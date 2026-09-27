"""Small, reproducible research-advisory checks; original data/code remain unchanged.

Run from any directory with a Python environment containing numpy/pandas/sklearn.
These are exploratory diagnostics, not the proposed completed paper experiments.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.splits import walk_forward_folds

OUT = ROOT / "research" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
panel_path = ROOT / "data/merged_macroeconomic_credit.csv"
df = pd.read_csv(panel_path, parse_dates=["Month_End"], index_col="Month_End")
s = df.Credit_Spread
daily_path = next((ROOT / "data/original").glob("ICE*.csv"))
raw = pd.read_csv(daily_path, parse_dates=["observation_date"], index_col="observation_date")
daily = pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna()
monthly = daily.resample("ME").agg(["mean", "last", "max", "min", "count"])
monthly["range"] = monthly["max"] - monthly["min"]
provenance = pd.DataFrame({"panel": s, "observed_daily_mean": monthly["mean"],
                           "forward_filled_daily_mean": raw.iloc[:, 0].ffill().resample("ME").mean()})
provenance["panel_minus_observed_mean"] = provenance.panel - provenance.observed_daily_mean
provenance.to_csv(OUT / "target_provenance.csv")
assert df.index.is_unique and df.index.is_monotonic_increasing
assert df.index.equals(pd.date_range(df.index.min(), df.index.max(), freq="ME"))
summary = {
    "python": platform.python_version(), "pandas": pd.__version__,
    "numpy": np.__version__, "sklearn": sklearn.__version__,
    "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [panel_path, daily_path]},
    "panel_rows": len(df), "start": str(df.index.min()), "end": str(df.index.max()),
    "missing_cells": int(df.isna().sum().sum()), "daily_nonmissing": len(daily),
    "monthly_mean_panel_max_abs_difference": float((monthly["mean"] - s).abs().max()),
    "panel_matches_forward_filled_mean_max_abs_error": float((provenance.panel-provenance.forward_filled_daily_mean).abs().max()),
    "months_differing_from_observed_daily_mean": int((provenance.panel_minus_observed_mean.abs()>1e-8).sum()),
    "target_reconstruction_max_abs_error": float((3*s.rolling(3).mean()-s.shift(1)-s.shift(2)-s).abs().max()),
    "march_2020": monthly.loc["2020-03-31"].to_dict(),
    "march_2020_mean_rank": int(monthly["mean"].rank(ascending=False, method="min").loc["2020-03-31"]),
    "march_2020_range_rank": int(monthly["range"].rank(ascending=False, method="min").loc["2020-03-31"]),
}
notebooks = []
for p in sorted((ROOT / "notebooks/v2").glob("*.ipynb")):
    cells = [c for c in json.loads(p.read_text())["cells"] if c["cell_type"] == "code"]
    notebooks.append({"notebook": p.name, "code_cells": len(cells),
                      "nonempty_code_cells": sum(bool("".join(c["source"]).strip()) for c in cells),
                      "cells_with_outputs": sum(bool(c.get("outputs")) for c in cells)})
summary["v2_notebooks"] = notebooks

# Hold target constant when comparing the two persistence forecasts.
benchmarks = []
for definition, mean in [("observed_daily_mean", monthly["mean"]), ("panel_ffill_mean", s)]:
    for start in ["1996-12-31", "2009-01-31"]:
        p = pd.DataFrame({"y": mean.shift(-1), "mean": mean,
                          "last": monthly["last"]}).loc[start:].dropna()
        for col in ["mean", "last"]:
            err = p.y - p[col]
            benchmarks.append({"origin_start": str(p.index.min().date()),
                               "origin_end": str(p.index.max().date()), "n": len(p),
                               "predictor": col, "target": "next_month_"+definition,
                               "rmse": float(np.sqrt((err**2).mean())),
                               "mae": float(err.abs().mean()),
                               "r2": float(1-(err**2).sum()/((p.y-p.y.mean())**2).sum())})
pd.DataFrame(benchmarks).to_csv(OUT / "aggregation_benchmarks.csv", index=False)

# Match the original spread-only classifier's features, annual refits, 12-month gap,
# and forest settings. Hold these fixed to isolate unknown-future-label handling.
X = pd.DataFrame({"sp_lag0": s, "sp_d3": s.diff(3),
                  "sp_rollstd6": s.rolling(6).std().shift(1)})
threshold = float(s.quantile(.75))  # Deliberately retained for a single-factor diagnostic.
class_rows = []
prediction_rows = []
for h in [1, 3, 6, 12]:
    future = s.shift(-h)
    for corrected in [False, True]:
        y = (future >= threshold).astype(float)
        if corrected:
            y = y.where(future.notna())
        D = X.join(y.rename("y")).dropna()
        pieces = []
        for f in walk_forward_folds(D.index, "2008-12-31", refit_freq=12, embargo=max(12,h)):
            tr, te = f.train_index, f.test_index
            model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                            random_state=42, n_jobs=-1, class_weight="balanced")
            model.fit(D.loc[tr, X.columns], D.loc[tr, "y"])
            p = pd.DataFrame({"actual": D.loc[te, "y"],
                              "rf_score": model.predict_proba(D.loc[te, X.columns])[:,1],
                              "current_spread_score": s.loc[te],
                              "future_observed": future.loc[te].notna()})
            pieces.append(p)
        p = pd.concat(pieces)
        for model in ["rf_score", "current_spread_score"]:
            class_rows.append({"h": h, "future_label_fix": corrected, "score": model,
                               "threshold_source": "full_sample_exploratory_only",
                               "n": len(p), "positive_months": int(p.actual.sum()),
                               "unknown_future_rows_scored": int((~p.future_observed).sum()),
                               "auc": roc_auc_score(p.actual,p[model]),
                               "average_precision": average_precision_score(p.actual,p[model])})
        p["h"] = h
        p["future_label_fix"] = corrected
        prediction_rows.append(p.reset_index(names="origin"))
    print(f"Completed classification horizon {h}", flush=True)
pd.DataFrame(class_rows).to_csv(OUT / "classification_label_diagnostic.csv", index=False)
pd.concat(prediction_rows).to_csv(OUT / "classification_diagnostic_predictions.csv", index=False)

# Count onset information without claiming an onset forecast has been evaluated.
threshold_rows = []
for name, q in [("full_sample_exploratory", threshold),
                ("frozen_pre_2009", float(s.loc[:"2008-12-31"].quantile(.75)))]:
    state = s >= q
    onsets = state & ~state.shift(1, fill_value=False)
    onset_dates = [str(t.date()) for t in s.loc["2009-01-31":].index if onsets.loc[t]]
    for h in [1,3,6,12]:
        # An at-risk forecast predicts at least one crossing in t+1,...,t+h.
        future_paths = pd.concat([s.shift(-j) for j in range(1,h+1)],axis=1)
        observed = future_paths.notna().all(axis=1)
        at_risk = (~state) & observed & (s.index >= pd.Timestamp("2009-01-31"))
        event = future_paths.ge(q).any(axis=1)
        threshold_rows.append({"threshold_source": name, "threshold": q, "h":h,
                               "at_risk_origins":int(at_risk.sum()),
                               "positive_origins":int((event & at_risk).sum()),
                               "distinct_onset_dates_in_test": ";".join(onset_dates)})
pd.DataFrame(threshold_rows).to_csv(OUT / "onset_counts.csv",index=False)
(OUT / "audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
print(pd.DataFrame(class_rows).to_string(index=False))
print(pd.DataFrame(benchmarks).to_string(index=False))
