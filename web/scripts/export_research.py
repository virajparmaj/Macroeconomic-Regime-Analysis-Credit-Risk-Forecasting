#!/usr/bin/env python3
"""Export existing research evidence for the website, without fitting any model.

Run from any directory with Python + pandas + numpy:
    python web/scripts/export_research.py

This script reads frozen inputs, checks alignment and recorded audit hashes,
recalculates simple metrics from saved predictions, and writes only under web/.
It never imports analysis scripts (many of which fit models at import time).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
PUBLIC = WEB / "public/evidence"
EVIDENCE = ROOT / "research/evidence"
MACROS = ["CPI", "FEDFUNDS", "Industrial_Production", "GDP", "Unemployment_Rate", "Consumer_Sentiment"]
INPUTS: set[Path] = set()


def read_csv(path: str, date: str) -> pd.DataFrame:
    absolute = ROOT / path
    INPUTS.add(absolute)
    return pd.read_csv(absolute, parse_dates=[date]).set_index(date).sort_index()


def read_json(path: str) -> dict:
    absolute = ROOT / path
    INPUTS.add(absolute)
    return json.loads(absolute.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def date(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def result_context(index: pd.DatetimeIndex, horizon: int, source: str, status: str) -> dict:
    return {
        "n": len(index), "originStart": date(index.min()), "originEnd": date(index.max()),
        "targetStart": date(index.min() + pd.offsets.MonthEnd(horizon)),
        "targetEnd": date(index.max() + pd.offsets.MonthEnd(horizon)),
        "horizonMonths": horizon, "source": source, "status": status,
    }


def metrics(actual: pd.Series, prediction: pd.Series, benchmark: pd.Series) -> dict:
    error = actual - prediction
    squared_error = float((error ** 2).sum())
    return {
        "rmse": float(np.sqrt((error ** 2).mean())), "mae": float(error.abs().mean()),
        "r2Mean": 1 - squared_error / float(((actual - actual.mean()) ** 2).sum()),
        "r2Persistence": 1 - squared_error / float(((actual - benchmark) ** 2).sum()),
    }


def ranking_metrics(actual: pd.Series, score: pd.Series) -> tuple[float, float]:
    """Check stored AUC/AP without fitting or evaluating any new model."""
    positives = int(actual.sum())
    negatives = len(actual) - positives
    ranks = score.rank(method="average")
    auc = (float(ranks[actual == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    groups = pd.DataFrame({"score": score, "actual": actual}).groupby("score").actual.agg(["sum", "count"]).sort_index(ascending=False)
    precision = groups["sum"].cumsum() / groups["count"].cumsum()
    average_precision = float((groups["sum"] / positives * precision).sum())
    return auc, average_precision


audit = read_json("research/evidence/audit_summary.json")
for name, expected in audit["input_sha256"].items():
    assert sha256(ROOT / name) == expected, f"Frozen audited input changed: {name}"

panel = read_csv("data/merged_macroeconomic_credit.csv", "Month_End")
regimes = read_csv("data/datamerged_macro_credit_with_regimes.csv", "Month_End")
stress = read_csv("research/evidence/macro_stress.csv", "Month_End")
provenance = read_csv("research/evidence/target_provenance.csv", "Unnamed: 0")
raw = read_csv("data/original/ICE BofA US High Yield Index Option-Adjusted Spread_BAMLH0A0HYM2.csv", "observation_date")
daily = pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna()
monthly = daily.resample("ME").agg(["mean", "last", "max", "min", "count"])
spread = panel.Credit_Spread
assert len(panel) == audit["panel_rows"] == 309
assert len(daily) == audit["daily_nonmissing"] == 6679
assert panel.index.is_unique and panel.index.equals(regimes.index)
assert panel.index.equals(pd.date_range(panel.index.min(), panel.index.max(), freq="ME"))
assert np.allclose(provenance.panel, spread)
assert np.allclose(provenance.observed_daily_mean, monthly["mean"])
assert int((provenance.panel_minus_observed_mean.abs() > 1e-8).sum()) == 80
assert np.allclose(regimes[[f"Regime_Prob_{i}" for i in range(10)]].sum(axis=1), 1)
assert np.array_equal(regimes.Regime_Label, regimes[[f"Regime_Prob_{i}" for i in range(10)]].values.argmax(axis=1))
threshold = float(spread.quantile(.75))
frozen_threshold = float(spread.loc[:"2008-12-31"].quantile(.75))
macro_threshold = float(stress.macro_stress.quantile(.75))
change = panel[MACROS].diff()

timeline = []
for origin, row in panel.iterrows():
    mo = monthly.loc[origin]
    regime = regimes.loc[origin]
    score = float(stress.loc[origin, "macro_stress"]) if origin in stress.index else None
    timeline.append({
        "date": date(origin), "legacyMean": float(row.Credit_Spread), "observedMean": float(mo["mean"]),
        "last": float(mo["last"]), "max": float(mo["max"]), "min": float(mo["min"]),
        "count": int(mo["count"]), "range": float(mo["max"] - mo["min"]),
        "targetDifference": float(provenance.loc[origin, "panel_minus_observed_mean"]),
        "macro": {key: float(row[key]) for key in MACROS},
        "macroChange1": {key: None if pd.isna(change.loc[origin, key]) else float(change.loc[origin, key]) for key in MACROS},
        "macroStress": score, "macroStressState": None if score is None else score >= macro_threshold,
        "spreadStress": bool(row.Credit_Spread >= threshold),
        "clusterId": int(regime.Regime_Label), "smoothedClusterId": int(regime.Regime_Label_Smoothed),
        "clusterProbabilities": [float(regime[f"Regime_Prob_{i}"]) for i in range(10)],
    })

macro_only = read_csv("research/evidence/preds_macro_only.csv", "idx")
macro_spread = read_csv("research/evidence/preds_macro+spread_history.csv", "idx")
assert macro_only.index.equals(macro_spread.index)
assert len(macro_only) == 163 and date(macro_only.index.min()) == "2009-01-31"
assert date(macro_only.index.max()) == "2022-07-31"
assert np.allclose(macro_only[["y", "rw"]], macro_spread[["y", "rw"]])
assert np.allclose(macro_only.y, spread.shift(-1).loc[macro_only.index])
assert np.allclose(macro_only.rw, spread.loc[macro_only.index])
predictions = [{
    "originDate": date(origin), "targetDate": date(origin + pd.offsets.MonthEnd(1)),
    "refitDate": f"{origin.year - 1}-12-31", "actual": float(row.y), "persistence": float(row.rw),
    "rfMacroOnly": float(row.model), "rfMacroSpread": float(macro_spread.loc[origin, "model"]),
    "lastDaily": float(monthly.loc[origin, "last"]),
} for origin, row in macro_only.iterrows()]

forecast_metrics = []
for model_id, label, feature_set, prediction, source in [
    ("persistence", "Monthly persistence", "Current legacy monthly mean", macro_only.rw, "research/evidence/preds_macro_only.csv"),
    ("rf_macro_level", "RF · macro only · level", "18 macro level / difference features", macro_only.model, "research/evidence/preds_macro_only.csv"),
    ("rf_macro_spread_level", "RF · macro + spread · level", "18 macro + 5 spread-history features", macro_spread.model, "research/evidence/preds_macro+spread_history.csv"),
]:
    forecast_metrics.append({
        "id": model_id, "label": label, "featureSet": feature_set, "target": "next_month_legacy_mean",
        **metrics(macro_only.y, prediction, macro_only.rw),
        **result_context(macro_only.index, 1, source, "Reproduced exploratory evaluation"),
        "benchmark": "Monthly persistence (origin monthly mean)", "hasPredictionRecords": True,
        "metricPrecision": "computed from stored predictions",
    })

# Only rounded summary metrics survive for these models. No prediction paths are fabricated.
fair_path = EVIDENCE / "fair_eval.log"
INPUTS.add(fair_path)
fair_log = fair_path.read_text()
for model_id, label, expression, feature_set in [
    ("rf_macro_change", "RF · macro only · change", r"RF\s+macro-only-Δ", "24 macro level / difference features"),
    ("rf_macro_spread_change", "RF · macro + spread · change", r"RF\s+macro\+spread-Δ", "24 macro + 4 spread-history features"),
    ("ridge_macro_spread_change", "Ridge · macro + spread · change", r"Ridge\s+macro\+spread-Δ", "24 macro + 4 spread-history features"),
]:
    match = re.search(expression + r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)", fair_log)
    assert match is not None, f"Missing stored summary: {label}"
    n, rmse, benchmark_rmse, r2 = match.groups()
    assert int(n) == 163 and abs(float(benchmark_rmse) - forecast_metrics[0]["rmse"]) < .00005
    forecast_metrics.append({
        "id": model_id, "label": label, "featureSet": feature_set, "target": "next_month_legacy_change",
        "rmse": float(rmse), "mae": None, "r2Mean": None, "r2Persistence": float(r2),
        **result_context(macro_only.index, 1, "research/evidence/fair_eval.log", "Reproduced exploratory summary; prediction records unavailable"),
        "benchmark": "Zero change (equivalent to monthly persistence)", "hasPredictionRecords": False,
        "metricPrecision": "rounded stored log",
    })

benchmark_path = EVIDENCE / "aggregation_benchmarks.csv"
INPUTS.add(benchmark_path)
benchmarks = pd.read_csv(benchmark_path)
aggregation = []
for _, row in benchmarks.iterrows():
    origins = pd.date_range(row.origin_start, row.origin_end, freq="ME")
    target = monthly["mean"] if row.target == "next_month_observed_daily_mean" else spread
    predictor = target if row.predictor == "mean" else monthly["last"]
    err = target.shift(-1).loc[origins] - predictor.loc[origins]
    assert len(origins) == row.n and np.isclose(np.sqrt((err ** 2).mean()), row.rmse)
    assert np.isclose(err.abs().mean(), row.mae)
    aggregation.append({
        **result_context(origins, 1, "research/evidence/aggregation_benchmarks.csv", "Verified exploratory benchmark"),
        "target": row.target, "predictor": row.predictor, "rmse": float(row.rmse),
        "mae": float(row.mae), "r2Mean": float(row.r2),
        "benchmark": "Previous monthly mean with the same target definition and origins",
    })

classification_path = EVIDENCE / "classification_label_diagnostic.csv"
INPUTS.add(classification_path)
class_metrics = pd.read_csv(classification_path)
class_predictions = read_csv("research/evidence/classification_diagnostic_predictions.csv", "origin")
classification = []
for h in [1, 3, 6, 12]:
    rows = class_metrics[(class_metrics.h == h) & class_metrics.future_label_fix].set_index("score")
    rf, current = rows.loc["rf_score"], rows.loc["current_spread_score"]
    records = class_predictions[(class_predictions.h == h) & class_predictions.future_label_fix]
    old = class_metrics[(class_metrics.h == h) & ~class_metrics.future_label_fix & (class_metrics.score == "rf_score")].iloc[0]
    assert len(records) == rf.n == current.n and int(records.actual.sum()) == rf.positive_months
    assert records.future_observed.all() and rf.unknown_future_rows_scored == 0
    assert int(old.n) - len(records) == old.unknown_future_rows_scored == h
    assert np.array_equal(records.actual, (spread.shift(-h).loc[records.index] >= threshold).astype(float))
    for key, stored in [("rf_score", rf), ("current_spread_score", current)]:
        auc, ap = ranking_metrics(records.actual, records[key])
        assert np.isclose(auc, stored.auc) and np.isclose(ap, stored.average_precision)
    classification.append({
        **result_context(records.index, h, "research/evidence/classification_label_diagnostic.csv", "Corrected exploratory diagnostic"),
        "rfAuc": float(rf.auc), "currentSpreadAuc": float(current.auc),
        "rfAveragePrecision": float(rf.average_precision), "currentSpreadAveragePrecision": float(current.average_precision),
        "positiveMonths": int(rf.positive_months), "unknownFutureExcluded": int(old.unknown_future_rows_scored),
        "threshold": threshold, "thresholdSource": "Full-sample q75 of legacy monthly mean; retrospective",
        "target": "Stress occupancy at t+h (legacy mean >= full-sample q75)",
        "benchmark": "Raw current-spread ranking score",
    })

onset_path = EVIDENCE / "onset_counts.csv"
INPUTS.add(onset_path)
onsets = []
for _, row in pd.read_csv(onset_path).iterrows():
    state = spread >= row.threshold
    entries = state & ~state.shift(1, fill_value=False)
    dates = [date(t) for t in entries.loc["2009-01-31":][lambda x: x].index]
    assert dates == row.distinct_onset_dates_in_test.split(";")
    paths = pd.concat([spread.shift(-j) for j in range(1, int(row.h) + 1)], axis=1)
    at_risk = ~state & paths.notna().all(axis=1) & (spread.index >= pd.Timestamp("2009-01-31"))
    assert int(at_risk.sum()) == row.at_risk_origins
    assert int((at_risk & paths.ge(row.threshold).any(axis=1)).sum()) == row.positive_origins
    onsets.append({
        "thresholdSource": row.threshold_source, "threshold": float(row.threshold),
        "horizonMonths": int(row.h), "atRiskOrigins": int(row.at_risk_origins),
        "positiveOrigins": int(row.positive_origins), "dates": dates,
        "source": "research/evidence/onset_counts.csv", "status": "Verified event counts; no entry forecast evaluated",
    })

# Public snapshots keep source links usable in an exported static website.
snapshots = sorted({
    *[p for p in EVIDENCE.iterdir() if p.suffix in {".csv", ".log", ".json"}],
    *list((ROOT / "research/paper_preparation").glob("*.md")),
    ROOT / "research/paper_preparation/validation_report.txt",
    ROOT / "research/RESEARCH_ADVISORY.md", ROOT / "research/EXPERIMENT_PROTOCOL.md",
    ROOT / "web/EVIDENCE_MAP.md", ROOT / "data/merged_macroeconomic_credit.csv",
    ROOT / "data/datamerged_macro_credit_with_regimes.csv",
    ROOT / "research/audit_checks.py", ROOT / "analysis/honest_eval.py",
    ROOT / "analysis/fair_eval.py", ROOT / "analysis/macro_regime.py",
    ROOT / "src/config_v2.py", ROOT / "src/metrics.py", ROOT / "src/splits.py",
    ROOT / "src/features.py", ROOT / "utilities/functions.py", ROOT / "utilities/data_processing.py",
    ROOT / "notebooks/viraj/p1_models.ipynb", ROOT / "notebooks/viraj/p2_models.ipynb",
    ROOT / "notebooks/viraj/unsupervised.ipynb", ROOT / "notebooks/v2/00_audit_of_v1.ipynb",
    ROOT / "notebooks/old_work/lasso_ridge.ipynb", ROOT / "notebooks/old_work/poly_reg.ipynb",
    ROOT / "notebooks/old_work/ Basic ML Time Series Models.ipynb",
    ROOT / "notebooks/old_work/Regime detection_Unsupervise_learning.ipynb",
})
PUBLIC.mkdir(parents=True, exist_ok=True)
for snapshot in snapshots:
    assert snapshot.exists(), f"Missing source snapshot: {snapshot}"
    relative = snapshot.relative_to(ROOT)
    destination = PUBLIC / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot, destination)
    INPUTS.add(snapshot)

payload = {
    "meta": {
        "schemaVersion": 1, "generatedBy": "web/scripts/export_research.py (exports existing evidence; no model fitting)",
        "sourceHashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(INPUTS)},
        "exportedSources": [str(p.relative_to(ROOT)) for p in snapshots],
        "spreadUnit": "percentage points; 1 pp = 100 basis points",
        "panel": {"n": len(panel), "start": date(panel.index.min()), "end": date(panel.index.max()), "missingCells": int(panel.isna().sum().sum())},
        "daily": {"n": len(daily), "sourceRows": len(raw), "missingSourceRows": len(raw) - len(daily), "start": date(daily.index.min()), "end": date(daily.index.max()), "lastMonthPartial": daily.index.max() < monthly.index.max(), "finalMonthObservedDays": int(monthly.iloc[-1]["count"])},
        "thresholds": {"spreadFullSample": threshold, "spreadFrozen2008": frozen_threshold, "macroStress": macro_threshold},
        "audit": {"differingMonths": audit["months_differing_from_observed_daily_mean"], "maxDifferencePp": audit["monthly_mean_panel_max_abs_difference"], "reconstructionError": audit["target_reconstruction_max_abs_error"], "emptyV2Cells": sum(n["code_cells"] - n["nonempty_code_cells"] for n in audit["v2_notebooks"]), "macroStressMonths": len(stress)},
        "forecast": {"refitBlocks": len({row["refitDate"] for row in predictions}), "refitEveryMonths": 12, "embargoMonths": 3, "timing": "Publication-lag approximations applied to revised macro data; refitDate is the nominal annual fold boundary before its test origins."},
        "timeline": {"source": ["data/merged_macroeconomic_credit.csv", "research/evidence/target_provenance.csv", "research/evidence/macro_stress.csv", "data/datamerged_macro_credit_with_regimes.csv"], "status": "Verified descriptive series; retrospective regime labels", "target": "Legacy forward-filled monthly mean and separately reconstructed observed-day mean", "horizon": "Descriptive; no forecast horizon", "benchmark": "Not applicable"},
        "classification": {"featureSet": "Current spread, three-month spread difference, lagged six-month spread standard deviation", "refitEveryMonths": 12, "embargoMonths": 12, "limitation": "Full-sample q75 retained to isolate unknown-label correction; scores do not establish calibrated probabilities or entry warning."},
    },
    "timeline": timeline, "march2020Daily": [{"date": date(t), "value": float(v)} for t, v in daily.loc["2020-03"].items()],
    "predictions": predictions, "metrics": forecast_metrics, "aggregation": aggregation,
    "classification": classification, "onsets": onsets,
    "aggregationExample": {"month": "2020-03-31", "observedMean": audit["march_2020"]["mean"], "last": audit["march_2020"]["last"], "max": audit["march_2020"]["max"], "min": audit["march_2020"]["min"], "range": audit["march_2020"]["range"], "observedDays": int(audit["march_2020"]["count"]), "meanRank": audit["march_2020_mean_rank"], "rangeRank": audit["march_2020_range_rank"], "source": "research/evidence/audit_summary.json"},
}

output = WEB / "src/data/research.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
manifest = {
    "exporter": "web/scripts/export_research.py", "exporterSha256": sha256(Path(__file__)),
    "output": "web/src/data/research.json", "outputSha256": sha256(output),
    "method": "No new fitting. Stored predictions, stored log summaries, corrected diagnostics, and descriptive source aggregation only.",
    "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bundled": p in snapshots} for p in sorted(INPUTS)],
    "rawDailySource": "Raw daily input is not bundled; the March 2020 observed values used in the detail chart are exported. Full-period daily counts and monthly summaries are verified from the raw source. The final month contains one source observation (2022-08-01).",
}
(PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Verified/exported {len(timeline)} monthly observations, {len(predictions)} saved forecasts, {len(classification)} corrected horizon comparisons, and {len(snapshots)} evidence snapshots.")
print(f"Output: {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes); manifest: web/public/evidence/manifest.json")
