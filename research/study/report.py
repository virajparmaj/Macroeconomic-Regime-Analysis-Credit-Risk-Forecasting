"""All tables derive from the saved ledger. Detailed ICE observations stay local."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, load_data
from .events import alarm_series, entry_dates, score_alarms
from .inference import comparison_rows, paired_inference, point_metrics

KEYS = ["scenario", "horizon", "task", "model", "feature_block"]


def read_predictions(run):
    out = ROOT / "results/research" / run
    df = pd.read_csv(out / "predictions.csv")
    for col in ["origin", "label_available", "target_date", "cutoff_timestamp"]:
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return out, df


def metrics_table(df):
    rows = []
    for key, g in df.groupby(KEYS):
        eligible = g.loc[g.eligible & g.actual.notna() & g.prediction.notna()]
        panels = [("at_risk" if key[2] == "entry" else "all", eligible)]
        if key[2] == "state":
            panels.append(("at_risk", eligible.loc[eligible.at_risk]))
        for risk, panel in panels:
            row = dict(zip(KEYS, key)) | {
                "risk_set": risk,
                "requested": len(g),
                "excluded": len(g) - len(eligible),
            }
            row.update(point_metrics(panel, key[2], g.score_type.iloc[0]))
            if key[2] == "level" and len(panel):
                for benchmark_name in ["mean_persistence", "endpoint_persistence"]:
                    b = df.loc[
                        (df.scenario == key[0])
                        & (df.horizon == key[1])
                        & (df.model == benchmark_name)
                        & df.eligible
                    ].set_index("origin")
                    paired = panel.set_index("origin")
                    if paired.index.isin(b.index).all():
                        loss = ((b.loc[paired.index, "prediction"] - paired.actual) ** 2).mean()
                        row["r2_vs_" + benchmark_name] = (
                            1 - row["mse"] / loss if loss > 0 else np.nan
                        )
            rows.append(row)
    return pd.DataFrame(rows)


def event_tables(all_predictions):
    summaries, alarms_out, events_out = [], [], []
    for key, g in all_predictions.loc[
        (all_predictions.task == "entry") & (all_predictions.score_type == "probability")
    ].groupby(KEYS):
        quantile = 0.7 if key[0] == "q70" else (0.8 if key[0] == "q80" else 0.75)
        data = load_data(quantile=quantile)
        alarms = alarm_series(g, data.target, data.threshold, key[1])
        valid = g.loc[(g.origin >= pd.Timestamp("2009-01-31")) & g.eligible].set_index("origin")
        if valid.empty:
            continue
        a = alarms.set_index("origin").reindex(valid.index)
        events = [
            e
            for e in entry_dates(data.target, data.threshold)
            if any(t < e <= t + pd.offsets.MonthEnd(key[1]) for t in valid.index)
        ]
        records, missed = score_alarms(valid.index, a.alarm, events, key[1])
        info = dict(zip(KEYS, key))
        false = sum(r["false_alarm"] for r in records)
        summaries.append(
            info
            | {
                "events": len(events),
                "warned": len(events) - len(missed),
                "missed": len(missed),
                "false_alarm_episodes": false,
                "at_risk_years": len(valid) / 12,
                "false_episodes_per_year": false / (len(valid) / 12),
                "calibration_unavailable": int(a.calibration_status.ne("calibrated").sum()),
            }
        )
        alarms_out.append(alarms.assign(**info))
        for r in records:
            events_out.append(info | r | {"missed": False})
        for e in missed:
            events_out.append(info | {"entry": e, "missed": True})
    return (
        pd.DataFrame(summaries),
        pd.concat(alarms_out, ignore_index=True) if alarms_out else pd.DataFrame(),
        pd.DataFrame(events_out),
    )


def calibration_table(df):
    rows = []
    for key, g in df.loc[df.eligible & (df.score_type == "probability")].groupby(KEYS):
        bins = np.minimum((g.prediction * 5).astype(int), 4)
        for b in range(5):
            p = g.loc[bins == b]
            rows.append(
                dict(zip(KEYS, key))
                | {
                    "bin_lower": b / 5,
                    "bin_upper": (b + 1) / 5,
                    "n": len(p),
                    "mean_probability": p.prediction.mean(),
                    "observed_rate": p.actual.mean(),
                }
            )
    return pd.DataFrame(rows)


def influence_table(df):
    data = load_data()
    g = df.loc[
        (df.scenario == "core")
        & (df.task == "level")
        & (df.model == "ridge")
        & (df.horizon == 1)
        & df.eligible
    ]
    a = g.loc[g.feature_block == "S_last+M+R"].set_index("origin")
    b = g.loc[g.feature_block == "S_last+M"].set_index("origin")
    if a.empty or b.empty:
        return pd.DataFrame()
    pairs = pd.DataFrame({"actual": a.actual, "model": a.prediction, "benchmark": b.prediction})
    rows = []
    for e in entry_dates(data.target, data.threshold):
        if e < pd.Timestamp("2009-01-31"):
            continue
        deletion = (pairs.index >= e - pd.offsets.MonthEnd(6)) & (
            pairs.index <= e + pd.offsets.MonthEnd(6)
        )
        masked = pairs.copy()
        masked.loc[deletion, :] = np.nan
        rows.append(
            {"deleted_event": e, "removed_origins": int(deletion.sum())} | paired_inference(masked)
        )
    return pd.DataFrame(rows)


def render(run):
    out, all_df = read_predictions(run)
    df = all_df.loc[all_df.origin >= pd.Timestamp("2009-01-31")].copy()
    tables = {
        "metrics": metrics_table(df),
        "comparisons": comparison_rows(df),
        "calibration": calibration_table(df),
        "event_influence": influence_table(df),
    }
    events, alarms, event_ledger = event_tables(all_df)
    tables["event_summary"] = events
    alarms.to_csv(out / "alarms.csv", index=False)
    event_ledger.to_csv(out / "event_ledger.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    # Public review artifacts contain aggregate results, never raw or per-origin OAS.
    public = ROOT / "research/study_results" / run
    public.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(public / f"{name}.csv", index=False)
    manifest = json.loads((out / "manifest.json").read_text())
    (public / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (public / "status.json").write_text((out / "status.json").read_text())
    failures = df.loc[df.exclusion_reason.fillna("").str.startswith("model_failure")]
    lines = [
        f"# Study run {run}",
        "",
        f"Profile: {manifest['profile']}. Historical analysis is exploratory.",
        "",
        f"Model failure rows: {len(failures)}.",
        "The local prediction ledger includes detailed observations; public files contain aggregate metrics only.",
        "",
        "Primary effects are incremental MSE gains, not RMSE reductions or trading returns.",
    ]
    comparisons = tables["comparisons"]
    if not comparisons.empty:
        p = comparisons.loc[
            (comparisons.scenario == "core")
            & (comparisons.horizon == 1)
            & (comparisons.model == "ridge")
            & (comparisons.contrast == "regime")
            & (comparisons.block_months == 12)
        ]
        if len(p):
            r = p.iloc[0]
            lines += [
                "",
                f"Primary n={int(r.n)}: gain {r.mse_gain:.4%}, 95% interval [{r.lower_95:.4%}, {r.upper_95:.4%}].",
                f"Regime RMSE {r.model_rmse:.6f} pp; no-regime RMSE {r.benchmark_rmse:.6f} pp.",
            ]
        plot_comparisons(comparisons, public)
    (public / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT {public}", flush=True)
    return public


def plot_comparisons(table, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = table.loc[(table.block_months == 12) & (table.contrast == "regime")]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, len(d) * 0.24)))
    y = np.arange(len(d))
    ax.hlines(y, 100 * d.lower_95, 100 * d.upper_95, color="#3670a3")
    ax.plot(100 * d.mse_gain, y, "o", color="#183753")
    ax.axvline(0, color="gray", lw=1)
    ax.axvline(5, color="gray", lw=1, ls="--")
    ax.set_yticks(y, [f"{r.scenario} / {r.model} / h{r.horizon}" for r in d.itertuples()])
    ax.set_xlabel("Incremental regime MSE gain (%) with 95% block interval")
    fig.tight_layout()
    fig.savefig(out / "regime_gains.png", dpi=170)
    plt.close(fig)
