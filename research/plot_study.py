"""Paper figures from aggregate metrics; detailed spread levels remain local."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "research/study_results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def plots():
    registry = json.loads((PUBLIC / "latest_runs.json").read_text())
    core = PUBLIC / registry["core"]
    metrics = pd.read_csv(core / "metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = metrics.loc[(metrics.task == "level") & (metrics.horizon == 1)].sort_values("rmse")
    ax.barh(
        np.arange(len(d)),
        100 * d.rmse,
        color=["#b06d36" if n == "endpoint_persistence" else "#3670a3" for n in d.model],
    )
    ax.set_yticks(np.arange(len(d)), [f"{r.model} / {r.feature_block}" for r in d.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("One-month RMSE (basis points), 162 common origins")
    fig.tight_layout()
    fig.savefig(core / "baseline_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    panels = [
        ("state", "all", "State: all origins"),
        ("state", "at_risk", "State: at risk"),
        ("entry", "at_risk", "Entry: at risk"),
    ]
    for ax, h in zip(axes, [1, 3]):
        for model, color, offset in [
            ("single_logistic", "#b06d36", -0.15),
            ("logistic", "#3670a3", 0),
            ("rf", "#52794d", 0.15),
        ]:
            rows = []
            for task, risk, label in panels:
                q = metrics.loc[
                    (metrics.horizon == h)
                    & (metrics.task == task)
                    & (metrics.risk_set == risk)
                    & (metrics.model == model)
                    & (
                        metrics.feature_block
                        == ("S_last" if model == "single_logistic" else "S_last+M+R")
                    )
                ]
                rows.append(q.iloc[0])
            ax.plot([r.auc for r in rows], np.arange(3) + offset, "o", label=model, color=color)
        ax.set_yticks(np.arange(3), [p[2] for p in panels])
        ax.invert_yaxis()
        ax.set_xlim(0.4, 1.01)
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_title(f"h={h} months")
        ax.set_xlabel("ROC AUC (descriptive; three entry events)")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Population changes matter: compare models within each panel")
    fig.tight_layout()
    fig.savefig(core / "state_entry_panels.png", dpi=180)
    plt.close(fig)

    alarms = pd.read_csv(
        ROOT / "results/research" / registry["core"] / "alarms.csv", parse_dates=["origin"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    from research.study.data import load_data
    from research.study.events import entry_dates

    data = load_data()
    event_dates = entry_dates(data.target, data.threshold)
    event_dates = event_dates[event_dates >= pd.Timestamp("2009-01-31")]
    for ax, h in zip(axes, [1, 3]):
        for y, model in enumerate(["single_logistic", "logistic", "rf"]):
            a = alarms.loc[
                (alarms.horizon == h)
                & (alarms.model == model)
                & (
                    alarms.feature_block
                    == ("S_last" if model == "single_logistic" else "S_last+M+R")
                )
                & alarms.alarm
            ]
            ax.scatter(a.origin, np.full(len(a), y), s=18, color="#b06d36")
        for event in event_dates:
            ax.axvline(event, color="#3670a3", lw=1, ls="--")
        ax.set_yticks(range(3), ["one-spread logistic", "macro+regime logistic", "macro+regime RF"])
        ax.set_title(f"h={h}: alarm months (orange); actual entry dates (blue)")
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlim(pd.Timestamp("2009-01-31"), pd.Timestamp("2022-07-31"))
    fig.tight_layout()
    fig.savefig(core / "alarm_timeline.png", dpi=180)
    plt.close(fig)

    ext = pd.read_csv(PUBLIC / registry["external"] / "metrics.csv")
    historical = pd.read_csv(PUBLIC / registry["sensitivity"] / "metrics.csv")
    historical = historical.loc[historical.scenario == "five_macro"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, h in zip(axes, [1, 3]):
        names = []
        for y, (model, block) in enumerate(
            [
                ("ridge", "S_last+M"),
                ("ridge", "S_last+M+R"),
                ("rf", "S_last+M"),
                ("rf", "S_last+M+R"),
            ]
        ):
            names.append(f"{model} / {block}")
            for frame, color, offset, label in [
                (historical, "#3670a3", -0.1, "Historical five-macro"),
                (ext, "#b06d36", 0.1, "Later-period holdout"),
            ]:
                r = frame.loc[
                    (frame.horizon == h) & (frame.model == model) & (frame.feature_block == block)
                ].iloc[0]
                ax.scatter(
                    1 - r.r2_vs_endpoint_persistence,
                    y + offset,
                    color=color,
                    label=label if y == 0 else None,
                )
        ax.axvline(1, color="gray", ls="--")
        ax.set_yticks(range(4), names)
        ax.invert_yaxis()
        ax.set_title(f"h={h} months")
        ax.set_xlabel("MSE / endpoint-persistence MSE (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PUBLIC / "external_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plots()
