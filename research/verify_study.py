"""Independently verify completed ledger runs and compare repeated forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["scenario", "origin", "horizon", "task", "model", "feature_block", "seed"]


def verify(run):
    out = ROOT / "results/research" / run
    status = json.loads((out / "status.json").read_text())
    assert status["complete"], "Incomplete jobs"
    manifest = json.loads((out / "manifest.json").read_text())
    data = pd.read_csv(out / "predictions.csv")
    assert not data.duplicated(KEYS).any(), "Duplicate forecast keys"
    errors = data.exclusion_reason.fillna("").str.startswith("model_failure")
    assert not errors.any(), (
        data.loc[errors, ["origin", "model", "exclusion_reason"]].head().to_dict()
    )
    valid = data.loc[data.eligible].copy()
    assert np.isfinite(valid.actual).all() and np.isfinite(valid.prediction).all()
    assert (
        pd.to_datetime(valid.last_training_label_available_at)
        <= pd.to_datetime(valid.cutoff_timestamp)
    ).all()
    assert (pd.to_datetime(valid.train_end) < pd.to_datetime(valid.cutoff_timestamp)).all()
    probs = valid.loc[valid.score_type == "probability", "prediction"]
    assert probs.between(0, 1).all()
    assert valid.loc[valid.task == "entry", "at_risk"].all()
    if manifest["profile"] == "core":
        for h, n in [(1, 162), (3, 160)]:
            counts = (
                valid.loc[(valid.task == "level") & (valid.horizon == h)]
                .groupby(["model", "feature_block"])
                .size()
            )
            assert counts.eq(n).all(), counts.to_dict()
    # Recompute aggregate MSE independently of the reporting module.
    metrics_path = out / "metrics.csv"
    checked = 0
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for row in metrics.loc[metrics.task == "level"].itertuples():
            p = valid.loc[
                (valid.scenario == row.scenario)
                & (valid.horizon == row.horizon)
                & (valid.task == "level")
                & (valid.model == row.model)
                & (valid.feature_block == row.feature_block)
            ]
            mse = np.mean((p.prediction.to_numpy() - p.actual.to_numpy()) ** 2)
            assert np.isclose(mse, row.mse, rtol=1e-12, atol=1e-12)
            checked += 1
    return {
        "run_id": run,
        "profile": manifest["profile"],
        "rows": len(data),
        "eligible": len(valid),
        "jobs": status["total_jobs"],
        "model_failures": int(errors.sum()),
        "independently_recomputed_metrics": checked,
        "passed": True,
    }


def compare(first, second):
    def read(run):
        return (
            pd.read_csv(ROOT / "results/research" / run / "predictions.csv")
            .set_index(KEYS)
            .sort_index()
        )

    a, b = read(first), read(second)
    assert a.index.equals(b.index)
    assert a.eligible.equals(b.eligible)
    assert a.exclusion_reason.fillna("").equals(b.exclusion_reason.fillna(""))
    np.testing.assert_allclose(a.prediction, b.prediction, rtol=1e-8, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(a.actual, b.actual, rtol=0, atol=0, equal_nan=True)
    return {
        "first": first,
        "repeat": second,
        "rows": len(a),
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_prediction_difference": float(np.nanmax(abs(a.prediction - b.prediction))),
        "passed": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--compare")
    a = p.parse_args()
    print(json.dumps(compare(a.run, a.compare) if a.compare else verify(a.run), indent=2))
