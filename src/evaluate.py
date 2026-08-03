"""The only module permitted to write ``results/metrics.csv``.

Every number that appears in a v2 write-up has to be traceable to a run that produced it.
v1's Phase 2 comparison table was a DataFrame literal -- ``"MSE": [0.524, ...]`` -- typed
next to a printed result of 5.723, and the accompanying prose claimed a stacked LSTM MSE
of 0.194 against a printed 15.001. Routing every metric through one writer, with the
run's provenance attached, removes the seam where a typed number can be substituted for a
computed one.

Notebook 06 reads ``results/metrics.csv`` and renders it. It does not compute metrics of
its own, and no other module writes this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from src.config_v2 import METRICS_CSV_PATH, RESULTS_DIR
from src.metrics import diebold_mariano, r2_oos, regression_metrics

#: Column order of ``results/metrics.csv``.
METRICS_COLUMNS: Sequence[str] = (
    "model",
    "fold",
    "n_test",
    "test_start",
    "test_end",
    "mse",
    "rmse",
    "mae",
    "r2",
    "r2_oos_vs_rw",
    "dm_stat_vs_rw",
    "dm_pvalue_vs_rw",
    "notes",
)


@dataclass(frozen=True)
class MetricRecord:
    """One model, evaluated on one fold.

    Attributes:
        model: Model name.
        fold: Fold index, or -1 for a single-split evaluation.
        n_test: Number of scored observations.
        test_start: First scored timestamp, ISO date.
        test_end: Last scored timestamp, ISO date.
        mse: Mean squared error.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        r2: R-squared against the test-period mean.
        r2_oos_vs_rw: Out-of-sample R-squared against the random walk.
        dm_stat_vs_rw: Diebold-Mariano statistic against the random walk.
        dm_pvalue_vs_rw: Two-sided p-value of that statistic.
        notes: Free text describing the configuration.
    """

    model: str
    fold: int
    n_test: int
    test_start: str
    test_end: str
    mse: float
    rmse: float
    mae: float
    r2: float
    r2_oos_vs_rw: float = float("nan")
    dm_stat_vs_rw: float = float("nan")
    dm_pvalue_vs_rw: float = float("nan")
    notes: str = ""


def score_fold(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    fold: int = -1,
    benchmark: Optional[pd.Series] = None,
    horizon: int = 1,
    notes: str = "",
) -> MetricRecord:
    """Score one model on one fold, including comparison against a benchmark.

    Args:
        model_name: Name to record.
        y_true: Observed values, indexed by timestamp.
        y_pred: Predictions, indexed by timestamp.
        fold: Fold index, or -1 for a single split.
        benchmark: Benchmark predictions, normally the random walk. When supplied,
            out-of-sample R-squared and a Diebold-Mariano test are added.
        horizon: Forecast horizon, passed to the Diebold-Mariano test.
        notes: Free text describing the configuration.

    Returns:
        A populated :class:`MetricRecord`.

    Raises:
        ValueError: If no complete observations remain after alignment.
    """
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if benchmark is not None:
        frame["benchmark"] = benchmark
    frame = frame.dropna()
    if frame.empty:
        raise ValueError(f"No complete observations for model {model_name!r} on fold {fold}")

    base = regression_metrics(frame["y_true"], frame["y_pred"])
    record: Dict[str, Any] = {
        "model": model_name,
        "fold": fold,
        "n_test": len(frame),
        "test_start": str(frame.index.min().date()),
        "test_end": str(frame.index.max().date()),
        **base,
        "notes": notes,
    }

    if benchmark is not None:
        try:
            record["r2_oos_vs_rw"] = r2_oos(frame["y_true"], frame["y_pred"], frame["benchmark"])
        except ValueError:
            record["r2_oos_vs_rw"] = float("nan")
        try:
            dm = diebold_mariano(
                frame["y_true"], frame["y_pred"], frame["benchmark"], horizon=horizon
            )
            record["dm_stat_vs_rw"] = dm["dm_stat"]
            record["dm_pvalue_vs_rw"] = dm["p_value"]
        except ValueError:
            record["dm_stat_vs_rw"] = float("nan")
            record["dm_pvalue_vs_rw"] = float("nan")

    return MetricRecord(**record)


def records_to_frame(records: Iterable[MetricRecord]) -> pd.DataFrame:
    """Convert records to a frame with the canonical column order.

    Args:
        records: Records to convert.

    Returns:
        Frame with columns in :data:`METRICS_COLUMNS` order.

    Raises:
        ValueError: If ``records`` is empty.
    """
    rows = [asdict(r) for r in records]
    if not rows:
        raise ValueError("No metric records to write")
    return pd.DataFrame(rows).reindex(columns=list(METRICS_COLUMNS))


def write_metrics(
    records: Iterable[MetricRecord],
    path: Path | str = METRICS_CSV_PATH,
    append: bool = False,
) -> Path:
    """Write metric records to ``results/metrics.csv``.

    This is the single permitted writer of that file.

    Args:
        records: Records to write.
        path: Destination path. Must sit under ``results/``.
        append: Append to an existing file rather than replacing it. Rows matching an
            existing (model, fold) pair are replaced so that a re-run does not duplicate.

    Returns:
        The path written.

    Raises:
        ValueError: If ``path`` is outside ``results/``, or if no records were supplied.
    """
    path = Path(path)
    if RESULTS_DIR not in path.parents and path.parent != RESULTS_DIR:
        raise ValueError(f"Refusing to write metrics outside {RESULTS_DIR}: {path}")

    frame = records_to_frame(records)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
        frame = combined.drop_duplicates(subset=["model", "fold"], keep="last")

    frame.to_csv(path, index=False)
    return path


def read_metrics(path: Path | str = METRICS_CSV_PATH) -> pd.DataFrame:
    """Read ``results/metrics.csv``.

    Args:
        path: Path to read.

    Returns:
        The metrics frame.

    Raises:
        FileNotFoundError: If the file has not been written yet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. It is written by src.evaluate.write_metrics, "
            "which notebook 05 calls once the walk-forward run completes."
        )
    return pd.read_csv(path)


def summarise_across_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean and standard deviation per model.

    Reporting a spread across folds rather than a single number is the point of the
    walk-forward design: one split on this sample gives roughly six independent
    observations, so a lone metric cannot be distinguished from noise.

    Args:
        frame: Per-fold metrics, as written by :func:`write_metrics`.

    Returns:
        One row per model with mean and standard deviation of each numeric metric and the
        number of folds contributing.
    """
    numeric = ["mse", "rmse", "mae", "r2", "r2_oos_vs_rw"]
    present = [c for c in numeric if c in frame.columns]
    grouped = frame.groupby("model")[present].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped["n_folds"] = frame.groupby("model").size()
    return grouped.sort_values("mse_mean")
