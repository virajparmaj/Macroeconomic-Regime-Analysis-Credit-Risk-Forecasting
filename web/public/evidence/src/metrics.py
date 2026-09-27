"""Error metrics, out-of-sample R-squared, Diebold-Mariano and regime-conditional error.

The metric that decides whether a forecast is useful is not R-squared against the test
mean -- it is error relative to a named benchmark. v1 reported R-squared alone, against a
test window whose variance is far below the training window's, which makes the number say
more about the split than about the model.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE and R-squared on aligned arrays.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys ``mse``, ``rmse``, ``mae`` and ``r2``.

    Raises:
        ValueError: If the inputs differ in length or contain no complete pairs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise ValueError("No complete (y_true, y_pred) pairs")
    y_true, y_pred = y_true[mask], y_pred[mask]

    err = y_true - y_pred
    mse = float(np.mean(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else float("nan")

    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(err))), "r2": r2}


def r2_oos(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_benchmark: np.ndarray | pd.Series,
) -> float:
    """Campbell-Thompson out-of-sample R-squared against an explicit benchmark.

    Defined as ``1 - SSE(model) / SSE(benchmark)``. Positive means the model beats the
    benchmark on squared error; zero means it ties; negative means it loses. Unlike plain
    R-squared this does not silently benchmark against the test-period mean, which no
    forecaster could have known in advance.

    Args:
        y_true: Observed values.
        y_pred: Model predictions.
        y_benchmark: Benchmark predictions, normally the random walk.

    Returns:
        Out-of-sample R-squared relative to the benchmark.

    Raises:
        ValueError: If lengths differ or the benchmark has zero squared error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_benchmark = np.asarray(y_benchmark, dtype=float)
    if not (y_true.shape == y_pred.shape == y_benchmark.shape):
        raise ValueError("y_true, y_pred and y_benchmark must share a shape")

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_benchmark)
    y_true, y_pred, y_benchmark = y_true[mask], y_pred[mask], y_benchmark[mask]

    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_bench = float(np.sum((y_true - y_benchmark) ** 2))
    if sse_bench == 0.0:
        raise ValueError("Benchmark has zero squared error; R2_OOS is undefined")
    return 1.0 - sse_model / sse_bench


def diebold_mariano(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    horizon: int = 1,
    loss: str = "squared",
    small_sample: bool = True,
) -> Dict[str, float]:
    """Test whether two forecasts differ in accuracy.

    Implements the Diebold-Mariano statistic with the Harvey-Leybourne-Newbold
    small-sample correction, which matters here: a 44-month test window is small enough
    that the uncorrected statistic over-rejects.

    The null is equal predictive accuracy. A negative statistic favours ``pred_a``.

    Args:
        y_true: Observed values.
        pred_a: First set of predictions.
        pred_b: Second set of predictions, normally the benchmark.
        horizon: Forecast horizon, which sets the autocovariance truncation at
            ``horizon - 1``.
        loss: ``"squared"`` or ``"absolute"``.
        small_sample: Apply the Harvey-Leybourne-Newbold correction.

    Returns:
        Dictionary with ``dm_stat``, ``p_value``, ``mean_loss_diff`` and ``n``.

    Raises:
        ValueError: If ``loss`` is unrecognised, the arrays differ in shape, or the loss
            differential has zero variance.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("y_true, pred_a and pred_b must share a shape")

    mask = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]

    if loss == "squared":
        d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    elif loss == "absolute":
        d = np.abs(y_true - pred_a) - np.abs(y_true - pred_b)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss!r}")

    n = len(d)
    if n < 3:
        raise ValueError(f"Need at least 3 observations for a DM test, got {n}")

    d_bar = float(np.mean(d))
    d_dev = d - d_bar
    gamma0 = float(np.mean(d_dev**2))
    gamma = [gamma0]
    for k in range(1, horizon):
        gamma.append(float(np.mean(d_dev[k:] * d_dev[:-k])))
    long_run_var = gamma[0] + 2.0 * sum(gamma[1:])

    if long_run_var <= 0:
        raise ValueError("Non-positive long-run variance of the loss differential")

    dm_stat = d_bar / np.sqrt(long_run_var / n)

    if small_sample:
        correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm_stat *= correction
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1)))
    else:
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "n": float(n),
    }


def regime_conditional_error(
    y_true: pd.Series,
    y_pred: pd.Series,
    regime_labels: pd.Series,
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Break forecast error down by regime.

    A headline metric averaged over a test window that is 43 quiet months and one crisis
    month says almost nothing about crisis behaviour. This splits the error so that the
    quiet-period and stress-period performance are reported separately.

    Args:
        y_true: Observed values, indexed by timestamp.
        y_pred: Model predictions, indexed by timestamp.
        regime_labels: Regime label per timestamp.
        benchmark: Optional benchmark predictions. When supplied, an ``r2_oos`` column is
            added per regime.

    Returns:
        One row per regime with ``n``, ``mse``, ``mae``, ``bias`` and optionally
        ``r2_oos``, sorted by regime label.
    """
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "regime": regime_labels})
    if benchmark is not None:
        frame["benchmark"] = benchmark
    frame = frame.dropna(subset=["y_true", "y_pred", "regime"])

    rows = []
    for regime, block in frame.groupby("regime", sort=True):
        err = block["y_true"] - block["y_pred"]
        row = {
            "regime": regime,
            "n": int(len(block)),
            "mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
        }
        if benchmark is not None and block["benchmark"].notna().all():
            try:
                row["r2_oos"] = r2_oos(block["y_true"], block["y_pred"], block["benchmark"])
            except ValueError:
                row["r2_oos"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).set_index("regime")
