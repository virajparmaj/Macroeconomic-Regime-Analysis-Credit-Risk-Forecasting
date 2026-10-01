"""Paired calendar-block inference; no independent-fold standard errors."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .data import PROTOCOL


def calendar_blocks(n, length, draws=5000, seed=42):
    if n < 1 or length < 1:
        raise ValueError("Empty calendar or invalid block length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (draws, int(np.ceil(n / length))))
    return ((starts[..., None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]


def paired_inference(frame, length=12, draws=5000):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Paired calendar must be unique and sorted")
    calendar = pd.date_range(frame.index.min(), frame.index.max(), freq="ME")
    data = frame.reindex(calendar)
    valid = data[["actual", "model", "benchmark"]].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No paired forecasts")
    lm = ((data.model - data.actual) ** 2).where(valid).to_numpy()
    lb = ((data.benchmark - data.actual) ** 2).where(valid).to_numpy()
    idx = calendar_blocks(len(data), length, draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_m = np.nanmean(lm[idx], axis=1)
        boot_b = np.nanmean(lb[idx], axis=1)
        gains = 1 - boot_m / boot_b
    gains = gains[np.isfinite(gains)]
    d = lb - lm
    centered = d - np.nanmean(d)
    null = np.nanmean(centered[idx], axis=1)
    null = null[np.isfinite(null)]
    lower, upper = (np.nan, np.nan)
    if len(gains) >= PROTOCOL["minimum_valid_draws"]:
        alpha = (1 - PROTOCOL["confidence"]) / 2
        lower, upper = np.quantile(gains, [alpha, 1 - alpha])
    denominator = np.nanmean(lb)
    return {
        "n": int(valid.sum()),
        "block_months": length,
        "model_rmse": float(np.sqrt(np.nanmean(lm))),
        "benchmark_rmse": float(np.sqrt(denominator)),
        "mse_gain": float(1 - np.nanmean(lm) / denominator) if denominator > 0 else np.nan,
        "lower_95": lower,
        "upper_95": upper,
        "paired_loss_gain": float(np.nanmean(d)),
        "p_value": float((1 + (np.abs(null) >= abs(np.nanmean(d))).sum()) / (len(null) + 1)),
        "valid_draws": len(gains),
        "invalid_draws": draws - len(gains),
    }


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    # Keep the declared family size even when a comparison is unavailable.
    adjusted = np.maximum.accumulate([(len(p) - i) * p[j] for i, j in enumerate(order)])
    out[order] = np.minimum(1, adjusted)
    return out


def point_metrics(frame, task, score_type):
    y, p = frame.actual.to_numpy(), frame.prediction.to_numpy()
    out = {"n": len(y)}
    if not len(y):
        return out
    if task == "level":
        out.update(
            mse=float(np.mean((p - y) ** 2)),
            rmse=float(np.sqrt(np.mean((p - y) ** 2))),
            mae=float(np.mean(abs(p - y))),
            rmse_bps=float(100 * np.sqrt(np.mean((p - y) ** 2))),
            mae_bps=float(100 * np.mean(abs(p - y))),
        )
    else:
        out["prevalence"] = float(np.mean(y))
        out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        out["average_precision"] = float(average_precision_score(y, p)) if y.sum() > 0 else np.nan
        if score_type == "probability":
            out["brier"] = float(brier_score_loss(y, p))
            out["log_loss"] = float(
                log_loss(
                    y,
                    np.clip(p, PROTOCOL["probability_clip"], 1 - PROTOCOL["probability_clip"]),
                    labels=[0, 1],
                )
            )
    return out


def comparison_rows(predictions):
    rows = []
    groups = ["scenario", "horizon", "model"]
    df = predictions.loc[(predictions.task == "level") & predictions.eligible]
    for key, group in df.groupby(groups):
        for suffix, a, b in [("regime", "S_last+M+R", "S_last+M"), ("macro", "S_last+M", "S_last")]:
            left = group.loc[group.feature_block == a].set_index("origin")
            right = group.loc[group.feature_block == b].set_index("origin")
            if left.empty or right.empty:
                continue
            if not left.index.equals(right.index):
                raise ValueError(f"Incomplete paired coverage: {key} {suffix}")
            if not np.array_equal(left.actual.to_numpy(), right.actual.to_numpy()):
                raise ValueError("Target mismatch")
            paired = pd.DataFrame(
                {"actual": left.actual, "model": left.prediction, "benchmark": right.prediction}
            )
            for length in PROTOCOL["bootstrap_lengths"]:
                rows.append(
                    dict(zip(groups, key)) | {"contrast": suffix} | paired_inference(paired, length)
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_p"] = np.nan
    selected = []
    for name in PROTOCOL["secondary_family"]:
        model, contrast, horizon = name.split("_")
        mask = (
            (result.scenario == "core")
            & (result.model == model)
            & (result.contrast == contrast)
            & (result.horizon == int(horizon[1:]))
            & (result.block_months == 12)
        )
        selected.append(result.index[mask][0] if mask.any() else None)
    p = [result.loc[i, "p_value"] if i is not None else np.nan for i in selected]
    for i, adjusted in zip(selected, holm(p)):
        if i is not None:
            result.loc[i, "holm_p"] = adjusted
    return result
