"""Backward-looking features and independently constructed future labels."""

import numpy as np
import pandas as pd

SPREAD = ["mean", "change1", "change3", "mean3", "vol6"]
ENDPOINT = ["endpoint", "endpoint_gap"]
BLOCKS = ["S_mean", "S_last", "S_last+M", "S_last+M+R"]


def spread_features(mean: pd.Series, endpoint: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": mean,
            "change1": mean.diff(),
            "change3": mean.diff(3),
            "mean3": mean.rolling(3).mean(),
            "vol6": mean.diff().rolling(6).std(ddof=1),
            "endpoint": endpoint,
            "endpoint_gap": endpoint - mean,
        }
    )


def labels(
    target: pd.Series, available: pd.Series, h: int, task: str, threshold: float
) -> pd.DataFrame:
    if h < 1 or task not in {"level", "state", "entry"}:
        raise ValueError("Invalid task or horizon")
    future = pd.concat([target.shift(-j) for j in range(1, h + 1)], axis=1)
    complete = future.notna().all(axis=1)
    y = target.shift(-h)
    if task == "state":
        y = y.ge(threshold).astype(float).where(y.notna())
    elif task == "entry":
        y = future.ge(threshold).any(axis=1).astype(float).where(complete)
    return pd.DataFrame(
        {
            "actual": y,
            "label_available": available.shift(-h),
            "target_date": target.index.to_series().shift(-h),
            "risk": target.lt(threshold) & target.notna(),
            "path_observed": complete,
        }
    )


def column_names(frame: pd.DataFrame, block: str) -> list[str]:
    if block not in BLOCKS:
        raise ValueError(block)
    cols = SPREAD.copy()
    if block != "S_mean":
        cols += ENDPOINT
    if "+M" in block:
        cols += [c for c in frame if c not in SPREAD + ENDPOINT]
    return cols
