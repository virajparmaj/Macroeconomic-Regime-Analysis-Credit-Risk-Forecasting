"""Purging depends on label availability, never shared past feature windows."""

import pandas as pd

from .data import PROTOCOL


def training_index(features, outcomes, cutoff, rolling=None, risk_only=False):
    valid = (
        features.notna().all(axis=1)
        & outcomes.actual.notna()
        & outcomes.label_available.le(cutoff)
        & (features.index < cutoff)
    )
    if risk_only:
        valid &= outcomes.risk
    if rolling:
        mature = outcomes.index[outcomes.label_available.le(cutoff)]
        if len(mature):
            start = mature.max() - pd.offsets.MonthEnd(rolling - 1)
            valid &= features.index >= start
    return features.index[valid]


def inner_folds(index, outcomes, min_train=60):
    """Three latest 12-calendar-month blocks, fitted before each validation block."""
    end = index.max()
    for back in reversed(range(PROTOCOL["inner_blocks"])):
        last = end - pd.offsets.MonthEnd(PROTOCOL["inner_block_months"] * back)
        first = last - pd.offsets.MonthEnd(PROTOCOL["inner_block_months"] - 1)
        cutoff = first - pd.offsets.MonthEnd(1)
        train = index[(index < first) & outcomes.loc[index, "label_available"].le(cutoff)]
        valid = index[(index >= first) & (index <= last)]
        if len(train) >= min_train and len(valid):
            yield train, valid
