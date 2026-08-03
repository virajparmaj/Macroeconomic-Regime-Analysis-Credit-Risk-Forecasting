"""Leakage-safe feature engineering.

This module mirrors the public API of :mod:`utilities.functions` -- ``add_lag_features``,
``add_rolling_features``, ``add_growth_rates``, ``add_interaction_terms`` -- so that
``notebooks/v2/00_audit_of_v1.ipynb`` can import both versions of the same function and
diff their output on identical input.

Three behavioural differences from v1, each correcting a specific defect:

1. **Rolling statistics exclude the current row.** v1 computed
   ``df[col].rolling(window).mean()``, which includes row ``t``. Because
   ``Credit_Spread_lag1`` and ``Credit_Spread_lag2`` were also features, the target was
   exactly recoverable as ``3 * rollmean3(t) - lag1(t) - lag2(t)``. Here every rolling
   statistic is shifted, and the shift is visible in the column name
   (``Credit_Spread_rollmean3_lag1``) so that no reader has to check the source to know
   whether row ``t`` is included.

2. **Interaction terms never touch the raw target.** v1 built
   ``Credit_Spread_x_FEDFUNDS`` from the contemporaneous target.
   :func:`add_interaction_terms` refuses to multiply a raw target column and requires its
   inputs to be lagged or non-target.

3. **The forecast target is named so the caller cannot miss it.** v1 created
   ``Target_1step_ahead`` while the consuming notebook guarded on ``col != "Target"``, so
   the guard never fired. :func:`make_forecast_target` returns the column name it created,
   and :func:`leakage_report` is the backstop.

Every function returns a new DataFrame; none mutates its input.
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config_v2 import ROLLING_WINDOW

#: Matches an explicit lag marker such as ``_lag1`` or ``_lag12`` anywhere in a name.
_LAG_MARKER = re.compile(r"_lag\d+")

#: Correlation above which a feature is treated as a restatement of the target.
LEAKAGE_CORR_THRESHOLD: float = 0.999


def add_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int] = (1, 2),
) -> pd.DataFrame:
    """Add lagged copies of the requested columns.

    Identical in behaviour to :func:`utilities.functions.add_lag_features` except that a
    new frame is returned rather than the input being mutated. Lagging is already
    leakage-safe in v1; it is reimplemented here only so that v2 has no import dependency
    on the frozen package.

    Args:
        df: Input frame indexed by a monotonically increasing period index.
        columns: Columns to lag.
        lags: Lag depths in periods. Must all be strictly positive.

    Returns:
        A new frame with one ``{col}_lag{k}`` column added per (column, lag) pair.

    Raises:
        ValueError: If any requested lag is not strictly positive.
    """
    if any(lag <= 0 for lag in lags):
        raise ValueError(f"All lags must be strictly positive, got {list(lags)}")

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame.", stacklevel=2)
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int = ROLLING_WINDOW,
    shift: int = 1,
) -> pd.DataFrame:
    """Add rolling mean and standard deviation that exclude the current row.

    This is the corrected counterpart to
    :func:`utilities.functions.add_rolling_features`. v1 wrote::

        df[f"{col}_rollmean{window}"] = df[col].rolling(window).mean()

    which is a trailing window *inclusive of row t*. Here the window is shifted by
    ``shift`` periods before being attached, so the value at row ``t`` is a function of
    ``t-shift`` back to ``t-shift-window+1`` only.

    Args:
        df: Input frame indexed by a monotonically increasing period index.
        columns: Columns to summarise.
        window: Rolling window length in periods.
        shift: Periods to shift the completed window by. The default of 1 excludes the
            current row. A shift of 0 reproduces the v1 behaviour and is rejected.

    Returns:
        A new frame with ``{col}_rollmean{window}_lag{shift}`` and
        ``{col}_rollstd{window}_lag{shift}`` added per column.

    Raises:
        ValueError: If ``window`` is below 2, or if ``shift`` is below 1. A zero shift is
            refused rather than merely discouraged: it is the exact defect this module
            exists to correct.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if shift < 1:
        raise ValueError(
            f"shift must be >= 1 so the current row is excluded, got {shift}. "
            "A shift of 0 reintroduces the v1 leak: with lag1 and lag2 also present, "
            "the target is recoverable as 3*rollmean3 - lag1 - lag2."
        )

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame.", stacklevel=2)
            continue
        rolling = out[col].rolling(window)
        out[f"{col}_rollmean{window}_lag{shift}"] = rolling.mean().shift(shift)
        out[f"{col}_rollstd{window}_lag{shift}"] = rolling.std().shift(shift)
    return out


def add_growth_rates(
    df: pd.DataFrame,
    columns: Sequence[str],
    periods: int = 1,
    as_percent: bool = True,
) -> pd.DataFrame:
    """Add period-over-period growth rates.

    A growth rate at row ``t`` uses ``t`` and ``t-periods``, so it is contemporaneous
    rather than forward-looking and is safe for a target dated ``t+h``. It is reimplemented
    here to avoid dividing by values at or near zero, which in v1 produced growth rates of
    -1583% and +268% on the Euro-Area GDP series (visible in the frozen notebook's own
    ``describe()`` output).

    Args:
        df: Input frame indexed by a monotonically increasing period index.
        columns: Columns to compute growth for.
        periods: Number of periods to difference over.
        as_percent: If True, scale the fractional change by 100.

    Returns:
        A new frame with one ``{col}_growth{periods}`` column added per column. Divisions
        by a denominator of zero yield NaN rather than an infinity.
    """
    out = df.copy()
    scale = 100.0 if as_percent else 1.0
    for col in columns:
        if col not in out.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame.", stacklevel=2)
            continue
        prev = out[col].shift(periods)
        # Guard the denominator explicitly rather than letting pct_change emit +/-inf,
        # which v1 then silently replaced with 0.0 via clip_infinities.
        denom = prev.where(prev != 0.0, other=np.nan)
        out[f"{col}_growth{periods}"] = (out[col] - prev) / denom.abs() * scale
    return out


def add_interaction_terms(
    df: pd.DataFrame,
    col_pairs: Iterable[Tuple[str, str]],
    target_stem: str | None = None,
) -> pd.DataFrame:
    """Add pairwise product interactions, refusing any that involve the raw target.

    v1's default pairs included ``('Credit_Spread', 'FEDFUNDS')``, producing
    ``Credit_Spread_x_FEDFUNDS`` -- the contemporaneous target multiplied by a known
    regressor, from which the target is recoverable by division. This function rejects any
    pair whose members are not lagged or rolling-derived when they share a stem with the
    target.

    Args:
        df: Input frame.
        col_pairs: Pairs of column names to multiply.
        target_stem: Stem of the target series, e.g. ``"Credit_Spread"``. Columns whose
            name contains this stem must carry an explicit ``_lag{k}`` marker to be
            eligible. Pass None to disable the check.

    Returns:
        A new frame with one ``{a}_x_{b}`` column added per accepted pair.

    Raises:
        ValueError: If a pair would multiply an unlagged column derived from the target.
    """
    out = df.copy()
    for col_a, col_b in col_pairs:
        missing = [c for c in (col_a, col_b) if c not in out.columns]
        if missing:
            warnings.warn(f"Columns not found for interaction: {missing}", stacklevel=2)
            continue

        if target_stem is not None:
            for col in (col_a, col_b):
                if target_stem in col and not _LAG_MARKER.search(col):
                    raise ValueError(
                        f"Refusing interaction ({col_a!r}, {col_b!r}): column {col!r} is "
                        f"derived from target stem {target_stem!r} but is not explicitly "
                        "lagged. This is the v1 Credit_Spread_x_FEDFUNDS defect."
                    )

        out[f"{col_a}_x_{col_b}"] = out[col_a] * out[col_b]
    return out


def make_forecast_target(
    df: pd.DataFrame,
    target_col: str,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, str]:
    """Create the forward-looking target column and return its name.

    v1 created ``Target_1step_ahead`` inside ``create_features`` and returned only the
    frame. The consuming notebook then dropped ``"Target"`` from its feature list, a name
    that never existed, so the target was fed back in as a predictor. Returning the name
    makes that failure mode impossible to reproduce by accident.

    Args:
        df: Input frame indexed by a monotonically increasing period index.
        target_col: Column to shift backwards into a forecast target.
        horizon: Forecast horizon in periods. Must be strictly positive.

    Returns:
        A tuple of (new frame, name of the created target column). The name has the form
        ``y_{target_col}_h{horizon}``.

    Raises:
        ValueError: If ``target_col`` is absent or ``horizon`` is not strictly positive.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in DataFrame")
    if horizon <= 0:
        raise ValueError(f"horizon must be strictly positive, got {horizon}")

    name = f"y_{target_col}_h{horizon}"
    out = df.copy()
    out[name] = out[target_col].shift(-horizon)
    return out, name


def leakage_report(
    df: pd.DataFrame,
    target: str,
    target_stem: str | None = None,
    corr_threshold: float = LEAKAGE_CORR_THRESHOLD,
) -> pd.DataFrame:
    """Flag columns that restate the target rather than predict it.

    Two independent checks are applied, because each catches a defect the other misses:

    * **Correlation.** Any column whose absolute Pearson correlation with the target
      exceeds ``corr_threshold`` is a restatement. This catches the v1
      ``Credit_Spread_x_FEDFUNDS`` style of leak.
    * **Name stem.** Any column sharing a stem with the target but carrying no explicit
      ``_lag{k}`` marker is flagged regardless of its correlation. This catches leaks that
      are nonlinear and therefore invisible to a correlation screen, and it is the check
      that would have caught v1's unshifted ``Credit_Spread_rollmean3``.

    Args:
        df: Frame containing both features and the target.
        target: Name of the target column.
        target_stem: Stem to match feature names against. Defaults to the target name
            with any ``y_`` prefix and ``_h{k}`` suffix stripped.
        corr_threshold: Absolute correlation above which a column is flagged.

    Returns:
        A frame indexed by flagged column name with columns ``abs_corr`` and ``reason``,
        sorted by ``abs_corr`` descending. Empty when no column is suspicious, which is
        the condition v2 asserts on.

    Raises:
        ValueError: If ``target`` is not present in ``df``.
    """
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame")

    if target_stem is None:
        target_stem = re.sub(r"^y_", "", re.sub(r"_h\d+$", "", target))

    y = df[target]
    rows: List[dict] = []

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        pair = pd.concat([series, y], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
            abs_corr = float("nan")
        else:
            abs_corr = abs(float(pair.corr().iloc[0, 1]))

        reasons: List[str] = []
        if np.isfinite(abs_corr) and abs_corr > corr_threshold:
            reasons.append(f"|corr| {abs_corr:.6f} > {corr_threshold}")
        if target_stem in col and not _LAG_MARKER.search(col):
            reasons.append(f"shares stem {target_stem!r} without an explicit _lag marker")

        if reasons:
            rows.append({"column": col, "abs_corr": abs_corr, "reason": "; ".join(reasons)})

    report = pd.DataFrame(rows, columns=["column", "abs_corr", "reason"])
    if report.empty:
        return report.set_index("column")
    return report.sort_values("abs_corr", ascending=False).set_index("column")


def rolling_identity_error(
    df: pd.DataFrame,
    rollmean_col: str,
    lag1_col: str,
    lag2_col: str,
    target_col: str,
    window: int = 3,
) -> float:
    """Return the maximum absolute error of the rolling-window reconstruction identity.

    For a trailing mean of length 3 that *includes* row ``t``::

        rollmean3(t) = (x(t) + x(t-1) + x(t-2)) / 3

    so ``x(t) = 3 * rollmean3(t) - x(t-1) - x(t-2)`` exactly. A near-zero return value
    therefore proves the target is algebraically recoverable from the feature set.

    Args:
        df: Frame containing all four named columns.
        rollmean_col: Rolling-mean column.
        lag1_col: One-period lag of the target.
        lag2_col: Two-period lag of the target.
        target_col: The target itself.
        window: Rolling window length used to build ``rollmean_col``.

    Returns:
        Maximum absolute reconstruction error over rows where all inputs are present.
        Values near machine epsilon indicate exact leakage; values of order the target's
        own scale indicate the identity does not hold.

    Raises:
        ValueError: If any named column is missing.
    """
    needed = [rollmean_col, lag1_col, lag2_col, target_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    frame = df[needed].dropna()
    if frame.empty:
        raise ValueError("No complete rows available to evaluate the identity")

    reconstructed = window * frame[rollmean_col] - frame[lag1_col] - frame[lag2_col]
    return float((reconstructed - frame[target_col]).abs().max())


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str,
    macro_columns: Sequence[str],
    lags: Sequence[int] = (1, 2, 3),
    window: int = ROLLING_WINDOW,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Assemble the standard v2 feature frame in one leakage-safe call.

    Args:
        df: Panel of macro series and the target, indexed by month end.
        target_col: Name of the target series.
        macro_columns: Macro regressor columns.
        lags: Lag depths applied to the target and to the macro regressors.
        window: Rolling window length.
        horizon: Forecast horizon in months.

    Returns:
        A tuple of (frame, target column name, feature column names). The frame retains
        all rows; callers decide where to drop incomplete ones.

    Note:
        The contemporaneous spread level is carried through as ``{target_col}_lag0``
        rather than under its bare name. It is a legitimate predictor -- the spread is a
        market price with no publication lag, so its value at ``t`` is known at ``t``, and
        it is exactly the random-walk forecast of ``t+1``. Dropping it would repeat v1's
        Phase 2 defect, where the sequence models were denied spread history entirely and
        then declared to have lost to a random forest that had it. Naming it ``_lag0``
        keeps the timing visible in the name and keeps :func:`leakage_report`'s stem rule
        meaningful for everything else.
    """
    out = df.copy()
    out[f"{target_col}_lag0"] = out[target_col]

    out = add_lag_features(out, [target_col, *macro_columns], lags=lags)
    out = add_rolling_features(out, [target_col, *macro_columns], window=window, shift=1)
    out = add_growth_rates(out, list(macro_columns), periods=1)
    out, target_name = make_forecast_target(out, target_col, horizon=horizon)

    exclude = {target_name, target_col}
    feature_cols = [c for c in out.columns if c not in exclude]
    return out, target_name, feature_cols
