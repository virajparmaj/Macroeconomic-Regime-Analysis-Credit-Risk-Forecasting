"""Naive benchmarks that any proposed model has to beat.

The word "baseline" in the naive sense appears nowhere in the thirty files of ``notes/``,
and no notebook in ``notebooks/viraj/`` fits one. That omission is what allowed a random
forest with a test R-squared of 0.61 on the honest task to be described as a success: the
random walk scores 0.65 on the same task, so the model is worse than assuming nothing
changes.

Every function returns predictions aligned to the index it was asked to predict, so that
the caller can feed them straight into :func:`src.metrics.r2_oos` and
:func:`src.metrics.diebold_mariano`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_forecast(level: pd.Series, test_index: pd.Index) -> pd.Series:
    """Predict the next value with the current one.

    For a target ``y(t) = spread(t + h)`` built by :func:`src.features.make_forecast_target`
    and features attached to row ``t``, the random-walk forecast is simply the spread level
    at row ``t``.

    Args:
        level: Contemporaneous target level, indexed by timestamp.
        test_index: Timestamps to predict.

    Returns:
        Predictions indexed by ``test_index``.

    Raises:
        ValueError: If any test timestamp is absent from ``level``.
    """
    missing = test_index.difference(level.index)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} test timestamps absent from level series")
    return level.reindex(test_index).rename("random_walk")


def train_mean_forecast(y_train: pd.Series, test_index: pd.Index) -> pd.Series:
    """Predict the training-sample mean for every test point.

    This is the honest constant benchmark. It is not the same as the test-period mean that
    plain R-squared implicitly compares against, which a forecaster could not have known.

    Args:
        y_train: Training targets.
        test_index: Timestamps to predict.

    Returns:
        A constant series indexed by ``test_index``.

    Raises:
        ValueError: If ``y_train`` has no finite values.
    """
    mean = float(y_train.dropna().mean())
    if not np.isfinite(mean):
        raise ValueError("Training target has no finite values")
    return pd.Series(mean, index=test_index, name="train_mean")


def ar1_forecast(
    level_train: pd.Series,
    y_train: pd.Series,
    level_test: pd.Series,
) -> pd.Series:
    """Fit ``y = a + b * level`` on the training block and apply it to the test block.

    With ``y`` the one-step-ahead spread and ``level`` the current spread, this is an AR(1)
    in levels estimated by ordinary least squares. Fitting is confined to the training
    block, so no test information reaches the coefficients.

    Args:
        level_train: Contemporaneous level over the training block.
        y_train: Target over the training block.
        level_test: Contemporaneous level over the test block.

    Returns:
        Predictions indexed like ``level_test``.

    Raises:
        ValueError: If fewer than three complete training pairs are available.
    """
    pair = pd.concat([level_train.rename("x"), y_train.rename("y")], axis=1).dropna()
    if len(pair) < 3:
        raise ValueError(f"Need at least 3 complete training pairs, got {len(pair)}")

    slope, intercept = np.polyfit(pair["x"].to_numpy(), pair["y"].to_numpy(), deg=1)
    return pd.Series(intercept + slope * level_test.to_numpy(), index=level_test.index, name="ar1")


def all_baselines(
    level_train: pd.Series,
    y_train: pd.Series,
    level_test: pd.Series,
) -> pd.DataFrame:
    """Compute every baseline for one fold.

    Args:
        level_train: Contemporaneous level over the training block.
        y_train: Target over the training block.
        level_test: Contemporaneous level over the test block.

    Returns:
        Frame indexed like ``level_test`` with one column per baseline.
    """
    return pd.DataFrame(
        {
            "random_walk": random_walk_forecast(level_test, level_test.index),
            "ar1": ar1_forecast(level_train, y_train, level_test),
            "train_mean": train_mean_forecast(y_train, level_test.index),
        }
    )
