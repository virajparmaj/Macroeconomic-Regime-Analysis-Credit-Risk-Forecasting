"""Expanding-window walk-forward splits with purge and embargo.

v1 evaluated on a single chronological split: train to 2018-12, test 2019-01 onward. One
split on 44 test months of a series with lag-1 autocorrelation 0.963 yields an effective
sample size of roughly six independent observations, so the reported metric carried no
usable error bar. Worse, the test window happens to contain far less spread variance than
the training window, which is why macro-only R-squared values come out deeply negative:
the denominator of R-squared is the test variance, not the training variance.

This module produces many origins instead of one, and removes the rows at the end of each
training block whose features overlap the test block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence

import pandas as pd

from src.config_v2 import embargo_months


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Attributes:
        origin: Last timestamp of the usable training data for this fold, before the
            embargo is applied.
        train_index: Timestamps used for fitting.
        test_index: Timestamps used for scoring.
        n_embargoed: Number of training rows dropped by the embargo.
    """

    origin: pd.Timestamp
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    n_embargoed: int


def assert_no_overlap(
    train_index: pd.Index,
    test_index: pd.Index,
    embargo: int = 0,
) -> None:
    """Raise if a fold's training and test blocks overlap or sit within the embargo.

    Args:
        train_index: Training timestamps.
        test_index: Test timestamps.
        embargo: Required gap, in index positions of the monthly grid, between the last
            training timestamp and the first test timestamp.

    Raises:
        ValueError: If the two blocks share timestamps, if training data postdates the
            first test timestamp, or if the realised gap is smaller than ``embargo``.
    """
    shared = train_index.intersection(test_index)
    if len(shared) > 0:
        raise ValueError(f"Train and test overlap on {len(shared)} timestamps: {list(shared[:5])}")

    if len(train_index) == 0 or len(test_index) == 0:
        return

    last_train = train_index.max()
    first_test = test_index.min()
    if last_train >= first_test:
        raise ValueError(
            f"Training data extends to {last_train.date()}, at or beyond the first test "
            f"timestamp {first_test.date()}"
        )

    gap_months = (first_test.year - last_train.year) * 12 + (first_test.month - last_train.month)
    if gap_months <= embargo:
        raise ValueError(
            f"Gap between last train ({last_train.date()}) and first test "
            f"({first_test.date()}) is {gap_months} month(s), which does not clear the "
            f"required embargo of {embargo}"
        )


def walk_forward_folds(
    index: pd.DatetimeIndex,
    initial_train_end: str | pd.Timestamp,
    refit_freq: int = 12,
    embargo: int | None = None,
    min_test: int = 1,
) -> Iterator[Fold]:
    """Yield expanding-window folds with an embargo at the end of each training block.

    Each fold trains on everything up to its origin, drops the final ``embargo`` rows of
    that block, and tests on the next ``refit_freq`` observations.

    Args:
        index: Sorted DatetimeIndex of the full sample.
        initial_train_end: Origin of the first fold. Training for fold 0 is all data at or
            before this timestamp.
        refit_freq: Number of observations in each test block, and the step between
            successive origins.
        embargo: Rows to drop from the end of each training block. Defaults to
            :func:`src.config_v2.embargo_months`, which is the longest lag or rolling
            window used in feature construction.
        min_test: Minimum test-block size. A trailing block shorter than this is dropped
            rather than scored on too few points.

    Yields:
        :class:`Fold` instances in chronological order.

    Raises:
        ValueError: If ``index`` is not sorted and unique, or if ``initial_train_end``
            leaves no training data.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be sorted and free of duplicates")

    if embargo is None:
        embargo = embargo_months()

    origin = pd.Timestamp(initial_train_end)
    n = len(index)

    start_pos = int(index.searchsorted(origin, side="right"))
    if start_pos == 0:
        raise ValueError(
            f"initial_train_end {origin.date()} precedes the sample start "
            f"{index[0].date()}; no training data available"
        )

    pos = start_pos
    while pos < n:
        test_slice = index[pos : pos + refit_freq]
        if len(test_slice) < min_test:
            break

        full_train = index[:pos]
        train_slice = full_train[: len(full_train) - embargo] if embargo else full_train
        if len(train_slice) == 0:
            pos += refit_freq
            continue

        fold = Fold(
            origin=index[pos - 1],
            train_index=pd.DatetimeIndex(train_slice),
            test_index=pd.DatetimeIndex(test_slice),
            n_embargoed=len(full_train) - len(train_slice),
        )
        assert_no_overlap(fold.train_index, fold.test_index, embargo=embargo)
        yield fold

        pos += refit_freq


def fold_summary(folds: Sequence[Fold]) -> pd.DataFrame:
    """Summarise a list of folds as a frame, for printing in a notebook.

    Args:
        folds: Folds to summarise.

    Returns:
        One row per fold with origin, train and test boundaries, sizes and embargo count.
    """
    rows: List[dict] = []
    for i, fold in enumerate(folds):
        rows.append(
            {
                "fold": i,
                "train_start": fold.train_index.min().date(),
                "train_end": fold.train_index.max().date(),
                "n_train": len(fold.train_index),
                "n_embargoed": fold.n_embargoed,
                "test_start": fold.test_index.min().date(),
                "test_end": fold.test_index.max().date(),
                "n_test": len(fold.test_index),
            }
        )
    return pd.DataFrame(rows).set_index("fold")
