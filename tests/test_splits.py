"""Tests for :mod:`src.splits`.

The properties that matter are that no training row ever postdates a test row, and that
the embargo actually removes the training rows whose features overlap the test block. v1
had neither guarantee to test, because it used a single hard-coded date split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.splits import Fold, assert_no_overlap, fold_summary, walk_forward_folds


@pytest.fixture
def index_240() -> pd.DatetimeIndex:
    """A 240-month month-end index starting January 2000."""
    return pd.date_range("2000-01-31", periods=240, freq="ME")


def test_folds_are_chronological_and_expanding(index_240: pd.DatetimeIndex) -> None:
    """Each fold trains on strictly more data than the last, and tests strictly later."""
    folds = list(walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=3))

    assert len(folds) > 1

    for prev, curr in zip(folds, folds[1:]):
        assert len(curr.train_index) > len(prev.train_index)
        assert curr.test_index.min() > prev.test_index.max()


def test_no_training_row_postdates_any_test_row(index_240: pd.DatetimeIndex) -> None:
    """The core anti-lookahead property."""
    for fold in walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=3):
        assert fold.train_index.max() < fold.test_index.min()
        assert len(fold.train_index.intersection(fold.test_index)) == 0


def test_embargo_removes_the_expected_number_of_rows(index_240: pd.DatetimeIndex) -> None:
    """An embargo of k drops exactly k rows from the end of every training block."""
    embargo = 3
    for fold in walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=embargo):
        assert fold.n_embargoed == embargo

        gap = (fold.test_index.min().year - fold.train_index.max().year) * 12 + (
            fold.test_index.min().month - fold.train_index.max().month
        )
        assert gap == embargo + 1, "gap must clear the embargo by one month"


def test_zero_embargo_leaves_training_block_intact(index_240: pd.DatetimeIndex) -> None:
    """With no embargo the training block runs right up to the first test row."""
    folds = list(walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=0))
    for fold in folds:
        assert fold.n_embargoed == 0
        assert fold.train_index.max() < fold.test_index.min()


def test_test_blocks_tile_the_remaining_sample(index_240: pd.DatetimeIndex) -> None:
    """Test blocks are contiguous and non-overlapping, so no month is scored twice."""
    folds = list(walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=3))
    scored = pd.DatetimeIndex([])
    for fold in folds:
        assert len(scored.intersection(fold.test_index)) == 0
        scored = scored.append(fold.test_index)

    assert scored.is_monotonic_increasing
    assert not scored.has_duplicates


def test_assert_no_overlap_rejects_overlapping_blocks(index_240: pd.DatetimeIndex) -> None:
    """The guard raises when the blocks share timestamps."""
    with pytest.raises(ValueError, match="overlap"):
        assert_no_overlap(index_240[:100], index_240[90:120])


def test_assert_no_overlap_rejects_an_insufficient_gap(index_240: pd.DatetimeIndex) -> None:
    """The guard raises when the blocks are adjacent but the embargo is not cleared."""
    with pytest.raises(ValueError, match="embargo"):
        assert_no_overlap(index_240[:100], index_240[100:120], embargo=3)


def test_rejects_unsorted_index() -> None:
    """An unsorted index is a programming error, not something to silently sort."""
    bad = pd.DatetimeIndex(["2000-03-31", "2000-01-31", "2000-02-29"])
    with pytest.raises(ValueError, match="sorted"):
        list(walk_forward_folds(bad, "2000-01-31", refit_freq=1, embargo=0))


def test_rejects_origin_before_sample_start(index_240: pd.DatetimeIndex) -> None:
    """An origin preceding the data leaves nothing to train on."""
    with pytest.raises(ValueError, match="no training data"):
        list(walk_forward_folds(index_240, "1990-01-31", refit_freq=12, embargo=0))


def test_trailing_short_block_is_dropped(index_240: pd.DatetimeIndex) -> None:
    """A final partial block below ``min_test`` is not scored."""
    folds = list(walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=3, min_test=12))
    assert all(len(f.test_index) == 12 for f in folds)


def test_fold_summary_shape(index_240: pd.DatetimeIndex) -> None:
    """The summary table has one row per fold and the expected columns."""
    folds = list(walk_forward_folds(index_240, "2009-12-31", refit_freq=12, embargo=3))
    summary = fold_summary(folds)

    assert len(summary) == len(folds)
    for col in ("train_start", "train_end", "n_train", "n_embargoed", "test_start", "n_test"):
        assert col in summary.columns


def test_real_sample_yields_multiple_origins() -> None:
    """On the project's own 309-month sample the design gives many origins, not one.

    v1 reported a single split. Refitting annually from the end of 2010 gives more than
    ten test blocks over the same data, which is what makes a standard deviation across
    folds meaningful.
    """
    from src.data import load_merged_panel

    panel = load_merged_panel()
    folds = list(walk_forward_folds(panel.index, "2010-12-31", refit_freq=12, embargo=3))

    assert len(folds) >= 10
    assert isinstance(folds[0], Fold)
    assert folds[-1].test_index.max() <= panel.index.max()
