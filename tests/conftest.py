"""Shared pytest configuration for the v2 test suite.

Puts the project root on ``sys.path`` so that both the new ``src`` package and the frozen
``utilities`` package are importable. The v1 package is imported deliberately: one of the
required tests asserts that the *old* behaviour still exhibits the leak, so that the fix
cannot be silently reverted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def monthly_index() -> pd.DatetimeIndex:
    """A 60-month month-end index starting January 2000."""
    return pd.date_range("2000-01-31", periods=60, freq="ME")


@pytest.fixture
def known_series(monthly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """A frame whose rolling means are known exactly by construction.

    ``x`` is 1, 2, 3, ... so a trailing 3-month mean inclusive of row t at position i
    (zero-based) is ``i`` for i >= 2, and the same mean excluding row t is ``i - 1``.
    """
    return pd.DataFrame({"x": np.arange(1.0, len(monthly_index) + 1.0)}, index=monthly_index)


@pytest.fixture
def real_panel() -> pd.DataFrame:
    """The real merged panel, used for the end-to-end leakage assertions."""
    from src.data import load_merged_panel

    return load_merged_panel()
