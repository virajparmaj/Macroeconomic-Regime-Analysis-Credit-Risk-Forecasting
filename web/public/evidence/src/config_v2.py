"""Configuration for the v2 rebuild: series metadata, publication lags, paths and seeds.

This module exists alongside the frozen root-level ``config.py``. It does not import it
and does not modify it.

The material addition over v1 is :data:`SERIES`, which records for every input series
both its true provenance (the FRED identifier it actually came from) and its real-world
publication lag in months. v1 merged every series on a month-end key with no lag at all,
which silently assumed that a forecaster standing at month end knew that month's CPI,
industrial production and OECD reference series. None of those are published then.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Tuple

# =========================================================
# Paths
# =========================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
ORIGINAL_DATA_DIR: Final[Path] = DATA_DIR / "original"
MERGED_PANEL_PATH: Final[Path] = DATA_DIR / "merged_macroeconomic_credit.csv"

# v2 writes only into these locations. Existing data/*.csv files are read-only inputs.
DATA_V2_DIR: Final[Path] = DATA_DIR / "v2"
PANEL_PIT_PATH: Final[Path] = DATA_V2_DIR / "panel_pit.parquet"

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
METRICS_CSV_PATH: Final[Path] = RESULTS_DIR / "metrics.csv"

# =========================================================
# Seeds and split boundaries
# =========================================================

RANDOM_SEED: Final[int] = 42

TRAIN_END_DATE: Final[str] = "2018-12-31"
TEST_START_DATE: Final[str] = "2019-01-01"
TEST_END_DATE: Final[str] = "2022-08-31"

#: Default forecast horizon in months. v2 forecasts t+1; v1's headline fitted t.
FORECAST_HORIZON: Final[int] = 1

#: Default rolling window for trend features.
ROLLING_WINDOW: Final[int] = 3

#: Default lag set for autoregressive features.
DEFAULT_LAGS: Final[Tuple[int, ...]] = (1, 2, 3)


@dataclass(frozen=True)
class SeriesSpec:
    """Provenance and release timing for a single input series.

    Attributes:
        column: Column name as it appears in ``data/merged_macroeconomic_credit.csv``.
        v2_column: Name used in v2. Differs from ``column`` only where the v1 name was
            actively misleading.
        source_id: The FRED / OECD identifier the series was actually downloaded from.
        description: Human-readable description of what the series measures.
        release_lag_months: Whole months between the reference period end and the first
            public release. Applied as a shift so that a model standing at month ``t``
            only sees values that had actually been published by then.
        revised: Whether the vintage in ``data/original/`` is a final revised series
            rather than a real-time first print. Where True, even a correct lag does not
            fully reconstruct what a forecaster would have seen.
    """

    column: str
    v2_column: str
    source_id: str
    description: str
    release_lag_months: int
    revised: bool


#: Metadata for every column of the merged panel.
#:
#: Release lags are whole months and are deliberately conservative (rounded down), so
#: they understate rather than overstate the alignment problem. Real release calendars:
#: UNRATE lands on the first Friday of M+1; CPIAUCSL and INDPRO mid M+1; the OECD
#: reference series runs two to three months behind and is revised for years afterwards.
SERIES: Final[Dict[str, SeriesSpec]] = {
    "CPI": SeriesSpec(
        column="CPI",
        v2_column="CPI",
        source_id="CPIAUCSL",
        description="Consumer Price Index for All Urban Consumers, All Items, US city average",
        release_lag_months=1,
        revised=True,
    ),
    "FEDFUNDS": SeriesSpec(
        column="FEDFUNDS",
        v2_column="FEDFUNDS",
        source_id="FEDFUNDS",
        description="Federal Funds Effective Rate, monthly average",
        release_lag_months=1,
        revised=False,
    ),
    "Industrial_Production": SeriesSpec(
        column="Industrial_Production",
        v2_column="Industrial_Production",
        source_id="INDPRO",
        description="Industrial Production Total Index",
        release_lag_months=1,
        revised=True,
    ),
    "GDP": SeriesSpec(
        column="GDP",
        # The v1 name is wrong in a way that changed how the result was described:
        # this is not US GDP, and it is the highest-loading variable in the regime PCA.
        v2_column="EA19_GDP_YoY_OECD",
        source_id="EA19LORSGPORGYSAM",
        description=(
            "OECD Leading Indicators reference series, GDP, Euro Area 19 countries, "
            "year-on-year growth, monthly interpolation of a quarterly series"
        ),
        release_lag_months=3,
        revised=True,
    ),
    "Unemployment_Rate": SeriesSpec(
        column="Unemployment_Rate",
        v2_column="Unemployment_Rate",
        source_id="UNRATE",
        description="Civilian Unemployment Rate",
        release_lag_months=1,
        revised=True,
    ),
    "Consumer_Sentiment": SeriesSpec(
        column="Consumer_Sentiment",
        v2_column="Consumer_Sentiment",
        source_id="UMCSENT",
        description="University of Michigan Consumer Sentiment Index",
        release_lag_months=0,
        revised=False,
    ),
    "Credit_Spread": SeriesSpec(
        column="Credit_Spread",
        v2_column="Credit_Spread",
        source_id="BAMLH0A0HYM2",
        description="ICE BofA US High Yield Index Option-Adjusted Spread, monthly mean of daily",
        release_lag_months=0,
        revised=False,
    ),
}

#: The forecast target.
TARGET_COLUMN: Final[str] = "Credit_Spread"

#: Macro regressors, under their v2 names.
MACRO_COLUMNS: Final[List[str]] = [
    spec.v2_column for spec in SERIES.values() if spec.column != TARGET_COLUMN
]

#: Mapping from v1 column names to v2 column names, for renaming a loaded panel.
V1_TO_V2_RENAME: Final[Dict[str, str]] = {
    spec.column: spec.v2_column for spec in SERIES.values() if spec.column != spec.v2_column
}

#: NBER US recession months in the sample window, used to sanity-check regime labels.
#: Peak and trough months inclusive, from the NBER business cycle reference dates.
NBER_RECESSIONS: Final[List[Tuple[str, str]]] = [
    ("2001-03-31", "2001-11-30"),
    ("2007-12-31", "2009-06-30"),
    ("2020-02-29", "2020-04-30"),
]


def embargo_months(max_lag: int = max(DEFAULT_LAGS), rolling_window: int = ROLLING_WINDOW) -> int:
    """Return the embargo length implied by the feature construction.

    Any feature built from a lag of ``max_lag`` or a rolling window of ``rolling_window``
    months carries information from up to that many months earlier. Training rows within
    that distance of the first test row therefore overlap the test period and must be
    dropped from the end of the training block.

    Args:
        max_lag: Longest lag used in feature construction.
        rolling_window: Longest rolling window used in feature construction.

    Returns:
        Number of months to remove from the end of each training block.
    """
    return max(max_lag, rolling_window)
