"""Loading and point-in-time alignment of the macro/credit panel.

v1 merged every series on a month-end key and applied no publication lag at all
(``notebooks/viraj/data_macroeconomic.ipynb``). That gives a model standing at the end of
month ``t`` the value of CPI, industrial production, the unemployment rate and an OECD
reference series for month ``t`` -- none of which had been published. It also uses final
revised FRED vintages throughout, so even the values that were published were not these
values.

This module applies the per-series lags recorded in :data:`src.config_v2.SERIES` and
renames the mislabelled ``GDP`` column. It reads ``data/merged_macroeconomic_credit.csv``
and never writes to it; v2 outputs go to ``data/v2/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config_v2 import (
    DATA_V2_DIR,
    MERGED_PANEL_PATH,
    PANEL_PIT_PATH,
    SERIES,
    V1_TO_V2_RENAME,
)


def load_merged_panel(path: Path | str = MERGED_PANEL_PATH) -> pd.DataFrame:
    """Load the merged macro/credit panel exactly as v1 sees it, with no alignment.

    Args:
        path: Path to the merged CSV.

    Returns:
        Frame indexed by ``Month_End`` (datetime), sorted ascending.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the expected index column or any configured series is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Merged panel not found: {path}")

    df = pd.read_csv(path, parse_dates=["Month_End"], index_col="Month_End").sort_index()

    missing = [name for name in SERIES if name not in df.columns]
    if missing:
        raise ValueError(f"Merged panel is missing configured series: {missing}")

    return df


def apply_publication_lags(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Shift each series forward by its publication lag.

    After this call, row ``t`` holds, for every series, the most recent value that had
    actually been released by the end of month ``t``.

    Args:
        df: Panel indexed by month end, using v1 column names.
        overrides: Optional per-column lag overrides in months, keyed by v1 column name.
            Use this to run the zero-lag counterfactual for comparison.

    Returns:
        A new frame with each configured column shifted by its lag. Columns not present
        in :data:`src.config_v2.SERIES` are passed through untouched.
    """
    overrides = overrides or {}
    out = df.copy()
    for name, spec in SERIES.items():
        if name not in out.columns:
            continue
        lag = overrides.get(name, spec.release_lag_months)
        if lag:
            out[name] = out[name].shift(lag)
    return out


def rename_to_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Rename v1 columns whose names misdescribe their contents.

    Currently this renames ``GDP`` to ``EA19_GDP_YoY_OECD``. That column is
    ``EA19LORSGPORGYSAM``: the OECD Leading Indicators reference series for Euro Area 19
    GDP, year on year, interpolated monthly from quarterly data. It is not US GDP, and it
    is the highest-loading variable in the v1 regime clustering.

    Args:
        df: Panel using v1 column names.

    Returns:
        A new frame with v2 column names.
    """
    return df.rename(columns=V1_TO_V2_RENAME)


def build_point_in_time_panel(
    path: Path | str = MERGED_PANEL_PATH,
    overrides: Optional[Dict[str, int]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Load, lag and rename the panel in one call.

    Args:
        path: Path to the merged CSV.
        overrides: Optional per-column publication-lag overrides in months.
        dropna: Drop the leading rows made incomplete by the longest lag.

    Returns:
        A point-in-time aligned panel under v2 column names.
    """
    panel = load_merged_panel(path)
    panel = apply_publication_lags(panel, overrides=overrides)
    panel = rename_to_v2(panel)
    return panel.dropna() if dropna else panel


def provenance_table() -> pd.DataFrame:
    """Return the series provenance and release-lag table for display in a notebook.

    Returns:
        One row per configured series with its source identifier, description, applied
        publication lag and whether the stored vintage is a revised series.
    """
    rows: List[dict] = []
    for spec in SERIES.values():
        rows.append(
            {
                "v1_name": spec.column,
                "v2_name": spec.v2_column,
                "source_id": spec.source_id,
                "release_lag_months": spec.release_lag_months,
                "revised_vintage": spec.revised,
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows).set_index("v1_name")


def save_panel(df: pd.DataFrame, path: Path | str = PANEL_PIT_PATH) -> Path:
    """Write a v2 panel to ``data/v2/``, creating the directory if needed.

    This writer is restricted to ``data/v2/`` so that the v1 CSVs under ``data/`` remain
    byte-identical.

    Args:
        df: Panel to write.
        path: Destination path. Must sit under ``data/v2/``.

    Returns:
        The path written.

    Raises:
        ValueError: If ``path`` is outside ``data/v2/``.
    """
    path = Path(path)
    if DATA_V2_DIR not in path.parents:
        raise ValueError(f"Refusing to write outside {DATA_V2_DIR}: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path
