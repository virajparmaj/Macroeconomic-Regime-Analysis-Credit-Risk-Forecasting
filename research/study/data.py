"""Source reconstruction. Values and their assumed availability are separate."""

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((Path(__file__).parent / "protocol.json").read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def source_path(series_id: str) -> Path:
    matches = []
    for path in (ROOT / "data/original").glob("*.csv"):
        if series_id in pd.read_csv(path, nrows=0).columns and not path.name.startswith("credit_"):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one source for {series_id}, found {len(matches)}")
    return matches[0]


def read_series(path: Path, series_id: str) -> pd.Series:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(raw.columns) != ["observation_date", series_id]:
        raise ValueError(f"Unexpected source schema: {path.name}")
    idx = pd.DatetimeIndex(pd.to_datetime(raw.observation_date, errors="raise"))
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("Source dates must be unique and sorted")
    values = raw[series_id].replace({"": np.nan, ".": np.nan, "NA": np.nan, "NaN": np.nan})
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Infinite source values are invalid")
    return pd.Series(numeric, index=idx, name=series_id)


def sources(extension: bool = False) -> tuple[dict, dict]:
    values, hashes = {}, {}
    for sid in PROTOCOL["source_ids"]:
        path = source_path(sid)
        series = read_series(path, sid)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        extra = ROOT / "data/research_external" / f"{sid}.csv"
        if extension and sid != "EA19LORSGPORGYSAM" and not extra.exists():
            raise ValueError(f"External snapshot unavailable: {sid}; run public acquisition first")
        if extension and extra.exists():
            new = read_series(extra, sid)
            # Preserve the original vintage, including the unscored partial boundary.
            series = pd.concat([series, new.loc[new.index > series.index.max()]])
            hashes[str(extra.relative_to(ROOT))] = digest(extra)
        values[sid] = series
    expected_path = Path(__file__).parent / "input_manifest.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        for name, value in expected.items():
            if hashes.get(name) != value:
                raise ValueError(
                    f"Historical input changed: {name}; declare a new protocol/input snapshot"
                )
    return values, hashes


def monthly_targets(daily: pd.Series) -> pd.DataFrame:
    numeric = daily.dropna()
    result = pd.DataFrame(
        {
            "mean_observed": daily.resample("ME").mean(),
            "mean_legacy_ffill": daily.ffill().resample("ME").mean(),
            "endpoint": daily.resample("ME").last(),
            "count": daily.resample("ME").count(),
        }
    )
    result["last_observation"] = numeric.index.to_series().resample("ME").max()
    # Empty calendar months stay missing, including the compatibility target.
    result.loc[result["count"] == 0, ["mean_observed", "mean_legacy_ffill", "endpoint"]] = np.nan
    return result


def asof_values(series: pd.Series, availability: pd.DatetimeIndex, origins) -> pd.Series:
    right = pd.DataFrame({"available": availability, "value": series.to_numpy()})
    right = right.dropna().sort_values("available").drop_duplicates("available", keep="last")
    left = pd.DataFrame({"origin": pd.DatetimeIndex(origins)})
    merged = pd.merge_asof(left, right, left_on="origin", right_on="available")
    return pd.Series(merged.value.to_numpy(), index=origins)


def macro_features(values: dict, index: pd.DatetimeIndex, five_macro=False) -> pd.DataFrame:
    raw = {}
    for sid in PROTOCOL["source_ids"][1:]:
        s = values[sid].copy()
        s.index = s.index.to_period("M").to_timestamp("M")
        if s.index.has_duplicates:
            raise ValueError(f"Duplicate reference months: {sid}")
        raw[sid] = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="ME"))
    out = pd.DataFrame(index=index)
    definitions = {
        "inflation": ("CPIAUCSL", 100 * np.log(raw["CPIAUCSL"] / raw["CPIAUCSL"].shift(12))),
        "ip_growth": ("INDPRO", 100 * np.log(raw["INDPRO"] / raw["INDPRO"].shift(1))),
        "rate": ("FEDFUNDS", raw["FEDFUNDS"]),
        "rate_change": ("FEDFUNDS", raw["FEDFUNDS"].diff()),
        "unemployment": ("UNRATE", raw["UNRATE"]),
        "unemployment_change": ("UNRATE", raw["UNRATE"].diff()),
        "sentiment_change": ("UMCSENT", raw["UMCSENT"].diff()),
        "ea_growth": ("EA19LORSGPORGYSAM", raw["EA19LORSGPORGYSAM"]),
    }
    for name, (sid, s) in definitions.items():
        if five_macro and name == "ea_growth":
            continue
        out[name] = s.shift(PROTOCOL["lag_months"][sid]).reindex(index)
    return out


@dataclass
class StudyData:
    target: pd.Series
    available: pd.Series
    features: pd.DataFrame
    current_known: pd.Series
    threshold: float
    hashes: dict
    quality: dict
    availability_ledger: pd.DataFrame


def load_data(
    target="mean_observed", delay=False, quantile=0.75, five_macro=False, extension=False
) -> StudyData:
    values, hashes = sources(extension)
    daily = values["BAMLH0A0HYM2"]
    monthly = monthly_targets(daily)
    # The historic boundary is known to end August 1; never treat it as a full month.
    monthly.loc["2022-08-31", :] = np.nan
    end = pd.Timestamp(PROTOCOL["end"])
    if extension:
        # Last complete calendar month before the frozen acquisition date.
        end = min(
            daily.index.max().to_period("M").start_time - pd.Timedelta(days=1),
            pd.Timestamp("2026-08-31"),
        )
    monthly = monthly.reindex(pd.date_range(PROTOCOL["start"], end, freq="ME"))
    s = monthly[target]
    available = pd.Series(s.index + pd.offsets.BDay(1) if delay else s.index, index=s.index)
    current = s.shift(1) if delay else s
    endpoints = monthly.endpoint
    if delay:
        endpoints = asof_values(daily, daily.index + pd.offsets.BDay(1), s.index)
        endpoints = endpoints.where(monthly["count"].gt(0))
    from .features import spread_features

    features = spread_features(current, endpoints)
    features = features.join(macro_features(values, s.index, five_macro))
    ledger = availability_ledger(values, delay)
    initial = s.loc[: PROTOCOL["threshold_end"]].dropna()
    quality = {
        "complete_month_rows": int(s.notna().sum()),
        "calendar_rows": len(s),
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
        "excluded_partial_months": ["1996-12", "2022-08"],
        "initial_threshold_n": len(initial),
        "threshold": float(initial.quantile(quantile)),
        "timing": PROTOCOL["availability"],
        "macro_vintages": "revised, not real-time",
    }
    return StudyData(
        s,
        available,
        features,
        s.notna() & ~pd.Series(delay, index=s.index),
        quality["threshold"],
        hashes,
        quality,
        ledger,
    )


def availability_ledger(values: dict, delay: bool) -> pd.DataFrame:
    rows = []
    for sid, s in values.items():
        reference = s.index if sid == "BAMLH0A0HYM2" else s.index.to_period("M").to_timestamp("M")
        if sid == "BAMLH0A0HYM2":
            available = reference + pd.offsets.BDay(1) if delay else reference
        else:
            available = reference + pd.offsets.MonthEnd(PROTOCOL["lag_months"][sid])
        original = source_path(sid)
        original_end = read_series(original, sid).index.max()
        extra = ROOT / "data/research_external" / f"{sid}.csv"
        source_hash = np.full(len(s), digest(original), dtype=object)
        if extra.exists():
            source_hash[s.index > original_end] = digest(extra)
        rows.append(
            pd.DataFrame(
                {
                    "series_id": sid,
                    "source_hash": source_hash,
                    "reference_date": reference,
                    "observation_date": s.index,
                    "available_at": available,
                    "vintage_date": None,
                    "value": s.to_numpy(),
                    "timing_assumption": "revised snapshot; assumed release lag",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
