"""Training-only alarm selection and one-to-one event accounting."""

import numpy as np
import pandas as pd

from .data import PROTOCOL


def entry_dates(target, q):
    return target.index[(target >= q) & (target.shift(1) < q)]


def score_alarms(origins, alarms, entries, h):
    episodes, current = [], []
    previous = None
    for date, alarm in zip(origins, alarms):
        adjacent = previous is not None and date == previous + pd.offsets.MonthEnd(1)
        if current and (not alarm or not adjacent):
            episodes.append(current)
            current = []
        if alarm:
            current.append(date)
        previous = date
    if current:
        episodes.append(current)
    used, records = set(), []
    for episode in episodes:
        candidates = [
            (event, warning)
            for event in entries
            if event not in used
            for warning in episode
            if warning < event <= warning + pd.offsets.MonthEnd(h)
        ]
        if candidates:
            event, warning = min(candidates)
            used.add(event)
            lead = (event.year - warning.year) * 12 + event.month - warning.month
        else:
            event, warning, lead = None, None, None
        records.append(
            {
                "alarm_start": episode[0],
                "alarm_end": episode[-1],
                "entry": event,
                "first_valid_warning": warning,
                "lead_months": lead,
                "false_alarm": event is None,
            }
        )
    return records, sorted(set(entries) - used)


def choose_alarm_threshold(calibration, entries, h, min_months=24):
    if len(calibration) < min_months or not len(entries):
        return np.inf, "insufficient_calibration"
    choices = np.unique(np.r_[calibration.prediction.to_numpy(), np.inf])
    candidates = []
    for threshold in choices:
        records, missed = score_alarms(
            calibration.index, calibration.prediction.ge(threshold), entries, h
        )
        false = sum(r["false_alarm"] for r in records)
        if false / (len(calibration) / 12) <= PROTOCOL["alarm_false_episodes_per_year"]:
            candidates.append((len(missed), false, -threshold, threshold))
    return min(candidates)[-1], "calibrated"  # Infinity is always a feasible no-alarm policy.


def alarm_series(predictions, target, q, h):
    entries = entry_dates(target, q)
    scores = predictions.set_index("origin").sort_index()
    records = []
    for origin, row in scores.loc["2009-01-31":].iterrows():
        start = origin - pd.offsets.MonthEnd(PROTOCOL["alarm_calibration_months"])
        cal = scores.loc[(scores.index >= start) & (scores.index < origin)]
        cal = cal.loc[cal.label_available.le(origin) & cal.eligible & cal.prediction.notna()]
        # Only events with observable outcomes and an eligible warning window count.
        seen = [
            e
            for e in entries
            if e <= origin and any(t < e <= t + pd.offsets.MonthEnd(h) for t in cal.index)
        ]
        threshold, status = choose_alarm_threshold(
            cal, seen, h, PROTOCOL["alarm_minimum_risk_months"]
        )
        records.append(
            {
                "origin": origin,
                "alarm_threshold": threshold,
                "alarm": bool(row.eligible and row.prediction >= threshold),
                "calibration_status": status,
            }
        )
    return pd.DataFrame(records)
