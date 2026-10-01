"""Adversarial checks of the forecasting contract, independent of real outcomes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research.study.data import asof_values, load_data, monthly_targets, read_series
from research.study.events import choose_alarm_threshold, score_alarms
from research.study.features import labels, spread_features
from research.study.inference import calendar_blocks, holm, paired_inference, point_metrics
from research.study.models import Representation, baseline, fitted_model, predict
from research.study.runner import atomic_json, forecast_origin, job_specs, scenarios
from research.study.splits import inner_folds, training_index


@pytest.fixture
def path_series():
    idx = pd.date_range("2000-01-31", periods=180, freq="ME")
    return pd.Series(5 + np.sin(np.arange(180) / 7), index=idx)


@pytest.mark.parametrize("h", [1, 3, 12])
def test_mature_labels_and_unknown_future(path_series, h):
    s = path_series
    y = labels(s, s.index.to_series(), h, "level", 6)
    x = spread_features(s, s)
    cutoff = s.index[100]
    tr = training_index(x, y, cutoff)
    assert (y.loc[tr, "label_available"] <= cutoff).all()
    assert y.actual.tail(h).isna().all()
    assert y.loc[s.index[100], "actual"] == s.iloc[100 + h]
    assert s.index[100 - h + 1] not in tr


def test_missing_intermediate_future_and_h1_identity(path_series):
    s = path_series.copy()
    s.iloc[40] = np.nan
    entry = labels(s, s.index.to_series(), 3, "entry", 5)
    assert pd.isna(entry.actual.iloc[38])
    state = labels(s, s.index.to_series(), 1, "state", 5)
    entry = labels(s, s.index.to_series(), 1, "entry", 5)
    assert state.actual.equals(entry.actual)


def test_cross_then_recede_and_risk():
    s = pd.Series([4.0, 8.0, 4.0, 4.0], index=pd.date_range("2000-01-31", periods=4, freq="ME"))
    state = labels(s, s.index.to_series(), 2, "state", 6)
    entry = labels(s, s.index.to_series(), 2, "entry", 6)
    assert state.actual.iloc[0] == 0
    assert entry.actual.iloc[0] == 1
    assert not entry.risk.iloc[1]


def test_future_perturbation_features_and_prediction():
    from copy import deepcopy

    data = load_data()
    cutoff = pd.Timestamp("2010-01-31")
    changed = deepcopy(data)
    changed.target.loc[changed.target.index > cutoff] = 1e6
    changed.features.loc[changed.features.index > cutoff, :] = -1e6
    job = {**scenarios("core")[0], "h": 1, "task": "level", "model": "ridge", "block": "S_last+M+R"}
    results = []
    for d in [data, changed]:
        y = labels(d.target, d.available, 1, "level", d.threshold)
        x = d.features
        results.append(forecast_origin(job, d, y, x, cutoff, {}))
    assert results[0]["prediction"] == results[1]["prediction"]
    assert results[0]["fit_details"] == results[1]["fit_details"]
    assert data.threshold == changed.threshold


def test_frozen_threshold_ignores_test_extremes(path_series):
    s = path_series
    cutoff = s.index[100]
    q = s.loc[:cutoff].quantile(0.75)
    changed = s.copy()
    changed.loc[changed.index > cutoff] = 1e8
    assert changed.loc[:cutoff].quantile(0.75) == q


def test_inner_maturity_and_train_only_scaling(path_series):
    s = path_series
    x = spread_features(s, s).dropna()
    y = labels(s, s.index.to_series(), 3, "level", 5)
    for train, validation in inner_folds(x.index[:-3], y):
        assert y.loc[train, "label_available"].max() < validation.min()
        r = Representation().fit(x.loc[train])
        expected = x.loc[train].mean().to_numpy()
        np.testing.assert_allclose(r.scale.mean_, expected)
        changed = x.copy()
        changed.loc[validation, :] = 1e9
        assert np.array_equal(r.scale.mean_, expected)


def test_availability_asof():
    dates = pd.to_datetime(["2020-01-01", "2020-02-01"])
    s = pd.Series([1, 999], index=dates)
    available = pd.to_datetime(["2020-02-15", "2020-03-15"])
    result = asof_values(s, available, pd.to_datetime(["2020-02-29", "2020-03-31"]))
    assert result.tolist() == [1, 999]


def test_real_boundary_and_target_reconciliation():
    from research.study.data import ROOT, sources

    d = load_data()
    assert len(d.target) == 307 and d.target.index[-1] == pd.Timestamp("2022-07-31")
    assert d.quality["initial_threshold_n"] == 144
    original = pd.read_csv(
        ROOT / "data/merged_macroeconomic_credit.csv", index_col="Month_End", parse_dates=True
    )
    raw, _ = sources()
    monthly = monthly_targets(raw["BAMLH0A0HYM2"])
    np.testing.assert_allclose(monthly.mean_legacy_ffill, original.Credit_Spread, atol=1e-12)
    assert monthly.loc["2022-08-31", "count"] == 1
    for h, n in [(1, 162), (3, 160)]:
        y = labels(d.target, d.available, h, "level", d.threshold)
        assert y.loc["2009-01-31":, "actual"].notna().sum() == n


def test_gap_remains_gap():
    dates = pd.to_datetime(["2020-01-15", "2020-03-15"])
    monthly = monthly_targets(pd.Series([4.0, 5.0], index=dates))
    assert pd.isna(monthly.loc["2020-02-29", "mean_legacy_ffill"])
    assert pd.isna(monthly.loc["2020-02-29", "mean_observed"])


def test_malformed_and_duplicate_sources(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("observation_date,X\n2020-01-01,nope\n")
    with pytest.raises(ValueError):
        read_series(path, "X")
    path.write_text("observation_date,X\n2020-01-01,1\n2020-01-01,2\n")
    with pytest.raises(ValueError, match="unique"):
        read_series(path, "X")


def test_constant_baselines_and_units(path_series):
    s = path_series * 0 + 5
    x = spread_features(s, s)
    y = labels(s, s.index.to_series(), 1, "level", 6)
    train = x.dropna().index[:-2]
    for name in ["mean_persistence", "endpoint_persistence", "training_mean", "ar1"]:
        assert baseline(name, s, y, train, s.index[-2], x, 1, 6, "level") == pytest.approx(5)
    m = point_metrics(pd.DataFrame({"actual": [4, 4], "prediction": [5, 5]}), "level", "level_pp")
    assert m["rmse_bps"] == 100


def test_single_class_and_undefined_auc(path_series):
    x = spread_features(path_series, path_series).dropna()
    y = pd.Series(0.0, index=x.index)
    fit = fitted_model(x, y, None, "logistic", False, "entry")
    assert fit[2]["fallback"] == "single_class_prevalence"
    p = predict(fit, x.iloc[:2], "entry")
    assert np.all(p > 0) and np.all(p < 1)
    m = point_metrics(pd.DataFrame({"actual": [0, 0], "prediction": p}), "entry", "probability")
    assert np.isnan(m["auc"]) and np.isfinite(m["log_loss"])


def test_regime_train_transform_is_deterministic(path_series):
    x = spread_features(path_series, path_series).dropna()
    x["macro"] = np.sin(np.arange(len(x)))
    a = Representation(True, True).fit(x.iloc[:100])
    b = Representation(True, True).fit(x.iloc[:100])
    np.testing.assert_array_equal(a.transform(x), b.transform(x))
    assert sum(a.occupancy) == 100


def test_calendar_blocks_and_paired_identity():
    idx = calendar_blocks(24, 6, 100)
    assert idx.shape == (100, 24)
    for start in range(0, 24, 6):
        assert np.all(np.diff(idx[:, start : start + 6], axis=1) % 24 == 1)
    dates = pd.date_range("2000-01-31", periods=24, freq="ME")
    frame = pd.DataFrame(
        {"actual": np.arange(24), "model": np.arange(24) + 1, "benchmark": np.arange(24) + 1},
        index=dates,
    )
    r = paired_inference(frame)
    assert r["mse_gain"] == 0 and r["p_value"] == 1
    frame.index = [dates[0]] * 24
    with pytest.raises(ValueError, match="unique"):
        paired_inference(frame)


def test_event_one_to_one_and_calendar_gap():
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    rows, missed = score_alarms(dates, [True, True, False, True, False, False], [dates[2]], 3)
    assert len(rows) == 2 and sum(not r["false_alarm"] for r in rows) == 1
    assert len(missed) == 0
    rows, _ = score_alarms(dates[[0, 2]], [True, True], [], 1)
    assert len(rows) == 2


def test_alarm_insufficient_and_false_budget():
    dates = pd.date_range("2000-01-31", periods=36, freq="ME")
    frame = pd.DataFrame({"prediction": np.linspace(0.01, 0.99, 36)}, index=dates)
    t, status = choose_alarm_threshold(frame, [], 3)
    assert np.isinf(t) and status == "insufficient_calibration"
    t, status = choose_alarm_threshold(frame, [dates[-1] + pd.offsets.MonthEnd(1)], 3)
    rows, _ = score_alarms(dates, frame.prediction >= t, [dates[-1] + pd.offsets.MonthEnd(1)], 3)
    assert sum(r["false_alarm"] for r in rows) / (len(frame) / 12) <= 1


def test_holm_and_job_uniqueness():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    for profile in ["core", "sensitivity", "external"]:
        jobs = job_specs(profile)
        assert len({j["id"] for j in jobs}) == len(jobs)


def test_atomic_checkpoint(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"done": True})
    assert json.loads(path.read_text()) == {"done": True}
    assert not path.with_suffix(".json.tmp").exists()


def test_resume_rejects_changed_manifest(tmp_path, monkeypatch):
    import research.study.runner as runner

    out = tmp_path / "results/research/example"
    out.mkdir(parents=True)
    frozen = {
        "profile": "smoke",
        "code_hash": "before",
        "protocol": {},
        "input_hashes": {},
        "environment": {},
    }
    (out / "manifest.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "manifest", lambda profile: frozen | {"code_hash": "after"})
    with pytest.raises(ValueError, match="manifest mismatch"):
        runner.execute("smoke", "example")


def test_completed_checkpoints_are_not_resubmitted(tmp_path):
    from research.study.runner import run_jobs

    marker = tmp_path / "completed.csv"
    marker.write_text("unchanged")
    import time

    run_jobs([], tmp_path, 1, time.monotonic() + 1)
    assert marker.read_text() == "unchanged"


def test_paired_comparison_rejects_missing_forecast():
    from research.study.inference import comparison_rows

    rows = []
    for block in ["S_last+M", "S_last+M+R"]:
        for date in pd.date_range("2009-01-31", periods=3, freq="ME"):
            if block.endswith("+R") and date.month == 2:
                continue
            rows.append(
                {
                    "scenario": "core",
                    "horizon": 1,
                    "model": "ridge",
                    "task": "level",
                    "eligible": True,
                    "feature_block": block,
                    "origin": date,
                    "actual": 5.0,
                    "prediction": 4.0,
                }
            )
    with pytest.raises(ValueError, match="paired coverage"):
        comparison_rows(pd.DataFrame(rows))


def test_delay_removes_current_state_and_uses_past_mean():
    primary, delayed = load_data(), load_data(delay=True)
    assert not delayed.current_known.any()
    np.testing.assert_allclose(delayed.features["mean"].iloc[1:], primary.target.iloc[:-1])
    assert (delayed.available > delayed.available.index).all()


def test_symmetric_winsorization_is_valid():
    from utilities.functions import winsorize_series

    values = pd.Series(range(10))
    result = winsorize_series(values, limits=(0.1, 0.1))
    assert result.min() == 1 and result.max() == 8
    for limits in [(-0.1, 0.1), (0.6, 0.1), (0.5, 0.5)]:
        with pytest.raises(ValueError):
            winsorize_series(values, limits=limits)


def test_future_perturbation_through_data_loader(monkeypatch):
    """Rebuild transformed features and threshold after replacing future source values."""
    from copy import deepcopy

    import research.study.data as module

    values, hashes = module.sources()
    before = module.load_data()
    cutoff = pd.Timestamp("2010-01-31")
    changed = deepcopy(values)
    for series in changed.values():
        series.loc[series.index > cutoff] = 1e6
    monkeypatch.setattr(module, "sources", lambda extension=False: (changed, hashes))
    after = module.load_data()
    assert_frame_equal(before.features.loc[:cutoff], after.features.loc[:cutoff])
    assert before.threshold == after.threshold


def test_ridge_inner_preprocessing_never_fits_validation(monkeypatch, path_series):
    import research.study.models as module
    from research.study.features import labels

    x = spread_features(path_series, path_series).dropna()
    x["macro"] = np.sin(np.arange(len(x)))
    outcomes = labels(path_series, path_series.index.to_series(), 3, "level", 5)
    x = x.loc[outcomes.actual.notna()]
    y = outcomes.loc[x.index, "actual"]
    observed = []
    original = module.Representation.fit

    def tracking_fit(self, frame):
        observed.append(frame.index.copy())
        return original(self, frame)

    monkeypatch.setattr(module.Representation, "fit", tracking_fit)
    module.choose_alpha(x, y, outcomes, True)
    expected = list(inner_folds(x.index, outcomes))
    assert len(observed) == len(expected)
    for actual, (train, validation) in zip(observed, expected):
        assert actual.equals(train)
        assert not len(actual.intersection(validation))


def test_alarm_selection_is_invariant_to_future_predictions():
    from research.study.events import alarm_series

    dates = pd.date_range("2005-01-31", periods=100, freq="ME")
    target = pd.Series(4.0, index=dates)
    target.iloc[[12, 48, 84]] = 9
    scores = pd.DataFrame(
        {
            "origin": dates,
            "label_available": dates + pd.offsets.MonthEnd(1),
            "eligible": target.lt(6).to_numpy(),
            "prediction": 0.2,
        }
    )
    first = alarm_series(scores, target, 6, 1)
    cutoff = pd.Timestamp("2010-01-31")
    changed = scores.copy()
    changed.loc[changed.origin > cutoff, "prediction"] = 0.99
    second = alarm_series(changed, target, 6, 1)
    assert_frame_equal(first.loc[first.origin <= cutoff], second.loc[second.origin <= cutoff])


def test_infinite_source_is_rejected(tmp_path):
    path = tmp_path / "infinite.csv"
    path.write_text("observation_date,X\n2020-01-01,inf\n")
    with pytest.raises(ValueError, match="Infinite"):
        read_series(path, "X")
