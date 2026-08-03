"""Tests for :mod:`src.features`.

The three tests named in the v2 specification are, in order:

* :func:`test_rolling_features_exclude_current_row`
* :func:`test_no_feature_reconstructs_target`
* :func:`test_v1_identity_holds`

The third is a regression test against the *frozen* ``utilities`` package. It asserts that
the old code still leaks. That sounds backwards, but it is the guard that stops the fix
from being quietly undone: if someone ever "tidies" ``utilities.functions`` into shifting
its rolling windows, this test fails and forces a conversation about the frozen record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config_v2 import TARGET_COLUMN
from src.features import (
    add_growth_rates,
    add_interaction_terms,
    add_lag_features,
    add_rolling_features,
    build_feature_frame,
    leakage_report,
    make_forecast_target,
    rolling_identity_error,
)

# ---------------------------------------------------------------------------
# Named test 1: rolling statistics must not see the current row
# ---------------------------------------------------------------------------


def test_rolling_features_exclude_current_row(known_series: pd.DataFrame) -> None:
    """A shifted 3-month mean at row t must be independent of x(t).

    With ``x = 1, 2, 3, ...`` the trailing mean over rows t-1, t-2, t-3 at zero-based
    position ``i`` equals ``i - 1``. The v1 version, which includes row t, equals ``i``.
    """
    out = add_rolling_features(known_series, ["x"], window=3, shift=1)
    col = "x_rollmean3_lag1"

    assert col in out.columns, "shift must be visible in the column name"

    # Positions 0..2 cannot have a complete shifted window.
    assert out[col].iloc[:3].isna().all()

    # Position 3 averages x at positions 0,1,2 -> (1+2+3)/3 = 2.0
    assert out[col].iloc[3] == pytest.approx(2.0)
    # Position 10 averages positions 7,8,9 -> (8+9+10)/3 = 9.0
    assert out[col].iloc[10] == pytest.approx(9.0)

    # The decisive property: perturbing x(t) must not move rollmean(t).
    perturbed = known_series.copy()
    perturbed.iloc[10, perturbed.columns.get_loc("x")] = 1_000_000.0
    out_perturbed = add_rolling_features(perturbed, ["x"], window=3, shift=1)

    assert out_perturbed[col].iloc[10] == pytest.approx(
        out[col].iloc[10]
    ), "rolling mean at row t changed when x(t) changed, so it still includes row t"


def test_rolling_features_reject_zero_shift(known_series: pd.DataFrame) -> None:
    """A shift of zero reproduces the v1 leak and must be refused outright."""
    with pytest.raises(ValueError, match="reintroduces the v1 leak"):
        add_rolling_features(known_series, ["x"], window=3, shift=0)


def test_functions_do_not_mutate_input(known_series: pd.DataFrame) -> None:
    """Every feature helper returns a new frame rather than editing in place."""
    before = known_series.copy(deep=True)

    add_lag_features(known_series, ["x"], lags=(1, 2))
    add_rolling_features(known_series, ["x"], window=3, shift=1)
    add_growth_rates(known_series, ["x"])
    make_forecast_target(known_series, "x", horizon=1)

    pd.testing.assert_frame_equal(known_series, before)


# ---------------------------------------------------------------------------
# Named test 2: nothing in the v2 feature set restates the target
# ---------------------------------------------------------------------------


def test_no_feature_reconstructs_target(real_panel: pd.DataFrame) -> None:
    """The v2 feature set must fail the v1 identity and produce an empty leakage report."""
    frame, target_name, feature_cols = build_feature_frame(
        real_panel,
        target_col=TARGET_COLUMN,
        macro_columns=[c for c in real_panel.columns if c != TARGET_COLUMN],
        lags=(1, 2, 3),
        window=3,
        horizon=1,
    )

    # The v1 identity is 3*rollmean3 - lag1 - lag2 == target. Against the v2 shifted
    # rolling column that reconstruction should be wrong by an economically large margin.
    error = rolling_identity_error(
        frame,
        rollmean_col=f"{TARGET_COLUMN}_rollmean3_lag1",
        lag1_col=f"{TARGET_COLUMN}_lag1",
        lag2_col=f"{TARGET_COLUMN}_lag2",
        target_col=TARGET_COLUMN,
        window=3,
    )
    assert error > 1e-6, (
        f"v2 features still reconstruct the target to within {error:.2e}; "
        "the rolling window is not actually shifted"
    )

    # And the same must hold against the forward target the models are fitted to.
    forward_error = rolling_identity_error(
        frame,
        rollmean_col=f"{TARGET_COLUMN}_rollmean3_lag1",
        lag1_col=f"{TARGET_COLUMN}_lag1",
        lag2_col=f"{TARGET_COLUMN}_lag2",
        target_col=target_name,
        window=3,
    )
    assert forward_error > 1e-6

    report = leakage_report(frame[feature_cols + [target_name]].dropna(), target=target_name)
    assert report.empty, f"leakage_report flagged columns:\n{report}"


def test_leakage_report_catches_the_v1_frame(real_panel: pd.DataFrame) -> None:
    """The detector must fire on a v1-style frame, or its silence on v2 means nothing."""
    v1_like = real_panel.copy()
    v1_like["Credit_Spread_rollmean3"] = v1_like[TARGET_COLUMN].rolling(3).mean()
    v1_like["Credit_Spread_lag1"] = v1_like[TARGET_COLUMN].shift(1)
    v1_like["Credit_Spread_lag2"] = v1_like[TARGET_COLUMN].shift(2)
    v1_like["Credit_Spread_x_FEDFUNDS"] = v1_like[TARGET_COLUMN] * v1_like["FEDFUNDS"]

    report = leakage_report(v1_like.dropna(), target=TARGET_COLUMN)

    assert not report.empty
    assert "Credit_Spread_rollmean3" in report.index
    assert "Credit_Spread_x_FEDFUNDS" in report.index


def test_interaction_terms_refuse_the_raw_target(real_panel: pd.DataFrame) -> None:
    """Building ``Credit_Spread_x_FEDFUNDS`` from the unlagged target must raise."""
    with pytest.raises(ValueError, match="Credit_Spread_x_FEDFUNDS defect"):
        add_interaction_terms(
            real_panel,
            [(TARGET_COLUMN, "FEDFUNDS")],
            target_stem=TARGET_COLUMN,
        )


def test_interaction_terms_allow_lagged_inputs(real_panel: pd.DataFrame) -> None:
    """The same interaction is fine once the target side is explicitly lagged."""
    lagged = add_lag_features(real_panel, [TARGET_COLUMN], lags=(1,))
    out = add_interaction_terms(
        lagged,
        [(f"{TARGET_COLUMN}_lag1", "FEDFUNDS")],
        target_stem=TARGET_COLUMN,
    )
    assert f"{TARGET_COLUMN}_lag1_x_FEDFUNDS" in out.columns


def test_make_forecast_target_returns_its_name(real_panel: pd.DataFrame) -> None:
    """The created target name is returned, so a caller cannot guard on the wrong string.

    This is the v1 defect where the notebook filtered on ``"Target"`` while the column was
    called ``Target_1step_ahead``.
    """
    out, name = make_forecast_target(real_panel, TARGET_COLUMN, horizon=1)
    assert name == f"y_{TARGET_COLUMN}_h1"
    assert name in out.columns
    # y(t) must equal the level at t+1.
    assert out[name].iloc[0] == pytest.approx(out[TARGET_COLUMN].iloc[1])


def test_growth_rates_do_not_emit_infinities() -> None:
    """A zero denominator yields NaN, not the infinity v1 replaced with 0.0."""
    frame = pd.DataFrame(
        {"g": [1.0, 0.0, 2.0, 4.0]},
        index=pd.date_range("2000-01-31", periods=4, freq="ME"),
    )
    out = add_growth_rates(frame, ["g"])
    values = out["g_growth1"]
    assert not np.isinf(values.to_numpy(dtype=float)).any()
    assert pd.isna(values.iloc[2]), "division by a zero previous value must be NaN"


# ---------------------------------------------------------------------------
# Named test 3: the v1 behaviour must keep failing, so the fix cannot be reverted
# ---------------------------------------------------------------------------


def test_v1_identity_holds(real_panel: pd.DataFrame) -> None:
    """The frozen ``utilities`` feature code must still reconstruct the target exactly.

    ``Credit_Spread(t) == 3*rollmean3(t) - lag1(t) - lag2(t)`` holds to machine precision
    whenever the rolling window includes row t. This test pins that property on the v1
    code so that any future change to ``utilities/functions.py`` that silently repairs the
    leak is caught, and so that notebook 00's demonstration cannot go stale.
    """
    from utilities.functions import add_lag_features as v1_add_lags
    from utilities.functions import add_rolling_features as v1_add_rolling

    v1_frame = v1_add_lags(real_panel.copy(), ["Credit_Spread"], lags=[1, 2])
    v1_frame = v1_add_rolling(v1_frame, ["Credit_Spread"], window=3)

    error = rolling_identity_error(
        v1_frame,
        rollmean_col="Credit_Spread_rollmean3",
        lag1_col="Credit_Spread_lag1",
        lag2_col="Credit_Spread_lag2",
        target_col="Credit_Spread",
        window=3,
    )

    assert error < 1e-12, (
        f"v1 rolling features no longer reconstruct the target (max error {error:.3e}). "
        "utilities/ is supposed to be frozen; if it changed, notebook 00's audit of the "
        "original result is no longer a faithful reproduction."
    )


def test_v1_and_v2_rolling_differ_on_the_same_input(known_series: pd.DataFrame) -> None:
    """The two implementations must disagree exactly one row apart.

    Notebook 00 shows this diff side by side; this pins the property it relies on.
    """
    from utilities.functions import add_rolling_features as v1_add_rolling

    v1_out = v1_add_rolling(known_series.copy(), ["x"], window=3)
    v2_out = add_rolling_features(known_series, ["x"], window=3, shift=1)

    v1_col = v1_out["x_rollmean3"]
    v2_col = v2_out["x_rollmean3_lag1"]

    # v2 at row t equals v1 at row t-1, exactly.
    pd.testing.assert_series_equal(
        v2_col.iloc[1:].reset_index(drop=True),
        v1_col.iloc[:-1].reset_index(drop=True),
        check_names=False,
    )
