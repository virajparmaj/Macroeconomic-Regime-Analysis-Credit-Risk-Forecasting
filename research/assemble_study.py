"""Build the paper handoff exclusively from registered, completed run artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "research/study_results"
PAPER = ROOT / "research/paper_preparation"


def table(frame, columns):
    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += [
        "| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows()
    ]
    return "\n".join(lines)


def read_profile(registry, profile, name):
    return pd.read_csv(PUBLIC / registry[profile] / f"{name}.csv")


def assemble():
    registry = json.loads((PUBLIC / "latest_runs.json").read_text())
    for profile in ["core", "sensitivity", "external", "repeat"]:
        status = json.loads((PUBLIC / registry[profile] / "status.json").read_text())
        assert status["complete"], profile
    metrics = read_profile(registry, "core", "metrics")
    comparisons = read_profile(registry, "core", "comparisons")
    primary = comparisons.loc[
        (comparisons.model == "ridge")
        & (comparisons.horizon == 1)
        & (comparisons.contrast == "regime")
        & (comparisons.block_months == 12)
    ].iloc[0]
    baseline = metrics.loc[(metrics.model == "endpoint_persistence") & (metrics.horizon == 1)].iloc[
        0
    ]
    mean = metrics.loc[(metrics.model == "mean_persistence") & (metrics.horizon == 1)].iloc[0]
    external = read_profile(registry, "external", "metrics")
    sensitivity = read_profile(registry, "sensitivity", "comparisons")
    events = read_profile(registry, "core", "event_summary")
    assert (
        events.events.eq(3).all() and events.warned.eq(0).all()
    ), "Update interpretation for changed results"
    influence = read_profile(registry, "core", "event_influence")
    covid = influence.loc[influence.deleted_event == "2020-03-31"].iloc[0]
    validation = json.loads((PUBLIC / "validation.json").read_text())
    reproduction = json.loads((PUBLIC / "reproducibility.json").read_text())
    assert reproduction["passed"]
    selected = metrics.loc[(metrics.task == "level") & (metrics.horizon == 1)]
    classification = metrics.loc[
        (metrics.task != "level")
        & (metrics.horizon == 1)
        & ((metrics.feature_block == "S_last+M+R") | (metrics.model == "single_logistic"))
    ]
    selected_events = events.loc[
        ((events.feature_block == "S_last+M+R") | (events.model == "single_logistic"))
    ]
    sensitivity = sensitivity.loc[
        (sensitivity.block_months == 12) & (sensitivity.contrast == "regime")
    ]
    effect = f"{primary.mse_gain:.2%}"
    interval = f"[{primary.lower_95:.2%}, {primary.upper_95:.2%}]"
    text = f"""# Persistence, Aggregation, and Incremental Regime Information in U.S. High-Yield Spread Forecasting

**Executed empirical draft — 30 September 2026 (America/Chicago).** Historical results are exploratory. This is a research-note draft, not a claim of publication acceptance, causal identification or profitable trading.

## Abstract

We assess whether macroeconomic regimes improve forecasts of monthly U.S. high-yield option-adjusted spreads beyond spread persistence. A source audit removes two partial boundary months, leaving 307 complete months from January 1997 through July 2022. Expanding monthly forecasts begin in January 2009. The primary comparison adds a fold-fitted two-state macro representation to a ridge model already containing current spread, endpoint and macro information. Across {int(primary.n)} one-month origins, regime features reduce MSE by {effect} relative to that macro model, but a 12-month calendar-block bootstrap gives a 95% interval of {interval}. The regime model's RMSE is {primary.model_rmse:.4f} percentage points, compared with {baseline.rmse:.4f} for the latest-spread persistence benchmark. Thus the favorable incremental point estimate does not establish useful prediction beyond the strongest simple benchmark. Stress-entry warnings are evaluated separately from stress occupancy. Only three entries occur under the frozen threshold; the prespecified calibrated alarm policies miss all three in the main study. Robustness and a separately labeled later-period holdout accompany the historical results. Revised macro vintages, rare events and the inspected historical sample limit interpretation.

## 1. Question and contribution

Do macroeconomic regime features improve monthly high-yield spread forecasts when all full-information models share the latest spread observation, and does any improvement extend to new stress entries? The contribution is the controlled empirical comparison, not a new algorithm, regime concept or aggregation benchmark. Regimes are learned transformations of the macro block, so gains measure representation value within a model family rather than new external information.

Prior work already studies regime-conditioned corporate spreads, including [Erlwein-Sayer et al.](https://doi.org/10.1016/j.ecosta.2023.12.002), and credit-regime early-warning models, including [Li and Xiao](https://www.bankofcanada.ca/2016/04/staff-working-paper-2016-21/). [Ellwanger and Snudden](https://www.lcerpa.org/files/LCERPA_2021_5.pdf) explain why endpoint observations matter for forecasts of aggregated series. This study applies those established controls jointly to the present U.S. HY dataset. The repository literature review records additional work and access limitations; uniqueness is provisional, not established by an exhaustive search.

## 2. Data and the boundary correction

The original merged panel contains 309 stored monthly rows, but its daily spread source begins December 31, 1996 and ends August 1, 2022. Each boundary month contains a single observation, not a full monthly average. We exclude both, retaining 307 complete months. Previous 163-origin diagnostics included August 2022 as an outcome and remain exploratory audit records; their values are not substituted into these results.

The main outcome is the observed-day monthly mean. Compatibility analysis reconstructs the earlier forward-filled source-grid mean without filling absent calendar months. Initial thresholds use the 144 complete months through December 2008, separately for each target definition. The q75 threshold is approximately 7.257 percentage points; the three post-2008 entries are September 2011, January 2016 and March 2020.

The macro block uses CPI inflation, industrial-production growth, federal-funds and unemployment levels/changes, sentiment change, and correctly identified Euro-area OECD growth. The stored GDP label was not U.S. GDP. Assumed release lags are one month for CPI, production, rates and unemployment, three months for Euro-area growth and zero for sentiment. These are revised snapshots with assumed timing, not real-time vintages. A weekday-delay sensitivity changes daily and monthly spread availability conservatively.

## 3. Methods

One- and three-month direct forecasts use expanding monthly refits, eligible label-mature training rows, and identical origin sets within each comparison. Ridge tunes alpha in {{0.1, 1, 10, 100}} using three chronological 12-month validation blocks. Scaling, two-state K-means and interactions are fitted inside each relevant training fold. Random forests use 200 trees, minimum leaf size five, all predictors and seed 42. Logistic models use fixed L2 regularization, C=1 and no class weighting. The explicit protocol, decisions and lock specify all transforms and defaults.

Learned level models predict a residual relative to their available spread anchor. Four feature blocks distinguish monthly spread information, latest daily information, macro additions and regime additions. Ridge regime models include state indicators and macro interactions. Baselines include monthly persistence, endpoint persistence, training mean and direct AR(1).

The primary estimand is one minus the regime model's MSE divided by the otherwise identical no-regime model's MSE. We use 5,000 paired circular calendar-block draws, primary length 12 months and sensitivities six and 24. Intervals describe uncertainty in this realized forecast-loss sequence; they do not absorb uncertainty in the retrospectively selected research design. Centered block-bootstrap tests and the prespecified Holm-adjusted secondary family are reported separately from intervals. The 5% practical margin is a study convention, not a profitability threshold.

Classification distinguishes terminal stress occupancy from any entry within the horizon, restricting entry forecasts to origins below the fixed threshold. All-origin state and at-risk state panels make population differences explicit. Raw spread is only a ranking score. Probability metrics include Brier score, log loss and calibration bins. Alarm thresholds use past out-of-fold predictions and a maximum one false-alarm episode per at-risk year where feasible. Fewer than 24 calibration months or no observed calibration entry produces a documented no-alarm policy. Consecutive alarms form one episode, matched to at most one event.

## 4. Historical regression results

{table(selected, ['model','feature_block','n','rmse','mae','r2_vs_endpoint_persistence'])}

RMSE and MAE are in percentage points; multiply by 100 for basis points. Endpoint persistence reduces RMSE by {1-baseline.rmse/mean.rmse:.2%} relative to monthly-mean persistence on the same corrected target window. The primary regime gain is {effect}, 95% interval {interval}; the interval crosses zero. The regime ridge model still has negative out-of-sample R² against endpoint persistence. A positive incremental gain over a weaker macro model is therefore insufficient evidence of practically useful overall forecasting skill.

{table(comparisons.loc[comparisons.block_months == 12], ['horizon','model','contrast','n','mse_gain','lower_95','upper_95','p_value','holm_p'])}

The null-centered bootstrap p-values and percentile effect intervals are different constructions and are not asserted to be exact inversions. Secondary inference is exploratory in the larger context of the inspected historical sample. Sensitivity to rare high-loss episodes is reported in the event-influence table rather than hidden by averaging across folds.

Removing the scoring window from six months before through six months after the March 2020 entry changes the primary gain to {covid.mse_gain:.2%}, with interval [{covid.lower_95:.2%}, {covid.upper_95:.2%}]. The fitted forecasts are unchanged. This indicates that the favorable full-sample point estimate is highly episode-dependent, rather than a stable improvement across ordinary months.

## 5. State discrimination and stress entry

{table(classification, ['task','model','feature_block','risk_set','n','prevalence','auc','average_precision','brier','log_loss'])}

Different risk sets have different prevalence. Do not subtract their AUCs and interpret the result as a causal effect of persistence. At one month state and entry labels coincide on the same at-risk origins, but state and entry models can differ because their training populations differ.

{table(selected_events, ['horizon','model','feature_block','events','warned','missed','false_episodes_per_year','calibration_unavailable'])}

No main-study alarm policy warns any of the three entries under the prespecified calibration procedure. This is a finding about the chosen policy, threshold, data and calibration window, not proof that all early warning is impossible. The insufficient-calibration counts matter: the no-alarm fallback deliberately abstains when past evidence is inadequate. Three entries and overlapping windows cannot support a broad reliability claim.

## 6. Robustness and later-period validation

{table(sensitivity, ['scenario','horizon','model','n','mse_gain','lower_95','upper_95'])}

Sensitivity settings change one factor at a time. Seed results are robustness checks, not independent statistical replications. The delay scenario evaluates regression only because the current month's completed state is unavailable at its assumed month-end cutoff. The five-macro historical variant removes the discontinued Euro-area series before external evaluation.

FRED's current free spread export begins October 2023, leaving a gap after the archived data. The later-period check freezes model parameters at July 2022, waits until seven contiguous months support spread features, and evaluates from April 2024 where outcomes exist. It does not interpolate the gap, refit on later outcomes or masquerade as a continuous September 2022 extension. Data are frozen through August 2026, the last complete target month permitted by the acquisition design. [FRED series notes](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) document the three-year access window and redistribution terms.

{table(external, ['horizon','model','feature_block','n','rmse','mae','r2_vs_endpoint_persistence'])}

The holdout uses the same index but a short later period, not an independent crisis sample. Vintage API requests were rejected without a configured FRED credential. No matched real-time-vintage experiment was completed; shifted revised data are not a substitute. Full continuity and vintage validation remain limitations.

## 7. Reproducibility and limitations

The delivered validation has {validation['tests_passed']} passing tests, including legacy checks, T01–T15 scientific-contract scenarios and the corrected symmetric-winsorization regression test. The original logging checks returned booleans that pytest did not enforce; assertion wrappers exposed a real validation bug, now fixed. All six reader notebooks execute from saved artifacts. A separate core run reproduces {reproduction['rows']} ledger rows and predictions within the declared tolerance (maximum absolute difference {reproduction['max_prediction_difference']:.3g}).

Historical inputs and code/configuration/environment hashes identify each run. Atomic job checkpoints support resumption; mismatched manifests are rejected. Detailed licensed spread observations and prediction ledgers remain local. Public review artifacts contain aggregate metrics, manifests, tests and documentation. Software tests establish implementation properties, not forecast validity by themselves.

Limitations include a single index, revised macro data, assumed availability, rare entries, serial dependence, the historically informed design, model-family dependence, publication restrictions and the external gap. Regime labels are statistical representations, not identified economic causes. Spread prediction does not calibrate borrower default probabilities or establish investment returns.

## 8. Conclusion

The measured ridge regime improvement over the macro model is substantial as a point estimate but statistically uncertain, and it does not overcome endpoint persistence on the main historical evaluation. Regime representation should therefore not be presented as an established successful forecasting system. The state/entry split further shows why good stress-state ranking alone is insufficient for advance warning. The credible contribution is this quantified, reproducible assessment with explicit limits, rather than a general forecasting breakthrough.

## Artifact register

- Core: `{registry['core']}`
- One-at-a-time sensitivities: `{registry['sensitivity']}`
- Later-period holdout: `{registry['external']}`
- Independent core repetition: `{registry['repeat']}`
- Tables/figures: `research/study_results/<run_id>/`
- Local detailed ledgers: `results/research/<run_id>/`
- Protocol and environment: `research/study/`
"""
    (PAPER / "13_EXECUTED_MANUSCRIPT.md").write_text(text)
    result_register = f"""# Executed results register

Updated 30 September 2026. Numerical results are generated from versioned ledgers. See [the executed manuscript](13_EXECUTED_MANUSCRIPT.md) and [run registry](../study_results/latest_runs.json).

| ID | Result | Status / artifact |
|---|---|---|
| P01 | Source identity, corrected complete-month window, timing assumptions | Executed; protocol, input manifest, run quality metadata; actual vintage reconstruction unavailable |
| P02 | Matched monthly versus endpoint information | Executed; core metrics |
| P03 | Primary incremental regime comparison | Executed; gain {effect}, 95% interval {interval} |
| P04 | Macro ablation | Executed; core comparisons and adjusted secondary p-values |
| P05 | State and at-risk entry panels | Executed; metrics and calibration tables |
| P06 | Event warnings and false alarms | Executed; three events; event summary and local alarm/event ledgers |
| P07 | Block length, window, refit, threshold, target, timing, seed and event influence | Executed; sensitivity comparisons and core event influence |
| P08 | External validation | Executed as separate April 2024-onward holdout; continuous post-2022 extension unavailable |
| P09 | Matched vintage validation | Not completed; public API requests rejected without configured credential |

The earlier A01–A07 audit records remain in `research/evidence/`. They used a panel containing partial boundary months and are not substituted for the corrected study. Model failures, exclusions, sample sizes and calendar gaps remain visible in ledgers/metrics.
"""
    (PAPER / "08_RESULTS_REGISTER.md").write_text(result_register)
    (PAPER / "09_CONCLUSIONS_AND_LIMITATIONS.md").write_text(
        f"""# Executed conclusions and limitations

The core historical ridge regime comparison has a **{effect} MSE gain**, with a **95% interval of {interval}**, relative to the same macro model without regimes. The interval includes a loss, and the regime model's RMSE ({primary.model_rmse:.4f} pp) remains above endpoint persistence ({baseline.rmse:.4f} pp). This does not establish useful forecasting beyond the strongest simple benchmark.

All main-study alarm policies miss the three frozen-threshold entries. This is conditional on the specified calibration procedure; scarce calibration evidence causes documented no-alarm periods. It is not evidence that crisis prediction is universally impossible.

The historical matrix, robustness checks, later-period validation, independent repetition and notebook readers are implemented and executed. A continuous post-2022 extension and matched historical vintage validation remain unavailable. Revised macro inputs, assumed release timing, one index, few events, inspected historical outcomes and a short external window limit generalization.

See [the executed manuscript](13_EXECUTED_MANUSCRIPT.md) for complete results, sources and run IDs. The modest contribution is a controlled empirical evaluation of representation value and benchmark sensitivity, not a new method, proven trading system, or accepted publication.
"""
    )
    return PAPER / "13_EXECUTED_MANUSCRIPT.md"


if __name__ == "__main__":
    print(assemble())
