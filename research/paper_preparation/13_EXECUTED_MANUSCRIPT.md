# Persistence, Aggregation, and Incremental Regime Information in U.S. High-Yield Spread Forecasting

**Executed empirical draft — 1 October 2026 (America/Chicago); protocol frozen 30 September.** Historical results are exploratory. This is a research-note draft, not a claim of publication acceptance, causal identification or profitable trading.

## Abstract

We assess whether macroeconomic regimes improve forecasts of monthly U.S. high-yield option-adjusted spreads beyond spread persistence. A source audit removes two partial boundary months, leaving 307 complete months from January 1997 through July 2022. Expanding monthly forecasts begin in January 2009. The primary comparison adds a fold-fitted two-state macro representation to a ridge model already containing current spread, endpoint and macro information. Across 162 one-month origins, regime features reduce MSE by 39.36% relative to that macro model, but a 12-month calendar-block bootstrap gives a 95% interval of [-7.92%, 55.79%]. The regime model's RMSE is 0.4854 percentage points, compared with 0.4234 for the latest-spread persistence benchmark. Thus the favorable incremental point estimate does not establish useful prediction beyond the strongest simple benchmark. Stress-entry warnings are evaluated separately from stress occupancy. Only three entries occur under the frozen threshold; the prespecified calibrated alarm policies miss all three in the main study. Robustness and a separately labeled later-period holdout accompany the historical results. Revised macro vintages, rare events and the inspected historical sample limit interpretation.

## 1. Question and contribution

Do macroeconomic regime features improve monthly high-yield spread forecasts when all full-information models share the latest spread observation, and does any improvement extend to new stress entries? The contribution is the controlled empirical comparison, not a new algorithm, regime concept or aggregation benchmark. Regimes are learned transformations of the macro block, so gains measure representation value within a model family rather than new external information.

Prior work already studies regime-conditioned corporate spreads, including [Erlwein-Sayer et al.](https://doi.org/10.1016/j.ecosta.2023.12.002), and credit-regime early-warning models, including [Li and Xiao](https://www.bankofcanada.ca/2016/04/staff-working-paper-2016-21/). [Ellwanger and Snudden](https://www.lcerpa.org/files/LCERPA_2021_5.pdf) explain why endpoint observations matter for forecasts of aggregated series. This study applies those established controls jointly to the present U.S. HY dataset. The repository literature review records additional work and access limitations; uniqueness is provisional, not established by an exhaustive search.

## 2. Data and the boundary correction

The original merged panel contains 309 stored monthly rows, but its daily spread source begins December 31, 1996 and ends August 1, 2022. Each boundary month contains a single observation, not a full monthly average. We exclude both, retaining 307 complete months. Previous 163-origin diagnostics included August 2022 as an outcome and remain exploratory audit records; their values are not substituted into these results.

The main outcome is the observed-day monthly mean. Compatibility analysis reconstructs the earlier forward-filled source-grid mean without filling absent calendar months. Initial thresholds use the 144 complete months through December 2008, separately for each target definition. The q75 threshold is approximately 7.257 percentage points; the three post-2008 entries are September 2011, January 2016 and March 2020.

The macro block uses CPI inflation, industrial-production growth, federal-funds and unemployment levels/changes, sentiment change, and correctly identified Euro-area OECD growth. The stored GDP label was not U.S. GDP. Assumed release lags are one month for CPI, production, rates and unemployment, three months for Euro-area growth and zero for sentiment. These are revised snapshots with assumed timing, not real-time vintages. A weekday-delay sensitivity changes daily and monthly spread availability conservatively.

## 3. Methods

One- and three-month direct forecasts use expanding monthly refits, eligible label-mature training rows, and identical origin sets within each comparison. Ridge tunes alpha in {0.1, 1, 10, 100} using three chronological 12-month validation blocks. Scaling, two-state K-means and interactions are fitted inside each relevant training fold. Random forests use 200 trees, minimum leaf size five, all predictors and seed 42. Logistic models use fixed L2 regularization, C=1 and no class weighting. The explicit protocol, decisions and lock specify all transforms and defaults.

Learned level models predict a residual relative to their available spread anchor. Four feature blocks distinguish monthly spread information, latest daily information, macro additions and regime additions. Ridge regime models include state indicators and macro interactions. Baselines include monthly persistence, endpoint persistence, training mean and direct AR(1).

The primary estimand is one minus the regime model's MSE divided by the otherwise identical no-regime model's MSE. We use 5,000 paired circular calendar-block draws, primary length 12 months and sensitivities six and 24. Intervals describe uncertainty in this realized forecast-loss sequence; they do not absorb uncertainty in the retrospectively selected research design. Centered block-bootstrap tests and the prespecified Holm-adjusted secondary family are reported separately from intervals. The 5% practical margin is a study convention, not a profitability threshold.

Classification distinguishes terminal stress occupancy from any entry within the horizon, restricting entry forecasts to origins below the fixed threshold. All-origin state and at-risk state panels make population differences explicit. Raw spread is only a ranking score. Probability metrics include Brier score, log loss and calibration bins. Alarm thresholds use past out-of-fold predictions and a maximum one false-alarm episode per at-risk year where feasible. Fewer than 24 calibration months or no observed calibration entry produces a documented no-alarm policy. Consecutive alarms form one episode, matched to at most one event.

## 4. Historical regression results

| model | feature_block | n | rmse | mae | r2_vs_endpoint_persistence |
| --- | --- | --- | --- | --- | --- |
| ar1 | S_last | 162 | 0.6134 | 0.3845 | -1.0985 |
| endpoint_persistence | S_last | 162 | 0.4234 | 0.2825 | 0.0000 |
| mean_persistence | S_last | 162 | 0.6129 | 0.3769 | -1.0949 |
| rf | S_last | 162 | 0.4307 | 0.2914 | -0.0344 |
| rf | S_last+M | 162 | 0.4352 | 0.2945 | -0.0562 |
| rf | S_last+M+R | 162 | 0.4349 | 0.2943 | -0.0547 |
| rf | S_mean | 162 | 0.6172 | 0.3717 | -1.1246 |
| ridge | S_last | 162 | 0.4257 | 0.2949 | -0.0109 |
| ridge | S_last+M | 162 | 0.6234 | 0.3387 | -1.1674 |
| ridge | S_last+M+R | 162 | 0.4854 | 0.3189 | -0.3143 |
| ridge | S_mean | 162 | 0.6438 | 0.3636 | -1.3116 |
| training_mean | S_last | 162 | 2.2584 | 1.6308 | -27.4464 |

RMSE and MAE are in percentage points; multiply by 100 for basis points. Endpoint persistence reduces RMSE by 30.91% relative to monthly-mean persistence on the same corrected target window. The primary regime gain is 39.36%, 95% interval [-7.92%, 55.79%]; the interval crosses zero. The regime ridge model still has negative out-of-sample R² against endpoint persistence. A positive incremental gain over a weaker macro model is therefore insufficient evidence of practically useful overall forecasting skill.

| horizon | model | contrast | n | mse_gain | lower_95 | upper_95 | p_value | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | rf | regime | 162 | 0.0014 | -0.0017 | 0.0043 | 0.3529 | 1.0000 |
| 1 | rf | macro | 162 | -0.0211 | -0.1131 | 0.0813 | 0.6937 | NA |
| 1 | ridge | regime | 162 | 0.3936 | -0.0792 | 0.5579 | 0.4507 | NA |
| 1 | ridge | macro | 162 | -1.1441 | -2.9546 | 0.0386 | 0.3331 | 1.0000 |
| 3 | rf | regime | 160 | 0.0015 | -0.0007 | 0.0059 | 0.2236 | 1.0000 |
| 3 | rf | macro | 160 | -0.0671 | -0.9697 | 0.2831 | 0.8154 | NA |
| 3 | ridge | regime | 160 | 0.3634 | -0.1160 | 0.6196 | 0.4013 | 1.0000 |
| 3 | ridge | macro | 160 | -0.6926 | -2.0663 | 0.0481 | 0.2182 | 1.0000 |

The null-centered bootstrap p-values and percentile effect intervals are different constructions and are not asserted to be exact inversions. Secondary inference is exploratory in the larger context of the inspected historical sample. Sensitivity to rare high-loss episodes is reported in the event-influence table rather than hidden by averaging across folds.

Removing the scoring window from six months before through six months after the March 2020 entry changes the primary gain to -2.37%, with interval [-11.04%, 6.53%]. The fitted forecasts are unchanged. This indicates that the favorable full-sample point estimate is highly episode-dependent, rather than a stable improvement across ordinary months.

## 5. State discrimination and stress entry

| task | model | feature_block | risk_set | n | prevalence | auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entry | logistic | S_last+M+R | at_risk | 142 | 0.0211 | 0.8993 | 0.1583 | 0.0245 | 0.0988 |
| entry | rf | S_last+M+R | at_risk | 142 | 0.0211 | 0.7506 | 0.1570 | 0.0221 | 0.1438 |
| entry | single_logistic | S_last | at_risk | 142 | 0.0211 | 0.8441 | 0.2664 | 0.0187 | 0.0846 |
| state | logistic | S_last+M+R | all | 162 | 0.1173 | 0.9591 | 0.7333 | 0.0452 | 0.2307 |
| state | logistic | S_last+M+R | at_risk | 142 | 0.0211 | 0.8561 | 0.1244 | 0.0211 | 0.0915 |
| state | rf | S_last+M+R | all | 162 | 0.1173 | 0.9520 | 0.8954 | 0.0331 | 0.1768 |
| state | rf | S_last+M+R | at_risk | 142 | 0.0211 | 0.7446 | 0.4237 | 0.0154 | 0.1303 |
| state | single_logistic | S_last | all | 162 | 0.1173 | 0.9691 | 0.9054 | 0.0337 | 0.1296 |
| state | single_logistic | S_last | at_risk | 142 | 0.0211 | 0.8465 | 0.4331 | 0.0233 | 0.1014 |

Different risk sets have different prevalence. Do not subtract their AUCs and interpret the result as a causal effect of persistence. At one month state and entry labels coincide on the same at-risk origins, but state and entry models can differ because their training populations differ.

| horizon | model | feature_block | events | warned | missed | false_episodes_per_year | calibration_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | logistic | S_last+M+R | 3 | 0 | 3 | 0.4225 | 54 |
| 1 | rf | S_last+M+R | 3 | 0 | 3 | 0.2535 | 54 |
| 1 | single_logistic | S_last | 3 | 0 | 3 | 0.2535 | 54 |
| 3 | logistic | S_last+M+R | 3 | 0 | 3 | 0.5143 | 56 |
| 3 | rf | S_last+M+R | 3 | 0 | 3 | 0.2571 | 56 |
| 3 | single_logistic | S_last | 3 | 0 | 3 | 0.1714 | 56 |

No main-study alarm policy warns any of the three entries under the prespecified calibration procedure. This is a finding about the chosen policy, threshold, data and calibration window, not proof that all early warning is impossible. The insufficient-calibration counts matter: the no-alarm fallback deliberately abstains when past evidence is inadequate. Three entries and overlapping windows cannot support a broad reliability claim.

## 6. Robustness and later-period validation

| scenario | horizon | model | n | mse_gain | lower_95 | upper_95 |
| --- | --- | --- | --- | --- | --- | --- |
| annual | 1 | rf | 162 | 0.0023 | 0.0001 | 0.0043 |
| annual | 1 | ridge | 162 | 0.2159 | -0.0063 | 0.2942 |
| annual | 3 | rf | 160 | 0.0012 | -0.0034 | 0.0095 |
| annual | 3 | ridge | 160 | 0.3382 | -0.0849 | 0.5864 |
| delay | 1 | rf | 162 | 0.0013 | -0.0003 | 0.0036 |
| delay | 1 | ridge | 162 | 0.1837 | -0.1771 | 0.2963 |
| delay | 3 | rf | 160 | 0.0008 | -0.0012 | 0.0028 |
| delay | 3 | ridge | 160 | 0.3984 | -0.1185 | 0.6287 |
| five_macro | 1 | rf | 162 | 0.0001 | -0.0012 | 0.0015 |
| five_macro | 1 | ridge | 162 | 0.1263 | -0.1533 | 0.2040 |
| five_macro | 3 | rf | 160 | 0.0001 | -0.0028 | 0.0045 |
| five_macro | 3 | ridge | 160 | 0.2854 | -0.1401 | 0.5433 |
| legacy | 1 | rf | 162 | 0.0029 | 0.0008 | 0.0042 |
| legacy | 1 | ridge | 162 | 0.4011 | -0.0412 | 0.5536 |
| legacy | 3 | rf | 160 | 0.0005 | -0.0009 | 0.0026 |
| legacy | 3 | ridge | 160 | 0.3639 | -0.1160 | 0.6253 |
| rolling120 | 1 | rf | 162 | -0.0013 | -0.0038 | 0.0020 |
| rolling120 | 1 | ridge | 162 | 0.0350 | -0.0390 | 0.0832 |
| rolling120 | 3 | rf | 160 | -0.0054 | -0.0220 | 0.0021 |
| rolling120 | 3 | ridge | 160 | -0.0879 | -0.3431 | 0.0385 |
| seed0 | 1 | rf | 162 | 0.0016 | -0.0012 | 0.0036 |
| seed0 | 3 | rf | 160 | 0.0000 | -0.0034 | 0.0023 |
| seed1 | 1 | rf | 162 | 0.0000 | -0.0014 | 0.0023 |
| seed1 | 3 | rf | 160 | 0.0002 | -0.0027 | 0.0022 |
| seed2 | 1 | rf | 162 | 0.0006 | -0.0011 | 0.0017 |
| seed2 | 3 | rf | 160 | 0.0040 | 0.0003 | 0.0102 |
| seed3 | 1 | rf | 162 | 0.0015 | -0.0002 | 0.0028 |
| seed3 | 3 | rf | 160 | -0.0010 | -0.0035 | 0.0019 |
| seed4 | 1 | rf | 162 | -0.0007 | -0.0022 | 0.0016 |
| seed4 | 3 | rf | 160 | 0.0014 | -0.0004 | 0.0029 |

Sensitivity settings change one factor at a time. Seed results are robustness checks, not independent statistical replications. The delay scenario evaluates regression only because the current month's completed state is unavailable at its assumed month-end cutoff. The five-macro historical variant removes the discontinued Euro-area series before external evaluation.

FRED's current free spread export begins October 2023, leaving a gap after the archived data. The later-period check freezes model parameters at July 2022, waits until seven contiguous months support spread features, and evaluates from April 2024 where outcomes exist. It does not interpolate the gap, refit on later outcomes or masquerade as a continuous September 2022 extension. Data are frozen through August 2026, the last complete target month permitted by the acquisition design. Missing inflation/unemployment features exclude November and December 2025. Consequently, the one-month panel has 26 origins through July 2026 and the three-month panel has 24 through May 2026; all models and baselines share these origins. [FRED series notes](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) document the three-year access window and redistribution terms.

| horizon | model | feature_block | n | rmse | mae | r2_vs_endpoint_persistence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ar1 | S_last | 26 | 0.2965 | 0.2293 | -1.2381 |
| 1 | endpoint_persistence | S_last | 26 | 0.1982 | 0.1357 | 0.0000 |
| 1 | mean_persistence | S_last | 26 | 0.2794 | 0.1951 | -0.9874 |
| 1 | rf | S_last+M | 26 | 0.1877 | 0.1351 | 0.1035 |
| 1 | rf | S_last+M+R | 26 | 0.1875 | 0.1351 | 0.1057 |
| 1 | ridge | S_last+M | 26 | 0.1902 | 0.1336 | 0.0794 |
| 1 | ridge | S_last+M+R | 26 | 0.1956 | 0.1399 | 0.0259 |
| 1 | training_mean | S_last | 26 | 2.4995 | 2.4821 | -157.9938 |
| 3 | ar1 | S_last | 24 | 0.6668 | 0.5940 | -1.3084 |
| 3 | endpoint_persistence | S_last | 24 | 0.4389 | 0.3048 | 0.0000 |
| 3 | mean_persistence | S_last | 24 | 0.4653 | 0.3291 | -0.1240 |
| 3 | rf | S_last+M | 24 | 0.7288 | 0.6753 | -1.7577 |
| 3 | rf | S_last+M+R | 24 | 0.7259 | 0.6730 | -1.7361 |
| 3 | ridge | S_last+M | 24 | 0.6328 | 0.5641 | -1.0794 |
| 3 | ridge | S_last+M+R | 24 | 0.6668 | 0.5958 | -1.3084 |
| 3 | training_mean | S_last | 24 | 2.5523 | 2.5342 | -32.8220 |

At one month the external learned models modestly outperform endpoint persistence, but adding regimes worsens ridge's error relative to the same macro model. At three months all learned models lose to endpoint persistence. These small samples do not establish stable gains. The holdout uses the same index but a short later period, not an independent crisis sample. Vintage API requests were rejected without a configured FRED credential. No matched real-time-vintage experiment was completed; shifted revised data are not a substitute. Full continuity and vintage validation remain limitations.

## 7. Reproducibility and limitations

The delivered validation has 58 passing tests, including legacy checks, T01–T15 scientific-contract scenarios and the corrected symmetric-winsorization regression test. The original logging checks returned booleans that pytest did not enforce; assertion wrappers exposed a real validation bug, now fixed. All six reader notebooks execute from saved artifacts. A separate core run reproduces 14040 ledger rows and predictions within the declared tolerance (maximum absolute difference 0).

Historical inputs and code/configuration/environment hashes identify each run. The repeat uses the same protocol and locked environment after equivalent orchestration/provenance refactoring; its distinct code hash and exact numerical agreement are disclosed in the reproduction record. Atomic job checkpoints support resumption; mismatched manifests are rejected and completed runs are unchanged. Detailed licensed spread observations and prediction ledgers remain local. Public review artifacts contain aggregate metrics, manifests, tests and documentation. Software tests establish implementation properties, not forecast validity by themselves.

Limitations include a single index, revised macro data, assumed availability, rare entries, serial dependence, the historically informed design, model-family dependence, publication restrictions and the external gap. Regime labels are statistical representations, not identified economic causes. Spread prediction does not calibrate borrower default probabilities or establish investment returns.

## 8. Conclusion

The measured ridge regime improvement over the macro model is substantial as a point estimate but statistically uncertain, and it does not overcome endpoint persistence on the main historical evaluation. Regime representation should therefore not be presented as an established successful forecasting system. The state/entry split further shows why good stress-state ranking alone is insufficient for advance warning. The credible contribution is this quantified, reproducible assessment with explicit limits, rather than a general forecasting breakthrough.

## Artifact register

- Core: `20261001T021109Z-947a55`
- One-at-a-time sensitivities: `20261001T022614Z-d43c7b`
- Later-period holdout: `20261001T024151Z-f6c6a9`
- Independent core repetition: `20261001T023825Z-9610b2`
- Tables/figures: `research/study_results/<run_id>/`
- Local detailed ledgers: `results/research/<run_id>/`
- Protocol and environment: `research/study/`
