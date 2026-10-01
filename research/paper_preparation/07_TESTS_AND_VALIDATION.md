# Tests and scientific validation

## Executed status — 1 October 2026

**58 tests passed in 25.36 seconds** under Python 3.13.15. See [captured output](../study_results/tests.txt) and [machine-readable validation](../study_results/validation.json). The new behavioral suite is `tests/study/test_validity.py`; legacy tests remain included. Old logging checks now assert their outcomes, exposing and fixing a genuine symmetric-winsorization validation bug. No assertions were weakened.

All 72 core, 236 sensitivity, 16 external and 72 repeated-core jobs completed with zero model-failure rows. Independent verification recomputed 204 aggregate regression MSE values across those runs, checked unique keys, finite values, probability bounds, training-label maturity and entry risk sets. The core has 162 one-month and 160 three-month origins. The repeated run's 14,040 keys/actuals/predictions match exactly (maximum absolute difference 0; allowed rtol 1e-8, atol 1e-10).

All six study readers execute. The main event analysis contains three entries; no main policy warns any. The suite checks future-data perturbation invariance, inner preprocessing isolation, alarm calibration isolation, manifest mismatch rejection, checkpoint resumption and completed-run immutability. Availability tests validate the implemented assumptions; they cannot validate missing historical vintage data.

The earlier [validation report](validation_report.txt) is retained as an audit-period record, not the current release gate. Passing tests establishes software behavior, not forecasting skill.

## Required tests with meaningful failure cases

| ID | Test | Adversarial example / acceptance condition |
|---|---|---|
| T01 | Label maturity | At cutoff T, reject a training row whose h-step label becomes available after T even if its feature date precedes T |
| T02 | Unknown future | Last h rows stay missing; entry labels require every future month; an absent intermediate month is not a negative |
| T03 | Future perturbation | Replace all post-cutoff inputs/targets with extreme values; every earlier feature, threshold, regime and prediction stays unchanged |
| T04 | Frozen quantile | Add extreme test-period spreads; pre-2009 threshold is unchanged |
| T05 | Train-only fitting | Spy on scaler, cluster and hyperparameter fit inputs; inner/outer validation rows never enter fitting |
| T06 | State/entry distinction | A path crosses then returns below q before h; endpoint state is 0, entry is 1; at h=1 both coincide on at-risk origins |
| T07 | Persistent stress | An origin already at/above q is excluded from entry risk set even if the next month is stressed |
| T08 | Availability | A revised/released value dated after cutoff is unavailable even if its reference month is earlier |
| T09 | Target identity | Legacy mean reconciles to source-grid forward fill; observed-day mean reconciles separately; units convert 1 pp to 100 bps |
| T10 | Pairing and uniqueness | A missing forecast or duplicate key makes the main comparison fail visibly; different targets cannot be paired |
| T11 | Baseline sanity | Constant observed spreads yield exact persistence forecasts; ledger and direct metric calculation agree |
| T12 | Probability edge cases | Single-class training/evaluation gives documented fallback/NA, never fabricated AUC; log loss handles bounded probabilities consistently |
| T13 | Calendar blocks | Bootstrap indices preserve model pairing and contiguous calendar blocks before at-risk masking |
| T14 | Event accounting | Consecutive alarms are one episode, one entry is not counted as several successes, at-risk time denominator is explicit |
| T15 | Reproduction | Same manifest/seed gives identical keys and numerically matching predictions; changed data yields a changed manifest |

The table is the original acceptance contract. Its behavioral scenarios are implemented in `tests/study/test_validity.py`; T15 also uses `research/verify_study.py` on the completed core runs. T08 verifies as-of availability logic and the stated lag assumption, not unobserved historical releases. Synthetic tests complement independent inspection of real ledgers.

## Release gate

Require all applicable tests to pass, inspect extreme-error months and all event matches, verify common-origin counts, and regenerate tables from saved predictions. Report the measured runtime, software environment and failed/undefined outputs. A statistical finding passes its evidence gate only after the paired comparison and uncertainty are produced; green pytest output is a separate engineering condition.
