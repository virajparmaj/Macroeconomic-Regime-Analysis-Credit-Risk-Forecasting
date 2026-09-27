# Tests and scientific validation

## Current status

The repository's existing pytest suite was rerun while preparing this folder. The captured outcome is in [validation_report.txt](validation_report.txt). These tests cover existing feature/split behavior; passing them does not certify target availability, vintage correctness, external validity or forecasting skill. Some split-test descriptions justify an embargo by feature overlap, which is not the appropriate general rationale for this forecasting design.

The earlier [audit summary](../evidence/audit_summary.json) records data identities, reconstruction error and notebook completeness. Classification and aggregation diagnostics are executed exploratory checks. The new paper pipeline and tests below are **planned**.

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

Add these as behavioral tests around the new modules, not assertions that mirror implementation lines. Retain legacy tests for old behavior while clearly naming the new scientific contract. Tests on synthetic paths complement, rather than replace, inspection of real forecast ledgers.

## Release gate

Require all applicable tests to pass, inspect extreme-error months and all event matches, verify common-origin counts, and regenerate tables from saved predictions. Report the measured runtime, software environment and failed/undefined outputs. A statistical finding passes its evidence gate only after the paired comparison and uncertainty are produced; green pytest output is a separate engineering condition.
