# Verified exploratory findings

These are executed diagnostics from the existing analysis, not results of the planned matched regime study. The source artifacts are linked for each table. No new forecasting experiment was run while assembling this paper-preparation package.

## 1. Information in the benchmark

Forecast origins January 2009–July 2022, 163 observations; target is the following month. Units are percentage points (pp); 1 pp = 100 basis points.

| Target | Predictor | RMSE (pp) | MAE (pp) | Test-mean R² |
|---|---|---:|---:|---:|
| Observed-day mean | Previous monthly mean | 0.612246 | 0.377659 | 0.924450 |
| Observed-day mean | Latest daily spread | 0.422144 | 0.280968 | 0.964082 |
| Legacy forward-filled mean | Previous monthly mean | 0.612416 | 0.377703 | 0.924430 |
| Legacy forward-filled mean | Latest daily spread | 0.421243 | 0.280453 | 0.964246 |

The latest daily value lowers legacy-target RMSE by **31.22%**. This is an RMSE reduction, not an MSE reduction. Hold the target and dates fixed when calculating it. [Source table](../evidence/aggregation_benchmarks.csv).

This motivates the stronger benchmark but does not establish a novel principle, statistical significance, or live availability. The final study must add endpoint information to the learned models too.

## 2. Reproduced regression models

Same 163-origin legacy-target evaluation. Results below are rounded from existing rerun logs. They use older annual fitting and timing choices, rather than the new primary specification.

| Existing specification | Model RMSE (pp) | Monthly persistence RMSE (pp) | R² relative to persistence |
|---|---:|---:|---:|
| RF future level, macro only | 2.4004 | 0.6124 | -14.3629 |
| RF future level, macro + spread history | 1.1131 | 0.6124 | -2.3037 |
| RF change, macro only | 0.8994 | 0.6124 | -1.1569 |
| RF change, macro + spread history | 0.8573 | 0.6124 | -0.9595 |
| Ridge change, macro + spread history | 1.1412 | 0.6124 | -2.4721 |

[Level rerun](../evidence/honest_eval.log); [change rerun](../evidence/fair_eval.log). All these reproduced specifications lose to monthly persistence. The ridge implementation needs scaled chronological tuning, and the macro inputs are lagged revised data. Old DM p-values are omitted because they are not the final dependent-data inference. Script prose calling these data point-in-time or the comparison the fairest test is not adopted here.

A separate stored 2019–2022 AR(1) comparison had MSE 0.4864 versus persistence 0.4998, with reported DM p=0.352 (see the original advisory). Therefore do not write that every model in the repository loses on every split.

## 3. Stress-state ranking after fixing unavailable future labels

The diagnostic deliberately retains the full-sample q75 threshold (6.432174 pp) and the old annual-fit/12-month-gap RF configuration to isolate label handling and the omitted baseline. It does not satisfy the proposed frozen-threshold protocol.

| Horizon (months) | n | Positive months | RF AUC | Current-spread AUC | RF AP | Current-spread AP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 163 | 31 | 0.9357 | 0.9538 | 0.8290 | 0.9002 |
| 3 | 161 | 29 | 0.7939 | 0.8370 | 0.4550 | 0.6514 |
| 6 | 158 | 26 | 0.6592 | 0.7512 | 0.2478 | 0.4862 |
| 12 | 152 | 20 | 0.5301 | 0.6705 | 0.1626 | 0.2633 |

AP = average precision. [Source table](../evidence/classification_label_diagnostic.csv); [origin-level predictions](../evidence/classification_diagnostic_predictions.csv). Current spread ranks better at each horizon in this diagnostic. It is a score, not a probability; no Brier score or calibration conclusion follows. Sample sizes differ across horizons, and no uncertainty for the differences has yet been computed.

The original classifier scored 1, 3, 6 and 12 unknown-future rows as negative at the respective horizons. The corrected diagnostic excludes these rows. This correction alone does not remove the full-sample-threshold problem.

## 4. How many actual stress entries?

| Threshold definition on legacy mean | q (pp) | Distinct post-2008 entries | h=1 at-risk origins | h=3 at-risk origins / positives |
|---|---:|---:|---:|---:|
| Full-sample q75, exploratory only | 6.432174 | 6 | 131 | 129 / 15 |
| q75 frozen through December 2008 | 7.245238 | 3 | 143 | 141 / 9 |

Frozen-threshold entry dates: **September 2011, January 2016, March 2020**. These are spread-threshold entries, not three independent recessions. At h=12 there are 36 positive windows but still only three entries. [Source counts](../evidence/onset_counts.csv). No event-warning model has been evaluated under this definition yet.

## 5. Project integrity and descriptive evidence

- The contemporaneous target is algebraically reconstructible from rolling mean and lag features; maximum numerical reconstruction error is 7.11e-15.
- The panel has 309 rows and no missing stored cells, but macro vintages and availability remain unresolved.
- Eighty monthly target values differ between legacy forward-filled and observed-day means; largest difference is 14.61 bps.
- March 2020 has observed mean 7.8541 pp, last 8.77 pp, maximum 10.87 pp and range 6.12 pp. Its mean ranks 43rd while its range ranks first in the diagnostic monthly series. Range describes within-month dispersion, not the speed or direction of a move.
- The six v2 implementation notebooks contain 47 empty code cells in total; the audit notebook is populated. A notebook filename is not evidence of an implemented experiment.

[Audit summary](../evidence/audit_summary.json) and [advisory](../RESEARCH_ADVISORY.md) supply provenance and broader context. The central regime-ablation estimate, its uncertainty, frozen-threshold probability forecasts and external replication remain pending.
