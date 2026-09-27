# Data, provenance and validity

## Existing assets

The merged panel has **309 months, December 1996–August 2022**, and no missing cells. This is a complete stored panel, not evidence that every value was observable at each historical forecast date. The daily HY spread source contains **6,679 nonmissing observations**. Input hashes and environment versions are in [audit_summary.json](../evidence/audit_summary.json).

| Stored column | Actual source | Existing assumed lag in months | Required treatment |
|---|---|---:|---|
| CPI | CPIAUCSL | 1 | CPI inflation transformation; distinguish revised values from original releases |
| FEDFUNDS | FEDFUNDS | 1 | Rate level/change, with monthly-average availability documented |
| Industrial_Production | INDPRO | 1 | Growth transformation; revised series |
| GDP | EA19LORSGPORGYSAM | 3 | Rename to Euro Area 19 OECD GDP growth reference series; not U.S. GDP |
| Unemployment_Rate | UNRATE | 1 | Rate level/change; revisions and release date matter |
| Consumer_Sentiment | UMCSENT | 0 | Verify historical preliminary/final timing; zero lag is an assumption |
| Credit_Spread | BAMLH0A0HYM2 | 0 | Separate observation date from actual availability date |

These lags describe the current implementation, not verified historical release calendars. Shifting final revised data is a **publication-lag approximation**, not a true point-in-time dataset. The `point_in_time` names and statements in existing scripts overstate what they establish.

## Target construction

The stored monthly target matches averaging the original daily series after forward-filling missing entries on its existing source-date grid. It differs from an observed-day-only mean in **80 months**, by up to **0.146111 percentage points = 14.6111 basis points**. Do not silently redefine the original target.

Keep two named targets: `mean_observed` for the main study, and `mean_legacy_ffill` for compatibility. Each target gets its own within-target comparisons. Include monthly endpoint and maximum only as separately labeled sensitivity targets; changing the outcome changes the forecasting problem.

Create an availability ledger with `series_id`, `reference_date`, `observation_date`, `available_at`, `vintage_date`, `value`, `timing_assumption`, and `source_hash`. If actual daily publication times cannot be recovered, call the baseline an idealized month-end information benchmark and include a one-business-day availability delay sensitivity. Do not call it deployable real-time performance.

## Main validity threats

| Threat | Evidence/impact | Remedy |
|---|---|---|
| Contemporaneous target encoding | Current spread reconstructed by `3*rolling_mean_3 - lag_1 - lag_2` to numerical precision | Future target and strict feature timestamps |
| Retrospective preprocessing | Full-sample scaling/clusters and centered smoothing appear in legacy analysis | Fold-only fitting; trailing/filtered state estimates |
| Unknown future labels set false | Comparisons with shifted NaN produce false labels in old classifier | Preserve unknown targets as missing; require complete future paths |
| Threshold chosen on test period | Old stress threshold uses the full sample | Freeze training quantile; report sensitivity separately |
| Too few stress entries | Three post-2008 entries under the frozen threshold | Event-level descriptive analysis; avoid claims of dependable warning |
| Revised macro data | Lagged values are not historical vintages | Explicit limitation; vintage subset or weaker retrospective claim |
| GDP identity and sample ending | Euro-area reference series limits panel to August 2022 | Correct naming; prespecify drop/replacement before extension |
| Inspected historical test sample | This audit already used 2009–2022 outcomes | Disclose exploratory design; untouched extension for confirmation |

A suitable extension is September 2022 onward with a prespecified five-macro specification excluding the discontinued Euro-area series; rerun the same five-variable specification historically so the extension is comparable. Acquiring and freezing it is planned, not completed. Record the data acquisition date and any prior exposure to the proposed holdout.
