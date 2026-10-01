# Executed results register

Updated 1 October 2026. Numerical results are generated from versioned ledgers. See [the executed manuscript](13_EXECUTED_MANUSCRIPT.md) and [run registry](../study_results/latest_runs.json).

| ID | Result | Status / artifact |
|---|---|---|
| P01 | Source identity, corrected complete-month window, timing assumptions | Executed; protocol, input manifest, run quality metadata; actual vintage reconstruction unavailable |
| P02 | Matched monthly versus endpoint information | Executed; core metrics |
| P03 | Primary incremental regime comparison | Executed; gain 39.36%, 95% interval [-7.92%, 55.79%] |
| P04 | Macro ablation | Executed; core comparisons and adjusted secondary p-values |
| P05 | State and at-risk entry panels | Executed; metrics and calibration tables |
| P06 | Event warnings and false alarms | Executed; three events; event summary and local alarm/event ledgers |
| P07 | Block length, window, refit, threshold, target, timing, seed and event influence | Executed; sensitivity comparisons and core event influence |
| P08 | External validation | Executed as separate April 2024-onward holdout; continuous post-2022 extension unavailable |
| P09 | Matched vintage validation | Not completed; public API requests rejected without configured credential |

The earlier A01–A07 audit records remain in `research/evidence/`. They used a panel containing partial boundary months and are not substituted for the corrected study. Model failures, exclusions, sample sizes and calendar gaps remain visible in ledgers/metrics.
