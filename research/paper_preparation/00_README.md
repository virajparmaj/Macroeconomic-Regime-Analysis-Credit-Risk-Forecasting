# Research paper preparation package

Updated 1 October 2026. **Status: historical study, sensitivity analysis, later-period holdout and empirical draft executed. Historical vintage validation remains unavailable.**

Recommended project: **Persistence, Aggregation, and Incremental Regime Information in U.S. High-Yield Spread Forecasting.** The modest contribution is a controlled empirical assessment of how benchmark information and event definitions change conclusions about macroeconomic regime forecasts. It is not a new forecasting algorithm or a claim that the underlying evaluation principles are new.

## Reading order

| File | Purpose |
|---|---|
| [01_PROJECT_SCOPE.md](01_PROJECT_SCOPE.md) | Research question, contribution, boundaries and completion criteria |
| [02_NOVELTY_AND_LITERATURE.md](02_NOVELTY_AND_LITERATURE.md) | Additional literature, overlap, defensible novelty and remaining search |
| [03_DATA_AND_VALIDITY.md](03_DATA_AND_VALIDITY.md) | Dataset, timing, provenance and threats to validity |
| [04_VERIFIED_FINDINGS.md](04_VERIFIED_FINDINGS.md) | Numerical evidence already obtained and its interpretation |
| [05_STUDY_DESIGN.md](05_STUDY_DESIGN.md) | Targets, comparisons, models and inference |
| [06_IMPLEMENTATION_PLAN.md](06_IMPLEMENTATION_PLAN.md) | Ordered engineering tasks, proposed paths and acceptance gates |
| [07_TESTS_AND_VALIDATION.md](07_TESTS_AND_VALIDATION.md) | Existing test status and necessary scientific validity tests |
| [08_RESULTS_REGISTER.md](08_RESULTS_REGISTER.md) | Completed versus pending results and manuscript table map |
| [09_CONCLUSIONS_AND_LIMITATIONS.md](09_CONCLUSIONS_AND_LIMITATIONS.md) | Conclusions justified now and conditional conclusions after experiments |
| [10_PAPER_BLUEPRINT.md](10_PAPER_BLUEPRINT.md) | Paper structure, preliminary abstract and figure specifications |
| [11_REPRODUCIBILITY.md](11_REPRODUCIBILITY.md) | Reproduction commands, environment and release checklist |
| [12_IMPLEMENTATION_PROMPT.md](12_IMPLEMENTATION_PROMPT.md) | Original execution brief retained for scope traceability |
| [13_EXECUTED_MANUSCRIPT.md](13_EXECUTED_MANUSCRIPT.md) | Generated empirical draft with actual tables, uncertainty and limitations |

The original broad [advisory](../RESEARCH_ADVISORY.md), [13-study literature review](../LITERATURE_REVIEW.md), and [experiment protocol](../EXPERIMENT_PROTOCOL.md) remain supporting material. This package narrows them into a practical study. The additional literature materially weakens any claim that the last-daily-observation benchmark itself is novel. The primary comparison is now explicitly the incremental contribution of regimes after macro and spread information, with the earlier macro-versus-spread comparison retained as secondary.

Use **verified exploratory**, **planned**, **unverified**, and **conditional** consistently. A successful software test does not establish forecast skill. An already-inspected historical period is not an untouched confirmatory holdout. Freeze the protocol before the new runs, disclose that the historical sample informed the design, and reserve stronger confirmation for new data.

## Delivered study

Start with [13_EXECUTED_MANUSCRIPT.md](13_EXECUTED_MANUSCRIPT.md), [08_RESULTS_REGISTER.md](08_RESULTS_REGISTER.md) and [11_REPRODUCIBILITY.md](11_REPRODUCIBILITY.md). The [run registry](../study_results/latest_runs.json) identifies the 72-job core, 236-job sensitivity, 16-job external and 72-job repeated-core runs. All complete without model failures. The test suite passes 58 tests, and all six study notebooks execute.

The primary regime MSE gain is 39.36%, with a 95% interval of −7.92% to 55.79%, versus ridge with the same macro information. It does not outperform endpoint persistence. All main calibrated alarm policies miss the three entries. These are exploratory empirical findings, not evidence of a generally successful warning system.

Earlier diagnostics and planning documents remain for traceability. Where they differ, the complete-month correction, frozen [protocol](../study/protocol.json), [decision log](../study/DECISIONS.md) and executed manuscript govern the delivered results. No institutional affiliation is asserted.
