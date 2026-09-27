# Research paper preparation package

Prepared 25 September 2026. **Status: verified exploratory evidence and implementation plan; the proposed paper experiments are not yet complete.**

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

The original broad [advisory](../RESEARCH_ADVISORY.md), [13-study literature review](../LITERATURE_REVIEW.md), and [experiment protocol](../EXPERIMENT_PROTOCOL.md) remain supporting material. This package narrows them into a practical study. The additional literature materially weakens any claim that the last-daily-observation benchmark itself is novel. The primary comparison is now explicitly the incremental contribution of regimes after macro and spread information, with the earlier macro-versus-spread comparison retained as secondary.

Use **verified exploratory**, **planned**, **unverified**, and **conditional** consistently. A successful software test does not establish forecast skill. An already-inspected historical period is not an untouched confirmatory holdout. Freeze the protocol before the new runs, disclose that the historical sample informed the design, and reserve stronger confirmation for new data.

## Recommended next action

Implement milestones M0–M3 in file 06: freeze the specification, reconstruct the data contract, enforce label maturity, and run the matched benchmark/model matrix. M4 adds dependent-data uncertainty and event analysis. M5 is a deliberately limited external validation. Write the final results and conclusion only from the resulting prediction ledger.
