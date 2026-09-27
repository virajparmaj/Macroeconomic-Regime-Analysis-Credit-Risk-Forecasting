# Project scope and contribution

## Central question

**Do macroeconomic regime features improve monthly U.S. high-yield spread forecasts once the model and its benchmarks share the latest available spread information, and does any improvement extend to new stress entries?**

The existing project studies aggregate credit-market conditions using public macroeconomic series, PCA/clustering, regression, and random forests. Its useful research asset is the combination of a reconstructible monthly panel, underlying daily spread observations, and enough historical variation to test specific evaluation choices. It does not contain borrower-level defaults, so its target is credit spreads and spread-defined stress, not individual default probability or realized credit losses.

## Proposed contribution

Quantify, within one controlled study, how conclusions change when researchers use a monthly-mean versus latest-daily information set, add fold-fitted macro regimes, and distinguish future stress occupancy from entry into stress. Report both improvements and failures, with uncertainty and a reproducible prediction ledger.

The useful empirical finding would be the **magnitude and robustness of an incremental effect**, or a sufficiently precise bound on an economically relevant effect. Finding a bug, introducing a familiar baseline, or reporting an insignificant p-value is insufficient by itself.

## Narrow scope

| Choice | Recommended scope |
|---|---|
| Market | ICE BofA U.S. High Yield OAS, one aggregate series |
| Main target | Next-month mean spread; primary analysis uses the explicitly reconstructed observed-day mean |
| Compatibility target | Existing forward-filled monthly mean, to reconcile earlier results |
| Primary forecast horizon | One month |
| Secondary horizon | Three months; six/twelve months exploratory only |
| Main comparison | Spread + macro + regime versus the same spread + macro model |
| Model families | Scaled ridge with regime interactions; random forest robustness |
| Event analysis | Frozen-threshold state and at-risk entry; descriptive with current event count |
| Stronger validation | One frozen out-of-time extension; if unavailable, a clearly identified second-series sensitivity |

## What would solidify the study most?

1. A fair, common-origin comparison where the regime model cannot win merely by seeing a more recent spread.
2. A causal-in-time regime construction and a documented publication-lag approximation, followed by a vintage-data subset if feasible.
3. Paired uncertainty estimates that respect time dependence; quantify whether a practically meaningful gain remains plausible.
4. A fully disclosed event ledger that makes persistent stress and new entries visibly different.
5. One frozen external check. More models on the same inspected sample are less useful than this.

## Completion criteria

The paper-preparation stage is complete when the specification, evidence, implementation tasks and manuscript structure are documented; this folder fulfills that stage. The empirical study is complete only when every required result row in file 08 has a prediction artifact, validity checks, uncertainty, and a traceable table. A publishable manuscript additionally needs a sufficiently distinct empirical finding, complete related-work comparison, and an honest account of data limitations. No venue acceptance is implied.
