# Paper blueprint and preliminary writing

## Working title

**Persistence, Aggregation, and Incremental Regime Information in U.S. High-Yield Spread Forecasting**

Alternative if the event analysis becomes sufficiently informative: **Persistence or Advance Warning? Evaluating Macroeconomic Regimes in U.S. High-Yield Spreads.**

## Preliminary abstract — exploratory evidence only

Macroeconomic regimes can describe credit-market stress without necessarily improving forecasts beyond information already contained in spreads. This study proposes a controlled evaluation of monthly U.S. high-yield option-adjusted spreads, distinguishing information from temporal aggregation, incremental regime features, and entry into stress. An audit of a 309-month panel identifies target-alignment and data-provenance issues. Exploratory comparisons show that the latest daily spread reduces next-month legacy-mean forecast RMSE from 0.6124 to 0.4212 percentage points relative to monthly-mean persistence. A simple current-spread ranking also exceeds the inspected random-forest stress-state classifier at each evaluated horizon. These diagnostics do not establish the performance of a final forecasting system: the classification comparison retains a retrospective threshold, and a training-frozen alternative identifies only three post-2008 stress entries. The planned study uses matched information sets, fold-fitted macro regimes, label-mature walk-forward evaluation, and dependent-data uncertainty to assess whether regimes add practically useful prediction. Final incremental-skill and external-validation results remain to be established.

Do not submit this as a completed-study abstract. After experiments, replace the proposal sentences with the actual primary estimate, interval, extension result and scoped conclusion. Preserve negative findings and uncertainty.

## Section map

| Section | Content to write | Evidence/source |
|---|---|---|
| 1 Introduction | Distinguish descriptive regimes from incremental prediction; motivate benchmark timing and warning task; state limited contribution | Files 01–02; no “first” claim |
| 2 Related work | Regime spread models, aggregation-aware evaluation, credit-stress warnings, leakage/time-series inference | Earlier review plus file 02 |
| 3 Data | Series identities, dates/units, daily aggregation, release approximation, input snapshots | File 03 and Table 1 |
| 4 Methods | Origin availability, targets/risk sets, model blocks, regime fitting, nested tuning, primary estimand | File 05 |
| 5 Results | Baselines first; incremental regimes; macro ablation; state/entry panels; event ledger | P02–P06 only after execution |
| 6 Robustness | Target construction, block dependence, q sensitivity, event influence, vintages and external period | P07–P09 |
| 7 Discussion | Economic interpretation, practical effect size, limits of event sample and representation | File 09 updated to actual findings |
| 8 Conclusion | Answer the primary question with effect and uncertainty; scope future work | No stronger than the result ledger |
| Appendices | Full feature definitions, split rules, test cases, all configurations, source/version manifests | Files 06–07 and release artifacts |

## Figures to produce from final results

1. Spread history: daily observations, observed/legacy monthly mean, frozen threshold, evaluation boundary and entry dates. Identify different target definitions rather than merging them visually.
2. Paired model losses and regime gain intervals: same target/origins, endpoint benchmark visible, block-length sensitivity explicit.
3. State versus entry: three risk-set panels with prevalence and valid sample counts. Do not imply directly comparable AUCs across different populations.
4. Event timeline: actual entries, alarm episodes, lead times and false alarms for the prespecified rule.
5. Historical versus external performance: separate periods, common frozen specification, no pooled headline concealing instability.

The existing [diagnostic figure](../evidence/research_diagnostics.png) is suitable for internal explanation. Its classification threshold and exploratory status must be disclosed if reused; it is not a substitute for final figures.

## Claim-to-evidence rule

Every number in the abstract and conclusion must identify a row of the final metrics table and its run ID. Label the current source audit as this project's audit, not a replication of another author's code. The paper's novelty statement must distinguish established methodology from the empirical finding delivered here.
