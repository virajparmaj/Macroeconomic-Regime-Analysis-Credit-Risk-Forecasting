# Research contribution assessment

25 September 2026 · Repository inspected directly; literature and selected numerical checks completed.

**Recommendation:** develop a narrowly scoped empirical paper about **persistence versus advance warning in credit-spread prediction**. The current project is not ready to support a paper claiming a successful macroeconomic regime forecasting system. Its strongest opportunity is a controlled study of how benchmark choice, information timing and event definition change the conclusions.

Proposed title: **Persistence or Advance Warning? Evaluating Macroeconomic Regimes in U.S. High-Yield Credit-Spread Forecasting**.

This is a defensible *research direction*, not a finding of guaranteed novelty or acceptance. The original methods and several economic observations already exist in close prior work. A standalone correction of an unpublished project is not automatically a research contribution. The proposed extension must establish a useful, reproducible empirical result beyond identifying implementation errors.

The detailed [literature review](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/LITERATURE_REVIEW.md) maps 13 relevant works. The [experiment protocol](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/EXPERIMENT_PROTOCOL.md) specifies every additional experiment, split, baseline, metric, statistical procedure, output and falsification condition.

![Exploratory diagnostics: persistence controls and event counts](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/research_diagnostics.png)

## 1. What the project actually investigates

The scientific object is the **ICE BofA US High Yield Index option-adjusted spread**, not individual default probabilities or realized loan losses. The question is whether a small macroeconomic information set and inferred states add information about future aggregate spread levels, changes, or threshold events beyond the spread's own recent history.

The usable panel has 309 complete monthly rows, December 1996–August 2022. Its six predictors are CPI, federal funds rate, industrial production, unemployment, consumer sentiment, and an OECD Euro-area growth reference series mislabeled `GDP`. The latter is explicitly identified by [FRED as Euro Area 19 year-over-year growth](https://fred.stlouisfed.org/series/EA19LORSGPORGYSAM), not US GDP. The daily spread source contains 6,679 nonmissing observations.

Actual implemented work spans data aggregation, PCA/clustering, RF and recurrent-model experiments, later leakage diagnostics, descriptive macro–spread associations, lead–lag tests, rolling regressions, walk-forward regression, multi-horizon stress classification, and daily-versus-monthly aggregation. The website visualizes selected conclusions; it is not independent evidence.

Several model comparisons do investigate substantive issues, but they do not all answer the same task. Contemporaneous spread fitting, one-month-ahead level prediction, predicting changes, classifying a future high-spread state, and predicting the *start* of a new episode must be separated.

### Demonstrated, suggested and unestablished

| Status | Evidence | Defensible interpretation |
|---|---|---|
| Demonstrated by fresh computation | Original rolling mean plus two lags reconstructs contemporaneous spread with maximum error 7.1×10⁻¹⁵ | The original feature construction contains the label algebraically; its R² is not evidence of forecasting skill |
| Demonstrated by stored audit outputs and code inspection | Original full-sample PCA/clustering and centered regime smoothing; misleading regime names; numerical prose/output mismatches | Original regime and deep-model claims need reconstruction, not editorial polishing |
| Reproduced in this review | On 163 origins, mean-persistence RMSE 0.6124 percentage points; macro+spread level RF 1.1131; change-target RF 0.8573 | These specific RF configurations lose to the stated benchmark |
| Reproduced in this review | Macro-only regime split gives contemporaneous R² 0.0743 calm / 0.4987 stress | Stronger descriptive association in this retrospective partition; not demonstrated causal-regime forecast value |
| Newly computed | Raw current-spread score beats the existing spread-only RF classifier at h=1,3,6,12 under the same exploratory labels | The classifier's apparent skill and horizon limit were judged without an adequate ranking baseline |
| Newly computed | Using the last daily observation predicts the legacy next-month mean with RMSE 0.4212 versus 0.6124 for mean persistence on post-2008 origins | Even the original random-walk comparison leaves a materially stronger available-information benchmark unused |
| Suggested | State persistence, aggregation and input timing account for much apparent model performance | A useful joint hypothesis; the fraction attributable to each factor has not been isolated |
| Not demonstrated | Reliable new-stress onset prediction, calibrated risk probabilities, incremental causal-regime skill, real-time vintage performance, or cross-market generalization | These require new experiments; neither the README nor completed-looking notebook names establishes them |

Sources: [fresh diagnostic summary](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/audit_summary.json), [regression rerun](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/honest_eval.log), [change-target rerun](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/fair_eval.log), [regime rerun](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/macro_regime.log), and [original completed audit notebook](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/notebooks/v2/00_audit_of_v1.ipynb).

The measured losses do not prove that macroeconomic information is absent. A finite-sample estimator can fail to exploit information, especially with weak predictors and few events. “No robust incremental gain found for these specifications” is a possible conclusion; “macroeconomics cannot forecast spreads” is not.

The reproduced level RF macro+spread comparison has nominal DM p=0.0561 under the existing implementation. The much smaller p-value quoted for macro-only belongs to a different comparison and cannot be transferred to the combined model; all such inference remains subject to the dependence corrections below.

## 2. Important corrections to the existing analytical report

1. **The original R² was compromised by more than its benchmark.** The rolling identity is s[t]=3×mean(s[t],s[t−1],s[t−2])−s[t−1]−s[t−2]. The current report's wording that the number was merely measured against the wrong null understates a target-contamination and task-definition problem. Its stored audit also failed to exactly reproduce the original 0.8084 headline from the structure printed in the original notebook; describe that discrepancy accurately.

2. **The classifier turns missing futures into negatives.** At [horizon_class.py:47](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/analysis/horizon_class.py:47), comparison with NaN returns false before conversion to integer. The last h months therefore enter scoring without observed outcomes. Fix labels with an explicit future-observed mask. This defect is real, but correcting it alone changes the headline AUCs only modestly; do not exaggerate its numerical impact.

3. **Threshold and regime look-ahead remain.** Classification uses a full-sample 75th percentile. [macro_regime.py](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/analysis/macro_regime.py) globally standardizes its index and uses its full-sample percentile. Omitting spreads from an index removes one source of circularity; it does not make the index available in real time. Its within-state regressions are in-sample, contemporaneous fits.

4. **Publication shifts are not historical vintages.** [src/data.py](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/src/data.py) shifts revised data by assumed whole months. This is a publication-lag approximation. The actual release calendars and revision histories remain unverified. Revisions need not always make accuracy better; unexpected results should be investigated, not rejected because they contradict an expectation.

5. **Most of the advertised v2 study is unimplemented.** Notebooks 01–06 contain 47 code cells, all empty. Only notebook 00 has executed code and outputs. `results/metrics.csv` and `data/v2/panel_pit.parquet` are absent. Existing analysis scripts nevertheless perform real computations; distinguish them from the unfinished integrated workflow.

6. **“Nothing beats the random walk” is too broad.** The completed audit's AR(1) has MSE 0.4864 versus 0.4998 for persistence on its 2019–2022 split, though its reported DM p≈0.352 does not establish superiority. The change-target RF also has lower error than the level-target RF, so 1.1131 is not the best error across all later specifications. Report exact tasks/configurations.

7. **Existing inferential claims need repair.** The DM implementation uses only h−1 autocovariance terms; at h=1 it makes no allowance for additional serial correlation in loss differentials. Macro-regime correlation intervals resample independent rows. Granger scripts select the minimum p-value over six lags without correction; crisis-excluded tests concatenate separated calendar periods before constructing lags. The break script calculates a sup-F statistic but labels the result sup-Wald and uses unsourced approximate critical values. These do not justify strong significance or stability statements.

8. **Several reported interpretations overreach.** A 0.54 AUC point estimate is not evidence of equality to chance without uncertainty; a model's horizon decay is not a universal predictability boundary. “Spread leads macro” does not logically imply macro cannot improve conditional spread forecasts. NBER agreement is descriptive external-label agreement, not out-of-sample forecasting validation. The earliest alarm in a 24-month pre-recession window, without counting all other alarms, is not evidence of useful warning.

9. **The effective-sample-size claim is misused.** n(1−ρ)/(1+ρ) is an approximation for particular mean-estimation settings, not a universal count of independent observations for every model, metric, or regression. Use dependence of the actual loss/statistic, plus distinct event counts. Differencing does not create new independent crises.

10. **Metric provenance is fragile.** `analysis/export.py` contains typed numbers; its 0.857 variance-share literal disagrees with the saved 0.832. It must not be the authoritative generator for a paper. The source monthly panel also differs from observed-only daily averages because of forward filling. This review reconstructed the forward-filled series to floating-point precision. The largest observed-only difference is September 2001: 0.146111 percentage points, or 14.61 basis points.

These are research-validity issues rather than reasons to abandon the project. A clean experiment ledger will resolve many at once.

## 3. Closest research and novelty-gap judgment

The detailed comparison is in the literature review. The most important competitors are the HMM–LSTM credit forecast paper (Erlwein-Sayer et al.), monthly macro/stacking study (Shao et al.), explainable spread-change models (Heger et al.), and the regime studies by Maalaoui Chun et al. Gilchrist–Zakrajšek and Faust et al. pre-empt the simple reverse-direction story. Anderson–Audzeyeva pre-empts claims that persistence benchmarks or robust forecast comparison are new.

| Possible novelty source | Judgment | What would make it defensible |
|---|---|---|
| Existing model/dataset combination | Very weak | No novelty claim; conventional components are appropriate tools |
| Macro correlations strengthen during stress | Low | A new controlled implication for *incremental forecasting*, not another descriptive split |
| Timing and hindsight interaction | Moderate, provisional | Same target and test dates, causal versus retrospective regimes, matched release-timing controls |
| State classification versus onset warning | Strongest relative candidate, still provisional | Persistence controls, fixed labels, event-level alarms and honest uncertainty |
| Aggregation-specific benchmark mismatch | Moderate as supporting result | Hold the target fixed, compare mean/last-observation controls, and quantify target-sensitive rankings |
| Negative incremental macro result | Low alone; moderate within controlled study | Good baselines, fair scaling/tuning, practical-effect intervals and event stability |
| Reproducibility/resource contribution | Low alone | Reusable transparent forecast ledger and computational ablations that teach more than one coding mistake |
| Geographic mismatch / feature importance | Low alone | Remove/replace the EA series and show a reproducible consequence; importance is not causal attribution |
| Regime label instability/generalization | Moderate support | Out-of-time labels, permutation-invariant agreement, feature drift and incremental forecast loss |
| Latency, memory or seed effects | Low priority | Only if performance is comparable; 309 monthly rows do not justify a systems-efficiency paper |
| Cross-dataset/temporal validation | Credibility multiplier | Frozen protocol on a genuinely new period/series, with shared-crisis dependence acknowledged |

The gap is **not** “nobody tried these six indicators.” It is the unresolved empirical distinction, in this application, between fitting persistent states and extracting additional information that warns of changes. Combining known controls becomes a contribution only if their measured consequences are useful and nontrivial.

## 4. Eight candidate research questions, ranked by defensibility

### RQ1 — Does high stress-classification AUC mostly reflect persistence rather than new-event warning?

**Hypothesis:** simple spread-ranking/calibrated-persistence controls match state discrimination, while onset performance is weaker. **Why it matters:** warning decisions depend on change, not repeatedly identifying an ongoing event. **Known/unknown:** regime persistence is established; its contribution to this classifier's apparent short-horizon skill has not been isolated. **Existing support:** the newly computed ranking baseline beats RF at every horizon. **Additional work:** E0, E3, E4; matched thresholds, calibrated probabilities, onset labels and false alarms. **Distinctness:** moderate and the most promising relative to required effort. **Risk:** only three post-2008 entries under the training q75; onset comparisons may be uninformative. This can support a warning against overinterpreting state AUC, not automatically a superior onset model.

### RQ2 — Do macro regimes add forecast information after controlling for the latest spread and available releases?

**Hypothesis:** descriptive regime association exceeds incremental causal-regime forecast value. **Why it matters:** macro state descriptions are often interpreted as predictive inputs. **Known/unknown:** regime-dependent dynamics and regime-feature models exist; the joint timing-controlled comparison for this panel is unfinished. **Existing support:** retrospective regime R² contrast and poor macro RF forecasts. **Additional work:** E1–E2 with no-regime controls and training-only clustering. **Distinctness:** moderate only when focused on the timing/benchmark interaction. **Risk:** causal regimes improve performance, or the comparison reproduces known findings without a useful new effect size. Both outcomes must be allowed.

### RQ3 — How does the aggregation of a daily spread change the appropriate monthly forecast benchmark?

**Hypothesis:** month-end information outperforms mean persistence for a next-month mean target; model rankings depend on the target/baseline pair. **Why it matters:** otherwise models get credit for recovering stale versus recent price information. **Known/unknown:** temporal aggregation and strong naive forecasts are standard; the magnitude and regime concentration here are testable. **Existing support:** post-2008 baseline RMSE falls from 0.6124 to 0.4212 with the target unchanged. **Additional work:** E1, R2, dependence intervals and period stability. **Distinctness:** moderate supporting contribution, low as a single obvious comparison. **Risk:** improvement is a predictable information-recency effect with no wider insight; do not present it as a novel algorithm.

### RQ4 — Can a small macro block deliver materially useful forecast gains over spread-only models?

**Hypothesis:** gains above a predeclared practical threshold cannot be established consistently. **Why it matters:** it tests the marginal value of data rather than a model leaderboard. **Known/unknown:** macro/spread prediction is established and context-dependent; this information set's conditional gain remains unmeasured fairly. **Existing support:** macro models lose to persistence, but some fair spread-only regression controls are missing. **Additional work:** E1, E4 and standardized chronological tuning; R1 strengthens interpretation. **Distinctness:** low alone, moderate in the integrated study. **Risk:** estimator weakness or insufficient precision is mistaken for absence of information. A nonsignificant p-value is not the hypothesized negative result.

### RQ5 — Are macro-derived states stable enough out of time to serve as forecast features?

**Hypothesis:** hindsight versus causal labels disagree most near transitions, where predictive use matters. **Why it matters:** post-hoc state diagrams can imply unavailable certainty. **Known/unknown:** filtered/smoothed distinctions are foundational; their practical effect on this regime construction is untested. **Existing support:** full-sample fitting, centered smoothing, tiny clusters and misleading labels in v1; notebook 04 is only an outline. **Additional work:** E2, two/three-state sensitivity, adjusted Rand agreement and forecast ablation. **Distinctness:** low-to-moderate. **Risk:** label permutation creates artificial disagreement, or regime instability merely reflects trend-driven clusters.

### RQ6 — Does monthly averaging underrepresent rapid spread shocks in a way that changes warnings?

**Hypothesis:** high-range events rank differently from high-level events and mean-based labels miss or delay some crossings. **Why it matters:** slow stress and rapid repricing can require different monitoring. **Known/unknown:** separate level/volatility regimes are already documented; target-sensitive warning consequences are not tested here. **Existing support:** March 2020 ranks first by range but 43rd by mean. **Additional work:** R2 with prespecified events and false-alarm comparisons; possibly O1. **Distinctness:** low alone, moderate paired with evaluation evidence. **Risk:** the result is merely the definitional fact that means are below maxima, or a single-event anecdote. Monthly range is not literally a speed measure because it ignores path ordering and elapsed time.

### RQ7 — Do the spread–macro lead–lag asymmetries survive publication timing and calendar-preserving inference?

**Hypothesis:** some asymmetries persist, but estimated strength depends on alignment and event concentration. **Why it matters:** it distinguishes economic precedence from reporting delay. **Known/unknown:** spreads predicting activity is well established; the proportion attributable to this dataset's alignment is not. **Existing support:** exploratory cross-correlations/Granger outputs. **Additional work:** O3, vintage subset, prespecified multivariate lag structure and multiple-testing control. **Distinctness:** low unless timing produces a strong, specific new finding. **Risk:** known result with weak causal identification; avoid making it the main paper.

### RQ8 — Do reported model rankings survive a controlled audit of information availability?

**Hypothesis:** rankings/gains change when target contamination, full-sample preprocessing, stale benchmarks and label maturity are corrected separately. **Why it matters:** it turns anecdotal debugging into a replicable empirical analysis. **Known/unknown:** leakage taxonomy/evaluation principles exist; their relative consequences in this workflow have not been isolated. **Existing support:** exact target reconstruction and report/code discrepancies. **Additional work:** E0–E4, a same-target factorial comparison, and preferably a second pipeline/series or O1 mechanism check. **Distinctness:** moderate as a research software/evaluation study, weak as self-audit alone. **Risk:** reviewers see only correction of preventable errors rather than a new scientific result.

Use RQ1 as the central story, RQ2 as its macro-regime test, and RQ3 as a necessary benchmark correction. RQ4–RQ5 are supporting analyses. RQ6–RQ8 should not become separate sprawling projects.

## 5. Hidden results and their implications

The following classifier numbers were **computed during this review**, reproducing the old spread-only RF settings and correcting only unavailable future labels. Both models use exactly the same valid labels/dates per horizon. The full-sample threshold is deliberately retained to isolate this correction, so these remain exploratory.

| Horizon | Valid origins | Positive months | RF ROC-AUC | Current spread ROC-AUC | RF average precision | Current spread average precision |
|---|---:|---:|---:|---:|---:|---:|
| 1 month | 163 | 31 | 0.936 | **0.954** | 0.829 | **0.900** |
| 3 months | 161 | 29 | 0.794 | **0.837** | 0.455 | **0.651** |
| 6 months | 158 | 26 | 0.659 | **0.751** | 0.248 | **0.486** |
| 12 months | 152 | 20 | 0.530 | **0.670** | 0.163 | **0.263** |

[Full diagnostic table](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/classification_label_diagnostic.csv). No confidence interval or statistical superiority claim is implied. Positive months are clustered. The current spread score requires no model fitting and is not a calibrated probability.

**This changes the central story.** The previous report recommended shipping the classifier and declared a three-month limit. Neither conclusion follows. The better baseline shows that the RF curve does not establish the market's forecastability boundary. More importantly, none of these AUCs tells us whether a model predicts a new crisis before spreads have already risen.

The current full-sample threshold produces six post-2008 entry dates; a threshold fixed using pre-2009 data produces only three. Multiple near-threshold crossings may still belong to the same economic episode. Threshold choice changes the task and effective event evidence, not just a tuning constant. See [onset counts](/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/research/evidence/onset_counts.csv).

The other consequential finding is the stronger last-observation benchmark. It reduces next-month legacy-mean RMSE by **31.2%** on the same 163 post-2008 origins, without changing the target or fitting a model. The roughly 35% figure in the old report applies to its full-sample observed-day aggregation comparison; it is not the same evaluation. This supports investigating information recency, not declaring a new forecasting method.

Other observations are useful but weaker: the retrospective stress/calm fit gap; the concentration of loss in a few events; original feature importance concentrated in target-derived inputs; and unstable semantics of the regime labels. Rolling R² falling before the GFC is an exploratory anecdote, not a validated early-warning indicator. Seed instability has not been established, and neither have useful accuracy/energy tradeoffs.

## 6. Minimum publishable extension

Run the five ESSENTIAL bundles in the experiment protocol: **E0 data/timing reconciliation; E1 matched benchmark/feature comparisons; E2 regime/timing ablation; E3 state versus onset; E4 dependent uncertainty and event sensitivity.** They share one forecast ledger and reuse the existing data and model families.

True historical vintages are strongly recommended and become essential if the paper claims real-time deployability. A frozen second period or series substantially improves peer-review credibility but is not needed to investigate the narrow within-sample evaluation question. Adding LSTM layers, transformers, SHAP plots or more arbitrary clusters has lower value than these controls.

The strongest required result is **not a higher AUC**. It is a stable, quantified contrast showing whether apparent aggregate state/level skill remains incremental after appropriate controls, and whether it translates to new-event warning. If the onset comparison lacks precision, that limit belongs in the primary findings. If macro features do help under the corrected design, report the gain and revise the narrative.

## 7. Strongest argument currently supported

“Previous research has established that credit spreads contain business-cycle information and that spread relationships can vary across regimes.

However, it remains unclear in this project how much apparent macro-regime forecasting skill exceeds spread persistence, and whether future high-spread classification demonstrates advance warning of new stress.

We investigate these distinctions using a monthly US high-yield spread panel and its underlying daily observations.

Our experiments evaluate target construction, benchmark choice, available information, causal versus retrospective regimes, and persistent-state versus onset labels. The full controlled design is proposed; only the explicitly identified diagnostic subset has been completed.

Our current results show algebraic target contamination in the original pipeline, failure of specified corrected RF forecasts to outperform persistence, and stronger stress-state rankings from the current spread than from the existing classifier under exploratory labels.

These findings suggest that headline fit and state discrimination require stronger evidence before they can be interpreted as incremental macroeconomic forecasting or early warning.

The proposed contribution is a reproducible, benchmark-controlled assessment of this distinction, with measured effect sizes and explicit limits from event scarcity.”

For a completed-paper version, remove the proposed/completed distinction only after E0–E4 exist. Do not fill the final results sentence with anticipated onset or regime effects.

## 8. Completed study design

| Component | Proposed final specification |
|---|---|
| Research question | Incremental macro/regime information versus persistence, and state discrimination versus advance warning |
| Hypotheses | H1–H3 in the protocol; target-aggregation hypothesis H4 secondary |
| Dataset | Existing monthly panel plus daily spread source; corrected identities, units and source processing |
| Experimental design | Chronological expanding fits, label-maturity controls, matched model dates, controlled timing/regime factors |
| Models/baselines | Mean and last-observation persistence, AR(1), ridge/RF; raw-spread ranking, calibrated persistence, transition model, logistic/RF |
| Evaluation metrics | Paired RMSE/MAE and benchmark R²; Brier/log loss, AUC/AP; event recall, lead time and false alarms |
| Statistical testing | Paired block intervals, appropriate HAC comparisons, nested-linear adjustment only where justified, prespecified practical margins and secondary multiplicity control |
| Robustness | Origin/window sensitivity, event deletion, q70/q75/q80, common horizon dates, revised versus vintage subset |
| Ablations | S/M/S+M/S+M+R, causal/hindsight regimes, lagged/unlagged releases, mean/last targets, EA-series exclusion |
| Figures needed | Benchmark skill over time; state versus onset performance; causal/hindsight regime timeline; aggregation panel as support |
| Tables needed | Data provenance; model/feature performance; timing/regime factorial; event/false-alarm ledger; robustness intervals |
| Primary result to estimate | Incremental macro/regime performance above strong persistence controls and whether state skill transfers to onset |
| Secondary results to estimate | Aggregation-specific benchmark effects, regime label revisions, concentration by episode |
| Limitations | One aggregate market, small macro set, few distinct events, revised inputs, early end date and post-selection exploration |
| Threats to validity | Leakage, availability assumptions, threshold choice, heteroskedastic/dependent losses, training-window changes, multiple comparisons, repeated use of the test period |
| Reproducibility | One prediction ledger, generated metrics, input hashes, fixed versions/seeds, runnable command and availability-invariance checks |

The primary and secondary result rows describe estimands, not fabricated results. A negative finding requires enough precision to distinguish absence of a useful gain from an underpowered comparison.

## 9. Proposed paper structure

1. **Abstract:** specify market, dataset, task distinction and evaluation controls; report only completed effect sizes and limitations. No generic claims of accurate “credit risk” prediction.
2. **Introduction:** explain why a model can rank persistent high-spread months well without giving a new warning. State the narrow incremental-information question and preview the controlled comparisons.
3. **Related Work:** separate spread determinants, regime forecasting, real-time macro prediction and evaluation methodology. Contrast directly with the closest competing studies; acknowledge all established ingredients.
4. **Research Question / Hypotheses:** define level, change, future-state and onset tasks; state primary versus secondary hypotheses and predeclared practical-effect thresholds.
5. **Data:** source identifiers, units, EA-series identity, observed/filled daily aggregation, observation/release/vintage times, coverage, missingness and episode counts. Explain sample termination.
6. **Methodology:** define causal features and regimes, benchmark forecasts, classification calibration and onset risk set. Put the original leakage identity in a short motivation box or appendix.
7. **Experimental Setup:** publish cutoff/refit rules, eligible label dates, training-only transforms/tuning, matched comparisons, inference design and exploratory/confirmatory boundaries.
8. **Results:** lead with incremental skill over strong benchmarks, then state-versus-onset evidence. Show uncertainty and events before optional model details. Include null/adverse results.
9. **Robustness / Ablation Analysis:** timing/regime factors, aggregation, origin/window, geographic input, thresholds and event influence. Explicitly report reversals and undefined metrics.
10. **Discussion:** distinguish market persistence from model value and descriptive macro states from advance information. Interpret mechanisms cautiously; explain practical implications without offering investment recommendations.
11. **Limitations:** event scarcity, vintage approximation, single-market scope, selected historical sample and no identified causal mechanism. Note that multiple correlated series would not create independent crises.
12. **Conclusion:** answer the exact empirical question within the tested design. State what additional data would change the conclusion. Do not end with a claim that a general-purpose early-warning system has been built.

## 10. Central story and submission decision

**Proposed title:** Persistence or Advance Warning? Evaluating Macroeconomic Regimes in U.S. High-Yield Credit-Spread Forecasting.

**One-sentence proposed contribution:** We quantify how persistence benchmarks, information timing and stress-event definitions alter the measured incremental value of macroeconomic regime features in US high-yield spread prediction.

**Research gap:** a controlled, reproducible separation of continued-state recognition from additional advance information in this application; priority remains provisional after the accessible literature search.

**Primary question:** do macro/regime models add useful information beyond the latest spread, especially before a new stress episode begins?

**Primary hypothesis:** much apparent level/state skill is matched by persistence, and its translation into new-event warning is weaker than headline scores suggest.

**Novelty claim:** a specific empirical comparison and its measured implications; no novel model architecture, causal economic law, or first use of regimes.

**Most important experiment:** E3's training-threshold state/onset comparison with calibrated spread-persistence controls, interpreted alongside E1's incremental macro ablation.

**Most important result needed:** a stable effect relative to strong controls, or an explicit precision limit that prevents the original warning claim. One attractive ROC curve is insufficient.

**Minimum additional work:** E0–E4 and a unified reproducible result ledger. R1/R3 are high-value additions; R2 becomes essential for a central aggregation claim.

**Three strongest proposed exhibits:** (1) a paired benchmark/feature-ablation table with intervals; (2) state-versus-onset performance with event counts and false alarms; (3) causal-versus-hindsight regime/timing ablation with identical targets and origins. A daily March 2020 panel is supporting illustration rather than the main scientific result.

**Closest competitors:** Erlwein-Sayer et al.; Shao et al.; Heger et al.; Maalaoui Chun et al. Their methods/questions overlap substantially; the proposed distinction lies in the controlled available-information and persistence/onset evaluation. Full-text follow-up remains necessary where access was limited.

### Potential abstract — accurate for the evidence currently available

Macroeconomic regime models are often evaluated through credit-spread fit or discrimination of high-spread periods, although these quantities need not establish incremental forecast information or advance warning. We examine this distinction in a 309-month US high-yield option-adjusted-spread panel and its daily source data. An audit identifies algebraic target contamination in the original contemporaneous specification. Recomputed expanding-window evaluations find that specified macroeconomic random-forest forecasts have higher one-month error than persistence. On 163 post-2008 origins, replacing previous-month-mean persistence with the latest daily observation reduces RMSE for the same monthly target from 0.6124 to 0.4212 percentage points. Under an exploratory full-sample stress threshold, current-spread rankings outperform the existing random-forest classifier at all four tested horizons after unavailable future labels are removed. These diagnostics motivate a controlled comparison of causal regime features, release timing, and persistent-state versus onset prediction. Revised macroeconomic data, hindsight thresholds and few independent stress episodes currently limit inference. The evidence supports stronger benchmark and event-definition controls, but does not yet establish a general absence of macroeconomic predictive information or a validated early-warning model.

This is a **working diagnostic abstract**, not a finished abstract pretending the proposed onset experiments have run. Once the full design is complete, replace the motivation/proposal sentences with the measured causal-regime and onset results, including uncertainty.

### Claims to avoid

- “We discovered that credit spreads lead the economy.”
- “This is the first macro-regime/ML credit-spread model.”
- “R²≈0.81 proves forecasting ability,” or that the original issue was only the benchmark.
- “Nothing beats a random walk,” “macro has no information,” or “ML cannot forecast credit spreads.”
- “AUC≈0.94 establishes useful early warning,” “the forecastable horizon is exactly three months,” or “0.54 proves chance performance.”
- “Point-in-time data” for revised inputs shifted by assumed release lags.
- “A regime causes spread changes,” “feature importance identifies economic drivers,” or “the spread forecast implies default/loss forecasts.”
- “The study tested three independent held-out recessions,” “monthly range measures repricing speed,” or “six independent observations is the effective sample size for every test.”
- “Publication ready” because the repository contains named notebook stages, an interactive site, or a large number of models.

**Current readiness:** enough for a serious research proposal and a reproducible diagnostic note; not enough for the intended completed empirical paper. E0–E4 are the exact path to a defensible narrow preprint. For a workshop or journal, a second period/series, authentic-vintage sensitivity, or a well-designed mechanism simulation would materially strengthen the contribution. Acceptance and novelty cannot be guaranteed. If corrected results remain event-specific or statistically indeterminate, publish that limited scope honestly or extend the evidence before making a broader claim.

## What was actually run for this advisory

Original data and existing scripts/notebooks were not edited. The frozen v1 notebooks were inspected as JSON, not executed. This review added `research/audit_checks.py` and reran `analysis/honest_eval.py`, `analysis/fair_eval.py`, and `analysis/macro_regime.py` with outputs redirected into `research/evidence/`. The new diagnostic reproduces the old spread-only classification scores before isolating the future-label correction and adding raw-spread controls. It also records source hashes, target identities, target aggregation and threshold-entry counts. The existing test suite passed: **23 tests**. Passing tests do not validate the statistical study design.

Runtime for the diagnostics: Python 3.11.14, pandas 3.0.0, NumPy 2.4.1 and scikit-learn 1.8.0. This differs from the project's declared Python ≥3.12 requirement; record it for reproduction and build a consistent locked environment before paper experiments. No exhaustive model retraining, vintage reconstruction, onset classifier validation or full proposed factorial study was completed as part of this advisory.
