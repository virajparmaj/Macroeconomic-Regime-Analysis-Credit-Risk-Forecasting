# Minimum publishable extension: executable study specification

Status: proposed on 25 September 2026 **after exploratory inspection**. This is not a retrospective preregistration. Existing exploratory results must be identified as such. Freeze this protocol before running the new experiment matrix and document deviations.

The smallest credible paper is a focused empirical evaluation study. It does not require a transformer, new neural architecture, proprietary dataset, trading system, or another large model sweep. It does require a common evaluation harness and stronger controls. Estimated effort below is advisor judgment, not a measured execution commitment.

## Common design

**Research question.** How much apparent skill in monthly US high-yield spread and stress forecasts remains after controlling for available spread information, macro release timing, regime hindsight, and the distinction between persistent stress and entry into stress?

**Primary hypotheses.** H1: much apparent level/state skill is matched by persistence controls. H2: macro and causal regime features provide smaller incremental gains than contemporaneous associations imply. H3: discrimination of all future stress months is materially better than warning of new stress entries. H4, secondary: the previous month's last observed spread is a stronger benchmark for next month's mean than the previous month's mean. These are falsifiable expectations, not desired answers.

**Data.** Preserve all originals. Reconcile 309 monthly observations, December 1996–August 2022, and 6,679 nonmissing daily OAS observations. The legacy monthly target averages daily rows after forward-filling missing values on the existing source-date grid. Observed-day means differ in 80 months, by up to 0.146111 percentage points. Use the legacy target for exact replication and observed-day mean as a prespecified sensitivity; never silently combine their metrics. A last-observation benchmark uses the latest valid quote available by the forecast cutoff. Audit whether that quote was published by the chosen decision time.

**Forecast clock.** Define origin t as a decision time after the final available quote of month t and before observing any t+1 target data. Record observation month, publication date/vintage, and availability cutoff separately. Latest-observation features at t are legitimate for predicting t+h; they were illegitimate only when the original pipeline treated that same observation as the unknown contemporaneous label. Do not automatically shift away usable information.

**Evaluation periods.** Replication uses origins January 2009–July 2022 for h=1, giving 163 predictions. For h=3,6,12 the last eligible origins are May 2022, February 2022 and August 2021. Use identical origin intersections across models within each task/horizon. A cross-horizon figure should additionally use the common January 2009–August 2021 origin window. The existing data after 2009 contain the GFC recovery, not its onset. The 2001 recession is training history. Do not describe this as testing advance warnings for three held-out recessions.

**Refitting.** Primary: monthly expanding-window fits; sensitivity: annual refits matching the old scripts. At a refit cutoff T, admit a training origin u only when all of its label window has been observed by T. For horizon h this requires u+h≤T. Purge by target end date, not by a blanket number equal to the longest trailing feature window. Legitimate overlap in *past* input histories is not itself look-ahead leakage. A three- or twelve-month arbitrary gap can unnecessarily discard the most recent crisis observations.

**Feature sets.** S = current spread, first and third differences, trailing three-month mean and six-month volatility computed only from available observations. M = the six existing macro series, correctly named, with prespecified levels/changes; fit transformations using training data. S+M isolates incremental macro information. S+M+R adds causal macro-only regime indicators. A second small specification uses inflation/IP log changes, rate changes, sentiment change and the OECD growth level instead of changes in trending raw levels. Do not optimize the transformation set on test results.

**Models.** Train-mean forecast; lagged-month-mean persistence; last-daily-observation forecast; OLS AR(1); standardized ridge; the existing RF. For classification: historical prevalence, raw current-spread ranking, training-only logistic calibration of current spread, a two-state transition-probability model, regularized logistic regression, and RF. Use ridge/logistic with training-local scaling. For a compact tuning budget use inner chronological splits and a short prespecified parameter grid, or prespecify conservative hyperparameters. Default RidgeCV leave-one-out selection is not a time-respecting inner forecast evaluation. Deep models are unnecessary for this contribution; if retained, give them the same information and tuning budget.

**Regimes.** A low-variance option is a two-state K-means fit to stationary macro features, with scaler and clustering estimated before each test block; labels are arbitrary state IDs or stress-ranked using training centroids. Add a deterministic macro stress index as a sensitivity. No centered smoothing, test-fitted PCA, or labels named after inspecting future spreads. If an HMM is later used, forecast from filtered probabilities, not full-sample smoothed probabilities. A new HMM is optional.

**Classification labels.** Freeze a threshold at the pre-2009 training 75th percentile: 7.245238 percentage points for the legacy panel. Keep the full-sample threshold 6.432174 only as an explicitly invalid deployment counterfactual. Secondary thresholds are training q70 and q80; no test-based selection. Define state as 1{s[t+h]≥q}. Define onset as 1{max(s[t+1],…,s[t+h])≥q} only at origins with s[t]<q. A sustained-onset sensitivity requires two consecutive stress months and a prespecified calm washout. All required future observations must exist; unknown labels remain missing. Count distinct events and merged episodes separately from positive origin-months.

**Metrics.** Regression: RMSE/MAE in percentage points and basis points, signed bias, and benchmark-relative R²=1−SSE(model)/SSE(benchmark). Show ordinary test-mean R² only as a secondary diagnostic. Classification: Brier score and log loss for probability forecasts; ROC-AUC, average precision and prevalence for ranking. Raw spread is a ranking score, not a probability. Average precision/prevalence is AP lift, not precision at a chosen operating threshold. Warning usefulness requires event recall, lead time, missed events and false-alarm episodes per year at an alarm threshold chosen on training validation only.

## ESSENTIAL experiments

### E0. Data and information-availability reconciliation

**Change:** compare observed-day mean, legacy forward-filled mean and month-end observations; audit source identities and the timestamp of every feature/label. Compare zero-lag revised inputs with the documented publication-lag approximation.

**Models:** none for identities; persistence forecasts for checking target alignment. **Split:** all dates for provenance identities; pre-2009 versus post-2008 explicitly separated for predictive metrics. **Metrics/tests:** mismatch counts, maximum absolute differences, unavailable-feature/label counts; exact assertions rather than a significance test. **Output:** data dictionary, time-line schematic and provenance table.

**Claim supported:** the evaluated task and information sets are well-defined and reproducible. **Failure condition:** targets cannot be reconstructed or predictor availability cannot be justified. Resolve that before modeling. If historical vintages are unavailable, retain the label “publication-lag-adjusted revised data”; do not claim true real-time evidence.

### E1. Matched forecast benchmark and feature-ablation matrix

**Change:** S versus M versus S+M; predict changes versus direct levels; compare each against both mean persistence and last-observation persistence. Hold the target constant for benchmark comparisons. Primary h=1, secondary h=3; h=6,12 are exploratory horizon checks.

**Models:** AR(1), standardized ridge and RF, plus nonlearned baselines. Fit identical estimator specifications for S and S+M; use a residual formulation forecast=benchmark+predicted residual as one fair test of incremental information. “Predicting a change” does not automatically give an RF the econometric nesting properties required by a Clark–West test.

**Split:** monthly expanding evaluation above; identical test rows for all models; inner chronological tuning only. **Metrics/tests:** paired MSE/MAE differences, R² against both persistence forecasts, dependent confidence intervals specified below. Use a nested-model adjustment only for an explicitly nested linear comparison meeting its assumptions; do not attach it indiscriminately to RF or penalized models. **Output:** main model×feature table; cumulative loss-difference plot; per-year/per-event contribution table.

**Claim supported:** the measured incremental value of the selected macro block under this protocol. **Weakening/falsification:** S+M beats S and both persistence forecasts by a material, stable, uncertainty-supported margin. Publish that result if obtained; the negative-result hypothesis is not a requirement for success. A wide interval means inconclusive, not “macro has no information.”

### E2. Controlled regime-hindsight and release-timing ablation

**Change:** cross (training-only causal regimes / full-sample hindsight regimes) with (publication-lag-adjusted revised macro / unlagged revised macro), plus no-regime controls. Keep future target, predictor transformations, eligible dates and model settings otherwise fixed. Retrospective conditions are diagnostics unavailable to a real forecaster, not competitor methods.

**Models:** ridge with regime interactions and RF with regime indicators; one small macro-only clustering specification. **Split:** same outer evaluation as E1; all causal fitting inside each training block. **Metrics/tests:** paired change in forecast loss and incremental R², block-bootstrap intervals; adjusted Rand agreement or training-based label matching for regime revisions. A paired difference of differences can describe an interaction between timing and regime treatment. **Output:** a 2×2 effects table and causal-versus-hindsight state timeline.

**Claim supported:** whether a measured regime advantage depends on unavailable information. **Weakening/falsification:** causal regime features retain the same material gain, or the retrospective condition offers no gain. Do not prewrite the conclusion. The original nowcast versus a future forecast is not a valid single-factor ablation: changing the target changes the task.

### E3. Persistent-state discrimination versus new-stress warning

**Change:** evaluate the frozen-threshold state and at-risk onset tasks separately. First fix only missing future labels to reproduce the diagnostic; then replace the hindsight threshold; then examine onset. Add last-daily-spread and raw monthly-spread ranking controls. Estimate probabilities using a training-fitted calibration model rather than interpreting raw scores as probabilities.

**Models:** current-spread score; calibrated single-spread logistic model; transition-probability baseline; S, S+M and S+M+R logistic/RF classifiers. **Split:** common eligible origins within each task; onset restricts both training and evaluation to at-risk origins; label maturation follows the full horizon. Show common-window sensitivity across horizons. **Metrics/tests:** ROC-AUC/AP with paired time-block uncertainty; Brier/log loss versus calibrated persistence; event recall and false-alarm episodes; event-deletion sensitivity. Report undefined metrics when only one class remains.

**Output:** state-versus-onset performance table, forecast timeline, and one row per actual onset showing prior alarm(s), lead time, and false alarms elsewhere. **Claim supported:** whether high state-classification performance demonstrates advance warning or mostly ranks already-elevated spreads. **Weakening/falsification:** onset gains remain substantial against calibrated persistence, with useful lead times and tolerable false alarms across episodes.

**Power gate:** the pre-2009 q75 produces only three post-2008 threshold entries (September 2011, January 2016, March 2020) in the current data. No resampling method creates additional independent crises. With these data, onset is primarily a falsification/diagnostic exercise; a claim of a validated general-purpose onset predictor requires more event evidence. If there are too few positives for model fitting, report the insufficiency rather than manufacture balanced samples or change thresholds after seeing outcomes.

### E4. Dependence, event concentration and practical significance

**Change:** vary inference assumptions and remove each event from the *scoring set* in turn while preserving calendar alignment. Separately vary initial origin (predeclared 2003-12 versus 2008-12) and expanding versus fixed 120-month training windows. Earlier-origin results are robustness, not a magically untouched holdout after this exploratory work.

**Models:** frozen E1–E3 predictions for score resampling/deletion; refit only when the training-window design changes. **Split:** paired common dates; chronological training, never “train on both sides of the excluded crisis” and call it forecasting. **Metrics/tests:** 5,000 paired circular/moving-block resamples with prespecified mean/block lengths 6,12,24 months, preserving the joint actual/prediction vector. Flag sensitivity to nonstationarity; supplement with event/year loss accounting and HAC loss-difference inference with bandwidth at least h−1 and longer-bandwidth sensitivity. Treat annual folds and seeds as dependent experimental repeats, not independent datasets.

**Output:** confidence intervals, event-deletion plot, inference-sensitivity appendix, and a power/precision statement. **Claim supported:** stability and size of observed effects. **Weakening/falsification:** sign reversals by event, an interval admitting large practical gains, or a result driven by one episode. Do not hide these as failed robustness checks.

Define fractional improvement I=(loss_baseline−loss_model)/loss_baseline. Before new runs, justify a practical margin; 5% loss reduction is a candidate reporting threshold, not a finance-industry standard. An upper confidence bound below +5% can rule out gains of that size *under the stated inference assumptions*. A nonsignificant zero-difference test cannot. Equivalence needs an appropriate dependent-data procedure and an interval contained within both −5% and +5%; large deterioration is not equivalence. Use one primary contrast (S+M versus S at h=1) and Holm adjustment for the prespecified secondary family; label the rest exploratory. Episode counts may be too small to support strong inferential claims regardless of p-values.

## STRONGLY RECOMMENDED experiments

### R1. True-vintage subset and geographic provenance ablation

**Change:** replace available US macro series with as-of vintages; compare the same dates against revised inputs. Remove the Euro-area OECD growth series; optionally add correctly timed US GDP, which is a separate new-data sensitivity. **Models/split:** S+M ridge/RF under the same outer origins; freeze all other choices. **Metrics/tests:** paired losses, block intervals, feature/rank stability. **Output:** vintage/geography ablation table. **Claim:** results survive revision and mislabeling concerns. **Failure:** gains vanish or reverse under authentic vintages. For any “real-time deployable” claim this becomes ESSENTIAL. ALFRED documents its [historical-vintage purpose](https://alfred.stlouisfed.org/help); verify coverage rather than assume every old series is available.

### R2. Target aggregation and shock measurement

**Change:** mean versus month-end versus monthly maximum as distinct targets; observed-only versus forward-filled averaging; add trailing daily range/volatility as predictors. **Models/split:** persistence/AR(1), ridge/RF, identical origin windows and information cutoffs. **Metrics/tests:** target-specific benchmark-relative losses and paired block intervals within target; crossing delays, peak attenuation and event recall. **Output:** a target×benchmark matrix and daily-event panel. **Claim:** a particular evaluation result is sensitive to aggregation. **Failure:** normalized rankings and warning behavior are stable. Do not compare raw RMSE across different target distributions and claim model improvement. Future monthly maxima/ranges are outcomes, never inputs at the earlier origin. This is ESSENTIAL if aggregation is placed in the title or central contribution.

### R3. Untouched temporal or second-series validation

**Change:** reserve and collect a later evaluation window, or add a second HY/rating/IG spread series. The obsolete OECD reference series needs an explicitly documented treatment if extending beyond August 2022. **Models/split:** freeze preprocessing, thresholds and models from the original study; score the extension once; adapt neither to its outcomes. **Metrics/tests:** same paired metrics, interval estimates and event counts; paired/block dependence across correlated series. **Output:** external-validation table. **Claim:** limited transportability beyond the discovery sample. **Failure:** direction reverses or incremental macro value is confined to one series. Correlated rating indices do not create independent crisis replications. Strongly recommended for peer review; essential for broad cross-market/generalization claims.

## OPTIONAL experiments

### O1. Stationary surrogate mechanism check

**Change:** simulate prespecified daily persistent processes (random walk and AR(1), with/without volatility episodes) and aggregate using the actual observation calendar. **Models/split:** persistence, AR(1), small RF; same-length chronological simulations, at least 1,000 independent simulated series. **Metrics/tests:** distribution of state/onset AUC, benchmark skill and aggregation gaps; Monte Carlo intervals. **Output:** empirical diagnostic overlaid on null distributions. **Claim:** persistence/aggregation can generate similar signatures without macro signal. **Failure:** real effects fall well outside these mechanisms. Simulation does not prove the real data-generating process; useful if the paper needs an explanatory mechanism beyond a case study.

### O2. Computational efficiency and seeds

**Change:** five fixed RF seeds and standardized single-thread timing versus AR/ridge/logistic. **Split/models:** same frozen matrix, no selecting the best seed. **Metrics/tests:** error spread, median training/inference time and peak memory on declared hardware; report paired distributions rather than infer across five seeds as if independent data. **Output:** accuracy–compute table. **Claim:** complexity costs exceed measured gains in this implementation. **Failure:** complex models offer repeatable gains. Energy claims require measurements; latency/memory alone are not energy consumption.

### O3. Lead–lag and rolling-fit diagnostics

**Change:** prespecify lag order and multivariate controls; calculate lags on the original calendar before excluding episodes; correct multiple testing. Compare rolling-fit alarms with a matched persistence/volatility alarm and all false alarms. **Models/split:** small VAR/ARX and predeclared rolling regressions, same training/evaluation boundaries. **Metrics/tests:** lag coefficients, adjusted tests, OOS macro forecast loss and event alarms. **Output:** descriptive appendix. **Claim:** predictive precedence under a stated design. **Failure:** dependence correction, calendar preservation or episode removal erases it. This should not displace the core experiments: the broad economic direction is already known.

## Completion gates and minimum workload

The **minimum** is E0–E4, with no new external dataset and only simple additional controls. Reuse the existing data, RF, linear models, daily source, split infrastructure and plots; correct evaluation assumptions before reusing the current helper defaults. Five experiment bundles are not five isolated model fits: E1–E3 share a single forecast ledger and E4 mainly analyzes it.

An experienced researcher could plausibly need 3–5 focused days for a narrow reliable harness, runs and first interpretation; a polished submission and literature follow-up may take longer. True-vintage acquisition and new external validation are additional, uncertain effort. These are planning estimates.

Stop expanding models when the core question is answered. If the evidence only shows that one buggy pipeline fails, the output is a useful technical audit. To justify a research paper, it should quantify a consequential evaluation effect across reasonable specifications, explain why it occurs, disclose uncertainty, and position that finding precisely against the closest literature. If all comparisons are inconclusive, report a precision-limited case study or gather more data; do not manufacture a stronger novelty claim.

## Reproducibility contract

Write one long-form forecast ledger with origin, cutoff, target end, data vintage, target definition, threshold, model, feature set, regime procedure, refit ID, training dates, tuning settings, seed, prediction and actual. Generate every table/plot from it. Hash inputs; preserve original notebooks read-only; record Python/package versions and hardware; provide a single command from a clean environment. Unit checks must cover label maturity, unknown labels, train-only scaling/clustering, identical score dates, source reconstruction and units. A future-data perturbation must not change an earlier causal prediction. Expose failed fits and missing forecasts; do not silently change the scoring sample per model. Check data-sharing permissions before promising redistribution; publish access instructions and hashes if raw data cannot be redistributed.

Methodological starting points: [Hewamalage et al., forecast evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/), and [Clark–West, nested predictive comparisons, author working paper](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf). The proposed block lengths and practical margins are study-design choices requiring justification, not prescriptions established by these citations.
