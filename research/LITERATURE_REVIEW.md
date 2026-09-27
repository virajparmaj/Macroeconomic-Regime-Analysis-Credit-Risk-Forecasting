# Literature review and novelty boundaries

Search date: 25 September 2026. Companion to `RESEARCH_ADVISORY.md` and `EXPERIMENT_PROTOCOL.md` in this directory.

**Verdict:** the original combination of macro indicators, clustering/regimes, and credit-spread prediction is already well represented. No exact duplicate of the proposed *joint evaluation of persistence benchmarks, information timing, aggregation, and state versus onset labels* was identified in the material accessible here. That is a provisional gap, not proof of priority.

## Search coverage and limits

Searched for credit-spread forecasting, high-yield OAS, regime switching, macroeconomic determinants, random-walk benchmarks, stress onset, forecast evaluation, leakage, and the exact FRED identifier BAMLH0A0HYM2. Recent work from 2021–2026 was prioritized; older work was retained when it directly pre-empts a proposed contribution.

Sources consulted include arXiv full text; publisher and author/university records; Federal Reserve and Bank of Canada papers; NeurIPS proceedings; and conference-hosted papers. Google Scholar and Semantic Scholar were explicitly attempted through both domain-targeted searches and their search pages. Their search pages did not return usable content in this environment. Consequently this is **not an exhaustive search of those two indexes**, and citation counts were not used. Before claiming priority, repeat exact-title and forward-citation searches there manually, particularly for papers 1–6 below.

Evidence grades: **F** = substantive full text inspected; **P** = publisher/author abstract or accessible preview inspected; **E** = indexed full-text excerpts additionally available. A statement that an issue was “not examined” below means outside the inspected study's stated design; where the full text was unavailable, an omission is explicitly marked unverified. Numerical results from different targets, samples, or metrics must not be compared as a leaderboard.

## Closest competing work

### 1. Erlwein-Sayer, Grimm, Pieper and Alsaç — *Forecasting corporate credit spreads: Regime-switching in LSTM* (online 2023)

**Evidence: P.** [Publisher preview](https://www.sciencedirect.com/science/article/pii/S2452306223000916); [author institution record](https://www.htw-berlin.de/forschung/online-forschungskatalog/publikationen/publikation?eid=15356). DOI: 10.1016/j.ecosta.2023.12.002.

- **Question/method:** do filtered latent market states improve one-step spread forecasts? A discretized Ornstein–Uhlenbeck HMM supplies state probabilities to an LSTM; comparisons include a plain LSTM and HMM forecasts.
- **Data/results:** European corporate spreads from three countries, 2004–2019, plus simulation. Adding regime information improves errors, especially during stronger fluctuations. No numerical effect is quoted because the accessible preview does not establish it.
- **Overlap:** very close to the project's original regime-plus-sequence-model framing. That combination is not new.
- **Boundary/gap:** this is daily European corporate forecasting rather than the proposed monthly US aggregate persistence/onset experiment. Whether every proposed timing and benchmark control appears in the full paper remains unverified. A new contribution must quantify an evaluation effect, not merely substitute K-means or a US series.

### 2. Shao, Bai, Hou, Zhou and Pan — *A Novel Methodology in Credit Spread Prediction Based on Ensemble Learning and Feature Selection* (2024)

**Evidence: F.** [arXiv full text, v1](https://arxiv.org/html/2412.09769v1).

- **Question/method:** can mutual-information feature selection and stacking improve monthly credit-spread estimates? MLP, random forest and nearest-neighbor predictions feed kernel ridge; features include macro variables and recent spread averages, with PCA whitening.
- **Data/results:** 120 monthly observations, January 2008–December 2017; chronological 70/30 split. Table IV reports selected-feature stacking test R² 0.920 and MSE 0.062. The exact spread-series identifier is insufficiently specified in the inspected text; it must not be assumed to be the project's HY OAS.
- **Overlap:** the closest small-sample, public-macro, monthly-ML comparison.
- **Boundary/gap:** the presented evaluation lacks a persistence benchmark, vintage reconstruction and stress-onset assessment. Your opportunity is a transparent benchmark-controlled study. This inspection does **not** establish that their implementation leaks or that their results would reverse. A formal replication requires their precise data and code.

### 3. Heger, Min and Zagst — *Analyzing credit spread changes using explainable artificial intelligence* (2024)

**Evidence: P/E.** [Publisher record](https://www.sciencedirect.com/science/article/pii/S1057521924002473); [author-university repository](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/year/2024/docId/112665). DOI: 10.1016/j.irfa.2024.103315.

- **Question/method:** which nonlinearities and interactions explain spread changes? Linear/local polynomial regression and ML are analyzed using partial-dependence plots, H-statistics and SHAP decomposition.
- **Data/results:** monthly US and Euro-area corporate and covered-bond changes across maturity buckets. The paper attributes ML advantages mainly to nonlinearities rather than interactions, and finds crisis-specific explanatory differences. The retrieved full-text excerpt reports average OOS RMSE 0.132 for local polynomial regression and 0.133 for RF. Exact sample dates and complete temporal alignment were not verified.
- **Overlap:** strongly pre-empts “use RF/SHAP to discover macro drivers” and generic crisis-dependent feature importance.
- **Boundary/gap:** explaining contemporaneous changes and forecasting changes from previously available releases are different estimands. The proposed study makes this separation explicit. Do not allege an evaluation defect in this paper without reviewing the full setup. Data redistribution is restricted according to its availability statement.

### 4. Boursicot, Gauthier and Djikeng Nguetsa — *Enhancing Credit Spread Forecasts using Macroeconomic Uncertainty Variables and Statistical Learning Approaches* (2025; revised March 2026)

**Evidence: P.** [Author-deposited SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5338990). DOI: 10.2139/ssrn.5338990.

- **Question/method:** can uncertainty variables improve spread term-structure prediction? The study estimates Nelson–Siegel parameters with a new penalty, then compares linear and ensemble predictors and uses SHAP.
- **Data/results:** firm-level spread curves with macroeconomic/financial uncertainty measures; boosted trees are reported to perform best. Sample dates, issuer count and numerical margins could not be verified from the accessible abstract.
- **Overlap:** closely related to macro-plus-ML spread forecasting; adding a generic uncertainty or explainability component alone is a weak novelty claim.
- **Boundary/gap:** firm-level curves differ from a single aggregate OAS and onset classification. The abstract does not establish whether it evaluates all timing/aggregation controls. Treat it as an important full-text follow-up, not as a paper whose omissions are settled.

### 5. Anderson and Audzeyeva — *Using diverse local optima for setting kernel parameters in support vector regression: Forecasting emerging market credit spreads* (2026)

**Evidence: P.** [Publisher](https://doi.org/10.1016/j.ijforecast.2026.03.004); [author repository](https://keele-repository.worktribe.com/output/1662595/using-diverse-local-optima-for-setting-kernel-parameters-in-support-vector-regression-forecasting-emerging-market-credit-spreads).

- **Question/method:** can sets of good SVR parameter solutions outperform selection of one optimum? The framework combines promising models and uses out-of-sample forecast comparisons/model-confidence methods.
- **Data/results:** sovereign spreads for Brazil, Mexico, Turkey and the Philippines. Forecast combinations outperform RF, standard SVR and conventional benchmarks in the reported application; exact numerical margins were not verified.
- **Overlap:** rigorous nonlinear credit forecasting and model comparison. Its [2019 Federal Reserve predecessor](https://www.federalreserve.gov/econres/feds/a-coherent-framework-for-predicting-emerging-market-credit-spreads-with-support-vector-regression.htm) explicitly compares against random walks and uses serial-dependence-aware validation.
- **Boundary/gap:** sovereign spread curves differ from aggregate HY OAS. Your negative result cannot establish that ML never beats persistence, nor can persistence benchmarking itself be claimed as novel. The contribution must concern specific evaluation conditions or the state/onset distinction.

### 6. Maalaoui Chun, Dionne and François — *Credit spread changes within switching regimes* (2014; accessible 2009 working version)

**Evidence: F for the 2009 version.** [Conference-hosted full paper](https://www.bankofcanada.ca/wp-content/uploads/2010/09/maalaoui.pdf); published version DOI 10.1016/j.jbankfin.2014.08.009. Do not assume numerical identity between versions.

- **Question/method:** does conditioning on endogenous credit regimes explain spread determinants better than a pooled or NBER-cycle specification? Markov regimes are inferred from credit spreads and then related to explanatory variables.
- **Data/results:** the working version analyzes rating/maturity-specific US corporate spreads, including 1994–2004 AA–BB series. It reports much greater explanatory power under credit regimes, averaging approximately 60% adjusted R² for 10-year series, with some coefficient sign reversals.
- **Overlap:** directly pre-empts “macro relationships differ in stress” as a standalone discovery.
- **Boundary/gap:** explaining regime-conditional changes is not the proposed common-origin forecast and stress-entry test. A comparison of descriptive regime fit against incremental forecast skill remains useful if it is measured under strict information timing.

### 7. Maalaoui Chun, Dionne and François — *Detecting Regime Shifts in Credit Spreads* (2014)

**Evidence: P/E.** [Publisher](https://www.cambridge.org/core/product/identifier/S0022109015000034/type/journal_article); [author working paper](https://chairegestiondesrisques.hec.ca/wp-content/uploads/pdf/cahiers-recherche/08-02.pdf).

- **Question/method:** are level and volatility regimes distinct? Sequential regime-shift detection separates mean shifts from variance shifts.
- **Data/results:** working-paper evidence combines Warga quotes (1987–1996), NAIC transactions (1994–2004) and TRACE transactions (2004–2009). Level regimes persist longer and relate to monetary/credit conditions; volatility regimes are shorter and also appear outside recessions.
- **Overlap:** directly anticipates the distinction between sustained high spreads and rapid repricing. Simply pointing to March 2020's range does not create a new economic concept.
- **Boundary/gap:** the proposed contribution is how those different event definitions change measured predictive performance and false alarms, rather than discovering that level and volatility are different.

### 8. Gilchrist and Zakrajšek — *Credit Spreads and Business Cycle Fluctuations* (2012)

**Evidence: P.** [American Economic Association](https://pubs.aeaweb.org/doi/10.1257/aer.102.4.1692); [NBER author paper](https://www.nber.org/papers/w17021).

- **Question/method:** what information about future economic activity is contained in credit spreads? Bond-level prices form an index decomposed into expected-default and excess-bond-premium components.
- **Data/results:** a US corporate-bond microdata panel and macroeconomic outcomes; predictive content is especially associated with the excess bond premium. Exact sample counts were not reverified here.
- **Overlap:** the proposed reverse-direction story—spreads lead macroeconomic activity—is firmly established.
- **Boundary/gap:** your aggregate OAS cannot identify their excess bond premium or their structural mechanism. A six-variable Granger exercise would be a replication-style illustration, not a new discovery. Keep it as context unless a new timing/measurement comparison changes the conclusion.

### 9. Faust, Gilchrist, Wright and Zakrajšek — *Credit Spreads as Predictors of Real-Time Economic Activity: A Bayesian Model-Averaging Approach* (2012 working paper; later journal publication)

**Evidence: F/P.** [Federal Reserve full-text record](https://www.federalreserve.gov/pubs/feds/2012/201277/).

- **Question/method:** do credit spreads improve real-time macro forecasts? Bayesian model averaging compares financial predictors against autoregressive forecasts.
- **Data/results:** real-time economic-activity measures and corporate-spread portfolios sorted by maturity/risk. Improvements extend from the current quarter to four quarters ahead and are attributed largely to spreads.
- **Overlap:** both publication timing and the reverse prediction direction already have strong precedents.
- **Boundary/gap:** the outcome is activity, not future HY OAS or credit-stress entry. Your fixed publication shifts on revised data are weaker evidence than their real-time design. A paper must call the existing treatment a publication-lag approximation, not vintage-correct real-time forecasting.

## Methodological and benchmark literature

### 10. Hewamalage, Ackermann and Bergmeir — *Forecast evaluation for data scientists: common pitfalls and best practices* (2023 journal; 2022 preprint)

**Evidence: F/P.** [Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/); [arXiv](https://arxiv.org/abs/2203.10716).

**Question/method/data/result:** a methodological synthesis using forecast-evaluation examples, rather than a credit dataset; addresses partitions, nonstationarity, metrics, competitive benchmarks and inference. **Overlap:** establishes the evaluation principles your corrected study should follow. **Not its question:** their empirical size and interaction in this particular credit task. **Possible contribution:** quantify those effects and publish a reproducible domain-specific case study. A checklist alone is not new methodology.

### 11. Kapoor and Narayanan — *Leakage and the reproducibility crisis in machine-learning-based science* (2023)

**Evidence: P.** [Author institution record](https://collaborate.princeton.edu/en/publications/leakage-and-the-reproducibility-crisis-in-machine-learning-based-/); [published article](https://doi.org/10.1016/j.patter.2023.100804).

**Question/method/data/result:** cross-field leakage taxonomy and a civil-war-prediction reproducibility study; the published record describes 294 affected papers across 17 fields, and complex-model advantages diminish after correction in the replication. **Overlap:** directly relevant to target reconstruction and preprocessing leakage. **Not its question:** aggregate-credit aggregation and persistence. **Possible contribution:** a transparent controlled credit study, with conclusions proportional to its single-market scope. Discovering bugs in your own unpublished code alone is unlikely to meet a substantial novelty threshold.

### 12. Godahewa et al. — *Monash Time Series Forecasting Archive* (NeurIPS Datasets and Benchmarks, 2021)

**Evidence: F/P.** [Conference paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/eddea82ad2755b24c4e168c5fc2ebd40-Paper-round2.pdf).

**Question/method/data/result:** standardized comparative forecasting across 25 datasets, heterogeneous frequencies and missingness, with baseline results under ten metrics. **Overlap:** reproducible forecasting benchmarks and transparent baseline comparisons. **Not its question:** credit-specific release calendars or onset versus persistence. **Possible contribution:** a small domain-specific evaluation protocol, not a general forecasting benchmark or evidence of globally superior models.

### 13. Aksu et al. — *GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation* (2024 arXiv version)

**Evidence: P.** [Versioned arXiv record](https://arxiv.org/abs/2410.10393v2).

**Question/method/data/result:** evaluates statistical, deep and foundation models across domains, frequencies and horizons while separating pretraining/evaluation data. This version describes 23 datasets and 17 baselines; later records differ, so version-lock any counts. **Overlap:** benchmark hygiene and multiple forecast horizons. **Not its question:** the economics of an aggregate spread label and its available information. **Possible contribution:** a focused test of incremental information, without adding foundation models solely for fashion.

## What this literature rules out

Do not claim to introduce regime switching for credit spreads, machine learning for spread prediction, macroeconomic determinants, state-dependent relationships, spread-led macro prediction, SHAP for spreads, random-walk benchmarking, leakage detection, or a general time-series benchmark.

The remaining plausible contribution is **an empirical identification of how forecast target, latest available spread information, and retrospective regime construction change conclusions about incremental macroeconomic skill and early warning**. “Identification” here concerns controlled computational comparisons, not an identified causal economic mechanism.

Two claims require additional work before submission: (1) that no prior study performs this exact evaluation; (2) that the proposed effects are stable rather than artifacts of one small sample. Access the full texts of papers 1, 3, 4 and 5, finish citation tracing, and complete the paired experiments before using language stronger than “we examine an under-tested evaluation question.”
