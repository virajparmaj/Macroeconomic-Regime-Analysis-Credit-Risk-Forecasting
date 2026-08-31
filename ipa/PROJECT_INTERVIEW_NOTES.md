# IPA Data Analyst Fellow — Project Interview Notes

> Use the current analytical review as the source of truth. The original notebooks are useful as an audit trail, but their headline Random Forest result is not a valid forecast result because the feature set leaked the contemporaneous target. The strongest interview story is that I found the problem, rebuilt the evaluation, and changed the conclusion.

## 1. The project in 20 seconds

I wanted to know whether broad economic conditions could identify changes in market credit risk. I combined six monthly macro indicators with the ICE BofA high-yield credit spread, analyzed regimes, timing, and forecast performance, and found that macro data describes stress much better than calm periods—but usually moves after the spread. That matters because a useful risk analysis needs correct timing, honest benchmarks, and claims that match what the data can support.

## 2. The project in 60–90 seconds

I started with a simple question: can macroeconomic conditions help explain or forecast changes in credit-market risk? I built a monthly panel from December 1996 through August 2022 using CPI, the Fed Funds rate, industrial production, an OECD Euro Area GDP reference series, unemployment, consumer sentiment, and the ICE BofA high-yield option-adjusted spread as the outcome.

The first version used feature engineering, regime clustering, and several forecasting models. When I audited it, I found that the original strong Random Forest result was inflated by target leakage and did not use a serious persistence benchmark. I rebuilt the analysis around publication timing, first differences, walk-forward testing, and a random-walk benchmark.

Three findings stood out. First, macro changes explained about 7% of spread changes in calm periods versus 50% in macro-stress periods. Second, the spread generally moved before the slower macro releases. Third, even the best corrected point-forecast model had 82% higher RMSE than simply assuming next month's spread would equal this month's. The practical value was therefore not a production-ready point forecast; it was a clearer early-warning and monitoring framework, plus a strong lesson in data timing, segmentation, and honest evaluation.

## 3. Know these facts

### Data and scope

- **Primary panel:** 309 monthly observations, December 1996–August 2022.
- **Variables:** six macro indicators plus `Credit_Spread`—seven numeric series after the date index.
- **Macro indicators:** CPI, Fed Funds, industrial production, Euro Area 19 OECD GDP reference growth, unemployment, and consumer sentiment.
- **Outcome:** monthly mean of the ICE BofA US High Yield Index Option-Adjusted Spread (`BAMLH0A0HYM2`), measured in percentage points.
- **Important scope:** this is market credit **pricing**, not borrower default or realized credit loss.
- **Daily source:** 6,762 rows with 83 missing spread entries; 6,679 valid daily observations. The model used 309 monthly means.
- **Final panel quality:** no missing values, duplicate rows, duplicate dates, or missing months.
- **Naming caveat:** the original `GDP` column is `EA19LORSGPORGYSAM`, an OECD Euro Area 19 reference series—not US GDP.

### Evaluation and results

- **Spread persistence:** lag-1 autocorrelation = **0.963**.
- **Walk-forward design:** 14 expanding origins, 163 scored months from 2009–2022, annual refits, three-month embargo, point-in-time macro alignment.
- **Best honest benchmark:** random-walk RMSE = **0.6124 pp**.
- **Random Forest with macro + spread history:** RMSE = **1.1131 pp**, or **81.7% higher** than the random walk; ordinary R² = **0.7503**, but R² versus the random walk = **−2.3037**. Its Diebold–Mariano p-value versus the random walk is **0.0561**, so describe the error as worse but do not call the difference significant at 5%.
- **Regime result:** joint in-sample R² of monthly spread changes on six macro changes = **0.0743 in calm months vs 0.4987 in macro-stress months**—a **6.71×** difference.
- **External regime check:** the macro-only stress index flagged **22 of 23** NBER recession months within its usable window.
- **Timing result:** spread → unemployment Granger p-value **<0.0001**; unemployment → spread p-value **0.1078**. This is predictive precedence, not causation.
- **Short-horizon stress ranking:** spread-only AUC = **0.935 / 0.793 / 0.656 / 0.543** at 1 / 3 / 6 / 12 months.
- **Aggregation result:** March 2020's daily range was **6.12 pp**, **13.6×** the median month's range; it ranked **1st of 309** by intra-month range but only **43rd** by monthly mean.
- **Original result to discuss only as an audit finding:** the first notebook reported R² = **0.8084**, but three of its features could reconstruct the contemporaneous spread exactly; three-column OLS produced R² = **1.0**. Do not present 0.8084 as valid forecast performance.

## 4. Three strongest data insights

### Insight 1 — Macro is much more informative during stress

**Finding:** The relationship between macro changes and spread changes is regime-dependent rather than stable across all months.

**Evidence:** In the macro-only regime split, joint in-sample R² was **0.0743 in 187 calm months** and **0.4987 in 63 stress months**. Separately, high-spread months were about **25% of 308 change observations** but contained **83.2% of spread-change variation**.

**Why it matters:** A pooled average hides that most useful macro information is concentrated in unusual periods. Segment-level KPIs are more informative than one headline metric.

**Possible implication:** Report calm and stress performance separately and monitor the regime indicator as context for escalation or investigation.

**Caveat:** These are contemporaneous, in-sample relationships. They do not prove that macro variables cause spread changes or that the 75th-percentile split is a natural boundary.

### Insight 2 — The market spread usually moves before the published macro data

**Finding:** The direction of predictive information generally ran from the spread toward later macro deterioration, not the other way around.

**Evidence:** Mean absolute cross-correlation peaked at **k = −1 month**, after the spread had moved. Spread changes predicted unemployment changes at **p < 0.0001**, while the reverse test was **p = 0.1078**. Macro led the spread for **0 of 6 indicators in both sample halves**.

**Why it matters:** Data availability is part of the analysis. A daily, forward-looking market price should not be expected to wait for backward-looking statistics released weeks later.

**Possible implication:** Use the spread as an early-warning input for macro monitoring, or add faster financial variables if the goal remains forecasting the spread.

**Caveat:** Granger precedence does not establish causality. Both series may respond to the same unobserved shock, and the recession-warning comparison contains only three recessions.

### Insight 3 — Monthly averaging hid the speed of the COVID shock

**Finding:** A monthly mean compressed the most violent credit repricing in the sample.

**Evidence:** In March 2020, the spread moved from **4.75 pp to 10.87 pp**, a **6.12 pp** range. That month ranked first by daily range but only 43rd by monthly mean; the mean of **7.85 pp** was **27.7% below** the peak.

**Why it matters:** Aggregation is an analytical choice, not neutral housekeeping. The monthly mean preserved broad level but discarded shock velocity.

**Possible implication:** Retain month-end, intra-month range, and realized volatility alongside the monthly mean, especially for monitoring tail events.

**Caveat:** Daily spreads are noisier, while the macro predictors remain monthly. This improves the outcome definition and monitoring resolution; it does not fix macro publication lags.

## 5. Why did I do it this way?

| Likely challenge | Short defensible answer |
|---|---|
| **Why these variables?** | They cover inflation, monetary policy, production, labor conditions, growth, and sentiment—the major dimensions of the economic cycle. I would now relabel the OECD series clearly and test whether a US-specific measure is more appropriate. |
| **Why use the high-yield spread?** | It is a continuous, observable market measure of priced credit stress with a long public history. It is not a default label, so I restrict the claim to market-level credit conditions. |
| **Why monthly data?** | Monthly frequency matches most macro releases and avoids pretending slow data are daily. The later audit showed that the mean alone hides fast events, so I would preserve daily summary features too. |
| **Why first differences?** | ADF/KPSS diagnostics identified four of seven series as non-stationary in levels. Differences reduce spurious level relationships and raise the effective information content of the series. |
| **Why a macro-only stress index?** | Splitting on the spread itself would condition on the outcome and make the result circular. The index used unemployment deterioration, negative industrial-production growth, weak sentiment, and the inverted OECD growth series, then was checked against NBER recession dates. |
| **Why the top quartile?** | It is simple, interpretable, and leaves enough stress observations for comparison. It is still a modeling choice; in production the threshold must be estimated on training data and sensitivity-tested. |
| **Why PCA and clustering originally?** | The macro variables were correlated and regimes were unlabeled, so PCA summarized common variation and clustering explored latent states. The audit showed the original 10-cluster result was too fragile for final forecast claims, so I treat it as exploration rather than the source of the strongest conclusions. |
| **Why Random Forest?** | It can capture nonlinear thresholds and interactions without imposing a linear functional form. I also tested a regularized linear model; neither beat persistence, which is more important than defending a favored algorithm. |
| **Why a random-walk benchmark?** | With lag-1 autocorrelation of 0.963, “next month equals this month” is the minimum credible bar. Plain R² compares against the unknowable test-period mean and made the model look much better than it was. |
| **Why walk-forward validation?** | It preserves time order and simulates repeated historical deployment. Expanding folds, publication lags, annual refits, and an embargo are more defensible than one lucky split or random cross-validation. |
| **Why classification after regression failed?** | “Will stress exceed a meaningful threshold?” is often more actionable than an unreliable exact point forecast. AUC and precision lift showed useful ranking at one to three months, but not at twelve. |
| **Why not keep adding complexity?** | The simpler persistence model won. More complexity is justified only if it improves out-of-time performance or provides a clear decision benefit. |

## 6. Data quality and cleaning

- **Missing values:** The raw daily spread file contained 83 missing entries; the original data notebook forward-filled them before monthly aggregation. It also calculated an interpolated alternative but chose forward fill. The final 309-month panel has no missing values. Engineered lag/rolling/forward-target rows were dropped only when the required history or future target was unavailable.
- **Duplicates and time grid:** The final panel has zero duplicate rows, zero duplicate dates, and all 309 expected month-end dates. The pipeline verified this; it did not need to remove duplicate panel rows.
- **Outliers:** The actual clustering and current analysis did **not** remove or winsorize extreme values. Crisis observations were treated as meaningful signal, not presumed errors.
- **Inconsistent data:** Macro dates and credit dates were standardized to month-end before an outer merge and chronological sort. The later audit corrected the misleading `GDP` label and documented source IDs and publication lags.
- **Transformations:** Daily spread values were aggregated to monthly mean; month-end was also saved. Current statistical tests use first differences where stationarity required it. Forecast features use lags, shifted rolling windows, changes, and publication-lag alignment.
- **Leakage controls:** The corrected rolling features exclude the current row, raw-target interactions are rejected, the forward target's exact name is returned to the caller, and a leakage report tests the finished feature matrix.
- **Validation:** Results were checked with point-in-time alignment, sample-half and crisis-exclusion tests, walk-forward folds, a named random-walk benchmark, alternative model families, NBER recession dates, and automated feature/split tests. At review time, **28 tests passed**.

**How did I know the data/results were reliable enough to analyze?**

I reconciled the merged panel back to the raw source files, verified completeness and time ordering, and reran the main analysis scripts. I then tested whether the conclusions survived more honest timing and benchmarking. “Reliable enough” here means suitable for exploratory historical analysis with explicit caveats—not production validation or causal inference.

## 7. Theory I must understand

1. **Correlation versus causation:** Correlation shows variables move together; it does not show one produces the other. Even Granger tests only show whether past values improve prediction, so the spread may move first without causing unemployment or sentiment to change.

2. **Stationarity and first differences:** A stationary series has a reasonably stable distribution over time. Trending levels can create misleading correlations, so the current analysis generally studies month-to-month changes after ADF/KPSS checks.

3. **Autocorrelation and the random walk:** Autocorrelation measures how much a series resembles its recent past. At 0.963 lag-1 autocorrelation, today's spread is already a powerful forecast of next month's, so any model must beat persistence rather than a flat historical mean.

4. **Plain R² versus out-of-sample R²:** Plain test R² compares errors with the test-period mean, which was unknown when forecasting. Campbell–Thompson out-of-sample R² compares directly with a usable benchmark; negative values mean the model loses to that benchmark.

5. **Walk-forward validation and leakage:** Time-series validation must train only on the past and score later periods. Leakage occurred originally because rolling features included the current target; publication timing is a second form of look-ahead if a model uses data before it was released.

6. **PCA, multicollinearity, and clustering:** PCA compresses correlated variables into a few combined dimensions; clustering then groups similar periods without labeled outcomes. Regime names are interpretations of centroids, not facts, and clusters can change when the sample changes.

7. **AUC, average precision, and base rate:** AUC measures ranking across thresholds, while average precision is more informative when stress cases are uncommon. Lift divides average precision by the positive base rate; neither metric proves the predicted probabilities are calibrated.

8. **Aggregation and measurement choice:** Mean, month-end, maximum, and intra-month range answer different questions. A mean is stable for monthly alignment but can hide the speed and peak of a shock.

## 8. Visualization and KPI defense

| Metric / graph | Why this metric? | Why this visualization? | What another view could hide |
|---|---|---|---|
| **Random-walk RMSE and R² vs random walk — Figure 4** | RMSE is in spread percentage points; R² versus a named benchmark answers whether the model adds forecast value. | Side-by-side bars expose how the same model can look good against the mean and bad against persistence; the time line shows the false alarm and lagging behavior. | Plain R² alone hides the benchmark. A single average error hides when the model fails. |
| **Calm/stress correlations and joint R² — Figure 1** | Conditional metrics test whether the relationship is stable across regimes. | Paired bars make the change for every indicator visible, and the variance-share bar shows concentration. | A pooled correlation hides that calm and stress months behave differently. |
| **Lead-lag cross-correlation and recession warnings — Figure 2** | Lead-lag measures timing, which is central when sources have different release schedules. | The directional axis separates “spread first” from “macro first”; grouped bars translate timing into months of warning. | A contemporaneous correlation cannot show which series moved first. |
| **AUC and precision lift by horizon — Figure 5** | AUC measures ranking; precision lift accounts for a changing positive base rate. | A horizon line shows signal decay and the bar chart shows practical gain over guessing. | One-horizon accuracy could hide class imbalance and the rapid loss of value after three months. |
| **Daily spread with monthly mean and range ranks — Figure 6** | Range measures repricing speed, while the mean measures monthly level. | Overlaying daily movement with flat monthly means makes lost information obvious. | A monthly line alone makes March 2020 look much less exceptional than it was. |

**If I could show only one visualization:** Figure 4, **“The model that looked like R² = 0.75 is 82% worse than assuming nothing changes.”** It communicates the most important analytical lesson in one view: define the decision, use a credible benchmark, inspect behavior over time, and do not let a flattering metric substitute for usefulness.

## 9. Limitations

### What the analysis supports

- Historical association between macro conditions and high-yield spread changes, especially during stress.
- Evidence that the spread generally moved before slower macro releases in this sample.
- Evidence that the tested point-forecast models did not beat persistence, plus promising short-horizon stress ranking that merits further validation.
- Evidence that monthly averaging compressed intra-month tail behavior.

### What I should not claim

- **Causality:** There is no exogenous design proving macro variables cause spread moves, or that spreads cause later economic changes.
- **Broad generalization:** The sample contains only three NBER recessions; 309 months are not 309 independent observations. Thresholds, AUCs, and regime relationships may shift in new cycles.
- **Borrower or policy impact:** The outcome is a market spread, not defaults, losses, customers, electricity prices, or program outcomes. It cannot directly justify a lending, investment, or public-policy action.

The three biggest practical limitations are: **few independent crisis episodes**, **final revised and sometimes slow/misaligned macro data**, and **a target/aggregation choice that measures market pricing while compressing fast shocks**. The mislabeled Euro Area series reinforces the provenance risk.

## 10. What I would do next

1. **Rebuild a true real-time panel.** Use historical data vintages rather than final revised values, replace or deliberately justify the Euro Area series, and add faster financial indicators such as the VIX, yield-curve measures, or a financial-conditions index. Success criterion: beat the random walk on the same walk-forward test.

2. **Test the direction the data supports.** Reverse the problem and forecast unemployment or industrial production from spread dynamics, comparing against each macro series' own autoregressive benchmark. This directly tests whether the observed predictive precedence creates useful early warning.

3. **Improve and externally validate the stress signal.** Add month-end, intra-month range, and realized volatility; estimate thresholds on training data only; then test the calm/stress result on a longer spread history such as the Moody's Baa–Aaa series. That addresses both aggregation loss and the three-recession limitation.

## 11. IPA / public-sector connection

1. **Data provenance and timing transfer directly.** Public-sector analysis often combines administrative or published series with different definitions, release schedules, and revisions. This project taught me to document sources, align information to when it was actually available, and challenge misleading labels before modeling.

2. **The KPI and segmentation discipline transfers.** I moved from one flattering average metric to named benchmarks, regime-specific reporting, and threshold sensitivity. In an IPA DAU role, the domain would change, but the habit is the same: define a KPI that matches the decision, segment heterogeneous conditions, and make clear what the number does and does not mean.

3. **Communication under uncertainty transfers.** The final charts turn technical checks into plain findings for nontechnical stakeholders and keep caveats visible. The credible connection is not credit-risk expertise; it is the ability to investigate a complex dataset, catch a validity problem, revise the conclusion, and communicate useful evidence without overstating certainty.
