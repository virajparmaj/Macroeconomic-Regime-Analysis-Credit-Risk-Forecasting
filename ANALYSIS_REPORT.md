# Macroeconomic Regimes & Credit Risk — Analytical Review

**Scope:** analysis side only. No website changes.
**Data:** `data/merged_macroeconomic_credit.csv` (309 monthly rows, Dec 1996 – Aug 2022) plus the previously unused daily source `data/original/ICE BofA…BAMLH0A0HYM2.csv` (6,679 daily observations).
**Figures:** `results/figures/` · **Numbers:** `results/key_numbers.csv`

---

## 0. What this project is actually asking, and what the data can answer

**The question as built:** can six monthly macro indicators — CPI, Fed Funds, industrial production, an OECD GDP reference series, unemployment and consumer sentiment — predict the ICE BofA US High Yield option-adjusted spread, used as a market-level proxy for credit risk?

**What the dataset can establish**
- How credit-market stress co-moves with the macro cycle, across four distinct episodes (2001–02, 2008–09, 2011–12, 2020).
- Whether those co-movements are *stable* — the regime question — and where they break.
- Which series carries information first, in *predictive precedence* terms.

**What it cannot establish**
- **Borrower-level default.** There are no loans, applicants, balances or delinquency labels. The spread is what lenders *charge* for risk, not what they *lose*.
- **Causation.** Six correlated aggregates over 309 months, with no exogenous variation, support association and precedence — nothing stronger.
- **Anything about fast events.** The panel is a monthly *mean* of a daily price (Insight 6).
- **Reliable inference from crisis observations.** There are three recessions, and that — not the row count — is the binding constraint. The spread's *level* has lag-1 autocorrelation 0.963, so 309 months carry roughly **6** independent observations of the level; differencing recovers most of it (~149), which is why every test below runs on changes rather than levels. But no amount of differencing creates a fourth crisis.

**What matters most:** `Credit_Spread` is the outcome. Among predictors, the finding below is that the spread's *own* recent dynamics dominate all six macro series combined.

**A note on provenance that changes interpretation:** the column named `GDP` is `EA19LORSGPORGYSAM` — the OECD leading-indicator reference series for **Euro Area 19** GDP, interpolated monthly from quarterly data, not US GDP. It is among the strongest correlates of the US high-yield spread in the panel, so the label matters for any economic reading. This was already flagged in `src/config_v2.py`; I use the corrected name throughout.

---

## Insight 1 — Macro data explains credit spreads only when the economy is already under stress. It is a switch, not a dial.

**Question.** The project treats the macro→spread relationship as a single fixed relationship estimated over the whole sample. Is it?

**Finding.** No. Splitting months by a stress index built **from macro indicators only** — no spread information — the six indicators jointly explain 7% of monthly spread changes in calm months and 50% in stress months. Every individual indicator strengthens. The full-sample correlation is an average of two regimes that barely resemble each other: CPI's correlation with spread changes is −0.04 in calm months and −0.49 in stress.

**Key numbers**
- **R² = 0.07 calm vs 0.50 stress — a 6.7× difference.**
- **High-spread months are 25% of the sample (78 of 308) but carry 83% of all spread variation.**
- The macro-only stress index flags **22 of the 23 NBER recession months inside its window** (96%). The window opens in Nov 2001 because the index needs a 60-month lookback, so it excludes most of the 2001 recession. The split is externally validated rather than fitted.

**Chart.** `results/figures/01_macro_is_a_switch_not_a_dial.png`

**What the chart shows.** Left: |correlation| with monthly spread changes, calm vs stress, per indicator — every bar grows, Fed Funds most (0.06 → 0.58). Centre: joint R², 0.07 vs 0.50. Right: the concentration — a quarter of the months, 83% of the variance.

**Why it matters.** A single pooled model is fitted mostly on months where its inputs carry almost no information, then judged on an average dominated by months it rarely sees. That is the mechanism behind the project's unstable results, and it reframes what the macro block is for: **state description, not continuous prediction.**

**Possible implication.** Report and monitor calm and stress separately rather than as one headline. Route them to different logic: in calm regimes lean on spread dynamics and treat macro as context; in stress regimes macro becomes genuinely informative and worth weighting. Make the regime label itself the monitored KPI.

**Limitation.** The regime split is drawn at the 75th percentile of a constructed index — a choice, not a natural boundary, though results hold at the 70th and 80th too. This is contemporaneous association: macro and spreads moving together in a crisis does not tell us which is driving which. Insight 2 addresses exactly that.

---

## Insight 2 — The credit spread moves *before* the macro data, not after it. The project's arrow of prediction points the wrong way.

**Question.** The project assumes macro → spread. Does the data support that direction?

**Finding.** It supports the reverse. Across leads from −6 to +6 months, the correlation between macro changes and spread changes peaks at **k = −1** — one month *after* the spread has already moved. Granger tests on first differences: spread changes predict unemployment changes (p < 0.0001) while unemployment changes do not predict spread changes (p = 0.108); the same asymmetry holds for consumer sentiment (p = 0.0009 vs p = 0.205). Splitting the sample in half, **macro leads the spread for 0 of 6 indicators in both halves.**

**Key numbers**
- **Mean |correlation| is 0.097 on the spread-moved-first side vs 0.061 on the macro-moved-first side.**
- Warning before recession starts: **credit spread 12 / 5 / 15 months** (2001 / 2008 / 2020) vs **unemployment 0 / 0 / never**.
- Under real-time publication-lag alignment the asymmetry *strengthens* — spread → macro reaches p < 0.001 for all six.

**Chart.** `results/figures/02_spread_leads_the_macro_data.png`

**What the chart shows.** Left: mean |correlation| by lead, with the mass piled on the spread-first side and a clear spike at k = −1. Right: months of warning before each recession, by signal — unemployment is a flat zero every time.

**Why it matters.** This explains the whole project in one line. The spread is a forward-looking market price set daily by people forecasting the economy; CPI, unemployment and industrial production are backward-looking statistics published weeks after the month they describe. Asking published macro data to predict a market price asks the slow series to anticipate the fast one. It is also *why* Insight 4 happens.

**Possible implication.** Invert the framing: use the spread as a **leading indicator of macro deterioration**, which the data supports, rather than as the thing macro predicts. That is a more defensible product and a more useful one — a credit-market-derived early-warning signal for the labour market and output.

**Limitation.** Granger precedence is predictive, not causal — it says the spread moves first, not that it *causes* the macro to move. Both plausibly respond to a common unobserved driver. Three recessions is a very small base for the warning-lead numbers, and the ±1 s.d. threshold is a reasonable but arbitrary rule. For CPI, Fed Funds and industrial production the relationship is significant in *both* directions rather than cleanly one-way; the honest claim is the asymmetry and the never-reversed direction, not exclusivity.

---

## Insight 3 — The macro model fits worst immediately before a credit crisis and best immediately after one.

**Question.** If the relationship changes over time, *when* does it work — and is that timing useful?

**Finding.** The timing is exactly backwards. The **five worst-fitting months in 26 years are November 2007 – August 2008**, on the eve of the largest credit crisis in the sample. The five best are March 2020 – February 2021, after the COVID shock had already repriced. Fit rises *after* stress arrives and falls *before* it does.

**Key numbers**
- **Rolling 60-month R² bottoms at 0.099 in May 2008 — its lowest in the sample. The spread then widened +13.6pp over the next seven months.**
- It peaks at **0.763 in April 2020**, after the shock. Sample mean is 0.36.
- Rolling R² correlates **+0.33 with the spread nine months *earlier*** and **−0.26 with the spread twelve months later** — the signature of a lagging indicator.

**Chart.** `results/figures/03_fit_collapses_before_crises.png`

**What the chart shows.** Top: rolling 60-month R², with the five worst-fit months in red and five best in teal. Bottom: the spread on the same axis. The red cluster sits in the flat, calm stretch immediately preceding the 2008 spike.

**Why it matters.** It challenges the natural assumption that a well-fitting model is a safe one. Here, deteriorating fit was the closest thing to an advance warning, and improving fit meant the damage was already done. A team monitoring model health in 2008 would have seen R² collapse and most likely concluded the model needed retraining — precisely when the collapse was the signal.

**Possible implication.** Track rolling R² as a **monitored risk KPI with an inverted reading**: a sustained fall below roughly 0.15 is grounds for escalation, not for a retrain ticket. It costs nothing to compute and would have fired in late 2007.

**Limitation.** Two crisis episodes drive this; n = 2 for the core claim. A formal sup-Wald test **cannot reject** parameter stability (statistic 6.11 against a ~17.5 critical value), so this is a cyclical fluctuation in fit, not a proven permanent break — and the 60-month window mechanically inflates R² for years after a crisis enters it. The direction is consistent and economically sensible, but I would not trade on the 0.15 threshold without more episodes.

---

## Insight 4 — The headline result is an artefact of the benchmark. Nothing built here beats "assume nothing changes."

**Question.** The project reports R² ≈ 0.81 for its best model. Is that good?

**Finding.** No. On an honest walk-forward test with point-in-time macro alignment, the random walk — predicting next month's spread with this month's — achieves **R² = 0.92**. The best macro model reaches R² = 0.75, which *looks* respectable against the test mean but is **82% higher error** than doing nothing. The result survives every attempt to rescue it: predicting the *change* so the random walk is nested, horizons of 1/3/6/12 months, Ridge as well as random forest. Macro-only is worse still, and produces real false alarms — in July 2020 it predicted 14.5pp when the actual spread was 5.1pp.

**Key numbers**
- **Random walk RMSE 0.61pp vs best model 1.11pp — the model is 82% worse.**
- **R² against the test mean: +0.75. R² against the random walk: −2.30.** Same predictions, two benchmarks.
- Diebold–Mariano vs the random walk: **p < 0.0001** for macro-only.

**Chart.** `results/figures/04_nothing_beats_the_random_walk.png`

**What the chart shows.** Left: the same forecasts scored two ways, +0.75 and −2.30. Centre: the RMSE ladder — random walk 0.61, macro+spread 1.11, macro-only 2.40. Right: predictions against actuals, where both models visibly *follow* the spread and never anticipate it, and the macro-only line invents a crisis in late 2020.

**Why it matters.** This is the single most important thing to be able to explain about the project, and it is a direct consequence of Insight 2: a model cannot lead a series that is already leading it. R² against the test-period mean flatters any model on a highly autocorrelated series, because the "null" it beats is one no forecaster would ever use. The number was never wrong — it was measured against the wrong thing.

**Possible implication.** Make a named benchmark mandatory in the reporting: every result quoted as R²-vs-random-walk with a Diebold–Mariano p-value, never R² alone. This is a reporting-standard change, and it is the cheapest quality improvement available to the project.

**Limitation.** This shows macro adds nothing *at these horizons, in this specification, for the level of this spread*. It does not show macro is uninformative — Insight 1 shows it is, contemporaneously, in stress. Nor does it rule out value from richer inputs (VIX, yield curve, financial-conditions indices, realised default rates), which is a data gap, not a modelling failure.

---

## Insight 5 — Reframed as classification, the problem is solvable — but only out to about three months.

**Question.** If forecasting the level fails, is there a version of the question the data *can* answer?

**Finding.** Yes. Asking "will the spread be in its top quartile *h* months from now?" instead of "what number will it be?" produces a genuinely useful model at short horizons — AUC 0.94 at one month and 0.79 at three. Then it dies: 0.66 at six months and 0.54 at twelve, indistinguishable from a coin flip. Crucially, **spread-only beats macro+spread at every horizon**, and macro-only at twelve months scores **0.43 — worse than random.**

**Key numbers**
- **AUC 0.94 → 0.79 → 0.66 → 0.54 at 1, 3, 6, 12 months.**
- **At one month the classifier is 4.4× more precise than the base rate; by six months it is 1.5×.**
- Adding six macro series to the spread's own history *lowers* AUC at all four horizons.

**Chart.** `results/figures/05_forecastable_horizon_is_three_months.png`

**What the chart shows.** Left: AUC decay by horizon for three feature sets, with the drop-off between three and six months marked. Right: precision lift over the base rate, showing where the model stops earning its keep.

**Why it matters.** It converts a negative result into a scoped deliverable. The useful artefact is a short-horizon credit-stress *flag*, not a spread forecast — and it defines the honest planning horizon. Anyone using this to set limits twelve months out is using noise.

**Possible implication.** Ship the one-to-three-month stress classifier with an explicit "no signal beyond three months" caveat, and drop point forecasts entirely. Where a longer view is genuinely needed, that is a case for new data, not a longer horizon on this data.

**Limitation.** The 6.43pp threshold is the full-sample 75th percentile — in production it must be set on training data only, which will move it. Positives are scarce and clustered (20–31 across 164 test months, concentrated in a handful of episodes), so the confidence interval on these AUCs is wide and effectively rests on three episodes. AUC also says nothing about calibration: it ranks well, but the probabilities are not yet trustworthy as levels.

---

## Insight 6 — Monthly averaging erased the fastest credit event in the sample.

**Question.** The panel is a monthly mean of a daily price. What does that cost?

**Finding.** March 2020 is the most violent month in 26 years — the daily spread ran from 4.75pp to 10.87pp within it, an intra-month range of 6.12pp, larger than any month of the GFC. In the monthly panel the project actually models, it is **the 43rd-highest month**, an unremarkable bad month. The averaging destroys precisely the event a credit-risk system exists to catch, and the repository already contains the daily source and a month-end alternative.

**Key numbers**
- **March 2020 ranks 1st of 309 by intra-month range and 43rd of 309 by monthly mean.**
- **Its 6.12pp range is 13.6× the median month's 0.45pp**; the monthly mean sits **28% below** the month's peak.
- **6,679 daily observations exist in the repo; 309 monthly rows are modelled.** Switching to month-end alone cuts the random walk's one-month RMSE by **35%**.

**Chart.** `results/figures/06_monthly_averaging_hides_the_shock.png`

**What the chart shows.** Left: the daily spread through COVID with monthly means overlaid as flat bars — the March bar sits far under the spike it is meant to represent. Right: the eight most violent months by intra-month range, with March 2020 at the top and the rank inversion stated.

**Why it matters.** A risk system is judged on how it handles the fast, severe tail. This aggregation choice systematically compresses exactly that: 4.7% understatement in a slow month like October 2002, 27.7% in March 2020. Speed is the discarded dimension, and speed is what distinguishes a manageable widening from a liquidity event.

**Possible implication.** Carry intra-month range and month-end alongside the mean — three cheap columns from data already in the repo — and treat **range as its own monitored indicator** of repricing velocity. It would have ranked March 2020 first in real time.

**Limitation.** The 35% RMSE improvement is partly definitional: a month-end value is temporally closer to the next month's mean, so it should predict it better. It does not mean month-end is a better *target*, only that the mean target is artificially smooth — a genuine 5.2-point drop in trivially-achievable R² when the target is switched. Daily data also introduces noise and non-trading-day gaps, and the macro series remain monthly regardless, so this improves the target, not the alignment.

---

# Top 3 interview-worthy insights

### 1. The model that looked like R² = 0.81 loses to assuming nothing changes

**Finding:** Scored against a random-walk benchmark instead of the test-period mean, the best model has 82% higher error than doing nothing.
**Important number:** Random walk RMSE 0.61pp vs model 1.11pp. R² +0.75 against the mean, −2.30 against the random walk.
**Why it matters:** The original number was never miscalculated — it was measured against a null no forecaster would use. On a series with 0.96 autocorrelation, R² against the test mean flatters everything.

**20–30 second version:** "My headline was R² of 0.81, and I wanted to know if that was actually good. So I built the dumbest possible benchmark — predict next month's spread with this month's. It scored 0.92. My model was 82% worse than doing nothing. The R² wasn't wrong, it was measured against the wrong null. Now I report R² against a named benchmark with a Diebold–Mariano test, never R² alone."

### 2. The spread leads the macro data, which is why the forecast could never work

**Finding:** Credit spread changes Granger-cause unemployment and sentiment changes; the reverse is not significant. In both sample halves, macro leads the spread for 0 of 6 indicators.
**Important number:** Spread → unemployment p < 0.0001; unemployment → spread p = 0.108. Warning before recessions: spread 12/5/15 months, unemployment 0/0/never.
**Why it matters:** It explains the failure mechanically. The spread is a market price set daily by people forecasting the economy; CPI and unemployment are published weeks after the month they describe. I had the slow series chasing the fast one.

**20–30 second version:** "When the model kept losing, I asked whether the direction was even right. I ran cross-correlations and Granger tests both ways. The spread predicts unemployment and sentiment; they don't predict the spread. That makes sense — the spread is a forward-looking price, unemployment is a backward-looking statistic published a month late. The project should be using the spread as a leading indicator *of* the macro, not the other way round."

### 3. Macro is a switch, not a dial — and the switch is off when you need it

**Finding:** The macro block explains 7% of spread changes in calm months and 50% in stress. But the rolling fit collapses to its 26-year low right before the GFC and peaks right after COVID.
**Important number:** R² 0.07 calm vs 0.50 stress (6.7×). Rolling R² bottoms at 0.099 in May 2008; the spread then widened 13.6pp in seven months.
**Why it matters:** Pooling both regimes into one model produces something fitted mostly on uninformative months. And rising model fit is confirmation that a crisis has arrived, not a warning that one is coming.

**20–30 second version:** "I split the sample by a stress index built only from macro data — no spread information — which flags 22 of the 23 NBER recession months in its window. Macro explains 7% of spread moves in calm periods and 50% in stress. It's a switch, not a dial. The uncomfortable part is the timing: rolling fit hit its 26-year low in May 2008, right before the spread widened 13.6 points. Falling fit was the warning, and I'd have read it as a retraining trigger."

---

## Likely interviewer follow-ups

**How did you discover this?** The random-walk finding came from asking what a naive benchmark actually scores. The notes use "baseline" constantly, but always to mean a simpler *model* — Random Forest as the baseline to beat — never a no-model null; "random walk" appears in exactly one of the 31 notes files, and no notebook fits one. Once it beat the model, the lead-lag question followed: if macro can't predict the spread, which direction *does* carry information? Insight 1 came from noticing the full-sample CPI correlation of −0.29 was hiding a rolling correlation spanning −0.86 to +0.94.

**Why this metric?** R²-vs-random-walk (Campbell–Thompson) rather than plain R², because plain R² benchmarks against the test-period mean, which no forecaster knows in advance and which flatters any model on an autocorrelated series. Diebold–Mariano with the Harvey small-sample correction because 163 test months is small enough that the uncorrected statistic over-rejects.

**Why segment that way?** I deliberately built the stress index from macro indicators *only*. Splitting on the spread's own level would condition on the outcome and make the conditional correlations partly circular. Validating against NBER dates externally — 22 of 23 recession months in the index window — meant the split wasn't just fitted to what I hoped to find.

**What could explain this?** Publication timing is the mechanism I find most plausible: the spread is priced daily, macro is published with a one-to-three-month lag and revised for years. A common-driver story also fits — both respond to something unobserved, and the spread reacts faster because it can.

**How did you validate it?** Every finding survives at least two robustness checks: differencing for stationarity (ADF/KPSS say four of seven series are I(1), so level correlations risk spurious regression), point-in-time alignment, sample-half splits, dropping crisis years, and alternative model families. The one that *didn't* survive is worth stating: the sup-Wald test cannot reject parameter stability, so I describe cyclical variation in fit, not a structural break.

**What could make this wrong?** The episode count. The spread's level has lag-1 autocorrelation of 0.963, which leaves about 6 independent observations of the level — differencing recovers most of that, to roughly 149, which is why I test on changes throughout. But the regime findings don't rest on months, they rest on episodes, and there are three: 2001, 2008 and 2020. A fourth would meaningfully change my confidence, in either direction. I'd also want to rule out that the lead-lag result is an artefact of the spread being a price and the macro series being revised aggregates.

**What data would you want?** Daily macro-adjacent series with no publication lag: VIX, the yield curve, a financial-conditions index. Realised default and recovery rates, so the target is credit *loss* rather than credit *pricing*. Real-time vintages rather than revised ones, since even the corrected lags use final revised data. And for a lender specifically, borrower-level performance — the macro layer would then be context, not the model.

**What action follows?** Three concrete things: report every forecast against a named benchmark; ship the one-to-three-month stress classifier and drop point forecasts; monitor rolling R² and intra-month range as risk KPIs, reading falling R² as escalation rather than a retraining trigger.

---

# WEBSITE HANDOFF

### Approved insights, in recommended order

| # | Headline | Chart file | Key numbers |
|---|---|---|---|
| 1 | **The model that looked like R² = 0.75 is 82% worse than assuming nothing changes** | `04_nothing_beats_the_random_walk.png` | RW RMSE **0.61pp** vs model **1.11pp**; R² **+0.75** vs mean, **−2.30** vs random walk; DM p < 0.0001 |
| 2 | **The credit spread moves before the macro data, not after it** | `02_spread_leads_the_macro_data.png` | Spread→unemployment **p < 0.0001**; unemployment→spread **p = 0.108**; warning **12/5/15 months** vs **0/0/never**; macro leads spread in **0 of 6** indicators, both sample halves |
| 3 | **Macro explains credit spreads only when the economy is already under stress** | `01_macro_is_a_switch_not_a_dial.png` | R² **0.07 calm vs 0.50 stress (6.7×)**; **25% of months carry 83% of variance**; regime index flags **22 of 23** NBER recession months in its window |
| 4 | **The macro model fits worst right before a crisis and best right after one** | `03_fit_collapses_before_crises.png` | Rolling R² low **0.099 (May 2008)**, then **+13.6pp** spread widening in 7 months; high **0.763 (Apr 2020)**; corr **+0.33** with spread 9 months earlier |
| 5 | **Credit stress is predictable three months out, and not at all at twelve** | `05_forecastable_horizon_is_three_months.png` | AUC **0.94 → 0.79 → 0.66 → 0.54** at 1/3/6/12 months; **4.4×** precision lift at h=1; macro-only at h=12 is **0.43**, below a coin flip |
| 6 | **Monthly averaging erased the fastest credit event in 26 years** | `06_monthly_averaging_hides_the_shock.png` | March 2020: **1st of 309** by intra-month range, **43rd of 309** by monthly mean; range **6.12pp = 13.6×** the median month; **6,679** daily observations unused |

**Ordering rationale:** open with the result that reframes everything (1), then the mechanism that explains it (2), then the structure that makes it more than a negative result (3, 4), then the constructive scope (5), and close with the data-design lesson (6). A reader who stops after three has the whole argument.

### Short interpretations

1. **Benchmark.** The project's headline R² was measured against the test-period mean — a null no forecaster would use on a series with 0.96 autocorrelation. Against a random walk the same predictions score −2.30. The number was never miscalculated; it was compared to the wrong thing.
2. **Direction.** Credit spreads Granger-cause unemployment and sentiment, not the reverse, and the spread gave 5–15 months of warning before each recession while unemployment gave none. A daily market price was being asked to wait for statistics published weeks late.
3. **Regime.** Macro's explanatory power is 6.7× higher in stress than in calm, and a quarter of months carry 83% of all spread variation. One pooled model is therefore fitted mostly on months where its inputs say almost nothing.
4. **Timing.** Rolling fit hit its 26-year low in May 2008, immediately before a 13.6pp widening, and peaked in April 2020 after the shock. Improving fit confirms a crisis has arrived; deteriorating fit is closer to a warning.
5. **Horizon.** Reframed as "will the spread be in its top quartile in *h* months", the problem is solvable at one to three months (AUC 0.94 and 0.79) and unsolvable at twelve (0.54). This defines the honest planning horizon.
6. **Aggregation.** The monthly mean ranks March 2020 as the 43rd-worst month; by how fast credit repriced within it, it is the worst in the sample. The daily data needed to see this is already in the repository.

### Candidate headline KPIs for the top of the page

- **0.61pp** — random-walk RMSE, the bar every model must clear *(pair with: best model 1.11pp)*
- **6.7×** — how much more macro explains in stress than in calm
- **3 months** — the horizon beyond which no signal survives
- **13.6×** — March 2020's repricing speed versus a median month

### Caveats that must stay visible

- **This is credit *pricing*, not credit *loss*.** No borrower, loan or default data exists here. The spread is what lenders charge for risk, not what they lose.
- **Association and precedence, never causation.** Granger tests show which series moves first, not what drives what. Both may respond to a common unobserved driver.
- **Three recessions carry the regime evidence.** The spread's level has lag-1 autocorrelation of 0.963, leaving ~6 independent observations of the level (differencing recovers ~149, and all tests here use changes). Every regime finding rests on 2001, 2008 and 2020.
- **A formal structural-break test does not reject stability** (sup-Wald 6.11 vs ~17.5). Insight 4 describes cyclical variation in fit, not a proven permanent break.
- **`GDP` is Euro Area 19, not US.** The series is `EA19LORSGPORGYSAM`, an OECD reference series interpolated monthly from quarterly data — and it is one of the strongest correlates in the panel.
- **Thresholds shown (6.43pp stress line, R² 0.15 escalation) are in-sample choices** and would need to be set on training data only in production.
