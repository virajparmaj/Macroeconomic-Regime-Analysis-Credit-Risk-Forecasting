/** Research copy traced to source cells and stored artifacts; no fitted results are generated here. */
export interface FeatureRecord {
  id: string
  code: string
  kind: 'Raw' | 'Engineered' | 'Regime output' | 'Target'
  family: string
  meaning: string
  series: string
  formula: string
  window: string
  experiment: string
  timing: string
  status: string
  source: string
}

export interface ExperimentRecord {
  id: string
  name: string
  purpose: 'Historical regression' | 'Sequence experiments' | 'Regime estimation' | 'Later evaluation' | 'Proposed work'
  status: string
  target: string
  features: string
  window: string
  sample: string
  horizon: string
  metrics: string
  benchmark: string
  limitation: string
  source: string
  sourceDetail?: string
}

export const macroSeries = [
  { code: 'CPI', name: 'U.S. consumer price index', series: 'CPIAUCSL', unit: 'index (1982–84 = 100)', lag: 1 },
  { code: 'FEDFUNDS', name: 'Effective federal funds rate', series: 'FEDFUNDS', unit: 'percent', lag: 1 },
  { code: 'Industrial_Production', name: 'U.S. industrial production index', series: 'INDPRO', unit: 'index', lag: 1 },
  { code: 'GDP', name: 'OECD GDP reference series, Euro Area 19', series: 'EA19LORSGPORGYSAM', unit: 'year-over-year growth, percent', lag: 3 },
  { code: 'Unemployment_Rate', name: 'U.S. unemployment rate', series: 'UNRATE', unit: 'percent', lag: 1 },
  { code: 'Consumer_Sentiment', name: 'University of Michigan consumer sentiment', series: 'UMCSENT', unit: 'index', lag: 0 },
] as const

const p1 = 'notebooks/viraj/p1_models.ipynb'
const p2 = 'notebooks/viraj/p2_models.ipynb'
const unsupervised = 'notebooks/viraj/unsupervised.ipynb'
const audit = 'notebooks/v2/00_audit_of_v1.ipynb'
const revised = 'Publication-lag approximations applied to revised data; historical vintages and exact release timestamps are not reconstructed.'
const seriesFor = (code: string) => macroSeries.find((s) => s.code === code)?.series ?? 'BAMLH0A0HYM2'

const raw: FeatureRecord[] = macroSeries.map((s) => ({
  id: `raw-${s.code}`, code: s.code, kind: 'Raw', family: 'Macro levels',
  meaning: `${s.name}; units: ${s.unit}.`, series: s.series,
  formula: 'Stored monthly source value, aligned on Month_End.', window: 'Monthly reference period',
  experiment: 'Legacy regression and regime inputs; later forecasts use availability-adjusted versions.',
  timing: `Legacy rows are contemporaneous. Later analysis shifts this series by ${s.lag} month${s.lag === 1 ? '' : 's'}. ${revised}`,
  status: 'Verified stored input', source: 'data/merged_macroeconomic_credit.csv',
}))

const historicalLags: FeatureRecord[] = ['GDP', 'CPI', 'Credit_Spread'].flatMap((code) => [1, 2].map((lag) => ({
  id: `legacy-${code}-lag${lag}`, code: `${code}_lag${lag}`, kind: 'Engineered', family: 'Lags',
  meaning: `${code === 'GDP' ? 'Euro-area GDP reference' : code.replaceAll('_', ' ')} from ${lag} month${lag === 1 ? '' : 's'} earlier.`,
  series: seriesFor(code), formula: `${code}(t − ${lag}) = ${code}.shift(${lag})`, window: `${lag}-month lag`,
  experiment: 'Frozen Phase 1: these six lag columns appear in the stored 23-column frame.',
  timing: 'A row lag does not prove publication availability. Spread lags alone do not leak the current target; paired with the unshifted rolling mean, they reconstruct it.',
  status: 'Historical feature; compromised experiment', source: p1,
})))

const historicalRolling: FeatureRecord[] = ['Unemployment_Rate', 'Credit_Spread'].flatMap((code) => ['mean', 'std'].map((stat) => ({
  id: `legacy-${code}-roll${stat}3`, code: `${code}_roll${stat}3`, kind: 'Engineered', family: 'Rolling statistics',
  meaning: stat === 'mean' ? 'Three-month average level.' : 'Three-month sample standard deviation (ddof = 1).',
  series: seriesFor(code), formula: stat === 'mean' ? `[${code}(t) + ${code}(t−1) + ${code}(t−2)] / 3` : `sample_std[${code}(t), ${code}(t−1), ${code}(t−2)]`,
  window: '3 months, including current row', experiment: 'Frozen Phase 1 stored feature frame.',
  timing: code === 'Credit_Spread' ? 'Includes Credit_Spread(t), which is the historical regression target. The rolling mean plus lag1 and lag2 reveals the answer algebraically.' : 'Includes the current unemployment observation before accounting for its publication lag.',
  status: 'Historical feature; timing defect', source: 'utilities/functions.py',
})))

const growth: FeatureRecord[] = ['GDP', 'CPI', 'Industrial_Production'].map((code) => ({
  id: `legacy-${code}-growth`, code: `${code}_growth`, kind: 'Engineered', family: 'Percentage changes',
  meaning: `One-month percentage change in the stored ${code} series.`, series: seriesFor(code),
  formula: `100 × [${code}(t) / ${code}(t−1) − 1]`, window: '1 month',
  experiment: 'Frozen Phase 1; CPI_growth is also a SARIMAX exogenous input.',
  timing: code === 'GDP' ? 'GDP is already a Euro-area growth reference series and crosses zero. A percentage change of it can explode (stored minimum about −1,583%); existence in code is not economic justification.' : 'Uses the current reference-period value. Publication timing must be respected before forecasting.',
  status: 'Historical transformation', source: 'utilities/functions.py',
}))

const currentUtilityExtras: FeatureRecord[] = [
  ...['GDP', 'CPI', 'Credit_Spread'].map((code): FeatureRecord => ({
    id: `current-${code}-lag3`, code: `${code}_lag3`, kind: 'Engineered', family: 'Lags',
    meaning: 'Additional third lag in the current legacy utility defaults.', series: seriesFor(code),
    formula: `${code}(t−3)`, window: '3 months', experiment: 'Current utilities.create_features, not the frozen Phase 1 23-column frame.',
    timing: 'A third lag is a current code capability; it does not prove that the stored historical model used it. Macro availability still needs release alignment.',
    status: 'Current code; differs from frozen output', source: 'utilities/data_processing.py',
  })),
  ...['Unemployment_Rate', 'Credit_Spread'].flatMap((code) => [6, 12].flatMap((window) => ['mean', 'std'].map((stat): FeatureRecord => ({
    id: `current-${code}-roll${stat}${window}`, code: `${code}_roll${stat}${window}`, kind: 'Engineered', family: 'Rolling statistics',
    meaning: `Current utility ${window}-month ${stat === 'mean' ? 'average' : 'sample standard deviation'}.`, series: seriesFor(code),
    formula: `${stat === 'mean' ? 'mean' : 'sample_std'}[${code}(t), …, ${code}(t−${window - 1})]`, window: `${window} months; includes current t`,
    experiment: 'Expanded current utilities defaults create 34 columns; this column is absent from the frozen Phase 1 23-column output.',
    timing: code === 'Credit_Spread' ? 'Contains the contemporaneous target when used for same-row regression. Extra columns are not proof of historical use.' : 'Current reference-period unemployment must be aligned to its later publication date before forecasting.',
    status: 'Current code; differs from frozen output', source: 'utilities/data_processing.py',
  })))),
]

const laterDifferences: FeatureRecord[] = macroSeries.flatMap((s) => [1, 3, 12].map((lag) => {
  const code = s.code === 'GDP' ? 'EA19_GDP_OECD' : s.code
  return {
    id: `later-${code}-d${lag}`, code: `${code}_d${lag}`, kind: 'Engineered', family: 'Differences',
    meaning: `${lag}-month change in the availability-adjusted ${s.name.toLowerCase()}; retains the source unit.`,
    series: s.series, formula: `available_${code}(t) − available_${code}(t−${lag})`, window: `${lag} months, after ${s.lag}-month source shift`,
    experiment: lag === 3 ? 'Spread-change RF/Ridge and original horizon-classification experiments; absent from the future-level RF macro block.' : 'Later future-level RF, spread-change RF/Ridge, and original horizon-classification experiments.',
    timing: revised, status: 'Reproduced exploratory specification', source: lag === 3 ? 'analysis/fair_eval.py' : 'analysis/honest_eval.py',
  }
}))

const laterSpread: FeatureRecord[] = [
  ['sp_lag0', 'Current monthly spread level', 'S(t)', 'Current origin month', 'Future-level RF, spread-change RF/Ridge, corrected stress classifier'],
  ['sp_d1', 'One-month spread change', 'S(t) − S(t−1)', '1 month', 'Future-level RF and spread-change RF/Ridge'],
  ['sp_d3', 'Three-month spread change', 'S(t) − S(t−3)', '3 months', 'Future-level RF, spread-change RF/Ridge, corrected stress classifier'],
  ['sp_rollmean3_lag1', 'Lagged three-month spread mean', '[S(t−1) + S(t−2) + S(t−3)] / 3', '3 months, shifted by 1', 'Future-level RF only'],
  ['sp_rollstd6_lag1', 'Lagged six-month spread variability', 'sample_std[S(t−1), …, S(t−6)]', '6 months, shifted by 1', 'Future-level RF and spread-change RF/Ridge'],
  ['sp_rollstd6', 'Lagged six-month spread variability', 'sample_std[S(t−1), …, S(t−6)]', '6 months, shifted by 1 despite the short name', 'Corrected stress classifier'],
].map(([code, meaning, formula, window, experiment]) => ({
  id: `later-${code}`, code, kind: 'Engineered', family: 'Spread history', meaning: `${meaning}; percentage points.`,
  series: 'BAMLH0A0HYM2', formula, window, experiment,
  timing: 'Information through origin t forecasts a future target, so current spread can be valid. The monthly aggregate must actually be available at the stated origin.',
  status: 'Reproduced exploratory specification', source: code === 'sp_rollstd6' ? 'research/audit_checks.py' : 'analysis/honest_eval.py',
}))

const regimeFeatures: FeatureRecord[] = [
  { id: 'regime-label', code: 'Regime_Label', kind: 'Regime output', family: 'Historical regimes', meaning: 'Hard assignment to one of ten GMM components; numeric IDs have no intrinsic economic ordering.', series: 'Six macro series via standardized four-component PCA', formula: 'argmax_k P(component k | macro PCA scores)', window: 'Full-sample fit: Dec 1996–Aug 2022', experiment: 'Phase 2 Random Forest uses one-hot encoded label.', timing: 'Retrospective: scaler, PCA, component selection and GMM are fitted using the full sample.', status: 'Historical retrospective output', source: unsupervised },
  ...Array.from({ length: 10 }, (_, i): FeatureRecord => ({
    id: `regime-prob-${i}`, code: `Regime_Prob_${i}`, kind: 'Regime output', family: 'Historical regimes',
    meaning: `GMM membership probability for component ${i}; not a credit-default probability.`, series: 'Six macro series via standardized four-component PCA',
    formula: `GMM.predict_proba(X_pca4)[:, ${i}]`, window: 'Full-sample fit: 309 months',
    experiment: 'Phase 2 LSTM, stacked LSTM, GRU and Transformer use all ten membership columns.',
    timing: 'Produced by the explicit GMM refit, not by K-Means. Full-sample construction makes these retrospective.', status: 'Historical retrospective output', source: unsupervised,
  })),
  { id: 'regime-smoothed', code: 'Regime_Label_Smoothed', kind: 'Regime output', family: 'Historical regimes', meaning: 'Smoothed numeric cluster assignment.', series: 'Regime_Label', formula: 'rolling(window=3, center=True).mode(), with edge filling', window: 't−1, t, t+1', experiment: 'Historical regime visualization and exported regime-enriched CSV.', timing: 'Centered smoothing uses the next month. This label is unavailable in real time.', status: 'Historical retrospective output', source: unsupervised },
  { id: 'regime-name', code: 'Regime_Name', kind: 'Regime output', family: 'Historical regimes', meaning: 'Analyst-assigned economic interpretation of a cluster.', series: 'Regime_Label mapped to a name', formula: 'Regime_Label.map(label_to_name)', window: 'Full-sample interpretation', experiment: 'Legacy regime descriptions.', timing: 'Mappings differ across artifacts. Names are interpretations, not verified economic states; inspect numeric-cluster profiles.', status: 'Historical interpretation; mapping conflicts', source: unsupervised },
  { id: 'macro-stress', code: 'macro_stress', kind: 'Regime output', family: 'Macro stress', meaning: 'Separate macro-only deterioration index; no spread input.', series: 'UNRATE, INDPRO, UMCSENT, EA19LORSGPORGYSAM', formula: 'mean(full_sample_z(unemp_chg12, ip_yoy, sent_z, gdp_neg))', window: 'Nov 2001–Aug 2022 after 60-month warm-up', experiment: 'Descriptive conditional macro/spread analysis.', timing: 'Full-sample standardization and the full-sample 75th-percentile threshold make this retrospective.', status: 'Verified descriptive construction', source: 'analysis/macro_regime.py' },
]

const stressComponents: FeatureRecord[] = [
  ['unemp_chg12', 'Unemployment above its trailing minimum', 'UNRATE', 'U(t) − min[U(t−11), …, U(t)]', '12 months'],
  ['ip_yoy', 'Negative industrial-production year-over-year growth', 'INDPRO', '−100 × [IP(t) / IP(t−12) − 1]', '12 months'],
  ['sent_z', 'Negative trailing sentiment z-score', 'UMCSENT', '−[sentiment(t) − mean60(t)] / sample_std60(t)', '60 months, including t'],
  ['gdp_neg', 'Negative Euro-area GDP reference value', 'EA19LORSGPORGYSAM', '−GDP(t)', 'Current reference period'],
].map(([code, meaning, series, formula, window]) => ({
  id: `stress-${code}`, code, kind: 'Engineered', family: 'Macro stress', meaning, series, formula, window,
  experiment: 'Four inputs to the separate macro-only stress index.', timing: 'Unadjusted revised monthly inputs; each component is standardized again over the full retained sample before averaging.',
  status: 'Verified descriptive construction', source: 'analysis/macro_regime.py',
}))

export const features: FeatureRecord[] = [
  ...raw,
  { id: 'spread', code: 'Credit_Spread', kind: 'Target', family: 'Targets', meaning: 'ICE BofA U.S. High Yield option-adjusted spread: an aggregate credit-market risk indicator, in percentage points.', series: 'BAMLH0A0HYM2', formula: 'Mean of source-date daily values within month, after forward filling missing source rows.', window: 'Monthly; Dec 1996–Aug 2022', experiment: 'Core legacy target; historical models fit S(t), later forecasts target S(t+h).', timing: 'Distinct from the observed-day-only monthly mean. The source ends Aug 1, 2022, so the final monthly bin is incomplete.', status: 'Verified stored target', source: 'data/merged_macroeconomic_credit.csv' },
  ...historicalLags, ...historicalRolling, ...growth,
  { id: 'growth-interaction', code: 'GDP_growth_x_CPI_growth', kind: 'Engineered', family: 'Interactions', meaning: 'Product of two legacy growth transformations.', series: 'EA19LORSGPORGYSAM × CPIAUCSL', formula: 'GDP_growth(t) × CPI_growth(t)', window: 'Current row', experiment: 'Frozen Phase 1.', timing: 'Inherits current-period macro timing and the instability of percentage changes in a GDP reference series that crosses zero.', status: 'Historical transformation', source: 'utilities/functions.py' },
  { id: 'spread-interaction', code: 'Credit_Spread_x_FEDFUNDS', kind: 'Engineered', family: 'Interactions', meaning: 'Product of contemporaneous spread and policy rate.', series: 'BAMLH0A0HYM2 × FEDFUNDS', formula: 'Credit_Spread(t) × FEDFUNDS(t)', window: 'Current row', experiment: 'Frozen Phase 1 current-target regression.', timing: 'Leaks the current target: divide by nonzero FEDFUNDS(t) to recover Credit_Spread(t). This is distinct from an available S(t) predicting S(t+1).', status: 'Historical feature; target leakage', source: 'utilities/functions.py' },
  { id: 'legacy-target', code: 'Target', kind: 'Target', family: 'Targets', meaning: 'Next-month spread column created in the historical feature frame, then excluded by the RF feature guard.', series: 'BAMLH0A0HYM2', formula: 'Credit_Spread.shift(−1)', window: 't+1', experiment: 'Stored Phase 1 23-column frame; actual model target remains Credit_Spread(t).', timing: 'The frozen RF cell explicitly removes Target. Do not attribute a renamed-target leak to that stored run.', status: 'Historical stored column; excluded predictor', source: p1 },
  { id: 'current-target', code: 'Target_1step_ahead', kind: 'Target', family: 'Targets', meaning: 'Future spread column emitted by the current legacy utility code.', series: 'BAMLH0A0HYM2', formula: 'Credit_Spread.shift(−1)', window: 't+1', experiment: 'Current utilities.create_features; not the column name in frozen outputs.', timing: 'The old notebook guard only drops Target, so it would fail to exclude this renamed future outcome if rerun against current utilities.', status: 'Current code; mismatch with frozen run', source: 'utilities/data_processing.py' },
  ...currentUtilityExtras,
  { id: 'safe-rolling', code: 'Credit_Spread_rollmean3_lag1', kind: 'Engineered', family: 'Rolling statistics', meaning: 'Corrected utility mean that excludes the current row.', series: 'BAMLH0A0HYM2', formula: '[S(t−1) + S(t−2) + S(t−3)] / 3', window: '3 months, shifted by 1', experiment: 'src.features correction demonstrated in v2 audit; downstream v2 study notebooks remain unexecuted.', timing: 'The reconstruction identity no longer holds. This feature correction alone does not validate a forecasting study.', status: 'Corrected utility; audit demonstration', source: 'src/features.py' },
  { id: 'safe-growth', code: 'GDP_growth1', kind: 'Engineered', family: 'Percentage changes', meaning: 'Corrected utility growth naming; denominator uses absolute prior value and masks exact zero.', series: 'EA19LORSGPORGYSAM', formula: '100 × [GDP(t) − GDP(t−1)] / |GDP(t−1)|; zero denominator → NaN', window: '1 month', experiment: 'src.features utility, not a verified final model feature specification.', timing: 'Near-zero denominators can still create large ratios. Numerical handling does not establish economic suitability.', status: 'Corrected utility; final study pending', source: 'src/features.py' },
  ...laterDifferences, ...laterSpread,
  { id: 'future-level', code: 'y', kind: 'Target', family: 'Targets', meaning: 'Future legacy monthly spread level, in percentage points.', series: 'BAMLH0A0HYM2', formula: 'S(t+1)', window: '1 month ahead', experiment: 'analysis/honest_eval.py reproduced RF level forecasts.', timing: 'Rows without a realized future target are dropped. Forecast-origin dates precede target dates by one month.', status: 'Reproduced exploratory target', source: 'analysis/honest_eval.py' },
  { id: 'future-change', code: 'dy', kind: 'Target', family: 'Targets', meaning: 'Future monthly spread change, in percentage points.', series: 'BAMLH0A0HYM2', formula: 'S(t+1) − S(t)', window: '1 month ahead', experiment: 'analysis/fair_eval.py RF/Ridge change forecasts; level forecast is S(t) + predicted dy.', timing: 'Persistence predicts zero change. This is a different training target from future levels even when errors can be compared on matched observations.', status: 'Reproduced exploratory target', source: 'analysis/fair_eval.py' },
  { id: 'safe-target', code: 'y_Credit_Spread_h1', kind: 'Target', family: 'Targets', meaning: 'Explicitly named future-level target returned with its column name by the corrected utility.', series: 'BAMLH0A0HYM2', formula: 'Credit_Spread.shift(−1)', window: '1 month ahead', experiment: 'src.features; v2 leakage audit. Later v2 study notebooks have no stored executions.', timing: 'Must be excluded from predictors; missing tail targets must remain unknown.', status: 'Corrected utility; final study pending', source: 'src/features.py' },
  { id: 'stress-target', code: 'y', kind: 'Target', family: 'Targets', meaning: 'Whether future spread is at or above the full-sample upper-quartile threshold; not a new-entry event.', series: 'BAMLH0A0HYM2', formula: '1[S(t+h) ≥ Q75(S)]; unknown S(t+h) → NaN', window: 'h = 1, 3, 6 or 12 months', experiment: 'Corrected spread-only RF classification diagnostic versus raw current-spread ranking.', timing: 'Future unknowns excluded, but full-sample threshold remains retrospective. Ranking scores are not calibrated probabilities.', status: 'Corrected exploratory diagnostic', source: 'research/audit_checks.py' },
  ...regimeFeatures, ...stressComponents,
]

const historicalLimit = 'Stored output only, with contemporaneous target leakage and notebook/code drift. It is not a valid final forecast ranking.'
const sequenceBase = {
  purpose: 'Sequence experiments' as const, status: 'Historical output; validity limits', target: 'Next-month legacy-mean Credit_Spread, pp',
  features: '12-month sequences of six macro levels and ten GMM membership probabilities; spread history is absent.',
  window: 'Target dates Jan 2019–Aug 2022; train through Dec 2018', sample: '44 test windows; 253 train windows', horizon: '1 month after the last input month', benchmark: 'No matched persistence benchmark in this notebook.',
  limitation: 'Full-sample upstream regime fitting is retrospective. Small sample and different feature information prevent comparison with the Phase 1 RF headline.', source: p2,
}

export const experiments: ExperimentRecord[] = [
  { id: 'sarimax', name: 'SARIMAX (1,1,1)', purpose: 'Historical regression', status: 'Historical conditional forecast', target: 'Credit_Spread: legacy monthly mean, pp', features: 'GDP, GDP_lag1, CPI_growth, Unemployment_Rate, FEDFUNDS', window: 'Train Feb 1997–Dec 2018; target dates Jan 2019–Jul 2022', sample: '263 training observations; 43 test targets', horizon: '43-step path from the Dec 2018 origin', metrics: 'Stored MSE 214.8933 pp²; MAE 7.7677 pp', benchmark: 'No persistence comparison in this historical execution.', limitation: 'Forecast receives realized test-period exogenous values. The order is an example configuration; no verified model-selection rationale.', source: p1, sourceDetail: 'Cells 6 and 9 (zero-based)' },
  { id: 'legacy-rf', name: 'Random Forest · Phase 1', purpose: 'Historical regression', status: 'Historical output; target leakage', target: 'Contemporaneous legacy-mean Credit_Spread(t), pp', features: 'Macro levels, macro/spread lags, current-inclusive rolling values, growth and interactions. Literal Target excluded.', window: 'Train Feb 1997–Dec 2018; test Jan 2019–Jul 2022', sample: '263 train; 43 test', horizon: 'Same-row regression, not next-month forecasting', metrics: 'Stored MSE 0.2714 pp²; RMSE 0.5209 pp; ordinary R² 0.8084', benchmark: 'Test-mean R²; no persistence benchmark.', limitation: historicalLimit, source: p1, sourceDetail: 'Cells 4 and 12; see v2/00_audit_of_v1 for reconstruction defects' },
  { id: 'legacy-rf-tuned', name: 'Tuned Random Forest · Phase 1', purpose: 'Historical regression', status: 'Historical output; target leakage', target: 'Contemporaneous legacy-mean Credit_Spread(t), pp', features: 'Same compromised Phase 1 features; 200 trees, max_depth=10.', window: 'Train Feb 1997–Dec 2018; test Jan 2019–Jul 2022', sample: '263 train; 43 test', horizon: 'Same-row regression', metrics: 'Stored MSE 0.2947 pp²; RMSE 0.5428 pp; ordinary R² 0.7920', benchmark: 'Ordinary cv=3 grid search, not time-series cross-validation.', limitation: historicalLimit, source: p1, sourceDetail: 'Cell 15' },
  { id: 'regularized', name: 'Lasso & Ridge regression', purpose: 'Historical regression', status: 'Historical in-sample diagnostics', target: 'Contemporaneous spread level, pp', features: 'Six macro inputs with standardized polynomial degree-two expansion; linear variants also exist.', window: 'Full 309-row input; exact fitting dates not printed in this artifact', sample: '309 fitted observations; no held-out metric in this cell', horizon: 'Same-row fitted regression', metrics: 'Degree-two Lasso MSE 2.0642 pp², R² 0.6942; Ridge MSE 1.2954 pp², R² 0.8081', benchmark: 'TimeSeriesSplit selects hyperparameters; displayed scores are then calculated on the fitted full sample.', limitation: 'Full-sample preprocessing and in-sample scoring. These scores cannot establish forecasting skill.', source: 'notebooks/old_work/lasso_ridge.ipynb', sourceDetail: 'Cell 0; linear alternatives in cell 2' },
  { id: 'polynomial', name: 'Polynomial regression', purpose: 'Historical regression', status: 'Historical random-split output', target: 'Contemporaneous spread level, pp', features: 'Six macro inputs; cubic regression and tuned degree-two Ridge.', window: 'Random 80/20 partition; calendar evaluation window not verified', sample: 'Test size = 20%; exact count not printed', horizon: 'Same-row regression', metrics: 'Cubic MSE 3.5885 pp², R² 0.2874; tuned quadratic Ridge MSE 1.0406 pp², R² 0.7934', benchmark: 'No chronological persistence evaluation.', limitation: 'Random splitting allows later periods into training. Not comparable with walk-forward forecasts.', source: 'notebooks/old_work/poly_reg.ipynb', sourceDetail: 'Stored outputs in cells 0 and 2 (execution counters are cleared)' },
  { id: 'knn-tree', name: 'k-nearest neighbors & Decision Tree', purpose: 'Historical regression', status: 'Historical random-split output', target: 'Contemporaneous legacy monthly spread, pp', features: 'Date ordinal, five macro levels and CPI first difference; full-data standardization.', window: 'Random 70/30 partition; calendar test window is not contiguous', sample: 'Input after first difference: 308 rows; split test fraction 30%', horizon: 'Same-row regression', metrics: 'kNN MSE 3.2679 pp², R² 0.6527; tree MSE 2.3681 pp², R² 0.7483', benchmark: 'No chronological persistence benchmark.', limitation: 'Full-data scaling and random time-series split; retained as exploratory breadth only.', source: 'notebooks/old_work/ Basic ML Time Series Models.ipynb', sourceDetail: 'Cells 3, 5 and 7' },
  { id: 'xgboost', name: 'XGBoost', purpose: 'Historical regression', status: 'Attempted; no successful result', target: 'Intended Phase 1 spread regression', features: 'Intended legacy feature frame.', window: 'No verified completed evaluation', sample: 'No successful fitted result', horizon: 'No completed forecast', metrics: 'No verified metric', benchmark: 'Not applicable.', limitation: 'XGBRegressor import is commented out; stored NameError stops the attempt.', source: p1, sourceDetail: 'Cell 18' },
  { id: 'phase2-rf', name: 'Random Forest · Phase 2', purpose: 'Historical regression', status: 'Historical output; conflicting narrative', target: 'Contemporaneous legacy-mean spread, pp', features: 'Six macro levels + one-hot Regime_Label. Executable feature list contains no spread lags.', window: 'Test Jan 2019–Aug 2022; train through Dec 2018', sample: '44 test months', horizon: 'Same-row regression', metrics: 'Stored MSE 5.723 pp²; MAE 1.889 pp; R² −3.194', benchmark: 'No matched persistence benchmark.', limitation: 'Full-sample regime construction. Narrative MSE 0.524/R² 0.616 and hard-coded summary table are not the executed result.', source: p2, sourceDetail: 'Cell 6; conflicts in cells 9 and 42' },
  { ...sequenceBase, id: 'lstm', name: 'LSTM', metrics: 'Stored MSE 8.455 pp²; MAE 2.568 pp; R² −5.081', sourceDetail: 'Cell 19; improved bidirectional LSTM' },
  { ...sequenceBase, id: 'stacked', name: 'Stacked LSTM', metrics: 'Final visible MSE 8.016 pp²; MAE 2.412 pp; R² −4.766', sourceDetail: 'Cell 29; several earlier, weaker stored variants also exist' },
  { ...sequenceBase, id: 'gru', name: 'GRU', metrics: 'Stored MSE 6.748 pp²; MAE 2.236 pp; R² −3.854', sourceDetail: 'Cell 33; GRU-QuickWin' },
  { ...sequenceBase, id: 'transformer', name: 'Transformer', metrics: 'Stored MSE 5.031 pp²; MAE 1.801 pp; R² −2.619', sourceDetail: 'Cell 38' },
  { id: 'hmm', name: 'Hidden Markov Model', purpose: 'Regime estimation', status: 'Historical retrospective exploration', target: 'Latent macro states; no forecast target', features: 'Six full-sample standardized macro series.', window: 'Dec 1996–Aug 2022', sample: '309 months', horizon: 'Retrospective state estimation', metrics: 'Stored search selects six states, full covariance; BIC 462.56; silhouette 0.280', benchmark: '2–9 states and full/diagonal covariance compared.', limitation: 'State names are analyst interpretations. Full-sample fit and convergence warnings limit interpretation; not a real-time recession detector.', source: 'notebooks/old_work/Regime detection_Unsupervise_learning.ipynb', sourceDetail: 'Cells 1–3 and 9' },
  { id: 'pca', name: 'PCA & Kernel PCA', purpose: 'Regime estimation', status: 'Historical retrospective exploration', target: 'Macro representation, not spread prediction', features: 'Six standardized macro series; credit spread excluded.', window: 'Dec 1996–Aug 2022', sample: '309 months', horizon: 'Full-sample descriptive fit', metrics: 'Four linear PCs: stored variance ratios 0.486, 0.303, 0.094, 0.063. Separate 2D Kernel PCA search silhouette ≈0.582.', benchmark: 'Kernel PCA dimensions 2–6 explored; final clustering uses four linear PCs.', limitation: 'PCA variance is not predictive value. The KPCA exploration is not the final exported regime path.', source: unsupervised, sourceDetail: 'Cells 14, 18, 23 and 30' },
  { id: 'kmeans-gmm', name: 'K-Means → GMM soft assignments', purpose: 'Regime estimation', status: 'Historical retrospective output', target: 'Macro-cluster membership', features: 'Four linear principal components from six standardized macro series.', window: 'Dec 1996–Aug 2022', sample: '309 months', horizon: 'Full-sample descriptive fit', metrics: 'K-Means k=10 selected: silhouette 0.515, Calinski–Harabasz 247.2. Then an explicit 10-component GMM is refitted.', benchmark: 'K-Means and GMM k=2–10 comparisons; Bayesian mixtures also explored.', limitation: 'Exported probabilities are from GMM. Centered smoothing sees t+1. Economic names conflict across artifacts; use cluster IDs and profiles.', source: unsupervised, sourceDetail: 'Cells 30, 41, 47, 49 and 52' },
  { id: 'monthly-persistence', name: 'Monthly persistence / random walk', purpose: 'Later evaluation', status: 'Reproduced exploratory benchmark', target: 'Next-month legacy mean, pp', features: 'Previous monthly mean S(t).', window: 'Origins Jan 2009–Jul 2022; targets Feb 2009–Aug 2022', sample: '163 forecasts', horizon: '1 month', metrics: 'RMSE ≈0.6124 pp', benchmark: 'Predict S(t+1) = S(t).', limitation: 'Revised historical dataset; last target bin has incomplete daily coverage. This result is specific to this target and evaluation window.', source: 'research/evidence/aggregation_benchmarks.csv' },
  { id: 'latest-daily', name: 'Latest daily spread benchmark', purpose: 'Later evaluation', status: 'Exploratory information comparison', target: 'Next-month legacy mean, or separately observed-day mean, pp', features: 'Last observed daily spread in the forecast-origin month.', window: 'Origins Jan 2009–Jul 2022; targets Feb 2009–Aug 2022', sample: '163 matched forecasts per target definition', horizon: '1 month', metrics: 'Legacy target RMSE ≈0.4212 pp; observed-day target ≈0.4221 pp', benchmark: 'Compare with monthly persistence for the same target and dates.', limitation: 'Does not show significance, live availability, or model performance when models receive identical endpoint information.', source: 'research/evidence/aggregation_benchmarks.csv' },
  { id: 'rf-level', name: 'Random Forest · future level', purpose: 'Later evaluation', status: 'Reproduced exploratory forecast', target: 'S(t+1), legacy monthly mean, pp', features: 'Macro levels + first/twelve-month differences; second variant adds current spread, changes, shifted mean and standard deviation.', window: 'Origins Jan 2009–Jul 2022; annual refits and 3-month training gap', sample: '163 forecasts per specification', horizon: '1 month', metrics: 'RMSE: macro-only 2.4004 pp; macro + spread history 1.1131 pp', benchmark: 'Monthly persistence RMSE 0.6124 pp on matched targets.', limitation: `${revised} Both tested specifications lose to persistence; this is not a universal result about macro predictors.`, source: 'research/evidence/honest_eval.log' },
  { id: 'change-models', name: 'Random Forest & Ridge · spread change', purpose: 'Later evaluation', status: 'Reproduced exploratory forecast', target: 'S(t+1) − S(t), pp', features: 'Macro levels and 1/3/12-month differences; spread variants add current spread, 1/3-month changes and shifted 6-month standard deviation.', window: 'Origins Jan 2009–Jul 2022; annual refits and 3-month training gap', sample: '163 forecasts per specification', horizon: '1 month', metrics: 'RMSE: RF macro-only 0.8994 pp; RF macro + spread 0.8573 pp; Ridge macro + spread 1.1412 pp', benchmark: 'Zero-change persistence RMSE 0.6124 pp.', limitation: 'Changed targets and feature blocks are explicit. These reproduced variants also lose to persistence; matched regime ablation has not been run.', source: 'research/evidence/fair_eval.log' },
  { id: 'classifier', name: 'Stress-state classification diagnostic', purpose: 'Later evaluation', status: 'Corrected exploratory diagnostic', target: 'Future occupancy at or above full-sample Q75 spread threshold', features: 'RF: current spread, three-month change, shifted six-month standard deviation. Comparator: raw current-spread ranking score.', window: 'Origins from Jan 2009; final origin depends on horizon', sample: '163 / 161 / 158 / 152 months for h=1 / 3 / 6 / 12', horizon: '1, 3, 6 and 12 months', metrics: 'AUC and average precision are displayed by horizon in the corrected classification table.', benchmark: 'Raw current-spread ranking evaluated side by side.', limitation: 'Unknown future outcomes excluded. Threshold still full-sample; scores are not calibrated probabilities. Occupancy differs from a new stress entry.', source: 'research/evidence/classification_label_diagnostic.csv' },
  { id: 'audit-ar', name: 'AR(1) & persistence · separate v2 audit', purpose: 'Later evaluation', status: 'Stored audit output; different window', target: 'Next-month legacy-mean spread, pp', features: 'AR(1): current spread alone.', window: 'Origins Jan 2019–Jul 2022; single split', sample: '43 forecasts', horizon: '1 month', metrics: 'Stored AR(1) MSE 0.4864 pp² versus persistence 0.4998 pp²; OOS R² 0.0268', benchmark: 'Same-window monthly persistence; stored DM p=0.352.', limitation: 'Small numerical improvement is not distinguishable in this stored test. Do not combine it with the 163-origin evaluation.', source: audit, sourceDetail: 'Cells 39–40' },
  { id: 'matched-study', name: 'Matched regime-ablation study', purpose: 'Proposed work', status: 'Proposed; not run', target: 'Future observed-day monthly mean; modeled residual m(t+h) − z(t), pp, where z(t) is the latest daily spread.', features: 'Primary Ridge comparison: S_last + M + R versus S_last + M. Both include spread history and the latest daily spread; R is a two-state K-Means representation fitted on training macros only.', window: 'Proposed evaluation beginning January 2009; no completed artifact', sample: 'No completed study sample', horizon: 'Primary h=1 month; secondary h=3 months', metrics: 'No result available', benchmark: 'Primary contrast: same-origin MSE with versus without regimes. Monthly-mean and latest-daily persistence are separate benchmarks on the same target.', limitation: 'The proposed design specifies monthly expanding fits, training-label maturity checks, fold-fitted preprocessing and dependent-data uncertainty. It remains unexecuted; later-period or external validation also remains pending.', source: 'research/paper_preparation/05_STUDY_DESIGN.md' },
  { id: 'onset-study', name: 'Frozen-threshold stress-entry forecasting', purpose: 'Proposed work', status: 'Event counts verified; forecast study pending', target: 'New crossing into stress, conditional on being below threshold', features: 'To be pre-specified.', window: 'Post-2008 event inventory; frozen pre-2009 threshold', sample: 'Three distinct entries: Sep 2011, Jan 2016, Mar 2020', horizon: 'Protocol considers 1 / 3 / 6 / 12 months', metrics: 'Event counts only; no completed early-warning score', benchmark: 'To be specified for at-risk origins.', limitation: 'Very few events. Entry cannot inherit occupancy AUC claims; historical vintage reconstruction and later-period validation remain pending.', source: 'research/evidence/onset_counts.csv' },
]

export const regimeMethods = [
  { name: 'Historical clusters', description: 'HMM exploration → standardized macros → PCA / Kernel PCA exploration → K-Means and mixture comparisons → K-Means k=10 selection → explicit GMM k=10 refit for membership probabilities. The final exported path uses linear PCA.', timing: 'Full-sample preprocessing and centered three-month smoothing are retrospective.', source: unsupervised },
  { name: 'Macro-only stress', description: 'Unemployment above its trailing minimum, negative industrial-production growth, negative sentiment z-score and negative Euro-area GDP reference value. Standardize each component, average, then apply the upper-quartile cutoff.', timing: 'No spread input, but full-sample normalization and threshold remain retrospective.', source: 'analysis/macro_regime.py' },
  { name: 'Spread-defined stress', description: 'A threshold on the credit-spread level defines occupancy. A new entry is a transition from below to at or above that threshold. Neither definition is the same as a macro cluster or recession.', timing: 'Full-sample thresholds describe history; a frozen pre-2009 threshold supports only the separate three-event inventory.', source: 'research/audit_checks.py' },
]
