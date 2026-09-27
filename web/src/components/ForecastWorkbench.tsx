import { useState } from 'react'
import { research, predictions, forecastMetrics, aggregationMetrics, classificationMetrics, type PredictionRow } from '@/lib/research'
import { ResearchPlot } from '@/charts/ResearchPlot'
import { Evidence, Source } from './Evidence'

const f = (v: number, digits = 4) => v.toFixed(digits)
const models = [
  { key: 'rfMacroSpread', label: 'RF · macro + spread history' },
  { key: 'rfMacroOnly', label: 'RF · macro only' },
] as const
type Model = typeof models[number]['key']

export function ForecastWorkbench() {
  const [model, setModel] = useState<Model>('rfMacroSpread')
  const [window, setWindow] = useState('all')
  const [index, setIndex] = useState(0)
  const [axis, setAxis] = useState<'targetDate' | 'originDate'>('targetDate')
  const rows = predictions.filter(r => window === 'all' || r.originDate.startsWith('2020') || r.originDate.startsWith('2019'))
  const selected = rows[Math.min(index, rows.length - 1)]
  const refits = rows.flatMap((r, i) => i === 0 || r.refitDate !== rows[i - 1].refitDate ? [i] : [])
  const value = (key: keyof Pick<PredictionRow, 'actual' | 'persistence' | Model>) => rows.map(r => r[key])
  return <>
    <div className="figure-header"><div><span className="eyebrow">01 / Forecast record</span><h3>Does the model beat persistence?</h3></div><span className="badge">Next month · legacy mean</span></div>
    <p>The benchmark carries the latest monthly mean forward. Both future-level Random Forest specifications have higher error on these 163 forecast origins. Change-target experiments also lose on the same evaluation.</p>
    <div className="controls">
      <label className="field">Prediction series<select className="control" value={model} onChange={e => setModel(e.target.value as Model)}>{models.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
      <label className="field">Window<select className="control" value={window} onChange={e => { setWindow(e.target.value); setIndex(0) }}><option value="all">All 163 forecast origins</option><option value="covid">2019–2020 origins</option></select></label>
      <label className="field">Time axis<select className="control" value={axis} onChange={e => setAxis(e.target.value as typeof axis)}><option value="targetDate">Realized target month</option><option value="originDate">Forecast origin month</option></select></label>
    </div>
    <ResearchPlot dates={rows.map(r => r[axis])} series={[
      { label: 'Actual', color: '#f1f0eb', values: value('actual') },
      { label: 'Monthly persistence', color: '#9aafa5', values: value('persistence'), dashed: true },
      { label: models.find(m => m.key === model)!.label, color: '#9eaab9', values: value(model) },
    ]} label={`Actual next-month legacy spread, monthly persistence and ${model}; ${axis === 'originDate' ? 'forecast origin' : 'target'} dates`} active={Math.min(index, rows.length - 1)} onActive={setIndex} refits={refits} />
    <div className="inspector">
      <label className="field">Inspect forecast record<input aria-label="Inspect forecast record" type="range" aria-valuetext={`Origin ${selected.originDate}, target ${selected.targetDate}, actual ${f(selected.actual,3)} percentage points`} min="0" max={rows.length - 1} value={Math.min(index, rows.length - 1)} onChange={e => setIndex(Number(e.target.value))} /></label>
      <div className="readout"><span>Origin <b>{selected.originDate}</b></span><span>Target <b>{selected.targetDate}</b></span><span>Refit block <b>{selected.refitDate.slice(0,7)}</b></span><span>Actual <b>{f(selected.actual,3)} pp</b></span><span>Persistence <b>{f(selected.persistence,3)} pp</b></span><span>RF <b>{f(selected[model],3)} pp</b></span></div>
    </div>
    <p className="source-note">Guides mark the first prediction in each annual block ({research.meta.forecast.refitBlocks} in the full sample). The inspector shows the nominal prior-December fold boundary; training has a three-month embargo. Each point is archived. The final August 2022 target is a partial month. No uncertainty bands are available.</p>
    <Evidence source={['research/evidence/preds_macro+spread_history.csv','research/evidence/preds_macro_only.csv','analysis/honest_eval.py']} target="Credit_Spread(t+1), legacy forward-filled monthly mean" dates={`Displayed origins ${rows[0].originDate}–${rows[rows.length - 1].originDate}; targets ${rows[0].targetDate}–${rows[rows.length - 1].targetDate}`} n={`${rows.length} displayed forecasts`} horizon="1 month" benchmark="Monthly persistence: ŷ(t+1) = spread(t)" status="Reproduced exploratory" />
    <div className="figure-header spaced"><div><span className="eyebrow">02 / Compatible error comparisons</span><h3>A positive R² is only part of the answer.</h3></div></div>
    <div className="table-wrap"><table><caption>Same 163 origins. Error units are pp. Change targets use zero change as persistence.</caption><thead><tr><th>Specification / trained target</th><th>RMSE ↓</th><th>MAE ↓</th><th>R² / test mean</th><th>R² / persistence</th></tr></thead><tbody>{forecastMetrics.map(m => <tr key={m.id} className={m.r2Persistence === 0 ? 'baseline-row' : ''}><th><span>{m.label}</span><small>{m.target === 'next_month_legacy_change' ? 'Spread change · rounded stored log' : 'Future level · stored prediction records'} · <Source path={m.source}>source</Source></small></th><td><div className="metric-cell"><span>{f(m.rmse)}</span><i style={{ width: `${m.rmse / 2.4004 * 100}%` }} /></div></td><td>{m.mae == null ? 'Not stored' : f(m.mae)}</td><td>{m.r2Mean == null ? 'Not stored' : f(m.r2Mean)}</td><td>{f(m.r2Persistence)}</td></tr>)}</tbody></table></div>
    <p className="source-note">The change-model logs do not retain prediction paths or MAE/test-mean R²; those fields stay unavailable. Comparing RMSE is valid here because forecast origins match and level errors equal change errors after adding the known origin spread.</p>
    <details className="technical"><summary>Read the metrics and the evaluation design</summary><div className="technical-body research-grid">
      <div><h4>Error in the target’s units</h4><p>MAE averages absolute errors. RMSE squares errors first and penalizes large misses more heavily. Both are in percentage points; MSE is in squared percentage points.</p><code className="formula">RMSE = √mean((actual − prediction)²)</code></div>
      <div><h4>Two different reference forecasts</h4><p>Ordinary R² compares squared error to the realized test-period mean. Out-of-sample R² here compares it to monthly persistence. The macro + spread RF has R² ≈ 0.7503 against the mean and −2.3037 against persistence: it clears one reference and loses to the stronger one.</p><code className="formula">R² vs persistence = 1 − SSE(model) / SSE(persistence)</code></div>
      <div><h4>Walk-forward, with remaining timing limits</h4><p>Expanding training windows, annual refits, a three-month embargo and lag approximations on revised macro data. Macro-only future-level models have levels, first and twelve-month differences; change models also use three-month macro differences. Feature sets differ. This is not a historical-vintage reconstruction.</p></div>
      <div><h4>Scope of the conclusion</h4><p>These five learned specifications lose on this sample. A separate stored 2019–2022 AR(1) comparison is a different experiment. Dependent-data uncertainty, chronological scaled Ridge tuning and a matched endpoint-information regime study remain unfinished.</p><Source path="research/paper_preparation/04_VERIFIED_FINDINGS.md">Evaluation scope</Source></div>
    </div></details>
    <Evidence source={['research/evidence/honest_eval.log','research/evidence/fair_eval.log']} target="Next-month legacy mean or its one-month change, identified per row" dates="Origins Jan 2009–Jul 2022; targets Feb 2009–Aug 2022" n="163 per specification" horizon="1 month" benchmark="Monthly persistence / zero change" status="Reproduced exploratory · change rows log only" />
  </>
}

export function AggregationComparison() {
  const [target, setTarget] = useState<'next_month_panel_ffill_mean' | 'next_month_observed_daily_mean'>('next_month_panel_ffill_mean')
  const rows = aggregationMetrics.filter(r => r.target === target && r.n === 163)
  const mean = rows.find(r => r.predictor === 'mean')!, last = rows.find(r => r.predictor === 'last')!
  const improvement = 100 * (1-last.rmse/mean.rmse)
  const march = research.march2020Daily
  return <div className="subsection">
    <div className="figure-header"><div><span className="eyebrow">03 / Information at the forecast origin</span><h3>The latest observation changes the benchmark.</h3></div></div>
    <p>The monthly mean and the last daily observation summarize different information. Keep the target and the 163 origins fixed to compare them.</p>
    <label className="field inline-field">Target aggregation<select className="control" value={target} onChange={e => setTarget(e.target.value as typeof target)}><option value="next_month_panel_ffill_mean">Legacy forward-filled mean</option><option value="next_month_observed_daily_mean">Observed-day-only mean</option></select></label>
    <div className="benchmark-comparison"><div><span>Previous monthly mean</span><strong>{f(mean.rmse,6)} <small>pp RMSE</small></strong></div><div><span>Latest daily spread</span><strong>{f(last.rmse,6)} <small>pp RMSE</small></strong></div><div className="accent"><span>Within-target RMSE reduction</span><strong>{f(improvement,2)}<small>%</small></strong></div></div>
    <p className="callout">Exploratory comparison. It establishes neither statistical significance nor live availability. Learned models still need the same latest-spread information before any regime contribution can be assessed.</p>
    <Evidence source="research/evidence/aggregation_benchmarks.csv" target={target === 'next_month_panel_ffill_mean' ? 'Next-month legacy forward-filled mean' : 'Next-month observed-day-only mean'} dates="Origins Jan 2009–Jul 2022; targets Feb 2009–Aug 2022" n="163 forecasts" horizon="1 month" benchmark="Previous monthly mean versus latest daily spread" status="Verified exploratory benchmark" />
    <details className="technical"><summary>March 2020: what averaging compresses</summary><div className="technical-body">
      <ResearchPlot dates={march.map(r => r.date)} daily label="Observed daily spreads in March 2020 and their observed-day monthly mean" series={[{label:'Daily observed spread',color:'#f1f0eb',values:march.map(r => r.value)},{label:'Observed-day mean',color:'#9eaab9',values:march.map(() => research.aggregationExample.observedMean),dashed:true}]} />
      <div className="readout"><span>Observed mean <b>{f(research.aggregationExample.observedMean,4)} pp</b></span><span>Daily maximum <b>10.87 pp</b></span><span>Last observation <b>8.77 pp</b></span><span>Within-month range <b>6.12 pp</b></span></div>
      <p>Range measures within-month dispersion, not the speed or direction of repricing. The mean, maximum and last observation answer different questions.</p>
      <Evidence source="research/evidence/audit_summary.json" target="Observed daily BAMLH0A0HYM2 and observed-day mean" dates="March 2020" n="22 observed daily values" />
    </div></details>
  </div>
}

export function StressClassification() {
  const [horizon, setHorizon] = useState(1)
  const row = classificationMetrics.find(r => r.horizonMonths === horizon)!
  return <div className="subsection">
    <div className="figure-header"><div><span className="eyebrow">04 / Stress-state diagnostic</span><h3>Ranking future stress is not warning of a new crisis.</h3></div><label className="field">Inspect horizon<select className="control" value={horizon} onChange={e => setHorizon(Number(e.target.value))}>{classificationMetrics.map(r => <option key={r.horizonMonths} value={r.horizonMonths}>{r.horizonMonths} month{r.horizonMonths > 1 ? 's' : ''}</option>)}</select></label></div>
    <p>After excluding unavailable future outcomes, raw current spread ranks future stress better than the RF score at every inspected horizon. The diagnostic retains a full-sample stress threshold.</p>
    <div className="ranking-chart" aria-label={`At ${horizon} months, RF AUC ${f(row.rfAuc)} versus current spread AUC ${f(row.currentSpreadAuc)}`}>
      {[{label:'Random Forest score',v:row.rfAuc,color:'#9eaab9'},{label:'Raw current-spread score',v:row.currentSpreadAuc,color:'#9aafa5'}].map(r => <div className="rank-bar" key={r.label}><span>{r.label}</span><div><i style={{width:`${r.v*100}%`,background:r.color}} /></div><b>{f(r.v)} <small>AUC</small></b></div>)}
      <p className="source-note">Bar scale 0–1 · AUC measures ranking discrimination; AP summarizes precision across recall levels.</p>
    </div>
    <div className="table-wrap"><table><caption>Corrected known-outcome rows only · RF versus raw current spread</caption><thead><tr><th>Horizon</th><th>n / positive</th><th>Unknown excluded</th><th>RF AUC / AP</th><th>Current spread AUC / AP</th></tr></thead><tbody>{classificationMetrics.map(r => <tr key={r.horizonMonths} className={r.horizonMonths === horizon ? 'baseline-row' : ''}><th>{r.horizonMonths} month{r.horizonMonths > 1 ? 's' : ''}</th><td>{r.n} / {r.positiveMonths}</td><td>{r.unknownFutureExcluded}</td><td>{f(r.rfAuc)} / {f(r.rfAveragePrecision)}</td><td>{f(r.currentSpreadAuc)} / {f(r.currentSpreadAveragePrecision)}</td></tr>)}</tbody></table></div>
    <p className="source-note">Threshold: future legacy spread ≥ {f(row.threshold,6)} pp (full-sample 75th percentile). A raw spread ranking is not a calibrated probability. Unknown labels are excluded rather than assigned zero; samples shrink with horizon. Differences have no completed uncertainty estimates. This RF diagnostic uses annual refits with a 12-month embargo, distinct from the regression evaluation.</p>
    <Evidence source={['research/evidence/classification_label_diagnostic.csv','research/evidence/classification_diagnostic_predictions.csv']} target="Future occupancy at or above full-sample legacy-spread q75" dates={`Selected horizon: origins ${row.originStart}–${row.originEnd}; targets ${row.targetStart}–${row.targetEnd}`} n={`${row.n} origins; ${row.positiveMonths} positive months; ${row.unknownFutureExcluded} unknown excluded`} units="AUC and AP (unitless); threshold in pp" horizon={`${horizon} months`} benchmark="Raw current-spread ranking" status="Corrected exploratory diagnostic" />
    <div className="entry-note"><span className="large-number">03</span><div><h4>Entries under a threshold frozen through 2008</h4><p>September 2011 · January 2016 · March 2020</p><p className="source-note">Legacy q75 = {f(research.meta.thresholds.spreadFrozen2008,6)} pp. Three post-2008 entries are verified counts, not three successful warnings. A completed frozen-threshold event forecast does not exist.</p></div></div>
    <Evidence source="research/evidence/onset_counts.csv" target="Entry from below to at or above frozen legacy-spread q75" dates="Post-2008 through Aug 2022; threshold estimated through Dec 2008" n="3 distinct entries" units="Distinct entries (count); threshold in pp" status="Verified event counts · forecasting pending" />
  </div>
}
