import { useMemo, useState } from 'react'
import { experiments, regimeMethods, type ExperimentRecord } from '@/lib/researchContent'
import { sourceHref } from '@/lib/research'

const purposes = ['Historical regression', 'Sequence experiments', 'Regime estimation', 'Later evaluation', 'Proposed work'] as const

function ExperimentDetail({ experiment, technical }: { experiment: ExperimentRecord; technical: boolean }) {
  return <details className="technical" open={technical || undefined}>
    <summary>{experiment.name} <span className="badge">{experiment.status}</span></summary>
    <p>{experiment.metrics}</p>
    <dl className="research-grid">
      <div><dt>Target & units</dt><dd>{experiment.target}</dd></div>
      <div><dt>Evaluation dates</dt><dd>{experiment.window}</dd></div>
      <div><dt>Sample</dt><dd>{experiment.sample}</dd></div>
      <div><dt>Horizon</dt><dd>{experiment.horizon}</dd></div>
      <div><dt>Inputs</dt><dd>{experiment.features}</dd></div>
      <div><dt>Benchmark / comparison</dt><dd>{experiment.benchmark}</dd></div>
    </dl>
    <p><strong>What the evidence supports.</strong> {experiment.limitation}</p>
    {experiment.id === 'sarimax' && <>
      <h4>How this specification works</h4>
      <p>After accounting for the exogenous inputs, <strong>AR (1)</strong> relates the differenced residual component to its previous value. <strong>Differencing (1)</strong> models month-to-month changes in that residual component. <strong>MA (1)</strong> accounts for the previous forecast innovation. The <strong>exogenous inputs</strong> add the five listed macro features. <code>seasonal_order=(0,0,0,0)</code> adds no seasonal terms.</p>
      <p>The notebook calls <code>order=(1,1,1)</code> an example. It does not establish that this order is optimal. Its forecast call supplies <code>exog_future=exog_test</code>, including macro values realized throughout the test period. This is a conditional historical forecast rather than a fully ex-ante path from December 2018.</p>
      <p>The stored summary reports coefficient standard errors and confidence intervals. Those describe uncertainty about fitted coefficients; they are not forecast intervals. No confidence band is manufactured here.</p>
      <p>Training diagnostics report non-normal and heteroskedastic residuals. Poor test metrics alone do not identify a single cause such as “nonlinearity” or justify a specific replacement model.</p>
    </>}
    <p className="source-note">Source: <a href={sourceHref(experiment.source)} target="_blank" rel="noreferrer">{experiment.source}</a>{experiment.sourceDetail && <> · {experiment.sourceDetail}</>}</p>
  </details>
}

export function ExperimentExplorer({ technical = false }: { technical?: boolean }) {
  const [purpose, setPurpose] = useState<string>('Historical regression')
  const [query, setQuery] = useState('')
  const selected = useMemo(() => experiments.filter((experiment) =>
    (purpose === 'All experiments' || experiment.purpose === purpose) &&
    [experiment.name, experiment.features, experiment.status, experiment.target].some((text) => text.toLowerCase().includes(query.trim().toLowerCase()))
  ), [purpose, query])

  return <div>
    <p>Open an experiment to inspect its target, inputs, evaluation window and evidence status. Historical scores remain separate from the reproduced benchmark evaluation.</p>
    <div className="controls">
      <label className="field" htmlFor="experiment-purpose">Research purpose
        <select className="control" id="experiment-purpose" value={purpose} onChange={(event) => setPurpose(event.target.value)}>
          {purposes.map((item) => <option key={item}>{item}</option>)}
          <option>All experiments</option>
        </select>
      </label>
      <label className="field" htmlFor="experiment-search">Search models or inputs
        <input className="control" id="experiment-search" type="search" placeholder="e.g. SARIMAX, LSTM, regime…" value={query} onChange={(event) => { setQuery(event.target.value); if (event.target.value) setPurpose('All experiments') }} />
      </label>
    </div>
    <p className="source-note" aria-live="polite">{selected.length} experiments shown. MSE uses pp²; MAE and RMSE use pp. One percentage point = 100 basis points.</p>
    <div className="experiment-list">
      {selected.map((experiment) => <ExperimentDetail key={experiment.id} experiment={experiment} technical={technical} />)}
      {!selected.length && <p role="status">No experiment matches this selection.</p>}
    </div>
  </div>
}

export function RegimeMethods() {
  return <details className="technical">
    <summary>Three different definitions of a “regime”</summary>
    {regimeMethods.map((method) => <article key={method.name}>
      <h4>{method.name}</h4>
      <p>{method.description}</p>
      <p>{method.timing}</p>
      <p className="source-note">Source: <a href={sourceHref(method.source)} target="_blank" rel="noreferrer">{method.source}</a></p>
    </article>)}
    <p>NBER recession periods are a separate historical reference. Cluster membership, macro deterioration, high spreads, recession and entry into a new stress episode are different objects.</p>
  </details>
}
