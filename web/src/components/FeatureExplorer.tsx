import { useMemo, useState } from 'react'
import { features, type FeatureRecord } from '@/lib/researchContent'
import { sourceHref } from '@/lib/research'

const kinds = ['All types', 'Raw', 'Engineered', 'Regime output', 'Target'] as const

function FeatureDetail({ feature }: { feature: FeatureRecord }) {
  return (
    <article className="feature-detail" aria-labelledby="selected-feature-name">
      <span className="badge">{feature.kind} · {feature.status}</span>
      <h4 id="selected-feature-name"><code>{feature.code}</code></h4>
      <p>{feature.meaning}</p>
      <div className="formula"><code>{feature.formula}</code></div>
      <dl className="research-grid">
        <div><dt>Source series</dt><dd>{feature.series}</dd></div>
        <div><dt>Lag / window</dt><dd>{feature.window}</dd></div>
        <div><dt>Experiment</dt><dd>{feature.experiment}</dd></div>
        <div><dt>Timing & interpretation</dt><dd>{feature.timing}</dd></div>
      </dl>
      <p className="source-note">Evidence: <a href={sourceHref(feature.source)} target="_blank" rel="noreferrer">{feature.source}</a></p>
    </article>
  )
}

export function FeatureExplorer() {
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState<string>('All types')
  const [selectedId, setSelectedId] = useState('spread')
  const filtered = useMemo(() => {
    const search = query.trim().toLowerCase()
    return features.filter((feature) => (kind === 'All types' || feature.kind === kind) &&
      [feature.code, feature.meaning, feature.series, feature.family, feature.experiment, feature.status]
        .some((text) => text.toLowerCase().includes(search)))
  }, [query, kind])
  const selected = filtered.find((feature) => feature.id === selectedId) ?? filtered[0]

  return (
    <div>
      <div className="controls">
        <label className="field" htmlFor="feature-search">Search code, source, or experiment
          <input className="control" id="feature-search" type="search" placeholder="e.g. CPI, rolling, GMM…" value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <label className="field" htmlFor="feature-kind">Feature type
          <select className="control" id="feature-kind" value={kind} onChange={(event) => setKind(event.target.value)}>
            {kinds.map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
      </div>
      <p className="source-note" aria-live="polite">{filtered.length} matching definitions. Feature sets differ by experiment.</p>
      {selected ? <div className="feature-layout">
        <div className="feature-list" role="group" aria-label="Choose a feature">
          {filtered.map((feature) => (
            <button type="button" key={feature.id} aria-pressed={feature.id === selected.id} onClick={() => setSelectedId(feature.id)}>
              <code>{feature.code}</code><span>{feature.family}</span>
            </button>
          ))}
        </div>
        <FeatureDetail feature={selected} />
      </div> : <p role="status">No feature matches this search. Try a source code, macro name, or another feature type.</p>}


    </div>
  )
}
