import type { ReactNode } from 'react'
import { sourceHref } from '@/lib/research'

export function Source({ path, children }: { path: string; children?: ReactNode }) {
  return <a href={sourceHref(path)} target="_blank" rel="noreferrer">{children ?? path.split('/').pop()} ↗</a>
}

export function Evidence({ source, target, dates, n, horizon = 'Descriptive · no forecast horizon', benchmark = 'Not applicable', status = 'Verified descriptive', units = 'Percentage points (pp)' }: {
  source: string | string[]; target: string; dates: string; n: string | number; horizon?: string; benchmark?: string; status?: string; units?: string
}) {
  return <details className="evidence">
    <summary><span className="status-dot" />{status}<span className="evidence-summary"> · {n} · {dates}</span><span className="evidence-action">Evidence +</span></summary>
    <dl className="evidence-grid">
      <div><dt>Target / series</dt><dd>{target}</dd></div><div><dt>Units</dt><dd>{units}</dd></div>
      <div><dt>Dates</dt><dd>{dates}</dd></div><div><dt>Sample</dt><dd>{n}</dd></div>
      <div><dt>Horizon</dt><dd>{horizon}</dd></div><div><dt>Benchmark</dt><dd>{benchmark}</dd></div>
      <div className="evidence-sources"><dt>Source artifacts</dt><dd>{(Array.isArray(source) ? source : [source]).map(p => <Source key={p} path={p}>{p}</Source>)}</dd></div>
    </dl>
  </details>
}
