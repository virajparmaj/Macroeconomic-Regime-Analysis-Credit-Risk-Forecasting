/** Insight 1: the same forecasts scored two ways.
 *  Toggling model lines on and off is the point — with the random walk visible
 *  it is immediately clear that both models *follow* the spread rather than
 *  anticipating it. */

import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line, curveMonotoneX } from 'd3-shape'
import { preds, modelMetrics } from '@/lib/data'
import { monthLabel, num, pval, toYear } from '@/lib/format'
import {
  AxisBottom, AxisLeft, C, Figure, GridLines, PointerLayer, TipRow, Tooltip, useMeasure,
} from './primitives'

type SeriesId = 'rw' | 'macroSpread' | 'macroOnly'
type Zoom = 'all' | 'covid' | 'taper'
const ZOOMS: { id: Zoom; label: string; from: string; to: string }[] = [
  { id: 'all',   label: 'All 163 months', from: '2000-01', to: '2030-01' },
  { id: 'covid', label: 'COVID 2020',     from: '2019-06', to: '2021-06' },
  { id: 'taper', label: '2015–16 oil',    from: '2015-01', to: '2016-12' },
]
const SERIES: { id: SeriesId; label: string; color: string; dash?: string }[] = [
  { id: 'rw',          label: 'Random walk',      color: C.moss, dash: '5 3' },
  { id: 'macroSpread', label: 'RF macro + spread',color: C.gold },
  { id: 'macroOnly',   label: 'RF macro only',    color: C.calm },
]

export function BenchmarkChart() {
  const [ref, w] = useMeasure<HTMLDivElement>()
  const [on, setOn] = useState<Record<SeriesId, boolean>>({ rw: true, macroSpread: true, macroOnly: false })
  const [zoom, setZoom] = useState<Zoom>('all')
  const [hover, setHover] = useState<number | null>(null)
  const zd = ZOOMS.find((z) => z.id === zoom)!
  const rows = useMemo(() => preds.filter((p) => p.d >= zd.from && p.d <= zd.to), [zd.from, zd.to])

  const height = w < 640 ? 260 : 300
  const m = { top: 18, right: w < 640 ? 12 : 18, bottom: 28, left: w < 640 ? 34 : 42 }
  const iw = Math.max(10, w - m.left - m.right), ih = Math.max(10, height - m.top - m.bottom)

  const x = useMemo(() => {
    const ys = rows.map((p) => toYear(p.d))
    return scaleLinear().domain([Math.min(...ys), Math.max(...ys)]).range([m.left, m.left + iw])
  }, [rows, m.left, iw])
  const y = useMemo(() => {
    const vals = rows.flatMap((p) => [p.actual, p.rw, on.macroSpread ? p.macroSpread : p.actual, on.macroOnly ? p.macroOnly : p.actual])
    const lo = Math.min(...vals), hi = Math.max(...vals)
    return scaleLinear().domain([zoom === 'all' ? 0 : lo - (hi - lo) * 0.12, hi * 1.06])
      .range([m.top + ih, m.top]).nice()
  }, [rows, zoom, on.macroSpread, on.macroOnly, m.top, ih])

  const mk = (key: 'actual' | SeriesId) =>
    d3line<(typeof rows)[0]>().x((p) => x(toYear(p.d))).y((p) => y(p[key])).curve(curveMonotoneX)(rows) ?? ''

  const yearTicks = useMemo(() => {
    const [a, b] = x.domain(); const step = b - a > 6 ? 2 : 1
    const t: number[] = []
    for (let yr = Math.ceil(a); yr <= b; yr += step) t.push(yr)
    return t.length > 1 ? t : [Math.round(a), Math.round(b)]
  }, [x])

  const hp = hover != null ? rows[hover] : null
  const showMarks = rows.length <= 30

  return (
    <Figure
      title="Both models trace the spread a month late. Neither anticipates it."
      subtitle="Walk-forward forecasts of next month's spread, 163 test months across 14 expanding origins with point-in-time macro alignment."
      controls={
        <div className="flex flex-col items-start gap-2 sm:items-end">
        <div className="inline-flex rounded-md border border-rule-strong p-0.5" role="group" aria-label="Zoom">
          {ZOOMS.map((z) => (
            <button
              key={z.id} onClick={() => { setZoom(z.id); setHover(null) }} aria-pressed={zoom === z.id}
              className={`rounded px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                zoom === z.id ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink'}`}
            >{z.label}</button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Show forecast series">
          {SERIES.map((s) => (
            <button
              key={s.id} onClick={() => setOn((o) => ({ ...o, [s.id]: !o[s.id] }))} aria-pressed={on[s.id]}
              className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                on[s.id] ? 'border-ink-faint bg-paper-sunk text-ink' : 'border-rule bg-paper text-ink-faint hover:text-ink-soft'
              }`}
            >
              <span className="h-[2.5px] w-3.5 rounded-full transition-opacity"
                    style={{ background: s.color, opacity: on[s.id] ? 1 : 0.3 }} />
              {s.label}
            </button>
          ))}
        </div>
        </div>
      }
      caption={
        <>Zoom into <strong className="font-semibold text-ink">COVID 2020</strong> to see the shape of the
        failure: every line turns a month after the black one does. The macro-only model is off by default because it distorts the axis — switch it on to see it call a
        <strong className="font-semibold text-ink"> 14.5pp crisis in July 2020</strong> when the spread was 5.1pp.
        That single month is a good illustration of why average error is not the whole story.</>
      }
    >
      <div ref={ref} className="relative select-none">
        {w > 0 && (
          <svg width={w} height={height} className="chart-svg block overflow-visible" role="img"
               aria-label="Actual credit spread against random-walk and model forecasts">
            <GridLines scale={y} ticks={y.ticks(5)} x0={m.left} x1={m.left + iw} />
            <AxisLeft scale={y} ticks={y.ticks(5)} x={m.left} format={(v) => `${v}`} />
            <text x={m.left} y={m.top - 6} fontSize={11} fill={C.mute}>Spread (pp)</text>
            <AxisBottom scale={x} ticks={yearTicks} y={m.top + ih} format={(v) => `${v}`} />

            {SERIES.filter((s) => on[s.id]).map((s) => (
              <path key={s.id} d={mk(s.id)} fill="none" stroke={s.color} strokeWidth={1.5}
                    strokeDasharray={s.dash} strokeLinejoin="round" opacity={0.92} />
            ))}
            <path d={mk('actual')} fill="none" stroke={C.ink} strokeWidth={2.1} strokeLinejoin="round" />

            {hp && (
              <g pointerEvents="none">
                <line x1={x(toYear(hp.d))} x2={x(toYear(hp.d))} y1={m.top} y2={m.top + ih}
                      stroke={C.ink} strokeWidth={1} opacity={0.3} />
                <circle cx={x(toYear(hp.d))} cy={y(hp.actual)} r={3.4} fill={C.ink} stroke={C.paper} strokeWidth={1.6} />
              </g>
            )}
            {showMarks && rows.map((p) => (
              <circle key={p.d} cx={x(toYear(p.d))} cy={y(p.actual)} r={2.4} fill={C.ink} />
            ))}
            <PointerLayer x0={m.left} y0={m.top} w={iw} h={ih} count={rows.length}
                          onIndex={(i) => setHover(i)} onLeave={() => setHover(null)} />
          </svg>
        )}
        {hp && (
          <Tooltip x={x(toYear(hp.d))} y={8} width={w}>
            <div className="mb-1 text-2xs font-semibold text-ink">{monthLabel(hp.d)}</div>
            <TipRow label="Actual" value={`${num(hp.actual)}pp`} color={C.ink} strong />
            {SERIES.filter((s) => on[s.id]).map((s) => (
              <TipRow key={s.id} label={s.label} value={`${num(hp[s.id])}pp`} color={s.color} />
            ))}
          </Tooltip>
        )}
      </div>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[30rem] border-collapse text-left">
          <thead>
            <tr className="border-b border-rule-strong">
              <th className="py-2 pr-3 text-2xs font-semibold uppercase tracking-wider text-ink-mute">Forecast</th>
              <th className="py-2 pr-3 text-right text-2xs font-semibold uppercase tracking-wider text-ink-mute">RMSE</th>
              <th className="py-2 pr-3 text-right text-2xs font-semibold uppercase tracking-wider text-ink-mute">R² vs mean</th>
              <th className="py-2 pr-3 text-right text-2xs font-semibold uppercase tracking-wider text-ink-mute">R² vs random walk</th>
              <th className="py-2 text-right text-2xs font-semibold uppercase tracking-wider text-ink-mute">Diebold–Mariano</th>
            </tr>
          </thead>
          <tbody>
            {modelMetrics.map((r, i) => (
              <tr key={r.sub} className={`border-b border-rule ${i === 0 ? 'bg-moss-soft/45' : ''}`}>
                <td className="py-2.5 pr-3">
                  <div className="text-[0.8rem] font-medium text-ink">{r.model}</div>
                  <div className="text-2xs text-ink-mute">{r.sub}</div>
                </td>
                <td className="tabular py-2.5 pr-3 text-right text-[0.8rem] font-semibold text-ink">{num(r.rmse)}</td>
                <td className="tabular py-2.5 pr-3 text-right text-[0.8rem] text-ink-soft">{num(r.r2Mean)}</td>
                <td className={`tabular py-2.5 pr-3 text-right text-[0.8rem] font-semibold ${
                  i === 0 ? 'text-ink-mute' : 'text-signal'}`}>
                  {i === 0 ? '— (benchmark)' : num(r.r2Rw)}
                </td>
                <td className="tabular py-2.5 text-right text-2xs text-ink-soft">{i === 0 ? '—' : pval(r.dmP)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2.5 text-2xs leading-relaxed text-ink-faint">
        R² vs random walk = 1 − SSE(model)/SSE(random walk); negative means the model loses. Diebold–Mariano
        uses the Harvey–Leybourne–Newbold small-sample correction. The same ranking holds when the target is
        the monthly <em>change</em>, at horizons of 1, 3, 6 and 12 months, and for Ridge as well as random forest.
      </p>
    </Figure>
  )
}
