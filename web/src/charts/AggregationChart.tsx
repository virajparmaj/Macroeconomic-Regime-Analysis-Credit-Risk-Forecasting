/** Insight 6: what monthly averaging costs.
 *  Switching between crisis windows is the analytical control — it shows the
 *  compression is not a COVID quirk but varies systematically with how fast the
 *  repricing happened. */

import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line, curveMonotoneX } from 'd3-shape'
import { aggregation } from '@/lib/data'
import { dayLabel, dayToYear, monthLabel, num } from '@/lib/format'
import { AxisLeft, C, Figure, GridLines, PointerLayer, TipRow, Tooltip, useMeasure } from './primitives'

type WinId = 'covid' | 'gfc' | 'telecom'
const WINDOWS: { id: WinId; label: string }[] = [
  { id: 'covid', label: 'COVID 2020' },
  { id: 'gfc', label: 'GFC 2008–09' },
  { id: 'telecom', label: 'Telecom 2002' },
]

export function AggregationChart() {
  const [ref, w] = useMeasure<HTMLDivElement>()
  const [win, setWin] = useState<WinId>('covid')
  const [hover, setHover] = useState<number | null>(null)
  const data = aggregation.windows[win]

  const height = w < 640 ? 250 : 290
  const m = { top: 20, right: w < 640 ? 12 : 18, bottom: 30, left: w < 640 ? 34 : 42 }
  const iw = Math.max(10, w - m.left - m.right), ih = Math.max(10, height - m.top - m.bottom)

  const x = useMemo(() => {
    const ys = data.daily.map((d) => dayToYear(d.d))
    return scaleLinear().domain([Math.min(...ys), Math.max(...ys)]).range([m.left, m.left + iw])
  }, [data, m.left, iw])
  const y = useMemo(() => {
    const vs = data.daily.map((d) => d.v)
    const lo = Math.min(...vs), hi = Math.max(...vs)
    const pad = (hi - lo) * 0.14
    return scaleLinear().domain([lo - pad, hi + pad]).range([m.top + ih, m.top]).nice()
  }, [data, m.top, ih])

  const path = d3line<{ d: string; v: number }>()
    .x((p) => x(dayToYear(p.d))).y((p) => y(p.v)).curve(curveMonotoneX)(data.daily) ?? ''

  const peak = data.daily.reduce((a, b) => (b.v > a.v ? b : a))
  const peakMonth = data.monthly.find((mm) => mm.m === peak.d.slice(0, 7))
  const compression = peakMonth ? ((peakMonth.max - peakMonth.mean) / peakMonth.max) * 100 : 0
  const hd = hover != null ? data.daily[hover] : null
  const hoverMonth = hd ? data.monthly.find((mm) => mm.m === hd.d.slice(0, 7)) : null

  return (
    <Figure
      title="The monthly average the project models never sees the worst of the month"
      subtitle="Daily high-yield spread against the monthly mean actually used in the panel. The flat bars are what the model was given."
      controls={
        <div className="inline-flex rounded-md border border-rule-strong p-0.5" role="group" aria-label="Crisis window">
          {WINDOWS.map((o) => (
            <button
              key={o.id} onClick={() => { setWin(o.id); setHover(null) }} aria-pressed={win === o.id}
              className={`rounded px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                win === o.id ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink'}`}
            >{o.label}</button>
          ))}
        </div>
      }
      caption={
        <>Compare the windows: in the slow 2002 repricing the monthly mean sits within about 5% of the month's
        peak, but in March 2020 it sits{' '}
        <strong className="font-semibold text-ink">{num(compression, 0)}% below</strong> it. The faster the
        credit event, the more the averaging hides — which is exactly backwards for a risk system.</>
      }
    >
      <div ref={ref} className="relative select-none">
        {w > 0 && (
          <svg width={w} height={height} className="chart-svg block overflow-visible" role="img"
               aria-label="Daily credit spread against monthly means during a crisis window">
            <GridLines scale={y} ticks={y.ticks(5)} x0={m.left} x1={m.left + iw} />
            <AxisLeft scale={y} ticks={y.ticks(5)} x={m.left} format={(v) => `${v}`} />
            <text x={m.left} y={m.top - 7} fontSize={11} fill={C.mute}>Spread (pp)</text>

            {data.monthly.map((mm) => {
              const [yy, mo] = mm.m.split('-').map(Number)
              const x0 = x(yy + (mo - 1) / 12)
              const x1 = x(yy + mo / 12)
              return (
                <g key={mm.m}>
                  <line x1={Math.max(m.left, x0)} x2={Math.min(m.left + iw, x1)} y1={y(mm.mean)} y2={y(mm.mean)}
                        stroke={C.signal} strokeWidth={3.4} strokeLinecap="round" opacity={0.95} />
                  <line x1={Math.max(m.left, x0)} x2={Math.max(m.left, x0)} y1={m.top} y2={m.top + ih}
                        stroke={C.rule} strokeWidth={1} strokeDasharray="2 4" />
                </g>
              )
            })}

            <path d={path} fill="none" stroke={C.soft} strokeWidth={1.5} strokeLinejoin="round" opacity={0.9} />

            <circle cx={x(dayToYear(peak.d))} cy={y(peak.v)} r={3.4} fill={C.ink} />
            <text x={x(dayToYear(peak.d))} y={y(peak.v) - 10} textAnchor="middle" fontSize={10}
                  fontWeight={600} fill={C.ink}>
              {num(peak.v)}pp
            </text>

            {data.monthly.map((mm) => {
              const [yy, mo] = mm.m.split('-').map(Number)
              const cx = (x(yy + (mo - 1) / 12) + x(yy + mo / 12)) / 2
              if (cx < m.left || cx > m.left + iw || w < 560) return null
              return (
                <text key={mm.m} x={cx} y={m.top + ih + 16} textAnchor="middle" fontSize={9.5} fill={C.faint}>
                  {monthLabel(mm.m).slice(0, 3)}
                </text>
              )
            })}

            {hd && (
              <g pointerEvents="none">
                <line x1={x(dayToYear(hd.d))} x2={x(dayToYear(hd.d))} y1={m.top} y2={m.top + ih}
                      stroke={C.ink} strokeWidth={1} opacity={0.3} />
                <circle cx={x(dayToYear(hd.d))} cy={y(hd.v)} r={3.4} fill={C.soft} stroke={C.paper} strokeWidth={1.5} />
              </g>
            )}
            <PointerLayer x0={m.left} y0={m.top} w={iw} h={ih} count={data.daily.length}
                          onIndex={(i) => setHover(i)} onLeave={() => setHover(null)} />
          </svg>
        )}
        {hd && (
          <Tooltip x={x(dayToYear(hd.d))} y={8} width={w}>
            <div className="mb-1 text-2xs font-semibold text-ink">{dayLabel(hd.d)}</div>
            <TipRow label="Daily spread" value={`${num(hd.v)}pp`} color={C.soft} strong />
            {hoverMonth && (
              <>
                <TipRow label="Month mean (modelled)" value={`${num(hoverMonth.mean)}pp`} color={C.signal} />
                <TipRow label="Month range" value={`${num(hoverMonth.min)} – ${num(hoverMonth.max)}pp`} />
              </>
            )}
          </Tooltip>
        )}
      </div>

      <div className="mt-6 grid gap-6 sm:grid-cols-[1.4fr_1fr]">
        <div>
          <h5 className="mb-2.5 text-2xs font-semibold uppercase tracking-[0.14em] text-ink-mute">
            The eight most violent months, 1997–2022
          </h5>
          <div className="space-y-1.5">
            {aggregation.topRanges.map((r) => {
              const pctW = (r.rng / aggregation.topRanges[0].rng) * 100
              const isCovid = r.era === 'COVID'
              return (
                <div key={r.m} className="flex items-center gap-2.5">
                  <span className={`w-[4.2rem] shrink-0 text-2xs ${isCovid ? 'font-semibold text-ink' : 'text-ink-soft'}`}>
                    {monthLabel(r.m).replace(/(\w{3})\w* /, '$1 ')}
                  </span>
                  <div className="h-3 flex-1 overflow-hidden rounded-[2px] bg-paper-sunk">
                    <div className="h-full rounded-[2px]"
                         style={{ width: `${pctW}%`, background: isCovid ? C.signal : r.era === 'GFC' ? C.gold : C.mute,
                                  opacity: isCovid ? 1 : 0.75, transition: 'width .6s cubic-bezier(.22,.61,.36,1)' }} />
                  </div>
                  <span className={`tabular w-9 shrink-0 text-right text-2xs ${isCovid ? 'font-semibold text-signal' : 'text-ink-mute'}`}>
                    {num(r.rng)}
                  </span>
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-2xs text-ink-faint">Intra-month range: highest minus lowest daily spread, in pp.</p>
        </div>

        <div className="rounded-lg border border-signal/25 bg-signal-soft/40 p-4">
          <div className="eyebrow mb-3 text-signal-deep">March 2020, ranked among {aggregation.nMonths} months</div>
          <dl className="space-y-2.5">
            <RankRow label="By how fast credit repriced" value={`${aggregation.rankByRange}st`} strong />
            <RankRow label="By monthly mean spread" value={`${aggregation.rankByMean}rd`} />
          </dl>
          <p className="mt-3.5 border-t border-signal/20 pt-3 text-2xs leading-relaxed text-ink-soft">
            Its {num(aggregation.topRanges[0].rng)}pp range is{' '}
            <strong className="font-semibold text-ink">
              {num(aggregation.topRanges[0].rng / aggregation.medianRange, 1)}×
            </strong>{' '}
            the median month's {num(aggregation.medianRange)}pp. The panel the project models ranks it
            43rd — an unremarkable bad month.
          </p>
        </div>
      </div>
    </Figure>
  )
}

function RankRow({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-2xs text-ink-soft">{label}</dt>
      <dd className={`tabular shrink-0 ${strong ? 'text-xl font-bold text-signal' : 'text-base font-semibold text-ink-mute'}`}>
        {value}
      </dd>
    </div>
  )
}
