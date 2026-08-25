/** Insight 2: which series moves first.
 *  Bars are the mean |correlation| across all six indicators at each lead; the
 *  faint lines behind are the individual indicators, so you can check the average
 *  is not hiding disagreement. Selecting an indicator isolates it. */

import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line, curveMonotoneX } from 'd3-shape'
import { xcorr, granger, MACRO_META, MACRO_KEYS, type MacroKey } from '@/lib/data'
import { num, pval } from '@/lib/format'
import { AxisLeft, C, Figure, GridLines, MACRO_COLOR, TipRow, Tooltip, useMeasure } from './primitives'

export function LeadLagChart() {
  const [ref, w] = useMeasure<HTMLDivElement>()
  const [sel, setSel] = useState<MacroKey | 'mean'>('mean')
  const [hover, setHover] = useState<number | null>(null)

  const height = w < 640 ? 250 : 290
  const m = { top: 30, right: w < 640 ? 12 : 18, bottom: 44, left: w < 640 ? 40 : 48 }
  const iw = Math.max(10, w - m.left - m.right), ih = Math.max(10, height - m.top - m.bottom)
  const lags = xcorr.lags

  const x = useMemo(
    () => scaleLinear().domain([lags[0] - 0.6, lags[lags.length - 1] + 0.6]).range([m.left, m.left + iw]),
    [lags, m.left, iw],
  )
  const y = useMemo(() => scaleLinear().domain([0, 0.45]).range([m.top + ih, m.top]), [m.top, ih])

  const values = sel === 'mean'
    ? xcorr.mean
    : xcorr.series.find((s) => s.key === sel)!.values.map((v) => Math.abs(v))

  const bw = Math.max(4, (iw / lags.length) * 0.62)
  const peakIdx = values.indexOf(Math.max(...values))

  return (
    <Figure
      title="The relationship peaks one month after the spread has already moved"
      subtitle="Correlation between each macro indicator's monthly change and the spread's monthly change, at leads from −6 to +6 months. Negative leads mean the spread moved first."
      controls={
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Indicator">
          <button
            onClick={() => setSel('mean')} aria-pressed={sel === 'mean'}
            className={`rounded-md border px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
              sel === 'mean' ? 'border-ink bg-ink text-paper' : 'border-rule-strong bg-paper text-ink-soft hover:border-ink-faint hover:text-ink'}`}
          >All six</button>
          {MACRO_KEYS.map((k) => (
            <button
              key={k} onClick={() => setSel(k)} aria-pressed={sel === k}
              className={`rounded-md border px-2 py-1 text-2xs font-medium transition-colors duration-200 ${
                sel === k ? 'border-ink bg-ink text-paper' : 'border-rule-strong bg-paper text-ink-soft hover:border-ink-faint hover:text-ink'}`}
            >{MACRO_META[k].short}</button>
          ))}
        </div>
      }
      caption={
        <>Averaged across all six indicators, the correlation on the spread-moved-first side is{' '}
        <strong className="font-semibold text-ink">{num(xcorr.spreadFirstSide, 3)}</strong> against{' '}
        <strong className="font-semibold text-ink">{num(xcorr.macroFirstSide, 3)}</strong> on the
        macro-moved-first side. Every indicator individually shows the same tilt.</>
      }
    >
      <div ref={ref} className="relative select-none">
        {w > 0 && (
          <svg width={w} height={height} className="chart-svg block overflow-visible" role="img"
               aria-label="Cross-correlation by lead, showing correlation concentrated where the spread moves first">
            <rect x={x(lags[0] - 0.6)} y={m.top} width={x(-0.5) - x(lags[0] - 0.6)} height={ih}
                  fill={C.signal} opacity={0.05} />
            <rect x={x(0.5)} y={m.top} width={x(lags[lags.length - 1] + 0.6) - x(0.5)} height={ih}
                  fill={C.calm} opacity={0.05} />
            <GridLines scale={y} ticks={y.ticks(4)} x0={m.left} x1={m.left + iw} />
            <AxisLeft scale={y} ticks={y.ticks(4)} x={m.left} format={(v) => v.toFixed(1)} />
            <text x={m.left} y={m.top - 16} fontSize={11} fill={C.mute}>|correlation| with monthly spread change</text>

            <text x={x(-3.25)} y={m.top + 12} textAnchor="middle" fontSize={11} fontWeight={600} fill={C.signal}>
              spread moves first
            </text>
            <text x={x(3.25)} y={m.top + 12} textAnchor="middle" fontSize={11} fontWeight={600} fill={C.calm}>
              macro moves first
            </text>

            {lags.map((k, i) => {
              const h = Math.max(0, y(0) - y(values[i]))
              const active = hover === i
              return (
                <rect key={k} x={x(k) - bw / 2} y={y(values[i])} width={bw} height={h} rx={1.5}
                      fill={k < 0 ? C.signal : k === 0 ? C.mute : C.calm}
                      opacity={active ? 1 : i === peakIdx ? 0.95 : 0.75}
                      style={{ transition: 'y .35s cubic-bezier(.22,.61,.36,1), height .35s cubic-bezier(.22,.61,.36,1)' }} />
              )
            })}

            {sel === 'mean' && xcorr.series.map((s) => {
              const p = d3line<number>().x((_, i) => x(lags[i])).y((v) => y(Math.abs(v)))
                .curve(curveMonotoneX)(s.values) ?? ''
              return <path key={s.key} d={p} fill="none" stroke={MACRO_COLOR[s.key]} strokeWidth={0.9} opacity={0.4} />
            })}

            <line x1={x(0)} x2={x(0)} y1={m.top} y2={m.top + ih} stroke={C.ruleStrong} strokeWidth={1} strokeDasharray="2 3" />
            <line x1={m.left} x2={m.left + iw} y1={y(0)} y2={y(0)} className="axis-line" strokeWidth={1} />
            {lags.map((k) => (
              <text key={k} x={x(k)} y={y(0) + 16} textAnchor="middle" fontSize={10}
                    fontWeight={k === -1 ? 700 : 400} fill={k === -1 ? C.signal : C.mute}>{k}</text>
            ))}
            <text x={m.left + iw / 2} y={height - 6} textAnchor="middle" fontSize={11} fill={C.mute}>
              lead k, in months
            </text>

            {lags.map((k, i) => (
              <rect key={`h${k}`} x={x(k) - (iw / lags.length) / 2} y={m.top}
                    width={iw / lags.length} height={ih} fill="transparent"
                    onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)} />
            ))}
          </svg>
        )}
        {hover != null && (
          <Tooltip x={x(lags[hover])} y={10} width={w}>
            <div className="mb-1 text-2xs font-semibold text-ink">
              {lags[hover] === 0 ? 'Same month' : lags[hover] < 0
                ? `Spread moved ${Math.abs(lags[hover])}m first`
                : `Macro moved ${lags[hover]}m first`}
            </div>
            <TipRow label={sel === 'mean' ? 'Mean |r|, six indicators' : `${MACRO_META[sel].short} |r|`}
                    value={num(values[hover], 3)}
                    color={sel === 'mean' ? (lags[hover] < 0 ? C.signal : C.calm) : MACRO_COLOR[sel]} strong />
          </Tooltip>
        )}
      </div>

      <div className="mt-6">
        <h5 className="mb-2.5 text-2xs font-semibold uppercase tracking-[0.14em] text-ink-mute">
          Granger tests, both directions
        </h5>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[26rem] border-collapse text-left">
            <thead>
              <tr className="border-b border-rule-strong">
                <th className="py-2 pr-3 text-2xs font-semibold text-ink-mute">Indicator</th>
                <th className="py-2 pr-3 text-right text-2xs font-semibold text-ink-mute">Spread → macro</th>
                <th className="py-2 pr-3 text-right text-2xs font-semibold text-ink-mute">Macro → spread</th>
                <th className="py-2 text-right text-2xs font-semibold text-ink-mute">Reads as</th>
              </tr>
            </thead>
            <tbody>
              {granger.map((g) => {
                const a = g.spreadToMacro < 0.05, b = g.macroToSpread < 0.05
                const verdict = a && !b ? 'Spread leads' : b && !a ? 'Macro leads' : a && b ? 'Both directions' : 'Neither'
                return (
                  <tr key={g.key} className="border-b border-rule">
                    <td className="py-2 pr-3 text-[0.8rem] text-ink">{g.label}</td>
                    <td className={`tabular py-2 pr-3 text-right text-2xs ${a ? 'font-semibold text-ink' : 'text-ink-faint'}`}>{pval(g.spreadToMacro)}</td>
                    <td className={`tabular py-2 pr-3 text-right text-2xs ${b ? 'font-semibold text-ink' : 'text-ink-faint'}`}>{pval(g.macroToSpread)}</td>
                    <td className="py-2 text-right text-2xs">
                      <span className={`rounded px-1.5 py-0.5 font-medium ${
                        verdict === 'Spread leads' ? 'bg-signal-soft text-signal-deep'
                        : verdict === 'Both directions' ? 'bg-gold-soft text-gold'
                        : 'bg-paper-sunk text-ink-mute'}`}>{verdict}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2.5 text-2xs leading-relaxed text-ink-faint">
          Tested on first differences with up to six lags. Three indicators are significant in both directions;
          what never happens is the reverse — in neither half of the sample does any indicator lead the spread
          without the spread also leading it. Granger tests establish predictive precedence, not causation.
        </p>
      </div>
    </Figure>
  )
}
