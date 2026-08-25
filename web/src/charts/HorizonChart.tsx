/** Insight 5: how far ahead the problem is answerable.
 *  Two linked views of the same walk-forward classification — ranking ability
 *  (AUC) and practical value (precision lift). The metric toggle matters because
 *  AUC alone flatters a rare-event problem. */

import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line } from 'd3-shape'
import { horizon, type FeatureSet } from '@/lib/data'
import { num } from '@/lib/format'
import { AxisBottom, AxisLeft, C, Figure, GridLines, TipRow, Tooltip, useMeasure } from './primitives'

const SETS: { id: FeatureSet; label: string; color: string }[] = [
  { id: 'spreadOnly',  label: 'Spread history only', color: C.signal },
  { id: 'macroSpread', label: 'Macro + spread',      color: C.gold },
  { id: 'macroOnly',   label: 'Macro only',          color: C.calm },
]
type Metric = 'auc' | 'lift'

export function HorizonChart() {
  const [ref, w] = useMeasure<HTMLDivElement>()
  const [metric, setMetric] = useState<Metric>('auc')
  const [hover, setHover] = useState<number | null>(null)

  const height = w < 640 ? 250 : 285
  const m = { top: 22, right: w < 640 ? 14 : 20, bottom: 42, left: w < 640 ? 38 : 46 }
  const iw = Math.max(10, w - m.left - m.right), ih = Math.max(10, height - m.top - m.bottom)
  const H = horizon.h

  const x = useMemo(() => scaleLinear().domain([0, 3]).range([m.left, m.left + iw]), [m.left, iw])
  const y = useMemo(
    () => metric === 'auc'
      ? scaleLinear().domain([0.4, 1]).range([m.top + ih, m.top])
      : scaleLinear().domain([0, 5]).range([m.top + ih, m.top]),
    [metric, m.top, ih],
  )
  const base = metric === 'auc' ? 0.5 : 1
  const data = horizon[metric]

  return (
    <Figure
      title="Credit stress is predictable three months out, and not at all at twelve"
      subtitle={`Walk-forward classification: will the spread sit in its top quartile (≥${horizon.threshold}pp) h months from now? 164 test months.`}
      controls={
        <div className="inline-flex rounded-md border border-rule-strong p-0.5" role="group" aria-label="Metric">
          {([['auc', 'Ranking (AUC)'], ['lift', 'Precision lift']] as const).map(([id, label]) => (
            <button
              key={id} onClick={() => setMetric(id)} aria-pressed={metric === id}
              className={`rounded px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                metric === id ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink'}`}
            >{label}</button>
          ))}
        </div>
      }
      caption={
        <>Adding the six macro indicators to the spread's own history <em>lowers</em> AUC at every horizon —
        they contribute noise, not signal. Switch to <em>Precision lift</em> for the practical reading: at one
        month the classifier is 4.4× more precise than the base rate; by twelve months every feature set has
        converged on doing nothing useful.</>
      }
      source="Random-forest classifiers, expanding walk-forward origins with a 12-month embargo and balanced class weights. Lift = average precision ÷ base rate. The 6.43pp threshold is the in-sample 75th percentile and would have to be set on training data alone in production."
    >
      <div ref={ref} className="relative select-none">
        {w > 0 && (
          <svg width={w} height={height} className="chart-svg block overflow-visible" role="img"
               aria-label={`Classification ${metric === 'auc' ? 'AUC' : 'precision lift'} by forecast horizon`}>
            <rect x={x(1)} y={m.top} width={x(2) - x(1)} height={ih} fill={C.signal} opacity={0.045} />
            <text x={(x(1) + x(2)) / 2} y={m.top + ih - 10} textAnchor="middle" fontSize={10}
                  fontWeight={600} fill={C.signal}>usable signal ends</text>

            <GridLines scale={y} ticks={y.ticks(5)} x0={m.left} x1={m.left + iw} />
            <AxisLeft scale={y} ticks={y.ticks(5)} x={m.left}
                      format={(v) => metric === 'auc' ? v.toFixed(1) : `${v}×`} />
            <text x={m.left} y={m.top - 8} fontSize={11} fill={C.mute}>
              {metric === 'auc' ? 'AUC — ranking stressed months above calm ones' : 'precision lift over the base rate'}
            </text>

            <line x1={m.left} x2={m.left + iw} y1={y(base)} y2={y(base)}
                  stroke={C.ruleStrong} strokeWidth={1.2} strokeDasharray="4 3" />
            <text x={m.left + 6} y={y(base) - 5} fontSize={10} fill={C.mute}>
              {metric === 'auc' ? 'coin flip' : 'no better than guessing'}
            </text>

            {SETS.map((s) => {
              const vals = data[s.id]
              const p = d3line<number>().x((_, i) => x(i)).y((v) => y(v))(vals) ?? ''
              return (
                <g key={s.id}>
                  <path d={p} fill="none" stroke={s.color} strokeWidth={s.id === 'spreadOnly' ? 2.2 : 1.6}
                        strokeLinejoin="round" strokeLinecap="round"
                        style={{ transition: 'd .4s cubic-bezier(.22,.61,.36,1)' }} />
                  {vals.map((v, i) => (
                    <circle key={i} cx={x(i)} cy={y(v)} r={hover === i ? 5 : 3.6} fill={s.color}
                            stroke={C.paper} strokeWidth={1.5}
                            style={{ transition: 'cy .4s cubic-bezier(.22,.61,.36,1), r .18s' }} />
                  ))}
                </g>
              )
            })}

            <AxisBottom scale={x} ticks={[0, 1, 2, 3]} y={m.top + ih} format={(i) => `${H[i]}`} />
            <text x={m.left + iw / 2} y={height - 6} textAnchor="middle" fontSize={11} fill={C.mute}>
              forecast horizon, months ahead
            </text>

            {H.map((_, i) => (
              <rect key={i} x={x(i) - iw / 8} y={m.top} width={iw / 4} height={ih} fill="transparent"
                    onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)} />
            ))}
          </svg>
        )}
        {hover != null && (
          <Tooltip x={x(hover)} y={10} width={w}>
            <div className="mb-1 text-2xs font-semibold text-ink">{H[hover]} month{H[hover] > 1 ? 's' : ''} ahead</div>
            {SETS.map((s) => (
              <TipRow key={s.id} label={s.label}
                      value={metric === 'auc' ? num(data[s.id][hover], 3) : `${num(data[s.id][hover], 2)}×`}
                      color={s.color} strong={s.id === 'spreadOnly'} />
            ))}
          </Tooltip>
        )}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-2xs text-ink-mute">
        {SETS.map((s) => (
          <span key={s.id} className="flex items-center gap-1.5">
            <span className="h-[2.5px] w-4 rounded-full" style={{ background: s.color }} />{s.label}
          </span>
        ))}
      </div>
    </Figure>
  )
}
