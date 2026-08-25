/** Insight 3: macro is a switch, not a dial.
 *  The regime-definition toggle is the analytical control. Splitting on the
 *  spread's own level is the obvious thing to do and is mildly circular; the
 *  macro-only index avoids that. Both give the same answer, which is the point. */

import { useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { regimeSplit, MACRO_META, type MacroKey } from '@/lib/data'
import { num, pct } from '@/lib/format'
import { C, Figure, useMeasure } from './primitives'

type Def = 'macroOnly' | 'spreadLevel'

export function RegimeSplitChart() {
  const [ref, w] = useMeasure<HTMLDivElement>()
  const [def, setDef] = useState<Def>('macroOnly')
  const v = regimeSplit[def]
  const ordered = [...v.rows].sort((a, b) => Math.abs(b.stress) - Math.abs(a.stress))

  const rowH = w < 640 ? 34 : 38
  const labelW = w < 640 ? 92 : 128
  const barW = Math.max(40, w - labelW - 58)
  const x = scaleLinear().domain([0, 0.65]).range([0, barW])
  const ratio = v.r2Stress / v.r2Calm

  return (
    <Figure
      title="Every indicator's link to credit spreads strengthens sharply under macro stress"
      subtitle="Absolute correlation with monthly spread changes, split by regime. Bars are calm months against stress months."
      controls={
        <div className="inline-flex rounded-md border border-rule-strong p-0.5" role="group" aria-label="Regime definition">
          {([['macroOnly', 'Macro-only index'], ['spreadLevel', 'Spread level']] as const).map(([id, label]) => (
            <button
              key={id} onClick={() => setDef(id)} aria-pressed={def === id}
              className={`rounded px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                def === id ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink'}`}
            >{label}</button>
          ))}
        </div>
      }
      caption={
        def === 'macroOnly' ? (
          <>The regime here is built from macro indicators <em>only</em> — no spread information — so the
          comparison is not circular. It flags 22 of the 23 NBER recession months inside its window.
          Switch to <em>Spread level</em> to see the same pattern under the more obvious (and mildly
          circular) definition: the conclusion does not depend on the choice.</>
        ) : (
          <>This splits months by whether the spread itself sat in its top quartile — the intuitive
          definition, but it conditions on the outcome. It is shown to demonstrate the finding is not an
          artefact of the macro-only construction: R² is{' '}
          <strong className="font-semibold text-ink">{num(v.r2Calm, 2)}</strong> calm against{' '}
          <strong className="font-semibold text-ink">{num(v.r2Stress, 2)}</strong> stress either way.</>
        )
      }
    >
      <div className="grid gap-8 lg:grid-cols-[1.55fr_1fr]">
        <div ref={ref}>
          {w > 0 && (
            <svg width={w} height={ordered.length * rowH + 34} className="chart-svg block overflow-visible" role="img"
                 aria-label="Correlation with spread changes by indicator, calm versus stress months">
              {[0, 0.2, 0.4, 0.6].map((t) => (
                <g key={t}>
                  <line x1={labelW + x(t)} x2={labelW + x(t)} y1={0} y2={ordered.length * rowH}
                        className="grid-line" strokeWidth={1} />
                  <text x={labelW + x(t)} y={ordered.length * rowH + 16} textAnchor="middle" fontSize={10}>
                    {t.toFixed(1)}
                  </text>
                </g>
              ))}
              {ordered.map((r, i) => {
                const yTop = i * rowH + 5
                const h = (rowH - 14) / 2
                return (
                  <g key={r.key}>
                    <text x={labelW - 10} y={i * rowH + rowH / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={C.soft}>
                      {MACRO_META[r.key as MacroKey].short}
                    </text>
                    <rect x={labelW} y={yTop} width={Math.max(1, x(Math.abs(r.calm)))} height={h} rx={1.5}
                          fill={C.mute} opacity={0.42}
                          style={{ transition: 'width .45s cubic-bezier(.22,.61,.36,1)' }} />
                    <rect x={labelW} y={yTop + h + 2} width={Math.max(1, x(Math.abs(r.stress)))} height={h} rx={1.5}
                          fill={C.signal}
                          style={{ transition: 'width .45s cubic-bezier(.22,.61,.36,1)' }} />
                    <text x={labelW + x(Math.abs(r.calm)) + 5} y={yTop + h / 2} dy="0.32em" fontSize={9.5} fill={C.mute}>
                      {num(Math.abs(r.calm), 2)}
                    </text>
                    <text x={labelW + x(Math.abs(r.stress)) + 5} y={yTop + h + 2 + h / 2} dy="0.32em"
                          fontSize={9.5} fontWeight={600} fill={C.signal}>
                      {num(Math.abs(r.stress), 2)}
                    </text>
                  </g>
                )
              })}
            </svg>
          )}
          <div className="mt-2 flex items-center gap-4 text-2xs text-ink-mute">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-3.5 rounded-[2px]" style={{ background: C.mute, opacity: 0.42 }} />
              Calm ({v.nCalm} months)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-3.5 rounded-[2px]" style={{ background: C.signal }} />
              Stress ({v.nStress} months)
            </span>
          </div>
        </div>

        <div className="flex flex-col justify-center gap-5">
          <div className="rounded-lg border border-rule bg-paper-warm p-5">
            <div className="eyebrow mb-3">Joint explanatory power</div>
            <div className="flex items-end gap-6">
              <Stat label="Calm" value={num(v.r2Calm, 2)} tone="mute" />
              <Stat label="Stress" value={num(v.r2Stress, 2)} tone="signal" />
              <div className="pb-1">
                <div className="text-2xl font-bold text-gold tabular">{num(ratio, 1)}×</div>
                <div className="text-2xs text-ink-mute">stronger</div>
              </div>
            </div>
            <p className="mt-3 text-2xs leading-relaxed text-ink-mute">
              R² of monthly spread changes on all six indicators, fitted separately within each regime.
            </p>
          </div>

          <div className="rounded-lg border border-rule bg-paper-warm p-5">
            <div className="eyebrow mb-3">Where the movement lives</div>
            <div className="space-y-2.5">
              <ConcBar label="Share of months" value={regimeSplit.monthShare} tone="mute" />
              <ConcBar label="Share of spread variation" value={regimeSplit.varianceShare} tone="signal" />
            </div>
            <p className="mt-3 text-2xs leading-relaxed text-ink-mute">
              High-spread months are {regimeSplit.nHighSpread} of {regimeSplit.nAll} — a quarter of the
              sample carrying {pct(regimeSplit.varianceShare)} of all month-to-month movement.
            </p>
          </div>
        </div>
      </div>
    </Figure>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone: 'mute' | 'signal' }) {
  return (
    <div>
      <div className={`tabular text-3xl font-bold ${tone === 'signal' ? 'text-signal' : 'text-ink-mute'}`}>{value}</div>
      <div className="mt-0.5 text-2xs text-ink-mute">{label}</div>
    </div>
  )
}

function ConcBar({ label, value, tone }: { label: string; value: number; tone: 'mute' | 'signal' }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-2xs text-ink-soft">{label}</span>
        <span className={`tabular text-[0.8rem] font-semibold ${tone === 'signal' ? 'text-signal' : 'text-ink-mute'}`}>
          {pct(value)}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-rule">
        <div className={`h-full rounded-full ${tone === 'signal' ? 'bg-signal' : 'bg-ink-faint'}`}
             style={{ width: `${value * 100}%`, transition: 'width .6s cubic-bezier(.22,.61,.36,1)' }} />
      </div>
    </div>
  )
}
