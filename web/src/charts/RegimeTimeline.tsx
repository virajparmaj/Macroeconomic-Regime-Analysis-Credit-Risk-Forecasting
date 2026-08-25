/** The centrepiece: 26 years of credit stress, with the macro regime and the
 *  stability of the macro relationship laid over the same time axis.
 *
 *  It answers one question that no static chart in the report could: *when* did
 *  the relationship between the macro economy and credit risk actually hold?
 *  The overlay switch is the analytical control — model fit, a single indicator,
 *  or that indicator's rolling correlation with the spread. */

import { useMemo, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3line, area as d3area, curveMonotoneX } from 'd3-shape'
import {
  timeline, ERAS, NBER_BANDS, MACRO_META, MACRO_KEYS,
  R2_WORST5, R2_BEST5, type EraId, type MacroKey,
} from '@/lib/data'
import { monthLabel, num, toYear } from '@/lib/format'
import {
  AxisBottom, AxisLeft, C, Figure, GridLines, MACRO_COLOR,
  PointerLayer, TipRow, Tooltip, useMeasure,
} from './primitives'

type Overlay = 'r2' | 'macro' | 'rollcorr'

const OVERLAYS: { id: Overlay; label: string; hint: string }[] = [
  { id: 'r2',       label: 'Model fit',      hint: 'Rolling 60-month R² of all six macro indicators explaining monthly spread changes.' },
  { id: 'rollcorr', label: 'Correlation',    hint: 'Rolling 60-month correlation between the selected indicator and monthly spread changes.' },
  { id: 'macro',    label: 'Indicator level',hint: 'The selected macro indicator in its own units.' },
]

const ANNOTATIONS = [
  { d: '2002-10', text: 'Telecom bust', dy: -6 },
  { d: '2008-12', text: 'GFC peak · 20.3pp', dy: -6 },
  { d: '2020-03', text: 'COVID', dy: -6 },
]

export function RegimeTimeline() {
  const [ref, w] = useMeasure<HTMLDivElement>()
  const [era, setEra] = useState<EraId>('all')
  const [overlay, setOverlay] = useState<Overlay>('r2')
  const [macro, setMacro] = useState<MacroKey>('unemp')
  const [hover, setHover] = useState<number | null>(null)
  const [showRegime, setShowRegime] = useState(true)

  const eraDef = ERAS.find((e) => e.id === era)!
  const rows = useMemo(
    () => timeline.filter((t) => t.d >= eraDef.from && t.d <= eraDef.to),
    [eraDef.from, eraDef.to],
  )

  const height = w < 640 ? 360 : 470
  const lowerH = w < 640 ? 104 : 136
  const nberH = 9
  const m = { top: 26, right: w < 640 ? 14 : 22, bottom: 40, left: w < 640 ? 38 : 48 }
  const iw = Math.max(10, w - m.left - m.right)
  const panelGap = 44
  const upperH = Math.max(10, height - m.top - m.bottom - lowerH - panelGap - nberH)

  const x = useMemo(() => {
    const ys = rows.map((r) => toYear(r.d))
    return scaleLinear().domain([Math.min(...ys), Math.max(...ys)]).range([m.left, m.left + iw])
  }, [rows, m.left, iw])

  const ySpread = useMemo(() => {
    const max = Math.max(...rows.map((r) => r.spread))
    return scaleLinear().domain([0, max * 1.12]).range([m.top + upperH, m.top]).nice()
  }, [rows, m.top, upperH])

  const lowerTop = m.top + upperH + panelGap
  const lowerBottom = lowerTop + lowerH

  // Overlay series + its own scale, in the lower panel.
  const { pts, yLow, lowTicks, lowFmt, lowColor, lowLabel, zeroLine } = useMemo(() => {
    if (overlay === 'r2') {
      const p = rows.map((r) => ({ d: r.d, v: r.r2 }))
      return {
        pts: p, yLow: scaleLinear().domain([0, 0.8]).range([lowerBottom, lowerTop]),
        lowTicks: [0, 0.2, 0.4, 0.6, 0.8], lowFmt: (v: number) => v.toFixed(1),
        lowColor: C.calm, lowLabel: 'rolling 60-month R²', zeroLine: null as number | null,
      }
    }
    if (overlay === 'rollcorr') {
      const key = ('rc_' + macro) as keyof (typeof rows)[0]
      const p = rows.map((r) => ({ d: r.d, v: r[key] as number | null }))
      return {
        pts: p, yLow: scaleLinear().domain([-0.8, 0.8]).range([lowerBottom, lowerTop]),
        lowTicks: [-0.8, -0.4, 0, 0.4, 0.8], lowFmt: (v: number) => v.toFixed(1),
        lowColor: MACRO_COLOR[macro], lowLabel: 'rolling correlation', zeroLine: 0,
      }
    }
    const p = rows.map((r) => ({ d: r.d, v: r[macro] as number }))
    const vals = p.map((q) => q.v!).filter((v) => v != null)
    const lo = Math.min(...vals), hi = Math.max(...vals)
    const pad = (hi - lo) * 0.12 || 1
    const sc = scaleLinear().domain([lo - pad, hi + pad]).range([lowerBottom, lowerTop]).nice()
    return {
      pts: p, yLow: sc, lowTicks: sc.ticks(4),
      lowFmt: (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)),
      lowColor: MACRO_COLOR[macro], lowLabel: MACRO_META[macro].unit, zeroLine: null,
    }
  }, [overlay, macro, rows, lowerTop, lowerBottom])

  const spreadPath = d3line<(typeof rows)[0]>()
    .x((r) => x(toYear(r.d))).y((r) => ySpread(r.spread)).curve(curveMonotoneX)(rows) ?? ''
  const spreadArea = d3area<(typeof rows)[0]>()
    .x((r) => x(toYear(r.d))).y0(ySpread(0)).y1((r) => ySpread(r.spread)).curve(curveMonotoneX)(rows) ?? ''
  const lowPath = d3line<{ d: string; v: number | null }>()
    .defined((p) => p.v != null).x((p) => x(toYear(p.d))).y((p) => yLow(p.v!)).curve(curveMonotoneX)(pts) ?? ''

  // Contiguous stress runs, drawn as bands behind everything.
  const bands = useMemo(() => {
    const out: { from: string; to: string }[] = []
    let start: string | null = null
    rows.forEach((r, i) => {
      if (r.regime === 'stress' && start === null) start = r.d
      const next = rows[i + 1]
      if (start !== null && (r.regime !== 'stress' || !next || next.regime !== 'stress')) {
        if (r.regime === 'stress') out.push({ from: start, to: r.d })
        start = null
      }
    })
    return out
  }, [rows])

  const yearTicks = useMemo(() => {
    const [a, b] = x.domain()
    const span = b - a
    const step = span > 20 ? 4 : span > 10 ? 2 : span > 4 ? 1 : 1
    const t: number[] = []
    for (let yr = Math.ceil(a); yr <= b; yr += step) t.push(yr)
    return t.length > 1 ? t : [Math.ceil(a), Math.floor(b)]
  }, [x])

  const hoverRow = hover != null ? rows[hover] : null
  const hoverX = hoverRow ? x(toYear(hoverRow.d)) : 0

  const overlayHint = OVERLAYS.find((o) => o.id === overlay)!.hint
  const needsMacro = overlay !== 'r2'

  return (
    <Figure
      title="Twenty-six years of credit stress, and the shifting ground beneath it"
      subtitle="The top panel is the high-yield spread — the price of credit risk. The lower panel is whichever diagnostic you choose. Shaded bands are months the macro-only stress index flags; grey bars are NBER recessions."
      controls={
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Time period">
          {ERAS.map((e) => (
            <button
              key={e.id} onClick={() => setEra(e.id)}
              aria-pressed={era === e.id}
              className={`rounded-full border px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                era === e.id
                  ? 'border-ink bg-ink text-paper'
                  : 'border-rule-strong bg-paper text-ink-soft hover:border-ink-faint hover:text-ink'
              }`}
            >
              {e.label}
            </button>
          ))}
        </div>
      }
      caption={
        <>
          <strong className="font-semibold text-ink">Read the lower panel against the upper one.</strong>{' '}
          {overlay === 'r2' ? (
            <>The five worst-fitting months in the sample (marked ▾) sit in the flat calm immediately before
            the 2008 spike, and the five best (▴) sit after COVID had already repriced. Fit rises once a
            crisis has arrived — it does not rise beforehand.</>
          ) : overlay === 'rollcorr' ? (
            <>{MACRO_META[macro].label}'s rolling correlation with spread changes crosses zero rather than
            settling anywhere. Every one of the six does. There is no single stable coefficient here to
            estimate — which is why a model fitted on the whole sample is averaging over relationships that
            disagree with each other.</>
          ) : (
            <>{MACRO_META[macro].label} in its own units, against the spread above. Note how little the two
            move together outside the shaded stress bands — the visual counterpart of Finding 3.</>
          )}
        </>
      }
      source="Sources: FRED (CPIAUCSL, FEDFUNDS, INDPRO, UNRATE, UMCSENT) and OECD EA19LORSGPORGYSAM; ICE BofA BAMLH0A0HYM2. 309 months, Dec 1996 – Aug 2022. Rolling windows need 60 months of history, so the lower panel begins in late 2001."
    >
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2.5">
        <div className="flex items-center gap-1.5" role="group" aria-label="Lower panel series">
          <span className="mr-0.5 text-2xs font-medium text-ink-mute">Lower panel</span>
          {OVERLAYS.map((o) => (
            <button
              key={o.id} onClick={() => setOverlay(o.id)} aria-pressed={overlay === o.id}
              className={`rounded-md border px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                overlay === o.id
                  ? 'border-calm bg-calm-soft text-ink'
                  : 'border-rule-strong bg-paper text-ink-soft hover:border-ink-faint hover:text-ink'
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>

        <div
          className={`flex items-center gap-1.5 transition-opacity duration-300 ${needsMacro ? 'opacity-100' : 'pointer-events-none opacity-35'}`}
          role="group" aria-label="Macro indicator"
        >
          <span className="mr-0.5 text-2xs font-medium text-ink-mute">Indicator</span>
          {MACRO_KEYS.map((k) => (
            <button
              key={k} onClick={() => setMacro(k)} aria-pressed={macro === k} disabled={!needsMacro}
              className={`rounded-md border px-2 py-1 text-2xs font-medium transition-colors duration-200 ${
                macro === k && needsMacro
                  ? 'border-ink bg-ink text-paper'
                  : 'border-rule-strong bg-paper text-ink-soft hover:border-ink-faint hover:text-ink'
              }`}
            >
              {MACRO_META[k].short}
            </button>
          ))}
        </div>

        <label className="flex cursor-pointer select-none items-center gap-1.5 text-2xs font-medium text-ink-mute">
          <input
            type="checkbox" checked={showRegime} onChange={(e) => setShowRegime(e.target.checked)}
            className="h-3.5 w-3.5 cursor-pointer accent-[#C8322B]"
          />
          Macro-stress bands
        </label>
      </div>

      <div ref={ref} className="relative select-none">
        {w > 0 && (
          <svg width={w} height={height} className="chart-svg block overflow-visible" role="img"
               aria-label="Credit spread over time with macro regime bands and a selectable diagnostic panel">
            {/* NBER recessions: faint full-height guide plus a solid ribbon that reads
                unambiguously against the macro-stress shading. */}
            {NBER_BANDS.map((b) => {
              if (b.to < eraDef.from || b.from > eraDef.to) return null
              const x0 = x(toYear(b.from > eraDef.from ? b.from : eraDef.from))
              const x1 = x(toYear(b.to < eraDef.to ? b.to : eraDef.to))
              return (
                <g key={b.label}>
                  <rect x={x0} y={m.top} width={Math.max(1, x1 - x0)}
                        height={lowerBottom - m.top} fill={C.ink} opacity={0.045} />
                  <rect x={x0} y={lowerBottom + 22} width={Math.max(2, x1 - x0)} height={nberH}
                        rx={1.5} fill={C.soft} opacity={0.85} />
                </g>
              )
            })}
            <text x={m.left - 10} y={lowerBottom + 22 + nberH / 2} dy="0.32em" textAnchor="end"
                  fontSize={9.5} fill={C.faint}>NBER</text>
            {/* Macro-stress bands */}
            {showRegime && bands.map((b) => {
              const x0 = x(toYear(b.from)), x1 = x(toYear(b.to))
              return (
                <rect key={b.from} x={x0} y={m.top} width={Math.max(1.5, x1 - x0)}
                      height={upperH} fill={C.signal} opacity={0.13} />
              )
            })}

            <GridLines scale={ySpread} ticks={ySpread.ticks(4)} x0={m.left} x1={m.left + iw} />
            <AxisLeft scale={ySpread} ticks={ySpread.ticks(4)} x={m.left} format={(v) => `${v}`} />
            <text x={m.left} y={m.top - 9} fontSize={11} fill={C.mute}>High-yield spread (pp)</text>

            <path d={spreadArea} fill={C.signal} opacity={0.07} />
            <path d={spreadPath} fill="none" stroke={C.signal} strokeWidth={1.9}
                  strokeLinejoin="round" strokeLinecap="round" />

            {/* Event annotations, full sample only */}
            {era === 'all' && ANNOTATIONS.map((a) => {
              const r = rows.find((q) => q.d === a.d)
              if (!r || w < 640) return null
              return (
                <g key={a.d}>
                  <circle cx={x(toYear(a.d))} cy={ySpread(r.spread)} r={2.6} fill={C.signalDeep} />
                  <text x={x(toYear(a.d))} y={ySpread(r.spread) + a.dy} textAnchor="middle"
                        fontSize={10} fill={C.signalDeep} fontWeight={600}>{a.text}</text>
                </g>
              )
            })}

            {/* Lower panel */}
            <line x1={m.left} x2={m.left + iw} y1={lowerTop - 24} y2={lowerTop - 24}
                  stroke={C.rule} strokeWidth={1} />
            <GridLines scale={yLow} ticks={lowTicks} x0={m.left} x1={m.left + iw} />
            <AxisLeft scale={yLow} ticks={lowTicks} x={m.left} format={lowFmt} />
            <text x={m.left} y={lowerTop - 9} fontSize={11} fill={C.mute}>{lowLabel}</text>
            {zeroLine != null && (
              <line x1={m.left} x2={m.left + iw} y1={yLow(zeroLine)} y2={yLow(zeroLine)}
                    stroke={C.ruleStrong} strokeWidth={1.2} />
            )}
            <path d={lowPath} fill="none" stroke={lowColor} strokeWidth={1.8}
                  strokeLinejoin="round" strokeLinecap="round" />

            {/* Worst / best fit markers */}
            {overlay === 'r2' && rows.map((r) => {
              const worst = R2_WORST5.includes(r.d), best = R2_BEST5.includes(r.d)
              if ((!worst && !best) || r.r2 == null) return null
              const cx = x(toYear(r.d)), cy = yLow(r.r2)
              return (
                <path key={r.d}
                      d={worst ? `M${cx - 5.5},${cy - 13}L${cx + 5.5},${cy - 13}L${cx},${cy - 4}Z`
                               : `M${cx - 5.5},${cy + 13}L${cx + 5.5},${cy + 13}L${cx},${cy + 4}Z`}
                      fill={worst ? C.signal : C.moss} stroke={C.paper} strokeWidth={0.9} />
              )
            })}

            <AxisBottom scale={x} ticks={yearTicks} y={lowerBottom} format={(v) => `${v}`} />

            {hoverRow && (
              <g pointerEvents="none">
                <line x1={hoverX} x2={hoverX} y1={m.top} y2={lowerBottom} stroke={C.ink} strokeWidth={1} opacity={0.35} />
                <circle cx={hoverX} cy={ySpread(hoverRow.spread)} r={3.6} fill={C.signal}
                        stroke={C.paper} strokeWidth={1.6} />
                {(() => {
                  const p = pts[hover!]
                  return p?.v != null ? (
                    <circle cx={hoverX} cy={yLow(p.v)} r={3.4} fill={lowColor} stroke={C.paper} strokeWidth={1.6} />
                  ) : null
                })()}
              </g>
            )}

            <PointerLayer x0={m.left} y0={m.top} w={iw} h={lowerBottom - m.top}
                          count={rows.length} onIndex={(i) => setHover(i)} onLeave={() => setHover(null)} />
          </svg>
        )}

        {hoverRow && (
          <Tooltip x={hoverX} y={12} width={w}>
            <div className="mb-1 text-2xs font-semibold text-ink">{monthLabel(hoverRow.d)}</div>
            <TipRow label="Credit spread" value={`${num(hoverRow.spread)}pp`} color={C.signal} strong />
            {overlay === 'r2' && (
              <TipRow label="Rolling R²" value={hoverRow.r2 != null ? num(hoverRow.r2, 3) : '—'} color={C.calm} />
            )}
            {overlay === 'rollcorr' && (
              <TipRow
                label={`${MACRO_META[macro].short} corr.`}
                value={(() => { const v = pts[hover!]?.v; return v != null ? num(v, 2) : '—' })()}
                color={MACRO_COLOR[macro]}
              />
            )}
            {overlay === 'macro' && (
              <TipRow label={MACRO_META[macro].short}
                      value={`${num(hoverRow[macro] as number)} ${MACRO_META[macro].unit}`}
                      color={MACRO_COLOR[macro]} />
            )}
            <div className="mt-1.5 flex items-center gap-1.5 border-t border-rule pt-1.5">
              <span className={`h-[7px] w-[7px] rounded-full ${
                hoverRow.regime === 'stress' ? 'bg-signal' : hoverRow.regime === 'calm' ? 'bg-rule-strong' : 'bg-transparent'
              }`} />
              <span className="text-2xs text-ink-mute">
                {hoverRow.regime === 'stress' ? 'Macro stress' : hoverRow.regime === 'calm' ? 'Calm' : 'Before index start'}
                {hoverRow.nber && ' · NBER recession'}
              </span>
            </div>
          </Tooltip>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-2xs text-ink-mute">
        <Key color={C.signal} label="High-yield spread" />
        <Key color={lowColor} label={overlay === 'r2' ? 'Rolling model fit' : overlay === 'rollcorr' ? `${MACRO_META[macro].short} correlation` : MACRO_META[macro].short} />
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-[2px]" style={{ background: C.signal, opacity: 0.22 }} />
          Macro-stress months
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-[7px] w-4 rounded-[2px]" style={{ background: C.soft, opacity: 0.85 }} />
          NBER recession
        </span>
        {overlay === 'r2' && <span className="text-ink-faint">▾ 5 worst-fit months &nbsp; ▴ 5 best-fit months</span>}
      </div>
      <p className="mt-2 text-2xs leading-relaxed text-ink-faint">{overlayHint}</p>
    </Figure>
  )
}

function Key({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-[2.5px] w-4 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}
