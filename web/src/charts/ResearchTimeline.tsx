import { useId, useMemo, useState, type ReactNode } from 'react'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import {
  macroMeta, macroMetadata, research, sourceHref, timeline,
  type MacroKey, type ResearchTimelineRow,
} from '@/lib/research'
import { monthLabel } from '@/lib/format'
import { AxisBottom, AxisLeft, C, GridLines, PointerLayer, useMeasure } from './primitives'

type Target = 'legacyMean' | 'observedMean'
type Transformation = 'level' | 'change'
type Overlay = 'cluster' | 'macro' | 'spread' | 'nber' | 'none'

const CLUSTER_COLORS = ['#a6b9c4', '#b69279', '#919cad', '#a8ad90', '#ad9aaa', '#b5ac8d', '#799ea1', '#b4a5a0', '#8593b4', '#a6b7a7']
const NBER_PERIODS = [
  { start: '2001-03', end: '2001-11' },
  { start: '2007-12', end: '2009-06' },
  { start: '2020-02', end: '2020-04' },
]
const OVERLAY_LABELS: Record<Overlay, string> = {
  cluster: 'GMM cluster IDs', macro: 'Macro-only stress', spread: 'Spread-defined stress',
  nber: 'NBER reference periods', none: 'No overlay',
}
const selectClass = 'mt-2 w-full min-w-0 border border-rule-strong bg-paper px-3 py-2.5 text-sm text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-calm'
const buttonClass = 'border border-rule-strong px-3 py-2 text-xs text-ink-soft hover:border-ink-mute focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-calm disabled:cursor-not-allowed disabled:opacity-40'

function isNber(row: ResearchTimelineRow) {
  const date = row.date.slice(0, 7)
  return NBER_PERIODS.some(({ start, end }) => date >= start && date <= end)
}

function Control({ label, children }: { label: string; children: ReactNode }) {
  return <label className="block min-w-0 text-xs text-ink-mute">{label}{children}</label>
}

function DataValue({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-ink-mute">{label}</dt>
      <dd className="mt-2 font-mono text-lg text-ink">{value}</dd>
      {detail && <dd className="mt-1 text-xs leading-relaxed text-ink-mute">{detail}</dd>}
    </div>
  )
}

/** All plotted values come from stored CSVs; switches select descriptive views. */
export function ResearchTimeline() {
  const id = useId()
  const [ref, width] = useMeasure<HTMLDivElement>()
  const [target, setTarget] = useState<Target>('legacyMean')
  const [macro, setMacro] = useState<MacroKey>('Unemployment_Rate')
  const [transformation, setTransformation] = useState<Transformation>('level')
  const [overlay, setOverlay] = useState<Overlay>('cluster')
  const [selected, setSelected] = useState(Math.max(0, timeline.findIndex((row) => row.date.startsWith('2020-03'))))
  const selectedRow = timeline[selected]
  const meta = macroMeta[macro]
  const lowerUnit = transformation === 'level' ? meta.unit : meta.unit.startsWith('%') ? 'pp change' : 'index-point change'
  const lowerValues = useMemo(() => timeline.map((row) => transformation === 'level' ? row.macro[macro] : row.macroChange1[macro]), [macro, transformation])

  const compact = width < 600
  const height = compact ? 408 : 472
  const left = compact ? 44 : 52
  const right = Math.max(left + 1, width - 14)
  const upperTop = 30
  const upperBottom = compact ? 170 : 206
  const ribbonTop = upperBottom + 13
  const ribbonHeight = 12
  const lowerTop = ribbonTop + ribbonHeight + 54
  const lowerBottom = height - 31
  const x = scaleLinear().domain([0, timeline.length - 1]).range([left, right])
  const ySpread = scaleLinear().domain([0, Math.max(...timeline.map((row) => row[target])) * 1.05]).nice().range([upperBottom, upperTop])
  const definedValues = lowerValues.filter((value): value is number => value !== null)
  const minMacro = Math.min(...definedValues)
  const maxMacro = Math.max(...definedValues)
  const padding = (maxMacro - minMacro) * 0.09 || 1
  const yMacro = scaleLinear().domain([minMacro - padding, maxMacro + padding]).nice().range([lowerBottom, lowerTop])
  const spreadPath = line<ResearchTimelineRow>().x((_, index) => x(index)).y((row) => ySpread(row[target]))(timeline) ?? ''
  const macroPath = line<number | null>().defined((value) => value !== null).x((_, index) => x(index)).y((value) => yMacro(value!))(lowerValues) ?? ''
  const yearStep = compact ? 6 : 4
  const yearTicks = timeline.flatMap((row, index) => row.date.slice(5, 7) === '01' && Number(row.date.slice(0, 4)) % yearStep === 0 ? [index] : [])
  const targetName = target === 'legacyMean' ? 'Legacy monthly mean' : 'Observed-day monthly mean'
  const macroValue = lowerValues[selected]
  const showOverlay = overlay !== 'none'

  function bandState(row: ResearchTimelineRow): boolean | null {
    if (overlay === 'macro') return row.macroStressState
    if (overlay === 'spread') return row.spreadStress
    if (overlay === 'nber') return isNber(row)
    return null
  }

  function overlayValue(row: ResearchTimelineRow): string {
    if (overlay === 'cluster') return `Cluster ${row.clusterId}`
    if (overlay === 'macro') return row.macroStressState === null ? 'Warmup · unavailable' : row.macroStressState ? 'Macro stress' : 'Below threshold'
    if (overlay === 'spread') return row.spreadStress ? 'High legacy spread' : 'Below threshold'
    if (overlay === 'nber') return isNber(row) ? 'Reference recession' : 'Outside reference period'
    return 'No state selected'
  }

  const overlayDescription = {
    cluster: 'Raw IDs from the stored 10-component Gaussian mixture refit. Full-sample preprocessing and fitting make these retrospective. The artifact also contains centered three-month smoothed labels; the ribbon uses raw IDs.',
    macro: `Macro-only stress uses no spread inputs. Full-sample standardization and the upper-quartile threshold (${research.meta.thresholds.macroStress.toFixed(4)}) make it retrospective. The first 59 months are unavailable because the longest component needs 60 months.`,
    spread: `High-spread states use the legacy monthly target at or above its full-sample 75th percentile (${research.meta.thresholds.spreadFullSample.toFixed(4)} pp). The definition remains tied to the legacy target when the plotted aggregation changes.`,
    nber: 'Historical NBER recession reference periods, using the inclusive month convention stored in this project. These dates are a separate reference series, not cluster labels or model predictions.',
    none: 'No regime or stress classification is applied to this view. Both series retain their own labeled units on separate aligned panels.',
  }[overlay]

  return (
    <figure className="m-0" aria-labelledby={`${id}-title`}>
      <div className="mb-6 flex flex-wrap items-baseline justify-between gap-3">
        <h3 id={`${id}-title`} className="text-xl font-medium tracking-tight text-ink">Conditions in context</h3>
        <span className="font-mono text-xs text-ink-mute">DEC 1996 — AUG 2022 / n = {timeline.length}</span>
      </div>
      <div className="grid gap-4 border-y border-rule py-5 sm:grid-cols-2 lg:grid-cols-4">
        <Control label="Spread aggregation">
          <select className={selectClass} value={target} onChange={(event) => setTarget(event.target.value as Target)}>
            <option value="legacyMean">Legacy monthly mean</option>
            <option value="observedMean">Observed-day monthly mean</option>
          </select>
        </Control>
        <Control label="Macro indicator">
          <select className={selectClass} value={macro} onChange={(event) => setMacro(event.target.value as MacroKey)}>
            {macroMetadata.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
          </select>
        </Control>
        <Control label="Macro transformation">
          <select className={selectClass} value={transformation} onChange={(event) => setTransformation(event.target.value as Transformation)}>
            <option value="level">Raw stored value</option>
            <option value="change">One-month difference</option>
          </select>
        </Control>
        <Control label="Regime or reference overlay">
          <select className={selectClass} value={overlay} onChange={(event) => setOverlay(event.target.value as Overlay)}>
            {(Object.keys(OVERLAY_LABELS) as Overlay[]).map((value) => <option key={value} value={value}>{OVERLAY_LABELS[value]}</option>)}
          </select>
        </Control>
      </div>

      <div ref={ref} className="timeline-plot relative mt-7 w-full" style={{ minHeight: height }}>
        {width > 0 && (
          <svg width={width} height={height} className="chart-svg block" role="img" aria-labelledby={`${id}-chart-title ${id}-chart-desc`}>
            <title id={`${id}-chart-title`}>High-yield credit spread and {meta.label}</title>
            <desc id={`${id}-chart-desc`}>{targetName} in percentage points above; {transformation === 'level' ? 'raw macro values' : 'one-month macro differences'} in {lowerUnit} below. {OVERLAY_LABELS[overlay]}. Use the month slider below for exact values.</desc>
            <text x={left} y={13} fill={C.mute} fontSize={11}>{targetName} · pp</text>
            <text x={left} y={lowerTop - 13} fill={C.mute} fontSize={11}>{compact ? meta.key : meta.label} · {lowerUnit}</text>
            <GridLines scale={ySpread} ticks={ySpread.ticks(4)} x0={left} x1={right} />
            <GridLines scale={yMacro} ticks={yMacro.ticks(3)} x0={left} x1={right} />
            {showOverlay && timeline.map((row, index) => {
              const x0 = index === 0 ? left : (x(index - 1) + x(index)) / 2
              const x1 = index === timeline.length - 1 ? right : (x(index) + x(index + 1)) / 2
              const active = bandState(row)
              const color = overlay === 'cluster' ? CLUSTER_COLORS[row.clusterId] : overlay === 'nber' ? C.soft : overlay === 'macro' ? C.gold : C.signal
              return (
                <g key={row.date} aria-hidden="true">
                  {active && <rect x={x0} y={upperTop} width={x1 - x0} height={upperBottom - upperTop} fill={color} opacity={0.1} />}
                  <rect x={x0} y={ribbonTop} width={x1 - x0 + 0.1} height={ribbonHeight} fill={overlay === 'cluster' || active ? color : C.rule} opacity={active === null && overlay !== 'cluster' ? 0.25 : 0.85} />
                </g>
              )
            })}
            {showOverlay && <text x={left} y={ribbonTop + ribbonHeight + 17} fill={C.mute} fontSize={10}>{OVERLAY_LABELS[overlay]}{overlay === 'macro' ? ' · warmup through Oct 2001' : ''}</text>}
            <AxisLeft scale={ySpread} ticks={ySpread.ticks(4)} x={left} format={(value) => value.toFixed(0)} />
            <AxisLeft scale={yMacro} ticks={yMacro.ticks(3)} x={left} format={(value) => Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1)} />
            {transformation === 'change' && <line x1={left} x2={right} y1={yMacro(0)} y2={yMacro(0)} stroke={C.mute} strokeDasharray="3 5" opacity={0.5} />}
            <path d={spreadPath} fill="none" stroke={C.ink} strokeWidth={1.8} />
            <path d={macroPath} fill="none" stroke={C.calm} strokeWidth={1.65} />
            <AxisBottom scale={x} ticks={yearTicks} y={lowerBottom} format={(index) => timeline[index].date.slice(0, 4)} />
            <line x1={x(selected)} x2={x(selected)} y1={upperTop} y2={lowerBottom} stroke={C.mute} strokeDasharray="3 5" opacity={0.7} />
            <circle cx={x(selected)} cy={ySpread(selectedRow[target])} r={4} fill={C.ink} stroke={C.paper} strokeWidth={2} />
            {macroValue !== null && <circle cx={x(selected)} cy={yMacro(macroValue)} r={4} fill={C.calm} stroke={C.paper} strokeWidth={2} />}
            <PointerLayer x0={left} y0={upperTop} w={right - left} h={lowerBottom - upperTop} count={timeline.length} onIndex={setSelected} onLeave={() => {}} />
          </svg>
        )}
      </div>

      {overlay === 'cluster' && (
        <div className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-ink-mute" aria-label="Cluster color key">
          <span>Cluster ID</span>
          {CLUSTER_COLORS.map((color, index) => <span className="inline-flex items-center gap-1.5 font-mono" key={index}><span className="h-2 w-2" style={{ background: color }} />{index}</span>)}
        </div>
      )}

      <div className="border-y border-rule py-5">
        <div className="mb-5 flex items-center gap-3 sm:gap-5">
          <button className={buttonClass} type="button" disabled={selected === 0} onClick={() => setSelected((value) => Math.max(0, value - 1))} aria-label="Inspect previous month">←</button>
          <label className="min-w-0 flex-1 text-xs text-ink-mute" htmlFor={`${id}-month`}>
            Inspect month · arrow keys move one month
            <input id={`${id}-month`} type="range" className="mt-3 block w-full accent-calm" min={0} max={timeline.length - 1} step={1} value={selected} onChange={(event) => setSelected(Number(event.target.value))} aria-valuetext={`${monthLabel(selectedRow.date)}, spread ${selectedRow[target].toFixed(4)} percentage points`} />
          </label>
          <button className={buttonClass} type="button" disabled={selected === timeline.length - 1} onClick={() => setSelected((value) => Math.min(timeline.length - 1, value + 1))} aria-label="Inspect next month">→</button>
        </div>
        <dl className="grid grid-cols-2 gap-x-5 gap-y-6 lg:grid-cols-4">
          <DataValue label="Selected month" value={monthLabel(selectedRow.date)} detail={`${selectedRow.count} observed daily spread ${selectedRow.count === 1 ? 'value' : 'values'}`} />
          <DataValue label={targetName} value={`${selectedRow[target].toFixed(4)} pp`} detail={`Legacy − observed mean: ${(selectedRow.targetDifference * 100).toFixed(2)} bp`} />
          <DataValue label={`${meta.key}${transformation === 'change' ? ' · Δ1' : ''}`} value={macroValue === null ? 'Unavailable' : macroValue.toFixed(3)} detail={macroValue === null ? 'No earlier observation in the stored panel' : lowerUnit} />
          <DataValue label={OVERLAY_LABELS[overlay]} value={overlayValue(selectedRow)} detail={overlay === 'macro' && selectedRow.macroStress !== null ? `Index: ${selectedRow.macroStress.toFixed(4)}` : overlay === 'cluster' ? 'Retrospective membership' : undefined} />
        </dl>
      </div>

      <figcaption className="mt-5 space-y-3 text-sm leading-relaxed text-ink-mute">
        <p>{overlayDescription}</p>
        <p>{target === 'legacyMean' ? 'Legacy target: mean of source-date values after forward filling missing spread cells.' : 'Observed-day target: mean of nonmissing daily spread observations only.'} The panels use separate units. {transformation === 'change' && <>Macro difference = value(t) − value(t−1); n = {timeline.length - 1} observed differences. </>}1 pp = 100 basis points.</p>
        <p className="border-l-2 border-gold pl-3">The final month is partial: August 2022 contains only the August 1 spread observation. Stored macro values are revised data; these descriptive views do not establish real-time availability.</p>
        <details className="border-t border-rule pt-3 text-xs">
          <summary className="cursor-pointer py-1 text-ink-soft">Sources and analytical context</summary>
          <div className="mt-3 space-y-2 leading-relaxed">
            <p>Status: verified descriptive export. Dates: December 1996–August 2022. Sample: {timeline.length} monthly observations; macro stress is available for 250 months. Horizon: contemporaneous description. Forecast benchmark: not applicable.</p>
            <p>Macro source: <span className="font-mono">{meta.source}</span>. Column: <span className="font-mono">{meta.key}</span>. {transformation === 'change' ? 'Transformation: first difference of adjacent stored monthly values.' : 'Transformation: raw stored monthly value.'}</p>
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              <a className="underline underline-offset-4" href={sourceHref('data/merged_macroeconomic_credit.csv')}>Core monthly panel</a>
              <a className="underline underline-offset-4" href={sourceHref('research/evidence/target_provenance.csv')}>Target provenance</a>
              <a className="underline underline-offset-4" href={sourceHref('data/datamerged_macro_credit_with_regimes.csv')}>Stored regime assignments</a>
              <a className="underline underline-offset-4" href={sourceHref('research/evidence/macro_stress.csv')}>Macro stress series</a>
              <a className="underline underline-offset-4" href={sourceHref('analysis/macro_regime.py')}>Macro index construction</a>
              <a className="underline underline-offset-4" href={sourceHref('src/config_v2.py')}>Reference-period convention</a>
            </div>
          </div>
        </details>
      </figcaption>
    </figure>
  )
}

export default ResearchTimeline
