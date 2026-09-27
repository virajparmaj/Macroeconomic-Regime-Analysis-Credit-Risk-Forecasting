import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { useMeasure, AxisLeft, GridLines } from './primitives'

export interface PlotSeries { label: string; color: string; values: (number | null)[]; dashed?: boolean }

/** Straight segments preserve the observations; no interpolated predictions or uncertainty. */
export function ResearchPlot({ dates, series, label, unit = 'Spread · pp', active, onActive, refits = [], daily = false }: {
  dates: string[]; series: PlotSeries[]; label: string; unit?: string; active?: number; onActive?: (index: number) => void; refits?: number[]; daily?: boolean
}) {
  const [ref, width] = useMeasure<HTMLDivElement>()
  const height = width < 600 ? 250 : 320
  const left = 44, right = 18, top = 30, bottom = height - 35
  const positions = dates.map((date, i) => daily ? Date.parse(date) : i)
  const xScale = scaleLinear().domain([positions[0], positions.at(-1)!]).range([left, Math.max(left + 1, width - right)])
  const x = (index: number) => xScale(positions[index])
  const values = series.flatMap(s => s.values.filter((v): v is number => v !== null))
  const min = Math.min(0, ...values), max = Math.max(1, ...values)
  const y = scaleLinear().domain([min, max * 1.08]).range([bottom, top]).nice()
  const tickCount = width < 500 ? 4 : 7
  const ticks = [...new Set(Array.from({ length: tickCount }, (_, i) => Math.round(i * (dates.length - 1) / (tickCount - 1))))]
  return <div className="research-plot" ref={ref}>
    {width > 0 && <svg width={width} height={height} className="chart-svg" role="img" aria-label={label}>
      <title>{label}</title>
      <text x={left} y={13} fontSize={11}>{unit}</text>
      <GridLines scale={y} ticks={y.ticks(5)} x0={left} x1={width - right} />
      <AxisLeft scale={y} ticks={y.ticks(5)} x={left} format={v => String(v)} />
      {refits.map(i => <line key={i} x1={x(i)} x2={x(i)} y1={top} y2={bottom} stroke="#343a41" strokeDasharray="2 6" />)}
      {series.map(s => <path key={s.label} d={line<number | null>().defined(v => v !== null).x((_, i) => x(i)).y(v => y(v!))(s.values) ?? ''} fill="none" stroke={s.color} strokeWidth={s.label === 'Actual' ? 2 : 1.7} strokeDasharray={s.dashed ? '5 4' : undefined} />)}
      {ticks.map(i => <text key={i} x={x(i)} y={height - 9} fontSize={10} textAnchor={i === 0 ? 'start' : i === dates.length - 1 ? 'end' : 'middle'}>{daily ? dates[i]?.slice(5) : dates[i]?.slice(0, 7)}</text>)}
      {active !== undefined && <g><line x1={x(active)} x2={x(active)} y1={top} y2={bottom} stroke="#b9c0c6" strokeDasharray="3 3" />{series.map(s => s.values[active] != null && <circle key={s.label} cx={x(active)} cy={y(s.values[active]!)} r="3" fill={s.color} />)}</g>}
      {onActive && <rect x={left} y={top} width={Math.max(1, width-left-right)} height={bottom-top} fill="transparent" onPointerMove={e => {
        const rect = e.currentTarget.ownerSVGElement!.getBoundingClientRect()
        const position = xScale.invert(e.clientX - rect.left)
        onActive(positions.reduce((nearest, value, i) => Math.abs(value-position) < Math.abs(positions[nearest]-position) ? i : nearest, 0))
      }} />}
    </svg>}
    <div className="legend">{series.map(s => <span key={s.label}><i style={{ background: s.dashed ? 'transparent' : s.color, borderTop: s.dashed ? `2px dashed ${s.color}` : undefined, height: s.dashed ? 0 : undefined }} />{s.label}</span>)}</div>
  </div>
}
