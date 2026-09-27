/** Shared chart scaffolding: responsive sizing, axes, grid and a pointer layer.
 *  All charts on the page are built from these so styling stays identical. */

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'

export const C = {
  ink: '#F1F0EB', soft: '#C4C7CA', mute: '#A5AAB0', faint: '#9299A0',
  rule: '#30363D', ruleStrong: '#4D555F', paper: '#101113', sunk: '#22262B',
  signal: '#B2A29D', signalSoft: '#2C2524', signalDeep: '#C8B4AD',
  calm: '#ACB7C1', calmSoft: '#252A30',
  gold: '#B6BEC8', goldSoft: '#282D34',
  moss: '#9AAFA5', mossSoft: '#242E29',
  violet: '#AEA8B8',
} as const

/** Stable, colour-blind-safe assignment for the six macro indicators. */
export const MACRO_COLOR: Record<string, string> = {
  fedfunds: C.signal, cpi: C.gold, eagdp: C.violet,
  sentiment: C.calm, indpro: C.moss, unemp: C.soft,
}

export function useMeasure<T extends HTMLElement>() {
  const ref = useRef<T | null>(null)
  const [w, setW] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    setW(el.getBoundingClientRect().width)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Reveals children once they scroll into view.
 *
 *  Fails open. If IntersectionObserver is unavailable, or never fires — which
 *  happens when the document is not being rendered, e.g. a backgrounded tab or a
 *  prerender — a timer reveals the content anyway. An entrance animation must
 *  never be able to hide the page's content permanently. */
export function Reveal({ children, className = '', delay = 0 }: {
  children: ReactNode; className?: string; delay?: number
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    const el = ref.current
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (!el || reduce || typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }

    // Safety net: reveal regardless if the observer has not reported by now.
    const fallback = window.setTimeout(() => setSeen(true), 1200)

    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          window.clearTimeout(fallback)
          setSeen(true)
          io.disconnect()
        }
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.08 },
    )
    io.observe(el)
    return () => { window.clearTimeout(fallback); io.disconnect() }
  }, [])

  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: seen ? 1 : 0,
        transform: seen ? 'none' : 'translateY(14px)',
        transition: `opacity .6s cubic-bezier(.22,.61,.36,1) ${delay}ms, transform .6s cubic-bezier(.22,.61,.36,1) ${delay}ms`,
      }}
    >
      {children}
    </div>
  )
}

export interface Margin { top: number; right: number; bottom: number; left: number }

export function AxisLeft({ scale, ticks, x, format, label }: {
  scale: (v: number) => number; ticks: number[]; x: number
  format: (v: number) => string; label?: string
}) {
  return (
    <g>
      {ticks.map((t) => (
        <text key={t} x={x - 8} y={scale(t)} dy="0.32em" textAnchor="end" fontSize={11}>
          {format(t)}
        </text>
      ))}
      {label && (
        <text
          transform={`translate(${x - 44},${scale(ticks[Math.floor(ticks.length / 2)])}) rotate(-90)`}
          textAnchor="middle" fontSize={11} fill={C.mute}
        >
          {label}
        </text>
      )}
    </g>
  )
}

export function GridLines({ scale, ticks, x0, x1 }: {
  scale: (v: number) => number; ticks: number[]; x0: number; x1: number
}) {
  return (
    <g aria-hidden>
      {ticks.map((t) => (
        <line key={t} x1={x0} x2={x1} y1={scale(t)} y2={scale(t)} className="grid-line" strokeWidth={1} />
      ))}
    </g>
  )
}

export function AxisBottom({ scale, ticks, y, format }: {
  scale: (v: number) => number; ticks: number[]; y: number; format: (v: number) => string
}) {
  return (
    <g>
      <line x1={scale(ticks[0])} x2={scale(ticks[ticks.length - 1])} y1={y} y2={y} className="axis-line" strokeWidth={1} />
      {ticks.map((t) => (
        <text key={t} x={scale(t)} y={y + 17} textAnchor="middle" fontSize={11}>
          {format(t)}
        </text>
      ))}
    </g>
  )
}

/** Floating tooltip anchored inside the chart's own bounding box. */
export function Tooltip({ x, y, width, children, align = 'auto' }: {
  x: number; y: number; width: number; children: ReactNode; align?: 'auto' | 'left' | 'right'
}) {
  const flip = align === 'auto' ? x > width * 0.58 : align === 'right'
  return (
    <div
      className="pointer-events-none absolute z-20 min-w-[9rem] max-w-[15rem] rounded-md border border-rule bg-paper/97 px-3 py-2 shadow-[0_6px_24px_-8px_rgba(18,21,26,.28)] backdrop-blur-[2px]"
      style={{ left: flip ? undefined : x + 14, right: flip ? width - x + 14 : undefined, top: y }}
      role="status"
    >
      {children}
    </div>
  )
}

export function TipRow({ label, value, color, strong }: {
  label: string; value: string; color?: string; strong?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[1px]">
      <span className="flex min-w-0 items-center gap-1.5">
        {color && <span className="h-[7px] w-[7px] shrink-0 rounded-full" style={{ background: color }} />}
        <span className={`truncate text-2xs ${strong ? 'text-ink' : 'text-ink-mute'}`}>{label}</span>
      </span>
      <span className={`tabular shrink-0 text-2xs ${strong ? 'font-semibold text-ink' : 'text-ink-soft'}`}>{value}</span>
    </div>
  )
}

/** Invisible rect that turns pointer / touch position into a data index. */
export function PointerLayer({ x0, y0, w, h, count, onIndex, onLeave }: {
  x0: number; y0: number; w: number; h: number; count: number
  onIndex: (i: number, px: number) => void; onLeave: () => void
}) {
  const handle = useCallback((clientX: number, rect: DOMRect) => {
    const px = clientX - rect.left - x0
    const i = Math.max(0, Math.min(count - 1, Math.round((px / w) * (count - 1))))
    onIndex(i, px + x0)
  }, [count, w, x0, onIndex])

  return (
    <rect
      x={x0} y={y0} width={Math.max(0, w)} height={Math.max(0, h)}
      fill="transparent" style={{ touchAction: 'pan-y' }}
      onPointerMove={(e) => handle(e.clientX, e.currentTarget.ownerSVGElement!.getBoundingClientRect())}
      onPointerDown={(e) => handle(e.clientX, e.currentTarget.ownerSVGElement!.getBoundingClientRect())}
      onPointerLeave={onLeave}
    />
  )
}

/** Chart frame: finding-based title, optional controls, caption underneath. */
export function Figure({ id, title, subtitle, controls, children, caption, source }: {
  id?: string; title: string; subtitle?: string; controls?: ReactNode
  children: ReactNode; caption?: ReactNode; source?: string
}) {
  return (
    <figure id={id} className="my-0">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-2xl">
          <h4 className="text-[0.95rem] font-semibold leading-snug text-ink">{title}</h4>
          {subtitle && <p className="mt-1 text-[0.8rem] leading-relaxed text-ink-mute">{subtitle}</p>}
        </div>
        {controls && <div className="shrink-0">{controls}</div>}
      </div>
      {children}
      {(caption || source) && (
        <figcaption className="mt-3 space-y-1.5">
          {caption && <div className="text-[0.8rem] leading-relaxed text-ink-soft">{caption}</div>}
          {source && <div className="text-2xs leading-relaxed text-ink-faint">{source}</div>}
        </figcaption>
      )}
    </figure>
  )
}
