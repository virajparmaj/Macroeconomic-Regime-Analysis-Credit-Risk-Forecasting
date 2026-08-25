/** Layout and narrative primitives shared across the page. */

import { useEffect, useRef, useState, type ReactNode } from 'react'

export function Section({ id, children, className = '', tone = 'paper' }: {
  id?: string; children: ReactNode; className?: string; tone?: 'paper' | 'warm' | 'sunk'
}) {
  const bg = tone === 'warm' ? 'bg-paper-warm' : tone === 'sunk' ? 'bg-paper-sunk' : 'bg-paper'
  return (
    <section id={id} className={`${bg} ${className}`}>
      <div className="mx-auto w-full max-w-content px-5 sm:px-8">{children}</div>
    </section>
  )
}

export function SectionHead({ eyebrow, title, lede, className = '' }: {
  eyebrow?: string; title: string; lede?: string; className?: string
}) {
  return (
    <header className={`max-w-2xl ${className}`}>
      {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
      <h2 className="text-[1.6rem] font-bold leading-[1.2] tracking-[-0.015em] text-ink sm:text-[2rem]">{title}</h2>
      {lede && <p className="lede mt-4">{lede}</p>}
    </header>
  )
}

/** A single headline figure. `note` carries the comparison that makes it mean
 *  something — a number without its benchmark is decoration. */
export function Metric({ value, unit, label, note, tone = 'ink' }: {
  value: string; unit?: string; label: string; note?: string; tone?: 'ink' | 'signal' | 'gold' | 'calm'
}) {
  const color = tone === 'signal' ? 'text-signal' : tone === 'gold' ? 'text-gold' : tone === 'calm' ? 'text-calm' : 'text-ink'
  return (
    <div className="min-w-0">
      <div className={`tabular flex items-baseline gap-1 text-[2rem] font-bold leading-none tracking-[-0.02em] sm:text-[2.4rem] ${color}`}>
        {value}
        {unit && <span className="text-base font-semibold sm:text-lg">{unit}</span>}
      </div>
      <div className="mt-2.5 text-[0.8rem] font-medium leading-snug text-ink">{label}</div>
      {note && <div className="mt-1 text-2xs leading-relaxed text-ink-mute">{note}</div>}
    </div>
  )
}

/** Progressive disclosure for methodology. Closed by default so the narrative
 *  stays readable; open reveals the detail a technical reviewer will want. */
export function Disclose({ summary, children, defaultOpen = false }: {
  summary: string; children: ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-lg border border-rule bg-paper-warm">
      <button
        onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left transition-colors duration-200 hover:bg-paper-sunk/60"
      >
        <span className="text-[0.82rem] font-medium text-ink">{summary}</span>
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden
             className="shrink-0 text-ink-mute transition-transform duration-300 ease-smooth"
             style={{ transform: open ? 'rotate(45deg)' : 'none' }}>
          <path d="M7 2v10M2 7h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </button>
      <div className="grid transition-[grid-template-rows] duration-300 ease-smooth"
           style={{ gridTemplateRows: open ? '1fr' : '0fr' }}>
        <div className="overflow-hidden">
          <div className="border-t border-rule px-4 py-3.5 text-[0.8rem] leading-relaxed text-ink-soft">
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}

export function Pill({ children, tone = 'neutral' }: {
  children: ReactNode; tone?: 'neutral' | 'signal' | 'moss' | 'gold'
}) {
  const cls = {
    neutral: 'border-rule-strong bg-paper text-ink-mute',
    signal: 'border-signal/30 bg-signal-soft text-signal-deep',
    moss: 'border-moss/30 bg-moss-soft text-moss',
    gold: 'border-gold/30 bg-gold-soft text-gold',
  }[tone]
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-2xs font-medium ${cls}`}>
      {children}
    </span>
  )
}

/** Sticky top navigation that tracks the section in view. */
export function Nav({ items }: { items: { id: string; label: string }[] }) {
  const [active, setActive] = useState(items[0]?.id ?? '')
  const [solid, setSolid] = useState(false)
  const ref = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (vis[0]) setActive(vis[0].target.id)
      },
      { rootMargin: '-18% 0px -62% 0px', threshold: [0.01, 0.25] },
    )
    items.forEach((i) => { const el = document.getElementById(i.id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [items])

  return (
    <nav ref={ref}
         className={`sticky top-0 z-40 border-b transition-all duration-300 ${
           solid ? 'border-rule bg-paper/92 backdrop-blur-md' : 'border-transparent bg-transparent'}`}>
      <div className="mx-auto flex w-full max-w-content items-center gap-4 px-5 py-2.5 sm:px-8">
        <a href="#top" className="shrink-0 text-2xs font-semibold tracking-tight text-ink">
          Macro&nbsp;·&nbsp;Credit
        </a>
        <div className="-mx-1 flex flex-1 gap-0.5 overflow-x-auto px-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {items.map((i) => (
            <a key={i.id} href={`#${i.id}`}
               className={`shrink-0 rounded-md px-2.5 py-1 text-2xs font-medium transition-colors duration-200 ${
                 active === i.id ? 'bg-paper-sunk text-ink' : 'text-ink-mute hover:text-ink'}`}>
              {i.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}
