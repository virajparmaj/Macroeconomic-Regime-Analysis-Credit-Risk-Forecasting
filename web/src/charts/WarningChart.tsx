/** Insight 2b: months of warning before each recession, by signal.
 *  Small multiples rather than grouped bars, because the comparison that matters
 *  is within each episode, not across them. */

import { warningTable } from '@/lib/data'
import { C, Figure } from './primitives'

const SIGNALS = [
  { key: 'spread' as const,    label: 'Credit spread',        color: C.signal, rule: 'above its 12-month mean +1 s.d.' },
  { key: 'sentiment' as const, label: 'Consumer sentiment',   color: C.calm,   rule: 'below its 12-month mean −1 s.d.' },
  { key: 'indpro' as const,    label: 'Industrial production',color: C.gold,   rule: 'year-on-year growth below zero' },
  { key: 'unemp' as const,     label: 'Unemployment',         color: C.mute,   rule: '+0.5pp off its 12-month low' },
]
const MAX = 20

export function WarningChart() {
  return (
    <Figure
      title="Unemployment gave no warning before any of the three recessions"
      subtitle="Months between a signal first firing and the NBER recession start. Longer is earlier."
      caption={
        <>The spread fired 5–15 months ahead every time; unemployment fired in the recession's own first month
        twice and never in 2020. Consumer sentiment was the earliest signal before 2008 — worth noting rather
        than smoothing over, since it is the one case where something beat the spread.</>
      }
      source="Signals evaluated over the 24 months preceding each NBER peak. Three episodes is a small base for this comparison."
    >
      <div className="grid gap-6 sm:grid-cols-3">
        {warningTable.map((ev) => (
          <div key={ev.event} className="rounded-lg border border-rule bg-paper-warm p-4">
            <div className="mb-3.5">
              <div className="text-[0.82rem] font-semibold text-ink">{ev.event}</div>
              <div className="text-2xs text-ink-mute">began {ev.start.replace('-', '·')}</div>
            </div>
            <div className="space-y-2.5">
              {SIGNALS.map((s) => {
                const v = ev.signals[s.key]
                return (
                  <div key={s.key}>
                    <div className="mb-1 flex items-baseline justify-between gap-2">
                      <span className="truncate text-2xs text-ink-soft">{s.label}</span>
                      <span className={`tabular shrink-0 text-2xs font-semibold ${
                        v == null ? 'text-ink-faint' : v === 0 ? 'text-ink-mute' : 'text-ink'}`}>
                        {v == null ? 'never' : v === 0 ? 'no lead' : `${v}m`}
                      </span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-rule">
                      {v != null && v > 0 && (
                        <div className="h-full rounded-full"
                             style={{ width: `${(v / MAX) * 100}%`, background: s.color,
                                      transition: 'width .6s cubic-bezier(.22,.61,.36,1)' }} />
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-2xs leading-relaxed text-ink-faint">
        Thresholds: {SIGNALS.map((s) => `${s.label.toLowerCase()} ${s.rule}`).join('; ')}. These are
        reasonable conventions, not optimised rules — different thresholds shift the months but not the ordering.
      </p>
    </Figure>
  )
}
