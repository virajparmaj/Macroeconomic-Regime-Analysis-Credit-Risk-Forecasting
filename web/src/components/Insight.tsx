/** The repeating unit of the narrative: finding → evidence → reading → so what.
 *  Interpretation and implication are deliberately separated, because conflating
 *  "what the data shows" with "what to do about it" is the failure mode this
 *  whole review was written against. */

import type { ReactNode } from 'react'
import { Reveal } from '@/charts/primitives'

export interface InsightNumber { value: string; label: string }

export function Insight({ index, headline, question, numbers, chart, interpretation, matters, implication, limitation }: {
  index: number
  headline: string
  question: string
  numbers: InsightNumber[]
  chart: ReactNode
  interpretation: ReactNode
  matters: ReactNode
  implication: ReactNode
  limitation: ReactNode
}) {
  return (
    <article className="scroll-mt-16 border-t border-rule py-14 first:border-t-0 sm:py-20">
      <Reveal>
        <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_15.5rem]">
          <div className="min-w-0">
            <div className="mb-3 flex items-baseline gap-3">
              <span className="tabular text-2xs font-semibold text-signal">
                {String(index).padStart(2, '0')}
              </span>
              <span className="text-2xs font-medium text-ink-mute">{question}</span>
            </div>
            <h3 className="max-w-3xl text-[1.35rem] font-bold leading-[1.25] tracking-[-0.015em] text-ink sm:text-[1.7rem]">
              {headline}
            </h3>
          </div>

          <div className="flex flex-row flex-wrap gap-x-8 gap-y-4 lg:flex-col lg:gap-y-5 lg:border-l lg:border-rule lg:pl-6">
            {numbers.map((n) => (
              <div key={n.label} className="min-w-0">
                <div className="tabular text-2xl font-bold leading-none tracking-[-0.02em] text-ink sm:text-[1.6rem]">
                  {n.value}
                </div>
                <div className="mt-1.5 text-2xs leading-snug text-ink-mute">{n.label}</div>
              </div>
            ))}
          </div>
        </div>
      </Reveal>

      <Reveal delay={60}>
        <div className="mt-9 rounded-xl border border-rule bg-paper-warm p-5 sm:p-7">{chart}</div>
      </Reveal>

      <Reveal delay={90}>
        <div className="mt-9 grid gap-x-10 gap-y-7 md:grid-cols-2">
          <Block label="What it means">{interpretation}</Block>
          <Block label="Why it matters">{matters}</Block>
          <Block label="What follows from it" tone="signal">{implication}</Block>
          <Block label="What it does not show" tone="mute">{limitation}</Block>
        </div>
      </Reveal>
    </article>
  )
}

function Block({ label, children, tone = 'default' }: {
  label: string; children: ReactNode; tone?: 'default' | 'signal' | 'mute'
}) {
  const accent = tone === 'signal' ? 'text-signal' : tone === 'mute' ? 'text-ink-faint' : 'text-ink-mute'
  return (
    <div>
      <div className={`eyebrow mb-2.5 ${accent}`}>{label}</div>
      <div className={`text-[0.88rem] leading-relaxed ${tone === 'mute' ? 'text-ink-mute' : 'text-ink-soft'}`}>
        {children}
      </div>
    </div>
  )
}
