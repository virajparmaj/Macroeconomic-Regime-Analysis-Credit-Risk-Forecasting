import { Reveal } from '@/charts/primitives'
import { RegimeTimeline } from '@/charts/RegimeTimeline'
import { BenchmarkChart } from '@/charts/BenchmarkChart'
import { LeadLagChart } from '@/charts/LeadLagChart'
import { WarningChart } from '@/charts/WarningChart'
import { RegimeSplitChart } from '@/charts/RegimeSplitChart'
import { HorizonChart } from '@/charts/HorizonChart'
import { AggregationChart } from '@/charts/AggregationChart'
import { Insight } from '@/components/Insight'
import { Disclose, Metric, Nav, Pill, Section, SectionHead } from '@/components/ui'
import { MACRO_META, MACRO_KEYS, R2_MIN, regimeSplit, timeline } from '@/lib/data'
import { monthLabel, num } from '@/lib/format'

const NAV = [
  { id: 'problem', label: 'Problem' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'findings', label: 'Findings' },
  { id: 'method', label: 'Method' },
  { id: 'limits', label: 'Limits' },
  { id: 'next', label: 'Next' },
]

export default function App() {
  return (
    <>
      <a href="#problem" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-ink focus:px-3 focus:py-2 focus:text-2xs focus:text-paper">
        Skip to content
      </a>
      <Nav items={NAV} />
      <Hero />
      <Problem />
      <TimelineSection />
      <Findings />
      <Methodology />
      <Limits />
      <NextSteps />
      <Footer />
    </>
  )
}

/* ------------------------------------------------------------------ hero */

function Hero() {
  return (
    <header id="top" className="relative overflow-hidden border-b border-rule bg-paper-warm">
      <div className="mx-auto w-full max-w-content px-5 pb-16 pt-14 sm:px-8 sm:pb-20 sm:pt-20">
        <Reveal>
          <div className="mb-6 flex flex-wrap items-center gap-2">
            <Pill tone="signal">Analytical review</Pill>
            <Pill>309 months · 1996–2022</Pill>
            <Pill>US high-yield credit</Pill>
          </div>
          <h1 className="max-w-4xl text-[2.1rem] font-bold leading-[1.08] tracking-[-0.028em] text-ink sm:text-[3.2rem]">
            Can macroeconomic data tell you when credit is about to get expensive?
          </h1>
          <p className="lede mt-6 max-w-2xl">
            I built a model that said yes, with an R² of 0.81. Then I checked it against a benchmark that
            assumes nothing changes — and the benchmark won. This is what 26 years of data actually
            supports, and what it does not.
          </p>
        </Reveal>

        <Reveal delay={120}>
          <div className="mt-12 grid grid-cols-2 gap-x-6 gap-y-8 border-t border-rule pt-9 sm:gap-x-10 lg:grid-cols-4">
            <Metric
              tone="signal" value="82" unit="%"
              label="Higher error than doing nothing"
              note="Best model 1.11pp RMSE against the random walk's 0.61pp."
            />
            <Metric
              tone="gold" value="6.7" unit="×"
              label="Macro's power in stress vs calm"
              note="R² of 0.50 in stressed months against 0.07 in calm ones."
            />
            <Metric
              value="3" unit="mo"
              label="How far ahead anything is predictable"
              note="Classification AUC falls from 0.94 to 0.54 between 1 and 12 months."
            />
            <Metric
              tone="calm" value="13.6" unit="×"
              label="March 2020's repricing speed"
              note="Versus a median month — yet the monthly panel ranks it only 43rd."
            />
          </div>
        </Reveal>
      </div>
    </header>
  )
}

/* --------------------------------------------------------------- problem */

function Problem() {
  return (
    <Section id="problem" className="scroll-mt-14 py-16 sm:py-24">
      <div className="grid gap-x-12 gap-y-10 lg:grid-cols-[1.25fr_1fr]">
        <div>
          <SectionHead
            eyebrow="The question"
            title="Credit spreads are what lenders charge for risk. Can the economy tell you where they go next?"
          />
          <div className="measure mt-6 space-y-4 text-[0.94rem] leading-relaxed text-ink-soft">
            <p>
              When lenders grow nervous they demand more compensation, and the high-yield spread widens.
              It is the cleanest publicly available read on credit-market stress, and it moves years
              before default statistics do. The obvious question is whether the macro economy — inflation,
              policy rates, output, jobs, sentiment — can tell you where it is heading.
            </p>
            <p>
              This project tested that with six monthly indicators over 26 years, covering the dot-com
              bust, the global financial crisis, the post-crisis calm and COVID. The original modelling
              reported an R² of 0.81 and called it a success.
            </p>
            <p className="border-l-2 border-signal pl-4 text-ink">
              The review below started from a simpler question: <strong className="font-semibold">compared
              to what?</strong> That one question changed every conclusion.
            </p>
          </div>
        </div>

        <aside className="rounded-xl border border-rule bg-paper-warm p-6">
          <div className="eyebrow mb-4">What is in the data</div>
          <dl className="space-y-3.5">
            <div className="border-b border-rule pb-3.5">
              <dt className="text-[0.82rem] font-semibold text-ink">Outcome · Credit spread</dt>
              <dd className="mt-1 text-2xs leading-relaxed text-ink-mute">
                ICE BofA US High Yield option-adjusted spread (BAMLH0A0HYM2), monthly mean of daily
                observations. 309 months, Dec 1996 – Aug 2022.
              </dd>
            </div>
            {MACRO_KEYS.map((k) => (
              <div key={k} className="flex items-baseline justify-between gap-3">
                <dt className="text-2xs text-ink-soft">{MACRO_META[k].label}</dt>
                <dd className="shrink-0 font-mono text-2xs text-ink-faint">{MACRO_META[k].source}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-5 rounded-md border border-gold/25 bg-gold-soft/45 p-3">
            <div className="text-2xs font-semibold text-gold">One label worth correcting</div>
            <p className="mt-1 text-2xs leading-relaxed text-ink-soft">
              The column named <span className="font-mono">GDP</span> is actually{' '}
              <span className="font-mono">EA19LORSGPORGYSAM</span> — an OECD reference series for{' '}
              <strong className="font-semibold">Euro Area</strong> GDP, interpolated monthly from quarterly
              data. Not US GDP, and one of the strongest correlates in the panel.
            </p>
          </div>
        </aside>
      </div>
    </Section>
  )
}

/* -------------------------------------------------------------- timeline */

function TimelineSection() {
  const stressMonths = timeline.filter((t) => t.regime === 'stress').length
  return (
    <Section id="timeline" tone="sunk" className="scroll-mt-14 border-y border-rule py-16 sm:py-24">
      <SectionHead
        eyebrow="The whole sample in one view"
        title="Twenty-six years, three crises, and a relationship that keeps moving"
        lede="Before any individual finding, this is the shape of the problem. The spread is stable for years at a time, then repricing violently — and the strength of its link to the macro economy is not constant either."
      />
      <Reveal className="mt-10">
        <div className="rounded-xl border border-rule bg-paper p-5 sm:p-7">
          <RegimeTimeline />
        </div>
      </Reveal>
      <Reveal delay={80}>
        <div className="mt-7 grid gap-4 sm:grid-cols-3">
          <Note title="Stress is rare and clustered">
            The macro-only index flags {stressMonths} of 250 months, and they arrive in three tight
            clusters rather than spread evenly. Any model averaging across all months is mostly
            learning the quiet ones.
          </Note>
          <Note title="The link is not constant">
            Switch the lower panel to <em>Correlation</em> and pick any indicator: the rolling
            correlation swings across zero. There is no single stable coefficient to estimate.
          </Note>
          <Note title="Fit arrives late">
            With <em>Model fit</em> selected, the low point is {monthLabel(R2_MIN.d)} at{' '}
            {num(R2_MIN.r2!, 2)} — immediately before the largest credit event in the sample.
          </Note>
        </div>
      </Reveal>
    </Section>
  )
}

function Note({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-rule bg-paper p-4">
      <div className="mb-1.5 text-[0.82rem] font-semibold text-ink">{title}</div>
      <p className="text-2xs leading-relaxed text-ink-mute">{children}</p>
    </div>
  )
}

/* -------------------------------------------------------------- findings */

function Findings() {
  return (
    <Section id="findings" className="scroll-mt-14 py-16 sm:py-24">
      <SectionHead
        eyebrow="What the data tells us"
        title="Six findings, in the order that changed my mind"
        lede="Each one survived at least two robustness checks — differencing for stationarity, point-in-time alignment, sample splits, dropping crisis years, or alternative model families. Where a check failed, that is stated."
        className="mb-4"
      />

      <Insight
        index={1}
        question="Is an R² of 0.81 actually good?"
        headline="The model that looked like R² = 0.75 is 82% worse than assuming nothing changes"
        numbers={[
          { value: '0.61pp', label: 'Random-walk RMSE — the bar to clear' },
          { value: '−2.30', label: 'Model R² measured against that benchmark' },
          { value: 'p < 0.0001', label: 'Diebold–Mariano, macro-only vs random walk' },
        ]}
        chart={<BenchmarkChart />}
        interpretation={
          <>On an honest walk-forward test, predicting next month's spread with this month's value scores
          R² = 0.92. The random forest reaches 0.75 — respectable against the test mean, but far behind the
          naive rule. (The original project reported 0.81 on a single chronological split; re-run across 14
          walk-forward origins with point-in-time macro it gives 0.75. The gap between those two is not the
          point — the gap to 0.92 is.) The result survives predicting the change instead of the level,
          horizons of 1, 3, 6 and 12 months, and Ridge in place of random forest.</>
        }
        matters={
          <>R² against the test-period mean flatters any model on a series with 0.96 autocorrelation, because
          the null it beats is one no forecaster would ever use. The original number was never
          miscalculated — it was compared to the wrong thing. Every downstream conclusion inherited that.</>
        }
        implication={
          <>Make a named benchmark mandatory in reporting: quote R² against the random walk with a
          Diebold–Mariano p-value, never R² alone. It is a reporting-standard change that costs nothing
          and would have caught this at the outset.</>
        }
        limitation={
          <>This shows macro adds nothing <em>at these horizons, in this specification, for this spread's
          level</em>. It does not show macro is uninformative — Finding 3 shows it is, contemporaneously,
          under stress. Nor does it rule out richer inputs like VIX or the yield curve.</>
        }
      />

      <Insight
        index={2}
        question="Why can't the model beat a naive rule?"
        headline="The credit spread moves before the macro data, not after it"
        numbers={[
          { value: '0 of 6', label: 'Indicators that lead the spread, in both sample halves' },
          { value: 'p = 0.108', label: 'Unemployment → spread. The reverse is p < 0.0001' },
          { value: '12 / 5 / 15', label: 'Months of warning the spread gave before each recession' },
        ]}
        chart={
          <div className="space-y-12">
            <LeadLagChart />
            <div className="border-t border-rule pt-10"><WarningChart /></div>
          </div>
        }
        interpretation={
          <>Correlation between macro changes and spread changes peaks at a lead of −1 month: the spread has
          already moved. Granger tests show spread changes predict unemployment and sentiment, while the
          reverse is not significant. Under real-time publication-lag alignment the asymmetry strengthens.</>
        }
        matters={
          <>This explains Finding 1 mechanically. The spread is a market price set daily by people
          forecasting the economy; CPI and unemployment are backward-looking statistics published weeks
          after the month they describe. The project asked the slow series to anticipate the fast one.</>
        }
        implication={
          <>Invert the framing. Use the spread as a leading indicator <em>of</em> macro deterioration — which
          the data supports — rather than as the thing macro predicts. That is both a more defensible
          product and a more useful one.</>
        }
        limitation={
          <>Granger precedence is predictive, not causal: it says the spread moves first, not that it causes
          the move. Both may respond to a common unobserved driver. For CPI, Fed Funds and industrial
          production the relationship is significant in <em>both</em> directions — the defensible claim is
          the asymmetry, not exclusivity.</>
        }
      />

      <Insight
        index={3}
        question="Does macro matter at all, then?"
        headline="Macro explains credit spreads only when the economy is already under stress"
        numbers={[
          { value: '0.07 → 0.50', label: 'R² in calm months versus stressed ones' },
          { value: '83%', label: `Of all spread variation, in ${Math.round(regimeSplit.monthShare * 100)}% of months` },
          { value: '22 of 23', label: 'NBER recession months the macro-only index flags' },
        ]}
        chart={<RegimeSplitChart />}
        interpretation={
          <>Split by a stress index built from macro indicators alone, the six series jointly explain 7% of
          monthly spread changes in calm periods and 50% in stressed ones. Every indicator strengthens —
          CPI's correlation goes from −0.04 to −0.49. It behaves like a switch rather than a dial.</>
        }
        matters={
          <>A single pooled model is fitted mostly on months where its inputs carry almost no information,
          then judged on an average dominated by months it rarely sees. That is the mechanism behind the
          project's unstable results, and it reframes what the macro block is for: describing the state,
          not predicting the level.</>
        }
        implication={
          <>Report calm and stress separately rather than as one headline, and route them to different
          logic — lean on spread dynamics in calm regimes, weight macro more heavily in stressed ones.
          Make the regime label itself a monitored indicator.</>
        }
        limitation={
          <>The split sits at the 75th percentile of a constructed index — a choice, though the result holds
          at the 70th and 80th too. This is contemporaneous association: macro and spreads moving together
          in a crisis does not establish which drives which.</>
        }
      />

      <Insight
        index={4}
        question="When does the relationship actually hold?"
        headline="The macro model fits worst right before a crisis and best right after one"
        numbers={[
          { value: '0.099', label: `Lowest rolling R² in 26 years — ${monthLabel(R2_MIN.d)}` },
          { value: '+13.6pp', label: 'How far the spread widened over the next seven months' },
          { value: '−0.26', label: 'Correlation of model fit with the spread 12 months later' },
        ]}
        chart={
          <div>
            <div className="mb-5 rounded-lg border border-calm/25 bg-calm-soft/40 p-4">
              <p className="text-[0.85rem] leading-relaxed text-ink-soft">
                This finding lives in the timeline above. Set the lower panel to{' '}
                <strong className="font-semibold text-ink">Model fit</strong> and select the{' '}
                <strong className="font-semibold text-ink">Global financial crisis</strong> period: the five
                worst-fitting months (▾) cluster in the calm immediately before the spike, and the five best
                (▴) land after COVID had already repriced.
              </p>
              <a href="#timeline"
                 className="mt-3 inline-flex items-center gap-1.5 text-2xs font-semibold text-calm transition-colors hover:text-ink">
                Back to the timeline
                <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden>
                  <path d="M2 10L10 2M10 2H4M10 2v6" stroke="currentColor" strokeWidth="1.5"
                        strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </a>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <FitCard tone="signal" label="Five worst-fit months"
                       months="Nov 2007 – Aug 2008" note="The eve of the global financial crisis." />
              <FitCard tone="moss" label="Five best-fit months"
                       months="Mar 2020 – Feb 2021" note="After COVID had already repriced credit." />
            </div>
          </div>
        }
        interpretation={
          <>Rolling 60-month fit bottoms at 0.099 in May 2008 and peaks at 0.763 in April 2020. It
          correlates +0.33 with the spread nine months <em>earlier</em> and −0.26 with the spread twelve
          months later — the signature of a lagging indicator.</>
        }
        matters={
          <>It challenges the assumption that a well-fitting model is a safe one. Deteriorating fit was the
          closest thing to an advance warning here, and improving fit meant the damage was already done. A
          team watching model health in 2008 would have seen R² collapse and filed a retraining ticket.</>
        }
        implication={
          <>Track rolling R² as a monitored risk indicator with an inverted reading — a sustained fall is
          grounds for escalation, not for retraining. It costs nothing to compute and would have fired in
          late 2007.</>
        }
        limitation={
          <>Two crisis episodes drive this. A formal sup-Wald test <strong className="font-semibold">cannot
          reject</strong> parameter stability (6.11 against a ~17.5 critical value), so this is cyclical
          variation in fit, not a proven structural break — and a 60-month window mechanically inflates R²
          for years after a crisis enters it.</>
        }
      />

      <Insight
        index={5}
        question="Is there a version of this question the data can answer?"
        headline="Credit stress is predictable three months out, and not at all at twelve"
        numbers={[
          { value: '0.94 → 0.54', label: 'Classification AUC, 1 month versus 12 months ahead' },
          { value: '4.4×', label: 'Precision lift over the base rate at one month' },
          { value: '0.43', label: 'Macro-only AUC at 12 months — below a coin flip' },
        ]}
        chart={<HorizonChart />}
        interpretation={
          <>Asking "will the spread be in its top quartile in h months" instead of "what number will it be"
          produces a genuinely useful model at short horizons — AUC 0.94 at one month, 0.79 at three. Then
          it dies. Spread history alone beats macro-plus-spread at every horizon.</>
        }
        matters={
          <>It converts a negative result into a scoped deliverable. The useful artefact is a short-horizon
          stress flag, not a spread forecast — and it defines the honest planning horizon. Anyone setting
          limits twelve months out on this data is using noise.</>
        }
        implication={
          <>Ship the one-to-three-month classifier with an explicit "no signal beyond three months" caveat
          and drop point forecasts. Where a longer view is genuinely needed, that is a case for new data,
          not a longer horizon on this data.</>
        }
        limitation={
          <>Positives are scarce and clustered — 20 to 31 across 164 test months, concentrated in a handful
          of episodes — so these AUCs carry wide intervals and rest on three crises. AUC also says nothing
          about calibration: the model ranks well, but its probabilities are not yet trustworthy as levels.</>
        }
      />

      <Insight
        index={6}
        question="Was the data itself set up to see a crisis?"
        headline="Monthly averaging erased the fastest credit event in twenty-six years"
        numbers={[
          { value: '1st vs 43rd', label: 'March 2020 ranked by repricing speed vs by monthly mean' },
          { value: '13.6×', label: "That month's range against a median month" },
          { value: '6,679', label: 'Daily observations in the repo; 309 monthly rows modelled' },
        ]}
        chart={<AggregationChart />}
        interpretation={
          <>March 2020's daily spread ran from 4.75pp to 10.87pp — a wider intra-month range than any month
          of the global financial crisis. In the monthly panel the project models, it is the 43rd-worst
          month. The averaging destroys precisely the event a credit-risk system exists to catch.</>
        }
        matters={
          <>A risk system is judged on the fast, severe tail, and this aggregation compresses exactly that:
          5% understatement in a slow month like October 2002, 28% in March 2020. Speed is the discarded
          dimension, and speed separates a manageable widening from a liquidity event.</>
        }
        implication={
          <>Carry intra-month range and month-end alongside the mean — three cheap columns from data already
          in the repository — and treat range as its own monitored indicator of repricing velocity. It
          would have ranked March 2020 first in real time.</>
        }
        limitation={
          <>Month-end predicting next month's mean better is partly definitional: it sits closer in time. It
          shows the mean target is artificially smooth, not that month-end is a better target. The macro
          series remain monthly regardless, so this improves the outcome measure, not the alignment.</>
        }
      />
    </Section>
  )
}

function FitCard({ tone, label, months, note }: {
  tone: 'signal' | 'moss'; label: string; months: string; note: string
}) {
  const cls = tone === 'signal'
    ? 'border-signal/25 bg-signal-soft/40' : 'border-moss/25 bg-moss-soft/40'
  const txt = tone === 'signal' ? 'text-signal-deep' : 'text-moss'
  return (
    <div className={`rounded-lg border p-4 ${cls}`}>
      <div className={`eyebrow mb-2 ${txt}`}>{label}</div>
      <div className="text-[0.95rem] font-semibold text-ink">{months}</div>
      <p className="mt-1 text-2xs leading-relaxed text-ink-mute">{note}</p>
    </div>
  )
}

/* ------------------------------------------------------------ methodology */

function Methodology() {
  return (
    <Section id="method" tone="warm" className="scroll-mt-14 border-y border-rule py-16 sm:py-24">
      <SectionHead
        eyebrow="Method"
        title="How this was tested"
        lede="Enough detail to reproduce the results, kept out of the narrative above. Everything was computed in Python; this page renders those outputs and derives nothing in the browser."
      />
      <div className="mt-9 grid gap-3 lg:grid-cols-2">
        <Disclose summary="Data, provenance and publication lags">
          <p className="mb-3">
            Six monthly macro series from FRED plus one OECD reference series, merged with the ICE BofA US
            High Yield option-adjusted spread on a month-end key. 309 complete months, Dec 1996 – Aug 2022,
            no missing values and no gaps in the monthly grid.
          </p>
          <p className="mb-3">
            The original merge applied <strong className="font-semibold text-ink">no publication lag</strong>,
            which hands a forecaster standing at month end that month's CPI, industrial production and
            unemployment — none of which had been published. Every forecasting test here shifts each series
            by its real release lag:
          </p>
          <ul className="mb-3 space-y-1">
            {MACRO_KEYS.map((k) => (
              <li key={k} className="flex justify-between gap-3 text-2xs">
                <span>{MACRO_META[k].label}</span>
                <span className="shrink-0 font-mono text-ink-faint">
                  {MACRO_META[k].lag === 0 ? 'same month' : `+${MACRO_META[k].lag}mo`}
                </span>
              </li>
            ))}
          </ul>
          <p>
            These lags are rounded down, so they understate rather than overstate the problem. The vintages
            are also final revised series, so even correct lags do not fully reconstruct what a forecaster
            would have seen in real time.
          </p>
        </Disclose>

        <Disclose summary="Stationarity and why everything is tested on changes">
          <p className="mb-3">
            ADF and KPSS tests agree that four of the seven series — CPI, Fed Funds, industrial production
            and consumer sentiment — are non-stationary in levels. Correlating non-stationary levels risks
            spurious regression, so every correlation, Granger test and rolling diagnostic here runs on
            first differences.
          </p>
          <p>
            This matters for effective sample size too. The spread's <em>level</em> has lag-1 autocorrelation
            of 0.963, leaving roughly 6 independent observations in 309 months; differencing recovers about
            149. What differencing cannot create is a fourth crisis.
          </p>
        </Disclose>

        <Disclose summary="Walk-forward validation, embargo and benchmarks">
          <p className="mb-3">
            The original evaluation used a single chronological split. This review uses expanding-window
            walk-forward validation with 14 origins from 2009 to 2022, refitting every 12 months, with a
            3-month embargo dropping training rows whose features overlap the test block.
          </p>
          <p className="mb-3">Three benchmarks, because a model has to beat something named:</p>
          <ul className="mb-3 space-y-1.5">
            <li><strong className="font-semibold text-ink">Random walk</strong> — next month equals this month. The one that wins.</li>
            <li><strong className="font-semibold text-ink">AR(1) in levels</strong> — fitted on the training block only.</li>
            <li><strong className="font-semibold text-ink">Training mean</strong> — the honest constant, unlike the test mean plain R² uses.</li>
          </ul>
          <p>
            Comparisons use Campbell–Thompson out-of-sample R² and Diebold–Mariano with the
            Harvey–Leybourne–Newbold small-sample correction, which matters at 163 test months.
          </p>
        </Disclose>

        <Disclose summary="Regime construction and robustness checks">
          <p className="mb-3">
            The stress index averages four z-scored macro signals: a Sahm-style rise in unemployment off its
            12-month low, negative industrial-production growth, sentiment below its 60-month norm, and the
            OECD growth series inverted. It uses{' '}
            <strong className="font-semibold text-ink">no credit-spread information</strong>, so conditional
            correlations are not circular. Stress is the top quartile of that index.
          </p>
          <p className="mb-3">External validation: it flags 22 of the 23 NBER recession months inside its
            window, which opens in Nov 2001 because of the 60-month lookback.</p>
          <p>Each finding was re-run: excluding 2008–09 and 2020, under point-in-time alignment, on both
            sample halves, and with alternative model families. The lead-lag asymmetry strengthens under
            real-time alignment; outside crises the macro–credit link largely disappears in both directions.</p>
        </Disclose>
      </div>

      <div className="mt-6 rounded-lg border border-rule bg-paper p-5">
        <div className="eyebrow mb-3">Tools</div>
        <p className="text-[0.82rem] leading-relaxed text-ink-soft">
          Python with pandas, statsmodels, scikit-learn and scipy for the analysis; matplotlib for the
          static figures in the repository. This page is React, TypeScript and D3 scales, rendering
          pre-computed JSON. Scripts live in{' '}
          <span className="font-mono text-2xs text-ink">/analysis</span>, and the full written review in{' '}
          <span className="font-mono text-2xs text-ink">ANALYSIS_REPORT.md</span>.
        </p>
      </div>
    </Section>
  )
}

/* ----------------------------------------------------------------- limits */

function Limits() {
  const items = [
    {
      title: 'This is credit pricing, not credit loss',
      body: 'There are no loans, borrowers, balances or default labels here. The spread is what lenders charge for risk, not what they lose. Nothing on this page transfers directly to borrower-level default modelling.',
    },
    {
      title: 'Association and precedence, never causation',
      body: 'Granger tests establish which series moves first, not what drives what. Six correlated aggregates with no exogenous variation cannot support a causal claim, and both series may respond to a common unobserved driver.',
    },
    {
      title: 'Three recessions carry the regime evidence',
      body: 'Every regime finding rests on 2001, 2008 and 2020. The row count is 309, but the episode count is three, and that is the binding constraint on confidence.',
    },
    {
      title: 'No structural break was proven',
      body: 'A sup-Wald test cannot reject parameter stability (6.11 against ~17.5). Finding 4 describes cyclical variation in how well the relationship holds — it is not evidence of a permanent regime change.',
    },
    {
      title: 'Thresholds shown are in-sample choices',
      body: 'The 6.43pp stress line and the suggested R² escalation level were set on the full sample. In production both would have to be fitted on training data alone, which will move them.',
    },
    {
      title: 'One series is not what its name says',
      body: 'The column labelled GDP is an OECD reference series for Euro Area 19, interpolated monthly from quarterly data — not US GDP. It is among the strongest correlates in the panel, so the label affects any economic reading.',
    },
  ]
  return (
    <Section id="limits" className="scroll-mt-14 py-16 sm:py-24">
      <SectionHead
        eyebrow="Limitations"
        title="What this analysis does not establish"
        lede="These are load-bearing, not boilerplate. Each one marks a place where a confident-sounding conclusion would outrun the evidence."
      />
      <div className="mt-9 grid gap-x-10 gap-y-7 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((it, i) => (
          <Reveal key={it.title} delay={i * 40}>
            <div className="border-t border-rule-strong pt-4">
              <h3 className="text-[0.88rem] font-semibold leading-snug text-ink">{it.title}</h3>
              <p className="mt-2 text-[0.82rem] leading-relaxed text-ink-mute">{it.body}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  )
}

/* ------------------------------------------------------------------- next */

function NextSteps() {
  const steps = [
    {
      n: '01',
      title: 'Test daily financial variables that have no publication lag',
      body: 'The core problem is that macro data arrives after the spread has moved. VIX, the 10y–2y term spread and the Chicago Fed NFCI are published daily or weekly and are plausibly contemporaneous with credit repricing. The specific test: do they beat the random walk on the same walk-forward setup where macro failed?',
      tag: 'Directly tests why Finding 2 happens',
    },
    {
      n: '02',
      title: 'Reverse the model and forecast the macro from the spread',
      body: 'Granger tests already point this way. Build the mirror of the failing model — predict unemployment and industrial production three to six months ahead from spread dynamics, benchmarked against each series\' own AR model. If the spread beats those benchmarks, the project has a working product rather than a negative result.',
      tag: 'Turns the finding into a deliverable',
    },
    {
      n: '03',
      title: 'Rebuild the panel at daily and weekly frequency for the spread',
      body: 'The daily series is already in the repository. Reconstruct the target as month-end plus intra-month range and realised volatility, then re-run Finding 5\'s classifier. The question is whether repricing velocity predicts stress persistence better than the level does — which the March 2020 ranking suggests it might.',
      tag: 'Uses data already on hand',
    },
    {
      n: '04',
      title: 'Validate the regime split on out-of-sample history',
      body: 'Three episodes is the binding constraint. BAMLH0A0HYM2 begins in 1996, but Moody\'s Baa–Aaa spread runs back to 1919 and covers roughly fifteen more credit cycles. Rebuilding the macro-only stress index on that history would show whether the 6.7× calm-to-stress ratio is a stable feature or an artefact of these three crises.',
      tag: 'Addresses the biggest limitation',
    },
  ]
  return (
    <Section id="next" tone="sunk" className="scroll-mt-14 border-t border-rule py-16 sm:py-24">
      <SectionHead
        eyebrow="What next"
        title="Four analyses that would move this forward"
        lede="Each follows from a specific finding above and has a stated success criterion — not more data for its own sake."
      />
      <div className="mt-10 grid gap-x-10 gap-y-8 md:grid-cols-2">
        {steps.map((s, i) => (
          <Reveal key={s.n} delay={i * 60}>
            <div className="flex gap-4">
              <span className="tabular shrink-0 text-2xs font-semibold text-signal">{s.n}</span>
              <div className="min-w-0">
                <h3 className="text-[0.95rem] font-semibold leading-snug text-ink">{s.title}</h3>
                <p className="mt-2 text-[0.85rem] leading-relaxed text-ink-soft">{s.body}</p>
                <div className="mt-2.5"><Pill tone="neutral">{s.tag}</Pill></div>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  )
}

function Footer() {
  return (
    <footer className="border-t border-rule bg-paper py-10">
      <div className="mx-auto w-full max-w-content px-5 sm:px-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-baseline sm:justify-between">
          <p className="text-2xs leading-relaxed text-ink-mute">
            Macroeconomic Regime Analysis &amp; Credit Risk — analytical review of 309 monthly observations,
            Dec 1996 – Aug 2022.
          </p>
          <p className="text-2xs text-ink-faint">
            Data: FRED · OECD · ICE BofA. Analysis in Python; page in React and D3.
          </p>
        </div>
      </div>
    </footer>
  )
}
