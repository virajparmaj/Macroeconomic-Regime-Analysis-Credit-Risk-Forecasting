import { useEffect, useState, type ReactNode } from 'react'
import { Nav } from '@/components/ui'
import { Evidence, Source } from '@/components/Evidence'
import { FeatureExplorer } from '@/components/FeatureExplorer'
import { ExperimentExplorer } from '@/components/ExperimentExplorer'
import { ForecastWorkbench, AggregationComparison, StressClassification } from '@/components/ForecastWorkbench'
import { ResearchTimeline } from '@/charts/ResearchTimeline'
import { research, macroMetadata, sourceHref } from '@/lib/research'

const NAV = [
  {id:'question',label:'01 Question'}, {id:'data',label:'02 Data'},
  {id:'conditions',label:'03 Conditions'}, {id:'experiments',label:'04 Experiments'},
  {id:'forecasts',label:'05 Forecasts'}, {id:'findings',label:'06 Findings'},
  {id:'validity',label:'07 Validity'}, {id:'technical',label:'08 Technical'},
]
const PANEL = 'data/merged_macroeconomic_credit.csv'

function Chapter({id,number,kicker,title,children}: {id:string;number:string;kicker:string;title:string;children:ReactNode}) {
  return <section id={id} className="chapter"><div className="chapter-heading"><span className="chapter-number">{number}</span><div><div className="eyebrow">{kicker}</div><h2>{title}</h2></div></div><div className="chapter-content">{children}</div></section>
}

export default function App() {
  const [view,setView] = useState<'summary'|'technical'>('summary')
  useEffect(() => { document.querySelectorAll<HTMLDetailsElement>('details.technical').forEach(d => { d.open = view === 'technical' }) },[view])
  return <>
    <a className="skip-link" href="#question">Skip to research</a>
    <div className="masthead"><a href="#top" className="wordmark"><span className="brand-mark" aria-hidden="true">M/C</span> ECONOMIC RESEARCH</a><span className="masthead-note">Research portfolio <span> / </span> 1996—2022</span></div>
    <Nav items={NAV} />
    <header className="hero" id="top">
      <picture className="hero-landscape" aria-hidden="true"><source media="(max-width:640px)" srcSet={`${import.meta.env.BASE_URL}images/alpine-research-small.jpg`} /><img src={`${import.meta.env.BASE_URL}images/alpine-research.jpg`} alt="" width="1536" height="1024" fetchPriority="high" /></picture>
      <div className="hero-topline"><span className="eyebrow"><span className="status-dot" /> Empirical case study / U.S. high-yield credit</span><span className="mono">RESEARCH NOTE 001</span></div>
      <h1>Macroeconomic Regime Analysis<br/><span>&amp; Credit Spread Forecasting</span></h1>
      <div className="hero-bottom"><p className="hero-deck">An empirical study of macroeconomic conditions, U.S. high-yield credit spreads, and the limits of forecasting beyond persistence.</p><div className="hero-thesis"><span className="eyebrow">The research question</span><p>What does macroeconomic structure add after we account for what the spread already tells us?</p><a className="text-link" href="#forecasts">Explore the evidence <span>↘</span></a></div></div>
      <div className="hero-stats">
        <a href="#data"><strong>309<small> months</small></strong><span>Six macro series, one aggregate spread</span><small>Core panel · Dec 1996–Aug 2022</small></a>
        <a href="#forecasts"><strong>0.6124<small> pp</small></strong><span>Monthly persistence RMSE</span><small>163 origins · next-month legacy mean</small></a>
        <a href="#information"><strong>31.22<small>%</small></strong><span>Lower RMSE with latest daily spread</span><small>Same origins and target · exploratory</small></a>
      </div>
    </header>
    <main>
      <div className="reading-toolbar"><span className="eyebrow">Read the argument. Inspect the evidence.</span><div className="segmented" aria-label="Reading detail"><button aria-pressed={view === 'summary'} onClick={() => setView('summary')}>Summary view</button><button aria-pressed={view === 'technical'} onClick={() => setView('technical')}>Technical view</button></div></div>
      <details className="brief" id="brief"><summary><span><span className="eyebrow">Start here</span><b>The 60-second hiring-manager view</b></span><span aria-hidden="true">↗</span></summary><div className="brief-body">
        <ol><li><b>Question.</b> Can macroeconomic conditions and statistical regimes improve forecasts of an aggregate high-yield credit spread beyond persistence?</li><li><b>Data.</b> A 309-month panel with six macro variables and BAMLH0A0HYM2; 6,679 observed daily spread values reveal what monthly aggregation compresses.</li><li><b>Methods.</b> Feature engineering, historical HMM/PCA/clustering, regression and sequence models, then walk-forward benchmarks and corrected stress diagnostics.</li><li><b>Strongest results.</b> Five reproduced learned specifications lose to monthly persistence on 163 origins. Using the latest daily spread reduces benchmark RMSE by 31.22% on the same legacy target. Current spread ranks future stress above RF at all four inspected horizons.</li><li><b>Judgment.</b> The original high fit is affected by target leakage. Revised macro data, retrospective regimes and three frozen-threshold stress entries limit interpretation. A matched regime-ablation study remains pending.</li></ol>
        <a className="text-link" href="#technical">Follow the sources and reproduction path ↘</a>
      </div></details>
      <Chapter id="question" number="01" kicker="Question" title="Useful prediction has to clear a credible benchmark.">
        <div className="split-copy"><p className="section-lede">Credit spreads summarize compensation demanded in a risky bond market. The research asks whether macroeconomic conditions help explain their behavior—and whether that relationship survives a forecasting test.</p><div><p>The target is <code>Credit_Spread</code>: the ICE BofA U.S. High Yield Index Option-Adjusted Spread, FRED series <code>BAMLH0A0HYM2</code>. It is an aggregate credit-market risk indicator.</p><p>Useful predictive value means lower out-of-sample error than a named benchmark, on the same target, dates and information set. Regime membership alone is not evidence of forecasting value.</p></div></div>
        <div className="scope-line"><b>Scope</b><p>Credit-market pricing. No borrower default probabilities, consumer credit scores, loan approvals, realized credit-loss forecasts, portfolio VaR or expected-loss estimates.</p></div>
        <p className="proposed"><span className="badge">Pending study</span> Give models and benchmarks the same latest spread information, then test whether regime features add predictive value.</p>
      </Chapter>
      <Chapter id="data" number="02" kicker="Data & provenance" title="One panel. Several processing stages.">
        <p className="section-lede">Dataset dimensions belong to artifacts, not to the project as a whole.</p>
        <div className="table-wrap"><table><caption>Verified stored stages</caption><thead><tr><th>Artifact</th><th>Dimensions</th><th>What it contains</th></tr></thead><tbody>
          <tr><th><Source path={PANEL}>Core monthly panel</Source></th><td>309 × 8</td><td>Six macro variables, spread target and date; Dec 1996–Aug 2022. No missing stored cells.</td></tr>
          <tr><th><Source path="notebooks/viraj/p1_models.ipynb">Phase 1 stored feature output</Source></th><td>306 × 23</td><td>Indexed notebook frame after historical feature construction. Current helper defaults produce a different feature set.</td></tr>
          <tr><th><Source path="data/datamerged_macro_credit_with_regimes.csv">Regime-enriched panel</Source></th><td>309 × 21</td><td>Date, core variables, hard label, ten GMM probabilities, smoothed label and analyst name.</td></tr>
          <tr><th><Source path="research/evidence/audit_summary.json">Underlying daily spread</Source></th><td>6,679 observed</td><td>83 missing source rows among 6,762; Dec 31, 1996–Aug 1, 2022.</td></tr>
        </tbody></table></div>
        <details className="technical"><summary>Source dictionary: six macro variables</summary><div className="technical-body table-wrap"><table><caption>Code names preserved; economic labels corrected</caption><thead><tr><th>Code</th><th>Economic meaning</th><th>Source series</th><th>Units / assumed lag</th></tr></thead><tbody>{macroMetadata.map(m => <tr key={m.key}><th><code>{m.key}</code></th><td>{m.label}</td><td><a href={`https://fred.stlouisfed.org/series/${m.source}`} target="_blank" rel="noreferrer">{m.source} ↗</a></td><td>{m.unit} / {m.lagMonths} months</td></tr>)}</tbody></table><p className="source-note">Lags describe the later evaluation’s publication-lag approximations, applied to revised data. They are not verified vintage-specific availability. The GDP column is Euro Area 19, never U.S. GDP.</p></div></details>
        <div className="research-grid aggregation-definitions"><div><span className="badge">Target A / legacy</span><h3>Forward-filled source-date mean</h3><p>The original workflow fills missing values on the dates present in the daily file, then averages by month. It does not create a full calendar-day grid.</p></div><div><span className="badge">Target B / reconstructed</span><h3>Observed-day-only mean</h3><p>A separate reconstruction averages only nonmissing daily observations. It differs from the legacy mean in <b>{research.meta.audit.differingMonths} months</b>, by up to <b>{(research.meta.audit.maxDifferencePp*100).toFixed(2)} basis points</b>.</p></div></div>
        <p className="callout">Spread levels, MAE and RMSE use percentage points (pp); MSE uses pp². <b>1 pp = 100 basis points.</b> August 2022 is a partial month: the daily source stops on August 1. Complete stored cells do not establish real-time completeness.</p>
        <Evidence source={['research/evidence/audit_summary.json','research/evidence/target_provenance.csv',PANEL]} target="Legacy and observed-day monthly spread means, separately defined" dates="Dec 1996–Aug 2022" n="309 months / 6,679 observed daily values" />
      </Chapter>
      <Chapter id="conditions" number="03" kicker="Macro conditions" title="Read the cycle without conflating its definitions.">
        <p className="section-lede">Explore the spread alongside one macro series. Separate panels preserve each variable’s units; overlays describe different statistical or economic reference states.</p>
        <ResearchTimeline />
        <details className="technical"><summary>Three regime concepts—and a recession reference</summary><div className="technical-body definition-grid">
          <article><span className="eyebrow">A / Historical unsupervised</span><h3>Clusters describe a statistical partition.</h3><p>Earlier work explored HMMs. The later workflow standardized macro inputs, explored PCA and Kernel PCA, compared K-Means and mixtures, and selected K-Means with k=10. It then explicitly refit a ten-component GMM to obtain soft membership probabilities.</p><p>GMM labels and probabilities in the saved CSV come from that refit. K-Means does not generate native probabilities. Full-sample preprocessing and centered three-month smoothing make these retrospective. Economic names are analyst interpretations; cluster IDs are the default.</p><Source path="notebooks/viraj/unsupervised.ipynb">Frozen regime workflow</Source></article>
          <article><span className="eyebrow">B / Macro-only stress</span><h3>A different construction, with different limits.</h3><p>Unemployment above its trailing 12-month minimum; negative industrial-production year-over-year growth; negative consumer-sentiment trailing 60-month z-score; and negative Euro-area GDP reference value.</p><p>Each component is standardized over the full available sample, then averaged. Values at or above its upper-quartile threshold are flagged. Spreads are not inputs, but full-sample scaling and the threshold remain retrospective.</p><Source path="analysis/macro_regime.py">Index construction</Source></article>
          <article><span className="eyebrow">C / Spread-defined stress</span><h3>The market variable defines its own state.</h3><p>The full-sample legacy spread’s 75th percentile defines the exploratory high-spread state. This conditions directly on the spread; it is not a macro regime or a recession designation.</p><p>NBER reference periods are separate historical date bands. Cluster membership, macro stress, high spreads, recession and entry into a new stress episode are distinct concepts.</p><Source path="research/audit_checks.py">Stress diagnostics</Source></article>
        </div></details>
      </Chapter>
      <Chapter id="experiments" number="04" kicker="Features & experiments" title="Follow the construction before trusting the fit.">
        <p className="section-lede">The project spans explicit features, statistical regimes, classical regressions and sequence models. Their targets, features and validation procedures differ.</p>
        <FeatureExplorer />
        <div className="leakage-note"><span className="eyebrow">A finding that changes the interpretation</span><h3>The current target can be reconstructed exactly.</h3><code className="formula">Credit_Spread(t) = 3 × Credit_Spread_rollmean3(t)<br/>− Credit_Spread_lag1(t) − Credit_Spread_lag2(t)</code><p>The historical three-month mean includes the current observation. Together with two lags, it exposes the contemporaneous target. The current spread × FEDFUNDS interaction also exposes it when the rate is nonzero.</p><p>This differs from using an available spread at time t to forecast t+1: that can be valid under explicit availability assumptions.</p><details className="technical"><summary>The future-target naming defect is a code-drift risk</summary><div className="technical-body"><p>The frozen Phase 1 frame contains <code>Target</code>, and the model excludes that column. Current helpers create <code>Target_1step_ahead</code>; the old exclusion would miss it if reused. The evidence proves rolling-feature leakage in the stored run, not that this renamed future target was used in that run.</p><Source path="src/features.py">Corrected feature helpers</Source></div></details><Evidence source={['research/evidence/audit_summary.json','notebooks/v2/00_audit_of_v1.ipynb','utilities/functions.py']} target="Contemporaneous Credit_Spread(t)" dates="Historical Phase 1 feature sample, Feb 1997–Jul 2022" n="306 feature rows" horizon="Same-month target reconstruction" benchmark="Algebraic identity; not forecast performance" status="Verified leakage diagnosis" /></div>
        <ExperimentExplorer technical={view === 'technical'} />
      </Chapter>
      <Chapter id="forecasts" number="05" kicker="Forecasts & benchmarks" title="The benchmark is part of the research design.">
        <ForecastWorkbench />
        <div id="information"><AggregationComparison /></div>
        <StressClassification />
      </Chapter>
      <Chapter id="findings" number="06" kicker="Supported findings" title="What the evidence supports today.">
        <div className="findings-list">
          <Finding n="01" metric="5 / 5" label="learned specifications lose" title="Persistence is a demanding baseline." body="The reproduced RF level, RF change and Ridge change specifications all have higher RMSE on the inspected 163 origins. This is a conclusion about these specifications and this evaluation." source="research/paper_preparation/04_VERIFIED_FINDINGS.md" href="#forecasts" />
          <Finding n="02" metric="31.22%" label="lower benchmark RMSE" title="The origin’s information matters." body="Latest daily spread improves on the previous monthly mean for the same next-month legacy target and origins. A matched comparison giving models that endpoint information is still pending." source="research/evidence/aggregation_benchmarks.csv" href="#information" />
          <Finding n="03" metric="4 / 4" label="horizons favor current spread" title="The simple ranking remains competitive." body="Current spread has higher AUC and average precision than RF after unknown future outcomes are excluded. This full-sample-threshold diagnostic does not establish crisis-warning performance." source="research/evidence/classification_label_diagnostic.csv" href="#forecasts" />
          <Finding n="04" metric="80" label="monthly definitions differ" title="Aggregation is a modeling choice." body="Observed-day and legacy forward-filled means differ in 80 of 309 months, by up to 14.61 bp. March 2020’s 6.12 pp range describes within-month dispersion." source="research/evidence/target_provenance.csv" href="#data" />
          <Finding n="05" metric="3" label="frozen-threshold entries" title="Many months do not mean many events." body="September 2011, January 2016 and March 2020 are the three post-2008 entries under the threshold frozen through 2008. Event forecasting and its uncertainty are unfinished." source="research/evidence/onset_counts.csv" href="#validity" />
        </div>
        <p className="source-note">Forecast findings use Jan 2009–Jul 2022 origins and the next-month legacy target unless a horizon is stated. Their full units, dates, counts, thresholds and evidence status appear in the linked result sections.</p>
      </Chapter>
      <Chapter id="validity" number="07" kicker="Validity & limitations" title="The boundaries are part of the result.">
        <div className="limitations">
          <article><span>01</span><h3>Target leakage</h3><p>Contemporaneous target-derived features invalidate a future-forecast interpretation of the original high fit. Corrected helpers are not proof that the final study ran.</p></article>
          <article><span>02</span><h3>Availability & revisions</h3><p>Publication-lag approximations use revised macro data. Historical vintages, live endpoint availability and the partial final month remain limitations.</p></article>
          <article><span>03</span><h3>Retrospective regimes</h3><p>Full-sample scaling, clustering, thresholds and centered smoothing look beyond individual historical forecast origins.</p></article>
          <article><span>04</span><h3>Few observations, fewer events</h3><p>309 months and only three frozen-threshold entries limit what can be learned. Serial dependence and overlapping outcomes require suitable uncertainty estimates.</p></article>
          <article><span>05</span><h3>Association & economic scope</h3><p>Statistical regimes do not identify causal mechanisms. OAS forecasting is not default calibration, trading profitability or portfolio loss modeling.</p></article>
          <article><span>06</span><h3>Incomplete experiments</h3><p>XGBoost has no verified successful result. Six v2 implementation notebooks remain empty. Planned studies cannot be represented by historical model scores.</p></article>
        </div>
        <details className="technical"><summary>Research agenda: explicitly not yet run</summary><div className="technical-body"><ol className="pending-list"><li><b>Matched regime ablation.</b> Give baselines and models the same latest-spread information; estimate the incremental contribution of regimes.</li><li><b>Frozen-threshold event forecasting.</b> Distinguish occupancy, at-risk origins and new entries; evaluate false alarms and missed events.</li><li><b>Dependent-data uncertainty.</b> Quantify paired error differences with methods that respect serial dependence and overlapping outcomes.</li><li><b>Historical vintages.</b> Reconstruct availability and revision timing.</li><li><b>External or later-period validation.</b> Test a frozen specification beyond the inspected sample.</li></ol><Source path="research/paper_preparation/08_RESULTS_REGISTER.md">Completed versus pending results register</Source></div></details>
      </Chapter>
      <Chapter id="technical" number="08" kicker="Technical details" title="A traceable path from artifact to claim.">
        <p className="section-lede">The interface exports and visualizes existing evidence. It does not train models, generate live forecasts or fill missing prediction artifacts.</p>
        <div className="pipeline"><span>Source CSVs</span><b>→</b><span>Documented transformations</span><b>→</b><span>Stored outputs</span><b>→</b><span>Checked export</span><b>→</b><span>Scoped interpretation</span></div>
        <div className="research-grid"><div><h3>Evidence map & source register</h3><p>The inventory separates verified descriptions, reproduced forecasts, historical compromised results, unsuccessful attempts and proposed work.</p><div className="source-links"><Source path="web/EVIDENCE_MAP.md">Read the evidence map</Source><Source path="research/paper_preparation/04_VERIFIED_FINDINGS.md">Verified findings</Source><Source path="research/paper_preparation/08_RESULTS_REGISTER.md">Results register</Source><a href={`${import.meta.env.BASE_URL}evidence/manifest.json`} target="_blank" rel="noreferrer">Source hashes and snapshot manifest ↗</a></div></div><div><h3>Reproduce the presentation</h3><pre><code># Python environment with NumPy + pandas<br/>python3 web/scripts/export_research.py<br/>cd web<br/>npm ci<br/>npm run typecheck<br/>npm run build</code></pre><p className="source-note">The exporter reads existing artifacts and verifies their alignment and metrics. It never executes original notebooks or fits models. Model regeneration requires its own documented environment and validation protocol.</p></div></div>
        <details className="technical"><summary>Corrected or omitted website claims</summary><div className="technical-body"><ul><li>The original RF R² ≈ 0.8084 remains historical, with target leakage disclosed.</li><li>Removed a universal three-month predictability limit and claims of reliable crisis warning.</li><li>Replaced “repricing speed” with within-month range and removed unsupported causal explanations.</li><li>Replaced “point-in-time” with lag approximations on revised data.</li><li>Kept the GDP code name while correcting its economic label to Euro Area 19.</li><li>Excluded old hard-coded performance and significance claims from the primary evidence layer.</li><li>Retained successful original model experiments separately from corrected forecast comparisons.</li></ul><Source path="web/EVIDENCE_MAP.md">Audit decisions</Source></div></details>
        <p className="source-note">Source artifacts, including frozen notebook snapshots, open locally. External FRED catalog links require a connection. The original notebooks remain frozen. Evidence is exploratory unless explicitly labeled otherwise.</p>
      </Chapter>
    </main>
    <footer><a className="wordmark" href="#top">M/C <span>Macroeconomic Regime Analysis</span></a><p>A research portfolio in economic data, modeling and statistical judgment.</p><a href={sourceHref('web/EVIDENCE_MAP.md')}>Evidence first ↗</a></footer>
  </>
}

function Finding({n,metric,label,title,body,source,href}: {n:string;metric:string;label:string;title:string;body:string;source:string;href:string}) {
  return <article className="finding"><span className="finding-number">{n}</span><div><h3>{title}</h3><p>{body}</p><div className="finding-links"><Source path={source}>Evidence source</Source><a href={href}>Inspect result ↗</a></div></div><div className="finding-metric"><strong>{metric}</strong><span>{label}</span></div></article>
}
