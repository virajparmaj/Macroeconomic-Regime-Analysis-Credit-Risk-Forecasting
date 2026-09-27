/** Verified exports of stored research artifacts. No models are fitted in the browser. */
import researchRaw from '@/data/research.json'

export type MacroKey = 'CPI' | 'FEDFUNDS' | 'Industrial_Production' | 'GDP' | 'Unemployment_Rate' | 'Consumer_Sentiment'
export type MacroValues = Record<MacroKey, number>
export interface ResearchTimelineRow {
  date: string
  legacyMean: number
  observedMean: number
  last: number
  max: number
  min: number
  count: number
  range: number
  targetDifference: number
  macro: MacroValues
  macroChange1: Record<MacroKey, number | null>
  macroStress: number | null
  macroStressState: boolean | null
  spreadStress: boolean
  clusterId: number
  smoothedClusterId: number
  clusterProbabilities: number[]
}
export interface PredictionRow {
  originDate: string
  targetDate: string
  refitDate: string
  actual: number
  persistence: number
  rfMacroOnly: number
  rfMacroSpread: number
  lastDaily: number
}
export interface ResultContext {
  n: number
  originStart: string
  originEnd: string
  targetStart: string
  targetEnd: string
  horizonMonths: number
  source: string
  status: string
}
export interface ForecastMetric extends ResultContext {
  id: string
  label: string
  featureSet: string
  target: 'next_month_legacy_mean' | 'next_month_legacy_change'
  rmse: number
  mae: number | null
  r2Mean: number | null
  r2Persistence: number
  benchmark: string
  hasPredictionRecords: boolean
  metricPrecision: 'computed from stored predictions' | 'rounded stored log'
}
export interface AggregationMetric extends ResultContext {
  target: 'next_month_observed_daily_mean' | 'next_month_panel_ffill_mean'
  predictor: 'mean' | 'last'
  rmse: number
  mae: number
  r2Mean: number
  benchmark: string
}
export interface ClassificationMetric extends ResultContext {
  rfAuc: number
  currentSpreadAuc: number
  rfAveragePrecision: number
  currentSpreadAveragePrecision: number
  positiveMonths: number
  unknownFutureExcluded: number
  threshold: number
  thresholdSource: string
  target: string
  benchmark: string
}
export interface OnsetCount {
  thresholdSource: string
  threshold: number
  horizonMonths: number
  atRiskOrigins: number
  positiveOrigins: number
  dates: string[]
  source: string
  status: string
}
export interface ResearchData {
  meta: {
    schemaVersion: number
    generatedBy: string
    sourceHashes: Record<string, string>
    exportedSources: string[]
    spreadUnit: string
    panel: { n: number; start: string; end: string; missingCells: number }
    daily: { n: number; sourceRows: number; missingSourceRows: number; start: string; end: string; lastMonthPartial: boolean; finalMonthObservedDays: number }
    thresholds: { spreadFullSample: number; spreadFrozen2008: number; macroStress: number }
    audit: { differingMonths: number; maxDifferencePp: number; reconstructionError: number; emptyV2Cells: number; macroStressMonths: number }
    forecast: { refitBlocks: number; refitEveryMonths: number; embargoMonths: number; timing: string }
    timeline: { source: string[]; status: string; target: string; horizon: string; benchmark: string }
    classification: { featureSet: string; refitEveryMonths: number; embargoMonths: number; limitation: string }
  }
  timeline: ResearchTimelineRow[]
  march2020Daily: { date: string; value: number }[]
  predictions: PredictionRow[]
  metrics: ForecastMetric[]
  aggregation: AggregationMetric[]
  classification: ClassificationMetric[]
  onsets: OnsetCount[]
  aggregationExample: { month: string; observedMean: number; last: number; max: number; min: number; range: number; observedDays: number; meanRank: number; rangeRank: number; source: string }
}

export const research = researchRaw as ResearchData
export const timeline = research.timeline
export const predictions = research.predictions
export const forecastMetrics = research.metrics
export const aggregationMetrics = research.aggregation
export const classificationMetrics = research.classification
export const onsetCounts = research.onsets

export const macroMetadata: { key: MacroKey; label: string; source: string; unit: string; lagMonths: number }[] = [
  { key: 'CPI', label: 'U.S. consumer price index', source: 'CPIAUCSL', unit: 'index', lagMonths: 1 },
  { key: 'FEDFUNDS', label: 'Effective federal funds rate', source: 'FEDFUNDS', unit: '%', lagMonths: 1 },
  { key: 'Industrial_Production', label: 'U.S. industrial production', source: 'INDPRO', unit: 'index', lagMonths: 1 },
  { key: 'GDP', label: 'OECD GDP reference · Euro Area 19', source: 'EA19LORSGPORGYSAM', unit: '% year over year', lagMonths: 3 },
  { key: 'Unemployment_Rate', label: 'U.S. unemployment rate', source: 'UNRATE', unit: '%', lagMonths: 1 },
  { key: 'Consumer_Sentiment', label: 'Consumer sentiment', source: 'UMCSENT', unit: 'index', lagMonths: 0 },
]
export const macroMeta = Object.fromEntries(macroMetadata.map((item) => [item.key, item])) as Record<MacroKey, (typeof macroMetadata)[number]>

/** Bundled snapshots work offline and under Vite's relative deployment base. */
export function sourceHref(path: string): string {
  const cleanPath = path.replace(/^\.?\//, '')
  if (research.meta.exportedSources.includes(cleanPath)) {
    // A literal + is valid in a URL path; Vite's static server misreads an encoded %2B.
    const assetPath = cleanPath.split('/').map((part) => encodeURIComponent(part).replace(/%2B/g, '+')).join('/')
    return `${import.meta.env.BASE_URL}evidence/${assetPath}`
  }
  // Raw source downloads are not redistributed; their hashes are in this manifest.
  if (cleanPath.startsWith('data/original/')) return `${import.meta.env.BASE_URL}evidence/manifest.json`
  return `https://github.com/virajparmaj/Macroeconomic-Regime-Analysis-Credit-Risk-Forecasting/blob/HEAD/${cleanPath.split('/').map(encodeURIComponent).join('/')}`
}
