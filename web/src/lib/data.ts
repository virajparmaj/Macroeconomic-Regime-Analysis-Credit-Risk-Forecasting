/** Typed access to the pre-computed analysis outputs.
 *
 *  Every figure was computed in Python (see /analysis) and exported by
 *  analysis/export_web.py. Nothing is recomputed in the browser: the site
 *  renders results, it does not derive them. */

import timelineRaw from '@/data/timeline.json'
import regimeSplitRaw from '@/data/regimeSplit.json'
import xcorrRaw from '@/data/xcorr.json'
import warningsRaw from '@/data/warnings.json'
import grangerRaw from '@/data/granger.json'
import predsRaw from '@/data/preds.json'
import modelMetricsRaw from '@/data/modelMetrics.json'
import horizonRaw from '@/data/horizon.json'
import aggregationRaw from '@/data/aggregation.json'

export type MacroKey = 'cpi' | 'fedfunds' | 'indpro' | 'eagdp' | 'unemp' | 'sentiment'

export interface TimelineRow {
  d: string
  spread: number
  nber: boolean
  stress: number | null
  regime: 'calm' | 'stress' | null
  r2: number | null
  cpi: number; fedfunds: number; indpro: number
  eagdp: number; unemp: number; sentiment: number
  rc_cpi: number | null; rc_fedfunds: number | null; rc_indpro: number | null
  rc_eagdp: number | null; rc_unemp: number | null; rc_sentiment: number | null
}

export interface RegimeRow {
  key: MacroKey
  full?: number; calm: number; stress: number
  calmLo?: number; calmHi?: number; stressLo?: number; stressHi?: number
}
export interface RegimeVariant {
  rows: RegimeRow[]; nCalm: number; nStress: number
  r2Calm: number; r2Stress: number; r2Full: number
}

export const timeline = timelineRaw as TimelineRow[]
export const regimeSplit = regimeSplitRaw as {
  macroOnly: RegimeVariant
  spreadLevel: RegimeVariant
  varianceShare: number; monthShare: number
  nHighSpread: number; nAll: number
}
export const xcorr = xcorrRaw as {
  lags: number[]
  series: { key: MacroKey; values: number[] }[]
  mean: number[]
  spreadFirstSide: number; macroFirstSide: number
}
export const warningTable = warningsRaw as {
  event: string; start: string
  signals: { spread: number | null; sentiment: number | null; indpro: number | null; unemp: number | null }
}[]
export const granger = grangerRaw as {
  key: MacroKey; label: string; spreadToMacro: number; macroToSpread: number
}[]
export const preds = predsRaw as {
  d: string; actual: number; rw: number; macroSpread: number; macroOnly: number
}[]
export const modelMetrics = modelMetricsRaw as {
  model: string; sub: string; rmse: number; r2Mean: number; r2Rw: number; dmP: number | null
}[]
export const horizon = horizonRaw as {
  h: number[]
  auc: Record<FeatureSet, number[]>
  lift: Record<FeatureSet, number[]>
  threshold: number
}
export type FeatureSet = 'spreadOnly' | 'macroSpread' | 'macroOnly'

export interface AggWindow {
  label: string; from: string; to: string
  daily: { d: string; v: number }[]
  monthly: { m: string; mean: number; max: number; min: number }[]
}
export const aggregation = aggregationRaw as {
  windows: Record<'covid' | 'gfc' | 'telecom', AggWindow>
  topRanges: { m: string; rng: number; mean: number; era: string }[]
  medianRange: number; rankByRange: number; rankByMean: number; nMonths: number
}

/** Display metadata for the six macro indicators. `source` is the FRED / OECD id
 *  the series was actually downloaded from — which for `eagdp` is not what the
 *  original column name claimed. */
export const MACRO_META: Record<MacroKey, {
  label: string; short: string; unit: string; source: string; lag: number
}> = {
  fedfunds:  { label: 'Fed Funds rate',        short: 'Fed Funds',  unit: '%',     source: 'FEDFUNDS',          lag: 1 },
  cpi:       { label: 'Consumer Price Index',  short: 'CPI',        unit: 'index', source: 'CPIAUCSL',          lag: 1 },
  eagdp:     { label: 'OECD GDP, Euro Area 19',short: 'OECD GDP',   unit: '% y/y', source: 'EA19LORSGPORGYSAM', lag: 3 },
  sentiment: { label: 'Consumer sentiment',    short: 'Sentiment',  unit: 'index', source: 'UMCSENT',           lag: 0 },
  indpro:    { label: 'Industrial production', short: 'Ind. prod.', unit: 'index', source: 'INDPRO',            lag: 1 },
  unemp:     { label: 'Unemployment rate',     short: 'Unemp.',     unit: '%',     source: 'UNRATE',            lag: 1 },
}
export const MACRO_KEYS = Object.keys(MACRO_META) as MacroKey[]

/** Named episodes used for the timeline's era shortcuts and annotations. */
export const ERAS = [
  { id: 'all',      label: 'Full sample', from: '1996-12', to: '2022-08' },
  { id: 'dotcom',   label: 'Dot-com & telecom', from: '1999-01', to: '2003-12' },
  { id: 'gfc',      label: 'Global financial crisis', from: '2006-01', to: '2010-12' },
  { id: 'postgfc',  label: 'Post-crisis calm', from: '2011-01', to: '2019-12' },
  { id: 'covid',    label: 'COVID', from: '2019-06', to: '2022-08' },
] as const
export type EraId = (typeof ERAS)[number]['id']

export const NBER_BANDS = [
  { from: '2001-03', to: '2001-11', label: '2001 recession' },
  { from: '2007-12', to: '2009-06', label: 'Global financial crisis' },
  { from: '2020-02', to: '2020-04', label: 'COVID recession' },
]

/** Derived constants quoted in the narrative. Computed once from the exported
 *  series rather than hard-coded, so prose and charts cannot drift apart. */
const r2rows = timeline.filter((t) => t.r2 != null)
export const R2_MIN = r2rows.reduce((a, b) => (b.r2! < a.r2! ? b : a))
export const R2_MAX = r2rows.reduce((a, b) => (b.r2! > a.r2! ? b : a))
export const R2_WORST5 = [...r2rows].sort((a, b) => a.r2! - b.r2!).slice(0, 5).map((t) => t.d)
export const R2_BEST5 = [...r2rows].sort((a, b) => b.r2! - a.r2!).slice(0, 5).map((t) => t.d)
export const SPREAD_MAX = timeline.reduce((a, b) => (b.spread > a.spread ? b : a))
