/** Shared number and date formatting. Every figure on the page routes through here
 *  so units and precision stay consistent between prose, tables and charts. */

export const pp = (v: number, d = 2) => `${v.toFixed(d)}pp`
export const num = (v: number, d = 2) => v.toFixed(d)
export const signed = (v: number, d = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`
export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`
export const mult = (v: number, d = 1) => `${v.toFixed(d)}×`

/** p-values below the resolution of the test are reported as an upper bound,
 *  never as a literal zero. */
export const pval = (p: number | null | undefined) => {
  if (p == null) return '—'
  if (p < 0.0001) return 'p < 0.0001'
  if (p < 0.001) return `p = ${p.toFixed(4)}`
  return `p = ${p.toFixed(3)}`
}

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

/** '2008-05' -> 'May 2008' */
export const monthLabel = (ym: string) => {
  const [y, m] = ym.split('-')
  return `${MONTHS[Number(m) - 1]} ${y}`
}
/** '2008-05' -> 'May ’08' */
export const monthShort = (ym: string) => {
  const [y, m] = ym.split('-')
  return `${MONTHS[Number(m) - 1]} ’${y.slice(2)}`
}
/** '2020-03-23' -> '23 Mar 2020' */
export const dayLabel = (ymd: string) => {
  const [y, m, d] = ymd.split('-')
  return `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}`
}
/** Fractional year, for continuous time scales. */
export const toYear = (ym: string) => {
  const [y, m] = ym.split('-').map(Number)
  return y + (m - 1) / 12
}
export const dayToYear = (ymd: string) => {
  const [y, m, d] = ymd.split('-').map(Number)
  return y + (m - 1) / 12 + (d - 1) / 365
}
