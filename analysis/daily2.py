import pandas as pd, numpy as np
pd.set_option('display.width',220)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
d = pd.read_csv(f'{ROOT}/data/original/ICE BofA US High Yield Index Option-Adjusted Spread_BAMLH0A0HYM2.csv',
                parse_dates=['observation_date'], index_col='observation_date').sort_index()
d.columns=['spread']; s=d['spread'].dropna()
m = s.resample('ME').agg(['mean','max','min','last','std','count'])
m['rng']=m['max']-m['min']
m=m.dropna(subset=["mean"])
m.index=m.index.to_period("M")
print("=== Peak compression (mean understates the month's worst day) ===")
for lbl, per in [('GFC peak','2008-12'),('GFC onset','2008-10'),('COVID','2020-03'),('COVID pre','2020-02'),
                 ('2002 telecom','2002-10'),('2011 EU','2011-10'),('2016 oil','2016-02'),('2018Q4','2018-12')]:
    r=m.loc[per]
    print(f"  {lbl:<13}{per}: mean={r['mean']:5.2f}  max={r['max']:5.2f}  last={r['last']:5.2f}  range={r['rng']:4.2f}  mean understates peak by {100*(r['max']-r['mean'])/r['max']:4.1f}%")

print("\n=== Ranks of March 2020 among 309 months ===")
for c in ['mean','max','rng','std']:
    print(f"  by monthly {c:<5}: rank {int((m[c]>m.loc['2020-03',c]).sum())+1} of {len(m)}")

print("\n=== Top 10 by intra-month RANGE (how fast credit repriced) ===")
t=m.sort_values('rng',ascending=False).head(10)[['mean','min','max','rng','last']]
t['era']=['GFC' if x.year in (2008,2009) else ('COVID' if x.year==2020 else str(x.year)) for x in t.index]
print(t.round(2).to_string())

print("\n=== Distribution of intra-month range ===")
print(m['rng'].describe(percentiles=[.5,.9,.95,.99]).round(3).to_string())
print(f"  2020-03 range = {m.loc['2020-03','rng']:.2f} = {m.loc['2020-03','rng']/m['rng'].median():.1f}x the median month")

# month-end vs month-mean as an EARLY WARNING device
print("\n\n=== Month-END (last) vs month-MEAN as a warning signal ===")
mm = m['mean']; ml=m['last']
# define stress event: monthly mean crosses above its own 90th pct (8.17) - use widening episodes instead
# Better: for each big widening episode, when does each series first exceed a threshold?
print("Cross-correlation: does last(t) lead mean(t+1)?")
for k in [0,1,2]:
    print(f"  corr( last(t), mean(t+{k}) ) = {ml.corr(mm.shift(-k)):.4f}")
print(f"  corr( mean(t), mean(t+1) )   = {mm.corr(mm.shift(-1)):.4f}")
# does last beat mean at predicting next month's mean? (random walk comparison)
cmp = pd.DataFrame({'y':mm.shift(-1),'from_mean':mm,'from_last':ml}).dropna()
for c in ['from_mean','from_last']:
    e=cmp['y']-cmp[c]; print(f"  RW using {c:<10}: RMSE={np.sqrt((e**2).mean()):.4f}  MAE={e.abs().mean():.4f}")
imp = (np.sqrt(((cmp['y']-cmp['from_mean'])**2).mean()) - np.sqrt(((cmp['y']-cmp['from_last'])**2).mean()))
print(f"  -> month-end RW is {100*imp/np.sqrt(((cmp['y']-cmp['from_mean'])**2).mean()):.1f}% lower RMSE")
m.to_csv('monthly_from_daily.csv')
