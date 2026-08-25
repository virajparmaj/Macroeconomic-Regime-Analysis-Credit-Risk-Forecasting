import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',250)
RR=pd.read_csv('rolling_r2.csv',parse_dates=['date'],index_col='date')['r2']
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
cs=df['Credit_Spread']
print("=== Rolling-60m macro R2 around each credit crisis ===")
for lbl,peak in [('2002 telecom','2002-10-31'),('GFC','2008-12-31'),('COVID','2020-03-31')]:
    pk=pd.Timestamp(peak)
    print(f"\n  {lbl} (spread peak {pk.date()}, spread={cs.loc[pk]:.2f}):")
    for off in [-24,-18,-12,-6,-3,0,3,6,12]:
        d=pk+pd.DateOffset(months=off)
        d=RR.index[RR.index.get_indexer([d],method='nearest')][0]
        print(f"    t{off:+4d}m ({d.date()}): rolling R2 = {RR.loc[d]:.3f}   spread = {cs.reindex([d]).iloc[0]:.2f}")

print("\n\n=== Is rolling R2 a LEADING or LAGGING indicator of spread stress? ===")
sp=cs.reindex(RR.index)
for k in range(-12,13,3):
    print(f"  corr( rollingR2(t), spread(t+{k:>3}) ) = {RR.corr(sp.shift(-k)):+.3f}")
print("\n  (positive k = R2 leads spread; the largest correlation shows which way information flows)")
best=max(range(-12,13),key=lambda k: RR.corr(sp.shift(-k)))
print(f"  max at k={best}  corr={RR.corr(sp.shift(-best)):+.3f}")

print("\n=== The pre-crisis blind spot, stated as numbers ===")
lo=RR.idxmin()
print(f"  All-time minimum rolling R2 = {RR.min():.3f} in {lo.strftime('%B %Y')}")
nxt=cs.loc[lo:lo+pd.DateOffset(months=7)]
print(f"  Spread over the following 7 months: {nxt.min():.2f} -> {nxt.max():.2f}  (+{nxt.max()-cs.loc[lo]:.2f}pp from that month)")
print(f"  Rolling R2 one year later ({(lo+pd.DateOffset(months=12)).strftime('%b %Y')}): "
      f"{RR.reindex([RR.index[RR.index.get_indexer([lo+pd.DateOffset(months=12)],method='nearest')][0]]).iloc[0]:.3f}")
print(f"\n  Bottom 5 months by rolling R2:"); print(RR.nsmallest(5).round(3).to_string())
print(f"\n  Top 5 months by rolling R2:"); print(RR.nlargest(5).round(3).to_string())
