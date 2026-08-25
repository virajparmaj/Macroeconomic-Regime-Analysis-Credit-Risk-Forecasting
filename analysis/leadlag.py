import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',250)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'
dd=df.diff()

print("=== CROSS-CORRELATION: corr( macro_change(t-k), spread_change(t) ) ===")
print("   k>0 => macro LEADS spread (useful for early warning). k<0 => spread leads macro.\n")
ks=list(range(-6,13))
out={}
for c in macro:
    out[c]={k: dd[c].shift(k).corr(dd[cs]) for k in ks}
X=pd.DataFrame(out).T
X.columns=[f"k={k}" for k in ks]
print(X.round(3).to_string())
print("\nBest LEADING lag per variable (k>=1, macro leads):")
for c in macro:
    row={k:out[c][k] for k in ks if k>=1}
    bk=max(row,key=lambda k:abs(row[k]))
    print(f"  {c:<24} best k={bk:>2}m  corr={row[bk]:+.3f}   (contemporaneous k=0: {out[c][0]:+.3f})")
print("\nBest LAGGING lag per variable (k<=-1, spread leads macro):")
for c in macro:
    row={k:out[c][k] for k in ks if k<=-1}
    bk=max(row,key=lambda k:abs(row[k]))
    print(f"  {c:<24} best k={bk:>2}m  corr={row[bk]:+.3f}")

print("\n\n=== Does the SPREAD lead the macro data, or vice versa? (Granger, spread<->unemployment) ===")
from statsmodels.tsa.stattools import grangercausalitytests
pairs=[('Unemployment_Rate',cs),('Industrial_Production',cs),('Consumer_Sentiment',cs)]
for a,b in pairs:
    for direction,(u,v) in [(f"{b} -> {a}",(a,b)),(f"{a} -> {b}",(b,a))]:
        d2=pd.concat([dd[u],dd[v]],axis=1).dropna()
        try:
            res=grangercausalitytests(d2,maxlag=6,verbose=False)
            ps={l:res[l][0]['ssr_ftest'][1] for l in range(1,7)}
            best=min(ps,key=ps.get)
            print(f"  {direction:<40} min p={ps[best]:.4f} at lag {best}  {'** significant' if ps[best]<0.05 else ''}")
        except Exception as e: print("  err",e)

print("\n\n=== EARLY WARNING: is the SPREAD itself the earliest recession signal? ===")
nber=[('2001-03-31','2001-11-30'),('2007-12-31','2009-06-30'),('2020-02-29','2020-04-30')]
sp=df[cs]
for start,end in nber:
    st=pd.Timestamp(start)
    print(f"\n  Recession starting {st.date()}:")
    # months before the recession start when each signal first crossed its warning threshold
    for name,series,rule in [
        ('Credit_Spread  > 12m avg +1sd', sp, lambda s: s > s.rolling(12).mean()+s.rolling(12).std()),
        ('Unemployment   +0.5pp off low', df['Unemployment_Rate'], lambda s: (s-s.rolling(12).min())>=0.5),
        ('Sentiment      < 12m avg -1sd', df['Consumer_Sentiment'], lambda s: s < s.rolling(12).mean()-s.rolling(12).std()),
        ('IndProd YoY    < 0', df['Industrial_Production'], lambda s: (s/s.shift(12)-1)<0),
    ]:
        flag=rule(series)
        win=flag.loc[st-pd.DateOffset(months=24):st]
        first=win[win].index.min() if win.any() else None
        if first is not None:
            lead=(st.year-first.year)*12+(st.month-first.month)
            print(f"    {name:<32} first fired {first.date()}  -> {lead:>2} months of lead")
        else:
            print(f"    {name:<32} did not fire in the 24m before")
