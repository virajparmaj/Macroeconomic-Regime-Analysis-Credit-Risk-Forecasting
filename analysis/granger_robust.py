import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',250)
from statsmodels.tsa.stattools import grangercausalitytests
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'

def gp(frame,cause,effect,maxlag=6):
    """p-value that `cause` Granger-causes `effect`. statsmodels tests col2 -> col1."""
    d=pd.concat([frame[effect],frame[cause]],axis=1).dropna()
    if len(d)<40: return np.nan,None
    r=grangercausalitytests(d,maxlag=maxlag,verbose=False)
    ps={l:r[l][0]['ssr_ftest'][1] for l in range(1,maxlag+1)}
    b=min(ps,key=ps.get); return ps[b],b

print("=== Granger p-values, BOTH directions, on first differences ===")
print(f"{'macro variable':<24} {'spread -> macro':>16} {'macro -> spread':>16}   verdict")
dd=df.diff().dropna()
for c in macro:
    p1,l1=gp(dd,cs,c); p2,l2=gp(dd,c,cs)
    v='SPREAD LEADS' if (p1<0.05 and p2>=0.05) else ('macro leads' if (p2<0.05 and p1>=0.05) else ('both' if p1<0.05 and p2<0.05 else 'neither'))
    print(f"{c:<24} {p1:>16.4f} {p2:>16.4f}   {v}")

print("\n\n=== ROBUSTNESS 1: exclude crisis periods (drop 2008-09 and 2020) ===")
mask=~(dd.index.year.isin([2008,2009,2020]))
sub=dd[mask]
print(f"n = {len(sub)} (dropped {len(dd)-len(sub)} crisis months)")
print(f"{'macro variable':<24} {'spread -> macro':>16} {'macro -> spread':>16}   verdict")
for c in macro:
    p1,_=gp(sub,cs,c); p2,_=gp(sub,c,cs)
    v='SPREAD LEADS' if (p1<0.05 and p2>=0.05) else ('macro leads' if (p2<0.05 and p1>=0.05) else ('both' if p1<0.05 and p2<0.05 else 'neither'))
    print(f"{c:<24} {p1:>16.4f} {p2:>16.4f}   {v}")

print("\n\n=== ROBUSTNESS 2: real-time / point-in-time alignment (macro shifted by publication lag) ===")
print("   This is the fair test: a forecaster at month t has NOT yet seen month t's CPI or unemployment.")
lags={'CPI':1,'FEDFUNDS':1,'Industrial_Production':1,'EA19_GDP_OECD':3,'Unemployment_Rate':1,'Consumer_Sentiment':0}
pit=df.copy()
for c,l in lags.items():
    if l: pit[c]=pit[c].shift(l)
dpit=pit.diff().dropna()
print(f"{'macro variable':<24} {'spread -> macro':>16} {'macro -> spread':>16}   verdict")
for c in macro:
    p1,_=gp(dpit,cs,c); p2,_=gp(dpit,c,cs)
    v='SPREAD LEADS' if (p1<0.05 and p2>=0.05) else ('macro leads' if (p2<0.05 and p1>=0.05) else ('both' if p1<0.05 and p2<0.05 else 'neither'))
    print(f"{c:<24} {p1:>16.4f} {p2:>16.4f}   {v}")

print("\n\n=== ROBUSTNESS 3: two halves of the sample ===")
for lbl,a,b in [('1997-2009',None,'2009-12-31'),('2010-2022','2010-01-01',None)]:
    s=dd.loc[a:b]
    n_lead=sum(1 for c in macro if (gp(s,cs,c)[0]<0.05) and (gp(s,c,cs)[0]>=0.05))
    n_rev=sum(1 for c in macro if (gp(s,c,cs)[0]<0.05) and (gp(s,cs,c)[0]>=0.05))
    print(f"  {lbl} (n={len(s)}): spread leads {n_lead}/6 macro vars ; macro leads spread for {n_rev}/6")
