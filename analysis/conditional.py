import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',240)
rng=np.random.default_rng(42)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'
dd=df.diff().dropna()

# Regime definition: STRESS = spread level at/above its trailing-expanding 80th pct... 
# simpler & fully defensible: stress = spread LEVEL in top quartile of full sample (>= p75 = 6.43)
# and a real-time variant. Report both.
lvl=df[cs]
thr=lvl.quantile(0.75)
stress=(lvl>=thr).reindex(dd.index)
print(f"Stress defined as Credit_Spread >= p75 = {thr:.2f}%.  n_stress={stress.sum()}  n_calm={(~stress).sum()}")

def boot_corr(x,y,n=5000):
    v=np.c_[x,y]; v=v[np.isfinite(v).all(1)]
    if len(v)<10: return np.nan,np.nan,np.nan
    base=np.corrcoef(v[:,0],v[:,1])[0,1]
    bs=[np.corrcoef(v[i][:,0],v[i][:,1])[0,1] for i in rng.integers(0,len(v),(n,len(v)))]
    return base, np.percentile(bs,2.5), np.percentile(bs,97.5)

print("\n=== Correlation of MACRO CHANGE with SPREAD CHANGE, by state (95% bootstrap CI) ===")
rows=[]
for c in macro:
    fa,fl,fh=boot_corr(dd[c].values,dd[cs].values)
    ca,cl,ch=boot_corr(dd.loc[~stress,c].values,dd.loc[~stress,cs].values)
    sa,sl,sh=boot_corr(dd.loc[stress,c].values,dd.loc[stress,cs].values)
    rows.append({'variable':c,'full':fa,'full_CI':f"[{fl:.2f},{fh:.2f}]",
                 'CALM':ca,'calm_CI':f"[{cl:.2f},{ch:.2f}]",
                 'STRESS':sa,'stress_CI':f"[{sl:.2f},{sh:.2f}]",
                 'gap':sa-ca,'ratio':abs(sa)/max(abs(ca),1e-9)})
r=pd.DataFrame(rows).set_index('variable')
print(r.round(3).to_string())
print(f"\n mean |corr| in CALM  = {r['CALM'].abs().mean():.3f}")
print(f" mean |corr| in STRESS = {r['STRESS'].abs().mean():.3f}   -> {r['STRESS'].abs().mean()/r['CALM'].abs().mean():.1f}x stronger")

# R2 of a joint macro regression, calm vs stress
import numpy.linalg as la
def r2(X,y):
    X=np.c_[np.ones(len(X)),X]; b=la.lstsq(X,y,rcond=None)[0]; e=y-X@b
    return 1-(e**2).sum()/((y-y.mean())**2).sum()
Xf=dd[macro].values; yf=dd[cs].values
print(f"\n=== Joint in-sample R2 of regressing spread CHANGE on all 6 macro CHANGES ===")
print(f"  full sample : R2 = {r2(Xf,yf):.4f}  (n={len(yf)})")
print(f"  CALM  months: R2 = {r2(dd.loc[~stress,macro].values, dd.loc[~stress,cs].values):.4f}  (n={(~stress).sum()})")
print(f"  STRESS months: R2 = {r2(dd.loc[stress,macro].values, dd.loc[stress,cs].values):.4f}  (n={stress.sum()})")

# Contribution: how much of the full-sample covariance comes from stress months?
print("\n=== Concentration: share of total spread VARIANCE contributed by stress months ===")
tot=((dd[cs]-dd[cs].mean())**2).sum()
st=((dd.loc[stress,cs]-dd[cs].mean())**2).sum()
print(f"  stress months are {100*stress.mean():.1f}% of months but carry {100*st/tot:.1f}% of squared spread-change variation")
