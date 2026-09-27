import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',240)
rng=np.random.default_rng(42)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'

# ---- MACRO-ONLY stress index: no spread information at all ----
# Use 12m change in unemployment (Sahm-like), 12m growth in IP, sentiment level z, EA GDP level
feat=pd.DataFrame(index=df.index)
feat['unemp_chg12']=df['Unemployment_Rate']-df['Unemployment_Rate'].rolling(12).min()   # Sahm-style rise off the low
feat['ip_yoy']=-100*(df['Industrial_Production']/df['Industrial_Production'].shift(12)-1)  # sign flipped: high = bad
feat['sent_z']=-(df['Consumer_Sentiment']-df['Consumer_Sentiment'].rolling(60).mean())/df['Consumer_Sentiment'].rolling(60).std()
feat['gdp_neg']=-df['EA19_GDP_OECD']
feat=feat.dropna()
z=(feat-feat.mean())/feat.std()
macro_stress=z.mean(axis=1).rename('macro_stress')
print("Macro-only stress index built from: Sahm-style unemployment rise, negative IP YoY, sentiment z-score, negative EA GDP.")
print("NO credit-spread information used.\n")
mthr=macro_stress.quantile(0.75)
mstress=(macro_stress>=mthr)
print(f"threshold p75 = {mthr:.3f} | n_stress={mstress.sum()} n_calm={(~mstress).sum()}")
print("\nStress months by year:")
print(mstress[mstress].groupby(mstress[mstress].index.year).size().to_string())

# NBER external validation
nber=[('2001-03-31','2001-11-30'),('2007-12-31','2009-06-30'),('2020-02-29','2020-04-30')]
inr=pd.Series(False,index=macro_stress.index)
for a,b in nber: inr.loc[a:b]=True
tp=(mstress&inr).sum(); print(f"\nOverlap with NBER recessions: {tp}/{inr.sum()} recession months flagged ({100*tp/inr.sum():.0f}% recall)")

dd=df.diff().dropna()
common=dd.index.intersection(macro_stress.index)
dd=dd.loc[common]; ms=mstress.loc[common]
def boot_corr(x,y,n=4000):
    v=np.c_[x,y]; v=v[np.isfinite(v).all(1)]
    if len(v)<10: return np.nan,np.nan,np.nan
    base=np.corrcoef(v[:,0],v[:,1])[0,1]
    bs=[np.corrcoef(v[i][:,0],v[i][:,1])[0,1] for i in rng.integers(0,len(v),(n,len(v)))]
    return base,np.percentile(bs,2.5),np.percentile(bs,97.5)
print(f"\n=== Spread-change correlation by MACRO-ONLY regime (n_calm={(~ms).sum()}, n_stress={ms.sum()}) ===")
rows=[]
for c in macro:
    fa,_,_=boot_corr(dd[c].values,dd[cs].values)
    ca,cl,ch=boot_corr(dd.loc[~ms,c].values,dd.loc[~ms,cs].values)
    sa,sl,sh=boot_corr(dd.loc[ms,c].values,dd.loc[ms,cs].values)
    rows.append({'variable':c,'full':fa,'CALM':ca,'calm_CI':f"[{cl:.2f},{ch:.2f}]",
                 'STRESS':sa,'stress_CI':f"[{sl:.2f},{sh:.2f}]",'gap':sa-ca})
r=pd.DataFrame(rows).set_index('variable'); print(r.round(3).to_string())
print(f"\n mean |corr| CALM={r['CALM'].abs().mean():.3f}  STRESS={r['STRESS'].abs().mean():.3f}  ratio={r['STRESS'].abs().mean()/r['CALM'].abs().mean():.1f}x")
import numpy.linalg as la
def r2(X,y):
    X=np.c_[np.ones(len(X)),X]; b=la.lstsq(X,y,rcond=None)[0]; e=y-X@b
    return 1-(e**2).sum()/((y-y.mean())**2).sum()
print(f"\n Joint R2 (spread change ~ 6 macro changes):")
print(f"   full   = {r2(dd[macro].values,dd[cs].values):.4f}")
print(f"   CALM   = {r2(dd.loc[~ms,macro].values,dd.loc[~ms,cs].values):.4f}  (n={(~ms).sum()})")
print(f"   STRESS = {r2(dd.loc[ms,macro].values,dd.loc[ms,cs].values):.4f}  (n={ms.sum()})")
print("\n=== Spread LEVEL behaviour by macro-only regime ===")
lv=df[cs].loc[common]
g=pd.DataFrame({'spread':lv,'stress':ms}).groupby('stress')['spread']
print(g.agg(['count','mean','median','std',lambda x:x.quantile(.95),'max']).rename(columns={'<lambda_0>':'p95'}).round(2).to_string())
macro_stress.to_frame().join(df[cs]).to_csv('macro_stress.csv')
