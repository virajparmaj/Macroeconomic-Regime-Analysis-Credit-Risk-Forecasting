"""Export the completed analysis to lightweight JSON for the web front end.

Every number here comes from the same code paths as ANALYSIS_REPORT.md. No new
analysis is performed. Output goes to web/src/data/.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'web' / 'src' / 'data'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

MACRO = ['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
KEY   = {'CPI':'cpi','FEDFUNDS':'fedfunds','Industrial_Production':'indpro',
         'EA19_GDP_OECD':'eagdp','Unemployment_Rate':'unemp','Consumer_Sentiment':'sentiment'}
CS = 'Credit_Spread'
LAGS = {'CPI':1,'FEDFUNDS':1,'Industrial_Production':1,'EA19_GDP_OECD':3,'Unemployment_Rate':1,'Consumer_Sentiment':0}
NBER = [('2001-03-31','2001-11-30'),('2007-12-31','2009-06-30'),('2020-02-29','2020-04-30')]

def r(x, n=4):
    if x is None or (isinstance(x,float) and not np.isfinite(x)): return None
    return round(float(x), n)

df = pd.read_csv(ROOT/'data/merged_macroeconomic_credit.csv', parse_dates=['Month_End'],
                 index_col='Month_End').sort_index().rename(columns={'GDP':'EA19_GDP_OECD'})
dd = df.diff()

# ---------- macro-only stress index (identical to analysis/macro_regime.py) ----------
feat = pd.DataFrame(index=df.index)
feat['a'] = df['Unemployment_Rate'] - df['Unemployment_Rate'].rolling(12).min()
feat['b'] = -100*(df['Industrial_Production']/df['Industrial_Production'].shift(12)-1)
feat['c'] = -(df['Consumer_Sentiment']-df['Consumer_Sentiment'].rolling(60).mean())/df['Consumer_Sentiment'].rolling(60).std()
feat['d'] = -df['EA19_GDP_OECD']
feat = feat.dropna()
z = (feat-feat.mean())/feat.std()
stress_idx = z.mean(axis=1)
thr = stress_idx.quantile(0.75)
is_stress = stress_idx >= thr

# ---------- rolling 60m R2 of macro -> spread change (identical to analysis/breaks.py) ----------
import statsmodels.api as sm
ddn = dd.dropna()
roll = {}
W = 60
for i in range(W, len(ddn)+1):
    sub = ddn.iloc[i-W:i]
    m = sm.OLS(sub[CS].values, sm.add_constant(sub[MACRO].values)).fit()
    roll[ddn.index[i-1]] = m.rsquared
roll = pd.Series(roll)

# ---------- rolling 60m correlations in differences ----------
rollcorr = pd.DataFrame({c: ddn[c].rolling(60).corr(ddn[CS]) for c in MACRO}).dropna()

# ---------- 1. TIMELINE ----------
nber_flag = pd.Series(False, index=df.index)
for a,b in NBER: nber_flag.loc[a:b] = True
timeline = []
for ts, row in df.iterrows():
    rec = {'d': ts.strftime('%Y-%m'), 'spread': r(row[CS],3), 'nber': bool(nber_flag.loc[ts])}
    for c in MACRO: rec[KEY[c]] = r(row[c],3)
    rec['stress']  = r(stress_idx.get(ts), 3)
    rec['regime']  = (None if ts not in stress_idx.index else ('stress' if bool(is_stress.loc[ts]) else 'calm'))
    rec['r2']      = r(roll.get(ts), 4)
    for c in MACRO:
        rec['rc_'+KEY[c]] = r(rollcorr[c].get(ts), 3) if ts in rollcorr.index else None
    timeline.append(rec)

# ---------- 2. CALM vs STRESS (macro-only regime, with bootstrap CIs) ----------
ix = ddn.index.intersection(stress_idx.index)
d2, st = ddn.loc[ix], is_stress.loc[ix]
def boot(x, y, n=4000):
    v = np.c_[x,y]; v = v[np.isfinite(v).all(1)]
    if len(v) < 10: return None, None, None
    base = np.corrcoef(v[:,0], v[:,1])[0,1]
    bs = [np.corrcoef(v[i][:,0], v[i][:,1])[0,1] for i in rng.integers(0,len(v),(n,len(v)))]
    return base, np.percentile(bs,2.5), np.percentile(bs,97.5)
import numpy.linalg as la
def jr2(X, y):
    X = np.c_[np.ones(len(X)), X]; b = la.lstsq(X,y,rcond=None)[0]; e = y-X@b
    return 1 - (e**2).sum()/((y-y.mean())**2).sum()
cs_rows = []
for c in MACRO:
    fa,_,_ = boot(d2[c].values, d2[CS].values)
    ca,cl,ch = boot(d2.loc[~st,c].values, d2.loc[~st,CS].values)
    sa,sl,sh = boot(d2.loc[st,c].values, d2.loc[st,CS].values)
    cs_rows.append({'key':KEY[c],'full':r(fa,3),'calm':r(ca,3),'calmLo':r(cl,3),'calmHi':r(ch,3),
                    'stress':r(sa,3),'stressLo':r(sl,3),'stressHi':r(sh,3)})
# spread-level regime as the alternate (circular) definition, for the "how did you segment" toggle
lvl_thr = df[CS].quantile(0.75)
st_lvl = (df[CS] >= lvl_thr).reindex(ix)
cs_rows_lvl = []
for c in MACRO:
    ca,_,_ = boot(d2.loc[~st_lvl,c].values, d2.loc[~st_lvl,CS].values)
    sa,_,_ = boot(d2.loc[st_lvl,c].values, d2.loc[st_lvl,CS].values)
    cs_rows_lvl.append({'key':KEY[c],'calm':r(ca,3),'stress':r(sa,3)})
regime_split = {
  'macroOnly': {'rows':cs_rows,'nCalm':int((~st).sum()),'nStress':int(st.sum()),
      'r2Calm':r(jr2(d2.loc[~st,MACRO].values,d2.loc[~st,CS].values),4),
      'r2Stress':r(jr2(d2.loc[st,MACRO].values,d2.loc[st,CS].values),4),
      'r2Full':r(jr2(d2[MACRO].values,d2[CS].values),4)},
  'spreadLevel': {'rows':cs_rows_lvl,'nCalm':int((~st_lvl).sum()),'nStress':int(st_lvl.sum()),
      'r2Calm':r(jr2(d2.loc[~st_lvl,MACRO].values,d2.loc[~st_lvl,CS].values),4),
      'r2Stress':r(jr2(d2.loc[st_lvl,MACRO].values,d2.loc[st_lvl,CS].values),4),
      'r2Full':r(jr2(d2[MACRO].values,d2[CS].values),4)},
}
# Concentration is computed on the FULL differenced sample so it carries no window caveat.
hi_full = (df[CS] >= lvl_thr).reindex(ddn.index)
tot = ((ddn[CS]-ddn[CS].mean())**2).sum()
regime_split['varianceShare'] = r(((ddn.loc[hi_full,CS]-ddn[CS].mean())**2).sum()/tot, 4)
regime_split['monthShare'] = r(hi_full.mean(), 4)
regime_split['nHighSpread'] = int(hi_full.sum()); regime_split['nAll'] = int(len(ddn))

# ---------- 3. CROSS-CORRELATION ----------
ks = list(range(-6,7))
xcorr = {'lags':ks,'series':[],'mean':[]}
M = []
for c in MACRO:
    v = [dd[c].shift(k).corr(dd[CS]) for k in ks]
    M.append([abs(x) for x in v])
    xcorr['series'].append({'key':KEY[c],'values':[r(x,3) for x in v]})
xcorr['mean'] = [r(x,3) for x in np.array(M).mean(0)]
xcorr['spreadFirstSide'] = r(np.array(M).mean(0)[[i for i,k in enumerate(ks) if k<0]].mean(),3)
xcorr['macroFirstSide']  = r(np.array(M).mean(0)[[i for i,k in enumerate(ks) if k>0]].mean(),3)

warnings_tbl = [
 {'event':'2001 recession','start':'2001-03','signals':{'spread':12,'sentiment':5,'indpro':1,'unemp':0}},
 {'event':'2008 recession','start':'2007-12','signals':{'spread':5,'sentiment':19,'indpro':None,'unemp':0}},
 {'event':'2020 recession','start':'2020-02','signals':{'spread':15,'sentiment':13,'indpro':10,'unemp':None}},
]
granger = [
 {'key':'unemp','label':'Unemployment','spreadToMacro':0.0000,'macroToSpread':0.1078},
 {'key':'sentiment','label':'Consumer sentiment','spreadToMacro':0.0009,'macroToSpread':0.2052},
 {'key':'indpro','label':'Industrial production','spreadToMacro':0.0000,'macroToSpread':0.0410},
 {'key':'cpi','label':'CPI','spreadToMacro':0.0000,'macroToSpread':0.0043},
 {'key':'fedfunds','label':'Fed Funds','spreadToMacro':0.0008,'macroToSpread':0.0170},
 {'key':'eagdp','label':'OECD GDP (EA19)','spreadToMacro':0.1253,'macroToSpread':0.5810},
]

# ---------- 4. WALK-FORWARD PREDICTIONS ----------
SCR = Path('/private/tmp/claude-501/-Users-veerr-89-Work-projects-Macroeconomic-Regime-Analysis-Credit-Risk/f7cdb370-a410-4735-9379-5dec2c2748d2/scratchpad/an')
pa = pd.read_csv(SCR/'preds_macro+spread_history.csv', parse_dates=['idx'], index_col='idx')
pb = pd.read_csv(SCR/'preds_macro_only.csv', parse_dates=['idx'], index_col='idx')
preds = [{'d':ts.strftime('%Y-%m'),'actual':r(row['y'],3),'rw':r(row['rw'],3),
          'macroSpread':r(row['model'],3),'macroOnly':r(pb['model'].get(ts),3)} for ts,row in pa.iterrows()]
model_metrics = [
 {'model':'Random walk','sub':'assume no change','rmse':0.6124,'r2Mean':0.9244,'r2Rw':0.0,'dmP':None},
 {'model':'Random forest','sub':'macro + spread history','rmse':1.1131,'r2Mean':0.7503,'r2Rw':-2.3037,'dmP':0.0561},
 {'model':'Random forest','sub':'macro only','rmse':2.4004,'r2Mean':-0.1610,'r2Rw':-14.3629,'dmP':0.0000},
]

# ---------- 5. HORIZON ----------
horizon = {'h':[1,3,6,12],
 'auc':{'spreadOnly':[0.935,0.793,0.656,0.543],'macroSpread':[0.920,0.782,0.613,0.519],'macroOnly':[0.751,0.686,0.639,0.431]},
 'lift':{'spreadOnly':[4.38,2.55,1.52,1.28],'macroSpread':[3.52,2.27,1.70,1.37],'macroOnly':[1.87,1.84,1.74,1.42]},
 'threshold':6.43}

# ---------- 6. DAILY vs MONTHLY ----------
d_raw = pd.read_csv(ROOT/'data/original/ICE BofA US High Yield Index Option-Adjusted Spread_BAMLH0A0HYM2.csv',
                    parse_dates=['observation_date'], index_col='observation_date').sort_index()
s = d_raw.iloc[:,0].dropna()
mo = s.resample('ME').agg(['mean','max','min','last']); mo['rng'] = mo['max']-mo['min']; mo = mo.dropna(subset=['mean'])
def window(a,b):
    w = s.loc[a:b]
    bars = [{'m':ts.strftime('%Y-%m'),'mean':r(row['mean'],3),'max':r(row['max'],3),'min':r(row['min'],3)}
            for ts,row in mo.loc[a:b].iterrows()]
    return {'daily':[{'d':ts.strftime('%Y-%m-%d'),'v':r(v,3)} for ts,v in w.items()],'monthly':bars}
aggregation = {
 'windows':{
   'covid':{'label':'COVID · Dec 2019 – Aug 2020','from':'2019-12-01','to':'2020-08-31', **window('2019-12-01','2020-08-31')},
   'gfc':{'label':'Global financial crisis · Jul 2008 – Jun 2009','from':'2008-07-01','to':'2009-06-30', **window('2008-07-01','2009-06-30')},
   'telecom':{'label':'Telecom bust · Jun 2002 – Feb 2003','from':'2002-06-01','to':'2003-02-28', **window('2002-06-01','2003-02-28')},
 },
 'topRanges':[{'m':ts.strftime('%Y-%m'),'rng':r(row['rng'],2),'mean':r(row['mean'],2),
               'era':('COVID' if ts.year==2020 else ('GFC' if ts.year in (2008,2009) else str(ts.year)))}
              for ts,row in mo.nlargest(8,'rng').iterrows()],
 'medianRange': r(mo['rng'].median(),3),
 'rankByRange': 1, 'rankByMean': 43, 'nMonths': int(len(mo)),
}

# ---------- 7. KEY NUMBERS ----------
kn = pd.read_csv(ROOT/'results/key_numbers.csv')
key_numbers = {row['metric']: {'value':row['value'],'desc':row['description'],'insight':int(row['insight'])}
               for _,row in kn.iterrows()}

for name, payload in [('timeline',timeline),('regimeSplit',regime_split),('xcorr',xcorr),
                      ('warnings',warnings_tbl),('granger',granger),('preds',preds),
                      ('modelMetrics',model_metrics),('horizon',horizon),
                      ('aggregation',aggregation),('keyNumbers',key_numbers)]:
    p = OUT/f'{name}.json'
    p.write_text(json.dumps(payload, separators=(',',':')))
    print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1024:.1f} KB")
print("\nexport complete")
