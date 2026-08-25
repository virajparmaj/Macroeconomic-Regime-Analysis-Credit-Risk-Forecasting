import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk')
pd.set_option('display.width',250)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from src.metrics import r2_oos, diebold_mariano, regression_metrics
from src.splits import walk_forward_folds
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'
lags={'CPI':1,'FEDFUNDS':1,'Industrial_Production':1,'EA19_GDP_OECD':3,'Unemployment_Rate':1,'Consumer_Sentiment':0}
pit=df.copy()
for c,l in lags.items():
    if l: pit[c]=pit[c].shift(l)

X=pd.DataFrame(index=pit.index)
for c in macro:
    X[c]=pit[c]; X[f'{c}_d1']=pit[c].diff(); X[f'{c}_d3']=pit[c].diff(3); X[f'{c}_d12']=pit[c].diff(12)
X['sp_lag0']=pit[cs]; X['sp_d1']=pit[cs].diff(); X['sp_d3']=pit[cs].diff(3)
X['sp_rollstd6_lag1']=pit[cs].rolling(6).std().shift(1)
# TARGET IS THE CHANGE -> random walk == predicting 0
dy=(pit[cs].shift(-1)-pit[cs]).rename('dy')
D=X.join(dy).dropna(); Xa=D.drop(columns='dy'); ya=D['dy']; lvl=pit[cs].reindex(D.index)
macro_cols=[c for c in Xa.columns if not c.startswith('sp_')]

rows=[];store={}
for label,cols,mk in [
    ('RF  macro-only-Δ',macro_cols,lambda:RandomForestRegressor(400,min_samples_leaf=3,random_state=42,n_jobs=-1)),
    ('RF  macro+spread-Δ',list(Xa.columns),lambda:RandomForestRegressor(400,min_samples_leaf=3,random_state=42,n_jobs=-1)),
    ('Ridge macro+spread-Δ',list(Xa.columns),lambda:RidgeCV(alphas=np.logspace(-3,3,25))),
]:
    P={'m':[],'y':[],'i':[],'lvl':[]}
    for fold in walk_forward_folds(Xa.index,initial_train_end='2008-12-31',refit_freq=12,embargo=3):
        tr=fold.train_index.intersection(Xa.index); te=fold.test_index.intersection(Xa.index)
        if len(tr)<24 or len(te)==0: continue
        mo=mk(); mo.fit(Xa.loc[tr,cols],ya.loc[tr])
        P['m']+=list(mo.predict(Xa.loc[te,cols])); P['y']+=list(ya.loc[te]); P['i']+=list(te); P['lvl']+=list(lvl.loc[te])
    p=pd.DataFrame(P).set_index('i'); store[label]=p
    zero=np.zeros(len(p))   # random walk = predict zero change
    mm=regression_metrics(p['y'],p['m']); rr=regression_metrics(p['y'],zero)
    ro=r2_oos(p['y'],p['m'],zero); dm=diebold_mariano(p['y'],p['m'],zero)
    rows.append({'model':label,'n':len(p),'RMSE_Δ':mm['rmse'],'RW_RMSE_Δ':rr['rmse'],
                 'R2_oos_vs_RW':ro,'DM_stat':dm['dm_stat'],'DM_p':dm['p_value']})
R=pd.DataFrame(rows).set_index('model')
print("=== FAIREST TEST: predict the CHANGE in spread. Random walk = predict 0. Model nests it. ===")
print(R.round(4).to_string())
print("\n  R2_oos_vs_RW > 0 means macro adds information beyond just knowing today's spread.\n")

# ---- regime-conditional performance ----
print("\n=== Performance by regime, best model ===")
best='RF  macro+spread-Δ'
p=store[best].copy()
p['lvl']=lvl.reindex(p.index)
thr=df[cs].quantile(0.75)
p['regime']=np.where(p['lvl']>=thr,'STRESS (spread>=6.43)','CALM')
for g,b in p.groupby('regime'):
    z=np.zeros(len(b))
    print(f"  {g:<24} n={len(b):>3}  model RMSE={np.sqrt(((b['y']-b['m'])**2).mean()):.4f}  "
          f"RW RMSE={np.sqrt((b['y']**2).mean()):.4f}  R2_oos={r2_oos(b['y'],b['m'],z):+.4f}")
print("\n=== How big are the actual monthly changes the model must catch? ===")
print(f"  sd of Δspread in CALM  : {p.loc[p.regime=='CALM','y'].std():.3f}")
print(f"  sd of Δspread in STRESS: {p.loc[p.regime!='CALM','y'].std():.3f}")
for lab,q in [('worst 1-month widening',p['y'].max()),('worst 1-month tightening',p['y'].min())]:
    print(f"  {lab}: {q:+.2f} pp")
