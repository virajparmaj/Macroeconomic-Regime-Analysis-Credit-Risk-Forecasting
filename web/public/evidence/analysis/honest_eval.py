import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk')
pd.set_option('display.width',240)
from sklearn.ensemble import RandomForestRegressor
from src.metrics import r2_oos, diebold_mariano, regression_metrics
from src.splits import walk_forward_folds

ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'

# point-in-time: publication lags per src/config_v2.py
lags={'CPI':1,'FEDFUNDS':1,'Industrial_Production':1,'EA19_GDP_OECD':3,'Unemployment_Rate':1,'Consumer_Sentiment':0}
pit=df.copy()
for c,l in lags.items():
    if l: pit[c]=pit[c].shift(l)

def build(frame, use_spread_history):
    X=pd.DataFrame(index=frame.index)
    for c in macro:
        X[c]=frame[c]
        X[f'{c}_d1']=frame[c].diff()
        X[f'{c}_d12']=frame[c].diff(12)
    if use_spread_history:
        X['sp_lag0']=frame[cs]
        X['sp_d1']=frame[cs].diff()
        X['sp_d3']=frame[cs].diff(3)
        X['sp_rollmean3_lag1']=frame[cs].rolling(3).mean().shift(1)
        X['sp_rollstd6_lag1']=frame[cs].rolling(6).std().shift(1)
    y=frame[cs].shift(-1).rename('y')
    d=X.join(y).dropna()
    return d.drop(columns='y'), d['y'], frame[cs].reindex(d.index)

results=[]
for label, use_sp in [('macro_only',False),('macro+spread_history',True)]:
    Xa,ya,lvl=build(pit,use_sp)
    preds={'model':[],'rw':[],'y':[],'idx':[]}
    for fold in walk_forward_folds(Xa.index, initial_train_end='2008-12-31', refit_freq=12, embargo=3):
        tr=fold.train_index.intersection(Xa.index); te=fold.test_index.intersection(Xa.index)
        if len(tr)<24 or len(te)==0: continue
        m=RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=-1)
        m.fit(Xa.loc[tr],ya.loc[tr])
        preds['model']+=list(m.predict(Xa.loc[te])); preds['rw']+=list(lvl.loc[te])
        preds['y']+=list(ya.loc[te]); preds['idx']+=list(te)
    P=pd.DataFrame(preds).set_index('idx')
    mm=regression_metrics(P['y'],P['model']); rr=regression_metrics(P['y'],P['rw'])
    ro=r2_oos(P['y'],P['model'],P['rw']); dm=diebold_mariano(P['y'],P['model'],P['rw'])
    results.append({'setup':label,'n':len(P),'model_RMSE':mm['rmse'],'model_R2':mm['r2'],
                    'RW_RMSE':rr['rmse'],'RW_R2':rr['r2'],'R2_oos_vs_RW':ro,
                    'DM_stat':dm['dm_stat'],'DM_p':dm['p_value']})
    P.to_csv(f'preds_{label}.csv')
R=pd.DataFrame(results).set_index('setup')
print("=== HONEST WALK-FORWARD (expanding, 14 origins 2009-2022, embargo=3, point-in-time macro, target = spread(t+1)) ===")
print(R.round(4).to_string())
print("""
Reading:
  model_R2       = R2 against the test-period mean (the number v1 reported)
  R2_oos_vs_RW   = 1 - SSE(model)/SSE(random walk). Positive = beats "assume no change".
  DM p           = Diebold-Mariano vs random walk. p<0.05 = difference is significant.
""")
for _,row in R.reset_index().iterrows():
    print(f"  {row['setup']:<22} R2 looks like {row['model_R2']:.3f}, but vs the random walk it is {row['R2_oos_vs_RW']:+.3f} "
          f"({'WORSE' if row['R2_oos_vs_RW']<0 else 'better'}), DM p={row['DM_p']:.3f}")
