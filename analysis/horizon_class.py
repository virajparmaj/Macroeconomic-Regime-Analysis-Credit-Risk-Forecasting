import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk')
pd.set_option('display.width',250)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.metrics import r2_oos, diebold_mariano
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
Xs=X.copy()
Xs['sp_lag0']=pit[cs]; Xs['sp_d3']=pit[cs].diff(3); Xs['sp_rollstd6']=pit[cs].rolling(6).std().shift(1)

print("=== A) REGRESSION at longer horizons: does macro beat the random walk anywhere? ===")
print(f"{'h':>3} {'set':<16} {'model RMSE':>11} {'RW RMSE':>9} {'R2_oos vs RW':>13} {'DM p':>7}")
for h in [1,3,6,12]:
    y=(pit[cs].shift(-h)-pit[cs]).rename('y')
    for nm,XX in [('macro-only',X),('macro+spread',Xs)]:
        D=XX.join(y).dropna(); A=D.drop(columns='y'); b=D['y']
        P={'m':[],'y':[],'i':[]}
        for f in walk_forward_folds(A.index,initial_train_end='2008-12-31',refit_freq=12,embargo=max(12,h)):
            tr=f.train_index.intersection(A.index); te=f.test_index.intersection(A.index)
            if len(tr)<24 or len(te)==0: continue
            mo=RandomForestRegressor(200,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(A.loc[tr],b.loc[tr])
            P['m']+=list(mo.predict(A.loc[te])); P['y']+=list(b.loc[te]); P['i']+=list(te)
        p=pd.DataFrame(P); z=np.zeros(len(p))
        rm=np.sqrt(((p['y']-p['m'])**2).mean()); rw=np.sqrt((p['y']**2).mean())
        dm=diebold_mariano(p['y'],p['m'],z,horizon=h)
        print(f"{h:>3} {nm:<16} {rm:>11.4f} {rw:>9.4f} {r2_oos(p['y'],p['m'],z):>13.4f} {dm['p_value']:>7.4f}")

print("\n\n=== B) CLASSIFICATION: will the spread be in the top quartile (>=6.43) h months from now? ===")
thr=df[cs].quantile(0.75)
print(f"    threshold = p75 = {thr:.2f}pp | base rate = {100*(df[cs]>=thr).mean():.1f}%")
print(f"{'h':>3} {'feature set':<16} {'n':>4} {'pos':>4} {'AUC':>7} {'AvgPrec':>8} {'base':>6} {'lift':>6}")
res={}
for h in [1,3,6,12]:
    ylab=(pit[cs].shift(-h)>=thr).astype(int).rename('y')
    for nm,XX in [('macro-only',X),('macro+spread',Xs),('spread-only',Xs[['sp_lag0','sp_d3','sp_rollstd6']])]:
        D=XX.join(ylab).dropna(); A=D.drop(columns='y'); b=D['y']
        P={'p':[],'y':[],'i':[]}
        for f in walk_forward_folds(A.index,initial_train_end='2008-12-31',refit_freq=12,embargo=max(12,h)):
            tr=f.train_index.intersection(A.index); te=f.test_index.intersection(A.index)
            if len(tr)<24 or len(te)==0 or b.loc[tr].nunique()<2: continue
            mo=RandomForestClassifier(200,min_samples_leaf=3,random_state=42,n_jobs=-1,class_weight='balanced').fit(A.loc[tr],b.loc[tr])
            P['p']+=list(mo.predict_proba(A.loc[te])[:,1]); P['y']+=list(b.loc[te]); P['i']+=list(te)
        p=pd.DataFrame(P)
        if p['y'].nunique()<2: print(f"{h:>3} {nm:<16} no positives"); continue
        auc=roc_auc_score(p['y'],p['p']); ap=average_precision_score(p['y'],p['p']); base=p['y'].mean()
        print(f"{h:>3} {nm:<16} {len(p):>4} {int(p['y'].sum()):>4} {auc:>7.3f} {ap:>8.3f} {base:>6.3f} {ap/base:>6.2f}x")
        res[(h,nm)]=p
        p.to_csv(f'clf_h{h}_{nm}.csv',index=False)
