import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller, kpss
pd.set_option('display.width',240)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'
print("=== ADF (H0: unit root) / KPSS (H0: stationary) ===")
rows=[]
for c in df.columns:
    a=adfuller(df[c].dropna(),autolag='AIC'); k=kpss(df[c].dropna(),regression='c',nlags='auto')
    ad=adfuller(df[c].diff().dropna(),autolag='AIC')
    rows.append({'series':c,'ADF_p_level':a[1],'KPSS_p_level':k[1],'ADF_p_diff':ad[1],
                 'verdict':('I(1) - nonstationary in levels' if a[1]>0.05 else 'level-stationary')})
print(pd.DataFrame(rows).set_index('series').round(4).to_string())

print("\n\n=== ROLLING 60m correlation in FIRST DIFFERENCES (stationary, defensible) ===")
dd=df.diff().dropna()
W=60
rc=pd.DataFrame({c: dd[c].rolling(W).corr(dd[cs]) for c in macro}).dropna()
summ=pd.DataFrame({'full_sample_diff':dd.corr()[cs][macro],'roll_min':rc.min(),'roll_max':rc.max(),
                   'roll_mean':rc.mean(),'roll_std':rc.std()})
summ['range']=summ['roll_max']-summ['roll_min']
summ['flips_sign']=[(rc[c].min()<0)&(rc[c].max()>0) for c in macro]
print(summ.round(3).sort_values('range',ascending=False).to_string())
print("\n--- sampled every 12 months ---")
print(rc.iloc[::12].round(2).to_string())
rc.to_csv('rollcorr_diff.csv')

print("\n\n=== Same, but 36-month window (faster-moving) ===")
rc36=pd.DataFrame({c: dd[c].rolling(36).corr(dd[cs]) for c in macro}).dropna()
s36=pd.DataFrame({'roll_min':rc36.min(),'roll_max':rc36.max(),'roll_std':rc36.std()})
s36['range']=s36['roll_max']-s36['roll_min']
print(s36.round(3).sort_values('range',ascending=False).to_string())
rc36.to_csv('rollcorr_diff36.csv')
