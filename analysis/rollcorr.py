import pandas as pd, numpy as np
pd.set_option('display.width',240)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'

print("=== FULL-SAMPLE correlation with Credit_Spread (levels) ===")
print(df.corr()[cs].drop(cs).round(3).sort_values().to_string())
print("\n=== FULL-SAMPLE correlation with Credit_Spread (month-over-month CHANGES) ===")
dd=df.diff().dropna()
print(dd.corr()[cs].drop(cs).round(3).sort_values().to_string())

print("\n\n=== ROLLING 60-MONTH correlation with spread (levels) — stability check ===")
W=60
rc={c: df[c].rolling(W).corr(df[cs]) for c in macro}
rc=pd.DataFrame(rc).dropna()
summ=pd.DataFrame({'full_sample':df.corr()[cs][macro],
                   'roll_min':rc.min(),'roll_max':rc.max(),'roll_mean':rc.mean(),'roll_std':rc.std()})
summ['range']=summ['roll_max']-summ['roll_min']
summ['flips_sign']=[(rc[c].min()<0)&(rc[c].max()>0) for c in macro]
print(summ.round(3).sort_values('range',ascending=False).to_string())

print("\n=== Where do the sign flips happen? (60m rolling corr, sampled every 12m) ===")
print(rc.iloc[::12].round(2).to_string())

print("\n\n=== Split-sample: pre-GFC / GFC+recovery / post-2013 ===")
eras={'1996-12..2007-11 (pre-GFC)':('1996-12','2007-11'),
      '2007-12..2013-12 (GFC+recovery)':('2007-12','2013-12'),
      '2014-01..2022-08 (post-GFC)':('2014-01','2022-08')}
rows={}
for name,(a,b) in eras.items():
    sub=df.loc[a:b]
    rows[name]=sub.corr()[cs][macro]
    rows[name]['n_months']=len(sub); rows[name]['mean_spread']=sub[cs].mean(); rows[name]['sd_spread']=sub[cs].std()
print(pd.DataFrame(rows).round(3).to_string())
