import pandas as pd, numpy as np
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50)
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df = pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv', parse_dates=['Month_End'], index_col='Month_End').sort_index()
print("SHAPE", df.shape, "|", df.index.min().date(), "->", df.index.max().date())
print("\n--- dtypes / missing / dupes ---")
print(pd.DataFrame({'dtype':df.dtypes,'n_missing':df.isna().sum(),'n_unique':df.nunique()}))
print("dup index:", df.index.duplicated().sum(), "| dup rows:", df.duplicated().sum())
# monthly grid gaps
full = pd.date_range(df.index.min(), df.index.max(), freq='ME')
print("expected months:", len(full), "actual:", len(df), "missing months:", sorted(set(full)-set(df.index))[:10])
print("\n--- describe ---")
print(df.describe().T[['count','mean','std','min','25%','50%','75%','max']].round(3))
print("\n--- percentiles of Credit_Spread ---")
cs = df['Credit_Spread']
for q in [0.01,0.05,0.10,0.25,0.5,0.75,0.90,0.95,0.99]:
    print(f"  p{q*100:>5.1f} = {cs.quantile(q):.3f}")
print(f"  skew={cs.skew():.3f} kurt={cs.kurtosis():.3f}  mean={cs.mean():.3f} median={cs.median():.3f}")
print("\n--- repeated/flat values (possible ffill) ---")
for c in df.columns:
    s=df[c]; flat=(s.diff()==0).sum()
    print(f"  {c:<24} exact-repeat months={flat:>3}  max run={(s.groupby((s!=s.shift()).cumsum()).transform('size')).max()}")
print("\n--- top 12 spread months ---")
print(cs.sort_values(ascending=False).head(12).round(2).to_string())
print("\n--- autocorrelation of Credit_Spread ---")
for k in [1,2,3,6,12]:
    print(f"  lag{k}: {cs.autocorr(k):.4f}")
d = cs.diff()
print("  diff lag1:", round(d.autocorr(1),4))
