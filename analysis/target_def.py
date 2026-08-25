import pandas as pd, numpy as np
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
d=pd.read_csv(f'{ROOT}/data/original/ICE BofA US High Yield Index Option-Adjusted Spread_BAMLH0A0HYM2.csv',
              parse_dates=['observation_date'],index_col='observation_date').sort_index()
s=d.iloc[:,0].dropna()
m=s.resample('ME').agg(['mean','last','max']); m.index=m.index.to_period('M')
print("=== Apples-to-apples: how hard is each target definition? (random-walk h=1) ===")
print(f"{'target':<10} {'AR(1)':>7} {'RW RMSE':>9} {'RW MAE':>8} {'sd(target)':>11} {'RW R2 vs test-mean':>19}")
for c in ['mean','last','max']:
    y=m[c].dropna(); yn=y.shift(-1); pair=pd.DataFrame({'y':yn,'x':y}).dropna()
    e=pair['y']-pair['x']; rmse=np.sqrt((e**2).mean())
    r2=1-(e**2).sum()/((pair['y']-pair['y'].mean())**2).sum()
    print(f"{c:<10} {y.autocorr(1):7.4f} {rmse:9.4f} {e.abs().mean():8.4f} {y.std():11.4f} {r2:19.4f}")
print("\nInterpretation: a higher AR(1) means the *target itself* is easier, independent of any model.")
print(f"  monthly-MEAN AR(1) = {m['mean'].autocorr(1):.4f}")
print(f"  monthly-LAST AR(1) = {m['last'].autocorr(1):.4f}")
print(f"  variance of month-over-month change: mean-target={m['mean'].diff().var():.4f}  last-target={m['last'].diff().var():.4f}")
print(f"  -> the mean target's monthly change has {100*(1-m['mean'].diff().var()/m['last'].diff().var()):.1f}% LESS variance to explain")

# how much of monthly-mean(t+1) is mechanically already known at end of month t?
# mean(t+1) = average of daily spreads in t+1. Decompose: R2 of regressing mean(t+1) on last(t)
import numpy.linalg as la
pair=pd.DataFrame({'y':m['mean'].shift(-1),'x':m['last']}).dropna()
X=np.c_[np.ones(len(pair)),pair['x']]; b=la.lstsq(X,pair['y'],rcond=None)[0]
res=pair['y']-X@b; r2=1-(res**2).sum()/((pair['y']-pair['y'].mean())**2).sum()
print(f"\n  R2 of mean(t+1) explained by the single number last(t): {r2:.4f}")
pair2=pd.DataFrame({'y':m['last'].shift(-1),'x':m['last']}).dropna()
X2=np.c_[np.ones(len(pair2)),pair2['x']]; b2=la.lstsq(X2,pair2['y'],rcond=None)[0]
res2=pair2['y']-X2@b2; r22=1-(res2**2).sum()/((pair2['y']-pair2['y'].mean())**2).sum()
print(f"  R2 of last(t+1) explained by the single number last(t): {r22:.4f}")
print(f"  -> switching the target from month-mean to month-end drops trivially-achievable R2 by {100*(r2-r22):.1f} points")
