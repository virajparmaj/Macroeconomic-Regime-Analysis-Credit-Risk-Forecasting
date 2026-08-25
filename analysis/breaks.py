import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',250)
import statsmodels.api as sm
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
cs='Credit_Spread'
dd=df.diff().dropna()
y=dd[cs].values; Xm=sm.add_constant(dd[macro].values); idx=dd.index

print("=== SUP-WALD / Quandt-Andrews break test on the macro->Δspread regression ===")
n=len(y); k=Xm.shape[1]; trim=0.15
lo,hi=int(n*trim),int(n*(1-trim))
full=sm.OLS(y,Xm).fit(); ssr_r=full.ssr
stats=[]
for b in range(lo,hi):
    s1=sm.OLS(y[:b],Xm[:b]).fit(); s2=sm.OLS(y[b:],Xm[b:]).fit()
    ssr_u=s1.ssr+s2.ssr
    F=((ssr_r-ssr_u)/k)/(ssr_u/(n-2*k))
    stats.append((idx[b],F))
S=pd.DataFrame(stats,columns=['date','F']).set_index('date')
best=S['F'].idxmax()
print(f"  supF = {S['F'].max():.2f} at break date {best.date()}")
print(f"  Andrews 5% critical value for k={k} params, 15% trim is ~{ {2:11.0,3:12.4,4:13.8,5:15.0,6:16.2,7:17.5}.get(k,17.5):.1f}"
      f"  -> {'REJECT stability' if S['F'].max()> {2:11.0,3:12.4,4:13.8,5:15.0,6:16.2,7:17.5}.get(k,17.5) else 'cannot reject'}")
print("\n  Top 8 candidate break dates by F:")
print(S.sort_values('F',ascending=False).head(8).round(2).to_string())
S.to_csv('supwald.csv')

print("\n\n=== Coefficients before vs after the dominant break ===")
b=list(idx).index(best)
pre=sm.OLS(y[:b],Xm[:b]).fit(); post=sm.OLS(y[b:],Xm[b:]).fit()
names=['const']+macro
C=pd.DataFrame({'pre_coef':pre.params,'pre_p':pre.pvalues,'post_coef':post.params,'post_p':post.pvalues},index=names)
C['sign_flip']=(np.sign(C.pre_coef)!=np.sign(C.post_coef))
C['pre_n']=b; C['post_n']=n-b
print(C.round(4).to_string())
print(f"\n  pre-break  R2 = {pre.rsquared:.4f} (n={b}, {idx[0].date()} .. {idx[b-1].date()})")
print(f"  post-break R2 = {post.rsquared:.4f} (n={n-b}, {idx[b].date()} .. {idx[-1].date()})")

print("\n\n=== ROLLING 60m R2 of macro->Δspread: when does the relationship work? ===")
W=60; rr=[]
for i in range(W,len(dd)+1):
    sub=dd.iloc[i-W:i]
    m=sm.OLS(sub[cs].values, sm.add_constant(sub[macro].values)).fit()
    rr.append((dd.index[i-1],m.rsquared))
RR=pd.DataFrame(rr,columns=['date','r2']).set_index('date')
print(RR.describe().round(3).to_string())
print(f"\n  min R2 {RR['r2'].min():.3f} at {RR['r2'].idxmin().date()}   max R2 {RR['r2'].max():.3f} at {RR['r2'].idxmax().date()}")
print("\n  sampled every 12m:"); print(RR.iloc[::12].round(3).to_string())
RR.to_csv('rolling_r2.csv')
