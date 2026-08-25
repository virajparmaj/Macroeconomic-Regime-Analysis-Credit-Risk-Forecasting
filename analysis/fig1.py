import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0,'.'); from style import *
setup()
FIG='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/results/figures'
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['FEDFUNDS','CPI','EA19_GDP_OECD','Consumer_Sentiment','Industrial_Production','Unemployment_Rate']
NICE={'FEDFUNDS':'Fed Funds','CPI':'CPI','EA19_GDP_OECD':'OECD GDP (EA19)','Consumer_Sentiment':'Consumer sentiment',
      'Industrial_Production':'Industrial production','Unemployment_Rate':'Unemployment'}
cs='Credit_Spread'
# macro-only stress index (no spread info)
feat=pd.DataFrame(index=df.index)
feat['a']=df['Unemployment_Rate']-df['Unemployment_Rate'].rolling(12).min()
feat['b']=-100*(df['Industrial_Production']/df['Industrial_Production'].shift(12)-1)
feat['c']=-(df['Consumer_Sentiment']-df['Consumer_Sentiment'].rolling(60).mean())/df['Consumer_Sentiment'].rolling(60).std()
feat['d']=-df['EA19_GDP_OECD']
feat=feat.dropna(); z=(feat-feat.mean())/feat.std(); ms=z.mean(1)
stress=(ms>=ms.quantile(0.75))
dd=df.diff().dropna(); ix=dd.index.intersection(stress.index); dd=dd.loc[ix]; st=stress.loc[ix]
import numpy.linalg as la
def r2(X,y):
    X=np.c_[np.ones(len(X)),X]; b=la.lstsq(X,y,rcond=None)[0]; e=y-X@b
    return 1-(e**2).sum()/((y-y.mean())**2).sum()
R2_calm=r2(dd.loc[~st,macro].values,dd.loc[~st,cs].values); R2_str=r2(dd.loc[st,macro].values,dd.loc[st,cs].values)
calm={c:dd.loc[~st,c].corr(dd.loc[~st,cs]) for c in macro}
strs={c:dd.loc[st,c].corr(dd.loc[st,cs]) for c in macro}

fig=plt.figure(figsize=(12.4,6.4))
gs=fig.add_gridspec(1,3,width_ratios=[1.5,1.05,0.85],wspace=0.42,left=0.055,right=0.985,top=0.80,bottom=0.13)
headline(fig,"Macro data explains credit spreads only when the economy is already under stress",
   "Monthly changes, 1997–2022 (n=250). Regime defined from macro indicators alone — no credit-spread information — and it flags 22 of the 23 NBER recession months inside its window.")

# panel A - grouped bars |corr|
ax=fig.add_subplot(gs[0,0])
yp=np.arange(len(macro))[::-1]
h=0.36
ax.barh(yp+h/2,[abs(calm[c]) for c in macro],height=h,color=SLATE,alpha=.55,label='Calm months (n=187)')
ax.barh(yp-h/2,[abs(strs[c]) for c in macro],height=h,color=RED,label='Macro-stress months (n=63)')
for i,c in zip(yp,macro):
    ax.text(abs(calm[c])+.012,i+h/2,f"{abs(calm[c]):.2f}",va='center',fontsize=8.4,color=MUTED)
    ax.text(abs(strs[c])+.012,i-h/2,f"{abs(strs[c]):.2f}",va='center',fontsize=8.4,color=RED,fontweight='bold')
ax.set_yticks(yp); ax.set_yticklabels([NICE[c] for c in macro],fontsize=9.3)
ax.set_xlabel('|correlation| with monthly change in credit spread')
ax.set_xlim(0,.72); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.legend(loc='lower right',bbox_to_anchor=(1.0,-0.005))
ax.set_title('Every indicator strengthens under stress',fontsize=10.6,loc='left',pad=8,color=INK)

# panel B - joint R2
ax2=fig.add_subplot(gs[0,1])
bars=ax2.bar(['Calm\nmonths','Macro-stress\nmonths'],[R2_calm,R2_str],color=[SLATE,RED],width=.56)
bars[0].set_alpha(.55)
for b,v in zip(bars,[R2_calm,R2_str]):
    ax2.text(b.get_x()+b.get_width()/2,v+.018,f"{v:.2f}",ha='center',fontsize=15,fontweight='bold',
             color=SLATE if v<.2 else RED)
ax2.annotate('',xy=(1,R2_str-0.03),xytext=(0,R2_calm+0.03),
             arrowprops=dict(arrowstyle='-|>',color=AMBER,lw=1.8,connectionstyle='arc3,rad=-0.32'))
ax2.text(0.5,R2_str*0.60,f"{R2_str/R2_calm:.1f}×",ha='center',fontsize=15,fontweight='bold',color=AMBER)
ax2.set_ylim(0,.60); ax2.set_ylabel('R² of spread change on all six macro indicators')
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
ax2.set_title('Joint explanatory power',fontsize=10.6,loc='left',pad=8,color=INK)

# panel C - concentration
ax3=fig.add_subplot(gs[0,2])
_ddf=df.diff().dropna(); thr=df[cs].quantile(.75); hi=(df[cs]>=thr).reindex(_ddf.index)
tot=((_ddf[cs]-_ddf[cs].mean())**2).sum(); shr=((_ddf.loc[hi,cs]-_ddf[cs].mean())**2).sum()/tot
share_months=hi.mean()
ax3.bar([0],[share_months*100],width=.5,color=SLATE,alpha=.55)
ax3.bar([1],[shr*100],width=.5,color=RED)
for x,v in [(0,share_months*100),(1,shr*100)]:
    ax3.text(x,v+2.2,f"{v:.0f}%",ha='center',fontsize=15,fontweight='bold',color=SLATE if x==0 else RED)
ax3.set_xticks([0,1]); ax3.set_xticklabels(['share of\nmonths','share of\nspread variance'],fontsize=9.2)
ax3.set_ylim(0,100); ax3.set_ylabel('%'); ax3.yaxis.grid(True); ax3.set_axisbelow(True)
ax3.set_title('High-spread months\ndominate the variance',fontsize=10.6,loc='left',pad=8,color=INK)
footer(fig,"Source: FRED (CPIAUCSL, FEDFUNDS, INDPRO, UNRATE, UMCSENT, EA19LORSGPORGYSAM) and ICE BofA BAMLH0A0HYM2. "
           "Correlations are contemporaneous on first differences; this is association, not causation.")
plt.savefig(f'{FIG}/01_macro_is_a_switch_not_a_dial.png')
print("saved 01 |",f"R2_calm={R2_calm:.4f} R2_stress={R2_str:.4f} ratio={R2_str/R2_calm:.2f} varshare={shr:.4f} monthshare={share_months:.4f}")
