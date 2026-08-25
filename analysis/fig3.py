import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from style import *
setup()
FIG='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/results/figures'
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
RR=pd.read_csv('rolling_r2.csv',parse_dates=['date'],index_col='date')['r2']
cs=df['Credit_Spread']
fig,(ax,axb)=plt.subplots(2,1,figsize=(12.6,7.4),sharex=True,gridspec_kw=dict(height_ratios=[1.25,1],hspace=0.10,
                          left=0.06,right=0.955,top=0.845,bottom=0.10))
headline(fig,"The macro model fits worst right before a credit crisis and best right after one",
  "Rolling 60-month R² of the six macro indicators explaining monthly changes in the high-yield spread, 2001–2022. Rising fit is confirmation, not warning.")
nber=[('2001-03-31','2001-11-30'),('2007-12-31','2009-06-30'),('2020-02-29','2020-04-30')]
for a,b in nber:
    for A in (ax,axb): A.axvspan(pd.Timestamp(a),pd.Timestamp(b),color=SLATE,alpha=.15,zorder=0)

ax.plot(RR.index,RR.values,lw=2.0,color=BLUE,zorder=3)
ax.fill_between(RR.index,0,RR.values,color=BLUE,alpha=.10,zorder=2)
lo5=RR.nsmallest(5); hi5=RR.nlargest(5)
ax.scatter(lo5.index,lo5.values,s=52,color=RED,zorder=5,edgecolor='white',linewidth=1.1)
ax.scatter(hi5.index,hi5.values,s=52,color=TEAL,zorder=5,edgecolor='white',linewidth=1.1)
ax.axhline(RR.mean(),color=MUTED,ls='--',lw=.9)
ax.text(RR.index[3],RR.mean()+.012,f'sample mean {RR.mean():.2f}',fontsize=8.2,color=MUTED)
mn=RR.idxmin()
ax.annotate(f"May 2008 — lowest fit in 26 years (R²={RR.min():.2f})\nthe spread then widened +13.6pp in 7 months",
            xy=(mn,RR.min()),xytext=(pd.Timestamp('2002-04-30'),0.055),fontsize=9.3,color=RED,fontweight='bold',
            arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6,connectionstyle='arc3,rad=0.18'))
mx=RR.idxmax()
ax.annotate(f"Apr 2020 — highest fit in 26 years (R²={RR.max():.2f})\nafter the shock had already happened",
            xy=(mx,RR.max()),xytext=(pd.Timestamp('2013-02-28'),0.70),fontsize=9.3,color=TEAL,fontweight='bold',
            arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1.6,connectionstyle='arc3,rad=-0.20'))
ax.set_ylabel('rolling 60-month R²'); ax.set_ylim(0,0.86); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.text(0.012,0.955,'●',transform=ax.transAxes,fontsize=11,color=RED,va='top')
ax.text(0.030,0.952,'5 worst-fit months',transform=ax.transAxes,fontsize=8.6,color=INK,va='top')
ax.text(0.150,0.955,'●',transform=ax.transAxes,fontsize=11,color=TEAL,va='top')
ax.text(0.168,0.952,'5 best-fit months',transform=ax.transAxes,fontsize=8.6,color=INK,va='top')

axb.plot(cs.index,cs.values,lw=1.7,color=RED)
axb.fill_between(cs.index,0,cs.values,color=RED,alpha=.09)
axb.set_ylabel('high-yield credit spread (pp)'); axb.set_ylim(0,22)
axb.yaxis.grid(True); axb.set_axisbelow(True)
for a,b in nber: pass
axb.axvspan(pd.Timestamp('2007-11-30'),pd.Timestamp('2008-08-31'),color=RED,alpha=.13,zorder=1)
axb.text(pd.Timestamp('2008-02-28'),19.0,'the 5 worst-fit\nmonths sit here',fontsize=8.6,color=RED,
         ha='center',fontweight='bold')
axb.set_xlabel('')
axb.set_xlim(RR.index.min(),RR.index.max())
footer(fig,"Grey bands = NBER recessions. R² is in-sample within each 60-month window and is a diagnostic of relationship stability, not a forecast. "
  "Rolling R² correlates +0.33 with the spread 9 months EARLIER and −0.26 with the spread 12 months later: it is a lagging indicator.")
plt.savefig(f'{FIG}/03_fit_collapses_before_crises.png')
print('saved 03')
