import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from style import *
setup()
FIG='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/results/figures'
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
df=pd.read_csv(f'{ROOT}/data/merged_macroeconomic_credit.csv',parse_dates=['Month_End'],index_col='Month_End').sort_index()
df=df.rename(columns={'GDP':'EA19_GDP_OECD'})
macro=['CPI','FEDFUNDS','Industrial_Production','EA19_GDP_OECD','Unemployment_Rate','Consumer_Sentiment']
NICE={'FEDFUNDS':'Fed Funds','CPI':'CPI','EA19_GDP_OECD':'OECD GDP (EA19)','Consumer_Sentiment':'Consumer sentiment',
      'Industrial_Production':'Industrial production','Unemployment_Rate':'Unemployment'}
cs='Credit_Spread'; dd=df.diff()
ks=list(range(-6,7))

fig=plt.figure(figsize=(12.4,6.3))
gs=fig.add_gridspec(1,2,width_ratios=[1.32,1.0],wspace=0.30,left=0.055,right=0.985,top=0.79,bottom=0.14)
headline(fig,"The credit spread moves before the macro data, not after it",
  "Left: cross-correlation of each macro indicator's change with the spread's change at different leads. Right: months of warning before each NBER recession start.")

ax=fig.add_subplot(gs[0,0])
ax.axvspan(-6.5,-0.5,color=RED,alpha=.055); ax.axvspan(0.5,6.5,color=BLUE,alpha=.055)
M=np.array([[abs(dd[c].shift(k).corr(dd[cs])) for k in ks] for c in macro])
mean_abs=M.mean(0)
bcols=[RED if k<0 else (SLATE if k==0 else BLUE) for k in ks]
ax.bar(ks,mean_abs,width=.74,color=bcols,alpha=.85,zorder=2)
cols=[RED,AMBER,TEAL,VIOLET,SLATE,BLUE]
for row,col in zip(M,cols):
    ax.plot(ks,row,lw=.9,color=col,alpha=.42,zorder=3)
ax.axvline(0,color=MUTED,lw=.9,ls=':')
ax.set_xlabel('lead k (months)   —   negative k = spread moved first')
ax.set_ylabel('mean |correlation| across the six indicators')
ax.set_xticks(ks); ax.yaxis.grid(True); ax.set_axisbelow(True)
top=ax.get_ylim()[1]*1.14; ax.set_ylim(0,top)
ax.text(-3.5,top*0.94,'spread moves FIRST',ha='center',fontsize=9.8,fontweight='bold',color=RED)
ax.text(3.5,top*0.94,'macro moves first',ha='center',fontsize=9.8,fontweight='bold',color=BLUE)
peak=ks[int(np.argmax(mean_abs))]
ax.annotate(f'peak at k={peak}\nmean |r| = {mean_abs.max():.2f}',xy=(peak,mean_abs.max()),
            xytext=(peak-2.6,top*0.78),fontsize=9,color=RED,fontweight='bold',
            arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.4))
kpos=mean_abs[[i for i,k in enumerate(ks) if k>0]].mean()
kneg=mean_abs[[i for i,k in enumerate(ks) if k<0]].mean()
ax.text(0.99,0.42,f'mean |r|, spread-first side: {kneg:.3f}\nmean |r|, macro-first side: {kpos:.3f}',
        transform=ax.transAxes,ha='right',fontsize=8.8,color=INK,
        bbox=dict(boxstyle='round,pad=0.45',fc='white',ec=GRID))
ax.set_title('The relationship peaks one month AFTER the spread has already moved',fontsize=10.6,loc='left',pad=8,color=INK)

ax2=fig.add_subplot(gs[0,1])
data={'2001 recession':{'Credit spread':12,'Consumer sentiment':5,'Industrial production':1,'Unemployment':0},
      '2008 recession':{'Credit spread':5,'Consumer sentiment':19,'Industrial production':0,'Unemployment':0},
      '2020 recession':{'Credit spread':15,'Consumer sentiment':13,'Industrial production':10,'Unemployment':0}}
sigs=['Credit spread','Consumer sentiment','Industrial production','Unemployment']
scol={'Credit spread':RED,'Consumer sentiment':TEAL,'Industrial production':AMBER,'Unemployment':SLATE}
xs=np.arange(3); w=.2
for i,s in enumerate(sigs):
    vals=[data[e][s] for e in data]
    b=ax2.bar(xs+(i-1.5)*w,vals,width=w,color=scol[s],label=s,alpha=1.0 if s=='Credit spread' else .62)
    for x,v in zip(xs+(i-1.5)*w,vals):
        ax2.text(x,v+0.35,('—' if v==0 else str(v)),ha='center',fontsize=8.3,
                 color=scol[s] if v>0 else MUTED,fontweight='bold' if s=='Credit spread' else 'normal')
ax2.set_xticks(xs); ax2.set_xticklabels(list(data.keys()),fontsize=9.4)
ax2.set_ylabel('months of warning before recession start')
ax2.yaxis.grid(True); ax2.set_axisbelow(True); ax2.legend(loc='upper left',fontsize=8.4)
ax2.set_ylim(0,23)
ax2.set_title('Unemployment never gave any warning; it fired at the start',fontsize=10.6,loc='left',pad=8,color=INK)
footer(fig,"Warning = first month a signal crossed its 12-month mean ±1 s.d. (unemployment: +0.5pp off its 12-month low) within 24 months of the recession start. "
  "Granger tests on first differences: in both sample halves, macro leads the spread for 0 of 6 indicators. Predictive precedence, not causation.")
plt.savefig(f'{FIG}/02_spread_leads_the_macro_data.png')
print('saved 02')
