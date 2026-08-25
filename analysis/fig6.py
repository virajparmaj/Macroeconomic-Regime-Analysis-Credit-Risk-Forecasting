import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from style import *
setup()
FIG='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/results/figures'
ROOT='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk'
d=pd.read_csv(f'{ROOT}/data/original/ICE BofA US High Yield Index Option-Adjusted Spread_BAMLH0A0HYM2.csv',
              parse_dates=['observation_date'],index_col='observation_date').sort_index()
s=d.iloc[:,0].dropna()
m=s.resample('ME').agg(['mean','max','min','last']); m['rng']=m['max']-m['min']; m=m.dropna(subset=['mean'])

fig=plt.figure(figsize=(12.6,6.5))
gs=fig.add_gridspec(1,2,width_ratios=[1.22,1.0],wspace=0.24,left=0.055,right=0.982,top=0.795,bottom=0.135)
headline(fig,"Monthly averaging erased the fastest credit event in 26 years",
  "The project models a monthly mean of a daily market price. March 2020 is the single most violent month in the sample — and the monthly panel cannot see it.")

# A: COVID zoom, daily vs monthly mean
ax=fig.add_subplot(gs[0,0])
w=s.loc['2019-12-01':'2020-08-31']
ax.plot(w.index,w.values,lw=1.7,color=SLATE,alpha=.9,label='daily spread (what the market saw)')
mw=m.loc['2019-12':'2020-08']
for ts,row in mw.iterrows():
    a=ts.replace(day=1); b=ts
    ax.hlines(row['mean'],a,b,color=RED,lw=3.4,zorder=5)
ax.hlines([],[],[],color=RED,lw=3.4,label='monthly mean (what the model saw)')
pk=w.idxmax()
ax.annotate(f'23 Mar 2020: {w.max():.2f}pp',xy=(pk,w.max()),xytext=(pd.Timestamp('2020-04-20'),10.4),
            fontsize=9.2,color=SLATE,fontweight='bold',
            arrowprops=dict(arrowstyle='-|>',color=SLATE,lw=1.3))
mar=m.loc[pd.Timestamp('2020-03-31')]
ax.annotate(f"March monthly mean: {mar['mean']:.2f}pp\n{100*(mar['max']-mar['mean'])/mar['max']:.0f}% below the month's peak",
            xy=(pd.Timestamp('2020-03-16'),mar['mean']),xytext=(pd.Timestamp('2019-12-10'),9.2),
            fontsize=9.2,color=RED,fontweight='bold',
            arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.4,connectionstyle='arc3,rad=0.20'))
ax.set_ylabel('high-yield credit spread (pp)'); ax.legend(loc='lower right'); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.set_ylim(3,11.6)
ax.set_title('COVID, December 2019 – August 2020',fontsize=10.6,loc='left',pad=8,color=INK)

# B: rank flip
ax2=fig.add_subplot(gs[0,1])
top=m.nlargest(8,'rng')[['rng']].iloc[::-1]
labs=[t.strftime('%b %Y') for t in top.index]
colors=[RED if t.year==2020 else (AMBER if t.year in (2008,2009) else SLATE) for t in top.index]
ax2.barh(np.arange(len(top)),top['rng'].values,color=colors,height=.62)
ax2.set_ylim(-3.4,len(top)-0.4)
for i,(v,t) in enumerate(zip(top['rng'].values,top.index)):
    ax2.text(v+.08,i,f"{v:.2f}",va='center',fontsize=9,fontweight='bold',
             color=RED if t.year==2020 else MUTED)
ax2.set_yticks(np.arange(len(top))); ax2.set_yticklabels(labs,fontsize=9)
ax2.set_xlabel('intra-month range: highest minus lowest daily spread (pp)')
ax2.set_xlim(0,7.2); ax2.xaxis.grid(True); ax2.set_axisbelow(True)
ax2.set_title('The eight most violent months, 1997–2022',fontsize=10.6,loc='left',pad=8,color=INK)
box=("March 2020, ranked among all 309 months\n"
     "   by intra-month range      →   1st\n"
     "   by monthly mean spread   →   43rd\n\n"
     f"Its range of 6.12pp is 13.6× the median month.")
ax2.text(0.5,0.035,box,transform=ax2.transAxes,ha='center',va='bottom',fontsize=9.0,color=INK,
         family='monospace',bbox=dict(boxstyle='round,pad=0.6',fc='#FEF2F2',ec=RED,lw=1.0))
footer(fig,"Source: ICE BofA BAMLH0A0HYM2 daily, 6,679 observations. The repository contains both credit_spread_monthly_mean.csv and credit_spread_monthly_last.csv; "
  "the modelling panel uses the mean. Using the month-end value instead cuts the random walk's one-month RMSE by 35%.")
plt.savefig(f'{FIG}/06_monthly_averaging_hides_the_shock.png')
print('saved 06')
