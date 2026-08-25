import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from style import *
setup()
FIG='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/results/figures'
P=pd.read_csv('preds_macro+spread_history.csv',parse_dates=['idx'],index_col='idx')
Pm=pd.read_csv('preds_macro_only.csv',parse_dates=['idx'],index_col='idx')

fig=plt.figure(figsize=(12.6,6.6))
gs=fig.add_gridspec(1,3,width_ratios=[1.0,1.0,1.28],wspace=0.36,left=0.055,right=0.982,top=0.795,bottom=0.135)
headline(fig,"The model that looked like R² = 0.75 is 82% worse than assuming nothing changes",
  "Walk-forward test, 14 expanding origins, 163 months (2009–2022), point-in-time macro alignment, forecasting the spread one month ahead.")

# A: the two ways of reporting
ax=fig.add_subplot(gs[0,0])
vals=[0.7503,-2.3037]; labs=['R² vs the\ntest-period mean','R²  vs the\nrandom walk']
b=ax.bar([0,1],vals,color=[TEAL,RED],width=.56)
ax.axhline(0,color=INK,lw=1.0)
ax.text(0,0.7503+.14,'+0.75',ha='center',fontsize=17,fontweight='bold',color=TEAL)
ax.text(1,-2.3037-.42,'−2.30',ha='center',fontsize=17,fontweight='bold',color=RED)
ax.text(0,0.7503+.62,'"looks good"',ha='center',fontsize=8.8,color=MUTED,style='italic')
ax.text(1,-2.3037-.80,'"loses badly"',ha='center',fontsize=8.8,color=MUTED,style='italic')
ax.set_xticks([0,1]); ax.set_xticklabels(labs,fontsize=9.2)
ax.set_ylim(-3.4,1.7); ax.set_ylabel('R²'); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.set_title('Same predictions, two benchmarks',fontsize=10.6,loc='left',pad=8,color=INK)

# B: RMSE ladder
ax2=fig.add_subplot(gs[0,1])
names=['Random walk\n(assume no change)','Random forest\nmacro + spread','Random forest\nmacro only']
rm=[0.6124,1.1131,2.4004]; cols=[TEAL,AMBER,RED]
bars=ax2.barh(np.arange(3)[::-1],rm,color=cols,height=.52)
for i,(v,c) in enumerate(zip(rm,cols)):
    ax2.text(v+.06,2-i,f"{v:.2f}",va='center',fontsize=12,fontweight='bold',color=c)
    if i>0: ax2.text(v+.06,2-i-0.28,f"+{100*(v/rm[0]-1):.0f}% error",va='center',fontsize=8.4,color=MUTED)
ax2.set_yticks(np.arange(3)[::-1]); ax2.set_yticklabels(names,fontsize=8.8)
ax2.set_xlabel('RMSE, percentage points of spread'); ax2.set_xlim(0,3.05)
ax2.xaxis.grid(True); ax2.set_axisbelow(True)
ax2.set_title('Error against the honest benchmark',fontsize=10.6,loc='left',pad=8,color=INK)

# C: actual vs predicted over time
ax3=fig.add_subplot(gs[0,2])
ax3.plot(P.index,P['y'],lw=2.0,color=INK,label='actual spread',zorder=4)
ax3.plot(P.index,P['rw'],lw=1.4,color=TEAL,ls='--',label='random walk',zorder=3)
ax3.plot(P.index,P['model'],lw=1.4,color=RED,label='RF (macro + spread)',zorder=3)
ax3.plot(Pm.index,Pm['model'],lw=1.2,color=SLATE,alpha=.75,label='RF (macro only)',zorder=2)
ax3.set_ylabel('high-yield spread (pp)'); ax3.legend(loc='upper right',fontsize=8.3)
ax3.yaxis.grid(True); ax3.set_axisbelow(True)
_fa=Pm['model'].idxmax()
ax3.annotate('Jul 2020: macro-only model predicts\n14.5pp. The actual spread was 5.1pp.',
             xy=(_fa,Pm['model'].max()),
             xytext=(pd.Timestamp('2011-06-30'),13.4),fontsize=8.6,color=SLATE,
             arrowprops=dict(arrowstyle='-|>',color=SLATE,lw=1.2,connectionstyle='arc3,rad=-0.18'))
ax3.set_title('The models track, but never lead',fontsize=10.6,loc='left',pad=8,color=INK)
footer(fig,"R² vs random walk = 1 − SSE(model)/SSE(random walk). Diebold–Mariano vs the random walk: p < 0.0001 for macro-only, p = 0.056 for macro+spread. "
  "The same conclusion holds when the target is the monthly CHANGE, at horizons of 1, 3, 6 and 12 months, and for Ridge as well as random forest.")
plt.savefig(f'{FIG}/04_nothing_beats_the_random_walk.png')
print('saved 04')
