import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from style import *
setup()
FIG='/Users/veerr_89/Work/projects/Macroeconomic-Regime-Analysis-Credit-Risk/results/figures'
H=[1,3,6,12]
auc={'spread-only':[0.935,0.793,0.656,0.543],'macro+spread':[0.920,0.782,0.613,0.519],'macro-only':[0.751,0.686,0.639,0.431]}
lift={'spread-only':[4.38,2.55,1.52,1.28],'macro+spread':[3.52,2.27,1.70,1.37],'macro-only':[1.87,1.84,1.74,1.42]}
cols={'spread-only':RED,'macro+spread':AMBER,'macro-only':SLATE}

fig=plt.figure(figsize=(12.4,6.2))
gs=fig.add_gridspec(1,2,width_ratios=[1.18,1.0],wspace=0.26,left=0.06,right=0.982,top=0.80,bottom=0.135)
headline(fig,"Credit stress is predictable three months out, and not at all at twelve",
  "Walk-forward classification: will the spread sit in its top quartile (≥6.43pp) h months from now? 164 test months, 2009–2022.")

ax=fig.add_subplot(gs[0,0])
ax.axhspan(0.5,0.7,color=SLATE,alpha=.07)
ax.axhline(0.5,color=MUTED,lw=1.1,ls='--')
ax.text(1.05,0.507,'coin flip',fontsize=8.6,color=MUTED,va='bottom',ha='left')
_off={'spread-only':(0,0.020),'macro+spread':(0,-0.032),'macro-only':(0,0.020)}
for k,v in auc.items():
    ax.plot(H,v,marker='o',ms=7,lw=2.2 if k=='spread-only' else 1.7,color=cols[k],
            label=k,alpha=1.0 if k!='macro-only' else .85)
    dx,dy=_off[k]
    for x,y in zip(H,v):
        ax.text(x+dx,y+dy,f"{y:.2f}",ha='center',fontsize=8.3,color=cols[k],
                va='bottom' if dy>0 else 'top',
                fontweight='bold' if k=='spread-only' else 'normal')
ax.axvspan(3,6,color=RED,alpha=.06)
ax.text(4.5,0.418,'usable signal\nends here',ha='center',fontsize=8.8,color=RED,fontweight='bold')
ax.set_xticks(H); ax.set_xlabel('forecast horizon h (months ahead)')
ax.set_ylabel('AUC — ability to rank stressed months above calm ones')
ax.set_ylim(0.40,0.99); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(loc='upper right')
ax.set_title('Adding macro data makes the classifier worse at every horizon',fontsize=10.6,loc='left',pad=8,color=INK)

ax2=fig.add_subplot(gs[0,1])
x=np.arange(len(H)); w=.26
for i,(k,v) in enumerate(lift.items()):
    b=ax2.bar(x+(i-1)*w,v,width=w,color=cols[k],label=k,alpha=1.0 if k=='spread-only' else .8)
    for xx,vv in zip(x+(i-1)*w,v):
        ax2.text(xx,vv+.06,f"{vv:.1f}×",ha='center',fontsize=8.4,color=cols[k],fontweight='bold')
ax2.axhline(1.0,color=INK,lw=1.1)
ax2.text(-0.42,1.07,'no better than the base rate',fontsize=8.4,color=MUTED,ha='left')
ax2.set_xticks(x); ax2.set_xticklabels([f'h={h}' for h in H])
ax2.set_ylabel('precision lift over the base rate'); ax2.set_ylim(0,5.1)
ax2.yaxis.grid(True); ax2.set_axisbelow(True); ax2.legend(loc='upper right')
ax2.set_title('At one month the spread alone is 4.4× better than guessing',fontsize=10.6,loc='left',pad=8,color=INK)
footer(fig,"Random-forest classifiers, expanding walk-forward origins with a 12-month embargo, balanced class weights. Lift = average precision ÷ base rate. "
  "Threshold 6.43pp is the in-sample 75th percentile and would need to be set on training data only in production.")
plt.savefig(f'{FIG}/05_forecastable_horizon_is_three_months.png')
print('saved 05')
