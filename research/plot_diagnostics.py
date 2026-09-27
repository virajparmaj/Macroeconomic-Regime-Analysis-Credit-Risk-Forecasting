"""Render the advisory's exploratory diagnostics from saved numeric artifacts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": "#25374a",
                     "text.color": "#25374a", "axes.edgecolor": "#c5cdd5"})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), gridspec_kw={"width_ratios":[1.2,1,1]})
fig.patch.set_facecolor("#f8fafb")
for ax in axes:
    ax.set_facecolor("#f8fafb")
    ax.grid(axis="y", color="#e4e9ee", linewidth=.7)
    ax.set_axisbelow(True)

c = pd.read_csv(E / "classification_label_diagnostic.csv")
c = c[c.future_label_fix]
for score, label, color in [("current_spread_score", "Current spread", "#167f78"),
                             ("rf_score", "Random forest", "#b85840")]:
    p = c[c.score == score]
    axes[0].plot(p.h, p.auc, marker="o", lw=2, color=color, label=label)
    for _, row in p.iterrows():
        axes[0].annotate(f"{row.auc:.3f}", (row.h,row.auc),
                         xytext=(0,8 if score == "current_spread_score" else -17),
                         textcoords="offset points",ha="center",fontsize=9,color=color)
axes[0].axhline(.5, color="#83909d", ls="--", lw=1)
axes[0].set(xticks=[1,3,6,12], ylim=(.46,1.01), xlabel="Forecast horizon (months)", ylabel="ROC-AUC")
axes[0].set_title("A  State ranking: baseline wins",loc="left",fontsize=12,pad=15)
axes[0].legend(loc="upper right",frameon=False,fontsize=9)

b = pd.read_csv(E / "aggregation_benchmarks.csv")
b = b[(b.origin_start == "2009-01-31") & (b.target == "next_month_panel_ffill_mean")]
axes[1].bar([0,1], b.rmse, color=["#8296ae","#167f78"],width=.58)
for i, row in enumerate(b.itertuples()):
    axes[1].text(i,row.rmse+.025,f"{row.rmse:.4f}",ha="center",fontweight="bold")
axes[1].set(xticks=[0,1], xticklabels=["Previous monthly\nmean", "Latest daily\nobservation"],
            ylim=(0,.75), ylabel="RMSE (percentage points)")
axes[1].set_title("B  Same target, stronger baseline",loc="left",fontsize=12,pad=15)
axes[1].text(.5,.93,"31.2% lower RMSE · 163 origins",transform=axes[1].transAxes,
              ha="center",fontsize=10)

o = pd.read_csv(E / "onset_counts.csv")
o = o[o.h == 1]
counts = o.distinct_onset_dates_in_test.str.split(";").map(len)
axes[2].bar([0,1], counts,color=["#8296ae","#b85840"],width=.58)
for i,n in enumerate(counts):
    axes[2].text(i,n+.15,str(n),ha="center",fontweight="bold",fontsize=12)
axes[2].set(xticks=[0,1],xticklabels=["Full-sample q75\n(hindsight)","Pre-2009 q75\n(frozen)"],
            ylim=(0,8),yticks=[0,2,4,6,8],ylabel="Post-2008 threshold entries")
axes[2].set_title("C  Event evidence is scarce",loc="left",fontsize=12,pad=15)
fig.suptitle("High scores do not yet establish advance warning",fontsize=19,fontweight="bold",x=.055,ha="left",y=.99)
fig.text(.055,.015,
         "Exploratory audit, not a completed forecasting study. A: unknown future labels removed; hindsight threshold retained; no confidence intervals.\n"
         "B: identical legacy monthly-mean target and dates. C: threshold entries are not necessarily independent economic episodes.",
         fontsize=9,color="#526170",va="bottom")
fig.subplots_adjust(left=.065,right=.985,top=.80,bottom=.23,wspace=.36)
fig.savefig(E / "research_diagnostics.png",dpi=180,facecolor=fig.get_facecolor())
plt.close(fig)
