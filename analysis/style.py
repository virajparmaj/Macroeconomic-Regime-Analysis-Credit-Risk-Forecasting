import matplotlib as mpl, matplotlib.pyplot as plt
INK='#1A1A1A'; MUTED='#6B7280'; GRID='#E5E7EB'
BLUE='#2563EB'; RED='#DC2626'; AMBER='#D97706'; TEAL='#0D9488'; SLATE='#64748B'; VIOLET='#7C3AED'
def setup():
    mpl.rcParams.update({
        'figure.dpi':130,'savefig.dpi':200,'savefig.bbox':'tight','savefig.facecolor':'white',
        'font.family':'sans-serif',
        'font.sans-serif':['Helvetica Neue','Helvetica','Arial','DejaVu Sans'],
        'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'axes.edgecolor':GRID,'axes.linewidth':1.0,'axes.labelcolor':INK,
        'axes.spines.top':False,'axes.spines.right':False,
        'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
        'xtick.labelsize':9,'ytick.labelsize':9,
        'grid.color':GRID,'grid.linewidth':0.8,'legend.frameon':False,'legend.fontsize':9,
        'figure.facecolor':'white','axes.facecolor':'white',
    })
def headline(fig, title, sub, y=0.988):
    fig.text(0.008,y,title,fontsize=14.5,fontweight='bold',color=INK,va='top',ha='left')
    fig.text(0.008,y-0.062,sub,fontsize=9.8,color=MUTED,va='top',ha='left')
def footer(fig,txt,y=0.005):
    fig.text(0.008,y,txt,fontsize=7.8,color=MUTED,va='bottom',ha='left',style='italic')
