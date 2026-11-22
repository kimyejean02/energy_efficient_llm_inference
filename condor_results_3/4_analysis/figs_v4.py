import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
 'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.5,
 'xtick.major.width':0.5,'ytick.major.width':0.5,'axes.edgecolor':'#555','xtick.color':'#333','ytick.color':'#333'})
v3=pd.read_csv('/home/claude/v3res/results/infer_results_graded.csv',dtype={'prompt_id':str})
v4=pd.read_csv('/home/claude/v4res/results_v4/infer_results_v4_graded.csv',dtype={'prompt_id':str})
g=pd.concat([v3,v4[v4.task!='factual_qa_shuffled']])
NAME={'small_1.5B':'1.5B','large_3B':'3B','coder_1.5B':'Coder-1.5B','coder_3B':'Coder-3B','math_1.5B':'Math-1.5B','summ_1.5B':'Summarizer-1.5B'}
C={'1.5B':'#2a78d6','3B':'#eb6834','Coder-1.5B':'#1baf7a','Coder-3B':'#7b4fd0','Math-1.5B':'#c2347a','Summarizer-1.5B':'#9a7b1c'}
M={'1.5B':'o','3B':'s','Coder-1.5B':'^','Coder-3B':'v','Math-1.5B':'D','Summarizer-1.5B':'P'}
rng=np.random.default_rng(42)
def ci(x):
    x=np.asarray(x,float); m=rng.choice(x,(5000,len(x))).mean(1); return np.percentile(m,[2.5,97.5])
def style(ax):
    ax.grid(True,color='#e6e6e6',linewidth=0.4); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2,pad=1.5)
panels=[('coding','Coding (MBPP+, n=378)',100,'Pass@1 (%)',['small_1.5B','large_3B','coder_1.5B','coder_3B']),
        ('math','Math (GSM8K, n=100)',100,'Accuracy (%)',['small_1.5B','large_3B','math_1.5B']),
        ('summarization','Summ. (XSum, n=200)',1,'ROUGE-L',['small_1.5B','large_3B','coder_1.5B','summ_1.5B'])]
fig,axs=plt.subplots(1,3,figsize=(3.5,1.95))
handles={}
for ax,(t,title,sc,yl,models) in zip(axs,panels):
    xs,ys=[],[]
    for mk in models:
        mn=NAME[mk]; d=g[(g.task==t)&(g.model_name==mk)]
        x=d.gpu_energy_avgp_latency_j.mean(); y=d.quality_score.mean()*sc; lo,hi=ci(d.quality_score)*sc
        ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='none',ecolor=C[mn],elinewidth=0.8,capsize=1.5,zorder=2)
        handles[mn]=ax.scatter(x,y,s=30 if mn.startswith('Summ') else 22,marker=M[mn],color=C[mn],edgecolor='none' if mn.startswith('Summ') else 'white',linewidth=0.7,zorder=3)
        xs.append(x); ys+=[lo,hi]
    xr=max(xs)-min(xs); yr=max(ys)-min(ys)
    ax.set_xlim(min(xs)-0.2*xr,max(xs)+0.2*xr); ax.set_ylim(min(ys)-0.12*yr,max(ys)+0.12*yr)
    ax.set_title(title,fontsize=6.2,pad=3); ax.set_ylabel(yl,labelpad=1); style(ax)
    ax.locator_params(axis='x',nbins=3); ax.locator_params(axis='y',nbins=4)
axs[1].set_xlabel('GPU energy per query (J)',labelpad=2)
order=['1.5B','3B','Coder-1.5B','Coder-3B','Math-1.5B','Summarizer-1.5B']
fig.legend([handles[o] for o in order],order,loc='upper center',ncol=3,frameon=False,fontsize=6.2,handletextpad=0.2,columnspacing=1.0,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(pad=0.3,w_pad=0.5,rect=(0,0,1,0.83)); fig.savefig('fig4_v4.png',dpi=400); plt.close(fig)
print('ok')
