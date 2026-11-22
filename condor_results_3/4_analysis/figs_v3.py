import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
 'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.5,
 'xtick.major.width':0.5,'ytick.major.width':0.5,'axes.edgecolor':'#555','xtick.color':'#333','ytick.color':'#333'})
R='/home/claude/v3res/results/'
g=pd.read_csv(R+'infer_results_graded.csv',dtype={'prompt_id':str})
NAME={'small_1.5B':'1.5B','large_3B':'3B','coder_1.5B':'Coder-1.5B'}
C={'1.5B':'#2a78d6','3B':'#eb6834','Coder-1.5B':'#1baf7a'}; M={'1.5B':'o','3B':'s','Coder-1.5B':'^'}
def style(ax):
    ax.grid(True,color='#e6e6e6',linewidth=0.4); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2,pad=1.5)
# ---------- Fig 1: quality vs energy with 95% CI
rng=np.random.default_rng(42)
def ci(x):
    x=np.asarray(x,float); m=rng.choice(x,(5000,len(x))).mean(1); return np.percentile(m,[2.5,97.5])
titles={'factual_qa':'QA (MMLU-Pro, n=1000)','summarization':'Summarization (XSum, n=200)','coding':'Coding (MBPP+, n=378)'}
scale={'factual_qa':100,'summarization':1,'coding':100}
ylab={'factual_qa':'Accuracy (%)','summarization':'ROUGE-L','coding':'Pass@1 (%)'}
fig,axs=plt.subplots(1,3,figsize=(3.5,1.8))
for ax,t in zip(axs,['factual_qa','summarization','coding']):
    xs,ys=[],[]
    for mk,mn in NAME.items():
        d=g[(g.task==t)&(g.model_name==mk)]; x=d.gpu_energy_avgp_latency_j.mean(); y=d.quality_score.mean()*scale[t]
        lo,hi=ci(d.quality_score)*scale[t]
        ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='none',ecolor=C[mn],elinewidth=0.8,capsize=1.5,zorder=2)
        ax.scatter(x,y,s=24,marker=M[mn],color=C[mn],edgecolor='white',linewidth=0.8,zorder=3,label=mn)
        xs.append(x); ys+= [lo,hi]
    xr=max(xs)-min(xs); yr=max(ys)-min(ys)
    ax.set_xlim(min(xs)-0.25*xr,max(xs)+0.25*xr); ax.set_ylim(min(ys)-0.15*yr,max(ys)+0.15*yr)
    ax.set_title(titles[t],fontsize=6.2,pad=3); ax.set_ylabel(ylab[t],labelpad=1); style(ax)
    ax.locator_params(axis='x',nbins=3); ax.locator_params(axis='y',nbins=4)
axs[1].set_xlabel('GPU energy per query (J)',labelpad=2)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=6.5,handletextpad=0.2,columnspacing=1.2,bbox_to_anchor=(0.5,1.03))
fig.tight_layout(pad=0.3,w_pad=0.5,rect=(0,0,1,0.9)); fig.savefig('fig1_v3.png',dpi=400); plt.close(fig)
# ---------- Fig 2: QA energy vs input tokens
fig,ax=plt.subplots(figsize=(3.5,1.9))
qa=g[g.task=='factual_qa']
for mk in ['large_3B','small_1.5B','coder_1.5B']:
    mn=NAME[mk]; d=qa[qa.model_name==mk]
    ax.scatter(d.input_tokens,d.gpu_energy_avgp_latency_j,s=3,marker=M[mn],color=C[mn],alpha=0.45,linewidth=0,label=mn,zorder=2)
    k=np.polyfit(d.input_tokens,d.gpu_energy_avgp_latency_j,1); xx=np.linspace(d.input_tokens.min(),d.input_tokens.max(),10)
    ax.plot(xx,np.polyval(k,xx),color=C[mn],linewidth=1.1,zorder=3)
ax.set_xlabel('Prompt length (input tokens)',labelpad=2); ax.set_ylabel('Energy per query (J)',labelpad=2); style(ax)
leg=ax.legend(frameon=False,fontsize=6.5,loc='upper left',handletextpad=0.3,markerscale=2.5)
for lh in leg.legend_handles: lh.set_alpha(1)
fig.tight_layout(pad=0.3); fig.savefig('fig2_v3.png',dpi=400); plt.close(fig)
# ---------- Fig 3: router trade-off on test split
sw=pd.read_csv(R+'router_sweep_test.csv'); ev=pd.read_csv(R+'router_eval.csv'); ev=ev[ev.split=='test']
fig,axs=plt.subplots(1,2,figsize=(3.5,2.1))
for ax,(t,first,sc,yl) in zip(axs,[('factual_qa','small_1.5B',100,'Accuracy (%)'),('coding','coder_1.5B',100,'Pass@1 (%)')]):
    s=sw[(sw.task==t)&(sw.first_model==first)].sort_values('energy_mean_j')
    ax.plot(s.energy_mean_j,s.quality*sc,color='#777',linewidth=1.0,zorder=1,label='Cascade sweep (cheap model → 3B)')
    for pol,mk in [('always small_1.5B','1.5B'),('always large_3B','3B'),('always coder_1.5B','Coder-1.5B')]:
        r=ev[(ev.policy==pol)&(ev.task==t)].iloc[0]
        ax.scatter(r.energy_mean_j,r.quality*sc,s=22,marker=M[mk],color=C[mk],edgecolor='white',linewidth=0.7,zorder=3)
    if t=='factual_qa':
        r=ev[(ev.policy=='cascade (small first, escalate to 3B)')&(ev.task==t)].iloc[0]
        ax.scatter(r.energy_mean_j,r.quality*sc,s=26,marker='D',facecolor='none',edgecolor='#222',linewidth=0.8,zorder=4,label='Fitted cascade')
        r=ev[(ev.policy=='task-aware + QA by subject')&(ev.task==t)].iloc[0]
        ax.scatter(r.energy_mean_j,r.quality*sc,s=30,marker='*',color='#222',zorder=4,label='By subject')
    r=ev[(ev.policy=='oracle (upper bound)')&(ev.task==t)].iloc[0]
    ax.scatter(r.energy_mean_j,r.quality*sc,s=22,marker='x',color='#222',linewidth=0.9,zorder=4,label='Oracle')
    ax.set_title('QA (test split)' if t=='factual_qa' else 'Coding (test split)',fontsize=6.5,pad=3)
    ax.set_ylabel(yl,labelpad=1); ax.set_xlabel('Energy per query (J)',labelpad=2); style(ax)
    ax.locator_params(axis='x',nbins=4)
from matplotlib.lines import Line2D
h1,l1=axs[0].get_legend_handles_labels()
h1=[Line2D([],[],marker=M[m],color=C[m],linestyle='none',markersize=4) for m in ['1.5B','3B','Coder-1.5B']]+h1
l1=['Always 1.5B','Always 3B','Always Coder']+l1
fig.legend(h1,l1,loc='upper center',ncol=4,frameon=False,fontsize=5.6,handletextpad=0.3,columnspacing=0.8,bbox_to_anchor=(0.5,0.995))
fig.tight_layout(pad=0.3,w_pad=0.8,rect=(0,0,1,0.8)); fig.savefig('fig3_v3.png',dpi=400); plt.close(fig)
print('ok')
