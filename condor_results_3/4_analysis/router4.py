import pandas as pd, numpy as np, json
from scipy import stats
LARGE="large_3B"; TOL={"factual_qa":0.02,"summarization":0.01,"coding":0.02,"math":0.02}
v3=pd.read_csv('v3res/results/infer_results_graded.csv',dtype={'prompt_id':str})
v4=pd.read_csv('v4res/results_v4/infer_results_v4_graded.csv',dtype={'prompt_id':str})
v4=v4[v4.task!='factual_qa_shuffled']
g=pd.concat([v3,v4],ignore_index=True)
g["energy"]=g["gpu_energy_avgp_latency_j"]
g["confidence"]=np.where(g.task=="factual_qa",g.first_logprob,g.mean_logprob)
def tables(task,split,models=None):
    d=g[(g.task==task)&(g.split==split)]
    if models is not None: d=d[d.model_name.isin(models)]
    p=lambda c:d.pivot(index="prompt_id",columns="model_name",values=c)
    return p("quality_score"),p("energy"),p("confidence"),d.drop_duplicates("prompt_id").set_index("prompt_id")["category"]
def full(Q): return [m for m in Q.columns if Q[m].notna().all()]
def apply(spec,Q,E,C,cat=None):
    if spec["type"]=="single": m=spec["model"]; return Q[m],E[m],pd.Series(False,index=Q.index)
    if spec["type"]=="cascade":
        f=spec["first"]; esc=~(C[f]>=spec["tau"]); return Q[f].where(~esc,Q[LARGE]),E[f]+esc*E[LARGE],esc
    if spec["type"]=="by_category":
        ms=cat.map(lambda c: spec["map"].get(c,LARGE))
        return pd.Series([Q.at[p,ms[p]] for p in Q.index],index=Q.index),pd.Series([E.at[p,ms[p]] for p in Q.index],index=Q.index),pd.Series(False,index=Q.index)
def fit(task,models,cascade):
    Q,E,C,_=tables(task,"dev",models); target=Q[LARGE].mean()-TOL[task]; ms=full(Q)
    cands=[{"type":"single","model":m} for m in ms]
    if cascade:
        for f in ms:
            if f==LARGE: continue
            for t in np.unique(np.quantile(C[f].dropna(),np.linspace(0,1,41))): cands.append({"type":"cascade","first":f,"tau":float(t)})
    best=None
    for s in cands:
        q,e,_=apply(s,Q,E,C)
        if q.mean()>=target-1e-12 and (best is None or e.mean()<best[1]-1e-9 or (abs(e.mean()-best[1])<1e-9 and q.mean()>best[2])): best=(s,e.mean(),q.mean())
    return best[0] if best else {"type":"single","model":LARGE}
def fit_cat(models):
    Q,E,C,cat=tables("factual_qa","dev",models); mp={}
    for c in sorted(cat.unique()):
        ids=cat.index[cat==c]; target=Q.loc[ids,LARGE].mean()-TOL["factual_qa"]
        ok=[m for m in Q.columns if Q.loc[ids,m].notna().all() and Q.loc[ids,m].mean()>=target-1e-12]
        mp[c]=min(ok,key=lambda m:E.loc[ids,m].mean())
    return {"type":"by_category","map":mp}
def evaluate(pol,split,tasks,models):
    per=[]
    for t in tasks:
        Q,E,C,cat=tables(t,split,models); q,e,esc=apply(pol[t],Q,E,C,cat)
        per.append(pd.DataFrame({"task":t,"quality":q,"energy":e,"esc":esc}))
    return pd.concat(per)
def oracle(split,tasks,models):
    per=[]
    for t in tasks:
        Q,E,_,_=tables(t,split,models); best=Q.max(axis=1)
        e=pd.Series([E.loc[p,[m for m in Q.columns if pd.notna(Q.at[p,m]) and Q.at[p,m]>=best[p]-1e-12]].min() for p in Q.index],index=Q.index)
        per.append(pd.DataFrame({"task":t,"quality":best,"energy":e,"esc":False}))
    return pd.concat(per)
V3M=["small_1.5B","large_3B","coder_1.5B"]; ALLM=V3M+["coder_3B","math_1.5B","summ_1.5B"]
rng=np.random.default_rng(42)
def run(label,tasks,models):
    pols={f"always {m}":{t:{"type":"single","model":m} for t in tasks} for m in ["small_1.5B","large_3B"]}
    pols["task-aware"]={t:fit(t,models,False) for t in tasks}
    pols["cascade"]={t:fit(t,models,True) for t in tasks}
    pc=dict(pols["task-aware"]); pc["factual_qa"]=fit_cat(models); pols["task-aware + QA by subject"]=pc
    res={k:evaluate(v,"test",tasks,models) for k,v in pols.items()}; res["oracle"]=oracle("test",tasks,models)
    base=res["always large_3B"]; rows=[]
    for k,per in res.items():
        row={"setting":label,"policy":k,"energy_kj":per.energy.sum()/1000,"saving_pct":100*(1-per.energy.sum()/base.energy.sum())}
        for t in tasks: row[f"q_{t}"]=per[per.task==t].quality.mean()
        if k not in("always large_3B",):
            sv=[]
            for _ in range(2000):
                idx=np.concatenate([rng.choice(np.where(per.task.values==t)[0],(per.task==t).sum()) for t in tasks])
                sv.append(100*(1-per.energy.values[idx].sum()/base.energy.values[idx].sum()))
            row["ci"]=f"[{np.percentile(sv,2.5):.1f}, {np.percentile(sv,97.5):.1f}]"
            for t in tasks:
                a=per[per.task==t].quality; b=base[base.task==t].quality
                if t=="summarization": p=stats.wilcoxon(a,b).pvalue if (a!=b).any() else 1.0
                else:
                    lo,hi=int(((a==1)&(b==0)).sum()),int(((a==0)&(b==1)).sum()); p=stats.binomtest(lo,lo+hi,.5).pvalue if lo+hi else 1.0
                row[f"p_{t}"]=p
        rows.append(row)
    return pd.DataFrame(rows),pols
pd.set_option('display.width',300); pd.set_option('display.max_columns',30)
out=[]; allpols={}
for label,tasks,models in [("v3 menu, 3 tasks",["factual_qa","summarization","coding"],V3M),
                           ("full menu, 3 tasks",["factual_qa","summarization","coding"],ALLM),
                           ("full menu, 4 tasks",["factual_qa","summarization","coding","math"],ALLM),
                           ("v3 menu, 4 tasks",["factual_qa","summarization","coding","math"],["small_1.5B","large_3B","coder_1.5B"])]:
    r,p=run(label,tasks,models); out.append(r); allpols[label]=p
R=pd.concat(out); print(R.round(3).to_string())
json.dump(allpols,open('/home/claude/v4res/router4_policies.json','w'),indent=1,default=str)
R.to_csv('/home/claude/v4res/router4_eval.csv',index=False)
for lab in allpols: print(lab, json.dumps({k:v for k,v in allpols[lab].items() if k not in('always small_1.5B','always large_3B')},default=str)[:900])
