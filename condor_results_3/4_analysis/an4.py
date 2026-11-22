import pandas as pd, numpy as np
from scipy.stats import wilcoxon, binomtest
pd.set_option('display.width',250)
v3=pd.read_csv('v3res/results/infer_results_graded.csv',dtype={'prompt_id':str})
v4=pd.read_csv('v4res/results_v4/infer_results_v4_graded.csv',dtype={'prompt_id':str})
E='gpu_energy_avgp_latency_j'
def mc(a,b):
    a=np.asarray(a);b=np.asarray(b); n01=int(((a==0)&(b==1)).sum()); n10=int(((a==1)&(b==0)).sum())
    return binomtest(n01,n01+n10,0.5).pvalue if n01+n10 else 1.0
def piv(d,task,col='quality_score'):
    return d[d.task==task].pivot(index='prompt_id',columns='model_name',values=col)
