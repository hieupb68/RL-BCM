"""Warmed single-process CPU quality/latency benchmark; run after training."""
import json
import time
import platform
import numpy as np
import torch
from src.revision_core import solve_policy,solve_baseline,brandes,features,incremental_values,nonedges
from experiments.revision_run import ROOT,array,graph_record,load_model,dump


def main():
    torch.set_num_threads(1)
    models={v:load_model(v,11) for v in ['exact','no_bc']}
    warm=array(graph_record('timingwarm','BA',2,20,0))
    for v in models: solve_policy(models[v],warm,0,5,v=='exact')
    solve_baseline(warm,0,5,'FullGreedy')
    records=[]
    for n in [50,100,256]:
        for i in range(3):
            r=graph_record('timing','BA',2,n,i);a=array(r)
            methods=['Degree','SampledGreedy20','IncidentGreedy','exact','no_bc']
            if n<=100:methods.insert(3,'FullGreedy')
            for repeat in range(3):
                for method in methods:
                    if method in models: result=solve_policy(models[method],a,r['target'],5,method=='exact')
                    else:result=solve_baseline(a,r['target'],5,method,92831+i)
                    records.append(dict(id=r['id'],n=n,repeat=repeat,method=method,**result))
            print('timed',n,i,flush=True)
    dump(ROOT/'timing.json',records)
    dump(ROOT/'timing_metadata.json',dict(device='cpu',threads=1,model_seed=11,
          repeats=3,graphs_per_size=3,compiled_warmup=True,model_loading_timed=False,
          processor=platform.processor(),torch=torch.__version__))


if __name__=='__main__':main()
