"""Tiny optimum and larger real-network experiments (no synthetic measurements)."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import networkx as nx
import torch
from src.revision_core import brandes, nonedges, candidate_values, solve_baseline, solve_policy, stable_seed
from experiments.revision_run import ROOT, SEEDS, dump, array, graph_record, load_model


def tiny():
    records=[]
    for family,param in [('ER',.3),('BA',2),('WS',.2)]:
        for i in range(20):
            r=graph_record('tiny',family,param,8,i)
            a=array(r); target=r['target']
            candidates=nonedges(a)
            optimum=-1.; best=[]; count=0
            for edges in itertools.combinations(candidates,2):
                h=a.copy()
                for u,v in edges: h[u,v]=h[v,u]=1
                bc=float(brandes(h)[target]); count+=1
                if bc > optimum:
                    optimum=bc; best=[list(map(int,e)) for e in edges]
            baseline=solve_baseline(a,target,2,'FullGreedy')
            incident=solve_baseline(a,target,2,'IncidentGreedy')
            records.append(dict(**r,optimum=optimum,optimal_edges=best,
                                combinations=count,greedy=baseline,incident=incident))
    dump(ROOT/'tiny_optima.json',records)
    print('tiny exact optimum',len(records),'greedy suboptimal',sum(r['greedy']['final']<r['optimum']-1e-12 for r in records),flush=True)


def prepare_real():
    records=[]; summaries=[]
    for dataset in ['email-Eu-core','ca-GrQc']:
        source=Path('data')/(dataset+'.txt')
        g=nx.read_edgelist(source,comments='#',nodetype=int,create_using=nx.Graph)
        raw_n,raw_m=g.number_of_nodes(),g.number_of_edges()
        g.remove_edges_from(list(nx.selfloop_edges(g)))
        g=g.subgraph(max(nx.connected_components(g),key=len)).copy()
        g=nx.convert_node_labels_to_integers(g,ordering='sorted',label_attribute='original_id')
        groups=[('LCC',g,0)]
        for sampling in ['BFS','RW']:
            for n in [80,256]:
                for i in range(3):
                    rng=np.random.default_rng(stable_seed('real',dataset,sampling,n,i))
                    root=int(rng.integers(len(g)))
                    if sampling=='BFS':
                        vertices=list(itertools.islice(nx.bfs_tree(g,root).nodes(),n))
                    else:
                        vertices={root}; current=root
                        while len(vertices)<n:
                            current=int(rng.choice(list(g.neighbors(current))))
                            vertices.add(current)
                        vertices=sorted(vertices)
                    h=nx.convert_node_labels_to_integers(g.subgraph(vertices).copy(),ordering='sorted')
                    groups.append((sampling,h,i))
        for sampling,h,i in groups:
            a=nx.to_numpy_array(h,nodelist=range(len(h)),dtype=np.uint8)
            bc=brandes(a)
            positive=np.where(bc>1e-12)[0]
            order=positive[np.argsort(bc[positive],kind='stable')]
            rng=np.random.default_rng(stable_seed('target',dataset,sampling,len(h),i))
            targets=rng.choice(order[:max(1,len(order)//4)],3 if sampling=='LCC' else 1,replace=False)
            summaries.append(dict(dataset=dataset,sampling=sampling,index=i,n=len(h),m=h.number_of_edges(),
                                  density=nx.density(h),clustering=nx.average_clustering(h),
                                  transitivity=nx.transitivity(h),mean_degree=2*h.number_of_edges()/len(h),
                                  raw_undirected_n=raw_n,raw_undirected_m=raw_m,
                                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            for j,target in enumerate(targets):
                records.append(dict(id=f'{dataset}_{sampling}_{len(h)}_{i}_target{j}',
                    family=dataset,sampling=sampling,n=len(h),target=int(target),
                    original_node_ids=[int(h.nodes[v]['original_id']) for v in range(len(h))],
                    edges=[list(map(int,e)) for e in h.edges()],
                    graph_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    dump(ROOT/'real_graphs.json',records)
    dump(ROOT/'real_structure.json',summaries)
    print('real instances',len(records),flush=True)


def run_real(mode, device='cpu'):
    torch.set_num_threads(1)
    records=json.loads((ROOT/'real_graphs.json').read_text())
    path=ROOT/('real_baselines_v2.jsonl' if mode=='baselines' else f'real_rl_{device}.jsonl')
    done=set()
    if path.exists(): done={(r['id'],r['method'],r.get('seed',0)) for r in map(json.loads,path.read_text().splitlines())}
    with path.open('a',encoding='utf-8') as out:
        if mode=='baselines':
            for r in records:
                methods=['Random','Degree','SampledGreedy20']
                if r['n']<=256: methods+=['IncidentGreedy']
                if r['n']==80: methods+=['FullGreedy']
                for method in methods:
                    if (r['id'],method,0) in done: continue
                    result=solve_baseline(array(r),r['target'],5,method,stable_seed(r['id'],method))
                    out.write(json.dumps(dict(id=r['id'],family=r['family'],sampling=r['sampling'],n=r['n'],method=method,**result))+'\n'); out.flush()
                print('real baseline',r['id'],flush=True)
        else:
            for variant in ['exact','no_bc']:
                for seed in SEEDS:
                    model=load_model(variant,seed).to(device)
                    for r in records:
                        if (r['id'],variant,seed) in done: continue
                        result=solve_policy(model,array(r),r['target'],5,variant!='no_bc',device=device,chunk=65536 if device=='cuda' else 8192)
                        out.write(json.dumps(dict(id=r['id'],family=r['family'],sampling=r['sampling'],n=r['n'],method=variant,seed=seed,device=device,**result))+'\n'); out.flush()
                        print('real rl',variant,seed,r['id'],round(result['seconds'],2),flush=True)


def tiny_rl():
    torch.set_num_threads(1)
    records=json.loads((ROOT/'tiny_optima.json').read_text())
    rows=[]
    for variant in ['exact','no_bc','myopic']:
        for seed in SEEDS:
            model=load_model(variant,seed)
            for r in records:
                result=solve_policy(model,array(r),r['target'],2,variant!='no_bc')
                rows.append(dict(id=r['id'],family=r['family'],method=variant,seed=seed,optimum=r['optimum'],**result))
    dump(ROOT/'tiny_rl.json',rows)


def budget():
    torch.set_num_threads(1)
    records=json.loads((ROOT/'test_graphs.json').read_text())
    records=[r for r in records if r['n']==50]
    path=ROOT/'budget.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for k in [1,3,10]:
            for r in records:
                for method in ['Degree','IncidentGreedy','FullGreedy']:
                    result=solve_baseline(array(r),r['target'],k,method,stable_seed(r['id'],method))
                    f.write(json.dumps(dict(id=r['id'],family=r['family'],budget=k,method=method,**result))+'\n');f.flush()
            for seed in SEEDS:
                model=load_model('exact',seed)
                for r in records:
                    result=solve_policy(model,array(r),r['target'],k,True)
                    f.write(json.dumps(dict(id=r['id'],family=r['family'],budget=k,method='exact',seed=seed,**result))+'\n');f.flush()
            print('budget',k,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tiny','prepare_real','real_baselines','real_rl','tiny_rl','budget'])
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=p.parse_args()
    if args.stage=='tiny': tiny()
    elif args.stage=='prepare_real': prepare_real()
    elif args.stage=='real_baselines': run_real('baselines')
    elif args.stage=='real_rl': run_real('rl',args.device)
    elif args.stage=='tiny_rl': tiny_rl()
    else: budget()
