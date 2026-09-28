"""Follow-up experiments (incident-sampled controls, matched timing), fixed before measurement.

Run quality stages before timing; timing is single-process, warmed, one thread.
Original 15 checkpoints are retained without tuning or selection.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from src.revision_core import solve_baseline, solve_policy, stable_seed, brandes
from experiments.revision_run import ROOT, array, graph_record, load_model, dump

OUT=ROOT/'followup'
SEEDS=[101,202,303,404,505]
METHODS=['SampledIncidentGreedy20','SampledIncidentGreedy100']

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def prepare():
    OUT.mkdir(exist_ok=True)
    protocol=dict(version='post-review-v1',baseline_seeds=SEEDS,methods=METHODS,budget=5,
        synthetic_graphs=240,real_instances=30,selection='All existing held-out instances; no new model training or selection',
        timing=dict(families=['ER','BA','WS'],sizes=[50,100,256],graphs_per_family_size=5,
                    repetitions=3,threads=1,model_seed=11,order='deterministic shuffled per repetition',
                    audit_bc_excluded=True,compilation_and_loading_excluded=True),
        cpu='Intel Core i7-9750H @ 2.60 GHz',ram_bytes=17023488000,gpu='NVIDIA GeForce GTX 1650',
        python=platform.python_version(),torch=torch.__version__,
        scope='Post-hoc follow-up to reviewer-style audit, not a prospectively selected primary hypothesis')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:dump(OUT/'protocol.json',protocol)
    print('Prepared follow-up protocol',flush=True)

def quality(split, tail=False):
    torch.set_num_threads(1)
    records=json.loads((ROOT/('test_graphs.json' if split=='synthetic' else 'real_graphs.json')).read_text())
    if tail:
        records=[r for r in records if r['family']=='ca-GrQc' and (r['sampling']!='LCC' or r['id'].endswith('target2'))]
    path=OUT/('real_tail.jsonl' if tail else f'{split}.jsonl')
    done={(x['id'],x['method'],x['seed']) for x in rows(path)}
    with path.open('a',encoding='utf-8') as f:
        for i,r in enumerate(records):
            a=array(r)
            for method in METHODS:
                for seed in SEEDS:
                    if (r['id'],method,seed) in done:continue
                    result=solve_baseline(a,r['target'],5,method,stable_seed('followup',r['id'],method,seed))
                    assert result['feasible_exact_budget']
                    row=dict(id=r['id'],family=r['family'],n=r['n'],sampling=r.get('sampling'),method=method,seed=seed,**result)
                    f.write(json.dumps(row)+'\n');f.flush()
            print(split,i+1,len(records),r['id'],flush=True)

def timing():
    assert len(rows(OUT/'synthetic.jsonl'))==2400 and len(rows(OUT/'real.jsonl'))==300, 'Finish quality experiments before isolated timing'
    torch.set_num_threads(1)
    models={v:load_model(v,11) for v in ['exact','no_bc','myopic']}
    warm=array(graph_record('followup-warm','BA',2,20,0))
    for v,model in models.items():solve_policy(model,warm,19,5,v!='no_bc')
    solve_baseline(warm,19,5,'FullGreedy')
    records=[graph_record('followup-timing',fam,param,n,i)
             for fam,param in [('ER',.1),('BA',2),('WS',.1)] for n in [50,100,256] for i in range(5)]
    dump(OUT/'timing_graphs.json',records)
    path=OUT/'timing.jsonl'
    done={(x['id'],x['method'],x['repeat']) for x in rows(path)}
    with path.open('a',encoding='utf-8') as f:
        for i,r in enumerate(records):
            a=array(r)
            methods=['Degree','SampledGreedy20',*METHODS,'IncidentGreedy',*models]
            if r['n']<=100:methods.append('FullGreedy')
            for repeat in range(3):
                order=np.random.default_rng(stable_seed('timing-order',r['id'],repeat)).permutation(methods)
                for method in order:
                    if (r['id'],method,repeat) in done:continue
                    if method in models:result=solve_policy(models[method],a,r['target'],5,method!='no_bc')
                    else:result=solve_baseline(a,r['target'],5,method,stable_seed('timing-action',r['id'],method))
                    row=dict(id=r['id'],family=r['family'],n=r['n'],method=str(method),repeat=repeat,model_seed=11 if method in models else None,**result)
                    f.write(json.dumps(row)+'\n');f.flush()
            print('timing',i+1,len(records),r['id'],flush=True)

def repair_tiny():
    path=ROOT/'tiny_optima.json';records=json.loads(path.read_text())
    changed=[]
    for r in records:
        old=r['incident']
        new=solve_baseline(array(r),r['target'],2,'IncidentGreedy')
        if new['feasible_exact_budget']:assert abs(new['final']-old['final'])<1e-12
        else:changed.append(dict(id=r['id'],old_steps=len(old['edges']),old_prefix_bc=old['final'],optimum_exact_k=r['optimum']))
        r['incident']=new
    dump(path,records)
    dump(OUT/'budget_correction.json',dict(status='corrected',infeasible_instances=changed,policy='No partial rollout is reported as a feasible exactly-k solution',original_snapshot='output/pre_followup_snapshot.zip'))
    print('Budget correction:',len(changed),'infeasible tiny instances',flush=True)

def annotate_budgets():
    """Add derived feasibility metadata without changing any measured score/time."""
    for filename in ['baselines_v2.jsonl','rl.jsonl','real_baselines_v2.jsonl','real_rl_cuda.jsonl','budget.jsonl']:
        path=ROOT/filename;data=rows(path)
        for r in data:
            requested=r.get('budget',5)
            assert len(r['edges'])==requested,(filename,r['id'])
            r.update(budget_requested=requested,steps_used=len(r['edges']),feasible_exact_budget=True,status='complete')
        path.write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
    print('Added budget metadata; all original main/real/budget trajectories use the requested budget',flush=True)

def audit():
    import networkx as nx
    import hashlib
    result={}
    for split,gfile,expected in [('synthetic','test_graphs.json',2400),('real','real_graphs.json',300)]:
        graphs={r['id']:r for r in json.loads((ROOT/gfile).read_text())}
        data=rows(OUT/f'{split}.jsonl');assert len(data)==expected
        assert len({(r['id'],r['method'],r['seed']) for r in data})==expected
        checks=0
        for i,r in enumerate(data):
            g=graphs[r['id']];a=array(g)
            assert r['steps_used']==r['budget_requested']==5 and r['feasible_exact_budget']
            for u,v in r['edges']:
                assert g['target'] in [u,v] and a[u,v]==0 and u!=v
                a[u,v]=a[v,u]=1
            assert abs(float(brandes(a)[g['target']])-r['final'])<1e-10
            assert abs(r['gain']-(r['final']-r['initial']))<1e-12
            if i%97==0 and g['n']<=256:
                assert abs(nx.betweenness_centrality(nx.from_numpy_array(a))[g['target']]-r['final'])<1e-10
                checks+=1
        result[split]=dict(rows=len(data),networkx_checks=checks,replayed_all_finals=True)
    t=rows(OUT/'timing.jsonl');assert len(t)==1170
    assert len({(r['id'],r['method'],r['repeat']) for r in t})==1170
    tg={r['id']:r for r in json.loads((OUT/'timing_graphs.json').read_text())}
    for identifier,method in {(r['id'],r['method']) for r in t}:
        repeats=[r for r in t if r['id']==identifier and r['method']==method]
        assert len(repeats)==3
        assert len({json.dumps(r['edges']) for r in repeats})==1
        assert all(r['seconds']>0 and len(r['edges'])==5 for r in repeats)
        g=tg[identifier];a=array(g)
        for u,v in repeats[0]['edges']:
            assert u!=v and a[u,v]==0
            a[u,v]=a[v,u]=1
        checked=float(brandes(a)[g['target']])
        assert all(abs(checked-r['final'])<1e-10 for r in repeats)
    result['timing']=dict(rows=len(t),graphs=45,replayed_all_distinct_trajectories=True)
    tiny=json.loads((ROOT/'tiny_optima.json').read_text())
    bad=[r for r in tiny if not r['incident']['feasible_exact_budget']]
    assert len(bad)==6 and all(r['incident']['final'] is None for r in bad)
    result['files']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['synthetic.jsonl','real.jsonl','timing.jsonl','protocol.json','timing_graphs.json']}
    result['status']='PASS';dump(OUT/'audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','synthetic','real','real_tail','timing','repair_tiny','annotate_budgets','audit']);a=p.parse_args()
    if a.stage in ['synthetic','real']:quality(a.stage)
    elif a.stage=='real_tail':quality('real',tail=True)
    else:globals()[a.stage]()
