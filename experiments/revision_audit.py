"""Independent post-hoc audit of records, checkpoints, graph IDs and metrics."""
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
from experiments.revision_run import ROOT,array,dump,SEEDS
from src.revision_core import brandes


def main():
    graphs={}
    for name in ['test_graphs.json','real_graphs.json']:
        graphs.update({r['id']:r for r in json.loads((ROOT/name).read_text())})
    summary={};audit_examples=[]
    for name,expected in [('baselines_v2.jsonl',1440),('rl.jsonl',3600),
                           ('real_baselines_v2.jsonl',126),('real_rl_cuda.jsonl',300),
                           ('budget.jsonl',1440)]:
        path=ROOT/name
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        # Real: 6 LCC targets*3 + 12 n80*5 + 12 n256*4 = 126.
        assert len(rows)==expected,(name,len(rows),expected)
        keys=[(r['id'],r['method'],r.get('seed',0),r.get('budget',5)) for r in rows]
        assert len(set(keys))==len(keys),(name,'duplicate records')
        for j,r in enumerate(rows):
            a=array(graphs[r['id']]); target=graphs[r['id']]['target']
            assert len(r['edges'])==r.get('budget',5)
            if 'feasible_exact_budget' in r:
                assert r['feasible_exact_budget'] and r['steps_used']==r['budget_requested']==r.get('budget',5)
            assert len(r['curve'])==len(r['edges'])+1
            assert abs(r['gain']-(r['final']-r['initial']))<1e-12
            assert abs(r['curve'][0]-r['initial'])<1e-12 and abs(r['curve'][-1]-r['final'])<1e-12
            if r['initial']>1e-12:assert abs(r['relative']-r['gain']/r['initial'])<1e-9
            else:assert r['relative'] is None
            for u,v in r['edges']:
                assert u!=v and a[u,v]==0
                a[u,v]=a[v,u]=1
            # All synthetic/budget outputs and all LCC outputs independently replayed.
            if name in ['baselines_v2.jsonl','rl.jsonl','budget.jsonl'] or graphs[r['id']]['n']>256:
                assert abs(float(brandes(a)[target])-r['final'])<1e-11,(name,r['id'])
            # Independent NetworkX path counts on a deterministic sample.
            if j%137==0 and len(a)<=256:
                b=nx.betweenness_centrality(nx.from_numpy_array(a))[target]
                assert abs(b-r['final'])<1e-11
                audit_examples.append(dict(file=name,id=r['id'],method=r['method'],networkx_final=b))
        summary[name]=dict(rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for variant in ['exact','no_bc','myopic']:
        for seed in SEEDS:
            d=json.loads((ROOT/f'{variant}_seed{seed}/complete.json').read_text())
            assert d['episodes']==5000
    dump(ROOT/'audit.json',dict(status='PASS',files=summary,networkx_crosschecks=audit_examples,
                               training_runs=15,training_episodes=75000))
    print('AUDIT PASS',summary,flush=True)


if __name__=='__main__':main()
