"""Reproducible revision experiments. Run from repository root as a module."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import hashlib
import numpy as np
import networkx as nx
import torch
from src.revision_core import (Policy, brandes, features, stable_seed,
                               solve_baseline, solve_policy)

ROOT = Path('results/revision')
CONFIGS = [('ER', .10), ('ER', .20), ('BA', 2), ('BA', 3), ('WS', .10), ('WS', .20)]
SEEDS = [11, 22, 33, 44, 55]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def graph_record(split, family, param, n, index):
    seed = stable_seed('revision-v1', split, family, param, n, index)
    rng = np.random.default_rng(seed)
    while True:
        s = int(rng.integers(2**31))
        if family == 'ER':
            g = nx.gnp_random_graph(n, param, seed=s)
        elif family == 'BA':
            g = nx.barabasi_albert_graph(n, int(param), seed=s)
        else:
            g = nx.watts_strogatz_graph(n, 4 if param == .1 else 6, param, seed=s)
        if nx.is_connected(g):
            break
    a = nx.to_numpy_array(g, dtype=np.uint8)
    bc = brandes(a)
    nz = np.where(bc > 1e-12)[0]
    order = nz[np.argsort(bc[nz], kind='stable')]
    target = int(rng.choice(order[:max(1, len(order)//4)])) if len(order) else int(rng.integers(n))
    return dict(id=f'{split}_{family}_{param}_{n}_{index}', split=split, family=family,
                param=param, n=n, seed=seed, target=target,
                edges=[list(map(int, e)) for e in sorted(g.edges())],
                graph_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def array(rec):
    a = np.zeros((rec['n'], rec['n']), dtype=np.uint8)
    for u, v in rec['edges']:
        a[u, v] = a[v, u] = 1
    return a


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for split, sizes, count in [('train', [20,30,40,50], 20),
                                ('validation', [30,50], 3),
                                ('test', [30,50,80,100], 10)]:
        records = [graph_record(split, family, param, n, i)
                   for family, param in CONFIGS for n in sizes for i in range(count)]
        all_records[split] = records
        dump(ROOT/f'{split}_graphs.json', records)
    hashes = [{x['graph_sha256'] for x in all_records[s]} for s in all_records]
    assert not any(hashes[i] & hashes[j] for i in range(3) for j in range(i))
    dump(ROOT/'protocol.json', dict(version='revision-v1', seeds=SEEDS, episodes=5000,
         budget=5, hidden=64, layers=3, batch_episodes=16, ppo_epochs=4,
         minibatch=32, lr=3e-4, gamma=1.0, gae_lambda=.95, clip=.2,
         entropy=.01, reward='100 * absolute BC increment',
         model_selection='last checkpoint; validation logged, never selected on test',
         primary='absolute final normalized BC',
         variants={'exact': {'use_bc': True, 'gamma': 1.0},
                   'no_bc': {'use_bc': False, 'gamma': 1.0},
                   'myopic': {'use_bc': True, 'gamma': 0.0}},
         python=platform.python_version(), torch=torch.__version__,
         networkx=nx.__version__, cpu=platform.processor(),
         hardware_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         data_counts={k:len(v) for k,v in all_records.items()}))
    print('Prepared', {k: len(v) for k,v in all_records.items()}, flush=True)


def train(seed, variant, episodes=5000, device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    use_bc, gamma = variant != 'no_bc', 0. if variant == 'myopic' else 1.
    out = ROOT/f'{variant}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    config = dict(seed=seed, variant=variant, episodes=episodes, device=device,
                  use_bc=use_bc, gamma=gamma)
    dump(out/'config.json', config)
    records = json.loads((ROOT/'train_graphs.json').read_text())
    by_size = {n: [r for r in records if r['n']==n] for n in [20,30,40,50]}
    cached = {r['id']: array(r) for r in records}
    val = json.loads((ROOT/'validation_graphs.json').read_text())
    model = Policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    start = time.perf_counter()
    history, completed = [], 0
    while completed < episodes:
        b = min(16, episodes-completed)
        n = int(rng.choice([20,30,40,50]))
        batch = [by_size[n][int(i)] for i in rng.integers(len(by_size[n]), size=b)]
        aa = np.stack([cached[r['id']].copy() for r in batch])
        targets = np.array([r['target'] for r in batch])
        tt = torch.tensor(targets, device=device)
        bc = np.stack([brandes(a) for a in aa])
        initial = bc[np.arange(b), targets].copy()
        xs, ads, acts, lps, vals, rewards = [], [], [], [], [], []
        pairs = torch.triu_indices(n,n,1).numpy()
        for step in range(5):
            xx = np.stack([features(a, int(t), c, (5-step)/5, use_bc) for a,t,c in zip(aa,targets,bc)])
            xt = torch.tensor(xx, device=device)
            at = torch.tensor(aa, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(xt, at, tt)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            xs.append(xt); ads.append(at); acts.append(action); lps.append(logp); vals.append(value)
            old = bc[np.arange(b), targets].copy()
            for j, ai in enumerate(action.cpu().numpy()):
                u,v = pairs[:,ai]
                assert aa[j,u,v] == 0
                aa[j,u,v] = aa[j,v,u] = 1
            bc = np.stack([brandes(a) for a in aa])
            rewards.append(torch.tensor(100*(bc[np.arange(b), targets]-old), dtype=torch.float32, device=device))
        values = torch.stack(vals)
        advantages = torch.zeros_like(values)
        gae = torch.zeros(b, device=device)
        for t in range(4,-1,-1):
            nextval = values[t+1] if t < 4 else torch.zeros(b, device=device)
            delta = rewards[t] + gamma*nextval-values[t]
            gae = delta + gamma*.95*gae
            advantages[t] = gae
        returns = (advantages+values).flatten()
        adv = advantages.flatten()
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        xx, at = torch.cat(xs), torch.cat(ads)
        action, oldlp = torch.cat(acts), torch.cat(lps)
        targett = tt.repeat(5)
        losses = []
        for epoch in range(4):
            perm = torch.randperm(b*5, device=device)
            for ix in perm.split(32):
                logits, value = model(xx[ix], at[ix], targett[ix])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(action[ix])-oldlp[ix]).exp()
                pl = -torch.minimum(ratio*adv[ix], ratio.clamp(.8,1.2)*adv[ix]).mean()
                vl = (value-returns[ix]).square().mean()
                loss = pl+.5*vl-.01*dist.entropy().mean()
                assert torch.isfinite(loss), 'Non-finite PPO loss'
                optimizer.zero_grad()
                loss.backward()
                nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
                losses.append(float(loss.detach()))
        completed += b
        row = dict(episode=completed, train_gain=float(np.mean(bc[np.arange(b), targets]-initial)),
                   loss=float(np.mean(losses)), elapsed=time.perf_counter()-start)
        if completed % 512 == 0 or completed == episodes:
            scores = [solve_policy(model,array(r),r['target'],5,use_bc,device)['final'] for r in val]
            row['validation_final'] = float(np.mean(scores))
            print(variant,seed,completed,row,flush=True)
            torch.save(dict(state=model.cpu().state_dict(), config=config, episode=completed),out/'model.pt')
            model.to(device)
        history.append(row)
        if completed % 128 == 0 or completed == episodes:
            dump(out/'history.json',history)
    dump(out/'complete.json',dict(**config, elapsed=time.perf_counter()-start, updates=len(history)))


def load_model(variant, seed):
    ckpt = torch.load(ROOT/f'{variant}_seed{seed}/model.pt', map_location='cpu', weights_only=False)
    model = Policy()
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model


def evaluate_baselines():
    records = json.loads((ROOT/'test_graphs.json').read_text())
    path = ROOT/'baselines_v2.jsonl'
    done = set()
    if path.exists():
        done = {(r['id'],r['method']) for r in map(json.loads,path.read_text().splitlines())}
    methods = ['Random','Degree','SampledGreedy20','SampledGreedy100','IncidentGreedy','FullGreedy']
    with path.open('a',encoding='utf-8') as f:
        for i,r in enumerate(records):
            for method in methods:
                if (r['id'],method) in done:
                    continue
                result = solve_baseline(array(r),r['target'],5,method,stable_seed(r['id'],method))
                f.write(json.dumps(dict(id=r['id'],family=r['family'],n=r['n'],method=method,**result))+'\n')
                f.flush()
            if i%10 == 0:
                print('baseline',i,len(records),r['id'],flush=True)


def evaluate_rl(variants):
    torch.set_num_threads(1)
    records = json.loads((ROOT/'test_graphs.json').read_text())
    path = ROOT/'rl.jsonl'
    done = set()
    if path.exists():
        done = {(r['id'],r['method'],r['seed']) for r in map(json.loads,path.read_text().splitlines())}
    with path.open('a',encoding='utf-8') as f:
        for variant in variants:
            for seed in SEEDS:
                model = load_model(variant,seed)
                for r in records:
                    if (r['id'],variant,seed) in done:
                        continue
                    result = solve_policy(model,array(r),r['target'],5,variant!='no_bc')
                    f.write(json.dumps(dict(id=r['id'],family=r['family'],n=r['n'],method=variant,seed=seed,**result))+'\n')
                    f.flush()
                print('evaluated',variant,seed,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','train','baselines','evaluate'])
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--variant',default='exact',choices=['exact','no_bc','myopic'])
    p.add_argument('--variants',nargs='+',default=['exact','no_bc','myopic'])
    p.add_argument('--episodes',type=int,default=5000)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='train':
        directory=ROOT/f'{args.variant}_seed{args.seed}'
        directory.mkdir(parents=True,exist_ok=True)
        lock=directory/'running.lock'
        if (directory/'complete.json').exists():
            print('Already complete',directory,flush=True)
        else:
            try:
                handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError:
                print('Another process owns this run',directory,flush=True)
            else:
                os.close(handle)
                try: train(args.seed,args.variant,args.episodes,args.device)
                finally: lock.unlink()
    elif args.stage=='baselines': evaluate_baselines()
    else: evaluate_rl(args.variants)
