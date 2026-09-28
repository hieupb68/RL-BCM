"""Auditable revision implementation: exact BC, exhaustive baselines, batched PPO.

Simple undirected unweighted graphs; vertices are contiguous integers. The compiled
Brandes implementation changes execution speed, not the objective or precision.
"""
import hashlib
import time
import itertools
import numpy as np
import networkx as nx
import torch
from torch import nn
from numba import njit


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], 'little')


@njit(cache=True)
def brandes(a):
    n = len(a)
    degree = a.sum(axis=1).astype(np.int64)
    offsets = np.zeros(n + 1, np.int64)
    for v in range(n):
        offsets[v + 1] = offsets[v] + degree[v]
    neighbors = np.empty(offsets[-1], np.int64)
    for v in range(n):
        p = offsets[v]
        for w in range(n):
            if a[v, w]:
                neighbors[p] = w
                p += 1
    bc = np.zeros(n)
    queue = np.empty(n, np.int64)
    for source in range(n):
        dist = np.full(n, -1, np.int64)
        sigma = np.zeros(n)
        delta = np.zeros(n)
        dist[source] = 0
        sigma[source] = 1
        queue[0] = source
        tail = 1
        head = 0
        while head < tail:
            v = queue[head]
            head += 1
            for p in range(offsets[v], offsets[v + 1]):
                w = neighbors[p]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue[tail] = w
                    tail += 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        for index in range(tail - 1, -1, -1):
            w = queue[index]
            for p in range(offsets[w], offsets[w + 1]):
                v = neighbors[p]
                if dist[v] == dist[w] - 1:
                    delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != source:
                bc[w] += delta[w]
    if n > 2:
        bc /= (n - 1) * (n - 2)
    return bc


@njit(cache=True)
def features(a, target, bc, remaining, use_bc=True):
    n = len(a)
    x = np.zeros((n, 6), np.float32)
    degree = a.sum(axis=1)
    dist = np.full(n, -1, np.int64)
    queue = np.empty(n, np.int64)
    queue[0] = target
    dist[target] = 0
    tail = 1
    head = 0
    while head < tail:
        v = queue[head]
        head += 1
        for w in range(n):
            if a[v, w] and dist[w] < 0:
                dist[w] = dist[v] + 1
                queue[tail] = w
                tail += 1
    maxdist = dist.max() + 1
    maxdeg = max(1, degree.max())
    for v in range(n):
        x[v, 0] = degree[v] / maxdeg
        x[v, 1] = 1.0 if v == target else 0.0
        if use_bc:
            x[v, 2] = bc[v]
        neighbors = np.where(a[v] > 0)[0]
        triangles = 0
        for j in range(len(neighbors)):
            for k in range(j):
                triangles += a[neighbors[j], neighbors[k]]
        if degree[v] > 1:
            x[v, 3] = 2 * triangles / (degree[v] * (degree[v] - 1))
        if dist[v] >= 0:
            x[v, 4] = 1 - dist[v] / maxdist
        x[v, 5] = remaining
    return x


@njit(cache=True)
def candidate_values(a, target, candidates):
    """Reference candidate scorer: recompute exact Brandes for every edge."""
    values = np.empty(len(candidates))
    for i in range(len(candidates)):
        u, v = candidates[i]
        a[u, v] = a[v, u] = 1
        values[i] = brandes(a)[target]
        a[u, v] = a[v, u] = 0
    return values


@njit(cache=True)
def shortest_path_counts(a):
    n=len(a)
    distances=np.full((n,n),n+1,np.int64)
    counts=np.zeros((n,n))
    queue=np.empty(n,np.int64)
    degree=a.sum(axis=1).astype(np.int64)
    offsets=np.zeros(n+1,np.int64)
    for v in range(n): offsets[v+1]=offsets[v]+degree[v]
    neighbors=np.empty(offsets[-1],np.int64)
    for v in range(n):
        k=offsets[v]
        for w in range(n):
            if a[v,w]: neighbors[k]=w; k+=1
    for s in range(n):
        distances[s,s]=0; counts[s,s]=1.; queue[0]=s
        head=0;tail=1
        while head<tail:
            v=queue[head];head+=1
            for p in range(offsets[v],offsets[v+1]):
                w=neighbors[p]
                if distances[s,w]==n+1:
                    distances[s,w]=distances[s,v]+1;queue[tail]=w;tail+=1
                if distances[s,w]==distances[s,v]+1: counts[s,w]+=counts[s,v]
    return distances,counts


@njit(cache=True)
def updated_pair(d,c,s,t,u,v):
    old=d[s,t];forward=d[s,u]+1+d[v,t];backward=d[s,v]+1+d[u,t]
    new=min(old,forward,backward)
    sigma=0.
    if old==new: sigma+=c[s,t]
    if forward==new: sigma+=c[s,u]*c[v,t]
    if backward==new: sigma+=c[s,v]*c[u,t]
    return new,sigma


@njit(cache=True)
def incremental_values(a,target,candidates):
    """Exact insertion formula, O(n*m + q*n^2) for q candidate edges.

    A new shortest simple path either avoids the new edge, traverses u->v,
    or v->u. Count each case attaining the new minimum distance. A repeated
    vertex would imply a shorter walk, so cannot occur in a minimizing case.
    All-pairs distances/counts are rebuilt after each committed modification.
    """
    n=len(a);d,c=shortest_path_counts(a)
    result=np.zeros(len(candidates))
    for index in range(len(candidates)):
        u,v=candidates[index]
        dt=np.empty(n,np.int64);ct=np.empty(n)
        for s in range(n): dt[s],ct[s]=updated_pair(d,c,s,target,u,v)
        total=0.
        for s in range(n):
            if s==target: continue
            for t in range(s+1,n):
                if t==target: continue
                distance,count=updated_pair(d,c,s,t,u,v)
                if count>0 and dt[s]+dt[t]==distance:
                    total+=ct[s]*ct[t]/count
        if n>2: result[index]=2*total/((n-1)*(n-2))
    return result


def nonedges(a):
    return np.column_stack(np.where(np.triu(1 - a, 1))).astype(np.int64)


def incident_nonedges(a, target):
    """Lexicographic incident candidates without materializing all graph pairs."""
    vertices = np.flatnonzero((a[target] == 0) & (np.arange(len(a)) != target))
    pairs = np.column_stack((np.minimum(vertices, target), np.maximum(vertices, target)))
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int64)


def solve_baseline(a, target, budget, method, seed=0):
    start = time.perf_counter()
    a = a.copy()
    audit_start = time.perf_counter()
    initial = float(brandes(a)[target])
    audit_seconds = time.perf_counter()-audit_start
    incident_only = method == 'IncidentGreedy' or method.startswith('SampledIncidentGreedy')
    available = len(incident_nonedges(a, target)) if incident_only else len(nonedges(a))
    if available < budget:
        # A prefix is not a feasible solution of the exactly-k problem.
        return dict(initial=initial, final=None, gain=None, relative=None,
                    last_prefix_bc=initial, edges=[], curve=[initial], steps_used=0,
                    budget_requested=budget, feasible_exact_budget=False,
                    status='infeasible_exact_budget', available_actions=available,
                    bc_calls=1, brandes_calls=1, apsp_builds=0,
                    seconds=time.perf_counter()-start-audit_seconds,
                    audit_seconds=audit_seconds)
    curve, edges, calls = [initial], [], 1
    rng = np.random.default_rng(seed)
    for step in range(budget):
        candidates = incident_nonedges(a, target) if incident_only else nonedges(a)
        if len(candidates) == 0:
            break
        if method == 'Random':
            edge = candidates[rng.integers(len(candidates))]
        elif method == 'Degree':
            incident = candidates[(candidates == target).any(axis=1)]
            pool = incident if len(incident) else candidates
            degree = a.sum(axis=1)
            edge = pool[np.argmax(degree[pool].sum(axis=1))]
        else:
            if method.startswith('SampledIncidentGreedy'):
                cap = int(method.replace('SampledIncidentGreedy', ''))
                candidates = candidates[np.sort(rng.choice(len(candidates), min(cap, len(candidates)), replace=False))]
            elif method == 'IncidentGreedy':
                pass
            elif method.startswith('SampledGreedy'):
                cap = int(method.replace('SampledGreedy', ''))
                candidates = candidates[np.sort(rng.choice(len(candidates), min(cap, len(candidates)), replace=False))]
            elif method != 'FullGreedy':
                raise ValueError(method)
            if not len(candidates):
                break
            values = incremental_values(a, target, candidates)
            calls += len(candidates)
            # Lexicographic first tie; all marginals recomputed on current graph.
            # Values within numerical precision are tied lexicographically.
            edge = candidates[np.where(values >= values.max()-1e-12)[0][0]]
        u, v = map(int, edge)
        a[u, v] = a[v, u] = 1
        edges.append([u, v])
        audit_start = time.perf_counter()
        curve.append(float(brandes(a)[target]))
        audit_seconds += time.perf_counter()-audit_start
        calls += 1
    assert len(edges) == budget, 'Feasible fixed-budget baseline terminated early'
    return dict(initial=initial, final=curve[-1], gain=curve[-1]-initial,
                relative=(curve[-1]-initial)/initial if initial > 1e-12 else None,
                steps_used=len(edges), budget_requested=budget, feasible_exact_budget=True,
                status='complete',
                edges=edges, curve=curve, bc_calls=calls,
                oracle_backend='exact insertion counts v1; tie tolerance 1e-12',
                brandes_calls=1+len(edges),
                apsp_builds=len(edges) if method not in ['Random','Degree'] else 0,
                seconds=time.perf_counter()-start-audit_seconds,
                audit_seconds=audit_seconds,
                timing_scope='decisions excluding initial/trajectory audit BC')


class Policy(nn.Module):
    """Mean GraphSAGE; symmetric unordered edge features; no dropout in PPO."""
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        dims = [6] + [hidden] * layers
        self.self_layers = nn.ModuleList([nn.Linear(dims[i], hidden) for i in range(layers)])
        self.neigh_layers = nn.ModuleList([nn.Linear(dims[i], hidden, bias=False) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.scorer = nn.Sequential(nn.Linear(3*hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.value = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, a):
        norm = a / a.sum(-1, keepdim=True).clamp_min(1)
        h = x
        for i, (sl, nl, ln) in enumerate(zip(self.self_layers, self.neigh_layers, self.norms)):
            h = ln(sl(h) + nl(torch.bmm(norm, h)))
            if i < len(self.norms)-1:
                h = torch.relu(h)
        return h

    def score(self, h, target, pairs):
        b = h.shape[0]
        ht = h[torch.arange(b, device=h.device), target]
        u, v = pairs
        z = torch.cat((h[:, u]+h[:, v], torch.abs(h[:, u]-h[:, v]), ht[:, None].expand(-1, len(u), -1)), -1)
        return self.scorer(z).squeeze(-1)

    def forward(self, x, a, target):
        h = self.encode(x, a)
        pairs = torch.triu_indices(a.shape[-1], a.shape[-1], 1, device=a.device)
        logits = self.score(h, target, pairs).masked_fill(a[:, pairs[0], pairs[1]] > 0, -1e9)
        ht = h[torch.arange(len(h), device=h.device), target]
        value = self.value(torch.cat((h.mean(1), ht), -1)).squeeze(-1)
        return logits, value


def solve_policy(model, a, target, budget, use_bc=True, device='cpu', chunk=8192):
    a = a.copy()
    audit_start = time.perf_counter()
    initial = float(brandes(a)[target])
    initial_audit_seconds = time.perf_counter()-audit_start
    start = time.perf_counter()
    curve, edges = [initial], []
    bc_time = 0.
    feature_time = 0.
    policy_time = 0.
    # Deployment excludes reward oracle: no-BC needs only initial/final auditing.
    with torch.no_grad():
        for step in range(budget):
            t = time.perf_counter()
            bc = brandes(a) if use_bc else np.zeros(len(a))
            bc_time += time.perf_counter()-t
            t = time.perf_counter()
            x = features(a, target, bc, (budget-step)/budget, use_bc)
            candidates = nonedges(a)
            feature_time += time.perf_counter()-t
            if not len(candidates):
                break
            t = time.perf_counter()
            at = torch.as_tensor(a, dtype=torch.float32, device=device)[None]
            xt = torch.as_tensor(x, device=device)[None]
            tt = torch.tensor([target], device=device)
            h = model.encode(xt, at)
            best_score, best_index = -float('inf'), 0
            for lo in range(0, len(candidates), chunk):
                pairs = torch.as_tensor(candidates[lo:lo+chunk].T.copy(), device=device)
                scores = model.score(h, tt, pairs)[0]
                ix = int(scores.argmax())
                score = float(scores[ix])
                if score > best_score:
                    best_score, best_index = score, lo+ix
            policy_time += time.perf_counter()-t
            u, v = map(int, candidates[best_index])
            a[u, v] = a[v, u] = 1
            edges.append([u, v])
    decision_seconds = time.perf_counter()-start
    # Separate trajectory audit, never fed to no-BC decisions.
    audit_start = time.perf_counter()
    for u, v in edges:
        a[u, v] = a[v, u] = 0
    for u, v in edges:
        a[u, v] = a[v, u] = 1
        curve.append(float(brandes(a)[target]))
    return dict(initial=initial, final=curve[-1], gain=curve[-1]-initial,
                relative=(curve[-1]-initial)/initial if initial > 1e-12 else None,
                edges=edges, curve=curve, seconds=decision_seconds,
                audit_seconds=initial_audit_seconds+time.perf_counter()-audit_start,
                bc_feature_seconds=bc_time, feature_seconds=feature_time,
                policy_seconds=policy_time, bc_calls=int(use_bc)*len(edges),
                audit_bc_calls=1+len(edges))
