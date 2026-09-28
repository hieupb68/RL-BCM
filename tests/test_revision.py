import itertools
import unittest
import numpy as np
import networkx as nx
import torch
from src.revision_core import brandes, candidate_values, incremental_values, nonedges, incident_nonedges, solve_baseline, Policy, features


class RevisionTests(unittest.TestCase):
    def test_exact_budget_infeasible_is_not_a_prefix_solution(self):
        a=nx.to_numpy_array(nx.path_graph(4),dtype=np.uint8)
        for method in ['IncidentGreedy','SampledIncidentGreedy20','SampledIncidentGreedy100']:
            r=solve_baseline(a,1,2,method)
            self.assertFalse(r['feasible_exact_budget'])
            self.assertIsNone(r['final'])
            self.assertEqual(r['status'],'infeasible_exact_budget')
            self.assertEqual(r['steps_used'],0)
        r=solve_baseline(a,1,2,'FullGreedy')
        self.assertTrue(r['feasible_exact_budget'])
        self.assertEqual(r['steps_used'],2)

    def test_incident_sampling_and_candidate_order(self):
        g=nx.barabasi_albert_graph(30,2,seed=31)
        a=nx.to_numpy_array(g,dtype=np.uint8)
        for target in [0,12,29]:
            allpairs=nonedges(a)
            expected=allpairs[(allpairs==target).any(axis=1)]
            np.testing.assert_array_equal(incident_nonedges(a,target),expected)
            full=solve_baseline(a,target,5,'IncidentGreedy')
            sampled=solve_baseline(a,target,5,'SampledIncidentGreedy100',4321)
            self.assertEqual(full['edges'],sampled['edges'])
            small=solve_baseline(a,target,5,'SampledIncidentGreedy20',4321)
            self.assertEqual(small['steps_used'],5)
            self.assertTrue(all(target in e for e in small['edges']))
            h=g.copy();h.add_edges_from(small['edges'])
            self.assertAlmostEqual(small['final'],nx.betweenness_centrality(h)[target],places=12)

    def test_incremental_candidate_oracle(self):
        for n in [4,9,20,50]:
            for p in [.05,.2,.6]:
                for seed in range(3):
                    a=nx.to_numpy_array(nx.gnp_random_graph(n,p,seed=seed),dtype=np.uint8)
                    candidates=nonedges(a)
                    for target in [0,n-1]:
                        np.testing.assert_allclose(incremental_values(a,target,candidates),
                                                   candidate_values(a,target,candidates),atol=1e-12,rtol=1e-12)

    def test_bc_against_networkx(self):
        graphs = [nx.path_graph(3), nx.star_graph(9), nx.complete_graph(8), nx.empty_graph(7),
                  nx.disjoint_union(nx.path_graph(4),nx.cycle_graph(5))]
        graphs += [nx.gnp_random_graph(n,p,seed=i) for n in [5,12,30,70] for p in [.1,.3,.8] for i in range(3)]
        for g in graphs:
            a=nx.to_numpy_array(g,dtype=np.uint8)
            np.testing.assert_allclose(brandes(a),list(nx.betweenness_centrality(g).values()),atol=1e-12)

    def test_full_greedy_recomputes_every_candidate(self):
        for seed in range(8):
            g=nx.gnp_random_graph(9,.25,seed=seed)
            a=nx.to_numpy_array(g,dtype=np.uint8)
            r=solve_baseline(a,0,3,'FullGreedy')
            for edge in r['edges']:
                scored=[]
                for e in sorted(nx.non_edges(g)):
                    h=g.copy(); h.add_edge(*e)
                    scored.append((nx.betweenness_centrality(h)[0],e))
                value=next(v for v,e in scored if list(e)==edge)
                self.assertAlmostEqual(value,max(v for v,e in scored),places=12)
                g.add_edge(*edge)
            self.assertAlmostEqual(r['final'],nx.betweenness_centrality(g)[0],places=12)
            np.testing.assert_array_equal(a,nx.to_numpy_array(nx.gnp_random_graph(9,.25,seed=seed),dtype=np.uint8))

    def test_features_and_no_bc(self):
        g=nx.barabasi_albert_graph(20,2,seed=1)
        a=nx.to_numpy_array(g,dtype=np.uint8)
        x=features(a,0,brandes(a),.6,True)
        np.testing.assert_allclose(x[:,3],list(nx.clustering(g).values()),atol=1e-7)
        self.assertTrue(np.all(features(a,0,np.ones(20),.6,False)[:,2]==0))
        self.assertTrue(np.all(x[:,5]==np.float32(.6)))

    def test_policy_permutation_equivariance(self):
        torch.set_num_threads(1); torch.manual_seed(1)
        model=Policy().eval()
        a=nx.to_numpy_array(nx.path_graph(9),dtype=np.uint8)
        x=features(a,0,brandes(a),1.,True)
        perm=np.random.default_rng(2).permutation(9)
        inv=np.argsort(perm)
        with torch.no_grad():
            h=model.encode(torch.tensor(x)[None],torch.tensor(a,dtype=torch.float32)[None])
            hp=model.encode(torch.tensor(x[perm])[None],torch.tensor(a[perm][:,perm],dtype=torch.float32)[None])
            pairs=torch.tensor([[0,1,3],[4,5,8]])
            scores=model.score(h,torch.tensor([0]),pairs)
            sp=model.score(hp,torch.tensor([int(inv[0])]),torch.tensor(inv[pairs.numpy()]))
            torch.testing.assert_close(scores,sp,atol=2e-6,rtol=2e-6)


if __name__=='__main__': unittest.main()
