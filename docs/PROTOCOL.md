# Experimental protocol

The protocol below was fixed before the test results were examined. All
measurements are written to `results/revision/`.

## Task and model

- Simple undirected, unweighted graphs; exactly `k = 5` distinct missing edges
  are added. The fixed-cardinality task matches execution. At-most-budget
  behavior can be audited separately by choosing the best prefix, including
  zero edges.
- Finite-horizon state includes the remaining budget. Reward is
  `100 x` the absolute BC increment and `gamma = 1`, so rewards telescope to the
  final-BC objective.
- Mean-aggregation GraphSAGE, 3 layers, width 64, six node features; symmetric
  edge scores `MLP([h_u + h_v, |h_u - h_v|, h_target])`; target-conditioned
  value head.
- PPO: Adam 3e-4, lambda = 0.95, clip = 0.2, entropy = 0.01, value
  coefficient = 0.5, gradient-norm clipping = 0.5; 16 episodes per update,
  four epochs, minibatches of 32 transitions; no dropout.
- Variants: `exact` (RL-BCM), `no_bc` (BC input feature set to zero; rewards
  still exact), and `myopic` (`gamma = 0`, a matched immediate-reward control,
  not exact greedy).
- Training seeds 11, 22, 33, 44, 55; 5,000 episodes each. The final checkpoint
  is used; no test-based architecture, checkpoint, or seed selection.

## Data

- 480 training graphs: 6 configurations x sizes 20/30/40/50 x 20 graphs.
- 36 validation graphs: 6 configurations x sizes 30/50 x 3 graphs.
- 240 test graphs: 6 configurations x sizes 30/50/80/100 x 10 graphs.
- Configurations: ER p = 0.1/0.2 (conditioned on connectedness), BA m = 2/3,
  WS degree/rewiring (4, 0.1) and (6, 0.2).
- One target per graph, uniform from the lowest quartile of vertices with
  BC > 1e-12. Graph and target seeds are SHA-256 hashes of split and
  configuration identifiers; graph hashes are checked for overlap across splits.
- 60 eight-vertex graphs (20 each of ER p = 0.3, BA m = 2, WS (4, 0.2)) for
  exhaustive two-edge optima; the 60 test graphs with n = 50 are re-solved
  with budgets 1, 3, 10.
- SNAP email-Eu-core and ca-GrQc, simple undirected LCCs; three LCC targets
  and three induced BFS / random-walk subgraphs of 80 and 256 vertices each.

## Baselines

- Full Greedy enumerates all missing unordered edges at every step; Incident
  Greedy enumerates all missing target-incident edges; an incident method with
  fewer than `k` incident non-edges is reported as infeasible.
- Global SG20/SG100 sample 20/100 missing edges uniformly per step; ISG20/ISG100
  sample 20/100 target non-neighbors (five seeds: 101, 202, 303, 404, 505).
- Degree and Random complete the baseline set.
- All methods use the same float64 exact Brandes oracle, tested against
  NetworkX. Greedy candidate values use an exact shortest-path insertion
  formula, checked against full recomputation on 36 graphs; ties within 1e-12
  are resolved lexicographically.

## Statistics and timing

- Primary outcome: final normalized BC (absolute gain is identical for paired
  comparisons). Median relative gain is secondary; it is undefined for
  zero initial BC.
- Crossed bootstrap (10,000 resamples) over model seeds and graph instances,
  graphs stratified by configuration and size. Seed-level sign-flip tests
  (all 2^5 flips) with Holm correction across the three families.
- Matched timing: 45 separate graphs (five per ER/BA/WS family at n = 50, 100,
  256), three repetitions, one CPU thread, model seed 11, shuffled method order,
  no concurrent workload. Decision time excludes audit BC, model loading,
  compilation, and training.
