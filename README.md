# RL-BCM: Reinforcement Learning for Betweenness Centrality Maximization

Code, data, trained models, and raw results for the paper

> Hieu B. Pham, Uyen T. Nguyen, Canh V. Pham.
> **Reinforcement Learning for Betweenness Centrality Maximization via Graph
> Structure Modification.** In: *Proceedings of the 15th International
> Conference on Computational Science and Network Intelligence (CSoNet 2026)*,
> Lecture Notes in Computer Science, Springer, 2026.

The task is to add exactly `k` edges to an undirected graph so that the
betweenness centrality (BC) of a designated target vertex is as large as
possible. RL-BCM is a target-conditioned GraphSAGE policy trained with PPO on a
finite-horizon MDP whose undiscounted rewards telescope to the terminal BC gain.
It is evaluated against exact full and incident greedy search, sampled greedy
search, degree and random heuristics, exact optima on small graphs, and two
SNAP networks.

## Main results (mean final normalized BC, 80 test graphs per family, k = 5)

| Method | ER | BA | WS |
|---|---:|---:|---:|
| Full / Incident Greedy (= ISG100) | 0.0835 | 0.0713 | 0.2823 |
| ISG20 (incident sampled greedy, 20 candidates) | 0.0817 | 0.0669 | 0.2728 |
| RL-BCM | 0.0801 | 0.0628 | 0.2639 |
| No-BC (RL-BCM without BC input) | 0.0785 | 0.0641 | 0.2592 |
| Myopic (RL-BCM with gamma = 0) | 0.0809 | 0.0663 | 0.2678 |
| Global SG100 | 0.0702 | 0.0514 | 0.2335 |
| Degree | 0.0456 | 0.0237 | 0.2081 |

RL-BCM reaches 95.8% / 88.0% / 93.5% of Full Greedy on ER / BA / WS and beats
degree and global sampled search, but incident search and the myopic control
are stronger, and quality drops for graphs larger than those used in training.
See the paper and `paper/revision/supplement.pdf` for intervals, statistics,
real-network transfer, and timing.

## Repository layout

```
src/revision_core.py        exact Brandes BC (Numba), exact incremental greedy oracle,
                            baselines, GraphSAGE policy/value network, PPO
experiments/                experiment pipeline (data, training, evaluation, reports)
tests/test_revision.py      unit tests (BC vs NetworkX, incremental updates, policy)
data/                       SNAP email-Eu-core and ca-GrQc edge lists
results/revision/           all graphs/targets, 15 checkpoints, histories, raw results
  exact_seed*/              RL-BCM checkpoints (model.pt), config, training history
  no_bc_seed*/, myopic_seed*/  ablation checkpoints
  *_graphs.json             train / validation / test graph-target instances
  baselines_v2.jsonl, rl.jsonl   synthetic quality results (one row per run)
  real_*_v2.jsonl, real_rl_cuda.jsonl  real-network results
  followup/                 incident-sampled controls and matched timing
  EXCLUDED_RUNS.md          which files are diagnostic and must not be aggregated
paper/main.tex              camera-ready LaTeX source (Springer LNCS)
paper/revision/supplement.pdf   detailed experiments and reproducibility report
docs/PROTOCOL.md            frozen experimental protocol
```

## Installation

Python 3.11 was used (PyTorch 2.0.1, NumPy 1.26.4, NetworkX 3.4.2,
SciPy 1.14.1, Numba 0.67.0, Matplotlib 3.10).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # install the PyTorch build for your platform first if needed
python -m unittest discover -s tests -v
```

## Reproducing the experiments

All commands run from the repository root and write to `results/revision/`.
Training skips runs whose directory already contains `complete.json`; move
`results/revision/` aside to reproduce from scratch.

```bash
# 1. data, training (3 variants x 5 seeds x 5,000 episodes), evaluation
python -m experiments.revision_run prepare
python -m experiments.revision_suite --workers 2
python -m experiments.revision_run baselines
python -m experiments.revision_run evaluate

# 2. exact optima on 8-vertex graphs, budget transfer, real networks
python -m experiments.revision_extra tiny
python -m experiments.revision_extra tiny_rl
python -m experiments.revision_extra budget
python -m experiments.revision_extra prepare_real
python -m experiments.revision_extra real_baselines
python -m experiments.revision_extra real_rl --device cuda   # or --device cpu

# 3. incident-sampled controls and matched CPU timing (run timing on an idle machine)
python -m experiments.followup_experiments prepare
python -m experiments.followup_experiments repair_tiny
python -m experiments.followup_experiments annotate_budgets
python -m experiments.followup_experiments synthetic
python -m experiments.followup_experiments real
python -m experiments.followup_experiments timing

# 4. audits, statistics, tables, figures, detailed report
python -m experiments.revision_timing
python -m experiments.revision_audit
python -m experiments.followup_experiments audit
python -m experiments.revision_report
python -m experiments.revision_figures
python -m experiments.followup_report
python -m experiments.revision_supplement
```

Compile the report with `pdflatex supplement.tex` (three passes) in
`paper/revision/`, and the paper with pdfLaTeX + BibTeX in `paper/`.

## Using a trained checkpoint

```python
import numpy as np, networkx as nx
from experiments.revision_run import load_model
from src.revision_core import solve_policy

model = load_model('exact', 11)            # RL-BCM, seed 11 ('no_bc' / 'myopic' for ablations)
g = nx.barabasi_albert_graph(60, 2, seed=0)
a = nx.to_numpy_array(g, dtype=np.uint8)   # dense 0/1 adjacency, vertices 0..n-1
result = solve_policy(model, a, target=17, budget=5)   # use_bc=False for the No-BC model
print(result['edges'], result['initial'], result['final'])
```

## Data sources

- email-Eu-core: H. Yin, A. R. Benson, J. Leskovec, D. F. Gleich, KDD 2017,
  <https://snap.stanford.edu/data/email-Eu-core.html>
- ca-GrQc: J. Leskovec, J. Kleinberg, C. Faloutsos, ACM TKDD 2007,
  <https://snap.stanford.edu/data/ca-GrQc.html>

## Citation

```bibtex
@inproceedings{Pham2026RLBCM,
  author    = {Pham, Hieu B. and Nguyen, Uyen T. and Pham, Canh V.},
  title     = {Reinforcement Learning for Betweenness Centrality Maximization
               via Graph Structure Modification},
  booktitle = {Computational Science and Network Intelligence (CSoNet 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The SNAP datasets are
redistributed under their original terms from the Stanford Network Analysis
Project.
