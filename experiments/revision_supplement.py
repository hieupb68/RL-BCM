"""Detailed standalone technical report, not assumed to be an accepted supplement."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.revision_run import ROOT
from experiments.revision_report import LABELS,read_rows


def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%')


def longtable(headers,rows):
    out=r'\begin{longtable}{'+('l'+'r'*(len(headers)-1))+'}\n'
    out+=r'\toprule '+' & '.join(headers)+r'\\ \midrule\endhead'+'\n'
    out+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    return out+'\n'+r'\bottomrule\end{longtable}'+'\n'


def main():
    root=Path('paper/revision')
    s=json.loads((ROOT/'summary.json').read_text())
    text=r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,url,fancyvrb,float,etoolbox}
\setlength{\LTpre}{8pt}\setlength{\LTpost}{8pt}
% Backport the finite-shrink fix in LaTeX tools longtable v4.25 (issue 1907).
% No-op when the installed version already uses finite shrink.
\makeatletter
\patchcmd{\LT@output}{\vss}{\vskip\z@\@plus1fil\@minus\normalbaselineskip}{}{}
\patchcmd{\LT@output}{\vss}{\vskip\z@\@plus1fil\@minus\normalbaselineskip}{}{}
\makeatother
\usepackage[protrusion=true,expansion=false]{microtype}
\title{RL-BCM: Detailed Revision Experiments and Reproducibility Report}
\author{Companion technical artifact to the camera-ready manuscript}
\date{25 September 2026}
\begin{document}\maketitle
This report contains additional measured results and implementation details.
It is a standalone research artifact; permission to upload a formal conference
supplement is not presumed. The manuscript is self-contained for its principal
claims. All reported experiments were executed locally; unrun baselines are
explicitly marked as such. Original single-seed results are excluded.
\tableofcontents

\section{Scope and corrections}
The revised evidence changes the scientific conclusion. Full Greedy is stronger
than RL in all three synthetic-family means, and the learned myopic control is
also stronger than the long-horizon model. The results support a learned
heuristic with a measurable quality/cost profile, not a demonstrated planning
advantage. A baseline denominator error in the earlier evaluation invalidated
relative comparisons: positive gains had been divided by final rather than
initial BC. The corrected evaluation replaces those comparisons entirely.

Other corrections include held-out validation, deterministic SHA-256 dataset
seeds, consistent dropout-free PPO likelihoods, a remaining-budget feature,
unordered symmetric edge scores, and absolute terminal-aligned reward.
The exact task adds a fixed number of edges; at-most-budget behavior is audited
separately because BC is not monotone under unrestricted insertion.

\section{Frozen experimental design}
Training uses 480 graphs, validation 36, and test 240. Synthetic configurations
are ER with probabilities 0.1/0.2, BA with attachment 2/3, and WS with
degree--rewiring pairs (4,0.1)/(6,0.2). Connectedness rejection is explicit.
Train sizes are 20,30,40,50; test sizes are 30,50,80,100. Each test family has
two configurations, four sizes, and ten graphs per cell. Targets come from the
bottom quartile of positive BC. Saved graph hashes verify disjoint splits.

Five seeds (11,22,33,44,55) are trained for each of exact-BC, no-BC-input,
and gamma-zero models. Every run executes 5,000 episodes of five decisions.
All variants use three width-64 mean GraphSAGE layers, a 192--64--32--1 edge
MLP, a 128--64--1 value head, and the same optimizer settings. Adam uses
$3\cdot10^{-4}$; PPO uses clip 0.2, four epochs, minibatches of 32, 16 episodes
per rollout, entropy 0.01, value coefficient 0.5, gradient clipping 0.5, and
GAE lambda 0.95. Main gamma is one; the control changes only gamma to zero.
The final checkpoint is used without test or validation selection.

\section{Metric definitions and uncertainty}
Primary outcomes are normalized endpoint-excluding final BC and its absolute
increase. Relative increase is $(B_k-B_0)/B_0$ only for $B_0>10^{-12}$.
The mean of relative increases is not a ratio of means and can be dominated
by small initial values. We report median relative gain as a secondary metric.

For learned models the measurement matrix has five seed rows and graph columns.
The crossed bootstrap independently samples seed indices and graph indices,
using the same sampled graph columns for both members of a contrast. Graphs
are stratified by configuration and size. We use 10,000 replicates with fixed
bootstrap seed 92831. This estimates both observed training-run and test-graph
variation; it does not incorporate new training corpora or new network families.
For deterministic baselines, only graph columns vary. Original Random and global-sampled
baselines have one predefined randomization per graph, so their within-graph
randomization variance is not independently estimated.

Seed-level sign-flip tests use graph-averaged paired differences and all 32
sign assignments. They are two-sided, require a symmetric null for these
differences, and have minimum attainable unadjusted $p=0.0625$. Holm correction
is applied across ER, BA, and WS for each specified comparator. A bootstrap
interval excluding zero and a coarse sign-flip p-value need not agree: the
two procedures use different sampling assumptions and resolution. We do not
describe this discrepancy as proof of statistical significance.

\section{Full primary and secondary metrics}
'''
    rows=[]
    for family in ['ER','BA','WS']:
        for method,x in s[family].items():
            rows.append([family,esc(LABELS[method]),f"{x['initial']:.5f}",f"{x['mean']:.5f}",f"{x['mean_gain']:.5f}",f"{x['median_relative']:.1f}"])
    # Two text columns, then four numeric columns.
    block=longtable(['Family','Method','$B_0$','$B_k$','$B_k-B_0$','Median rel. (\%)'],rows)
    text+=block.replace('{lrrrrr}','{llrrrr}',1)
    text+=r'\subsection{All training-seed means}'+'\n'+r'\input{seed_table}'+'\n'
    text+=r'\begin{figure}[ht]\centering\includegraphics[width=\textwidth]{figures/training.pdf}\caption{Held-out validation for all seeds; final checkpoints are used without selection.}\end{figure}'+'\n'
    text+=r'\subsection{Paired comparisons}'+'\n'
    differences=json.loads((ROOT/'contrasts.json').read_text())
    rows=[]
    for d in differences:
        rows.append([d['family'],esc(d['contrast']),f"{d['mean']:+.5f}",f"[{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]",f"{100*d['win']:.1f}",f"{100*d['tie']:.1f}",f"{d['p_holm']:.3f}"])
    text+=longtable(['Family','Contrast','Difference','95\% CI','Win\%','Tie\%','$p_H$'],rows).replace('{lrrrrrr}','{llrrrrr}',1)
    text+=r'\subsection{Breakdown by graph size}'+'\n'
    allrows=read_rows(ROOT/'baselines_v2.jsonl')+read_rows(ROOT/'rl.jsonl')
    rows=[]
    for family in ['ER','BA','WS']:
        for n in [30,50,80,100]:
            cells=[]
            for method in ['Degree','SampledGreedy20','SampledGreedy100','FullGreedy','exact','no_bc','myopic']:
                rr=[r for r in allrows if r['family']==family and r['n']==n and r['method']==method]
                cells.append(f"{np.mean([r['final'] for r in rr]):.5f}")
            rows.append([f'{family}, {n}']+cells)
    text+=longtable(['Family, $n$','Degree','SG20','SG100','Full G.','RL','No BC','Myopic'],rows)
    text+=r'\subsection{At-most-budget diagnostic}'+'\n'
    text+='The following means select the best prefix of the stored five-step trajectory, including no modification. This is an oracle-assisted diagnostic, not a separately trained stopping policy. No-BC deployment would require additional BC evaluation to choose such a prefix.\n'
    rows=[]
    for family in ['ER','BA','WS']:
        for method in ['Degree','FullGreedy','exact','no_bc','myopic']:
            x=s[family][method]
            rows.append([family,esc(LABELS[method]),f"{x['mean']:.6f}",f"{x['mean_best_prefix']:.6f}"])
    text+=longtable(['Family','Method','Fixed budget','Best prefix'],rows).replace('{lrrr}','{llrr}',1)
    text+=r'''\section{Exact-optimum and budget diagnostics}
\input{diagnostics_results}
The optimal benchmark enumerates all unordered two-edge sets. With $q$ absent
edges this means $\binom{q}{2}$ final graphs. Enumeration of sets is sufficient
because the final graph is order-independent, even though greedy's choices
depend on intermediate graphs. No global-optimality claim is made for greedy.

\subsection{Recorded counterexample}
Vertices are 0 through 7, target 6, and edges are
$\{02,03,04,16,47,56,57,67\}$. The greedy trajectory adds 06 and 26 and ends
at 0.5238095238. The optimal set adds 26 and 36 and ends at 0.5317460317.
All 190 missing-edge pairs were evaluated. Its complete graph and target record
is \texttt{tiny\_ER\_0.3\_8\_4}. This is a verified greedy failure, not a
claim that RL solves this particular graph.

\subsection{Budget transfer by family}
The model always trained at budget five. Initial budget ratios equal one at
every test horizon, so that feature alone does not reveal the absolute horizon
at the first step. Transfer to budgets one, three, and ten is therefore a
stronger distribution shift than merely changing remaining steps in a fixed
five-step episode. Dedicated horizon-conditioned retraining would be a separate
experiment and is not claimed here.
'''
    budget=read_rows(ROOT/'budget.jsonl');rows=[]
    for family in ['ER','BA','WS']:
        for k in [1,3,10]:
            rows.append([family,str(k)]+[f"{np.mean([r['final'] for r in budget if r['family']==family and r['budget']==k and r['method']==m]):.5f}" for m in ['Degree','IncidentGreedy','FullGreedy','exact']])
    text+=longtable(['Family','$k$','Degree','Incident G.','Full G.','RL'],rows)
    text+=r'''\section{Real-network preprocessing and structure}
SNAP email-Eu-core is converted to an undirected simple graph. Both datasets
have self-loops removed before retaining their largest connected component.
The original email directed edge count must not be compared directly to the
processed undirected count. GrQc's published LCC edge count also includes
edges removed by the loop-cleaning step. Source file checksums and vertex-ID
maps are provided with the graph records.

BFS and random-walk samplers select vertices, then retain the induced graph.
The same source network can contribute overlapping samples. No confidence
interval across these samples is interpreted as independent-network uncertainty.
Degree-biased visitation and locality bias remain for random walks and BFS,
respectively; neither is called a representative uniform network sample.
\input{real_table}
\input{real_results}
\subsection{Measured structural properties}
'''
    structures=json.loads((ROOT/'real_structure.json').read_text());rows=[]
    for r in structures:
        rows.append([('Email' if r['dataset']=='email-Eu-core' else 'GrQc')+f" {r['sampling']} {r['index']}",str(r['n']),str(r['m']),f"{r['mean_degree']:.2f}",f"{r['density']:.4f}",f"{r['clustering']:.4f}",f"{r['transitivity']:.4f}"])
    text+=longtable(['Network/sample','$n$','$m$','Mean degree','Density','Clustering','Transitivity'],rows)
    text+=r'''\section{Cost measurements and implementation}
\input{timing_table}
\input{timing_results}
Training elapsed times below are process durations measured while some jobs
overlapped. They include validation evaluations. Summing them is not total
wall-clock time and is not energy consumption.
'''
    runs=json.loads((ROOT/'training_summary.json').read_text());rows=[]
    for r in runs:rows.append([esc(r['variant']),str(r['seed']),str(r['episodes']),str(r['updates']),f"{r['elapsed']:.2f}"])
    text+=longtable(['Variant','Seed','Episodes','PPO updates','Elapsed (s)'],rows)
    text+=r'''\subsection{Exact insertion oracle}
For an inserted edge $uv$, new shortest paths between $s,t$ either avoid that
edge or traverse it in one of its two orientations. Candidate distances are
$d_{st}$, $d_{su}+1+d_{vt}$, and $d_{sv}+1+d_{ut}$. Counts for all tied
minimizers are added. An alleged minimizing walk with repeated vertices would
contain a removable positive-length cycle, contradicting minimality; products
of the original path counts are therefore valid for the minimizing cases.
Disconnected pairs have zero counts and do not contribute. Candidate target
BC is then obtained from all updated pair distances/counts, including distances
to the target. The tables are rebuilt after each committed edge, so this does
not reuse obsolete marginals. Tie tolerance is $10^{-12}$.

\subsection{Source-level checks and artifact audit}
The automated suite compares Brandes to NetworkX; compares all candidate
insertions to independent full BC recomputation; checks exhaustive greedy's
stepwise maximum; tests clustering features and the absent BC channel; and
tests score equivariance under vertex relabeling. A post-hoc audit checks
exact row counts, unique method/seed/graph keys, valid new edges, trajectory
lengths, metric identities, checkpoint completion, and replays final BC.
Deterministically selected outcomes are independently recomputed with NetworkX.

The files named \texttt{baselines.jsonl}, \texttt{real\_baselines.jsonl}, and
\texttt{real\_rl.jsonl} are explicitly excluded development records. Final
quality data use the \texttt{v2} baseline files and
\texttt{real\_rl\_cuda.jsonl}. See \texttt{EXCLUDED\_RUNS.md}.

\section{Reproduction}
Run from the repository root with the recorded Python environment. The train
stage skips completed runs. Remove or move a run directory only when deliberately
starting a fresh reproduction, and keep the original artifact separate.
\begin{Verbatim}[fontsize=\small]
python -m unittest discover -s tests -v
python -m experiments.revision_run prepare
python -m experiments.revision_suite --workers 2
python -m experiments.revision_run baselines
python -m experiments.revision_run evaluate
python -m experiments.revision_extra tiny
python -m experiments.revision_extra tiny_rl
python -m experiments.revision_extra budget
python -m experiments.revision_extra prepare_real
python -m experiments.revision_extra real_baselines
python -m experiments.revision_extra real_rl --device cuda
python -m experiments.revision_timing
python -m experiments.revision_report
python -m experiments.revision_build all
python -m experiments.followup_experiments prepare
python -m experiments.followup_experiments repair_tiny
python -m experiments.followup_experiments annotate_budgets
python -m experiments.followup_experiments synthetic
python -m experiments.followup_experiments real
python -m experiments.followup_experiments timing
python -m experiments.revision_audit
python -m experiments.followup_experiments audit
python -m experiments.followup_report
python -m experiments.revision_supplement
python -m experiments.revision_compile
\end{Verbatim}
CPU execution can replace GPU execution for real-network quality evaluation,
but the output filename and reporting input must be changed consistently.
GPU and CPU floating-point scoring can break near ties differently; the
deployed edge trajectory is always retained and scored with the exact CPU
objective. Timing requires matching hardware and exclusion of concurrent
training or evaluation load. Dataset source files are local copies of the
public SNAP edge lists; no new account or paid compute is required.

\section{Remaining limitations}
There is no evidence here that longer-horizon PPO beats a matched immediate
reward policy. Exhaustive greedy is unrun on large LCCs. Five training seeds
give only coarse inferential resolution. Low-BC target selection, a fixed
training corpus, two real datasets, potential subgraph overlap, one stochastic
baseline realization per graph, dense adjacency, and quadratic pair scoring
limit the scope of conclusions. The report makes no universal optimization,
scalability, reproducibility-across-hardware, or state-of-the-art claim.
\end{document}
'''
    # Keep the report's tables with the sections that discuss them; main-paper
    # tables retain their original LNCS floating placement.
    for name in ['seed_table','real_table','timing_table']:
        content=(root/(name+'.tex')).read_text(encoding='utf-8').replace('\\begin{table}[t]','\\begin{table}[H]')
        text=text.replace('\\input{'+name+'}',content)
    text=text.replace('\\begin{figure}[ht]','\\begin{figure}[H]')
    timing_text=(root/'timing_results.tex').read_text(encoding='utf-8')
    timing_text=timing_text.replace(r'Fig.~\ref{fig:main}', 'the main manuscript figure')
    text=text.replace(r'\input{timing_results}',timing_text)
    if (root/'followup_supplement.tex').exists():
        text=text.replace(r'\section{Remaining limitations}',r'\input{followup_supplement}'+'\n'+r'\section{Remaining limitations}')
        text=text.replace('one stochastic\nbaseline realization per graph', 'one original global-sampled\nbaseline realization per graph (the added incident controls have five)')
    (root/'supplement.tex').write_text(text,encoding='utf-8')
    print('Wrote detailed supplementary report',flush=True)


if __name__=='__main__':main()
