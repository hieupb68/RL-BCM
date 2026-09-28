"""Generate follow-up tables/text directly from completed measurement records."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.revision_report import bootstrap, table, LABELS, read_rows
from experiments.revision_run import ROOT, dump
from experiments.followup_experiments import OUT, rows

P=Path('paper/revision')
FAMS=['ER','BA','WS']
NAMES={'SampledIncidentGreedy20':'Incident SG20','SampledIncidentGreedy100':'Incident SG100',
       'exact':'RL-BCM','no_bc':'No BC','myopic':'Myopic','IncidentGreedy':'Incident Greedy',
       'FullGreedy':'Full Greedy','Degree':'Degree','SampledGreedy20':'Global SG20',
       'SampledGreedy100':'Global SG100'}

def write(name,text): (P/name).write_text(text+'\n',encoding='utf-8')
def fmt(x):
    if 0<abs(x)<.00005:
        coefficient,exponent=f'{x:.2e}'.split('e')
        return f'${coefficient}\\!\\times\\!10^{{{int(exponent)}}}$'
    return f'{x:.4f}'

def main():
    sg=rows(OUT/'synthetic.jsonl');real=rows(OUT/'real.jsonl');timing=rows(OUT/'timing.jsonl')
    assert len(sg)==2400 and len(real)==300 and len(timing)==1170
    old=json.loads((ROOT/'summary.json').read_text())
    gs=json.loads((ROOT/'test_graphs.json').read_text());gmap={g['id']:g for g in gs}
    added={};tab=[]
    for fam in FAMS:
        added[fam]={}
        graphs=[g for g in gs if g['family']==fam]
        strata=[f"{g['param']}_{g['n']}" for g in graphs]
        for method in ['SampledIncidentGreedy20','SampledIncidentGreedy100']:
            index={(x['seed'],x['id']):x for x in sg if x['family']==fam and x['method']==method}
            vals=np.array([[index[s,g['id']]['final'] for g in graphs] for s in [101,202,303,404,505]])
            added[fam][method]=dict(mean=float(vals.mean()),ci=bootstrap(vals,strata),n_graphs=len(graphs),n_seeds=5)
    for method in ['Random','Degree','SampledGreedy20','SampledGreedy100','SampledIncidentGreedy20','SampledIncidentGreedy100','FullGreedy','exact','no_bc','myopic']:
        cells=[]
        for fam in FAMS:
            x=added[fam][method] if method.startswith('SampledIncident') else old[fam][method]
            cells.append(r'\shortstack[r]{'+f"{x['mean']:.4f}"+r'\\{\scriptsize '+f"[{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]"+'}}')
        label=NAMES.get(method,LABELS.get(method,method))
        if method=='FullGreedy':label='Full / Incident Greedy'
        tab.append([label]+cells)
    table('main_table.tex','Mean final BC and 95\\% crossed-bootstrap intervals on 80 graphs per family. Learned policies and incident-sampled baselines have five seeds each; original global-sampled baselines have one run per graph. Full and Incident Greedy coincide on every instance.',['Method','ER','BA','WS'],tab,'tab:main')
    write('abstract_results.tex',
          'RL-BCM reaches 95.8\\%, 88.0\\%, and 93.5\\% of full-greedy mean final BC on ER, BA, and WS, respectively. Both the 20-candidate incident baseline and the myopic control have higher family means, so these experiments do not establish a non-myopic advantage.')
    write('main_results.tex',
          'RL reaches 95.8\\%, 88.0\\%, and 93.5\\% of Full Greedy mean final BC on ER, BA, and WS (Table~\\ref{tab:main}). These are ratios of means, not per-instance approximation guarantees. Full and Incident Greedy coincide on all 240 instances, making incident search the practical reference. RL exceeds Degree and global-sampled greedy in family means. Absolute-gain and median-relative-gain values are in the detailed report.')
    write('conclusion_results.tex',
          'The GraphSAGE--PPO policy improves on global random candidate sampling, but exact incident search and even the 20-candidate incident baseline have higher mean quality on the synthetic test set. The gamma-zero control also exceeds the main policy. Thus the evidence supports a measured learned heuristic, without establishing a long-horizon advantage. Future changes should be tested against incident search on a new holdout set, especially beyond the training sizes.')
    # Keep the historical paired analysis, with reader-facing labels.
    c=(P/'contrast_table.tex').read_text()
    for a,b in [('FullGreedy','Full Greedy'),(r'no\_bc','No BC'),('/ myopic','/ Myopic')]:c=c.replace(a,b)
    write('contrast_table.tex',c)
    synthetic_text='Sampling within the incident action set changes the comparison. Incident SG20 reaches '
    synthetic_text+=', '.join(f"{added[f]['SampledIncidentGreedy20']['mean']:.4f}" for f in FAMS)+' mean final BC on ER, BA, and WS. '
    synthetic_text+='Incident SG100 evaluates every incident candidate when $n\\le100$ and therefore reproduces Incident Greedy on this test set. '
    synthetic_text+='These controls distinguish a limited search budget from spending that budget on a much larger action space.'
    write('incident_results.tex',synthetic_text)
    rl=read_rows(ROOT/'rl.jsonl');base=read_rows(ROOT/'baselines_v2.jsonl')
    incidences={m:dict(incident=sum(gmap[r['id']]['target'] in e for r in data if r['method']==m for e in r['edges']),total=sum(len(r['edges']) for r in data if r['method']==m))
                for m,data in [('exact',rl),('no_bc',rl),('myopic',rl),('FullGreedy',base)]}
    bysize=[]
    for fam in FAMS:
        for n in [30,50,80,100]:
            a=np.mean([r['final'] for r in rl if r['family']==fam and r['n']==n and r['method']=='exact'])
            b=np.mean([r['final'] for r in base if r['family']==fam and r['n']==n and r['method']=='IncidentGreedy'])
            bysize.append(dict(family=fam,n=n,rl=float(a),incident=float(b),ratio=float(a/b)))
    write('transfer_results.tex',
          'Size transfer is uneven (Fig.~\\ref{fig:main}). On BA, the ratio of mean final BC to Incident Greedy falls from 96.0\\% at $n=30$ to 74.3\\% at $n=100$; WS falls from 97.8\\% to 86.1\\%. ER stays near 95--96\\%. The family averages therefore conceal losses beyond the maximum training size of 50. All 6,000 actions of the main policy and all 1,200 Full Greedy actions on the synthetic test set touch the target. Thus these instances do not demonstrate a benefit from unrestricted actions, although they do not establish incident sufficiency in general.')
    # All real methods use common instances; baseline seed repetition is kept explicit.
    rr=read_rows(ROOT/'real_baselines_v2.jsonl')+read_rows(ROOT/'real_rl_cuda.jsonl')+real
    groups=defaultdict(list)
    for r in rr:groups[r['family'],r['sampling'],r['n'],r['method']].append(r)
    realtab=[];realsummary=[]
    methods=['Degree','SampledIncidentGreedy20','SampledIncidentGreedy100','IncidentGreedy','exact','no_bc']
    for fam in ['email-Eu-core','ca-GrQc']:
        for sampling,n in [('BFS',80),('RW',80),('BFS',256),('RW',256),('LCC',986 if fam=='email-Eu-core' else 4158)]:
            cells=[]
            for m in methods:
                values=groups.get((fam,sampling,n,m),[])
                cells.append(fmt(np.mean([x['final'] for x in values])) if values else '--')
                if values:realsummary.append(dict(family=fam,sampling=sampling,n=n,method=m,mean=float(np.mean([x['final'] for x in values])),outcomes=len(values)))
            realtab.append([('Email' if fam=='email-Eu-core' else 'GrQc')+f' {sampling} {n}']+cells)
    table('real_table.tex','Mean final BC on three targets/subgraphs per row. Incident SG20/SG100 and learned models average five seeds; these are not independent source networks. Dashes denote unrun methods. RL quality uses GPU inference; no cross-device speedup is inferred.',['Network/sample','Degree','ISG20','ISG100','Inc. G.','RL','No BC'],realtab,'tab:real')
    rt='The incident-sampled controls extend the stronger comparison to both full LCCs. '
    for fam,n in [('email-Eu-core',986),('ca-GrQc',4158)]:
        get=lambda m:np.mean([x['final'] for x in groups[fam,'LCC',n,m]])
        rt+=f"On {fam}, Incident SG20/SG100 attain {get('SampledIncidentGreedy20'):.4f}/{get('SampledIncidentGreedy100'):.4f}, compared with {get('exact'):.4f} for RL. "
    rt+='Transfer also has substantial failures: on GrQc BFS-256, RL attains 0.0398 versus 0.1386 for Incident Greedy. Global SG20 on Email LCC is $1.865\\times10^{-5}$, not zero. Full Greedy remains unrun above 80 vertices on these networks.'
    write('real_results.tex',rt)
    # Collapse technical repetitions within each graph before summarizing quality.
    ts=[]
    for fam in FAMS:
        for n in [50,100,256]:
            for method in sorted({r['method'] for r in timing if r['family']==fam and r['n']==n}):
                rr0=[r for r in timing if r['family']==fam and r['n']==n and r['method']==method]
                unique={r['id']:r for r in rr0}
                ts.append(dict(family=fam,n=n,method=method,mean_final=float(np.mean([r['final'] for r in unique.values()])),
                    median_seconds=float(np.median([r['seconds'] for r in rr0])),graphs=len(unique),repetitions=3))
    lookup={(x['family'],x['n'],x['method']):x for x in ts}
    timerows=[]
    display=['Degree','SampledIncidentGreedy20','IncidentGreedy','exact','no_bc','myopic']
    for fam in FAMS:
        cells=[]
        for m in display:
            x=lookup[fam,256,m]
            cells.append(r'\shortstack[r]{'+f"{x['mean_final']:.4f}"+r'\\{\scriptsize '+f"{x['median_seconds']:.4f} s"+'}}')
        timerows.append([fam]+cells)
    table('timing_table.tex','Matched quality/cost at $n=256$: mean final BC (top) and median CPU decision seconds (bottom) on five graphs per family, three timing repetitions, model seed 11. One thread; audit BC, loading, compilation and training excluded. ISG20 samples incident edges. All sizes/methods are in the report.',['Family','Degree','ISG20','Inc. G.','RL','No BC','Myopic'],timerows,'tab:timing')
    # Dominance is descriptive of these mean-quality/median-time points.
    dominance=[]
    for fam in FAMS:
        for n in [50,100,256]:
            a=lookup[fam,n,'exact']
            for method in ['Degree','SampledIncidentGreedy20','SampledIncidentGreedy100','IncidentGreedy','no_bc','myopic']:
                b=lookup[fam,n,method]
                if b['mean_final']>=a['mean_final'] and b['median_seconds']<=a['median_seconds'] and (b['mean_final']>a['mean_final'] or b['median_seconds']<a['median_seconds']):
                    dominance.append(dict(family=fam,n=n,method=method))
    tt='Table~\\ref{tab:timing} pairs quality and time on identical graphs; Fig.~\\ref{fig:main} shows the BA-256 tradeoff. '
    for method in ['SampledIncidentGreedy20','IncidentGreedy','exact','no_bc']:
        x=lookup['BA',256,method]
        tt+=f"{NAMES[method]} obtains BC {x['mean_final']:.4f} in {x['median_seconds']:.4f}\\,s. "
    tt+='A latency saving is useful only if the associated quality is acceptable. The five main-model training runs cost 37.1 cumulative process-minutes (101.7 for all 15 runs), including validation; overlapping processes prevent interpreting these sums as wall-clock duration. '
    train=json.loads((ROOT/'exact_seed11/complete.json').read_text())['elapsed']
    saved=lookup['BA',100,'IncidentGreedy']['median_seconds']-lookup['BA',100,'exact']['median_seconds']
    if saved>0:
        tt+=f"Using seed 11's {train:.1f}\\,s training duration and the BA-100 median saving over Incident Greedy gives an illustrative break-even of {train/saved:,.0f} uses, before accounting for unequal quality. "
    else:tt+='On BA-100 the measured main-policy latency does not improve on Incident Greedy, so these medians give no training-cost break-even against that comparator. '
    tt+='This is a conditional calculation, not a universal deployment threshold.'
    count=len({(x['family'],x['n']) for x in dominance})
    er=lookup['ER',100,'exact'];eg=lookup['ER',100,'IncidentGreedy']
    ba=lookup['BA',256,'exact'];bi=lookup['BA',256,'SampledIncidentGreedy20']
    compact='Table~\\ref{tab:timing} pairs quality and time on identical graphs; Fig.~\\ref{fig:main} shows BA-256. '
    compact+=f'Another measured method dominates the main policy in {count}/9 family--size cells when comparing mean BC and median time. '
    compact+=f"For ER-100, RL attains BC {er['mean_final']:.4f} in {er['median_seconds']:.4f}\\,s, versus {eg['mean_final']:.4f} in {eg['median_seconds']:.4f}\\,s for Incident Greedy: a conditional tradeoff, not superiority. "
    compact+=f"For BA-256, ISG20 has both higher BC ({bi['mean_final']:.4f} versus {ba['mean_final']:.4f}) and lower latency ({bi['median_seconds']:.4f} versus {ba['median_seconds']:.4f}\\,s). "
    compact+='These are descriptive comparisons on five graphs per cell and one model seed. The five main training runs cost 37.1 cumulative process-minutes, including validation; overlapping runs prevent interpreting this sum as wall-clock duration. Any training-cost amortization requires repeated uses, positive latency savings, and acceptable quality.'
    write('timing_results.tex',compact)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.25),layout='constrained')
    for fam in FAMS:
        vals=[r for r in bysize if r['family']==fam]
        axes[0].plot([r['n'] for r in vals],[r['ratio']*100 for r in vals],marker='o',label=fam)
    axes[0].axvspan(50,100,color='gray',alpha=.08)
    axes[0].set(xlabel='Vertices',ylabel='RL / Incident Greedy (%)',xticks=[30,50,80,100],ylim=(65,102))
    axes[0].legend(fontsize=8)
    short={'SampledIncidentGreedy20':'ISG20','SampledIncidentGreedy100':'ISG100','IncidentGreedy':'Inc. G.','exact':'RL','no_bc':'No BC','myopic':'Myopic','Degree':'Degree'}
    for m in ['Degree','SampledIncidentGreedy20','SampledIncidentGreedy100','IncidentGreedy','exact','no_bc','myopic']:
        x=lookup['BA',256,m];axes[1].scatter(x['median_seconds'],x['mean_final'],s=18)
        offset=(4,-10) if m=='exact' else (4,5)
        axes[1].annotate(short[m],(x['median_seconds'],x['mean_final']),xytext=offset,textcoords='offset points',fontsize=8)
    axes[1].set(xscale='log',xlabel='CPU seconds, BA-256',ylabel='Mean final BC')
    axes[1].margins(x=.3,y=.25)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(P/'figures/followup.pdf');plt.close(fig)
    tiny=json.loads((ROOT/'tiny_optima.json').read_text());tr=json.loads((ROOT/'tiny_rl.json').read_text())
    bad={x['id'] for x in tiny if x['greedy']['final']<x['optimum']-1e-12}
    tx={x['id']:x for x in tiny};rs=[x for x in tr if x['method']=='exact' and x['id'] in bad]
    familyopt={f:float(np.mean([abs(x['final']-x['optimum'])<1e-12 for x in tr if x['method']=='exact' and x['family']==f])) for f in FAMS}
    diagnostics='The pooled optimum rate hides ER/BA/WS rates of 22\\%, 22\\%, and 95\\%. On the six instances where Full Greedy is suboptimal, the main policy beats greedy and reaches the optimum in only 1/30 seed--graph outcomes. Six tiny instances cannot accommodate two incident additions; the artifact now marks Incident Greedy infeasible there and reports no fixed-budget final score.'
    write('tiny_followup.tex',diagnostics)
    # Detailed appendix is separate from the stable ten original tables.
    extra=r'\section{Incident-sampled controls and matched timing}'+'\n'
    extra+='This follow-up was specified after the initial evaluation. It retains all 15 checkpoints without further tuning. Each incident-sampled baseline has five random seeds (101, 202, 303, 404, 505) on 240 synthetic and 30 real instances: 2,700 added quality outcomes. The timing set contains 45 new graphs (five per family and size 50, 100, 256), with three repetitions, one thread, and model seed 11. Technical repetitions do not increase the number of independent graphs.\n\n'
    extra+=synthetic_text+'\n\n'+diagnostics+'\n\n'
    extra+=tt.replace(r'Fig.~\ref{fig:main}','the main manuscript figure')+'\n\n'
    extra+='The incident candidate generator directly lists target non-neighbors in lexicographic order. It avoids constructing all nonedges for incident methods; earlier latency records remain archived and are not pooled with this new benchmark. Infeasible fixed-budget requests return a null final score, a status flag, and the requested/used budget. No unrestricted fallback is silently substituted.\n\n'
    extra+='Incident monotonicity follows directly from the path-fraction definition. For endpoints distinct from the target, every new path using an inserted target-incident edge passes through the target. If new shortest paths are shorter, the contribution becomes one; if they tie existing shortest paths, the fraction changes from $a/b$ to $(a+c)/(b+c)$ with $a\\le b$; otherwise it is unchanged. Previously disconnected pairs cannot decrease their zero contribution. Summing with a fixed normalization proves nondecrease, not global optimality of an incident restriction.\n\n'
    extra+=r'\begin{longtable}{llrrr}\toprule Family & Method & $n$ & Mean final BC & Median seconds\\\midrule\endhead'+'\n'
    for x in ts:extra+=f"{x['family']} & {NAMES.get(x['method'],x['method'])} & {x['n']} & {x['mean_final']:.6f} & {x['median_seconds']:.6f}"+r'\\'+'\n'
    extra+=r'\bottomrule\end{longtable}'+'\n'
    extra+='The matched mean-quality/median-time comparison finds the main policy dominated by another measured method in '+str(len({(x['family'],x['n']) for x in dominance}))+' of nine family--size cells. This is descriptive evidence on five graphs per cell, not a significance or universal dominance claim. Full records retain all seed and graph outcomes.\n'
    write('followup_supplement.tex',extra)
    dump(OUT/'summary.json',dict(synthetic=added,real=realsummary,timing=ts,dominance=dominance,action_incidence=incidences,size_transfer=bysize,tiny_optimum_by_family=familyopt,tiny_greedy_failure_outcomes=len(rs),tiny_beats_greedy=sum(x['final']>tx[x['id']]['greedy']['final']+1e-12 for x in rs)))
    print(json.dumps({'synthetic':added,'dominance':dominance,'real_LCC':[r for r in realsummary if r['sampling']=='LCC']},indent=2),flush=True)

if __name__=='__main__':main()
