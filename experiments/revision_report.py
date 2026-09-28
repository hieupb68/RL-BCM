"""Generate tables/figures solely from complete per-instance measurement records."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
from experiments.revision_run import ROOT, SEEDS, dump

PAPER=Path('paper/revision')
LABELS={'exact':'RL-BCM','no_bc':'RL without BC','myopic':'RL myopic',
        'FullGreedy':'Full Greedy','IncidentGreedy':'Incident Greedy',
        'SampledGreedy20':'Sampled Greedy (20)','SampledGreedy100':'Sampled Greedy (100)',
        'Degree':'Degree','Random':'Random'}


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]


def bootstrap(values, strata, repetitions=10000, seed=92831):
    """Crossed seed x graph bootstrap; resample graph columns by fixed stratum."""
    rng=np.random.default_rng(seed)
    groups=[np.where(np.array(strata)==s)[0] for s in sorted(set(strata))]
    samples=np.empty(repetitions)
    for b in range(repetitions):
        si=rng.integers(values.shape[0],size=values.shape[0])
        gi=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        samples[b]=values[np.ix_(si,gi)].mean()
    return np.quantile(samples,[.025,.975]).tolist()


def signflip(d):
    obs=abs(np.mean(d))
    null=[abs(np.mean(np.array(signs)*d)) for signs in itertools.product([-1,1],repeat=len(d))]
    return float(np.mean(np.array(null)>=obs-1e-14))


def holm(ps):
    order=np.argsort(ps); adjusted=np.zeros(len(ps)); last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(len(ps)-rank)*ps[index])); adjusted[index]=last
    return adjusted.tolist()


def table(name, caption, columns, rows, label):
    s='\\begin{table}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n'
    s+='\\begin{tabular}{'+('l'+'r'*(len(columns)-1))+'}\n\\toprule\n'
    s+=' & '.join(columns)+' \\\\\n\\midrule\n'
    s+=''.join(' & '.join(row)+' \\\\\n' for row in rows)
    s+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (PAPER/name).write_text(s,encoding='utf-8')


def main():
    PAPER.mkdir(parents=True,exist_ok=True)
    (PAPER/'figures').mkdir(exist_ok=True)
    records=json.loads((ROOT/'test_graphs.json').read_text())
    baselines=read_rows(ROOT/'baselines_v2.jsonl'); rl=read_rows(ROOT/'rl.jsonl')
    assert len(baselines)==240*6, len(baselines)
    assert len(rl)==240*5*3,len(rl)
    index={(r['id'],r['method'],r.get('seed',0)):r for r in baselines+rl}
    assert len(index)==len(baselines)+len(rl)
    methods=list(LABELS)
    summary={}; differences=[]; seed_rows=[]
    matrices={}
    for family in ['ER','BA','WS']:
        rr=[r for r in records if r['family']==family]
        strata=[f"{r['param']}_{r['n']}" for r in rr]
        summary[family]={}
        for method in methods:
            seeds=SEEDS if method in ['exact','no_bc','myopic'] else [0]
            rows=[[index[r['id'],method,s] for r in rr] for s in seeds]
            vals=np.array([[x['final'] for x in row] for row in rows])
            matrices[family,method]=vals
            rel=[x['relative']*100 for row in rows for x in row if x['relative'] is not None]
            means=vals.mean(axis=1)
            result=dict(mean=float(vals.mean()),ci=bootstrap(vals,strata),
                        seed_means=means.tolist(), seed_sd=float(means.std(ddof=1)) if len(means)>1 else None,
                        initial=float(np.mean([row[0]['initial'] for row in zip(*rows)])),
                        mean_gain=float(np.mean([x['gain'] for row in rows for x in row])),
                        median_relative=float(np.median(rel)),
                        mean_seconds=float(np.mean([x['seconds'] for row in rows for x in row])),
                        mean_oracle_calls=float(np.mean([x['bc_calls'] for row in rows for x in row])),
                        mean_best_prefix=float(np.mean([max(x['curve']) for row in rows for x in row])),
                        n_graphs=len(rr),n_seeds=len(seeds))
            summary[family][method]=result
            if len(seeds)>1:
                for seed,m in zip(seeds,means): seed_rows.append(dict(family=family,method=method,seed=seed,mean_final=float(m)))
        for other in ['FullGreedy','IncidentGreedy','Degree','no_bc','myopic']:
            d=matrices[family,'exact']-matrices[family,other]
            dm=d.mean(axis=1)
            differences.append(dict(family=family,contrast='exact - '+other,mean=float(d.mean()),
                ci=bootstrap(d,strata),seed_differences=dm.tolist(),p_signflip=signflip(dm),
                win=float(np.mean(d>1e-12)),tie=float(np.mean(abs(d)<=1e-12)),loss=float(np.mean(d < -1e-12))))
    for other in ['FullGreedy','IncidentGreedy','Degree','no_bc','myopic']:
        group=[d for d in differences if d['contrast']=='exact - '+other]
        for d,p in zip(group,holm([d['p_signflip'] for d in group])): d['p_holm']=p
    dump(ROOT/'summary.json',summary);dump(ROOT/'contrasts.json',differences);dump(ROOT/'seed_means.json',seed_rows)
    rows=[]
    for method in ['Random','Degree','SampledGreedy20','SampledGreedy100','IncidentGreedy','FullGreedy','exact','no_bc','myopic']:
        cells=[]
        for family in ['ER','BA','WS']:
            x=summary[family][method]
            cells.append('\\shortstack[r]{'+f"{x['mean']:.4f}"+'\\\\{\\scriptsize '+f"[{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]"+'}}')
        rows.append([LABELS[method]]+cells)
    table('main_table.tex','Mean final normalized BC and 95\\% crossed-bootstrap intervals. Each family has 80 test graphs; RL results include five independent training seeds. Baselines have one predefined run per graph.',['Method','ER','BA','WS'],rows,'tab:main')
    rows=[]
    for family in ['ER','BA','WS']:
        for method in ['exact','no_bc','myopic']:
            x=summary[family][method]
            rows.append([family+' / '+LABELS[method]]+[f'{v:.4f}' for v in x['seed_means']]+[f"{x['seed_sd']:.4f}"])
    table('seed_table.tex','Mean final BC per training seed on the same 80 graphs per family. SD is between the five seed means, not between graph instances.',['Family / model']+[str(s) for s in SEEDS]+['SD'],rows,'tab:seeds')
    rows=[]
    for d in differences:
        if d['contrast'] not in ['exact - FullGreedy','exact - myopic','exact - no_bc']:continue
        rows.append([d['family']+' / '+d['contrast'].replace('exact - ','') .replace('_','\\_'),f"{d['mean']:+.4f}",f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]",f"{100*d['win']:.1f}",f"{d['p_holm']:.3f}"])
    table('contrast_table.tex','Paired final-BC difference (RL-BCM minus comparator), crossed-bootstrap interval, strict win percentage across paired seed--graph outcomes, and two-sided seed sign-flip $p$ after Holm correction across three families per contrast.',['Family / comparator','$\\Delta$','95\\% CI','Win (\\%)','$p_H$'],rows,'tab:contrasts')
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=False)
    selected=['Degree','SampledGreedy20','FullGreedy','exact','no_bc','myopic']
    for ax,fam in zip(axes,['ER','BA','WS']):
        means=[summary[fam][m]['mean'] for m in selected]
        err=np.array([[v-summary[fam][m]['ci'][0],summary[fam][m]['ci'][1]-v] for m,v in zip(selected,means)]).T
        ax.bar(range(6),means,yerr=err,color=['#9da8b5','#a6c7cc','#355070','#e56b6f','#eaac8b','#6d597a'],capsize=3)
        ax.set_xticks(range(6),['Degree','SG20','Full G.','RL','No BC','Myopic'],rotation=45,ha='right');ax.set_title(fam)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Final normalized BC');fig.tight_layout();fig.savefig(PAPER/'figures/main.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
    complete=[]
    for ax,variant in zip(axes,['exact','no_bc','myopic']):
        for seed in SEEDS:
            history=json.loads((ROOT/f'{variant}_seed{seed}/history.json').read_text())
            points=[h for h in history if 'validation_final' in h]
            ax.plot([h['episode'] for h in points],[h['validation_final'] for h in points],label=str(seed),alpha=.8)
            complete.append(json.loads((ROOT/f'{variant}_seed{seed}/complete.json').read_text()))
        ax.set_title(LABELS[variant]);ax.set_xlabel('Training episodes');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Validation mean final BC');axes[-1].legend(title='Seed',fontsize=6)
    fig.tight_layout();fig.savefig(PAPER/'figures/training.pdf');plt.close(fig)
    dump(ROOT/'training_summary.json',complete)
    lines=['# Revision results (generated from measurements)','',
           '240 graphs, 5 training seeds per variant, 5,000 episodes per run; 15 completed runs.','',
           '| Family | Method | Mean final BC | 95% CI | Between-seed SD | Median relative gain % |',
           '|---|---|---:|---|---:|---:|']
    for fam in summary:
        for method,x in summary[fam].items():
            lines.append(f"| {fam} | {LABELS[method]} | {x['mean']:.6f} | {x['ci']} | {x['seed_sd']} | {x['median_relative']:.2f} |")
    lines+=['','## Paired contrasts','',json.dumps(differences,indent=2),'',
            'Five seed-level two-sided sign-flip tests have minimum attainable p=0.0625 before correction. Bootstrap intervals are descriptive uncertainty estimates, not a substitute for additional independent training runs.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({f:{m:round(x['mean'],5) for m,x in v.items()} for f,v in summary.items()},indent=2),flush=True)


if __name__=='__main__': main()
