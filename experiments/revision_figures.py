"""Regenerate vector publication figures without repeating statistical tests."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.revision_run import ROOT,SEEDS
from experiments.revision_report import PAPER,LABELS

plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
s=json.loads((ROOT/'summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(10,2.8))
selected=['Degree','SampledGreedy20','FullGreedy','exact','no_bc','myopic']
for ax,fam in zip(axes,['ER','BA','WS']):
    means=[s[fam][m]['mean'] for m in selected]
    err=np.array([[v-s[fam][m]['ci'][0],s[fam][m]['ci'][1]-v] for m,v in zip(selected,means)]).T
    ax.bar(range(6),means,yerr=err,color=['#9da8b5','#a6c7cc','#355070','#e56b6f','#eaac8b','#6d597a'],capsize=3)
    ax.set_xticks(range(6),['Degree','SG20','Full G.','RL','No BC','Myopic'],rotation=45,ha='right');ax.set_title(fam)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Final normalized BC');fig.tight_layout();fig.savefig(PAPER/'figures/main.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
for ax,variant in zip(axes,['exact','no_bc','myopic']):
    for seed in SEEDS:
        history=json.loads((ROOT/f'{variant}_seed{seed}/history.json').read_text())
        points=[h for h in history if 'validation_final' in h]
        ax.plot([h['episode'] for h in points],[h['validation_final'] for h in points],label=str(seed),alpha=.8)
    ax.set_title(LABELS[variant]);ax.set_xlabel('Training episodes');ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Validation mean final BC');axes[-1].legend(title='Seed',fontsize=6)
fig.tight_layout();fig.savefig(PAPER/'figures/training.pdf');plt.close(fig)
print('Vector figures regenerated with embedded TrueType fonts.')
