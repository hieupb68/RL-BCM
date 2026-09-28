"""Schedule independent training jobs; experiments are real local processes."""
import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path
from experiments.revision_run import ROOT, SEEDS


def run(job):
    variant, seed = job
    directory=ROOT/f'{variant}_seed{seed}'
    if (directory/'complete.json').exists():
        return f'skip completed {job}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u','-m','experiments.revision_run','train',
                               '--variant',variant,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{job} failed; see {directory}/run.log')
    return f'completed {job}'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--exclude-exact11',action='store_true')
    p.add_argument('--variants',nargs='+',default=['exact','no_bc','myopic'])
    args=p.parse_args()
    jobs=[(v,s) for v in args.variants for s in SEEDS
          if not (args.exclude_exact11 and v=='exact' and s==11)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run,jobs):
            print(result,flush=True)
