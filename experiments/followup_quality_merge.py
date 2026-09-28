"""Coordinate disjoint quality workers; never used for runtime benchmarking."""
import json
import time
from pathlib import Path
import psutil
from experiments.followup_experiments import OUT, rows
from experiments.revision_run import dump

prefix=OUT/'real.jsonl'
tail=OUT/'real_tail.jsonl'
while True:
    data=rows(prefix)
    finished={(r['id'],r['method'],r['seed']) for r in data}
    required=[r for r in data if r['id']=='ca-GrQc_LCC_4158_0_target1']
    if len(required)==10:
        stopped=[]
        candidates=[]
        for p in psutil.process_iter(['pid','name','cmdline']):
            cmd=p.info['cmdline'] or []
            if p.info['name'].lower().startswith('python') and '-m' in cmd and 'experiments.followup_experiments' in cmd and cmd[-1]=='real':
                if Path(p.cwd()).resolve()==Path.cwd().resolve():candidates.append(p)
        assert len(candidates)<=1, 'Ambiguous worker ownership; do not terminate multiple jobs'
        for p in candidates:
            stopped.append(p.pid)
            p.terminate()
            p.wait(timeout=15)
        print('Prefix complete; stopped only matching real-quality worker',stopped,flush=True)
        break
    time.sleep(3)
# Keep the completed prefix as provenance before merging disjoint target jobs.
original=prefix.read_bytes()
(OUT/'real_prefix_raw.jsonl').write_bytes(original)
while len(rows(tail))<130:time.sleep(3)
data=rows(prefix)+rows(tail)
unique={}
for r in data:
    key=(r['id'],r['method'],r['seed'])
    if key in unique:
        assert abs(r['final']-unique[key]['final'])<1e-12 and r['edges']==unique[key]['edges']
    else:unique[key]=r
assert len(unique)==300
prefix.write_text(''.join(json.dumps(r)+'\n' for r in unique.values()),encoding='utf-8')
dump(OUT/'quality_execution.json',dict(mode='Two disjoint target workers; quality only',rows=300,
     prefix_rows=len(rows(OUT/'real_prefix_raw.jsonl')),tail_rows=130,
     note='Quality-stage seconds are not used for speedup claims; the later timing stage is isolated.'))
print('Merged all 300 unique real quality outcomes',flush=True)
