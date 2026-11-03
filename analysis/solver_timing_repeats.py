import sys,json,time,random,statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(ROOT))
R=str(ROOT)+'/'
from cops_robbers import solve, core_tree
inst=[json.loads(l) for l in open(R+'dataset/instances_test_240.jsonl')]
gs=[i for i in inst if i['n']==26][:3]
for k in (2,3,4):
    ts=[]
    for g in gs:
        t=time.perf_counter(); solve(26,g['edges'],k); ts.append(time.perf_counter()-t)
    print(26,k,round(statistics.median(ts),3))
for n in (40,60,80):
    e=core_tree(n,random.Random(n))[1]; t=time.perf_counter(); solve(n,e,2); print(n,2,round(time.perf_counter()-t,2))
