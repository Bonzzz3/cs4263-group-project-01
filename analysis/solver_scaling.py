import sys, json, time, random, itertools, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(ROOT))
R=str(ROOT)+'/'
from cops_robbers import solve, core_tree
inst=[json.loads(l) for l in open(R+'dataset/instances_test_240.jsonl')]
def states(n,k): return 2*math.comb(n+k-1,k)*n
def run(n,edges,k):
    t=time.perf_counter(); solve(n,edges,k); return time.perf_counter()-t
print("guards scaling on test graphs (median of 3 graphs)")
for n in (12,20,26):
    gs=[i for i in inst if i['n']==n][:3]
    for k in (1,2,3,4):
        if states(n,k)>3e6: print(n,k,states(n,k),'skipped'); continue
        ts=sorted(run(n,g['edges'],k) for g in gs)
        print(n,k,states(n,k),round(ts[1],3))
print("rooms scaling, 2 guards (core-tree graphs)")
for n in (26,40,60,80):
    rng=random.Random(n); 
    try: edges=core_tree(n,rng)
    except Exception as e: print('gen fail',e); break
    if isinstance(edges,tuple): edges=edges[-1] if isinstance(edges[-1],list) else edges[0]
    print(n,2,states(n,2),round(run(n,edges,2),2))
