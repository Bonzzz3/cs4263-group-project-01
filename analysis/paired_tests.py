import json, sys, random, statistics, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(ROOT))
from cops_robbers import score_one
R=str(ROOT)+'/'
inst=[json.loads(l) for l in open(R+'dataset/instances_test_240.jsonl')]
def load(v):
    return {r['id']:r for r in map(json.loads, open(R+f'work/luna_medium_{v}_test240.jsonl'))}
runs={v:load(v) for v in ['B0','B1','B2','B3','B4','B5']}
sc={v:{i['id']:score_one(i, runs[v][i['id']]['response']) for i in inst} for v in runs}
def ex(r): return r['parsed'] and not r.get('abstained') and r['exact']
def binom_two_sided(b,c):
    n=b+c; k=min(b,c)
    p=sum(math.comb(n,j) for j in range(k+1))/2**n*2
    return min(1,p)
ids=[i['id'] for i in inst]; byid={i['id']:i for i in inst}
random.seed(0)
print("paired vs B0: b=only B0 right, c=only Bk right (exact match); McNemar exact p; bootstrap 95% CI of diffs")
for v in ['B1','B2','B3','B4','B5']:
    use=[i for i in ids if not sc[v][i].get('abstained')]
    b=sum(ex(sc['B0'][i]) and not ex(sc[v][i]) for i in use)
    c=sum(ex(sc[v][i]) and not ex(sc['B0'][i]) for i in use)
    p=binom_two_sided(b,c)
    def stats(sample):
        ed=sum(ex(sc[v][i])-ex(sc['B0'][i]) for i in sample)/len(sample)
        rooms=sum(byid[i]['n'] for i in sample)
        u0=sum(sc['B0'][i].get('false_win',byid[i]['n']) for i in sample)/rooms
        u1=sum(sc[v][i].get('false_win',byid[i]['n']) for i in sample)/rooms
        return ed,u1-u0
    boots=[stats([random.choice(use) for _ in use]) for _ in range(4000)]
    e,u=stats(use)
    q=lambda xs:(sorted(xs)[int(.025*len(xs))],sorted(xs)[int(.975*len(xs))])
    print(f"{v}: n={len(use)} b={b} c={c} p={p:.4f} exactdiff={e*100:+.1f} CI{tuple(round(x*100,1) for x in q([x[0] for x in boots]))} unsafediff={u*100:+.1f} CI{tuple(round(x*100,1) for x in q([x[1] for x in boots]))}")
# cost per correct instance
print("\ntokens per exactly-correct instance / seconds per correct (sum latency / #correct)")
for v in runs:
    toks=sum(runs[v][i].get('total_tokens') or 0 for i in ids); lat=sum(runs[v][i].get('latency_seconds') or 0 for i in ids)
    k=sum(ex(sc[v][i]) for i in ids)
    print(v, k, round(toks/k), round(lat/k,1))
# B3 unanimous wrong check
from cops_robbers import parse_response
uw=0
for i in ids:
    calls=[parse_response(c['response']) for c in runs['B3'][i]['calls']]
    t=byid[i]['truth']
    for room in t:
        vals=[c['capture_guaranteed'].get(room) for c in calls if c and isinstance(c.get('capture_guaranteed'),dict)]
        if len(vals)==3 and all(x is not None and x!=t[room] for x in vals): uw+=1
print("\nB3 unanimous wrong rooms:",uw)
# 26-room medians
for v in ['B0','B4']:
    print(v,'26-room median latency',statistics.median(runs[v][i]['latency_seconds'] for i in ids if byid[i]['n']==26))
print('26-room median solve_ms',statistics.median(byid[i]['solve_ms'] for i in ids if byid[i]['n']==26))
# relaxation: safety when restricted; unsafe by size for B0/B4
print("\nunsafe room % by size band")
for v in ['B0','B1','B2','B4']:
    row=[]
    for lo,hi in [(12,14),(16,18),(20,22),(24,26)]:
        s=[i for i in ids if lo<=byid[i]['n']<=hi]
        row.append(round(100*sum(sc[v][i]['false_win'] for i in s)/sum(byid[i]['n'] for i in s),1))
    print(v,row)
print("\nB4 McNemar p =", binom_two_sided(29,71))
print("share of true ESCAPE rooms wrongly called CAPTURE (of answered)")
for v in runs:
    use=[i for i in ids if sc[v][i]['parsed'] and not sc[v][i].get('abstained')]
    esc=sum(sum(1 for r,t in byid[i]['truth'].items() if not t) for i in use)
    print(v, round(100*sum(sc[v][i]['false_win'] for i in use)/esc,1))
print("\nunsafe by band B3,B5(answered)")
for v in ['B3','B5']:
    row=[]
    for lo,hi in [(12,14),(16,18),(20,22),(24,26)]:
        s=[i for i in ids if lo<=byid[i]['n']<=hi and sc[v][i]['parsed'] and not sc[v][i].get('abstained')]
        row.append(round(100*sum(sc[v][i]['false_win'] for i in s)/sum(byid[i]['n'] for i in s),1))
    print(v,row)
print("\nAI cost by size band")
for v in ['B0','B4']:
    for lo,hi in [(12,14),(24,26)]:
        s=[i for i in ids if lo<=byid[i]['n']<=hi]
        print(v,lo,hi,round(statistics.mean(runs[v][i]['total_tokens'] for i in s)),round(statistics.median(runs[v][i]['latency_seconds'] for i in s),1))
