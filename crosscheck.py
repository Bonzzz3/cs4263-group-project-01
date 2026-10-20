#!/usr/bin/env python3
"""Independent re-implementation of the game, written from the rules text only.

It does not import the solver. It builds the winning sets W_m literally from the
definition in the report (section 1) and compares, for every instance in the
dataset, the first m at which each (guards, intruder, guard-turn) state is won
against the labels and 'rounds' stored in the dataset.
Run: python3 crosscheck.py   (needs dataset/ built)"""
import itertools, json, sys

def labels(n, edges, k, S):
    N = [{v} for v in range(n)]
    for a, b in edges:
        N[a].add(b); N[b].add(a)
    guards = list(itertools.combinations_with_replacement(range(n), k))
    gmoves = {g: {tuple(sorted(m)) for m in itertools.product(*[N[x] for x in g])} for g in guards}
    W = {(g, i, t) for g in guards for i in range(n) for t in "GI" if i in g}
    first = {s: 0 for s in W}
    m = 0
    while True:
        m += 1
        new = set()
        for g in guards:
            for i in range(n):
                if i in g:
                    continue
                if (g, i, "G") not in W and any((g2, i, "I") in W for g2 in gmoves[g]):
                    new.add((g, i, "G"))
                if (g, i, "I") not in W and all((g, j, "G") in W for j in N[i]):
                    new.add((g, i, "I"))
        if not new:
            break
        for s in new:
            first[s] = m
        W |= new
    S = tuple(sorted(S))
    return [first.get((S, r, "G")) for r in range(n)]

bad = total = 0
for split in ("dev", "test"):
    for line in open(f"dataset/instances_{split}.jsonl"):
        inst = json.loads(line); total += 1
        got = labels(inst["n"], [tuple(e) for e in inst["edges"]], inst["k"], inst["guard_start"])
        exp = [inst["rounds"][nm] for nm in inst["truth"]]
        if got != exp:
            bad += 1; print("MISMATCH", inst["id"])
# hand-checked examples from section 1: 4-room loop A-B-C-D
print("loop, 1 guard at A:", labels(4, [(0,1),(1,2),(2,3),(3,0)], 1, [0]))
print("loop, 2 guards at A:", labels(4, [(0,1),(1,2),(2,3),(3,0)], 2, [0,0]))
print(f"cross-checked {total} instances, {bad} mismatches")
sys.exit(1 if bad else 0)
