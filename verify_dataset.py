#!/usr/bin/env python3
"""
Independent verification of the dataset ground truth.

The solver in cops_robbers.py finds the winning region by a fixpoint loop.
This script does NOT trust that loop. It takes the solver's output and checks
it against two certificates, written separately:

  CAPTURE certificate (ranking function): every state claimed won with depth
    d > 0 has a guard move (guard turn) or only intruder moves (intruder turn)
    leading to states with strictly smaller depth, and depth 0 means a guard
    shares the intruder's room.  Following the ranking, capture is forced in
    at most d half-moves, whatever the intruder does.

  ESCAPE certificate (invariant): the set U of states NOT claimed won is
    closed in the intruder's favour: from every guard-turn state in U, EVERY
    guard move lands in U (and never on the intruder), and from every
    intruder-turn state in U, SOME intruder move lands in U.  By induction the
    intruder survives forever from any state in U.

Both certificates are checked on the full state space of each instance, plus
dataset-level invariants (connected graphs, mixed outcomes, disjoint splits,
labels agree with the certified regions).
"""
import itertools
import json
import sys

from cops_robbers import room_names, solve


def closed_nbrs(n, edges):
    adj = [{v} for v in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def guard_moves(G, adj):
    return {tuple(sorted(m)) for m in itertools.product(*[adj[g] for g in G])}


def verify_instance(inst):
    n, k = inst["n"], inst["k"]
    edges = [tuple(e) for e in inst["edges"]]
    adj = closed_nbrs(n, edges)
    errors = []

    # graph sanity: no self loops, connected
    if any(a == b for a, b in edges):
        errors.append("self loop")
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    if len(seen) != n:
        errors.append("graph not connected")

    depth = solve(n, edges, k)
    guard_sets = list(itertools.combinations_with_replacement(range(n), k))

    for G in guard_sets:
        gm = guard_moves(G, adj)
        for i in range(n):
            # ---- capture certificate for every claimed-won state
            for t in (0, 1):
                d = depth.get((G, i, t))
                if d is None:
                    continue
                if (d == 0) != (i in G):
                    errors.append(f"depth-0 mismatch at {(G, i, t)}")
                    continue
                if d == 0:
                    continue
                if t == 0:
                    ok = any(depth.get((G2, i, 1)) == d - 1 for G2 in gm)
                else:
                    ds = [depth.get((G, i2, 0)) for i2 in adj[i]]
                    ok = all(x is not None and x <= d - 1 for x in ds) and (d - 1) in ds
                if not ok:
                    errors.append(f"capture certificate fails at {(G, i, t)} d={d}")
            # ---- escape certificate for every state outside the winning region
            if i in G:
                continue
            if (G, i, 0) not in depth:
                if any((G2, i, 1) in depth for G2 in gm):
                    errors.append(f"escape invariant fails (guard turn) {(G, i)}")
            if (G, i, 1) not in depth:
                if not any((G, i2, 0) not in depth and i2 not in G for i2 in adj[i]):
                    errors.append(f"escape invariant fails (intruder turn) {(G, i)}")

    # labels in the file must agree with the certified regions
    S = tuple(sorted(inst["guard_start"]))
    names = room_names(n)
    for r in range(n):
        d = depth.get((S, r, 0))
        if (d is not None) != inst["truth"][names[r]]:
            errors.append(f"label mismatch for room {names[r]}")
        if inst["rounds"][names[r]] != d:
            errors.append(f"rounds mismatch for room {names[r]}")
    vals = list(inst["truth"].values())
    if all(vals) or not any(vals):
        errors.append("outcomes not mixed")
    return errors


def main():
    ids, bad, total = set(), 0, 0
    files = {"dev": "dataset/instances_dev.jsonl", "test": "dataset/instances_test.jsonl"}
    for split, path in files.items():
        for line in open(path):
            inst = json.loads(line)
            total += 1
            if inst["id"] in ids:
                print("DUPLICATE ID", inst["id"])
                bad += 1
            ids.add(inst["id"])
            if inst["split"] != split:
                print("WRONG SPLIT", inst["id"])
                bad += 1
            errs = verify_instance(inst)
            if errs:
                bad += 1
                print(inst["id"], errs[:3])
    print(f"verified {total} instances, {bad} with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
