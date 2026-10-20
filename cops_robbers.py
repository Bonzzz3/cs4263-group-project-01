#!/usr/bin/env python3
"""
Cops-and-robbers security patrol: exact solver, instance generator,
LLM prompt builder and scorer.

SOLUTION A (classical): backward induction over joint positions
(an attractor computation, equivalent to minimax on a finite game).

IMPORTANT: the solver and the scorer are EVALUATION instruments only.
They must never be called inside Solution B or inside any safeguard.

Subcommands
  solve     solve one instance given on the command line
  generate  write a set of instances (with ground truth) to JSONL
  prompt    print the B0 prompt for one instance from a JSONL file
  score     score LLM responses against ground truth

Rules of the game (exactly as stated in the prompt):
  - Guards move first, then the intruder, alternating.
  - On its turn each guard stays or moves along one corridor.
  - The intruder stays or moves along one corridor.
  - Caught as soon as any guard shares the intruder's room.
  - The intruder sees the guards and plays optimally.
  - Capture is "guaranteed" if guards can force it in finitely many turns.
"""
import argparse
import itertools
import json
import random
import string
import sys
from collections import defaultdict


# ----------------------------------------------------------------- solver
def neighbours(edges, n):
    """Closed neighbourhoods: a piece may stay or move along a corridor."""
    adj = {v: {v} for v in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return {v: sorted(s) for v, s in adj.items()}


def solve(n, edges, k):
    """
    Return (win, depth) where win[(G, i, t)] is True if the guards can force
    capture from guard multiset G (sorted tuple), intruder room i, and turn t
    (0 = guards to move, 1 = intruder to move). depth is the number of
    half-moves needed under optimal play (capture states have depth 0).
    """
    adj = neighbours(edges, n)
    guard_sets = list(itertools.combinations_with_replacement(range(n), k))

    def guard_moves(G):
        opts = [adj[g] for g in G]
        return {tuple(sorted(m)) for m in itertools.product(*opts)}

    gmoves = {G: guard_moves(G) for G in guard_sets}

    depth = {}
    for G in guard_sets:
        for i in range(n):
            if i in G:
                depth[(G, i, 0)] = 0
                depth[(G, i, 1)] = 0

    changed = True
    level = 0
    while changed:
        changed = False
        level += 1
        new = {}
        for G in guard_sets:
            for i in range(n):
                if i in G:
                    continue
                # guard turn: some move reaches a winning intruder-turn state
                if (G, i, 0) not in depth:
                    best = None
                    for G2 in gmoves[G]:
                        d = depth.get((G2, i, 1))
                        if d is not None and (best is None or d < best):
                            best = d
                    if best is not None:
                        new[(G, i, 0)] = best + 1
                # intruder turn: every move reaches a winning guard-turn state
                if (G, i, 1) not in depth:
                    worst = 0
                    ok = True
                    for i2 in adj[i]:
                        d = depth.get((G, i2, 0))
                        if d is None:
                            ok = False
                            break
                        worst = max(worst, d)
                    if ok:
                        new[(G, i, 1)] = worst + 1
        for key, d in new.items():
            if key not in depth:
                depth[key] = d
                changed = True
    return depth


def ground_truth(n, edges, guard_start, k=None):
    """Per intruder start room: (capture_guaranteed, rounds_if_guaranteed)."""
    k = len(guard_start)
    G = tuple(sorted(guard_start))
    depth = solve(n, edges, k)
    out = {}
    for i in range(n):
        d = depth.get((G, i, 0))
        out[i] = (d is not None, d)
    return out


# -------------------------------------------------------------- generator
def room_names(n):
    return list(string.ascii_uppercase[:n]) if n <= 26 else [f"R{i}" for i in range(n)]


def is_connected(n, edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def random_graph(n, extra, rng):
    """Random spanning tree plus `extra` extra edges (creates loops)."""
    edges = set()
    nodes = list(range(n))
    rng.shuffle(nodes)
    for idx in range(1, n):
        a = nodes[idx]
        b = nodes[rng.randrange(idx)]
        edges.add((min(a, b), max(a, b)))
    attempts = 0
    while extra > 0 and attempts < 1000:
        attempts += 1
        a, b = rng.sample(range(n), 2)
        e = (min(a, b), max(a, b))
        if e not in edges:
            edges.add(e)
            extra -= 1
    return sorted(edges)


def gen_petersen(m, step=2):
    """Generalised Petersen graph GP(m, step): 2m rooms, 3 corridors each."""
    e = set()
    for i in range(m):
        for a, b in [(i, (i + 1) % m), (i, m + i), (m + i, m + (i + step) % m)]:
            e.add((min(a, b), max(a, b)))
    return 2 * m, sorted(e)


def relabel(n, edges, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    return sorted({(min(perm[a], perm[b]), max(perm[a], perm[b])) for a, b in edges})


def core_tree(total, rng, core_m=5):
    """
    Petersen core GP(core_m, 2) (10 rooms for core_m=5) plus pendant rooms
    attached one by one to a random existing room, until `total` rooms.
    Two guards cannot trap an intruder inside the core, but the pendant trees
    create starts where capture needs a multi-step chase, so labels depend on
    real lookahead instead of simple adjacency.
    """
    n0, base = gen_petersen(core_m, 2)
    edges = set(base)
    nodes = list(range(n0))
    for v in range(n0, total):
        a = rng.choice(nodes)
        edges.add((min(a, v), max(a, v)))
        nodes.append(v)
    return total, sorted(edges)


def make_instance(idx, n, extra, k, rng, family="random"):
    if family == "structured":
        # Plain generalised Petersen graph; `n` is the GP parameter m and
        # rooms = 2m. NOTE: with 2 guards every won room is won within one
        # half-move (guard room or adjacent), so this family is too easy to
        # separate real reasoning from an adjacency check. Kept for reference;
        # the dataset uses "coretree" instead.
        n, base = gen_petersen(n, 2)
        edges = relabel(n, base, rng)
    elif family == "coretree":
        n, base = core_tree(n, rng)
        edges = relabel(n, base, rng)
    else:
        edges = random_graph(n, extra, rng)
    guard_start = [rng.randrange(n) for _ in range(k)]
    truth = ground_truth(n, edges, guard_start)
    names = room_names(n)
    return {
        "id": f"inst{idx:04d}",
        "n": n,
        "k": k,
        "extra_edges": extra,
        "edges": edges,
        "guard_start": guard_start,
        "truth": {names[i]: truth[i][0] for i in range(n)},
        "rounds": {names[i]: truth[i][1] for i in range(n)},
    }


def build_prompt(inst):
    names = room_names(inst["n"])
    corridors = ", ".join(f"{names[a]}-{names[b]}" for a, b in inst["edges"])
    starts = ", ".join(names[g] for g in inst["guard_start"])
    room_list = ", ".join(names)
    answer_fields = ", ".join(f'"{r}": true/false' for r in names)
    return f"""You are given a pursuit game on an undirected graph of rooms.

RULES
- Rooms: {room_list}.
- Corridors (edges): {corridors}.
- There are {inst['k']} guard(s) and 1 intruder. All positions are rooms.
- Turn order: the guards move first. Then the intruder moves. Then the
  guards move again, and so on.
- On a guard turn, EACH guard either stays or moves along one corridor to
  an adjacent room. Guards may share a room.
- On the intruder's turn, the intruder either stays or moves along one
  corridor to an adjacent room.
- The intruder is caught as soon as any guard is in the same room as the
  intruder, whether the guard moved there or the intruder did.
- The intruder always sees the guards' positions and plays the best
  possible strategy to avoid capture forever.
- "Guarantee capture" means the guards can force a capture in finitely
  many turns against every possible intruder strategy.

QUESTION
The guards start in rooms {starts}. The intruder may start in any room.
For each possible intruder start room ({room_list}), decide whether the
guards can guarantee capture.

Respond with ONLY this JSON, no other text:
{{
  "capture_guaranteed": {{{answer_fields}}},
  "confidence": a number from 0 to 1
}}"""


# ----------------------------------------------------------------- scorer
def parse_response(text):
    """Extract the first JSON object from a model response, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:j + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def score_one(inst, response_text):
    obj = parse_response(response_text)
    names = room_names(inst["n"])
    if not obj or "capture_guaranteed" not in obj:
        return {"parsed": False}
    pred = obj["capture_guaranteed"]
    if not all(r in pred and isinstance(pred[r], bool) for r in names):
        return {"parsed": False}
    truth = inst["truth"]
    false_win = sum(1 for r in names if pred[r] and not truth[r])    # unsafe
    false_loss = sum(1 for r in names if (not pred[r]) and truth[r])  # wasteful
    exact = false_win == 0 and false_loss == 0
    conf = obj.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None
    return {
        "parsed": True,
        "exact": exact,
        "false_win": false_win,
        "false_loss": false_loss,
        "rooms": len(names),
        "confidence": conf,
    }


def summarise(results):
    total = len(results)
    parsed = [r for r in results if r["parsed"]]
    out = {"instances": total, "unparseable": total - len(parsed)}
    if not parsed:
        return out
    out["exact_match_rate"] = sum(r["exact"] for r in parsed) / len(parsed)
    out["instances_with_unsafe_claim"] = (
        sum(1 for r in parsed if r["false_win"] > 0) / len(parsed))
    out["unsafe_room_rate"] = (
        sum(r["false_win"] for r in parsed) / sum(r["rooms"] for r in parsed))
    out["wasteful_room_rate"] = (
        sum(r["false_loss"] for r in parsed) / sum(r["rooms"] for r in parsed))
    with_conf = [r for r in parsed if r["confidence"] is not None]
    if with_conf:
        # Brier score of stated confidence against exact-match correctness
        out["brier"] = sum((r["confidence"] - (1.0 if r["exact"] else 0.0)) ** 2
                           for r in with_conf) / len(with_conf)
        out["mean_confidence"] = sum(r["confidence"] for r in with_conf) / len(with_conf)
        wrong = [r for r in with_conf if not r["exact"]]
        out["mean_confidence_when_wrong"] = (
            sum(r["confidence"] for r in wrong) / len(wrong)) if wrong else None
    return out


# -------------------------------------------------------------------- CLI
def cmd_solve(a):
    edges = [tuple(map(int, e.split("-"))) for e in a.edges.split(",")]
    guards = [int(x) for x in a.guards.split(",")]
    res = ground_truth(a.n, edges, guards)
    names = room_names(a.n)
    for i in range(a.n):
        ok, d = res[i]
        print(f"intruder starts {names[i]}: "
              f"{'CAPTURE GUARANTEED' if ok else 'INTRUDER ESCAPES'}"
              + (f" (within {d} half-moves)" if ok else ""))


def cmd_generate(a):
    rng = random.Random(a.seed)
    sizes = [int(x) for x in a.sizes.split(",")]
    if a.family == "structured":
        sizes = [m for m in sizes if m != 6]
    count = 0
    with open(a.out, "w") as f:
        for n in sizes:
            made, tries = 0, 0
            while made < a.per_size and tries < a.per_size * 200:
                tries += 1
                extra = rng.randint(0, max(1, n // 2))
                inst = make_instance(count, n, extra, a.guards, rng, a.family)
                vals = list(inst["truth"].values())
                if a.mixed and (all(vals) or not any(vals)):
                    continue  # keep only instances with both outcomes present
                f.write(json.dumps(inst) + "\n")
                made += 1
                count += 1
            print(f"n={n}: wrote {made} instances", file=sys.stderr)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_prompt(a):
    insts = load(a.instances)
    inst = next(i for i in insts if i["id"] == a.id) if a.id else insts[0]
    print(build_prompt(inst))


def cmd_score(a):
    insts = {i["id"]: i for i in load(a.instances)}
    results, by_n = [], defaultdict(list)
    for line in open(a.responses):
        if not line.strip():
            continue
        rec = json.loads(line)  # {"id": ..., "response": "..."}
        inst = insts[rec["id"]]
        r = score_one(inst, rec["response"])
        results.append(r)
        by_n[inst["n"]].append(r)
    print("OVERALL", json.dumps(summarise(results), indent=2))
    for n in sorted(by_n):
        print(f"\nn={n}", json.dumps(summarise(by_n[n]), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--edges", required=True, help="e.g. 0-1,1-2,2-3,3-0")
    s.add_argument("--guards", required=True, help="e.g. 0,0")
    s.set_defaults(fn=cmd_solve)

    g = sub.add_parser("generate")
    g.add_argument("--sizes", default="6,8,10,12")
    g.add_argument("--per-size", type=int, default=20)
    g.add_argument("--guards", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--family", choices=["random", "coretree", "structured"],
                   default="random",
                   help="random: tree plus extra loops (use with 1 guard). "
                        "coretree: Petersen core plus pendant trees, size = total "
                        "rooms >= 12 (use with 2 guards; this is the dataset family). "
                        "structured: plain generalised Petersen graphs, rooms = 2*size "
                        "(too easy: every win is within one half-move)")
    g.add_argument("--mixed", action="store_true",
                   help="keep only instances where some starts win and some lose")
    g.add_argument("--out", default="instances.jsonl")
    g.set_defaults(fn=cmd_generate)

    pr = sub.add_parser("prompt")
    pr.add_argument("--instances", default="instances.jsonl")
    pr.add_argument("--id")
    pr.set_defaults(fn=cmd_prompt)

    sc = sub.add_parser("score")
    sc.add_argument("--instances", default="instances.jsonl")
    sc.add_argument("--responses", required=True,
                    help='JSONL with {"id": ..., "response": "<model text>"}')
    sc.set_defaults(fn=cmd_score)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
