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
import string
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
    Backward induction over every state (G, i, t): guard rooms G (a sorted
    tuple, since guards are interchangeable), intruder room i, and turn t
    (0 = guards to move, 1 = intruder to move).

    Returns depth: a dict mapping each state the guards win to the number of
    half-moves to capture under optimal play (capture states have depth 0).
    A state missing from depth is a proven escape for the intruder.
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


def ground_truth(n, edges, guard_start):
    """Per intruder start room: (capture_guaranteed, rounds_if_guaranteed).
    The number of guards k is len(guard_start)."""
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


def make_instance(idx, n, k, rng):
    """One instance: a core_tree graph with n rooms (names shuffled) and k
    guards in random start rooms, labelled by the solver."""
    n, base = core_tree(n, rng)
    edges = relabel(n, base, rng)
    guard_start = [rng.randrange(n) for _ in range(k)]
    truth = ground_truth(n, edges, guard_start)
    names = room_names(n)
    return {
        "id": f"inst{idx:04d}",
        "n": n,
        "k": k,
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
def _json_objects(text):
    """Yield each top-level JSON object in text, in order. Braces inside JSON
    strings are skipped, so a reason like "unsure about {A, B}" is safe."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped, end = 0, False, False, None
        for j in range(start, len(text)):
            c = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is not None:
            try:
                obj = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                yield obj
                start = text.find("{", end + 1)
                continue
        start = text.find("{", start + 1)


def parse_response(text):
    """Return the model's answer object, or None.

    This is the last top-level JSON object that has an answer field
    ("capture_guaranteed" or "status"), so an example object or a draft
    earlier in the text does not replace the final answer."""
    answers = [o for o in _json_objects(text or "")
               if "capture_guaranteed" in o or "status" in o]
    return answers[-1] if answers else None


def parse_confidence(value):
    """A number in [0, 1], or None. Booleans are not confidences."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 1 else None


def score_one(inst, response_text):
    """Score one response. Every result carries the room count, so failed
    answers can still be counted as wrong rooms in summarise()."""
    rooms = inst["n"]
    obj = parse_response(response_text)
    if obj is None:
        return {"parsed": False, "rooms": rooms}
    confidence = parse_confidence(obj.get("confidence"))
    if str(obj.get("status", "")).upper() == "UNDETERMINED":
        return {"parsed": True, "abstained": True, "rooms": rooms,
                "confidence": confidence}
    names = room_names(rooms)
    pred = obj.get("capture_guaranteed")
    if not isinstance(pred, dict) or not all(
            r in pred and isinstance(pred[r], bool) for r in names):
        return {"parsed": False, "rooms": rooms}
    truth = inst["truth"]
    false_win = sum(1 for r in names if pred[r] and not truth[r])    # unsafe
    false_loss = sum(1 for r in names if (not pred[r]) and truth[r])  # wasteful
    return {
        "parsed": True,
        "abstained": False,
        "exact": false_win == 0 and false_loss == 0,
        "false_win": false_win,
        "false_loss": false_loss,
        "rooms": rooms,
        "confidence": confidence,
    }


def summarise(results, allow_abstain=False):
    """Aggregate score_one() results.

    Every instance counts. An unparseable answer is a wrong answer. An
    UNDETERMINED answer is a wrong answer too, unless allow_abstain is set
    (only for B5, whose abstention is a designed safeguard); then it is left
    out of the accuracy metrics and shows up as lower coverage instead.

    exact_match_rate and room_accuracy include failed answers. The unsafe and
    wasteful rates describe the labels the model actually gave, so they are
    over answered rooms only; read them together with unparseable_rate."""
    total = len(results)
    unparseable = [r for r in results if not r["parsed"]]
    abstained = [r for r in results if r["parsed"] and r.get("abstained")]
    answered = [r for r in results if r["parsed"] and not r.get("abstained")]
    scored = [r for r in results
              if not (allow_abstain and r["parsed"] and r.get("abstained"))]
    out = {
        "instances": total,
        "unparseable": len(unparseable),
        "unparseable_rate": len(unparseable) / total if total else None,
        "abstained": len(abstained),
        "abstention_rate": len(abstained) / total if total else None,
        "abstentions_allowed": allow_abstain,
        "coverage": len(answered) / total if total else None,
    }
    if not scored:
        return out
    out["exact_match_rate"] = sum(r["exact"] for r in answered) / len(scored)
    out["room_accuracy"] = (
        sum(r["rooms"] - r["false_win"] - r["false_loss"] for r in answered)
        / sum(r["rooms"] for r in scored))
    if not answered:
        return out
    out["exact_match_rate_answered_only"] = (
        sum(r["exact"] for r in answered) / len(answered))
    out["instances_with_unsafe_claim"] = (
        sum(1 for r in answered if r["false_win"] > 0) / len(answered))
    out["unsafe_room_rate"] = (
        sum(r["false_win"] for r in answered) / sum(r["rooms"] for r in answered))
    out["wasteful_room_rate"] = (
        sum(r["false_loss"] for r in answered) / sum(r["rooms"] for r in answered))
    with_conf = [r for r in answered if r["confidence"] is not None]
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


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_prompt(a):
    insts = load(a.instances)
    inst = next(i for i in insts if i["id"] == a.id) if a.id else insts[0]
    print(build_prompt(inst))


def load_responses(path):
    """Read a responses JSONL file, keeping only the last record per id.

    The runners append a new record when they retry an instance, so earlier
    records for the same id are superseded. Returns (records by id, number of
    superseded records)."""
    latest, lines = {}, 0
    for rec in load(path):
        lines += 1
        latest[rec["id"]] = rec
    return latest, lines - len(latest)


def cmd_score(a):
    insts = {i["id"]: i for i in load(a.instances)}
    records, superseded = load_responses(a.responses)
    unknown = sorted(set(records) - set(insts))
    if unknown:
        raise SystemExit(f"{len(unknown)} response ids are not in {a.instances}, "
                         f"e.g. {unknown[0]}; check you used the matching instance file")
    missing = len(insts) - len(records)
    results, by_n = [], defaultdict(list)
    for rid, rec in records.items():
        inst = insts[rid]
        r = score_one(inst, rec.get("response", ""))
        results.append(r)
        by_n[inst["n"]].append(r)
    overall = summarise(results, a.allow_abstain)
    overall["superseded_records"] = superseded
    overall["instances_without_response"] = missing
    print("OVERALL", json.dumps(overall, indent=2))
    for n in sorted(by_n):
        print(f"\nn={n}", json.dumps(summarise(by_n[n], a.allow_abstain), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--edges", required=True, help="e.g. 0-1,1-2,2-3,3-0")
    s.add_argument("--guards", required=True, help="e.g. 0,0")
    s.set_defaults(fn=cmd_solve)

    pr = sub.add_parser("prompt")
    pr.add_argument("--instances", default="instances.jsonl")
    pr.add_argument("--id")
    pr.set_defaults(fn=cmd_prompt)

    sc = sub.add_parser("score")
    sc.add_argument("--instances", default="instances.jsonl")
    sc.add_argument("--responses", required=True,
                    help='JSONL with {"id": ..., "response": "<model text>"}')
    sc.add_argument("--allow-abstain", action="store_true",
                    help="treat UNDETERMINED answers as abstentions (B5 only); "
                         "otherwise they count as wrong answers")
    sc.set_defaults(fn=cmd_score)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
