#!/usr/bin/env python3
"""
Runtime and state counts of the Solution A solver against the number of
rooms n, with k = 2 guards as in the dataset.

For each n, the script generates graphs from the dataset's family (a 10-room
Petersen-type core plus attached rooms, cops_robbers.core_tree), solves each
one several times and keeps the median runtime. One row per graph goes to
results/solver_benchmark.csv, and a summary per n is printed, with fitted
growth exponents (quantity ~ n^x).

Besides runtime and state counts, it records the two quantities that explain
the runtime:
  transitions  moves out of every non-capture state (the size of the game)
  passes       sweeps the solver makes over all states. Pass d adds the
               states captured in d half-moves, and one last pass finds
               nothing, so passes = longest capture + 1.
The solver checks about passes * transitions moves in total, so runtime
divided by that product (ns_per_move_check, in nanoseconds) should stay
roughly constant across sizes.

Runtimes depend on the machine and Python version, so report both with the
results. Run: uv run python bench_solver.py
"""
import argparse
import csv
import itertools
import math
import os
import platform
import random
import statistics
import time

from cops_robbers import core_tree, neighbours, solve

K = 2  # fixed for the project


def count_transitions(n, edges):
    """Moves out of every non-capture state: distinct joint guard moves on the
    guards' turn plus intruder moves (including staying) on the intruder's."""
    adj = neighbours(edges, n)
    total = 0
    for G in itertools.combinations_with_replacement(range(n), K):
        guard_moves = len({tuple(sorted(m))
                           for m in itertools.product(*(adj[g] for g in G))})
        for i in range(n):
            if i not in G:
                total += guard_moves + len(adj[i])
    return total


def fit_exponent(xs, ys):
    """Least-squares slope of log(y) against log(x)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    return (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
            / sum((a - mx) ** 2 for a in lx))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="12,16,20,24,28,32,36,40",
                   help="room counts to test (each at least 10)")
    p.add_argument("--graphs", type=int, default=5, help="graphs per size")
    p.add_argument("--repeats", type=int, default=3, help="timed solves per graph")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/solver_benchmark.csv")
    a = p.parse_args()

    sizes = [int(s) for s in a.sizes.split(",")]
    print(f"Python {platform.python_version()} on {platform.platform()}, "
          f"k = {K}, {a.graphs} graphs x {a.repeats} repeats per size\n")

    rows = []
    for n in sizes:
        rng = random.Random(a.seed * 1000 + n)
        for g in range(a.graphs):
            _, edges = core_tree(n, rng)
            times = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                depth = solve(n, edges, K)
                times.append((time.perf_counter() - t0) * 1000)
            ms = statistics.median(times)
            max_depth = max(depth.values())
            transitions = count_transitions(n, edges)
            rows.append({
                "n": n,
                "k": K,
                "graph": g,
                "corridors": len(edges),
                "states": math.comb(n + K - 1, K) * n * 2,
                "won_states": len(depth),
                "transitions": transitions,
                "max_depth": max_depth,
                "passes": max_depth + 1,
                "median_ms": round(ms, 2),
                "ns_per_move_check": round(ms * 1e6 / ((max_depth + 1) * transitions), 1),
            })

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'rooms':>5} {'states':>8} {'transitions':>11} {'passes':>6} "
          f"{'runtime ms':>10} {'ns per move check':>17}")
    summary = []
    for n in sizes:
        r = [x for x in rows if x["n"] == n]
        mean = lambda key: statistics.fmean(x[key] for x in r)  # noqa: E731
        summary.append((n, r[0]["states"], mean("transitions"),
                        mean("passes"), mean("median_ms")))
        print(f"{n:>5} {r[0]['states']:>8} {mean('transitions'):>11.0f} "
              f"{mean('passes'):>6.1f} {mean('median_ms'):>10.1f} "
              f"{mean('ns_per_move_check'):>17.1f}")

    ns = [s[0] for s in summary]
    print()
    for label, col in [("states", 1), ("transitions", 2), ("passes", 3), ("runtime", 4)]:
        print(f"{label:>11} ~ n^{fit_exponent(ns, [s[col] for s in summary]):.2f}")
    print("(states is exactly n^2 (n+1), so its exponent tends to 3)")
    print(f"\nwrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
