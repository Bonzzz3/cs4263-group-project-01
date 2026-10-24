#!/usr/bin/env python3
"""
Build the shared dataset for the cops-and-robbers project.

One graph family with a fixed guard count k = 2, two splits:
  coretree-2g    2 guards, Petersen core (10 rooms) plus pendant trees up to
                 the target room count, rooms shuffled. A plain Petersen graph
                 was rejected: with 2 guards every win is within 1 half-move,
                 so labels reduce to an adjacency check.
  dev   : small split for the one-day pilot and prompt development
  test  : held-out split for all reported results (never tune prompts on it)

Every instance keeps only graphs where SOME intruder starts are won and SOME
are lost, so a model cannot score well by answering all-true or all-false.

Outputs (in ./dataset):
  instances_dev.jsonl, instances_test.jsonl   instance + exact ground truth
  prompts_B0_dev.jsonl, prompts_B0_test.jsonl {"id", "prompt"} baseline prompts
  dataset_stats.csv                           composition per split/family/size
"""
import csv
import json
import os
import random
import time

from cops_robbers import build_prompt, make_instance

CORETREE_SIZES = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # total rooms
PER_SIZE = {"dev": 10, "test": 30}
SEEDS = {"dev": 1000, "test": 2000}
OUT = "dataset"


def build_split(split, seen):
    rng = random.Random(SEEDS[split])
    rows = []
    plan = [("coretree-2g", 2, CORETREE_SIZES)]
    for fam_name, k, sizes in plan:
        for size in sizes:
            made = tries = 0
            while made < PER_SIZE[split]:
                tries += 1
                if tries > 5000:
                    raise RuntimeError(f"could not fill {split} {fam_name} {size}")
                t0 = time.perf_counter()
                inst = make_instance(0, size, k, rng)
                solve_ms = (time.perf_counter() - t0) * 1000
                vals = list(inst["truth"].values())
                if all(vals) or not any(vals):
                    continue
                key = (inst["n"], tuple(map(tuple, inst["edges"])),
                       tuple(sorted(inst["guard_start"])))
                if key in seen:
                    continue
                seen.add(key)
                made += 1
                inst["id"] = f"{split}-{fam_name}-r{inst['n']:02d}-{made:03d}"
                inst["split"] = split
                inst["family"] = fam_name
                inst["corridors"] = len(inst["edges"])
                inst["escape_rooms"] = sum(1 for v in vals if not v)
                inst["max_half_moves"] = max(
                    r for r in inst["rounds"].values() if r is not None)
                inst["solve_ms"] = round(solve_ms, 2)
                rows.append(inst)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    seen = set()
    stats = []
    for split in ("dev", "test"):
        rows = build_split(split, seen)
        with open(f"{OUT}/instances_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(f"{OUT}/prompts_B0_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "prompt": build_prompt(r)}) + "\n")
        groups = {}
        for r in rows:
            groups.setdefault((r["family"], r["n"]), []).append(r)
        for (fam, n), g in sorted(groups.items()):
            cnt = len(g)
            stats.append({
                "split": split, "family": fam, "guards": g[0]["k"], "rooms": n,
                "instances": cnt,
                "mean_corridors": round(sum(x["corridors"] for x in g) / cnt, 1),
                "mean_escape_rooms": round(sum(x["escape_rooms"] for x in g) / cnt, 2),
                "mean_escape_fraction": round(
                    sum(x["escape_rooms"] / x["n"] for x in g) / cnt, 3),
                "max_half_moves": max(x["max_half_moves"] for x in g),
                "mean_solve_ms": round(sum(x["solve_ms"] for x in g) / cnt, 1),
            })
    with open(f"{OUT}/dataset_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    print("wrote", sum(s["instances"] for s in stats), "instances")


if __name__ == "__main__":
    main()
