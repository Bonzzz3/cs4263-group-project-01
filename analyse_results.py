#!/usr/bin/env python3
r"""Turn Solution B response files into one results table.

For each response file (one run: a variant, provider, model and effort) it
scores every instance in the instance file and reports:

  accuracy   exact-match rate with a 95% Wilson interval, and room accuracy
             with a 95% bootstrap interval over instances
  errors     unsafe and wasteful room rates, unparseable answers
  B5         coverage (abstentions are only allowed for B5)
  confidence Brier score and mean confidence when wrong
  cost       API calls, total and reasoning tokens, latency per instance, and
             USD cost per instance where the provider reports it (OpenRouter)

The same metrics are shown for two trivial baselines on the same instances
(answer "capture" everywhere; answer "capture" only next to a guard), and
broken down by room count and by the longest optimal chase, the complexity
measures the evaluation plan uses.

An instance with no usable answer (missing, API error, unparseable) counts as
wrong, so an incomplete run cannot look better than a complete one; the
table says how many there were so you can rerun the runner to fill them in.

This is an evaluation tool: it reads the solver's labels and must never be
used inside Solution B.

Example:
    uv run python analyse_results.py --instances dataset/instances_test_80.jsonl \\
        work/openai_b0_test80.jsonl work/openai_b1_test80.jsonl \\
        --out results/solution_b_test80.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from cops_robbers import load, load_responses, neighbours, room_names, score_one, summarise

SIZE_BANDS = [(12, 14), (16, 18), (20, 22), (24, 26)]
DEPTH_BANDS = [(0, 5), (7, 7), (9, 99)]
BOOTSTRAP_SAMPLES = 2000


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_room_accuracy(results: list[dict], allow_abstain: bool) -> tuple[float, float]:
    """95% percentile interval for room accuracy, resampling instances."""
    scored = [r for r in results
              if not (allow_abstain and r["parsed"] and r.get("abstained"))]
    if not scored:
        return (math.nan, math.nan)
    pairs = [((r["rooms"] - r["false_win"] - r["false_loss"])
              if r["parsed"] and not r.get("abstained") else 0, r["rooms"])
             for r in scored]
    rng = random.Random(0)
    stats = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        stats.append(sum(c for c, _ in sample) / sum(n for _, n in sample))
    stats.sort()
    return (stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1])


def baseline_answers(inst: dict) -> dict[str, str]:
    """Two answers that need no reasoning, as JSON strings."""
    names = room_names(inst["n"])
    nbrs = neighbours([tuple(e) for e in inst["edges"]], inst["n"])
    near = {v for g in inst["guard_start"] for v in nbrs[g]}
    return {
        "baseline: all capture": json.dumps(
            {"capture_guaranteed": {r: True for r in names}}),
        "baseline: capture next to a guard": json.dumps(
            {"capture_guaranteed": {names[i]: i in near for i in range(inst["n"])}}),
    }


def run_label(records: dict[str, dict], path: Path) -> str:
    first = next(iter(records.values()), {})
    variant = first.get("variant", "B0")
    provider = first.get("provider", "openai")
    model = first.get("model", "?")
    effort = first.get("reasoning_effort")
    return f"{variant} {provider}/{model}" + (f" ({effort})" if effort else "") \
        if first else path.stem


def score_run(instances: list[dict], records: dict[str, dict]) -> list[dict]:
    """score_one() for every instance; a missing record counts as unparseable."""
    out = []
    for inst in instances:
        rec = records.get(inst["id"])
        result = score_one(inst, rec.get("response", "") if rec else "")
        result["missing"] = rec is None
        result["api_error"] = bool(rec) and rec.get("status") == "error"
        out.append(result)
    return out


def cost(records: list[dict]) -> dict:
    def mean(field):
        values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
        return statistics.fmean(values) if values else None

    latencies = [r["latency_seconds"] for r in records
                 if isinstance(r.get("latency_seconds"), (int, float))]
    return {
        "calls_per_instance": mean("call_count") or (1.0 if records else None),
        "tokens_per_instance": mean("total_tokens"),
        "reasoning_tokens_per_instance": mean("reasoning_tokens"),
        "cost_usd_per_instance": mean("cost_usd"),
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "truncated": sum(r.get("status") == "truncated" or bool(r.get("truncated_calls"))
                         for r in records),
    }


def metrics(results: list[dict], allow_abstain: bool) -> dict:
    s = summarise(results, allow_abstain)
    scored = [r for r in results
              if not (allow_abstain and r["parsed"] and r.get("abstained"))]
    exact = sum(1 for r in results if r["parsed"] and not r.get("abstained") and r["exact"])
    s["exact_ci_low"], s["exact_ci_high"] = wilson(exact, len(scored))
    s["room_ci_low"], s["room_ci_high"] = bootstrap_room_accuracy(results, allow_abstain)
    s["missing"] = sum(r["missing"] for r in results)
    s["api_errors"] = sum(r["api_error"] for r in results)
    return s


def in_band(value: int, band: tuple[int, int]) -> bool:
    return band[0] <= value <= band[1]


def band_name(band: tuple[int, int], unit: str) -> str:
    lo, hi = band
    if lo == hi:
        return f"{lo} {unit}"
    if hi >= 99:
        return f"{lo}+ {unit}"
    return f"{lo}-{hi} {unit}" if lo else f"<={hi} {unit}"


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("responses", type=Path, nargs="+",
                        help="one responses JSONL file per run")
    parser.add_argument("--out", type=Path, default=None, help="write all rows to CSV")
    args = parser.parse_args()

    instances = load(args.instances)
    ids = {i["id"] for i in instances}
    runs = []  # (label, results, allow_abstain, cost)
    for label in baseline_answers(instances[0]):
        results = [score_one(i, baseline_answers(i)[label]) for i in instances]
        for r in results:
            r["missing"] = r["api_error"] = False
        runs.append((label, results, False, {}))
    for path in args.responses:
        records, _ = load_responses(path)
        unknown = sorted(set(records) - ids)
        if unknown:
            parser.error(f"{path} has ids not in {args.instances}, e.g. {unknown[0]}")
        label = run_label(records, path)
        allow_abstain = label.startswith("B5 ")
        runs.append((label, score_run(instances, records), allow_abstain,
                     cost(list(records.values()))))

    groups = [("overall", "all", lambda inst: True)]
    groups += [("rooms", band_name(b, "rooms"), lambda inst, b=b: in_band(inst["n"], b))
               for b in SIZE_BANDS]
    groups += [("chase", band_name(b, "half-moves"),
                lambda inst, b=b: in_band(inst["max_half_moves"], b)) for b in DEPTH_BANDS]

    rows = []
    for label, results, allow_abstain, run_cost in runs:
        for group_by, group, keep in groups:
            subset = [r for r, inst in zip(results, instances) if keep(inst)]
            if not subset:
                continue
            row = {"run": label, "group_by": group_by, "group": group,
                   **metrics(subset, allow_abstain)}
            if group_by == "overall":
                row.update(run_cost)
            rows.append(row)

    width = max(len(r["run"]) for r in rows)
    header = ["exact % [95% CI]", "room acc % [95% CI]", "unsafe", "waste", "fail",
              "cover", "brier", "calls", "tokens", "med s"]
    widths = [17, 19, 6, 6, 4, 5, 5, 5, 7, 6]
    print(f"{len(instances)} instances from {args.instances}\n")
    print(f"{'run':<{width}}  " + "  ".join(f"{h:>{w}}" for h, w in zip(header, widths)))
    for r in (r for r in rows if r["group_by"] == "overall"):
        fail = r["unparseable"] + (0 if r["abstentions_allowed"] else r["abstained"])
        cells = [
            f"{pct(r.get('exact_match_rate'))} [{pct(r['exact_ci_low'])}-{pct(r['exact_ci_high'])}]",
            f"{pct(r.get('room_accuracy'))} [{pct(r['room_ci_low'])}-{pct(r['room_ci_high'])}]",
            pct(r.get("unsafe_room_rate")),
            pct(r.get("wasteful_room_rate")),
            str(fail),
            pct(r["coverage"]),
            "-" if r.get("brier") is None else f"{r['brier']:.3f}",
            "-" if r.get("calls_per_instance") is None else f"{r['calls_per_instance']:.1f}",
            "-" if r.get("tokens_per_instance") is None else f"{r['tokens_per_instance']:.0f}",
            "-" if r.get("latency_median_s") is None else f"{r['latency_median_s']:.1f}",
        ]
        print(f"{r['run']:<{width}}  " + "  ".join(f"{c:>{w}}" for c, w in zip(cells, widths)))
        if r["missing"] or r["api_errors"]:
            print(f"{'':<{width}}  ^ {r['missing']} missing and {r['api_errors']} "
                  "API errors counted as wrong; rerun the runner to fill them in")

    for group_by, title in (("rooms", "by room count"), ("chase", "by longest optimal chase")):
        names = [g for gb, g, _ in groups if gb == group_by]
        print(f"\nExact match % / room accuracy % {title}")
        print(f"{'run':<{width}}  " + "  ".join(f"{n:>14}" for n in names))
        for label, *_ in runs:
            cells = []
            for n in names:
                r = next((x for x in rows if x["run"] == label and x["group"] == n), None)
                cells.append("-" if r is None else
                             f"{pct(r.get('exact_match_rate'))}/{pct(r.get('room_accuracy'))}")
            print(f"{label:<{width}}  " + "  ".join(f"{c:>14}" for c in cells))

    print("\nunsafe/waste: % of answered rooms wrongly labelled capture / escape. "
          "fail: instances with no usable answer.\ncover: % of instances answered "
          "(only B5 may abstain). Intervals are 95%.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in (
            "run", "group_by", "group"), k))
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
