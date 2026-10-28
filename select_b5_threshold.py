#!/usr/bin/env python3
"""Select a B5 abstention threshold using development data only.

The script prints the full threshold/coverage/risk table and a deterministic
selection.  It never runs the solver and must only be run on the development
split.  The selected threshold is then passed to run_openai_safeguards.py for
the held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cops_robbers import (
    load_responses,
    parse_confidence,
    parse_response,
    score_one,
    summarise,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_threshold(
    instances: dict[str, dict], records: list[dict], threshold: float
) -> dict:
    """Score the dev responses as if B5 had used this threshold, with the
    runner's rules: an answer that is not valid in the required format is a
    failure; a valid answer abstains if the model said UNDETERMINED or its
    confidence is missing or below the threshold."""
    results = []
    for record in records:
        instance = instances[record["id"]]
        raw = record.get("response", "")
        result = score_one(instance, raw)
        if not result["parsed"] or result.get("abstained"):
            results.append(result)
            continue
        confidence = parse_confidence(parse_response(raw).get("confidence"))
        if confidence is None or confidence < threshold:
            results.append({"parsed": True, "abstained": True,
                            "rooms": instance["n"], "confidence": confidence})
        else:
            results.append(result)
    return summarise(results, allow_abstain=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.20,
        help="development coverage required for a feasible threshold",
    )
    parser.add_argument(
        "--max-unsafe-room-rate",
        type=float,
        default=0.05,
        help="maximum development unsafe-room rate for a feasible threshold",
    )
    args = parser.parse_args()
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1")
    if not 0 <= args.max_unsafe_room_rate <= 1:
        parser.error("--max-unsafe-room-rate must be between 0 and 1")

    instances = {row["id"]: row for row in load_jsonl(args.instances)}
    if not instances or any(row.get("split") != "dev" for row in instances.values()):
        parser.error("B5 threshold selection must use a development split file")
    latest, _ = load_responses(args.responses)
    records = list(latest.values())
    unknown = [r["id"] for r in records if r["id"] not in instances]
    if unknown:
        parser.error(f"response ids not in {args.instances}, e.g. {unknown[0]}")
    thresholds = [round(i / 100, 2) for i in range(101)]
    table = []
    for threshold in thresholds:
        metrics = evaluate_threshold(instances, records, threshold)
        table.append({"threshold": threshold, **metrics})
        print(json.dumps(table[-1], sort_keys=True))

    feasible = [
        row
        for row in table
        if row.get("coverage") is not None
        and row["coverage"] >= args.min_coverage
        and row.get("unsafe_room_rate") is not None
        and row["unsafe_room_rate"] <= args.max_unsafe_room_rate
    ]
    if feasible:
        chosen = max(
            feasible,
            key=lambda row: (
                row["coverage"],
                row.get("exact_match_rate", 0),
                -row["threshold"],
            ),
        )
        policy = "highest coverage within the development safety constraint"
    else:
        chosen = min(
            table,
            key=lambda row: (
                row.get("unsafe_room_rate", 1),
                -row.get("coverage", 0),
                -row.get("exact_match_rate", 0),
            ),
        )
        policy = "no feasible threshold; lowest observed development risk"
    print(json.dumps({"selected": chosen, "selection_policy": policy}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

