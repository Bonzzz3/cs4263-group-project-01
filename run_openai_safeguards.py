#!/usr/bin/env python3
"""Run Solution B1--B5 through the direct OpenAI Responses API.

The runner is deliberately separate from the B0 runner.  It never loads or
calls the classical solver, never reads truth labels, and never sends checker
feedback to the model.  B3 voting and B4/B5 post-processing use only model
outputs and the original task prompt.

Examples:
    .venv-openai/bin/python run_openai_safeguards.py \
        --variant B1 --limit 2 --timeout 600 \
        --output work/openai_b1_smoke.jsonl

    .venv-openai/bin/python run_openai_safeguards.py \
        --variant B5 --abstain-threshold 0.70 --ids-from \
        dataset/instances_test_100.jsonl --prompts dataset/prompts_B0_test.jsonl \
        --timeout 600 --output work/openai_b5_test100.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from run_openai_b0 import (
    read_ids,
    read_prompts,
    reasoning_tokens,
    usage_value,
)
from solution_b_prompts import (
    apply_abstention,
    build_b1_prompt,
    build_b2_prompt,
    build_b3_prompt,
    build_b4_critic_prompt,
    build_b4_revision_prompt,
    build_b5_prompt,
    majority_vote,
    parse_candidate,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "dataset" / "prompts_B0_dev.jsonl"
DEFAULT_OUTPUT = ROOT / "work" / "openai_safeguard_responses.jsonl"
VARIANTS = ("B1", "B2", "B3", "B4", "B5")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def existing_ids(path: Path, model: str, effort: str, variant: str) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("model") != model
                or row.get("reasoning_effort") != effort
                or row.get("variant") != variant
            ):
                raise ValueError(
                    f"{path}:{line_number}: existing output uses different settings"
                )
            case_id = row.get("id")
            if not isinstance(case_id, str) or case_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            seen.add(case_id)
            if row.get("status") == "ok":
                done.add(case_id)
    return done


def select_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    prompts = read_prompts(args.prompts)
    if args.ids_from is not None:
        requested_ids = read_ids(args.ids_from)
        prompt_ids = {row["id"] for row in prompts}
        missing = requested_ids - prompt_ids
        if missing:
            sample = ", ".join(sorted(missing)[:3])
            raise ValueError(
                f"{args.ids_from} contains ids absent from {args.prompts}"
                f" (for example: {sample})"
            )
        prompts = [row for row in prompts if row["id"] in requested_ids]
    return prompts[: args.limit] if args.limit is not None else prompts


def call_model(
    client: OpenAI,
    model: str,
    effort: str,
    prompt: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    call = {"started_at": utc_now()}
    try:
        request: dict[str, Any] = {
            "model": model,
            "reasoning": {"effort": effort},
            "store": False,
            "input": prompt,
        }
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        response = client.responses.create(**request)
        usage = response.usage
        text = response.output_text or ""
        call.update(
            {
                "response": text,
                "response_id": response.id,
                "input_tokens": usage_value(usage, "input_tokens"),
                "output_tokens": usage_value(usage, "output_tokens"),
                "reasoning_tokens": reasoning_tokens(usage),
                "total_tokens": usage_value(usage, "total_tokens"),
                "status": "ok" if text else "empty_output",
            }
        )
    except Exception as exc:  # preserve diagnostic evidence in the JSONL
        call.update(
            {
                "response": "",
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    call["finished_at"] = utc_now()
    call["latency_seconds"] = round(time.perf_counter() - started, 3)
    return call


def add_call_totals(record: dict[str, Any]) -> None:
    calls = record.get("calls", [])
    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
        values = [call.get(field) for call in calls if isinstance(call.get(field), int)]
        record[field] = sum(values) if values else None
    record["call_count"] = len(calls)
    record["call_latencies_seconds"] = [
        call.get("latency_seconds") for call in calls
    ]


def answer_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def run_variant(
    variant: str,
    base_prompt: str,
    room_names: list[str],
    client: OpenAI,
    model: str,
    effort: str,
    max_output_tokens: int | None,
    abstain_threshold: float | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    calls: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    if variant == "B1":
        calls.append(call_model(client, model, effort, build_b1_prompt(base_prompt), max_output_tokens))
    elif variant == "B2":
        calls.append(call_model(client, model, effort, build_b2_prompt(base_prompt), max_output_tokens))
    elif variant == "B3":
        prompt = build_b3_prompt(base_prompt)
        for _ in range(3):
            calls.append(call_model(client, model, effort, prompt, max_output_tokens))
        candidates = [
            parse_candidate(call["response"], room_names)
            if call.get("status") == "ok"
            else None
            for call in calls
        ]
        final = majority_vote(candidates, room_names)
    elif variant == "B4":
        draft_prompt = base_prompt
        draft_call = call_model(client, model, effort, draft_prompt, max_output_tokens)
        calls.append(draft_call)
        draft = draft_call.get("response", "")
        critic_prompt = build_b4_critic_prompt(base_prompt, draft)
        critic_call = call_model(client, model, effort, critic_prompt, max_output_tokens)
        calls.append(critic_call)
        critique = critic_call.get("response", "")
        revision_prompt = build_b4_revision_prompt(base_prompt, draft, critique)
        revision_call = call_model(client, model, effort, revision_prompt, max_output_tokens)
        calls.append(revision_call)
        final_text = revision_call.get("response", "")
        if parse_candidate(final_text, room_names) is not None:
            final = parse_candidate(final_text, room_names)
        elif parse_candidate(draft, room_names) is not None:
            final = parse_candidate(draft, room_names)
    elif variant == "B5":
        calls.append(call_model(client, model, effort, build_b5_prompt(base_prompt), max_output_tokens))
        if abstain_threshold is None:
            raise ValueError("B5 requires --abstain-threshold selected on development data")
        candidate = (
            parse_candidate(calls[0]["response"], room_names)
            if calls[0].get("status") == "ok"
            else None
        )
        final = apply_abstention(candidate, room_names, abstain_threshold)
    else:
        raise ValueError(f"unknown safeguard variant: {variant}")

    if final is not None:
        return answer_json(final), calls, final
    if variant == "B4" and calls:
        return calls[-1].get("response", ""), calls, None
    return calls[0].get("response", "") if calls else "", calls, None


def run(args: argparse.Namespace) -> int:
    if args.variant == "B5" and args.abstain_threshold is None:
        raise ValueError("B5 requires --abstain-threshold")
    if args.abstain_threshold is not None and not 0 <= args.abstain_threshold <= 1:
        raise ValueError("--abstain-threshold must be between 0 and 1")

    prompts = select_prompts(args)
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5.4")
    effort = args.reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "low")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(output, model, effort, args.variant)
    pending = [row for row in prompts if row["id"] not in done]

    print(f"variant={args.variant} model={model} reasoning_effort={effort}")
    print(f"selected={len(prompts)} pending={len(pending)}")
    if not pending:
        return 0

    client = OpenAI(max_retries=0, timeout=args.timeout)
    with output.open("a", encoding="utf-8") as handle:
        for row in pending:
            started = time.perf_counter()
            # Derive room names from the required JSON contract rather than
            # from graph text.  This covers both A..Z and R0..R29 names.
            contract = row["prompt"].split('"capture_guaranteed": {', 1)[1].split("}", 1)[0]
            room_names = [part.split(":", 1)[0].strip().strip('"') for part in contract.split(",")]
            record: dict[str, Any] = {
                "id": row["id"],
                "variant": args.variant,
                "model": model,
                "api": "responses",
                "reasoning_effort": effort,
                "abstain_threshold": args.abstain_threshold,
                "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                "started_at": utc_now(),
            }
            try:
                response_text, calls, final = run_variant(
                    args.variant,
                    row["prompt"],
                    room_names,
                    client,
                    model,
                    effort,
                    args.max_output_tokens,
                    args.abstain_threshold,
                )
                record["calls"] = calls
                record["response"] = response_text
                record["final_object"] = final
                record["status"] = "ok" if response_text else "empty_output"
            except Exception as exc:
                record.update(
                    {
                        "calls": record.get("calls", []),
                        "response": "",
                        "final_object": None,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            add_call_totals(record)
            record["finished_at"] = utc_now()
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{row['id']}: {record['status']} {record['latency_seconds']}s "
                f"calls={record.get('call_count')} out={record.get('output_tokens')} "
                f"reasoning={record.get('reasoning_tokens')}"
            )
            if record["status"] == "error":
                return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--ids-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--abstain-threshold", type=float, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

