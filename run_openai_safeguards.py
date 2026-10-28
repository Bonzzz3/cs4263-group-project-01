#!/usr/bin/env python3
r"""Run Solution B1--B5 with OpenAI, SoCLaaS or OpenRouter (see llm_client.py).

The runner is deliberately separate from the B0 runner.  It never loads or
calls the classical solver, never reads truth labels, and never sends checker
feedback to the model.  B3 voting and B4/B5 post-processing use only model
outputs and the original task prompt.

If any API call for an instance fails, the whole instance is recorded as an
error and retried on the next run, so B3 never votes with a missing sample
and B4 never critiques a draft that was not produced.

Examples:
    uv run --env-file .env python run_openai_safeguards.py \
        --variant B1 --limit 2 --output work/openai_b1_smoke.jsonl

    uv run --env-file .env python run_openai_safeguards.py \
        --variant B5 --abstain-threshold 0.70 --ids-from \
        dataset/instances_test_80.jsonl --prompts dataset/prompts_B0_test.jsonl \
        --output work/openai_b5_test80.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from llm_client import (
    add_run_arguments,
    call_model,
    check_run_arguments,
    load_done,
    make_client,
    resolve_settings,
    run_jobs,
    utc_now,
)
from run_openai_b0 import select_prompts
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


class CallFailed(Exception):
    """An API call for this instance failed; the instance is retried later."""


def add_call_totals(record: dict[str, Any]) -> None:
    calls = record.get("calls", [])
    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
                  "cost_usd"):
        values = [call.get(field) for call in calls
                  if isinstance(call.get(field), (int, float))]
        record[field] = sum(values) if values else None
    record["call_count"] = len(calls)
    record["truncated_calls"] = sum(call.get("status") == "truncated" for call in calls)
    record["call_latencies_seconds"] = [
        call.get("latency_seconds") for call in calls
    ]


def answer_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def run_variant(
    variant: str,
    base_prompt: str,
    room_names: list[str],
    ask: Callable[[str], dict[str, Any]],
    calls: list[dict[str, Any]],
    abstain_threshold: float | None,
) -> tuple[str, dict[str, Any] | None]:
    """Run one safeguard on one instance. Each API call is appended to calls.
    Raises CallFailed as soon as a call fails, so later calls never build on
    a missing answer."""

    def call(prompt: str) -> str:
        result = ask(prompt)
        calls.append(result)
        if result["status"] == "error":
            raise CallFailed(f"{result.get('error_type')}: {result.get('error')}")
        return result["response"]

    final: dict[str, Any] | None = None
    if variant == "B1":
        text = call(build_b1_prompt(base_prompt))
    elif variant == "B2":
        text = call(build_b2_prompt(base_prompt))
    elif variant == "B3":
        prompt = build_b3_prompt(base_prompt)
        samples = [call(prompt) for _ in range(3)]
        # A tie or fewer than two valid samples gives UNDETERMINED, which the
        # scorer counts as a wrong answer (only B5 may abstain).
        final = majority_vote([parse_candidate(t, room_names) for t in samples], room_names)
        text = samples[-1]
    elif variant == "B4":
        draft = call(base_prompt)
        critique = call(build_b4_critic_prompt(base_prompt, draft))
        text = call(build_b4_revision_prompt(base_prompt, draft, critique))
        final = parse_candidate(text, room_names) or parse_candidate(draft, room_names)
    elif variant == "B5":
        if abstain_threshold is None:
            raise ValueError("B5 requires --abstain-threshold selected on development data")
        text = call(build_b5_prompt(base_prompt))
        final = apply_abstention(parse_candidate(text, room_names), room_names,
                                 abstain_threshold)
    else:
        raise ValueError(f"unknown safeguard variant: {variant}")
    return (answer_json(final) if final is not None else text), final


def room_names_from_prompt(prompt: str) -> list[str]:
    """Room names from the required JSON contract rather than the graph text.
    This covers both A..Z and R0..R29 names."""
    contract = prompt.split('"capture_guaranteed": {', 1)[1].split("}", 1)[0]
    return [part.split(":", 1)[0].strip().strip('"') for part in contract.split(",")]


def run(args: argparse.Namespace) -> int:
    if args.variant == "B5" and args.abstain_threshold is None:
        raise ValueError("B5 requires --abstain-threshold")
    if args.abstain_threshold is not None and not 0 <= args.abstain_threshold <= 1:
        raise ValueError("--abstain-threshold must be between 0 and 1")

    prompts = select_prompts(args)
    model, effort = resolve_settings(args.provider, args.model, args.reasoning_effort)
    settings = {
        "provider": args.provider,
        "model": model,
        "reasoning_effort": effort,
        "variant": args.variant,
        "abstain_threshold": args.abstain_threshold,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output, settings)
    pending = [row for row in prompts if row["id"] not in done]

    print(f"variant={args.variant} provider={args.provider} model={model} "
          f"reasoning_effort={effort}")
    print(f"selected={len(prompts)} pending={len(pending)} workers={args.workers}")
    if not pending:
        return 0

    client = make_client(args.provider, args.timeout, args.max_retries)

    def ask(prompt: str) -> dict[str, Any]:
        return call_model(client, args.provider, model, effort, prompt,
                          args.max_output_tokens, args.stream)

    def job(row: dict[str, str]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": row["id"],
            **settings,
            "api": "responses" if args.provider == "openai" else "chat.completions",
            "max_output_tokens": args.max_output_tokens,
            "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
            "started_at": utc_now(),
        }
        calls: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            text, final = run_variant(args.variant, row["prompt"],
                                      room_names_from_prompt(row["prompt"]),
                                      ask, calls, args.abstain_threshold)
            record.update({"response": text, "final_object": final,
                           "status": "ok" if text else "empty_output"})
        except CallFailed:
            failed = calls[-1]
            record.update({"response": "", "final_object": None, "status": "error",
                           "error_type": failed.get("error_type"),
                           "error": failed.get("error")})
        record["calls"] = calls
        add_call_totals(record)
        record["finished_at"] = utc_now()
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        return record

    return run_jobs(pending, job, output, args.workers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--ids-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--abstain-threshold", type=float, default=None)
    add_run_arguments(parser)
    args = parser.parse_args()
    check_run_arguments(parser, args)
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

