#!/usr/bin/env python3
r"""Run the Solution B0 baseline: the plain prompt, one call per instance.

B0 deliberately uses the existing plain prompts without tools, code,
classical-solver access, structured-output enforcement, or post-processing.
The scorer is run separately after responses have been saved.

Works with OpenAI (Responses API), and SoCLaaS and OpenRouter
(OpenAI-compatible Chat Completions); see llm_client.py. Rerunning the same
command resumes: finished instances are skipped and failed API calls are
retried.

Examples:
    uv run --env-file .env python run_openai_b0.py --limit 1 \
        --output work/openai_b0_smoke.jsonl

    uv run --env-file .env python run_openai_b0.py --provider soclaas \
        --model qwen3.6:35b --limit 1 --output work/soclaas_b0_smoke.jsonl

    uv run --env-file .env python run_openai_b0.py --provider openrouter \
        --model openai/gpt-5.4 --reasoning-effort low --limit 1 \
        --output work/openrouter_b0_smoke.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "dataset" / "prompts_B0_dev.jsonl"
DEFAULT_OUTPUT = ROOT / "work" / "openai_b0_responses.jsonl"


def read_prompts(path: Path) -> list[dict[str, str]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "prompt"}:
                raise ValueError(f"{path}:{line_number}: expected id and prompt")
            if not isinstance(row["id"], str) or not isinstance(row["prompt"], str):
                raise ValueError(f"{path}:{line_number}: id and prompt must be strings")
            if not row["id"] or not row["prompt"] or row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: blank or duplicate id/prompt")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def read_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("id")
            if not isinstance(case_id, str) or not case_id or case_id in ids:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            ids.add(case_id)
    if not ids:
        raise ValueError(f"No ids found in {path}")
    return ids


def select_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    prompts = read_prompts(args.prompts)
    if args.ids_from is not None:
        requested_ids = read_ids(args.ids_from)
        missing = requested_ids - {row["id"] for row in prompts}
        if missing:
            sample = ", ".join(sorted(missing)[:3])
            raise ValueError(
                f"{args.ids_from} contains ids absent from {args.prompts}"
                f" (for example: {sample})"
            )
        prompts = [row for row in prompts if row["id"] in requested_ids]
    return prompts[: args.limit] if args.limit is not None else prompts


def run(args: argparse.Namespace) -> int:
    prompts = select_prompts(args)
    model, effort = resolve_settings(args.provider, args.model, args.reasoning_effort)
    settings = {"provider": args.provider, "model": model, "reasoning_effort": effort}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output, settings)
    pending = [row for row in prompts if row["id"] not in done]

    print(f"provider={args.provider} model={model} reasoning_effort={effort}")
    print(f"selected={len(prompts)} pending={len(pending)} workers={args.workers}")
    if not pending:
        return 0

    client = make_client(args.provider, args.timeout, args.max_retries)

    def job(row: dict[str, str]) -> dict:
        record = {
            "id": row["id"],
            **settings,
            "api": "responses" if args.provider == "openai" else "chat.completions",
            "max_output_tokens": args.max_output_tokens,
            "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
            "recorded_at": utc_now(),
        }
        record.update(call_model(client, args.provider, model, effort,
                                 row["prompt"], args.max_output_tokens, args.stream))
        return record

    return run_jobs(pending, job, output, args.workers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--ids-from",
        type=Path,
        default=None,
        help="Optional JSONL file whose id fields select a prompt subset",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    add_run_arguments(parser)
    args = parser.parse_args()
    check_run_arguments(parser, args)
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
