#!/usr/bin/env python3
"""Run the Solution B0 baseline through the direct OpenAI Responses API.

B0 deliberately uses the existing plain prompts without tools, code,
classical-solver access, structured-output enforcement, or post-processing.
The scorer is run separately after responses have been saved.

Example:
    .venv-openai/bin/python run_openai_b0.py \
        --limit 1 --max-output-tokens 512 \
        --output work/openai_b0_smoke.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "dataset" / "prompts_B0_dev.jsonl"
DEFAULT_OUTPUT = ROOT / "work" / "openai_b0_responses.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def existing_ids(path: Path, model: str, effort: str) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") != model or row.get("reasoning_effort") != effort:
                raise ValueError(
                    f"{path}:{line_number}: existing output uses different model/settings"
                )
            case_id = row.get("id")
            if not isinstance(case_id, str) or case_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            seen.add(case_id)
            if row.get("status") == "ok":
                done.add(case_id)
    return done


def usage_value(usage, name: str):
    return getattr(usage, name, None) if usage is not None else None


def reasoning_tokens(usage):
    details = getattr(usage, "output_tokens_details", None) if usage else None
    return getattr(details, "reasoning_tokens", None) if details else None


def run(args: argparse.Namespace) -> int:
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
    selected = prompts[: args.limit] if args.limit is not None else prompts
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    effort = args.reasoning_effort or os.environ.get(
        "OPENAI_REASONING_EFFORT", "medium"
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(output, model, effort)
    pending = [row for row in selected if row["id"] not in done]

    print(f"model={model} reasoning_effort={effort}")
    print(f"selected={len(selected)} pending={len(pending)}")
    if not pending:
        return 0

    client = OpenAI(max_retries=0, timeout=args.timeout)
    with output.open("a", encoding="utf-8") as handle:
        for row in pending:
            started = time.perf_counter()
            record = {
                "id": row["id"],
                "model": model,
                "api": "responses",
                "reasoning_effort": effort,
                "prompt_sha256": hashlib.sha256(
                    row["prompt"].encode("utf-8")
                ).hexdigest(),
                "started_at": utc_now(),
            }
            try:
                request = {
                    "model": model,
                    "reasoning": {"effort": effort},
                    "store": False,
                    "input": row["prompt"],
                }
                if args.max_output_tokens is not None:
                    request["max_output_tokens"] = args.max_output_tokens
                response = client.responses.create(**request)
                usage = response.usage
                text = response.output_text or ""
                record.update(
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
            except Exception as exc:  # preserve failure evidence and stop safely
                record.update(
                    {
                        "response": "",
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            record["finished_at"] = utc_now()
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{row['id']}: {record['status']} "
                f"{record['latency_seconds']}s "
                f"out={record.get('output_tokens')} "
                f"reasoning={record.get('reasoning_tokens')}"
            )
            if record["status"] == "error":
                return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--ids-from",
        type=Path,
        default=None,
        help="Optional JSONL file whose id fields select a prompt subset",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh"), default=None)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional cap; omit to let the API use its model/context limit",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
