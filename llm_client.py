#!/usr/bin/env python3
"""Shared API access for the Solution B runners.

All providers go through the OpenAI Python SDK:
  openai      OpenAI's Responses API. Needs OPENAI_API_KEY.
  soclaas     NUS SoC's SoCLaaS gateway, an OpenAI-compatible Chat
              Completions API. Needs SOCLAAS_API_KEY.
  openrouter  OpenRouter's OpenAI-compatible Chat Completions API, which
              routes to many vendors' models (e.g. openai/gpt-5.4).
              Needs OPENROUTER_API_KEY.
SOCLAAS_BASE_URL and OPENROUTER_BASE_URL override the default endpoints.

This module only sends prompts and records what came back. It never loads the
solver, the scorer or any labels, so Solution B stays separate from Solution A.

No provider is sent a temperature, so each uses its default sampling.
That matters for B3: with temperature 0 its three samples would be near
copies of each other and the vote would mean little.

List the models a provider offers:
    uv run --env-file .env python llm_client.py --provider soclaas
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROVIDERS = ("openai", "soclaas", "openrouter")
# Not every model accepts every level; a rejected one stops the run cleanly.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
BASE_URLS = {
    "soclaas": "https://soclaas-api.comp.nus.edu.sg/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
DEFAULT_MODEL = {"openai": "gpt-5.6-luna", "soclaas": None, "openrouter": None}
DEFAULT_EFFORT = {"openai": "medium", "soclaas": None, "openrouter": None}
ENV_PREFIX = {"openai": "OPENAI", "soclaas": "SOCLAAS", "openrouter": "OPENROUTER"}

# A record whose status is one of these is a finished model outcome and is not
# retried on resume. Only "error" (the API call itself failed) is retried.
FINAL_STATUSES = {"ok", "truncated", "empty_output"}

# Errors that mean the run is misconfigured (bad key, unknown model, rejected
# parameter). Retrying will not help, so the run stops submitting new work.
CONFIG_ERRORS = {
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "UnprocessableEntityError",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------- settings
def resolve_settings(
    provider: str, model: str | None, effort: str | None
) -> tuple[str, str | None]:
    """Fill in model and reasoning effort from flags, then the provider's
    environment variables (OPENAI_MODEL, SOCLAAS_REASONING_EFFORT, ...), then
    the defaults. For SoCLaaS and OpenRouter an unset effort is not sent,
    so the model uses its own default."""
    prefix = ENV_PREFIX[provider]
    model = model or os.environ.get(f"{prefix}_MODEL") or DEFAULT_MODEL[provider]
    effort = (effort or os.environ.get(f"{prefix}_REASONING_EFFORT")
              or DEFAULT_EFFORT[provider])
    if not model:
        raise ValueError(
            f"{provider} needs --model or {prefix}_MODEL. List the available "
            f"models with: python llm_client.py --provider {provider}")
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"reasoning effort must be one of {', '.join(EFFORTS)}")
    return model, effort


def make_client(provider: str, timeout: float, max_retries: int):
    """An SDK client. The SDK retries rate limits, timeouts and server errors
    max_retries times with exponential backoff before giving up."""
    from openai import OpenAI

    if provider == "openai":
        return OpenAI(timeout=timeout, max_retries=max_retries)
    prefix = ENV_PREFIX[provider]
    key = os.environ.get(f"{prefix}_API_KEY")
    if not key:
        raise ValueError(f"{prefix}_API_KEY is not set")
    return OpenAI(
        api_key=key,
        base_url=os.environ.get(f"{prefix}_BASE_URL", BASE_URLS[provider]),
        timeout=timeout,
        max_retries=max_retries,
    )


# -------------------------------------------------------------------- calls
_THINK = re.compile(r"<think>.*?</think>", re.S)


def split_thinking(content: str) -> tuple[str, int]:
    """Remove <think> blocks that some open models put in the reply.
    Returns (answer text, number of thinking characters removed)."""
    removed = sum(len(m) for m in _THINK.findall(content))
    text = _THINK.sub("", content)
    if "</think>" in text:  # opening tag omitted by the chat template
        head, text = text.rsplit("</think>", 1)
        removed += len(head) + len("</think>")
    if "<think>" in text:  # cut off mid-thought
        head, tail = text.split("<think>", 1)
        removed += len(tail) + len("<think>")
        text = head
    return text.strip(), removed


def _get(obj: Any, *path: str) -> Any:
    for name in path:
        obj = getattr(obj, name, None) if obj is not None else None
    return obj


def _call_openai(client, model, effort, prompt, max_output_tokens) -> dict:
    request: dict[str, Any] = {"model": model, "store": False, "input": prompt}
    if effort is not None:
        request["reasoning"] = {"effort": effort}
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens
    r = client.responses.create(**request)
    return {
        "response": r.output_text or "",
        "response_id": r.id,
        "api_status": getattr(r, "status", None),
        "incomplete_reason": _get(r, "incomplete_details", "reason"),
        "input_tokens": _get(r, "usage", "input_tokens"),
        "output_tokens": _get(r, "usage", "output_tokens"),
        "reasoning_tokens": _get(r, "usage", "output_tokens_details", "reasoning_tokens"),
        "total_tokens": _get(r, "usage", "total_tokens"),
    }


def _call_chat(client, model, effort, prompt, max_output_tokens,
               provider="soclaas", stream=True) -> dict:
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if effort is not None and provider == "openrouter":
        request["extra_body"] = {"reasoning": {"effort": effort}}
    elif effort is not None:
        request["reasoning_effort"] = effort
    if max_output_tokens is not None:
        request["max_tokens"] = max_output_tokens
    if stream:
        content, thinking_chars, finish, usage, rid, routed = _read_stream(client, request)
    else:
        r = client.chat.completions.create(**request)
        choice = r.choices[0]
        content, finish, usage, rid = choice.message.content or "", choice.finish_reason, r.usage, r.id
        routed = getattr(r, "provider", None)
        thinking_chars = sum(len(v) for v in (getattr(choice.message, f, None)
                                              for f in ("reasoning_content", "reasoning"))
                             if isinstance(v, str))
    text, removed = split_thinking(content)
    if finish is None and stream:
        # The stream closed without the model finishing. SoCLaaS does this to
        # every request at 10 minutes, so it is a cut-off, not an empty answer.
        finish = "stream_closed"
    return {
        "response": text,
        "response_id": rid,
        "streamed": stream,
        "api_status": "completed" if finish in ("stop", None) else "incomplete",
        "incomplete_reason": None if finish in ("stop", None) else finish,
        "finish_reason": finish,
        "thinking_chars": thinking_chars + removed,
        "input_tokens": _get(usage, "prompt_tokens"),
        "output_tokens": _get(usage, "completion_tokens"),
        "reasoning_tokens": _get(usage, "completion_tokens_details", "reasoning_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
        # OpenRouter only: the vendor that served the call, and its cost in USD
        "routed_to": routed,
        "cost_usd": _get(usage, "cost"),
    }


def _read_stream(client, request):
    """Make the call as a stream and put the pieces back together.

    Long reasoning calls need this: a gateway that sees no response for a few
    minutes gives up with a 502 (SoCLaaS cut off a non-streamed DeepSeek call
    at about 5 minutes), while a stream keeps data flowing. The timeout then
    limits the gap between chunks rather than the whole call."""
    parts: list[str] = []
    thinking_chars, finish, usage, rid, routed = 0, None, None, None, None
    with client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}) as chunks:
        for chunk in chunks:
            rid = rid or chunk.id
            routed = routed or getattr(chunk, "provider", None)
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = choice.delta
                if delta is not None:
                    if delta.content:
                        parts.append(delta.content)
                    for field in ("reasoning_content", "reasoning"):
                        value = getattr(delta, field, None)
                        if isinstance(value, str):
                            thinking_chars += len(value)
                if choice.finish_reason:
                    finish = choice.finish_reason
    return "".join(parts), thinking_chars, finish, usage, rid, routed


def call_model(
    client,
    provider: str,
    model: str,
    effort: str | None,
    prompt: str,
    max_output_tokens: int | None,
    stream: bool = True,
) -> dict[str, Any]:
    """One API call. Never raises: failures are recorded with status "error".
    SoCLaaS and OpenRouter calls are streamed unless stream is False.

    status is "ok", "truncated" (the API stopped early, usually at the token
    limit), "empty_output" (finished but no answer text) or "error"."""
    started = time.perf_counter()
    call: dict[str, Any] = {"started_at": utc_now()}
    try:
        if provider == "openai":
            call.update(_call_openai(client, model, effort, prompt, max_output_tokens))
        else:
            call.update(_call_chat(client, model, effort, prompt, max_output_tokens,
                                   provider, stream))
        if call["api_status"] == "incomplete":
            call["status"] = "truncated"
        else:
            call["status"] = "ok" if call["response"] else "empty_output"
    except Exception as exc:  # keep the evidence; the runner decides what next
        call.update({"response": "", "status": "error",
                     "error_type": type(exc).__name__, "error": str(exc)})
    call["finished_at"] = utc_now()
    call["latency_seconds"] = round(time.perf_counter() - started, 3)
    return call


# ------------------------------------------------------------ run plumbing
def load_done(path: Path, settings: dict[str, Any]) -> set[str]:
    """Ids already finished in an output file, for resuming a run.

    The last record per id wins, so a retried instance is judged by its latest
    attempt. Records made with different settings are refused, so one file
    never mixes models, efforts or variants."""
    if not path.exists():
        return set()
    latest: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("provider", "openai")  # files written before SoCLaaS
            for key, want in settings.items():
                if row.get(key) != want:
                    raise ValueError(
                        f"{path}:{line_number}: existing output has {key}="
                        f"{row.get(key)!r}, this run uses {want!r}; "
                        "use a different --output file")
            latest[row["id"]] = row.get("status")
    return {i for i, status in latest.items() if status in FINAL_STATUSES}


def run_jobs(
    rows: list[dict[str, Any]],
    job: Callable[[dict[str, Any]], dict[str, Any]],
    output: Path,
    workers: int,
) -> int:
    """Run job(row) for every row on a thread pool, appending each record to
    output as it finishes. Returns 1 if any record failed, else 0.

    A configuration error (bad key, unknown model, rejected parameter) stops
    new work from starting; calls already in flight finish and are saved."""
    abort = threading.Event()

    def guarded(row):
        if abort.is_set():
            return None
        record = job(row)
        if record.get("error_type") in CONFIG_ERRORS:
            abort.set()
        return record

    failed = finished = 0
    with output.open("a", encoding="utf-8") as handle, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded, row) for row in rows]
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            finished += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            failed += record["status"] == "error"
            print(f"[{finished}/{len(rows)}] {record['id']}: {record['status']} "
                  f"{record.get('latency_seconds')}s "
                  f"calls={record.get('call_count', 1)} "
                  f"out={record.get('output_tokens')} "
                  f"reasoning={record.get('reasoning_tokens')}"
                  + (f" {record.get('error_type')}: {record.get('error')}"
                     if record["status"] == "error" else ""),
                  flush=True)
    if abort.is_set():
        print(f"Stopped early after a configuration error; "
              f"{len(rows) - finished} instances were not started.")
    if failed:
        print(f"{failed} instances failed; run the same command again to retry them.")
    return 1 if failed or abort.is_set() else 0


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by both runners."""
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--model", default=None,
                        help="default: $OPENAI_MODEL / $SOCLAAS_MODEL, then "
                             f"{DEFAULT_MODEL['openai']} for openai")
    parser.add_argument("--reasoning-effort", choices=EFFORTS, default=None,
                        help="default: medium for openai; not sent otherwise")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="optional cap; omit to use the model's own limit")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="seconds per API call; when streaming, the longest "
                             "wait for the next chunk")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="SoCLaaS/OpenRouter: wait for the whole answer instead of "
                             "streaming it (gateways may cut long calls off)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="SDK retries for rate limits, timeouts, server errors")
    parser.add_argument("--workers", type=int, default=4,
                        help="instances run in parallel")


def check_run_arguments(parser: argparse.ArgumentParser, args) -> None:
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")


def main() -> int:
    parser = argparse.ArgumentParser(description="List a provider's models.")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    args = parser.parse_args()
    client = make_client(args.provider, timeout=60, max_retries=1)
    for m in sorted(client.models.list(), key=lambda m: m.id):
        print(m.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
