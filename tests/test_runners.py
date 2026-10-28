"""
Tests for the Solution B runners, against a fake API server on localhost.

The real OpenAI SDK talks to the fake server over HTTP, so these tests check
the request formats for all three providers (OpenAI Responses, SoCLaaS and
OpenRouter Chat Completions), retries, resuming, truncation and error handling, without
calling any real model or spending anything.

Run with: uv run pytest
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_client import split_thinking  # noqa: E402

PROMPTS = ROOT / "dataset" / "prompts_B0_dev.jsonl"
FIRST_IDS = [json.loads(line)["id"] for line in PROMPTS.open()][:3]


# ------------------------------------------------------------ fake server
class FakeAPI:
    """Answers every room TRUE. `plan` maps a prompt substring to a list of
    behaviours used one per request: "ok", "500", "401", "incomplete",
    "think", "reasoning" (thinking in a separate field), "cut" (the stream
    closes before the model finishes, as SoCLaaS does at 10 minutes) or "slow" (the answer
    takes about 2.5 s, sent in small pieces when streaming). Chat requests
    with "stream": true get a server-sent event stream like the real APIs.
    Requests are logged for the tests to inspect."""

    def __init__(self):
        self.plan = {}
        self.requests = []
        self.lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.reply(200, {"object": "list", "data": [
                    {"id": "fake-model", "object": "model", "created": 0, "owned_by": "x"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body.get("input") or body["messages"][0]["content"]
                with api.lock:
                    api.requests.append({"path": self.path, "body": body})
                    behaviour = "ok"
                    for key, steps in api.plan.items():
                        if key in prompt and steps:
                            behaviour = steps.pop(0)
                            break
                if behaviour in ("500", "401"):
                    return self.reply(int(behaviour), {"error": {"message": "fake failure"}})
                rooms = re.findall(r'"(\w+)": true/false', prompt)
                text = json.dumps({"capture_guaranteed": {r: True for r in rooms},
                                   "confidence": 0.8})
                if self.path.endswith("/responses"):
                    self.reply(200, api.responses_body(text, behaviour))
                elif body.get("stream"):
                    self.stream(api.chat_body(text, behaviour), behaviour)
                else:
                    if behaviour == "slow":
                        time.sleep(2.5)
                    self.reply(200, api.chat_body(text, behaviour))

            def stream(self, full, behaviour):
                """Send `full` as chunks: thinking, the answer in pieces, the
                finish reason, then a usage-only chunk, then [DONE]."""
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                message = full["choices"][0]["message"]
                content = message["content"]
                base = {"id": full["id"], "object": "chat.completion.chunk",
                        "created": 0, "model": "fake", "provider": full["provider"]}
                deltas = []
                if message.get("reasoning"):
                    deltas.append({"role": "assistant", "reasoning": message["reasoning"]})
                pieces = 10 if behaviour == "slow" else 3
                step = max(1, len(content) // pieces)
                deltas += [{"content": content[i:i + step]}
                           for i in range(0, len(content), step)]
                events = [{**base, "choices": [{"index": 0, "delta": d,
                                                "finish_reason": None}]} for d in deltas]
                if behaviour == "cut":
                    events = [e for e in events if "reasoning" in e["choices"][0]["delta"]]
                else:
                    events.append({**base, "choices": [{"index": 0, "delta": {},
                                   "finish_reason": full["choices"][0]["finish_reason"]}]})
                    events.append({**base, "choices": [], "usage": full["usage"]})
                for event in events:
                    if behaviour == "slow":
                        time.sleep(0.25)
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def reply(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @staticmethod
    def responses_body(text, behaviour):
        incomplete = behaviour == "incomplete"
        return {
            "id": "resp_1", "object": "response", "created_at": 0, "model": "fake",
            "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
            "output": [] if incomplete else [{
                "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 30}},
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        }

    @staticmethod
    def chat_body(text, behaviour):
        content = f"<think>rooms and corridors {{A}}</think>\n{text}" if behaviour == "think" else text
        message = {"role": "assistant", "content": content}
        if behaviour in ("reasoning", "cut"):
            message["reasoning"] = "x" * 40
        return {
            "id": "chat_1", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "length" if behaviour == "incomplete" else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                      "cost": 0.0012},
            "provider": "FakeUpstream",
        }


@pytest.fixture
def api():
    fake = FakeAPI()
    yield fake
    fake.server.shutdown()


def run(script, api, *args):
    env = {**os.environ, "OPENAI_API_KEY": "test", "OPENAI_BASE_URL": api.url,
           "SOCLAAS_API_KEY": "test", "SOCLAAS_BASE_URL": api.url,
           "OPENROUTER_API_KEY": "test", "OPENROUTER_BASE_URL": api.url}
    for key in ("OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "SOCLAAS_MODEL",
                "SOCLAAS_REASONING_EFFORT", "OPENROUTER_MODEL",
                "OPENROUTER_REASONING_EFFORT"):
        env.pop(key, None)
    return subprocess.run([sys.executable, str(ROOT / script), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def records(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def out_file():
    return os.path.join(tempfile.mkdtemp(), "out.jsonl")


# ---------------------------------------------------------------- B0, OpenAI
def test_b0_openai_records_answers_tokens_and_settings(api):
    out = out_file()
    p = run("run_openai_b0.py", api, "--limit", "3", "--workers", "2", "--output", out)
    assert p.returncode == 0, p.stderr
    rows = records(out)
    assert sorted(r["id"] for r in rows) == sorted(FIRST_IDS)
    r = rows[0]
    assert r["status"] == "ok" and r["provider"] == "openai" and r["api"] == "responses"
    assert r["model"] == "gpt-5.6-luna" and r["reasoning_effort"] == "medium"
    assert r["reasoning_tokens"] == 30 and r["api_status"] == "completed"
    assert all(q["path"].endswith("/responses") for q in api.requests)
    assert api.requests[0]["body"]["reasoning"] == {"effort": "medium"}
    assert "temperature" not in api.requests[0]["body"]


def test_sdk_retries_a_server_error(api):
    api.plan["Rooms:"] = ["500"]
    out = out_file()
    p = run("run_openai_b0.py", api, "--limit", "1", "--max-retries", "2", "--output", out)
    assert p.returncode == 0, p.stderr
    assert records(out)[0]["status"] == "ok" and len(api.requests) == 2


def test_failed_call_is_retried_on_resume_and_scored_once(api):
    api.plan["Rooms:"] = ["500"]
    out = out_file()
    first = run("run_openai_b0.py", api, "--limit", "1", "--max-retries", "0", "--output", out)
    assert first.returncode == 1 and records(out)[0]["status"] == "error"
    second = run("run_openai_b0.py", api, "--limit", "1", "--max-retries", "0", "--output", out)
    assert second.returncode == 0 and [r["status"] for r in records(out)] == ["error", "ok"]
    third = run("run_openai_b0.py", api, "--limit", "1", "--output", out)
    assert third.returncode == 0 and "pending=0" in third.stdout   # used to crash
    score = subprocess.run([sys.executable, str(ROOT / "cops_robbers.py"), "score",
                            "--instances", str(ROOT / "dataset/instances_dev.jsonl"),
                            "--responses", out], capture_output=True, text=True, check=True)
    overall = json.loads(score.stdout.split("OVERALL", 1)[1].split("\nn=", 1)[0])
    assert overall["instances"] == 1 and overall["unparseable"] == 0
    assert overall["superseded_records"] == 1


def test_truncated_answer_is_recorded_and_not_retried(api):
    api.plan["Rooms:"] = ["incomplete"]
    out = out_file()
    p = run("run_openai_b0.py", api, "--limit", "1", "--max-output-tokens", "64",
            "--output", out)
    assert p.returncode == 0, p.stderr
    r = records(out)[0]
    assert r["status"] == "truncated" and r["incomplete_reason"] == "max_output_tokens"
    assert api.requests[0]["body"]["max_output_tokens"] == 64
    again = run("run_openai_b0.py", api, "--limit", "1", "--max-output-tokens", "64",
                "--output", out)
    assert "pending=0" in again.stdout


def test_configuration_error_stops_the_run(api):
    api.plan["Rooms:"] = ["401"] * 10
    out = out_file()
    p = run("run_openai_b0.py", api, "--limit", "3", "--workers", "1", "--output", out)
    assert p.returncode == 1 and "Stopped early" in p.stdout
    assert len(records(out)) == 1 and records(out)[0]["error_type"] == "AuthenticationError"


def test_resume_refuses_a_file_from_other_settings(api):
    out = out_file()
    assert run("run_openai_b0.py", api, "--limit", "1", "--output", out).returncode == 0
    p = run("run_openai_b0.py", api, "--limit", "1", "--reasoning-effort", "low",
            "--output", out)
    assert p.returncode != 0 and "different --output file" in p.stderr


# ---------------------------------------------------------------- SoCLaaS
def test_soclaas_needs_a_model(api):
    p = run("run_openai_b0.py", api, "--provider", "soclaas", "--limit", "1",
            "--output", out_file())
    assert p.returncode != 0 and "SOCLAAS_MODEL" in p.stderr


def test_b0_soclaas_uses_chat_completions_and_strips_thinking(api):
    api.plan["Rooms:"] = ["think"]
    out = out_file()
    p = run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "qwen-test",
            "--limit", "1", "--output", out)
    assert p.returncode == 0, p.stderr
    r = records(out)[0]
    body = api.requests[0]["body"]
    assert api.requests[0]["path"].endswith("/chat/completions")
    assert body["model"] == "qwen-test" and "reasoning_effort" not in body
    assert "temperature" not in body
    assert r["provider"] == "soclaas" and r["api"] == "chat.completions"
    assert r["response"].startswith("{") and r["thinking_chars"] > 0
    assert r["input_tokens"] == 100 and r["output_tokens"] == 50


def test_soclaas_sends_reasoning_effort_only_when_asked(api):
    p = run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "qwen-test",
            "--reasoning-effort", "low", "--limit", "1", "--output", out_file())
    assert p.returncode == 0, p.stderr
    assert api.requests[0]["body"]["reasoning_effort"] == "low"


def test_soclaas_length_cutoff_is_truncated(api):
    api.plan["Rooms:"] = ["incomplete"]
    out = out_file()
    run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "q",
        "--max-output-tokens", "32", "--limit", "1", "--output", out)
    r = records(out)[0]
    assert r["status"] == "truncated" and r["finish_reason"] == "length"
    assert api.requests[0]["body"]["max_tokens"] == 32


# ------------------------------------------------------------- OpenRouter
def test_openrouter_needs_a_model(api):
    p = run("run_openai_b0.py", api, "--provider", "openrouter", "--limit", "1",
            "--output", out_file())
    assert p.returncode != 0 and "OPENROUTER_MODEL" in p.stderr


def test_b0_openrouter_sends_reasoning_object_and_records_routing(api):
    out = out_file()
    p = run("run_openai_b0.py", api, "--provider", "openrouter", "--model",
            "openai/gpt-5.4", "--reasoning-effort", "low", "--limit", "1", "--output", out)
    assert p.returncode == 0, p.stderr
    body = api.requests[0]["body"]
    assert api.requests[0]["path"].endswith("/chat/completions")
    assert body["model"] == "openai/gpt-5.4"
    assert body["reasoning"] == {"effort": "low"} and "reasoning_effort" not in body
    r = records(out)[0]
    assert r["provider"] == "openrouter" and r["status"] == "ok"
    assert r["routed_to"] == "FakeUpstream" and r["cost_usd"] == 0.0012


def test_openrouter_without_effort_sends_no_reasoning_setting(api):
    p = run("run_openai_b0.py", api, "--provider", "openrouter", "--model", "m",
            "--limit", "1", "--output", out_file())
    assert p.returncode == 0, p.stderr
    assert "reasoning" not in api.requests[0]["body"]


# -------------------------------------------------------------- streaming
@pytest.mark.parametrize("provider", ["soclaas", "openrouter"])
def test_chat_providers_stream_by_default(api, provider):
    api.plan["Rooms:"] = ["reasoning"]
    out = out_file()
    p = run("run_openai_b0.py", api, "--provider", provider, "--model", "m",
            "--limit", "1", "--output", out)
    assert p.returncode == 0, p.stderr
    body = api.requests[0]["body"]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    r = records(out)[0]
    assert r["status"] == "ok" and r["streamed"] is True
    assert json.loads(r["response"])["confidence"] == 0.8   # pieces put back together
    assert r["thinking_chars"] == 40 and r["output_tokens"] == 50
    assert r["routed_to"] == "FakeUpstream" and r["cost_usd"] == 0.0012


def test_no_stream_still_works(api):
    api.plan["Rooms:"] = ["reasoning"]
    out = out_file()
    p = run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "m",
            "--no-stream", "--limit", "1", "--output", out)
    assert p.returncode == 0, p.stderr
    assert "stream" not in api.requests[0]["body"]
    r = records(out)[0]
    assert r["status"] == "ok" and r["streamed"] is False and r["thinking_chars"] == 40


def test_streamed_think_tags_and_truncation_are_handled(api):
    api.plan["Rooms:"] = ["think", "incomplete"]
    out = out_file()
    run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "m",
        "--limit", "2", "--workers", "1", "--output", out)
    first, second = sorted(records(out), key=lambda r: r["id"])
    assert first["response"].startswith("{") and first["thinking_chars"] > 0
    assert second["status"] == "truncated" and second["finish_reason"] == "length"


def test_stream_closed_before_the_answer_is_truncated_not_empty(api):
    api.plan["Rooms:"] = ["cut"]
    out = out_file()
    p = run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "m",
            "--limit", "1", "--output", out)
    assert p.returncode == 0, p.stderr
    r = records(out)[0]
    assert r["status"] == "truncated" and r["incomplete_reason"] == "stream_closed"
    assert r["response"] == "" and r["thinking_chars"] == 40


def test_streaming_survives_a_call_longer_than_the_timeout(api):
    """The timeout bounds the gap between chunks when streaming, so a long
    call that keeps sending survives; the same call without streaming is cut
    off. This is the SoCLaaS 502 situation in miniature."""
    api.plan["Rooms:"] = ["slow", "slow"]
    streamed, waited = out_file(), out_file()
    run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "m", "--limit", "1",
        "--timeout", "1", "--max-retries", "0", "--output", streamed)
    run("run_openai_b0.py", api, "--provider", "soclaas", "--model", "m", "--limit", "1",
        "--timeout", "1", "--max-retries", "0", "--no-stream", "--output", waited)
    assert records(streamed)[0]["status"] == "ok"
    assert records(streamed)[0]["latency_seconds"] > 1
    assert records(waited)[0]["status"] == "error"
    assert records(waited)[0]["error_type"] == "APITimeoutError"


def test_split_thinking_handles_missing_and_unclosed_tags():
    assert split_thinking("<think>a</think> {}") == ("{}", len("<think>a</think>"))
    assert split_thinking("reasoning only</think>{}")[0] == "{}"
    assert split_thinking("{} <think>cut off")[0] == "{}"


# ------------------------------------------------------------ safeguards
@pytest.mark.parametrize("provider", ["openai", "soclaas", "openrouter"])
def test_b3_makes_three_calls_and_votes(api, provider):
    out = out_file()
    p = run("run_openai_safeguards.py", api, "--variant", "B3", "--provider", provider,
            "--model", "m", "--limit", "1", "--output", out)
    assert p.returncode == 0, p.stderr
    r = records(out)[0]
    assert r["call_count"] == 3 and len(api.requests) == 3
    assert r["final_object"]["status"] == "ANSWERED" and r["total_tokens"] == 450
    if provider != "openai":
        assert abs(r["cost_usd"] - 3 * 0.0012) < 1e-12


def test_b3_with_a_failed_sample_is_an_error_not_a_vote(api):
    api.plan["INDEPENDENT SAMPLE"] = ["ok", "500"]
    out = out_file()
    p = run("run_openai_safeguards.py", api, "--variant", "B3", "--limit", "1",
            "--max-retries", "0", "--output", out)
    assert p.returncode == 1
    r = records(out)[0]
    assert r["status"] == "error" and r["call_count"] == 2   # stopped at the failure
    again = run("run_openai_safeguards.py", api, "--variant", "B3", "--limit", "1",
                "--output", out)
    assert again.returncode == 0 and records(out)[-1]["status"] == "ok"


def test_b4_runs_draft_critique_revision(api):
    out = out_file()
    p = run("run_openai_safeguards.py", api, "--variant", "B4", "--limit", "1",
            "--output", out)
    assert p.returncode == 0, p.stderr
    prompts = [q["body"]["input"] for q in api.requests]
    assert len(prompts) == 3
    assert "CRITIQUE THE DRAFT" in prompts[1] and "CRITIQUE AND REVISION" in prompts[2]


def test_safeguard_cut_off_before_the_answer_is_truncated_not_empty(api):
    api.plan["CONFIDENCE AND ABSTENTION"] = ["cut"]
    out = out_file()
    p = run("run_openai_safeguards.py", api, "--variant", "B5", "--abstain-threshold",
            "0.0", "--provider", "soclaas", "--model", "m", "--limit", "1",
            "--output", out)
    assert p.returncode == 0, p.stderr
    r = records(out)[0]
    assert r["status"] == "truncated" and r["truncated_calls"] == 1
    assert r["response"] == "" and r["final_object"] is None


def test_b5_threshold_is_part_of_the_run_settings(api):
    out = out_file()
    assert run("run_openai_safeguards.py", api, "--variant", "B5", "--abstain-threshold",
               "0.5", "--limit", "1", "--output", out).returncode == 0
    p = run("run_openai_safeguards.py", api, "--variant", "B5", "--abstain-threshold",
            "0.9", "--limit", "1", "--output", out)
    assert p.returncode != 0 and "abstain_threshold" in p.stderr


def test_list_models(api):
    p = run("llm_client.py", api, "--provider", "soclaas")
    assert p.returncode == 0 and p.stdout.strip() == "fake-model"
