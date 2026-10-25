#!/usr/bin/env python3
"""Run Solution B0 (plain prompt, no tools, no code) over a prompts file via the API.
Usage: python3 run_b0.py dataset/prompts_B0_dev.jsonl responses_B0_dev.jsonl [N]
Provider: if SOCLAAS_API_KEY is set, uses SoCLaaS (OpenAI-compatible; model from
SOCLAAS_MODEL, base URL from SOCLAAS_BASE_URL); otherwise Anthropic (ANTHROPIC_API_KEY).
B0_MAX_TOKENS overrides the output token cap (default 2000).
Resumable: ids already answered in the output file are skipped; failed ids are retried.
Score with cops_robbers.py score."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

MAX_TOKENS = int(os.environ.get("B0_MAX_TOKENS", 2000))

if os.environ.get("SOCLAAS_API_KEY"):
    from openai import OpenAI  # uv run --with openai
    PROVIDER = "soclaas"
    MODEL = os.environ.get("SOCLAAS_MODEL", "qwen3.6:35b")
    client = OpenAI(api_key=os.environ["SOCLAAS_API_KEY"],
                    base_url=os.environ.get("SOCLAAS_BASE_URL",
                                            "https://soclaas-api.comp.nus.edu.sg/v1"))
else:
    import anthropic  # uv run --with anthropic
    PROVIDER = "anthropic"
    MODEL = "claude-opus-5-5"
    client = anthropic.Anthropic()

def call(prompt):
    if PROVIDER == "soclaas":
        m = client.chat.completions.create(model=MODEL, max_tokens=MAX_TOKENS, temperature=0,
                                           messages=[{"role": "user", "content": prompt}])
        return m.choices[0].message.content or "", m.usage.prompt_tokens, m.usage.completion_tokens
    m = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, temperature=0,
                               messages=[{"role": "user", "content": prompt}])
    return m.content[0].text, m.usage.input_tokens, m.usage.output_tokens

def ask(row):
    t0 = time.time()
    for attempt in range(5):
        try:
            text, tin, tout = call(row["prompt"])
            return {"id": row["id"], "response": text, "model": MODEL,
                    "in_tokens": tin, "out_tokens": tout,
                    "latency_s": round(time.time() - t0, 2)}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"{row['id']} attempt {attempt + 1}: {err}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return {"id": row["id"], "response": "", "model": MODEL, "error": err}

def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = [json.loads(l) for l in open(src)][:limit]
    done, kept = set(), []
    if os.path.exists(dst):
        for l in open(dst):
            rec = json.loads(l)
            if "error" in rec:  # dropped here, retried below
                continue
            if rec.get("model", MODEL) != MODEL:
                sys.exit(f"{dst} has responses from {rec['model']}, not {MODEL}; use another file")
            done.add(rec["id"]); kept.append(l)
        with open(dst, "w") as f:
            f.writelines(kept)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{PROVIDER} {MODEL}: {len(todo)} to run", file=sys.stderr)
    with open(dst, "a") as f, ThreadPoolExecutor(4) as ex:
        for out in ex.map(ask, todo):
            f.write(json.dumps(out) + "\n"); f.flush()
    print("done", len(todo))
main()
