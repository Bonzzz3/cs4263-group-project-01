#!/usr/bin/env python3
"""Run Solution B0 (plain prompt, no tools, no code) over a prompts file via the API.
Usage: ANTHROPIC_API_KEY=... python3 run_b0.py dataset/prompts_B0_dev.jsonl responses_B0_dev.jsonl [N]
Resumable: ids already in the output file are skipped. Score with cops_robbers.py score."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import anthropic  # pip install anthropic

MODEL = "claude-opus-5-5"
client = anthropic.Anthropic()

def ask(row):
    t0 = time.time()
    for attempt in range(5):
        try:
            m = client.messages.create(model=MODEL, max_tokens=2000, temperature=0,
                                       messages=[{"role": "user", "content": row["prompt"]}])
            return {"id": row["id"], "response": m.content[0].text,
                    "in_tokens": m.usage.input_tokens, "out_tokens": m.usage.output_tokens,
                    "latency_s": round(time.time() - t0, 2)}
        except Exception as e:
            time.sleep(2 ** attempt)
    return {"id": row["id"], "response": "", "error": "failed"}

def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = [json.loads(l) for l in open(src)][:limit]
    done = {json.loads(l)["id"] for l in open(dst)} if os.path.exists(dst) else set()
    todo = [r for r in rows if r["id"] not in done]
    with open(dst, "a") as f, ThreadPoolExecutor(4) as ex:
        for out in ex.map(ask, todo):
            f.write(json.dumps(out) + "\n"); f.flush()
    print("done", len(todo))
main()
