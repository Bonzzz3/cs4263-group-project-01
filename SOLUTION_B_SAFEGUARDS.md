# Solution B safeguards

This implementation keeps B0 as the plain-prompt control and adds five
separate inference-time conditions.  No condition sends classical labels,
solver states, checker feedback, or a generated/executed classical solver to
the model.

## Conditions

| Condition | Additional calls | Main risk targeted | Final answer rule |
| --- | ---: | --- | --- |
| B0 | 1 | Baseline | Raw JSON response |
| B1 | 1 | Misread graph, positions, or turns | Structured private ledger, raw JSON |
| B2 | 1 | Missing an escape route | Adversarial refutation before accepting TRUE |
| B3 | 3 | One-off instability | Per-room majority vote across independent calls |
| B4 | 3 | Unchecked errors | Draft, separate critic, then revision |
| B5 | 1 | Overconfident mistakes | Development-selected confidence threshold and abstention |

The B3 majority vote is performed by the wrapper using only the three model
outputs.  The B4 critic receives the original task and the draft, but no
ground-truth answer.  B5 threshold selection is performed on development data
only; the selected threshold must be frozen before held-out evaluation.

These are safeguards because each adds a defined error-control mechanism, not
because each is expected to improve every metric on every model:

| Safeguard | Mechanism that can improve reliability | Evidence that must be reported |
| --- | --- | --- |
| B1 | An explicit state/adjacency/turn checklist targets graph-tracking errors. | Exact match and unsafe-room rate versus B0. |
| B2 | Active falsification targets missed escape routes and the possible-versus-guaranteed mistake. | Unsafe-room rate, especially on larger/deeper instances. |
| B3 | Independent samples reduce some one-off errors through aggregation. | Accuracy, consensus strength, three-call cost, and correlated-error failures. |
| B4 | A separate critique pass can expose illegal moves, invented edges, and contradictions before revision. | Accuracy, unsafe-room rate, and three-call latency/token cost. |
| B5 | Selective prediction avoids forcing a low-confidence answer. | Coverage, conditional accuracy/risk, and abstention rate. |

B3 agreement must not be described as calibrated confidence: correlated model
errors can make all three samples agree on the same wrong answer.  If B3 does
not beat B0, that is a valid result showing the limitation of self-consistency
for this task, rather than a reason to remove the condition.  Likewise, B5 is
not allowed to trade away coverage invisibly; all abstentions and answered-only
metrics must be reported.

The safeguards remain prompt-level controls over the contemporary model's own
reasoning.  B1 and B2 do not enumerate the complete game state space and do
not call, generate, or reproduce the classical solver.  The runner also never
sends classical labels, solver states, or checker feedback to the model.  This
boundary is important because the project clarification prohibits turning
Solution B into a classical planning component or hybrid pipeline.

## Running

All runners work with three providers through the same OpenAI SDK:

| Provider | Flag | API | Needs | Reasoning effort sent as |
| --- | --- | --- | --- | --- |
| OpenAI (default) | `--provider openai` | Responses | `OPENAI_API_KEY` | `reasoning.effort` (default medium) |
| SoCLaaS | `--provider soclaas` | Chat Completions | `SOCLAAS_API_KEY`, and `--model` or `SOCLAAS_MODEL` | `reasoning_effort`, only if given |
| OpenRouter | `--provider openrouter` | Chat Completions | `OPENROUTER_API_KEY`, and `--model` or `OPENROUTER_MODEL` | `reasoning.effort`, only if given |

Put the keys in a local `.env` (git-ignored) and pass it with
`uv run --env-file .env`. List a provider's models with
`uv run --env-file .env python llm_client.py --provider soclaas`.
OpenRouter names models by vendor (e.g. `openai/gpt-5.4`) and each record
notes which vendor served the call (`routed_to`) and its cost (`cost_usd`).
A model reached through OpenRouter is the same model as through its vendor's
own API, but the request goes through Chat Completions rather than the
Responses API, so say which route was used when comparing results.

Runs are resumable: rerun the same command and finished instances are skipped
while API failures are retried. `--workers` (default 4) sets how many
instances run in parallel, and `--max-retries` (default 3) how often the SDK
retries a rate limit, timeout or server error. Use a separate `--output` file
per provider, model, effort and variant; the runner refuses to mix them.

SoCLaaS and OpenRouter calls are streamed. A long reasoning call sends nothing
until it finishes, and SoCLaaS's gateway gives up after about 5 minutes with a
502; streaming keeps data flowing, so the call survives. While streaming,
`--timeout` is the longest wait for the next chunk rather than for the whole
answer. `--no-stream` turns streaming off.

The runner constructs B1--B5 prompts at runtime from the existing B0 prompts:

```bash
uv run --env-file .env python run_openai_safeguards.py \
  --variant B1 --prompts dataset/prompts_B0_test.jsonl \
  --ids-from dataset/instances_test_80.jsonl \
  --model gpt-5.6-luna --reasoning-effort medium \
  --output work/openai_b1_test80.jsonl

uv run --env-file .env python run_openai_safeguards.py \
  --variant B1 --provider soclaas --model qwen3.6:35b \
  --prompts dataset/prompts_B0_test.jsonl \
  --ids-from dataset/instances_test_80.jsonl \
  --output work/soclaas_b1_test80.jsonl
```

B0 uses `run_openai_b0.py` with the same flags (no `--variant`). Use the same
command with `B2`, `B3`, or `B4`. B3 and B4 intentionally make three calls per
instance, so their latency and token totals must be reported. If any call for
an instance fails, the whole instance is retried, so B3 never votes with a
missing sample.

For B5, first run the raw B5 condition on the development set, inspect the
threshold table, and freeze the selected threshold:

```bash
uv run --env-file .env python run_openai_safeguards.py \
  --variant B5 --prompts dataset/prompts_B0_dev.jsonl \
  --abstain-threshold 0.0 \
  --model gpt-5.6-luna --reasoning-effort medium \
  --output work/openai_b5_dev.jsonl

uv run python select_b5_threshold.py \
  --instances dataset/instances_dev.jsonl \
  --responses work/openai_b5_dev.jsonl
```

Then rerun B5 with the selected threshold on the held-out set. Do not choose a
threshold after looking at held-out labels.

## Evaluation

Put every run on the same instances into one table with
`analyse_results.py`. It reports exact-match rate (with a 95% Wilson
interval), room accuracy (with a 95% bootstrap interval), unsafe and wasteful
room rates, failed answers, B5 coverage, Brier score, and calls, tokens and
latency per instance, alongside two no-reasoning baselines. It also breaks the
accuracy down by room count and by longest optimal chase, the
complexity-versus-correctness relationship the professor suggested:

```bash
uv run python analyse_results.py --instances dataset/instances_test_80.jsonl \
  work/openai_b0_test80.jsonl work/openai_b1_test80.jsonl \
  work/soclaas_b0_test80.jsonl work/soclaas_b1_test80.jsonl \
  --out results/solution_b_test80.csv
```

To inspect a single run, `cops_robbers.py score` still works; pass
`--allow-abstain` only for B5:

```bash
uv run python cops_robbers.py score \
  --instances dataset/instances_test_80.jsonl \
  --responses work/openai_b1_test80.jsonl
```

An unparseable or missing answer counts as wrong, and so does an
UNDETERMINED answer from any variant other than B5 (for example a B3 tie).
For B5, abstentions are reported as coverage, and accuracy is conditional on
the answered instances; an abstention is never silently counted as a correct
label.

An improvement claim should be made only when the recorded results support it.
The intended safeguard success criterion is lower unsafe-room rate than B0,
with exact-match rate and coverage reported alongside it.  If a safeguard does
not improve the selected metric, report that negative result rather than
quietly changing the protocol.

## Literature support

- Graph reasoning is a known weakness for language models; ProGraph reports
  poor direct graph-reasoning performance and motivates explicit graph
  structure handling: [ProGraph](https://arxiv.org/abs/2409.19667).
- B3 follows the independent-path aggregation idea in
  [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171).
- B4 is an inference-time feedback/revision condition inspired by
  [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651).
  The report should also acknowledge the negative result that language models
  do not reliably self-correct reasoning errors:
  [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798).
- B2 should be treated as a measurable counterexample-search safeguard, not as
  a guaranteed proof method; counterexample generation itself is difficult in
  [REFUTE](https://arxiv.org/abs/2502.19414).
- B5 follows selective prediction/abstention principles from
  [The Art of Abstention](https://aclanthology.org/2021.acl-long.84/) and
  should be evaluated with coverage-risk trade-offs rather than accuracy alone.

