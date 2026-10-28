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

The runner constructs B1--B5 prompts at runtime from the existing B0 prompts:

```bash
.venv-openai/bin/python run_openai_safeguards.py \
  --variant B1 --prompts dataset/prompts_B0_test.jsonl \
  --ids-from dataset/instances_test_80.jsonl \
  --model gpt-5.6-luna --reasoning-effort medium --timeout 600 \
  --output work/openai_b1_test80.jsonl
```

Use the same command with `B2`, `B3`, or `B4`. B3 and B4 intentionally make
three calls per instance, so their latency and token totals must be reported.

For B5, first run the raw B5 condition on the development set, inspect the
threshold table, and freeze the selected threshold:

```bash
.venv-openai/bin/python run_openai_safeguards.py \
  --variant B5 --prompts dataset/prompts_B0_dev.jsonl \
  --abstain-threshold 0.0 \
  --model gpt-5.6-luna --reasoning-effort medium --timeout 600 \
  --output work/openai_b5_dev.jsonl

.venv-openai/bin/python select_b5_threshold.py \
  --instances dataset/instances_dev.jsonl \
  --responses work/openai_b5_dev.jsonl
```

Then rerun B5 with the selected threshold on the held-out set. Do not choose a
threshold after looking at held-out labels.

## Evaluation

Score every condition with the matching instance file:

```bash
.venv-openai/bin/python cops_robbers.py score \
  --instances dataset/instances_test_80.jsonl \
  --responses work/openai_b1_test80.jsonl
```

Report exact-match rate, unsafe-room rate, wasteful-room rate, unparseable
rate, confidence/Brier metrics, latency, total tokens, and API calls per
instance.  Break the results down by room count and maximum classical capture
depth so the report captures the professor's suggested complexity-versus-
correctness relationship.  For B5 also report coverage and conditional
accuracy/risk among answered instances; an abstention is not silently counted
as a correct label.

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

