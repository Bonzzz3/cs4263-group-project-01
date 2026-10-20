# Cops-and-robbers dataset package (CS4263 group project)

Exact ground truth for a pursuit game on graphs, plus the tools to regenerate,
verify, prompt and score. Needs Python 3 only; no dependencies.

## Files

| File | Purpose |
| --- | --- |
| `cops_robbers.py` | Solver (Solution A), instance generator, B0 prompt builder, scorer |
| `build_dataset.py` | Builds the dev and test splits (fixed seeds) |
| `verify_dataset.py` | Independent certificate check of every label |
| `crosscheck.py` | Second implementation written from the rules text alone; must agree on every label and capture length |
| `run_b0.py` | Scripted API runner for Solution B0 (needs `ANTHROPIC_API_KEY`; resumable) |
| `dataset/instances_dev.jsonl` | 100 instances for the pilot and prompt development |
| `dataset/instances_test.jsonl` | 300 held-out instances for reported results |
| `dataset/prompts_B0_*.jsonl` | `{"id", "prompt"}` baseline prompts, one per instance |
| `dataset/dataset_stats.csv` | Composition per split, family and room count |

## Commands

```
python3 build_dataset.py
python3 verify_dataset.py
python3 crosscheck.py
ANTHROPIC_API_KEY=... python3 run_b0.py dataset/prompts_B0_dev.jsonl responses_B0_dev.jsonl 3
python3 cops_robbers.py score --instances dataset/instances_test.jsonl --responses responses.jsonl
python3 cops_robbers.py solve --n 4 --edges 0-1,1-2,2-3,3-0 --guards 0
```

Seeds are fixed: a rebuild reproduces the instances, labels and prompts exactly
(checked). Only the wall-clock fields `solve_ms` and `mean_solve_ms` vary between runs.

`responses.jsonl` has one object per line: `{"id": "...", "response": "<model text>"}`.
The model must answer with JSON: `{"capture_guaranteed": {"A": true, ...}, "confidence": 0.8}`.

## Modelling choice

The guards are one player choosing a joint move for all guards each turn (no
communication or negotiation is modelled), and k is fixed at 2 for every instance.

## Game rules (exactly as in the prompt)

Guards move first, then the intruder, alternating. On its turn each guard stays
or moves along one corridor, and so does the intruder. The intruder is caught as
soon as it shares a room with any guard, including at the start. The intruder
sees the guards and plays to avoid capture forever. "Capture guaranteed" means
the guards can force capture in finitely many turns against every intruder
strategy.

## Instance fields

`id, split, family, n (rooms), k (guards), edges (room indices), guard_start,
truth (room name -> capture guaranteed), rounds (half-moves to capture, null if
the intruder escapes), corridors, escape_rooms, max_half_moves, solve_ms`.
Room names are letters A, B, C, ... in index order.

## Families

- `coretree-2g` (the only family): 2 guards, 10-room Petersen-type core plus
  pendant rooms up to 12, 14, ..., 30 rooms in total, rooms shuffled.
  2 guards on random sparse graphs were tried and rejected: the guards won from
  every room, so there was nothing to label. A plain Petersen-type graph was rejected: with 2 guards every
  win was within one half-move, so labels reduced to an adjacency check.

Every instance has both outcomes among its intruder start rooms.

## Important

The solver, verifier and scorer are evaluation tools only. They must never run
inside Solution B or inside any safeguard, or the two solutions stop being
separate as the project brief requires.
