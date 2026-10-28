# Cops-and-Robbers Security Patrol (CS4263 group project)

Can two guards always catch an intruder in a building, and from which start
rooms? We answer this exactly with a classical solver (Solution A) and compare
it with an LLM reasoning directly over the problem (Solution B), using the
solver's labels as ground truth.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
```

This installs Python 3.12 (pinned in `.python-version`) and pytest. The project
code itself uses the standard library only.

## Repository layout

| Path | Part | Purpose |
| --- | --- | --- |
| `cops_robbers.py` | Shared | Solver, instance generator, prompt builder and scorer, with a command-line interface |
| `build_dataset.py` | Dataset | Builds the dev and test splits from fixed seeds |
| `verify_dataset.py` | Dataset | Checks every label with capture and escape certificates, without trusting the solver |
| `crosscheck.py` | Dataset | Second implementation written from the rules alone; must agree on every label and capture length |
| `dataset/` | Dataset | Instances, B0 prompts and composition statistics (see [Dataset](#dataset)) |
| `tests/test_solver.py` | Solution A | Solver tests: hand-worked cases, a graph-theory theorem, the Petersen graph and an independent minimax search |
| `bench_solver.py` | Solution A | Runtime and state counts against room count; writes `results/solver_benchmark.csv` |
| `run_b0.py` | Solution B | Runs the B0 baseline prompts through an LLM API (see the script for usage) |

Each script's docstring has the details; this README covers how the pieces fit.

## Common commands

```
uv run pytest                          # solver tests
uv run python verify_dataset.py        # certificate check of every label
uv run python crosscheck.py            # independent re-implementation check
uv run python build_dataset.py         # rebuild dataset/ from fixed seeds
uv run python bench_solver.py          # solver scaling benchmark
uv run python cops_robbers.py solve --n 4 --edges 0-1,1-2,2-3,3-0 --guards 0,0
uv run python cops_robbers.py score --instances dataset/instances_test.jsonl --responses responses.jsonl
```

Rebuilding reproduces the dataset exactly; only the timing fields (`solve_ms`,
`mean_solve_ms`) change between runs.

## The problem

Rooms and corridors form a connected graph. Two guards and one intruder each
occupy a room, and everyone can see everyone.

- The guards move first, then the intruder, alternating.
- On a turn, each piece stays put or moves along one corridor. Guards may share
  a room.
- The intruder is caught as soon as it shares a room with a guard, including at
  the start.
- The two guards act as one player choosing a joint move; no communication
  between them is modelled.

Given the graph and the guards' start rooms, each possible intruder start room
is labelled "capture guaranteed" (the guards can force a capture against every
intruder strategy) or "intruder escapes". The B0 prompt states the same rules.

## Dataset

One family, `coretree-2g`: a 10-room Petersen-type core with rooms attached one
at a time up to 12, 14, ..., 26 rooms, room names shuffled, and 2 guards in
random start rooms. Every instance has both outcomes among its start rooms.
Two other families were rejected: with 2 guards, random sparse graphs were won
from every room (nothing to label), and on plain Petersen-type graphs every
capture took one move (labels reduced to an adjacency check).

- `dev`: 10 instances per size (80 in total), for prompt development.
- `test`: 30 instances per size (240 in total), for reported results only.

Each line of `dataset/instances_*.jsonl` is one instance:

| Field | Meaning |
| --- | --- |
| `id`, `split`, `family` | Identity |
| `n`, `k` | Number of rooms and guards (k = 2) |
| `edges`, `corridors` | Corridors as pairs of room indices, and their count |
| `guard_start` | Guards' start rooms (indices) |
| `truth` | Room name → capture guaranteed |
| `rounds` | Room name → half-moves to capture under optimal play, or `null` if the intruder escapes |
| `escape_rooms`, `max_half_moves` | Number of escape rooms, and the longest capture |
| `solve_ms` | Solver time when the dataset was built (machine-dependent) |

Room names are letters A, B, C, ... in index order.

## Scoring LLM responses

`responses.jsonl` has one object per line: `{"id": "...", "response": "<model text>"}`.
The model must answer with JSON: `{"capture_guaranteed": {"A": true, ...}, "confidence": 0.8}`.
