"""
Tests for the scorer in cops_robbers.py, which every Solution B result
depends on. They pin down how failed, abstained and retried answers count.

Run with: uv run pytest
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cops_robbers import (  # noqa: E402
    load_responses,
    parse_response,
    score_one,
    summarise,
)

# Three rooms: capture is guaranteed from A and C, the intruder escapes from B.
INST = {"id": "x", "n": 3, "truth": {"A": True, "B": False, "C": True}}


def answer(labels, confidence=0.9, **extra):
    return json.dumps({"capture_guaranteed": labels, "confidence": confidence, **extra})


RIGHT = answer({"A": True, "B": False, "C": True})
UNSAFE = answer({"A": True, "B": True, "C": True})       # claims capture at B
WASTEFUL = answer({"A": False, "B": False, "C": True})   # gives up on A
GARBAGE = "I think the guards win."
ABSTAIN = json.dumps({"status": "UNDETERMINED", "confidence": 0.3})


def scores(*responses, allow_abstain=False):
    return summarise([score_one(INST, r) for r in responses], allow_abstain)


# ------------------------------------------------------------ single answers
def test_correct_answer_is_exact():
    r = score_one(INST, RIGHT)
    assert r["exact"] and r["false_win"] == 0 and r["false_loss"] == 0


def test_unsafe_and_wasteful_rooms_are_counted_separately():
    assert score_one(INST, UNSAFE)["false_win"] == 1
    assert score_one(INST, WASTEFUL)["false_loss"] == 1


def test_missing_room_or_non_boolean_label_is_unparseable():
    assert not score_one(INST, answer({"A": True, "B": False}))["parsed"]
    assert not score_one(INST, answer({"A": True, "B": "no", "C": True}))["parsed"]


def test_boolean_confidence_is_not_a_confidence():
    assert score_one(INST, answer({"A": True, "B": False, "C": True}, True))["confidence"] is None


# ------------------------------------------------------------------ parsing
def test_parser_takes_the_final_answer_after_prose_and_drafts():
    text = ("Room sets like {A, B} are not JSON. Draft: "
            + answer({"A": False, "B": False, "C": False})
            + " On reflection, the final answer is: " + RIGHT)
    assert parse_response(text)["capture_guaranteed"] == {"A": True, "B": False, "C": True}


def test_parser_ignores_braces_inside_strings():
    text = json.dumps({"status": "UNDETERMINED", "abstention_reason": "unsure about {A, B}"})
    assert parse_response(text)["status"] == "UNDETERMINED"


def test_parser_ignores_objects_without_an_answer_field():
    assert parse_response('Adjacency: {"A": ["B"], "B": ["A", "C"]}') is None


# ------------------------------------------------------------- aggregation
def test_unparseable_answers_count_as_wrong():
    s = scores(RIGHT, GARBAGE)
    assert s["exact_match_rate"] == 0.5          # previously reported as 1.0
    assert s["room_accuracy"] == 3 / 6
    assert s["unparseable"] == 1 and s["unparseable_rate"] == 0.5
    assert s["exact_match_rate_answered_only"] == 1.0


def test_abstention_counts_as_wrong_unless_allowed():
    s = scores(RIGHT, ABSTAIN)
    assert s["exact_match_rate"] == 0.5 and s["coverage"] == 0.5
    s = scores(RIGHT, ABSTAIN, allow_abstain=True)
    assert s["exact_match_rate"] == 1.0 and s["coverage"] == 0.5
    assert s["abstained"] == 1


def test_room_rates_cover_answered_rooms_only():
    s = scores(UNSAFE, WASTEFUL)
    assert s["unsafe_room_rate"] == 1 / 6 and s["wasteful_room_rate"] == 1 / 6
    assert s["room_accuracy"] == 4 / 6 and s["exact_match_rate"] == 0.0


def test_brier_score_uses_whole_instance_correctness():
    s = scores(answer({"A": True, "B": False, "C": True}, 0.8),
               answer({"A": True, "B": True, "C": True}, 0.6))
    assert abs(s["brier"] - ((0.8 - 1) ** 2 + 0.6 ** 2) / 2) < 1e-12


# ------------------------------------------------------- retried records
def write_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    f.close()
    return f.name


def test_a_retried_instance_is_scored_once_from_its_last_record():
    path = write_jsonl([{"id": "x", "status": "error", "response": ""},
                        {"id": "x", "status": "ok", "response": RIGHT}])
    latest, superseded = load_responses(path)
    assert superseded == 1 and latest["x"]["response"] == RIGHT


def test_score_command_end_to_end():
    instances = write_jsonl([INST])
    responses = write_jsonl([{"id": "x", "status": "error", "response": ""},
                             {"id": "x", "status": "ok", "response": RIGHT}])
    out = subprocess.run(
        [sys.executable, str(ROOT / "cops_robbers.py"), "score",
         "--instances", instances, "--responses", responses],
        capture_output=True, text=True, check=True).stdout
    overall = json.loads(out.split("OVERALL", 1)[1].split("\nn=", 1)[0])
    assert overall["instances"] == 1 and overall["exact_match_rate"] == 1.0
    assert overall["superseded_records"] == 1
