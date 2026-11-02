"""
Tests for the Solution B prompt builders and B5 answer handling.

Run with: uv run pytest
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cops_robbers import score_one, summarise  # noqa: E402
from select_b5_threshold import evaluate_threshold  # noqa: E402
from solution_b_prompts import (  # noqa: E402
    apply_abstention,
    build_b1_prompt,
    build_b2_prompt,
    build_b3_prompt,
    build_b5_prompt,
    parse_candidate,
)

PROMPT = json.loads((ROOT / "dataset" / "prompts_B0_dev.jsonl").open().readline())["prompt"]
INST = json.loads((ROOT / "dataset" / "instances_dev.jsonl").open().readline())
ROOMS = list(INST["truth"])


def test_every_variant_keeps_the_per_room_answer_template():
    for build in (build_b1_prompt, build_b2_prompt, build_b3_prompt, build_b5_prompt):
        prompt = build(PROMPT)
        assert all(f'"{room}": true/false' in prompt for room in ROOMS), build.__name__


def test_b5_prompt_has_no_placeholder_values_to_copy():
    prompt = build_b5_prompt(PROMPT)
    assert '"ROOM"' not in prompt and '"confidence":0.0' not in prompt


def test_b5_invalid_output_is_a_failure_not_an_abstention():
    assert apply_abstention(parse_candidate("not json", ROOMS), ROOMS, 0.5) is None
    low = {"capture_guaranteed": INST["truth"], "confidence": 0.2}
    assert apply_abstention(parse_candidate(json.dumps(low), ROOMS), ROOMS, 0.5)["status"] == "UNDETERMINED"


def test_b5_boolean_confidence_does_not_bypass_threshold():
    answer = {"capture_guaranteed": INST["truth"], "confidence": True}
    candidate = parse_candidate(json.dumps(answer), ROOMS)
    assert candidate["confidence"] is None
    assert apply_abstention(candidate, ROOMS, 0.5)["status"] == "UNDETERMINED"


def test_threshold_selection_counts_invalid_output_as_failure():
    records = [{"id": INST["id"], "response": "not json"}]
    s = evaluate_threshold({INST["id"]: INST}, records, 0.5)
    assert s["unparseable"] == 1 and s["abstained"] == 0 and s["exact_match_rate"] == 0.0


def test_threshold_selection_abstains_on_low_confidence():
    good = json.dumps({"status": "ANSWERED", "capture_guaranteed": INST["truth"],
                       "confidence": 0.4})
    s = evaluate_threshold({INST["id"]: INST}, [{"id": INST["id"], "response": good}], 0.5)
    assert s["abstained"] == 1 and s["coverage"] == 0.0
    s = evaluate_threshold({INST["id"]: INST}, [{"id": INST["id"], "response": good}], 0.3)
    assert s["exact_match_rate"] == 1.0 and s["coverage"] == 1.0


def test_abstention_rate_is_reported():
    s = summarise([score_one(INST, json.dumps({"status": "UNDETERMINED"}))], True)
    assert s["abstention_rate"] == 1.0
