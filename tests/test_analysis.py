"""
Tests for analyse_results.py.

Run with: uv run pytest
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyse_results import wilson  # noqa: E402

INSTANCES = ROOT / "dataset" / "instances_dev.jsonl"


def test_wilson_interval():
    low, high = wilson(0, 80)
    assert low == 0 and abs(high - 0.0458) < 1e-3
    low, high = wilson(40, 80)
    assert abs(low - 0.393) < 1e-3 and abs(high - 0.607) < 1e-3


def test_a_run_answering_capture_everywhere_matches_that_baseline():
    insts = [json.loads(line) for line in INSTANCES.open()]
    tmp = Path(tempfile.mkdtemp())
    responses = tmp / "all_true.jsonl"
    responses.write_text("".join(json.dumps({
        "id": i["id"], "variant": "B1", "provider": "openai", "model": "fake",
        "reasoning_effort": "low", "status": "ok", "call_count": 1,
        "response": json.dumps({"capture_guaranteed": {r: True for r in i["truth"]},
                                "confidence": 0.5})}) + "\n" for i in insts))
    out = tmp / "results.csv"
    subprocess.run([sys.executable, str(ROOT / "analyse_results.py"),
                    "--instances", str(INSTANCES), str(responses), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    rows = {(r["run"], r["group"]): r for r in csv.DictReader(out.open())}
    baseline = rows[("baseline: all capture", "all")]
    run = rows[("B1 openai/fake (low)", "all")]
    for metric in ("exact_match_rate", "room_accuracy", "unsafe_room_rate"):
        assert run[metric] == baseline[metric]
    assert float(run["brier"]) == 0.25   # confidence 0.5, never exactly right


def test_missing_instances_count_as_wrong():
    insts = [json.loads(line) for line in INSTANCES.open()][:10]
    tmp = Path(tempfile.mkdtemp())
    responses = tmp / "partial.jsonl"
    responses.write_text("".join(json.dumps({
        "id": i["id"], "status": "ok",
        "response": json.dumps({"capture_guaranteed": i["truth"]})}) + "\n" for i in insts))
    p = subprocess.run([sys.executable, str(ROOT / "analyse_results.py"),
                        "--instances", str(INSTANCES), str(responses)],
                       check=True, capture_output=True, text=True)
    assert "70 missing" in p.stdout
    line = next(x for x in p.stdout.splitlines() if x.startswith("B0 "))
    assert " 12.5 [" in line   # 10 of 80 exactly right, not 100%
