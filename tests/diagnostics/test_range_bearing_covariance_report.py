"""CPU-only synthetic reporting diagnostics; no sampler or promotion evidence."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("range_bearing_report", ROOT / "scripts/run_range_bearing_covariance_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path):
    args = SimpleNamespace(output=tmp_path, particles=1008, replicates=4,
                           dtype="float64", no_tf32=True)
    reference = dict(valid=True, replicates=16, value=0., score=[0., 0.],
                     mcse=[.001]*3, replicate_values=[0.]*16,
                     replicate_scores=[[0., 0.]]*16)
    rows = []
    for ds, seed in enumerate(runner.HELDOUT):
        for point in range(2):
            for n in (16384, 65536, 1008):
                row = dict(reference)
                if n == 1008:
                    row["replicate_scores"] = [[1.5, 0.]]*16
                runner.save(tmp_path / "references" / f"{seed}-{point}-{n}.json", row)
            for arm, error in (("new", .5), ("old", 2.), ("prior", .7), ("global_guide", 1.)):
                for rep in range(4):
                    rows.append(dict(data_seed=seed, point=point, arm=arm,
                        seed=31001+100*ds+rep, valid=True, score=[error+.01*rep, 0.],
                        value=.01, wall_seconds=.1))
    return args, rows


def report(args, rows):
    (args.output / "compare.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    runner.report(args)
    return runner.load(args.output / "report.json")


def test_complete_primary_and_heuristics(tmp_path):
    args, rows = fixture(tmp_path)
    result = report(args, rows)
    assert result["complete"] and result["heuristic_suite_complete"]
    assert all(c["verdict"] == "conditional_score_improvement_with_reference_sensitivity" for c in result["cells"])


def test_invalid_heuristic_keeps_primary_interval(tmp_path):
    args, rows = fixture(tmp_path)
    next(r for r in rows if r["arm"] == "global_guide")["valid"] = False
    result = report(args, rows)
    assert result["complete"] and not result["heuristic_suite_complete"]
    cell = result["cells"][0]
    assert cell["primary_complete"] and cell["paired_95_student_interval"] is not None
    assert cell["heuristic_dominance"] == "incomplete_heuristic_screen"


def test_invalid_primary_blocks_interval(tmp_path):
    args, rows = fixture(tmp_path)
    next(r for r in rows if r["arm"] == "new")["valid"] = False
    result = report(args, rows)
    assert not result["complete"]
    assert result["cells"][0]["paired_95_student_interval"] is None


def test_duplicate_design_blocks_primary(tmp_path):
    args, rows = fixture(tmp_path)
    primary = [r for r in rows if r["arm"] == "new" and r["data_seed"] == 1704 and r["point"] == 0]
    primary[1]["seed"] = primary[0]["seed"]
    result = report(args, rows)
    assert not result["complete"]
    assert result["cells"][0]["paired_95_student_interval"] is None


def test_invalid_reference_rejected(tmp_path):
    args, rows = fixture(tmp_path)
    path = args.output / "references/1704-0-65536.json"
    ref = runner.load(path)
    ref["mcse"][0] = -1.
    runner.save(path, ref)
    with pytest.raises(ValueError, match="Invalid independent reference"):
        report(args, rows)
