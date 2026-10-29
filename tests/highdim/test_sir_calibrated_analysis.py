"""Synthetic reporting checks; no scientific performance evidence."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("sir_calibrated_analysis", ROOT / "scripts/analyze_sir_calibrated_guidance.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def fixture(directory):
    errors = dict(plain=2., csmc_quadratic=1., kernel=3., plain_quadratic_beta=2., plain_kernel_beta=2., prior=.5, local=2.5)
    manifest = dict(design="independent_beta_matched_guidance_v1", status="complete",
        completed_rows=112, arms=list(errors), trace_fields=["valid", "moment_valid", "correction_cap_valid"],
        seeds={}, arms_by_dataset={}, reference=dict(datasets={}))
    rows, smoke = [], []
    for data_seed in (26100831, 26100832):
        identity = str(data_seed)
        manifest["seeds"][identity] = list(range(8))
        manifest["arms_by_dataset"][identity] = {arm: [1., 0., 0.] for arm in errors}
        manifest["reference"]["datasets"][identity] = dict(score=[0., 0., 0.])
        for arm, error in errors.items():
            for seed in range(8):
                rows.append(dict(data_seed=data_seed, seed=seed, arm=arm, beta=[1., 0., 0.],
                    reset_epsilon=1., reset_steps=5, valid=True, steps_completed=[50] * 3,
                    value=0., score=[error, 0., 0.], seconds=1., traces=[[[1., 1., 1.]] * 50] * 3))
            smoke.append(dict(data_seed=data_seed, arm=arm, valid=True,
                derivative_checks=[dict(passed=True) for _ in range(6)],
                richardson=[dict(passed=True) for _ in range(3)]))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "smoke.json").write_text(json.dumps(smoke))
    return manifest, rows


def test_matched_contrasts_and_heuristic_veto(tmp_path):
    fixture(tmp_path)
    ANALYSIS.analyse(tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert analysis["decision"]["complete"]
    assert analysis["decision"]["conditional_gain_both_datasets"] == dict(csmc_quadratic=True, kernel=False)
    assert analysis["decision"]["heuristic_dominance"]["csmc_quadratic"] == "fail"
    assert analysis["bootstrap"]["family_contrasts"] == 8
    for dataset in analysis["datasets"]:
        assert dataset["contrasts"]["quadratic_matched"]["interval_99_375"] == [-3., -3.]
        assert dataset["contrasts"]["kernel_matched"]["interval_99_375"] == [5., 5.]


def test_failed_seed_does_not_become_survivor_ranking(tmp_path):
    manifest, rows = fixture(tmp_path)
    rows[8]["valid"] = False
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    ANALYSIS.analyse(tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert not analysis["decision"]["complete"]
    assert not analysis["decision"]["conditional_gain_both_datasets"]["csmc_quadratic"]
    assert analysis["datasets"][0]["contrasts"]["quadratic_matched"]["interval_99_375"] is None
    assert "incomplete; no ranking" in (tmp_path / "result.md").read_text()


@pytest.mark.parametrize("field,value", [("reset_steps", 40), ("beta", [.1, .9, 0.])])
def test_changed_controls_are_rejected(tmp_path, field, value):
    _, rows = fixture(tmp_path)
    rows[0][field] = value
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        ANALYSIS.analyse(tmp_path)
