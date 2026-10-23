"""Synthetic reporting tests, not scientific comparison evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/analyze_ledh_sir_guidance.py"
SPEC = importlib.util.spec_from_file_location("sir_guidance_analysis", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def fixture(directory):
    errors = {"plain": 2., "csmc_quadratic": 1., "kernel": 3.,
              "prior": .5, "global": 1.5, "local": 2.5}
    arms = {arm: [1., 0., 0.] for arm in errors}
    manifest = dict(status="complete", completed_rows=96, arms=arms,
                    trace_fields=["valid", "moment_valid", "correction_cap_valid"],
                    seeds={}, fits={}, reference=dict(datasets={}))
    rows = []
    for data_seed in (26100831, 26100832):
        manifest["seeds"][str(data_seed)] = list(range(8))
        manifest["reference"]["datasets"][str(data_seed)] = dict(score=[0., 0., 0.])
        fit = dict(projected=[False], active=[False], projected_error=[0.], quadratic_error=[0.],
                   qr_condition=[1.], design_margin=[1.], precision=[[[0.]]], linear=[[0.]])
        fit_path = directory / f"guides-{data_seed}.json"
        fit_path.write_text(json.dumps(fit))
        manifest["fits"][str(data_seed)] = dict(sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest())
        for seed in range(8):
            for arm, error in errors.items():
                rows.append(dict(data_seed=data_seed, seed=seed, arm=arm, beta=arms[arm],
                                 reset_steps=5, reset_epsilon=1., valid=True, steps_completed=[50] * 3,
                                 score=[error, 0., 0.], value=0., seconds=1.,
                                 traces=[[[1., 1., 1.] for _ in range(50)] for _ in range(3)]))
    smoke = [dict(arm=arm, valid=True,
                  derivative_checks=[dict(coordinate=coordinate, step=step, passed=True, relative_error=0.)
                                     for coordinate in range(3) for step in (2.5e-5, 1.25e-5)],
                  richardson=[dict(coordinate=coordinate, passed=True, relative_error=0.)
                              for coordinate in range(3)])
             for arm in ("plain", "csmc_quadratic", "kernel")]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "smoke.json").write_text(json.dumps(smoke))
    return manifest, rows, smoke


def test_paired_score_metric_and_heuristic_veto(tmp_path):
    fixture(tmp_path)
    ANALYSIS.analyse(tmp_path)
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["decision"]["complete"]
    assert result["decision"]["conditional_gain_both_datasets"] == dict(csmc_quadratic=True, kernel=False)
    assert result["decision"]["heuristic_dominance"]["csmc_quadratic"] == "fail"
    assert not result["decision"]["default_ready"]
    for dataset in result["datasets"]:
        assert dataset["comparisons"]["csmc_quadratic"]["interval_98_75"] == [-3., -3.]
        assert dataset["comparisons"]["kernel"]["interval_98_75"] == [5., 5.]
    assert result["traces_checked"] == 14400


def test_incomplete_design_remains_descriptive(tmp_path):
    manifest, rows, _ = fixture(tmp_path)
    partial = [row for row in rows if row["seed"] < 3]
    partial.append(next(row for row in rows if row["seed"] == 3 and row["arm"] == "plain"))
    manifest.update(status="budget_exhausted", completed_rows=len(partial))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in partial))
    ANALYSIS.analyse(tmp_path)
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert not result["decision"]["complete"]
    assert not any(result["decision"]["conditional_gain_both_datasets"].values())
    for dataset in result["datasets"]:
        assert dataset["balanced_descriptive"]["seeds"] == [0, 1, 2]
        assert all(summary["count"] == 3 for summary in dataset["balanced_descriptive"]["arms"].values())
        assert all(comparison["interval_98_75"] is None for comparison in dataset["comparisons"].values())


def test_failed_comparator_is_not_silently_dropped(tmp_path):
    _, rows, _ = fixture(tmp_path)
    failed = next(row for row in rows if row["arm"] == "global")
    failed.update(valid=False, steps_completed=[4] * 3)
    for trace in failed["traces"]:
        trace[4:] = [["nan"] * 3 for _ in range(46)]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    ANALYSIS.analyse(tmp_path)
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["decision"]["evaluation_complete"]
    assert not result["decision"]["complete"]
    assert len(result["decision"]["failures"]) == 1
    assert result["datasets"][0]["arms"]["global"]["failed_count"] == 1
    assert result["datasets"][0]["arms"]["global"]["count"] == 7
    assert result["decision"]["conditional_gain_both_datasets"]["csmc_quadratic"]
    assert result["decision"]["heuristic_dominance"]["csmc_quadratic"] == "fail"
    assert "not comparable: failed seed" in (tmp_path / "result.md").read_text()


@pytest.mark.parametrize("mutation,message", [
    ("unexpected_arm", "Unexpected paired evaluation design"),
    ("wrong_ot", "fixed OT setting changed"),
    ("wrong_beta", "proposal mixture setting changed"),
    ("missing_derivative", "Incomplete derivative evidence"),
    ("changed_fit", "Frozen guide checksum changed"),
])
def test_rejects_corrupted_comparison_evidence(tmp_path, mutation, message):
    _, rows, smoke = fixture(tmp_path)
    if mutation == "unexpected_arm":
        rows[0]["arm"] = "unknown"
    elif mutation == "wrong_ot":
        rows[0]["reset_steps"] = 40
    elif mutation == "wrong_beta":
        rows[0]["beta"] = [0., 1., 0.]
    elif mutation == "missing_derivative":
        smoke[0]["derivative_checks"].pop()
    elif mutation == "changed_fit":
        with (tmp_path / "guides-26100831.json").open("a") as stream:
            stream.write(" ")
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "smoke.json").write_text(json.dumps(smoke))
    with pytest.raises(ValueError, match=message):
        ANALYSIS.analyse(tmp_path)
