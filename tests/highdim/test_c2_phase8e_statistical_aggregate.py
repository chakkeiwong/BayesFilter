"""Contract tests for the fixture-cluster Phase 8E analyzer."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "docs/benchmarks/analyze_c2_phase8e_statistical_replication_20260907.py"
SPEC = importlib.util.spec_from_file_location("c2_phase8e_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_run(
    path: Path,
    *,
    candidate_ess: list[float] | None,
    baseline_ess: list[float] | None,
    branch_count: int = 2,
    fixture_name: str | None = None,
    fixture_offset: float = 0.0,
) -> None:
    path.mkdir()
    records = []
    for branch_index in range(branch_count):
        candidate = {
            "label": MODULE.CANDIDATE_LABEL,
            "branch_index": branch_index,
            "seed": 98200 + 4096 + 1009 * branch_index,
            "all_checks_pass": candidate_ess is not None,
        }
        baseline = {
            "label": "bootstrap_conditional",
            "branch_index": branch_index,
            "seed": 98200 + 4096 + 1009 * branch_index,
            "all_checks_pass": baseline_ess is not None,
        }
        if candidate_ess is not None:
            candidate["ess_by_time"] = candidate_ess
        else:
            candidate["error"] = "fixed Laplace result has invalid rows"
        if baseline_ess is not None:
            baseline["ess_by_time"] = baseline_ess
        records.extend((candidate, baseline))
    fixture_path = path / (fixture_name or "fixture.json")
    horizon = (
        len(candidate_ess)
        if candidate_ess is not None
        else len(baseline_ess or ())
    )
    observation_seed = 424300 + int(fixture_offset)
    fixture_path.write_text(
        json.dumps(
            {
                "model_seed": 20260912,
                "observation_seed": observation_seed,
                "observations": [
                    [0.1 + float(index) + float(fixture_offset)]
                    for index in range(horizon)
                ],
            }
        ),
        encoding="utf-8",
    )
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    payload = {
        "proposal_active_start_time": 1,
        "phase": "c2_exact_likelihood_laplace_phase8e_statistical_replication_v1",
        "include_repair_schedules": True,
        "schedule_selection_mode": "fixed_requested_config_after_independent_validity_check",
        "schedule_ladder_id": "baseline_plus_reviewed_repair_hypotheses",
        "selected_schedule": {"config_id": "quarter_long_12"},
        "particle_count": 4096,
        "horizon": horizon,
        "branch_count": branch_count,
        "replication": {
            "set_id": "c2_phase8e_primary_n4096_v1",
            "analysis_seed": 20260907,
            "model_seed": 20260912,
            "observation_seed": observation_seed,
            "branch_seed_formula": {
                "base": 98200,
                "particle_count_multiplier": 1,
                "branch_index_stride": 1009,
                "expression": "98200 + particle_count + 1009 * branch_index",
            },
        },
        "records": records,
        "sources": {
            "fixture": {"path": str(fixture_path), "sha256": fixture_sha256}
        },
        "status": "PASS_PHASE8D_VALIDITY_DIAGNOSTIC",
        "continuation_veto": False,
        "environment": {"jit_compile": True},
        "workspace": {"git_commit": "0123456789abcdef", "git_status": ""},
        "checks": {
            "records_complete": True,
            "source_files_present": True,
            "snapshot_bank_complete": True,
            "gpu_memory_growth_verified": True,
            "all_branch_validity": candidate_ess is not None and baseline_ess is not None,
        },
    }
    (path / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_fixture_cluster_bootstrap_and_sign_test(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        run = tmp_path / f"run{index}"
        _write_run(
            run,
            candidate_ess=[1.0, 4.0, 4.0],
            baseline_ess=[1.0, 2.0, 2.0],
            fixture_offset=float(index),
        )
        paths.append(run)
    result = MODULE.analyze(
        paths,
        expected_branch_count=2,
        heuristics=("bootstrap_conditional",),
        bootstrap_resamples=200,
        bootstrap_seed=7,
        expected_analysis_seed=20260907,
        expected_observation_seeds=(424300, 424301, 424302),
        boundaries=MODULE.BinBoundaries(near_zero_upper=0.5, ordinary_upper=3.0),
    )
    entry = result["statistical"]["bootstrap_conditional"]
    assert result["status"] == "NOMINATED_STATISTICAL_CONTRAST_DIAGNOSTIC"
    assert result["fixture_count"] == 3
    assert entry["bootstrap"]["resampling_unit"] == "independent_observation_fixture"
    assert entry["sign_test"]["positive_count"] == 3
    assert entry["sign_test"]["two_sided_exact_p_value"] == 0.25
    assert entry["statistically_nominated"] is True


def test_invalid_candidate_blocks_inference_without_inventing_ess(tmp_path: Path) -> None:
    run = tmp_path / "invalid"
    _write_run(run, candidate_ess=None, baseline_ess=[1.0, 2.0, 2.0])
    result = MODULE.analyze(
        [run],
        expected_branch_count=2,
        heuristics=("bootstrap_conditional",),
        bootstrap_resamples=100,
        expected_observation_seeds=(424300,),
        boundaries=MODULE.BinBoundaries(near_zero_upper=0.5, ordinary_upper=3.0),
    )
    entry = result["statistical"]["bootstrap_conditional"]
    assert result["status"] == "VETO_PHASE8E_VALIDITY"
    assert result["valid_fixture_count"] == 0
    assert entry["fixture_values"] == []
    assert entry["bootstrap"] is None
    assert entry["statistically_nominated"] is False


def test_observation_bin_loss_remains_a_promotion_veto(tmp_path: Path) -> None:
    run = tmp_path / "bin-loss"
    _write_run(run, candidate_ess=[1.0, 4.0, 4.0], baseline_ess=[1.0, 5.0, 2.0], branch_count=1)
    boundaries = MODULE.BinBoundaries(near_zero_upper=0.5, ordinary_upper=2.5)
    result = MODULE.analyze(
        [run],
        expected_branch_count=1,
        heuristics=("bootstrap_conditional",),
        boundaries=boundaries,
        expected_observation_seeds=(424300,),
        bootstrap_resamples=100,
    )
    row = result["fixture_rows"][0]
    assert result["status"] == "PASS_VALIDITY_WITH_PROMOTION_VETO"
    assert row["observation_bins_checked"] is True
    assert row["observation_bin_losses"]
    assert row["observation_bins"][1] == "ordinary"


def test_frozen_boundary_artifact_recomputes_from_calibration_partition(tmp_path: Path) -> None:
    boundary_path = (
        ROOT / "docs/benchmarks/fixtures/c2_phase8e_observation_bin_boundaries_v1.json"
    )
    boundaries = MODULE._read_boundaries(boundary_path, expected_model_seed=20260912)
    assert boundaries.near_zero_upper == pytest.approx(0.3093389771526106)
    assert boundaries.ordinary_upper == pytest.approx(0.4621355191302)
    assert boundaries.source_sha256 == hashlib.sha256(boundary_path.read_bytes()).hexdigest()

    tampered_path = tmp_path / "tampered-boundaries.json"
    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    payload["near_zero_upper"] = float(payload["near_zero_upper"]) + 0.01
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.AnalysisError, match="frozen calibration partition"):
        MODULE._read_boundaries(tampered_path, expected_model_seed=20260912)


def test_wrong_branch_seed_formula_blocks_artifact_inference(tmp_path: Path) -> None:
    run = tmp_path / "wrong-branch-seed"
    _write_run(run, candidate_ess=[1.0, 4.0, 4.0], baseline_ess=[1.0, 2.0, 2.0])
    result_path = run / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        record["seed"] = int(record["seed"]) + 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    result = MODULE.analyze(
        [run],
        expected_branch_count=2,
        heuristics=("bootstrap_conditional",),
        expected_observation_seeds=(424300,),
        boundaries=MODULE.BinBoundaries(near_zero_upper=0.5, ordinary_upper=3.0),
        bootstrap_resamples=100,
    )
    assert result["status"] == "BLOCKED_ARTIFACT_PARSE"
    assert "deterministic seed" in result["parse_errors"][0]["error"]


def test_missing_fixture_is_reported_as_parse_error(tmp_path: Path) -> None:
    run = tmp_path / "missing-fixture"
    _write_run(run, candidate_ess=[1.0, 4.0, 4.0], baseline_ess=[1.0, 2.0, 2.0])
    result_path = run / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    fixture_path = Path(payload["sources"]["fixture"]["path"])
    fixture_path.unlink()
    result = MODULE.analyze(
        [run],
        expected_branch_count=2,
        heuristics=("bootstrap_conditional",),
        expected_observation_seeds=(424300,),
        boundaries=MODULE.BinBoundaries(near_zero_upper=0.5, ordinary_upper=3.0),
        bootstrap_resamples=100,
    )
    assert result["status"] == "BLOCKED_ARTIFACT_PARSE"
    assert "cannot read" in result["parse_errors"][0]["error"]
