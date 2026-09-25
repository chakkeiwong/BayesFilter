"""Backend-free orchestration tests, never evidence of filtering accuracy."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bayesfilter.score_study.contracts import seed_pair, validate_study
from bayesfilter.score_study.coordinator import execute, report
from bayesfilter.score_study.registry import default_registry


@pytest.fixture
def registry():
    registry = default_registry()
    registry.proposals["bootstrap"] = replace(registry.proposals["bootstrap"], status="active", prerequisite="")
    registry.estimators["analytical_filter"] = replace(
        registry.estimators["analytical_filter"],
        endpoint="tests.highdim.test_younis_score_master_coordinator:synthetic_result")
    return registry


@pytest.fixture
def study():
    return {"schema": "younis_score_study_v1", "phase": "0B", "version": 1,
            "plan": "docs/plans/younis-kdm-score-master-program-2026-09-14.md", "seed": 123,
            "budget": {"wall_seconds": 120, "max_attempts": 8, "max_attempts_per_row": 3},
            "partitions": {"calibration": [10], "validation": [20], "claim": [30]},
            "required_proposals": ["bootstrap"],
            "rows": [{"id": "a", "model": "gaussian_all_parameters", "proposal": "bootstrap",
                      "estimator": "analytical_filter", "comparison_target": "model_score",
                      "comparison": "approximation_error", "dataset": 10, "replicate": 0,
                      "role": "mechanics"}]}


def synthetic_result(row, context):
    return {"value": 1.0, "score": [2.0], "oracle_value": 1.5, "oracle_score": [2.5],
            "value_target": "finite_particle_log_likelihood", "derivative_target": "finite_program_score",
            "comparison_target": "model_score", "derivative_id": "analytical_recursion",
            "proposal_law": "transition_prior", "initial_terms": True,
            "numerical_validity": "pass", "inference_status": "mechanics_only",
            "runtime": {"device": "none", "backend": "synthetic_test_only"},
            "diagnostics": {"not_a_filter": True}}


def loader(_):
    return synthetic_result


def test_validation_imports_no_tensorflow():
    environment = {**os.environ, "BAYESFILTER_PRELOAD_CUSTOM_OP": "0", "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run([sys.executable, "-c",
        "import sys; from bayesfilter.score_study.registry import default_registry; "
        "from bayesfilter.score_study.coordinator import execute; default_registry(); "
        "assert 'tensorflow' not in sys.modules; assert 'numpy' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["required_proposals"].append("kdm"), "missing requested"),
    (lambda s: s["rows"][0].update(comparison="same_target"), "different mathematical targets"),
    (lambda s: s["partitions"]["claim"].append(10), "overlap"),
    (lambda s: s["rows"][0].update(depends_on=["a"]), "cyclic"),
    (lambda s: s["budget"].update(wall_seconds=0), "positive budget"),
])
def test_invalid_study_fails_before_execution(study, registry, mutation, match):
    mutation(study)
    with pytest.raises(ValueError, match=match):
        validate_study(study, registry)


def test_support_and_initial_law_rejection(study, registry):
    registry.models["gaussian_all_parameters"] = replace(registry.models["gaussian_all_parameters"], support="degenerate")
    with pytest.raises(ValueError, match="support"):
        validate_study(study, registry)
    registry.models["gaussian_all_parameters"] = replace(registry.models["gaussian_all_parameters"], support="regular_euclidean")
    registry.estimators["analytical_filter"] = replace(registry.estimators["analytical_filter"], includes_initial_terms=False)
    with pytest.raises(ValueError, match="initialization"):
        validate_study(study, registry)


def test_random_streams_independent_of_scheduling():
    keys = dict(master_seed=7, model="m", dataset=5, replicate=9, stream="particles", coupling_group="g")
    first = seed_pair(**keys)
    seed_pair(**{**keys, "replicate": 17})
    assert first == seed_pair(**keys)
    assert first != seed_pair(**{**keys, "stream": "observations"})
    assert first != seed_pair(**{**keys, "perturbation": "independent_node_1"})


def test_execute_reuse_and_corrupt_result_repair(study, registry, tmp_path):
    output = tmp_path / "run"
    state = execute(study, registry, output, endpoint_loader=loader)
    assert state["execution_status"] == "complete"
    assert len(execute(study, registry, output, resume=True, endpoint_loader=loader)["attempts"]) == 1
    result_path = output / state["rows"]["a"]["result_path"]
    result_path.write_text('{}')
    assert report(output, registry)["execution_status"] == "incomplete"
    state = execute(study, registry, output, resume=True, endpoint_loader=loader)
    assert state["execution_status"] == "complete"
    assert len(state["attempts"]) == 2
    assert result_path.read_text() == '{}'
    assert any(e["event"] == "invalidated_evidence" for e in state["events"])


def test_failure_repair_refresh_and_independent_branch(study, registry, tmp_path):
    study["rows"] += [{**study["rows"][0], "id": "b"},
                      {**study["rows"][0], "id": "c", "depends_on": ["a"]},
                      {**study["rows"][0], "id": "kdm", "proposal": "kdm", "estimator": "iwsg"}]
    def fail_a(row, context):
        if row["id"] == "a":
            raise RuntimeError("localized harness failure")
        return synthetic_result(row, context)
    output = tmp_path / "run"
    state = execute(study, registry, output, endpoint_loader=lambda _: fail_a)
    assert [state["rows"][r]["execution_status"] for r in ("a", "b", "c", "kdm")] == ["failed", "complete", "blocked", "blocked"]
    state = execute(study, registry, output, resume=True, endpoint_loader=loader)
    assert state["rows"]["c"]["execution_status"] == "complete"
    assert state["attempts"][0]["status"] == "failed"
    assert (output / state["attempts"][0]["path"] / "error.log").exists()
    study["version"] = 2
    state = execute(study, registry, output, resume=True, endpoint_loader=loader)
    assert any(e["event"] == "repair_refresh" for e in state["events"])


def test_interruption_preserved_and_resumed(study, registry, tmp_path):
    def interrupt(row, context):
        raise KeyboardInterrupt()
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        execute(study, registry, output, endpoint_loader=lambda _: interrupt)
    state = json.loads((output / "state.json").read_text())
    assert state["attempts"][0]["status"] == "interrupted"
    state = execute(study, registry, output, resume=True, endpoint_loader=loader)
    assert state["execution_status"] == "complete"
    assert len(state["attempts"]) == 2


def test_computed_failure_diagnostics_survive_coordinator(study, registry, tmp_path):
    from bayesfilter.score_study.contracts import DiagnosticFailure
    def failed_fit(row, context):
        raise DiagnosticFailure("fit failed", {"history": [1., float("nan")],
                                                "valid": False, "iterations": 7})
    output = tmp_path / "run"
    state = execute(study, registry, output, endpoint_loader=lambda _: failed_fit)
    attempt = state["attempts"][0]
    saved = json.loads((output / attempt["failure_diagnostics_path"]).read_text())
    assert attempt["status"] == "failed"
    assert saved["details"] == {"history": [1., "nan"], "valid": False, "iterations": 7}
    assert state["rows"]["a"]["numerical_validity"] == "not_admitted"


def test_incomplete_output_and_attempt_budget(study, registry, tmp_path):
    output = tmp_path / "run"
    study["budget"]["max_attempts_per_row"] = 1
    state = execute(study, registry, output, endpoint_loader=lambda _: lambda row, context: {})
    assert state["rows"]["a"]["execution_status"] == "failed"
    state = execute(study, registry, output, resume=True, endpoint_loader=loader)
    assert state["rows"]["a"]["execution_status"] == "budget_exhausted"
    assert len(state["attempts"]) == 1


def test_source_change_invalidates_consumers(study, registry, tmp_path, monkeypatch):
    from bayesfilter.score_study import coordinator
    source = {"test.py": "first"}
    monkeypatch.setattr(coordinator, "source_closure", lambda _: dict(source))
    output = tmp_path / "run"
    execute(study, registry, output, endpoint_loader=loader)
    source["test.py"] = "repaired"
    state = execute(study, registry, output, resume=True, endpoint_loader=loader)
    assert len(state["attempts"]) == 2
    assert state["events"][-1]["event"] == "repair_refresh"
