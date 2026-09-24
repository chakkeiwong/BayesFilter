"""Diagnostic evidence checks for M30; no posterior or timing claims."""
from copy import deepcopy
import json

import pytest

from docs.benchmarks.replay_hmc_m30_2026_09_23 import compare_payloads


def fingerprints():
    tensor = {"dtype": "float64", "shape": [8, 4, 2], "sha256": "original"}
    return {"samples": dict(tensor), "trace": {
        "initial_momentum": dict(tensor), "final_momentum": dict(tensor),
        "proposed_state": dict(tensor),
        "is_accepted": {"dtype": "bool", "shape": [8, 4], "sha256": "accepted"},
        "target_status_telemetry": {"status_code": dict(tensor)}}}


@pytest.mark.parametrize("field", ["initial_momentum", "final_momentum", "proposed_state", "is_accepted"])
def test_complete_trace_comparison_rejects_each_corrupted_field(field):
    expected = fingerprints()
    actual = deepcopy(expected)
    actual["trace"][field]["sha256"] = "changed"
    with pytest.raises(AssertionError, match=field):
        compare_payloads(actual, expected)


def test_missing_nested_health_trace_is_not_a_parity_pass():
    expected = fingerprints()
    actual = deepcopy(expected)
    actual["trace"]["target_status_telemetry"].clear()
    with pytest.raises(AssertionError, match="tensor keys differ"):
        compare_payloads(actual, expected)


def test_exact_complete_evidence_and_shapes_are_checked():
    expected = fingerprints()
    compare_payloads(deepcopy(expected), expected)
    actual = deepcopy(expected)
    actual["samples"]["shape"] = [4, 8, 2]
    with pytest.raises(AssertionError, match="samples/shape"):
        compare_payloads(actual, expected)


def test_full_fit_comparison_keeps_numerical_l_and_seeds():
    from docs.benchmarks.audit_hmc_m30_2026_09_23 import normalize
    runtime = {"runtime": "tfp.mcmc.sample_chain_independent_chain_ensemble",
        "ensemble_call_count": 4, "dynamic_num_leapfrog_steps": True,
        "num_leapfrog_steps": 7, "num_leapfrog_steps_source": "runtime_tensor_argument",
        "root_seed": [2, 3]}
    with pytest.raises(AssertionError):
        normalize(runtime, expected_l=3)
    checked = normalize(runtime, expected_l=7)
    assert checked["root_seed"] == [2, 3]
    assert normalize({"num_leapfrog_steps": 7}, expected_l=3) == {"num_leapfrog_steps": 7}


def test_completed_fit_cannot_be_relabelled_with_another_reuse_policy(tmp_path):
    from bayesfilter.testing.inference_validation.engines.pipeline import run_replication
    path = tmp_path / "replication-0000"
    (path / "tuning").mkdir(parents=True)
    (path / "independent_assessment.json").write_text('{"saved": true}')
    (path / "tuning/execution_spec.json").write_text(json.dumps({"execution": {
        "config": {"reuse_leapfrog_graphs": True}}}))
    assert run_replication(None, tmp_path, 0, reuse_leapfrog_graphs=True) == {"saved": True}
    with pytest.raises(ValueError, match="original runner reuse policy"):
        run_replication(None, tmp_path, 0)
