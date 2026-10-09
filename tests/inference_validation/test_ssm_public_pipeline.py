"""Actual state-space adapters through both public validation routes.

These are bounded CPU integration tests.  They check target wiring, candidate
retention, reload, and warmup exclusion; they do not make posterior accuracy
or nonlinear approximation claims.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
from bayesfilter.testing.inference_validation.engines.pipeline import run_replication
from bayesfilter.testing.inference_validation.procedures import execute_pipeline
from bayesfilter.testing.inference_validation.ssm_targets import get_ssm_profile
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor


def _design(target: str, route: str) -> ValidationDesign:
    if target.startswith("ssm_campaign_"):
        from bayesfilter.testing.inference_validation.ssm_campaign_profiles import generate_data
        observations = generate_data(target, (20260926, 71))
    else:
        observations = list(get_ssm_profile(target).default_observations)
    return ValidationDesign(
        design_id=f"ssm-public-{target}-{route}",
        engine="accuracy",
        scenario=ScenarioSpec(target, route),
        replications=1,
        draws=8,
        seed=2026092601,
        budget_seconds=180,
        purpose="actual BayesFilter state-space public call-chain regression",
        numerical_provenance="bounded CPU integration fixture; no posterior promotion",
        device="cpu_reference",
        l_grid=(2, 3),
        step_size=0.25,
        leapfrog_steps=2,
        measurement_draws=64,
        posterior_cap=8,
        mcse_tolerance=1.0,
        accuracy_tolerance=1.0,
        options={
            "data": observations,
            "acceptance_policy": {
                "target": 0.70,
                "practical_region": (0.50, 0.90),
                "repair_region": (0.41, 0.99),
            },
            "search": {
                "pilot_enabled": False,
                "refinement_rounds": 0,
                # K0's concentrated location posterior needs more same-L
                # repairs from this fixed epsilon. Fund both length families
                # and their fresh verification; do not weaken acceptance.
                "total_budget_units": 32 if target == "ssm_campaign_location" else 8,
                "repair_reserve_units": 8 if target == "ssm_campaign_location" else 2,
                "evidence_rungs": (1,),
            },
            "posterior_members": "selected",
            "member_rule": "first_verified",
            "posterior_settings": {
                "warmup_chunk_results": 4,
                "warmup_min_results": 4,
                "warmup_check_window_results": 4,
                "warmup_max_results": 8,
                "retained_chunk_results": 4,
                "retained_min_results": 4,
                "retained_max_results": 8,
            },
            "posterior_count_budget": {
                "max_results_per_chain": 8,
                "count_budget_reason": "bounded public state-space regression",
            },
        },
    )


@pytest.mark.parametrize("target", ["ssm_lgssm_qr", "ssm_nonlinear",
                                   "ssm_campaign_location", "ssm_campaign_nonlinear"])
@pytest.mark.parametrize("route", ["ordinary", "prepared"])
def test_actual_ssm_adapter_uses_public_tuning_and_retained_pipeline(
    tmp_path, target: str, route: str
) -> None:
    design = _design(target, route)
    # Use the executor's canonical replication path so the independent
    # assessor consumes this completed fit instead of launching a second one.
    root = tmp_path / "replication-0000"
    try:
        result = execute_pipeline(design, root, data=design.options["data"])
    except Exception as exc:
        # Include the nested preparation cause in a preserved pytest receipt.
        pytest.fail(f"{exc!r}; details={getattr(exc, 'details', None)}")

    assert result["verified_candidate_ids"], result
    assert result["numerical_route"] == route
    tuning = read_json(result["tuning_path"])
    assert tuning["scope"]["target_signature"] if isinstance(tuning.get("scope"), dict) else True
    assert set(result["verified_candidate_ids"]) <= {
        member["candidate_id"] for member in result["members"]
    }
    assessed = [member for member in result["members"] if member["status"] == "assessed"]
    assert assessed
    for member in assessed:
        assert member["warmup_exclusion_matches"]
        assert read_tensor(member["draws_path"]).shape[0] == member["recorded_retained_count"]

    # A second invocation must reload the same public target and preserve the
    # candidate/member inventory instead of silently retuning a different law.
    resumed = execute_pipeline(design, root, data=design.options["data"])
    assert resumed["verified_candidate_ids"] == result["verified_candidate_ids"]

    record = run_replication(design, tmp_path, 0)
    rows = [row for row in record["members"] if row.get("status") == "assessed"]
    assert rows
    if target.startswith("ssm_nonlinear"):
        assert all(row["assessment"]["finding"] == "reference_unavailable" for row in rows)
    else:
        assert all(row["assessment"]["accuracy_established"] is False for row in rows)
    if target.startswith("ssm_campaign_"):
        # A completed-fit fast path must still refuse changed observations.
        from dataclasses import replace
        changed = [design.options["data"][0]+.1, *design.options["data"][1:]]
        altered = replace(design, options={**design.options, "data":changed})
        with pytest.raises(ValueError, match="identity|source|design"):
            execute_pipeline(altered, root, data=changed)
        assert result["reference_contract"]["data"] == design.options["data"]
        spec = read_json(root / "tuning" / "execution_spec.json")
        assert spec["execution"]["config"]["target_status_trace_policy"] == "per_chain_step"


def test_campaign_isolated_child_keeps_data_status_and_graph_reuse(tmp_path):
    import json
    import subprocess
    import sys
    from dataclasses import replace
    from bayesfilter.testing.inference_validation.ssm_campaign import suite
    from bayesfilter.testing.inference_validation.storage import write_json
    design = _design("ssm_campaign_location","ordinary")
    design = replace(design,budget_seconds=180.,options={**design.options,
        "isolate_fits":True,"fit_process_timeout_seconds":170.})
    source = write_json(tmp_path/"suite.json",suite("isolated-ssm-regression",[design]))
    output = tmp_path/"run"
    run = subprocess.run([sys.executable,"-m","bayesfilter.testing.inference_validation",
        "run",str(source),"--output",str(output),"--reuse-leapfrog-graphs"],
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1"},capture_output=True,text=True,timeout=210)
    assert run.returncode == 0,run.stdout+run.stderr
    job = read_json(output/"run_index.json")["jobs"][design.design_id]
    assert job["status"] == "complete",job
    root = output/design.design_id/"replication-0000"
    child = read_json(root/"process-attempt-001-manifest.json")
    assert child["reuse_leapfrog_graphs"] is True
    assert child["runtime"]["gpu_intentionally_hidden"] is True
    assert child["data"] == design.options["data"]
    payload = read_json(root/"pipeline.json")
    assert payload["verified_candidate_ids"]
    assert payload["reference_contract"]["data"] == design.options["data"]
    assert read_json(root/"tuning/execution_spec.json")["execution"]["config"]["target_status_trace_policy"] == "per_chain_step"


@pytest.mark.parametrize(
    "target",
    ["ssm_lgssm_qr", "ssm_lgssm_near_unit", "ssm_lgssm_small_noise", "ssm_lgssm_long_horizon"],
)
def test_named_qr_stress_profiles_have_finite_filter_value_and_independent_oracle(target):
    import numpy as np
    import tensorflow as tf

    from bayesfilter.testing.inference_validation.references import analytic
    from bayesfilter.testing.inference_validation.targets import ValidationTarget

    target_adapter = ValidationTarget(target, jit_compile=False)
    q = tf.constant([[0.20, -1.05], [-0.30, -0.70]], tf.float64)
    value, score = target_adapter.log_prob_and_grad(q)
    reference = analytic.log_density(target, q.numpy(), target_adapter.parameters, target_adapter.data)
    np.testing.assert_allclose(value.numpy(), reference, rtol=0.0, atol=1.0e-6)
    assert np.isfinite(score.numpy()).all()
    assert target_adapter.target_status_telemetry(q)["valid_pre_regularized_score"].numpy().all()


@pytest.mark.parametrize(
    "target",
    ["ssm_lgssm_qr", "ssm_lgssm_near_unit", "ssm_lgssm_small_noise", "ssm_lgssm_long_horizon"],
)
def test_qr_public_fixture_matches_independent_mechanics_oracle(tmp_path, target):
    from bayesfilter.testing.inference_validation.engines.mechanics import run

    profile = get_ssm_profile(target)
    design = ValidationDesign(
        design_id=f"ssm-mechanics-{target}", engine="mechanics",
        scenario=ScenarioSpec(target, "frozen"), replications=4, draws=64,
        seed=2026092602, budget_seconds=120,
        purpose="QR state-space value and score call-chain regression",
        numerical_provenance="independent scalar Kalman recursion; no posterior claim",
        device="cpu_reference", options={"data": list(profile.default_observations)},
    )
    result = run(design, tmp_path / target)
    assert result["finding"] == "mechanics_passed", result
    assert result["density_tolerance"] == 1.0e-6


def test_nonlinear_profile_declares_approximation_reference_boundary():
    from bayesfilter.testing.inference_validation.catalog import get_target
    from bayesfilter.testing.inference_validation.targets import ValidationTarget

    spec = get_target("ssm_nonlinear")
    target = ValidationTarget("ssm_nonlinear", jit_compile=False)
    assert spec.reference.kind == "reference_unavailable"
    assert target.value_score_capability().target_scope == "inference_validation"
    assert "deterministic sigma-point approximation only" in target.value_score_capability().nonclaims
