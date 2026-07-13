from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc_coordinates import KernelState, WarmupTrajectoryPolicy
from bayesfilter.inference.hmc_tuning_artifacts import (
    KillableChildSpec,
    atomic_write_json,
    build_hmc_tuning_engineering_artifact,
    kernel_state_summary,
    load_and_replay_hmc_tuning_artifact,
    private_start_bank_summary,
    run_killable_child,
    transition_ledger_payload,
    validate_hmc_tuning_engineering_artifact,
    validate_killable_child_closeout,
)
from bayesfilter.inference.hmc_tuning_state import HMCTuningTransition
from bayesfilter.inference.hmc_kernel_selection import (
    private_start_bank_content_signature,
)
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)
from tests.test_hmc_kernel_tuning_windowed_mass import (
    _operational_budget,
    _operational_inputs,
)
import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning


def _evidence(probability: float):
    draws = np.arange(64, dtype=float)[:, None, None]
    chains = np.arange(4, dtype=float)[None, :, None]
    values = np.full((64, 4), probability)
    return evaluate_hmc_acceptance_evidence(
        samples=draws + chains,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
    ).payload()


def _operational_payloads(*, selected_l: int = 3):
    adapter, geometry, bootstrap = _operational_inputs()
    windowed = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 640),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        _attempt_budget_policy=_operational_budget(),
    )
    operational = windowed.operational_warmup_result
    assert operational is not None
    warmup_state = operational.final_kernel_state
    final_policy = WarmupTrajectoryPolicy(
        selected_l,
        warmup_state.trajectory_policy.max_leapfrog_steps,
    )
    final_state = KernelState(
        canonical_theta=warmup_state.canonical_theta,
        active_latent=warmup_state.active_latent,
        transform=warmup_state.transform,
        momentum_metric=warmup_state.momentum_metric,
        epsilon=None,
        trajectory_policy=final_policy,
        adaptation_generation=warmup_state.adaptation_generation,
        seed_lineage=warmup_state.seed_lineage,
        evidence_status="exact_l_retuned",
    ).with_epsilon(
        float(warmup_state.epsilon),
        evidence_status="exact_l_retuned",
    )
    active_bank = np.asarray(
        warmup_state.transform.theta_to_latent(
            operational.private_start_bank_theta
        ).numpy(),
        dtype=float,
    )
    active_bank_signature = private_start_bank_content_signature(
        active_bank,
        final_state.transform.signature,
    )
    return (
        operational.public_payload(),
        kernel_state_summary(final_state),
        private_start_bank_summary(
            operational,
            active_signature=active_bank_signature,
        ),
        {
            "schema": "bayesfilter.hmc_trajectory_handoff.v2",
            "warmup_trajectory_signature": warmup_state.trajectory_policy.signature,
            "final_trajectory_signature": final_policy.signature,
            "warmup_num_leapfrog_steps": warmup_state.trajectory_policy.num_leapfrog_steps,
            "selected_num_leapfrog_steps": selected_l,
            "max_leapfrog_steps": final_policy.max_leapfrog_steps,
            "selection_signature": "selection-signature",
            "candidate_signature": "candidate-signature",
            "exact_l_retune_signature": "retune-signature",
            "exact_l_retune_seed": (3, 103),
            "coordinate_signature": final_state.transform.signature,
            "metric_signature": final_state.momentum_metric.signature,
            "start_bank_signature": active_bank_signature,
            "exact_l_retuned": True,
        },
    )


def _repair_ledger(coordinate: str, metric: str, trajectory: str):
    signature_fields = {
        "coordinate_signature": coordinate,
        "metric_signature": metric,
        "trajectory_signature": trajectory,
    }
    return transition_ledger_payload(
        (
            HMCTuningTransition(
                source="initialized",
                target="warming",
                reason="operational warmup started",
            ),
            HMCTuningTransition(
                source="warming",
                target="metric_frozen",
                reason="operational metric frozen",
            ),
            HMCTuningTransition(
                source="metric_frozen",
                target="step_tuned",
                reason="exact-L step tune complete",
                **signature_fields,
            ),
            HMCTuningTransition(
                source="step_tuned",
                target="verifying",
                reason="independent verification started",
                **signature_fields,
            ),
            HMCTuningTransition(
                source="verifying",
                target="repair_required",
                reason="supported high-acceptance repair",
                **signature_fields,
            ),
            HMCTuningTransition(
                source="repair_required",
                target="step_repaired",
                reason="bounded epsilon repair applied",
                **signature_fields,
            ),
            HMCTuningTransition(
                source="step_repaired",
                target="verifying",
                reason="reserved repair verification started",
                **signature_fields,
            ),
            HMCTuningTransition(
                source="verifying",
                target="passed",
                reason="bounded engineering verification passed",
                **signature_fields,
            ),
        )
    )


def _seed_domains():
    return {
        "warmup": (1, 101),
        "candidate_selection": (2, 102),
        "exact_final_l_epsilon_tune": (3, 103),
        "independent_final_verification": (4, 104),
        "repair_verification": (5, 105),
        "evidence_extension": (6, 106),
    }


def _artifact(*, purpose: str = "repair_loop_validation", slots: int = 2, validated: bool = True):
    warmup, kernel, bank, handoff = _operational_payloads()
    return build_hmc_tuning_engineering_artifact(
        evidence_purpose=purpose,
        configured_attempt_slots=slots,
        warmup_payload=warmup,
        kernel_state_payload=kernel,
        start_bank_payload=bank,
        trajectory_handoff=handoff,
        acceptance_evidence_payloads=(_evidence(0.90), _evidence(0.70)),
        transition_ledger=_repair_ledger(
            kernel["coordinate_signature"],
            kernel["metric_signature"],
            kernel["trajectory_signature"],
        ),
        seed_domains=_seed_domains(),
        terminal_state="passed",
        repair_loop_validated=validated,
        old_v1_compatibility={
            "schema": "bayesfilter.hmc_public_frozen_kernel_handoff.v1",
            "readable": True,
            "operational_authority": False,
        },
    )


def test_v2_artifact_replay_is_deterministic_and_preserves_v1_attachment(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / "tuning.json"
    atomic_write_json(path, artifact)

    first = load_and_replay_hmc_tuning_artifact(path)
    second = load_and_replay_hmc_tuning_artifact(path)

    assert first == second
    assert first["evidence_decisions"] == ("repair_step_higher", "passed")
    assert first["transition_count"] == 8
    assert first["repair_loop_validated"] is True
    assert first["old_v1_compatibility_present"] is True


def test_legacy_v2_artifact_replays_only_as_non_authoritative_migration_view(
    tmp_path: Path,
) -> None:
    from bayesfilter.inference.hmc_tuning_artifacts import canonical_sha256

    legacy_evidence = {
        "schema": "bayesfilter.hmc_acceptance_evidence.v2",
        "decision": "candidate_local_veto",
        "passed": False,
        "repair_direction": None,
        "pooled_mean": None,
        "interval": None,
        "standard_error": None,
        "chain_means": [],
        "block_means_by_chain": [],
        "movement_rate_by_chain": [],
        "repeated_state_fraction_by_chain": [],
        "normalized_return_displacement_by_chain": [],
        "usable_decisions_per_chain": 0,
        "excluded_remainder_per_chain": 0,
        "native_divergence_status": "not_exposed_by_kernel",
        "native_divergence_count": None,
        "max_abs_log_accept_energy_proxy": 1001.0,
        "policy": {
            **HMCAcceptancePolicy().payload(),
            "schema": "bayesfilter.hmc_acceptance_policy.v2",
        },
        "hard_health_failures": ["log_accept_energy_proxy_exceeded"],
        "explanatory_notes": [],
        "raw_traces_exposed": False,
        "reports_posterior_convergence": False,
    }
    core = {
        "schema": "bayesfilter.hmc_tuning_engineering_artifact.v2",
        "acceptance_evidence": [legacy_evidence],
        "raw_start_bank_exposed": False,
        "raw_states_exposed": False,
        "raw_samples_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
    }
    path = tmp_path / "legacy-v2.json"
    atomic_write_json(path, {**core, "artifact_sha256": canonical_sha256(core)})

    view = load_and_replay_hmc_tuning_artifact(path)

    assert view["operational_authority"] is False
    assert view["repair_loop_validated_under_v3"] is False
    assert view["evidence_migration_views"][0][
        "repair_direction_under_v3"
    ] == "unavailable"


def test_v2_artifact_accepts_exact_retuned_selected_l_different_from_warmup() -> None:
    warmup, kernel, bank, handoff = _operational_payloads(selected_l=2)
    artifact = build_hmc_tuning_engineering_artifact(
        evidence_purpose="repair_loop_validation",
        configured_attempt_slots=2,
        warmup_payload=warmup,
        kernel_state_payload=kernel,
        start_bank_payload=bank,
        trajectory_handoff=handoff,
        acceptance_evidence_payloads=(_evidence(0.90), _evidence(0.70)),
        transition_ledger=_repair_ledger(
            kernel["coordinate_signature"],
            kernel["metric_signature"],
            kernel["trajectory_signature"],
        ),
        seed_domains=_seed_domains(),
        terminal_state="passed",
        repair_loop_validated=True,
    )

    assert validate_hmc_tuning_engineering_artifact(artifact)[
        "repair_loop_validated"
    ] is True


def test_v2_artifact_rejects_unretuned_or_mismatched_trajectory_handoff() -> None:
    payload = json.loads(json.dumps(_artifact()))
    payload["trajectory_handoff"]["exact_l_retuned"] = False
    from bayesfilter.inference.hmc_tuning_artifacts import canonical_sha256

    core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    payload["artifact_sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="trajectory handoff lineage"):
        validate_hmc_tuning_engineering_artifact(payload)


def test_v2_artifact_rejects_repair_transitions_without_directional_evidence() -> None:
    payload = json.loads(json.dumps(_artifact()))
    payload["acceptance_evidence"] = [_evidence(0.70), _evidence(0.70)]
    from bayesfilter.inference.hmc_tuning_artifacts import canonical_sha256

    core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    payload["artifact_sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="directional evidence"):
        validate_hmc_tuning_engineering_artifact(payload)


def test_one_attempt_or_seam_artifact_cannot_claim_repair_loop_validation() -> None:
    with pytest.raises(ValueError, match="seam_execution_only"):
        _artifact(purpose="seam_execution_only", slots=1, validated=True)
    with pytest.raises(ValueError, match="at least two"):
        _artifact(purpose="repair_loop_validation", slots=1, validated=True)

    with pytest.raises(ValueError, match="integer scalar"):
        _artifact(slots=2.5)
    with pytest.raises(ValueError, match="must be boolean"):
        _artifact(validated="false")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("configured_attempt_slots",), 2.5, "integer scalar"),
        (("repair_loop_validated",), 1, "must be boolean"),
        (("kernel_state", "adaptation_generation"), 1.5, "integer scalar"),
        (("kernel_state", "seed_lineage"), (1.5, 2), "integer scalar"),
        (("private_start_bank", "count"), 4.5, "integer scalar"),
        (("private_start_bank", "seed_root"), (1, False), "integer scalar"),
        (("trajectory_handoff", "selected_num_leapfrog_steps"), 2.5, "integer scalar"),
        (("trajectory_handoff", "exact_l_retune_seed"), (3.5, 103), "integer scalar"),
        (("transition_ledger", "record_count"), 8.5, "integer scalar"),
        (("seed_domains", "repair_verification"), (7, 8.5), "integer scalar"),
    ],
)
def test_v3_artifact_rejects_noninteger_authority_fields(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = json.loads(json.dumps(_artifact()))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_hmc_tuning_engineering_artifact(payload)


def test_v3_artifact_rejects_unexpected_top_level_payload() -> None:
    payload = json.loads(json.dumps(_artifact()))
    payload["raw_samples"] = [[0.0]]

    with pytest.raises(ValueError, match="field set"):
        validate_hmc_tuning_engineering_artifact(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stale_coordinate", "stale coordinate"),
        ("stale_metric", "stale metric"),
        ("transition_gap", "transition counts"),
        ("fabricated_update_count", "metric-update count"),
        ("final_coordinate", "final kernel lineage"),
        ("fractional_divergence", "integer scalar"),
    ],
)
def test_v3_artifact_recomputes_operational_warmup_claims(
    mutation: str,
    message: str,
) -> None:
    payload = json.loads(json.dumps(_artifact()))
    warmup = payload["warmup"]
    update_index = next(
        index
        for index, window in enumerate(warmup["windows"])
        if window["metric_decision"] is not None
        and window["metric_decision"]["update_applied"] is True
    )
    if mutation == "stale_coordinate":
        warmup["windows"][update_index + 1]["coordinate_signature_used"] = "stale"
    elif mutation == "stale_metric":
        warmup["windows"][update_index + 1]["metric_signature_used"] = "stale"
    elif mutation == "transition_gap":
        warmup["windows"][1]["transition_count_before_window"] += 1
    elif mutation == "fabricated_update_count":
        warmup["operational_metric_update_count"] += 1
    elif mutation == "final_coordinate":
        warmup["final_coordinate_signature"] = "stale-final"
    else:
        warmup["windows"][0]["native_divergence_status"] = "available"
        warmup["windows"][0]["native_divergence_count"] = 1.5
    warmup["every_update_used_by_later_transition"] = True

    with pytest.raises(ValueError, match=message):
        validate_hmc_tuning_engineering_artifact(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("selected_step", "selection lacks final usable evidence"),
        ("usable_flag", "usable flag is inconsistent"),
        ("nonfinite_acceptance", "nonfinite attempt carries acceptance"),
        ("bracket_role", "diagnostic role is invalid"),
        ("bracket_nonclaims", "nonclaims changed"),
        ("warmup_nonclaims", "operational warmup nonclaims changed"),
        ("schedule_config", "configured schedule"),
        ("target_status_policy", "target-status policy is inconsistent"),
        ("target_status_disabled_count", "disabled warmup target status carries"),
        ("target_status_failed_count", "passed warmup carries a target-status failure"),
    ),
)
def test_v3_artifact_rejects_forged_warmup_authority(
    mutation: str,
    message: str,
) -> None:
    payload = json.loads(json.dumps(_artifact()))
    payload.pop("artifact_sha256")
    warmup = payload["warmup"]
    bracket = warmup["reasonable_epsilon"]
    if mutation == "selected_step":
        bracket["selected_step_size"] *= 2.0
    elif mutation == "usable_flag":
        bracket["attempts"][-1]["usable"] = False
    elif mutation == "nonfinite_acceptance":
        bracket["attempts"][0]["finite"] = False
        bracket["attempts"][0]["usable"] = False
    elif mutation == "bracket_role":
        bracket["diagnostic_role"] = "posterior_gate"
    elif mutation == "bracket_nonclaims":
        bracket["nonclaims"] = ["posterior convergence claim"]
    elif mutation == "warmup_nonclaims":
        warmup["nonclaims"] = ["posterior convergence claim"]
    elif mutation == "schedule_config":
        warmup["config"]["initial_buffer"] += 1
    elif mutation == "target_status_policy":
        warmup["windows"][0]["target_status_trace_policy"] = "per_chain_step"
    elif mutation == "target_status_disabled_count":
        warmup["windows"][0]["target_status_failure_count"] = 0
    else:
        warmup["target_status_trace_policy"] = "per_chain_step"
        for window in warmup["windows"]:
            window["target_status_trace_policy"] = "per_chain_step"
            window["target_status_failure_count"] = 0
        warmup["windows"][0]["target_status_failure_count"] = 1

    with pytest.raises(ValueError, match=message):
        validate_hmc_tuning_engineering_artifact(payload, require_hash=False)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["kernel_state"].__setitem__(
                "metric_signature", "corrupt"
            ),
            "metric signature mismatch",
        ),
        (
            lambda payload: payload["private_start_bank"].__setitem__(
                "values", [[0.0]]
            ),
            "unexpected fields",
        ),
        (
            lambda payload: payload["private_start_bank"].__setitem__(
                "source_signature", "corrupt-canonical-source"
            ),
            "private start-bank lineage",
        ),
        (
            lambda payload: payload["private_start_bank"].__setitem__(
                "signature", "corrupt-active-content"
            ),
            "trajectory handoff lineage",
        ),
        (
            lambda payload: payload["seed_domains"].__setitem__(
                "repair_verification",
                payload["seed_domains"]["candidate_selection"],
            ),
            "distinct",
        ),
        (
            lambda payload: payload["transition_ledger"]["records"][5].__setitem__(
                "metric_signature", "changed"
            ),
            "ledger hash mismatch",
        ),
    ],
)
def test_v2_artifact_corruption_and_raw_start_leak_fail_closed(mutator, message: str) -> None:
    payload = json.loads(json.dumps(_artifact()))
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        validate_hmc_tuning_engineering_artifact(payload)


def test_validly_rehashed_signature_change_still_fails_repair_scope() -> None:
    payload = json.loads(json.dumps(_artifact()))
    ledger = payload["transition_ledger"]
    ledger["records"][5]["metric_signature"] = "changed"
    core = {
        "schema": ledger["schema"],
        "records": tuple(ledger["records"]),
        "record_count": ledger["record_count"],
    }
    from bayesfilter.inference.hmc_tuning_artifacts import canonical_sha256

    ledger["ledger_sha256"] = canonical_sha256(core)
    artifact_core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    payload["artifact_sha256"] = canonical_sha256(artifact_core)
    with pytest.raises(ValueError, match="changed or omitted"):
        validate_hmc_tuning_engineering_artifact(payload)


def test_killable_child_timeout_is_parent_finalized_and_hashed(tmp_path: Path) -> None:
    closeout_path = tmp_path / "timeout-closeout.json"
    spec = KillableChildSpec(
        command=(sys.executable, "-c", "import time; time.sleep(10)"),
        timeout_s=0.2,
        closeout_path=closeout_path,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
    )

    closeout = run_killable_child(spec)

    assert closeout["classification"] == "hard_timeout"
    assert closeout["parent_finalized"] is True
    assert closeout["gpu_intentionally_hidden"] is True
    assert closeout["returncode"] != 0
    assert closeout["stdout_retained"] is False
    assert closeout["stderr_retained"] is False
    assert closeout_path.is_file()
    loaded = json.loads(closeout_path.read_text(encoding="ascii"))
    validate_killable_child_closeout(loaded)


def test_killable_child_success_hashes_child_artifact(tmp_path: Path) -> None:
    child_artifact = tmp_path / "child.json"
    closeout_path = tmp_path / "success-closeout.json"
    code = (
        "import json, os, pathlib; "
        "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'; "
        f"pathlib.Path({str(child_artifact)!r}).write_text(json.dumps({{'ok': True}}))"
    )
    closeout = run_killable_child(
        KillableChildSpec(
            command=(sys.executable, "-c", code),
            timeout_s=5.0,
            closeout_path=closeout_path,
            child_artifact_path=child_artifact,
        )
    )

    assert closeout["classification"] == "completed"
    assert closeout["child_artifact_exists"] is True
    assert len(closeout["child_artifact_sha256"]) == 64
    validate_killable_child_closeout(closeout)
