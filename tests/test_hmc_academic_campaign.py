from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from bayesfilter.inference.hmc_academic_campaign import (
    ACADEMIC_BLOCK_DECISION,
    ACADEMIC_CONTEXT_KIND,
    ACADEMIC_DIAGNOSTIC_DEFINITIONS,
    ACADEMIC_DIAGNOSTIC_NONCLAIMS,
    ACADEMIC_FAILURE_SCHEMA,
    ACADEMIC_NONCLAIMS,
    ACADEMIC_PASS_DECISION,
    ACADEMIC_PROGRESS_SCHEMA,
    ACADEMIC_RESULT_SCHEMA,
    CAMPAIGN_MAX_ATTEMPTS,
    CAMPAIGN_WALL_TIME_CAP_SECONDS,
    AcademicCampaignError,
    finalize_academic_attempt,
    load_attempt_history,
    prepare_academic_launch,
    release_academic_launch,
    validate_academic_launch_context,
    verify_checksum_manifest,
    write_infrastructure_failure,
)
from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.runtime import atomic_write_json
from bayesfilter.testing import deterministic_lgssm_hmc_phase7_tf as controller
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
    DEFAULT_CONFIG_PATH,
    DeterministicLGSSMPhase7Config,
    DeterministicLGSSMPhase7Error,
    run_phase7,
    validate_phase7_inputs,
)
from scripts.run_hmc_phase7_academic_campaign import (
    classify_controller_result,
    classify_uncaught_error,
)


@pytest.fixture(scope="module")
def config() -> DeterministicLGSSMPhase7Config:
    return DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def preflight(config: DeterministicLGSSMPhase7Config) -> dict:
    return dict(validate_phase7_inputs(config))


def _root(config: DeterministicLGSSMPhase7Config, tmp_path: Path) -> Path:
    return config.artifact_root / f".test-academic-{tmp_path.name}"


def _prepare(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    root: Path,
    *,
    second: int = 0,
    invocation_elapsed_seconds: float = 0.0,
):
    return prepare_academic_launch(
        config=config,
        preflight=preflight,
        command=("python", "scripts/run_hmc_phase7_academic_campaign.py"),
        campaign_root=root,
        now=datetime(2026, 7, 13, 1, 2, second, tzinfo=timezone.utc),
        invocation_elapsed_seconds=invocation_elapsed_seconds,
    )


def _rehash(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _diagnostic_payload(
    preflight: dict,
    *,
    draw_count: int,
    passed: bool,
) -> dict:
    if passed:
        rank_rhat, folded_rhat, bulk, lower, upper = 1.001, 1.002, 1500.0, 800.0, 900.0
    else:
        rank_rhat, folded_rhat, bulk, lower, upper = 1.02, 1.01, 900.0, 300.0, 350.0
    rhat = max(rank_rhat, folded_rhat)
    tail = min(lower, upper)
    rows = [
        {
            "parameter": name,
            "rank_normalized_split_rhat": rank_rhat,
            "folded_rank_normalized_split_rhat": folded_rhat,
            "rhat": rhat,
            "bulk_ess": bulk,
            "tail_ess": tail,
            "lower_tail_ess": lower,
            "upper_tail_ess": upper,
            "passed": passed,
        }
        for name in preflight["parameter_names"]
    ]
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": passed,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "draw_count_per_chain": draw_count,
        "chain_count": 4,
        "parameter_count": 18,
        "split_draw_count_per_chain": draw_count // 2,
        "split_chain_count": 8,
        "thresholds": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
        },
        "definitions": ACADEMIC_DIAGNOSTIC_DEFINITIONS,
        "max_rhat": rhat,
        "min_bulk_ess": bulk,
        "min_tail_ess": tail,
        "parameter_diagnostics": rows,
        "hard_vetoes": [],
        "nonclaims": ACADEMIC_DIAGNOSTIC_NONCLAIMS,
    }


def _progress_check(*, stage: str, completed: int, diagnostic: dict) -> dict:
    return {
        "stage": stage,
        "completed_results_per_chain": completed,
        "passed": diagnostic["passed"],
        "max_rhat": diagnostic["max_rhat"],
        "min_bulk_ess": diagnostic["min_bulk_ess"],
        "min_tail_ess": diagnostic["min_tail_ess"],
        "input_all_finite": diagnostic["input_all_finite"],
        "diagnostics_all_finite": diagnostic["diagnostics_all_finite"],
        "hard_vetoes": diagnostic["hard_vetoes"],
    }


def _write_strict_pass_graph(context, preflight: dict) -> tuple[dict, dict]:
    diagnostic = _diagnostic_payload(preflight, draw_count=4000, passed=True)
    np.savez_compressed(
        context.paths["private_samples_path"],
        retained_raw_samples=np.zeros((4000, 4, 18), dtype=np.float64),
        final_worker_states=np.zeros((2, 2, 18), dtype=np.float64),
        config_hash=np.asarray(context.config.hash),
        private_replay_hash=np.asarray(preflight["private_replay_artifact_hash"]),
    )
    private_path = context.paths["private_samples_path"]
    pids = [101, 202]
    result = _rehash(
        {
            "schema": ACADEMIC_RESULT_SCHEMA,
            "passed": True,
            "decision": ACADEMIC_PASS_DECISION,
            "smoke": False,
            "campaign_id": context.campaign_id,
            "attempt_number": context.attempt_number,
            "run_manifest_artifact_hash": context.manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
            "config_hash": context.config.hash,
            "preflight_before_runtime": preflight,
            "burnin_results_per_chain": 2000,
            "retained_results_per_chain": 4000,
            "final_diagnostics": diagnostic,
            "worker_count": 2,
            "chains_per_worker": 2,
            "chain_count": 4,
            "worker_pids": pids,
            "worker_metadata": [
                {
                    "worker_index": index,
                    "pid": pid,
                    "jit_compile": True,
                    "use_xla": True,
                    "compile_trace_count": 1,
                    "cuda_visible_devices": "-1",
                    "child_transition_identity_hash": context.config.payload[
                        "adopted_identities"
                    ]["transition_identity_hash"],
                    "child_transition_identity_verified": True,
                    "child_implementation_verification_status": (
                        "launch_inventory_only_not_child_byte_verified"
                    ),
                    "launch_implementation_inventory_hash": context.manifest[
                        "implementation_inventory_hash"
                    ],
                }
                for index, pid in enumerate(pids)
            ],
            "worker_teardown": {"all_exited": True, "worker_pids": pids},
            "private_retained_sample_reference": {
                "file_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
                "byte_count": private_path.stat().st_size,
                "shape_verified": True,
                "finite_verified": True,
                "provenance_verified": True,
                "path_publicized": False,
                "raw_samples_publicized": False,
            },
            "jit_compile": True,
            "jit_compile_false_runtime_executed": False,
            "cuda_visible_devices": "-1",
            "elapsed_seconds": 1.0,
            "controller_entered": True,
            "workers_started": True,
            "hmc_transition_executed": True,
            "serious_runtime_executed": True,
            "phase8_executed": False,
            "neutra_executed": False,
            "nonclaims": ACADEMIC_NONCLAIMS,
        }
    )
    progress = _rehash(
        {
            "schema": ACADEMIC_PROGRESS_SCHEMA,
            "status": "result_written",
            "config_hash": context.config.hash,
            "smoke": False,
            "campaign_id": context.campaign_id,
            "attempt_number": context.attempt_number,
            "run_manifest_artifact_hash": context.manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
            "burnin_checks": [
                _progress_check(stage="burnin", completed=2000, diagnostic=diagnostic)
            ],
            "retained_checks": [
                _progress_check(stage="retained", completed=4000, diagnostic=diagnostic)
            ],
            "completed": True,
            "passed": True,
            "result_artifact_hash": result["artifact_hash"],
        }
    )
    atomic_write_json(context.paths["public_result_path"], result)
    atomic_write_json(context.paths["public_progress_path"], progress)
    return result, progress


def _failure_payload(
    context,
    *,
    classification: str = "infrastructure_failure",
    reason: str = "RuntimeError: worker process failed",
) -> dict:
    return _rehash(
        {
            "schema": ACADEMIC_FAILURE_SCHEMA,
            "passed": False,
            "decision": ACADEMIC_BLOCK_DECISION,
            "campaign_id": context.campaign_id,
            "attempt_number": context.attempt_number,
            "failure_classification": classification,
            "stage": "test",
            "reason": reason,
            "config_hash": context.config.hash,
            "run_manifest_artifact_hash": context.manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": context.preflight[
                "artifact_hash"
            ],
            "transition_identity_hash": context.config.payload[
                "adopted_identities"
            ]["transition_identity_hash"],
            "serious_execution_contract_hash": context.config.payload[
                "adopted_identities"
            ]["serious_execution_contract_hash"],
            "elapsed_seconds": 1.0,
            "controller_entered": True,
            "workers_started": False,
            "hmc_transition_executed": False,
            "serious_runtime_executed": False,
            "phase8_executed": False,
            "neutra_executed": False,
            "nonclaims": ["test failure only"],
        }
    )


def _finish_failure(
    context,
    *,
    elapsed: float = 10.0,
    classification: str = "infrastructure_failure",
) -> dict:
    atomic_write_json(
        context.paths["failure_path"],
        _failure_payload(context, classification=classification),
    )
    return dict(
        finalize_academic_attempt(
            context,
            elapsed_seconds=elapsed,
            terminal_path=context.paths["failure_path"],
        )
    )


def test_real_preflight_pins_academic_typed_identities(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
) -> None:
    assert preflight["identity_hashes"]["transition_identity_hash"] == (
        "sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a"
    )
    assert preflight["identity_hashes"]["serious_execution_contract_hash"] == (
        "sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4"
    )
    assert config.runtime_authority is False


def test_context_round_trip_has_environment_source_inventory_and_no_authority(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root, invocation_elapsed_seconds=2.0)
        validate_academic_launch_context(context, config=config)
        manifest = json.loads(
            context.paths["run_manifest_path"].read_text(encoding="utf-8")
        )
        assert context.context_kind == ACADEMIC_CONTEXT_KIND
        assert context.attempt_number == 1
        assert manifest["python_executable"]
        assert manifest["python_version"]
        assert manifest["data_version"] == config.payload[
            "governed_source_references"
        ]["fixture"]["file_sha256"]
        assert len(manifest["implementation_references"]) > 20
        assert manifest["controller_wall_time_cap_seconds"] < (
            CAMPAIGN_WALL_TIME_CAP_SECONDS - 2.0
        )
        text = json.dumps(manifest, sort_keys=True)
        assert "authority_artifact_hash" not in text
        assert "launch_claim_artifact_hash" not in text
        assert not any(
            "authority" in path.name or "claim" in path.name
            for path in root.rglob("*")
        )
    finally:
        if "context" in locals():
            release_academic_launch(context)
        shutil.rmtree(root, ignore_errors=True)


def test_context_rejects_manifest_budget_or_deadline_drift(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        for forged in (
            replace(
                context,
                remaining_wall_time_seconds=context.remaining_wall_time_seconds + 1.0,
            ),
            replace(
                context,
                controller_wall_time_cap_seconds=(
                    context.controller_wall_time_cap_seconds + 1.0
                ),
            ),
            replace(
                context,
                controller_deadline_monotonic=(
                    context.controller_deadline_monotonic + 100.0
                ),
            ),
        ):
            with pytest.raises(AcademicCampaignError, match="context/manifest|deadline"):
                validate_academic_launch_context(forged, config=config)
    finally:
        if "context" in locals():
            release_academic_launch(context)
        shutil.rmtree(root, ignore_errors=True)


def test_typed_identity_drift_is_rejected_before_directory_creation(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    tampered = copy.deepcopy(preflight)
    tampered["identity_hashes"]["transition_identity_hash"] = "sha256:" + "0" * 64
    with pytest.raises(AcademicCampaignError, match="preflight identity drift"):
        prepare_academic_launch(
            config=config,
            preflight=tampered,
            command=("python", "launcher.py"),
            campaign_root=root,
        )
    assert not root.exists()


def test_run_directory_collision_is_fail_closed(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        root.mkdir()
        (root / "attempt-001-20260713T010200Z").write_text(
            "occupied\n", encoding="utf-8"
        )
        with pytest.raises(AcademicCampaignError, match="run directory collision"):
            _prepare(config, preflight, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_concurrent_preparation_is_blocked(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        with pytest.raises(AcademicCampaignError, match="launch is active"):
            _prepare(config, preflight, root, second=1)
    finally:
        if "context" in locals():
            release_academic_launch(context)
        shutil.rmtree(root, ignore_errors=True)


def test_infrastructure_failure_allows_contiguous_retry_and_charges_time(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        first = _prepare(config, preflight, root, second=0)
        summary = _finish_failure(first, elapsed=123.5)
        assert summary["classification"] == "infrastructure_failure"
        second = _prepare(config, preflight, root, second=1)
        assert second.attempt_number == 2
        assert second.prior_cumulative_wall_time_seconds == 123.5
        assert second.remaining_wall_time_seconds == (
            CAMPAIGN_WALL_TIME_CAP_SECONDS - 123.5
        )
    finally:
        if "second" in locals():
            release_academic_launch(second)
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(
    "classification",
    ["continuation_veto", "diagnostic_cap_failure"],
)
def test_non_infrastructure_terminal_blocks_retry(
    classification: str,
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        if classification == "diagnostic_cap_failure":
            payload = _failure_payload(
                context,
                classification=classification,
                reason="retained_diagnostics_failed_at_cap",
            )
            payload["final_diagnostics"] = _diagnostic_payload(
                preflight, draw_count=40000, passed=False
            )
            payload["workers_started"] = True
            payload["hmc_transition_executed"] = True
            payload["serious_runtime_executed"] = True
            payload = _rehash(payload)
            atomic_write_json(context.paths["failure_path"], payload)
            finalize_academic_attempt(
                context,
                elapsed_seconds=10.0,
                terminal_path=context.paths["failure_path"],
            )
        else:
            _finish_failure(context, classification=classification)
        with pytest.raises(AcademicCampaignError, match="terminal after"):
            _prepare(config, preflight, root, second=1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_attempt_cap_blocks_fourth_infrastructure_retry(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        for index in range(CAMPAIGN_MAX_ATTEMPTS):
            context = _prepare(config, preflight, root, second=index)
            _finish_failure(context, elapsed=1.0)
        with pytest.raises(AcademicCampaignError, match="attempt cap reached"):
            _prepare(config, preflight, root, second=10)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cleanup_overrun_is_terminal_and_history_remains_readable(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        summary = _finish_failure(
            context,
            elapsed=float(CAMPAIGN_WALL_TIME_CAP_SECONDS) + 2.0,
        )
        assert summary["remaining_wall_time_seconds"] == 0.0
        assert summary["budget_overrun_seconds"] == 2.0
        assert len(load_attempt_history(root)) == 1
        with pytest.raises(AcademicCampaignError, match="wall-time budget exhausted"):
            _prepare(config, preflight, root, second=1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_terminal_semantic_tamper_is_rejected_before_checksums(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        payload = _failure_payload(context)
        payload["campaign_id"] = "wrong"
        atomic_write_json(context.paths["failure_path"], _rehash(payload))
        with pytest.raises(AcademicCampaignError, match="campaign link"):
            finalize_academic_attempt(
                context,
                elapsed_seconds=1.0,
                terminal_path=context.paths["failure_path"],
            )
        assert not context.paths["checksums_path"].exists()
        assert not context.paths["attempt_summary_path"].exists()
    finally:
        if "context" in locals():
            release_academic_launch(context)
        shutil.rmtree(root, ignore_errors=True)


def test_complete_strict_pass_graph_finalizes_and_blocks_retry(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        _write_strict_pass_graph(context, preflight)
        summary = finalize_academic_attempt(
            context,
            elapsed_seconds=12.0,
            terminal_path=context.paths["public_result_path"],
        )
        assert summary["classification"] == "strict_pass"
        assert summary["exit_code"] == 0
        verify_checksum_manifest(context.paths["checksums_path"], context.run_directory)
        with pytest.raises(AcademicCampaignError, match="terminal after"):
            _prepare(config, preflight, root, second=1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(
    "tamper",
    ["missing_progress", "schedule", "derived_metric", "draw_count"],
)
def test_strict_pass_graph_tamper_is_rejected(
    tamper: str,
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        result, progress = _write_strict_pass_graph(context, preflight)
        if tamper == "missing_progress":
            context.paths["public_progress_path"].unlink()
        elif tamper == "schedule":
            progress["retained_checks"][0]["completed_results_per_chain"] = 6000
            atomic_write_json(context.paths["public_progress_path"], _rehash(progress))
        elif tamper == "derived_metric":
            result["final_diagnostics"]["parameter_diagnostics"][0]["rhat"] = 1.0
            result["final_diagnostics"]["max_rhat"] = 1.0
            atomic_write_json(context.paths["public_result_path"], _rehash(result))
        else:
            result["final_diagnostics"]["draw_count_per_chain"] = 6000
            result["final_diagnostics"]["split_draw_count_per_chain"] = 3000
            atomic_write_json(context.paths["public_result_path"], _rehash(result))
        with pytest.raises(AcademicCampaignError):
            finalize_academic_attempt(
                context,
                elapsed_seconds=1.0,
                terminal_path=context.paths["public_result_path"],
            )
        assert not context.paths["checksums_path"].exists()
    finally:
        if "context" in locals():
            release_academic_launch(context)
        shutil.rmtree(root, ignore_errors=True)


def test_terminal_checksum_tamper_is_detected(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        _finish_failure(context)
        verify_checksum_manifest(context.paths["checksums_path"], context.run_directory)
        context.paths["failure_path"].write_text("{}\n", encoding="utf-8")
        with pytest.raises(AcademicCampaignError, match="checksum mismatch"):
            verify_checksum_manifest(context.paths["checksums_path"], context.run_directory)
        with pytest.raises(AcademicCampaignError, match="terminal artifact drift"):
            load_attempt_history(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_launcher_written_failure_has_truthful_execution_facts(
    config: DeterministicLGSSMPhase7Config,
    preflight: dict,
    tmp_path: Path,
) -> None:
    root = _root(config, tmp_path)
    try:
        context = _prepare(config, preflight, root)
        payload = write_infrastructure_failure(
            context,
            stage="launcher_supervision",
            error=RuntimeError("pre-worker failure"),
            elapsed_seconds=1.0,
        )
        assert payload["controller_entered"] is True
        assert payload["workers_started"] is None
        assert payload["hmc_transition_executed"] is None
        assert payload["serious_runtime_executed"] is None
    finally:
        if "context" in locals():
            release_academic_launch(context)
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"passed": True}, "strict_pass"),
        (
            {"passed": False, "reason": "retained_diagnostics_failed_at_cap"},
            "diagnostic_cap_failure",
        ),
        (
            {"passed": False, "failure_classification": "infrastructure_failure"},
            "infrastructure_failure",
        ),
        ({"passed": False, "reason": "runtime_error:ValueError"}, "continuation_veto"),
    ],
)
def test_launcher_failure_classification(payload: dict, expected: str) -> None:
    assert classify_controller_result(payload) == expected


def test_uncaught_contract_error_is_not_retryable() -> None:
    assert classify_uncaught_error(DeterministicLGSSMPhase7Error("drift")) == (
        "continuation_veto"
    )
    assert classify_uncaught_error(RuntimeError("worker died")) == (
        "infrastructure_failure"
    )
    for error in (
        DeterministicLGSSMPhase7Error("child governed snapshots are incomplete"),
        DeterministicLGSSMPhase7Error("source contract path mismatch"),
        ValueError("governed source snapshot reference mismatch"),
    ):
        assert controller._classify_academic_runtime_failure(
            error, stage="preflight_passed"
        ) == "continuation_veto"


def test_generic_v2_runtime_remains_rejected_without_context(
    config: DeterministicLGSSMPhase7Config,
    tmp_path: Path,
) -> None:
    with pytest.raises(DeterministicLGSSMPhase7Error, match="runtime is not authorized"):
        run_phase7(
            config,
            output_override=tmp_path / "result.json",
            progress_override=tmp_path / "progress.json",
            private_samples_override=tmp_path / "samples.npz",
        )


def test_controller_rejects_multiple_context_kinds_before_worker_creation(
    config: DeterministicLGSSMPhase7Config,
) -> None:
    with pytest.raises(DeterministicLGSSMPhase7Error, match="mutually exclusive"):
        run_phase7(
            config,
            smoke_launch_context=object(),
            academic_launch_context=object(),
        )


def test_academic_worker_reports_launch_inventory_not_child_byte_verification() -> None:
    request = {
        "secure_source_verification": False,
        "launch_implementation_inventory_hash": "sha256:" + "1" * 64,
    }
    evidence = controller._cached_child_implementation_identity(request)
    assert evidence == {
        "child_implementation_references_verified": False,
        "child_implementation_verification_status": (
            "launch_inventory_only_not_child_byte_verified"
        ),
        "launch_implementation_inventory_hash": "sha256:" + "1" * 64,
    }


def test_remaining_time_and_inventory_are_forwarded_to_worker_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFuture:
        def result(self, *, timeout: float):
            assert 0.0 < timeout <= 7.0
            return {"response": True}

    class FakeExecutor:
        def submit(self, *_args, **_kwargs):
            return FakeFuture()

    observed: list[dict] = []
    monkeypatch.setattr(
        controller,
        "_assert_worker_response",
        lambda *_args, **kwargs: observed.append(kwargs),
    )
    monkeypatch.setattr(
        controller,
        "_worker_request",
        lambda *_args, **kwargs: {"seed": kwargs["seed"]},
    )
    config = type("Config", (), {"payload": {"execution": {"wall_time_cap_seconds": 99}}})()
    inventory_hash = "sha256:" + "2" * 64
    controller._run_worker_round(
        [FakeExecutor(), FakeExecutor()],
        config,
        action="retained",
        count=1,
        stage_index=2,
        check_index=0,
        root_seed=(1, 2),
        worker_env={},
        smoke=False,
        target_scope="target",
        expected_worker_pids=(1, 2),
        start=controller.time.monotonic(),
        wall_time_cap_seconds=7.0,
        launch_implementation_inventory_hash=inventory_hash,
    )
    assert all(
        item["expected_launch_implementation_inventory_hash"] == inventory_hash
        for item in observed
    )


def test_worker_transition_dispatch_is_recorded_before_response_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFuture:
        def result(self, *, timeout: float):
            return {"response": True}

    class FakeExecutor:
        def submit(self, *_args, **_kwargs):
            return FakeFuture()

    monkeypatch.setattr(
        controller,
        "_worker_request",
        lambda *_args, **kwargs: {"seed": kwargs["seed"]},
    )
    monkeypatch.setattr(
        controller,
        "_assert_worker_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DeterministicLGSSMPhase7Error("response mismatch")
        ),
    )
    config = type("Config", (), {
        "payload": {"execution": {"wall_time_cap_seconds": 99}}
    })()
    dispatched: list[bool] = []
    with pytest.raises(DeterministicLGSSMPhase7Error, match="response mismatch"):
        controller._run_worker_round(
            [FakeExecutor(), FakeExecutor()],
            config,
            action="retained",
            count=1,
            stage_index=2,
            check_index=0,
            root_seed=(1, 2),
            worker_env={},
            smoke=False,
            target_scope="target",
            expected_worker_pids=(1, 2),
            start=controller.time.monotonic(),
            wall_time_cap_seconds=7.0,
            on_transition_dispatched=lambda: dispatched.append(True),
        )
    assert dispatched == [True]


def test_verified_teardown_rejects_surviving_worker() -> None:
    class Process:
        pid = 101

        def __init__(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            pass

        def join(self, timeout: float) -> None:
            pass

        def kill(self) -> None:
            pass

    class Executor:
        def __init__(self) -> None:
            self._processes = {101: Process()}

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            pass

    with pytest.raises(DeterministicLGSSMPhase7Error, match="teardown failed"):
        controller._terminate_executors_verified(
            [Executor()],
            worker_pids=(101,),
            deadline=time.monotonic() + 0.01,
        )


def test_historical_authority_files_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256(
        (root / "bayesfilter/inference/hmc_serious_authority.py").read_bytes()
    ).hexdigest() == "4cb310f1845372c0857693f0e519d6b3f91b779d5502c30fb942e0716f1e2e29"
    assert hashlib.sha256(
        (root / "tests/test_hmc_serious_authority.py").read_bytes()
    ).hexdigest() == "58427c3d66dc7eb4fb9fb5694b5ebd2099419e093364170abb24655c49cdf201"
