from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.testing import deterministic_lgssm_hmc_phase7_tf as phase7_controller
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
    DEFAULT_CONFIG_PATH,
    HISTORICAL_V1_CONFIG_PATH,
    PHASE7_CONFIG_SCHEMA_V2,
    DeterministicLGSSMPhase7Config,
    DeterministicLGSSMPhase7Error,
    controller_action,
    derive_worker_seed,
    map_final_hmc_samples_to_raw,
    validate_phase7_inputs,
)


def test_phase7_config_pins_runtime_and_diagnostic_contract() -> None:
    config = DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)

    assert config.payload["schema"] == PHASE7_CONFIG_SCHEMA_V2
    assert config.runtime_authority is False
    assert config.worker_count == 2
    assert config.chains_per_worker == 2
    assert config.chain_count == 4
    assert config.payload["execution"]["root_seed"] == [20260711, 701]
    assert config.payload["execution"]["jit_compile"] is True
    assert config.payload["execution"]["use_xla"] is True
    assert config.payload["diagnostics"]["rhat_max"] == 1.01
    assert config.payload["diagnostics"]["bulk_ess_min"] == 1000.0
    assert config.payload["diagnostics"]["tail_ess_min"] == 400.0


def test_phase7_config_rejects_threshold_or_worker_drift(tmp_path: Path) -> None:
    payload = json.loads(HISTORICAL_V1_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["execution"]["worker_count"] = 3
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="two workers"):
        DeterministicLGSSMPhase7Config.load(path)


def test_seed_derivation_is_stable_unique_and_stage_separated() -> None:
    seeds = {
        derive_worker_seed(
            (20260711, 701),
            stage_index=stage,
            check_index=check,
            worker_index=worker,
        )
        for stage in (1, 2)
        for check in range(3)
        for worker in range(2)
    }

    assert len(seeds) == 12
    assert derive_worker_seed(
        (20260711, 701), stage_index=1, check_index=0, worker_index=0
    ) == (20360711, 10701)
    assert derive_worker_seed(
        (20260711, 701), stage_index=2, check_index=1, worker_index=1
    ) == (20461757, 20819)


@pytest.mark.parametrize(
    ("completed", "passed", "expected"),
    [
        (0, False, "run_initial"),
        (2000, False, "extend"),
        (2000, True, "pass"),
        (16000, False, "fail_at_cap"),
        (16000, True, "pass"),
    ],
)
def test_controller_action_is_deterministic(
    completed: int,
    passed: bool,
    expected: str,
) -> None:
    assert controller_action(
        completed=completed,
        initial=2000,
        extension=1000,
        maximum=16000,
        diagnostics_passed=passed,
    ) == expected


class DeterministicLGSSMPosteriorAdapter:
    pass


class _Transform:
    def __init__(self, base_adapter, *, shift: float):
        self.base_adapter = base_adapter
        self.shift = float(shift)

    def latent_to_position(self, values):
        return np.asarray(values) + self.shift


def test_two_transform_mapping_reaches_exact_terminal_target() -> None:
    raw = DeterministicLGSSMPosteriorAdapter()
    phase4 = _Transform(raw, shift=2.0)
    final = _Transform(phase4, shift=3.0)

    mapped = map_final_hmc_samples_to_raw(final, np.zeros((2, 4, 3)))

    np.testing.assert_allclose(mapped, 5.0)


def test_two_transform_mapping_rejects_unknown_depth() -> None:
    raw = object()
    phase4 = _Transform(raw, shift=2.0)
    final = _Transform(phase4, shift=3.0)

    with pytest.raises(DeterministicLGSSMPhase7Error, match="LGSSM target"):
        map_final_hmc_samples_to_raw(final, np.zeros((2, 4, 3)))


def test_config_payload_is_not_mutated_by_load() -> None:
    original = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    loaded = DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)
    assert copy.deepcopy(loaded.payload) == original


def test_preflight_fails_closed_before_private_replay_refresh() -> None:
    base = DeterministicLGSSMPhase7Config.load(HISTORICAL_V1_CONFIG_PATH)
    payload = copy.deepcopy(base.payload)
    payload["artifacts"]["private_replay"] = "/tmp/phase7-test-missing-replay.json"
    config = DeterministicLGSSMPhase7Config(payload=payload, path=base.path)

    with pytest.raises(
        (DeterministicLGSSMPhase7Error, FileNotFoundError),
    ):
        validate_phase7_inputs(config)


@pytest.mark.parametrize("control_flow", [False, True])
def test_failing_worker_hung_peer_uses_bounded_teardown_and_preserves_control_flow(
    monkeypatch: pytest.MonkeyPatch,
    control_flow: bool,
) -> None:
    config = DeterministicLGSSMPhase7Config.load(HISTORICAL_V1_CONFIG_PATH)
    events: list[str] = []

    class FakeProcess:
        def __init__(self, index: int) -> None:
            self.index = index
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            events.append(f"terminate:{self.index}")

        def join(self, timeout: float) -> None:
            assert 0.0 <= timeout <= 5.0
            events.append(f"join:{self.index}")

        def kill(self) -> None:
            events.append(f"kill:{self.index}")
            self.alive = False

    failure: BaseException = (
        KeyboardInterrupt("controller interrupt")
        if control_flow
        else RuntimeError("worker failed while peer hung")
    )

    class FakeFuture:
        def result(self, *, timeout: float):
            assert timeout > 0.0
            raise failure

    class FakeExecutor:
        def __init__(self, index: int) -> None:
            self.index = index
            self._processes = {index: FakeProcess(index)}

        def submit(self, *_args, **_kwargs):
            events.append(f"submit:{self.index}")
            return FakeFuture()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is False
            assert cancel_futures is True
            events.append(f"shutdown:{self.index}")

    executors = [FakeExecutor(0), FakeExecutor(1)]

    def executor_factory(*_args, **_kwargs):
        return executors.pop(0)

    monkeypatch.setattr(
        phase7_controller.concurrent.futures,
        "ProcessPoolExecutor",
        executor_factory,
    )
    monkeypatch.setattr(
        phase7_controller,
        "validate_phase7_inputs",
        lambda _cfg: {
            "target_scope": "test_target",
            "parameter_names": tuple(f"p{index}" for index in range(18)),
        },
    )
    monkeypatch.setattr(
        phase7_controller,
        "_write_runtime_json",
        lambda *_args, **_kwargs: None,
    )
    failure_payload = {"passed": False, "decision": "test_failure"}
    monkeypatch.setattr(
        phase7_controller,
        "_write_controller_failure",
        lambda *_args, **_kwargs: failure_payload,
    )

    if control_flow:
        with pytest.raises(KeyboardInterrupt, match="controller interrupt"):
            phase7_controller.run_phase7(config)
    else:
        assert phase7_controller.run_phase7(config) == failure_payload

    assert events.index("terminate:0") < events.index("join:0")
    assert events.index("terminate:1") < events.index("join:0")
    assert events.count("shutdown:0") == 1
    assert events.count("shutdown:1") == 1
    assert "kill:0" in events and "kill:1" in events


def test_bounded_teardown_finishes_all_peers_then_reraises_cleanup_interrupt() -> None:
    events: list[str] = []

    class FakeProcess:
        def __init__(self, index: int) -> None:
            self.index = index

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            events.append(f"terminate:{self.index}")

        def join(self, timeout: float) -> None:
            events.append(f"join:{self.index}")

        def kill(self) -> None:
            events.append(f"kill:{self.index}")

    class FakeExecutor:
        def __init__(self, index: int) -> None:
            self.index = index
            self._processes = {index: FakeProcess(index)}

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is False
            events.append(f"shutdown:{self.index}")
            if self.index == 0:
                raise KeyboardInterrupt("cleanup interrupt")

    with pytest.raises(KeyboardInterrupt, match="cleanup interrupt"):
        phase7_controller._terminate_executors(
            (FakeExecutor(0), FakeExecutor(1)),
            deadline=phase7_controller.time.monotonic(),
        )
    assert "terminate:0" in events and "terminate:1" in events
    assert "shutdown:0" in events and "shutdown:1" in events


def test_smoke_diagnostics_treat_finite_threshold_failures_as_explanatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import hmc_convergence

    config = DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)
    rows = tuple(
        {
            "parameter": f"p{index}",
            "rank_normalized_split_rhat": 2.0,
            "folded_rank_normalized_split_rhat": 1.5,
            "rhat": 2.0,
            "bulk_ess": 0.5,
            "tail_ess": 0.25,
            "lower_tail_ess": 0.25,
            "upper_tail_ess": 0.5,
            "passed": False,
        }
        for index in range(18)
    )
    payload = {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": False,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "draw_count_per_chain": 8,
        "chain_count": 4,
        "parameter_count": 18,
        "split_draw_count_per_chain": 4,
        "split_chain_count": 8,
        "thresholds": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
        },
        "definitions": {},
        "max_rhat": 2.0,
        "min_bulk_ess": 0.5,
        "min_tail_ess": 0.25,
        "parameter_diagnostics": rows,
        "hard_vetoes": (),
        "nonclaims": (),
    }
    monkeypatch.setattr(
        hmc_convergence,
        "rank_normalized_hmc_diagnostics",
        lambda *_args, **_kwargs: payload,
    )

    observed = phase7_controller._aggregate_diagnostics(
        (np.zeros((8, 2, 18)), np.zeros((8, 2, 18))),
        parameter_names=tuple(f"p{index}" for index in range(18)),
        cfg=config,
        smoke=True,
    )

    assert observed["passed"] is True
    assert all(row["passed"] is True for row in observed["parameter_diagnostics"])
    assert observed["max_rhat"] == 2.0
    assert observed["smoke_gate"] == "finite_diagnostics_only_non_promoting"


def _secure_test_context(*, authority_kind: str = "phase7_serious") -> SimpleNamespace:
    return SimpleNamespace(
        authority_kind=authority_kind,
        authority={"artifact_hash": "sha256:" + "1" * 64},
        claim={"artifact_hash": "sha256:" + "2" * 64},
        proposal_manifest={"artifact_hash": "sha256:" + "3" * 64},
        implementation_source_bundle={"repository_file:test.py": b"test source\n"},
    )


def _secure_test_seal(*, worker_index: int = 0) -> dict:
    config = DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)
    return dict(
        phase7_controller._secure_worker_cache_seal(
            config,
            worker_index=worker_index,
            smoke=False,
            target_scope="test_target",
            launch_context=_secure_test_context(),
        )
    )


def _rehash_seal(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def test_secure_worker_cache_seal_round_trip_and_session_binding() -> None:
    seal = _secure_test_seal()
    assert phase7_controller._parse_secure_worker_cache_seal(seal) == seal

    for field, replacement in (
        ("claim_artifact_hash", "sha256:" + "4" * 64),
        ("worker_index", 1),
        ("target_scope", "other_target"),
        ("transition_identity_hash", "sha256:" + "5" * 64),
        ("implementation_source_bundle_hash", "sha256:" + "6" * 64),
        ("total_chain_count", 8),
    ):
        mutated = copy.deepcopy(seal)
        mutated[field] = replacement
        mutated = _rehash_seal(mutated)
        if field == "total_chain_count":
            match = "topology"
        else:
            match = "request/cache seal|cached launch seal|mismatch"
        request = {
            "secure_source_verification": True,
            "action": "burnin",
            "worker_index": 0,
            "smoke": False,
            "target_scope": "test_target",
            "chains_per_worker": 2,
            "total_chain_count": 4,
            "worker_cache_seal": mutated,
        }
        original_cache = dict(phase7_controller._WORKER_CACHE)
        try:
            phase7_controller._WORKER_CACHE.clear()
            phase7_controller._WORKER_CACHE["worker_cache_seal"] = seal
            with pytest.raises(DeterministicLGSSMPhase7Error, match=match):
                phase7_controller._verify_secure_worker_cache_seal(request)
        finally:
            phase7_controller._WORKER_CACHE.clear()
            phase7_controller._WORKER_CACHE.update(original_cache)


def _valid_secure_worker_response() -> tuple[dict, dict]:
    seal = _secure_test_seal()
    response = {
        "schema": "bayesfilter.deterministic_lgssm_hmc_phase7_worker_result.v1",
        "action": "initialize",
        "worker_index": 0,
        "pid": 1234,
        "passed": True,
        "seed": (11, 12),
        "final_state": np.zeros((2, 18)),
        "raw_samples": np.empty((0, 2, 18)),
        "diagnostics": {},
        "metadata": {
            "jit_compile": True,
            "use_xla": True,
            "compile_trace_count": 1,
            "cuda_visible_devices": "-1",
            "target_scope": "test_target",
            "child_source_references_verified": True,
            "child_implementation_references_verified": True,
            "child_loaded_source_bytes_verified": True,
            "child_implementation_source_bundle_hash": seal[
                "implementation_source_bundle_hash"
            ],
            "child_transition_identity_verified": True,
            "child_transition_identity_hash": seal["transition_identity_hash"],
            "child_worker_cache_seal_hash": seal["artifact_hash"],
        },
    }
    return response, seal


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(worker_index="0"), "ordering"),
        (lambda item: item.update(pid=True), "PID"),
        (lambda item: item.update(pid=0), "PID"),
        (lambda item: item.update(seed=("11", 12)), "seed"),
        (
            lambda item: item["metadata"].update(compile_trace_count=None),
            "retraced",
        ),
        (
            lambda item: item["metadata"].update(target_scope="other"),
            "target scope",
        ),
        (
            lambda item: item["metadata"].update(
                child_transition_identity_hash="sha256:" + "7" * 64
            ),
            "provenance",
        ),
        (
            lambda item: item["metadata"].update(
                child_implementation_source_bundle_hash="sha256:" + "8" * 64
            ),
            "provenance",
        ),
        (
            lambda item: item["metadata"].update(
                child_worker_cache_seal_hash="sha256:" + "9" * 64
            ),
            "provenance",
        ),
    ],
)
def test_secure_worker_response_rejects_forged_provenance(mutation, match: str) -> None:
    response, seal = _valid_secure_worker_response()
    mutation(response)
    with pytest.raises(DeterministicLGSSMPhase7Error, match=match):
        phase7_controller._assert_worker_response(
            response,
            action="initialize",
            expected_worker_index=0,
            expected_pid=None,
            expected_seed=(11, 12),
            expected_target_scope="test_target",
            expected_transition_identity_hash=seal["transition_identity_hash"],
            expected_implementation_source_bundle_hash=seal[
                "implementation_source_bundle_hash"
            ],
            expected_worker_cache_seal_hash=seal["artifact_hash"],
            secure_source_verification=True,
        )


def test_secure_worker_response_round_trip_and_duplicate_pid_veto() -> None:
    response, seal = _valid_secure_worker_response()
    phase7_controller._assert_worker_response(
        response,
        action="initialize",
        expected_worker_index=0,
        expected_pid=None,
        expected_seed=(11, 12),
        expected_target_scope="test_target",
        expected_transition_identity_hash=seal["transition_identity_hash"],
        expected_implementation_source_bundle_hash=seal[
            "implementation_source_bundle_hash"
        ],
        expected_worker_cache_seal_hash=seal["artifact_hash"],
        secure_source_verification=True,
    )
    pids: list[int] = []
    phase7_controller._record_initialized_worker_pid(pids, response["pid"])
    with pytest.raises(DeterministicLGSSMPhase7Error, match="duplicate PID"):
        phase7_controller._record_initialized_worker_pid(pids, response["pid"])
