"""Deterministic multicore Phase 7 controller for the LGSSM HMC runbook."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesfilter.runtime import atomic_write_json, stable_config_hash


PHASE7_CONFIG_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_phase7_config.v1"
PHASE7_CONFIG_SCHEMA_V2 = "bayesfilter.deterministic_lgssm_hmc_phase7_config.v2"
PHASE7_CONFIG_SCHEMA_V3 = "bayesfilter.deterministic_lgssm_hmc_phase7_config.v3"
PHASE7_RESULT_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_phase7_result.v1"
PHASE7_PROGRESS_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_phase7_progress.v1"
ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_V1_CONFIG_PATH = ROOT / (
    "docs/benchmarks/configs/multidim_lgssm_phase7_burnin_sampling_2026_07_11.json"
)
DEFAULT_CONFIG_PATH = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_phase7_typed_identity_baseline_2026_07_11.json"
)
PHASE3_PUBLIC_ARTIFACT_ROOT = ROOT / (
    "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11"
)
PHASE3_PRIVATE_SIDECAR_NAME = "hmc_semantic_identity_phase3_sidecar.json"
PHASE3_INPUT_MANIFEST_NAME = (
    "hmc_semantic_identity_phase3_input_integrity_manifest.json"
)
PHASE3_PUBLIC_RECORD_NAME = "candidate_semantic_validation.json"
PHASE3_OUTPUT_MANIFEST_NAME = "output_integrity_manifest.json"
PHASE5_ADOPTION_RECORD_PATH = PHASE3_PUBLIC_ARTIFACT_ROOT / (
    "typed_identity_baseline_adoption_record.json"
)
_WORKER_CACHE: dict[str, Any] = {}
SECURE_WORKER_CACHE_SEAL_SCHEMA = (
    "bayesfilter.hmc_phase7_secure_worker_cache_seal.v1"
)


class _SpawnMainBootstrapOverride:
    def __init__(self, bootstrap_path: str | Path) -> None:
        self.bootstrap_path = Path(bootstrap_path).resolve()
        self.main_module = sys.modules["__main__"]
        self.previous_file = getattr(self.main_module, "__file__", None)

    def __enter__(self) -> "_SpawnMainBootstrapOverride":
        self.main_module.__file__ = str(self.bootstrap_path)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.previous_file is None:
            try:
                delattr(self.main_module, "__file__")
            except AttributeError:
                pass
        else:
            self.main_module.__file__ = self.previous_file


class DeterministicLGSSMPhase7Error(RuntimeError):
    """Raised when a Phase 7 continuation veto fires."""


@dataclass(frozen=True)
class DeterministicLGSSMPhase7Config:
    payload: Mapping[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "DeterministicLGSSMPhase7Config":
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("schema") not in {
            PHASE7_CONFIG_SCHEMA,
            PHASE7_CONFIG_SCHEMA_V2,
            PHASE7_CONFIG_SCHEMA_V3,
        }:
            raise ValueError("unsupported Phase 7 config schema")
        config = cls(payload=payload, path=config_path)
        config.validate()
        return config

    @property
    def hash(self) -> str:
        return f"sha256:{stable_config_hash(self.payload)}"

    @property
    def artifact_root(self) -> Path:
        return ROOT / str(self.payload["artifact_root"])

    def artifact_path(self, key: str) -> Path:
        return ROOT / str(self.payload["artifacts"][key])

    @property
    def worker_count(self) -> int:
        return int(self.payload["execution"]["worker_count"])

    @property
    def chains_per_worker(self) -> int:
        return int(self.payload["execution"]["chains_per_worker"])

    @property
    def chain_count(self) -> int:
        return self.worker_count * self.chains_per_worker

    def validate(self) -> None:
        schema = self.payload.get("schema")
        if schema == PHASE7_CONFIG_SCHEMA_V2:
            from bayesfilter.inference.hmc_identity_adoption import (
                parse_phase7_v2_config,
            )

            parse_phase7_v2_config(self.payload)
        elif schema == PHASE7_CONFIG_SCHEMA_V3:
            _validate_phase7_v3_config_payload(self.payload)
        elif schema != PHASE7_CONFIG_SCHEMA:
            raise ValueError("unsupported Phase 7 config schema")
        execution = self.payload["execution"]
        if int(execution["worker_count"]) != 2:
            raise ValueError("Phase 7 requires exactly two workers")
        if int(execution["chains_per_worker"]) != 2:
            raise ValueError("Phase 7 requires exactly two chains per worker")
        if execution.get("cuda_visible_devices") != "-1":
            raise ValueError("Phase 7 requires CUDA_VISIBLE_DEVICES=-1")
        if execution.get("jit_compile") is not True or execution.get("use_xla") is not True:
            raise ValueError("Phase 7 requires XLA/JIT")
        if execution.get("chain_execution_mode") != "tf_function":
            raise ValueError("Phase 7 requires tf_function chain execution")
        if execution.get("compile_workers_sequentially") is not True:
            raise ValueError("Phase 7 requires sequential worker compilation")
        expected_root_seed = (
            (20260713, 701)
            if schema == PHASE7_CONFIG_SCHEMA_V3
            else (20260711, 701)
        )
        if tuple(int(item) for item in execution["root_seed"]) != expected_root_seed:
            raise ValueError("Phase 7 root seed mismatch")
        if int(execution["wall_time_cap_seconds"]) != 28800:
            raise ValueError("Phase 7 wall-time cap mismatch")
        expected_threads = {
            "TF_NUM_INTRAOP_THREADS": "8",
            "TF_NUM_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        if dict(execution.get("thread_environment", {})) != expected_threads:
            raise ValueError("Phase 7 thread environment mismatch")
        diagnostics = self.payload["diagnostics"]
        if (
            float(diagnostics["rhat_max"]) != 1.01
            or float(diagnostics["bulk_ess_min"]) != 1000.0
            or float(diagnostics["tail_ess_min"]) != 400.0
            or diagnostics.get("all_parameters_required") is not True
        ):
            raise ValueError("Phase 7 diagnostic thresholds mismatch")
        _validate_controller_counts(self.payload)

    @property
    def runtime_authority(self) -> bool:
        if self.payload.get("schema") == PHASE7_CONFIG_SCHEMA_V2:
            return bool(self.payload["runtime_authority"])
        return True


@dataclass(frozen=True)
class Phase7LiveReplayBundle:
    fixture: Mapping[str, Any]
    mass: Mapping[str, Any]
    kernel: Mapping[str, Any]
    private_replay: Mapping[str, Any]
    source_config: Any
    source_contract_path: Path
    governed_source_snapshots: Mapping[str, Mapping[str, Any]]
    replay: Any


@dataclass(frozen=True)
class Phase7LiveIdentityBundle:
    replay_bundle: Phase7LiveReplayBundle
    transition: Any
    serious_execution: Any
    smoke_execution: Any
    provenance: Any


def phase7_governed_source_paths(
    config: DeterministicLGSSMPhase7Config,
) -> Mapping[str, Path]:
    """Return the exact immutable sources owned by the Phase 5 baseline."""

    cfg = config
    if cfg.payload.get("schema") == PHASE7_CONFIG_SCHEMA_V3:
        references = cfg.payload["governed_source_references"]
        return {
            name: ROOT / str(references[name]["path"])
            for name in _PHASE7_V3_SOURCE_KEYS
        }
    source_config_path = ROOT / str(cfg.payload["source_tuning_config_path"])
    source_payload = _read_json(source_config_path)
    source_contract = source_payload.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise DeterministicLGSSMPhase7Error("source contract reference is missing")
    source_contract_path = ROOT / str(source_contract.get("path"))
    return {
        "fixture": cfg.artifact_root / "fixture_T120_seed20260709_301.json",
        "xla_compile": cfg.artifact_root / "xla_compile_gate.json",
        "geometry": cfg.artifact_root / "geometry.json",
        "mass": cfg.artifact_root / "mass.json",
        "kernel": cfg.artifact_root / "kernel_tuning.json",
        "private_replay": cfg.artifact_path("private_replay"),
        "source_tuning_config": source_config_path,
        "historical_v1_config": HISTORICAL_V1_CONFIG_PATH,
        "source_contract": source_contract_path,
    }


def build_phase7_live_replay_from_paths(
    *,
    fixture_path: str | Path,
    mass_path: str | Path,
    kernel_path: str | Path,
    private_replay_path: str | Path,
    source_tuning_config_path: str | Path,
    xla_evidence_path: str | Path,
    source_contract_path: str | Path | None = None,
    historical_v1_config_path: str | Path | None = HISTORICAL_V1_CONFIG_PATH,
) -> Phase7LiveReplayBundle:
    """Reconstruct the exact replay consumed by preflight and the worker."""

    from bayesfilter.inference import (
        build_retained_frozen_kernel_hmc_adapter_from_tuning_payload,
    )
    from docs.benchmarks import (
        run_multidim_lgssm_serious_hmc_tuning_2026_07_09 as tuning_driver,
    )

    paths: dict[str, Path] = {
        "fixture": Path(fixture_path),
        "xla_compile": Path(xla_evidence_path),
        "geometry": Path(mass_path).with_name("geometry.json"),
        "mass": Path(mass_path),
        "kernel": Path(kernel_path),
        "private_replay": Path(private_replay_path),
        "source_tuning_config": Path(source_tuning_config_path),
    }
    if historical_v1_config_path is not None:
        paths["historical_v1_config"] = Path(historical_v1_config_path)
    source_config_snapshot = _snapshot_json_source(paths["source_tuning_config"])
    source_config_payload = source_config_snapshot["payload"]
    if source_config_payload.get("schema") != tuning_driver.CONFIG_SCHEMA:
        raise DeterministicLGSSMPhase7Error("source tuning config schema mismatch")
    contract_reference = source_config_payload.get("source_contract")
    if not isinstance(contract_reference, Mapping):
        raise DeterministicLGSSMPhase7Error("source contract reference is missing")
    contract_path = Path(source_contract_path) if source_contract_path is not None else (
        ROOT / str(contract_reference.get("path"))
    )
    expected_contract_path = ROOT / str(contract_reference.get("path"))
    if contract_path.resolve() != expected_contract_path.resolve():
        raise DeterministicLGSSMPhase7Error("source contract path mismatch")
    paths["source_contract"] = contract_path
    snapshots = {
        name: (
            source_config_snapshot
            if name == "source_tuning_config"
            else _snapshot_json_source(path)
        )
        for name, path in paths.items()
    }
    fixture = snapshots["fixture"]["payload"]
    mass = snapshots["mass"]["payload"]
    kernel = snapshots["kernel"]["payload"]
    private_replay = snapshots["private_replay"]["payload"]
    source_config = tuning_driver.DeterministicLGSSMHMCConfig(
        payload=source_config_payload,
        path=paths["source_tuning_config"],
    )
    contract = snapshots["source_contract"]["payload"]
    base_adapter = tuning_driver.DeterministicLGSSMPosteriorAdapter(
        observations=fixture["observations"],
        contract=contract,
        parameter_names=fixture["parameter_names"],
        evidence_path=str(xla_evidence_path),
    )
    target_scope = str(kernel.get("target_scope"))
    if private_replay.get("target_scope") != target_scope:
        raise DeterministicLGSSMPhase7Error("kernel/private replay target mismatch")
    tuning_payload = private_replay.get("tuning_payload")
    if not isinstance(tuning_payload, Mapping):
        raise DeterministicLGSSMPhase7Error("private replay tuning payload is missing")
    replay = build_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
        adapter=base_adapter,
        tuning_payload=tuning_payload,
        initial_position=np.asarray(mass["center"], dtype=float),
        initial_covariance=np.asarray(mass["mass_covariance"], dtype=float),
        parameter_scales=np.asarray(mass["scale"], dtype=float),
        target_scope=target_scope,
    )
    return Phase7LiveReplayBundle(
        fixture=fixture,
        mass=mass,
        kernel=kernel,
        private_replay=private_replay,
        source_config=source_config,
        source_contract_path=contract_path,
        governed_source_snapshots=snapshots,
        replay=replay,
    )


def build_phase7_live_replay(
    config: DeterministicLGSSMPhase7Config,
) -> Phase7LiveReplayBundle:
    paths = phase7_governed_source_paths(config)
    return build_phase7_live_replay_from_paths(
        fixture_path=paths["fixture"],
        mass_path=paths["mass"],
        kernel_path=paths["kernel"],
        private_replay_path=paths["private_replay"],
        source_tuning_config_path=paths["source_tuning_config"],
        xla_evidence_path=paths["xla_compile"],
        source_contract_path=paths["source_contract"],
        historical_v1_config_path=paths.get("historical_v1_config"),
    )


def build_phase7_live_identity_bundle(
    config: DeterministicLGSSMPhase7Config,
) -> Phase7LiveIdentityBundle:
    """Build typed identities from the exact live replay without running HMC."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.inference.hmc_identity import (
        FrozenHMCExecutionContractV1,
        FrozenHMCTransitionIdentityV1,
    )
    from bayesfilter.inference.hmc_identity_integration import (
        build_selection_provenance_from_tuning_payload,
    )

    replay_bundle = build_phase7_live_replay(config)
    transition = FrozenHMCTransitionIdentityV1.from_replay(replay_bundle.replay)
    versions = {
        "tensorflow_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "python_version": platform.python_version(),
    }
    serious = FrozenHMCExecutionContractV1.from_phase7_config(
        transition=transition,
        config=config,
        smoke=False,
        **versions,
    )
    smoke = FrozenHMCExecutionContractV1.from_phase7_config(
        transition=transition,
        config=config,
        smoke=True,
        **versions,
    )
    provenance = build_selection_provenance_from_tuning_payload(
        tuning_payload=replay_bundle.private_replay["tuning_payload"],
        tuning_config_hash=config.payload["source_tuning_config_hash"],
    )
    return Phase7LiveIdentityBundle(
        replay_bundle=replay_bundle,
        transition=transition,
        serious_execution=serious,
        smoke_execution=smoke,
        provenance=provenance,
    )


def derive_worker_seed(
    root_seed: Sequence[int],
    *,
    stage_index: int,
    check_index: int,
    worker_index: int,
) -> tuple[int, int]:
    root = tuple(int(item) for item in root_seed)
    if len(root) != 2:
        raise ValueError("root_seed must contain two integers")
    if stage_index not in {1, 2}:
        raise ValueError("stage_index must be burn-in (1) or retained (2)")
    if check_index < 0 or worker_index < 0:
        raise ValueError("check_index and worker_index must be nonnegative")
    return (
        root[0] + 100000 * stage_index + 1009 * check_index + 37 * worker_index,
        root[1] + 10000 * stage_index + 101 * check_index + 17 * worker_index,
    )


def controller_action(
    *,
    completed: int,
    initial: int,
    extension: int,
    maximum: int,
    diagnostics_passed: bool,
) -> str:
    """Return the deterministic next action for one controller stage."""

    completed = int(completed)
    initial = int(initial)
    extension = int(extension)
    maximum = int(maximum)
    if initial <= 0 or extension <= 0 or maximum < initial:
        raise ValueError("invalid controller counts")
    if completed < initial:
        return "run_initial"
    if diagnostics_passed:
        return "pass"
    if completed >= maximum:
        return "fail_at_cap"
    return "extend"


def run_phase7(
    config: DeterministicLGSSMPhase7Config | None = None,
    *,
    smoke: bool = False,
    output_override: Path | None = None,
    progress_override: Path | None = None,
    private_samples_override: Path | None = None,
    smoke_launch_context: Any | None = None,
    serious_launch_context: Any | None = None,
    academic_launch_context: Any | None = None,
) -> Mapping[str, Any]:
    """Run deterministic burn-in and retained sampling through persistent workers."""

    cfg = DeterministicLGSSMPhase7Config.load() if config is None else config
    v2_smoke = smoke_launch_context is not None
    v2_serious = serious_launch_context is not None
    v2_academic = academic_launch_context is not None
    if sum((v2_smoke, v2_serious, v2_academic)) > 1:
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 launch contexts are mutually exclusive"
        )
    secure_launch_context = smoke_launch_context or serious_launch_context
    launch_context = secure_launch_context or academic_launch_context
    if v2_smoke:
        _validate_smoke_launch_context(
            cfg,
            smoke=smoke,
            context=smoke_launch_context,
            output_override=output_override,
            progress_override=progress_override,
            private_samples_override=private_samples_override,
        )
    elif v2_serious:
        _validate_serious_launch_context(
            cfg,
            smoke=smoke,
            context=serious_launch_context,
            output_override=output_override,
            progress_override=progress_override,
            private_samples_override=private_samples_override,
        )
    elif v2_academic:
        _validate_academic_launch_context(
            cfg,
            smoke=smoke,
            context=academic_launch_context,
            output_override=output_override,
            progress_override=progress_override,
            private_samples_override=private_samples_override,
        )
    elif not cfg.runtime_authority:
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 runtime is not authorized by the active V2 config"
        )
    start = time.monotonic()
    wall_time_cap_seconds = (
        min(
            float(academic_launch_context.controller_wall_time_cap_seconds),
            float(academic_launch_context.controller_deadline_monotonic) - start,
        )
        if v2_academic
        else float(cfg.payload["execution"]["wall_time_cap_seconds"])
    )
    if wall_time_cap_seconds <= 0.0:
        raise DeterministicLGSSMPhase7Error("academic controller deadline reached")
    if launch_context is not None:
        public_output = launch_context.paths["public_result_path"]
        progress_output = launch_context.paths["public_progress_path"]
        private_samples = launch_context.paths["private_samples_path"]
        preflight = launch_context.preflight
    else:
        public_output = output_override or cfg.artifact_path("public_result")
        progress_output = progress_override or cfg.artifact_path("public_progress")
        private_samples = private_samples_override or cfg.artifact_path(
            "private_retained_samples"
        )
        preflight = {"passed": False, "status": "not_run"}
    worker_env = _worker_environment(cfg)
    for key, value in worker_env.items():
        os.environ[str(key)] = str(value)
    root_seed = tuple(int(item) for item in cfg.payload["execution"]["root_seed"])
    counts = _runtime_counts(cfg, smoke=smoke)
    progress: dict[str, Any] = {
        "schema": _progress_schema(launch_context),
        "status": "preflight_passed",
        "config_hash": cfg.hash,
        "smoke": bool(smoke),
        **_runtime_authority_links(launch_context),
        "burnin_checks": [],
        "retained_checks": [],
        "completed": False,
    }
    executors: list[concurrent.futures.ProcessPoolExecutor] = []
    worker_pids: list[int] = []
    current_states: list[np.ndarray | None] = [None] * cfg.worker_count
    burnin_windows: list[np.ndarray | None] = [None] * cfg.worker_count
    retained_chunks: list[list[np.ndarray]] = [[] for _ in range(cfg.worker_count)]
    worker_metadata: list[Mapping[str, Any] | None] = [None] * cfg.worker_count
    worker_cache_seals: list[Mapping[str, Any] | None] = [None] * cfg.worker_count
    hmc_transition_executed = False
    executors_torn_down = False
    primary_result: Mapping[str, Any] | None = None

    def record_hmc_transition_dispatch() -> None:
        nonlocal hmc_transition_executed
        hmc_transition_executed = True

    try:
        if secure_launch_context is not None:
            secure_launch_context.output_session.validate_for_runtime()
        if secure_launch_context is None:
            preflight = validate_phase7_inputs(cfg)
        progress["status"] = "preflight_passed"
        _write_runtime_json(
            progress_output,
            progress,
            smoke_launch_context=launch_context,
            role="public_progress_path",
        )
        if secure_launch_context is not None:
            secure_launch_context.consumed_evidence_session.verify()
        context = multiprocessing.get_context("spawn")
        executor_options: dict[str, Any] = {}
        if secure_launch_context is not None:
            from bayesfilter.inference.hmc_smoke_authority import (
                child_source_loader_initializer,
            )

            initializer, initargs = child_source_loader_initializer(
                references=secure_launch_context.proposal[
                    "implementation_references"
                ],
                source_bundle=secure_launch_context.implementation_source_bundle,
                worker_environment=worker_env,
                python_executable=secure_launch_context.command[0],
                implementation_paths=getattr(
                    secure_launch_context, "implementation_paths", None
                ),
            )
            executor_options = {"initializer": initializer, "initargs": initargs}
        # Spawn executes this empty kernel device before the initializer, so no
        # mutable repository or standard-library source runs first in the child.
        bootstrap_path = (
            Path(os.devnull)
            if secure_launch_context is not None
            else Path(__file__).resolve()
        )
        with _SpawnMainBootstrapOverride(bootstrap_path):
            for _ in range(cfg.worker_count):
                executors.append(
                    concurrent.futures.ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=context,
                        **executor_options,
                    )
                )
            # Compile one worker at a time to avoid simultaneous XLA codegen peaks.
            for worker_index, executor in enumerate(executors):
                initialize_seed = derive_worker_seed(
                    root_seed,
                    stage_index=1,
                    check_index=9999,
                    worker_index=worker_index,
                )
                cache_seal = (
                    _secure_worker_cache_seal(
                        cfg,
                        worker_index=worker_index,
                        smoke=smoke,
                        target_scope=str(preflight["target_scope"]),
                        launch_context=secure_launch_context,
                    )
                    if secure_launch_context is not None
                    else None
                )
                worker_cache_seals[worker_index] = cache_seal
                response = executor.submit(
                    _phase7_worker_command,
                    _worker_request(
                        cfg,
                        worker_index=worker_index,
                        action="initialize",
                        count=0,
                        seed=initialize_seed,
                        state=None,
                        worker_env=worker_env,
                        smoke=smoke,
                        target_scope=str(preflight["target_scope"]),
                        implementation_references=(
                            secure_launch_context.proposal[
                                "implementation_references"
                            ]
                            if secure_launch_context is not None
                            else None
                        ),
                        implementation_source_bundle=(
                            secure_launch_context.implementation_source_bundle
                            if secure_launch_context is not None
                            else None
                        ),
                        launch_implementation_inventory_hash=(
                            academic_launch_context.manifest[
                                "implementation_inventory_hash"
                            ]
                            if v2_academic
                            else None
                        ),
                        secure_source_verification=(
                            secure_launch_context is not None
                        ),
                        worker_cache_seal=cache_seal,
                    ),
                ).result(
                    timeout=_remaining_wall_time(
                        cfg,
                        start,
                        cap_seconds=wall_time_cap_seconds,
                    )
                )
                _assert_worker_response(
                    response,
                    action="initialize",
                    expected_worker_index=worker_index,
                    expected_pid=None,
                    expected_seed=initialize_seed,
                    expected_target_scope=str(preflight["target_scope"]),
                    expected_transition_identity_hash=(
                        _expected_transition_identity_hash(cfg)
                    ),
                    expected_implementation_source_bundle_hash=(
                        cache_seal["implementation_source_bundle_hash"]
                        if cache_seal is not None
                        else None
                    ),
                    expected_worker_cache_seal_hash=(
                        cache_seal["artifact_hash"]
                        if cache_seal is not None
                        else None
                    ),
                    expected_launch_implementation_inventory_hash=(
                        academic_launch_context.manifest[
                            "implementation_inventory_hash"
                        ]
                        if v2_academic
                        else None
                    ),
                    secure_source_verification=secure_launch_context is not None,
                )
                _record_initialized_worker_pid(worker_pids, response["pid"])
                current_states[worker_index] = np.asarray(
                    response["final_state"], dtype=float
                )
                worker_metadata[worker_index] = {
                    **response["metadata"],
                    "worker_index": response["worker_index"],
                    "pid": response["pid"],
                }

        burnin_completed = 0
        burnin_check_index = 0
        burnin_passed = False
        final_burnin_diagnostic: Mapping[str, Any] | None = None
        while True:
            action = controller_action(
                completed=burnin_completed,
                initial=counts["burnin_initial"],
                extension=counts["burnin_extension"],
                maximum=counts["burnin_maximum"],
                diagnostics_passed=burnin_passed,
            )
            if action == "pass":
                break
            if action == "fail_at_cap":
                return _write_controller_failure(
                    cfg,
                    public_output=public_output,
                    progress_output=progress_output,
                    progress=progress,
                    stage="burnin",
                    reason="burnin_diagnostics_failed_at_cap",
                    preflight=preflight,
                    worker_pids=worker_pids,
                    elapsed=time.monotonic() - start,
                    smoke=smoke,
                    smoke_launch_context=launch_context,
                    final_diagnostic=final_burnin_diagnostic,
                    failure_classification=(
                        "diagnostic_cap_failure" if v2_academic else None
                    ),
                    workers_started=bool(worker_pids),
                    hmc_transition_executed=hmc_transition_executed,
                )
            count = (
                counts["burnin_initial"]
                if action == "run_initial"
                else counts["burnin_extension"]
            )
            responses = _run_worker_round(
                executors,
                cfg,
                action="burnin",
                count=count,
                stage_index=1,
                check_index=burnin_check_index,
                root_seed=root_seed,
                worker_env=worker_env,
                smoke=smoke,
                target_scope=str(preflight["target_scope"]),
                expected_worker_pids=worker_pids,
                start=start,
                wall_time_cap_seconds=wall_time_cap_seconds,
                launch_implementation_inventory_hash=(
                    academic_launch_context.manifest[
                        "implementation_inventory_hash"
                    ]
                    if v2_academic
                    else None
                ),
                secure_source_verification=secure_launch_context is not None,
                expected_worker_cache_seals=worker_cache_seals,
                on_transition_dispatched=record_hmc_transition_dispatch,
            )
            for index, response in enumerate(responses):
                current_states[index] = np.asarray(response["final_state"], dtype=float)
                burnin_windows[index] = np.asarray(response["raw_samples"], dtype=float)[
                    -counts["burnin_window"] :
                ]
                worker_metadata[index] = {
                    **response["metadata"],
                    "worker_index": response["worker_index"],
                    "pid": response["pid"],
                }
            burnin_completed += count
            diagnostic = _aggregate_diagnostics(
                burnin_windows,
                parameter_names=preflight["parameter_names"],
                cfg=cfg,
                smoke=smoke,
            )
            _assert_diagnostic_no_hard_vetoes(diagnostic)
            final_burnin_diagnostic = diagnostic
            burnin_passed = bool(diagnostic["passed"])
            progress["status"] = "burnin_check"
            progress["burnin_checks"].append(
                _public_check_summary(
                    diagnostic,
                    completed=burnin_completed,
                    stage="burnin",
                )
            )
            _write_runtime_json(
                progress_output,
                progress,
                smoke_launch_context=launch_context,
                role="public_progress_path",
            )
            burnin_check_index += 1
            _check_wall_time(cfg, start, cap_seconds=wall_time_cap_seconds)

        retained_completed = 0
        retained_check_index = 0
        retained_passed = False
        final_diagnostic: Mapping[str, Any] | None = None
        while True:
            action = controller_action(
                completed=retained_completed,
                initial=counts["retained_initial"],
                extension=counts["retained_extension"],
                maximum=counts["retained_maximum"],
                diagnostics_passed=retained_passed,
            )
            if action == "pass":
                break
            if action == "fail_at_cap":
                return _write_controller_failure(
                    cfg,
                    public_output=public_output,
                    progress_output=progress_output,
                    progress=progress,
                    stage="retained",
                    reason="retained_diagnostics_failed_at_cap",
                    preflight=preflight,
                    worker_pids=worker_pids,
                    elapsed=time.monotonic() - start,
                    smoke=smoke,
                    smoke_launch_context=launch_context,
                    final_diagnostic=final_diagnostic,
                    failure_classification=(
                        "diagnostic_cap_failure" if v2_academic else None
                    ),
                    workers_started=bool(worker_pids),
                    hmc_transition_executed=hmc_transition_executed,
                )
            count = (
                counts["retained_initial"]
                if action == "run_initial"
                else counts["retained_extension"]
            )
            responses = _run_worker_round(
                executors,
                cfg,
                action="retained",
                count=count,
                stage_index=2,
                check_index=retained_check_index,
                root_seed=root_seed,
                worker_env=worker_env,
                smoke=smoke,
                target_scope=str(preflight["target_scope"]),
                expected_worker_pids=worker_pids,
                start=start,
                wall_time_cap_seconds=wall_time_cap_seconds,
                launch_implementation_inventory_hash=(
                    academic_launch_context.manifest[
                        "implementation_inventory_hash"
                    ]
                    if v2_academic
                    else None
                ),
                secure_source_verification=secure_launch_context is not None,
                expected_worker_cache_seals=worker_cache_seals,
                on_transition_dispatched=record_hmc_transition_dispatch,
            )
            for index, response in enumerate(responses):
                current_states[index] = np.asarray(response["final_state"], dtype=float)
                retained_chunks[index].append(
                    np.asarray(response["raw_samples"], dtype=float)
                )
                worker_metadata[index] = {
                    **response["metadata"],
                    "worker_index": response["worker_index"],
                    "pid": response["pid"],
                }
            retained_completed += count
            cumulative = [np.concatenate(chunks, axis=0) for chunks in retained_chunks]
            final_diagnostic = _aggregate_diagnostics(
                cumulative,
                parameter_names=preflight["parameter_names"],
                cfg=cfg,
                smoke=smoke,
            )
            _assert_diagnostic_no_hard_vetoes(final_diagnostic)
            retained_passed = bool(final_diagnostic["passed"])
            progress["status"] = "retained_check"
            progress["retained_checks"].append(
                _public_check_summary(
                    final_diagnostic,
                    completed=retained_completed,
                    stage="retained",
                )
            )
            _write_runtime_json(
                progress_output,
                progress,
                smoke_launch_context=launch_context,
                role="public_progress_path",
            )
            retained_check_index += 1
            _check_wall_time(cfg, start, cap_seconds=wall_time_cap_seconds)

        if final_diagnostic is None:
            raise DeterministicLGSSMPhase7Error("retained diagnostics are missing")
        samples_by_worker = [np.concatenate(chunks, axis=0) for chunks in retained_chunks]
        retained = np.concatenate(samples_by_worker, axis=1)
        _write_private_samples(
            private_samples,
            retained=retained,
            final_worker_states=current_states,
            config_hash=cfg.hash,
            private_replay_hash=preflight["private_replay_artifact_hash"],
            smoke_launch_context=launch_context,
        )
        private_inspection = _inspect_private_samples(
            private_samples,
            expected_draw_count=retained_completed,
            expected_chain_count=cfg.chain_count,
            expected_parameter_count=len(preflight["parameter_names"]),
            expected_config_hash=cfg.hash,
            expected_private_replay_hash=preflight["private_replay_artifact_hash"],
            smoke_launch_context=launch_context,
        )
        private_hash = private_inspection["file_sha256"]
        elapsed = time.monotonic() - start
        academic_teardown: Mapping[str, Any] | None = None
        if v2_academic:
            academic_teardown = _terminate_executors_verified(
                executors,
                worker_pids=worker_pids,
                deadline=time.monotonic() + 10.0,
            )
            executors_torn_down = True
        result = {
            "schema": _result_schema(launch_context),
            "passed": True,
            "decision": _pass_decision(launch_context),
            "smoke": bool(smoke),
            **_runtime_authority_links(launch_context),
            "config_hash": cfg.hash,
            **_preflight_result_evidence(preflight, launch_context),
            "burnin_results_per_chain": burnin_completed,
            "retained_results_per_chain": retained_completed,
            "final_diagnostics": final_diagnostic,
            "worker_count": cfg.worker_count,
            "chains_per_worker": cfg.chains_per_worker,
            "chain_count": cfg.chain_count,
            "worker_pids": worker_pids,
            "worker_metadata": _public_worker_metadata(
                worker_metadata, launch_context
            ),
            **(
                {"worker_teardown": academic_teardown}
                if academic_teardown is not None
                else {}
            ),
            "private_retained_sample_reference": {
                "file_sha256": private_hash,
                "byte_count": private_inspection["byte_count"],
                "shape_verified": private_inspection["shape_verified"],
                "finite_verified": private_inspection["finite_verified"],
                "provenance_verified": private_inspection["provenance_verified"],
                "path_publicized": False,
                "raw_samples_publicized": False,
            },
            "jit_compile": True,
            "jit_compile_false_runtime_executed": False,
            "cuda_visible_devices": "-1",
            "elapsed_seconds": elapsed,
            **_runtime_execution_state(
                launch_context,
                workers_started=bool(worker_pids),
                hmc_transition_executed=hmc_transition_executed,
            ),
            "phase8_executed": False,
            "nonclaims": _result_nonclaims(cfg, launch_context),
        }
        result = _finalize_runtime_artifact(result, launch_context)
        _write_runtime_json(
            public_output,
            result,
            smoke_launch_context=launch_context,
            role="public_result_path",
        )
        primary_result = result
        progress.update(
            {
                "status": "result_written",
                "completed": True,
                "passed": True,
                "result_artifact_hash": result["artifact_hash"],
            }
        )
        _write_runtime_json(
            progress_output,
            _finalize_progress_artifact(progress, launch_context),
            smoke_launch_context=launch_context,
            role="public_progress_path",
        )
        return result
    except BaseException as error:
        consumed_evidence_drift = False
        if secure_launch_context is not None:
            from bayesfilter.inference.hmc_smoke_authority import (
                ConsumedAttempt1EvidenceDriftError,
            )

            consumed_evidence_drift = isinstance(
                error, ConsumedAttempt1EvidenceDriftError
            )
        try:
            _terminate_executors(
                executors,
                deadline=_executor_teardown_deadline(
                    cfg,
                    start,
                    cap_seconds=wall_time_cap_seconds,
                ),
            )
        except BaseException:
            executors_torn_down = True
            if consumed_evidence_drift:
                raise error.with_traceback(error.__traceback__)
            if not isinstance(error, Exception):
                raise error.with_traceback(error.__traceback__)
            raise
        executors_torn_down = True
        if consumed_evidence_drift:
            raise error.with_traceback(error.__traceback__)
        if not isinstance(error, Exception):
            raise
        if primary_result is not None and not v2_academic:
            return primary_result
        if secure_launch_context is not None and secure_launch_context.output_session.nonempty(
            "public_result_path"
        ):
            try:
                existing_primary = secure_launch_context.output_session.read_json(
                    "public_result_path"
                )
                if v2_smoke:
                    from bayesfilter.inference.hmc_smoke_authority import (
                        parse_smoke_terminal_result,
                    )

                    parse_smoke_terminal_result(existing_primary)
                else:
                    from bayesfilter.inference.hmc_serious_authority import (
                        parse_serious_terminal_result,
                    )

                    parse_serious_terminal_result(existing_primary)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                return existing_primary
        failure = _write_controller_failure(
            cfg,
            public_output=public_output,
            progress_output=progress_output,
            progress=progress,
            stage=str(progress.get("status", "unknown")),
            reason=f"runtime_error:{type(error).__name__}",
            preflight=preflight,
            worker_pids=worker_pids,
            elapsed=time.monotonic() - start,
            smoke=smoke,
            smoke_launch_context=launch_context,
            failure_classification=(
                _classify_academic_runtime_failure(
                    error,
                    stage=str(progress.get("status", "unknown")),
                )
                if v2_academic
                else None
            ),
            failure_detail=(str(error) if v2_academic else None),
            workers_started=bool(worker_pids),
            hmc_transition_executed=hmc_transition_executed,
        )
        return failure
    finally:
        if not executors_torn_down:
            _terminate_executors(
                executors,
                deadline=_executor_teardown_deadline(
                    cfg,
                    start,
                    cap_seconds=wall_time_cap_seconds,
                ),
            )


def validate_phase7_v1_inputs(
    config: DeterministicLGSSMPhase7Config,
) -> Mapping[str, Any]:
    """Preserve the historical whole-payload validator without modification."""

    cfg = config
    if cfg.payload.get("schema") != PHASE7_CONFIG_SCHEMA:
        raise ValueError("historical validator requires the Phase 7 V1 schema")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DeterministicLGSSMPhase7Error("CUDA_VISIBLE_DEVICES=-1 is required")
    root = cfg.artifact_root
    artifact_names = {
        "fixture": "fixture_T120_seed20260709_301.json",
        "xla_compile": "xla_compile_gate.json",
        "geometry": "geometry.json",
        "mass": "mass.json",
        "kernel": "kernel_tuning.json",
    }
    artifacts = {
        key: _read_json(root / filename) for key, filename in artifact_names.items()
    }
    expected = cfg.payload["expected_hashes"]
    for key in ("fixture", "xla_compile", "geometry", "mass"):
        _verify_embedded_artifact_hash(artifacts[key], label=key)
        if artifacts[key].get("artifact_hash") != expected[key]:
            raise DeterministicLGSSMPhase7Error(f"{key} artifact hash mismatch")
    kernel = artifacts["kernel"]
    _verify_embedded_artifact_hash(kernel, label="kernel")
    source_config = _read_json(ROOT / str(cfg.payload["source_tuning_config_path"]))
    observed_source_config_hash = f"sha256:{stable_config_hash(source_config)}"
    if observed_source_config_hash != cfg.payload["source_tuning_config_hash"]:
        raise DeterministicLGSSMPhase7Error("source tuning config file hash mismatch")
    if kernel.get("config_hash") != cfg.payload["source_tuning_config_hash"]:
        raise DeterministicLGSSMPhase7Error("source tuning config hash mismatch")
    if kernel.get("final_kernel_hash") != expected["public_final_kernel"]:
        raise DeterministicLGSSMPhase7Error("public final kernel hash mismatch")
    public_final = kernel.get("final_kernel_payload", {})
    checks = {
        "private_loop_final_kernel": public_final.get("phase7_final_kernel_hash"),
        "selected_step": public_final.get("selected_step_hash"),
        "selected_trajectory": public_final.get("selected_trajectory_hash"),
        "adapter": kernel.get("adapter_signature"),
    }
    for key, observed in checks.items():
        if observed != expected[key]:
            raise DeterministicLGSSMPhase7Error(f"{key} hash mismatch")
    private_replay = _read_json(cfg.artifact_path("private_replay"))
    if private_replay.get("schema") != (
        "bayesfilter.deterministic_lgssm_hmc_private_tuning_replay.v1"
    ):
        raise DeterministicLGSSMPhase7Error("private replay schema mismatch")
    private_replay_without_hash = {
        key: value for key, value in private_replay.items() if key != "artifact_hash"
    }
    recomputed_private_hash = (
        f"sha256:{stable_config_hash(private_replay_without_hash)}"
    )
    if private_replay.get("artifact_hash") != recomputed_private_hash:
        raise DeterministicLGSSMPhase7Error("private replay artifact hash mismatch")
    tuning_payload = private_replay.get("tuning_payload")
    if not isinstance(tuning_payload, Mapping):
        raise DeterministicLGSSMPhase7Error("private replay tuning payload is missing")
    loop_payload = tuning_payload.get("tune_verify_repair_loop")
    if not isinstance(loop_payload, Mapping):
        raise DeterministicLGSSMPhase7Error("private replay loop payload is missing")
    private_final = loop_payload.get("final_kernel_payload")
    if not isinstance(private_final, Mapping):
        raise DeterministicLGSSMPhase7Error("private final kernel payload is missing")
    recomputed_loop_hash = stable_config_hash(private_final)
    if recomputed_loop_hash != expected["private_loop_final_kernel"]:
        raise DeterministicLGSSMPhase7Error("private final kernel payload hash mismatch")
    if loop_payload.get("final_kernel_hash") != recomputed_loop_hash:
        raise DeterministicLGSSMPhase7Error("private loop recorded kernel hash mismatch")
    private_file_sha256 = _file_sha256(cfg.artifact_path("private_replay"))
    public_reference = kernel.get("private_replay_reference")
    if not isinstance(public_reference, Mapping) or public_reference.get("available") is not True:
        raise DeterministicLGSSMPhase7Error("public private-replay reference is missing")
    if (
        public_reference.get("artifact_hash") != recomputed_private_hash
        or public_reference.get("file_sha256") != private_file_sha256
        or int(public_reference.get("byte_count", -1))
        != cfg.artifact_path("private_replay").stat().st_size
    ):
        raise DeterministicLGSSMPhase7Error("public private-replay reference mismatch")
    for key, observed in {
        "fixture": private_replay.get("fixture_hash"),
        "xla_compile": private_replay.get("xla_compile_hash"),
        "geometry": private_replay.get("geometry_hash"),
        "mass": private_replay.get("mass_hash"),
        "public_final_kernel": private_replay.get("public_final_kernel_hash"),
        "private_loop_final_kernel": private_replay.get("private_loop_final_kernel_hash"),
        "selected_step": private_replay.get("selected_step_hash"),
        "selected_trajectory": private_replay.get("selected_trajectory_hash"),
        "adapter": private_replay.get("adapter_signature"),
    }.items():
        if observed != expected[key]:
            raise DeterministicLGSSMPhase7Error(f"private replay {key} mismatch")
    fixture = artifacts["fixture"]
    return {
        "passed": True,
        "source_tuning_config_hash": kernel["config_hash"],
        "fixture_hash": fixture["artifact_hash"],
        "xla_compile_hash": artifacts["xla_compile"]["artifact_hash"],
        "geometry_hash": artifacts["geometry"]["artifact_hash"],
        "mass_hash": artifacts["mass"]["artifact_hash"],
        "public_final_kernel_hash": kernel["final_kernel_hash"],
        "private_loop_final_kernel_hash": private_replay["private_loop_final_kernel_hash"],
        "private_replay_artifact_hash": private_replay["artifact_hash"],
        "private_replay_file_sha256": private_file_sha256,
        "adapter_signature": kernel["adapter_signature"],
        "parameter_names": tuple(fixture["parameter_names"]),
        "target_scope": kernel["target_scope"],
    }


def validate_phase7_v2_inputs(
    config: DeterministicLGSSMPhase7Config,
    *,
    adoption_record_path: str | Path = PHASE5_ADOPTION_RECORD_PATH,
) -> Mapping[str, Any]:
    """Validate the approved typed baseline without launching HMC runtime."""

    from bayesfilter.inference.hmc_identity import (
        artifact_file_sha256,
        canonical_artifact_payload_hash,
    )
    from bayesfilter.inference.hmc_identity_adoption import (
        GOVERNED_SOURCE_KEYS,
        HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1,
        LEGACY_GATE_STATUS,
        build_phase5_artifact_reference,
        build_phase5_preflight_report,
        parse_phase7_v2_config,
        validate_phase7_v2_config_against_historical,
        verify_phase5_adoption_record,
        verify_phase5_artifact_reference,
        verify_phase7_v2_sources,
    )
    from bayesfilter.inference.hmc_identity_migration_certificate import (
        verify_certificate_output_manifest,
        verify_migration_certificate_sources,
    )

    cfg = config
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DeterministicLGSSMPhase7Error("CUDA_VISIBLE_DEVICES=-1 is required")
    parse_phase7_v2_config(cfg.payload)
    certificate_path = cfg.artifact_root / (
        "private_diagnostics/hmc_semantic_identity_migration_certificate.json"
    )
    public_proposal_path = PHASE3_PUBLIC_ARTIFACT_ROOT / (
        "migration_certificate_proposal.json"
    )
    phase4_manifest_path = PHASE3_PUBLIC_ARTIFACT_ROOT / (
        "migration_certificate_output_manifest.json"
    )
    phase3_sidecar_path = cfg.artifact_root / (
        "private_diagnostics/hmc_semantic_identity_phase3_sidecar.json"
    )
    phase3_input_manifest_path = cfg.artifact_root / (
        "private_diagnostics/hmc_semantic_identity_phase3_input_integrity_manifest.json"
    )
    phase3_public_record_path = PHASE3_PUBLIC_ARTIFACT_ROOT / (
        "candidate_semantic_validation.json"
    )
    phase3_output_manifest_path = PHASE3_PUBLIC_ARTIFACT_ROOT / (
        "output_integrity_manifest.json"
    )
    certificate = _read_json(certificate_path)
    public_proposal = _read_json(public_proposal_path)
    phase4_manifest = _read_json(phase4_manifest_path)
    historical_config = _read_json(HISTORICAL_V1_CONFIG_PATH)
    validate_phase7_v2_config_against_historical(
        cfg.payload,
        historical_config=historical_config,
        certificate=certificate,
    )
    governed_paths = phase7_governed_source_paths(cfg)
    verify_phase7_v2_sources(cfg.payload, source_paths=governed_paths)
    baseline = cfg.payload["baseline_adoption"]
    for name, path in (
        ("certificate_reference", certificate_path),
        ("public_proposal_reference", public_proposal_path),
        ("phase4_output_manifest_reference", phase4_manifest_path),
    ):
        verify_phase5_artifact_reference(baseline[name], path=path)
    phase3_source_paths = {
        "phase7_config": HISTORICAL_V1_CONFIG_PATH,
        "refreshed_kernel": governed_paths["kernel"],
        "refreshed_private_replay": governed_paths["private_replay"],
        "phase3_sidecar": phase3_sidecar_path,
        "phase3_input_manifest": phase3_input_manifest_path,
        "phase3_public_record": phase3_public_record_path,
        "phase3_output_manifest": phase3_output_manifest_path,
    }
    verify_migration_certificate_sources(
        certificate,
        source_paths=phase3_source_paths,
    )
    verify_certificate_output_manifest(
        phase4_manifest,
        certificate_path=certificate_path,
        public_proposal_path=public_proposal_path,
    )
    adoption_path = Path(adoption_record_path)
    adoption_record = _read_json(adoption_path)
    verify_phase5_adoption_record(
        adoption_record,
        v2_config_path=cfg.path,
        historical_v1_config_path=HISTORICAL_V1_CONFIG_PATH,
        certificate_path=certificate_path,
        public_proposal_path=public_proposal_path,
        phase4_output_manifest_path=phase4_manifest_path,
    )

    live = build_phase7_live_identity_bundle(cfg)
    adopted = cfg.payload["adopted_identities"]
    replay = live.replay_bundle.private_replay
    tuning_payload = replay["tuning_payload"]
    identity_hashes = {
        "transition_identity_hash": live.transition.identity_hash,
        "serious_execution_contract_hash": live.serious_execution.identity_hash,
        "smoke_execution_contract_hash": live.smoke_execution.identity_hash,
        "selection_provenance_hash": live.provenance.identity_hash,
        "complete_tuning_payload_hash": canonical_artifact_payload_hash(
            tuning_payload
        ),
        "legacy_replay_canonical_payload_hash": canonical_artifact_payload_hash(
            replay
        ),
    }
    identity_checks = {
        "transition": (
            identity_hashes["transition_identity_hash"]
            == adopted["transition_identity_hash"]
        ),
        "serious_execution": (
            identity_hashes["serious_execution_contract_hash"]
            == adopted["serious_execution_contract_hash"]
        ),
        "smoke_execution": (
            identity_hashes["smoke_execution_contract_hash"]
            == adopted["smoke_execution_contract_hash"]
        ),
        "provenance": (
            identity_hashes["selection_provenance_hash"]
            == adopted["selection_provenance_hash"]
            and identity_hashes["complete_tuning_payload_hash"]
            == adopted["complete_tuning_payload_hash"]
            and identity_hashes["legacy_replay_canonical_payload_hash"]
            == adopted["legacy_replay_canonical_payload_hash"]
        ),
    }
    failed = tuple(name for name, passed in identity_checks.items() if not passed)
    if failed:
        raise DeterministicLGSSMPhase7Error(
            f"Phase 7 V2 typed identity mismatch: {failed}"
        )
    replay_file_sha256 = artifact_file_sha256(governed_paths["private_replay"])
    if replay_file_sha256 != adopted["legacy_replay_file_sha256"]:
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V2 private replay exact-file mismatch"
        )
    historical = cfg.payload["historical_legacy_hashes"]
    refreshed = certificate["refreshed_legacy_hashes"]
    legacy_audit = {
        "status": LEGACY_GATE_STATUS,
        "public_final_kernel": (
            "different"
            if historical["public_final_kernel"] != refreshed["public_final_kernel"]
            else "equal"
        ),
        "private_loop_final_kernel": (
            "different"
            if historical["private_loop_final_kernel"]
            != refreshed["private_loop_final_kernel"]
            else "equal"
        ),
        "selected_trajectory": (
            "different"
            if historical["selected_trajectory"] != refreshed["selected_trajectory"]
            else "equal"
        ),
    }
    integrity_checks = {
        **{name: True for name in GOVERNED_SOURCE_KEYS},
        "certificate": True,
        "public_proposal": True,
        "phase4_output_manifest": True,
        "adoption_record": True,
    }
    return build_phase5_preflight_report(
        config_reference=build_phase5_artifact_reference(
            cfg.path,
            embedded_hash_rule="canonical_without_hash",
        ),
        adoption_record_reference=build_phase5_artifact_reference(
            adoption_path,
            embedded_hash_rule="canonical_without_hash",
        ),
        identity_hashes=identity_hashes,
        identity_checks=identity_checks,
        integrity_checks=integrity_checks,
        legacy_audit=legacy_audit,
        private_replay_artifact_hash=replay["artifact_hash"],
        private_replay_file_sha256=replay_file_sha256,
        parameter_names=live.replay_bundle.fixture["parameter_names"],
        target_scope=str(live.replay_bundle.kernel["target_scope"]),
    )


_PHASE7_V3_SOURCE_KEYS = (
    "fixture",
    "xla_compile",
    "geometry",
    "mass",
    "kernel",
    "private_replay",
    "source_tuning_config",
    "source_contract",
)
_PHASE7_V3_SOURCE_SCHEMAS = {
    "fixture": "bayesfilter.deterministic_lgssm_hmc_tuning_fixture.v1",
    "xla_compile": "bayesfilter.deterministic_lgssm_hmc_tuning_xla_score_gate.v1",
    "geometry": "bayesfilter.deterministic_lgssm_hmc_tuning_geometry.v1",
    "mass": "bayesfilter.deterministic_lgssm_hmc_tuning_mass.v1",
    "kernel": "bayesfilter.deterministic_lgssm_hmc_tuning_kernel.v1",
    "private_replay": (
        "bayesfilter.deterministic_lgssm_hmc_private_tuning_replay.v1"
    ),
    "source_tuning_config": "bayesfilter.deterministic_lgssm_hmc_tuning_config.v1",
    "source_contract": "bayesfilter.multidim_triangular_lgssm.contract.v1",
}
_PHASE7_V3_IDENTITY_KEYS = (
    "transition_identity_hash",
    "serious_execution_contract_hash",
    "smoke_execution_contract_hash",
    "selection_provenance_hash",
    "complete_tuning_payload_hash",
)
_MODERN_RHAT_DEFINITION = (
    "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
)


def _validate_phase7_v3_config_payload(payload: Mapping[str, Any]) -> None:
    fields = {
        "schema",
        "config_id",
        "plan_path",
        "source_tuning_config_path",
        "source_tuning_config_hash",
        "artifact_root",
        "execution",
        "burnin",
        "retained",
        "diagnostics",
        "artifacts",
        "governed_source_references",
        "expected_identities",
        "fresh_run_policy",
        "nonclaims",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Phase 7 V3 config fields mismatch")
    if payload.get("schema") != PHASE7_CONFIG_SCHEMA_V3:
        raise ValueError("Phase 7 V3 schema mismatch")
    for name in (
        "config_id",
        "plan_path",
        "source_tuning_config_path",
        "artifact_root",
    ):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"Phase 7 V3 {name} is invalid")
    _require_tagged_sha256(
        payload.get("source_tuning_config_hash"),
        label="Phase 7 V3 source tuning config hash",
    )
    references = payload.get("governed_source_references")
    if not isinstance(references, Mapping) or set(references) != set(
        _PHASE7_V3_SOURCE_KEYS
    ):
        raise ValueError("Phase 7 V3 governed source inventory mismatch")
    for name in _PHASE7_V3_SOURCE_KEYS:
        reference = references[name]
        if not isinstance(reference, Mapping) or set(reference) != {
            "path",
            "schema",
            "file_sha256",
            "byte_count",
            "artifact_hash",
        }:
            raise ValueError(f"Phase 7 V3 source reference fields mismatch: {name}")
        path = Path(str(reference.get("path", "")))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"Phase 7 V3 source path is not repository relative: {name}")
        if reference.get("schema") != _PHASE7_V3_SOURCE_SCHEMAS[name]:
            raise ValueError(f"Phase 7 V3 source schema mismatch: {name}")
        _require_bare_sha256(
            reference.get("file_sha256"),
            label=f"Phase 7 V3 {name} file hash",
        )
        if type(reference.get("byte_count")) is not int or reference["byte_count"] <= 0:
            raise ValueError(f"Phase 7 V3 source byte count mismatch: {name}")
        if name in {
            "fixture",
            "xla_compile",
            "geometry",
            "mass",
            "kernel",
            "private_replay",
        }:
            _require_tagged_sha256(
                reference.get("artifact_hash"),
                label=f"Phase 7 V3 {name} artifact hash",
            )
        elif reference.get("artifact_hash") is not None:
            raise ValueError(f"Phase 7 V3 {name} must not invent an artifact hash")
    if references["source_tuning_config"]["path"] != payload[
        "source_tuning_config_path"
    ]:
        raise ValueError("Phase 7 V3 source tuning config path mismatch")
    identities = payload.get("expected_identities")
    if not isinstance(identities, Mapping) or set(identities) != set(
        _PHASE7_V3_IDENTITY_KEYS
    ):
        raise ValueError("Phase 7 V3 expected identity inventory mismatch")
    for name in _PHASE7_V3_IDENTITY_KEYS:
        _require_tagged_sha256(
            identities[name],
            label=f"Phase 7 V3 {name}",
        )
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get(
        "rhat_definition"
    ) != _MODERN_RHAT_DEFINITION:
        raise ValueError("Phase 7 V3 R-hat definition mismatch")
    policy = payload.get("fresh_run_policy")
    expected_policy = {
        "historical_identity_inputs_allowed": False,
        "migration_certificates_required": False,
        "approval_manifests_required": False,
        "no_overwrite": True,
        "tuning_root_seed": [20260709, 501],
        "serious_root_seed": [20260713, 701],
    }
    if policy != expected_policy:
        raise ValueError("Phase 7 V3 fresh-run policy mismatch")
    root = Path(str(payload["artifact_root"]))
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("Phase 7 V3 artifact root must be repository relative")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "public_result",
        "public_progress",
        "private_replay",
        "private_retained_samples",
    }:
        raise ValueError("Phase 7 V3 artifact inventory mismatch")
    for name, value in artifacts.items():
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
            raise ValueError(f"Phase 7 V3 output escapes artifact root: {name}")
    if artifacts["private_replay"] != references["private_replay"]["path"]:
        raise ValueError("Phase 7 V3 private replay path mismatch")


def _require_bare_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be a bare lowercase SHA-256")
    return value


def _require_tagged_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a tagged SHA-256")
    _require_bare_sha256(value[7:], label=label)
    return value


def _verify_v3_source_reference_snapshot(
    reference: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    name: str,
) -> None:
    expected_path = ROOT / str(reference["path"])
    observed_path = Path(snapshot["path"])
    if observed_path.resolve() != expected_path.resolve() or observed_path != expected_path:
        raise DeterministicLGSSMPhase7Error(
            f"Phase 7 V3 governed source path mismatch: {name}"
        )
    if snapshot.get("file_sha256") != reference["file_sha256"] or int(
        snapshot.get("byte_count", -1)
    ) != int(reference["byte_count"]):
        raise DeterministicLGSSMPhase7Error(
            f"Phase 7 V3 governed source file mismatch: {name}"
        )
    source = snapshot.get("payload")
    if not isinstance(source, Mapping) or source.get("schema") != reference["schema"]:
        raise DeterministicLGSSMPhase7Error(
            f"Phase 7 V3 governed source schema mismatch: {name}"
        )
    if reference["artifact_hash"] is not None:
        _verify_embedded_artifact_hash(source, label=name)
        if source.get("artifact_hash") != reference["artifact_hash"]:
            raise DeterministicLGSSMPhase7Error(
                f"Phase 7 V3 governed source artifact mismatch: {name}"
            )


def _phase7_v3_terminal_tuning_diagnostics(
    private_replay: Mapping[str, Any],
) -> Mapping[str, Any]:
    tuning = private_replay.get("tuning_payload")
    loop = tuning.get("tune_verify_repair_loop") if isinstance(tuning, Mapping) else None
    attempts = loop.get("attempts") if isinstance(loop, Mapping) else None
    if (
        not isinstance(loop, Mapping)
        or loop.get("passed") is not True
        or loop.get("final_status") != "passed"
        or tuple(loop.get("hard_vetoes", ()))
        or not isinstance(attempts, Sequence)
        or isinstance(attempts, (str, bytes))
        or not attempts
    ):
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 private tuning loop did not pass"
        )
    attempt = attempts[-1]
    diagnostics = attempt.get("verification_diagnostics") if isinstance(
        attempt, Mapping
    ) else None
    config = attempt.get("verification_config_payload") if isinstance(
        attempt, Mapping
    ) else None
    if (
        not isinstance(diagnostics, Mapping)
        or not isinstance(config, Mapping)
        or attempt.get("passed") is not True
        or attempt.get("final_status") != "passed"
        or tuple(attempt.get("hard_vetoes", ()))
    ):
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 terminal tuning attempt did not pass"
        )
    rank = diagnostics.get("max_rank_normalized_split_rhat")
    folded = diagnostics.get("max_folded_rank_normalized_split_rhat")
    combined = diagnostics.get("max_finite_rhat")
    numbers = (rank, folded, combined, diagnostics.get("acceptance_rate"))
    if any(type(value) not in (int, float) or not np.isfinite(value) for value in numbers):
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 terminal tuning diagnostics are nonfinite or missing"
        )
    if (
        diagnostics.get("passed") is not True
        or diagnostics.get("rhat_definition") != _MODERN_RHAT_DEFINITION
        or diagnostics.get("all_finite_rhat_at_or_below_threshold") is not True
        or diagnostics.get("minimum_retained_pass_gate_satisfied") is not True
        or int(diagnostics.get("retained_sample_count", 0)) < 1000
        or float(combined) != max(float(rank), float(folded))
        or float(combined) > 1.01
        or not 0.65 <= float(diagnostics["acceptance_rate"]) <= 0.75
        or tuple(diagnostics.get("hard_vetoes", ()))
        or config.get("rhat_definition") != _MODERN_RHAT_DEFINITION
        or float(config.get("rhat_threshold", float("nan"))) != 1.01
        or list(config.get("acceptance_band", ())) != [0.65, 0.75]
        or int(config.get("minimum_retained_results_for_pass", 0)) != 1000
    ):
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 corrected tuning handoff mismatch"
        )
    return diagnostics


def validate_phase7_v3_inputs(
    config: DeterministicLGSSMPhase7Config,
    *,
    require_fresh_outputs: bool = True,
) -> Mapping[str, Any]:
    """Validate a direct fresh-run config without historical authority inputs."""

    from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash

    cfg = config
    cfg.validate()
    if cfg.payload.get("schema") != PHASE7_CONFIG_SCHEMA_V3:
        raise ValueError("fresh validator requires the Phase 7 V3 schema")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DeterministicLGSSMPhase7Error("CUDA_VISIBLE_DEVICES=-1 is required")
    paths = phase7_governed_source_paths(cfg)
    if tuple(paths) != _PHASE7_V3_SOURCE_KEYS:
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 governed source inventory mismatch"
        )
    snapshots = {name: _snapshot_json_source(path) for name, path in paths.items()}
    references = cfg.payload["governed_source_references"]
    for name in _PHASE7_V3_SOURCE_KEYS:
        _verify_v3_source_reference_snapshot(
            references[name], snapshot=snapshots[name], name=name
        )
    fixture = snapshots["fixture"]["payload"]
    xla_gate = snapshots["xla_compile"]["payload"]
    geometry = snapshots["geometry"]["payload"]
    mass = snapshots["mass"]["payload"]
    kernel = snapshots["kernel"]["payload"]
    private_replay = snapshots["private_replay"]["payload"]
    source_config = snapshots["source_tuning_config"]["payload"]
    source_hash = f"sha256:{stable_config_hash(source_config)}"
    if source_hash != cfg.payload["source_tuning_config_hash"]:
        raise DeterministicLGSSMPhase7Error("Phase 7 V3 tuning config hash mismatch")
    artifact_hashes = {
        "fixture": fixture["artifact_hash"],
        "xla_compile": xla_gate["artifact_hash"],
        "geometry": geometry["artifact_hash"],
        "mass": mass["artifact_hash"],
    }
    if (
        fixture.get("config_hash") != source_hash
        or xla_gate.get("config_hash") != source_hash
        or xla_gate.get("fixture_hash") != artifact_hashes["fixture"]
        or xla_gate.get("passed") is not True
        or xla_gate.get("jit_compile") is not True
        or xla_gate.get("jit_compile_false_runtime_executed") is not False
        or geometry.get("config_hash") != source_hash
        or geometry.get("fixture_hash") != artifact_hashes["fixture"]
        or geometry.get("xla_compile_hash") != artifact_hashes["xla_compile"]
        or geometry.get("passed") is not True
        or mass.get("config_hash") != source_hash
        or mass.get("geometry_hash") != artifact_hashes["geometry"]
        or mass.get("passed") is not True
        or kernel.get("config_hash") != source_hash
        or kernel.get("fixture_hash") != artifact_hashes["fixture"]
        or kernel.get("xla_compile_hash") != artifact_hashes["xla_compile"]
        or kernel.get("geometry_hash") != artifact_hashes["geometry"]
        or kernel.get("mass_hash") != artifact_hashes["mass"]
        or kernel.get("passed") is not True
        or kernel.get("final_status") != "passed"
        or tuple(kernel.get("hard_vetoes", ()))
        or kernel.get("xla_confirmed") is not True
        or kernel.get("final_kernel_requires_serious_sampling_pass") is not True
    ):
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 tuning artifact lineage or gate mismatch"
        )
    public_replay = kernel.get("private_replay_reference")
    if not isinstance(public_replay, Mapping) or (
        public_replay.get("available") is not True
        or public_replay.get("artifact_hash") != private_replay["artifact_hash"]
        or public_replay.get("file_sha256")
        != snapshots["private_replay"]["file_sha256"]
        or int(public_replay.get("byte_count", -1))
        != snapshots["private_replay"]["byte_count"]
    ):
        raise DeterministicLGSSMPhase7Error(
            "Phase 7 V3 public/private replay reference mismatch"
        )
    for name, expected in {
        "config_hash": source_hash,
        "fixture_hash": artifact_hashes["fixture"],
        "xla_compile_hash": artifact_hashes["xla_compile"],
        "geometry_hash": artifact_hashes["geometry"],
        "mass_hash": artifact_hashes["mass"],
        "public_final_kernel_hash": kernel.get("final_kernel_hash"),
        "adapter_signature": kernel.get("adapter_signature"),
        "target_scope": kernel.get("target_scope"),
    }.items():
        if private_replay.get(name) != expected:
            raise DeterministicLGSSMPhase7Error(
                f"Phase 7 V3 private replay lineage mismatch: {name}"
            )
    terminal_tuning = _phase7_v3_terminal_tuning_diagnostics(private_replay)
    live = build_phase7_live_identity_bundle(cfg)
    expected = cfg.payload["expected_identities"]
    observed = {
        "transition_identity_hash": live.transition.identity_hash,
        "serious_execution_contract_hash": live.serious_execution.identity_hash,
        "smoke_execution_contract_hash": live.smoke_execution.identity_hash,
        "selection_provenance_hash": live.provenance.identity_hash,
        "complete_tuning_payload_hash": canonical_artifact_payload_hash(
            private_replay["tuning_payload"]
        ),
    }
    if observed != expected:
        raise DeterministicLGSSMPhase7Error("Phase 7 V3 typed identity mismatch")
    output_paths = {
        name: cfg.artifact_path(name)
        for name in ("public_result", "public_progress", "private_retained_samples")
    }
    if require_fresh_outputs:
        collisions = tuple(name for name, path in output_paths.items() if path.exists())
        if collisions:
            raise DeterministicLGSSMPhase7Error(
                f"Phase 7 V3 no-overwrite output collision: {collisions}"
            )
    report = {
        "schema": "bayesfilter.deterministic_lgssm_hmc_phase7_v3_preflight.v1",
        "passed": True,
        "config_hash": cfg.hash,
        "source_tuning_config_hash": source_hash,
        "target_scope": str(kernel["target_scope"]),
        "parameter_names": tuple(fixture["parameter_names"]),
        "private_replay_artifact_hash": private_replay["artifact_hash"],
        "identity_hashes": observed,
        "governed_source_file_sha256": {
            name: snapshots[name]["file_sha256"] for name in _PHASE7_V3_SOURCE_KEYS
        },
        "corrected_tuning_handoff": {
            "rhat_definition": terminal_tuning["rhat_definition"],
            "max_rank_normalized_split_rhat": terminal_tuning[
                "max_rank_normalized_split_rhat"
            ],
            "max_folded_rank_normalized_split_rhat": terminal_tuning[
                "max_folded_rank_normalized_split_rhat"
            ],
            "max_rhat": terminal_tuning["max_finite_rhat"],
            "retained_sample_count": terminal_tuning["retained_sample_count"],
            "acceptance_rate": terminal_tuning["acceptance_rate"],
        },
        "runtime_authority": "direct_local_academic_execution",
        "historical_identity_inputs_consumed": False,
        "migration_or_approval_artifacts_consumed": False,
        "fresh_outputs_verified": bool(require_fresh_outputs),
        "nonclaims": tuple(cfg.payload["nonclaims"]),
    }
    return _with_artifact_hash(report)


def validate_phase7_inputs(
    config: DeterministicLGSSMPhase7Config,
) -> Mapping[str, Any]:
    """Dispatch historical V1/V2 or direct fresh-run V3 preflight."""

    schema = config.payload.get("schema")
    if schema == PHASE7_CONFIG_SCHEMA:
        return validate_phase7_v1_inputs(config)
    if schema == PHASE7_CONFIG_SCHEMA_V2:
        return validate_phase7_v2_inputs(config)
    if schema == PHASE7_CONFIG_SCHEMA_V3:
        return validate_phase7_v3_inputs(config)
    raise ValueError("unsupported Phase 7 config schema")


def generate_phase3_candidate_identity_evidence(
    config: DeterministicLGSSMPhase7Config,
    *,
    private_sidecar_path: str | Path | None = None,
    input_manifest_path: str | Path | None = None,
    public_record_path: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Persist candidate identities, then preserve the unchanged legacy veto.

    This is an opt-in Phase 3 evidence lane. It does not replace or weaken
    ``validate_phase7_inputs`` and it never launches an HMC transition.
    """

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DeterministicLGSSMPhase7Error("CUDA_VISIBLE_DEVICES=-1 is required")

    from bayesfilter.inference.hmc_identity import (
        FrozenHMCExecutionContractV1,
        FrozenHMCTransitionIdentityV1,
        SelectionProvenanceIdentityV1,
    )
    from bayesfilter.inference.hmc_identity_integration import (
        LegacyValidatorResultV1,
        PHASE3_CANDIDATE_CHECK_KEYS,
        PHASE3_LEGACY_VETO_CODE,
        assert_public_validation_redacted,
        build_input_integrity_manifest,
        build_output_integrity_manifest,
        build_private_identity_sidecar,
        build_public_validation_record,
        parse_input_integrity_manifest,
        parse_output_integrity_manifest,
        parse_private_identity_sidecar,
        parse_public_validation_record,
        public_record_matches_private_sidecar,
        snapshot_governed_inputs,
        verify_input_integrity_manifest,
        verify_output_integrity_manifest,
        write_private_identity_sidecar,
    )

    cfg = config
    cfg.validate()
    if cfg.payload.get("schema") != PHASE7_CONFIG_SCHEMA:
        raise ValueError("Phase 3 evidence requires the historical V1 config")
    source_paths = phase7_governed_source_paths(cfg)
    governed_paths = tuple(source_paths[name] for name in (
        "fixture",
        "xla_compile",
        "geometry",
        "mass",
        "kernel",
        "private_replay",
        "source_tuning_config",
        "historical_v1_config",
        "source_contract",
    ))
    pre_snapshot = snapshot_governed_inputs(governed_paths)
    live = build_phase7_live_identity_bundle(cfg)
    fixture = live.replay_bundle.fixture
    kernel = live.replay_bundle.kernel
    private_replay = live.replay_bundle.private_replay
    tuning_payload = private_replay["tuning_payload"]
    replay = live.replay_bundle.replay
    transition = live.transition
    serious_execution = live.serious_execution
    smoke_execution = live.smoke_execution
    provenance = live.provenance
    runtime_versions = {
        "tensorflow_version": serious_execution.tensorflow_version,
        "tfp_version": serious_execution.tfp_version,
        "python_version": serious_execution.python_version,
    }

    # Reconstruct from the serialized typed payloads before consulting legacy gates.
    candidate_checks = {
        "transition_reconstructed": (
            FrozenHMCTransitionIdentityV1.from_payload(transition.payload())
            == transition
        ),
        "serious_execution_reconstructed": (
            FrozenHMCExecutionContractV1.from_payload(serious_execution.payload())
            == serious_execution
        ),
        "smoke_execution_reconstructed": (
            FrozenHMCExecutionContractV1.from_payload(smoke_execution.payload())
            == smoke_execution
        ),
        "selection_provenance_reconstructed": (
            SelectionProvenanceIdentityV1.from_payload(provenance.payload())
            == provenance
        ),
        "private_sidecar_round_trip": False,
        "public_private_hashes_match": False,
        "governed_inputs_unchanged": False,
        "public_redaction_passed": False,
    }
    if not all(candidate_checks[name] for name in PHASE3_CANDIDATE_CHECK_KEYS[:4]):
        raise DeterministicLGSSMPhase7Error("candidate identity reconstruction failed")

    try:
        validate_phase7_v1_inputs(cfg)
    except DeterministicLGSSMPhase7Error as error:
        if str(error) != "public final kernel hash mismatch":
            raise DeterministicLGSSMPhase7Error(
                f"unexpected legacy validator veto: {error}"
            ) from error
        legacy_error = error
    else:
        raise DeterministicLGSSMPhase7Error(
            "legacy validator unexpectedly passed during Phase 3"
        )
    legacy_result = LegacyValidatorResultV1(
        passed=False,
        exception_type=type(legacy_error).__name__,
        message=str(legacy_error),
        veto_code=PHASE3_LEGACY_VETO_CODE,
        remains_binding=True,
    )

    private_root = cfg.artifact_root / "private_diagnostics"
    sidecar_path = Path(private_sidecar_path or private_root / PHASE3_PRIVATE_SIDECAR_NAME)
    input_path = Path(input_manifest_path or private_root / PHASE3_INPUT_MANIFEST_NAME)
    public_path = Path(
        public_record_path or PHASE3_PUBLIC_ARTIFACT_ROOT / PHASE3_PUBLIC_RECORD_NAME
    )
    output_path = Path(
        output_manifest_path or PHASE3_PUBLIC_ARTIFACT_ROOT / PHASE3_OUTPUT_MANIFEST_NAME
    )
    if len({path.resolve() for path in (sidecar_path, input_path, public_path, output_path)}) != 4:
        raise ValueError("Phase 3 evidence output paths must be distinct")
    if any(path.resolve() in {item.resolve() for item in governed_paths} for path in (
        sidecar_path,
        input_path,
        public_path,
        output_path,
    )):
        raise ValueError("Phase 3 outputs must not overwrite governed inputs")

    public_legacy_reference = kernel.get("private_replay_reference")
    if not isinstance(public_legacy_reference, Mapping):
        raise DeterministicLGSSMPhase7Error("public private-replay reference is missing")
    bounded_legacy_reference = {
        name: public_legacy_reference.get(name)
        for name in ("artifact_hash", "file_sha256", "byte_count")
    }
    sidecar = build_private_identity_sidecar(
        transition=transition,
        serious_execution=serious_execution,
        smoke_execution=smoke_execution,
        selection_provenance=provenance,
        complete_tuning_payload=tuning_payload,
        legacy_private_replay_payload=private_replay,
        legacy_private_replay_path=source_paths["private_replay"],
        legacy_private_replay_reference=bounded_legacy_reference,
        replay=replay,
        legacy_validator_result=legacy_result,
    )
    restored_sidecar = write_private_identity_sidecar(sidecar_path, sidecar)
    parse_private_identity_sidecar(restored_sidecar)
    candidate_checks["private_sidecar_round_trip"] = True

    post_snapshot = snapshot_governed_inputs(governed_paths)
    input_manifest = build_input_integrity_manifest(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
    )
    parse_input_integrity_manifest(input_manifest)
    atomic_write_json(input_path, input_manifest)
    restored_input_manifest = _read_json(input_path)
    verify_input_integrity_manifest(restored_input_manifest)
    candidate_checks["governed_inputs_unchanged"] = True

    preliminary_checks = {
        **candidate_checks,
        "public_private_hashes_match": True,
        "public_redaction_passed": True,
    }
    public_record = build_public_validation_record(
        sidecar_payload=restored_sidecar,
        sidecar_path=sidecar_path,
        input_integrity_manifest=restored_input_manifest,
        legacy_private_replay_reference=bounded_legacy_reference,
        candidate_checks=preliminary_checks,
    )
    assert_public_validation_redacted(
        public_record,
        forbidden_values=(
            str(private_replay["target_scope"]),
            replay.contract["base_adapter_signature"],
            replay.contract["phase4_hmc_adapter_signature"],
            replay.contract["final_hmc_adapter_signature"],
            replay.contract["geometry_mass_artifact_signature"],
            replay.contract["adapted_mass_artifact_signature"],
            str(replay.final_kernel_payload["step_size"]),
            *runtime_versions.values(),
        ),
    )
    candidate_checks["public_redaction_passed"] = True
    candidate_checks["public_private_hashes_match"] = (
        public_record_matches_private_sidecar(
            public_record=public_record,
            sidecar_payload=restored_sidecar,
            sidecar_path=sidecar_path,
            input_integrity_manifest=restored_input_manifest,
        )
    )
    if not all(candidate_checks.values()):
        raise DeterministicLGSSMPhase7Error("candidate Phase 3 evidence checks failed")
    final_public_record = build_public_validation_record(
        sidecar_payload=restored_sidecar,
        sidecar_path=sidecar_path,
        input_integrity_manifest=restored_input_manifest,
        legacy_private_replay_reference=bounded_legacy_reference,
        candidate_checks=candidate_checks,
    )
    parse_public_validation_record(final_public_record)
    atomic_write_json(public_path, final_public_record)
    restored_public_record = _read_json(public_path)
    parse_public_validation_record(restored_public_record)

    output_manifest = build_output_integrity_manifest(
        sidecar_path=sidecar_path,
        input_manifest_path=input_path,
        public_record_path=public_path,
    )
    parse_output_integrity_manifest(output_manifest)
    atomic_write_json(output_path, output_manifest)
    restored_output_manifest = _read_json(output_path)
    verify_output_integrity_manifest(
        restored_output_manifest,
        sidecar_path=sidecar_path,
        input_manifest_path=input_path,
        public_record_path=public_path,
    )
    raise legacy_error


def map_final_hmc_samples_to_raw(adapter: Any, samples: Any) -> Any:
    """Apply exactly two frozen mass transforms and reach the LGSSM target."""

    final_base = getattr(adapter, "base_adapter", None)
    if final_base is None or not callable(getattr(adapter, "latent_to_position", None)):
        raise DeterministicLGSSMPhase7Error("final replay adapter transform is missing")
    phase4 = getattr(final_base, "base_adapter", None)
    if phase4 is None or not callable(getattr(final_base, "latent_to_position", None)):
        raise DeterministicLGSSMPhase7Error("Phase 4 replay adapter transform is missing")
    if getattr(phase4, "base_adapter", None) is not None:
        raise DeterministicLGSSMPhase7Error("unexpected replay transform depth")
    if phase4.__class__.__name__ != "DeterministicLGSSMPosteriorAdapter":
        raise DeterministicLGSSMPhase7Error("terminal replay adapter is not the LGSSM target")
    return final_base.latent_to_position(adapter.latent_to_position(samples))


def _validate_smoke_launch_context(
    cfg: DeterministicLGSSMPhase7Config,
    *,
    smoke: bool,
    context: Any,
    output_override: Path | None,
    progress_override: Path | None,
    private_samples_override: Path | None,
) -> None:
    from bayesfilter.inference.hmc_smoke_authority import (
        HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1,
        HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1,
        HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_SCHEMA_V1,
        HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1,
        parse_launch_claim,
        parse_smoke_authority,
        parse_smoke_authority_proposal,
        parse_smoke_authority_proposal_manifest,
        verify_prepared_smoke_launch_context,
    )

    if cfg.payload.get("schema") != PHASE7_CONFIG_SCHEMA_V2:
        raise DeterministicLGSSMPhase7Error("smoke authority requires the V2 config")
    if smoke is not True:
        raise DeterministicLGSSMPhase7Error("smoke authority cannot run serious mode")
    if any(
        value is not None
        for value in (output_override, progress_override, private_samples_override)
    ):
        raise DeterministicLGSSMPhase7Error(
            "smoke authority forbids caller output overrides"
        )
    try:
        verify_prepared_smoke_launch_context(context, consume=True)
    except (AttributeError, TypeError, ValueError) as error:
        raise DeterministicLGSSMPhase7Error(
            "smoke launch context was not prepared by the verified launcher"
        ) from error
    if context.output_session is None:
        raise DeterministicLGSSMPhase7Error("secure smoke output session is missing")
    context.output_session.validate_for_runtime()
    if context.config.path.resolve() != cfg.path.resolve() or context.config.hash != cfg.hash:
        raise DeterministicLGSSMPhase7Error("smoke launch config mismatch")
    parse_smoke_authority_proposal(context.proposal)
    parse_smoke_authority_proposal_manifest(context.proposal_manifest)
    parse_smoke_authority(context.authority)
    parse_launch_claim(context.claim)
    if context.proposal.get("schema") != HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_SCHEMA_V1:
        raise DeterministicLGSSMPhase7Error("smoke proposal schema mismatch")
    if context.proposal_manifest.get("schema") != (
        HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1
    ) or context.authority.get("schema") != HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1:
        raise DeterministicLGSSMPhase7Error("smoke authority evidence mismatch")
    if context.claim.get("schema") != HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1:
        raise DeterministicLGSSMPhase7Error("smoke claim schema mismatch")
    if context.claim["authority_artifact_hash"] != context.authority["artifact_hash"]:
        raise DeterministicLGSSMPhase7Error("smoke claim authority mismatch")
    if context.claim["proposal_manifest_artifact_hash"] != (
        context.proposal_manifest["artifact_hash"]
    ):
        raise DeterministicLGSSMPhase7Error("smoke claim manifest mismatch")
    if tuple(context.command) != tuple(context.claim["command"]):
        raise DeterministicLGSSMPhase7Error("smoke claim command mismatch")
    for name, path in context.paths.items():
        if path != ROOT / context.proposal["paths"][name]:
            raise DeterministicLGSSMPhase7Error("smoke context output path mismatch")
    from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash

    if canonical_artifact_payload_hash(
        context.output_session.read_json("claim_path")
    ) != (
        canonical_artifact_payload_hash(context.claim)
    ):
        raise DeterministicLGSSMPhase7Error("durable smoke claim bytes mismatch")
    if context.preflight.get("runtime_authority") is not False or (
        context.preflight.get("runtime_executed") is not False
    ):
        raise DeterministicLGSSMPhase7Error("smoke context preflight boundary mismatch")


def _validate_serious_launch_context(
    cfg: DeterministicLGSSMPhase7Config,
    *,
    smoke: bool,
    context: Any,
    output_override: Path | None,
    progress_override: Path | None,
    private_samples_override: Path | None,
) -> None:
    from bayesfilter.inference.hmc_serious_authority import (
        verify_prepared_serious_launch_context,
    )

    if cfg.payload.get("schema") != PHASE7_CONFIG_SCHEMA_V2:
        raise DeterministicLGSSMPhase7Error(
            "serious authority requires the V2 config"
        )
    if smoke is not False:
        raise DeterministicLGSSMPhase7Error(
            "serious authority cannot run smoke mode"
        )
    if any(
        value is not None
        for value in (output_override, progress_override, private_samples_override)
    ):
        raise DeterministicLGSSMPhase7Error(
            "serious authority forbids caller output overrides"
        )
    try:
        verify_prepared_serious_launch_context(context)
    except (AttributeError, TypeError, ValueError) as error:
        raise DeterministicLGSSMPhase7Error(
            "serious launch context was not prepared by the verified launcher"
        ) from error
    if context.output_session is None:
        raise DeterministicLGSSMPhase7Error(
            "secure serious output session is missing"
        )
    context.output_session.validate_for_runtime()
    if context.config.path.resolve() != cfg.path.resolve() or (
        context.config.hash != cfg.hash
    ):
        raise DeterministicLGSSMPhase7Error("serious launch config mismatch")
    if context.preflight.get("runtime_authority") is not False or (
        context.preflight.get("runtime_executed") is not False
    ):
        raise DeterministicLGSSMPhase7Error(
            "serious context preflight boundary mismatch"
        )
    from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
    from bayesfilter.inference.hmc_identity_adoption import (
        parse_phase5_preflight_report,
    )
    from bayesfilter.inference.hmc_serious_authority import (
        HISTORICAL_ARCHIVE_MANIFEST_PATH,
        parse_serious_authority,
        parse_serious_authority_proposal,
        parse_serious_authority_proposal_manifest,
        parse_serious_launch_claim,
    )

    parse_serious_authority_proposal(context.proposal)
    parse_serious_authority_proposal_manifest(context.proposal_manifest)
    parse_serious_authority(context.authority)
    parse_serious_launch_claim(context.claim)
    parse_phase5_preflight_report(context.preflight)
    if (
        context.proposal_manifest["proposal_reference"]["embedded_artifact_hash"]
        != context.proposal["artifact_hash"]
        or context.proposal_manifest["proposal_reference"]["canonical_payload_hash"]
        != canonical_artifact_payload_hash(context.proposal)
    ):
        raise DeterministicLGSSMPhase7Error(
            "serious proposal/manifest context mismatch"
        )
    if (
        context.authority["proposal_manifest_reference"]["embedded_artifact_hash"]
        != context.proposal_manifest["artifact_hash"]
        or context.claim["authority_artifact_hash"]
        != context.authority["artifact_hash"]
        or context.claim["proposal_manifest_artifact_hash"]
        != context.proposal_manifest["artifact_hash"]
        or context.claim["historical_archive_manifest_artifact_hash"]
        != context.proposal["historical_archive_manifest_reference"][
            "embedded_artifact_hash"
        ]
    ):
        raise DeterministicLGSSMPhase7Error(
            "serious authority/claim context mismatch"
        )
    if tuple(context.command) != tuple(context.claim["command"]):
        raise DeterministicLGSSMPhase7Error("serious claim command mismatch")
    for name, path in context.paths.items():
        if path != ROOT / context.proposal["paths"][name]:
            raise DeterministicLGSSMPhase7Error(
                "serious context output path mismatch"
            )
    durable_claim = context.output_session.read_json("claim_path")
    if canonical_artifact_payload_hash(durable_claim) != (
        canonical_artifact_payload_hash(context.claim)
    ):
        raise DeterministicLGSSMPhase7Error("durable serious claim bytes mismatch")
    archive_manifest = context.proposal["historical_archive_manifest_reference"]
    if archive_manifest["resolved_path_sha256"] != __import__("hashlib").sha256(
        str(HISTORICAL_ARCHIVE_MANIFEST_PATH.resolve()).encode()
    ).hexdigest():
        raise DeterministicLGSSMPhase7Error(
            "serious archive manifest path mismatch"
        )
    verify_prepared_serious_launch_context(context, consume=True)


def _validate_academic_launch_context(
    cfg: DeterministicLGSSMPhase7Config,
    *,
    smoke: bool,
    context: Any,
    output_override: Path | None,
    progress_override: Path | None,
    private_samples_override: Path | None,
) -> None:
    from bayesfilter.inference.hmc_academic_campaign import (
        AcademicCampaignError,
        validate_academic_launch_context,
    )

    if smoke:
        raise DeterministicLGSSMPhase7Error(
            "academic campaign context cannot run smoke mode"
        )
    if any(
        item is not None
        for item in (output_override, progress_override, private_samples_override)
    ):
        raise DeterministicLGSSMPhase7Error(
            "academic campaign context forbids caller output overrides"
        )
    try:
        validate_academic_launch_context(context, config=cfg)
    except AcademicCampaignError as error:
        raise DeterministicLGSSMPhase7Error(str(error)) from error


def _verify_child_live_identity(
    request: Mapping[str, Any], live: Phase7LiveReplayBundle
) -> Mapping[str, Any]:
    if "v3_config_payload" in request:
        from bayesfilter.inference.hmc_identity import FrozenHMCTransitionIdentityV1

        config_payload = request["v3_config_payload"]
        _validate_phase7_v3_config_payload(config_payload)
        expected = request.get("expected_transition_identity_hash")
        if expected != config_payload["expected_identities"][
            "transition_identity_hash"
        ]:
            raise DeterministicLGSSMPhase7Error(
                "child transition identity is not the V3 transition"
            )
        snapshots = live.governed_source_snapshots
        if tuple(snapshots) != _PHASE7_V3_SOURCE_KEYS:
            raise DeterministicLGSSMPhase7Error(
                "child V3 governed snapshots are incomplete"
            )
        requested_paths = request.get("governed_source_paths")
        if not isinstance(requested_paths, Mapping) or tuple(
            requested_paths
        ) != _PHASE7_V3_SOURCE_KEYS:
            raise DeterministicLGSSMPhase7Error(
                "child V3 governed source path inventory is incomplete"
            )
        references = config_payload["governed_source_references"]
        for name in _PHASE7_V3_SOURCE_KEYS:
            snapshot_path = Path(str(snapshots[name]["path"]))
            requested_path = Path(str(requested_paths[name]))
            if snapshot_path.resolve() != requested_path.resolve() or (
                snapshot_path != requested_path
            ):
                raise DeterministicLGSSMPhase7Error(
                    f"child V3 governed source path mismatch: {name}"
                )
            _verify_v3_source_reference_snapshot(
                references[name], snapshot=snapshots[name], name=name
            )
        transition = FrozenHMCTransitionIdentityV1.from_replay(live.replay)
        if transition.identity_hash != expected:
            raise DeterministicLGSSMPhase7Error(
                "child V3 transition identity mismatch before compile"
            )
        if str(live.kernel["target_scope"]) != str(request["target_scope"]):
            raise DeterministicLGSSMPhase7Error("child V3 target scope mismatch")
        return {
            "child_source_references_verified": True,
            "child_transition_identity_verified": True,
            "child_transition_identity_hash": transition.identity_hash,
        }
    if "v2_config_payload" not in request:
        if request.get("smoke") is True or request.get(
            "secure_source_verification"
        ) is True:
            raise DeterministicLGSSMPhase7Error(
                "secure child request is missing V2 identity evidence"
            )
        return {
            "child_source_references_verified": True,
            "child_transition_identity_verified": True,
        }
    from bayesfilter.inference.hmc_identity import FrozenHMCTransitionIdentityV1
    from bayesfilter.inference.hmc_identity_adoption import (
        GOVERNED_SOURCE_KEYS,
        parse_phase7_v2_config,
    )
    from bayesfilter.inference.hmc_smoke_authority import (
        verify_artifact_reference_snapshot,
    )

    config_payload = request["v2_config_payload"]
    parse_phase7_v2_config(config_payload)
    expected = request["expected_transition_identity_hash"]
    if expected != config_payload["adopted_identities"][
        "transition_identity_hash"
    ]:
        raise DeterministicLGSSMPhase7Error(
            "child transition identity is not the adopted V2 transition"
        )
    snapshots = live.governed_source_snapshots
    if tuple(sorted(snapshots)) != tuple(sorted(GOVERNED_SOURCE_KEYS)):
        raise DeterministicLGSSMPhase7Error("child governed snapshots are incomplete")
    requested_paths = request.get("governed_source_paths")
    if not isinstance(requested_paths, Mapping) or tuple(
        sorted(requested_paths)
    ) != tuple(sorted(GOVERNED_SOURCE_KEYS)):
        raise DeterministicLGSSMPhase7Error(
            "child governed source path inventory is incomplete"
        )
    for name in GOVERNED_SOURCE_KEYS:
        snapshot_path = Path(str(snapshots[name]["path"]))
        requested_path = Path(str(requested_paths[name]))
        if snapshot_path.resolve() != requested_path.resolve() or (
            snapshot_path != requested_path
        ):
            raise DeterministicLGSSMPhase7Error(
                f"child governed source path mismatch: {name}"
            )
        verify_artifact_reference_snapshot(
            config_payload["governed_source_references"][name],
            snapshot=snapshots[name],
        )
    transition = FrozenHMCTransitionIdentityV1.from_replay(live.replay)
    if transition.identity_hash != expected:
        raise DeterministicLGSSMPhase7Error(
            "child transition identity mismatch before compile"
        )
    if str(live.kernel["target_scope"]) != str(request["target_scope"]):
        raise DeterministicLGSSMPhase7Error("child target scope mismatch")
    return {
        "child_source_references_verified": True,
        "child_transition_identity_verified": True,
        "child_transition_identity_hash": transition.identity_hash,
    }


def _verify_child_implementation_identity(
    request: Mapping[str, Any],
    *,
    require_runtime_imports: bool = False,
) -> Mapping[str, Any]:
    references = request.get("implementation_references")
    if references is None:
        if request.get("secure_source_verification") is True and (
            request.get("action") == "initialize"
        ):
            raise DeterministicLGSSMPhase7Error(
                "secure child request is missing implementation evidence"
            )
        return _cached_child_implementation_identity(request)
    from bayesfilter.inference.hmc_smoke_authority import (
        implementation_source_bundle_hash,
        verify_implementation_reference_inventory,
    )

    implementation_paths = {
        role: (
            Path(sys.executable).resolve()
            if role == "python_executable"
            else ROOT / role.removeprefix("repository_file:")
        )
        for role in references
    }
    verify_implementation_reference_inventory(
        references,
        python_executable=sys.executable,
        implementation_paths=implementation_paths,
    )
    import builtins

    bootstrap = getattr(builtins, "_BAYESFILTER_PHASE6_SOURCE_BOOTSTRAP", None)
    expected_bundle_hash = request.get("implementation_source_bundle_hash")
    if not isinstance(bootstrap, Mapping) or (
        bootstrap.get("bundle_hash") != expected_bundle_hash
    ):
        raise DeterministicLGSSMPhase7Error(
            "smoke child source bootstrap evidence is missing"
        )
    source_bundle = request.get("implementation_source_bundle")
    if not isinstance(source_bundle, Mapping) or (
        implementation_source_bundle_hash(source_bundle) != expected_bundle_hash
    ):
        raise DeterministicLGSSMPhase7Error(
            "smoke child loaded source bundle mismatch"
        )
    _verify_loaded_child_modules(
        references,
        expected_bundle_hash=expected_bundle_hash,
        loaded_modules=sys.modules,
        require_runtime_imports=require_runtime_imports,
    )
    return {
        "child_implementation_references_verified": True,
        "child_loaded_source_bytes_verified": True,
        "child_implementation_source_bundle_hash": expected_bundle_hash,
    }


def _verify_loaded_child_modules(
    references: Mapping[str, Mapping[str, Any]],
    *,
    expected_bundle_hash: str,
    loaded_modules: Mapping[str, Any],
    require_runtime_imports: bool,
) -> None:
    """Reject imported project modules whose executed bytes were not retained."""

    synthetic_namespace_names = ("docs", "docs.benchmarks")
    if require_runtime_imports:
        for name in synthetic_namespace_names:
            module = loaded_modules.get(name)
            if (
                module is None
                or getattr(module, "__phase6_synthetic_namespace__", None) != name
                or getattr(module, "__phase6_source_bundle_hash__", None)
                != expected_bundle_hash
            ):
                raise DeterministicLGSSMPhase7Error(
                    "smoke child benchmark namespace evidence is missing"
                )
    expected_module_roles = {
        role
        for role in references
        if role.startswith("repository_file:bayesfilter/")
        or role
        == (
            "repository_file:docs/benchmarks/"
            "run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
        )
    }
    loaded_roles: set[str] = set()
    for name, module in tuple(loaded_modules.items()):
        if name in synthetic_namespace_names:
            continue
        is_bayesfilter = name == "bayesfilter" or name.startswith("bayesfilter.")
        is_benchmark_driver = name == (
            "docs.benchmarks."
            "run_multidim_lgssm_serious_hmc_tuning_2026_07_09"
        )
        is_docs_module = name == "docs" or name.startswith("docs.")
        if is_docs_module and not is_benchmark_driver:
            raise DeterministicLGSSMPhase7Error(
                "smoke child imported unverified docs code"
            )
        if not is_bayesfilter and not is_benchmark_driver:
            continue
        role = getattr(module, "__phase6_source_role__", None)
        source_hash = getattr(module, "__phase6_source_sha256__", None)
        bundle_hash = getattr(module, "__phase6_source_bundle_hash__", None)
        if role not in expected_module_roles or (
            source_hash != references[role]["file_sha256"]
        ) or bundle_hash != expected_bundle_hash:
            raise DeterministicLGSSMPhase7Error(
                "smoke child imported unverified BayesFilter code"
            )
        loaded_roles.add(role)
    controller_role = (
        "repository_file:bayesfilter/testing/"
        "deterministic_lgssm_hmc_phase7_tf.py"
    )
    if controller_role not in loaded_roles:
        raise DeterministicLGSSMPhase7Error(
            "smoke child controller loaded-byte evidence is missing"
        )
    benchmark_role = (
        "repository_file:docs/benchmarks/"
        "run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
    )
    if require_runtime_imports and benchmark_role not in loaded_roles:
        raise DeterministicLGSSMPhase7Error(
            "smoke child benchmark loaded-byte evidence is missing"
        )


def _cached_child_implementation_identity(
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    cached = _WORKER_CACHE.get("child_identity")
    if request.get("secure_source_verification") is not True:
        inventory_hash = request.get("launch_implementation_inventory_hash")
        if inventory_hash is not None:
            return {
                "child_implementation_references_verified": False,
                "child_implementation_verification_status": (
                    "launch_inventory_only_not_child_byte_verified"
                ),
                "launch_implementation_inventory_hash": inventory_hash,
            }
        return {"child_implementation_references_verified": True}
    if not isinstance(cached, Mapping) or (
        cached.get("child_implementation_references_verified") is not True
        or cached.get("child_loaded_source_bytes_verified") is not True
    ):
        raise DeterministicLGSSMPhase7Error(
            "smoke child cached implementation evidence is missing"
        )
    seal = _parse_secure_worker_cache_seal(request.get("worker_cache_seal"))
    if cached.get("child_implementation_source_bundle_hash") != seal[
        "implementation_source_bundle_hash"
    ] or cached.get("child_transition_identity_hash") != seal[
        "transition_identity_hash"
    ]:
        raise DeterministicLGSSMPhase7Error(
            "secure child cached identity differs from launch seal"
        )
    return dict(cached)


def _secure_worker_cache_seal(
    cfg: DeterministicLGSSMPhase7Config,
    *,
    worker_index: int,
    smoke: bool,
    target_scope: str,
    launch_context: Any,
) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
    from bayesfilter.inference.hmc_smoke_authority import (
        implementation_source_bundle_hash,
    )

    payload = {
        "schema": SECURE_WORKER_CACHE_SEAL_SCHEMA,
        "config_hash": cfg.hash,
        "authority_kind": (
            "phase7_serious"
            if _is_serious_launch_context(launch_context)
            else "phase6_smoke"
        ),
        "authority_artifact_hash": launch_context.authority["artifact_hash"],
        "claim_artifact_hash": launch_context.claim["artifact_hash"],
        "proposal_manifest_artifact_hash": launch_context.proposal_manifest[
            "artifact_hash"
        ],
        "worker_index": worker_index,
        "smoke": smoke,
        "target_scope": target_scope,
        "transition_identity_hash": _expected_transition_identity_hash(cfg),
        "implementation_source_bundle_hash": implementation_source_bundle_hash(
            launch_context.implementation_source_bundle
        ),
        "chains_per_worker": cfg.chains_per_worker,
        "total_chain_count": cfg.chain_count,
    }
    return {
        **payload,
        "artifact_hash": canonical_artifact_payload_hash(payload),
    }


def _parse_secure_worker_cache_seal(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash

    fields = {
        "schema",
        "config_hash",
        "authority_kind",
        "authority_artifact_hash",
        "claim_artifact_hash",
        "proposal_manifest_artifact_hash",
        "worker_index",
        "smoke",
        "target_scope",
        "transition_identity_hash",
        "implementation_source_bundle_hash",
        "chains_per_worker",
        "total_chain_count",
        "artifact_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal fields mismatch"
        )
    if payload.get("schema") != SECURE_WORKER_CACHE_SEAL_SCHEMA or (
        payload.get("authority_kind") not in {"phase6_smoke", "phase7_serious"}
    ):
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal identity mismatch"
        )
    if type(payload.get("worker_index")) is not int or payload["worker_index"] < 0:
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal worker index mismatch"
        )
    if type(payload.get("smoke")) is not bool or (
        payload["smoke"] is not (payload["authority_kind"] == "phase6_smoke")
    ):
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal mode mismatch"
        )
    if payload.get("chains_per_worker") != 2 or payload.get(
        "total_chain_count"
    ) != 4:
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal topology mismatch"
        )
    for name in (
        "config_hash",
        "authority_artifact_hash",
        "claim_artifact_hash",
        "proposal_manifest_artifact_hash",
        "transition_identity_hash",
        "implementation_source_bundle_hash",
        "artifact_hash",
    ):
        value = payload.get(name)
        if not isinstance(value, str) or not value.startswith("sha256:") or (
            len(value) != 71
        ) or any(char not in "0123456789abcdef" for char in value[7:]):
            raise DeterministicLGSSMPhase7Error(
                f"secure worker cache seal {name} mismatch"
            )
    if not isinstance(payload.get("target_scope"), str) or not payload[
        "target_scope"
    ]:
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal target scope mismatch"
        )
    expected = canonical_artifact_payload_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    if payload["artifact_hash"] != expected:
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache seal hash mismatch"
        )
    return payload


def _verify_secure_worker_cache_seal(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if request.get("secure_source_verification") is not True:
        return None
    seal = _parse_secure_worker_cache_seal(request.get("worker_cache_seal"))
    if (
        seal["worker_index"] != request.get("worker_index")
        or seal["smoke"] is not request.get("smoke")
        or seal["target_scope"] != request.get("target_scope")
        or seal["chains_per_worker"] != request.get("chains_per_worker")
        or seal["total_chain_count"] != request.get("total_chain_count")
    ):
        raise DeterministicLGSSMPhase7Error(
            "secure worker request/cache seal mismatch"
        )
    if request.get("action") == "initialize":
        config = request.get("v2_config_payload") or request.get(
            "v3_config_payload"
        )
        if not isinstance(config, Mapping) or seal["config_hash"] != (
            "sha256:" + stable_config_hash(config)
        ) or seal["transition_identity_hash"] != request.get(
            "expected_transition_identity_hash"
        ) or seal["implementation_source_bundle_hash"] != request.get(
            "implementation_source_bundle_hash"
        ):
            raise DeterministicLGSSMPhase7Error(
                "secure worker initialize/cache seal mismatch"
            )
    else:
        cached = _WORKER_CACHE.get("worker_cache_seal")
        if not isinstance(cached, Mapping) or dict(cached) != dict(seal):
            raise DeterministicLGSSMPhase7Error(
                "secure worker cached launch seal mismatch"
            )
    return seal


def _is_serious_launch_context(launch_context: Any | None) -> bool:
    return launch_context is not None and getattr(
        launch_context, "authority_kind", None
    ) == "phase7_serious"


def _is_academic_launch_context(launch_context: Any | None) -> bool:
    return launch_context is not None and getattr(
        launch_context, "context_kind", None
    ) == "phase7_academic_campaign"


def _progress_schema(launch_context: Any | None) -> str:
    if launch_context is None:
        return PHASE7_PROGRESS_SCHEMA
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import (
            HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1,
        )

        return HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import (
            ACADEMIC_PROGRESS_SCHEMA,
        )

        return ACADEMIC_PROGRESS_SCHEMA
    from bayesfilter.inference.hmc_smoke_authority import (
        HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1,
    )

    return HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1


def _result_schema(launch_context: Any | None) -> str:
    if launch_context is None:
        return PHASE7_RESULT_SCHEMA
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import (
            HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1,
        )

        return HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import ACADEMIC_RESULT_SCHEMA

        return ACADEMIC_RESULT_SCHEMA
    from bayesfilter.inference.hmc_smoke_authority import (
        HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1,
    )

    return HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1


def _failure_schema(launch_context: Any | None) -> str:
    if launch_context is None:
        return PHASE7_RESULT_SCHEMA
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import (
            HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1,
        )

        return HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import ACADEMIC_FAILURE_SCHEMA

        return ACADEMIC_FAILURE_SCHEMA
    from bayesfilter.inference.hmc_smoke_authority import (
        HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1,
    )

    return HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1


def _runtime_authority_links(launch_context: Any | None) -> Mapping[str, Any]:
    if launch_context is None:
        return {}
    if _is_serious_launch_context(launch_context):
        return {
            "serious_authority_artifact_hash": launch_context.authority[
                "artifact_hash"
            ],
            "serious_launch_claim_artifact_hash": launch_context.claim[
                "artifact_hash"
            ],
            "serious_proposal_manifest_artifact_hash": (
                launch_context.proposal_manifest["artifact_hash"]
            ),
            "preflight_before_runtime_artifact_hash": (
                launch_context.preflight["artifact_hash"]
            ),
        }
    if _is_academic_launch_context(launch_context):
        return {
            "campaign_id": launch_context.campaign_id,
            "attempt_number": launch_context.attempt_number,
            "run_manifest_artifact_hash": launch_context.manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": (
                launch_context.preflight["artifact_hash"]
            ),
        }
    return {
        "smoke_authority_artifact_hash": launch_context.authority["artifact_hash"],
        "smoke_launch_claim_artifact_hash": launch_context.claim["artifact_hash"],
        "smoke_proposal_manifest_artifact_hash": (
            launch_context.proposal_manifest["artifact_hash"]
        ),
        "preflight_before_runtime_artifact_hash": (
            launch_context.preflight["artifact_hash"]
        ),
    }


def _runtime_execution_state(
    launch_context: Any | None,
    *,
    workers_started: bool = True,
    hmc_transition_executed: bool = True,
) -> Mapping[str, Any]:
    if launch_context is None:
        return {}
    if _is_academic_launch_context(launch_context):
        return {
            "controller_entered": True,
            "workers_started": bool(workers_started),
            "hmc_transition_executed": bool(hmc_transition_executed),
            "serious_runtime_executed": bool(hmc_transition_executed),
            "neutra_executed": False,
        }
    if _is_serious_launch_context(launch_context):
        return {
            "serious_runtime_executed": True,
            "neutra_executed": False,
        }
    return {
        "serious_runtime_executed": False,
        "neutra_executed": False,
    }


def _preflight_result_evidence(
    preflight: Mapping[str, Any], launch_context: Any | None
) -> Mapping[str, Any]:
    if launch_context is None:
        return {"preflight": preflight}
    return {"preflight_before_runtime": preflight}


def _public_worker_metadata(
    metadata: Sequence[Mapping[str, Any] | None],
    launch_context: Any | None,
) -> Sequence[Mapping[str, Any] | None]:
    if launch_context is None:
        return metadata
    allowed = (
        "jit_compile",
        "use_xla",
        "compile_trace_count",
        "first_call_s",
        "warm_call_s",
        "tensorflow_version",
        "tfp_version",
        "python_version",
        "cuda_visible_devices",
        "thread_environment",
        "child_source_references_verified",
        "child_implementation_references_verified",
        "child_loaded_source_bytes_verified",
        "child_implementation_source_bundle_hash",
        "child_transition_identity_verified",
        "child_transition_identity_hash",
    )
    if _is_serious_launch_context(launch_context):
        allowed = (
            "worker_index",
            "pid",
            "child_worker_cache_seal_hash",
            *allowed,
        )
    elif _is_academic_launch_context(launch_context):
        allowed = (
            "worker_index",
            "pid",
            "child_implementation_verification_status",
            "launch_implementation_inventory_hash",
            *allowed,
        )
    return tuple(
        None
        if item is None
        else {name: item.get(name) for name in allowed}
        for item in metadata
    )


def _finalize_runtime_artifact(
    payload: Mapping[str, Any], launch_context: Any | None
) -> Mapping[str, Any]:
    if launch_context is None:
        return _with_artifact_hash(payload)
    from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash

    result = dict(payload)
    if "artifact_hash" in result:
        raise ValueError("runtime artifact hash must not be prepopulated")
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _finalize_progress_artifact(
    payload: Mapping[str, Any], launch_context: Any | None
) -> Mapping[str, Any]:
    if launch_context is None:
        return dict(payload)
    return _finalize_runtime_artifact(payload, launch_context)


def _pass_decision(launch_context: Any | None) -> str:
    if launch_context is None:
        return "PASS_PHASE7_TO_PHASE8_APPROVAL_BOUNDARY"
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import (
            SERIOUS_PASS_DECISION,
        )

        return SERIOUS_PASS_DECISION
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import (
            ACADEMIC_PASS_DECISION,
        )

        return ACADEMIC_PASS_DECISION
    from bayesfilter.inference.hmc_smoke_authority import SMOKE_PASS_DECISION

    return SMOKE_PASS_DECISION


def _block_decision(launch_context: Any | None) -> str:
    if launch_context is None:
        return "BLOCK_PHASE7"
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import (
            SERIOUS_BLOCK_DECISION,
        )

        return SERIOUS_BLOCK_DECISION
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import (
            ACADEMIC_BLOCK_DECISION,
        )

        return ACADEMIC_BLOCK_DECISION
    from bayesfilter.inference.hmc_smoke_authority import SMOKE_BLOCK_DECISION

    return SMOKE_BLOCK_DECISION


def _result_nonclaims(
    cfg: DeterministicLGSSMPhase7Config, launch_context: Any | None
) -> tuple[str, ...]:
    if launch_context is None:
        return tuple(cfg.payload["nonclaims"])
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import SERIOUS_NONCLAIMS

        return SERIOUS_NONCLAIMS
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import ACADEMIC_NONCLAIMS

        return ACADEMIC_NONCLAIMS
    from bayesfilter.inference.hmc_smoke_authority import SMOKE_NONCLAIMS

    return SMOKE_NONCLAIMS


def _failure_nonclaims(
    cfg: DeterministicLGSSMPhase7Config, launch_context: Any | None
) -> tuple[str, ...]:
    if launch_context is None:
        return tuple(cfg.payload["nonclaims"])
    if _is_serious_launch_context(launch_context):
        from bayesfilter.inference.hmc_serious_authority import (
            SERIOUS_FAILURE_NONCLAIMS,
        )

        return SERIOUS_FAILURE_NONCLAIMS
    if _is_academic_launch_context(launch_context):
        from bayesfilter.inference.hmc_academic_campaign import (
            ACADEMIC_FAILURE_NONCLAIMS,
        )

        return ACADEMIC_FAILURE_NONCLAIMS
    from bayesfilter.inference.hmc_smoke_authority import SMOKE_FAILURE_NONCLAIMS

    return SMOKE_FAILURE_NONCLAIMS


def _phase7_worker_command(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Spawn-safe worker command that initializes TensorFlow only in the child."""

    for key, value in request["worker_environment"].items():
        os.environ[str(key)] = str(value)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DeterministicLGSSMPhase7Error("worker GPU hiding failed")
    action = request.get("action")
    if action not in {"initialize", "burnin", "retained"}:
        raise DeterministicLGSSMPhase7Error("worker action is invalid")
    worker_cache_seal = _verify_secure_worker_cache_seal(request)
    child_implementation = _verify_child_implementation_identity(request)

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.inference import FixedSizeHMCChunkConfig, build_fixed_size_hmc_chunk_runner

    chains = int(request["chains_per_worker"])
    worker_index = int(request["worker_index"])
    if action == "initialize":
        if _WORKER_CACHE:
            raise DeterministicLGSSMPhase7Error("worker cache was already initialized")
        live = build_phase7_live_replay_from_paths(
            fixture_path=request["fixture_path"],
            mass_path=request["mass_path"],
            kernel_path=request["kernel_path"],
            private_replay_path=request["private_replay_path"],
            source_tuning_config_path=request["source_tuning_config_path"],
            xla_evidence_path=request["xla_evidence_path"],
            source_contract_path=request["source_contract_path"],
            historical_v1_config_path=request["historical_v1_config_path"],
        )
        child_implementation = _verify_child_implementation_identity(
            request,
            require_runtime_imports=True,
        )
        child_identity = {
            **child_implementation,
            **_verify_child_live_identity(request, live),
        }
        replay = live.replay
        final_kernel = replay.final_kernel_payload
        dimension = int(final_kernel["target_dimension"])
        offsets = np.linspace(-0.15, 0.15, int(request["total_chain_count"]))
        global_start = worker_index * chains
        pattern = 1.0 - 2.0 * (np.arange(dimension) % 2)
        initial_state = offsets[global_start : global_start + chains, None] * pattern[None, :]
        max_results = int(request["max_chunk_results"])
        chunk_config = FixedSizeHMCChunkConfig(
            max_results=max_results,
            num_burnin_steps=0,
            step_size=float(final_kernel["step_size"]),
            num_leapfrog_steps=int(final_kernel["num_leapfrog_steps"]),
            seed=tuple(int(item) for item in request["seed"]),
            use_xla=True,
            trace_policy="reduced",
            target_status_trace_policy="none",
            target_scope=str(request["target_scope"]),
            chain_execution_mode="tf_function",
        )
        _WORKER_CACHE.update(
            {
                "worker_index": worker_index,
                "replay": replay,
                "runner": build_fixed_size_hmc_chunk_runner(
                    replay.adapter,
                    initial_state,
                    chunk_config,
                ),
                "final_kernel": final_kernel,
                "dimension": dimension,
                "initial_state": initial_state,
                "max_results": max_results,
                "current_state": initial_state,
                "child_identity": child_identity,
                "worker_cache_seal": worker_cache_seal,
            }
        )
    elif not _WORKER_CACHE:
        raise DeterministicLGSSMPhase7Error("worker cache is not initialized")
    if int(_WORKER_CACHE["worker_index"]) != worker_index:
        raise DeterministicLGSSMPhase7Error("worker cache identity mismatch")
    replay = _WORKER_CACHE["replay"]
    runner = _WORKER_CACHE["runner"]
    final_kernel = _WORKER_CACHE["final_kernel"]
    dimension = int(_WORKER_CACHE["dimension"])
    current_state = np.asarray(_WORKER_CACHE["current_state"], dtype=float)
    if current_state.shape != (chains, dimension):
        raise DeterministicLGSSMPhase7Error("worker state shape mismatch")

    count = int(request["count"])
    if action == "initialize":
        # Compile one transition but do not advance the persisted chain state.
        active_results = 1
    else:
        active_results = count
    if active_results > int(_WORKER_CACHE["max_results"]):
        raise DeterministicLGSSMPhase7Error("worker chunk exceeds compiled maximum")
    result = runner.run(
        active_results=active_results,
        current_state=current_state,
        seed=tuple(int(item) for item in request["seed"]),
        step_size=float(final_kernel["step_size"]),
    )
    valid = tf.boolean_mask(result.samples, result.valid_mask)
    raw = map_final_hmc_samples_to_raw(replay.adapter, valid)
    diagnostics = result.diagnostics
    hard_vetoes: list[str] = []
    if int(diagnostics.get("nonfinite_valid_sample_count", -1)) != 0:
        hard_vetoes.append("nonfinite_samples")
    if int(diagnostics.get("log_accept_ratio_nonfinite_count", -1)) != 0:
        hard_vetoes.append("nonfinite_log_accept")
    if int(diagnostics.get("target_log_prob_nonfinite_count", -1)) != 0:
        hard_vetoes.append("nonfinite_target_log_prob")
    divergence_count = diagnostics.get("divergence_count")
    if divergence_count is not None and int(divergence_count) != 0:
        hard_vetoes.append("nonzero_divergence_count")
    metadata = dict(result.metadata)
    if metadata.get("jit_compile") is not True or metadata.get("use_xla") is not True:
        hard_vetoes.append("xla_jit_not_confirmed")
    if hard_vetoes:
        raise DeterministicLGSSMPhase7Error(f"worker hard vetoes: {hard_vetoes}")
    persisted_final_state = current_state if action == "initialize" else result.final_state.numpy()
    _WORKER_CACHE["current_state"] = np.asarray(persisted_final_state, dtype=float)
    raw_samples = np.empty((0, chains, dimension)) if action == "initialize" else raw.numpy()
    return {
        "schema": "bayesfilter.deterministic_lgssm_hmc_phase7_worker_result.v1",
        "action": action,
        "worker_index": worker_index,
        "pid": os.getpid(),
        "passed": True,
        "seed": tuple(int(item) for item in request["seed"]),
        "final_state": persisted_final_state,
        "raw_samples": raw_samples,
        "diagnostics": diagnostics,
        "metadata": {
            "jit_compile": metadata.get("jit_compile"),
            "use_xla": metadata.get("use_xla"),
            "compile_trace_count": metadata.get("compile_trace_count"),
            "first_call_s": metadata.get("first_call_s"),
            "warm_call_s": metadata.get("warm_call_s"),
            "tensorflow_version": tf.__version__,
            "tfp_version": tfp.__version__,
            "python_version": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_environment": dict(request["worker_environment"]),
            "target_scope": request["target_scope"],
            "adapter_signature": replay.contract["final_hmc_adapter_signature"],
            "child_worker_cache_seal_hash": (
                None
                if worker_cache_seal is None
                else worker_cache_seal["artifact_hash"]
            ),
            **dict(_WORKER_CACHE["child_identity"]),
        },
    }


def _runtime_counts(
    config: DeterministicLGSSMPhase7Config,
    *,
    smoke: bool,
) -> Mapping[str, int]:
    if smoke:
        return {
            "burnin_initial": 4,
            "burnin_extension": 4,
            "burnin_window": 4,
            "burnin_maximum": 4,
            "retained_initial": 8,
            "retained_extension": 8,
            "retained_maximum": 8,
        }
    burnin = config.payload["burnin"]
    retained = config.payload["retained"]
    return {
        "burnin_initial": int(burnin["initial_results_per_chain"]),
        "burnin_extension": int(burnin["extension_results_per_chain"]),
        "burnin_window": int(burnin["check_window_results_per_chain"]),
        "burnin_maximum": int(burnin["max_results_per_chain"]),
        "retained_initial": int(retained["initial_results_per_chain"]),
        "retained_extension": int(retained["extension_results_per_chain"]),
        "retained_maximum": int(retained["max_results_per_chain"]),
    }


def _aggregate_diagnostics(
    worker_samples: Sequence[np.ndarray | None],
    *,
    parameter_names: Sequence[str],
    cfg: DeterministicLGSSMPhase7Config,
    smoke: bool,
) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )

    if any(samples is None for samples in worker_samples):
        raise DeterministicLGSSMPhase7Error("worker samples are incomplete")
    combined = np.concatenate([np.asarray(item) for item in worker_samples], axis=1)
    configured = cfg.payload["diagnostics"]
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=float(configured["rhat_max"]),
        bulk_ess_min=float(configured["bulk_ess_min"]),
        tail_ess_min=float(configured["tail_ess_min"]),
    )
    payload = dict(
        rank_normalized_hmc_diagnostics(
            combined,
            parameter_names=parameter_names,
            thresholds=thresholds,
        )
    )
    if smoke:
        metric_fields = (
            "rank_normalized_split_rhat",
            "folded_rank_normalized_split_rhat",
            "rhat",
            "bulk_ess",
            "tail_ess",
            "lower_tail_ess",
            "upper_tail_ess",
        )
        rows = tuple(
            {
                **dict(row),
                "passed": all(
                    not isinstance(row.get(name), (bool, np.bool_))
                    and isinstance(row.get(name), (int, float, np.number))
                    and bool(np.isfinite(row[name]))
                    for name in metric_fields
                ),
            }
            for row in payload["parameter_diagnostics"]
        )
        payload["parameter_diagnostics"] = rows
        payload["passed"] = bool(
            payload["input_all_finite"] and payload["diagnostics_all_finite"]
            and all(row["passed"] for row in rows)
        )
        payload["nonclaims"] = (
            "finite-only smoke engineering diagnostic screen",
            "R-hat and ESS values are explanatory only",
            "no posterior recovery or HMC convergence claim",
            "no sampler superiority, production, or default readiness claim",
        )
        payload["smoke_gate"] = "finite_diagnostics_only_non_promoting"
    return payload


def _run_worker_round(
    executors: Sequence[concurrent.futures.ProcessPoolExecutor],
    cfg: DeterministicLGSSMPhase7Config,
    *,
    action: str,
    count: int,
    stage_index: int,
    check_index: int,
    root_seed: Sequence[int],
    worker_env: Mapping[str, str],
    smoke: bool,
    target_scope: str,
    expected_worker_pids: Sequence[int],
    start: float,
    wall_time_cap_seconds: float | None = None,
    launch_implementation_inventory_hash: str | None = None,
    secure_source_verification: bool = False,
    expected_worker_cache_seals: Sequence[Mapping[str, Any] | None] = (),
    on_transition_dispatched: Callable[[], None] | None = None,
) -> list[Mapping[str, Any]]:
    if secure_source_verification and len(expected_worker_cache_seals) != len(
        executors
    ):
        raise DeterministicLGSSMPhase7Error(
            "secure worker cache-seal inventory mismatch"
        )
    requests = [
        _worker_request(
                cfg,
                worker_index=index,
                action=action,
                count=count,
                seed=derive_worker_seed(
                    root_seed,
                    stage_index=stage_index,
                    check_index=check_index,
                    worker_index=index,
                ),
                state=None,
                worker_env=worker_env,
                smoke=smoke,
                target_scope=target_scope,
                launch_implementation_inventory_hash=(
                    launch_implementation_inventory_hash
                ),
                secure_source_verification=secure_source_verification,
                worker_cache_seal=(
                    expected_worker_cache_seals[index]
                    if secure_source_verification
                    else None
                ),
            )
        for index in range(len(executors))
    ]
    futures = []
    for executor, request in zip(executors, requests, strict=True):
        futures.append(executor.submit(
            _phase7_worker_command,
            request,
        ))
        if len(futures) == 1 and on_transition_dispatched is not None:
            on_transition_dispatched()
    responses = [
        future.result(
            timeout=_remaining_wall_time(
                cfg,
                start,
                cap_seconds=wall_time_cap_seconds,
            )
        )
        for future in futures
    ]
    for index, (request, response) in enumerate(
        zip(requests, responses, strict=True)
    ):
        seal = (
            expected_worker_cache_seals[index]
            if secure_source_verification
            else None
        )
        _assert_worker_response(
            response,
            action=action,
            expected_worker_index=index,
            expected_pid=expected_worker_pids[index],
            expected_seed=request["seed"],
            expected_target_scope=target_scope,
            expected_transition_identity_hash=(
                seal["transition_identity_hash"]
                if seal is not None
                else _expected_transition_identity_hash(cfg)
            ),
            expected_implementation_source_bundle_hash=(
                seal["implementation_source_bundle_hash"]
                if seal is not None
                else None
            ),
            expected_worker_cache_seal_hash=(
                seal["artifact_hash"] if seal is not None else None
            ),
            secure_source_verification=secure_source_verification,
            expected_launch_implementation_inventory_hash=(
                launch_implementation_inventory_hash
            ),
        )
    return responses


def _worker_request(
    cfg: DeterministicLGSSMPhase7Config,
    *,
    worker_index: int,
    action: str,
    count: int,
    seed: Sequence[int],
    state: np.ndarray | None,
    worker_env: Mapping[str, str],
    smoke: bool,
    target_scope: str | None = None,
    implementation_references: Mapping[str, Mapping[str, Any]] | None = None,
    implementation_source_bundle: Mapping[str, bytes] | None = None,
    launch_implementation_inventory_hash: str | None = None,
    secure_source_verification: bool | None = None,
    worker_cache_seal: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    schema = cfg.payload.get("schema")
    if schema in {PHASE7_CONFIG_SCHEMA_V2, PHASE7_CONFIG_SCHEMA_V3} and not target_scope:
        raise ValueError("typed worker request requires preflight target scope")
    governed_paths = phase7_governed_source_paths(cfg)
    secure_verification = (
        bool(smoke) and schema != PHASE7_CONFIG_SCHEMA_V3
        if secure_source_verification is None
        else bool(secure_source_verification)
    )
    request = {
        "worker_index": int(worker_index),
        "action": str(action),
        "count": int(count),
        "seed": tuple(int(item) for item in seed),
        "state": state,
        "worker_environment": dict(worker_env),
        "chains_per_worker": cfg.chains_per_worker,
        "total_chain_count": cfg.chain_count,
        "fixture_path": str(governed_paths["fixture"]),
        "mass_path": str(governed_paths["mass"]),
        "kernel_path": str(governed_paths["kernel"]),
        "private_replay_path": str(cfg.artifact_path("private_replay")),
        "source_tuning_config_path": str(governed_paths["source_tuning_config"]),
        "source_contract_path": str(governed_paths["source_contract"]),
        "historical_v1_config_path": (
            str(governed_paths["historical_v1_config"])
            if "historical_v1_config" in governed_paths
            else None
        ),
        "xla_evidence_path": str(governed_paths["xla_compile"]),
        "target_scope": str(target_scope) if target_scope else str(
            _read_json(governed_paths["kernel"])["target_scope"]
        ),
        "max_chunk_results": max(
            _runtime_counts(cfg, smoke=smoke)[key]
            for key in (
                "burnin_initial",
                "burnin_extension",
                "retained_initial",
                "retained_extension",
            )
        ),
        "smoke": bool(smoke),
        "secure_source_verification": secure_verification,
        "launch_implementation_inventory_hash": (
            launch_implementation_inventory_hash
        ),
    }
    if secure_verification:
        request["worker_cache_seal"] = (
            None if worker_cache_seal is None else dict(worker_cache_seal)
        )
    if cfg.payload.get("schema") == PHASE7_CONFIG_SCHEMA_V2 and action == "initialize":
        if secure_verification and (
            implementation_references is None or implementation_source_bundle is None
        ):
            raise ValueError(
                "V2 secure worker request requires implementation source evidence"
            )
        if implementation_source_bundle is not None:
            from bayesfilter.inference.hmc_smoke_authority import (
                implementation_source_bundle_hash,
            )
        request.update(
            {
                "v2_config_payload": dict(cfg.payload),
                "governed_source_paths": {
                    name: str(path)
                    for name, path in governed_paths.items()
                },
                "expected_transition_identity_hash": cfg.payload[
                    "adopted_identities"
                ]["transition_identity_hash"],
                "implementation_references": (
                    None
                    if implementation_references is None
                    else {
                        name: dict(reference)
                        for name, reference in implementation_references.items()
                    }
                ),
                "implementation_source_bundle": (
                    None
                    if implementation_source_bundle is None
                    else dict(implementation_source_bundle)
                ),
                "implementation_source_bundle_hash": (
                    None
                    if implementation_source_bundle is None
                    else implementation_source_bundle_hash(
                        implementation_source_bundle
                    )
                ),
            }
        )
    if schema == PHASE7_CONFIG_SCHEMA_V3 and action == "initialize":
        if secure_verification and (
            implementation_references is None or implementation_source_bundle is None
        ):
            raise ValueError(
                "V3 secure worker request requires implementation source evidence"
            )
        if implementation_source_bundle is not None:
            from bayesfilter.inference.hmc_smoke_authority import (
                implementation_source_bundle_hash,
            )
        request.update(
            {
                "v3_config_payload": dict(cfg.payload),
                "governed_source_paths": {
                    name: str(path) for name, path in governed_paths.items()
                },
                "expected_transition_identity_hash": cfg.payload[
                    "expected_identities"
                ]["transition_identity_hash"],
                "implementation_references": (
                    None
                    if implementation_references is None
                    else {
                        name: dict(reference)
                        for name, reference in implementation_references.items()
                    }
                ),
                "implementation_source_bundle": (
                    None
                    if implementation_source_bundle is None
                    else dict(implementation_source_bundle)
                ),
                "implementation_source_bundle_hash": (
                    None
                    if implementation_source_bundle is None
                    else implementation_source_bundle_hash(
                        implementation_source_bundle
                    )
                ),
            }
        )
    return request


def _expected_transition_identity_hash(
    cfg: DeterministicLGSSMPhase7Config,
) -> str | None:
    schema = cfg.payload.get("schema")
    if schema == PHASE7_CONFIG_SCHEMA_V2:
        return str(cfg.payload["adopted_identities"]["transition_identity_hash"])
    if schema == PHASE7_CONFIG_SCHEMA_V3:
        return str(cfg.payload["expected_identities"]["transition_identity_hash"])
    return None


def _assert_worker_response(
    response: Mapping[str, Any],
    *,
    action: str,
    expected_worker_index: int,
    expected_pid: int | None,
    expected_seed: Sequence[int],
    expected_target_scope: str,
    expected_transition_identity_hash: str | None = None,
    expected_implementation_source_bundle_hash: str | None = None,
    expected_worker_cache_seal_hash: str | None = None,
    expected_launch_implementation_inventory_hash: str | None = None,
    secure_source_verification: bool = False,
) -> None:
    expected_fields = {
        "schema",
        "action",
        "worker_index",
        "pid",
        "passed",
        "seed",
        "final_state",
        "raw_samples",
        "diagnostics",
        "metadata",
    }
    if not isinstance(response, Mapping) or set(response) != expected_fields or (
        response.get("schema")
        != "bayesfilter.deterministic_lgssm_hmc_phase7_worker_result.v1"
    ):
        raise DeterministicLGSSMPhase7Error("worker response schema mismatch")
    if response.get("passed") is not True:
        raise DeterministicLGSSMPhase7Error("worker response did not pass")
    if response.get("action") != action:
        raise DeterministicLGSSMPhase7Error("worker action mismatch")
    if type(response.get("worker_index")) is not int or response[
        "worker_index"
    ] != expected_worker_index:
        raise DeterministicLGSSMPhase7Error("worker response ordering mismatch")
    if type(response.get("pid")) is not int or response["pid"] < 1 or (
        expected_pid is not None and response["pid"] != expected_pid
    ):
        raise DeterministicLGSSMPhase7Error("persistent worker PID mismatch")
    if not isinstance(response.get("seed"), Sequence) or isinstance(
        response["seed"], (str, bytes)
    ) or tuple(response["seed"]) != tuple(expected_seed) or any(
        type(item) is not int for item in response["seed"]
    ):
        raise DeterministicLGSSMPhase7Error("worker response seed mismatch")
    metadata = response.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DeterministicLGSSMPhase7Error("worker metadata is missing")
    if metadata.get("jit_compile") is not True or metadata.get("use_xla") is not True:
        raise DeterministicLGSSMPhase7Error("worker XLA metadata mismatch")
    if metadata.get("cuda_visible_devices") != "-1":
        raise DeterministicLGSSMPhase7Error("worker GPU hiding mismatch")
    compile_count = metadata.get("compile_trace_count")
    if type(compile_count) is not int or compile_count != 1:
        raise DeterministicLGSSMPhase7Error("worker XLA runner retraced")
    if metadata.get("target_scope") != expected_target_scope:
        raise DeterministicLGSSMPhase7Error("worker target scope mismatch")
    if metadata.get("child_source_references_verified") is not True:
        raise DeterministicLGSSMPhase7Error("worker source identity was not verified")
    if expected_launch_implementation_inventory_hash is not None:
        if metadata.get("child_implementation_references_verified") is not False or (
            metadata.get("child_implementation_verification_status")
            != "launch_inventory_only_not_child_byte_verified"
        ) or metadata.get("launch_implementation_inventory_hash") != (
            expected_launch_implementation_inventory_hash
        ):
            raise DeterministicLGSSMPhase7Error(
                "worker launch implementation inventory mismatch"
            )
    elif metadata.get("child_implementation_references_verified") is not True:
        raise DeterministicLGSSMPhase7Error(
            "worker implementation identity was not verified"
        )
    if secure_source_verification and (
        metadata.get("child_loaded_source_bytes_verified") is not True
    ):
        raise DeterministicLGSSMPhase7Error(
            "worker loaded source bytes were not verified"
        )
    if metadata.get("child_transition_identity_verified") is not True:
        raise DeterministicLGSSMPhase7Error("worker transition identity was not verified")
    if expected_transition_identity_hash is not None and (
        metadata.get("child_transition_identity_hash")
        != expected_transition_identity_hash
    ):
        raise DeterministicLGSSMPhase7Error(
            "worker secure source/cache provenance mismatch"
            if secure_source_verification
            else "worker transition identity hash mismatch"
        )
    if secure_source_verification and (
        metadata.get("child_implementation_source_bundle_hash")
        != expected_implementation_source_bundle_hash
        or metadata.get("child_worker_cache_seal_hash")
        != expected_worker_cache_seal_hash
    ):
        raise DeterministicLGSSMPhase7Error(
            "worker secure source/cache provenance mismatch"
        )


def _record_initialized_worker_pid(worker_pids: list[int], pid: int) -> None:
    if type(pid) is not int or pid < 1:
        raise DeterministicLGSSMPhase7Error(
            "worker initialization returned an invalid PID"
        )
    if pid in worker_pids:
        raise DeterministicLGSSMPhase7Error(
            "worker initialization returned a duplicate PID"
        )
    worker_pids.append(pid)


def _worker_environment(cfg: DeterministicLGSSMPhase7Config) -> Mapping[str, str]:
    environment = {
        str(key): str(value)
        for key, value in cfg.payload["execution"]["thread_environment"].items()
    }
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "1"
    environment["MPLCONFIGDIR"] = "/tmp/matplotlib-bayesfilter-phase7-worker"
    return environment


def _public_check_summary(
    diagnostic: Mapping[str, Any],
    *,
    completed: int,
    stage: str,
) -> Mapping[str, Any]:
    return {
        "stage": stage,
        "completed_results_per_chain": int(completed),
        "passed": bool(diagnostic["passed"]),
        "max_rhat": diagnostic.get("max_rhat"),
        "min_bulk_ess": diagnostic.get("min_bulk_ess"),
        "min_tail_ess": diagnostic.get("min_tail_ess"),
        "input_all_finite": diagnostic.get("input_all_finite"),
        "diagnostics_all_finite": diagnostic.get("diagnostics_all_finite"),
        "hard_vetoes": diagnostic.get("hard_vetoes", ()),
    }


def _write_private_samples(
    path: Path,
    *,
    retained: np.ndarray,
    final_worker_states: Sequence[np.ndarray | None],
    config_hash: str,
    private_replay_hash: str,
    smoke_launch_context: Any | None = None,
) -> None:
    states = np.stack([np.asarray(item) for item in final_worker_states], axis=0)
    if smoke_launch_context is not None and not _is_academic_launch_context(
        smoke_launch_context
    ):
        session = smoke_launch_context.output_session
        with session.begin_binary_write("private_samples_path") as handle:
            np.savez_compressed(
                handle,
                retained_raw_samples=np.asarray(retained, dtype=np.float64),
                final_worker_states=np.asarray(states, dtype=np.float64),
                config_hash=np.asarray(config_hash),
                private_replay_hash=np.asarray(private_replay_hash),
            )
            handle.flush()
            os.fsync(handle.fileno())
        session.finish_binary_write("private_samples_path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        retained_raw_samples=np.asarray(retained, dtype=np.float64),
        final_worker_states=np.asarray(states, dtype=np.float64),
        config_hash=np.asarray(config_hash),
        private_replay_hash=np.asarray(private_replay_hash),
    )
    os.replace(temporary, path)


def _inspect_private_samples(
    path: Path,
    *,
    expected_draw_count: int,
    expected_chain_count: int,
    expected_parameter_count: int,
    expected_config_hash: str,
    expected_private_replay_hash: str,
    smoke_launch_context: Any | None = None,
) -> Mapping[str, Any]:
    if smoke_launch_context is None or _is_academic_launch_context(
        smoke_launch_context
    ):
        archive_source: Any = path
    else:
        archive_source = io.BytesIO(
            smoke_launch_context.output_session.read_bytes("private_samples_path")
        )
    with np.load(archive_source, allow_pickle=False) as archive:
        retained = np.asarray(archive["retained_raw_samples"], dtype=np.float64)
        states = np.asarray(archive["final_worker_states"], dtype=np.float64)
        config_hash = str(archive["config_hash"].item())
        replay_hash = str(archive["private_replay_hash"].item())
    expected_shape = (
        int(expected_draw_count),
        int(expected_chain_count),
        int(expected_parameter_count),
    )
    if retained.shape != expected_shape:
        raise DeterministicLGSSMPhase7Error("private retained sample shape mismatch")
    if states.shape != (2, 2, int(expected_parameter_count)):
        raise DeterministicLGSSMPhase7Error("private final worker state shape mismatch")
    if not np.all(np.isfinite(retained)) or not np.all(np.isfinite(states)):
        raise DeterministicLGSSMPhase7Error("private sample archive is nonfinite")
    if config_hash != expected_config_hash or replay_hash != expected_private_replay_hash:
        raise DeterministicLGSSMPhase7Error("private sample provenance mismatch")
    if smoke_launch_context is None or _is_academic_launch_context(
        smoke_launch_context
    ):
        file_sha256 = _file_sha256(path)
        byte_count = path.stat().st_size
    else:
        private_reference = smoke_launch_context.output_session.file_reference(
            "private_samples_path"
        )
        file_sha256 = private_reference["file_sha256"]
        byte_count = private_reference["byte_count"]
    return {
        "file_sha256": file_sha256,
        "byte_count": byte_count,
        "shape_verified": True,
        "finite_verified": True,
        "provenance_verified": True,
    }


def _write_controller_failure(
    cfg: DeterministicLGSSMPhase7Config,
    *,
    public_output: Path,
    progress_output: Path,
    progress: Mapping[str, Any],
    stage: str,
    reason: str,
    preflight: Mapping[str, Any],
    worker_pids: Sequence[int],
    elapsed: float,
    smoke: bool,
    smoke_launch_context: Any | None = None,
    final_diagnostic: Mapping[str, Any] | None = None,
    failure_classification: str | None = None,
    failure_detail: str | None = None,
    workers_started: bool = False,
    hmc_transition_executed: bool = False,
) -> Mapping[str, Any]:
    result = {
        "schema": _failure_schema(smoke_launch_context),
        "passed": False,
        "decision": _block_decision(smoke_launch_context),
        "smoke": bool(smoke),
        **_runtime_authority_links(smoke_launch_context),
        "stage": str(stage),
        "reason": str(reason),
        **(
            {"failure_classification": failure_classification}
            if failure_classification is not None
            else {}
        ),
        **(
            {"failure_detail": failure_detail}
            if failure_detail is not None
            else {}
        ),
        "config_hash": cfg.hash,
        **_preflight_result_evidence(preflight, smoke_launch_context),
        "worker_pids": tuple(int(item) for item in worker_pids),
        "final_diagnostics": final_diagnostic,
        "jit_compile_false_runtime_executed": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "elapsed_seconds": float(elapsed),
        **_runtime_execution_state(
            smoke_launch_context,
            workers_started=workers_started,
            hmc_transition_executed=hmc_transition_executed,
        ),
        "phase8_executed": False,
        "nonclaims": _failure_nonclaims(cfg, smoke_launch_context),
    }
    result = _finalize_runtime_artifact(result, smoke_launch_context)
    _write_runtime_json(
        public_output,
        result,
        smoke_launch_context=smoke_launch_context,
        role="public_result_path",
    )
    updated_progress = {
        **dict(progress),
        "status": "blocked_result_written",
        "completed": True,
        "passed": False,
        "result_artifact_hash": result["artifact_hash"],
    }
    _write_runtime_json(
        progress_output,
        _finalize_progress_artifact(updated_progress, smoke_launch_context),
        smoke_launch_context=smoke_launch_context,
        role="public_progress_path",
    )
    return result


def _write_runtime_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    smoke_launch_context: Any | None,
    role: str,
) -> None:
    if smoke_launch_context is None or _is_academic_launch_context(
        smoke_launch_context
    ):
        if smoke_launch_context is not None and path != smoke_launch_context.paths[role]:
            raise DeterministicLGSSMPhase7Error(
                "academic campaign output path mismatch"
            )
        atomic_write_json(path, payload)
        return
    expected = smoke_launch_context.paths[role]
    if path != expected:
        raise DeterministicLGSSMPhase7Error("secure smoke output path mismatch")
    smoke_launch_context.output_session.write_json(role, payload)


def _validate_controller_counts(payload: Mapping[str, Any]) -> None:
    burnin = payload["burnin"]
    retained = payload["retained"]
    if (
        int(burnin["initial_results_per_chain"]) != 2000
        or int(burnin["extension_results_per_chain"]) != 1000
        or int(burnin["check_window_results_per_chain"]) != 1000
        or int(burnin["max_results_per_chain"]) != 16000
    ):
        raise ValueError("Phase 7 burn-in counts mismatch")
    if (
        int(retained["initial_results_per_chain"]) != 4000
        or int(retained["extension_results_per_chain"]) != 2000
        or int(retained["check_interval_results_per_chain"]) != 2000
        or int(retained["max_results_per_chain"]) != 40000
    ):
        raise ValueError("Phase 7 retained counts mismatch")


def _check_wall_time(
    config: DeterministicLGSSMPhase7Config,
    start: float,
    *,
    cap_seconds: float | None = None,
) -> None:
    cap = (
        float(config.payload["execution"]["wall_time_cap_seconds"])
        if cap_seconds is None
        else float(cap_seconds)
    )
    if time.monotonic() - start >= cap:
        raise DeterministicLGSSMPhase7Error("machine wall-time cap reached")


def _remaining_wall_time(
    config: DeterministicLGSSMPhase7Config,
    start: float,
    *,
    cap_seconds: float | None = None,
) -> float:
    cap = (
        float(config.payload["execution"]["wall_time_cap_seconds"])
        if cap_seconds is None
        else float(cap_seconds)
    )
    remaining = cap - (time.monotonic() - start)
    if remaining <= 0.0:
        raise TimeoutError("machine wall-time cap reached")
    return remaining


def _terminate_executors(
    executors: Sequence[concurrent.futures.ProcessPoolExecutor],
    *,
    deadline: float | None = None,
) -> None:
    """Terminate every worker under one deadline without masking the caller."""

    shared_deadline = time.monotonic() + 5.0 if deadline is None else float(deadline)
    processes: list[Any] = []
    control_flow: BaseException | None = None
    for executor in executors:
        try:
            processes.extend(tuple(getattr(executor, "_processes", {}).values()))
        except BaseException as error:
            if not isinstance(error, Exception) and control_flow is None:
                control_flow = error

    # Signal every peer before waiting for any one process.
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except BaseException as error:
            if not isinstance(error, Exception) and control_flow is None:
                control_flow = error
            try:
                process.terminate()
            except BaseException as fallback_error:
                if not isinstance(fallback_error, Exception) and control_flow is None:
                    control_flow = fallback_error

    for process in processes:
        try:
            remaining = max(0.0, shared_deadline - time.monotonic())
            if remaining > 0.0:
                process.join(timeout=remaining)
            if process.is_alive():
                process.kill()
        except BaseException as error:
            if not isinstance(error, Exception) and control_flow is None:
                control_flow = error
            try:
                process.kill()
            except BaseException as fallback_error:
                if not isinstance(fallback_error, Exception) and control_flow is None:
                    control_flow = fallback_error

    for executor in executors:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except BaseException as error:
            if not isinstance(error, Exception) and control_flow is None:
                control_flow = error
    if control_flow is not None:
        raise control_flow.with_traceback(control_flow.__traceback__)


def _terminate_executors_verified(
    executors: Sequence[concurrent.futures.ProcessPoolExecutor],
    *,
    worker_pids: Sequence[int],
    deadline: float,
) -> Mapping[str, Any]:
    """Terminate and confirm every academic worker before a strict pass."""

    processes: list[Any] = []
    errors: list[str] = []
    for executor in executors:
        try:
            processes.extend(tuple(getattr(executor, "_processes", {}).values()))
        except Exception as error:
            errors.append(f"inventory:{type(error).__name__}")
    observed_pids = tuple(
        int(getattr(process, "pid", 0) or 0) for process in processes
    )
    if tuple(sorted(observed_pids)) != tuple(sorted(int(pid) for pid in worker_pids)):
        raise DeterministicLGSSMPhase7Error(
            "academic worker teardown PID inventory mismatch"
        )
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception as error:
            errors.append(f"terminate:{type(error).__name__}")
    for process in processes:
        try:
            remaining = max(0.0, float(deadline) - time.monotonic())
            process.join(timeout=remaining)
            if process.is_alive():
                process.kill()
                remaining = max(0.0, float(deadline) - time.monotonic())
                process.join(timeout=remaining)
            if process.is_alive():
                errors.append(f"pid_{process.pid}_still_alive")
        except Exception as error:
            errors.append(f"join:{type(error).__name__}")
    for executor in executors:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as error:
            errors.append(f"shutdown:{type(error).__name__}")
    if errors:
        raise DeterministicLGSSMPhase7Error(
            f"academic worker teardown failed: {tuple(errors)}"
        )
    return {
        "all_exited": True,
        "worker_pids": tuple(int(pid) for pid in worker_pids),
        "verification": "terminate_join_kill_rejoin_is_alive_false",
    }


def _executor_teardown_deadline(
    config: DeterministicLGSSMPhase7Config,
    start: float,
    *,
    cap_seconds: float | None = None,
) -> float:
    cap = (
        float(config.payload["execution"]["wall_time_cap_seconds"])
        if cap_seconds is None
        else float(cap_seconds)
    )
    global_deadline = float(start) + cap
    return min(global_deadline, time.monotonic() + 5.0)


def _classify_academic_runtime_failure(
    error: BaseException,
    *,
    stage: str,
) -> str | None:
    if isinstance(error, (TimeoutError, concurrent.futures.TimeoutError)):
        return "continuation_veto"
    if isinstance(error, (DeterministicLGSSMPhase7Error, ValueError)):
        return "continuation_veto"
    return "infrastructure_failure"


def _assert_diagnostic_no_hard_vetoes(diagnostic: Mapping[str, Any]) -> None:
    hard_vetoes = tuple(str(item) for item in diagnostic.get("hard_vetoes", ()))
    if hard_vetoes:
        raise DeterministicLGSSMPhase7Error(
            "convergence diagnostic hard veto"
        )


def _verify_embedded_artifact_hash(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed = payload.get("artifact_hash")
    if not isinstance(observed, str):
        raise DeterministicLGSSMPhase7Error(f"{label} artifact hash is missing")
    without_hash = {key: value for key, value in payload.items() if key != "artifact_hash"}
    expected = f"sha256:{stable_config_hash(without_hash)}"
    if observed != expected:
        raise DeterministicLGSSMPhase7Error(f"{label} embedded artifact hash mismatch")


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload


def _snapshot_json_source(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return {
        "path": path,
        "payload": payload,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_artifact_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized
