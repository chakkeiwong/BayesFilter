#!/usr/bin/env python3
"""Run the authorized two-seed SSL-LSTM DSGE-parity NeuTra program."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    DSGE_PAPER_NEUTRA_FAMILY,
    DSGE_PAPER_TRAINING_BATCH_SIZE,
    DSGE_PAPER_TRAINING_STEPS,
    NeuTraReverseKLTrainer,
    dsge_paper_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    FREE_PARAMETER_NAMES,
    PRIOR_CENTER_VALUES,
    TARGET_SEMANTIC_SHA256,
    locked_ssl_lstm_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm_neutra.dsge_parity_material_candidate.v1"
PROGRAM_SCHEMA = "bayesfilter.ssl_lstm_neutra.dsge_parity_material_program.v1"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-plan-2026-07-15.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-result-2026-07-15.md"
)
CANARY_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "procedure-parity-repair/timing-canary-r4.json"
)
HISTORICAL_HELPER_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_neutra_phase4_bounded_training_2026_07_14.py"
)
MATERIAL_TEST_PATH = Path(
    "tests/test_ssl_lstm_neutra_dsge_parity_material_training.py"
)
DSGE_SOURCE_ROOT = Path("/home/ubuntu/python/dsge_hmc")
DSGE_SOURCE_COMMIT = "d94566c9f70b3143e599a56eba7cb461ff2bda88"
CANARY_SHA256 = "d20e1219026dcd3b62b2218b1978ec6979647d41022062db578ac766fdefb001"
CANARY_SOURCE_SHA256 = {
    "artifact_loader": "f3dbb9ad2f750679f2d67b63bb8cce4db2c16622907512a701de799abf9194ec",
    "parity_test": "c6b15ab6eb09505700ef9fc65b3216c7e9c625e84be323c5690d4c1731293e12",
    "runner": "26077ed39f9a586214532a6b29ca83097c669b22197bc01e916030b81ab57ab5",
    "target": "6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667",
    "trainer": "211d93ec5bd228ae444b814d891af9a7714c5f3026e52ad0fcf29d992b3469ae",
}
SOURCE_PATHS = {
    "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
    "parity_test": Path("tests/test_neutra_dsge_procedure_parity.py"),
    "target": Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
    "trainer": Path("bayesfilter/inference/neutra_training.py"),
}

SHARED_GPU_SECONDS = 36_000.0
PER_SEED_GPU_SECONDS = 18_000.0
FINALIZATION_RESERVE_SECONDS = 300.0
CHECKPOINT_EVERY = 100
VALIDATION_EVERY = 250
VALIDATION_BATCH_SIZE = 64
INVERSE_RADIUS_MAX = 4.30
ROUNDTRIP_MAX_ABS = 1.0e-9
DENSE_SATURATION_FRACTION_MAX = 0.05


class MaterialTrainingError(RuntimeError):
    """Raised when the material program violates its prospective contract."""


class ResourceStop(MaterialTrainingError):
    """Raised before a prospective GPU-time boundary can be crossed."""


@dataclass(frozen=True)
class SeedSpec:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]


SEEDS = (
    SeedSpec("seed-a", (20260715, 4101), (20260715, 5101), (20260715, 5201)),
    SeedSpec("seed-b", (20260715, 4102), (20260715, 5102), (20260715, 5202)),
)


def _load_historical_helpers() -> Any:
    path = ROOT / HISTORICAL_HELPER_PATH
    spec = importlib.util.spec_from_file_location("ssl_lstm_phase4_probe_helpers", path)
    if spec is None or spec.loader is None:
        raise MaterialTrainingError("historical probe helper import unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = _load_historical_helpers()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, payload: Any) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise MaterialTrainingError(f"output already exists: {path}")
    absolute.write_bytes(_canonical_bytes(payload))


def _strict_load(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise MaterialTrainingError(f"nonfinite JSON constant {value}: {path}")

    value = json.loads((ROOT / path).read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise MaterialTrainingError(f"expected JSON object: {path}")
    return value


def _require_fresh_root(output_root: Path) -> None:
    absolute = ROOT / output_root
    if absolute.exists() and any(absolute.iterdir()):
        raise MaterialTrainingError(f"program output root is not fresh: {output_root}")
    absolute.mkdir(parents=True, exist_ok=True)


def _assert_seed_contract() -> None:
    roles = [
        seed
        for spec in SEEDS
        for seed in (
            spec.initialization_seed,
            spec.training_seed,
            spec.validation_seed,
        )
    ]
    if len(roles) != len(set(roles)):
        raise MaterialTrainingError("material seed roles overlap")
    if (20260715, 4099) in roles or (20260714, 3301) in roles:
        raise MaterialTrainingError("material seed overlaps canary or probe role")


def _assert_source_bindings() -> dict[str, Any]:
    if _sha256(CANARY_PATH) != CANARY_SHA256:
        raise MaterialTrainingError("authoritative timing-canary identity drift")
    canary = _strict_load(CANARY_PATH)
    if canary.get("status") != "DSGE_PROCEDURE_PARITY_GPU_XLA_TIMING_CANARY_PASSED":
        raise MaterialTrainingError("authoritative timing canary did not pass")
    recorded = canary.get("run_manifest", {}).get("source_sha256")
    if recorded != CANARY_SOURCE_SHA256:
        raise MaterialTrainingError("authoritative timing-canary source manifest drift")
    for role, path in SOURCE_PATHS.items():
        if _sha256(path) != CANARY_SOURCE_SHA256[role]:
            raise MaterialTrainingError(f"final parity source drift: {role}")
    sibling_commit = _git("rev-parse", "HEAD", cwd=DSGE_SOURCE_ROOT)
    if sibling_commit != DSGE_SOURCE_COMMIT:
        raise MaterialTrainingError("dsge_hmc source commit drift")
    return {
        "authoritative_canary_path": CANARY_PATH.as_posix(),
        "authoritative_canary_sha256": CANARY_SHA256,
        "authoritative_canary_source_sha256": dict(CANARY_SOURCE_SHA256),
        "dsge_source_root": DSGE_SOURCE_ROOT.as_posix(),
        "dsge_source_commit": sibling_commit,
        "material_runner_sha256": _sha256(Path(__file__).resolve().relative_to(ROOT)),
        "material_test_sha256": _sha256(MATERIAL_TEST_PATH),
        "historical_probe_helper_sha256": _sha256(HISTORICAL_HELPER_PATH),
        "plan_sha256": _sha256(PLAN_PATH),
    }


def _source_files(bindings: dict[str, Any]) -> list[dict[str, str]]:
    rows = (
        (Path(__file__).resolve().relative_to(ROOT), "material_program_runner"),
        (MATERIAL_TEST_PATH, "material_program_tests"),
        (HISTORICAL_HELPER_PATH, "probe_and_gate_helpers_only"),
        (Path("bayesfilter/inference/neutra_training.py"), "source_parity_trainer"),
        (Path("bayesfilter/inference/neutra_artifacts.py"), "frozen_artifact_loader"),
        (Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"), "locked_target"),
        (Path("tests/test_neutra_dsge_procedure_parity.py"), "direct_source_parity_tests"),
        (PLAN_PATH, "prospective_plan"),
        (CANARY_PATH, "authoritative_timing_canary"),
    )
    source_rows = [
        {"path": path.as_posix(), "role": role, "sha256": _sha256(path)}
        for path, role in rows
    ]
    source_rows.append(
        {
            "path": bindings["dsge_source_root"],
            "role": "dsge_source_commit",
            "sha256": bindings["dsge_source_commit"],
        }
    )
    return source_rows


def _config(target: Any, spec: SeedSpec) -> Any:
    return dsge_paper_neutra_config(
        dimension=4,
        fixed_translation=PRIOR_CENTER_VALUES,
        target_parameter_names=FREE_PARAMETER_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        initialization_seed=spec.initialization_seed,
        jit_compile=True,
    )


def _step_batch(spec: SeedSpec, logical_step: int) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(spec.training_seed, tf.int32),
        int(logical_step),
    )
    return tf.random.stateless_normal(
        (DSGE_PAPER_TRAINING_BATCH_SIZE, 4),
        seed=seed,
        dtype=tf.float64,
    )


def _validation_batch(spec: SeedSpec) -> tf.Tensor:
    return tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4),
        seed=tf.constant(spec.validation_seed, tf.int32),
        dtype=tf.float64,
    )


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step:04d}.json"


def _candidate_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "result": output_dir / "result.json",
        "failure": output_dir / "failure.json",
        "final_state": output_dir / "final-state.json",
        "frozen_payload": output_dir / "frozen-payload.json",
    }


def _resource_stop_state_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"resource-stop-state-{step:04d}.json"


def _check_budget_and_preserve_state(
    trainer: NeuTraReverseKLTrainer,
    *,
    output_dir: Path,
    candidate_started: float,
    program_started: float,
) -> None:
    try:
        _check_budget(
            candidate_started=candidate_started,
            program_started=program_started,
            reserve_finalization=True,
        )
    except ResourceStop:
        step = int(trainer.step.numpy())
        _write_json(_resource_stop_state_path(output_dir, step), trainer.state_payload())
        raise


def _check_budget(
    *,
    candidate_started: float,
    program_started: float,
    reserve_finalization: bool,
) -> None:
    reserve = FINALIZATION_RESERVE_SECONDS if reserve_finalization else 0.0
    candidate_elapsed = time.perf_counter() - candidate_started
    program_elapsed = time.perf_counter() - program_started
    if candidate_elapsed + reserve >= PER_SEED_GPU_SECONDS:
        raise ResourceStop("per-seed GPU-time boundary reached")
    if program_elapsed + reserve >= SHARED_GPU_SECONDS:
        raise ResourceStop("shared GPU-time boundary reached")


def candidate_decision(
    *,
    final_validation: dict[str, Any],
    loss_interval: dict[str, float],
    probes: dict[str, Any],
    reload_exact: bool,
) -> tuple[str, list[str], list[str]]:
    hard_vetoes = []
    promotion_vetoes = []
    if not probes["all_finite"]:
        hard_vetoes.append("probe_nonfinite")
    if not reload_exact:
        hard_vetoes.append("frozen_reload_mismatch")
    if probes["roundtrip_max_abs"] > ROUNDTRIP_MAX_ABS:
        promotion_vetoes.append("roundtrip_residual_above_threshold")
    if probes["original_neighborhood_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        promotion_vetoes.append("original_neighborhood_missing_support")
    if probes["moderate_shell_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        promotion_vetoes.append("moderate_shell_missing_support")
    if loss_interval["one_sided_95_upper"] >= 0.0:
        promotion_vetoes.append("heldout_loss_improvement_not_established")
    if final_validation["saturation_fraction"] > DENSE_SATURATION_FRACTION_MAX:
        promotion_vetoes.append("dense_scale_saturation_above_cap")
    if hard_vetoes:
        return "INVALID_HARD_VETO", hard_vetoes, promotion_vetoes
    if promotion_vetoes:
        return "CANDIDATE_NOT_NOMINATED", hard_vetoes, promotion_vetoes
    return "VIABLE_FROZEN_CANDIDATE", hard_vetoes, promotion_vetoes


def should_continue_after_candidate(decision: str) -> bool:
    if decision in {"VIABLE_FROZEN_CANDIDATE", "CANDIDATE_NOT_NOMINATED"}:
        return True
    if decision == "INVALID_HARD_VETO":
        return False
    raise ValueError(f"unknown candidate decision: {decision}")


def can_start_next_seed(program_elapsed_seconds: float) -> bool:
    elapsed = float(program_elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("program elapsed seconds must be finite and nonnegative")
    return elapsed + PER_SEED_GPU_SECONDS <= SHARED_GPU_SECONDS


def classify_program(rows: list[dict[str, Any]], wall_seconds: float) -> tuple[bool, str]:
    wall = float(wall_seconds)
    if not math.isfinite(wall) or wall < 0.0:
        raise ValueError("program wall seconds must be finite and nonnegative")
    within_cap = wall <= SHARED_GPU_SECONDS
    completed = within_cap and len(rows) == len(SEEDS) and all(
        row.get("decision") != "INVALID_HARD_VETO" for row in rows
    )
    decisions = [row.get("decision") for row in rows]
    if completed and decisions == ["VIABLE_FROZEN_CANDIDATE"] * 2:
        return True, "TWO_INDEPENDENT_CANDIDATES_NOMINATED"
    if completed and decisions.count("VIABLE_FROZEN_CANDIDATE") == 1:
        return True, "SEED_INSTABILITY_REPAIR_REQUIRED"
    if completed:
        return True, "SOURCE_MATCHED_CANDIDATE_REJECTED_UNDER_DECLARED_GATES"
    return False, "PROGRAM_STOPPED_BY_CONTINUATION_VETO"


def _reload_exact(
    trainer: NeuTraReverseKLTrainer,
    loaded: Any,
    validation_z: tf.Tensor,
) -> bool:
    live_theta, live_logdet = trainer.forward_and_logdet(validation_z)
    frozen_theta = loaded.transport.forward_batch(validation_z)
    frozen_logdet = loaded.transport.log_abs_det_jacobian_batch(validation_z)
    return bool(
        tf.reduce_all(tf.equal(live_theta, frozen_theta)).numpy()
        and tf.reduce_all(tf.equal(live_logdet, frozen_logdet)).numpy()
    )


def run_candidate(
    spec: SeedSpec,
    *,
    output_dir: Path,
    program_started: float,
    physical_gpus: list[Any],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    absolute = ROOT / output_dir
    if absolute.exists() and any(absolute.iterdir()):
        raise MaterialTrainingError(f"candidate directory is not fresh: {output_dir}")
    absolute.mkdir(parents=True, exist_ok=True)
    paths = _candidate_paths(output_dir)
    candidate_started_at = _now()
    candidate_started = time.perf_counter()
    training_history = []
    validation_history = []
    checkpoint_rows = []
    resume_replay = None
    target = locked_ssl_lstm_posterior_target()
    config = _config(target, spec)
    if config.family != DSGE_PAPER_NEUTRA_FAMILY:
        raise MaterialTrainingError("material runner did not select source-parity preset")
    trainer = NeuTraReverseKLTrainer(target, config)
    validation_z = _validation_batch(spec)
    initial_validation = HELPERS._host_validation(  # noqa: SLF001
        trainer.validation_batch(validation_z),
        family="dense_iaf",
        s_max=config.s_max,
    )
    validation_history.append({"step": 0, **initial_validation})

    warmup_batch = _step_batch(spec, 1)
    hlo_text = trainer._compiled_train_step.experimental_get_compiler_ir(  # noqa: SLF001
        warmup_batch
    )(stage="hlo")
    if not isinstance(hlo_text, str) or "HloModule" not in hlo_text:
        raise MaterialTrainingError("material trainer did not expose HLO evidence")

    for logical_step in range(1, DSGE_PAPER_TRAINING_STEPS + 1):
        _check_budget_and_preserve_state(
            trainer,
            output_dir=output_dir,
            candidate_started=candidate_started,
            program_started=program_started,
        )
        z = warmup_batch if logical_step == 1 else _step_batch(spec, logical_step)
        if logical_step == CHECKPOINT_EVERY + 1:
            before_replay = trainer.state_payload()
            first_row = HELPERS._host_step(trainer.train_step(z))  # noqa: SLF001
            expected_state = trainer.state_payload()
            trainer.restore_state(before_replay)
            replay_row = HELPERS._host_step(trainer.train_step(z))  # noqa: SLF001
            replay_state = trainer.state_payload()
            replay_passed = (
                first_row == replay_row
                and _canonical_bytes(expected_state) == _canonical_bytes(replay_state)
            )
            if not replay_passed:
                raise MaterialTrainingError("material exact resume replay mismatch")
            resume_replay = {
                "logical_step": logical_step,
                "passed": True,
                "pre_state_hash": before_replay["state_hash"],
                "post_state_hash": replay_state["state_hash"],
            }
            row = replay_row
        else:
            row = HELPERS._host_step(trainer.train_step(z))  # noqa: SLF001
        training_history.append(row)

        if logical_step % CHECKPOINT_EVERY == 0:
            state = trainer.state_payload()
            checkpoint = _checkpoint_path(output_dir, logical_step)
            _write_json(checkpoint, state)
            checkpoint_rows.append(
                {
                    "step": logical_step,
                    "path": checkpoint.as_posix(),
                    "sha256": _sha256(checkpoint),
                    "state_hash": state["state_hash"],
                }
            )
        if logical_step % VALIDATION_EVERY == 0:
            validation = HELPERS._host_validation(  # noqa: SLF001
                trainer.validation_batch(validation_z),
                family="dense_iaf",
                s_max=config.s_max,
            )
            validation_history.append({"step": logical_step, **validation})

    _check_budget_and_preserve_state(
        trainer,
        output_dir=output_dir,
        candidate_started=candidate_started,
        program_started=program_started,
    )
    final_state = trainer.state_payload()
    _write_json(paths["final_state"], final_state)
    frozen_payload = trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-dsge-parity-material-{spec.label}",
        target_signature=target.target_signature(),
    )
    _write_json(paths["frozen_payload"], frozen_payload)
    loaded = load_frozen_neutra_artifact(
        frozen_payload,
        expected_target_signature=target.target_signature(),
    )
    reload_exact = _reload_exact(trainer, loaded, validation_z)
    probes = HELPERS._probe_diagnostics(target, loaded.transport)  # noqa: SLF001
    final_validation = validation_history[-1]
    loss_interval = HELPERS.paired_loss_upper_bound(
        initial_validation["per_sample_loss"],
        final_validation["per_sample_loss"],
    )
    decision, hard_vetoes, promotion_vetoes = candidate_decision(
        final_validation=final_validation,
        loss_interval=loss_interval,
        probes=probes,
        reload_exact=reload_exact,
    )
    output_devices = sorted(
        {
            *(variable.device for variable in trainer.variables),
            *initial_validation["output_devices"],
            *final_validation["output_devices"],
            *probes["output_devices"],
        }
    )
    if not output_devices or not all("GPU:" in device for device in output_devices):
        hard_vetoes.append("outputs_not_gpu_resident")
        decision = "INVALID_HARD_VETO"
    if int(trainer.step.numpy()) != DSGE_PAPER_TRAINING_STEPS:
        hard_vetoes.append("optimizer_step_count_mismatch")
        decision = "INVALID_HARD_VETO"
    post_bindings = _assert_source_bindings()
    if post_bindings != bindings:
        hard_vetoes.append("source_binding_changed_during_candidate")
        decision = "INVALID_HARD_VETO"

    wall_time = time.perf_counter() - candidate_started
    if wall_time > PER_SEED_GPU_SECONDS:
        hard_vetoes.append("per_seed_gpu_cap_overrun")
        decision = "INVALID_HARD_VETO"
    if time.perf_counter() - program_started > SHARED_GPU_SECONDS:
        hard_vetoes.append("shared_gpu_cap_overrun")
        decision = "INVALID_HARD_VETO"
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "decision": decision,
        "candidate": {
            "label": spec.label,
            "family": config.family,
            "initialization_seed": list(spec.initialization_seed),
            "training_seed": list(spec.training_seed),
            "validation_seed": list(spec.validation_seed),
            "steps": DSGE_PAPER_TRAINING_STEPS,
            "batch_size": DSGE_PAPER_TRAINING_BATCH_SIZE,
            "config": config.manifest_payload(),
        },
        "hard_vetoes": hard_vetoes,
        "promotion_vetoes": promotion_vetoes,
        "training": {
            "history": training_history,
            "checkpoints": checkpoint_rows,
            "resume_replay": resume_replay,
            "final_state_path": paths["final_state"].as_posix(),
            "final_state_sha256": _sha256(paths["final_state"]),
            "final_state_hash": final_state["state_hash"],
        },
        "validation": {
            "history": validation_history,
            "paired_final_minus_initial": loss_interval,
        },
        "frozen_transport": {
            "path": paths["frozen_payload"].as_posix(),
            "sha256": _sha256(paths["frozen_payload"]),
            "reload_exact": reload_exact,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "topology_hash": loaded.manifest.topology_hash,
            "tensor_hash": loaded.manifest.tensor_hash,
            "training_state_hash": loaded.manifest.training_state_hash,
            "component_order": list(frozen_payload["component_order"]),
        },
        "probe_diagnostics": probes,
        "thresholds": {
            "inverse_radius_max": INVERSE_RADIUS_MAX,
            "roundtrip_max_abs": ROUNDTRIP_MAX_ABS,
            "dense_saturation_fraction_max": DENSE_SATURATION_FRACTION_MAX,
            "heldout_loss_upper_bound_max": 0.0,
        },
        "source_bindings": bindings,
        "source_files": _source_files(bindings),
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "physical_gpus": [device.name for device in physical_gpus],
            "output_devices": output_devices,
            "dtype": "float64",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "soft_device_placement_during_run": False,
            "hlo_sha256": hashlib.sha256(hlo_text.encode("utf-8")).hexdigest(),
            "hlo_characters": len(hlo_text),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "started_at_utc": candidate_started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "charged_gpu_seconds": wall_time,
            "per_seed_gpu_cap_seconds": PER_SEED_GPU_SECONDS,
            "shared_gpu_cap_seconds": SHARED_GPU_SECONDS,
            "output_dir": output_dir.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "inference_status": {
            "hard_veto_evidence": list(hard_vetoes),
            "candidate_nomination_vetoes": list(promotion_vetoes),
            "statistically_supported_ranking": False,
            "continuous_differences_role": "descriptive_only",
        },
        "nonclaims": (
            "no posterior correctness or complete mode/tail coverage claim",
            "no HMC, convergence, or predictive claim",
            "no A/B ranking, superiority, or default-readiness claim",
            "no general NeuTra paper-fidelity or research-direction claim",
        ),
    }
    _write_json(paths["result"], payload)
    return payload


def _failure_payload(
    *,
    error: Exception,
    spec: SeedSpec,
    output_dir: Path,
    candidate_started_at: str,
    candidate_started: float,
    program_started: float,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    wall = time.perf_counter() - candidate_started
    return {
        "schema": SCHEMA,
        "status": "FAILED",
        "decision": "INVALID_HARD_VETO",
        "candidate": {
            "label": spec.label,
            "initialization_seed": list(spec.initialization_seed),
            "training_seed": list(spec.training_seed),
            "validation_seed": list(spec.validation_seed),
        },
        "hard_vetoes": [
            "resource_stop" if isinstance(error, ResourceStop) else "execution_failure"
        ],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "source_bindings": bindings,
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "started_at_utc": candidate_started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall,
            "charged_gpu_seconds": wall,
            "cumulative_program_seconds": time.perf_counter() - program_started,
            "per_seed_gpu_cap_seconds": PER_SEED_GPU_SECONDS,
            "shared_gpu_cap_seconds": SHARED_GPU_SECONDS,
            "trust_basis": "failure_receipt_only_gpu_provenance_not_established",
            "output_dir": output_dir.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "nonclaims": ["failed or truncated artifact; no candidate or scientific conclusion"],
    }


def run_program(output_root: Path) -> dict[str, Any]:
    _require_fresh_root(output_root)
    _assert_seed_contract()
    bindings = _assert_source_bindings()
    started_at = _now()
    started = time.perf_counter()
    previous_soft_placement = tf.config.get_soft_device_placement()
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise MaterialTrainingError("trusted material training requires a visible GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    program_start_path = output_root / "program-start.json"
    _write_json(
        program_start_path,
        {
            "schema": PROGRAM_SCHEMA,
            "status": "RUNNING",
            "started_at_utc": started_at,
            "pid": os.getpid(),
            "command": " ".join(sys.argv),
            "seeds": [spec.__dict__ for spec in SEEDS],
            "shared_gpu_cap_seconds": SHARED_GPU_SECONDS,
            "per_seed_gpu_cap_seconds": PER_SEED_GPU_SECONDS,
            "source_bindings": bindings,
        },
    )
    rows = []
    stop_reason = None
    try:
        with tf.device("/GPU:0"):
            for index, spec in enumerate(SEEDS):
                elapsed = time.perf_counter() - started
                if index > 0 and not can_start_next_seed(elapsed):
                    stop_reason = "insufficient_shared_budget_for_complete_next_seed"
                    break
                output_dir = output_root / spec.label
                candidate_started_at = _now()
                candidate_started = time.perf_counter()
                try:
                    payload = run_candidate(
                        spec,
                        output_dir=output_dir,
                        program_started=started,
                        physical_gpus=physical_gpus,
                        bindings=bindings,
                    )
                except Exception as error:
                    failure_path = output_dir / "failure.json"
                    failure = _failure_payload(
                        error=error,
                        spec=spec,
                        output_dir=output_dir,
                        candidate_started_at=candidate_started_at,
                        candidate_started=candidate_started,
                        program_started=started,
                        bindings=bindings,
                    )
                    if not (ROOT / failure_path).exists():
                        _write_json(failure_path, failure)
                    rows.append(
                        {
                            "label": spec.label,
                            "decision": "INVALID_HARD_VETO",
                            "path": failure_path.as_posix(),
                            "sha256": _sha256(failure_path),
                            "charged_gpu_seconds": failure["run_manifest"][
                                "charged_gpu_seconds"
                            ],
                        }
                    )
                    stop_reason = f"{spec.label}_hard_evidence_veto"
                    break
                result_path = output_dir / "result.json"
                rows.append(
                    {
                        "label": spec.label,
                        "decision": payload["decision"],
                        "path": result_path.as_posix(),
                        "sha256": _sha256(result_path),
                        "charged_gpu_seconds": payload["run_manifest"][
                            "charged_gpu_seconds"
                        ],
                    }
                )
                if not should_continue_after_candidate(payload["decision"]):
                    stop_reason = f"{spec.label}_hard_evidence_veto"
                    break
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    wall = time.perf_counter() - started
    within_cap = wall <= SHARED_GPU_SECONDS
    completed, program_decision = classify_program(rows, wall)
    payload = {
        "schema": PROGRAM_SCHEMA,
        "status": "COMPLETED" if completed else "STOPPED",
        "decision": program_decision,
        "stop_reason": stop_reason,
        "candidate_rows": rows,
        "budget": {
            "charged_gpu_seconds": wall,
            "shared_gpu_cap_seconds": SHARED_GPU_SECONDS,
            "remaining_gpu_seconds": max(0.0, SHARED_GPU_SECONDS - wall),
            "within_cap": within_cap,
        },
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "physical_gpus": [device.name for device in physical_gpus],
            "dtype": "float64",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "soft_device_placement_during_run": False,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall,
            "output_root": output_root.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "source_bindings": bindings,
        "inference_status": {
            "statistically_supported_ranking": False,
            "candidate_rejection_is_research_direction_rejection": False,
            "continuous_differences_role": "descriptive_only",
        },
        "nonclaims": (
            "no posterior, HMC, predictive, superiority, or readiness claim",
            "no general paper-fidelity or NeuTra research-direction claim",
        ),
    }
    _write_json(output_root / "program-result.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = run_program(args.program_output_root)
    print(payload["decision"], flush=True)
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
