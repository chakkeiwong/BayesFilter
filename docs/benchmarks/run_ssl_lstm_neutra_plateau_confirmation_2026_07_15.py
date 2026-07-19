#!/usr/bin/env python3
"""Confirm a frozen SSL-LSTM NeuTra tuning policy on two fresh streams."""

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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.neutra_training import (  # noqa: E402
    DSGE_PAPER_TRAINING_BATCH_SIZE,
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.neutra_training_control import (  # noqa: E402
    NeuTraPlateauConfig,
    NeuTraPlateauController,
    joint_training_checkpoint_payload,
    validate_joint_training_checkpoint,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    FREE_PARAMETER_NAMES,
    PRIOR_CENTER_VALUES,
    locked_ssl_lstm_posterior_target,
)


HELPER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_phase4_bounded_training_2026_07_14.py"
HELPER_SPEC = importlib.util.spec_from_file_location(
    "plateau_confirmation_helpers", HELPER_PATH
)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise RuntimeError("unable to load SSL-LSTM diagnostic helpers")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HELPERS
HELPER_SPEC.loader.exec_module(HELPERS)

SCHEMA = "bayesfilter.ssl_lstm_neutra.plateau_confirmation.v1"
POLICY_SCHEMA = "bayesfilter.ssl_lstm_neutra.tuning_policy.v1"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-optuna-plateau-training-repair-plan-2026-07-15.md"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/plateau-confirmation"
)
VALIDATION_BATCH_SIZE = 64
INVERSE_RADIUS_MAX = 4.30


class ConfirmationError(RuntimeError):
    """Raised when confirmation evidence is invalid."""


class ResourceStop(ConfirmationError):
    """Raised when the shared confirmation cap is exhausted."""


@dataclass(frozen=True)
class FreshStream:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]


FRESH_STREAMS = (
    FreshStream("fresh-c", (20260715, 7101), (20260715, 8101), (20260715, 8201)),
    FreshStream("fresh-d", (20260715, 7102), (20260715, 8102), (20260715, 8202)),
)
HISTORICAL_SEED_ROWS = frozenset(
    {
        (20260715, 4101),
        (20260715, 4102),
        (20260715, 5101),
        (20260715, 5102),
        (20260715, 5201),
        (20260715, 5202),
    }
)


def canonical(payload: Any) -> bytes:
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


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ConfirmationError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_SCHEMA:
        raise ConfirmationError("unsupported tuning policy schema")
    study_summary = ROOT / str(payload.get("study_summary_path"))
    if not study_summary.is_file():
        raise ConfirmationError("policy study summary is unavailable")
    if sha256(study_summary) != payload.get("study_summary_sha256"):
        raise ConfirmationError("policy study summary hash mismatch")
    declared_streams = tuple(
        FreshStream(
            label=str(row["label"]),
            initialization_seed=tuple(int(value) for value in row["initialization_seed"]),
            training_seed=tuple(int(value) for value in row["training_seed"]),
            validation_seed=tuple(int(value) for value in row["validation_seed"]),
        )
        for row in payload.get("fresh_streams", [])
    )
    if declared_streams != FRESH_STREAMS:
        raise ConfirmationError("fresh confirmation streams do not match frozen policy")
    all_fresh_seeds = {
        seed
        for stream in declared_streams
        for seed in (
            stream.initialization_seed,
            stream.training_seed,
            stream.validation_seed,
        )
    }
    if len(all_fresh_seeds) != 6 or all_fresh_seeds & HISTORICAL_SEED_ROWS:
        raise ConfirmationError("fresh confirmation seed separation failed")
    return payload


def source_bindings(policy_path: Path) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve().relative_to(ROOT),
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "controller": Path("bayesfilter/inference/neutra_training_control.py"),
        "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
        "plan": PLAN_PATH,
        "policy": policy_path.relative_to(ROOT),
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_paths": {key: value.as_posix() for key, value in paths.items()},
        "source_sha256": {
            key: sha256(ROOT / value) for key, value in paths.items()
        },
    }


def step_batch(stream: FreshStream, step: int) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(stream.training_seed, tf.int32), int(step)
    )
    return tf.random.stateless_normal(
        (DSGE_PAPER_TRAINING_BATCH_SIZE, 4), seed=seed, dtype=tf.float64
    )


def validation_batch(stream: FreshStream) -> tf.Tensor:
    return tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4),
        seed=tf.constant(stream.validation_seed, tf.int32),
        dtype=tf.float64,
    )


def trainer_config(target: Any, stream: FreshStream, policy: dict[str, Any]) -> Any:
    params = policy["selected_hyperparameters"]
    return ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=PRIOR_CENTER_VALUES,
        target_parameter_names=FREE_PARAMETER_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=float(params["learning_rate"]),
        initialization_scale=float(params["initialization_scale"]),
        gradient_clip_norm=float(params["gradient_clip_norm"]),
        initialization_seed=stream.initialization_seed,
        jit_compile=True,
    )


def controller_config(policy: dict[str, Any]) -> NeuTraPlateauConfig:
    row = policy["plateau_policy"]
    return NeuTraPlateauConfig(
        validation_check_every=int(row["validation_check_every"]),
        patience_steps=int(row["patience_steps"]),
        max_steps=int(row["max_steps"]),
        initial_learning_rate=float(policy["selected_hyperparameters"]["learning_rate"]),
        learning_rate_factor=float(row["learning_rate_factor"]),
        minimum_learning_rate_fraction=float(row["minimum_learning_rate_fraction"]),
        absolute_min_delta=float(row["absolute_min_delta"]),
        one_sided_critical_value=float(row["one_sided_critical_value"]),
        saturation_max=float(row["saturation_max"]),
    )


def host_validation(trainer: Any, z: tf.Tensor, step: int) -> dict[str, Any]:
    row = HELPERS._host_validation(  # noqa: SLF001
        trainer.validation_batch(z), family="dense_iaf", s_max=1.0
    )
    return {
        "step": int(step),
        "learning_rate": float(trainer.learning_rate_at(step).numpy()),
        **row,
    }


def run_stream(
    *,
    stream: FreshStream,
    policy: dict[str, Any],
    output_root: Path,
    program_start: float,
    cap_seconds: float,
    prior_charged_seconds: float,
    resume: bool,
) -> dict[str, Any]:
    output_dir = output_root / stream.label
    output_dir.mkdir(parents=True, exist_ok=resume)
    started = time.perf_counter()
    target = locked_ssl_lstm_posterior_target()
    trainer = NeuTraReverseKLTrainer(target, trainer_config(target, stream, policy))
    controller = NeuTraPlateauController(controller_config(policy))
    validation_z = validation_batch(stream)
    progress_path = output_dir / "progress.json"
    if resume:
        checkpoint_path = output_dir / "resource-stop-checkpoint.json"
        if not checkpoint_path.is_file() or not progress_path.is_file():
            raise ConfirmationError("resume checkpoint or progress artifact is missing")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        validate_joint_training_checkpoint(checkpoint)
        trainer.restore_state(checkpoint["trainer_state"])
        controller.restore_state(checkpoint["controller_state"])
        best_state = checkpoint["best_trainer_state"]
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("schema") != SCHEMA or progress.get("stream") != asdict(stream):
            raise ConfirmationError("resume progress artifact mismatch")
        history = list(progress.get("history", []))
        checkpoints = list(progress.get("checkpoints", []))
        if not history or best_state is None:
            raise ConfirmationError("resume progress lacks history or best state")
        initial = history[0]
        start_step = int(trainer.step.numpy()) + 1
        if not math.isclose(
            float(trainer.learning_rate_at(start_step).numpy()),
            controller.current_learning_rate,
            rel_tol=1.0e-6,
            abs_tol=0.0,
        ):
            raise ConfirmationError("resumed trainer/controller learning rate mismatch")
    else:
        initial = host_validation(trainer, validation_z, 0)
        initial_state = trainer.state_payload()
        initial_action = controller.observe(
            step=0,
            per_sample_loss=initial["per_sample_loss"],
            saturation_fraction=initial["saturation_fraction"],
            trainer_state_hash=initial_state["state_hash"],
        )
        if initial_action.kind != "initialize_best":
            raise ConfirmationError("initial validation did not initialize best state")
        best_state = initial_state
        history = [{**initial, "controller_action": initial_action.payload()}]
        checkpoints = []
        start_step = 1
        write_progress(progress_path, stream, history, checkpoints, trainer)
    warmup = step_batch(stream, start_step)
    hlo = trainer._compiled_train_step.experimental_get_compiler_ir(warmup)(  # noqa: SLF001
        stage="hlo"
    )
    if not isinstance(hlo, str) or "HloModule" not in hlo:
        raise ConfirmationError("confirmation did not expose XLA HLO")

    stop_reason = None
    for step in range(start_step, controller.config.max_steps + 1):
        if (
            prior_charged_seconds
            + time.perf_counter()
            - program_start
            + 60.0
            >= cap_seconds
        ):
            emergency = joint_training_checkpoint_payload(
                trainer_state=trainer.state_payload(),
                controller_state=controller.state_payload(),
                best_trainer_state=best_state,
            )
            write_json(
                output_dir / "resource-stop-checkpoint.json",
                emergency,
                replace=True,
            )
            write_progress(
                progress_path,
                stream,
                history,
                checkpoints,
                trainer,
                replace=True,
            )
            raise ResourceStop("shared confirmation GPU-time cap exhausted")
        trainer.train_step(warmup if step == 1 else step_batch(stream, step))
        if step % controller.config.validation_check_every != 0:
            continue
        validation = host_validation(trainer, validation_z, step)
        current_state = trainer.state_payload()
        action = controller.observe(
            step=step,
            per_sample_loss=validation["per_sample_loss"],
            saturation_fraction=validation["saturation_fraction"],
            trainer_state_hash=current_state["state_hash"],
        )
        if action.meaningful_improvement:
            best_state = current_state
        if action.should_reduce_learning_rate:
            trainer.set_learning_rate(action.current_learning_rate)
            current_state = trainer.state_payload()
        joint = joint_training_checkpoint_payload(
            trainer_state=current_state,
            controller_state=controller.state_payload(),
            best_trainer_state=best_state,
        )
        checkpoint_path = output_dir / f"checkpoint-{step:04d}.json"
        write_json(checkpoint_path, joint)
        checkpoints.append(
            {
                "step": step,
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(checkpoint_path),
                "checkpoint_hash": joint["checkpoint_hash"],
            }
        )
        history.append({**validation, "controller_action": action.payload()})
        write_progress(
            progress_path,
            stream,
            history,
            checkpoints,
            trainer,
            replace=True,
        )
        if action.should_stop:
            stop_reason = action.stop_reason
            break

    if stop_reason is None:
        raise ConfirmationError("confirmation ended without an explicit stop reason")
    terminal_state = trainer.state_payload()
    terminal_state_path = output_dir / "terminal-state.json"
    write_json(terminal_state_path, terminal_state)
    best_state_path = output_dir / "best-state.json"
    write_json(best_state_path, best_state)

    best_trainer = NeuTraReverseKLTrainer(target, trainer.config)
    best_trainer.restore_state(best_state)
    frozen = best_trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-plateau-confirmation-{stream.label}",
        target_signature=target.target_signature(),
    )
    frozen_path = output_dir / "best-frozen-payload.json"
    write_json(frozen_path, frozen)
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    probes = HELPERS._probe_diagnostics(target, loaded.transport)  # noqa: SLF001
    paired = HELPERS.paired_loss_upper_bound(  # noqa: SLF001
        initial["per_sample_loss"], list(controller.best_per_sample_loss)
    )
    vetoes = []
    if stop_reason == "scale_saturation_above_cap":
        vetoes.append("dense_scale_saturation_above_cap")
    if not probes["all_finite"]:
        vetoes.append("probe_nonfinite")
    if probes["roundtrip_max_abs"] > 1.0e-9:
        vetoes.append("roundtrip_residual_above_threshold")
    if probes["moderate_shell_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        vetoes.append("moderate_shell_missing_support")
    if paired["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_loss_improvement_not_established")
    return {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "decision": "CONFIRMATION_PASSED" if not vetoes else "CONFIRMATION_VETOED",
        "stream": asdict(stream),
        "trainer_config": trainer.config.manifest_payload(),
        "plateau_config": controller.config.manifest_payload(),
        "stop_reason": stop_reason,
        "best_step": controller.best_step,
        "terminal_step": int(trainer.step.numpy()),
        "learning_rate_reductions": controller.learning_rate_reductions,
        "history": history,
        "checkpoints": checkpoints,
        "vetoes": vetoes,
        "paired_best_minus_initial": paired,
        "best_probes": probes,
        "terminal_state_path": terminal_state_path.relative_to(ROOT).as_posix(),
        "terminal_state_sha256": sha256(terminal_state_path),
        "best_state_path": best_state_path.relative_to(ROOT).as_posix(),
        "best_state_sha256": sha256(best_state_path),
        "best_frozen_path": frozen_path.relative_to(ROOT).as_posix(),
        "best_frozen_sha256": sha256(frozen_path),
        "run_manifest": {
            "runtime_seconds": time.perf_counter() - started,
            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "device": sorted({variable.device for variable in trainer.variables}),
            "dtype": "float64",
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": [
            "transport-training confirmation only",
            "maximum-step stop is not a convergence claim",
            "no HMC, posterior-correctness, superiority, or scientific claim",
        ],
    }


def write_progress(
    path: Path,
    stream: FreshStream,
    history: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    trainer: NeuTraReverseKLTrainer,
    *,
    replace: bool = False,
) -> None:
    """Persist runner-only history beside the hash-bound numerical checkpoint."""

    write_json(
        path,
        {
            "schema": SCHEMA,
            "status": "RUNNING",
            "stream": asdict(stream),
            "trainer_step": int(trainer.step.numpy()),
            "trainer_state_hash": trainer.state_payload()["state_hash"],
            "history": history,
            "checkpoints": checkpoints,
        },
        replace=replace,
    )


def configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise ConfirmationError("confirmation requires a visible GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    return gpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-cap-seconds", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.gpu_cap_seconds) or args.gpu_cap_seconds <= 0.0:
        parser.error("--gpu-cap-seconds must be finite and positive")
    policy_path = (ROOT / args.policy).resolve()
    if not policy_path.is_relative_to(ROOT):
        parser.error("--policy must be inside the repository")
    policy = load_policy(policy_path)
    bindings = source_bindings(policy_path)
    gpus = configure_gpu()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=args.resume)
    previous_summary_path = output_root / "confirmation-summary.json"
    prior_charged_seconds = 0.0
    previous_results: dict[str, dict[str, Any]] = {}
    if args.resume:
        if not previous_summary_path.is_file():
            parser.error("--resume requires an existing confirmation summary")
        previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("policy_sha256") != sha256(policy_path):
            raise ConfirmationError("resume policy hash mismatch")
        prior_charged_seconds = float(
            previous_summary.get("run_manifest", {}).get("charged_gpu_seconds", 0.0)
        )
        if prior_charged_seconds >= args.gpu_cap_seconds:
            raise ConfirmationError("declared total confirmation cap is already exhausted")
        previous_results = {row["label"]: row for row in previous_summary.get("results", [])}
    program_start = time.perf_counter()
    results = []
    resource_stop = None
    for stream in FRESH_STREAMS:
        if stream.label in previous_results:
            row = previous_results[stream.label]
            result_path = ROOT / row["path"]
            if not result_path.is_file() or sha256(result_path) != row["sha256"]:
                raise ConfirmationError("completed stream result hash mismatch on resume")
            results.append(row)
            continue
        try:
            result = run_stream(
                stream=stream,
                policy=policy,
                output_root=output_root,
                program_start=program_start,
                cap_seconds=args.gpu_cap_seconds,
                prior_charged_seconds=prior_charged_seconds,
                resume=args.resume and (output_root / stream.label).exists(),
            )
        except ResourceStop as exc:
            resource_stop = str(exc)
            write_json(
                output_root / f"{stream.label}-resource-stop.json",
                {
                    "schema": SCHEMA,
                    "status": "RESOURCE_STOP",
                    "stream": asdict(stream),
                    "error": resource_stop,
                    "candidate_veto": False,
                    "scientific_interpretation": "none",
                },
                replace=args.resume,
            )
            break
        result_path = output_root / stream.label / "result.json"
        write_json(result_path, result)
        results.append(
            {
                "label": stream.label,
                "decision": result["decision"],
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(result_path),
            }
        )
    passed = (
        len(results) == len(FRESH_STREAMS)
        and all(row["decision"] == "CONFIRMATION_PASSED" for row in results)
    )
    summary = {
        "schema": SCHEMA,
        "status": "RESOURCE_STOP" if resource_stop else "COMPLETED",
        "decision": "FRESH_CONFIRMATION_PASSED" if passed else "FRESH_CONFIRMATION_NOT_PASSED",
        "results": results,
        "resource_stop": resource_stop,
        "policy_path": policy_path.relative_to(ROOT).as_posix(),
        "policy_sha256": sha256(policy_path),
        "source_bindings": bindings,
        "run_manifest": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "physical_gpus": [gpu.name for gpu in gpus],
            "charged_gpu_seconds": time.perf_counter() - program_start,
            "gpu_cap_seconds": args.gpu_cap_seconds,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": [
            "no HMC or posterior-correctness claim",
            "no statistical superiority or default-readiness claim",
        ],
    }
    summary["run_manifest"]["charged_gpu_seconds"] += prior_charged_seconds
    write_json(
        output_root / "confirmation-summary.json",
        summary,
        replace=args.resume,
    )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "charged_gpu_seconds": summary["run_manifest"]["charged_gpu_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
