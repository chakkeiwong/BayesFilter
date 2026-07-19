#!/usr/bin/env python3
"""Run the bounded paired `(32,32)` SSL-LSTM NeuTra capacity diagnostic."""

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

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_training import (
    DSGE_PAPER_TRAINING_BATCH_SIZE,
    SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
    NeuTraReverseKLTrainer,
    ssl_lstm_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (
    FREE_PARAMETER_NAMES,
    PRIOR_CENTER_VALUES,
    locked_ssl_lstm_posterior_target,
)


HELPER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_phase4_bounded_training_2026_07_14.py"
HELPER_SPEC = importlib.util.spec_from_file_location("capacity_phase4_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise RuntimeError("unable to load probe helpers")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HELPERS
HELPER_SPEC.loader.exec_module(HELPERS)

SCHEMA = "bayesfilter.ssl_lstm_neutra.capacity_32x32_diagnostic.v1"
PLAN_PATH = Path("docs/plans/bayesfilter-ssl-lstm-neutra-seed-instability-stabilization-repair-plan-2026-07-15.md")
DEFAULT_OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/capacity-32x32-diagnostic")
RESULT_NOTE_PATH = Path("docs/plans/bayesfilter-ssl-lstm-neutra-capacity-32x32-diagnostic-result-2026-07-15.md")
BASELINE_A_PATH = Path("docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training/seed-a/result.json")
BASELINE_B_PATH = Path("docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training/seed-b/result.json")
BASELINE_SHA256 = {
    BASELINE_A_PATH: "6b0b5ff525e9081870b707784715b6e31e1c8f47d8fec59ff7b96a1bc7bc8186",
    BASELINE_B_PATH: "3cfd5f1d936c99d1f42e4d7f5b4900da9d49403bcac902f55e912ac7b04ab40c",
}
STEPS = 1200
CHECKPOINT_EVERY = 100
VALIDATION_EVERY = 100
VALIDATION_BATCH_SIZE = 64
PER_STREAM_SECONDS = 4500.0
SHARED_SECONDS = 9000.0
INVERSE_RADIUS_MAX = 4.30
SATURATION_MAX = 0.05


@dataclass(frozen=True)
class Stream:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]


STREAMS = (
    Stream("seed-a", (20260715, 4101), (20260715, 5101), (20260715, 5201)),
    Stream("seed-b", (20260715, 4102), (20260715, 5102), (20260715, 5202)),
)


class DiagnosticError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def source_bindings() -> dict[str, Any]:
    for path, expected in BASELINE_SHA256.items():
        if sha256(ROOT / path) != expected:
            raise DiagnosticError(f"immutable baseline hash drift: {path}")
    paths = {
        "runner": Path(__file__).resolve().relative_to(ROOT),
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
        "capacity_tests": Path("tests/test_ssl_lstm_neutra_capacity_32x32_diagnostic.py"),
        "parity_tests": Path("tests/test_neutra_dsge_procedure_parity.py"),
        "probe_helpers": HELPER_PATH.relative_to(ROOT),
        "plan": PLAN_PATH,
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {role: sha256(ROOT / path) for role, path in paths.items()},
        "source_paths": {role: path.as_posix() for role, path in paths.items()},
        "baseline_sha256": {path.as_posix(): value for path, value in BASELINE_SHA256.items()},
    }


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DiagnosticError(f"output exists: {path}")
    path.write_bytes(canonical(payload))


def classify_program(results: list[dict[str, Any]], wall_seconds: float) -> str:
    if not math.isfinite(float(wall_seconds)) or float(wall_seconds) > SHARED_SECONDS:
        return "INVALID_EVIDENCE"
    decisions = [row.get("decision") for row in results]
    if "INVALID_EVIDENCE" in decisions or len(results) != len(STREAMS):
        return "INVALID_EVIDENCE"
    if decisions == ["R2_CAPACITY_REPAIR_NOMINATED"] * len(STREAMS):
        return "R2_CAPACITY_REPAIR_NOMINATED"
    return "R2_CAPACITY_REPAIR_NOT_NOMINATED"


def stream_config(target: Any, stream: Stream) -> Any:
    return ssl_lstm_capacity_neutra_config(
        dimension=4,
        fixed_translation=PRIOR_CENTER_VALUES,
        target_parameter_names=FREE_PARAMETER_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        initialization_seed=stream.initialization_seed,
        jit_compile=True,
    )


def step_batch(stream: Stream, step: int) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(tf.constant(stream.training_seed, tf.int32), step)
    return tf.random.stateless_normal(
        (DSGE_PAPER_TRAINING_BATCH_SIZE, 4), seed=seed, dtype=tf.float64
    )


def validation_batch(stream: Stream) -> tf.Tensor:
    return tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4), seed=tf.constant(stream.validation_seed, tf.int32), dtype=tf.float64
    )


def checkpoint_payload(trainer: Any, target: Any, step: int) -> dict[str, Any]:
    payload = trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-capacity-32x32-diagnostic-{step}",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=target.target_signature())
    probes = HELPERS._probe_diagnostics(target, loaded.transport)  # noqa: SLF001
    return {
        "step": step,
        "state_hash": trainer.state_payload()["state_hash"],
        "frozen_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "procedure": payload["procedure"],
        "hidden_layers": [
            component["hidden_layers"]
            for component in payload["components"]
            if component["kind"] == "dense_autoregressive_iaf"
        ],
        "probes": probes,
    }


def stage_saturation(trainer: Any, z: tf.Tensor) -> list[dict[str, float]]:
    values = z
    rows = []
    for component in trainer.transport.components:
        if component.__class__.__name__ == "_TrainableDenseIAF":
            scale_log = component.scale_log(values)
            rows.append(
                {
                    "fraction": float(
                        tf.reduce_mean(
                            tf.cast(tf.abs(scale_log) >= 0.95 * trainer.config.s_max, tf.float64)
                        ).numpy()
                    ),
                    "minimum": float(tf.reduce_min(scale_log).numpy()),
                    "maximum": float(tf.reduce_max(scale_log).numpy()),
                }
            )
        values, _ = component.forward_and_logdet(values)
    if len(rows) != 3:
        raise DiagnosticError("capacity diagnostic expected exactly three IAF stages")
    return rows


def run_stream(
    stream: Stream,
    output_root: Path,
    program_start: float,
    bindings: dict[str, Any],
    physical_gpus: list[Any],
) -> dict[str, Any]:
    output_dir = output_root / stream.label
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target = locked_ssl_lstm_posterior_target()
    config = stream_config(target, stream)
    trainer = NeuTraReverseKLTrainer(target, config)
    parameter_count = sum(int(tf.size(variable).numpy()) for variable in trainer.variables)
    if parameter_count != 4440:
        raise DiagnosticError(f"capacity parameter-count mismatch: {parameter_count}")
    validation_z = validation_batch(stream)
    initial = HELPERS._host_validation(  # noqa: SLF001
        trainer.validation_batch(validation_z), family="dense_iaf", s_max=config.s_max
    )
    history: list[dict[str, Any]] = [{"step": 0, **initial}]
    checkpoints: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    warmup = step_batch(stream, 1)
    hlo = trainer._compiled_train_step.experimental_get_compiler_ir(warmup)(stage="hlo")  # noqa: SLF001
    if not isinstance(hlo, str) or "HloModule" not in hlo:
        raise DiagnosticError("capacity diagnostic did not expose XLA HLO")

    for step in range(1, STEPS + 1):
        elapsed = time.perf_counter() - started
        if elapsed + 120.0 >= PER_STREAM_SECONDS or time.perf_counter() - program_start + 120.0 >= SHARED_SECONDS:
            raise DiagnosticError("capacity diagnostic resource stop")
        result = trainer.train_step(warmup if step == 1 else step_batch(stream, step))
        row = HELPERS._host_step(result)  # noqa: SLF001
        rows.append(row)
        if step % CHECKPOINT_EVERY == 0:
            state = trainer.state_payload()
            checkpoint_path = output_dir / f"checkpoint-{step:04d}.json"
            write_json(checkpoint_path, state)
            cp = checkpoint_payload(trainer, target, step)
            cp["checkpoint_path"] = checkpoint_path.as_posix()
            cp["checkpoint_sha256"] = sha256(checkpoint_path)
            checkpoints.append(cp)
            validation = HELPERS._host_validation(  # noqa: SLF001
                trainer.validation_batch(validation_z), family="dense_iaf", s_max=config.s_max
            )
            validation["stage_saturation"] = stage_saturation(trainer, validation_z)
            history.append({"step": step, **validation})
            if not all(math.isfinite(float(value)) for value in (validation["mean_loss"], validation["saturation_fraction"])):
                raise DiagnosticError(f"nonfinite validation at step {step}")
            if validation["saturation_fraction"] > SATURATION_MAX:
                return {
                    "status": "COMPLETED_R2_VETO",
                    "decision": "R2_CAPACITY_REPAIR_NOT_NOMINATED",
                    "veto": "dense_scale_saturation_above_cap",
                    "veto_step": step,
                    "candidate": {"label": stream.label, "config": config.manifest_payload()},
                    "training": {"history": rows, "validation": history, "checkpoints": checkpoints},
                    "source_bindings": bindings,
                    "run_manifest": {
                        "wall_time_seconds": time.perf_counter() - started,
                        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                        "physical_gpus": [device.name for device in physical_gpus],
                        "parameter_count": parameter_count,
                    },
                }

    final = history[-1]
    initial_loss = initial["per_sample_loss"]
    final_loss = final["per_sample_loss"]
    interval = HELPERS.paired_loss_upper_bound(initial_loss, final_loss)
    final_probe = checkpoints[-1]["probes"]
    vetoes = []
    if final_probe["moderate_shell_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        vetoes.append("moderate_shell_missing_support")
    if interval["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_loss_improvement_not_established")
    return {
        "status": "COMPLETED",
        "decision": "R2_CAPACITY_REPAIR_NOMINATED" if not vetoes else "R2_CAPACITY_REPAIR_NOT_NOMINATED",
        "vetoes": vetoes,
        "candidate": {"label": stream.label, "family": config.family, "config": config.manifest_payload()},
        "training": {"history": rows, "validation": history, "checkpoints": checkpoints},
        "paired_final_minus_initial": interval,
        "final_probe": final_probe,
        "source_bindings": bindings,
        "run_manifest": {
            "wall_time_seconds": time.perf_counter() - started,
            "charged_gpu_seconds": time.perf_counter() - started,
            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "hlo_characters": len(hlo),
            "device": sorted({v.device for v in trainer.variables}),
            "physical_gpus": [device.name for device in physical_gpus],
            "parameter_count": parameter_count,
            "dtype": "float64",
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = ROOT / args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise DiagnosticError(f"diagnostic output root is not fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    bindings = source_bindings()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise DiagnosticError("capacity diagnostic requires a visible GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    started = time.perf_counter()
    start_payload = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": now(),
        "streams": [stream.__dict__ for stream in STREAMS],
        "shared_gpu_cap_seconds": SHARED_SECONDS,
        "per_stream_gpu_cap_seconds": PER_STREAM_SECONDS,
        "plan": PLAN_PATH.as_posix(),
        "family": SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
        "hidden_layers": [32, 32],
        "physical_gpus": [device.name for device in gpus],
        "source_bindings": bindings,
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "output_root": args.output_root.as_posix(),
        "result_note": RESULT_NOTE_PATH.as_posix(),
    }
    write_json(output_root / "diagnostic-start.json", start_payload)
    results = []
    for stream in STREAMS:
        if time.perf_counter() - started + PER_STREAM_SECONDS > SHARED_SECONDS and results:
            raise DiagnosticError("insufficient shared budget for second stream")
        try:
            result = run_stream(stream, output_root, started, bindings, gpus)
        except Exception as error:
            failure = {
                "schema": SCHEMA,
                "status": "INVALID_HARD_VETO",
                "decision": "INVALID_EVIDENCE",
                "candidate": stream.__dict__,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "source_bindings": bindings,
                "nonclaims": ["failure receipt only; no capacity or scientific conclusion"],
            }
            failure_path = output_root / stream.label / "failure.json"
            write_json(failure_path, failure)
            results.append({"label": stream.label, "decision": "INVALID_EVIDENCE", "path": failure_path.as_posix(), "sha256": sha256(failure_path)})
            break
        result_path = output_root / stream.label / "result.json"
        write_json(result_path, result)
        results.append({"label": stream.label, "decision": result["decision"], "path": result_path.as_posix(), "sha256": sha256(result_path)})
    wall = time.perf_counter() - started
    program = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "decision": classify_program(results, wall),
        "results": results,
        "budget": {"charged_gpu_seconds": wall, "shared_gpu_cap_seconds": SHARED_SECONDS, "within_cap": wall <= SHARED_SECONDS},
        "source_bindings": bindings,
        "run_manifest": {
            "git_commit": bindings["git_commit"],
            "git_dirty": bindings["git_dirty"],
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "completed_at_utc": now(),
            "physical_gpus": [device.name for device in gpus],
            "dtype": "float64",
            "tf32": True,
            "jit_compile": True,
            "soft_device_placement_during_run": False,
            "output_root": args.output_root.as_posix(),
            "plan": PLAN_PATH.as_posix(),
            "result_note": RESULT_NOTE_PATH.as_posix(),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
    }
    write_json(output_root / "diagnostic-result.json", program)


if __name__ == "__main__":
    main()
