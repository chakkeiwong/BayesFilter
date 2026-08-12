#!/usr/bin/env python3
"""Bounded q=20 localization of host-callback versus GPU-native eigh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mode() -> str:
    if "--mode" not in sys.argv:
        return "inspect"
    index = sys.argv.index("--mode")
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else "inspect"


MODE = _mode()
if MODE == "gpu":
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("GPU localization requires TF_FORCE_GPU_ALLOW_GROWTH=true")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise RuntimeError("GPU localization requires an explicit visible GPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from bayesfilter.inference.neutra_batching import (
    bound_batch_native_neutra_training_target,
    require_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_training import (
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES, PRIOR_CENTER
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=MODE == "gpu")
if MODE == "gpu":
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)


SCHEMA = "bayesfilter.ssl_lstm.q20_gpu_native_eigh_localization.v1"
PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-gpu-native-eigh-localization-plan-2026-07-31.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-gpu-native-eigh-localization-2026-07-31/r1"
)
SOURCE_PATHS = {
    "runner": SCRIPT,
    "plan": PLAN,
    "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"),
    "filter": Path("bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"),
    "custom_op": Path("bayesfilter/ops/symmetric_sylvester_op.cc"),
    "binding": Path("bayesfilter/inference/neutra_batching.py"),
    "trainer": Path("bayesfilter/inference/neutra_training.py"),
    "memory_policy": Path("bayesfilter/runtime/gpu_memory_policy.py"),
}
SOURCE_SHA256 = {
    key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for key, path in SOURCE_PATHS.items()
}
MATERIAL_CAP_SECONDS = 1800.0
AUTHORIZED_REMAINING_SECONDS = 2152.4224067549985
RESERVED_NONMATERIAL_SECONDS = AUTHORIZED_REMAINING_SECONDS - MATERIAL_CAP_SECONDS
CUSTOM_BASELINE_MEDIAN_SECONDS = 565.9442223530059
VALUE_ATOL = 1.0e-8
SCORE_ATOL = 1.0e-7
SCORE_RTOL = 1.0e-7


class LocalizationError(RuntimeError):
    pass


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
        raise LocalizationError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def allocator_payload() -> Mapping[str, Mapping[str, int]]:
    return {
        device.name: {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info(
                f"GPU:{index}"
            ).items()
        }
        for index, device in enumerate(tf.config.list_logical_devices("GPU"))
    }


def device_payload() -> Mapping[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    growth = {
        device.name: bool(tf.config.experimental.get_memory_growth(device))
        for device in physical
    }
    if MODE == "gpu" and (not logical or not growth or not all(growth.values())):
        raise LocalizationError("trusted GPU memory growth was not verified")
    return {
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "memory_growth": growth,
        "memory_policy": GPU_MEMORY_POLICY,
        "allocator_memory_bytes": allocator_payload(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


class MaterialBudget:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 300.0) -> None:
        if self.elapsed + reserve_seconds >= MATERIAL_CAP_SECONDS:
            self.persist("RESOURCE_STOP")
            raise LocalizationError("localization material cap exhausted")

    def persist(self, status: str) -> None:
        elapsed = self.elapsed
        write_json(
            ROOT / self.output_root / "material-budget-ledger.json",
            {
                "schema": f"{SCHEMA}.material_budget_ledger",
                "status": status,
                "cap_seconds": MATERIAL_CAP_SECONDS,
                "charged_seconds": elapsed,
                "remaining_seconds": MATERIAL_CAP_SECONDS - elapsed,
            },
            replace=True,
        )


class Progress:
    def __init__(self, output_root: Path, budget: MaterialBudget) -> None:
        self.path = ROOT / output_root / "progress.json"
        self.budget = budget
        self.payload: dict[str, Any] = {
            "schema": f"{SCHEMA}.progress",
            "status": "RUNNING",
            "active_operation": None,
            "operations": [],
            "source_sha256": SOURCE_SHA256,
        }
        self._write()

    def _write(self) -> None:
        self.payload["process_elapsed_seconds"] = self.budget.elapsed
        self.payload["allocator_memory_bytes"] = allocator_payload()
        write_json(self.path, self.payload, replace=True)
        self.budget.persist("RUNNING")

    def run(self, name: str, operation: Callable[[], Any]) -> Any:
        self.budget.require()
        started = time.perf_counter()
        self.payload["active_operation"] = {
            "name": name,
            "started_unix_seconds": time.time(),
        }
        self._write()
        result = operation()
        self.payload["operations"].append(
            {
                "name": name,
                "duration_seconds": time.perf_counter() - started,
                "result": result,
            }
        )
        self.payload["active_operation"] = None
        self._write()
        return result


def target_program(backend: str) -> tuple[Any, Callable[[tf.Tensor], Any]]:
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend=backend
    )

    @tf.function(
        input_signature=[tf.TensorSpec([2, 4], tf.float64)],
        jit_compile=True,
        reduce_retracing=False,
    )
    def program(theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
        return (
            value,
            score,
            status["status_code"],
            status["valid_pre_regularized_score"],
            status["floor_count_value"],
            status["min_innovation_eigenvalue"],
        )

    return target, program


def target_call_payload(rows: tuple[tf.Tensor, ...]) -> Mapping[str, Any]:
    value, score, status_code, valid, floors, min_eigenvalue = rows
    hard_valid = bool(
        tf.reduce_all(
            tf.logical_and(
                valid,
                tf.logical_and(
                    tf.equal(status_code, 0), tf.equal(floors, 0)
                ),
            )
        ).numpy()
    )
    finite = bool(
        tf.reduce_all(
            tf.concat(
                (
                    tf.reshape(tf.math.is_finite(value), [-1]),
                    tf.reshape(tf.math.is_finite(score), [-1]),
                    tf.reshape(tf.math.is_finite(min_eigenvalue), [-1]),
                ),
                axis=0,
            )
        ).numpy()
    )
    if not finite or not hard_valid:
        raise LocalizationError("target call failed finite/status screen")
    return {
        "value": [float(x) for x in tf.reshape(value, [-1]).numpy()],
        "score": [float(x) for x in tf.reshape(score, [-1]).numpy()],
        "all_finite": finite,
        "all_hard_valid": hard_valid,
        "floor_count_total": int(tf.reduce_sum(floors).numpy()),
        "min_innovation_eigenvalue": float(tf.reduce_min(min_eigenvalue).numpy()),
    }


def parity_payload(
    custom_rows: tuple[tf.Tensor, ...], native_rows: tuple[tf.Tensor, ...]
) -> Mapping[str, Any]:
    custom_value, custom_score = custom_rows[:2]
    native_value, native_score = native_rows[:2]
    value_max_abs = float(tf.reduce_max(tf.abs(native_value - custom_value)).numpy())
    score_abs = tf.abs(native_score - custom_score)
    score_scale = tf.maximum(tf.abs(custom_score), tf.constant(1.0, tf.float64))
    score_max_abs = float(tf.reduce_max(score_abs).numpy())
    score_max_rel = float(tf.reduce_max(score_abs / score_scale).numpy())
    passed = value_max_abs <= VALUE_ATOL and bool(
        tf.reduce_all(
            tf.logical_or(score_abs <= SCORE_ATOL, score_abs / score_scale <= SCORE_RTOL)
        ).numpy()
    )
    payload = {
        "passed": passed,
        "value_max_abs": value_max_abs,
        "score_max_abs": score_max_abs,
        "score_max_relative_scaled": score_max_rel,
        "value_atol": VALUE_ATOL,
        "score_atol": SCORE_ATOL,
        "score_rtol": SCORE_RTOL,
    }
    if not passed:
        raise LocalizationError("strict TensorFlow backend parity vetoed")
    return payload


def make_native_trainer(native_owner: Any) -> NeuTraReverseKLTrainer:
    binding = require_batch_native_neutra_target(
        native_owner,
        target_signature=native_owner.target_signature(),
        batch_size=100,
    )
    target = bound_batch_native_neutra_training_target(binding)
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(x) for x in PRIOR_CENTER),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=2.0e-4,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260731, 10101),
        jit_compile=True,
    )
    return NeuTraReverseKLTrainer(target, config)


def stateless_batch(step: int) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant((20260731, 10201), tf.int32), int(step)
    )
    return tf.random.stateless_normal([100, 4], seed, dtype=tf.float64)


def step_payload(step: Any) -> Mapping[str, Any]:
    return {
        "step": int(step.step.numpy()),
        "loss": float(step.loss.numpy()),
        "gradient_norm": float(step.gradient_norm.numpy()),
        "clipped_gradient_norm": float(step.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(step.clipping_applied.numpy()),
    }


def run_gpu(args: argparse.Namespace) -> Mapping[str, Any]:
    budget = MaterialBudget(args.output_root)
    progress = Progress(args.output_root, budget)
    state: dict[str, Any] = {}
    try:
        def construct() -> Mapping[str, Any]:
            custom, custom_program = target_program("compiled_custom_op")
            native, native_program = target_program("tensorflow_eigh_strict")
            if custom.target_signature() != native.target_signature():
                raise LocalizationError("backend changed the target signature")
            state.update(
                custom=custom,
                custom_program=custom_program,
                native=native,
                native_program=native_program,
            )
            return {
                "target_signature": custom.target_signature(),
                "custom_adapter_signature": custom.adapter_signature(),
                "native_adapter_signature": native.adapter_signature(),
            }

        progress.run("construct_targets", construct)
        theta = tf.stack(
            (
                PRIOR_CENTER,
                PRIOR_CENTER + tf.constant((0.01, -0.02, 0.015, -0.01), tf.float64),
            )
        )
        custom_first = progress.run(
            "custom_batch2_first",
            lambda: target_call_payload(
                state.setdefault("custom_first_rows", state["custom_program"](theta))
            ),
        )
        del custom_first
        progress.run(
            "custom_batch2_warm",
            lambda: target_call_payload(state["custom_program"](theta)),
        )
        progress.run(
            "native_batch2_first",
            lambda: target_call_payload(
                state.setdefault("native_first_rows", state["native_program"](theta))
            ),
        )
        progress.run(
            "native_batch2_warm",
            lambda: target_call_payload(state["native_program"](theta)),
        )
        progress.run(
            "backend_parity",
            lambda: parity_payload(
                state["custom_first_rows"], state["native_first_rows"]
            ),
        )

        def construct_trainer() -> Mapping[str, Any]:
            trainer = make_native_trainer(state["native"])
            state["trainer"] = trainer
            return {
                "variable_devices": sorted({v.device for v in trainer.variables}),
                "trainable_variable_count": len(trainer.variables),
            }

        progress.run("construct_native_trainer", construct_trainer)
        trainer = state["trainer"]
        for index in range(1, 4):
            progress.run(
                f"native_optimizer_update_{index}",
                lambda index=index: step_payload(trainer.train_step(stateless_batch(index))),
            )

        def hlo() -> Mapping[str, Any]:
            text = trainer._compiled_train_step.experimental_get_compiler_ir(
                stateless_batch(99)
            )(stage="hlo")
            encoded = text if isinstance(text, bytes) else str(text).encode("utf-8")
            if not encoded or b"ENTRY" not in encoded:
                raise LocalizationError("strict backend HLO is missing an ENTRY")
            return {
                "hlo_sha256": hashlib.sha256(encoded).hexdigest(),
                "hlo_byte_count": len(encoded),
            }

        progress.run("native_hlo_extraction", hlo)
        progress.payload["status"] = "GPU_NATIVE_LOCALIZATION_COMPLETED"
        progress._write()
        operations = {
            row["name"]: row for row in progress.payload["operations"]
        }
        warm = [
            float(operations[f"native_optimizer_update_{i}"]["duration_seconds"])
            for i in (2, 3)
        ]
        result = {
            **progress.payload,
            "schema": f"{SCHEMA}.result",
            "status": "GPU_NATIVE_LOCALIZATION_COMPLETED",
            "native_warm_update_seconds": warm,
            "native_warm_update_median_seconds": statistics.median(warm),
            "historical_custom_warm_update_median_seconds": CUSTOM_BASELINE_MEDIAN_SECONDS,
            "descriptive_speed_ratio": CUSTOM_BASELINE_MEDIAN_SECONDS / statistics.median(warm),
            "run_manifest": {
                "git_commit": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain")),
                "command": " ".join(sys.argv),
                "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "device": device_payload(),
                "dtype": "float64",
                "jit_compile": True,
                "source_paths": {k: v.as_posix() for k, v in SOURCE_PATHS.items()},
                "source_sha256": SOURCE_SHA256,
                "plan": PLAN.as_posix(),
                "authorized_remaining_seconds": AUTHORIZED_REMAINING_SECONDS,
                "material_cap_seconds": MATERIAL_CAP_SECONDS,
                "reserved_nonmaterial_seconds": RESERVED_NONMATERIAL_SECONDS,
            },
            "nonclaims": [
                "localization and repair nomination only",
                "no campaign budget, tuning, training quality, convergence, HMC, posterior, or default claim",
            ],
        }
        write_json(ROOT / args.output_root / "result.json", result)
        budget.persist(result["status"])
        return result
    except Exception:
        budget.persist("FAILED_ATTEMPT")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gpu",), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.is_absolute() or not (ROOT / args.output_root).resolve().is_relative_to(ROOT):
        raise LocalizationError("output root must be repository-relative")
    result = run_gpu(args)
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()

