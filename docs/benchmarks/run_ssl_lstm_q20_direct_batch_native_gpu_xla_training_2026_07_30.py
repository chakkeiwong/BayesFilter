#!/usr/bin/env python3
"""Direct batch-native q=20 NeuTra GPU/XLA training campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mode_from_argv() -> str:
    if "--mode" not in sys.argv:
        return "contract"
    index = sys.argv.index("--mode")
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else "contract"


MODE_AT_IMPORT = _mode_from_argv()
MATERIAL_MODES = {"mechanics", "tuning-arm", "final-stream"}
if MODE_AT_IMPORT not in MATERIAL_MODES:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("material modes require TF_FORCE_GPU_ALLOW_GROWTH=true")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise RuntimeError("material modes require an explicit visible physical GPU")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf,
    require_gpu=MODE_AT_IMPORT in MATERIAL_MODES,
)
if MODE_AT_IMPORT in MATERIAL_MODES:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)


from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_batching import (  # noqa: E402
    bound_batch_native_neutra_training_target,
    require_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
    ssl_lstm_wide_capacity_neutra_config,
)
from bayesfilter.inference.neutra_training_control import (  # noqa: E402
    NeuTraPlateauConfig,
    NeuTraPlateauController,
    joint_training_checkpoint_payload,
    validate_joint_training_checkpoint,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    FREE_NAMES,
    PRIOR_CENTER,
)


SCHEMA = "bayesfilter.ssl_lstm.q20_direct_batch_native_gpu_xla_training.v1"
TUNING_SCHEMA = "bayesfilter.ssl_lstm.q20_direct_gpu_xla_tuning_selection.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-plan-2026-07-30.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
SOURCE_PATHS = {
    "runner": SCRIPT,
    "plan": PLAN,
    "target": Path(
        "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
    ),
    "scalar_target": Path(
        "bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"
    ),
    "binding": Path("bayesfilter/inference/neutra_batching.py"),
    "trainer": Path("bayesfilter/inference/neutra_training.py"),
    "controller": Path("bayesfilter/inference/neutra_training_control.py"),
    "memory_policy": Path("bayesfilter/runtime/gpu_memory_policy.py"),
}
IMPORTED_SOURCE_SHA256 = {
    key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for key, path in SOURCE_PATHS.items()
}
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-direct-batch-native-gpu-xla-training-2026-07-30/r1"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
BATCH_SIZE = 100
VALIDATION_SIZE = 64
AUDIT_SIZE = 256
TUNING_STEPS = 100
FINAL_MAX_STEPS = 1000
CHECK_EVERY = 100
SHELL_RADIUS = 4.0
SHELL_RADIUS_MAX = 4.30
ROUNDTRIP_MAX = 1.0e-9
MATERIAL_CAP_SECONDS = 18000.0
AUDIT_CRITICAL_VALUE = 1.6508515817258696


class CampaignError(RuntimeError):
    pass


class CampaignBudget:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.started = time.perf_counter()
        ledger_path = ROOT / output_root / "material-budget-ledger.json"
        if ledger_path.exists():
            ledger = read_json(ledger_path)
            if ledger.get("schema") != f"{SCHEMA}.material_budget_ledger":
                raise CampaignError("material budget ledger schema mismatch")
            self.prior_seconds = float(ledger.get("charged_seconds", -1.0))
            self.prior_attempts = list(ledger.get("attempts", ()))
        else:
            self.prior_seconds = 0.0
            self.prior_attempts = []
        if (
            not math.isfinite(self.prior_seconds)
            or self.prior_seconds < 0.0
            or self.prior_seconds >= MATERIAL_CAP_SECONDS
        ):
            raise CampaignError("material campaign budget is exhausted or invalid")

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 0.0) -> None:
        if self.elapsed + float(reserve_seconds) >= MATERIAL_CAP_SECONDS:
            self.persist(status="RESOURCE_STOP")
            raise CampaignError("declared material campaign cap exhausted")

    def persist(self, *, status: str) -> None:
        process_seconds = time.perf_counter() - self.started
        attempt = {
            "mode": MODE_AT_IMPORT,
            "command": " ".join(sys.argv),
            "process_seconds": process_seconds,
            "status": str(status),
        }
        payload = {
            "schema": f"{SCHEMA}.material_budget_ledger",
            "cap_seconds": MATERIAL_CAP_SECONDS,
            "charged_seconds": self.prior_seconds + process_seconds,
            "attempts": [*self.prior_attempts, attempt],
        }
        write_json(
            ROOT / self.output_root / "material-budget-ledger.json",
            payload,
            replace=True,
        )


@dataclass(frozen=True)
class Stream:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]
    audit_seed: tuple[int, int] | None = None


MECHANICS_STREAM = Stream(
    "mechanics", (20260730, 5101), (20260730, 5201), (20260730, 5301)
)
TUNING_STREAM = Stream(
    "tuning", (20260730, 6101), (20260730, 6201), (20260730, 6301)
)
FINAL_STREAMS = {
    "seed-a": Stream(
        "seed-a",
        (20260730, 7101),
        (20260730, 7201),
        (20260730, 7301),
        (20260730, 7401),
    ),
    "seed-b": Stream(
        "seed-b",
        (20260730, 7102),
        (20260730, 7202),
        (20260730, 7302),
        (20260730, 7402),
    ),
}
TUNING_CANDIDATES = {
    "arch32-lr2e4": ((32, 32), 2.0e-4),
    "arch32-lr4e4": ((32, 32), 4.0e-4),
    "arch64-lr2e4": ((64, 64), 2.0e-4),
    "arch64-lr4e4": ((64, 64), 4.0e-4),
}


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
        raise CampaignError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, Mapping):
        raise CampaignError(f"artifact is not a mapping: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def batch(seed: tuple[int, int], fold: int, size: int) -> tf.Tensor:
    folded = tf.random.experimental.stateless_fold_in(
        tf.constant(seed, tf.int32), int(fold)
    )
    return tf.random.stateless_normal(
        (int(size), 4), seed=folded, dtype=tf.float64
    )


def allocator_memory() -> Mapping[str, Mapping[str, int]]:
    return {
        device.name: {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info(
                f"GPU:{index}"
            ).items()
        }
        for index, device in enumerate(tf.config.list_logical_devices("GPU"))
    }


def device_manifest() -> Mapping[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    if MODE_AT_IMPORT in MATERIAL_MODES and not logical:
        raise CampaignError("material mode has no logical GPU")
    growth = {
        device.name: bool(tf.config.experimental.get_memory_growth(device))
        for device in physical
    }
    if MODE_AT_IMPORT in MATERIAL_MODES and (not growth or not all(growth.values())):
        raise CampaignError("GPU memory growth was not verified")
    return {
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "memory_growth": growth,
        "memory_policy": GPU_MEMORY_POLICY,
        "allocator_memory_bytes": allocator_memory(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        "trust_basis": TRUST_BASIS,
    }


def run_manifest(args: argparse.Namespace, started: float) -> Mapping[str, Any]:
    return {
        "schema": f"{SCHEMA}.run_manifest",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "device": device_manifest(),
        "dtype": "float64",
        "jit_compile": True,
        "xla_required": True,
        "batch_size": BATCH_SIZE,
        "target_backend": "direct_batch_native_compiled_custom_op",
        "sample_wise_loop_used": False,
        "scalar_fallback_used": False,
        "row_mapped_scalar_target_used": False,
        "source_paths": {key: value.as_posix() for key, value in SOURCE_PATHS.items()},
        "source_sha256_at_import": IMPORTED_SOURCE_SHA256,
        "plan": PLAN.as_posix(),
        "output_root": args.output_root.as_posix(),
        "mode": args.mode,
        "candidate": args.candidate,
        "stream": args.stream,
        "declared_material_cap_seconds": MATERIAL_CAP_SECONDS,
        "process_wall_seconds": time.perf_counter() - started,
    }


def target_and_binding() -> tuple[Any, Any, Any]:
    owner = batch_native_complexity_posterior_target(
        20,
        jit_compile=True,
        principal_sqrt_backend="compiled_custom_op",
    )
    binding = require_batch_native_neutra_target(
        owner,
        target_signature=owner.target_signature(),
        batch_size=BATCH_SIZE,
    )
    return owner, binding, bound_batch_native_neutra_training_target(binding)


def make_trainer(
    target: Any,
    *,
    architecture: tuple[int, int],
    learning_rate: float,
    initialization_seed: tuple[int, int],
) -> NeuTraReverseKLTrainer:
    factory = (
        ssl_lstm_tuned_capacity_neutra_config
        if architecture == (32, 32)
        else ssl_lstm_wide_capacity_neutra_config
    )
    config = factory(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=float(learning_rate),
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=initialization_seed,
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, config)
    variable_devices = sorted({variable.device for variable in trainer.variables})
    if MODE_AT_IMPORT in MATERIAL_MODES and not variable_devices:
        raise CampaignError("trainer has no trainable variable placement")
    return trainer


def host_step(result: Any) -> Mapping[str, Any]:
    return {
        "step": int(result.step.numpy()),
        "loss": float(result.loss.numpy()),
        "surrogate": float(result.surrogate.numpy()),
        "target_value_mean": float(result.target_value_mean.numpy()),
        "logdet_mean": float(result.logdet_mean.numpy()),
        "gradient_norm": float(result.gradient_norm.numpy()),
        "clipped_gradient_norm": float(result.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(result.clipping_applied.numpy()),
    }


def status_payload(target: Any, theta: tf.Tensor) -> Mapping[str, Any]:
    value, score, status = target.batch_value_score_status(theta)
    hard_valid = tf.convert_to_tensor(status["hard_valid_for_training"], tf.bool)
    if not bool(tf.reduce_all(hard_valid).numpy()):
        raise CampaignError("target status rejected one or more rows")
    return {
        "row_count": int(theta.shape[0]),
        "all_hard_valid_for_training": True,
        "status_code_max": int(tf.reduce_max(status["status_code"]).numpy()),
        "valid_pre_regularized_score_all": bool(
            tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        ),
        "floor_count_total": int(tf.reduce_sum(status["floor_count_value"]).numpy()),
        "min_innovation_eigenvalue": float(
            tf.reduce_min(status["min_innovation_eigenvalue"]).numpy()
        ),
        "condition_estimate_available": bool(
            tf.reduce_all(
                status["innovation_condition_estimate_available"]
            ).numpy()
        ),
        "value_all_finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
        "score_all_finite": bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
    }


def validation_payload(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    z: tf.Tensor,
    *,
    step: int,
    learning_rate: float,
) -> Mapping[str, Any]:
    validation = trainer.validation_batch(z)
    status = status_payload(target, validation.theta)
    losses = [float(value) for value in validation.per_sample_loss.numpy().tolist()]
    scale_log = validation.scale_log.numpy().tolist()
    scale_values = [value for row in scale_log for value in row]
    hidden = validation.hidden_preactivations.numpy().tolist()
    hidden_values = [
        value for row in hidden for stage in row for layer in stage for value in layer
    ]
    return {
        "step": int(step),
        "learning_rate": float(learning_rate),
        "per_sample_loss": losses,
        "mean_loss": math.fsum(losses) / len(losses),
        "saturation_fraction": (
            sum(abs(value) >= 0.95 for value in scale_values) / len(scale_values)
        ),
        "scale_log_min": min(scale_values),
        "scale_log_max": max(scale_values),
        "hidden_abs_tail_fraction": (
            sum(abs(value) >= 5.0 for value in hidden_values) / len(hidden_values)
            if hidden_values
            else 0.0
        ),
        "target_status": status,
    }


def frozen_support(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    *,
    transport_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    frozen = trainer.frozen_transport_payload(
        transport_id=transport_id,
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen,
        expected_target_signature=target.target_signature(),
    )
    rows = [tf.zeros([4], tf.float64)]
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64) * SHELL_RADIUS
        rows.extend((direction, -direction))
    z = tf.stack(rows)
    theta = loaded.transport.forward_batch(z)
    value, score, status = target.batch_value_score_status(theta)
    replay_z = loaded.transport.inverse_theta_to_z_batch(theta)
    replay_theta = loaded.transport.forward_batch(replay_z)
    transformed_score = (
        loaded.transport.pullback_score_batch(z, score)
        + loaded.transport.log_abs_det_jacobian_score_batch(z)
    )
    residual = float(
        tf.reduce_max(
            tf.concat(
                (
                    tf.reshape(tf.abs(replay_z - z), [-1]),
                    tf.reshape(tf.abs(replay_theta - theta), [-1]),
                ),
                axis=0,
            )
        ).numpy()
    )
    radius = float(tf.reduce_max(tf.linalg.norm(replay_z, axis=-1)).numpy())
    all_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
        for value in (theta, value, score, replay_z, replay_theta, transformed_score)
    )
    status_valid = bool(tf.reduce_all(status["hard_valid_for_training"]).numpy())
    if (
        not all_finite
        or not status_valid
        or residual > ROUNDTRIP_MAX
        or radius > SHELL_RADIUS_MAX
    ):
        raise CampaignError("frozen support probe failed")
    return frozen, {
        "all_finite": all_finite,
        "all_target_status_valid": status_valid,
        "roundtrip_max_abs": residual,
        "moderate_shell_max_inverse_radius": radius,
        "transformed_score_max_abs": float(
            tf.reduce_max(tf.abs(transformed_score)).numpy()
        ),
        "probe_definition": "origin_plus_coordinate_shell_radius_4_in_neutra_z_chart",
    }


def paired_summary(
    initial: Sequence[float],
    trained: Sequence[float],
    *,
    critical_value: float,
) -> Mapping[str, float]:
    differences = [
        float(right) - float(left)
        for left, right in zip(initial, trained, strict=True)
    ]
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    standard_error = math.sqrt(max(variance, 0.0) / len(differences))
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "one_sided_95_upper": mean + float(critical_value) * standard_error,
        "critical_value": float(critical_value),
    }


def audit_losses(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    z: tf.Tensor,
) -> list[float]:
    validation = trainer.validation_batch(z)
    status_payload(target, validation.theta)
    return [float(value) for value in validation.per_sample_loss.numpy().tolist()]


def contract(args: argparse.Namespace) -> Mapping[str, Any]:
    owner, binding, target = target_and_binding()
    payload = {
        "schema": SCHEMA,
        "status": "CONTRACT_CHECK_PASSED",
        "mode": "contract",
        "target": owner.signature_payload(),
        "binding": binding.payload(),
        "training_proxy": {
            "target_signature": target.target_signature(),
            "adapter_signature": target.adapter_signature(),
            "parameter_dim": target.parameter_dim,
            "parameter_names": list(target.parameter_names),
        },
        "device": device_manifest(),
        "source_sha256": IMPORTED_SOURCE_SHA256,
        "nonclaims": [
            "CPU-hidden contract inspection only",
            "no target evaluation, optimizer update, or GPU/XLA claim",
        ],
    }
    write_json(ROOT / args.output_root / "contract.json", payload)
    return payload


def mechanics(
    args: argparse.Namespace,
    started: float,
    budget: CampaignBudget,
) -> Mapping[str, Any]:
    budget.require(300.0)
    owner, binding, target = target_and_binding()
    trainer = make_trainer(
        target,
        architecture=(32, 32),
        learning_rate=4.0e-4,
        initialization_seed=MECHANICS_STREAM.initialization_seed,
    )
    before = allocator_memory()
    z = batch(MECHANICS_STREAM.training_seed, 1, BATCH_SIZE)
    theta, _ = trainer.forward_and_logdet(z)
    pre_status = status_payload(target, theta)
    step_started = time.perf_counter()
    step = trainer.train_step(z)
    step_seconds = time.perf_counter() - step_started
    frozen, support = frozen_support(
        trainer, target, transport_id="q20-direct-gpu-xla-mechanics"
    )
    frozen_path = ROOT / args.output_root / "mechanics-frozen.json"
    write_json(frozen_path, frozen)
    hlo = trainer._compiled_train_step.experimental_get_compiler_ir(z)(stage="hlo")
    hlo_bytes = hlo if isinstance(hlo, bytes) else str(hlo).encode("utf-8")
    payload = {
        "schema": SCHEMA,
        "status": "GPU_XLA_MECHANICS_PASSED",
        "mode": "mechanics",
        "q": 20,
        "batch_size": BATCH_SIZE,
        "architecture": [32, 32],
        "learning_rate": 4.0e-4,
        "stream": asdict(MECHANICS_STREAM),
        "binding": binding.payload(),
        "pre_update_status": pre_status,
        "training_step": host_step(step),
        "step_wall_seconds_including_first_compile": step_seconds,
        "support_probe": support,
        "frozen_path": frozen_path.relative_to(ROOT).as_posix(),
        "frozen_sha256": sha256(frozen_path),
        "xla_hlo_sha256": hashlib.sha256(hlo_bytes).hexdigest(),
        "variable_devices": sorted({variable.device for variable in trainer.variables}),
        "allocator_before_bytes": before,
        "allocator_after_bytes": allocator_memory(),
        "run_manifest": run_manifest(args, started),
        "execution_eligibility": {
            "tuning_gate_passed": True,
            "hmc_eligible": False,
            "posterior_claim_eligible": False,
        },
        "nonclaims": [
            "one-update mechanics evidence only",
            "no training-quality, convergence, HMC, posterior, or promotion claim",
        ],
    }
    write_json(ROOT / args.output_root / "mechanics-result.json", payload)
    return payload


def tuning_arm(
    args: argparse.Namespace,
    started: float,
    budget: CampaignBudget,
) -> Mapping[str, Any]:
    if args.candidate not in TUNING_CANDIDATES:
        raise CampaignError("tuning-arm requires a known --candidate")
    architecture, learning_rate = TUNING_CANDIDATES[args.candidate]
    mechanics_path = ROOT / args.output_root / "mechanics-result.json"
    mechanics_result = read_json(mechanics_path)
    if mechanics_result.get("status") != "GPU_XLA_MECHANICS_PASSED":
        raise CampaignError("tuning requires a passing mechanics artifact")
    owner, binding, target = target_and_binding()
    if mechanics_result.get("binding") != binding.payload():
        raise CampaignError("mechanics binding does not match tuning binding")
    trainer = make_trainer(
        target,
        architecture=architecture,
        learning_rate=learning_rate,
        initialization_seed=TUNING_STREAM.initialization_seed,
    )
    validation_z = batch(TUNING_STREAM.validation_seed, 0, VALIDATION_SIZE)
    history = [
        validation_payload(
            trainer,
            target,
            validation_z,
            step=0,
            learning_rate=learning_rate,
        )
    ]
    steps = []
    for step_index in range(1, TUNING_STEPS + 1):
        budget.require(300.0)
        z = batch(TUNING_STREAM.training_seed, step_index, BATCH_SIZE)
        step = trainer.train_step(z)
        if step_index in {1, 50, 100}:
            steps.append(host_step(step))
        if step_index in {50, 100}:
            history.append(
                validation_payload(
                    trainer,
                    target,
                    validation_z,
                    step=step_index,
                    learning_rate=learning_rate,
                )
            )
            write_json(
                ROOT / args.output_root / "tuning" / args.candidate / "progress.json",
                {
                    "schema": SCHEMA,
                    "status": "RUNNING",
                    "candidate": args.candidate,
                    "last_step": step_index,
                    "history": history,
                },
                replace=True,
            )
    frozen, support = frozen_support(
        trainer, target, transport_id=f"q20-direct-gpu-xla-tuning-{args.candidate}"
    )
    paired = paired_summary(
        history[0]["per_sample_loss"],
        history[-1]["per_sample_loss"],
        critical_value=1.6694022215079607,
    )
    arm_root = ROOT / args.output_root / "tuning" / args.candidate
    frozen_path = arm_root / "frozen.json"
    write_json(frozen_path, frozen)
    payload = {
        "schema": SCHEMA,
        "status": "TUNING_ARM_COMPLETED",
        "candidate": args.candidate,
        "architecture": list(architecture),
        "learning_rate": learning_rate,
        "initialization_scale": 0.01,
        "gradient_clip_norm": 10.0,
        "stream": asdict(TUNING_STREAM),
        "binding": binding.payload(),
        "history": history,
        "training_step_snapshots": steps,
        "paired_step100_minus_step0": paired,
        "support_probe": support,
        "frozen_path": frozen_path.relative_to(ROOT).as_posix(),
        "frozen_sha256": sha256(frozen_path),
        "run_manifest": run_manifest(args, started),
        "vetoes": [],
        "nonclaims": [
            "single-seed bounded tuning arm",
            "descriptive candidate-selection evidence only",
            "no architecture ranking, HMC, posterior, or default claim",
        ],
    }
    write_json(arm_root / "result.json", payload)
    return payload


def select_tuning(args: argparse.Namespace) -> Mapping[str, Any]:
    candidates = []
    for label, (architecture, learning_rate) in TUNING_CANDIDATES.items():
        path = ROOT / args.output_root / "tuning" / label / "result.json"
        result = read_json(path)
        if result.get("status") != "TUNING_ARM_COMPLETED" or result.get("vetoes"):
            continue
        if result.get("architecture") != list(architecture):
            raise CampaignError(f"tuning architecture mismatch: {label}")
        if float(result.get("learning_rate")) != learning_rate:
            raise CampaignError(f"tuning learning-rate mismatch: {label}")
        if result.get("run_manifest", {}).get("source_sha256_at_import") != IMPORTED_SOURCE_SHA256:
            raise CampaignError(f"tuning source mismatch: {label}")
        candidates.append(
            (
                float(result["paired_step100_minus_step0"]["mean_difference"]),
                sum(architecture),
                learning_rate,
                label,
                result,
                path,
            )
        )
    if not candidates:
        raise CampaignError("no hard-valid tuning candidate remains")
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    best = candidates[0]
    near = [row for row in candidates if row[0] <= best[0] + 0.05]
    selected = min(near, key=lambda row: (row[1], row[2], row[3]))
    _mean, _width, learning_rate, label, result, path = selected
    payload = {
        "schema": TUNING_SCHEMA,
        "status": "TUNING_SELECTION_ISSUED",
        "selection_rule": (
            "lowest_step100_minus_step0_paired_mean_with_0p05_indifference_"
            "then_lower_capacity_then_lower_rate"
        ),
        "selected_candidate": label,
        "architecture": result["architecture"],
        "learning_rate": learning_rate,
        "initialization_scale": 0.01,
        "gradient_clip_norm": 10.0,
        "target_signature": result["binding"]["target_signature"],
        "adapter_signature": result["binding"]["adapter_signature"],
        "binding_dependency_closure_sha256": result["binding"][
            "dependency_closure_sha256"
        ],
        "source_sha256": IMPORTED_SOURCE_SHA256,
        "selected_result_path": path.relative_to(ROOT).as_posix(),
        "selected_result_sha256": sha256(path),
        "candidate_table": [
            {
                "candidate": row[3],
                "paired_mean_difference": row[0],
                "architecture": row[4]["architecture"],
                "learning_rate": row[2],
                "result_path": row[5].relative_to(ROOT).as_posix(),
                "result_sha256": sha256(row[5]),
            }
            for row in candidates
        ],
        "ranking_status": "descriptive_selection_only_not_statistical_ranking",
        "nonclaims": [
            "repository-issued bounded tuning selection",
            "not evidence of architecture superiority or global optimality",
        ],
    }
    unsigned = dict(payload)
    payload["selection_hash"] = hashlib.sha256(canonical(unsigned)).hexdigest()
    write_json(ROOT / args.output_root / "tuning-selection.json", payload)
    return payload


def validated_selection(args: argparse.Namespace, binding: Any) -> Mapping[str, Any]:
    path = ROOT / args.output_root / "tuning-selection.json"
    payload = dict(read_json(path))
    selection_hash = str(payload.pop("selection_hash", ""))
    if selection_hash != hashlib.sha256(canonical(payload)).hexdigest():
        raise CampaignError("tuning selection hash mismatch")
    payload["selection_hash"] = selection_hash
    if payload.get("schema") != TUNING_SCHEMA:
        raise CampaignError("tuning selection schema mismatch")
    if payload.get("source_sha256") != IMPORTED_SOURCE_SHA256:
        raise CampaignError("tuning selection source scope mismatch")
    if payload.get("target_signature") != binding.target_signature:
        raise CampaignError("tuning target signature mismatch")
    if payload.get("adapter_signature") != binding.adapter_signature:
        raise CampaignError("tuning adapter signature mismatch")
    if payload.get("binding_dependency_closure_sha256") != binding.dependency_closure_sha256:
        raise CampaignError("tuning binding closure mismatch")
    selected_path = ROOT / str(payload["selected_result_path"])
    if sha256(selected_path) != payload.get("selected_result_sha256"):
        raise CampaignError("selected tuning result hash mismatch")
    return payload


def final_stream(
    args: argparse.Namespace,
    started: float,
    budget: CampaignBudget,
) -> Mapping[str, Any]:
    if args.stream not in FINAL_STREAMS:
        raise CampaignError("final-stream requires seed-a or seed-b")
    stream = FINAL_STREAMS[args.stream]
    owner, binding, target = target_and_binding()
    selection = validated_selection(args, binding)
    architecture = tuple(int(value) for value in selection["architecture"])
    learning_rate = float(selection["learning_rate"])
    trainer = make_trainer(
        target,
        architecture=architecture,
        learning_rate=learning_rate,
        initialization_seed=stream.initialization_seed,
    )
    validation_z = batch(stream.validation_seed, 0, VALIDATION_SIZE)
    initial_validation = validation_payload(
        trainer,
        target,
        validation_z,
        step=0,
        learning_rate=learning_rate,
    )
    if stream.audit_seed is None:
        raise CampaignError("final stream is missing audit seed")
    audit_z = batch(stream.audit_seed, 0, AUDIT_SIZE)
    untrained_audit = audit_losses(trainer, target, audit_z)
    initial_state = trainer.state_payload()
    controller = NeuTraPlateauController(
        NeuTraPlateauConfig(
            validation_check_every=CHECK_EVERY,
            patience_steps=200,
            max_steps=FINAL_MAX_STEPS,
            initial_learning_rate=learning_rate,
            learning_rate_factor=0.5,
            post_repair_no_improvement_cycles=2,
            saturation_repair_enabled=False,
            roundtrip_max_abs=ROUNDTRIP_MAX,
            moderate_shell_max_inverse_radius=SHELL_RADIUS_MAX,
        )
    )
    initial_frozen, initial_support = frozen_support(
        trainer,
        target,
        transport_id=f"q20-direct-final-{stream.label}-initial",
    )
    action = controller.observe(
        step=0,
        per_sample_loss=initial_validation["per_sample_loss"],
        saturation_fraction=initial_validation["saturation_fraction"],
        all_finite=initial_support["all_finite"],
        roundtrip_max_abs=initial_support["roundtrip_max_abs"],
        moderate_shell_max_inverse_radius=initial_support[
            "moderate_shell_max_inverse_radius"
        ],
        trainer_state_hash=initial_state["state_hash"],
    )
    history = [{**initial_validation, "controller_action": action.payload()}]
    best_state = initial_state
    checkpoints = []
    step_snapshots = []
    for step_index in range(1, FINAL_MAX_STEPS + 1):
        budget.require(600.0)
        z = batch(stream.training_seed, step_index, BATCH_SIZE)
        step = trainer.train_step(z)
        if step_index == 1 or step_index % CHECK_EVERY == 0:
            step_snapshots.append(host_step(step))
        if step_index % CHECK_EVERY:
            continue
        validation = validation_payload(
            trainer,
            target,
            validation_z,
            step=step_index,
            learning_rate=controller.current_learning_rate,
        )
        _, support = frozen_support(
            trainer,
            target,
            transport_id=f"q20-direct-final-{stream.label}-step-{step_index}",
        )
        state = trainer.state_payload()
        action = controller.observe(
            step=step_index,
            per_sample_loss=validation["per_sample_loss"],
            saturation_fraction=validation["saturation_fraction"],
            all_finite=support["all_finite"],
            roundtrip_max_abs=support["roundtrip_max_abs"],
            moderate_shell_max_inverse_radius=support[
                "moderate_shell_max_inverse_radius"
            ],
            trainer_state_hash=state["state_hash"],
        )
        if action.meaningful_improvement:
            best_state = state
        if action.should_reduce_learning_rate:
            trainer.restore_state(best_state)
            trainer.set_learning_rate(action.current_learning_rate)
        joint = joint_training_checkpoint_payload(
            trainer_state=trainer.state_payload(),
            controller_state=controller.state_payload(),
            best_trainer_state=best_state,
        )
        validate_joint_training_checkpoint(joint)
        checkpoint_path = (
            ROOT
            / args.output_root
            / "final"
            / stream.label
            / f"checkpoint-{step_index:04d}.json"
        )
        write_json(checkpoint_path, joint)
        checkpoints.append(
            {
                "step": step_index,
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(checkpoint_path),
                "checkpoint_hash": joint["checkpoint_hash"],
            }
        )
        history.append({**validation, "support_probe": support, "controller_action": action.payload()})
        write_json(
            ROOT / args.output_root / "final" / stream.label / "progress.json",
            {
                "schema": SCHEMA,
                "status": "RUNNING",
                "stream": asdict(stream),
                "last_program_step": step_index,
                "history": history,
                "checkpoints": checkpoints,
                "wall_seconds": time.perf_counter() - started,
            },
            replace=True,
        )
        if action.should_stop:
            break
    if controller.stop_reason is None:
        raise CampaignError("final stream ended without a controller stop")
    best_trainer = make_trainer(
        target,
        architecture=architecture,
        learning_rate=learning_rate,
        initialization_seed=stream.initialization_seed,
    )
    best_trainer.restore_state(best_state)
    trained_audit = audit_losses(best_trainer, target, audit_z)
    paired = paired_summary(
        untrained_audit,
        trained_audit,
        critical_value=AUDIT_CRITICAL_VALUE,
    )
    frozen, support = frozen_support(
        best_trainer,
        target,
        transport_id=f"q20-direct-final-{stream.label}-best",
    )
    stream_root = ROOT / args.output_root / "final" / stream.label
    frozen_path = stream_root / "frozen-best.json"
    write_json(frozen_path, frozen)
    vetoes = []
    if paired["one_sided_95_upper"] >= 0.0:
        vetoes.append("paired_untouched_audit_improvement_failed")
    payload = {
        "schema": SCHEMA,
        "status": (
            "GPU_XLA_FINAL_STREAM_SCREEN_PASSED"
            if not vetoes
            else "GPU_XLA_FINAL_STREAM_SCREEN_VETOED"
        ),
        "q": 20,
        "stream": asdict(stream),
        "selected_tuning_hash": selection["selection_hash"],
        "selected_candidate": selection["selected_candidate"],
        "architecture": list(architecture),
        "learning_rate": learning_rate,
        "initialization_scale": 0.01,
        "gradient_clip_norm": 10.0,
        "best_step": controller.best_step,
        "terminal_program_step": history[-1]["step"],
        "terminal_optimizer_step": int(trainer.step.numpy()),
        "stop_reason": controller.stop_reason,
        "learning_rate_reductions": controller.learning_rate_reductions,
        "history": history,
        "training_step_snapshots": step_snapshots,
        "checkpoints": checkpoints,
        "audit": {
            "definition": "stateless_final_stream_audit_seed_fold_0_untouched",
            "batch_size": AUDIT_SIZE,
            "untrained_mean_loss": math.fsum(untrained_audit) / len(untrained_audit),
            "trained_mean_loss": math.fsum(trained_audit) / len(trained_audit),
            "paired_trained_minus_untrained": paired,
            "untrained_per_sample_loss": untrained_audit,
            "trained_per_sample_loss": trained_audit,
        },
        "support_probe": support,
        "frozen_path": frozen_path.relative_to(ROOT).as_posix(),
        "frozen_sha256": sha256(frozen_path),
        "vetoes": vetoes,
        "run_manifest": run_manifest(args, started),
        "execution_eligibility": {
            "gpu_xla_training_screen_eligible": not vetoes,
            "hmc_eligible": False,
            "posterior_claim_eligible": False,
            "default_ready": False,
        },
        "nonclaims": [
            "one of two predeclared final training replications",
            "loss screen is not convergence or posterior evidence",
            "no HMC, architecture-ranking, default, or scientific-validity claim",
        ],
    }
    write_json(stream_root / "result.json", payload)
    return payload


def summarize(args: argparse.Namespace) -> Mapping[str, Any]:
    selection = read_json(ROOT / args.output_root / "tuning-selection.json")
    results = []
    for label in ("seed-a", "seed-b"):
        path = ROOT / args.output_root / "final" / label / "result.json"
        result = read_json(path)
        if result.get("selected_tuning_hash") != selection.get("selection_hash"):
            raise CampaignError(f"final selection mismatch: {label}")
        results.append((label, result, path))
    passed = all(
        result.get("status") == "GPU_XLA_FINAL_STREAM_SCREEN_PASSED"
        for _label, result, _path in results
    )
    payload = {
        "schema": SCHEMA,
        "status": (
            "GPU_XLA_TRAINING_SCREEN_PASSED"
            if passed
            else "GPU_XLA_TRAINING_SCREEN_VETOED"
        ),
        "selected_candidate": selection["selected_candidate"],
        "architecture": selection["architecture"],
        "learning_rate": selection["learning_rate"],
        "final_streams": [
            {
                "label": label,
                "status": result["status"],
                "best_step": result["best_step"],
                "terminal_program_step": result["terminal_program_step"],
                "trained_audit_mean_loss": result["audit"]["trained_mean_loss"],
                "paired_audit": result["audit"]["paired_trained_minus_untrained"],
                "vetoes": result["vetoes"],
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for label, result, path in results
        ],
        "hard_veto_screen": "PASS" if passed else "VETO",
        "statistically_supported_ranking": "none; no method or architecture ranking was tested",
        "descriptive_only_differences": [
            "per-seed loss, runtime, gradient, saturation, and allocator telemetry"
        ],
        "default_readiness": "not established",
        "next_evidence_needed": (
            "downstream HMC validation under a separate reviewed plan; not authorized here"
            if passed
            else "repair the failed frozen protocol with fresh validation and audit partitions"
        ),
        "execution_eligibility": {
            "gpu_xla_training_screen_passed": passed,
            "hmc_launched": False,
            "hmc_eligible_from_this_artifact_alone": False,
            "posterior_claim_eligible": False,
            "default_ready": False,
        },
        "nonclaims": [
            "no HMC was launched",
            "no convergence, posterior, architecture-superiority, default, or scientific-validity claim",
        ],
    }
    write_json(ROOT / args.output_root / "summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("contract", "mechanics", "tuning-arm", "select", "final-stream", "summarize"),
        default="contract",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate", choices=tuple(TUNING_CANDIDATES), default=None)
    parser.add_argument("--stream", choices=tuple(FINAL_STREAMS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output_root.is_absolute():
        raise CampaignError("output root must be repository-relative")
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise CampaignError("output root escapes the repository")
    budget = CampaignBudget(args.output_root) if args.mode in MATERIAL_MODES else None
    try:
        if args.mode == "contract":
            result = contract(args)
        elif args.mode == "mechanics":
            assert budget is not None
            result = mechanics(args, started, budget)
        elif args.mode == "tuning-arm":
            assert budget is not None
            result = tuning_arm(args, started, budget)
        elif args.mode == "select":
            result = select_tuning(args)
        elif args.mode == "final-stream":
            assert budget is not None
            result = final_stream(args, started, budget)
        else:
            result = summarize(args)
    except Exception:
        if budget is not None:
            budget.persist(status="FAILED_ATTEMPT")
        raise
    if budget is not None:
        budget.persist(status=str(result["status"]))
    print(json.dumps({"status": result["status"], "mode": args.mode}, sort_keys=True))


if __name__ == "__main__":
    main()
