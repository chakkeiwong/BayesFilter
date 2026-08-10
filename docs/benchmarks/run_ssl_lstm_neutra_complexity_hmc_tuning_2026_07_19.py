#!/usr/bin/env python3
"""Q-general transformed-target preflight and fixed-kernel HMC tuning.

Material preflight/tuning modes require explicit authorization and a cumulative
cap. Contract-smoke mode hides GPUs and performs no target or HMC evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _configure_visibility_before_tensorflow_import() -> str:
    mode = None
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv):
            mode = sys.argv[index + 1]
    if mode == "contract-smoke":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu-hidden-contract-smoke"
    if mode not in {"preflight", "tune"} and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return str(os.environ["CUDA_VISIBLE_DEVICES"])
    probe = subprocess.run(
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    available = {
        int(line.strip())
        for line in probe.stdout.splitlines()
        if line.strip().isdigit()
    }
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _configure_visibility_before_tensorflow_import()

import numpy as np
import tensorflow as tf


def _enable_memory_growth_before_project_imports() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU memory growth must be established before project imports"
            ) from exc
        if tf.config.experimental.get_memory_growth(gpu) is not True:
            raise RuntimeError("GPU memory growth verification failed")


_enable_memory_growth_before_project_imports()


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter  # noqa: E402
from bayesfilter.inference.hmc import (  # noqa: E402
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_hmc_tuning.v1"
CHECKPOINT_SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_hmc_tuning.checkpoint.v1"
PHASE3_SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_training.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-32x32-hmc-tuning-plan-2026-07-21.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
Q_VALUES = (1, 2, 5, 10, 20)
ROOT_SEED = 20260719
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
INITIAL_SCALE_GRID = (0.05, 0.10, 0.20, 0.40)
HIGH_SCALE_EXPANSION = (0.80, 1.60)
LOW_SCALE_EXPANSION = (0.025, 0.0125)
TRAJECTORY_GRID = (2, 4, 8, 16)
TRAJECTORY_PRIORITY = (8, 4, 16, 2)
PILOT_ACCEPTANCE_BAND = (0.50, 0.90)
CONFIRMATION_ACCEPTANCE_BAND = (0.60, 0.80)
TARGET_ACCEPTANCE = 0.70
PILOT_MIN_MOVEMENT = 0.25
CONFIRMATION_MIN_MOVEMENT = 0.50
MIN_RMS_JUMP_DISTANCE = 0.05
FD_ABS_TOL = 2.0e-5
FD_REL_TOL = 2.0e-5
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3)
FD_REQUIRED_ERROR_REDUCTION = 0.50
PILOT_RESULTS = 16
PILOT_BURNIN = 8
CONFIRMATION_RESULTS = 64
CONFIRMATION_BURNIN = 32
HOST_RAM_CAP_BYTES = 64 * 1024**3
FIRST_COMPILED_ARM_RESERVE_SECONDS = 600.0
ARM_COST_MARGIN = 1.50


class ComplexityHMCError(RuntimeError):
    """Raised when a transformed-HMC contract is invalid."""


class ResourceStop(ComplexityHMCError):
    """Raised after the cumulative HMC cap is exhausted."""


class HostMemoryVeto(ComplexityHMCError):
    """Raised if the HMC parent exceeds the 64 GiB host limit."""


class Budget:
    def __init__(self, seconds: float, *, prior_seconds: float = 0.0) -> None:
        self.seconds = float(seconds)
        self.prior_seconds = float(prior_seconds)
        self.started = time.perf_counter()
        self.observed_seconds_per_transition_leapfrog: list[float] = []

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 0.0) -> None:
        if self.elapsed + float(reserve_seconds) >= self.seconds:
            raise ResourceStop("declared transformed-HMC cap exhausted")

    def observe(self, seconds_per_transition_leapfrog: float) -> None:
        value = float(seconds_per_transition_leapfrog)
        if math.isfinite(value) and value > 0.0:
            self.observed_seconds_per_transition_leapfrog.append(value)

    def arm_reserve(
        self,
        *,
        transitions: int,
        leapfrog_steps: int,
        cold_runner: bool,
    ) -> float:
        execution = 0.0
        if self.observed_seconds_per_transition_leapfrog:
            execution = (
                max(self.observed_seconds_per_transition_leapfrog)
                * int(transitions)
                * int(leapfrog_steps)
                * ARM_COST_MARGIN
            )
        compile_reserve = FIRST_COMPILED_ARM_RESERVE_SECONDS if cold_runner else 0.0
        return max(60.0, execution + compile_reserve)


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ComplexityHMCError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


@contextlib.contextmanager
def output_writer_lock(output: Path):
    """Allow at most one material run to own an output root."""
    lock_path = output / ".material-run.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ComplexityHMCError(
                f"output root is already locked by another material run: {output}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComplexityHMCError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ComplexityHMCError(f"{label} must remain inside the repository")
    return resolved


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


class TargetBridge:
    """Grant HMC authority only to one q-specific principal-root target."""

    def __init__(self, target: Any, *, evidence_path: str) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:phase4_fixed_transport_hmc"
        self.evidence_path = str(evidence_path)

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            (self.target.adapter_signature() + ":phase4-fixed-transport-hmc").encode(
                "ascii"
            )
        ).hexdigest()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_complexity_phase4_target_bridge",
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=(
                "q-specific transformed-HMC tuning authority only",
                "does not promote the base target globally",
                "no convergence or posterior correctness claim",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("target bridge requires rank-one or rank-two positions")


def load_binding(q: int, label: str, result_path: Path) -> tuple[Any, dict[str, Any]]:
    receipt_path = repo_path(result_path, label=f"{label} Phase 3 result")
    receipt = strict_json(receipt_path)
    if receipt.get("schema") != PHASE3_SCHEMA:
        raise ComplexityHMCError(f"{label} Phase 3 result schema mismatch")
    if receipt.get("status") != "ADMITTED":
        raise ComplexityHMCError(f"{label} Phase 3 result is not ADMITTED")
    if int(receipt.get("q", -1)) != q:
        raise ComplexityHMCError(f"{label} Phase 3 q mismatch")
    stream = receipt.get("stream")
    if not isinstance(stream, Mapping) or not str(stream.get("label", "")):
        raise ComplexityHMCError(f"{label} Phase 3 stream receipt is missing")
    payload_relative = receipt.get("best_frozen_payload_path")
    expected_payload_hash = receipt.get("best_frozen_payload_sha256")
    if not isinstance(payload_relative, str) or not isinstance(
        expected_payload_hash, str
    ):
        raise ComplexityHMCError(
            f"{label} Phase 3 result lacks externalized frozen-payload binding"
        )
    path = repo_path(Path(payload_relative), label=f"{label} transport payload")
    actual_payload_hash = sha256(path)
    if actual_payload_hash != expected_payload_hash:
        raise ComplexityHMCError(f"{label} frozen-payload hash mismatch")
    payload = strict_json(path)
    target = complexity_posterior_target(q, jit_compile=True)
    artifact = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=target.target_signature(),
    )
    replayed = load_frozen_neutra_artifact(
        json.loads(canonical(payload)),
        expected_target_signature=target.target_signature(),
    )
    if artifact.artifact_signature != replayed.artifact_signature:
        raise ComplexityHMCError(f"{label} serialized artifact replay mismatch")
    if int(artifact.manifest.dimension) != int(target.parameter_dim):
        raise ComplexityHMCError(f"{label} transport dimension mismatch")
    bridge = TargetBridge(target, evidence_path=path.relative_to(ROOT).as_posix())
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=f"{bridge.target_scope}:{label}",
        runtime_backend="ssl_lstm_complexity_phase4_fixed_transport_hmc",
        evidence_path=path.relative_to(ROOT).as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "fixed frozen transport for tuning only",
            "identity HMC mass",
            "tuning samples are excluded from retained evidence",
        ),
    )
    capability = adapter.value_score_capability()
    if not capability.is_accepted_full_chain_xla_diagnostic_authority:
        raise ComplexityHMCError("scoped transformed target lacks HMC authority")
    binding = {
        "label": label,
        "phase3_result_path": receipt_path.relative_to(ROOT).as_posix(),
        "phase3_result_sha256": sha256(receipt_path),
        "phase3_stream_label": str(stream["label"]),
        "payload_path": path.relative_to(ROOT).as_posix(),
        "payload_sha256": actual_payload_hash,
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "transport_hash": artifact.manifest.transport_hash,
        "tensor_hash": artifact.manifest.tensor_hash,
        "topology_hash": artifact.manifest.topology_hash,
        "artifact_signature": artifact.artifact_signature,
        "serialization_replay_artifact_signature_match": True,
        "target_scope": adapter.target_scope,
    }
    binding["binding_signature"] = payload_sha256(binding)
    return adapter, binding


def validate_distinct_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    if set(bindings) != {"chart-a", "chart-b"}:
        raise ComplexityHMCError("exactly chart-a and chart-b bindings are required")
    for field in (
        "phase3_result_path",
        "phase3_stream_label",
        "payload_path",
        "payload_sha256",
        "artifact_signature",
        "binding_signature",
    ):
        values = [str(bindings[label][field]) for label in ("chart-a", "chart-b")]
        if len(set(values)) != 2:
            raise ComplexityHMCError(
                f"Phase 4 requires two independent bindings; duplicate {field}"
            )


def finite_difference_ladder_decision(
    residuals: Sequence[float],
) -> tuple[int, float, bool]:
    if len(residuals) != len(FD_STEPS) or not all(
        math.isfinite(float(value)) and float(value) >= 0.0 for value in residuals
    ):
        raise ComplexityHMCError("finite-difference ladder residuals are invalid")
    best_index = min(range(len(residuals)), key=lambda index: float(residuals[index]))
    first = float(residuals[0])
    best = float(residuals[best_index])
    reduction = best / first if first > 0.0 else (0.0 if best == 0.0 else math.inf)
    converged = best_index > 0 and reduction <= FD_REQUIRED_ERROR_REDUCTION
    return best_index, reduction, converged


def transformed_preflight(adapter: Any) -> dict[str, Any]:
    transport = adapter.transport
    z = tf.constant(INITIAL_Z, tf.float64)
    theta = transport.forward_batch(z)
    replay_z = transport.inverse_theta_to_z_batch(theta)
    replay_theta = transport.forward_batch(replay_z)
    base_value, base_score = adapter.base_adapter.log_prob_and_grad(theta)
    expected_value = base_value + transport.log_abs_det_jacobian_batch(z)
    expected_score = transport.pullback_score_batch(z, base_score)
    expected_score += transport.log_abs_det_jacobian_score_batch(z)
    actual_value, actual_score = adapter.log_prob_and_grad(z)
    value_residual = float(tf.reduce_max(tf.abs(actual_value - expected_value)).numpy())
    score_residual = float(tf.reduce_max(tf.abs(actual_score - expected_score)).numpy())
    roundtrip = float(
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
    origin = tf.zeros((4,), tf.float64)
    _origin_value, origin_score = adapter.log_prob_and_grad(origin)
    fd_rows = []
    for step in FD_STEPS:
        epsilon = tf.constant(step, tf.float64)
        finite_difference = []
        for coordinate in range(4):
            direction = tf.one_hot(coordinate, 4, dtype=tf.float64)
            plus, _ = adapter.log_prob_and_grad(origin + epsilon * direction)
            minus, _ = adapter.log_prob_and_grad(origin - epsilon * direction)
            finite_difference.append((plus - minus) / (2.0 * epsilon))
        finite_difference_tensor = tf.stack(finite_difference)
        fd_residual = float(
            tf.reduce_max(tf.abs(finite_difference_tensor - origin_score)).numpy()
        )
        fd_scale = max(
            1.0,
            float(tf.reduce_max(tf.abs(finite_difference_tensor)).numpy()),
            float(tf.reduce_max(tf.abs(origin_score)).numpy()),
        )
        fd_rows.append(
            {
                "step": step,
                "finite_difference": finite_difference_tensor,
                "max_abs_residual": fd_residual,
                "scale": fd_scale,
                "tolerance": FD_ABS_TOL + FD_REL_TOL * fd_scale,
            }
        )
    best_fd_index, fd_error_reduction, fd_converged = (
        finite_difference_ladder_decision(
            [float(row["max_abs_residual"]) for row in fd_rows]
        )
    )
    best_fd = fd_rows[best_fd_index]
    tensors = (
        z,
        theta,
        replay_z,
        replay_theta,
        base_value,
        base_score,
        expected_value,
        expected_score,
        actual_value,
        actual_score,
        origin_score,
        *(row["finite_difference"] for row in fd_rows),
    )
    all_finite = all(bool(tf.reduce_all(tf.math.is_finite(row)).numpy()) for row in tensors)
    vetoes = []
    if not all_finite:
        vetoes.append("nonfinite_transformed_target_preflight")
    if roundtrip > 1.0e-9:
        vetoes.append("transport_roundtrip_above_threshold")
    if value_residual > 1.0e-10:
        vetoes.append("change_of_variables_value_identity_failed")
    if score_residual > 1.0e-9:
        vetoes.append("change_of_variables_score_identity_failed")
    if (
        float(best_fd["max_abs_residual"]) > float(best_fd["tolerance"])
        or not fd_converged
    ):
        vetoes.append("transformed_score_finite_difference_failed")
    return {
        "status": "PASSED" if not vetoes else "VETOED",
        "all_finite": all_finite,
        "roundtrip_max_abs": roundtrip,
        "value_identity_max_abs": value_residual,
        "score_identity_max_abs": score_residual,
        "finite_difference_max_abs": best_fd["max_abs_residual"],
        "finite_difference_scale": best_fd["scale"],
        "finite_difference_tolerance": best_fd["tolerance"],
        "finite_difference_tolerance_rule": "atol + rtol * max(1,abs(fd),abs(score))",
        "finite_difference_steps": list(FD_STEPS),
        "finite_difference_rows": [
            {
                "step": row["step"],
                "finite_difference": json_safe(row["finite_difference"]),
                "max_abs_residual": row["max_abs_residual"],
                "scale": row["scale"],
                "tolerance": row["tolerance"],
            }
            for row in fd_rows
        ],
        "finite_difference_best_step": best_fd["step"],
        "finite_difference_error_reduction": fd_error_reduction,
        "finite_difference_required_error_reduction": FD_REQUIRED_ERROR_REDUCTION,
        "finite_difference_converged_before_cancellation": fd_converged,
        "initial_z": [list(row) for row in INITIAL_Z],
        "initial_theta": json_safe(theta),
        "vetoes": vetoes,
    }


def diagnose_run(
    *,
    samples: Any,
    initial_state: Any,
    trace: Mapping[str, Any],
    acceptance_band: tuple[float, float] | None,
    min_movement: float,
    min_rms_jump: float,
) -> dict[str, Any]:
    sample_tensor = tf.convert_to_tensor(samples, tf.float64)
    initial_tensor = tf.convert_to_tensor(initial_state, tf.float64)
    if sample_tensor.shape.rank != 3 or tuple(sample_tensor.shape[1:]) != (4, 4):
        raise ComplexityHMCError("HMC samples must have shape [draw,4,4]")
    accepted = tf.convert_to_tensor(trace["is_accepted"], tf.bool)
    log_accept = tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64)
    target = tf.convert_to_tensor(trace["target_log_prob"], tf.float64)
    expected_shape = tuple(sample_tensor.shape[:2])
    if any(tuple(row.shape) != expected_shape for row in (accepted, log_accept, target)):
        raise ComplexityHMCError("HMC trace shape mismatch")
    previous = tf.concat((initial_tensor[tf.newaxis, ...], sample_tensor[:-1]), axis=0)
    jump_norm = tf.linalg.norm(sample_tensor - previous, axis=-1)
    movement = tf.reduce_mean(tf.cast(jump_norm > 0.0, tf.float64), axis=0)
    any_moved = tf.reduce_any(jump_norm > 0.0, axis=0)
    rms_jump = tf.sqrt(tf.reduce_mean(tf.square(jump_norm), axis=0))
    acceptance_probability = tf.exp(tf.minimum(log_accept, 0.0))
    acceptance = tf.reduce_mean(acceptance_probability, axis=0)
    binary_acceptance = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
    finite = {
        "samples": bool(tf.reduce_all(tf.math.is_finite(sample_tensor)).numpy()),
        "log_accept_ratio": bool(tf.reduce_all(tf.math.is_finite(log_accept)).numpy()),
        "target_log_prob": bool(tf.reduce_all(tf.math.is_finite(target)).numpy()),
    }
    for key in ("proposed_target_log_prob", "log_acceptance_correction"):
        if key in trace:
            row = tf.convert_to_tensor(trace[key], tf.float64)
            if tuple(row.shape) != expected_shape:
                raise ComplexityHMCError(f"{key} trace shape mismatch")
            finite[key] = bool(tf.reduce_all(tf.math.is_finite(row)).numpy())
    divergence_status = "unavailable_not_zero"
    divergence_count = None
    divergence_by_chain = None
    if "divergence" in trace:
        divergence = tf.convert_to_tensor(trace["divergence"], tf.bool)
        divergence_status = "available"
        divergence_count = int(tf.reduce_sum(tf.cast(divergence, tf.int32)).numpy())
        divergence_by_chain = json_safe(
            tf.reduce_sum(tf.cast(divergence, tf.int32), axis=0)
        )
    hard_vetoes = []
    if not all(finite.values()):
        hard_vetoes.append("nonfinite_hmc_telemetry")
    if not bool(tf.reduce_all(any_moved).numpy()):
        hard_vetoes.append("unmoved_chain")
    if divergence_count is not None and divergence_count > 0:
        hard_vetoes.append("positive_native_divergence")
    explanatory_flags = []
    if bool(tf.reduce_any(movement < min_movement).numpy()):
        explanatory_flags.append("per_chain_movement_below_descriptive_reference")
    if bool(tf.reduce_any(rms_jump < min_rms_jump).numpy()):
        explanatory_flags.append("per_chain_rms_jump_below_descriptive_reference")
    acceptance_vetoes = []
    if acceptance_band is not None:
        low, high = acceptance_band
        if bool(tf.reduce_any(acceptance < low).numpy()):
            acceptance_vetoes.append("per_chain_acceptance_below_band")
        if bool(tf.reduce_any(acceptance > high).numpy()):
            acceptance_vetoes.append("per_chain_acceptance_above_band")
    return {
        "viable": not hard_vetoes and not acceptance_vetoes,
        "finite": finite,
        "acceptance_rate": float(tf.reduce_mean(acceptance).numpy()),
        "acceptance_rate_by_chain": json_safe(acceptance),
        "acceptance_rate_semantics": "mean_metropolis_acceptance_probability",
        "binary_acceptance_rate_by_chain": json_safe(binary_acceptance),
        "binary_acceptance_role": "explanatory_movement_diagnostic",
        "movement_rate_by_chain": json_safe(movement),
        "chain_moved": json_safe(any_moved),
        "rms_jump_distance_by_chain": json_safe(rms_jump),
        "native_divergence_status": divergence_status,
        "native_divergence_count": divergence_count,
        "native_divergence_count_by_chain": divergence_by_chain,
        "hard_vetoes": hard_vetoes,
        "acceptance_vetoes": acceptance_vetoes,
        "explanatory_flags": explanatory_flags,
        "thresholds": {
            "acceptance_band": acceptance_band,
            "descriptive_movement_rate_reference": min_movement,
            "descriptive_rms_jump_distance_reference": min_rms_jump,
        },
    }


def build_runner(
    adapter: Any,
    *,
    num_results: int,
    num_burnin_steps: int,
    leapfrog_steps: int,
) -> Any:
    initial = tf.constant(INITIAL_Z, tf.float64)
    config = FullChainHMCConfig(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        step_size=0.01,
        num_leapfrog_steps=leapfrog_steps,
        seed=(ROOT_SEED, 1),
        use_xla=True,
        trace_policy="standard",
        target_scope=adapter.target_scope,
    )
    return build_reusable_full_chain_tfp_hmc_runner(
        adapter,
        initial,
        config,
        dynamic_num_leapfrog_steps=True,
    )


def run_arm(
    runner: Any,
    *,
    step_size: float,
    leapfrog_steps: int,
    seed_word: int,
    role: str,
    acceptance_band: tuple[float, float] | None,
    min_movement: float,
    min_rms_jump: float,
) -> dict[str, Any]:
    initial = tf.constant(INITIAL_Z, tf.float64)
    result = runner.run(
        current_state=initial,
        seed=(ROOT_SEED, seed_word),
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps,
    )
    diagnostics = diagnose_run(
        samples=result.samples,
        initial_state=initial,
        trace=result.trace,
        acceptance_band=acceptance_band,
        min_movement=min_movement,
        min_rms_jump=min_rms_jump,
    )
    transitions = int(runner.config.num_results + runner.config.num_burnin_steps)
    wall = float(result.metadata["sample_chain_call_s"])
    normalized = wall / (transitions * int(leapfrog_steps))
    host_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if host_rss > HOST_RAM_CAP_BYTES:
        raise HostMemoryVeto("HMC parent RSS exceeded 64 GiB")
    return {
        "role": role,
        "seed": [ROOT_SEED, seed_word],
        "step_size": float(step_size),
        "num_leapfrog_steps": int(leapfrog_steps),
        "trajectory_length": float(step_size) * int(leapfrog_steps),
        "diagnostics": diagnostics,
        "runner_diagnostics": json_safe(result.diagnostics),
        "timing": {
            "sample_chain_seconds": wall,
            "transitions": transitions,
            "leapfrog_steps": int(leapfrog_steps),
            "seconds_per_transition_leapfrog": normalized,
            "timing_role": "descriptive_cost_normalization_only",
        },
        "samples_retained_as_posterior_evidence": False,
    }


def select_scale(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    viable = [row for row in rows if row["diagnostics"]["viable"]]
    if not viable:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float]:
        rates = row["diagnostics"]["acceptance_rate_by_chain"]
        worst = max(abs(float(rate) - TARGET_ACCEPTANCE) for rate in rates)
        return round(worst, 12), -float(row["step_size"])

    return min(viable, key=key)


def scale_expansion(rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    if not rows or select_scale(rows) is not None:
        return ()
    boundary = max(rows, key=lambda row: float(row["step_size"]))
    rates = [
        float(rate) for rate in boundary["diagnostics"]["acceptance_rate_by_chain"]
    ]
    pooled = sum(rates) / len(rates)
    if pooled > PILOT_ACCEPTANCE_BAND[1]:
        return HIGH_SCALE_EXPANSION
    if pooled < PILOT_ACCEPTANCE_BAND[0]:
        return LOW_SCALE_EXPANSION
    return ()


def scale_bracket_repair(rows: Sequence[Mapping[str, Any]]) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row["step_size"]))
    for lower, upper in zip(ordered, ordered[1:]):
        lower_rates = [
            float(value) for value in lower["diagnostics"]["acceptance_rate_by_chain"]
        ]
        upper_rates = [
            float(value) for value in upper["diagnostics"]["acceptance_rate_by_chain"]
        ]
        lower_reaches_or_exceeds_band = (
            min(lower_rates) >= PILOT_ACCEPTANCE_BAND[0]
            and max(lower_rates) > PILOT_ACCEPTANCE_BAND[1]
        )
        upper_reaches_or_crosses_band = (
            max(upper_rates) <= PILOT_ACCEPTANCE_BAND[1]
            and min(upper_rates) < PILOT_ACCEPTANCE_BAND[0]
        )
        upper_hard_vetoes = upper["diagnostics"].get("hard_vetoes", [])
        lower_hard_vetoes = lower["diagnostics"].get("hard_vetoes", [])
        if (
            lower_reaches_or_exceeds_band
            and upper_reaches_or_crosses_band
            and not lower_hard_vetoes
            and not upper_hard_vetoes
        ):
            return math.sqrt(float(lower["step_size"]) * float(upper["step_size"]))
    return None


def select_trajectory(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    viable = {
        int(row["num_leapfrog_steps"]): row
        for row in rows
        if row["diagnostics"]["viable"]
    }
    return next((viable[value] for value in TRAJECTORY_PRIORITY if value in viable), None)


def adjacent_repair_candidate(
    confirmation: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    diagnostics = confirmation["diagnostics"]
    if diagnostics["hard_vetoes"]:
        return None
    vetoes = diagnostics["acceptance_vetoes"]
    current = int(confirmation["num_leapfrog_steps"])
    viable_by_l = {
        int(row["num_leapfrog_steps"]): row
        for row in trajectory_rows
        if row["diagnostics"]["viable"]
    }
    if vetoes == ["per_chain_acceptance_below_band"]:
        choices = [value for value in viable_by_l if value < current]
        return viable_by_l[max(choices)] if choices else None
    if vetoes == ["per_chain_acceptance_above_band"]:
        choices = [value for value in viable_by_l if value > current]
        return viable_by_l[min(choices)] if choices else None
    return None


def arm_path(output: Path, label: str, stage: str, index: int) -> Path:
    return output / "arms" / label / f"{stage}-{index:02d}.json"


def run_or_reuse_arm(
    *,
    output: Path,
    label: str,
    stage: str,
    index: int,
    runner: Any,
    budget: Budget,
    step_size: float,
    leapfrog_steps: int,
    seed_word: int,
    acceptance_band: tuple[float, float] | None,
    min_movement: float,
    min_rms_jump: float,
    binding_signature: str,
    source_signature: str,
    resume: bool,
    checkpoint: Callable[[float], None],
) -> dict[str, Any]:
    path = arm_path(output, label, stage, index)
    contract = {
        "label": label,
        "stage": stage,
        "index": index,
        "step_size": float(step_size),
        "num_leapfrog_steps": int(leapfrog_steps),
        "seed": [ROOT_SEED, seed_word],
        "acceptance_band": acceptance_band,
        "min_movement": min_movement,
        "min_rms_jump": min_rms_jump,
        "binding_signature": binding_signature,
        "execution_source_signature": source_signature,
        "runner_contract": {
            "num_results": int(runner.config.num_results),
            "num_burnin_steps": int(runner.config.num_burnin_steps),
            "use_xla": bool(runner.config.use_xla),
            "trace_policy": str(runner.config.trace_policy),
            "target_scope": str(runner.config.target_scope),
            "dynamic_num_leapfrog_steps": bool(
                runner.dynamic_num_leapfrog_steps
            ),
        },
    }
    if resume and path.is_file():
        existing = strict_json(path)
        if existing.get("contract") != json_safe(contract):
            raise ComplexityHMCError("existing arm contract mismatch")
        result = dict(existing["result"])
        budget.observe(result["timing"]["seconds_per_transition_leapfrog"])
        return result
    transitions = int(runner.config.num_results + runner.config.num_burnin_steps)
    reserve = budget.arm_reserve(
        transitions=transitions,
        leapfrog_steps=leapfrog_steps,
        cold_runner=int(getattr(runner, "_call_count", 0)) == 0,
    )
    budget.require(reserve)
    checkpoint(reserve)
    result = run_arm(
        runner,
        step_size=step_size,
        leapfrog_steps=leapfrog_steps,
        seed_word=seed_word,
        role=stage,
        acceptance_band=acceptance_band,
        min_movement=min_movement,
        min_rms_jump=min_rms_jump,
    )
    result["resource_reserve_seconds_before_launch"] = reserve
    budget.observe(result["timing"]["seconds_per_transition_leapfrog"])
    write_json(path, {"schema": SCHEMA, "contract": contract, "result": result})
    checkpoint(0.0)
    return result


def tune_transport(
    *,
    output: Path,
    label: str,
    adapter: Any,
    binding: Mapping[str, Any],
    source_signature: str,
    budget: Budget,
    resume: bool,
    seed_offset: int,
    checkpoint: Callable[[float], None],
) -> dict[str, Any]:
    pilot_runner = build_runner(
        adapter,
        num_results=PILOT_RESULTS,
        num_burnin_steps=PILOT_BURNIN,
        leapfrog_steps=4,
    )
    scale_rows = []
    for index, step in enumerate(INITIAL_SCALE_GRID):
        scale_rows.append(
            run_or_reuse_arm(
                output=output,
                label=label,
                stage="scale",
                index=index,
                runner=pilot_runner,
                budget=budget,
                step_size=step,
                leapfrog_steps=4,
                seed_word=seed_offset + index,
                acceptance_band=PILOT_ACCEPTANCE_BAND,
                min_movement=PILOT_MIN_MOVEMENT,
                min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                binding_signature=str(binding["binding_signature"]),
                source_signature=source_signature,
                resume=resume,
                checkpoint=checkpoint,
            )
        )
    expansion = scale_expansion(scale_rows)
    expansion_executed = []
    for local_index, step in enumerate(expansion):
        index = len(INITIAL_SCALE_GRID) + local_index
        scale_rows.append(
            run_or_reuse_arm(
                output=output,
                label=label,
                stage="scale",
                index=index,
                runner=pilot_runner,
                budget=budget,
                step_size=step,
                leapfrog_steps=4,
                seed_word=seed_offset + index,
                acceptance_band=PILOT_ACCEPTANCE_BAND,
                min_movement=PILOT_MIN_MOVEMENT,
                min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                binding_signature=str(binding["binding_signature"]),
                source_signature=source_signature,
                resume=resume,
                checkpoint=checkpoint,
            )
        )
        expansion_executed.append(step)
        if select_scale(scale_rows) is not None or scale_bracket_repair(scale_rows) is not None:
            break
    selected_scale = select_scale(scale_rows)
    bracket_repair_step = None
    if selected_scale is None:
        bracket_repair_step = scale_bracket_repair(scale_rows)
        if bracket_repair_step is not None:
            scale_rows.append(
                run_or_reuse_arm(
                    output=output,
                    label=label,
                    stage="scale-bracket-repair",
                    index=0,
                    runner=pilot_runner,
                    budget=budget,
                    step_size=bracket_repair_step,
                    leapfrog_steps=4,
                    seed_word=seed_offset + 50,
                    acceptance_band=PILOT_ACCEPTANCE_BAND,
                    min_movement=PILOT_MIN_MOVEMENT,
                    min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                    binding_signature=str(binding["binding_signature"]),
                    source_signature=source_signature,
                    resume=resume,
                    checkpoint=checkpoint,
                )
            )
            selected_scale = select_scale(scale_rows)
    trajectory_rows = []
    selected_trajectory = None
    confirmation = None
    repair = None
    if selected_scale is not None:
        step_size = float(selected_scale["step_size"])
        for index, leapfrog in enumerate(TRAJECTORY_GRID):
            trajectory_rows.append(
                run_or_reuse_arm(
                    output=output,
                    label=label,
                    stage="trajectory",
                    index=index,
                    runner=pilot_runner,
                    budget=budget,
                    step_size=step_size,
                    leapfrog_steps=leapfrog,
                    seed_word=seed_offset + 100 + index,
                    acceptance_band=PILOT_ACCEPTANCE_BAND,
                    min_movement=PILOT_MIN_MOVEMENT,
                    min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                    binding_signature=str(binding["binding_signature"]),
                    source_signature=source_signature,
                    resume=resume,
                    checkpoint=checkpoint,
                )
            )
        selected_trajectory = select_trajectory(trajectory_rows)
    if selected_trajectory is not None:
        leapfrog = int(selected_trajectory["num_leapfrog_steps"])
        confirmation_runner = build_runner(
            adapter,
            num_results=CONFIRMATION_RESULTS,
            num_burnin_steps=CONFIRMATION_BURNIN,
            leapfrog_steps=leapfrog,
        )
        confirmation = run_or_reuse_arm(
            output=output,
            label=label,
            stage="confirmation",
            index=0,
            runner=confirmation_runner,
            budget=budget,
            step_size=float(selected_trajectory["step_size"]),
            leapfrog_steps=leapfrog,
            seed_word=seed_offset + 200,
            acceptance_band=CONFIRMATION_ACCEPTANCE_BAND,
            min_movement=CONFIRMATION_MIN_MOVEMENT,
            min_rms_jump=MIN_RMS_JUMP_DISTANCE,
            binding_signature=str(binding["binding_signature"]),
            source_signature=source_signature,
            resume=resume,
            checkpoint=checkpoint,
        )
        if not confirmation["diagnostics"]["viable"]:
            adjacent = adjacent_repair_candidate(confirmation, trajectory_rows)
            if adjacent is not None:
                repair_l = int(adjacent["num_leapfrog_steps"])
                repair_runner = build_runner(
                    adapter,
                    num_results=CONFIRMATION_RESULTS,
                    num_burnin_steps=CONFIRMATION_BURNIN,
                    leapfrog_steps=repair_l,
                )
                repair = run_or_reuse_arm(
                    output=output,
                    label=label,
                    stage="adjacent-repair",
                    index=0,
                    runner=repair_runner,
                    budget=budget,
                    step_size=float(adjacent["step_size"]),
                    leapfrog_steps=repair_l,
                    seed_word=seed_offset + 201,
                    acceptance_band=CONFIRMATION_ACCEPTANCE_BAND,
                    min_movement=CONFIRMATION_MIN_MOVEMENT,
                    min_rms_jump=MIN_RMS_JUMP_DISTANCE,
                    binding_signature=str(binding["binding_signature"]),
                    source_signature=source_signature,
                    resume=resume,
                    checkpoint=checkpoint,
                )
    admitted = (
        repair
        if repair is not None and repair["diagnostics"]["viable"]
        else confirmation
        if confirmation is not None and confirmation["diagnostics"]["viable"]
        else None
    )
    return {
        "label": label,
        "status": "ADMITTED" if admitted is not None else "VETOED",
        "scale_rows": scale_rows,
        "scale_expansion": list(expansion),
        "scale_expansion_executed": list(expansion_executed),
        "scale_bracket_repair_step": bracket_repair_step,
        "selected_scale": selected_scale,
        "trajectory_rows": trajectory_rows,
        "selected_trajectory": selected_trajectory,
        "confirmation": confirmation,
        "adjacent_repair": repair,
        "selected_kernel": (
            None
            if admitted is None
            else {
                "mass_matrix": "identity",
                "step_size": admitted["step_size"],
                "num_leapfrog_steps": admitted["num_leapfrog_steps"],
                "trajectory_length": admitted["trajectory_length"],
                "target_acceptance": TARGET_ACCEPTANCE,
                "confirmation_seed": admitted["seed"],
            }
        ),
    }


def source_bindings() -> dict[str, Any]:
    paths = {
        "runner": SCRIPT,
        "plan": PLAN,
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "adapter": Path("bayesfilter/inference/batched_value_score.py"),
        "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        "hmc": Path("bayesfilter/inference/hmc.py"),
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_paths": {key: path.as_posix() for key, path in paths.items()},
        "source_sha256": {key: sha256(ROOT / path) for key, path in paths.items()},
    }


def execution_source_signature() -> str:
    paths = (
        SCRIPT,
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        Path("bayesfilter/inference/batched_value_score.py"),
        Path("bayesfilter/inference/neutra_artifacts.py"),
        Path("bayesfilter/inference/hmc.py"),
    )
    return payload_sha256({path.as_posix(): sha256(ROOT / path) for path in paths})


def material_contract(args: argparse.Namespace, source_signature: str) -> dict[str, Any]:
    receipts = {}
    for label, value in (
        ("chart-a", args.phase3_result_a),
        ("chart-b", args.phase3_result_b),
    ):
        path = repo_path(value, label=f"{label} Phase 3 result")
        receipts[label] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }
    return {
        "q": int(args.q),
        "phase3_receipts": receipts,
        "execution_source_signature": source_signature,
        "selected_hmc_topology": "single_tfp_sample_chain_batched_four_chain_xla",
    }


def run_manifest(args: argparse.Namespace, budget: Budget) -> dict[str, Any]:
    try:
        gpu_allocator_memory = json_safe(
            tf.config.experimental.get_memory_info("GPU:0")
        )
    except (ValueError, RuntimeError):
        gpu_allocator_memory = {"status": "unavailable"}
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "selected_physical_gpu": SELECTED_GPU,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "gpu_memory_growth_verified": all(
            tf.config.experimental.get_memory_growth(gpu) is True
            for gpu in tf.config.list_physical_devices("GPU")
        ),
        "jit_compile": True,
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "charged_seconds": budget.elapsed,
        "cap_seconds": budget.seconds,
        "host_ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "gpu_allocator_memory": gpu_allocator_memory,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "plan": PLAN.as_posix(),
        "output_root": args.output_root.as_posix(),
    }


def configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise ComplexityHMCError("transformed-HMC execution requires a visible GPU")
    for gpu in gpus:
        if tf.config.experimental.get_memory_growth(gpu) is not True:
            raise ComplexityHMCError("GPU memory growth verification failed")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (ValueError, RuntimeError):
        pass
    return gpus


def validate_material_args(args: argparse.Namespace) -> None:
    if args.mode != "contract-smoke" and not args.authorize_material_run:
        raise ComplexityHMCError("preflight/tune requires --authorize-material-run")
    if args.mode != "contract-smoke":
        if args.cap_seconds is None or args.cap_seconds <= 0.0:
            raise ComplexityHMCError("material modes require a positive cumulative cap")
        if args.output_root is None:
            raise ComplexityHMCError("material modes require an explicit output root")
        if args.phase3_result_a is None or args.phase3_result_b is None:
            raise ComplexityHMCError("material modes require two Phase 3 result receipts")
        repo_path(args.output_root, label="output root")
        repo_path(args.phase3_result_a, label="Phase 3 result a")
        repo_path(args.phase3_result_b, label="Phase 3 result b")


def resume_prior_seconds(
    *,
    args: argparse.Namespace,
    summary_path: Path,
    checkpoint_path: Path,
    contract: Mapping[str, Any],
) -> float:
    if not args.resume:
        return 0.0
    if summary_path.is_file():
        previous = strict_json(summary_path)
        expected_schema = SCHEMA
        charged = previous.get("run_manifest", {}).get("charged_seconds")
    elif checkpoint_path.is_file():
        previous = strict_json(checkpoint_path)
        expected_schema = CHECKPOINT_SCHEMA
        charged = previous.get("charged_seconds")
    else:
        raise ComplexityHMCError("resume requires summary.json or checkpoint.json")
    if previous.get("schema") != expected_schema or int(previous.get("q", -1)) != args.q:
        raise ComplexityHMCError("resume artifact mismatch")
    if previous.get("material_contract") != contract:
        raise ComplexityHMCError("resume material contract mismatch")
    value = float(charged)
    if not math.isfinite(value) or value < 0.0:
        raise ComplexityHMCError("resume charged-seconds value is invalid")
    return value


def contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": args.q,
        "required_phase3_result_count": 2,
        "initial_z": [list(row) for row in INITIAL_Z],
        "scale_grid": list(INITIAL_SCALE_GRID),
        "trajectory_grid": list(TRAJECTORY_GRID),
        "trajectory_priority": list(TRAJECTORY_PRIORITY),
        "pilot_acceptance_band": list(PILOT_ACCEPTANCE_BAND),
        "confirmation_acceptance_band": list(CONFIRMATION_ACCEPTANCE_BAND),
        "target_acceptance": TARGET_ACCEPTANCE,
        "repair": "one adjacent trajectory confirmation for acceptance-only failure",
        "timing_normalization": "seconds_per_transition_leapfrog",
        "selected_hmc_topology": "single_tfp_sample_chain_batched_four_chain_xla",
        "material_execution_authorized": False,
        "resume_binding": (
            "each arm binds exact admitted Phase 3 receipt, payload hash, target, "
            "transport, runner contract, and execution source signature"
        ),
        "resource_guard": {
            "cumulative_cap": True,
            "first_compiled_arm_reserve_seconds": FIRST_COMPILED_ARM_RESERVE_SECONDS,
            "observed_arm_cost_margin": ARM_COST_MARGIN,
            "boundary": (
                "a non-preemptive HMC arm may overshoot its prospective reserve; "
                "the next arm cannot start after the cap guard fires"
            ),
        },
        "source_bindings": source_bindings(),
        "nonclaims": [
            "contract/import smoke only",
            "no transport loading or target evaluation",
            "no HMC execution or kernel nomination",
        ],
    }


def _run_material_locked(
    args: argparse.Namespace, output: Path
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    checkpoint_path = output / "checkpoint.json"
    source_signature = execution_source_signature()
    contract = material_contract(args, source_signature)
    prior_seconds = resume_prior_seconds(
        args=args,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        contract=contract,
    )
    budget = Budget(args.cap_seconds, prior_seconds=prior_seconds)

    def checkpoint(pending_reserve_seconds: float) -> None:
        write_json(
            checkpoint_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "q": args.q,
                "material_contract": contract,
                "charged_seconds": budget.elapsed + float(pending_reserve_seconds),
                "pending_nonpreemptive_reserve_seconds": float(
                    pending_reserve_seconds
                ),
            },
            replace=True,
        )

    bindings = {}
    adapters = {}
    preflights = {}
    preflight_timing = {}
    resource_stop = None
    hard_veto = None
    try:
        budget.require(2.0 * FIRST_COMPILED_ARM_RESERVE_SECONDS)
        preflight_inputs = (
            ("chart-a", args.phase3_result_a),
            ("chart-b", args.phase3_result_b),
        )
        for index, (label, path) in enumerate(preflight_inputs):
            checkpoint(
                (len(preflight_inputs) - index) * FIRST_COMPILED_ARM_RESERVE_SECONDS
            )
            started = time.perf_counter()
            adapter, binding = load_binding(args.q, label, path)
            adapters[label] = adapter
            bindings[label] = binding
            preflights[label] = transformed_preflight(adapter)
            preflight_timing[label] = time.perf_counter() - started
            host_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            if host_rss > HOST_RAM_CAP_BYTES:
                raise HostMemoryVeto(
                    "transformed-target preflight RSS exceeded 64 GiB"
                )
        checkpoint(0.0)
        validate_distinct_bindings(bindings)
    except ResourceStop as exc:
        resource_stop = str(exc)
    except HostMemoryVeto as exc:
        hard_veto = str(exc)
    except ComplexityHMCError as exc:
        hard_veto = str(exc)
    preflight_vetoes = {
        label: row["vetoes"] for label, row in preflights.items() if row["vetoes"]
    }
    results = {}
    if (
        args.mode == "tune"
        and not preflight_vetoes
        and resource_stop is None
        and hard_veto is None
    ):
        try:
            for index, label in enumerate(("chart-a", "chart-b")):
                results[label] = tune_transport(
                    output=output,
                    label=label,
                    adapter=adapters[label],
                    binding=bindings[label],
                    source_signature=source_signature,
                    budget=budget,
                    resume=args.resume,
                    seed_offset=15100 + 1000 * args.q + index * 300,
                    checkpoint=checkpoint,
                )
        except ResourceStop as exc:
            resource_stop = str(exc)
        except HostMemoryVeto as exc:
            hard_veto = str(exc)
    status = (
        "HARD_VETO"
        if hard_veto is not None
        else "RESOURCE_STOP"
        if resource_stop is not None
        else "PREFLIGHT_VETO"
        if preflight_vetoes
        else "PREFLIGHT_PASSED"
        if args.mode == "preflight"
        else "KERNELS_FROZEN"
        if len(results) == 2 and all(row["status"] == "ADMITTED" for row in results.values())
        else "TUNING_REPAIR_REQUIRED"
    )
    payload = {
        "schema": SCHEMA,
        "mode": args.mode,
        "status": status,
        "q": args.q,
        "material_contract": contract,
        "bindings": bindings,
        "preflights": preflights,
        "preflight_timing_seconds": preflight_timing,
        "preflight_vetoes": preflight_vetoes,
        "tuning": results,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "candidate_veto": False if resource_stop or hard_veto else None,
        "continuation_veto": hard_veto is not None,
        "scientific_interpretation": "none",
        "source_bindings": source_bindings(),
        "execution_source_signature": source_signature,
        "run_manifest": run_manifest(args, budget),
        "inference_status": {
            "hard_veto_screen": "see_preflight_and_per_arm_diagnostics",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "acceptance",
                "movement",
                "RMS jump distance",
                "runtime",
                "seconds per transition-leapfrog",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "independent retained-chain admission"
                if status == "KERNELS_FROZEN"
                else "declared repair or additional authority"
            ),
        },
        "nonclaims": [
            "tuning viability only",
            "no transport or kernel ranking",
            "no retained posterior evidence or convergence claim",
            "native divergence unavailability is not zero divergences",
        ],
    }
    write_json(summary_path, payload, replace=args.resume)
    return payload


def run_material(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    with output_writer_lock(output):
        return _run_material_locked(args, output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "preflight", "tune"), required=True)
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--phase3-result-a", type=Path)
    parser.add_argument("--phase3-result-b", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cap-seconds", type=float)
    parser.add_argument("--authorize-material-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cap_seconds is not None and (
        not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0
    ):
        parser.error("--cap-seconds must be finite and positive")
    if args.mode == "contract-smoke" and args.resume:
        parser.error("contract smoke cannot resume")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_material_args(args)
    if args.mode == "contract-smoke":
        payload = contract_payload(args)
    else:
        configure_gpu()
        payload = run_material(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {"mode": payload["mode"], "status": payload["status"], "q": payload["q"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
