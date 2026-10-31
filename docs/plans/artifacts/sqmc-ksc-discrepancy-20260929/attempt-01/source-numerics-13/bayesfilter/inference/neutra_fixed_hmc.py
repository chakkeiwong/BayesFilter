"""Fixed-length HMC for a frozen nonlinear NeuTra transport.

The controller deliberately has no NUTS or adaptive-mass path. A scalar step
size is calibrated on discarded transitions, then a new fixed HMC kernel is
used for discarded qualification and retained shards. The target remains the
exact transformed density supplied by the adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.hmc_posterior_diagnostics import (
    compute_coordinate_diagnostics,
    per_chain_ebfmi,
)


tfd = tfp.distributions
FIXED_HMC_POLICY_ID = "bayesfilter_neutra_fixed_hmc_v1"
FIXED_HMC_SHARD_SCHEMA = "bayesfilter.neutra.fixed_hmc_tensor_shard.v1"
FIXED_HMC_CALIBRATION_SCHEMA = "bayesfilter.neutra.fixed_hmc_calibration.v1"


class FixedNeuTraHMCError(RuntimeError):
    """Raised when the fixed-HMC evidence contract is violated."""


@dataclass(frozen=True)
class FixedNeuTraHMCConfig:
    """Prospective fixed-HMC and discarded scalar calibration settings."""

    dimension: int
    chain_count: int = 4
    leapfrog_steps: int = 3
    initial_step_size: float = 0.1
    calibration_updates: int = 6
    calibration_transitions_per_update: int = 32
    calibration_target_accept: float = 0.70
    calibration_multiplier: float = 1.5
    warmup_chunk_size: int = 500
    warmup_min_results: int = 2000
    warmup_max_results: int = 10000
    retained_chunk_size: int = 500
    retained_max_results: int = 10000
    rhat_max: float = 1.01
    warmup_rhat_max: float = 1.05
    bulk_ess_min: float = 400.0
    tail_ess_min: float = 400.0
    mcse_sd_ratio_max: float = 0.05
    ebfmi_min: float = 0.30
    delta_h_abs_max: float = 1000.0
    use_xla: bool = True
    seed: tuple[int, int] = (20260730, 9305)

    def __post_init__(self) -> None:
        for name in (
            "dimension", "chain_count", "leapfrog_steps", "calibration_updates",
            "calibration_transitions_per_update", "warmup_chunk_size",
            "warmup_min_results", "warmup_max_results", "retained_chunk_size",
            "retained_max_results",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.chain_count < 2:
            raise ValueError("fixed HMC requires at least two chains")
        if self.warmup_min_results > self.warmup_max_results:
            raise ValueError("warmup minimum exceeds cap")
        for total, chunk in ((self.warmup_min_results, self.warmup_chunk_size),
                             (self.warmup_max_results, self.warmup_chunk_size),
                             (self.retained_max_results, self.retained_chunk_size)):
            if total % chunk:
                raise ValueError("HMC caps must be multiples of chunk size")
        for name in ("initial_step_size", "calibration_multiplier", "delta_h_abs_max",
                     "rhat_max", "warmup_rhat_max", "bulk_ess_min", "tail_ess_min",
                     "mcse_sd_ratio_max", "ebfmi_min"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        target = float(self.calibration_target_accept)
        if not 0.0 < target < 1.0 or not math.isfinite(target):
            raise ValueError("calibration_target_accept must lie in (0, 1)")
        object.__setattr__(self, "calibration_target_accept", target)
        if self.rhat_max <= 1.0 or self.warmup_rhat_max <= 1.0:
            raise ValueError("R-hat thresholds must exceed one")
        object.__setattr__(self, "seed", tuple(int(v) for v in self.seed))
        if len(self.seed) != 2:
            raise ValueError("seed must contain two integers")

    def payload(self) -> Mapping[str, Any]:
        return {"policy_id": FIXED_HMC_POLICY_ID, **asdict(self)}


@dataclass(frozen=True)
class FixedNeuTraHMCKernel:
    """Hash-bound scalar step size and fixed trajectory length."""

    dimension: int
    chain_count: int
    step_size: float
    leapfrog_steps: int
    use_xla: bool = True
    momentum_metric: str = "identity"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if int(self.dimension) <= 0 or int(self.chain_count) < 2 or int(self.leapfrog_steps) <= 0:
            raise ValueError("invalid fixed kernel dimensions or trajectory")
        object.__setattr__(self, "step_size", float(self.step_size))
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "chain_count", int(self.chain_count))
        object.__setattr__(self, "leapfrog_steps", int(self.leapfrog_steps))
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        if self.momentum_metric != "identity":
            raise ValueError("only identity momentum is permitted")

    def payload(self) -> Mapping[str, Any]:
        body = {"schema": "bayesfilter.neutra.fixed_hmc_kernel.v1", **asdict(self)}
        return {**body, "kernel_hash": _stable_hash(body)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FixedNeuTraHMCKernel":
        body = dict(payload)
        supplied = str(body.pop("kernel_hash", ""))
        if body.pop("schema", None) != "bayesfilter.neutra.fixed_hmc_kernel.v1":
            raise FixedNeuTraHMCError("unsupported fixed-HMC kernel schema")
        if supplied != _stable_hash({"schema": "bayesfilter.neutra.fixed_hmc_kernel.v1", **body}):
            raise FixedNeuTraHMCError("fixed-HMC kernel hash mismatch")
        return cls(**body)


@dataclass(frozen=True)
class FixedNeuTraHMCChunk:
    initial_state: tf.Tensor
    samples: tf.Tensor
    trace: Mapping[str, tf.Tensor]
    final_state: tf.Tensor


def _stable_hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _status(adapter: Any, state: tf.Tensor) -> Mapping[str, tf.Tensor]:
    raw = adapter.target_status_telemetry(state)
    required = ("status_code", "valid_pre_regularized_score", "invalid_count", "roundoff_repair_count")
    missing = [name for name in required if name not in raw]
    if missing:
        raise FixedNeuTraHMCError("target status telemetry missing: " + ", ".join(missing))
    return {
        "target_status_code": tf.cast(raw["status_code"], tf.int32),
        "target_valid_pre_regularized_score": tf.cast(raw["valid_pre_regularized_score"], tf.bool),
        "target_invalid_count": tf.cast(raw["invalid_count"], tf.int32),
        "target_roundoff_repair_count": tf.cast(raw["roundoff_repair_count"], tf.int32),
    }


def _trace(adapter: Any, state: tf.Tensor, results: Any) -> Mapping[str, tf.Tensor]:
    accepted = results.accepted_results
    final_momentum = tf.cast(accepted.final_momentum[0], tf.float64)
    target_log_prob = tf.cast(accepted.target_log_prob, tf.float64)
    energy = -target_log_prob + 0.5 * tf.reduce_sum(tf.square(final_momentum), axis=-1)
    return {
        "target_log_prob": target_log_prob,
        "target_score": tf.cast(accepted.grads_target_log_prob[0], tf.float64),
        "transport_log_abs_det_jacobian": tf.cast(adapter.log_abs_det_jacobian(state), tf.float64),
        "log_accept_ratio": tf.cast(results.log_accept_ratio, tf.float64),
        "delta_h": tf.cast(-results.log_accept_ratio, tf.float64),
        "is_accepted": tf.cast(results.is_accepted, tf.bool),
        "energy": energy,
        "step_size": tf.fill(tf.shape(target_log_prob), tf.constant(0.0, tf.float64)),
        "leapfrog_steps": tf.fill(tf.shape(results.is_accepted), tf.constant(0, tf.int32)),
        **_status(adapter, state),
    }


class FixedNeuTraHMCChunkRunner:
    """Reusable compiled fixed-HMC runner; kernel settings cannot vary."""

    def __init__(
        self,
        adapter: Any,
        state_template: Any,
        kernel: FixedNeuTraHMCKernel,
        *,
        chunk_size: int,
        mutable_step_size: bool = False,
    ):
        state = tf.convert_to_tensor(state_template, tf.float64)
        expected = (kernel.chain_count, kernel.dimension)
        if state.shape != expected:
            raise ValueError(f"state_template must have shape {expected}")
        self.kernel = kernel
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._step_size = tf.Variable(
            kernel.step_size, dtype=tf.float64, trainable=False, name="fixed_hmc_step_size"
        )
        self._mutable_step_size = bool(mutable_step_size)
        target = reviewed_value_score_target_fn(adapter, dtype=tf.float64, require_batched=True)
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target,
            step_size=self._step_size,
            num_leapfrog_steps=kernel.leapfrog_steps,
        )

        def run(current_state: tf.Tensor, seed: tf.Tensor) -> Any:
            return tfp.mcmc.sample_chain(
                num_results=self.chunk_size,
                num_burnin_steps=0,
                current_state=current_state,
                kernel=hmc,
                trace_fn=lambda values, results: _trace(adapter, values, results),
                seed=seed,
            )

        self._run = tf.function(
            run,
            input_signature=(tf.TensorSpec(expected, tf.float64), tf.TensorSpec((2,), tf.int32)),
            jit_compile=bool(kernel.use_xla),
            reduce_retracing=True,
        )

    def run(self, initial_state: Any, seed: Sequence[int]) -> FixedNeuTraHMCChunk:
        state = tf.convert_to_tensor(initial_state, tf.float64)
        samples, trace = self._run(state, tf.constant(tuple(int(v) for v in seed), tf.int32))
        trace = {str(k): tf.convert_to_tensor(v) for k, v in trace.items()}
        trace["step_size"] = tf.fill(tf.shape(trace["log_accept_ratio"]), self._step_size)
        trace["leapfrog_steps"] = tf.fill(tf.shape(trace["is_accepted"]), tf.constant(self.kernel.leapfrog_steps, tf.int32))
        return FixedNeuTraHMCChunk(state, tf.cast(samples, tf.float64), trace, tf.cast(samples[-1], tf.float64))

    def set_step_size(self, step_size: float) -> None:
        """Update calibration step size without rebuilding the compiled graph."""
        if not self._mutable_step_size:
            raise FixedNeuTraHMCError("step size is immutable for this chunk runner")
        value = float(step_size)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("step_size must be positive and finite")
        self._step_size.assign(value)


def _all_finite(tensors: Sequence[tf.Tensor]) -> bool:
    return bool(all(tf.reduce_all(tf.math.is_finite(tf.cast(t, tf.float64))).numpy() for t in tensors))


def calibrate_fixed_hmc_step_size(
    adapter: Any,
    initial_state: Any,
    config: FixedNeuTraHMCConfig,
) -> tuple[FixedNeuTraHMCKernel, Mapping[str, Any], tf.Tensor]:
    """Calibrate one scalar step size; every transition is excluded evidence."""
    state = tf.convert_to_tensor(initial_state, tf.float64)
    step = float(config.initial_step_size)
    rows = []
    initial_kernel = FixedNeuTraHMCKernel(
        config.dimension, config.chain_count, step, config.leapfrog_steps, config.use_xla
    )
    runner = FixedNeuTraHMCChunkRunner(
        adapter,
        state,
        initial_kernel,
        chunk_size=config.calibration_transitions_per_update,
        mutable_step_size=True,
    )
    for update in range(config.calibration_updates):
        runner.set_step_size(step)
        chunk = runner.run(state, (config.seed[0], config.seed[1] + update + 1))
        accepted = tf.reduce_mean(tf.cast(chunk.trace["is_accepted"], tf.float64), axis=0)
        mean_accept = float(tf.reduce_mean(accepted).numpy())
        if not _all_finite((chunk.samples, chunk.trace["target_log_prob"], chunk.trace["delta_h"], chunk.trace["energy"])):
            raise FixedNeuTraHMCError("step-size calibration produced nonfinite values")
        rows.append({"update": update, "step_size": step, "acceptance_by_chain": _json_ready(accepted), "mean_acceptance": mean_accept})
        state = chunk.final_state
        if mean_accept > config.calibration_target_accept + 0.05:
            step *= config.calibration_multiplier
        elif mean_accept < config.calibration_target_accept - 0.05:
            step /= config.calibration_multiplier
    kernel = FixedNeuTraHMCKernel(config.dimension, config.chain_count, step, config.leapfrog_steps, config.use_xla)
    payload = {"schema": FIXED_HMC_CALIBRATION_SCHEMA, "kernel": kernel.payload(), "updates": rows, "all_draws_discarded": True, "nonclaims": ["step-size calibration only", "no convergence or posterior claim"]}
    return kernel, payload, state


def compute_fixed_hmc_diagnostics(adapter: Any, samples: tf.Tensor, trace_parts: Sequence[Mapping[str, tf.Tensor]], *, rhat_max: float = 1.01, bulk_ess_min: float = 400.0, tail_ess_min: float = 400.0, mcse_sd_ratio_max: float = 0.05, ebfmi_min: float = 0.30, delta_h_abs_max: float = 1000.0) -> Mapping[str, Any]:
    chain_major = tf.transpose(samples, (1, 0, 2))
    mapped = tf.reshape(adapter.latent_to_position(tf.reshape(samples, (-1, int(samples.shape[-1])))), tf.shape(samples))
    reports = {"neutra_latent_z": compute_coordinate_diagnostics(chain_major), "physical_theta": compute_coordinate_diagnostics(tf.transpose(tf.reshape(mapped, tf.shape(samples)), (1, 0, 2)))}
    coordinate_passed = True
    extrema = {}
    for name, report in reports.items():
        rhat = tf.cast(report["rank_normalized_split_rhat"]["maximum"], tf.float64)
        bulk = tf.cast(report["rank_normalized_ess"]["bulk"], tf.float64)
        tail = tf.cast(report["rank_normalized_ess"]["tail"], tf.float64)
        ratio = tf.cast(report["mean"]["mcse_sd_ratio"], tf.float64)
        passed = bool(tf.reduce_all(rhat < rhat_max).numpy() and tf.reduce_all(bulk >= bulk_ess_min).numpy() and tf.reduce_all(tail >= tail_ess_min).numpy() and tf.reduce_all(ratio <= mcse_sd_ratio_max).numpy())
        coordinate_passed = coordinate_passed and passed
        extrema[name] = {"passed": passed, "max_rhat": float(tf.reduce_max(rhat).numpy()), "min_bulk_ess": float(tf.reduce_min(bulk).numpy()), "min_tail_ess": float(tf.reduce_min(tail).numpy()), "max_mcse_sd_ratio": float(tf.reduce_max(ratio).numpy())}
    energy = tf.transpose(tf.concat([part["energy"] for part in trace_parts], axis=0))
    delta_h = tf.concat([part["delta_h"] for part in trace_parts], axis=0)
    accepted = tf.concat([part["is_accepted"] for part in trace_parts], axis=0)
    ebfmi = per_chain_ebfmi(energy)
    finite = _all_finite((samples, mapped, energy, delta_h))
    mechanics = bool(finite and tf.reduce_all(ebfmi > ebfmi_min).numpy() and tf.reduce_all(tf.abs(delta_h) <= delta_h_abs_max).numpy() and tf.reduce_all(tf.reduce_any(tf.not_equal(samples[1:], samples[:-1]), axis=(0, 2))).numpy())
    return {"schema": "bayesfilter.neutra.fixed_hmc_convergence.v1", "passed": bool(coordinate_passed and mechanics), "coordinate_passed": coordinate_passed, "mechanics_passed": mechanics, "coordinate_extrema": extrema, "ebfmi_by_chain": _json_ready(ebfmi), "acceptance_fraction_by_chain": _json_ready(tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)), "maximum_abs_delta_h": float(tf.reduce_max(tf.abs(delta_h)).numpy()), "all_finite": finite, "thresholds": {"rhat_max_exclusive": rhat_max, "bulk_ess_min": bulk_ess_min, "tail_ess_min": tail_ess_min, "mcse_sd_ratio_max": mcse_sd_ratio_max, "ebfmi_min_exclusive": ebfmi_min, "delta_h_abs_max": delta_h_abs_max}}


__all__ = ["FIXED_HMC_POLICY_ID", "FixedNeuTraHMCConfig", "FixedNeuTraHMCKernel", "FixedNeuTraHMCChunk", "FixedNeuTraHMCChunkRunner", "FixedNeuTraHMCError", "calibrate_fixed_hmc_step_size", "compute_fixed_hmc_diagnostics"]
