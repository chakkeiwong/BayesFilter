"""Adaptive NUTS and frozen-kernel sampling for fixed nonlinear NeuTra maps.

The controller uses TFP's windowed dual-averaging/diagonal-mass adaptation,
then reconstructs a separate preconditioned NUTS kernel from persisted numeric
state. Adaptation draws and production draws are returned through distinct
APIs so callers cannot retain warmup accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
tfb = tfp.bijectors

ADAPTIVE_NEUTRA_NUTS_POLICY_ID = "bayesfilter_neutra_windowed_nuts_v1"
NUTS_SHARD_SCHEMA = "bayesfilter.neutra.nuts_tensor_shard.v1"
NUTS_ADAPTATION_SHARD_SCHEMA = "bayesfilter.neutra.nuts_adaptation_shard.v1"
PINNED_TFP_VERSION = "0.25.0"

NUTS_TRACE_FIELDS = (
    "target_log_prob",
    "target_score",
    "transport_log_abs_det_jacobian",
    "log_accept_ratio",
    "step_size",
    "leapfrogs_taken",
    "is_accepted",
    "reach_max_depth",
    "has_divergence",
    "negative_hamiltonian",
    "target_status_code",
    "target_valid_pre_regularized_score",
    "target_invalid_count",
    "target_roundoff_repair_count",
)


class AdaptiveNeuTraNUTSError(RuntimeError):
    """Raised when adaptive or frozen NeuTra NUTS violates its contract."""


@dataclass(frozen=True)
class AdaptiveNeuTraNUTSConfig:
    """Prospective windowed-adaptation and frozen NUTS settings."""

    dimension: int
    chain_count: int = 4
    adaptation_results: int = 2000
    target_accept_prob: float = 0.80
    max_tree_depth: int = 10
    max_energy_diff: float = 500.0
    parallel_iterations: int = 10
    seed: tuple[int, int] = (20260730, 9305)

    def __post_init__(self) -> None:
        for name in (
            "dimension",
            "chain_count",
            "adaptation_results",
            "max_tree_depth",
            "parallel_iterations",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.chain_count < 2:
            raise ValueError("adaptive NUTS requires at least two chains")
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must lie in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        energy = float(self.max_energy_diff)
        if not math.isfinite(energy) or energy <= 0.0:
            raise ValueError("max_energy_diff must be positive and finite")
        object.__setattr__(self, "max_energy_diff", energy)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        object.__setattr__(self, "seed", seed)

    def payload(self) -> Mapping[str, Any]:
        return {"policy_id": ADAPTIVE_NEUTRA_NUTS_POLICY_ID, **asdict(self)}


@dataclass(frozen=True)
class FrozenNeuTraNUTSKernel:
    """Persistable numeric state for one post-adaptation NUTS kernel."""

    dimension: int
    chain_count: int
    step_size: float
    position_variance: tuple[tuple[float, ...], ...]
    target_accept_prob: float
    max_tree_depth: int
    max_energy_diff: float
    source_policy_id: str = ADAPTIVE_NEUTRA_NUTS_POLICY_ID

    def __post_init__(self) -> None:
        dimension = int(self.dimension)
        chains = int(self.chain_count)
        if dimension <= 0 or chains < 2:
            raise ValueError("frozen kernel dimension/chains are invalid")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "chain_count", chains)
        step = float(self.step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("frozen step_size must be positive and finite")
        object.__setattr__(self, "step_size", step)
        variance = tuple(tuple(float(item) for item in row) for row in self.position_variance)
        if len(variance) != chains or any(len(row) != dimension for row in variance):
            raise ValueError("position_variance must have shape [chain, dimension]")
        if any(not math.isfinite(item) or item <= 0.0 for row in variance for item in row):
            raise ValueError("position_variance must be positive and finite")
        object.__setattr__(self, "position_variance", variance)
        if self.source_policy_id != ADAPTIVE_NEUTRA_NUTS_POLICY_ID:
            raise ValueError("unsupported adaptive NUTS source policy")

    def payload(self) -> Mapping[str, Any]:
        body = {
            "schema": "bayesfilter.neutra.frozen_nuts_kernel.v1",
            **asdict(self),
        }
        return {**body, "kernel_hash": _stable_hash(body)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenNeuTraNUTSKernel":
        body = dict(payload)
        supplied = str(body.pop("kernel_hash", ""))
        if body.pop("schema", None) != "bayesfilter.neutra.frozen_nuts_kernel.v1":
            raise AdaptiveNeuTraNUTSError("unsupported frozen NUTS kernel schema")
        check = {"schema": "bayesfilter.neutra.frozen_nuts_kernel.v1", **body}
        if supplied != _stable_hash(check):
            raise AdaptiveNeuTraNUTSError("frozen NUTS kernel hash mismatch")
        return cls(**body)


@dataclass(frozen=True)
class AdaptiveNeuTraNUTSResult:
    """Adaptation tensors and the extracted frozen-kernel state."""

    states: tf.Tensor
    trace: Mapping[str, tf.Tensor]
    frozen_kernel: FrozenNeuTraNUTSKernel


@dataclass(frozen=True)
class FrozenNeuTraNUTSChunk:
    """One production chunk from an immutable frozen kernel."""

    initial_state: tf.Tensor
    samples: tf.Tensor
    trace: Mapping[str, tf.Tensor]
    final_state: tf.Tensor


class _PotentialDistribution(tfd.Distribution):
    """Unnormalized vector density with an explicit identity event chart."""

    def __init__(self, target_log_prob_fn: Any, dimension: int) -> None:
        self._target_log_prob_fn = target_log_prob_fn
        self._dimension = int(dimension)
        super().__init__(
            dtype=tf.float64,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
            parameters={},
            name="neutra_latent_potential",
        )

    def _batch_shape(self) -> tf.TensorShape:
        return tf.TensorShape([])

    def _batch_shape_tensor(self) -> tf.Tensor:
        return tf.constant([], tf.int32)

    def _event_shape(self) -> tf.TensorShape:
        return tf.TensorShape([self._dimension])

    def _event_shape_tensor(self) -> tf.Tensor:
        return tf.constant([self._dimension], tf.int32)

    def _log_prob(self, value: Any) -> tf.Tensor:
        return self._target_log_prob_fn(value)

    def _unnormalized_log_prob(self, value: Any) -> tf.Tensor:
        return self._target_log_prob_fn(value)

    def _sample_n(self, n: Any, seed: Any = None) -> tf.Tensor:
        sanitized = tfp.random.sanitize_seed(seed, salt="neutra_potential_sample")
        return tf.random.stateless_normal(
            (n, self._dimension), seed=sanitized, dtype=tf.float64
        )

    def _default_event_space_bijector(self) -> tfb.Bijector:
        return tfb.Identity(validate_args=False)


def _core_status(adapter: Any, state: tf.Tensor) -> Mapping[str, tf.Tensor]:
    method = getattr(adapter, "target_status_telemetry", None)
    if not callable(method):
        raise AdaptiveNeuTraNUTSError("adapter must expose target_status_telemetry")
    status = method(state)
    if not isinstance(status, Mapping):
        raise AdaptiveNeuTraNUTSError("target status telemetry must be a mapping")
    required = (
        "status_code",
        "valid_pre_regularized_score",
        "invalid_count",
        "roundoff_repair_count",
    )
    missing = tuple(name for name in required if name not in status)
    if missing:
        raise AdaptiveNeuTraNUTSError(
            "target status telemetry missing core fields: " + ", ".join(missing)
        )
    return {
        "target_status_code": tf.cast(status["status_code"], tf.int32),
        "target_valid_pre_regularized_score": tf.cast(
            status["valid_pre_regularized_score"], tf.bool
        ),
        "target_invalid_count": tf.cast(status["invalid_count"], tf.int32),
        "target_roundoff_repair_count": tf.cast(
            status["roundoff_repair_count"], tf.int32
        ),
    }


def _nuts_trace(adapter: Any, state: tf.Tensor, results: Any) -> Mapping[str, tf.Tensor]:
    logdet = getattr(adapter, "log_abs_det_jacobian", None)
    if not callable(logdet):
        raise AdaptiveNeuTraNUTSError("adapter must expose log_abs_det_jacobian")
    return {
        "target_log_prob": tf.cast(results.target_log_prob, tf.float64),
        "target_score": tf.cast(
            _single_tensor(results.grads_target_log_prob, "NUTS target score"),
            tf.float64,
        ),
        "transport_log_abs_det_jacobian": tf.cast(logdet(state), tf.float64),
        "log_accept_ratio": tf.cast(results.log_accept_ratio, tf.float64),
        "step_size": tf.cast(results.step_size, tf.float64),
        "leapfrogs_taken": tf.cast(results.leapfrogs_taken, tf.int32),
        "is_accepted": tf.cast(results.is_accepted, tf.bool),
        "reach_max_depth": tf.cast(results.reach_max_depth, tf.bool),
        "has_divergence": tf.cast(results.has_divergence, tf.bool),
        # TFP stores log-density minus kinetic energy, the negative Hamiltonian.
        "negative_hamiltonian": tf.cast(results.energy, tf.float64),
        **_core_status(adapter, state),
    }


def _windowed_trace(
    adapter: Any,
    state: tf.Tensor,
    _bijector: Any,
    is_adapting: tf.Tensor,
    results: Any,
) -> Mapping[str, tf.Tensor]:
    state_tensor = _single_tensor(state, "windowed trace state")
    return {
        "is_adapting": tf.cast(is_adapting, tf.bool),
        **_nuts_trace(adapter, state_tensor, results),
    }


def run_windowed_adaptive_neutra_nuts(
    adapter: Any,
    initial_state: Any,
    config: AdaptiveNeuTraNUTSConfig,
) -> AdaptiveNeuTraNUTSResult:
    """Run adaptation only and extract persistable frozen NUTS state."""

    state = tf.convert_to_tensor(initial_state, tf.float64)
    expected = (config.chain_count, config.dimension)
    if state.shape != expected:
        raise ValueError(f"initial_state must have shape {expected}")
    target = reviewed_value_score_target_fn(adapter, dtype=tf.float64, require_batched=True)
    joint = tfd.JointDistributionSequential([_PotentialDistribution(target, config.dimension)])
    result = tfp.experimental.mcmc.windowed_adaptive_nuts(
        n_draws=0,
        joint_dist=joint,
        n_chains=config.chain_count,
        num_adaptation_steps=config.adaptation_results,
        current_state=[state],
        init_step_size=None,
        dual_averaging_kwargs={"target_accept_prob": config.target_accept_prob},
        max_tree_depth=config.max_tree_depth,
        max_energy_diff=config.max_energy_diff,
        parallel_iterations=config.parallel_iterations,
        trace_fn=lambda values, bijector, adapting, results: _windowed_trace(
            adapter, values, bijector, adapting, results
        ),
        return_final_kernel_results=True,
        discard_tuning=False,
        seed=config.seed,
    )
    states = _single_tensor(result.all_states, "adaptation states")
    if states.shape != (config.adaptation_results, *expected):
        raise AdaptiveNeuTraNUTSError("adaptation state shape mismatch")
    windowed = result.final_kernel_results
    mass = _require_inner(windowed, "windowed adaptation")
    dual = _require_inner(mass, "diagonal mass adaptation")
    _require_inner(dual, "dual averaging")
    step_size = _single_tensor(dual.new_step_size, "adapted step size")
    if step_size.shape.rank != 0:
        raise AdaptiveNeuTraNUTSError("adapted step size must be scalar")
    running = tuple(mass.running_variance)
    if len(running) != 1:
        raise AdaptiveNeuTraNUTSError("adaptive NUTS requires one state part")
    variance = tf.cast(running[0].variance(), tf.float64)
    if variance.shape != expected:
        raise AdaptiveNeuTraNUTSError("adapted position variance shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(step_size)).numpy()) or not bool(
        step_size > 0.0
    ):
        raise AdaptiveNeuTraNUTSError("adapted step size is nonfinite or nonpositive")
    if not bool(tf.reduce_all(tf.math.is_finite(variance)).numpy()) or not bool(
        tf.reduce_all(variance > 0.0).numpy()
    ):
        raise AdaptiveNeuTraNUTSError("adapted position variance is nonfinite or nonpositive")
    frozen = FrozenNeuTraNUTSKernel(
        dimension=config.dimension,
        chain_count=config.chain_count,
        step_size=float(step_size.numpy()),
        position_variance=tuple(
            tuple(float(item) for item in row) for row in variance.numpy().tolist()
        ),
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        max_energy_diff=config.max_energy_diff,
    )
    trace = {str(name): tf.convert_to_tensor(value) for name, value in result.trace.items()}
    if tuple(trace) != ("is_adapting", *NUTS_TRACE_FIELDS):
        raise AdaptiveNeuTraNUTSError("adaptation trace schema mismatch")
    return AdaptiveNeuTraNUTSResult(states=states, trace=trace, frozen_kernel=frozen)


class FrozenNeuTraNUTSChunkRunner:
    """Reusable compiled runner for one persisted frozen NUTS kernel."""

    def __init__(
        self,
        adapter: Any,
        state_template: Any,
        frozen_kernel: FrozenNeuTraNUTSKernel,
        *,
        chunk_size: int,
        use_xla: bool = False,
    ) -> None:
        state = tf.convert_to_tensor(state_template, tf.float64)
        expected = (frozen_kernel.chain_count, frozen_kernel.dimension)
        if state.shape != expected:
            raise ValueError(f"state_template must have shape {expected}")
        self.adapter = adapter
        self.frozen_kernel = frozen_kernel
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        target = reviewed_value_score_target_fn(adapter, dtype=tf.float64, require_batched=True)
        variance = tf.constant(frozen_kernel.position_variance, tf.float64)
        momentum = _make_momentum_distribution(state, variance)
        kernel = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=target,
            step_size=tf.constant(frozen_kernel.step_size, tf.float64),
            momentum_distribution=momentum,
            max_tree_depth=frozen_kernel.max_tree_depth,
            max_energy_diff=frozen_kernel.max_energy_diff,
        )

        def run(current_state: tf.Tensor, seed: tf.Tensor) -> Any:
            return tfp.mcmc.sample_chain(
                num_results=self.chunk_size,
                current_state=current_state,
                kernel=kernel,
                num_burnin_steps=0,
                trace_fn=lambda values, results: _nuts_trace(adapter, values, results),
                seed=seed,
            )

        self._run = tf.function(
            run,
            input_signature=(
                tf.TensorSpec(expected, tf.float64),
                tf.TensorSpec((2,), tf.int32),
            ),
            jit_compile=bool(use_xla),
            reduce_retracing=True,
        )

    def run(self, initial_state: Any, seed: Sequence[int]) -> FrozenNeuTraNUTSChunk:
        state = tf.convert_to_tensor(initial_state, tf.float64)
        samples, trace = self._run(state, tf.constant(tuple(int(item) for item in seed), tf.int32))
        samples = tf.cast(samples, tf.float64)
        trace = {str(name): tf.convert_to_tensor(value) for name, value in trace.items()}
        if tuple(trace) != NUTS_TRACE_FIELDS:
            raise AdaptiveNeuTraNUTSError("production NUTS trace schema mismatch")
        return FrozenNeuTraNUTSChunk(
            initial_state=state,
            samples=samples,
            trace=trace,
            final_state=samples[-1],
        )


def write_nuts_tensor_shard(
    chunk: FrozenNeuTraNUTSChunk,
    *,
    path: str | Path,
    role: str,
    block_index: int,
    global_start_index: int,
    kernel_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Persist one immutable TensorFlow shard and verify exact readback."""

    if role not in {"frozen_qualification", "posterior"}:
        raise ValueError("invalid NUTS shard role")
    root = Path(path)
    if root.exists():
        raise FileExistsError(root)
    temporary = root.with_name(root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    tensors = {
        "initial_state": chunk.initial_state,
        "samples": chunk.samples,
        "final_state": chunk.final_state,
        **chunk.trace,
    }
    receipts = _write_tensor_receipts(temporary, tensors)
    count = int(chunk.samples.shape[0])
    manifest = {
        "schema": NUTS_SHARD_SCHEMA,
        "role": role,
        "block_index": int(block_index),
        "global_start_index": int(global_start_index),
        "global_end_index_exclusive": int(global_start_index) + count,
        "kernel_payload": _json_ready(kernel_payload),
        "metadata": _json_ready({} if metadata is None else metadata),
        "tensors": receipts,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    temporary.rename(root)
    readback = read_nuts_tensor_shard(root)
    for name, tensor in tensors.items():
        tf.debugging.assert_equal(readback["tensors"][name], tf.convert_to_tensor(tensor))
    return {
        **manifest,
        "path": str(root.resolve()),
        "manifest_sha256": _sha256_file(root / "manifest.json"),
        "readback_verified": True,
    }


def write_nuts_adaptation_shard(
    result: AdaptiveNeuTraNUTSResult,
    *,
    initial_state: Any,
    path: str | Path,
    kernel_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Persist all excluded adaptation states and traces in a distinct schema."""

    root = Path(path)
    if root.exists():
        raise FileExistsError(root)
    temporary = root.with_name(root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    state = tf.convert_to_tensor(initial_state, tf.float64)
    tensors = {
        "initial_state": state,
        "states": result.states,
        "final_state": result.states[-1],
        **result.trace,
    }
    receipts = _write_tensor_receipts(temporary, tensors)
    manifest = {
        "schema": NUTS_ADAPTATION_SHARD_SCHEMA,
        "role": "adaptation_excluded",
        "excluded_from_posterior": True,
        "global_start_index": 0,
        "global_end_index_exclusive": int(result.states.shape[0]),
        "kernel_payload": _json_ready(kernel_payload),
        "metadata": _json_ready({} if metadata is None else metadata),
        "tensors": receipts,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    temporary.rename(root)
    readback = read_nuts_adaptation_shard(root)
    for name, tensor in tensors.items():
        tf.debugging.assert_equal(readback["tensors"][name], tf.convert_to_tensor(tensor))
    return {
        **manifest,
        "path": str(root.resolve()),
        "manifest_sha256": _sha256_file(root / "manifest.json"),
        "readback_verified": True,
    }


def read_nuts_adaptation_shard(path: str | Path) -> Mapping[str, Any]:
    """Read an excluded adaptation shard; it is never a retained-shard reader."""

    root = Path(path)
    manifest, tensors = _read_tensor_manifest(root, NUTS_ADAPTATION_SHARD_SCHEMA)
    expected = {
        "initial_state",
        "states",
        "final_state",
        "is_adapting",
        *NUTS_TRACE_FIELDS,
    }
    if set(tensors) != expected:
        raise AdaptiveNeuTraNUTSError("NUTS adaptation tensor schema mismatch")
    if manifest.get("role") != "adaptation_excluded" or not bool(
        manifest.get("excluded_from_posterior", False)
    ):
        raise AdaptiveNeuTraNUTSError("NUTS adaptation exclusion marker is missing")
    count = int(tensors["states"].shape[0])
    if int(manifest.get("global_start_index", -1)) != 0 or int(
        manifest.get("global_end_index_exclusive", -1)
    ) != count:
        raise AdaptiveNeuTraNUTSError("NUTS adaptation index count mismatch")
    if not bool(tf.reduce_all(tensors["is_adapting"]).numpy()):
        raise AdaptiveNeuTraNUTSError("NUTS adaptation shard contains non-adapting draws")
    if not bool(
        tf.reduce_all(tf.equal(tensors["final_state"], tensors["states"][-1])).numpy()
    ):
        raise AdaptiveNeuTraNUTSError("NUTS adaptation final-state handoff mismatch")
    return {"manifest": manifest, "tensors": tensors}


def read_nuts_tensor_shard(path: str | Path) -> Mapping[str, Any]:
    """Read and hash-validate one immutable TensorFlow NUTS shard."""

    root = Path(path)
    manifest, tensors = _read_tensor_manifest(root, NUTS_SHARD_SCHEMA)
    expected = {"initial_state", "samples", "final_state", *NUTS_TRACE_FIELDS}
    if set(tensors) != expected:
        raise AdaptiveNeuTraNUTSError("NUTS shard tensor schema mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(tensors["initial_state"])).numpy()):
        raise AdaptiveNeuTraNUTSError("NUTS shard initial state is nonfinite")
    if not bool(tf.reduce_all(tf.equal(tensors["final_state"], tensors["samples"][-1])).numpy()):
        raise AdaptiveNeuTraNUTSError("NUTS shard final-state handoff mismatch")
    return {"manifest": manifest, "tensors": tensors}


def compute_neutra_nuts_adaptation_readiness(
    adapter: Any,
    shard: Mapping[str, Any],
    *,
    recent_rhat_draws: int = 1000,
    recent_mechanics_draws: int = 500,
    rhat_max_inclusive: float = 1.05,
    max_depth_fraction_exclusive: float = 0.10,
) -> Mapping[str, Any]:
    """Apply the prospective late-window gate to excluded adaptation draws."""

    if shard.get("manifest", {}).get("schema") != NUTS_ADAPTATION_SHARD_SCHEMA:
        raise AdaptiveNeuTraNUTSError("adaptation readiness requires adaptation schema")
    tensors = shard["tensors"]
    states = tf.cast(tensors["states"], tf.float64)
    draws = int(states.shape[0])
    rhat_draws = int(recent_rhat_draws)
    mechanics_draws = int(recent_mechanics_draws)
    if rhat_draws < 4 or rhat_draws % 2 or mechanics_draws < 1:
        raise ValueError("adaptation readiness windows are invalid")
    if draws < max(rhat_draws, mechanics_draws):
        raise AdaptiveNeuTraNUTSError("adaptation shard is shorter than readiness windows")
    latent = tf.transpose(states[-rhat_draws:], (1, 0, 2))
    flat = tf.reshape(states[-rhat_draws:], (-1, int(states.shape[-1])))
    physical_rows = tf.cast(adapter.latent_to_position(flat), tf.float64)
    physical = tf.transpose(
        tf.reshape(physical_rows, tf.shape(states[-rhat_draws:])), (1, 0, 2)
    )
    reports = {
        "neutra_latent_z": compute_coordinate_diagnostics(latent),
        "physical_theta": compute_coordinate_diagnostics(physical),
    }
    extrema = {}
    rhat_passed = True
    for name, report in reports.items():
        values = tf.cast(report["rank_normalized_split_rhat"]["maximum"], tf.float64)
        passed = bool(
            tf.reduce_all(tf.math.is_finite(values)).numpy()
            and tf.reduce_all(values <= rhat_max_inclusive).numpy()
        )
        rhat_passed = rhat_passed and passed
        extrema[name] = {"passed": passed, "max_rhat": float(tf.reduce_max(values).numpy())}
    tail = slice(draws - mechanics_draws, draws)
    divergence = tensors["has_divergence"][tail]
    max_depth = tensors["reach_max_depth"][tail]
    max_depth_fraction = tf.reduce_mean(tf.cast(max_depth, tf.float64), axis=0)
    pre = tf.concat((tensors["initial_state"][None, ...], states[:-1]), axis=0)
    moved = tf.reduce_any(tf.not_equal(states, pre), axis=(0, 2))
    float_trace_names = (
        "target_log_prob",
        "target_score",
        "transport_log_abs_det_jacobian",
        "step_size",
        "negative_hamiltonian",
    )
    all_finite = bool(
        tf.reduce_all(tf.math.is_finite(states)).numpy()
        and tf.reduce_all(tf.math.is_finite(physical_rows)).numpy()
        and all(
            bool(tf.reduce_all(tf.math.is_finite(tensors[name])).numpy())
            for name in float_trace_names
        )
    )
    mechanics_passed = bool(
        all_finite
        and not tf.reduce_any(divergence).numpy()
        and tf.reduce_all(max_depth_fraction < max_depth_fraction_exclusive).numpy()
        and tf.reduce_all(moved).numpy()
        and tf.reduce_all(tf.equal(tensors["target_status_code"], 0)).numpy()
        and tf.reduce_all(tensors["target_valid_pre_regularized_score"]).numpy()
        and tf.reduce_all(tf.equal(tensors["target_invalid_count"], 0)).numpy()
        and tf.reduce_all(tf.equal(tensors["target_roundoff_repair_count"], 0)).numpy()
    )
    return {
        "schema": "bayesfilter.neutra.nuts_adaptation_readiness.v1",
        "draws_per_chain": draws,
        "passed": bool(rhat_passed and mechanics_passed),
        "rhat_passed": rhat_passed,
        "mechanics_passed": mechanics_passed,
        "coordinate_extrema": extrema,
        "coordinate_diagnostics": _json_ready(reports),
        "divergence_count_by_chain_recent": _json_ready(
            tf.reduce_sum(tf.cast(divergence, tf.int64), axis=0)
        ),
        "max_depth_fraction_by_chain_recent": _json_ready(max_depth_fraction),
        "chain_moved": _json_ready(moved),
        "all_finite": all_finite,
        "thresholds": {
            "recent_rhat_draws": rhat_draws,
            "recent_mechanics_draws": mechanics_draws,
            "rhat_max_inclusive": rhat_max_inclusive,
            "max_depth_fraction_exclusive": max_depth_fraction_exclusive,
        },
    }


def compute_frozen_neutra_nuts_qualification(
    shard: Mapping[str, Any],
    frozen_kernel: FrozenNeuTraNUTSKernel,
    *,
    max_depth_fraction_exclusive: float = 0.10,
) -> Mapping[str, Any]:
    """Check one excluded segment produced by the reconstructed frozen kernel."""

    manifest = shard.get("manifest", {})
    tensors = shard.get("tensors", {})
    if manifest.get("schema") != NUTS_SHARD_SCHEMA or manifest.get(
        "role"
    ) != "frozen_qualification":
        raise AdaptiveNeuTraNUTSError(
            "frozen qualification requires a frozen_qualification shard"
        )
    if manifest.get("kernel_payload") != _json_ready(frozen_kernel.payload()):
        raise AdaptiveNeuTraNUTSError("frozen qualification kernel lineage mismatch")
    samples = tf.cast(tensors["samples"], tf.float64)
    pre = tf.concat((tensors["initial_state"][None, ...], samples[:-1]), axis=0)
    moved = tf.reduce_any(tf.not_equal(samples, pre), axis=(0, 2))
    divergence = tensors["has_divergence"]
    max_depth = tensors["reach_max_depth"]
    max_depth_fraction = tf.reduce_mean(tf.cast(max_depth, tf.float64), axis=0)
    float_trace_names = (
        "target_log_prob",
        "target_score",
        "transport_log_abs_det_jacobian",
        "log_accept_ratio",
        "step_size",
        "negative_hamiltonian",
    )
    all_finite = bool(
        tf.reduce_all(tf.math.is_finite(samples)).numpy()
        and all(
            bool(tf.reduce_all(tf.math.is_finite(tensors[name])).numpy())
            for name in float_trace_names
        )
    )
    step_size_matches = bool(
        tf.reduce_all(
            tf.equal(
                tf.cast(tensors["step_size"], tf.float64),
                tf.constant(frozen_kernel.step_size, tf.float64),
            )
        ).numpy()
    )
    passed = bool(
        all_finite
        and step_size_matches
        and not tf.reduce_any(divergence).numpy()
        and tf.reduce_all(max_depth_fraction < max_depth_fraction_exclusive).numpy()
        and tf.reduce_all(moved).numpy()
        and tf.reduce_all(tf.equal(tensors["target_status_code"], 0)).numpy()
        and tf.reduce_all(tensors["target_valid_pre_regularized_score"]).numpy()
        and tf.reduce_all(tf.equal(tensors["target_invalid_count"], 0)).numpy()
        and tf.reduce_all(tf.equal(tensors["target_roundoff_repair_count"], 0)).numpy()
    )
    return {
        "schema": "bayesfilter.neutra.frozen_nuts_qualification.v1",
        "draws_per_chain": int(samples.shape[0]),
        "passed": passed,
        "all_finite": all_finite,
        "step_size_matches": step_size_matches,
        "chain_moved": _json_ready(moved),
        "divergence_count_by_chain": _json_ready(
            tf.reduce_sum(tf.cast(divergence, tf.int64), axis=0)
        ),
        "max_depth_fraction_by_chain": _json_ready(max_depth_fraction),
        "target_status_nonzero_count": int(
            tf.reduce_sum(
                tf.cast(tf.not_equal(tensors["target_status_code"], 0), tf.int64)
            ).numpy()
        ),
        "target_invalid_count": int(
            tf.reduce_sum(tf.cast(tensors["target_invalid_count"], tf.int64)).numpy()
        ),
        "target_roundoff_repair_count": int(
            tf.reduce_sum(
                tf.cast(tensors["target_roundoff_repair_count"], tf.int64)
            ).numpy()
        ),
        "thresholds": {
            "max_depth_fraction_exclusive": max_depth_fraction_exclusive,
            "zero_divergences": True,
        },
    }


def compute_retained_neutra_nuts_diagnostics(
    adapter: Any,
    shards: Sequence[Mapping[str, Any]],
    *,
    rhat_max_exclusive: float = 1.01,
    bulk_ess_min: float = 400.0,
    tail_ess_min: float = 400.0,
    mcse_sd_ratio_max: float = 0.05,
    ebfmi_min_exclusive: float = 0.30,
    max_depth_fraction_exclusive: float = 0.05,
) -> Mapping[str, Any]:
    """Compute terminal diagnostics from validated immutable shard tensors."""

    if not shards:
        raise ValueError("at least one retained shard is required")
    expected_index = 0
    previous_final = None
    kernel_hash = None
    for shard in shards:
        manifest = shard.get("manifest", {})
        tensors = shard.get("tensors", {})
        if (
            manifest.get("schema") != NUTS_SHARD_SCHEMA
            or manifest.get("role") != "posterior"
        ):
            raise AdaptiveNeuTraNUTSError("retained diagnostics require posterior shards")
        current_kernel_hash = _stable_hash(manifest.get("kernel_payload", {}))
        if kernel_hash is None:
            kernel_hash = current_kernel_hash
        elif current_kernel_hash != kernel_hash:
            raise AdaptiveNeuTraNUTSError("retained shard kernel lineage mismatch")
        if int(manifest.get("global_start_index", -1)) != expected_index:
            raise AdaptiveNeuTraNUTSError("retained shard indices are not contiguous")
        count = int(tensors["samples"].shape[0])
        if int(manifest.get("global_end_index_exclusive", -1)) != expected_index + count:
            raise AdaptiveNeuTraNUTSError("retained shard index count mismatch")
        if previous_final is not None and not bool(
            tf.reduce_all(tf.equal(previous_final, tensors["initial_state"])).numpy()
        ):
            raise AdaptiveNeuTraNUTSError("retained shard state handoff mismatch")
        expected_index += count
        previous_final = tensors["final_state"]
    samples = tf.concat([row["tensors"]["samples"] for row in shards], axis=0)
    chain_major = tf.transpose(samples, (1, 0, 2))
    flat = tf.reshape(samples, (-1, int(samples.shape[-1])))
    mapped = tf.reshape(adapter.latent_to_position(flat), tf.shape(samples))
    physical = tf.transpose(mapped, (1, 0, 2))
    reports = {
        "neutra_latent_z": compute_coordinate_diagnostics(chain_major),
        "physical_theta": compute_coordinate_diagnostics(physical),
    }
    extrema = {}
    coordinate_passed = True
    for name, report in reports.items():
        rhat = tf.cast(report["rank_normalized_split_rhat"]["maximum"], tf.float64)
        bulk = tf.cast(report["rank_normalized_ess"]["bulk"], tf.float64)
        tail = tf.cast(report["rank_normalized_ess"]["tail"], tf.float64)
        ratio = tf.cast(report["mean"]["mcse_sd_ratio"], tf.float64)
        finite = tf.reduce_all(
            tf.math.is_finite(tf.concat((rhat, bulk, tail, ratio), axis=0))
        )
        passed = bool(
            finite.numpy()
            and tf.reduce_all(rhat < rhat_max_exclusive).numpy()
            and tf.reduce_all(bulk >= bulk_ess_min).numpy()
            and tf.reduce_all(tail >= tail_ess_min).numpy()
            and tf.reduce_all(ratio <= mcse_sd_ratio_max).numpy()
        )
        coordinate_passed = coordinate_passed and passed
        extrema[name] = {
            "passed": passed,
            "max_rhat": float(tf.reduce_max(rhat).numpy()),
            "min_bulk_ess": float(tf.reduce_min(bulk).numpy()),
            "min_tail_ess": float(tf.reduce_min(tail).numpy()),
            "max_mcse_sd_ratio": float(tf.reduce_max(ratio).numpy()),
        }
    energy = tf.transpose(
        tf.concat([row["tensors"]["negative_hamiltonian"] for row in shards], axis=0)
    )
    ebfmi = per_chain_ebfmi(energy)
    divergence = tf.concat([row["tensors"]["has_divergence"] for row in shards], axis=0)
    max_depth = tf.concat([row["tensors"]["reach_max_depth"] for row in shards], axis=0)
    status = tf.concat([row["tensors"]["target_status_code"] for row in shards], axis=0)
    valid = tf.concat(
        [row["tensors"]["target_valid_pre_regularized_score"] for row in shards], axis=0
    )
    invalid = tf.concat([row["tensors"]["target_invalid_count"] for row in shards], axis=0)
    repair = tf.concat(
        [row["tensors"]["target_roundoff_repair_count"] for row in shards], axis=0
    )
    max_depth_fraction = tf.reduce_mean(tf.cast(max_depth, tf.float64), axis=0)
    pre = tf.concat((shards[0]["tensors"]["initial_state"][None, ...], samples[:-1]), axis=0)
    moved = tf.reduce_any(tf.not_equal(samples, pre), axis=(0, 2))
    duplicate_pairs = []
    for left in range(int(samples.shape[1])):
        for right in range(left + 1, int(samples.shape[1])):
            if bool(tf.reduce_all(tf.equal(samples[:, left], samples[:, right])).numpy()):
                duplicate_pairs.append((left, right))
    all_finite = bool(
        tf.reduce_all(tf.math.is_finite(samples)).numpy()
        and tf.reduce_all(tf.math.is_finite(mapped)).numpy()
        and tf.reduce_all(tf.math.is_finite(energy)).numpy()
        and tf.reduce_all(
            tf.math.is_finite(
                tf.concat([row["tensors"]["target_log_prob"] for row in shards], axis=0)
            )
        ).numpy()
        and tf.reduce_all(
            tf.math.is_finite(
                tf.concat([row["tensors"]["target_score"] for row in shards], axis=0)
            )
        ).numpy()
        and tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [row["tensors"]["transport_log_abs_det_jacobian"] for row in shards],
                    axis=0,
                )
            )
        ).numpy()
        and tf.reduce_all(
            tf.math.is_finite(
                tf.concat([row["tensors"]["log_accept_ratio"] for row in shards], axis=0)
            )
        ).numpy()
        and tf.reduce_all(
            tf.math.is_finite(
                tf.concat([row["tensors"]["step_size"] for row in shards], axis=0)
            )
        ).numpy()
    )
    mechanics_passed = bool(
        all_finite
        and tf.reduce_all(tf.math.is_finite(ebfmi)).numpy()
        and tf.reduce_all(ebfmi > ebfmi_min_exclusive).numpy()
        and not tf.reduce_any(divergence).numpy()
        and tf.reduce_all(max_depth_fraction < max_depth_fraction_exclusive).numpy()
        and tf.reduce_all(moved).numpy()
        and not duplicate_pairs
        and tf.reduce_all(tf.equal(status, 0)).numpy()
        and tf.reduce_all(valid).numpy()
        and tf.reduce_all(tf.equal(invalid, 0)).numpy()
        and tf.reduce_all(tf.equal(repair, 0)).numpy()
    )
    return {
        "schema": "bayesfilter.neutra.nuts_convergence_diagnostics.v1",
        "draws_per_chain": int(samples.shape[0]),
        "passed": bool(coordinate_passed and mechanics_passed),
        "coordinate_passed": coordinate_passed,
        "mechanics_passed": mechanics_passed,
        "thresholds": {
            "rhat_max_exclusive": rhat_max_exclusive,
            "bulk_ess_min": bulk_ess_min,
            "tail_ess_min": tail_ess_min,
            "mcse_sd_ratio_max": mcse_sd_ratio_max,
            "ebfmi_min_exclusive": ebfmi_min_exclusive,
            "max_depth_fraction_exclusive": max_depth_fraction_exclusive,
        },
        "coordinate_extrema": extrema,
        "coordinate_diagnostics": _json_ready(reports),
        "ebfmi_by_chain": _json_ready(ebfmi),
        "acceptance_fraction_by_chain": _json_ready(
            tf.reduce_mean(
                tf.cast(
                    tf.concat(
                        [row["tensors"]["is_accepted"] for row in shards], axis=0
                    ),
                    tf.float64,
                ),
                axis=0,
            )
        ),
        "mean_accept_probability_by_chain": _json_ready(
            tf.reduce_mean(
                tf.exp(
                    tf.minimum(
                        tf.concat(
                            [row["tensors"]["log_accept_ratio"] for row in shards],
                            axis=0,
                        ),
                        tf.constant(0.0, tf.float64),
                    )
                ),
                axis=0,
            )
        ),
        "mean_leapfrogs_by_chain": _json_ready(
            tf.reduce_mean(
                tf.cast(
                    tf.concat(
                        [row["tensors"]["leapfrogs_taken"] for row in shards], axis=0
                    ),
                    tf.float64,
                ),
                axis=0,
            )
        ),
        "divergence_count_by_chain": _json_ready(
            tf.reduce_sum(tf.cast(divergence, tf.int64), axis=0)
        ),
        "max_depth_fraction_by_chain": _json_ready(max_depth_fraction),
        "chain_moved": _json_ready(moved),
        "duplicate_chain_pairs": duplicate_pairs,
        "target_status_nonzero_count": int(
            tf.reduce_sum(tf.cast(tf.not_equal(status, 0), tf.int64)).numpy()
        ),
        "target_invalid_count": int(tf.reduce_sum(tf.cast(invalid, tf.int64)).numpy()),
        "target_roundoff_repair_count": int(
            tf.reduce_sum(tf.cast(repair, tf.int64)).numpy()
        ),
        "all_finite": all_finite,
    }


def _make_momentum_distribution(state: tf.Tensor, position_variance: tf.Tensor) -> Any:
    # This helper is part of TFP's adaptation module surface and reconstructs
    # the same precision convention used by DiagonalMassMatrixAdaptation.
    if tfp.__version__ != PINNED_TFP_VERSION:
        raise AdaptiveNeuTraNUTSError(
            f"momentum reconstruction requires TFP {PINNED_TFP_VERSION}; got {tfp.__version__}"
        )
    from tensorflow_probability.python.experimental.mcmc import preconditioning_utils

    return preconditioning_utils.make_momentum_distribution(
        [state], tf.shape(state)[:-1], [position_variance]
    )


def _require_inner(value: Any, label: str) -> Any:
    inner = getattr(value, "inner_results", None)
    if inner is None:
        raise AdaptiveNeuTraNUTSError(f"{label} result nesting mismatch")
    return inner


def _single_tensor(value: Any, label: str) -> tf.Tensor:
    flat = tf.nest.flatten(value)
    if len(flat) != 1:
        raise AdaptiveNeuTraNUTSError(f"{label} must contain one tensor")
    return tf.convert_to_tensor(flat[0])


def _stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _write_tensor_receipts(root: Path, tensors: Mapping[str, Any]) -> Mapping[str, Any]:
    receipts = {}
    for name, tensor in tensors.items():
        value = tf.convert_to_tensor(tensor)
        blob = bytes(tf.io.serialize_tensor(value).numpy())
        destination = root / f"{name}.tftensor"
        destination.write_bytes(blob)
        receipts[name] = {
            "file": destination.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "shape": list(value.shape),
            "dtype": value.dtype.name,
        }
    return receipts


def _read_tensor_manifest(
    root: Path, schema: str
) -> tuple[Mapping[str, Any], Mapping[str, tf.Tensor]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise AdaptiveNeuTraNUTSError("NUTS shard schema mismatch")
    tensors = {}
    for name, receipt in manifest.get("tensors", {}).items():
        source = root / str(receipt["file"])
        blob = source.read_bytes()
        if hashlib.sha256(blob).hexdigest() != receipt.get("sha256"):
            raise AdaptiveNeuTraNUTSError(f"NUTS shard hash mismatch: {name}")
        tensor = tf.io.parse_tensor(blob, out_type=tf.as_dtype(receipt["dtype"]))
        tensors[str(name)] = tf.ensure_shape(tensor, receipt["shape"])
    return manifest, tensors


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if tf.is_tensor(value):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ADAPTIVE_NEUTRA_NUTS_POLICY_ID",
    "NUTS_ADAPTATION_SHARD_SCHEMA",
    "PINNED_TFP_VERSION",
    "AdaptiveNeuTraNUTSError",
    "AdaptiveNeuTraNUTSConfig",
    "AdaptiveNeuTraNUTSResult",
    "FrozenNeuTraNUTSChunk",
    "FrozenNeuTraNUTSChunkRunner",
    "FrozenNeuTraNUTSKernel",
    "compute_frozen_neutra_nuts_qualification",
    "compute_retained_neutra_nuts_diagnostics",
    "compute_neutra_nuts_adaptation_readiness",
    "read_nuts_adaptation_shard",
    "read_nuts_tensor_shard",
    "run_windowed_adaptive_neutra_nuts",
    "write_nuts_tensor_shard",
    "write_nuts_adaptation_shard",
]
