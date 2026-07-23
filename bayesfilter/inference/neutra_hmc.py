"""TensorFlow/TFP sequential HMC for fixed BayesFilter NeuTra transports.

Warm-up and retained samples are archived in separate TensorFlow tensor shards.
Warm-up readiness and retained admission use rank-normalized split/folded R-hat;
retained admission additionally uses rank-normalized bulk/tail ESS. Passing is
a finite-sample operational screen, not proof of stationarity or correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_bulk_tail_ess,
    rank_normalized_split_rhat,
)


NEUTRA_SEQUENTIAL_HMC_POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"
SEQUENTIAL_NEUTRA_HMC_SCHEMA = "bayesfilter.neutra.sequential_hmc_result.v1"


class NeuTraHMCError(RuntimeError):
    """Raised when sequential NeuTra-HMC execution violates its contract."""


@dataclass(frozen=True)
class SequentialNeuTraHMCConfig:
    """Fixed-kernel sequential warm-up and retained-sampling policy."""

    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    warmup_chunk_size: int = 500
    warmup_min_results: int = 2000
    warmup_window_results: int = 1000
    warmup_max_results: int = 10000
    retained_chunk_size: int = 500
    retained_min_results: int = 1000
    retained_max_results: int = 10000
    warmup_rhat_max: float = 1.05
    retained_rhat_max: float = 1.01
    bulk_ess_min: float = 400.0
    tail_ess_min: float = 400.0
    delta_h_abs_max: float = 1000.0
    chain_count: int = 4
    use_xla: bool = True
    target_status_required: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", float(self.step_size))
        for name in (
            "num_leapfrog_steps",
            "warmup_chunk_size",
            "warmup_min_results",
            "warmup_window_results",
            "warmup_max_results",
            "retained_chunk_size",
            "retained_min_results",
            "retained_max_results",
            "chain_count",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.chain_count < 2:
            raise ValueError("sequential HMC requires at least two chains")
        if self.warmup_window_results > self.warmup_min_results:
            raise ValueError("warmup window cannot exceed minimum warm-up")
        if self.warmup_window_results < 4:
            raise ValueError("warmup diagnostic window requires at least four draws")
        if self.warmup_min_results > self.warmup_max_results:
            raise ValueError("warmup minimum cannot exceed maximum")
        if self.retained_min_results > self.retained_max_results:
            raise ValueError("retained minimum cannot exceed maximum")
        if self.retained_min_results < 4:
            raise ValueError("retained diagnostics require at least four draws")
        for total_name, chunk_name in (
            ("warmup_min_results", "warmup_chunk_size"),
            ("warmup_window_results", "warmup_chunk_size"),
            ("warmup_max_results", "warmup_chunk_size"),
            ("retained_min_results", "retained_chunk_size"),
            ("retained_max_results", "retained_chunk_size"),
        ):
            if int(getattr(self, total_name)) % int(getattr(self, chunk_name)):
                raise ValueError(f"{total_name} must be a multiple of {chunk_name}")
        for name in (
            "warmup_rhat_max",
            "retained_rhat_max",
            "bulk_ess_min",
            "tail_ess_min",
            "delta_h_abs_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.warmup_rhat_max <= 1.0 or self.retained_rhat_max <= 1.0:
            raise ValueError("R-hat thresholds must exceed one")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        object.__setattr__(self, "target_status_required", bool(self.target_status_required))

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            **asdict(self),
        }


@dataclass(frozen=True)
class SequentialNeuTraHMCResult:
    """Public-safe sequential run result; raw draws remain in private shards."""

    passed: bool
    stop_reason: str
    warmup_results_per_chain: int
    retained_results_per_chain: int
    diagnostics: Mapping[str, Any]
    archive: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def payload(self) -> Mapping[str, Any]:
        return {"schema": SEQUENTIAL_NEUTRA_HMC_SCHEMA, **asdict(self)}


def sequential_chunk_seed(
    root_seed: tuple[int, int], *, phase_index: int, chunk_index: int
) -> tuple[int, int]:
    """Return deterministic, phase-separated chunk seeds."""

    first, second = (int(item) for item in root_seed)
    return first, second + 1000003 * int(phase_index) + 1009 * (int(chunk_index) + 1)


def _tensor_tree_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _tensor_tree_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_tree_python(item) for item in value]
    if tf.is_tensor(value):
        return _tensor_tree_python(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    tensor = tf.convert_to_tensor(value)
    serialized = tf.io.serialize_tensor(tensor).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise NeuTraHMCError(f"archive shard already exists: {path}")
    path.write_bytes(serialized)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise NeuTraHMCError(f"artifact already exists: {path}")
    text = json.dumps(_tensor_tree_python(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _coordinate_diagnostics(
    samples: tf.Tensor, *, rhat_max: float
) -> Mapping[str, Any]:
    chain_major = tf.transpose(samples, (1, 0, 2))
    rhat = rank_normalized_split_rhat(chain_major)
    ess = rank_normalized_bulk_tail_ess(chain_major)
    maximum = tf.convert_to_tensor(rhat["maximum"], tf.float64)
    bulk = tf.convert_to_tensor(ess["bulk"], tf.float64)
    tail = tf.convert_to_tensor(ess["tail"], tf.float64)
    finite = tf.reduce_all(
        tf.math.is_finite(tf.concat((maximum, bulk, tail), axis=0))
    )
    return {
        "all_finite": bool(finite.numpy()),
        "max_rhat": float(tf.reduce_max(maximum).numpy()),
        "min_bulk_ess": float(tf.reduce_min(bulk).numpy()),
        "min_tail_ess": float(tf.reduce_min(tail).numpy()),
        "rhat_by_parameter": _tensor_tree_python(maximum),
        "bulk_ess_by_parameter": _tensor_tree_python(bulk),
        "tail_ess_by_parameter": _tensor_tree_python(tail),
        "rhat_threshold": float(rhat_max),
    }


def _mapped_model_samples(adapter: Any, samples: tf.Tensor) -> tf.Tensor:
    mapper = getattr(adapter, "latent_to_position", None)
    if not callable(mapper):
        raise NeuTraHMCError(
            "NeuTra admission requires a latent-to-model-parameter map"
        )
    if samples.shape.rank != 3 or any(dim is None for dim in samples.shape):
        raise NeuTraHMCError("samples must have static [draw, chain, parameter] shape")
    flat = tf.reshape(samples, (-1, int(samples.shape[-1])))
    mapped = tf.convert_to_tensor(mapper(flat), tf.float64)
    if mapped.shape != flat.shape:
        raise NeuTraHMCError("mapped model-parameter samples changed the sample shape")
    return tf.reshape(mapped, tf.shape(samples))


def _diagnostics(
    adapter: Any, samples: tf.Tensor, *, rhat_max: float
) -> Mapping[str, Any]:
    latent = _coordinate_diagnostics(samples, rhat_max=rhat_max)
    model = _coordinate_diagnostics(
        _mapped_model_samples(adapter, samples), rhat_max=rhat_max
    )
    return {
        "hmc_coordinates": latent,
        "model_parameters": model,
        "all_finite": bool(latent["all_finite"] and model["all_finite"]),
        "max_rhat": max(float(latent["max_rhat"]), float(model["max_rhat"])),
        "min_bulk_ess": min(
            float(latent["min_bulk_ess"]), float(model["min_bulk_ess"])
        ),
        "min_tail_ess": min(
            float(latent["min_tail_ess"]), float(model["min_tail_ess"])
        ),
        "rhat_threshold": float(rhat_max),
    }


def _target_status(adapter: Any, samples: tf.Tensor) -> Mapping[str, Any]:
    status_method = getattr(adapter, "log_prob_and_grad_status", None)
    if not callable(status_method):
        raise NeuTraHMCError("adapter lacks combined value/score/status telemetry")
    if samples.shape.rank != 3 or any(dim is None for dim in samples.shape):
        raise NeuTraHMCError(
            "target status samples must have static [draw, chain, parameter] shape"
        )
    # Compile/evaluate small fixed batches.  The q=20 status target maps over
    # a static leading dimension; flattening a full 500-draw chunk would build
    # an unnecessarily large XLA graph, while one call per draw adds avoidable
    # host overhead.  Four transitions per chain is the reviewed compromise.
    draw_count = int(samples.shape[0])
    chain_count = int(samples.shape[1])
    flat = tf.reshape(samples, (-1, int(samples.shape[-1])))
    audit_rows = chain_count * min(4, draw_count)
    audit_batches = [
        flat[start : start + audit_rows]
        for start in range(0, int(flat.shape[0]), audit_rows)
    ]
    values_by_batch: list[tf.Tensor] = []
    scores_by_batch: list[tf.Tensor] = []
    status_by_batch: dict[str, list[tf.Tensor]] = {}
    for audit_batch in audit_batches:
        values, scores, status = status_method(audit_batch)
        if not isinstance(status, Mapping):
            raise NeuTraHMCError("target status telemetry schema mismatch")
        values_by_batch.append(tf.convert_to_tensor(values, tf.float64))
        scores_by_batch.append(tf.convert_to_tensor(scores, tf.float64))
        for key, value in status.items():
            status_by_batch.setdefault(str(key), []).append(tf.convert_to_tensor(value))
    required = (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    )
    if any(key not in status_by_batch for key in required):
        raise NeuTraHMCError("target status telemetry schema mismatch")
    combined_status = {
        key: tf.concat(values, axis=0) for key, values in status_by_batch.items()
    }
    combined_values = tf.concat(values_by_batch, axis=0)
    combined_scores = tf.concat(scores_by_batch, axis=0)
    code = tf.convert_to_tensor(combined_status["status_code"], tf.int32)
    valid = tf.convert_to_tensor(
        combined_status["valid_pre_regularized_score"], tf.bool
    )
    floors = tf.convert_to_tensor(combined_status["floor_count_value"], tf.int32)
    min_eigen = tf.convert_to_tensor(
        combined_status["min_innovation_eigenvalue"], tf.float64
    )
    condition = tf.convert_to_tensor(
        combined_status["innovation_condition_estimate"], tf.float64
    )
    target_finite = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(combined_values)),
        tf.reduce_all(tf.math.is_finite(combined_scores)),
    )
    status_valid = tf.reduce_all(tf.logical_and(tf.equal(code, 0), valid))
    diagnostics_finite = tf.reduce_all(
        tf.math.is_finite(tf.concat((min_eigen, condition), axis=0))
    )
    passed = tf.logical_and(target_finite, tf.logical_and(status_valid, diagnostics_finite))
    return {
        "passed": bool(passed.numpy()),
        "target_value_score_all_finite": bool(target_finite.numpy()),
        "status_nonvalid_count": int(
            tf.reduce_sum(tf.cast(tf.logical_or(tf.not_equal(code, 0), ~valid), tf.int32)).numpy()
        ),
        "floor_count_total": int(tf.reduce_sum(floors).numpy()),
        "minimum_innovation_eigenvalue": float(tf.reduce_min(min_eigen).numpy()),
        "maximum_innovation_condition_estimate": float(tf.reduce_max(condition).numpy()),
        "audit_batch_size": audit_rows,
        "audit_batch_count": len(audit_batches),
        "audited_state_count": int(samples.shape[0]) * int(samples.shape[1]),
    }


class _ChunkRunner:
    def __init__(self, adapter: Any, state: tf.Tensor, config: SequentialNeuTraHMCConfig):
        self.adapter = adapter
        self.config = config
        self.state_shape = tuple(int(dim) for dim in state.shape)
        self.dtype = state.dtype
        target = reviewed_value_score_target_fn(adapter, dtype=self.dtype, require_batched=True)

        def run(current_state: tf.Tensor, seed: tf.Tensor):
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target,
                step_size=tf.constant(config.step_size, self.dtype),
                num_leapfrog_steps=config.num_leapfrog_steps,
            )

            def trace_fn(_state: Any, results: Any) -> Mapping[str, tf.Tensor]:
                return {
                    "is_accepted": results.is_accepted,
                    "log_accept_ratio": results.log_accept_ratio,
                    "target_log_prob": results.accepted_results.target_log_prob,
                    "proposed_target_log_prob": results.proposed_results.target_log_prob,
                    # For identity-mass TFP HMC, log(alpha ratio) = -Delta H.
                    "delta_h": -results.log_accept_ratio,
                }

            return tfp.mcmc.sample_chain(
                num_results=config.warmup_chunk_size,
                num_burnin_steps=0,
                current_state=current_state,
                kernel=kernel,
                trace_fn=trace_fn,
                seed=seed,
            )

        self.run = tf.function(
            run,
            input_signature=(
                tf.TensorSpec(self.state_shape, self.dtype),
                tf.TensorSpec((2,), tf.int32),
            ),
            jit_compile=config.use_xla,
            reduce_retracing=True,
        )


def _chain_moved(pre_chunk_state: tf.Tensor, samples: tf.Tensor) -> tf.Tensor:
    sequence = tf.concat((pre_chunk_state[None, ...], samples), axis=0)
    return tf.reduce_any(tf.not_equal(sequence[1:], sequence[:-1]), axis=(0, 2))


def run_sequential_neutra_hmc(
    adapter: Any,
    initial_state: Any,
    config: SequentialNeuTraHMCConfig,
    *,
    archive_root: str | Path,
    archive_label: str,
    budget_check: Callable[[int], bool | None] | None = None,
) -> SequentialNeuTraHMCResult:
    """Run fixed-kernel sequential warm-up and retained HMC."""

    state = tf.convert_to_tensor(initial_state, tf.float64)
    if state.shape.rank != 2 or int(state.shape[0]) != config.chain_count:
        raise ValueError("initial_state must have shape [chain, parameter]")
    if any(dim is None for dim in state.shape):
        raise ValueError("initial_state must have a fully static shape")
    if int(config.retained_chunk_size) != int(config.warmup_chunk_size):
        raise ValueError("current compiled controller requires equal chunk sizes")
    root = Path(archive_root)
    if root.exists() and any(root.iterdir()):
        raise NeuTraHMCError("archive_root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    runner = _ChunkRunner(adapter, state, config)
    started = time.perf_counter()
    phase_rows: dict[str, list[Mapping[str, Any]]] = {"warmup": [], "retained": []}
    phase_samples: dict[str, list[tf.Tensor]] = {"warmup": [], "retained": []}
    hard_vetoes: list[str] = []
    warmup_ready = False
    retained_passed = False
    warmup_count = 0
    retained_count = 0
    last_warmup_diagnostics: Mapping[str, Any] = {}
    last_retained_diagnostics: Mapping[str, Any] = {}

    for phase_index, phase in enumerate(("warmup", "retained")):
        maximum = (
            config.warmup_max_results if phase == "warmup" else config.retained_max_results
        )
        chunk_size = (
            config.warmup_chunk_size if phase == "warmup" else config.retained_chunk_size
        )
        chunk_count = maximum // chunk_size
        for chunk_index in range(chunk_count):
            if budget_check is not None:
                budget_allowed = budget_check(
                    chunk_size * config.num_leapfrog_steps
                )
                if budget_allowed is False:
                    hard_vetoes.append("campaign_resource_cap")
                    break
            seed = sequential_chunk_seed(
                config.seed, phase_index=phase_index, chunk_index=chunk_index
            )
            pre_chunk_state = state
            chunk_started = time.perf_counter()
            samples, trace = runner.run(state, tf.constant(seed, tf.int32))
            chunk_seconds = time.perf_counter() - chunk_started
            samples = tf.convert_to_tensor(samples, tf.float64)
            state = samples[-1]
            trace = {key: tf.convert_to_tensor(value) for key, value in trace.items()}
            moved = _chain_moved(pre_chunk_state, samples)
            samples_finite = tf.reduce_all(tf.math.is_finite(samples))
            log_accept_finite = tf.reduce_all(
                tf.math.is_finite(tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64))
            )
            target_finite = tf.reduce_all(
                tf.math.is_finite(tf.convert_to_tensor(trace["target_log_prob"], tf.float64))
            )
            proposed_finite = tf.reduce_all(
                tf.math.is_finite(
                    tf.convert_to_tensor(trace["proposed_target_log_prob"], tf.float64)
                )
            )
            delta_h = tf.convert_to_tensor(trace["delta_h"], tf.float64)
            delta_h_finite = tf.reduce_all(tf.math.is_finite(delta_h))
            delta_h_within_limit = tf.reduce_all(
                tf.abs(delta_h) <= tf.constant(config.delta_h_abs_max, tf.float64)
            )
            status = _target_status(adapter, samples) if config.target_status_required else {"passed": True}
            chunk_hard = []
            if not bool(samples_finite.numpy()):
                chunk_hard.append("nonfinite_state")
            if not bool(log_accept_finite.numpy()):
                chunk_hard.append("nonfinite_log_accept_ratio")
            if not bool(target_finite.numpy()) or not bool(proposed_finite.numpy()):
                chunk_hard.append("nonfinite_target_log_prob")
            if not bool(delta_h_finite.numpy()):
                chunk_hard.append("nonfinite_delta_h")
            elif not bool(delta_h_within_limit.numpy()):
                chunk_hard.append("absolute_delta_h_above_hard_limit")
            if not bool(tf.reduce_all(moved).numpy()):
                chunk_hard.append("unmoved_chain")
            if not bool(status["passed"]):
                chunk_hard.append("target_status_veto")
            checkpoint_diagnostics: Mapping[str, Any] = {}
            if not chunk_hard:
                phase_samples[phase].append(samples)
                if phase == "warmup":
                    warmup_count += chunk_size
                    if warmup_count >= config.warmup_min_results:
                        combined = tf.concat(phase_samples[phase], axis=0)
                        window = combined[-config.warmup_window_results :]
                        last_warmup_diagnostics = _diagnostics(
                            adapter, window, rhat_max=config.warmup_rhat_max
                        )
                        warmup_ready = bool(
                            last_warmup_diagnostics["all_finite"]
                            and last_warmup_diagnostics["max_rhat"]
                            <= config.warmup_rhat_max
                        )
                        checkpoint_diagnostics = last_warmup_diagnostics
                else:
                    retained_count += chunk_size
                    if retained_count >= config.retained_min_results:
                        combined = tf.concat(phase_samples[phase], axis=0)
                        last_retained_diagnostics = _diagnostics(
                            adapter, combined, rhat_max=config.retained_rhat_max
                        )
                        retained_passed = bool(
                            last_retained_diagnostics["all_finite"]
                            and last_retained_diagnostics["max_rhat"]
                            <= config.retained_rhat_max
                            and last_retained_diagnostics["min_bulk_ess"]
                            >= config.bulk_ess_min
                            and last_retained_diagnostics["min_tail_ess"]
                            >= config.tail_ess_min
                        )
                        checkpoint_diagnostics = last_retained_diagnostics
            prefix = f"{archive_label}-{phase}-{chunk_index:03d}"
            sample_receipt = _write_tensor(root / phase / f"{prefix}-samples.tftensor", samples)
            trace_receipts = {
                key: _write_tensor(root / phase / f"{prefix}-{key}.tftensor", value)
                for key, value in trace.items()
            }
            row = {
                "phase": phase,
                "chunk_index": chunk_index,
                "seed": list(seed),
                "sample_receipt": sample_receipt,
                "trace_receipts": trace_receipts,
                "chunk_seconds": chunk_seconds,
                "acceptance_probability_by_chain": _tensor_tree_python(
                    tf.reduce_mean(
                        tf.exp(tf.minimum(tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64), 0.0)),
                        axis=0,
                    )
                ),
                "chain_moved": _tensor_tree_python(moved),
                "target_status": status,
                "energy_error": {
                    "identity": "delta_h_equals_negative_log_accept_ratio",
                    "maximum_absolute_delta_h": float(
                        tf.reduce_max(tf.abs(delta_h)).numpy()
                    ),
                    "absolute_delta_h_hard_limit": config.delta_h_abs_max,
                    "all_finite": bool(delta_h_finite.numpy()),
                    "within_hard_limit": bool(delta_h_within_limit.numpy()),
                },
                "checkpoint_diagnostics": checkpoint_diagnostics,
                "hard_vetoes": chunk_hard,
                "native_divergence_status": "not_exposed_by_kernel",
            }
            _write_json(root / phase / f"{prefix}-receipt.json", row)
            phase_rows[phase].append(row)
            hard_vetoes.extend(chunk_hard)
            if chunk_hard:
                break
            if (phase == "warmup" and warmup_ready) or (
                phase == "retained" and retained_passed
            ):
                break
        if hard_vetoes or (phase == "warmup" and not warmup_ready):
            break

    stop_reason = (
        "hard_veto"
        if hard_vetoes
        else "warmup_cap_not_ready"
        if not warmup_ready
        else "retained_diagnostics_passed"
        if retained_passed
        else "retained_cap_not_passed"
    )
    manifest = {
        "schema": SEQUENTIAL_NEUTRA_HMC_SCHEMA,
        "policy": config.payload(),
        "archive_label": archive_label,
        "warmup_chunks": phase_rows["warmup"],
        "retained_chunks": phase_rows["retained"],
        "warmup_excluded_from_posterior": True,
        "native_divergence_status": "not_exposed_by_kernel",
        "nonclaims": [
            "native divergence unavailability is not zero divergences",
            "finite-sample operational screen only",
            "no posterior correctness or stationarity proof",
        ],
    }
    manifest_path = root / f"{archive_label}-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return SequentialNeuTraHMCResult(
        passed=retained_passed and not hard_vetoes,
        stop_reason=stop_reason,
        warmup_results_per_chain=warmup_count,
        retained_results_per_chain=retained_count,
        diagnostics={
            "warmup": last_warmup_diagnostics,
            "retained": last_retained_diagnostics,
            "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
            "acceptance_role": "explanatory_only",
            "native_divergence_status": "not_exposed_by_kernel",
        },
        archive={
            "root": root.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_hash,
            "warmup_chunk_count": len(phase_rows["warmup"]),
            "retained_chunk_count": len(phase_rows["retained"]),
        },
        metadata={
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "wall_seconds": time.perf_counter() - started,
            "use_xla": config.use_xla,
            "warmup_excluded_from_posterior": True,
        },
    )


__all__ = [
    "NEUTRA_SEQUENTIAL_HMC_POLICY_ID",
    "NeuTraHMCError",
    "SequentialNeuTraHMCConfig",
    "SequentialNeuTraHMCResult",
    "run_sequential_neutra_hmc",
    "sequential_chunk_seed",
]
