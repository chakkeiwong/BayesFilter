"""BayesFilter-owned staged fixed-kernel HMC estimation workflow.

This module owns the generic staged sampler-health policy.  Callers provide an
already-authorized value/score adapter, a four-chain start bank, and one frozen
step size/leapfrog pair.  Burn-in diagnostics are discarded; retained samples
are streamed to the existing private archive sink.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import tensorflow as tf

from bayesfilter.inference.hmc import (
    FixedSizeHMCChunkConfig,
    HMCStreamingSampleArchiveSink,
    build_fixed_size_hmc_chunk_runner,
)
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_bulk_tail_ess,
    rank_normalized_split_rhat,
)
from bayesfilter.inference.posterior_adapter import value_score_capability


ROUTE = "bayesfilter_staged_fixed_kernel_hmc_estimation_v1"
NONCLAIMS = (
    "fixed-kernel staged sampler-health and retained archive workflow only",
    "R-hat thresholds are finite-sample health gates, not convergence proofs",
    "no posterior validity, identification, sampler superiority, or scientific claim",
)


def _seed(value: Any, *, name: str) -> tuple[int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    return result


@dataclass(frozen=True)
class StagedFixedKernelHMCConfig:
    """Exact user-requested burn-in and retained-sample ladder."""

    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    burnin_block_size: int = 500
    burnin_max_transitions: int = 5000
    burnin_rhat_threshold: float = 1.05
    retained_initial_samples: int = 3000
    retained_extension_block_size: int = 1000
    retained_max_samples: int = 20000
    retained_rhat_threshold: float = 1.01
    chain_count: int = 4
    use_xla: bool = False
    target_scope: str | None = None
    archive_label: str = "dz5_staged_hmc_retained"

    def __post_init__(self) -> None:
        step = float(self.step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step)
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "seed", _seed(self.seed, name="seed"))
        for name in (
            "burnin_block_size",
            "burnin_max_transitions",
            "retained_initial_samples",
            "retained_extension_block_size",
            "retained_max_samples",
            "chain_count",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.chain_count != 4:
            raise ValueError("staged estimation requires exactly four chains")
        if self.burnin_max_transitions % self.burnin_block_size:
            raise ValueError("burnin_max_transitions must be divisible by burnin_block_size")
        if self.retained_initial_samples % self.retained_extension_block_size:
            raise ValueError("retained_initial_samples must be divisible by retained_extension_block_size")
        if self.retained_max_samples < self.retained_initial_samples:
            raise ValueError("retained_max_samples must cover retained_initial_samples")
        for name in ("burnin_rhat_threshold", "retained_rhat_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 1.0:
                raise ValueError(f"{name} must be finite and greater than one")
            object.__setattr__(self, name, value)
        if self.retained_rhat_threshold >= self.burnin_rhat_threshold:
            raise ValueError("retained R-hat threshold must be stricter than burn-in threshold")
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        if self.use_xla:
            raise ValueError("staged estimation requires explicit full-chain XLA authority; use_xla must be false")
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))
        if not str(self.archive_label).strip() or "/" in str(self.archive_label) or "\\" in str(self.archive_label):
            raise ValueError("archive_label must be a non-empty path-free label")

    def payload(self) -> Mapping[str, Any]:
        return {**asdict(self), "route": ROUTE, "retained_sampling_authorized": True}


@dataclass(frozen=True)
class StagedFixedKernelHMCResult:
    config: StagedFixedKernelHMCConfig
    burnin_blocks: tuple[Mapping[str, Any], ...]
    retained_blocks: tuple[Mapping[str, Any], ...]
    burnin_status: str
    retained_status: str
    final_status: str
    retained_sample_count: int
    final_state: Any
    archive_manifest: Mapping[str, Any] | None
    elapsed_seconds: float
    artifact_path: str | None = None

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.staged_fixed_kernel_hmc_estimation_result.v1",
            "route": ROUTE,
            "config": self.config.payload(),
            "burnin_blocks": self.burnin_blocks,
            "retained_blocks": self.retained_blocks,
            "burnin_status": self.burnin_status,
            "retained_status": self.retained_status,
            "final_status": self.final_status,
            "retained_sample_count": self.retained_sample_count,
            "archive_manifest": self.archive_manifest,
            "elapsed_seconds": self.elapsed_seconds,
            "artifact_path": self.artifact_path,
            "retained_sampling_authorized": self.final_status == "retained_rhat_passed",
            "nonclaims": NONCLAIMS,
        }


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json(value.numpy())
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if hasattr(value, "item"):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _diag(
    samples: list[tf.Tensor],
    threshold: float,
    *,
    diagnostic_transform: Callable[[Any], Any] | None,
    window: str,
) -> Mapping[str, Any]:
    if window not in {"latest", "cumulative"}:
        raise ValueError("diagnostic window must be latest or cumulative")
    active = samples[-1] if window == "latest" else tf.concat(samples, axis=0)
    chain_major = tf.transpose(active, perm=(1, 0, 2))
    if not bool(tf.reduce_all(tf.math.is_finite(chain_major)).numpy()):
        raise ValueError("nonfinite samples in staged HMC diagnostics")
    latent_raw = rank_normalized_split_rhat(chain_major)
    physical = chain_major if diagnostic_transform is None else tf.cast(
        tf.convert_to_tensor(diagnostic_transform(chain_major)), tf.float64
    )
    if tuple(physical.shape) != tuple(chain_major.shape):
        raise ValueError("diagnostic_transform must preserve [chain, draw, parameter] shape")
    if not bool(tf.reduce_all(tf.math.is_finite(physical)).numpy()):
        raise ValueError("diagnostic_transform returned nonfinite physical samples")
    raw = rank_normalized_split_rhat(physical)
    maximum = tf.maximum(tf.cast(raw["bulk"], tf.float64), tf.cast(raw["folded"], tf.float64))
    ess = rank_normalized_bulk_tail_ess(physical)
    latent_maximum = tf.maximum(
        tf.cast(latent_raw["bulk"], tf.float64),
        tf.cast(latent_raw["folded"], tf.float64),
    )
    max_rhat = float(tf.reduce_max(maximum).numpy())
    return {
        "draw_count_per_chain": int(chain_major.shape[1]),
        "diagnostic_coordinate": "physical_parameter",
        "diagnostic_window": window,
        "physical_rhat": _json(raw),
        "latent_rhat": _json(latent_raw),
        "latent_maximum_rhat": float(tf.reduce_max(latent_maximum).numpy()),
        "maximum_rhat": max_rhat,
        "rhat_threshold": float(threshold),
        "rhat_passed": bool(math.isfinite(max_rhat) and max_rhat < float(threshold)),
        "ess": _json(ess),
    }


def _validate_chunk(
    *,
    adapter: Any,
    run: Any,
    current: tf.Tensor,
    previous: tf.Tensor,
    expected_active: int,
    chain_count: int,
    parameter_dim: int,
) -> tf.Tensor:
    """Apply shared hard health checks and evaluate model-owned status telemetry."""

    mask = tf.cast(tf.convert_to_tensor(run.valid_mask), tf.bool)
    if tuple(mask.shape) != (int(run.samples.shape[0]),):
        raise ValueError("chunk valid-mask shape mismatch")
    if int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()) != expected_active:
        raise ValueError("chunk did not return the expected active transition count")
    samples = tf.convert_to_tensor(run.samples, tf.float64)
    valid = tf.boolean_mask(samples, mask)
    if tuple(valid.shape) != (expected_active, chain_count, parameter_dim):
        raise ValueError("chunk sample layout mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(valid)).numpy()):
        raise ValueError("nonfinite chunk samples")
    if not bool(tf.reduce_all(tf.equal(tf.convert_to_tensor(run.final_state, tf.float64), valid[-1])).numpy()):
        raise ValueError("chunk final_state does not match last valid sample")
    displacement = tf.linalg.norm(valid - previous[tf.newaxis, :, :], axis=-1)
    moved = tf.reduce_any(displacement > 0.0, axis=0)
    if not bool(tf.reduce_all(moved).numpy()):
        raise ValueError("at least one chain did not move in the chunk")
    diagnostics = run.diagnostics
    for name in ("acceptance_rate", "log_accept_ratio_max_abs_finite"):
        value = diagnostics.get(name)
        if value is not None and not bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy()):
            raise ValueError(f"nonfinite chunk diagnostic: {name}")
    divergence = diagnostics.get("divergence_count")
    if divergence is not None and int(tf.reduce_sum(tf.cast(divergence, tf.int32)).numpy()) > 0:
        raise ValueError("positive native divergence count")
    trace = run.trace
    for name in ("log_accept_ratio", "target_log_prob"):
        values = tf.boolean_mask(tf.cast(tf.convert_to_tensor(trace[name]), tf.float64), mask)
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(f"nonfinite chunk trace: {name}")
    telemetry_fn = getattr(adapter, "target_status_telemetry", None)
    if not callable(telemetry_fn):
        raise ValueError("adapter must expose target_status_telemetry")
    status = telemetry_fn(current)
    code = tf.cast(tf.convert_to_tensor(status["status_code"]), tf.int32)
    valid_score = tf.cast(tf.convert_to_tensor(status["valid_pre_regularized_score"]), tf.bool)
    if bool(tf.reduce_any(tf.not_equal(code, 0)).numpy()) or not bool(tf.reduce_all(valid_score).numpy()):
        raise ValueError("target status telemetry reported an invalid row")
    return valid


def run_staged_fixed_kernel_hmc_estimation(
    *,
    adapter: Any,
    initial_state: Any,
    config: StagedFixedKernelHMCConfig,
    archive_dir: str | Path,
    output_path: str | Path | None = None,
    diagnostic_transform: Callable[[Any], Any] | None = None,
    progress_path: str | Path | None = None,
) -> StagedFixedKernelHMCResult:
    """Execute the exact staged policy with fixed kernel and four-chain handoff."""

    if not isinstance(config, StagedFixedKernelHMCConfig):
        raise TypeError("config must be StagedFixedKernelHMCConfig")
    capability = value_score_capability(adapter)
    if not capability.is_accepted_xla_hmc_authority:
        raise ValueError("staged estimation requires accepted target value/score authority")
    if config.target_scope is not None and capability.target_scope != config.target_scope:
        raise ValueError("adapter and config target_scope mismatch")
    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    if state.shape.rank != 2 or tuple(state.shape) != (config.chain_count, int(getattr(adapter, "parameter_dim"))):
        raise ValueError("initial_state must have shape [4, parameter_dim]")
    if not bool(tf.reduce_all(tf.math.is_finite(state)).numpy()):
        raise ValueError("initial_state must be finite")
    started = time.perf_counter()
    progress_file = None if progress_path is None else Path(progress_path)
    progress = {
        "schema": "bayesfilter.staged_fixed_kernel_hmc_progress.v1",
        "route": ROUTE,
        "config": config.payload(),
        "burnin_blocks": [],
        "retained_blocks": [],
        "burnin_status": "not_started",
        "retained_status": "not_started",
        "retained_count": 0,
        "last_state_hash": _hash(state),
    }

    def write_progress() -> None:
        if progress_file is None:
            return
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_file.with_name(progress_file.name + ".tmp")
        temporary.write_text(json.dumps(_json(progress), indent=2, sort_keys=True) + "\n")
        temporary.replace(progress_file)

    write_progress()
    runner_config = FixedSizeHMCChunkConfig(
        max_results=max(config.burnin_block_size, config.retained_extension_block_size),
        num_burnin_steps=0,
        step_size=config.step_size,
        num_leapfrog_steps=config.num_leapfrog_steps,
        seed=config.seed,
        use_xla=False,
        trace_policy="standard",
        target_scope=config.target_scope,
        chain_execution_mode="tf_function",
    )
    runner = build_fixed_size_hmc_chunk_runner(adapter, state, runner_config)
    burnin_blocks: list[Mapping[str, Any]] = []
    burnin_samples: list[tf.Tensor] = []
    burnin_status = "burnin_cap_exhausted"
    current = state
    for block_index in range(config.burnin_max_transitions // config.burnin_block_size):
        run = runner.run(
            active_results=config.burnin_block_size,
            current_state=current,
            seed=(config.seed[0], config.seed[1] + 1009 * block_index),
        )
        previous = current
        current = tf.convert_to_tensor(run.final_state, tf.float64)
        valid = _validate_chunk(
            adapter=adapter,
            run=run,
            current=current,
            previous=previous,
            expected_active=config.burnin_block_size,
            chain_count=config.chain_count,
            parameter_dim=int(getattr(adapter, "parameter_dim")),
        )
        burnin_samples.append(valid)
        diag = dict(_diag(
            burnin_samples,
            config.burnin_rhat_threshold,
            diagnostic_transform=diagnostic_transform,
            window="latest",
        ))
        diag.update({
            "block_index": block_index,
            "cumulative_transitions_per_chain": (block_index + 1) * config.burnin_block_size,
            "acceptance_rate": _json(run.diagnostics.get("acceptance_rate")),
            "acceptance_rate_by_chain": _json(run.diagnostics.get("acceptance_rate_by_chain")),
            "divergence_count": _json(run.diagnostics.get("divergence_count")),
            "final_state_hash": _hash(current),
        })
        burnin_blocks.append(diag)
        progress["burnin_blocks"] = _json(burnin_blocks)
        progress["last_state_hash"] = _hash(current)
        progress["burnin_status"] = "burnin_rhat_passed" if diag["rhat_passed"] else "burnin_in_progress"
        write_progress()
        if diag["rhat_passed"]:
            burnin_status = "burnin_rhat_passed"
            break
    if burnin_status != "burnin_rhat_passed":
        result = StagedFixedKernelHMCResult(config, tuple(burnin_blocks), (), burnin_status, "not_started", "burnin_cap_exhausted", 0, current, None, time.perf_counter() - started)
        if output_path is not None:
            path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(_json(result.payload()), indent=2, sort_keys=True) + "\n")
        return result

    sink_config = FixedSizeHMCChunkConfig(
        max_results=config.retained_extension_block_size,
        num_burnin_steps=0,
        step_size=config.step_size,
        num_leapfrog_steps=config.num_leapfrog_steps,
        seed=config.seed,
        use_xla=False,
        trace_policy="standard",
        target_scope=config.target_scope,
        chain_execution_mode="tf_function",
    )
    sink = HMCStreamingSampleArchiveSink(
        archive_dir=archive_dir,
        config=sink_config,
        archive_label=config.archive_label,
        metadata={"route": ROUTE, "target_scope": config.target_scope, "nonclaims": NONCLAIMS},
    )
    retained_blocks: list[Mapping[str, Any]] = []
    retained_samples: list[tf.Tensor] = []
    retained_count = 0
    retained_status = "retained_rhat_cap_exhausted"
    block_index = 0
    while retained_count < config.retained_max_samples:
        run = runner.run(
            active_results=config.retained_extension_block_size,
            current_state=current,
            seed=(config.seed[0], config.seed[1] + 100000 + 1009 * block_index),
        )
        previous = current
        current = tf.convert_to_tensor(run.final_state, tf.float64)
        valid = _validate_chunk(
            adapter=adapter,
            run=run,
            current=current,
            previous=previous,
            expected_active=config.retained_extension_block_size,
            chain_count=config.chain_count,
            parameter_dim=int(getattr(adapter, "parameter_dim")),
        )
        sink.write_chunk(run, chunk_index=block_index)
        retained_samples.append(valid)
        retained_count += config.retained_extension_block_size
        should_check = retained_count >= config.retained_initial_samples
        progress["retained_count"] = retained_count
        progress["last_state_hash"] = _hash(current)
        if should_check:
            diag = dict(_diag(
                retained_samples,
                config.retained_rhat_threshold,
                diagnostic_transform=diagnostic_transform,
                window="cumulative",
            ))
            diag.update({
                "block_index": block_index,
                "cumulative_retained_samples_per_chain": retained_count,
                "acceptance_rate": _json(run.diagnostics.get("acceptance_rate")),
                "acceptance_rate_by_chain": _json(run.diagnostics.get("acceptance_rate_by_chain")),
                "divergence_count": _json(run.diagnostics.get("divergence_count")),
                "final_state_hash": _hash(current),
            })
            retained_blocks.append(diag)
            progress["retained_blocks"] = _json(retained_blocks)
            progress["retained_status"] = "retained_rhat_passed" if diag["rhat_passed"] else "retained_in_progress"
            if diag["rhat_passed"]:
                retained_status = "retained_rhat_passed"
        write_progress()
        if retained_status == "retained_rhat_passed":
            break
        block_index += 1
    manifest = sink.finalize()
    final_status = "retained_rhat_passed" if retained_status == "retained_rhat_passed" else "retained_rhat_cap_exhausted"
    result = StagedFixedKernelHMCResult(config, tuple(burnin_blocks), tuple(retained_blocks), burnin_status, retained_status, final_status, retained_count, current, manifest.public_payload(), time.perf_counter() - started)
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(_json(result.payload()), indent=2, sort_keys=True) + "\n")
    return result


__all__ = [
    "ROUTE",
    "NONCLAIMS",
    "StagedFixedKernelHMCConfig",
    "StagedFixedKernelHMCResult",
    "run_staged_fixed_kernel_hmc_estimation",
]
