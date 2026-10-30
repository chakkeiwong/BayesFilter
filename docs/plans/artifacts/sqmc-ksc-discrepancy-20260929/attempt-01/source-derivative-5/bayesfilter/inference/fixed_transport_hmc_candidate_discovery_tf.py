"""Candidate-only fixed-transport HMC tuning in TensorFlow/TFP.

This module discovers an unranked set of fixed-HMC ``(L, epsilon)`` pairs.  It
does not refine the trajectory grid, select a representative, confirm a kernel,
or authorize retained sampling.  The numerical transition primitive is shared
with the legacy fixed-transport kernel-admission tuner while the orchestration
and result contracts remain deliberately separate.
"""

from __future__ import annotations

import json
import math
import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import tensorflow as tf

import bayesfilter.inference.fixed_transport_hmc_mechanics_tf as _mechanics
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_bulk_tail_ess,
    rank_normalized_split_rhat,
)
from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    FixedTransportFullChainConfig as _FullChainHMCConfig,
    FixedTransportHMCPolicy as _TuningPolicy,
    RunFullChainFn,
    build_fixed_transport_value_score_adapter,
    build_fixed_transport_reusable_runner as _build_reusable_runner,
    fixed_transport_base_adapter_signature as _base_adapter_signature,
    fixed_transport_json_ready as _json_ready,
    fixed_transport_stable_hash as _stable_hash,
    fixed_transport_terminal_step_size as _terminal_step_size,
    fixed_transport_target_status_diagnostics as _target_status_diagnostics,
    fixed_transport_tensor_diagnostics as _tensor_diagnostics,
    offset_fixed_transport_seed as _offset_seed,
    run_fixed_transport_full_chain_tfp_hmc as _run_full_chain_tfp_hmc,
)
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.inference.neutra_hmc import (
    _ArchivedSequentialNeuTraHMCConfig,
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    SequentialNeuTraHMCXLAQualificationReceipt,
    validate_sequential_neutra_hmc_xla_receipt,
)


FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE = (
    "bayesfilter_fixed_transport_hmc_candidate_discovery_v1"
)
FIXED_TRANSPORT_CANDIDATE_CAMPAIGN_ROUTE = (
    "bayesfilter_fixed_transport_hmc_candidate_campaign_v2"
)
PRIMARY_L_GRID = (3, 5, 9, 13, 18, 25)
GAUSSIAN_EPSILON_WARM_START_ROUTE = (
    "bayesfilter_standard_normal_harmonic_epsilon_warm_start_v1"
)
CANDIDATE_DISCOVERY_NONCLAIMS = (
    "discarded fixed-transport HMC candidate-discovery draws only",
    "no representative selection",
    "no trajectory refinement",
    "no exact-L retuning",
    "no frozen-kernel confirmation",
    "no convergence or retained-sampling claim",
    "no sampler ranking, superiority, or default-readiness claim",
)

CANDIDATE_REFINEMENT_ROUTE = "bayesfilter_fixed_transport_hmc_candidate_refinement_v1"
CANDIDATE_REFINEMENT_NONCLAIMS = (
    "discard-only fixed-kernel refinement diagnostics",
    "no stationarity or posterior convergence proof",
    "no retained posterior samples",
    "no sampler superiority or default-readiness claim",
)


class _RefinementCheckpointError(RuntimeError):
    """Raised when resumable candidate evidence is missing or inconsistent."""


def _refinement_tensor_receipt(path: Path, value: Any) -> Mapping[str, Any]:
    tensor = tf.convert_to_tensor(value)
    encoded = tf.io.serialize_tensor(tensor).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise _RefinementCheckpointError(
            f"candidate refinement checkpoint tensor exists: {path}"
        )
    path.write_bytes(encoded)
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "shape": [int(item) for item in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def _read_refinement_tensor_receipt(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise _RefinementCheckpointError(
            f"candidate refinement checkpoint tensor is missing: {path}"
        ) from error
    if len(encoded) != int(receipt["bytes"]):
        raise _RefinementCheckpointError(
            "candidate refinement checkpoint tensor byte count mismatch"
        )
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise _RefinementCheckpointError(
            "candidate refinement checkpoint tensor hash mismatch"
        )
    tensor = tf.io.parse_tensor(
        encoded, out_type=tf.dtypes.as_dtype(str(receipt["dtype"]))
    )
    if tuple(tensor.shape) != tuple(int(item) for item in receipt["shape"]):
        raise _RefinementCheckpointError(
            "candidate refinement checkpoint tensor shape mismatch"
        )
    return tensor


def _write_refinement_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _positive_int(value: Any, *, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _band(value: Sequence[float]) -> tuple[float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 2 or not 0.0 < result[0] <= result[1] < 1.0:
        raise ValueError("nomination_band must satisfy 0 < low <= high < 1")
    return result


def _seed(value: Sequence[int], *, name: str) -> tuple[int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    return result


def gaussian_harmonic_epsilon_warm_starts(
    leapfrog_grid: Sequence[int] = PRIMARY_L_GRID,
    *,
    dimension: int = 1,
    target_accept_prob: float = 0.70,
    epsilon_min: float = 1.0e-3,
    epsilon_max: float = 1.95,
    epsilon_grid_size: int = 257,
    draw_count: int = 4096,
    seed: tuple[int, int] = (20260805, 7701),
) -> Mapping[int, Mapping[str, float | int | str | tuple[int, int]]]:
    """Return deterministic Gaussian fixed-``L`` epsilon warm starts.

    The pilot is a standard-normal harmonic oscillator with identity mass.  It
    calibrates the *initial* epsilon to the requested mean Metropolis
    acceptance, using common stateless Gaussian position/momentum draws for
    every ``L``.  It is a warm-start hypothesis only: the DZ5 target's own
    dual-averaging adaptation and fresh screens remain authoritative.
    """
    grid = tuple(int(item) for item in leapfrog_grid)
    if not grid or any(item <= 0 for item in grid):
        raise ValueError("leapfrog_grid must contain positive integers")
    target = float(target_accept_prob)
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target_accept_prob must be finite and in (0, 1)")
    lo, hi = float(epsilon_min), float(epsilon_max)
    if not math.isfinite(lo) or not math.isfinite(hi) or not 0.0 < lo < hi < 2.0:
        raise ValueError("epsilon bounds must satisfy 0 < min < max < 2")
    count = int(epsilon_grid_size)
    draws = int(draw_count)
    dim = int(dimension)
    if count < 3 or draws <= 0 or dim <= 0:
        raise ValueError("epsilon_grid_size >= 3, draw_count, and dimension must be positive")
    root = _seed(seed, name="seed")
    # Stateless NumPy draws keep the pilot independent of TensorFlow runtime
    # and make the initializer reproducible in CPU-hidden tests and launchers.
    import numpy as np

    generator = np.random.default_rng(
        (int(root[0]) << 32) ^ (int(root[1]) & 0xFFFFFFFF)
    )
    position = generator.normal(size=(draws, dim))
    momentum = generator.normal(size=(draws, dim))
    epsilons = np.linspace(lo, hi, count, dtype=float)
    result: dict[int, Mapping[str, float | int | str | tuple[int, int]]] = {}
    for leapfrog in grid:
        acceptance = np.empty(count, dtype=float)
        for index, epsilon in enumerate(epsilons):
            q = position.copy()
            p = momentum.copy()
            p -= 0.5 * epsilon * q
            for step in range(leapfrog):
                q += epsilon * p
                if step != leapfrog - 1:
                    p -= epsilon * q
            p -= 0.5 * epsilon * q
            delta_h = 0.5 * (q * q + p * p - position * position - momentum * momentum)
            acceptance[index] = float(
                np.minimum(1.0, np.exp(np.minimum(-delta_h.sum(axis=1), 0.0))).mean()
            )
        # Fixed-L Gaussian acceptance is oscillatory at large epsilon.  Use the
        # first stable-branch crossing from acceptance near one, not a later
        # resonance that happens to be numerically closer to the target.
        crossings = np.flatnonzero(acceptance <= target)
        if crossings.size:
            right = int(crossings[0])
            left = max(0, right - 1)
            selected = min((left, right), key=lambda index: abs(acceptance[index] - target))
            crossing_bracket = (float(epsilons[left]), float(epsilons[right]))
        else:
            selected = int(np.argmin(np.abs(acceptance - target)))
            crossing_bracket = None
        result[leapfrog] = {
            "epsilon": float(epsilons[selected]),
            "pilot_acceptance": float(acceptance[selected]),
            "target_accept_prob": target,
            "epsilon_grid_min": lo,
            "epsilon_grid_max": hi,
            "epsilon_grid_size": count,
            "draw_count": draws,
            "dimension": dim,
            "seed": root,
            "route": GAUSSIAN_EPSILON_WARM_START_ROUTE,
            "diagnostic_role": "warm_start_only",
            "selection_rule": "first_stable_branch_target_acceptance_crossing",
            "crossing_bracket": crossing_bracket,
        }
    return result


@dataclass(frozen=True)
class FixedTransportHMCXLAQualificationReceipt:
    """Route-bound evidence that the exact full-chain XLA shape was qualified."""

    base_adapter_signature: str
    fixed_transport_manifest_hash: str
    transformed_adapter_signature: str
    initial_state_shape: tuple[int, int]
    target_scope: str
    evidence_path: str
    evidence_sha256: str
    qualification_code_hash: str
    status: str = "passed"
    route: str = FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE

    def __post_init__(self) -> None:
        for name in (
            "base_adapter_signature",
            "fixed_transport_manifest_hash",
            "transformed_adapter_signature",
            "target_scope",
            "evidence_path",
            "evidence_sha256",
            "qualification_code_hash",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty")
        shape = tuple(int(item) for item in self.initial_state_shape)
        if len(shape) != 2 or any(item <= 0 for item in shape):
            raise ValueError("initial_state_shape must be positive [chain, parameter]")
        object.__setattr__(self, "initial_state_shape", shape)
        if self.status != "passed":
            raise ValueError("XLA qualification receipt must have status='passed'")
        if self.route != FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE:
            raise ValueError("XLA qualification receipt route mismatch")

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_hmc_xla_qualification.v2",
            **asdict(self),
        }

    @property
    def receipt_hash(self) -> str:
        return _stable_hash(self.payload())


@dataclass(frozen=True)
class FixedTransportHMCXLAQualificationConfig:
    """Explicit bounded probe settings for the shared discovery route."""

    initial_step_size: float
    adaptation_steps: int
    tune_num_results: int
    screen_num_results: int
    screen_num_burnin_steps: int
    seed: tuple[int, int]
    value_atol: float
    value_rtol: float
    score_atol: float
    score_rtol: float
    transition_atol: float
    transition_rtol: float
    primary_l_grid: tuple[int, ...] = PRIMARY_L_GRID
    target_accept_prob: float = 0.70
    target_status_trace_policy: str = "per_chain_step"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_step_size",
            _positive_float(self.initial_step_size, name="initial_step_size"),
        )
        for name in (
            "adaptation_steps",
            "tune_num_results",
            "screen_num_results",
            "screen_num_burnin_steps",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        grid = tuple(int(item) for item in self.primary_l_grid)
        if grid != PRIMARY_L_GRID:
            raise ValueError(f"primary_l_grid must equal {PRIMARY_L_GRID}")
        object.__setattr__(self, "primary_l_grid", grid)
        object.__setattr__(self, "seed", _seed(self.seed, name="seed"))
        for name in (
            "value_atol",
            "value_rtol",
            "score_atol",
            "score_rtol",
            "transition_atol",
            "transition_rtol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
            object.__setattr__(self, name, value)
        if float(self.target_accept_prob) != 0.70:
            raise ValueError("qualification target_accept_prob must equal 0.70")
        object.__setattr__(self, "target_accept_prob", 0.70)
        if self.target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("target_status_trace_policy is invalid")

    def payload(self) -> Mapping[str, Any]:
        return asdict(self)


def _comparison(
    eager: Any,
    compiled: Any,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> Mapping[str, Any]:
    left = tf.convert_to_tensor(eager)
    right = tf.convert_to_tensor(compiled)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"{label} shape or dtype mismatch")
    if left.dtype == tf.bool or left.dtype.is_integer:
        equal = bool(tf.reduce_all(tf.equal(left, right)).numpy())
        if not equal:
            raise ValueError(f"{label} exact comparison failed")
        return {"label": label, "passed": True, "comparison": "exact"}
    left = tf.cast(left, tf.float64)
    right = tf.cast(right, tf.float64)
    if not bool(
        tf.reduce_all(tf.math.is_finite(left)).numpy()
        and tf.reduce_all(tf.math.is_finite(right)).numpy()
    ):
        raise ValueError(f"{label} contains nonfinite values")
    residual = tf.abs(left - right)
    tolerance = tf.cast(atol, tf.float64) + tf.cast(rtol, tf.float64) * tf.abs(left)
    passed = bool(tf.reduce_all(residual <= tolerance).numpy())
    maximum = float(tf.reduce_max(residual).numpy()) if int(tf.size(residual)) else 0.0
    if not passed:
        raise ValueError(f"{label} XLA comparison failed")
    return {
        "label": label,
        "passed": True,
        "comparison": "absolute_plus_relative",
        "atol": atol,
        "rtol": rtol,
        "maximum_absolute_residual": maximum,
    }


def _mapping_comparisons(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
    prefix: str,
) -> tuple[Mapping[str, Any], ...]:
    if set(eager) != set(compiled):
        raise ValueError(f"{prefix} trace schema mismatch")
    rows: list[Mapping[str, Any]] = []
    for key in sorted(eager):
        left = eager[key]
        right = compiled[key]
        label = f"{prefix}.{key}"
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                raise ValueError(f"{label} nested trace mismatch")
            rows.extend(
                _mapping_comparisons(
                    left,
                    right,
                    atol=atol,
                    rtol=rtol,
                    prefix=label,
                )
            )
        else:
            rows.append(
                _comparison(left, right, atol=atol, rtol=rtol, label=label)
            )
    return tuple(rows)


def _explanatory_route_comparisons(
    control: Any,
    compiled: Any,
    *,
    atol: float,
    rtol: float,
    prefix: str,
) -> tuple[Mapping[str, Any], ...]:
    """Compare stochastic routes for telemetry without imposing replay equality."""
    def explain_leaf(left: Any, right: Any, label: str) -> Mapping[str, Any]:
        try:
            row = dict(_comparison(left, right, atol=atol, rtol=rtol, label=label))
            row.update(
                diagnostic_role="explanatory_stochastic_route_telemetry",
                hard_veto=False,
            )
            return row
        except Exception as error:  # noqa: BLE001 - route telemetry is non-vetoing.
            left_tensor = tf.convert_to_tensor(left)
            right_tensor = tf.convert_to_tensor(right)
            residual = None
            if left_tensor.shape == right_tensor.shape:
                try:
                    residual = float(
                        tf.reduce_max(
                            tf.abs(
                                tf.cast(left_tensor, tf.float64)
                                - tf.cast(right_tensor, tf.float64)
                            )
                        ).numpy()
                    )
                except (TypeError, ValueError):
                    residual = None
            return {
                "label": label,
                "passed": False,
                "comparison": "explanatory_only",
                "maximum_absolute_residual": residual,
                "error": f"{type(error).__name__}: {error}",
                "diagnostic_role": "explanatory_stochastic_route_telemetry",
                "hard_veto": False,
            }

    def explain_value(left: Any, right: Any, label: str) -> list[Mapping[str, Any]]:
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                return [
                    {
                        "label": label,
                        "passed": False,
                        "comparison": "explanatory_only",
                        "error": "mapping/scalar disagreement between stochastic routes",
                        "diagnostic_role": "explanatory_stochastic_route_telemetry",
                        "hard_veto": False,
                    }
                ]
            nested: list[Mapping[str, Any]] = []
            for key in sorted(set(left) | set(right), key=str):
                child = f"{label}.{key}"
                if key not in left or key not in right:
                    nested.append(
                        {
                            "label": child,
                            "passed": False,
                            "comparison": "explanatory_only",
                            "error": "trace key present in only one stochastic route",
                            "diagnostic_role": "explanatory_stochastic_route_telemetry",
                            "hard_veto": False,
                        }
                    )
                else:
                    nested.extend(explain_value(left[key], right[key], child))
            return nested
        try:
            return [explain_leaf(left, right, label)]
        except Exception as error:  # noqa: BLE001 - malformed telemetry is explanatory.
            return [
                {
                    "label": label,
                    "passed": False,
                    "comparison": "explanatory_only",
                    "error": f"{type(error).__name__}: {error}",
                    "diagnostic_role": "explanatory_stochastic_route_telemetry",
                    "hard_veto": False,
                }
            ]

    rows = explain_value(control.samples, compiled.samples, f"{prefix}.samples")
    control_trace = dict(control.trace)
    compiled_trace = dict(compiled.trace)
    for key in sorted(set(control_trace) | set(compiled_trace), key=str):
        if key not in control_trace or key not in compiled_trace:
            rows.append(
                {
                    "label": f"{prefix}.trace.{key}",
                    "passed": False,
                    "comparison": "explanatory_only",
                    "error": "trace key present in only one stochastic route",
                    "diagnostic_role": "explanatory_stochastic_route_telemetry",
                    "hard_veto": False,
                }
            )
        else:
            rows.extend(
                explain_value(
                    control_trace[key], compiled_trace[key], f"{prefix}.trace.{key}"
                )
            )
    return tuple(rows)


def _explanatory_value_comparison(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> Mapping[str, Any]:
    """Record a stochastic scalar/tensor difference without imposing equality."""
    try:
        row = dict(_comparison(left, right, atol=atol, rtol=rtol, label=label))
        row.update(
            diagnostic_role="explanatory_stochastic_route_telemetry",
            hard_veto=False,
        )
        return row
    except Exception as error:  # noqa: BLE001 - route telemetry is non-vetoing.
        return {
            "label": label,
            "passed": False,
            "comparison": "explanatory_only",
            "error": f"{type(error).__name__}: {error}",
            "diagnostic_role": "explanatory_stochastic_route_telemetry",
            "hard_veto": False,
        }


def _qualification_code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve(), Path(_mechanics.__file__).resolve())):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _qualification_chain_config(
    config: FixedTransportHMCXLAQualificationConfig,
    *,
    use_xla: bool,
    adaptive: bool,
    target_scope: str,
    num_leapfrog_steps: int,
) -> _FullChainHMCConfig:
    policy = (
        _TuningPolicy.dual_averaging(
            steps=config.adaptation_steps,
            target=config.target_accept_prob,
            source=FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
        )
        if adaptive
        else _TuningPolicy.fixed(source=FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE)
    )
    return _FullChainHMCConfig(
        num_results=config.tune_num_results if adaptive else config.screen_num_results,
        num_burnin_steps=(
            config.adaptation_steps if adaptive else config.screen_num_burnin_steps
        ),
        step_size=config.initial_step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        seed=config.seed,
        use_xla=use_xla,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        tuning_policy=policy,
        target_scope=target_scope,
        chain_execution_mode="tf_function",
    )


def qualify_fixed_transport_hmc_candidate_discovery_xla(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    initial_state: Any,
    config: FixedTransportHMCXLAQualificationConfig,
    evidence_path: str | Path,
    target_scope: str | None = None,
) -> FixedTransportHMCXLAQualificationReceipt:
    """Qualify the exact transformed target and both discovery HMC phases."""

    if not isinstance(config, FixedTransportHMCXLAQualificationConfig):
        raise TypeError("config must be FixedTransportHMCXLAQualificationConfig")
    output = Path(evidence_path).resolve()
    if output.exists():
        raise FileExistsError(f"XLA qualification artifact already exists: {output}")
    capability = value_score_capability(base_adapter)
    scope = target_scope or (
        None
        if capability.target_scope is None
        else f"{capability.target_scope}:fixed_transport_candidate_discovery"
    )
    if not scope:
        raise ValueError("target_scope is required when the base adapter has none")
    adapter = _adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=scope,
    )
    state = _validate_start_bank(
        initial_state,
        chain_count=4,
        parameter_dim=adapter.parameter_dim,
        require_distinct=False,
        tolerance=0.0,
    )

    eager_value, eager_score = adapter.log_prob_and_grad(state)
    compiled_target = tf.function(
        adapter.log_prob_and_grad, jit_compile=True, reduce_retracing=True
    )
    compiled_value, compiled_score = compiled_target(state)
    comparisons: list[Mapping[str, Any]] = [
        _comparison(
            eager_value,
            compiled_value,
            atol=config.value_atol,
            rtol=config.value_rtol,
            label="transformed_target.value",
        ),
        _comparison(
            eager_score,
            compiled_score,
            atol=config.score_atol,
            rtol=config.score_rtol,
            label="transformed_target.score",
        ),
    ]
    run_records = []
    for leapfrog in config.primary_l_grid:
        for adaptive in (True, False):
            phase = "dual_averaging" if adaptive else "fixed_screen"
            label = f"L{leapfrog}.{phase}"
            control = _run_full_chain_tfp_hmc(
                adapter,
                state,
                _qualification_chain_config(
                    config,
                    use_xla=False,
                    adaptive=adaptive,
                    target_scope=scope,
                    num_leapfrog_steps=leapfrog,
                ),
            )
            compiled = _run_full_chain_tfp_hmc(
                adapter,
                state,
                _qualification_chain_config(
                    config,
                    use_xla=True,
                    adaptive=adaptive,
                    target_scope=scope,
                    num_leapfrog_steps=leapfrog,
                ),
            )
            comparisons.extend(
                _explanatory_route_comparisons(
                    control,
                    compiled,
                    atol=config.transition_atol,
                    rtol=config.transition_rtol,
                    prefix=label,
                )
            )
            if compiled.metadata.get("jit_compile") is not True:
                raise ValueError(f"{label} did not execute with jit_compile=True")
            run_records.append(
                {
                    "phase": phase,
                    "num_leapfrog_steps": leapfrog,
                    "control_metadata": _json_ready(control.metadata),
                    "xla_metadata": _json_ready(compiled.metadata),
                    "control_diagnostics": _json_ready(control.diagnostics),
                    "xla_diagnostics": _json_ready(compiled.diagnostics),
                }
            )

    base_signature = _base_adapter_signature(base_adapter)
    code_hash = _qualification_code_hash()
    artifact = {
        "schema": "bayesfilter.fixed_transport_hmc_xla_qualification_evidence.v1",
        "status": "passed",
        "route": FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
        "base_adapter_signature": base_signature,
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "transformed_adapter_signature": adapter.adapter_signature(),
        "initial_state_shape": tuple(int(item) for item in state.shape),
        "target_scope": scope,
        "qualification_code_hash": code_hash,
        "config": config.payload(),
        "comparisons": tuple(comparisons),
        "runs": tuple(run_records),
        "nonclaims": (
            "bounded exact-route XLA qualification only",
            "no candidate, convergence, retained-sampling, or posterior claim",
        ),
    }
    encoded = (json.dumps(_json_ready(artifact), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return FixedTransportHMCXLAQualificationReceipt(
        base_adapter_signature=base_signature,
        fixed_transport_manifest_hash=adapter.transport_manifest_hash,
        transformed_adapter_signature=adapter.adapter_signature(),
        initial_state_shape=tuple(int(item) for item in state.shape),
        target_scope=scope,
        evidence_path=str(output),
        evidence_sha256=hashlib.sha256(encoded).hexdigest(),
        qualification_code_hash=code_hash,
    )


@dataclass(frozen=True)
class FixedTransportHMCCandidateDiscoveryConfig:
    """Explicit bounded work policy for candidate discovery, not admission."""

    initial_step_size: float
    adaptation_steps: int
    tune_num_results: int
    screen_num_results: int
    screen_num_burnin_steps: int
    xla_qualification: FixedTransportHMCXLAQualificationReceipt
    primary_l_grid: tuple[int, ...] = PRIMARY_L_GRID
    chain_count: int = 4
    replication_count: int = 2
    target_accept_prob: float = 0.70
    nomination_band: tuple[float, float] = (0.65, 0.75)
    require_distinct_starts: bool = False
    start_distinct_tolerance: float = 0.0
    movement_tolerance: float = 0.0
    tune_seed_base: tuple[int, int] = (20260803, 1000)
    screen_seed_base: tuple[int, int] = (20260803, 2000)
    target_scope: str | None = None
    target_status_trace_policy: str = "per_chain_step"
    output_filename: str = "fixed_transport_hmc_candidate_discovery.json"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_step_size",
            _positive_float(self.initial_step_size, name="initial_step_size"),
        )
        for name in (
            "adaptation_steps",
            "tune_num_results",
            "screen_num_results",
            "screen_num_burnin_steps",
            "chain_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        if self.chain_count != 4:
            raise ValueError("candidate discovery requires exactly four chains")
        if int(self.replication_count) != 2:
            raise ValueError("candidate discovery requires exactly two replications")
        object.__setattr__(self, "replication_count", 2)
        grid = tuple(int(item) for item in self.primary_l_grid)
        if grid != PRIMARY_L_GRID:
            raise ValueError(f"primary_l_grid must equal {PRIMARY_L_GRID}")
        object.__setattr__(self, "primary_l_grid", grid)
        target = float(self.target_accept_prob)
        if target != 0.70:
            raise ValueError("candidate-discovery target_accept_prob must equal 0.70")
        object.__setattr__(self, "target_accept_prob", target)
        band = _band(self.nomination_band)
        if band != (0.65, 0.75):
            raise ValueError("candidate-discovery nomination_band must equal (0.65, 0.75)")
        object.__setattr__(self, "nomination_band", band)
        for name in ("start_distinct_tolerance", "movement_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tune_seed_base", _seed(self.tune_seed_base, name="tune_seed_base"))
        object.__setattr__(
            self,
            "screen_seed_base",
            _seed(self.screen_seed_base, name="screen_seed_base"),
        )
        if self.target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("target_status_trace_policy is invalid")
        if not isinstance(self.xla_qualification, FixedTransportHMCXLAQualificationReceipt):
            raise TypeError("xla_qualification must be a qualification receipt")

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["xla_qualification"] = self.xla_qualification.payload()
        payload.update(
            {
                "route": FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
                "runtime_numerical_backend": "tensorflow_tfp_xla_only",
                "selection_performed": False,
                "confirmation_performed": False,
            }
        )
        return payload


@dataclass(frozen=True)
class FixedTransportHMCCandidateEvidence:
    num_leapfrog_steps: int
    tuned_step_size: float | None
    replication_means: tuple[float, ...]
    grand_mean: float | None
    sample_standard_deviation: float | None
    nomination_interval: tuple[float, float] | None
    disposition: str
    hard_rejection_reasons: tuple[str, ...]
    tune_seed: tuple[int, int]
    screen_seeds: tuple[tuple[int, int], ...]
    start_bank_signature: str

    @property
    def nominated(self) -> bool:
        return self.disposition == "candidate_nominated"

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_hmc_candidate_evidence.v1",
            **asdict(self),
            "nominated": self.nominated,
            "statistical_unit": "fresh_replication_mean_across_four_chains",
            "nomination_rule": "[grand_mean-sd,grand_mean+sd] intersects [0.65,0.75]",
            "standard_deviation_role": "sample_sd_not_standard_error",
            "selection_performed": False,
            "confirmation_performed": False,
        }


@dataclass(frozen=True)
class FixedTransportHMCCandidateDiscoveryResult:
    config: FixedTransportHMCCandidateDiscoveryConfig
    base_adapter_signature: str
    transformed_adapter_signature: str
    fixed_transport_manifest_hash: str
    start_bank_signature: str
    candidates: tuple[FixedTransportHMCCandidateEvidence, ...]
    final_status: str
    artifact_path: str | None = None

    @property
    def nominated_candidates(self) -> tuple[FixedTransportHMCCandidateEvidence, ...]:
        return tuple(item for item in self.candidates if item.nominated)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_hmc_candidate_discovery_result.v1",
            "route": FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
            "config": self.config.payload(),
            "base_adapter_signature": self.base_adapter_signature,
            "transformed_adapter_signature": self.transformed_adapter_signature,
            "fixed_transport_manifest_hash": self.fixed_transport_manifest_hash,
            "start_bank_signature": self.start_bank_signature,
            "candidates": tuple(item.payload() for item in self.candidates),
            "nominated_l_values": tuple(
                item.num_leapfrog_steps for item in self.nominated_candidates
            ),
            "final_status": self.final_status,
            "artifact_path": self.artifact_path,
            "selected_candidate_index": None,
            "final_kernel_payload": None,
            "refinement_performed": False,
            "exact_l_retuning_performed": False,
            "confirmation_performed": False,
            "retained_sampling_authorized": False,
            "nonclaims": CANDIDATE_DISCOVERY_NONCLAIMS,
        }


@dataclass(frozen=True)
class FixedTransportHMCCandidateCampaignConfig:
    """One-process qualification and candidate-discovery work contract."""

    initial_step_size: float
    adaptation_steps: int
    tune_num_results: int
    screen_num_results: int
    screen_num_burnin_steps: int
    value_atol: float
    value_rtol: float
    score_atol: float
    score_rtol: float
    transition_atol: float
    transition_rtol: float
    wall_cap_seconds: float
    primary_l_grid: tuple[int, ...] = PRIMARY_L_GRID
    chain_count: int = 4
    replication_count: int = 2
    target_accept_prob: float = 0.70
    nomination_band: tuple[float, float] = (0.65, 0.75)
    require_distinct_starts: bool = True
    start_distinct_tolerance: float = 0.0
    movement_tolerance: float = 0.0
    qualification_seed_base: tuple[int, int] = (20260804, 100)
    tune_seed_base: tuple[int, int] = (20260804, 1000)
    screen_seed_base: tuple[int, int] = (20260804, 2000)
    target_scope: str | None = None
    target_status_trace_policy: str = "per_chain_step"
    output_filename: str = "fixed_transport_hmc_candidate_campaign.json"
    gaussian_warm_start: bool = True
    gaussian_pilot_epsilon_min: float = 1.0e-3
    gaussian_pilot_epsilon_max: float = 1.95
    gaussian_pilot_grid_size: int = 257
    gaussian_pilot_draw_count: int = 4096
    gaussian_pilot_seed: tuple[int, int] = (20260805, 7701)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_step_size",
            _positive_float(self.initial_step_size, name="initial_step_size"),
        )
        for name in (
            "adaptation_steps",
            "tune_num_results",
            "screen_num_results",
            "screen_num_burnin_steps",
            "chain_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        if self.chain_count != 4 or int(self.replication_count) != 2:
            raise ValueError("candidate campaign requires four chains and two replications")
        object.__setattr__(self, "replication_count", 2)
        grid = tuple(int(item) for item in self.primary_l_grid)
        if grid != PRIMARY_L_GRID:
            raise ValueError(f"primary_l_grid must equal {PRIMARY_L_GRID}")
        object.__setattr__(self, "primary_l_grid", grid)
        if float(self.target_accept_prob) != 0.70:
            raise ValueError("candidate-campaign target_accept_prob must equal 0.70")
        object.__setattr__(self, "target_accept_prob", 0.70)
        band = _band(self.nomination_band)
        if band != (0.65, 0.75):
            raise ValueError("candidate-campaign nomination_band must equal (0.65, 0.75)")
        object.__setattr__(self, "nomination_band", band)
        for name in (
            "value_atol",
            "value_rtol",
            "score_atol",
            "score_rtol",
            "transition_atol",
            "transition_rtol",
            "start_distinct_tolerance",
            "movement_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "wall_cap_seconds",
            _positive_float(self.wall_cap_seconds, name="wall_cap_seconds"),
        )
        for name in ("qualification_seed_base", "tune_seed_base", "screen_seed_base"):
            object.__setattr__(self, name, _seed(getattr(self, name), name=name))
        if self.target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("target_status_trace_policy is invalid")
        if not str(self.output_filename):
            raise ValueError("output_filename must be non-empty")
        object.__setattr__(self, "gaussian_warm_start", bool(self.gaussian_warm_start))
        for name in ("gaussian_pilot_epsilon_min", "gaussian_pilot_epsilon_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("gaussian_pilot_grid_size", "gaussian_pilot_draw_count"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "gaussian_pilot_seed",
            _seed(self.gaussian_pilot_seed, name="gaussian_pilot_seed"),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            **asdict(self),
            "route": FIXED_TRANSPORT_CANDIDATE_CAMPAIGN_ROUTE,
            "runtime_numerical_backend": "tensorflow_tfp_xla_with_non_xla_oracle",
            "selection_performed": False,
            "confirmation_performed": False,
        }


@dataclass(frozen=True)
class FixedTransportHMCCandidateCampaignResult:
    config: FixedTransportHMCCandidateCampaignConfig
    base_adapter_signature: str
    transformed_adapter_signature: str
    fixed_transport_manifest_hash: str
    start_bank_signature: str
    candidates: tuple[FixedTransportHMCCandidateEvidence, ...]
    target_qualification: Mapping[str, Any]
    arm_evidence: tuple[Mapping[str, Any], ...]
    runner_evidence: Mapping[str, Any]
    hmc_call_count: int
    elapsed_seconds: float
    final_status: str
    epsilon_warm_starts: Mapping[int, Mapping[str, Any]]
    campaign_blockers: tuple[str, ...] = ()
    artifact_path: str | None = None

    @property
    def nominated_candidates(self) -> tuple[FixedTransportHMCCandidateEvidence, ...]:
        return tuple(item for item in self.candidates if item.nominated)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_hmc_candidate_campaign_result.v2",
            "route": FIXED_TRANSPORT_CANDIDATE_CAMPAIGN_ROUTE,
            "config": self.config.payload(),
            "base_adapter_signature": self.base_adapter_signature,
            "transformed_adapter_signature": self.transformed_adapter_signature,
            "fixed_transport_manifest_hash": self.fixed_transport_manifest_hash,
            "start_bank_signature": self.start_bank_signature,
            "target_qualification": self.target_qualification,
            "runner_evidence": self.runner_evidence,
            "arm_evidence": self.arm_evidence,
            "candidates": tuple(item.payload() for item in self.candidates),
            "nominated_l_values": tuple(
                item.num_leapfrog_steps for item in self.nominated_candidates
            ),
            "hmc_call_count": self.hmc_call_count,
            "elapsed_seconds": self.elapsed_seconds,
            "final_status": self.final_status,
            "epsilon_warm_starts": self.epsilon_warm_starts,
            "campaign_blockers": self.campaign_blockers,
            "artifact_path": self.artifact_path,
            "draw_role": "all qualification adaptation and screen draws discarded",
            "selected_candidate_index": None,
            "final_kernel_payload": None,
            "refinement_performed": False,
            "exact_l_retuning_performed": False,
            "confirmation_performed": False,
            "retained_sampling_authorized": False,
            "nonclaims": CANDIDATE_DISCOVERY_NONCLAIMS,
        }


@dataclass(frozen=True)
class FixedTransportCandidateRefinementConfig:
    """Bounded fresh fixed-kernel candidate refinement policy."""

    # Candidate nomination uses a discarded warm-up block followed by a
    # continuation diagnostic block. These are not convergence claims.
    stage_num_results: tuple[int, int] = (500, 500)
    num_burnin_steps: int = 500
    chain_count: int = 4
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    seed_base: tuple[int, int] = (20260808, 2600)
    repair_lower_multiplier: float = 0.80
    repair_higher_multiplier: float = 1.20
    max_epsilon_repairs_per_stage: int = 1
    wall_cap_seconds: float = 24.0 * 3600.0
    target_scope: str | None = None
    target_status_trace_policy: str = "per_chain_step"
    output_filename: str = "fixed_transport_candidate_refinement.json"
    xla_qualification: SequentialNeuTraHMCXLAQualificationReceipt | None = None
    xla_qualification_required: bool = False

    def __post_init__(self) -> None:
        stages = tuple(int(item) for item in self.stage_num_results)
        if stages != (500, 500):
            raise ValueError("stage_num_results must equal (500, 500)")
        object.__setattr__(self, "stage_num_results", stages)
        burnin = int(self.num_burnin_steps)
        if burnin <= 0:
            raise ValueError("num_burnin_steps must be positive")
        object.__setattr__(self, "num_burnin_steps", burnin)
        if int(self.chain_count) != 4:
            raise ValueError("candidate refinement requires four chains")
        object.__setattr__(self, "chain_count", 4)
        if float(self.target_accept_prob) != 0.70:
            raise ValueError("target_accept_prob must equal 0.70")
        object.__setattr__(self, "target_accept_prob", 0.70)
        band = _band(self.acceptance_band)
        if band != (0.65, 0.75):
            raise ValueError("acceptance_band must equal (0.65, 0.75)")
        object.__setattr__(self, "acceptance_band", band)
        object.__setattr__(self, "seed_base", _seed(self.seed_base, name="seed_base"))
        for name in ("repair_lower_multiplier", "repair_higher_multiplier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if not (0.0 < self.repair_lower_multiplier < 1.0 < self.repair_higher_multiplier):
            raise ValueError("epsilon repair multipliers must straddle one")
        if int(self.max_epsilon_repairs_per_stage) != 1:
            raise ValueError("at most one epsilon repair per stage is supported")
        object.__setattr__(self, "max_epsilon_repairs_per_stage", 1)
        wall_cap = float(self.wall_cap_seconds)
        if not math.isfinite(wall_cap) or wall_cap <= 0.0:
            raise ValueError("wall_cap_seconds must be positive and finite")
        object.__setattr__(self, "wall_cap_seconds", wall_cap)
        if self.target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("target_status_trace_policy is invalid")
        if not str(self.output_filename):
            raise ValueError("output_filename must be non-empty")
        required = bool(self.xla_qualification_required)
        object.__setattr__(self, "xla_qualification_required", required)
        if required and not isinstance(
            self.xla_qualification, SequentialNeuTraHMCXLAQualificationReceipt
        ):
            raise ValueError("candidate refinement requires exact XLA qualification")

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["xla_qualification"] = (
            None
            if self.xla_qualification is None
            else self.xla_qualification.payload()
        )
        return {
            **payload,
            "route": CANDIDATE_REFINEMENT_ROUTE,
            "runtime_numerical_backend": "tensorflow_tfp_xla_fixed_kernel",
            "draw_role": "burnin and diagnostic draws discarded",
            "selection_performed": True,
            "retained_sampling_authorized": False,
        }


@dataclass(frozen=True)
class FixedTransportCandidateRefinementResult:
    """Immutable summaries from the bounded refinement and optional selection."""

    config: FixedTransportCandidateRefinementConfig
    base_adapter_signature: str
    transformed_adapter_signature: str
    fixed_transport_manifest_hash: str
    start_bank_signature: str
    stages: tuple[Mapping[str, Any], ...]
    selected_candidate: Mapping[str, Any] | None
    elapsed_seconds: float
    final_status: str
    artifact_path: str | None = None

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.fixed_transport_candidate_refinement_result.v1",
            "route": CANDIDATE_REFINEMENT_ROUTE,
            "config": self.config.payload(),
            "base_adapter_signature": self.base_adapter_signature,
            "transformed_adapter_signature": self.transformed_adapter_signature,
            "fixed_transport_manifest_hash": self.fixed_transport_manifest_hash,
            "start_bank_signature": self.start_bank_signature,
            "stages": self.stages,
            "selected_candidate": self.selected_candidate,
            "elapsed_seconds": self.elapsed_seconds,
            "selected_candidate_index": None,
            "final_status": self.final_status,
            "artifact_path": self.artifact_path,
            "retained_sampling_authorized": False,
            "nonclaims": CANDIDATE_REFINEMENT_NONCLAIMS,
        }


def _validate_start_bank(
    value: Any,
    *,
    chain_count: int,
    parameter_dim: int,
    require_distinct: bool,
    tolerance: float,
) -> tf.Tensor:
    state = tf.cast(tf.convert_to_tensor(value), tf.float64)
    if state.shape != (chain_count, parameter_dim):
        raise ValueError("initial_state must have static shape [chain, parameter]")
    if not bool(tf.reduce_all(tf.math.is_finite(state)).numpy()):
        raise ValueError("initial_state must be finite")
    if require_distinct:
        difference = tf.abs(state[:, tf.newaxis, :] - state[tf.newaxis, :, :])
        pairwise = tf.reduce_max(difference, axis=-1)
        diagonal = tf.eye(chain_count, dtype=tf.bool)
        off_diagonal = tf.boolean_mask(pairwise, tf.logical_not(diagonal))
        if bool(tf.reduce_any(off_diagonal <= tolerance).numpy()):
            raise ValueError("initial_state rows must be pairwise distinct")
    return state


def _adapter(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    target_scope: str,
    full_chain_xla_diagnostic_ready: bool = False,
) -> FixedTransportValueScoreAdapter:
    capability = value_score_capability(base_adapter)
    if not capability.is_accepted_xla_hmc_authority:
        raise ValueError("candidate discovery requires accepted target-XLA authority")
    return build_fixed_transport_value_score_adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
        evidence_path=capability.evidence_path,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=full_chain_xla_diagnostic_ready,
    )


def _validate_receipt(
    receipt: FixedTransportHMCXLAQualificationReceipt,
    *,
    adapter: FixedTransportValueScoreAdapter,
    base_signature: str,
    state: tf.Tensor,
    config: FixedTransportHMCCandidateDiscoveryConfig,
) -> None:
    expected = {
        "base_adapter_signature": base_signature,
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "transformed_adapter_signature": adapter.adapter_signature(),
        "initial_state_shape": tuple(int(item) for item in state.shape),
        "target_scope": adapter.target_scope,
    }
    mismatches = tuple(
        name for name, value in expected.items() if getattr(receipt, name) != value
    )
    if mismatches:
        raise ValueError("XLA qualification receipt mismatch: " + ", ".join(mismatches))
    if receipt.qualification_code_hash != _qualification_code_hash():
        raise ValueError("XLA qualification code hash is stale")
    evidence = Path(receipt.evidence_path)
    if not evidence.is_file():
        raise ValueError("XLA qualification evidence artifact is missing")
    encoded = evidence.read_bytes()
    observed_hash = hashlib.sha256(encoded).hexdigest()
    if observed_hash != receipt.evidence_sha256:
        raise ValueError("XLA qualification evidence artifact hash mismatch")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("XLA qualification evidence artifact is not valid JSON") from error
    required = {
        "schema": "bayesfilter.fixed_transport_hmc_xla_qualification_evidence.v1",
        "status": "passed",
        "route": FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
        "base_adapter_signature": base_signature,
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "transformed_adapter_signature": adapter.adapter_signature(),
        "initial_state_shape": list(int(item) for item in state.shape),
        "target_scope": adapter.target_scope,
        "qualification_code_hash": receipt.qualification_code_hash,
    }
    payload_values = {name: payload.get(name) for name in required}
    if payload_values != required:
        raise ValueError("XLA qualification evidence payload mismatch")
    qualification = payload.get("config")
    if not isinstance(qualification, Mapping):
        raise ValueError("XLA qualification evidence config is missing")
    exact = {
        "initial_step_size": config.initial_step_size,
        "adaptation_steps": config.adaptation_steps,
        "tune_num_results": config.tune_num_results,
        "screen_num_results": config.screen_num_results,
        "screen_num_burnin_steps": config.screen_num_burnin_steps,
        "primary_l_grid": list(config.primary_l_grid),
        "target_accept_prob": config.target_accept_prob,
        "target_status_trace_policy": config.target_status_trace_policy,
    }
    observed = {name: qualification.get(name) for name in exact}
    if observed != exact:
        raise ValueError("XLA qualification evidence does not match discovery config")


def _chain_config(
    config: FixedTransportHMCCandidateDiscoveryConfig,
    *,
    num_results: int,
    burnin: int,
    step_size: float,
    leapfrog: int,
    seed: tuple[int, int],
    adaptive: bool,
) -> _FullChainHMCConfig:
    policy = (
        _TuningPolicy.dual_averaging(
            steps=config.adaptation_steps,
            target=config.target_accept_prob,
            source=FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
        )
        if adaptive
        else _TuningPolicy.fixed(source=FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE)
    )
    return _FullChainHMCConfig(
        num_results=num_results,
        num_burnin_steps=burnin,
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        seed=seed,
        use_xla=True,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        tuning_policy=policy,
        target_scope=str(config.target_scope),
        chain_execution_mode="tf_function",
    )


def _screen_summary(
    run: Any,
    *,
    initial_state: tf.Tensor,
    movement_tolerance: float,
    status_required: bool,
) -> tuple[float | None, tuple[str, ...]]:
    diagnostics = _tensor_diagnostics(run.samples, run.trace)
    runtime = dict(run.diagnostics)
    if "target_status_telemetry" in runtime:
        diagnostics["target_status_telemetry"] = runtime["target_status_telemetry"]
    reasons: list[str] = []
    for name in ("samples_all_finite", "log_accept_ratio_finite", "target_log_prob_finite"):
        if diagnostics.get(name) is not True:
            reasons.append(name)
    if diagnostics.get("proposed_target_log_prob_finite") is False:
        reasons.append("proposed_target_log_prob_nonfinite")
    if diagnostics.get("target_score_finite") is False:
        reasons.append("target_score_nonfinite")
    divergence = diagnostics.get("divergence_count")
    if diagnostics.get("divergence_status") == "available" and divergence is not None and int(divergence) > 0:
        reasons.append("native_divergence_detected")
    telemetry = diagnostics.get("target_status_telemetry")
    if status_required:
        status = telemetry
        if status is None and "target_status_telemetry" in run.trace:
            status = _target_status_diagnostics(run.trace["target_status_telemetry"])
        if not isinstance(status, Mapping):
            reasons.append("target_status_telemetry_missing")
        elif bool(status.get("telemetry_failure_veto")):
            reasons.append("target_status_telemetry_failure")
    samples = tf.cast(tf.convert_to_tensor(run.samples), tf.float64)
    displacement = tf.reduce_max(tf.abs(samples - initial_state[tf.newaxis, :, :]), axis=(0, 2))
    if bool(tf.reduce_any(displacement <= movement_tolerance).numpy()):
        reasons.append("chain_without_movement")
    acceptance = diagnostics.get("acceptance_rate")
    if acceptance is None or not math.isfinite(float(acceptance)):
        reasons.append("acceptance_missing_or_nonfinite")
        return None, tuple(dict.fromkeys(reasons))
    return float(acceptance), tuple(dict.fromkeys(reasons))


def _discover_one_l(
    config: FixedTransportHMCCandidateDiscoveryConfig,
    *,
    adapter: FixedTransportValueScoreAdapter,
    initial_state: tf.Tensor,
    start_signature: str,
    leapfrog: int,
    index: int,
    run_full_chain: RunFullChainFn,
) -> FixedTransportHMCCandidateEvidence:
    tune_seed = _offset_seed(config.tune_seed_base, index)
    screen_seeds = tuple(
        _offset_seed(config.screen_seed_base, index * config.replication_count + rep)
        for rep in range(config.replication_count)
    )
    try:
        tune = run_full_chain(
            adapter,
            initial_state,
            _chain_config(
                config,
                num_results=config.tune_num_results,
                burnin=config.adaptation_steps,
                step_size=config.initial_step_size,
                leapfrog=leapfrog,
                seed=tune_seed,
                adaptive=True,
            ),
        )
        tune_diagnostics = _tensor_diagnostics(tune.samples, tune.trace)
        if any(
            tune_diagnostics.get(name) is not True
            for name in (
                "samples_all_finite",
                "log_accept_ratio_finite",
                "target_log_prob_finite",
            )
        ):
            raise ValueError("adaptation returned nonfinite required diagnostics")
        divergence = tune_diagnostics.get("divergence_count")
        if (
            tune_diagnostics.get("divergence_status") == "available"
            and divergence is not None
            and int(divergence) > 0
        ):
            raise ValueError("adaptation exposed a positive native divergence")
        if config.target_status_trace_policy == "per_chain_step":
            telemetry = tune_diagnostics.get("target_status_telemetry")
            if not isinstance(telemetry, Mapping):
                raise ValueError("adaptation target status telemetry is missing")
            if bool(telemetry.get("telemetry_failure_veto")):
                raise ValueError("adaptation target status telemetry failed")
        step_tensor = tf.reshape(tf.convert_to_tensor(tune.diagnostics["final_step_size"]), (-1,))
        tuned_step = float(step_tensor[-1].numpy())
        if not math.isfinite(tuned_step) or tuned_step <= 0.0:
            raise ValueError("adaptation returned invalid final step size")
    except Exception as error:  # noqa: BLE001 - candidate-local typed failure.
        return FixedTransportHMCCandidateEvidence(
            leapfrog,
            None,
            (),
            None,
            None,
            None,
            "adaptation_failed",
            (f"{type(error).__name__}: {error}",),
            tune_seed,
            screen_seeds,
            start_signature,
        )
    replication_means: list[float] = []
    reasons: list[str] = []
    for seed in screen_seeds:
        try:
            run = run_full_chain(
                adapter,
                initial_state,
                _chain_config(
                    config,
                    num_results=config.screen_num_results,
                    burnin=config.screen_num_burnin_steps,
                    step_size=tuned_step,
                    leapfrog=leapfrog,
                    seed=seed,
                    adaptive=False,
                ),
            )
            mean, local = _screen_summary(
                run,
                initial_state=initial_state,
                movement_tolerance=config.movement_tolerance,
                status_required=config.target_status_trace_policy == "per_chain_step",
            )
            reasons.extend(local)
            if mean is not None:
                replication_means.append(mean)
        except Exception as error:  # noqa: BLE001 - candidate-local typed failure.
            reasons.append(f"{type(error).__name__}: {error}")
    if reasons or len(replication_means) != 2:
        return FixedTransportHMCCandidateEvidence(
            leapfrog,
            tuned_step,
            tuple(replication_means),
            None,
            None,
            None,
            "hard_rejected",
            tuple(dict.fromkeys(reasons or ("incomplete_replications",))),
            tune_seed,
            screen_seeds,
            start_signature,
        )
    grand_mean = math.fsum(replication_means) / 2.0
    sd = math.sqrt(math.fsum((item - grand_mean) ** 2 for item in replication_means))
    interval = (max(0.0, grand_mean - sd), min(1.0, grand_mean + sd))
    low, high = config.nomination_band
    nominated = interval[1] >= low and interval[0] <= high
    return FixedTransportHMCCandidateEvidence(
        leapfrog,
        tuned_step,
        tuple(replication_means),
        grand_mean,
        sd,
        interval,
        "candidate_nominated" if nominated else "statistically_incompatible",
        (),
        tune_seed,
        screen_seeds,
        start_signature,
    )


def discover_fixed_transport_hmc_candidates(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    initial_state: Any,
    config: FixedTransportHMCCandidateDiscoveryConfig,
    output_dir: str | Path | None = None,
    run_full_chain: RunFullChainFn = _run_full_chain_tfp_hmc,
) -> FixedTransportHMCCandidateDiscoveryResult:
    """Return all nominated primary-grid arms without selecting a kernel."""

    if not isinstance(config, FixedTransportHMCCandidateDiscoveryConfig):
        raise TypeError("config must be FixedTransportHMCCandidateDiscoveryConfig")
    path = None if output_dir is None else Path(output_dir) / config.output_filename
    if path is not None and path.exists():
        raise FileExistsError(f"candidate-discovery artifact already exists: {path}")
    capability = value_score_capability(base_adapter)
    target_scope = config.target_scope or (
        None if capability.target_scope is None else f"{capability.target_scope}:fixed_transport_candidate_discovery"
    )
    if not target_scope:
        raise ValueError("target_scope is required when the base adapter has none")
    config = replace(config, target_scope=target_scope)
    adapter = _adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
    )
    state = _validate_start_bank(
        initial_state,
        chain_count=config.chain_count,
        parameter_dim=adapter.parameter_dim,
        require_distinct=config.require_distinct_starts,
        tolerance=config.start_distinct_tolerance,
    )
    base_signature = _base_adapter_signature(base_adapter)
    _validate_receipt(
        config.xla_qualification,
        adapter=adapter,
        base_signature=base_signature,
        state=state,
        config=config,
    )
    start_signature = _stable_hash(
        {
            "coordinate_system": "fixed_transport_z",
            "initial_state": _json_ready(state),
        }
    )
    candidates = tuple(
        _discover_one_l(
            config,
            adapter=adapter,
            initial_state=state,
            start_signature=start_signature,
            leapfrog=leapfrog,
            index=index,
            run_full_chain=run_full_chain,
        )
        for index, leapfrog in enumerate(config.primary_l_grid)
    )
    nominated = tuple(item for item in candidates if item.nominated)
    result = FixedTransportHMCCandidateDiscoveryResult(
        config=config,
        base_adapter_signature=base_signature,
        transformed_adapter_signature=adapter.adapter_signature(),
        fixed_transport_manifest_hash=adapter.transport_manifest_hash,
        start_bank_signature=start_signature,
        candidates=candidates,
        final_status="candidate_set_found" if nominated else "no_candidate_nominated",
    )
    if output_dir is None:
        return result
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    result = FixedTransportHMCCandidateDiscoveryResult(
        **{**result.__dict__, "artifact_path": str(path)}
    )
    path.write_text(
        json.dumps(_json_ready(result.payload()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _campaign_chain_config(
    config: FixedTransportHMCCandidateCampaignConfig,
    *,
    target_scope: str,
    adaptive: bool,
    use_xla: bool,
    initial_step_size: float,
) -> _FullChainHMCConfig:
    policy = (
        _TuningPolicy.dual_averaging(
            steps=config.adaptation_steps,
            target=config.target_accept_prob,
            source=FIXED_TRANSPORT_CANDIDATE_CAMPAIGN_ROUTE,
        )
        if adaptive
        else _TuningPolicy.fixed(source=FIXED_TRANSPORT_CANDIDATE_CAMPAIGN_ROUTE)
    )
    return _FullChainHMCConfig(
        num_results=config.tune_num_results if adaptive else config.screen_num_results,
        num_burnin_steps=(
            config.adaptation_steps if adaptive else config.screen_num_burnin_steps
        ),
        step_size=float(initial_step_size),
        num_leapfrog_steps=config.primary_l_grid[0],
        seed=(0, 0),
        use_xla=use_xla,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        tuning_policy=policy,
        target_scope=target_scope,
        chain_execution_mode="tf_function",
    )


def _campaign_target_qualification(
    adapter: FixedTransportValueScoreAdapter,
    state: tf.Tensor,
    config: FixedTransportHMCCandidateCampaignConfig,
) -> tuple[Mapping[str, Any], Any]:
    eager_value, eager_score = adapter.log_prob_and_grad(state)
    target_program = tf.function(
        adapter.log_prob_and_grad,
        input_signature=(tf.TensorSpec(tuple(state.shape), state.dtype),),
        jit_compile=True,
        reduce_retracing=True,
    )
    compiled_value, compiled_score = target_program(state)
    comparisons = (
        _comparison(
            eager_value,
            compiled_value,
            atol=config.value_atol,
            rtol=config.value_rtol,
            label="transformed_target.value",
        ),
        _comparison(
            eager_score,
            compiled_score,
            atol=config.score_atol,
            rtol=config.score_rtol,
            label="transformed_target.score",
        ),
    )
    getter = getattr(target_program, "experimental_get_tracing_count", None)
    trace_count = None if getter is None else int(getter())
    if trace_count != 1:
        raise ValueError("target XLA program tracing count must equal one")
    return (
        {
            "status": "passed",
            "comparisons": comparisons,
            "program_signature": _stable_hash(
                {
                    "adapter_signature": adapter.adapter_signature(),
                    "input_shape": tuple(int(item) for item in state.shape),
                    "dtype": state.dtype.name,
                    "jit_compile": True,
                }
            ),
            "tracing_count": trace_count,
        },
        target_program,
    )


def _campaign_run_checks(
    run: Any,
    *,
    phase: str,
    initial_state: tf.Tensor,
    movement_tolerance: float,
    status_required: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    diagnostics = _tensor_diagnostics(run.samples, run.trace)
    reasons: list[str] = []
    for name in ("samples_all_finite", "log_accept_ratio_finite", "target_log_prob_finite"):
        if diagnostics.get(name) is not True:
            reasons.append(name)
    if diagnostics.get("proposed_target_log_prob_finite") is False:
        reasons.append("proposed_target_log_prob_nonfinite")
    if diagnostics.get("target_score_finite") is False:
        reasons.append("target_score_nonfinite")
    if (
        diagnostics.get("divergence_status") == "available"
        and diagnostics.get("divergence_count") is not None
        and int(diagnostics["divergence_count"]) > 0
    ):
        reasons.append("native_divergence_detected")
    if status_required:
        status = diagnostics.get("target_status_telemetry")
        if not isinstance(status, Mapping):
            reasons.append("target_status_telemetry_missing")
        elif bool(status.get("telemetry_failure_veto")):
            reasons.append("target_status_telemetry_failure")
    if phase == "fresh_screen":
        samples = tf.cast(tf.convert_to_tensor(run.samples), tf.float64)
        displacement = tf.reduce_max(
            tf.abs(samples - initial_state[tf.newaxis, :, :]), axis=(0, 2)
        )
        moved = displacement > movement_tolerance
        diagnostics["maximum_displacement_by_chain"] = displacement.numpy().tolist()
        diagnostics["moved_by_chain"] = moved.numpy().tolist()
        if bool(tf.reduce_any(tf.logical_not(moved)).numpy()):
            reasons.append("chain_without_movement")
    return diagnostics, tuple(dict.fromkeys(reasons))


def _campaign_compare_runs(
    control: Any,
    compiled: Any,
    *,
    config: FixedTransportHMCCandidateCampaignConfig,
    prefix: str,
) -> tuple[Mapping[str, Any], ...]:
    return _explanatory_route_comparisons(
        control,
        compiled,
        atol=config.transition_atol,
        rtol=config.transition_rtol,
        prefix=prefix,
    )


def _runner_receipt(runner: Any) -> Mapping[str, Any]:
    return {
        "program_signature": runner.program_signature,
        "call_count": runner.call_count,
        "tracing_count": runner.tracing_count,
        "use_xla": runner.config.use_xla,
        "adaptation_policy": runner.config.adaptation_policy,
        "dynamic_inputs": ("current_state", "seed", "step_size", "num_leapfrog_steps"),
    }


def run_fixed_transport_hmc_candidate_campaign(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    initial_state: Any,
    config: FixedTransportHMCCandidateCampaignConfig,
    output_dir: str | Path | None = None,
    runner_factory: Any = _build_reusable_runner,
) -> FixedTransportHMCCandidateCampaignResult:
    """Qualify and discover candidates in one process using four runners."""

    if not isinstance(config, FixedTransportHMCCandidateCampaignConfig):
        raise TypeError("config must be FixedTransportHMCCandidateCampaignConfig")
    path = None if output_dir is None else Path(output_dir) / config.output_filename
    if path is not None and path.exists():
        raise FileExistsError(f"candidate campaign artifact already exists: {path}")
    capability = value_score_capability(base_adapter)
    target_scope = config.target_scope or (
        None
        if capability.target_scope is None
        else f"{capability.target_scope}:fixed_transport_candidate_campaign"
    )
    if not target_scope:
        raise ValueError("target_scope is required when the base adapter has none")
    config = replace(config, target_scope=target_scope)
    adapter = _adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
    )
    state = _validate_start_bank(
        initial_state,
        chain_count=config.chain_count,
        parameter_dim=adapter.parameter_dim,
        require_distinct=config.require_distinct_starts,
        tolerance=config.start_distinct_tolerance,
    )
    start_signature = _stable_hash(
        {"coordinate_system": "fixed_transport_z", "initial_state": _json_ready(state)}
    )
    started = time.monotonic()

    def check_cap() -> None:
        if time.monotonic() - started > config.wall_cap_seconds:
            raise TimeoutError("candidate campaign wall-time cap exhausted")

    target_qualification, _target_program = _campaign_target_qualification(
        adapter, state, config
    )
    epsilon_warm_starts = (
        gaussian_harmonic_epsilon_warm_starts(
            config.primary_l_grid,
            dimension=adapter.parameter_dim,
            target_accept_prob=config.target_accept_prob,
            epsilon_min=config.gaussian_pilot_epsilon_min,
            epsilon_max=config.gaussian_pilot_epsilon_max,
            epsilon_grid_size=config.gaussian_pilot_grid_size,
            draw_count=config.gaussian_pilot_draw_count,
            seed=config.gaussian_pilot_seed,
        )
        if config.gaussian_warm_start
        else {
            int(leapfrog): {
                "epsilon": config.initial_step_size,
                "pilot_acceptance": None,
                "target_accept_prob": config.target_accept_prob,
                "route": "caller_scalar_legacy_warm_start",
                "diagnostic_role": "warm_start_only",
            }
            for leapfrog in config.primary_l_grid
        }
    )
    # The exact target-only XLA comparison above is the campaign's local
    # authority receipt.  Only after it passes may the four full-chain XLA
    # runner contracts be constructed.
    adapter = _adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
        full_chain_xla_diagnostic_ready=True,
    )
    runner_names = ("adaptive_control", "adaptive_xla", "fixed_control", "fixed_xla")
    runners = {
        name: runner_factory(
            adapter,
            state,
            _campaign_chain_config(
                config,
                target_scope=target_scope,
                adaptive=name.startswith("adaptive"),
                use_xla=name.endswith("xla"),
                initial_step_size=config.initial_step_size,
            ),
        )
        for name in runner_names
    }
    candidates: list[FixedTransportHMCCandidateEvidence] = []
    arm_records: list[Mapping[str, Any]] = []
    hmc_call_count = 0
    for index, leapfrog in enumerate(config.primary_l_grid):
        tune_seed = _offset_seed(config.tune_seed_base, index)
        qualification_seed = _offset_seed(config.qualification_seed_base, index)
        screen_seeds = tuple(
            _offset_seed(
                config.screen_seed_base,
                index * config.replication_count + replication,
            )
            for replication in range(config.replication_count)
        )
        record: dict[str, Any] = {
            "num_leapfrog_steps": leapfrog,
            "tune_seed": tune_seed,
            "qualification_seed": qualification_seed,
            "screen_seeds": screen_seeds,
            "start_bank_signature": start_signature,
            "gaussian_warm_start": epsilon_warm_starts[leapfrog],
        }
        replication_means: list[float] = []
        replication_rows: list[Mapping[str, Any]] = []
        try:
            check_cap()
            adaptation_runs = {}
            adaptation_checks = {}
            for name in ("adaptive_control", "adaptive_xla"):
                adaptation_runs[name] = runners[name].run(
                    current_state=state,
                    seed=tune_seed,
                    step_size=tf.constant(
                        float(epsilon_warm_starts[leapfrog]["epsilon"]), tf.float64
                    ),
                    num_leapfrog_steps=tf.constant(leapfrog, tf.int32),
                )
                hmc_call_count += 1
                diagnostics, reasons = _campaign_run_checks(
                    adaptation_runs[name],
                    phase="adaptation",
                    initial_state=state,
                    movement_tolerance=config.movement_tolerance,
                    status_required=config.target_status_trace_policy == "per_chain_step",
                )
                if reasons:
                    raise ValueError(f"{name} adaptation failed: {', '.join(reasons)}")
                adaptation_checks[name] = _json_ready(diagnostics)
            adaptation_comparisons = _campaign_compare_runs(
                adaptation_runs["adaptive_control"],
                adaptation_runs["adaptive_xla"],
                config=config,
                prefix=f"L{leapfrog}.adaptation",
            )
            control_step = _terminal_step_size(adaptation_runs["adaptive_control"].trace)
            tuned_step_tensor = _terminal_step_size(adaptation_runs["adaptive_xla"].trace)
            step_comparison = _explanatory_value_comparison(
                control_step,
                tuned_step_tensor,
                atol=config.transition_atol,
                rtol=config.transition_rtol,
                label=f"L{leapfrog}.terminal_epsilon",
            )
            tuned_step = float(tuned_step_tensor.numpy())
            record.update(
                {
                    "adapted_epsilon": tuned_step,
                    "adaptation_checks": adaptation_checks,
                    "adaptation_comparisons": adaptation_comparisons,
                    "terminal_epsilon_comparison": step_comparison,
                }
            )
            fixed_runs = {}
            fixed_checks = {}
            for name in ("fixed_control", "fixed_xla"):
                check_cap()
                fixed_runs[name] = runners[name].run(
                    current_state=state,
                    seed=qualification_seed,
                    step_size=tuned_step_tensor,
                    num_leapfrog_steps=tf.constant(leapfrog, tf.int32),
                )
                hmc_call_count += 1
                diagnostics, reasons = _campaign_run_checks(
                    fixed_runs[name],
                    phase="qualification_screen",
                    initial_state=state,
                    movement_tolerance=config.movement_tolerance,
                    status_required=config.target_status_trace_policy == "per_chain_step",
                )
                if reasons:
                    raise ValueError(f"{name} qualification failed: {', '.join(reasons)}")
                fixed_checks[name] = _json_ready(diagnostics)
            fixed_comparisons = _campaign_compare_runs(
                fixed_runs["fixed_control"],
                fixed_runs["fixed_xla"],
                config=config,
                prefix=f"L{leapfrog}.exact_epsilon_screen",
            )
            record.update(
                {
                    "exact_epsilon_checks": fixed_checks,
                    "exact_epsilon_comparisons": fixed_comparisons,
                    "fresh_replications": tuple(replication_rows),
                }
            )
            for seed in screen_seeds:
                check_cap()
                run = runners["fixed_xla"].run(
                    current_state=state,
                    seed=seed,
                    step_size=tuned_step_tensor,
                    num_leapfrog_steps=tf.constant(leapfrog, tf.int32),
                )
                hmc_call_count += 1
                diagnostics, reasons = _campaign_run_checks(
                    run,
                    phase="fresh_screen",
                    initial_state=state,
                    movement_tolerance=config.movement_tolerance,
                    status_required=config.target_status_trace_policy == "per_chain_step",
                )
                if diagnostics.get("acceptance_rate_semantics") != "mean_metropolis_acceptance_probability":
                    raise ValueError("fresh screen acceptance has wrong semantics")
                mean = float(diagnostics["acceptance_rate"])
                row = {
                    "seed": seed,
                    "acceptance_mean": mean,
                    "acceptance_rate_semantics": diagnostics[
                        "acceptance_rate_semantics"
                    ],
                    "acceptance_probability_by_chain": diagnostics.get(
                        "acceptance_probability_by_chain"
                    ),
                    "binary_acceptance_rate_explanatory_only": diagnostics.get(
                        "binary_acceptance_rate"
                    ),
                    "binary_acceptance_by_chain_explanatory_only": diagnostics.get(
                        "binary_acceptance_by_chain"
                    ),
                    "hard_rejection_reasons": reasons,
                    "diagnostics": _json_ready(diagnostics),
                }
                replication_means.append(mean)
                replication_rows.append(row)
                record["fresh_replications"] = tuple(replication_rows)
                if reasons:
                    raise ValueError("fresh screen failed: " + ", ".join(reasons))
            grand_mean = math.fsum(replication_means) / 2.0
            sd = math.sqrt(
                math.fsum((item - grand_mean) ** 2 for item in replication_means)
            )
            interval = (max(0.0, grand_mean - sd), min(1.0, grand_mean + sd))
            low, high = config.nomination_band
            disposition = (
                "candidate_nominated"
                if interval[1] >= low and interval[0] <= high
                else "statistically_incompatible"
            )
            candidate = FixedTransportHMCCandidateEvidence(
                leapfrog,
                tuned_step,
                tuple(replication_means),
                grand_mean,
                sd,
                interval,
                disposition,
                (),
                tune_seed,
                screen_seeds,
                start_signature,
            )
            record.update(
                {
                    "disposition": disposition,
                    "fresh_replications": tuple(replication_rows),
                }
            )
        except TimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - arm-local evidence is preserved.
            candidate = FixedTransportHMCCandidateEvidence(
                leapfrog,
                record.get("adapted_epsilon"),
                tuple(replication_means),
                None,
                None,
                None,
                "hard_rejected",
                (f"{type(error).__name__}: {error}",),
                tune_seed,
                screen_seeds,
                start_signature,
            )
            record.update(
                {
                    "disposition": "hard_rejected",
                    "hard_rejection_reasons": candidate.hard_rejection_reasons,
                }
            )
        candidates.append(candidate)
        arm_records.append(record)

    runner_evidence = {name: _runner_receipt(runner) for name, runner in runners.items()}
    invalid_traces = tuple(
        name
        for name, receipt in runner_evidence.items()
        if receipt["call_count"] > 0 and receipt["tracing_count"] != 1
    )
    campaign_blockers = tuple(
        "runner_tracing_count_invalid:"
        + ",".join(
            f"{name}={runner_evidence[name]['tracing_count']}"
            for name in invalid_traces
        )
        for _ in (0,)
    ) if invalid_traces else ()
    nominated = tuple(item for item in candidates if item.nominated)
    elapsed = time.monotonic() - started
    result = FixedTransportHMCCandidateCampaignResult(
        config=config,
        base_adapter_signature=_base_adapter_signature(base_adapter),
        transformed_adapter_signature=adapter.adapter_signature(),
        fixed_transport_manifest_hash=adapter.transport_manifest_hash,
        start_bank_signature=start_signature,
        candidates=tuple(candidates),
        target_qualification=target_qualification,
        arm_evidence=tuple(arm_records),
        runner_evidence=runner_evidence,
        hmc_call_count=hmc_call_count,
        elapsed_seconds=elapsed,
        final_status=(
            "campaign_invalid_runner_tracing"
            if campaign_blockers
            else ("candidate_set_found" if nominated else "no_candidate_nominated")
        ),
        epsilon_warm_starts=epsilon_warm_starts,
        campaign_blockers=campaign_blockers,
    )
    if path is None:
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    result = FixedTransportHMCCandidateCampaignResult(
        **{**result.__dict__, "artifact_path": str(path)}
    )
    path.write_text(
        json.dumps(_json_ready(result.payload()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _refinement_candidate_payload(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        leapfrog = candidate.get("num_leapfrog_steps")
        epsilon = candidate.get("tuned_step_size", candidate.get("epsilon"))
    else:
        leapfrog = getattr(candidate, "num_leapfrog_steps", None)
        epsilon = getattr(candidate, "tuned_step_size", getattr(candidate, "epsilon", None))
    leapfrog = int(leapfrog)
    epsilon = float(epsilon)
    if leapfrog <= 0 or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("refinement candidates require positive L and epsilon")
    return {"num_leapfrog_steps": leapfrog, "epsilon": epsilon}


def _refinement_acceptance_interval(
    chain_means: Sequence[float],
) -> tuple[float, float, float, float]:
    values = tuple(float(item) for item in chain_means)
    if len(values) != 4 or any(not math.isfinite(item) for item in values):
        raise ValueError("refinement requires four finite chain acceptance means")
    mean = math.fsum(values) / 4.0
    sd = math.sqrt(math.fsum((item - mean) ** 2 for item in values) / 3.0)
    interval = (max(0.0, mean - sd), min(1.0, mean + sd))
    return mean, sd, interval[0], interval[1]


def _refinement_screen_one(
    run: Any,
    *,
    initial_state: tf.Tensor,
    config: FixedTransportCandidateRefinementConfig,
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    diagnostics, reasons = _campaign_run_checks(
        run,
        phase="fresh_screen",
        initial_state=initial_state,
        movement_tolerance=0.0,
        status_required=config.target_status_trace_policy == "per_chain_step",
    )
    chain_means = diagnostics.get("acceptance_probability_by_chain")
    if chain_means is None or len(chain_means) != config.chain_count:
        reasons = tuple(dict.fromkeys((*reasons, "acceptance_by_chain_missing")))
    rhat_payload: Mapping[str, Any] = {}
    ess_payload: Mapping[str, Any] = {}
    if not reasons:
        chain_major = tf.transpose(tf.cast(run.samples, tf.float64), perm=(1, 0, 2))
        try:
            raw_rhat = rank_normalized_split_rhat(chain_major)
            maximum_rhat = tf.cast(raw_rhat["maximum"], tf.float64)
            if not bool(tf.reduce_all(tf.math.is_finite(maximum_rhat)).numpy()):
                raise ValueError("rank-normalized split R-hat is nonfinite")
            rhat_payload = {
                **_mechanics.fixed_transport_json_ready(raw_rhat),
                "maximum_over_parameters": float(tf.reduce_max(maximum_rhat).numpy()),
            }
            ess_payload = _mechanics.fixed_transport_json_ready(
                rank_normalized_bulk_tail_ess(chain_major)
            )
        except Exception as error:  # diagnostic failure is candidate-local.
            reasons = (f"{type(error).__name__}: {error}",)
    if reasons:
        return {
            **candidate,
            "chain_acceptance_means": chain_means,
            "hard_rejection_reasons": tuple(dict.fromkeys(reasons)),
            "disposition": "hard_rejected",
            "diagnostics": _mechanics.fixed_transport_json_ready(diagnostics),
            "diagnostic_coordinate": "hmc_coordinates_z",
            "rhat": rhat_payload,
            "ess": ess_payload,
        }
    mean, sd, low, high = _refinement_acceptance_interval(chain_means)
    band_low, band_high = config.acceptance_band
    return {
        **candidate,
        "chain_acceptance_means": tuple(float(item) for item in chain_means),
        "grand_mean": mean,
        "sample_sd": sd,
        "acceptance_interval": (low, high),
        "acceptance_band": config.acceptance_band,
        "diagnostic_coordinate": "hmc_coordinates_z",
        "rhat": rhat_payload,
        "ess": ess_payload,
        "diagnostics": _mechanics.fixed_transport_json_ready(diagnostics),
        "hard_rejection_reasons": (),
        "disposition": (
            "candidate_survived"
            if high >= band_low and low <= band_high
            else "acceptance_incompatible"
        ),
    }


def _refinement_repaired_candidates(
    stage_candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    config: FixedTransportCandidateRefinementConfig,
) -> tuple[Mapping[str, Any], ...]:
    by_l = {int(row["num_leapfrog_steps"]): row for row in rows}
    repaired = []
    for candidate in stage_candidates:
        row = by_l[int(candidate["num_leapfrog_steps"])]
        interval = row.get("acceptance_interval")
        if (
            isinstance(interval, (tuple, list))
            and len(interval) == 2
            and float(interval[0]) > config.acceptance_band[1]
        ):
            multiplier = config.repair_higher_multiplier
            direction = "higher_epsilon_for_high_acceptance"
        else:
            multiplier = config.repair_lower_multiplier
            direction = (
                "lower_epsilon_for_low_acceptance"
                if interval is not None
                else "lower_epsilon_after_hard_veto"
            )
        repaired.append(
            {
                "num_leapfrog_steps": int(candidate["num_leapfrog_steps"]),
                "epsilon": float(candidate["epsilon"]) * multiplier,
                "epsilon_repair_direction": direction,
                "epsilon_repair_multiplier": multiplier,
            }
        )
    return tuple(repaired)


def refine_fixed_transport_hmc_candidates(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    initial_state: Any,
    candidates: Sequence[Any],
    config: FixedTransportCandidateRefinementConfig,
    output_dir: str | Path | None = None,
    runner_factory: Any = _build_reusable_runner,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> FixedTransportCandidateRefinementResult:
    """Run the user's 500+500 continuation screen using BayesFilter mechanics only."""

    if not isinstance(config, FixedTransportCandidateRefinementConfig):
        raise TypeError("config must be FixedTransportCandidateRefinementConfig")
    if not candidates:
        raise ValueError("at least one nominated candidate is required")
    path = None if output_dir is None else Path(output_dir) / config.output_filename
    if path is not None and path.exists():
        raise FileExistsError(f"candidate refinement artifact exists: {path}")
    capability = value_score_capability(base_adapter)
    target_scope = config.target_scope or capability.target_scope
    if not target_scope:
        raise ValueError("target_scope is required")
    adapter = _adapter(
        base_adapter=base_adapter,
        fixed_transport=fixed_transport,
        target_scope=target_scope,
        full_chain_xla_diagnostic_ready=bool(config.xla_qualification_required),
    )
    state = _validate_start_bank(
        initial_state,
        chain_count=config.chain_count,
        parameter_dim=adapter.parameter_dim,
        require_distinct=True,
        tolerance=0.0,
    )
    normalized = tuple(_refinement_candidate_payload(item) for item in candidates)
    if len({(row["num_leapfrog_steps"], row["epsilon"]) for row in normalized}) != len(normalized):
        raise ValueError("refinement candidates must be distinct")
    if len({row["num_leapfrog_steps"] for row in normalized}) != len(normalized):
        raise ValueError("refinement requires one nominated epsilon per L")
    if config.xla_qualification_required:
        qualification_config = _ArchivedSequentialNeuTraHMCConfig(
            step_size=float(normalized[0]["epsilon"]),
            num_leapfrog_steps=int(normalized[0]["num_leapfrog_steps"]),
            seed=config.seed_base,
            warmup_chunk_size=config.stage_num_results[0],
            warmup_min_results=config.stage_num_results[0],
            warmup_window_results=config.stage_num_results[0],
            warmup_max_results=config.stage_num_results[0],
            retained_chunk_size=config.stage_num_results[0],
            retained_min_results=config.stage_num_results[0],
            retained_max_results=config.stage_num_results[0],
            bulk_ess_min=1.0,
            tail_ess_min=1.0,
            acceptance_min=0.0,
            acceptance_max=1.0,
            chain_count=config.chain_count,
            use_xla=True,
            target_status_required=(
                config.target_status_trace_policy == "per_chain_step"
            ),
            primary_diagnostic_coordinate="hmc_coordinates_z",
            retained_ess_required=False,
        )
        validate_sequential_neutra_hmc_xla_receipt(
            config.xla_qualification,
            adapter=adapter,
            initial_state=state,
            config=qualification_config,
        )
    start_signature = _stable_hash({"coordinate_system": "fixed_transport_z", "initial_state": _json_ready(state)})
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir).resolve()
    checkpoint_path = (
        None
        if checkpoint_root is None
        else checkpoint_root / "candidate-refinement-checkpoint.json"
    )
    run_contract = {
        "schema": "bayesfilter.fixed_transport_candidate_refinement_contract.v1",
        "route": CANDIDATE_REFINEMENT_ROUTE,
        "config": config.payload(),
        "base_adapter_signature": _base_adapter_signature(base_adapter),
        "transformed_adapter_signature": adapter.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "start_bank_signature": start_signature,
        "candidates": normalized,
    }
    run_contract_hash = _stable_hash(run_contract)
    checkpoint_entries: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    if checkpoint_root is not None:
        if resume:
            if checkpoint_path is None or not checkpoint_path.is_file():
                raise _RefinementCheckpointError(
                    "candidate refinement resume requires a checkpoint"
                )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("schema") != "bayesfilter.fixed_transport_candidate_refinement_checkpoint.v1":
                raise _RefinementCheckpointError(
                    "candidate refinement checkpoint schema mismatch"
                )
            if checkpoint.get("terminal") is not False:
                raise _RefinementCheckpointError(
                    "candidate refinement checkpoint is terminal"
                )
            if checkpoint.get("run_contract_hash") != _stable_hash(
                checkpoint.get("run_contract")
            ):
                raise _RefinementCheckpointError(
                    "candidate refinement checkpoint contract hash is invalid"
                )
            if checkpoint.get("run_contract_hash") != run_contract_hash:
                raise _RefinementCheckpointError(
                    "candidate refinement checkpoint run contract mismatch"
                )
            for entry in checkpoint.get("entries", ()):
                key = (
                    int(entry["stage_index"]),
                    int(entry["repair_attempt"]),
                    int(entry["candidate_index"]),
                )
                if key in checkpoint_entries:
                    raise _RefinementCheckpointError(
                        "candidate refinement checkpoint contains duplicate entries"
                    )
                for receipt_name in ("warmup_endpoint", "final_endpoint"):
                    receipt = entry.get(receipt_name)
                    if receipt is not None:
                        _read_refinement_tensor_receipt(receipt)
                checkpoint_entries[key] = dict(entry)
            expected_paths = {checkpoint_path.resolve()}
            for entry in checkpoint_entries.values():
                for receipt_name in ("warmup_endpoint", "final_endpoint"):
                    receipt = entry.get(receipt_name)
                    if receipt is not None:
                        expected_paths.add(Path(receipt["path"]).resolve())
            observed_paths = {
                item.resolve() for item in checkpoint_root.rglob("*") if item.is_file()
            }
            orphan_paths = sorted(observed_paths - expected_paths)
            if orphan_paths:
                raise _RefinementCheckpointError(
                    "candidate refinement checkpoint contains orphan artifacts: "
                    + ", ".join(path.as_posix() for path in orphan_paths)
                )
        else:
            if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
                raise _RefinementCheckpointError(
                    "candidate refinement checkpoint root must be new or empty"
                )
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            _write_refinement_checkpoint(
                checkpoint_path,
                {
                    "schema": "bayesfilter.fixed_transport_candidate_refinement_checkpoint.v1",
                    "run_contract": run_contract,
                    "run_contract_hash": run_contract_hash,
                    "entries": (),
                    "terminal": False,
                },
            )

    def persist_checkpoint() -> None:
        if checkpoint_path is None:
            return
        _write_refinement_checkpoint(
            checkpoint_path,
            {
                "schema": "bayesfilter.fixed_transport_candidate_refinement_checkpoint.v1",
                "run_contract": run_contract,
                "run_contract_hash": run_contract_hash,
                "entries": tuple(
                    checkpoint_entries[key] for key in sorted(checkpoint_entries)
                ),
                "terminal": False,
            },
        )
    stages: list[Mapping[str, Any]] = []
    survivors = normalized
    state_by_l = {
        int(row["num_leapfrog_steps"]): tf.identity(state) for row in normalized
    }
    started = time.monotonic()

    def check_cap() -> None:
        if time.monotonic() - started > config.wall_cap_seconds:
            raise TimeoutError("candidate refinement wall-time cap exhausted")

    for stage_index, num_results in enumerate(config.stage_num_results):
        if stage_index == 1 and len(survivors) <= 1:
            break
        stage_candidates = tuple(_refinement_candidate_payload(row) for row in survivors)
        stage_start_by_l = {
            int(row["num_leapfrog_steps"]): tf.identity(
                state_by_l[int(row["num_leapfrog_steps"])]
            )
            for row in stage_candidates
        }
        attempts: list[Mapping[str, Any]] = []
        for repair_attempt in range(config.max_epsilon_repairs_per_stage + 1):
            if repair_attempt:
                active = _refinement_repaired_candidates(
                    stage_candidates,
                    attempts[-1]["candidates"],
                    config,
                )
            else:
                active = stage_candidates
            runner = runner_factory(
                adapter,
                stage_start_by_l[int(active[0]["num_leapfrog_steps"])],
                _FullChainHMCConfig(
                    num_results=num_results,
                    # The active route uses two explicit calls to the same
                    # zero-burnin XLA graph. Legacy injected factories may
                    # still inspect this field in CPU fixtures.
                    num_burnin_steps=0 if config.xla_qualification_required else (
                        config.num_burnin_steps if stage_index == 0 else 0
                    ),
                    step_size=float(active[0]["epsilon"]),
                    num_leapfrog_steps=int(active[0]["num_leapfrog_steps"]),
                    seed=config.seed_base,
                    use_xla=True,
                    trace_policy="standard",
                    target_status_trace_policy=config.target_status_trace_policy,
                    tuning_policy=_TuningPolicy.fixed(
                        source=(
                            NEUTRA_SEQUENTIAL_HMC_POLICY_ID
                            if config.xla_qualification_required
                            else CANDIDATE_REFINEMENT_ROUTE
                        )
                    ),
                    target_scope=target_scope,
                    chain_execution_mode="tf_function",
                ),
            )
            if config.xla_qualification_required:
                observed_program = getattr(runner, "program_signature", None)
                if observed_program != config.xla_qualification.program_signature:
                    raise ValueError(
                        "candidate refinement runner does not match the exact "
                        "sequential XLA qualification program"
                    )
            rows: list[Mapping[str, Any]] = []
            next_state_by_l: dict[int, tf.Tensor] = {}
            for candidate_index, candidate in enumerate(active):
                check_cap()
                leapfrog = int(candidate["num_leapfrog_steps"])
                candidate_start = stage_start_by_l[leapfrog]
                seed = _offset_seed(
                    config.seed_base,
                    stage_index * 10000 + repair_attempt * 1000 + candidate_index,
                )
                checkpoint_key = (stage_index, repair_attempt, candidate_index)
                saved = checkpoint_entries.get(checkpoint_key)
                if saved is not None:
                    expected_saved = {
                        "candidate": _json_ready(candidate),
                        "seed": _json_ready(seed),
                    }
                    observed_saved = {
                        name: saved.get(name) for name in expected_saved
                    }
                    if observed_saved != expected_saved:
                        raise _RefinementCheckpointError(
                            "candidate refinement checkpoint entry mismatch"
                        )
                    if saved.get("phase") == "candidate_complete":
                        row = dict(saved["row"])
                        if row.get("disposition") == "candidate_survived":
                            next_state_by_l[leapfrog] = tf.cast(
                                _read_refinement_tensor_receipt(
                                    saved["final_endpoint"]
                                ),
                                tf.float64,
                            )
                        rows.append(row)
                        continue
                    if saved.get("phase") != "warmup_complete":
                        raise _RefinementCheckpointError(
                            "candidate refinement checkpoint phase is invalid"
                        )
                try:
                    if config.xla_qualification_required and stage_index == 0:
                        if saved is None:
                            warmup = runner.run(
                                current_state=candidate_start,
                                seed=_offset_seed(seed, 500000),
                                step_size=tf.constant(candidate["epsilon"], tf.float64),
                                num_leapfrog_steps=tf.constant(candidate["num_leapfrog_steps"], tf.int32),
                            )
                            warmup_endpoint = tf.cast(tf.convert_to_tensor(warmup.samples[-1]), tf.float64)
                            warmup_receipt = (
                                None
                                if checkpoint_root is None
                                else _refinement_tensor_receipt(
                                    checkpoint_root
                                    / "states"
                                    / f"s{stage_index}-a{repair_attempt}-c{candidate_index}-warmup.tftensor",
                                    warmup_endpoint,
                                )
                            )
                            if warmup_receipt is not None:
                                checkpoint_entries[checkpoint_key] = {
                                    "stage_index": stage_index,
                                    "repair_attempt": repair_attempt,
                                    "candidate_index": candidate_index,
                                    "candidate": _json_ready(candidate),
                                    "seed": _json_ready(seed),
                                    "phase": "warmup_complete",
                                    "warmup_endpoint": warmup_receipt,
                                }
                                persist_checkpoint()
                        else:
                            warmup_endpoint = tf.cast(
                                _read_refinement_tensor_receipt(
                                    saved["warmup_endpoint"]
                                ),
                                tf.float64,
                            )
                        run = runner.run(
                            current_state=warmup_endpoint,
                            seed=seed,
                            step_size=tf.constant(candidate["epsilon"], tf.float64),
                            num_leapfrog_steps=tf.constant(candidate["num_leapfrog_steps"], tf.int32),
                        )
                        candidate_start = warmup_endpoint
                    else:
                        run = runner.run(
                            current_state=candidate_start,
                            seed=seed,
                            step_size=tf.constant(candidate["epsilon"], tf.float64),
                            num_leapfrog_steps=tf.constant(candidate["num_leapfrog_steps"], tf.int32),
                        )
                    row = _refinement_screen_one(
                        run,
                        initial_state=candidate_start,
                        config=config,
                        candidate={
                            **candidate,
                            "seed": seed,
                            "discarded_burnin_transitions": (
                                config.num_burnin_steps if stage_index == 0 else 0
                            ),
                            "continued_from_previous_stage": stage_index > 0,
                            "initial_state_hash": _stable_hash(
                                {
                                    "coordinate_system": "fixed_transport_z",
                                    "state": _json_ready(candidate_start),
                                }
                            ),
                        },
                    )
                    endpoint = tf.cast(tf.convert_to_tensor(run.samples[-1]), tf.float64)
                    row = {
                        **row,
                        "final_state_hash": _stable_hash(
                            {
                                "coordinate_system": "fixed_transport_z",
                                "state": _json_ready(endpoint),
                            }
                        ),
                    }
                    if row.get("disposition") == "candidate_survived":
                        next_state_by_l[leapfrog] = endpoint
                except Exception as error:  # candidate-local hard veto.
                    if isinstance(error, _RefinementCheckpointError):
                        raise
                    row = {
                        **candidate,
                        "seed": seed,
                        "disposition": "hard_rejected",
                        "hard_rejection_reasons": (f"{type(error).__name__}: {error}",),
                    }
                    endpoint = None
                if checkpoint_root is not None:
                    prior = checkpoint_entries.get(checkpoint_key, {})
                    final_receipt = (
                        _refinement_tensor_receipt(
                            checkpoint_root
                            / "states"
                            / f"s{stage_index}-a{repair_attempt}-c{candidate_index}-final.tftensor",
                            endpoint,
                        )
                        if endpoint is not None
                        and row.get("disposition") == "candidate_survived"
                        else None
                    )
                    checkpoint_entries[checkpoint_key] = {
                        "stage_index": stage_index,
                        "repair_attempt": repair_attempt,
                        "candidate_index": candidate_index,
                        "candidate": _json_ready(candidate),
                        "seed": _json_ready(seed),
                        "phase": "candidate_complete",
                        "warmup_endpoint": prior.get("warmup_endpoint"),
                        "final_endpoint": final_receipt,
                        "row": _json_ready(row),
                    }
                    persist_checkpoint()
                rows.append(row)
            check_cap()
            attempts.append(
                {
                    "repair_attempt": repair_attempt,
                    "epsilon_policy": (
                        "exact_nominated_epsilon"
                        if repair_attempt == 0
                        else "candidate_specific_directional_repair"
                    ),
                    "candidates": tuple(rows),
                }
            )
            survivors = tuple(row for row in rows if row.get("disposition") == "candidate_survived")
            if survivors or repair_attempt == config.max_epsilon_repairs_per_stage:
                state_by_l = next_state_by_l
                break
        stages.append(
            {
                "num_results": num_results,
                "num_burnin_steps": config.num_burnin_steps if stage_index == 0 else 0,
                "stage_role": (
                    "discarded_burnin_then_candidate_diagnostic"
                    if stage_index == 0
                    else "continued_candidate_diagnostic"
                ),
                "diagnostic_coordinate": "hmc_coordinates_z",
                "attempts": tuple(attempts),
                "survivor_count": len(survivors),
            }
        )
        if not survivors:
            break
    selected = None
    if survivors:
        if len(survivors) == 1:
            selected = dict(survivors[0])
        else:
            final_rows = stages[-1]["attempts"][-1]["candidates"]
            selected = dict(min(
                (row for row in final_rows if row.get("disposition") == "candidate_survived"),
                key=lambda row: (
                    float(row["rhat"]["maximum_over_parameters"]),
                    int(row["num_leapfrog_steps"]),
                    float(row["epsilon"]),
                ),
            ))
    result = FixedTransportCandidateRefinementResult(
        config=config,
        base_adapter_signature=_base_adapter_signature(base_adapter),
        transformed_adapter_signature=adapter.adapter_signature(),
        fixed_transport_manifest_hash=adapter.transport_manifest_hash,
        start_bank_signature=start_signature,
        stages=tuple(stages),
        selected_candidate=selected,
        elapsed_seconds=time.monotonic() - started,
        final_status=("candidate_selected" if selected is not None else "no_candidate_after_refinement"),
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        result = FixedTransportCandidateRefinementResult(
            **{**result.__dict__, "artifact_path": str(path)}
        )
        path.write_text(
            json.dumps(_json_ready(result.payload()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if checkpoint_path is not None:
        terminal = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        terminal.update(
            terminal=True,
            final_status=result.final_status,
            selected_candidate=_json_ready(selected),
            result_path=None if path is None else str(path.resolve()),
            result_sha256=(
                None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()
            ),
        )
        _write_refinement_checkpoint(checkpoint_path, terminal)
    return result


__all__ = [
    "CANDIDATE_DISCOVERY_NONCLAIMS",
    "FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE",
    "FIXED_TRANSPORT_CANDIDATE_CAMPAIGN_ROUTE",
    "GAUSSIAN_EPSILON_WARM_START_ROUTE",
    "PRIMARY_L_GRID",
    "gaussian_harmonic_epsilon_warm_starts",
    "FixedTransportHMCCandidateDiscoveryConfig",
    "FixedTransportHMCCandidateDiscoveryResult",
    "FixedTransportHMCCandidateCampaignConfig",
    "FixedTransportHMCCandidateCampaignResult",
    "FixedTransportHMCCandidateEvidence",
    "FixedTransportHMCXLAQualificationConfig",
    "FixedTransportHMCXLAQualificationReceipt",
    "discover_fixed_transport_hmc_candidates",
    "qualify_fixed_transport_hmc_candidate_discovery_xla",
    "run_fixed_transport_hmc_candidate_campaign",
    "CANDIDATE_REFINEMENT_ROUTE",
    "FixedTransportCandidateRefinementConfig",
    "FixedTransportCandidateRefinementResult",
    "refine_fixed_transport_hmc_candidates",
]
