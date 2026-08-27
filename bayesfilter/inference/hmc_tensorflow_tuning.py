"""TensorFlow-native HMC tuning for a repository-issued transition binding.

This module never imports NumPy or SciPy. It keeps tuning state in TensorFlow,
reuses the bound BayesFilter transition kernel, and leaves public legacy-versus-
TensorFlow dispatch to :mod:`bayesfilter.inference.hmc_tuning_dispatch`.
The current graph is diagnostic mechanics only: it does not implement the
ordinary fresh-R-hat handoff gate or XLA qualification and cannot issue a
retained-kernel handoff.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.tuning_contract import HMCTuningRunnerBinding


TENSORFLOW_HMC_TUNING_SCHEMA = "bayesfilter.tensorflow_hmc_tuning.v1"
FOUR_CHAIN_ACCEPTANCE_SCHEMA = "bayesfilter.four_chain_mean_band_acceptance.v1"
BOUND_RETAINED_HMC_ARCHIVE_SCHEMA = "bayesfilter.bound_retained_hmc_archive.v1"
_CHAIN_COUNT = 4
_DTYPE = tf.float64


def _nonempty(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    integer = int(value)
    if integer < 0 or integer != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _positive_finite(value: Any, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return scalar


def _closed_probability_band(value: Any, name: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = values
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} values must be finite")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= lower <= upper <= 1")
    return lower, upper


@dataclass(frozen=True)
class FourChainMeanBandAcceptancePolicy:
    """Exact equal-length four-chain mean-acceptance heuristic."""

    overall_band: tuple[float, float]
    per_chain_band: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overall_band",
            _closed_probability_band(self.overall_band, "overall_band"),
        )
        object.__setattr__(
            self,
            "per_chain_band",
            _closed_probability_band(self.per_chain_band, "per_chain_band"),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": FOUR_CHAIN_ACCEPTANCE_SCHEMA,
            "chain_count": _CHAIN_COUNT,
            "chain_weighting": "equal_chain_means",
            "overall_band": self.overall_band,
            "per_chain_band": self.per_chain_band,
            "boundaries": "inclusive",
            "negative_infinity_log_acceptance": "probability_zero",
            "nan_or_positive_infinity_log_acceptance": "invalid",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FourChainMeanBandAcceptancePolicy":
        if payload.get("schema") != FOUR_CHAIN_ACCEPTANCE_SCHEMA:
            raise ValueError("unsupported four-chain acceptance policy schema")
        return cls(
            overall_band=tuple(payload["overall_band"]),
            per_chain_band=tuple(payload["per_chain_band"]),
        )

    def evaluate(
        self,
        log_accept_ratio: Any,
        *,
        divergence_count: Any,
    ) -> "FourChainAcceptanceDecision":
        values = tf.cast(tf.convert_to_tensor(log_accept_ratio), _DTYPE)
        if values.shape.rank is not None and values.shape.rank != 2:
            raise ValueError("log_accept_ratio must have shape [draw, 4]")
        if values.shape.rank == 2 and values.shape[1] not in {None, _CHAIN_COUNT}:
            raise ValueError("log_accept_ratio must have shape [draw, 4]")
        if (
            values.shape.rank == 2
            and values.shape[0] is not None
            and int(values.shape[0]) <= 0
        ):
            raise ValueError("log_accept_ratio must contain at least one draw")
        divergence = tf.cast(tf.convert_to_tensor(divergence_count), tf.int32)
        assertions = (
            tf.debugging.assert_rank(
                values, 2, message="log_accept_ratio must have shape [draw, 4]"
            ),
            tf.debugging.assert_equal(
                tf.shape(values)[1],
                _CHAIN_COUNT,
                message="log_accept_ratio must contain exactly four chains",
            ),
            tf.debugging.assert_positive(
                tf.shape(values)[0],
                message="log_accept_ratio must contain at least one draw",
            ),
            tf.debugging.assert_rank(
                divergence, 0, message="divergence_count must be scalar"
            ),
            tf.debugging.assert_greater_equal(
                divergence,
                tf.constant(0, tf.int32),
                message="divergence_count must be non-negative",
            ),
        )
        with tf.control_dependencies(assertions):
            values = tf.identity(values)
            divergence = tf.identity(divergence)
        defined = tf.logical_and(
            tf.logical_not(tf.math.is_nan(values)),
            tf.logical_not(tf.logical_and(tf.math.is_inf(values), values > 0.0)),
        )
        safe_values = tf.where(
            defined,
            values,
            tf.fill(tf.shape(values), tf.constant(float("-inf"), _DTYPE)),
        )
        probabilities = tf.exp(tf.minimum(safe_values, tf.zeros_like(safe_values)))
        chain_means = tf.reduce_mean(probabilities, axis=0)
        overall_mean = tf.reduce_mean(chain_means)
        overall_lower, overall_upper = (
            tf.constant(item, _DTYPE) for item in self.overall_band
        )
        chain_lower, chain_upper = (
            tf.constant(item, _DTYPE) for item in self.per_chain_band
        )
        negative_infinity = tf.constant(float("-inf"), _DTYPE)
        positive_infinity = tf.constant(float("inf"), _DTYPE)
        # The policy receives log ratios. One representable step preserves an
        # inclusive decimal boundary after exp/log and mean roundoff.
        overall_lower_inclusive = tf.math.nextafter(
            overall_lower, negative_infinity
        )
        overall_upper_inclusive = tf.math.nextafter(
            overall_upper, positive_infinity
        )
        chain_lower_inclusive = tf.math.nextafter(chain_lower, negative_infinity)
        chain_upper_inclusive = tf.math.nextafter(chain_upper, positive_infinity)
        overall_pass = tf.logical_and(
            overall_mean >= overall_lower_inclusive,
            overall_mean <= overall_upper_inclusive,
        )
        chain_passes = tf.logical_and(
            chain_means >= chain_lower_inclusive,
            chain_means <= chain_upper_inclusive,
        )
        defined_pass = tf.reduce_all(defined)
        divergence_pass = tf.equal(divergence, tf.constant(0, tf.int32))
        passed = tf.logical_and(
            defined_pass,
            tf.logical_and(
                divergence_pass,
                tf.logical_and(overall_pass, tf.reduce_all(chain_passes)),
            ),
        )
        any_high = tf.logical_or(
            overall_mean > overall_upper_inclusive,
            tf.reduce_any(chain_means > chain_upper_inclusive),
        )
        any_low = tf.logical_or(
            overall_mean < overall_lower_inclusive,
            tf.reduce_any(chain_means < chain_lower_inclusive),
        )
        direction = tf.where(
            tf.logical_and(any_high, tf.logical_not(any_low)),
            tf.constant(1, tf.int32),
            tf.where(
                tf.logical_and(any_low, tf.logical_not(any_high)),
                tf.constant(-1, tf.int32),
                tf.constant(0, tf.int32),
            ),
        )
        return FourChainAcceptanceDecision(
            chain_means=chain_means,
            overall_mean=overall_mean,
            log_acceptance_defined=defined_pass,
            overall_band_pass=overall_pass,
            per_chain_band_passes=chain_passes,
            divergence_count=divergence,
            divergence_pass=divergence_pass,
            passed=passed,
            repair_direction=direction,
        )


class FourChainAcceptanceDecision(NamedTuple):
    chain_means: tf.Tensor
    overall_mean: tf.Tensor
    log_acceptance_defined: tf.Tensor
    overall_band_pass: tf.Tensor
    per_chain_band_passes: tf.Tensor
    divergence_count: tf.Tensor
    divergence_pass: tf.Tensor
    passed: tf.Tensor
    repair_direction: tf.Tensor


@dataclass(frozen=True)
class TensorFlowHMCKernelTuningConfig:
    """Explicit-budget config for non-promoting TensorFlow tuning diagnostics."""

    parameter_dimension: int
    evidence_role: str
    mass_window_results: tuple[int, ...]
    step_adaptation_results: int
    verification_results: int
    max_leapfrog_steps: int
    initial_step_size: float
    budget_provenance: str
    initial_step_size_provenance: str
    geometry_provenance: str
    target_scope: str
    acceptance_policy: FourChainMeanBandAcceptancePolicy
    target_accept_prob: float
    verification_repair_rounds: int
    step_repair_factor: float
    mass_shrinkage: float
    covariance_jitter: float
    eigenvalue_floor: float
    max_condition_number: float
    seed: tuple[int, int]
    chain_execution_mode: str = "tf_function"
    target_status_trace_policy: str = "none"
    use_xla: bool = False

    def __post_init__(self) -> None:
        dimension = _positive_int(self.parameter_dimension, "parameter_dimension")
        role = str(self.evidence_role)
        if role not in {"diagnostic_only", "diagnostic_candidate_screen"}:
            raise ValueError(
                "evidence_role must be diagnostic_only or diagnostic_candidate_screen"
            )
        windows = tuple(
            _positive_int(value, "mass_window_results")
            for value in self.mass_window_results
        )
        if not windows:
            raise ValueError("mass_window_results must contain at least one window")
        if (
            role == "diagnostic_candidate_screen"
            and min(windows) * _CHAIN_COUNT < dimension + 1
        ):
            raise ValueError(
                "candidate dense covariance requires at least d + 1 states in every window"
            )
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        shrinkage = float(self.mass_shrinkage)
        if not math.isfinite(shrinkage) or not 0.0 < shrinkage <= 1.0:
            raise ValueError("mass_shrinkage must be finite and in (0, 1]")
        jitter = float(self.covariance_jitter)
        floor = float(self.eigenvalue_floor)
        condition = float(self.max_condition_number)
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("covariance_jitter must be finite and non-negative")
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("eigenvalue_floor must be positive and finite")
        if not math.isfinite(condition) or condition <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than one")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        if self.chain_execution_mode != "tf_function":
            raise ValueError("TensorFlow tuning requires chain_execution_mode='tf_function'")
        if self.target_status_trace_policy != "none":
            raise ValueError("TensorFlow bound tuning currently requires target status none")
        if self.use_xla:
            raise ValueError("TensorFlow bound tuning has not qualified XLA")
        if not isinstance(self.acceptance_policy, FourChainMeanBandAcceptancePolicy):
            raise TypeError(
                "acceptance_policy must be FourChainMeanBandAcceptancePolicy"
            )
        object.__setattr__(self, "parameter_dimension", dimension)
        object.__setattr__(self, "evidence_role", role)
        object.__setattr__(self, "mass_window_results", windows)
        object.__setattr__(
            self,
            "step_adaptation_results",
            _positive_int(self.step_adaptation_results, "step_adaptation_results"),
        )
        object.__setattr__(
            self,
            "verification_results",
            _positive_int(self.verification_results, "verification_results"),
        )
        object.__setattr__(
            self,
            "max_leapfrog_steps",
            _positive_int(self.max_leapfrog_steps, "max_leapfrog_steps"),
        )
        object.__setattr__(
            self,
            "verification_repair_rounds",
            _nonnegative_int(
                self.verification_repair_rounds, "verification_repair_rounds"
            ),
        )
        object.__setattr__(
            self,
            "initial_step_size",
            _positive_finite(self.initial_step_size, "initial_step_size"),
        )
        object.__setattr__(
            self,
            "step_repair_factor",
            _positive_finite(self.step_repair_factor, "step_repair_factor"),
        )
        if self.step_repair_factor <= 1.0:
            raise ValueError("step_repair_factor must be greater than one")
        object.__setattr__(self, "target_accept_prob", target)
        object.__setattr__(self, "mass_shrinkage", shrinkage)
        object.__setattr__(self, "covariance_jitter", jitter)
        object.__setattr__(self, "eigenvalue_floor", floor)
        object.__setattr__(self, "max_condition_number", condition)
        object.__setattr__(self, "seed", seed)
        for name in (
            "budget_provenance",
            "initial_step_size_provenance",
            "geometry_provenance",
            "target_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))

    @property
    def chain_count(self) -> int:
        return _CHAIN_COUNT

    @property
    def trajectory_candidates(self) -> tuple[int, ...]:
        values = [1]
        while values[-1] * 2 <= self.max_leapfrog_steps:
            values.append(values[-1] * 2)
        if values[-1] != self.max_leapfrog_steps:
            values.append(self.max_leapfrog_steps)
        return tuple(values)

    @property
    def metric_state_count_by_window(self) -> tuple[int, ...]:
        return tuple(_CHAIN_COUNT * value for value in self.mass_window_results)

    @property
    def metric_rank_eligible(self) -> bool:
        return min(self.metric_state_count_by_window) >= self.parameter_dimension + 1

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": TENSORFLOW_HMC_TUNING_SCHEMA,
            "numerical_backend": "tensorflow_only",
            "dtype": _DTYPE.name,
            "parameter_dimension": self.parameter_dimension,
            "chain_count": self.chain_count,
            "initial_chain_state_policy": (
                "four_identical_zero_states_in_current_affine_coordinates"
            ),
            "mass_window_leapfrog_steps": 1,
            "evidence_role": self.evidence_role,
            "mass_window_results": self.mass_window_results,
            "metric_state_count_by_window": self.metric_state_count_by_window,
            "metric_rank_eligible": self.metric_rank_eligible,
            "step_adaptation_results": self.step_adaptation_results,
            "verification_results": self.verification_results,
            "max_leapfrog_steps": self.max_leapfrog_steps,
            "trajectory_candidates": self.trajectory_candidates,
            "trajectory_candidate_policy": "powers_of_two_then_explicit_cap",
            "seed_derivation_policy": (
                "stateless_fold_in_mass_100_plus_window_candidate_1000_plus_index"
            ),
            "dual_averaging_internal_policy": (
                "tensorflow_probability_defaults_except_explicit_adaptation_steps_"
                "and_target_accept_prob"
            ),
            "initial_step_size": self.initial_step_size,
            "budget_provenance": self.budget_provenance,
            "initial_step_size_provenance": self.initial_step_size_provenance,
            "geometry_provenance": self.geometry_provenance,
            "target_scope": self.target_scope,
            "acceptance_policy": self.acceptance_policy.payload(),
            "target_accept_prob": self.target_accept_prob,
            "verification_repair_rounds": self.verification_repair_rounds,
            "step_repair_factor": self.step_repair_factor,
            "mass_shrinkage": self.mass_shrinkage,
            "covariance_jitter": self.covariance_jitter,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "seed": self.seed,
            "chain_execution_mode": self.chain_execution_mode,
            "target_status_trace_policy": self.target_status_trace_policy,
            "use_xla": self.use_xla,
            "candidate_selection": "first_passing_predeclared_candidate",
            "ranking_claim": False,
            "artifact_authority": False,
            "admission_supported": False,
            "handoff_eligibility": "not_supported",
            "fresh_rhat_verification": "not_implemented",
            "rhat_role": "not_computed",
            "xla_qualification": "not_implemented",
            "xla_mode": "diagnostic_non_xla_only",
            "nonclaims": (
                "the diagnostic screen cannot issue a tuning handoff",
                "fresh tuning R-hat and retained ESS are not computed",
                "the route is not XLA qualified",
                "no retained posterior convergence claim",
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TensorFlowHMCKernelTuningConfig":
        if payload.get("schema") != TENSORFLOW_HMC_TUNING_SCHEMA:
            raise ValueError("unsupported TensorFlow HMC tuning config schema")
        if payload.get("artifact_authority") is not False:
            raise ValueError("TensorFlow diagnostic config cannot claim artifact authority")
        if payload.get("admission_supported") is not False:
            raise ValueError("TensorFlow diagnostic config cannot claim admission")
        if payload.get("handoff_eligibility") != "not_supported":
            raise ValueError("TensorFlow config has inconsistent handoff eligibility")
        expected_diagnostic_fields = {
            "numerical_backend": "tensorflow_only",
            "dtype": _DTYPE.name,
            "chain_count": _CHAIN_COUNT,
            "initial_chain_state_policy": (
                "four_identical_zero_states_in_current_affine_coordinates"
            ),
            "mass_window_leapfrog_steps": 1,
            "trajectory_candidate_policy": "powers_of_two_then_explicit_cap",
            "seed_derivation_policy": (
                "stateless_fold_in_mass_100_plus_window_candidate_1000_plus_index"
            ),
            "dual_averaging_internal_policy": (
                "tensorflow_probability_defaults_except_explicit_adaptation_steps_"
                "and_target_accept_prob"
            ),
            "fresh_rhat_verification": "not_implemented",
            "rhat_role": "not_computed",
            "xla_qualification": "not_implemented",
            "xla_mode": "diagnostic_non_xla_only",
        }
        for name, expected in expected_diagnostic_fields.items():
            if payload.get(name) != expected:
                raise ValueError(f"TensorFlow diagnostic config has invalid {name}")
        return cls(
            parameter_dimension=payload["parameter_dimension"],
            evidence_role=payload["evidence_role"],
            mass_window_results=tuple(payload["mass_window_results"]),
            step_adaptation_results=payload["step_adaptation_results"],
            verification_results=payload["verification_results"],
            max_leapfrog_steps=payload["max_leapfrog_steps"],
            initial_step_size=payload["initial_step_size"],
            budget_provenance=payload["budget_provenance"],
            initial_step_size_provenance=payload[
                "initial_step_size_provenance"
            ],
            geometry_provenance=payload["geometry_provenance"],
            target_scope=payload["target_scope"],
            acceptance_policy=FourChainMeanBandAcceptancePolicy.from_payload(
                payload["acceptance_policy"]
            ),
            target_accept_prob=payload["target_accept_prob"],
            verification_repair_rounds=payload["verification_repair_rounds"],
            step_repair_factor=payload["step_repair_factor"],
            mass_shrinkage=payload["mass_shrinkage"],
            covariance_jitter=payload["covariance_jitter"],
            eigenvalue_floor=payload["eigenvalue_floor"],
            max_condition_number=payload["max_condition_number"],
            seed=tuple(payload["seed"]),
            chain_execution_mode=payload["chain_execution_mode"],
            target_status_trace_policy=payload["target_status_trace_policy"],
            use_xla=payload["use_xla"],
        )


@dataclass(frozen=True)
class _TensorFlowAffineTransform:
    center: Any
    factor: Any
    log_jacobian_convention: str = "constant_omitted"

    def __post_init__(self) -> None:
        center = tf.cast(tf.convert_to_tensor(self.center), _DTYPE)
        factor = tf.cast(tf.convert_to_tensor(self.factor), _DTYPE)
        if center.shape.rank != 1 or center.shape[0] is None:
            raise ValueError("affine center must have static shape [dimension]")
        dimension = int(center.shape[0])
        if factor.shape != (dimension, dimension):
            raise ValueError("affine factor must have static shape [dimension, dimension]")
        if self.log_jacobian_convention != "constant_omitted":
            raise ValueError("affine transform requires constant_omitted convention")
        with tf.control_dependencies(
            (
                tf.debugging.assert_all_finite(center, "affine center must be finite"),
                tf.debugging.assert_all_finite(factor, "affine factor must be finite"),
            )
        ):
            object.__setattr__(self, "center", tf.identity(center))
            object.__setattr__(self, "factor", tf.identity(factor))


@dataclass(frozen=True)
class _TensorFlowAffineAdapter:
    transform: _TensorFlowAffineTransform
    base_adapter_signature: str
    target_scope: str

    def latent_to_position(self, value: Any) -> tf.Tensor:
        latent = tf.cast(tf.convert_to_tensor(value), _DTYPE)
        return self.transform.center + tf.linalg.matmul(
            latent, self.transform.factor, transpose_b=True
        )

    def position_to_latent(self, value: Any) -> tf.Tensor:
        position = tf.cast(tf.convert_to_tensor(value), _DTYPE)
        delta = position - self.transform.center
        return tf.transpose(
            tf.linalg.triangular_solve(
                self.transform.factor, tf.transpose(delta), lower=True
            )
        )


class _TensorChunk(NamedTuple):
    states: tf.Tensor
    log_accept_ratio: tf.Tensor
    divergence: tf.Tensor
    force_fallback: tf.Tensor
    delta_h: tf.Tensor
    is_accepted: tf.Tensor
    final_state: tf.Tensor
    final_step_size: tf.Tensor


def _run_tensor_steps(
    *,
    kernel: tfp.mcmc.TransitionKernel,
    initial_state: tf.Tensor,
    num_results: int,
    seed: tf.Tensor,
    adaptive: bool,
) -> _TensorChunk:
    results = kernel.bootstrap_results(initial_state)
    states = tf.TensorArray(
        _DTYPE,
        size=num_results,
        element_shape=initial_state.shape,
        clear_after_read=False,
    )
    vectors = tf.TensorShape([_CHAIN_COUNT])
    log_accept = tf.TensorArray(_DTYPE, num_results, element_shape=vectors)
    divergence = tf.TensorArray(tf.bool, num_results, element_shape=vectors)
    fallback = tf.TensorArray(tf.bool, num_results, element_shape=vectors)
    delta_h = tf.TensorArray(_DTYPE, num_results, element_shape=vectors)
    accepted = tf.TensorArray(tf.bool, num_results, element_shape=vectors)

    def cond(index: tf.Tensor, *_: Any) -> tf.Tensor:
        return index < tf.constant(num_results, tf.int32)

    def body(
        index: tf.Tensor,
        state: tf.Tensor,
        kernel_results: Any,
        state_array: tf.TensorArray,
        log_array: tf.TensorArray,
        divergence_array: tf.TensorArray,
        fallback_array: tf.TensorArray,
        delta_array: tf.TensorArray,
        accepted_array: tf.TensorArray,
    ) -> tuple[Any, ...]:
        step_seed = tf.random.experimental.stateless_fold_in(seed, index)
        next_state, next_results = kernel.one_step(
            state, kernel_results, seed=step_seed
        )
        inner = next_results.inner_results if adaptive else next_results
        return (
            index + 1,
            next_state,
            next_results,
            state_array.write(index, next_state),
            log_array.write(index, inner.log_accept_ratio),
            divergence_array.write(index, inner.divergence),
            fallback_array.write(index, inner.force_fallback),
            delta_array.write(index, inner.delta_h),
            accepted_array.write(index, inner.is_accepted),
        )

    result = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, tf.int32),
            initial_state,
            results,
            states,
            log_accept,
            divergence,
            fallback,
            delta_h,
            accepted,
        ),
        parallel_iterations=1,
    )
    final_results = result[2]
    final_step = (
        final_results.new_step_size
        if adaptive
        else final_results.step_size
    )
    final_step = tf.ensure_shape(tf.cast(tf.convert_to_tensor(final_step), _DTYPE), ())
    with tf.control_dependencies(
        (
            tf.debugging.assert_all_finite(
                final_step, "adapted step size must be finite"
            ),
            tf.debugging.assert_positive(
                final_step, message="adapted step size must be positive"
            ),
        )
    ):
        final_step = tf.identity(final_step)
    return _TensorChunk(
        states=result[3].stack(),
        log_accept_ratio=result[4].stack(),
        divergence=result[5].stack(),
        force_fallback=result[6].stack(),
        delta_h=result[7].stack(),
        is_accepted=result[8].stack(),
        final_state=result[1],
        final_step_size=final_step,
    )


def _welford_covariance(states: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.cast(tf.convert_to_tensor(states), _DTYPE)
    if values.shape.rank != 2 or values.shape[0] is None or values.shape[1] is None:
        raise ValueError("Welford states must have static shape [state, dimension]")
    count = int(values.shape[0])
    dimension = int(values.shape[1])
    if count < 2:
        raise ValueError("Welford covariance requires at least two states")
    mean = tf.zeros([dimension], _DTYPE)
    m2 = tf.zeros([dimension, dimension], _DTYPE)

    def cond(index: tf.Tensor, *_: tf.Tensor) -> tf.Tensor:
        return index < tf.constant(count, tf.int32)

    def body(
        index: tf.Tensor, current_mean: tf.Tensor, current_m2: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        row = values[index]
        next_count = tf.cast(index + 1, _DTYPE)
        delta = row - current_mean
        next_mean = current_mean + delta / next_count
        delta2 = row - next_mean
        next_m2 = current_m2 + delta[:, None] * delta2[None, :]
        return index + 1, next_mean, next_m2

    _, mean, m2 = tf.while_loop(
        cond,
        body,
        (tf.constant(0, tf.int32), mean, m2),
        parallel_iterations=1,
    )
    covariance = m2 / tf.cast(count - 1, _DTYPE)
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    return mean, covariance


def _regularize_covariance(
    empirical: tf.Tensor,
    *,
    target: tf.Tensor,
    config: TensorFlowHMCKernelTuningConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    weight = tf.constant(config.mass_shrinkage, _DTYPE)
    covariance = (1.0 - weight) * empirical + weight * target
    dimension = config.parameter_dimension
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    covariance += tf.constant(config.covariance_jitter, _DTYPE) * tf.eye(
        dimension, dtype=_DTYPE
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    largest = tf.maximum(
        tf.reduce_max(eigenvalues), tf.constant(config.eigenvalue_floor, _DTYPE)
    )
    effective_floor = tf.maximum(
        tf.constant(config.eigenvalue_floor, _DTYPE),
        largest / tf.constant(config.max_condition_number, _DTYPE),
    )
    regularized_values = tf.maximum(eigenvalues, effective_floor)
    regularized = tf.matmul(
        eigenvectors * regularized_values[None, :],
        eigenvectors,
        transpose_b=True,
    )
    regularized = 0.5 * (regularized + tf.transpose(regularized))
    with tf.control_dependencies(
        (
            tf.debugging.assert_all_finite(
                regularized, "regularized covariance must be finite"
            ),
            tf.debugging.assert_positive(
                regularized_values,
                message="regularized covariance eigenvalues must be positive",
            ),
        )
    ):
        regularized = tf.identity(regularized)
    return regularized, tf.linalg.cholesky(regularized), regularized_values


def _adaptive_kernel(
    *,
    binding: HMCTuningRunnerBinding,
    adapter: _TensorFlowAffineAdapter,
    step_size: tf.Tensor,
    leapfrog_steps: int,
    adaptation_steps: int,
    config: TensorFlowHMCKernelTuningConfig,
) -> tfp.mcmc.TransitionKernel:
    inner = binding.build_tensor_kernel(
        adapter,
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps,
        target_scope=config.target_scope,
        target_status_trace_policy=config.target_status_trace_policy,
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
    )
    return tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner,
        num_adaptation_steps=adaptation_steps,
        target_accept_prob=tf.constant(config.target_accept_prob, _DTYPE),
    )


def _fixed_kernel(
    *,
    binding: HMCTuningRunnerBinding,
    adapter: _TensorFlowAffineAdapter,
    step_size: tf.Tensor,
    leapfrog_steps: Any,
    config: TensorFlowHMCKernelTuningConfig,
) -> tfp.mcmc.TransitionKernel:
    return binding.build_tensor_kernel(
        adapter,
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps,
        target_scope=config.target_scope,
        target_status_trace_policy=config.target_status_trace_policy,
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
    )


def _phase_seed(root: tf.Tensor, phase: int) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(
        root, tf.constant(phase, tf.int32)
    )


def _candidate_verification(
    *,
    binding: HMCTuningRunnerBinding,
    adapter: _TensorFlowAffineAdapter,
    initial_state: tf.Tensor,
    initial_step: tf.Tensor,
    leapfrog_steps: int,
    config: TensorFlowHMCKernelTuningConfig,
    seed: tf.Tensor,
) -> tuple[_TensorChunk, FourChainAcceptanceDecision]:
    adaptive = _adaptive_kernel(
        binding=binding,
        adapter=adapter,
        step_size=initial_step,
        leapfrog_steps=leapfrog_steps,
        adaptation_steps=config.step_adaptation_results,
        config=config,
    )
    adapted = _run_tensor_steps(
        kernel=adaptive,
        initial_state=initial_state,
        num_results=config.step_adaptation_results,
        seed=_phase_seed(seed, 0),
        adaptive=True,
    )
    step = adapted.final_step_size
    state = adapted.final_state

    def run_verification(
        current_state: tf.Tensor, current_step: tf.Tensor, round_index: int
    ) -> tuple[_TensorChunk, FourChainAcceptanceDecision]:
        fixed = _fixed_kernel(
            binding=binding,
            adapter=adapter,
            step_size=current_step,
            leapfrog_steps=leapfrog_steps,
            config=config,
        )
        chunk = _run_tensor_steps(
            kernel=fixed,
            initial_state=current_state,
            num_results=config.verification_results,
            seed=_phase_seed(seed, round_index + 1),
            adaptive=False,
        )
        decision = config.acceptance_policy.evaluate(
            chunk.log_accept_ratio,
            divergence_count=tf.reduce_sum(tf.cast(chunk.divergence, tf.int32)),
        )
        return chunk, decision

    verified, decision = run_verification(state, step, 0)
    for repair_index in range(config.verification_repair_rounds):
        direction = decision.repair_direction
        factor = tf.constant(config.step_repair_factor, _DTYPE)
        repaired_step = tf.where(
            direction > 0,
            verified.final_step_size * factor,
            tf.where(
                direction < 0,
                verified.final_step_size / factor,
                verified.final_step_size,
            ),
        )
        needs_repair = tf.logical_and(
            tf.logical_not(decision.passed),
            tf.logical_and(
                decision.log_acceptance_defined,
                tf.logical_and(
                    decision.divergence_pass, tf.not_equal(direction, 0)
                ),
            ),
        )

        def repair() -> tuple[_TensorChunk, FourChainAcceptanceDecision]:
            return run_verification(
                verified.final_state, repaired_step, repair_index + 1
            )

        def retain() -> tuple[_TensorChunk, FourChainAcceptanceDecision]:
            return verified, decision

        verified, decision = tf.cond(needs_repair, repair, retain)
    return verified, decision


def _build_tuning_graph(
    config: TensorFlowHMCKernelTuningConfig,
    binding: HMCTuningRunnerBinding,
    adapter_signature: str,
) -> Any:
    dimension = config.parameter_dimension
    root_seed = tf.constant(config.seed, tf.int32)
    candidate_count = len(config.trajectory_candidates)

    @tf.function(
        input_signature=(
            tf.TensorSpec([dimension], _DTYPE),
            tf.TensorSpec([dimension], _DTYPE),
        ),
        autograph=False,
        jit_compile=False,
        reduce_retracing=True,
    )
    def run(initial_position: tf.Tensor, parameter_scales: tf.Tensor) -> Mapping[str, tf.Tensor]:
        with tf.control_dependencies(
            (
                tf.debugging.assert_all_finite(
                    initial_position, "initial_position must be finite"
                ),
                tf.debugging.assert_all_finite(
                    parameter_scales, "parameter_scales must be finite"
                ),
                tf.debugging.assert_positive(
                    parameter_scales, message="parameter_scales must be positive"
                ),
            )
        ):
            center = tf.identity(initial_position)
            scales = tf.identity(parameter_scales)
        initial_covariance = tf.linalg.diag(tf.square(scales))
        covariance = initial_covariance
        factor = tf.linalg.diag(scales)
        state = tf.zeros([_CHAIN_COUNT, dimension], _DTYPE)
        step_size = tf.constant(config.initial_step_size, _DTYPE)
        adaptation_divergence_count = tf.constant(0, tf.int32)
        adaptation_fallback_count = tf.constant(0, tf.int32)
        final_metric_eigenvalues = tf.square(scales)

        for window_index, result_count in enumerate(config.mass_window_results):
            adapter = _TensorFlowAffineAdapter(
                transform=_TensorFlowAffineTransform(center=center, factor=factor),
                base_adapter_signature=adapter_signature,
                target_scope=config.target_scope,
            )
            kernel = _adaptive_kernel(
                binding=binding,
                adapter=adapter,
                step_size=step_size,
                leapfrog_steps=1,
                adaptation_steps=result_count,
                config=config,
            )
            chunk = _run_tensor_steps(
                kernel=kernel,
                initial_state=state,
                num_results=result_count,
                seed=_phase_seed(root_seed, 100 + window_index),
                adaptive=True,
            )
            raw_states = adapter.latent_to_position(chunk.states)
            raw_flat = tf.reshape(raw_states, [result_count * _CHAIN_COUNT, dimension])
            next_center, empirical = _welford_covariance(raw_flat)
            next_covariance, next_factor, final_metric_eigenvalues = (
                _regularize_covariance(
                    empirical,
                    target=initial_covariance,
                    config=config,
                )
            )
            raw_final = adapter.latent_to_position(chunk.final_state)
            next_adapter = _TensorFlowAffineAdapter(
                transform=_TensorFlowAffineTransform(
                    center=next_center, factor=next_factor
                ),
                base_adapter_signature=adapter_signature,
                target_scope=config.target_scope,
            )
            state = next_adapter.position_to_latent(raw_final)
            center = next_center
            covariance = next_covariance
            factor = next_factor
            step_size = chunk.final_step_size
            adaptation_divergence_count += tf.reduce_sum(
                tf.cast(chunk.divergence, tf.int32)
            )
            adaptation_fallback_count += tf.reduce_sum(
                tf.cast(chunk.force_fallback, tf.int32)
            )

        adapter = _TensorFlowAffineAdapter(
            transform=_TensorFlowAffineTransform(center=center, factor=factor),
            base_adapter_signature=adapter_signature,
            target_scope=config.target_scope,
        )
        candidate_selected = tf.constant(False)
        selected_index = tf.constant(-1, tf.int32)
        reported_index = tf.constant(-1, tf.int32)
        reported_l = tf.constant(1, tf.int32)
        reported_step = step_size
        reported_state = state
        reported_log_accept = tf.fill(
            [config.verification_results, _CHAIN_COUNT],
            tf.constant(float("nan"), _DTYPE),
        )
        reported_divergence_count = tf.constant(0, tf.int32)
        reported_fallback_count = tf.constant(0, tf.int32)
        candidate_summary = tf.TensorArray(
            _DTYPE,
            candidate_count,
            element_shape=tf.TensorShape([7]),
            clear_after_read=False,
        )

        for candidate_index, leapfrog_steps in enumerate(config.trajectory_candidates):
            def evaluate_candidate() -> tuple[Any, ...]:
                chunk, decision = _candidate_verification(
                    binding=binding,
                    adapter=adapter,
                    initial_state=state,
                    initial_step=step_size,
                    leapfrog_steps=leapfrog_steps,
                    config=config,
                    seed=_phase_seed(root_seed, 1000 + candidate_index),
                )
                divergence_count = tf.reduce_sum(
                    tf.cast(chunk.divergence, tf.int32)
                )
                fallback_count = tf.reduce_sum(
                    tf.cast(chunk.force_fallback, tf.int32)
                )
                return (
                    tf.constant(True),
                    chunk.final_state,
                    chunk.final_step_size,
                    chunk.log_accept_ratio,
                    divergence_count,
                    fallback_count,
                    decision.passed,
                    decision.overall_mean,
                )

            def skip_candidate() -> tuple[Any, ...]:
                return (
                    tf.constant(False),
                    state,
                    step_size,
                    tf.zeros([config.verification_results, _CHAIN_COUNT], _DTYPE),
                    tf.constant(0, tf.int32),
                    tf.constant(0, tf.int32),
                    tf.constant(False),
                    tf.constant(float("nan"), _DTYPE),
                )

            (
                evaluated,
                candidate_state,
                candidate_step,
                candidate_log_accept,
                candidate_divergence,
                candidate_fallback,
                candidate_passed,
                candidate_overall_mean,
            ) = tf.cond(candidate_selected, skip_candidate, evaluate_candidate)
            choose = tf.logical_and(
                tf.logical_not(candidate_selected),
                tf.logical_and(evaluated, candidate_passed),
            )
            selected_index = tf.where(
                choose, tf.constant(candidate_index, tf.int32), selected_index
            )
            reported_index = tf.where(
                evaluated, tf.constant(candidate_index, tf.int32), reported_index
            )
            reported_l = tf.where(
                evaluated, tf.constant(leapfrog_steps, tf.int32), reported_l
            )
            reported_step = tf.where(evaluated, candidate_step, reported_step)
            reported_state = tf.where(evaluated, candidate_state, reported_state)
            reported_log_accept = tf.where(
                evaluated, candidate_log_accept, reported_log_accept
            )
            reported_divergence_count = tf.where(
                evaluated, candidate_divergence, reported_divergence_count
            )
            reported_fallback_count = tf.where(
                evaluated, candidate_fallback, reported_fallback_count
            )
            candidate_selected = tf.logical_or(candidate_selected, choose)
            candidate_summary = candidate_summary.write(
                candidate_index,
                tf.stack(
                    (
                        tf.cast(evaluated, _DTYPE),
                        tf.cast(leapfrog_steps, _DTYPE),
                        candidate_step,
                        candidate_overall_mean,
                        tf.cast(candidate_divergence, _DTYPE),
                        tf.cast(candidate_fallback, _DTYPE),
                        tf.cast(candidate_passed, _DTYPE),
                    )
                ),
            )

        health = config.acceptance_policy.evaluate(
            reported_log_accept,
            divergence_count=reported_divergence_count,
        )
        metric_rank_eligible = tf.constant(config.metric_rank_eligible)
        metric_update_count = tf.constant(len(config.mass_window_results), tf.int32)
        metric_update_valid = tf.logical_and(
            tf.reduce_all(tf.math.is_finite(covariance)),
            tf.logical_and(
                tf.reduce_all(tf.math.is_finite(factor)),
                tf.reduce_all(final_metric_eigenvalues > 0.0),
            ),
        )
        candidate_role = tf.constant(
            config.evidence_role == "diagnostic_candidate_screen"
        )
        heuristic_screen_passed = tf.logical_and(
            candidate_selected,
            tf.logical_and(
                health.passed,
                tf.logical_and(
                    metric_rank_eligible,
                    tf.logical_and(candidate_role, metric_update_valid),
                ),
            ),
        )
        # This graph does not implement the ordinary fresh-R-hat admission gate
        # or XLA qualification. The heuristic remains useful diagnostic output,
        # but it cannot become a public-tuner handoff condition.
        handoff_eligible = tf.constant(False)
        passed = tf.constant(False)
        final_raw_state = adapter.latent_to_position(reported_state)
        health_summary = tf.concat(
            (
                health.chain_means,
                tf.reshape(health.overall_mean, [1]),
                tf.cast(
                    tf.stack(
                        (
                            health.log_acceptance_defined,
                            health.overall_band_pass,
                            tf.reduce_all(health.per_chain_band_passes),
                            health.divergence_pass,
                            health.passed,
                            metric_rank_eligible,
                            metric_update_valid,
                            candidate_role,
                            candidate_selected,
                            heuristic_screen_passed,
                            handoff_eligible,
                            passed,
                        )
                    ),
                    _DTYPE,
                ),
            ),
            axis=0,
        )
        return {
            "initial_position": initial_position,
            "parameter_scales": parameter_scales,
            "center": center,
            "factor": factor,
            "covariance": covariance,
            "metric_eigenvalues": final_metric_eigenvalues,
            "final_chain_state": reported_state,
            "final_raw_state": final_raw_state,
            "step_size": reported_step,
            "num_leapfrog_steps": reported_l,
            "selected_candidate_index": selected_index,
            "reported_candidate_index": reported_index,
            "candidate_selected": candidate_selected,
            "verification_log_accept_ratio": reported_log_accept,
            "verification_divergence_count": reported_divergence_count,
            "verification_force_fallback_count": reported_fallback_count,
            "adaptation_divergence_count": adaptation_divergence_count,
            "adaptation_force_fallback_count": adaptation_fallback_count,
            "candidate_summary": candidate_summary.stack(),
            "health_summary": health_summary,
            "metric_rank_eligible": metric_rank_eligible,
            "metric_update_count": metric_update_count,
            "metric_update_valid": metric_update_valid,
            "health_passed": health.passed,
            "heuristic_screen_passed": heuristic_screen_passed,
            "handoff_eligible": handoff_eligible,
            "passed": passed,
        }

    return run


@dataclass(frozen=True)
class TensorFlowHMCKernelTuningResult:
    config: TensorFlowHMCKernelTuningConfig
    adapter_signature: str
    binding_payload: Mapping[str, Any]
    runner_binding: HMCTuningRunnerBinding = field(repr=False, compare=False)
    initial_position: tf.Tensor
    parameter_scales: tf.Tensor
    center: tf.Tensor
    factor: tf.Tensor
    covariance: tf.Tensor
    metric_eigenvalues: tf.Tensor
    final_chain_state: tf.Tensor
    final_raw_state: tf.Tensor
    step_size: tf.Tensor
    num_leapfrog_steps: tf.Tensor
    selected_candidate_index: tf.Tensor
    reported_candidate_index: tf.Tensor
    candidate_selected: tf.Tensor
    verification_log_accept_ratio: tf.Tensor
    verification_divergence_count: tf.Tensor
    verification_force_fallback_count: tf.Tensor
    adaptation_divergence_count: tf.Tensor
    adaptation_force_fallback_count: tf.Tensor
    candidate_summary: tf.Tensor
    health_summary: tf.Tensor
    metric_rank_eligible: tf.Tensor
    metric_update_count: tf.Tensor
    metric_update_valid: tf.Tensor
    health_passed: tf.Tensor
    heuristic_screen_passed: tf.Tensor
    handoff_eligible: tf.Tensor
    passed: tf.Tensor
    graph_function: Any = field(repr=False, compare=False, default=None)
    artifact_manifest_path: str | None = None

    @property
    def posterior_admission_authority(self) -> bool:
        return False

    @property
    def admission_supported(self) -> bool:
        return False

    @property
    def acceptance_decision(self) -> FourChainAcceptanceDecision:
        return self.config.acceptance_policy.evaluate(
            self.verification_log_accept_ratio,
            divergence_count=self.verification_divergence_count,
        )


_TUNING_TENSOR_FIELDS = (
    "initial_position",
    "parameter_scales",
    "center",
    "factor",
    "covariance",
    "metric_eigenvalues",
    "final_chain_state",
    "final_raw_state",
    "step_size",
    "num_leapfrog_steps",
    "selected_candidate_index",
    "reported_candidate_index",
    "candidate_selected",
    "verification_log_accept_ratio",
    "verification_divergence_count",
    "verification_force_fallback_count",
    "adaptation_divergence_count",
    "adaptation_force_fallback_count",
    "candidate_summary",
    "health_summary",
    "metric_rank_eligible",
    "metric_update_count",
    "metric_update_valid",
    "health_passed",
    "heuristic_screen_passed",
    "handoff_eligible",
    "passed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="ascii")
    os.replace(temporary, path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp")
    tf.io.write_file(str(temporary), tf.io.serialize_tensor(tf.convert_to_tensor(value)))
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "serialization": "tf.io.serialize_tensor",
    }


def _prepare_output_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("TensorFlow HMC output directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_tuning_artifact(
    output_dir: str | Path,
    result: TensorFlowHMCKernelTuningResult,
) -> str:
    root = _prepare_output_dir(output_dir)
    tensors = {}
    for name in _TUNING_TENSOR_FIELDS:
        tensors[name] = _write_tensor(root / f"{name}.tfbin", getattr(result, name))
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": TENSORFLOW_HMC_TUNING_SCHEMA,
            "artifact_role": result.config.evidence_role,
            "config": result.config.payload(),
            "adapter_signature": result.adapter_signature,
            "runner_binding": dict(result.binding_payload),
            "tensors": tensors,
            "candidate_summary_fields": (
                "evaluated",
                "num_leapfrog_steps",
                "step_size",
                "overall_acceptance_mean",
                "divergence_count",
                "force_fallback_count",
                "health_passed",
            ),
            "health_summary_fields": (
                "chain_0_acceptance_mean",
                "chain_1_acceptance_mean",
                "chain_2_acceptance_mean",
                "chain_3_acceptance_mean",
                "overall_acceptance_mean",
                "log_acceptance_defined",
                "overall_band_pass",
                "all_per_chain_bands_pass",
                "divergence_pass",
                "health_passed",
                "metric_rank_eligible",
                "metric_update_valid",
                "candidate_role",
                "candidate_selected",
                "heuristic_screen_passed",
                "handoff_eligible",
                "passed",
            ),
            "numerical_backend": "tensorflow_only",
            "arrays_materialized_on_host": False,
            "artifact_authority": False,
            "admission_supported": False,
            "handoff_eligibility": "not_supported",
            "fresh_rhat_verification": "not_implemented",
            "rhat_role": "not_computed",
            "xla_qualification": "not_implemented",
            "xla_mode": "diagnostic_non_xla_only",
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
        },
    )
    return str(manifest_path)


def _read_tensor_record(root: Path, record: Mapping[str, Any], dtype: tf.dtypes.DType) -> tf.Tensor:
    path = root / str(record["path"])
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"missing or hash-mismatched tensor shard: {path.name}")
    return tf.io.parse_tensor(tf.io.read_file(str(path)), out_type=dtype)


def load_tensorflow_hmc_tuning_result(
    manifest_path: str | Path,
    *,
    adapter: Any,
    runner_binding: HMCTuningRunnerBinding,
) -> TensorFlowHMCKernelTuningResult:
    """Reload TensorFlow mechanics while requiring the original typed binding."""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="ascii"))
    if payload.get("schema") != TENSORFLOW_HMC_TUNING_SCHEMA:
        raise ValueError("unsupported TensorFlow HMC tuning artifact")
    if payload.get("artifact_authority") is not False:
        raise ValueError("TensorFlow diagnostic artifact cannot claim authority")
    if payload.get("admission_supported") is not False:
        raise ValueError("TensorFlow diagnostic artifact cannot claim admission")
    if payload.get("handoff_eligibility") != "not_supported":
        raise ValueError("TensorFlow artifact has inconsistent handoff eligibility")
    expected_diagnostic_fields = {
        "fresh_rhat_verification": "not_implemented",
        "rhat_role": "not_computed",
        "xla_qualification": "not_implemented",
        "xla_mode": "diagnostic_non_xla_only",
    }
    for name, expected in expected_diagnostic_fields.items():
        if payload.get(name) != expected:
            raise ValueError(f"TensorFlow diagnostic artifact has invalid {name}")
    config = TensorFlowHMCKernelTuningConfig.from_payload(payload["config"])
    if not isinstance(runner_binding, HMCTuningRunnerBinding):
        raise TypeError("runner_binding must be HMCTuningRunnerBinding")
    if payload["runner_binding"].get("binding_hash") != runner_binding.binding_hash:
        raise ValueError("runner binding hash mismatch")
    adapter_signature = _adapter_signature(adapter)
    if payload.get("adapter_signature") != adapter_signature:
        raise ValueError("adapter signature mismatch")
    root = path.parent
    tensor_payload = payload["tensors"]
    bool_fields = {
        "candidate_selected",
        "metric_rank_eligible",
        "metric_update_valid",
        "health_passed",
        "heuristic_screen_passed",
        "handoff_eligible",
        "passed",
    }
    int_fields = {
        "num_leapfrog_steps",
        "selected_candidate_index",
        "reported_candidate_index",
        "verification_divergence_count",
        "verification_force_fallback_count",
        "adaptation_divergence_count",
        "adaptation_force_fallback_count",
        "metric_update_count",
    }
    values = {}
    for name in _TUNING_TENSOR_FIELDS:
        dtype = tf.bool if name in bool_fields else tf.int32 if name in int_fields else _DTYPE
        values[name] = _read_tensor_record(root, tensor_payload[name], dtype)
    tf.debugging.assert_equal(
        values["handoff_eligible"], False, message="diagnostic cannot issue handoff"
    )
    tf.debugging.assert_equal(
        values["passed"], False, message="diagnostic cannot pass tuning admission"
    )
    return TensorFlowHMCKernelTuningResult(
        config=config,
        adapter_signature=adapter_signature,
        binding_payload=runner_binding.payload(),
        runner_binding=runner_binding,
        graph_function=None,
        artifact_manifest_path=str(path),
        **values,
    )


def _adapter_signature(adapter: Any) -> str:
    value = getattr(adapter, "adapter_signature", None)
    if not callable(value):
        raise TypeError("adapter must expose adapter_signature()")
    return _nonempty(value(), "adapter_signature")


def _run_tensorflow_hmc_tuning(
    *,
    adapter: Any,
    initial_position: Any,
    config: TensorFlowHMCKernelTuningConfig,
    output_dir: str | Path | None,
    parameter_scales: Any,
    runner_binding: HMCTuningRunnerBinding,
) -> TensorFlowHMCKernelTuningResult:
    if not isinstance(runner_binding, HMCTuningRunnerBinding):
        raise TypeError("TensorFlow tuning requires HMCTuningRunnerBinding")
    if parameter_scales is None:
        raise ValueError("TensorFlow tuning requires parameter_scales")
    adapter_signature = _adapter_signature(adapter)
    adapter_scope = str(getattr(adapter, "target_scope", config.target_scope))
    if adapter_scope != config.target_scope:
        raise ValueError("adapter and config target_scope mismatch")
    runner_binding.validate_public_context(
        target_scope=config.target_scope,
        target_status_trace_policy=config.target_status_trace_policy,
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
    )
    position = tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(initial_position), _DTYPE),
        [config.parameter_dimension],
    )
    scales = tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(parameter_scales), _DTYPE),
        [config.parameter_dimension],
    )
    graph = _build_tuning_graph(config, runner_binding, adapter_signature)
    values = graph(position, scales)
    result = TensorFlowHMCKernelTuningResult(
        config=config,
        adapter_signature=adapter_signature,
        binding_payload=runner_binding.payload(),
        runner_binding=runner_binding,
        graph_function=graph,
        **values,
    )
    if output_dir is not None:
        manifest = _write_tuning_artifact(output_dir, result)
        result = TensorFlowHMCKernelTuningResult(
            **{
                **result.__dict__,
                "artifact_manifest_path": manifest,
            }
        )
    return result


@dataclass(frozen=True)
class BoundRetainedHMCArchiveConfig:
    num_results: int
    seed: tuple[int, int]
    output_dir: str | Path
    budget_provenance: str
    continuation_manifest: str | Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_results", _positive_int(self.num_results, "num_results"))
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "budget_provenance", _nonempty(self.budget_provenance, "budget_provenance")
        )
        if self.continuation_manifest is not None:
            object.__setattr__(
                self, "continuation_manifest", Path(self.continuation_manifest)
            )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": BOUND_RETAINED_HMC_ARCHIVE_SCHEMA,
            "num_results": self.num_results,
            "seed": self.seed,
            "budget_provenance": self.budget_provenance,
            "continuation_manifest": (
                None
                if self.continuation_manifest is None
                else str(self.continuation_manifest)
            ),
        }


@dataclass(frozen=True)
class BoundRetainedHMCArchiveResult:
    initial_chain_state: tf.Tensor
    final_raw_state: tf.Tensor
    final_chain_state: tf.Tensor
    chain_acceptance_means: tf.Tensor
    overall_acceptance_mean: tf.Tensor
    divergence_count: tf.Tensor
    force_fallback_count: tf.Tensor
    health_passed: tf.Tensor
    archive_manifest_path: str
    binding_hash: str


def _load_retained_continuation(
    manifest_path: Path,
    *,
    tuning: TensorFlowHMCKernelTuningResult,
    runner_binding: HMCTuningRunnerBinding,
) -> tuple[tf.Tensor, tf.Tensor]:
    if not manifest_path.is_file():
        raise ValueError("retained continuation manifest does not exist")
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    if payload.get("schema") != BOUND_RETAINED_HMC_ARCHIVE_SCHEMA:
        raise ValueError("unsupported retained continuation manifest")
    if payload.get("adapter_signature") != tuning.adapter_signature:
        raise ValueError("retained continuation adapter signature mismatch")
    if payload.get("runner_binding", {}).get("binding_hash") != (
        runner_binding.binding_hash
    ):
        raise ValueError("retained continuation binding hash mismatch")
    if payload.get("same_bound_transition_as_tuning") is not True:
        raise ValueError("retained continuation changed the bound transition")
    if tuning.artifact_manifest_path is None:
        raise ValueError("retained continuation requires a durable tuning manifest")
    previous_tuning = Path(str(payload.get("tuning_manifest", ""))).resolve()
    current_tuning = Path(tuning.artifact_manifest_path).resolve()
    if previous_tuning != current_tuning:
        raise ValueError("retained continuation tuning manifest mismatch")

    tensors = payload.get("tensors", {})
    state_record = tensors.get("final_chain_state.tfbin", {})
    health_record = tensors.get("health_passed.tfbin", {})
    state_path = manifest_path.parent / "final_chain_state.tfbin"
    health_path = manifest_path.parent / "health_passed.tfbin"
    for path, record in (
        (state_path, state_record),
        (health_path, health_record),
    ):
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(
                f"missing or hash-mismatched continuation tensor: {path.name}"
            )
    state = tf.ensure_shape(
        tf.io.parse_tensor(tf.io.read_file(str(state_path)), out_type=_DTYPE),
        [_CHAIN_COUNT, tuning.config.parameter_dimension],
    )
    health = tf.ensure_shape(
        tf.io.parse_tensor(tf.io.read_file(str(health_path)), out_type=tf.bool),
        [],
    )
    tf.debugging.assert_all_finite(state, "retained continuation state")
    return state, health


@dataclass(frozen=True)
class BoundRetainedHMCArchiveRunner:
    tuning_result: TensorFlowHMCKernelTuningResult = field(repr=False)
    runner_binding: HMCTuningRunnerBinding = field(repr=False)

    def run(self, config: BoundRetainedHMCArchiveConfig) -> BoundRetainedHMCArchiveResult:
        if not isinstance(config, BoundRetainedHMCArchiveConfig):
            raise TypeError("config must be BoundRetainedHMCArchiveConfig")
        tuning = self.tuning_result
        if not tuning.admission_supported:
            raise ValueError(
                "TensorFlow diagnostic tuning is not admission-capable and cannot run retained sampling"
            )
        tf.debugging.assert_equal(
            tuning.handoff_eligible,
            True,
            message="retained archive requires a handoff-eligible tuning result",
        )
        tf.debugging.assert_equal(
            tuning.passed,
            True,
            message="retained archive requires a passed candidate result",
        )
        initial_state = tuning.final_chain_state
        continuation_health = tf.constant(True)
        if config.continuation_manifest is not None:
            initial_state, continuation_health = _load_retained_continuation(
                config.continuation_manifest,
                tuning=tuning,
                runner_binding=self.runner_binding,
            )
            tf.debugging.assert_equal(
                continuation_health,
                True,
                message="retained continuation requires a passing predecessor",
            )
        initial_state = tf.ensure_shape(
            initial_state, [_CHAIN_COUNT, tuning.config.parameter_dimension]
        )

        root = _prepare_output_dir(config.output_dir)
        sample_path = root / "samples.tfbin"
        log_accept_path = root / "log_accept_ratio.tfbin"
        divergence_path = root / "divergence.tfbin"
        fallback_path = root / "force_fallback.tfbin"
        delta_h_path = root / "delta_h.tfbin"
        accepted_path = root / "is_accepted.tfbin"
        initial_chain_path = root / "initial_chain_state.tfbin"
        final_raw_path = root / "final_raw_state.tfbin"
        final_chain_path = root / "final_chain_state.tfbin"
        health_summary_path = root / "health_summary.tfbin"
        health_passed_path = root / "health_passed.tfbin"
        adapter = _TensorFlowAffineAdapter(
            transform=_TensorFlowAffineTransform(
                center=tuning.center, factor=tuning.factor
            ),
            base_adapter_signature=tuning.adapter_signature,
            target_scope=tuning.config.target_scope,
        )
        kernel = _fixed_kernel(
            binding=self.runner_binding,
            adapter=adapter,
            step_size=tuning.step_size,
            leapfrog_steps=tuning.num_leapfrog_steps,
            config=tuning.config,
        )

        @tf.function(input_signature=(), autograph=False, jit_compile=False)
        def execute() -> tuple[tf.Tensor, ...]:
            chunk = _run_tensor_steps(
                kernel=kernel,
                initial_state=tf.identity(initial_state),
                num_results=config.num_results,
                seed=tf.constant(config.seed, tf.int32),
                adaptive=False,
            )
            raw_samples = adapter.latent_to_position(chunk.states)
            final_raw = adapter.latent_to_position(chunk.final_state)
            divergence_count = tf.reduce_sum(tf.cast(chunk.divergence, tf.int32))
            fallback_count = tf.reduce_sum(tf.cast(chunk.force_fallback, tf.int32))
            decision = tuning.config.acceptance_policy.evaluate(
                chunk.log_accept_ratio,
                divergence_count=divergence_count,
            )
            health_summary = tf.concat(
                (
                    decision.chain_means,
                    tf.reshape(decision.overall_mean, [1]),
                    tf.cast(
                        tf.stack(
                            (
                                decision.log_acceptance_defined,
                                decision.overall_band_pass,
                                tf.reduce_all(decision.per_chain_band_passes),
                                decision.divergence_pass,
                                decision.passed,
                            )
                        ),
                        _DTYPE,
                    ),
                ),
                axis=0,
            )
            writes = (
                tf.io.write_file(
                    str(sample_path), tf.io.serialize_tensor(raw_samples)
                ),
                tf.io.write_file(
                    str(log_accept_path),
                    tf.io.serialize_tensor(chunk.log_accept_ratio),
                ),
                tf.io.write_file(
                    str(divergence_path), tf.io.serialize_tensor(chunk.divergence)
                ),
                tf.io.write_file(
                    str(fallback_path),
                    tf.io.serialize_tensor(chunk.force_fallback),
                ),
                tf.io.write_file(
                    str(delta_h_path), tf.io.serialize_tensor(chunk.delta_h)
                ),
                tf.io.write_file(
                    str(accepted_path),
                    tf.io.serialize_tensor(chunk.is_accepted),
                ),
                tf.io.write_file(
                    str(initial_chain_path), tf.io.serialize_tensor(initial_state)
                ),
                tf.io.write_file(
                    str(final_raw_path), tf.io.serialize_tensor(final_raw)
                ),
                tf.io.write_file(
                    str(final_chain_path),
                    tf.io.serialize_tensor(chunk.final_state),
                ),
                tf.io.write_file(
                    str(health_summary_path),
                    tf.io.serialize_tensor(health_summary),
                ),
                tf.io.write_file(
                    str(health_passed_path),
                    tf.io.serialize_tensor(decision.passed),
                ),
            )
            with tf.control_dependencies(writes):
                return (
                    tf.identity(initial_state),
                    tf.identity(final_raw),
                    tf.identity(chunk.final_state),
                    tf.identity(decision.chain_means),
                    tf.identity(decision.overall_mean),
                    tf.identity(divergence_count),
                    tf.identity(fallback_count),
                    tf.identity(decision.passed),
                )

        (
            archived_initial,
            final_raw,
            final_chain,
            chain_means,
            overall_mean,
            divergence_count,
            fallback_count,
            health_passed,
        ) = execute()
        tensor_records = {
            path.name: {"sha256": _sha256(path), "serialization": "tf.io.serialize_tensor"}
            for path in (
                sample_path,
                log_accept_path,
                divergence_path,
                fallback_path,
                delta_h_path,
                accepted_path,
                initial_chain_path,
                final_raw_path,
                final_chain_path,
                health_summary_path,
                health_passed_path,
            )
        }
        manifest_path = root / "manifest.json"
        _write_json(
            manifest_path,
            {
                "schema": BOUND_RETAINED_HMC_ARCHIVE_SCHEMA,
                "config": config.payload(),
                "tuning_manifest": tuning.artifact_manifest_path,
                "adapter_signature": tuning.adapter_signature,
                "runner_binding": dict(self.runner_binding.payload()),
                "tensors": tensor_records,
                "same_bound_transition_as_tuning": True,
                "initial_state_source": (
                    "tuning_final_chain_state"
                    if config.continuation_manifest is None
                    else "retained_archive_final_chain_state"
                ),
                "health_summary_fields": (
                    "chain_0_acceptance_mean",
                    "chain_1_acceptance_mean",
                    "chain_2_acceptance_mean",
                    "chain_3_acceptance_mean",
                    "overall_acceptance_mean",
                    "log_acceptance_defined",
                    "overall_band_pass",
                    "all_per_chain_bands_pass",
                    "divergence_pass",
                    "health_passed",
                ),
                "returned_sample_tensor": False,
                "numerical_backend": "tensorflow_only",
                "reports_posterior_convergence": False,
            },
        )
        return BoundRetainedHMCArchiveResult(
            initial_chain_state=archived_initial,
            final_raw_state=final_raw,
            final_chain_state=final_chain,
            chain_acceptance_means=chain_means,
            overall_acceptance_mean=overall_mean,
            divergence_count=divergence_count,
            force_fallback_count=fallback_count,
            health_passed=health_passed,
            archive_manifest_path=str(manifest_path),
            binding_hash=self.runner_binding.binding_hash,
        )


def build_retained_bound_hmc_archive_runner_from_tuning_result(
    *,
    tuning_result: TensorFlowHMCKernelTuningResult,
    runner_binding: HMCTuningRunnerBinding,
) -> BoundRetainedHMCArchiveRunner:
    if not isinstance(tuning_result, TensorFlowHMCKernelTuningResult):
        raise TypeError("tuning_result must be TensorFlowHMCKernelTuningResult")
    if not isinstance(runner_binding, HMCTuningRunnerBinding):
        raise TypeError("runner_binding must be HMCTuningRunnerBinding")
    if not tuning_result.admission_supported:
        raise ValueError(
            "TensorFlow diagnostic tuning is not admission-capable and cannot build a retained runner"
        )
    if tuning_result.config.evidence_role != "diagnostic_candidate_screen":
        raise ValueError("retained sampling requires a diagnostic candidate screen")
    if tuning_result.runner_binding is not runner_binding:
        raise ValueError("retained sampling requires the exact loaded runner binding")
    if tuning_result.binding_payload.get("binding_hash") != runner_binding.binding_hash:
        raise ValueError("retained sampling runner binding hash mismatch")
    if tuning_result.artifact_manifest_path is None or not Path(
        tuning_result.artifact_manifest_path
    ).is_file():
        raise ValueError("retained sampling requires a durable tuning manifest")
    return BoundRetainedHMCArchiveRunner(
        tuning_result=tuning_result,
        runner_binding=runner_binding,
    )


__all__ = [
    "BOUND_RETAINED_HMC_ARCHIVE_SCHEMA",
    "BoundRetainedHMCArchiveConfig",
    "BoundRetainedHMCArchiveResult",
    "BoundRetainedHMCArchiveRunner",
    "FOUR_CHAIN_ACCEPTANCE_SCHEMA",
    "FourChainAcceptanceDecision",
    "FourChainMeanBandAcceptancePolicy",
    "TENSORFLOW_HMC_TUNING_SCHEMA",
    "TensorFlowHMCKernelTuningConfig",
    "TensorFlowHMCKernelTuningResult",
    "build_retained_bound_hmc_archive_runner_from_tuning_result",
    "load_tensorflow_hmc_tuning_result",
]
