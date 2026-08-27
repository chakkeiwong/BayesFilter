"""Exact true-energy-corrected HMC with a frozen position-only neural force.

The learned force is used only inside a fixed symmetric kick-drift-kick map.
Metropolis correction always uses the declared deterministic target potential
and both endpoint kinetic energies.  The active transition and chain paths are
TensorFlow-native and use ``tf.while_loop`` for leapfrog and sample loops.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.tuning_contract import (
    HMCTuningRunnerBinding,
    _issue_hmc_tuning_runner_binding,
)


NEURAL_FORCE_HMC_SCHEMA = "bayesfilter.neural_force_hmc.v1"
POSITION_ONLY_FORCE_SEMANTICS = "position_only_scalar_potential_gradient"
DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS = (
    "deterministic_position_only_proposal_field"
)
_POSITION_ONLY_FORCE_SEMANTICS = {
    POSITION_ONLY_FORCE_SEMANTICS,
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
}
# A finite log-probability representation for a numerically invalid proposal
# that is forced to reject.  This is a representation of the decision, not an
# energy threshold and is never used to define divergence.
NUMERICAL_DIVERGENCE_LOG_ACCEPTANCE = -1.0e30


class InvalidNeuralForceHMCConfiguration(ValueError):
    """Raised when a binding is outside the proved neural-force HMC route."""


def _require_nonempty(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise InvalidNeuralForceHMCConfiguration(f"{name} must be nonempty")
    return text


def _require_position_only_callable(function: Callable[..., Any], name: str) -> None:
    if not callable(function):
        raise TypeError(f"{name} must be callable")
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return
    marker = object()
    try:
        signature.bind(marker)
    except TypeError as exc:
        raise InvalidNeuralForceHMCConfiguration(
            f"{name} must accept exactly one position argument; a required second "
            "argument such as momentum is forbidden"
        ) from exc
    try:
        signature.bind(marker, marker)
    except TypeError:
        return
    raise InvalidNeuralForceHMCConfiguration(
        f"{name} accepts a second argument; momentum-dependent APIs are forbidden"
    )


@dataclass(frozen=True)
class FrozenPositionOnlyForce:
    """Binding for one frozen deterministic position-only proposal field."""

    function: Callable[[tf.Tensor], tf.Tensor] = field(repr=False, compare=False)
    identity: str
    semantics: str = POSITION_ONLY_FORCE_SEMANTICS
    frozen: bool = True
    momentum_dependent: bool = False
    direct_state_update: bool = False
    symmetric_schedule: bool = True
    coordinate_system: str = "raw"

    def __post_init__(self) -> None:
        _require_position_only_callable(self.function, "force function")
        object.__setattr__(self, "identity", _require_nonempty(self.identity, "force identity"))
        if self.semantics not in _POSITION_ONLY_FORCE_SEMANTICS:
            raise InvalidNeuralForceHMCConfiguration(
                "force semantics must be a supported frozen position-only semantic"
            )
        if not self.frozen:
            raise InvalidNeuralForceHMCConfiguration("force must be frozen before sampling")
        if self.momentum_dependent:
            raise InvalidNeuralForceHMCConfiguration("momentum-dependent force is forbidden")
        if self.direct_state_update:
            raise InvalidNeuralForceHMCConfiguration("direct neural state updates are forbidden")
        if not self.symmetric_schedule:
            raise InvalidNeuralForceHMCConfiguration("force schedule must be symmetric")
        coordinate_system = _require_nonempty(
            self.coordinate_system, "force coordinate_system"
        )
        if coordinate_system not in {"raw", "transformed"}:
            raise InvalidNeuralForceHMCConfiguration(
                "force coordinate_system must be 'raw' or 'transformed'"
            )
        object.__setattr__(self, "coordinate_system", coordinate_system)


@dataclass(frozen=True)
class FrozenTargetPotential:
    """Binding for the deterministic true endpoint potential.

    ``includes_chart_log_jacobian`` records a caller-established property of
    the model target. The generic ``transformed`` coordinate label may also
    denote BayesFilter's affine mass coordinates, so the label alone cannot
    establish whether a separate model chart Jacobian exists or is required.
    """

    function: Callable[[tf.Tensor], tf.Tensor] = field(repr=False, compare=False)
    identity: str
    coordinate_system: str = "raw"
    includes_chart_log_jacobian: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        _require_position_only_callable(self.function, "target potential function")
        object.__setattr__(self, "identity", _require_nonempty(self.identity, "target identity"))
        if self.coordinate_system not in {"raw", "transformed"}:
            raise InvalidNeuralForceHMCConfiguration(
                "coordinate_system must be 'raw' or 'transformed'"
            )
        if not self.deterministic:
            raise InvalidNeuralForceHMCConfiguration(
                "the P1 kernel requires a deterministic endpoint potential"
            )


@dataclass(frozen=True)
class NeuralForceHMCConfig:
    """Immutable fixed-step configuration for the corrected kernel."""

    step_size: Any
    num_leapfrog_steps: Any
    inverse_mass_diagonal: tuple[float, ...]
    dtype: str = "float64"

    def __post_init__(self) -> None:
        try:
            dtype = tf.as_dtype(self.dtype)
        except TypeError as exc:
            raise InvalidNeuralForceHMCConfiguration(
                "dtype is not a TensorFlow dtype"
            ) from exc
        if tf.is_tensor(self.step_size):
            step_size = tf.convert_to_tensor(self.step_size, dtype=dtype)
            with tf.control_dependencies(
                (
                    tf.debugging.assert_rank(
                        step_size, 0, message="step_size must be scalar"
                    ),
                    tf.debugging.assert_all_finite(
                        step_size, "step_size must be finite"
                    ),
                    tf.debugging.assert_positive(
                        step_size, message="step_size must be positive"
                    ),
                )
            ):
                step_size = tf.identity(step_size)
        else:
            step_size = float(self.step_size)
            if not math.isfinite(step_size) or step_size <= 0.0:
                raise InvalidNeuralForceHMCConfiguration(
                    "step_size must be positive and finite"
                )
        if tf.is_tensor(self.num_leapfrog_steps):
            supplied_steps = tf.convert_to_tensor(self.num_leapfrog_steps)
            if not supplied_steps.dtype.is_integer:
                raise InvalidNeuralForceHMCConfiguration(
                    "num_leapfrog_steps tensor must have an integer dtype"
                )
            steps = tf.cast(supplied_steps, tf.int32)
            with tf.control_dependencies(
                (
                    tf.debugging.assert_rank(
                        steps, 0, message="num_leapfrog_steps must be scalar"
                    ),
                    tf.debugging.assert_positive(
                        steps, message="num_leapfrog_steps must be positive"
                    ),
                )
            ):
                steps = tf.identity(steps)
        else:
            steps = int(self.num_leapfrog_steps)
            if steps < 1 or steps != self.num_leapfrog_steps:
                raise InvalidNeuralForceHMCConfiguration(
                    "num_leapfrog_steps must be a positive integer"
                )
        inverse_mass = tuple(float(value) for value in self.inverse_mass_diagonal)
        if not inverse_mass or any(
            not math.isfinite(value) or value <= 0.0 for value in inverse_mass
        ):
            raise InvalidNeuralForceHMCConfiguration(
                "inverse_mass_diagonal must contain positive finite values"
            )
        if dtype not in {tf.float32, tf.float64}:
            raise InvalidNeuralForceHMCConfiguration("dtype must be float32 or float64")
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "num_leapfrog_steps", steps)
        object.__setattr__(self, "inverse_mass_diagonal", inverse_mass)
        object.__setattr__(self, "dtype", dtype.name)

    @property
    def tf_dtype(self) -> tf.dtypes.DType:
        return tf.as_dtype(self.dtype)


class NeuralForceProposal(NamedTuple):
    position: tf.Tensor
    momentum: tf.Tensor
    force_call_count: tf.Tensor
    divergence: tf.Tensor
    force_fallback: tf.Tensor


class NeuralForceHMCTrace(NamedTuple):
    accepted: tf.Tensor
    log_acceptance_ratio: tf.Tensor
    initial_potential: tf.Tensor
    final_potential: tf.Tensor
    initial_kinetic: tf.Tensor
    final_kinetic: tf.Tensor
    delta_h: tf.Tensor
    finite_status: tf.Tensor
    endpoint_out_of_support: tf.Tensor
    endpoint_call_count: tf.Tensor
    force_call_count: tf.Tensor
    initial_momentum: tf.Tensor
    final_momentum: tf.Tensor
    divergence: tf.Tensor
    force_fallback: tf.Tensor


class NeuralForceHMCTransition(NamedTuple):
    position: tf.Tensor
    potential: tf.Tensor
    trace: NeuralForceHMCTrace


class NeuralForceEndpointKernelResults(NamedTuple):
    target_log_prob: tf.Tensor


class NeuralForceTransitionKernelResults(NamedTuple):
    """TFP-compatible results for endpoint-corrected neural-force HMC."""

    accepted_results: NeuralForceEndpointKernelResults
    is_accepted: tf.Tensor
    log_accept_ratio: tf.Tensor
    proposed_results: NeuralForceEndpointKernelResults
    step_size: tf.Tensor
    delta_h: tf.Tensor
    finite_status: tf.Tensor
    divergence: tf.Tensor
    force_fallback: tf.Tensor


class NeuralForceHMCChain(NamedTuple):
    positions: tf.Tensor
    potentials: tf.Tensor
    accepted: tf.Tensor
    log_acceptance_ratio: tf.Tensor
    initial_potential: tf.Tensor
    final_potential: tf.Tensor
    initial_kinetic: tf.Tensor
    final_kinetic: tf.Tensor
    delta_h: tf.Tensor
    finite_status: tf.Tensor
    endpoint_call_count: tf.Tensor
    force_call_count: tf.Tensor
    num_warmup: tf.Tensor
    divergence: tf.Tensor
    force_fallback: tf.Tensor


def _assert_rank_and_dimension(
    value: tf.Tensor,
    *,
    rank: int,
    dimension: int | None,
    name: str,
) -> tf.Tensor:
    if value.shape.rank is not None and value.shape.rank != rank:
        raise ValueError(f"{name} must have rank {rank}")
    checks = [tf.debugging.assert_rank(value, rank, message=f"{name} rank")]
    if dimension is not None:
        if value.shape.rank == rank and value.shape[-1] not in {None, dimension}:
            raise ValueError(f"{name} final dimension must be {dimension}")
        checks.append(
            tf.debugging.assert_equal(
                tf.shape(value)[-1], dimension, message=f"{name} final dimension"
            )
        )
    with tf.control_dependencies(checks):
        return tf.identity(value)


def _assert_all_finite(value: tf.Tensor, name: str) -> tf.Tensor:
    with tf.control_dependencies([tf.debugging.assert_all_finite(value, name)]):
        return tf.identity(value)


def _sanitize_batch_matrix(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a finite matrix and a per-row arithmetic-invalid flag.

    The custom kernel rejects rows carrying this flag.  Replacing the invalid
    candidate before endpoint evaluation keeps the rejection path finite and
    deterministic, so a numerical failure cannot escape as an undefined
    TensorFlow value.
    """

    finite = tf.reduce_all(tf.math.is_finite(value), axis=-1)
    safe = tf.where(finite[:, tf.newaxis], value, tf.zeros_like(value))
    return safe, tf.logical_not(finite)


def _sanitize_batch_vector(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a finite vector and a per-row arithmetic-invalid flag."""

    finite = tf.math.is_finite(value)
    safe = tf.where(finite, value, tf.zeros_like(value))
    return safe, tf.logical_not(finite)


def _evaluate_force(
    force: FrozenPositionOnlyForce,
    position: tf.Tensor,
    *,
    dimension: int,
    dtype: tf.dtypes.DType,
) -> tuple[tf.Tensor, tf.Tensor]:
    value = tf.convert_to_tensor(force.function(position), dtype=dtype)
    value = _assert_rank_and_dimension(
        value, rank=2, dimension=dimension, name="position-only force"
    )
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(value), tf.shape(position), message="force shape must match position"
            )
        ]
    ):
        # A non-finite reported field is extended by the deterministic zero
        # field.  The flag is telemetry, not a divergence: the proposal remains
        # a finite position-only map and the endpoint target still decides
        # support and acceptance.
        return _sanitize_batch_matrix(tf.identity(value))


def _evaluate_endpoint_potential(
    target: FrozenTargetPotential,
    position: tf.Tensor,
    *,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    value = tf.convert_to_tensor(target.function(position), dtype=dtype)
    value = _assert_rank_and_dimension(value, rank=1, dimension=None, name="target potential")
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(value)[0],
                tf.shape(position)[0],
                message="target potential batch must match position batch",
            )
        ]
    ):
        value = tf.identity(value)
    defined = tf.logical_not(
        tf.logical_or(tf.math.is_nan(value), tf.equal(value, tf.constant(-float("inf"), dtype)))
    )
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.reduce_all(defined),
                True,
                message="target potential must be finite or declared +inf support rejection",
            )
        ]
    ):
        return tf.identity(value)


def kinetic_energy(momentum: tf.Tensor, config: NeuralForceHMCConfig) -> tf.Tensor:
    """Return batched diagonal-mass kinetic energy."""

    momentum = tf.convert_to_tensor(momentum, dtype=config.tf_dtype)
    dimension = len(config.inverse_mass_diagonal)
    momentum = _assert_rank_and_dimension(
        momentum, rank=2, dimension=dimension, name="momentum"
    )
    inverse_mass = tf.constant(config.inverse_mass_diagonal, dtype=config.tf_dtype)
    return 0.5 * tf.reduce_sum(tf.square(momentum) * inverse_mass, axis=-1)


def neural_force_proposal(
    position: tf.Tensor,
    momentum: tf.Tensor,
    force: FrozenPositionOnlyForce,
    config: NeuralForceHMCConfig,
    *,
    step_size: tf.Tensor | float | None = None,
) -> NeuralForceProposal:
    """Execute the fixed symmetric proposal and terminal momentum flip."""

    dtype = config.tf_dtype
    dimension = len(config.inverse_mass_diagonal)
    position = _assert_all_finite(
        _assert_rank_and_dimension(
            tf.convert_to_tensor(position, dtype=dtype),
            rank=2,
            dimension=dimension,
            name="position",
        ),
        "initial position must be finite",
    )
    momentum = _assert_all_finite(
        _assert_rank_and_dimension(
            tf.convert_to_tensor(momentum, dtype=dtype),
            rank=2,
            dimension=dimension,
            name="momentum",
        ),
        "initial momentum must be finite",
    )
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(position), tf.shape(momentum), message="position/momentum shape"
            )
        ]
    ):
        position = tf.identity(position)

    step_size = tf.convert_to_tensor(
        config.step_size if step_size is None else step_size,
        dtype=dtype,
    )
    with tf.control_dependencies(
        [
            tf.debugging.assert_rank(step_size, 0, message="step size must be scalar"),
            tf.debugging.assert_all_finite(step_size, "step size must be finite"),
            tf.debugging.assert_positive(step_size, message="step size must be positive"),
        ]
    ):
        step_size = tf.identity(step_size)
    inverse_mass = tf.constant(config.inverse_mass_diagonal, dtype=dtype)
    initial_force, initial_force_fallback = _evaluate_force(
        force, position, dimension=dimension, dtype=dtype
    )
    momentum, initial_momentum_invalid = _sanitize_batch_matrix(
        momentum - 0.5 * step_size * initial_force
    )
    divergence = tf.identity(initial_momentum_invalid)
    force_fallback = tf.identity(initial_force_fallback)

    def cond(index: tf.Tensor, *_: tf.Tensor) -> tf.Tensor:
        return index < config.num_leapfrog_steps

    def body(
        index: tf.Tensor,
        current_position: tf.Tensor,
        current_momentum: tf.Tensor,
        current_divergence: tf.Tensor,
        current_force_fallback: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        next_position, position_invalid = _sanitize_batch_matrix(
            current_position + step_size * current_momentum * inverse_mass
        )

        def interior_kick() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            force_value, force_invalid = _evaluate_force(
                force, next_position, dimension=dimension, dtype=dtype
            )
            next_momentum, momentum_invalid = _sanitize_batch_matrix(
                current_momentum - step_size * force_value
            )
            return next_momentum, force_invalid, momentum_invalid

        def final_drift() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            return current_momentum, tf.zeros_like(current_force_fallback), tf.zeros_like(
                current_divergence
            )

        next_momentum, force_invalid, momentum_invalid = tf.cond(
            index + 1 < config.num_leapfrog_steps,
            interior_kick,
            final_drift,
        )
        return (
            index + 1,
            next_position,
            next_momentum,
            tf.logical_or(
                current_divergence,
                tf.logical_or(position_invalid, momentum_invalid),
            ),
            tf.logical_or(current_force_fallback, force_invalid),
        )

    _, final_position, final_momentum, divergence, force_fallback = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, tf.int32),
            position,
            momentum,
            divergence,
            force_fallback,
        ),
        parallel_iterations=1,
    )
    final_force, final_force_fallback = _evaluate_force(
        force, final_position, dimension=dimension, dtype=dtype
    )
    final_momentum, final_momentum_invalid = _sanitize_batch_matrix(
        final_momentum - 0.5 * step_size * final_force
    )
    divergence = tf.logical_or(divergence, final_momentum_invalid)
    force_fallback = tf.logical_or(force_fallback, final_force_fallback)
    return NeuralForceProposal(
        position=final_position,
        momentum=-final_momentum,
        force_call_count=(
            tf.cast(config.num_leapfrog_steps, tf.int32) + tf.constant(1, tf.int32)
        ),
        divergence=divergence,
        force_fallback=force_fallback,
    )


def neural_force_hmc_transition(
    position: tf.Tensor,
    current_potential: tf.Tensor,
    force: FrozenPositionOnlyForce,
    target: FrozenTargetPotential,
    config: NeuralForceHMCConfig,
    seed: tf.Tensor | tuple[int, int],
    *,
    step_size: tf.Tensor | float | None = None,
) -> NeuralForceHMCTransition:
    """Run one corrected batched transition with one new endpoint evaluation."""

    dtype = config.tf_dtype
    dimension = len(config.inverse_mass_diagonal)
    position = _assert_all_finite(
        _assert_rank_and_dimension(
            tf.convert_to_tensor(position, dtype=dtype),
            rank=2,
            dimension=dimension,
            name="position",
        ),
        "current position must be finite",
    )
    current_potential = _assert_all_finite(
        _assert_rank_and_dimension(
            tf.convert_to_tensor(current_potential, dtype=dtype),
            rank=1,
            dimension=None,
            name="current potential",
        ),
        "cached current potential must be finite",
    )
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(current_potential)[0],
                tf.shape(position)[0],
                message="cached potential batch must match position batch",
            )
        ]
    ):
        current_potential = tf.identity(current_potential)

    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    seed = _assert_rank_and_dimension(seed, rank=1, dimension=2, name="stateless seed")
    momentum_seed, acceptance_seed = tf.unstack(tf.random.experimental.stateless_split(seed, 2))
    inverse_mass = tf.constant(config.inverse_mass_diagonal, dtype=dtype)
    momentum = tf.random.stateless_normal(
        tf.shape(position), momentum_seed, dtype=dtype
    ) * tf.math.rsqrt(inverse_mass)
    momentum, generated_momentum_invalid = _sanitize_batch_matrix(momentum)
    proposal = neural_force_proposal(
        position,
        momentum,
        force,
        config,
        step_size=step_size,
    )
    proposed_potential = _evaluate_endpoint_potential(
        target, proposal.position, dtype=dtype
    )
    initial_kinetic_raw = kinetic_energy(momentum, config)
    final_kinetic_raw = kinetic_energy(proposal.momentum, config)
    initial_kinetic, initial_kinetic_invalid = _sanitize_batch_vector(
        initial_kinetic_raw
    )
    final_kinetic, final_kinetic_invalid = _sanitize_batch_vector(final_kinetic_raw)
    proposed_potential = tf.identity(proposed_potential)
    endpoint_out_of_support = tf.logical_and(
        tf.math.is_inf(proposed_potential), proposed_potential > 0.0
    )
    raw_delta_h = (
        proposed_potential
        + final_kinetic
        - current_potential
        - initial_kinetic
    )
    delta_nonfinite = tf.logical_and(
        tf.logical_not(endpoint_out_of_support),
        tf.logical_not(tf.math.is_finite(raw_delta_h)),
    )
    divergence = tf.logical_or(
        tf.logical_or(proposal.divergence, generated_momentum_invalid),
        tf.logical_or(
            tf.logical_or(initial_kinetic_invalid, final_kinetic_invalid),
            delta_nonfinite,
        ),
    )
    delta_h = tf.where(
        divergence,
        tf.fill(tf.shape(raw_delta_h), tf.constant(-NUMERICAL_DIVERGENCE_LOG_ACCEPTANCE, dtype)),
        raw_delta_h,
    )
    log_acceptance_ratio = tf.minimum(tf.zeros_like(delta_h), -delta_h)
    log_acceptance_ratio = tf.where(
        divergence,
        tf.fill(
            tf.shape(log_acceptance_ratio),
            tf.constant(NUMERICAL_DIVERGENCE_LOG_ACCEPTANCE, dtype),
        ),
        log_acceptance_ratio,
    )
    uniform = tf.random.stateless_uniform(
        tf.shape(log_acceptance_ratio),
        acceptance_seed,
        minval=tf.constant(0.0, dtype),
        maxval=tf.constant(1.0, dtype),
        dtype=dtype,
    )
    accepted = tf.logical_and(
        tf.math.log(uniform) < log_acceptance_ratio,
        tf.logical_not(divergence),
    )
    next_position = tf.where(accepted[:, tf.newaxis], proposal.position, position)
    next_potential = tf.where(accepted, proposed_potential, current_potential)
    finite_status = tf.logical_and(
        tf.math.is_finite(proposed_potential),
        tf.logical_and(tf.math.is_finite(raw_delta_h), tf.logical_not(divergence)),
    )
    trace = NeuralForceHMCTrace(
        accepted=accepted,
        log_acceptance_ratio=log_acceptance_ratio,
        initial_potential=current_potential,
        final_potential=proposed_potential,
        initial_kinetic=initial_kinetic,
        final_kinetic=final_kinetic,
        delta_h=delta_h,
        finite_status=finite_status,
        endpoint_out_of_support=endpoint_out_of_support,
        endpoint_call_count=tf.constant(1, tf.int32),
        force_call_count=proposal.force_call_count,
        initial_momentum=momentum,
        final_momentum=proposal.momentum,
        divergence=divergence,
        force_fallback=proposal.force_fallback,
    )
    return NeuralForceHMCTransition(next_position, next_potential, trace)


class NeuralForceTransitionKernel(tfp.mcmc.TransitionKernel):
    """Calibrated TFP kernel for the BayesFilter neural-force proposal.

    The kernel exposes the conventional ``step_size`` and ``log_accept_ratio``
    result fields, so TFP's dual-averaging adapter can tune the existing
    endpoint-only proposal without changing its mechanics.
    """

    def __init__(
        self,
        *,
        force: FrozenPositionOnlyForce,
        target: FrozenTargetPotential,
        config: NeuralForceHMCConfig,
        name: str = "neural_force_hmc",
    ) -> None:
        if not isinstance(force, FrozenPositionOnlyForce):
            raise TypeError("force must be FrozenPositionOnlyForce")
        if not isinstance(target, FrozenTargetPotential):
            raise TypeError("target must be FrozenTargetPotential")
        if not isinstance(config, NeuralForceHMCConfig):
            raise TypeError("config must be NeuralForceHMCConfig")
        self._parameters = {
            "force": force,
            "target": target,
            "config": config,
            "name": str(name),
        }

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    @property
    def is_calibrated(self) -> bool:
        return True

    def bootstrap_results(self, init_state: tf.Tensor) -> NeuralForceTransitionKernelResults:
        state = tf.convert_to_tensor(init_state, dtype=self.config.tf_dtype)
        potential = _evaluate_endpoint_potential(
            self.target,
            state,
            dtype=self.config.tf_dtype,
        )
        target_log_prob = -potential
        zeros = tf.zeros_like(target_log_prob)
        return NeuralForceTransitionKernelResults(
            accepted_results=NeuralForceEndpointKernelResults(target_log_prob),
            is_accepted=tf.ones_like(target_log_prob, dtype=tf.bool),
            log_accept_ratio=zeros,
            proposed_results=NeuralForceEndpointKernelResults(target_log_prob),
            step_size=tf.convert_to_tensor(
                self.config.step_size, dtype=self.config.tf_dtype
            ),
            delta_h=zeros,
            finite_status=tf.math.is_finite(target_log_prob),
            divergence=tf.zeros_like(target_log_prob, dtype=tf.bool),
            force_fallback=tf.zeros_like(target_log_prob, dtype=tf.bool),
        )

    def one_step(
        self,
        current_state: tf.Tensor,
        previous_kernel_results: NeuralForceTransitionKernelResults,
        seed: tf.Tensor | tuple[int, int] | None = None,
    ) -> tuple[tf.Tensor, NeuralForceTransitionKernelResults]:
        if seed is None:
            raise ValueError("NeuralForceTransitionKernel requires a stateless seed")
        current_potential = -tf.convert_to_tensor(
            previous_kernel_results.accepted_results.target_log_prob,
            dtype=self.config.tf_dtype,
        )
        transition = neural_force_hmc_transition(
            current_state,
            current_potential,
            self.force,
            self.target,
            self.config,
            seed,
            step_size=previous_kernel_results.step_size,
        )
        trace = transition.trace
        return transition.position, NeuralForceTransitionKernelResults(
            accepted_results=NeuralForceEndpointKernelResults(-transition.potential),
            is_accepted=trace.accepted,
            log_accept_ratio=-trace.delta_h,
            proposed_results=NeuralForceEndpointKernelResults(-trace.final_potential),
            step_size=tf.convert_to_tensor(
                previous_kernel_results.step_size,
                dtype=self.config.tf_dtype,
            ),
            delta_h=trace.delta_h,
            finite_status=trace.finite_status,
            divergence=trace.divergence,
            force_fallback=trace.force_fallback,
        )

    @property
    def force(self) -> FrozenPositionOnlyForce:
        return self._parameters["force"]

    @property
    def target(self) -> FrozenTargetPotential:
        return self._parameters["target"]

    @property
    def config(self) -> NeuralForceHMCConfig:
        return self._parameters["config"]


def run_full_chain_neural_force_hmc(
    adapter: Any,
    initial_state: Any,
    config: Any,
    *,
    force: FrozenPositionOnlyForce,
    target: FrozenTargetPotential,
) -> Any:
    """Run the endpoint-only neural-force kernel through native TFP adaptation.

    ``adapter`` is the BayesFilter fixed-mass affine wrapper supplied by the
    native tuning ladder. Its coordinate map is applied to both the frozen
    force and exact endpoint potential, so tuning and verification remain bound
    to the same mass artifact as the exact-gradient arm.
    """

    from bayesfilter.inference.hmc import FullChainHMCConfig, FullChainHMCRunResult
    from bayesfilter.inference.hmc_tuning import HMCTuningPolicy

    if not isinstance(config, FullChainHMCConfig):
        raise TypeError("config must be FullChainHMCConfig")
    if config.target_status_trace_policy != "none":
        raise ValueError("neural-force runner does not expose target-status telemetry")
    latent_to_position = getattr(adapter, "latent_to_position", None)
    transform = getattr(adapter, "transform", None)
    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    if state.shape.rank != 2 or state.shape[-1] is None:
        raise ValueError("initial state must have static shape [chain, parameter]")
    dimension = int(state.shape[-1])
    if callable(latent_to_position) and transform is not None:
        coordinate_to_neutra = latent_to_position
        factor = tf.convert_to_tensor(transform.factor, dtype=state.dtype)
        center = tf.convert_to_tensor(transform.center, dtype=state.dtype)
        coordinate_route = "native_fixed_mass_affine"
    else:
        coordinate_to_neutra = lambda value: tf.convert_to_tensor(value, state.dtype)
        factor = tf.eye(dimension, dtype=state.dtype)
        center = tf.zeros((dimension,), dtype=state.dtype)
        coordinate_route = "direct_fixed_transport_z"
    tf.debugging.assert_equal(
        tf.shape(factor),
        tf.constant((dimension, dimension), tf.int32),
        message="mass affine factor shape mismatch",
    )

    def mass_coordinate_force(value: tf.Tensor) -> tf.Tensor:
        position = coordinate_to_neutra(value)
        position_force = tf.convert_to_tensor(force.function(position), value.dtype)
        return tf.tensordot(position_force, factor, axes=[[-1], [0]])

    def mass_coordinate_target(value: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(
            target.function(coordinate_to_neutra(value)),
            value.dtype,
        )

    bound_force = FrozenPositionOnlyForce(
        function=mass_coordinate_force,
        identity=f"{force.identity}:native-fixed-mass-affine",
        semantics=force.semantics,
        coordinate_system="transformed",
    )
    bound_target = FrozenTargetPotential(
        function=mass_coordinate_target,
        identity=f"{target.identity}:native-fixed-mass-affine",
        coordinate_system="transformed",
        includes_chart_log_jacobian=target.includes_chart_log_jacobian,
    )
    kernel_config = NeuralForceHMCConfig(
        step_size=config.step_size,
        num_leapfrog_steps=config.num_leapfrog_steps,
        inverse_mass_diagonal=(1.0,) * dimension,
        dtype=state.dtype.name,
    )
    kernel: tfp.mcmc.TransitionKernel = NeuralForceTransitionKernel(
        force=bound_force,
        target=bound_target,
        config=kernel_config,
    )
    if config.tuning_policy.uses_dual_averaging:
        if config.tuning_policy.label != "fixed_mass_dual_averaging":
            raise ValueError("neural-force adaptation requires fixed-mass dual averaging")
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
            target_accept_prob=tf.constant(
                config.tuning_policy.target_accept_prob,
                dtype=state.dtype,
            ),
        )

    adaptive = bool(config.tuning_policy.uses_dual_averaging)

    def trace_fn(_state: tf.Tensor, results: Any) -> dict[str, tf.Tensor]:
        inner = results.inner_results if adaptive else results
        payload = {
            "is_accepted": inner.is_accepted,
            "log_accept_ratio": inner.log_accept_ratio,
            "target_log_prob": inner.accepted_results.target_log_prob,
            "proposed_target_log_prob": inner.proposed_results.target_log_prob,
            "delta_h": inner.delta_h,
            "finite_status": inner.finite_status,
            "divergence": inner.divergence,
            "force_fallback": inner.force_fallback,
        }
        if adaptive:
            payload.update(
                {
                    "step_size": results.new_step_size,
                    "target_accept_prob": results.target_accept_prob,
                    "num_adaptation_steps": tf.convert_to_tensor(
                        config.tuning_policy.num_adaptation_steps,
                        tf.int32,
                    ),
                }
            )
        return payload

    def run_chain() -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        return tfp.mcmc.sample_chain(
            num_results=config.num_results,
            num_burnin_steps=config.num_burnin_steps,
            current_state=state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=tf.constant(config.seed, tf.int32),
        )

    runner = (
        tf.function(run_chain, jit_compile=True, reduce_retracing=True)
        if config.use_xla
        else run_chain
        if config.chain_execution_mode == "eager"
        else tf.function(run_chain, reduce_retracing=True)
    )
    started = time.perf_counter()
    samples, trace = runner()
    elapsed = time.perf_counter() - started
    finite_samples = tf.reduce_all(tf.math.is_finite(samples), axis=-1)
    log_accept = tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64)
    acceptance_probability = tf.exp(tf.minimum(log_accept, tf.zeros_like(log_accept)))
    diagnostics: dict[str, Any] = {
        "finite_sample_count": tf.reduce_sum(tf.cast(finite_samples, tf.int32)),
        "nonfinite_sample_count": tf.reduce_sum(
            tf.cast(tf.logical_not(finite_samples), tf.int32)
        ),
        "sample_shape": tuple(int(item) for item in samples.shape),
        "trace_policy": config.trace_policy,
        "acceptance_rate": tf.reduce_mean(acceptance_probability),
        "acceptance_rate_semantics": "mean_metropolis_acceptance_probability",
        "binary_acceptance_rate": tf.reduce_mean(
            tf.cast(trace["is_accepted"], tf.float64)
        ),
        "log_accept_ratio_finite_count": tf.reduce_sum(
            tf.cast(tf.math.is_finite(log_accept), tf.int32)
        ),
        "log_accept_ratio_nonfinite_count": tf.reduce_sum(
            tf.cast(tf.logical_not(tf.math.is_finite(log_accept)), tf.int32)
        ),
        "log_accept_ratio_max_abs_finite": tf.reduce_max(tf.abs(log_accept)),
        "maximum_absolute_delta_h": tf.reduce_max(tf.abs(trace["delta_h"])),
        "target_log_prob_finite_count": tf.reduce_sum(
            tf.cast(tf.math.is_finite(trace["target_log_prob"]), tf.int32)
        ),
        "target_log_prob_nonfinite_count": tf.reduce_sum(
            tf.cast(tf.logical_not(tf.math.is_finite(trace["target_log_prob"])), tf.int32)
        ),
        "target_log_prob_min_finite": tf.reduce_min(trace["target_log_prob"]),
        "target_log_prob_max_finite": tf.reduce_max(trace["target_log_prob"]),
        "native_divergence_status": "available",
        "divergence_status": "available",
        "divergence_count": tf.reduce_sum(
            tf.cast(trace["divergence"], tf.int32)
        ),
        "divergence_count_by_chain": tf.reduce_sum(
            tf.cast(trace["divergence"], tf.int32), axis=0
        ),
        "force_fallback_count": tf.reduce_sum(
            tf.cast(trace["force_fallback"], tf.int32)
        ),
        "force_fallback_count_by_chain": tf.reduce_sum(
            tf.cast(trace["force_fallback"], tf.int32), axis=0
        ),
    }
    if adaptive:
        step_trace = tf.convert_to_tensor(trace["step_size"], tf.float64)
        diagnostics.update(
            {
                "final_step_size": step_trace[-1],
                "final_step_size_finite": tf.reduce_all(tf.math.is_finite(step_trace)),
                "target_accept_prob": tf.reshape(trace["target_accept_prob"], [-1])[-1],
                "num_adaptation_steps": tf.reshape(
                    trace["num_adaptation_steps"], [-1]
                )[-1],
            }
        )
    return FullChainHMCRunResult(
        samples=samples,
        trace=trace,
        diagnostics=diagnostics,
        metadata={
            "runtime": "bayesfilter.neural_force_tfp_transition_kernel",
            "jit_compile": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "target_scope": config.target_scope,
            "force_identity": force.identity,
            "target_identity": target.identity,
            "force_semantics": force.semantics,
            "target_coordinate_system": target.coordinate_system,
            "target_includes_chart_log_jacobian": (
                target.includes_chart_log_jacobian
            ),
            "affine_log_jacobian_convention": "constant_omitted",
            "mass_coordinate_factor_shape": tuple(int(item) for item in factor.shape),
            "mass_coordinate_center_shape": tuple(
                int(item) for item in center.shape
            ),
            "coordinate_route": coordinate_route,
            "sample_chain_call_s": elapsed,
            "endpoint_only_exact_value": True,
            "exact_filter_gradient_inside_leapfrog": False,
            "trace_unavailability": {},
        },
    )


def build_affine_neural_force_transition_kernel(
    *,
    adapter: Any,
    force: FrozenPositionOnlyForce,
    target: FrozenTargetPotential,
    step_size: Any,
    num_leapfrog_steps: Any,
) -> NeuralForceTransitionKernel:
    """Build the typed endpoint-corrected kernel in native affine coordinates."""

    latent_to_position = getattr(adapter, "latent_to_position", None)
    transform = getattr(adapter, "transform", None)
    if not callable(latent_to_position) or transform is None:
        raise InvalidNeuralForceHMCConfiguration(
            "tensor kernel construction requires a native affine adapter"
        )
    factor = tf.cast(tf.convert_to_tensor(transform.factor), tf.float64)
    center = tf.cast(tf.convert_to_tensor(transform.center), tf.float64)
    if factor.shape.rank != 2 or factor.shape[0] is None or factor.shape[1] is None:
        raise InvalidNeuralForceHMCConfiguration(
            "affine factor must have a static square shape"
        )
    dimension = int(factor.shape[0])
    if factor.shape != (dimension, dimension) or center.shape != (dimension,):
        raise InvalidNeuralForceHMCConfiguration(
            "affine center/factor shape mismatch"
        )
    convention = str(
        getattr(transform, "log_jacobian_convention", "constant_omitted")
    )
    if convention != "constant_omitted":
        raise InvalidNeuralForceHMCConfiguration(
            "native affine neural-force HMC requires constant_omitted convention"
        )

    def active_force(value: tf.Tensor) -> tf.Tensor:
        raw_position = latent_to_position(value)
        raw_force = tf.convert_to_tensor(force.function(raw_position), value.dtype)
        return tf.tensordot(raw_force, factor, axes=[[-1], [0]])

    def active_target(value: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(
            target.function(latent_to_position(value)), value.dtype
        )

    bound_force = FrozenPositionOnlyForce(
        function=active_force,
        identity=f"{force.identity}:native-fixed-mass-affine",
        semantics=force.semantics,
        coordinate_system="transformed",
    )
    bound_target = FrozenTargetPotential(
        function=active_target,
        identity=f"{target.identity}:native-fixed-mass-affine",
        coordinate_system="transformed",
        includes_chart_log_jacobian=target.includes_chart_log_jacobian,
    )
    return NeuralForceTransitionKernel(
        force=bound_force,
        target=bound_target,
        config=NeuralForceHMCConfig(
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            inverse_mass_diagonal=(1.0,) * dimension,
            dtype="float64",
        ),
    )


def _neural_force_tuning_source_dependency_closure() -> dict[str, Any]:
    """Hash the executable boundary of the typed neural-force tuning binding."""

    inference_root = Path(__file__).resolve().parent
    package_root = inference_root.parent
    repository_root = package_root.parent
    paths = (
        inference_root / "neural_force_hmc.py",
        inference_root / "hmc_tuning_dispatch.py",
        inference_root / "hmc_tensorflow_tuning.py",
        inference_root / "hmc_kernel_tuning.py",
        inference_root / "hmc.py",
        inference_root / "hmc_tuning.py",
        inference_root / "hmc_coordinates.py",
        inference_root / "tuning_contract.py",
        inference_root / "posterior_adapter.py",
        inference_root / "__init__.py",
        package_root / "__init__.py",
    )
    missing = tuple(str(path) for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(
            "neural-force tuning source closure is incomplete: " + ", ".join(missing)
        )
    return {
        "schema": "bayesfilter.neural_force_hmc_tuning_source_closure.v1",
        "files": tuple(
            {
                "path": str(path.relative_to(repository_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ),
    }


def bind_neural_force_hmc_tuning_runner(
    *,
    force: FrozenPositionOnlyForce,
    target: FrozenTargetPotential,
    target_scope: str,
) -> HMCTuningRunnerBinding:
    """Bind neural-force mechanics to the ordinary public tuning ladder.

    The ordinary tuner owns mass adaptation, epsilon, and leapfrog-count
    selection. This binding only supplies the fixed-configuration transition
    mechanics and exact endpoint potential. Direct identity-mass coordinates
    are rejected by the binding after every stage call.
    """

    if not isinstance(force, FrozenPositionOnlyForce):
        raise TypeError("force must be FrozenPositionOnlyForce")
    if not isinstance(target, FrozenTargetPotential):
        raise TypeError("target must be FrozenTargetPotential")
    scope = _require_nonempty(target_scope, "target_scope")
    if force.coordinate_system != target.coordinate_system:
        raise InvalidNeuralForceHMCConfiguration(
            "force and endpoint target must use the same coordinate system"
        )
    if force.coordinate_system != "raw":
        raise InvalidNeuralForceHMCConfiguration(
            "ordinary neural-force tuning requires raw adapter coordinates; "
            "a transformed target belongs to the fixed-transport contract"
        )

    def bound_runner(adapter: Any, initial_state: Any, config: Any) -> Any:
        return run_full_chain_neural_force_hmc(
            adapter,
            initial_state,
            config,
            force=force,
            target=target,
        )

    def tensor_kernel_factory(
        *, adapter: Any, step_size: Any, num_leapfrog_steps: Any
    ) -> NeuralForceTransitionKernel:
        return build_affine_neural_force_transition_kernel(
            adapter=adapter,
            force=force,
            target=target,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )

    return _issue_hmc_tuning_runner_binding(
        runner=bound_runner,
        tensor_kernel_factory=tensor_kernel_factory,
        runner_identity=(
            "bayesfilter.inference.neural_force_hmc."
            "run_full_chain_neural_force_hmc"
        ),
        algorithm_family="endpoint_corrected_frozen_position_force_hmc",
        target_scope=scope,
        coordinate_scope=force.coordinate_system,
        force_identity=force.identity,
        force_semantics=force.semantics,
        endpoint_target_identity=target.identity,
        endpoint_target_coordinate_system=target.coordinate_system,
        endpoint_target_includes_chart_log_jacobian=(
            target.includes_chart_log_jacobian
        ),
        affine_log_jacobian_convention="constant_omitted",
        target_status_evidence=(
            "exact_endpoint_target_finite_health_and_transition_finite_status_fail_closed"
        ),
        supported_target_status_trace_policies=("none",),
        supported_chain_execution_modes=("tf_function", "eager"),
        backend="tensorflow_probability",
        dtype="float64",
        xla_capable=True,
        required_diagnostic_fields=(
            "finite_sample_count",
            "nonfinite_sample_count",
            "log_accept_ratio_finite_count",
            "log_accept_ratio_nonfinite_count",
            "maximum_absolute_delta_h",
            "target_log_prob_finite_count",
            "target_log_prob_nonfinite_count",
            "divergence_count",
            "force_fallback_count",
        ),
        required_metadata_fields=(
            "force_identity",
            "force_semantics",
            "target_identity",
            "target_coordinate_system",
            "target_includes_chart_log_jacobian",
            "affine_log_jacobian_convention",
            "coordinate_route",
            "target_scope",
            "chain_execution_mode",
        ),
        source_dependency_closure=_neural_force_tuning_source_dependency_closure(),
    )


def sample_neural_force_hmc(
    initial_position: tf.Tensor,
    initial_potential: tf.Tensor,
    force: FrozenPositionOnlyForce,
    target: FrozenTargetPotential,
    config: NeuralForceHMCConfig,
    *,
    num_warmup: int,
    num_results: int,
    seed: tf.Tensor | tuple[int, int],
) -> NeuralForceHMCChain:
    """Run fixed-kernel chains and retain both warm-up and evidence states."""

    if int(num_warmup) != num_warmup or num_warmup < 0:
        raise ValueError("num_warmup must be a nonnegative integer")
    if int(num_results) != num_results or num_results < 1:
        raise ValueError("num_results must be a positive integer")
    total = int(num_warmup) + int(num_results)
    dtype = config.tf_dtype
    initial_position = tf.convert_to_tensor(initial_position, dtype=dtype)
    initial_potential = tf.convert_to_tensor(initial_potential, dtype=dtype)
    seed = tf.convert_to_tensor(seed, tf.int32)
    positions = tf.TensorArray(dtype, size=total, clear_after_read=False)
    potentials = tf.TensorArray(dtype, size=total, clear_after_read=False)
    accepted = tf.TensorArray(tf.bool, size=total, clear_after_read=False)
    log_ratios = tf.TensorArray(dtype, size=total, clear_after_read=False)
    initial_potentials = tf.TensorArray(dtype, size=total, clear_after_read=False)
    final_potentials = tf.TensorArray(dtype, size=total, clear_after_read=False)
    initial_kinetics = tf.TensorArray(dtype, size=total, clear_after_read=False)
    final_kinetics = tf.TensorArray(dtype, size=total, clear_after_read=False)
    delta_h = tf.TensorArray(dtype, size=total, clear_after_read=False)
    finite_status = tf.TensorArray(tf.bool, size=total, clear_after_read=False)
    divergence = tf.TensorArray(tf.bool, size=total, clear_after_read=False)
    force_fallback = tf.TensorArray(tf.bool, size=total, clear_after_read=False)
    endpoint_counts = tf.TensorArray(tf.int32, size=total, clear_after_read=False)
    force_counts = tf.TensorArray(tf.int32, size=total, clear_after_read=False)

    def cond(index: tf.Tensor, *_: Any) -> tf.Tensor:
        return index < total

    def body(
        index: tf.Tensor,
        position: tf.Tensor,
        potential: tf.Tensor,
        position_array: tf.TensorArray,
        potential_array: tf.TensorArray,
        accepted_array: tf.TensorArray,
        ratio_array: tf.TensorArray,
        initial_potential_array: tf.TensorArray,
        final_potential_array: tf.TensorArray,
        initial_kinetic_array: tf.TensorArray,
        final_kinetic_array: tf.TensorArray,
        delta_array: tf.TensorArray,
        finite_array: tf.TensorArray,
        divergence_array: tf.TensorArray,
        force_fallback_array: tf.TensorArray,
        endpoint_count_array: tf.TensorArray,
        force_count_array: tf.TensorArray,
    ) -> tuple[Any, ...]:
        iteration_seed = tf.random.experimental.stateless_fold_in(seed, index)
        transition = neural_force_hmc_transition(
            position, potential, force, target, config, iteration_seed
        )
        return (
            index + 1,
            transition.position,
            transition.potential,
            position_array.write(index, transition.position),
            potential_array.write(index, transition.potential),
            accepted_array.write(index, transition.trace.accepted),
            ratio_array.write(index, transition.trace.log_acceptance_ratio),
            initial_potential_array.write(index, transition.trace.initial_potential),
            final_potential_array.write(index, transition.trace.final_potential),
            initial_kinetic_array.write(index, transition.trace.initial_kinetic),
            final_kinetic_array.write(index, transition.trace.final_kinetic),
            delta_array.write(index, transition.trace.delta_h),
            finite_array.write(index, transition.trace.finite_status),
            divergence_array.write(index, transition.trace.divergence),
            force_fallback_array.write(index, transition.trace.force_fallback),
            endpoint_count_array.write(index, transition.trace.endpoint_call_count),
            force_count_array.write(index, transition.trace.force_call_count),
        )

    result = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, tf.int32),
            initial_position,
            initial_potential,
            positions,
            potentials,
            accepted,
            log_ratios,
            initial_potentials,
            final_potentials,
            initial_kinetics,
            final_kinetics,
            delta_h,
            finite_status,
            divergence,
            force_fallback,
            endpoint_counts,
            force_counts,
        ),
        parallel_iterations=1,
    )
    return NeuralForceHMCChain(
        positions=result[3].stack(),
        potentials=result[4].stack(),
        accepted=result[5].stack(),
        log_acceptance_ratio=result[6].stack(),
        initial_potential=result[7].stack(),
        final_potential=result[8].stack(),
        initial_kinetic=result[9].stack(),
        final_kinetic=result[10].stack(),
        delta_h=result[11].stack(),
        finite_status=result[12].stack(),
        endpoint_call_count=result[15].stack(),
        force_call_count=result[16].stack(),
        num_warmup=tf.constant(num_warmup, tf.int32),
        divergence=result[13].stack(),
        force_fallback=result[14].stack(),
    )


__all__ = [
    "DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS",
    "FrozenPositionOnlyForce",
    "FrozenTargetPotential",
    "InvalidNeuralForceHMCConfiguration",
    "NEURAL_FORCE_HMC_SCHEMA",
    "NeuralForceHMCChain",
    "NeuralForceHMCConfig",
    "NeuralForceHMCTrace",
    "NeuralForceHMCTransition",
    "NeuralForceProposal",
    "POSITION_ONLY_FORCE_SEMANTICS",
    "bind_neural_force_hmc_tuning_runner",
    "build_affine_neural_force_transition_kernel",
    "kinetic_energy",
    "neural_force_hmc_transition",
    "neural_force_proposal",
    "sample_neural_force_hmc",
    "run_full_chain_neural_force_hmc",
]
