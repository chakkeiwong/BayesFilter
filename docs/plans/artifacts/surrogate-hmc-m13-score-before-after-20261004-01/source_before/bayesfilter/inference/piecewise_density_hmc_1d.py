"""Bounded one-dimensional event HMC for a piecewise-density target.

This module is deliberately narrower than the ordinary HMC runners.  It
implements the scalar reflection/refraction mechanics used by the BGS ZLB
reopen plan; the exact endpoint log density remains the Metropolis authority.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp


DTYPE = tf.float64


def scalar_reflection_or_refraction(
    momentum: tf.Tensor,
    potential_jump: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply the unit-mass normal energy rule to a scalar momentum.

    ``potential_jump`` is ``U_after - U_before``.  The returned tensors are
    ``(new_momentum, reflected, refracted)``.  Reflection is selected when the
    available kinetic energy cannot pay an uphill jump; downhill jumps always
    refract and increase the available kinetic energy.
    """

    momentum = tf.cast(tf.convert_to_tensor(momentum), DTYPE)
    potential_jump = tf.cast(tf.convert_to_tensor(potential_jump), DTYPE)
    kinetic = 0.5 * tf.square(momentum)
    can_cross = kinetic >= potential_jump
    reflected_momentum = -momentum
    remaining = tf.maximum(tf.square(momentum) - 2.0 * potential_jump, 0.0)
    refracted_momentum = tf.sign(momentum) * tf.sqrt(remaining)
    new_momentum = tf.where(can_cross, refracted_momentum, reflected_momentum)
    reflected = tf.logical_not(can_cross)
    refracted = can_cross
    return new_momentum, reflected, refracted


def scalar_event_drift(
    position: tf.Tensor,
    momentum: tf.Tensor,
    *,
    boundary: tf.Tensor | float,
    potential_jump: tf.Tensor | float,
    step_size: tf.Tensor | float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Advance a scalar linear drift through at most one known boundary.

    The crossing time is solved from the linear drift exactly.  The event map
    then changes only the momentum and spends the remaining drift time on the
    new side.  This is a scalar primitive; simultaneous or multiple BGS
    surfaces are outside its contract.
    """

    position = tf.cast(tf.convert_to_tensor(position), DTYPE)
    momentum = tf.cast(tf.convert_to_tensor(momentum), DTYPE)
    boundary = tf.cast(tf.convert_to_tensor(boundary), DTYPE)
    potential_jump = tf.cast(tf.convert_to_tensor(potential_jump), DTYPE)
    step_size = tf.cast(tf.convert_to_tensor(step_size), DTYPE)
    raw_position = position + step_size * momentum
    crossed = tf.math.logical_xor(
        position < boundary,
        raw_position < boundary,
    )
    safe_momentum = tf.where(
        tf.abs(momentum) > tf.constant(1.0e-30, DTYPE),
        momentum,
        tf.ones_like(momentum),
    )
    event_time = tf.clip_by_value(
        (boundary - position) / safe_momentum,
        tf.zeros_like(position),
        step_size,
    )
    moving_up = tf.logical_and(position < boundary, raw_position >= boundary)
    signed_jump = tf.where(moving_up, potential_jump, -potential_jump)
    event_momentum, reflected, refracted = scalar_reflection_or_refraction(
        momentum, signed_jump
    )
    remaining = tf.maximum(step_size - event_time, tf.zeros_like(step_size))
    event_position = boundary + event_momentum * remaining
    return (
        tf.where(crossed, event_position, raw_position),
        tf.where(crossed, event_momentum, momentum),
        tf.cast(crossed, tf.bool),
        tf.math.logical_and(crossed, reflected),
        tf.math.logical_and(crossed, refracted),
    )


class PiecewiseDensity1DTransitionKernelResults(NamedTuple):
    accepted_results: Any
    proposed_results: Any
    is_accepted: tf.Tensor
    log_accept_ratio: tf.Tensor
    delta_h: tf.Tensor
    event_count: tf.Tensor
    reflection_count: tf.Tensor
    refraction_count: tf.Tensor
    initial_momentum: tf.Tensor
    final_momentum: tf.Tensor
    finite_status: tf.Tensor


def piecewise_density_leapfrog_proposal_1d(
    position: tf.Tensor,
    momentum: tf.Tensor,
    *,
    score_fn: Callable[[tf.Tensor], tf.Tensor],
    boundary: tf.Tensor | float,
    potential_jump: tf.Tensor | float,
    step_size: tf.Tensor | float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """One symmetric scalar kick-event-drift-kick proposal."""

    position = tf.cast(tf.convert_to_tensor(position), DTYPE)
    momentum = tf.cast(tf.convert_to_tensor(momentum), DTYPE)
    step_size = tf.cast(tf.convert_to_tensor(step_size), DTYPE)
    score = tf.cast(score_fn(position), DTYPE)
    half_momentum = momentum + 0.5 * step_size * score
    proposal, event_momentum, crossed, reflected, refracted = scalar_event_drift(
        position,
        half_momentum,
        boundary=boundary,
        potential_jump=potential_jump,
        step_size=step_size,
    )
    proposal_score = tf.cast(score_fn(proposal), DTYPE)
    final_momentum = event_momentum + 0.5 * step_size * proposal_score
    return proposal, final_momentum, crossed, reflected, refracted


class PiecewiseDensity1DTransitionKernel(tfp.mcmc.TransitionKernel):
    """Scalar reflection/refraction HMC with exact endpoint correction.

    The target and score are supplied separately.  ``score_fn`` is used only
    for smooth kicks; endpoint acceptance always calls ``target_log_prob_fn``.
    The known scalar boundary and one-sided potential jump are explicit API
    inputs, so this class refuses to infer a multidimensional BGS normal.
    """

    def __init__(
        self,
        *,
        target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
        score_fn: Callable[[tf.Tensor], tf.Tensor],
        boundary: float,
        potential_jump: float,
        step_size: float,
        name: str = "piecewise_density_hmc_1d",
    ) -> None:
        if not callable(target_log_prob_fn) or not callable(score_fn):
            raise TypeError("target_log_prob_fn and score_fn must be callable")
        if not tf.math.is_finite(tf.constant(boundary, DTYPE)):
            raise ValueError("boundary must be finite")
        if not tf.math.is_finite(tf.constant(potential_jump, DTYPE)):
            raise ValueError("potential_jump must be finite")
        if not tf.math.is_finite(tf.constant(step_size, DTYPE)) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        self._parameters = {
            "target_log_prob_fn": target_log_prob_fn,
            "score_fn": score_fn,
            "boundary": float(boundary),
            "potential_jump": float(potential_jump),
            "step_size": float(step_size),
            "name": str(name),
        }

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def target_log_prob_fn(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return self._parameters["target_log_prob_fn"]

    @property
    def score_fn(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return self._parameters["score_fn"]

    @property
    def boundary(self) -> tf.Tensor:
        return tf.constant(self._parameters["boundary"], DTYPE)

    @property
    def potential_jump(self) -> tf.Tensor:
        return tf.constant(self._parameters["potential_jump"], DTYPE)

    @property
    def step_size(self) -> tf.Tensor:
        return tf.constant(self._parameters["step_size"], DTYPE)

    def _endpoint(self, state: tf.Tensor) -> tf.Tensor:
        state = tf.ensure_shape(tf.cast(tf.convert_to_tensor(state), DTYPE), [None])
        value = tf.cast(self.target_log_prob_fn(state), DTYPE)
        return tf.ensure_shape(value, [None])

    def bootstrap_results(
        self, init_state: tf.Tensor
    ) -> PiecewiseDensity1DTransitionKernelResults:
        state = tf.ensure_shape(tf.cast(tf.convert_to_tensor(init_state), DTYPE), [None])
        target_log_prob = self._endpoint(state)
        zeros = tf.zeros_like(state)
        finite = tf.reduce_all(tf.math.is_finite(target_log_prob))
        return PiecewiseDensity1DTransitionKernelResults(
            accepted_results=target_log_prob,
            proposed_results=target_log_prob,
            is_accepted=tf.ones_like(state, tf.bool),
            log_accept_ratio=zeros,
            delta_h=zeros,
            event_count=tf.zeros_like(state, tf.int32),
            reflection_count=tf.zeros_like(state, tf.int32),
            refraction_count=tf.zeros_like(state, tf.int32),
            initial_momentum=zeros,
            final_momentum=zeros,
            finite_status=tf.fill(tf.shape(state), finite),
        )

    def one_step(
        self,
        current_state: tf.Tensor,
        previous_kernel_results: PiecewiseDensity1DTransitionKernelResults,
        seed: tf.Tensor | tuple[int, int] | None = None,
    ) -> tuple[tf.Tensor, PiecewiseDensity1DTransitionKernelResults]:
        if seed is None:
            raise ValueError("PiecewiseDensity1DTransitionKernel requires a stateless seed")
        seed = tf.ensure_shape(tf.cast(tf.convert_to_tensor(seed), tf.int32), [2])
        momentum_seed, acceptance_seed = tf.unstack(tf.random.experimental.stateless_split(seed, 2))
        state = tf.ensure_shape(tf.cast(tf.convert_to_tensor(current_state), DTYPE), [None])
        current_log_prob = tf.cast(previous_kernel_results.accepted_results, DTYPE)
        current_potential = -current_log_prob
        momentum = tf.random.stateless_normal(tf.shape(state), momentum_seed, dtype=DTYPE)
        initial_kinetic = 0.5 * tf.square(momentum)
        proposal, final_momentum, crossed, reflected, refracted = (
            piecewise_density_leapfrog_proposal_1d(
            state,
            momentum,
            score_fn=self.score_fn,
            boundary=self.boundary,
            potential_jump=self.potential_jump,
            step_size=self.step_size,
        )
        )
        proposed_log_prob = self._endpoint(proposal)
        proposed_potential = -proposed_log_prob
        final_kinetic = 0.5 * tf.square(final_momentum)
        delta_h = proposed_potential + final_kinetic - current_potential - initial_kinetic
        log_accept_ratio = tf.minimum(tf.zeros_like(delta_h), -delta_h)
        uniform = tf.random.stateless_uniform(tf.shape(state), acceptance_seed, dtype=DTYPE)
        accepted = tf.logical_and(
            tf.math.is_finite(delta_h), tf.math.log(uniform) < log_accept_ratio
        )
        next_state = tf.where(accepted, proposal, state)
        next_log_prob = tf.where(accepted, proposed_log_prob, current_log_prob)
        finite = tf.logical_and(tf.math.is_finite(proposed_log_prob), tf.math.is_finite(delta_h))
        event = tf.cast(crossed, tf.int32)
        return next_state, PiecewiseDensity1DTransitionKernelResults(
            accepted_results=next_log_prob,
            proposed_results=proposed_log_prob,
            is_accepted=accepted,
            log_accept_ratio=log_accept_ratio,
            delta_h=delta_h,
            event_count=event,
            reflection_count=tf.cast(tf.logical_and(crossed, reflected), tf.int32),
            refraction_count=tf.cast(tf.logical_and(crossed, refracted), tf.int32),
            initial_momentum=momentum,
            final_momentum=final_momentum,
            finite_status=finite,
        )


__all__ = [
    "PiecewiseDensity1DTransitionKernel",
    "PiecewiseDensity1DTransitionKernelResults",
    "piecewise_density_leapfrog_proposal_1d",
    "scalar_event_drift",
    "scalar_reflection_or_refraction",
]
