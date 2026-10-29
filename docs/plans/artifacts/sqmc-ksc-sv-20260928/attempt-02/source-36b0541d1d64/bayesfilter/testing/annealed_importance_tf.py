"""Diagnostic linear AIS with bridge-correct fixed-HMC rejuvenation.

TFP HMC kernel results cache target values and gradients.  A changing AIS bridge
therefore requires a fresh bootstrap under every new bridge target before HMC's
`one_step`.  This diagnostic driver performs that refresh explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import tensorflow as tf
import tensorflow_probability as tfp


LogProbFn = Callable[[tf.Tensor], tf.Tensor]


def run_linear_ais_fixed_hmc(
    proposal_log_prob_fn: LogProbFn,
    target_log_prob_fn: LogProbFn,
    initial_state: Any,
    *,
    num_steps: int,
    step_size: float,
    num_leapfrog_steps: int,
    seed: tuple[int, int],
    rejuvenation_interval: int = 1,
    jit_compile: bool = True,
) -> Mapping[str, tf.Tensor]:
    """Run linear AIS with bridge-correct fixed-HMC rejuvenation.

    Bridges without HMC use the identity transition, which leaves every bridge
    invariant. Raw proposal and target values are carried across identity steps
    so sparse rejuvenation avoids redundant evaluations of an unchanged state.
    """

    if isinstance(num_steps, bool) or int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if not float(step_size) > 0.0:
        raise ValueError("step_size must be positive")
    if isinstance(num_leapfrog_steps, bool) or int(num_leapfrog_steps) < 2:
        raise ValueError("num_leapfrog_steps must be at least two")
    if isinstance(rejuvenation_interval, bool) or int(rejuvenation_interval) <= 0:
        raise ValueError("rejuvenation_interval must be positive")
    if int(num_steps) % int(rejuvenation_interval) != 0:
        raise ValueError("rejuvenation_interval must divide num_steps")
    state = tf.convert_to_tensor(initial_state)
    if not state.dtype.is_floating or state.shape.rank is None or state.shape.rank < 2:
        raise ValueError("initial_state must be floating [path, event...]")
    step = tf.constant(float(step_size), state.dtype)
    total_steps = tf.constant(int(num_steps), tf.int32)
    interval = tf.constant(int(rejuvenation_interval), tf.int32)
    total_rejuvenations = tf.constant(
        int(num_steps) // int(rejuvenation_interval), tf.int32
    )
    seed_tensor = tf.constant(seed, tf.int32)

    @tf.function(jit_compile=jit_compile, reduce_retracing=False)
    def sample(initial: tf.Tensor) -> Mapping[str, tf.Tensor]:
        initial_proposal = tf.convert_to_tensor(proposal_log_prob_fn(initial), state.dtype)
        initial_target = tf.convert_to_tensor(target_log_prob_fn(initial), state.dtype)
        if initial_proposal.shape.rank != 1 or initial_target.shape != initial_proposal.shape:
            raise ValueError("proposal and target must return one log density per path")
        path_count = tf.shape(initial_proposal)[0]
        initial_finite = tf.logical_and(
            tf.math.is_finite(initial_proposal), tf.math.is_finite(initial_target)
        )

        def condition(
            index: tf.Tensor,
            _current: tf.Tensor,
            _weights: tf.Tensor,
            _proposal: tf.Tensor,
            _target: tf.Tensor,
            _accepted: tf.Tensor,
            _path_finite: tf.Tensor,
            _max_abs_log_accept: tf.Tensor,
        ) -> tf.Tensor:
            return index < total_steps

        def body(
            index: tf.Tensor,
            current: tf.Tensor,
            weights: tf.Tensor,
            proposal_value: tf.Tensor,
            target_value: tf.Tensor,
            accepted_count: tf.Tensor,
            path_finite: tf.Tensor,
            max_abs_log_accept: tf.Tensor,
        ) -> tuple[tf.Tensor, ...]:
            weights = weights + (target_value - proposal_value) / tf.cast(
                total_steps, state.dtype
            )
            beta = tf.cast(index + 1, state.dtype) / tf.cast(total_steps, state.dtype)

            def bridge_log_prob(value: tf.Tensor) -> tf.Tensor:
                proposal = tf.convert_to_tensor(proposal_log_prob_fn(value), state.dtype)
                target = tf.convert_to_tensor(target_log_prob_fn(value), state.dtype)
                return beta * target + (1.0 - beta) * proposal

            def rejuvenate() -> tuple[tf.Tensor, ...]:
                kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=bridge_log_prob,
                    step_size=step,
                    num_leapfrog_steps=int(num_leapfrog_steps),
                )
                # This refresh is required because beta changes between HMC moves.
                bootstrap = kernel.bootstrap_results(current)
                next_state, results = kernel.one_step(
                    current,
                    bootstrap,
                    seed=tf.random.experimental.stateless_fold_in(seed_tensor, index),
                )
                log_accept = tf.convert_to_tensor(results.log_accept_ratio, state.dtype)
                next_proposal = tf.convert_to_tensor(
                    proposal_log_prob_fn(next_state), state.dtype
                )
                next_bridge = tf.convert_to_tensor(
                    results.accepted_results.target_log_prob, state.dtype
                )
                next_target = (
                    next_bridge - (tf.constant(1.0, state.dtype) - beta) * next_proposal
                ) / beta
                step_finite = tf.logical_and(
                    tf.math.is_finite(proposal_value), tf.math.is_finite(target_value)
                )
                step_finite = tf.logical_and(step_finite, tf.math.is_finite(log_accept))
                step_finite = tf.logical_and(
                    step_finite,
                    tf.reduce_all(
                        tf.math.is_finite(next_state),
                        axis=tf.range(1, tf.rank(next_state)),
                    ),
                )
                step_finite = tf.logical_and(
                    step_finite,
                    tf.logical_and(
                        tf.math.is_finite(next_proposal), tf.math.is_finite(next_target)
                    ),
                )
                step_finite = tf.logical_and(
                    step_finite,
                    tf.math.is_finite(results.proposed_results.target_log_prob),
                )
                return (
                    next_state,
                    next_proposal,
                    next_target,
                    tf.cast(results.is_accepted, tf.int32),
                    step_finite,
                    tf.abs(log_accept),
                )

            def identity() -> tuple[tf.Tensor, ...]:
                return (
                    current,
                    proposal_value,
                    target_value,
                    tf.zeros_like(accepted_count),
                    tf.ones_like(path_finite),
                    tf.zeros_like(max_abs_log_accept),
                )

            moved = tf.cond(
                tf.equal(tf.math.floormod(index + 1, interval), 0),
                rejuvenate,
                identity,
            )
            return (
                index + 1,
                moved[0],
                weights,
                moved[1],
                moved[2],
                accepted_count + moved[3],
                tf.logical_and(path_finite, moved[4]),
                tf.maximum(max_abs_log_accept, moved[5]),
            )

        outputs = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0, tf.int32),
                initial,
                tf.zeros_like(initial_proposal),
                initial_proposal,
                initial_target,
                tf.zeros((path_count,), tf.int32),
                initial_finite,
                tf.zeros_like(initial_proposal),
            ),
            parallel_iterations=1,
        )
        terminal = outputs[1]
        terminal_proposal = outputs[3]
        terminal_target = outputs[4]
        path_finite = tf.logical_and(
            outputs[6],
            tf.logical_and(
                tf.math.is_finite(terminal_proposal),
                tf.math.is_finite(terminal_target),
            ),
        )
        return {
            "terminal_state": terminal,
            "log_weights": outputs[2],
            "accepted_count": outputs[5],
            "rejuvenation_count": total_rejuvenations,
            "acceptance_fraction": tf.cast(outputs[5], state.dtype)
            / tf.cast(total_rejuvenations, state.dtype),
            "path_all_finite": path_finite,
            "maximum_absolute_log_accept_ratio": outputs[7],
            "terminal_proposal_log_prob": terminal_proposal,
            "terminal_target_log_prob": terminal_target,
        }

    return sample(state)
