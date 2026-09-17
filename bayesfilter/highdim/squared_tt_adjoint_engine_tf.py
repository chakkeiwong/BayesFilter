"""Full-path manual adjoint score engine for the branch-axis squared-TT filter.

P2 implementation of UB-1 Addendum A: forward pass identical to
`run_value_filter_branch_axis` (same frozen program) with per-update
checkpoints, then one reverse sweep chaining the node adjoints of
`squared_tt_adjoint_tf`. Gradient cost is O(1) x value in flops
(one transposed solve per update; no per-parameter environment work) —
the P2A-selected mode.

Adapter VJP contract (per UB-1 A.2.4, transposes of the JVP obligations):
    transition_vjp(x_c, x_p, cotangent_rows) -> bar_theta [p]
    observation_vjp(x_c, y, cotangent_rows) -> bar_theta [p]
    initial_vjp(x_c, cotangent_rows)        -> bar_theta [p]

Telescoping note: for interior steps the retained-normalizer cotangent
cancels exactly (increment_t carries +1/Zc_t, increment_{t+1} carries
-1/Zc_t); the implementation keeps both terms explicit so the identity is
an emergent check, not an assumption.
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_native_adjoint_engine_tf import make_adjoint_filter

DTYPE = tf.float64


def run_adjoint_score_filter(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    transition_vjp: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    observation_vjp: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    initial_vjp: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    parameter_dim: int,
    gram_condition_veto: float = 1e12,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Execute the complete native analytical adjoint, then check its vetoes.

    The scaled augmented solve uses the existing XLA CholeskyQR2 backend;
    same-objective value parity and analytical-score FD remain required gates.
    Use ``make_adjoint_filter`` to retain the stable compiled callable across
    repeated evaluations of an identical adapter and observation signature.
    """
    observations = tf.convert_to_tensor(observations, DTYPE)
    call = make_adjoint_filter(adapter, observations.shape, config,
        transition_vjp=transition_vjp, observation_vjp=observation_vjp,
        initial_vjp=initial_vjp, parameter_dim=parameter_dim)
    value, score, status = call(observations)
    if not bool(status['valid'].numpy()):
        raise ValueError('non-finite manual adjoint (fail-closed)')
    if bool((status['worst_condition'] > config.condition_number_veto).numpy()):
        raise ValueError('condition number veto in fixed ALS fit')
    if bool((status['gram_condition'] > gram_condition_veto).numpy()):
        raise ValueError('retained Gram conditioning veto (P2 claim gate)')
    return value, score


__all__ = ['make_adjoint_filter', 'run_adjoint_score_filter']
