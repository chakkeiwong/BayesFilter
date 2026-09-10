"""Single-cloud adapters for the shared Contract-E reset kernel.

The semantic implementation lives in ``ledh_unified_reset_tf`` with explicit
batch and direction axes. These adapters preserve the historical [N,...]
public contract by promoting to B=1/K=1 and unwrapping the result. No autodiff
is used in either path.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.ledh_unified_reset_tf import (
    batched_sinkhorn_contract_e_reset_triple_with_tangent,
)

Tensor = tf.Tensor


def sinkhorn_contract_e_reset_with_tangent(
    children: Tensor,
    d_children: Tensor,
    weights: Tensor,
    d_weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    """Return reset particles and one analytical score-direction tangent."""
    children = tf.convert_to_tensor(children)
    count = tf.shape(children)[0]
    dimension = tf.shape(children)[1]
    dummy_covariances = tf.zeros(
        [count, dimension, dimension], dtype=children.dtype
    )
    result = batched_sinkhorn_contract_e_reset_triple_with_tangent(
        children[None, ...],
        d_children[None, None, ...],
        dummy_covariances[None, ...],
        tf.zeros_like(dummy_covariances)[None, None, ...],
        weights[None, ...],
        d_weights[None, None, ...],
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
    )
    return result[0][0], result[1][0, 0]


def sinkhorn_contract_e_reset_triple_with_tangent(
    children: Tensor,
    d_children: Tensor,
    covariances: Tensor,
    d_covariances: Tensor,
    weights: Tensor,
    d_weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reset states and carry covariances using the same shared transport."""
    result = batched_sinkhorn_contract_e_reset_triple_with_tangent(
        children[None, ...],
        d_children[None, None, ...],
        covariances[None, ...],
        d_covariances[None, None, ...],
        weights[None, ...],
        d_weights[None, None, ...],
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
    )
    return (
        result[0][0],
        result[1][0, 0],
        result[2][0],
        result[3][0, 0],
        result[4][0],
        result[5][0, 0],
    )


__all__ = [
    "sinkhorn_contract_e_reset_triple_with_tangent",
    "sinkhorn_contract_e_reset_with_tangent",
]
