"""Forward-autodiff ORACLE for the canonical LEDH lane (parity judge only).

C-9 contract: this module is the ONLY place autodiff may touch the
canonical lane. It is never claim-bearing; it exists to gate the P4
analytical score derivations stage by stage.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

Tensor = tf.Tensor


def oracle_forward_autodiff_score(
    value_fn: Callable[[Tensor], Tensor],
    theta: Tensor,
) -> Tensor:
    """Forward-mode JVP score of ``value_fn`` at ``theta`` (one-hot sweep)."""

    theta = tf.convert_to_tensor(theta)
    parameter_count = int(theta.shape[-1])
    directions = []
    for index in range(parameter_count):
        direction = tf.one_hot(index, parameter_count, dtype=theta.dtype)
        with tf.autodiff.ForwardAccumulator(theta, direction) as accumulator:
            value = value_fn(theta)
        directions.append(accumulator.jvp(value))
    return tf.stack(directions, axis=-1)


__all__ = ["oracle_forward_autodiff_score"]
