"""Importable deterministic factories for CPU/XLA cloud boundary tests."""

from __future__ import annotations

from typing import Any, Mapping


def quadratic_value_score_factory(config: Mapping[str, Any]):
    """Return a batched negative-quadratic value/score function."""

    import tensorflow as tf

    precision = tf.convert_to_tensor(config["precision"], tf.float64)
    center = tf.convert_to_tensor(config["center"], tf.float64)

    def value_and_score(rows: Any) -> tuple[Any, Any]:
        delta = tf.convert_to_tensor(rows, tf.float64) - center[None, :]
        scores = -tf.einsum("ij,bj->bi", precision, delta)
        values = 0.5 * tf.reduce_sum(delta * scores, axis=1)
        return values, scores

    return value_and_score
