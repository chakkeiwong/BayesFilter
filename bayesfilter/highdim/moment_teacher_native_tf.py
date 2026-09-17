"""Native direction recursion for the existing diagnostic TT moment teacher.

This preserves the extension_or_invention finite program, including every
fitting direction. It does not establish canonical LEDH score admission.
"""

import math
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim.bases import _legendre_values
from bayesfilter.highdim.zhao_cui_moment_teacher_xla import (
    padded_fixed_teacher_recursion_shape_xla,
)
from bayesfilter.ops.quadrature_tf import gauss_legendre
from bayesfilter.ops.stateless_random_tf import philox_normal_float64


def teacher_normal(shape, seed, dtype):
    """Preserve the existing TSL Philox/BoxMuller stream under XLA."""
    if dtype == tf.float64:
        return philox_normal_float64(shape, seed)
    count = math.prod(shape)
    words = tf.random.stateless_uniform(
        [(count + 1) // 2, 2], seed, minval=None, maxval=None,
        dtype=tf.uint32, alg="philox",
    )
    bits = tf.bitwise.bitwise_or(tf.bitwise.bitwise_and(words, 0x7FFFFF), 127 << 23)
    uniforms = tf.bitcast(bits, tf.float32) - 1.0
    radius = tf.sqrt(-2.0 * tf.math.log(tf.maximum(uniforms[:, 0], 1e-7)))
    # TSL BoxMullerFloat multiplies by double M_PI before rounding the angle.
    angle = tf.cast(tf.constant(2.0 * math.pi, tf.float64)
                    * tf.cast(uniforms[:, 1], tf.float64), tf.float32)
    values = tf.stack([tf.sin(angle) * radius, tf.cos(angle) * radius], axis=1)
    return tf.reshape(tf.reshape(values, [-1])[:count], shape)


def teacher_uniform_float32(shape, seed):
    words = tf.random.stateless_uniform(
        shape, seed, minval=None, maxval=None, dtype=tf.uint32, alg="philox")
    bits = tf.bitwise.bitwise_or(tf.bitwise.bitwise_and(words, 0x7FFFFF), 127 << 23)
    return tf.bitcast(bits, tf.float32) - 1.0


@lru_cache(maxsize=16)
def operator_power_program(basis_size, *, jit_compile=True):
    """Pack the original power-specific exact rules, then contract all powers."""
    width = basis_size + 2
    orders = (max(2, basis_size), max(2, basis_size + 1), width)
    rules = tuple(gauss_legendre(order, jit_compile=jit_compile) for order in orders)
    nodes = tuple(tf.pad(rule[0], [[0, width - order]]) for rule, order in zip(rules, orders))
    weights = tuple(tf.pad(rule[1], [[0, width - order]]) for rule, order in zip(rules, orders))
    x = tf.stack([nodes[0], nodes[0], nodes[1], nodes[1], nodes[2]])
    w = tf.stack([weights[0], weights[0], weights[1], weights[1], weights[2]])

    @tf.function(input_signature=[], jit_compile=jit_compile, autograph=False)
    def calculate():
        values = _legendre_values(x, basis_size - 1) * tf.sqrt(
            tf.cast(2 * tf.range(basis_size) + 1, tf.float64))
        powers = tf.pow(x, tf.cast(tf.range(5)[:, None], tf.float64))
        return tf.einsum("pn,pnl,pnm->plm", (0.5 * w) * powers, values, values)

    return calculate


def freeze_scale_shift_core(base, indices, targets_at_indices, maximum_iterations):
    """Return fixed rows and status, preserving early invalid-fit failure."""
    def step(iteration, indices, valid, stable):
        targets = targets_at_indices(indices)
        previous = tf.concat([tf.ones_like(targets["marginal_values"][:1]),
                              targets["marginal_values"][:-1]], axis=0)
        augmented = base + tf.where(
            tf.range(tf.shape(base)[0])[:, None] > 0,
            tf.math.log(tf.maximum(previous, tf.cast(1e-30, base.dtype))), tf.zeros_like(base),
        )
        following = tf.argmax(augmented, axis=1, output_type=tf.int32)
        return iteration + 1, following, targets["valid"], tf.reduce_all(following == indices)

    _, indices, valid, stable = tf.while_loop(
        lambda iteration, indices, valid, stable: (iteration < maximum_iterations) & valid & ~stable,
        step, (tf.constant(0), indices, tf.constant(True), tf.constant(False)),
        maximum_iterations=maximum_iterations, parallel_iterations=1,
    )
    return indices, valid, stable


def teacher_directions(base, base_tangents, physical_valid, prepared, controls, *, setup_static):
    dtype = base.dtype
    recursion = (padded_fixed_teacher_recursion_shape_xla.python_function
                 if setup_static else padded_fixed_teacher_recursion_shape_xla)

    def direction(index):
        return recursion(
            prepared["basis_values"], prepared["active_mask"], prepared["schedule"],
            base, base_tangents[:, :, index], prepared["weights"],
            tf.zeros_like(prepared["weights"]), prepared["initial_cores"],
            tf.zeros_like(prepared["initial_cores"]), prepared["scale_shift_indices"],
            prepared["defensive_weights"], tf.zeros_like(prepared["defensive_weights"]),
            prepared["query_basis_values"], prepared["keep_mask"], prepared["mass_operators"],
            prepared["defensive_marginal_values"], tf.zeros_like(prepared["defensive_marginal_values"]),
            prepared["defensive_mass"], tf.zeros([], dtype), prepared["operator_powers"],
            prepared["defensive_power_moments"], prepared["state_offset"],
            tf.zeros_like(prepared["state_offset"]), prepared["state_matrix"],
            tf.zeros_like(prepared["state_matrix"]), prepared["pair_indices"],
            tf.cast(controls.tt_ridge, dtype), tf.cast(controls.column_scale_floor, dtype),
            tf.cast(controls.condition_number_veto, dtype), tf.cast(controls.fit_residual_veto, dtype),
        )

    first = direction(tf.constant(0))
    count = base_tangents.shape[-1]
    skew = tf.TensorArray(dtype, size=count, element_shape=first[9].shape).write(0, first[9])
    kurtosis = tf.TensorArray(dtype, size=count, element_shape=first[10].shape).write(0, first[10])
    co_skew = tf.TensorArray(dtype, size=count, element_shape=first[11].shape).write(0, first[11])
    co_kurtosis = tf.TensorArray(dtype, size=count, element_shape=first[12].shape).write(0, first[12])

    def step(index, skew, kurtosis, co_skew, co_kurtosis, valid):
        result = direction(index)
        return (index + 1, skew.write(index, result[9]), kurtosis.write(index, result[10]),
                co_skew.write(index, result[11]), co_kurtosis.write(index, result[12]), valid & result[-1])

    _, skew, kurtosis, co_skew, co_kurtosis, valid = tf.while_loop(
        lambda index, *_: index < count, step,
        (tf.constant(1), skew, kurtosis, co_skew, co_kurtosis, first[-1]),
        maximum_iterations=count - 1, parallel_iterations=1,
    )
    return {
        "marginal_values": first[2], "normalizers": first[4], "skew": first[5],
        "kurtosis": first[6], "co_skew": first[7], "co_kurtosis": first[8],
        "skew_tangent": tf.transpose(skew.stack(), [1, 2, 0]),
        "kurtosis_tangent": tf.transpose(kurtosis.stack(), [1, 2, 0]),
        "co_skew_tangent": tf.transpose(co_skew.stack(), [1, 2, 0]),
        "co_kurtosis_tangent": tf.transpose(co_kurtosis.stack(), [1, 2, 0]),
        "valid": physical_valid & valid,
    }
