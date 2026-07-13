from __future__ import annotations

import json
import os
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    batched_annealed_transport_core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float64
annealed_transport_tf.DTYPE = DTYPE
experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
VALUE_ATOL = 1.0e-10
VJP_ATOL = 1.0e-8
JVP_ATOL = 1.0e-8
FIXTURES = ((1, 3, 1), (1, 4, 2), (2, 3, 2))


@pytest.fixture(autouse=True)
def _use_float64_transport_dtype():
    old_transport_dtype = annealed_transport_tf.DTYPE
    old_core_dtype = experimental_batched_ledh_pfpf_ot_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
    yield
    annealed_transport_tf.DTYPE = old_transport_dtype
    experimental_batched_ledh_pfpf_ot_tf.DTYPE = old_core_dtype


def _fixture(batch_size: int, num_particles: int, state_dim: int) -> tuple[tf.Tensor, tf.Tensor]:
    count = batch_size * num_particles * state_dim
    x = tf.reshape(
        tf.linspace(tf.constant(-0.35, DTYPE), tf.constant(0.55, DTYPE), count),
        [batch_size, num_particles, state_dim],
    )
    batch_shift = tf.reshape(
        tf.linspace(tf.constant(0.0, DTYPE), tf.constant(0.08, DTYPE), batch_size),
        [batch_size, 1, 1],
    )
    x = x + batch_shift
    raw_logw = tf.reshape(
        tf.linspace(
            tf.constant(-0.25, DTYPE),
            tf.constant(0.20, DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    logw = raw_logw - tf.reduce_logsumexp(raw_logw, axis=1, keepdims=True)
    return x, logw


def _cotangent(shape: tf.TensorShape | tuple[int, ...], scale: float = 0.07) -> tf.Tensor:
    dims = [int(dim) for dim in tuple(shape)]
    count = 1
    for dim in dims:
        count *= dim
    return tf.reshape(
        tf.linspace(tf.constant(-scale, DTYPE), tf.constant(scale, DTYPE), count),
        dims,
    )


def _max_abs(values: list[tf.Tensor]) -> float:
    if not values:
        return 0.0
    maxima = [tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))) for value in values]
    return float(tf.reduce_max(tf.stack(maxima)).numpy())


def _assert_near(lhs: tf.Tensor, rhs: tf.Tensor, atol: float) -> float:
    delta = tf.reduce_max(tf.abs(tf.convert_to_tensor(lhs, DTYPE) - tf.convert_to_tensor(rhs, DTYPE)))
    assert float(delta.numpy()) <= atol
    return float(delta.numpy())


def _manual_barycentric_vjp(
    transport_matrix: tf.Tensor,
    particles: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    d_transport = tf.einsum("bid,bjd->bij", upstream, particles)
    d_particles = tf.einsum("bij,bid->bjd", transport_matrix, upstream)
    return d_transport, d_particles


def _manual_softmin_vjp(
    epsilon: tf.Tensor,
    cost_matrix: tf.Tensor,
    values: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    epsilon = tf.reshape(tf.cast(epsilon, DTYPE), [-1, 1, 1])
    logits = values[:, None, :] - cost_matrix / epsilon
    probs = tf.nn.softmax(logits, axis=2)
    d_values = -tf.reshape(epsilon, [-1, 1]) * tf.reduce_sum(upstream[:, :, None] * probs, axis=1)
    d_cost = upstream[:, :, None] * probs
    return d_cost, d_values


def _softmin_vjp_fixture(
    batch_size: int,
    num_rows: int,
    num_cols: int,
    state_dim: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    query = tf.reshape(
        tf.linspace(
            tf.constant(-0.45, DTYPE),
            tf.constant(0.35, DTYPE),
            batch_size * num_rows * state_dim,
        ),
        [batch_size, num_rows, state_dim],
    )
    key = tf.reshape(
        tf.linspace(
            tf.constant(0.25, DTYPE),
            tf.constant(-0.30, DTYPE),
            batch_size * num_cols * state_dim,
        ),
        [batch_size, num_cols, state_dim],
    )
    batch_shift = tf.reshape(
        tf.linspace(tf.constant(0.0, DTYPE), tf.constant(0.06, DTYPE), batch_size),
        [batch_size, 1, 1],
    )
    query = query + batch_shift
    key = key - 0.5 * batch_shift
    values = tf.reshape(
        tf.linspace(
            tf.constant(-0.2, DTYPE),
            tf.constant(0.3, DTYPE),
            batch_size * num_cols,
        ),
        [batch_size, num_cols],
    )
    upstream = _cotangent((batch_size, num_rows), scale=0.04)
    epsilon = tf.linspace(
        tf.constant(0.55, DTYPE),
        tf.constant(0.75, DTYPE),
        batch_size,
    )
    return epsilon, query, key, values, upstream


def _dense_softmin_query_key_value_vjp(
    epsilon: tf.Tensor,
    query: tf.Tensor,
    key: tf.Tensor,
    values: tf.Tensor,
    upstream: tf.Tensor,
    *,
    stop_keys: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    cost = 0.5 * annealed_transport_tf._pairwise_squared_cross(query, key)  # noqa: SLF001
    d_cost, d_values = _manual_softmin_vjp(epsilon, cost, values, upstream)
    diff = query[:, :, None, :] - key[:, None, :, :]
    d_query = tf.reduce_sum(d_cost[:, :, :, None] * diff, axis=2)
    if stop_keys:
        d_key = tf.zeros_like(key)
    else:
        d_key = tf.reduce_sum(d_cost[:, :, :, None] * (-diff), axis=1)
    return d_query, d_key, d_values


def _transport_from_potentials(
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
) -> tf.Tensor:
    n = tf.cast(tf.shape(particles)[1], DTYPE)
    return annealed_transport_tf._filterflow_exact_transport_from_potentials(  # noqa: SLF001
        particles,
        f,
        g,
        eps,
        logw,
        n,
    )


def _streaming_transport_from_potentials_value(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
) -> tf.Tensor:
    n = tf.cast(tf.shape(scaled_x)[1], DTYPE)
    transported, _ = annealed_transport_tf._filterflow_streaming_transport_from_potentials(  # noqa: SLF001
        scaled_x,
        particles,
        f,
        g,
        eps,
        logw,
        n,
        row_chunk_size=2,
        col_chunk_size=2,
    )
    return transported


def _manual_transport_from_potentials_vjp(
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    del g  # The column potential cancels in the code-defined column normalization.
    eps_ = tf.reshape(tf.cast(eps, DTYPE), [-1, 1, 1])
    n = tf.cast(tf.shape(particles)[1], DTYPE)
    diff = particles[:, :, None, :] - particles[:, None, :, :]
    cost = 0.5 * tf.reduce_sum(diff * diff, axis=3)
    logits = (f[:, :, None] + tf.zeros_like(cost) - cost) / eps_
    probs = tf.nn.softmax(logits, axis=1)
    transport = tf.exp(logw[:, None, :]) * n * probs
    d_temp = upstream * transport
    d_logits = d_temp - probs * tf.reduce_sum(d_temp, axis=1, keepdims=True)
    d_logw = tf.reduce_sum(d_temp, axis=1)
    d_f = tf.reduce_sum(d_logits / eps_, axis=2)
    d_g = tf.reduce_sum(d_logits / eps_, axis=1)
    d_cost = -d_logits / eps_
    row_part = tf.reduce_sum(d_cost[:, :, :, None] * diff, axis=2)
    col_part = tf.reduce_sum(d_cost[:, :, :, None] * (-diff), axis=1)
    d_particles = row_part + col_part
    return d_particles, d_f, d_g, d_logw


def _manual_streaming_transport_from_potentials_vjp_reference(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    del g
    eps_ = tf.reshape(tf.cast(eps, DTYPE), [-1, 1, 1])
    n = tf.cast(tf.shape(scaled_x)[1], DTYPE)
    diff = scaled_x[:, :, None, :] - scaled_x[:, None, :, :]
    cost = 0.5 * tf.reduce_sum(diff * diff, axis=3)
    logits = (f[:, :, None] - cost) / eps_
    probs = tf.nn.softmax(logits, axis=1)
    transport = tf.exp(logw[:, None, :]) * n * probs
    d_transport = tf.reduce_sum(upstream[:, :, None, :] * particles[:, None, :, :], axis=3)
    weighted = d_transport * transport
    d_logits = weighted - probs * tf.reduce_sum(weighted, axis=1, keepdims=True)
    d_logw = tf.reduce_sum(weighted, axis=1)
    d_f = tf.reduce_sum(d_logits / eps_, axis=2)
    d_g = tf.zeros_like(f)
    d_cost = -d_logits / eps_
    row_part = tf.reduce_sum(d_cost[:, :, :, None] * diff, axis=2)
    col_part = tf.reduce_sum(d_cost[:, :, :, None] * (-diff), axis=1)
    d_scaled_x = row_part + col_part
    d_particles = tf.matmul(transport, upstream, transpose_a=True)
    return d_scaled_x, d_particles, d_f, d_g, d_logw


def _transport_vjp_fixture(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    base = tf.reshape(
        tf.linspace(
            tf.constant(-0.55, DTYPE),
            tf.constant(0.45, DTYPE),
            batch_size * num_particles * state_dim,
        ),
        [batch_size, num_particles, state_dim],
    )
    batch_shift = tf.reshape(
        tf.linspace(tf.constant(0.0, DTYPE), tf.constant(0.05, DTYPE), batch_size),
        [batch_size, 1, 1],
    )
    scaled_x = base + batch_shift
    particles = 0.7 * base + tf.constant(0.11, DTYPE) - 0.25 * batch_shift
    f = tf.reshape(
        tf.linspace(
            tf.constant(-0.18, DTYPE),
            tf.constant(0.23, DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    g = tf.math.sin(
        tf.reshape(
            tf.linspace(
                tf.constant(0.2, DTYPE),
                tf.constant(1.1, DTYPE),
                batch_size * num_particles,
            ),
            [batch_size, num_particles],
        )
    )
    raw_logw = tf.reshape(
        tf.linspace(
            tf.constant(-0.25, DTYPE),
            tf.constant(0.15, DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    logw = raw_logw - tf.reduce_logsumexp(raw_logw, axis=1, keepdims=True)
    upstream = _cotangent((batch_size, num_particles, state_dim), scale=0.035)
    eps = tf.linspace(tf.constant(0.58, DTYPE), tf.constant(0.74, DTYPE), batch_size)
    return scaled_x, particles, f, g, eps, logw, upstream


def _sinkhorn_recursion_fixture(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.reshape(
        tf.linspace(
            tf.constant(-0.35, DTYPE),
            tf.constant(0.55, DTYPE),
            batch_size * num_particles * state_dim,
        ),
        [batch_size, num_particles, state_dim],
    )
    raw_alpha = tf.reshape(
        tf.linspace(
            tf.constant(-0.22, DTYPE),
            tf.constant(0.18, DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    raw_beta = tf.reshape(
        tf.linspace(
            tf.constant(0.16, DTYPE),
            tf.constant(-0.19, DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    log_alpha = raw_alpha - tf.reduce_logsumexp(raw_alpha, axis=1, keepdims=True)
    log_beta = raw_beta - tf.reduce_logsumexp(raw_beta, axis=1, keepdims=True)
    upstream_alpha = _cotangent((batch_size, num_particles), scale=0.041)
    upstream_beta = _cotangent((batch_size, num_particles), scale=0.027)
    epsilon = tf.constant(0.43, DTYPE)
    epsilon0 = tf.linspace(
        tf.constant(0.85, DTYPE),
        tf.constant(0.95, DTYPE),
        batch_size,
    )
    scaling = tf.constant(0.82, DTYPE)
    return log_alpha, log_beta, x, upstream_alpha, upstream_beta, epsilon, epsilon0, scaling


def _stopped_key_cost_vjp_to_x(x: tf.Tensor, d_cost: tf.Tensor) -> tf.Tensor:
    key_x = tf.stop_gradient(x)
    diff = x[:, :, None, :] - key_x[:, None, :, :]
    return tf.reduce_sum(d_cost[:, :, :, None] * diff, axis=2)


def _manual_streaming_sinkhorn_recursion_vjp_reference(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    upstream_alpha: tf.Tensor,
    upstream_beta: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    key_x = tf.stop_gradient(x)
    cost = annealed_transport_tf._filterflow_exact_cost(x, key_x)  # noqa: SLF001
    zero = tf.zeros_like(upstream_alpha)
    (
        d_log_alpha,
        d_log_beta,
        d_cost_xy,
        d_cost_yx,
        d_cost_xx,
        d_cost_yy,
    ) = annealed_transport_tf._filterflow_manual_dense_finite_sinkhorn_vjp(  # noqa: SLF001
        log_alpha,
        log_beta,
        cost,
        cost,
        cost,
        cost,
        (upstream_alpha, upstream_beta, zero, zero),
        epsilon=epsilon,
        epsilon0=epsilon0,
        scaling=scaling,
        steps=steps,
    )
    d_x = tf.add_n(
        [
            _stopped_key_cost_vjp_to_x(x, d_cost)
            for d_cost in (d_cost_xy, d_cost_yx, d_cost_xx, d_cost_yy)
        ]
    )
    return d_log_alpha, d_log_beta, d_x


def _sinkhorn_recursion_custom_gradient_value(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    @tf.custom_gradient
    def _value(
        inner_log_alpha: tf.Tensor,
        inner_log_beta: tf.Tensor,
        inner_x: tf.Tensor,
        inner_epsilon: tf.Tensor,
        inner_epsilon0: tf.Tensor,
        inner_scaling: tf.Tensor,
    ):
        alpha, beta = annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(  # noqa: SLF001
            inner_log_alpha,
            inner_log_beta,
            inner_x,
            inner_epsilon,
            inner_epsilon0,
            inner_scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )

        def grad(d_alpha: tf.Tensor, d_beta: tf.Tensor):
            d_log_alpha, d_log_beta, d_x = (
                annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_vjp_stopped_scale_keys(  # noqa: SLF001
                    inner_log_alpha,
                    inner_log_beta,
                    inner_x,
                    d_alpha,
                    d_beta,
                    inner_epsilon,
                    inner_epsilon0,
                    inner_scaling,
                    steps=steps,
                    row_chunk_size=row_chunk_size,
                    col_chunk_size=col_chunk_size,
                )
            )
            return d_log_alpha, d_log_beta, d_x, None, None, None

        return (alpha, beta), grad

    return _value(log_alpha, log_beta, x, epsilon, epsilon0, scaling)


def _fixed_softmin(epsilon: tf.Tensor, cost_matrix: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    return annealed_transport_tf._filterflow_exact_softmin(  # noqa: SLF001
        tf.cast(epsilon, DTYPE),
        tf.cast(cost_matrix, DTYPE),
        tf.cast(values, DTYPE),
    )


def _fixed_sinkhorn_outputs(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost_xy: tf.Tensor,
    cost_yx: tf.Tensor,
    cost_xx: tf.Tensor,
    cost_yy: tf.Tensor,
    *,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    running = tf.cast(epsilon0, DTYPE)
    eps = tf.cast(epsilon, DTYPE)
    scaling_factor = tf.cast(scaling, DTYPE) ** 2
    a_y = _fixed_softmin(running, cost_yx, log_alpha)
    b_x = _fixed_softmin(running, cost_xy, log_beta)
    a_x = _fixed_softmin(running, cost_xx, log_alpha)
    b_y = _fixed_softmin(running, cost_yy, log_beta)
    for _ in range(steps):
        running_ = tf.reshape(running, [-1, 1])
        at_y = _fixed_softmin(running, cost_yx, log_alpha + b_x / running_)
        bt_x = _fixed_softmin(running, cost_xy, log_beta + a_y / running_)
        at_x = _fixed_softmin(running, cost_xx, log_alpha + a_x / running_)
        bt_y = _fixed_softmin(running, cost_yy, log_beta + b_y / running_)
        a_y = 0.5 * (a_y + at_y)
        b_x = 0.5 * (b_x + bt_x)
        a_x = 0.5 * (a_x + at_x)
        b_y = 0.5 * (b_y + bt_y)
        running = tf.maximum(running * scaling_factor, eps)
    eps_ = tf.reshape(eps, [-1, 1])
    return (
        _fixed_softmin(eps, cost_yx, log_alpha + b_x / eps_),
        _fixed_softmin(eps, cost_xy, log_beta + a_y / eps_),
        _fixed_softmin(eps, cost_xx, log_alpha + a_x / eps_),
        _fixed_softmin(eps, cost_yy, log_beta + b_y / eps_),
    )


def _manual_fixed_sinkhorn_vjp(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost_xy: tf.Tensor,
    cost_yx: tf.Tensor,
    cost_xx: tf.Tensor,
    cost_yy: tf.Tensor,
    upstreams: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    *,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    running = tf.cast(epsilon0, DTYPE)
    eps = tf.cast(epsilon, DTYPE)
    scaling_factor = tf.cast(scaling, DTYPE) ** 2
    a_y = _fixed_softmin(running, cost_yx, log_alpha)
    b_x = _fixed_softmin(running, cost_xy, log_beta)
    a_x = _fixed_softmin(running, cost_xx, log_alpha)
    b_y = _fixed_softmin(running, cost_yy, log_beta)
    states = []
    for _ in range(steps):
        states.append((running, a_y, b_x, a_x, b_y))
        running_ = tf.reshape(running, [-1, 1])
        at_y = _fixed_softmin(running, cost_yx, log_alpha + b_x / running_)
        bt_x = _fixed_softmin(running, cost_xy, log_beta + a_y / running_)
        at_x = _fixed_softmin(running, cost_xx, log_alpha + a_x / running_)
        bt_y = _fixed_softmin(running, cost_yy, log_beta + b_y / running_)
        a_y = 0.5 * (a_y + at_y)
        b_x = 0.5 * (b_x + bt_x)
        a_x = 0.5 * (a_x + at_x)
        b_y = 0.5 * (b_y + bt_y)
        running = tf.maximum(running * scaling_factor, eps)

    d_log_alpha = tf.zeros_like(log_alpha)
    d_log_beta = tf.zeros_like(log_beta)
    d_cost_xy = tf.zeros_like(cost_xy)
    d_cost_yx = tf.zeros_like(cost_yx)
    d_cost_xx = tf.zeros_like(cost_xx)
    d_cost_yy = tf.zeros_like(cost_yy)
    eps_ = tf.reshape(eps, [-1, 1])

    d_a_y = tf.zeros_like(log_alpha)
    d_b_x = tf.zeros_like(log_beta)
    d_a_x = tf.zeros_like(log_alpha)
    d_b_y = tf.zeros_like(log_beta)

    dc, dv = _manual_softmin_vjp(eps, cost_yx, log_alpha + b_x / eps_, upstreams[0])
    d_cost_yx += dc
    d_log_alpha += dv
    d_b_x += dv / eps_
    dc, dv = _manual_softmin_vjp(eps, cost_xy, log_beta + a_y / eps_, upstreams[1])
    d_cost_xy += dc
    d_log_beta += dv
    d_a_y += dv / eps_
    dc, dv = _manual_softmin_vjp(eps, cost_xx, log_alpha + a_x / eps_, upstreams[2])
    d_cost_xx += dc
    d_log_alpha += dv
    d_a_x += dv / eps_
    dc, dv = _manual_softmin_vjp(eps, cost_yy, log_beta + b_y / eps_, upstreams[3])
    d_cost_yy += dc
    d_log_beta += dv
    d_b_y += dv / eps_

    for running, old_a_y, old_b_x, old_a_x, old_b_y in reversed(states):
        running_ = tf.reshape(running, [-1, 1])
        bar_at_y = 0.5 * d_a_y
        bar_bt_x = 0.5 * d_b_x
        bar_at_x = 0.5 * d_a_x
        bar_bt_y = 0.5 * d_b_y

        old_d_a_y = 0.5 * d_a_y
        old_d_b_x = 0.5 * d_b_x
        old_d_a_x = 0.5 * d_a_x
        old_d_b_y = 0.5 * d_b_y

        dc, dv = _manual_softmin_vjp(running, cost_yx, log_alpha + old_b_x / running_, bar_at_y)
        d_cost_yx += dc
        d_log_alpha += dv
        old_d_b_x += dv / running_
        dc, dv = _manual_softmin_vjp(running, cost_xy, log_beta + old_a_y / running_, bar_bt_x)
        d_cost_xy += dc
        d_log_beta += dv
        old_d_a_y += dv / running_
        dc, dv = _manual_softmin_vjp(running, cost_xx, log_alpha + old_a_x / running_, bar_at_x)
        d_cost_xx += dc
        d_log_alpha += dv
        old_d_a_x += dv / running_
        dc, dv = _manual_softmin_vjp(running, cost_yy, log_beta + old_b_y / running_, bar_bt_y)
        d_cost_yy += dc
        d_log_beta += dv
        old_d_b_y += dv / running_

        d_a_y, d_b_x, d_a_x, d_b_y = old_d_a_y, old_d_b_x, old_d_a_x, old_d_b_y

    dc, dv = _manual_softmin_vjp(epsilon0, cost_yy, log_beta, d_b_y)
    d_cost_yy += dc
    d_log_beta += dv
    dc, dv = _manual_softmin_vjp(epsilon0, cost_xx, log_alpha, d_a_x)
    d_cost_xx += dc
    d_log_alpha += dv
    dc, dv = _manual_softmin_vjp(epsilon0, cost_xy, log_beta, d_b_x)
    d_cost_xy += dc
    d_log_beta += dv
    dc, dv = _manual_softmin_vjp(epsilon0, cost_yx, log_alpha, d_a_y)
    d_cost_yx += dc
    d_log_alpha += dv

    return d_log_alpha, d_log_beta, d_cost_xy, d_cost_yx, d_cost_xx, d_cost_yy


def _loop_fixture(batch_size: int, num_particles: int) -> tuple[tf.Tensor, ...]:
    grid = tf.reshape(
        tf.linspace(tf.constant(-0.4, DTYPE), tf.constant(0.6, DTYPE), batch_size * num_particles),
        [batch_size, num_particles],
    )
    log_alpha = grid - tf.reduce_logsumexp(grid, axis=1, keepdims=True)
    log_beta = -0.3 * grid - tf.reduce_logsumexp(-0.3 * grid, axis=1, keepdims=True)
    ids = tf.cast(tf.range(num_particles), DTYPE)
    base = 0.07 + 0.5 * tf.square(ids[:, None] - ids[None, :])
    cost_xy = tf.tile(base[None, :, :], [batch_size, 1, 1])
    cost_yx = cost_xy + tf.reshape(tf.linspace(tf.constant(0.01, DTYPE), tf.constant(0.03, DTYPE), num_particles), [1, -1, 1])
    cost_xx = cost_xy + tf.constant(0.04, DTYPE)
    cost_yy = cost_xy + tf.constant(0.08, DTYPE)
    return log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy


def _finite_difference_residual(
    scalar_fn,
    primals: tuple[tf.Tensor, ...],
    tangents: tuple[tf.Tensor, ...],
    grad_dot_tangent: tf.Tensor,
) -> float:
    errors = []
    for step in (1.0e-3, 5.0e-4, 2.5e-4):
        h = tf.constant(step, DTYPE)
        plus = tuple(p + h * t for p, t in zip(primals, tangents, strict=True))
        minus = tuple(p - h * t for p, t in zip(primals, tangents, strict=True))
        fd = (scalar_fn(*plus) - scalar_fn(*minus)) / (2.0 * h)
        errors.append(tf.abs(fd - grad_dot_tangent))
    return float(tf.reduce_min(tf.stack(errors)).numpy())


def _diagnostics_for_fixture(batch_size: int, num_particles: int, state_dim: int) -> dict[str, float]:
    particles, logw = _fixture(batch_size, num_particles, state_dim)
    transport = tf.nn.softmax(
        tf.reshape(
            tf.linspace(tf.constant(-0.3, DTYPE), tf.constant(0.4, DTYPE), batch_size * num_particles * num_particles),
            [batch_size, num_particles, num_particles],
        ),
        axis=2,
    )
    upstream_particles = _cotangent((batch_size, num_particles, state_dim), scale=0.09)
    with tf.GradientTape() as tape:
        tape.watch([transport, particles])
        transported = tf.linalg.matmul(transport, particles)
        scalar = tf.reduce_sum(transported * upstream_particles)
    ref_d_transport, ref_d_particles = tape.gradient(scalar, [transport, particles])
    man_d_transport, man_d_particles = _manual_barycentric_vjp(
        transport,
        particles,
        upstream_particles,
    )
    barycentric_error = max(
        _assert_near(man_d_transport, ref_d_transport, VJP_ATOL),
        _assert_near(man_d_particles, ref_d_particles, VJP_ATOL),
    )

    eps = tf.fill([batch_size], tf.constant(0.7, DTYPE))
    cost = annealed_transport_tf._filterflow_exact_cost(particles, particles)  # noqa: SLF001
    upstream_soft = _cotangent((batch_size, num_particles), scale=0.05)
    with tf.GradientTape() as tape:
        tape.watch([cost, logw])
        soft = _fixed_softmin(eps, cost, logw)
        scalar = tf.reduce_sum(soft * upstream_soft)
    ref_d_cost, ref_d_logw = tape.gradient(scalar, [cost, logw])
    man_d_cost, man_d_logw = _manual_softmin_vjp(eps, cost, logw, upstream_soft)
    softmin_error = max(
        _assert_near(man_d_cost, ref_d_cost, VJP_ATOL),
        _assert_near(man_d_logw, ref_d_logw, VJP_ATOL),
    )

    f = _cotangent((batch_size, num_particles), scale=0.04)
    g = _cotangent((batch_size, num_particles), scale=0.03)
    eps_transport = tf.constant(0.7, DTYPE)
    upstream_transport = _cotangent((batch_size, num_particles, num_particles), scale=0.02)
    with tf.GradientTape() as tape:
        tape.watch([particles, f, g, logw])
        matrix = _transport_from_potentials(particles, f, g, eps_transport, logw)
        scalar = tf.reduce_sum(matrix * upstream_transport)
    ref_d_particles, ref_d_f, ref_d_g, ref_d_logw = tape.gradient(
        scalar,
        [particles, f, g, logw],
    )
    man_d_particles, man_d_f, man_d_g, man_d_logw = _manual_transport_from_potentials_vjp(
        particles,
        f,
        g,
        eps_transport,
        logw,
        upstream_transport,
    )
    transport_error = max(
        _assert_near(man_d_particles, ref_d_particles, VJP_ATOL),
        _assert_near(man_d_f, ref_d_f, VJP_ATOL),
        _assert_near(man_d_g, ref_d_g, VJP_ATOL),
        _assert_near(man_d_logw, ref_d_logw, VJP_ATOL),
    )

    (
        log_alpha,
        log_beta,
        cost_xy,
        cost_yx,
        cost_xx,
        cost_yy,
    ) = _loop_fixture(batch_size, num_particles)
    epsilon0 = tf.fill([batch_size], tf.constant(0.9, DTYPE))
    epsilon = tf.fill([batch_size], tf.constant(0.45, DTYPE))
    scaling = tf.constant(0.8, DTYPE)
    steps = 2
    with tf.GradientTape() as tape:
        tape.watch([log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy])
        outputs = _fixed_sinkhorn_outputs(
            log_alpha,
            log_beta,
            cost_xy,
            cost_yx,
            cost_xx,
            cost_yy,
            epsilon=epsilon,
            epsilon0=epsilon0,
            scaling=scaling,
            steps=steps,
        )
        upstreams = tuple(_cotangent(output.shape, scale=0.06) for output in outputs)
        scalar = tf.add_n([tf.reduce_sum(output * upstream) for output, upstream in zip(outputs, upstreams, strict=True)])
    ref_loop = tape.gradient(scalar, [log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy])
    man_loop = _manual_fixed_sinkhorn_vjp(
        log_alpha,
        log_beta,
        cost_xy,
        cost_yx,
        cost_xx,
        cost_yy,
        upstreams,
        epsilon=epsilon,
        epsilon0=epsilon0,
        scaling=scaling,
        steps=steps,
    )
    loop_error = max(_assert_near(manual, reference, VJP_ATOL) for manual, reference in zip(man_loop, ref_loop, strict=True))

    tangents = tuple(_cotangent(tensor.shape, scale=0.01) for tensor in (log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy))
    with tf.autodiff.ForwardAccumulator(
        (log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy),
        tangents,
    ) as acc:
        outputs = _fixed_sinkhorn_outputs(
            log_alpha,
            log_beta,
            cost_xy,
            cost_yx,
            cost_xx,
            cost_yy,
            epsilon=epsilon,
            epsilon0=epsilon0,
            scaling=scaling,
            steps=steps,
        )
        scalar = tf.add_n([tf.reduce_sum(output * upstream) for output, upstream in zip(outputs, upstreams, strict=True)])
    jvp = acc.jvp(scalar)
    vjp_dot = tf.add_n([tf.reduce_sum(grad * tangent) for grad, tangent in zip(man_loop, tangents, strict=True)])
    jvp_error = _assert_near(jvp, vjp_dot, JVP_ATOL)

    def loop_scalar(*args: tf.Tensor) -> tf.Tensor:
        outs = _fixed_sinkhorn_outputs(
            *args,
            epsilon=epsilon,
            epsilon0=epsilon0,
            scaling=scaling,
            steps=steps,
        )
        return tf.add_n([tf.reduce_sum(output * upstream) for output, upstream in zip(outs, upstreams, strict=True)])

    fd_residual = _finite_difference_residual(
        loop_scalar,
        (log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy),
        tangents,
        vjp_dot,
    )

    return {
        "barycentric_vjp_max_abs_error": barycentric_error,
        "softmin_vjp_max_abs_error": softmin_error,
        "transport_from_potentials_vjp_max_abs_error": transport_error,
        "finite_sinkhorn_loop_vjp_max_abs_error": loop_error,
        "finite_sinkhorn_loop_jvp_vjp_error": jvp_error,
        "finite_difference_min_residual": fd_residual,
    }


def _dense_custom_gradient_diagnostics(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> dict[str, float]:
    particles, logw = _fixture(batch_size, num_particles, state_dim)
    eps = tf.constant(0.45, DTYPE)
    epsilon0 = tf.fill([batch_size], tf.constant(0.9, DTYPE))
    scaling = tf.constant(0.8, DTYPE)
    steps = 2
    upstream = _cotangent((batch_size, num_particles, num_particles), scale=0.025)
    raw_value = (
        annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(  # noqa: SLF001
            particles,
            logw,
            eps,
            epsilon0,
            scaling,
            steps=steps,
        )
    )
    custom_value = (
        annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys(  # noqa: SLF001
            particles,
            logw,
            eps,
            epsilon0,
            scaling,
            steps=steps,
        )
    )
    value_error = _assert_near(custom_value, raw_value, VALUE_ATOL)

    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        raw_matrix = (
            annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(  # noqa: SLF001
                particles,
                logw,
                eps,
                epsilon0,
                scaling,
                steps=steps,
            )
        )
        raw_scalar = tf.reduce_sum(raw_matrix * upstream)
    raw_dx, raw_dlogw = tape.gradient(raw_scalar, [particles, logw])

    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        custom_matrix = (
            annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys(  # noqa: SLF001
                particles,
                logw,
                eps,
                epsilon0,
                scaling,
                steps=steps,
            )
        )
        custom_scalar = tf.reduce_sum(custom_matrix * upstream)
    custom_dx, custom_dlogw = tape.gradient(custom_scalar, [particles, logw])

    gradient_error = max(
        _assert_near(custom_dx, raw_dx, VJP_ATOL),
        _assert_near(custom_dlogw, raw_dlogw, VJP_ATOL),
    )
    finite_value = bool(tf.reduce_all(tf.math.is_finite(custom_value)).numpy())
    finite_grad = bool(
        tf.reduce_all(tf.math.is_finite(custom_dx)).numpy()
        and tf.reduce_all(tf.math.is_finite(custom_dlogw)).numpy()
    )
    assert finite_value
    assert finite_grad
    return {
        "dense_custom_value_max_abs_error": value_error,
        "dense_custom_gradient_max_abs_error": gradient_error,
        "dense_custom_value_finite": float(finite_value),
        "dense_custom_gradient_finite": float(finite_grad),
    }


def _scaled_particles_for_core(particles: tf.Tensor) -> tf.Tensor:
    center = tf.reduce_mean(particles, axis=1, keepdims=True)
    centered = particles - tf.stop_gradient(center)
    scale = annealed_transport_tf._filterflow_scale(particles)  # noqa: SLF001
    return centered / tf.stop_gradient(scale)[:, None, None]


def _m5_manual_core_diagnostics(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> dict[str, float]:
    particles, logw = _fixture(batch_size, num_particles, state_dim)
    eps = tf.constant(0.45, DTYPE)
    scaling = tf.constant(0.8, DTYPE)
    steps = 2
    mask = tf.ones([batch_size], dtype=tf.bool)
    result = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=eps,
        scaling=scaling,
        max_iterations=steps,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    scaled = _scaled_particles_for_core(particles)
    epsilon0 = tf.stop_gradient(annealed_transport_tf._filterflow_epsilon_start(scaled))  # noqa: SLF001
    reference = (
        annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(  # noqa: SLF001
            scaled,
            logw,
            eps,
            epsilon0,
            scaling,
            steps=steps,
        )
    )
    value_error = _assert_near(result.transport_matrix, reference, VALUE_ATOL)

    upstream = _cotangent((batch_size, num_particles, num_particles), scale=0.025)
    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        raw_scaled = _scaled_particles_for_core(particles)
        raw_epsilon0 = tf.stop_gradient(
            annealed_transport_tf._filterflow_epsilon_start(raw_scaled)  # noqa: SLF001
        )
        raw_matrix = (
            annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(  # noqa: SLF001
                raw_scaled,
                logw,
                eps,
                raw_epsilon0,
                scaling,
                steps=steps,
            )
        )
        raw_scalar = tf.reduce_sum(raw_matrix * upstream)
    raw_dx, raw_dlogw = tape.gradient(raw_scalar, [particles, logw])

    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        custom_result = batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=eps,
            scaling=scaling,
            max_iterations=steps,
            transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        )
        custom_scalar = tf.reduce_sum(custom_result.transport_matrix * upstream)
    custom_dx, custom_dlogw = tape.gradient(custom_scalar, [particles, logw])
    gradient_error = max(
        _assert_near(custom_dx, raw_dx, VJP_ATOL),
        _assert_near(custom_dlogw, raw_dlogw, VJP_ATOL),
    )
    return {
        "m5_manual_core_transport_value_max_abs_error": value_error,
        "m5_manual_core_transport_gradient_max_abs_error": gradient_error,
    }


def _m6_manual_streaming_diagnostics(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> dict[str, float]:
    particles, logw = _fixture(batch_size, num_particles, state_dim)
    eps = tf.constant(0.45, DTYPE)
    scaling = tf.constant(0.8, DTYPE)
    steps = 2
    mask = tf.ones([batch_size], dtype=tf.bool)
    dense = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=eps,
        scaling=scaling,
        max_iterations=steps,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    streaming = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=eps,
        scaling=scaling,
        max_iterations=steps,
        transport_gradient_mode=MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_plan_mode="streaming",
        row_chunk_size=2,
        col_chunk_size=2,
    )
    value_error = _assert_near(streaming.particles, dense.particles, VALUE_ATOL)
    assert streaming.transport_matrix.shape == (batch_size, 0, 0)

    upstream = _cotangent((batch_size, num_particles, state_dim), scale=0.025)
    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        dense_result = batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=eps,
            scaling=scaling,
            max_iterations=steps,
            transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        )
        dense_scalar = tf.reduce_sum(dense_result.particles * upstream)
    dense_dx, dense_dlogw = tape.gradient(dense_scalar, [particles, logw])

    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        streaming_result = batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=eps,
            scaling=scaling,
            max_iterations=steps,
            transport_gradient_mode=MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            transport_plan_mode="streaming",
            row_chunk_size=2,
            col_chunk_size=2,
        )
        streaming_scalar = tf.reduce_sum(streaming_result.particles * upstream)
    streaming_dx, streaming_dlogw = tape.gradient(streaming_scalar, [particles, logw])
    gradient_error = max(
        _assert_near(streaming_dx, dense_dx, VJP_ATOL),
        _assert_near(streaming_dlogw, dense_dlogw, VJP_ATOL),
    )
    return {
        "m6_manual_streaming_particle_value_max_abs_error": value_error,
        "m6_manual_streaming_particle_gradient_max_abs_error": gradient_error,
        "m6_manual_streaming_transport_matrix_size": float(tf.size(streaming.transport_matrix).numpy()),
    }


def _blockwise_route_diagnostics(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> dict[str, float]:
    particles, logw = _fixture(batch_size, num_particles, state_dim)
    eps = tf.constant(0.45, DTYPE)
    scaling = tf.constant(0.8, DTYPE)
    steps = 2
    mask = tf.ones([batch_size], dtype=tf.bool)
    blockwise = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=eps,
        scaling=scaling,
        max_iterations=steps,
        transport_gradient_mode=MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_plan_mode="streaming",
        row_chunk_size=2,
        col_chunk_size=2,
    )
    old_streaming = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=eps,
        scaling=scaling,
        max_iterations=steps,
        transport_gradient_mode=MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_plan_mode="streaming",
        row_chunk_size=2,
        col_chunk_size=2,
    )
    dense = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=eps,
        scaling=scaling,
        max_iterations=steps,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    value_error = max(
        _assert_near(blockwise.particles, old_streaming.particles, VALUE_ATOL),
        _assert_near(blockwise.particles, dense.particles, VALUE_ATOL),
    )
    upstream = _cotangent((batch_size, num_particles, state_dim), scale=0.025)
    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        blockwise_result = batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=eps,
            scaling=scaling,
            max_iterations=steps,
            transport_gradient_mode=MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
            transport_plan_mode="streaming",
            row_chunk_size=2,
            col_chunk_size=2,
        )
        blockwise_scalar = tf.reduce_sum(blockwise_result.particles * upstream)
    blockwise_dx, blockwise_dlogw = tape.gradient(blockwise_scalar, [particles, logw])

    with tf.GradientTape() as tape:
        tape.watch([particles, logw])
        dense_result = batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=eps,
            scaling=scaling,
            max_iterations=steps,
            transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        )
        dense_scalar = tf.reduce_sum(dense_result.particles * upstream)
    dense_dx, dense_dlogw = tape.gradient(dense_scalar, [particles, logw])
    gradient_error = max(
        _assert_near(blockwise_dx, dense_dx, VJP_ATOL),
        _assert_near(blockwise_dlogw, dense_dlogw, VJP_ATOL),
    )
    return {
        "blockwise_route_value_max_abs_error": value_error,
        "blockwise_route_gradient_max_abs_error": gradient_error,
        "blockwise_route_transport_matrix_size": float(tf.size(blockwise.transport_matrix).numpy()),
        "blockwise_route_old_streaming_transport_matrix_size": float(tf.size(old_streaming.transport_matrix).numpy()),
    }


def _streaming_softmin_vjp_diagnostics(
    *,
    batch_size: int,
    num_rows: int,
    num_cols: int,
    state_dim: int,
    row_chunk_size: int,
    col_chunk_size: int,
    stop_keys: bool,
) -> dict[str, float]:
    epsilon, query, key, values, upstream = _softmin_vjp_fixture(
        batch_size,
        num_rows,
        num_cols,
        state_dim,
    )
    dense = _dense_softmin_query_key_value_vjp(
        epsilon,
        query,
        key,
        values,
        upstream,
        stop_keys=stop_keys,
    )
    streaming = annealed_transport_tf._filterflow_streaming_softmin_vjp(  # noqa: SLF001
        epsilon,
        query,
        key,
        values,
        upstream,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=stop_keys,
    )

    with tf.GradientTape() as tape:
        tape.watch([query, key, values])
        key_for_forward = tf.stop_gradient(key) if stop_keys else key
        soft = annealed_transport_tf._filterflow_streaming_softmin(  # noqa: SLF001
            epsilon,
            query,
            key_for_forward,
            values,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        scalar = tf.reduce_sum(soft * upstream)
    autodiff = tape.gradient(
        scalar,
        [query, key, values],
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    dense_error = max(
        _assert_near(streaming_part, dense_part, VJP_ATOL)
        for streaming_part, dense_part in zip(streaming, dense, strict=True)
    )
    autodiff_error = max(
        _assert_near(streaming_part, autodiff_part, VJP_ATOL)
        for streaming_part, autodiff_part in zip(streaming, autodiff, strict=True)
    )
    stopped_key_leakage = float(tf.reduce_max(tf.abs(streaming[1])).numpy()) if stop_keys else 0.0
    finite = float(
        tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(part))
                    for part in streaming
                ]
            )
        ).numpy()
    )
    return {
        "streaming_softmin_vjp_dense_max_abs_error": dense_error,
        "streaming_softmin_vjp_autodiff_max_abs_error": autodiff_error,
        "streaming_softmin_vjp_stopped_key_leakage": stopped_key_leakage,
        "streaming_softmin_vjp_finite": finite,
    }


def _streaming_same_points_softmin_vjp_diagnostics(
    *,
    batch_size: int,
    num_particles: int,
    state_dim: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, float]:
    epsilon, query, _key, _fixture_values, _fixture_upstream = _softmin_vjp_fixture(
        batch_size,
        num_particles,
        num_particles,
        state_dim,
    )
    points = query
    values = tf.reshape(
        tf.linspace(
            tf.constant(-0.17, DTYPE),
            tf.constant(0.23, DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    upstream = _cotangent((batch_size, num_particles), scale=0.035)
    generic = annealed_transport_tf._filterflow_streaming_softmin_vjp(  # noqa: SLF001
        epsilon,
        points,
        points,
        values,
        upstream,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=False,
        return_epsilon=True,
    )
    same_points = annealed_transport_tf._filterflow_streaming_same_points_softmin_vjp(  # noqa: SLF001
        epsilon,
        points,
        values,
        upstream,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    generic_d_points = generic[0] + generic[1]
    dense = _dense_softmin_query_key_value_vjp(
        epsilon,
        points,
        points,
        values,
        upstream,
        stop_keys=False,
    )
    dense_d_points = dense[0] + dense[1]
    generic_error = max(
        _assert_near(same_points[0], generic_d_points, VJP_ATOL),
        _assert_near(same_points[1], generic[2], VJP_ATOL),
        _assert_near(same_points[2], generic[3], VJP_ATOL),
    )
    dense_error = max(
        _assert_near(same_points[0], dense_d_points, VJP_ATOL),
        _assert_near(same_points[1], dense[2], VJP_ATOL),
    )
    finite = float(
        tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(part))
                    for part in same_points
                ]
            )
        ).numpy()
    )
    return {
        "streaming_same_points_softmin_vjp_generic_max_abs_error": generic_error,
        "streaming_same_points_softmin_vjp_dense_max_abs_error": dense_error,
        "streaming_same_points_softmin_vjp_finite": finite,
    }


def _streaming_transport_from_potentials_vjp_diagnostics(
    *,
    batch_size: int,
    num_particles: int,
    state_dim: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, float]:
    scaled_x, particles, f, g, eps, logw, upstream = _transport_vjp_fixture(
        batch_size,
        num_particles,
        state_dim,
    )
    float_n = tf.cast(num_particles, DTYPE)
    streaming = annealed_transport_tf._filterflow_streaming_transport_from_potentials_vjp(  # noqa: SLF001
        scaled_x,
        particles,
        f,
        g,
        eps,
        logw,
        float_n,
        upstream,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    manual = _manual_streaming_transport_from_potentials_vjp_reference(
        scaled_x,
        particles,
        f,
        g,
        eps,
        logw,
        upstream,
    )
    with tf.GradientTape() as tape:
        tape.watch([scaled_x, particles, f, g, logw])
        transported, _ = annealed_transport_tf._filterflow_streaming_transport_from_potentials(  # noqa: SLF001
            scaled_x,
            particles,
            f,
            g,
            eps,
            logw,
            float_n,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        scalar = tf.reduce_sum(transported * upstream)
    autodiff = tape.gradient(
        scalar,
        [scaled_x, particles, f, g, logw],
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    manual_error = max(
        _assert_near(streaming_part, manual_part, VJP_ATOL)
        for streaming_part, manual_part in zip(streaming, manual, strict=True)
    )
    autodiff_error = max(
        _assert_near(streaming_part, autodiff_part, VJP_ATOL)
        for streaming_part, autodiff_part in zip(streaming, autodiff, strict=True)
    )
    g_is_nontrivial = float(tf.reduce_max(tf.abs(g - tf.reduce_mean(g, axis=1, keepdims=True))).numpy())
    value_shift = float(tf.reduce_max(tf.abs(scaled_x - particles)).numpy())
    dg_norm = float(tf.reduce_max(tf.abs(streaming[3])).numpy())
    finite = float(
        tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(part))
                    for part in streaming
                ]
            )
        ).numpy()
    )
    return {
        "streaming_transport_from_potentials_vjp_manual_max_abs_error": manual_error,
        "streaming_transport_from_potentials_vjp_autodiff_max_abs_error": autodiff_error,
        "streaming_transport_from_potentials_vjp_dg_max_abs": dg_norm,
        "streaming_transport_from_potentials_vjp_g_nontrivial": g_is_nontrivial,
        "streaming_transport_from_potentials_vjp_scaled_x_particles_gap": value_shift,
        "streaming_transport_from_potentials_vjp_finite": finite,
    }


def _streaming_sinkhorn_recursion_vjp_diagnostics(
    *,
    batch_size: int,
    num_particles: int,
    state_dim: int,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, float]:
    (
        log_alpha,
        log_beta,
        x,
        upstream_alpha,
        upstream_beta,
        epsilon,
        epsilon0,
        scaling,
    ) = _sinkhorn_recursion_fixture(batch_size, num_particles, state_dim)
    streaming = annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_vjp_stopped_scale_keys(  # noqa: SLF001
        log_alpha,
        log_beta,
        x,
        upstream_alpha,
        upstream_beta,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    manual = _manual_streaming_sinkhorn_recursion_vjp_reference(
        log_alpha,
        log_beta,
        x,
        upstream_alpha,
        upstream_beta,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
    )
    manual_error = max(
        _assert_near(streaming_part, manual_part, VJP_ATOL)
        for streaming_part, manual_part in zip(streaming, manual, strict=True)
    )
    with tf.GradientTape() as tape:
        tape.watch([log_alpha, log_beta, x])
        alpha, beta = annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(  # noqa: SLF001
            log_alpha,
            log_beta,
            x,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        scalar = tf.reduce_sum(alpha * upstream_alpha) + tf.reduce_sum(beta * upstream_beta)
    autodiff = tape.gradient(
        scalar,
        [log_alpha, log_beta, x],
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    autodiff_error = max(
        _assert_near(streaming_part, autodiff_part, VJP_ATOL)
        for streaming_part, autodiff_part in zip(streaming, autodiff, strict=True)
    )
    tangents = (
        _cotangent(log_alpha.shape, scale=0.012),
        _cotangent(log_beta.shape, scale=0.011),
        _cotangent(x.shape, scale=0.010),
    )
    with tf.autodiff.ForwardAccumulator(
        (log_alpha, log_beta, x),
        tangents,
    ) as acc:
        alpha, beta = annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(  # noqa: SLF001
            log_alpha,
            log_beta,
            x,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        directional_scalar = tf.reduce_sum(alpha * upstream_alpha) + tf.reduce_sum(beta * upstream_beta)
    jvp = acc.jvp(directional_scalar)
    vjp_dot = tf.add_n(
        [
            tf.reduce_sum(grad * tangent)
            for grad, tangent in zip(streaming, tangents, strict=True)
        ]
    )
    directional_error = _assert_near(jvp, vjp_dot, JVP_ATOL)
    with tf.GradientTape() as tape:
        tape.watch([log_alpha, log_beta, x, epsilon, epsilon0, scaling])
        alpha, beta = _sinkhorn_recursion_custom_gradient_value(
            log_alpha,
            log_beta,
            x,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        custom_scalar = tf.reduce_sum(alpha * upstream_alpha) + tf.reduce_sum(beta * upstream_beta)
    custom_grads = tape.gradient(
        custom_scalar,
        [log_alpha, log_beta, x, epsilon, epsilon0, scaling],
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    custom_error = max(
        _assert_near(custom_part, streaming_part, VJP_ATOL)
        for custom_part, streaming_part in zip(custom_grads[:3], streaming, strict=True)
    )
    stopped_scale_leakage = _max_abs(custom_grads[3:])
    finite = float(
        tf.reduce_all(
            tf.stack(
                [
                    tf.reduce_all(tf.math.is_finite(part))
                    for part in streaming
                ]
            )
        ).numpy()
    )
    return {
        "streaming_sinkhorn_recursion_vjp_manual_max_abs_error": manual_error,
        "streaming_sinkhorn_recursion_vjp_autodiff_max_abs_error": autodiff_error,
        "streaming_sinkhorn_recursion_vjp_directional_error": directional_error,
        "streaming_sinkhorn_recursion_vjp_custom_gradient_max_abs_error": custom_error,
        "streaming_sinkhorn_recursion_vjp_stopped_scale_leakage": stopped_scale_leakage,
        "streaming_sinkhorn_recursion_vjp_finite": finite,
    }


def collect_diagnostics() -> dict[str, Any]:
    fixture_results = {}
    custom_results = {}
    m5_results = {}
    m6_results = {}
    for batch_size, num_particles, state_dim in FIXTURES:
        fixture_results[f"B{batch_size}_N{num_particles}_D{state_dim}"] = _diagnostics_for_fixture(
            batch_size,
            num_particles,
            state_dim,
        )
        custom_results[f"B{batch_size}_N{num_particles}_D{state_dim}"] = (
            _dense_custom_gradient_diagnostics(batch_size, num_particles, state_dim)
        )
        m5_results[f"B{batch_size}_N{num_particles}_D{state_dim}"] = (
            _m5_manual_core_diagnostics(batch_size, num_particles, state_dim)
        )
        m6_results[f"B{batch_size}_N{num_particles}_D{state_dim}"] = (
            _m6_manual_streaming_diagnostics(batch_size, num_particles, state_dim)
        )
    maxima: dict[str, float] = {}
    for row in (
        list(fixture_results.values())
        + list(custom_results.values())
        + list(m5_results.values())
        + list(m6_results.values())
    ):
        for key, value in row.items():
            maxima[key] = max(maxima.get(key, 0.0), float(value))
    return {
        "route": "manual_dense_finite_sinkhorn_stopped_scale_keys",
        "dtype": "float64",
        "execution_class": "cpu_oracle_style_tiny_dense",
        "fixtures": fixture_results,
        "dense_custom_gradient_fixtures": custom_results,
        "m5_manual_core_fixtures": m5_results,
        "m6_manual_streaming_fixtures": m6_results,
        "maxima": maxima,
        "tolerances": {
            "value_atol": VALUE_ATOL,
            "vjp_atol": VJP_ATOL,
            "jvp_atol": JVP_ATOL,
            "finite_difference": "explanatory_only",
        },
        "nonclaims": [
            "no_gpu_tf32_evidence",
            "no_streaming_memory_claim",
            "no_p82_validation",
            "no_hmc_or_default_readiness",
        ],
    }


@pytest.mark.parametrize("batch_size,num_particles,state_dim", FIXTURES)
def test_manual_adjoint_primitives_match_tiny_autodiff_oracle(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> None:
    diagnostics = _diagnostics_for_fixture(batch_size, num_particles, state_dim)

    assert diagnostics["barycentric_vjp_max_abs_error"] <= VJP_ATOL
    assert diagnostics["softmin_vjp_max_abs_error"] <= VJP_ATOL
    assert diagnostics["transport_from_potentials_vjp_max_abs_error"] <= VJP_ATOL
    assert diagnostics["finite_sinkhorn_loop_vjp_max_abs_error"] <= VJP_ATOL
    assert diagnostics["finite_sinkhorn_loop_jvp_vjp_error"] <= JVP_ATOL
    assert diagnostics["finite_difference_min_residual"] < 1.0e-8


@pytest.mark.parametrize(
    "case",
    [
        {
            "batch_size": 1,
            "num_rows": 4,
            "num_cols": 4,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 2,
            "stop_keys": False,
        },
        {
            "batch_size": 2,
            "num_rows": 5,
            "num_cols": 3,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 2,
            "stop_keys": False,
        },
        {
            "batch_size": 2,
            "num_rows": 5,
            "num_cols": 3,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 2,
            "stop_keys": True,
        },
    ],
)
def test_streaming_softmin_vjp_matches_dense_and_tiny_autodiff(case: dict[str, Any]) -> None:
    diagnostics = _streaming_softmin_vjp_diagnostics(**case)

    assert diagnostics["streaming_softmin_vjp_dense_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_softmin_vjp_autodiff_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_softmin_vjp_stopped_key_leakage"] <= VJP_ATOL
    assert diagnostics["streaming_softmin_vjp_finite"] == 1.0


@pytest.mark.parametrize(
    "case",
    [
        {
            "batch_size": 1,
            "num_particles": 4,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 2,
        },
        {
            "batch_size": 2,
            "num_particles": 5,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 3,
        },
    ],
)
def test_streaming_same_points_softmin_vjp_matches_generic_route(case: dict[str, Any]) -> None:
    diagnostics = _streaming_same_points_softmin_vjp_diagnostics(**case)

    assert diagnostics["streaming_same_points_softmin_vjp_generic_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_same_points_softmin_vjp_dense_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_same_points_softmin_vjp_finite"] == 1.0


@pytest.mark.parametrize(
    "case",
    [
        {
            "batch_size": 1,
            "num_particles": 4,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 2,
        },
        {
            "batch_size": 2,
            "num_particles": 5,
            "state_dim": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 3,
        },
    ],
)
def test_streaming_transport_from_potentials_vjp_matches_manual_and_tiny_autodiff(
    case: dict[str, Any],
) -> None:
    diagnostics = _streaming_transport_from_potentials_vjp_diagnostics(**case)

    assert diagnostics["streaming_transport_from_potentials_vjp_manual_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_transport_from_potentials_vjp_autodiff_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_transport_from_potentials_vjp_dg_max_abs"] <= VJP_ATOL
    assert diagnostics["streaming_transport_from_potentials_vjp_g_nontrivial"] > 1.0e-3
    assert diagnostics["streaming_transport_from_potentials_vjp_scaled_x_particles_gap"] > 1.0e-3
    assert diagnostics["streaming_transport_from_potentials_vjp_finite"] == 1.0


@pytest.mark.parametrize(
    "case",
    [
        {
            "batch_size": 1,
            "num_particles": 4,
            "state_dim": 2,
            "steps": 0,
            "row_chunk_size": 2,
            "col_chunk_size": 2,
        },
        {
            "batch_size": 2,
            "num_particles": 5,
            "state_dim": 2,
            "steps": 2,
            "row_chunk_size": 2,
            "col_chunk_size": 3,
        },
    ],
)
def test_streaming_sinkhorn_recursion_vjp_matches_manual_and_tiny_autodiff(
    case: dict[str, Any],
) -> None:
    diagnostics = _streaming_sinkhorn_recursion_vjp_diagnostics(**case)

    assert diagnostics["streaming_sinkhorn_recursion_vjp_manual_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_sinkhorn_recursion_vjp_autodiff_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_sinkhorn_recursion_vjp_directional_error"] <= JVP_ATOL
    assert diagnostics["streaming_sinkhorn_recursion_vjp_custom_gradient_max_abs_error"] <= VJP_ATOL
    assert diagnostics["streaming_sinkhorn_recursion_vjp_stopped_scale_leakage"] <= VJP_ATOL
    assert diagnostics["streaming_sinkhorn_recursion_vjp_finite"] == 1.0


@pytest.mark.parametrize("batch_size,num_particles,state_dim", ((1, 4, 2), (2, 3, 2)))
def test_blockwise_route_matches_dense_and_preserves_streaming_metadata(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> None:
    diagnostics = _blockwise_route_diagnostics(batch_size, num_particles, state_dim)

    assert diagnostics["blockwise_route_value_max_abs_error"] <= VALUE_ATOL
    assert diagnostics["blockwise_route_gradient_max_abs_error"] <= VJP_ATOL
    assert diagnostics["blockwise_route_transport_matrix_size"] == 0.0
    assert diagnostics["blockwise_route_old_streaming_transport_matrix_size"] == 0.0


def test_blockwise_route_rejects_unsupported_combinations_and_preserves_defaults() -> None:
    particles, logw = _fixture(2, 3, 2)
    mask = tf.constant([True, False], dtype=tf.bool)
    kwargs = {
        "epsilon": tf.constant(0.45, DTYPE),
        "scaling": tf.constant(0.8, DTYPE),
        "max_iterations": 2,
        "transport_gradient_mode": MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        "transport_plan_mode": "streaming",
        "row_chunk_size": 2,
        "col_chunk_size": 2,
    }
    assert (
        experimental_batched_ledh_pfpf_ot_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
        == "manual_streaming_finite_sinkhorn_stopped_scale_keys"
    )
    assert (
        experimental_batched_ledh_pfpf_ot_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE
        == "manual_streaming_blockwise_vjp_finite_sinkhorn_stopped_scale_keys"
    )
    assert experimental_batched_ledh_pfpf_ot_tf.DEFAULT_EXECUTION_TARGET == "gpu"

    with pytest.raises(ValueError, match="streaming transport only"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            transport_plan_mode="dense",
            transport_gradient_mode=MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        )
    with pytest.raises(ValueError, match="warmstarts"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            warmstart_state=annealed_transport_tf.build_annealed_transport_coldstart_state_tf(
                particles,
                logw,
                epsilon=tf.constant(0.45, DTYPE),
                scaling=tf.constant(0.8, DTYPE),
                transport_plan_mode="dense",
            ),
            **kwargs,
        )
    with pytest.raises(ValueError, match="transport_ad_mode='stabilized'"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            transport_ad_mode="full",
            **kwargs,
        )
    with pytest.raises(ValueError, match="scalar epsilon only"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=tf.fill([2], tf.constant(0.45, DTYPE)),
            scaling=tf.constant(0.8, DTYPE),
            max_iterations=2,
            transport_gradient_mode=MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
            transport_plan_mode="streaming",
            row_chunk_size=2,
            col_chunk_size=2,
        )


@pytest.mark.parametrize("batch_size,num_particles,state_dim", FIXTURES)
def test_m3_private_dense_custom_gradient_matches_tiny_raw_autodiff_oracle(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> None:
    diagnostics = _dense_custom_gradient_diagnostics(
        batch_size,
        num_particles,
        state_dim,
    )

    assert diagnostics["dense_custom_value_max_abs_error"] <= VALUE_ATOL
    assert diagnostics["dense_custom_gradient_max_abs_error"] <= VJP_ATOL
    assert diagnostics["dense_custom_value_finite"] == 1.0
    assert diagnostics["dense_custom_gradient_finite"] == 1.0


@pytest.mark.parametrize("batch_size,num_particles,state_dim", FIXTURES)
def test_m5_manual_dense_core_matches_stopped_key_helper_and_raw_autodiff(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> None:
    diagnostics = _m5_manual_core_diagnostics(
        batch_size,
        num_particles,
        state_dim,
    )

    assert diagnostics["m5_manual_core_transport_value_max_abs_error"] <= VALUE_ATOL
    assert diagnostics["m5_manual_core_transport_gradient_max_abs_error"] <= VJP_ATOL


@pytest.mark.parametrize("batch_size,num_particles,state_dim", FIXTURES)
def test_m6_manual_streaming_matches_dense_route_without_dense_matrix(
    batch_size: int,
    num_particles: int,
    state_dim: int,
) -> None:
    diagnostics = _m6_manual_streaming_diagnostics(
        batch_size,
        num_particles,
        state_dim,
    )

    assert diagnostics["m6_manual_streaming_particle_value_max_abs_error"] <= VALUE_ATOL
    assert diagnostics["m6_manual_streaming_particle_gradient_max_abs_error"] <= VJP_ATOL
    assert diagnostics["m6_manual_streaming_transport_matrix_size"] == 0.0


def test_m2_unsupported_route_guards_reject_streaming_warmstart_and_n10000() -> None:
    def validate_m2_route(
        *,
        transport_plan_mode: str,
        warmstart: bool,
        num_particles: int,
        governed_validation: bool,
    ) -> None:
        if transport_plan_mode != "dense":
            raise ValueError("M2 primitive route supports dense transport only")
        if warmstart:
            raise ValueError("M2 primitive route does not support warmstarts")
        if governed_validation or num_particles >= 10000:
            raise ValueError("M2 primitive route must not launch governed N10000 validation")

    with pytest.raises(ValueError, match="dense transport only"):
        validate_m2_route(
            transport_plan_mode="streaming",
            warmstart=False,
            num_particles=3,
            governed_validation=False,
        )
    with pytest.raises(ValueError, match="warmstarts"):
        validate_m2_route(
            transport_plan_mode="dense",
            warmstart=True,
            num_particles=3,
            governed_validation=False,
        )
    with pytest.raises(ValueError, match="N10000"):
        validate_m2_route(
            transport_plan_mode="dense",
            warmstart=False,
            num_particles=10000,
            governed_validation=True,
        )

    particles, logw = _fixture(1, 3, 1)
    mask = tf.constant([True], dtype=tf.bool)
    with pytest.raises(ValueError, match="streaming transport currently supports"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            transport_plan_mode="streaming",
            transport_gradient_mode="filterflow_custom_op",
        )


def test_m5_manual_dense_core_mixed_mask_blends_identity_and_uniform_weights() -> None:
    particles, logw = _fixture(2, 3, 2)
    mask = tf.constant([True, False], dtype=tf.bool)
    result = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=tf.constant(0.45, DTYPE),
        scaling=tf.constant(0.8, DTYPE),
        max_iterations=2,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    active_reference = batched_annealed_transport_core_tf(
        particles[:1],
        logw[:1],
        tf.constant([True], dtype=tf.bool),
        epsilon=tf.constant(0.45, DTYPE),
        scaling=tf.constant(0.8, DTYPE),
        max_iterations=2,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )
    identity = tf.eye(3, dtype=DTYPE)
    uniform_log = tf.fill([3], -tf.math.log(tf.constant(3.0, DTYPE)))

    _assert_near(result.transport_matrix[:1], active_reference.transport_matrix, VALUE_ATOL)
    _assert_near(result.transport_matrix[1], identity, VALUE_ATOL)
    _assert_near(result.log_weights[0], uniform_log, VALUE_ATOL)
    _assert_near(result.log_weights[1], logw[1], VALUE_ATOL)
    _assert_near(result.particles[1], particles[1], VALUE_ATOL)


def test_m5_manual_dense_core_rejects_unsupported_combinations() -> None:
    particles, logw = _fixture(2, 3, 2)
    mask = tf.constant([True, False], dtype=tf.bool)
    kwargs = {
        "epsilon": tf.constant(0.45, DTYPE),
        "scaling": tf.constant(0.8, DTYPE),
        "max_iterations": 2,
        "transport_gradient_mode": MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    }
    with pytest.raises(ValueError, match="dense transport only"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            transport_plan_mode="streaming",
            **kwargs,
        )
    with pytest.raises(ValueError, match="warmstarts"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            warmstart_state=annealed_transport_tf.build_annealed_transport_coldstart_state_tf(
                particles,
                logw,
                epsilon=tf.constant(0.45, DTYPE),
                scaling=tf.constant(0.8, DTYPE),
                transport_plan_mode="dense",
            ),
            **kwargs,
        )
    with pytest.raises(ValueError, match="transport_ad_mode='stabilized'"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            transport_ad_mode="full",
            **kwargs,
        )
    with pytest.raises(ValueError, match="scalar epsilon only"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=tf.fill([2], tf.constant(0.45, DTYPE)),
            scaling=tf.constant(0.8, DTYPE),
            max_iterations=2,
            transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        )


def test_m6_manual_streaming_core_rejects_unsupported_combinations() -> None:
    particles, logw = _fixture(2, 3, 2)
    mask = tf.constant([True, False], dtype=tf.bool)
    kwargs = {
        "epsilon": tf.constant(0.45, DTYPE),
        "scaling": tf.constant(0.8, DTYPE),
        "max_iterations": 2,
        "transport_gradient_mode": MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        "transport_plan_mode": "streaming",
        "row_chunk_size": 2,
        "col_chunk_size": 2,
    }
    with pytest.raises(ValueError, match="streaming transport only"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            transport_plan_mode="dense",
            transport_gradient_mode=MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        )
    with pytest.raises(ValueError, match="warmstarts"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            warmstart_state=annealed_transport_tf.build_annealed_transport_coldstart_state_tf(
                particles,
                logw,
                epsilon=tf.constant(0.45, DTYPE),
                scaling=tf.constant(0.8, DTYPE),
                transport_plan_mode="dense",
            ),
            **kwargs,
        )
    full_result = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        transport_ad_mode="full",
        **kwargs,
    )
    assert bool(tf.reduce_all(tf.math.is_finite(full_result.particles)).numpy())
    with pytest.raises(ValueError, match="scalar epsilon only"):
        batched_annealed_transport_core_tf(
            particles,
            logw,
            mask,
            epsilon=tf.fill([2], tf.constant(0.45, DTYPE)),
            scaling=tf.constant(0.8, DTYPE),
            max_iterations=2,
            transport_gradient_mode=MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            transport_plan_mode="streaming",
            row_chunk_size=2,
            col_chunk_size=2,
        )


def test_m3_private_dense_custom_gradient_rejects_vector_epsilon_scope() -> None:
    particles, logw = _fixture(2, 3, 2)
    with pytest.raises(ValueError, match="scalar epsilon only"):
        annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys(  # noqa: SLF001
            particles,
            logw,
            tf.fill([2], tf.constant(0.45, DTYPE)),
            tf.fill([2], tf.constant(0.9, DTYPE)),
            tf.constant(0.8, DTYPE),
            steps=2,
        )


def test_m3_private_dense_custom_gradient_blocks_hyperparameter_gradients() -> None:
    particles, logw = _fixture(2, 3, 2)
    eps = tf.Variable(tf.constant(0.45, DTYPE))
    epsilon0 = tf.Variable(tf.fill([2], tf.constant(0.9, DTYPE)))
    scaling = tf.Variable(tf.constant(0.8, DTYPE))
    upstream = _cotangent((2, 3, 3), scale=0.025)
    with tf.GradientTape() as tape:
        matrix = annealed_transport_tf._filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys(  # noqa: SLF001
            particles,
            logw,
            eps,
            epsilon0,
            scaling,
            steps=2,
        )
        scalar = tf.reduce_sum(matrix * upstream)
    d_eps, d_epsilon0, d_scaling = tape.gradient(
        scalar,
        [eps, epsilon0, scaling],
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    assert float(tf.reduce_max(tf.abs(d_eps)).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(d_epsilon0)).numpy()) == 0.0
    assert float(tf.reduce_max(tf.abs(d_scaling)).numpy()) == 0.0


def test_m5_manual_dense_route_is_opt_in_to_batched_core_only() -> None:
    particles, logw = _fixture(1, 3, 1)
    mask = tf.constant([True], dtype=tf.bool)
    result = batched_annealed_transport_core_tf(
        particles,
        logw,
        mask,
        epsilon=tf.constant(0.45, DTYPE),
        scaling=tf.constant(0.8, DTYPE),
        max_iterations=2,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )

    assert result.particles.shape == particles.shape
    assert result.log_weights.shape == logw.shape
    assert result.transport_matrix.shape == (1, 3, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(result.particles)).numpy())

    with pytest.raises(ValueError, match="transport_gradient_mode"):
        annealed_transport_tf.annealed_transport_resample_tf(
            particles,
            logw,
            transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        )


if __name__ == "__main__":
    print(json.dumps(collect_diagnostics(), indent=2, sort_keys=True))
