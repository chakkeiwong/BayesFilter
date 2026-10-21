"""Native forward/reverse recurrences for the existing branch-axis TT adjoint.

This preserves the fixed ALS schedule, relative defensive mass, smooth shift,
checkpoint replay and explicit manual node adjoints. Python builds static
heterogeneous schemas; dates and solve updates execute in TensorFlow loops.
"""

from __future__ import annotations

import math
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.retained_quadratic_form_tf import (
    prefix_gram_matrix,
    prefix_row_vectors,
    suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_adjoint_tf import (
    cholesky_vjp,
    design_assembly_adjoint,
    gram_chain_adjoint,
    prefix_rows_adjoint,
    solve_node_adjoint,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    _gauss_rows,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.squared_tt_engine_xla_tf import _solve_scaled_qr
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import add_core_tensors, basis_masses
from bayesfilter.highdim.tt_native_control_tf import gram_adjoint
from bayesfilter.highdim.tt_native_control_tf import random_core_start_program
from bayesfilter.highdim.tt_preparation_tf import frozen_design_rows

DTYPE = tf.float64


def _arrays(value, size):
    return tf.nest.map_structure(lambda x: tf.TensorArray(
        x.dtype, size=size, element_shape=x.shape, clear_after_read=False), value)


def _write(arrays, index, values):
    return tf.nest.map_structure(lambda array, value: array.write(index, value), arrays, values)


def _read(values, index):
    return tf.nest.map_structure(lambda value: value[index], values)


def _stack(arrays):
    return tf.nest.map_structure(lambda array: array.stack(), arrays)


def _fit(basis, rows, target, weights, starts, config):
    fitter = FixedTTFitter()
    count = len(starts)
    updates = config.sweeps * count
    history = _write(_arrays(starts, updates + 1), 0, starts)
    def body(index, values, worst, history):
        def branch(axis):
            def update():
                cores = tuple(TTCore(value) for value in values)
                design = fitter._build_design_matrix(basis, rows, cores, axis)
                solution, condition = _solve_scaled_qr(design, weights, target, tf.constant(config.ridge, DTYPE))
                new_values = values[:axis] + (tf.reshape(solution, values[axis].shape),) + values[axis + 1:]
                return new_values, tf.maximum(worst, condition)
            return update
        values, worst = tf.switch_case(index % count, tuple(branch(axis) for axis in range(count)))
        return index + 1, values, worst, _write(history, index + 1, values)
    _, values, worst, history = tf.while_loop(lambda i, *_: i < updates, body,
        (tf.constant(0), starts, tf.constant(0., DTYPE), history),
        maximum_iterations=updates, parallel_iterations=1)
    return values, worst, _stack(history)


def _reverse_fit(basis, rows, target, weights, history, bar_cores, config):
    fitter = FixedTTFitter()
    count = len(history)
    updates = config.sweeps * count
    def body(index, bar_cores, bar_target):
        update_index = updates - 1 - index
        cores = tuple(TTCore(value) for value in _read(history, update_index))
        after = _read(history, update_index + 1)
        def branch(axis):
            def reverse():
                design = fitter._build_design_matrix(basis, rows, cores, axis)
                bar_g, bar_a = solve_node_adjoint(design, weights, target,
                    tf.reshape(after[axis], [-1]), config.ridge, tf.reshape(bar_cores[axis], [-1]))
                additions = design_assembly_adjoint(basis, rows, cores, axis, bar_a)
                values = add_core_tensors(bar_cores, tuple(core.values for core in additions))
                values = values[:axis] + (tf.zeros_like(values[axis]),) + values[axis + 1:]
                return values, bar_target + bar_g
            return reverse
        bars, bar_target = tf.switch_case(update_index % count, tuple(branch(axis) for axis in range(count)))
        return index + 1, bars, bar_target
    _, _, bar_target = tf.while_loop(lambda i, *_: i < updates, body,
        (tf.constant(0), bar_cores, tf.zeros_like(target)),
        maximum_iterations=updates, parallel_iterations=1)
    return bar_target


@lru_cache(maxsize=8)
def make_adjoint_filter(adapter, observation_shape, config, *, transition_vjp,
                        observation_vjp, initial_vjp, parameter_dim, jit_compile=True):
    n, horizon = adapter.state_dim, int(observation_shape[0])
    if horizon < 1:
        raise ValueError("positive observation horizon required")
    basis = _product_basis(n, config.basis_degree)
    extended_basis = _product_basis(n + 1, config.basis_degree)
    basis_dim = int(basis.bases[0].basis_dim)
    half = tf.constant(config.coordinate_half_width, DTYPE)
    conversion = tf.cast(n, DTYPE) * (tf.math.log(half) + tf.math.log(tf.constant(2., DTYPE)))
    tau = tf.constant(config.tau, DTYPE)
    initial_cores = tuple(core.values for core in _initial_tt_cores(n, basis_dim, config.rank))
    if config.quadrature_order is not None:
        initial_rows, initial_weights = _gauss_rows(n, config.quadrature_order, jit_compile=jit_compile)
        rows, row_weights = _gauss_rows(2 * n, config.quadrature_order, jit_compile=jit_compile)
        all_rows = tf.repeat(rows[None], max(1, horizon - 1), axis=0)
    else:
        initial_rows = frozen_design_rows(config, config.row_count, n, tf.constant([0]), 17,
            jit_compile=jit_compile)[0]
        initial_weights = tf.fill([config.row_count], tf.constant(1. / config.row_count, DTYPE))
        all_rows = frozen_design_rows(config, config.row_count, 2 * n,
            tf.range(1, max(2, horizon)), 100, jit_compile=jit_compile)
        row_weights = tf.fill([config.row_count], tf.constant(1. / config.row_count, DTYPE))

    def phase(previous_rank, dates):
        branch_count = previous_rank + 1
        mixed_basis = ProductBasis(list(basis.bases) + [DiscreteIndicatorBasis1D(branch_count)]
                                    + list(basis.bases), basis.convention)
        dims = (basis_dim,) * n + (branch_count,) + (basis_dim,) * n
        shapes = tuple((1 if axis == 0 else config.rank, dim,
                        1 if axis == 2 * n else config.rank) for axis, dim in enumerate(dims))
        starts = random_core_start_program(shapes, dates.shape[0], jit_compile=jit_compile)(
            dates, tf.constant(config.seed, tf.int32))

        def forward(t, observations, prefix, gram, zc, packed):
            eigenvalues = tf.linalg.eigvalsh(gram)
            condition = eigenvalues[-1] / tf.maximum(eigenvalues[0], tf.constant(1e-300, DTYPE))
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(gram + tf.constant(config.branch_gram_floor, DTYPE)
                * floor_scale * tf.eye(tf.shape(gram)[0], dtype=DTYPE))
            rows = all_rows[t - 1]
            x_current, z_previous = rows[:, :n] * half, rows[:, n:]
            x_previous = z_previous * half
            log_g = adapter.transition_log_density(x_current, x_previous) + adapter.observation_log_density(x_current, observations[t]) + conversion
            v = prefix_row_vectors(tuple(TTCore(value) for value in prefix), basis, z_previous)
            u = tf.einsum("na,ab->nb", v, chol)
            tau_abs = tau * (zc / (1. + tau))
            sum_sq = tf.reduce_sum(tf.square(u), axis=1) + tau_abs
            log_f = tf.math.log(sum_sq) + log_g
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
            sqrt_g = tf.exp(.5 * (log_g - shift))
            amplitudes = tf.concat((u, tf.ones([rows.shape[0], 1], DTYPE) * tf.sqrt(tau_abs)), axis=1)
            target = tf.reshape(amplitudes * sqrt_g[:, None], [-1])
            codes = tf.tile(tf.range(branch_count, dtype=DTYPE)[None], [rows.shape[0], 1])
            full_rows = tf.concat((tf.repeat(rows[:, :n], branch_count, axis=0),
                tf.reshape(codes, [-1, 1]), tf.repeat(rows[:, n:], branch_count, axis=0)), axis=1)
            weights = tf.reshape(tf.repeat(row_weights, branch_count, axis=0), [-1])
            initial = tuple(tf.reshape(packed[axis, :math.prod(shape)], shape) for axis, shape in enumerate(shapes))
            fitted, worst, updates = _fit(mixed_basis, full_rows, target, weights, initial, config)
            cores = tuple(TTCore(value) for value in fitted)
            new_gram = suffix_gram_matrix(cores[n:], mixed_basis, axis_offset=n)
            prefix_gram = prefix_gram_matrix(cores[:n], mixed_basis)
            new_zc = (1. + tau) * tf.einsum("ab,ab->", prefix_gram, new_gram)
            increment = shift + tf.math.log(new_zc) - tf.math.log(zc)
            step = {"x_current": x_current, "x_previous": x_previous, "z_previous": z_previous,
                "prev_prefix": prefix, "prev_zc": zc, "chol": chol, "v": v, "u": u, "tau_abs": tau_abs,
                "sum_sq": sum_sq, "sqrt_g": sqrt_g, "amplitudes": amplitudes,
                "softmax": tf.nn.softmax(log_f), "target": target, "rows": full_rows, "weights": weights,
                "cores": fitted, "updates": updates, "gram": new_gram, "zc": new_zc,
                "increment": increment, "worst": worst, "gram_condition": condition}
            return fitted[:n], new_gram, new_zc, step

        def reverse(step, y, bar_prefix, bar_gram, bar_zh):
            bar_target = reverse_normalizer_and_fit(step, mixed_basis, bar_prefix, bar_gram, bar_zh)
            bar_targets = tf.reshape(bar_target, [all_rows.shape[1], branch_count])
            bar_amplitudes = bar_targets * step['sqrt_g'][:, None]
            bar_sqrt_g = tf.reduce_sum(bar_targets * step['amplitudes'], axis=1)
            bar_log_g = .5 * step['sqrt_g'] * bar_sqrt_g
            bar_shift = -tf.reduce_sum(bar_log_g)
            bar_logf = (tf.constant(1., DTYPE) + bar_shift) * step['softmax']
            bar_sum_sq = bar_logf / step['sum_sq']
            bar_log_g += bar_logf
            bar_u = bar_amplitudes[:, :previous_rank] + 2. * step['u'] * bar_sum_sq[:, None]
            # With fixed tau=0 the defensive amplitude is constant in theta.
            bar_tau = tf.reduce_sum(bar_sum_sq) + tf.math.divide_no_nan(
                tf.reduce_sum(bar_amplitudes[:, previous_rank]), 2. * tf.sqrt(step['tau_abs']))
            bar_v = tf.einsum('nb,ab->na', bar_u, step['chol'])
            bar_chol = tf.einsum('na,nb->ab', step['v'], bar_u)
            bar_gram = cholesky_vjp(step['chol'], bar_chol)
            # A = G + rho * trace(G)/r * I; include the relative floor's pullback.
            floor_cotangent = (tf.constant(config.branch_gram_floor, DTYPE)
                               * tf.linalg.trace(bar_gram) / tf.cast(previous_rank, DTYPE))
            bar_gram += floor_cotangent * tf.eye(previous_rank, dtype=DTYPE)
            prev_cores = tuple(TTCore(value) for value in step['prev_prefix'])
            bar_prefix = tuple(core.values for core in prefix_rows_adjoint(prev_cores, basis, step['z_previous'], bar_v))
            bar_zh = -1. / (step['prev_zc'] / (1. + tau)) + tau * bar_tau
            gradient = transition_vjp(step['x_current'], step['x_previous'], bar_log_g)
            gradient += observation_vjp(step['x_current'], y, bar_log_g)
            return gradient, bar_prefix, bar_gram, bar_zh
        return forward, reverse, starts, mixed_basis

    def reverse_normalizer_and_fit(step, fit_basis, bar_prefix, bar_gram, bar_zh, *, initial=False):
        fitted = tuple(TTCore(value) for value in step['cores'])
        cores = fitted
        gram_basis = fit_basis
        if initial:
            suffix = tf.tensor_scatter_nd_update(tf.zeros([fitted[-1].right_rank, basis_dim, 1], DTYPE), [[0, 0, 0]], [1.])
            cores = fitted + (TTCore(suffix),)
            gram_basis = extended_basis
        prefix, suffix = cores[:n], cores[n:]
        z_h = step['zc'] / (1. + tau)
        bar_zh += 1. / z_h
        suffix_mass = suffix_gram_matrix(suffix, gram_basis, axis_offset=n)
        prefix_mass = prefix_gram_matrix(prefix, gram_basis)
        bar_suffix = bar_zh * prefix_mass + bar_gram
        bar_prefix_mass = bar_zh * suffix_mass
        masses = basis_masses(gram_basis, cores[:n])
        prefix_bars = gram_adjoint(prefix, masses, bar_prefix_mass)
        prefix_bars = add_core_tensors(prefix_bars, bar_prefix)
        suffix_bars = tuple(core.values for core in gram_chain_adjoint(suffix, gram_basis, axis_offset=n, bar_gram=bar_suffix))
        return _reverse_fit(fit_basis, step['rows'], step['target'], step['weights'], step['updates'],
                             (prefix_bars + suffix_bars)[:len(fitted)], config)

    first_forward, first_reverse, first_starts, _ = phase(1, tf.constant([1], tf.int32))
    forward, reverse, starts, _ = phase(config.rank, tf.range(2, max(3, horizon)))

    def evaluate(observations):
        x0 = initial_rows * half
        log_f = adapter.initial_log_density(x0) + adapter.observation_log_density(x0, observations[0]) + conversion
        shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
        target = tf.exp(.5 * (log_f - shift))
        fitted, worst, updates = _fit(basis, initial_rows, target, initial_weights, initial_cores, config)
        suffix = tf.tensor_scatter_nd_update(tf.zeros([fitted[-1].shape[-1], basis_dim, 1], DTYPE), [[0, 0, 0]], [1.])
        gram = suffix_gram_matrix((TTCore(suffix),), extended_basis, axis_offset=n)
        zc = (1. + tau) * tf.einsum('ab,ab->', prefix_gram_matrix(tuple(TTCore(value) for value in fitted), extended_basis), gram)
        total = shift + tf.math.log(zc)
        initial = {"cores": fitted, "updates": updates, "zc": zc, "target": target,
                       "rows": initial_rows, "weights": initial_weights}
        valid = tf.math.is_finite(total) & tf.math.is_finite(worst)
        worst_gram = tf.constant(1., DTYPE)
        bar_prefix = tuple(tf.zeros_like(value) for value in fitted)
        bar_gram = tf.zeros_like(gram)
        bar_zh = tf.constant(0., DTYPE)
        gradient = tf.zeros([parameter_dim], DTYPE)
        if horizon > 1:
            prefix, gram, zc, first = first_forward(tf.constant(1), observations, fitted, gram, zc, first_starts[0])
            total += first['increment']
            worst = tf.maximum(worst, first['worst'])
            worst_gram = tf.maximum(worst_gram, first['gram_condition'])
            valid &= tf.math.is_finite(first['increment']) & tf.math.is_finite(first['worst']) & tf.math.is_finite(first['gram_condition'])
            # First transition changes the boundary rank and the branch axis.
            if horizon > 2:
                prefix, gram, zc, second = forward(tf.constant(2), observations, prefix, gram, zc, starts[0])
                total += second['increment']
                worst = tf.maximum(worst, second['worst'])
                worst_gram = tf.maximum(worst_gram, second['gram_condition'])
                valid &= tf.math.is_finite(second['increment']) & tf.math.is_finite(second['worst']) & tf.math.is_finite(second['gram_condition'])
                history = _write(_arrays(second, horizon - 2), 0, second)
                def date_body(t, prefix, gram, zc, total, worst, worst_gram, valid, history):
                    prefix, gram, zc, step = forward(t, observations, prefix, gram, zc, starts[t - 2])
                    valid &= tf.math.is_finite(step['increment']) & tf.math.is_finite(step['worst']) & tf.math.is_finite(step['gram_condition'])
                    return (t + 1, prefix, gram, zc, total + step['increment'], tf.maximum(worst, step['worst']),
                            tf.maximum(worst_gram, step['gram_condition']), valid, _write(history, t - 2, step))
                if horizon > 3:
                    _, prefix, gram, zc, total, worst, worst_gram, valid, history = tf.while_loop(
                        lambda t, *_: t < horizon, date_body,
                        (tf.constant(3), prefix, gram, zc, total, worst, worst_gram, valid, history),
                        maximum_iterations=horizon - 3, parallel_iterations=1)
                history = _stack(history)
                bar_prefix = tuple(tf.zeros_like(value) for value in prefix)
                bar_gram = tf.zeros_like(gram)
                def reverse_date(t, gradient, bar_prefix, bar_gram, bar_zh):
                    direct, bar_prefix, bar_gram, bar_zh = reverse(_read(history, t - 2), observations[t], bar_prefix, bar_gram, bar_zh)
                    return t - 1, gradient + direct, bar_prefix, bar_gram, bar_zh
                _, gradient, bar_prefix, bar_gram, bar_zh = tf.while_loop(lambda t, *_: t >= 2, reverse_date,
                    (tf.constant(horizon - 1), gradient, bar_prefix, bar_gram, bar_zh),
                    maximum_iterations=horizon - 2, parallel_iterations=1)
            else:
                bar_prefix = tuple(tf.zeros_like(value) for value in prefix)
                bar_gram = tf.zeros_like(gram)
            direct, bar_prefix, bar_gram, bar_zh = first_reverse(first, observations[1], bar_prefix, bar_gram, bar_zh)
            gradient += direct
        bar_target = reverse_normalizer_and_fit(initial, basis, bar_prefix, bar_gram, bar_zh, initial=True)
        half_target = .5 * target * bar_target
        bar_shift = -tf.reduce_sum(half_target)
        bar_logf = half_target + (tf.constant(1., DTYPE) + bar_shift) * tf.nn.softmax(log_f)
        gradient += initial_vjp(x0, bar_logf)
        gradient += observation_vjp(x0, observations[0], bar_logf)
        valid &= tf.math.is_finite(total) & tf.reduce_all(tf.math.is_finite(gradient))
        return total, gradient, {"valid": valid, "worst_condition": worst, "gram_condition": worst_gram}
    return tf.function(evaluate, input_signature=[tf.TensorSpec(observation_shape, DTYPE)],
                       jit_compile=jit_compile, autograph=False)
