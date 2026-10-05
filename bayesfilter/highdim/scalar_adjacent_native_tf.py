"""Complete native-date execution of the existing scalar adjacent-TT extension.

Paper context: Zhao--Cui Algorithm 2, equations (15)/(16), and local author
source models/full_sol.m:73-125. The fixed weighted ALS fit is an existing local
extension; this execution refactor supplies no source-faithfulness promotion.
"""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.fixed_tt_native_fit_tf import NativeFixedTTFit
from bayesfilter.highdim.tt_native_control_tf import make_fixed_squared_marginal

D = tf.float64
_CACHE = OrderedDict()


def make_scalar_adjacent_state_fixed_tt(
    model, config, observation_shape, *, jit_compile=True
):
    """Return cached complete value/report and value/total-AD-score callables."""
    from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
        _log_reference_density,
        _reference_quadrature,
    )

    key = (id(model), id(config), tuple(observation_shape), bool(jit_compile))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key][2]
    horizon = int(observation_shape[0])
    if horizon < 1:
        raise ValueError("observations must be nonempty")
    previous_rows = _reference_quadrature(
        config.adjacent.product_basis, config.adjacent.fit_quadrature_order
    )[0][:, 1:2]

    def prepare(active):
        basis = active.product_basis
        rows, weights = _reference_quadrature(basis, active.fit_quadrature_order)
        fit = NativeFixedTTFit(
            basis,
            rows,
            weights,
            active.fit_config,
            active.initial_cores,
            jit_compile=jit_compile,
        )
        physical, logdet = config.scalar_coordinate_map.forward(rows[:, :1])
        retained_basis = ProductBasis([basis.bases[0]], basis.convention)
        mass_rows, mass_weights = _reference_quadrature(retained_basis, 65)
        volume = (
            tf.reduce_prod(tf.stack(tuple(axis.domain.length for axis in basis.bases)))
            if basis.convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE
            else tf.constant(1.0, D)
        )
        marginal_volume = (
            basis.bases[1].domain.length
            if basis.dimension == 2
            and basis.convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE
            else 1.0
        )

        previous_marginal = make_fixed_squared_marginal(
            basis, active.initial_cores, (0,), previous_rows, jit_compile=jit_compile
        )
        mass_marginal = make_fixed_squared_marginal(
            basis, active.initial_cores, (0,), mass_rows, jit_compile=jit_compile
        )
        normalizer = make_fixed_squared_marginal(
            basis, active.initial_cores, (), tf.zeros([1, 0], D), jit_compile=jit_compile
        )

        def marginal(packed):
            return (
                previous_marginal(fit.unpack(packed))
                + tf.constant(active.density_tau, D) * marginal_volume
            )

        def evaluate(log_target, initial):
            shift, shift_index = tf.reduce_max(log_target), tf.argmax(log_target)
            target = tf.exp(0.5 * (log_target - shift))
            result = fit(target, initial)
            z = (
                normalizer(fit.unpack(result["cores"]))[0]
                + tf.constant(active.density_tau, D) * volume
            )
            mass_values = (
                mass_marginal(fit.unpack(tf.stop_gradient(result["cores"])))
                + tf.constant(active.density_tau, D) * marginal_volume
            )
            mass = tf.reduce_sum(mass_weights * mass_values / tf.stop_gradient(z))
            valid = (
                result["valid"] & tf.math.is_finite(z) & (z > active.normalizer_floor)
            )
            return {
                "fit": result,
                "target": target,
                "initial": initial,
                "shift": shift,
                "shift_index": shift_index,
                "z": z,
                "mass": mass,
                "increment": tf.math.log(z) + shift,
                "valid": valid,
            }

        return fit, rows, physical, logdet, marginal, evaluate

    initial = prepare(config.initial)
    adjacent = prepare(config.adjacent)
    afit, arows, current_x, current_logdet, adj_marginal, adj_fit = adjacent
    previous_x, previous_logdet = config.scalar_coordinate_map.forward(arows[:, 1:2])
    log_reference = _log_reference_density(config.adjacent.product_basis)
    log_current_reference = _log_reference_density(
        ProductBasis(
            [config.adjacent.product_basis.bases[0]], config.adjacent.measure_convention
        )
    )

    def evaluate(theta, observations):
        if config.transition_before_first_observation:
            log_target = (
                model.initial_log_density(theta, previous_x)
                + model.transition_log_density(theta, previous_x, current_x, t=0)
                + model.observation_log_density(theta, current_x, observations[0], t=0)
                + current_logdet
                + previous_logdet
                - log_reference
            )
            first = adj_fit(log_target, afit.initial)
            first_marginal = adj_marginal
        else:
            ifit, _, physical, logdet, first_marginal, ifit_call = initial
            log_target = (
                model.initial_log_density(theta, physical)
                + model.observation_log_density(theta, physical, observations[0], t=0)
                + logdet
                - _log_reference_density(config.initial.product_basis)
            )
            first = ifit_call(log_target, ifit.initial)

        if horizon == 1:
            return first["increment"][None], first, ()

        def next_fit(date, previous_values, starting):
            log_target = (
                tf.math.log(previous_values)
                + model.transition_log_density(theta, previous_x, current_x, t=date)
                + model.observation_log_density(
                    theta, current_x, observations[date], t=date
                )
                + current_logdet
                - log_current_reference
            )
            return adj_fit(log_target, starting)

        # Preserve the original warm-start rule: t=1 starts from the configured
        # adjacent cores, including when t=0 already fits an adjacent state.
        second = next_fit(
            tf.constant(1),
            first_marginal(first["fit"]["cores"]) / first["z"],
            afit.initial,
        )
        if horizon == 2:
            remaining = tf.nest.map_structure(lambda value: value[None], second)
            return tf.stack((first["increment"], second["increment"])), first, remaining
        buffers = tf.nest.map_structure(
            lambda value: tf.TensorArray(
                value.dtype, horizon - 1, element_shape=value.shape
            ).write(0, value),
            second,
        )

        def step(date, previous, normalizer, buffers):
            values = adj_marginal(previous) / normalizer
            row = next_fit(date, values, previous)
            buffers = tf.nest.map_structure(
                lambda buffer, value: buffer.write(date - 1, value), buffers, row
            )
            return date + 1, row["fit"]["cores"], row["z"], buffers

        _, _, _, buffers = tf.while_loop(
            lambda date, *_: date < horizon,
            step,
            (tf.constant(2), second["fit"]["cores"], second["z"], buffers),
            maximum_iterations=max(0, horizon - 2),
            parallel_iterations=1,
        )
        remaining = tf.nest.map_structure(lambda buffer: buffer.stack(), buffers)
        return (
            tf.concat((first["increment"][None], remaining["increment"]), 0),
            first,
            remaining,
        )

    signature = [
        tf.TensorSpec([model.parameter_dim()], D),
        tf.TensorSpec(observation_shape, D),
    ]
    value = tf.function(
        evaluate, input_signature=signature, jit_compile=jit_compile, autograph=False
    )

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def score(theta, observations):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = evaluate(theta, observations)
            total = tf.reduce_sum(result[0])
        return result, tape.gradient(total, theta)

    result = (value, score, initial[0], afit)
    _CACHE[key] = (model, config, result)
    while len(_CACHE) > 8:  # Bounded callable cache, not numerical iteration.
        _CACHE.popitem(last=False)
    return result
