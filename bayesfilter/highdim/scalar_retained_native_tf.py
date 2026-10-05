"""Native scalar retained-grid filtering and recursive analytical sensitivities.

This preserves the existing weighted ALS/retained-grid extension. Paper context
is Zhao--Cui Algorithm 2, (15)--(16), and author models/full_sol.m:73--125;
this local quadrature route makes no source-faithfulness claim. Numerical date
updates and all parameter columns execute in one enclosing XLA program.
"""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.fixed_tt_native_fit_tf import NativeFixedTTFit
from bayesfilter.ops.qr_lstsq_tf import condition_number

D = tf.float64
_CACHE = OrderedDict()


class ScalarRetainedProgram:
    """Fixed schema and completed tensor histories for host artifact assembly."""

    def __init__(self, model, config, observation_shape, moment_order,
                 propagation_order, derivative_config, jit_compile):
        from bayesfilter.highdim import filtering as old

        basis = config.product_basis
        coordinate_map = old._coordinate_map_for_config(config, 1)
        old._validate_scalar_adjacent_target_common(
            model, basis, coordinate_map, config.measure_convention,
            config.fit_quadrature_order, "native_scalar_retained", "native_target",
        )
        self.rows, self.weights = old._tensor_product_reference_quadrature(
            basis, config.fit_quadrature_order
        )
        self.physical, self.logdet = coordinate_map.forward(self.rows)
        nodes, self.propagation_weights = old.legendre_gauss_nodes_weights(propagation_order)
        self.propagation_rows = nodes[:, None]
        self.propagation_physical, self.propagation_logdet = coordinate_map.forward(
            self.propagation_rows
        )
        nodes, moment_weights = old.legendre_gauss_nodes_weights(moment_order)
        moment_rows = nodes[:, None]
        moment_physical, _ = coordinate_map.forward(moment_rows)
        moment_weights = 0.5 * moment_weights
        initial = config.initial_cores or old._default_initial_cores(basis, config.fit_config)
        self.fit = NativeFixedTTFit(basis, self.rows, self.weights, config.fit_config,
                                    initial, jit_compile=jit_compile)
        @tf.function(input_signature=[tf.TensorSpec(self.rows.shape, D),
            tf.TensorSpec(moment_rows.shape, D), tf.TensorSpec(self.propagation_rows.shape, D)],
            jit_compile=jit_compile, autograph=False)
        def prepare_basis(rows, moment_rows, propagation_rows):
            return (basis.bases[0].evaluate(rows[:, 0]),
                    basis.bases[0].evaluate(moment_rows[:, 0]),
                    basis.bases[0].evaluate(propagation_rows[:, 0]),
                    basis.bases[0].mass_matrix(config.measure_convention.mass_measure))

        prepare = prepare_basis.python_function if tf.inside_function() else prepare_basis
        phi, moment_phi, propagation_phi, mass_matrix = prepare(
            self.rows, moment_rows, self.propagation_rows)
        self.basis_preparation = prepare_basis
        volume = (basis.bases[0].domain.length
                  if config.measure_convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE
                  else 1.0)
        tau = tf.constant(config.density_tau, D)
        log_reference_weight = old._log_uniform_reference_weight_density(basis)
        horizon = int(observation_shape[0])
        if horizon < 1:
            raise ValueError("observations must contain at least one row")
        count, previous_count = self.rows.shape[0], propagation_order
        old._check_pairwise_transition_tensor_budget(count, previous_count, 1)
        if (previous_count * 4 + 3) * D.size > config.retained_storage_byte_budget:
            raise ValueError(old.HighDimStatus.RETAINED_STORAGE_BUDGET_EXCEEDED.value)
        next_points = tf.repeat(self.physical, previous_count, axis=0)
        previous_points = tf.tile(self.propagation_physical, [count, 1])
        with_score = derivative_config is not None
        indices = () if not with_score else tuple(derivative_config.parameter_indices)
        if with_score:
            old._require_explicit_parameter_score_methods(model)
            if not indices or min(indices) < 0 or max(indices) >= model.parameter_dim():
                raise ValueError("parameter_indices: INVALID_SHAPE")
        parameter_count = len(indices)

        def select_scores(value, rows):
            value = tf.ensure_shape(value, [rows, model.parameter_dim()])
            return tf.gather(value, indices, axis=1)

        def evaluate(theta, observations, runtime_model=None):
            active_model = model if runtime_model is None else runtime_model
            # The design is fixed and scalar, so every parameter RHS shares the
            # same normal matrix; no AD or repeated scalar score is involved.
            normal = tf.matmul(phi, phi * self.weights[:, None], transpose_a=True)
            normal += tf.constant(config.fit_config.ridge, D) * tf.eye(phi.shape[1], dtype=D)
            derivative_condition = (condition_number(normal, jit_compile=jit_compile)
                                    if with_score else tf.constant(0.0, D))

            def date_step(date, previous_log, previous_dot, starting):
                def initial_target():
                    value = active_model.initial_log_density(theta, self.physical)
                    dot = (select_scores(active_model.initial_log_density_parameter_score(
                        theta, self.physical), count) if with_score
                        else tf.zeros([count, 0], D))
                    return value, dot

                def transitioned_target():
                    transition = tf.reshape(active_model.transition_log_density(
                        theta, previous_points, next_points, t=date), [count, previous_count])
                    terms = (tf.math.log(self.propagation_weights)[None, :]
                             + self.propagation_logdet[None, :] + previous_log[None, :]
                             + transition)
                    value = tf.reduce_logsumexp(terms, axis=1)
                    if with_score:
                        local = select_scores(active_model.transition_log_density_parameter_score(
                            theta, previous_points, next_points, t=date), count * previous_count)
                        dot_terms = tf.reshape(local, [count, previous_count, parameter_count])
                        dot = tf.reduce_sum(tf.nn.softmax(terms, axis=1)[:, :, None]
                                            * (dot_terms + previous_dot[None, :, :]), axis=1)
                    else:
                        dot = tf.zeros([count, 0], D)
                    return value, dot

                prior, dot_prior = tf.cond(date == 0, initial_target, transitioned_target)
                log_target = (prior + active_model.observation_log_density(
                    theta, self.physical, observations[date], t=date)
                    + self.logdet - log_reference_weight)
                shift = tf.reduce_max(log_target)
                target = tf.exp(0.5 * (log_target - shift))
                fit_result = self.fit(target, starting)
                coefficients = fit_result["cores"][0]
                z = tf.einsum("i,ij,j->", coefficients, mass_matrix, coefficients) + tau * volume
                increment = tf.math.log(z) + shift
                propagation_h = tf.linalg.matvec(propagation_phi, coefficients)
                propagation_raw = tf.square(propagation_h) + tau
                propagation_log = (tf.math.log(propagation_raw) - tf.math.log(z)
                                   - self.propagation_logdet + log_reference_weight)
                moment_h = tf.linalg.matvec(moment_phi, coefficients)
                moment_raw = tf.square(moment_h) + tau
                moment_values = tf.exp(tf.math.log(moment_raw) - tf.math.log(z))
                weighted_values = moment_weights * moment_values
                mass = tf.reduce_sum(weighted_values)
                mean = tf.reduce_sum(weighted_values * moment_physical[:, 0])
                second = tf.reduce_sum(weighted_values * tf.square(moment_physical[:, 0]))
                variance = tf.maximum(second - tf.square(mean), tf.constant(0.0, D))
                if with_score:
                    dot_target_log = dot_prior + select_scores(
                        active_model.observation_log_density_parameter_score(
                            theta, self.physical, observations[date], t=date), count)
                    dot_shift = tf.gather(dot_target_log, tf.argmax(log_target))
                    dot_target = 0.5 * target[:, None] * (dot_target_log - dot_shift[None, :])
                    rhs = tf.matmul(phi, self.weights[:, None] * dot_target, transpose_a=True)
                    dot_c = tf.linalg.solve(normal, rhs)
                    dot_z = 2.0 * tf.einsum("i,ij,jp->p", coefficients, mass_matrix, dot_c)
                    dot_log_z = dot_z / z
                    score = dot_log_z + dot_shift
                    propagation_values = tf.exp(tf.math.log(propagation_raw) - tf.math.log(z))
                    dot_propagation_values = (2.0 * propagation_h[:, None]
                                              * tf.matmul(propagation_phi, dot_c) / z
                                              - propagation_values[:, None] * dot_log_z[None, :])
                    dot_propagation = dot_propagation_values / tf.maximum(
                        propagation_values[:, None], tf.constant(1e-300, D))
                    dot_moment = (2.0 * moment_h[:, None] * tf.matmul(moment_phi, dot_c)
                                  / z - moment_raw[:, None] * dot_z[None, :] / tf.square(z))
                    dot_mass = tf.reduce_sum(moment_weights[:, None] * dot_moment, axis=0)
                    dot_mean_numerator = tf.reduce_sum(
                        (moment_weights * moment_physical[:, 0])[:, None] * dot_moment, axis=0)
                    dot_second_numerator = tf.reduce_sum(
                        (moment_weights * tf.square(moment_physical[:, 0]))[:, None] * dot_moment, axis=0)
                    # Preserve the legacy derivative report's quotient moments.
                    dot_mean = (dot_mean_numerator * mass - mean * dot_mass) / tf.square(mass)
                    dot_second = (dot_second_numerator * mass - second * dot_mass) / tf.square(mass)
                    dot_variance = dot_second - 2.0 * (mean / mass) * dot_mean
                    derivative_valid = (tf.math.is_finite(derivative_condition)
                                        & (derivative_condition <= derivative_config.solve_condition_number_veto))
                    derivative_finite = (tf.reduce_all(tf.math.is_finite(dot_c))
                                         & tf.reduce_all(tf.math.is_finite(dot_target_log))
                                         & tf.reduce_all(tf.math.is_finite(dot_propagation))
                                         & tf.reduce_all(tf.math.is_finite(score))
                                         & tf.reduce_all(tf.math.is_finite(dot_mean))
                                         & tf.reduce_all(tf.math.is_finite(dot_variance)))
                else:
                    dot_propagation = tf.zeros([previous_count, 0], D)
                    score = dot_log_z = dot_mean = dot_variance = tf.zeros([0], D)
                    derivative_valid = tf.constant(True)
                    derivative_finite = tf.constant(True)
                return {
                    "fit": fit_result, "initial": starting, "target": target,
                    "log_target": log_target, "shift": shift, "z": z,
                    "increment": increment, "propagation_log": propagation_log,
                    "propagation_dot": dot_propagation, "mass": mass, "mean": mean,
                    "variance": variance, "score": score, "dot_log_z": dot_log_z,
                    "dot_mean": dot_mean, "dot_variance": dot_variance,
                    "derivative_valid": derivative_valid,
                    "derivative_finite": derivative_finite,
                    "target_valid": tf.reduce_all(tf.math.is_finite(log_target)),
                    "retained_valid": (tf.reduce_all(tf.math.is_finite(propagation_log))
                                       & tf.math.is_finite(mean) & tf.math.is_finite(variance)),
                    "normalizer_valid": tf.math.is_finite(z) & (z > config.normalizer_floor)
                        & tf.math.is_finite(mass) & (mass > 0.0),
                }

            first = date_step(tf.constant(0), tf.zeros([previous_count], D),
                              tf.zeros([previous_count, parameter_count], D), self.fit.initial)
            histories = tf.nest.map_structure(lambda value: tf.TensorArray(
                value.dtype, horizon, element_shape=value.shape).write(0, value), first)

            def advance(date, previous_log, previous_dot, starting, histories):
                row = date_step(date, previous_log, previous_dot, starting)
                histories = tf.nest.map_structure(
                    lambda buffer, value: buffer.write(date, value), histories, row)
                return (date + 1, row["propagation_log"], row["propagation_dot"],
                        row["fit"]["cores"] if with_score else self.fit.initial, histories)

            _, _, _, _, histories = tf.while_loop(
                lambda date, *_: date < horizon, advance,
                (tf.constant(1), first["propagation_log"], first["propagation_dot"],
                 first["fit"]["cores"] if with_score else self.fit.initial, histories),
                maximum_iterations=horizon - 1, parallel_iterations=1,
            )
            history = tf.nest.map_structure(lambda value: value.stack(), histories)
            return {"history": history,
                    "log_likelihood": tf.reduce_sum(history["increment"]),
                    "score": tf.reduce_sum(history["score"], axis=0)}

        self.numerical = evaluate
        self.call = tf.function(lambda theta, observations: evaluate(theta, observations), input_signature=[
            tf.TensorSpec([model.parameter_dim()], D), tf.TensorSpec(observation_shape, D),
        ], jit_compile=jit_compile, autograph=False)

    def validate(self, histories):
        """Fail closed at the completed-program boundary, before reporting."""
        from bayesfilter.highdim.diagnostics import HighDimStatus

        valid = (histories["target_valid"] & histories["fit"]["valid"]
                 & histories["normalizer_valid"] & histories["retained_valid"]
                 & histories["derivative_valid"] & histories["derivative_finite"])
        if bool(tf.reduce_all(valid)):
            return
        first_failure = tf.argmax(tf.cast(~valid, tf.int32))
        row = tf.nest.map_structure(lambda value: value[first_failure], histories)
        if not bool(row["target_valid"]):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        if not bool(row["fit"]["valid"]):
            codes = row["fit"]["codes"]
            code = int(codes[tf.argmax(tf.cast(codes > 0, tf.int32))])
            if code == 6:
                raise ValueError(self.fit.gates[0]["status"])
            status = (HighDimStatus.CONDITION_NUMBER_VETO if code in (3, 4)
                      else HighDimStatus.NONFINITE_VALUE)
            raise ValueError(status.value)
        if not bool(row["normalizer_valid"]):
            raise ValueError(HighDimStatus.NORMALIZER_FLOOR_EXCEEDED.value)
        if not bool(row["retained_valid"]):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        if not bool(row["derivative_valid"]):
            raise ValueError(HighDimStatus.DERIVATIVE_SOLVE_FAILURE.value)
        if not bool(row["derivative_finite"]):
            raise ValueError(HighDimStatus.NONFINITE_RETAINED_DERIVATIVE.value)


def make_scalar_retained_program(model, config, observation_shape, *,
                                moment_order=257, propagation_order=321,
                                derivative_config=None, jit_compile=True):
    key = (id(model), id(config), tuple(observation_shape), moment_order,
           propagation_order, derivative_config, bool(jit_compile))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key][2]
    program = ScalarRetainedProgram(model, config, observation_shape, moment_order,
                                    propagation_order, derivative_config, jit_compile)
    _CACHE[key] = (model, config, program)
    if len(_CACHE) > 8:
        _CACHE.popitem(last=False)
    return program
