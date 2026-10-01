"""Enclosing quadratic initializer with ordered native recentering.

Cloud preparation and completed-record formatting live outside this boundary.
Each factory owns its joint-locator resources exclusively. Host owners serialize
invocations; no target, incumbent, terminal or mass decision returns to Python.
"""

from threading import RLock

import tensorflow as tf

from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.joint_center_tf import make_joint_center_program
from bayesfilter.inference.mass_matrix_tf import precision_program
from bayesfilter.inference.quadratic_geometry_control_tf import validate_callback_graph
from bayesfilter.inference.quadratic_geometry_full_tf import make_geometry_program
from bayesfilter.inference.quadratic_map_covariance import (
    _material_move,
    _precision_transform,
    _scale_summary,
)

D = tf.float64
EVALUATION_STATUSES = ('finite', 'position_nonfinite', 'exception', 'score_shape_mismatch', 'nonfinite')
# -1 is the nonterminal native state. Strings are only emitted after completion.
STATUSES = ('initial_value_or_score_nonfinite', 'center_value_or_score_nonfinite',
    'geometry_rejected', 'maximum_refinement_steps_after_exact_incumbent_move',
    'mass_matrix_regularization_failed', 'maximum_refinement_steps_without_terminal_score',
    'center_refinement_rejected', 'usable', 'covariance_center_mismatch_requires_refit')


def make_initializer_target(callback, dimension, *, jit_compile=True):
    """Retain the wrapper's typed validation without host numerical feedback."""
    @tf.function(input_signature=[tf.TensorSpec([dimension], D)],
                 autograph=False, jit_compile=jit_compile)
    def evaluate(point):
        missing = tf.fill([dimension], tf.constant(float('nan'), D))

        def call():
            try:
                value, score = callback(point)
                value = tf.reshape(tf.convert_to_tensor(value, D), [])
                score = tf.reshape(tf.convert_to_tensor(score, D), [-1])
            except Exception:  # noqa: BLE001 - construction/conversion boundary only.
                return tf.constant(float('nan'), D), missing, tf.constant(2)
            if score.shape != (dimension,):
                return value, missing, tf.constant(3)
            valid = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
            return value, score, tf.where(valid, 0, 4)

        return tf.cond(tf.reduce_all(tf.math.is_finite(point)), call,
            lambda: (tf.constant(float('nan'), D), missing, tf.constant(1)))

    validate_callback_graph(evaluate, jit_compile=jit_compile)
    return evaluate


def make_quadratic_initializer_program(callback, dimension, locator_config, geometry_config,
        mass_config, *, iterative_config=None, batched_callback=None, jit_compile=True):
    """Compile initialization through final geometry and mass decisions.

    A non-None iterative_config selects the original bounded recentering route.
    The single-shot route omits the extra per-iteration center evaluation.
    Resource-bearing locator handles are private to this enclosing program.
    """
    iterative = iterative_config is not None
    if iterative and not geometry_config.constrain_center_refinement_to_trust_region:
        raise ValueError('iterative quadratic recentering requires constrained center refinement')
    target = make_initializer_target(callback, dimension, jit_compile=jit_compile)
    geometry = make_geometry_program(callback, dimension, geometry_config,
        batched_callback=batched_callback, jit_compile=jit_compile)
    geometry_schema = geometry.get_concrete_function().structured_outputs
    fit_possible = bool(geometry_schema['fit_result'])
    locator = (make_joint_center_program(callback, dimension, JointCenterLocatorConfig(
        max_iterations=locator_config.max_iterations, gradient_tolerance=locator_config.tolerance,
        parallel_iterations=locator_config.parallel_iterations, jit_compile=jit_compile,
        max_objective_evaluations=max(601, locator_config.max_iterations * 25 + 1)),
        jit_compile=jit_compile) if locator_config.enabled else None)
    locator_schema = locator.get_concrete_function().structured_outputs if locator is not None else {}
    mass = precision_program(dimension, dense=mass_config.dense, jit_compile=jit_compile)
    mass_schema = mass.get_concrete_function().structured_outputs
    capacity = iterative_config.max_refinement_steps + 1 if iterative else 1

    def zero(schema):
        return tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), schema)

    def geometry_decisions(raw, center, scale):
        if not fit_possible:
            return tf.constant(False), tf.constant(False), center, tf.constant(False), center, tf.eye(dimension, dtype=D)
        fit = raw['fit_result']
        accepted = (raw['stage'] == 3) & (fit['status'] == 1)
        promoted = accepted & fit['has_incumbent'] & _material_move(fit['best_position'],
            center, scale, fit['best_value'], raw['center_value'],
            tf.constant(geometry_config.center_log_prob_tolerance, D))
        return (accepted, promoted, fit['best_position'], fit['refinement']['accepted'],
                fit['refinement']['refined_center'], fit['fit']['precision'])

    def locate(initial, initial_value, initial_score, valid):
        initial_norm = tf.linalg.norm(initial_score)
        if locator is None:
            return initial, {'raw': {}, 'accepted': tf.constant(False),
                'candidate_finite': valid, 'candidate_value': initial_value,
                'candidate_norm': initial_norm, 'initial_norm': initial_norm}
        raw = tf.cond(valid, lambda: locator(initial, tf.ones([dimension], D)), lambda: zero(locator_schema))
        value = tf.where(raw['best_present'], raw['best_objective'], initial_value)
        score = tf.where(raw['best_present'], raw['best_score'], initial_score)
        finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
        # The wrapper deliberately permits a better callback after endpoint failure.
        accepted = valid & finite & (value >= initial_value - locator_config.log_prob_tolerance)
        accepted &= ~(raw['best_present'] & (raw['best_source'] == 0))
        candidate = tf.where(raw['best_present'], raw['best_position'], initial)
        return tf.where(accepted, candidate, initial), {'raw': raw, 'accepted': accepted,
            'candidate_finite': finite, 'candidate_value': value,
            'candidate_norm': tf.linalg.norm(score), 'initial_norm': initial_norm}

    @tf.function(input_signature=geometry.input_signature, autograph=False, jit_compile=jit_compile)
    def initialize(initial, scale, directions, offsets, permutation_seed):
        initial_value, initial_score, initial_status = target(initial)
        valid = initial_status == 0
        with tf.control_dependencies([initial_value, initial_score, initial_status]):
            position, located = locate(initial, initial_value, initial_score, valid)
        row_schema = {'geometry': geometry_schema, 'center': initial, 'value': initial_value,
            'score_norm': initial_value, 'score_max': initial_value,
            'terminal': valid, 'promoted': valid}
        # Fixed-capacity buffers retain only completed records. Unwritten slots
        # are initialized explicitly and never consumed as observations.
        history = tf.nest.map_structure(lambda value: tf.zeros([capacity, *value.shape], value.dtype), row_schema)

        def body(index, current, status, count, records, final_precision):
            if iterative:
                value, score, evaluation_status = target(current)
                finite = evaluation_status == 0
                scaled = score * scale
                score_norm, score_max = tf.linalg.norm(scaled), tf.reduce_max(tf.abs(scaled))
                terminal = score_max <= iterative_config.terminal_score_max_abs
            else:
                value, score_norm, score_max = initial_value, tf.constant(0., D), tf.constant(0., D)
                evaluation_status, finite, terminal = tf.constant(0), tf.constant(True), tf.constant(True)

            def fit():
                with tf.control_dependencies([value, evaluation_status]):
                    raw = geometry(current, scale, directions, offsets, permutation_seed)
                accepted, promoted, best, refined, refined_center, precision = geometry_decisions(raw, current, scale)
                if iterative:
                    limit = index == capacity - 1
                    next_status = tf.where(~accepted, 2, tf.where(promoted,
                        tf.where(limit, 3, -1), tf.where(terminal, 7,
                        tf.where(limit, 5, tf.where(~refined, 6, -1)))))
                    next_center = tf.where(next_status == -1, tf.where(promoted, best, refined_center), current)
                else:
                    next_status = tf.where(~accepted, 2, tf.where(promoted, 8, 7))
                    next_center = current
                record = {'geometry': raw, 'center': current, 'value': value,
                    'score_norm': score_norm, 'score_max': score_max, 'terminal': terminal, 'promoted': promoted}
                updated = tf.nest.map_structure(lambda buffer, item: buffer if item.shape.num_elements() == 0
                    else tf.tensor_scatter_nd_update(buffer, index[None, None], item[None]), records, record)
                return next_center, next_status, count + 1, updated, precision

            next_center, next_status, count, records, final_precision = tf.cond(finite, fit,
                lambda: (current, tf.constant(1), count, records, final_precision))
            return index + 1, next_center, next_status, count, records, final_precision

        index, final_position, status, count, history, fitted_precision = tf.while_loop(
            lambda _index, _center, state, *_: state == -1, body,
            (tf.constant(0), position, tf.where(valid, -1, 0), tf.constant(0), history,
             tf.eye(dimension, dtype=D)), maximum_iterations=capacity, parallel_iterations=1)

        def build_mass():
            theta_precision = _precision_transform(fitted_precision, scale)
            return mass(theta_precision, tf.constant(mass_config.jitter, D),
                tf.constant(mass_config.eigenvalue_floor or 0., D),
                tf.constant(mass_config.max_condition_number or 0., D))

        mass_attempted = status == 7
        mass_result = tf.cond(mass_attempted, build_mass, lambda: zero(mass_schema))
        _, _, _, flags, diagonal_valid = mass_result
        mass_valid = tf.reduce_all(flags[:3]) & (tf.constant(mass_config.dense) | diagonal_valid)
        status = tf.where(mass_attempted & ~mass_valid, 4, status)
        return tf.nest.map_structure(tf.stop_gradient, {
            'status': status, 'initial_status': initial_status, 'initial_value': initial_value,
            'initial_score': initial_score, 'locator_position': position, 'locator': located,
            'history': history, 'fit_count': count, 'last_index': index - 1,
            'final_position': final_position, 'mass_attempted': mass_attempted, 'mass': mass_result,
            'scale_summary': _scale_summary(scale)})

    initialize.invocation_lock = RLock()
    initialize.locator_construction_error = {} if locator is None else locator.construction_error
    return initialize
