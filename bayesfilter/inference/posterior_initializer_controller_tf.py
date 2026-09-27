"""Internal complete prepared posterior initializer in one native program.

The public endpoint dispatches here after complete CPU/GPU comparisons.
Numerical authorities and the tracker are shared across every stage.
"""

from dataclasses import replace
from threading import RLock
from types import MethodType

import tensorflow as tf

from bayesfilter.inference.joint_center_tf import (
    make_joint_center_program,
    rounded_affine_position,
)
from bayesfilter.inference.posterior_candidate_ledger_tf import candidate_ledger_program
from bayesfilter.inference.posterior_cloud_preparation_tf import (
    PosteriorCloudPreparation,
    posterior_seed_keys,
)
from bayesfilter.inference.posterior_curvature_controller_tf import (
    make_posterior_curvature_program,
)
from bayesfilter.inference.posterior_local_initializer import (
    _cloud_row_counts,
    _EligibilityTrackingEvaluator,
)
from bayesfilter.inference.posterior_movement_tf import make_posterior_movement_program
from bayesfilter.inference.program_cache_scope import ProgramCacheScope
from bayesfilter.inference.quadratic_geometry_control_tf import _callback_program
from bayesfilter.inference.quadratic_geometry_full_tf import geometry_extents

D = tf.float64
I = tf.int32
_OWNER_LOCK = RLock()
_LAST_OWNER = None


def _same_callback(left, right):
    return left is right or (isinstance(left, MethodType) and isinstance(right, MethodType)
        and left.__self__ is right.__self__ and left.__func__ is right.__func__)


def posterior_initializer_owner(callback, dimension, config, movement_config, thresholds, *, device,
        batched_callback=None, eligibility_callback=None, batched_eligibility_callback=None):
    """Retain one owner by callback identity and configuration, never input values."""
    global _LAST_OWNER
    callbacks = (callback, batched_callback, eligibility_callback, batched_eligibility_callback)
    key = (dimension, config, movement_config, thresholds, device)
    with _OWNER_LOCK:
        previous = _LAST_OWNER
        if previous is not None and all(map(_same_callback, previous[0], callbacks)) and previous[1] == key:
            return previous[2]
        _LAST_OWNER = None
        del previous
        with tf.device(device):
            owner = PreparedPosteriorInitializer(callback, dimension, config, movement_config, thresholds,
                batched_callback=batched_callback, eligibility_callback=eligibility_callback,
                batched_eligibility_callback=batched_eligibility_callback)
        _LAST_OWNER = (callbacks, key, owner)
        return owner


def clear_posterior_initializer_cache():
    """Release Python ownership; no native executable eviction is implied."""
    global _LAST_OWNER
    with _OWNER_LOCK:
        _LAST_OWNER = None


@tf.function(input_signature=[tf.TensorSpec([None, None], D)], jit_compile=True, autograph=False)
def posterior_eigen_summary(matrix):
    """The existing public payload formula, with completed tensor outputs."""
    values = tf.linalg.eigvalsh(.5 * (matrix + tf.transpose(matrix)))
    return tf.nest.map_structure(tf.stop_gradient, {
        "minimum": tf.reduce_min(values), "maximum": tf.reduce_max(values),
        "condition_number": tf.reduce_max(values) / tf.reduce_min(values),
        "positive": tf.reduce_all(values > 0.),
    })


class PreparedPosteriorInitializer:
    """One independent owner for valid static configuration and prepared clouds.

    Configuration-only construction keeps caller inputs out of the traced
    resources. Local derivative engines must remain enabled during tracing;
    frozen outputs disconnect the unrelated caller tape. Initial and scale
    values are operands, including the bounded locator's chart state.
    """

    def __init__(self, callback, dimension, config, movement_config, thresholds, *,
            batched_callback=None, eligibility_callback=None, batched_eligibility_callback=None):
        jit_compile = config.locator_config.jit_compile
        if config.locator_config.max_wall_seconds is not None:
            raise ValueError("native posterior initialization requires an independent parent wall deadline")
        _, _, samples, directions = geometry_extents(dimension, movement_config)
        try:
            training, selection, audit = _cloud_row_counts(config, dimension)
            preparation_config, curvature_error = config, None
        except ValueError as error:
            # Only these static row-count errors occur at this original boundary.
            # Valid extents describe unused storage; no curvature will execute.
            curvature_error = str(error)
            preparation_config = replace(config, training_rows_per_replicate=None,
                selection_rows_per_replicate=None, audit_rows=None)
            training, selection, audit = _cloud_row_counts(preparation_config, dimension)
        self.curvature_configuration_error = curvature_error
        self.preparation = PosteriorCloudPreparation(dimension, preparation_config, movement_config)
        self.seed_keys = posterior_seed_keys(dimension, preparation_config, movement_config)
        self.radii = (tf.constant(movement_config.trust_radius, D), tf.constant(config.curvature_radius, D))
        movement_attempts, curvature_attempts = config.max_movement_attempts, config.max_curvature_attempts
        partitions, capacity = 2 * config.replicate_count + 1, max(training, selection, audit)
        scope = ProgramCacheScope()
        self.invocation_lock = RLock()
        self.dependency_scope = scope
        with scope.activate():
            tracker = _EligibilityTrackingEvaluator(callback, dimension=dimension,
                max_rows=config.max_exact_evaluations, batched_fn=batched_callback,
                eligibility_fn=eligibility_callback, batched_eligibility_fn=batched_eligibility_callback)
            origin = tf.Variable(tf.zeros([dimension], D), trainable=False)
            units = tf.Variable(tf.ones([dimension], D), trainable=False)
            radius = tf.constant(config.locator_box_radius, D)

            def chart(point):
                tanh = tf.math.tanh(point / radius)
                z = radius * tanh
                theta = origin.read_value() + units.read_value() * z
                value, score = tracker.scalar(theta)
                return value, score * units.read_value() * (1. - tf.square(tanh))

            target = _callback_program(tracker.scalar, dimension, jit_compile)
            locator = make_joint_center_program(chart, dimension, config.locator_config, jit_compile=jit_compile)
            # The public locator JIT option never disabled the movement or
            # curvature authorities; preserve their original XLA defaults.
            movement = make_posterior_movement_program(tracker, dimension, config, movement_config,
                jit_compile=True)
            curvature = (make_posterior_curvature_program(tracker, dimension, config, thresholds,
                jit_compile=True) if curvature_error is None else None)
            choose = candidate_ledger_program(2, dimension, jit_compile=jit_compile)
            locator_template = locator.get_concrete_function().structured_outputs
            movement_template = movement.get_concrete_function().structured_outputs
            curvature_template = (curvature.get_concrete_function().structured_outputs if curvature is not None else {
                "status": tf.constant(0, I), "center": tf.zeros([dimension], D),
                "value": tf.constant(0., D), "score": tf.zeros([dimension], D),
                "covariance_theta": tf.zeros([dimension, dimension], D),
                "marginal": tf.zeros([dimension], D),
                "fit": {"fit": {"selection": {"precision": tf.zeros([dimension, dimension], D)}}}})

        @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
            tf.TensorSpec([movement_attempts, directions, dimension], D),
            tf.TensorSpec([movement_attempts, samples, dimension], D),
            tf.TensorSpec([movement_attempts, 2], I),
            tf.TensorSpec([curvature_attempts, partitions, capacity, dimension], D)],
            jit_compile=jit_compile, autograph=False)
        def initialize(initial, scale, raw_directions, movement_offsets, permutation_keys, curvature_offsets):
            resets = (tracker.evaluated_rows.assign(0), tracker.invalid_rows.assign(0),
                tracker.mismatch_rows.assign(0), tracker.budget_exhausted.assign(False),
                origin.assign(initial), units.assign(scale))
            with tf.control_dependencies(resets):
                initial_value, initial_score = target(initial)
            initial_ok = tf.math.is_finite(initial_value) & tf.reduce_all(tf.math.is_finite(initial_score))
            location = tf.cond(initial_ok, lambda: locator(tf.zeros([dimension], D), tf.ones([dimension], D)),
                lambda: tf.nest.map_structure(lambda x: tf.zeros(x.shape, x.dtype), locator_template))

            def replay():
                z = radius * tf.math.tanh(location["best_position"] / radius)
                position = rounded_affine_position(initial, scale, z)
                value, score = target(position)
                valid = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
                return position, value, score, valid

            with tf.control_dependencies(tf.nest.flatten(location)):
                point, value, score, valid = tf.cond(initial_ok & location["best_present"], replay,
                    lambda: (initial, tf.constant(0., D), tf.zeros([dimension], D), tf.constant(False)))
            positions, values, scores = tf.stack((initial, point)), tf.stack((initial_value, value)), tf.stack((initial_score, score))
            initial_count = tf.cast(initial_ok, I) + tf.cast(valid, I)
            ledger = choose(positions, values, scores, tf.stack((initial_ok, valid)), scale, initial_count)
            with tf.control_dependencies((value, score)):
                mismatch = tracker.mismatch_rows.read_value() > 0
                exhausted = tracker.budget_exhausted.read_value()
            tracker_status = tf.where(mismatch, 1, tf.where(exhausted, 2, 0))
            movement_ran = initial_ok & (tracker_status == 0)
            moved = tf.cond(movement_ran,
                lambda: movement(ledger["position"], ledger["value"], ledger["score"], scale,
                    raw_directions, movement_offsets, permutation_keys),
                lambda: tf.nest.map_structure(lambda x: tf.zeros(x.shape, x.dtype), movement_template))
            curvature_ran = movement_ran & (moved["status"] == 0)
            empty_curvature = tf.nest.map_structure(lambda x: tf.zeros(x.shape, x.dtype), curvature_template)
            curved = (tf.cond(curvature_ran,
                lambda: curvature(moved["center"], moved["value"], moved["score"], scale, curvature_offsets),
                lambda: empty_curvature) if curvature is not None else empty_curvature)
            curvature_completed = curvature_ran & (curvature is not None)
            accepted = curvature_ran & (curved["status"] == 1)
            empty_summary = {"minimum": tf.constant(0., D), "maximum": tf.constant(0., D),
                "condition_number": tf.constant(0., D), "positive": tf.constant(False)}
            precision_summary = tf.cond(accepted,
                lambda: posterior_eigen_summary.python_function(curved["fit"]["fit"]["selection"]["precision"]),
                lambda: empty_summary)
            covariance_summary = tf.cond(accepted,
                lambda: posterior_eigen_summary.python_function(curved["covariance_theta"]), lambda: empty_summary)
            center = tf.where(initial_ok, ledger["position"], initial)
            center_value = tf.where(initial_ok, ledger["value"], initial_value)
            center_score = tf.where(initial_ok, ledger["score"], initial_score)
            center = tf.where(curvature_completed, curved["center"], tf.where(movement_ran, moved["center"], center))
            center_value = tf.where(curvature_completed, curved["value"], tf.where(movement_ran, moved["value"], center_value))
            center_score = tf.where(curvature_completed, curved["score"], tf.where(movement_ran, moved["score"], center_score))
            with tf.control_dependencies(tf.nest.flatten(curved)):
                accounting = {"evaluated_rows": tracker.evaluated_rows.read_value(),
                    "invalid_rows": tracker.invalid_rows.read_value(), "mismatch_rows": tracker.mismatch_rows.read_value(),
                    "budget_exhausted": tracker.budget_exhausted.read_value()}
            return tf.nest.map_structure(tf.stop_gradient, {
                "stage": tf.where(curvature_ran, 3 if curvature is not None else 4,
                    tf.where(movement_ran, 2, tf.where(initial_ok, 1, 0))),
                "accepted": accepted, "initial_ok": initial_ok, "initial_positions": positions,
                "initial_values": values, "initial_scores": scores, "initial_ledger": ledger,
                "tracker_status_after_locator": tracker_status, "locator": location,
                "movement": moved, "curvature": curved, "accounting": accounting,
                "center": center, "value": center_value, "score": center_score, "scale": scale,
                "precision_eigen_summary": precision_summary, "covariance_eigen_summary": covariance_summary,
            })

        with scope.activate():
            initialize.get_concrete_function()
        self.compiled = initialize

    def prepare_clouds(self):
        return self.preparation(*self.seed_keys, *self.radii)

    def __call__(self, initial, scale, directions, movement_offsets, permutation_keys, curvature_offsets):
        with self.invocation_lock:
            result = self.compiled(initial, scale, directions, movement_offsets, permutation_keys, curvature_offsets)
            int(result["stage"])
            return result
