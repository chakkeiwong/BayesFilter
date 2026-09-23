"""Reusable dense initializer recurrence over prepared attempt clouds.

Locator, ordered cloud evaluation and validated fitting execute within one XLA
boundary. Random-cloud preparation and completed report/file output are separate.
The returned geometry is frozen, as in the original host-materialized initializer.
"""

from threading import RLock

import tensorflow as tf

from bayesfilter.inference.batched_local_center_tf import BatchedLocalCenterProgram
from bayesfilter.inference.dense_initializer_attempt_tf import (
    make_dense_initializer_attempt_program,
)

D = tf.float64


class DenseInitializerProgram:
    """Own the callback, locator resources and fixed-shape attempt recurrence.

    Construction settings are static metadata. Initial center, coordinate scale
    and all prepared offset clouds are operands and may change without retracing.
    Calls through this owner serialize access to the resettable locator resources.
    """

    def __init__(self, callback, dimension, replicates, training_rows, selection_rows,
            audit_rows, *, locator_config, thresholds, max_attempts,
            max_exact_evaluations, center_score_max, factor_max=2,
            dense_eigenvalue_floor=1e-8, max_condition_number=1e8,
            shrinkage_weights=(0., .25, .5, .75, 1.), structured_target_family=None):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.invocation_lock = RLock()
        self.planned_rows = max_attempts * (locator_config.maximum_physical_rows_multiplier
            + replicates * (training_rows + selection_rows) + audit_rows)
        if self.planned_rows > max_exact_evaluations:
            raise ValueError("dense initializer planned exact rows exceed max_exact_evaluations")
        self.locator = BatchedLocalCenterProgram(callback, 1, dimension, locator_config)
        self.attempt = make_dense_initializer_attempt_program(callback, dimension, replicates,
            training_rows, selection_rows, audit_rows, thresholds=thresholds, factor_max=factor_max,
            dense_eigenvalue_floor=dense_eigenvalue_floor, max_condition_number=max_condition_number,
            shrinkage_weights=shrinkage_weights, structured_target_family=structured_target_family,
            jit_compile=locator_config.jit_compile)
        locate, evaluate = self.locator.compiled, self.attempt
        template = evaluate.get_concrete_function().structured_outputs
        zero_attempt = tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), template)
        location_template = locate.get_concrete_function().structured_outputs
        signature = [tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
            tf.TensorSpec([max_attempts, 2 * replicates + 1,
                max(training_rows, selection_rows, audit_rows), dimension], D)]

        @tf.function(input_signature=signature, jit_compile=locator_config.jit_compile, autograph=False)
        def run(initial, scale, offsets):
            def allocate(value):
                return tf.zeros([max_attempts, *value.shape], value.dtype)

            initial_state = {"attempt_count": tf.constant(0), "status_code": tf.constant(0),
                "center": initial, "exact_rows": tf.constant(0, tf.int64),
                "location_history": tf.nest.map_structure(allocate, location_template),
                "attempt_history": tf.nest.map_structure(allocate, template),
                "scaled_score_norm_history": tf.zeros([max_attempts], D),
                "cloud_ran_history": tf.zeros([max_attempts], tf.bool),
                "initial_output_shift": tf.zeros([dimension], D),
                "initial_output_scale_log": tf.zeros([dimension], D)}

            def step(state):
                index = state["attempt_count"]
                location = locate(state["center"][None, :], scale)
                score_norm = tf.cond(location["accepted"],
                    lambda: tf.linalg.norm(location["center_score"] * scale), lambda: tf.constant(0., D))
                cloud_ran = location["accepted"] & tf.math.is_finite(score_norm) & (score_norm <= center_score_max)
                attempt = tf.cond(cloud_ran,
                    lambda: evaluate(location["center"], scale, location["center_value"],
                        location["center_score"], tf.gather(offsets, index)), lambda: zero_attempt)
                # 0 retries a strictly better cloud incumbent.  Exhaustion is
                # resolved after the loop, preserving the last partial record.
                mapped = tf.gather(tf.constant([0, 3, 0, 5, 7, 8, 9]), attempt["status_code"])
                mapped = tf.where((attempt["status_code"] == 4) &
                    (attempt["fit"]["fit_error_code"] != 0), 6, mapped)
                status = tf.where(~location["accepted"], 1, tf.where(~cloud_ran, 2, mapped))
                center = tf.where(location["accepted"], location["center"], state["center"])
                center = tf.where(cloud_ran & (attempt["status_code"] == 2), attempt["candidate_center"], center)

                def append(history, value):
                    return tf.tensor_scatter_nd_update(history, tf.reshape(index, [1, 1]), value[None])

                return ({"attempt_count": index + 1, "status_code": status, "center": center,
                    "exact_rows": state["exact_rows"] + location["physical_target_rows"] +
                        attempt["exact_evaluation_rows"],
                    "location_history": tf.nest.map_structure(append, state["location_history"], location),
                    "attempt_history": tf.nest.map_structure(append, state["attempt_history"], attempt),
                    "scaled_score_norm_history": append(state["scaled_score_norm_history"], score_norm),
                    "cloud_ran_history": append(state["cloud_ran_history"], cloud_ran),
                    "initial_output_shift": attempt["initial_output_shift"],
                    "initial_output_scale_log": attempt["initial_output_scale_log"]},)

            completed, = tf.while_loop(lambda state: (state["attempt_count"] < max_attempts) &
                (state["status_code"] == 0), step, (initial_state,),
                maximum_iterations=max_attempts, parallel_iterations=1)
            completed["status_code"] = tf.where(completed["status_code"] == 0, 4, completed["status_code"])
            completed["usable"] = completed["status_code"] == 9
            completed["jit_compile"] = tf.constant(locator_config.jit_compile, tf.bool)
            return tf.nest.map_structure(tf.stop_gradient, completed)

        self.compiled = run

    def __call__(self, initial, scale, offsets):
        with self.invocation_lock:
            result = self.compiled(initial, scale, offsets)
            int(result["status_code"])
            return result
