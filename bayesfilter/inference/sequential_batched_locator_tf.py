"""Native batched localization with optional buffered objective diagnostics.

Progress delivery occurs after the compiled call. Optional trace storage is
bounded; overflow is an explicit reporting failure, never silent truncation or
a change to the optimizer. Host-only target callbacks are unsupported.
"""

import threading
from functools import lru_cache

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.sequential_selection_tf import selection_numerics

D = tf.float64
MAX_TRACE_BYTES = 16 * 1024 * 1024
MAX_TRACE_EVENTS = 4096


def _max_abs_observation(values):
    # XLA's all-NaN max reduction may yield its -inf identity. Preserve the
    # original host diagnostic's NaN rather than reporting a finite/inf score.
    return tf.where(tf.reduce_any(tf.math.is_nan(values), axis=1),
        tf.constant(float("nan"), D), tf.reduce_max(tf.abs(values), axis=1))


def _program(scalar, batched, count, dimension, box_radius, tolerance,
             max_iterations, max_line_search_iterations, stopping_condition,
             jit_compile, recorder=None):
    stopping = (tfp.optimizer.converged_all if stopping_condition == "converged_all"
                else tfp.optimizer.converged_any)

    @tf.function(input_signature=[tf.TensorSpec([count, dimension], D),
                                  tf.TensorSpec([dimension], D)],
                 jit_compile=jit_compile, autograph=False)
    def locate(starts, scale):
        if recorder is not None:
            recorder.calls.assign(0)
            recorder.rows.assign(tf.zeros_like(recorder.rows))
        radius = tf.constant(box_radius, D)

        def objective(unconstrained):
            z = radius * tf.math.tanh(unconstrained / radius)
            values, scores = batched(starts + scale[None, :] * z)
            values = tf.ensure_shape(tf.convert_to_tensor(values, D), [count])
            scores = tf.ensure_shape(tf.convert_to_tensor(scores, D), [count, dimension])
            derivative = 1.0 - tf.square(z / radius)
            if recorder is not None:
                recorder.record(tf.concat([values[:, None],
                    _max_abs_observation(scores * scale[None, :])[:, None],
                    _max_abs_observation(scores * scale[None, :] * derivative)[:, None], z], axis=1))
            return -values, -(scores * scale[None, :] * derivative)

        optimizer = tfp.optimizer.lbfgs_minimize(objective,
            initial_position=tf.zeros([count, dimension], D),
            tolerance=tf.constant(tolerance, D), max_iterations=max_iterations,
            max_line_search_iterations=max_line_search_iterations,
            parallel_iterations=1, stopping_condition=stopping)
        endpoint_z = radius * tf.math.tanh(optimizer.position / radius)
        endpoints = starts + scale[None, :] * endpoint_z
        endpoint_values, endpoint_scores = batched(endpoints)
        endpoint_values = tf.ensure_shape(tf.convert_to_tensor(endpoint_values, D), [count])
        endpoint_scores = tf.ensure_shape(tf.convert_to_tensor(endpoint_scores, D), [count, dimension])
        valid = tf.math.is_finite(endpoint_values) & tf.reduce_all(tf.math.is_finite(endpoint_scores), axis=1)
        positions = tf.concat([starts, endpoints], axis=0)
        eligible = tf.concat([tf.ones([count], tf.bool), valid], axis=0)
        values = tf.TensorArray(D, size=2 * count, element_shape=[])
        scores = tf.TensorArray(D, size=2 * count, element_shape=[dimension])

        def evaluate(point):
            value, score = scalar(point)
            value = tf.reshape(tf.convert_to_tensor(value, D), [])
            score = tf.reshape(tf.convert_to_tensor(score, D), [-1])
            if score.shape != (dimension,):
                raise ValueError("target score must have one entry per parameter")
            return value, score

        def replay(index, values, scores):
            value, score = tf.cond(eligible[index], lambda: evaluate(positions[index]),
                lambda: (tf.constant(float("nan"), D), tf.zeros([dimension], D)))
            return index + 1, values.write(index, value), scores.write(index, score)

        _, values, scores = tf.while_loop(lambda index, *_: index < 2 * count, replay,
            (tf.constant(0), values, scores), maximum_iterations=2 * count, parallel_iterations=1)
        result = {"selected": selection_numerics(positions, values.stack(), scores.stack()),
            "endpoint_finite": valid, "endpoint_standardized_norm": tf.linalg.norm(endpoint_z, axis=1),
            "converged": optimizer.converged, "failed": optimizer.failed,
            "iterations": optimizer.num_iterations, "objective_calls": optimizer.num_objective_evaluations,
            "exact_evaluations": tf.math.count_nonzero(eligible, dtype=tf.int32),
            "objective_evaluations": (optimizer.num_objective_evaluations + 1) * count}
        if recorder is not None:
            result.update(trace=recorder.rows.read_value(), trace_count=recorder.calls.read_value(),
                trace_overflow=recorder.calls.read_value() > recorder.capacity)
        # These are completed preparation records, disconnected from external
        # differentiation in the original public endpoint. Internal analytical
        # target scores and optimizer updates are unchanged.
        return tf.nest.map_structure(tf.stop_gradient, result)

    return locate


@lru_cache(maxsize=64)
def batched_locator_program(scalar, batched, count, dimension, box_radius, tolerance,
                            max_iterations, max_line_search_iterations,
                            stopping_condition, *, jit_compile=True):
    return _program(scalar, batched, count, dimension, box_radius, tolerance,
        max_iterations, max_line_search_iterations, stopping_condition, jit_compile)


class BufferedBatchedLocator:
    """Serialize access to optional device-resident objective observations."""

    def __init__(self, scalar, batched, count, dimension, box_radius, tolerance,
                 max_iterations, max_line_search_iterations, stopping_condition,
                 *, device, jit_compile=True, capacity=None):
        if count <= 0 or dimension <= 0:
            raise ValueError("trace dimensions must be positive")
        limit = min(MAX_TRACE_EVENTS, MAX_TRACE_BYTES // (8 * count * (dimension + 3)))
        self.capacity = limit if capacity is None else int(capacity)
        if not 0 < self.capacity <= limit:
            raise ValueError("trace capacity must fit the 16 MiB / 4096-event observation limit")
        self.lock = threading.Lock()
        with tf.device(device):
            # TensorFlow pins int32 variables to host memory. int64 keeps this
            # counter with its trace buffer inside the same GPU/XLA program.
            self.calls = tf.Variable(0, trainable=False, dtype=tf.int64)
            self.rows = tf.Variable(tf.zeros([self.capacity, count, dimension + 3], D), trainable=False)
        self.compiled = _program(scalar, batched, count, dimension, box_radius, tolerance,
            max_iterations, max_line_search_iterations, stopping_condition, jit_compile, self)

    def record(self, row):
        index = self.calls.read_value()

        def save():
            update = self.rows.scatter_nd_update(tf.reshape(index, [1, 1]), row[None, :])
            with tf.control_dependencies([update]):
                return tf.constant(0)

        saved = tf.cond(index < self.capacity, save, lambda: tf.constant(0))
        with tf.control_dependencies([saved]):
            self.calls.assign_add(1)

    def __call__(self, starts, scale):
        with self.lock:
            result = self.compiled(starts, scale)
            # Completion gives each invocation immutable output snapshots before
            # another caller resets the shared resources. No target replay.
            if bool(result["trace_overflow"]):
                raise RuntimeError(f"Batched locator progress capacity {self.capacity} exceeded; "
                    f"{int(result['trace_count'])} objective calls completed. No partial progress was delivered.")
            return result


@lru_cache(maxsize=4)
def buffered_batched_locator_program(scalar, batched, count, dimension, box_radius,
                                    tolerance, max_iterations, max_line_search_iterations,
                                    stopping_condition, *, device, jit_compile=True):
    return BufferedBatchedLocator(scalar, batched, count, dimension, box_radius, tolerance,
        max_iterations, max_line_search_iterations, stopping_condition,
        device=device, jit_compile=jit_compile)
