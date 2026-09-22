"""Native full-vector L-BFGS locator with exact callback accounting.

This internal dependency preserves the joint locator's optimizer and selection
semantics. Each factory owns its resource state. An enclosing controller must
own its handle exclusively; host callers serialize calls with invocation_lock.
There is no host numerical feedback or eager optimizer fallback.
"""

from threading import RLock

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.quadratic_geometry_control_tf import _callback_program

D = tf.float64
I = tf.int64
STATUSES = (
    "initial_target_invalid", "optimizer_exception", "evaluation_accounting_invalid",
    "evaluation_cap_exhausted", "optimizer_failed", "endpoint_target_invalid",
    "endpoint_objective_decrease", "converged", "iteration_limit",
)
SOURCES = ("initial", "optimizer_callback", "endpoint_replay")


def make_joint_center_program(callback, dimension, config, *, jit_compile=True):
    """Own initialization, L-BFGS, replay, incumbent and status in one graph.

    Resource variables are needed because TFP's objective interface has no
    auxiliary-state return. Reset every resource inside each call. Device
    placement belongs to construction; int64 counters also work on GPU/XLA.
    The public wall-clock diagnostic route cannot be embedded in this program.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if config.max_wall_seconds is not None:
        raise ValueError("native locator requires an independent parent wall deadline")
    target = _callback_program(callback, dimension, jit_compile)
    attempts = tf.Variable(0, dtype=I, trainable=False)
    rows = tf.Variable(0, dtype=I, trainable=False)
    best_value = tf.Variable(0., dtype=D, trainable=False)
    best_z = tf.Variable(tf.zeros([dimension], D), trainable=False)
    best_score = tf.Variable(tf.zeros([dimension], D), trainable=False)
    best_index = tf.Variable(-1, dtype=I, trainable=False)
    exhausted = tf.Variable(False, trainable=False)
    cap = config.max_objective_evaluations
    construction_error = {}

    def host_affine_position(initial, scale, z):
        # Original endpoint/record reconstruction ran outside the optimizer:
        # multiplication rounded before addition. Preserve that boundary so
        # replay cannot change a strict incumbent decision by fused multiply-add.
        return rounded_affine_position(initial, scale, z)

    def base(initial, value, score):
        return {
            "status": tf.constant(0), "endpoint_accepted": tf.constant(False),
            "initial_position": initial, "endpoint_position": initial,
            "initial_objective": value, "endpoint_objective": value,
            "initial_score": score, "endpoint_score": score,
            "best_position": initial, "best_objective": value, "best_score": score,
            "best_present": tf.constant(False), "best_source": tf.constant(-1),
            "best_callback_index": tf.constant(-1, I), "best_is_endpoint": tf.constant(False),
            "optimizer_converged": tf.constant(False), "optimizer_failed": tf.constant(False),
            "optimizer_iterations": tf.constant(0, I), "reported_evaluations": tf.constant(0, I),
            "callback_attempts": tf.constant(0, I), "optimizer_target_rows": tf.constant(0, I),
            "physical_target_rows": tf.constant(1, I), "cap_exhausted": tf.constant(False),
        }

    def optimize(initial, scale, initial_value, initial_score):
        def objective(z):
            attempted = attempts.assign_add(1)

            def evaluate():
                row = rows.assign_add(1)
                with tf.control_dependencies([row]):
                    value, score = target(initial + scale * z)
                    objective_value, gradient = -value, -scale * score
                # Match the original standardized gradient's inverse mapping,
                # including its floating-point rounding before eligibility.
                candidate_score = -gradient / scale
                eligible = tf.math.is_finite(value) & tf.reduce_all(
                    tf.math.is_finite(z) & tf.math.is_finite(candidate_score))

                def promote():
                    updates = (best_value.assign(value), best_z.assign(z),
                               best_score.assign(candidate_score), best_index.assign(row - 1))
                    with tf.control_dependencies(updates):
                        return tf.identity(objective_value), tf.identity(gradient)

                return tf.cond(eligible & (value > best_value.read_value()), promote,
                               lambda: (objective_value, gradient))

            def decline():
                flag = exhausted.assign(True)
                with tf.control_dependencies([flag]):
                    return tf.constant(float("inf"), D), tf.fill([dimension], tf.constant(float("nan"), D))

            with tf.control_dependencies([attempted]):
                return tf.cond(rows.read_value() < cap, evaluate, decline)

        # This catches Python construction errors only. Native compilation or
        # runtime errors propagate without executing an eager retry.
        try:
            optimizer = tfp.optimizer.lbfgs_minimize(objective,
                initial_position=tf.zeros([dimension], D),
                num_correction_pairs=config.num_correction_pairs,
                tolerance=tf.constant(config.gradient_tolerance, D),
                x_tolerance=tf.constant(config.x_tolerance, D),
                f_relative_tolerance=tf.constant(config.f_relative_tolerance, D),
                f_absolute_tolerance=tf.constant(config.f_absolute_tolerance, D),
                max_iterations=config.max_iterations,
                parallel_iterations=config.parallel_iterations,
                max_line_search_iterations=config.max_line_search_iterations)
        except Exception as exc:  # noqa: BLE001 - typed construction-only failure.
            construction_error.update(exception_type=type(exc).__name__)
            return {**base(initial, initial_value, initial_score), "status": tf.constant(1),
                    "best_present": tf.constant(True), "best_source": tf.constant(0)}

        endpoint = host_affine_position(initial, scale, optimizer.position)
        # Ensure tracking reads and endpoint replay follow every optimizer call.
        with tf.control_dependencies(tf.nest.flatten(optimizer)):
            callback_attempts, target_rows = attempts.read_value(), rows.read_value()
            cap_exhausted = exhausted.read_value()
            callback_value, callback_z = best_value.read_value(), best_z.read_value()
            callback_score, callback_index = best_score.read_value(), best_index.read_value()
            endpoint_value, endpoint_score = target(endpoint)
        endpoint_valid = tf.math.is_finite(endpoint_value) & tf.reduce_all(tf.math.is_finite(endpoint_score))
        reported = tf.cast(optimizer.num_objective_evaluations, I)
        accounting = (reported == callback_attempts) & (
            (~cap_exhausted & (callback_attempts == target_rows) & (target_rows <= cap)) |
            (cap_exhausted & (callback_attempts > target_rows) & (target_rows == cap)))
        accepted = accounting & ~cap_exhausted & ~optimizer.failed & endpoint_valid & (endpoint_value >= initial_value)
        positions = tf.stack((initial, host_affine_position(initial, scale, callback_z), endpoint))
        values = tf.stack((initial_value, callback_value, endpoint_value))
        scores = tf.stack((initial_score, callback_score, endpoint_score))
        eligible = tf.stack((tf.constant(True), callback_index >= 0, endpoint_valid))
        eligible &= tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(positions) & tf.math.is_finite(scores), 1)
        selected = tf.argmax(tf.where(eligible, values, tf.constant(float("-inf"), D)), output_type=tf.int32)
        status = tf.where(~accounting, 2, tf.where(cap_exhausted, 3,
            tf.where(optimizer.failed, 4, tf.where(~endpoint_valid, 5,
            tf.where(endpoint_value < initial_value, 6, tf.where(optimizer.converged, 7, 8))))))
        return {**base(initial, initial_value, initial_score),
            "status": status, "endpoint_accepted": accepted,
            "endpoint_position": endpoint, "endpoint_objective": endpoint_value, "endpoint_score": endpoint_score,
            "best_position": tf.gather(positions, selected), "best_objective": tf.gather(values, selected),
            "best_score": tf.gather(scores, selected), "best_present": tf.reduce_any(eligible),
            "best_source": selected, "best_callback_index": tf.where(selected == 1, callback_index, tf.constant(-1, I)),
            "best_is_endpoint": selected == 2, "optimizer_converged": optimizer.converged,
            "optimizer_failed": optimizer.failed, "optimizer_iterations": tf.cast(optimizer.num_iterations, I),
            "reported_evaluations": reported, "callback_attempts": callback_attempts,
            "optimizer_target_rows": target_rows, "physical_target_rows": target_rows + 2,
            "cap_exhausted": cap_exhausted}

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D)],
                 jit_compile=jit_compile, autograph=False)
    def locate(initial, scale):
        resets = (attempts.assign(0), rows.assign(0), best_value.assign(0.),
                  best_z.assign(tf.zeros([dimension], D)), best_score.assign(tf.zeros([dimension], D)),
                  best_index.assign(-1), exhausted.assign(False))
        with tf.control_dependencies(resets):
            initial_value, initial_score = target(initial)
        seeds = (best_value.assign(initial_value), best_score.assign(initial_score))
        valid = tf.math.is_finite(initial_value) & tf.reduce_all(tf.math.is_finite(initial_score))
        with tf.control_dependencies(seeds):
            result = tf.cond(valid, lambda: optimize(initial, scale, initial_value, initial_score),
                             lambda: base(initial, initial_value, initial_score))
        initial_scaled, endpoint_scaled = initial_score * scale, result['endpoint_score'] * scale
        result.update(initial_score_l2=tf.linalg.norm(initial_scaled),
            initial_score_max_abs=tf.reduce_max(tf.abs(initial_scaled)),
            endpoint_score_l2=tf.linalg.norm(endpoint_scaled),
            endpoint_score_max_abs=tf.reduce_max(tf.abs(endpoint_scaled)))
        return tf.nest.map_structure(tf.stop_gradient, result)

    locate.invocation_lock = RLock()
    locate.construction_error = construction_error
    return locate


def rounded_affine_position(initial, scale, z):
    """Retain the original two rounded operations, including signed zeros.

    nextafter(x, x) is an exact identity on finite binary64 values. Unlike an
    HLO optimization barrier, its integer implementation also blocks LLVM from
    contracting the multiply/add. This diagnostic initializer is disconnected
    from external differentiation, as the original host records were.
    """
    product = scale * z
    product = tf.math.nextafter(product, product)
    position = initial + product
    return tf.math.nextafter(position, position)


def joint_center_result(raw, *, jit_compile=True, construction_error=None):
    """Format a completed native record; it cannot steer numerical execution."""
    from bayesfilter.inference.joint_center import JointCenterResult

    present = bool(raw['best_present'])
    callback_index = int(raw['best_callback_index'])
    return JointCenterResult(
        status=STATUSES[int(raw['status'])], endpoint_accepted=bool(raw['endpoint_accepted']),
        initial_position=raw['initial_position'], endpoint_position=raw['endpoint_position'],
        initial_score=raw['initial_score'], endpoint_score=raw['endpoint_score'],
        initial_objective=float(raw['initial_objective']), endpoint_objective=float(raw['endpoint_objective']),
        best_evaluated_position=raw['best_position'] if present else None,
        best_evaluated_score=raw['best_score'] if present else None,
        best_evaluated_objective=float(raw['best_objective']) if present else None,
        best_evaluated_source=SOURCES[int(raw['best_source'])] if present else None,
        best_evaluated_callback_index=callback_index if callback_index >= 0 else None,
        best_evaluated_is_endpoint=bool(raw['best_is_endpoint']),
        initial_score_l2=float(raw['initial_score_l2']), endpoint_score_l2=float(raw['endpoint_score_l2']),
        initial_score_max_abs=float(raw['initial_score_max_abs']), endpoint_score_max_abs=float(raw['endpoint_score_max_abs']),
        optimizer_converged=bool(raw['optimizer_converged']), optimizer_failed=bool(raw['optimizer_failed']),
        optimizer_iterations=int(raw['optimizer_iterations']), reported_objective_evaluations=int(raw['reported_evaluations']),
        callback_attempts=int(raw['callback_attempts']), optimizer_target_rows=int(raw['optimizer_target_rows']),
        physical_target_rows=int(raw['physical_target_rows']), cap_exhausted=bool(raw['cap_exhausted']),
        wall_time_exhausted=False, jit_compile=jit_compile,
        exception_type=(construction_error or {}).get('exception_type') if int(raw['status']) == 1 else None)
