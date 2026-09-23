"""Internal operand-bound checkpoint and same-state continuation for L-BFGS.

This program owns numerical stages only. The external validator is a host
boundary and must run under invocation_lock between checkpoint and continuation.
Public integration is qualified separately; no eager retry is provided here.
"""

from threading import RLock

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.optimizer.lbfgs import LBfgsOptimizerResults

from bayesfilter.inference.joint_center_tf import STATUSES, rounded_affine_position
from bayesfilter.inference.quadratic_geometry_control_tf import _callback_program

D = tf.float64
I = tf.int64
SOURCES = ("initial", "optimizer_callback", "checkpoint_replay", "endpoint_replay")


class StagedJointCenterProgram:
    """Own reusable numerical stages; continuation carries all state explicitly."""

    def __init__(self, callback, dimension, config):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if config.max_wall_seconds is not None:
            raise ValueError("native staged locator requires an independent parent wall deadline")
        self.invocation_lock = RLock()
        self.construction_errors = {}
        errors = self.construction_errors
        target = _callback_program(callback, dimension, config.jit_compile)
        attempts = tf.Variable(0, dtype=I, trainable=False)
        rows = tf.Variable(0, dtype=I, trainable=False)
        best_value = tf.Variable(0., dtype=D, trainable=False)
        best_z = tf.Variable(tf.zeros([dimension], D), trainable=False)
        best_score = tf.Variable(tf.zeros([dimension], D), trainable=False)
        best_index = tf.Variable(-1, dtype=I, trainable=False)
        exhausted = tf.Variable(False, trainable=False)
        cap = config.max_objective_evaluations

        def empty_optimizer():
            return LBfgsOptimizerResults(tf.constant(False), tf.constant(False),
                tf.constant(0), tf.constant(0), tf.zeros([dimension], D),
                tf.constant(0., D), tf.zeros([dimension], D),
                tf.zeros([config.num_correction_pairs, dimension], D),
                tf.zeros([config.num_correction_pairs, dimension], D))

        def state():
            return {"attempts": attempts.read_value(), "rows": rows.read_value(),
                "value": best_value.read_value(), "z": best_z.read_value(),
                "score": best_score.read_value(), "index": best_index.read_value(),
                "exhausted": exhausted.read_value()}

        def restore(saved):
            return (attempts.assign(saved["attempts"]), rows.assign(saved["rows"]),
                best_value.assign(saved["value"]), best_z.assign(saved["z"]),
                best_score.assign(saved["score"]), best_index.assign(saved["index"]),
                exhausted.assign(saved["exhausted"]))

        def objective(initial, scale):
            def evaluate(z):
                attempted = attempts.assign_add(1)

                def available():
                    row = rows.assign_add(1)
                    with tf.control_dependencies([row]):
                        value, score = target(initial + scale * z)
                        loss, gradient = -value, -scale * score
                    candidate_score = -gradient / scale
                    valid = tf.math.is_finite(value) & tf.reduce_all(
                        tf.math.is_finite(z) & tf.math.is_finite(candidate_score))

                    def promote():
                        updates = (best_value.assign(value), best_z.assign(z),
                            best_score.assign(candidate_score), best_index.assign(row - 1))
                        with tf.control_dependencies(updates):
                            return tf.identity(loss), tf.identity(gradient)

                    return tf.cond(valid & (value > best_value.read_value()),
                        promote, lambda: (loss, gradient))

                def capped():
                    flag = exhausted.assign(True)
                    with tf.control_dependencies([flag]):
                        return tf.constant(float("inf"), D), tf.fill([dimension], tf.constant(float("nan"), D))

                with tf.control_dependencies([attempted]):
                    return tf.cond(rows.read_value() < cap, available, capped)
            return evaluate

        def optimize(initial, scale, previous=None):
            return tfp.optimizer.lbfgs_minimize(objective(initial, scale),
                initial_position=tf.zeros([dimension], D) if previous is None else None,
                previous_optimizer_results=previous,
                num_correction_pairs=config.num_correction_pairs,
                tolerance=tf.constant(config.gradient_tolerance, D),
                x_tolerance=tf.constant(config.x_tolerance, D),
                f_relative_tolerance=tf.constant(config.f_relative_tolerance, D),
                f_absolute_tolerance=tf.constant(config.f_absolute_tolerance, D),
                max_iterations=config.checkpoint_iterations if previous is None else config.total_iterations,
                parallel_iterations=config.parallel_iterations,
                max_line_search_iterations=config.max_line_search_iterations)

        def summarize(initial, scale, initial_value, initial_score, endpoint,
                      endpoint_value, endpoint_score, optimizer, saved,
                      checkpoint=None, status_override=None, replayed=True):
            endpoint_valid = tf.math.is_finite(endpoint_value) & tf.reduce_all(tf.math.is_finite(endpoint_score))
            count, attempted, capped = saved["rows"], saved["attempts"], saved["exhausted"]
            reported = tf.cast(optimizer.num_objective_evaluations, I)
            accounting = (reported == attempted) & (
                (~capped & (attempted == count) & (count <= cap)) |
                (capped & (attempted > count) & (count == cap)))
            threshold = initial_value if checkpoint is None else checkpoint["endpoint_objective"]
            accepted = accounting & ~capped & ~optimizer.failed & endpoint_valid & (endpoint_value >= threshold)
            status = tf.where(~accounting, 2, tf.where(capped, 3, tf.where(optimizer.failed, 4,
                tf.where(~endpoint_valid, 5, tf.where(endpoint_value < threshold, 6,
                tf.where(optimizer.converged, 7, 8))))))
            callback_position = rounded_affine_position(initial, scale, saved["z"])
            checkpoint_position = endpoint if checkpoint is None else checkpoint["endpoint_position"]
            checkpoint_value = endpoint_value if checkpoint is None else checkpoint["endpoint_objective"]
            checkpoint_score = endpoint_score if checkpoint is None else checkpoint["endpoint_score"]
            positions = tf.stack((initial, callback_position, checkpoint_position, endpoint))
            values = tf.stack((initial_value, saved["value"], checkpoint_value, endpoint_value))
            scores = tf.stack((initial_score, saved["score"], checkpoint_score, endpoint_score))
            eligible = tf.stack((tf.constant(True), saved["index"] >= 0,
                tf.constant(replayed or checkpoint is not None), tf.constant(replayed and checkpoint is not None)))
            eligible &= tf.math.is_finite(values) & tf.reduce_all(
                tf.math.is_finite(positions) & tf.math.is_finite(scores), axis=1)
            selected = tf.argmax(tf.where(eligible, values, tf.constant(float("-inf"), D)), output_type=tf.int32)
            if status_override is not None:
                status, accepted = tf.constant(status_override), tf.constant(False)
                reported = attempted
            initial_scaled, endpoint_scaled = initial_score * scale, endpoint_score * scale
            return {"status": status, "endpoint_accepted": accepted,
                "initial_position": initial, "initial_objective": initial_value, "initial_score": initial_score,
                "endpoint_position": endpoint, "endpoint_objective": endpoint_value, "endpoint_score": endpoint_score,
                "best_position": tf.gather(positions, selected), "best_objective": tf.gather(values, selected),
                "best_score": tf.gather(scores, selected), "best_present": tf.reduce_any(eligible), "best_source": selected,
                "best_callback_index": tf.where(selected == 1, saved["index"], tf.constant(-1, I)),
                "best_is_endpoint": selected == 3,
                "initial_score_l2": tf.linalg.norm(initial_scaled),
                "initial_score_max_abs": tf.reduce_max(tf.abs(initial_scaled)),
                "endpoint_score_l2": tf.linalg.norm(endpoint_scaled),
                "endpoint_score_max_abs": tf.reduce_max(tf.abs(endpoint_scaled)),
                "optimizer_converged": optimizer.converged, "optimizer_failed": optimizer.failed,
                "optimizer_iterations": tf.cast(optimizer.num_iterations, I), "reported_evaluations": reported,
                "callback_attempts": attempted, "optimizer_target_rows": count,
                "physical_target_rows": count + 1 + int(replayed) + int(checkpoint is not None),
                "cap_exhausted": capped}

        def checkpoint_impl(initial, scale):
            resets = restore({"attempts": 0, "rows": 0, "value": 0.,
                "z": tf.zeros([dimension], D), "score": tf.zeros([dimension], D), "index": -1, "exhausted": False})
            with tf.control_dependencies(resets):
                initial_value, initial_score = target(initial)
            seeds = (best_value.assign(initial_value), best_score.assign(initial_score))

            def run():
                try:
                    optimizer = optimize(initial, scale)
                except Exception as exc:  # noqa: BLE001 - trace-time construction failure only.
                    errors["checkpoint"] = type(exc).__name__
                    saved = state()
                    record = summarize(initial, scale, initial_value, initial_score, initial,
                        initial_value, initial_score, empty_optimizer(), saved, status_override=1, replayed=False)
                    return empty_optimizer(), saved, record
                with tf.control_dependencies(tf.nest.flatten(optimizer)):
                    saved = state()
                    endpoint = rounded_affine_position(initial, scale, optimizer.position)
                    value, score = target(endpoint)
                return optimizer, saved, summarize(initial, scale, initial_value, initial_score,
                    endpoint, value, score, optimizer, saved)

            def invalid():
                optimizer, saved = empty_optimizer(), state()
                return optimizer, saved, summarize(initial, scale, initial_value, initial_score,
                    initial, initial_value, initial_score, optimizer, saved, status_override=0, replayed=False)

            with tf.control_dependencies(seeds):
                valid = tf.math.is_finite(initial_value) & tf.reduce_all(tf.math.is_finite(initial_score))
                result = tf.cond(valid, run, invalid)
            return tf.nest.map_structure(tf.stop_gradient, result)

        operands = [tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D)]
        self.checkpoint = tf.function(checkpoint_impl, input_signature=operands,
            jit_compile=config.jit_compile, autograph=False)
        state_spec = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype),
            self.checkpoint.get_concrete_function().structured_outputs)

        def continue_impl(initial, scale, previous):
            previous_optimizer, previous_state, checkpoint = previous
            with tf.control_dependencies(restore(previous_state)):
                try:
                    optimizer = optimize(initial, scale, previous_optimizer)
                except Exception as exc:  # noqa: BLE001 - trace-time construction failure only.
                    errors["continuation"] = type(exc).__name__
                    saved = state()
                    return summarize(initial, scale, checkpoint["initial_objective"], checkpoint["initial_score"],
                        checkpoint["endpoint_position"], checkpoint["endpoint_objective"], checkpoint["endpoint_score"],
                        previous_optimizer, saved, checkpoint=checkpoint, status_override=1, replayed=False)
            with tf.control_dependencies(tf.nest.flatten(optimizer)):
                saved = state()
                endpoint = rounded_affine_position(initial, scale, optimizer.position)
                value, score = target(endpoint)
            result = summarize(initial, scale, checkpoint["initial_objective"], checkpoint["initial_score"],
                endpoint, value, score, optimizer, saved, checkpoint=checkpoint)
            return tf.nest.map_structure(tf.stop_gradient, result)

        self.continuation = tf.function(continue_impl, input_signature=[*operands, state_spec],
            jit_compile=config.jit_compile, autograph=False)


def checkpoint_record(raw):
    """Translate completed numerical fields for the external validator."""
    from bayesfilter.inference.joint_center import JointCenterCheckpoint

    return JointCenterCheckpoint(status=STATUSES[int(raw["status"])],
        endpoint_accepted=bool(raw["endpoint_accepted"]), position=raw["endpoint_position"],
        score=raw["endpoint_score"], objective=float(raw["endpoint_objective"]),
        score_l2=float(raw["endpoint_score_l2"]), score_max_abs=float(raw["endpoint_score_max_abs"]),
        optimizer_converged=bool(raw["optimizer_converged"]), optimizer_failed=bool(raw["optimizer_failed"]),
        optimizer_iterations=int(raw["optimizer_iterations"]),
        reported_objective_evaluations=int(raw["reported_evaluations"]), callback_attempts=int(raw["callback_attempts"]),
        optimizer_target_rows=int(raw["optimizer_target_rows"]), physical_target_rows=int(raw["physical_target_rows"]))


def staged_result(raw, checkpoint, *, validated, validator_calls, continuation_started,
                  jit_compile, status=None, exception_type=None):
    """Host formatting only; every numerical decision was computed in a stage."""
    from bayesfilter.inference.joint_center import JointCenterStagedResult

    present = bool(raw["best_present"])
    callback_index = int(raw["best_callback_index"])
    return JointCenterStagedResult(status=STATUSES[int(raw["status"])] if status is None else status,
        endpoint_accepted=bool(raw["endpoint_accepted"]) if status is None else False,
        initial_position=raw["initial_position"], initial_score=raw["initial_score"],
        initial_objective=float(raw["initial_objective"]), initial_score_l2=float(raw["initial_score_l2"]),
        initial_score_max_abs=float(raw["initial_score_max_abs"]), checkpoint=checkpoint,
        checkpoint_validated=validated, checkpoint_validator_calls=validator_calls, continuation_started=continuation_started,
        endpoint_position=raw["endpoint_position"], endpoint_score=raw["endpoint_score"],
        endpoint_objective=float(raw["endpoint_objective"]), endpoint_score_l2=float(raw["endpoint_score_l2"]),
        endpoint_score_max_abs=float(raw["endpoint_score_max_abs"]), optimizer_converged=bool(raw["optimizer_converged"]),
        optimizer_failed=bool(raw["optimizer_failed"]), optimizer_iterations=int(raw["optimizer_iterations"]),
        reported_objective_evaluations=int(raw["reported_evaluations"]), callback_attempts=int(raw["callback_attempts"]),
        optimizer_target_rows=int(raw["optimizer_target_rows"]), physical_target_rows=int(raw["physical_target_rows"]),
        cap_exhausted=bool(raw["cap_exhausted"]), wall_time_exhausted=False, jit_compile=jit_compile,
        exception_type=exception_type, best_evaluated_position=raw["best_position"] if present else None,
        best_evaluated_score=raw["best_score"] if present else None,
        best_evaluated_objective=float(raw["best_objective"]) if present else None,
        best_evaluated_source=SOURCES[int(raw["best_source"])] if present else None,
        best_evaluated_callback_index=callback_index if callback_index >= 0 else None,
        best_evaluated_is_endpoint=bool(raw["best_is_endpoint"]))


def run_staged_program(program, initial, scale, validator):
    """One external validator boundary; serialize the complete invocation."""
    with program.invocation_lock:
        first = program.checkpoint(initial, scale)
        raw = first[2]
        checkpoint = checkpoint_record(raw)
        validated, calls, continued, status, error = False, 0, False, None, None
        if checkpoint.endpoint_accepted:
            calls = 1
            try:
                validated = bool(validator(checkpoint))
            except Exception as exc:  # noqa: BLE001 - existing host-validator semantics.
                status, error = "checkpoint_validator_exception", type(exc).__name__
            if validated:
                continued = True
                raw = program.continuation(initial, scale, first)
            elif status is None:
                status = "checkpoint_rejected"
        if int(raw["status"]) == 1:
            error = program.construction_errors.get("continuation" if continued else "checkpoint")
        return staged_result(raw, checkpoint, validated=validated, validator_calls=calls,
            continuation_started=continued, jit_compile=bool(program.checkpoint.function_spec.jit_compile),
            status=status, exception_type=error)
