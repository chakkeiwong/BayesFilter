"""Complete paired quadratic refinement as a bounded TensorFlow/XLA program.

The paired-local public initializer calls this controller after validating its
inputs and row budget. The callback supplies the unchanged analytical score;
no optimizer, curvature model, acceptance threshold or random stream is added.
"""

from dataclasses import replace
from threading import RLock

import tensorflow as tf

from bayesfilter.inference.paired_score_pilot_tf import (
    make_paired_score_precision_program,
)
from bayesfilter.inference.quadratic_batch_evaluation_tf import (
    make_quadratic_batch_evaluator,
)
from bayesfilter.inference.quadratic_probe_evaluation_tf import (
    make_paired_probe_evaluator,
)

D = tf.float64
STATUS = (
    "running", "initial_target_invalid", "localizer_refinement_replay_mismatch",
    "replay_mismatch", "curvature_target_invalid", "nonfinite_scaled_score",
    "quadratic_model_rejected", "pilot_factorization_failed", "nonfinite_centeredness",
    "trust_solve_failed", "refinement_round_limit", "local_center_candidate",
)
ROLES = ("initial", "replay", "paired_fit_large", "paired_fit_small", "model_check", "proposal")
_CONTROLLER_LOCK = RLock()
_LAST_CONTROLLER = None


def paired_quadratic_controller(callback, dimension, config, *, jit_compile=True):
    """Reuse the most recent target/configuration; keep seed a runtime operand.

    One retained program bounds Python target ownership. Replacing this entry
    does not promise native executable eviction. Callback identity avoids
    conflating equal or unhashable callable objects with distinct targets.
    """
    global _LAST_CONTROLLER
    static = replace(config, seed=0, paired_steps=tuple(config.paired_steps))
    with _CONTROLLER_LOCK:
        previous = _LAST_CONTROLLER
        if (previous is not None and previous[0] is callback
                and previous[1:4] == (dimension, static, jit_compile)):
            return previous[4]
        _LAST_CONTROLLER = None
        del previous
        program = make_paired_quadratic_controller(callback, dimension, static, jit_compile=jit_compile)
        _LAST_CONTROLLER = (callback, dimension, static, jit_compile, program)
        return program


def clear_paired_quadratic_controller_cache():
    """Release the single Python cache entry, without claiming native cleanup."""
    global _LAST_CONTROLLER
    with _CONTROLLER_LOCK:
        _LAST_CONTROLLER = None


def make_paired_quadratic_controller(callback, dimension, config, *, jit_compile=True):
    """Bind static capacities; seed, anchor, scale and initial evidence are inputs.

    The wrapper must validate finite center, positive finite scale, dimensions
    and the whole planned physical-row budget before invoking this program.
    ``jit_compile=False`` is an explicit graph-reference exception for the
    complete controller, including trust solving and paired fitting.
    """
    from bayesfilter.inference.batched_quadratic_center import (
        _norm,
        solve_spd_quadratic_trust_region_tf,
    )

    cfg = config
    if cfg.pilot_method != "paired_local":
        raise ValueError("paired controller requires paired_local")
    planned = cfg.planned_rows(dimension)
    if dimension < 1 or planned > cfg.max_physical_rows:
        raise ValueError("invalid dimension or whole refinement physical-row budget")
    batch, rounds = cfg.batch_size, cfg.max_fit_rounds
    capacity = planned // batch
    probe_batches = (2 * dimension + batch - 1) // batch
    single = make_quadratic_batch_evaluator(callback, dimension, batch, 1, jit_compile=jit_compile)
    probes = make_paired_probe_evaluator(callback, dimension, batch, steps=cfg.paired_steps, jit_compile=jit_compile)
    fitted = make_paired_score_precision_program(dimension, steps=cfg.paired_steps,
        precision_condition_cap=cfg.precision_condition_cap,
        model_relative_rmse_cap=cfg.model_relative_rmse_cap, jit_compile=jit_compile)
    model_template = fitted.get_concrete_function().structured_outputs
    trust = tf.function(solve_spd_quadratic_trust_region_tf, autograph=False, jit_compile=jit_compile,
        input_signature=[tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], D)])

    def put(history, index, value):
        return tf.tensor_scatter_nd_update(history, tf.reshape(index, [1, 1]), value[None])

    def retain(state, result, positions, values, scores, valid, roles):
        """Append executed callbacks only; physical charges can include no-call rows."""
        start, first_index = state["callback_batches"], state["physical_rows"]

        def append(index, history):
            slot = start + index
            return index + 1, {
                "positions": put(history["positions"], slot, tf.gather(positions, index)),
                "values": put(history["values"], slot, tf.gather(values, index)),
                "scores": put(history["scores"], slot, tf.gather(scores, index)),
                "valid": put(history["valid"], slot, tf.gather(valid, index)),
                "roles": put(history["roles"], slot, tf.gather(roles, index)),
                "first_index": put(history["first_index"], slot, first_index + tf.cast(index * batch, tf.int64)),
            }

        _, history = tf.while_loop(lambda index, _: index < result["callback_batches"], append,
            (tf.constant(0), state["history"]), parallel_iterations=1)
        return {**state, "history": history,
            "center": result["center"], "center_value": result["center_value"], "center_score": result["center_score"],
            "selected_index": result["selected_index"],
            "physical_rows": state["physical_rows"] + result["physical_rows"],
            "padded_rows": state["padded_rows"] + result["padded_rows"],
            "invalid_rows": state["invalid_rows"] + result["invalid_rows"],
            "callback_batches": state["callback_batches"] + result["callback_batches"],
        }

    def evaluate(state, point, role, record):
        result = single(point[None], 1, record, state["center"], state["center_value"], state["center_score"],
            state["physical_rows"], state["selected_index"])
        state = retain(state, result, result["positions"], result["values"], result["scores"], result["valid"],
            tf.constant([role], tf.int32))
        return state, result

    def close(actual, expected):
        return tf.reduce_all(tf.abs(actual - expected) <= cfg.replay_atol + cfg.replay_rtol * tf.abs(expected))

    def replay(state):
        state, result = evaluate(state, state["center"], 1, False)
        ok = result["ok"] & close(result["values"][0], state["center_value"])
        ok &= close(result["scores"][0], state["center_score"])
        return {**state, "status": tf.where(ok, state["status"], 3)}

    signature = [tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.bool), tf.TensorSpec([], D), tf.TensorSpec([dimension], D)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=jit_compile)
    def execute(center, scale, seed, has_initial_evidence, initial_value, initial_score):
        state = {
            "status": tf.constant(0), "iterations": tf.constant(0),
            "center": center, "center_value": tf.constant(-float("inf"), D), "center_score": tf.zeros([dimension], D),
            "physical_rows": tf.constant(0, tf.int64), "padded_rows": tf.constant(0, tf.int64),
            "invalid_rows": tf.constant(0, tf.int64), "selected_index": tf.constant(-1, tf.int64),
            "callback_batches": tf.constant(0), "fit_calls": tf.constant(0), "trust_calls": tf.constant(0),
            "radius": tf.constant(cfg.initial_trust_radius, D),
            "factor": tf.zeros([dimension, dimension], D), "precision": tf.zeros([dimension, dimension], D),
            "history": {"positions": tf.zeros([capacity, batch, dimension], D),
                "values": tf.zeros([capacity, batch], D), "scores": tf.zeros([capacity, batch, dimension], D),
                "valid": tf.zeros([capacity, batch], tf.bool), "roles": tf.zeros([capacity], tf.int32),
                "first_index": tf.zeros([capacity], tf.int64)},
            "rounds": {"anchor": tf.zeros([rounds, dimension], D), "anchor_value": tf.zeros([rounds], D),
                "radius": tf.zeros([rounds], D), "cloud_changed_incumbent": tf.zeros([rounds], tf.bool),
                "model_present": tf.zeros([rounds], tf.bool), "centeredness_present": tf.zeros([rounds], tf.bool),
                "step_present": tf.zeros([rounds], tf.bool), "centeredness": tf.zeros([rounds], D),
                "models": tf.nest.map_structure(lambda value: tf.zeros([rounds, *value.shape], value.dtype), model_template),
                "steps": {"valid": tf.zeros([rounds], tf.bool), "step": tf.zeros([rounds, dimension], D),
                    "multiplier": tf.zeros([rounds], D), "boundary_active": tf.zeros([rounds], tf.bool),
                    "predicted_improvement": tf.zeros([rounds], D)},
                "actual_improvement": tf.zeros([rounds], D), "ratio": tf.zeros([rounds], D),
                "trust_step_accepted": tf.zeros([rounds], tf.bool)},
        }
        state, first = evaluate(state, center, 0, True)
        evidence_ok = close(state["center_value"], initial_value) & close(state["center_score"], initial_score)
        use_evidence = first["ok"] & has_initial_evidence & evidence_ok
        state = {**state, "status": tf.where(~first["ok"], 1, tf.where(has_initial_evidence & ~evidence_ok, 2, 0)),
            "center_value": tf.where(use_evidence, initial_value, state["center_value"]),
            "center_score": tf.where(use_evidence, initial_score, state["center_score"])}

        def fit_round(state):
            index = state["iterations"]
            state = replay(state)
            anchor, anchor_value, anchor_score = state["center"], state["center_value"], state["center_score"]

            def prepare(state):
                result = probes(seed, index, scale, anchor, anchor_value, anchor_score,
                    state["physical_rows"], state["selected_index"])
                state = retain(state, result, tf.reshape(result["positions"], [-1, batch, dimension]),
                    tf.reshape(result["values"], [-1, batch]), tf.reshape(result["scores"], [-1, batch, dimension]),
                    tf.reshape(result["valid"], [-1, batch]), tf.repeat(tf.constant([2, 3, 4]), probe_batches))
                center_score = anchor_score * scale
                finite = tf.reduce_all(tf.math.is_finite(center_score))
                finite &= tf.reduce_all(tf.math.is_finite(result["scaled_scores"]))
                finite &= tf.reduce_all(tf.math.is_finite(result["check_offsets"]))
                state = {**state, "status": tf.where(~result["ok"], 4, tf.where(finite, 0, 5))}

                def fit_model(state):
                    model = fitted(center_score, result["scaled_scores"][0], result["scaled_scores"][1],
                        result["check_offsets"], result["scaled_scores"][2])
                    precision = model["raw_precision"]
                    good = model["raw_spd"] & (model["design_rank"] == dimension)
                    good &= tf.reduce_all(tf.math.is_finite(precision))
                    good &= model["precision_condition"] <= cfg.precision_condition_cap
                    good &= model["selection_relative_rmse"] <= cfg.model_relative_rmse_cap
                    good &= model["accepted"]
                    changed = tf.reduce_any(state["center"] != anchor)
                    report = {**state["rounds"],
                        "anchor": put(state["rounds"]["anchor"], index, anchor),
                        "anchor_value": put(state["rounds"]["anchor_value"], index, anchor_value),
                        "radius": put(state["rounds"]["radius"], index, state["radius"]),
                        "model_present": put(state["rounds"]["model_present"], index, tf.constant(True)),
                        "cloud_changed_incumbent": put(state["rounds"]["cloud_changed_incumbent"], index, changed),
                        "models": tf.nest.map_structure(lambda history, value: put(history, index, value), state["rounds"]["models"], model)}
                    state = {**state, "rounds": report, "fit_calls": state["fit_calls"] + 1,
                        "status": tf.where(good, 0, 6)}

                    def factorize(state):
                        cholesky = tf.linalg.cholesky(precision)
                        solved = tf.linalg.triangular_solve(cholesky, tf.linalg.diag(scale))
                        covariance = tf.matmul(solved, solved, transpose_a=True)
                        factor = tf.linalg.cholesky(covariance)
                        reconstructed = factor @ tf.transpose(factor)
                        covariance_scale = tf.reduce_max(tf.abs(covariance))
                        reconstruction = tf.linalg.norm((reconstructed - covariance) / covariance_scale)
                        reconstruction /= tf.linalg.norm(covariance / covariance_scale)
                        valid = tf.reduce_all(tf.math.is_finite(factor)) & tf.reduce_all(tf.linalg.diag_part(factor) > 0)
                        valid &= tf.math.is_finite(reconstruction) & (reconstruction <= 1e-10)
                        state = {**state, "status": tf.where(valid, 0, 7)}

                        def use_factor(state):
                            centeredness = _norm(tf.linalg.matvec(factor, anchor_score, transpose_a=True))
                            report = {**state["rounds"],
                                "centeredness": put(state["rounds"]["centeredness"], index, centeredness),
                                "centeredness_present": put(state["rounds"]["centeredness_present"], index, tf.constant(True))}
                            state = {**state, "rounds": report, "status": tf.where(tf.math.is_finite(centeredness), 0, 8)}

                            def accept(state):
                                state = replay(state)
                                return {**state, "status": tf.where(state["status"] == 0, 11, state["status"]),
                                    "factor": factor, "precision": precision}

                            def move(state):
                                step = trust(precision, center_score, state["radius"])
                                state = {**state, "trust_calls": state["trust_calls"] + 1,
                                    "status": tf.where(step["valid"], 0, 9)}

                                def propose(state):
                                    state, proposed = evaluate(state, anchor + scale * step["step"], 5, True)
                                    predicted = step["predicted_improvement"]
                                    actual = tf.where(proposed["ok"], proposed["values"][0, 0] - anchor_value,
                                        tf.constant(-float("inf"), D))
                                    ratio = tf.where(predicted > 0, actual / predicted, tf.constant(-float("inf"), D))
                                    report = {**state["rounds"],
                                        "steps": tf.nest.map_structure(lambda history, value: put(history, index, value), state["rounds"]["steps"], step),
                                        "step_present": put(state["rounds"]["step_present"], index, tf.constant(True)),
                                        "actual_improvement": put(state["rounds"]["actual_improvement"], index, actual),
                                        "ratio": put(state["rounds"]["ratio"], index, ratio),
                                        "trust_step_accepted": put(state["rounds"]["trust_step_accepted"], index,
                                            tf.math.is_finite(ratio) & (actual > 0) & (ratio > .1))}
                                    radius = tf.where(~tf.math.is_finite(ratio) | (ratio < .25), state["radius"] * .25,
                                        tf.where((ratio > .75) & step["boundary_active"],
                                            tf.minimum(2 * state["radius"], cfg.maximum_trust_radius), state["radius"]))
                                    return {**state, "rounds": report, "radius": radius}

                                return tf.cond(state["status"] == 0, lambda: propose(state), lambda: state)

                            return tf.cond(state["status"] == 0,
                                lambda: tf.cond((centeredness <= cfg.centeredness_cap) & ~changed,
                                    lambda: accept(state), lambda: move(state)), lambda: state)

                        return tf.cond(state["status"] == 0, lambda: use_factor(state), lambda: state)

                    return tf.cond(state["status"] == 0, lambda: factorize(state), lambda: state)

                return tf.cond(state["status"] == 0, lambda: fit_model(state), lambda: state)

            state = tf.cond(state["status"] == 0, lambda: prepare(state), lambda: state)
            return ({**state, "iterations": index + 1},)

        state, = tf.while_loop(lambda state: (state["status"] == 0) & (state["iterations"] < rounds),
            fit_round, (state,), maximum_iterations=rounds, parallel_iterations=1)
        return {**state, "status": tf.where(state["status"] == 0, 10, state["status"])}

    return execute
