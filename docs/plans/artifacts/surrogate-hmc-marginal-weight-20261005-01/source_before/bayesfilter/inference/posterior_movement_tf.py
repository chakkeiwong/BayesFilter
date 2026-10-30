"""Internal prepared-cloud movement recurrence for posterior initialization.

The public initializer is not wired here yet. Geometry and exact selection
remain configurations of their shared numerical authorities.
"""

import tensorflow as tf

from bayesfilter.inference.posterior_candidate_ledger_tf import candidate_ledger_program
from bayesfilter.inference.quadratic_geometry_control_tf import _callback_program
from bayesfilter.inference.quadratic_geometry_full_tf import (
    geometry_extents,
    make_geometry_program,
)

D = tf.float64
I = tf.int32
STATUSES = ("movement_complete", "insufficient_successful_movement_fits",
            "movement_not_centered_within_attempt_budget", "eligibility_contract_mismatch",
            "exact_evaluation_budget_exhausted")


def make_posterior_movement_program(evaluator, dimension, config, movement_config, *, jit_compile=True):
    """Continue from a finite exact incumbent, retaining every attempt record.

    The enclosing owner supplies/reset its eligibility tracker and serializes
    invocation. Prepared directions, offsets and permutation seeds are operands.
    No random draw or source selection happens on the host inside the recurrence.
    """
    attempts = config.max_movement_attempts
    _, required, samples, directions = geometry_extents(dimension, movement_config)
    geometry = make_geometry_program(evaluator.scalar, dimension, movement_config,
        batched_callback=evaluator.batched if evaluator.batched_fn is not None else None,
        jit_compile=jit_compile)
    target = _callback_program(evaluator.scalar, dimension, jit_compile)
    choose = candidate_ledger_program(attempts + 1, dimension, jit_compile=jit_compile)
    template = geometry.get_concrete_function().structured_outputs
    maximum = attempts + 1

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([attempts, directions, dimension], D),
        tf.TensorSpec([attempts, samples, dimension], D), tf.TensorSpec([attempts, 2], I)],
        jit_compile=jit_compile, autograph=False)
    def run(initial, value, score, scale, raw_directions, offsets, permutation_seeds):
        state = {"attempt_count": tf.constant(0, I), "successful_fits": tf.constant(0, I),
            "status": tf.constant(0, I), "material_move": tf.constant(False),
            "center": initial, "value": value, "score": score, "ledger_count": tf.constant(1, I),
            "ledger_positions": tf.concat((initial[None], tf.zeros([attempts, dimension], D)), 0),
            "ledger_values": tf.concat((value[None], tf.zeros([attempts], D)), 0),
            "ledger_scores": tf.concat((score[None], tf.zeros([attempts, dimension], D)), 0),
            "ledger_scaled_score_l2": tf.concat((tf.linalg.norm(score * scale)[None], tf.zeros([attempts], D)), 0),
            "ledger_attempts": tf.fill([maximum], tf.constant(-1, I)),
            "ledger_sources": tf.fill([maximum], tf.constant(-1, I)),
            "ledger_promoted": tf.concat((tf.constant([True]), tf.zeros([attempts], tf.bool)), 0),
            "geometry_history": tf.nest.map_structure(lambda x: tf.zeros([attempts, *x.shape], x.dtype), template),
            "fit_centers": tf.zeros([attempts, dimension], D),
            "geometry_accepted": tf.zeros([attempts], tf.bool),
            "geometry_best_present": tf.zeros([attempts], tf.bool),
            "geometry_best_sources": tf.fill([attempts], tf.constant(-1, I)),
            "candidate_promoted": tf.zeros([attempts], tf.bool),
            "center_moved": tf.zeros([attempts], tf.bool),
            "material_moves": tf.zeros([attempts], tf.bool),
            "scaled_moves": tf.zeros([attempts], D), "improvements": tf.zeros([attempts], D),
            "successful_fit_counts": tf.zeros([attempts], I)}

        def condition(current):
            return ((current["attempt_count"] < attempts) & (current["status"] == 0) &
                    ((current["successful_fits"] < config.min_movement_fits) | current["material_move"]))

        def step(current):
            index = current["attempt_count"]
            raw = geometry(current["center"], scale, tf.gather(raw_directions, index),
                           tf.gather(offsets, index), tf.gather(permutation_seeds, index))
            old_best = raw["incumbent"]
            if samples >= required:
                fitted = raw["fit_result"]
                best = {"present": tf.where(raw["stage"] == 3, fitted["has_incumbent"], old_best["present"]),
                    "position": tf.where(raw["stage"] == 3, fitted["best_position"], old_best["position"]),
                    "source": tf.where(raw["stage"] == 3, fitted["best_source"], old_best["source"])}
                accepted = (raw["stage"] == 3) & (fitted["status"] == 1)
            else:
                best, accepted = old_best, tf.constant(False)

            def replay():
                replay_value, replay_score = target(best["position"])
                valid = tf.math.is_finite(replay_value) & tf.reduce_all(tf.math.is_finite(replay_score))
                return replay_value, replay_score, valid

            with tf.control_dependencies(tf.nest.flatten(raw)):
                replay_value, replay_score, replay_valid = tf.cond(best["present"], replay,
                    lambda: (tf.constant(0., D), tf.zeros([dimension], D), tf.constant(False)))

            def append():
                address = tf.reshape(current["ledger_count"], [1, 1])
                positions = tf.tensor_scatter_nd_update(current["ledger_positions"], address, best["position"][None])
                values = tf.tensor_scatter_nd_update(current["ledger_values"], address, replay_value[None])
                scores = tf.tensor_scatter_nd_update(current["ledger_scores"], address, replay_score[None])
                count = current["ledger_count"] + 1
                selection = choose(positions, values, scores, tf.ones([maximum], tf.bool), scale, count)
                promoted = tf.gather(selection["promoted"], count - 1)
                return {"ledger_positions": positions, "ledger_values": values, "ledger_scores": scores,
                    "ledger_scaled_score_l2": selection["scaled_score_l2"],
                    "ledger_count": count, "center": selection["position"], "value": selection["value"],
                    "score": selection["score"], "promoted": promoted,
                    "ledger_attempts": tf.tensor_scatter_nd_update(current["ledger_attempts"], address, index[None]),
                    "ledger_sources": tf.tensor_scatter_nd_update(current["ledger_sources"], address, best["source"][None]),
                    "ledger_promoted": tf.tensor_scatter_nd_update(current["ledger_promoted"], address, promoted[None])}

            chosen = tf.cond(replay_valid, append, lambda: {
                "ledger_positions": current["ledger_positions"], "ledger_values": current["ledger_values"],
                "ledger_scores": current["ledger_scores"], "ledger_count": current["ledger_count"],
                "ledger_scaled_score_l2": current["ledger_scaled_score_l2"],
                "center": current["center"], "value": current["value"], "score": current["score"],
                "ledger_attempts": current["ledger_attempts"], "ledger_sources": current["ledger_sources"],
                "ledger_promoted": current["ledger_promoted"],
                "promoted": tf.constant(False)})
            improved = chosen["value"] > current["value"]
            moved = improved & ~tf.reduce_all(chosen["center"] == current["center"])
            displacement = tf.linalg.norm((chosen["center"] - current["center"]) / scale)
            improvement = chosen["value"] - current["value"]
            material = moved & (improvement > config.objective_improvement_tolerance) & (displacement > config.scaled_center_tolerance)
            successes = current["successful_fits"] + tf.cast(accepted, I)
            with tf.control_dependencies((replay_value, replay_score)):
                status = tf.where(evaluator.mismatch_rows.read_value() > 0, 3,
                                  tf.where(evaluator.budget_exhausted.read_value(), 4, 0))
            address = tf.reshape(index, [1, 1])

            def record(history, item):
                if item.shape.num_elements() == 0:
                    return history
                return tf.tensor_scatter_nd_update(history, address, item[None])

            promoted = chosen.pop("promoted")
            result = {**current, **chosen,
                "attempt_count": index + 1, "successful_fits": successes,
                "status": status, "material_move": material,
                "geometry_history": tf.nest.map_structure(record, current["geometry_history"], raw)}
            result.update(fit_centers=record(current["fit_centers"], current["center"]),
                geometry_accepted=record(current["geometry_accepted"], accepted),
                geometry_best_present=record(current["geometry_best_present"], best["present"]),
                geometry_best_sources=record(current["geometry_best_sources"], best["source"]),
                candidate_promoted=record(current["candidate_promoted"], promoted),
                center_moved=record(current["center_moved"], moved),
                material_moves=record(current["material_moves"], material),
                scaled_moves=record(current["scaled_moves"], displacement),
                improvements=record(current["improvements"], improvement),
                successful_fit_counts=record(current["successful_fit_counts"], successes))
            return (result,)

        result, = tf.while_loop(condition, step, (state,), parallel_iterations=1)
        result["status"] = tf.where(result["status"] != 0, result["status"],
            tf.where(result["successful_fits"] < config.min_movement_fits, 1,
                     tf.where(result["material_move"], 2, 0)))
        return tf.nest.map_structure(tf.stop_gradient, result)

    return run
