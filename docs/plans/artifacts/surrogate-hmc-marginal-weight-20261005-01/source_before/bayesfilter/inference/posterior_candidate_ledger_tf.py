"""Internal native candidate ledger for posterior-initializer enclosure.

Selection calls the shared exact-incumbent authority. This dependency does not
yet replace the public initializer's movement or curvature host controller.
"""

import tensorflow as tf

from bayesfilter.inference._exact_incumbent import _incumbent_selection
from bayesfilter.inference.program_cache_scope import scoped_program_cache

D = tf.float64
I = tf.int32


@scoped_program_cache(maxsize=16)
def candidate_ledger_program(capacity, dimension, *, jit_compile=True):
    """Record exact earliest incumbents with a fixed capacity and active count.

    Invalid counts produce an explicit invalid-input status and no selection.
    Scale is an already validated positive finite coordinate scale, as in the
    enclosing initializer. Numerical records never escape to a host decision.
    """
    if capacity < 0 or dimension < 1:
        raise ValueError("capacity must be nonnegative and dimension positive")
    select = _incumbent_selection.python_function

    @tf.function(input_signature=[
        tf.TensorSpec([capacity, dimension], D), tf.TensorSpec([capacity], D),
        tf.TensorSpec([capacity, dimension], D), tf.TensorSpec([capacity], tf.bool),
        tf.TensorSpec([dimension], D), tf.TensorSpec([], I),
    ], jit_compile=jit_compile, autograph=False)
    def ledger(positions, values, scores, eligibility, scale, active_count):
        valid_count = (active_count >= 0) & (active_count <= capacity)
        initial = (tf.constant(0, I), tf.constant(-1, I), tf.zeros([dimension], D),
            tf.constant(0., D), tf.zeros([dimension], D), tf.zeros([capacity], tf.bool),
            tf.zeros([capacity], tf.bool), tf.fill([capacity], tf.constant(-1, I)),
            tf.zeros([capacity], D))

        def condition(index, *unused):
            return valid_count & (index < active_count)

        def body(index, best_index, best_position, best_value, best_score,
                 eligible_rows, promotions, prefix_indices, score_norms):
            point, value, score = (tf.gather(positions, index), tf.gather(values, index),
                                   tf.gather(scores, index))
            mask, selected = select(tf.concat((best_position, point), 0),
                tf.concat((best_score, score), 0), tf.stack((best_value, value)),
                tf.stack((best_index >= 0, tf.gather(eligibility, index))),
                tf.constant([dimension, 2 * dimension], I))
            promoted = selected == 1
            chosen_index = tf.where(promoted, index, best_index)
            address = tf.reshape(index, [1, 1])
            eligible_rows = tf.tensor_scatter_nd_update(eligible_rows, address, mask[1:2])
            promotions = tf.tensor_scatter_nd_update(promotions, address, promoted[None])
            prefix_indices = tf.tensor_scatter_nd_update(prefix_indices, address, chosen_index[None])
            score_norms = tf.tensor_scatter_nd_update(score_norms, address, tf.linalg.norm(score * scale)[None])
            return (index + 1, chosen_index, tf.where(promoted, point, best_position),
                tf.where(promoted, value, best_value), tf.where(promoted, score, best_score),
                eligible_rows, promotions, prefix_indices, score_norms)

        # A zero-capacity fixture has no valid gather operation to trace.
        completed = tf.while_loop(condition, body, initial, parallel_iterations=1) if capacity else initial
        result = {"input_valid": valid_count, "recorded_count": completed[0],
            "selected_index": completed[1], "selected_present": completed[1] >= 0,
            "position": completed[2], "value": completed[3], "score": completed[4],
            "candidate_eligible": completed[5], "promoted": completed[6],
            "selected_prefix_indices": completed[7], "scaled_score_l2": completed[8]}
        return tf.nest.map_structure(tf.stop_gradient, result)

    return ledger
