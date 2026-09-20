"""Native ordered proposals, exact-incumbent promotion and radius update.

The second-fit callback is a native fixed-signature program; it must return
precision and usability plus its complete tensor record for later reporting.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference.sequential_proposal_tf import proposal_program

D = tf.float64


def _empty_proposal(dimension, old_norm, active):
    return {'position': tf.zeros([dimension], D), 'value': tf.constant(float('-inf'), D),
        'score': tf.zeros([dimension], D), 'step': tf.zeros([dimension], D),
        'actual': tf.constant(float('-inf'), D), 'predicted': tf.constant(float('-inf'), D),
        'old_norm': old_norm, 'new_norm': old_norm, 'finite': tf.constant(False),
        'boundary': tf.constant(False), 'rho': tf.constant(float('-inf'), D),
        'score_passed': tf.constant(not active), 'legacy_passed': tf.constant(False),
        'required_norm_max': tf.constant(0., D), 'resolution_floor': tf.constant(0., D),
        'accepted': tf.constant(False)}


@lru_cache(maxsize=32)
def attempts_program(scalar, second_fit, dimension, config, *, jit_compile=True):
    # The fixed second callback has explicit input/output signatures. Its empty
    # record is schema construction only and never evaluates a target or fit.
    # Bind resource ownership before creating the consuming loop graph.
    second_concrete = second_fit.get_concrete_function()
    second_signature = second_fit.input_signature
    second_shape = second_concrete.structured_outputs
    propose = proposal_program(scalar, dimension, config.proposal_score_acceptance_policy,
        config.require_proposal_score_reduction, jit_compile=jit_compile).python_function
    count = 2 if config.refinement_geometry_policy == 'factor_correlation' and config.structured_max_factors == 2 else 1

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.bool), tf.TensorSpec([dimension, dimension], D),
        second_signature], jit_compile=jit_compile, autograph=False)
    def attempt(center, center_value, center_score, scale, radius, stalled, first_usable, first_precision, second_inputs):
        first = _empty_proposal(dimension, tf.linalg.norm(scale * center_score), config.require_proposal_score_reduction)
        empty_second = tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), second_shape)
        histories = tf.nest.map_structure(lambda value: tf.TensorArray(value.dtype, size=count,
            element_shape=value.shape, clear_after_read=False), first)
        used = tf.TensorArray(tf.bool, size=count, clear_after_read=False)
        evaluated = tf.TensorArray(tf.bool, size=count, clear_after_read=False)
        # Fixed-size histories remain initialized even if the first proposal
        # accepts. Unused rows are flagged and excluded from host records.
        def initialize(index, histories, used, evaluated):
            return (index + 1, tf.nest.map_structure(lambda rows, value: rows.write(index, value), histories, first),
                used.write(index, False), evaluated.write(index, False))
        _, histories, used, evaluated = tf.while_loop(lambda index, *_: index < count, initialize,
            (tf.constant(0), histories, used, evaluated), maximum_iterations=count, parallel_iterations=1)

        def step(index, accepted, best_exists, best_value, best_position, best_score,
                 last_evaluated, last, second, histories, used, evaluated):
            def fit_second():
                return second_fit(*second_inputs)

            if count == 2:
                second = tf.cond(index == 1, fit_second, lambda: second)
            usable = tf.where(index == 0, first_usable, second['usable'])
            precision = tf.where(index == 0, first_precision, second['precision'])

            def execute():
                return propose(center, center_value, center_score, scale, precision, radius,
                    tf.constant(config.score_reduction_factor, D), tf.constant(config.acceptance_ratio, D))

            current = tf.cond(usable, execute, lambda: last)
            improved = usable & current['finite'] & (current['value'] > best_value)
            return (index + 1, usable & current['accepted'], best_exists | improved,
                tf.where(improved, current['value'], best_value),
                tf.where(improved, current['position'], best_position),
                tf.where(improved, current['score'], best_score), last_evaluated | usable, current, second,
                tf.nest.map_structure(lambda rows, value: rows.write(index, value), histories, current),
                used.write(index, True), evaluated.write(index, usable))

        (attempted, accepted, best_exists, best_value, best_position, best_score,
         last_evaluated, last, second, histories, used, evaluated) = tf.while_loop(
            lambda index, accepted, *_: (index < count) & ~accepted, step,
            (tf.constant(0), tf.constant(False), tf.constant(False), center_value, center, center_score,
             tf.constant(False), first, empty_second, histories, used, evaluated),
            maximum_iterations=count, parallel_iterations=1)
        promoted_without_acceptance = best_exists & (~accepted | ~last_evaluated | (best_value > last['value']))
        contract = (last['rho'] < tf.constant(config.shrink_threshold, D)) | ~accepted
        expand = (last['rho'] >= tf.constant(config.expansion_threshold, D)) & last['boundary']
        expanded = radius * tf.constant(config.expansion_factor, D)
        bounded_expansion = tf.where(expanded < tf.constant(config.maximum_radius, D),
            expanded, tf.constant(config.maximum_radius, D))
        radius_after = tf.where(contract, radius * tf.constant(config.shrink_factor, D),
            tf.where(expand, bounded_expansion, radius))
        return {'attempted': attempted, 'evaluations': tf.reduce_sum(tf.cast(evaluated.stack(), tf.int32)),
            'accepted': accepted, 'best_exists': best_exists, 'center_value': best_value,
            'center': best_position, 'center_score': best_score,
            'stalled': tf.where(best_exists, 0, stalled + 1), 'last_evaluated': last_evaluated,
            'last': last, 'second_fit': second, 'used': used.stack(), 'evaluated': evaluated.stack(),
            'histories': tf.nest.map_structure(lambda rows: rows.stack(), histories),
            'promoted_without_acceptance': promoted_without_acceptance,
            'radius_after': radius_after, 'radius_action': tf.where(contract, 0, tf.where(expand, 1, 2))}

    return attempt
