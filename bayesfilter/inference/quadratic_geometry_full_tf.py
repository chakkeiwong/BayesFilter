"""Complete native geometry attempt on versioned, pre-generated probe inputs.

This boundary owns every numerical decision from center evaluation to replay.
Random cloud generation is CPU/XLA preparation; formatting is a separate host
boundary. The public initializer uses this program without host numerical feedback.
"""

import math
from threading import RLock

import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_control_tf import _callback_program
from bayesfilter.inference.quadratic_geometry_fit_tf import make_geometry_fit_program
from bayesfilter.inference.quadratic_geometry_pilot_tf import (
    make_geometry_pilot_program,
)
from bayesfilter.inference.quadratic_geometry_prepare_tf import (
    make_direction_preparation_program,
    make_geometry_design_program,
    make_geometry_seeded_partition_program,
)
from bayesfilter.ops.geometry_random_tf import GeometryTensorStream

D = tf.float64
STAGES = ("center_value_or_score_nonfinite", "pilot_eigenbasis_ill_conditioned",
          "insufficient_finite_samples", "fit_completed")
_LOCK = RLock()
_LAST_PROGRAM = None


def geometry_extents(dimension, config):
    """Configuration-only shape bounds from the original sample-ratio rule."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    rank = min(config.rank, dimension - 1)
    parameters = dimension + rank + 2
    required = config.min_samples_per_parameter * parameters
    samples = config.sample_count if config.sample_count is not None else max(required + 20, 8 * parameters)
    directions = (config.pilot_direction_count if config.pilot_direction_count is not None
                  else max(4 * dimension, 2 * rank + 8)) if rank else 0
    return rank, required, samples, directions


def prepare_geometry_inputs(dimension, config):
    """Preserve normal/ball/permutation seed order without target decisions."""
    rank, _, samples, directions = geometry_extents(dimension, config)
    stream = GeometryTensorStream(config.seed)
    raw = stream.normal(size=(directions, dimension)) if rank else tf.zeros([0, dimension], D)
    offsets = stream.ball(samples, dimension, radius=config.trust_radius)
    return raw, offsets, stream._next_seed("permutation")


def geometry_program(callback, dimension, config, *, batched_callback=None, jit_compile=True):
    """Keep one complete callback-dependent program, comparing by identity."""
    global _LAST_PROGRAM
    settings = (dimension, config, jit_compile)
    with _LOCK:
        previous = _LAST_PROGRAM
        if (previous is not None and previous[0] is callback and previous[1] is batched_callback
                and previous[2] == settings):
            return previous[3]
        _LAST_PROGRAM = None
        del previous
        program = make_geometry_program(callback, dimension, config,
            batched_callback=batched_callback, jit_compile=jit_compile)
        _LAST_PROGRAM = (callback, batched_callback, settings, program)
        return program


def clear_geometry_program_cache():
    """Release this Python owner; native executable residency is separate."""
    global _LAST_PROGRAM
    with _LOCK:
        _LAST_PROGRAM = None


def _incumbent(center, center_value, center_score, center_valid, pilot, design, design_ran):
    """Earliest finite maximum, with logical indices excluding inactive rows."""
    pilot_capacity = pilot['values'].shape[0]
    design_capacity = design['values'].shape[0]
    pilot_rows = 2 * pilot['direction_count']
    positions = tf.concat((center[None], pilot['positions'], design['positions']), 0)
    values = tf.concat((center_value[None], pilot['values'], design['values']), 0)
    scores = tf.concat((center_score[None], pilot['scores'], design['scores']), 0)
    active = tf.concat((center_valid[None], center_valid & pilot['valid'] & (tf.range(pilot_capacity) < pilot_rows),
                        design_ran & design['valid']), 0)
    eligible = active & tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
    eligible &= tf.reduce_all(tf.math.is_finite(positions), axis=1)
    selected = tf.argmax(tf.where(eligible, values, tf.constant(float('-inf'), D)), output_type=tf.int32)
    present = tf.reduce_any(eligible)
    logical = tf.concat((tf.zeros([1], tf.int32), 1 + tf.range(pilot_capacity),
                         1 + pilot_rows + tf.range(design_capacity)), 0)
    role = tf.where(selected == 0, 0, tf.where(selected <= pilot_capacity, 1, 2))
    return {'position': tf.gather(positions, selected), 'value': tf.gather(values, selected),
        'score': tf.gather(scores, selected), 'index': tf.cast(tf.where(present, tf.gather(logical, selected), -1), tf.int64),
        'source': role, 'present': present}


def make_geometry_program(callback, dimension, config, *, batched_callback=None, jit_compile=True):
    """Compile one complete attempt; cloud values and permutation seed are operands."""
    rank, required, samples, directions = geometry_extents(dimension, config)
    target = _callback_program(callback, dimension, jit_compile)
    cloud_callback = callback if batched_callback is None else batched_callback
    batched = batched_callback is not None
    prepare_directions = make_direction_preparation_program(dimension, directions, jit_compile=jit_compile)
    pilot = make_geometry_pilot_program(cloud_callback, dimension, rank, directions,
        batched=batched, active_rows=True, jit_compile=jit_compile)
    design = make_geometry_design_program(cloud_callback, dimension, samples,
        batched=batched, jit_compile=jit_compile)
    partition = make_geometry_seeded_partition_program(dimension, samples, required,
        config.holdout_fraction, jit_compile=jit_compile)
    # H(N) and N-H(N) are nondecreasing for required <= N <= samples.
    maximum_holdout = min(math.floor(config.holdout_fraction * samples), max(0, samples - required))
    maximum_train = samples - maximum_holdout
    fit = (make_geometry_fit_program(callback, dimension, rank, maximum_train, maximum_holdout,
        config, active_rows=True, minimum_train_rows=required, jit_compile=jit_compile)
        if samples >= required else None)
    # These inspect output schemas only. No target is executed during construction.
    pilot_template = pilot.get_concrete_function().structured_outputs
    design_template = design.get_concrete_function().structured_outputs
    partition_template = partition.get_concrete_function().structured_outputs
    fit_template = fit.get_concrete_function().structured_outputs if fit is not None else {}

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([directions, dimension], D), tf.TensorSpec([samples, dimension], D),
        tf.TensorSpec([2], tf.int32)], autograph=False, jit_compile=jit_compile)
    def evaluate(center, scale, raw_directions, offsets, permutation_seed):
        center_value, center_score = target(center)
        center_valid = tf.math.is_finite(center_value) & tf.reduce_all(tf.math.is_finite(center_score))
        center_score_z = center_score * scale
        prepared = prepare_directions(raw_directions)
        pilot_result = tf.cond(center_valid, lambda: pilot(center, scale, prepared['directions'],
            tf.constant(config.pilot_radius, D), center_score_z, prepared['count']),
            lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), pilot_template))
        design_ran = center_valid & pilot_result['basis_resolved']
        design_result = tf.cond(design_ran, lambda: design(center, scale, offsets),
            lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), design_template))
        partition_result = tf.cond(design_ran & (design_result['finite_count'] >= required),
            lambda: partition(offsets, design_result['values'], design_result['scores'], scale, permutation_seed),
            lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), partition_template))
        incumbent = _incumbent(center, center_value, center_score, center_valid, pilot_result, design_result, design_ran)
        prefix_count = tf.cast(1 + 2 * pilot_result['direction_count'] + tf.where(design_ran, samples, 0), tf.int64)
        fit_ran = design_ran & partition_result['ready']
        if fit is not None:
            fit_result = tf.cond(fit_ran, lambda: fit(center, scale, pilot_result['q_basis'],
                partition_result['z_train'][:maximum_train], partition_result['y_train'][:maximum_train],
                partition_result['score_train'][:maximum_train], partition_result['z_holdout'][:maximum_holdout],
                partition_result['y_holdout'][:maximum_holdout], center_value, center_score_z,
                incumbent['position'], incumbent['value'], incumbent['score'], incumbent['index'],
                incumbent['source'], incumbent['present'], prefix_count,
                partition_result['training_count'], partition_result['holdout_count']),
                lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), fit_template))
            attempted = tf.where(fit_ran, fit_result['evaluation_count'], prefix_count)
        else:
            fit_result, attempted = {}, prefix_count
        # Original early-center results record zero rows after the attempted call.
        recorded = tf.where(center_valid, attempted, tf.constant(0, tf.int64))
        return {'stage': tf.where(~center_valid, 0, tf.where(~design_ran, 1, tf.where(~fit_ran, 2, 3))),
            'center_value': center_value, 'center_score_norm': tf.linalg.norm(center_score_z),
            'pilot': pilot_result, 'design': design_result, 'partition': partition_result,
            'fit_result': fit_result, 'incumbent': incumbent,
            'evaluation_count': recorded, 'attempted_evaluation_count': attempted}

    return evaluate
