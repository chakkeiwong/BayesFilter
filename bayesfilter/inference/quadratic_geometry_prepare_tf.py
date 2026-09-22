"""Internal XLA preparation with fixed capacities and explicit active counts.

Random draws/permutations are inputs. Callers must respect the returned counts;
padding is neither a callback input nor a fitting observation. Public geometry
initializers are unchanged until their enclosing control is qualified.
"""

import math

import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_pilot_tf import (
    make_geometry_cloud_program,
)

D = tf.float64


def _retained_indices(mask, capacity):
    rows = tf.range(capacity)
    return tf.argsort(tf.where(mask, rows, rows + capacity), stable=True)


def _subnormal_square_units(raw):
    """Exact round-to-even square in 2**-1074 units for subnormal products."""
    one = tf.constant(1, tf.uint64)

    def left(value, amount):
        return tf.bitwise.left_shift(value, tf.cast(amount, tf.uint64))

    def right(value, amount):
        return tf.bitwise.right_shift(value, tf.cast(amount, tf.uint64))

    bits = tf.bitcast(raw, tf.uint64)
    exponent = tf.cast(tf.bitwise.bitwise_and(right(bits, 52), tf.constant(2047, tf.uint64)), tf.int32)
    significand = tf.bitwise.bitwise_or(tf.bitwise.bitwise_and(bits, left(one, 52) - one), left(one, 52))
    lower = tf.bitwise.bitwise_and(significand, left(one, 32) - one)
    upper = right(significand, 32)
    cross = 2 * lower * upper
    low_product = lower * lower
    low = low_product + left(cross, 32)
    # The high word is below 2**42, so signed addition is exact. Grappler
    # combines this three-term sum into AddN, which has no uint64 GPU kernel.
    high = tf.cast(tf.cast(upper * upper, tf.int64) + tf.cast(right(cross, 32), tf.int64)
                   + tf.cast(low < low_product, tf.int64), tf.uint64)
    # x = significand * 2**(exponent-1075), so x*x in subnormal
    # units is significand**2 shifted right by 1076-2*exponent.
    shift = 1076 - 2 * exponent
    low_shift = tf.clip_by_value(shift, 1, 63)
    high_shift = tf.clip_by_value(shift - 64, 1, 63)
    quotient_low = tf.bitwise.bitwise_or(right(low, low_shift), left(high, 64 - low_shift))
    remainder_low = tf.bitwise.bitwise_and(low, left(one, low_shift) - one)
    halfway_low = left(one, low_shift - 1)
    remainder_high = tf.bitwise.bitwise_and(high, left(one, high_shift) - one)
    halfway_high = left(one, high_shift - 1)
    quotient = tf.where(shift < 64, quotient_low, tf.where(shift == 64, high, right(high, high_shift)))
    greater = tf.where(shift < 64, remainder_low > halfway_low,
        tf.where(shift == 64, low > left(one, 63),
            (remainder_high > halfway_high) | ((remainder_high == halfway_high) & (low > 0))))
    tie = tf.where(shift < 64, remainder_low == halfway_low,
        tf.where(shift == 64, low == left(one, 63), (remainder_high == halfway_high) & (low == 0)))
    odd = tf.bitwise.bitwise_and(quotient, one) != 0
    rounded = quotient + tf.cast(greater | (tie & odd), tf.uint64)
    return tf.cast(tf.where((exponent >= 485) & (exponent <= 511), rounded, tf.zeros_like(rounded)), D)


def _direction_norms(raw):
    """Preserve gradual underflow in the original binary64 sum of squares."""
    small = tf.reduce_max(tf.abs(raw), axis=1) <= tf.constant(math.ldexp(1., -256), D)
    safe = tf.where(small[:, None], raw, tf.zeros_like(raw))
    scaled = safe * tf.constant(math.ldexp(1., 537), D)
    units = tf.where(tf.abs(safe) < tf.constant(math.ldexp(1., -511), D),
                     _subnormal_square_units(safe), tf.square(scaled))
    restored = tf.sqrt(tf.reduce_sum(units, axis=1)) * tf.constant(math.ldexp(1., -537), D)
    return tf.where(small, restored, tf.linalg.norm(raw, axis=1))


def make_direction_preparation_program(dimension, capacity, *, jit_compile=True):
    """Preserve norm > 0 and source order, including its nonfinite semantics."""
    if dimension < 1 or capacity < 0:
        raise ValueError("invalid direction capacity")

    @tf.function(input_signature=[tf.TensorSpec([capacity, dimension], D)],
                 autograph=False, jit_compile=jit_compile)
    def prepare(raw):
        norms = _direction_norms(raw)
        retained = norms > 0.
        count = tf.math.count_nonzero(retained, dtype=tf.int32)
        indices = _retained_indices(retained, capacity)
        active = tf.range(capacity) < count
        selected = tf.gather(raw, indices)
        denominator = tf.where(active, tf.gather(norms, indices), tf.ones([capacity], D))
        directions = tf.where(active[:, None], selected / denominator[:, None], tf.zeros_like(selected))
        return {"directions": directions, "count": count,
                "indices": tf.where(active, indices, -1)}

    return prepare


def make_geometry_design_program(callback, dimension, capacity, *, batched=False, jit_compile=True):
    """Construct and evaluate exactly the requested design cloud in order."""
    evaluate = make_geometry_cloud_program(callback, dimension, capacity,
        batched=batched, jit_compile=jit_compile)

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([capacity, dimension], D)], autograph=False, jit_compile=jit_compile)
    def design(center, scale, offsets):
        positions = center[None] + offsets * scale[None]
        cloud = evaluate(positions)
        nan = tf.constant(float("nan"), D)
        return {"positions": positions, "values": tf.where(cloud["valid"], cloud["values"], nan),
            "scores": tf.where(cloud["valid"][:, None], cloud["scores"], nan), "valid": cloud["valid"],
            "finite_count": tf.math.count_nonzero(cloud["valid"], dtype=tf.int32)}

    return design


def make_geometry_partition_program(dimension, capacity, required_finite, holdout_fraction, *, jit_compile=True):
    """Compact then permute the finite design; unused output rows are zero.

    The active permutation prefix indexes the compacted finite rows, exactly as
    in the original initializer. Its trailing capacity is ignored. Malformed
    active permutations return permutation_valid=False and no fitting rows.
    """
    if dimension < 1 or capacity < 1 or required_finite < 1:
        raise ValueError("invalid design capacity or finite-row requirement")
    if not math.isfinite(holdout_fraction) or not 0. <= holdout_fraction < 1.:
        raise ValueError("invalid holdout fraction")

    @tf.function(input_signature=[tf.TensorSpec([capacity, dimension], D),
        tf.TensorSpec([capacity], D), tf.TensorSpec([capacity, dimension], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([capacity], tf.int32)],
        autograph=False, jit_compile=jit_compile)
    def partition(offsets, values, scores, scale, permutation):
        finite = tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
        count = tf.math.count_nonzero(finite, dtype=tf.int32)
        indices = _retained_indices(finite, capacity)
        rows = tf.range(capacity)
        active = rows < count
        sufficient = count >= required_finite
        # Sorting the active prefix checks range, uniqueness and completeness.
        sorted_order = tf.sort(tf.where(active, permutation, capacity))
        permutation_valid = tf.reduce_all(tf.where(active, sorted_order == rows, True))
        ready = sufficient & permutation_valid
        holdout = tf.minimum(tf.cast(tf.floor(tf.constant(holdout_fraction, D) * tf.cast(count, D)), tf.int32),
                             tf.maximum(0, count - required_finite))
        holdout = tf.where(ready, holdout, 0)
        training = tf.where(ready, count - holdout, 0)
        # Invalid inputs cannot escape as duplicate/missing numerical samples.
        order = tf.gather(indices, tf.clip_by_value(permutation, 0, capacity - 1))
        train_slots = tf.minimum(rows + holdout, capacity - 1)
        train_indices = tf.gather(order, train_slots)
        train_active, holdout_active = rows < training, rows < holdout
        train_z = tf.gather(offsets, train_indices)
        train_score = tf.gather(scores, train_indices) * scale[None]
        return {"finite_mask": finite, "finite_count": count, "sufficient": sufficient,
            "permutation_valid": permutation_valid, "ready": ready,
            "finite_indices": tf.where(active, indices, -1),
            "training_count": training, "holdout_count": holdout,
            "train_indices": tf.where(train_active, train_indices, -1),
            "holdout_indices": tf.where(holdout_active, order, -1),
            "z_train": tf.where(train_active[:, None], train_z, tf.zeros_like(train_z)),
            "y_train": tf.where(train_active, tf.gather(values, train_indices), tf.zeros_like(values)),
            "score_train": tf.where(train_active[:, None], train_score, tf.zeros_like(train_score)),
            "z_holdout": tf.where(holdout_active[:, None], tf.gather(offsets, order), tf.zeros_like(offsets)),
            "y_holdout": tf.where(holdout_active, tf.gather(values, order), tf.zeros_like(values))}

    return partition
