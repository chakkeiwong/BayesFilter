"""Internal XLA center proposal and exact replay for quadratic initialization.

The public geometry wrapper is unchanged pending complete compatibility checks.
These kernels consume a TensorFlow scalar target with a same-dimension score;
they do not infer callback exceptions or claim validity of the enclosing fit.
"""

from threading import RLock

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.quadratic_geometry import _trust_region_kernel

D = tf.float64
_LOCK = RLock()
_LAST_REFINEMENT = None

STEP_FAILURES = (
    "trust_region_invalid_input",
    "trust_region_precision_not_spd",
    "trust_region_bracket_failed",
    "trust_region_solution_invalid",
    "precision_solve_failed",
)
REJECTION_REASONS = (
    "outside_trust_radius",
    "log_prob_decreased",
    "score_norm_not_improved_enough",
    "actual_improvement_not_positive",
    "predicted_improvement_not_positive",
    "improvement_ratio_nonfinite",
)


def _target(callback, point):
    value, score = callback(point)
    return (tf.reshape(tf.convert_to_tensor(value, D), []),
            tf.ensure_shape(tf.reshape(tf.convert_to_tensor(score, D), [-1]), point.shape))


def _finite(value):
    return tf.reduce_all(tf.math.is_finite(value))


def _has_zero_lu_pivot(matrix):
    """Use partial pivoting's exact-zero test without unsupported LU/det ops.

    This is a solve-error status, not a tolerance-based numerical rank rule.
    The actual solution still comes from the original TensorFlow linear solve.
    """
    dimension = matrix.shape[0]
    indices = tf.range(dimension)

    def eliminate(index, current, _singular):
        column = tf.gather(current, index, axis=1)
        pivot = tf.argmax(tf.where(indices >= index, tf.abs(column), tf.constant(-1., D)), output_type=tf.int32)
        permutation = tf.where(indices == index, pivot, tf.where(indices == pivot, index, indices))
        swapped = tf.gather(current, permutation)
        leading = tf.gather(tf.gather(swapped, index), index)
        singular = leading == 0.0
        multipliers = tf.gather(swapped, index, axis=1) / tf.where(singular, tf.constant(1., D), leading)
        updated = swapped - multipliers[:, None] * tf.gather(swapped, index)[None, :]
        trailing = (indices[:, None] > index) & (indices[None, :] > index)
        return index + 1, tf.where(trailing, updated, swapped), singular

    return tf.while_loop(lambda index, _current, singular: (index < dimension) & ~singular,
        eliminate, (tf.constant(0), matrix, tf.constant(False)),
        maximum_iterations=dimension, parallel_iterations=1)[2]


def center_refinement_program(callback, dimension, config, *, jit_compile=True):
    """Bound callback ownership to one program and compare callables by identity."""
    global _LAST_REFINEMENT
    settings = (dimension, bool(config.constrain_center_refinement_to_trust_region),
                config.trust_radius, config.center_log_prob_tolerance,
                config.center_score_improvement_factor, jit_compile)
    with _LOCK:
        previous = _LAST_REFINEMENT
        if previous is not None and previous[0] is callback and previous[1] == settings:
            return previous[2]
        _LAST_REFINEMENT = None
        del previous
        program = make_center_refinement_program(callback, dimension, config, jit_compile=jit_compile)
        _LAST_REFINEMENT = (callback, settings, program)
        return program


def clear_center_refinement_cache():
    """Release Python ownership; native executable eviction remains unproved."""
    global _LAST_REFINEMENT
    with _LOCK:
        _LAST_REFINEMENT = None


def make_center_refinement_program(callback, dimension, config, *, jit_compile=True):
    """Enclose proposal, one target call and every acceptance decision."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    constrained = bool(config.constrain_center_refinement_to_trust_region)
    eigenpairs = eigenpair_program(dimension) if jit_compile else tf.linalg.eigh

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([], D), tf.TensorSpec([], D)], autograph=False, jit_compile=jit_compile)
    def evaluate(center, scale, precision, linear, center_value, center_score_norm):
        radius = tf.constant(config.trust_radius, D)
        value_tolerance = tf.constant(config.center_log_prob_tolerance, D)
        score_factor = tf.constant(config.center_score_improvement_factor, D)
        empty_step = {"status": tf.constant(1), "step": tf.zeros([dimension], D),
            "boundary": tf.constant(False), "multiplier": tf.constant(0.0, D), "predicted": tf.constant(0.0, D)}
        if constrained:
            valid_inputs = _finite(precision) & _finite(linear) & tf.math.is_finite(radius) & (radius > 0.0)
            safe_precision = tf.where(valid_inputs, precision, tf.eye(dimension, dtype=D))
            safe_linear = tf.where(valid_inputs, linear, tf.zeros_like(linear))
            result = _trust_region_kernel(safe_precision, safe_linear, radius, eigenpairs=eigenpairs)
            step = {"status": tf.where(~valid_inputs, 1,
                tf.where(result["status_code"] == 0, 0, result["status_code"] + 1)),
                "step": result["step"], "boundary": result["boundary_active"],
                "multiplier": result["lagrange_multiplier"], "predicted": result["predicted_improvement"]}
        else:
            # Native solves need an explicit singular status: XLA cannot raise
            # the eager solve's InvalidArgumentError from inside the program.
            singular = _has_zero_lu_pivot(precision)

            def solve():
                position = tf.linalg.solve(precision, linear[:, None])[:, 0]
                predicted = tf.reduce_sum(linear * position) - .5 * tf.reduce_sum(
                    position * tf.linalg.matvec(precision, position))
                return {**empty_step, "status": tf.constant(0), "step": position, "predicted": predicted}

            step = tf.cond(singular, lambda: {**empty_step, "status": tf.constant(5)}, solve)
        empty = {"status": step["status"], "accepted": tf.constant(False),
            "reason_mask": tf.zeros([6], tf.bool), "target_called": tf.constant(False),
            "target_finite": tf.constant(False), "refined_center": center,
            "refined_value": tf.constant(float("nan"), D), "refined_score": tf.zeros([dimension], D),
            "score_norm": tf.constant(0.0, D), "z_norm": tf.constant(0.0, D),
            "actual": tf.constant(0.0, D), "ratio": tf.constant(0.0, D),
            "has_ratio": tf.constant(False), "boundary": step["boundary"],
            "multiplier": step["multiplier"], "predicted": step["predicted"]}

        def replay():
            refined = center + step["step"] * scale
            value, score = _target(callback, refined)
            finite = tf.math.is_finite(value) & _finite(score)
            z_norm = tf.linalg.norm(step["step"])
            score_norm = tf.linalg.norm(score * scale)
            actual = value - center_value
            has_ratio = step["predicted"] > 0.0
            ratio = tf.where(has_ratio, actual / step["predicted"], tf.constant(float("nan"), D))
            reasons = tf.stack((z_norm > radius, value < center_value - value_tolerance,
                score_norm > score_factor * center_score_norm,
                tf.constant(constrained) & (actual <= 0.0),
                tf.constant(constrained) & (step["predicted"] <= 0.0),
                tf.constant(constrained) & (~has_ratio | ~tf.math.is_finite(ratio))))
            # Keep the original positive predicates, including their NaN
            # behavior; the diagnostic reason list is not the acceptance rule.
            accepted = (finite & (z_norm <= radius) & (value >= center_value - value_tolerance)
                & (score_norm <= score_factor * center_score_norm))
            if constrained:
                accepted &= (actual > 0.0) & has_ratio & tf.math.is_finite(ratio)
            return {**empty, "status": tf.where(finite, 0, 6), "accepted": accepted,
                "reason_mask": reasons, "target_called": tf.constant(True), "target_finite": finite,
                "refined_center": refined, "refined_value": value, "refined_score": score,
                "score_norm": score_norm, "z_norm": z_norm, "actual": actual, "ratio": ratio,
                "has_ratio": has_ratio}

        return tf.cond(step["status"] == 0, replay, lambda: empty)

    return evaluate


def make_exact_replay_program(callback, dimension, *, jit_compile=True):
    """Replay an incumbent without changing its original provenance or values."""
    if dimension < 1:
        raise ValueError("dimension must be positive")

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([], tf.bool)], autograph=False, jit_compile=jit_compile)
    def replay(point, incumbent_value, incumbent_score, has_incumbent):
        def evaluate():
            value, score = _target(callback, point)
            valid = tf.math.is_finite(value) & _finite(score)
            matches = (valid & (tf.abs(value - incumbent_value) <= 1e-12 + 1e-10 * tf.abs(incumbent_value))
                & tf.reduce_all(tf.abs(score - incumbent_score) <= 1e-11 + 1e-9 * tf.abs(incumbent_score)))
            return {"attempted": tf.constant(True), "valid": valid, "matches": matches, "value": value}

        return tf.cond(has_incumbent, evaluate, lambda: {"attempted": tf.constant(False),
            "valid": tf.constant(False), "matches": tf.constant(False), "value": tf.constant(float("nan"), D)})

    return replay
