"""Native family, consensus and shrinkage selection for fixed-center fits.

Python factories describe static family/partition shapes. Every numerical
eligibility check, pair comparison, reduction and candidate decision executes
inside the returned program; completed reports are reconstructed by the caller.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference import fixed_center_stability_tf as stability

D = tf.float64


@lru_cache(maxsize=64)
def mean_error_program(score_kernel, dimension, partition_rows, *, jit_compile=True):
    """Evaluate each whole partition in order, with no row-wise target calls."""
    count, rows = len(partition_rows), max(partition_rows, default=0)
    distinct_rows = tuple(sorted(set(partition_rows)))
    row_branches = tuple(distinct_rows.index(size) for size in partition_rows)

    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([count, rows, dimension], D),
        tf.TensorSpec([count, rows, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def compute(precision, center, offsets, scores):
        if not count:
            return tf.constant(float("nan"), D)
        branch_ids = tf.constant(row_branches, tf.int32)

        def step(index, errors):
            # Branches differ only in the static partition row extent.
            def branch(size):
                return lambda: score_kernel(precision, center, offsets[index, :size], scores[index, :size])
            error = tf.switch_case(branch_ids[index], tuple(branch(size) for size in distinct_rows))
            return index + 1, tf.tensor_scatter_nd_update(errors, [[index]], [error])

        _, errors = tf.while_loop(lambda index, _: index < count, step,
            (tf.constant(0), tf.zeros([count], D)), maximum_iterations=count, parallel_iterations=1)
        return tf.reduce_mean(errors)

    return compute


@lru_cache(maxsize=64)
def selection_program(comparison_kernel, score_kernel, dimension, families,
        family_counts, partition_rows, weight_count, structured_target_family, *, jit_compile=True):
    """Enclose all family checks and ordered selection in one stable graph."""
    family_count = len(families)
    capacity = max(family_counts, default=0)
    max_pairs = capacity * (capacity - 1) // 2
    report_width = 3 * dimension + 10
    distinct_counts = tuple(sorted(set(family_counts)))
    branch_ids = tuple(distinct_counts.index(count) for count in family_counts)
    partition_count, rows = len(partition_rows), max(partition_rows, default=0)
    errors_program = mean_error_program(score_kernel, dimension, partition_rows, jit_compile=jit_compile)
    programs = tuple(stability.stability_program(comparison_kernel, dimension, count,
        jit_compile=jit_compile) for count in distinct_counts)
    first_index = families.index("factor_1") if "factor_1" in families else -1
    second_index = families.index("factor_2") if "factor_2" in families else -1
    dense_index = families.index("dense") if "dense" in families else -1
    target_index = families.index(structured_target_family) if structured_target_family in families else -1

    @tf.function(input_signature=[tf.TensorSpec([family_count, capacity, dimension, dimension], D),
        tf.TensorSpec([family_count, capacity, 3], tf.bool), tf.TensorSpec([dimension], D),
        tf.TensorSpec([partition_count, rows, dimension], D), tf.TensorSpec([partition_count, rows, dimension], D),
        tf.TensorSpec([4], D), tf.TensorSpec([4], tf.bool), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([weight_count], D), tf.TensorSpec([], D)],
        jit_compile=jit_compile, autograph=False)
    def compute(matrices, flags, center, offsets, scores, caps, enabled, rank, weights, holdout_cap):
        states = {"complete": tf.zeros([family_count], tf.bool),
            "usable_count": tf.zeros([family_count], tf.int32), "error": tf.zeros([family_count], tf.int32),
            "reports": tf.zeros([family_count, max_pairs, report_width], D),
            "checks": tf.zeros([family_count, max_pairs, 4], tf.bool),
            "pair_passed": tf.zeros([family_count, max_pairs], tf.bool),
            "passed": tf.zeros([family_count], tf.bool)}

        if family_count:
            def check_family(index, error, states):
                def branch(count, program):
                    def run():
                        result = program(matrices[index, :count], flags[index, :count, :2], caps, enabled, rank)
                        padding = max_pairs - count * (count - 1) // 2
                        return {**result, "reports": tf.pad(result["reports"], [[0, padding], [0, 0]]),
                            "checks": tf.pad(result["checks"], [[0, padding], [0, 0]]),
                            "pair_passed": tf.pad(result["pair_passed"], [[0, padding]])}
                    return run
                result = tf.switch_case(tf.constant(branch_ids, tf.int32)[index],
                    tuple(branch(count, program) for count, program in zip(distinct_counts, programs, strict=True)))
                states = tf.nest.map_structure(lambda all_rows, row:
                    tf.tensor_scatter_nd_update(all_rows, [[index]], [row]), states, result)
                return index + 1, result["error"], states

            _, _, states = tf.while_loop(lambda index, error, _: (index < family_count) & (error == 0),
                check_family, (tf.constant(0), tf.constant(0), states),
                maximum_iterations=family_count, parallel_iterations=1)

        def mean(family_index, accepted, symmetric=False):
            if family_index < 0:
                return tf.zeros([dimension, dimension], D), tf.constant(0)
            mask = flags[family_index, :, 0] & flags[family_index, :, 2 if accepted else 1]
            values = matrices[family_index]
            if symmetric:
                values = .5 * (values + tf.linalg.matrix_transpose(values))
            count = tf.math.count_nonzero(mask, dtype=tf.int32)
            values = tf.where(mask[:, None, None], values, tf.zeros_like(values))
            return tf.reduce_sum(values, axis=0) / tf.cast(tf.maximum(count, 1), D), count

        def stable(index):
            return tf.constant(False) if index < 0 else states["passed"][index]

        first, first_count = mean(first_index, True)
        second, second_count = mean(second_index, True)
        direct = tf.constant(0)
        if structured_target_family is None:
            direct = tf.where(stable(first_index) & (first_count >= 2), 1,
                tf.where(stable(second_index) & (second_count >= 2), 2, 0))
        raw_consensus, dense_count = mean(dense_index, False)
        consensus, _ = mean(dense_index, False, symmetric=True)
        if structured_target_family is None:
            target = tf.linalg.diag(tf.linalg.diag_part(raw_consensus))
            target_ok = tf.constant(True)
            consensus_code = 3
        else:
            target, target_count = mean(target_index, True)
            target_ok = stable(target_index) & (target_count > 0)
            consensus_code = 4 if structured_target_family == "factor_1" else 5
        target = .5 * (target + tf.transpose(target))
        can_shrink = (direct == 0) & stable(dense_index) & (dense_count > 0) & target_ok

        def shrink():
            def step(index, error_code, best_index, best_error, best_weight, best, errors):
                weight = weights[index]
                candidate = (1. - weight) * consensus + weight * target
                candidate = .5 * (candidate + tf.transpose(candidate))
                # Match the public SPD veto, including its NaN comparison semantics.
                error_code = tf.where(tf.reduce_min(tf.linalg.eigvalsh(candidate)) <= 0., 4, 0)
                error = tf.cond(error_code == 0, lambda: errors_program(candidate, center, offsets, scores),
                    lambda: tf.constant(0., D))
                better = (index == 0) | (error < best_error) | ((error == best_error) & (weight < best_weight))
                return (index + 1, error_code, tf.where(better, index, best_index),
                    tf.where(better, error, best_error), tf.where(better, weight, best_weight),
                    tf.where(better, candidate, best), tf.tensor_scatter_nd_update(errors, [[index]], [error]))
            return tf.while_loop(lambda index, error, *_: (index < weight_count) & (error == 0), step,
                (tf.constant(0), tf.constant(0), tf.constant(-1), tf.constant(0., D), tf.constant(0., D),
                    tf.zeros([dimension, dimension], D), tf.zeros([weight_count], D)),
                maximum_iterations=weight_count, parallel_iterations=1)

        visited, error, best_index, best_error, best_weight, candidate, errors = tf.cond(
            can_shrink & ~tf.reduce_any(states["error"] != 0), shrink,
            lambda: (tf.constant(0), tf.constant(0), tf.constant(-1), tf.constant(0., D),
                tf.constant(0., D), tf.zeros([dimension, dimension], D), tf.zeros([weight_count], D)))
        selected = tf.where(direct != 0, direct,
            tf.where(can_shrink & (error == 0) & ~(best_error > holdout_cap), consensus_code, 0))
        selected = tf.where(tf.reduce_any(states["error"] != 0), 0, selected)
        return {"stability": states, "family_code": selected,
            "precision": tf.where(direct == 1, first, tf.where(direct == 2, second, candidate)),
            "visited": visited, "error": error, "errors": errors,
            "selected_index": tf.where(selected >= 3, best_index, -1), "weight": best_weight,
            "diagonal_only": (selected == 3) & (best_weight == 1.)}

    return compute
