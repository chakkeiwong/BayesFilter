"""Paired bundle/path diagnostics for classifier-score path-count scaling."""

from __future__ import annotations

import math
from collections.abc import Sequence

import tensorflow as tf


DTYPE = tf.float64


def _outputs(values: tf.Tensor, counts: Sequence[int]) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if tensor.shape.rank != 4:
        raise ValueError("outputs must have shape [count,bundle,path,cell]")
    if any(dimension is None for dimension in tensor.shape):
        raise ValueError("output shape must be static")
    if int(tensor.shape[0]) != len(counts):
        raise ValueError("output count dimension differs from path counts")
    if int(tensor.shape[1]) < 2 or int(tensor.shape[2]) < 2:
        raise ValueError("at least two bundles and two audit paths are required")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError("outputs contain non-finite values")
    return tensor


def _fixed(values: tf.Tensor, levels: int, bundles: int, cells: int) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if tensor.shape != (levels, bundles, cells):
        raise ValueError("fixed outputs must have shape [count,bundle,cell]")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError("fixed outputs contain non-finite values")
    return tensor


def sample_variance(values: tf.Tensor, *, axis: int) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    count = int(tensor.shape[axis])
    if count < 2:
        raise ValueError("sample variance needs at least two rows")
    centered = tensor - tf.reduce_mean(tensor, axis=axis, keepdims=True)
    return tf.reduce_sum(tf.square(centered), axis=axis) / tf.cast(count - 1, DTYPE)


def _percentile(values: tf.Tensor, probability: float) -> float:
    ordered = tf.sort(tf.cast(tf.reshape(values, [-1]), DTYPE))
    index = max(
        0,
        min(
            int(ordered.shape[0]) - 1,
            math.ceil(float(probability) * int(ordered.shape[0])) - 1,
        ),
    )
    return float(ordered[index].numpy())


def _positive_floor(values: tf.Tensor) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    return tf.maximum(
        tf.reduce_max(tf.abs(tensor)) * tf.constant(1.0e-15, DTYPE),
        tf.constant(1.0e-18, DTYPE),
    )


def _scaling_classification(lower: float, upper: float) -> str:
    if upper >= 1.0:
        return "no_supported_variance_reduction"
    if upper < 0.5:
        return "faster_than_1_over_n"
    if lower > 0.5:
        return "slower_than_1_over_n"
    return "compatible_with_1_over_n"


def _adjacent_rows(
    counts: tuple[int, ...], point: tf.Tensor, draws: tf.Tensor
) -> list[dict[str, object]]:
    floor = _positive_floor(draws)
    rows: list[dict[str, object]] = []
    for index in range(len(counts) - 1):
        if counts[index + 1] != 2 * counts[index]:
            raise ValueError("path-count levels must be adjacent doublings")
        point_ratio = point[index + 1] / tf.maximum(point[index], floor)
        ratio_draws = draws[:, index + 1] / tf.maximum(draws[:, index], floor)
        exponent_draws = -tf.math.log(tf.maximum(ratio_draws, floor)) / tf.math.log(
            tf.constant(2.0, DTYPE)
        )
        ratio = float(point_ratio.numpy())
        lower = _percentile(ratio_draws, 0.025)
        upper = _percentile(ratio_draws, 0.975)
        rows.append(
            {
                "from_count": counts[index],
                "to_count": counts[index + 1],
                "variance_ratio": ratio,
                "ratio_lower_95": lower,
                "ratio_upper_95": upper,
                "normalized_1_over_n_efficiency": ratio / 0.5,
                "scaling_exponent": -math.log(max(ratio, 1.0e-300), 2.0),
                "exponent_lower_95": _percentile(exponent_draws, 0.025),
                "exponent_upper_95": _percentile(exponent_draws, 0.975),
                "classification": _scaling_classification(lower, upper),
            }
        )
    return rows


def _global_exponent(
    counts: tuple[int, ...], point: tf.Tensor, draws: tf.Tensor
) -> dict[str, float] | None:
    if len(counts) < 3:
        return None
    x = tf.math.log(tf.cast(counts, DTYPE))
    centered_x = x - tf.reduce_mean(x)
    denominator = tf.reduce_sum(tf.square(centered_x))
    floor = _positive_floor(draws)

    def exponent(values: tf.Tensor) -> tf.Tensor:
        log_values = tf.math.log(tf.maximum(values, floor))
        centered_y = log_values - tf.reduce_mean(log_values, axis=-1, keepdims=True)
        slope = tf.reduce_sum(centered_y * centered_x, axis=-1) / denominator
        return -slope

    point_exponent = exponent(point[None, :])[0]
    exponent_draws = exponent(draws)
    return {
        "point": float(point_exponent.numpy()),
        "lower_95": _percentile(exponent_draws, 0.025),
        "upper_95": _percentile(exponent_draws, 0.975),
    }


def summarize_path_count_scaling(
    outputs: tf.Tensor,
    fixed_outputs: tf.Tensor,
    *,
    counts: Sequence[int],
    exact_scores: tf.Tensor | None = None,
    exact_fixed_score: tf.Tensor | None = None,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: tuple[int, int] = (4171, 6199),
    chunk_size: int = 100,
) -> dict[str, object]:
    """Summarize adjacent variance scaling for exact path-count doublings."""

    frozen_counts = tuple(int(value) for value in counts)
    if len(frozen_counts) not in (2, 3):
        raise ValueError("the scaling ladder requires two or three count levels")
    if frozen_counts[0] != 8192 or any(value <= 0 for value in frozen_counts):
        raise ValueError("the ladder must start at the frozen 8192 baseline")
    crossed = _outputs(outputs, frozen_counts)
    levels, bundles, paths, cells = (int(value) for value in crossed.shape)
    fixed = _fixed(fixed_outputs, levels, bundles, cells)
    if int(bootstrap_replicates) < 100:
        raise ValueError("at least 100 bootstrap replicates are required")

    bundle_variance_by_path_cell = sample_variance(crossed, axis=1)
    bundle_variance_by_cell = tf.reduce_mean(bundle_variance_by_path_cell, axis=1)
    fixed_bundle_variance_by_cell = sample_variance(fixed, axis=1)
    baseline_bundle_mean = tf.reduce_mean(crossed[0], axis=0)
    natural_path_variance = sample_variance(baseline_bundle_mean, axis=0)
    scale_floor = tf.maximum(
        tf.reduce_max(natural_path_variance) * tf.constant(1.0e-12, DTYPE),
        tf.constant(1.0e-18, DTYPE),
    )
    scale_variance = tf.maximum(natural_path_variance, scale_floor)
    joint_audit = tf.reduce_mean(bundle_variance_by_cell / scale_variance[None, :], axis=1)
    joint_fixed = tf.reduce_mean(
        fixed_bundle_variance_by_cell / scale_variance[None, :], axis=1
    )

    exact = None
    exact_fixed = None
    joint_mse = None
    joint_fixed_mse = None
    if exact_scores is not None:
        exact = tf.cast(tf.convert_to_tensor(exact_scores), DTYPE)
        exact_fixed = tf.cast(tf.convert_to_tensor(exact_fixed_score), DTYPE)
        if exact.shape != (paths, cells) or exact_fixed.shape != (cells,):
            raise ValueError("exact scores do not match ladder dimensions")
        errors = crossed - exact[None, None, :, :]
        fixed_errors = fixed - exact_fixed[None, None, :]
        mse_by_cell = tf.reduce_mean(tf.square(errors), axis=(1, 2))
        fixed_mse_by_cell = tf.reduce_mean(tf.square(fixed_errors), axis=1)
        joint_mse = tf.reduce_mean(mse_by_cell / scale_variance[None, :], axis=1)
        joint_fixed_mse = tf.reduce_mean(
            fixed_mse_by_cell / scale_variance[None, :], axis=1
        )

    audit_draws: list[tf.Tensor] = []
    fixed_draws: list[tf.Tensor] = []
    mse_draws: list[tf.Tensor] = []
    fixed_mse_draws: list[tf.Tensor] = []
    completed = 0
    while completed < int(bootstrap_replicates):
        draw_count = min(int(chunk_size), int(bootstrap_replicates) - completed)
        bundle_indices = tf.random.stateless_uniform(
            [draw_count, bundles],
            [int(bootstrap_seed[0]), int(bootstrap_seed[1]) + completed],
            minval=0,
            maxval=bundles,
            dtype=tf.int32,
        )
        path_indices = tf.random.stateless_uniform(
            [draw_count, paths],
            [int(bootstrap_seed[0]) + 1, int(bootstrap_seed[1]) + completed],
            minval=0,
            maxval=paths,
            dtype=tf.int32,
        )
        selected_bundles = tf.transpose(
            tf.gather(crossed, bundle_indices, axis=1), [1, 0, 2, 3, 4]
        )
        selected = tf.gather(selected_bundles, path_indices, axis=3, batch_dims=1)
        variance_by_cell = tf.reduce_mean(sample_variance(selected, axis=2), axis=2)
        audit_draws.append(
            tf.reduce_mean(variance_by_cell / scale_variance[None, None, :], axis=2)
        )

        selected_fixed = tf.transpose(
            tf.gather(fixed, bundle_indices, axis=1), [1, 0, 2, 3]
        )
        fixed_variance = sample_variance(selected_fixed, axis=2)
        fixed_draws.append(
            tf.reduce_mean(fixed_variance / scale_variance[None, None, :], axis=2)
        )

        if exact is not None:
            selected_exact = tf.gather(exact, path_indices, axis=0)
            selected_errors = selected - selected_exact[:, None, None, :, :]
            cell_mse = tf.reduce_mean(tf.square(selected_errors), axis=(2, 3))
            mse_draws.append(
                tf.reduce_mean(cell_mse / scale_variance[None, None, :], axis=2)
            )
            fixed_errors = selected_fixed - exact_fixed[None, None, :]
            fixed_cell_mse = tf.reduce_mean(tf.square(fixed_errors), axis=2)
            fixed_mse_draws.append(
                tf.reduce_mean(
                    fixed_cell_mse / scale_variance[None, None, :], axis=2
                )
            )
        completed += draw_count

    audit_bootstrap = tf.concat(audit_draws, axis=0)
    fixed_bootstrap = tf.concat(fixed_draws, axis=0)
    result: dict[str, object] = {
        "counts": frozen_counts,
        "bundle_count": bundles,
        "audit_path_count": paths,
        "cell_count": cells,
        "bundle_variance_by_cell": bundle_variance_by_cell,
        "fixed_bundle_variance_by_cell": fixed_bundle_variance_by_cell,
        "natural_path_variance_by_cell": natural_path_variance,
        "scale_variance_by_cell": scale_variance,
        "joint_audit_bundle_variance": joint_audit,
        "joint_fixed_bundle_variance": joint_fixed,
        "audit_adjacent_scaling": _adjacent_rows(
            frozen_counts, joint_audit, audit_bootstrap
        ),
        "fixed_adjacent_scaling": _adjacent_rows(
            frozen_counts, joint_fixed, fixed_bootstrap
        ),
        "audit_global_exponent": _global_exponent(
            frozen_counts, joint_audit, audit_bootstrap
        ),
        "fixed_global_exponent": _global_exponent(
            frozen_counts, joint_fixed, fixed_bootstrap
        ),
        "bootstrap_replicates": int(bootstrap_replicates),
    }
    if exact is not None and joint_mse is not None and joint_fixed_mse is not None:
        mse_bootstrap = tf.concat(mse_draws, axis=0)
        fixed_mse_bootstrap = tf.concat(fixed_mse_draws, axis=0)
        result.update(
            {
                "joint_exact_mse": joint_mse,
                "joint_exact_fixed_mse": joint_fixed_mse,
                "exact_mse_adjacent_scaling": _adjacent_rows(
                    frozen_counts, joint_mse, mse_bootstrap
                ),
                "exact_fixed_mse_adjacent_scaling": _adjacent_rows(
                    frozen_counts, joint_fixed_mse, fixed_mse_bootstrap
                ),
            }
        )
    return result


__all__ = [
    "DTYPE",
    "sample_variance",
    "summarize_path_count_scaling",
]
