"""Crossed bundle/path diagnostics for classifier-score variance reduction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import tensorflow as tf


DTYPE = tf.float64
ARM_NAMES = (
    "independent_n2048",
    "crn_n2048",
    "independent_n8192",
    "crn_n8192",
)
EFFECT_COMPARISONS = (
    ("crn_at_n2048", 1, 0),
    ("crn_at_n8192", 3, 2),
    ("more_paths_independent", 2, 0),
    ("more_paths_crn", 3, 1),
    ("combined", 3, 0),
)


def _crossed(values: tf.Tensor) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if tensor.shape.rank != 4:
        raise ValueError("outputs must have shape [arm,bundle,path,cell]")
    if any(dimension is None for dimension in tensor.shape):
        raise ValueError("crossed output shape must be static")
    if int(tensor.shape[0]) != len(ARM_NAMES):
        raise ValueError("crossed outputs must contain the four frozen arms")
    if int(tensor.shape[1]) < 2 or int(tensor.shape[2]) < 2:
        raise ValueError("at least two bundles and two paths are required")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError("crossed outputs contain non-finite values")
    return tensor


def _fixed(values: tf.Tensor, bundles: int, cells: int) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if tensor.shape != (len(ARM_NAMES), bundles, cells):
        raise ValueError("fixed outputs must have shape [arm,bundle,cell]")
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


def _percentile(values: tf.Tensor, probability: float) -> tf.Tensor:
    ordered = tf.sort(tf.cast(tf.reshape(values, [-1]), DTYPE))
    index = max(0, min(int(ordered.shape[0]) - 1, math.ceil(float(probability) * int(ordered.shape[0])) - 1))
    return ordered[index]


def _interval_rows(point: tf.Tensor, draws: tf.Tensor) -> list[dict[str, float]]:
    rows = []
    for arm_index, arm_name in enumerate(ARM_NAMES):
        arm_draws = draws[:, arm_index]
        rows.append(
            {
                "arm": arm_name,
                "point": float(point[arm_index].numpy()),
                "lower_95": float(_percentile(arm_draws, 0.025).numpy()),
                "upper_95": float(_percentile(arm_draws, 0.975).numpy()),
            }
        )
    return rows


def _effect_interval_rows(point: tf.Tensor, draws: tf.Tensor) -> list[dict[str, object]]:
    rows = []
    for name, numerator, denominator in EFFECT_COMPARISONS:
        floor = tf.maximum(
            tf.reduce_max(tf.abs(draws)) * tf.constant(1.0e-15, DTYPE),
            tf.constant(1.0e-18, DTYPE),
        )
        point_ratio = point[numerator] / tf.maximum(point[denominator], floor)
        draw_ratio = draws[:, numerator] / tf.maximum(draws[:, denominator], floor)
        rows.append(
            {
                "effect": name,
                "numerator_arm": ARM_NAMES[numerator],
                "denominator_arm": ARM_NAMES[denominator],
                "point": float(point_ratio.numpy()),
                "lower_95": float(_percentile(draw_ratio, 0.025).numpy()),
                "upper_95": float(_percentile(draw_ratio, 0.975).numpy()),
            }
        )
    return rows


def _baseline_ratio(values: tf.Tensor) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    floor = tf.maximum(
        tf.reduce_max(tf.abs(tensor)) * tf.constant(1.0e-15, DTYPE),
        tf.constant(1.0e-18, DTYPE),
    )
    return tensor / tf.maximum(tensor[..., 0:1], floor)


def summarize_crossed_outputs(
    outputs: tf.Tensor,
    fixed_outputs: tf.Tensor,
    *,
    exact_scores: tf.Tensor | None = None,
    exact_fixed_score: tf.Tensor | None = None,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: tuple[int, int] = (1601, 2903),
    chunk_size: int = 100,
) -> dict[str, object]:
    """Return paired variance ratios and optional exact-score MSE ratios."""

    crossed = _crossed(outputs)
    arms, bundles, paths, cells = (int(value) for value in crossed.shape)
    fixed = _fixed(fixed_outputs, bundles, cells)
    if int(bootstrap_replicates) < 100:
        raise ValueError("at least 100 bootstrap replicates are required")

    bundle_variance_by_path_cell = sample_variance(crossed, axis=1)
    bundle_variance_by_cell = tf.reduce_mean(bundle_variance_by_path_cell, axis=1)
    fixed_bundle_variance_by_cell = sample_variance(fixed, axis=1)
    baseline_bundle_mean = tf.reduce_mean(crossed[0], axis=0)
    natural_path_variance = sample_variance(baseline_bundle_mean, axis=0)
    numerical_floor = tf.maximum(
        tf.reduce_max(natural_path_variance) * tf.constant(1.0e-12, DTYPE),
        tf.constant(1.0e-18, DTYPE),
    )
    scale_variance = tf.maximum(natural_path_variance, numerical_floor)
    joint_audit_variance = tf.reduce_mean(bundle_variance_by_cell / scale_variance[None, :], axis=1)
    joint_fixed_variance = tf.reduce_mean(fixed_bundle_variance_by_cell / scale_variance[None, :], axis=1)
    audit_ratio = joint_audit_variance / joint_audit_variance[0]
    fixed_ratio = joint_fixed_variance / joint_fixed_variance[0]

    exact = None
    exact_fixed = None
    mse_by_cell = None
    bias_by_cell = None
    rmse_by_cell = None
    fixed_mse_by_cell = None
    fixed_bias_by_cell = None
    fixed_rmse_by_cell = None
    joint_mse = None
    mse_ratio = None
    joint_fixed_mse = None
    fixed_mse_ratio = None
    if exact_scores is not None:
        exact = tf.cast(tf.convert_to_tensor(exact_scores), DTYPE)
        exact_fixed = tf.cast(tf.convert_to_tensor(exact_fixed_score), DTYPE)
        if exact.shape != (paths, cells) or exact_fixed.shape != (cells,):
            raise ValueError("exact scores do not match audit/fixed dimensions")
        exact_errors = crossed - exact[None, None, :, :]
        fixed_exact_errors = fixed - exact_fixed[None, None, :]
        bias_by_cell = tf.reduce_mean(exact_errors, axis=(1, 2))
        mse_by_cell = tf.reduce_mean(tf.square(exact_errors), axis=(1, 2))
        rmse_by_cell = tf.sqrt(mse_by_cell)
        fixed_bias_by_cell = tf.reduce_mean(fixed_exact_errors, axis=1)
        fixed_mse_by_cell = tf.reduce_mean(tf.square(fixed_exact_errors), axis=1)
        fixed_rmse_by_cell = tf.sqrt(fixed_mse_by_cell)
        joint_mse = tf.reduce_mean(mse_by_cell / scale_variance[None, :], axis=1)
        mse_ratio = joint_mse / joint_mse[0]
        joint_fixed_mse = tf.reduce_mean(fixed_mse_by_cell / scale_variance[None, :], axis=1)
        fixed_mse_ratio = joint_fixed_mse / joint_fixed_mse[0]

    audit_draws: list[tf.Tensor] = []
    fixed_draws: list[tf.Tensor] = []
    mse_draws: list[tf.Tensor] = []
    fixed_mse_draws: list[tf.Tensor] = []
    completed = 0
    while completed < int(bootstrap_replicates):
        count = min(int(chunk_size), int(bootstrap_replicates) - completed)
        bundle_indices = tf.random.stateless_uniform(
            [count, bundles],
            [int(bootstrap_seed[0]), int(bootstrap_seed[1]) + completed],
            minval=0,
            maxval=bundles,
            dtype=tf.int32,
        )
        path_indices = tf.random.stateless_uniform(
            [count, paths],
            [int(bootstrap_seed[0]) + 1, int(bootstrap_seed[1]) + completed],
            minval=0,
            maxval=paths,
            dtype=tf.int32,
        )
        selected_bundles = tf.transpose(tf.gather(crossed, bundle_indices, axis=1), [1, 0, 2, 3, 4])
        selected = tf.gather(selected_bundles, path_indices, axis=3, batch_dims=1)
        variance_by_path_cell = sample_variance(selected, axis=2)
        variance_by_cell = tf.reduce_mean(variance_by_path_cell, axis=2)
        joint = tf.reduce_mean(variance_by_cell / scale_variance[None, None, :], axis=2)
        audit_draws.append(_baseline_ratio(joint))

        selected_fixed = tf.transpose(tf.gather(fixed, bundle_indices, axis=1), [1, 0, 2, 3])
        fixed_variance = sample_variance(selected_fixed, axis=2)
        fixed_joint = tf.reduce_mean(fixed_variance / scale_variance[None, None, :], axis=2)
        fixed_draws.append(_baseline_ratio(fixed_joint))

        if exact is not None:
            selected_exact = tf.gather(exact, path_indices, axis=0, batch_dims=0)
            errors = selected - selected_exact[:, None, None, :, :]
            cell_mse = tf.reduce_mean(tf.square(errors), axis=(2, 3))
            mse_joint = tf.reduce_mean(cell_mse / scale_variance[None, None, :], axis=2)
            mse_draws.append(_baseline_ratio(mse_joint))
            fixed_errors = selected_fixed - exact_fixed[None, None, :]
            fixed_cell_mse = tf.reduce_mean(tf.square(fixed_errors), axis=2)
            fixed_mse_joint = tf.reduce_mean(fixed_cell_mse / scale_variance[None, None, :], axis=2)
            fixed_mse_draws.append(_baseline_ratio(fixed_mse_joint))
        completed += count

    audit_bootstrap = tf.concat(audit_draws, axis=0)
    fixed_bootstrap = tf.concat(fixed_draws, axis=0)
    result: dict[str, object] = {
        "arm_names": ARM_NAMES,
        "bundle_count": bundles,
        "audit_path_count": paths,
        "cell_count": cells,
        "bundle_variance_by_cell": bundle_variance_by_cell,
        "fixed_bundle_variance_by_cell": fixed_bundle_variance_by_cell,
        "natural_path_variance_by_cell": natural_path_variance,
        "scale_variance_by_cell": scale_variance,
        "joint_audit_bundle_variance": joint_audit_variance,
        "joint_fixed_bundle_variance": joint_fixed_variance,
        "audit_variance_ratio": _interval_rows(audit_ratio, audit_bootstrap),
        "fixed_variance_ratio": _interval_rows(fixed_ratio, fixed_bootstrap),
        "audit_effect_ratio": _effect_interval_rows(joint_audit_variance, audit_bootstrap),
        "fixed_effect_ratio": _effect_interval_rows(joint_fixed_variance, fixed_bootstrap),
        "bootstrap_replicates": int(bootstrap_replicates),
    }
    if exact is not None and mse_ratio is not None and fixed_mse_ratio is not None:
        mse_bootstrap = tf.concat(mse_draws, axis=0)
        fixed_mse_bootstrap = tf.concat(fixed_mse_draws, axis=0)
        result.update(
            {
                "exact_mse_by_cell": mse_by_cell,
                "exact_bias_by_cell": bias_by_cell,
                "exact_rmse_by_cell": rmse_by_cell,
                "exact_fixed_mse_by_cell": fixed_mse_by_cell,
                "exact_fixed_bias_by_cell": fixed_bias_by_cell,
                "exact_fixed_rmse_by_cell": fixed_rmse_by_cell,
                "joint_exact_mse": joint_mse,
                "joint_exact_fixed_mse": joint_fixed_mse,
                "exact_mse_ratio": _interval_rows(mse_ratio, mse_bootstrap),
                "exact_fixed_mse_ratio": _interval_rows(fixed_mse_ratio, fixed_mse_bootstrap),
                "exact_mse_effect_ratio": _effect_interval_rows(joint_mse, mse_bootstrap),
                "exact_fixed_mse_effect_ratio": _effect_interval_rows(joint_fixed_mse, fixed_mse_bootstrap),
            }
        )
    return result


def classify_combined_arm(summary: dict[str, object]) -> dict[str, object]:
    audit = summary["audit_variance_ratio"][3]
    fixed = summary["fixed_variance_ratio"][3]
    variance_supported = audit["upper_95"] < 1.0 and fixed["upper_95"] < 1.0
    accuracy_harmed = False
    if "exact_mse_ratio" in summary:
        accuracy_harmed = (
            summary["exact_mse_ratio"][3]["lower_95"] > 1.0
            or summary["exact_fixed_mse_ratio"][3]["lower_95"] > 1.0
        )
    if accuracy_harmed:
        status = "accuracy_harmed"
    elif variance_supported:
        status = "variance_reduction_supported"
    elif audit["point"] < 1.0 and fixed["point"] < 1.0:
        status = "descriptively_favorable"
    else:
        status = "no_supported_reduction"
    return {
        "arm": ARM_NAMES[3],
        "status": status,
        "variance_reduction_supported": variance_supported,
        "accuracy_harmed": accuracy_harmed,
    }


__all__ = [
    "ARM_NAMES",
    "DTYPE",
    "EFFECT_COMPARISONS",
    "classify_combined_arm",
    "sample_variance",
    "summarize_crossed_outputs",
]
