"""Dense initialization glue for the shared BayesFilter estimation procedure.

The center locator, exact replay, curvature estimation and stability selection
belong to canonical BayesFilter. This module prepares independent diagnostic
clouds and converts their scores under theta = center + diag(scale) z. A usable
local covariance initializes guide marginal scales; it is not posterior or HMC
evidence. Authority: docs/plans/dz5_equity_phase_repair_20260909.md.
"""

from __future__ import annotations

import math
from typing import Any

from bayesfilter_estimation import file_hash, json_ready


def initialize_dense_local(model: Any, recipe: Any, context: Any) -> dict[str, Any]:
    """Locate from model-declared starts, then fit fresh fixed-center geometry."""
    import numpy as np
    import tensorflow as tf
    from bayesfilter.inference.batched_local_center import (
        BatchedLocalCenterConfig,
        locate_batched_local_center,
    )
    from bayesfilter.inference.fixed_center_curvature import fit_fixed_center_curvature

    from bayesfilter_estimation_runtime import (
        diagnostic_json_ready,
        initializer_configurations,
    )

    configuration, _movement, thresholds = initializer_configurations(recipe)
    target = model.training_target
    dimension = target.parameter_dim
    scale = tf.convert_to_tensor(model.coordinate_scale, tf.float64)
    center = tf.convert_to_tensor(model.initial_position, tf.float64)
    default_rows = max(2 * dimension, 3 * dimension - 1)
    training_rows = configuration.training_rows_per_replicate or default_rows
    selection_rows = configuration.selection_rows_per_replicate or default_rows
    audit_rows = configuration.audit_rows or 2 * dimension
    partition_rows = ([training_rows] * configuration.replicate_count
                      + [selection_rows] * configuration.replicate_count + [audit_rows])
    locator = configuration.locator_config
    locator_config = BatchedLocalCenterConfig(
        box_radius=configuration.locator_box_radius, trust_refinement_rounds=1,
        num_correction_pairs=locator.num_correction_pairs, max_iterations=locator.max_iterations,
        max_line_search_iterations=locator.max_line_search_iterations,
        gradient_tolerance=locator.gradient_tolerance,
        max_optimizer_callback_batches_per_round=locator.max_objective_evaluations,
        jit_compile=False,
    )
    planned_rows = configuration.max_curvature_attempts * (
        locator_config.maximum_physical_rows_multiplier + sum(partition_rows)
    )
    if planned_rows > configuration.max_exact_evaluations:
        raise ValueError("dense initializer planned exact rows exceed max_exact_evaluations")
    exact_rows = 0
    archives = {}
    attempts = []
    result = {"passed": False, "method": "dense_local", "attempts": attempts,
              "initial_output_shift": None, "initial_output_scale_log": None,
              "planned_exact_row_ceiling": planned_rows,
              "nonclaims": ["local guide initializer only", "not posterior covariance",
                            "not HMC or retained-sampling qualification"]}

    def status(positions):
        telemetry = target.target_status_telemetry(positions)
        return tf.cast(telemetry["valid_pre_regularized_score"], tf.bool) & (telemetry["status_code"] == 0)

    def combined(positions):
        atomic = getattr(target, "batch_value_score_and_validity", None)
        if atomic is not None:
            return atomic(positions)
        value, score = target.batch_value_and_score(positions)
        return value, score, status(positions)

    def finish(reason):
        result.update(status=reason, exact_evaluation_count=exact_rows,
                      target_evaluation_accounting="logical value/score rows; status callbacks may repeat physical work")
        if archives:
            path = context.root / "initializer_evaluations.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **archives)
            result["artifacts"] = {str(path): file_hash(path)}
        return diagnostic_json_ready(result)

    for attempt in range(configuration.max_curvature_attempts):
        with context.boundary(f"dense_center_{attempt}"):
            location = locate_batched_local_center(combined, center[None, :], scale, config=locator_config)
        exact_rows += int(location.physical_target_rows.numpy())
        record = {"attempt": attempt, "locator": location.payload()}
        attempts.append(record)
        if not bool(location.accepted.numpy()):
            return finish(f"locator_{location.status}")
        center = location.center
        center_value = float(location.center_value.numpy())
        center_score = location.center_score
        score_norm = float(tf.linalg.norm(center_score * scale).numpy())
        record["scaled_center_score_l2"] = score_norm
        if not math.isfinite(score_norm) or score_norm > recipe.dense_center_score_max:
            return finish("dense_center_score_above_cap")
        partitions = []
        candidate_center = center
        candidate_value = center_value
        for partition_index, rows in enumerate(partition_rows):
            seed = (configuration.seed[0], configuration.seed[1] + 1000 * attempt + partition_index)
            radius_seed = (seed[0], seed[1] + 100)
            directions = tf.random.stateless_normal((rows, dimension), seed, dtype=tf.float64)
            norms = tf.linalg.norm(directions, axis=1, keepdims=True)
            radii = configuration.curvature_radius * tf.random.stateless_uniform(
                (rows, 1), radius_seed, dtype=tf.float64
            ) ** (1.0 / dimension)
            offsets = directions / norms * radii
            positions = center[None, :] + offsets * scale
            with context.boundary(f"dense_cloud_{attempt}_{partition_index}"):
                values, scores, valid = combined(positions)
            exact_rows += rows
            prefix = f"attempt_{attempt}_partition_{partition_index}"
            for name, tensor in (("positions", positions), ("values", values), ("scores", scores), ("valid", valid)):
                archives[f"{prefix}_{name}"] = tensor.numpy()
            finite = tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(scores))
            if not bool((tf.reduce_all(valid) & finite).numpy()):
                return finish("dense_curvature_cloud_invalid")
            best = int(tf.argmax(values).numpy())
            if float(values[best].numpy()) > candidate_value:
                candidate_value = float(values[best].numpy())
                candidate_center = positions[best]
            partitions.append((offsets.numpy(), (scores * scale).numpy()))
        record.update(partition_rows=partition_rows, objective_improvement=candidate_value - center_value)
        if candidate_value > center_value:
            center = candidate_center
            record["curvature_center_moved"] = True
            continue
        count = configuration.replicate_count
        with context.boundary(f"dense_curvature_fit_{attempt}"):
            curvature = fit_fixed_center_curvature(
                center.numpy(), (center_score * scale).numpy(),
                np.stack([row[0] for row in partitions[:count]]),
                np.stack([row[1] for row in partitions[:count]]),
                np.stack([row[0] for row in partitions[count:2 * count]]),
                np.stack([row[1] for row in partitions[count:2 * count]]),
                *partitions[-1], thresholds=thresholds, factor_max=configuration.factor_max,
                dense_eigenvalue_floor=configuration.dense_eigenvalue_floor,
                max_condition_number=configuration.max_condition_number,
                shrinkage_weights=configuration.shrinkage_weights,
                structured_target_family=configuration.structured_target_family,
                lineage={"role": "fresh_dense_initializer", "seed": configuration.seed,
                         "attempt": attempt, "truth_blind_prior_start": True},
            )
        record["curvature"] = curvature.payload()
        if not curvature.accepted or curvature.selected_covariance_z is None:
            return finish(f"dense_{curvature.status}")
        covariance = tf.convert_to_tensor(curvature.selected_covariance_z, tf.float64)
        marginal_scale = scale * tf.sqrt(tf.linalg.diag_part(covariance))
        if not bool(tf.reduce_all(tf.math.is_finite(marginal_scale) & (marginal_scale > 0)).numpy()):
            return finish("dense_initializer_scale_invalid")
        result.update(passed=True, initial_output_shift=json_ready(center),
                      initial_output_scale_log=json_ready(tf.math.log(marginal_scale)))
        return finish("usable_dense_local_initializer")
    return finish("dense_curvature_not_centered_within_attempt_budget")
