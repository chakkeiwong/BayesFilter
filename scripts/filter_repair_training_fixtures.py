"""Matched finite-loss and fitted-initializer execution diagnostics.

These fresh exact parents are not trained historical artifacts. Every arm
uses the same parameter and target tensors, finite loss and regularizers.
"""

from dataclasses import fields

from filter_repair_centered_fixtures import exact_parent

FIXTURES = ("core_affine_score", "centered_additive_initializer", "centered_pair_initializer")


def fixture(tf, name, size, jit):
    del jit
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_parameter_density_training_tf as training,
    )
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256

    parent = exact_parent(tf)
    dtype, rows = tf.float64, 4 * size
    points = tf.reshape(tf.linspace(tf.constant(-.3, dtype), .4, rows * 36), [rows, 36])
    targets = tf.reshape(tf.sin(tf.cast(tf.range(rows * 3), dtype)), [rows, 3])
    weights = tf.linspace(tf.constant(-.2, dtype), .3, rows)
    prefix_errors = tf.ones([2, 3], dtype)
    dimensions = {"dimension": 36, "rows": rows, "parent_rank": 1, "basis_width": 3,
        "prefix_dimension": 18, "prefix_rows": 2, "parameter_count": 3,
        "parent_core_sha256": tuple(tensor_sha256(core) for core in parent.cores),
        "classification": "existing_extension_or_invention", "canonical_admitted": False}
    if name == "core_affine_score":
        position = .01 * tf.sin(tf.cast(tf.range(3 * 36 * 3), dtype))
        basis = training.centered_lane_b_product_basis(order=2, num_elems=1)

        def evaluate(position, points, targets, weights, prefix_errors):
            with tf.GradientTape() as tape:
                tape.watch(position)
                result = training.core_affine_origin_total_score_loss_arrays(
                    parent=parent, position=position, point_local_points=points,
                    point_target_score=targets, point_importance_log_weight=weights,
                    global_target_score=targets[0], global_score_standard_error=prefix_errors[0],
                    prefix_local_points=points[:2, :18], prefix_target_score=targets[:2],
                    prefix_score_standard_error=prefix_errors, point_weight=1.,
                    global_weight=.2, prefix_weight=.1, l2_weight=.001, basis=basis)
            return result, tape.gradient(result[0], position)

        dimensions.update(boundary="complete_core_affine_point_global_prefix_loss_and_gradient",
                          point_weight=1., global_weight=.2, prefix_weight=.1, l2_weight=.001)
        return evaluate, (position, points, targets, weights, prefix_errors), dimensions

    if name in ("centered_additive_initializer", "centered_pair_initializer"):
        initialize = (training.target_informed_additive_score_initialization
                      if name == "centered_additive_initializer"
                      else training.target_informed_within_region_pair_score_initialization)

        def evaluate(points, targets, weights, prefix_errors):
            result = initialize(parent=parent, local_points=points,
                target_complete_data_score=targets, importance_log_weight=weights,
                ridge_fraction=.1, global_score_weight=1., prefix_local_points=points[:2, :18],
                prefix_target_score=targets[:2], prefix_score_standard_error=prefix_errors, prefix_weight=.2)
            return tuple(tf.convert_to_tensor(value, dtype) for value in tf.nest.flatten(
                tuple(getattr(result, field.name) for field in fields(result))))

        dimensions.update(boundary="complete_fitted_initializer_all_numerical_fields",
                          ridge_fraction=.1, global_score_weight=1., prefix_weight=.2)
        return evaluate, (points, targets, weights, prefix_errors), dimensions
    raise ValueError(f"Unregistered training fixture: {name}")
