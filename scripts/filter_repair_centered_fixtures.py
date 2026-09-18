"""Fresh exact parents for isolated centered-density execution comparisons.

No historical fit is loaded. The parent amplitude is exactly one, so its
defensive normalizer is exactly 1+tau. Fixed residuals exercise heterogeneous
ranks without changing any scientific training or selection protocol.
"""

import math

FIXTURES = ("centered_child", "centered_solver")


def exact_parent(tf):
    from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
        LaneBLogNormalizerEstimate,
        LaneBT1Artifact,
        LaneBT1Settings,
        issue_lane_b_t1_identity,
        source_closure,
    )

    dtype = tf.float64
    settings = LaneBT1Settings(arm_id="filter_repair_exact_reference", rank=1, basis_order=2,
        basis_num_elems=1, learning_rate=.001, l1_weight=1e-6, l2_weight=0., batch_size=4,
        train_steps=1, expansion_factor=1., covariance_jitter=1e-5, quantile_fraction=.1,
        use_quantile_scale=False, tau=.05, gradient_clip_norm=10., cdf_grid_size=16,
        cdf_bisection_steps=8, cdf_max_working_bytes=2**24)
    cores = tuple(tf.ones([1, 3, 1], dtype) for _ in range(36))
    exact = tf.math.log(tf.constant(1.05, dtype))

    def estimate(role):
        return LaneBLogNormalizerEstimate(role=role, seed=0, sample_count=4,
            shift_constant=tf.zeros([], dtype), log_evidence=exact, log_shifted_normalizer=exact,
            log_standard_error=tf.zeros([], dtype), log_likelihood_sha256=tensor_sha256(tf.fill([4], exact)))

    values = {"settings": settings, "frame": SourceRouteCoordinateFrame(mu=tf.zeros([36], dtype),
        matrix=tf.eye(36, dtype=dtype), expansion_factor=1.), "cores": cores, "shift_constant": tf.zeros([], dtype),
        "calibration_estimate": estimate("exact_test_calibration"), "validation_estimate": estimate("exact_test_validation"),
        "frozen_reference_points": tf.fill([36, 4], tf.constant(.5, dtype)),
        "training_cloud_manifest": {"role": "exact_constant_reference_no_training"},
        "validation_cloud_manifest": {"role": "exact_constant_reference_no_training"}, "source_hashes": source_closure()}
    return LaneBT1Artifact(**values, identity=issue_lane_b_t1_identity(**values))


def fixture(tf, name, size, jit):
    del jit
    dtype = tf.float64
    if name == "centered_child":
        from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
            LaneBCenteredResidualChild,
        )
        from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
            tensor_sha256,
        )

        parent = exact_parent(tf)
        components = []
        rank = size + 1
        for component in range(3):
            cores = []
            for axis in range(36):
                shape = (1 if axis == 0 else rank, 3, 1 if axis == 35 else rank)
                values = [((1. if left == right == 0 else 0.)
                           + .005 * math.sin(component + axis + left * 7 + basis * 3 + right))
                          for left in range(shape[0]) for basis in range(3) for right in range(shape[2])]
                cores.append(tf.reshape(tf.constant(values, dtype), shape))
            components.append(tuple(cores))
        child = LaneBCenteredResidualChild(parent, tuple(components))
        points = tf.reshape(tf.linspace(tf.constant(-.3, dtype), .4, size * 4 * 36), [size * 4, 36])
        inputs = (tf.constant([.02, -.01, .03], dtype), points)

        def evaluate(theta, points):
            return (child.increment_and_score(theta), child.point_log_density_and_score(theta, points),
                    child.prefix_log_marginal_and_score(theta, points[:, :18]))

        return evaluate, inputs, {"dimension": 36, "residual_rank": rank, "query_rows": size * 4,
            "feature_count": 3, "prefix_dimension": 18, "tau": .05,
            "frozen_core_sha256": tuple(tensor_sha256(core) for core in tf.nest.flatten(child.components)),
            "boundary": "complete_centered_increment_point_prefix_values_and_analytical_scores",
            "classification": "existing_extension_or_invention", "canonical_admitted": False}
    if name == "centered_solver":
        from bayesfilter.highdim import (
            zhao_cui_austria_sir_parameter_density_training_tf as training,
        )

        dimension, maximum, interval = size * 4, 32, 3
        diagonal = tf.constant([1. + axis for axis in range(dimension)], dtype)
        rhs = tf.constant([.5 * math.cos(axis) for axis in range(dimension)], dtype)

        def callback(position):
            return (.5 * tf.reduce_sum(diagonal * position**2) - tf.reduce_sum(rhs * position),
                    diagonal * position - rhs)

        inputs = (tf.zeros([dimension], dtype), tf.constant(1e-10, dtype))
        if hasattr(training, "quadratic_cg_program"):
            program = training.quadratic_cg_program(callback, tf.TensorSpec([dimension], dtype), maximum, interval)

            def evaluate(position, tolerance):
                result = program.python_function(position, tolerance)
                return (result["position"], result["converged"], result["failed"], result["num_iterations"],
                    result["initial_residual_norm"], result["residual_norm"], result["relative_residual_norm"],
                    result["minimum_curvature"], result["trace_keep"],
                    tf.where(result["trace_keep"], result["trace_norms"], 0.),
                    tf.where(result["trace_keep"], result["trace_relative"], 0.))
        else:
            def evaluate(position, tolerance):
                result = training.solve_quadratic_value_gradient_with_conjugate_gradient(callback,
                    initial_position=position, tolerance=float(tolerance), max_iterations=maximum, trace_interval=interval)
                indices = tf.constant([[row[0]] for row in result.trace])
                keep = tf.scatter_nd(indices, tf.ones([len(result.trace)], tf.bool), [maximum + 1])
                norms = tf.scatter_nd(indices, tf.stack([row[1] for row in result.trace]), [maximum + 1])
                relative = tf.scatter_nd(indices, tf.stack([row[2] for row in result.trace]), [maximum + 1])
                return (result.position, tf.constant(result.converged), tf.constant(result.failed), tf.constant(result.num_iterations),
                    result.initial_residual_norm, result.residual_norm, result.relative_residual_norm,
                    result.minimum_curvature, keep, norms, relative)

        return evaluate, inputs, {"dimension": dimension, "maximum_iterations": maximum, "trace_interval": interval,
            "diagonal": [1. + axis for axis in range(dimension)], "rhs": [.5 * math.cos(axis) for axis in range(dimension)],
            "boundary": "complete_quadratic_callback_solver_status_and_trace", "canonical_admitted": False}
    raise ValueError(f"Unregistered centered fixture: {name}")
