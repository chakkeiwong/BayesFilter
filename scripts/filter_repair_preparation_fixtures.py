"""Matched preparation diagnostics using frozen tensors in isolated source arms.

No fixture imports a second source tree or changes runtime functions. Baseline
host-only APIs retain their real tracing failures; eager results are separate
reference measurements. These fixtures do not grant scientific admission.
"""

import inspect

FIXTURES = ("source_recenter", "gamma_preparation", "student_proposal",
            "ukf_initializer", "moment_teacher", "austria_preparation",
            "exact_incumbent", "sequential_score_fit", "mass_precision", "mass_structured", "block_score_geometry")


def fixture(tf, name, size, jit, *, public_boundary=False):
    dtype = tf.float64
    if name == "block_score_geometry":
        from bayesfilter.inference import block_score_geometry as geometry

        dimension, replicates = 4 * size, 2 * size
        training_rows, selection_rows, audit_rows = 12 * size, 9 * size, 11 * size
        blocks = tuple(geometry.ScoreGeometryBlock(str(index), 2 * index, 2 * index + 2)
            for index in range(dimension // 2))
        config = geometry.BlockScoreGeometryConfig(ridge=1e-12, principal_subspace_rank=2)
        controls = (config.ridge, config.max_condition_number, config.selection_relative_rmse_cap,
            config.audit_relative_rmse_cap, config.unexplained_response_fraction_cap,
            config.generalized_eigenvalue_spread_cap, config.trace_normalized_frobenius_cap,
            config.trace_normalized_operator_cap, config.principal_angle_degrees_cap)
        coordinate = tf.range(dimension)
        precision = tf.linalg.diag(tf.linspace(tf.constant(1., dtype), 4., dimension)) + .15 * tf.cast(
            coordinate[:, None] // 2 == coordinate[None, :] // 2, dtype)
        center = tf.linspace(tf.constant(-.2, dtype), .3, dimension)

        def offsets(shape, count, phase):
            indices = tf.cast(tf.range(count), dtype)
            return tf.reshape(.2 * tf.sin(.17 * indices ** 2 + phase), shape)

        training = offsets([replicates, training_rows, dimension], replicates * training_rows * dimension, .1)
        selection = offsets([replicates, selection_rows, dimension], replicates * selection_rows * dimension, .3)
        audit = offsets([audit_rows, dimension], audit_rows * dimension, .7)
        scale = tf.linspace(tf.constant(.5, dtype), 2., dimension)
        inputs = (center, training, center - training @ precision, selection, center - selection @ precision,
            audit, center - audit @ precision, tf.constant(controls, dtype),
            tf.constant(config.principal_subspace_rank, tf.int32), scale)
        native = hasattr(geometry, "native")
        rank = min(config.principal_subspace_rank, dimension - 1)
        block_fields = ("design_rank", "offset_rank", "raw_minimum_eigenvalue", "raw_maximum_eigenvalue",
            "raw_nonpositive_eigenvalue_count", "raw_condition_number")
        selection_fields = ("selection_relative_rmse", "selection_unexplained_response_fraction")
        summary_fields = ("consensus_selection_relative_rmse", "consensus_selection_unexplained_response_fraction",
            "audit_relative_rmse", "audit_unexplained_response_fraction", "precision_covariance_identity_max_abs")
        check_fields = ("generalized_eigenvalue_spread", "trace_normalized_frobenius", "trace_normalized_operator",
            "principal_angle_degrees")
        if native and not public_boundary:
            program = geometry.native.fit_program(dimension, replicates, training_rows, selection_rows,
                audit_rows, tuple((block.start, block.stop) for block in blocks), jit_compile=jit).python_function
            physical = geometry.native.position_program(dimension, jit_compile=jit).python_function

            def evaluate(*arguments):
                result = program(*arguments[:-1])
                pairs = result["pairs"]
                reported_pairs = tf.concat([pairs[:, :2 * dimension + rank], pairs[:, 3 * dimension:]], 1)
                return (result["status"], result["precision"], result["covariance"], result["blocks"],
                    result["selection"], reported_pairs, result["checks"], result["summary"],
                    *physical(result["precision"], result["covariance"], arguments[-1])[:3])
        else:
            def evaluate(center, training, training_scores, selection, selection_scores,
                         audit, audit_scores, _controls, _rank, scale):
                result = geometry.fit_block_diagonal_score_geometry(center_score_z=center,
                    training_offsets_z=training, training_scores_z=training_scores,
                    selection_offsets_z=selection, selection_scores_z=selection_scores,
                    audit_offsets_z=audit, audit_scores_z=audit_scores, blocks=blocks, config=config)
                if not result.accepted:
                    raise ValueError("Frozen block-score comparison rejected: " + result.status)
                report = result.diagnostics
                block_reports = [[[block[field] for field in block_fields] + [0.]
                    for block in replicate["blocks"]] for replicate in report["replicates"]]
                selections = [[replicate[field] for field in selection_fields] for replicate in report["replicates"]]
                pairs, checks = [], []
                for pair in report["stability"]["comparisons"]:
                    metric = pair["metrics"]
                    generalized = metric["generalized_eigenvalues"]
                    pairs.append([*metric["left_raw_eigenvalues"], *metric["right_raw_eigenvalues"],
                        *metric["principal_angles_degrees"], metric["positive_subspace_rank"],
                        generalized["minimum"], generalized["maximum"], generalized["spread"],
                        metric["trace_normalized_frobenius"], metric["trace_normalized_operator"],
                        metric["maximum_principal_angle_degrees"], metric["left_nonpositive_count"],
                        metric["right_nonpositive_count"], 1.])
                    checks.append([pair["checks"][field] for field in check_fields])
                physical = result.position_geometry(scale)
                return (tf.constant(0, tf.int32), tf.convert_to_tensor(result.precision_z, dtype),
                    tf.convert_to_tensor(result.covariance_z, dtype), tf.constant(block_reports, dtype),
                    tf.constant(selections, dtype), tf.constant(pairs, dtype), tf.constant(checks, tf.bool),
                    tf.constant([report[field] for field in summary_fields], dtype),
                    *(tf.convert_to_tensor(physical[field], dtype) for field in ("precision", "covariance", "factor")))

        evaluate.timing_scope = ("complete_tensor_block_fit_stability_qualification_and_scaling"
            if native and not public_boundary else "complete_public_block_fit_records_and_scaling")
        evaluate.execution_backend = "tensorflow" if native else "legacy_numpy_geometry_reporting_diagnostic_only"
        return evaluate, inputs, {"dimension": dimension, "replicates": replicates, "block_width": 2,
            "blocks": dimension // 2, "training_rows": training_rows, "selection_rows": selection_rows,
            "audit_rows": audit_rows, "boundary": "complete_block_score_geometry_and_position_scaling",
            "random_inputs": False}

    if name in ("mass_precision", "mass_structured"):
        from bayesfilter.inference import mass_matrix as mass

        dimension = 6 * size
        matrix = tf.linalg.diag(tf.linspace(tf.constant(.2, dtype), 3., dimension)) + .03
        inputs = (matrix, tf.constant(.1, dtype), tf.constant(.25, dtype), tf.constant(8., dtype))
        blocks = tuple({"name": str(i), "start": 3 * i, "stop": 3 * i + 3} for i in range(2 * size))
        native = hasattr(mass, "native")
        precision_fields = ("effective_eigenvalue_floor", "raw_min_eigenvalue", "raw_max_eigenvalue",
            "regularized_min_eigenvalue", "regularized_max_eigenvalue", "raw_nonpositive_eigenvalue_count",
            "clipped_eigenvalue_count", "input_asymmetry_max_abs")
        structured_fields = ("raw_min_block_eigenvalue", "regularized_min_block_eigenvalue",
            "regularized_max_block_eigenvalue")

        if native and not public_boundary:
            if name == "mass_precision":
                program = mass.native.precision_program(dimension, dense=True, jit_compile=jit).python_function
            else:
                program = mass.native.structured_program(dimension,
                    tuple((block["start"], block["stop"]) for block in blocks), jit_compile=jit).inline_function
            summary = mass.native.summary_program(dimension, jit_compile=jit).python_function

            def evaluate(matrix, weight, floor, cap):
                if name == "mass_precision":
                    precision, covariance, diagnostics, _flags, _valid = program(matrix, tf.constant(1e-9, dtype), floor, cap)
                    return precision, covariance, diagnostics, *summary(precision)[:2], *summary(covariance)[:2]
                covariance, diagnostics, _valid = program(matrix, weight, floor, cap)
                return covariance, diagnostics, *summary(covariance)[:2]
        else:
            def summary(matrix_summary):
                return (tf.constant(matrix_summary["eigenvalues"], dtype),
                    tf.constant([matrix_summary[field] for field in ("min", "max", "condition_number")], dtype))

            def evaluate(matrix, weight, floor, cap):
                if name == "mass_precision":
                    result = mass.covariance_from_precision(matrix, source="comparison", jitter=1e-9,
                        eigenvalue_floor=.25, max_condition_number=8.)
                    diagnostics = tf.constant([result.regularization_report[field] for field in precision_fields], dtype)
                    return (result.regularized_precision, result.covariance, diagnostics,
                        *summary(result.precision_eigen_summary), *summary(result.covariance_eigen_summary))
                result = mass.structured_covariance_from_empirical(matrix, blocks=blocks,
                    shrinkage=.1, eigenvalue_floor=.25, max_condition_number=8.)
                diagnostics = tf.constant([result.regularization_report[field] for field in structured_fields], dtype)
                return result.covariance, diagnostics, *summary(result.covariance_eigen_summary)

        evaluate.timing_scope = ("complete_tensor_mass_and_spectral_summaries" if native and not public_boundary
            else "complete_public_mass_and_records")
        evaluate.execution_backend = "tensorflow" if native else "legacy_tensorflow_eager_diagnostic_only"
        return evaluate, inputs, {"dimension": dimension, "blocks": 2 * size,
            "boundary": "complete_mass_preparation_and_spectral_reports", "random_inputs": False}

    if name == "exact_incumbent":
        from bayesfilter.inference import _exact_incumbent as selector

        count, dimension = 32 * size, 3
        positions = tf.reshape(tf.sin(tf.cast(tf.range(count * dimension), dtype)), [count, dimension])
        values = -tf.reduce_sum(positions ** 2, axis=1)
        inputs = (positions, -positions, values, tf.range(count) % 3 != 0)
        native = hasattr(selector, "_incumbent_selection")

        def evaluate(positions, scores, values, flags):
            if native and not public_boundary:
                _mask, index = selector._incumbent_selection.python_function(
                    tf.reshape(positions, [-1]), tf.reshape(scores, [-1]), values,
                    flags, tf.range(1, count + 1) * dimension)
                return index, values[index], positions[index], scores[index]
            records = selector.candidates_from_rows(positions, values, scores,
                start_index=0, source_role="comparison", eligibility=flags)
            winner = selector.select_exact_incumbent(records)
            if winner is None:
                raise ValueError("frozen incumbent fixture has no eligible candidate")
            return (tf.constant(winner.evaluation_index, tf.int32), tf.constant(winner.value, dtype),
                tf.convert_to_tensor(winner.position, dtype), tf.convert_to_tensor(winner.score, dtype))

        evaluate.timing_scope = ("complete_tensor_selection" if native and not public_boundary
            else "complete_public_record_creation_and_selection")
        evaluate.execution_backend = "tensorflow" if native else "legacy_numpy_selection_diagnostic_only"
        return evaluate, inputs, {"records": count, "dimension": dimension,
            "boundary": "complete_exact_finite_incumbent_selection", "source_role": "fresh_execution_comparison"}

    if name == "sequential_score_fit":
        from bayesfilter.inference import sequential_map_covariance as sequential

        count, dimension = 16 * size, 2 * size
        config = sequential.SequentialMapCovarianceConfig()
        precision = tf.linalg.diag(tf.linspace(tf.constant(1.5, dtype), 3., dimension)) + .1

        def scalar(point):
            score = -tf.linalg.matvec(precision, point)
            return .5 * tf.reduce_sum(point * score), score

        def batch(points):
            scores = -tf.einsum("ij,bj->bi", precision, points)
            return .5 * tf.reduce_sum(points * scores, 1), scores

        center = tf.linspace(tf.constant(-.13, dtype), .22, dimension)
        inputs = (center, scalar(center)[1], tf.linspace(tf.constant(.7, dtype), 1.2, dimension),
            tf.constant(.3, dtype), tf.constant([2026, 919], tf.int32), tf.constant(config.ridge, dtype),
            tf.constant(config.eigenvalue_floor, dtype), tf.constant(config.max_condition_number, dtype),
            tf.constant(config.score_holdout_relative_rmse, dtype))
        native = hasattr(sequential, "score_fit_program")
        fields = ("rank", "train_score_rmse", "holdout_score_relative_rmse", "raw_eigenvalues",
            "projected_eigenvalues", "projection_relative_frobenius", "projected_precision_z")
        if native and not public_boundary:
            from bayesfilter.inference.sequential_score_fit_tf import partition_schema
            train, heldout = partition_schema(count, config.holdout_fraction, pair_disjoint=False)
            program = sequential.score_fit_program(scalar, batch, count, dimension, train, heldout,
                jit_compile=jit).python_function

            def evaluate(*args):
                result = program(*args)
                return (result["status"], *(result[field] for field in fields), result["best_value"],
                    result["best_position"], result["best_score"])
        else:
            def evaluate(center, center_score, scale, radius, seed, ridge, floor, condition_cap, tolerance):
                result, _count = sequential._fit_score_curvature(scalar, center, center_score, scale,
                    dimension=dimension, radius=float(radius), sample_count=count,
                    seed=tuple(int(x) for x in seed), config=config, evaluations=0,
                    batched_value_and_score_fn=batch)
                if result["status"] != "usable":
                    raise ValueError(result["status"])
                return (tf.constant(1, tf.int32), *(tf.convert_to_tensor(result[field],
                    tf.int32 if field == "rank" else dtype) for field in fields),
                    tf.convert_to_tensor(result["best_exact_value"], dtype),
                    tf.convert_to_tensor(result["best_exact_position"], dtype),
                    tf.convert_to_tensor(result["best_exact_score"], dtype))

        evaluate.timing_scope = ("complete_tensor_cloud_evaluation_score_fit_and_selection"
            if native and not public_boundary else "complete_public_cloud_evaluation_score_fit_and_selection")
        evaluate.execution_backend = ("tensorflow" if native
            else "legacy_tensorflow_fit_with_numpy_record_selection_diagnostic_only")
        return evaluate, inputs, {"samples": count, "dimension": dimension, "seed": [2026, 919],
            "boundary": "complete_existing_symmetric_score_fit", "frozen_result": True}

    if name == "austria_preparation":
        from bayesfilter.highdim.zhao_cui_austria_sir_parameter_density_training_tf import (
            batch_native_t1_from_common_noise,
        )

        count = 8 * size
        theta = tf.constant([[0., 0., 0.], [.03, -.02, .01], [-.03, .02, -.01]], dtype)
        noise = tf.reshape(tf.sin(tf.cast(tf.range(count * 18), dtype)), [count, 18])
        inputs = (theta, noise, .4 * noise, tf.linspace(tf.constant(4., dtype), 8., 9))
        core = getattr(batch_native_t1_from_common_noise, "python_function", batch_native_t1_from_common_noise)
        return core, inputs, {"parameter_rows": 3, "samples": count, "state_dimension": 18,
            "substeps": 4, "boundary": "complete_common_noise_proposal_density_and_analytical_score",
            "route": "existing_half_step_fourth_stage_execution_adaptation"}

    if name == "source_recenter":
        from bayesfilter.highdim import source_route

        count = 32 * size
        values = tf.cast(tf.range(count), dtype)
        inputs = (tf.stack([tf.sin(.37 * values), tf.cos(.53 * values), .1 * values]),
                  tf.math.log(.5 + tf.square(tf.cos(.23 * values))))
        if hasattr(source_route, "preparation_tf"):
            core = source_route.preparation_tf.recenter.python_function

            def evaluate(samples, weights):
                mean, matrix, _valid = core(samples, weights, tf.constant(1.3, dtype),
                    tf.constant(1e-5, dtype), tf.constant(.1, dtype),
                    tf.constant(0., dtype), use_quantile_scale=True)
                return mean, matrix
        else:
            def evaluate(samples, weights):
                result = source_route.source_route_recenter(samples=samples,
                    log_weights=weights, expansion_factor=1.3, covariance_jitter=1e-5,
                    quantile_fraction=.1, min_ess_for_quantile_scale=0., use_quantile_scale=True)
                return result.mu, result.matrix

        return evaluate, inputs, {"particles": count, "dimension": 3,
            "boundary": "complete_weighted_recenter_and_quantile_stretch",
            "route": "existing_computeL_adaptation"}

    if name == "gamma_preparation":
        from bayesfilter.highdim import (
            c2_transformed_observation_student_proposal_tf as proposal,
        )

        count = 24 * size
        inputs = (tf.constant([-1729, 13], tf.int64), tf.constant(2.5, dtype))
        if hasattr(proposal, "philox_gamma_float64"):
            def evaluate(seed, alpha):
                return proposal.philox_gamma_float64([count], seed, alpha, tf.constant(.5, dtype))
        else:
            def evaluate(seed, alpha):
                return tf.random.stateless_gamma([count], seed, alpha=alpha,
                    beta=tf.constant(.5, dtype), dtype=dtype)

        return evaluate, inputs, {"draws": count, "alpha": 2.5, "rate": .5,
            "stream": "preserved_TensorFlow_2.19.1_scalar_alpha_Philox",
            "boundary": "complete_seeded_gamma_draws"}

    if name == "student_proposal":
        from bayesfilter.highdim import (
            c2_transformed_observation_student_proposal_tf as student,
        )

        dimension, count, seed = 2 * size, 11, (-17, 41)
        proposal = student.build_c2_transformed_observation_student_proposal(
            transition_matrix=tf.eye(dimension, dtype=dtype) * .6 + .02,
            process_covariance=tf.eye(dimension, dtype=dtype) * .3 + .01,
            observation=tf.linspace(tf.constant(.1, dtype), .3, dimension),
            theta_reference=tf.constant([.6, -.91], dtype), nu=3.5, time_index=1)
        inputs = (tf.reshape(tf.linspace(tf.constant(-.4, dtype), .6, count * dimension),
                             [count, dimension]),)
        if hasattr(proposal, "compiled_seeded_sampler"):
            sampler = proposal.compiled_seeded_sampler(count, jit_compile=jit).python_function

            def evaluate(parents):
                return sampler(parents, tf.constant(seed, tf.int64))
        else:
            def evaluate(parents):
                return proposal.sample_with_seed(parents, count, seed, jit_compile=jit)

        return evaluate, inputs, {"particles": count, "dimension": dimension, "seed": seed,
            "nu": 3.5, "boundary": "seeded_proposal_and_density_after_frozen_geometry",
            "canonical_admitted": False}

    if name == "ukf_initializer":
        from bayesfilter.highdim import ukf_initializer as initializer
        from bayesfilter.highdim.bases import (
            BoundedInterval,
            LegendreBasis1D,
            ProductBasis,
        )
        from bayesfilter.highdim.diagnostics import (
            DensityMeasure,
            MassMeasure,
            MeasureConvention,
        )

        dimension, rank, order = 2 * size, 3, 16
        convention = MeasureConvention(density_measure=DensityMeasure.REFERENCE_MEASURE,
            mass_measure=MassMeasure.REFERENCE_MEASURE, reference_weight_name="omega")
        basis = ProductBasis(tuple(LegendreBasis1D(BoundedInterval(-1., 1.), 2 + axis % 3)
                                   for axis in range(dimension)), convention)
        ranks = (1, *([rank] * (dimension - 1)), 1)
        config = initializer.P76UKFInitializerConfig(product_basis=basis, ranks=ranks,
                                                    quadrature_order=order)
        # Distinct eigenvalues and off-diagonal covariance exercise the actual
        # frame calculation. Compare its reconstructed covariance, because
        # eigenvector signs are mathematically arbitrary across backends.
        covariance = tf.linalg.diag(tf.linspace(tf.constant(.6, dtype), 1.4, dimension)) + .03
        inputs = (tf.linspace(tf.constant(-.1, dtype), .2, dimension), covariance)
        if "jit_compile" in inspect.signature(initializer.p76_build_ukf_initializer).parameters:
            from bayesfilter.highdim.ukf_initializer_tf import initializer_program

            program = initializer_program(basis, ranks, order, jit_compile=jit).python_function

            def evaluate(center, covariance):
                coefficients, cores, linear_map, stabilized, _valid = program(
                    center, covariance, tf.constant(config.gamma, dtype),
                    tf.constant(config.covariance_abs_floor, dtype),
                    tf.constant(config.covariance_rel_floor, dtype), tf.zeros([dimension], dtype),
                    tf.eye(dimension, dtype=dtype), tf.constant(config.seed_epsilon / (rank - 1), dtype))
                return (tuple(coefficients[axis, :local.basis_dim] for axis, local in enumerate(basis.bases)),
                    tuple(cores[axis, :ranks[axis], :local.basis_dim, :ranks[axis+1]]
                          for axis, local in enumerate(basis.bases)),
                    linear_map @ tf.transpose(linear_map), stabilized[0], stabilized[2], stabilized[3])
        else:
            def evaluate(center, covariance):
                moments = initializer.P76AdjacentUKFMoments(center, covariance, 1, 0, "scout_not_truth")
                _, linear_map, stabilized = initializer.p76_local_frame_from_moments(moments, config)
                coefficients = initializer.p76_gaussian_sqrt_projection_coefficients(
                    basis, gamma=config.gamma, quadrature_order=order, center=center,
                    linear_map=linear_map, reference_offset=tf.zeros([dimension], dtype),
                    reference_matrix=tf.eye(dimension, dtype=dtype))
                cores = initializer.p76_embed_rank_one_with_seeded_channels(
                    coefficients, ranks=ranks, seed_epsilon=config.seed_epsilon)
                return (coefficients, tuple(core.values for core in cores),
                    linear_map @ tf.transpose(linear_map), stabilized.covariance,
                    stabilized.raw_eigenvalues, stabilized.floored_eigenvalues)

        return evaluate, inputs, {"dimension": dimension, "ranks": ranks,
            "quadrature_order": order, "boundary": "moments_to_frame_projection_and_TT_cores",
            "route": "extension_or_invention", "canonical_admitted": False}

    if name == "moment_teacher":
        from bayesfilter.highdim import zhao_cui_moment_teacher_lgssm_tf as teacher

        horizon, rows, dimension = 2 * size, 24, 6
        theta = tf.constant([.55, .45, .35, .8, .6], dtype)
        # An explicit frozen tensor design avoids drawing or fitting a different
        # fixture in the two source arms. Degree-one operators are exact integrals
        # of 1 and sqrt(3)*x under the uniform measure on [-1, 1].
        reference = tf.reshape(.85 * tf.sin(tf.cast(tf.range(rows * dimension), dtype)), [rows, dimension])
        basis_values = tf.stack([tf.ones([dimension, rows], dtype), tf.sqrt(tf.constant(3., dtype))
                                 * tf.transpose(reference)], axis=-1)
        root3 = 3. ** .5
        powers = tf.constant([[[1., 0.], [0., 1.]], [[0., 1. / root3], [1. / root3, 0.]],
            [[1. / 3., 0.], [0., 3. / 5.]], [[0., root3 / 5.], [root3 / 5., 0.]],
            [[1. / 5., 0.], [0., 3. / 7.]]], dtype)
        prepared = {
            "observations": tf.reshape(tf.linspace(tf.constant(-.1, dtype), .2, horizon * 3), [horizon, 3]),
            "reference_points": reference, "basis_values": basis_values,
            "active_mask": tf.ones([dimension, 1, 2, 1], dtype),
            "schedule": tf.range(dimension), "weights": tf.fill([rows], tf.constant(1. / rows, dtype)),
            "initial_cores": tf.reshape(tf.stack([tf.ones([dimension], dtype),
                .03 * tf.sin(tf.cast(tf.range(dimension), dtype))], -1), [dimension, 1, 2, 1]),
            "scale_shift_indices": tf.zeros([horizon], tf.int32),
            "defensive_weights": tf.fill([horizon], tf.constant(.05, dtype)),
            "query_basis_values": tf.concat([basis_values[3:], basis_values[3:]], 0),
            "keep_mask": tf.constant([True, True, True, False, False, False]),
            "mass_operators": tf.eye(2, batch_shape=[dimension], dtype=dtype),
            "defensive_marginal_values": tf.ones([horizon, rows], dtype),
            "defensive_mass": tf.constant(1., dtype),
            "operator_powers": tf.broadcast_to(powers, [dimension, 5, 2, 2]),
            "defensive_power_moments": tf.tile(tf.constant([[1., 0., 1./3., 0., 1./5.]], dtype), [dimension, 1]),
            "state_offset": tf.zeros([3], dtype),
            "state_matrix": tf.concat([2.5 * tf.eye(3, dtype=dtype), tf.zeros([3, 3], dtype)], 1),
            "pair_indices": tf.constant(teacher.PAIR_INDICES, tf.int32),
            "chart_scale": tf.constant(2.5, dtype), "center_theta": theta,
        }
        controls = teacher.MomentTeacherControls(sinkhorn_steps=2, balance_steps=100,
            correction_steps=0, correction_strength=0., correction_floor=1e-6,
            pairwise_correction_steps=0, pairwise_strength=0., pairwise_floor=1e-6,
            tt_ridge=1e-5, column_scale_floor=1e-6, condition_number_veto=1e10, fit_residual_veto=2.)
        keys = tuple(prepared)

        def evaluate(theta, *values):
            return teacher._teacher_targets(theta, dict(zip(keys, values, strict=True)), controls, setup_static=True)

        return evaluate, (theta, *prepared.values()), {"horizon": horizon, "fit_rows": rows,
            "dimension": dimension, "rank": 1, "basis_size": 2, "parameters": 5,
            "boundary": "complete_fixed_TT_teacher_moments_and_derivatives",
            "design": "fixed_sinusoidal_v1", "canonical_admitted": False}
    raise ValueError(f"Unregistered preparation fixture: {name}")
