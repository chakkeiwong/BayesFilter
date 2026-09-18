"""Matched preparation diagnostics using frozen tensors in isolated source arms.

No fixture imports a second source tree or changes runtime functions. Baseline
host-only APIs retain their real tracing failures; eager results are separate
reference measurements. These fixtures do not grant scientific admission.
"""

import inspect

FIXTURES = ("source_recenter", "gamma_preparation", "student_proposal",
            "ukf_initializer", "moment_teacher", "austria_preparation")


def fixture(tf, name, size, jit):
    dtype = tf.float64
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
