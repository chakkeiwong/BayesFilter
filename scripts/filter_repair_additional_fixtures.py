"""Matched diagnostic fixtures for routes added during the closure review.

Both isolated source arms receive the same frozen arrays and FP64 settings.
These preserve existing finite-program/statistical scores without LEDH admission.
"""

from dataclasses import replace

FIXTURES = ("latent_sir", "initial_rqmc", "kalman_information")


def fixture(tf, name, size, jit):
    del jit
    dtype, horizon = tf.float64, 2 * size
    if name == "latent_sir":
        from bayesfilter.highdim import ledh_contract_e_latent_sir_tf as sir

        spec = sir.LatentSIRStaticSpec(
            state_dimension=2, observation_dimension=1, compartments=1,
            initial_mean=tf.constant([0.3, 0.2], dtype),
            initial_covariance=tf.linalg.diag(tf.constant([0.25, 0.16], dtype)),
            process_covariance=tf.linalg.diag(tf.constant([0.25, 0.16], dtype)),
            base_observation_covariance=tf.constant([[0.16]], dtype),
            base_kappa=tf.constant([0.1], dtype), base_nu=tf.constant([1.], dtype),
            adjacency=tf.zeros([1, 1], dtype), neighbor_degree=tf.zeros([1], dtype),
            step=tf.constant(0.005, dtype), substeps=4, zhao_cui_rk4_variant=False,
        )
        cloud = tf.constant([
            [-1.5, -1.], [-1., 0.5], [-0.5, 1.], [-0.2, -1.5],
            [0.2, 1.5], [0.5, -0.5], [1., -1.], [1.5, 1.],
        ], dtype)
        residual = tf.constant([
            [-1., -0.75], [-0.75, 0.25], [-0.5, 0.5], [-0.25, -1.],
            [0.25, 1.], [0.5, -0.5], [0.75, -0.25], [1., 0.75],
        ], dtype)
        inputs = (
            tf.constant([0.04, -0.03, 0.02], dtype),
            tf.tile(tf.constant([[0.15], [0.1]], dtype), [size, 1]), cloud[None],
            tf.broadcast_to((0.3 * cloud)[None, None], [1, horizon - 1, 8, 2]),
            tf.ones([1, horizon], tf.bool),
            tf.broadcast_to(residual[None, None], [1, horizon, 8, 2]),
            tf.fill([1, horizon], tf.constant(1e-5, dtype)),
            tf.constant(0.25, dtype), tf.constant(0.9, dtype),
        )

        def evaluate(theta, observations, initial, transition, mask, residual, ridge, epsilon, scaling):
            prepared = {"observations": observations, "initial_noise": initial,
                "transition_noise": transition, "fixed_reset_mask": mask,
                "residual_design": residual, "prepared_ridge": ridge, "epsilon": epsilon, "scaling": scaling}
            row = sir.latent_sir_contract_e_value_and_score_core(theta, prepared, spec,
                steps=20, balance_steps=100, row_chunk_size=8, col_chunk_size=8)
            return (row["objective"], row["score"], row["final_particles"],
                    row["final_particles_tangent"], row["increment_history"],
                    row["increment_score_history"], row["valid_chart"], row["reset_valid_history"])

        return evaluate, inputs, {"horizon": horizon, "particles": 8, "state": 2, "parameters": 3,
            "annealing_steps": 20, "balance_steps": 100, "score": "finite_program_manual_jvp_diagnostic_only",
            "canonical_admitted": False}

    if name == "initial_rqmc":
        from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as rqmc
        from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
            diagonal_lgssm_callbacks,
        )
        from experiments.dpf_implementation.tf_tfp.filters import (
            experimental_batched_ledh_pfpf_ot_tf as flow,
        )
        from experiments.dpf_implementation.tf_tfp.resampling import (
            annealed_transport_tf as transport,
        )

        flow.DTYPE = transport.DTYPE = dtype
        callbacks = diagonal_lgssm_callbacks()
        matrix = tf.cast(callbacks.model.observation_matrix, dtype)

        def observation_callbacks(_theta, _time):
            return (
                lambda points: tf.einsum("bnd,od->bno", points, matrix),
                lambda points: tf.broadcast_to(matrix[None, None], [*points.shape[:2], 3, 3]),
                lambda predicted, observed: observed[None, None] - predicted,
            )

        callbacks = replace(callbacks, observation_callbacks=observation_callbacks)
        design = tf.constant([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]], dtype)
        inputs = (tf.constant([0.2, 0.3, 0.4, 0.5, 0.8], dtype),
            tf.reshape(tf.linspace(tf.constant(-0.1, dtype), 0.2, horizon * 3), [horizon, 3]),
            0.3 * design, tf.broadcast_to((0.2 * design)[None], [horizon - 1, 4, 3]), design)

        def evaluate(theta, observations, initial, process, design):
            value, score, diagnostics = rqmc.finite_value_standard_score_initial_rqmc(
                callbacks, theta, observations, initial, process, design,
                functional_time_loop=False, reset_policy="none", ancestry_policy="existing_one_to_one",
                epsilon=2., sinkhorn_steps=2, balance_steps=4)
            return value, score, diagnostics

        return evaluate, inputs, {"horizon": horizon, "particles": 4, "state": 3, "parameters": 5,
            "reset": "none", "score": "analytical_statistical_backward_score_not_finite_value_gradient",
            "canonical_admitted": False, "precision": "explicit_FP64_reference"}

    if name == "kalman_information":
        from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as information

        inputs = (tf.constant([0.2, 0.3, 0.4, 0.5, 0.8], dtype),
            tf.reshape(tf.linspace(tf.constant(-0.1, dtype), 0.2, horizon * 3), [horizon, 3]),
            tf.constant([[0.1, 0.2, -0.1], [-0.3, 0.1, 0.4]], dtype))
        kalman = getattr(information.exact_kalman_value, "python_function", information.exact_kalman_value)

        def evaluate(theta, observations, points):
            with tf.GradientTape() as tape:
                tape.watch(theta)
                value = kalman(theta, observations)
            return (value, tape.gradient(value, theta),
                information._conditional_future_log_likelihood(theta, points, observations),
                *information._backward_information_parameters(theta, observations),
                *information._finite_lookahead_information_parameters(theta, observations, 2))

        return evaluate, inputs, {"horizon": horizon, "state": 3, "parameters": 5, "conditional_points": 2,
            "lookahead": 2, "score": "autodiff_reference_for_Kalman_value"}
    raise ValueError(f"Unregistered additional fixture: {name}")
