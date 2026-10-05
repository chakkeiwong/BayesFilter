"""Complete seeded-initializer and prefix-target execution diagnostics.

Seeds retain the original Philox draws. Baseline host/tracing failures remain
visible and use their valid eager reference, without modifying runtime code.
"""

from dataclasses import replace

from filter_repair_centered_fixtures import exact_parent

FIXTURES = ("centered_seeded_initializer", "centered_prefix_targets")


def fixture(tf, name, size, jit):
    del jit
    from bayesfilter.highdim import zhao_cui_austria_sir_lane_b_tf as lane
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_parameter_density_training_tf as training,
    )
    from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
        CenteredThetaFeatures,
        centered_lane_b_product_basis,
    )

    dtype, seed = tf.float64, 1729
    dimensions = {"seed": seed, "stream": "preserved_original_TensorFlow_Philox",
        "classification": "existing_extension_or_invention", "canonical_admitted": False}
    if name == "centered_seeded_initializer":
        parent = exact_parent(tf)
        rank = size + 1
        settings = replace(parent.settings, rank=rank)
        basis = centered_lane_b_product_basis(order=settings.basis_order,
                                             num_elems=settings.basis_num_elems)
        features = CenteredThetaFeatures()

        def evaluate():
            balanced = lane.balanced_initial_cores(settings, basis)
            residual = training.fixed_rank_initial_residual_components(parent=parent,
                features=features, rank=rank, seed=seed, amplitude_scale=.002, perturbation_scale=.003)
            connected = training.embed_residual_component_with_connected_channels(residual[0],
                target_rank=rank + 2, seed=seed, seeded_channel_epsilon=.03)
            return balanced, residual, connected

        dimensions.update(dimension=36, rank=rank, connected_rank=rank + 2,
            amplitude_scale=.002, perturbation_scale=.003, seeded_channel_epsilon=.03,
            basis_width=3, feature_count=features.feature_count,
            boundary="complete_balanced_residual_and_connected_seeded_initialization")
        return evaluate, (), dimensions

    if name == "centered_prefix_targets":
        rows, samples = 2 * size, 8 * size
        model = training.parameterized_zhao_cui_sir_austria_model()
        mean = model.base_model.transition_mean(model.base_model.initial_mean[None])[0]
        points = mean[None] + tf.reshape(.2 * tf.sin(tf.cast(tf.range(rows * 18), dtype)), [rows, 18])

        def evaluate(points, global_score, global_se):
            estimate = training.RatioScoreEstimate(value=tf.constant(.1, dtype), score=global_score,
                score_standard_error=global_se, effective_sample_size=tf.constant(6., dtype))
            output = training.estimate_t1_prefix_scores(prefix_points=points, global_score=estimate,
                sample_count=samples, seed=seed)
            return tuple(tf.stack(tuple(getattr(row, key) for row in output))
                         for key in ("value", "score", "score_standard_error", "effective_sample_size"))

        dimensions.update(rows=rows, samples=samples, state_dimension=18, parameter_count=3,
            boundary="complete_seeded_prefix_mass_score_standard_error_and_ess",
            sealed_dataset_validation="host_preparation_before_candidate_numerical_trace")
        return evaluate, (points, tf.constant([.2, -.1, .3], dtype), tf.constant([.02, .03, .01], dtype)), dimensions
    raise ValueError(f"Unregistered initialization fixture: {name}")
