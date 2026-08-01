from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_coupled_nonlinear import (
    CoupledNonlinearGaussianModel,
    GaussianQuantileCoordinateMap,
    ShiftedAlgebraicCoordinateMap,
)


def _states(block_count: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    previous_block = tf.constant([[0.7, 0.25], [0.9, 0.15]], tf.float64)
    current_block = tf.constant([[0.66, 0.22], [0.88, 0.17]], tf.float64)
    previous = tf.tile(previous_block, [1, block_count])
    current = tf.tile(current_block, [1, block_count])
    observation = tf.tile(tf.constant([0.19], tf.float64), [block_count])
    return previous, current, observation


def test_exact_conditional_identity_matches_transition_times_observation() -> None:
    model = CoupledNonlinearGaussianModel(3, dtype=tf.float64)
    theta = tf.constant([0.12, -0.08, 0.03], tf.float64)
    previous, current, observation = _states(model.block_count)

    conditional = model.conditional_log_density(
        theta, previous, current, observation
    )
    predictive = model.predictive_observation_mean(theta, previous)
    predictive_variance = model.predictive_observation_variance()
    residual = observation[None, :] - predictive
    log_predictive = -0.5 * tf.reduce_sum(
        tf.constant(1.8378770664093453, tf.float64)
        + tf.math.log(predictive_variance)
        + tf.square(residual) / predictive_variance,
        axis=1,
    )
    joint = model.transition_log_density(
        theta, previous, current, 1
    ) + model.observation_log_density(theta, current, observation, 1)

    tf.debugging.assert_near(conditional, joint - log_predictive, atol=2e-12)


def test_analytical_parameter_scores_match_density_finite_differences() -> None:
    model = CoupledNonlinearGaussianModel(2, dtype=tf.float64)
    theta = tf.constant([0.12, -0.08, 0.03], tf.float64)
    previous, current, observation = _states(model.block_count)
    step = tf.constant(1e-5, tf.float64)

    transition_score = model.transition_log_density_parameter_score(
        theta, previous, current, 1
    )
    observation_score = model.observation_log_density_parameter_score(
        theta, current, observation, 1
    )
    transition_fd = []
    observation_fd = []
    for parameter_index in range(model.parameter_dim()):
        direction = tf.one_hot(parameter_index, model.parameter_dim(), dtype=tf.float64)
        transition_fd.append(
            (
                model.transition_log_density(theta + step * direction, previous, current, 1)
                - model.transition_log_density(theta - step * direction, previous, current, 1)
            )
            / (2.0 * step)
        )
        observation_fd.append(
            (
                model.observation_log_density(
                    theta + step * direction, current, observation, 1
                )
                - model.observation_log_density(
                    theta - step * direction, current, observation, 1
                )
            )
            / (2.0 * step)
        )

    tf.debugging.assert_near(
        transition_score, tf.stack(transition_fd, axis=1), atol=2e-8
    )
    tf.debugging.assert_near(
        observation_score, tf.stack(observation_fd, axis=1), atol=2e-8
    )


def test_float32_model_core_compiles_with_xla_on_cpu() -> None:
    model = CoupledNonlinearGaussianModel(2)
    theta = tf.constant([0.12, -0.08, 0.03], tf.float32)
    previous64, current64, observation64 = _states(model.block_count)
    previous = tf.cast(previous64, tf.float32)
    current = tf.cast(current64, tf.float32)
    observation = tf.cast(observation64, tf.float32)

    @tf.function(jit_compile=True, autograph=False)
    def evaluate() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            model.transition_mean(theta, previous),
            model.transition_log_density(theta, previous, current, 1),
            model.transition_log_density_parameter_score(
                theta, previous, current, 1
            ),
            model.observation_log_density_parameter_score(
                theta, current, observation, 1
            ),
        )

    mean, log_density, transition_score, observation_score = evaluate()

    assert mean.shape == (2, 4)
    assert log_density.shape == (2,)
    assert transition_score.shape == (2, 3)
    assert observation_score.shape == (2, 3)
    for value in (mean, log_density, transition_score, observation_score):
        assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def test_shifted_algebraic_map_roundtrip_and_jacobian_directions() -> None:
    coordinate_map = ShiftedAlgebraicCoordinateMap(
        tf.constant([0.8, 0.2], tf.float64),
        tf.constant([0.5, 0.3], tf.float64),
    )
    reference = tf.constant(
        [[-0.8, 0.4], [-0.2, 0.0], [0.6, -0.7]], tf.float64
    )

    physical, log_dxdz = coordinate_map.forward(reference)
    reconstructed, log_dzdx = coordinate_map.inverse(physical)
    forward_components = coordinate_map.forward_log_det_components(reference)
    inverse_components = coordinate_map.inverse_log_det_components(physical)

    tf.debugging.assert_near(reconstructed, reference, atol=2e-12)
    tf.debugging.assert_near(log_dxdz + log_dzdx, tf.zeros([3], tf.float64), atol=2e-12)
    tf.debugging.assert_near(
        tf.reduce_sum(forward_components, axis=1), log_dxdz, atol=2e-12
    )
    tf.debugging.assert_near(
        forward_components + inverse_components,
        tf.zeros([3, 2], tf.float64),
        atol=2e-12,
    )
    assert coordinate_map.manifest_payload()["route_classification"] == "extension_or_invention"


def test_gaussian_quantile_map_roundtrip_and_jacobian_directions() -> None:
    coordinate_map = GaussianQuantileCoordinateMap(
        tf.constant([0.8, 0.2], tf.float64),
        tf.constant([0.2, 0.15], tf.float64),
    )
    reference = tf.constant(
        [[-0.95, 0.4], [-0.2, 0.0], [0.8, -0.7]], tf.float64
    )

    physical, log_dxdz = coordinate_map.forward(reference)
    reconstructed, log_dzdx = coordinate_map.inverse(physical)
    forward_components = coordinate_map.forward_log_det_components(reference)
    inverse_components = coordinate_map.inverse_log_det_components(physical)

    tf.debugging.assert_near(reconstructed, reference, atol=2e-12)
    tf.debugging.assert_near(log_dxdz + log_dzdx, tf.zeros([3], tf.float64), atol=2e-12)
    tf.debugging.assert_near(
        tf.reduce_sum(forward_components, axis=1), log_dxdz, atol=2e-12
    )
    tf.debugging.assert_near(
        forward_components + inverse_components,
        tf.zeros([3, 2], tf.float64),
        atol=2e-12,
    )
    manifest = coordinate_map.manifest_payload()
    assert manifest["family"] == "GaussianQuantileCoordinateMap"
    assert manifest["route_classification"] == "extension_or_invention"
