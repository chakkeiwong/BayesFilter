from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.models import (
    ParameterizedZhaoCuiSIRSSM,
    SpatialSIRSSM,
    parameterized_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.sir_latent_preclip_tf import (
    LATENT_PRECLIP_CANONICAL_STATUS,
    LATENT_PRECLIP_REPRESENTATION_CLASS,
    LatentPreclipSIRSSM,
    latent_preclip_zhao_cui_sir_austria_model,
)


DTYPE = tf.float64


def _small_model(compartments: int) -> LatentPreclipSIRSSM:
    initial_pairs = tf.stack(
        [
            tf.fill([compartments], tf.constant(4.0, DTYPE)),
            tf.fill([compartments], tf.constant(0.2, DTYPE)),
        ],
        axis=1,
    )
    base = SpatialSIRSSM(
        kappa=tf.fill([compartments], tf.constant(0.1, DTYPE)),
        nu=tf.fill([compartments], tf.constant(1.0, DTYPE)),
        initial_mean=tf.reshape(initial_pairs, [-1]),
        neighbor_sets=tuple(() for _ in range(compartments)),
        delta=0.02,
        rk4_internal_step=0.005,
        process_covariance=tf.eye(2 * compartments, dtype=DTYPE),
        observation_covariance=tf.eye(compartments, dtype=DTYPE),
        initial_covariance=tf.eye(2 * compartments, dtype=DTYPE),
        process_noise_policy="clip_susceptible_after_noise",
    )
    return LatentPreclipSIRSSM(ParameterizedZhaoCuiSIRSSM(base))


def _source_style_path(
    model: LatentPreclipSIRSSM,
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    observation_noise: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    scaled = model.physical_model.scaled_model(theta)
    initial_chol = tf.linalg.cholesky(scaled.initial_covariance)
    process_chol = tf.linalg.cholesky(scaled.process_covariance)
    observation_chol = tf.linalg.cholesky(scaled.observation_covariance)
    state = scaled.initial_mean + tf.linalg.matvec(initial_chol, initial_noise)
    states = [state]
    observations = [
        model.physical_model.infectious_components(state)[0]
        + tf.linalg.matvec(observation_chol, observation_noise[0])
    ]
    for index in range(int(transition_noise.shape[0])):
        mean = model.physical_model.transition_mean(theta, state[tf.newaxis, :])[0]
        unclipped = mean + tf.linalg.matvec(process_chol, transition_noise[index])
        state = model.physical_state(unclipped, time_index=index + 1)[0]
        states.append(state)
        observations.append(
            model.physical_model.infectious_components(state)[0]
            + tf.linalg.matvec(observation_chol, observation_noise[index + 1])
        )
    return tf.stack(states), tf.stack(observations)


def test_latent_target_identity_is_explicitly_noncanonical_extension() -> None:
    payload = latent_preclip_zhao_cui_sir_austria_model().manifest_payload()
    assert payload["representation_classification"] == LATENT_PRECLIP_REPRESENTATION_CLASS
    assert payload["canonical_contract_e_status"] == LATENT_PRECLIP_CANONICAL_STATUS
    assert payload["physical_state_time_order"] == (
        "x_0=z_0; x_t=clip_susceptible(z_t) for t>=1"
    )
    assert "contract_e_chol_canonical_admission" in payload["what_is_not_claimed"]


def test_physical_projection_preserves_unclipped_initial_and_clips_later() -> None:
    model = _small_model(1)
    latent = tf.constant([[-2.0, -0.5]], DTYPE)
    tf.debugging.assert_equal(model.physical_state(latent, time_index=0), latent)
    tf.debugging.assert_equal(
        model.physical_state(latent, time_index=1),
        tf.constant([[0.0, -0.5]], DTYPE),
    )


def test_paired_noise_reproduces_source_physical_law_for_j1_j2_j9() -> None:
    for compartments in (1, 2, 9):
        model = _small_model(compartments)
        theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
        initial_noise = tf.linspace(
            tf.constant(-0.2, DTYPE), tf.constant(0.2, DTYPE), model.state_dim()
        )
        transition_noise = tf.zeros([2, model.state_dim()], DTYPE)
        transition_noise = tf.tensor_scatter_nd_update(
            transition_noise,
            indices=tf.stack(
                [
                    tf.zeros([compartments], tf.int32),
                    tf.range(0, model.state_dim(), 2, dtype=tf.int32),
                ],
                axis=1,
            ),
            updates=tf.fill([compartments], tf.constant(-10.0, DTYPE)),
        )
        observation_noise = tf.reshape(
            tf.linspace(
                tf.constant(-0.1, DTYPE),
                tf.constant(0.1, DTYPE),
                3 * model.observation_dim(),
            ),
            [3, model.observation_dim()],
        )

        latent = model.simulate_from_standard_normals(
            theta, initial_noise, transition_noise, observation_noise
        )
        source_states, source_observations = _source_style_path(
            model, theta, initial_noise, transition_noise, observation_noise
        )

        tf.debugging.assert_equal(latent["physical_path"], source_states)
        tf.debugging.assert_equal(latent["observations"], source_observations)
        assert bool(tf.reduce_any(latent["physical_path"][1:, 0::2] == 0.0).numpy())


def test_wrong_initial_clipping_changes_first_transition() -> None:
    model = _small_model(1)
    theta = tf.zeros([3], DTYPE)
    latent_initial = tf.constant([[-2.0, 0.4]], DTYPE)
    correct = model.transition_mean(theta, latent_initial, time_index=1)
    wrongly_clipped = model.physical_model.transition_mean(
        theta, model.physical_state(latent_initial, time_index=1)
    )
    assert float(tf.reduce_max(tf.abs(correct - wrongly_clipped)).numpy()) > 1.0e-6


def test_latent_transition_is_unclipped_and_has_matching_gaussian_density() -> None:
    model = _small_model(1)
    theta = tf.constant([0.01, -0.02, 0.03], DTYPE)
    previous = tf.constant([[4.0, 0.2]], DTYPE)
    noise = tf.constant([[-10.0, 0.25]], DTYPE)
    latent_next = model.transition_push_from_standard_normal(
        theta, previous, noise, 1
    )
    assert float(latent_next[0, 0].numpy()) < 0.0
    density = model.transition_log_density(theta, previous, latent_next, t=1)
    assert bool(tf.reduce_all(tf.math.is_finite(density)).numpy())


def test_austria_wrapper_has_declared_dimensions() -> None:
    model = latent_preclip_zhao_cui_sir_austria_model()
    assert model.state_dim() == 18
    assert model.observation_dim() == 9
    assert model.parameter_dim() == 3
    assert isinstance(model.physical_model, type(parameterized_zhao_cui_sir_austria_model()))
