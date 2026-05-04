import numpy as np
import pytest

from bayesfilter.filters import (
    ParticleFilterConfig,
    kalman_log_likelihood,
    particle_filter_log_likelihood,
)
from bayesfilter.testing.structural_fixtures import AR2StructuralModel


def _ar2_identity_residual(previous_particles, next_particles):
    return next_particles[:, 1] - previous_particles[:, 0]


def test_structural_bootstrap_particle_likelihood_is_finite_and_near_kalman():
    model = AR2StructuralModel(phi1=0.35, phi2=-0.10, sigma=0.25, observation_sigma=0.15)
    observations = np.array([[0.2], [0.05], [-0.1], [0.15]], dtype=float)

    particle = particle_filter_log_likelihood(
        model,
        observations,
        config=ParticleFilterConfig(num_particles=25000, random_seed=123),
        identity_diagnostic=_ar2_identity_residual,
    )
    kalman = kalman_log_likelihood(observations, model.as_lgssm())

    assert np.isfinite(particle.log_likelihood)
    assert abs(particle.log_likelihood - kalman.log_likelihood) < 0.08
    assert particle.metadata.filter_name == "structural_bootstrap_particle"
    assert particle.metadata.integration_space == "innovation"
    assert particle.metadata.differentiability_status == "monte_carlo_value_only"
    assert particle.diagnostics["max_identity_residual"] < 1e-14


def test_particle_filter_preserves_deterministic_completion_in_final_particles():
    model = AR2StructuralModel()

    result = particle_filter_log_likelihood(
        model,
        np.array([[0.0]], dtype=float),
        config=ParticleFilterConfig(num_particles=128, random_seed=9),
        return_particles=True,
        identity_diagnostic=_ar2_identity_residual,
    )

    assert result.final_particles.shape == (128, 2)
    assert result.diagnostics["max_identity_residual"] < 1e-14


def test_particle_filter_blocks_artificial_deterministic_noise_without_label():
    with pytest.raises(ValueError, match="deterministic-coordinate noise"):
        ParticleFilterConfig(deterministic_noise_scale=0.01)


def test_particle_filter_allows_labeled_artificial_noise_as_approximation_only():
    config = ParticleFilterConfig(
        deterministic_noise_scale=0.01,
        approximation_label="off_manifold_debug_noise",
        proposal_correction="declared_approximation",
    )

    assert config.approximation_label == "off_manifold_debug_noise"
    assert config.proposal_correction == "declared_approximation"
