import numpy as np
import pytest

from bayesfilter import StructuralFilterConfig, validate_filter_config
from bayesfilter.filters.particles import ParticleFilterConfig
from bayesfilter.filters.sigma_points import StructuralSVDSigmaPointFilter
from bayesfilter.testing.structural_fixtures import AR2StructuralModel


def test_sigma_point_result_records_structural_metadata():
    model = AR2StructuralModel()
    result = StructuralSVDSigmaPointFilter().filter(model, [[0.1], [0.0]])

    assert result.metadata.filter_name == "structural_svd_sigma_point"
    assert result.metadata.partition == model.partition
    assert result.metadata.integration_space == "innovation"
    assert result.metadata.deterministic_completion == "required"
    assert result.metadata.approximation_label == "sigma_point_gaussian_closure"


def test_missing_structural_metadata_fails_closed_for_structural_config():
    config = StructuralFilterConfig(
        integration_space="innovation",
        deterministic_completion="required",
    )

    with pytest.raises(ValueError, match="requires partition metadata"):
        validate_filter_config(None, config)


def test_particle_namespace_records_audited_structural_metadata():
    from bayesfilter.filters import particle_filter_log_likelihood

    model = AR2StructuralModel()
    result = particle_filter_log_likelihood(
        model,
        np.array([[0.1]], dtype=float),
        config=ParticleFilterConfig(num_particles=128, random_seed=2),
    )

    assert result.metadata.filter_name == "structural_bootstrap_particle"
    assert result.metadata.partition == model.partition
    assert result.metadata.integration_space == "innovation"
    assert result.metadata.deterministic_completion == "required"
