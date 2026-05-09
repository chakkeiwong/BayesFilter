from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter import (
    StatePartition,
    StructuralFilterConfig,
    make_affine_structural_tf,
    pointwise_deterministic_residuals,
    structural_block_metadata,
    structural_filter_diagnostics,
    structural_filter_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def _ar2_structural_model(config: StructuralFilterConfig | None = None):
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        config=config,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.constant([[0.35, -0.10], [1.0, 0.0]], dtype=tf.float64),
        innovation_matrix=tf.constant([[0.25], [0.0]], dtype=tf.float64),
        innovation_covariance=tf.eye(1, dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.15**2]], dtype=tf.float64),
    )


def test_affine_structural_transition_completes_deterministic_state_pointwise() -> None:
    model = _ar2_structural_model()
    previous = tf.constant(
        [[0.7, -0.2], [0.1, 0.3], [-0.4, 0.5]],
        dtype=tf.float64,
    )
    innovation = tf.constant([[1.1], [-0.2], [0.4]], dtype=tf.float64)

    next_points = model.transition(previous, innovation)
    residuals = pointwise_deterministic_residuals(model, previous, innovation)

    np.testing.assert_allclose(next_points.numpy()[:, 1], previous.numpy()[:, 0], atol=1e-14)
    np.testing.assert_allclose(residuals.numpy(), np.zeros([3, 1]), atol=1e-14)


def test_structural_metadata_separates_blocks_and_completion_policy() -> None:
    model = _ar2_structural_model()

    metadata = structural_filter_metadata(model, filter_name="tf_structural_protocol_test")
    diagnostics = structural_filter_diagnostics(model)
    blocks = structural_block_metadata(model)

    assert metadata.integration_space == "innovation"
    assert metadata.deterministic_completion == "required"
    assert metadata.partition is model.partition
    assert diagnostics.extra["stochastic_indices"] == (0,)
    assert diagnostics.extra["deterministic_indices"] == (1,)
    assert diagnostics.extra["exogenous_block"]["innovation_dim"] == 1
    assert diagnostics.extra["deterministic_completion_block"]["indices"] == (1,)
    assert blocks["collapsed_full_state"] is False


def test_collapsed_full_state_route_is_metadata_distinct_from_structural_route() -> None:
    collapsed = _ar2_structural_model(
        StructuralFilterConfig(
            integration_space="full_state",
            deterministic_completion="approximate",
            approximation_label="collapsed_full_state_singular_covariance",
            allow_full_state_for_mixed=True,
        )
    )

    metadata = structural_filter_metadata(collapsed, filter_name="tf_structural_collapsed_test")
    diagnostics = structural_filter_diagnostics(collapsed)

    assert metadata.integration_space == "full_state"
    assert metadata.deterministic_completion == "approximate"
    assert metadata.approximation_label == "collapsed_full_state_singular_covariance"
    assert diagnostics.extra["collapsed_full_state"] is True
    assert diagnostics.extra["deterministic_completion_block"]["policy"] == "approximate"


def test_structural_protocol_rejects_hidden_full_state_for_mixed_models() -> None:
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    with pytest.raises(ValueError, match="full-state integration for mixed models"):
        make_affine_structural_tf(
            partition=partition,
            config=StructuralFilterConfig(
                integration_space="full_state",
                deterministic_completion="required",
            ),
            initial_mean=tf.zeros([2], dtype=tf.float64),
            initial_covariance=tf.eye(2, dtype=tf.float64),
            transition_offset=tf.zeros([2], dtype=tf.float64),
            transition_matrix=tf.eye(2, dtype=tf.float64),
            innovation_matrix=tf.constant([[1.0], [0.0]], dtype=tf.float64),
            innovation_covariance=tf.eye(1, dtype=tf.float64),
            observation_offset=tf.zeros([1], dtype=tf.float64),
            observation_matrix=tf.constant([[1.0, 0.0]], dtype=tf.float64),
            observation_covariance=tf.eye(1, dtype=tf.float64),
        )


def test_structural_tf_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (ROOT / "bayesfilter" / "structural_tf.py").read_text(encoding="utf-8")

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text
