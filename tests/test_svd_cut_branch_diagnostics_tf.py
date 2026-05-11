import os

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter import StatePartition
from bayesfilter.structural_tf import make_affine_structural_tf
from bayesfilter.testing import svd_cut_branch_frequency_summary


pytestmark = pytest.mark.skipif(
    os.environ.get("BAYESFILTER_RUN_EXTENDED_CPU") != "1",
    reason="SVD-CUT branch-frequency diagnostics are opt-in extended CPU tests.",
)


def _model_from_params(params: tf.Tensor, *, repeated_spectrum: bool = False):
    phi1 = params[0]
    sigma = params[1]
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    initial_covariance = (
        tf.eye(2, dtype=tf.float64)
        if repeated_spectrum
        else tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64))
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.2], dtype=tf.float64),
        initial_covariance=initial_covariance,
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.stack(
            [
                tf.stack([phi1, tf.constant(-0.12, dtype=tf.float64)]),
                tf.constant([1.0, 0.0], dtype=tf.float64),
            ]
        ),
        innovation_matrix=tf.reshape(
            tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]),
            [2, 1],
        ),
        innovation_covariance=tf.constant([[0.43]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.25]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )


@pytest.mark.extended
def test_svd_cut_branch_frequency_summary_quantifies_smooth_parameter_box() -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    parameter_grid = tf.constant(
        [
            [0.27, 0.23],
            [0.31, 0.27],
            [0.35, 0.31],
        ],
        dtype=tf.float64,
    )

    summary = svd_cut_branch_frequency_summary(
        observations,
        parameter_grid,
        _model_from_params,
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )

    assert summary.total_count == 3
    assert summary.smooth_count == 3
    assert summary.active_floor_count == 0
    assert summary.weak_spectral_gap_count == 0
    assert summary.nonfinite_count == 0
    assert summary.other_blocked_count == 0
    assert summary.smooth_fraction == 1.0
    assert summary.max_point_count > 0
    assert summary.max_integration_rank > 0
    assert np.isfinite(summary.min_placement_eigen_gap)
    assert np.isfinite(summary.min_innovation_eigen_gap) or np.isinf(
        summary.min_innovation_eigen_gap
    )


@pytest.mark.extended
def test_svd_cut_branch_frequency_summary_counts_blocked_weak_gap() -> None:
    observations = tf.constant([[0.2]], dtype=tf.float64)
    parameter_grid = tf.constant([[0.31, 0.27], [0.32, 0.28]], dtype=tf.float64)

    def repeated_builder(values: tf.Tensor):
        return _model_from_params(values, repeated_spectrum=True)

    summary = svd_cut_branch_frequency_summary(
        observations,
        parameter_grid,
        repeated_builder,
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )

    assert summary.total_count == 2
    assert summary.smooth_count == 0
    assert summary.weak_spectral_gap_count == 2
    assert summary.smooth_fraction == 0.0
