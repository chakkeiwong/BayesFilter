"""
Metrics Module
==============
Implements accuracy and stability metrics for Bayesian filters.
"""

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def rmse(est, truth):
    """
    Root Mean Squared Error (RMSE).

    A standard measure of the difference between the estimated state
    and the ground truth.

    :param est: Estimated state vector or trajectory.
    :param truth: Ground truth state vector or trajectory.
    :return: Scalar RMSE value.
    """
    if isinstance(est, tf.Tensor): est = est.numpy()
    if isinstance(truth, tf.Tensor): truth = truth.numpy()
    return np.sqrt(np.mean((est - truth)**2))

def compute_ess(weights):
    """
    Effective Sample Size (ESS).

    Measures the degree of particle degeneracy in a filter.
    ESS = 1 / sum(weights^2).

    :param weights: Normalized importance weights [N].
    :return: Scalar ESS value.
    """
    if isinstance(weights, tf.Tensor): weights = weights.numpy()
    return 1.0 / np.sum(np.square(weights))

def compute_omat(x_true, x_est, n_targets=4):
    """
    Optimal Mass Transfer (OMAT) Metric.

    Specifically used for the Acoustic Tracking experiment (Li 17) where
    multiple targets are tracked. It uses linear sum assignment to find
    the best matching between predicted and true target positions.

    :param x_true: Ground truth targets [Target_Dim].
    :param x_est: Estimated targets [Target_Dim].
    :param n_targets: Number of targets being tracked.
    """
    if isinstance(x_true, tf.Tensor): x_true = x_true.numpy()
    if isinstance(x_est, tf.Tensor): x_est = x_est.numpy()

    # Reshape to extract (x, y) coordinates for each target
    pos_true = x_true.reshape(n_targets, -1)[:, :2]
    pos_est = x_est.reshape(n_targets, -1)[:, :2]

    from scipy.spatial.distance import cdist as sp_cdist
    # Calculate pairwise distances between all true and estimated targets
    C = sp_cdist(pos_true, pos_est)
    # Solve the assignment problem
    row, col = linear_sum_assignment(C)
    return C[row, col].mean()
