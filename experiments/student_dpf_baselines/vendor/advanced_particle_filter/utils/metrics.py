"""
Evaluation metrics for filtering and tracking.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Optional


def compute_omat(
    xs_true: np.ndarray,
    xs_est: np.ndarray,
    num_targets: int = 4,
    p: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Compute Optimal Mass Transfer (OMAT) metric for multi-target tracking.
    
    The OMAT metric finds the optimal assignment between true and estimated
    target positions using the Hungarian algorithm, then computes the average
    distance error.
    
    Reference: Schuhmacher et al. (2008), "A consistent metric for performance
    evaluation of multi-object filters"
    
    Args:
        xs_true: [T+1, nx] True states (including initial)
        xs_est: [T+1, nx] or [T, nx] Estimated states
        num_targets: Number of targets (C)
        p: Power parameter (default 1)
        
    Returns:
        omat_per_step: [T] OMAT at each time step (excluding initial if aligned)
        omat_mean: Mean OMAT over all time steps
    """
    # Handle case where xs_est doesn't include initial state
    if xs_est.shape[0] == xs_true.shape[0] - 1:
        xs_true_aligned = xs_true[1:]  # Skip initial state
    else:
        xs_true_aligned = xs_true
    
    T = min(xs_true_aligned.shape[0], xs_est.shape[0])
    omat_per_step = np.zeros(T)
    
    for t in range(T):
        # Extract positions for each target
        pos_true = np.zeros((num_targets, 2))
        pos_est = np.zeros((num_targets, 2))
        
        for c in range(num_targets):
            pos_true[c, 0] = xs_true_aligned[t, 4*c]
            pos_true[c, 1] = xs_true_aligned[t, 4*c + 1]
            pos_est[c, 0] = xs_est[t, 4*c]
            pos_est[c, 1] = xs_est[t, 4*c + 1]
        
        # Build cost matrix: distance^p between all pairs
        cost_matrix = np.zeros((num_targets, num_targets))
        for i in range(num_targets):
            for j in range(num_targets):
                dist = np.sqrt(np.sum((pos_true[i] - pos_est[j])**2))
                cost_matrix[i, j] = dist ** p
        
        # Optimal assignment via Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        # OMAT: (1/C * sum of costs)^(1/p)
        omat_per_step[t] = (total_cost / num_targets) ** (1/p)
    
    return omat_per_step, np.mean(omat_per_step)


def compute_rmse(
    xs_true: np.ndarray,
    xs_est: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute Root Mean Square Error per time step.
    
    Args:
        xs_true: [T+1, nx] True states
        xs_est: [T+1, nx] or [T, nx] Estimated states
        
    Returns:
        rmse_per_step: [T] RMSE at each time step
        rmse_mean: Mean RMSE over all time steps
    """
    # Handle alignment
    if xs_est.shape[0] == xs_true.shape[0] - 1:
        xs_true_aligned = xs_true[1:]
    else:
        xs_true_aligned = xs_true
    
    T = min(xs_true_aligned.shape[0], xs_est.shape[0])
    
    rmse_per_step = np.sqrt(np.mean((xs_true_aligned[:T] - xs_est[:T])**2, axis=1))
    
    return rmse_per_step, np.mean(rmse_per_step)


def compute_position_rmse(
    xs_true: np.ndarray,
    xs_est: np.ndarray,
    num_targets: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Compute RMSE only for position components (not velocity).
    
    For multi-target tracking where state is [x1, y1, vx1, vy1, x2, y2, ...].
    
    Args:
        xs_true: [T+1, nx] True states
        xs_est: [T+1, nx] or [T, nx] Estimated states
        num_targets: Number of targets
        
    Returns:
        rmse_per_step: [T] Position RMSE at each time step
        rmse_mean: Mean position RMSE
    """
    # Handle alignment
    if xs_est.shape[0] == xs_true.shape[0] - 1:
        xs_true_aligned = xs_true[1:]
    else:
        xs_true_aligned = xs_true
    
    T = min(xs_true_aligned.shape[0], xs_est.shape[0])
    rmse_per_step = np.zeros(T)
    
    for t in range(T):
        pos_error_sq = 0.0
        for c in range(num_targets):
            dx = xs_true_aligned[t, 4*c] - xs_est[t, 4*c]
            dy = xs_true_aligned[t, 4*c + 1] - xs_est[t, 4*c + 1]
            pos_error_sq += dx**2 + dy**2
        rmse_per_step[t] = np.sqrt(pos_error_sq / num_targets)
    
    return rmse_per_step, np.mean(rmse_per_step)
