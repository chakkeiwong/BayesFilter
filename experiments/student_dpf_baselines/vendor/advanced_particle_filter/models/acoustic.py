"""
Multi-target Acoustic Tracking State Space Model.

Based on Li & Coates (2017): "Particle Filtering With Invertible Particle Flow"

State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...] for num_targets targets
Observations: Amplitude measurements at sensor grid
"""

import numpy as np
from typing import Tuple, Optional
from numpy.random import Generator

from .base import StateSpaceModel


def make_acoustic_ssm(
    num_targets: int = 4,
    num_sensors_per_side: int = 5,
    sensor_range: Tuple[float, float] = (0.0, 40.0),
    dt: float = 1.0,
    source_amplitude: float = 10.0,
    obs_noise_var: float = 0.01,
    m0_positions: Optional[np.ndarray] = None,
    m0_velocities: Optional[np.ndarray] = None,
    P0_diag: Optional[np.ndarray] = None,
    dist_offset: float = 0.1,
    Q: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> StateSpaceModel:
    """
    Multi-target Acoustic Tracking SSM from Li & Coates (2017).
    
    Each target emits sound with amplitude Ψ. Sensors measure the sum of
    attenuated amplitudes from all targets:
    
        z_s = Σ_c Ψ / (||p_c - R_s|| + d0) + w_s
    
    Args:
        num_targets: Number of acoustic sources (default 4)
        num_sensors_per_side: Grid size (total sensors = num_sensors_per_side^2)
        sensor_range: (min, max) for sensor grid positions
        dt: Time step
        source_amplitude: Ψ in paper (default 10.0)
        obs_noise_var: Observation noise variance σ_w^2 (default 0.01)
        m0_positions: [num_targets, 2] Initial positions (paper defaults if None)
        m0_velocities: [num_targets, 2] Initial velocities (paper defaults if None)
        P0_diag: [4] Initial covariance diagonal per target (default from paper)
        dist_offset: d0, prevents division by zero (default 0.1)
        Q: Process noise covariance. If None, uses realistic Q from paper.
        eps: Small constant for numerical stability
        
    Returns:
        StateSpaceModel instance with additional attributes:
            - sensor_locs: [num_sensors, 2] sensor positions
            - num_targets: number of targets
            - source_amplitude: Ψ
            - dist_offset: d0
    """
    nx = num_targets * 4
    num_sensors = num_sensors_per_side ** 2
    ny = num_sensors
    
    # --- Sensor grid ---
    sensor_1d = np.linspace(sensor_range[0], sensor_range[1], num_sensors_per_side)
    sensor_xx, sensor_yy = np.meshgrid(sensor_1d, sensor_1d)
    sensor_locs = np.stack([sensor_xx.ravel(), sensor_yy.ravel()], axis=1)  # [ny, 2]
    
    # --- Dynamics ---
    # Single target: constant velocity model
    A_single = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    A = np.kron(np.eye(num_targets), A_single)  # Block diagonal
    
    # Process noise (realistic, from paper)
    if Q is None:
        # Paper's Q_real: (1/20) * Gamma where Gamma has specific structure
        Gamma_single = np.array([
            [1/3, 0, 0.5, 0],
            [0, 1/3, 0, 0.5],
            [0.5, 0, 1, 0],
            [0, 0.5, 0, 1],
        ])
        Q_single = (1.0 / 20.0) * Gamma_single
        Q = np.kron(np.eye(num_targets), Q_single)
    Q = 0.5 * (Q + Q.T) + eps * np.eye(nx)
    
    # --- Initial state ---
    if m0_positions is None:
        m0_positions = np.array([
            [12, 6],
            [32, 32],
            [20, 13],
            [15, 35],
        ][:num_targets])
    
    if m0_velocities is None:
        m0_velocities = np.array([
            [0.001, 0.001],
            [-0.001, -0.005],
            [-0.1, 0.01],
            [0.002, 0.002],
        ][:num_targets])
    
    m0 = np.zeros(nx)
    for i in range(num_targets):
        m0[4*i] = m0_positions[i, 0]
        m0[4*i + 1] = m0_positions[i, 1]
        m0[4*i + 2] = m0_velocities[i, 0]
        m0[4*i + 3] = m0_velocities[i, 1]
    
    # Initial covariance
    if P0_diag is None:
        # Paper: std = [10, 10, 1, 1] per target
        P0_diag = np.array([100.0, 100.0, 1.0, 1.0])
    P0_full_diag = np.tile(P0_diag, num_targets)
    P0 = np.diag(P0_full_diag)
    P0 = 0.5 * (P0 + P0.T) + eps * np.eye(nx)
    
    # --- Observation noise ---
    R = obs_noise_var * np.eye(ny)
    
    # --- Observation functions ---
    
    def obs_mean(x: np.ndarray) -> np.ndarray:
        """
        Acoustic observation function.
        
        Args:
            x: [N, nx] or [nx] states
            
        Returns:
            y: [N, ny] or [ny] observations (amplitude at each sensor)
        """
        single = x.ndim == 1
        if single:
            x = x[None, :]
        
        N = x.shape[0]
        
        # Extract positions: [N, num_targets, 2]
        positions = np.zeros((N, num_targets, 2))
        for i in range(num_targets):
            positions[:, i, 0] = x[:, 4*i]
            positions[:, i, 1] = x[:, 4*i + 1]
        
        # Distance from each target to each sensor
        # positions: [N, num_targets, 1, 2]
        # sensor_locs: [1, 1, num_sensors, 2]
        diff = positions[:, :, None, :] - sensor_locs[None, None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + eps)  # [N, num_targets, num_sensors]
        
        # Amplitude contribution: Ψ / (dist + d0)
        contributions = source_amplitude / (dist + dist_offset)
        
        # Sum over targets
        y = np.sum(contributions, axis=1)  # [N, num_sensors]
        
        if single:
            return y[0]
        return y
    
    def obs_jacobian(x: np.ndarray) -> np.ndarray:
        """
        Jacobian of acoustic observation function.
        
        Args:
            x: [nx] state vector
            
        Returns:
            H: [ny, nx] Jacobian matrix
        """
        # Extract positions
        positions = np.zeros((num_targets, 2))
        for i in range(num_targets):
            positions[i, 0] = x[4*i]
            positions[i, 1] = x[4*i + 1]
        
        # Distances
        diff = positions[:, None, :] - sensor_locs[None, :, :]  # [num_targets, num_sensors, 2]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + eps)  # [num_targets, num_sensors]
        
        # Gradient: d/dp_c [Ψ / (||p_c - R_s|| + d0)]
        #         = -Ψ * (p_c - R_s) / (||p_c - R_s|| * (||p_c - R_s|| + d0)^2)
        denom = dist * ((dist + dist_offset) ** 2) + eps
        grad_coeff = -source_amplitude / denom  # [num_targets, num_sensors]
        grad_pos = grad_coeff[:, :, None] * diff  # [num_targets, num_sensors, 2]
        
        # Build Jacobian
        H = np.zeros((ny, nx))
        for i in range(num_targets):
            H[:, 4*i] = grad_pos[i, :, 0]      # d/dx_i
            H[:, 4*i + 1] = grad_pos[i, :, 1]  # d/dy_i
            # d/dvx_i = 0, d/dvy_i = 0
        
        return H
    
    # Create model
    model = StateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0,
        initial_cov=P0,
        # Dynamics (linear)
        dynamics_mean=lambda x: x @ A.T,
        dynamics_cov=Q,
        dynamics_jacobian=lambda x: A,
        # Observation (nonlinear, Gaussian noise)
        obs_mean=obs_mean,
        obs_cov=R,
        obs_jacobian=obs_jacobian,
    )
    
    # Store additional attributes for reference
    model.sensor_locs = sensor_locs
    model.num_targets = num_targets
    model.source_amplitude = source_amplitude
    model.dist_offset = dist_offset
    model.A = A  # Store for easy access
    
    return model


def make_acoustic_Q_filter(num_targets: int = 4, eps: float = 1e-6) -> np.ndarray:
    """
    Create the inflated Q matrix used for filtering in Li & Coates (2017).
    
    The paper uses a larger Q for filtering to account for model uncertainty.
    This can be passed to filter methods via Q_override parameter.
    
    Args:
        num_targets: Number of targets
        eps: Small constant for numerical stability
        
    Returns:
        Q_filter: [nx, nx] inflated process noise covariance
    """
    nx = num_targets * 4
    
    Q_filter_single = np.array([
        [3.0, 0.0, 0.1, 0.0],
        [0.0, 3.0, 0.0, 0.1],
        [0.1, 0.0, 0.03, 0.0],
        [0.0, 0.1, 0.0, 0.03],
    ])
    
    Q_filter = np.kron(np.eye(num_targets), Q_filter_single)
    Q_filter = 0.5 * (Q_filter + Q_filter.T) + eps * np.eye(nx)
    
    return Q_filter
