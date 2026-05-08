"""
Range-Bearing State Space Model with Student-t measurement noise.

State: [px, py, vx, vy] - position and velocity
Observation: [range, bearing] with Student-t noise
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
from numpy.random import Generator

from .base import StateSpaceModel


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _h_range_bearing(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Range-bearing observation function.
    
    Args:
        x: [N, 4] or [4] states with [px, py, vx, vy]
        eps: Small constant for numerical stability
        
    Returns:
        y: [N, 2] or [2] observations [range, bearing]
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]
    
    px, py = x[:, 0], x[:, 1]
    r = np.sqrt(px**2 + py**2 + eps)
    th = np.arctan2(py, px)
    
    y = np.stack([r, th], axis=-1)
    
    if single:
        return y[0]
    return y


def _H_jac_range_bearing(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Jacobian of range-bearing observation.
    
    Args:
        x: [4] state vector
        eps: Small constant for numerical stability
        
    Returns:
        H: [2, 4] Jacobian matrix
    """
    px, py = x[0], x[1]
    r2 = px**2 + py**2 + eps
    r = np.sqrt(r2)
    
    # d(range)/d(state)
    dr_dpx = px / r
    dr_dpy = py / r
    
    # d(bearing)/d(state)
    dth_dpx = -py / r2
    dth_dpy = px / r2
    
    H = np.array([
        [dr_dpx, dr_dpy, 0.0, 0.0],
        [dth_dpx, dth_dpy, 0.0, 0.0]
    ])
    
    return H


def make_range_bearing_ssm(
    dt: float = 0.01,
    q_diag: float = 0.01,
    nu: float = 2.0,
    s_r: float = 0.01,
    s_th: float = 0.01,
    m0: Tuple[float, ...] = (1.0, 0.5, 0.01, 0.01),
    P0_diag: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    eps: float = 1e-6,
) -> StateSpaceModel:
    """
    Range-Bearing SSM with Student-t measurement noise.
    
    State: x = [px, py, vx, vy]
    Dynamics: Constant velocity model (linear)
    Observation: [range, bearing] with Student-t noise
    
    Args:
        dt: Time step
        q_diag: Process noise diagonal value
        nu: Degrees of freedom for Student-t (nu=2 gives heavy tails)
        s_r: Scale for range noise
        s_th: Scale for bearing noise
        m0: Initial state mean (px, py, vx, vy)
        P0_diag: Initial state variance diagonal
        eps: Small constant for numerical stability
        
    Returns:
        StateSpaceModel instance
    """
    nx = 4
    ny = 2
    
    # Dynamics: constant velocity
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    
    # Process noise
    Q = np.diag(np.full(nx, q_diag))
    Q = 0.5 * (Q + Q.T) + eps * np.eye(nx)
    
    # Initial state
    m0_vec = np.array(m0)
    P0_mat = np.diag(np.array(P0_diag))
    P0_mat = 0.5 * (P0_mat + P0_mat.T) + eps * np.eye(nx)
    
    # Observation "covariance" for flow computation (Gaussian approximation)
    # Flow algorithms assume Gaussian; this is used for A(λ), b(λ) computation
    R = np.diag([s_r**2, s_th**2])
    
    # --- Custom observation functions for Student-t ---
    
    def obs_mean(x: np.ndarray) -> np.ndarray:
        """Observation mean function."""
        return _h_range_bearing(x, eps)
    
    def obs_jacobian(x: np.ndarray) -> np.ndarray:
        """Observation Jacobian at x."""
        return _H_jac_range_bearing(x, eps)
    
    def observation_log_prob(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Log probability of observation y given state x.
        
        Uses Student-t distribution for both range and bearing.
        
        Args:
            x: [N, 4] particles
            y: [2] observation [range, bearing]
            
        Returns:
            log_prob: [N] log-likelihoods
        """
        y_pred = _h_range_bearing(x, eps)
        
        # Range residual (no wrapping needed)
        res_r = y[0] - y_pred[:, 0]
        
        # Bearing residual (wrap to [-pi, pi])
        res_th = _wrap_angle(y[1] - y_pred[:, 1])
        
        # Student-t log-likelihoods
        loglik_r = stats.t.logpdf(res_r, df=nu, loc=0, scale=s_r)
        loglik_th = stats.t.logpdf(res_th, df=nu, loc=0, scale=s_th)
        
        return loglik_r + loglik_th
    
    def sample_observation(x: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Sample observation from Student-t distribution.
        
        Args:
            x: [4] single state
            rng: NumPy random generator
            
        Returns:
            y: [2] observation [range, bearing]
        """
        y_mean = _h_range_bearing(x, eps)
        
        # Sample Student-t noise using inverse CDF
        u_r = rng.uniform()
        u_th = rng.uniform()
        noise_r = stats.t.ppf(u_r, df=nu, loc=0, scale=s_r)
        noise_th = stats.t.ppf(u_th, df=nu, loc=0, scale=s_th)
        
        r = y_mean[0] + noise_r
        th = _wrap_angle(y_mean[1] + noise_th)
        
        return np.array([r, th])
    
    return StateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0_vec,
        initial_cov=P0_mat,
        # Dynamics (linear)
        dynamics_mean=lambda x: x @ A.T,
        dynamics_cov=Q,
        dynamics_jacobian=lambda x: A,
        # Observation (nonlinear with Student-t noise)
        obs_mean=obs_mean,
        obs_cov=R,  # Gaussian approximation for flow
        obs_jacobian=obs_jacobian,
        obs_log_prob=observation_log_prob,
        obs_sample=sample_observation,
    )
