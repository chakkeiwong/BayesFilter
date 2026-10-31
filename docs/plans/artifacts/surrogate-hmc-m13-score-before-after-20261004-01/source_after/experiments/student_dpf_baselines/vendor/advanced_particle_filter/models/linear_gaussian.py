"""
Linear Gaussian State Space Model.

x_t = A @ x_{t-1} + v_t,  v_t ~ N(0, Q)
y_t = C @ x_t + w_t,      w_t ~ N(0, R)
"""

import numpy as np
from .base import StateSpaceModel


def make_lgssm(
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> StateSpaceModel:
    """
    Create a Linear Gaussian State Space Model.
    
    Dynamics:    x_t = A @ x_{t-1} + v_t,  v_t ~ N(0, Q)
    Observation: y_t = C @ x_t + w_t,      w_t ~ N(0, R)
    
    Args:
        A: [nx, nx] State transition matrix
        C: [ny, nx] Observation matrix
        Q: [nx, nx] Process noise covariance
        R: [ny, ny] Observation noise covariance
        m0: [nx] Initial state mean
        P0: [nx, nx] Initial state covariance
        
    Returns:
        StateSpaceModel instance
    """
    # Convert to numpy arrays
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    P0 = np.asarray(P0, dtype=np.float64)
    
    nx = A.shape[0]
    ny = C.shape[0]
    
    # Ensure symmetry
    Q = 0.5 * (Q + Q.T)
    R = 0.5 * (R + R.T)
    P0 = 0.5 * (P0 + P0.T)
    
    # Define dynamics functions
    def dynamics_mean(x: np.ndarray) -> np.ndarray:
        """x: [N, nx] -> [N, nx]"""
        return x @ A.T
    
    def dynamics_jacobian(x: np.ndarray) -> np.ndarray:
        """x: [nx] -> [nx, nx]"""
        return A
    
    # Define observation functions
    def obs_mean(x: np.ndarray) -> np.ndarray:
        """x: [N, nx] -> [N, ny]"""
        return x @ C.T
    
    def obs_jacobian(x: np.ndarray) -> np.ndarray:
        """x: [nx] -> [ny, nx]"""
        return C
    
    return StateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0,
        initial_cov=P0,
        dynamics_mean=dynamics_mean,
        dynamics_cov=Q,
        dynamics_jacobian=dynamics_jacobian,
        obs_mean=obs_mean,
        obs_cov=R,
        obs_jacobian=obs_jacobian,
        # obs_log_prob=None -> uses default Gaussian
        # obs_sample=None -> uses default Gaussian
    )


def make_lgssm_from_chol(
    A: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> StateSpaceModel:
    """
    Create LGSSM from noise Cholesky factors.
    
    Q = B @ B.T
    R = D @ D.T
    
    Args:
        A: [nx, nx] State transition matrix
        C: [ny, nx] Observation matrix
        B: [nx, nv] Process noise factor (Q = B @ B.T)
        D: [ny, nw] Observation noise factor (R = D @ D.T)
        m0: [nx] Initial state mean
        P0: [nx, nx] Initial state covariance
        
    Returns:
        StateSpaceModel instance
    """
    B = np.asarray(B, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    
    Q = B @ B.T
    R = D @ D.T
    
    return make_lgssm(A, C, Q, R, m0, P0)
