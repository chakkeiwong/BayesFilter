"""
Kalman Filter implementations.

- KalmanFilter: Standard Kalman filter for linear Gaussian models
- ExtendedKalmanFilter: EKF for nonlinear models with Gaussian noise
- UnscentedKalmanFilter: UKF using sigma points (2n+1 scheme)
"""

import numpy as np
from typing import Optional
from numpy.random import Generator, default_rng

from .base import FilterResult
from ..models.base import StateSpaceModel


class KalmanFilter:
    """
    Standard Kalman Filter for linear Gaussian models.
    
    Optimal for linear dynamics and linear observations with Gaussian noise.
    """
    
    def __init__(self):
        pass
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
    ) -> FilterResult:
        """
        Run Kalman filter on observations.
        
        Args:
            model: StateSpaceModel (should be linear Gaussian)
            observations: [T, ny] Observations (y_1, ..., y_T)
            
        Returns:
            FilterResult with filtered means and covariances
        """
        T = observations.shape[0]
        nx = model.state_dim
        ny = model.obs_dim
        
        # Get model matrices (assumes linear model)
        # F = dynamics Jacobian (constant for linear)
        # H = observation Jacobian (constant for linear)
        dummy_x = np.zeros(nx)
        F = model.dynamics_jacobian(dummy_x)
        H = model.obs_jacobian(dummy_x)
        Q = model.dynamics_cov
        R = model.obs_cov
        
        # Storage
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        log_likelihoods = np.zeros(T)
        
        # Initialize
        means[0] = model.initial_mean
        covariances[0] = model.initial_cov
        
        for t in range(T):
            y = observations[t]
            m_prev = means[t]
            P_prev = covariances[t]
            
            # Predict
            m_pred = F @ m_prev
            P_pred = F @ P_prev @ F.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)  # Symmetrize
            
            # Update
            m_upd, P_upd, log_lik = self._update(m_pred, P_pred, y, H, R)
            
            means[t + 1] = m_upd
            covariances[t + 1] = P_upd
            log_likelihoods[t] = log_lik
        
        return FilterResult(
            means=means,
            covariances=covariances,
            log_likelihood=np.sum(log_likelihoods),
            log_likelihood_increments=log_likelihoods,
        )
    
    def _update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> tuple:
        """
        Kalman update step.
        
        Args:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
            y: [ny] Observation
            H: [ny, nx] Observation matrix
            R: [ny, ny] Observation noise covariance
            
        Returns:
            m_upd: [nx] Updated mean
            P_upd: [nx, nx] Updated covariance
            log_lik: Log likelihood of observation
        """
        nx = len(m_pred)
        ny = len(y)
        
        # Innovation
        y_pred = H @ m_pred
        v = y - y_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        
        # Kalman gain: K = P_pred @ H.T @ S^{-1}
        # Solve S @ K.T = H @ P_pred.T for K.T
        K = np.linalg.solve(S, H @ P_pred).T
        
        # Updated mean
        m_upd = m_pred + K @ v
        
        # Updated covariance (Joseph form for numerical stability)
        I = np.eye(nx)
        IKH = I - K @ H
        P_upd = IKH @ P_pred @ IKH.T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        
        # Log likelihood
        S_chol = np.linalg.cholesky(S)
        S_logdet = 2.0 * np.sum(np.log(np.diag(S_chol)))
        solved = np.linalg.solve(S_chol, v)
        mahal_sq = np.sum(solved ** 2)
        log_lik = -0.5 * (ny * np.log(2 * np.pi) + S_logdet + mahal_sq)
        
        return m_upd, P_upd, log_lik
    
    def predict(
        self,
        m: np.ndarray,
        P: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        Single prediction step.
        
        Args:
            m: [nx] Current mean
            P: [nx, nx] Current covariance
            model: StateSpaceModel
            
        Returns:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
        """
        F = model.dynamics_jacobian(m)
        Q = model.dynamics_cov
        
        m_pred = F @ m
        P_pred = F @ P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        return m_pred, P_pred
    
    def update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        Single update step.
        
        Args:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
            y: [ny] Observation
            model: StateSpaceModel
            
        Returns:
            m_upd: [nx] Updated mean
            P_upd: [nx, nx] Updated covariance
        """
        H = model.obs_jacobian(m_pred)
        R = model.obs_cov
        
        m_upd, P_upd, _ = self._update(m_pred, P_pred, y, H, R)
        return m_upd, P_upd


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear models.
    
    Linearizes dynamics and observation functions around current estimate.
    Works with Gaussian noise only.
    """
    
    def __init__(self):
        pass
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
    ) -> FilterResult:
        """
        Run EKF on observations.
        
        Args:
            model: StateSpaceModel with dynamics_jacobian and obs_jacobian
            observations: [T, ny] Observations
            
        Returns:
            FilterResult
        """
        T = observations.shape[0]
        nx = model.state_dim
        
        Q = model.dynamics_cov
        R = model.obs_cov
        
        # Storage
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        log_likelihoods = np.zeros(T)
        
        # Initialize
        means[0] = model.initial_mean
        covariances[0] = model.initial_cov
        
        for t in range(T):
            y = observations[t]
            m_prev = means[t]
            P_prev = covariances[t]
            
            # Predict
            m_pred, P_pred = self.predict(m_prev, P_prev, model)
            
            # Update
            m_upd, P_upd, log_lik = self._update(m_pred, P_pred, y, model)
            
            means[t + 1] = m_upd
            covariances[t + 1] = P_upd
            log_likelihoods[t] = log_lik
        
        return FilterResult(
            means=means,
            covariances=covariances,
            log_likelihood=np.sum(log_likelihoods),
            log_likelihood_increments=log_likelihoods,
        )
    
    def predict(
        self,
        m: np.ndarray,
        P: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        EKF prediction step.
        
        Args:
            m: [nx] Current mean
            P: [nx, nx] Current covariance
            model: StateSpaceModel
            
        Returns:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
        """
        # Linearize dynamics at current mean
        F = model.dynamics_jacobian(m)
        Q = model.dynamics_cov
        
        # Predicted mean: f(m)
        m_pred = model.dynamics_mean(m[np.newaxis, :])[0]
        
        # Predicted covariance
        P_pred = F @ P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        return m_pred, P_pred
    
    def update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        EKF update step.
        
        Args:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
            y: [ny] Observation
            model: StateSpaceModel
            
        Returns:
            m_upd: [nx] Updated mean
            P_upd: [nx, nx] Updated covariance
        """
        m_upd, P_upd, _ = self._update(m_pred, P_pred, y, model)
        return m_upd, P_upd
    
    def _update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """Internal update with log likelihood."""
        nx = len(m_pred)
        ny = len(y)
        
        # Linearize observation at predicted mean
        H = model.obs_jacobian(m_pred)
        R = model.obs_cov
        
        # Predicted observation: h(m_pred)
        y_pred = model.obs_mean(m_pred[np.newaxis, :])[0]
        
        # Innovation
        v = y - y_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        
        # Kalman gain
        K = np.linalg.solve(S, H @ P_pred).T
        
        # Update
        m_upd = m_pred + K @ v
        
        # Joseph form
        I = np.eye(nx)
        IKH = I - K @ H
        P_upd = IKH @ P_pred @ IKH.T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        
        # Log likelihood
        S_chol = np.linalg.cholesky(S)
        S_logdet = 2.0 * np.sum(np.log(np.diag(S_chol)))
        solved = np.linalg.solve(S_chol, v)
        mahal_sq = np.sum(solved ** 2)
        log_lik = -0.5 * (ny * np.log(2 * np.pi) + S_logdet + mahal_sq)
        
        return m_upd, P_upd, log_lik


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter using sigma points.
    
    Uses 2n+1 sigma points for better nonlinear approximation than EKF.
    Standard unscented transform with parameters alpha, beta, kappa.
    
    Default parameters match the ekfukf MATLAB toolbox commonly used
    in particle flow literature.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: Optional[float] = None,
    ):
        """
        Standard Unscented Kalman Filter.
        
        Default parameters (alpha=1, beta=2, kappa=3-n) are recommended for
        Gaussian distributions and match the ekfukf MATLAB toolbox.
        
        Args:
            alpha: Spread of sigma points. Default 1.0.
            beta: Prior knowledge parameter. Default 2.0 for Gaussian.
            kappa: Secondary scaling parameter. Default 3-n (set dynamically).
                   If None, will be computed as 3-n where n is state dimension.
        """
        self.alpha = alpha
        self.beta = beta
        self._kappa = kappa  # None means compute dynamically as 3-n
    
    def _compute_weights(self, n: int) -> tuple:
        """
        Compute sigma point weights.
        
        Args:
            n: State dimension
            
        Returns:
            Wm: [2n+1] Weights for mean
            Wc: [2n+1] Weights for covariance
            lam: Lambda parameter
        """
        # Use kappa=3-n if not specified (common choice for Gaussian)
        kappa = self._kappa if self._kappa is not None else (3 - n)
        
        lam = self.alpha ** 2 * (n + kappa) - n
        
        # Mean weights
        Wm = np.full(2 * n + 1, 0.5 / (n + lam))
        Wm[0] = lam / (n + lam)
        
        # Covariance weights
        Wc = Wm.copy()
        Wc[0] = Wm[0] + (1 - self.alpha ** 2 + self.beta)
        
        return Wm, Wc, lam
    
    def _sigma_points(
        self,
        m: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray:
        """
        Generate sigma points.
        
        Args:
            m: [n] Mean
            P: [n, n] Covariance
            
        Returns:
            sigma: [2n+1, n] Sigma points
        """
        n = len(m)
        _, _, lam = self._compute_weights(n)
        
        # Scaling factor: (n + lambda)
        # This can be negative for small alpha, so we take abs and handle carefully
        scaling = n + lam
        if scaling <= 0:
            # Fallback to standard scaling when lambda is too negative
            scaling = n  # Equivalent to alpha=1, kappa=0
        
        # Ensure P is positive definite with regularization
        P_reg = 0.5 * (P + P.T)  # Symmetrize
        min_eig = np.min(np.linalg.eigvalsh(P_reg))
        if min_eig < 1e-10:
            P_reg = P_reg + (1e-6 - min(min_eig, 0)) * np.eye(n)
        
        # Square root of scaling * P via Cholesky
        # L @ L.T = scaling * P
        try:
            L = np.linalg.cholesky(scaling * P_reg)
        except np.linalg.LinAlgError:
            # Last resort: use eigendecomposition for square root
            eigvals, eigvecs = np.linalg.eigh(P_reg)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(scaling * eigvals))
        
        # Sigma points using COLUMNS of L (not rows!)
        # sigma[i+1] = m + L[:, i] for i = 0..n-1
        # sigma[n+i+1] = m - L[:, i] for i = 0..n-1
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = m
        for i in range(n):
            sigma[i + 1] = m + L[:, i]
            sigma[n + i + 1] = m - L[:, i]
        
        return sigma
    
    def filter(
        self,
        model: StateSpaceModel,
        observations: np.ndarray,
    ) -> FilterResult:
        """
        Run UKF on observations.
        
        Args:
            model: StateSpaceModel
            observations: [T, ny] Observations
            
        Returns:
            FilterResult
        """
        T = observations.shape[0]
        nx = model.state_dim
        
        # Storage
        means = np.zeros((T + 1, nx))
        covariances = np.zeros((T + 1, nx, nx))
        log_likelihoods = np.zeros(T)
        
        # Initialize
        means[0] = model.initial_mean
        covariances[0] = model.initial_cov
        
        for t in range(T):
            y = observations[t]
            m_prev = means[t]
            P_prev = covariances[t]
            
            # Predict
            m_pred, P_pred = self.predict(m_prev, P_prev, model)
            
            # Update
            m_upd, P_upd, log_lik = self._update(m_pred, P_pred, y, model)
            
            means[t + 1] = m_upd
            covariances[t + 1] = P_upd
            log_likelihoods[t] = log_lik
        
        return FilterResult(
            means=means,
            covariances=covariances,
            log_likelihood=np.sum(log_likelihoods),
            log_likelihood_increments=log_likelihoods,
        )
    
    def predict(
        self,
        m: np.ndarray,
        P: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        UKF prediction step.
        
        Args:
            m: [nx] Current mean
            P: [nx, nx] Current covariance
            model: StateSpaceModel
            
        Returns:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
        """
        nx = len(m)
        Q = model.dynamics_cov
        
        Wm, Wc, _ = self._compute_weights(nx)
        
        # Generate sigma points
        sigma = self._sigma_points(m, P)  # [2n+1, nx]
        
        # Propagate through dynamics
        sigma_pred = model.dynamics_mean(sigma)  # [2n+1, nx]
        
        # Predicted mean
        m_pred = np.sum(Wm[:, np.newaxis] * sigma_pred, axis=0)
        
        # Predicted covariance
        diff = sigma_pred - m_pred
        P_pred = np.einsum('i,ij,ik->jk', Wc, diff, diff) + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        return m_pred, P_pred
    
    def update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """
        UKF update step.
        
        Args:
            m_pred: [nx] Predicted mean
            P_pred: [nx, nx] Predicted covariance
            y: [ny] Observation
            model: StateSpaceModel
            
        Returns:
            m_upd: [nx] Updated mean
            P_upd: [nx, nx] Updated covariance
        """
        m_upd, P_upd, _ = self._update(m_pred, P_pred, y, model)
        return m_upd, P_upd
    
    def _update(
        self,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        model: StateSpaceModel,
    ) -> tuple:
        """Internal update with log likelihood.
        
        Uses numerically stable Cholesky-based update.
        """
        nx = len(m_pred)
        ny = len(y)
        R = model.obs_cov
        
        Wm, Wc, _ = self._compute_weights(nx)
        
        # Generate sigma points from predicted distribution
        sigma = self._sigma_points(m_pred, P_pred)  # [2n+1, nx]
        
        # Propagate through observation function
        sigma_y = model.obs_mean(sigma)  # [2n+1, ny]
        
        # Predicted observation
        y_pred = np.sum(Wm[:, np.newaxis] * sigma_y, axis=0)
        
        # Innovation
        v = y - y_pred
        
        # Cross-covariance and innovation covariance
        diff_x = sigma - m_pred  # [2n+1, nx]
        diff_y = sigma_y - y_pred  # [2n+1, ny]
        
        Pxy = np.einsum('i,ij,ik->jk', Wc, diff_x, diff_y)  # [nx, ny]
        S = np.einsum('i,ij,ik->jk', Wc, diff_y, diff_y) + R  # [ny, ny]
        S = 0.5 * (S + S.T)
        
        # Cholesky of innovation covariance (with regularization)
        try:
            S_chol = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S_chol = np.linalg.cholesky(S + 1e-6 * np.eye(ny))
        
        # Kalman gain: K = Pxy @ S^{-1}, where K is [nx, ny]
        # Solve S @ K.T = Pxy.T for K.T, then transpose
        # S_chol @ S_chol.T @ K.T = Pxy.T
        # First solve S_chol @ temp = Pxy.T for temp [ny, nx]
        # Then solve S_chol.T @ K.T = temp for K.T [ny, nx]
        temp = np.linalg.solve(S_chol, Pxy.T)  # [ny, nx]
        K = np.linalg.solve(S_chol.T, temp).T  # [nx, ny]
        
        # Updated mean
        m_upd = m_pred + K @ v
        
        # Updated covariance using stable Cholesky-based formula
        # P_upd = P_pred - K @ S @ K.T
        # Rewrite: K @ S @ K.T = K @ S_chol @ S_chol.T @ K.T = (K @ S_chol) @ (K @ S_chol).T
        KS_chol = K @ S_chol  # [nx, ny]
        P_upd = P_pred - KS_chol @ KS_chol.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(P_upd))
        if min_eig < 0:
            P_upd = P_upd + (1e-6 - min_eig) * np.eye(nx)
        
        # Log likelihood
        S_logdet = 2.0 * np.sum(np.log(np.diag(S_chol)))
        solved = np.linalg.solve(S_chol, v)
        mahal_sq = np.sum(solved ** 2)
        log_lik = -0.5 * (ny * np.log(2 * np.pi) + S_logdet + mahal_sq)
        
        return m_upd, P_upd, log_lik
