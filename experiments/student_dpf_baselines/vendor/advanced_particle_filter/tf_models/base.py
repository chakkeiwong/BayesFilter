"""
TensorFlow State Space Model base class.

Mirrors: models/base.py (NumPy version)

All functions operate on batched inputs where first axis is batch dimension.
All callables must be tf.function-compatible (no NumPy, no Python control flow
that depends on tensor values).

Design decisions for JIT compatibility:
  - No dataclass (tf.function traces through __init__; we use a plain class)
  - All arrays are tf.Tensor (float64 by default for numerical parity with NumPy)
  - Cholesky factors precomputed once, stored as tf.constant
  - Callables (dynamics_mean, obs_mean, etc.) must accept/return tf.Tensor
"""

import tensorflow as tf
from typing import Callable, Optional


class TFStateSpaceModel:
    """
    TensorFlow State Space Model specification.

    Dynamics:    x_t = f(x_{t-1}) + v_t,  v_t ~ N(0, Q)
    Observation: y_t = h(x_t) + w_t,      w_t ~ N(0, R) or custom

    All mean functions take batched input [N, nx] and return [N, *].
    Jacobian functions take single input [nx] and return matrix.

    Attributes:
        state_dim: int
        obs_dim: int
        initial_mean: [nx] tf.Tensor
        initial_cov: [nx, nx] tf.Tensor
        dynamics_mean: f(x), maps [N, nx] -> [N, nx]
        dynamics_cov: [nx, nx] tf.Tensor  (Q)
        dynamics_jacobian: F(x) = df/dx, maps [nx] -> [nx, nx]
        obs_mean: h(x), maps [N, nx] -> [N, ny]
        obs_cov: [ny, ny] tf.Tensor  (R)
        obs_jacobian: H(x) = dh/dx, maps [nx] -> [ny, nx]
        obs_log_prob: Optional custom log p(y|x), maps ([N, nx], [ny]) -> [N]
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        initial_mean: tf.Tensor,
        initial_cov: tf.Tensor,
        dynamics_mean: Callable[[tf.Tensor], tf.Tensor],
        dynamics_cov: tf.Tensor,
        dynamics_jacobian: Callable[[tf.Tensor], tf.Tensor],
        obs_mean: Callable[[tf.Tensor], tf.Tensor],
        obs_cov: tf.Tensor,
        obs_jacobian: Callable[[tf.Tensor], tf.Tensor],
        obs_log_prob: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
        dtype: tf.DType = tf.float64,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dtype = dtype

        # Store as tf.constant for graph embedding
        self.initial_mean = tf.constant(initial_mean, dtype=dtype)
        self.initial_cov = tf.constant(initial_cov, dtype=dtype)
        self.dynamics_cov = tf.constant(dynamics_cov, dtype=dtype)
        self.obs_cov = tf.constant(obs_cov, dtype=dtype)

        # Callables (must be tf.function-friendly)
        self.dynamics_mean = dynamics_mean
        self.dynamics_jacobian = dynamics_jacobian
        self.obs_mean = obs_mean
        self.obs_jacobian = obs_jacobian
        self.obs_log_prob = obs_log_prob

        # Precompute Cholesky factors and related quantities
        eps = tf.constant(1e-8, dtype=dtype)
        eye_nx = tf.eye(state_dim, dtype=dtype)
        eye_ny = tf.eye(obs_dim, dtype=dtype)

        P0 = self.initial_cov + eps * eye_nx
        self._initial_cov_chol = tf.linalg.cholesky(P0)

        Q = self.dynamics_cov + eps * eye_nx
        self._dynamics_cov_chol = tf.linalg.cholesky(Q)

        R = self.obs_cov + eps * eye_ny
        self._obs_cov_chol = tf.linalg.cholesky(R)
        self._obs_cov_inv = tf.linalg.inv(R)
        self._obs_cov_logdet = tf.linalg.slogdet(R)[1]

    # -----------------------------------------------------------------
    # Sampling methods
    # -----------------------------------------------------------------

    def sample_initial(self, n: int, rng: tf.random.Generator) -> tf.Tensor:
        """
        Sample n particles from initial distribution.

        Args:
            n: Number of samples
            rng: tf.random.Generator

        Returns:
            x: [n, nx] samples
        """
        noise = rng.normal(shape=[n, self.state_dim], dtype=self.dtype)
        # x = m0 + noise @ L.T  where P0 = L @ L.T
        return self.initial_mean + tf.linalg.matvec(
            tf.transpose(self._initial_cov_chol), noise
        )  # broadcasting: noise @ L.T  <=>  (L @ noise.T).T

    def sample_dynamics(self, x: tf.Tensor, rng: tf.random.Generator) -> tf.Tensor:
        """
        Sample x_t from p(x_t | x_{t-1}).

        Args:
            x: [N, nx] current states
            rng: tf.random.Generator

        Returns:
            x_next: [N, nx] next states
        """
        n = tf.shape(x)[0]
        mean = self.dynamics_mean(x)  # [N, nx]
        noise = rng.normal(shape=[n, self.state_dim], dtype=self.dtype)
        return mean + noise @ tf.transpose(self._dynamics_cov_chol)

    def sample_observation(self, x: tf.Tensor, rng: tf.random.Generator) -> tf.Tensor:
        """
        Sample y from p(y | x).

        Args:
            x: [nx] single state (not batched)
            rng: tf.random.Generator

        Returns:
            y: [ny] observation
        """
        mean = self.obs_mean(x[tf.newaxis, :])[0]  # [ny]
        noise = rng.normal(shape=[self.obs_dim], dtype=self.dtype)
        return mean + tf.linalg.matvec(self._obs_cov_chol, noise)

    # -----------------------------------------------------------------
    # Log-probability methods
    # -----------------------------------------------------------------

    def observation_log_prob(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(y | x) for all particles.

        Args:
            x: [N, nx] particles
            y: [ny] single observation

        Returns:
            log_prob: [N]
        """
        if self.obs_log_prob is not None:
            return self.obs_log_prob(x, y)
        return self._gaussian_obs_log_prob(x, y)

    def _gaussian_obs_log_prob(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Gaussian observation log-likelihood.

        Args:
            x: [N, nx] particles
            y: [ny] observation

        Returns:
            log_prob: [N]
        """
        y_pred = self.obs_mean(x)  # [N, ny]
        residual = y - y_pred  # [N, ny]

        # Solve L @ z = residual.T  =>  z = L^{-1} residual.T
        # residual.T is [ny, N], solved is [ny, N]
        solved = tf.linalg.triangular_solve(
            self._obs_cov_chol,
            tf.transpose(residual),
            lower=True,
        )
        mahal_sq = tf.reduce_sum(solved ** 2, axis=0)  # [N]

        ny_f = tf.cast(self.obs_dim, self.dtype)
        pi = tf.constant(3.141592653589793, dtype=self.dtype)
        log_prob = -0.5 * (ny_f * tf.math.log(2.0 * pi) + self._obs_cov_logdet + mahal_sq)
        return log_prob

    def dynamics_log_prob(self, x_next: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(x_next | x) for all particles.

        Args:
            x_next: [N, nx] next states
            x: [N, nx] current states

        Returns:
            log_prob: [N]
        """
        mean = self.dynamics_mean(x)  # [N, nx]
        residual = x_next - mean  # [N, nx]

        solved = tf.linalg.triangular_solve(
            self._dynamics_cov_chol,
            tf.transpose(residual),
            lower=True,
        )
        mahal_sq = tf.reduce_sum(solved ** 2, axis=0)  # [N]

        Q_logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(self._dynamics_cov_chol))
        )
        nx_f = tf.cast(self.state_dim, self.dtype)
        pi = tf.constant(3.141592653589793, dtype=self.dtype)
        log_prob = -0.5 * (nx_f * tf.math.log(2.0 * pi) + Q_logdet + mahal_sq)
        return log_prob

    # -----------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------

    def simulate(self, T: int, rng: tf.random.Generator):
        """
        Simulate a trajectory from the model.
        Retracing-defensive implementation.
        Args:
            T: Number of time steps
            rng: tf.random.Generator

        Returns:
            states: [T+1, nx] states (x_0, ..., x_T)
            observations: [T, ny] observations (y_1, ..., y_T)
        """
        # Use TensorArray for dynamic writes inside tf.function
        states_ta = tf.TensorArray(dtype=self.dtype, size=T + 1, dynamic_size=False)
        obs_ta = tf.TensorArray(dtype=self.dtype, size=T, dynamic_size=False)

        # Initial state
        x0 = self.sample_initial(1, rng)[0]  # [nx]
        states_ta = states_ta.write(0, x0)

        x_curr = x0
        for t in tf.range(T):
            x_next = self.sample_dynamics(x_curr[tf.newaxis, :], rng)[0]
            y = self.sample_observation(x_next, rng)
            states_ta = states_ta.write(t + 1, x_next)
            obs_ta = obs_ta.write(t, y)
            x_curr = x_next

        return states_ta.stack(), obs_ta.stack()

    def __repr__(self) -> str:
        return f"TFStateSpaceModel(nx={self.state_dim}, ny={self.obs_dim})"
