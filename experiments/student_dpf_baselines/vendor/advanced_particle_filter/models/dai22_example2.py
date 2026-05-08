"""
Dai & Daum (2022) Example 2: 3D Angle-Only Target Tracking.

9-dimensional state (position, velocity, acceleration) with a single
angle-measuring sensor at the origin providing azimuth and elevation.

Dynamics are deterministic (continuous-time, discretized via matrix exponential).
Observation model is nonlinear (arctan).

Reference:
    Dai & Daum, "Stiffness Mitigation in Stochastic Particle Flow Filters",
    IEEE Trans. Aerosp. Electron. Syst., Vol. 58, No. 4, August 2022.
"""

import numpy as np
from scipy.linalg import expm
from typing import Optional
from numpy.random import Generator

from .base import StateSpaceModel


def make_dai22_example2_ssm(
    dt: float = 1.0,
    epsilon: float = 1e-2,
    R_diag: float = 1e-6,
    Q_diag: float = 1e-12,
    eps: float = 1e-12,
) -> StateSpaceModel:
    """
    Create the state space model for Dai (2022) Example 2.

    State: s = [x, y, z, vx, vy, vz, ax, ay, az]^T ∈ R^9
    Dynamics: ds = A s dt  (deterministic, damped constant-acceleration)
    Observation: z = [arctan(x/y), arctan(z/r)]^T + v,  v ~ N(0, R)
                 where r = sqrt(x^2 + y^2)

    Args:
        dt: Discrete time step (default 1.0)
        epsilon: Damping constant (default 1e-2)
        R_diag: Observation noise variance (default 1e-6, very informative)
        Q_diag: Nominal process noise for the SSM interface.
                Set very small since dynamics are deterministic.
                This is NOT the flow diffusion Q; it's only needed so that
                dynamics_cov is valid for the base class.
        eps: Numerical stability constant

    Returns:
        StateSpaceModel with additional attributes:
            - A_cont: [9, 9] continuous-time dynamics matrix
            - Phi: [9, 9] discrete-time state transition matrix expm(A*dt)
            - truth_initial_state: [9] true initial state s_0
            - dt: time step
            - epsilon: damping constant
    """
    nx = 9
    ny = 2

    # --- Continuous-time dynamics matrix ---
    # A = [ -εI₃   I₃    0₃ ]
    #     [  0₃  -εI₃   I₃ ]
    #     [  0₃   0₃   -εI₃ ]
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    A_cont = np.block([
        [-epsilon * I3, I3,             Z3],
        [Z3,            -epsilon * I3,  I3],
        [Z3,            Z3,             -epsilon * I3],
    ])

    # Discrete-time state transition: Φ = expm(A * dt)
    Phi = expm(A_cont * dt)

    # --- Process noise (nominally zero; small for numerical validity) ---
    Q = Q_diag * np.eye(nx)
    Q = 0.5 * (Q + Q.T) + eps * np.eye(nx)

    # --- Initial state ---
    # Truth
    s0_truth = np.array([40.0, 40.0, 40.0, 8.0, 0.0, -3.0, 0.0, 0.0, 0.0])

    # Prior (filter initialization) — significantly off from truth
    s0_prior = np.array([40.0, 50.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    #s0_prior = np.array([40.0, 40.0, 40.0, 8.0, 0.0, -3.0, 0.0, 0.0, 0.0])
    # Prior covariance
    P0_diag = np.array([10.0, 10.0, 10.0, 1e4, 1e4, 1e4, 10.0, 10.0, 10.0])
    P0 = np.diag(P0_diag)
    P0 = 0.5 * (P0 + P0.T) + eps * np.eye(nx)

    # --- Observation noise ---
    R = R_diag * np.eye(ny)

    # --- Dynamics functions ---
    def dynamics_mean(x: np.ndarray) -> np.ndarray:
        """
        Deterministic dynamics: x_{t+1} = Φ x_t.

        Args:
            x: [N, 9] or [9] states

        Returns:
            x_next: [N, 9] or [9]
        """
        return x @ Phi.T

    def dynamics_jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian of dynamics (constant, = Φ)."""
        return Phi

    # --- Observation functions ---
    def obs_mean(x: np.ndarray) -> np.ndarray:
        """
        Observation function: [arctan(x/y), arctan(z/r)]

        Args:
            x: [N, 9] or [9] states

        Returns:
            z: [N, 2] or [2] observations (azimuth, elevation)
        """
        single = x.ndim == 1
        if single:
            x = x[None, :]

        px, py, pz = x[:, 0], x[:, 1], x[:, 2]

        # Azimuth: arctan(x/y)
        azimuth = np.arctan2(px, py)

        # Elevation: arctan(z/r), r = sqrt(x^2 + y^2)
        r = np.sqrt(px**2 + py**2 + eps)
        elevation = np.arctan2(pz, r)

        z = np.stack([azimuth, elevation], axis=-1)

        if single:
            return z[0]
        return z

    def obs_jacobian(x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function.

        H is [2, 9]. Only first 3 columns (position) are nonzero.

        For azimuth = arctan(x/y):
            d/dx = y / (x^2 + y^2)
            d/dy = -x / (x^2 + y^2)
            d/dz = 0

        For elevation = arctan(z/r), r = sqrt(x^2 + y^2):
            d/dx = -x z / (r (r^2 + z^2))
            d/dy = -y z / (r (r^2 + z^2))
            d/dz = r / (r^2 + z^2)

        Args:
            x: [9] single state

        Returns:
            H: [2, 9] Jacobian
        """
        px, py, pz = x[0], x[1], x[2]

        # Azimuth derivatives
        r_xy_sq = px**2 + py**2 + eps
        daz_dx = py / r_xy_sq
        daz_dy = -px / r_xy_sq
        daz_dz = 0.0

        # Elevation derivatives
        r_xy = np.sqrt(r_xy_sq)
        r_total_sq = r_xy_sq + pz**2 + eps
        del_dx = -px * pz / (r_xy * r_total_sq + eps)
        del_dy = -py * pz / (r_xy * r_total_sq + eps)
        del_dz = r_xy / (r_total_sq + eps)

        H = np.zeros((2, 9))
        H[0, 0] = daz_dx
        H[0, 1] = daz_dy
        H[0, 2] = daz_dz
        H[1, 0] = del_dx
        H[1, 1] = del_dy
        H[1, 2] = del_dz

        return H

    # --- Build model ---
    model = StateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=s0_prior,
        initial_cov=P0,
        dynamics_mean=dynamics_mean,
        dynamics_cov=Q,
        dynamics_jacobian=dynamics_jacobian,
        obs_mean=obs_mean,
        obs_cov=R,
        obs_jacobian=obs_jacobian,
    )

    # Store extra attributes
    model.A_cont = A_cont
    model.Phi = Phi
    model.truth_initial_state = s0_truth
    model.dt = dt
    model.epsilon = epsilon

    return model


def simulate_dai22_example2(
    model: StateSpaceModel,
    T: int = 20,
    rng: Optional[Generator] = None,
    seed: int = 42,
) -> tuple:
    """
    Simulate truth trajectory and observations for Dai (2022) Example 2.

    Uses the TRUE initial state (not the prior) to generate the trajectory.
    Dynamics are deterministic; only observation noise is random.

    Args:
        model: StateSpaceModel from make_dai22_example2_ssm()
        T: Number of time steps (default 20)
        rng: NumPy random generator
        seed: Random seed (used if rng is None)

    Returns:
        true_states: [T+1, 9] True state trajectory (s_0, ..., s_T)
        observations: [T, 2] Observations (z_1, ..., z_T)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    nx = model.state_dim
    ny = model.obs_dim

    true_states = np.zeros((T + 1, nx))
    observations = np.zeros((T, ny))

    # Start from TRUTH initial state
    true_states[0] = model.truth_initial_state.copy()

    for t in range(T):
        # Deterministic dynamics
        true_states[t + 1] = model.dynamics_mean(true_states[t:t+1])[0]

        # Noisy observation
        h_true = model.obs_mean(true_states[t + 1:t + 2])[0]
        noise = rng.standard_normal(ny)
        observations[t] = h_true + model._obs_cov_chol @ noise

    return true_states, observations
