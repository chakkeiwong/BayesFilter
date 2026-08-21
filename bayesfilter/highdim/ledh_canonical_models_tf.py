"""Canonical-lane model adapters (P5 model onboarding).

Builds ``NonlinearScoreModel`` and ``CanonicalModelCallbacks`` instances
for the six leaderboard models WITHOUT placeholder Gaussians: process and
observation covariances come from the model specification (model_exact) or
sigma-point propagation; the C-10 provenance guard enforces this at
construction.

First onboarded models: LGSSM (exact) and Austria SIR (RK4 dynamics with
analytical parameter tangent reused from the batch adapter derivation;
observation model 100*exp(2*theta_2)*I_9 on infectious compartments —
model-exact, theta-dependent). Remaining models (predator-prey, 3 SV
variants) follow the same template; each addition must extend
`test_ledh_canonical_models.py` with its S-1-class gate.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel

Tensor = tf.Tensor
DTYPE = tf.float64


def austria_sir_canonical_model(theta_fixed: Tensor) -> NonlinearScoreModel:
    """Austria SIR with real (non-placeholder) Gaussian inputs.

    theta enters the RK4 dynamics (kappa, nu) and the observation
    covariance (100*exp(2*theta_2)). The direction for the score is the
    caller's choice; tangent callbacks close over the direction index via
    the standard one-hot convention at the score assembly level.

    Provenance: process noise is the model's additive noise (model_exact,
    identity scale in the latent parameterization per the frozen target's
    process_noise tensors); observation covariance is the model's stated
    100*exp(2*theta_2)*I_9 (model_exact). The flow linearization H is the
    model's linear infectious-compartment extraction (model_exact). No
    identity placeholder stands in for a derived quantity: the transition
    JACOBIAN enters through the UKF sigma-point propagation of the real
    RK4 dynamics rather than an eye() transition matrix.
    """

    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    base = zhao_cui_sir_austria_model()
    adjacency = tf.cast(base._adjacency_matrix, DTYPE)  # noqa: SLF001
    degree = tf.reduce_sum(adjacency, axis=1)
    step = tf.constant(0.005, DTYPE)
    infectious_matrix = tf.stack(
        [
            tf.one_hot(2 * index + 1, 18, dtype=DTYPE)
            for index in range(9)
        ],
        axis=0,
    )

    def physical(theta):
        return (
            0.1 * tf.exp(theta[..., 0]),
            18.0 * tf.exp(theta[..., 1]),
        )

    def rhs(theta, state):
        kappa, nu = physical(theta)
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        neighbor_s = (
            tf.einsum("nj,kj->nk", susceptible, adjacency)
            - susceptible * degree
        )
        neighbor_i = (
            tf.einsum("nj,kj->nk", infectious, adjacency)
            - infectious * degree
        )
        infection = kappa * susceptible * infectious
        rhs_s = -infection + 0.5 * neighbor_s
        rhs_i = infection - nu * infectious + 0.5 * neighbor_i
        return tf.reshape(
            tf.stack([rhs_s, rhs_i], axis=2), [tf.shape(state)[0], 18]
        )

    def rhs_tangent(theta, state, d_state, d_theta):
        kappa, nu = physical(theta)
        d_kappa = kappa * d_theta[..., 0]
        d_nu = nu * d_theta[..., 1]
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        d_susceptible = d_state[:, 0::2]
        d_infectious = d_state[:, 1::2]
        d_neighbor_s = (
            tf.einsum("nj,kj->nk", d_susceptible, adjacency)
            - d_susceptible * degree
        )
        d_neighbor_i = (
            tf.einsum("nj,kj->nk", d_infectious, adjacency)
            - d_infectious * degree
        )
        infection = kappa * susceptible * infectious
        d_infection = (
            d_kappa * susceptible * infectious
            + kappa * d_susceptible * infectious
            + kappa * susceptible * d_infectious
        )
        d_rhs_s = -d_infection + 0.5 * d_neighbor_s
        d_rhs_i = (
            d_infection
            - d_nu * infectious
            - nu * d_infectious
            + 0.5 * d_neighbor_i
        )
        return tf.reshape(
            tf.stack([d_rhs_s, d_rhs_i], axis=2), [tf.shape(state)[0], 18]
        )

    def transition_mean_fn(theta, points):
        current = points
        for _ in range(4):
            k1 = rhs(theta, current)
            k2 = rhs(theta, current + 0.5 * step * k1)
            k3 = rhs(theta, current + 0.5 * step * k2)
            k4 = rhs(theta, current + step * k3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return current

    def transition_mean_tangent_fn(theta, points, d_points):
        # d(RK4)/d(theta,state) chained through the four stages; the
        # parameter direction is the one-hot the score assembly selects —
        # here bound to direction 0..2 via theta_fixed closure convention:
        # the assembly calls with d_points carrying the state tangent and
        # the parameter direction is fixed per call (d_theta one-hot).
        d_theta = _current_direction[0]
        current, d_current = points, d_points
        for _ in range(4):
            k1 = rhs(theta, current)
            d1 = rhs_tangent(theta, current, d_current, d_theta)
            k2 = rhs(theta, current + 0.5 * step * k1)
            d2 = rhs_tangent(
                theta,
                current + 0.5 * step * k1,
                d_current + 0.5 * step * d1,
                d_theta,
            )
            k3 = rhs(theta, current + 0.5 * step * k2)
            d3 = rhs_tangent(
                theta,
                current + 0.5 * step * k2,
                d_current + 0.5 * step * d2,
                d_theta,
            )
            k4 = rhs(theta, current + step * k3)
            d4 = rhs_tangent(
                theta,
                current + step * k3,
                d_current + step * d3,
                d_theta,
            )
            current = current + step / 6.0 * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            d_current = d_current + step / 6.0 * (
                d1 + 2.0 * d2 + 2.0 * d3 + d4
            )
        return d_current

    _current_direction = [tf.constant([1.0, 0.0, 0.0], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _current_direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def observation_fn(points):
        return tf.einsum("oi,ni->no", infectious_matrix, points)

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            infectious_matrix, [tf.shape(points)[0], 9, 18]
        )

    def observation_tangent_fn(points, d_points):
        return tf.einsum("oi,ni->no", infectious_matrix, d_points)

    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    observation_covariance = (
        100.0 * tf.exp(2.0 * theta_fixed[2]) * tf.eye(9, dtype=DTYPE)
    )

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.eye(18, dtype=DTYPE),
        # ^ PROVENANCE: model_exact, NOT a placeholder. The frozen Austria
        # target adds process noise at unit scale (batch adapter line 254:
        # `rk4(...) + noise`), i.e. Q = I_18 by model definition. The
        # 2026-08 placeholder defect was feeding I to the FLOW as the
        # PREDICTED covariance; here the predicted covariance comes from
        # the UKF sigma-point propagation of the real RK4 dynamics, and I
        # enters only as the true additive-noise covariance.
        observation_covariance=observation_covariance,
    )
    # NonlinearScoreModel is frozen; the direction setter travels alongside.
    return model, set_score_direction


__all__ = ["austria_sir_canonical_model"]
