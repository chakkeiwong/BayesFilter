"""P4 scaffold: flow value + analytical parameter tangent, LGSSM slice.

Scaffold naming per rule R-D (expiry: absorbed into
``ledh_canonical_score_tf`` when all stages land; G-2 enforces).

Derivation (note S3, LGSSM slice, theta scalar scaling F = theta*F0):

  P^i(theta) = F P_prev F^T + Q            (UKF exact on linear dynamics)
  dP^i = dF P F^T + F P dF^T
  anchor a = F x_prev; da = dF x_prev
  per substep with lam fixed:
    S = lam H P H^T + R;      dS = lam H dP H^T
    K = P H^T S^{-1};         dK = dP H^T S^{-1} - P H^T S^{-1} dS S^{-1}
    A = -0.5 K H;             dA = -0.5 dK H
    zres = z - (h(a) - H a) = z  (linear H: residual e = 0 offset)
    phrz = P H^T R^{-1} z_eff; dphrz = dP H^T R^{-1} z_eff + P H^T R^{-1} dz_eff
    inner = phrz + lam A phrz + A m;  d(inner) by product rule
    b = inner + 2 lam A inner; db by product rule
    state step x <- (I + eps A) x + eps b
    dx <- (I + eps A) dx + eps dA x + eps db
  All matrix operations elementwise-analytical; no autodiff.
"""

from __future__ import annotations

import tensorflow as tf

Tensor = tf.Tensor


def flow_value_and_parameter_tangent_lgssm(
    theta: Tensor,
    f0: Tensor,
    process_covariance: Tensor,
    observation_matrix: Tensor,
    observation_covariance: Tensor,
    states: Tensor,
    covariances: Tensor,
    noise: Tensor,
    observation: Tensor,
    *,
    substeps: int,
    with_tangent: bool,
) -> tuple[Tensor, Tensor | None]:
    """Return post-flow states and (optionally) d(post_flow)/dtheta.

    theta: [1] scalar parameter scaling the transition matrix.
    """

    dtype = states.dtype
    dim = int(states.shape[1])
    count = tf.shape(states)[0]
    transition = theta[0] * f0
    d_transition = f0  # d(theta*F0)/dtheta

    process_chol = tf.linalg.cholesky(process_covariance)
    anchors = tf.einsum("ij,nj->ni", transition, states)
    d_anchors = tf.einsum("ij,nj->ni", d_transition, states)
    pre_flow = anchors + tf.einsum("ij,nj->ni", process_chol, noise)
    d_pre_flow = d_anchors

    predicted = (
        tf.einsum("ij,njk,lk->nil", transition, covariances, transition)
        + process_covariance[None]
    )
    d_predicted = tf.einsum(
        "ij,njk,lk->nil", d_transition, covariances, transition
    ) + tf.einsum("ij,njk,lk->nil", transition, covariances, d_transition)

    eye = tf.eye(dim, dtype=dtype)
    eps = tf.constant(1.0 / substeps, dtype=dtype)
    h = observation_matrix
    r = observation_covariance
    r_chol = tf.linalg.cholesky(r)
    r_inv_z = tf.linalg.cholesky_solve(r_chol, observation[:, None])[:, 0]

    actual = pre_flow
    d_actual = d_pre_flow
    prior_means = anchors
    d_prior_means = d_anchors
    log_det = tf.zeros([tf.shape(states)[0]], dtype)
    d_log_det = tf.zeros([tf.shape(states)[0]], dtype)

    for step_index in range(substeps):
        lam = tf.constant((step_index + 1) / substeps, dtype=dtype)
        php = tf.einsum("oi,nij,pj->nop", h, predicted, h)
        d_php = tf.einsum("oi,nij,pj->nop", h, d_predicted, h)
        innovation = lam * php + r[None]
        innovation_chol = tf.linalg.cholesky(innovation)
        ph_t = tf.einsum("nij,oj->nio", predicted, h)
        d_ph_t = tf.einsum("nij,oj->nio", d_predicted, h)
        # K_lam = P H^T S^{-1}
        k_lam = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(ph_t)
            )
        )
        d_s = lam * d_php
        # d(K_lam) = dP H^T S^{-1} - K_lam dS S^{-1}
        # implemented via cholesky_solve on the transposed systems:
        term_one = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(d_ph_t)
            )
        )
        k_ds = tf.einsum("nio,nop->nip", k_lam, d_s)
        term_two = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol, tf.linalg.matrix_transpose(k_ds)
            )
        )
        d_k_lam = term_one - term_two
        a_matrix = -0.5 * tf.einsum("nio,oj->nij", k_lam, h)
        d_a_matrix = -0.5 * tf.einsum("nio,oj->nij", d_k_lam, h)

        phrz = tf.einsum("nio,o->ni", ph_t, r_inv_z)
        d_phrz = tf.einsum("nio,o->ni", d_ph_t, r_inv_z)
        a_phrz = tf.einsum("nij,nj->ni", a_matrix, phrz)
        d_a_phrz = tf.einsum("nij,nj->ni", d_a_matrix, phrz) + tf.einsum(
            "nij,nj->ni", a_matrix, d_phrz
        )
        a_mean = tf.einsum("nij,nj->ni", a_matrix, prior_means)
        d_a_mean = tf.einsum("nij,nj->ni", d_a_matrix, prior_means) + tf.einsum(
            "nij,nj->ni", a_matrix, d_prior_means
        )
        inner = phrz + lam * a_phrz + a_mean
        d_inner = d_phrz + lam * d_a_phrz + d_a_mean
        a_inner = tf.einsum("nij,nj->ni", a_matrix, inner)
        d_a_inner = tf.einsum("nij,nj->ni", d_a_matrix, inner) + tf.einsum(
            "nij,nj->ni", a_matrix, d_inner
        )
        b_vector = inner + 2.0 * lam * a_inner
        d_b_vector = d_inner + 2.0 * lam * d_a_inner

        new_actual = (
            actual
            + eps * (tf.einsum("nij,nj->ni", a_matrix, actual) + b_vector)
        )
        if with_tangent:
            d_actual = (
                d_actual
                + eps
                * (
                    tf.einsum("nij,nj->ni", d_a_matrix, actual)
                    + tf.einsum("nij,nj->ni", a_matrix, d_actual)
                    + d_b_vector
                )
            )
        actual = new_actual
        # theta-product and its parameter derivative:
        # d log|det(I+eps A)| = eps tr[(I+eps A)^{-1} dA]
        step_matrix = eye[None] + eps * a_matrix
        log_det = log_det + tf.math.log(
            tf.abs(tf.linalg.det(step_matrix))
        )
        if with_tangent:
            step_inv = tf.linalg.inv(step_matrix)
            d_log_det = d_log_det + eps * tf.linalg.trace(
                tf.einsum("nij,njk->nik", step_inv, d_a_matrix)
            )

    if with_tangent:
        return actual, d_actual[..., None], log_det, d_log_det
    return actual, None, log_det, None


def multi_step_value_and_score_lgssm(
    theta: Tensor,
    f0: Tensor,
    process_covariance: Tensor,
    observation_matrix: Tensor,
    observation_covariance: Tensor,
    initial_states: Tensor,
    covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    *,
    substeps: int,
    with_tangent: bool,
) -> tuple[Tensor, Tensor | None]:
    """Multi-step value + analytical score, gated recursion slice.

    Slice semantics (recorded honestly): particles carry as post-flow
    children step to step with equal-weight reset (reset_policy='none'
    equivalent); per-particle covariances held FIXED across steps, so this
    gate covers the S2-S4+S8 tangent recursion (state-path and weight
    tangents through the full horizon) but NOT yet the covariance-recursion
    tangent (S1/S5 chaining) or the Contract-E reset tangent (S6) — both
    remain open ledger stages. The state tangent d(children)/dtheta
    propagates through the anchor of the NEXT step: d(anchor_{t+1}) =
    dF x_t + F dx_t.
    """

    dtype = initial_states.dtype
    horizon = int(observations.shape[0])
    count = tf.shape(initial_states)[0]
    transition = theta[0] * f0
    d_transition = f0
    process_chol = tf.linalg.cholesky(process_covariance)
    obs_chol = tf.linalg.cholesky(observation_covariance)
    eye = tf.eye(int(initial_states.shape[1]), dtype=dtype)
    eps = tf.constant(1.0 / substeps, dtype=dtype)
    h = observation_matrix
    r_inv = tf.linalg.cholesky_solve(obs_chol, tf.eye(int(observation_matrix.shape[0]), dtype=dtype))

    states = initial_states
    d_states = tf.zeros_like(initial_states)
    weights_log = tf.fill([count], -tf.math.log(tf.cast(count, dtype)))
    total = tf.zeros([], dtype)
    d_total = tf.zeros([], dtype)

    predicted = (
        tf.einsum("ij,njk,lk->nil", transition, covariances, transition)
        + process_covariance[None]
    )
    base_d_predicted = tf.einsum(
        "ij,njk,lk->nil", d_transition, covariances, transition
    ) + tf.einsum("ij,njk,lk->nil", transition, covariances, d_transition)

    for time_index in range(horizon):
        observation = observations[time_index]
        noise = noises[time_index]
        anchors = tf.einsum("ij,nj->ni", transition, states)
        d_anchors = tf.einsum("ij,nj->ni", d_transition, states) + tf.einsum(
            "ij,nj->ni", transition, d_states
        )
        pre_flow = anchors + tf.einsum("ij,nj->ni", process_chol, noise)
        d_pre_flow = d_anchors

        r_inv_z = tf.linalg.matvec(r_inv, observation)
        actual = pre_flow
        d_actual = d_pre_flow
        log_det = tf.zeros([count], dtype)
        d_log_det = tf.zeros([count], dtype)
        for step_index in range(substeps):
            lam = tf.constant((step_index + 1) / substeps, dtype=dtype)
            php = tf.einsum("oi,nij,pj->nop", h, predicted, h)
            d_php = tf.einsum("oi,nij,pj->nop", h, base_d_predicted, h)
            innovation_chol = tf.linalg.cholesky(
                lam * php + observation_covariance[None]
            )
            ph_t = tf.einsum("nij,oj->nio", predicted, h)
            d_ph_t = tf.einsum("nij,oj->nio", base_d_predicted, h)
            k_lam = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(ph_t)
                )
            )
            d_s = lam * d_php
            term_one = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(d_ph_t)
                )
            )
            k_ds = tf.einsum("nio,nop->nip", k_lam, d_s)
            term_two = tf.linalg.matrix_transpose(
                tf.linalg.cholesky_solve(
                    innovation_chol, tf.linalg.matrix_transpose(k_ds)
                )
            )
            d_k_lam = term_one - term_two
            a_matrix = -0.5 * tf.einsum("nio,oj->nij", k_lam, h)
            d_a_matrix = -0.5 * tf.einsum("nio,oj->nij", d_k_lam, h)
            phrz = tf.einsum("nio,o->ni", ph_t, r_inv_z)
            d_phrz = tf.einsum("nio,o->ni", d_ph_t, r_inv_z)
            a_phrz = tf.einsum("nij,nj->ni", a_matrix, phrz)
            d_a_phrz = tf.einsum(
                "nij,nj->ni", d_a_matrix, phrz
            ) + tf.einsum("nij,nj->ni", a_matrix, d_phrz)
            a_mean = tf.einsum("nij,nj->ni", a_matrix, anchors)
            d_a_mean = tf.einsum(
                "nij,nj->ni", d_a_matrix, anchors
            ) + tf.einsum("nij,nj->ni", a_matrix, d_anchors)
            inner = phrz + lam * a_phrz + a_mean
            d_inner = d_phrz + lam * d_a_phrz + d_a_mean
            a_inner = tf.einsum("nij,nj->ni", a_matrix, inner)
            d_a_inner = tf.einsum(
                "nij,nj->ni", d_a_matrix, inner
            ) + tf.einsum("nij,nj->ni", a_matrix, d_inner)
            b_vector = inner + 2.0 * lam * a_inner
            d_b_vector = d_inner + 2.0 * lam * d_a_inner
            new_actual = actual + eps * (
                tf.einsum("nij,nj->ni", a_matrix, actual) + b_vector
            )
            d_actual = d_actual + eps * (
                tf.einsum("nij,nj->ni", d_a_matrix, actual)
                + tf.einsum("nij,nj->ni", a_matrix, d_actual)
                + d_b_vector
            )
            actual = new_actual
            step_matrix = eye[None] + eps * a_matrix
            log_det += tf.math.log(tf.abs(tf.linalg.det(step_matrix)))
            step_inv = tf.linalg.inv(step_matrix)
            d_log_det += eps * tf.linalg.trace(
                tf.einsum("nij,njk->nik", step_inv, d_a_matrix)
            )

        children, d_children = actual, d_actual
        transition_log, d_transition_log = _gaussian_log_density_and_tangent(
            children, d_children, anchors, d_anchors, process_chol
        )
        observed = tf.einsum("oi,ni->no", h, children)
        d_observed = tf.einsum("oi,ni->no", h, d_children)
        obs_target = tf.broadcast_to(observation[None, :], tf.shape(observed))
        observation_log, d_observation_log = _gaussian_log_density_and_tangent(
            obs_target, None, observed, d_observed, obs_chol
        )
        proposal_log, d_proposal_log = _gaussian_log_density_and_tangent(
            pre_flow, d_pre_flow, anchors, d_anchors, process_chol
        )
        logits = (
            weights_log
            + transition_log
            + observation_log
            + log_det
            - proposal_log
        )
        d_logits = (
            d_transition_log + d_observation_log + d_log_det - d_proposal_log
        )
        increment = tf.reduce_logsumexp(logits)
        softmax = tf.exp(logits - increment)
        total += increment
        d_total += tf.reduce_sum(softmax * d_logits)

        states = children
        d_states = d_children
        weights_log = tf.fill([count], -tf.math.log(tf.cast(count, dtype)))

    if with_tangent:
        return total, d_total[None]
    return total, None


def _cholesky_forward_differential(
    chol: Tensor, d_matrix: Tensor
) -> Tensor:
    """dL for L L^T = Sigma: dL = L Phi(L^{-1} dSigma L^{-T}).

    Phi = lower triangle with halved diagonal (standard forward-mode
    Cholesky differential; same identity as the repo's `_cholesky_jvp`).
    Batched over the leading axis.
    """

    inv_d = tf.linalg.triangular_solve(chol, d_matrix)
    inv_d_inv_t = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(
            chol, tf.linalg.matrix_transpose(inv_d)
        )
    )
    lower = tf.linalg.band_part(inv_d_inv_t, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(inv_d_inv_t))
    return tf.linalg.matmul(chol, phi)


def _sigma_points_with_tangent(
    means: Tensor,
    covariances: Tensor,
    d_means: Tensor,
    d_covariances: Tensor,
    scale: float,
    jitter: float,
) -> tuple[Tensor, Tensor]:
    """Sigma points and their analytical tangents (Cholesky differential)."""

    dim = tf.shape(means)[1]
    dtype = means.dtype
    eye = tf.eye(dim, dtype=dtype)
    stabilized = 0.5 * (
        covariances + tf.linalg.matrix_transpose(covariances)
    ) + tf.constant(jitter, dtype=dtype) * eye
    d_stabilized = 0.5 * (
        d_covariances + tf.linalg.matrix_transpose(d_covariances)
    )
    scale_c = tf.constant(scale, dtype=dtype)
    chol = tf.linalg.cholesky(scale_c * stabilized)
    d_chol = _cholesky_forward_differential(chol, scale_c * d_stabilized)
    offsets = tf.linalg.matrix_transpose(chol)
    d_offsets = tf.linalg.matrix_transpose(d_chol)
    points = tf.concat(
        [
            means[:, None, :],
            means[:, None, :] + offsets,
            means[:, None, :] - offsets,
        ],
        axis=1,
    )
    d_points = tf.concat(
        [
            d_means[:, None, :],
            d_means[:, None, :] + d_offsets,
            d_means[:, None, :] - d_offsets,
        ],
        axis=1,
    )
    return points, d_points


def ukf_predict_with_parameter_tangent(
    states: Tensor,
    covariances: Tensor,
    d_states: Tensor,
    d_covariances: Tensor,
    transition_mean_fn,
    transition_mean_tangent_fn,
    process_noise_covariance: Tensor,
    *,
    d_process_noise_covariance: Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    jitter: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """S1: unscented prediction with analytical parameter tangent.

    ``transition_mean_tangent_fn(points, d_points)`` must return the TOTAL
    tangent of the dynamics at ``points`` (partial-theta term plus Jacobian
    times ``d_points``) — the same contract as the repo's model tangent
    callbacks.
    """

    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import _unscented_weights

    dtype = states.dtype
    dim = int(states.shape[1])
    count = tf.shape(states)[0]
    mean_w, cov_w, scale = _unscented_weights(
        dim, dtype, alpha=alpha, beta=beta, kappa=kappa
    )
    points, d_points = _sigma_points_with_tangent(
        states, covariances, d_states, d_covariances, scale, jitter
    )
    flat = tf.reshape(points, [-1, dim])
    d_flat = tf.reshape(d_points, [-1, dim])
    pushed = tf.reshape(
        transition_mean_fn(flat), [count, 2 * dim + 1, dim]
    )
    d_pushed = tf.reshape(
        transition_mean_tangent_fn(flat, d_flat),
        [count, 2 * dim + 1, dim],
    )
    predicted_means = tf.einsum("s,nsd->nd", mean_w, pushed)
    d_predicted_means = tf.einsum("s,nsd->nd", mean_w, d_pushed)
    centered = pushed - predicted_means[:, None, :]
    d_centered = d_pushed - d_predicted_means[:, None, :]
    predicted_covs = 0.5 * (
        lambda m: m + tf.linalg.matrix_transpose(m)
    )(
        tf.einsum("s,nsi,nsj->nij", cov_w, centered, centered)
    ) + tf.cast(process_noise_covariance, dtype)[None]
    d_predicted_covs = tf.einsum(
        "s,nsi,nsj->nij", cov_w, d_centered, centered
    ) + tf.einsum("s,nsi,nsj->nij", cov_w, centered, d_centered)
    d_predicted_covs = 0.5 * (
        d_predicted_covs + tf.linalg.matrix_transpose(d_predicted_covs)
    )
    if d_process_noise_covariance is not None:
        d_predicted_covs += tf.cast(d_process_noise_covariance, dtype)[None]
    return predicted_means, predicted_covs, d_predicted_means, d_predicted_covs


def ukf_update_with_parameter_tangent(
    predicted_means: Tensor,
    predicted_covariances: Tensor,
    d_predicted_means: Tensor,
    d_predicted_covariances: Tensor,
    observation_mean_fn,
    observation_mean_tangent_fn,
    observation_covariance: Tensor,
    observation: Tensor,
    *,
    d_observation_covariance: Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    jitter: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """S5: unscented update with analytical parameter tangent."""

    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import _unscented_weights

    dtype = predicted_means.dtype
    dim = int(predicted_means.shape[1])
    obs_dim = int(observation.shape[-1])
    count = tf.shape(predicted_means)[0]
    mean_w, cov_w, scale = _unscented_weights(
        dim, dtype, alpha=alpha, beta=beta, kappa=kappa
    )
    points, d_points = _sigma_points_with_tangent(
        predicted_means,
        predicted_covariances,
        d_predicted_means,
        d_predicted_covariances,
        scale,
        jitter,
    )
    flat = tf.reshape(points, [-1, dim])
    d_flat = tf.reshape(d_points, [-1, dim])
    observed = tf.reshape(
        observation_mean_fn(flat), [count, 2 * dim + 1, obs_dim]
    )
    d_observed = tf.reshape(
        observation_mean_tangent_fn(flat, d_flat),
        [count, 2 * dim + 1, obs_dim],
    )
    observed_means = tf.einsum("s,nso->no", mean_w, observed)
    d_observed_means = tf.einsum("s,nso->no", mean_w, d_observed)
    centered_x = points - predicted_means[:, None, :]
    d_centered_x = d_points - d_predicted_means[:, None, :]
    centered_y = observed - observed_means[:, None, :]
    d_centered_y = d_observed - d_observed_means[:, None, :]
    innovation_cov = tf.einsum(
        "s,nsi,nsj->nij", cov_w, centered_y, centered_y
    ) + tf.cast(observation_covariance, dtype)[None]
    innovation_cov = 0.5 * (
        innovation_cov + tf.linalg.matrix_transpose(innovation_cov)
    )
    d_innovation_cov = tf.einsum(
        "s,nsi,nsj->nij", cov_w, d_centered_y, centered_y
    ) + tf.einsum("s,nsi,nsj->nij", cov_w, centered_y, d_centered_y)
    d_innovation_cov = 0.5 * (
        d_innovation_cov + tf.linalg.matrix_transpose(d_innovation_cov)
    )
    if d_observation_covariance is not None:
        d_innovation_cov += tf.cast(d_observation_covariance, dtype)[None]
    cross_cov = tf.einsum(
        "s,nsi,nsj->nij", cov_w, centered_x, centered_y
    )
    d_cross_cov = tf.einsum(
        "s,nsi,nsj->nij", cov_w, d_centered_x, centered_y
    ) + tf.einsum("s,nsi,nsj->nij", cov_w, centered_x, d_centered_y)
    chol = tf.linalg.cholesky(
        innovation_cov
        + tf.constant(jitter, dtype=dtype) * tf.eye(obs_dim, dtype=dtype)
    )
    gain = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(cross_cov))
    )
    # dK = (dC - K dS) S^{-1}
    residual_matrix = d_cross_cov - tf.einsum(
        "nio,nop->nip", gain, d_innovation_cov
    )
    d_gain = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(
            chol, tf.linalg.matrix_transpose(residual_matrix)
        )
    )
    innovation = observation[None, :] - observed_means
    d_innovation = -d_observed_means
    post_means = predicted_means + tf.einsum("nio,no->ni", gain, innovation)
    d_post_means = (
        d_predicted_means
        + tf.einsum("nio,no->ni", d_gain, innovation)
        + tf.einsum("nio,no->ni", gain, d_innovation)
    )
    ksk = tf.einsum("nio,nop,njp->nij", gain, innovation_cov, gain)
    d_ksk = (
        tf.einsum("nio,nop,njp->nij", d_gain, innovation_cov, gain)
        + tf.einsum("nio,nop,njp->nij", gain, d_innovation_cov, gain)
        + tf.einsum("nio,nop,njp->nij", gain, innovation_cov, d_gain)
    )
    post_covs = 0.5 * (
        (predicted_covariances - ksk)
        + tf.linalg.matrix_transpose(predicted_covariances - ksk)
    )
    d_post_covs = 0.5 * (
        (d_predicted_covariances - d_ksk)
        + tf.linalg.matrix_transpose(d_predicted_covariances - d_ksk)
    )
    return post_means, post_covs, d_post_means, d_post_covs


__all__ = [
    "flow_value_and_parameter_tangent_lgssm",
    "one_step_increment_and_parameter_tangent_lgssm",
    "multi_step_value_and_score_lgssm",
    "ukf_predict_with_parameter_tangent",
    "ukf_update_with_parameter_tangent",
]


def _gaussian_log_density_and_tangent(
    points: Tensor,
    d_points: Tensor | None,
    means: Tensor,
    d_means: Tensor | None,
    covariance_chol: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """log N(points; means, C) with analytical tangent through both args."""

    residual = points - means
    solved = tf.linalg.triangular_solve(
        tf.broadcast_to(
            covariance_chol, [tf.shape(points)[0], *covariance_chol.shape]
        ),
        residual[:, :, None],
    )[:, :, 0]
    dim = int(points.shape[1])
    import math as _math

    log_norm = tf.constant(
        dim * _math.log(2.0 * _math.pi), dtype=points.dtype
    ) + 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(covariance_chol))
    )
    value = -0.5 * (tf.reduce_sum(tf.square(solved), axis=1) + log_norm)
    if d_points is None and d_means is None:
        return value, None
    d_residual = (d_points if d_points is not None else 0.0) - (
        d_means if d_means is not None else 0.0
    )
    d_solved = tf.linalg.triangular_solve(
        tf.broadcast_to(
            covariance_chol, [tf.shape(points)[0], *covariance_chol.shape]
        ),
        d_residual[:, :, None],
    )[:, :, 0]
    d_value = -tf.reduce_sum(solved * d_solved, axis=1)
    return value, d_value


def one_step_increment_and_parameter_tangent_lgssm(
    theta: Tensor,
    f0: Tensor,
    process_covariance: Tensor,
    observation_matrix: Tensor,
    observation_covariance: Tensor,
    states: Tensor,
    covariances: Tensor,
    noise: Tensor,
    observation: Tensor,
    weights: Tensor,
    *,
    substeps: int,
    with_tangent: bool,
) -> tuple[Tensor, Tensor | None]:
    """S4+S8: one-step PF-PF log-likelihood increment and its d/dtheta.

    Weight per Li(17): log w = log w_prev + log p(x_k|anc) + log p(z|x_k)
    + log theta - log p(eta_0|anc); increment = logsumexp(log w). The
    analytical tangent flows through every term: post-flow states (S3),
    anchors, and the log-det.
    """

    dtype = states.dtype
    transition = theta[0] * f0
    d_transition = f0
    anchors = tf.einsum("ij,nj->ni", transition, states)
    d_anchors = tf.einsum("ij,nj->ni", d_transition, states)
    process_chol = tf.linalg.cholesky(process_covariance)
    pre_flow = anchors + tf.einsum("ij,nj->ni", process_chol, noise)
    d_pre_flow = d_anchors

    children, d_children, log_det, d_log_det = (
        flow_value_and_parameter_tangent_lgssm(
            theta,
            f0,
            process_covariance,
            observation_matrix,
            observation_covariance,
            states,
            covariances,
            noise,
            observation,
            substeps=substeps,
            with_tangent=with_tangent,
        )
    )
    d_children_vec = d_children[..., 0] if with_tangent else None

    transition_log, d_transition_log = _gaussian_log_density_and_tangent(
        children,
        d_children_vec,
        anchors,
        d_anchors if with_tangent else None,
        process_chol,
    )
    obs_chol = tf.linalg.cholesky(observation_covariance)
    observed = tf.einsum(
        "oi,ni->no", observation_matrix, children
    )
    d_observed = (
        tf.einsum("oi,ni->no", observation_matrix, d_children_vec)
        if with_tangent
        else None
    )
    obs_target = tf.broadcast_to(
        observation[None, :], tf.shape(observed)
    )
    observation_log, d_observation_log_neg = _gaussian_log_density_and_tangent(
        obs_target,
        None,
        observed,
        d_observed,
        obs_chol,
    )
    proposal_log, d_proposal_log = _gaussian_log_density_and_tangent(
        pre_flow,
        d_pre_flow if with_tangent else None,
        anchors,
        d_anchors if with_tangent else None,
        process_chol,
    )

    logits = (
        tf.math.log(weights)
        + transition_log
        + observation_log
        + log_det
        - proposal_log
    )
    increment = tf.reduce_logsumexp(logits)
    if not with_tangent:
        return increment, None
    d_logits = (
        d_transition_log
        + d_observation_log_neg
        + d_log_det
        - d_proposal_log
    )
    softmax = tf.exp(logits - increment)
    d_increment = tf.reduce_sum(softmax * d_logits)
    return increment, d_increment[None]
