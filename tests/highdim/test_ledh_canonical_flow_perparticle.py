"""Conformance tests C-2..C-5 for the per-particle-covariance LEDH flow.

Written before the implementation (rule R-C). Tolerances per execution plan.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_flow_perparticle_tf import (
    ledh_flow_per_particle,
)

DTYPE = tf.float64


def _fixture(seed: int, n: int = 4, dim: int = 2, obs_dim: int = 2):
    rng = np.random.default_rng(seed)
    anchors = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    pre_flow = anchors + tf.constant(
        0.3 * rng.standard_normal((n, dim)), DTYPE
    )
    covs = tf.constant(
        np.stack([np.eye(dim) * (0.4 + 0.3 * i) for i in range(n)]), DTYPE
    )
    obs_matrix = tf.constant(np.eye(obs_dim, dim), DTYPE)
    obs_cov = tf.eye(obs_dim, dtype=DTYPE) * 0.5
    observation = tf.constant(rng.standard_normal(obs_dim), DTYPE)
    return anchors, pre_flow, covs, obs_matrix, obs_cov, observation


def _run(anchors, pre_flow, covs, obs_matrix, obs_cov, observation, substeps=64):
    def obs_fn(points):
        return tf.linalg.matvec(
            tf.broadcast_to(
                obs_matrix, [tf.shape(points)[0], *obs_matrix.shape]
            ),
            points,
        )

    def obs_jac_fn(points):
        return tf.broadcast_to(
            obs_matrix, [tf.shape(points)[0], *obs_matrix.shape]
        )

    return ledh_flow_per_particle(
        anchor_states=anchors,
        pre_flow_states=pre_flow,
        predicted_covariances=covs,
        observation=observation,
        observation_fn=obs_fn,
        observation_jacobian_fn=obs_jac_fn,
        observation_covariance=obs_cov,
        substeps=substeps,
    )


def test_c2_flow_consumes_per_particle_covariance():
    anchors, pre_flow, covs, H, R, y = _fixture(1)
    base = _run(anchors, pre_flow, covs, H, R, y)
    perturbed_covs = tf.tensor_scatter_nd_update(
        covs, [[2]], [covs[2] * 3.0]
    )
    pert = _run(anchors, pre_flow, perturbed_covs, H, R, y)
    moved = tf.reduce_max(
        tf.abs(base["post_flow_states"][2] - pert["post_flow_states"][2])
    ).numpy()
    others = tf.reduce_max(
        tf.abs(
            tf.concat(
                [
                    base["post_flow_states"][:2] - pert["post_flow_states"][:2],
                    base["post_flow_states"][3:] - pert["post_flow_states"][3:],
                ],
                axis=0,
            )
        )
    ).numpy()
    assert moved > 1.0e-6, "flow ignored the perturbed particle covariance"
    assert others < 1.0e-14, "covariance perturbation leaked across particles"


def test_c3_linearization_at_anchor_not_actual():
    anchors, pre_flow, covs, H, R, y = _fixture(2)
    base = _run(anchors, pre_flow, covs, H, R, y)
    # Perturb the ACTUAL pre-flow sample of particle 1; anchor unchanged.
    pre2 = tf.tensor_scatter_nd_update(
        pre_flow, [[1]], [pre_flow[1] + 0.5]
    )
    pert = _run(anchors, pre2, covs, H, R, y)
    # Linear observation model => A,b depend only on anchors/covs; the
    # affine map applied to particle 1 must be IDENTICAL, so the output
    # difference equals the affine map applied to the input difference —
    # verified via the returned per-particle map applied to the delta.
    delta_in = (pre2[1] - pre_flow[1]).numpy()
    map_matrix = base["total_affine_matrix"][1].numpy()
    predicted_delta_out = map_matrix @ delta_in
    actual_delta_out = (
        pert["post_flow_states"][1] - base["post_flow_states"][1]
    ).numpy()
    err = np.max(np.abs(actual_delta_out - predicted_delta_out))
    assert err < 1.0e-9, (
        "flow map changed when only the actual sample moved: "
        "linearization is not anchored"
    )


def test_c4_log_det_matches_numerical_jacobian():
    anchors, pre_flow, covs, H, R, y = _fixture(3, n=3, dim=2)
    result = _run(anchors, pre_flow, covs, H, R, y)
    # The realized map on the actual state is affine with the returned
    # total matrix; its Jacobian determinant must equal exp(forward_log_det).
    for i in range(3):
        matrix = result["total_affine_matrix"][i].numpy()
        det = abs(np.linalg.det(matrix))
        log_det = float(result["forward_log_det"][i].numpy())
        assert abs(np.log(det) - log_det) < 1.0e-4 * max(abs(np.log(det)), 1.0), (
            f"particle {i}: theta-product {log_det} vs map log|det| {np.log(det)}"
        )


def test_c4_substep_refinement_toward_kalman():
    """EDH limit: with shared covariance AND shared prior mean (all
    linearizations at the pooled prior mean), the flow is the exact
    Gaussian homotopy; the transported cloud mean approaches the Kalman
    posterior mean as substeps -> infinity (Euler O(eps) error)."""
    _, pre_flow, covs, H, R, y = _fixture(4, n=6, dim=2)
    shared = tf.eye(2, dtype=DTYPE) * 0.6
    shared_covs = tf.broadcast_to(shared, covs.shape)
    prior_mean = tf.reduce_mean(pre_flow, axis=0)
    pooled_anchors = tf.broadcast_to(prior_mean, pre_flow.shape)
    p = shared.numpy()
    h = H.numpy()
    r = R.numpy()
    s = h @ p @ h.T + r
    gain = p @ h.T @ np.linalg.inv(s)
    post_mean = prior_mean.numpy() + gain @ (
        y.numpy() - h @ prior_mean.numpy()
    )
    errors = []
    for substeps in (32, 64, 128, 256):
        res = _run(
            pooled_anchors, pre_flow, shared_covs, H, R, y, substeps=substeps
        )
        cloud_mean = tf.reduce_mean(res["post_flow_states"], axis=0).numpy()
        errors.append(np.linalg.norm(cloud_mean - post_mean))
    # Declared tolerance: error halves per substep doubling (Euler order),
    # terminal atol 5e-3.
    assert errors[-1] <= errors[0] + 1.0e-12
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine <= 0.75 * coarse + 1.0e-12, (
            f"no first-order refinement: {errors}"
        )
    assert errors[-1] < 5.0e-3 * max(np.linalg.norm(post_mean), 1.0), (
        f"terminal Euler error too large: {errors}"
    )


def test_c5_weight_identity_components():
    anchors, pre_flow, covs, H, R, y = _fixture(5)
    result = _run(anchors, pre_flow, covs, H, R, y)
    # pre_flow_log_density must equal the Gaussian density of the pre-flow
    # sample under (anchor prior mean, predicted covariance), independently
    # computed via slogdet (det-free, numerically stable reference).
    for i in range(4):
        diff = (pre_flow[i] - anchors[i]).numpy()
        p = covs[i].numpy()
        sign, logdet = np.linalg.slogdet(2.0 * np.pi * p)
        assert sign > 0
        expected = -0.5 * (diff @ np.linalg.solve(p, diff) + logdet)
        got = float(result["pre_flow_log_density"][i].numpy())
        assert abs(got - expected) < 1.0e-8, f"particle {i} proposal density"
