"""Canonical LEDH-PF-PF OT single-cloud filter (P3 assembly).

Per-step program, per Li(2017) Algorithm 1 as contracted in ch19c and
``ledh_alg1_contract``:

  UKF predict (per-particle P^i)
  -> zero-noise anchor + pre-flow sample
  -> dual-state LEDH flow (per-particle A/b, theta-product)
  -> PF-PF weight (transition at post-flow / proposal at pre-flow / theta)
  -> UKF update (posterior P_k^i)
  -> OT/Contract-E reset with optional dual-cap trust-region correction
  -> triple ancestry {x, P, w}

Value path only (P3). The analytical recursive score is P4; autodiff is
forbidden on this path (contract step ``analytical_score``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
from bayesfilter.highdim.ledh_alg1_contract import COVARIANCE_PROVENANCE_KINDS
from bayesfilter.highdim.ledh_flow_perparticle_tf import ledh_flow_per_particle
from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
    triple_gather,
    ukf_predict_per_particle,
    ukf_update_per_particle,
)

Tensor = tf.Tensor


@dataclass(frozen=True)
class CanonicalModelCallbacks:
    model_id: str
    state_dim: int
    observation_dim: int
    transition_mean_fn: Callable[[Tensor, int], Tensor]
    transition_log_density_fn: Callable[[Tensor, Tensor, int], Tensor]
    process_noise_covariance: Tensor
    process_noise_covariance_provenance: str
    observation_fn: Callable[[Tensor, int], Tensor]
    observation_jacobian_fn: Callable[[Tensor, int], Tensor]
    observation_covariance: Tensor
    observation_log_density_fn: Callable[[Tensor, Tensor, int], Tensor]
    initial_mean: Tensor
    initial_covariance: Tensor
    initial_covariance_provenance: str


def _require_provenance(kind: str, field: str) -> None:
    if kind not in COVARIANCE_PROVENANCE_KINDS:
        raise ValueError(
            f"{field}: covariance provenance {kind!r} is not an accepted "
            f"kind {COVARIANCE_PROVENANCE_KINDS}; identity/constant "
            "placeholders require a reviewed exception (contract C-10)"
        )


def canonical_value_and_diagnostics(
    callbacks: CanonicalModelCallbacks,
    observations: Tensor,
    *,
    particle_count: int,
    seed: int,
    flow_substeps: int = 24,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    dual_cap_enabled: bool = False,
    trust_region_enabled: bool = False,
    trust_region_lm_damping: float = 1.0e-2,
    trust_region_lm_scale_floor: float = 1.0e-4,
    trust_region_radius: float = 0.5,
) -> dict[str, Tensor]:
    """Run the canonical filter; return value + mandatory diagnostics."""

    _require_provenance(
        callbacks.initial_covariance_provenance, "initial_covariance"
    )
    _require_provenance(
        callbacks.process_noise_covariance_provenance,
        "process_noise_covariance",
    )
    dtype = tf.convert_to_tensor(callbacks.initial_mean).dtype
    observations = tf.convert_to_tensor(observations, dtype)
    horizon = int(observations.shape[0])
    dim = callbacks.state_dim
    generator = tf.random.Generator.from_seed(seed)

    initial_chol = tf.linalg.cholesky(
        tf.convert_to_tensor(callbacks.initial_covariance, dtype)
    )
    noise = generator.normal([particle_count, dim], dtype=dtype)
    states = callbacks.initial_mean[None, :] + tf.linalg.matvec(
        tf.broadcast_to(initial_chol, [particle_count, dim, dim]), noise
    )
    covariances = tf.broadcast_to(
        tf.convert_to_tensor(callbacks.initial_covariance, dtype),
        [particle_count, dim, dim],
    )
    weights = tf.fill(
        [particle_count], tf.cast(1.0 / particle_count, dtype)
    )
    process_cov = tf.convert_to_tensor(
        callbacks.process_noise_covariance, dtype
    )
    process_chol = tf.linalg.cholesky(process_cov)

    total = tf.zeros([], dtype)
    valid = tf.constant(True)
    ess_history = []
    dual_cap_active = tf.constant(dual_cap_enabled)

    design = tf.linalg.matrix_transpose(
        tf.stack(
            [
                tf.eye(dim, dtype=dtype)[i % dim]
                * (1.0 if (i // dim) % 2 == 0 else -1.0)
                for i in range(2 * dim)
            ]
        )
    )
    design = tf.tile(
        tf.transpose(design), [particle_count // (2 * dim) + 1, 1]
    )[:particle_count]

    for time_index in range(horizon):
        observation = observations[time_index]

        def mean_fn(points, _t=time_index):
            return callbacks.transition_mean_fn(points, _t)

        predicted_means, predicted_covs = ukf_predict_per_particle(
            states, covariances, mean_fn, process_cov
        )
        anchors = mean_fn(states)
        process_noise = generator.normal(
            [particle_count, dim], dtype=dtype
        )
        pre_flow = anchors + tf.linalg.matvec(
            tf.broadcast_to(process_chol, [particle_count, dim, dim]),
            process_noise,
        )

        flow = ledh_flow_per_particle(
            anchor_states=anchors,
            pre_flow_states=pre_flow,
            predicted_covariances=predicted_covs,
            observation=observation,
            observation_fn=lambda p, _t=time_index: callbacks.observation_fn(p, _t),
            observation_jacobian_fn=lambda p, _t=time_index: callbacks.observation_jacobian_fn(p, _t),
            observation_covariance=tf.convert_to_tensor(
                callbacks.observation_covariance, dtype
            ),
            prior_means=anchors,
            substeps=flow_substeps,
        )
        children = flow["post_flow_states"]

        transition_log = callbacks.transition_log_density_fn(
            children, states, time_index
        )
        observation_log = callbacks.observation_log_density_fn(
            children, observation, time_index
        )
        # Li(17) eq. bf-pfpf-alg1-weight: the denominator is the TRANSITION
        # density of the pre-flow sample, p(eta_0 | ancestor) — NOT the
        # density under the UKF-predicted covariance (which the flow module
        # reports for diagnostics). Verified against the exact 1D marginal
        # in the P3 gate debugging (2026-08-21).
        proposal_log = callbacks.transition_log_density_fn(
            pre_flow, states, time_index
        )
        logits = (
            tf.math.log(weights)
            + transition_log
            + observation_log
            + flow["forward_log_det"]
            - proposal_log
        )
        increment = tf.reduce_logsumexp(logits)
        step_weights = tf.exp(logits - increment)
        step_valid = (
            tf.reduce_all(tf.math.is_finite(children))
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(step_weights))
        )
        ess = 1.0 / tf.reduce_sum(tf.square(step_weights))
        ess_history.append(ess)

        # Fail-closed gate BEFORE the reset: Contract-E's eigvalsh raises on
        # nonfinite input rather than returning NaN, so poisoned clouds must
        # be sanitized here with validity recorded (S-4).
        safe_weights = tf.where(
            step_valid,
            step_weights,
            tf.fill([particle_count], tf.cast(1.0 / particle_count, dtype)),
        )
        safe_children = tf.where(
            tf.math.is_finite(children), children, tf.zeros_like(children)
        )

        post_means, post_covs = ukf_update_per_particle(
            predicted_means,
            predicted_covs,
            lambda p, _t=time_index: callbacks.observation_fn(p, _t),
            tf.convert_to_tensor(callbacks.observation_covariance, dtype),
            observation,
        )

        restored = _restore_cloud_primal(
            tf.cast(safe_children, tf.float32),
            tf.cast(safe_weights, tf.float32),
            tf.cast(design, tf.float32),
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
            reset_policy="contract_e",
            dual_cap_enabled=dual_cap_enabled,
            trust_region_enabled=trust_region_enabled,
            trust_region_lm_damping=trust_region_lm_damping,
            trust_region_lm_scale_floor=trust_region_lm_scale_floor,
            trust_region_radius=trust_region_radius,
        )
        reset_states = tf.cast(restored["particles"], dtype)
        reset_valid = tf.reduce_all(tf.math.is_finite(reset_states))

        # OT reset outputs an equal-weight cloud whose rows are transport
        # barycenters of the weighted children; the covariance triple
        # follows the maximum-weight ancestor of each reset row when the
        # reset does not expose an explicit ancestry (Contract E rows are
        # positionally aligned with children).
        states = reset_states
        covariances = post_covs
        weights = tf.fill(
            [particle_count], tf.cast(1.0 / particle_count, dtype)
        )
        total += tf.where(step_valid, increment, tf.cast(float("nan"), dtype))
        valid = valid & step_valid & reset_valid

    nan = tf.cast(float("nan"), dtype)
    return {
        "value": tf.where(valid, total, nan),
        "program_valid": valid,
        "per_step_ess": tf.stack(ess_history),
        "dual_cap_active": dual_cap_active,
        "model_id": tf.constant(callbacks.model_id),
    }


__all__ = [
    "CanonicalModelCallbacks",
    "canonical_value_and_diagnostics",
]
