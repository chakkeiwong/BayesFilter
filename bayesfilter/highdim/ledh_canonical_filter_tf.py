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

import numpy as np
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


def _replication_generator(seed: int) -> tf.random.Generator:
    """Independent per-seed stream for replication runs.

    `tf.random.Generator.from_seed(s)` and `from_seed(s+1)` are the SAME
    Philox stream shifted by one draw row, so consecutive-seed
    "replications" share almost the entire particle cloud (fidelity item
    #7, found 2026-08-25 via a seed-invariant heavy-weight event on the
    linear leaderboard anchor). SeedSequence hashing spreads seeds far
    apart, making stream overlap between any two replication seeds
    negligible while staying deterministic per seed.
    """

    material = np.random.SeedSequence(int(seed)).generate_state(
        2, dtype=np.uint32
    )
    return tf.random.Generator.from_seed(
        (int(material[0]) << 31) ^ int(material[1])
    )


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
    temper_stages: int = 1,
    annealed_resampling: bool = False,
    flow_prior_cap: float = float("inf"),
    resample_seed: int = 1,
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
    generator = _replication_generator(seed)

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
        # Class-B fail-closed guard (2026-08-26): nonfinite lifecycle
        # state must become a RECORDED veto, never an eigh/cholesky
        # crash. Added during the first claim-scale full-production
        # Austria run; the crash was later localized to the SCORE
        # lane's S6 gap Cholesky (this value-lane guard has not fired —
        # it stands as no-fire Class-B protection).
        state_finite = bool(
            tf.reduce_all(tf.math.is_finite(states)).numpy()
        ) and bool(
            tf.reduce_all(tf.math.is_finite(covariances)).numpy()
        )
        if not state_finite:
            valid = tf.constant(False)
            total = tf.cast(float("nan"), dtype)
            for _ in range(time_index, horizon):
                ess_history.append(tf.cast(float("nan"), dtype))
            break
        observation = observations[time_index]

        def mean_fn(points, _t=time_index):
            return callbacks.transition_mean_fn(points, _t)

        predicted_means, predicted_covs = ukf_predict_per_particle(
            states, covariances, mean_fn, process_cov
        )
        predicted_finite = bool(
            tf.reduce_all(tf.math.is_finite(predicted_covs)).numpy()
        ) and bool(
            tf.reduce_all(tf.math.is_finite(predicted_means)).numpy()
        )
        if not predicted_finite:
            # same guard class: dynamics blowup from finite states
            valid = tf.constant(False)
            total = tf.cast(float("nan"), dtype)
            for _ in range(time_index, horizon):
                ess_history.append(tf.cast(float("nan"), dtype))
            break
        anchors = mean_fn(states)
        process_noise = generator.normal(
            [particle_count, dim], dtype=dtype
        )
        pre_flow = anchors + tf.linalg.matvec(
            tf.broadcast_to(process_chol, [particle_count, dim, dim]),
            process_noise,
        )

        # Staged (tempered) flow: the homotopy is split into temper_stages
        # invertible sub-flows with likelihood tempering (P/k, R*k per
        # stage). Exactness is guaranteed by the PF-PF importance identity
        # for ANY invertible composed map (total forward log-det
        # accumulated); staging is an efficiency lever calibrated in P6.
        # Evidence: 2026-08-21/22 repair-arm evaluation — combined with a
        # model-faithful initial covariance it restored Austria-scope ESS
        # from 1/256 to 98/256 at the final step.
        stage_count = max(1, int(temper_stages))
        if np.isfinite(flow_prior_cap):
            cap_eigenvalues, cap_vectors = tf.linalg.eigh(predicted_covs)
            flow_prior = tf.einsum(
                "nij,nj,nkj->nik",
                cap_vectors,
                tf.minimum(
                    cap_eigenvalues, tf.cast(flow_prior_cap, dtype)
                ),
                cap_vectors,
            )
        else:
            flow_prior = predicted_covs

        if annealed_resampling:
            # Within-step annealed SMC (P6 contract PASSED 2026-08-22,
            # artifact annealed_smc_probe.json): tempered flow moves with
            # SYSTEMATIC RESAMPLING of the (particle, ancestor, covariance)
            # triple between stages; increment = SMC-sampler normalizer
            # telescope, unbiased on the extended space. On frozen Austria
            # this held takeoff-step stage-ESS at 59-88% vs ~2% historical.
            stage_rng = np.random.default_rng(resample_seed + time_index)
            current = pre_flow
            stage_anchors = anchors
            stage_states = states
            stage_prior = flow_prior
            stage_pm, stage_pc = predicted_means, predicted_covs
            increment = tf.zeros([], dtype)
            min_stage_ess = tf.cast(particle_count, dtype)
            prev_obs_log = callbacks.observation_log_density_fn(
                current, observation, time_index
            )
            prev_trans_log = callbacks.transition_log_density_fn(
                current, stage_states, time_index
            )
            for stage in range(1, stage_count + 1):
                flow = ledh_flow_per_particle(
                    anchor_states=stage_anchors,
                    pre_flow_states=current,
                    predicted_covariances=stage_prior
                    / tf.cast(stage_count, dtype),
                    observation=observation,
                    observation_fn=lambda p, _t=time_index: callbacks.observation_fn(p, _t),
                    observation_jacobian_fn=lambda p, _t=time_index: callbacks.observation_jacobian_fn(p, _t),
                    observation_covariance=tf.convert_to_tensor(
                        callbacks.observation_covariance, dtype
                    )
                    * tf.cast(stage_count, dtype),
                    prior_means=stage_anchors,
                    substeps=flow_substeps,
                )
                moved = flow["post_flow_states"]
                new_obs_log = callbacks.observation_log_density_fn(
                    moved, observation, time_index
                )
                new_trans_log = callbacks.transition_log_density_fn(
                    moved, stage_states, time_index
                )
                fraction = tf.cast(stage / stage_count, dtype)
                prev_fraction = tf.cast((stage - 1) / stage_count, dtype)
                stage_logits = (
                    new_trans_log
                    + fraction * new_obs_log
                    + flow["forward_log_det"]
                    - prev_trans_log
                    - prev_fraction * prev_obs_log
                )
                increment += tf.reduce_logsumexp(
                    stage_logits
                    - tf.math.log(tf.cast(particle_count, dtype))
                )
                stage_w = tf.exp(
                    stage_logits - tf.reduce_logsumexp(stage_logits)
                )
                min_stage_ess = tf.minimum(
                    min_stage_ess,
                    1.0 / tf.reduce_sum(tf.square(stage_w)),
                )
                positions = (
                    stage_rng.uniform() + np.arange(particle_count)
                ) / particle_count
                cumulative = np.cumsum(stage_w.numpy())
                cumulative[-1] = 1.0
                idx = tf.constant(
                    np.searchsorted(cumulative, positions).astype(np.int64),
                    tf.int32,
                )
                # Triple discipline: everything ancestry-linked gathers
                current = tf.gather(moved, idx)
                stage_anchors = tf.gather(stage_anchors, idx)
                stage_states = tf.gather(stage_states, idx)
                stage_prior = tf.gather(stage_prior, idx)
                stage_pm = tf.gather(stage_pm, idx)
                stage_pc = tf.gather(stage_pc, idx)
                prev_obs_log = callbacks.observation_log_density_fn(
                    current, observation, time_index
                )
                prev_trans_log = callbacks.transition_log_density_fn(
                    current, stage_states, time_index
                )
            children = current
            predicted_means, predicted_covs = stage_pm, stage_pc
            states = stage_states
            step_weights = tf.fill(
                [particle_count], tf.cast(1.0 / particle_count, dtype)
            )
            ess_proxy = min_stage_ess
        else:
            current = pre_flow
            total_log_det = tf.zeros([particle_count], dtype)
            for _stage in range(stage_count):
                flow = ledh_flow_per_particle(
                    anchor_states=anchors,
                    pre_flow_states=current,
                    predicted_covariances=flow_prior
                    / tf.cast(stage_count, dtype),
                    observation=observation,
                    observation_fn=lambda p, _t=time_index: callbacks.observation_fn(p, _t),
                    observation_jacobian_fn=lambda p, _t=time_index: callbacks.observation_jacobian_fn(p, _t),
                    observation_covariance=tf.convert_to_tensor(
                        callbacks.observation_covariance, dtype
                    )
                    * tf.cast(stage_count, dtype),
                    prior_means=anchors,
                    substeps=flow_substeps,
                )
                current = flow["post_flow_states"]
                total_log_det += flow["forward_log_det"]
            children = current

            transition_log = callbacks.transition_log_density_fn(
                children, states, time_index
            )
            observation_log = callbacks.observation_log_density_fn(
                children, observation, time_index
            )
            # Li(17) eq. bf-pfpf-alg1-weight: the denominator is the
            # TRANSITION density of the pre-flow sample, p(eta_0|ancestor).
            proposal_log = callbacks.transition_log_density_fn(
                pre_flow, states, time_index
            )
            logits = (
                tf.math.log(weights)
                + transition_log
                + observation_log
                + total_log_det
                - proposal_log
            )
            increment = tf.reduce_logsumexp(logits)
            step_weights = tf.exp(logits - increment)
            ess_proxy = None
        step_valid = (
            tf.reduce_all(tf.math.is_finite(children))
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(step_weights))
        )
        ess = (
            ess_proxy
            if ess_proxy is not None
            else 1.0 / tf.reduce_sum(tf.square(step_weights))
        )
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
