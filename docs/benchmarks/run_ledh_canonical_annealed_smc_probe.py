"""P6 within-step annealed-SMC probe (contract in execution-plan ledger).

Per step: k annealing stages; tempered flow move; systematic resampling of
(particle, ancestor) pairs between stages; SMC-sampler normalizer
telescope for the increment. Frozen Austria tensors, float64 CPU.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def systematic_indices(weights: np.ndarray, count: int, offset: float) -> np.ndarray:
    positions = (offset + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions).astype(np.int64)


def main() -> None:
    started = time.time()
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        make_canonical_neutra_target,
    )
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
        ukf_predict_per_particle,
        ukf_update_per_particle,
    )
    from bayesfilter.highdim.ledh_flow_perparticle_tf import (
        ledh_flow_per_particle,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    with tf.device("/CPU:0"):
        target = make_canonical_neutra_target(
            "austria_sir", particle_count=1008
        )
    n = 1008
    horizon = int(target.observations.shape[0])
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, _sd = austria_sir_canonical_model(theta0)
    observations = target.observations
    process_noise = target.noises
    initial = target.initial_states
    variance = 100.0
    obs_norm = 4.5 * np.log(2.0 * np.pi * variance)
    cap_c = 8.0

    def spectral_cap(covs):
        eigenvalues, eigenvectors = tf.linalg.eigh(covs)
        capped = tf.minimum(eigenvalues, tf.cast(cap_c, DTYPE))
        return tf.einsum(
            "nij,nj,nkj->nik", eigenvectors, capped, eigenvectors
        )

    def obs_log(points, obs):
        observed = model.observation_fn(points)
        return (
            -0.5
            * tf.reduce_sum(tf.square(obs[None, :] - observed), axis=1)
            / variance
            - obs_norm
        )

    def trans_log(points, anchors):
        return -0.5 * tf.reduce_sum(
            tf.square(points - anchors), axis=1
        ) - 9.0 * np.log(2.0 * np.pi)

    def annealed_lane(k_stages: int, seed: int):
        rng = np.random.default_rng(seed)
        states = initial
        covariances = tf.eye(18, batch_shape=[n], dtype=DTYPE)
        min_stage_ess, total = {}, 0.0
        for t in range(horizon):
            obs = observations[t]
            mean_fn = lambda p: model.transition_mean_fn(theta0, p)
            pm, pc = ukf_predict_per_particle(
                states, covariances, mean_fn, model.process_covariance
            )
            flow_prior = spectral_cap(pc)
            anchors = mean_fn(states)
            current = anchors + process_noise[t]
            stage_ess_track = []
            prev_ol = obs_log(current, obs)
            prev_tl = trans_log(current, anchors)
            for s in range(1, k_stages + 1):
                flow = ledh_flow_per_particle(
                    anchor_states=anchors,
                    pre_flow_states=current,
                    predicted_covariances=flow_prior / k_stages,
                    observation=obs,
                    observation_fn=model.observation_fn,
                    observation_jacobian_fn=model.observation_jacobian_fn,
                    observation_covariance=model.observation_covariance
                    * k_stages,
                    prior_means=anchors,
                    substeps=12,
                )
                moved = flow["post_flow_states"]
                new_ol = obs_log(moved, obs)
                new_tl = trans_log(moved, anchors)
                lw = (
                    new_tl
                    + (s / k_stages) * new_ol
                    + flow["forward_log_det"]
                    - prev_tl
                    - ((s - 1) / k_stages) * prev_ol
                )
                lse = tf.reduce_logsumexp(lw - np.log(n))
                total += float(lse.numpy())
                w = tf.exp(lw - tf.reduce_logsumexp(lw))
                ess = float(1.0 / tf.reduce_sum(tf.square(w)).numpy())
                stage_ess_track.append(ess)
                indices = systematic_indices(
                    w.numpy(), n, rng.uniform()
                )
                idx = tf.constant(indices, tf.int32)
                current = tf.gather(moved, idx)
                anchors = tf.gather(anchors, idx)
                pc_gathered = tf.gather(pc, idx)
                flow_prior = tf.gather(flow_prior, idx)
                states_prev = tf.gather(states, idx)
                prev_ol = obs_log(current, obs)
                prev_tl = trans_log(current, anchors)
            min_stage_ess[t] = round(min(stage_ess_track), 1)
            _pm2, covariances = ukf_update_per_particle(
                tf.gather(pm, idx), pc_gathered,
                model.observation_fn, model.observation_covariance, obs,
            )
            states = current
        return total, min_stage_ess

    cells = {}
    for k in (4, 8):
        value, stage_ess = annealed_lane(k, seed=17)
        cells[f"annealed_k{k}"] = {
            "value": value,
            "takeoff_min_stage_ess": [stage_ess[2], stage_ess[4]],
            "worst_step_min_stage_ess": min(stage_ess.values()),
            "per_step_min_stage_ess": stage_ess,
            "pass_10pct_takeoff": bool(
                stage_ess[2] > 0.10 * n and stage_ess[4] > 0.10 * n
            ),
        }
        print(f"annealed_k{k}: value={value:.3f} takeoff={cells[f'annealed_k{k}']['takeoff_min_stage_ess']} worst={cells[f'annealed_k{k}']['worst_step_min_stage_ess']}")

    payload = {
        "schema": "bayesfilter.ledh_canonical_annealed_smc_probe.v1",
        "contract": "P6 Within-Step Annealed-SMC Contract (ledger)",
        "flow_prior_cap": cap_c,
        "cells": cells,
        "wall_seconds": time.time() - started,
    }
    out = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "annealed_smc_probe.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)


if __name__ == "__main__":
    main()
