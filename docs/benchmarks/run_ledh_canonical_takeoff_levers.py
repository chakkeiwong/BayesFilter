"""P6 takeoff-lever ladder (contract in the execution-plan ledger).

Lever A: spectral cap on the flow-prior covariance, P_capped =
U min(Lambda, c) U^T (Q = I on this scope, so c is in Q units).
Lever B: temper stages. Ladder: c in {2, 8, 32, inf} x stages in {4, 8}.
Primary: ESS at steps 2 and 4; threshold > 10% of N declared in contract.
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


def main() -> None:
    started = time.time()
    from bayesfilter.highdim.cubature_genut_neutra_targets import (
        make_genut_neutra_target,
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
        target = make_genut_neutra_target(
            "austria_sir", particle_count=1008
        )
    n = 1008
    horizon = int(target.observations.shape[0])
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, _sd = austria_sir_canonical_model(theta0)
    observations = tf.cast(target.observations, DTYPE)
    initial_noise = tf.cast(target.initial_noise, DTYPE)
    process_noise = tf.cast(target.process_noise, DTYPE)
    initial_mean = tf.cast(zhao_cui_sir_austria_model().initial_mean, DTYPE)
    initial = initial_mean[None, :] + initial_noise
    variance = 100.0
    obs_norm = 4.5 * np.log(2.0 * np.pi * variance)

    def spectral_cap(covs: tf.Tensor, cap: float) -> tf.Tensor:
        if not np.isfinite(cap):
            return covs
        eigenvalues, eigenvectors = tf.linalg.eigh(covs)
        capped = tf.minimum(eigenvalues, tf.cast(cap, DTYPE))
        return tf.einsum(
            "nij,nj,nkj->nik", eigenvectors, capped, eigenvectors
        )

    def lane(cap: float, stages: int):
        states = initial
        covariances = tf.eye(18, batch_shape=[n], dtype=DTYPE)
        ess_track, total = [], 0.0
        for t in range(horizon):
            obs = observations[t]
            mean_fn = lambda p: model.transition_mean_fn(theta0, p)
            pm, pc = ukf_predict_per_particle(
                states, covariances, mean_fn, model.process_covariance
            )
            flow_prior = spectral_cap(pc, cap)
            anchors = mean_fn(states)
            pre = anchors + process_noise[t]
            current = pre
            log_det = tf.zeros([n], DTYPE)
            for _ in range(stages):
                flow = ledh_flow_per_particle(
                    anchor_states=anchors,
                    pre_flow_states=current,
                    predicted_covariances=flow_prior / stages,
                    observation=obs,
                    observation_fn=model.observation_fn,
                    observation_jacobian_fn=model.observation_jacobian_fn,
                    observation_covariance=model.observation_covariance
                    * stages,
                    prior_means=anchors,
                    substeps=16,
                )
                current = flow["post_flow_states"]
                log_det += flow["forward_log_det"]
            children = current
            tl = -0.5 * tf.reduce_sum(tf.square(children - anchors), axis=1)
            observed = model.observation_fn(children)
            ol = (
                -0.5
                * tf.reduce_sum(
                    tf.square(obs[None, :] - observed), axis=1
                )
                / variance
                - obs_norm
            )
            pl = -0.5 * tf.reduce_sum(tf.square(pre - anchors), axis=1)
            logits = -np.log(n) + tl + ol + log_det - pl
            lse = tf.reduce_logsumexp(logits)
            total += float(lse.numpy())
            w = tf.exp(logits - lse)
            ess_track.append(float(1.0 / tf.reduce_sum(tf.square(w))))
            _pm2, covariances = ukf_update_per_particle(
                pm, pc, model.observation_fn,
                model.observation_covariance, obs,
            )
            states = children
        return total, ess_track

    cells = {}
    for cap in (2.0, 8.0, 32.0, float("inf")):
        for stages in (4, 8):
            value, ess = lane(cap, stages)
            key = f"cap_{cap:g}_stages_{stages}"
            cells[key] = {
                "value": value,
                "takeoff_ess": [round(ess[2], 1), round(ess[4], 1)],
                "min_ess": round(min(ess), 1),
                "pass_10pct": bool(
                    ess[2] > 0.10 * n and ess[4] > 0.10 * n
                ),
            }
            print(key, "->", cells[key])

    payload = {
        "schema": "bayesfilter.ledh_canonical_takeoff_levers.v1",
        "contract": "P6 Takeoff-Lever Contract (execution-plan ledger 2026-08-22)",
        "cells": cells,
        "wall_seconds": time.time() - started,
    }
    out = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "takeoff_levers.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)


if __name__ == "__main__":
    main()
