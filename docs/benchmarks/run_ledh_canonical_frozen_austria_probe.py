"""Canonical lane on the REAL frozen Austria target (hard-regime probe).

Question: on the frozen T=20/N=1008 Austria tensors — the scope where the
bootstrap NeuTra lane collapsed to ESS ~23/1008 — does the canonical
UKF+flow+tempered lane maintain healthy ESS?

Diagnostic lane, float64 CPU (frozen float32 tensors upcast; this is a
proposal-quality measurement on the frozen data, NOT a frozen-scope
claim-bearing run — hashes change under upcast by construction).
Both lanes share the model functions and frozen tensors; only the
proposal mechanism differs. Descriptive; no promotion.
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

    with tf.device("/CPU:0"):
        target = make_canonical_neutra_target(
            "austria_sir", particle_count=1008
        )
    n = 1008
    horizon = int(target.observations.shape[0])
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, _set_dir = austria_sir_canonical_model(theta0)

    observations = target.observations
    process_noise = target.noises
    initial = target.initial_states
    variance = 100.0
    obs_norm = 4.5 * np.log(2.0 * np.pi * variance)

    def canonical_lane(temper_stages: int, substeps: int):
        states = initial
        covariances = tf.eye(18, batch_shape=[n], dtype=DTYPE)
        ess_track, total = [], 0.0
        for t in range(horizon):
            obs = observations[t]
            mean_fn = lambda p: model.transition_mean_fn(theta0, p)
            pm, pc = ukf_predict_per_particle(
                states, covariances, mean_fn, model.process_covariance
            )
            anchors = mean_fn(states)
            pre = anchors + process_noise[t]
            current = pre
            log_det = tf.zeros([n], DTYPE)
            for _ in range(temper_stages):
                flow = ledh_flow_per_particle(
                    anchor_states=anchors,
                    pre_flow_states=current,
                    predicted_covariances=pc / temper_stages,
                    observation=obs,
                    observation_fn=model.observation_fn,
                    observation_jacobian_fn=model.observation_jacobian_fn,
                    observation_covariance=model.observation_covariance
                    * temper_stages,
                    prior_means=anchors,
                    substeps=substeps,
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

    def bootstrap_lane():
        states = initial
        ess_track, total = [], 0.0
        for t in range(horizon):
            obs = observations[t]
            anchors = model.transition_mean_fn(theta0, states)
            children = anchors + process_noise[t]
            observed = model.observation_fn(children)
            ol = (
                -0.5
                * tf.reduce_sum(
                    tf.square(obs[None, :] - observed), axis=1
                )
                / variance
                - obs_norm
            )
            logits = -np.log(n) + ol
            lse = tf.reduce_logsumexp(logits)
            total += float(lse.numpy())
            w = tf.exp(logits - lse)
            ess_track.append(float(1.0 / tf.reduce_sum(tf.square(w))))
            states = children
        return total, ess_track

    b_val, b_ess = bootstrap_lane()
    ladder = {}
    for stages in (1, 2, 4):
        value, ess = canonical_lane(temper_stages=stages, substeps=16)
        ladder[f"temper_{stages}"] = {
            "value": value,
            "min_ess": min(ess),
            "ess_at_hard_steps": [round(ess[2], 1), round(ess[4], 1)],
            "ess": [round(e, 1) for e in ess],
        }
    c_val, c_ess = ladder["temper_4"]["value"], ladder["temper_4"]["ess"]

    payload = {
        "schema": "bayesfilter.ledh_canonical_frozen_austria_probe.v1",
        "scope": "frozen Austria tensors T=20 N=1008, float64 upcast, theta=0",
        "note": "diagnostic proposal-quality probe; not a frozen-scope claim run",
        "bootstrap": {
            "value": b_val,
            "min_ess": min(b_ess),
            "ess": [round(e, 1) for e in b_ess],
        },
        "canonical_ladder": ladder,
        "wall_seconds": time.time() - started,
    }
    out_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08",
    )
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "frozen_austria_probe.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
