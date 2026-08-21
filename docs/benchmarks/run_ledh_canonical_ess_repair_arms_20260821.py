"""P5-ESS repair-arm evaluation (evidence-contract diagnostic).

Question: which lever restores canonical-lane ESS on the Austria scope —
(a) initial-covariance fidelity (model-spread init vs unit init),
(b) UKF-update contraction verified through the filter's own loop,
(c) tempered weighting (ESS-floor-triggered annealing, the 2026-08-20
    degeneracy remedy),
(d) flow substeps.
Comparator: bootstrap ESS on the identical fixture (176/128/73 recorded).
Primary: min per-step ESS fraction per arm. Descriptive only — no
promotion; results feed the P6 calibration contract.
Hard veto per arm: nonfinite value. Everything else is data.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "highdim"))

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def main() -> None:
    from test_ledh_canonical_models import _austria_fixture
    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
        ukf_predict_per_particle,
        ukf_update_per_particle,
    )
    from bayesfilter.highdim.ledh_flow_perparticle_tf import (
        ledh_flow_per_particle,
    )

    n, horizon = 256, 3
    model, _sd, theta0, initial, covs, noises, observations = (
        _austria_fixture(83, n=n, horizon=horizon)
    )
    variance = 100.0

    def run_lane(
        *,
        initial_states,
        initial_cov_scale: float,
        substeps: int,
        temper_stages: int,
        with_update: bool,
    ):
        states = initial_states
        covariances = tf.constant(
            np.stack([np.eye(18) * initial_cov_scale] * n), DTYPE
        )
        ess_track = []
        for t in range(horizon):
            obs = observations[t]
            mean_fn = lambda p: model.transition_mean_fn(theta0, p)
            pm, pc = ukf_predict_per_particle(
                states, covariances, mean_fn, model.process_covariance
            )
            anchors = mean_fn(states)
            pre = anchors + noises[t]
            current = pre
            log_temper_weight = tf.zeros([n], DTYPE)
            for stage in range(temper_stages):
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
                log_temper_weight += flow["forward_log_det"]
            children = current
            res_t = children - anchors
            tl = -0.5 * tf.reduce_sum(tf.square(res_t), axis=1)
            observed = model.observation_fn(children)
            ol = -0.5 * tf.reduce_sum(
                tf.square(obs[None, :] - observed), axis=1
            ) / variance
            pl = -0.5 * tf.reduce_sum(tf.square(pre - anchors), axis=1)
            logits = tl + ol + log_temper_weight - pl
            logits -= tf.reduce_max(logits)
            w = tf.exp(logits)
            w = w / tf.reduce_sum(w)
            ess_track.append(float(1.0 / tf.reduce_sum(tf.square(w))))
            if with_update:
                _pm2, covariances = ukf_update_per_particle(
                    pm,
                    pc,
                    model.observation_fn,
                    model.observation_covariance,
                    obs,
                )
            else:
                covariances = pc
            # equal-weight carry (reset='none' smoke slice, as the failing gate)
            states = children
        return ess_track

    results = {}
    results["baseline_unit_init"] = run_lane(
        initial_states=initial,
        initial_cov_scale=1.0,
        substeps=16,
        temper_stages=1,
        with_update=True,
    )
    # (a) tight model-faithful initial spread: the fixture draws initial
    # particles at unit spread around the mean; arm uses matched cov
    results["arm_a_tight_init_cloud"] = run_lane(
        initial_states=(
            tf.reduce_mean(initial, axis=0)[None, :]
            + 0.1 * (initial - tf.reduce_mean(initial, axis=0)[None, :])
        ),
        initial_cov_scale=0.01,
        substeps=16,
        temper_stages=1,
        with_update=True,
    )
    # (b) no-update ablation (isolate the update's contribution)
    results["arm_b_no_update"] = run_lane(
        initial_states=initial,
        initial_cov_scale=1.0,
        substeps=16,
        temper_stages=1,
        with_update=False,
    )
    # (c) tempering ladder
    for stages in (2, 4):
        results[f"arm_c_temper_{stages}"] = run_lane(
            initial_states=initial,
            initial_cov_scale=1.0,
            substeps=16,
            temper_stages=stages,
            with_update=True,
        )
    # (d) substep ladder
    for sub in (48,):
        results[f"arm_d_substeps_{sub}"] = run_lane(
            initial_states=initial,
            initial_cov_scale=1.0,
            substeps=sub,
            temper_stages=1,
            with_update=True,
        )

    results["arm_ac_combined"] = run_lane(
        initial_states=(
            tf.reduce_mean(initial, axis=0)[None, :]
            + 0.1 * (initial - tf.reduce_mean(initial, axis=0)[None, :])
        ),
        initial_cov_scale=0.01,
        substeps=16,
        temper_stages=4,
        with_update=True,
    )
    print(json.dumps({k: [round(v, 2) for v in vals] for k, vals in results.items()}, indent=1))


if __name__ == "__main__":
    main()

