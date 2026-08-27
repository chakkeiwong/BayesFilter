"""Canonical leaderboard slice 1: LGSSM + Austria, CPU float64 (Part 4).

Contract: bayesfilter-ledh-canonical-leaderboard-rerun-plan-2026-08-22.md.
Descriptive results; hard veto = nonfinite/invalid rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "highdim"))

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def lgssm_cells() -> dict:
    from test_ledh_canonical_filter import (
        _callbacks_for_lgssm,
        _kalman_log_likelihood,
        _lgssm_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        canonical_value_and_diagnostics,
    )

    rows = []
    for model_seed in (101, 102, 103):
        model = _lgssm_model(model_seed)
        reference = _kalman_log_likelihood(model)
        callbacks = _callbacks_for_lgssm(model)
        values = []
        for particle_seed in (0, 1, 2):
            result = canonical_value_and_diagnostics(
                callbacks,
                tf.constant(model["observations"], DTYPE),
                particle_count=4096,
                seed=particle_seed,
            )
            values.append(float(result["value"].numpy()))
        rows.append(
            {
                "model_seed": model_seed,
                "kalman_exact": reference,
                "canonical_values": values,
                "canonical_mean": float(np.mean(values)),
                "abs_error_of_mean": abs(float(np.mean(values)) - reference),
                "per_seed_spread": float(np.std(values)),
                "finite": all(np.isfinite(v) for v in values),
            }
        )
    return {"lgssm_value_vs_exact_kalman": rows}


def lgssm_score_cell() -> dict:
    from test_ledh_canonical_score_full import _model
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )
    from bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf import (
        oracle_forward_autodiff_score,
    )

    rng = np.random.default_rng(111)
    n, dim, horizon = 8, 2, 4
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    model = _model()
    theta0 = tf.constant([0.6], DTYPE)

    def value_fn(theta):
        value, _ = canonical_value_and_analytical_score(
            model, theta, initial, covs, noises, observations,
            flow_substeps=10, with_score=False,
        )
        return value

    oracle = float(
        oracle_forward_autodiff_score(value_fn, theta0)[0].numpy()
    )
    _, score = canonical_value_and_analytical_score(
        model, theta0, initial, covs, noises, observations,
        flow_substeps=10, with_score=True,
    )
    analytical = float(score[0].numpy())
    return {
        "nonlinear_score_vs_oracle": {
            "analytical": analytical,
            "oracle": oracle,
            "rel_error": abs(analytical - oracle) / max(abs(oracle), 1.0),
        }
    }


def austria_cells() -> dict:
    from test_ledh_canonical_models import _austria_fixture
    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import (
        ukf_predict_per_particle,
        ukf_update_per_particle,
    )
    from bayesfilter.highdim.ledh_flow_perparticle_tf import (
        ledh_flow_per_particle,
    )

    rows = []
    n, horizon = 256, 3
    for seed in (83, 84, 85):
        model, _sd, theta0, initial, covs, noises, observations = (
            _austria_fixture(seed, n=n, horizon=horizon)
        )
        tight_initial = (
            tf.reduce_mean(initial, axis=0)[None, :]
            + 0.1 * (initial - tf.reduce_mean(initial, axis=0)[None, :])
        )

        def canonical_lane():
            states = tight_initial
            covariances = tf.constant(
                np.stack([np.eye(18) * 0.01] * n), DTYPE
            )
            ess_track, total = [], 0.0
            stages = 4
            for t in range(horizon):
                obs = observations[t]
                mean_fn = lambda p: model.transition_mean_fn(theta0, p)
                pm, pc = ukf_predict_per_particle(
                    states, covariances, mean_fn, model.process_covariance
                )
                anchors = mean_fn(states)
                pre = anchors + noises[t]
                current = pre
                log_det = tf.zeros([n], DTYPE)
                for _ in range(stages):
                    flow = ledh_flow_per_particle(
                        anchor_states=anchors,
                        pre_flow_states=current,
                        predicted_covariances=pc / stages,
                        observation=obs,
                        observation_fn=model.observation_fn,
                        observation_jacobian_fn=model.observation_jacobian_fn,
                        observation_covariance=model.observation_covariance
                        * stages,
                        prior_means=anchors,
                        flow_substeps=16,
                    )
                    current = flow["post_flow_states"]
                    log_det += flow["forward_log_det"]
                children = current
                tl = -0.5 * tf.reduce_sum(
                    tf.square(children - anchors), axis=1
                )
                observed = model.observation_fn(children)
                ol = -0.5 * tf.reduce_sum(
                    tf.square(obs[None, :] - observed), axis=1
                ) / 100.0 - 4.5 * np.log(2.0 * np.pi * 100.0)
                pl = -0.5 * tf.reduce_sum(
                    tf.square(pre - anchors), axis=1
                )
                logits = (
                    -np.log(n) + tl + ol + log_det - pl
                )
                increment = float(tf.reduce_logsumexp(logits).numpy())
                total += increment
                w = tf.exp(logits - tf.reduce_logsumexp(logits))
                ess_track.append(
                    float(1.0 / tf.reduce_sum(tf.square(w)).numpy())
                )
                _pm2, covariances = ukf_update_per_particle(
                    pm, pc, model.observation_fn,
                    model.observation_covariance, obs,
                )
                states = children
            return total, ess_track

        def bootstrap_lane():
            states = tight_initial
            ess_track, total = [], 0.0
            for t in range(horizon):
                obs = observations[t]
                anchors = model.transition_mean_fn(theta0, states)
                children = anchors + noises[t]
                observed = model.observation_fn(children)
                ol = -0.5 * tf.reduce_sum(
                    tf.square(obs[None, :] - observed), axis=1
                ) / 100.0 - 4.5 * np.log(2.0 * np.pi * 100.0)
                logits = -np.log(n) + ol
                increment = float(tf.reduce_logsumexp(logits).numpy())
                total += increment
                w = tf.exp(logits - tf.reduce_logsumexp(logits))
                ess_track.append(
                    float(1.0 / tf.reduce_sum(tf.square(w)).numpy())
                )
                states = children
            return total, ess_track

        c_val, c_ess = canonical_lane()
        b_val, b_ess = bootstrap_lane()
        rows.append(
            {
                "seed": seed,
                "canonical_value": c_val,
                "bootstrap_value": b_val,
                "canonical_min_ess": min(c_ess),
                "bootstrap_min_ess": min(b_ess),
                "canonical_ess": [round(e, 1) for e in c_ess],
                "bootstrap_ess": [round(e, 1) for e in b_ess],
                "finite": bool(np.isfinite(c_val) and np.isfinite(b_val)),
            }
        )
    return {"austria_canonical_vs_bootstrap": rows}


def main() -> None:
    started = time.time()
    payload = {
        "schema": "bayesfilter.ledh_canonical_leaderboard_slice1.v1",
        "plan": "docs/plans/bayesfilter-ledh-canonical-leaderboard-rerun-plan-2026-08-22.md",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
        "environment": sys.prefix,
        "dtype": "float64 (CPU reference lane)",
        "conformance_suite": "canonical gates 30/30 at commit",
        "results": {},
    }
    payload["results"].update(lgssm_cells())
    payload["results"].update(lgssm_score_cell())
    payload["results"].update(austria_cells())
    payload["wall_seconds"] = time.time() - started
    out_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08",
    )
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "slice1_result.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload["results"], indent=1))
    print("artifact:", out)


if __name__ == "__main__":
    main()
