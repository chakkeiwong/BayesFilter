"""Q2 Curve 7: Austria Fisher-identity gate at GPU-replication scale.

Contract: `docs/plans/bayesfilter-q2-calibration-campaign-plan-2026-08-24.md`
(Curve 7). E_{data ~ p(.|theta0)}[score(theta0)] = 0 for the exact
likelihood score; the particle estimate adds O(1/N) bias + MC spread, so
the gate is |mean| < 3*SE + bias_slack per direction, with the
non-vacuity guard (identically-zero scores are a hard failure).

Directions 0..2 (kappa, nu, observation log-scale). theta_2's density/
covariance tangents were wired 2026-08-24 and oracle-gated
(`test_austria_r_direction_score_matches_oracle`).

Simulation note (recorded): truth trajectories use the model's own
`transition_mean_fn` (RK4 with the vendored half-step-k4 quirk) plus
independent numpy noise; the mean map is shared with the model under
test, the noise/density structure is independent. Sharing the mean map
does not weaken the identity's validity (simulation law == model law is
exactly what the identity requires); it does mean this gate probes
density families and score derivation, not RK4-implementation identity
(that has its own vendored gate).

Score lane is single-cloud float64 by design; GPU f64 is accepted for
this diagnostic and recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402  (simulation + diagnostics)
import tensorflow as tf  # noqa: E402

DTYPE = tf.float64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=40)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--bias-slack", type=float, default=0.05)
    args = parser.parse_args()
    started = time.time()

    gpus = tf.config.list_physical_devices("GPU")
    memory_growth_verified = False
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        memory_growth_verified = all(
            tf.config.experimental.get_memory_growth(gpu) for gpu in gpus
        )
        if not memory_growth_verified:
            raise SystemExit("FAIL-CLOSED: memory growth not verified")
    device = "/GPU:0" if gpus else "/CPU:0"

    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, set_direction = austria_sir_canonical_model(theta0)
    initial_mean = zhao_cui_sir_austria_model().initial_mean.numpy()
    obs_matrix = np.zeros((9, 18))
    for o in range(9):
        obs_matrix[o, 2 * o + 1] = 1.0
    n, horizon = args.particles, args.horizon
    obs_scale = 10.0  # sqrt(100*exp(2*0))

    rng = np.random.default_rng(3001)
    results = {}
    veto = False
    with tf.device(device):
        for direction_index in range(3):
            one_hot = np.zeros(3)
            one_hot[direction_index] = 1.0
            set_direction(tf.constant(one_hot, DTYPE))
            scores = []
            for _rep in range(args.replications):
                # simulate truth from the model law (see module note)
                truth = initial_mean + rng.standard_normal(18)
                observations = []
                for _t in range(horizon):
                    truth = model.transition_mean_fn(
                        theta0, tf.constant(truth[None, :], DTYPE)
                    ).numpy()[0] + rng.standard_normal(18)
                    observations.append(
                        obs_matrix @ truth
                        + obs_scale * rng.standard_normal(9)
                    )
                observations_tf = tf.constant(
                    np.array(observations), DTYPE
                )
                initial = tf.constant(
                    initial_mean[None, :]
                    + rng.standard_normal((n, 18)),
                    DTYPE,
                )
                covs = tf.constant(np.stack([np.eye(18)] * n), DTYPE)
                noises = tf.constant(
                    rng.standard_normal((horizon, n, 18)), DTYPE
                )
                value, score = canonical_value_and_analytical_score(
                    model,
                    theta0,
                    initial,
                    covs,
                    noises,
                    observations_tf,
                    substeps=8,
                    with_score=True,
                )
                if not np.isfinite(float(value.numpy())):
                    veto = True
                scores.append(float(score[0].numpy()))
            scores = np.array(scores)
            spread = float(np.std(scores))
            mean = float(np.mean(scores))
            se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
            vacuous = spread < 1.0e-12
            passed = (not vacuous) and abs(mean) < 3.0 * se + args.bias_slack
            if vacuous:
                veto = True
            results[f"direction_{direction_index}"] = {
                "mean_score": mean,
                "se": se,
                "std": spread,
                "gate": f"|mean| < 3*SE + {args.bias_slack}",
                "vacuous": vacuous,
                "passed": bool(passed),
            }
            print(
                f"[dir {direction_index}] mean={mean:.4f} se={se:.4f} "
                f"passed={passed}",
                flush=True,
            )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema": "bayesfilter.q2_austria_fisher_gpu.v1",
        "plan": "docs/plans/bayesfilter-q2-calibration-campaign-plan-2026-08-24.md",
        "manifest": {
            "commit": commit,
            "command": " ".join(sys.argv),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "unset"
            ),
            "device": device,
            "memory_growth_verified": memory_growth_verified,
            "dtype": "float64",
            "particles": n,
            "horizon": horizon,
            "replications": args.replications,
            "sim_seed": 3001,
            "wall_seconds": round(time.time() - started, 1),
        },
        "hard_veto_fired": veto,
        "directions": results,
        "all_passed": bool(
            (not veto)
            and all(r["passed"] for r in results.values())
        ),
    }
    out_dir = os.path.join(
        _ROOT, "docs", "benchmarks", "q2_calibration_20260824",
        f"austria_fisher_gpu_{int(started)}",
    )
    os.makedirs(out_dir, exist_ok=False)
    out_path = os.path.join(out_dir, "result.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(
        f"[done] all_passed={payload['all_passed']} artifact={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
