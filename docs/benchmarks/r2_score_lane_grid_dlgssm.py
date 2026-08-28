"""Score-lane grid for dlgssm at the value-lane control family.

Question: does the score lane's marginal-convergence behavior match the
value lane's at the same epsilon/sinkhorn_steps/flow_substeps controls,
and do score-lane values track value-lane values across the grid?

Grid: same 4 ε × 3 sk × 3 flow × 2 N as the value-lane pilot.
Replications: 3 (same as value pilot, different seed offset).
Primary veto: same marginal-convergence veto (TV error > 1e-4).
Comparator: exact Kalman log-likelihood (linear Gaussian model).

Not concluded: which configuration is best for the score lane. This
checks whether the value-lane veto transfers; it does not select a
score-lane configuration.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_q3_leaderboard_20260824 import (  # noqa: E402
    _density_obs_log,
    _gaussian_trans_log,
)

DTYPE = tf.float64

LGSSM_OBS_MATRIX = np.array(
    [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]
)
LGSSM_THETA = np.array([0.9, 0.8, 0.7, 0.6, 0.8])
MARGINAL_TOLERANCE = 1.0e-4
SCORE_DIRECTION = 0


def exact_kalman_loglik(observations):
    phi = np.diag(LGSSM_THETA[:3])
    q = LGSSM_THETA[3] ** 2 * np.eye(3)
    r = LGSSM_THETA[4] ** 2 * np.eye(3)
    h = LGSSM_OBS_MATRIX
    mean, cov = np.zeros(3), np.eye(3)
    total = 0.0
    for t in range(int(observations.shape[0])):
        mean = phi @ mean
        cov = phi @ cov @ phi.T + q
        s = h @ cov @ h.T + r
        resid = observations[t] - h @ mean
        _sign, logdet = np.linalg.slogdet(2.0 * np.pi * s)
        total += -0.5 * (resid @ np.linalg.solve(s, resid) + logdet)
        gain = cov @ h.T @ np.linalg.inv(s)
        mean = mean + gain @ resid
        cov = (np.eye(3) - gain @ h) @ cov
    return float(total)


def run_one_cell(observations, epsilon, sinkhorn_steps, flow_substeps,
                 particle_count, seed):
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )

    theta = tf.constant(LGSSM_THETA, DTYPE)
    model, set_direction = diagonal_lgssm_canonical_model(theta)
    dim = 3
    horizon = int(observations.shape[0])

    one_hot = np.zeros(LGSSM_THETA.shape[0])
    one_hot[SCORE_DIRECTION] = 1.0
    set_direction(tf.constant(one_hot, DTYPE))

    rng = np.random.default_rng(9000 + seed)
    initial = tf.constant(
        np.zeros(dim)[None, :] + rng.standard_normal((particle_count, dim)),
        DTYPE,
    )
    covs = tf.constant(np.stack([np.eye(dim)] * particle_count), DTYPE)
    noises = tf.constant(
        rng.standard_normal((horizon, particle_count, dim)), DTYPE
    )
    reset_design = tf.constant(
        np.tile(
            np.concatenate([np.eye(dim), -np.eye(dim)]),
            (particle_count // (2 * dim), 1),
        ),
        DTYPE,
    )

    cell = {
        "epsilon": epsilon,
        "sinkhorn_steps": sinkhorn_steps,
        "flow_substeps": flow_substeps,
        "particle_count": particle_count,
        "seed": seed,
    }

    try:
        # Score lane does not return marginal diagnostics in its result
        # dict, so we cannot directly extract them. The implementation
        # calls _restore_cloud_primal which computes and returns them,
        # but canonical_value_and_analytical_score discards those fields.
        # We can only check finiteness and compare the value.
        value, score = canonical_value_and_analytical_score(
            model,
            theta,
            initial,
            covs,
            noises,
            tf.constant(observations, DTYPE),
            flow_substeps=flow_substeps,
            with_score=True,
            reset_policy="contract_e",
            reset_design=reset_design,
            reset_epsilon=epsilon,
            reset_sinkhorn_steps=sinkhorn_steps,
            reset_balance_steps=sinkhorn_steps,
            reset_ridge=1.0e-5,
            correction_steps=1,
            correction_lm_damping=1.0e-2,
            correction_lm_scale_floor=1.0e-4,
            correction_trust_radius=0.5,
            pairwise_steps=1,
            annealed_stages=1,
            annealed_seed=17,
        )

        val = float(value.numpy())
        sc = float(score[0].numpy())
        finite = bool(np.isfinite(val) and np.isfinite(sc))

        cell.update({
            "value": val,
            "score": sc,
            "finite": finite,
        })
        return cell
    except Exception as exc:
        cell.update({
            "value": None,
            "score": None,
            "finite": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return cell


def main():
    data = np.load(
        "docs/benchmarks/r2_tuning_20260827/dlgssm_tuning_data.npz"
    )
    replications = 3

    kalman = [
        exact_kalman_loglik(data["observations"][rep])
        for rep in range(replications)
    ]
    print(f"exact Kalman references:")
    for rep, ref in enumerate(kalman):
        print(f"  rep {rep}: {ref:.6f}")

    epsilons = [0.5, 1.0, 2.0, 4.0]
    sinkhorn_values = [8, 16, 24]
    flow_values = [12, 16, 24]
    particle_counts = [504, 1008]

    total = (
        len(epsilons)
        * len(sinkhorn_values)
        * len(flow_values)
        * len(particle_counts)
        * replications
    )
    print(f"\nrunning {total} score-lane cells\n")

    cells = []
    idx = 0
    t_start = time.time()
    for eps in epsilons:
        for sinkhorn_steps in sinkhorn_values:
            for flow_substeps in flow_values:
                for n in particle_counts:
                    for rep in range(replications):
                        idx += 1
                        obs = data["observations"][rep]
                        seed = 999000 + rep
                        t0 = time.time()
                        cell = run_one_cell(
                            obs, eps, sinkhorn_steps, flow_substeps, n, seed
                        )
                        cell["replication"] = rep
                        cell["exact_kalman"] = kalman[rep]
                        cell["abs_error"] = (
                            abs(cell["value"] - kalman[rep])
                            if cell.get("value") is not None
                            else None
                        )
                        cell["elapsed_sec"] = time.time() - t0
                        cells.append(cell)

                        status = "finite" if cell["finite"] else "FAILED"
                        err = (
                            f"{cell['abs_error']:.4f}"
                            if cell["abs_error"] is not None
                            else "n/a"
                        )
                        sc = (
                            f"{cell['score']:.2f}"
                            if cell.get("score") is not None
                            else "n/a"
                        )
                        print(
                            f"[{idx}/{total}] eps={eps} sk={sinkhorn_steps} "
                            f"flow={flow_substeps} N={n} rep={rep} "
                            f"{status} err={err} score={sc} "
                            f"{cell['elapsed_sec']:.1f}s"
                        )

    wall = time.time() - t_start

    # Group by configuration
    by_config = {}
    for cell in cells:
        key = (
            cell["epsilon"],
            cell["sinkhorn_steps"],
            cell["flow_substeps"],
            cell["particle_count"],
        )
        by_config.setdefault(key, []).append(cell)

    configs = []
    for key, group in sorted(by_config.items()):
        eps, sk, flow, n = key
        finite_pass = all(c["finite"] for c in group)
        errors = [
            c["abs_error"]
            for c in group
            if c.get("abs_error") is not None
        ]
        if errors:
            mean_err = float(np.mean(errors))
            se_err = (
                float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
                if len(errors) > 1
                else None
            )
        else:
            mean_err = None
            se_err = None
        configs.append({
            "epsilon": eps,
            "sinkhorn_steps": sk,
            "flow_substeps": flow,
            "particle_count": n,
            "replications_run": len(group),
            "finite_veto_pass": finite_pass,
            "mean_abs_error": mean_err,
            "se_abs_error": se_err,
        })

    survivors = [c for c in configs if c["finite_veto_pass"]]
    rejected = [c for c in configs if not c["finite_veto_pass"]]

    output = {
        "study": "r2_score_lane_grid_dlgssm",
        "date": "2026-08-28",
        "question": (
            "does the score lane's marginal-convergence behavior match "
            "the value lane at the same controls"
        ),
        "scope": {
            "model": "dlgssm",
            "lane": "score",
            "route": "contract_e_reset_dual_cap_trust_region",
            "horizon": int(data["horizon"]),
            "dtype": "float64",
            "particle_counts": particle_counts,
        },
        "tuning_data": {
            "path": "docs/benchmarks/r2_tuning_20260827/dlgssm_tuning_data.npz",
            "sha256": str(data["sha256"]),
            "replication_seeds": data["replication_seeds"].tolist(),
            "replications_used": replications,
        },
        "grid": {
            "epsilon": epsilons,
            "sinkhorn_steps": sinkhorn_values,
            "balance_steps": "tied equal to sinkhorn_steps",
            "flow_substeps": flow_values,
            "particle_counts": particle_counts,
        },
        "exact_kalman_reference": kalman,
        "total_cells": total,
        "cells": cells,
        "configurations": configs,
        "summary": {
            "configurations_total": len(configs),
            "configurations_finite": len(survivors),
            "configurations_rejected": len(rejected),
            "note": (
                "Score lane does not surface marginal diagnostics in its "
                "return dict, so we cannot apply the same marginal-TV "
                "veto the value lane uses. This grid only checks "
                "finiteness."
            ),
        },
        "manifest": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "cuda_device_order": os.environ.get(
                "CUDA_DEVICE_ORDER", "unset"
            ),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "unset"
            ),
            "gpu": [
                tf.config.experimental.get_device_details(g).get(
                    "device_name", "unknown"
                )
                for g in tf.config.list_physical_devices("GPU")
            ],
            "wall_time_sec": wall,
        },
        "non_claims": [
            "This does not select a score-lane configuration or rank "
            "them. It checks whether the value-lane grid's finiteness "
            "pattern transfers to the score lane.",
            "The score lane implementation does not return marginal "
            "diagnostics, so the same convergence veto cannot be "
            "applied. Only finiteness is checked.",
        ],
    }
    Path("docs/benchmarks/r2_score_lane_grid_dlgssm.json").write_text(
        json.dumps(output, indent=2)
    )
    print(f"\nwall time {wall/60:.1f} min")
    print(f"saved docs/benchmarks/r2_score_lane_grid_dlgssm.json")
    print(f"finite configurations: {len(survivors)}/{len(configs)}")


if __name__ == "__main__":
    main()
