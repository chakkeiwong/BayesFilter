"""R2-TUNE Phase 2: Per-scope tuning grid pilot (dlgssm only).

Pilot run before the full six-model campaign. Produces the first per-scope
tuning artifact for the dlgssm value lane and exercises the grid
infrastructure that the remaining five models will reuse.

Scope bound fields (LEDH Per-Scope Tuning Rule): model=dlgssm value lane,
route=Contract-E reset + dual-cap trust region, horizon=50, dtype=float64,
device=single GPU, chunk policy=central selector.

Control family (route-specific: entropic reset controls + flow resolution)
  epsilon        in {0.5, 1.0, 2.0, 4.0}
  sinkhorn_steps in {8, 16, 24}   (balance_steps tied equal)
  flow_substeps  in {12, 16, 24}
  particle_count in {504, 1008}

Evidence contract
  Question: which reset-control cell reproduces the exact Kalman
    log-likelihood most closely on tuning data disjoint from the claim
    partition, among cells whose Sinkhorn marginals actually converge?
  Comparator: exact Kalman filter log-likelihood (this model is linear
    Gaussian, so the Kalman recursion is the exact value, not a proxy).
  Primary criterion: mean absolute error against the exact Kalman value
    across replications, with standard error.
  Hard veto A: non-finite value or program_valid False.
  Hard veto B: Sinkhorn marginal convergence -- any step with
    post-quotient column TV error above marginal_tolerance (1e-4).
  Explanatory only: per-step ESS, wall time, monotone epsilon trend.
  Not concluded: that the selected cell is superior to the others. With
    three replications the ranking is descriptive; ties inside one SE are
    statistically indistinguishable and resolved by control cost, not by
    a superiority claim.
  Artifact: docs/benchmarks/r2_tuning_pilot_dlgssm.json

Device
  Requires CUDA_DEVICE_ORDER=PCI_BUS_ID. Without it CUDA uses
  FASTEST_FIRST ordering and CUDA_VISIBLE_DEVICES=1 selects the wrong
  card. The 2026-08-27 pre-fix pilot ran on the 5080 for this reason.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    PYTHONPATH=. conda run -n tftwogpu \
    python docs/benchmarks/run_r2_tuning_grid_pilot.py
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Anchor the repository root that contains THIS script. `conda run` does
# not propagate PYTHONPATH, and a site entry for the main checkout
# (/home/chakwong/BayesFilter) precedes cwd on sys.path, so a worktree run
# would otherwise import the main repo's `bayesfilter` and fail on modules
# that exist only here. Front-insert makes the worktree win.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# The Q3 leaderboard runner holds the claim-lane callback builders; reuse
# them so the tuning grid drives the same log-density construction the
# claim runs use rather than a re-derived copy. It must be imported HERE,
# at module scope, because its own import body calls
# set_memory_growth -- which raises once TF devices are initialized. A
# lazy import inside the cell loop fails for that reason.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_q3_leaderboard_20260824 import (  # noqa: E402
    _density_obs_log,
    _gaussian_trans_log,
)

DTYPE = tf.float64

# Frozen reference observation matrix and filter theta for the dlgssm
# canonical model (must match diagonal_lgssm_canonical_model and the Q3
# claim lane's row_diagonal_lgssm).
LGSSM_OBS_MATRIX = np.array(
    [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]]
)
LGSSM_THETA = np.array([0.9, 0.8, 0.7, 0.6, 0.8])

MARGINAL_TOLERANCE = 1.0e-4


def load_tuning_data(
    model_name: str,
    data_dir: str = "docs/benchmarks/r2_tuning_20260827",
):
    """Load the Phase-2 tuning dataset for one model."""
    path = Path(data_dir) / f"{model_name}_tuning_data.npz"
    data = np.load(path)
    return {
        "observations": data["observations"],
        "replication_seeds": data["replication_seeds"],
        "model_seed": int(data["model_seed"]),
        "horizon": int(data["horizon"]),
        "replications": int(data["replications"]),
        "sha256": str(data["sha256"]),
        "path": str(path),
    }


def exact_kalman_loglik(observations: np.ndarray) -> float:
    """Exact log-likelihood for the diagonal LGSSM.

    The model is linear Gaussian, so the Kalman recursion is the exact
    marginal likelihood -- not an approximation and not a proxy metric.
    This is a NumPy independent reference solution (permitted diagnostic
    use under the repository backend rule).
    """
    phi = np.diag(LGSSM_THETA[:3])
    q = LGSSM_THETA[3] ** 2 * np.eye(3)
    r = LGSSM_THETA[4] ** 2 * np.eye(3)
    h = LGSSM_OBS_MATRIX
    mean = np.zeros(3)
    cov = np.eye(3)
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


def run_one_cell(
    observations,
    epsilon,
    sinkhorn_steps,
    flow_substeps,
    particle_count,
    seed,
):
    """Run one grid cell through the canonical value lane."""
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        diagonal_lgssm_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        canonical_value_and_diagnostics,
        CanonicalModelCallbacks,
    )

    theta0 = tf.constant(LGSSM_THETA, DTYPE)
    model, _ = diagonal_lgssm_canonical_model(theta0)

    dim = 3
    obs_tensor = tf.constant(observations, DTYPE)

    callbacks = CanonicalModelCallbacks(
        model_id="dlgssm_tuning",
        state_dim=dim,
        observation_dim=dim,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta0, p),
        transition_log_density_fn=_gaussian_trans_log(model, theta0),
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=_density_obs_log(model, theta0),
        initial_mean=tf.zeros([dim], DTYPE),
        initial_covariance=tf.eye(dim, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )

    cell = {
        "epsilon": epsilon,
        "sinkhorn_steps": sinkhorn_steps,
        "flow_substeps": flow_substeps,
        "particle_count": particle_count,
        "seed": seed,
    }

    try:
        result = canonical_value_and_diagnostics(
            callbacks=callbacks,
            observations=obs_tensor,
            particle_count=particle_count,
            seed=seed,
            flow_substeps=flow_substeps,
            temper_stages=1,
            annealed_resampling=False,
            flow_prior_cap=float("inf"),
            resample_seed=seed,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=sinkhorn_steps,
            ridge=1.0e-5,
            dual_cap_enabled=True,
            trust_region_enabled=True,
            trust_region_lm_damping=1.0e-2,
            trust_region_lm_scale_floor=1.0e-4,
            trust_region_radius=0.5,
        )

        value = float(result["value"].numpy())
        program_valid = bool(result["program_valid"].numpy())
        finite = bool(np.isfinite(value)) and program_valid

        ess = result["per_step_ess"].numpy()
        ess_finite = ess[np.isfinite(ess)]

        steps_done = int(result["marginal_steps_completed"].numpy())
        max_tv = float(result["max_marginal_tv_error"].numpy())
        all_marginals_valid = bool(result["all_marginals_valid"].numpy())
        per_step_valid = result["per_step_marginal_valid"].numpy()
        horizon = int(observations.shape[0])

        # Non-vacuity: the veto passes only if the reset actually ran at
        # every step. A run that broke out early has no marginal evidence
        # for the steps it skipped and must not be credited with passing.
        marginal_veto_pass = (
            all_marginals_valid and steps_done == horizon
        )

        cell.update(
            {
                "value": value,
                "finite": finite,
                "program_valid": program_valid,
                "ess_min": (
                    float(ess_finite.min()) if ess_finite.size else None
                ),
                "ess_mean": (
                    float(ess_finite.mean()) if ess_finite.size else None
                ),
                "ess_min_fraction": (
                    float(ess_finite.min()) / particle_count
                    if ess_finite.size
                    else None
                ),
                "ess_nonfinite_steps": int(
                    ess.size - ess_finite.size
                ),
                "max_marginal_tv_error": max_tv,
                "marginal_steps_completed": steps_done,
                "marginal_failed_steps": int(
                    (~per_step_valid).sum()
                ),
                "all_marginals_valid": all_marginals_valid,
                "marginal_veto_pass": marginal_veto_pass,
            }
        )
        return cell
    except Exception as exc:  # localized cell failure, recorded not raised
        cell.update(
            {
                "value": None,
                "finite": False,
                "program_valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "ess_min": None,
                "ess_mean": None,
                "ess_min_fraction": None,
                "ess_nonfinite_steps": None,
                "max_marginal_tv_error": None,
                "marginal_steps_completed": 0,
                "marginal_failed_steps": None,
                "all_marginals_valid": False,
                "marginal_veto_pass": False,
            }
        )
        return cell


def _config_key(cell):
    return (
        cell["epsilon"],
        cell["sinkhorn_steps"],
        cell["flow_substeps"],
        cell["particle_count"],
    )


def select_configuration(cells, replications):
    """Apply the two hard vetoes, then rank surviving configurations.

    Returns the selection record. The ranking is by mean absolute error
    against the exact Kalman value with its standard error; the reported
    selection is the cheapest configuration inside one SE of the lowest
    mean error, so no superiority claim is attached to the winner.
    """
    by_config = {}
    for cell in cells:
        by_config.setdefault(_config_key(cell), []).append(cell)

    configs = []
    for key, group in sorted(by_config.items()):
        eps, sk, flow, n = key
        finite_pass = all(c["finite"] for c in group)
        veto_pass = all(c["marginal_veto_pass"] for c in group)
        complete = len(group) == replications
        errors = [
            abs(c["abs_error"])
            for c in group
            if c.get("abs_error") is not None
        ]
        if errors:
            mean_err = float(np.mean(errors))
            se_err = (
                float(np.std(errors, ddof=1) / math.sqrt(len(errors)))
                if len(errors) > 1
                else None
            )
        else:
            mean_err = None
            se_err = None
        tvs = [
            c["max_marginal_tv_error"]
            for c in group
            if c.get("max_marginal_tv_error") is not None
        ]
        ess_fracs = [
            c["ess_min_fraction"]
            for c in group
            if c.get("ess_min_fraction") is not None
        ]
        configs.append(
            {
                "epsilon": eps,
                "sinkhorn_steps": sk,
                "flow_substeps": flow,
                "particle_count": n,
                "replications_run": len(group),
                "replications_complete": complete,
                "finite_veto_pass": finite_pass,
                "marginal_veto_pass": veto_pass,
                "survives": bool(finite_pass and veto_pass and complete),
                "mean_abs_error": mean_err,
                "se_abs_error": se_err,
                "worst_marginal_tv_error": (
                    float(max(tvs)) if tvs else None
                ),
                "worst_ess_min_fraction": (
                    float(min(ess_fracs)) if ess_fracs else None
                ),
                # Control cost, used only to break statistical ties.
                "control_cost": flow * sk * n,
            }
        )

    survivors = [
        c for c in configs if c["survives"] and c["mean_abs_error"] is not None
    ]

    rejected_by_marginal = [
        c for c in configs if not c["marginal_veto_pass"]
    ]
    rejected_by_finite = [c for c in configs if not c["finite_veto_pass"]]

    selection = {
        "rule": (
            "Reject any configuration with a non-finite value or an "
            "unconverged Sinkhorn marginal at any step (post-quotient "
            "column TV error above marginal_tolerance, or a reset that "
            "did not run for the full horizon). Rank survivors by mean "
            "absolute error against the exact Kalman log-likelihood. "
            "Report the cheapest control setting whose mean error is "
            "within one standard error of the lowest, since differences "
            "inside that band are not statistically separated at this "
            "replication count."
        ),
        "comparator": "exact Kalman log-likelihood (model is linear Gaussian)",
        "marginal_tolerance": MARGINAL_TOLERANCE,
        "configurations_total": len(configs),
        "configurations_surviving": len(survivors),
        "rejected_by_marginal_veto": len(rejected_by_marginal),
        "rejected_by_finite_veto": len(rejected_by_finite),
        # Non-vacuity guard: a veto that rejects nothing has no bite, and
        # the artifact must say so rather than implying the screen worked.
        "marginal_veto_had_bite": bool(rejected_by_marginal),
        "finite_veto_had_bite": bool(rejected_by_finite),
        "configurations": configs,
    }

    if not survivors:
        selection.update(
            {
                "selected": None,
                "note": "no configuration survived both hard vetoes",
            }
        )
        return selection

    best = min(survivors, key=lambda c: c["mean_abs_error"])
    band = best["mean_abs_error"] + (best["se_abs_error"] or 0.0)
    indistinguishable = [
        c for c in survivors if c["mean_abs_error"] <= band
    ]
    chosen = min(
        indistinguishable,
        key=lambda c: (c["control_cost"], c["mean_abs_error"]),
    )
    selection.update(
        {
            "lowest_mean_abs_error": {
                k: best[k]
                for k in (
                    "epsilon",
                    "sinkhorn_steps",
                    "flow_substeps",
                    "particle_count",
                    "mean_abs_error",
                    "se_abs_error",
                )
            },
            "statistically_indistinguishable_count": len(indistinguishable),
            "selected": {
                k: chosen[k]
                for k in (
                    "epsilon",
                    "sinkhorn_steps",
                    "flow_substeps",
                    "particle_count",
                    "mean_abs_error",
                    "se_abs_error",
                    "worst_marginal_tv_error",
                    "worst_ess_min_fraction",
                )
            },
            "selection_basis": (
                "cheapest control setting within one SE of the lowest "
                "mean absolute error"
            ),
        }
    )
    return selection


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _device_record():
    gpus = tf.config.list_physical_devices("GPU")
    details = []
    for gpu in gpus:
        try:
            info = tf.config.experimental.get_device_details(gpu)
        except Exception:
            info = {}
        details.append(
            {
                "name": gpu.name,
                "device_name": info.get("device_name", "unknown"),
                "compute_capability": str(
                    info.get("compute_capability", "unknown")
                ),
            }
        )
    return {
        "visible_gpus": details,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "unset"
        ),
        "memory_growth_verified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="/GPU:0",
        help=(
            "TensorFlow logical device. With CUDA_VISIBLE_DEVICES masking "
            "the selected card is always /GPU:0."
        ),
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=3,
        help="tuning replications per grid cell",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmarks/r2_tuning_pilot_dlgssm.json",
    )
    args = parser.parse_args()

    device_record = _device_record()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # TensorFlow GPU Memory Rule: fail closed if growth cannot be set.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in gpus:
            if not tf.config.experimental.get_memory_growth(gpu):
                raise RuntimeError(
                    f"memory growth not active on {gpu.name}"
                )
        device_record["memory_growth_verified"] = True
        print(
            "GPU memory growth verified on: "
            + ", ".join(d["device_name"] for d in device_record["visible_gpus"])
        )
    else:
        print("no GPU visible; running on CPU")

    data = load_tuning_data("dlgssm")
    replications = min(args.replications, data["replications"])
    print(f"dlgssm tuning data {data['observations'].shape}")
    print(f"  sha256 {data['sha256'][:16]}...")
    print(f"  replications used {replications}/{data['replications']}")

    kalman = [
        exact_kalman_loglik(data["observations"][rep])
        for rep in range(replications)
    ]
    for rep, ref in enumerate(kalman):
        print(f"  exact Kalman rep {rep}: {ref:.6f}")

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
    print(f"\nrunning {total} cells\n")

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
                        with tf.device(args.device):
                            cell = run_one_cell(
                                obs,
                                eps,
                                sinkhorn_steps,
                                flow_substeps,
                                n,
                                seed,
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

                        veto = (
                            "pass"
                            if cell["marginal_veto_pass"]
                            else "VETO"
                        )
                        err = (
                            f"{cell['abs_error']:.4f}"
                            if cell["abs_error"] is not None
                            else "n/a"
                        )
                        tv = (
                            f"{cell['max_marginal_tv_error']:.3e}"
                            if cell["max_marginal_tv_error"] is not None
                            else "n/a"
                        )
                        essf = (
                            f"{cell['ess_min_fraction']:.3f}"
                            if cell["ess_min_fraction"] is not None
                            else "n/a"
                        )
                        print(
                            f"[{idx}/{total}] eps={eps} sk={sinkhorn_steps} "
                            f"flow={flow_substeps} N={n} rep={rep} "
                            f"{veto} err={err} tv={tv} essfrac={essf} "
                            f"{cell['elapsed_sec']:.1f}s"
                        )

    wall = time.time() - t_start
    selection = select_configuration(cells, replications)

    output = {
        "study": "r2_tune_phase2_pilot_dlgssm",
        "date": "2026-08-27",
        "scope": {
            "model": "dlgssm",
            "lane": "value",
            "route": "contract_e_reset_dual_cap_trust_region",
            "horizon": data["horizon"],
            "dtype": "float64",
            "particle_counts": particle_counts,
            "chunk_policy": "dpf_transport_exact_divisor_cap3000_v1",
        },
        "tuning_data": {
            "path": data["path"],
            "sha256": data["sha256"],
            "model_seed": data["model_seed"],
            "replication_seeds": data["replication_seeds"].tolist(),
            "replications_used": replications,
            "disjoint_from_claim_partition": True,
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
        "selection": selection,
        "manifest": {
            "git_commit": _git_commit(),
            "command": " ".join(
                ["python", "docs/benchmarks/run_r2_tuning_grid_pilot.py"]
            ),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "device": device_record,
            "wall_time_sec": wall,
            "plan_file": "docs/plans/bayesfilter-r2-execution-plan-2026-08-27.md",
        },
        "non_claims": [
            "The selected configuration is not claimed superior to other "
            "surviving configurations; differences within one standard "
            "error are not statistically separated at this replication "
            "count.",
            "Passing the marginal-convergence veto does not establish "
            "posterior correctness, HMC readiness, or dense Sinkhorn "
            "equivalence.",
            "This pilot covers dlgssm only. It is a warm-start candidate "
            "for no other model or route.",
        ],
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nwall time {wall/60:.1f} min")
    print(f"saved {args.output}")
    print(
        f"configurations surviving both vetoes: "
        f"{selection['configurations_surviving']}"
        f"/{selection['configurations_total']}"
    )
    print(
        f"marginal veto had bite: {selection['marginal_veto_had_bite']} "
        f"(rejected {selection['rejected_by_marginal_veto']})"
    )
    if selection.get("selected"):
        sel = selection["selected"]
        print(
            f"selected eps={sel['epsilon']} sk={sel['sinkhorn_steps']} "
            f"flow={sel['flow_substeps']} N={sel['particle_count']} "
            f"mean|err|={sel['mean_abs_error']:.4f}"
        )


if __name__ == "__main__":
    main()
