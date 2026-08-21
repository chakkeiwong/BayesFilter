"""P1B LGSSM value ladder: rank-sufficiency curve for the branch-axis engine.

Plan: docs/plans/bayesfilter-p1b-lgssm-value-ladder-plan-2026-08-17.md
Scattered frozen rows only (V2). Exact Kalman is the same-target oracle.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DensityKernelAdapter,
    EngineConfig,
    run_value_filter_branch_axis,
)

DTYPE = tf.float64
PLAN = "docs/plans/bayesfilter-p1b-lgssm-value-ladder-plan-2026-08-17.md"

PER_STEP_TOLERANCE = 2.5e-3  # declared primary criterion (plan)
HORIZON = 8
ROWS_BY_N = {2: 4096, 4: 8192, 8: 16384}
RANKS = (2, 4, 6, 8)
SEEDS = (42, 142, 242)
STOP_WALL_SECONDS = 45 * 60


def _mvn_log_density(x: tf.Tensor, mean: tf.Tensor, covariance: np.ndarray) -> tf.Tensor:
    d = int(covariance.shape[0])
    chol = np.linalg.cholesky(covariance)
    solve = tf.linalg.triangular_solve(
        tf.constant(chol, DTYPE), tf.transpose(x - mean), lower=True
    )
    quad = tf.reduce_sum(tf.square(solve), axis=0)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


def _case(n: int, seed: int, adversarial: bool):
    rng = np.random.default_rng(seed)
    scale = 0.25 if adversarial else 0.1
    A = 0.7 * np.eye(n) + scale * rng.standard_normal((n, n)) / max(1, n - 1)
    Q = 0.4 * np.eye(n)
    H = np.eye(n)
    R = 0.5 * np.eye(n)
    m0 = np.zeros(n)
    P0 = np.eye(n)
    x = rng.multivariate_normal(m0, P0)
    ys = []
    for t in range(HORIZON):
        if t > 0:
            x = A @ x + rng.multivariate_normal(np.zeros(n), Q)
        ys.append(H @ x + rng.multivariate_normal(np.zeros(n), R))
    ys = np.stack(ys)
    mean, cov = m0.copy(), P0.copy()
    exact = 0.0
    for i, y in enumerate(ys):
        if i > 0:
            mean = A @ mean
            cov = A @ cov @ A.T + Q
        S = H @ cov @ H.T + R
        innov = y - H @ mean
        exact += float(
            -0.5
            * (
                n * np.log(2.0 * np.pi)
                + np.linalg.slogdet(S)[1]
                + innov @ np.linalg.solve(S, innov)
            )
        )
        K = cov @ H.T @ np.linalg.inv(S)
        mean = mean + K @ innov
        cov = cov - K @ S @ K.T
    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=lambda xc, xp: _mvn_log_density(
            xc, tf.linalg.matvec(tf.constant(A, DTYPE), xp), Q
        ),
        observation_log_density=lambda xc, y=None: None,  # replaced below
        initial_log_density=lambda xc: _mvn_log_density(xc, tf.constant(m0, DTYPE), P0),
    )
    # rebuild with observation closure capturing R, H properly
    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=adapter.transition_log_density,
        observation_log_density=lambda xc, y: _mvn_log_density(
            tf.linalg.matvec(tf.constant(H, DTYPE), xc), tf.convert_to_tensor(y, DTYPE), R
        ),
        initial_log_density=adapter.initial_log_density,
    )
    return adapter, tf.constant(ys, DTYPE), exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dims", default="2,4,8")
    parser.add_argument("--output", required=True)
    # Amendment A1 (2026-08-18): attempt03 overrides; defaults reproduce
    # the original attempt01/02 design.
    parser.add_argument("--row-design", default="mc", choices=("mc", "sobol"))
    parser.add_argument("--rows", type=int, default=None, help="override ROWS_BY_N for all n")
    parser.add_argument("--ranks", default=None, help="override rank list, e.g. 4,6")
    args = parser.parse_args()

    started = time.time()
    dims = [int(part) for part in args.dims.split(",") if part.strip()]
    ranks = (
        tuple(int(part) for part in args.ranks.split(",") if part.strip())
        if args.ranks
        else RANKS
    )
    rows_list: list[dict[str, Any]] = []
    stop_reason = None
    for n in dims:
        for r in ranks:
            for arm, seed_list in (
                ("standard", SEEDS),
                ("adversarial", (SEEDS[0],) if n >= 4 else ()),
            ):
                for seed in seed_list:
                    adapter, ys, exact = _case(n, seed + n, arm == "adversarial")
                    config = EngineConfig(
                        basis_degree=12,
                        rank=r,
                        row_count=args.rows if args.rows else ROWS_BY_N[n],
                        sweeps=3,
                        ridge=1e-10,
                        tau=1e-6,
                        coordinate_half_width=3.0,
                        seed=91000 + 10 * n + r,
                        row_design=args.row_design,
                    )
                    cell_start = time.time()
                    try:
                        value, diags = run_value_filter_branch_axis(adapter, ys, config)
                        gap = abs(float(value.numpy()) - exact)
                        status = "ok"
                    except Exception as error:  # hard veto surface
                        gap, diags, status = float("nan"), [], f"veto:{error}"
                    wall = time.time() - cell_start
                    rows_list.append(
                        {
                            "n": n,
                            "rank": r,
                            "arm": arm,
                            "seed": seed,
                            "gap": gap,
                            "per_step_gap": gap / HORIZON if np.isfinite(gap) else None,
                            "passes_declared": bool(
                                np.isfinite(gap) and gap / HORIZON <= PER_STEP_TOLERANCE
                            ),
                            "wall_seconds": wall,
                            "status": status,
                            "max_fit_rms": (
                                max(d["weighted_fit_rms"] for d in diags) if diags else None
                            ),
                            "max_gram_condition": (
                                max(d.get("gram_condition", 0.0) for d in diags)
                                if diags
                                else None
                            ),
                        }
                    )
                    print(json.dumps(rows_list[-1]))
                    if wall > STOP_WALL_SECONDS:
                        stop_reason = f"resource bound at n={n} r={r} ({wall:.0f}s)"
                        break
                if stop_reason:
                    break
            if stop_reason:
                break
        if stop_reason:
            break

    # r*(n): smallest rank whose standard-arm seeds all pass
    r_star: dict[str, Any] = {}
    for n in dims:
        passing = [
            r
            for r in ranks
            if all(
                row["passes_declared"]
                for row in rows_list
                if row["n"] == n and row["rank"] == r and row["arm"] == "standard"
            )
            and any(row["n"] == n and row["rank"] == r for row in rows_list)
        ]
        r_star[str(n)] = min(passing) if passing else None

    result = {
        "schema_version": "p1b_lgssm_value_ladder.v2",
        "plan": PLAN,
        "row_design": args.row_design,
        "rows_override": args.rows,
        "ranks_run": list(ranks),
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "per_step_tolerance": PER_STEP_TOLERANCE,
        "horizon": HORIZON,
        "rows": rows_list,
        "r_star": r_star,
        "stop_reason": stop_reason,
        "wall_time_seconds": time.time() - started,
        "nonclaims": [
            "LGSSM-family fixtures only; no cross-model or T=120 claim",
            "3 seeds: descriptive mean/range only, no ranking language",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"r_star": r_star, "stop_reason": stop_reason}))


if __name__ == "__main__":
    main()
