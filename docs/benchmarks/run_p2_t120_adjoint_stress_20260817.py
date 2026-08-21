"""P2 T=120 adjoint-state stress: resource measurement vs value-only engine.

Plan: docs/plans/bayesfilter-p2-t120-adjoint-stress-plan-2026-08-17.md
Engineering resource measurement only (peak RSS, wall); no scientific claim.
Run each mode in its OWN process so ru_maxrss isolates the arm.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

DTYPE = tf.float64
PLAN = "docs/plans/bayesfilter-p2-t120-adjoint-stress-plan-2026-08-17.md"
HORIZON = 120
N_STATE = 2


def _build():
    # I-P2-4-verified n=2 regime (test_p2_adjoint_engine_fd._config) with
    # the shift-family LGSSM adapter; data horizon extended to T=120.
    from tests.highdim.test_p2_adjoint_engine_fd import _config

    from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter

    n = N_STATE
    rng = np.random.default_rng(62)
    a_matrix = 0.7 * np.eye(n)
    q_matrix = 0.4 * np.eye(n)
    r_matrix = 0.5 * np.eye(n)
    m0 = np.zeros(n)
    p0 = np.eye(n)
    q_inv = np.linalg.inv(q_matrix)

    def mvn(x, mean, cov):
        d = int(cov.shape[0])
        chol = np.linalg.cholesky(cov)
        solve = tf.linalg.triangular_solve(
            tf.constant(chol, DTYPE), tf.transpose(x - mean), lower=True
        )
        quad = tf.reduce_sum(tf.square(solve), axis=0)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)

    adapter = DensityKernelAdapter(
        state_dim=n,
        transition_log_density=lambda xc, xp: mvn(
            xc, tf.linalg.matvec(tf.constant(a_matrix, DTYPE), xp), q_matrix
        ),
        observation_log_density=lambda xc, y: mvn(xc, tf.convert_to_tensor(y, DTYPE), r_matrix),
        initial_log_density=lambda xc: mvn(xc, tf.constant(m0, DTYPE), p0),
    )

    def transition_vjp(xc, xp, cot):
        residual = xc - tf.linalg.matvec(tf.constant(a_matrix, DTYPE), xp)
        rows = tf.einsum("nd,de->ne", residual, tf.constant(q_inv, DTYPE))
        return tf.einsum("n,ne->e", cot, rows)

    def observation_vjp(xc, y, cot):
        return tf.zeros([n], DTYPE)

    def initial_vjp(xc, cot):
        return tf.zeros([n], DTYPE)

    x = rng.multivariate_normal(m0, p0)
    ys = []
    for t in range(HORIZON):
        if t > 0:
            x = a_matrix @ x + rng.multivariate_normal(np.zeros(n), q_matrix)
        ys.append(x + rng.multivariate_normal(np.zeros(n), r_matrix))
    return adapter, (transition_vjp, observation_vjp, initial_vjp), tf.constant(
        np.stack(ys), DTYPE
    ), _config(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("value", "adjoint"), required=True)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter, (tvjp, ovjp, ivjp), ys, config = _build()
    ys = ys[: args.horizon]
    started = time.time()
    if args.mode == "value":
        from bayesfilter.highdim.squared_tt_engine_v0_tf import run_value_filter_branch_axis

        value, _diags = run_value_filter_branch_axis(adapter, ys, config)
        grad_norm = None
    else:
        from bayesfilter.highdim.squared_tt_adjoint_engine_tf import run_adjoint_score_filter

        value, grad = run_adjoint_score_filter(
            adapter, ys, config,
            transition_vjp=tvjp, observation_vjp=ovjp, initial_vjp=ivjp,
            parameter_dim=N_STATE,
        )
        grad_norm = float(tf.norm(grad).numpy())
    wall = time.time() - started
    if not np.isfinite(float(value.numpy())):
        raise SystemExit("veto: non-finite value")
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    result = {
        "schema_version": "p2_t120_adjoint_stress.v1",
        "plan": PLAN,
        "mode": args.mode,
        "horizon": args.horizon,
        "n": N_STATE,
        "value": float(value.numpy()),
        "grad_norm": grad_norm,
        "wall_seconds": wall,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": peak_rss_bytes / 2**30,
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "load_average": os.getloadavg(),
        "nonclaims": [
            "resource measurement only; no accuracy/HMC/scaling claim",
            "single run per mode: wall and RSS are descriptive",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mode", "wall_seconds", "peak_rss_gib", "value")}))


if __name__ == "__main__":
    main()
