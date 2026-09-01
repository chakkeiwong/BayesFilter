"""Run a CPU/XLA linear-Gaussian LEDH-PFPF density/Jacobian fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("LEDH fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("LEDH fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("LEDH fixture found a visible GPU")

from bayesfilter.testing.particle_authority_ledh_tf import (
    gaussian_log_density,
    ledh_flow,
    ledh_inverse,
)


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_ledhpfpf_fixture_2026_08_25.py"
MODULE = ROOT / "bayesfilter/testing/particle_authority_ledh_tf.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase23-ledhpfpf-source-fixture-subplan-2026-08-25.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise RuntimeError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    points = tf.constant(
        [[-1.4, -0.8], [-1.0, 0.3], [-0.6, 1.1], [-0.2, -1.3], [0.1, 0.2],
         [0.4, 1.4], [0.8, -0.4], [1.1, 0.7], [1.5, -1.0], [1.8, 0.1],
         [2.0, 1.2], [2.4, -0.2]],
        tf.float64,
    )
    prior_mean = tf.constant([0.2, -0.1], tf.float64)
    prior_covariance = tf.constant([[1.0, 0.2], [0.2, 1.5]], tf.float64)
    observation_matrix = tf.constant([[1.0, 0.3], [-0.2, 1.0]], tf.float64)
    observation_covariance = tf.constant([[0.4, 0.05], [0.05, 0.6]], tf.float64)
    observation = tf.constant([0.7, -0.4], tf.float64)
    step_sizes = tf.fill([10], tf.constant(0.1, tf.float64))
    final_points, flow = ledh_flow(
        points,
        prior_mean,
        prior_covariance,
        observation_matrix,
        observation_covariance,
        observation,
        step_sizes,
    )
    recovered = ledh_inverse(final_points, flow)
    roundtrip = tf.reduce_max(tf.abs(recovered - points))
    composed = tf.eye(2, dtype=tf.float64)
    for matrix in flow["matrices"]:
        composed = tf.matmul(matrix, composed)
    composed_logdet = tf.math.log(tf.abs(tf.linalg.det(composed)))
    determinant_residual = tf.abs(composed_logdet - flow["logdet"])
    prior_log = gaussian_log_density(points, prior_mean, prior_covariance)
    transformed_proposal_log = prior_log - flow["logdet"]
    recovered_prior_log = gaussian_log_density(recovered, prior_mean, prior_covariance)
    density_identity_residual = tf.reduce_max(
        tf.abs(transformed_proposal_log - (recovered_prior_log - flow["logdet"]))
    )
    posterior_precision = tf.linalg.inv(prior_covariance) + tf.matmul(
        observation_matrix,
        tf.matmul(tf.linalg.inv(observation_covariance), observation_matrix),
        transpose_a=True,
    )
    posterior_covariance = tf.linalg.inv(posterior_precision)
    posterior_mean = tf.matmul(
        posterior_covariance,
        (
            tf.matmul(tf.linalg.inv(prior_covariance), prior_mean[..., None])
            + tf.matmul(
                observation_matrix,
                tf.matmul(tf.linalg.inv(observation_covariance), observation[..., None]),
                transpose_a=True,
            )
        ),
    )[:, 0]
    target_log = gaussian_log_density(
        final_points, posterior_mean, posterior_covariance
    )
    log_weights = target_log - transformed_proposal_log
    hard = {
        "finite": bool(
            tf.reduce_all(tf.math.is_finite(final_points)).numpy()
            and tf.reduce_all(tf.math.is_finite(log_weights)).numpy()
        ),
        "all_step_determinants_nonzero": bool(
            tf.reduce_all(tf.abs(flow["determinants"]) > 0.0).numpy()
        ),
        "inverse_roundtrip_residual": float(roundtrip.numpy()),
        "determinant_product_residual": float(determinant_residual.numpy()),
        "density_identity_residual": float(density_identity_residual.numpy()),
    }
    status = (
        "PASS_SOURCE_FAITHFUL_LEDHPFPF_FIXTURE"
        if hard["finite"]
        and hard["all_step_determinants_nonzero"]
        and hard["inverse_roundtrip_residual"] <= 1.0e-10
        and hard["determinant_product_residual"] <= 1.0e-10
        and hard["density_identity_residual"] <= 1.0e-10
        else "LEDHPFPF_FIXTURE_FAIL_REPAIR"
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.ledhpfpf_fixture.v1",
        "status": status,
        "role": "source_faithful_invertible_ledh_pfpf_fixture_candidate",
        "source_anchor": "Li-Coates equations 6-20 and Algorithm 1",
        "hard_gates": hard,
        "flow": flow,
        "target": {
            "posterior_mean": posterior_mean,
            "posterior_covariance": posterior_covariance,
            "target_log_density": target_log,
            "proposal_log_density": transformed_proposal_log,
            "log_weights": log_weights,
        },
        "fixture": {
            "points": points,
            "prior_mean": prior_mean,
            "prior_covariance": prior_covariance,
            "observation_matrix": observation_matrix,
            "observation_covariance": observation_covariance,
            "observation": observation,
            "step_sizes": step_sizes,
            "final_points": final_points,
            "recovered_points": recovered,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_intentionally_hidden": True,
            "jit_compile": False,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"runner": _sha(RUNNER), "module": _sha(MODULE), "plan": _sha(PLAN)},
        },
        "nonclaims": [
            "finite affine-flow density identity does not establish nonlinear q20 posterior correctness",
            "the fixture does not establish mode discovery, IID samples, or HMC readiness",
            "no default or authority promotion claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 23 Source-Faithful LEDH-PFPF Fixture\n\n"
        f"Status: `{status}`\n\n"
        "This is a linear-Gaussian invertible-flow density/Jacobian fixture, "
        "not q20 posterior evidence.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status == "PASS_SOURCE_FAITHFUL_LEDHPFPF_FIXTURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
