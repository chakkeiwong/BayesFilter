"""Run a reduced innovation-coordinate density/Jacobian fixture for q20."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("reduced LEDH fixture requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("reduced LEDH fixture requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("reduced LEDH fixture found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)


TARGET = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-particle-authority-phase26-reduced-coordinate-"
    "ledh-subplan-2026-08-25.md"
)
RUNNER = Path(__file__).resolve()


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


def _finite(value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _standard_log_density(points: tf.Tensor) -> tf.Tensor:
    dimension = tf.cast(tf.shape(points)[-1], tf.float64)
    return -0.5 * (
        dimension * tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
        + tf.reduce_sum(tf.square(points), axis=-1)
    )


def build_fixture() -> dict[str, Any]:
    started = time.perf_counter()
    target = batch_native_complexity_posterior_target(
        20, jit_compile=False, principal_sqrt_backend="tensorflow_eigh"
    )
    free = tf.constant(
        [[-0.35, -0.20, 0.15, 0.25], [0.40, 0.10, -0.25, -0.10]],
        dtype=tf.float64,
    )
    model, _derivatives = target._batched_components(free)
    covariance = model.innovation_covariance
    chol = tf.linalg.cholesky(covariance)
    batch_size = int(free.shape[0])
    point_count = 4
    dimension = int(model.innovation_dim)
    axis = tf.linspace(tf.constant(-1.0, tf.float64), tf.constant(1.0, tf.float64), dimension)
    base = tf.broadcast_to(axis[tf.newaxis, tf.newaxis, :], [batch_size, point_count, dimension])
    mapped = tf.einsum("bpk,bjk->bpj", base, chol)
    recovered = tf.linalg.triangular_solve(
        chol[:, tf.newaxis, :, :], mapped[..., tf.newaxis]
    )[..., 0]
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    base_log = _standard_log_density(base)
    transformed_log = base_log - logdet[:, tf.newaxis]
    recovered_log = _standard_log_density(recovered) - logdet[:, tf.newaxis]
    inverse_residual = tf.reduce_max(tf.abs(recovered - base))
    density_residual = tf.reduce_max(tf.abs(transformed_log - recovered_log))
    eigvals = tf.linalg.eigvalsh(covariance)
    value, score, status = target.neutra_batch_log_prob_and_grad_status(free)
    hard = {
        "covariance_finite": _finite(covariance),
        "cholesky_finite": _finite(chol),
        "mapped_finite": _finite(mapped),
        "inverse_finite": _finite(recovered),
        "target_finite": _finite(value) and _finite(score),
        "target_status_present": bool(status),
        "inverse_roundtrip_residual": float(inverse_residual.numpy()),
        "density_identity_residual": float(density_residual.numpy()),
        "minimum_covariance_eigenvalue": float(tf.reduce_min(eigvals).numpy()),
    }
    mechanics_pass = (
        all(hard[key] for key in (
            "covariance_finite",
            "cholesky_finite",
            "mapped_finite",
            "inverse_finite",
            "target_finite",
            "target_status_present",
        ))
        and hard["minimum_covariance_eigenvalue"] > 0.0
        and hard["inverse_roundtrip_residual"] <= 1.0e-10
        and hard["density_identity_residual"] <= 1.0e-10
    )
    status_code = (
        "REDUCED_COORDINATE_DENSITY_FIXTURE_PASS_TARGET_UNBOUND"
        if mechanics_pass
        else "REDUCED_COORDINATE_DENSITY_FIXTURE_FAIL_REPAIR"
    )
    return {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.ledh_reduced_fixture.v1",
        "status": status_code,
        "role": "reduced_innovation_density_jacobian_mechanics_only",
        "target": {
            "q": 20,
            "target_scope": target.target_scope,
            "target_signature": target.target_signature(),
            "adapter_signature": target.adapter_signature(),
            "parameter_dim": target.parameter_dim,
            "innovation_dim": dimension,
            "binding_to_parameter_target": False,
            "binding_reason": "fixture maps innovation coordinates using Q; no target-to-innovation proposal callback exists",
        },
        "hard_checks": hard,
        "fixture": {
            "batch": batch_size,
            "points": point_count,
            "base_coordinates": base,
            "mapped_coordinates": mapped,
            "recovered_coordinates": recovered,
            "covariance_eigenvalues": eigvals,
            "logdet_cholesky": logdet,
            "base_log_density": base_log,
            "mapped_log_density": transformed_log,
        },
        "decision": {
            "mechanics": "pass" if mechanics_pass else "fail",
            "target_binding": "vetoed",
            "direct_q20_ledh": "closed_wrong_relative_to_declared_four_parameter_target",
            "wider_campaign": "ETPF_SMC_GenUT_NeuTra_unaffected",
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
            "random_seed": "deterministic_fixed_tensor_no_rng",
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "runner": _sha(RUNNER),
                "target": _sha(TARGET),
                "plan": _sha(PLAN),
            },
        },
        "nonclaims": [
            "reduced innovation mechanics do not establish a q20 parameter proposal",
            "no source-faithful LEDH admission or posterior equivalence",
            "no whitening, mode-discovery, HMC, or default-readiness claim",
        ],
    }


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
    result = build_fixture()
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 26 Reduced Innovation LEDH Fixture\n\n"
        f"Status: `{result['status']}`\n\n"
        "The reduced density/Jacobian mechanics are tested separately from "
        "binding to the four-parameter q20 target.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "REDUCED_COORDINATE_DENSITY_FIXTURE_PASS_TARGET_UNBOUND" else 2


if __name__ == "__main__":
    raise SystemExit(main())
