#!/usr/bin/env python3
"""Diagnose frozen three-mode NeuTra score parity and component support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "three-mode/component-aware-budget1000-r1/trainer_states.json"
)
PLAN = ROOT / "docs/plans/bayesfilter-neutra-gap-closure-plan-2026-08-17.md"


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(v) for v in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cloud-size", type=int, default=64)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    if not CHECKPOINT.is_file() or not PLAN.is_file():
        raise FileNotFoundError("gap diagnostic inputs are missing")
    if int(args.cloud_size) < 8:
        raise ValueError("cloud-size must be at least eight")
    root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
        AnalyticGaussianMixtureValueScoreAdapter,
        analytic_three_mode_target,
        load_weighted_neutra_transport,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical}")
    loaded = load_weighted_neutra_transport(CHECKPOINT, required_dimension=4)
    target = analytic_three_mode_target()
    base = AnalyticGaussianMixtureValueScoreAdapter(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope="weighted_neutra_three_mode_gap_diagnostic_v1",
        runtime_backend="tensorflow_exact_three_mode_gap_diagnostic",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    means = target["means"]
    factors = tf.linalg.cholesky(target["covariances"])
    noise = tf.random.stateless_normal(
        [3, int(args.cloud_size), 4], seed=(20260817, 71001), dtype=tf.float64
    )
    physical_clouds = means[:, tf.newaxis, :] + tf.einsum("mij,mnj->mni", factors, noise)
    physical = tf.reshape(physical_clouds, [-1, 4])
    latent, forward_logdet = loaded.transport.inverse_and_forward_logdet(physical)
    reconstructed, inverse_logdet = loaded.transport.forward_and_logdet(latent)
    values, score = base.log_prob_and_grad(physical)
    with tf.GradientTape() as tape:
        tape.watch(physical)
        value_tape, _ = base.log_prob_and_grad(physical)
        score_reference = tf.reduce_sum(value_tape)
    autodiff_score = tape.gradient(score_reference, physical)
    transformed_value, transformed_score = adapter.log_prob_and_grad_batch(latent)
    with tf.GradientTape() as tape:
        tape.watch(latent)
        theta = loaded.transport.forward_batch(latent)
        physical_value, _ = base.log_prob_and_grad(theta)
        total = tf.reduce_sum(physical_value + loaded.transport.log_abs_det_jacobian_batch(latent))
    transformed_autodiff = tape.gradient(total, latent)
    interpolation = tf.concat(
        [means[0] * (1.0 - alpha) + means[1] * alpha for alpha in tf.constant((0.25, 0.5, 0.75), tf.float64)],
        axis=0,
    )
    interpolation = tf.reshape(interpolation, [-1, 4])
    interp_latent, interp_logdet = loaded.transport.inverse_and_forward_logdet(interpolation)
    component_latent = tf.reshape(latent, [3, int(args.cloud_size), 4])
    latent_means = tf.reduce_mean(component_latent, axis=1)
    latent_cov = tfp_covariance(component_latent)
    payload = {
        "schema": "bayesfilter.neutra.three_mode_gap_diagnostic.v1",
        "plan": PLAN.as_posix(),
        "checkpoint": CHECKPOINT.as_posix(),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "target_signature": target["target_signature"],
        "transport_manifest": loaded.transport.manifest_payload(),
        "cloud_size_per_component": int(args.cloud_size),
        "component_latent_mean": latent_means,
        "component_latent_covariance": latent_cov,
        "component_latent_logdet_mean": tf.reduce_mean(tf.reshape(forward_logdet, [3, int(args.cloud_size)]), axis=1),
        "reconstruction_max_abs": tf.reduce_max(tf.abs(reconstructed - physical)),
        "logdet_inverse_forward_max_abs": tf.reduce_max(tf.abs(inverse_logdet - forward_logdet)),
        "score_max_abs_error_vs_autodiff": tf.reduce_max(tf.abs(score - autodiff_score)),
        "transformed_score_max_abs_error_vs_autodiff": tf.reduce_max(tf.abs(transformed_score - transformed_autodiff)),
        "interpolation_latent": interp_latent,
        "interpolation_logdet": interp_logdet,
        "all_finite": bool(tf.reduce_all(tf.math.is_finite(tf.concat([physical, latent, reconstructed, values[:, tf.newaxis], score, transformed_value[:, tf.newaxis], transformed_score], axis=1))).numpy()),
        "memory_policy": memory,
        "device": str(logical[0]),
        "dtype": "float64",
        "jit_compile": False,
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "wall_seconds": time.perf_counter() - started,
        "nonclaims": ["local support diagnostic only", "no HMC convergence claim", "no posterior correctness claim"],
    }
    _write(root / "result.json", payload)
    _write(root / "run_manifest.json", {"schema": "bayesfilter.neutra.three_mode_gap_manifest.v1", "command": sys.argv, "plan": PLAN.as_posix(), "checkpoint_sha256": _sha256(CHECKPOINT), "memory_policy": memory, "device": str(logical[0]), "jit_compile": False, "dtype": "float64"})
    _write(root / "artifact_hashes.json", {"schema": "bayesfilter.neutra.three_mode_gap_hashes.v1", "artifacts": {p.name: _sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": root.as_posix(), "all_finite": payload["all_finite"], "wall_seconds": payload["wall_seconds"]}))
    return 0


def tfp_covariance(values: Any) -> Any:
    import tensorflow as tf

    centered = values - tf.reduce_mean(values, axis=1, keepdims=True)
    count = tf.cast(tf.shape(values)[1] - 1, values.dtype)
    return tf.einsum("mni,mnj->mij", centered, centered) / count


if __name__ == "__main__":
    raise SystemExit(main())
