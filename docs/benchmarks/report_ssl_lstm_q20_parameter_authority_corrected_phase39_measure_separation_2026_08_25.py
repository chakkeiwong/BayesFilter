"""Report train/validation/audit empirical-measure separation for Phase 39."""

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
    raise RuntimeError("Phase 39 is a CPU diagnostic; set CUDA_VISIBLE_DEVICES=-1 before import")

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"

import tensorflow as tf


class Phase39Error(RuntimeError):
    """Raised when a Phase 39 source receipt is invalid."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
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


def _load_tensor(root: Path, receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase39Error(f"tensor hash mismatch: {path}")
    dtype = getattr(tf, str(receipt["dtype"]))
    value = tf.io.parse_tensor(encoded, out_type=dtype)
    return tf.ensure_shape(value, receipt["shape"])


def _weighted_summary(
    theta: tf.Tensor,
    target_log: tf.Tensor,
    proposal_log: tf.Tensor,
    weights: tf.Tensor,
    indices: list[int],
    *,
    affine_mean: tf.Tensor | None = None,
    affine_chol: tf.Tensor | None = None,
) -> Mapping[str, Any]:
    if len(indices) <= 1:
        raise Phase39Error("each partition needs at least two rows")
    idx = tf.constant(indices, tf.int32)
    rows = tf.gather(theta, idx)
    target = tf.gather(target_log, idx)
    proposal = tf.gather(proposal_log, idx)
    raw_weights = tf.gather(weights, idx)
    normalized = raw_weights / tf.reduce_sum(raw_weights)
    mean = tf.reduce_sum(normalized[:, tf.newaxis] * rows, axis=0)
    centered = rows - mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", normalized, centered, centered)
    sign = rows[:, 2] < 0.0
    latent = rows
    latent_mean = mean
    latent_covariance = covariance
    if affine_mean is not None and affine_chol is not None:
        latent = tf.transpose(
            tf.linalg.triangular_solve(
                affine_chol, tf.transpose(rows - affine_mean[tf.newaxis, :]), lower=True
            )
        )
        latent_mean = tf.reduce_sum(normalized[:, tf.newaxis] * latent, axis=0)
        latent_centered = latent - latent_mean[tf.newaxis, :]
        latent_covariance = tf.einsum(
            "n,ni,nj->ij", normalized, latent_centered, latent_centered
        )
    ratio = target - proposal
    return {
        "count": len(indices),
        "raw_weight_sum": tf.reduce_sum(raw_weights),
        "effective_sample_size_fraction": tf.math.reciprocal(
            tf.cast(len(indices), tf.float64) * tf.reduce_sum(tf.square(normalized))
        ),
        "maximum_normalized_weight": tf.reduce_max(normalized),
        "weighted_negative_mode_fraction": tf.reduce_sum(tf.where(sign, normalized, tf.zeros_like(normalized))),
        "negative_count": tf.reduce_sum(tf.cast(sign, tf.int32)),
        "positive_count": tf.reduce_sum(tf.cast(tf.logical_not(sign), tf.int32)),
        "theta_mean": mean,
        "theta_covariance": covariance,
        "latent_mean": latent_mean,
        "latent_covariance": latent_covariance,
        "target_log_theta_min": tf.reduce_min(target),
        "target_log_theta_max": tf.reduce_max(target),
        "proposal_log_theta_min": tf.reduce_min(proposal),
        "proposal_log_theta_max": tf.reduce_max(proposal),
        "log_ratio_min": tf.reduce_min(ratio),
        "log_ratio_max": tf.reduce_max(ratio),
        "log_ratio_mean": tf.reduce_sum(normalized * ratio),
        "finite": tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(rows, [-1]),
                        tf.reshape(target, [-1]),
                        tf.reshape(proposal, [-1]),
                        tf.reshape(normalized, [-1]),
                    ],
                    axis=0,
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.authority_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase39Error("paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase39Error(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    authority = ROOT / args.authority_root
    pilot_path = authority / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase39Error("authority pilot is not passing")
    m0 = pilot["arms"].get("M0")
    if m0 is None or m0.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase39Error("M0 is not bound to theta_R4")
    target_signature = str(m0["target_signature"])
    if target_signature != "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278":
        raise Phase39Error("unexpected target signature")
    theta = _load_tensor(ROOT, m0["receipts"]["final_theta"])
    weights = _load_tensor(ROOT, m0["receipts"]["final_normalized_weights"])
    target_log = _load_tensor(ROOT, m0["receipts"]["final_target_log_theta"])
    proposal_log = _load_tensor(ROOT, m0["receipts"]["final_proposal_log_theta"])
    if theta.shape.rank != 2 or theta.shape[1] != 4:
        raise Phase39Error(f"wrong theta shape: {theta.shape}")
    n = int(theta.shape[0])
    if any(tuple(value.shape) != (n,) for value in (weights, target_log, proposal_log)):
        raise Phase39Error("receipt vector shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(theta)).numpy()):
        raise Phase39Error("theta contains non-finite values")
    weights = tf.maximum(tf.cast(weights, tf.float64), tf.constant(1.0e-300, tf.float64))
    weights = weights / tf.reduce_sum(weights)
    negative = [int(index) for index in tf.reshape(tf.where(theta[:, 2] < 0.0), [-1]).numpy().tolist()]
    positive = [int(index) for index in tf.reshape(tf.where(theta[:, 2] >= 0.0), [-1]).numpy().tolist()]
    audit = negative[:6] + positive[:6]
    remaining = [index for index in range(n) if index not in set(audit)]
    validation = remaining[: min(12, len(remaining) // 3)]
    train = remaining[len(validation) :]
    if len(train) != 232 or len(validation) != 12 or len(audit) != 12:
        raise Phase39Error(f"unexpected reconstructed split: {len(train)}/{len(validation)}/{len(audit)}")
    train_idx = tf.constant(train, tf.int32)
    train_weights = tf.gather(weights, train_idx)
    train_weights = train_weights / tf.reduce_sum(train_weights)
    train_theta = tf.gather(theta, train_idx)
    affine_mean = tf.reduce_sum(train_weights[:, tf.newaxis] * train_theta, axis=0)
    centered = train_theta - affine_mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", train_weights, centered, centered)
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    eigenvalues = tf.linalg.eigvalsh(covariance)
    if not bool(tf.reduce_all(eigenvalues > 0.0).numpy()):
        raise Phase39Error("train covariance is not positive definite")
    chol = tf.linalg.cholesky(covariance)
    partitions = {"train": train, "validation": validation, "audit": audit}
    summaries = {
        name: _weighted_summary(
            theta,
            target_log,
            proposal_log,
            weights,
            indices,
        )
        for name, indices in partitions.items()
    }
    latent_summaries = {
        name: _weighted_summary(
            theta,
            target_log,
            proposal_log,
            weights,
            indices,
            affine_mean=affine_mean,
            affine_chol=chol,
        )
        for name, indices in partitions.items()
    }
    for name, summary in summaries.items():
        if not bool(summary["finite"].numpy()):
            raise Phase39Error(f"non-finite summary: {name}")
    train_latent = latent_summaries["train"]
    oracle_mean = tf.reduce_max(tf.abs(train_latent["latent_mean"]))
    oracle_cov = tf.reduce_max(tf.abs(train_latent["latent_covariance"] - tf.eye(4, dtype=tf.float64)))
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_measure_separation.v1",
        "status": "PASS_THETA_MEASURE_SEPARATION_DIAGNOSTIC",
        "plan_version": "v2.1-training-measure-bound",
        "role": "read_only_partition_support_diagnostic",
        "measure": "theta_R4",
        "target_signature": target_signature,
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "m0_protocol_hash": m0["configuration"]["protocol_hash"],
            "geometry_role": m0["protocol"]["geometry_role"],
        },
        "split": {
            "train": len(train),
            "validation": len(validation),
            "audit": len(audit),
            "negative_axis2_total": len(negative),
            "positive_axis2_total": len(positive),
            "selection_frozen_before_audit": True,
            "construction": "Phase38 deterministic negative[:6]+positive[:6]; validation=remaining[:12]; train=remaining[12:]",
        },
        "affine_training_oracle": {
            "max_abs_mean": oracle_mean,
            "max_abs_covariance_residual": oracle_cov,
            "eigenvalues": eigenvalues,
            "condition_estimate": tf.reduce_max(eigenvalues) / tf.reduce_min(eigenvalues),
        },
        "partitions": {
            name: {"physical": summaries[name], "train_measure_affine": latent_summaries[name]}
            for name in partitions
        },
        "nonclaims": [
            "Partition summaries do not establish target coverage, mode discovery, posterior correctness, or IID whitening.",
            "Small validation and audit partitions make covariance and tail summaries descriptive only.",
            "No objective retuning, checkpoint selection, SMC-U authority, HMC, canonical LEDH, or default promotion.",
        ],
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "pilot": _sha(pilot_path)},
        },
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    lines = [
        "# Corrected Theta Measure Separation",
        "",
        f"Status: `{result['status']}`",
        "",
        "| Partition | N | ESS fraction | max normalized weight | negative-mode fraction | log-ratio min | log-ratio max | affine latent mean max | affine latent covariance offdiag max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in partitions:
        physical = summaries[name]
        latent = latent_summaries[name]
        latent_mean_max = tf.reduce_max(tf.abs(latent["latent_mean"]))
        latent_cov_offdiag_max = tf.reduce_max(
            tf.abs(
                latent["latent_covariance"]
                - tf.linalg.diag(tf.linalg.diag_part(latent["latent_covariance"]))
            )
        )
        lines.append(
            f"| {name} | {physical['count']} | {float(physical['effective_sample_size_fraction']):.6f} | "
            f"{float(physical['maximum_normalized_weight']):.6f} | {float(physical['weighted_negative_mode_fraction']):.6f} | "
            f"{float(physical['log_ratio_min']):.6f} | {float(physical['log_ratio_max']):.6f} | "
            f"{float(latent_mean_max.numpy()):.6f} | {float(latent_cov_offdiag_max.numpy()):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Train affine oracle: mean max `{float(oracle_mean.numpy()):.3e}`, covariance residual max `{float(oracle_cov.numpy()):.3e}`.",
            "",
            "This is an explanatory diagnostic. It does not establish IID Gaussian whitening, posterior correctness, or mode discovery.",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
