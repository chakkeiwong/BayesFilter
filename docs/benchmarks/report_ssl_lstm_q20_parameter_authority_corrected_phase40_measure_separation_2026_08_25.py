"""Audit the exact v2.2 root-group-stratified train/validation/audit split."""

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
    raise RuntimeError("Phase 40 measure audit is CPU diagnostic-only; hide CUDA before import")

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
EXPECTED_VERSION = "v2.2-root-group-stratified"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v3_root_group_stratified_split"

import tensorflow as tf


class Phase40MeasureError(RuntimeError):
    """Raised when the active v2.2 measure receipt is malformed."""


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


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase40MeasureError(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _summary(
    theta: tf.Tensor,
    target_log: tf.Tensor,
    proposal_log: tf.Tensor,
    weights: tf.Tensor,
    roots: tf.Tensor,
    indices: list[int],
    *,
    affine_mean: tf.Tensor | None = None,
    affine_chol: tf.Tensor | None = None,
) -> Mapping[str, Any]:
    if len(indices) < 2:
        raise Phase40MeasureError("partition has fewer than two rows")
    index = tf.constant(indices, tf.int32)
    rows = tf.gather(theta, index)
    target = tf.gather(target_log, index)
    proposal = tf.gather(proposal_log, index)
    raw_weights = tf.gather(weights, index)
    normalized = raw_weights / tf.reduce_sum(raw_weights)
    mean = tf.reduce_sum(normalized[:, tf.newaxis] * rows, axis=0)
    centered = rows - mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", normalized, centered, centered)
    latent = rows
    if affine_mean is not None and affine_chol is not None:
        latent = tf.transpose(
            tf.linalg.triangular_solve(
                affine_chol,
                tf.transpose(rows - affine_mean[tf.newaxis, :]),
                lower=True,
            )
        )
    latent_mean = tf.reduce_sum(normalized[:, tf.newaxis] * latent, axis=0)
    latent_centered = latent - latent_mean[tf.newaxis, :]
    latent_covariance = tf.einsum(
        "n,ni,nj->ij", normalized, latent_centered, latent_centered
    )
    ratio = target - proposal
    sign = rows[:, 2] < 0.0
    root_values = tf.gather(roots, index)
    return {
        "count": len(indices),
        "root_count": tf.size(tf.unique(root_values).y),
        "negative_count": tf.reduce_sum(tf.cast(sign, tf.int32)),
        "positive_count": tf.reduce_sum(tf.cast(tf.logical_not(sign), tf.int32)),
        "effective_sample_size_fraction": tf.math.reciprocal(
            tf.cast(len(indices), tf.float64) * tf.reduce_sum(tf.square(normalized))
        ),
        "maximum_normalized_weight": tf.reduce_max(normalized),
        "weighted_negative_mode_fraction": tf.reduce_sum(
            tf.where(sign, normalized, tf.zeros_like(normalized))
        ),
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
        "weighted_log_ratio_mean": tf.reduce_sum(normalized * ratio),
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
    parser.add_argument("--identity-root", required=True, type=Path)
    parser.add_argument("--affine-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.identity_root, args.affine_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase40MeasureError("paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase40MeasureError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    trace_paths = {
        "identity": ROOT / args.identity_root / "result.json",
        "affine": ROOT / args.affine_root / "result.json",
    }
    traces: dict[str, Mapping[str, Any]] = {}
    for name, path in trace_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS_NEUTRA_BOUNDARY_ROLE_LIMITED":
            raise Phase40MeasureError(f"non-passing trace: {path}")
        if payload.get("plan_version") != EXPECTED_VERSION or payload.get("schema") != EXPECTED_SCHEMA:
            raise Phase40MeasureError(f"trace is not active v2.2 schema: {path}")
        split = payload.get("split", {})
        if split.get("policy") != "root_group_stratified_v1":
            raise Phase40MeasureError(f"wrong split policy: {path}")
        if not all(split.get(key) is True for key in ("root_disjoint", "row_partition_complete", "row_partition_disjoint")):
            raise Phase40MeasureError(f"split invariant failed: {path}")
        traces[name] = payload
    identity_split = traces["identity"]["split"]
    if traces["affine"]["split"]["indices"] != identity_split["indices"]:
        raise Phase40MeasureError("identity and affine splits differ")
    authority_root = ROOT / args.identity_root
    # Both traces must point to the same authority pilot; this keeps the
    # partition comparison tied to one declared particle measure.
    authority = traces["identity"]["authority"]
    if traces["affine"]["authority"] != authority:
        raise Phase40MeasureError("identity and affine authority metadata differ")
    pilot_path = ROOT / authority["root"] / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    m0 = pilot.get("arms", {}).get("M0")
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT" or m0 is None:
        raise Phase40MeasureError("authority pilot is not a passing M0 receipt")
    if m0.get("target_signature") != authority["target_signature"]:
        raise Phase40MeasureError("target signature mismatch")
    if m0.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase40MeasureError("authority measure is not theta_R4")
    if _sha(pilot_path) != authority["pilot_sha256"]:
        raise Phase40MeasureError("pilot hash does not match boundary receipt")
    theta = _load_tensor(m0["receipts"]["final_theta"])
    weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    target_log = _load_tensor(m0["receipts"]["final_target_log_theta"])
    proposal_log = _load_tensor(m0["receipts"]["final_proposal_log_theta"])
    roots = _load_tensor(m0["receipts"]["final_roots"])
    n = int(theta.shape[0])
    if theta.shape.rank != 2 or theta.shape[1] != 4:
        raise Phase40MeasureError(f"wrong theta shape: {theta.shape}")
    if any(tuple(value.shape) != (n,) for value in (weights, target_log, proposal_log, roots)):
        raise Phase40MeasureError("authority vector shape mismatch")
    weights = tf.maximum(tf.cast(weights, tf.float64), tf.constant(1.0e-300, tf.float64))
    weights = weights / tf.reduce_sum(weights)
    partitions = identity_split["indices"]
    names = ("train", "validation", "audit")
    if set(partitions) != set(names):
        raise Phase40MeasureError("partition names are incomplete")
    row_union = set(partitions["train"]) | set(partitions["validation"]) | set(partitions["audit"])
    if len(row_union) != n or sum(len(partitions[name]) for name in names) != n:
        raise Phase40MeasureError("partition does not cover each row exactly once")
    root_sets = {
        name: set(int(value) for value in tf.gather(roots, tf.constant(partitions[name], tf.int32)).numpy().tolist())
        for name in names
    }
    if any(root_sets[left] & root_sets[right] for left, right in (("train", "validation"), ("train", "audit"), ("validation", "audit"))):
        raise Phase40MeasureError("root overlap found on source tensors")
    train_idx = tf.constant(partitions["train"], tf.int32)
    train_weights = tf.gather(weights, train_idx)
    train_weights = train_weights / tf.reduce_sum(train_weights)
    train_theta = tf.gather(theta, train_idx)
    affine_mean = tf.reduce_sum(train_weights[:, tf.newaxis] * train_theta, axis=0)
    centered = train_theta - affine_mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", train_weights, centered, centered)
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    eigenvalues = tf.linalg.eigvalsh(covariance)
    if not bool(tf.reduce_all(eigenvalues > 0.0).numpy()):
        raise Phase40MeasureError("train covariance is not positive definite")
    chol = tf.linalg.cholesky(covariance)
    physical = {
        name: _summary(theta, target_log, proposal_log, weights, roots, partitions[name])
        for name in names
    }
    latent = {
        name: _summary(
            theta,
            target_log,
            proposal_log,
            weights,
            roots,
            partitions[name],
            affine_mean=affine_mean,
            affine_chol=chol,
        )
        for name in names
    }
    if not all(bool(physical[name]["finite"].numpy()) for name in names):
        raise Phase40MeasureError("non-finite partition summary")
    oracle_mean = tf.reduce_max(tf.abs(latent["train"]["latent_mean"]))
    oracle_cov = tf.reduce_max(tf.abs(latent["train"]["latent_covariance"] - tf.eye(4, dtype=tf.float64)))
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_measure_separation.v2_root_group_stratified",
        "status": "PASS_V2_2_THETA_MEASURE_SEPARATION_DIAGNOSTIC",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_root_group_partition_support_diagnostic",
        "measure": "theta_R4",
        "target_signature": authority["target_signature"],
        "authority": authority,
        "split": {
            **identity_split,
            "source_root_sets": {name: sorted(root_sets[name]) for name in names},
        },
        "affine_training_oracle": {
            "max_abs_mean": oracle_mean,
            "max_abs_covariance_residual": oracle_cov,
            "eigenvalues": eigenvalues,
            "condition_estimate": tf.reduce_max(eigenvalues) / tf.reduce_min(eigenvalues),
        },
        "partitions": {
            name: {"physical": physical[name], "train_measure_affine": latent[name]}
            for name in names
        },
        "nonclaims": [
            "Root-disjoint partition summaries are explanatory and do not establish target coverage, mode discovery, posterior correctness, or IID whitening.",
            "The validation and audit partitions are finite samples; their moment differences are descriptive only.",
            "No objective retuning, SMC-U authority, HMC, canonical LEDH, or default promotion.",
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
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "pilot": _sha(pilot_path),
                "identity_trace": _sha(trace_paths["identity"]),
                "affine_trace": _sha(trace_paths["affine"]),
            },
        },
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    lines = [
        "# v2.2 Root-Group-Stratified Theta Measure Separation",
        "",
        f"Status: `{result['status']}`",
        "",
        "| Partition | rows | roots | ESS fraction | max weight | negative-mode fraction | theta mean[0] | affine latent mean max | affine covariance offdiag max | log-ratio min | log-ratio max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        p = physical[name]
        l = latent[name]
        latent_mean_max = tf.reduce_max(tf.abs(l["latent_mean"]))
        latent_cov_offdiag = tf.reduce_max(
            tf.abs(l["latent_covariance"] - tf.linalg.diag(tf.linalg.diag_part(l["latent_covariance"])))
        )
        lines.append(
            f"| {name} | {p['count']} | {p['root_count']} | {float(p['effective_sample_size_fraction']):.6f} | "
            f"{float(p['maximum_normalized_weight']):.6f} | {float(p['weighted_negative_mode_fraction']):.6f} | "
            f"{float(p['theta_mean'][0]):.6f} | {float(latent_mean_max):.6f} | {float(latent_cov_offdiag):.6f} | "
            f"{float(p['log_ratio_min']):.6f} | {float(p['log_ratio_max']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Train affine oracle: mean max `{float(oracle_mean):.3e}`, covariance residual max `{float(oracle_cov):.3e}`; roots are disjoint across partitions.",
            "",
            "This is an explanatory diagnostic. It does not establish IID Gaussian whitening, posterior correctness, or exhaustive mode discovery.",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
