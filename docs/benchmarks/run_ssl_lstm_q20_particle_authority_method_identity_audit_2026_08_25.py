"""Audit q20 modular-arm method identity against checked local sources.

This is a CPU-hidden diagnostic/reporting lane. It does not implement or
promote ETPF, GenUT, LEDH-PFPF, ET-PF, NeuTra, or HMC.
"""

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
    raise RuntimeError("method identity audit requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("method identity audit requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("method identity audit found a visible GPU")


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_method_identity_audit_2026_08_25.py"
MODULAR_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_modular_arms_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase17-source-faithful-modular-contracts-subplan-2026-08-25.md"
ACEVEDO = ROOT / ".localresources/papers/ledh_replay_solution_20260824/acevedo-dewiljes-reich-2017-second-order-etpf.txt"
GENUT = ROOT / ".localresources/papers/ebeigbe-et-al-genut-2104.01958.txt"
LI_COATES = ROOT / ".localresources/papers/ledh_replay_solution_20260824/li-coates-2017-particle-filtering-invertible-flow.txt"


class IdentityAuditError(RuntimeError):
    """Raised when the method identity audit cannot preserve its evidence."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise IdentityAuditError(f"refusing to overwrite artifact: {path}")
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _function_source(source: str, start: str, end: str) -> str:
    start_index = source.find(start)
    end_index = source.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        raise IdentityAuditError(f"cannot isolate bounded source block: {start}")
    return source[start_index:end_index]


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = ROOT / str(receipt["path"])
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != receipt["sha256"]:
        raise IdentityAuditError(f"tensor hash mismatch: {path}")
    dtype = getattr(tf, str(receipt["dtype"]))
    value = tf.io.parse_tensor(encoded, out_type=dtype)
    value = tf.ensure_shape(value, receipt["shape"])
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise IdentityAuditError(f"non-finite tensor: {path}")
    return value


def _weighted_marginal_moments(
    points: tf.Tensor, weights: tf.Tensor
) -> Mapping[str, tf.Tensor]:
    weights = tf.cast(weights, tf.float64)
    weights = weights / tf.reduce_sum(weights)
    points = tf.cast(points, tf.float64)
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    third = tf.reduce_sum(weights[:, None] * tf.pow(centered, 3), axis=0)
    fourth = tf.reduce_sum(weights[:, None] * tf.pow(centered, 4), axis=0)
    return {
        "mean": mean,
        "covariance": 0.5 * (covariance + tf.transpose(covariance)),
        "marginal_third_central": third,
        "marginal_fourth_central": fourth,
    }


def _current_m2_rule(
    source_moments: Mapping[str, tf.Tensor], dimension: int
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    mean = source_moments["mean"]
    covariance = source_moments["covariance"]
    ridge = tf.constant(1.0e-8, tf.float64)
    chol = tf.linalg.cholesky(covariance + ridge * tf.eye(dimension, dtype=tf.float64))
    central_weight = tf.constant(0.10, tf.float64)
    side_weight = (1.0 - central_weight) / tf.cast(2 * dimension, tf.float64)
    scale = tf.sqrt(tf.cast(dimension, tf.float64) / (1.0 - central_weight))
    columns = tf.transpose(chol) * scale
    points = tf.concat(
        (mean[None, :], mean[None, :] + columns, mean[None, :] - columns), axis=0
    )
    weights = tf.concat(
        (central_weight[None], tf.fill((2 * dimension,), side_weight)), axis=0
    )
    return points, weights, _weighted_marginal_moments(points, weights)


def _max_abs(value: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(value)).numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.authority_root.is_absolute() or args.output_root.is_absolute():
        raise IdentityAuditError("paths must be repository-relative")
    if ".." in args.authority_root.parts or ".." in args.output_root.parts:
        raise IdentityAuditError("paths may not contain parent traversal")
    output_root = ROOT / args.output_root
    if output_root.exists():
        raise IdentityAuditError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()

    for source_path in (ACEVEDO, GENUT, LI_COATES, MODULAR_RUNNER, PLAN):
        if not source_path.is_file():
            raise IdentityAuditError(f"required source is missing: {source_path}")

    pilot_path = ROOT / args.authority_root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_GATE":
        raise IdentityAuditError("authority pilot did not pass")
    m0 = pilot["arms"]["M0"]
    if m0["protocol"].get("mode_axis") != 2:
        raise IdentityAuditError("authority mode_axis is not the bound q20 axis 2")
    if m0["protocol"].get("target_signature") != m0.get("target_signature"):
        raise IdentityAuditError("target signature mismatch")
    theta = _load_tensor(m0["receipts"]["final_theta"])
    weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    if theta.shape != (300, 4) or weights.shape != (300,):
        raise IdentityAuditError("authority bank shape mismatch")

    modular_source = MODULAR_RUNNER.read_text(encoding="utf-8")
    m1_source = _function_source(
        modular_source, "def _m1_second_order_transform", "def _m2_genut_sigma_points"
    )
    m2_source = _function_source(
        modular_source, "def _m2_genut_sigma_points", "def _evaluate_status"
    )
    m3_source = _function_source(
        modular_source, "def _m3_affine_density_scaffold", "def _arm_markdown"
    )

    source_moments = _weighted_marginal_moments(theta, weights)
    _m2_points, _m2_weights, m2_moments = _current_m2_rule(source_moments, 4)
    m2_third_residual = _max_abs(
        m2_moments["marginal_third_central"]
        - source_moments["marginal_third_central"]
    )
    m2_fourth_residual = _max_abs(
        m2_moments["marginal_fourth_central"]
        - source_moments["marginal_fourth_central"]
    )
    if m2_third_residual <= 1.0e-12:
        raise IdentityAuditError(
            "measured bank has no discriminating marginal-third-moment signal"
        )

    arms = {
        "M1": {
            "claimed_target": "Acevedo second-order corrected ETPF",
            "computed_quantity": "affine Cholesky finite-cloud mean/covariance match",
            "relation": "different",
            "classification": "wrong_relative_to_named_etpf_source_identity",
            "scaffold_status": "validity_not_rejected_for_explicit_affine_moment_scaffold",
            "required_operations_present": {
                "letf_transport_matrix_D": "transport_matrix" in m1_source,
                "optimal_transport_or_sinkhorn": "sinkhorn" in m1_source.lower(),
                "riccati_correction": "riccati" in m1_source.lower(),
            },
            "source_anchor": "Acevedo et al. equations 16,20,26,42-44,48-57",
        },
        "M2": {
            "claimed_target": "Ebeigbe et al. generalized unscented transform",
            "computed_quantity": "symmetric 2d+1 mean/covariance unscented rule",
            "relation": "different_for_measured_nonzero_skewness",
            "classification": "wrong_relative_to_named_genut_source_identity",
            "scaffold_status": "validity_not_rejected_for_explicit_symmetric_ut_scaffold",
            "required_operations_present": {
                "marginal_skewness_input": "skew" in m2_source.lower(),
                "marginal_kurtosis_input": "kurt" in m2_source.lower(),
                "asymmetric_sigma_offsets": "minus" not in m2_source.lower(),
            },
            "dynamic_moment_receipt": {
                "source_marginal_third_central": source_moments[
                    "marginal_third_central"
                ],
                "sigma_marginal_third_central": m2_moments[
                    "marginal_third_central"
                ],
                "maximum_third_moment_residual": m2_third_residual,
                "maximum_fourth_moment_residual": m2_fourth_residual,
            },
            "source_anchor": "Ebeigbe et al. selected diagonal skewness/kurtosis contract",
        },
        "M3": {
            "claimed_target": "Li-Coates invertible LEDH-PFPF proposal",
            "computed_quantity": "single fixed affine map and tautological log-density comparison",
            "relation": "different",
            "classification": "wrong_relative_to_named_ledh_pfpf_source_identity",
            "scaffold_status": "explicit_affine_density_scaffold_only",
            "required_operations_present": {
                "pseudo_time_step_product": "pseudo" in m3_source.lower(),
                "pre_and_post_flow_target_terms": "target" in m3_source.lower(),
                "covariance_lifecycle": "covariance" in m3_source.lower(),
            },
            "source_anchor": "Li-Coates equations 16,19,20 and Algorithm 1",
        },
        "M4": {
            "claimed_target": "full second-order ET-PF comparator",
            "computed_quantity": "alias of M1 affine finite-moment scaffold",
            "relation": "different",
            "classification": "wrong_relative_to_named_full_etpf_identity",
            "scaffold_status": "descriptive_affine_comparator_only",
            "required_operations_present": {
                "independent_filter_route": False,
                "reference_filter_comparison": False,
            },
            "source_anchor": "project M4 contract in reviewed modular plan",
        },
    }
    if any(
        any(values.values())
        for values in (
            arms["M1"]["required_operations_present"],
            arms["M2"]["required_operations_present"],
            arms["M3"]["required_operations_present"],
        )
    ):
        raise IdentityAuditError("bounded source classification assumptions changed")

    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.method_identity_audit.v1",
        "status": "PASS_METHOD_IDENTITY_AUDIT_REPAIR_REQUIRED",
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha256(pilot_path),
            "protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "mode_axis": 2,
            "particle_count": 300,
        },
        "source_hashes": {
            "acevedo": _sha256(ACEVEDO),
            "genut": _sha256(GENUT),
            "li_coates": _sha256(LI_COATES),
            "modular_runner": _sha256(MODULAR_RUNNER),
            "audit_runner": _sha256(RUNNER),
            "plan": _sha256(PLAN),
        },
        "arms": arms,
        "engineering_ledger": "audit runner and immutable input/hash gates passed",
        "numerical_ledger": "measured M2 third-moment mismatch is finite and nonzero",
        "scientific_ledger": "no current q20 modular arm implements its named source method",
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ("git", "status", "--short"), cwd=ROOT, text=True
                ).strip()
            ),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_intentionally_hidden": True,
            "jit_compile": False,
            "role": "cpu_reference_method_identity_audit",
            "wall_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "absence in the bounded q20 runner is not absence elsewhere in the repository",
            "the audit does not reject ETPF, GenUT, LEDH-PFPF, or ET-PF as scientific ideas",
            "the audit does not establish posterior correctness, IID sampling, HMC readiness, or default status",
        ],
    }
    _write_json(output_root / "result.json", result)
    (output_root / "result.md").write_text(
        "# Phase 17 Method Identity Audit\n\n"
        "Status: `PASS_METHOD_IDENTITY_AUDIT_REPAIR_REQUIRED`\n\n"
        "The bounded q20 M1--M4 runner computes four explicit scaffolds, not the "
        "named source methods. Preserve their diagnostic results, repair their "
        "labels, and implement one source-faithful arm at a time.\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {"status": result["status"], "output_root": args.output_root.as_posix()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
