#!/usr/bin/env python3
"""Build target-specific Laplace geometry for the P6 SIR-SGQF comparator."""

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

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "SIR-SGQF"
IDENTITY_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-02"
)
IDENTITY_RESULT_SHA256 = "5cca9efae6147dbdcbd5ad12d0371451b58b6d26cc879ad1c267c0f40d100ea2"
TYPED_SIGNATURE = "0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc"
PLAN_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p6-r2-sir-sgqf-comparator-subplan-2026-07-16.md"
)
ITERATION_FD_STEP = 1.0e-4
TERMINAL_FD_STEP = 5.0e-5
MAX_ITERATIONS = 8
TRUST_RADIUS_INF = 1.0
LINE_MULTIPLIERS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)
SCORE_NORM_MAX = 1.0e-4
STEP_NORM_MAX = 1.0e-5
VALUE_IMPROVEMENT_MAX = 1.0e-8
HESSIAN_RELATIVE_GAP_MAX = 1.0e-3
ABSOLUTE_EIGENVALUE_FLOOR = 1.0e-8
RELATIVE_EIGENVALUE_FLOOR = 1.0e-6


class AffineMassTargetAdapter:
    """Exact three-dimensional affine coordinate change around SIR-SGQF."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        center: Any,
        factor: Any,
        target_signature: str,
        mass_artifact_sha256: str,
    ) -> None:
        import tensorflow as tf

        self.base_adapter = base_adapter
        self.center = tf.convert_to_tensor(center, tf.float64)
        self.factor = tf.convert_to_tensor(factor, tf.float64)
        if self.center.shape != (3,) or self.factor.shape != (3, 3):
            raise ValueError("SIR affine center/factor shape mismatch")
        sign, log_abs_det = tf.linalg.slogdet(self.factor)
        if float(sign.numpy()) <= 0.0 or not bool(tf.math.is_finite(log_abs_det).numpy()):
            raise ValueError("SIR affine factor must have a positive determinant")
        self.log_abs_det = log_abs_det
        self.parameter_dim = 3
        self.target_signature = str(target_signature)
        self.mass_artifact_sha256 = str(mass_artifact_sha256)

    def forward(self, z: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(z, tf.float64)
        return self.center + tf.tensordot(values, self.factor, axes=[[-1], [1]])

    def inverse_batch(self, theta: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(theta, tf.float64) - self.center
        return tf.transpose(
            tf.linalg.triangular_solve(
                self.factor, tf.transpose(values), lower=True
            )
        )

    def log_prob_and_grad(self, z: Any) -> tuple[Any, Any]:
        import tensorflow as tf

        values = tf.convert_to_tensor(z, tf.float64)
        theta = self.forward(values)
        raw_value, raw_score = self.base_adapter.log_prob_and_grad(theta)
        latent_score = tf.tensordot(raw_score, self.factor, axes=[[-1], [0]])
        return raw_value + self.log_abs_det, latent_score

    def target_status_telemetry(self, z: Any) -> Mapping[str, Any]:
        return self.base_adapter.target_status_telemetry(self.forward(z))

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": "bayesfilter.multimodel_neutra_p6_sir_affine_mass_target.v1",
                    "target_signature": self.target_signature,
                    "mass_artifact_sha256": self.mass_artifact_sha256,
                    "coordinate_program": "theta=center+z@factor.T",
                    "score_program": "score_z=score_theta@factor",
                    "constant_log_abs_det_included": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_rows(tf: Any, value: Any, score: Any, status: Mapping[str, Any]) -> Any:
    return tf.logical_and(
        tf.math.is_finite(value),
        tf.logical_and(
            tf.reduce_all(tf.math.is_finite(score), axis=1),
            tf.logical_and(
                tf.equal(status["status_code"], 0),
                status["valid_pre_regularized_score"],
            ),
        ),
    )


def _score_hessian(tf: Any, evaluate: Any, point: Any, *, step: float):
    basis = tf.eye(3, dtype=tf.float64)
    epsilon = tf.constant(step, tf.float64)
    stencil = tf.concat(
        (point[None, :] + epsilon * basis, point[None, :] - epsilon * basis),
        axis=0,
    )
    value, score, status = evaluate(stencil)
    valid = _valid_rows(tf, value, score, status)
    raw = tf.transpose((score[:3] - score[3:]) / (2.0 * epsilon))
    symmetric = 0.5 * (raw + tf.transpose(raw))
    finite = tf.reduce_all(tf.math.is_finite(symmetric))
    evidence = {
        "step": step,
        "stencil_valid": valid,
        "stencil_value": value,
        "raw_hessian": raw,
        "symmetric_hessian": symmetric,
        "symmetry_max_gap": tf.reduce_max(tf.abs(raw - tf.transpose(raw))),
        "finite": finite,
    }
    return symmetric if bool(tf.reduce_all(valid).numpy() and finite.numpy()) else None, evidence


def _regularized_geometry(tf: Any, precision_raw: Any):
    raw = 0.5 * (
        tf.convert_to_tensor(precision_raw, tf.float64)
        + tf.transpose(tf.convert_to_tensor(precision_raw, tf.float64))
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(raw)
    largest_abs = tf.reduce_max(tf.abs(eigenvalues))
    floor = tf.maximum(
        tf.constant(ABSOLUTE_EIGENVALUE_FLOOR, tf.float64),
        tf.constant(RELATIVE_EIGENVALUE_FLOOR, tf.float64) * largest_abs,
    )
    regularized = tf.maximum(eigenvalues, floor)
    precision = tf.einsum("ik,k,jk->ij", eigenvectors, regularized, eigenvectors)
    precision = 0.5 * (precision + tf.transpose(precision))
    precision_factor = tf.linalg.cholesky(precision)
    covariance = tf.linalg.cholesky_solve(
        precision_factor, tf.eye(3, dtype=tf.float64)
    )
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    factor = tf.linalg.cholesky(covariance)
    return precision, covariance, factor, {
        "raw_eigenvalues": eigenvalues,
        "largest_absolute_eigenvalue": largest_abs,
        "realized_eigenvalue_floor": floor,
        "regularized_eigenvalues": regularized,
        "clipped_eigenvalue_count": tf.math.count_nonzero(eigenvalues < floor),
        "regularized_condition_number": regularized[-1] / regularized[0],
    }


def main() -> None:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"geometry output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    from bayesfilter.inference.neutra_campaign import (
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIRSGQFLikelihoodRecomposer,
        generate_frozen_sir_dataset_tf,
        make_sir_sgqf_neutra_adapter,
        sir_identity_chart_jacobian_value_score,
        sir_prior_value_score,
    )

    if _hash(IDENTITY_ROOT / "result.json") != IDENTITY_RESULT_SHA256:
        raise RuntimeError("SIR-SGQF identity result hash mismatch")
    expected_identity = json.loads((IDENTITY_ROOT / "target_identity.json").read_text())
    registry = json.loads((IDENTITY_ROOT / "repaired_registry.json").read_text())
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    recomposer = SIRSGQFLikelihoodRecomposer(adapter)
    points = tf.concat(
        [
            tf.zeros([1, 3], tf.float64),
            0.5 * tf.eye(3, dtype=tf.float64),
            -0.5 * tf.eye(3, dtype=tf.float64),
            tf.eye(3, dtype=tf.float64),
            -tf.eye(3, dtype=tf.float64),
        ],
        axis=0,
    )
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=points,
        prior_value_score_fn=sir_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=sir_identity_chart_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    identity = issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="model_cell",
        scope_id=CELL_ID,
        adapter=adapter,
        recomposition=recomposition,
        registry_row=registry,
        registry_artifact_sha256=_hash(IDENTITY_ROOT / "repaired_registry.json"),
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if _jsonable(identity.payload()) != expected_identity or identity.target_signature != TYPED_SIGNATURE:
        raise RuntimeError("SIR-SGQF identity drift before geometry")

    @tf.function(input_signature=[tf.TensorSpec([1, 3], tf.float64)], jit_compile=True)
    def evaluate_one(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    @tf.function(input_signature=[tf.TensorSpec([6, 3], tf.float64)], jit_compile=True)
    def evaluate_stencil(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    @tf.function(input_signature=[tf.TensorSpec([6, 3], tf.float64)], jit_compile=True)
    def evaluate_line(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    current = tf.zeros([3], tf.float64)
    iterations = []
    converged = False
    terminal_reason = "maximum_iterations_reached"
    for iteration in range(MAX_ITERATIONS):
        value, score, status = evaluate_one(current[None, :])
        if not bool(_valid_rows(tf, value, score, status)[0].numpy()):
            terminal_reason = "current_point_invalid"
            break
        score_norm = float(tf.reduce_max(tf.abs(score[0])).numpy())
        if score_norm <= SCORE_NORM_MAX:
            converged = True
            terminal_reason = "score_norm_passed"
            iterations.append({"iteration": iteration, "point": current, "value": value[0], "score": score[0], "score_norm_inf": score_norm, "stopped_before_step": True})
            break
        hessian, hessian_evidence = _score_hessian(
            tf, evaluate_stencil, current, step=ITERATION_FD_STEP
        )
        if hessian is None:
            terminal_reason = "iteration_hessian_invalid"
            break
        _precision, covariance, _factor, regularization = _regularized_geometry(tf, -hessian)
        direction = tf.linalg.matvec(covariance, score[0])
        raw_norm = tf.reduce_max(tf.abs(direction))
        scale = tf.minimum(
            tf.constant(1.0, tf.float64),
            tf.constant(TRUST_RADIUS_INF, tf.float64) / tf.maximum(raw_norm, 1.0e-30),
        )
        direction = scale * direction
        multipliers = tf.constant(LINE_MULTIPLIERS, tf.float64)
        candidates = current[None, :] + multipliers[:, None] * direction[None, :]
        line_value, line_score, line_status = evaluate_line(candidates)
        line_valid = _valid_rows(tf, line_value, line_score, line_status)
        eligible = tf.where(line_valid, line_value, tf.fill([6], tf.constant(float("-inf"), tf.float64)))
        selected_index = int(tf.argmax(eligible).numpy())
        improvement = line_value[selected_index] - value[0]
        realized_step = multipliers[selected_index] * direction
        iterations.append(
            {
                "iteration": iteration,
                "point": current,
                "value": value[0],
                "score": score[0],
                "score_norm_inf": score_norm,
                "hessian": hessian_evidence,
                "regularization": regularization,
                "direction": direction,
                "line_values": line_value,
                "line_valid": line_valid,
                "selected_index": selected_index,
                "selected_multiplier": multipliers[selected_index],
                "value_improvement": improvement,
                "realized_step": realized_step,
            }
        )
        if not bool(line_valid[selected_index].numpy()):
            terminal_reason = "no_valid_line_candidate"
            break
        current = candidates[selected_index]
        if bool(
            tf.reduce_max(tf.abs(realized_step)) <= STEP_NORM_MAX
            and improvement <= VALUE_IMPROVEMENT_MAX
            and improvement >= -1.0e-12
        ):
            converged = True
            terminal_reason = "step_and_value_stagnation_passed"
            break

    final_value, final_score, final_status = evaluate_one(current[None, :])
    final_valid = bool(_valid_rows(tf, final_value, final_score, final_status)[0].numpy())
    final_score_norm = float(tf.reduce_max(tf.abs(final_score[0])).numpy())
    if final_valid and final_score_norm <= SCORE_NORM_MAX:
        converged = True
        terminal_reason = "terminal_score_norm_passed"
    hessian_a, evidence_a = _score_hessian(tf, evaluate_stencil, current, step=ITERATION_FD_STEP)
    hessian_b, evidence_b = _score_hessian(tf, evaluate_stencil, current, step=TERMINAL_FD_STEP)
    relative_gap = None
    geometry = None
    affine_checks = None
    if hessian_a is not None and hessian_b is not None:
        relative_gap = tf.linalg.norm(hessian_a - hessian_b) / tf.maximum(tf.linalg.norm(hessian_b), 1.0e-30)
        precision, covariance, factor, regularization = _regularized_geometry(tf, -hessian_b)
        geometry = {
            "center": current,
            "negative_hessian_raw": -hessian_b,
            "precision_regularized": precision,
            "covariance": covariance,
            "cholesky_factor": factor,
            "regularization": regularization,
            "coordinate_program": "theta=center+z@factor.T",
            "inverse_program": "solve(factor,(theta-center).T).T",
            "score_program": "score_z=score_theta@factor",
            "constant_log_abs_det_included": True,
        }
        geometry_hash = hashlib.sha256(
            json.dumps(_jsonable(geometry), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        affine = AffineMassTargetAdapter(
            base_adapter=adapter,
            center=current,
            factor=factor,
            target_signature=identity.target_signature,
            mass_artifact_sha256=geometry_hash,
        )
        z = tf.stack([tf.zeros([3], tf.float64), tf.constant([0.1, -0.05, 0.08], tf.float64)], axis=0)
        theta = affine.forward(z)
        round_trip = affine.inverse_batch(theta)
        raw_value, raw_score = adapter.log_prob_and_grad(theta)
        affine_value, affine_score = affine.log_prob_and_grad(z)
        affine_checks = {
            "round_trip_max_gap": tf.reduce_max(tf.abs(round_trip - z)),
            "value_chain_rule_max_gap": tf.reduce_max(tf.abs(affine_value - (raw_value + affine.log_abs_det))),
            "score_chain_rule_max_gap": tf.reduce_max(tf.abs(affine_score - tf.tensordot(raw_score, factor, axes=[[-1], [0]]))),
        }
        affine_checks["passed"] = bool(
            affine_checks["round_trip_max_gap"] <= 1.0e-10
            and affine_checks["value_chain_rule_max_gap"] <= 1.0e-10
            and affine_checks["score_chain_rule_max_gap"] <= 1.0e-10
        )
    passed = bool(
        converged
        and final_valid
        and relative_gap is not None
        and relative_gap <= HESSIAN_RELATIVE_GAP_MAX
        and geometry is not None
        and affine_checks is not None
        and affine_checks["passed"]
    )
    result = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_geometry.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "passed": passed,
        "decision": "ADMIT_SIR_SGQF_LAPLACE_GEOMETRY" if passed else "COMPARATOR_BLOCKED_GEOMETRY",
        "target_identity": identity.payload(),
        "iterations": iterations,
        "converged": converged,
        "terminal_reason": terminal_reason,
        "final_point": current,
        "final_value": final_value[0],
        "final_score": final_score[0],
        "final_score_norm_inf": final_score_norm,
        "final_valid": final_valid,
        "terminal_hessian_step_a": evidence_a,
        "terminal_hessian_step_b": evidence_b,
        "terminal_hessian_relative_gap": relative_gap,
        "geometry": geometry,
        "affine_checks": affine_checks,
        "memory_policy": memory_policy,
        "output_devices": [final_value.device, final_score.device],
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "elapsed_seconds": time.perf_counter() - started,
        "nonclaims": ["local tuning geometry only", "no HMC, NeuTra, exactness, calibration, or readiness claim"],
    }
    _write(args.output_root / "result.json", result)
    _write(
        args.output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_geometry_manifest.v1",
            "program_id": PROGRAM_ID,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "memory_policy": memory_policy,
            "jit_compile": True,
            "tf32_enabled": True,
            "target_signature": identity.target_signature,
            "plan_file": PLAN_FILE,
            "wall_time_seconds": result["elapsed_seconds"],
        },
    )
    hashes = {
        str(path.relative_to(args.output_root)): _hash(path)
        for path in sorted(args.output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write(args.output_root / "artifact_hashes.json", {"schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_geometry_hashes.v1", "artifacts": hashes})


if __name__ == "__main__":
    main()
