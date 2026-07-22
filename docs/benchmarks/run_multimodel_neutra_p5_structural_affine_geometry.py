#!/usr/bin/env python3
"""Build the P5 STR-UKF posterior-mode affine geometry diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base
from docs.benchmarks import run_multimodel_neutra_p5_structural_plain_hmc as structural


CELL_ID = "STR-UKF"
SOURCE_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p5/STR-UKF/plain-hmc/attempt-02"
)
SOURCE_RESULT_SHA256 = (
    "5c01b1f9b36e1c23b4b09b22c13b98dded8f2200d180f98eb1c2b4706bacec46"
)
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p5-r2a-structural-affine-hmc-repair-subplan-2026-07-16.md"
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
RAW_PRECISION_MIN_EIGENVALUE = 1.0e-8
NONCLAIMS = (
    "STR-UKF target-bound posterior-mode geometry diagnostic only",
    "failed source warm-up mean is a mode-start hypothesis only",
    "local curvature is tuning evidence, not posterior evidence",
    "no HMC convergence, NeuTra quality, UKF exactness, calibration, robustness, or readiness claim",
)


class StructuralAffineTargetAdapter:
    """Exact five-dimensional affine chart around the typed structural target."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        center: Any,
        factor: Any,
        target_signature: str,
        geometry_sha256: str,
    ) -> None:
        import tensorflow as tf

        self.base_adapter = base_adapter
        self.center = tf.convert_to_tensor(center, tf.float64)
        self.factor = tf.convert_to_tensor(factor, tf.float64)
        if self.center.shape != (5,) or self.factor.shape != (5, 5):
            raise base.P4PlainHMCError("structural affine center/factor shape mismatch")
        sign, log_abs_det = tf.linalg.slogdet(self.factor)
        if float(sign.numpy()) <= 0.0 or not bool(tf.math.is_finite(log_abs_det).numpy()):
            raise base.P4PlainHMCError("structural affine factor must have positive determinant")
        self.log_abs_det = log_abs_det
        self.parameter_dim = 5
        self.target_signature = str(target_signature)
        self.geometry_sha256 = str(geometry_sha256)

    def forward(self, z: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(z, tf.float64)
        return self.center + tf.tensordot(values, self.factor, axes=[[-1], [1]])

    def inverse_batch(self, theta: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(theta, tf.float64) - self.center
        return tf.transpose(
            tf.linalg.triangular_solve(self.factor, tf.transpose(values), lower=True)
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
                    "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_target.v1",
                    "target_signature": self.target_signature,
                    "geometry_sha256": self.geometry_sha256,
                    "coordinate_program": "theta=center+z@factor.T",
                    "score_program": "score_z=score_theta@factor",
                    "constant_log_abs_det_included": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def run_geometry(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"structural geometry output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    adapter, identity, identity_reference = _rebuild_identity(tf)
    start, start_reference = _load_source_start(tf)
    basis = tf.eye(5, dtype=tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([1, 5], tf.float64)], jit_compile=True)
    def evaluate_one(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    @tf.function(input_signature=[tf.TensorSpec([10, 5], tf.float64)], jit_compile=True)
    def evaluate_stencil(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    @tf.function(input_signature=[tf.TensorSpec([6, 5], tf.float64)], jit_compile=True)
    def evaluate_line(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    current = start
    iteration_rows = []
    converged = False
    terminal_reason = "maximum_iterations_reached"
    for iteration in range(MAX_ITERATIONS):
        value, score, status = evaluate_one(current[None, :])
        point_valid = _valid_rows(tf, value, score, status)[0]
        if not bool(point_valid.numpy()):
            terminal_reason = "current_point_invalid"
            break
        score_norm = float(tf.reduce_max(tf.abs(score[0])).numpy())
        if score_norm <= SCORE_NORM_MAX:
            converged = True
            terminal_reason = "score_norm_passed"
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "point": base._json_ready(current),
                    "value": float(value[0].numpy()),
                    "score": base._json_ready(score[0]),
                    "score_norm_inf": score_norm,
                    "stopped_before_step": True,
                    "stopping_reason": terminal_reason,
                }
            )
            break
        hessian, hessian_evidence = _score_hessian(
            tf, evaluate_stencil, current, basis, step=ITERATION_FD_STEP
        )
        if hessian is None:
            terminal_reason = "iteration_hessian_invalid"
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "point": base._json_ready(current),
                    "value": float(value[0].numpy()),
                    "score": base._json_ready(score[0]),
                    "score_norm_inf": score_norm,
                    "hessian": hessian_evidence,
                    "stopping_reason": terminal_reason,
                }
            )
            break
        _precision, covariance, _factor, regularization = _regularized_geometry(
            tf, -hessian
        )
        raw_direction = tf.linalg.matvec(covariance, score[0])
        raw_norm = tf.reduce_max(tf.abs(raw_direction))
        scale = tf.minimum(
            tf.constant(1.0, tf.float64),
            tf.constant(TRUST_RADIUS_INF, tf.float64)
            / tf.maximum(raw_norm, tf.constant(1.0e-30, tf.float64)),
        )
        direction = scale * raw_direction
        multipliers = tf.constant(LINE_MULTIPLIERS, tf.float64)
        candidates = current[None, :] + multipliers[:, None] * direction[None, :]
        line_value, _line_score, line_status = evaluate_line(candidates)
        line_valid = _valid_rows(tf, line_value, _line_score, line_status)
        eligible_value = tf.where(
            line_valid,
            line_value,
            tf.fill(tf.shape(line_value), tf.constant(float("-inf"), tf.float64)),
        )
        selected_index = int(tf.argmax(eligible_value).numpy())
        selected_valid = bool(line_valid[selected_index].numpy())
        selected_multiplier = float(LINE_MULTIPLIERS[selected_index])
        selected_value = float(line_value[selected_index].numpy())
        current_value = float(value[0].numpy())
        improvement = selected_value - current_value
        realized_step = selected_multiplier * direction
        realized_norm = float(tf.reduce_max(tf.abs(realized_step)).numpy())
        iteration_rows.append(
            {
                "iteration": iteration,
                "point": base._json_ready(current),
                "value": current_value,
                "score": base._json_ready(score[0]),
                "score_norm_inf": score_norm,
                "hessian": hessian_evidence,
                "regularization": regularization,
                "raw_direction": base._json_ready(raw_direction),
                "raw_direction_norm_inf": float(raw_norm.numpy()),
                "direction_scale": float(scale.numpy()),
                "line_multipliers": LINE_MULTIPLIERS,
                "line_values": base._json_ready(line_value),
                "line_valid": base._json_ready(line_valid),
                "selected_index": selected_index,
                "selected_multiplier": selected_multiplier,
                "selected_valid": selected_valid,
                "value_improvement": improvement,
                "realized_step": base._json_ready(realized_step),
                "realized_step_norm_inf": realized_norm,
            }
        )
        if not selected_valid:
            terminal_reason = "no_valid_line_candidate"
            break
        current = candidates[selected_index]
        if (
            realized_norm <= STEP_NORM_MAX
            and -1.0e-12 <= improvement <= VALUE_IMPROVEMENT_MAX
        ):
            converged = True
            terminal_reason = "step_and_value_stagnation_candidate"
            break

    final_value, final_score, final_status = evaluate_one(current[None, :])
    final_valid = bool(_valid_rows(tf, final_value, final_score, final_status)[0].numpy())
    final_score_norm = float(tf.reduce_max(tf.abs(final_score[0])).numpy())
    score_gate_passed = bool(final_valid and final_score_norm <= SCORE_NORM_MAX)
    if score_gate_passed:
        converged = True
        terminal_reason = "terminal_score_norm_passed"

    hessian_a, evidence_a = _score_hessian(
        tf, evaluate_stencil, current, basis, step=ITERATION_FD_STEP
    )
    hessian_b, evidence_b = _score_hessian(
        tf, evaluate_stencil, current, basis, step=TERMINAL_FD_STEP
    )
    hessian_relative_gap = None
    hessian_stable = False
    raw_precision_spd = False
    final_geometry = None
    affine_checks = None
    if hessian_a is not None and hessian_b is not None:
        denominator = tf.maximum(tf.linalg.norm(hessian_b), tf.constant(1.0e-30, tf.float64))
        hessian_relative_gap = float(
            (tf.linalg.norm(hessian_a - hessian_b) / denominator).numpy()
        )
        hessian_stable = hessian_relative_gap <= HESSIAN_RELATIVE_GAP_MAX
        raw_precision = -hessian_b
        raw_eigenvalues = tf.linalg.eigvalsh(raw_precision)
        raw_precision_spd = bool(
            tf.reduce_min(raw_eigenvalues).numpy() > RAW_PRECISION_MIN_EIGENVALUE
        )
        precision, covariance, factor, regularization = _regularized_geometry(
            tf, raw_precision
        )
        final_geometry = {
            "center": base._json_ready(current),
            "negative_hessian_raw": base._json_ready(raw_precision),
            "raw_precision_eigenvalues": base._json_ready(raw_eigenvalues),
            "raw_precision_spd": raw_precision_spd,
            "raw_precision_min_eigenvalue_required": RAW_PRECISION_MIN_EIGENVALUE,
            "precision_regularized": base._json_ready(precision),
            "covariance": base._json_ready(covariance),
            "cholesky_factor": base._json_ready(factor),
            "regularization": regularization,
            "coordinate_program": "theta=center+z@factor.T",
            "inverse_program": "solve(factor,(theta-center).T).T",
            "score_program": "score_z=score_theta@factor",
            "constant_log_abs_det_included": True,
        }
        provisional_hash = hashlib.sha256(
            json.dumps(final_geometry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        affine = StructuralAffineTargetAdapter(
            base_adapter=adapter,
            center=current,
            factor=factor,
            target_signature=identity.target_signature,
            geometry_sha256=provisional_hash,
        )
        check_points = tf.stack((tf.zeros([5], tf.float64), basis[0] * 0.1), axis=0)
        affine_checks = _affine_checks(tf, affine, check_points)

    passed = bool(
        converged
        and score_gate_passed
        and hessian_stable
        and raw_precision_spd
        and final_geometry is not None
        and affine_checks is not None
        and affine_checks["passed"] is True
    )
    result = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_geometry.v1",
        "program_id": structural.PROGRAM_ID,
        "cell_id": CELL_ID,
        "completed": True,
        "passed": passed,
        "decision": (
            "ADMIT_STR_UKF_AFFINE_GEOMETRY"
            if passed
            else "BLOCK_STR_UKF_AFFINE_GEOMETRY"
        ),
        "identity_reference": identity_reference,
        "target_identity": identity.payload(),
        "source_attempt": start_reference,
        "config": {
            "iteration_fd_step": ITERATION_FD_STEP,
            "terminal_fd_step": TERMINAL_FD_STEP,
            "max_iterations": MAX_ITERATIONS,
            "trust_radius_inf": TRUST_RADIUS_INF,
            "line_multipliers": LINE_MULTIPLIERS,
            "score_norm_max": SCORE_NORM_MAX,
            "step_norm_max": STEP_NORM_MAX,
            "value_improvement_max": VALUE_IMPROVEMENT_MAX,
            "hessian_relative_gap_max": HESSIAN_RELATIVE_GAP_MAX,
            "raw_precision_min_eigenvalue": RAW_PRECISION_MIN_EIGENVALUE,
        },
        "iteration_rows": iteration_rows,
        "converged": converged,
        "score_gate_passed": score_gate_passed,
        "terminal_reason": terminal_reason,
        "final_point": base._json_ready(current),
        "final_value": float(final_value[0].numpy()),
        "final_score": base._json_ready(final_score[0]),
        "final_score_norm_inf": final_score_norm,
        "final_valid": final_valid,
        "terminal_hessian_step_a": evidence_a,
        "terminal_hessian_step_b": evidence_b,
        "terminal_hessian_relative_gap": hessian_relative_gap,
        "terminal_hessian_stable": hessian_stable,
        "raw_precision_spd": raw_precision_spd,
        "final_geometry": final_geometry,
        "affine_checks": affine_checks,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    base._write_new_json(output_root / "result.json", result)
    base._write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=identity.target_signature,
            wall_time=time.monotonic() - started,
        ),
    )
    _write_hashes(output_root)
    return result


def _rebuild_identity(tf: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_campaign import (
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        StructuralUKFLikelihoodRecomposer,
        generate_frozen_structural_dataset_tf,
        make_structural_ukf_neutra_adapter,
        structural_source_probit_jacobian_value_score,
        structural_source_uniform_prior_value_score,
        structural_truth_source,
    )

    identity_reference = base._verify_source_root(structural.IDENTITY_ROOT)
    expected_identity = base._read_mapping(structural.IDENTITY_ROOT / "target_identity.json")
    registry = base._read_mapping(structural.IDENTITY_ROOT / "repaired_registry.json")
    registry_hash = base._file_sha256(structural.IDENTITY_ROOT / "repaired_registry.json")
    _states, observations = generate_frozen_structural_dataset_tf()
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    audit_points = structural._audit_points(tf, structural_truth_source)
    recomposer = StructuralUKFLikelihoodRecomposer(adapter)
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=audit_points,
        prior_value_score_fn=structural_source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=structural_source_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    identity = issue_typed_neutra_target_identity(
        program_id=structural.PROGRAM_ID,
        scope_kind="model_cell",
        scope_id=CELL_ID,
        adapter=adapter,
        recomposition=recomposition,
        registry_row=registry,
        registry_artifact_sha256=registry_hash,
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if base._json_ready(identity.payload()) != expected_identity:
        raise base.P4PlainHMCError("structural typed identity drift in geometry run")
    if identity.target_signature != structural.EXPECTED_TYPED_SIGNATURE:
        raise base.P4PlainHMCError("structural target signature drift in geometry run")
    return adapter, identity, identity_reference


def _load_source_start(tf: Any) -> tuple[Any, Mapping[str, Any]]:
    result_path = SOURCE_ROOT / "result.json"
    if base._file_sha256(result_path) != SOURCE_RESULT_SHA256:
        raise base.P4PlainHMCError("structural source HMC result hash mismatch")
    declared = base._read_mapping(SOURCE_ROOT / "artifact_hashes.json")["artifacts"]
    for relative_path, expected in declared.items():
        if base._file_sha256(SOURCE_ROOT / relative_path) != expected:
            raise base.P4PlainHMCError(f"structural source artifact hash mismatch: {relative_path}")
    result = base._read_mapping(result_path)
    sequential = result["sequential_run"]
    first_health = sequential["warmup_checks"][0]["health"]
    if not (
        result["passed"] is False
        and sequential["warmup_results_per_chain"] == 1000
        and sequential["retained_results_per_chain"] == 0
        and sequential["hard_vetoes"] == ["warmup_chunk_health_failed"]
        and first_health["energy_error_divergence_count"] == 1
    ):
        raise base.P4PlainHMCError("source attempt is not the frozen one-divergence failure")
    metadata_path = SOURCE_ROOT / "samples/warmup/cumulative/metadata.json"
    metadata = base._read_mapping(metadata_path)
    tensor_path = Path(str(metadata["model_path"]))
    draws = tf.io.parse_tensor(tf.io.read_file(str(tensor_path)), out_type=tf.float64)
    draws = tf.ensure_shape(draws, (1000, 4, 5))
    if not bool(tf.reduce_all(tf.math.is_finite(draws)).numpy()):
        raise base.P4PlainHMCError("source tuning-only warm-up contains nonfinite states")
    start = tf.reduce_mean(draws, axis=(0, 1))
    return start, {
        "root": str(SOURCE_ROOT),
        "result_sha256": SOURCE_RESULT_SHA256,
        "artifact_hashes_sha256": base._file_sha256(SOURCE_ROOT / "artifact_hashes.json"),
        "warmup_tensor_sha256": base._file_sha256(tensor_path),
        "source_failure": (
            "historical_finite_extreme_log_acceptance_misclassified_as_energy_veto"
        ),
        "source_failure_interpretation_retired": True,
        "source_role": "pooled_mean_is_mode_locator_start_hypothesis_only",
        "source_draws_never_used_as_geometry_covariance_or_inference": True,
        "point": base._json_ready(start),
    }


def _score_hessian(
    tf: Any, evaluate_stencil: Any, point: Any, basis: Any, *, step: float
) -> tuple[Any | None, Mapping[str, Any]]:
    epsilon = tf.constant(step, tf.float64)
    stencil = tf.concat(
        (point[None, :] + epsilon * basis, point[None, :] - epsilon * basis), axis=0
    )
    value, score, status = evaluate_stencil(stencil)
    valid = _valid_rows(tf, value, score, status)
    raw = tf.transpose((score[:5] - score[5:]) / (2.0 * epsilon))
    symmetric = 0.5 * (raw + tf.transpose(raw))
    all_valid = bool(tf.reduce_all(valid).numpy())
    finite = bool(tf.reduce_all(tf.math.is_finite(symmetric)).numpy())
    evidence = {
        "step": step,
        "stencil_valid": base._json_ready(valid),
        "all_stencil_valid": all_valid,
        "stencil_value": base._json_ready(value),
        "raw_hessian": base._json_ready(raw),
        "symmetric_hessian": base._json_ready(symmetric),
        "symmetry_max_gap": float(tf.reduce_max(tf.abs(raw - tf.transpose(raw))).numpy()),
        "finite": finite,
    }
    return (symmetric if all_valid and finite else None), evidence


def _regularized_geometry(
    tf: Any, precision_raw: Any
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    raw = 0.5 * (
        tf.convert_to_tensor(precision_raw, tf.float64)
        + tf.transpose(tf.convert_to_tensor(precision_raw, tf.float64))
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(raw)
    largest_abs = tf.reduce_max(tf.abs(eigenvalues))
    floor = tf.maximum(
        tf.constant(1.0e-8, tf.float64),
        tf.constant(1.0e-6, tf.float64) * largest_abs,
    )
    regularized_eigenvalues = tf.maximum(eigenvalues, floor)
    precision = tf.einsum("ik,k,jk->ij", eigenvectors, regularized_eigenvalues, eigenvectors)
    precision = 0.5 * (precision + tf.transpose(precision))
    precision_factor = tf.linalg.cholesky(precision)
    covariance = tf.linalg.cholesky_solve(precision_factor, tf.eye(5, dtype=tf.float64))
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    factor = tf.linalg.cholesky(covariance)
    return precision, covariance, factor, {
        "raw_eigenvalues": base._json_ready(eigenvalues),
        "realized_eigenvalue_floor": float(floor.numpy()),
        "regularized_eigenvalues": base._json_ready(regularized_eigenvalues),
        "clipped_eigenvalue_count": int(tf.math.count_nonzero(eigenvalues < floor).numpy()),
        "regularized_condition_number": float(
            (regularized_eigenvalues[-1] / regularized_eigenvalues[0]).numpy()
        ),
    }


def _affine_checks(tf: Any, adapter: StructuralAffineTargetAdapter, z: Any) -> Mapping[str, Any]:
    theta = adapter.forward(z)
    round_trip = adapter.inverse_batch(theta)
    raw_value, raw_score = adapter.base_adapter.log_prob_and_grad(theta)
    affine_value, affine_score = adapter.log_prob_and_grad(z)
    round_trip_gap = float(tf.reduce_max(tf.abs(round_trip - z)).numpy())
    value_gap = float(
        tf.reduce_max(tf.abs(affine_value - (raw_value + adapter.log_abs_det))).numpy()
    )
    score_gap = float(
        tf.reduce_max(
            tf.abs(
                affine_score
                - tf.tensordot(raw_score, adapter.factor, axes=[[-1], [0]])
            )
        ).numpy()
    )
    return {
        "round_trip_max_gap": round_trip_gap,
        "value_chain_rule_max_gap": value_gap,
        "score_chain_rule_max_gap": score_gap,
        "passed": bool(
            round_trip_gap <= 1.0e-10
            and value_gap <= 1.0e-10
            and score_gap <= 1.0e-10
        ),
    }


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


def _write_hashes(output_root: Path) -> None:
    hashes = {
        str(path.relative_to(output_root)): base._file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    base._write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_geometry_hashes.v1",
            "artifacts": hashes,
        },
    )


def _run_manifest(
    *,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str,
    wall_time: float,
) -> Mapping[str, Any]:
    git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_geometry_manifest.v1",
        "program_id": structural.PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p5_structural_affine_geometry.py "
            f"--output-root {output_root}"
        ),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "data_version": "chapter18b-structural-T100-seed-20260716-15001",
        "random_seeds": "N/A; deterministic geometry diagnostic",
        "target_signature": target_signature,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_geometry(args.output_root)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "passed": result["passed"],
                "decision": result["decision"],
                "terminal_reason": result["terminal_reason"],
                "final_score_norm_inf": result["final_score_norm_inf"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
