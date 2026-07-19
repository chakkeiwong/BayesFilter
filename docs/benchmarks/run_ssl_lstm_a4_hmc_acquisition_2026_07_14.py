#!/usr/bin/env python3
"""Calibration-only four-chain HMC acquisition for SSL-LSTM Phase A4.

This Tier 2 harness delegates posterior values and scores to the locked A1
target. It adds only the A0 affine sampler chart, bounded full-chain authority,
private retained-tensor readback, and prospective admission diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-"
    "subplan-2026-07-11.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-"
    "result-2026-07-14.md"
)
HARNESS_PATH = Path("docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py")
TEST_PATH = Path("tests/test_ssl_lstm_a4_hmc_acquisition.py")
A0_LOCK_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"
)
A4_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition"
)
A0_LOCK_SHA256 = "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
TARGET_SEMANTIC_SHA256 = (
    "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
)
TARGET_ADAPTER_SHA256 = (
    "004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556"
)
TARGET_SCOPE = "ssl_lstm_completion:a1:masked_svd_ukf_four_parameter"
ACQUISITION_SCOPE = TARGET_SCOPE + ":a4_calibration_only_four_chain_hmc"
ROOT_SEED = (20260714, 1404)
GPU_BUDGET_SECONDS = 8.0 * 60.0 * 60.0
R_HAT_MAX = 1.05
ESS_MIN = 100.0
MCSE_SD_RATIO_MAX = 0.10
ACCEPTANCE_MIN = 0.20
ACCEPTANCE_MAX = 0.95
FD_STEP = 1.0e-5
FD_RTOL = 5.0e-3
FD_ATOL = 8.0e-4

INITIAL_STATES = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
KERNEL_CANDIDATES = (
    ("balanced", 0.3925, 4),
    ("smaller_step", 0.19625, 8),
    ("larger_step", 0.785, 2),
)
SEGMENT_DRAWS = (250, 250, 500, 1000)


class AcquisitionError(RuntimeError):
    """Raised when an A4 acquisition contract cannot be satisfied."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _replace_nonfinite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _replace_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_replace_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(_canonical_bytes(_replace_nonfinite(dict(value))) + b"\n")


def _strict_load(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise AcquisitionError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise AcquisitionError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise AcquisitionError(f"expected JSON object: {path}")
    return value


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _file_row(path: Path, role: str) -> dict[str, Any]:
    absolute = ROOT / path
    return {
        "path": path.as_posix(),
        "role": role,
        "bytes": absolute.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_bindings() -> list[dict[str, Any]]:
    return [
        _file_row(PLAN_PATH, "prospective_evidence_contract"),
        _file_row(HARNESS_PATH, "calibration_only_hmc_harness"),
        _file_row(TEST_PATH, "focused_harness_tests"),
        _file_row(A0_LOCK_PATH, "locked_sampler_geometry"),
        _file_row(
            Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
            "locked_a1_target",
        ),
        _file_row(
            Path("bayesfilter/inference/hmc.py"),
            "retained_hmc_archive_runtime",
        ),
        _file_row(
            Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
            "rank_normalized_admission_diagnostics",
        ),
    ]


def _assert_current_source_bindings(payload: Mapping[str, Any], path: Path) -> None:
    if payload.get("source_files") != _source_bindings():
        raise AcquisitionError(f"source binding drift since prior GPU artifact: {path}")


def _geometry() -> tuple[Any, Any, Any, dict[str, Any]]:
    import tensorflow as tf

    if _sha256(A0_LOCK_PATH) != A0_LOCK_SHA256:
        raise AcquisitionError("A0 target-lock byte identity drift")
    lock = _strict_load(A0_LOCK_PATH)
    if lock.get("schema_version") != "bayesfilter.ssl_lstm_completion.phase_a0_target_lock.v1":
        raise AcquisitionError("unexpected A0 target-lock schema")
    if lock.get("signatures", {}).get("target_semantic_sha256") != TARGET_SEMANTIC_SHA256:
        raise AcquisitionError("A0 target semantic mismatch")
    geometry = lock.get("sampler_geometry")
    if not isinstance(geometry, dict):
        raise AcquisitionError("A0 sampler geometry missing")
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    covariance_z = tf.constant(geometry["covariance_z"]["values"], tf.float64)
    factor = tf.linalg.diag(scale) @ factor_z
    residual = float(tf.reduce_max(tf.abs(factor_z @ tf.transpose(factor_z) - covariance_z)))
    if tuple(center.shape) != (4,) or tuple(factor.shape) != (4, 4):
        raise AcquisitionError("A0 affine geometry shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(center)).numpy()):
        raise AcquisitionError("A0 affine center is nonfinite")
    if not bool(tf.reduce_all(tf.math.is_finite(factor)).numpy()):
        raise AcquisitionError("A0 affine factor is nonfinite")
    tolerance = float(geometry["checks"]["factor_covariance_tolerance"])
    if residual > tolerance:
        raise AcquisitionError("A0 factor_z covariance reconstruction failed")
    report = {
        "coordinate_formula": "theta = center + z @ factor.T",
        "score_formula": "score_z = score_theta @ factor",
        "factor_formula": "factor = diag(scale) @ factor_z",
        "factor_orientation": "row_right_transpose",
        "constant_log_jacobian": "omitted_parameter_independent_constant",
        "center": _json_safe(center),
        "scale": _json_safe(scale),
        "factor": _json_safe(factor),
        "factor_z_covariance_residual_max_abs": residual,
        "factor_z_covariance_tolerance": tolerance,
        "sampler_geometry_sha256": lock["signatures"]["sampler_geometry_sha256"],
        "role": "initialization_and_tuning_context_only_not_target_definition",
    }
    return center, factor, factor_z, report


class A4CalibrationHMCAdapter:
    """Affine wrapper over the locked A1 target for this acquisition only."""

    parameter_dim = 4

    def __init__(self) -> None:
        import tensorflow as tf
        from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (
            MASKED_POSTERIOR_CONTRACT_SHA256,
            TARGET_SEMANTIC_SHA256 as A1_TARGET_SHA256,
            locked_ssl_lstm_posterior_target,
        )

        if A1_TARGET_SHA256 != TARGET_SEMANTIC_SHA256:
            raise AcquisitionError("imported A1 target semantic mismatch")
        if MASKED_POSTERIOR_CONTRACT_SHA256 != TARGET_ADAPTER_SHA256:
            raise AcquisitionError("imported A1 adapter contract mismatch")
        self.base = locked_ssl_lstm_posterior_target()
        if self.base.adapter_signature() != TARGET_ADAPTER_SHA256:
            raise AcquisitionError("live A1 adapter signature mismatch")
        if self.base.target_signature() != TARGET_SEMANTIC_SHA256:
            raise AcquisitionError("live A1 target signature mismatch")
        self.center, self.factor, _factor_z, self.geometry_report = _geometry()
        self._batch_programs: dict[int, Any] = {}
        self._tf = tf

    def adapter_signature(self) -> str:
        payload = {
            "schema_version": "bayesfilter.ssl_lstm.a4_calibration_hmc_adapter.v1",
            "base_adapter_sha256": TARGET_ADAPTER_SHA256,
            "base_target_sha256": TARGET_SEMANTIC_SHA256,
            "sampler_geometry_sha256": self.geometry_report["sampler_geometry_sha256"],
            "target_scope": ACQUISITION_SCOPE,
        }
        return _sha256_bytes(_canonical_bytes(payload))

    def value_score_capability(self) -> Any:
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability

        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_benchmark_local_affine_a1_adapter",
            evidence_path=PLAN_PATH.as_posix(),
            target_scope=ACQUISITION_SCOPE,
            nonclaims=(
                "A4 calibration-only four-chain HMC acquisition authority",
                "delegates locked A1 values and scores without changing the posterior",
                "does not change A1 capability metadata or public/default policy",
                "no posterior correctness, convergence proof, or HMC readiness claim",
            ),
        )

    def free_from_latent(self, latent: Any) -> Any:
        z = self._tf.convert_to_tensor(latent, self._tf.float64)
        return self.center + self._tf.tensordot(
            z,
            self.factor,
            axes=[[-1], [1]],
        )

    def latent_score(self, free_score: Any) -> Any:
        score = self._tf.convert_to_tensor(free_score, self._tf.float64)
        return self._tf.tensordot(
            score,
            self.factor,
            axes=[[-1], [0]],
        )

    def _batch_program(self, batch_size: int) -> Any:
        program = self._batch_programs.get(batch_size)
        if program is None:
            tf = self._tf
            base = self.base
            center = self.center
            factor = self.factor

            @tf.function(
                input_signature=[tf.TensorSpec([batch_size, 4], tf.float64)],
                jit_compile=True,
                reduce_retracing=True,
            )
            def compiled(latent: Any) -> tuple[Any, Any]:
                free = center + latent @ tf.transpose(factor)
                value, score = base.batch_value_and_score(free)
                return value, score @ factor

            program = compiled
            self._batch_programs[batch_size] = program
        return program

    def log_prob_and_grad(self, latent: Any) -> tuple[Any, Any]:
        tf = self._tf
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            if tuple(values.shape) != (4,):
                raise ValueError("latent scalar state must have shape [4]")
            free = self.free_from_latent(values)
            value, score = self.base.value_and_score(free)
            return value, self.latent_score(score)
        if values.shape.rank == 2:
            batch_size = values.shape[0]
            if batch_size is None or tuple(values.shape) != (int(batch_size), 4):
                raise ValueError("latent batch state must have static shape [batch,4]")
            return self._batch_program(int(batch_size))(values)
        raise ValueError("latent state must have rank one or two")


def _transform_checks(adapter: A4CalibrationHMCAdapter) -> dict[str, Any]:
    import tensorflow as tf

    latent = tf.constant(INITIAL_STATES, tf.float64)
    free = adapter.free_from_latent(latent)
    value, score = adapter.log_prob_and_grad(latent)
    base_value, base_score = adapter.base.batch_value_and_score(free)
    expected_score = base_score @ adapter.factor
    value_residual = float(tf.reduce_max(tf.abs(value - base_value)))
    score_residual = float(tf.reduce_max(tf.abs(score - expected_score)))

    scalar_rows = []
    for index in range(4):
        scalar_value, scalar_score = adapter.log_prob_and_grad(latent[index])
        scalar_rows.append((scalar_value, scalar_score))
    scalar_values = tf.stack([row[0] for row in scalar_rows])
    scalar_scores = tf.stack([row[1] for row in scalar_rows])
    batch_scalar_value_residual = float(tf.reduce_max(tf.abs(value - scalar_values)))
    batch_scalar_score_residual = float(tf.reduce_max(tf.abs(score - scalar_scores)))

    finite_difference = []
    center = tf.zeros([4], tf.float64)
    center_value, center_score = adapter.log_prob_and_grad(center)
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64)
        plus, _ = adapter.log_prob_and_grad(center + FD_STEP * direction)
        minus, _ = adapter.log_prob_and_grad(center - FD_STEP * direction)
        finite_difference.append((plus - minus) / (2.0 * FD_STEP))
    finite_difference_score = tf.stack(finite_difference)
    fd_abs = tf.abs(center_score - finite_difference_score)
    fd_passed = bool(
        tf.reduce_all(fd_abs <= FD_ATOL + FD_RTOL * tf.abs(finite_difference_score)).numpy()
    )
    output_devices = sorted({str(value.device), str(score.device)})
    checked_values = tf.concat(
        [
            tf.reshape(value, [-1]),
            tf.reshape(score, [-1]),
            tf.reshape(center_value, [-1]),
            tf.reshape(finite_difference_score, [-1]),
        ],
        axis=0,
    )
    all_finite = bool(tf.reduce_all(tf.math.is_finite(checked_values)).numpy())
    passed = (
        all_finite
        and value_residual <= 1.0e-10
        and score_residual <= 1.0e-8
        and batch_scalar_value_residual <= 1.0e-10
        and batch_scalar_score_residual <= 1.0e-8
        and fd_passed
    )
    return {
        "passed": passed,
        "all_finite": all_finite,
        "value_transform_residual_max_abs": value_residual,
        "score_transform_residual_max_abs": score_residual,
        "batch_scalar_value_residual_max_abs": batch_scalar_value_residual,
        "batch_scalar_score_residual_max_abs": batch_scalar_score_residual,
        "finite_difference_residual_max_abs": float(tf.reduce_max(fd_abs)),
        "finite_difference_atol": FD_ATOL,
        "finite_difference_rtol": FD_RTOL,
        "output_devices": output_devices,
        "compiled_batch_sizes": list(adapter.base.compiled_batch_sizes()),
        "nonclaims": ["target/transform engineering checks only"],
    }


def _environment_manifest(
    *,
    started: str,
    completed: str,
    wall_time: float,
    output_paths: Sequence[Path] = (),
    random_seeds: Any = ROOT_SEED,
) -> dict[str, Any]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    physical = tf.config.list_physical_devices()
    logical = tf.config.list_logical_devices()
    return {
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "packages": {
            "tensorflow": str(tf.__version__),
            "tensorflow_probability": str(tfp.__version__),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONPYCACHEPREFIX",
                "TMPDIR",
                "CUDA_CACHE_PATH",
                "XLA_FLAGS",
            )
        },
        "physical_devices": [
            {"name": item.name, "device_type": item.device_type} for item in physical
        ],
        "logical_devices": [
            {"name": item.name, "device_type": item.device_type} for item in logical
        ],
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "dtype": "float64",
        "cpu_gpu_status": (
            "cpu_hidden_reference"
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            else "trusted_gpu_xla"
        ),
        "data_version": TARGET_SEMANTIC_SHA256,
        "random_seeds": _json_safe(random_seeds),
        "started_at_utc": started,
        "completed_at_utc": completed,
        "wall_time_seconds": wall_time,
        "output_paths": [path.as_posix() for path in output_paths],
        "trust_basis": (
            "cpu_hidden_reference_exception_not_gpu_evidence"
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            else "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "plan_path": PLAN_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
    }


def build_existing_artifact_audit() -> dict[str, Any]:
    near_misses = [
        {
            "path": "docs/benchmarks/minimal_ssl_lstm_zhaocui_hmc_validity_phase3_longer_gpu_xla_2026-07-06.json",
            "observed": {
                "chain_count": 4,
                "retained_draws": 64,
                "sample_shape": [64, 4, 24],
                "target_scope": "minimal_ssl_lstm_zhaocui_hmc_ladder:zhaocui_fixed:phase1",
                "promotion_screen": "failed",
            },
            "rejection_reasons": [
                "target_scope_and_parameter_chart_do_not_match_locked_A1",
                "promotion_screen_failed_split_rhat_and_ess",
                "no_reusable_retained_chain_archive_found",
            ],
        },
        {
            "path": "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json",
            "observed": {
                "cpu_hidden": True,
                "retained_transitions": 128,
                "sample_shape": [128, 4],
                "chain_count": 1,
                "native_divergence_status": "not_exposed_by_kernel",
            },
            "rejection_reasons": [
                "not_a_four_chain_archive",
                "no_chain_shaped_retained_parameter_archive",
                "finite_acceptance_screen_is_not_sampler_admission",
            ],
        },
        {
            "path": "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_replicated_diagnostic_cpu_hidden_2026-07-08.json",
            "observed": {
                "independent_seed_runs": 3,
                "retained_transitions_per_run": 16,
                "single_chain_per_run": True,
            },
            "rejection_reasons": [
                "independent_short_runs_are_not_one_four_chain_archive",
                "no_rank_normalized_four_chain_admission_diagnostics",
            ],
        },
    ]
    archive_candidates = sorted(
        path.relative_to(ROOT).as_posix()
        for suffix in ("*.npz", "*.npy", "*.tftensor")
        for path in (ROOT / "docs").rglob(suffix)
        if "ssl_lstm" in path.as_posix().lower()
    )
    reusable = [path for path in archive_candidates if "retained_samples" in path]
    return {
        "schema_version": "bayesfilter.ssl_lstm.a4_existing_hmc_artifact_audit.v1",
        "created_at_utc": _now(),
        "question": "Does an existing locked-A1 four-chain retained HMC archive satisfy the A4 admission contract?",
        "required_contract": {
            "target_scope": TARGET_SCOPE,
            "sample_shape": "[draw,4,4]",
            "minimum_retained_draws": 250,
            "sampler_validity": "finite_movement_rhat_ess_mcse_acceptance",
            "native_divergence": "positive_veto_if_available_unavailability_qualified",
        },
        "near_misses": near_misses,
        "ssl_lstm_numeric_archive_candidates": archive_candidates,
        "reusable_locked_a1_retained_archives": reusable,
        "qualifying_artifact_count": 0,
        "decision": "NO_EXISTING_ARTIFACT_QUALIFIES_RUN_AUTHORIZED_CALIBRATION_ONLY_ACQUISITION",
        "source_files": _source_bindings(),
        "nonclaims": [
            "historical target compatibility does not repair archive shape or validity evidence",
            "no historical HMC artifact is being called invalid for its original purpose",
        ],
    }


def _parse_tensor(path: Path, dtype: Any) -> Any:
    import tensorflow as tf

    data = (ROOT / path).read_bytes()
    return tf.io.parse_tensor(data, out_type=dtype)


def _read_archive(archive_dir: Path, label: str) -> tuple[Any, Any, dict[str, Any]]:
    import tensorflow as tf

    manifest_path = archive_dir / f"{label}_private_manifest.json"
    manifest = _strict_load(manifest_path)
    if manifest.get("artifact_type") != "bayesfilter_private_retained_sample_hmc_archive":
        raise AcquisitionError("unexpected retained archive manifest type")
    shards = manifest.get("sample_shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise AcquisitionError("retained archive must contain exactly one sample shard")
    shard = shards[0]
    sample_path = Path(str(shard["path"]))
    if _sha256(sample_path) != shard["sha256"]:
        raise AcquisitionError("retained sample shard hash mismatch")
    state = manifest.get("sidecars", {}).get("final_state")
    if not isinstance(state, dict):
        raise AcquisitionError("retained archive final-state sidecar missing")
    state_path = Path(str(state["path"]))
    if _sha256(state_path) != state["sha256"]:
        raise AcquisitionError("retained final-state hash mismatch")
    samples = _parse_tensor(sample_path, tf.float64)
    final_state = _parse_tensor(state_path, tf.float64)
    expected_shape = tuple(int(item) for item in shard["shape"])
    if tuple(samples.shape) != expected_shape or expected_shape[1:] != (4, 4):
        raise AcquisitionError("retained sample shard shape mismatch")
    if tuple(final_state.shape) != (4, 4):
        raise AcquisitionError("retained final-state shape mismatch")
    return samples, final_state, manifest


def _coordinate_diagnostics(chain_major: Any) -> dict[str, Any]:
    from bayesfilter.inference.hmc_posterior_diagnostics import (
        compute_coordinate_diagnostics,
    )

    return _json_safe(compute_coordinate_diagnostics(chain_major))


def _admission_diagnostics(
    *,
    latent_draw_major: Any,
    adapter: A4CalibrationHMCAdapter,
    segment_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import tensorflow as tf

    latent = tf.convert_to_tensor(latent_draw_major, tf.float64)
    if latent.shape.rank != 3 or tuple(latent.shape[1:]) != (4, 4):
        raise AcquisitionError("cumulative latent archive must have shape [draw,4,4]")
    draws = int(latent.shape[0])
    if draws < 4 or draws % 2:
        raise AcquisitionError("diagnostic draw count must be even and at least four")
    free = adapter.free_from_latent(latent)
    finite = bool(
        tf.reduce_all(tf.math.is_finite(latent)).numpy()
        and tf.reduce_all(tf.math.is_finite(free)).numpy()
    )
    chain_major_latent = tf.transpose(latent, [1, 0, 2])
    chain_major_free = tf.transpose(free, [1, 0, 2])
    moved = tf.reduce_any(
        tf.not_equal(chain_major_latent[:, 1:, :], chain_major_latent[:, :-1, :]),
        axis=(1, 2),
    )
    accepted_by_segment = []
    total_by_segment = []
    divergence_statuses = []
    divergence_counts = []
    log_accept_nonfinite = 0
    target_nonfinite = 0
    for manifest in segment_manifests:
        diagnostics = manifest["diagnostics_private_metadata"]
        health = diagnostics["sampler_health_diagnostics"]
        segment_draws = int(manifest["retained_sample_count"])
        rates = [float(value) for value in health["acceptance_rate_by_chain"]]
        if len(rates) != 4:
            raise AcquisitionError("segment acceptance telemetry must contain four chains")
        raw_counts = [rate * segment_draws for rate in rates]
        rounded_counts = [int(round(value)) for value in raw_counts]
        if any(
            abs(raw - rounded) > 1.0e-9
            for raw, rounded in zip(raw_counts, rounded_counts, strict=True)
        ):
            raise AcquisitionError("segment acceptance rates do not reconstruct exact counts")
        accepted_by_segment.append(rounded_counts)
        total_by_segment.append([segment_draws] * 4)
        divergence_statuses.append(diagnostics["native_divergence_status"])
        if diagnostics.get("divergence_count") is not None:
            divergence_counts.append(int(diagnostics["divergence_count"]))
        log_accept_nonfinite += int(health["log_accept_ratio"]["nonfinite_count"])
        target_nonfinite += int(health["target_log_prob"]["nonfinite_count"])
    accepted_by_chain = tf.reduce_sum(
        tf.constant(accepted_by_segment, tf.int64), axis=0
    )
    total_by_chain = tf.reduce_sum(tf.constant(total_by_segment, tf.int64), axis=0)
    acceptance_by_chain = tf.cast(accepted_by_chain, tf.float64) / tf.cast(
        total_by_chain, tf.float64
    )
    aggregate_acceptance = float(
        tf.cast(tf.reduce_sum(accepted_by_chain), tf.float64)
        / tf.cast(tf.reduce_sum(total_by_chain), tf.float64)
    )

    def coordinate_failures(name: str, values: Mapping[str, Any]) -> list[str]:
        failures = []
        maximum = values["rank_normalized_split_rhat"]["maximum"]
        bulk = values["rank_normalized_ess"]["bulk"]
        tail = values["rank_normalized_ess"]["tail"]
        ratio = values["mean"]["mcse_sd_ratio"]
        arrays = (maximum, bulk, tail, ratio)
        if not all(math.isfinite(float(item)) for array in arrays for item in array):
            return [f"{name}:nonfinite_rank_ess_or_mcse"]
        if max(float(item) for item in maximum) > R_HAT_MAX:
            failures.append(f"{name}:rank_normalized_split_rhat_above_threshold")
        if min(float(item) for item in bulk) < ESS_MIN:
            failures.append(f"{name}:bulk_ess_below_threshold")
        if min(float(item) for item in tail) < ESS_MIN:
            failures.append(f"{name}:tail_ess_below_threshold")
        if max(float(item) for item in ratio) > MCSE_SD_RATIO_MAX:
            failures.append(f"{name}:mcse_sd_ratio_above_threshold")
        return failures

    hard_vetoes = []
    promotion_vetoes = []
    if not finite:
        hard_vetoes.append("latent_or_free_samples_nonfinite")
    if not all(bool(item) for item in _json_safe(moved)):
        hard_vetoes.append("unmoved_chain")
    if log_accept_nonfinite:
        hard_vetoes.append("log_accept_ratio_nonfinite")
    if target_nonfinite:
        hard_vetoes.append("target_log_prob_nonfinite")
    if any(count > 0 for count in divergence_counts):
        hard_vetoes.append("positive_native_divergence_count")
    acceptance_list = [float(item) for item in _json_safe(acceptance_by_chain)]
    if not ACCEPTANCE_MIN <= aggregate_acceptance <= ACCEPTANCE_MAX:
        promotion_vetoes.append("aggregate_acceptance_outside_threshold")
    if any(not ACCEPTANCE_MIN <= item <= ACCEPTANCE_MAX for item in acceptance_list):
        promotion_vetoes.append("per_chain_acceptance_outside_threshold")
    if hard_vetoes:
        coordinate_diagnostics: Mapping[str, Any] = {
            "status": "not_computed_because_hard_veto_fired"
        }
    else:
        latent_diag = _coordinate_diagnostics(chain_major_latent)
        free_diag = _coordinate_diagnostics(chain_major_free)
        promotion_vetoes.extend(coordinate_failures("latent", latent_diag))
        promotion_vetoes.extend(coordinate_failures("free", free_diag))
        coordinate_diagnostics = {"latent": latent_diag, "free": free_diag}
    admitted = not hard_vetoes and not promotion_vetoes
    return {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_cumulative_diagnostics.v1",
        "created_at_utc": _now(),
        "decision": "ADMITTED_FOR_A4_CALIBRATION" if admitted else (
            "HARD_VETO_STOP" if hard_vetoes else "PROMOTION_VETO_EXTEND_IF_BUDGET_ALLOWS"
        ),
        "admitted": admitted,
        "draw_count_per_chain": draws,
        "chain_count": 4,
        "parameter_count": 4,
        "sample_shape": list(latent.shape),
        "all_samples_finite": finite,
        "chain_moved": _json_safe(moved),
        "acceptance_rate": aggregate_acceptance,
        "acceptance_rate_by_chain": acceptance_list,
        "native_divergence_statuses": divergence_statuses,
        "native_divergence_interpretation": (
            "not_exposed_by_kernel_is_unavailability_not_zero_divergences"
        ),
        "log_accept_ratio_nonfinite_count": log_accept_nonfinite,
        "target_log_prob_nonfinite_count": target_nonfinite,
        "hard_vetoes": hard_vetoes,
        "promotion_vetoes": list(dict.fromkeys(promotion_vetoes)),
        "thresholds": {
            "rank_normalized_split_rhat_max": R_HAT_MAX,
            "bulk_ess_min": ESS_MIN,
            "tail_ess_min": ESS_MIN,
            "mcse_sd_ratio_max": MCSE_SD_RATIO_MAX,
            "acceptance_min": ACCEPTANCE_MIN,
            "acceptance_max": ACCEPTANCE_MAX,
        },
        "coordinate_diagnostics": coordinate_diagnostics,
        "coordinate_transform": adapter.geometry_report,
        "nonclaims": [
            "finite-sample calibration-input admission screen only",
            "no convergence or stationarity proof",
            "no posterior correctness, sampler superiority, or readiness claim",
            "native divergence unavailability is not zero divergences",
        ],
    }


def _require_gpu() -> None:
    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    if not physical or not logical:
        raise AcquisitionError("trusted GPU/XLA mode requires visible physical and logical GPU")


def _classify_canary_movement(moved: Any) -> tuple[list[str], list[str]]:
    import tensorflow as tf

    values = tf.convert_to_tensor(moved, tf.bool)
    if tuple(values.shape) != (4,):
        raise AcquisitionError("canary movement must contain exactly four chains")
    if not bool(tf.reduce_any(values).numpy()):
        return ["all_canary_chains_unmoved"], []
    if not bool(tf.reduce_all(values).numpy()):
        return [], ["subset_of_canary_chains_unmoved_tuning_attention"]
    return [], []


def _run_archive(
    *,
    adapter: A4CalibrationHMCAdapter,
    archive_dir: Path,
    label: str,
    current_state: Any,
    num_results: int,
    num_burnin_steps: int,
    step_size: float,
    leapfrog_steps: int,
    seed: tuple[int, int],
    role: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], float]:
    from bayesfilter.inference.hmc import (
        RetainedSampleHMCArchiveConfig,
        build_retained_sample_hmc_archive_runner,
    )

    config = RetainedSampleHMCArchiveConfig(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps,
        seed=seed,
        use_xla=True,
        target_scope=ACQUISITION_SCOPE,
        chain_execution_mode="tf_function",
    )
    runner = build_retained_sample_hmc_archive_runner(adapter, current_state, config)
    started = time.perf_counter()
    result = runner.run(
        archive_dir=ROOT / archive_dir,
        archive_label=label,
        current_state=current_state,
        seed=seed,
        metadata={
            "role": role,
            "plan_path": PLAN_PATH.as_posix(),
            "target_semantic_sha256": TARGET_SEMANTIC_SHA256,
            "target_adapter_sha256": TARGET_ADAPTER_SHA256,
            "adapter_signature": adapter.adapter_signature(),
            "root_seed": ROOT_SEED,
            "segment_seed": seed,
        },
        overwrite=False,
    )
    elapsed = time.perf_counter() - started
    samples, final_state, private_manifest = _read_archive(archive_dir, label)
    diagnostics = _json_safe(result.diagnostics)
    evidence_output_devices = sorted(
        {
            str(result.final_state.device),
            str(result.final_target_log_prob.device),
        }
    )
    metadata_output_devices = [str(result.final_index.device)]
    if not evidence_output_devices or not all(
        "GPU:" in device for device in evidence_output_devices
    ):
        raise AcquisitionError(
            "trusted GPU/XLA state/target outputs were not placed on GPU: "
            f"{evidence_output_devices}"
        )
    metadata = _json_safe(result.metadata)
    metadata["evidence_output_devices"] = evidence_output_devices
    metadata["metadata_output_devices"] = metadata_output_devices
    return (samples, final_state, private_manifest, elapsed), diagnostics, metadata, elapsed


def run_cpu_check(output: Path) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    adapter = A4CalibrationHMCAdapter()
    checks = _transform_checks(adapter)
    completed_at = _now()
    wall_time = time.perf_counter() - started
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_cpu_check.v1",
        "status": "PASSED" if checks["passed"] else "FAILED",
        "artifact_role": "cpu_hidden_engineering_check_not_sampler_evidence",
        "target": {
            "scope": TARGET_SCOPE,
            "semantic_sha256": TARGET_SEMANTIC_SHA256,
            "adapter_sha256": TARGET_ADAPTER_SHA256,
            "calibration_adapter_signature": adapter.adapter_signature(),
        },
        "geometry": adapter.geometry_report,
        "checks": checks,
        "run_manifest": _environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output_paths=(output,),
        ),
        "source_files": _source_bindings(),
        "nonclaims": [
            "CPU-hidden target/transform engineering check only",
            "not HMC, sampler, posterior, GPU, or calibration evidence",
        ],
    }
    _write_json(output, payload)
    if not checks["passed"]:
        raise AcquisitionError("CPU target/transform checks failed")
    return payload


def run_canary(
    output: Path,
    archive_dir: Path,
    *,
    archive_label: str,
    seed_tail: int,
    prior_gpu_artifacts: Sequence[Path],
) -> dict[str, Any]:
    import tensorflow as tf

    _require_gpu()
    if not archive_label.strip():
        raise AcquisitionError("canary archive label must be non-empty")
    if seed_tail not in {1411, 1412}:
        raise AcquisitionError("canary seed tail is outside the reviewed attempts")
    prior_gpu_seconds = _load_prior_gpu_seconds(prior_gpu_artifacts)
    if prior_gpu_seconds + 900.0 > GPU_BUDGET_SECONDS:
        raise AcquisitionError("canary projected cost exceeds remaining GPU budget")
    started_at = _now()
    started = time.perf_counter()
    adapter = A4CalibrationHMCAdapter()
    checks = _transform_checks(adapter)
    if not checks["passed"]:
        raise AcquisitionError("GPU target/transform checks failed before canary")
    if not checks["output_devices"] or not all(
        "GPU:" in device for device in checks["output_devices"]
    ):
        raise AcquisitionError(
            "GPU target/transform outputs were not placed on GPU: "
            f"{checks['output_devices']}"
        )
    state = tf.constant(INITIAL_STATES, tf.float64)
    (samples, _final_state, private_manifest, call_s), diagnostics, metadata, _ = _run_archive(
        adapter=adapter,
        archive_dir=archive_dir,
        label=archive_label,
        current_state=state,
        num_results=8,
        num_burnin_steps=8,
        step_size=KERNEL_CANDIDATES[0][1],
        leapfrog_steps=KERNEL_CANDIDATES[0][2],
        seed=(20260714, seed_tail),
        role="trusted_gpu_xla_four_chain_canary",
    )
    moved = tf.reduce_any(tf.not_equal(samples[1:], samples[:-1]), axis=(0, 2))
    hard_vetoes = []
    if not bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()):
        hard_vetoes.append("nonfinite_canary_samples")
    movement_vetoes, repair_triggers = _classify_canary_movement(moved)
    hard_vetoes.extend(movement_vetoes)
    health = private_manifest["diagnostics_private_metadata"]["sampler_health_diagnostics"]
    if int(health["log_accept_ratio"]["nonfinite_count"]):
        hard_vetoes.append("nonfinite_canary_log_accept_ratio")
    if int(health["target_log_prob"]["nonfinite_count"]):
        hard_vetoes.append("nonfinite_canary_target_log_prob")
    completed_at = _now()
    wall_time = time.perf_counter() - started
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1",
        "status": (
            "PASSED_WITH_TUNING_REPAIR_TRIGGER"
            if not hard_vetoes and repair_triggers
            else "PASSED" if not hard_vetoes else "FAILED"
        ),
        "artifact_role": "trusted_gpu_xla_four_chain_hmc_canary",
        "target_transform_checks": checks,
        "archive_summary": {
            "sample_shape": list(samples.shape),
            "chain_moved": _json_safe(moved),
            "private_manifest_sha256": _sha256(
                archive_dir / f"{archive_label}_private_manifest.json"
            ),
        },
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "source_files": _source_bindings(),
        "hard_vetoes": hard_vetoes,
        "repair_triggers": repair_triggers,
        "budget_lineage_artifacts": [
            path.as_posix() for path in prior_gpu_artifacts
        ],
        "run_manifest": _environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output_paths=(output, archive_dir),
            random_seeds=(ROOT_SEED, (20260714, seed_tail)),
        ),
        "gpu_budget": {
            "cap_seconds": GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_gpu_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds_lower_bound": (
                GPU_BUDGET_SECONDS - prior_gpu_seconds - wall_time
            ),
        },
        "nonclaims": [
            "tiny four-chain GPU/XLA mechanics canary only; partial movement is a tuning trigger",
            "not tuning, convergence, posterior correctness, or HMC readiness evidence",
        ],
    }
    _write_json(output, payload)
    if hard_vetoes:
        raise AcquisitionError(f"GPU canary hard vetoes: {hard_vetoes}")
    return payload


def _load_prior_gpu_seconds(paths: Sequence[Path]) -> float:
    normalized = [path.as_posix() for path in paths]
    if len(normalized) != len(set(normalized)):
        raise AcquisitionError("prior GPU artifacts must not be counted twice")
    total = 0.0
    for path in paths:
        if not (ROOT / path).is_file():
            raise AcquisitionError(f"prior GPU artifact does not exist: {path}")
        payload = _strict_load(path)
        manifest = payload.get("run_manifest", {})
        if manifest.get("trust_basis") != "owner_designated_managed_session_visible_gpu_trusted":
            raise AcquisitionError(f"prior artifact is not trusted GPU evidence: {path}")
        if manifest.get("cpu_gpu_status") != "trusted_gpu_xla":
            raise AcquisitionError(f"prior artifact is not a GPU/XLA run: {path}")
        wall_time = manifest.get("wall_time_seconds")
        if wall_time is None or not math.isfinite(float(wall_time)) or float(wall_time) < 0.0:
            raise AcquisitionError(f"prior GPU artifact has invalid wall time: {path}")
        total += float(wall_time)
    return total


def _validate_tuning_budget_lineage(
    paths: Sequence[Path], *, candidate_index: int
) -> float:
    payloads = [_strict_load(path) for path in paths]
    allowed_schemas = {
        "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1",
        "bayesfilter.ssl_lstm.a4_hmc_tuning_screen.v1",
    }
    if any(payload.get("schema_version") not in allowed_schemas for payload in payloads):
        raise AcquisitionError("tuning budget lineage contains an unrelated artifact")
    canaries = [
        (path, payload)
        for path, payload in zip(paths, payloads, strict=True)
        if payload.get("schema_version") == "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1"
        and payload.get("status") in {
            "PASSED",
            "PASSED_WITH_TUNING_REPAIR_TRIGGER",
        }
    ]
    if len(canaries) != 1:
        raise AcquisitionError("tuning budget lineage requires exactly one passed GPU canary")
    qualified_canary_path, qualified_canary = canaries[0]
    _assert_current_source_bindings(qualified_canary, qualified_canary_path)
    inherited_canary = {
        str(path)
        for path in qualified_canary.get("budget_lineage_artifacts", ())
    }
    qualified_set = inherited_canary | {qualified_canary_path.as_posix()}
    earlier_tuning = [
        (path, payload)
        for path, payload in zip(paths, payloads, strict=True)
        if payload.get("schema_version") == "bayesfilter.ssl_lstm.a4_hmc_tuning_screen.v1"
    ]
    if candidate_index == 0 and earlier_tuning:
        raise AcquisitionError("first tuning candidate must follow the canary directly")
    if candidate_index > 0:
        failed_candidates = {
            int(payload.get("candidate", {}).get("index", -1)): (path, payload)
            for path, payload in earlier_tuning
            if payload.get("status") == "NOT_SELECTED"
        }
        attempted_indices = set(failed_candidates)
        if 0 not in attempted_indices:
            raise AcquisitionError(
                "fallback tuning budget lineage requires failed candidate 0"
            )
        if any(index >= candidate_index for index in attempted_indices):
            raise AcquisitionError("fallback tuning lineage contains a non-earlier candidate")
        first_rates = [
            float(value)
            for value in failed_candidates[0][1].get("acceptance_rate_by_chain", ())
        ]
        if len(first_rates) != 4:
            raise AcquisitionError("failed candidate 0 lacks four-chain acceptance")
        low = any(value < ACCEPTANCE_MIN for value in first_rates)
        high = any(value > ACCEPTANCE_MAX for value in first_rates)
        if low and high:
            raise AcquisitionError(
                "candidate 0 has mixed low/high chain acceptance; fallback direction is ambiguous"
            )
        expected = 1 if low else 2 if high else None
        if expected is None or candidate_index != expected:
            raise AcquisitionError(
                "fallback candidate does not match candidate 0 acceptance direction"
            )
        failed_path, failed_payload = failed_candidates[0]
        _assert_current_source_bindings(failed_payload, failed_path)
        if (
            {path.as_posix() for path in paths}
            != qualified_set | {failed_path.as_posix()}
            or len(earlier_tuning) != 1
        ):
            raise AcquisitionError(
                "fallback tuning lineage must contain canary ancestry and failed candidate 0"
            )
    elif {path.as_posix() for path in paths} != qualified_set:
        raise AcquisitionError(
            "candidate 0 tuning lineage must contain exactly the qualified canary ancestry"
        )
    return _load_prior_gpu_seconds(paths)


def _validate_segment_budget_lineage(
    paths: Sequence[Path],
    *,
    selected_tuning: Path,
    previous_segment_outputs: Sequence[Path],
) -> float:
    normalized = {path.as_posix() for path in paths}
    required = {
        selected_tuning.as_posix(),
        *(path.as_posix() for path in previous_segment_outputs),
    }
    payloads = [_strict_load(path) for path in paths]
    canaries = [
        payload
        for payload in payloads
        if payload.get("schema_version") == "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1"
        and payload.get("status") in {
            "PASSED",
            "PASSED_WITH_TUNING_REPAIR_TRIGGER",
        }
    ]
    if len(canaries) != 1:
        raise AcquisitionError("segment budget lineage requires exactly one passed GPU canary")
    selected = _strict_load(selected_tuning)
    if (
        selected.get("schema_version") != "bayesfilter.ssl_lstm.a4_hmc_tuning_screen.v1"
        or selected.get("status") != "SELECTED"
    ):
        raise AcquisitionError("segment budget lineage requires the selected tuning artifact")
    _assert_current_source_bindings(selected, selected_tuning)
    for path in previous_segment_outputs:
        _assert_current_source_bindings(_strict_load(path), path)
    inherited = {
        str(path)
        for path in selected.get("budget_lineage_artifacts", ())
    }
    expected = required | inherited
    missing = sorted(expected - normalized)
    extra = sorted(normalized - expected)
    if missing:
        raise AcquisitionError(
            f"segment budget lineage omits required artifacts: {missing}"
        )
    if extra:
        raise AcquisitionError(f"segment budget lineage contains extra artifacts: {extra}")
    return _load_prior_gpu_seconds(paths)


def run_tune(
    *, output: Path,
    archive_dir: Path,
    candidate_index: int,
    prior_gpu_artifacts: Sequence[Path],
) -> dict[str, Any]:
    import tensorflow as tf

    _require_gpu()
    if candidate_index < 0 or candidate_index >= len(KERNEL_CANDIDATES):
        raise AcquisitionError("invalid kernel candidate index")
    prior_gpu_seconds = _validate_tuning_budget_lineage(
        prior_gpu_artifacts,
        candidate_index=candidate_index,
    )
    name, step_size, leapfrog = KERNEL_CANDIDATES[candidate_index]
    projected = 900.0
    if prior_gpu_seconds + projected > GPU_BUDGET_SECONDS:
        raise AcquisitionError("tuning projected cost exceeds remaining GPU budget")
    started_at = _now()
    started = time.perf_counter()
    adapter = A4CalibrationHMCAdapter()
    initial = tf.constant(INITIAL_STATES, tf.float64)
    label = f"tune_{candidate_index}_{name}"
    seed = (20260714, 1420 + candidate_index)
    (samples, _final_state, manifest, call_s), diagnostics, metadata, _ = _run_archive(
        adapter=adapter,
        archive_dir=archive_dir,
        label=label,
        current_state=initial,
        num_results=64,
        num_burnin_steps=32,
        step_size=step_size,
        leapfrog_steps=leapfrog,
        seed=seed,
        role="trusted_gpu_xla_kernel_tuning_screen",
    )
    private = manifest["diagnostics_private_metadata"]
    health = private["sampler_health_diagnostics"]
    acceptance_by_chain = [float(value) for value in health["acceptance_rate_by_chain"]]
    moved = tf.reduce_any(tf.not_equal(samples[1:], samples[:-1]), axis=(0, 2))
    hard_vetoes = []
    if not bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()):
        hard_vetoes.append("nonfinite_tuning_samples")
    if not bool(tf.reduce_all(moved).numpy()):
        hard_vetoes.append("unmoved_tuning_chain")
    if int(health["log_accept_ratio"]["nonfinite_count"]):
        hard_vetoes.append("nonfinite_tuning_log_accept_ratio")
    if int(health["target_log_prob"]["nonfinite_count"]):
        hard_vetoes.append("nonfinite_tuning_target_log_prob")
    divergence = private.get("divergence_count")
    if divergence is not None and int(divergence) > 0:
        hard_vetoes.append("positive_native_divergence_count")
    acceptance_passed = all(
        ACCEPTANCE_MIN <= value <= ACCEPTANCE_MAX for value in acceptance_by_chain
    )
    selected = not hard_vetoes and acceptance_passed
    completed_at = _now()
    wall_time = time.perf_counter() - started
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_tuning_screen.v1",
        "status": "SELECTED" if selected else ("HARD_VETO" if hard_vetoes else "NOT_SELECTED"),
        "candidate": {
            "index": candidate_index,
            "name": name,
            "step_size": step_size,
            "num_leapfrog_steps": leapfrog,
            "trajectory_length": step_size * leapfrog,
            "seed": seed,
        },
        "selection_policy": "first_candidate_passing_all_chain_acceptance_and_hard_veto_screens",
        "acceptance_rate_by_chain": acceptance_by_chain,
        "acceptance_bounds": [ACCEPTANCE_MIN, ACCEPTANCE_MAX],
        "acceptance_passed": acceptance_passed,
        "chain_moved": _json_safe(moved),
        "hard_vetoes": hard_vetoes,
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "private_manifest_sha256": _sha256(archive_dir / f"{label}_private_manifest.json"),
        "budget_lineage_artifacts": [path.as_posix() for path in prior_gpu_artifacts],
        "source_files": _source_bindings(),
        "run_manifest": _environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output_paths=(output, archive_dir),
            random_seeds=(ROOT_SEED, seed),
        ),
        "gpu_budget": {
            "cap_seconds": GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_gpu_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds": GPU_BUDGET_SECONDS - prior_gpu_seconds - wall_time,
        },
        "native_divergence_interpretation": "not_exposed_by_kernel_is_not_zero_divergences",
        "nonclaims": [
            "fixed-kernel calibration acquisition tuning screen only",
            "passing acceptance does not establish convergence or superiority",
        ],
    }
    _write_json(output, payload)
    if hard_vetoes:
        raise AcquisitionError(f"tuning hard vetoes: {hard_vetoes}")
    return payload


def run_segment(
    *,
    output: Path,
    archive_dir: Path,
    segment_index: int,
    selected_tuning: Path,
    previous_segment_outputs: Sequence[Path],
    prior_gpu_artifacts: Sequence[Path],
) -> dict[str, Any]:
    import tensorflow as tf

    _require_gpu()
    if segment_index < 0 or segment_index >= len(SEGMENT_DRAWS):
        raise AcquisitionError("invalid acquisition segment index")
    tuning = _strict_load(selected_tuning)
    if tuning.get("status") != "SELECTED":
        raise AcquisitionError("acquisition requires a selected tuning artifact")
    candidate = tuning["candidate"]
    expected_previous = segment_index
    if len(previous_segment_outputs) != expected_previous:
        raise AcquisitionError("previous segment count does not match segment index")
    previous_payloads = [_strict_load(path) for path in previous_segment_outputs]
    for expected_index, item in enumerate(previous_payloads):
        if (
            item.get("schema_version")
            != "bayesfilter.ssl_lstm.a4_hmc_acquisition_segment.v1"
            or item.get("status") != "EXTEND"
            or int(item.get("segment", {}).get("index", -1)) != expected_index
        ):
            raise AcquisitionError(
                "previous segments must be consecutive non-admitted EXTEND artifacts"
            )

    prior_gpu_seconds = _validate_segment_budget_lineage(
        prior_gpu_artifacts,
        selected_tuning=selected_tuning,
        previous_segment_outputs=previous_segment_outputs,
    )

    projected_transition_count = SEGMENT_DRAWS[segment_index] + (
        250 if segment_index == 0 else 0
    )
    projected = 3600.0 * (projected_transition_count / 250.0)
    if prior_gpu_seconds + projected > GPU_BUDGET_SECONDS:
        raise AcquisitionError("segment projected cost exceeds remaining GPU budget")
    adapter = A4CalibrationHMCAdapter()
    if segment_index == 0:
        current_state = tf.constant(INITIAL_STATES, tf.float64)
        burnin = 250
    else:
        previous_label = f"segment_{segment_index - 1}"
        _previous_samples, current_state, _previous_manifest = _read_archive(
            archive_dir, previous_label
        )
        burnin = 0
    segment_draws = SEGMENT_DRAWS[segment_index]
    seed = (20260714, 1430 + segment_index)
    label = f"segment_{segment_index}"
    started_at = _now()
    started = time.perf_counter()
    (samples, _final_state, manifest, call_s), diagnostics, metadata, _ = _run_archive(
        adapter=adapter,
        archive_dir=archive_dir,
        label=label,
        current_state=current_state,
        num_results=segment_draws,
        num_burnin_steps=burnin,
        step_size=float(candidate["step_size"]),
        leapfrog_steps=int(candidate["num_leapfrog_steps"]),
        seed=seed,
        role="trusted_gpu_xla_calibration_only_hmc_acquisition_segment",
    )
    cumulative = []
    manifests = []
    for prior_index in range(segment_index):
        prior_samples, _prior_state, prior_manifest = _read_archive(
            archive_dir, f"segment_{prior_index}"
        )
        cumulative.append(prior_samples)
        manifests.append(prior_manifest)
    cumulative.append(samples)
    manifests.append(manifest)
    latent = tf.concat(cumulative, axis=0)
    admission = _admission_diagnostics(
        latent_draw_major=latent,
        adapter=adapter,
        segment_manifests=manifests,
    )
    completed_at = _now()
    wall_time = time.perf_counter() - started
    hard_vetoes = list(admission["hard_vetoes"])
    status = "ADMITTED" if admission["admitted"] else (
        "HARD_VETO" if hard_vetoes else "EXTEND"
    )
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_acquisition_segment.v1",
        "status": status,
        "segment": {
            "index": segment_index,
            "draw_count": segment_draws,
            "burnin_count": burnin,
            "cumulative_draw_count": int(latent.shape[0]),
            "seed": seed,
            "label": label,
        },
        "selected_kernel": candidate,
        "prior_segment_artifacts": [path.as_posix() for path in previous_segment_outputs],
        "admission_diagnostics": admission,
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "private_manifest_sha256": _sha256(archive_dir / f"{label}_private_manifest.json"),
        "cumulative_private_sample_sha256": _sha256_bytes(
            bytes(tf.io.serialize_tensor(latent).numpy())
        ),
        "budget_lineage_artifacts": [path.as_posix() for path in prior_gpu_artifacts],
        "source_files": _source_bindings(),
        "run_manifest": _environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output_paths=(output, archive_dir),
            random_seeds=(ROOT_SEED, seed),
        ),
        "gpu_budget": {
            "cap_seconds": GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_gpu_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds": GPU_BUDGET_SECONDS - prior_gpu_seconds - wall_time,
        },
        "nonclaims": admission["nonclaims"],
    }
    _write_json(output, payload)
    if hard_vetoes:
        raise AcquisitionError(f"acquisition hard vetoes: {hard_vetoes}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit-existing")
    audit_parser.add_argument("--output", type=Path, required=True)

    cpu_parser = subparsers.add_parser("cpu-check")
    cpu_parser.add_argument("--output", type=Path, required=True)

    canary_parser = subparsers.add_parser("gpu-canary")
    canary_parser.add_argument("--output", type=Path, required=True)
    canary_parser.add_argument("--archive-dir", type=Path, required=True)
    canary_parser.add_argument("--archive-label", default="canary")
    canary_parser.add_argument("--seed-tail", type=int, default=1411)
    canary_parser.add_argument(
        "--prior-gpu-artifact", type=Path, action="append", default=[]
    )

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--output", type=Path, required=True)
    tune_parser.add_argument("--archive-dir", type=Path, required=True)
    tune_parser.add_argument("--candidate-index", type=int, required=True)
    tune_parser.add_argument("--prior-gpu-artifact", type=Path, action="append", default=[])

    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("--output", type=Path, required=True)
    segment_parser.add_argument("--archive-dir", type=Path, required=True)
    segment_parser.add_argument("--segment-index", type=int, required=True)
    segment_parser.add_argument("--selected-tuning", type=Path, required=True)
    segment_parser.add_argument("--previous-segment-output", type=Path, action="append", default=[])
    segment_parser.add_argument("--prior-gpu-artifact", type=Path, action="append", default=[])

    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "audit-existing":
        payload = build_existing_artifact_audit()
        _write_json(args.output, payload)
    elif args.command == "cpu-check":
        payload = run_cpu_check(args.output)
    elif args.command == "gpu-canary":
        payload = run_canary(
            args.output,
            args.archive_dir,
            archive_label=args.archive_label,
            seed_tail=args.seed_tail,
            prior_gpu_artifacts=args.prior_gpu_artifact,
        )
    elif args.command == "tune":
        payload = run_tune(
            output=args.output,
            archive_dir=args.archive_dir,
            candidate_index=args.candidate_index,
            prior_gpu_artifacts=args.prior_gpu_artifact,
        )
    else:
        payload = run_segment(
            output=args.output,
            archive_dir=args.archive_dir,
            segment_index=args.segment_index,
            selected_tuning=args.selected_tuning,
            previous_segment_outputs=args.previous_segment_output,
            prior_gpu_artifacts=args.prior_gpu_artifact,
        )
    print(
        _canonical_bytes(
            {
                "command": args.command,
                "status": payload.get("status", payload.get("decision")),
                "output": args.output.as_posix(),
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcquisitionError as exc:
        print(f"A4_HMC_ACQUISITION_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
