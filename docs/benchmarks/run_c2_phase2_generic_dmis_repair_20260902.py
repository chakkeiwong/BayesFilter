"""Frozen C2 Phase 2 diagnostic for complete DMIS and a TT control variate.

This script is deliberately a diagnostic adapter around the generic
``frozen_dmis_control_variate_tf`` kernel.  It loads the preserved t=2,3,4
frozen snapshots, evaluates the exact carried target, and compares a
standard-normal/product-Student deterministic mixture with the stored TT Gram
normalizer.  It does not modify the legacy C2 driver or claim a production
likelihood.

The target returned by the snapshot evaluator is ``E_t(u)`` whose expectation
under the standard-normal reference is the existing ``z_t``.  The generic
Lebesgue target supplied here is therefore ``gamma_t(u)=eta(u) E_t(u)``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/benchmarks"
SNAPSHOT_ATTEMPT = (
    ROOT
    / "docs/benchmarks/artifacts/c2_n4_root_cause_20260828/"
    "attempt05_coherent_20260831"
)
SNAPSHOT_ROOT = SNAPSHOT_ATTEMPT / "snapshots"
DEFAULT_OUTPUT_PARENT = (
    ROOT / "docs/benchmarks/artifacts/c2_phase2_generic_dmis_repair_20260902"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCHMARK_DIR))

import tensorflow as tf

# TensorFlow-dependent fixture imports can inspect devices during import.  Set
# the repository allocator policy before importing those modules so the first
# device interaction cannot preempt memory-growth configuration.
CPU_ONLY = os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
EARLY_MEMORY_POLICY = None
if not CPU_ONLY:
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    EARLY_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=True
    )

import tensorflow_probability as tfp

import sv_fixture_c2_20260826 as sv
from bayesfilter.highdim.frozen_dmis_control_variate_tf import (
    DTYPE,
    ROUTE_CLASSIFICATION,
    ROUTE_ID,
    complete_mixture_log_density,
    frozen_importance_directional_estimate,
    frozen_importance_estimate,
)
import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as snapshot_api
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    _hermite_product_basis,
    _log_eta,
    _log_student_t_ratio,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import DiscreteIndicatorBasis1D
from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.retained_quadratic_form_tf import prefix_row_vectors
from bayesfilter.highdim.tt import TTCore


N = 4
JOINT_DIM = 2 * N
CAPTURE_STEPS = (2, 3, 4)
MODEL_SEED = 52
OBS_SEED = 42
STUDENT_NU = 5.0
ALPHAS = (0.25, 0.5, 0.75)
DEFAULT_ROW_COUNTS = (8192, 16384, 32768)
DEFAULT_SCRAMBLES = 4
DEFAULT_CALIBRATION_SCRAMBLES = 2
QMC_HALF_WIDTH_LIMIT = 0.00125
QMC_T_CRITICAL_95 = 3.182446305284263
TANGENT_STEP = 1.0e-6
TANGENT_TOLERANCE = 2.0e-6
SMALL_CONVENTION_ROWS = 32
SMALL_CONVENTION_SEED = (20260902, 201)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--row-counts",
        default=",".join(str(value) for value in DEFAULT_ROW_COUNTS),
        help="comma-separated rows per proposal component",
    )
    parser.add_argument("--scrambles", type=int, default=DEFAULT_SCRAMBLES)
    parser.add_argument(
        "--calibration-scrambles",
        type=int,
        default=DEFAULT_CALIBRATION_SCRAMBLES,
    )
    parser.add_argument(
        "--jit-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("row counts must be positive")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("row counts must be strictly increasing")
    return counts


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value {value!r}")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _workspace_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256(status + tracked_diff)
    untracked = []
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        relative = entry[3:].decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if path.is_file():
            file_hash = _sha256_file(path)
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(file_hash.encode("ascii"))
            untracked.append({"path": relative, "sha256": file_hash})
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "workspace_state_sha256": digest.hexdigest(),
        "untracked_inputs": untracked,
    }


def _load_snapshot(step: int):
    snapshot_dir = SNAPSHOT_ROOT / f"t{step:02d}"
    metadata_path = snapshot_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensors = {}
    for name, descriptor in metadata["tensor_files"].items():
        relative = Path(str(descriptor["path"]))
        path = SNAPSHOT_ATTEMPT / relative
        actual_hash = _sha256_file(path)
        if actual_hash != str(descriptor["sha256"]):
            raise ValueError(f"snapshot tensor hash mismatch: {path}")
        tensor = tf.io.parse_tensor(tf.io.read_file(str(path)), out_type=DTYPE)
        tensors[name] = tensor
    snapshot = snapshot_api.gaussian_xla_frozen_snapshot_from_parts(metadata, tensors)
    fingerprint = snapshot_api.gaussian_xla_frozen_snapshot_fingerprint(snapshot)
    if fingerprint != str(metadata["snapshot_fingerprint"]):
        raise ValueError(f"snapshot fingerprint mismatch at t={step}")
    return snapshot, {
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": _sha256_file(metadata_path),
        "fingerprint": fingerprint,
        "tensor_hashes": {
            name: str(descriptor["sha256"])
            for name, descriptor in metadata["tensor_files"].items()
        },
    }


def _sobol_uniform(count: int, dimension: int, seed: tuple[int, int]) -> tf.Tensor:
    base = tf.math.sobol_sample(dimension, count, dtype=DTYPE)
    shift = tf.random.stateless_uniform(
        [1, dimension], tf.constant(seed, tf.int32), dtype=DTYPE
    )
    epsilon = tf.constant(1.0e-14, DTYPE)
    return tf.clip_by_value(tf.math.floormod(base + shift, 1.0), epsilon, 1.0 - epsilon)


def _proposal_bank(
    count: int, seed: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    uniform = _sobol_uniform(count, JOINT_DIM, seed)
    normal = tfp.distributions.Normal(
        tf.constant(0.0, DTYPE), tf.constant(1.0, DTYPE)
    )
    student = tfp.distributions.StudentT(
        df=tf.constant(STUDENT_NU, DTYPE),
        loc=tf.constant(0.0, DTYPE),
        scale=tf.constant(1.0, DTYPE),
    )
    standard_rows = normal.quantile(uniform)
    student_rows = student.quantile(uniform)
    return standard_rows, student_rows, uniform


def _log_student(rows: tf.Tensor) -> tf.Tensor:
    student = tfp.distributions.StudentT(
        df=tf.constant(STUDENT_NU, DTYPE),
        loc=tf.constant(0.0, DTYPE),
        scale=tf.constant(1.0, DTYPE),
    )
    return tf.reduce_sum(student.log_prob(rows), axis=1)


def _row_shells(
    rows: tf.Tensor,
    target: tf.Tensor,
    control: tf.Tensor,
    proposal_log: tf.Tensor,
    masses: tf.Tensor,
) -> list[dict[str, float | int]]:
    row_radius = tf.reduce_max(tf.abs(rows), axis=1)
    inverse = tf.exp(-proposal_log)
    target_weight = masses * target * inverse
    residual_weight = masses * (target - control) * inverse
    target_energy = tf.reduce_sum(target_weight)
    residual_energy = tf.reduce_sum(tf.square(residual_weight))
    output = []
    for label, mask in (
        ("[0,2]", row_radius <= 2.0),
        ("(2,4]", tf.logical_and(row_radius > 2.0, row_radius <= 4.0)),
        ("(4,inf)", row_radius > 4.0),
    ):
        mask_f = tf.cast(mask, DTYPE)
        shell_target = tf.reduce_sum(target_weight * mask_f)
        shell_residual = tf.reduce_sum(
            tf.square(residual_weight) * mask_f
        )
        output.append(
            {
                "shell": label,
                "row_count": int(tf.reduce_sum(tf.cast(mask, tf.int64)).numpy()),
                "target_mass": float(shell_target.numpy()),
                "target_mass_fraction": float(
                    (shell_target / tf.maximum(target_energy, 1.0e-300)).numpy()
                ),
                "residual_second_moment": float(shell_residual.numpy()),
                "residual_second_moment_fraction": float(
                    (shell_residual / tf.maximum(residual_energy, 1.0e-300)).numpy()
                ),
            }
        )
    return output


def _stats(values: Sequence[float]) -> dict[str, object]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    se = sd / math.sqrt(len(numbers))
    half = QMC_T_CRITICAL_95 * se
    return {
        "n": len(numbers),
        "values": numbers,
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "half_width_95": half,
        "ci95": [mean - half, mean + half],
    }


def _independent_energy_check(snapshot, adapter, rows: tf.Tensor) -> tf.Tensor:
    """Reassemble the branch target without calling the snapshot evaluator."""

    n = snapshot.state_dim
    current_basis = _hermite_product_basis(n, snapshot.basis_degree)
    prefix_shapes = tuple(tuple(value.shape.as_list()) for value in snapshot.prefix_values)
    prefix_cores = tuple(
        TTCore(tf.reshape(value, shape))
        for value, shape in zip(snapshot.prefix_values, prefix_shapes)
    )
    gram = tf.convert_to_tensor(snapshot.suffix_gram, DTYPE)
    floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
    chol = tf.linalg.cholesky(
        gram
        + tf.constant(snapshot.branch_gram_floor, DTYPE)
        * floor_scale
        * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
    )
    u_c = rows[:, :n]
    u_p = rows[:, n:]
    x_current = snapshot.joint_mean[:n][None, :] + tf.einsum(
        "ij,nj->ni", snapshot.joint_chol[:n, :n], u_c
    )
    x_previous = (
        snapshot.joint_mean[n:][None, :]
        + tf.einsum("ij,nj->ni", snapshot.joint_chol[n:, :n], u_c)
        + tf.einsum("ij,nj->ni", snapshot.joint_chol[n:, n:], u_p)
    )
    u_old = tf.transpose(
        tf.linalg.triangular_solve(
            snapshot.old_coordinate_matrix,
            tf.transpose(x_previous - snapshot.old_coordinate_offset[None, :]),
            lower=True,
        )
    )
    logdet_c = tf.reduce_sum(
        tf.math.log(tf.abs(tf.linalg.diag_part(snapshot.joint_chol[:n, :n])))
    )
    logdet_p = tf.reduce_sum(
        tf.math.log(tf.abs(tf.linalg.diag_part(snapshot.joint_chol[n:, n:])))
    )
    logdet_old = tf.reduce_sum(
        tf.math.log(tf.abs(tf.linalg.diag_part(snapshot.old_coordinate_matrix)))
    )
    conversion = (
        logdet_c
        + logdet_p
        - logdet_old
        + _log_eta(u_old)
        - _log_eta(rows)
    )
    log_g = (
        adapter.transition_log_density(x_current, x_previous)
        + adapter.observation_log_density(x_current, snapshot.observation)
        + conversion
    )
    v_prev = tf.einsum(
        "na,ab->nb",
        prefix_row_vectors(prefix_cores, current_basis, u_old),
        chol,
    )
    if snapshot.defensive_nu is None:
        floor_values = snapshot.tau_abs_previous * tf.ones(
            [tf.shape(rows)[0]], DTYPE
        )
    else:
        floor_values = snapshot.tau_abs_previous * tf.exp(
            _log_student_t_ratio(u_old, snapshot.defensive_nu)
        )
    sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + floor_values
    return tf.exp(tf.math.log(sum_sq) + log_g - snapshot.frozen_shift)


def _evaluate_rows(
    snapshot,
    adapter,
    rows: tf.Tensor,
    *,
    jit_compile: bool,
) -> dict[str, tf.Tensor]:
    weights = tf.fill(
        [tf.shape(rows)[0]],
        tf.cast(1.0, DTYPE) / tf.cast(tf.shape(rows)[0], DTYPE),
    )
    return snapshot_api.evaluate_gaussian_xla_frozen_transition(
        snapshot, adapter, rows, weights, jit_compile=jit_compile
    )


def _one_bank_record(
    snapshot,
    adapter,
    standard_rows: tf.Tensor,
    student_rows: tf.Tensor,
    *,
    alpha: float,
    jit_compile: bool,
    tangent_check: bool,
) -> dict[str, object]:
    count = int(standard_rows.shape[0])
    rows = tf.concat([standard_rows, student_rows], axis=0)
    evaluation = _evaluate_rows(snapshot, adapter, rows, jit_compile=jit_compile)
    energy = evaluation["row_target_energy"]
    prediction = evaluation["row_prediction_energy"]
    eta_log = _log_eta(rows)
    student_log = _log_student(rows)
    component_logs = tf.stack([eta_log, student_log], axis=1)
    component_weights = tf.constant([1.0 - alpha, alpha], DTYPE)
    base_masses = tf.concat(
        [
            tf.fill([count], tf.constant((1.0 - alpha) / count, DTYPE)),
            tf.fill([count], tf.constant(alpha / count, DTYPE)),
        ],
        axis=0,
    )
    eta = tf.exp(eta_log)
    target = eta * energy
    control = eta * prediction
    value = frozen_importance_estimate(
        target,
        component_logs,
        component_weights,
        base_masses,
        control_values=control,
        control_normalizer=snapshot.z_h,
    )
    plain = frozen_importance_estimate(
        target, component_logs, component_weights, base_masses
    )
    log_q = complete_mixture_log_density(component_logs, component_weights)
    tangent_payload: dict[str, object] = {
        "checked": False,
        "maximum_abs_error": None,
        "tolerance": TANGENT_TOLERANCE,
    }
    if tangent_check:
        zeros_component = tf.zeros([2 * count, 2, 1], DTYPE)
        tangent = frozen_importance_directional_estimate(
            target,
            component_logs,
            component_weights,
            base_masses,
            target[:, None],
            zeros_component,
            control_values=control,
            control_normalizer=snapshot.z_h,
            control_tangent=tf.zeros([2 * count, 1], DTYPE),
            control_normalizer_tangent=tf.zeros([1], DTYPE),
        )
        step = tf.constant(TANGENT_STEP, DTYPE)
        plus = frozen_importance_estimate(
            target * tf.exp(step),
            component_logs,
            component_weights,
            base_masses,
            control_values=control,
            control_normalizer=snapshot.z_h,
        )["normalizer"]
        minus = frozen_importance_estimate(
            target * tf.exp(-step),
            component_logs,
            component_weights,
            base_masses,
            control_values=control,
            control_normalizer=snapshot.z_h,
        )["normalizer"]
        fd = (plus - minus) / (2.0 * step)
        tangent_error = tf.abs(tangent["normalizer_tangent"][0] - fd)
        tangent_payload = {
            "checked": True,
            "maximum_abs_error": float(tangent_error.numpy()),
            "finite_difference": float(fd.numpy()),
            "explicit_tangent": float(
                tangent["normalizer_tangent"][0].numpy()
            ),
            "valid": bool(tangent["tangent_valid"].numpy()),
            "tolerance": TANGENT_TOLERANCE,
        }
        if not bool(tangent["tangent_valid"].numpy()) or float(
            tangent_error.numpy()
        ) > TANGENT_TOLERANCE:
            raise RuntimeError("frozen DMIS tangent parity failed")
    if not bool(value["valid"].numpy()) or not bool(plain["dmis_valid"].numpy()):
        raise RuntimeError("frozen DMIS/CV validity flag failed")
    return {
        "alpha": alpha,
        "row_count_per_component": count,
        "normalizer": float(value["normalizer"].numpy()),
        "log_normalizer": float(value["log_normalizer"].numpy()),
        "cv_normalizer": float(value["normalizer"].numpy()),
        "cv_log_normalizer": float(value["log_normalizer"].numpy()),
        "plain_normalizer": float(plain["normalizer"].numpy()),
        "plain_log_normalizer": float(plain["log_normalizer"].numpy()),
        "gram_normalizer": float(snapshot.z_h.numpy()),
        "gram_log_normalizer": float(tf.math.log(snapshot.z_h).numpy()),
        "target_ess": float(value["target_effective_sample_size"].numpy()),
        "target_ess_fraction": float(
            value["target_effective_sample_size_fraction"].numpy()
        ),
        "maximum_normalized_target_weight": float(
            value["maximum_normalized_target_weight"].numpy()
        ),
        "residual_second_moment": float(
            value["residual_second_moment"].numpy()
        ),
        "target_support_valid": bool(value["target_support_valid"].numpy()),
        "residual_support_valid": bool(value["residual_support_valid"].numpy()),
        "complete_mixture_recomposed": bool(
            tf.reduce_all(tf.math.is_finite(log_q)).numpy()
        ),
        "target_all_finite": bool(
            evaluation["target_all_finite"].numpy()
        ),
        "branch_closure_relative_max": float(
            evaluation["target_branch_closure_relative_max"].numpy()
        ),
        "shells": _row_shells(
            rows, target, control, log_q, base_masses
        ),
        "tangent": tangent_payload,
    }


def _convention_check(snapshot, adapter, *, jit_compile: bool) -> dict[str, object]:
    standard_rows, student_rows, _ = _proposal_bank(
        SMALL_CONVENTION_ROWS, SMALL_CONVENTION_SEED
    )
    rows = standard_rows
    evaluation = _evaluate_rows(snapshot, adapter, rows, jit_compile=jit_compile)
    energy = evaluation["row_target_energy"]
    eta_log = _log_eta(rows)
    eta = tf.exp(eta_log)
    masses = tf.fill(
        [SMALL_CONVENTION_ROWS],
        tf.constant(1.0 / SMALL_CONVENTION_ROWS, DTYPE),
    )
    one_component = frozen_importance_estimate(
        eta * energy,
        eta_log[:, None],
        tf.constant([1.0], DTYPE),
        masses,
    )
    direct_eta = tf.reduce_mean(energy)
    standard_error = tf.abs(one_component["normalizer"] - direct_eta)
    independent = _independent_energy_check(snapshot, adapter, rows)
    independent_error = tf.reduce_max(tf.abs(independent - energy))
    if float(standard_error.numpy()) > 2.0e-12:
        raise RuntimeError("standard-normal gamma convention check failed")
    if float(independent_error.numpy()) > 2.0e-10:
        raise RuntimeError("independent target assembly check failed")
    return {
        "rows": SMALL_CONVENTION_ROWS,
        "direct_eta_mean": float(direct_eta.numpy()),
        "generic_one_component": float(one_component["normalizer"].numpy()),
        "generic_minus_direct_abs": float(standard_error.numpy()),
        "independent_target_max_abs": float(independent_error.numpy()),
        "target_all_finite": bool(evaluation["target_all_finite"].numpy()),
        "passed": True,
    }


def _calibrate_alpha(
    snapshots: Mapping[int, object],
    adapter,
    *,
    count: int,
    scrambles: int,
    jit_compile: bool,
) -> tuple[float, dict[str, object]]:
    records = {alpha: {step: [] for step in CAPTURE_STEPS} for alpha in ALPHAS}
    for scramble in range(scrambles):
        for step in CAPTURE_STEPS:
            seed = (20260902 + step, 310000 + 100 * step + scramble)
            standard, student, _ = _proposal_bank(count, seed)
            for alpha in ALPHAS:
                record = _one_bank_record(
                    snapshots[step],
                    adapter,
                    standard,
                    student,
                    alpha=alpha,
                    jit_compile=jit_compile,
                    tangent_check=False,
                )
                records[alpha][step].append(record)
    summary = {}
    for alpha in ALPHAS:
        by_step = {}
        maximum_half_width = 0.0
        for step in CAPTURE_STEPS:
            stats = _stats(
                [item["log_normalizer"] for item in records[alpha][step]]
            )
            by_step[str(step)] = stats
            maximum_half_width = max(
                maximum_half_width, float(stats["half_width_95"])
            )
        summary[str(alpha)] = {
            "per_step": by_step,
            "maximum_half_width_95": maximum_half_width,
        }
    selected = min(
        ALPHAS,
        key=lambda alpha: (
            float(summary[str(alpha)]["maximum_half_width_95"]),
            abs(alpha - 0.5),
        ),
    )
    return selected, {
        "calibration_row_count_per_component": count,
        "calibration_scrambles": scrambles,
        "candidates": summary,
        "selected_alpha_student": selected,
    }


def _run_claim_ladder(
    snapshots: Mapping[int, object],
    adapter,
    *,
    row_counts: Sequence[int],
    scrambles: int,
    alpha: float,
    jit_compile: bool,
) -> dict[str, object]:
    records: dict[str, list[dict[str, object]]] = {}
    for count in row_counts:
        count_records = []
        for scramble in range(scrambles):
            for step in CAPTURE_STEPS:
                seed = (20260902 + 10 * step, 410000 + 100 * step + scramble)
                standard, student, _ = _proposal_bank(count, seed)
                record = _one_bank_record(
                    snapshots[step],
                    adapter,
                    standard,
                    student,
                    alpha=alpha,
                    jit_compile=jit_compile,
                    tangent_check=(scramble == 0),
                )
                record.update(
                    {
                        "time_index": step,
                        "scramble": scramble,
                        "seed": list(seed),
                    }
                )
                count_records.append(record)
        records[str(count)] = count_records
    summaries = {}
    precision_pass = True
    for count in row_counts:
        by_step = {}
        for step in CAPTURE_STEPS:
            rows = [
                item
                for item in records[str(count)]
                if int(item["time_index"]) == step
            ]
            log_stats = _stats([item["log_normalizer"] for item in rows])
            plain_stats = _stats([item["plain_log_normalizer"] for item in rows])
            cv_stats = _stats([item["normalizer"] for item in rows])
            half = float(log_stats["half_width_95"])
            precision_pass = precision_pass and half <= QMC_HALF_WIDTH_LIMIT
            by_step[str(step)] = {
                "log_normalizer": log_stats,
                "cv_log_normalizer": log_stats,
                "plain_log_normalizer": plain_stats,
                "cv_normalizer": cv_stats,
                "gram_log_normalizer": float(rows[0]["gram_log_normalizer"]),
                "normalizer_gap_mean": float(
                    statistics.fmean(
                        [
                            float(item["log_normalizer"])
                            - float(item["gram_log_normalizer"])
                            for item in rows
                        ]
                    )
                ),
                "mean_target_ess_fraction": float(
                    statistics.fmean(
                        [float(item["target_ess_fraction"]) for item in rows]
                    )
                ),
                "mean_residual_second_moment": float(
                    statistics.fmean(
                        [float(item["residual_second_moment"]) for item in rows]
                    )
                ),
                "all_valid": all(bool(item["target_support_valid"]) for item in rows),
                "maximum_tangent_error": max(
                    float(item["tangent"]["maximum_abs_error"] or 0.0)
                    for item in rows
                    if item["tangent"]["checked"]
                ),
            }
        summaries[str(count)] = by_step
    return {
        "selected_alpha_student": alpha,
        "row_counts": list(row_counts),
        "scrambles": scrambles,
        "records": records,
        "summaries": summaries,
        "precision_pass": precision_pass,
        "precision_half_width_limit": QMC_HALF_WIDTH_LIMIT,
    }


def _result_markdown(result: Mapping[str, object]) -> str:
    calibration = result["calibration"]
    claim = result["claim_ladder"]
    lines = [
        "# C2 Phase 2 Generic-DMIS Frozen Integration",
        "",
        f"Status: `{result['status']}` (diagnostic; no production claim).",
        "",
        "The harness loads the preserved t=2,3,4 frozen snapshots and supplies "
        "the exact branch-summed target to the model-independent complete-DMIS "
        "kernel. The target is `gamma=eta*E` in the whitened coordinates; the "
        "stored TT Gram is used only as a known-integral control variate.",
        "",
        f"Selected Student mixture weight (calibration): "
        f"`{calibration['selected_alpha_student']}`",
        "",
        "## Convention Check",
        "",
    ]
    for step, check in result["convention_checks"].items():
        lines.append(
            f"- t={step}: generic/direct absolute difference "
            f"`{check['generic_minus_direct_abs']:.3e}`, independent target "
            f"maximum difference `{check['independent_target_max_abs']:.3e}`"
        )
    lines.extend(
        [
            "",
            "## Precision Ladder",
            "",
            "| rows/component | t | CV log Z mean +/- 95% half-width | plain DMIS log Z mean +/- 95% half-width | CV Z mean +/- 95% half-width | Gram log Z | ESS fraction |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for count in claim["row_counts"]:
        for step in CAPTURE_STEPS:
            summary = claim["summaries"][str(count)][str(step)]
            log_stats = summary["cv_log_normalizer"]
            cv_stats = summary["cv_normalizer"]
            plain_stats = summary["plain_log_normalizer"]
            lines.append(
                f"| {count} | {step} | {log_stats['mean']:+.8f} +/- "
                f"{log_stats['half_width_95']:.3g} | "
                f"{plain_stats['mean']:+.8f} +/- "
                f"{plain_stats['half_width_95']:.3g} | "
                f"{cv_stats['mean']:+.8f} +/- "
                f"{cv_stats['half_width_95']:.3g} | "
                f"{summary['gram_log_normalizer']:+.8f} | "
                f"{summary['mean_target_ess_fraction']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Precision screen: `{'PASS' if claim['precision_pass'] else 'FAIL'}` "
            f"(limit `{QMC_HALF_WIDTH_LIMIT}`).",
            "- `CV log Z` is the known-integral squared-TT control-variate estimate; "
            "`plain DMIS log Z` is the uncorrected complete-mixture estimate.",
            "- The convention and tangent checks are engineering diagnostics for "
            "the frozen finite program.",
            "- A precision failure does not classify the TT fit, state recursion, "
            "or basis as causal.",
            "- Recursive moment-map testing is permitted only after this screen "
            "passes and requires a separate executable callable.",
            "",
            "Full provenance is in `result.json` and `run_manifest.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.parent != DEFAULT_OUTPUT_PARENT.resolve():
        raise ValueError(
            f"output root must be a direct child of {DEFAULT_OUTPUT_PARENT}"
        )
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    row_counts = _parse_counts(args.row_counts)
    if args.scrambles < 2 or args.calibration_scrambles < 2:
        raise ValueError("at least two scrambles are required for uncertainty")
    started = time.perf_counter()
    snapshots = {}
    snapshot_records = {}
    for step in CAPTURE_STEPS:
        snapshots[step], snapshot_records[str(step)] = _load_snapshot(step)

    memory_policy = EARLY_MEMORY_POLICY
    cpu_only = CPU_ONLY
    logical_gpus = ()
    if not cpu_only:
        logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
        if not logical_gpus:
            raise RuntimeError("serious run requires a visible TensorFlow GPU")
        with tf.device("/GPU:0"):
            placement_probe = tf.reduce_sum(tf.ones([32], DTYPE))
        if "GPU" not in placement_probe.device.upper():
            raise RuntimeError("placement probe did not execute on GPU")
        placement_device = placement_probe.device
    else:
        placement_device = "/CPU:0"

    model = sv.sv_model(N, MODEL_SEED)
    adapter = sv.sv_adapter(model)
    convention_checks = {
        str(step): _convention_check(
            snapshots[step], adapter, jit_compile=bool(args.jit_compile)
        )
        for step in CAPTURE_STEPS
    }
    selected_alpha, calibration = _calibrate_alpha(
        snapshots,
        adapter,
        count=row_counts[0],
        scrambles=int(args.calibration_scrambles),
        jit_compile=bool(args.jit_compile),
    )
    claim_ladder = _run_claim_ladder(
        snapshots,
        adapter,
        row_counts=row_counts,
        scrambles=int(args.scrambles),
        alpha=selected_alpha,
        jit_compile=bool(args.jit_compile),
    )
    status = (
        "PHASE2_DMIS_PRECISION_REPAIRED"
        if claim_ladder["precision_pass"]
        else "PHASE2_INTEGRATION_UNRESOLVED"
    )
    script_path = Path(__file__).resolve()
    result = {
        "schema_id": "c2_phase2_generic_dmis_repair_result_v1",
        "status": status,
        "classification": "diagnostic_holdout_only",
        "scientific_claims": [],
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "target_definition": "gamma_t(u)=eta_2n(u)*E_t(u)",
        "target_boundary": "finite carried-density target, not true C2 likelihood",
        "model_id": "zc24_sv_vector_extension_v1",
        "model_seed": MODEL_SEED,
        "observation_seed": OBS_SEED,
        "capture_steps": list(CAPTURE_STEPS),
        "snapshot_records": snapshot_records,
        "convention_checks": convention_checks,
        "calibration": calibration,
        "claim_ladder": claim_ladder,
        "recursive_stage": {
            "executed": False,
            "reason": (
                "gated on frozen precision screen and requires an executable "
                "recursive moment-map callable"
            ),
        },
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "python_version": platform.python_version(),
        "dtype": DTYPE.name,
        "jit_compile": bool(args.jit_compile),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "cpu_only": cpu_only,
        "logical_gpus": [str(device) for device in logical_gpus],
        "placement_probe_device": placement_device,
        "memory_policy": memory_policy,
        "student_nu": STUDENT_NU,
        "alpha_candidates": list(ALPHAS),
        "command": [sys.executable, *sys.argv],
        "script_sha256": _sha256_file(script_path),
        "module_sha256": _sha256_file(
            ROOT / "bayesfilter/highdim/frozen_dmis_control_variate_tf.py"
        ),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "result.json", result)
    (output_root / "result.md").write_text(
        _result_markdown(result), encoding="utf-8"
    )
    manifest = {
        "schema_id": "c2_phase2_generic_dmis_repair_manifest_v1",
        "plan": "docs/plans/bayesfilter-c2-phase2-generic-dmis-recursive-repair-plan-20260902.md",
        "plan_review": "docs/plans/bayesfilter-c2-phase2-generic-dmis-recursive-repair-plan-review-20260902.md",
        "classification": "diagnostic_holdout_only",
        "trust_basis": TRUST_BASIS if not cpu_only else "cpu_only_reference_diagnostic",
        "workspace": _workspace_state(),
        "command": [sys.executable, *sys.argv],
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "result_sha256": _sha256_file(output_root / "result.json"),
        "result_markdown_sha256": _sha256_file(output_root / "result.md"),
        "snapshot_attempt": str(SNAPSHOT_ATTEMPT.relative_to(ROOT)),
        "snapshot_fingerprints": {
            step: record["fingerprint"]
            for step, record in snapshot_records.items()
        },
        "attempt_budget": {
            "row_counts": list(row_counts),
            "scrambles": int(args.scrambles),
            "calibration_scrambles": int(args.calibration_scrambles),
            "maximum_gpu_hours": 6,
            "maximum_cpu_hours": 4,
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "selected_alpha": selected_alpha}, indent=2))


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
