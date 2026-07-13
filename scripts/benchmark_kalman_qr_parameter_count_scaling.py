#!/usr/bin/env python
"""Benchmark QR score timing as parameter dimension varies.

The timed kernels are TensorFlow QR/square-root Kalman computations. Fixture
construction is outside the timed region. Transition and observation matrices
are lower triangular; covariance parameters are assigned to lower-triangular
covariance factors and then converted to SPD covariance matrices for the public
Kalman API.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import kalman_qr_benchmark_contract as benchmark_contract


PHASE6_IMPORT_DISCOVERY_PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
PHASE6_IMPORT_DISCOVERY_SCRIPT = (
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
)
PHASE6_IMPORT_DISCOVERY_OUTPUT = (
    "/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/import_discovery.json"
)
PHASE6_IMPORT_DISCOVERY_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}
PHASE6_IMPORT_DISCOVERY_ARGUMENTS = [
    "--phase6-import-discovery",
    "--device",
    "cpu",
    "--cpu-threads",
    "1",
    "--output-json",
    PHASE6_IMPORT_DISCOVERY_OUTPUT,
]


def _phase6_import_discovery_preimport_guard() -> bool:
    """Fail closed before TensorFlow import when discovery authority is present."""
    if "--phase6-import-discovery" not in sys.argv[1:]:
        return False
    expected_argv = [PHASE6_IMPORT_DISCOVERY_SCRIPT, *PHASE6_IMPORT_DISCOVERY_ARGUMENTS]
    problems: list[str] = []
    if sys.executable != PHASE6_IMPORT_DISCOVERY_PYTHON:
        problems.append("interpreter")
    if sys.argv != expected_argv:
        problems.append("argv")
    expected_script = Path(__file__).resolve().parents[1] / PHASE6_IMPORT_DISCOVERY_SCRIPT
    try:
        invoked_script = (Path.cwd() / sys.argv[0]).resolve(strict=True)
    except OSError:
        invoked_script = None
    if invoked_script != expected_script.resolve(strict=True):
        problems.append("script")
    if {
        name: os.environ.get(name)
        for name in PHASE6_IMPORT_DISCOVERY_ENVIRONMENT
    } != PHASE6_IMPORT_DISCOVERY_ENVIRONMENT:
        problems.append("environment")
    if problems:
        raise SystemExit(
            "Phase 6 import discovery rejected before import: "
            + ", ".join(problems)
        )
    return True


def _phase6_preimport_exact_process_argv() -> list[str]:
    raw = getattr(sys, "orig_argv", None)
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or any(not isinstance(value, str) for value in raw)
        or raw[0] != PHASE6_IMPORT_DISCOVERY_PYTHON
        or raw[2:] != sys.argv[1:]
    ):
        raise ValueError("Phase 6 child cannot establish its exact process argv")
    invoked_script = Path(raw[1])
    if not invoked_script.is_absolute():
        invoked_script = Path.cwd() / invoked_script
    try:
        script_matches = invoked_script.resolve(strict=True) == Path(__file__).resolve(
            strict=True
        )
    except OSError:
        script_matches = False
    if not script_matches:
        raise ValueError("Phase 6 child process argv names a different script")
    return list(raw)


def _phase6_write_authority_failure(
    snapshot: Mapping[str, Any],
    *,
    command_argv: Sequence[str],
    error: BaseException,
) -> None:
    """Write only to child paths already bound by a valid reviewed snapshot."""

    if not benchmark_contract.phase6_child_authority_snapshot_valid(snapshot):
        return
    row = snapshot["schedule_row"]
    if row.get("child_command_argv") != list(command_argv):
        return
    identity = row["identity"]
    paths = benchmark_contract._phase6_child_artifact_paths(identity)
    options = benchmark_contract._phase6_command_options(command_argv)
    if (
        not isinstance(options, Mapping)
        or options.get("--output-json") != [str(paths["artifact"])]
        or options.get("--progress-journal") != [str(paths["journal"])]
    ):
        return
    now_ns = time.perf_counter_ns()
    failure = {
        "schema": benchmark_contract.PHASE6_CHILD_AUTHORITY_FAILURE_SCHEMA,
        "state": "failed",
        "stage": "child_entry_authority_guard",
        "identity": identity,
        "case_id": row["case_id"],
        "attempt_id": row["attempt_id"],
        **row["fingerprints"],
        "resume_key": row["resume_key"],
        "command_argv": list(command_argv),
        "started_ns": now_ns,
        "finished_ns": now_ns,
        "elapsed_seconds": 0.0,
        "target_work": {
            "tensorflow_imported": False,
            "fixture_constructed": False,
            "selected_method_constructed": False,
            "trace_requested": False,
            "xla_requested": False,
            "kalman_invocations": 0,
        },
        "error": {"type": type(error).__name__, "message": str(error)},
        "nonclaims": list(benchmark_contract.PHASE6_NONCLAIMS),
    }
    event = {
        "schema": benchmark_contract.PHASE6_CHILD_AUTHORITY_FAILURE_SCHEMA,
        "attempt_id": row["attempt_id"],
        "case_id": row["case_id"],
        "method_id": identity["method_id"],
        "stage": "child_entry_authority_guard",
        "resume_key": row["resume_key"],
        **row["fingerprints"],
    }
    benchmark_contract.durable_atomic_write_text(
        paths["journal"], benchmark_contract.strict_json_dumps(event) + "\n"
    )
    benchmark_contract.durable_atomic_write_json(paths["artifact"], failure)


def _phase6_child_authority_preimport_guard() -> Mapping[str, Any] | None:
    if "--phase6-authority-snapshot" not in sys.argv[1:]:
        return None
    command_argv: list[str] = []
    snapshot: Any = None
    try:
        command_argv = _phase6_preimport_exact_process_argv()
        indices = [
            index
            for index, value in enumerate(command_argv)
            if value == "--phase6-authority-snapshot"
        ]
        if len(indices) != 1 or indices[0] + 1 >= len(command_argv):
            raise ValueError("Phase 6 child requires one authority snapshot path")
        snapshot_path = Path(command_argv[indices[0] + 1])
        snapshot = benchmark_contract.read_strict_json(snapshot_path)
        if not benchmark_contract.phase6_child_authority_snapshot_valid(snapshot):
            raise ValueError("Phase 6 child authority snapshot is invalid")
        expected_sha256 = os.environ.get(
            benchmark_contract.PHASE6_CHILD_AUTHORITY_SHA256_ENV
        )
        if (
            expected_sha256 is None
            or {
                name: os.environ.get(name)
                for name in benchmark_contract.PHASE6_ENVIRONMENT
            }
            != benchmark_contract.PHASE6_ENVIRONMENT
            or benchmark_contract.file_sha256(snapshot_path) != expected_sha256
            or benchmark_contract.durable_json_sha256(snapshot) != expected_sha256
        ):
            raise ValueError("Phase 6 child authority environment or digest mismatch")
        benchmark_contract.validate_phase6_child_authority_snapshot(
            snapshot,
            repo_root=REPO_ROOT,
            command_argv=command_argv,
        )
        return snapshot
    except Exception as exc:
        if isinstance(snapshot, Mapping) and command_argv:
            _phase6_write_authority_failure(
                snapshot, command_argv=command_argv, error=exc
            )
        raise SystemExit(0) from exc


PHASE6_IMPORT_DISCOVERY_PREIMPORT_VALIDATED = (
    _phase6_import_discovery_preimport_guard()
)
PHASE6_CHILD_AUTHORITY_CAPTURE = _phase6_child_authority_preimport_guard()

# Deliberate CPU debug/reference runs must hide GPU devices before TensorFlow
# import so a CPU artifact does not accidentally probe CUDA.
if any(argument == "--device=cpu" for argument in sys.argv) or (
    "--device" in sys.argv
    and sys.argv[sys.argv.index("--device") + 1 : sys.argv.index("--device") + 2]
    == ["cpu"]
):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

if "--cpu-threads" in sys.argv:
    value = sys.argv[sys.argv.index("--cpu-threads") + 1 : sys.argv.index("--cpu-threads") + 2]
    if value:
        os.environ.setdefault("OMP_NUM_THREADS", value[0])
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", value[0])
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", value[0])

import tensorflow as tf

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_sqrt_kalman_score,
    tf_qr_sqrt_kalman_score_batched_static,
)
from bayesfilter.linear.kalman_qr_tf import tf_qr_sqrt_kalman_log_likelihood_while_loop
SUPPORTED_DTYPES = {
    "float32": tf.float32,
    "float64": tf.float64,
}
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-subplan-2026-07-09.md"
)
DEFAULT_JSON = (
    "docs/benchmarks/"
    "kalman_qr_parameter_count_scaling_2026-07-09.json"
)
DEFAULT_MD = (
    "docs/benchmarks/"
    "kalman_qr_parameter_count_scaling_2026-07-09.md"
)
PHASE2_DIAGNOSTIC_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase2.v1"
PHASE2_PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-subplan-2026-07-11.md"
)
PHASE2_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-result-2026-07-11.md"
)
PHASE2_DIAGNOSTIC_JSON = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase2_graphdef_2026-07-11.json"
)
PHASE3_DIAGNOSTIC_SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase3.v1"
PHASE3_PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase3-parameter-vectorization-subplan-2026-07-11.md"
)
PHASE3_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase3-parameter-vectorization-result-2026-07-11.md"
)
PHASE3_DIAGNOSTIC_JSON = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase3_parameter_graphdef_2026-07-11.json"
)
PHASE4_DIAGNOSTIC_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase4.autodiff.v1"
)
PHASE4_XLA_SMOKE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase4.autodiff_cpu_xla.v1"
)
PHASE4_PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase4-batched-autodiff-subplan-2026-07-11.md"
)
PHASE4_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase4-batched-autodiff-result-2026-07-11.md"
)
PHASE4_DIAGNOSTIC_JSON = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json"
)
PHASE4_XLA_SMOKE_JSON = (
    "docs/benchmarks/"
    "kalman_qr_batched_xla_repair_phase4_autodiff_cpu_xla_smoke_2026-07-11.json"
)
PHASE4_DIAGNOSTIC_LOG = "/tmp/kalman_qr_phase4_autodiff/phase4_autodiff.log"
PHASE4_XLA_SMOKE_LOG = "/tmp/kalman_qr_phase4_autodiff/cpu_xla_smoke.log"
PHASE4_DECLARED_PATHS = (
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py",
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
    "scripts/kalman_qr_benchmark_contract.py",
    "tests/test_kalman_qr_batch_native_autodiff.py",
    "tests/test_kalman_qr_batched_fixture.py",
    "tests/test_kalman_qr_benchmark_contract.py",
    "tests/test_kalman_qr_parameter_count_scaling_harness.py",
    "tests/test_linear_qr_batched_parameter_vectorization_tf.py",
)
PHASE4_TOLERANCES = {
    "float32": {
        "value": {"rtol": 2.0e-4, "atol": 2.0e-4},
        "score": {"rtol": 2.0e-4, "atol": 2.0e-4},
        "off_diagonal_atol": 2.0e-6,
    },
    "float64": {
        "value": {"rtol": 1.0e-10, "atol": 1.0e-10},
        "score": {"rtol": 1.0e-8, "atol": 1.0e-9},
        "off_diagonal_atol": 2.0e-12,
    },
}
PHASE4_NONCLAIMS = (
    "no warm-runtime improvement claim",
    "no CPU or GPU scalability claim",
    "no method ranking claim",
    "no GPU readiness claim",
    "no HMC or posterior correctness claim",
    "no default, production, or scientific validity claim",
)
PHASE6_PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-subplan-2026-07-12.md"
)
PHASE6_TRACE_CHILD_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace_child.v1"
)
PHASE6_IMPORT_DISCOVERY_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.import_discovery.v1"
)
PHASE2_TENSOR_NAMES = (
    "transition_offset",
    "transition_matrix",
    "transition_covariance",
    "observation_offset",
    "observation_matrix",
    "observation_covariance",
    "initial_state_mean",
    "initial_state_covariance",
    "d_initial_state_mean",
    "d_initial_state_covariance",
    "d_transition_offset",
    "d_transition_matrix",
    "d_transition_covariance",
    "d_observation_offset",
    "d_observation_matrix",
    "d_observation_covariance",
)
PARAMETER_COUNTS = [50, 150]
MAX_SCALING_PARAMETER_COUNT = 150
CANONICAL_PROPOSAL_BATCH_SIZE = 16
PROPOSAL_ROW_IDS = {
    1: (7,),
    4: (2, 7, 8, 13),
    16: tuple(range(16)),
}


class ParameterSlot(NamedTuple):
    group: str
    row: int
    col: int


@dataclass(frozen=True)
class Fixture:
    state_dim: int
    observation_dim: int
    timesteps: int
    parameter_count: int
    parameters: tf.Tensor
    observations: tf.Tensor
    base_initial_mean: tf.Tensor
    base_initial_covariance_factor: tf.Tensor
    base_transition_offset: tf.Tensor
    base_transition_matrix: tf.Tensor
    base_transition_covariance_factor: tf.Tensor
    base_observation_offset: tf.Tensor
    base_observation_matrix: tf.Tensor
    base_observation_covariance_factor: tf.Tensor
    d_initial_mean: tf.Tensor
    d_initial_covariance_factor: tf.Tensor
    d_transition_offset: tf.Tensor
    d_transition_matrix: tf.Tensor
    d_transition_covariance_factor: tf.Tensor
    d_observation_offset: tf.Tensor
    d_observation_matrix: tf.Tensor
    d_observation_covariance_factor: tf.Tensor
    slot_allocation: dict[str, int]
    parameter_capacity: int
    dtype: tf.DType


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tf.TensorShape):
        return value.as_list()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - manifest best effort.
        return f"{type(exc).__name__}: {exc}"
    text = completed.stdout.strip()
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        return f"returncode={completed.returncode}; stdout={text}; stderr={stderr}"
    return text


def _lower_triangular_slot_count(dim: int) -> int:
    return dim * (dim + 1) // 2


def parameter_capacity(dimension: int) -> int:
    return 5 * _lower_triangular_slot_count(dimension) + 3 * dimension


def _lower_triangular_indices(dimension: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(dimension) for col in range(row + 1)]


def _resolve_dtype(name: str) -> tf.DType:
    try:
        return SUPPORTED_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; expected float32 or float64") from exc


def _make_slots(dimension: int, parameter_count: int) -> list[ParameterSlot]:
    slots: list[ParameterSlot] = []
    matrix_groups = [
        "transition_matrix",
        "observation_matrix",
        "transition_covariance_factor",
        "observation_covariance_factor",
        "initial_covariance_factor",
    ]
    for row, col in _lower_triangular_indices(dimension):
        for group in matrix_groups:
            slots.append(ParameterSlot(group, row, col))
    for group in ("transition_offset", "observation_offset", "initial_mean"):
        for row in range(dimension):
            slots.append(ParameterSlot(group, row, 0))
    return slots[:parameter_count]


def _slot_allocation(slots: list[ParameterSlot]) -> dict[str, int]:
    groups = [
        "transition_matrix",
        "observation_matrix",
        "transition_covariance_factor",
        "observation_covariance_factor",
        "initial_covariance_factor",
        "transition_offset",
        "observation_offset",
        "initial_mean",
    ]
    allocation = {group: 0 for group in groups}
    for slot in slots:
        allocation[slot.group] += 1
    return allocation


def _basis_matrix(
    slots: list[ParameterSlot],
    group: str,
    rows: int,
    cols: int,
    scale: float,
    dtype: tf.DType,
) -> tf.Tensor:
    indices = [[index, slot.row, slot.col] for index, slot in enumerate(slots) if slot.group == group]
    if not indices:
        return tf.zeros([len(slots), rows, cols], dtype=dtype)
    values = [scale] * len(indices)
    return tf.scatter_nd(
        tf.constant(indices, dtype=tf.int64),
        tf.constant(values, dtype=dtype),
        [len(slots), rows, cols],
    )


def _basis_vector(
    slots: list[ParameterSlot],
    group: str,
    size: int,
    scale: float,
    dtype: tf.DType,
) -> tf.Tensor:
    indices = [[index, slot.row] for index, slot in enumerate(slots) if slot.group == group]
    if not indices:
        return tf.zeros([len(slots), size], dtype=dtype)
    values = [scale] * len(indices)
    return tf.scatter_nd(
        tf.constant(indices, dtype=tf.int64),
        tf.constant(values, dtype=dtype),
        [len(slots), size],
    )


def _make_lower_factor(
    dimension: int,
    *,
    diag_start: float,
    offdiag_scale: float,
    dtype: tf.DType,
) -> tf.Tensor:
    row = tf.cast(tf.range(dimension)[:, tf.newaxis], dtype)
    col = tf.cast(tf.range(dimension)[tf.newaxis, :], dtype)
    lower = row >= col
    strict_lower = row > col
    diag = diag_start + 0.01 * tf.cast(tf.range(dimension), dtype)
    offdiag = offdiag_scale * tf.math.sin((row + 1.0) * (col + 1.0) * 0.07)
    return tf.where(
        lower,
        tf.where(strict_lower, offdiag, tf.linalg.diag(diag)),
        tf.zeros([dimension, dimension], dtype=dtype),
    )


def _make_base_transition(dimension: int, *, dtype: tf.DType) -> tf.Tensor:
    indices = tf.range(dimension, dtype=dtype)
    denom = tf.cast(tf.maximum(dimension - 1, 1), dtype)
    diagonal = 0.68 - 0.08 * indices / denom
    base = tf.linalg.diag(diagonal)
    if dimension > 1:
        subdiag = 0.012 * tf.ones([dimension - 1], dtype=dtype)
        base += tf.linalg.diag(subdiag, k=-1)
    return tf.linalg.band_part(base, -1, 0)


def _make_base_observation(dimension: int, *, dtype: tf.DType) -> tf.Tensor:
    row = tf.cast(tf.range(dimension)[:, tf.newaxis] + 1, dtype)
    col = tf.cast(tf.range(dimension)[tf.newaxis, :] + 1, dtype)
    smooth = 0.018 * tf.math.cos(row * col / tf.cast(dimension + 5, dtype))
    return tf.linalg.band_part(tf.eye(dimension, dtype=dtype) + smooth, -1, 0)


def _factor_covariance_and_derivative(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    covariance = factor @ tf.transpose(factor)
    dcovariance = (
        tf.einsum("pij,kj->pik", dfactor, factor)
        + tf.einsum("ij,pkj->pik", factor, dfactor)
    )
    return covariance, dcovariance


def _batched_factor_covariance_and_derivative(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    covariance = factor @ tf.linalg.matrix_transpose(factor)
    dcovariance = (
        tf.einsum("pij,bkj->bpik", dfactor, factor)
        + tf.einsum("bij,pkj->bpik", factor, dfactor)
    )
    return covariance, dcovariance


def _model_tensors(fixture: Fixture, parameters: tf.Tensor) -> tuple[tf.Tensor, ...]:
    params = tf.convert_to_tensor(parameters, dtype=fixture.dtype)
    transition_offset = fixture.base_transition_offset + tf.einsum(
        "p,pi->i",
        params,
        fixture.d_transition_offset,
    )
    transition_matrix = fixture.base_transition_matrix + tf.einsum(
        "p,pij->ij",
        params,
        fixture.d_transition_matrix,
    )
    transition_covariance_factor = fixture.base_transition_covariance_factor + tf.einsum(
        "p,pij->ij",
        params,
        fixture.d_transition_covariance_factor,
    )
    transition_covariance, d_transition_covariance = _factor_covariance_and_derivative(
        transition_covariance_factor,
        fixture.d_transition_covariance_factor,
    )

    observation_offset = fixture.base_observation_offset + tf.einsum(
        "p,pi->i",
        params,
        fixture.d_observation_offset,
    )
    observation_matrix = fixture.base_observation_matrix + tf.einsum(
        "p,pij->ij",
        params,
        fixture.d_observation_matrix,
    )
    observation_covariance_factor = fixture.base_observation_covariance_factor + tf.einsum(
        "p,pij->ij",
        params,
        fixture.d_observation_covariance_factor,
    )
    observation_covariance, d_observation_covariance = _factor_covariance_and_derivative(
        observation_covariance_factor,
        fixture.d_observation_covariance_factor,
    )

    initial_state_mean = fixture.base_initial_mean + tf.einsum(
        "p,pi->i",
        params,
        fixture.d_initial_mean,
    )
    initial_covariance_factor = fixture.base_initial_covariance_factor + tf.einsum(
        "p,pij->ij",
        params,
        fixture.d_initial_covariance_factor,
    )
    initial_state_covariance, d_initial_state_covariance = _factor_covariance_and_derivative(
        initial_covariance_factor,
        fixture.d_initial_covariance_factor,
    )

    return (
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        fixture.d_initial_mean,
        d_initial_state_covariance,
        fixture.d_transition_offset,
        fixture.d_transition_matrix,
        d_transition_covariance,
        fixture.d_observation_offset,
        fixture.d_observation_matrix,
        d_observation_covariance,
    )


def _proposal_row_ids(batch_size: int) -> tuple[int, ...]:
    try:
        return PROPOSAL_ROW_IDS[batch_size]
    except KeyError as exc:
        raise ValueError(
            f"unsupported batch_size={batch_size}; expected one of {sorted(PROPOSAL_ROW_IDS)}"
        ) from exc


def _make_parameter_cloud(fixture: Fixture) -> tf.Tensor:
    row_coordinates = tf.linspace(
        tf.constant(-1.0, dtype=fixture.dtype),
        tf.constant(1.0, dtype=fixture.dtype),
        CANONICAL_PROPOSAL_BATCH_SIZE,
    )
    row_offsets = tf.constant(0.02, dtype=fixture.dtype) * row_coordinates
    parameter_axis = tf.cast(tf.range(fixture.parameter_count) + 1, fixture.dtype)
    pattern = tf.math.sin(parameter_axis * tf.constant(0.37, dtype=fixture.dtype))
    return fixture.parameters[tf.newaxis, :] + row_offsets[:, tf.newaxis] * pattern[tf.newaxis, :]


def _make_parameter_batch(fixture: Fixture, batch_size: int) -> tf.Tensor:
    return tf.gather(
        _make_parameter_cloud(fixture),
        tf.constant(_proposal_row_ids(batch_size), dtype=tf.int32),
        axis=0,
    )


def _broadcast_vector_basis(parameters_batch: tf.Tensor, basis: tf.Tensor) -> tf.Tensor:
    batch_zeros = tf.reduce_sum(parameters_batch[:, :0], axis=1)
    return basis[tf.newaxis, :, :] + batch_zeros[:, tf.newaxis, tf.newaxis]


def _broadcast_matrix_basis(parameters_batch: tf.Tensor, basis: tf.Tensor) -> tf.Tensor:
    batch_zeros = tf.reduce_sum(parameters_batch[:, :0], axis=1)
    return basis[tf.newaxis, :, :, :] + batch_zeros[:, tf.newaxis, tf.newaxis, tf.newaxis]


def _batched_model_tensors(fixture: Fixture, parameters_batch: tf.Tensor) -> tuple[tf.Tensor, ...]:
    params_batch = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
    transition_offset = fixture.base_transition_offset[tf.newaxis, :] + tf.einsum(
        "bp,pi->bi", params_batch, fixture.d_transition_offset
    )
    transition_matrix = fixture.base_transition_matrix[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_transition_matrix
    )
    transition_factor = fixture.base_transition_covariance_factor[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_transition_covariance_factor
    )
    transition_covariance, d_transition_covariance = (
        _batched_factor_covariance_and_derivative(
            transition_factor,
            fixture.d_transition_covariance_factor,
        )
    )

    observation_offset = fixture.base_observation_offset[tf.newaxis, :] + tf.einsum(
        "bp,pi->bi", params_batch, fixture.d_observation_offset
    )
    observation_matrix = fixture.base_observation_matrix[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_observation_matrix
    )
    observation_factor = fixture.base_observation_covariance_factor[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_observation_covariance_factor
    )
    observation_covariance, d_observation_covariance = (
        _batched_factor_covariance_and_derivative(
            observation_factor,
            fixture.d_observation_covariance_factor,
        )
    )

    initial_mean = fixture.base_initial_mean[tf.newaxis, :] + tf.einsum(
        "bp,pi->bi", params_batch, fixture.d_initial_mean
    )
    initial_factor = fixture.base_initial_covariance_factor[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_initial_covariance_factor
    )
    initial_covariance, d_initial_covariance = _batched_factor_covariance_and_derivative(
        initial_factor,
        fixture.d_initial_covariance_factor,
    )

    return (
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_mean,
        initial_covariance,
        _broadcast_vector_basis(params_batch, fixture.d_initial_mean),
        d_initial_covariance,
        _broadcast_vector_basis(params_batch, fixture.d_transition_offset),
        _broadcast_matrix_basis(params_batch, fixture.d_transition_matrix),
        d_transition_covariance,
        _broadcast_vector_basis(params_batch, fixture.d_observation_offset),
        _broadcast_matrix_basis(params_batch, fixture.d_observation_matrix),
        d_observation_covariance,
    )


def _batched_model_value_tensors(
    fixture: Fixture,
    parameters_batch: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    """Build only the model tensors consumed by the autodiff likelihood."""

    params_batch = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
    transition_offset = fixture.base_transition_offset[tf.newaxis, :] + tf.einsum(
        "bp,pi->bi", params_batch, fixture.d_transition_offset
    )
    transition_matrix = fixture.base_transition_matrix[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_transition_matrix
    )
    transition_factor = fixture.base_transition_covariance_factor[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_transition_covariance_factor
    )
    transition_covariance = transition_factor @ tf.linalg.matrix_transpose(
        transition_factor
    )

    observation_offset = fixture.base_observation_offset[tf.newaxis, :] + tf.einsum(
        "bp,pi->bi", params_batch, fixture.d_observation_offset
    )
    observation_matrix = fixture.base_observation_matrix[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_observation_matrix
    )
    observation_factor = fixture.base_observation_covariance_factor[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_observation_covariance_factor
    )
    observation_covariance = observation_factor @ tf.linalg.matrix_transpose(
        observation_factor
    )

    initial_mean = fixture.base_initial_mean[tf.newaxis, :] + tf.einsum(
        "bp,pi->bi", params_batch, fixture.d_initial_mean
    )
    initial_factor = fixture.base_initial_covariance_factor[tf.newaxis, :, :] + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_initial_covariance_factor
    )
    initial_covariance = initial_factor @ tf.linalg.matrix_transpose(initial_factor)

    return (
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_mean,
        initial_covariance,
    )


def _explicit_batched_base(base: tf.Tensor, batch_size: int) -> tf.Tensor:
    shape = base.shape.as_list()
    if any(dimension is None for dimension in shape):
        raise ValueError("explicit batched base requires fully defined fixture shape")
    return tf.broadcast_to(base, [batch_size, *shape])


def _batched_model_value_tensors_explicit(
    fixture: Fixture,
    parameters_batch: tf.Tensor,
    *,
    batch_size: int,
) -> tuple[tf.Tensor, ...]:
    """Build autodiff model tensors with explicit static batch-shaped bases."""

    params_batch = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
    transition_offset = _explicit_batched_base(
        fixture.base_transition_offset, batch_size
    ) + tf.einsum("bp,pi->bi", params_batch, fixture.d_transition_offset)
    transition_matrix = _explicit_batched_base(
        fixture.base_transition_matrix, batch_size
    ) + tf.einsum("bp,pij->bij", params_batch, fixture.d_transition_matrix)
    transition_factor = _explicit_batched_base(
        fixture.base_transition_covariance_factor, batch_size
    ) + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_transition_covariance_factor
    )
    transition_covariance = transition_factor @ tf.linalg.matrix_transpose(
        transition_factor
    )

    observation_offset = _explicit_batched_base(
        fixture.base_observation_offset, batch_size
    ) + tf.einsum("bp,pi->bi", params_batch, fixture.d_observation_offset)
    observation_matrix = _explicit_batched_base(
        fixture.base_observation_matrix, batch_size
    ) + tf.einsum("bp,pij->bij", params_batch, fixture.d_observation_matrix)
    observation_factor = _explicit_batched_base(
        fixture.base_observation_covariance_factor, batch_size
    ) + tf.einsum(
        "bp,pij->bij", params_batch, fixture.d_observation_covariance_factor
    )
    observation_covariance = observation_factor @ tf.linalg.matrix_transpose(
        observation_factor
    )

    initial_mean = _explicit_batched_base(
        fixture.base_initial_mean, batch_size
    ) + tf.einsum("bp,pi->bi", params_batch, fixture.d_initial_mean)
    initial_factor = _explicit_batched_base(
        fixture.base_initial_covariance_factor, batch_size
    ) + tf.einsum("bp,pij->bij", params_batch, fixture.d_initial_covariance_factor)
    initial_covariance = initial_factor @ tf.linalg.matrix_transpose(initial_factor)

    return (
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_mean,
        initial_covariance,
    )


def _graph_input_node_name(tensor: tf.Tensor) -> str:
    return tensor.name.split(":", 1)[0]


def _graph_input_base_name(input_name: str) -> str:
    return input_name.lstrip("^").split(":", 1)[0]


def _normalize_leading_batch_shapes(node: Any, batch_size: int) -> None:
    shape_attrs = []
    if "shape" in node.attr:
        shape_attrs.append(node.attr["shape"].shape)
    if "_output_shapes" in node.attr:
        shape_attrs.extend(node.attr["_output_shapes"].list.shape)
    for shape in shape_attrs:
        if shape.unknown_rank or not shape.dim:
            continue
        if shape.dim[0].size == batch_size:
            shape.dim[0].size = -31337


def _normalized_graphdef_payload(
    graph_def: Any,
    *,
    parameter_input_name: str,
    batch_size: int,
) -> bytes:
    """Normalize only leading-B shape metadata on parameter descendants."""

    normalized = copy.deepcopy(graph_def)
    descendants = {parameter_input_name}
    for node in normalized.node:
        if node.name == parameter_input_name or any(
            _graph_input_base_name(input_name) in descendants for input_name in node.input
        ):
            descendants.add(node.name)
            _normalize_leading_batch_shapes(node, batch_size)
    return normalized.SerializeToString(deterministic=True)


def _normalized_graphdef_digest(
    graph_def: Any,
    *,
    parameter_input_name: str,
    batch_size: int,
) -> str:
    return hashlib.sha256(
        _normalized_graphdef_payload(
            graph_def,
            parameter_input_name=parameter_input_name,
            batch_size=batch_size,
        )
    ).hexdigest()


def _trace_batched_fixture_graph(
    fixture: Fixture,
    *,
    batch_size: int,
) -> dict[str, Any]:
    tensor_fields = (
        "base_initial_mean",
        "base_initial_covariance_factor",
        "base_transition_offset",
        "base_transition_matrix",
        "base_transition_covariance_factor",
        "base_observation_offset",
        "base_observation_matrix",
        "base_observation_covariance_factor",
        "d_initial_mean",
        "d_initial_covariance_factor",
        "d_transition_offset",
        "d_transition_matrix",
        "d_transition_covariance_factor",
        "d_observation_offset",
        "d_observation_matrix",
        "d_observation_covariance_factor",
    )

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def fixture_wrapper(parameters_batch: tf.Tensor) -> tuple[tf.Tensor, ...]:
        traced_fixture = replace(
            fixture,
            **{
                name: tf.constant(getattr(fixture, name), name=f"fixture_{name}")
                for name in tensor_fields
            },
        )
        return _batched_model_tensors(traced_fixture, parameters_batch)

    concrete = fixture_wrapper.get_concrete_function()
    graph_def = concrete.graph.as_graph_def(add_shapes=True)
    input_name = _graph_input_node_name(concrete.inputs[0])
    return {
        "batch_size": batch_size,
        "node_count": len(graph_def.node),
        "serialized_bytes": len(graph_def.SerializeToString(deterministic=True)),
        "normalized_structural_digest": _normalized_graphdef_digest(
            graph_def,
            parameter_input_name=input_name,
            batch_size=batch_size,
        ),
        "parameter_input_name": input_name,
        "output_shapes": [tensor.shape.as_list() for tensor in concrete.outputs],
        "graph_def": graph_def,
    }


def _tensor_sha256(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()
    return hashlib.sha256(serialized).hexdigest()


def _combined_tensor_sha256(values: tuple[tf.Tensor, ...]) -> str:
    return hashlib.sha256(
        "".join(_tensor_sha256(value) for value in values).encode("ascii")
    ).hexdigest()


def _fixture_identity_record(fixture: Fixture, *, batch_size: int) -> dict[str, Any]:
    base_values = (
        fixture.base_initial_mean,
        fixture.base_initial_covariance_factor,
        fixture.base_transition_offset,
        fixture.base_transition_matrix,
        fixture.base_transition_covariance_factor,
        fixture.base_observation_offset,
        fixture.base_observation_matrix,
        fixture.base_observation_covariance_factor,
    )
    derivative_values = (
        fixture.d_initial_mean,
        fixture.d_initial_covariance_factor,
        fixture.d_transition_offset,
        fixture.d_transition_matrix,
        fixture.d_transition_covariance_factor,
        fixture.d_observation_offset,
        fixture.d_observation_matrix,
        fixture.d_observation_covariance_factor,
    )
    selected = _make_parameter_batch(fixture, batch_size)
    return {
        "parameter_count": fixture.parameter_count,
        "batch_size": batch_size,
        "dtype": fixture.dtype.name,
        "base_model_hash": _combined_tensor_sha256(base_values),
        "observation_hash": _tensor_sha256(fixture.observations),
        "parameter_hash": _tensor_sha256(fixture.parameters),
        "derivative_basis_hash": _combined_tensor_sha256(derivative_values),
        "proposal_cloud_hash": _tensor_sha256(_make_parameter_cloud(fixture)),
        "selected_parameter_hash": _tensor_sha256(selected),
        "proposal_row_ids": list(_proposal_row_ids(batch_size)),
    }


def _tensor_exact_equal(left: tf.Tensor, right: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.equal(left, right)).numpy())


def _fixture_parity_record(fixture: Fixture, *, batch_size: int) -> dict[str, Any]:
    parameters_batch = _make_parameter_batch(fixture, batch_size)
    actual = _batched_model_tensors(fixture, parameters_batch)
    scalar_rows = [
        _model_tensors(fixture, parameters_batch[row]) for row in range(batch_size)
    ]
    expected = tuple(
        tf.stack(values, axis=0) for values in zip(*scalar_rows, strict=True)
    )
    tolerance = 2.0e-6 if fixture.dtype == tf.float32 else 2.0e-13
    exact_tensor_indices = {8, 10, 11, 13, 14}
    tensor_rows = []
    for index, (name, actual_tensor, expected_tensor) in enumerate(
        zip(PHASE2_TENSOR_NAMES, actual, expected, strict=True)
    ):
        residual = tf.abs(actual_tensor - expected_tensor)
        allowed = tolerance + tolerance * tf.abs(expected_tensor)
        exact_required = index in exact_tensor_indices
        exact_passed = _tensor_exact_equal(actual_tensor, expected_tensor)
        near_passed = bool(tf.reduce_all(residual <= allowed).numpy())
        tensor_rows.append(
            {
                "name": name,
                "actual_shape": actual_tensor.shape.as_list(),
                "expected_shape": expected_tensor.shape.as_list(),
                "actual_dtype": actual_tensor.dtype.name,
                "expected_dtype": expected_tensor.dtype.name,
                "shape_matches": actual_tensor.shape == expected_tensor.shape,
                "dtype_matches": actual_tensor.dtype == expected_tensor.dtype,
                "exact_required": exact_required,
                "exact_passed": exact_passed,
                "near_passed": near_passed,
                "rtol": tolerance,
                "atol": tolerance,
                "max_abs_residual": float(tf.reduce_max(residual).numpy()),
            }
        )
    passed = all(
        row["shape_matches"]
        and row["dtype_matches"]
        and row["near_passed"]
        and (row["exact_passed"] if row["exact_required"] else True)
        for row in tensor_rows
    )
    return {
        "dtype": fixture.dtype.name,
        "parameter_count": fixture.parameter_count,
        "batch_size": batch_size,
        "passed": passed,
        "tensor_count": len(tensor_rows),
        "tensors": tensor_rows,
    }


def _constant_inventory(graph_def: Any) -> dict[str, Any]:
    consumers: dict[str, list[str]] = {node.name: [] for node in graph_def.node}
    for node in graph_def.node:
        for edge in node.input:
            source = _graph_input_base_name(edge)
            if source in consumers:
                consumers[source].append(node.name)
    rows = []
    for node in graph_def.node:
        if node.op != "Const":
            continue
        tensor = node.attr["value"].tensor
        shape = [int(dimension.size) for dimension in tensor.tensor_shape.dim]
        element_count = math.prod(shape) if shape else 1
        rows.append(
            {
                "name": node.name,
                "dtype_enum": int(tensor.dtype),
                "rank": len(shape),
                "shape": shape,
                "element_count": element_count,
                "payload_sha256": hashlib.sha256(
                    tensor.SerializeToString(deterministic=True)
                ).hexdigest(),
                "consumers": consumers[node.name],
            }
        )
    return {
        "constant_count": len(rows),
        "constant_inventory_digest": benchmark_contract.canonical_sha256(rows),
    }


def _phase2_nested_fixture_record(dtype: tf.DType) -> dict[str, Any]:
    fixtures = {
        parameter_count: make_fixture(10, parameter_count, 8, dtype=dtype)
        for parameter_count in (50, 150)
    }
    identity_rows = [
        _fixture_identity_record(fixtures[parameter_count], batch_size=batch_size)
        for parameter_count in (50, 150)
        for batch_size in (1, 4, 16)
    ]
    small = fixtures[50]
    large = fixtures[150]
    derivative_fields = (
        "d_initial_mean",
        "d_initial_covariance_factor",
        "d_transition_offset",
        "d_transition_matrix",
        "d_transition_covariance_factor",
        "d_observation_offset",
        "d_observation_matrix",
        "d_observation_covariance_factor",
    )
    derivative_prefix_checks = {
        name: _tensor_exact_equal(getattr(small, name), getattr(large, name)[:50])
        for name in derivative_fields
    }
    cloud_prefix_exact = _tensor_exact_equal(
        _make_parameter_cloud(small), _make_parameter_cloud(large)[:, :50]
    )
    selected = {
        parameter_count: {
            batch_size: _make_parameter_batch(fixtures[parameter_count], batch_size)
            for batch_size in (1, 4, 16)
        }
        for parameter_count in (50, 150)
    }
    row_maps = {
        parameter_count: {
            batch_size: {
                row_id: selected[parameter_count][batch_size][index]
                for index, row_id in enumerate(_proposal_row_ids(batch_size))
            }
            for batch_size in (1, 4, 16)
        }
        for parameter_count in (50, 150)
    }
    batch_subset_exact = all(
        _tensor_exact_equal(row_maps[parameter_count][small_batch][row_id], row_maps[parameter_count][large_batch][row_id])
        for parameter_count in (50, 150)
        for small_batch, large_batch in ((1, 4), (4, 16))
        for row_id in _proposal_row_ids(small_batch)
    )
    parameter_prefix_exact = all(
        _tensor_exact_equal(
            row_maps[50][batch_size][row_id],
            row_maps[150][batch_size][row_id][:50],
        )
        for batch_size in (1, 4, 16)
        for row_id in _proposal_row_ids(batch_size)
    )
    checks = {
        "base_model_hash_common": len(
            {row["base_model_hash"] for row in identity_rows}
        )
        == 1,
        "observation_hash_common": len(
            {row["observation_hash"] for row in identity_rows}
        )
        == 1,
        "parameter_prefix_exact": _tensor_exact_equal(
            small.parameters, large.parameters[:50]
        ),
        "derivative_prefix_exact": all(derivative_prefix_checks.values()),
        "proposal_cloud_prefix_exact": cloud_prefix_exact,
        "selected_rows_nested_across_batch_exact": batch_subset_exact,
        "selected_rows_prefix_across_parameter_count_exact": parameter_prefix_exact,
        "locked_row_ids_exact": all(
            _proposal_row_ids(batch_size) == expected
            for batch_size, expected in PROPOSAL_ROW_IDS.items()
        ),
    }
    return {
        "dtype": dtype.name,
        "checks": checks,
        "passed": all(checks.values()),
        "derivative_prefix_checks": derivative_prefix_checks,
        "identity_rows": identity_rows,
    }


def _phase2_graph_record() -> dict[str, Any]:
    fixture = make_fixture(10, 50, 8, dtype=tf.float32)
    rows = []
    for batch_size in (1, 4, 16):
        traced = _trace_batched_fixture_graph(fixture, batch_size=batch_size)
        graph_def = traced.pop("graph_def")
        rows.append({**traced, **_constant_inventory(graph_def)})
    checks = {
        "node_count_equal": len({row["node_count"] for row in rows}) == 1,
        "normalized_structural_digest_equal": len(
            {row["normalized_structural_digest"] for row in rows}
        )
        == 1,
        "constant_count_equal": len({row["constant_count"] for row in rows}) == 1,
        "constant_inventory_digest_equal": len(
            {row["constant_inventory_digest"] for row in rows}
        )
        == 1,
        "output_batch_shapes_exact": all(
            len(row["output_shapes"]) == len(PHASE2_TENSOR_NAMES)
            and all(shape[0] == row["batch_size"] for shape in row["output_shapes"])
            for row in rows
        ),
    }
    return {"checks": checks, "passed": all(checks.values()), "rows": rows}


def _phase2_declared_source_manifest() -> dict[str, Any]:
    paths = (
        "scripts/kalman_qr_benchmark_contract.py",
        "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
        "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py",
        "tests/test_kalman_qr_batched_fixture.py",
        "tests/test_kalman_qr_benchmark_contract.py",
        "tests/test_kalman_qr_parameter_count_scaling_harness.py",
        "bayesfilter/linear/kalman_qr_derivatives_tf.py",
        "bayesfilter/linear/kalman_qr_tf.py",
        "bayesfilter/linear/qr_factor_tf.py",
    )
    files = [
        {
            "path": path,
            "sha256": benchmark_contract.file_sha256(REPO_ROOT / path),
        }
        for path in paths
    ]
    return {
        "files": files,
        "declared_source_fingerprint": benchmark_contract.canonical_sha256(files),
    }


def _phase3_source_structure_checks() -> dict[str, bool]:
    source_path = REPO_ROOT / "bayesfilter/linear/kalman_qr_derivatives_tf.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    target_names = {
        "_batched_stack_qr_lower_factor_first_derivatives",
        "_batched_qr_factor_derivative",
        "_batched_right_solve_upper",
        "_batched_omega_from_a",
        "_batched_cholesky_factor_first_derivatives",
        "_batched_factor_covariance_first_derivatives",
        "_batched_symmetrize",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in target_names
    }
    forbidden_nodes = (
        ast.For,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    nodes = [node for function in functions.values() for node in ast.walk(function)]
    called_names = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else None
        for node in nodes
        if isinstance(node, ast.Call)
    }
    return {
        "all_helpers_found": set(functions) == target_names,
        "no_python_loop_or_comprehension": not any(
            isinstance(node, forbidden_nodes)
            for function in functions.values()
            for node in ast.walk(function)
        ),
        "no_static_parameter_dim": "_static_dim" not in called_names,
        "no_tensorflow_mapping": called_names.isdisjoint(
            {"map_fn", "vectorized_map", "numpy_function"}
        ),
        "no_scalar_score_call": "tf_qr_sqrt_kalman_score" not in called_names,
    }


def _phase3_graph_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row.get("parameter_count") for row in rows]
    distinct_rows = len(rows) == 2 and labels == [50, 150] and rows[0] is not rows[1]
    if not distinct_rows:
        checks = {
            "distinct_ordered_parameter_rows": False,
            "node_count_equal": False,
            "ordered_op_sequence_equal": False,
            "op_histogram_equal": False,
            "constant_count_equal": False,
            "output_shapes_exact": False,
            "source_structure": False,
        }
        return {"checks": checks, "state": "failed", "returncode": 1}

    small, large = rows
    checks = {
        "distinct_ordered_parameter_rows": True,
        "node_count_equal": small.get("node_count") == large.get("node_count"),
        "ordered_op_sequence_equal": small.get("ordered_op_sequence_digest")
        == large.get("ordered_op_sequence_digest"),
        "op_histogram_equal": small.get("op_histogram") == large.get("op_histogram"),
        "constant_count_equal": small.get("constant_count") == large.get("constant_count"),
        "output_shapes_exact": small.get("output_shapes") == [[4], [4, 50]]
        and large.get("output_shapes") == [[4], [4, 150]],
        "source_structure": all(small.get("source_structure_checks", {}).values())
        and small.get("source_structure_checks") == large.get("source_structure_checks"),
    }
    state = "passed" if all(checks.values()) else "failed"
    return {"checks": checks, "state": state, "returncode": 0 if state == "passed" else 1}


def _phase3_trace_score_graph(
    fixture: Fixture,
    *,
    batch_size: int,
    build_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    builder = build_fn or build_batch_native_analytic_fn
    score_fn = builder(fixture, batch_size=batch_size, jit_compile=False)
    parameters_batch = _make_parameter_batch(fixture, batch_size)
    concrete = score_fn.get_concrete_function(parameters_batch)
    graph_def = concrete.graph.as_graph_def(add_shapes=True)
    op_sequence = [node.op for node in graph_def.node]
    op_histogram: dict[str, int] = {}
    for op in op_sequence:
        op_histogram[op] = op_histogram.get(op, 0) + 1
    return {
        "parameter_count": fixture.parameter_count,
        "batch_size": batch_size,
        "dtype": fixture.dtype.name,
        "node_count": len(graph_def.node),
        "serialized_bytes": len(graph_def.SerializeToString(deterministic=True)),
        "ordered_op_sequence_digest": benchmark_contract.canonical_sha256(op_sequence),
        "op_histogram": op_histogram,
        "constant_count": sum(node.op == "Const" for node in graph_def.node),
        "input_shapes": [tensor.shape.as_list() for tensor in concrete.inputs],
        "output_shapes": [tensor.shape.as_list() for tensor in concrete.outputs],
        "trace_wall_time_seconds": time.perf_counter() - started,
        "fixture_identity": _fixture_identity_record(fixture, batch_size=batch_size),
        "source_structure_checks": _phase3_source_structure_checks(),
    }


def _phase3_declared_source_manifest() -> dict[str, Any]:
    paths = (
        "bayesfilter/linear/kalman_qr_derivatives_tf.py",
        "bayesfilter/linear/kalman_qr_tf.py",
        "bayesfilter/linear/qr_factor_tf.py",
        "scripts/kalman_qr_benchmark_contract.py",
        "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
        "tests/test_linear_qr_batched_parameter_vectorization_tf.py",
        "tests/test_linear_qr_batched_analytical_score_tf.py",
        "tests/test_linear_kalman_qr_derivatives_tf.py",
        "tests/test_linear_qr_factor_tf.py",
        "tests/test_kalman_qr_batched_fixture.py",
    )
    files = [
        {"path": path, "sha256": benchmark_contract.file_sha256(REPO_ROOT / path)}
        for path in paths
    ]
    return {
        "files": files,
        "declared_source_fingerprint": benchmark_contract.canonical_sha256(files),
    }


def run_phase3_parameter_graph_diagnostic(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if args.device != "cpu":
        raise ValueError("Phase 3 diagnostic requires --device cpu")
    if args.jit_compile:
        raise ValueError("Phase 3 diagnostic requires --no-jit-compile")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Phase 3 diagnostic requires CUDA_VISIBLE_DEVICES=-1 before import")

    cpu_thread_manifest = _configure_cpu_threads(args.cpu_threads)
    rows = [
        _phase3_trace_score_graph(
            make_fixture(10, parameter_count, 8, dtype=tf.float32),
            batch_size=4,
        )
        for parameter_count in (50, 150)
    ]
    gate = _phase3_graph_gate(rows)
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    payload = {
        "schema": PHASE3_DIAGNOSTIC_SCHEMA,
        "state": gate["state"],
        "checks": gate["checks"],
        "rows": rows,
        "versions": {
            "fixture_contract_version": benchmark_contract.FIXTURE_CONTRACT_VERSION,
            "parameter_batch_version": benchmark_contract.PARAMETER_BATCH_VERSION,
            "observation_generation_version": benchmark_contract.OBSERVATION_GENERATION_VERSION,
        },
        "source_manifest": benchmark_contract.source_manifest(
            REPO_ROOT, include_supervisor=True
        ),
        "declared_source_manifest": _phase3_declared_source_manifest(),
        "runtime_manifest": benchmark_contract.runtime_manifest(),
        "run_manifest": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run_text(["git", "status", "--short"]),
            "command_argv": list(sys.argv),
            "cwd": str(REPO_ROOT),
            "python_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "UNSET"),
            "conda_prefix": os.environ.get("CONDA_PREFIX", "UNSET"),
            "cpu_thread_manifest": cpu_thread_manifest,
            "device_status": {
                "requested_device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_detection_by_harness": "not_called",
                "trust_basis": "cpu_debug_or_reference_exception",
            },
            "jit_compile": False,
            "xla_execution": "not_run",
            "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"),
            "tf32_status": "not_queried_not_relevant_to_cpu_trace_only_diagnostic",
            "output_json": (
                str(output_json.relative_to(REPO_ROOT))
                if output_json.is_relative_to(REPO_ROOT)
                else str(output_json)
            ),
            "log_path": args.phase3_log_path,
            "plan_path": PHASE3_PLAN_PATH,
            "result_path": PHASE3_RESULT_PATH,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "no warm-runtime improvement claim",
            "no analytical versus autodiff ranking claim",
            "no CPU or GPU scalability claim",
            "no GPU readiness claim",
            "no HMC or posterior correctness claim",
            "no default, production, or scientific validity claim",
        ],
    }
    benchmark_contract.atomic_write_json(output_json, payload)
    print(
        benchmark_contract.strict_json_dumps(
            {
                "state": payload["state"],
                "checks": payload["checks"],
                "output_json": str(output_json),
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            indent=2,
        )
    )
    return gate["returncode"]


def run_phase2_fixture_diagnostic(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if args.device != "cpu":
        raise ValueError("Phase 2 diagnostic requires --device cpu")
    if args.jit_compile:
        raise ValueError("Phase 2 diagnostic requires --no-jit-compile")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Phase 2 diagnostic requires CUDA_VISIBLE_DEVICES=-1 before import")

    cpu_thread_manifest = _configure_cpu_threads(args.cpu_threads)
    parity_rows = [
        _fixture_parity_record(
            make_fixture(10, parameter_count, 8, dtype=dtype),
            batch_size=batch_size,
        )
        for dtype in (tf.float32, tf.float64)
        for parameter_count in (3, 50)
        for batch_size in (1, 4)
    ]
    nested_rows = [
        _phase2_nested_fixture_record(dtype) for dtype in (tf.float32, tf.float64)
    ]
    graph = _phase2_graph_record()
    checks = {
        "all_16_tensor_parity": all(row["passed"] for row in parity_rows),
        "nested_fixture_identity": all(row["passed"] for row in nested_rows),
        "graph_structure": graph["passed"],
    }
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    runtime_manifest = benchmark_contract.runtime_manifest()
    source_manifest = benchmark_contract.source_manifest(
        REPO_ROOT, include_supervisor=True
    )
    payload = {
        "schema": PHASE2_DIAGNOSTIC_SCHEMA,
        "state": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "versions": {
            "fixture_contract_version": benchmark_contract.FIXTURE_CONTRACT_VERSION,
            "parameter_batch_version": benchmark_contract.PARAMETER_BATCH_VERSION,
            "observation_generation_version": benchmark_contract.OBSERVATION_GENERATION_VERSION,
        },
        "fixture_contract": {
            "dimension": 10,
            "timesteps": 8,
            "parity_parameter_counts": [3, 50],
            "nested_parameter_counts": [50, 150],
            "batch_sizes": [1, 4, 16],
            "dtypes": ["float32", "float64"],
            "parameter_formula": "theta[j] = -0.2 + 0.4 * j / 149",
            "canonical_proposal_row_coordinates": "linspace(-1, 1, 16)",
            "proposal_row_ids": {
                str(batch_size): list(row_ids)
                for batch_size, row_ids in PROPOSAL_ROW_IDS.items()
            },
            "observation_source": "parameter-independent base model",
            "deterministic": True,
            "random_seeds": [],
        },
        "parity_rows": parity_rows,
        "nested_fixture_rows": nested_rows,
        "graph": graph,
        "source_manifest": source_manifest,
        "declared_source_manifest": _phase2_declared_source_manifest(),
        "runtime_manifest": runtime_manifest,
        "run_manifest": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run_text(["git", "status", "--short"]),
            "command_argv": list(sys.argv),
            "cwd": str(REPO_ROOT),
            "python_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "UNSET"),
            "conda_prefix": os.environ.get("CONDA_PREFIX", "UNSET"),
            "cpu_thread_manifest": cpu_thread_manifest,
            "device_status": {
                "requested_device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_detection_by_harness": "not_called",
                "trust_basis": "cpu_debug_or_reference_exception",
            },
            "jit_compile": False,
            "xla_execution": "not_run",
            "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"),
            "tf32_status": "not_queried_not_relevant_to_cpu_trace_only_diagnostic",
            "output_json": str(output_json.relative_to(REPO_ROOT)),
            "log_path": args.phase2_log_path,
            "plan_path": PHASE2_PLAN_PATH,
            "result_path": PHASE2_RESULT_PATH,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "no analytical score correctness claim",
            "no autodiff correctness claim",
            "no XLA viability claim",
            "no warm runtime or method ranking claim",
            "no GPU readiness claim",
            "no HMC or posterior correctness claim",
            "no default, production, or scientific validity claim",
        ],
    }
    benchmark_contract.atomic_write_json(output_json, payload)
    print(
        benchmark_contract.strict_json_dumps(
            {
                "state": payload["state"],
                "checks": checks,
                "output_json": str(output_json),
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            indent=2,
        )
    )
    return 0 if payload["state"] == "passed" else 1


def _generate_observations(fixture_without_observations: Fixture) -> tf.Tensor:
    fixture = fixture_without_observations
    dtype = fixture.dtype
    transition_offset = fixture.base_transition_offset
    transition_matrix = fixture.base_transition_matrix
    observation_offset = fixture.base_observation_offset
    observation_matrix = fixture.base_observation_matrix
    state = fixture.base_initial_mean
    obs_index = tf.cast(tf.range(fixture.observation_dim) + 1, dtype)
    obs_values = []
    for t in range(fixture.timesteps):
        state = transition_offset + tf.linalg.matvec(transition_matrix, state)
        deterministic_noise = 0.03 * tf.math.sin(
            tf.cast(t + 1, dtype) * obs_index * tf.constant(0.067, dtype=dtype)
        )
        obs_values.append(
            observation_offset
            + tf.linalg.matvec(observation_matrix, state)
            + deterministic_noise
        )
    return tf.stack(obs_values, axis=0)


def make_fixture(
    dimension: int,
    parameter_count: int,
    timesteps: int,
    *,
    dtype: tf.DType,
) -> Fixture:
    if parameter_count > parameter_capacity(dimension):
        raise ValueError(
            f"parameter_count={parameter_count} exceeds independent lower-triangular "
            f"slot capacity {parameter_capacity(dimension)} for dimension {dimension}"
    )
    slots = _make_slots(dimension, parameter_count)
    index = tf.cast(tf.range(dimension), dtype)
    if parameter_count > MAX_SCALING_PARAMETER_COUNT:
        raise ValueError(
            f"parameter_count={parameter_count} exceeds the nested scaling maximum "
            f"{MAX_SCALING_PARAMETER_COUNT}"
        )
    parameter_indices = tf.cast(tf.range(parameter_count), dtype)
    parameters = (
        tf.constant(-0.2, dtype=dtype)
        + tf.constant(0.4 / (MAX_SCALING_PARAMETER_COUNT - 1), dtype=dtype)
        * parameter_indices
    )
    base_initial_mean = 0.025 * tf.math.sin(index + 1.0)
    base_transition_offset = 0.008 * tf.math.cos((index + 1.0) * 0.13)
    base_observation_offset = 0.015 * tf.math.sin((index + 1.0) * 0.11)
    fixture = Fixture(
        state_dim=dimension,
        observation_dim=dimension,
        timesteps=timesteps,
        parameter_count=parameter_count,
        parameters=parameters,
        observations=tf.zeros([timesteps, dimension], dtype=dtype),
        base_initial_mean=base_initial_mean,
        base_initial_covariance_factor=_make_lower_factor(
            dimension,
            diag_start=0.65,
            offdiag_scale=0.006,
            dtype=dtype,
        ),
        base_transition_offset=base_transition_offset,
        base_transition_matrix=_make_base_transition(dimension, dtype=dtype),
        base_transition_covariance_factor=_make_lower_factor(
            dimension,
            diag_start=0.28,
            offdiag_scale=0.003,
            dtype=dtype,
        ),
        base_observation_offset=base_observation_offset,
        base_observation_matrix=_make_base_observation(dimension, dtype=dtype),
        base_observation_covariance_factor=_make_lower_factor(
            dimension,
            diag_start=0.36,
            offdiag_scale=0.004,
            dtype=dtype,
        ),
        d_initial_mean=_basis_vector(slots, "initial_mean", dimension, 0.0010, dtype),
        d_initial_covariance_factor=_basis_matrix(
            slots,
            "initial_covariance_factor",
            dimension,
            dimension,
            0.0007,
            dtype,
        ),
        d_transition_offset=_basis_vector(slots, "transition_offset", dimension, 0.0010, dtype),
        d_transition_matrix=_basis_matrix(
            slots,
            "transition_matrix",
            dimension,
            dimension,
            0.0008,
            dtype,
        ),
        d_transition_covariance_factor=_basis_matrix(
            slots,
            "transition_covariance_factor",
            dimension,
            dimension,
            0.0007,
            dtype,
        ),
        d_observation_offset=_basis_vector(
            slots,
            "observation_offset",
            dimension,
            0.0010,
            dtype,
        ),
        d_observation_matrix=_basis_matrix(
            slots,
            "observation_matrix",
            dimension,
            dimension,
            0.0008,
            dtype,
        ),
        d_observation_covariance_factor=_basis_matrix(
            slots,
            "observation_covariance_factor",
            dimension,
            dimension,
            0.0007,
            dtype,
        ),
        slot_allocation=_slot_allocation(slots),
        parameter_capacity=parameter_capacity(dimension),
        dtype=dtype,
    )
    return Fixture(
        **{
            **fixture.__dict__,
            "observations": _generate_observations(fixture),
        }
    )


def build_analytic_fn(
    fixture: Fixture,
    *,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[tf.TensorSpec([fixture.parameter_count], fixture.dtype)],
    )
    def analytical_score(parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        tensors = _model_tensors(fixture, parameters)
        return tf_qr_sqrt_kalman_score.python_function(
            observations=fixture.observations,
            transition_offset=tensors[0],
            transition_matrix=tensors[1],
            transition_covariance=tensors[2],
            observation_offset=tensors[3],
            observation_matrix=tensors[4],
            observation_covariance=tensors[5],
            initial_state_mean=tensors[6],
            initial_state_covariance=tensors[7],
            d_initial_state_mean=tensors[8],
            d_initial_state_covariance=tensors[9],
            d_transition_offset=tensors[10],
            d_transition_matrix=tensors[11],
            d_transition_covariance=tensors[12],
            d_observation_offset=tensors[13],
            d_observation_matrix=tensors[14],
            d_observation_covariance=tensors[15],
            jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
            jitter_updates_filtered_covariance=True,
        )

    return analytical_score


def build_autodiff_fn(
    fixture: Fixture,
    *,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[tf.TensorSpec([fixture.parameter_count], fixture.dtype)],
    )
    def autodiff_score(parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=fixture.dtype)
        with tf.GradientTape() as tape:
            tape.watch(params)
            tensors = _model_tensors(fixture, params)
            value = tf_qr_sqrt_kalman_log_likelihood_while_loop.python_function(
                observations=fixture.observations,
                transition_offset=tensors[0],
                transition_matrix=tensors[1],
                transition_covariance=tensors[2],
                observation_offset=tensors[3],
                observation_matrix=tensors[4],
                observation_covariance=tensors[5],
                initial_state_mean=tensors[6],
                initial_state_covariance=tensors[7],
                jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                jitter_updates_filtered_covariance=True,
            )
        score = tape.gradient(value, params)
        if score is None:
            score = tf.fill(tf.shape(params), tf.constant(float("nan"), dtype=fixture.dtype))
        return value, score

    return autodiff_score


def build_batch_native_analytic_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def batch_native_score(parameters_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        tensors = _batched_model_tensors(fixture, parameters_batch)
        return tf_qr_sqrt_kalman_score_batched_static.python_function(
            observations=fixture.observations,
            transition_offset=tensors[0],
            transition_matrix=tensors[1],
            transition_covariance=tensors[2],
            observation_offset=tensors[3],
            observation_matrix=tensors[4],
            observation_covariance=tensors[5],
            initial_state_mean=tensors[6],
            initial_state_covariance=tensors[7],
            d_initial_state_mean=tensors[8],
            d_initial_state_covariance=tensors[9],
            d_transition_offset=tensors[10],
            d_transition_matrix=tensors[11],
            d_transition_covariance=tensors[12],
            d_observation_offset=tensors[13],
            d_observation_matrix=tensors[14],
            d_observation_covariance=tensors[15],
            jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
            jitter_updates_filtered_covariance=True,
        )

    return batch_native_score


def build_scalar_analytic_row_loop_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def scalar_row_loop_score(parameters_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = []
        scores = []
        for batch_index in range(batch_size):
            tensors = _model_tensors(fixture, parameters_batch[batch_index])
            value, score = tf_qr_sqrt_kalman_score.python_function(
                observations=fixture.observations,
                transition_offset=tensors[0],
                transition_matrix=tensors[1],
                transition_covariance=tensors[2],
                observation_offset=tensors[3],
                observation_matrix=tensors[4],
                observation_covariance=tensors[5],
                initial_state_mean=tensors[6],
                initial_state_covariance=tensors[7],
                d_initial_state_mean=tensors[8],
                d_initial_state_covariance=tensors[9],
                d_transition_offset=tensors[10],
                d_transition_matrix=tensors[11],
                d_transition_covariance=tensors[12],
                d_observation_offset=tensors[13],
                d_observation_matrix=tensors[14],
                d_observation_covariance=tensors[15],
                jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                jitter_updates_filtered_covariance=True,
            )
            values.append(value)
            scores.append(score)
        return tf.stack(values, axis=0), tf.stack(scores, axis=0)

    return scalar_row_loop_score


def build_autodiff_row_loop_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def autodiff_row_loop_score(parameters_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
        values = []
        scores = []
        for batch_index in range(batch_size):
            row_params = params[batch_index]
            with tf.GradientTape() as tape:
                tape.watch(row_params)
                tensors = _model_tensors(fixture, row_params)
                value = tf_qr_sqrt_kalman_log_likelihood_while_loop.python_function(
                    observations=fixture.observations,
                    transition_offset=tensors[0],
                    transition_matrix=tensors[1],
                    transition_covariance=tensors[2],
                    observation_offset=tensors[3],
                    observation_matrix=tensors[4],
                    observation_covariance=tensors[5],
                    initial_state_mean=tensors[6],
                    initial_state_covariance=tensors[7],
                    jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                    jitter_updates_filtered_covariance=True,
                )
            score = tape.gradient(value, row_params)
            if score is None:
                score = tf.fill(
                    tf.shape(row_params),
                    tf.constant(float("nan"), dtype=fixture.dtype),
                )
            values.append(value)
            scores.append(score)
        return tf.stack(values, axis=0), tf.stack(scores, axis=0)

    return autodiff_row_loop_score


def build_batch_native_autodiff_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Differentiate the batch-static likelihood with one reverse-mode VJP."""

    from bayesfilter.linear.kalman_qr_tf import (
        tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
    )

    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def batch_native_autodiff_score(
        parameters_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
        with tf.GradientTape() as tape:
            tape.watch(params)
            tensors = _batched_model_tensors(fixture, params)
            value = tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function(
                observations=fixture.observations,
                transition_offset=tensors[0],
                transition_matrix=tensors[1],
                transition_covariance=tensors[2],
                observation_offset=tensors[3],
                observation_matrix=tensors[4],
                observation_covariance=tensors[5],
                initial_state_mean=tensors[6],
                initial_state_covariance=tensors[7],
                jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                jitter_updates_filtered_covariance=True,
            )
        score = tape.gradient(
            value,
            params,
            output_gradients=tf.ones_like(value),
        )
        if score is None:
            raise RuntimeError("batch-native QR likelihood gradient is disconnected")
        return value, score

    return batch_native_autodiff_score


def build_batch_native_autodiff_reduce_sum_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Test a scalar reduction instead of an explicit all-ones VJP seed."""

    from bayesfilter.linear.kalman_qr_tf import (
        tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
    )

    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def batch_native_autodiff_score(
        parameters_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
        with tf.GradientTape() as tape:
            tape.watch(params)
            tensors = _batched_model_tensors(fixture, params)
            value = tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function(
                observations=fixture.observations,
                transition_offset=tensors[0],
                transition_matrix=tensors[1],
                transition_covariance=tensors[2],
                observation_offset=tensors[3],
                observation_matrix=tensors[4],
                observation_covariance=tensors[5],
                initial_state_mean=tensors[6],
                initial_state_covariance=tensors[7],
                jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                jitter_updates_filtered_covariance=True,
            )
            reduced_value = tf.reduce_sum(value)
        score = tape.gradient(reduced_value, params)
        if score is None:
            raise RuntimeError("batch-native QR likelihood gradient is disconnected")
        return value, score

    return batch_native_autodiff_score


def _build_batch_native_autodiff_value_only_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
    explicit_batch_shape: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    from bayesfilter.linear.kalman_qr_tf import (
        tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
    )

    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                [batch_size, fixture.parameter_count],
                fixture.dtype,
                name="parameters_batch",
            )
        ],
    )
    def batch_native_autodiff_score(
        parameters_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters_batch, dtype=fixture.dtype)
        with tf.GradientTape() as tape:
            tape.watch(params)
            if explicit_batch_shape:
                tensors = _batched_model_value_tensors_explicit(
                    fixture,
                    params,
                    batch_size=batch_size,
                )
            else:
                tensors = _batched_model_value_tensors(fixture, params)
            value = tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function(
                observations=fixture.observations,
                transition_offset=tensors[0],
                transition_matrix=tensors[1],
                transition_covariance=tensors[2],
                observation_offset=tensors[3],
                observation_matrix=tensors[4],
                observation_covariance=tensors[5],
                initial_state_mean=tensors[6],
                initial_state_covariance=tensors[7],
                jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
                jitter_updates_filtered_covariance=True,
            )
        score = tape.gradient(
            value,
            params,
            output_gradients=tf.ones_like(value),
        )
        if score is None:
            raise RuntimeError("batch-native QR likelihood gradient is disconnected")
        return value, score

    return batch_native_autodiff_score


def build_batch_native_autodiff_value_only_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Counterfactual using only model tensors consumed by autodiff."""

    return _build_batch_native_autodiff_value_only_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=jit_compile,
        explicit_batch_shape=False,
    )


def build_batch_native_autodiff_value_only_explicit_fn(
    fixture: Fixture,
    *,
    batch_size: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Counterfactual adding explicit static batch shapes to value-only tensors."""

    return _build_batch_native_autodiff_value_only_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=jit_compile,
        explicit_batch_shape=True,
    )


def _selected_method_builder(
    method_id: str,
    *,
    fixture: Fixture,
    batch_size: int,
    jit_compile: bool,
    builder_registry: dict[str, Callable[[], Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]]]
    | None = None,
) -> tuple[Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]], list[str]]:
    """Dispatch exactly one benchmark builder and return its invocation ledger."""

    registry = builder_registry or {
        "batch_native_analytical_qr_score": lambda: build_batch_native_analytic_fn(
            fixture,
            batch_size=batch_size,
            jit_compile=jit_compile,
        ),
        "batch_native_autodiff_qr_score": lambda: build_batch_native_autodiff_fn(
            fixture,
            batch_size=batch_size,
            jit_compile=jit_compile,
        ),
        "scalar_analytical_row_loop": lambda: build_scalar_analytic_row_loop_fn(
            fixture,
            batch_size=batch_size,
            jit_compile=jit_compile,
        ),
        "autodiff_row_loop_qr_score": lambda: build_autodiff_row_loop_fn(
            fixture,
            batch_size=batch_size,
            jit_compile=jit_compile,
        ),
    }
    if set(registry) != set(benchmark_contract.METHOD_IDS):
        raise ValueError("method builder registry does not match the closed method contract")
    try:
        builder = registry[method_id]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark method {method_id!r}") from exc
    invocation_ledger = [method_id]
    return builder(), invocation_ledger


def _phase6_tensor_signature(tensor: tf.Tensor) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
    }


def _phase6_exact_process_argv() -> list[str]:
    return _phase6_preimport_exact_process_argv()


def _load_phase6_supervisor_for_discovery() -> None:
    path = REPO_ROOT / "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"
    module_name = "bayesfilter_phase6_supervisor_discovery"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Phase 6 supervisor from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def run_phase6_import_discovery(args: argparse.Namespace) -> int:
    if not PHASE6_IMPORT_DISCOVERY_PREIMPORT_VALIDATED:
        raise ValueError("Phase 6 import discovery requires its closed child invocation")
    if (
        args.device != "cpu"
        or args.cpu_threads != 1
        or args.output_json != PHASE6_IMPORT_DISCOVERY_OUTPUT
        or {
            name: os.environ.get(name)
            for name in PHASE6_IMPORT_DISCOVERY_ENVIRONMENT
        }
        != PHASE6_IMPORT_DISCOVERY_ENVIRONMENT
    ):
        raise ValueError("Phase 6 import discovery authority changed after validation")
    _load_phase6_supervisor_for_discovery()
    payload = {
        "schema": PHASE6_IMPORT_DISCOVERY_SCHEMA,
        "kind": "import_only_no_fixture_trace_or_execution",
        "command_argv": [
            sys.executable,
            *sys.argv,
        ],
        "environment": dict(PHASE6_IMPORT_DISCOVERY_ENVIRONMENT),
        "fixture_constructed": False,
        "trace_requested": False,
        "selected_method_constructed": False,
        "concrete_function_invocations": 0,
        "manifest": benchmark_contract.repository_module_manifest(REPO_ROOT),
        "nonclaims": list(benchmark_contract.PHASE6_NONCLAIMS),
    }
    benchmark_contract.durable_atomic_write_json(Path(args.output_json).resolve(), payload)
    return 0


def run_phase6_trace_only(args: argparse.Namespace) -> int:
    if args.method not in benchmark_contract.PRIMARY_METHOD_IDS:
        raise ValueError("Phase 6 trace-only mode requires one primary method")
    if len(args.dimensions) != 1 or len(args.parameter_counts) != 1:
        raise ValueError("Phase 6 trace-only mode requires one dimension and parameter count")
    if args.jit_compile:
        raise ValueError("Phase 6 trace-only mode must use --no-jit-compile")
    required_identity = {
        "case_id": args.case_id,
        "attempt_id": args.attempt_id,
        "progress_journal": args.progress_journal,
        "source_fingerprint": args.source_fingerprint,
        "config_fingerprint": args.config_fingerprint,
        "runtime_fingerprint": args.runtime_fingerprint,
        "fixture_fingerprint": args.fixture_fingerprint,
        "schedule_fingerprint": args.schedule_fingerprint,
        "resume_key": args.resume_key,
    }
    if any(not value for value in required_identity.values()):
        raise ValueError("Phase 6 trace-only mode requires reviewed identity fields")
    command_argv = _phase6_exact_process_argv()
    dtype = _resolve_dtype(args.dtype)
    identity = benchmark_contract.phase6_identity(
        dimension=args.dimensions[0],
        parameter_count=args.parameter_counts[0],
        batch_size=args.batch_size,
        dtype=args.dtype,
        method_id=args.method,
        operation="trace",
    )
    started_ns = time.perf_counter_ns()
    stage = "fixture"
    before_manifest: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    state = "failed"

    def enter(next_stage: str) -> None:
        nonlocal stage
        stage = next_stage
        benchmark_contract.append_progress_event(
            Path(args.progress_journal),
            {
                "attempt_id": args.attempt_id,
                "case_id": args.case_id,
                "method_id": args.method,
                "stage": next_stage,
                "resume_key": args.resume_key,
                **{
                    field: getattr(args, field)
                    for field in benchmark_contract.FINGERPRINT_FIELDS
                },
            },
            allowed_stages=benchmark_contract.PHASE6_TRACE_STAGES,
        )

    try:
        enter("fixture")
        fixture = make_fixture(
            args.dimensions[0], args.parameter_counts[0], args.timesteps, dtype=dtype
        )
        enter("pre_builder_provenance")
        before_manifest = benchmark_contract.repository_module_manifest(REPO_ROOT)
        enter("selected_method_construction")
        selected, invoked_method_ids = _selected_method_builder(
            args.method,
            fixture=fixture,
            batch_size=args.batch_size,
            jit_compile=False,
        )
        enter("get_concrete_function")
        trace_started_ns = time.perf_counter_ns()
        concrete = selected.get_concrete_function()
        trace_finished_ns = time.perf_counter_ns()
        enter("graphdef_extraction")
        graph_def = concrete.graph.as_graph_def(add_shapes=True)
        raw = graph_def.SerializeToString(deterministic=True)
        ordered_ops = [node.op for node in graph_def.node]
        histogram: dict[str, int] = {}
        for op in ordered_ops:
            histogram[op] = histogram.get(op, 0) + 1
        structured_args, structured_kwargs = concrete.structured_input_signature
        if structured_kwargs or len(structured_args) != 1:
            raise RuntimeError("trace concrete function does not have one positional user input")
        user_spec = structured_args[0]
        if not isinstance(user_spec, tf.TensorSpec):
            raise RuntimeError("trace concrete function input is not a TensorSpec")
        expected_spec = tf.TensorSpec(
            [args.batch_size, args.parameter_counts[0]], dtype, name="parameters_batch"
        )
        if user_spec != expected_spec:
            raise RuntimeError(f"unexpected trace input signature: {user_spec!r}")
        inputs = [_phase6_tensor_signature(tensor) for tensor in concrete.inputs]
        capture_count = len(concrete.captured_inputs)
        captures = inputs[-capture_count:] if capture_count else []
        if len(inputs) - capture_count != 1:
            raise RuntimeError("trace concrete function does not expose exactly one user input")
        outputs = [_phase6_tensor_signature(tensor) for tensor in concrete.outputs]
        expected_outputs = [
            {"dtype": args.dtype, "shape": [args.batch_size]},
            {"dtype": args.dtype, "shape": [args.batch_size, args.parameter_counts[0]]},
        ]
        if [
            {"dtype": output["dtype"], "shape": output["shape"]} for output in outputs
        ] != expected_outputs:
            raise RuntimeError(f"unexpected trace outputs: {outputs!r}")
        graph_bytes = benchmark_contract.graphdef_bytes_record(raw)
        tokens = benchmark_contract.graphdef_token_stream(raw)
        evidence = {
            "identity": identity,
            "timesteps": args.timesteps,
            "requested_device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "jit_compile": False,
            "tf32_queried": False,
            "device_enumeration_api_calls": 0,
            "invoked_method_ids": invoked_method_ids,
            "get_concrete_function_calls": 1,
            "concrete_function_invocations": 0,
            "trace_started_ns": trace_started_ns,
            "trace_finished_ns": trace_finished_ns,
            "trace_seconds": (trace_finished_ns - trace_started_ns) / 1.0e9,
            "structured_user_input": {
                "name": user_spec.name,
                "dtype": user_spec.dtype.name,
                "shape": user_spec.shape.as_list(),
            },
            "concrete_inputs": inputs,
            "captured_inputs": captures,
            "concrete_outputs": [
                {**output, "result_position": "value" if index == 0 else "score"}
                for index, output in enumerate(outputs)
            ],
            "graphdef_bytes": graph_bytes,
            "graphdef_versions": {
                "producer": graph_def.versions.producer,
                "min_consumer": graph_def.versions.min_consumer,
                "bad_consumers": list(graph_def.versions.bad_consumers),
            },
            "graphdef_extraction_version": "as_graph_def-add_shapes-deterministic-v1",
            "top_level_node_count": len(graph_def.node),
            "function_count": len(graph_def.library.function),
            "ordered_op_sha256": benchmark_contract.canonical_sha256(ordered_ops),
            "op_histogram": dict(sorted(histogram.items())),
            "typed_token_stream": tokens,
            "typed_token_stream_sha256": benchmark_contract.canonical_sha256(tokens),
        }
        state = "passed"
        enter("terminal_provenance")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    after_manifest = benchmark_contract.repository_module_manifest(REPO_ROOT)
    terminal_stage = stage
    enter("envelope_write")
    finished_ns = time.perf_counter_ns()
    payload = {
        "schema": PHASE6_TRACE_CHILD_SCHEMA,
        "state": state,
        "identity": identity,
        "case_id": args.case_id,
        "attempt_id": args.attempt_id,
        **{
            field: getattr(args, field)
            for field in benchmark_contract.FINGERPRINT_FIELDS
        },
        "resume_key": args.resume_key,
        "stage": terminal_stage,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "elapsed_seconds": (finished_ns - started_ns) / 1.0e9,
        "command_argv": command_argv,
        "dependency_manifest_before_builder": before_manifest,
        "dependency_manifest_after_terminal": after_manifest,
        "evidence": evidence,
        "error": error,
        "nonclaims": list(benchmark_contract.PHASE6_NONCLAIMS),
    }
    benchmark_contract.durable_atomic_write_json(Path(args.output_json).resolve(), payload)
    return 0 if state == "passed" else 1


def _phase4_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_phase4_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            kind = "nan"
        elif value > 0:
            kind = "positive_infinity"
        else:
            kind = "negative_infinity"
        return {"nonfinite": kind}
    return value


def _phase4_materialize(outputs: tuple[tf.Tensor, tf.Tensor]) -> dict[str, Any]:
    value, score = outputs
    return {
        "value": _phase4_json_safe(value.numpy().tolist()),
        "score": _phase4_json_safe(score.numpy().tolist()),
        "value_dtype": value.dtype.name,
        "score_dtype": score.dtype.name,
        "value_shape": value.shape.as_list(),
        "score_shape": score.shape.as_list(),
    }


def _phase4_nested_shape(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return [] if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    child_shapes = [_phase4_nested_shape(item) for item in value]
    if any(shape is None for shape in child_shapes):
        return None
    if child_shapes and any(shape != child_shapes[0] for shape in child_shapes[1:]):
        return None
    return [len(value), *(child_shapes[0] if child_shapes else [])]


def _phase4_flatten_numeric(value: Any) -> list[float] | None:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            child = _phase4_flatten_numeric(item)
            if child is None:
                return None
            flattened.extend(child)
        return flattened
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return [number] if math.isfinite(number) else None


def _phase4_directed_comparison(
    candidate: Any,
    reference: Any,
    *,
    expected_shape: Sequence[int],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    candidate_shape = _phase4_nested_shape(candidate)
    reference_shape = _phase4_nested_shape(reference)
    candidate_values = _phase4_flatten_numeric(candidate)
    reference_values = _phase4_flatten_numeric(reference)
    shapes_match = (
        candidate_shape == list(expected_shape)
        and reference_shape == list(expected_shape)
    )
    finite = candidate_values is not None and reference_values is not None
    passed = shapes_match and finite
    max_abs_residual: float | None = None
    max_allowed: float | None = None
    if passed:
        residuals = [
            abs(candidate_value - reference_value)
            for candidate_value, reference_value in zip(
                candidate_values, reference_values, strict=True
            )
        ]
        allowed = [atol + rtol * abs(value) for value in reference_values]
        max_abs_residual = max(residuals, default=0.0)
        max_allowed = max(allowed, default=atol)
        passed = all(
            residual <= limit
            for residual, limit in zip(residuals, allowed, strict=True)
        )
    return {
        "passed": bool(passed),
        "candidate_shape": candidate_shape,
        "reference_shape": reference_shape,
        "finite": finite,
        "max_abs_residual": max_abs_residual,
        "max_allowed": max_allowed,
        "rtol": rtol,
        "atol": atol,
    }


def _phase4_output_metadata_valid(
    output: Any,
    *,
    dtype: str,
    batch_size: int,
    parameter_count: int,
) -> bool:
    if not isinstance(output, Mapping):
        return False
    return (
        output.get("value_dtype") == dtype
        and output.get("score_dtype") == dtype
        and output.get("value_shape") == [batch_size]
        and output.get("score_shape") == [batch_size, parameter_count]
        and _phase4_nested_shape(output.get("value")) == [batch_size]
        and _phase4_nested_shape(output.get("score")) == [batch_size, parameter_count]
        and _phase4_flatten_numeric(output.get("value")) is not None
        and _phase4_flatten_numeric(output.get("score")) is not None
    )


def _phase4_declared_path_manifest() -> dict[str, Any]:
    files = []
    for relative in PHASE4_DECLARED_PATHS:
        path = REPO_ROOT / relative
        files.append(
            {
                "path": relative,
                "sha256": benchmark_contract.file_sha256(path),
                "git_status_short": _run_text(
                    ["git", "status", "--short", "--", relative]
                ),
            }
        )
    return {
        "files": files,
        "declared_source_fingerprint": benchmark_contract.canonical_sha256(files),
    }


def _phase4_expected_argv(mode: str) -> list[str]:
    script = "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
    if mode == "diagnostic":
        return [
            script,
            "--phase4-autodiff-diagnostic",
            "--device",
            "cpu",
            "--cpu-threads",
            "1",
            "--no-jit-compile",
            "--output-json",
            PHASE4_DIAGNOSTIC_JSON,
            "--phase4-log-path",
            PHASE4_DIAGNOSTIC_LOG,
        ]
    if mode == "cpu_xla_smoke":
        return [
            script,
            "--phase4-autodiff-xla-smoke",
            "--device",
            "cpu",
            "--cpu-threads",
            "1",
            "--jit-compile",
            "--output-json",
            PHASE4_XLA_SMOKE_JSON,
            "--phase4-log-path",
            PHASE4_XLA_SMOKE_LOG,
        ]
    raise ValueError(f"unknown Phase 4 mode {mode!r}")


def _phase4_fixture_identities(mode: str) -> list[dict[str, Any]]:
    rows = [
        _fixture_identity_record(
            make_fixture(2, 3, 4, dtype=dtype),
            batch_size=batch_size,
        )
        for dtype in (tf.float32, tf.float64)
        for batch_size in (1, 4)
    ]
    if mode == "diagnostic":
        return rows
    if mode == "cpu_xla_smoke":
        return [row for row in rows if row["dtype"] == "float32" and row["batch_size"] == 4]
    raise ValueError(f"unknown Phase 4 mode {mode!r}")


def phase4_expected_contract(
    mode: str,
    *,
    output_json: str | None = None,
    log_path: str | None = None,
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if mode == "diagnostic":
        schema = PHASE4_DIAGNOSTIC_SCHEMA
        canonical_output = PHASE4_DIAGNOSTIC_JSON
        canonical_log = PHASE4_DIAGNOSTIC_LOG
        jit_compile = False
        xla_execution = "not_run"
        tf32_status = "not_queried"
    elif mode == "cpu_xla_smoke":
        schema = PHASE4_XLA_SMOKE_SCHEMA
        canonical_output = PHASE4_XLA_SMOKE_JSON
        canonical_log = PHASE4_XLA_SMOKE_LOG
        jit_compile = True
        xla_execution = "executed"
        tf32_status = "not_queried_cpu_xla_irrelevant"
    else:
        raise ValueError(f"unknown Phase 4 mode {mode!r}")
    output = output_json or canonical_output
    log = log_path or canonical_log
    return {
        "schema": schema,
        "mode": mode,
        "method_schema": benchmark_contract.SCHEMA,
        "method_contract_version": benchmark_contract.METHOD_CONTRACT_VERSION,
        "primary_method_ids": list(benchmark_contract.PRIMARY_METHOD_IDS),
        "reference_method_ids": list(benchmark_contract.REFERENCE_METHOD_IDS),
        "versions": {
            "fixture_contract_version": benchmark_contract.FIXTURE_CONTRACT_VERSION,
            "parameter_batch_version": benchmark_contract.PARAMETER_BATCH_VERSION,
            "observation_generation_version": benchmark_contract.OBSERVATION_GENERATION_VERSION,
        },
        "case_contract": {
            "dimension": 2,
            "timesteps": 4,
            "parameter_count": 3,
            "batch_sizes": [1, 4] if mode == "diagnostic" else [4],
            "dtypes": ["float32", "float64"] if mode == "diagnostic" else ["float32"],
        },
        "tolerances": copy.deepcopy(PHASE4_TOLERANCES),
        "fixture_identities": _phase4_fixture_identities(mode),
        "declared_path_manifest": _phase4_declared_path_manifest(),
        "runtime_manifest": benchmark_contract.runtime_manifest(),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "command_argv": list(command_argv or _phase4_expected_argv(mode)),
        "cwd": str(REPO_ROOT),
        "python_executable": str(Path(sys.executable).resolve()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "UNSET"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "UNSET"),
        "output_json": output,
        "log_path": log,
        "plan_path": PHASE4_PLAN_PATH,
        "result_path": PHASE4_RESULT_PATH,
        "requested_device": "cpu",
        "cuda_visible_devices": "-1",
        "gpu_detection_by_harness": "not_called",
        "requested_cpu_threads": 1,
        "thread_environment": {
            "omp_num_threads": "1",
            "tf_num_intraop_threads_env": "1",
            "tf_num_interop_threads_env": "1",
        },
        "jit_compile": jit_compile,
        "xla_execution": xla_execution,
        "tf32_status": tf32_status,
        "nonclaims": list(PHASE4_NONCLAIMS),
    }


def _phase4_common_checks(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, bool]:
    provenance = raw.get("provenance")
    methods = raw.get("methods")
    versions = raw.get("versions")
    if not isinstance(provenance, Mapping):
        provenance = {}
    if not isinstance(methods, Mapping):
        methods = {}
    return {
        "schema_and_mode": raw.get("schema") == expected["schema"]
        and raw.get("mode") == expected["mode"],
        "method_contract_identity": methods.get("schema") == expected["method_schema"]
        and methods.get("contract_version") == expected["method_contract_version"]
        and methods.get("primary_ids") == expected["primary_method_ids"]
        and methods.get("reference_ids") == expected["reference_method_ids"],
        "fixture_version_identity": versions == expected["versions"],
        "case_contract_identity": raw.get("case_contract") == expected["case_contract"],
        "tolerance_identity": raw.get("tolerances") == expected["tolerances"],
        "fixture_hash_identity": raw.get("fixture_identities")
        == expected["fixture_identities"],
        "declared_path_identity": raw.get("declared_path_manifest")
        == expected["declared_path_manifest"],
        "source_fingerprint_identity": isinstance(
            raw.get("declared_path_manifest"), Mapping
        )
        and raw["declared_path_manifest"].get("declared_source_fingerprint")
        == expected["declared_path_manifest"]["declared_source_fingerprint"],
        "runtime_identity": raw.get("runtime_manifest") == expected["runtime_manifest"],
        "git_commit_identity": bool(provenance.get("git_commit"))
        and provenance.get("git_commit") == expected["git_commit"],
        "argv_identity": provenance.get("command_argv") == expected["command_argv"],
        "path_identity": provenance.get("cwd") == expected["cwd"]
        and provenance.get("python_executable") == expected["python_executable"]
        and provenance.get("conda_default_env") == expected["conda_default_env"]
        and provenance.get("conda_prefix") == expected["conda_prefix"]
        and provenance.get("output_json") == expected["output_json"]
        and provenance.get("log_path") == expected["log_path"]
        and provenance.get("plan_path") == expected["plan_path"]
        and provenance.get("result_path") == expected["result_path"],
        "cpu_device_identity": provenance.get("requested_device")
        == expected["requested_device"]
        and provenance.get("cuda_visible_devices")
        == expected["cuda_visible_devices"]
        and provenance.get("gpu_detection_by_harness")
        == expected["gpu_detection_by_harness"],
        "thread_identity": provenance.get("requested_cpu_threads")
        == expected["requested_cpu_threads"]
        and provenance.get("effective_intra_op_threads") == 1
        and provenance.get("effective_inter_op_threads") == 1
        and provenance.get("thread_environment") == expected["thread_environment"],
        "jit_xla_tf32_identity": provenance.get("jit_compile")
        == expected["jit_compile"]
        and provenance.get("xla_execution") == expected["xla_execution"]
        and provenance.get("tf32_status") == expected["tf32_status"],
        "nonclaims_identity": raw.get("nonclaims") == expected["nonclaims"],
        "collection_succeeded": raw.get("collection_error") is None,
        "positive_internal_wall_time": isinstance(
            raw.get("internal_wall_time_seconds"), (int, float)
        )
        and not isinstance(raw.get("internal_wall_time_seconds"), bool)
        and math.isfinite(float(raw.get("internal_wall_time_seconds", 0.0)))
        and float(raw.get("internal_wall_time_seconds", 0.0)) > 0.0,
    }


def _phase4_raw_provenance(
    args: argparse.Namespace,
    *,
    mode: str,
    output_json: str,
    cpu_thread_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "command_argv": list(sys.argv),
        "cwd": str(REPO_ROOT),
        "python_executable": str(Path(sys.executable).resolve()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "UNSET"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "UNSET"),
        "output_json": output_json,
        "log_path": args.phase4_log_path,
        "plan_path": PHASE4_PLAN_PATH,
        "result_path": PHASE4_RESULT_PATH,
        "requested_device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_detection_by_harness": "not_called",
        "requested_cpu_threads": args.cpu_threads,
        "effective_intra_op_threads": cpu_thread_manifest.get(
            "tf_intra_op_parallelism_threads"
        ),
        "effective_inter_op_threads": cpu_thread_manifest.get(
            "tf_inter_op_parallelism_threads"
        ),
        "thread_environment": {
            "omp_num_threads": cpu_thread_manifest.get("omp_num_threads"),
            "tf_num_intraop_threads_env": cpu_thread_manifest.get(
                "tf_num_intraop_threads_env"
            ),
            "tf_num_interop_threads_env": cpu_thread_manifest.get(
                "tf_num_interop_threads_env"
            ),
        },
        "jit_compile": mode == "cpu_xla_smoke",
        "xla_execution": "executed" if mode == "cpu_xla_smoke" else "not_run",
        "tf32_status": (
            "not_queried_cpu_xla_irrelevant"
            if mode == "cpu_xla_smoke"
            else "not_queried"
        ),
    }


def _phase4_raw_shell(
    args: argparse.Namespace,
    *,
    mode: str,
    output_json: str,
    cpu_thread_manifest: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = expected or phase4_expected_contract(mode)
    return {
        "schema": expected["schema"],
        "mode": mode,
        "methods": {
            "schema": benchmark_contract.SCHEMA,
            "contract_version": benchmark_contract.METHOD_CONTRACT_VERSION,
            "primary_ids": list(benchmark_contract.PRIMARY_METHOD_IDS),
            "reference_ids": list(benchmark_contract.REFERENCE_METHOD_IDS),
        },
        "versions": copy.deepcopy(expected["versions"]),
        "case_contract": copy.deepcopy(expected["case_contract"]),
        "tolerances": copy.deepcopy(PHASE4_TOLERANCES),
        "fixture_identities": _phase4_fixture_identities(mode),
        "declared_path_manifest": _phase4_declared_path_manifest(),
        "runtime_manifest": benchmark_contract.runtime_manifest(),
        "provenance": _phase4_raw_provenance(
            args,
            mode=mode,
            output_json=output_json,
            cpu_thread_manifest=cpu_thread_manifest,
        ),
        "nonclaims": list(PHASE4_NONCLAIMS),
        "collection_error": None,
    }


def _phase4_raw_batch_value(
    fixture: Fixture, parameters_batch: tf.Tensor
) -> tf.Tensor:
    from bayesfilter.linear.kalman_qr_tf import (
        tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop,
    )

    tensors = _batched_model_tensors(fixture, parameters_batch)
    return tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop.python_function(
        observations=fixture.observations,
        transition_offset=tensors[0],
        transition_matrix=tensors[1],
        transition_covariance=tensors[2],
        observation_offset=tensors[3],
        observation_matrix=tensors[4],
        observation_covariance=tensors[5],
        initial_state_mean=tensors[6],
        initial_state_covariance=tensors[7],
        jitter=tf.constant(1.0e-9, dtype=fixture.dtype),
        jitter_updates_filtered_covariance=True,
    )


def _phase4_collect_parity_row(dtype: tf.DType, batch_size: int) -> dict[str, Any]:
    fixture = make_fixture(2, 3, 4, dtype=dtype)
    parameters_batch = _make_parameter_batch(fixture, batch_size)
    batch_autodiff = build_batch_native_autodiff_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    scalar_autodiff = build_autodiff_row_loop_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    analytical = build_batch_native_analytic_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    with tf.device("/CPU:0"):
        return {
            "dtype": dtype.name,
            "batch_size": batch_size,
            "parameter_count": 3,
            "batch_native_autodiff": _phase4_materialize(
                batch_autodiff(parameters_batch)
            ),
            "scalar_autodiff": _phase4_materialize(
                scalar_autodiff(parameters_batch)
            ),
            "batch_native_analytical": _phase4_materialize(
                analytical(parameters_batch)
            ),
        }


def _phase4_collect_jacobian_row(dtype: tf.DType) -> dict[str, Any]:
    fixture = make_fixture(2, 3, 4, dtype=dtype)
    parameters_batch = _make_parameter_batch(fixture, 4)
    batch_autodiff = build_batch_native_autodiff_fn(
        fixture,
        batch_size=4,
        jit_compile=False,
    )
    with tf.device("/CPU:0"):
        value, score = batch_autodiff(parameters_batch)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(parameters_batch)
            raw_value = _phase4_raw_batch_value(fixture, parameters_batch)
        jacobian = tape.jacobian(
            raw_value,
            parameters_batch,
            experimental_use_pfor=False,
        )
        del tape
        if jacobian is None:
            raise RuntimeError("full batch likelihood Jacobian is disconnected")
        perturbation = tf.constant([0.01, -0.015, 0.02], dtype=dtype)
        perturbed_parameters = tf.tensor_scatter_nd_add(
            parameters_batch,
            [[2]],
            [perturbation],
        )
        perturbed_value, perturbed_score = batch_autodiff(perturbed_parameters)
    return {
        "dtype": dtype.name,
        "batch_size": 4,
        "parameter_count": 3,
        "value": _phase4_json_safe(value.numpy().tolist()),
        "score": _phase4_json_safe(score.numpy().tolist()),
        "value_dtype": value.dtype.name,
        "score_dtype": score.dtype.name,
        "value_shape": value.shape.as_list(),
        "score_shape": score.shape.as_list(),
        "jacobian": _phase4_json_safe(jacobian.numpy().tolist()),
        "jacobian_dtype": jacobian.dtype.name,
        "jacobian_shape": jacobian.shape.as_list(),
        "perturbation": [0.01, -0.015, 0.02],
        "perturbed_value": _phase4_json_safe(perturbed_value.numpy().tolist()),
        "perturbed_score": _phase4_json_safe(perturbed_score.numpy().tolist()),
        "perturbed_value_dtype": perturbed_value.dtype.name,
        "perturbed_score_dtype": perturbed_score.dtype.name,
        "perturbed_value_shape": perturbed_value.shape.as_list(),
        "perturbed_score_shape": perturbed_score.shape.as_list(),
    }


def evaluate_phase4_diagnostic(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    checks = _phase4_common_checks(raw, expected)
    parity_details = []
    parity_rows = raw.get("parity_rows")
    expected_cases = [
        (dtype, batch_size)
        for dtype in expected["case_contract"]["dtypes"]
        for batch_size in expected["case_contract"]["batch_sizes"]
    ]
    rows_by_case: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    if isinstance(parity_rows, list):
        for row in parity_rows:
            if isinstance(row, Mapping):
                key = (row.get("dtype"), row.get("batch_size"))
                if key not in rows_by_case:
                    rows_by_case[key] = row
                else:
                    rows_by_case[key] = {}
    parity_complete = len(parity_rows) == len(expected_cases) if isinstance(parity_rows, list) else False
    parity_valid = parity_complete and set(rows_by_case) == set(expected_cases)
    for dtype, batch_size in expected_cases:
        row = rows_by_case.get((dtype, batch_size), {})
        autodiff = row.get("batch_native_autodiff")
        scalar = row.get("scalar_autodiff")
        analytical = row.get("batch_native_analytical")
        metadata_valid = all(
            _phase4_output_metadata_valid(
                output,
                dtype=dtype,
                batch_size=batch_size,
                parameter_count=3,
            )
            for output in (autodiff, scalar, analytical)
        )
        tolerances = expected["tolerances"][dtype]
        value_scalar = _phase4_directed_comparison(
            autodiff.get("value") if isinstance(autodiff, Mapping) else None,
            scalar.get("value") if isinstance(scalar, Mapping) else None,
            expected_shape=[batch_size],
            **tolerances["value"],
        )
        score_scalar = _phase4_directed_comparison(
            autodiff.get("score") if isinstance(autodiff, Mapping) else None,
            scalar.get("score") if isinstance(scalar, Mapping) else None,
            expected_shape=[batch_size, 3],
            **tolerances["score"],
        )
        value_analytical = _phase4_directed_comparison(
            autodiff.get("value") if isinstance(autodiff, Mapping) else None,
            analytical.get("value") if isinstance(analytical, Mapping) else None,
            expected_shape=[batch_size],
            **tolerances["value"],
        )
        score_analytical = _phase4_directed_comparison(
            autodiff.get("score") if isinstance(autodiff, Mapping) else None,
            analytical.get("score") if isinstance(analytical, Mapping) else None,
            expected_shape=[batch_size, 3],
            **tolerances["score"],
        )
        passed = metadata_valid and all(
            detail["passed"]
            for detail in (
                value_scalar,
                score_scalar,
                value_analytical,
                score_analytical,
            )
        )
        parity_valid = parity_valid and passed
        parity_details.append(
            {
                "dtype": dtype,
                "batch_size": batch_size,
                "metadata_valid": metadata_valid,
                "batch_vs_scalar_value": value_scalar,
                "batch_vs_scalar_score": score_scalar,
                "batch_vs_analytical_value": value_analytical,
                "batch_vs_analytical_score": score_analytical,
                "passed": passed,
            }
        )
    checks["parity_rows_complete"] = parity_complete
    checks["parity_value_score"] = parity_valid

    jacobian_rows = raw.get("jacobian_rows")
    jacobian_by_dtype: dict[str, Mapping[str, Any]] = {}
    if isinstance(jacobian_rows, list):
        for row in jacobian_rows:
            if isinstance(row, Mapping):
                dtype = str(row.get("dtype"))
                if dtype not in jacobian_by_dtype:
                    jacobian_by_dtype[dtype] = row
                else:
                    jacobian_by_dtype[dtype] = {}
    jacobian_complete = (
        isinstance(jacobian_rows, list)
        and len(jacobian_rows) == 2
        and set(jacobian_by_dtype) == {"float32", "float64"}
    )
    jacobian_details = []
    jacobian_valid = jacobian_complete
    for dtype in ("float32", "float64"):
        row = jacobian_by_dtype.get(dtype, {})
        jacobian = row.get("jacobian")
        score = row.get("score")
        value = row.get("value")
        perturbed_value = row.get("perturbed_value")
        perturbed_score = row.get("perturbed_score")
        metadata_valid = (
            row.get("batch_size") == 4
            and row.get("parameter_count") == 3
            and row.get("value_dtype") == dtype
            and row.get("score_dtype") == dtype
            and row.get("jacobian_dtype") == dtype
            and row.get("perturbed_value_dtype") == dtype
            and row.get("perturbed_score_dtype") == dtype
            and row.get("value_shape") == [4]
            and row.get("score_shape") == [4, 3]
            and row.get("jacobian_shape") == [4, 4, 3]
            and row.get("perturbed_value_shape") == [4]
            and row.get("perturbed_score_shape") == [4, 3]
            and _phase4_nested_shape(jacobian) == [4, 4, 3]
            and _phase4_nested_shape(value) == [4]
            and _phase4_nested_shape(score) == [4, 3]
            and _phase4_nested_shape(perturbed_value) == [4]
            and _phase4_nested_shape(perturbed_score) == [4, 3]
            and _phase4_flatten_numeric(jacobian) is not None
            and _phase4_flatten_numeric(value) is not None
            and _phase4_flatten_numeric(score) is not None
            and _phase4_flatten_numeric(perturbed_value) is not None
            and _phase4_flatten_numeric(perturbed_score) is not None
            and row.get("perturbation") == [0.01, -0.015, 0.02]
        )
        diagonal = None
        off_diagonal_values = None
        if _phase4_nested_shape(jacobian) == [4, 4, 3]:
            diagonal = [jacobian[index][index] for index in range(4)]
            off_diagonal_values = [
                jacobian[row_index][column_index]
                for row_index in range(4)
                for column_index in range(4)
                if row_index != column_index
            ]
        tolerances = expected["tolerances"][dtype]
        diagonal_comparison = _phase4_directed_comparison(
            diagonal,
            score,
            expected_shape=[4, 3],
            **tolerances["score"],
        )
        off_diagonal_numeric = _phase4_flatten_numeric(off_diagonal_values)
        off_diagonal_max = (
            max((abs(item) for item in off_diagonal_numeric), default=0.0)
            if off_diagonal_numeric is not None
            else None
        )
        off_diagonal_passed = (
            off_diagonal_max is not None
            and off_diagonal_max <= tolerances["off_diagonal_atol"]
        )
        unaffected_value = None
        unaffected_perturbed_value = None
        unaffected_score = None
        unaffected_perturbed_score = None
        if _phase4_nested_shape(value) == [4] and _phase4_nested_shape(perturbed_value) == [4]:
            unaffected_value = [value[index] for index in (0, 1, 3)]
            unaffected_perturbed_value = [perturbed_value[index] for index in (0, 1, 3)]
        if _phase4_nested_shape(score) == [4, 3] and _phase4_nested_shape(perturbed_score) == [4, 3]:
            unaffected_score = [score[index] for index in (0, 1, 3)]
            unaffected_perturbed_score = [perturbed_score[index] for index in (0, 1, 3)]
        perturb_value_comparison = _phase4_directed_comparison(
            unaffected_perturbed_value,
            unaffected_value,
            expected_shape=[3],
            **tolerances["value"],
        )
        perturb_score_comparison = _phase4_directed_comparison(
            unaffected_perturbed_score,
            unaffected_score,
            expected_shape=[3, 3],
            **tolerances["score"],
        )
        perturbed_row_finite = (
            _phase4_nested_shape(perturbed_value) == [4]
            and _phase4_nested_shape(perturbed_score) == [4, 3]
            and _phase4_flatten_numeric(perturbed_value[2]) is not None
            and _phase4_flatten_numeric(perturbed_score[2]) is not None
        )
        passed = (
            metadata_valid
            and diagonal_comparison["passed"]
            and off_diagonal_passed
            and perturb_value_comparison["passed"]
            and perturb_score_comparison["passed"]
            and perturbed_row_finite
        )
        jacobian_valid = jacobian_valid and passed
        jacobian_details.append(
            {
                "dtype": dtype,
                "metadata_valid": metadata_valid,
                "diagonal_score": diagonal_comparison,
                "off_diagonal_max_abs": off_diagonal_max,
                "off_diagonal_atol": tolerances["off_diagonal_atol"],
                "off_diagonal_passed": off_diagonal_passed,
                "unaffected_value": perturb_value_comparison,
                "unaffected_score": perturb_score_comparison,
                "perturbed_row_finite": perturbed_row_finite,
                "passed": passed,
            }
        )
    checks["jacobian_rows_complete"] = jacobian_complete
    checks["jacobian_row_independence"] = jacobian_valid
    state = "passed" if checks and all(checks.values()) else "failed"
    return {
        "state": state,
        "returncode": 0 if state == "passed" else 1,
        "checks": checks,
        "parity_details": parity_details,
        "jacobian_details": jacobian_details,
    }


def evaluate_phase4_xla_smoke(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    checks = _phase4_common_checks(raw, expected)
    compiled = raw.get("compiled")
    non_jit = raw.get("non_jit")
    metadata_valid = all(
        _phase4_output_metadata_valid(
            output,
            dtype="float32",
            batch_size=4,
            parameter_count=3,
        )
        for output in (compiled, non_jit)
    )
    tolerance = expected["tolerances"]["float32"]
    value_comparison = _phase4_directed_comparison(
        compiled.get("value") if isinstance(compiled, Mapping) else None,
        non_jit.get("value") if isinstance(non_jit, Mapping) else None,
        expected_shape=[4],
        **tolerance["value"],
    )
    score_comparison = _phase4_directed_comparison(
        compiled.get("score") if isinstance(compiled, Mapping) else None,
        non_jit.get("score") if isinstance(non_jit, Mapping) else None,
        expected_shape=[4, 3],
        **tolerance["score"],
    )
    concrete_count = raw.get("concrete_function_count")
    wall_time = raw.get("internal_wall_time_seconds")
    checks.update(
        {
            "compiled_non_jit_metadata": metadata_valid,
            "compiled_non_jit_value_parity": value_comparison["passed"],
            "compiled_non_jit_score_parity": score_comparison["passed"],
            "one_concrete_function": type(concrete_count) is int
            and concrete_count == 1,
            "positive_internal_wall_time": isinstance(wall_time, (int, float))
            and not isinstance(wall_time, bool)
            and math.isfinite(float(wall_time))
            and float(wall_time) > 0.0,
        }
    )
    state = "passed" if checks and all(checks.values()) else "failed"
    return {
        "state": state,
        "returncode": 0 if state == "passed" else 1,
        "checks": checks,
        "value_comparison": value_comparison,
        "score_comparison": score_comparison,
    }


def _phase4_output_path(path: str) -> tuple[Path, str]:
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    recorded = (
        str(output.relative_to(REPO_ROOT))
        if output.is_relative_to(REPO_ROOT)
        else str(output)
    )
    return output, recorded


def run_phase4_autodiff_diagnostic(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if args.device != "cpu":
        raise ValueError("Phase 4 diagnostic requires --device cpu")
    if args.jit_compile:
        raise ValueError("Phase 4 diagnostic requires --no-jit-compile")
    if args.cpu_threads != 1:
        raise ValueError("Phase 4 diagnostic requires --cpu-threads 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Phase 4 diagnostic requires CUDA_VISIBLE_DEVICES=-1 before import")
    output_path, recorded_output = _phase4_output_path(args.output_json)
    cpu_threads = _configure_cpu_threads(args.cpu_threads)
    expected = phase4_expected_contract("diagnostic")
    raw = _phase4_raw_shell(
        args,
        mode="diagnostic",
        output_json=recorded_output,
        cpu_thread_manifest=cpu_threads,
        expected=expected,
    )
    try:
        raw["parity_rows"] = [
            _phase4_collect_parity_row(dtype, batch_size)
            for dtype in (tf.float32, tf.float64)
            for batch_size in (1, 4)
        ]
        raw["jacobian_rows"] = [
            _phase4_collect_jacobian_row(dtype)
            for dtype in (tf.float32, tf.float64)
        ]
    except Exception as exc:
        raw["collection_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raw.setdefault("parity_rows", [])
        raw.setdefault("jacobian_rows", [])
    raw["internal_wall_time_seconds"] = time.perf_counter() - started
    evaluation = evaluate_phase4_diagnostic(raw, expected)
    payload = {
        **raw,
        "state": evaluation["state"],
        "checks": evaluation["checks"],
        "parity_details": evaluation["parity_details"],
        "jacobian_details": evaluation["jacobian_details"],
        "internal_wall_time_seconds": raw["internal_wall_time_seconds"],
    }
    benchmark_contract.atomic_write_json(output_path, payload)
    print(
        benchmark_contract.strict_json_dumps(
            {
                "state": payload["state"],
                "checks": payload["checks"],
                "output_json": str(output_path),
                "internal_wall_time_seconds": payload["internal_wall_time_seconds"],
            },
            indent=2,
        )
    )
    return evaluation["returncode"]


def run_phase4_autodiff_xla_smoke(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if args.device != "cpu":
        raise ValueError("Phase 4 XLA smoke requires --device cpu")
    if not args.jit_compile:
        raise ValueError("Phase 4 XLA smoke requires --jit-compile")
    if args.cpu_threads != 1:
        raise ValueError("Phase 4 XLA smoke requires --cpu-threads 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Phase 4 XLA smoke requires CUDA_VISIBLE_DEVICES=-1 before import")
    output_path, recorded_output = _phase4_output_path(args.output_json)
    cpu_threads = _configure_cpu_threads(args.cpu_threads)
    expected = phase4_expected_contract("cpu_xla_smoke")
    raw = _phase4_raw_shell(
        args,
        mode="cpu_xla_smoke",
        output_json=recorded_output,
        cpu_thread_manifest=cpu_threads,
        expected=expected,
    )
    try:
        fixture = make_fixture(2, 3, 4, dtype=tf.float32)
        parameters_batch = _make_parameter_batch(fixture, 4)
        non_jit = build_batch_native_autodiff_fn(
            fixture,
            batch_size=4,
            jit_compile=False,
        )
        compiled = build_batch_native_autodiff_fn(
            fixture,
            batch_size=4,
            jit_compile=True,
        )
        with tf.device("/CPU:0"):
            raw["non_jit"] = _phase4_materialize(non_jit(parameters_batch))
            raw["compiled"] = _phase4_materialize(compiled(parameters_batch))
            _phase4_materialize(compiled(tf.identity(parameters_batch)))
        raw["concrete_function_count"] = len(
            compiled._list_all_concrete_functions_for_serialization()
        )
    except Exception as exc:
        raw["collection_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raw.setdefault("non_jit", None)
        raw.setdefault("compiled", None)
        raw.setdefault("concrete_function_count", None)
    raw["internal_wall_time_seconds"] = time.perf_counter() - started
    evaluation = evaluate_phase4_xla_smoke(raw, expected)
    payload = {**raw, "state": evaluation["state"], **evaluation}
    benchmark_contract.atomic_write_json(output_path, payload)
    print(
        benchmark_contract.strict_json_dumps(
            {
                "state": payload["state"],
                "checks": payload["checks"],
                "output_json": str(output_path),
                "internal_wall_time_seconds": payload["internal_wall_time_seconds"],
            },
            indent=2,
        )
    )
    return evaluation["returncode"]


def _nested_float(value: Any) -> Any:
    if isinstance(value, list):
        return [_nested_float(item) for item in value]
    return float(value)


def _materialize(
    outputs: tuple[tf.Tensor, tf.Tensor],
) -> dict[str, Any]:
    value, score = outputs
    value_size = math.prod(value.shape.as_list())
    packed = tf.concat([tf.reshape(value, [-1]), tf.reshape(score, [-1])], axis=0)
    packed_np = packed.numpy()
    value_np = packed_np[:value_size].reshape(value.shape.as_list())
    score_np = packed_np[value_size:].reshape(score.shape.as_list())
    return {
        "value": _nested_float(value_np.tolist()),
        "score": _nested_float(score_np.tolist()),
        "value_flat": [float(x) for x in value_np.reshape([-1])],
        "score_flat": [float(x) for x in score_np.reshape([-1])],
        "devices": [value.device, score.device],
        "value_dtype": value.dtype.name,
        "score_dtype": score.dtype.name,
        "value_shape": value.shape.as_list(),
        "score_shape": score.shape.as_list(),
    }


def _synchronize_outputs(
    outputs: tuple[tf.Tensor, tf.Tensor],
    *,
    async_wait: Callable[[], Any] | None = None,
) -> tuple[str, int, str | None]:
    if async_wait is None:
        candidate = getattr(tf.experimental, "async_wait", None)
        async_wait = candidate if callable(candidate) else None
    if async_wait is not None:
        async_wait()
        return "tf.experimental.async_wait", 0, None
    value, score = outputs
    sentinel = tf.reduce_sum(value) + tf.reduce_sum(score)
    sentinel.numpy()
    return "scalar_sentinel", 1, "reduce_sum(value)+reduce_sum(score)"


def _historical_materializing_time_call(
    fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    parameters: tf.Tensor,
    *,
    device_name: str,
) -> tuple[float, dict[str, Any]]:
    """Preserve the pre-Phase-5 boundary for historical grid artifacts only."""

    start = time.perf_counter()
    with tf.device(device_name):
        outputs = fn(parameters)
    materialized = _materialize(outputs)
    return time.perf_counter() - start, materialized


def _timed_stage(
    stage_events: list[dict[str, Any]],
    stage: str,
    action: Callable[[], Any],
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[Any, float]:
    entered_ns = clock_ns()
    try:
        result = action()
    finally:
        finished_ns = clock_ns()
        stage_events.append(
            {
                "sequence_index": len(stage_events),
                "stage": stage,
                "entered_ns": entered_ns,
                "finished_ns": finished_ns,
            }
        )
    return result, (finished_ns - entered_ns) / 1.0e9


def _progress_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attempt_id": args.attempt_id,
        "case_id": args.case_id,
        "method_id": args.method,
        "source_fingerprint": args.source_fingerprint,
        "config_fingerprint": args.config_fingerprint,
        "runtime_fingerprint": args.runtime_fingerprint,
        "fixture_fingerprint": args.fixture_fingerprint,
        "schedule_fingerprint": args.schedule_fingerprint,
        "resume_key": args.resume_key,
    }


def _enter_stage(args: argparse.Namespace, stage: str) -> None:
    event = {**_progress_identity(args), "stage": stage}
    benchmark_contract.append_progress_event(Path(args.progress_journal), event)


def benchmark_selected_method_case(
    *,
    args: argparse.Namespace,
    dimension: int,
    parameter_count: int,
    device_name: str,
    dtype: tf.DType,
) -> dict[str, Any]:
    """Execute one method/case child under the separated v4 timing contract."""

    identity = _progress_identity(args)
    stage_events: list[dict[str, Any]] = []
    durations: dict[str, Any] = {}
    invoked_method_ids: list[str] = []
    materialized: dict[str, Any] | None = None
    measurement: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    current_stage: str | None = None
    failure_stage: str | None = None
    invocation_count = 0
    scalar_materialization_count = 0
    synchronization_method: str | None = None
    sentinel_definition: str | None = None

    try:
        current_stage = "fixture"
        _enter_stage(args, "fixture")
        def build_fixture() -> tuple[Fixture, tf.Tensor]:
            fixture = make_fixture(dimension, parameter_count, args.timesteps, dtype=dtype)
            return fixture, _make_parameter_batch(fixture, args.batch_size)

        (fixture, parameters_batch), durations["fixture_seconds"] = _timed_stage(
            stage_events, "fixture", build_fixture
        )

        current_stage = "trace"
        _enter_stage(args, "trace")
        def trace_selected() -> tuple[Any, int, int]:
            if args.phase6_dependency_before is not None:
                benchmark_contract.durable_atomic_write_json(
                    args.phase6_dependency_before.resolve(),
                    benchmark_contract.repository_module_manifest(REPO_ROOT),
                )
            method_fn, ledger = _selected_method_builder(
                args.method,
                fixture=fixture,
                batch_size=args.batch_size,
                jit_compile=args.jit_compile,
            )
            invoked_method_ids[:] = ledger
            with tf.device(device_name):
                concrete = method_fn.get_concrete_function(parameters_batch)
            graph_def = concrete.graph.as_graph_def(add_shapes=True)
            return concrete, len(graph_def.node), len(graph_def.SerializeToString())

        (concrete, graph_nodes, graph_bytes), durations["trace_seconds"] = _timed_stage(
            stage_events, "trace", trace_selected
        )

        def invoke_and_synchronize() -> tuple[tf.Tensor, tf.Tensor]:
            nonlocal invocation_count, scalar_materialization_count
            nonlocal synchronization_method, sentinel_definition
            invocation_count += 1
            with tf.device(device_name):
                outputs = concrete(parameters_batch)
            method, count, definition = _synchronize_outputs(outputs)
            if synchronization_method is not None and method != synchronization_method:
                raise RuntimeError("synchronization method changed within one child")
            synchronization_method = method
            sentinel_definition = definition
            scalar_materialization_count += count
            return outputs

        current_stage = "first_executable_call"
        _enter_stage(args, "first_executable_call")
        outputs, durations["first_executable_call_seconds"] = _timed_stage(
            stage_events, "first_executable_call", invoke_and_synchronize
        )
        invocation_after_first = invocation_count

        current_stage = "warm_execution"
        _enter_stage(args, "warm_execution")
        warm_seconds: list[float] = []
        warm_entered = time.perf_counter_ns()
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            outputs = invoke_and_synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1.0e9
            warm_seconds.append(elapsed)
        stage_events.append(
            {
                "sequence_index": len(stage_events),
                "stage": "warm_execution",
                "entered_ns": warm_entered,
                "finished_ns": time.perf_counter_ns(),
            }
        )
        durations["warm_execution_seconds"] = warm_seconds
        invocation_after_warm = invocation_count

        current_stage = "materialization"
        _enter_stage(args, "materialization")
        materialized, durations["materialization_seconds"] = _timed_stage(
            stage_events, "materialization", lambda: _materialize(outputs)
        )
        all_finite = _finite_vector(materialized["value_flat"]) and _finite_vector(
            materialized["score_flat"]
        )
        output_metadata = {
            "all_finite": all_finite,
            "value_shape": materialized["value_shape"],
            "score_shape": materialized["score_shape"],
            "value_dtype": materialized["value_dtype"],
            "score_dtype": materialized["score_dtype"],
            "devices": materialized["devices"],
        }

        current_stage = "parity"
        _enter_stage(args, "parity")
        def direct_parity() -> dict[str, Any]:
            nonlocal invocation_count
            invocation_count += 1
            with tf.device(device_name):
                reference_value, reference_score = concrete(parameters_batch)
                timed_value = tf.convert_to_tensor(materialized["value"], dtype=dtype)
                timed_score = tf.convert_to_tensor(materialized["score"], dtype=dtype)
                residuals = tf.stack(
                    [
                        tf.reduce_max(tf.abs(reference_value - timed_value)),
                        tf.reduce_max(tf.abs(reference_score - timed_score)),
                    ]
                )
            value_residual, score_residual = [float(value) for value in residuals.numpy()]
            value_reference_max = max(abs(value) for value in materialized["value_flat"])
            score_reference_max = max(abs(value) for value in materialized["score_flat"])
            tolerance = PHASE4_TOLERANCES[dtype.name]
            passed = (
                value_residual
                <= tolerance["value"]["atol"]
                + tolerance["value"]["rtol"] * value_reference_max
                and score_residual
                <= tolerance["score"]["atol"]
                + tolerance["score"]["rtol"] * score_reference_max
            )
            return {
                "passed": passed,
                "dtype": dtype.name,
                "value_rtol": tolerance["value"]["rtol"],
                "value_atol": tolerance["value"]["atol"],
                "score_rtol": tolerance["score"]["rtol"],
                "score_atol": tolerance["score"]["atol"],
                "value_reference_max_abs": value_reference_max,
                "score_reference_max_abs": score_reference_max,
                "value_max_abs_residual": value_residual,
                "score_max_abs_residual": score_residual,
            }

        direct_output_parity, durations["parity_seconds"] = _timed_stage(
            stage_events, "parity", direct_parity
        )
        expected_value_shape = [args.batch_size]
        expected_score_shape = [args.batch_size, parameter_count]
        dtype_shape_passed = (
            materialized["value_dtype"] == dtype.name
            and materialized["score_dtype"] == dtype.name
            and materialized["value_shape"] == expected_value_shape
            and materialized["score_shape"] == expected_score_shape
        )
        state = (
            "passed"
            if all_finite and dtype_shape_passed and direct_output_parity["passed"]
            else "failed"
        )
        if state == "failed":
            error = {
                "type": "MethodOutputValidityFailure",
                "message": "selected method output failed finite/dtype/shape checks",
            }
        record_outputs = None
        if all_finite:
            record_outputs = {
                "value": materialized["value"],
                "score": materialized["score"],
            }

        payload = {
            "case_id": args.case_id,
            "method_id": args.method,
            "output_metadata": output_metadata,
            "outputs": record_outputs,
            "graphdef": {"node_count": graph_nodes, "serialized_bytes": graph_bytes},
            "direct_output_parity": direct_output_parity,
        }
        current_stage = "payload_encoding"
        _enter_stage(args, "payload_encoding")
        encoded_payload, durations["payload_encoding_seconds"] = _timed_stage(
            stage_events,
            "payload_encoding",
            lambda: benchmark_contract.strict_json_dumps(payload, indent=2) + "\n",
        )
        sidecar_path = Path(args.output_json).with_suffix(".payload.json")
        current_stage = "payload_write"
        _enter_stage(args, "payload_write")
        _, durations["artifact_write_seconds"] = _timed_stage(
            stage_events,
            "payload_write",
            lambda: benchmark_contract.atomic_write_encoded_json(sidecar_path, encoded_payload),
        )
        measurement = {
            "timing_boundary_version": benchmark_contract.TIMING_BOUNDARY_VERSION,
            "requested_repeats": args.repeats,
            "stage_events": stage_events,
            "durations": durations,
            "synchronization": {
                "method": synchronization_method,
                "sentinel_definition": sentinel_definition,
                "scalar_materialization_count": scalar_materialization_count,
                "full_output_materialization_count": 1,
                "parity_residual_materialization_count": 1,
            },
            "invocation_counts": {
                "before_first_executable_call": 0,
                "after_first_executable_call": invocation_after_first,
                "after_warm_execution": invocation_after_warm,
                "after_reference_call": invocation_count,
            },
            "graphdef": {"node_count": graph_nodes, "serialized_bytes": graph_bytes},
            "direct_output_parity": direct_output_parity,
            "payload_sidecar": {
                "path": str(sidecar_path),
                "sha256": benchmark_contract.file_sha256(sidecar_path),
                "write_count": 1,
            },
            "envelope_write_measured": False,
        }
        if not benchmark_contract.measurement_record_is_valid(
            {"measurement": measurement, "output_metadata": output_metadata}
        ):
            raise benchmark_contract.ContractError("constructed measurement record is invalid")
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - exercised by child integration tests.
        failure_stage = current_stage
        state = "failed"
        output_metadata = None
        record_outputs = None
        error = {"type": type(exc).__name__, "message": str(exc)}

    current_stage = "envelope_write"
    _enter_stage(args, "envelope_write")
    return {
        "schema": benchmark_contract.SCHEMA,
        "method_contract_version": benchmark_contract.METHOD_CONTRACT_VERSION,
        **identity,
        "state": state,
        "last_entered_stage": "envelope_write",
        "terminal_stage": "envelope_write",
        "failure_stage": failure_stage,
        "invoked_method_ids": invoked_method_ids,
        "measurement": measurement,
        "returncode": 0 if state == "passed" else 1,
        "timed_out": False,
        "output_metadata": output_metadata,
        "outputs": record_outputs,
        "aggregate_parity_status": "deferred_to_supervisor",
        "error": error,
    }


def _write_v4_method_markdown(record: dict[str, Any], path: Path) -> None:
    lines = [
        "# Kalman QR Batched XLA Repair Method Record",
        "",
        f"- Case: `{record['case_id']}`",
        f"- Method: `{record['method_id']}`",
        f"- State: `{record['state']}`",
        f"- Last stage: `{record['last_entered_stage']}`",
        f"- JIT compile: `{record.get('jit_compile', 'recorded in config fingerprint')}`",
        f"- Timing boundary: `{benchmark_contract.TIMING_BOUNDARY_VERSION}`",
        "",
        "This method-local artifact does not establish comparator parity, XLA ",
        "viability, timing rank, GPU readiness, or scientific/default readiness.",
        "",
    ]
    benchmark_contract.atomic_write_text(path, "\n".join(lines))


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": len(values),
        "mean_seconds": float(statistics.fmean(values)),
        "median_seconds": float(statistics.median(values)),
        "min_seconds": float(min(values)),
        "max_seconds": float(max(values)),
    }


def _finite_vector(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _max_abs_residual(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    return max(abs(left[index] - right[index]) for index in range(len(left)))


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _dtype_tolerances(dtype: tf.DType) -> tuple[float, float]:
    if dtype == tf.float32:
        return 5.0e-3, 5.0e-3
    return 1.0e-8, 1.0e-5


def benchmark_case(
    *,
    dimension: int,
    parameter_count: int,
    timesteps: int,
    repeats: int,
    batch_size: int,
    jit_compile: bool,
    device_name: str,
    dtype: tf.DType,
) -> dict[str, Any]:
    requested_dtype = dtype.name
    capacity = parameter_capacity(dimension)
    if parameter_count > capacity:
        return {
            "state_dim": dimension,
            "observation_dim": dimension,
            "timesteps": timesteps,
            "parameter_count": parameter_count,
            "batch_size": batch_size,
            "parameter_capacity": capacity,
            "applicable": False,
            "applicability_reason": (
                "requested parameter count exceeds independent lower-triangular "
                "slot capacity"
            ),
            "jit_compile": jit_compile,
            "device_name": device_name,
            "requested_dtype": requested_dtype,
            "observed_dtype_check": {
                "passed": None,
                "reason": "not applicable",
            },
        }

    fixture = make_fixture(dimension, parameter_count, timesteps, dtype=dtype)
    parameters_batch = _make_parameter_batch(fixture, batch_size)
    batch_native_analytical_fn = build_batch_native_analytic_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=jit_compile,
    )
    scalar_analytical_row_loop_fn = build_scalar_analytic_row_loop_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=jit_compile,
    )
    autodiff_row_loop_fn = build_autodiff_row_loop_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=jit_compile,
    )
    methods = {
        "batch_native_analytical_qr_score": batch_native_analytical_fn,
        "scalar_analytical_row_loop": scalar_analytical_row_loop_fn,
        "autodiff_row_loop_qr_score": autodiff_row_loop_fn,
    }

    compile_first_calls: dict[str, float] = {}
    warm_start_calls: dict[str, float] = {}
    repeated_calls: dict[str, list[float]] = {name: [] for name in methods}
    outputs: dict[str, dict[str, Any]] = {}

    for name, fn in methods.items():
        elapsed, materialized = _historical_materializing_time_call(
            fn, parameters_batch, device_name=device_name
        )
        compile_first_calls[name] = elapsed
        outputs[name] = materialized

    for name, fn in methods.items():
        elapsed, materialized = _historical_materializing_time_call(
            fn, parameters_batch, device_name=device_name
        )
        warm_start_calls[name] = elapsed
        outputs[name] = materialized

    method_order = list(methods.items())
    for repeat_index in range(repeats):
        ordered = method_order if repeat_index % 2 == 0 else list(reversed(method_order))
        for name, fn in ordered:
            elapsed, materialized = _historical_materializing_time_call(
                fn, parameters_batch, device_name=device_name
            )
            repeated_calls[name].append(elapsed)
            outputs[name] = materialized

    batch_output = outputs["batch_native_analytical_qr_score"]
    scalar_output = outputs["scalar_analytical_row_loop"]
    autodiff_output = outputs["autodiff_row_loop_qr_score"]
    observed_value_dtypes = {
        name: output["value_dtype"] for name, output in outputs.items()
    }
    observed_score_dtypes = {
        name: output["score_dtype"] for name, output in outputs.items()
    }
    dtype_check_passed = all(
        observed == requested_dtype
        for observed in list(observed_value_dtypes.values()) + list(observed_score_dtypes.values())
    )
    expected_value_shape = [batch_size]
    expected_score_shape = [batch_size, parameter_count]
    shape_check_passed = all(
        output["value_shape"] == expected_value_shape and output["score_shape"] == expected_score_shape
        for output in outputs.values()
    )
    batch_scalar_value_residual = _max_abs_residual(
        batch_output["value_flat"],
        scalar_output["value_flat"],
    )
    batch_scalar_score_residual = _max_abs_residual(
        batch_output["score_flat"],
        scalar_output["score_flat"],
    )
    batch_autodiff_value_residual = _max_abs_residual(
        batch_output["value_flat"],
        autodiff_output["value_flat"],
    )
    batch_autodiff_score_residual = _max_abs_residual(
        batch_output["score_flat"],
        autodiff_output["score_flat"],
    )
    score_relative_residual = batch_autodiff_score_residual / max(
        1.0,
        _l2_norm(autodiff_output["score_flat"]),
    )
    all_finite = (
        _finite_vector(batch_output["value_flat"])
        and _finite_vector(scalar_output["value_flat"])
        and _finite_vector(autodiff_output["value_flat"])
        and _finite_vector(batch_output["score_flat"])
        and _finite_vector(scalar_output["score_flat"])
        and _finite_vector(autodiff_output["score_flat"])
    )
    value_tolerance, score_tolerance = _dtype_tolerances(dtype)
    parity_passed = (
        all_finite
        and dtype_check_passed
        and shape_check_passed
        and batch_scalar_value_residual <= value_tolerance
        and batch_scalar_score_residual <= score_tolerance
        and batch_autodiff_value_residual <= value_tolerance
        and batch_autodiff_score_residual <= score_tolerance
    )
    warm_summaries = {name: _summary(times) for name, times in repeated_calls.items()}
    batch_median = warm_summaries["batch_native_analytical_qr_score"]["median_seconds"]
    scalar_median = warm_summaries["scalar_analytical_row_loop"]["median_seconds"]
    autodiff_median = warm_summaries["autodiff_row_loop_qr_score"]["median_seconds"]
    autodiff_ratio = None
    scalar_ratio = None
    if batch_median and autodiff_median:
        autodiff_ratio = float(autodiff_median / batch_median)
    if batch_median and scalar_median:
        scalar_ratio = float(scalar_median / batch_median)

    first_minus_warm = {
        name: float(compile_first_calls[name] - warm_start_calls[name])
        for name in methods
    }
    batch_native_autodiff = build_batch_native_autodiff_fn(
        fixture,
        batch_size=batch_size,
        jit_compile=False,
    )
    _probe_elapsed, probe_output = _historical_materializing_time_call(
        batch_native_autodiff,
        parameters_batch,
        device_name=device_name,
    )
    batched_static_probe_finite = (
        _finite_vector(probe_output["value_flat"])
        and _finite_vector(probe_output["score_flat"])
    )
    return {
        "state_dim": dimension,
        "observation_dim": dimension,
        "timesteps": timesteps,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "parameter_capacity": fixture.parameter_capacity,
        "applicable": True,
        "slot_allocation": fixture.slot_allocation,
        "matrix_parameterization": {
            "transition_matrix": "lower_triangular",
            "observation_matrix": "lower_triangular",
            "covariance_inputs": "SPD matrices formed from lower_triangular_factors",
        },
        "jit_compile": jit_compile,
        "device_name": device_name,
        "requested_dtype": requested_dtype,
        "observed_value_dtypes": observed_value_dtypes,
        "observed_score_dtypes": observed_score_dtypes,
        "observed_dtype_check": {
            "passed": dtype_check_passed,
            "expected": requested_dtype,
        },
        "input_shapes": {
            "observations": fixture.observations.shape.as_list(),
            "parameters": fixture.parameters.shape.as_list(),
            "parameters_batch": parameters_batch.shape.as_list(),
        },
        "compile_first_call_seconds": compile_first_calls,
        "warm_start_call_seconds": warm_start_calls,
        "first_minus_warm_start_seconds": first_minus_warm,
        "repeated_call_seconds": repeated_calls,
        "warm_call_summary": warm_summaries,
        "descriptive_autodiff_row_loop_over_batch_native_analytical_median_ratio": autodiff_ratio,
        "descriptive_scalar_row_loop_over_batch_native_analytical_median_ratio": scalar_ratio,
        "diagnostics": {
            "batch_native_autodiff_qr_score": {
                "timed_method": False,
                "jit_compile": False,
                "elapsed_seconds": _probe_elapsed,
                "all_finite": batched_static_probe_finite,
                "value_shape": probe_output["value_shape"],
                "score_shape": probe_output["score_shape"],
                "score_head": probe_output["score_flat"][: min(5, batch_size * parameter_count)],
                "reason_not_timed": (
                    "Phase 4 repairs correctness; Phase 5 owns timing-boundary migration."
                ),
            },
        },
        "outputs": {
            name: {
                "value": output["value"],
                "score_head": output["score_flat"][: min(5, batch_size * parameter_count)],
                "score_tail": output["score_flat"][-min(5, batch_size * parameter_count) :],
                "devices": output["devices"],
                "value_shape": output["value_shape"],
                "score_shape": output["score_shape"],
            }
            for name, output in outputs.items()
        },
        "agreement": {
            "all_finite": all_finite,
            "shape_check_passed": shape_check_passed,
            "parity_passed": parity_passed,
            "value_abs_tolerance": value_tolerance,
            "score_abs_tolerance": score_tolerance,
            "batch_vs_scalar_value_max_abs_residual": batch_scalar_value_residual,
            "batch_vs_scalar_score_max_abs_residual": batch_scalar_score_residual,
            "batch_vs_autodiff_value_max_abs_residual": batch_autodiff_value_residual,
            "batch_vs_autodiff_score_max_abs_residual": batch_autodiff_score_residual,
            "score_relative_residual": score_relative_residual,
        },
    }


def _select_device(requested: str) -> tuple[str, dict[str, Any]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    if requested == "auto":
        selected = "/GPU:0" if logical_gpus else "/CPU:0"
    elif requested == "gpu":
        if not logical_gpus:
            raise RuntimeError("requested GPU benchmark but no logical GPU is visible")
        selected = "/GPU:0"
    elif requested == "cpu":
        selected = "/CPU:0"
    else:
        selected = requested
    return selected, {
        "requested_device": requested,
        "selected_device": selected,
        "physical_gpus": [device.name for device in physical_gpus],
        "logical_gpus": [device.name for device in logical_gpus],
        "cpu_only_exception": selected.upper().startswith("/CPU"),
        "trust_basis": (
            "owner_designated_managed_session_visible_gpu_trusted"
            if selected.upper().startswith("/GPU")
            else "cpu_debug_or_reference_exception"
        ),
    }


def _configure_cpu_threads(cpu_threads: int | None) -> dict[str, Any]:
    if cpu_threads is not None:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
            intra_status = "set"
        except RuntimeError as exc:
            intra_status = f"runtime_already_initialized: {exc}"
        try:
            tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
            inter_status = "set"
        except RuntimeError as exc:
            inter_status = f"runtime_already_initialized: {exc}"
    else:
        intra_status = "default"
        inter_status = "default"
    return {
        "requested_cpu_threads": cpu_threads,
        "tf_intra_op_parallelism_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "tf_inter_op_parallelism_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "intra_op_set_status": intra_status,
        "inter_op_set_status": inter_status,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "tf_num_intraop_threads_env": os.environ.get("TF_NUM_INTRAOP_THREADS", "UNSET"),
        "tf_num_interop_threads_env": os.environ.get("TF_NUM_INTEROP_THREADS", "UNSET"),
    }


def _manifest(
    args: argparse.Namespace,
    device_manifest: dict[str, Any],
    cpu_thread_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(REPO_ROOT),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": _run_text(["git", "status", "--short"]),
        "plan_path": args.plan_path,
        "json_path": args.output_json,
        "markdown_path": args.output_md,
        "dimensions": args.dimensions,
        "parameter_counts": args.parameter_counts,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "repeats": args.repeats,
        "jit_compile": args.jit_compile,
        "requested_dtype": args.dtype,
        "isolate_each_row": args.isolate_each_row,
        "row_subprocess_timeout_seconds": args.row_subprocess_timeout_seconds,
        "benchmark_methods": [
            "batch_native_analytical_qr_score",
            "scalar_analytical_row_loop",
            "autodiff_row_loop_qr_score",
        ],
        "autodiff_value_backend": "scalar_while_loop_row_loop",
        "autodiff_execution": "compiled_static_batch_row_loop",
        "batch_native_autodiff_gradient_status": (
            "corrected_phase4_not_timed_in_historical_grid"
        ),
        "device_manifest": device_manifest,
        "cpu_thread_manifest": cpu_thread_manifest,
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "xla_flags": os.environ.get("XLA_FLAGS", "UNSET"),
    }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6g}"


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    manifest = payload["manifest"]
    rows = payload["rows"]
    lines = [
        "# Kalman QR Parameter Count Scaling Result",
        "",
        f"- JSON artifact: `{manifest['json_path']}`",
        f"- Plan: `{manifest['plan_path']}`",
        f"- Command: `{manifest['command']}`",
        f"- Device: `{manifest['device_manifest']['selected_device']}`",
        f"- JIT compile: `{manifest['jit_compile']}`",
        f"- Requested dtype: `{manifest['requested_dtype']}`",
        f"- Batch size: `{manifest['batch_size']}`",
        f"- CPU threads: `{manifest['cpu_threads']}`",
        f"- TF32 execution enabled: `{manifest['tf32_execution_enabled']}`",
        f"- Benchmark methods: `{manifest['benchmark_methods']}`",
        f"- Autodiff value backend: `{manifest['autodiff_value_backend']}`",
        f"- Autodiff execution: `{manifest['autodiff_execution']}`",
        f"- Trust basis: `{manifest['device_manifest']['trust_basis']}`",
        "",
        "## Execution Note",
        "",
        (
            "All measured arms are compiled TensorFlow functions when "
            "`jit_compile=True`. The batch-native analytical arm calls "
            "`tf_qr_sqrt_kalman_score_batched_static`; the scalar comparator "
            "loops over batch rows inside one compiled function and calls the "
            "scalar analytical score; the autodiff comparator loops over batch "
            "rows inside one compiled function and differentiates the scalar QR "
            "value for each row. The "
            "first timed call is compile+first-call, the second timed call is "
            "the first warm-start call, and repeated calls provide the "
            "warm-call summary. The first-minus-warm value is explanatory only "
            "and is not a pure compiler-only measurement."
        ),
        "",
        (
            "Requested dtype is checked against observed analytical/autodiff "
            "value and score tensor dtypes. A mismatch fails the row parity "
            "screen. TF32 mode is reported separately and is not treated as the "
            "requested tensor dtype."
        ),
        "",
        (
            "Transition and observation matrices are lower triangular. "
            "Covariance inputs are SPD matrices formed from lower-triangular "
            "factors because the public Kalman API consumes covariance matrices, "
            "not covariance factors."
        ),
        "",
        "## Decision Table",
        "",
        "| Field | Status | Notes |",
        "| --- | --- | --- |",
        (
            "| Decision | `DESCRIPTIVE_TIMING_RECORDED` | "
            "No default, HMC, or scientific promotion claim. |"
        ),
        (
            f"| Primary criterion | `{payload['summary']['all_applicable_rows_parity_passed']}` | "
            "Finite outputs, requested/observed dtype match, and analytical/autodiff value-score parity for applicable rows. |"
        ),
        "| Veto diagnostics | `see rows` | Nonfinite outputs or parity failure invalidates a row timing ratio. |",
        "| Applicability | `capacity_checked` | Rows above independent lower-triangular slot capacity are marked N/A. |",
        "| Main uncertainty | `single-run wall timing` | Repeats are descriptive and not a statistical ranking. |",
        (
            "| Not concluded | `no promotion` | No HMC readiness, posterior correctness, "
            "or universal speed superiority. |"
        ),
        "",
        "## Timing Table",
        "",
        (
            "| dims `(n,m)` | params | batch | batch-native compile+first s | "
            "batch-native warm-start s | batch-native warm median s | "
            "scalar row-loop warm median s | autodiff row-loop warm median s | "
            "autodiff row-loop / batch-native warm | scalar row-loop / batch-native warm | "
            "observed dtypes | batch/autodiff score max abs residual | parity |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        dims = f"({row.get('state_dim')},{row.get('observation_dim')})"
        if not row.get("applicable", True):
            lines.append(
                "| "
                f"{dims} | {row.get('parameter_count')} | {row.get('batch_size', manifest.get('batch_size', 'N/A'))} | "
                "N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `N/A capacity` |"
            )
            continue
        if "error" in row:
            lines.append(
                "| "
                f"{dims} | {row.get('parameter_count')} | {row.get('batch_size', manifest.get('batch_size', 'N/A'))} | "
                "N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | "
                f"`error: {row['error']['type']}` |"
            )
            continue
        batch_summary = row["warm_call_summary"]["batch_native_analytical_qr_score"]
        scalar_summary = row["warm_call_summary"]["scalar_analytical_row_loop"]
        autodiff_summary = row["warm_call_summary"]["autodiff_row_loop_qr_score"]
        autodiff_ratio = row[
            "descriptive_autodiff_row_loop_over_batch_native_analytical_median_ratio"
        ]
        scalar_ratio = row[
            "descriptive_scalar_row_loop_over_batch_native_analytical_median_ratio"
        ]
        observed_dtypes = (
            f"value={row['observed_value_dtypes']}; "
            f"score={row['observed_score_dtypes']}"
        )
        lines.append(
            "| "
            f"{dims} | {row['parameter_count']} | {row['batch_size']} "
            f"| {_format_seconds(row['compile_first_call_seconds']['batch_native_analytical_qr_score'])} "
            f"| {_format_seconds(row['warm_start_call_seconds']['batch_native_analytical_qr_score'])} "
            f"| {_format_seconds(batch_summary['median_seconds'])} "
            f"| {_format_seconds(scalar_summary['median_seconds'])} "
            f"| {_format_seconds(autodiff_summary['median_seconds'])} "
            f"| {_format_seconds(autodiff_ratio)} "
            f"| {_format_seconds(scalar_ratio)} "
            f"| `{observed_dtypes}` "
            f"| {row['agreement']['batch_vs_autodiff_score_max_abs_residual']:.3e} "
            f"| `{row['agreement']['parity_passed']}` |"
        )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| Evidence class | Status |",
            "| --- | --- |",
            (
                f"| Hard veto screen | `"
                f"{payload['summary']['all_applicable_rows_parity_passed']}` |"
            ),
            "| Statistically supported ranking | `not assessed` |",
            "| Descriptive-only differences | `compile+first, warm-start, warm medians, and ratios only` |",
            "| Default-readiness | `not assessed` |",
            "| Next evidence needed | `replicate runs and broaden model families if making a speed claim` |",
            "",
            "## Run Manifest",
            "",
            f"- Git commit: `{manifest['git_commit']}`",
            f"- TensorFlow: `{manifest['tensorflow_version']}`",
            f"- Requested dtype: `{manifest['requested_dtype']}`",
            f"- Batch size: `{manifest['batch_size']}`",
            f"- CPU thread manifest: `{manifest['cpu_thread_manifest']}`",
            f"- TF32 execution enabled: `{manifest['tf32_execution_enabled']}`",
            f"- Physical GPUs: `{manifest['device_manifest']['physical_gpus']}`",
            f"- Logical GPUs: `{manifest['device_manifest']['logical_gpus']}`",
            f"- CUDA_VISIBLE_DEVICES: `{manifest['cuda_visible_devices']}`",
            "- Data version: `deterministic synthetic lower-triangular LGSSM fixture generated by this script`",
            "- Random seeds: `N/A deterministic fixture`",
            "",
            "## Post-Run Red Team",
            "",
            (
                "The strongest alternative explanation is device/runtime noise or "
                "XLA compile/runtime behavior specific to this synthetic lower-"
                "triangular parameterization. A result that would overturn a "
                "speed interpretation is a replicated run on the target deployment "
                "device where warm median ratios change materially or parity fails."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--parameter-counts", nargs="+", type=int, default=PARAMETER_COUNTS)
    parser.add_argument("--timesteps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Set TensorFlow intra/inter-op and common CPU thread env vars for CPU scaling runs.",
    )
    parser.add_argument("--device", default="auto", help="auto, gpu, cpu, or explicit TF device")
    parser.add_argument("--dtype", choices=sorted(SUPPORTED_DTYPES), default="float64")
    parser.add_argument("--plan-path", default=PLAN_PATH)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_MD)
    parser.add_argument(
        "--phase2-fixture-diagnostic",
        action="store_true",
        help="Run only the reviewed GPU-hidden, non-JIT Phase 2 fixture/GraphDef gate.",
    )
    parser.add_argument(
        "--phase2-log-path",
        default="/tmp/kalman_qr_phase2_fixture/phase2_graphdef.log",
        help="Predeclared external log path recorded by the Phase 2 diagnostic.",
    )
    parser.add_argument(
        "--phase3-parameter-graph-diagnostic",
        action="store_true",
        help="Trace only the reviewed GPU-hidden, non-JIT Phase 3 P-axis graph gate.",
    )
    parser.add_argument(
        "--phase3-log-path",
        default="/tmp/kalman_qr_phase3_vectorization/phase3_parameter_graphdef.log",
        help="Predeclared external log path recorded by the Phase 3 diagnostic.",
    )
    parser.add_argument(
        "--phase4-autodiff-diagnostic",
        action="store_true",
        help="Run only the reviewed GPU-hidden non-JIT Phase 4 autodiff gate.",
    )
    parser.add_argument(
        "--phase4-autodiff-xla-smoke",
        action="store_true",
        help="Run only the reviewed tiny GPU-hidden CPU-XLA Phase 4 smoke.",
    )
    parser.add_argument(
        "--phase4-log-path",
        default=PHASE4_DIAGNOSTIC_LOG,
        help="Predeclared external log path recorded by a Phase 4 diagnostic.",
    )
    parser.add_argument(
        "--phase6-import-discovery",
        action="store_true",
        help="Import the Phase 6 execution closure without fixture, trace, or execution.",
    )
    parser.add_argument(
        "--phase6-trace-only",
        action="store_true",
        help="Trace exactly one Phase 6 primary method without invoking it or XLA.",
    )
    parser.add_argument("--method", choices=benchmark_contract.METHOD_IDS, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--progress-journal", default=None)
    parser.add_argument("--source-fingerprint", default=None)
    parser.add_argument("--config-fingerprint", default=None)
    parser.add_argument("--runtime-fingerprint", default=None)
    parser.add_argument("--fixture-fingerprint", default=None)
    parser.add_argument("--schedule-fingerprint", default=None)
    parser.add_argument("--resume-key", default=None)
    parser.add_argument("--phase6-authority-snapshot", type=Path)
    parser.add_argument("--phase6-dependency-before", type=Path)
    parser.add_argument("--phase6-dependency-after", type=Path)
    tf32_group = parser.add_mutually_exclusive_group()
    tf32_group.add_argument("--tf32-enabled", dest="tf32_enabled", action="store_true")
    tf32_group.add_argument("--no-tf32", dest="tf32_enabled", action="store_false")
    parser.set_defaults(tf32_enabled=None)
    parser.add_argument(
        "--isolate-each-row",
        action="store_true",
        help=(
            "Run each dimension/parameter-count row in a fresh child process "
            "and aggregate results to avoid cumulative XLA codegen memory growth."
        ),
    )
    parser.add_argument(
        "--row-subprocess-timeout-seconds",
        type=float,
        default=1800.0,
        help="Per-row timeout used only with --isolate-each-row.",
    )
    parser.add_argument(
        "--flush-after-row",
        action="store_true",
        help="Write JSON/Markdown artifacts after each completed grid row.",
    )
    jit_group = parser.add_mutually_exclusive_group()
    jit_group.add_argument("--jit-compile", dest="jit_compile", action="store_true")
    jit_group.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.set_defaults(jit_compile=True)
    return parser.parse_args()


def _text_tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:]


def _isolated_child_command(
    args: argparse.Namespace,
    *,
    dimension: int,
    parameter_count: int,
    row_json: Path,
    row_md: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dimensions",
        str(dimension),
        "--parameter-counts",
        str(parameter_count),
        "--timesteps",
        str(args.timesteps),
        "--repeats",
        str(args.repeats),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--plan-path",
        str(args.plan_path),
        "--output-json",
        str(row_json),
        "--output-md",
        str(row_md),
    ]
    if args.cpu_threads is not None:
        command.extend(["--cpu-threads", str(args.cpu_threads)])
    command.append("--jit-compile" if args.jit_compile else "--no-jit-compile")
    return command


def _run_isolated_grid(
    args: argparse.Namespace,
    *,
    selected_device: str,
    device_manifest: dict[str, Any],
    cpu_thread_manifest: dict[str, Any],
    output_json: Path,
    output_md: Path,
) -> int:
    manifest = _manifest(args, device_manifest, cpu_thread_manifest)
    manifest["selected_device_for_parent"] = selected_device
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema": "kalman_qr_parameter_count_scaling_v1",
        "manifest": manifest,
        "summary": {
            "all_applicable_rows_parity_passed": False,
            "applicable_row_count": 0,
            "inapplicable_row_count": 0,
            "wall_time_seconds": None,
            "run_status": "running",
            "nonclaims": [
                "descriptive timing only",
                "no HMC readiness claim",
                "no posterior correctness claim",
                "no production default change",
                "no statistically supported ranking",
            ],
        },
        "rows": rows,
    }

    scratch_root = Path("/tmp") / f"kalman_qr_parameter_count_rows_{os.getpid()}"
    scratch_root.mkdir(parents=True, exist_ok=True)

    def flush(run_status: str = "running") -> None:
        applicable_rows = [row for row in rows if row.get("applicable", True)]
        payload["summary"]["run_status"] = run_status
        payload["summary"]["wall_time_seconds"] = time.perf_counter() - started
        payload["summary"]["applicable_row_count"] = len(applicable_rows)
        payload["summary"]["inapplicable_row_count"] = len(rows) - len(applicable_rows)
        payload["summary"]["all_applicable_rows_parity_passed"] = (
            bool(applicable_rows)
            and all(row.get("agreement", {}).get("parity_passed", False) for row in applicable_rows)
        )
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        _write_markdown(payload, output_md)

    for dimension in args.dimensions:
        for parameter_count in args.parameter_counts:
            row_json = scratch_root / f"row_n{dimension}_p{parameter_count}.json"
            row_md = scratch_root / f"row_n{dimension}_p{parameter_count}.md"
            command = _isolated_child_command(
                args,
                dimension=dimension,
                parameter_count=parameter_count,
                row_json=row_json,
                row_md=row_md,
            )
            row_started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.row_subprocess_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                rows.append(
                    {
                        "state_dim": dimension,
                        "observation_dim": dimension,
                        "timesteps": args.timesteps,
                        "parameter_count": parameter_count,
                        "batch_size": args.batch_size,
                        "parameter_capacity": parameter_capacity(dimension),
                        "applicable": True,
                        "jit_compile": args.jit_compile,
                        "device_name": selected_device,
                        "requested_dtype": args.dtype,
                        "row_process": {
                            "isolated": True,
                            "timeout_seconds": args.row_subprocess_timeout_seconds,
                            "elapsed_seconds": time.perf_counter() - row_started,
                            "command": " ".join(command),
                            "stdout_tail": _text_tail(exc.stdout or ""),
                            "stderr_tail": _text_tail(exc.stderr or ""),
                        },
                        "error": {
                            "type": "TimeoutExpired",
                            "message": str(exc),
                        },
                    }
                )
                flush()
                continue

            child_rows: list[dict[str, Any]] = []
            if row_json.exists():
                try:
                    child_payload = json.loads(row_json.read_text(encoding="utf-8"))
                    child_rows = child_payload.get("rows", [])
                except Exception as exc:  # pragma: no cover - corrupt child artifact.
                    rows.append(
                        {
                            "state_dim": dimension,
                            "observation_dim": dimension,
                            "timesteps": args.timesteps,
                            "parameter_count": parameter_count,
                            "batch_size": args.batch_size,
                            "parameter_capacity": parameter_capacity(dimension),
                            "applicable": True,
                            "jit_compile": args.jit_compile,
                            "device_name": selected_device,
                            "requested_dtype": args.dtype,
                            "row_process": {
                                "isolated": True,
                                "returncode": completed.returncode,
                                "elapsed_seconds": time.perf_counter() - row_started,
                                "command": " ".join(command),
                                "stdout_tail": _text_tail(completed.stdout),
                                "stderr_tail": _text_tail(completed.stderr),
                            },
                            "error": {
                                "type": type(exc).__name__,
                                "message": f"failed to read child row artifact: {exc}",
                            },
                        }
                    )
                    flush()
                    continue

            if child_rows:
                row = child_rows[0]
                row["row_process"] = {
                    "isolated": True,
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.perf_counter() - row_started,
                    "command": " ".join(command),
                    "stdout_tail": _text_tail(completed.stdout),
                    "stderr_tail": _text_tail(completed.stderr),
                }
                rows.append(row)
            else:
                rows.append(
                    {
                        "state_dim": dimension,
                        "observation_dim": dimension,
                        "timesteps": args.timesteps,
                        "parameter_count": parameter_count,
                        "batch_size": args.batch_size,
                        "parameter_capacity": parameter_capacity(dimension),
                        "applicable": True,
                        "jit_compile": args.jit_compile,
                        "device_name": selected_device,
                        "requested_dtype": args.dtype,
                        "row_process": {
                            "isolated": True,
                            "returncode": completed.returncode,
                            "elapsed_seconds": time.perf_counter() - row_started,
                            "command": " ".join(command),
                            "stdout_tail": _text_tail(completed.stdout),
                            "stderr_tail": _text_tail(completed.stderr),
                        },
                        "error": {
                            "type": "MissingChildRows",
                            "message": "child process did not produce a row artifact",
                        },
                    }
                )
            flush()

    flush("complete")
    print(
        json.dumps(
            {
                "json": str(output_json),
                "markdown": str(output_md),
                "all_applicable_rows_parity_passed": payload["summary"][
                    "all_applicable_rows_parity_passed"
                ],
                "applicable_row_count": payload["summary"]["applicable_row_count"],
                "inapplicable_row_count": payload["summary"]["inapplicable_row_count"],
                "wall_time_seconds": payload["summary"]["wall_time_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    applicable_rows = [row for row in rows if row.get("applicable", True)]
    has_error = any("error" in row for row in applicable_rows)
    if has_error:
        return 1
    if not applicable_rows:
        return 0
    return 0 if payload["summary"]["all_applicable_rows_parity_passed"] else 1


def main() -> int:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive when provided")
    if args.row_subprocess_timeout_seconds <= 0:
        raise ValueError("--row-subprocess-timeout-seconds must be positive")
    if any(dimension <= 0 for dimension in args.dimensions):
        raise ValueError("--dimensions must be positive")
    if any(parameter_count <= 0 for parameter_count in args.parameter_counts):
        raise ValueError("--parameter-counts must be positive")
    diagnostic_modes = (
        args.phase2_fixture_diagnostic,
        args.phase3_parameter_graph_diagnostic,
        args.phase4_autodiff_diagnostic,
        args.phase4_autodiff_xla_smoke,
        args.phase6_import_discovery,
        args.phase6_trace_only,
    )
    if sum(bool(mode) for mode in diagnostic_modes) > 1:
        raise ValueError("diagnostic modes are mutually exclusive")
    if args.phase6_import_discovery:
        if args.method is not None or args.isolate_each_row:
            raise ValueError("Phase 6 import discovery cannot run a method or grid")
        return run_phase6_import_discovery(args)
    if args.phase6_trace_only:
        if args.isolate_each_row:
            raise ValueError("Phase 6 trace-only mode cannot run the legacy isolated grid")
        return run_phase6_trace_only(args)
    if args.phase4_autodiff_diagnostic or args.phase4_autodiff_xla_smoke:
        if args.method is not None or args.isolate_each_row:
            raise ValueError("Phase 4 diagnostic modes cannot be combined with method or grid execution")
        if args.phase4_autodiff_diagnostic:
            return run_phase4_autodiff_diagnostic(args)
        return run_phase4_autodiff_xla_smoke(args)
    if args.phase2_fixture_diagnostic:
        if args.method is not None or args.isolate_each_row or args.phase3_parameter_graph_diagnostic:
            raise ValueError(
                "--phase2-fixture-diagnostic cannot be combined with method or grid execution"
            )
        return run_phase2_fixture_diagnostic(args)
    if args.phase3_parameter_graph_diagnostic:
        if args.method is not None or args.isolate_each_row:
            raise ValueError(
                "--phase3-parameter-graph-diagnostic cannot be combined with method or grid execution"
            )
        return run_phase3_parameter_graph_diagnostic(args)
    dtype = _resolve_dtype(args.dtype)

    if args.method is not None:
        required_child_fields = {
            "case_id": args.case_id,
            "attempt_id": args.attempt_id,
            "progress_journal": args.progress_journal,
            "source_fingerprint": args.source_fingerprint,
            "config_fingerprint": args.config_fingerprint,
            "runtime_fingerprint": args.runtime_fingerprint,
            "fixture_fingerprint": args.fixture_fingerprint,
            "schedule_fingerprint": args.schedule_fingerprint,
            "resume_key": args.resume_key,
        }
        missing = sorted(name for name, value in required_child_fields.items() if not value)
        if missing:
            raise ValueError(f"v4 method child is missing identity fields: {missing}")
        if len(args.dimensions) != 1 or len(args.parameter_counts) != 1:
            raise ValueError("v4 method child requires exactly one dimension and parameter count")
        if args.tf32_enabled is None:
            raise ValueError("v4 method child requires an explicit TF32 setting")
        if (args.phase6_dependency_before is None) != (
            args.phase6_dependency_after is None
        ):
            raise ValueError("Phase 6 dependency outputs must be provided together")

    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    cpu_thread_manifest = _configure_cpu_threads(args.cpu_threads)
    if args.tf32_enabled is not None:
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_enabled)
    selected_device, device_manifest = _select_device(args.device)
    if args.method is not None:
        record = benchmark_selected_method_case(
            args=args,
            dimension=args.dimensions[0],
            parameter_count=args.parameter_counts[0],
            device_name=selected_device,
            dtype=dtype,
        )
        record["device_manifest"] = device_manifest
        record["cpu_thread_manifest"] = cpu_thread_manifest
        if args.phase6_dependency_after is not None:
            benchmark_contract.durable_atomic_write_json(
                args.phase6_dependency_after.resolve(),
                benchmark_contract.repository_module_manifest(REPO_ROOT),
            )
        _write_v4_method_markdown(record, output_md)
        benchmark_contract.atomic_write_json(output_json, record)
        return 0 if record["state"] == "passed" else 1
    if args.isolate_each_row:
        return _run_isolated_grid(
            args,
            selected_device=selected_device,
            device_manifest=device_manifest,
            cpu_thread_manifest=cpu_thread_manifest,
            output_json=output_json,
            output_md=output_md,
        )

    manifest = _manifest(args, device_manifest, cpu_thread_manifest)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema": "kalman_qr_parameter_count_scaling_v1",
        "manifest": manifest,
        "summary": {
            "all_applicable_rows_parity_passed": False,
            "applicable_row_count": 0,
            "inapplicable_row_count": 0,
            "wall_time_seconds": None,
            "run_status": "running",
            "nonclaims": [
                "descriptive timing only",
                "no HMC readiness claim",
                "no posterior correctness claim",
                "no production default change",
                "no statistically supported ranking",
            ],
        },
        "rows": rows,
    }

    def flush(run_status: str = "running") -> None:
        applicable_rows = [row for row in rows if row.get("applicable", True)]
        payload["summary"]["run_status"] = run_status
        payload["summary"]["wall_time_seconds"] = time.perf_counter() - started
        payload["summary"]["applicable_row_count"] = len(applicable_rows)
        payload["summary"]["inapplicable_row_count"] = len(rows) - len(applicable_rows)
        payload["summary"]["all_applicable_rows_parity_passed"] = (
            bool(applicable_rows)
            and all(row.get("agreement", {}).get("parity_passed", False) for row in applicable_rows)
        )
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        _write_markdown(payload, output_md)

    for dimension in args.dimensions:
        for parameter_count in args.parameter_counts:
            try:
                rows.append(
                    benchmark_case(
                        dimension=dimension,
                        parameter_count=parameter_count,
                        timesteps=args.timesteps,
                        repeats=args.repeats,
                        batch_size=args.batch_size,
                        jit_compile=args.jit_compile,
                        device_name=selected_device,
                        dtype=dtype,
                    )
                )
            except KeyboardInterrupt:
                flush("interrupted")
                raise
            except Exception as exc:  # pragma: no cover - runtime artifact path.
                rows.append(
                    {
                        "state_dim": dimension,
                        "observation_dim": dimension,
                        "timesteps": args.timesteps,
                        "parameter_count": parameter_count,
                        "batch_size": args.batch_size,
                        "parameter_capacity": parameter_capacity(dimension),
                        "applicable": True,
                        "jit_compile": args.jit_compile,
                        "device_name": selected_device,
                        "requested_dtype": dtype.name,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
            if args.flush_after_row:
                flush()

    flush("complete")
    print(
        json.dumps(
            {
                "json": str(output_json),
                "markdown": str(output_md),
                "all_applicable_rows_parity_passed": payload["summary"][
                    "all_applicable_rows_parity_passed"
                ],
                "applicable_row_count": payload["summary"]["applicable_row_count"],
                "inapplicable_row_count": payload["summary"]["inapplicable_row_count"],
                "wall_time_seconds": payload["summary"]["wall_time_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    applicable_rows = [row for row in rows if row.get("applicable", True)]
    has_error = any("error" in row for row in applicable_rows)
    if has_error:
        return 1
    if not applicable_rows:
        return 0
    return 0 if payload["summary"]["all_applicable_rows_parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
