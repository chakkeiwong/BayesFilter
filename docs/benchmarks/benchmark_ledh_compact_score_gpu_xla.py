"""Trusted GPU/XLA harness for per-seed compact LEDH score evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import inspect
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.ledh_fd_policy import (
    LEDH_FD_BASE_RELATIVE_TOLERANCE,
    LEDH_FD_DENOMINATOR,
    LEDH_FD_DIAGNOSTIC_SCOPE,
    LEDH_FD_PASS_RULE,
    LEDH_FD_POLICY_ID,
    LEDH_FD_STATISTICAL_STATUS,
    coordinate_central_difference_step,
    evaluate_ledh_fd_policy,
    ledh_fd_step_policy_metadata,
    validate_declared_ledh_fd_policy,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from bayesfilter.highdim.ledh_historical_raw_policy import (
    require_historical_raw_diagnostic_opt_in,
)
from docs.benchmarks import benchmark_ledh_same_target_actual_sv_score as actual_sv
from docs.benchmarks import benchmark_ledh_same_target_fixed_sir_score as fixed_sir
from docs.benchmarks import benchmark_ledh_same_target_generalized_sv_score as generalized_sv
from docs.benchmarks import benchmark_ledh_same_target_ksc_sv_score as ksc_sv
from docs.benchmarks import (
    benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter as lgssm,
)
from docs.benchmarks import benchmark_ledh_same_target_predator_prey_score as predator_prey


GPU_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
SCHEMA_VERSION = "bayesfilter.ledh.compact_score_gpu_xla.v5"
LEGACY_SCHEMA_VERSION = "bayesfilter.ledh.compact_score_gpu_xla.v1"
CANONICAL_TARGETS_PATH = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase1-canonical-targets-2026-07-11.json"
)
CANONICAL_TARGETS_SHA256 = (
    "1cc83076b491b7c059fadbef85cacbb138c974a39502f5418d9018c17ef8fec8"
)
P1A_RECEIPT_PATH = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase1-p1a-gate-receipt-2026-07-11.json"
)
P1A_RECEIPT_SHA256 = (
    "41fafd0eed4abb10a002d525ccb10e3544a2f52ce81f7b0e76c1d57e040edaef"
)
P1B_RECEIPT_PATH = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase1-p1b-gate-receipt-2026-07-11.json"
)
P1B_RECEIPT_SHA256 = (
    "af0547a53097cb5af6579c8ae993c1868dcd75a570dcfe3e08c2248c57dd1718"
)
PHASE1_ENTRY_AMENDMENT_PATH = (
    "docs/plans/bayesfilter-complete-highdim-leaderboard-"
    "phase1-entry-amendment-2026-07-12.md"
)
PHASE1_ENTRY_AMENDMENT_SHA256 = (
    "cdf6439185a65c4121b51d179179ee8b331ec809c44e6310465878562efb8869"
)
PLAN_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md"
HISTORICAL_RESULT_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md"
FD_POLICY_CORRECTION_PLAN_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-subplan-2026-07-11.md"
RESULT_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md"
EXECUTION_MANIFEST_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-execution-manifest-2026-07-10.md"
GATE_A_RESULT_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-a-harness-result-2026-07-10.md"
EXACT_COMMANDS_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json"
GATE_B_REVIEW_PATH = "docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-a-manifest-codex-substitute-review-iter2-2026-07-10.md"
GATE_B_REPAIR_REVIEW_PATH = "docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-b-cross-row-extraction-repair-codex-substitute-review-2026-07-11.md"
GATE_B_RESULT_PATH = "docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-b-result-2026-07-11.md"
GATE_B_RESULT_REVIEW_PATH = "docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-b-result-codex-substitute-review-iter2b-2026-07-11.md"
ROOT_CAUSE_REPAIR_PLAN_PATH = (
    "docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-subplan-2026-07-11.md"
)
ROOT_CAUSE_REPAIR_COMMANDS_PATH = (
    "docs/plans/ledh-predator-generalized-fd-root-cause-repair-gpu-commands-2026-07-11.json"
)
COMPLETE_HIGHDIM_COMMAND_BUILDER_PATH = (
    "docs/benchmarks/build_complete_highdim_ledh_phase2_phase3_command_manifest.py"
)
COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH = (
    "docs/plans/complete-highdim-leaderboard-ledh-phase2-phase3-"
    "exact-commands-repair1-2026-07-12.json"
)
COMPLETE_HIGHDIM_PHASE2_EXECUTION_AUTHORITY_PATH = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase2-execution-authority-repair1-2026-07-12.json"
)
COMPLETE_HIGHDIM_PHASE3_EXECUTION_AUTHORITY_PATH = (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase3-execution-authority-repair1-2026-07-12.json"
)
FULL_ROW_BATCH_SEEDS = (81120, 81121, 81122, 81123, 81124)
MEMORY_BUDGET_MIB = 14000.0
JIT_COMPILE = True
COMPACT_GRADIENT_MODE = "manual_streaming_finite_sinkhorn_stopped_scale_keys"
GPU_EXECUTION_AUTHORIZED = False
ROOT_CAUSE_REPAIR_GPU_EXECUTION_AUTHORIZED = True


@dataclass(frozen=True)
class RowSpec:
    name: str
    module: Any
    row_id: str
    source_value_artifact: str
    truth_theta: tuple[float, ...]
    parameter_names: tuple[str, ...]
    score_route: str
    full_time_steps: int
    full_num_particles: int
    row_chunk_size: int
    col_chunk_size: int
    particle_chunk_size: int
    sinkhorn_epsilon: float
    flow_observation_variance: float | None
    legacy_module_fd_step: float


ROW_SPECS = {
    "lgssm": RowSpec(
        name="lgssm",
        module=lgssm,
        row_id=lgssm.ROW_ID,
        source_value_artifact=(
            "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"
        ),
        truth_theta=tuple(lgssm.TRUTH_THETA),
        parameter_names=tuple(lgssm.PARAMETER_NAMES),
        score_route=lgssm.COMPACT_SCORE_ROUTE_ID,
        full_time_steps=50,
        full_num_particles=10000,
        row_chunk_size=512,
        col_chunk_size=512,
        particle_chunk_size=256,
        sinkhorn_epsilon=lgssm.FULL_ROW_SINKHORN_EPSILON,
        flow_observation_variance=None,
        legacy_module_fd_step=1.0e-3,
    ),
    "fixed-sir": RowSpec(
        name="fixed-sir",
        module=fixed_sir,
        row_id=fixed_sir.FIXED_SIR_AUSTRIA_ROW_ID,
        source_value_artifact="docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json",
        truth_theta=(0.0, 0.0, 0.0),
        parameter_names=tuple(fixed_sir.PARAMETER_NAMES),
        score_route=fixed_sir.FIXED_SIR_COMPACT_SCORE_ROUTE_ID,
        full_time_steps=20,
        full_num_particles=10000,
        row_chunk_size=1024,
        col_chunk_size=1024,
        particle_chunk_size=512,
        sinkhorn_epsilon=1.0,
        flow_observation_variance=None,
        legacy_module_fd_step=1.0e-3,
    ),
    "predator-prey": RowSpec(
        name="predator-prey",
        module=predator_prey,
        row_id=predator_prey.PREDATOR_PREY_ROW_ID,
        source_value_artifact="docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json",
        truth_theta=tuple(predator_prey.TRUTH_THETA),
        parameter_names=tuple(predator_prey.PARAMETER_NAMES),
        score_route=predator_prey.PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID,
        full_time_steps=20,
        full_num_particles=10000,
        row_chunk_size=512,
        col_chunk_size=512,
        particle_chunk_size=512,
        sinkhorn_epsilon=1.0,
        flow_observation_variance=None,
        legacy_module_fd_step=1.0e-4,
    ),
    "actual-sv": RowSpec(
        name="actual-sv",
        module=actual_sv,
        row_id=actual_sv.ACTUAL_SV_ROW_ID,
        source_value_artifact="docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json",
        truth_theta=tuple(actual_sv.TRUTH_THETA),
        parameter_names=tuple(actual_sv.PARAMETER_NAMES),
        score_route=actual_sv.ACTUAL_SV_COMPACT_SCORE_ROUTE_ID,
        full_time_steps=1000,
        full_num_particles=10000,
        row_chunk_size=512,
        col_chunk_size=512,
        particle_chunk_size=512,
        sinkhorn_epsilon=1.0,
        flow_observation_variance=math.pi * math.pi / 2.0,
        legacy_module_fd_step=1.0e-4,
    ),
    "generalized-sv": RowSpec(
        name="generalized-sv",
        module=generalized_sv,
        row_id=generalized_sv.GENERALIZED_SV_ROW_ID,
        source_value_artifact="docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json",
        truth_theta=tuple(generalized_sv.TRUTH_THETA),
        parameter_names=tuple(generalized_sv.PARAMETER_NAMES),
        score_route=generalized_sv.GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID,
        full_time_steps=1008,
        full_num_particles=10000,
        row_chunk_size=512,
        col_chunk_size=512,
        particle_chunk_size=512,
        sinkhorn_epsilon=1.0,
        flow_observation_variance=2.0,
        legacy_module_fd_step=1.0e-4,
    ),
    "ksc-sv": RowSpec(
        name="ksc-sv",
        module=ksc_sv,
        row_id=ksc_sv.KSC_SV_ROW_ID,
        source_value_artifact="docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json",
        truth_theta=tuple(ksc_sv.TRUTH_THETA),
        parameter_names=tuple(ksc_sv.PARAMETER_NAMES),
        score_route=ksc_sv.KSC_SV_COMPACT_SCORE_ROUTE_ID,
        full_time_steps=1000,
        full_num_particles=10000,
        row_chunk_size=512,
        col_chunk_size=512,
        particle_chunk_size=512,
        sinkhorn_epsilon=1.0,
        flow_observation_variance=math.pi * math.pi / 2.0,
        legacy_module_fd_step=1.0e-4,
    ),
}


def _parse_int_csv(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("expected at least one seed")
    return output


def _parse_path_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    output = [item.strip() for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("expected at least one shard path")
    return output


def _parse_args(
    argv: Sequence[str] | None = None,
    *,
    validate: bool = True,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--row", choices=tuple(ROW_SPECS), required=True)
    parser.add_argument("--stage", choices=("score-only", "fd-only", "aggregate"), required=True)
    parser.add_argument("--batch-seeds", default="81120")
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--transport-policy", choices=("active-all", "active-odd", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=None)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=None)
    parser.add_argument("--col-chunk-size", type=int, default=None)
    parser.add_argument("--particle-chunk-size", type=int, default=None)
    parser.add_argument("--transport-plan-mode", choices=("streaming",), default="streaming")
    parser.add_argument("--transport-ad-mode", choices=("full",), default="full")
    parser.add_argument("--transport-gradient-mode", choices=(COMPACT_GRADIENT_MODE,), default=COMPACT_GRADIENT_MODE)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--tf32-mode", choices=("enabled",), default="enabled")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--score-reference-json", default=None)
    parser.add_argument("--score-shards", default=None)
    parser.add_argument("--fd-shards", default=None)
    parser.add_argument("--source-value-artifact", default=None)
    parser.add_argument("--memory-budget-mib", type=float, default=MEMORY_BUDGET_MIB)
    parser.add_argument("--command-timeout-seconds", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--historical-raw-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    spec = ROW_SPECS[args.row]
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    args.time_steps = spec.full_time_steps if args.time_steps is None else args.time_steps
    args.num_particles = spec.full_num_particles if args.num_particles is None else args.num_particles
    args.sinkhorn_epsilon = (
        spec.sinkhorn_epsilon
        if args.sinkhorn_epsilon is None
        else args.sinkhorn_epsilon
    )
    args.row_chunk_size = spec.row_chunk_size if args.row_chunk_size is None else args.row_chunk_size
    args.col_chunk_size = spec.col_chunk_size if args.col_chunk_size is None else args.col_chunk_size
    args.particle_chunk_size = spec.particle_chunk_size if args.particle_chunk_size is None else args.particle_chunk_size
    args.flow_observation_variance = spec.flow_observation_variance
    # Row modules still require this argument for their standalone diagnostics;
    # the GPU FD stage uses the coordinate policy below, never this legacy value.
    args.fd_step = spec.legacy_module_fd_step
    args.score_shards = _parse_path_csv(args.score_shards)
    args.fd_shards = _parse_path_csv(args.fd_shards)
    args.source_value_artifact = args.source_value_artifact or spec.source_value_artifact
    if validate:
        _validate_args(args, spec)
    return args


def _validate_args(args: argparse.Namespace, spec: RowSpec) -> None:
    if args.jit_compile is not True:
        raise ValueError("production compact score harness requires jit_compile=True")
    if args.stage != "aggregate" and len(args.batch_seeds) != 1:
        raise ValueError("runtime score/FD shards require exactly one seed")
    if args.stage != "aggregate" and int(args.batch_seeds[0]) not in FULL_ROW_BATCH_SEEDS:
        raise ValueError("runtime shard seed must belong to the frozen full-row seed set")
    if args.stage == "aggregate" and tuple(args.batch_seeds) != FULL_ROW_BATCH_SEEDS:
        raise ValueError("aggregate requires the exact five full-row seeds")
    if args.stage == "fd-only" and not args.score_reference_json:
        raise ValueError("fd-only requires --score-reference-json")
    if args.stage == "aggregate" and (not args.score_shards or not args.fd_shards):
        raise ValueError("aggregate requires score and FD shard paths")
    if args.stage != "aggregate" and (args.score_shards or args.fd_shards):
        raise ValueError("raw shard stages do not accept aggregate shard lists")
    if args.time_steps <= 0 or args.time_steps > spec.full_time_steps:
        raise ValueError("time_steps exceeds the row contract")
    if args.num_particles <= 1:
        raise ValueError("num_particles must exceed one")
    for name in ("sinkhorn_iterations", "row_chunk_size", "col_chunk_size", "particle_chunk_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    expected_transport = (
        "active-all",
        10,
        spec.sinkhorn_epsilon,
        0.9,
        1.0e-3,
        spec.row_chunk_size,
        spec.col_chunk_size,
        spec.particle_chunk_size,
        "streaming",
        "full",
        COMPACT_GRADIENT_MODE,
    )
    observed_transport = (
        args.transport_policy,
        int(args.sinkhorn_iterations),
        float(args.sinkhorn_epsilon),
        float(args.annealed_scaling),
        float(args.annealed_convergence_threshold),
        int(args.row_chunk_size),
        int(args.col_chunk_size),
        int(args.particle_chunk_size),
        args.transport_plan_mode,
        args.transport_ad_mode,
        args.transport_gradient_mode,
    )
    if observed_transport != expected_transport:
        raise ValueError("runtime evidence requires the frozen admitted transport policy")
    if float(args.memory_budget_mib) != MEMORY_BUDGET_MIB:
        raise ValueError("runtime evidence requires the frozen 14000 MiB score budget")
    if args.command_timeout_seconds is not None and args.command_timeout_seconds <= 0:
        raise ValueError("command timeout must be positive when declared")
    if args.source_value_artifact != spec.source_value_artifact:
        raise ValueError("runtime evidence requires the frozen admitted source value artifact")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("CPU scope cannot claim GPU output")
    if args.stage != "aggregate" and args.device_scope == "visible":
        if args.expect_device_kind != "gpu" or "GPU" not in args.device.upper():
            raise ValueError("visible trusted runtime requires an explicit GPU device expectation")
    require_historical_raw_diagnostic_opt_in(
        args, route_name="unified raw-barycentric LEDH score harness"
    )


def _configure_precision(spec: RowSpec, args: argparse.Namespace) -> dict[str, Any]:
    precision = spec.module._configure_precision(args)  # noqa: SLF001
    if args.tf32_mode != "enabled" or args.dtype != "float32":
        raise ValueError("production score precision must be float32 with TF32 enabled")
    return precision


def _configure_devices() -> tuple[list[str], list[str]]:
    physical = tf.config.list_physical_devices("GPU")
    for gpu in physical:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical = tf.config.list_logical_devices("GPU")
    return [str(item) for item in physical], [str(item) for item in logical]


def _gpu_memory_info() -> dict[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        return {"status": "unavailable"}


def _reset_gpu_memory_stats() -> bool:
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        return True
    except (ValueError, RuntimeError):
        return False


def _validate_devices(outputs: Sequence[tf.Tensor], expected: str) -> list[str]:
    devices = sorted({str(value.device) for value in outputs})
    token = "GPU" if expected == "gpu" else "CPU"
    if not devices or not all(token in device.upper() for device in devices):
        raise ValueError(f"expected {expected} outputs, got {devices}")
    return devices


def _materialize(outputs: Sequence[tf.Tensor]) -> None:
    for output in outputs:
        output.numpy()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@functools.lru_cache(maxsize=1)
def _canonical_target_sha256_by_row() -> dict[str, str]:
    path = ROOT / CANONICAL_TARGETS_PATH
    if _sha256(path) != CANONICAL_TARGETS_SHA256:
        raise ValueError("canonical target artifact hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("canonical target artifact rows must be a list")
    output = {
        str(row.get("row_id")): str(row.get("row_sha256"))
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = {spec.row_id for spec in ROW_SPECS.values()}
    if set(output) != expected or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in output.values()
    ):
        raise ValueError("canonical target row identity set mismatch")
    return output


def _canonical_target_sha256(spec: RowSpec) -> str:
    return _canonical_target_sha256_by_row()[spec.row_id]


@functools.lru_cache(maxsize=1)
def _phase1_gate_bindings() -> dict[str, str]:
    bindings = {
        CANONICAL_TARGETS_PATH: CANONICAL_TARGETS_SHA256,
        P1A_RECEIPT_PATH: P1A_RECEIPT_SHA256,
        P1B_RECEIPT_PATH: P1B_RECEIPT_SHA256,
        PHASE1_ENTRY_AMENDMENT_PATH: PHASE1_ENTRY_AMENDMENT_SHA256,
    }
    for relative, expected in bindings.items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"Phase 1 gate artifact hash mismatch: {relative}")
    return bindings


def _configuration_identity(args: argparse.Namespace, spec: RowSpec) -> dict[str, Any]:
    payload = {
        "schema_version": "bayesfilter.ledh.configuration_identity.v1",
        "row_id": spec.row_id,
        "canonical_target_sha256": _canonical_target_sha256(spec),
        "source_value_artifact_sha256": _source_value_sha256(spec),
        "score_parameter_names": list(spec.parameter_names),
        "score_evaluation_theta": [
            float(tf.constant(value, dtype=tf.float32).numpy())
            for value in spec.truth_theta
        ],
        "time_steps": int(args.time_steps),
        "num_particles": int(args.num_particles),
        "transport_policy": args.transport_policy,
        "sinkhorn_iterations": int(args.sinkhorn_iterations),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "annealed_scaling": float(args.annealed_scaling),
        "annealed_convergence_threshold": float(
            args.annealed_convergence_threshold
        ),
        "row_chunk_size": int(args.row_chunk_size),
        "col_chunk_size": int(args.col_chunk_size),
        "particle_chunk_size": int(args.particle_chunk_size),
        "transport_plan_mode": args.transport_plan_mode,
        "transport_ad_mode": args.transport_ad_mode,
        "transport_gradient_mode": args.transport_gradient_mode,
        "flow_observation_variance": args.flow_observation_variance,
        "jit_compile": True,
        "dtype": args.dtype,
        "tf32_mode": args.tf32_mode,
        "fd_policy_id": LEDH_FD_POLICY_ID,
        "fd_step_policy": ledh_fd_step_policy_metadata(),
    }
    return {"payload": payload, "sha256": _canonical_json_sha256(payload)}


def _route_identity(args: argparse.Namespace, spec: RowSpec) -> dict[str, Any]:
    payload = {
        "schema_version": "bayesfilter.ledh.route_identity.v1",
        "row_id": spec.row_id,
        "score_route": spec.score_route,
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "transport_plan_mode": args.transport_plan_mode,
        "transport_ad_mode": args.transport_ad_mode,
        "transport_gradient_mode": args.transport_gradient_mode,
        "code_source_sha256": _code_source_sha256(spec),
    }
    return {"payload": payload, "sha256": _canonical_json_sha256(payload)}


def _fd_endpoint_contract(spec: RowSpec) -> dict[str, Any]:
    center = tf.constant(spec.truth_theta, dtype=tf.float32)
    directions = []
    for index, name in enumerate(spec.parameter_names):
        nominal_step = coordinate_central_difference_step(float(center[index].numpy()))
        basis = tf.one_hot(index, len(spec.parameter_names), dtype=tf.float32)
        step = tf.constant(nominal_step, dtype=tf.float32)
        minus = center - step * basis
        plus = center + step * basis
        denominator = plus[index] - minus[index]
        directions.append(
            {
                "parameter": name,
                "direction_index": index,
                "nominal_step": nominal_step,
                "minus_theta": [float(value) for value in minus.numpy().reshape(-1)],
                "plus_theta": [float(value) for value in plus.numpy().reshape(-1)],
                "effective_denominator": float(denominator.numpy()),
            }
        )
    payload = {
        "schema_version": "bayesfilter.ledh.fd_endpoint_contract.v1",
        "row_id": spec.row_id,
        "parameter_names": list(spec.parameter_names),
        "center_theta": [float(value) for value in center.numpy().reshape(-1)],
        "step_policy": ledh_fd_step_policy_metadata(),
        "directions": directions,
    }
    return {"payload": payload, "sha256": _canonical_json_sha256(payload)}


def _command_identity_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    argv = manifest.get("command_argv")
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("run manifest command_argv must be a nonempty sequence")
    normalized_argv = [str(item) for item in argv]
    template_payload = {
        "schema_version": "bayesfilter.ledh.command_template_family.v1",
        "runner_path": manifest.get("runner_path"),
        "working_directory": manifest.get("working_directory"),
        "python_executable": manifest.get("python_executable"),
        "row": manifest.get("row"),
        "row_id": manifest.get("row_id"),
        "stage": manifest.get("stage"),
        "configuration_sha256": _require_mapping(
            "manifest configuration identity",
            manifest.get("configuration_identity"),
        ).get("sha256"),
        "route_sha256": _require_mapping(
            "manifest route identity",
            manifest.get("route_identity"),
        ).get("sha256"),
        "canonical_target_sha256": manifest.get("canonical_target_sha256"),
        "fd_endpoint_contract_sha256": _require_mapping(
            "manifest FD endpoint contract",
            manifest.get("fd_endpoint_contract"),
        ).get("sha256"),
    }
    exact_payload = {
        **template_payload,
        "schema_version": "bayesfilter.ledh.exact_command.v1",
        "command_argv": normalized_argv,
        "output": manifest.get("output"),
        "markdown_output": manifest.get("markdown_output"),
        "batch_seeds": list(manifest.get("batch_seeds") or ()),
        "score_reference_json": manifest.get("score_reference_json"),
        "device_scope": manifest.get("device_scope"),
        "cuda_visible_devices": manifest.get("cuda_visible_devices"),
        "device": manifest.get("device"),
        "expect_device_kind": manifest.get("expect_device_kind"),
        "command_timeout_seconds": manifest.get("command_timeout_seconds"),
    }
    return {
        "template_payload": template_payload,
        "template_family_sha256": _canonical_json_sha256(template_payload),
        "exact_payload": exact_payload,
        "exact_command_sha256": _canonical_json_sha256(exact_payload),
    }


@functools.lru_cache(maxsize=None)
def _frozen_source_value_sha256(row: str) -> str:
    return _sha256(ROOT / ROW_SPECS[row].source_value_artifact)


def _source_value_sha256(spec: RowSpec) -> str:
    return _frozen_source_value_sha256(spec.name)


def _git_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}:{exc}"


@functools.lru_cache(maxsize=None)
def _code_source_paths(row: str) -> tuple[Path, ...]:
    modules = [ROW_SPECS[row].module]
    visited: set[str] = set()
    paths = {
        Path(__file__).resolve(),
        ROOT / "bayesfilter/ledh_fd_policy.py",
        ROOT / "docs/benchmarks/build_ledh_phase9_gpu_command_manifest.py",
        ROOT / COMPLETE_HIGHDIM_COMMAND_BUILDER_PATH,
    }
    while modules:
        module = modules.pop()
        name = str(getattr(module, "__name__", ""))
        if not name or name in visited:
            continue
        visited.add(name)
        raw_path = getattr(module, "__file__", None)
        if raw_path is not None:
            path = Path(raw_path).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                paths.add(path)
        for value in vars(module).values():
            dependency = value if isinstance(value, types.ModuleType) else None
            if dependency is None and (inspect.isfunction(value) or inspect.isclass(value)):
                dependency = inspect.getmodule(value)
            if dependency is None:
                continue
            value_path = getattr(dependency, "__file__", None)
            if value_path is None:
                continue
            path = Path(value_path).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                modules.append(dependency)
    return tuple(sorted(paths))


@functools.lru_cache(maxsize=None)
def _frozen_code_source_sha256(row: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(path.relative_to(ROOT)), _sha256(path))
        for path in _code_source_paths(row)
    )


def _code_source_sha256(spec: RowSpec) -> dict[str, str]:
    return dict(_frozen_code_source_sha256(spec.name))


@functools.lru_cache(maxsize=1)
def _frozen_governance_artifact_sha256() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, _sha256(ROOT / path))
        for path in (
            PLAN_PATH,
            FD_POLICY_CORRECTION_PLAN_PATH,
            GATE_A_RESULT_PATH,
            EXECUTION_MANIFEST_PATH,
            EXACT_COMMANDS_PATH,
            GATE_B_REVIEW_PATH,
            GATE_B_REPAIR_REVIEW_PATH,
            GATE_B_RESULT_PATH,
            GATE_B_RESULT_REVIEW_PATH,
            ROOT_CAUSE_REPAIR_PLAN_PATH,
            ROOT_CAUSE_REPAIR_COMMANDS_PATH,
            COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH,
        )
    )


def _governance_artifact_sha256() -> dict[str, str]:
    return dict(_frozen_governance_artifact_sha256())


def _prepared_input_fingerprint(prepared: Mapping[str, Any]) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []

    def visit(path: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(f"{path}.{key}" if path else str(key), value[key])
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(f"{path}[{index}]", item)
            return
        if not tf.is_tensor(value):
            raise ValueError(f"prepared input leaf {path} must be a TensorFlow tensor")
        tensor = tf.convert_to_tensor(value)
        serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
        leaves.append(
            {
                "path": path,
                "dtype": tensor.dtype.name,
                "shape": tensor.shape.as_list(),
                "sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )

    tensor_inputs = {key: value for key, value in prepared.items() if key != "semantics"}
    visit("", tensor_inputs)
    canonical = json.dumps(leaves, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "algorithm": "sha256_tf_serialize_tensor_tree_v1",
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "tensor_leaf_count": len(leaves),
        "tensor_leaves": leaves,
    }


def _randomness_identity(
    *,
    seed: int,
    prepared_input_fingerprint: Mapping[str, Any],
    configuration_identity: Mapping[str, Any],
    route_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "bayesfilter.ledh.randomness_identity.v1",
        "seed": int(seed),
        "prepared_input_aggregate_sha256": prepared_input_fingerprint.get(
            "aggregate_sha256"
        ),
        "configuration_sha256": configuration_identity.get("sha256"),
        "route_sha256": route_identity.get("sha256"),
    }
    return {"payload": payload, "sha256": _canonical_json_sha256(payload)}


@functools.lru_cache(maxsize=1)
def _exact_execution_commands() -> tuple[Mapping[str, Any], ...]:
    payload = json.loads((ROOT / EXACT_COMMANDS_PATH).read_text(encoding="utf-8"))
    commands = (
        list(payload.get("gate_b_commands") or ())
        + list(payload.get("gate_c_commands") or ())
        + list(payload.get("gate_d_commands") or ())
        + list(payload.get("aggregate_commands") or ())
    )
    if len(commands) != 91:
        raise ValueError("frozen exact command manifest must contain 91 commands")
    return tuple(commands)


@functools.lru_cache(maxsize=1)
def _complete_highdim_execution_commands() -> tuple[Mapping[str, Any], ...]:
    path = ROOT / COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "bayesfilter.complete_highdim.ledh_phase2_phase3_exact_commands.v1"
    ):
        raise ValueError("complete-highdim exact-command manifest schema mismatch")
    commands = tuple(
        list(payload.get("phase2_commands") or ())
        + list(payload.get("phase3_commands") or ())
        + list(payload.get("aggregate_commands") or ())
    )
    expected_count = int(payload.get("command_count", -1))
    if not commands or len(commands) != expected_count:
        raise ValueError("complete-highdim exact-command manifest count mismatch")
    exact_hashes = [str(command.get("exact_command_sha256")) for command in commands]
    if len(set(exact_hashes)) != len(exact_hashes):
        raise ValueError("complete-highdim exact-command hashes must be unique")
    return commands


def _validate_complete_highdim_execution_authority(
    command: Mapping[str, Any],
) -> Mapping[str, Any]:
    phase = str(command.get("phase"))
    authority_paths = {
        "phase2": COMPLETE_HIGHDIM_PHASE2_EXECUTION_AUTHORITY_PATH,
        "phase3": COMPLETE_HIGHDIM_PHASE3_EXECUTION_AUTHORITY_PATH,
    }
    if phase not in authority_paths:
        raise ValueError("complete-highdim command has no phase authority class")
    path = ROOT / authority_paths[phase]
    if not path.is_file():
        raise ValueError(f"complete-highdim {phase} execution is not authorized")
    payload = _require_mapping(
        "complete-highdim execution authority",
        json.loads(path.read_text(encoding="utf-8")),
    )
    expected = {
        "schema_version": "bayesfilter.complete_highdim.execution_authority.v1",
        "run_id": "complete-highdim-leaderboard-local-20260712-134906",
        "status": "authorized",
        "authorized_phase": phase,
        "exact_command_manifest_path": COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH,
        "exact_command_manifest_sha256": _sha256(
            ROOT / COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
        ),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"complete-highdim execution authority {key} mismatch")
    for key in (
        "prior_phase_result_path",
        "prior_phase_result_sha256",
        "phase_subplan_path",
        "phase_subplan_sha256",
        "phase_entry_review_path",
        "phase_entry_review_sha256",
        "owner_approval",
    ):
        _nonempty_string(f"complete-highdim execution authority {key}", payload.get(key))
    for path_key, hash_key in (
        ("prior_phase_result_path", "prior_phase_result_sha256"),
        ("phase_subplan_path", "phase_subplan_sha256"),
        ("phase_entry_review_path", "phase_entry_review_sha256"),
    ):
        if _sha256(ROOT / str(payload[path_key])) != payload[hash_key]:
            raise ValueError(
                f"complete-highdim execution authority {hash_key} mismatch"
            )
    return payload


@functools.lru_cache(maxsize=1)
def _root_cause_repair_execution_commands() -> tuple[Mapping[str, Any], ...]:
    payload = json.loads((ROOT / ROOT_CAUSE_REPAIR_COMMANDS_PATH).read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "bayesfilter.ledh.predator_generalized_fd_repair_gpu_commands.v1"
    ):
        raise ValueError("root-cause repair command manifest schema mismatch")
    commands = tuple(payload.get("commands") or ())
    if len(commands) != 4:
        raise ValueError("root-cause repair command manifest must contain four commands")
    return commands


def _matching_execution_commands(
    commands: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    spec: RowSpec,
) -> list[Mapping[str, Any]]:
    candidates = [
        command
        for command in commands
        if command.get("row") == spec.name
        and command.get("stage") == args.stage
        and int(command.get("time_steps")) == int(args.time_steps)
        and int(command.get("num_particles")) == int(args.num_particles)
        and command.get("output") == args.output
        and command.get("score_reference_json") == args.score_reference_json
        and command.get("command_timeout_seconds") == args.command_timeout_seconds
    ]
    if args.stage == "aggregate":
        candidates = [
            command
            for command in candidates
            if tuple(command.get("seeds") or ()) == tuple(args.batch_seeds)
            and list(command.get("score_shards") or ()) == list(args.score_shards)
            and list(command.get("fd_shards") or ()) == list(args.fd_shards)
        ]
    else:
        candidates = [
            command
            for command in candidates
            if int(command.get("seed")) == int(args.batch_seeds[0])
        ]
    return candidates


def _validate_exact_execution_command(
    args: argparse.Namespace,
    spec: RowSpec,
    invoked_argv: Sequence[str] | None = None,
) -> None:
    if args.stage != "aggregate" and not (
        args.device_scope == "visible" and args.expect_device_kind == "gpu"
    ):
        return
    if Path.cwd().resolve() != ROOT:
        raise ValueError("reviewed Phase 9 commands must run from the repository root")
    active_candidates = _matching_execution_commands(
        _complete_highdim_execution_commands(),
        args,
        spec,
    )
    active_candidates = [
        command
        for command in active_candidates
        if command.get("markdown_output") == args.markdown_output
        and command.get("canonical_target_sha256")
        == _canonical_target_sha256(spec)
        and command.get("source_value_artifact_sha256")
        == _source_value_sha256(spec)
        and command.get("configuration_identity")
        == _configuration_identity(args, spec)
        and command.get("route_identity") == _route_identity(args, spec)
        and command.get("fd_endpoint_contract") == _fd_endpoint_contract(spec)
        and args.device_scope == ("cpu" if args.stage == "aggregate" else "visible")
        and args.expect_device_kind == ("cpu" if args.stage == "aggregate" else "gpu")
        and (
            args.stage == "aggregate"
            or (args.cuda_visible_devices == "0" and args.device == "/GPU:0")
        )
    ]
    if len(active_candidates) == 1:
        expected_argv = list(active_candidates[0]["argv"][1:])
        if invoked_argv is not None and list(invoked_argv) != expected_argv:
            raise ValueError(
                "runtime argv does not exactly match the complete-highdim command freeze"
            )
        _validate_complete_highdim_execution_authority(active_candidates[0])
        return
    repair_candidates = _matching_execution_commands(
        _root_cause_repair_execution_commands(),
        args,
        spec,
    )
    if len(repair_candidates) == 1:
        if not (
            args.device_scope == "visible"
            and args.expect_device_kind == "gpu"
            and args.cuda_visible_devices == "0"
            and args.device == "/GPU:0"
        ):
            raise ValueError("root-cause repair command requires trusted visible GPU execution")
        if invoked_argv is not None and list(invoked_argv) != list(repair_candidates[0]["argv"][1:]):
            raise ValueError("runtime argv does not exactly match the reviewed repair command")
        if not ROOT_CAUSE_REPAIR_GPU_EXECUTION_AUTHORIZED:
            raise ValueError("root-cause repair GPU execution is not authorized")
        return

    historical_candidates = _matching_execution_commands(_exact_execution_commands(), args, spec)
    historical_candidates = [
        command
        for command in historical_candidates
        if command.get("markdown_output") == args.markdown_output
        and args.device_scope == ("cpu" if args.stage == "aggregate" else "visible")
        and args.expect_device_kind == ("cpu" if args.stage == "aggregate" else "gpu")
        and (
            args.stage == "aggregate"
            or (args.cuda_visible_devices == "0" and args.device == "/GPU:0")
        )
    ]
    if len(historical_candidates) == 1:
        if invoked_argv is not None and list(invoked_argv) != list(historical_candidates[0]["argv"][1:]):
            raise ValueError("runtime argv does not exactly match the reviewed Phase 9 command")
        raise ValueError(
            "the historical Phase 9 exact-command manifest is superseded; "
            "new GPU or aggregate execution requires a reviewed command plan"
        )
    raise ValueError("runtime command is not an exact reviewed Phase 9 command")


def _manifest(args: argparse.Namespace, spec: RowSpec) -> dict[str, Any]:
    invoked_argv = getattr(args, "invoked_command_argv", sys.argv)
    configuration_identity = _configuration_identity(args, spec)
    route_identity = _route_identity(args, spec)
    manifest = {
        "command": shlex.join(invoked_argv),
        "command_argv": [str(item) for item in invoked_argv],
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "output": args.output,
        "markdown_output": args.markdown_output,
        "working_directory": str(Path.cwd().resolve()),
        "git_commit": _git_output(("git", "rev-parse", "HEAD")),
        "git_status_short": _git_output(("git", "status", "--short")),
        "code_source_sha256": _code_source_sha256(spec),
        "governance_artifact_sha256": _governance_artifact_sha256(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "tensorflow_version": tf.__version__,
        "host": platform.node(),
        "platform": platform.platform(),
        "gpu_trust_basis": (
            GPU_TRUST_BASIS
            if args.stage != "aggregate" and args.device_scope == "visible"
            else None
        ),
        "device_scope": args.device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
        "jit_compile": JIT_COMPILE,
        "dtype": args.dtype,
        "tf32_mode": args.tf32_mode,
        "row": spec.name,
        "row_id": spec.row_id,
        "canonical_target_artifact": CANONICAL_TARGETS_PATH,
        "canonical_target_artifact_sha256": CANONICAL_TARGETS_SHA256,
        "canonical_target_sha256": _canonical_target_sha256(spec),
        "fd_endpoint_contract": _fd_endpoint_contract(spec),
        "phase1_gate_artifact_sha256": _phase1_gate_bindings(),
        "configuration_identity": configuration_identity,
        "route_identity": route_identity,
        "source_value_artifact": args.source_value_artifact,
        "source_value_artifact_sha256": _source_value_sha256(spec),
        "score_parameter_names": list(spec.parameter_names),
        "truth_theta": list(spec.truth_theta),
        "stage": args.stage,
        "score_reference_json": args.score_reference_json,
        "time_steps": int(args.time_steps),
        "num_particles": int(args.num_particles),
        "batch_seeds": list(args.batch_seeds),
        "transport_policy": args.transport_policy,
        "sinkhorn_iterations": int(args.sinkhorn_iterations),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "annealed_scaling": float(args.annealed_scaling),
        "annealed_convergence_threshold": float(args.annealed_convergence_threshold),
        "row_chunk_size": int(args.row_chunk_size),
        "col_chunk_size": int(args.col_chunk_size),
        "particle_chunk_size": int(args.particle_chunk_size),
        "transport_plan_mode": args.transport_plan_mode,
        "transport_ad_mode": args.transport_ad_mode,
        "transport_gradient_mode": args.transport_gradient_mode,
        "flow_observation_variance": args.flow_observation_variance,
        "memory_budget_mib": float(args.memory_budget_mib),
        "command_timeout_seconds": args.command_timeout_seconds,
        "legacy_module_fd_step_not_used_by_gpu_fd": spec.legacy_module_fd_step,
        "fd_step_policy": ledh_fd_step_policy_metadata(),
        "fd_policy_id": LEDH_FD_POLICY_ID,
        "fd_diagnostic_scope": LEDH_FD_DIAGNOSTIC_SCOPE,
        "fd_base_relative_tolerance": LEDH_FD_BASE_RELATIVE_TOLERANCE,
        "fd_coordinate_relative_error_denominator": LEDH_FD_DENOMINATOR,
        "fd_pass_rule": LEDH_FD_PASS_RULE,
        "fd_statistical_interpretation": LEDH_FD_STATISTICAL_STATUS,
        "gpu_execution_authorized": GPU_EXECUTION_AUTHORIZED,
        "root_cause_repair_gpu_execution_authorized": (
            ROOT_CAUSE_REPAIR_GPU_EXECUTION_AUTHORIZED
        ),
        "plan_path": PLAN_PATH,
        "fd_policy_correction_plan_path": FD_POLICY_CORRECTION_PLAN_PATH,
        "historical_result_path": HISTORICAL_RESULT_PATH,
        "result_path": RESULT_PATH,
        "execution_manifest_path": EXECUTION_MANIFEST_PATH,
        "exact_commands_path": EXACT_COMMANDS_PATH,
        "gate_b_review_path": GATE_B_REVIEW_PATH,
        "gate_b_repair_review_path": GATE_B_REPAIR_REVIEW_PATH,
        "gate_b_result_path": GATE_B_RESULT_PATH,
        "gate_b_result_review_path": GATE_B_RESULT_REVIEW_PATH,
        "root_cause_repair_plan_path": ROOT_CAUSE_REPAIR_PLAN_PATH,
        "root_cause_repair_commands_path": ROOT_CAUSE_REPAIR_COMMANDS_PATH,
        "complete_highdim_exact_commands_path": COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH,
        "complete_highdim_exact_commands_sha256": _sha256(
            ROOT / COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
        ),
    }
    manifest["command_identity"] = _command_identity_from_manifest(manifest)
    return manifest


def _progress(args: argparse.Namespace, spec: RowSpec, status: str, terminal: bool, started: float, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": status,
        "terminal_artifact": terminal,
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "row_id": spec.row_id,
        "score_route": spec.score_route,
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW,
        "run_manifest": _manifest(args, spec),
        "nonclaims": [
            "raw score and FD shards are not score admission",
            "prefix results are not full-row evidence",
            "segmented execution is not monolithic batch memory or runtime evidence",
            "not HMC readiness or posterior correctness evidence",
            "not a runtime or scientific superiority claim",
        ],
    }
    payload.update(extra)
    return payload


def _load_source_value(path: str, spec: RowSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = ROOT / path
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _source_value_sha256(spec):
        raise ValueError("source value artifact changed after process provenance was frozen")
    payload = json.loads(raw)
    normalized = validate_ledh_forward_scalar_artifact(
        payload,
        expected_row_id=spec.row_id,
        require_admitted=False,
    )
    return payload, normalized


def _compiled_score(spec: RowSpec, args: argparse.Namespace, prepared: Mapping[str, Any]):
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return spec.module._compact_score_tensor_outputs(args, theta, prepared)  # noqa: SLF001

    return compiled


def _compiled_value(spec: RowSpec, args: argparse.Namespace, prepared: Mapping[str, Any]):
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return spec.module._value_tensor_outputs(args, theta, prepared)  # noqa: SLF001

    return compiled


def _score_only(args: argparse.Namespace, spec: RowSpec, started: float) -> dict[str, Any]:
    source_payload, source = _load_source_value(args.source_value_artifact, spec)
    precision = _configure_precision(spec, args)
    physical, logical = _configure_devices()
    prepared = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    prepared_fingerprint = _prepared_input_fingerprint(prepared)
    manifest = _manifest(args, spec)
    configuration_identity = _require_mapping(
        "score configuration identity",
        manifest["configuration_identity"],
    )
    route_identity = _require_mapping(
        "score route identity",
        manifest["route_identity"],
    )
    randomness_identity = _randomness_identity(
        seed=int(args.batch_seeds[0]),
        prepared_input_fingerprint=prepared_fingerprint,
        configuration_identity=configuration_identity,
        route_identity=route_identity,
    )
    semantics = _require_mapping("prepared semantics", prepared.get("semantics"))
    if semantics.get("row_id") != spec.row_id:
        raise ValueError("prepared score inputs have the wrong row identity")
    if (
        semantics.get("target_observation_policy") is not None
        and semantics.get("target_observation_policy") != source["target_observation_policy"]
    ):
        raise ValueError("prepared score inputs have the wrong target observation policy")
    theta = tf.constant(spec.truth_theta, dtype=tf.float32)
    compiled = _compiled_score(spec, args, prepared)
    reset = _reset_gpu_memory_stats()
    before = _gpu_memory_info()
    with tf.device(args.device):
        call_started = time.perf_counter()
        outputs = compiled(theta)
        _materialize(outputs)
        call_seconds = time.perf_counter() - call_started
    after = _gpu_memory_info()
    devices = _validate_devices(outputs, args.expect_device_kind)
    objective, log_likelihood, score, per_seed_score = outputs
    finite = bool(all(tf.reduce_all(tf.math.is_finite(value)).numpy() for value in outputs))
    if not finite:
        raise ValueError("compact score outputs are nonfinite")
    if args.expect_device_kind == "gpu" and not reset:
        raise ValueError("trusted GPU score evidence requires reset memory statistics")
    peak_bytes = after.get("peak")
    peak_mib = None if peak_bytes is None else float(peak_bytes) / (1024.0 * 1024.0)
    if args.expect_device_kind == "gpu" and peak_mib is None:
        raise ValueError("trusted GPU score evidence requires a finite reset memory peak")
    memory_pass = peak_mib is not None and peak_mib <= args.memory_budget_mib
    del source_payload
    return _progress(
        args,
        spec,
        "completed" if args.expect_device_kind == "cpu" or memory_pass else "blocked_memory_budget",
        True,
        started,
        evidence_class=GPU_TRUST_BASIS if args.device_scope == "visible" else "cpu_hidden_debug_only",
        source_value_artifact=args.source_value_artifact,
        source_value_artifact_sha256=_source_value_sha256(spec),
        target_observation_policy=source["target_observation_policy"],
        theta_coordinate_system=source["theta_coordinate_system"],
        score_parameter_names=list(spec.parameter_names),
        truth_theta=list(spec.truth_theta),
        score_evaluation_theta=[
            float(tf.constant(value, dtype=tf.float32).numpy())
            for value in spec.truth_theta
        ],
        canonical_target_sha256=_canonical_target_sha256(spec),
        configuration_identity=dict(configuration_identity),
        route_identity=dict(route_identity),
        randomness_identity=randomness_identity,
        prepared_input_fingerprint=prepared_fingerprint,
        objective=float(objective.numpy()),
        total_log_likelihood=float(log_likelihood.numpy().reshape(-1)[0]),
        log_likelihood_by_seed=[float(value) for value in log_likelihood.numpy().reshape(-1)],
        score=[float(value) for value in score.numpy().reshape(-1)],
        per_seed_score=per_seed_score.numpy().tolist(),
        score_derivative_provenance=spec.score_route,
        value_score_route_status=LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        value_score_same_transport_algorithm=True,
        no_autodiff_score_route=True,
        uses_gradient_tape=False,
        uses_forward_accumulator=False,
        uses_stopped_partial_derivative=False,
        score_correctness={"kind": "same_scalar_finite_difference", "status": "not_run_score_only"},
        score_finite=True,
        score_output_devices=devices,
        physical_gpus=physical,
        logical_gpus=logical,
        precision=precision,
        score_call_seconds=call_seconds,
        score_compile_first_call_seconds=call_seconds,
        score_gpu_memory_stats_reset=reset,
        score_gpu_memory_info_before=before,
        score_gpu_memory_info_after=after,
        memory_diagnostics={
            "score_memory_budget_pass": memory_pass,
            "full_row_memory_gate_applicable": bool(
                args.num_particles == spec.full_num_particles
                and args.time_steps == spec.full_time_steps
            ),
            "n10000_memory_pass": memory_pass if args.num_particles == spec.full_num_particles else None,
            "peak_mib": peak_mib,
            "budget_mib": args.memory_budget_mib,
            "source": "score_gpu_memory_info_after" if peak_mib is not None else None,
        },
    )


def _manifest_args(manifest: Mapping[str, Any]) -> argparse.Namespace:
    command_argv = manifest.get("command_argv")
    if not isinstance(command_argv, Sequence) or isinstance(
        command_argv, (str, bytes)
    ):
        raise ValueError("run manifest command_argv must be a sequence")
    normalized = [str(item) for item in command_argv]
    try:
        argument_start = normalized.index("--row")
    except ValueError as exc:
        raise ValueError("run manifest command_argv must contain --row") from exc
    return _parse_args(normalized[argument_start:])


def _load_score_reference(path: str, args: argparse.Namespace, spec: RowSpec) -> dict[str, Any]:
    reference_path = Path(path)
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest = _require_mapping("run_manifest", payload.get("run_manifest"))
    declared_output = Path(str(manifest.get("output")))
    if not declared_output.is_absolute():
        declared_output = ROOT / declared_output
    if declared_output.resolve() != reference_path.resolve():
        raise ValueError("score reference run manifest output path mismatch")
    reference_args = _manifest_args(manifest)
    if reference_args.stage != "score-only":
        raise ValueError("score reference must record a score-only command")
    _validate_raw_score_shard(
        payload,
        reference_args,
        spec,
        require_gpu=args.expect_device_kind == "gpu",
    )
    shared_fields = (
        "batch_seeds",
        "time_steps",
        "num_particles",
        "source_value_artifact",
        "memory_budget_mib",
        "device_scope",
        "cuda_visible_devices",
        "device",
        "expect_device_kind",
        "dtype",
        "tf32_mode",
    )
    for field in shared_fields:
        if getattr(reference_args, field) != getattr(args, field):
            raise ValueError(f"score reference {field} does not match FD command")
    if _configuration_identity(reference_args, spec) != _configuration_identity(
        args, spec
    ):
        raise ValueError("score reference configuration identity mismatch")
    if _route_identity(reference_args, spec) != _route_identity(args, spec):
        raise ValueError("score reference route identity mismatch")
    return payload


def _fd_only(args: argparse.Namespace, spec: RowSpec, started: float) -> dict[str, Any]:
    reference = _load_score_reference(args.score_reference_json, args, spec)
    _source_payload, source = _load_source_value(args.source_value_artifact, spec)
    precision = _configure_precision(spec, args)
    physical, logical = _configure_devices()
    prepared = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    prepared_fingerprint = _prepared_input_fingerprint(prepared)
    if prepared_fingerprint != reference.get("prepared_input_fingerprint"):
        raise ValueError("FD prepared inputs do not match the score reference")
    compiled = _compiled_value(spec, args, prepared)
    theta = tf.constant(spec.truth_theta, dtype=tf.float32)
    score = tf.constant(reference["score"], dtype=tf.float32)
    fd_values = []
    fd_diagnostics = []
    output_devices: set[str] = set()
    fd_started = time.perf_counter()
    value_compile_first_call_seconds = None
    for index, name in enumerate(spec.parameter_names):
        basis = tf.one_hot(index, len(spec.parameter_names), dtype=tf.float32)
        theta_value = float(theta[index].numpy())
        nominal_step_value = coordinate_central_difference_step(theta_value)
        nominal_step = tf.constant(nominal_step_value, dtype=tf.float32)
        plus_theta = theta + nominal_step * basis
        minus_theta = theta - nominal_step * basis
        effective_denominator = plus_theta[index] - minus_theta[index]
        if float(effective_denominator.numpy()) == 0.0:
            raise ValueError(f"FD parameter endpoints collapsed for {name}")
        with tf.device(args.device):
            call_started = time.perf_counter()
            plus = compiled(plus_theta)
            _materialize(plus)
            if value_compile_first_call_seconds is None:
                value_compile_first_call_seconds = time.perf_counter() - call_started
            minus = compiled(minus_theta)
            _materialize(minus)
        output_devices.update(_validate_devices((*plus, *minus), args.expect_device_kind))
        numerator = plus[0] - minus[0]
        fd = numerator / effective_denominator
        diagnostic = {
            "parameter": name,
            "direction_index": index,
            "theta": theta_value,
            "nominal_step": nominal_step_value,
            "minus_parameter": float(minus_theta[index].numpy()),
            "plus_parameter": float(plus_theta[index].numpy()),
            "center_theta": [float(value) for value in theta.numpy().reshape(-1)],
            "minus_endpoint": {
                "role": "minus",
                "theta": [
                    float(value) for value in minus_theta.numpy().reshape(-1)
                ],
                "total_log_likelihood": float(minus[0].numpy()),
            },
            "plus_endpoint": {
                "role": "plus",
                "theta": [
                    float(value) for value in plus_theta.numpy().reshape(-1)
                ],
                "total_log_likelihood": float(plus[0].numpy()),
            },
            "effective_step": float((effective_denominator / 2.0).numpy()),
            "effective_denominator": float(effective_denominator.numpy()),
            "minus_objective": float(minus[0].numpy()),
            "plus_objective": float(plus[0].numpy()),
            "objective_numerator": float(numerator.numpy()),
            "endpoint_objectives_equal": bool(float(plus[0].numpy()) == float(minus[0].numpy())),
            "finite_difference": float(fd.numpy()),
        }
        numeric_fields = (
            "theta",
            "nominal_step",
            "minus_parameter",
            "plus_parameter",
            "effective_step",
            "effective_denominator",
            "minus_objective",
            "plus_objective",
            "objective_numerator",
            "finite_difference",
        )
        if not all(
            math.isfinite(float(diagnostic[field])) for field in numeric_fields
        ):
            raise ValueError(f"nonfinite FD endpoint diagnostic for {name}")
        fd_values.append(fd)
        fd_diagnostics.append(diagnostic)
    fd_policy = evaluate_ledh_fd_policy(
        [float(value) for value in score.numpy().reshape(-1)],
        [float(value) for value in tf.stack(fd_values).numpy().reshape(-1)],
        spec.parameter_names,
    )
    passed = fd_policy["status"] == "pass"
    return _progress(
        args,
        spec,
        "completed" if passed else "failed_fd",
        True,
        started,
        evidence_class=GPU_TRUST_BASIS if args.device_scope == "visible" else "cpu_hidden_debug_only",
        source_value_artifact=args.source_value_artifact,
        source_value_artifact_sha256=_source_value_sha256(spec),
        score_reference_json=args.score_reference_json,
        score_reference_sha256=_sha256(Path(args.score_reference_json)),
        target_observation_policy=source["target_observation_policy"],
        theta_coordinate_system=source["theta_coordinate_system"],
        score_parameter_names=list(spec.parameter_names),
        truth_theta=list(spec.truth_theta),
        score_evaluation_theta=[
            float(tf.constant(value, dtype=tf.float32).numpy())
            for value in spec.truth_theta
        ],
        canonical_target_sha256=reference["canonical_target_sha256"],
        configuration_identity=reference["configuration_identity"],
        route_identity=reference["route_identity"],
        randomness_identity=reference["randomness_identity"],
        prepared_input_fingerprint=prepared_fingerprint,
        score=list(reference["score"]),
        score_derivative_provenance=spec.score_route,
        value_score_route_status=LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        score_correctness={
            "kind": "same_scalar_finite_difference",
            "status": "pass" if passed else "fail",
            "step_policy": ledh_fd_step_policy_metadata(),
            "finite_difference_diagnostics": fd_diagnostics,
            "fd_policy": fd_policy,
            "uses_value_only_scalar_route": True,
        },
        value_output_devices=sorted(output_devices),
        physical_gpus=physical,
        logical_gpus=logical,
        precision=precision,
        value_compile_first_call_seconds=value_compile_first_call_seconds,
        value_fd_elapsed_seconds=time.perf_counter() - fd_started,
    )


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_vector(name: str, value: object, length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    output = [float(item) for item in value]
    if len(output) != length or any(not math.isfinite(item) for item in output):
        raise ValueError(f"{name} must contain {length} finite values")
    return output


def _finite_scalar(name: str, value: object) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_prepared_input_fingerprint(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fingerprint = _require_mapping(
        "prepared_input_fingerprint",
        payload.get("prepared_input_fingerprint"),
    )
    if fingerprint.get("algorithm") != "sha256_tf_serialize_tensor_tree_v1":
        raise ValueError("prepared input fingerprint algorithm mismatch")
    aggregate_sha256 = _nonempty_string(
        "prepared input aggregate SHA-256",
        fingerprint.get("aggregate_sha256"),
    )
    if len(aggregate_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in aggregate_sha256
    ):
        raise ValueError("prepared input aggregate SHA-256 must be lowercase hexadecimal")
    leaves = fingerprint.get("tensor_leaves")
    if not isinstance(leaves, Sequence) or isinstance(leaves, (str, bytes)) or not leaves:
        raise ValueError("prepared input fingerprint must contain tensor leaves")
    if fingerprint.get("tensor_leaf_count") != len(leaves):
        raise ValueError("prepared input fingerprint leaf count mismatch")
    paths: list[str] = []
    normalized_leaves = []
    for raw_leaf in leaves:
        leaf = _require_mapping("prepared input fingerprint leaf", raw_leaf)
        path = _nonempty_string("prepared input tensor path", leaf.get("path"))
        dtype = _nonempty_string("prepared input tensor dtype", leaf.get("dtype"))
        shape = leaf.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError("prepared input tensor shape must be a sequence")
        normalized_shape = [int(dimension) for dimension in shape]
        if any(dimension < 0 for dimension in normalized_shape):
            raise ValueError("prepared input tensor shape must be fully known and nonnegative")
        sha256 = _nonempty_string("prepared input tensor SHA-256", leaf.get("sha256"))
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("prepared input tensor SHA-256 must be lowercase hexadecimal")
        paths.append(path)
        normalized_leaves.append(
            {"path": path, "dtype": dtype, "shape": normalized_shape, "sha256": sha256}
        )
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("prepared input tensor paths must be unique and sorted")
    canonical = json.dumps(normalized_leaves, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != aggregate_sha256:
        raise ValueError("prepared input aggregate SHA-256 mismatch")
    return fingerprint


def _float32_theta(spec: RowSpec) -> list[float]:
    return [
        float(tf.constant(value, dtype=tf.float32).numpy())
        for value in spec.truth_theta
    ]


def _validate_identity(
    name: str,
    observed: object,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    identity = _require_mapping(name, observed)
    if dict(identity) != dict(expected):
        raise ValueError(f"{name} mismatch")
    payload = _require_mapping(f"{name} payload", identity.get("payload"))
    if identity.get("sha256") != _canonical_json_sha256(payload):
        raise ValueError(f"{name} SHA-256 mismatch")
    return identity


def _validate_command_identity(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = _require_mapping(
        "run manifest command identity",
        manifest.get("command_identity"),
    )
    expected = _command_identity_from_manifest(manifest)
    if dict(identity) != expected:
        raise ValueError("run manifest command identity mismatch")
    argv = manifest.get("command_argv")
    if manifest.get("command") != shlex.join([str(item) for item in argv]):
        raise ValueError("run manifest command string/argv mismatch")
    return identity


def _single_seed(payload: Mapping[str, Any]) -> int:
    manifest = _require_mapping("run_manifest", payload.get("run_manifest"))
    seeds = manifest.get("batch_seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or len(seeds) != 1:
        raise ValueError("raw shards must contain exactly one seed")
    return int(seeds[0])


def _validate_common_shard(payload: Mapping[str, Any], args: argparse.Namespace, spec: RowSpec, *, require_gpu: bool) -> int:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("terminal_artifact") is not True
        or payload.get("artifact_status") != "completed"
    ):
        raise ValueError("shard must be a terminal compact-score artifact")
    if payload.get("row_id") != spec.row_id or payload.get("score_route") != spec.score_route:
        raise ValueError("shard row or compact provenance mismatch")
    try:
        dt.datetime.fromisoformat(_nonempty_string("shard timestamp_utc", payload.get("timestamp_utc")))
    except ValueError as exc:
        raise ValueError("shard timestamp_utc must be an ISO timestamp") from exc
    if _finite_scalar("shard elapsed_seconds", payload.get("elapsed_seconds")) < 0.0:
        raise ValueError("shard elapsed_seconds must be nonnegative")
    manifest = _require_mapping("run_manifest", payload.get("run_manifest"))
    for key in (
        "command",
        "runner_path",
        "output",
        "working_directory",
        "git_commit",
        "python_executable",
        "python_version",
        "tensorflow_version",
        "host",
        "platform",
    ):
        _nonempty_string(f"shard run manifest {key}", manifest.get(key))
    command_argv = manifest.get("command_argv")
    if (
        not isinstance(command_argv, Sequence)
        or isinstance(command_argv, (str, bytes))
        or not command_argv
        or any(not isinstance(item, str) or not item for item in command_argv)
    ):
        raise ValueError("shard run manifest command_argv must be nonempty strings")
    _validate_command_identity(manifest)
    commit = str(manifest["git_commit"])
    if len(commit) != 40 or any(character not in "0123456789abcdefABCDEF" for character in commit):
        raise ValueError("shard run manifest git_commit must be a full hexadecimal commit")
    if commit != _git_output(("git", "rev-parse", "HEAD")):
        raise ValueError("shard run manifest git_commit does not match the current HEAD")
    if not isinstance(manifest.get("git_status_short"), str):
        raise ValueError("shard run manifest git_status_short must disclose the worktree state")
    if manifest.get("code_source_sha256") != _code_source_sha256(spec):
        raise ValueError("shard run manifest code source hashes mismatch")
    if manifest.get("governance_artifact_sha256") != _governance_artifact_sha256():
        raise ValueError("shard run manifest governance artifact hashes mismatch")
    if manifest.get("jit_compile") is not True:
        raise ValueError("shard must record jit_compile=true")
    if manifest.get("dtype") != "float32" or manifest.get("tf32_mode") != "enabled":
        raise ValueError("shard must use production score precision")
    if int(manifest.get("time_steps")) != int(args.time_steps):
        raise ValueError("shard time_steps mismatch")
    if int(manifest.get("num_particles")) != int(args.num_particles):
        raise ValueError("shard num_particles mismatch")
    expected_manifest = {
        "runner_path": "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py",
        "row": spec.name,
        "row_id": spec.row_id,
        "canonical_target_artifact": CANONICAL_TARGETS_PATH,
        "canonical_target_artifact_sha256": CANONICAL_TARGETS_SHA256,
        "canonical_target_sha256": _canonical_target_sha256(spec),
        "fd_endpoint_contract": _fd_endpoint_contract(spec),
        "phase1_gate_artifact_sha256": _phase1_gate_bindings(),
        "configuration_identity": _configuration_identity(args, spec),
        "route_identity": _route_identity(args, spec),
        "source_value_artifact": spec.source_value_artifact,
        "source_value_artifact_sha256": _source_value_sha256(spec),
        "score_parameter_names": list(spec.parameter_names),
        "truth_theta": list(spec.truth_theta),
        "transport_policy": args.transport_policy,
        "sinkhorn_iterations": int(args.sinkhorn_iterations),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "annealed_scaling": float(args.annealed_scaling),
        "annealed_convergence_threshold": float(args.annealed_convergence_threshold),
        "row_chunk_size": int(args.row_chunk_size),
        "col_chunk_size": int(args.col_chunk_size),
        "particle_chunk_size": int(args.particle_chunk_size),
        "transport_plan_mode": args.transport_plan_mode,
        "transport_ad_mode": args.transport_ad_mode,
        "transport_gradient_mode": args.transport_gradient_mode,
        "flow_observation_variance": args.flow_observation_variance,
        "memory_budget_mib": float(args.memory_budget_mib),
        "command_timeout_seconds": args.command_timeout_seconds,
        "legacy_module_fd_step_not_used_by_gpu_fd": spec.legacy_module_fd_step,
        "fd_step_policy": ledh_fd_step_policy_metadata(),
        "fd_policy_id": LEDH_FD_POLICY_ID,
        "fd_diagnostic_scope": LEDH_FD_DIAGNOSTIC_SCOPE,
        "fd_base_relative_tolerance": LEDH_FD_BASE_RELATIVE_TOLERANCE,
        "fd_coordinate_relative_error_denominator": LEDH_FD_DENOMINATOR,
        "fd_pass_rule": LEDH_FD_PASS_RULE,
        "fd_statistical_interpretation": LEDH_FD_STATISTICAL_STATUS,
        "gpu_execution_authorized": GPU_EXECUTION_AUTHORIZED,
        "plan_path": PLAN_PATH,
        "fd_policy_correction_plan_path": FD_POLICY_CORRECTION_PLAN_PATH,
        "historical_result_path": HISTORICAL_RESULT_PATH,
        "result_path": RESULT_PATH,
        "execution_manifest_path": EXECUTION_MANIFEST_PATH,
        "exact_commands_path": EXACT_COMMANDS_PATH,
        "gate_b_review_path": GATE_B_REVIEW_PATH,
        "gate_b_repair_review_path": GATE_B_REPAIR_REVIEW_PATH,
        "gate_b_result_path": GATE_B_RESULT_PATH,
        "gate_b_result_review_path": GATE_B_RESULT_REVIEW_PATH,
        "root_cause_repair_plan_path": ROOT_CAUSE_REPAIR_PLAN_PATH,
        "root_cause_repair_commands_path": ROOT_CAUSE_REPAIR_COMMANDS_PATH,
        "complete_highdim_exact_commands_path": COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH,
        "complete_highdim_exact_commands_sha256": _sha256(
            ROOT / COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
        ),
        "working_directory": str(ROOT),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"shard run manifest {key} mismatch")
    if require_gpu and payload.get("evidence_class") != GPU_TRUST_BASIS:
        raise ValueError("shard must use the managed-session GPU trust basis")
    if require_gpu and manifest.get("gpu_trust_basis") != GPU_TRUST_BASIS:
        raise ValueError("shard run manifest trust basis mismatch")
    if require_gpu:
        if (
            manifest.get("device_scope") != "visible"
            or manifest.get("expect_device_kind") != "gpu"
            or "GPU" not in str(manifest.get("device", "")).upper()
            or manifest.get("cuda_visible_devices") in ("", "-1")
        ):
            raise ValueError("trusted GPU shard run manifest device provenance mismatch")
    if tuple(payload.get("score_parameter_names") or ()) != spec.parameter_names:
        raise ValueError("shard parameter order mismatch")
    if payload.get("canonical_target_sha256") != _canonical_target_sha256(spec):
        raise ValueError("shard canonical target signature mismatch")
    configuration_identity = _validate_identity(
        "shard configuration identity",
        payload.get("configuration_identity"),
        _configuration_identity(args, spec),
    )
    route_identity = _validate_identity(
        "shard route identity",
        payload.get("route_identity"),
        _route_identity(args, spec),
    )
    if payload.get("source_value_artifact") != spec.source_value_artifact:
        raise ValueError("shard source value artifact mismatch")
    if payload.get("source_value_artifact_sha256") != _source_value_sha256(spec):
        raise ValueError("shard source value artifact hash mismatch")
    source_payload, source = _load_source_value(spec.source_value_artifact, spec)
    del source_payload
    if payload.get("target_observation_policy") != source["target_observation_policy"]:
        raise ValueError("shard target observation policy mismatch")
    if payload.get("theta_coordinate_system") != source["theta_coordinate_system"]:
        raise ValueError("shard theta coordinate system mismatch")
    if _finite_vector("shard truth theta", payload.get("truth_theta"), len(spec.truth_theta)) != list(spec.truth_theta):
        raise ValueError("shard truth theta mismatch")
    if _finite_vector(
        "shard score evaluation theta",
        payload.get("score_evaluation_theta"),
        len(spec.truth_theta),
    ) != _float32_theta(spec):
        raise ValueError("shard score evaluation theta mismatch")
    prepared_fingerprint = _validate_prepared_input_fingerprint(payload)
    precision = _require_mapping("precision", payload.get("precision"))
    expected_precision = {
        "dtype": "float32",
        "active_dtype": "float32",
        "tf_dtype": "float32",
        "tf32_mode": "enabled",
        "tf32_execution_enabled": True,
    }
    for key, expected in expected_precision.items():
        if precision.get(key) != expected:
            raise ValueError(f"shard precision {key} mismatch")
    if require_gpu:
        physical = payload.get("physical_gpus")
        logical = payload.get("logical_gpus")
        if (
            not isinstance(physical, Sequence)
            or isinstance(physical, (str, bytes))
            or not physical
            or not all("GPU" in str(item).upper() for item in physical)
            or not isinstance(logical, Sequence)
            or isinstance(logical, (str, bytes))
            or not logical
            or not all("GPU" in str(item).upper() for item in logical)
        ):
            raise ValueError("trusted GPU shard must record physical and logical GPUs")
    seed = _single_seed(payload)
    if seed not in FULL_ROW_BATCH_SEEDS:
        raise ValueError("raw shard seed is outside the frozen full-row seed set")
    if args.stage != "aggregate" and seed != int(args.batch_seeds[0]):
        raise ValueError("raw shard seed does not match the requested seed")
    _validate_identity(
        "shard randomness identity",
        payload.get("randomness_identity"),
        _randomness_identity(
            seed=seed,
            prepared_input_fingerprint=prepared_fingerprint,
            configuration_identity=configuration_identity,
            route_identity=route_identity,
        ),
    )
    return seed


def _validate_raw_score_shard(payload: Mapping[str, Any], args: argparse.Namespace, spec: RowSpec, *, require_gpu: bool) -> int:
    seed = _validate_common_shard(payload, args, spec, require_gpu=require_gpu)
    manifest = _require_mapping("run_manifest", payload["run_manifest"])
    if manifest.get("stage") != "score-only":
        raise ValueError("score shard stage mismatch")
    score = _finite_vector("score", payload.get("score"), len(spec.parameter_names))
    objective = _finite_scalar("score objective", payload.get("objective"))
    total_log_likelihood = _finite_scalar(
        "score total_log_likelihood",
        payload.get("total_log_likelihood"),
    )
    log_likelihood = _finite_vector("score log_likelihood_by_seed", payload.get("log_likelihood_by_seed"), 1)
    if not math.isclose(objective, log_likelihood[0], rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError("singleton score objective must equal its log likelihood")
    if total_log_likelihood != log_likelihood[0] or not math.isclose(
        objective,
        total_log_likelihood,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise ValueError("score shard paired total value mismatch")
    per_seed_score = payload.get("per_seed_score")
    if not isinstance(per_seed_score, Sequence) or isinstance(per_seed_score, (str, bytes)) or len(per_seed_score) != 1:
        raise ValueError("score per_seed_score must contain exactly one seed vector")
    if _finite_vector("score per_seed_score[0]", per_seed_score[0], len(spec.parameter_names)) != score:
        raise ValueError("singleton score must match its per-seed score")
    if payload.get("score_derivative_provenance") != spec.score_route:
        raise ValueError("score shard derivative provenance mismatch")
    if (
        payload.get("value_score_route_status") != LEDH_SCORE_VALUE_ROUTE_STATUS_SAME
        or payload.get("value_score_same_transport_algorithm") is not True
    ):
        raise ValueError("score shard must preserve the same value/score route")
    if payload.get("score_admission_status") != LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW:
        raise ValueError("raw score shard must be explicitly historical")
    if payload.get("no_autodiff_score_route") is not True:
        raise ValueError("score shard must declare the no-autodiff route")
    for key in ("uses_gradient_tape", "uses_forward_accumulator", "uses_stopped_partial_derivative"):
        if payload.get(key) is not False:
            raise ValueError(f"score shard {key} must be false")
    if payload.get("score_finite") is not True:
        raise ValueError("score shard must be finite")
    devices = payload.get("score_output_devices")
    if require_gpu and (not devices or not all("GPU" in str(item).upper() for item in devices)):
        raise ValueError("score shard must report GPU outputs")
    if require_gpu:
        before = _require_mapping(
            "score_gpu_memory_info_before",
            payload.get("score_gpu_memory_info_before"),
        )
        memory = _require_mapping(
            "score_gpu_memory_info_after",
            payload.get("score_gpu_memory_info_after"),
        )
        for name, snapshot in (("before", before), ("after", memory)):
            try:
                current = _finite_scalar(f"score GPU memory {name} current", snapshot.get("current"))
                peak = _finite_scalar(f"score GPU memory {name} peak", snapshot.get("peak"))
            except ValueError as exc:
                raise ValueError("score shard reset peak memory must be finite") from exc
            if current < 0.0 or peak < 0.0:
                raise ValueError("score shard must report nonnegative TensorFlow memory")
        if payload.get("score_gpu_memory_stats_reset") is not True:
            raise ValueError("score shard must reset TensorFlow memory stats")
        peak_mib = float(memory["peak"]) / (1024.0 * 1024.0)
        budget_pass = peak_mib <= float(args.memory_budget_mib)
        full_row_applicable = bool(
            args.num_particles == spec.full_num_particles
            and args.time_steps == spec.full_time_steps
        )
        diagnostics = _require_mapping("memory_diagnostics", payload.get("memory_diagnostics"))
        expected_diagnostics = {
            "score_memory_budget_pass": budget_pass,
            "full_row_memory_gate_applicable": full_row_applicable,
            "n10000_memory_pass": budget_pass if args.num_particles == spec.full_num_particles else None,
            "peak_mib": peak_mib,
            "budget_mib": float(args.memory_budget_mib),
            "source": "score_gpu_memory_info_after",
        }
        for key, expected in expected_diagnostics.items():
            if diagnostics.get(key) != expected:
                raise ValueError(f"score shard memory diagnostic {key} mismatch")
    return seed


def _validate_raw_fd_shard(payload: Mapping[str, Any], args: argparse.Namespace, spec: RowSpec, *, require_gpu: bool) -> int:
    seed = _validate_common_shard(payload, args, spec, require_gpu=require_gpu)
    manifest = _require_mapping("run_manifest", payload["run_manifest"])
    if manifest.get("stage") != "fd-only":
        raise ValueError("FD shard stage mismatch")
    if payload.get("score_admission_status") != LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW:
        raise ValueError("raw FD shard must be explicitly historical")
    if payload.get("score_derivative_provenance") != spec.score_route:
        raise ValueError("FD shard derivative provenance mismatch")
    if payload.get("value_score_route_status") != LEDH_SCORE_VALUE_ROUTE_STATUS_SAME:
        raise ValueError("FD shard must preserve the same value/score route")
    correctness = _require_mapping("score_correctness", payload.get("score_correctness"))
    if correctness.get("status") != "pass" or correctness.get("kind") != "same_scalar_finite_difference":
        raise ValueError("every FD shard must pass same-scalar correctness")
    if correctness.get("step_policy") != ledh_fd_step_policy_metadata():
        raise ValueError("FD shard step policy mismatch")
    if correctness.get("uses_value_only_scalar_route") is not True:
        raise ValueError("FD shard must use the value-only route")
    score = _finite_vector("FD shard score", payload.get("score"), len(spec.parameter_names))
    fd_policy = _require_mapping("FD policy", correctness.get("fd_policy"))
    entries = fd_policy.get("parameters")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or len(entries) != len(spec.parameter_names):
        raise ValueError("FD shard must contain every parameter exactly once")
    finite_differences = []
    diagnostics = correctness.get("finite_difference_diagnostics")
    if (
        not isinstance(diagnostics, Sequence)
        or isinstance(diagnostics, (str, bytes))
        or len(diagnostics) != len(spec.parameter_names)
    ):
        raise ValueError("FD shard must contain every endpoint diagnostic exactly once")
    for index, name in enumerate(spec.parameter_names):
        entry = _require_mapping("FD parameter", entries[index])
        if entry.get("parameter") != name:
            raise ValueError("FD parameter order mismatch")
        entry_score = _finite_scalar("FD parameter score", entry.get("score"))
        finite_difference = _finite_scalar("FD parameter finite difference", entry.get("finite_difference"))
        if entry_score != score[index]:
            raise ValueError("FD parameter score does not match the referenced score")
        diagnostic = _require_mapping("FD endpoint diagnostic", diagnostics[index])
        if diagnostic.get("parameter") != name:
            raise ValueError("FD endpoint diagnostic parameter order mismatch")
        if diagnostic.get("direction_index") != index:
            raise ValueError("FD endpoint diagnostic direction index mismatch")
        expected_center = _float32_theta(spec)
        center_theta = _finite_vector(
            "FD endpoint diagnostic center theta",
            diagnostic.get("center_theta"),
            len(spec.parameter_names),
        )
        if center_theta != expected_center:
            raise ValueError("FD endpoint diagnostic center theta mismatch")
        minus_endpoint = _require_mapping(
            "FD minus endpoint",
            diagnostic.get("minus_endpoint"),
        )
        plus_endpoint = _require_mapping(
            "FD plus endpoint",
            diagnostic.get("plus_endpoint"),
        )
        if minus_endpoint.get("role") != "minus":
            raise ValueError("FD minus endpoint role mismatch")
        if plus_endpoint.get("role") != "plus":
            raise ValueError("FD plus endpoint role mismatch")
        minus_theta = _finite_vector(
            "FD minus endpoint theta",
            minus_endpoint.get("theta"),
            len(spec.parameter_names),
        )
        plus_theta = _finite_vector(
            "FD plus endpoint theta",
            plus_endpoint.get("theta"),
            len(spec.parameter_names),
        )
        theta_value = float(tf.constant(spec.truth_theta[index], dtype=tf.float32).numpy())
        nominal_step = coordinate_central_difference_step(theta_value)
        basis_value = tf.constant(nominal_step, dtype=tf.float32)
        expected_minus = float((tf.constant(theta_value, tf.float32) - basis_value).numpy())
        expected_plus = float((tf.constant(theta_value, tf.float32) + basis_value).numpy())
        expected_denominator = float(
            (tf.constant(expected_plus, tf.float32) - tf.constant(expected_minus, tf.float32)).numpy()
        )
        expected_step = float((tf.constant(expected_denominator, tf.float32) / 2.0).numpy())
        expected_minus_theta = list(expected_center)
        expected_plus_theta = list(expected_center)
        expected_minus_theta[index] = expected_minus
        expected_plus_theta[index] = expected_plus
        if minus_theta != expected_minus_theta:
            raise ValueError("FD minus endpoint theta mismatch")
        if plus_theta != expected_plus_theta:
            raise ValueError("FD plus endpoint theta mismatch")
        for field, expected in (
            ("theta", theta_value),
            ("nominal_step", nominal_step),
            ("minus_parameter", expected_minus),
            ("plus_parameter", expected_plus),
            ("effective_step", expected_step),
            ("effective_denominator", expected_denominator),
        ):
            if _finite_scalar(f"FD endpoint diagnostic {field}", diagnostic.get(field)) != expected:
                raise ValueError(f"FD endpoint diagnostic {field} mismatch")
        minus_objective = _finite_scalar(
            "FD endpoint diagnostic minus_objective",
            diagnostic.get("minus_objective"),
        )
        plus_objective = _finite_scalar(
            "FD endpoint diagnostic plus_objective",
            diagnostic.get("plus_objective"),
        )
        if _finite_scalar(
            "FD minus endpoint total log likelihood",
            minus_endpoint.get("total_log_likelihood"),
        ) != minus_objective:
            raise ValueError("FD minus endpoint total log likelihood mismatch")
        if _finite_scalar(
            "FD plus endpoint total log likelihood",
            plus_endpoint.get("total_log_likelihood"),
        ) != plus_objective:
            raise ValueError("FD plus endpoint total log likelihood mismatch")
        numerator = _finite_scalar(
            "FD endpoint diagnostic objective_numerator",
            diagnostic.get("objective_numerator"),
        )
        expected_numerator = float(
            (tf.constant(plus_objective, tf.float32) - tf.constant(minus_objective, tf.float32)).numpy()
        )
        if numerator != expected_numerator:
            raise ValueError("FD endpoint diagnostic objective numerator mismatch")
        if diagnostic.get("endpoint_objectives_equal") is not (plus_objective == minus_objective):
            raise ValueError("FD endpoint diagnostic equality flag mismatch")
        expected_fd = float(
            (tf.constant(numerator, tf.float32) / tf.constant(expected_denominator, tf.float32)).numpy()
        )
        diagnostic_fd = _finite_scalar(
            "FD endpoint diagnostic finite_difference",
            diagnostic.get("finite_difference"),
        )
        if diagnostic_fd != expected_fd or diagnostic_fd != finite_difference:
            raise ValueError("FD endpoint diagnostic finite difference mismatch")
        finite_differences.append(finite_difference)
    recomputed = validate_declared_ledh_fd_policy(
        fd_policy,
        score,
        finite_differences,
        spec.parameter_names,
    )
    if recomputed["status"] != "pass":
        raise ValueError("every FD shard must pass recomputed maximum-direction FD check")
    devices = payload.get("value_output_devices")
    if require_gpu and (not devices or not all("GPU" in str(item).upper() for item in devices)):
        raise ValueError("FD shard must report GPU value outputs")
    return seed


def _by_seed(
    paths: Sequence[str],
    validator,
    args: argparse.Namespace,
    spec: RowSpec,
    require_gpu: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    output: dict[int, dict[str, Any]] = {}
    hashes: dict[int, str] = {}
    for path in paths:
        shard_path = Path(path)
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        manifest = _require_mapping("run_manifest", payload.get("run_manifest"))
        declared_output = Path(str(manifest.get("output")))
        if not declared_output.is_absolute():
            declared_output = ROOT / declared_output
        if declared_output.resolve() != shard_path.resolve():
            raise ValueError("shard run manifest output path mismatch")
        seed = validator(payload, args, spec, require_gpu=require_gpu)
        if seed not in FULL_ROW_BATCH_SEEDS:
            raise ValueError(f"unexpected shard seed: {seed}")
        if seed in output:
            raise ValueError(f"duplicate shard seed: {seed}")
        output[seed] = payload
        hashes[seed] = _sha256(shard_path)
    missing = [seed for seed in FULL_ROW_BATCH_SEEDS if seed not in output]
    if missing:
        raise ValueError(f"missing shard seeds: {missing}")
    return output, hashes


def _aggregate(args: argparse.Namespace, spec: RowSpec, started: float) -> dict[str, Any]:
    if args.time_steps != spec.full_time_steps or args.num_particles != spec.full_num_particles:
        raise ValueError("admission aggregation requires the exact full row shape")
    source_payload, source = _load_source_value(args.source_value_artifact, spec)
    if tuple(source["batch_seeds"]) != FULL_ROW_BATCH_SEEDS:
        raise ValueError("source value artifact fixed-seed set mismatch")
    scores, score_hashes = _by_seed(
        args.score_shards,
        _validate_raw_score_shard,
        args,
        spec,
        True,
    )
    fds, fd_hashes = _by_seed(
        args.fd_shards,
        _validate_raw_fd_shard,
        args,
        spec,
        True,
    )
    for seed in FULL_ROW_BATCH_SEEDS:
        if fds[seed].get("score_reference_sha256") != score_hashes[seed]:
            raise ValueError("FD shard score-reference hash mismatch")
        if _finite_vector("FD shard score", fds[seed].get("score"), len(spec.parameter_names)) != _finite_vector(
            "score shard score",
            scores[seed].get("score"),
            len(spec.parameter_names),
        ):
            raise ValueError("FD shard score does not match its score shard")
        if fds[seed].get("prepared_input_fingerprint") != scores[seed].get(
            "prepared_input_fingerprint"
        ):
            raise ValueError("FD shard prepared inputs do not match its score shard")
        for field in (
            "canonical_target_sha256",
            "configuration_identity",
            "route_identity",
            "randomness_identity",
        ):
            if fds[seed].get(field) != scores[seed].get(field):
                raise ValueError(f"FD shard {field} does not match its score shard")
    score_template_hashes = {
        seed: str(
            _require_mapping(
                "score command identity",
                _require_mapping(
                    "score run manifest",
                    scores[seed].get("run_manifest"),
                ).get("command_identity"),
            ).get("template_family_sha256")
        )
        for seed in FULL_ROW_BATCH_SEEDS
    }
    fd_template_hashes = {
        seed: str(
            _require_mapping(
                "FD command identity",
                _require_mapping(
                    "FD run manifest",
                    fds[seed].get("run_manifest"),
                ).get("command_identity"),
            ).get("template_family_sha256")
        )
        for seed in FULL_ROW_BATCH_SEEDS
    }
    if len(set(score_template_hashes.values())) != 1:
        raise ValueError("score shard command template family mismatch")
    if len(set(fd_template_hashes.values())) != 1:
        raise ValueError("FD shard command template family mismatch")
    score_exact_hashes = {
        seed: str(scores[seed]["run_manifest"]["command_identity"]["exact_command_sha256"])
        for seed in FULL_ROW_BATCH_SEEDS
    }
    fd_exact_hashes = {
        seed: str(fds[seed]["run_manifest"]["command_identity"]["exact_command_sha256"])
        for seed in FULL_ROW_BATCH_SEEDS
    }
    if len(set(score_exact_hashes.values())) != len(FULL_ROW_BATCH_SEEDS):
        raise ValueError("score shard exact command identities must be unique by seed")
    if len(set(fd_exact_hashes.values())) != len(FULL_ROW_BATCH_SEEDS):
        raise ValueError("FD shard exact command identities must be unique by seed")
    score_outputs = [str(scores[seed]["run_manifest"]["output"]) for seed in FULL_ROW_BATCH_SEEDS]
    fd_outputs = [str(fds[seed]["run_manifest"]["output"]) for seed in FULL_ROW_BATCH_SEEDS]
    if len(set((*score_outputs, *fd_outputs))) != 2 * len(FULL_ROW_BATCH_SEEDS):
        raise ValueError("score and FD shard output paths must be unique")
    total_log_likelihood_by_seed = {
        seed: _finite_scalar(
            "score shard paired total log likelihood",
            scores[seed].get("total_log_likelihood"),
        )
        for seed in FULL_ROW_BATCH_SEEDS
    }
    aggregate_total_log_likelihood = statistics.fmean(
        total_log_likelihood_by_seed.values()
    )
    aggregate_average_log_likelihood = (
        aggregate_total_log_likelihood / float(spec.full_time_steps)
    )
    aggregate_score = [
        statistics.fmean(float(scores[seed]["score"][index]) for seed in FULL_ROW_BATCH_SEEDS)
        for index in range(len(spec.parameter_names))
    ]
    aggregate_fd = []
    for index, name in enumerate(spec.parameter_names):
        values = []
        for seed in FULL_ROW_BATCH_SEEDS:
            correctness = _require_mapping("score_correctness", fds[seed]["score_correctness"])
            policy = _require_mapping("FD policy", correctness.get("fd_policy"))
            entries = policy["parameters"]
            entry = _require_mapping("FD parameter", entries[index])
            if entry.get("parameter") != name:
                raise ValueError("FD parameter order mismatch")
            values.append(float(entry["finite_difference"]))
        aggregate_fd.append(statistics.fmean(values))
    aggregate_fd_policy = evaluate_ledh_fd_policy(
        aggregate_score,
        aggregate_fd,
        spec.parameter_names,
    )
    per_seed_records = []
    for seed in FULL_ROW_BATCH_SEEDS:
        correctness = _require_mapping(
            "per-seed FD correctness",
            fds[seed]["score_correctness"],
        )
        per_seed_records.append(
            {
                "seed": seed,
                "total_log_likelihood": total_log_likelihood_by_seed[seed],
                "score": _finite_vector(
                    "per-seed score",
                    scores[seed]["score"],
                    len(spec.parameter_names),
                ),
                "canonical_target_sha256": scores[seed][
                    "canonical_target_sha256"
                ],
                "source_value_artifact_sha256": scores[seed][
                    "source_value_artifact_sha256"
                ],
                "configuration_sha256": scores[seed]["configuration_identity"][
                    "sha256"
                ],
                "route_sha256": scores[seed]["route_identity"]["sha256"],
                "randomness_sha256": scores[seed]["randomness_identity"][
                    "sha256"
                ],
                "score_shard_path": score_outputs[
                    FULL_ROW_BATCH_SEEDS.index(seed)
                ],
                "score_shard_sha256": score_hashes[seed],
                "score_exact_command_sha256": score_exact_hashes[seed],
                "fd_shard_path": fd_outputs[FULL_ROW_BATCH_SEEDS.index(seed)],
                "fd_shard_sha256": fd_hashes[seed],
                "fd_exact_command_sha256": fd_exact_hashes[seed],
                "fd_policy": correctness["fd_policy"],
                "finite_difference_diagnostics": correctness[
                    "finite_difference_diagnostics"
                ],
            }
        )
    per_seed_peak_mib = {
        seed: float(scores[seed]["score_gpu_memory_info_after"]["peak"]) / (1024.0 * 1024.0)
        for seed in FULL_ROW_BATCH_SEEDS
    }
    peak_mib = max(per_seed_peak_mib.values())
    memory_pass = peak_mib <= args.memory_budget_mib
    score_artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": spec.row_id,
        "source_value_artifact": args.source_value_artifact,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": source["target_observation_policy"],
        "theta_coordinate_system": source["theta_coordinate_system"],
        "score_parameter_names": list(spec.parameter_names),
        "score": aggregate_score,
        "score_derivative_provenance": spec.score_route,
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "claims_exact_native_actual_sv_likelihood": False,
        "score_correctness": {
            "kind": "same_scalar_finite_difference",
            "status": "pass",
            "step_policy": ledh_fd_step_policy_metadata(),
            "per_seed_individual_direction_status": "pass",
            "per_seed_count": len(per_seed_records),
            "aggregate_fd_policy": aggregate_fd_policy,
            "aggregate_fd_policy_role": "explanatory_only_not_admission",
        },
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW,
        "score_precision": {
            "dtype": "float32",
            "active_dtype": "float32",
            "tf_dtype": "float32",
            "tf32_mode": "enabled",
            "tf32_execution_enabled": True,
        },
        "memory_diagnostics": {
            "n10000_memory_pass": memory_pass,
            "peak_mib": peak_mib,
            "budget_mib": args.memory_budget_mib,
            "source": "max_per_seed_score_gpu_memory_info_after",
        },
    }
    validate_ledh_score_artifact(
        score_artifact,
        source_value_artifact=source_payload,
        expected_row_id=spec.row_id,
        require_admitted=False,
    )
    return _progress(
        args,
        spec,
        "completed" if memory_pass else "blocked_memory_budget",
        True,
        started,
        evidence_class="offline_aggregate_of_validated_trusted_gpu_shards",
        score_admission_status=score_artifact["score_admission_status"],
        source_value_artifact=args.source_value_artifact,
        source_value_artifact_sha256=_source_value_sha256(spec),
        canonical_target_sha256=_canonical_target_sha256(spec),
        configuration_identity=_configuration_identity(args, spec),
        route_identity=_route_identity(args, spec),
        score_parameter_names=list(spec.parameter_names),
        total_log_likelihood=aggregate_total_log_likelihood,
        average_log_likelihood=aggregate_average_log_likelihood,
        total_log_likelihood_by_seed={
            str(seed): total_log_likelihood_by_seed[seed]
            for seed in FULL_ROW_BATCH_SEEDS
        },
        score=aggregate_score,
        aggregate_finite_difference=aggregate_fd,
        per_seed_records=per_seed_records,
        seed_invariant_score_command_template_family_sha256=next(
            iter(score_template_hashes.values())
        ),
        seed_invariant_fd_command_template_family_sha256=next(
            iter(fd_template_hashes.values())
        ),
        score_correctness=score_artifact["score_correctness"],
        score_derivative_provenance=spec.score_route,
        value_score_route_status=LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        memory_diagnostics=score_artifact["memory_diagnostics"],
        execution_strategy={
            "kind": "seed_sharded_trusted_gpu_processes",
            "segmented_execution_disclosed": True,
            "monolithic_batch_memory_claim": False,
            "monolithic_batch_runtime_claim": False,
        },
        per_seed_score_peak_mib={str(seed): per_seed_peak_mib[seed] for seed in FULL_ROW_BATCH_SEEDS},
        score_artifact=score_artifact,
    )


def _write_markdown(path: Path, result: Mapping[str, Any], json_path: Path) -> None:
    lines = [
        "# Compact LEDH Score GPU/XLA Artifact",
        "",
        f"- JSON: `{json_path}`",
        f"- Status: `{result.get('artifact_status')}`",
        f"- Row: `{result.get('row_id')}`",
        f"- Stage: `{result.get('run_manifest', {}).get('stage')}`",
        f"- Evidence class: `{result.get('evidence_class')}`",
        f"- Score correctness: `{result.get('score_correctness')}`",
        f"- Memory: `{result.get('memory_diagnostics', result.get('score_gpu_memory_info_after'))}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result.get("nonclaims", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv, validate=False)
    args.invoked_command_argv = [sys.argv[0], *raw_argv]
    spec = ROW_SPECS[args.row]
    output = Path(args.output)
    started = time.perf_counter()
    _write_json_atomic(output, _progress(args, spec, "started", False, started))
    try:
        _validate_args(args, spec)
        _validate_exact_execution_command(args, spec, args.invoked_command_argv)
        _write_json_atomic(output, _progress(args, spec, "initialized", False, started))
        if args.stage == "score-only":
            result = _score_only(args, spec, started)
        elif args.stage == "fd-only":
            result = _fd_only(args, spec, started)
        else:
            result = _aggregate(args, spec, started)
    except BaseException as exc:  # noqa: BLE001
        failure = _progress(
            args,
            spec,
            "failed",
            True,
            started,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        _write_json_atomic(output, failure)
        raise
    _write_json_atomic(output, result)
    if args.markdown_output:
        _write_markdown(Path(args.markdown_output), result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("artifact_status") != "completed":
        raise SystemExit(f"hard gate failed: {result.get('artifact_status')}")


if __name__ == "__main__":
    main()
