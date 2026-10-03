#!/usr/bin/env python3
"""Measure q=20 Phase 9B P1 executable readiness without posterior claims."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PHASE0_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md"
P1_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
FACTOR_BACKEND = "tensorflow_eigh_strict_factor_cached"
STRICT_BACKEND = "tensorflow_eigh_strict"
POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"
ARM_NAMES = ("factor", "strict")
CHAIN_COUNT = 4
DIMENSION = 4
DEFAULT_CHUNK_RESULTS = 500
REQUIRED_CHUNKS_PER_ARM = 6
CLAIM_BOUNDARY = "phase9b_m4_p0_executable_readiness_only"
ROUTE_POLICY_MARKER = "run_batched_hmc"
MANAGED_SESSION_TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
GPU_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "TF_FORCE_GPU_ALLOW_GROWTH",
    "BAYESFILTER_SELECTED_GPU_UUID",
    "BAYESFILTER_GPU_SELECTION_POLICY_ID",
    "TF_XLA_FLAGS",
    "XLA_FLAGS",
    "NVIDIA_TF32_OVERRIDE",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.campaign_budget_ledger import (
    CampaignBudgetLedger,
    CampaignBudgetLedgerError,
)
from bayesfilter.runtime.display_gpu_policy import (
    GPUPlacementError,
    POLICY_ID as GPU_PLACEMENT_POLICY_ID,
    check_runtime_headroom,
    select_and_pin_gpu,
)


class ReadinessDiagnosticError(RuntimeError):
    """Raised when the bounded executable-readiness contract fails."""


def _select_phase9b_gpu() -> Mapping[str, Any]:
    try:
        return select_and_pin_gpu()
    except GPUPlacementError as exc:
        raise ReadinessDiagnosticError(str(exc)) from exc


def _handle_signal(signum: int, _frame: Any) -> None:
    raise ReadinessDiagnosticError(f"diagnostic interrupted by signal {signum}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not value == value or value in (float("inf"), float("-inf")):
            raise ReadinessDiagnosticError("diagnostic values must be finite")
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ReadinessDiagnosticError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _git_payload() -> Mapping[str, Any]:
    status = _git(("git", "status", "--porcelain"))
    return {
        "commit": _git(("git", "rev-parse", "HEAD")),
        "worktree_dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _runtime_provenance() -> Mapping[str, Any]:
    """Capture bounded process and GPU-launch context for a readiness receipt."""
    return {
        "schema": "bayesfilter.runtime_provenance.v1",
        "trust_basis": MANAGED_SESSION_TRUST_BASIS,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "conda": {
            "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefix": os.environ.get("CONDA_PREFIX"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "working_directory": str(Path.cwd()),
        "gpu_environment": {
            name: os.environ.get(name) for name in GPU_ENVIRONMENT_KEYS
        },
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReadinessDiagnosticError(f"unable to load source-owned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _allocator_info(tf: Any) -> Mapping[str, Any]:
    try:
        value = tf.config.experimental.get_memory_info("GPU:0")
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise ReadinessDiagnosticError(
            f"GPU allocator telemetry is unavailable: {type(exc).__name__}"
        ) from exc
    result = {str(key): int(item) for key, item in dict(value).items()}
    if not {"current", "peak"}.issubset(result):
        raise ReadinessDiagnosticError("GPU allocator telemetry lacks current and peak bytes")
    return result


def _finite_tensor(tf: Any, value: Any, label: str) -> bool:
    try:
        return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ReadinessDiagnosticError(f"cannot verify finite {label}") from exc


def _movement_receipt(
    tf: Any,
    *,
    pre_chunk_state: Any,
    samples: Any,
    label: str,
) -> Mapping[str, Any]:
    """Apply the shared controller's per-chain movement semantics to one call."""
    from bayesfilter.inference.neutra_hmc import _chain_moved

    try:
        moved = _chain_moved(
            tf.convert_to_tensor(pre_chunk_state), tf.convert_to_tensor(samples)
        )
        per_chain = tuple(bool(item) for item in moved.numpy().tolist())
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ReadinessDiagnosticError(f"cannot verify {label} movement") from exc
    if len(per_chain) != CHAIN_COUNT:
        raise ReadinessDiagnosticError(
            f"{label} movement shape does not match {CHAIN_COUNT} chains"
        )
    stationary = [index for index, did_move in enumerate(per_chain) if not did_move]
    if stationary:
        raise ReadinessDiagnosticError(
            f"{label} had a stationary chain: {stationary}"
        )
    return {
        "status": "PASS_ALL_CHAINS_MOVED",
        "criterion": "per_chain_any_sequential_state_change",
        "per_chain_moved": per_chain,
    }


def _check_deadline(started: float, max_seconds: float, label: str) -> None:
    if time.monotonic() - started > max_seconds:
        raise ReadinessDiagnosticError(
            f"readiness diagnostic exceeded its {max_seconds:g}s cap at {label}"
        )


def _complete_schedule_forecast(
    results: Mapping[str, Mapping[str, Any]], *, startup_seconds: float, chunk_results: int
) -> Mapping[str, Any]:
    if chunk_results != DEFAULT_CHUNK_RESULTS:
        raise ReadinessDiagnosticError("readiness timing must use the P1 500-result chunk")
    startup = float(startup_seconds)
    if not math.isfinite(startup) or startup < 0.0:
        raise ReadinessDiagnosticError("startup timing must be finite and nonnegative")
    per_arm: dict[str, float] = {}
    chunk_forecasts: list[float] = []
    for arm in ARM_NAMES:
        try:
            timing = results[arm]["timing"]
            measured = {
                key: float(timing[key])
                for key in (
                    "profile_seconds", "bridge_seconds", "chart_seconds",
                    "tuning_seconds", "first_compiled_call_seconds",
                    "steady_state_chunk_seconds", "serialization_seconds",
                    "cleanup_seconds",
                )
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ReadinessDiagnosticError(f"{arm} timing record is incomplete") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in measured.values()):
            raise ReadinessDiagnosticError(f"{arm} timing values must be finite and nonnegative")
        steady = measured["steady_state_chunk_seconds"]
        if steady <= 0.0 or measured["first_compiled_call_seconds"] <= 0.0:
            raise ReadinessDiagnosticError(f"{arm} controller timings must be positive")
        setup = sum(
            measured[key]
            for key in ("profile_seconds", "bridge_seconds", "chart_seconds", "tuning_seconds")
        )
        serialization = measured["serialization_seconds"]
        chunk_forecast = steady + 2 * serialization
        chunk_forecasts.append(chunk_forecast)
        per_arm[arm] = (
            setup
            + measured["first_compiled_call_seconds"]
            + (REQUIRED_CHUNKS_PER_ARM - 1) * steady
            + 2 * REQUIRED_CHUNKS_PER_ARM * serialization
            + measured["cleanup_seconds"]
            + chunk_forecast
            + steady
        )
    return {
        "p1_required_chunks_per_arm": REQUIRED_CHUNKS_PER_ARM,
        "p1_forecast_seconds_per_arm": per_arm,
        "p1_complete_schedule_forecast_seconds": startup + sum(per_arm.values()),
        "sequential_chunk_forecast_seconds": max(chunk_forecasts),
        "startup_seconds": startup,
        "serialization_derivation": "two sample copies per P1 draw: chunk and cumulative archive; use the full measured two-chunk serialization cost for each copy conservatively",
        "reserve_derivation": "one measured steady-state chunk plus its forecast serialization per arm",
        "termination_grace_derivation": "one additional measured steady-state chunk per arm, distinct from reserve",
    }


def _xla_receipt(program: Any, state: Any, seed: Any) -> Mapping[str, Any]:
    compiler_ir = getattr(program, "experimental_get_compiler_ir", None)
    if not callable(compiler_ir):
        raise ReadinessDiagnosticError("compiled controller does not expose XLA compiler IR")
    try:
        ir = compiler_ir(state, seed)(stage="hlo")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ReadinessDiagnosticError("XLA compiler IR probe failed") from exc
    encoded = str(ir).encode("utf-8")
    if not encoded:
        raise ReadinessDiagnosticError("XLA compiler IR probe returned empty output")
    return {
        "status": "verified_by_compiler_ir_probe",
        "hlo_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _run_arm(
    *,
    tf: Any,
    p1_runner: Any,
    source_module: Any,
    output: Path,
    arm_name: str,
    chunk_results: int,
    started: float,
    max_seconds: float,
) -> Mapping[str, Any]:
    arm_started = time.monotonic()
    backend = FACTOR_BACKEND if arm_name == "factor" else STRICT_BACKEND
    arm_root = output / arm_name
    arm_root.mkdir(parents=True, exist_ok=False)
    p1_runner._ACTIVE_CONTEXT["output"] = output
    seed_map = p1_runner._seed_map(output, arm_name)
    stages: dict[str, Any] = {}
    profile_started = time.monotonic()
    profile = p1_runner._arm_profile(source_module, backend, arm_name)
    stages["profile_seconds"] = time.monotonic() - profile_started
    _check_deadline(started, max_seconds, f"{arm_name}:profile")

    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    bridge_started = time.monotonic()
    bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=backend
    )
    if str(bridge.target_signature) != TARGET_SIGNATURE:
        raise ReadinessDiagnosticError(f"{arm_name} target signature changed")
    stages["bridge_seconds"] = time.monotonic() - bridge_started
    chart_started = time.monotonic()
    chart_map, checkpoints, preflight = source_module._build_fresh_chart(
        tf,
        bridge,
        p1_runner.CHART_INDEX,
        p1_runner.COMPONENT_ID,
        profile,
        arm_root,
    )
    chart = p1_runner._select_chart(chart_map, p1_runner.BETA)
    stages["chart_seconds"] = time.monotonic() - chart_started
    _check_deadline(started, max_seconds, f"{arm_name}:chart")

    tuning_started = time.monotonic()
    tuning_root = arm_root / "tuning" / f"chart-{p1_runner.CHART_INDEX}" / f"beta-{p1_runner.BETA:g}"
    tuning = source_module._tune_scope(
        tf,
        bridge,
        chart,
        chart_index=p1_runner.CHART_INDEX,
        beta=p1_runner.BETA,
        output_dir=tuning_root,
        profile=profile,
        scope_index=p1_runner.SCOPE_INDEX,
    )
    handoff = tuning.get("_live_handoff")
    if handoff is None:
        raise ReadinessDiagnosticError(f"{arm_name} did not produce a live handoff")
    stages["tuning_seconds"] = time.monotonic() - tuning_started
    _check_deadline(started, max_seconds, f"{arm_name}:tuning")

    from bayesfilter.inference.neutra_hmc import _build_batched_hmc_program

    initial_state = tf.convert_to_tensor(profile.initial_state_bank, tf.float64)
    if tuple(initial_state.shape) != (CHAIN_COUNT, DIMENSION):
        raise ReadinessDiagnosticError(f"{arm_name} initial state shape is not P1 shape")
    program = _build_batched_hmc_program(
        adapter=handoff.transformed_adapter,
        num_results=chunk_results,
        num_burnin_steps=0,
        step_size=float(handoff.step_size),
        num_leapfrog_steps=int(handoff.num_leapfrog_steps),
        state_shape=(CHAIN_COUNT, DIMENSION),
        jit_compile=True,
    )
    input_signature = tuple(
        {
            "shape": tuple(item.shape.as_list()),
            "dtype": item.dtype.name,
        }
        for item in program.input_signature
    )
    first_seed = tf.constant(seed_map["warmup"], tf.int32)
    steady_seed = tf.constant(seed_map["retained"], tf.int32)
    allocator_before = _allocator_info(tf)
    first_started = time.monotonic()
    first_samples, first_trace = program(initial_state, first_seed)
    first_samples.shape.assert_is_compatible_with((chunk_results, CHAIN_COUNT, DIMENSION))
    first_finite = _finite_tensor(tf, first_samples, f"{arm_name} first samples")
    first_target_finite = _finite_tensor(
        tf, first_trace["target_log_prob"], f"{arm_name} first target log probability"
    )
    first_seconds = time.monotonic() - first_started
    stages["first_compiled_call_seconds"] = first_seconds
    allocator_after_first = _allocator_info(tf)
    if not all((first_finite, first_target_finite)):
        raise ReadinessDiagnosticError(f"{arm_name} first controller call returned nonfinite values")
    first_movement = _movement_receipt(
        tf,
        pre_chunk_state=initial_state,
        samples=first_samples,
        label=f"{arm_name} first compiled controller call",
    )
    _check_deadline(started, max_seconds, f"{arm_name}:first-call")

    steady_input = first_samples[-1]
    steady_started = time.monotonic()
    steady_samples, steady_trace = program(steady_input, steady_seed)
    steady_samples.shape.assert_is_compatible_with((chunk_results, CHAIN_COUNT, DIMENSION))
    steady_finite = _finite_tensor(tf, steady_samples, f"{arm_name} steady samples")
    steady_target_finite = _finite_tensor(
        tf, steady_trace["target_log_prob"], f"{arm_name} steady target log probability"
    )
    steady_seconds = time.monotonic() - steady_started
    stages["steady_state_chunk_seconds"] = steady_seconds
    allocator_after_steady = _allocator_info(tf)
    if not all((steady_finite, steady_target_finite)):
        raise ReadinessDiagnosticError(f"{arm_name} steady controller call returned nonfinite values")
    steady_movement = _movement_receipt(
        tf,
        pre_chunk_state=steady_input,
        samples=steady_samples,
        label=f"{arm_name} steady-state controller call",
    )
    tracing_count = int(program.experimental_get_tracing_count())
    if tracing_count != 1:
        raise ReadinessDiagnosticError(
            f"{arm_name} controller retraced {tracing_count} times"
        )
    xla = _xla_receipt(program, initial_state, first_seed)
    _check_deadline(started, max_seconds, f"{arm_name}:steady-state")

    serialization_started = time.monotonic()
    archive = p1_runner._archive_callback(arm_root, tf, arm_name)
    sample_archives = []
    for chunk_index, (samples, seed) in enumerate(
        ((first_samples, first_seed), (steady_samples, steady_seed))
    ):
        model_samples = tf.reshape(
            chart.forward_batch(tf.reshape(samples, (-1, DIMENSION))), samples.shape
        )
        sample_archives.append(
            archive(
                stage="warmup",
                chunk_index=chunk_index,
                latent_samples=samples,
                model_samples=model_samples,
                seed=seed,
                cumulative=False,
            )
        )
    sample_diagnostics = p1_runner._diagnostic_summary(tf, steady_samples)
    receipt = {
        "schema": "bayesfilter.ssl_lstm_q20.phase9b_executable_readiness_arm.v1",
        "status": "PASS_READINESS_ARM",
        "arm": arm_name,
        "backend": backend,
        "policy_id": POLICY_ID,
        "scope": {
            "scope_index": p1_runner.SCOPE_INDEX,
            "chart_index": p1_runner.CHART_INDEX,
            "beta": p1_runner.BETA,
        },
        "target_signature": TARGET_SIGNATURE,
        "handoff": tuning.get("handoff"),
        "tuning_artifact": tuning.get("tuning_artifact"),
        "checkpoint_count": len(checkpoints),
        "preflight_count": len(preflight),
        "input_signature": input_signature,
        "chunk_results_per_chain": chunk_results,
        "chain_count": CHAIN_COUNT,
        "dimension": DIMENSION,
        "first_compiled_call_seconds": first_seconds,
        "steady_state_chunk_seconds": steady_seconds,
        "controller_tracing_count": tracing_count,
        "xla": xla,
        "allocator_before": allocator_before,
        "allocator_after_first": allocator_after_first,
        "allocator_after_steady": allocator_after_steady,
        "movement": {
            "first_compiled_call": first_movement,
            "steady_state_call": steady_movement,
        },
        "seed_namespace": seed_map,
        "sample_archives": sample_archives,
        "sample_diagnostics": sample_diagnostics,
        "claim_boundary": CLAIM_BOUNDARY,
        "nonclaims": [
            "no posterior correctness",
            "no convergence promotion",
            "no sampler ranking",
            "no scientific or default promotion",
        ],
    }
    _write_json(arm_root / "readiness.json", receipt)
    stages["serialization_seconds"] = time.monotonic() - serialization_started
    cleanup_started = time.monotonic()
    del first_samples, first_trace, steady_samples, steady_trace, program, samples, model_samples
    gc.collect()
    stages["cleanup_seconds"] = time.monotonic() - cleanup_started
    stages["arm_elapsed_seconds"] = time.monotonic() - arm_started
    _check_deadline(started, max_seconds, f"{arm_name}:archive-cleanup")
    return {**receipt, "timing": stages}


def _failure_payload(
    *,
    output: Path,
    started: float,
    started_at_utc: str,
    exc: BaseException,
    stages: Mapping[str, Any],
    ledger: CampaignBudgetLedger | None,
    runtime_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema": "bayesfilter.ssl_lstm_q20.phase9b_executable_readiness.v1",
        "status": "FAIL_PHASE9B_M4_P0_READINESS",
        "error_type": type(exc).__name__,
        "error": str(exc)[:12000],
        "traceback": traceback.format_exc()[:20000],
        "output_dir": str(output),
        "started_at_utc": started_at_utc,
        "elapsed_seconds": time.monotonic() - started,
        "stages": dict(stages),
        "target_signature": TARGET_SIGNATURE,
        "policy_id": POLICY_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "runtime_provenance": dict(runtime_provenance),
        "git": _git_payload(),
        "source_hashes": {
            "diagnostic": _sha256(SCRIPT),
            "phase0_plan": _sha256(PHASE0_PLAN),
            "p1_runner": _sha256(P1_RUNNER),
        },
        "nonclaims": [
            "no posterior correctness",
            "no convergence promotion",
            "no sampler ranking",
            "no scientific or default promotion",
        ],
    }
    if ledger is not None:
        payload["budget_ledger"] = {
            "path": str(ledger.path.relative_to(ROOT)),
            "checksum": ledger.checksum(),
            "remaining_seconds": ledger.remaining_seconds(),
        }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-results", type=int, default=DEFAULT_CHUNK_RESULTS)
    parser.add_argument("--max-seconds", required=True, type=float)
    args = parser.parse_args(argv)
    if args.chunk_results != DEFAULT_CHUNK_RESULTS:
        raise SystemExit("readiness diagnostic requires the P1 500-result chunk")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0.0:
        raise SystemExit("max-seconds must be positive and finite")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_PHASE9B_M4_P0_READINESS", "error": "output directory already exists"}), file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    started = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    runtime_provenance = _runtime_provenance()
    stages: dict[str, Any] = {}
    ledger: CampaignBudgetLedger | None = None
    budget_settled = False
    current_arm: str | None = None
    gpu_selection: Mapping[str, Any] | None = None
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").strip().lower() != "true":
            raise ReadinessDiagnosticError("TF_FORCE_GPU_ALLOW_GROWTH=true is required before TensorFlow import")
        gpu_selection = _select_phase9b_gpu()
        runtime_provenance = _runtime_provenance()
        stages["gpu_selection"] = gpu_selection
        _write_json(output / "gpu-selection.json", gpu_selection)
        p1_runner = _load_module(P1_RUNNER, "phase9b_p1_readiness_runner")
        source_inputs = p1_runner._verify_source_inputs()
        source_hashes = dict(p1_runner._readiness_source_hashes())
        source_hashes["phase0_plan"] = _sha256(PHASE0_PLAN)
        _write_json(
            output / "run_start.json",
            {
                "schema": "bayesfilter.ssl_lstm_q20.phase9b_executable_readiness_run_start.v1",
                "status": "RUN_STARTED",
                "started_at_utc": started_at_utc,
                "command": list(sys.argv),
                "output_dir": str(output),
                "max_seconds": args.max_seconds,
                "chunk_results": args.chunk_results,
                "target_signature": TARGET_SIGNATURE,
                "policy_id": POLICY_ID,
                "gpu_placement_policy_id": GPU_PLACEMENT_POLICY_ID,
                "claim_boundary": CLAIM_BOUNDARY,
                "source_hashes": source_hashes,
                "gpu_selection": {
                    "path": str((output / "gpu-selection.json").relative_to(ROOT)),
                    "policy_id": gpu_selection["policy_id"],
                    "selected": gpu_selection["selected"],
                    "reason": gpu_selection["reason"],
                },
                "runtime_provenance": runtime_provenance,
                "git": _git_payload(),
            },
        )
        ledger = p1_runner._open_campaign_ledger(
            output, minimum_remaining_seconds=args.max_seconds
        )
        ledger.reserve_chunk(
            attempt_id=output.name,
            arm="readiness_diagnostic",
            chunk_index=0,
            reserve_seconds=args.max_seconds,
            output_root=output,
            seed_namespace={arm: p1_runner._seed_map(output, arm) for arm in ARM_NAMES},
            source_hash=p1_runner._campaign_source_hash(),
            plan_hash=p1_runner._campaign_plan_hash(),
        )
        stages["tensorflow_import_started"] = time.monotonic() - started
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_selection["selected"]["uuid"]:
            raise ReadinessDiagnosticError("GPU environment changed after placement selection")
        import tensorflow as tf

        stages["tensorflow_import_seconds"] = time.monotonic() - started - stages["tensorflow_import_started"]
        _check_deadline(started, args.max_seconds, "tensorflow-import")
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        device_started = time.monotonic()
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
        logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
        if len(physical_gpus) != 1 or len(logical_gpus) != 1:
            raise ReadinessDiagnosticError(
                f"expected one visible GPU, got physical={len(physical_gpus)} logical={len(logical_gpus)}"
            )
        if not bool(tf.config.experimental.tensor_float_32_execution_enabled()):
            raise ReadinessDiagnosticError("TF32 execution is not enabled")
        stages["device_initialization_seconds"] = time.monotonic() - device_started
        stages["memory_policy"] = memory_policy
        stages["physical_gpus"] = [str(item.name) for item in physical_gpus]
        stages["logical_gpus"] = [str(item.name) for item in logical_gpus]
        stages["gpu_selection"] = gpu_selection
        stages["tf32_execution_enabled"] = True
        stages["tensorflow_version"] = tf.__version__
        import tensorflow_probability as tfp

        stages["tensorflow_probability_version"] = tfp.__version__
        _check_deadline(started, args.max_seconds, "device-initialization")
        source_module = p1_runner._load_phase9a_runner()
        results: dict[str, Mapping[str, Any]] = {}
        startup_seconds = time.monotonic() - started
        for current_arm in ARM_NAMES:
            _check_deadline(started, args.max_seconds, f"{current_arm}:entry")
            results[current_arm] = _run_arm(
                tf=tf,
                p1_runner=p1_runner,
                source_module=source_module,
                output=output,
                arm_name=current_arm,
                chunk_results=args.chunk_results,
                started=started,
                max_seconds=args.max_seconds,
            )
            headroom = check_runtime_headroom(gpu_selection, int(_allocator_info(tf)["peak"]))
            _write_json(output / current_arm / "gpu-headroom.json", headroom)
        factor = results["factor"]["timing"]
        strict = results["strict"]["timing"]
        forecast = _complete_schedule_forecast(
            results, startup_seconds=startup_seconds, chunk_results=args.chunk_results
        )
        campaign_forecast = forecast["p1_complete_schedule_forecast_seconds"]
        current_sources = dict(p1_runner._readiness_source_hashes())
        current_sources["phase0_plan"] = _sha256(PHASE0_PLAN)
        if current_sources != source_hashes:
            raise ReadinessDiagnosticError("readiness source changed during the diagnostic")
        _check_deadline(started, args.max_seconds, "closeout")
        elapsed = time.monotonic() - started
        ledger.settle_arm(
            attempt_id=output.name,
            arm="readiness_diagnostic",
            measured_seconds=elapsed,
            status="completed",
        )
        budget_settled = True
        ledger.finish_attempt(attempt_id=output.name, status="completed")
        result = {
            "schema": "bayesfilter.ssl_lstm_q20.phase9b_executable_readiness.v1",
            "status": "PASS_PHASE9B_M4_P0_RUNTIME_DIAGNOSTIC",
            "policy_id": POLICY_ID,
            "target_signature": TARGET_SIGNATURE,
            "scope": {"scope_index": 2, "chart_index": 0, "beta": 1.0},
            "source_inputs": source_inputs,
            "command": list(sys.argv),
            "started_at_utc": started_at_utc,
            "timing": {
                "process_to_import_seconds": stages["tensorflow_import_started"],
                "tensorflow_import_seconds": stages["tensorflow_import_seconds"],
                "device_initialization_seconds": stages["device_initialization_seconds"],
                "factor": factor,
                "strict": strict,
                **forecast,
            },
            "gpu": {
                "placement_policy_id": GPU_PLACEMENT_POLICY_ID,
                "selection": gpu_selection,
                "physical_devices": stages["physical_gpus"],
                "logical_devices": stages["logical_gpus"],
                "memory_policy": memory_policy,
                "tf32_execution_enabled": True,
                "tensorflow_version": stages["tensorflow_version"],
                "tensorflow_probability_version": stages["tensorflow_probability_version"],
                "allocator_after_diagnostic": _allocator_info(tf),
            },
            "source_hashes": source_hashes,
            "runtime_provenance": runtime_provenance,
            "git": _git_payload(),
            "elapsed_seconds": elapsed,
            "budget_ledger": {
                "path": str(ledger.path.relative_to(ROOT)),
                "checksum": ledger.checksum(),
                "remaining_seconds": ledger.remaining_seconds(),
            },
            "claim_boundary": CLAIM_BOUNDARY,
            "nonclaims": [
                "no posterior correctness",
                "no convergence promotion",
                "no sampler ranking",
                "no scientific or default promotion",
            ],
        }
        encoded = _canonical_bytes(result)
        _write_json(output / "result.json", result)
        _write_json(
            output / "run_manifest.json",
            {**result, "manifest_sha256": hashlib.sha256(encoded).hexdigest()},
        )
        print(json.dumps({"status": result["status"], "output_dir": str(output), "forecast_seconds": campaign_forecast}, sort_keys=True))
        return 0
    except Exception as exc:
        stages["failed_arm"] = current_arm
        if ledger is not None and not budget_settled:
            try:
                ledger.settle_arm(
                    attempt_id=output.name,
                    arm="readiness_diagnostic",
                    measured_seconds=max(0.0, time.monotonic() - started),
                    status="failed",
                    failure_class=type(exc).__name__,
                )
            except Exception:
                pass
        if ledger is not None:
            try:
                ledger.finish_attempt(
                    attempt_id=output.name,
                    status="failed",
                    failure_class=type(exc).__name__,
                )
            except Exception:
                pass
        payload = _failure_payload(
            output=output,
            started=started,
            started_at_utc=started_at_utc,
            exc=exc,
            stages=stages,
            ledger=ledger,
            runtime_provenance=runtime_provenance,
        )
        _write_json(output / "failure.json", payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
