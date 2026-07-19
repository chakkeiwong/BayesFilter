#!/usr/bin/env python3
"""Dual-averaging repair for the SSL-LSTM A4 HMC acquisition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_HARNESS_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py"
)
HARNESS_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_2026_07_14.py"
)
TEST_PATH = Path("tests/test_ssl_lstm_a4_hmc_repair_02.py")
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-"
    "repair-02-plan-2026-07-14.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-"
    "repair-02-result-2026-07-14.md"
)
REPAIR_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
    "hmc-acquisition/repair-02"
)
ARCHIVE_DIR = REPAIR_ROOT / "private"
ADAPTATION_OUTPUT = REPAIR_ROOT / "adaptation.json"
SEGMENT_OUTPUT = REPAIR_ROOT / "segment-0.json"
ADAPTATION_LABEL = "repair_02_adaptation"
SEGMENT_LABEL = "repair_02_segment_0"
ADAPTATION_SEED = (20260714, 1620)
SEGMENT_SEED = (20260714, 1630)
INITIAL_STEP_SIZE = 0.3925
NUM_LEAPFROG_STEPS = 4
NUM_ADAPTATION_STEPS = 256
NUM_WARMUP_STEPS = 320
SCREEN_DRAWS = 64
TARGET_ACCEPTANCE = 0.70
REPAIR_ACCEPTANCE_MIN = 0.55
REPAIR_ACCEPTANCE_MAX = 0.85
PER_CHAIN_SAFETY_MIN = 0.20
PER_CHAIN_SAFETY_MAX = 0.95
FROZEN_STEP_MIN = 1.0e-4
FROZEN_STEP_MAX = 2.0
FROZEN_STEP_SPREAD_RTOL = 1.0e-12

PRIOR_RECEIPTS = (
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/gpu-canary.json"
        ),
        "d5aa099cc4835d427b570a7a22430a7b79498760dc99d8ed280c9bf39692c048",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/gpu-canary-repair-01.json"
        ),
        "b30098f573fb2a7a22f8a1a71b910d2b931fac7c169f049ac9e9efe6af87ab2d",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/tune-0.json"
        ),
        "9e70e8dbd04de09c0bc3946d100d24d67ce520c18f63e58c8b5d3502762fa76f",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/segment-0.json"
        ),
        "d12e7aeb1c9760b9d4bba9f9827c027e371d227a3cf5b84d7775f3a922021892",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/repair-01/tune.json"
        ),
        "c374b2e8197ee39272c020d8bcac6e29d598e28ff8142a708126fe89ada52dde",
    ),
)


def _load_base() -> ModuleType:
    path = ROOT / BASE_HARNESS_PATH
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_hmc_repair02_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load A4 acquisition harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PLAN_PATH = PLAN_PATH
    module.RESULT_PATH = RESULT_PATH
    return module


base = _load_base()
RepairError = base.AcquisitionError


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


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
        _file_row(PLAN_PATH, "prospective_repair_02_evidence_contract"),
        _file_row(HARNESS_PATH, "repair_02_harness"),
        _file_row(TEST_PATH, "focused_repair_02_tests"),
        _file_row(BASE_HARNESS_PATH, "reviewed_acquisition_authority"),
        _file_row(base.A0_LOCK_PATH, "locked_sampler_geometry"),
        _file_row(
            Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
            "locked_a1_target",
        ),
        _file_row(Path("bayesfilter/inference/hmc.py"), "reviewed_hmc_runtime"),
        _file_row(
            Path("bayesfilter/inference/hmc_tuning.py"),
            "reviewed_fixed_mass_dual_averaging_policy",
        ),
        _file_row(
            Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
            "rank_normalized_admission_diagnostics",
        ),
    ]


def _prior_paths() -> tuple[Path, ...]:
    return tuple(path for path, _digest in PRIOR_RECEIPTS)


def validate_prior_receipts() -> float:
    total = 0.0
    for path, expected_digest in PRIOR_RECEIPTS:
        if not (ROOT / path).is_file():
            raise RepairError(f"missing prior GPU receipt: {path}")
        if _sha256(path) != expected_digest:
            raise RepairError(f"prior GPU receipt SHA-256 drift: {path}")
        payload = base._strict_load(path)
        manifest = payload.get("run_manifest", {})
        if (
            manifest.get("trust_basis")
            != "owner_designated_managed_session_visible_gpu_trusted"
            or manifest.get("cpu_gpu_status") != "trusted_gpu_xla"
            or manifest.get("data_version") != base.TARGET_SEMANTIC_SHA256
        ):
            raise RepairError(f"prior receipt is not trusted locked-target evidence: {path}")
        wall_time = float(manifest.get("wall_time_seconds", math.nan))
        if not math.isfinite(wall_time) or wall_time < 0.0:
            raise RepairError(f"invalid prior GPU wall time: {path}")
        total += wall_time
    balanced = base._strict_load(PRIOR_RECEIPTS[3][0])
    if (
        balanced.get("status") != "HARD_VETO"
        or "unmoved_chain"
        not in balanced.get("admission_diagnostics", {}).get("hard_vetoes", ())
    ):
        raise RepairError("balanced-kernel unmoved-chain repair trigger drift")
    smaller = base._strict_load(PRIOR_RECEIPTS[4][0])
    if (
        smaller.get("status") != "NOT_SELECTED"
        or smaller.get("acceptance_passed") is not False
        or smaller.get("chain_moved") != [True, True, True, True]
    ):
        raise RepairError("smaller-step over-acceptance repair trigger drift")
    return total


def _adaptation_private_paths() -> dict[str, Path]:
    return {
        "samples": ARCHIVE_DIR / f"{ADAPTATION_LABEL}_screen_samples.tftensor",
        "final_state": ARCHIVE_DIR / f"{ADAPTATION_LABEL}_final_state.tftensor",
        "step_trace": ARCHIVE_DIR / f"{ADAPTATION_LABEL}_step_trace.tftensor",
        "manifest": ARCHIVE_DIR / f"{ADAPTATION_LABEL}_private_manifest.json",
    }


def _segment_private_paths() -> tuple[Path, ...]:
    return tuple(
        ARCHIVE_DIR / f"{SEGMENT_LABEL}_{suffix}"
        for suffix in (
            "retained_samples.tftensor",
            "final_state.tftensor",
            "final_target_log_prob.tftensor",
            "private_manifest.json",
        )
    )


def _require_fresh(output: Path, private_paths: Sequence[Path]) -> None:
    collisions = [path for path in (output, *private_paths) if (ROOT / path).exists()]
    if collisions:
        raise RepairError(
            "repair-02 artifact collision; refusing overwrite: "
            + ", ".join(path.as_posix() for path in collisions)
        )


def _write_tensor(path: Path, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise RepairError(f"private tensor already exists: {path}")
    tensor = tf.convert_to_tensor(value)
    data = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(data)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def _read_tensor(row: Mapping[str, Any]) -> Any:
    import tensorflow as tf

    path = Path(str(row["path"]))
    data = (ROOT / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != row.get("sha256"):
        raise RepairError(f"private tensor hash mismatch: {path}")
    tensor = tf.io.parse_tensor(data, out_type=tf.dtypes.as_dtype(row["dtype"]))
    tensor = tf.ensure_shape(tensor, tuple(int(dim) for dim in row["shape"]))
    return tensor


def build_adaptation_config() -> Any:
    from bayesfilter.inference.hmc import FullChainHMCConfig
    from bayesfilter.inference.hmc_tuning import HMCTuningPolicy

    policy = HMCTuningPolicy.fixed_mass_dual_averaging(
        num_adaptation_steps=NUM_ADAPTATION_STEPS,
        target_accept_prob=TARGET_ACCEPTANCE,
        source=PLAN_PATH.as_posix(),
    )
    return FullChainHMCConfig(
        num_results=SCREEN_DRAWS,
        num_burnin_steps=NUM_WARMUP_STEPS,
        step_size=INITIAL_STEP_SIZE,
        num_leapfrog_steps=NUM_LEAPFROG_STEPS,
        seed=ADAPTATION_SEED,
        use_xla=True,
        trace_policy="standard",
        tuning_policy=policy,
        target_scope=base.ACQUISITION_SCOPE,
        chain_execution_mode="tf_function",
    )


def classify_adaptation(samples: Any, trace: Mapping[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    sample_tensor = tf.convert_to_tensor(samples, tf.float64)
    accepted = tf.convert_to_tensor(trace["is_accepted"], tf.bool)
    step_trace = tf.convert_to_tensor(trace["step_size"], tf.float64)
    log_accept = tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64)
    target = tf.convert_to_tensor(trace["target_log_prob"], tf.float64)
    hard_vetoes = []
    if tuple(sample_tensor.shape) != (SCREEN_DRAWS, 4, 4):
        hard_vetoes.append("unexpected_adaptation_sample_shape")
    if tuple(accepted.shape) != (SCREEN_DRAWS, 4):
        hard_vetoes.append("unexpected_adaptation_acceptance_shape")
    if tuple(step_trace.shape) != (SCREEN_DRAWS,):
        hard_vetoes.append("adapted_step_trace_not_scalar_per_draw")
    all_finite = bool(tf.reduce_all(tf.math.is_finite(sample_tensor)).numpy())
    telemetry_finite = bool(
        tf.reduce_all(tf.math.is_finite(log_accept)).numpy()
        and tf.reduce_all(tf.math.is_finite(target)).numpy()
    )
    if not all_finite:
        hard_vetoes.append("nonfinite_adaptation_samples")
    if not telemetry_finite:
        hard_vetoes.append("nonfinite_adaptation_telemetry")
    moved = tf.reduce_any(
        tf.not_equal(sample_tensor[1:], sample_tensor[:-1]), axis=(0, 2)
    )
    moved_list = [bool(value) for value in moved.numpy().tolist()]
    if moved_list != [True, True, True, True]:
        hard_vetoes.append("unmoved_adaptation_screen_chain")
    acceptance = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
    acceptance_by_chain = [float(value) for value in acceptance.numpy().tolist()]
    aggregate_acceptance = float(tf.reduce_mean(acceptance).numpy())
    if tuple(step_trace.shape) == (SCREEN_DRAWS,):
        step_values = step_trace.numpy()
        final_step = float(step_values[-1])
        step_spread = float(step_values.max() - step_values.min())
        step_tolerance = FROZEN_STEP_SPREAD_RTOL * max(1.0, abs(final_step))
        if not math.isfinite(final_step) or not FROZEN_STEP_MIN <= final_step <= FROZEN_STEP_MAX:
            hard_vetoes.append("unsafe_or_nonfinite_frozen_step")
        if not math.isfinite(step_spread) or step_spread > step_tolerance:
            hard_vetoes.append("step_changed_after_warmup")
    else:
        final_step = None
        step_spread = None
        step_tolerance = None
    divergence_status = "available" if "divergence" in trace else "not_exposed_by_kernel"
    divergence_count = None
    if "divergence" in trace:
        divergence_count = int(
            tf.reduce_sum(tf.cast(trace["divergence"], tf.int32)).numpy()
        )
        if divergence_count > 0:
            hard_vetoes.append("positive_native_divergence_count")
    acceptance_passed = (
        REPAIR_ACCEPTANCE_MIN <= aggregate_acceptance <= REPAIR_ACCEPTANCE_MAX
        and all(
            PER_CHAIN_SAFETY_MIN <= value <= PER_CHAIN_SAFETY_MAX
            for value in acceptance_by_chain
        )
    )
    selected = not hard_vetoes and acceptance_passed
    return {
        "selected": selected,
        "status": "SELECTED" if selected else ("HARD_VETO" if hard_vetoes else "NOT_SELECTED"),
        "hard_vetoes": hard_vetoes,
        "acceptance_passed": acceptance_passed,
        "acceptance_rate": aggregate_acceptance,
        "acceptance_rate_by_chain": acceptance_by_chain,
        "chain_moved": moved_list,
        "all_samples_finite": all_finite,
        "telemetry_finite": telemetry_finite,
        "final_step_size": final_step,
        "post_warmup_step_spread": step_spread,
        "post_warmup_step_spread_tolerance": step_tolerance,
        "native_divergence_status": divergence_status,
        "native_divergence_count": divergence_count,
    }


def _environment_manifest(
    *, started: str, completed: str, wall_time: float, output: Path, seeds: Any
) -> dict[str, Any]:
    return base._environment_manifest(
        started=started,
        completed=completed,
        wall_time=wall_time,
        output_paths=(output, ARCHIVE_DIR),
        random_seeds=(base.ROOT_SEED, seeds),
    )


def run_adaptation() -> dict[str, Any]:
    import tensorflow as tf
    from bayesfilter.inference.hmc import build_reusable_full_chain_tfp_hmc_runner

    base._require_gpu()
    private_paths = _adaptation_private_paths()
    _require_fresh(ADAPTATION_OUTPUT, tuple(private_paths.values()))
    prior_seconds = validate_prior_receipts()
    projected_seconds = 7200.0
    if prior_seconds + projected_seconds > base.GPU_BUDGET_SECONDS:
        raise RepairError("adaptation projection exceeds remaining shared GPU budget")
    adapter = base.A4CalibrationHMCAdapter()
    initial = tf.constant(base.INITIAL_STATES, tf.float64)
    config = build_adaptation_config()
    started_at = base._now()
    started = time.perf_counter()
    runner = build_reusable_full_chain_tfp_hmc_runner(adapter, initial, config)
    result = runner.run(
        current_state=initial,
        seed=ADAPTATION_SEED,
        step_size=INITIAL_STEP_SIZE,
    )
    completed_at = base._now()
    wall_time = time.perf_counter() - started
    classification = classify_adaptation(result.samples, result.trace)
    output_devices = sorted(
        {
            str(tf.convert_to_tensor(result.samples).device),
            str(tf.convert_to_tensor(result.trace["target_log_prob"]).device),
        }
    )
    if not output_devices or not all("GPU:" in item for item in output_devices):
        classification["hard_vetoes"].append("adaptation_outputs_not_on_gpu")
        classification["selected"] = False
        classification["status"] = "HARD_VETO"

    tensor_rows = {
        "screen_samples": _write_tensor(private_paths["samples"], result.samples),
        "final_state": _write_tensor(private_paths["final_state"], result.samples[-1]),
        "step_trace": _write_tensor(private_paths["step_trace"], result.trace["step_size"]),
    }
    private_manifest = {
        "artifact_type": "bayesfilter_private_ssl_lstm_a4_hmc_repair_02_adaptation",
        "schema_version": 1,
        "target_scope": base.TARGET_SCOPE,
        "target_semantic_sha256": base.TARGET_SEMANTIC_SHA256,
        "initial_state_policy": "original_fixed_four_dispersed_starts",
        "config": {
            "initial_step_size": INITIAL_STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "num_adaptation_steps": NUM_ADAPTATION_STEPS,
            "num_warmup_steps": NUM_WARMUP_STEPS,
            "screen_draws": SCREEN_DRAWS,
            "target_acceptance": TARGET_ACCEPTANCE,
            "seed": ADAPTATION_SEED,
        },
        "tensors": tensor_rows,
        "classification": classification,
        "privacy_contract": {
            "screen_samples_are_diagnostic_only": True,
            "screen_samples_must_not_enter_calibration": True,
            "final_state_may_seed_only_the_frozen_repair_02_retained_run": True,
        },
    }
    base._write_json(private_paths["manifest"], private_manifest)
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_repair_02_adaptation.v1",
        "status": classification["status"],
        "classification": classification,
        "adaptation_contract": {
            "policy": "fixed_mass_dual_averaging",
            "initial_step_size": INITIAL_STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "num_adaptation_steps": NUM_ADAPTATION_STEPS,
            "num_warmup_steps": NUM_WARMUP_STEPS,
            "screen_draws": SCREEN_DRAWS,
            "target_acceptance": TARGET_ACCEPTANCE,
            "repair_acceptance_band": [REPAIR_ACCEPTANCE_MIN, REPAIR_ACCEPTANCE_MAX],
            "per_chain_safety_interval": [
                PER_CHAIN_SAFETY_MIN,
                PER_CHAIN_SAFETY_MAX,
            ],
            "frozen_step_safety_interval": [FROZEN_STEP_MIN, FROZEN_STEP_MAX],
            "seed": ADAPTATION_SEED,
        },
        "initial_state_policy": "original_fixed_four_dispersed_starts",
        "runner_diagnostics": base._json_safe(result.diagnostics),
        "runner_metadata": {
            **base._json_safe(result.metadata),
            "evidence_output_devices": output_devices,
        },
        "private_manifest_sha256": _sha256(private_paths["manifest"]),
        "budget_lineage_artifacts": [path.as_posix() for path in _prior_paths()],
        "source_files": _source_bindings(),
        "run_manifest": _environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output=ADAPTATION_OUTPUT,
            seeds=ADAPTATION_SEED,
        ),
        "gpu_budget": {
            "cap_seconds": base.GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_seconds,
            "projected_seconds_before_run": projected_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": float(result.metadata["sample_chain_call_s"]),
            "remaining_seconds": base.GPU_BUDGET_SECONDS - prior_seconds - wall_time,
        },
        "nonclaims": [
            "dual-averaging adaptation screen only",
            "screen samples are not retained calibration draws",
            "no convergence, posterior correctness, or sampler superiority claim",
        ],
    }
    base._write_json(ADAPTATION_OUTPUT, payload)
    if classification["hard_vetoes"]:
        raise RepairError(
            f"repair-02 adaptation hard vetoes: {classification['hard_vetoes']}"
        )
    return payload


def run_segment() -> dict[str, Any]:
    import tensorflow as tf

    base._require_gpu()
    _require_fresh(SEGMENT_OUTPUT, _segment_private_paths())
    adaptation, current_state, frozen_step = _load_selected_adaptation()
    prior_seconds = (
        validate_prior_receipts()
        + float(adaptation["run_manifest"]["wall_time_seconds"])
    )
    projected_seconds = 7200.0
    if prior_seconds + projected_seconds > base.GPU_BUDGET_SECONDS:
        raise RepairError("retained projection exceeds remaining shared GPU budget")
    adapter = base.A4CalibrationHMCAdapter()
    started_at = base._now()
    started = time.perf_counter()
    (samples, _state, manifest, call_s), diagnostics, metadata, _ = base._run_archive(
        adapter=adapter,
        archive_dir=ARCHIVE_DIR,
        label=SEGMENT_LABEL,
        current_state=current_state,
        num_results=250,
        num_burnin_steps=250,
        step_size=frozen_step,
        leapfrog_steps=NUM_LEAPFROG_STEPS,
        seed=SEGMENT_SEED,
        role="trusted_gpu_xla_a4_hmc_repair_02_frozen_step_retained_segment",
    )
    admission = base._admission_diagnostics(
        latent_draw_major=samples,
        adapter=adapter,
        segment_manifests=(manifest,),
    )
    completed_at = base._now()
    wall_time = time.perf_counter() - started
    status = "ADMITTED" if admission["admitted"] else (
        "HARD_VETO" if admission["hard_vetoes"] else "NOT_ADMITTED"
    )
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_repair_02_segment.v1",
        "status": status,
        "segment": {
            "index": 0,
            "draw_count": 250,
            "burnin_count": 250,
            "seed": SEGMENT_SEED,
            "label": SEGMENT_LABEL,
        },
        "initial_state_policy": "exact_repair_02_adaptation_screen_final_state",
        "frozen_kernel": {
            "step_size": frozen_step,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "trajectory_length": frozen_step * NUM_LEAPFROG_STEPS,
            "source": ADAPTATION_OUTPUT.as_posix(),
        },
        "admission_diagnostics": admission,
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "private_manifest_sha256": _sha256(
            ARCHIVE_DIR / f"{SEGMENT_LABEL}_private_manifest.json"
        ),
        "cumulative_private_sample_sha256": hashlib.sha256(
            bytes(tf.io.serialize_tensor(samples).numpy())
        ).hexdigest(),
        "budget_lineage_artifacts": [
            *(path.as_posix() for path in _prior_paths()),
            ADAPTATION_OUTPUT.as_posix(),
        ],
        "source_files": _source_bindings(),
        "run_manifest": _environment_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output=SEGMENT_OUTPUT,
            seeds=SEGMENT_SEED,
        ),
        "gpu_budget": {
            "cap_seconds": base.GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_seconds,
            "projected_seconds_before_run": projected_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds": base.GPU_BUDGET_SECONDS - prior_seconds - wall_time,
        },
        "nonclaims": admission["nonclaims"],
    }
    base._write_json(SEGMENT_OUTPUT, payload)
    if admission["hard_vetoes"]:
        raise RepairError(f"repair-02 retained hard vetoes: {admission['hard_vetoes']}")
    return payload


def _load_selected_adaptation() -> tuple[dict[str, Any], Any, float]:
    payload = base._strict_load(ADAPTATION_OUTPUT)
    if (
        payload.get("schema_version")
        != "bayesfilter.ssl_lstm.a4_hmc_repair_02_adaptation.v1"
        or payload.get("status") != "SELECTED"
        or payload.get("classification", {}).get("selected") is not True
    ):
        raise RepairError("retained run requires selected repair-02 adaptation")
    if payload.get("source_files") != _source_bindings():
        raise RepairError("repair-02 source binding drift since adaptation")
    if payload.get("budget_lineage_artifacts") != [
        path.as_posix() for path in _prior_paths()
    ]:
        raise RepairError("repair-02 adaptation ancestry drift")
    run_manifest = payload.get("run_manifest", {})
    if (
        run_manifest.get("trust_basis")
        != "owner_designated_managed_session_visible_gpu_trusted"
        or run_manifest.get("cpu_gpu_status") != "trusted_gpu_xla"
        or run_manifest.get("data_version") != base.TARGET_SEMANTIC_SHA256
        or run_manifest.get("plan_path") != PLAN_PATH.as_posix()
        or run_manifest.get("result_path") != RESULT_PATH.as_posix()
    ):
        raise RepairError("repair-02 adaptation trusted manifest drift")
    private_path = _adaptation_private_paths()["manifest"]
    if payload.get("private_manifest_sha256") != _sha256(private_path):
        raise RepairError("repair-02 adaptation private-manifest hash drift")
    private = base._strict_load(private_path)
    if private.get("target_semantic_sha256") != base.TARGET_SEMANTIC_SHA256:
        raise RepairError("repair-02 private target identity drift")
    if private.get("initial_state_policy") != "original_fixed_four_dispersed_starts":
        raise RepairError("repair-02 private initial-state policy drift")
    expected_private_config = {
        "initial_step_size": INITIAL_STEP_SIZE,
        "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
        "num_adaptation_steps": NUM_ADAPTATION_STEPS,
        "num_warmup_steps": NUM_WARMUP_STEPS,
        "screen_draws": SCREEN_DRAWS,
        "target_acceptance": TARGET_ACCEPTANCE,
        "seed": list(ADAPTATION_SEED),
    }
    if private.get("config") != expected_private_config:
        raise RepairError("repair-02 private adaptation config drift")
    if private.get("classification") != payload.get("classification"):
        raise RepairError("repair-02 private/public classification drift")
    final_state = _read_tensor(private["tensors"]["final_state"])
    step_trace = _read_tensor(private["tensors"]["step_trace"])
    screen_samples = _read_tensor(private["tensors"]["screen_samples"])
    if tuple(final_state.shape) != (4, 4) or tuple(screen_samples.shape) != (64, 4, 4):
        raise RepairError("repair-02 private state/sample shape drift")
    if not bool(tf_reduce_all_finite(final_state, screen_samples)):
        raise RepairError("repair-02 private state/sample nonfinite")
    import tensorflow as tf

    try:
        tf.debugging.assert_equal(final_state, screen_samples[-1])
    except tf.errors.InvalidArgumentError as exc:
        raise RepairError(
            "repair-02 final state does not equal final adaptation-screen draw"
        ) from exc
    if tuple(step_trace.shape) != (64,):
        raise RepairError("repair-02 private step trace shape drift")
    frozen_step = float(step_trace[-1].numpy())
    spread = float((step_trace.numpy().max() - step_trace.numpy().min()))
    if (
        frozen_step != float(payload["classification"]["final_step_size"])
        or spread > FROZEN_STEP_SPREAD_RTOL * max(1.0, abs(frozen_step))
    ):
        raise RepairError("repair-02 frozen step drift")
    return payload, final_state, frozen_step


def tf_reduce_all_finite(*values: Any) -> bool:
    import tensorflow as tf

    return all(
        bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())
        for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-lineage", "adapt", "segment"))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "validate-lineage":
        prior = validate_prior_receipts()
        payload = {
            "status": "PASSED",
            "prior_gpu_seconds": prior,
            "remaining_gpu_seconds": base.GPU_BUDGET_SECONDS - prior,
        }
        output = None
    elif args.command == "adapt":
        payload = run_adaptation()
        output = ADAPTATION_OUTPUT.as_posix()
    else:
        payload = run_segment()
        output = SEGMENT_OUTPUT.as_posix()
    print(
        base._canonical_bytes(
            {"command": args.command, "status": payload["status"], "output": output}
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as exc:
        print(f"A4_HMC_REPAIR_02_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
