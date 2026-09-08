#!/usr/bin/env python3
"""Diagnose the R-hat trajectory of one rejected SSL-LSTM q=20 NeuTra kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-rhat-trajectory-diagnostic-plan-2026-08-21.md"
)
CONTINUATION_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_global_mixing_continuation_2026_08_20.py"
)
TRAINING_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py"
)
R2_ROOT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc"
)
R2_RESULT = R2_ROOT / "result.json"
R2_MANIFEST = R2_ROOT / "manifest.json"
R2_HASHES = R2_ROOT / "artifact-hashes.json"
R2_TUNING = R2_ROOT / (
    "candidate-01-seed-2-L-05/tuning/L-05/tuning-result.json"
)
FIXED_HMC_MECHANICS = (
    ROOT / "bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py"
)
HMC_CONVERGENCE = ROOT / "bayesfilter/inference/hmc_convergence.py"
ROUTE_LEDGER = ROOT / (
    "docs/plans/artifacts/"
    "neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/"
    "route_ledger.json"
)
FAILED_LAUNCH_ROOT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/"
    "r3-rhat-trajectory"
)
FAILED_LAUNCH_RESULT = FAILED_LAUNCH_ROOT / "result.json"
FAILED_LAUNCH_MANIFEST = FAILED_LAUNCH_ROOT / "manifest.json"
FAILED_LAUNCH_HASHES = FAILED_LAUNCH_ROOT / "artifact-hashes.json"
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/"
    "r3-rhat-trajectory-retry-01"
)

CONTINUATION_RUNNER_SHA256 = (
    "eeef1880cb26a7649ccf76230b909518fa1ca4a3e94e3bbc35e38de654d57723"
)
R2_RESULT_SHA256 = "9092b82d25f8e63d1708c63c7d48284ef3c55a5edc3e225833ba505f0b65e706"
R2_MANIFEST_SHA256 = "c662e56370fe6dd111916232dfede31ac169ce214d03a95431583fe9ca6a92d6"
R2_HASHES_SHA256 = "d874cf937fe6b7cca69be6c3aa0274ad3ca3ba80a0d409a5a943370425f5a14e"
R2_TUNING_SHA256 = "b9006654df5fe52a44c1057dc52646ab403de427afdc8b0bfe56e780575987fe"
FIXED_HMC_MECHANICS_SHA256 = (
    "a04aaea2824bb972881ce9af023d89ba473a6343cf1d437bca2812a90f564616"
)
HMC_CONVERGENCE_SHA256 = (
    "b7544346f4beb63946c7482a8a9a2341f4d5cabe72a2b29fab7c4f4ab8408dd7"
)
FAILED_LAUNCH_RESULT_SHA256 = (
    "f3a0112caf08738c8fcc83af0b28b4ac41504295edb076ace3018cf89e7befa8"
)
FAILED_LAUNCH_MANIFEST_SHA256 = (
    "198339c610841f43341e31be5085fd4dd52ed8e03648adc446ab2b063294019c"
)
FAILED_LAUNCH_HASHES_SHA256 = (
    "e607f09c9d9f61b4b0b236cd049ca6b98220feef2403e154b5afcd15b5b98f0e"
)

AGGREGATE_GRANT_SECONDS = 64800.0
PRIOR_R2_GPU_WALL_SECONDS = 23280.603976539
FAILED_LAUNCH_PROCESS_WALL_SECONDS = 0.050816870003473014
EXTERNAL_PROCESS_CAP_SECONDS = 36000
INTERNAL_WORK_CAP_SECONDS = 35820.0
CLOSEOUT_RESERVE_SECONDS = 180.0
PRIOR_L5_CALL_SECONDS = 13188.602818158004
PRIOR_L5_CALL_STATES = 2064
FORECAST_ALLOWANCE = 1.25

TRANSPORT_SEED = 2
NUM_LEAPFROG_STEPS = 5
STEP_SIZE = 0.2460072308515237
NUM_RESULTS = 4000
NUM_BURNIN_STEPS = 64
CHAIN_COUNT = 4
PARAMETER_COUNT = 4
HMC_SEED = (20260820, 52000)
CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
RECENT_WINDOW = 1000
BASELINE_CHECKPOINT = 2000
RHAT_THRESHOLD = 1.01
OBSERVATION_WEIGHT_INDEX = 2
PARAMETER_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
TARGET_SCOPE = "ssl_lstm_q20_neutra_global_mixing_continuation:seed-2:L-5"
EXPECTED_POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"
SHARED_HMC_RUNTIME = "tfp.mcmc.sample_chain"


class DiagnosticBudgetExhausted(RuntimeError):
    """Raised when the one authorized call cannot fit its declared budget."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic artifact: {path}")
    temporary.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic tensor: {path}")
    temporary.write_bytes(value)
    temporary.replace(path)


def _reserve_output_root(path: Path) -> Path:
    absolute = path if path.is_absolute() else ROOT / path
    if absolute.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("diagnostic output root is frozen by the reviewed plan")
    if absolute.exists():
        raise RuntimeError(f"refusing to reuse diagnostic output root: {absolute}")
    absolute.mkdir(parents=True, exist_ok=False)
    return absolute


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty)}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="1")
    parser.add_argument(
        "--time-cap-seconds", type=float, default=INTERNAL_WORK_CAP_SECONDS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).isdigit():
        raise SystemExit("device must be one nonnegative physical GPU index")
    if float(args.time_cap_seconds) != INTERNAL_WORK_CAP_SECONDS:
        raise SystemExit("internal diagnostic cap is frozen to 35820 seconds")
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise SystemExit("output root is frozen to r3-rhat-trajectory-retry-01")


def _forecast() -> Mapping[str, Any]:
    scaled = PRIOR_L5_CALL_SECONDS * (
        (NUM_RESULTS + NUM_BURNIN_STEPS) / PRIOR_L5_CALL_STATES
    )
    allowed = scaled * FORECAST_ALLOWANCE
    prior_campaign_process_wall = (
        PRIOR_R2_GPU_WALL_SECONDS + FAILED_LAUNCH_PROCESS_WALL_SECONDS
    )
    aggregate_at_external_cap = (
        prior_campaign_process_wall + EXTERNAL_PROCESS_CAP_SECONDS
    )
    return {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_resource_forecast.v1",
        "role": "engineering_resource_admission_only",
        "prior_r2_gpu_wall_seconds": PRIOR_R2_GPU_WALL_SECONDS,
        "failed_pre_tensorflow_launcher_wall_seconds": (
            FAILED_LAUNCH_PROCESS_WALL_SECONDS
        ),
        "prior_campaign_process_wall_seconds": prior_campaign_process_wall,
        "aggregate_grant_seconds": AGGREGATE_GRANT_SECONDS,
        "remaining_aggregate_grant_seconds": (
            AGGREGATE_GRANT_SECONDS - prior_campaign_process_wall
        ),
        "external_process_cap_seconds": EXTERNAL_PROCESS_CAP_SECONDS,
        "internal_work_cap_seconds": INTERNAL_WORK_CAP_SECONDS,
        "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
        "prior_l5_call_seconds": PRIOR_L5_CALL_SECONDS,
        "prior_l5_call_states_per_chain": PRIOR_L5_CALL_STATES,
        "requested_states_per_chain": NUM_RESULTS + NUM_BURNIN_STEPS,
        "linear_scaled_seconds": scaled,
        "forecast_allowance": FORECAST_ALLOWANCE,
        "allowance_scaled_seconds": allowed,
        "aggregate_wall_at_external_cap_seconds": aggregate_at_external_cap,
        "fits_internal_with_closeout": bool(
            allowed + CLOSEOUT_RESERVE_SECONDS <= INTERNAL_WORK_CAP_SECONDS
        ),
        "fits_prior_hmc_envelope": bool(aggregate_at_external_cap <= 61200.0),
        "fits_aggregate_grant": bool(
            aggregate_at_external_cap <= AGGREGATE_GRANT_SECONDS
        ),
        "nonclaim": "runtime proportionality is a feasibility estimate, not a sampler diagnostic",
    }


def _validated_inputs() -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    fixed = (
        (CONTINUATION_RUNNER, CONTINUATION_RUNNER_SHA256),
        (R2_RESULT, R2_RESULT_SHA256),
        (R2_MANIFEST, R2_MANIFEST_SHA256),
        (R2_HASHES, R2_HASHES_SHA256),
        (R2_TUNING, R2_TUNING_SHA256),
        (FIXED_HMC_MECHANICS, FIXED_HMC_MECHANICS_SHA256),
        (HMC_CONVERGENCE, HMC_CONVERGENCE_SHA256),
        (FAILED_LAUNCH_RESULT, FAILED_LAUNCH_RESULT_SHA256),
        (FAILED_LAUNCH_MANIFEST, FAILED_LAUNCH_MANIFEST_SHA256),
        (FAILED_LAUNCH_HASHES, FAILED_LAUNCH_HASHES_SHA256),
    )
    for path, expected in fixed:
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"immutable r2 input SHA-256 mismatch: {path}")

    continuation = _load_module(
        CONTINUATION_RUNNER, "ssl_lstm_q20_rhat_frozen_continuation"
    )
    training_result, prior_budget = continuation._validated_training_artifacts()

    failed_hashes = _read_json(FAILED_LAUNCH_HASHES)
    if failed_hashes.get("schema") != "bayesfilter.ssl_lstm.q20_neutra_rhat_hashes.v1":
        raise RuntimeError("failed-launch inventory schema mismatch")
    failed_inventory = failed_hashes.get("artifacts")
    if not isinstance(failed_inventory, Mapping) or len(failed_inventory) != 2:
        raise RuntimeError("failed-launch inventory must contain exactly two artifacts")
    for relative, expected in failed_inventory.items():
        path = FAILED_LAUNCH_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"failed-launch artifact graph mismatch: {relative}")
    failed_result = _read_json(FAILED_LAUNCH_RESULT)
    if (
        failed_result.get("status") != "DIAGNOSTIC_HARNESS_FAILURE"
        or failed_result.get("error_type") != "ModuleNotFoundError"
        or failed_result.get("reason") != "No module named 'bayesfilter'"
        or float(failed_result.get("process_wall_seconds", -1.0))
        != FAILED_LAUNCH_PROCESS_WALL_SECONDS
    ):
        raise RuntimeError("failed-launch repair provenance mismatch")
    if failed_result.get("partial_artifacts") != {}:
        raise RuntimeError("failed launch unexpectedly reached a diagnostic phase")

    inventory_payload = _read_json(R2_HASHES)
    if (
        inventory_payload.get("schema")
        != "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc_hashes.v1"
    ):
        raise RuntimeError("r2 HMC inventory schema mismatch")
    inventory = inventory_payload.get("artifacts")
    if not isinstance(inventory, Mapping) or len(inventory) != 15:
        raise RuntimeError("r2 HMC inventory must contain exactly 15 artifacts")
    for relative, expected in inventory.items():
        path = R2_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"r2 HMC artifact graph mismatch: {relative}")

    result = _read_json(R2_RESULT)
    manifest = _read_json(R2_MANIFEST)
    tuning = _read_json(R2_TUNING)
    if result.get("status") != "HMC_NO_CANDIDATE_ADMITTED":
        raise RuntimeError("r2 baseline terminal status mismatch")
    if result.get("decision") != "NO_HMC_CANDIDATE_ADMITTED":
        raise RuntimeError("r2 baseline decision mismatch")
    if float(result.get("hmc_phase_wall_seconds", -1.0)) != PRIOR_R2_GPU_WALL_SECONDS:
        raise RuntimeError("r2 GPU wall mismatch")
    if manifest.get("runner", {}).get("sha256") != CONTINUATION_RUNNER_SHA256:
        raise RuntimeError("r2 manifest continuation-runner identity mismatch")

    candidates = tuning.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("seed-2 L=5 tuning result must contain one candidate")
    candidate = candidates[0]
    diagnostics = candidate.get("verification_diagnostics")
    config = candidate.get("verification_config_payload")
    if not isinstance(diagnostics, Mapping) or not isinstance(config, Mapping):
        raise RuntimeError("r2 candidate verification payload is incomplete")
    expected_config = {
        "num_results": 2000,
        "num_burnin_steps": NUM_BURNIN_STEPS,
        "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
        "step_size": STEP_SIZE,
        "seed": list(HMC_SEED),
        "target_scope": TARGET_SCOPE,
        "use_xla": True,
        "chain_execution_mode": "tf_function",
        "target_status_trace_policy": "per_chain_step",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"r2 candidate config drift: {key}")
    if tuning.get("final_status") != "no_viable_candidate":
        raise RuntimeError("r2 seed-2 L=5 candidate was not rejected")
    if tuning.get("hard_vetoes") != ["verification_modern_rank_folded_rhat_failed"]:
        raise RuntimeError("r2 seed-2 L=5 rejection identity mismatch")
    if float(candidate.get("selected_step_size", -1.0)) != STEP_SIZE:
        raise RuntimeError("r2 seed-2 L=5 step-size identity mismatch")
    tuning_config = tuning.get("config")
    if not isinstance(tuning_config, Mapping):
        raise RuntimeError("r2 tuning configuration missing")
    if tuning_config.get("proposal_dynamics_identity") != "exact_transformed_gradient":
        raise RuntimeError("r2 proposal-dynamics identity mismatch")
    mass = tuning.get("identity_z_mass_artifact_payload")
    if not isinstance(mass, Mapping):
        raise RuntimeError("r2 identity-z mass artifact missing")
    if (
        mass.get("covariance_source") != "fixed_identity_z"
        or mass.get("matrix_used_for_square_root") != "identity_z"
        or mass.get("dimension") != PARAMETER_COUNT
    ):
        raise RuntimeError("r2 identity-z mass policy mismatch")
    identity = [
        [1.0 if row == column else 0.0 for column in range(PARAMETER_COUNT)]
        for row in range(PARAMETER_COUNT)
    ]
    if mass.get("covariance") != identity or mass.get("factor") != identity:
        raise RuntimeError("r2 identity-z mass matrices mismatch")
    modern = diagnostics.get("modern_rank_normalized_verification")
    if not isinstance(modern, Mapping):
        raise RuntimeError("r2 seed-2 L=5 modern R-hat payload missing")
    if float(modern.get("max_finite_rhat", -1.0)) != 1.0875996310350042:
        raise RuntimeError("r2 seed-2 L=5 baseline R-hat drift")

    baseline = {
        "r2_result": result,
        "r2_manifest": manifest,
        "tuning": tuning,
        "candidate": candidate,
        "verification_diagnostics": diagnostics,
        "modern_rhat": modern,
        "initial_state_bank": diagnostics.get("initial_state_bank"),
        "kernel": {
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "transformed_adapter_signature": tuning.get(
                "transformed_adapter_signature"
            ),
            "base_adapter_signature": tuning.get("base_adapter_signature"),
            "fixed_transport_manifest_hash": tuning.get(
                "fixed_transport_manifest_hash"
            ),
            "mass_policy": "fixed_identity_z",
            "identity_z_mass_artifact_payload": mass,
            "identity_z_mass_artifact_signature": tuning.get(
                "identity_z_mass_artifact_signature"
            ),
            "proposal_dynamics_identity": "exact_transformed_gradient",
            "use_xla": True,
        },
        "verification_config": config,
    }
    budget = {
        **_forecast(),
        "authorization": "user_continue_as_suggested_2026-08-21",
        "prior_r2_budget": prior_budget,
        "launcher_attempt_limit": 2,
        "gpu_initializing_attempt_limit": 1,
        "retry_index": 1,
        "failed_pre_tensorflow_launch": {
            "root": FAILED_LAUNCH_ROOT.as_posix(),
            "result_sha256": FAILED_LAUNCH_RESULT_SHA256,
            "manifest_sha256": FAILED_LAUNCH_MANIFEST_SHA256,
            "artifact_hashes_sha256": FAILED_LAUNCH_HASHES_SHA256,
            "process_wall_seconds": FAILED_LAUNCH_PROCESS_WALL_SECONDS,
            "gpu_or_tensorflow_initialized": False,
        },
        "predictive_reserve_used": False,
    }
    return continuation, training_result, baseline, budget


def _route_policy_audit() -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc_policy import (
        load_neutra_hmc_route_ledger,
        require_neutra_hmc_route_policy,
    )

    ledger = load_neutra_hmc_route_ledger(ROUTE_LEDGER)
    audit = require_neutra_hmc_route_policy(ROOT, ledger)
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    records = {
        str(item.get("path")): item
        for item in ledger.get("routes", ())
        if isinstance(item, Mapping)
    }
    record = records.get(relative)
    if (
        not isinstance(record, Mapping)
        or record.get("classification") != "smoke_mechanics_or_reference"
    ):
        raise RuntimeError("R-hat diagnostic lacks non-promotional route classification")
    if audit.get("canonical_policy_id") != EXPECTED_POLICY_ID:
        raise RuntimeError("canonical route-policy identity drift")
    return {
        "ledger_path": ROUTE_LEDGER.as_posix(),
        "ledger_sha256": _sha256(ROUTE_LEDGER),
        "canonical_policy_id": audit["canonical_policy_id"],
        "passed": audit["passed"],
        "discovered_route_count": len(audit["discovered_routes"]),
        "classified_route_count": len(audit["classified_routes"]),
        "current_route": record,
        "claim_bearing_route_unchanged": True,
    }


def _acceptance_summary(tf: Any, trace: Mapping[str, Any]) -> Mapping[str, Any]:
    log_accept = tf.cast(tf.convert_to_tensor(trace["log_accept_ratio"]), tf.float64)
    accepted = tf.cast(tf.convert_to_tensor(trace["is_accepted"]), tf.float64)
    probability = tf.exp(tf.minimum(log_accept, 0.0))
    return {
        "acceptance_rate": float(tf.reduce_mean(probability).numpy()),
        "acceptance_probability_by_chain": tf.reduce_mean(
            probability, axis=0
        ).numpy().tolist(),
        "binary_acceptance_rate": float(tf.reduce_mean(accepted).numpy()),
        "binary_acceptance_by_chain": tf.reduce_mean(
            accepted, axis=0
        ).numpy().tolist(),
    }


def _mode_summary(tf: Any, labels_draw_chain: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_global_mixing import assess_retained_mode_mixing

    labels = tf.cast(tf.convert_to_tensor(labels_draw_chain), tf.int32)
    report = assess_retained_mode_mixing(
        tf.transpose(labels, (1, 0)), region_count=2
    )
    return {
        **_json_ready(report.payload()),
        "labels_shape": [int(value) for value in labels.shape],
        "label_semantics": "1 iff observation_weight.0.0 < 0; 0 otherwise",
    }


def _diagnose_block(tf: Any, physical: Any, trace: Mapping[str, Any]) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_convergence import (
        rank_normalized_split_rhat_summary,
    )

    values = tf.cast(tf.convert_to_tensor(physical), tf.float64)
    labels = tf.cast(values[:, :, OBSERVATION_WEIGHT_INDEX] < 0.0, tf.int32)
    physical_rhat = rank_normalized_split_rhat_summary(
        values, rhat_max=RHAT_THRESHOLD
    )
    sign_rhat = rank_normalized_split_rhat_summary(
        tf.cast(labels[:, :, tf.newaxis], tf.float64),
        rhat_max=RHAT_THRESHOLD,
    )
    mode = _mode_summary(tf, labels)
    return {
        "draw_count_per_chain": int(values.shape[0]),
        "physical_rhat": physical_rhat,
        "observation_weight_rhat": {
            "rank": physical_rhat["rank_normalized_split_rhat"][
                OBSERVATION_WEIGHT_INDEX
            ],
            "folded": physical_rhat["folded_rank_normalized_split_rhat"][
                OBSERVATION_WEIGHT_INDEX
            ],
            "maximum": physical_rhat["rhat"][OBSERVATION_WEIGHT_INDEX],
        },
        "sign_indicator_rhat": sign_rhat,
        "sign_indicator_maximum_rhat": sign_rhat["rhat"][0],
        "mode_mixing": mode,
        "acceptance": _acceptance_summary(tf, trace),
        "screen_passed": bool(
            physical_rhat["passed"]
            and sign_rhat["passed"]
            and mode["passed"] is True
        ),
    }


def _slice_trace(trace: Mapping[str, Any], start: int, stop: int) -> Mapping[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in trace.items():
        if isinstance(value, Mapping):
            sliced[str(key)] = _slice_trace(value, start, stop)
        else:
            sliced[str(key)] = value[start:stop]
    return sliced


def _trajectory_summary(
    rows: Sequence[Mapping[str, Any]], *, baseline_checkpoint: int
) -> Mapping[str, Any]:
    cumulative = {
        int(row["checkpoint_draws"]): float(
            row["cumulative"]["observation_weight_rhat"]["maximum"]
        )
        for row in rows
    }
    if baseline_checkpoint not in cumulative:
        raise ValueError("baseline checkpoint missing from trajectory")
    ordered = [cumulative[int(row["checkpoint_draws"])] for row in rows]
    changes = [right - left for left, right in zip(ordered, ordered[1:])]
    endpoint_draws = int(rows[-1]["checkpoint_draws"])
    endpoint = cumulative[endpoint_draws]
    baseline = cumulative[baseline_checkpoint]
    recent_rows = [
        row for row in rows if isinstance(row.get("recent_window"), Mapping)
    ]
    recent_observation_weight = [
        {
            "checkpoint_draws": int(row["checkpoint_draws"]),
            "maximum_rhat": float(
                row["recent_window"]["observation_weight_rhat"]["maximum"]
            ),
        }
        for row in recent_rows
    ]
    return {
        "baseline_checkpoint_draws": baseline_checkpoint,
        "endpoint_checkpoint_draws": endpoint_draws,
        "observation_weight_rhat_at_baseline": baseline,
        "observation_weight_rhat_at_endpoint": endpoint,
        "observation_weight_rhat_change_baseline_to_endpoint": endpoint - baseline,
        "observation_weight_rhat_dropped_baseline_to_endpoint": endpoint < baseline,
        "adjacent_checkpoint_changes": changes,
        "adjacent_decrease_count": sum(change < 0.0 for change in changes),
        "adjacent_increase_count": sum(change > 0.0 for change in changes),
        "all_adjacent_changes_nonpositive": all(change <= 0.0 for change in changes),
        "recent_window_observation_weight_rhat": recent_observation_weight,
        "role": "descriptive_trajectory_without_extrapolation",
    }


def _checkpoint_diagnostics(
    tf: Any,
    physical: Any,
    trace: Mapping[str, Any],
    *,
    checkpoints: Sequence[int] = CHECKPOINTS,
    recent_window: int = RECENT_WINDOW,
    baseline_checkpoint: int = BASELINE_CHECKPOINT,
) -> Mapping[str, Any]:
    values = tf.cast(tf.convert_to_tensor(physical), tf.float64)
    draws = int(values.shape[0])
    points = tuple(int(value) for value in checkpoints)
    if (
        not points
        or tuple(sorted(set(points))) != points
        or points[-1] > draws
        or baseline_checkpoint not in points
    ):
        raise ValueError("invalid R-hat checkpoint schedule")
    rows = []
    for checkpoint in points:
        cumulative = _diagnose_block(
            tf,
            values[:checkpoint],
            _slice_trace(trace, 0, checkpoint),
        )
        recent = None
        if checkpoint >= int(recent_window):
            start = checkpoint - int(recent_window)
            recent = _diagnose_block(
                tf,
                values[start:checkpoint],
                _slice_trace(trace, start, checkpoint),
            )
        rows.append(
            {
                "checkpoint_draws": checkpoint,
                "cumulative": cumulative,
                "recent_window": recent,
                "recent_window_draws": (
                    None if recent is None else int(recent_window)
                ),
            }
        )
    endpoint = rows[-1]["cumulative"]
    return {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_trajectory.v1",
        "checkpoint_schedule": list(points),
        "recent_window_draws": int(recent_window),
        "rhat_threshold": RHAT_THRESHOLD,
        "rows": rows,
        "trajectory": _trajectory_summary(
            rows, baseline_checkpoint=baseline_checkpoint
        ),
        "endpoint": endpoint,
        "diagnostic_screen_passed_at_endpoint": endpoint["screen_passed"],
        "posterior_admission": False,
        "nonclaims": [
            "checkpoint diagnostics are descriptive and not repeated experiments",
            "no ESS or canonical sequential-HMC admission",
            "no posterior or predictive use",
        ],
    }


def _max_float_residual(left: Any, right: Any) -> float | None:
    residuals: list[float] = []

    def visit(a: Any, b: Any) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in set(a).intersection(b):
                visit(a[key], b[key])
            return
        if isinstance(a, list) and isinstance(b, list):
            for item_a, item_b in zip(a, b):
                visit(item_a, item_b)
            return
        if (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and not isinstance(b, bool)
        ):
            residuals.append(abs(float(a) - float(b)))

    visit(_json_ready(left), _json_ready(right))
    return max(residuals) if residuals else None


def _replay_tieout(
    baseline: Mapping[str, Any], checkpoint_payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    row = next(
        item
        for item in checkpoint_payload["rows"]
        if int(item["checkpoint_draws"]) == BASELINE_CHECKPOINT
    )
    prior_rhat = _json_ready(baseline["modern_rhat"])
    current_rhat = _json_ready(row["cumulative"]["physical_rhat"])
    prior_diagnostics = baseline["verification_diagnostics"]
    acceptance_keys = (
        "acceptance_rate",
        "acceptance_probability_by_chain",
        "binary_acceptance_rate",
        "binary_acceptance_by_chain",
    )
    prior_acceptance = {
        key: _json_ready(prior_diagnostics.get(key)) for key in acceptance_keys
    }
    current_acceptance = _json_ready(row["cumulative"]["acceptance"])
    rhat_equal = current_rhat == prior_rhat
    acceptance_equal = current_acceptance == prior_acceptance
    return {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_replay_tieout.v1",
        "checkpoint_draws": BASELINE_CHECKPOINT,
        "same_initial_bank_step_leapfrog_burnin_and_stateless_seed": True,
        "rhat_summary_exactly_equal": rhat_equal,
        "acceptance_summary_exactly_equal": acceptance_equal,
        "deterministic_summary_replay_passed": bool(rhat_equal and acceptance_equal),
        "rhat_maximum_absolute_float_residual": _max_float_residual(
            current_rhat, prior_rhat
        ),
        "acceptance_maximum_absolute_float_residual": _max_float_residual(
            current_acceptance, prior_acceptance
        ),
        "raw_prefix_identity_proved": False,
        "interpretation": (
            "summary-level replay supports, but cannot prove, raw-prefix identity"
            if rhat_equal and acceptance_equal
            else "cross-run exact-extension interpretation is forbidden; within-run trajectory remains usable"
        ),
    }


def _write_tensor_receipt(tf: Any, path: Path, value: Any) -> Mapping[str, Any]:
    tensor = tf.convert_to_tensor(value)
    serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
    _write_bytes(path, serialized)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
        "shape": [int(value) for value in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def _verify_tensor_receipt(tf: Any, receipt: Mapping[str, Any]) -> None:
    path = Path(str(receipt["path"]))
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != receipt.get("sha256"):
        raise RuntimeError(f"tensor archive hash mismatch: {path}")
    if len(blob) != int(receipt.get("bytes", -1)):
        raise RuntimeError(f"tensor archive byte-count mismatch: {path}")
    tensor = tf.io.parse_tensor(blob, out_type=tf.as_dtype(str(receipt["dtype"])))
    if [int(value) for value in tensor.shape] != list(receipt["shape"]):
        raise RuntimeError(f"tensor archive shape mismatch: {path}")


def _archive_raw(
    tf: Any,
    output: Path,
    *,
    latent: Any,
    physical: Any,
    trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = output / "raw"
    receipts: dict[str, Mapping[str, Any]] = {
        "latent_samples": _write_tensor_receipt(
            tf, root / "latent-samples.tftensor", latent
        ),
        "physical_samples": _write_tensor_receipt(
            tf, root / "physical-samples.tftensor", physical
        ),
        "observation_weight_sign_labels": _write_tensor_receipt(
            tf,
            root / "observation-weight-sign-labels.tftensor",
            tf.cast(
                tf.convert_to_tensor(physical)[:, :, OBSERVATION_WEIGHT_INDEX] < 0.0,
                tf.int32,
            ),
        ),
    }

    def archive_tree(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                archive_tree(f"{prefix}-{key}", item)
            return
        key = prefix.replace("_", "-")
        receipts[f"trace.{prefix}"] = _write_tensor_receipt(
            tf, root / f"{key}.tftensor", value
        )

    for key, value in trace.items():
        archive_tree(str(key), value)
    for receipt in receipts.values():
        _verify_tensor_receipt(tf, receipt)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_raw_archive.v1",
        "sample_layout": "draw_chain_parameter",
        "sign_layout": "draw_chain",
        "trace_layout": "draw_chain_or_draw_chain_parameter",
        "verified_after_write": True,
        "receipts": receipts,
        "posterior_eligible": False,
    }
    _write(output / "raw-archive.json", payload)
    return payload


def _write_artifact_inventory(output: Path) -> Mapping[str, Any]:
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name != "artifact-hashes.json"
        and not path.name.endswith(".tmp")
    }
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_hashes.v1",
        "artifacts": artifacts,
    }
    _write(output / "artifact-hashes.json", payload)
    return payload


def _execute(args: argparse.Namespace, output: Path, started: float) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    continuation, training_result, baseline, budget = _validated_inputs()
    route_policy = _route_policy_audit()
    forecast = _forecast()
    if not all(
        forecast[key]
        for key in (
            "fits_internal_with_closeout",
            "fits_prior_hmc_envelope",
            "fits_aggregate_grant",
        )
    ):
        raise DiagnosticBudgetExhausted("the single diagnostic call is under-budgeted")
    _write(output / "resource-admission.json", forecast)

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    import tensorflow as tf

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical}")

    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportFullChainConfig,
        FixedTransportHMCPolicy,
        build_fixed_transport_value_score_adapter,
        run_fixed_transport_full_chain_tfp_hmc,
    )
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    training_module = _load_module(
        TRAINING_RUNNER, "ssl_lstm_q20_rhat_frozen_training"
    )
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    if target.target_signature() != continuation.TARGET_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 target signature drift")
    if target.adapter_signature() != continuation.ADAPTER_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 adapter signature drift")
    base = BatchNativeBoundAdapter(target, target_signature=target.target_signature())
    nomination = next(
        item
        for item in training_result["nominations"]
        if int(item["seed"]) == TRANSPORT_SEED
    )
    trainer, transport_identity = continuation._load_candidate(
        tf,
        training_module,
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
        nomination,
    )
    geometry = _read_json(continuation.GEOMETRY)
    representatives = tf.constant(
        [
            geometry["representatives"][label]["position"]
            for label in ("plus", "minus")
        ],
        tf.float64,
    )
    initial = continuation._initial_state(tf, trainer.transport, representatives)
    if _json_ready(initial) != baseline["initial_state_bank"]:
        raise RuntimeError("r2 initial-state bank replay mismatch")
    adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base,
        fixed_transport=trainer.transport,
        target_scope=TARGET_SCOPE,
        evidence_path=None,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    kernel = baseline["kernel"]
    if adapter.adapter_signature() != kernel["transformed_adapter_signature"]:
        raise RuntimeError("r2 transformed-adapter identity mismatch")
    if adapter.transport_manifest_hash != kernel["fixed_transport_manifest_hash"]:
        raise RuntimeError("r2 transport-manifest identity mismatch")
    if base.adapter_signature() != kernel["base_adapter_signature"]:
        raise RuntimeError("r2 base-adapter identity mismatch")
    parity = training_module._parity_report(
        tf, trainer, target, adapter, representatives
    )
    if not bool(tf.convert_to_tensor(parity["passed"], tf.bool).numpy()):
        raise RuntimeError("diagnostic consumer exact pullback parity failed")

    launch = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_launch.v1",
        "status": "DIAGNOSTIC_STARTED",
        "timestamp_utc": _utc_now(),
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "runner": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "immutable_r2": {
            "result": {"path": R2_RESULT.as_posix(), "sha256": R2_RESULT_SHA256},
            "manifest": {
                "path": R2_MANIFEST.as_posix(),
                "sha256": R2_MANIFEST_SHA256,
            },
            "artifact_hashes": {
                "path": R2_HASHES.as_posix(),
                "sha256": R2_HASHES_SHA256,
            },
            "tuning": {"path": R2_TUNING.as_posix(), "sha256": R2_TUNING_SHA256},
        },
        "failed_pre_tensorflow_launch": budget["failed_pre_tensorflow_launch"],
        "transport_identity": transport_identity,
        "kernel_identity": kernel,
        "initial_state": initial,
        "consumer_exact_pullback_parity": parity,
        "config": {
            "num_results": NUM_RESULTS,
            "num_burnin_steps": NUM_BURNIN_STEPS,
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "seed": HMC_SEED,
            "checkpoints": CHECKPOINTS,
            "recent_window": RECENT_WINDOW,
            "target_scope": TARGET_SCOPE,
        },
        "budget": budget,
        "route_policy": route_policy,
        "shared_hmc_runtime": SHARED_HMC_RUNTIME,
        "memory_policy": memory_policy,
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": [str(device) for device in logical],
        "managed_session_trust_basis": (
            "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "launcher_attempt_index": 2,
        "gpu_initializing_attempt_limit": 1,
        "promotion_eligible": False,
    }
    _write(output / "launch.json", launch)

    prior_config = baseline["verification_config"]
    config = FixedTransportFullChainConfig(
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN_STEPS,
        step_size=STEP_SIZE,
        num_leapfrog_steps=NUM_LEAPFROG_STEPS,
        seed=HMC_SEED,
        use_xla=True,
        trace_policy=str(prior_config["trace_policy"]),
        target_status_trace_policy=str(prior_config["target_status_trace_policy"]),
        tuning_policy=FixedTransportHMCPolicy.fixed(
            source=str(prior_config["tuning_policy"]["source"])
        ),
        target_scope=TARGET_SCOPE,
        chain_execution_mode="tf_function",
    )
    call_started = time.perf_counter()
    run = run_fixed_transport_full_chain_tfp_hmc(adapter, initial, config)
    call_seconds = time.perf_counter() - call_started
    latent = tf.ensure_shape(
        tf.cast(run.samples, tf.float64),
        (NUM_RESULTS, CHAIN_COUNT, PARAMETER_COUNT),
    )
    physical = tf.reshape(
        trainer.transport.forward_batch(tf.reshape(latent, (-1, PARAMETER_COUNT))),
        (NUM_RESULTS, CHAIN_COUNT, PARAMETER_COUNT),
    )
    tf.debugging.assert_all_finite(physical, "diagnostic physical samples")
    archive = _archive_raw(
        tf, output, latent=latent, physical=physical, trace=run.trace
    )
    checkpoints = _checkpoint_diagnostics(tf, physical, run.trace)
    tieout = _replay_tieout(baseline, checkpoints)
    _write(output / "checkpoint-diagnostics.json", checkpoints)
    _write(output / "replay-tieout.json", tieout)

    diagnostics = _json_ready(run.diagnostics)
    status_telemetry = diagnostics.get("target_status_telemetry")
    chain_moved = tf.reduce_any(
        tf.not_equal(latent, initial[tf.newaxis, :, :]), axis=(0, 2)
    )
    hard_vetoes: list[str] = []
    for key, reason in (
        ("samples_all_finite", "samples_nonfinite"),
        ("log_accept_ratio_finite", "log_accept_ratio_nonfinite"),
        ("target_log_prob_finite", "target_log_prob_nonfinite"),
        ("proposed_target_log_prob_finite", "proposed_target_log_prob_nonfinite"),
        ("target_score_finite", "target_score_nonfinite"),
    ):
        if diagnostics.get(key) is not True:
            hard_vetoes.append(reason)
    if (
        not isinstance(status_telemetry, Mapping)
        or status_telemetry.get("all_status_valid") is not True
    ):
        hard_vetoes.append("target_status_invalid_or_missing")
    if not bool(tf.reduce_all(chain_moved).numpy()):
        hard_vetoes.append("one_or_more_chains_did_not_move")
    if (
        diagnostics.get("divergence_status") == "available"
        and int(diagnostics.get("divergence_count") or 0) > 0
    ):
        hard_vetoes.append("native_divergence_detected")
    if archive.get("verified_after_write") is not True:
        hard_vetoes.append("raw_archive_not_verified")

    run_valid = not hard_vetoes
    endpoint_passed = bool(
        run_valid and checkpoints["diagnostic_screen_passed_at_endpoint"]
    )
    if not run_valid:
        terminal_status = "DIAGNOSTIC_RUN_INVALID"
        decision = "NO_SAMPLE_LENGTH_CONCLUSION_RUN_INVALID"
    elif endpoint_passed:
        terminal_status = "DIAGNOSTIC_SCREEN_PASSED_AT_4000"
        decision = "LONGER_HORIZON_SUFFICIENT_FOR_THIS_DIAGNOSTIC_SCREEN_ONLY"
    else:
        terminal_status = "DIAGNOSTIC_SCREEN_FAILED_AT_4000"
        decision = "DOUBLING_TO_4000_INSUFFICIENT_FOR_DECLARED_SCREEN"

    elapsed = time.perf_counter() - started
    result = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_result.v1",
        "status": terminal_status,
        "decision": decision,
        "run_valid": run_valid,
        "hard_vetoes": hard_vetoes,
        "diagnostic_screen_passed": endpoint_passed,
        "posterior_admitted": False,
        "predictive_authorized": False,
        "candidate_reinstated": False,
        "transport_seed": TRANSPORT_SEED,
        "kernel": kernel,
        "call_wall_seconds": call_seconds,
        "process_wall_seconds": elapsed,
        "aggregate_gpu_wall_seconds": PRIOR_R2_GPU_WALL_SECONDS + elapsed,
        "aggregate_campaign_process_wall_seconds": (
            PRIOR_R2_GPU_WALL_SECONDS
            + FAILED_LAUNCH_PROCESS_WALL_SECONDS
            + elapsed
        ),
        "diagnostics": diagnostics,
        "all_chains_moved": bool(tf.reduce_all(chain_moved).numpy()),
        "chain_moved": chain_moved,
        "endpoint": checkpoints["endpoint"],
        "trajectory": checkpoints["trajectory"],
        "replay_tieout": tieout,
        "raw_archive": archive,
        "budget": budget,
        "plan": launch["plan"],
        "runner": launch["runner"],
        "route_policy": route_policy,
        "memory_policy": memory_policy,
        "git": _git_manifest(),
        "timestamp_completed_utc": _utc_now(),
        "nonclaims": [
            "diagnostic screen is not canonical sequential-HMC admission",
            "no ESS, posterior, mode-weight, predictive, superiority, or default claim",
            "a failed 4000-draw screen rejects only sufficiency at this horizon",
            "native divergence unavailability is not zero divergences",
        ],
    }
    _write(output / "result.json", result)
    manifest = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_manifest.v1",
        "status": terminal_status,
        "command": list(sys.argv),
        "cwd": str(Path.cwd()),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": sys.version.split()[0],
        "git": result["git"],
        "plan": launch["plan"],
        "runner": launch["runner"],
        "output_root": output.as_posix(),
        "result": (output / "result.json").as_posix(),
        "checkpoint_diagnostics": (output / "checkpoint-diagnostics.json").as_posix(),
        "raw_archive": (output / "raw-archive.json").as_posix(),
        "wall_seconds": elapsed,
        "call_wall_seconds": call_seconds,
        "aggregate_gpu_wall_seconds": result["aggregate_gpu_wall_seconds"],
        "aggregate_campaign_process_wall_seconds": result[
            "aggregate_campaign_process_wall_seconds"
        ],
        "budget": budget,
        "random_seed": list(HMC_SEED),
        "transport_seed": TRANSPORT_SEED,
        "kernel": kernel,
        "checkpoint_schedule": list(CHECKPOINTS),
        "recent_window_draws": RECENT_WINDOW,
        "route_policy": route_policy,
        "memory_policy": memory_policy,
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": [str(device) for device in logical],
        "managed_session_trust_basis": launch["managed_session_trust_basis"],
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "source_dependencies": {
            "continuation_runner": {
                "path": CONTINUATION_RUNNER.as_posix(),
                "sha256": CONTINUATION_RUNNER_SHA256,
            },
            "fixed_hmc_mechanics": {
                "path": FIXED_HMC_MECHANICS.as_posix(),
                "sha256": FIXED_HMC_MECHANICS_SHA256,
            },
            "hmc_convergence": {
                "path": HMC_CONVERGENCE.as_posix(),
                "sha256": HMC_CONVERGENCE_SHA256,
            },
        },
        "timestamp_completed_utc": result["timestamp_completed_utc"],
    }
    _write(output / "manifest.json", manifest)
    inventory = _write_artifact_inventory(output)
    for relative, expected in inventory["artifacts"].items():
        if _sha256(output / relative) != expected:
            raise RuntimeError(f"terminal artifact inventory mismatch: {relative}")
    print(
        json.dumps(
            {
                "status": terminal_status,
                "decision": decision,
                "output": output.as_posix(),
                "observation_weight_rhat_change": checkpoints["trajectory"][
                    "observation_weight_rhat_change_baseline_to_endpoint"
                ],
                "endpoint_screen_passed": endpoint_passed,
            },
            sort_keys=True,
        )
    )
    return 0


def _write_abort_terminal(
    output: Path,
    *,
    status: str,
    error: Exception,
    started: float,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    elapsed = time.perf_counter() - started
    under_budgeted = status == "DIAGNOSTIC_UNDER_BUDGETED"
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_result.v1",
        "status": status,
        "decision": (
            "NO_SAMPLE_LENGTH_CONCLUSION_RESOURCE_STOP"
            if under_budgeted
            else "NO_SAMPLE_LENGTH_CONCLUSION_HARNESS_FAILURE"
        ),
        "run_valid": False,
        "diagnostic_screen_passed": False,
        "posterior_admitted": False,
        "predictive_authorized": False,
        "reason": str(error),
        "error_type": type(error).__name__,
        "traceback": None if under_budgeted else traceback.format_exc(),
        "failure_classification": (
            "resource_budget_exhaustion_not_sampler_failure"
            if under_budgeted
            else "implementation_or_infrastructure_failure_not_scientific_evidence"
        ),
        "process_wall_seconds": elapsed,
        "aggregate_gpu_wall_seconds": PRIOR_R2_GPU_WALL_SECONDS + elapsed,
        "aggregate_campaign_process_wall_seconds": (
            PRIOR_R2_GPU_WALL_SECONDS
            + FAILED_LAUNCH_PROCESS_WALL_SECONDS
            + elapsed
        ),
        "partial_artifacts": {
            path.relative_to(output).as_posix(): _sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name not in {"result.json", "manifest.json", "artifact-hashes.json"}
            and not path.name.endswith(".tmp")
        },
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "runner": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "requested_physical_device_selector": str(args.device),
        "git": _git_manifest(),
        "timestamp_completed_utc": _utc_now(),
        "nonclaims": [
            "resource or harness failure provides no sample-length evidence",
            "no candidate, posterior, predictive, scientific, or default admission",
        ],
    }
    _write(output / "result.json", payload)
    _write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_rhat_manifest.v1",
            "status": status,
            "command": list(sys.argv),
            "cwd": str(Path.cwd()),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_version": sys.version.split()[0],
            "git": payload["git"],
            "plan": payload["plan"],
            "runner": payload["runner"],
            "output_root": output.as_posix(),
            "result": (output / "result.json").as_posix(),
            "wall_seconds": elapsed,
            "aggregate_gpu_wall_seconds": payload["aggregate_gpu_wall_seconds"],
            "aggregate_campaign_process_wall_seconds": payload[
                "aggregate_campaign_process_wall_seconds"
            ],
            "requested_physical_device_selector": str(args.device),
            "timestamp_completed_utc": payload["timestamp_completed_utc"],
        },
    )
    _write_artifact_inventory(output)
    return payload


def main() -> int:
    args = _args()
    _validate_args(args)
    output = _reserve_output_root(args.output_root)
    started = time.perf_counter()
    try:
        return _execute(args, output, started)
    except DiagnosticBudgetExhausted as error:
        payload = _write_abort_terminal(
            output,
            status="DIAGNOSTIC_UNDER_BUDGETED",
            error=error,
            started=started,
            args=args,
        )
        print(json.dumps({"status": payload["status"], "output": output.as_posix()}))
        return 3
    except Exception as error:
        payload = _write_abort_terminal(
            output,
            status="DIAGNOSTIC_HARNESS_FAILURE",
            error=error,
            started=started,
            args=args,
        )
        print(json.dumps({"status": payload["status"], "output": output.as_posix()}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
