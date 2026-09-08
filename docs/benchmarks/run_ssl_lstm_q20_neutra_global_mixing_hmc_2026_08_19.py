#!/usr/bin/env python3
"""Tune and validate one common SSL-LSTM q=20 fixed-NeuTra HMC kernel."""

from __future__ import annotations

import argparse
import gc
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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md"
)
TRAINING_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py"
)
TRAINING_ROOT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "training-screen"
)
TRAINING_RESULT = TRAINING_ROOT / "result.json"
TRAINING_MANIFEST = TRAINING_ROOT / "manifest.json"
TRAINING_HASHES = TRAINING_ROOT / "artifact-hashes.json"
PREFLIGHT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "gpu-preflight.json"
)
CANARY_RESULT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/"
    "gpu-canary-retry-01/result.json"
)
GEOMETRY = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc"
)
ROUTE_LEDGER = ROOT / (
    "docs/plans/artifacts/"
    "neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/"
    "route_ledger.json"
)

TRAINING_RUNNER_SHA256 = "8c17fafcb25a5656ac1e90734f4b157703ddfad490b45ce01063c52779cd0c9f"
TRAINING_RESULT_SHA256 = "886a617eb60895bc97bc6530b74ef9e2578abee64771992fb29495c471cd92c7"
TRAINING_MANIFEST_SHA256 = "556e34a3ad9975c10cd5db327fbff2b0c71f82f46da4b840eb1ed11b7f6f1c76"
TRAINING_HASHES_SHA256 = "01af201e2350c58dcfd85e3e2a0e5d298584b752bf5df1441f2e47b4a3c6da90"
TRAINING_PLAN_SHA256 = "00b82d58140da2f64ba7722771c55b64f48f14fa58c9f7e4107d311a858f7a09"
PREFLIGHT_SHA256 = "28be07fcc83be539b9b643f2127094f025706d0a92b733873a85ddeb56b50a45"
CANARY_RESULT_SHA256 = "27685b3f22936659b5b5b34bc07c675eb16c47ddb86a465bfd83fb31c31d7bfa"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"

CAMPAIGN_WALL_CAP_SECONDS = 28800.0
PREDICTIVE_RESERVE_SECONDS = 3600.0
DERIVED_HMC_REMAINDER_SECONDS = 23503.589557993997
EXTERNAL_HMC_CAP_SECONDS = 23503
INTERNAL_HMC_CAP_SECONDS = 23323.0
CLOSEOUT_RESERVE_SECONDS = 180.0
CANARY_HMC_CALL_SECONDS = 771.3013279580045
CANARY_HMC_LEAPFROG_TRANSITIONS = (64 + 64) * 5
TUNING_CALL_OVERRUN_ALLOWANCE = 1.25

TRANSPORT_SEED_ORDER = (2, 3)
LEAPFROG_ORDER = (3, 5, 10, 15)
PARAMETER_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
OBSERVATION_WEIGHT_INDEX = 2
TUNING_VERIFICATION_RESULTS = 2000
TUNING_VERIFICATION_BURNIN = 64
SEQUENTIAL_CHUNK_RESULTS = 500
SEQUENTIAL_WARMUP_MIN = 2000
SEQUENTIAL_WARMUP_WINDOW = 1000
SEQUENTIAL_WARMUP_MAX = 10000
SEQUENTIAL_RETAINED_MIN = 2000
SEQUENTIAL_RETAINED_MAX = 10000
MECHANICS_RESULTS = 64
MECHANICS_BURNIN = 16
EXPECTED_NEUTRA_SEQUENTIAL_HMC_POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"


class HMCBudgetExhausted(RuntimeError):
    """Raised when the authorized remainder cannot cover the next required step."""


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
        raise RuntimeError(f"refusing to overwrite HMC artifact: {path}")
    temporary.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _write_tensor(path: Path, serialized: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite HMC tensor artifact: {path}")
    temporary.write_bytes(serialized)
    temporary.replace(path)


def _reserve_output_root(path: Path) -> Path:
    absolute = path if path.is_absolute() else ROOT / path
    if absolute.exists():
        raise RuntimeError(f"refusing to reuse HMC output root: {absolute}")
    absolute.mkdir(parents=True, exist_ok=False)
    return absolute


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
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument(
        "--campaign-wall-cap-seconds", type=float, default=CAMPAIGN_WALL_CAP_SECONDS
    )
    parser.add_argument(
        "--predictive-reserve-seconds", type=float, default=PREDICTIVE_RESERVE_SECONDS
    )
    parser.add_argument("--time-cap-seconds", type=float, default=INTERNAL_HMC_CAP_SECONDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).isdigit():
        raise SystemExit("device must be one nonnegative physical GPU index")
    training = args.training_root if args.training_root.is_absolute() else ROOT / args.training_root
    if training.resolve() != TRAINING_ROOT.resolve():
        raise SystemExit("training root is frozen to the reviewed Phase 3 artifact")
    if float(args.campaign_wall_cap_seconds) != CAMPAIGN_WALL_CAP_SECONDS:
        raise SystemExit("campaign wall cap is frozen to 28800 seconds")
    if float(args.predictive_reserve_seconds) != PREDICTIVE_RESERVE_SECONDS:
        raise SystemExit("predictive reserve is frozen to 3600 seconds")
    if float(args.time_cap_seconds) != INTERNAL_HMC_CAP_SECONDS:
        raise SystemExit("internal HMC cap is frozen to 23323 seconds")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validated_training_artifacts() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    fixed = (
        (TRAINING_RUNNER, TRAINING_RUNNER_SHA256),
        (TRAINING_RESULT, TRAINING_RESULT_SHA256),
        (TRAINING_MANIFEST, TRAINING_MANIFEST_SHA256),
        (TRAINING_HASHES, TRAINING_HASHES_SHA256),
        (PREFLIGHT, PREFLIGHT_SHA256),
        (CANARY_RESULT, CANARY_RESULT_SHA256),
        (GEOMETRY, GEOMETRY_SHA256),
    )
    for path, expected in fixed:
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen HMC input SHA-256 mismatch: {path}")
    hashes = _read_json(TRAINING_HASHES)
    if hashes.get("schema") != "bayesfilter.ssl_lstm.q20_neutra_training_hashes.v1":
        raise RuntimeError("training hash graph schema mismatch")
    inventory = hashes.get("artifacts")
    if not isinstance(inventory, Mapping) or len(inventory) != 30:
        raise RuntimeError("training hash graph must contain exactly 30 artifacts")
    for relative, expected in inventory.items():
        path = TRAINING_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"training artifact graph mismatch: {relative}")
    result = _read_json(TRAINING_RESULT)
    if result.get("status") != "TRAINING_SCREEN_AND_FROZEN_AUDIT_COMPLETED":
        raise RuntimeError("training screen did not pass its frozen audit phase")
    if result.get("plan", {}).get("sha256") != TRAINING_PLAN_SHA256:
        raise RuntimeError("training plan identity mismatch")
    if result.get("runner", {}).get("sha256") != TRAINING_RUNNER_SHA256:
        raise RuntimeError("training runner identity mismatch")
    nominations = result.get("nominations")
    audits = result.get("frozen_audits")
    if not isinstance(nominations, list) or [item.get("seed") for item in nominations] != [2, 3]:
        raise RuntimeError("training nominations must contain seeds 2 then 3")
    if not isinstance(audits, list) or [item.get("seed") for item in audits] != [2, 3]:
        raise RuntimeError("training audits must contain seeds 2 then 3")
    for nomination, audit in zip(nominations, audits, strict=True):
        if audit.get("arm_id") != nomination.get("arm_id") or audit.get("passed") is not True:
            raise RuntimeError("a nominated transport did not pass frozen audit")
        if audit.get("exact_pullback_parity", {}).get("passed") is not True:
            raise RuntimeError("a nominated transport failed exact pullback parity")
        state_path = Path(str(nomination.get("state_path", "")))
        if not state_path.is_file() or _sha256(state_path) != nomination.get("state_sha256"):
            raise RuntimeError("nominated state file identity mismatch")
    preflight = _read_json(PREFLIGHT)
    canary = _read_json(CANARY_RESULT)
    spent = (
        float(preflight["wall_time_seconds"])
        + float(canary["wall_seconds"])
        + float(result["wall_seconds"])
    )
    derived = CAMPAIGN_WALL_CAP_SECONDS - PREDICTIVE_RESERVE_SECONDS - spent
    if abs(derived - DERIVED_HMC_REMAINDER_SECONDS) > 1.0e-6:
        raise RuntimeError("derived HMC wall remainder mismatch")
    budget = {
        "campaign_wall_cap_seconds": CAMPAIGN_WALL_CAP_SECONDS,
        "predictive_reserve_seconds": PREDICTIVE_RESERVE_SECONDS,
        "preflight_wall_seconds": float(preflight["wall_time_seconds"]),
        "canary_wall_seconds": float(canary["wall_seconds"]),
        "training_wall_seconds": float(result["wall_seconds"]),
        "prior_gpu_wall_seconds": spent,
        "derived_hmc_remainder_seconds": derived,
        "external_hmc_cap_seconds": EXTERNAL_HMC_CAP_SECONDS,
        "internal_hmc_cap_seconds": INTERNAL_HMC_CAP_SECONDS,
        "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
    }
    return result, budget


def _route_policy_audit() -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc_policy import (
        load_neutra_hmc_route_ledger,
        require_neutra_hmc_route_policy,
    )

    ledger = load_neutra_hmc_route_ledger(ROUTE_LEDGER)
    audit = require_neutra_hmc_route_policy(ROOT, ledger)
    relative_runner = Path(__file__).resolve().relative_to(ROOT).as_posix()
    records = {
        str(item.get("path")): item
        for item in ledger.get("routes", ())
        if isinstance(item, Mapping)
    }
    record = records.get(relative_runner)
    if not isinstance(record, Mapping) or record.get("classification") != "active_claim_bearing":
        raise RuntimeError("current HMC runner lacks active route-ledger classification")
    if audit.get("canonical_policy_id") != EXPECTED_NEUTRA_SEQUENTIAL_HMC_POLICY_ID:
        raise RuntimeError("canonical NeuTra HMC policy identity drift")
    return {
        "ledger_path": ROUTE_LEDGER.as_posix(),
        "ledger_sha256": _sha256(ROUTE_LEDGER),
        "canonical_policy_id": audit["canonical_policy_id"],
        "passed": audit["passed"],
        "discovered_route_count": len(audit["discovered_routes"]),
        "classified_route_count": len(audit["classified_routes"]),
        "current_route": record,
    }


def _remaining(started: float, cap: float) -> float:
    return cap - (time.perf_counter() - started)


def _require_remaining(started: float, cap: float, context: str) -> float:
    remaining = _remaining(started, cap)
    if remaining <= CLOSEOUT_RESERVE_SECONDS:
        raise HMCBudgetExhausted(
            f"fewer than {CLOSEOUT_RESERVE_SECONDS:g} seconds remain before {context}"
        )
    return remaining


def _load_candidate(
    tf: Any,
    training_module: Any,
    trainer_type: Any,
    config_type: Any,
    nomination: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    state_path = Path(str(nomination["state_path"]))
    if _sha256(state_path) != nomination["state_sha256"]:
        raise RuntimeError("candidate state artifact drift")
    wrapper = _read_json(state_path)
    if wrapper.get("schema") != "bayesfilter.ssl_lstm.q20_neutra_training_state.v1":
        raise RuntimeError("candidate state schema mismatch")
    state = wrapper.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("candidate state payload missing")
    config = training_module._config_from_state(state, config_type)
    trainer = trainer_type(config)
    training_module._restore_trainer_state(tf, trainer, state)
    tensor_hash = training_module._stable_hash({"variables": state["variables"]})
    if state.get("state_hash") != nomination.get("training_state_hash"):
        raise RuntimeError("candidate training state hash mismatch")
    if tensor_hash != nomination.get("transport_tensor_hash"):
        raise RuntimeError("candidate transport tensor hash mismatch")
    trainer.transport.bind_frozen_identity(
        {
            "checkpoint_sha256": str(nomination["state_sha256"]),
            "training_state_hash": str(state["state_hash"]),
            "transport_tensor_hash": tensor_hash,
        }
    )
    return trainer, {
        "arm_id": nomination["arm_id"],
        "seed": int(nomination["seed"]),
        "state_path": state_path.as_posix(),
        "state_sha256": nomination["state_sha256"],
        "training_state_hash": state["state_hash"],
        "transport_tensor_hash": tensor_hash,
        "transport_manifest": trainer.transport.manifest_payload(),
    }


def _initial_state(tf: Any, transport: Any, representatives: Any) -> Any:
    latent, _ = transport.inverse_and_forward_logdet(representatives)
    perturbation = tf.constant((0.05, 0.0, 0.0, 0.0), tf.float64)
    initial = tf.stack(
        (
            latent[0],
            latent[1],
            latent[0] + perturbation,
            latent[1] - perturbation,
        ),
        axis=0,
    )
    tf.debugging.assert_all_finite(initial, "HMC initial latent bank")
    return tf.ensure_shape(initial, (4, 4))


def _tuning_seeds(seed: int, leapfrog_index: int) -> Mapping[str, tuple[int, int]]:
    return {
        "tune": (20260819, 30000 + 1000 * seed + 100 * leapfrog_index),
        "screen": (20260819, 40000 + 1000 * seed + 100 * leapfrog_index),
        "verification": (20260819, 50000 + 1000 * seed + leapfrog_index),
    }


def _ordered_tune(
    tf: Any,
    base: Any,
    transport: Any,
    initial: Any,
    *,
    seed: int,
    output: Path,
    campaign_started: float,
    time_cap_seconds: float,
) -> tuple[Mapping[str, Any] | None, str | None, list[Mapping[str, Any]]]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        run_fixed_transport_full_chain_tfp_hmc,
    )

    attempts = []
    for leapfrog_index, leapfrog in enumerate(LEAPFROG_ORDER):
        before_remaining = _require_remaining(
            campaign_started, time_cap_seconds, f"seed {seed} L={leapfrog} tuning"
        )
        scope = f"ssl_lstm_q20_neutra_global_mixing:seed-{seed}:L-{leapfrog}"
        seeds = _tuning_seeds(seed, leapfrog_index)
        config = FixedTransportHMCKernelTuningConfig(
            initial_step_size=0.05,
            leapfrog_grid=(leapfrog,),
            chain_count=4,
            initial_state_bank=tuple(
                tuple(float(value) for value in row) for row in initial.numpy().tolist()
            ),
            target_accept_prob=0.70,
            acceptance_band=(0.55, 0.90),
            repair_band=(0.40, 0.95),
            fixed_grid_fallback_acceptance_max=0.95,
            budget_schedule=(32, 64, 128),
            tune_num_results=16,
            screen_num_results=64,
            screen_num_burnin_steps=16,
            verification_num_results=TUNING_VERIFICATION_RESULTS,
            verification_num_burnin_steps=TUNING_VERIFICATION_BURNIN,
            require_modern_rank_normalized_verification=True,
            verification_min_retained_results_per_chain=TUNING_VERIFICATION_RESULTS,
            verification_rhat_max=1.01,
            verification_coordinate_system="raw_target_coordinates",
            tune_seed_base=seeds["tune"],
            screen_seed_base=seeds["screen"],
            verification_seed_base=seeds["verification"],
            chain_execution_mode="tf_function",
            use_xla=True,
            target_scope=scope,
            target_status_trace_policy="per_chain_step",
            output_filename="tuning-result.json",
            source=PLAN.as_posix(),
        )
        tuning_root = output / f"L-{leapfrog:02d}"
        tuning_started = time.perf_counter()
        tuning_calls: list[Mapping[str, Any]] = []
        resource_refusals: list[Mapping[str, Any]] = []

        def bounded_full_chain(active_adapter: Any, state: Any, chain_config: Any) -> Any:
            requested_work = (
                int(chain_config.num_results) + int(chain_config.num_burnin_steps)
            ) * int(chain_config.num_leapfrog_steps)
            predicted = (
                CANARY_HMC_CALL_SECONDS
                * requested_work
                / CANARY_HMC_LEAPFROG_TRANSITIONS
                * TUNING_CALL_OVERRUN_ALLOWANCE
            )
            remaining = _remaining(campaign_started, time_cap_seconds)
            allowed = remaining >= predicted + CLOSEOUT_RESERVE_SECONDS
            row: dict[str, Any] = {
                "num_results": int(chain_config.num_results),
                "num_burnin_steps": int(chain_config.num_burnin_steps),
                "num_leapfrog_steps": int(chain_config.num_leapfrog_steps),
                "requested_leapfrog_transitions": requested_work,
                "canary_hmc_call_seconds": CANARY_HMC_CALL_SECONDS,
                "canary_leapfrog_transitions": CANARY_HMC_LEAPFROG_TRANSITIONS,
                "overrun_allowance": TUNING_CALL_OVERRUN_ALLOWANCE,
                "predicted_call_seconds": predicted,
                "remaining_internal_hmc_wall_seconds": remaining,
                "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
                "allowed": allowed,
                "role": "engineering_resource_veto_only",
            }
            tuning_calls.append(row)
            if not allowed:
                resource_refusals.append(dict(row))
                raise HMCBudgetExhausted(
                    "canary-anchored forecast refused a fixed-HMC tuning call"
                )
            call_started = time.perf_counter()
            value = run_fixed_transport_full_chain_tfp_hmc(
                active_adapter, state, chain_config
            )
            row["actual_call_seconds"] = time.perf_counter() - call_started
            return value

        result = tune_fixed_transport_hmc_kernel(
            base_adapter=base,
            fixed_transport=transport,
            initial_position=initial[0],
            config=config,
            output_dir=tuning_root,
            run_full_chain=bounded_full_chain,
        )
        payload = _json_ready(result.payload())
        artifact = tuning_root / "tuning-result.json"
        if not artifact.is_file():
            raise RuntimeError("repository tuner did not emit its result artifact")
        receipt = {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_ordered_tuning_attempt.v1",
            "transport_seed": seed,
            "leapfrog_order_index": leapfrog_index,
            "num_leapfrog_steps": leapfrog,
            "ordered_stop_rule": "first complete singleton-grid verification pass",
            "scope": scope,
            "seeds": seeds,
            "remaining_wall_seconds_before": before_remaining,
            "elapsed_wall_seconds": time.perf_counter() - tuning_started,
            "remaining_wall_seconds_after": _remaining(campaign_started, time_cap_seconds),
            "tuning_artifact": artifact.as_posix(),
            "tuning_artifact_sha256": _sha256(artifact),
            "passed": bool(result.passed),
            "final_status": result.final_status,
            "hard_vetoes": payload.get("hard_vetoes", []),
            "repair_triggers": payload.get("repair_triggers", []),
            "final_kernel_hash": payload.get("final_kernel_hash"),
            "resource_budget_calls": tuning_calls,
            "resource_refusal": resource_refusals[-1] if resource_refusals else None,
        }
        _write(tuning_root / "attempt-receipt.json", receipt)
        attempts.append(receipt)
        _write(
            output / f"tuning-progress-{leapfrog_index + 1:02d}.json",
            {
                "schema": "bayesfilter.ssl_lstm.q20_neutra_ordered_tuning_progress.v1",
                "transport_seed": seed,
                "attempts": attempts,
                "selected_num_leapfrog_steps": leapfrog if result.passed else None,
            },
        )
        if resource_refusals:
            raise HMCBudgetExhausted(
                "canary-anchored forecast refused a fixed-HMC tuning call"
            )
        if result.passed:
            kernel = payload.get("final_kernel_payload")
            if not isinstance(kernel, Mapping):
                raise RuntimeError("passing tuner omitted the final kernel payload")
            return kernel, scope, attempts
    return None, None, attempts


def _mode_report(tf: Any, physical_samples: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_global_mixing import assess_retained_mode_mixing

    samples = tf.convert_to_tensor(physical_samples, tf.float64)
    if samples.shape.rank != 3 or samples.shape[1:] != (4, 4):
        raise RuntimeError("mode diagnostic requires [draw, 4, 4] physical samples")
    labels_draw_chain = tf.cast(
        samples[:, :, OBSERVATION_WEIGHT_INDEX] < 0.0, tf.int32
    )
    labels_chain_draw = tf.transpose(labels_draw_chain, (1, 0))
    report = assess_retained_mode_mixing(labels_chain_draw, region_count=2)
    return {
        **_json_ready(report.payload()),
        "labels_shape": [int(value) for value in labels_chain_draw.shape],
        "label_semantics": "1 iff observation_weight.0.0 < 0; 0 otherwise",
    }


def _run_mechanics(
    tf: Any,
    adapter: Any,
    transport: Any,
    initial: Any,
    kernel: Mapping[str, Any],
    *,
    seed: int,
    scope: str,
    output: Path,
) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportFullChainConfig,
        FixedTransportHMCPolicy,
        run_fixed_transport_full_chain_tfp_hmc,
    )

    config = FixedTransportFullChainConfig(
        num_results=MECHANICS_RESULTS,
        num_burnin_steps=MECHANICS_BURNIN,
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        seed=(20260819, 80000 + seed),
        use_xla=True,
        trace_policy="standard",
        target_status_trace_policy="per_chain_step",
        tuning_policy=FixedTransportHMCPolicy.fixed(source=PLAN.as_posix()),
        target_scope=scope,
        chain_execution_mode="tf_function",
    )
    result = run_fixed_transport_full_chain_tfp_hmc(adapter, initial, config)
    latent = tf.ensure_shape(result.samples, (MECHANICS_RESULTS, 4, 4))
    physical = tf.reshape(
        transport.forward_batch(tf.reshape(latent, (-1, 4))),
        (MECHANICS_RESULTS, 4, 4),
    )
    moved = tf.reduce_any(tf.not_equal(latent, initial[tf.newaxis, :, :]), axis=(0, 2))
    diagnostics = _json_ready(result.diagnostics)
    status = diagnostics.get("target_status_telemetry", {})
    health = bool(
        diagnostics.get("samples_all_finite") is True
        and diagnostics.get("log_accept_ratio_finite") is True
        and diagnostics.get("target_log_prob_finite") is True
        and diagnostics.get("target_score_finite") is True
        and isinstance(status, Mapping)
        and status.get("all_status_valid") is True
        and bool(tf.reduce_all(moved).numpy())
    )
    latent_path = output / "mechanics-latent.tftensor"
    physical_path = output / "mechanics-physical.tftensor"
    _write_tensor(latent_path, bytes(tf.io.serialize_tensor(latent).numpy()))
    _write_tensor(physical_path, bytes(tf.io.serialize_tensor(physical).numpy()))
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_fixed_hmc_mechanics.v1",
        "passed_health_gate": health,
        "mode_crossing_role": "explanatory_only_before_canonical_sequential_gate",
        "mode_mixing": _mode_report(tf, physical),
        "all_chains_moved": bool(tf.reduce_all(moved).numpy()),
        "chain_moved": moved,
        "diagnostics": result.diagnostics,
        "metadata": result.metadata,
        "config": config.signature_payload(),
        "latent_samples": {
            "path": latent_path.as_posix(),
            "sha256": _sha256(latent_path),
            "shape": [MECHANICS_RESULTS, 4, 4],
        },
        "physical_samples": {
            "path": physical_path.as_posix(),
            "sha256": _sha256(physical_path),
            "shape": [MECHANICS_RESULTS, 4, 4],
        },
    }
    _write(output / "mechanics-result.json", payload)
    return _json_ready(payload)


def _archive_callback(root: Path, tf: Any, label: str):
    def archive(
        *,
        stage: str,
        chunk_index: Any,
        latent_samples: Any,
        model_samples: Any,
        seed: Any,
        cumulative: bool,
    ) -> Mapping[str, Any]:
        suffix = "cumulative" if cumulative else f"chunk-{int(chunk_index):03d}"
        stage_root = root / stage
        latent_path = stage_root / f"{label}-{suffix}-latent.tftensor"
        physical_path = stage_root / f"{label}-{suffix}-physical.tftensor"
        latent_bytes = bytes(tf.io.serialize_tensor(latent_samples).numpy())
        physical_bytes = bytes(tf.io.serialize_tensor(model_samples).numpy())
        _write_tensor(latent_path, latent_bytes)
        _write_tensor(physical_path, physical_bytes)
        receipt = {
            "stage": stage,
            "chunk_index": chunk_index,
            "cumulative": bool(cumulative),
            "seed": None if seed is None else list(seed),
            "latent_path": latent_path.as_posix(),
            "latent_sha256": hashlib.sha256(latent_bytes).hexdigest(),
            "physical_path": physical_path.as_posix(),
            "physical_sha256": hashlib.sha256(physical_bytes).hexdigest(),
            "sample_shape": [int(value) for value in latent_samples.shape],
            "dtype": "float64",
        }
        _write(stage_root / f"{label}-{suffix}-receipt.json", receipt)
        return receipt

    return archive


def _retained_diagnostic(tf: Any, samples: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )

    physical = tf.convert_to_tensor(samples, tf.float64)
    sign = tf.cast(
        physical[:, :, OBSERVATION_WEIGHT_INDEX] < 0.0, tf.float64
    )[:, :, tf.newaxis]
    augmented = tf.concat((physical, sign), axis=-1)
    convergence = rank_normalized_hmc_diagnostics(
        augmented,
        parameter_names=(*PARAMETER_NAMES, "observation_weight_sign_indicator"),
        thresholds=RankNormalizedHMCThresholds(
            rhat_max=1.01,
            bulk_ess_min=1000.0,
            tail_ess_min=400.0,
        ),
    )
    mixing = _mode_report(tf, physical)
    passed = bool(convergence["passed"] and mixing["passed"] is True)
    return {
        "passed": passed,
        "convergence": convergence,
        "global_mixing": mixing,
        "indicator_tie_handling": (
            "Blom average-rank normalization in rank_normalized_hmc_diagnostics; "
            "direct per-chain sign coverage and transitions remain authoritative"
        ),
        "diagnostic_dimensions": [*PARAMETER_NAMES, "observation_weight_sign_indicator"],
    }


def _run_sequential(
    tf: Any,
    adapter: Any,
    transport: Any,
    initial: Any,
    kernel: Mapping[str, Any],
    *,
    seed: int,
    output: Path,
    campaign_started: float,
    time_cap_seconds: float,
    verification_call_seconds: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    from bayesfilter.inference.neutra_hmc import (
        NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )

    if NEUTRA_SEQUENTIAL_HMC_POLICY_ID != EXPECTED_NEUTRA_SEQUENTIAL_HMC_POLICY_ID:
        raise RuntimeError("canonical sequential HMC policy identity mismatch")

    config = SequentialNeuTraHMCConfig(
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        warmup_seed=(20260819, 60000 + 1000 * seed),
        retained_seed=(20260819, 70000 + 1000 * seed),
        warmup_chunk_results=SEQUENTIAL_CHUNK_RESULTS,
        warmup_min_results=SEQUENTIAL_WARMUP_MIN,
        warmup_check_window_results=SEQUENTIAL_WARMUP_WINDOW,
        warmup_max_results=SEQUENTIAL_WARMUP_MAX,
        warmup_rhat_max=1.05,
        retained_chunk_results=SEQUENTIAL_CHUNK_RESULTS,
        retained_min_results=SEQUENTIAL_RETAINED_MIN,
        retained_max_results=SEQUENTIAL_RETAINED_MAX,
        retained_rhat_max=1.01,
        minimum_chain_count=4,
        jit_compile=True,
    )

    def transform(samples: Any) -> Any:
        shape = tf.shape(samples)
        physical = transport.forward_batch(tf.reshape(samples, (-1, 4)))
        return tf.reshape(physical, shape)

    leapfrog = int(kernel["num_leapfrog_steps"])
    verification_work = (
        TUNING_VERIFICATION_RESULTS + TUNING_VERIFICATION_BURNIN
    ) * leapfrog
    seconds_per_leapfrog_transition = verification_call_seconds / verification_work
    budget_checks: list[Mapping[str, Any]] = []

    def budget_check(next_leapfrog_transitions: int) -> bool:
        remaining = _remaining(campaign_started, time_cap_seconds)
        predicted = seconds_per_leapfrog_transition * int(next_leapfrog_transitions)
        allowed = remaining >= predicted + CLOSEOUT_RESERVE_SECONDS
        budget_checks.append(
            {
                "next_leapfrog_transitions": int(next_leapfrog_transitions),
                "predicted_next_chunk_seconds": predicted,
                "remaining_internal_hmc_wall_seconds": remaining,
                "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
                "allowed": allowed,
                "role": "engineering_resource_veto_only",
            }
        )
        return allowed

    result = run_sequential_neutra_hmc(
        adapter=adapter,
        initial_state=initial,
        model_transform=transform,
        parameter_names=PARAMETER_NAMES,
        config=config,
        retained_diagnostic_fn=lambda samples: _retained_diagnostic(tf, samples),
        archive_callback=_archive_callback(
            output / "archive", tf, f"seed-{seed}-common-kernel"
        ),
        budget_check=budget_check,
    )
    if result.get("policy_id") != NEUTRA_SEQUENTIAL_HMC_POLICY_ID:
        raise RuntimeError("sequential controller result policy identity mismatch")
    retained = result["private_retained_raw"]
    final_diagnostic = (
        _retained_diagnostic(tf, retained)
        if int(retained.shape[0]) >= SEQUENTIAL_RETAINED_MIN
        else None
    )
    summary = {
        key: value
        for key, value in result.items()
        if not str(key).startswith("private_")
    }
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_sequential_hmc.v1",
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        **summary,
        "final_retained_diagnostic": final_diagnostic,
        "resource_budget_checks": budget_checks,
        "warmup_excluded_from_posterior": True,
        "candidate_transport_seed": seed,
    }
    _write(output / "sequential-result.json", payload)
    retained_archive = None
    cumulative = summary.get("cumulative_archives")
    if isinstance(cumulative, Mapping):
        retained_archive = cumulative.get("retained")
    return _json_ready(payload), _json_ready(retained_archive)


def _adapter_for_kernel(
    base: Any,
    transport: Any,
    kernel: Mapping[str, Any],
    scope: str,
) -> Any:
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        build_fixed_transport_value_score_adapter,
    )

    adapter = build_fixed_transport_value_score_adapter(
        base_adapter=base,
        fixed_transport=transport,
        target_scope=scope,
        evidence_path=None,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    if adapter.adapter_signature() != kernel.get("transformed_adapter_signature"):
        raise RuntimeError("selected kernel transformed-adapter identity mismatch")
    if adapter.transport_manifest_hash != kernel.get("fixed_transport_manifest_hash"):
        raise RuntimeError("selected kernel transport-manifest identity mismatch")
    if kernel.get("base_adapter_signature") != base.adapter_signature():
        raise RuntimeError("selected kernel base-adapter identity mismatch")
    if int(kernel.get("num_leapfrog_steps", 0)) < 2:
        raise RuntimeError("L=1 is forbidden")
    return adapter


def _write_terminal(
    output: Path,
    *,
    status: str,
    started: float,
    budget: Mapping[str, Any],
    attempts: list[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    retained_archive: Mapping[str, Any] | None,
    memory_policy: Mapping[str, Any],
    logical: tuple[Any, ...],
    args: argparse.Namespace,
    route_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc.v1",
        "policy_id": route_policy["canonical_policy_id"],
        "status": status,
        "passed": selected is not None,
        "decision": (
            "ADMIT_ONE_COMMON_KERNEL_FOR_PREDICTIVE"
            if selected is not None
            else "NO_HMC_CANDIDATE_ADMITTED"
        ),
        "selected_candidate": selected,
        "retained_posterior_archive": retained_archive,
        "candidate_attempts": attempts,
        "candidate_order": list(TRANSPORT_SEED_ORDER),
        "ordered_leapfrog_feasibility_ladder": list(LEAPFROG_ORDER),
        "pooling_across_candidates_or_mode_locked_chains": False,
        "budget": budget,
        "hmc_phase_wall_seconds": time.perf_counter() - started,
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "runner": {"path": Path(__file__).as_posix(), "sha256": _sha256(Path(__file__))},
        "training_inputs": {
            "result": {"path": TRAINING_RESULT.as_posix(), "sha256": TRAINING_RESULT_SHA256},
            "manifest": {
                "path": TRAINING_MANIFEST.as_posix(),
                "sha256": TRAINING_MANIFEST_SHA256,
            },
            "artifact_hashes": {
                "path": TRAINING_HASHES.as_posix(),
                "sha256": TRAINING_HASHES_SHA256,
            },
        },
        "memory_policy": memory_policy,
        "route_policy": route_policy,
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": [str(device) for device in logical],
        "managed_session_trust_basis": (
            "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "git": _git_manifest(),
        "nonclaims": [
            "tuning acceptance does not establish convergence",
            "unrun leapfrog lengths or fallback candidates are not failures",
            "known-sign traversal does not prove exhaustive mode discovery",
            "no sampler superiority, predictive equivalence, scientific validity, or default readiness",
        ],
    }
    _write(output / "result.json", payload)
    _write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc_manifest.v1",
            "status": status,
            "policy_id": route_policy["canonical_policy_id"],
            "command": list(sys.argv),
            "cwd": str(Path.cwd()),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_version": sys.version.split()[0],
            "git": payload["git"],
            "plan": payload["plan"],
            "runner": payload["runner"],
            "result": (output / "result.json").as_posix(),
            "output_root": output.as_posix(),
            "wall_seconds": payload["hmc_phase_wall_seconds"],
            "random_seed_policy": {
                "transport_seed_order": list(TRANSPORT_SEED_ORDER),
                "tuning": "phase-separated deterministic stateless seeds",
                "sequential": "distinct warmup and retained stateless roots",
            },
            "gpu_memory_growth_required": True,
            "memory_policy": memory_policy,
            "route_policy": route_policy,
            "requested_physical_device_selector": str(args.device),
            "visible_logical_gpus": [str(device) for device in logical],
            "managed_session_trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "dtype": "float64",
            "jit_compile": True,
            "tf32_enabled": False,
            "data_version": payload["training_inputs"],
            "timestamp_completed_utc": _utc_now(),
        },
    )
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    _write(
        output / "artifact-hashes.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc_hashes.v1",
            "artifacts": artifacts,
        },
    )
    return payload


def _write_abort_terminal(
    output: Path,
    *,
    status: str,
    error: Exception,
    started: float,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    launch_path = output / "launch.json"
    launch = _read_json(launch_path) if launch_path.is_file() else {}
    route_policy = launch.get("route_policy")
    policy_id = (
        route_policy.get("canonical_policy_id")
        if isinstance(route_policy, Mapping)
        else "NOT_VERIFIED_BEFORE_TERMINAL_STOP"
    )
    partial_artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"result.json", "manifest.json", "artifact-hashes.json"}
        and not path.name.endswith(".tmp")
    }
    under_budgeted = status == "UNDER_BUDGETED_HMC"
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc.v1",
        "policy_id": policy_id,
        "status": status,
        "passed": False,
        "decision": (
            "STOP_UNDER_BUDGETED_WITHOUT_CANDIDATE_REJECTION"
            if under_budgeted
            else "STOP_HMC_HARNESS_FAILURE"
        ),
        "reason": str(error),
        "error_type": type(error).__name__,
        "failure_classification": (
            "resource_budget_exhaustion_not_sampler_failure"
            if under_budgeted
            else "implementation_or_infrastructure_failure_not_scientific_evidence"
        ),
        "traceback": None if under_budgeted else traceback.format_exc(),
        "hmc_phase_wall_seconds": time.perf_counter() - started,
        "budget": launch.get(
            "budget",
            {
                "campaign_wall_cap_seconds": CAMPAIGN_WALL_CAP_SECONDS,
                "predictive_reserve_seconds": PREDICTIVE_RESERVE_SECONDS,
                "derived_hmc_remainder_seconds": DERIVED_HMC_REMAINDER_SECONDS,
                "external_hmc_cap_seconds": EXTERNAL_HMC_CAP_SECONDS,
                "internal_hmc_cap_seconds": INTERNAL_HMC_CAP_SECONDS,
                "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
            },
        ),
        "partial_artifacts": partial_artifacts,
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "runner": {"path": Path(__file__).as_posix(), "sha256": _sha256(Path(__file__))},
        "route_policy": route_policy,
        "memory_policy": launch.get("memory_policy"),
        "requested_physical_device_selector": str(args.device),
        "visible_logical_gpus": launch.get("visible_logical_gpus", []),
        "managed_session_trust_basis": launch.get("managed_session_trust_basis"),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "git": _git_manifest(),
        "timestamp_completed_utc": _utc_now(),
        "nonclaims": [
            "no HMC candidate, posterior, predictive, scientific, or default-readiness admission",
            "unrun kernels and transport candidates are not rejected",
            (
                "budget exhaustion is not a convergence failure"
                if under_budgeted
                else "a harness failure is not evidence against the target or NeuTra mechanism"
            ),
        ],
    }
    _write(output / "result.json", payload)
    _write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc_manifest.v1",
            "status": status,
            "policy_id": policy_id,
            "command": list(sys.argv),
            "cwd": str(Path.cwd()),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_version": sys.version.split()[0],
            "git": payload["git"],
            "plan": payload["plan"],
            "runner": payload["runner"],
            "result": (output / "result.json").as_posix(),
            "output_root": output.as_posix(),
            "wall_seconds": payload["hmc_phase_wall_seconds"],
            "budget": payload["budget"],
            "route_policy": route_policy,
            "memory_policy": payload["memory_policy"],
            "requested_physical_device_selector": str(args.device),
            "visible_logical_gpus": payload["visible_logical_gpus"],
            "managed_session_trust_basis": payload["managed_session_trust_basis"],
            "dtype": "float64",
            "jit_compile": True,
            "tf32_enabled": False,
            "timestamp_completed_utc": payload["timestamp_completed_utc"],
        },
    )
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact-hashes.json"
    }
    _write(
        output / "artifact-hashes.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc_hashes.v1",
            "artifacts": artifacts,
        },
    )
    return payload


def _execute(args: argparse.Namespace, output: Path, started: float) -> int:
    training_result, budget = _validated_training_artifacts()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    import tensorflow as tf

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical}")
    route_policy = _route_policy_audit()

    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    training_module = _load_module(
        TRAINING_RUNNER, "ssl_lstm_q20_global_mixing_training_frozen"
    )
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    if target.target_signature() != TARGET_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 target signature drift")
    if target.adapter_signature() != ADAPTER_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 adapter signature drift")
    base = BatchNativeBoundAdapter(target, target_signature=target.target_signature())
    geometry = _read_json(GEOMETRY)
    representatives = tf.constant(
        [
            geometry["representatives"][label]["position"]
            for label in ("plus", "minus")
        ],
        tf.float64,
    )
    _write(
        output / "launch.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_hmc_launch.v1",
            "status": "HMC_ORDERED_FEASIBILITY_STARTED",
            "timestamp_utc": _utc_now(),
            "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
            "runner": {"path": Path(__file__).as_posix(), "sha256": _sha256(Path(__file__))},
            "training_result": {
                "path": TRAINING_RESULT.as_posix(),
                "sha256": TRAINING_RESULT_SHA256,
            },
            "budget": budget,
            "memory_policy": memory_policy,
            "visible_logical_gpus": [str(device) for device in logical],
            "requested_physical_device_selector": str(args.device),
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
            "dtype": "float64",
            "managed_session_trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "route_policy": route_policy,
        },
    )

    attempts: list[Mapping[str, Any]] = []
    selected_candidate: Mapping[str, Any] | None = None
    retained_archive: Mapping[str, Any] | None = None
    nominations = training_result["nominations"]
    for expected_seed, nomination in zip(TRANSPORT_SEED_ORDER, nominations, strict=True):
        if int(nomination["seed"]) != expected_seed:
            raise RuntimeError("transport candidate order drift")
        _require_remaining(started, float(args.time_cap_seconds), "candidate load")
        candidate_root = output / f"candidate-seed-{expected_seed}"
        candidate_root.mkdir(parents=True, exist_ok=False)
        trainer, identity = _load_candidate(
            tf,
            training_module,
            WeightedForwardKLNeuTraTrainer,
            WeightedNeuTraConfig,
            nomination,
        )
        initial = _initial_state(tf, trainer.transport, representatives)
        candidate_started = time.perf_counter()
        kernel, scope, tuning_attempts = _ordered_tune(
            tf,
            base,
            trainer.transport,
            initial,
            seed=expected_seed,
            output=candidate_root / "tuning",
            campaign_started=started,
            time_cap_seconds=float(args.time_cap_seconds),
        )
        candidate_summary: dict[str, Any] = {
            "transport_seed": expected_seed,
            "identity": identity,
            "initial_state": _json_ready(initial),
            "tuning_attempts": tuning_attempts,
            "selected_kernel": kernel,
            "selected_tuning_scope": scope,
            "mechanics": None,
            "sequential": None,
            "status": "TUNING_NO_VIABLE_KERNEL",
            "passed": False,
            "wall_seconds": None,
        }
        if kernel is not None and scope is not None:
            adapter = _adapter_for_kernel(base, trainer.transport, kernel, scope)
            parity = training_module._parity_report(
                tf, trainer, target, adapter, representatives
            )
            if not bool(tf.convert_to_tensor(parity.get("passed"), tf.bool).numpy()):
                raise RuntimeError("HMC consumer exact pullback parity failed")
            mechanics = _run_mechanics(
                tf,
                adapter,
                trainer.transport,
                initial,
                kernel,
                seed=expected_seed,
                scope=scope,
                output=candidate_root / "mechanics",
            )
            candidate_summary["consumer_exact_pullback_parity"] = parity
            candidate_summary["mechanics"] = mechanics
            if mechanics.get("passed_health_gate") is not True:
                candidate_summary["status"] = "MECHANICS_HEALTH_VETO"
            else:
                verification_call = float(
                    kernel["verification_diagnostics"]["runtime_metadata"][
                        "sample_chain_call_s"
                    ]
                )
                verification_transitions = (
                    TUNING_VERIFICATION_RESULTS + TUNING_VERIFICATION_BURNIN
                )
                estimated_minimum = (
                    verification_call
                    * (SEQUENTIAL_WARMUP_MIN + SEQUENTIAL_RETAINED_MIN)
                    / verification_transitions
                )
                remaining = _remaining(started, float(args.time_cap_seconds))
                sufficiency = {
                    "role": "engineering_budget_sufficiency_only",
                    "same_kernel_verification_call_seconds": verification_call,
                    "verification_transition_count": verification_transitions,
                    "derived_seconds_per_transition": (
                        verification_call / verification_transitions
                    ),
                    "canonical_minimum_sequential_transitions": (
                        SEQUENTIAL_WARMUP_MIN + SEQUENTIAL_RETAINED_MIN
                    ),
                    "estimated_minimum_sequential_wall_seconds": estimated_minimum,
                    "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
                    "remaining_hmc_wall_seconds": remaining,
                    "sufficient": bool(
                        remaining >= estimated_minimum + CLOSEOUT_RESERVE_SECONDS
                    ),
                    "nonclaim": "runtime proportionality is an engineering estimate, not a sampler diagnostic",
                }
                candidate_summary["sequential_budget_sufficiency"] = sufficiency
                _write(candidate_root / "sequential-budget-sufficiency.json", sufficiency)
                if sufficiency["sufficient"] is not True:
                    candidate_summary["status"] = "UNDER_BUDGETED_BEFORE_SEQUENTIAL"
                    candidate_summary["wall_seconds"] = (
                        time.perf_counter() - candidate_started
                    )
                    _write(candidate_root / "candidate-result.json", candidate_summary)
                    attempts.append(_json_ready(candidate_summary))
                    raise HMCBudgetExhausted(
                        "same-kernel timing estimate cannot cover the canonical sequential minimum"
                    )
                sequential, archive = _run_sequential(
                    tf,
                    adapter,
                    trainer.transport,
                    initial,
                    kernel,
                    seed=expected_seed,
                    output=candidate_root / "sequential",
                    campaign_started=started,
                    time_cap_seconds=float(args.time_cap_seconds),
                    verification_call_seconds=verification_call,
                )
                candidate_summary["sequential"] = sequential
                candidate_summary["status"] = (
                    "COMMON_KERNEL_SEQUENTIAL_PASSED"
                    if sequential.get("passed") is True
                    else "COMMON_KERNEL_SEQUENTIAL_REJECTED"
                )
                candidate_summary["passed"] = sequential.get("passed") is True
                if candidate_summary["passed"]:
                    retained_archive = archive
                    selected_candidate = {
                        "transport_seed": expected_seed,
                        "arm_id": identity["arm_id"],
                        "kernel": kernel,
                        "tuning_scope": scope,
                        "retained_archive": archive,
                    }
        candidate_summary["wall_seconds"] = time.perf_counter() - candidate_started
        _write(candidate_root / "candidate-result.json", candidate_summary)
        attempts.append(_json_ready(candidate_summary))
        _write(
            output / f"candidate-progress-{len(attempts):02d}.json",
            {
                "schema": "bayesfilter.ssl_lstm.q20_neutra_hmc_candidate_progress.v1",
                "attempts": attempts,
                "selected_candidate": selected_candidate,
                "remaining_internal_hmc_wall_seconds": _remaining(
                    started, float(args.time_cap_seconds)
                ),
            },
        )
        if selected_candidate is not None:
            break
        if (
            isinstance(candidate_summary.get("sequential"), Mapping)
            and "campaign_resource_cap"
            in candidate_summary["sequential"].get("hard_vetoes", ())
        ):
            raise HMCBudgetExhausted(
                "measured same-kernel forecast refused the next sequential chunk"
            )
        del trainer
        gc.collect()

    status = (
        "HMC_ADMITTED_FOR_PREDICTIVE"
        if selected_candidate is not None
        else "HMC_NO_CANDIDATE_ADMITTED"
    )
    payload = _write_terminal(
        output,
        status=status,
        started=started,
        budget=budget,
        attempts=attempts,
        selected=selected_candidate,
        retained_archive=retained_archive,
        memory_policy=memory_policy,
        logical=logical,
        args=args,
        route_policy=route_policy,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "passed": payload["passed"],
                "output": output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _args()
    _validate_args(args)
    output = _reserve_output_root(args.output_root)
    started = time.perf_counter()
    try:
        return _execute(args, output, started)
    except HMCBudgetExhausted as error:
        payload = _write_abort_terminal(
            output,
            status="UNDER_BUDGETED_HMC",
            error=error,
            started=started,
            args=args,
        )
        print(json.dumps({"status": payload["status"], "output": output.as_posix()}))
        return 3
    except Exception as error:
        payload = _write_abort_terminal(
            output,
            status="HMC_HARNESS_FAILURE",
            error=error,
            started=started,
            args=args,
        )
        print(json.dumps({"status": payload["status"], "output": output.as_posix()}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
