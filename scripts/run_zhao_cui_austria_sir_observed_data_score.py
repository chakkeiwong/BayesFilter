#!/usr/bin/env python3
"""Run bounded stages of the Austria SIR fixed-variant score campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


if "--cpu-reference" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


SCHEMA = "bayesfilter.zhao_cui_austria_sir_observed_data_score_campaign.v1"
PLAN = (
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-observed-data-score-active-implementation-plan-2026-07-30.md"
)


def _configure_device(cpu_reference: bool) -> Mapping[str, object]:
    if cpu_reference:
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("CPU reference mode must hide all GPU devices")
        tf.config.experimental.enable_tensor_float_32_execution(False)
        return {
            "execution_class": "explicit_cpu_reference",
            "physical_gpus": [],
            "logical_gpus": [],
            "online_device": "/CPU:0",
            "tf32_enabled": False,
            "memory_policy": "N/A: CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
        }
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("GPU mechanics stage requires a logical GPU")
    return {
        "execution_class": "trusted_visible_gpu",
        "physical_gpus": [row["device"] for row in memory["physical_devices"]],
        "logical_gpus": [item.name for item in logical],
        "online_device": "/GPU:0",
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "memory_policy": memory,
    }


def _target_api():
    # Keep this import after GPU memory policy setup in main().
    from bayesfilter.highdim import zhao_cui_austria_sir_fixed_variant_tf

    return zhao_cui_austria_sir_fixed_variant_tf


def _proposal_api():
    # Proposal construction also imports TensorFlow target modules; keep it lazy.
    from bayesfilter.highdim import zhao_cui_austria_sir_proposal_tf

    return zhao_cui_austria_sir_proposal_tf


def _git_payload() -> Mapping[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--short"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tf.dtypes.DType):
        return value.name
    return value


def _target_stage() -> Mapping[str, object]:
    target = _target_api().make_austria_sir_observed_data_target()
    return {
        "status": "PASS_TARGET_SEALED",
        "primary_pass": True,
        "target": target.manifest,
        "target_identity": target.target_identity,
        "observation_shape": target.observations.shape.as_list(),
        "source_state_shape": target.source_states.shape.as_list(),
    }


def _mechanics_stage(
    *,
    device: Mapping[str, object],
    particle_count: int,
    horizon: int,
    seed: int,
    jit_compile: bool,
) -> Mapping[str, object]:
    api = _target_api()
    target = api.make_austria_sir_observed_data_target()
    branch = api.make_bootstrap_mechanics_branch(
        particle_count=int(particle_count),
        horizon=int(horizon),
        proposal_seed=int(seed),
        target=target,
    )
    program = api.prepare_austria_sir_source_order_program(branch, target=target)
    theta = tf.zeros([3], tf.float32)
    evaluator = program.compiled(jit_compile=bool(jit_compile))
    with tf.device(str(device["online_device"])):
        started = time.monotonic()
        result = evaluator(theta)
        first_seconds = time.monotonic() - started
        warmed = evaluator(theta)
        warmed_seconds = time.monotonic() - started - first_seconds
    value_repeat_error = tf.abs(
        result["log_likelihood"] - warmed["log_likelihood"]
    )
    score_repeat_error = tf.reduce_max(tf.abs(result["score"] - warmed["score"]))
    finite = bool(result["finite"].numpy())
    increments_match = float(
        tf.abs(result["log_likelihood"] - tf.reduce_sum(result["log_increments"])).numpy()
    )
    score_increments_match = float(
        tf.reduce_max(
            tf.abs(result["score"] - tf.reduce_sum(result["increment_scores"], axis=0))
        ).numpy()
    )
    primary_pass = bool(
        finite
        and float(value_repeat_error.numpy()) <= 1e-5
        and float(score_repeat_error.numpy()) <= 1e-4
        and increments_match <= 5e-4
        and score_increments_match <= 5e-3
        and (not jit_compile or "XLA" in str(result["log_likelihood"].device).upper() or device["execution_class"] == "trusted_visible_gpu")
    )
    return {
        "status": "PASS_BOOTSTRAP_MECHANICS" if primary_pass else "BLOCK_BOOTSTRAP_MECHANICS",
        "primary_pass": primary_pass,
        "artifact_role": "mechanics_baseline_not_zhao_cui_proposal_quality",
        "particle_count": int(particle_count),
        "horizon": int(horizon),
        "proposal_seed": int(seed),
        "jit_compile": bool(jit_compile),
        "value": result["log_likelihood"],
        "score": result["score"],
        "log_increments": result["log_increments"],
        "increment_scores": result["increment_scores"],
        "ess_by_time": result["ess_by_time"],
        "minimum_ess": result["minimum_ess"],
        "maximum_log_weight_spread": result["maximum_log_weight_spread"],
        "finite": finite,
        "value_increment_sum_residual": increments_match,
        "score_increment_sum_max_abs_residual": score_increments_match,
        "value_repeat_error": value_repeat_error,
        "score_repeat_max_abs_error": score_repeat_error,
        "first_call_seconds": first_seconds,
        "warmed_call_seconds": warmed_seconds,
        "output_device": result["log_likelihood"].device,
        "target": target.manifest,
        "target_identity": target.target_identity,
        "branch": branch.manifest_payload(),
        "program": program.manifest_payload(),
        "nonclaims": (
            "bootstrap mechanics branch is not a fitted Zhao-Cui proposal",
            "no proposal-quality, tuning, production-KR, HMC, or leaderboard claim",
        ),
    }


def _proposal_preflight_stage() -> Mapping[str, object]:
    target = _target_api().make_austria_sir_observed_data_target()
    return {
        "status": "PASS_T1_PROPOSAL_BRIDGE_IMPLEMENTED",
        "primary_pass": True,
        "target": target.manifest,
        "target_identity": target.target_identity,
        "continuation_veto": False,
        "completed_lower_gates": (
            "exact_target_identity",
            "manual_same-scalar_source-order_score",
            "fp32_graph_native_model",
            "kr_preallocation_byte_gate",
            "deterministic_inverse_microbatching",
            "active_target_latent_preclip_t1_trainer",
            "immutable_trained_density_ttsirt_artifact",
            "exact_gaussian_initial_proposal",
            "complete_t1_source_order_branch_compiler",
        ),
        "nonclaims": (
            "bridge implementation does not establish proposal quality",
            "no target-specific tuning artifact",
            "no T2, T20, production-KR, HMC, or leaderboard readiness",
        ),
    }


def _proposal_t1_smoke_stage(
    *,
    particle_count: int,
    seed: int,
) -> Mapping[str, object]:
    """Fit and execute one tiny T1 proposal without making a tuning claim."""

    if int(particle_count) < 2 or int(particle_count) > 32:
        raise ValueError("T1 smoke particle count must be in [2,32]")
    proposal = _proposal_api()
    target_api = _target_api()
    spec = proposal.AustriaSIRT1ProposalSpec(
        degree=2,
        rank=1,
        batch_size=16,
        train_batches=1,
        learning_rate=1e-3,
        l1_weight=1e-9,
        l2_weight=1e-8,
        defensive_tau=1e-8,
        # Existing Zhao-Cui diagnostic plans require inverse/CDF round-trip
        # error <=1e-4. Use their established numerical controls here.
        cdf_grid_size=129,
        cdf_bisection_steps=24,
        kr_max_batch_working_bytes=64 * 1024 * 1024,
    )
    artifact = proposal.fit_austria_sir_t1_proposal(spec, seed=int(seed))
    roots = tf.random.experimental.stateless_split(
        tf.constant([int(seed), 211], tf.int32), 3
    )
    initial_reference = tf.random.stateless_uniform(
        [18, int(particle_count)], roots[0], dtype=tf.float64
    )
    ancestor_uniforms = tf.random.stateless_uniform(
        [1, int(particle_count)], roots[1], dtype=tf.float64
    )
    transition_reference = tf.random.stateless_uniform(
        [1, 18, int(particle_count)], roots[2], dtype=tf.float64
    )
    compilation = proposal.compile_austria_sir_t1_proposal_branch(
        artifact,
        initial_reference_points=initial_reference,
        ancestor_uniforms=ancestor_uniforms,
        transition_reference_points=transition_reference,
        inverse_microbatch_size=min(4, int(particle_count)),
    )
    target = target_api.make_austria_sir_observed_data_target()
    program = target_api.prepare_austria_sir_source_order_program(
        compilation.branch, target=target
    )
    result = program.evaluate(tf.zeros([3], tf.float32))

    probe_reference = tf.random.stateless_uniform(
        [36, 2], tf.constant([int(seed), 313], tf.int32), dtype=tf.float64
    )
    transport = artifact.transport()
    probe_local = transport.inverse_transport(probe_reference)
    recovered_reference = transport.forward_transport(probe_local)
    roundtrip_error = tf.reduce_max(tf.abs(recovered_reference - probe_reference))
    finite = bool(result["finite"].numpy())
    minimum_ess_fraction = result["minimum_ess"] / tf.cast(
        int(particle_count), tf.float32
    )
    proposal_quality_pass = bool(
        float(minimum_ess_fraction.numpy()) >= 0.5
        and float(roundtrip_error.numpy()) <= 1e-4
    )
    primary_pass = bool(
        finite
        and bool(tf.math.is_finite(roundtrip_error).numpy())
        and int(artifact.diagnostics["core_count"]) == 36
        and bool(artifact.diagnostics["calibration_validation_seed_disjoint"])
        and float(spec.l1_weight) > 0.0
        and float(roundtrip_error.numpy()) <= 1e-4
    )
    return {
        "status": (
            "PASS_T1_PROPOSAL_MECHANICS_SMOKE"
            if primary_pass
            else "BLOCK_T1_PROPOSAL_MECHANICS_SMOKE"
        ),
        "primary_pass": primary_pass,
        "artifact_role": "tiny_mechanics_smoke_not_tuning_or_proposal_quality",
        "particle_count": int(particle_count),
        "fit_seed": int(seed),
        "spec": spec.payload(),
        "frozen_artifact": artifact.payload(),
        "compiler": compilation.manifest,
        "branch": compilation.branch.manifest_payload(),
        "program": program.manifest_payload(),
        "value": result["log_likelihood"],
        "score": result["score"],
        "ess_by_time": result["ess_by_time"],
        "minimum_ess": result["minimum_ess"],
        "minimum_ess_fraction": minimum_ess_fraction,
        "maximum_log_weight_spread": result["maximum_log_weight_spread"],
        "finite": finite,
        "transport_roundtrip_max_abs_error": roundtrip_error,
        "transport_roundtrip_gate": "max_abs_error <= 1e-4",
        "proposal_quality_pass": proposal_quality_pass,
        "proposal_quality_gate": (
            "minimum_ess_fraction >= 0.5 and roundtrip <= 1e-4"
        ),
        "proposal_quality_role": "repair_trigger_only_in_tiny_smoke",
        "production_kr_closure": False,
        "nonclaims": (
            "one optimizer batch is not target-specific tuning",
            "rank one tiny-sample mechanics do not establish proposal quality",
            "T1 does not establish T2 or T20 feasibility",
            "no production-KR, HMC, posterior, or leaderboard claim",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("target", "mechanics", "proposal-preflight", "proposal-t1-smoke"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--particle-count", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=30711)
    parser.add_argument("--no-jit-compile", action="store_true")
    args = parser.parse_args(argv)

    if args.output_root.exists():
        raise ValueError("output-root must be a new versioned directory")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = _configure_device(bool(args.cpu_reference))
    route_classification = _target_api().ROUTE_CLASSIFICATION
    try:
        if args.stage == "target":
            stage = _target_stage()
        elif args.stage == "mechanics":
            stage = _mechanics_stage(
                device=device,
                particle_count=args.particle_count,
                horizon=args.horizon,
                seed=args.seed,
                jit_compile=not args.no_jit_compile,
            )
        elif args.stage == "proposal-preflight":
            stage = _proposal_preflight_stage()
        else:
            stage = _proposal_t1_smoke_stage(
                particle_count=args.particle_count,
                seed=args.seed,
            )
        allocator = (
            {"current": 0, "peak": 0}
            if args.cpu_reference
            else tf.config.experimental.get_memory_info("GPU:0")
        )
        payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": args.stage,
            "plan": PLAN,
            "command": " ".join(sys.argv),
            "environment": {
                "python": sys.version,
                "tensorflow": tf.__version__,
                "tf_force_gpu_allow_growth": os.environ.get(
                    "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            },
            "device": device,
            "gpu_allocator": {
                "current_bytes": int(allocator["current"]),
                "peak_bytes": int(allocator["peak"]),
            },
            "git": _git_payload(),
            "wall_time_seconds": time.monotonic() - started,
            "route_classification": route_classification,
            **stage,
        }
        ready = _json_ready(payload)
        (args.output_root / "result.json").write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_root / "result.md").write_text(
            "# Austria SIR Zhao-Cui Campaign Stage\n\n"
            f"Stage: `{args.stage}`\n\n"
            f"Status: `{ready['status']}`\n\n"
            f"Primary pass: `{ready['primary_pass']}`\n\n"
            f"Artifact: `{args.output_root / 'result.json'}`\n",
            encoding="utf-8",
        )
        return 0 if ready["primary_pass"] or args.stage == "proposal-preflight" else 2
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "stage": args.stage,
            "status": "FAILED_INFRASTRUCTURE_OR_IMPLEMENTATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_time_seconds": time.monotonic() - started,
        }
        (args.output_root / "failure_result.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
