#!/usr/bin/env python3
"""Tune and run corrected HMC behind the frozen three-mode weighted NeuTra map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-neutra-three-mode-provenance-and-evidence-closure-plan-2026-08-17.md"
)
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "three-mode/component-aware-width128-depth6-updates10000-r1/trainer_states.json"
)
DEFAULT_ROOT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "three-mode/hmc-width128-depth6-r2"
)
CHAIN_COUNT = 4
ROOT_SEED = (20260812, 14001)
EXPECTED_CHECKPOINT_SHA256 = (
    "b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6"
)
EXPECTED_TARGET_SIGNATURE = (
    "3f5c692fa2d6c985c652ddad7394031d837f3dbd3e31ee14bbc8db62ad4a3a55"
)
EXPECTED_HIDDEN_LAYERS = (128, 128)
EXPECTED_STAGES = 6
EXPECTED_SELECTED_STEP = 8750


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_line_count": len(status)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("canary", "run"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--candidate-source",
        choices=("reviewed", "fresh-replication"),
        default="reviewed",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--cap-seconds", type=float, default=5400.0)
    return parser.parse_args()


def _configure_gpu(tf: Any) -> Mapping[str, Any]:
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    devices = tuple(tf.config.list_logical_devices("GPU"))
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible logical GPU, found {devices}")
    return _json_ready(policy)


def _require_reviewed_candidate(loaded: Any, target: Mapping[str, Any]) -> None:
    """Reject intact but scientifically ineligible three-mode checkpoints."""

    observed = {
        "checkpoint_sha256": str(loaded.checkpoint_sha256),
        "target_signature": str(target.get("target_signature", "")),
        "hidden_layers": tuple(int(value) for value in loaded.config.hidden_layers),
        "stages": int(loaded.config.stages),
        "selected_step": int(loaded.selected_step),
        "jit_compile": bool(loaded.config.jit_compile),
    }
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "hidden_layers": EXPECTED_HIDDEN_LAYERS,
        "stages": EXPECTED_STAGES,
        "selected_step": EXPECTED_SELECTED_STEP,
        "jit_compile": True,
    }
    mismatches = tuple(
        key for key, expected_value in expected.items() if observed[key] != expected_value
    )
    if mismatches:
        details = ", ".join(
            f"{key}={observed[key]!r} (expected {expected[key]!r})"
            for key in mismatches
        )
        raise RuntimeError(f"three-mode checkpoint is not the reviewed candidate: {details}")


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _nested_numeric_close(left: Any, right: Any, *, atol: float = 1.0e-14) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_numeric_close(left[key], right[key], atol=atol) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _nested_numeric_close(a, b, atol=atol) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    return left == right


def _require_fresh_replication_candidate(
    loaded: Any, target: Mapping[str, Any], checkpoint: Path
) -> None:
    """Admit only a hashed, passing fresh replica from the active campaign."""

    root = checkpoint.parent
    result_path = root / "result.json"
    manifest_path = root / "run_manifest.json"
    hashes_path = root / "artifact_hashes.json"
    for path in (result_path, manifest_path, hashes_path):
        if not path.is_file():
            raise RuntimeError(f"fresh replication evidence is missing: {path}")
    hashes = _read_json_object(hashes_path)
    if hashes.get("schema") != "bayesfilter.defensive_weighted_neutra_analytic_hashes.v1":
        raise RuntimeError("fresh replication hash schema mismatch")
    receipts = hashes.get("artifacts")
    if not isinstance(receipts, Mapping):
        raise RuntimeError("fresh replication hash receipts are missing")
    for path in (result_path, manifest_path, checkpoint):
        if receipts.get(path.name) != _sha256(path):
            raise RuntimeError(f"fresh replication artifact hash mismatch: {path.name}")

    result = _read_json_object(result_path)
    manifest = _read_json_object(manifest_path)
    training_target = result.get("target")
    decision = result.get("decision")
    selection = result.get("checkpoint_selection")
    config = result.get("config")
    capacity = result.get("capacity_arm")
    if not all(
        isinstance(value, Mapping)
        for value in (training_target, decision, selection, config, capacity)
    ):
        raise RuntimeError("fresh replication scientific metadata is incomplete")

    replication = result.get("replication")
    expected_target = {
        "identity": target["identity"],
        "target_probabilities": _json_ready(target["probabilities"]),
        "means": _json_ready(target["means"]),
        "covariances": _json_ready(target["covariances"]),
    }
    observed_target = {
        key: training_target.get(key) for key in expected_target
    }
    checks = {
        "mode": result.get("mode") == "three-mode-canary",
        "replication": isinstance(replication, int) and replication in (1, 2),
        "decision": decision.get("candidate_passed") is True,
        "target": _nested_numeric_close(observed_target, expected_target),
        "config": config == _json_ready(loaded.config.manifest_payload()),
        "hidden_width": capacity.get("hidden_width") == 128,
        "stages": capacity.get("stages") == 6,
        "selected_step": selection.get("weighted_update") == loaded.selected_step,
        "plan": manifest.get("active_plan_file") == PLAN.resolve().as_posix(),
        "jit_compile": manifest.get("jit_compile") is True,
        "memory_growth": (
            manifest.get("gpu_memory_policy", {}).get(
                "configured_before_logical_device_initialization"
            )
            is True
            and manifest.get("gpu_memory_policy", {}).get(
                "all_physical_devices_memory_growth"
            )
            is True
        ),
    }
    failed = tuple(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "fresh three-mode replication is ineligible: " + ", ".join(failed)
        )


def _build_runtime(
    checkpoint: Path, *, candidate_source: str = "reviewed"
) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
        AnalyticGaussianMixtureValueScoreAdapter,
        analytic_three_mode_target,
        component_aware_initial_state,
        load_weighted_neutra_transport,
    )

    memory_policy = _configure_gpu(tf)
    target = analytic_three_mode_target()
    loaded = load_weighted_neutra_transport(checkpoint, required_dimension=4)
    if candidate_source == "reviewed":
        _require_reviewed_candidate(loaded, target)
    elif candidate_source == "fresh-replication":
        _require_fresh_replication_candidate(loaded, target, checkpoint)
    else:
        raise ValueError(f"unsupported candidate source: {candidate_source}")
    base = AnalyticGaussianMixtureValueScoreAdapter(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope="weighted_neutra_three_mode_hmc:transformed_v1",
        runtime_backend="tensorflow_exact_three_component_mixture_weighted_iaf_hmc",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "one frozen weighted transport only",
            "component-aware initialization is a geometry diagnostic",
            "analytic three-component target only",
            "no general NeuTra or SSL-LSTM claim",
        ),
    )
    initial = component_aware_initial_state(
        loaded.transport, target, chain_count=CHAIN_COUNT
    )
    return tf, loaded, base, adapter, {
        "memory_policy": memory_policy,
        "target": target,
        "initial_state": initial,
    }


def _run_tuning(*, base: Any, loaded: Any, initial: Any, output: Path) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.10,
        leapfrog_grid=(3, 5, 10, 15, 20, 25),
        chain_count=CHAIN_COUNT,
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
        verification_num_results=2000,
        verification_num_burnin_steps=64,
        require_modern_rank_normalized_verification=True,
        verification_coordinate_system="hmc_coordinates",
        verification_min_retained_results_per_chain=2000,
        tune_seed_base=(20260812, 14101),
        screen_seed_base=(20260812, 14201),
        verification_seed_base=(20260812, 14301),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope="weighted_neutra_three_mode_hmc:tuning_v1",
        output_filename="tuning_result.json",
    )
    return tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=loaded.transport,
        initial_position=initial[0],
        config=config,
        output_dir=output / "tuning",
    ).payload()


def _run_sequential(
    *, adapter: Any, initial: Any, tuning: Mapping[str, Any], output: Path, cap: float
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )

    kernel = tuning.get("final_kernel_payload")
    if not isinstance(kernel, Mapping) or tuning.get("passed") is not True:
        raise RuntimeError("tuning did not produce a viable fixed kernel")
    leapfrog = int(kernel.get("num_leapfrog_steps", 0))
    step_size = float(kernel.get("step_size", 0.0))
    if leapfrog < 2:
        raise RuntimeError("L=1 is forbidden")
    if not step_size > 0.0:
        raise RuntimeError("tuning step size is invalid")
    started = time.perf_counter()
    config = SequentialNeuTraHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        seed=ROOT_SEED,
        warmup_chunk_size=500,
        warmup_min_results=2000,
        warmup_window_results=1000,
        warmup_max_results=10000,
        retained_chunk_size=500,
        retained_min_results=1000,
        retained_max_results=10000,
        retained_check_interval_results=1000,
        warmup_rhat_max=1.05,
        retained_rhat_max=1.01,
        bulk_ess_min=400.0,
        tail_ess_min=400.0,
        delta_h_abs_max=1000.0,
        acceptance_min=0.35,
        acceptance_max=0.95,
        chain_count=CHAIN_COUNT,
        use_xla=True,
        target_status_required=True,
        retained_ess_required=True,
        xla_qualification_required=False,
    )
    result = run_sequential_neutra_hmc(
        adapter,
        initial,
        config,
        archive_root=output / "archive",
        archive_label="weighted-three-mode",
        budget_check=lambda _transitions: time.perf_counter() - started < float(cap),
    )
    payload = {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **result.__dict__}
    _write(output / "sequential_result.json", payload)
    return payload


def _load_retained_samples(tf: Any, sequential: Mapping[str, Any]) -> Any:
    archive_root = Path(sequential["archive"]["root"])
    manifest_path = archive_root / "weighted-three-mode-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("retained_chunks", ())
    if not rows:
        raise RuntimeError("sequential archive has no retained chunks")
    samples = []
    for row in rows:
        receipt = row["sample_receipt"]
        path = Path(receipt["path"])
        if _sha256(path) != receipt["sha256"]:
            raise RuntimeError("retained sample receipt hash mismatch")
        samples.append(tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64))
    return tf.concat(samples, axis=0)


def main() -> int:
    args = _parse_args()
    if not float(args.cap_seconds) > 0.0:
        raise ValueError("cap_seconds must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen checkpoint is missing: {checkpoint}")
    args.output_root.mkdir(parents=True)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    tf, loaded, base, adapter, runtime = _build_runtime(
        checkpoint, candidate_source=args.candidate_source
    )
    manifest = {
        "schema": "bayesfilter.weighted_neutra_three_mode_hmc_manifest.v1",
        "plan": PLAN.as_posix(),
        "checkpoint": loaded.manifest_payload(),
        "candidate_source": args.candidate_source,
        "base_adapter_signature": base.adapter_signature(),
        "transformed_adapter_signature": adapter.adapter_signature(),
        "target": runtime["target"]["signature_payload"],
        "memory_policy": runtime["memory_policy"],
        "device": tuple(str(device) for device in tf.config.list_logical_devices("GPU")),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "initial_state": runtime["initial_state"],
        "git": _git_manifest(),
        "started_monotonic": started,
    }
    _write(args.output_root / "run_manifest.json", manifest)
    if args.mode == "canary":
        from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
            FixedTransportFullChainConfig,
            FixedTransportHMCPolicy,
            run_fixed_transport_full_chain_tfp_hmc,
        )

        canary = run_fixed_transport_full_chain_tfp_hmc(
            adapter,
            runtime["initial_state"],
            FixedTransportFullChainConfig(
                num_results=64,
                num_burnin_steps=64,
                step_size=0.05,
                num_leapfrog_steps=3,
                seed=(20260812, 14401),
                use_xla=True,
                trace_policy="full",
                target_status_trace_policy="per_chain_step",
                tuning_policy=FixedTransportHMCPolicy.fixed(source=PLAN.as_posix()),
                target_scope="weighted_neutra_three_mode_hmc:canary_v1",
                chain_execution_mode="tf_function",
            ),
        )
        moved = tf.reduce_any(
            tf.not_equal(
                canary.samples, runtime["initial_state"][tf.newaxis, ...]
            ),
            axis=(0, 2),
        )
        passed = bool(
            canary.diagnostics.get("samples_all_finite", False)
            and canary.diagnostics.get("target_log_prob_finite", False)
            and canary.diagnostics.get("target_score_finite", False)
            and canary.diagnostics.get("target_status_telemetry", {}).get(
                "all_status_valid", False
            )
            and bool(tf.reduce_all(moved).numpy())
            and canary.metadata.get("use_xla") is True
        )
        _write(
            args.output_root / "canary.json",
            {
                "schema": "bayesfilter.weighted_neutra_three_mode_hmc_canary.v1",
                "passed": passed,
                "samples_shape": tuple(int(value) for value in canary.samples.shape),
                "diagnostics": canary.diagnostics,
                "metadata": canary.metadata,
                "chain_moved": moved,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        if not passed:
            raise RuntimeError("three-mode GPU/XLA HMC canary failed")
        print(json.dumps({"mode": "canary", "passed": True, "output_root": args.output_root.as_posix()}))
        return 0

    tuning = _run_tuning(
        base=base,
        loaded=loaded,
        initial=runtime["initial_state"],
        output=args.output_root,
    )
    if tuning.get("passed") is not True:
        result_path = args.output_root / "result.json"
        _write(
            result_path,
            {
                "schema": "bayesfilter.weighted_neutra_three_mode_hmc_result.v1",
                "manifest": manifest,
                "tuning": tuning,
                "decision": {
                    "status": "three_mode_hmc_candidate_rejected_at_tuning",
                    "sequential_passed": False,
                    "analytic_primary_screens_passed": False,
                    "repair_trigger": "no_viable_fixed_hmc_kernel",
                    "nonclaims": (
                        "no posterior claim because sequential sampling did not run",
                        "no conclusion about the weighted training direction",
                    ),
                },
                "wall_seconds": time.perf_counter() - started,
            },
        )
        _write(
            args.output_root / "artifact_hashes.json",
            {
                "schema": "bayesfilter.weighted_neutra_three_mode_hmc_hashes.v1",
                "artifacts": {
                    path.name: _sha256(path)
                    for path in sorted(args.output_root.iterdir())
                    if path.is_file() and path.name != "artifact_hashes.json"
                },
            },
        )
        print(
            json.dumps(
                {
                    "mode": "run",
                    "passed": False,
                    "status": "three_mode_hmc_candidate_rejected_at_tuning",
                    "wall_seconds": time.perf_counter() - started,
                    "output_root": args.output_root.as_posix(),
                },
                sort_keys=True,
            )
        )
        return 0
    sequential = _run_sequential(
        adapter=adapter,
        initial=runtime["initial_state"],
        tuning=tuning,
        output=args.output_root,
        cap=min(3600.0, float(args.cap_seconds)),
    )
    from bayesfilter.testing.gaussian_mixture_diagnostics_tf import (
        retained_gaussian_mixture_diagnostics,
    )

    retained_latent = _load_retained_samples(tf, sequential)
    flat_latent = tf.reshape(retained_latent, (-1, 4))
    retained_physical = tf.reshape(
        loaded.transport.forward_batch(flat_latent), tf.shape(retained_latent)
    )
    analytic = retained_gaussian_mixture_diagnostics(
        retained_physical,
        runtime["target"]["probabilities"],
        runtime["target"]["means"],
        runtime["target"]["covariances"],
    )
    sequential_passed = bool(sequential.get("passed"))
    analytic_passed = bool(analytic["passed_primary_screens"])
    final_passed = sequential_passed and analytic_passed
    result_path = args.output_root / "result.json"
    _write(
        result_path,
        {
            "schema": "bayesfilter.weighted_neutra_three_mode_hmc_result.v1",
            "manifest": manifest,
            "tuning": tuning,
            "sequential": sequential,
            "retained_analytic_diagnostics": analytic,
            "decision": {
                "status": "three_mode_hmc_candidate_passed"
                if final_passed
                else "three_mode_hmc_candidate_rejected",
                "sequential_passed": sequential_passed,
                "analytic_primary_screens_passed": analytic_passed,
                "marginal_moment_role": "explanatory_only_not_joint_veto",
                "nonclaims": (
                    "no posterior equality proof",
                    "no reverse-KL comparison or ranking",
                    "no component-discovery claim",
                    "no cross-transport or SSL-LSTM claim",
                ),
            },
            "wall_seconds": time.perf_counter() - started,
        },
    )
    _write(
        args.output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.weighted_neutra_three_mode_hmc_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(args.output_root.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "mode": "run",
                "passed": final_passed,
                "wall_seconds": time.perf_counter() - started,
                "output_root": args.output_root.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
