#!/usr/bin/env python3
"""Run the bounded analytic fixed-length HMC ladder behind weighted NeuTra."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-plan-2026-08-12.md"
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/"
    "r1-two-mode/capacity-depth6-width128-updates10000-confirmation-1-v1/"
    "trainer_states.json"
)
DEFAULT_ROOT = ROOT / "docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12"
ROOT_SEED = (20260812, 91001)
CHAIN_COUNT = 4


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
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_line_count": len(status)}


def _fold_seed(seed: tuple[int, int], chain: int) -> tuple[int, int]:
    return int(seed[0]) + chain + 1, int(seed[1]) + 100_003 * (chain + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "canary", "run"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--cap-seconds", type=float, default=6000.0)
    return parser.parse_args()


def _configure_gpu(tf: Any) -> Mapping[str, Any]:
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    devices = tuple(tf.config.list_logical_devices("GPU"))
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible GPU, found {devices}")
    return _json_ready(policy)


def _build_runtime() -> tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import (
        AnalyticGaussianMixtureValueScoreAdapter,
        analytic_two_mode_target,
        load_weighted_neutra_transport,
        mode_aware_initial_state,
    )

    memory_policy = _configure_gpu(tf)
    loaded = load_weighted_neutra_transport(CHECKPOINT)
    target = analytic_two_mode_target()
    base = AnalyticGaussianMixtureValueScoreAdapter(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope="defensive_weighted_neutra_analytic_hmc:transformed_v1",
        runtime_backend="tensorflow_exact_mixture_weighted_iaf_hmc",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "one frozen training replication only",
            "analytic target only",
            "no SSL-LSTM or general NeuTra claim",
        ),
    )
    initial = mode_aware_initial_state(loaded.transport, target)
    return tf, loaded, base, adapter, {
        "memory_policy": memory_policy,
        "target": target,
        "initial_state": initial,
    }


def _run_tuning(
    *, tf: Any, base: Any, loaded: Any, adapter: Any, initial: Any, output: Path
) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )

    latent_reference = initial[0]
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        leapfrog_grid=(3, 5, 10, 15, 20, 25),
        chain_count=CHAIN_COUNT,
        initial_state_bank=tuple(
            tuple(float(value) for value in row) for row in initial.numpy().tolist()
        ),
        target_accept_prob=0.70,
        acceptance_band=(0.55, 0.90),
        repair_band=(0.40, 0.95),
        selection_policy="acceptance_target_distance",
        selection_replications=1,
        fixed_grid_fallback_acceptance_max=0.95,
        budget_schedule=(64, 128, 256),
        tune_num_results=16,
        screen_num_results=64,
        screen_num_burnin_steps=16,
        verification_num_results=4000,
        verification_num_burnin_steps=64,
        require_modern_rank_normalized_verification=True,
        verification_coordinate_system="hmc_coordinates",
        verification_min_retained_results_per_chain=4000,
        tune_seed_base=(20260812, 92001),
        screen_seed_base=(20260812, 93001),
        verification_seed_base=(20260812, 94001),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope="defensive_weighted_neutra_analytic_hmc:tuning_v1",
        tuning_policy=FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        output_filename="tuning_result.json",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=loaded.transport,
        initial_position=latent_reference,
        config=config,
        output_dir=output / "tuning",
    )
    payload = result.payload()
    return payload


def _run_sequential(
    *, tf: Any, adapter: Any, initial: Any, tuning: Mapping[str, Any], output: Path, cap: float
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )

    kernel = tuning.get("final_kernel_payload")
    if not isinstance(kernel, Mapping) or tuning.get("passed") is not True:
        raise RuntimeError("tuning did not produce a viable fixed kernel")
    leapfrog = int(kernel.get("num_leapfrog_steps", 0))
    if leapfrog < 2:
        raise RuntimeError("L=1 is forbidden")
    step_size = float(kernel.get("step_size", 0.0))
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

    def budget_check(_transitions: int) -> bool:
        return time.perf_counter() - started < float(cap)

    result = run_sequential_neutra_hmc(
        adapter,
        initial,
        config,
        archive_root=output / "archive",
        archive_label="weighted-analytic",
        budget_check=budget_check,
    )
    payload = {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **result.__dict__}
    _write(output / "sequential_result.json", payload)
    return payload


def _load_retained_samples(tf: Any, sequential: Mapping[str, Any]) -> Any:
    archive_root = Path(sequential["archive"]["root"])
    manifest_path = archive_root / "weighted-analytic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("retained_chunks", ())
    if not rows:
        raise RuntimeError("sequential archive has no retained chunks")
    tensors = []
    for row in rows:
        receipt = row["sample_receipt"]
        path = Path(receipt["path"])
        if _sha256(path) != receipt["sha256"]:
            raise RuntimeError("retained sample receipt hash mismatch")
        tensors.append(tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64))
    return tf.concat(tensors, axis=0)


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    if args.mode == "preflight":
        _write(
            args.output_root / "preflight.json",
            {
                "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_preflight.v1",
                "plan": PLAN.as_posix(),
                "checkpoint": CHECKPOINT.as_posix(),
                "checkpoint_sha256": _sha256(CHECKPOINT),
                "git": _git_manifest(),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            },
        )
        return 0
    tf, loaded, base, adapter, runtime = _build_runtime()
    manifest = {
        "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_run_manifest.v1",
        "plan": PLAN.as_posix(),
        "checkpoint": loaded.manifest_payload(),
        "base_adapter_signature": base.adapter_signature(),
        "transformed_adapter_signature": adapter.adapter_signature(),
        "target": runtime["target"]["signature_payload"],
        "memory_policy": runtime["memory_policy"],
        "device": tuple(str(device) for device in tf.config.list_logical_devices("GPU")),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
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

        config = FixedTransportFullChainConfig(
            num_results=32,
            num_burnin_steps=32,
            step_size=0.05,
            num_leapfrog_steps=3,
            seed=(20260812, 95001),
            use_xla=True,
            trace_policy="full",
            target_status_trace_policy="per_chain_step",
            tuning_policy=FixedTransportHMCPolicy.fixed(source=PLAN.as_posix()),
            target_scope="defensive_weighted_neutra_analytic_hmc:canary_v1",
            chain_execution_mode="tf_function",
        )
        result = run_fixed_transport_full_chain_tfp_hmc(adapter, runtime["initial_state"], config)
        _write(
            args.output_root / "canary.json",
            {
                "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_canary.v1",
                "samples_shape": tuple(int(value) for value in result.samples.shape),
                "diagnostics": result.diagnostics,
                "metadata": result.metadata,
                "all_finite": bool(result.diagnostics.get("samples_all_finite", False)),
            },
        )
        return 0
    tuning = _run_tuning(
        tf=tf,
        base=base,
        loaded=loaded,
        adapter=adapter,
        initial=runtime["initial_state"],
        output=args.output_root,
    )
    sequential = _run_sequential(
        tf=tf,
        adapter=adapter,
        initial=runtime["initial_state"],
        tuning=tuning,
        output=args.output_root,
        cap=min(3600.0, float(args.cap_seconds)),
    )
    from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import (
        retained_analytic_diagnostics,
    )

    retained_latent = _load_retained_samples(tf, sequential)
    flat_latent = tf.reshape(retained_latent, (-1, 4))
    flat_physical = loaded.transport.forward_batch(flat_latent)
    retained_physical = tf.reshape(flat_physical, tf.shape(retained_latent))
    analytic = retained_analytic_diagnostics(retained_physical)
    sequential_passed = bool(sequential.get("passed"))
    analytic_passed = bool(analytic["passed_primary_screens"])
    final_passed = sequential_passed and analytic_passed
    _write(
        args.output_root / "result.json",
        {
            "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_result.v1",
            "manifest": manifest,
            "tuning": tuning,
            "sequential": sequential,
            "retained_analytic_diagnostics": analytic,
            "decision": {
                "status": (
                    "analytic_hmc_candidate_passed"
                    if final_passed
                    else "analytic_hmc_candidate_rejected"
                ),
                "sequential_passed": sequential_passed,
                "analytic_screens_passed": analytic_passed,
                "promotion_claim": final_passed,
                "scope": "one frozen transport on one analytic target",
            },
            "wall_seconds": time.perf_counter() - started,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
