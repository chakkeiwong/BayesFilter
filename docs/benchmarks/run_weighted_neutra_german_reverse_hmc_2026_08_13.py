#!/usr/bin/env python3
"""Run corrected fixed-length HMC for the frozen German reverse comparator."""

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

PLAN = ROOT / "docs/plans/bayesfilter-weighted-forward-kl-german-credit-plan-2026-08-13.md"
STATE_SCHEMA = "bayesfilter.weighted_neutra_german_reverse_state.v1"
CHAIN_COUNT = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cap-seconds", type=float, default=5400.0)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "as_list"):
        return _ready(value.as_list())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_tuning(base: Any, transport: Any, initial: Any, output: Path) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.05,
        leapfrog_grid=(3, 5, 10, 15, 20, 25, 32),
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
        tune_seed_base=(20260813, 45001),
        screen_seed_base=(20260813, 46001),
        verification_seed_base=(20260813, 47001),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope="weighted_neutra_german_credit:reverse_comparator:tuning_v1",
        output_filename="tuning_result.json",
    )
    return tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=initial[0],
        config=config,
        output_dir=output / "tuning",
    ).payload()


def _run_sequential(
    adapter: Any,
    initial: Any,
    tuning: Mapping[str, Any],
    output: Path,
    cap: float,
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc

    kernel = tuning.get("final_kernel_payload")
    if not isinstance(kernel, Mapping) or tuning.get("passed") is not True:
        raise RuntimeError("German reverse HMC tuning produced no viable kernel")
    leapfrog = int(kernel.get("num_leapfrog_steps", 0))
    if leapfrog < 2:
        raise RuntimeError("L=1 is forbidden")
    step_size = float(kernel.get("step_size", 0.0))
    if not step_size > 0.0:
        raise RuntimeError("German reverse HMC step size is invalid")
    started = time.perf_counter()
    config = SequentialNeuTraHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        seed=(20260813, 48001),
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
        archive_label="weighted-german-reverse",
        budget_check=lambda _transitions: time.perf_counter() - started < cap,
    )
    payload = {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **result.__dict__}
    _write(output / "sequential_result.json", payload)
    return payload


def _batch_means_mcse(values: Any) -> Any:
    import tensorflow as tf

    # values: [draw, chain, parameter]. Ten consecutive batches per chain are
    # available at the canonical 1,000 retained draws.
    draw_count = int(values.shape[0])
    chain_count = int(values.shape[1])
    batch_count_per_chain = 10
    batch_size = draw_count // batch_count_per_chain
    trimmed = values[: batch_count_per_chain * batch_size]
    chain_major = tf.transpose(trimmed, (1, 0, 2))
    batches = tf.reshape(
        chain_major, (chain_count * batch_count_per_chain, batch_size, int(values.shape[2]))
    )
    means = tf.reduce_mean(batches, axis=1)
    batch_variance = tf.math.reduce_variance(means, axis=0)
    return tf.sqrt(batch_variance / tf.cast(tf.shape(means)[0], tf.float64))


def _reference_diagnostics(
    tf: Any,
    spec: Any,
    transport: Any,
    archive_root: Path,
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_german_credit_target import constrained_from_unconstrained

    sample_paths = sorted((archive_root / "retained").glob("*-samples.tftensor"))
    if not sample_paths:
        raise RuntimeError("German reverse retained sample archive is missing")
    latent_chunks = [
        tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64) for path in sample_paths
    ]
    latent = tf.concat(latent_chunks, axis=0)
    if latent.shape.rank != 3 or latent.shape[1:] != (CHAIN_COUNT, spec.dimension):
        raise RuntimeError("German reverse retained sample shape mismatch")
    flat = tf.reshape(latent, (-1, spec.dimension))
    unconstrained = transport.forward_batch(flat)
    constrained = constrained_from_unconstrained(spec, unconstrained)
    chains = tf.reshape(constrained, tf.shape(latent))
    mean = tf.reduce_mean(chains, axis=(0, 1))
    square_rows = tf.square(chains)
    square = tf.reduce_mean(square_rows, axis=(0, 1))
    mean_mcse = _batch_means_mcse(chains)
    square_mcse = _batch_means_mcse(square_rows)
    reference_mean = tf.constant(spec.reference_mean, tf.float64)
    reference_square = tf.constant(spec.reference_square, tf.float64)
    mean_delta = mean - reference_mean
    square_delta = square - reference_square
    mean_z = tf.abs(mean_delta) / tf.maximum(mean_mcse, tf.constant(1.0e-12, tf.float64))
    square_z = tf.abs(square_delta) / tf.maximum(square_mcse, tf.constant(1.0e-12, tf.float64))
    return {
        "coordinate": "constrained=[z,local_scale,global_scale]",
        "draws_per_chain": int(latent.shape[0]),
        "chain_count": int(latent.shape[1]),
        "reference_has_stored_mcse": False,
        "candidate_batch_means_batches_per_chain": 10,
        "mean": mean,
        "square": square,
        "reference_mean": reference_mean,
        "reference_square": reference_square,
        "mean_delta": mean_delta,
        "square_delta": square_delta,
        "mean_batch_means_mcse": mean_mcse,
        "square_batch_means_mcse": square_mcse,
        "mean_abs_error_max": tf.reduce_max(tf.abs(mean_delta)),
        "mean_abs_error_median": _median(tf.sort(tf.abs(mean_delta))),
        "square_abs_error_max": tf.reduce_max(tf.abs(square_delta)),
        "square_abs_error_median": _median(tf.sort(tf.abs(square_delta))),
        "mean_abs_z_candidate_mcse_max": tf.reduce_max(mean_z),
        "square_abs_z_candidate_mcse_max": tf.reduce_max(square_z),
        "role": "required_posterior_diagnostic_without_calibrated_joint_threshold",
        "nonclaims": (
            "candidate MCSE does not include unknown reference MCSE",
            "coordinatewise z values are not a joint equality test",
        ),
    }


def _median(sorted_values: Any) -> Any:
    import tensorflow as tf

    count = int(sorted_values.shape[0])
    middle = count // 2
    if count % 2:
        return sorted_values[middle]
    return tf.constant(0.5, sorted_values.dtype) * (
        sorted_values[middle - 1] + sorted_values[middle]
    )


def main() -> int:
    args = _parse_args()
    root = args.output_root.resolve()
    training_root = args.training_root.resolve()
    if root.exists():
        raise FileExistsError(f"output root must be fresh: {root}")
    if not float(args.cap_seconds) > 0.0:
        raise ValueError("cap-seconds must be positive")
    required = (
        PLAN,
        args.data.resolve(),
        args.reference.resolve(),
        training_root / "trainer_state.json",
        training_root / "artifact_hashes.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("German reverse HMC inputs are missing")
    root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_german_credit_proposal import load_frozen_german_transport
    from bayesfilter.inference.neutra_german_credit_target import (
        GermanCreditValueScoreAdapter,
        load_german_credit_target_spec,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    started = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical_gpus}")
    spec = load_german_credit_target_spec(args.data, args.reference)
    frozen = load_frozen_german_transport(
        training_root / "trainer_state.json",
        training_root / "artifact_hashes.json",
        expected_schema=STATE_SCHEMA,
    )
    if (
        frozen.target_name != spec.name
        or frozen.target_data_sha256 != spec.data_sha256
        or frozen.target_reference_sha256 != spec.reference_sha256
    ):
        raise RuntimeError("German reverse HMC checkpoint target binding mismatch")
    base = GermanCreditValueScoreAdapter(spec)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=frozen.transport,
        target_scope="weighted_neutra_german_credit:reverse_comparator:hmc_v1",
        runtime_backend="tensorflow_exact_german_credit_reverse_iaf_hmc",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "frozen reverse-KL comparator only",
            "no weighted-candidate or objective-ranking claim",
        ),
    )
    initial = tf.random.stateless_normal(
        (CHAIN_COUNT, spec.dimension), seed=(20260813, 45000), dtype=tf.float64
    ) * tf.constant(0.1, tf.float64)
    manifest = {
        "schema": "bayesfilter.weighted_neutra_german_reverse_hmc_manifest.v1",
        "plan": PLAN.as_posix(),
        "training_root": training_root.as_posix(),
        "training_state_sha256": frozen.state_sha256,
        "training_state_hash": frozen.state_hash,
        "target": spec.manifest_payload(),
        "adapter_signature": adapter.adapter_signature(),
        "transport_manifest": frozen.transport.manifest_payload(),
        "memory_policy": memory_policy,
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "initial_state": initial,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    _write(root / "run_manifest.json", manifest)
    tuning = _run_tuning(base, frozen.transport, initial, root)
    if tuning.get("passed") is not True:
        result = {
            "schema": "bayesfilter.weighted_neutra_german_reverse_hmc_result.v1",
            "manifest": manifest,
            "tuning": tuning,
            "decision": {
                "status": "reverse_comparator_rejected_at_tuning",
                "weighted_training_reopened": False,
            },
            "wall_seconds": time.perf_counter() - started,
        }
        _write(root / "result.json", result)
        _write(
            root / "artifact_hashes.json",
            {
                "schema": "bayesfilter.weighted_neutra_german_reverse_hmc_hashes.v1",
                "artifacts": {
                    path.relative_to(root).as_posix(): _sha256(path)
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path.name != "artifact_hashes.json"
                },
            },
        )
        print(json.dumps({"passed": False, "output_root": root.as_posix()}))
        return 0
    sequential = _run_sequential(
        adapter,
        initial,
        tuning,
        root,
        min(float(args.cap_seconds), 3600.0),
    )
    reference = _reference_diagnostics(tf, spec, frozen.transport, root / "archive")
    passed = bool(sequential.get("passed"))
    result = {
        "schema": "bayesfilter.weighted_neutra_german_reverse_hmc_result.v1",
        "manifest": manifest,
        "tuning": tuning,
        "sequential": sequential,
        "reference_diagnostics": reference,
        "decision": {
            "status": "reverse_comparator_sampler_passed" if passed else "reverse_comparator_sampler_rejected",
            "weighted_training_reopened": False,
            "promotion": False,
            "primary_criterion": "canonical sequential R-hat/ESS and numerical status",
            "nonclaims": (
                "no weighted-candidate evidence",
                "no objective ranking",
                "no default promotion",
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(root / "result.json", result)
    _write(
        root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.weighted_neutra_german_reverse_hmc_hashes.v1",
            "artifacts": {
                path.relative_to(root).as_posix(): _sha256(path)
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"passed": passed, "output_root": root.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
