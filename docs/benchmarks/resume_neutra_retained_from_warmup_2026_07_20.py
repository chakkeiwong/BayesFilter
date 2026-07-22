#!/usr/bin/env python3
"""Resume retained NeuTra HMC from a validated archived warm-up boundary."""

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


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument("--admitted-kernel-replay", type=Path, required=True)
    parser.add_argument("--frozen-transport", type=Path, required=True)
    parser.add_argument("--frozen-transport-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cell", default="LGSSM-EXACT")
    parser.add_argument("--seed-offset", type=int, default=1000)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"resume output must be fresh: {args.output_root}")
    if _sha256(args.frozen_transport) != str(args.frozen_transport_sha256).lower():
        raise ValueError("frozen transport SHA-256 mismatch")

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
        rank_normalized_split_rhat_summary,
    )
    from bayesfilter.inference.hmc_kernel_tuning import (
        build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_end_to_end import (
        BatchNativeBoundAdapter,
        TensorArchive,
        _fixed_transport_adapter,
        _target_signature,
        _truth_tail,
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        run_batched_hmc,
        sequential_chunk_seed,
    )
    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == args.cell)
    adapter = spec.adapter_factory()
    if _target_signature(adapter) != spec.target_signature:
        raise ValueError("current target signature mismatch")
    bound = BatchNativeBoundAdapter(adapter, target_signature=spec.target_signature)
    loaded = load_frozen_neutra_artifact(
        _read_mapping(args.frozen_transport),
        expected_target_signature=spec.target_signature,
    )
    target_scope = f"{spec.cell_id}:fixed_neutra_native_tuning"
    tuned_adapter = _fixed_transport_adapter(bound, loaded.transport, target_scope)
    execution = {
        "dtype": "float64",
        "backend": "tensorflow_probability",
        "jit_compile": True,
        "tf32_execution_enabled": True,
        "mass_policy": "fixed_identity",
    }
    mechanics = _read_mapping(args.admitted_kernel_replay)
    replay = build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
        adapter=tuned_adapter,
        mechanics_payload=mechanics,
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        target_signature=spec.target_signature,
        target_scope=target_scope,
        execution=execution,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
    )

    expected_warmup_root_seed = (
        20260718,
        spec.initial_seed[1] + 100 + int(args.seed_offset),
    )
    warmup_latent = []
    warmup_model = []
    warmup_rows = []
    for index in range(2):
        chunk = args.warmup_root / f"chunk-{index:04d}"
        metadata = _read_mapping(chunk / "metadata.json")
        expected_seed = sequential_chunk_seed(expected_warmup_root_seed, index)
        if (
            metadata.get("schema")
            != "bayesfilter.neutra.all_models.tensor_archive.v1"
            or metadata.get("stage") != "warmup"
            or metadata.get("target_signature") != spec.target_signature
            or tuple(metadata.get("seed", ())) != expected_seed
            or tuple(metadata.get("sample_shape", ()))
            != (1000, 4, spec.parameter_dim)
        ):
            raise ValueError(f"warm-up archive contract mismatch: {chunk}")
        latent = tf.io.parse_tensor(
            tf.io.read_file(str(chunk / "latent.tensor")), out_type=tf.float64
        )
        model = tf.io.parse_tensor(
            tf.io.read_file(str(chunk / "model.tensor")), out_type=tf.float64
        )
        latent.set_shape((1000, 4, spec.parameter_dim))
        model.set_shape((1000, 4, spec.parameter_dim))
        if not bool(
            tf.reduce_all(tf.math.is_finite(latent)).numpy()
            and tf.reduce_all(tf.math.is_finite(model)).numpy()
        ):
            raise ValueError("warm-up archive contains nonfinite values")
        warmup_latent.append(latent)
        warmup_model.append(model)
        warmup_rows.append(
            {
                "chunk_index": index,
                "metadata_path": str(chunk / "metadata.json"),
                "latent_sha256": _sha256(chunk / "latent.tensor"),
                "model_sha256": _sha256(chunk / "model.tensor"),
                "seed": expected_seed,
            }
        )
    warmup_window = warmup_model[-1]
    warmup_rhat = rank_normalized_split_rhat_summary(
        warmup_window, rhat_max=1.05
    )
    if warmup_rhat.get("passed") is not True:
        raise ValueError("archived warm-up does not pass the declared R-hat gate")
    initial_state = warmup_latent[-1][-1]

    retained_root_seed = (
        20260718,
        spec.initial_seed[1] + 101 + int(args.seed_offset),
    )
    retained_seed = sequential_chunk_seed(retained_root_seed, 0)
    retained = run_batched_hmc(
        adapter=replay.adapter,
        initial_state=initial_state,
        config=BatchedHMCConfig(
            num_results=1000,
            num_burnin_steps=0,
            step_size=replay.step_size,
            num_leapfrog_steps=replay.num_leapfrog_steps,
            seed=retained_seed,
            jit_compile=True,
        ),
    )
    latent_samples = tf.convert_to_tensor(retained["samples"], tf.float64)
    shape = tf.shape(latent_samples)
    flat = tf.reshape(latent_samples, (-1, spec.parameter_dim))
    raw = loaded.transport.forward_batch(flat)
    physical = spec.physical_transform(tf, raw)
    model_samples = tf.reshape(physical, shape)
    health = retained["diagnostics"]
    thresholds = RankNormalizedHMCThresholds(1.01, 1000.0, 400.0)
    convergence = rank_normalized_hmc_diagnostics(
        model_samples,
        parameter_names=spec.parameter_names,
        thresholds=thresholds,
    )
    passed = bool(health.get("health_passed") is True and convergence["passed"])
    truth_tail = (
        _truth_tail(spec, model_samples)
        if passed
        else {
            "status": "NOT_EVALUATED_INVALID_SAMPLER",
            "minimum_p_truth": None,
            "parameter_rows": (),
        }
    )
    args.output_root.mkdir(parents=True)
    archive = TensorArchive(args.output_root / "samples", spec.target_signature)
    retained_archive = archive(
        stage="retained",
        chunk_index=0,
        latent_samples=latent_samples,
        model_samples=model_samples,
        seed=retained_seed,
        cumulative=False,
    )
    decision = (
        "PASS_TWO_SEED_TRUTH_TAIL"
        if passed and truth_tail["status"] == "PASS"
        else (
            str(truth_tail["status"])
            if passed
            else "HMC_CONVERGENCE_OR_HEALTH_FAILURE"
        )
    )
    result = {
        "schema": "bayesfilter.neutra.retained_from_warmup_resume_result.v1",
        "cell_id": spec.cell_id,
        "target_signature": spec.target_signature,
        "passed": bool(passed and truth_tail["status"] == "PASS"),
        "decision": decision,
        "admitted_kernel_replay": {
            "path": str(args.admitted_kernel_replay),
            "mechanics_sha256": mechanics.get("mechanics_sha256"),
        },
        "warmup": {
            "source_root": str(args.warmup_root),
            "results_per_chain": 2000,
            "warmup_excluded_from_posterior": True,
            "chunks": tuple(warmup_rows),
            "modern_rhat": warmup_rhat,
            "passed": True,
        },
        "retained": {
            "seed": retained_seed,
            "results_per_chain": 1000,
            "health": health,
            "convergence": convergence,
            "archive": retained_archive,
            "passed": passed,
        },
        "truth_tail": truth_tail,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": (
            "one second-seed truth-tail diagnostic only",
            "no sampler superiority or default-readiness claim",
        ),
    }
    atomic_write_json(args.output_root / "result.json", result)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    atomic_write_json(
        args.output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.neutra.retained_from_warmup_resume_manifest.v1",
            "git_commit": commit,
            "command": tuple(sys.argv),
            "gpu_memory_policy": memory_policy,
            "tensorflow_version": tf.__version__,
            "jit_compile": True,
            "tf32_execution_enabled": True,
            "warmup_source_root": str(args.warmup_root),
            "admitted_kernel_replay": str(args.admitted_kernel_replay),
            "frozen_transport": str(args.frozen_transport),
            "frozen_transport_sha256": str(args.frozen_transport_sha256).lower(),
            "retained_seed": retained_seed,
            "result_path": str(args.output_root / "result.json"),
            "wall_time_seconds": time.monotonic() - started,
            "tuning_invoked": False,
            "training_invoked": False,
        },
    )
    print(json.dumps({"passed": result["passed"], "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
