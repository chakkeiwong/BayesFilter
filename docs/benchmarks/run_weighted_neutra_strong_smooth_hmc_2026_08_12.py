#!/usr/bin/env python3
"""Run corrected fixed-length HMC behind a frozen varying-Hessian transport."""

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

CHAIN_COUNT = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--cap-seconds", type=float, default=5400.0)
    parser.add_argument("--target-name", choices=("nk_like_mild_smooth", "nk_like_strong_smooth"), default="nk_like_strong_smooth")
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
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


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _load_runtime(
    tf: Any,
    target_name: str,
    constants_path: Path,
    training_root: Path,
    plan_path: Path,
) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_varying_hessian_target import (
        FrozenAffineLiftWeightedTransport,
        VaryingHessianValueScoreAdapter,
        load_varying_hessian_target_spec,
    )
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected exactly one visible logical GPU, found {logical_gpus}")
    trainer_state_path = training_root / "trainer_state.json"
    hashes_path = training_root / "artifact_hashes.json"
    training_manifest_path = training_root / "run_manifest.json"
    state = _load_json(trainer_state_path)
    hashes = _load_json(hashes_path)
    training_manifest = _load_json(training_manifest_path)
    artifact_hashes = hashes.get("artifacts", {})
    if artifact_hashes.get(trainer_state_path.name) != _sha256(trainer_state_path):
        raise RuntimeError("training state artifact hash mismatch")
    if artifact_hashes.get(training_manifest_path.name) != _sha256(training_manifest_path):
        raise RuntimeError("training manifest artifact hash mismatch")
    if state.get("schema") != "bayesfilter.weighted_neutra_strong_smooth_local_state.v1":
        raise RuntimeError("training state schema mismatch")
    config_payload = dict(state.get("config", {}))
    config_payload.pop("schema", None)
    config_payload["hidden_layers"] = tuple(config_payload["hidden_layers"])
    config_payload["initialization_seed"] = tuple(config_payload["initialization_seed"])
    config = WeightedNeuTraConfig(**config_payload)
    transport_local = WeightedDenseIAFTransport(config)
    variables = state.get("variables")
    if not isinstance(variables, list) or len(variables) != len(transport_local.trainable_variables):
        raise RuntimeError("training state variable count mismatch")
    for variable, raw in zip(transport_local.trainable_variables, variables):
        tensor = tf.convert_to_tensor(raw, tf.float64)
        if tensor.shape != variable.shape:
            raise RuntimeError("training state variable shape mismatch")
        tf.debugging.assert_all_finite(tensor, "training state variable")
        variable.assign(tensor)
    state_hash_payload = {
        "config": state["config"],
        "selected_update": state["selected_update"],
        "variables": state["variables"],
    }
    if state.get("state_hash") != _stable_hash(state_hash_payload):
        raise RuntimeError("training state semantic hash mismatch")
    tensor_hash = _stable_hash([variable.read_value() for variable in transport_local.trainable_variables])
    transport_local.bind_frozen_identity(
        {
            "checkpoint_sha256": _sha256(trainer_state_path),
            "training_state_hash": str(state["state_hash"]),
            "transport_tensor_hash": tensor_hash,
        }
    )
    spec = load_varying_hessian_target_spec(constants_path, expected_name=target_name)
    training_target = training_manifest.get("target")
    if not isinstance(training_target, Mapping):
        raise RuntimeError("training manifest target is missing")
    if training_target.get("name") != target_name:
        raise RuntimeError("training checkpoint target name mismatch")
    if training_target.get("constants_sha256") != spec.constants_sha256:
        raise RuntimeError("training checkpoint constants SHA-256 mismatch")
    transport = FrozenAffineLiftWeightedTransport(spec, transport_local)
    base = VaryingHessianValueScoreAdapter(spec)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"weighted_neutra_varying_hessian:{target_name}:hmc_v1",
        runtime_backend="tensorflow_source_bound_varying_hessian_affine_weighted_iaf_hmc",
        evidence_path=plan_path.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "source-bound smooth target only",
            "no normalized posterior authority is available for this target",
            "no posterior-reference or predictive-equivalence claim",
        ),
    )
    local_starts = tf.constant(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        tf.float64,
    )
    initial = transport_local.inverse_and_forward_logdet(local_starts)[0]
    return tf, transport_local, base, adapter, {
        "memory_policy": _ready(memory_policy),
        "logical_gpu": str(logical_gpus[0]),
        "spec": spec,
        "transport": transport,
        "initial_state": initial,
        "config": config,
        "state_hash": state["state_hash"],
        "training_state_path": trainer_state_path,
        "training_state_sha256": _sha256(trainer_state_path),
        "training_manifest_path": training_manifest_path,
        "training_manifest_sha256": _sha256(training_manifest_path),
    }


def _run_tuning(
    base: Any,
    initial: Any,
    output: Path,
    target_name: str,
) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.10,
        leapfrog_grid=(3, 5, 10, 15, 20, 25),
        chain_count=CHAIN_COUNT,
        initial_state_bank=tuple(tuple(float(v) for v in row) for row in initial.numpy().tolist()),
        target_accept_prob=0.70,
        acceptance_band=(0.55, 0.90),
        repair_band=(0.40, 0.95),
        selection_policy="acceptance_target_distance",
        selection_replications=1,
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
        tune_seed_base=(20260813, 17001),
        screen_seed_base=(20260813, 18001),
        verification_seed_base=(20260813, 19001),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=f"weighted_neutra_varying_hessian:{target_name}:tuning_v1",
        tuning_policy=FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        output_filename="tuning_result.json",
    )
    return tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=RUNTIME_TRANSPORT,
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
    target_name: str,
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc

    kernel = tuning.get("final_kernel_payload")
    if not isinstance(kernel, Mapping) or tuning.get("passed") is not True:
        raise RuntimeError("HMC tuning did not produce a viable fixed kernel")
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
        seed=(20260813, 20001),
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
        archive_label=f"weighted-{target_name.removeprefix('nk_like_').replace('_', '-')}",
        budget_check=lambda _transitions: time.perf_counter() - started < cap,
    )
    payload = {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **result.__dict__}
    _write(output / "sequential_result.json", payload)
    return payload


RUNTIME_TRANSPORT: Any = None


def main() -> int:
    args = _parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    if not float(args.cap_seconds) > 0.0:
        raise ValueError("cap_seconds must be positive")
    plan_path = args.plan.resolve()
    constants_path = args.constants.resolve()
    training_root = args.training_root.resolve()
    required_inputs = (
        plan_path,
        constants_path,
        training_root / "trainer_state.json",
        training_root / "artifact_hashes.json",
        training_root / "run_manifest.json",
    )
    if any(not path.is_file() for path in required_inputs):
        raise FileNotFoundError("varying-Hessian HMC inputs are missing")
    args.output_root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    global RUNTIME_TRANSPORT
    tf, _local, base, adapter, runtime = _load_runtime(
        __import__("tensorflow"),
        args.target_name,
        constants_path,
        training_root,
        plan_path,
    )
    RUNTIME_TRANSPORT = runtime["transport"]
    manifest = {
        "schema": "bayesfilter.weighted_neutra_varying_hessian_hmc_manifest.v1",
        "plan": plan_path.as_posix(),
        "training_root": training_root.as_posix(),
        "training_state": runtime["training_state_path"],
        "training_state_sha256": runtime["training_state_sha256"],
        "training_state_hash": runtime["state_hash"],
        "training_manifest": runtime["training_manifest_path"],
        "training_manifest_sha256": runtime["training_manifest_sha256"],
        "target": runtime["spec"].manifest_payload(),
        "adapter_signature": adapter.adapter_signature(),
        "transport_manifest": runtime["transport"].manifest_payload(),
        "memory_policy": runtime["memory_policy"],
        "gpu": runtime["logical_gpu"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "initial_state": runtime["initial_state"],
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    _write(args.output_root / "run_manifest.json", manifest)
    tuning = _run_tuning(base, runtime["initial_state"], args.output_root, args.target_name)
    if tuning.get("passed") is not True:
        _write(args.output_root / "result.json", {"schema": "bayesfilter.weighted_neutra_varying_hessian_hmc_result.v1", "manifest": manifest, "tuning": tuning, "decision": {"status": "hmc_candidate_rejected_at_tuning", "promotion": False, "repair_trigger": "no_viable_fixed_kernel", "nonclaims": ["no posterior correctness claim"]}})
        return 0
    sequential = _run_sequential(
        adapter,
        runtime["initial_state"],
        tuning,
        args.output_root,
        min(float(args.cap_seconds), 3600.0),
        args.target_name,
    )
    passed = bool(sequential.get("passed"))
    _write(args.output_root / "result.json", {
        "schema": "bayesfilter.weighted_neutra_varying_hessian_hmc_result.v1",
        "manifest": manifest,
        "tuning": tuning,
        "sequential": sequential,
        "decision": {
            "status": "candidate_sampler_evidence_passed" if passed else "candidate_sampler_evidence_rejected",
            "promotion": False,
            "primary_criterion": "canonical sequential R-hat/ESS and numerical status screens",
            "posterior_reference": "not_available_for_this_unnormalized_source_bound_target",
            "nonclaims": ["no posterior correctness claim", "no default promotion", "no ranking"],
        },
        "wall_seconds": time.perf_counter() - started,
    })
    _write(args.output_root / "artifact_hashes.json", {
        "schema": "bayesfilter.weighted_neutra_varying_hessian_hmc_hashes.v1",
        "artifacts": {path.name: _sha256(path) for path in args.output_root.iterdir() if path.is_file() and path.name != "artifact_hashes.json"},
    })
    print(json.dumps({"passed": passed, "output_root": args.output_root.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
