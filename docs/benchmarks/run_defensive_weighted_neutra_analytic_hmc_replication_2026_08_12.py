#!/usr/bin/env python3
"""Run one frozen-kernel weighted-NeuTra analytic HMC replication."""

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

PLAN = ROOT / "docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-replication-plan-2026-08-12.md"
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/"
    "r1-two-mode/capacity-depth6-width128-updates10000-confirmation-1-v1/"
    "trainer_states.json"
)
TUNING = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5/"
    "tuning/tuning_result.json"
)
V5_MANIFEST = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5/"
    "run_manifest.json"
)
EXPECTED_CHECKPOINT_SHA256 = "af961871dcc3b626216d7500e695534f147ecfd9ba4fe0f9907f59018d40e8e5"
EXPECTED_TUNING_SHA256 = "6dfe2b8145040a18831a08032bfd61854189f2651e76c70842e59d4e4e12eb4f"
EXPECTED_V5_MANIFEST_SHA256 = "172613cd7a979edc1e324d538890a70c76ec97e54f7cf382c69fd5b68cc3dcfb"
EXPECTED_STEP_SIZE = 0.14091138276334744
EXPECTED_LEAPFROG = 20
V5_BASE_ADAPTER_SIGNATURE = "2253ee2bf30b54269201a898d661fb462e12805af8040fa9b6049d51a534b8e0"
CPU_MIGRATED_BASE_ADAPTER_SIGNATURE = "f99e3c7e19373f595322e927e31c8dd3ae07ad2ae8d02c778b836c58dc2f4a6b"
EXPECTED_TRANSPORT_MANIFEST_HASH = "e32d8e4c4762858baf6b2a13ab1e860c7308a4fa152e549890a3b809b5e8b938"
V5_LIVE_TRANSFORMED_ADAPTER_SIGNATURE = "6b4e5c287309163c105fd803ca9d3f97ad0ff25c991dad15a57d1d4b29d1a6ba"
V5_TUNING_TRANSFORMED_ADAPTER_SIGNATURE = "7d188d2373215877cd393ec10f0df83d0e2c19f15484957420bf6e2785317f46"
CPU_MIGRATED_TRANSFORMED_ADAPTER_SIGNATURE = "39ca3400712732e3b0427532e9c4bf23cc308b0f2fdc5fc45ccd7a4a5c22974d"
CHAIN_COUNT = 4


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
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "run"), default="run")
    parser.add_argument("--replication", type=int, choices=range(4), required=True)
    parser.add_argument("--seed-second", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--cap-seconds", type=float, default=3600.0)
    return parser.parse_args()


def _validate_frozen_inputs() -> Mapping[str, Any]:
    if _sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    if _sha256(TUNING) != EXPECTED_TUNING_SHA256:
        raise RuntimeError("tuning artifact SHA-256 mismatch")
    payload = json.loads(TUNING.read_text(encoding="utf-8"))
    kernel = payload.get("final_kernel_payload")
    if payload.get("passed") is not True or not isinstance(kernel, Mapping):
        raise RuntimeError("frozen tuning artifact is not admitted")
    if int(kernel.get("num_leapfrog_steps", -1)) != EXPECTED_LEAPFROG:
        raise RuntimeError("frozen leapfrog count differs from reviewed kernel")
    if float(kernel.get("step_size", -1.0)) != EXPECTED_STEP_SIZE:
        raise RuntimeError("frozen step size differs from reviewed kernel")
    if kernel.get("mass_policy") != "fixed_identity_z":
        raise RuntimeError("frozen mass policy differs from reviewed kernel")
    if kernel.get("transformed_adapter_signature") != V5_TUNING_TRANSFORMED_ADAPTER_SIGNATURE:
        raise RuntimeError("tuner-internal transformed adapter identity mismatch")
    return {
        "path": TUNING.as_posix(),
        "sha256": EXPECTED_TUNING_SHA256,
        "kernel_sha256": hashlib.sha256(
            json.dumps(_ready(kernel), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "kernel": kernel,
    }


def _target_signature_compatibility(tf: Any, current: Mapping[str, Any]) -> Mapping[str, Any]:
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob_responsibilities_score,
    )

    if _sha256(V5_MANIFEST) != EXPECTED_V5_MANIFEST_SHA256:
        raise RuntimeError("immutable v5 manifest SHA-256 mismatch")
    old = json.loads(V5_MANIFEST.read_text(encoding="utf-8"))
    if old.get("base_adapter_signature") != V5_BASE_ADAPTER_SIGNATURE:
        raise RuntimeError("v5 base adapter identity mismatch")
    target = old.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError("v5 target payload is missing")
    current_payload = _ready(current["signature_payload"])
    for key in ("schema", "identity", "dtype", "probabilities", "means"):
        if current_payload[key] != target[key]:
            raise RuntimeError(f"current target differs from v5 in {key}")
    old_covariance = target["covariances"]
    current_covariance = current_payload["covariances"]
    differences = []
    for i in range(2):
        for j in range(4):
            for k in range(4):
                before = float(old_covariance[i][j][k])
                after = float(current_covariance[i][j][k])
                delta = abs(before - after)
                if delta > max(math.ulp(before), math.ulp(after)):
                    raise RuntimeError("current covariance differs from v5 by more than one ULP")
                if delta:
                    differences.append({"index": [i, j, k], "v5": before, "current": after, "absolute_delta": delta})
    old_probabilities = tf.constant(target["probabilities"], tf.float64)
    old_means = tf.constant(target["means"], tf.float64)
    old_covariances = tf.constant(target["covariances"], tf.float64)
    deterministic = tf.reshape(
        tf.linspace(tf.constant(-12.0, tf.float64), tf.constant(12.0, tf.float64), 4096 * 4),
        (4096, 4),
    )
    points = tf.concat(
        (
            old_means,
            tf.zeros((1, 4), tf.float64),
            tf.ones((1, 4), tf.float64),
            -tf.ones((1, 4), tf.float64),
            tf.constant(((1.0, -2.0, 3.0, -4.0),), tf.float64),
            10.0 * tf.eye(4, dtype=tf.float64),
            -10.0 * tf.eye(4, dtype=tf.float64),
            deterministic,
        ),
        axis=0,
    )
    old_outputs = gaussian_mixture_log_prob_responsibilities_score(
        points, old_probabilities, old_means, old_covariances
    )
    current_outputs = gaussian_mixture_log_prob_responsibilities_score(
        points, current["probabilities"], current["means"], current["covariances"]
    )
    labels = ("log_prob", "responsibilities", "score")
    maximum_deltas = {
        label: float(tf.reduce_max(tf.abs(before - after)).numpy())
        for label, before, after in zip(labels, old_outputs, current_outputs)
    }
    if any(delta != 0.0 for delta in maximum_deltas.values()):
        raise RuntimeError("current target value/score differs from v5 on compatibility bank")
    return {
        "schema": "bayesfilter.defensive_weighted_neutra_target_signature_compatibility.v1",
        "v5_manifest_path": V5_MANIFEST.as_posix(),
        "v5_manifest_sha256": EXPECTED_V5_MANIFEST_SHA256,
        "v5_base_adapter_signature": V5_BASE_ADAPTER_SIGNATURE,
        "v5_live_transformed_adapter_signature": V5_LIVE_TRANSFORMED_ADAPTER_SIGNATURE,
        "v5_tuning_transformed_adapter_signature": V5_TUNING_TRANSFORMED_ADAPTER_SIGNATURE,
        "current_target_signature": current["target_signature"],
        "covariance_differences": differences,
        "maximum_covariance_absolute_delta": max((row["absolute_delta"] for row in differences), default=0.0),
        "comparison_point_count": int(points.shape[0]),
        "maximum_output_absolute_deltas": maximum_deltas,
        "passed": True,
        "interpretation": "one_ulp_serialization_migration_same_checked_value_score_program",
    }


def _build_runtime() -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import (
        AnalyticGaussianMixtureValueScoreAdapter,
        analytic_two_mode_target,
        load_weighted_neutra_transport,
        mode_aware_initial_state,
    )

    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    devices = tuple(tf.config.list_logical_devices("GPU"))
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible GPU, found {devices}")
    target = analytic_two_mode_target()
    compatibility = _target_signature_compatibility(tf, target)
    loaded = load_weighted_neutra_transport(CHECKPOINT)
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
            "frozen kernel root-seed replication only",
            "analytic target only",
            "no SSL-LSTM or general NeuTra claim",
        ),
    )
    signature_pair = (base.adapter_signature(), adapter.adapter_signature())
    allowed_signature_pairs = {
        (
            V5_BASE_ADAPTER_SIGNATURE,
            V5_LIVE_TRANSFORMED_ADAPTER_SIGNATURE,
        ): "exact_v5_gpu_pair",
        (
            CPU_MIGRATED_BASE_ADAPTER_SIGNATURE,
            CPU_MIGRATED_TRANSFORMED_ADAPTER_SIGNATURE,
        ): "one_ulp_cpu_reference_pair",
    }
    if signature_pair not in allowed_signature_pairs:
        raise RuntimeError("live adapter signatures are not a reviewed hardware pair")
    if adapter.transport_manifest_hash != EXPECTED_TRANSPORT_MANIFEST_HASH:
        raise RuntimeError("live transport manifest differs from frozen tuning authority")
    initial = mode_aware_initial_state(loaded.transport, target)
    return tf, loaded, adapter, {
        "target": target,
        "initial_state": initial,
        "memory_policy": _ready(policy),
        "target_signature_compatibility": compatibility,
        "adapter_signature_pair_status": allowed_signature_pairs[signature_pair],
    }


def _load_retained_samples(tf: Any, sequential: Mapping[str, Any]) -> Any:
    manifest = json.loads(
        Path(sequential["archive"]["root"], "weighted-analytic-manifest.json").read_text()
    )
    tensors = []
    for row in manifest.get("retained_chunks", ()):
        receipt = row["sample_receipt"]
        path = Path(receipt["path"])
        if _sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"retained receipt mismatch: {path}")
        tensors.append(tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64))
    if not tensors:
        raise RuntimeError("no retained chunks")
    return tf.concat(tensors, axis=0)


def _mode_transition_diagnostics(tf: Any, physical_samples: Any) -> Mapping[str, Any]:
    from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import analytic_two_mode_target
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob_responsibilities_score,
    )

    samples = tf.convert_to_tensor(physical_samples, tf.float64)
    target = analytic_two_mode_target()
    _, responsibilities, _ = gaussian_mixture_log_prob_responsibilities_score(
        tf.reshape(samples, (-1, 4)),
        target["probabilities"],
        target["means"],
        target["covariances"],
    )
    assignments = tf.reshape(
        tf.argmax(responsibilities, axis=1, output_type=tf.int32),
        tf.shape(samples)[:2],
    )
    transitions = tf.reduce_sum(
        tf.cast(assignments[1:] != assignments[:-1], tf.int32), axis=0
    )
    visits = tf.stack(
        [tf.reduce_sum(tf.cast(assignments == mode, tf.int32), axis=0) for mode in (0, 1)],
        axis=1,
    )
    return {
        "schema": "bayesfilter.defensive_weighted_neutra_mode_transitions.v1",
        "hard_assignment_transition_count_by_chain": transitions,
        "hard_assignment_visit_count_by_chain_and_mode": visits,
        "all_chains_transitioned": bool(tf.reduce_all(transitions > 0).numpy()),
        "role": "explanatory_only_not_convergence_or_equality_evidence",
    }


def main() -> int:
    args = _parse_args()
    if args.seed_second != 91011 + args.replication:
        raise ValueError("seed-second must be the predeclared replication seed")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    output.mkdir(parents=True)
    frozen = _validate_frozen_inputs()
    started = time.perf_counter()
    manifest = {
        "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_replication_manifest.v1",
        "command": " ".join(sys.argv),
        "environment": "tfgpu",
        "python": sys.version,
        "plan": PLAN.as_posix(),
        "replication": args.replication,
        "root_seed": [20260812, args.seed_second],
        "checkpoint": {"path": CHECKPOINT.as_posix(), "sha256": _sha256(CHECKPOINT)},
        "tuning": {key: frozen[key] for key in ("path", "sha256", "kernel_sha256")},
        "frozen_kernel": {
            "num_leapfrog_steps": EXPECTED_LEAPFROG,
            "step_size": EXPECTED_STEP_SIZE,
            "mass_policy": "fixed_identity_z",
        },
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        "git": _git_manifest(),
    }
    _write(
        output / "run_manifest.json",
        manifest,
    )
    tf, loaded, adapter, runtime = _build_runtime()
    manifest.update(
        tensorflow=tf.__version__,
        tensorflow_probability=__import__("tensorflow_probability").__version__,
        logical_gpu_devices=[str(device) for device in tf.config.list_logical_devices("GPU")],
        memory_policy=runtime["memory_policy"],
        checkpoint_manifest=loaded.manifest_payload(),
        base_adapter_signature=adapter.base_adapter.adapter_signature(),
        transformed_adapter_signature=adapter.adapter_signature(),
        adapter_signature_pair_status=runtime["adapter_signature_pair_status"],
        transport_manifest_hash=adapter.transport_manifest_hash,
        trust_basis="owner_designated_managed_session_visible_gpu_trusted",
        route_ledger_status="preexisting_referenced_ledger_artifact_absent_not_used_for_scientific_adjudication",
        target_signature_compatibility=runtime["target_signature_compatibility"],
    )
    _write(output / "run_manifest.json", manifest)
    if runtime["adapter_signature_pair_status"] != "exact_v5_gpu_pair":
        raise RuntimeError("claim-bearing GPU route must reproduce exact v5 signatures")
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
                num_results=32,
                num_burnin_steps=32,
                step_size=EXPECTED_STEP_SIZE,
                num_leapfrog_steps=EXPECTED_LEAPFROG,
                seed=(20260812, 97001),
                use_xla=True,
                trace_policy="full",
                target_status_trace_policy="per_chain_step",
                tuning_policy=FixedTransportHMCPolicy.fixed(source=PLAN.as_posix()),
                target_scope="defensive_weighted_neutra_analytic_hmc:replication_canary_v1",
                chain_execution_mode="tf_function",
            ),
        )
        diagnostics = canary.diagnostics
        chain_moved = tf.reduce_any(
            tf.not_equal(canary.samples, runtime["initial_state"][tf.newaxis, ...]),
            axis=(0, 2),
        )
        passed = bool(
            diagnostics.get("samples_all_finite", False)
            and diagnostics.get("target_log_prob_finite", False)
            and diagnostics.get("target_score_finite", False)
            and diagnostics.get("target_status_telemetry", {}).get("all_status_valid", False)
            and bool(tf.reduce_all(chain_moved).numpy())
            and canary.metadata.get("use_xla") is True
        )
        _write(
            output / "canary.json",
            {
                "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_replication_canary.v1",
                "passed": passed,
                "samples_shape": tuple(int(value) for value in canary.samples.shape),
                "diagnostics": diagnostics,
                "metadata": canary.metadata,
                "chain_moved": chain_moved,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        if not passed:
            raise RuntimeError("current-contract GPU/XLA canary failed")
        print(json.dumps({"mode": "canary", "passed": True, "output_root": output.as_posix()}))
        return 0
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc
    from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import retained_analytic_diagnostics

    config = SequentialNeuTraHMCConfig(
        step_size=EXPECTED_STEP_SIZE,
        num_leapfrog_steps=EXPECTED_LEAPFROG,
        seed=(20260812, args.seed_second),
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
    sequential = run_sequential_neutra_hmc(
        adapter,
        runtime["initial_state"],
        config,
        archive_root=output / "archive",
        archive_label="weighted-analytic",
        budget_check=lambda _transitions: time.perf_counter() - started < args.cap_seconds,
    )
    sequential_payload = {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **sequential.__dict__}
    _write(output / "sequential_result.json", sequential_payload)
    retained_latent = _load_retained_samples(tf, sequential_payload)
    retained_physical = tf.reshape(
        loaded.transport.forward_batch(tf.reshape(retained_latent, (-1, 4))),
        tf.shape(retained_latent),
    )
    analytic = retained_analytic_diagnostics(retained_physical)
    mode_transitions = _mode_transition_diagnostics(tf, retained_physical)
    gates = analytic["gates"]
    primary_names = (
        "all_finite",
        "minority_mass_99pct_interval_contains_truth",
        "both_modes_observed_overall",
        "both_hard_modes_observed_per_chain",
    )
    analytic_primary_passed = all(bool(gates[name]) for name in primary_names)
    final_passed = bool(sequential_payload["passed"]) and analytic_primary_passed
    _write(
        output / "result.json",
        {
            "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_replication_result.v1",
            "replication": args.replication,
            "root_seed": [20260812, args.seed_second],
            "frozen_inputs": frozen,
            "sequential": sequential_payload,
            "retained_analytic_diagnostics": analytic,
            "mode_transition_diagnostics": mode_transitions,
            "adjudication": {
                "status": "passed_frozen_seed_replication" if final_passed else "rejected_frozen_seed_replication",
                "sequential_passed": bool(sequential_payload["passed"]),
                "analytic_primary_passed": analytic_primary_passed,
                "analytic_primary_gate_names": primary_names,
                "marginal_mean_covariance_role": "descriptive_only_not_joint_veto",
                "nonclaims": [
                    "no stationarity or distributional equality proof",
                    "no seed ranking or sampler superiority claim",
                    "no cross-transport or SSL-LSTM claim",
                ],
            },
            "wall_seconds": time.perf_counter() - started,
            "git": _git_manifest(),
        },
    )
    result_path = output / "result.json"
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.defensive_weighted_neutra_replication_hashes.v1", "artifacts": {"result.json": _sha256(result_path), "run_manifest.json": _sha256(output / "run_manifest.json"), "sequential_result.json": _sha256(output / "sequential_result.json")}})
    print(json.dumps({"replication": args.replication, "passed": final_passed, "wall_seconds": time.perf_counter() - started, "output_root": output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
