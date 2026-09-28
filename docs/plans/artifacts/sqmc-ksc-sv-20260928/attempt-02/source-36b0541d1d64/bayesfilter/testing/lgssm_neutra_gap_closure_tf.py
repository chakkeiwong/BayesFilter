"""TensorFlow-only LGSSM NeuTra scientific gap-closure campaign helpers.

This module deliberately does not import the legacy generic HMC stack.  It
provides the narrow batched TensorFlow/TFP mechanics, artifact binding, tuning
selection, tensor serialization, and posterior summaries required by the
2026-07-15 LGSSM NeuTra gap-closure plan.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import (
    FixedTransportValueScoreAdapter,
)
from bayesfilter.inference.hmc_convergence import (
    RankNormalizedHMCThresholds,
    rank_normalized_hmc_diagnostics,
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference import neutra_hmc as _shared_neutra_hmc
from bayesfilter.inference.neutra_hmc_policy import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md"
)
ARTIFACT_ROOT = ROOT / "docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15"
SELECTION_PATH = ROOT / (
    "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/"
    "phase7/screen-500/selected_recipe.json"
)
COMPARATOR_PATH = ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/"
    "final_recovery_result.json"
)
EXPECTED_SELECTION_FILE_SHA256 = (
    "1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97"
)
EXPECTED_COMPARATOR_FILE_SHA256 = (
    "bcc6e71a1067dc648758a5aac9c87ef7e94fdd4b1ac53d5601ef4e9fdf6741b5"
)
EXPECTED_TARGET_SIGNATURE = (
    "f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30"
)
EXPECTED_ADAPTER_SIGNATURE = (
    "42dc7bad0137fd9c31aa1d618bb4e560f68d1bbe3a7ab4f5ef95e458b2abc985"
)
EXPECTED_RECIPE_ID = "wide_2x_lr5e3"
SCREEN_WIDE_RESULT_PATH = ROOT / (
    "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/"
    "screen-500/screen/candidates/wide_2x_lr5e3/attempt_1_graph_native/result.json"
)
FRESH_CANDIDATE_SOURCES = {
    "dense_seed1201": {
        "record_path": ARTIFACT_ROOT / "phase1" / "seed1201_post_validation.json",
        "record_schema": "bayesfilter.lgssm_neutra_completed_training_post_validation.v1",
        "payload_file_sha256": (
            "6429977ba1754ce5f36248104c82fa18639311a0727298bc3ed436b4a670a745"
        ),
    },
    "dense_seed1202": {
        "record_path": ROOT / (
            "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/"
            "long-training-attempt-01/phase4/training_jobs/dense_seed1202/"
            "attempt_1_graph_native/result.json"
        ),
        "record_schema": "bayesfilter.lgssm_neutra_strict_training_job.v1",
        "payload_file_sha256": (
            "92e5ca376fd9660be138e8badc2ff871deff09ca97784ad247add54692352e31"
        ),
    },
}
DIMENSION = 18
CHAIN_COUNT = 4
FINAL_STEPS = 5000
RHAT_MAX = 1.01
BULK_ESS_MIN = 1000.0
TAIL_ESS_MIN = 400.0
POSTERIOR_AGREEMENT_MAX_Z = 4.0
RECOVERY_MAX_Z = 3.0
ENERGY_ERROR_DIVERGENCE_LOG_ACCEPT_THRESHOLD = -1000.0
TUNING_PRIMARY_STEPS = (0.025, 0.05, 0.1, 0.2, 0.4)
TUNING_REPAIR_STEPS = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
TUNING_LEAPFROG_STEPS = 10
TUNING_ACCEPTANCE_BAND = (0.60, 0.90)
SEQUENTIAL_REPAIR_ROOT = ARTIFACT_ROOT / "sequential-repair-attempt-01"
CONFIRMATION_ROOT = SEQUENTIAL_REPAIR_ROOT / "confirmation-attempt-01"
SEQUENTIAL_REPAIR_SEEDS = {
    "dense_seed1201": {
        "warmup": (20260715, 4101),
        "retained": (20260715, 4201),
    },
    "dense_seed1202": {
        "warmup": (20260715, 4301),
        "retained": (20260715, 4401),
    },
}

NONCLAIMS = (
    "one favorably truth-centered 18D LGSSM fixture only",
    "plain-HMC agreement is not exact-posterior correctness evidence",
    "training loss and acceptance alone are explanatory only",
    "no sampler superiority, calibration, robustness, or generalization claim",
    "no production or default-readiness claim",
)


class LGSSMNeuTraGapClosureError(RuntimeError):
    """Raised when a gap-closure identity or evidence gate fails closed."""


def _archive_sequential_chunk(
    *,
    root: Path,
    stage: str,
    chunk_index: int,
    z_samples: tf.Tensor,
    raw_samples: tf.Tensor,
    seed: tuple[int, int],
) -> Mapping[str, Any]:
    prefix = root / stage / f"chunk_{chunk_index + 1:04d}"
    metadata = {"stage": stage, "chunk_index": chunk_index, "seed": seed}
    return {
        "z": write_tensor_archive(
            prefix.with_name(prefix.name + "_z.tftensor"),
            z_samples,
            metadata={**metadata, "coordinate_system": "latent_z"},
        ),
        "raw": write_tensor_archive(
            prefix.with_name(prefix.name + "_raw.tftensor"),
            raw_samples,
            metadata={**metadata, "coordinate_system": "raw_parameters"},
        ),
    }


# These compatibility bindings keep historical campaign imports stable while
# making the shared controller the sole active implementation.
TensorHMCConfig = _shared_neutra_hmc.TensorHMCConfig
SequentialNeuTraHMCConfig = _shared_neutra_hmc.SequentialNeuTraHMCConfig


def _lgssm_target_status_summary(target_status: Any) -> Mapping[str, Any]:
    status_code = tf.convert_to_tensor(target_status["status_code"], tf.int32)
    valid_score = tf.convert_to_tensor(
        target_status["valid_pre_regularized_score"], tf.bool
    )
    invalid = tf.logical_or(tf.not_equal(status_code, 0), tf.logical_not(valid_score))
    return {
        "available": True,
        "all_status_valid": bool(tf.reduce_all(tf.logical_not(invalid)).numpy()),
        "status_nonvalid_count": int(
            tf.reduce_sum(tf.cast(invalid, tf.int32)).numpy()
        ),
        "floor_count_total": int(
            tf.reduce_sum(tf.cast(target_status["floor_count_value"], tf.int64)).numpy()
        ),
        "min_innovation_eigenvalue": float(
            tf.reduce_min(target_status["min_innovation_eigenvalue"]).numpy()
        ),
        "max_innovation_condition_estimate": float(
            tf.reduce_max(target_status["innovation_condition_estimate"]).numpy()
        ),
        "trace_scope": "sampled_transition_states",
    }


def run_batched_hmc(
    *,
    adapter: Any,
    initial_state: Any,
    config: TensorHMCConfig,
) -> Mapping[str, Any]:
    """Compatibility wrapper around the canonical shared batched controller."""

    telemetry = getattr(adapter, "target_status_telemetry", None)
    return _shared_neutra_hmc.run_batched_hmc(
        adapter=adapter,
        initial_state=initial_state,
        config=config,
        target_status_summary_fn=(
            _lgssm_target_status_summary if callable(telemetry) else None
        ),
    )


def run_sequential_neutra_hmc(
    *,
    adapter: Any,
    initial_state: Any,
    raw_transform: Callable[[tf.Tensor], Any],
    parameter_names: Sequence[str],
    config: SequentialNeuTraHMCConfig,
    archive_root: str | Path | None = None,
    retained_diagnostic_fn: Callable[[tf.Tensor], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Preserve the LGSSM archive schema while delegating sampling control."""

    root = None if archive_root is None else Path(archive_root).resolve()
    if root is not None and root.exists():
        raise FileExistsError(f"sequential HMC archive root already exists: {root}")

    def archive_callback(
        *,
        stage: str,
        chunk_index: int | None,
        latent_samples: tf.Tensor,
        model_samples: tf.Tensor,
        seed: tuple[int, int] | None,
        cumulative: bool,
    ) -> Mapping[str, Any]:
        if cumulative:
            assert root is not None
            return {
                "z": write_tensor_archive(
                    root / stage / "all_z.tftensor",
                    latent_samples,
                    metadata={"stage": stage, "coordinate_system": "latent_z"},
                ),
                "raw": write_tensor_archive(
                    root / stage / "all_raw.tftensor",
                    model_samples,
                    metadata={
                        "stage": stage,
                        "coordinate_system": "raw_parameters",
                    },
                ),
            }
        assert root is not None and chunk_index is not None and seed is not None
        return _archive_sequential_chunk(
            root=root,
            stage=stage,
            chunk_index=chunk_index,
            z_samples=latent_samples,
            raw_samples=model_samples,
            seed=seed,
        )

    telemetry = getattr(adapter, "target_status_telemetry", None)
    result = dict(
        _shared_neutra_hmc.run_sequential_neutra_hmc(
            adapter=adapter,
            initial_state=initial_state,
            model_transform=raw_transform,
            parameter_names=parameter_names,
            config=config,
            retained_diagnostic_fn=retained_diagnostic_fn,
            archive_callback=archive_callback if root is not None else None,
            target_status_summary_fn=(
                _lgssm_target_status_summary if callable(telemetry) else None
            ),
        )
    )
    cumulative = result.get("cumulative_archives")
    if isinstance(cumulative, Mapping):
        flattened = {
            "warmup_z": cumulative["warmup"]["z"],
            "warmup_raw": cumulative["warmup"]["raw"],
        }
        if "retained" in cumulative:
            flattened.update(
                {
                    "retained_z": cumulative["retained"]["z"],
                    "retained_raw": cumulative["retained"]["raw"],
                }
            )
        result["cumulative_archives"] = flattened
    return result


def run_gaussian_phase0_diagnostic(
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run a small CPU-hidden XLA mechanics and diagnostic gate."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraGapClosureError(
            "Gaussian Phase 0 diagnostic requires CUDA_VISIBLE_DEVICES=-1"
        )

    class GaussianAdapter:
        parameter_dim = 2

        @staticmethod
        def log_prob_and_grad(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
            values = tf.convert_to_tensor(theta, tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    config = TensorHMCConfig(
        num_results=128,
        num_burnin_steps=128,
        step_size=0.55,
        num_leapfrog_steps=4,
        seed=(20260715, 1),
    )
    run = run_batched_hmc(
        adapter=GaussianAdapter(),
        initial_state=tf.constant(
            [[-1.5, 0.5], [-0.5, -1.0], [0.5, 1.0], [1.5, -0.5]],
            tf.float64,
        ),
        config=config,
    )
    rhat = rank_normalized_split_rhat_summary(run["samples"], rhat_max=RHAT_MAX)
    passed = bool(run["diagnostics"]["health_passed"] and rhat["passed"])
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_gap_closure_phase0_gaussian.v1",
            "passed": passed,
            "decision": (
                "PASS_PHASE0_GAUSSIAN_XLA_HMC"
                if passed
                else "BLOCK_PHASE0_GAUSSIAN_XLA_HMC"
            ),
            "hmc": {"config": run["config"], "diagnostics": run["diagnostics"]},
            "modern_rhat": rhat,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow_version": tf.__version__,
            "evidence_role": "engineering_fixture_only",
            "nonclaims": NONCLAIMS,
        }
    )
    target = (
        ARTIFACT_ROOT / "phase0" / "gaussian_xla_hmc.json"
        if output_path is None
        else Path(output_path)
    )
    _write_new_json(target, result)
    return result


def run_screen_frozen_phase0_smoke(
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run a short CPU/XLA HMC integration smoke on the historical screen fit."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraGapClosureError(
            "frozen Phase 0 smoke requires CUDA_VISIBLE_DEVICES=-1"
        )
    screen = _read_mapping(SCREEN_WIDE_RESULT_PATH, "wide screen training result")
    if (
        screen.get("schema") != "bayesfilter.lgssm_neutra_strict_training_job.v1"
        or screen.get("passed") is not True
        or screen.get("recipe", {}).get("recipe_id") != EXPECTED_RECIPE_ID
        or int(screen.get("steps", -1)) != 500
        or screen.get("target_signature") != EXPECTED_TARGET_SIGNATURE
    ):
        raise LGSSMNeuTraGapClosureError("historical wide screen fixture is invalid")
    payload_path = _verify_file_reference(screen.get("payload"), "screen frozen payload")
    loaded = load_frozen_neutra_artifact(
        _read_mapping(payload_path, "screen frozen payload"),
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope="lgssm_neutra_gap_closure_phase0_historical_screen_smoke",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    config = TensorHMCConfig(
        num_results=8,
        num_burnin_steps=8,
        step_size=0.01,
        num_leapfrog_steps=2,
        seed=(20260715, 2),
    )
    run = run_batched_hmc(
        adapter=adapter,
        initial_state=tf.zeros((CHAIN_COUNT, DIMENSION), tf.float64),
        config=config,
    )
    flat_z = tf.reshape(run["samples"], (-1, DIMENSION))
    raw = loaded.transport.forward_batch(flat_z)
    status = bundle.adapter.target_status_telemetry(raw)
    status_valid = bool(
        tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
    )
    passed = bool(run["diagnostics"]["health_passed"] and status_valid)
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_gap_closure_phase0_frozen_smoke.v1",
            "passed": passed,
            "decision": (
                "PASS_PHASE0_HISTORICAL_SCREEN_FROZEN_HMC_SMOKE"
                if passed
                else "BLOCK_PHASE0_HISTORICAL_SCREEN_FROZEN_HMC_SMOKE"
            ),
            "source_result": {
                "path": str(SCREEN_WIDE_RESULT_PATH.relative_to(ROOT)),
                "file_sha256": _file_sha256(SCREEN_WIDE_RESULT_PATH),
                "evidence_role": "historical_500_step_structural_fixture_only",
            },
            "payload_file_sha256": _file_sha256(payload_path),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "hmc": {"config": run["config"], "diagnostics": run["diagnostics"]},
            "target_status_all_valid": status_valid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": True,
            "evidence_role": "engineering_integration_smoke_only",
            "nonclaims": NONCLAIMS,
        }
    )
    target = (
        ARTIFACT_ROOT / "phase0" / "historical_screen_frozen_hmc_smoke.json"
        if output_path is None
        else Path(output_path)
    )
    _write_new_json(target, result)
    return result


def run_fresh_frozen_objective_probe(
    candidate_id: str,
    *,
    device_mode: str,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run the canonical deterministic probe for one fresh frozen candidate."""

    if device_mode == "cpu_hidden_xla":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise LGSSMNeuTraGapClosureError("CPU probe requires CUDA_VISIBLE_DEVICES=-1")
        device = "/CPU:0"
        memory_policy: Mapping[str, Any] | None = None
    elif device_mode == "trusted_gpu_xla":
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            raise LGSSMNeuTraGapClosureError("GPU probe cannot run with CUDA hidden")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        device = "/GPU:0"
    else:
        raise ValueError("device_mode must be cpu_hidden_xla or trusted_gpu_xla")
    tf.config.set_soft_device_placement(False)
    candidate, source, loaded = _load_fresh_candidate(candidate_id)
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    if bundle.adapter.adapter_signature() != EXPECTED_ADAPTER_SIGNATURE:
        raise LGSSMNeuTraGapClosureError("fresh probe adapter signature mismatch")
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=f"lgssm_neutra_gap_closure_phase3_{candidate}",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    probe = _canonical_probe_points()

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg: tf.Tensor):
        theta = loaded.transport.forward_batch(z_arg)
        logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        value, score = adapter.log_prob_and_grad_batch(z_arg)
        status = bundle.adapter.target_status_telemetry(theta)
        return theta, logdet, value, score, status

    with tf.device(device):
        first = compiled(probe)
        second = compiled(probe)
    theta, logdet, value, score, status = first
    numeric = (theta, logdet, value, score)
    all_finite = bool(
        all(tf.reduce_all(tf.math.is_finite(item)).numpy() for item in numeric)
    )
    status_valid = bool(
        tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
    )
    exact_repeat = bool(
        all(tf.reduce_all(tf.equal(left, right)).numpy() for left, right in zip(first[:4], second[:4]))
    )
    output_devices = tuple(str(item.device) for item in numeric)
    expected_token = "GPU" if device_mode == "trusted_gpu_xla" else "CPU"
    correct_device = all(expected_token in item.upper() for item in output_devices)
    passed = bool(all_finite and status_valid and exact_repeat and correct_device)
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_fresh_frozen_objective_probe.v1",
            "candidate_id": candidate,
            "device_mode": device_mode,
            "passed": passed,
            "decision": (
                "PASS_FRESH_FROZEN_OBJECTIVE_PROBE"
                if passed
                else "BLOCK_FRESH_FROZEN_OBJECTIVE_PROBE"
            ),
            "source_record": source,
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "payload_file_sha256": source["payload_file_sha256"],
            "probe_hash": _tensor_sha256(probe),
            "probe_shape": tuple(int(item) for item in probe.shape),
            "theta": theta,
            "logdet": logdet,
            "value": value,
            "score": score,
            "target_status": status,
            "all_outputs_finite": all_finite,
            "target_status_all_valid": status_valid,
            "second_call_exact": exact_repeat,
            "output_devices": output_devices,
            "all_outputs_on_requested_device": correct_device,
            "jit_compile": True,
            "dtype": "float64",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "gpu_memory_policy": memory_policy,
            "evidence_role": "frozen_objective_identity_and_cross_device_parity",
            "nonclaims": NONCLAIMS,
        }
    )
    target = (
        ARTIFACT_ROOT / "phase3" / candidate / f"{device_mode}_probe.json"
        if output_path is None
        else Path(output_path)
    )
    _write_new_json(target, result)
    return result


def finalize_frozen_objective_validation(
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Compare both candidates' GPU and CPU probes and freeze viability."""

    rows = []
    tolerances = {
        "theta_max_abs": 1.0e-12,
        "logdet_max_abs": 1.0e-12,
        "value_max_abs": 1.0e-8,
        "score_max_abs": 1.0e-8,
    }
    for candidate in FRESH_CANDIDATE_SOURCES:
        directory = ARTIFACT_ROOT / "phase3" / candidate
        gpu_path = directory / "trusted_gpu_xla_probe.json"
        cpu_path = directory / "cpu_hidden_xla_probe.json"
        gpu = _read_mapping(gpu_path, f"{candidate} GPU probe")
        cpu = _read_mapping(cpu_path, f"{candidate} CPU probe")
        for payload, mode in ((gpu, "trusted_gpu_xla"), (cpu, "cpu_hidden_xla")):
            if (
                payload.get("schema")
                != "bayesfilter.lgssm_neutra_fresh_frozen_objective_probe.v1"
                or payload.get("candidate_id") != candidate
                or payload.get("device_mode") != mode
                or payload.get("passed") is not True
                or not _artifact_hash_matches(payload)
            ):
                raise LGSSMNeuTraGapClosureError(f"{candidate} {mode} probe is invalid")
        identity_keys = (
            "target_signature",
            "adapter_signature",
            "artifact_signature",
            "transport_hash",
            "fixed_transport_adapter_signature",
            "payload_file_sha256",
            "probe_hash",
        )
        identity_match = all(gpu.get(key) == cpu.get(key) for key in identity_keys)
        differences = {}
        for key in ("theta", "logdet", "value", "score"):
            left = tf.convert_to_tensor(gpu[key], tf.float64)
            right = tf.convert_to_tensor(cpu[key], tf.float64)
            if left.shape != right.shape:
                raise LGSSMNeuTraGapClosureError(f"{candidate} {key} shape mismatch")
            differences[f"{key}_max_abs"] = float(
                tf.reduce_max(tf.abs(left - right)).numpy()
            )
        parity_passed = bool(
            identity_match
            and all(differences[key] <= tolerances[key] for key in tolerances)
        )
        rows.append(
            {
                "candidate_id": candidate,
                "passed": parity_passed,
                "identity_match": identity_match,
                "differences": differences,
                "tolerances": tolerances,
                "gpu_probe": {
                    "path": str(gpu_path.relative_to(ROOT)),
                    "file_sha256": _file_sha256(gpu_path),
                    "artifact_hash": gpu["artifact_hash"],
                },
                "cpu_probe": {
                    "path": str(cpu_path.relative_to(ROOT)),
                    "file_sha256": _file_sha256(cpu_path),
                    "artifact_hash": cpu["artifact_hash"],
                },
                "artifact_signature": gpu["artifact_signature"],
                "transport_hash": gpu["transport_hash"],
                "fixed_transport_adapter_signature": gpu[
                    "fixed_transport_adapter_signature"
                ],
            }
        )
    viable = tuple(row["candidate_id"] for row in rows if row["passed"])
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_frozen_objective_validation.v1",
            "phase": 3,
            "passed": bool(viable),
            "decision": (
                "PASS_PHASE3_FROZEN_OBJECTIVE_VALIDATION"
                if viable
                else "BLOCK_PHASE3_NO_VALID_FROZEN_OBJECTIVE"
            ),
            "candidate_results": tuple(rows),
            "viable_candidates": viable,
            "rejected_candidates": tuple(
                row["candidate_id"] for row in rows if not row["passed"]
            ),
            "hmc_seed_ledger": {
                "dense_seed1201": {
                    "primary_probe": (20260715, 2101),
                    "primary_verification": (20260715, 2201),
                    "repair_probe": (20260715, 2151),
                    "repair_verification": (20260715, 2251),
                    "confirmation": (20260715, 3101),
                },
                "dense_seed1202": {
                    "primary_probe": (20260715, 2301),
                    "primary_verification": (20260715, 2401),
                    "repair_probe": (20260715, 2351),
                    "repair_verification": (20260715, 2451),
                    "confirmation": (20260715, 3301),
                },
            },
            "serious_sampling_executed": False,
            "evidence_role": "frozen_objective_validation_before_hmc_tuning",
            "nonclaims": NONCLAIMS,
        }
    )
    target = ARTIFACT_ROOT / "phase3" / "result.json" if output_path is None else Path(output_path)
    _write_new_json(target, result)
    return result


def run_hmc_tuning_candidate(
    candidate_id: str,
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run primary and, when eligible, one repair fixed-grid HMC tuning round."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraGapClosureError("HMC tuning requires CUDA_VISIBLE_DEVICES=-1")
    candidate, source, loaded = _load_fresh_candidate(candidate_id)
    phase3 = _read_mapping(ARTIFACT_ROOT / "phase3" / "result.json", "Phase 3 result")
    if (
        phase3.get("schema") != "bayesfilter.lgssm_neutra_frozen_objective_validation.v1"
        or phase3.get("passed") is not True
        or candidate not in tuple(phase3.get("viable_candidates", ()))
        or not _artifact_hash_matches(phase3)
    ):
        raise LGSSMNeuTraGapClosureError("candidate is not Phase 3 viable")
    seed_ledger = phase3.get("hmc_seed_ledger", {}).get(candidate)
    if not isinstance(seed_ledger, Mapping):
        raise LGSSMNeuTraGapClosureError("candidate HMC seed ledger is missing")
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=f"lgssm_neutra_gap_closure_hmc_{candidate}",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    root = ARTIFACT_ROOT / "phase4" / candidate
    if root.exists():
        raise FileExistsError(f"candidate tuning root already exists: {root}")
    started = time.monotonic()
    primary = _run_hmc_tuning_round(
        candidate=candidate,
        loaded=loaded,
        adapter=adapter,
        bundle=bundle,
        round_id="primary",
        step_sizes=TUNING_PRIMARY_STEPS,
        probe_seed=tuple(seed_ledger["primary_probe"]),
        verification_seed=tuple(seed_ledger["primary_verification"]),
        root=root / "primary",
    )
    selected_round = primary
    repair = None
    if primary["admitted"] is not True and primary["repair_eligible"] is True:
        repair = _run_hmc_tuning_round(
            candidate=candidate,
            loaded=loaded,
            adapter=adapter,
            bundle=bundle,
            round_id="repair",
            step_sizes=TUNING_REPAIR_STEPS,
            probe_seed=tuple(seed_ledger["repair_probe"]),
            verification_seed=tuple(seed_ledger["repair_verification"]),
            root=root / "repair",
        )
        selected_round = repair
    admitted = bool(selected_round["admitted"])
    final_kernel = selected_round.get("final_kernel") if admitted else None
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_hmc_tuning_candidate.v1",
            "phase": 4,
            "candidate_id": candidate,
            "passed": admitted,
            "admitted": admitted,
            "decision": (
                "ADMIT_FIXED_NEUTRA_HMC_KERNEL"
                if admitted
                else "REJECT_CANDIDATE_AFTER_DECLARED_TUNING_REPAIR"
            ),
            "source_record": source,
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "seed_ledger": dict(seed_ledger),
            "primary_round": primary,
            "repair_round": repair,
            "selected_round": selected_round["round_id"],
            "final_kernel": final_kernel,
            "final_kernel_hash": (
                None if final_kernel is None else f"sha256:{_stable_json_hash(final_kernel)}"
            ),
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "physical_gpus": tuple(
                    str(item) for item in tf.config.list_physical_devices("GPU")
                ),
                "jit_compile": True,
                "dtype": "float64",
                "chain_count": CHAIN_COUNT,
                "batched_chain_execution": True,
            },
            "serious_sampling_executed": False,
            "evidence_role": "fresh_modern_rhat_hmc_tuning_admission",
            "nonclaims": NONCLAIMS,
        }
    )
    target = root / "result.json" if output_path is None else Path(output_path)
    _write_new_json(target, result)
    return result


def _run_hmc_tuning_round(
    *,
    candidate: str,
    loaded: Any,
    adapter: Any,
    bundle: Any,
    round_id: str,
    step_sizes: Sequence[float],
    probe_seed: tuple[int, int],
    verification_seed: tuple[int, int],
    root: Path,
) -> Mapping[str, Any]:
    rows = []
    initial_state = _canonical_probe_points()
    hard_veto = False
    for index, step_size in enumerate(step_sizes):
        seed = (int(probe_seed[0]), int(probe_seed[1]) + 10_000 + index)
        run = run_batched_hmc(
            adapter=adapter,
            initial_state=initial_state,
            config=TensorHMCConfig(
                num_results=64,
                num_burnin_steps=128,
                step_size=float(step_size),
                num_leapfrog_steps=TUNING_LEAPFROG_STEPS,
                seed=seed,
            ),
        )
        diagnostics = dict(run["diagnostics"])
        health_passed = bool(diagnostics["health_passed"])
        if not health_passed:
            hard_veto = True
        rows.append(
            {
                "grid_index": index,
                "step_size": float(step_size),
                "num_leapfrog_steps": TUNING_LEAPFROG_STEPS,
                "trajectory_length": float(step_size) * TUNING_LEAPFROG_STEPS,
                "seed": seed,
                "acceptance_rate": diagnostics["acceptance_rate"],
                "health_passed": health_passed,
                "diagnostics": diagnostics,
            }
        )
    selected = select_tuning_candidate(rows)
    verification = None
    admitted = False
    final_kernel = None
    rhat_failed = False
    if selected is not None and not hard_veto:
        run = run_batched_hmc(
            adapter=adapter,
            initial_state=initial_state,
            config=TensorHMCConfig(
                num_results=1000,
                num_burnin_steps=1000,
                step_size=float(selected["step_size"]),
                num_leapfrog_steps=TUNING_LEAPFROG_STEPS,
                seed=verification_seed,
            ),
        )
        z_samples = tf.convert_to_tensor(run["samples"], tf.float64)
        flat_raw = loaded.transport.forward_batch(tf.reshape(z_samples, (-1, DIMENSION)))
        raw_samples = tf.reshape(flat_raw, tf.shape(z_samples))
        admission = tuning_admission(
            probe_rows=rows,
            verification_samples=raw_samples,
            parameter_names=bundle.parameter_names,
            verification_health=run["diagnostics"],
        )
        admitted = bool(admission["admitted"])
        rhat_failed = admission["verification_modern_rhat"]["passed"] is not True
        archive_dir = root / "verification"
        z_archive = write_tensor_archive(
            archive_dir / "retained_z.tftensor",
            z_samples,
            metadata={
                "candidate_id": candidate,
                "round_id": round_id,
                "coordinate_system": "frozen_neutra_latent_z",
            },
        )
        raw_archive = write_tensor_archive(
            archive_dir / "retained_raw.tftensor",
            raw_samples,
            metadata={
                "candidate_id": candidate,
                "round_id": round_id,
                "coordinate_system": "lgssm_raw_parameters",
            },
        )
        final_kernel = {
            "schema": "bayesfilter.lgssm_neutra_fixed_hmc_kernel.v1",
            "candidate_id": candidate,
            "round_id": round_id,
            "step_size": float(selected["step_size"]),
            "num_leapfrog_steps": TUNING_LEAPFROG_STEPS,
            "trajectory_length": float(selected["step_size"])
            * TUNING_LEAPFROG_STEPS,
            "target_signature": bundle.target_signature,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "verification_seed": verification_seed,
            "verification_draws_per_chain": 1000,
            "verification_burnin": 1000,
            "verification_rhat_definition": (
                "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
            ),
            "verification_rhat_max": RHAT_MAX,
        }
        verification = {
            "admission": admission,
            "hmc_diagnostics": run["diagnostics"],
            "z_archive": z_archive,
            "raw_archive": raw_archive,
            "final_kernel": final_kernel,
        }
    repair_eligible = bool(
        not admitted
        and not hard_veto
        and (selected is None or rhat_failed)
        and round_id == "primary"
    )
    return {
        "round_id": round_id,
        "step_sizes": tuple(float(item) for item in step_sizes),
        "probe_seed_root": probe_seed,
        "probe_rows": tuple(rows),
        "selected_probe": selected,
        "verification_seed": verification_seed,
        "verification": verification,
        "admitted": admitted,
        "repair_eligible": repair_eligible,
        "hard_veto": hard_veto,
        "final_kernel": final_kernel if admitted else None,
        "selection_rule": "healthy_acceptance_0.60_to_0.90_closest_to_0.75_then_grid_order",
        "admission_rule": "fresh_1000_draw_raw_coordinate_rank_and_folded_rhat_le_1.01",
    }


def finalize_hmc_tuning(
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Freeze all candidate tuning admissions before confirmation."""

    phase3 = _read_mapping(ARTIFACT_ROOT / "phase3" / "result.json", "Phase 3 result")
    rows = []
    for candidate in tuple(phase3.get("viable_candidates", ())):
        path = ARTIFACT_ROOT / "phase4" / candidate / "result.json"
        row = _read_mapping(path, f"{candidate} tuning result")
        if (
            row.get("schema") != "bayesfilter.lgssm_neutra_hmc_tuning_candidate.v1"
            or row.get("candidate_id") != candidate
            or not _artifact_hash_matches(row)
        ):
            raise LGSSMNeuTraGapClosureError(f"{candidate} tuning result is invalid")
        rows.append(row)
    admitted = tuple(row["candidate_id"] for row in rows if row.get("admitted") is True)
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_hmc_tuning_result.v1",
            "phase": 4,
            "passed": bool(admitted),
            "decision": (
                "PASS_PHASE4_HMC_ADMISSIONS_FROZEN"
                if admitted
                else "BLOCK_PHASE4_NO_ADMITTED_HMC_CANDIDATE"
            ),
            "all_phase3_candidates_processed": True,
            "candidate_order": tuple(phase3.get("viable_candidates", ())),
            "admitted_candidates": admitted,
            "rejected_candidates": tuple(
                row["candidate_id"] for row in rows if row.get("admitted") is not True
            ),
            "candidate_results": tuple(
                {
                    "candidate_id": row["candidate_id"],
                    "admitted": row["admitted"],
                    "selected_round": row["selected_round"],
                    "final_kernel_hash": row.get("final_kernel_hash"),
                    "result_path": str(
                        (ARTIFACT_ROOT / "phase4" / row["candidate_id"] / "result.json")
                        .relative_to(ROOT)
                    ),
                    "result_artifact_hash": row["artifact_hash"],
                }
                for row in rows
            ),
            "post_admission_retuning_allowed": False,
            "serious_sampling_executed": False,
            "nonclaims": NONCLAIMS,
        }
    )
    target = ARTIFACT_ROOT / "phase4" / "result.json" if output_path is None else Path(output_path)
    _write_new_json(target, result)
    return result


def run_corrected_sequential_hmc_candidate(
    candidate_id: str,
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run the fresh sequential warm-up and admission repair for one candidate."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraGapClosureError(
            "corrected sequential HMC requires CUDA_VISIBLE_DEVICES=-1"
        )
    candidate, source, loaded = _load_fresh_candidate(candidate_id)
    historical_path = ARTIFACT_ROOT / "phase4" / candidate / "result.json"
    historical = _read_mapping(historical_path, f"{candidate} historical Phase 4")
    selected = historical.get("repair_round", {}).get("selected_probe")
    if (
        historical.get("schema") != "bayesfilter.lgssm_neutra_hmc_tuning_candidate.v1"
        or historical.get("candidate_id") != candidate
        or not _artifact_hash_matches(historical)
        or not isinstance(selected, Mapping)
        or float(selected.get("step_size", float("nan"))) != 0.8
        or int(selected.get("num_leapfrog_steps", -1)) != TUNING_LEAPFROG_STEPS
        or selected.get("health_passed") is not True
    ):
        raise LGSSMNeuTraGapClosureError(
            "historical selected fixed kernel is missing or invalid"
        )
    try:
        seeds = SEQUENTIAL_REPAIR_SEEDS[candidate]
    except KeyError as exc:
        raise LGSSMNeuTraGapClosureError(
            f"corrected sequential seeds are missing for {candidate}"
        ) from exc
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=f"lgssm_neutra_sequential_repair_{candidate}",
        evidence_path=(
            "docs/plans/"
            "bayesfilter-lgssm-neutra-sequential-hmc-repair-plan-2026-07-15.md"
        ),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    def raw_transform(z_samples: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(z_samples)
        raw = loaded.transport.forward_batch(tf.reshape(z_samples, (-1, DIMENSION)))
        return tf.reshape(raw, shape)

    candidate_root = SEQUENTIAL_REPAIR_ROOT / candidate
    target = candidate_root / "result.json" if output_path is None else Path(output_path)
    config = SequentialNeuTraHMCConfig(
        step_size=0.8,
        num_leapfrog_steps=TUNING_LEAPFROG_STEPS,
        warmup_seed=tuple(seeds["warmup"]),
        retained_seed=tuple(seeds["retained"]),
    )
    run = run_sequential_neutra_hmc(
        adapter=adapter,
        initial_state=_canonical_probe_points(),
        raw_transform=raw_transform,
        parameter_names=bundle.parameter_names,
        config=config,
        archive_root=candidate_root / "samples",
    )
    public_run = {
        key: value for key, value in run.items() if not key.startswith("private_")
    }
    kernel = {
        "schema": "bayesfilter.lgssm_neutra_fixed_hmc_kernel.v2",
        "candidate_id": candidate,
        "step_size": 0.8,
        "num_leapfrog_steps": TUNING_LEAPFROG_STEPS,
        "trajectory_length": 8.0,
        "target_signature": bundle.target_signature,
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "fixed_transport_adapter_signature": adapter.adapter_signature(),
        "sequential_config": config.payload(),
    }
    command = (
        sys.executable,
        "docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py",
        "sequential-candidate",
        "--job-id",
        candidate,
    )
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_sequential_hmc_candidate.v1",
            "phase": "R1",
            "candidate_id": candidate,
            "passed": bool(run["passed"]),
            "decision": run["decision"],
            "source_record": source,
            "historical_fixed_budget_result": {
                "path": str(historical_path.relative_to(ROOT)),
                "file_sha256": _file_sha256(historical_path),
                "artifact_hash": historical["artifact_hash"],
                "evidence_role": "superseded_incomplete_fixed_budget_diagnostic",
            },
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "fixed_kernel": kernel,
            "fixed_kernel_hash": f"sha256:{_stable_json_hash(kernel)}",
            "sequential_run": public_run,
            "runtime_manifest": runtime_manifest(
                command=command,
                output_paths=(target, candidate_root / "samples"),
            ),
            "evidence_role": "corrected_sequential_warmup_and_tuning_admission",
            "serious_confirmation_executed": False,
            "nonclaims": NONCLAIMS,
        }
    )
    _write_new_json(target, result)
    return result


def finalize_corrected_sequential_hmc(
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Aggregate corrected candidate admissions without ranking candidates."""

    rows = []
    for candidate in tuple(FRESH_CANDIDATE_SOURCES):
        path = SEQUENTIAL_REPAIR_ROOT / candidate / "result.json"
        row = _read_mapping(path, f"{candidate} corrected sequential result")
        if (
            row.get("schema") != "bayesfilter.lgssm_neutra_sequential_hmc_candidate.v1"
            or row.get("candidate_id") != candidate
            or not _artifact_hash_matches(row)
        ):
            raise LGSSMNeuTraGapClosureError(
                f"{candidate} corrected sequential result is invalid"
            )
        rows.append(row)
    admitted = tuple(row["candidate_id"] for row in rows if row.get("passed") is True)
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_sequential_hmc_result.v1",
            "phase": "R1",
            "passed": bool(admitted),
            "decision": (
                "PASS_SEQUENTIAL_HMC_ADMISSION"
                if admitted
                else "REJECT_ALL_FIXED_KERNELS_AT_SEQUENTIAL_CAPS"
            ),
            "candidate_order": tuple(FRESH_CANDIDATE_SOURCES),
            "admitted_candidates": admitted,
            "nonadmitted_candidates": tuple(
                row["candidate_id"] for row in rows if row.get("passed") is not True
            ),
            "candidate_results": tuple(
                {
                    "candidate_id": row["candidate_id"],
                    "passed": row["passed"],
                    "decision": row["decision"],
                    "warmup_results_per_chain": row["sequential_run"][
                        "warmup_results_per_chain"
                    ],
                    "retained_results_per_chain": row["sequential_run"][
                        "retained_results_per_chain"
                    ],
                    "result_path": str(
                        (
                            SEQUENTIAL_REPAIR_ROOT
                            / row["candidate_id"]
                            / "result.json"
                        ).relative_to(ROOT)
                    ),
                    "result_artifact_hash": row["artifact_hash"],
                    "fixed_kernel_hash": row["fixed_kernel_hash"],
                }
                for row in rows
            ),
            "statistically_supported_ranking": None,
            "next_step": (
                "refresh_and_execute_confirmatory_phase5"
                if admitted
                else "close_current_fixed_kernel_candidates_without_direction_rejection"
            ),
            "nonclaims": NONCLAIMS,
        }
    )
    target = SEQUENTIAL_REPAIR_ROOT / "result.json" if output_path is None else Path(output_path)
    _write_new_json(target, result)
    return result


def run_confirmatory_hmc_candidate(
    candidate_id: str,
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run fresh adaptive confirmation for one sequentially admitted candidate."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LGSSMNeuTraGapClosureError(
            "confirmatory HMC requires CUDA_VISIBLE_DEVICES=-1"
        )
    candidate, source, loaded = _load_fresh_candidate(candidate_id)
    r1_path = SEQUENTIAL_REPAIR_ROOT / candidate / "result.json"
    r1 = _read_mapping(r1_path, f"{candidate} R1 result")
    aggregate = _read_mapping(SEQUENTIAL_REPAIR_ROOT / "result.json", "R1 aggregate")
    if (
        r1.get("schema") != "bayesfilter.lgssm_neutra_sequential_hmc_candidate.v1"
        or r1.get("candidate_id") != candidate
        or r1.get("passed") is not True
        or not _artifact_hash_matches(r1)
        or aggregate.get("passed") is not True
        or candidate not in tuple(aggregate.get("admitted_candidates", ()))
        or not _artifact_hash_matches(aggregate)
    ):
        raise LGSSMNeuTraGapClosureError("candidate is not R1-admitted")
    historical = _read_mapping(
        ARTIFACT_ROOT / "phase4" / candidate / "result.json",
        f"{candidate} historical Phase 4",
    )
    confirmation_seed = tuple(historical.get("seed_ledger", {}).get("confirmation", ()))
    if len(confirmation_seed) != 2:
        raise LGSSMNeuTraGapClosureError("confirmation seed is missing")
    retained_seed = (int(confirmation_seed[0]), int(confirmation_seed[1]) + 10000)
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope=f"lgssm_neutra_confirmation_{candidate}",
        evidence_path=(
            "docs/plans/"
            "bayesfilter-lgssm-neutra-sequential-hmc-r2-subplan-2026-07-15.md"
        ),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    def raw_transform(z_samples: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(z_samples)
        raw = loaded.transport.forward_batch(tf.reshape(z_samples, (-1, DIMENSION)))
        return tf.reshape(raw, shape)

    def convergence(raw_samples: tf.Tensor) -> Mapping[str, Any]:
        return full_convergence_diagnostics(
            raw_samples,
            parameter_names=bundle.parameter_names,
        )

    candidate_root = CONFIRMATION_ROOT / candidate
    target = candidate_root / "result.json" if output_path is None else Path(output_path)
    config = SequentialNeuTraHMCConfig(
        step_size=float(r1["fixed_kernel"]["step_size"]),
        num_leapfrog_steps=int(r1["fixed_kernel"]["num_leapfrog_steps"]),
        warmup_seed=confirmation_seed,
        retained_seed=retained_seed,
        warmup_chunk_results=1000,
        warmup_min_results=2000,
        warmup_check_window_results=1000,
        warmup_max_results=10000,
        warmup_rhat_max=1.05,
        retained_chunk_results=2000,
        retained_min_results=4000,
        retained_max_results=10000,
        retained_rhat_max=RHAT_MAX,
    )
    run = run_sequential_neutra_hmc(
        adapter=adapter,
        initial_state=_canonical_probe_points(),
        raw_transform=raw_transform,
        parameter_names=bundle.parameter_names,
        config=config,
        archive_root=candidate_root / "samples",
        retained_diagnostic_fn=convergence,
    )
    convergence_result = (
        run["retained_checks"][-1].get("full_convergence")
        if run["retained_checks"]
        else None
    )
    comparator = load_plain_hmc_comparator_summary()
    posterior = None
    if run["retained_results_per_chain"] > 0:
        posterior = posterior_summary(
            candidate_samples=run["private_retained_raw"],
            parameter_names=bundle.parameter_names,
            comparator=comparator,
        )
    passed = bool(
        run["passed"]
        and isinstance(convergence_result, Mapping)
        and convergence_result.get("passed") is True
        and isinstance(posterior, Mapping)
        and posterior.get("all_finite") is True
        and posterior.get("posterior_agreement_passed") is True
        and posterior.get("recovery_passed") is True
    )
    public_run = {
        key: value for key, value in run.items() if not key.startswith("private_")
    }
    command = (
        sys.executable,
        "docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py",
        "confirm-candidate",
        "--job-id",
        candidate,
    )
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_confirmatory_hmc_candidate.v1",
            "phase": "R2",
            "candidate_id": candidate,
            "passed": passed,
            "decision": (
                "PASS_EXACT_FIXTURE_NEUTRA_CONFIRMATION"
                if passed
                else "REJECT_CANDIDATE_CONFIRMATORY_GATES"
            ),
            "source_record": source,
            "r1_admission": {
                "path": str(r1_path.relative_to(ROOT)),
                "file_sha256": _file_sha256(r1_path),
                "artifact_hash": r1["artifact_hash"],
                "fixed_kernel_hash": r1["fixed_kernel_hash"],
            },
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "fixed_kernel": r1["fixed_kernel"],
            "sequential_run": public_run,
            "final_full_convergence": convergence_result,
            "posterior_summary": posterior,
            "plain_hmc_comparator": {
                "path": str(Path(comparator["path"]).relative_to(ROOT)),
                "file_sha256": comparator["file_sha256"],
                "artifact_hash": comparator["artifact_hash"],
                "parameter_names": comparator["parameter_names"],
            },
            "runtime_manifest": runtime_manifest(
                command=command,
                output_paths=(target, candidate_root / "samples"),
            ),
            "evidence_role": "fresh_confirmatory_exact_fixture_neutra_hmc",
            "nonclaims": NONCLAIMS,
        }
    )
    _write_new_json(target, result)
    return result


def finalize_confirmatory_hmc(
    *,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Aggregate confirmatory candidates and emit the exact-fixture verdict."""

    rows = []
    for candidate in tuple(FRESH_CANDIDATE_SOURCES):
        path = CONFIRMATION_ROOT / candidate / "result.json"
        row = _read_mapping(path, f"{candidate} confirmatory result")
        if (
            row.get("schema") != "bayesfilter.lgssm_neutra_confirmatory_hmc_candidate.v1"
            or row.get("candidate_id") != candidate
            or not _artifact_hash_matches(row)
        ):
            raise LGSSMNeuTraGapClosureError(
                f"{candidate} confirmatory result is invalid"
            )
        rows.append(row)
    passing = tuple(row["candidate_id"] for row in rows if row.get("passed") is True)
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_confirmatory_hmc_result.v1",
            "phase": "R2",
            "passed": bool(passing),
            "decision": (
                "PASS_NEUTRA_ON_EXACT_FAVORABLE_LGSSM_FIXTURE"
                if passing
                else "NO_CANDIDATE_PASSED_EXACT_FIXTURE_CONFIRMATION"
            ),
            "candidate_order": tuple(FRESH_CANDIDATE_SOURCES),
            "passing_candidates": passing,
            "nonpassing_candidates": tuple(
                row["candidate_id"] for row in rows if row.get("passed") is not True
            ),
            "candidate_results": tuple(
                _confirmatory_candidate_summary(row)
                for row in rows
            ),
            "statistically_supported_ranking": None,
            "positive_claim_scope": (
                "at_least_one_specific_frozen_neutra_candidate_passes_recorded_"
                "hmc_convergence_plain_hmc_agreement_and_truth_recovery_gates_"
                "on_this_exact_favorable_18d_lgssm_fixture"
                if passing
                else None
            ),
            "nonclaims": NONCLAIMS,
        }
    )
    target = CONFIRMATION_ROOT / "result.json" if output_path is None else Path(output_path)
    _write_new_json(target, result)
    return result


def _confirmatory_candidate_summary(row: Mapping[str, Any]) -> Mapping[str, Any]:
    convergence = row.get("final_full_convergence")
    posterior = row.get("posterior_summary")
    sequential = row["sequential_run"]
    return {
        "candidate_id": row["candidate_id"],
        "passed": row["passed"],
        "decision": row["decision"],
        "warmup_results_per_chain": sequential["warmup_results_per_chain"],
        "retained_results_per_chain": sequential["retained_results_per_chain"],
        "hard_vetoes": sequential["hard_vetoes"],
        "max_rhat": convergence.get("max_rhat") if isinstance(convergence, Mapping) else None,
        "min_bulk_ess": (
            convergence.get("min_bulk_ess") if isinstance(convergence, Mapping) else None
        ),
        "min_tail_ess": (
            convergence.get("min_tail_ess") if isinstance(convergence, Mapping) else None
        ),
        "max_posterior_agreement_combined_mcse": (
            posterior.get("max_posterior_agreement_combined_mcse")
            if isinstance(posterior, Mapping)
            else None
        ),
        "max_abs_mean_minus_truth_over_sd": (
            posterior.get("max_abs_mean_minus_truth_over_sd")
            if isinstance(posterior, Mapping)
            else None
        ),
        "result_path": str(
            (CONFIRMATION_ROOT / row["candidate_id"] / "result.json").relative_to(ROOT)
        ),
        "result_artifact_hash": row["artifact_hash"],
    }


def select_tuning_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    acceptance_band: tuple[float, float] = TUNING_ACCEPTANCE_BAND,
) -> Mapping[str, Any] | None:
    """Select a finite probe by band/midpoint while preserving grid order."""

    lower, upper = (float(item) for item in acceptance_band)
    if not (0.0 < lower <= upper < 1.0):
        raise ValueError("acceptance_band must be within (0, 1)")
    midpoint = 0.5 * (lower + upper)
    eligible = []
    for index, row in enumerate(rows):
        acceptance = float(row.get("acceptance_rate", float("nan")))
        if (
            row.get("health_passed") is True
            and lower <= acceptance <= upper
            and math.isfinite(acceptance)
        ):
            eligible.append((abs(acceptance - midpoint), index, row))
    return None if not eligible else dict(min(eligible, key=lambda item: item[:2])[2])


def tuning_admission(
    *,
    probe_rows: Sequence[Mapping[str, Any]],
    verification_samples: Any,
    parameter_names: Sequence[str],
    verification_health: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Admit only a nominated finite probe with fresh modern R-hat evidence."""

    selected = select_tuning_candidate(probe_rows)
    rhat = rank_normalized_split_rhat_summary(
        verification_samples,
        rhat_max=RHAT_MAX,
    )
    names = tuple(str(item) for item in parameter_names)
    if len(names) != int(tf.convert_to_tensor(verification_samples).shape[-1]):
        raise LGSSMNeuTraGapClosureError("parameter names do not match verification draws")
    admitted = bool(
        selected is not None
        and verification_health.get("health_passed") is True
        and rhat["passed"] is True
        and int(rhat["draw_count_per_chain"]) >= 1000
    )
    return {
        "admitted": admitted,
        "selected_probe": selected,
        "verification_modern_rhat": rhat,
        "verification_health": dict(verification_health),
        "acceptance_role": "nomination_only",
        "admission_rule": "finite_probe_plus_fresh_rank_and_folded_rhat_le_1.01",
    }


def validate_strict_training_result(
    path: str | Path,
    *,
    expected_job_id: str,
) -> Mapping[str, Any]:
    """Validate a current strict 5,000-step result and its frozen payload."""

    result_path = Path(path).resolve()
    result = _read_mapping(result_path, "strict training result")
    if result.get("schema") != "bayesfilter.lgssm_neutra_strict_training_job.v1":
        raise LGSSMNeuTraGapClosureError("strict training result schema mismatch")
    if result.get("passed") is not True or not _artifact_hash_matches(result):
        raise LGSSMNeuTraGapClosureError("strict training result did not pass integrity")
    if result.get("job_kind") != "final" or result.get("job_id") != expected_job_id:
        raise LGSSMNeuTraGapClosureError("strict training job identity mismatch")
    if int(result.get("steps", -1)) != FINAL_STEPS:
        raise LGSSMNeuTraGapClosureError("strict training step count mismatch")
    recipe = result.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("recipe_id") != EXPECTED_RECIPE_ID:
        raise LGSSMNeuTraGapClosureError("strict training recipe mismatch")
    if result.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise LGSSMNeuTraGapClosureError("strict training target signature mismatch")
    if result.get("adapter_signature") != EXPECTED_ADAPTER_SIGNATURE:
        raise LGSSMNeuTraGapClosureError("strict training adapter signature mismatch")
    if int(result.get("compiled_training_program_invocations", -1)) != 1:
        raise LGSSMNeuTraGapClosureError("strict training invocation count mismatch")
    if result.get("compiled_training_control_flow") != "tf_while_loop":
        raise LGSSMNeuTraGapClosureError("strict training control-flow mismatch")
    runtime = result.get("runtime_metadata")
    if not isinstance(runtime, Mapping) or runtime.get("jit_compile") is not True:
        raise LGSSMNeuTraGapClosureError("strict training did not record XLA")
    parity = result.get("frozen_reload_and_score_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise LGSSMNeuTraGapClosureError("strict frozen reload/score parity failed")
    gpu = result.get("gpu_manifest")
    if not isinstance(gpu, Mapping) or not gpu.get("physical_gpus"):
        raise LGSSMNeuTraGapClosureError("strict training GPU evidence is missing")
    memory = gpu.get("gpu_memory_policy")
    if not isinstance(memory, Mapping) or memory.get("mode") != "memory_growth":
        raise LGSSMNeuTraGapClosureError("strict training memory-growth evidence failed")
    selection = result.get("selected_recipe_source")
    if not isinstance(selection, Mapping):
        raise LGSSMNeuTraGapClosureError("selected recipe source is missing")
    selected_reference = selection.get("selected_recipe")
    if (
        not isinstance(selected_reference, Mapping)
        or selected_reference.get("file_sha256") != EXPECTED_SELECTION_FILE_SHA256
        or selection.get("recipe_id") != EXPECTED_RECIPE_ID
        or selection.get("screen_weights_reused") is not False
    ):
        raise LGSSMNeuTraGapClosureError("selected recipe source hash mismatch")
    payload_path = _verify_file_reference(result.get("payload"), "frozen payload")
    loaded = load_frozen_neutra_artifact(
        _read_mapping(payload_path, "frozen payload"),
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    if loaded.artifact_signature != result.get("artifact_signature"):
        raise LGSSMNeuTraGapClosureError("frozen artifact signature mismatch")
    if loaded.manifest.transport_hash != result.get("transport_hash"):
        raise LGSSMNeuTraGapClosureError("frozen transport hash mismatch")
    return {
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_artifact_hash": result["artifact_hash"],
        "job_id": expected_job_id,
        "payload_path": str(payload_path),
        "payload_file_sha256": _file_sha256(payload_path),
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "recipe_id": EXPECTED_RECIPE_ID,
        "passed": True,
    }


def post_validate_completed_training_attempt(
    attempt_root: str | Path,
    *,
    expected_job_id: str,
    output_path: str | Path,
) -> Mapping[str, Any]:
    """Recover a completed job rejected only by the old import-closure defect."""

    root = Path(attempt_root).resolve()
    rejected = _read_mapping(root / "result.json", "rejected strict result")
    if (
        rejected.get("schema") != "bayesfilter.lgssm_neutra_strict_training_job.v1"
        or rejected.get("passed") is not False
        or rejected.get("job_kind") != "final"
        or rejected.get("job_id") != expected_job_id
        or rejected.get("error", {}).get("message")
        != "repository import closure uses NumPy"
    ):
        raise LGSSMNeuTraGapClosureError("attempt is not the known closure-rejection case")
    completion_path = root / "training_completion.json"
    completion = _read_mapping(completion_path, "training completion")
    if (
        completion.get("schema")
        != "bayesfilter.lgssm_neutra_strict_training_completion.v1"
        or completion.get("job_kind") != "final"
        or completion.get("job_id") != expected_job_id
        or int(completion.get("steps", -1)) != FINAL_STEPS
        or completion.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or completion.get("adapter_signature") != EXPECTED_ADAPTER_SIGNATURE
        or completion.get("target_status_all_valid") is not True
        or not _artifact_hash_matches(completion)
    ):
        raise LGSSMNeuTraGapClosureError("durable training completion is invalid")
    selection = completion.get("selected_recipe_source")
    if (
        not isinstance(selection, Mapping)
        or selection.get("recipe_id") != EXPECTED_RECIPE_ID
        or selection.get("screen_weights_reused") is not False
        or selection.get("selected_recipe", {}).get("file_sha256")
        != EXPECTED_SELECTION_FILE_SHA256
    ):
        raise LGSSMNeuTraGapClosureError("training completion recipe identity mismatch")
    checkpoint_path = _verify_file_reference(completion.get("checkpoint"), "checkpoint")
    payload_path = _verify_file_reference(completion.get("payload"), "frozen payload")
    training_config_path = root / "training" / "training_config.json"
    training_config_payload = _read_mapping(training_config_path, "training config")

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
    )
    from bayesfilter.testing import lgssm_neutra_strict_training_tf as strict

    config = PlainDenseIAFTrainingConfig(
        target_signature=str(training_config_payload["target_signature"]),
        dimension=int(training_config_payload["dimension"]),
        affine_center=training_config_payload["affine_center"],
        affine_factor=training_config_payload["affine_factor"],
        output_dir=root / "training",
        seed=tuple(training_config_payload["seed"]),
        hidden_layers=tuple(training_config_payload["hidden_layers"]),
        stage_count=int(training_config_payload["stage_count"]),
        activation=str(training_config_payload["activation"]),
        s_max=float(training_config_payload["s_max"]),
        init_scale=float(training_config_payload["init_scale"]),
        steps=int(training_config_payload["steps"]),
        batch_size=int(training_config_payload["batch_size"]),
        learning_rate=float(training_config_payload["learning_rate"]),
        final_learning_rate_fraction=float(
            training_config_payload["final_learning_rate_fraction"]
        ),
        clip_norm=float(training_config_payload["clip_norm"]),
        beta1=float(training_config_payload["beta1"]),
        beta2=float(training_config_payload["beta2"]),
        epsilon=float(training_config_payload["epsilon"]),
        checkpoint_every=int(training_config_payload["checkpoint_every"]),
        heartbeat_every=int(training_config_payload["heartbeat_every"]),
        jit_compile=bool(training_config_payload["jit_compile"]),
        device=str(training_config_payload["device"]),
        require_gpu=bool(training_config_payload["require_gpu"]),
    )
    if config.config_hash != training_config_payload.get("config_hash"):
        raise LGSSMNeuTraGapClosureError("restored training config hash mismatch")
    if config.hidden_layers != (36, 36) or config.stage_count != 3:
        raise LGSSMNeuTraGapClosureError("restored training architecture mismatch")
    loaded = load_frozen_neutra_artifact(
        _read_mapping(payload_path, "frozen payload"),
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    flow = restore_plain_dense_iaf_flow(config=config, state_path=checkpoint_path)
    parity = strict._compiled_parity(tf, flow, loaded)
    if parity.get("passed") is not True:
        raise LGSSMNeuTraGapClosureError("recovered frozen reload/score parity failed")
    closure = strict.audit_imported_bayesfilter_closure()
    if closure.get("passed") is not True:
        raise LGSSMNeuTraGapClosureError("repaired repository import closure failed")
    latest = _read_mapping(root / "training" / "training_latest.json", "training latest")
    if (
        int(latest.get("step", -1)) != FINAL_STEPS
        or latest.get("target_status_all_valid") is not True
        or latest.get("target_values_finite") is not True
    ):
        raise LGSSMNeuTraGapClosureError("terminal training diagnostic failed")
    result = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_completed_training_post_validation.v1",
            "passed": True,
            "decision": "ACCEPT_COMPLETED_SEED_AFTER_IMPORT_CLOSURE_HARNESS_REPAIR",
            "job_kind": "final",
            "job_id": expected_job_id,
            "steps": FINAL_STEPS,
            "recipe_id": EXPECTED_RECIPE_ID,
            "seed": tuple(config.seed),
            "target_signature": EXPECTED_TARGET_SIGNATURE,
            "adapter_signature": EXPECTED_ADAPTER_SIGNATURE,
            "training_state_hash": completion["training_state_hash"],
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "original_rejected_result": {
                "path": str((root / "result.json").relative_to(ROOT)),
                "file_sha256": _file_sha256(root / "result.json"),
                "classification": "post_training_unrelated_eager_import_false_positive",
            },
            "training_completion": {
                "path": str(completion_path.relative_to(ROOT)),
                "file_sha256": _file_sha256(completion_path),
                "artifact_hash": completion["artifact_hash"],
            },
            "checkpoint": {
                **dict(completion["checkpoint"]),
                "state_hash_verified_by_restore": True,
            },
            "payload": dict(completion["payload"]),
            "frozen_reload_and_score_parity": parity,
            "terminal_training_diagnostic": latest,
            "compiled_training_program_invocations": completion[
                "compiled_training_program_invocations"
            ],
            "compiled_training_control_flow": completion[
                "compiled_training_control_flow"
            ],
            "checkpoint_policy": completion["checkpoint_policy"],
            "repository_import_closure_after_repair": closure,
            "gpu_memory_policy": memory_policy,
            "gpu_devices": tuple(str(item) for item in tf.config.list_logical_devices("GPU")),
            "original_elapsed_seconds": float(rejected["elapsed_seconds"]),
            "evidence_role": "fresh_5000_step_engineering_candidate_for_phase3",
            "repair_non_effect": (
                "runtime package import was made lazy; target, optimizer, seed, "
                "training tensors, checkpoint, and frozen payload were not changed"
            ),
            "nonclaims": NONCLAIMS,
        }
    )
    _write_new_json(Path(output_path), result)
    return result


def load_plain_hmc_comparator_summary(
    path: str | Path = COMPARATOR_PATH,
    *,
    expected_file_sha256: str = EXPECTED_COMPARATOR_FILE_SHA256,
) -> Mapping[str, Any]:
    """Bind the immutable plain-HMC summary without reading its NumPy archive."""

    comparator_path = Path(path).resolve()
    if _file_sha256(comparator_path) != expected_file_sha256:
        raise LGSSMNeuTraGapClosureError("plain-HMC comparator file hash mismatch")
    payload = _read_mapping(comparator_path, "plain-HMC comparator")
    if payload.get("schema") != "bayesfilter.deterministic_lgssm_hmc_final_recovery_result.v1":
        raise LGSSMNeuTraGapClosureError("plain-HMC comparator schema mismatch")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("passed") is not True:
        raise LGSSMNeuTraGapClosureError("plain-HMC modern diagnostics did not pass")
    if diagnostics.get("definitions", {}).get("rhat") != (
        "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
    ):
        raise LGSSMNeuTraGapClosureError("plain-HMC R-hat definition mismatch")
    rows = payload.get("parameter_recovery")
    if not isinstance(rows, list) or len(rows) != DIMENSION:
        raise LGSSMNeuTraGapClosureError("plain-HMC comparator parameter count mismatch")
    names = tuple(str(row.get("parameter")) for row in rows)
    for row in rows:
        values = tuple(
            float(row.get(key, float("nan")))
            for key in ("posterior_mean", "posterior_sd", "mean_mcse", "truth")
        )
        if not all(math.isfinite(item) for item in values) or values[1] <= 0.0 or values[2] <= 0.0:
            raise LGSSMNeuTraGapClosureError("plain-HMC comparator contains invalid values")
    return {
        "path": str(comparator_path),
        "file_sha256": expected_file_sha256,
        "artifact_hash": payload.get("artifact_hash"),
        "parameter_names": names,
        "posterior_mean": tf.constant(
            [row["posterior_mean"] for row in rows], tf.float64
        ),
        "posterior_sd": tf.constant([row["posterior_sd"] for row in rows], tf.float64),
        "mean_mcse": tf.constant([row["mean_mcse"] for row in rows], tf.float64),
        "truth": tf.constant([row["truth"] for row in rows], tf.float64),
        "diagnostics": diagnostics,
    }


def write_tensor_archive(
    path: str | Path,
    tensor: Any,
    *,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write one immutable TensorFlow tensor plus a standard-library sidecar."""

    target = Path(path).resolve()
    sidecar = target.with_suffix(target.suffix + ".json")
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"tensor archive already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    values = tf.convert_to_tensor(tensor)
    tf.io.write_file(str(target), tf.io.serialize_tensor(values))
    payload = _with_artifact_hash(
        {
            "schema": "bayesfilter.lgssm_neutra_tensor_archive.v1",
            "tensor_path": str(target),
            "tensor_file_sha256": _file_sha256(target),
            "shape": tuple(int(item) for item in values.shape),
            "dtype": values.dtype.name,
            "all_finite": bool(tf.reduce_all(tf.math.is_finite(values)).numpy()),
            "metadata": dict(metadata),
        }
    )
    _write_new_json(sidecar, payload)
    return payload


def read_tensor_archive(sidecar_path: str | Path) -> tf.Tensor:
    """Verify and read an immutable TensorFlow tensor archive."""

    sidecar = _read_mapping(Path(sidecar_path), "tensor archive sidecar")
    if sidecar.get("schema") != "bayesfilter.lgssm_neutra_tensor_archive.v1":
        raise LGSSMNeuTraGapClosureError("tensor archive schema mismatch")
    if not _artifact_hash_matches(sidecar):
        raise LGSSMNeuTraGapClosureError("tensor archive sidecar hash mismatch")
    path = Path(str(sidecar["tensor_path"]))
    if _file_sha256(path) != sidecar.get("tensor_file_sha256"):
        raise LGSSMNeuTraGapClosureError("tensor archive file hash mismatch")
    dtype = tf.dtypes.as_dtype(str(sidecar["dtype"]))
    value = tf.io.parse_tensor(tf.io.read_file(str(path)), out_type=dtype)
    value = tf.ensure_shape(value, tuple(int(item) for item in sidecar["shape"]))
    if sidecar.get("all_finite") is True and not bool(
        tf.reduce_all(tf.math.is_finite(value)).numpy()
    ):
        raise LGSSMNeuTraGapClosureError("tensor archive became nonfinite")
    return value


def posterior_summary(
    *,
    candidate_samples: Any,
    parameter_names: Sequence[str],
    comparator: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Compute TensorFlow/TFP posterior agreement, recovery, and uncertainty."""

    samples = tf.convert_to_tensor(candidate_samples, tf.float64)
    if samples.shape.rank != 3 or samples.shape[1:] != (CHAIN_COUNT, DIMENSION):
        raise LGSSMNeuTraGapClosureError("candidate samples must have shape [draw,4,18]")
    names = tuple(str(item) for item in parameter_names)
    if names != tuple(comparator["parameter_names"]):
        raise LGSSMNeuTraGapClosureError("candidate/comparator parameter order mismatch")
    pooled = tf.reshape(samples, (-1, DIMENSION))
    candidate_mean = tf.reduce_mean(pooled, axis=0)
    candidate_sd = tf.math.reduce_std(pooled, axis=0)
    half = int(samples.shape[0]) // 2
    split = tf.reshape(
        tf.stack((samples[:half], samples[-half:]), axis=2),
        (half, 2 * CHAIN_COUNT, DIMENSION),
    )
    mean_ess = tfp.mcmc.effective_sample_size(
        split,
        filter_beyond_positive_pairs=True,
        cross_chain_dims=1,
    )
    candidate_mcse = candidate_sd / tf.sqrt(mean_ess)
    combined_mcse = tf.sqrt(
        tf.square(candidate_mcse) + tf.square(comparator["mean_mcse"])
    )
    agreement = tf.abs(candidate_mean - comparator["posterior_mean"]) / combined_mcse
    recovery = tf.abs(candidate_mean - comparator["truth"]) / candidate_sd
    quantiles = tfp.stats.percentile(
        pooled,
        (5.0, 50.0, 95.0),
        axis=0,
        interpolation="linear",
    )
    all_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    (candidate_mean, candidate_sd, mean_ess, candidate_mcse, agreement, recovery),
                    axis=0,
                )
            )
        ).numpy()
    )
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "parameter": name,
                "truth": float(comparator["truth"][index].numpy()),
                "neutra_mean": float(candidate_mean[index].numpy()),
                "neutra_sd": float(candidate_sd[index].numpy()),
                "neutra_mean_ess": float(mean_ess[index].numpy()),
                "neutra_mean_mcse": float(candidate_mcse[index].numpy()),
                "plain_hmc_mean": float(comparator["posterior_mean"][index].numpy()),
                "plain_hmc_mean_mcse": float(comparator["mean_mcse"][index].numpy()),
                "agreement_combined_mcse": float(agreement[index].numpy()),
                "agreement_passed": bool(
                    agreement[index].numpy() <= POSTERIOR_AGREEMENT_MAX_Z
                ),
                "recovery_posterior_sd": float(recovery[index].numpy()),
                "recovery_passed": bool(recovery[index].numpy() <= RECOVERY_MAX_Z),
                "q05": float(quantiles[0, index].numpy()),
                "q50": float(quantiles[1, index].numpy()),
                "q95": float(quantiles[2, index].numpy()),
            }
        )
    return {
        "all_finite": all_finite,
        "posterior_agreement_passed": bool(
            all_finite
            and tf.reduce_all(agreement <= POSTERIOR_AGREEMENT_MAX_Z).numpy()
        ),
        "max_posterior_agreement_combined_mcse": float(
            tf.reduce_max(agreement).numpy()
        ),
        "recovery_passed": bool(
            all_finite and tf.reduce_all(recovery <= RECOVERY_MAX_Z).numpy()
        ),
        "max_abs_mean_minus_truth_over_sd": float(tf.reduce_max(recovery).numpy()),
        "parameter_rows": tuple(rows),
        "mean_mcse_definition": "posterior_sd / sqrt(split-chain cross-chain ESS)",
        "nonclaims": NONCLAIMS,
    }


def full_convergence_diagnostics(
    samples: Any,
    *,
    parameter_names: Sequence[str],
) -> Mapping[str, Any]:
    """Compute the fixed serious modern R-hat and ESS gate."""

    return rank_normalized_hmc_diagnostics(
        samples,
        parameter_names=parameter_names,
        thresholds=RankNormalizedHMCThresholds(
            rhat_max=RHAT_MAX,
            bulk_ess_min=BULK_ESS_MIN,
            tail_ess_min=TAIL_ESS_MIN,
        ),
    )


def audit_new_route_import_closure() -> Mapping[str, Any]:
    """Reject NumPy, host callbacks, and legacy HMC imports in this route."""

    paths = (
        Path(__file__).resolve(),
        ROOT / "docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py",
    )
    violations = []
    forbidden_modules = {
        "numpy",
        "bayesfilter.inference.hmc",
        "bayesfilter.inference.fixed_transport_hmc_tuning",
        "bayesfilter.testing.lgssm_neutra_serious_validation_tf",
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == name or alias.name.startswith(name + ".")
                        for name in forbidden_modules
                    ):
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = str(node.module)
                if any(module == name or module.startswith(name + ".") for name in forbidden_modules):
                    violations.append(f"{path.name}:{node.lineno}:{module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tf"
                    and node.func.attr in {"numpy_function", "py_function"}
                ):
                    violations.append(f"{path.name}:{node.lineno}:tf.{node.func.attr}")
    return {
        "passed": not violations,
        "paths": tuple(str(path.relative_to(ROOT)) for path in paths),
        "violations": tuple(violations),
        "policy": "tensorflow_tfp_only_no_legacy_numpy_hmc_imports",
    }


def phase0_local_checks() -> Mapping[str, Any]:
    """Bind immutable inputs and report the local Phase 0 static gate."""

    selection_hash = _file_sha256(SELECTION_PATH)
    if selection_hash != EXPECTED_SELECTION_FILE_SHA256:
        raise LGSSMNeuTraGapClosureError("selected recipe file hash mismatch")
    selected = _read_mapping(SELECTION_PATH, "selected recipe")
    if selected.get("selected_recipe", {}).get("recipe_id") != EXPECTED_RECIPE_ID:
        raise LGSSMNeuTraGapClosureError("selected recipe identity mismatch")
    comparator = load_plain_hmc_comparator_summary()
    closure = audit_new_route_import_closure()
    passed = bool(PLAN_PATH.is_file() and closure["passed"])
    return {
        "passed": passed,
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "selection_file_sha256": selection_hash,
        "selected_recipe": EXPECTED_RECIPE_ID,
        "comparator_file_sha256": comparator["file_sha256"],
        "comparator_parameter_names": comparator["parameter_names"],
        "import_closure": closure,
    }


def runtime_manifest(*, command: Sequence[str], output_paths: Sequence[Path]) -> Mapping[str, Any]:
    """Return the required serious-run provenance fields."""

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "command": tuple(str(item) for item in command),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "jit_compile": True,
        "dtype": "float64",
        "plan_file": str(PLAN_PATH.relative_to(ROOT)),
        "output_paths": tuple(str(path) for path in output_paths),
    }


def _verify_file_reference(reference: Any, label: str) -> Path:
    if not isinstance(reference, Mapping):
        raise LGSSMNeuTraGapClosureError(f"{label} reference must be a mapping")
    path = (ROOT / str(reference.get("path", ""))).resolve()
    if not path.is_file():
        raise LGSSMNeuTraGapClosureError(f"{label} file is missing")
    if _file_sha256(path) != reference.get("file_sha256"):
        raise LGSSMNeuTraGapClosureError(f"{label} file hash mismatch")
    if path.stat().st_size != int(reference.get("byte_count", -1)):
        raise LGSSMNeuTraGapClosureError(f"{label} byte count mismatch")
    return path


def _load_fresh_candidate(candidate_id: str):
    candidate = str(candidate_id)
    try:
        specification = FRESH_CANDIDATE_SOURCES[candidate]
    except KeyError as exc:
        raise LGSSMNeuTraGapClosureError(f"unknown fresh candidate: {candidate}") from exc
    record_path = Path(specification["record_path"]).resolve()
    record = _read_mapping(record_path, f"{candidate} source record")
    if (
        record.get("schema") != specification["record_schema"]
        or record.get("passed") is not True
        or record.get("job_id") != candidate
        or int(record.get("steps", -1)) != FINAL_STEPS
        or record.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or record.get("adapter_signature") != EXPECTED_ADAPTER_SIGNATURE
        or not _artifact_hash_matches(record)
    ):
        raise LGSSMNeuTraGapClosureError(f"{candidate} source record is invalid")
    payload_path = _verify_file_reference(record.get("payload"), f"{candidate} payload")
    payload_hash = _file_sha256(payload_path)
    if payload_hash != specification["payload_file_sha256"]:
        raise LGSSMNeuTraGapClosureError(f"{candidate} payload hash mismatch")
    loaded = load_frozen_neutra_artifact(
        _read_mapping(payload_path, f"{candidate} frozen payload"),
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    if (
        loaded.artifact_signature != record.get("artifact_signature")
        or loaded.manifest.transport_hash != record.get("transport_hash")
    ):
        raise LGSSMNeuTraGapClosureError(f"{candidate} frozen identity mismatch")
    source = {
        "path": str(record_path.relative_to(ROOT)),
        "file_sha256": _file_sha256(record_path),
        "artifact_hash": record["artifact_hash"],
        "payload_path": str(payload_path.relative_to(ROOT)),
        "payload_file_sha256": payload_hash,
    }
    return candidate, source, loaded


def _canonical_probe_points() -> tf.Tensor:
    values = tf.range(4 * DIMENSION, dtype=tf.float64)
    midpoint = tf.constant(0.5 * (4 * DIMENSION - 1), tf.float64)
    return tf.reshape(
        (values - midpoint) / tf.constant(97.0, tf.float64),
        (4, DIMENSION),
    )


def _tensor_sha256(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LGSSMNeuTraGapClosureError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise LGSSMNeuTraGapClosureError(f"{label} must contain a JSON object")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _artifact_hash_matches(payload: Mapping[str, Any]) -> bool:
    normalized = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_hash", "artifact_hash_semantics"}
    }
    return payload.get("artifact_hash") == f"sha256:{_stable_json_hash(normalized)}"


def _with_artifact_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = f"sha256:{_stable_json_hash(result)}"
    result["artifact_hash_semantics"] = (
        "stable_json_sha256_excluding_artifact_hash_fields"
    )
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        materialized = value.numpy()
        if hasattr(materialized, "tolist"):
            return _json_ready(materialized.tolist())
        if hasattr(materialized, "item"):
            return _json_ready(materialized.item())
        return _json_ready(materialized)
    return value
