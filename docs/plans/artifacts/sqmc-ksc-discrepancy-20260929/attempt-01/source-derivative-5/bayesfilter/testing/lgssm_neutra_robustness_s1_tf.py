"""Same-fixture third-seed NeuTra HMC robustness harness."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_hmc import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    SequentialNeuTraHMCConfig,
    run_batched_hmc,
    run_sequential_neutra_hmc,
    TensorHMCConfig,
)
from bayesfilter.testing import lgssm_neutra_gap_closure_tf as reference


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "docs/plans/bayesfilter-neutra-hmc-robustness-s1-subplan-2026-07-15.md"
ARTIFACT_ROOT = ROOT / (
    "docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-"
    "2026-07-15/s1"
)
TRAINING_RESULT = ARTIFACT_ROOT / (
    "training-attempt-01/phase4/training_jobs/dense_seed1203/"
    "attempt_1_graph_native/result.json"
)
CANDIDATE_ID = "dense_seed1203"
EXPECTED_PAYLOAD_SHA256 = (
    "a1a2f10c4f642c6daf6e359537183db27b30d05eea920a7135fe25adb83fe571"
)
TUNING_ROOT = ARTIFACT_ROOT / "hmc-tuning-attempt-01"
CONFIRMATION_ROOT = ARTIFACT_ROOT / "confirmation-attempt-01"
STEP_SIZES = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
NUM_LEAPFROG_STEPS = 10
PROBE_SEED = (20260715, 5101)
ADMISSION_WARMUP_SEED = (20260715, 5201)
ADMISSION_RETAINED_SEED = (20260715, 5301)
CONFIRMATION_WARMUP_SEED = (20260715, 5401)
CONFIRMATION_RETAINED_SEED = (20260715, 5501)

NONCLAIMS = (
    "one additional training seed on the same favorable 18D LGSSM fixture only",
    "training loss and acceptance are explanatory or nomination-only",
    "no sampler superiority, calibration, population reliability, or broad robustness claim",
    "no new-fixture, production-readiness, or universal NeuTra claim",
)


class S1NeuTraHMCError(RuntimeError):
    """Raised when S1 identity, execution, or evidence checks fail closed."""


def run_s1_tuning_and_admission() -> Mapping[str, Any]:
    """Nominate a fixed kernel, then admit it through the shared controller."""

    _require_cpu_hidden()
    if TUNING_ROOT.exists():
        raise FileExistsError(f"S1 tuning root exists: {TUNING_ROOT}")
    source, loaded, bundle, adapter = _load_candidate()
    started = time.monotonic()
    rows = []
    hard_vetoes = []
    initial_state = reference._canonical_probe_points()
    for index, step_size in enumerate(STEP_SIZES):
        seed = (PROBE_SEED[0], PROBE_SEED[1] + 10_000 + index)
        run = run_batched_hmc(
            adapter=adapter,
            initial_state=initial_state,
            config=TensorHMCConfig(
                num_results=64,
                num_burnin_steps=128,
                step_size=step_size,
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=seed,
            ),
            target_status_summary_fn=reference._lgssm_target_status_summary,
        )
        health = dict(run["diagnostics"])
        if health["health_passed"] is not True:
            hard_vetoes.append(f"probe_health_failed:{index}")
        rows.append(
            {
                "grid_index": index,
                "step_size": step_size,
                "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
                "trajectory_length": step_size * NUM_LEAPFROG_STEPS,
                "seed": seed,
                "acceptance_rate": health["acceptance_rate"],
                "health_passed": health["health_passed"],
                "diagnostics": health,
            }
        )
    selected = reference.select_tuning_candidate(rows)
    admission = None
    if selected is not None and not hard_vetoes:
        config = SequentialNeuTraHMCConfig(
            step_size=float(selected["step_size"]),
            num_leapfrog_steps=NUM_LEAPFROG_STEPS,
            warmup_seed=ADMISSION_WARMUP_SEED,
            retained_seed=ADMISSION_RETAINED_SEED,
            warmup_chunk_results=1000,
            warmup_min_results=2000,
            warmup_check_window_results=1000,
            warmup_max_results=10000,
            warmup_rhat_max=1.05,
            retained_chunk_results=1000,
            retained_min_results=1000,
            retained_max_results=10000,
            retained_rhat_max=1.01,
        )
        admission = run_sequential_neutra_hmc(
            adapter=adapter,
            initial_state=initial_state,
            model_transform=_raw_transform(loaded),
            parameter_names=bundle.parameter_names,
            config=config,
            archive_callback=_archive_callback(TUNING_ROOT / "samples"),
            target_status_summary_fn=reference._lgssm_target_status_summary,
        )
    passed = bool(
        selected is not None
        and not hard_vetoes
        and isinstance(admission, Mapping)
        and admission.get("passed") is True
    )
    public_admission = (
        None
        if admission is None
        else {key: value for key, value in admission.items() if not key.startswith("private_")}
    )
    fixed_kernel = (
        {
            "schema": "bayesfilter.neutra_s1_fixed_hmc_kernel.v1",
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "candidate_id": CANDIDATE_ID,
            "step_size": float(selected["step_size"]),
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "trajectory_length": float(selected["step_size"]) * NUM_LEAPFROG_STEPS,
            "target_signature": bundle.target_signature,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "adapter_signature": adapter.adapter_signature(),
        }
        if passed
        else None
    )
    result_path = TUNING_ROOT / "result.json"
    result = reference._with_artifact_hash(
        {
            "schema": "bayesfilter.neutra_robustness_s1_tuning.v1",
            "candidate_id": CANDIDATE_ID,
            "passed": passed,
            "decision": (
                "ADMIT_S1_FIXED_KERNEL" if passed else "REJECT_S1_FIXED_KERNEL"
            ),
            "source_record": source,
            "probe_rows": tuple(rows),
            "acceptance_role": "nomination_only",
            "selected_probe": selected,
            "hard_vetoes": tuple(hard_vetoes),
            "sequential_admission": public_admission,
            "fixed_kernel": fixed_kernel,
            "fixed_kernel_hash": (
                None
                if fixed_kernel is None
                else f"sha256:{reference._stable_json_hash(fixed_kernel)}"
            ),
            "runtime_manifest": _runtime_manifest(
                command=(
                    sys.executable,
                    "docs/benchmarks/run_lgssm_neutra_robustness_s1_2026_07_15.py",
                    "tune-and-admit",
                ),
                elapsed_seconds=time.monotonic() - started,
                output_paths=(result_path, TUNING_ROOT / "samples"),
            ),
            "evidence_role": "fresh_kernel_nomination_and_shared_controller_admission",
            "confirmation_executed": False,
            "nonclaims": NONCLAIMS,
        }
    )
    reference._write_new_json(result_path, result)
    return result


def run_s1_confirmation() -> Mapping[str, Any]:
    """Confirm the admitted S1 kernel with fresh sequential draws."""

    _require_cpu_hidden()
    if CONFIRMATION_ROOT.exists():
        raise FileExistsError(f"S1 confirmation root exists: {CONFIRMATION_ROOT}")
    source, loaded, bundle, adapter = _load_candidate()
    tuning_path = TUNING_ROOT / "result.json"
    tuning = reference._read_mapping(tuning_path, "S1 tuning result")
    if (
        tuning.get("schema") != "bayesfilter.neutra_robustness_s1_tuning.v1"
        or tuning.get("passed") is not True
        or not reference._artifact_hash_matches(tuning)
        or not isinstance(tuning.get("fixed_kernel"), Mapping)
    ):
        raise S1NeuTraHMCError("S1 fixed kernel was not admitted")
    kernel = tuning["fixed_kernel"]
    started = time.monotonic()
    config = SequentialNeuTraHMCConfig(
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        warmup_seed=CONFIRMATION_WARMUP_SEED,
        retained_seed=CONFIRMATION_RETAINED_SEED,
        warmup_chunk_results=1000,
        warmup_min_results=2000,
        warmup_check_window_results=1000,
        warmup_max_results=10000,
        warmup_rhat_max=1.05,
        retained_chunk_results=2000,
        retained_min_results=4000,
        retained_max_results=10000,
        retained_rhat_max=1.01,
    )
    run = run_sequential_neutra_hmc(
        adapter=adapter,
        initial_state=reference._canonical_probe_points(),
        model_transform=_raw_transform(loaded),
        parameter_names=bundle.parameter_names,
        config=config,
        retained_diagnostic_fn=lambda draws: reference.full_convergence_diagnostics(
            draws, parameter_names=bundle.parameter_names
        ),
        archive_callback=_archive_callback(CONFIRMATION_ROOT / "samples"),
        target_status_summary_fn=reference._lgssm_target_status_summary,
    )
    convergence = (
        run["retained_checks"][-1].get("full_convergence")
        if run["retained_checks"]
        else None
    )
    comparator = reference.load_plain_hmc_comparator_summary()
    posterior = (
        reference.posterior_summary(
            candidate_samples=run["private_retained_raw"],
            parameter_names=bundle.parameter_names,
            comparator=comparator,
        )
        if run["retained_results_per_chain"]
        else None
    )
    passed = bool(
        run["passed"]
        and isinstance(convergence, Mapping)
        and convergence.get("passed") is True
        and isinstance(posterior, Mapping)
        and posterior.get("all_finite") is True
        and posterior.get("posterior_agreement_passed") is True
        and posterior.get("recovery_passed") is True
    )
    public_run = {
        key: value for key, value in run.items() if not key.startswith("private_")
    }
    result_path = CONFIRMATION_ROOT / "result.json"
    result = reference._with_artifact_hash(
        {
            "schema": "bayesfilter.neutra_robustness_s1_confirmation.v1",
            "candidate_id": CANDIDATE_ID,
            "passed": passed,
            "decision": (
                "PASS_S1_SAME_FIXTURE_THIRD_SEED"
                if passed
                else "REJECT_S1_SAME_FIXTURE_THIRD_SEED"
            ),
            "source_record": source,
            "tuning_result": {
                "path": str(tuning_path.relative_to(ROOT)),
                "file_sha256": reference._file_sha256(tuning_path),
                "artifact_hash": tuning["artifact_hash"],
                "fixed_kernel_hash": tuning["fixed_kernel_hash"],
            },
            "target_signature": bundle.target_signature,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_kernel": kernel,
            "sequential_run": public_run,
            "final_full_convergence": convergence,
            "posterior_summary": posterior,
            "plain_hmc_comparator": {
                "path": str(Path(comparator["path"]).relative_to(ROOT)),
                "file_sha256": comparator["file_sha256"],
                "artifact_hash": comparator["artifact_hash"],
                "parameter_names": comparator["parameter_names"],
            },
            "runtime_manifest": _runtime_manifest(
                command=(
                    sys.executable,
                    "docs/benchmarks/run_lgssm_neutra_robustness_s1_2026_07_15.py",
                    "confirm",
                ),
                elapsed_seconds=time.monotonic() - started,
                output_paths=(result_path, CONFIRMATION_ROOT / "samples"),
            ),
            "evidence_role": "same_fixture_additional_training_seed_downstream_hmc",
            "nonclaims": NONCLAIMS,
        }
    )
    reference._write_new_json(result_path, result)
    return result


def _load_candidate():
    record = reference._read_mapping(TRAINING_RESULT, "S1 training result")
    if (
        record.get("schema") != "bayesfilter.lgssm_neutra_strict_training_job.v1"
        or record.get("passed") is not True
        or record.get("job_id") != CANDIDATE_ID
        or tuple(record.get("seed", ())) != (20260715, 1203)
        or int(record.get("steps", -1)) != 5000
        or record.get("target_signature") != reference.EXPECTED_TARGET_SIGNATURE
        or record.get("adapter_signature") != reference.EXPECTED_ADAPTER_SIGNATURE
        or not reference._artifact_hash_matches(record)
    ):
        raise S1NeuTraHMCError("S1 training record identity or integrity failed")
    if record.get("frozen_reload_and_score_parity", {}).get("passed") is not True:
        raise S1NeuTraHMCError("S1 frozen parity failed")
    if record.get("repository_import_closure", {}).get("passed") is not True:
        raise S1NeuTraHMCError("S1 training import closure failed")
    memory = record.get("gpu_manifest", {}).get("gpu_memory_policy", {})
    if (
        memory.get("mode") != "memory_growth"
        or memory.get("all_physical_devices_memory_growth") is not True
        or record.get("gpu_manifest", {}).get("jit_compile") is not True
    ):
        raise S1NeuTraHMCError("S1 GPU/XLA memory policy evidence failed")
    payload_path = reference._verify_file_reference(record["payload"], "S1 payload")
    if reference._file_sha256(payload_path) != EXPECTED_PAYLOAD_SHA256:
        raise S1NeuTraHMCError("S1 frozen payload hash mismatch")
    loaded = load_frozen_neutra_artifact(
        reference._read_mapping(payload_path, "S1 frozen payload"),
        expected_target_signature=reference.EXPECTED_TARGET_SIGNATURE,
    )
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=reference.EXPECTED_TARGET_SIGNATURE
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope="lgssm_neutra_robustness_s1_dense_seed1203",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    source = {
        "path": str(TRAINING_RESULT.relative_to(ROOT)),
        "file_sha256": reference._file_sha256(TRAINING_RESULT),
        "artifact_hash": record["artifact_hash"],
        "payload_path": str(payload_path.relative_to(ROOT)),
        "payload_file_sha256": EXPECTED_PAYLOAD_SHA256,
        "training_seed": tuple(record["seed"]),
    }
    return source, loaded, bundle, adapter


def _raw_transform(loaded: Any):
    def transform(z_samples: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(z_samples)
        raw = loaded.transport.forward_batch(
            tf.reshape(z_samples, (-1, reference.DIMENSION))
        )
        return tf.reshape(raw, shape)

    return transform


def _archive_callback(root: Path):
    def archive(
        *,
        stage: str,
        chunk_index: int | None,
        latent_samples: tf.Tensor,
        model_samples: tf.Tensor,
        seed: tuple[int, int] | None,
        cumulative: bool,
    ) -> Mapping[str, Any]:
        suffix = "all" if cumulative else f"chunk_{int(chunk_index) + 1:04d}"
        metadata = {
            "stage": stage,
            "chunk_index": chunk_index,
            "seed": seed,
            "cumulative": cumulative,
        }
        return {
            "z": reference.write_tensor_archive(
                root / stage / f"{suffix}_z.tftensor",
                latent_samples,
                metadata={**metadata, "coordinate_system": "latent_z"},
            ),
            "raw": reference.write_tensor_archive(
                root / stage / f"{suffix}_raw.tftensor",
                model_samples,
                metadata={**metadata, "coordinate_system": "raw_parameters"},
            ),
        }

    return archive


def _require_cpu_hidden() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise S1NeuTraHMCError("S1 HMC requires CUDA_VISIBLE_DEVICES=-1")


def _runtime_manifest(
    *,
    command: Sequence[str],
    elapsed_seconds: float,
    output_paths: Sequence[Path],
) -> Mapping[str, Any]:
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
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "hardware": "CPU with CUDA devices intentionally hidden",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": True,
        "dtype": "float64",
        "training_seed": (20260715, 1203),
        "hmc_seeds": {
            "probe": PROBE_SEED,
            "admission_warmup": ADMISSION_WARMUP_SEED,
            "admission_retained": ADMISSION_RETAINED_SEED,
            "confirmation_warmup": CONFIRMATION_WARMUP_SEED,
            "confirmation_retained": CONFIRMATION_RETAINED_SEED,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "output_paths": tuple(str(path.relative_to(ROOT)) for path in output_paths),
        "plan_file": str(PLAN_PATH.relative_to(ROOT)),
        "result_file": str(output_paths[0].relative_to(ROOT)),
    }
