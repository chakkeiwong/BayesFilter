"""New-fixture NeuTra HMC admission and confirmation for Phase F2."""

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
    TensorHMCConfig,
    run_batched_hmc,
    run_sequential_neutra_hmc,
)
from bayesfilter.testing import lgssm_neutra_gap_closure_tf as reference
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "docs/plans/bayesfilter-neutra-hmc-robustness-f2-subplan-2026-07-15.md"
PROGRAM_ROOT = ROOT / (
    "docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-"
    "2026-07-15"
)
F0_ROOT = PROGRAM_ROOT / "f0"
F1_ROOT = PROGRAM_ROOT / "f1"
F2_ROOT = PROGRAM_ROOT / "f2"
CONFIG_PATH = F0_ROOT / "config.json"
FIXTURE_PATH = F0_ROOT / "plain-hmc/fixture_T120_seed20260715_701.json"
COMPARATOR_PATH = F0_ROOT / "plain-hmc/comparator-repair-attempt-02/result.json"
TRAINING_RESULT = F1_ROOT / "final/inherited_wide_lr5e3/attempt-01/result.json"
EXPECTED_TARGET_SIGNATURE = (
    "312d2f4ceb5d65bf18251fa53ae1276781c62fd2daefaba0bda8dc3d46a5d283"
)
EXPECTED_PAYLOAD_SHA256 = (
    "cab56a88caabe557ff8287f399902beddf839f4d43c482c4c132dc46075a5920"
)
EXPECTED_COMPARATOR_ARTIFACT_HASH = (
    "sha256:c0370eca9594b2660734236a7e1cdd55e638054b8a200c5317ef1600cb68b44e"
)
TUNING_ROOT = F2_ROOT / "tuning-attempt-01"
CONFIRMATION_ROOT = F2_ROOT / "confirmation-attempt-01"
STEP_SIZES = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
NUM_LEAPFROG_STEPS = 10
PROBE_SEED = (20260715, 9101)
ADMISSION_WARMUP_SEED = (20260715, 9201)
ADMISSION_RETAINED_SEED = (20260715, 9301)
CONFIRMATION_WARMUP_SEED = (20260715, 9401)
CONFIRMATION_RETAINED_SEED = (20260715, 9501)

NONCLAIMS = (
    "one frozen NeuTra candidate on one additional deterministic LGSSM fixture",
    "acceptance and training loss are nomination or explanatory diagnostics only",
    "no sampler superiority, calibration, population reliability, or broad robustness claim",
    "no production or universal default-readiness claim",
)


class F2NeuTraHMCError(RuntimeError):
    """Raised when F2 identity or evidence fails closed."""


def run_f2_tuning_and_admission() -> Mapping[str, Any]:
    _require_cpu_hidden()
    if TUNING_ROOT.exists():
        raise FileExistsError(f"F2 tuning root exists: {TUNING_ROOT}")
    source, loaded, bundle, adapter, comparator = _load_inputs()
    started = time.monotonic()
    rows = []
    configuration_vetoes = []
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
        diagnostics = dict(run["diagnostics"])
        if diagnostics["health_passed"] is not True:
            configuration_vetoes.append(f"probe_health_failed:{index}")
        rows.append(
            {
                "grid_index": index,
                "step_size": step_size,
                "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
                "trajectory_length": step_size * NUM_LEAPFROG_STEPS,
                "seed": seed,
                "acceptance_rate": diagnostics["acceptance_rate"],
                "health_passed": diagnostics["health_passed"],
                "diagnostics": diagnostics,
            }
        )
    selected = reference.select_tuning_candidate(rows)
    admission = None
    if selected is not None:
        admission = run_sequential_neutra_hmc(
            adapter=adapter,
            initial_state=initial_state,
            model_transform=_raw_transform(loaded),
            parameter_names=bundle.parameter_names,
            config=SequentialNeuTraHMCConfig(
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
            ),
            archive_callback=_archive_callback(TUNING_ROOT / "samples"),
            target_status_summary_fn=reference._lgssm_target_status_summary,
        )
    passed = bool(isinstance(admission, Mapping) and admission.get("passed") is True)
    public = None if admission is None else {
        key: value for key, value in admission.items() if not key.startswith("private_")
    }
    kernel = (
        {
            "schema": "bayesfilter.neutra_robustness_f2_fixed_kernel.v1",
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "step_size": float(selected["step_size"]),
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "trajectory_length": float(selected["step_size"]) * NUM_LEAPFROG_STEPS,
            "target_signature": bundle.target_signature,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "adapter_signature": adapter.adapter_signature(),
        }
        if passed else None
    )
    result_path = TUNING_ROOT / "result.json"
    result = reference._with_artifact_hash(
        {
            "schema": "bayesfilter.neutra_robustness_f2_tuning.v1",
            "passed": passed,
            "decision": "ADMIT_F2_NEUTRA_KERNEL" if passed else "REJECT_F2_NEUTRA_KERNEL",
            "source_record": source,
            "comparator": _comparator_reference(comparator),
            "probe_rows": tuple(rows),
            "acceptance_role": "nomination_only",
            "selected_probe": selected,
            "configuration_vetoes": tuple(configuration_vetoes),
            "sequential_admission": public,
            "fixed_kernel": kernel,
            "fixed_kernel_hash": None if kernel is None else f"sha256:{reference._stable_json_hash(kernel)}",
            "runtime_manifest": _manifest(
                stage="tune-and-admit",
                elapsed=time.monotonic() - started,
                outputs=(result_path, TUNING_ROOT / "samples"),
            ),
            "evidence_role": "new_fixture_neutra_kernel_nomination_and_admission",
            "confirmation_executed": False,
            "nonclaims": NONCLAIMS,
        }
    )
    reference._write_new_json(result_path, result)
    return result


def run_f2_confirmation() -> Mapping[str, Any]:
    _require_cpu_hidden()
    if CONFIRMATION_ROOT.exists():
        raise FileExistsError(f"F2 confirmation root exists: {CONFIRMATION_ROOT}")
    source, loaded, bundle, adapter, comparator = _load_inputs()
    tuning_path = TUNING_ROOT / "result.json"
    tuning = reference._read_mapping(tuning_path, "F2 tuning result")
    if (
        tuning.get("passed") is not True
        or not reference._artifact_hash_matches(tuning)
        or not isinstance(tuning.get("fixed_kernel"), Mapping)
    ):
        raise F2NeuTraHMCError("F2 kernel was not admitted")
    kernel = tuning["fixed_kernel"]
    started = time.monotonic()
    run = run_sequential_neutra_hmc(
        adapter=adapter,
        initial_state=reference._canonical_probe_points(),
        model_transform=_raw_transform(loaded),
        parameter_names=bundle.parameter_names,
        config=SequentialNeuTraHMCConfig(
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
        ),
        retained_diagnostic_fn=lambda draws: reference.full_convergence_diagnostics(
            draws, parameter_names=bundle.parameter_names
        ),
        archive_callback=_archive_callback(CONFIRMATION_ROOT / "samples"),
        target_status_summary_fn=reference._lgssm_target_status_summary,
    )
    convergence = run["retained_checks"][-1].get("full_convergence") if run["retained_checks"] else None
    posterior = (
        _posterior_comparison(
            samples=run["private_retained_raw"],
            parameter_names=bundle.parameter_names,
            comparator=comparator,
        )
        if run["retained_results_per_chain"] else None
    )
    passed = bool(
        run["passed"]
        and isinstance(convergence, Mapping) and convergence.get("passed") is True
        and isinstance(posterior, Mapping)
        and posterior.get("all_finite") is True
        and posterior.get("posterior_agreement_passed") is True
        and posterior.get("recovery_passed") is True
    )
    public = {key: value for key, value in run.items() if not key.startswith("private_")}
    result_path = CONFIRMATION_ROOT / "result.json"
    result = reference._with_artifact_hash(
        {
            "schema": "bayesfilter.neutra_robustness_f2_confirmation.v1",
            "passed": passed,
            "decision": "PASS_F2_NEW_FIXTURE_NEUTRA" if passed else "REJECT_F2_NEW_FIXTURE_NEUTRA",
            "source_record": source,
            "tuning_result": {
                "path": str(tuning_path.relative_to(ROOT)),
                "file_sha256": reference._file_sha256(tuning_path),
                "artifact_hash": tuning["artifact_hash"],
                "fixed_kernel_hash": tuning["fixed_kernel_hash"],
            },
            "comparator": _comparator_reference(comparator),
            "target_signature": bundle.target_signature,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "fixed_kernel": kernel,
            "sequential_run": public,
            "final_full_convergence": convergence,
            "posterior_summary": posterior,
            "runtime_manifest": _manifest(
                stage="confirm",
                elapsed=time.monotonic() - started,
                outputs=(result_path, CONFIRMATION_ROOT / "samples"),
            ),
            "evidence_role": "new_fixture_neutra_downstream_hmc_confirmation",
            "nonclaims": NONCLAIMS,
        }
    )
    reference._write_new_json(result_path, result)
    return result


def _load_inputs():
    training = reference._read_mapping(TRAINING_RESULT, "F1 training result")
    comparator = reference._read_mapping(COMPARATOR_PATH, "F0 comparator")
    if (
        training.get("schema") != "bayesfilter.neutra_robustness_f1_training_job.v1"
        or training.get("job_kind") != "final"
        or training.get("passed") is not True
        or training.get("screen_weights_reused") is not False
        or training.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or not _artifact_hash_matches_f1(training)
    ):
        raise F2NeuTraHMCError("F1 training record failed identity or integrity")
    if (
        comparator.get("schema") != "bayesfilter.neutra_robustness_f0_plain_hmc_comparator.v1"
        or comparator.get("passed") is not True
        or comparator.get("artifact_hash") != EXPECTED_COMPARATOR_ARTIFACT_HASH
        or comparator.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or not reference._artifact_hash_matches(comparator)
    ):
        raise F2NeuTraHMCError("F0 comparator failed identity or integrity")
    payload_path = reference._verify_file_reference(training["payload"], "F1 payload")
    if reference._file_sha256(payload_path) != EXPECTED_PAYLOAD_SHA256:
        raise F2NeuTraHMCError("F1 payload hash mismatch")
    loaded = load_frozen_neutra_artifact(
        reference._read_mapping(payload_path, "F1 payload"),
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    bundle = load_deterministic_lgssm_exact_target(
        config_path=CONFIG_PATH,
        fixture_path=FIXTURE_PATH,
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope="lgssm_new_fixture_neutra_f2",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    source = {
        "path": str(TRAINING_RESULT.relative_to(ROOT)),
        "file_sha256": reference._file_sha256(TRAINING_RESULT),
        "artifact_hash": training["artifact_hash"],
        "payload_path": str(payload_path.relative_to(ROOT)),
        "payload_file_sha256": EXPECTED_PAYLOAD_SHA256,
        "training_seed": tuple(training["seed"]),
    }
    return source, loaded, bundle, adapter, comparator


def _artifact_hash_matches_f1(payload: Mapping[str, Any]) -> bool:
    clean = {key: value for key, value in payload.items() if key not in {"artifact_hash", "artifact_hash_semantics"}}
    from bayesfilter.testing.lgssm_new_fixture_neutra_training_f1_tf import _stable_hash
    return payload.get("artifact_hash") == "sha256:" + _stable_hash(clean)


def _posterior_comparison(
    *, samples: Any, parameter_names: Sequence[str], comparator: Mapping[str, Any]
) -> Mapping[str, Any]:
    summary = comparator["posterior_summary"]
    converted = {
        "path": COMPARATOR_PATH,
        "file_sha256": reference._file_sha256(COMPARATOR_PATH),
        "artifact_hash": comparator["artifact_hash"],
        "parameter_names": tuple(summary["parameter_names"]),
        "posterior_mean": tf.constant(summary["posterior_mean"], tf.float64),
        "mean_mcse": tf.constant(summary["mean_mcse"], tf.float64),
        "truth": tf.constant(summary["truth"], tf.float64),
    }
    return reference.posterior_summary(
        candidate_samples=samples,
        parameter_names=parameter_names,
        comparator=converted,
    )


def _comparator_reference(comparator: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "path": str(COMPARATOR_PATH.relative_to(ROOT)),
        "file_sha256": reference._file_sha256(COMPARATOR_PATH),
        "artifact_hash": comparator["artifact_hash"],
        "target_signature": comparator["target_signature"],
        "parameter_names": comparator["posterior_summary"]["parameter_names"],
    }


def _raw_transform(loaded: Any):
    def transform(z: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(z)
        raw = loaded.transport.forward_batch(tf.reshape(z, (-1, 18)))
        return tf.reshape(raw, shape)
    return transform


def _archive_callback(root: Path):
    def archive(
        *, stage: str, chunk_index: int | None, latent_samples: tf.Tensor,
        model_samples: tf.Tensor, seed: tuple[int, int] | None, cumulative: bool,
    ) -> Mapping[str, Any]:
        suffix = "all" if cumulative else f"chunk_{int(chunk_index) + 1:04d}"
        metadata = {"stage": stage, "chunk_index": chunk_index, "seed": seed, "cumulative": cumulative}
        return {
            "z": reference.write_tensor_archive(root / stage / f"{suffix}_z.tftensor", latent_samples, metadata={**metadata, "coordinate_system": "neutra_latent_z"}),
            "raw": reference.write_tensor_archive(root / stage / f"{suffix}_raw.tftensor", model_samples, metadata={**metadata, "coordinate_system": "raw_parameters"}),
        }
    return archive


def _require_cpu_hidden() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise F2NeuTraHMCError("F2 HMC requires CUDA_VISIBLE_DEVICES=-1")


def _manifest(*, stage: str, elapsed: float, outputs: Sequence[Path]) -> Mapping[str, Any]:
    commit = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "git_commit": commit,
        "command": (sys.executable, "docs/benchmarks/run_lgssm_new_fixture_neutra_hmc_f2_2026_07_15.py", stage),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "hardware": "CPU with CUDA devices intentionally hidden",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": True,
        "dtype": "float64",
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "training_seed": (20260715, 8201),
        "hmc_seeds": {
            "probe": PROBE_SEED,
            "admission_warmup": ADMISSION_WARMUP_SEED,
            "admission_retained": ADMISSION_RETAINED_SEED,
            "confirmation_warmup": CONFIRMATION_WARMUP_SEED,
            "confirmation_retained": CONFIRMATION_RETAINED_SEED,
        },
        "elapsed_seconds": float(elapsed),
        "output_paths": tuple(str(path.relative_to(ROOT)) for path in outputs),
        "plan_file": str(PLAN_PATH.relative_to(ROOT)),
        "result_file": str(outputs[0].relative_to(ROOT)),
    }
