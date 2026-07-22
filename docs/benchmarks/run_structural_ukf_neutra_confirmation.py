#!/usr/bin/env python3
"""Run transformed HMC and truth-tail diagnostics for structural UKF NeuTra."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_training as common
from docs.benchmarks import run_structural_ukf_neutra_training as training


PLAN_PATH = training.PLAN_PATH
PROBE_RESULTS = 128
PROBE_BURNIN = 64
TUNING_RESULTS = 1000
TUNING_BURNIN = 1000
INITIAL_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.10, -0.10, 0.08, -0.08, 0.06),
    (-0.10, 0.10, -0.08, 0.08, -0.06),
    (0.16, 0.08, -0.12, -0.10, 0.12),
)
PHYSICAL_PARAMETER_NAMES = ("rho", "sigma", "phi", "gamma", "R")
PASS_THRESHOLD = 0.05
SEVERE_THRESHOLD = 0.003
NONCLAIMS = (
    "one frozen noncentral-truth synthetic fixture only",
    "the truth-tail quantity is a posterior-tail diagnostic, not a frequentist p-value",
    "acceptance, short probes, runtime, posterior means, and intervals are descriptive",
    "no UKF exactness, calibration theorem, universal reliability, plain-HMC superiority, or readiness claim",
)


@dataclass(frozen=True)
class KernelProfile:
    profile_id: str
    step_sizes: tuple[float, ...]
    num_leapfrog_steps: int
    tuning_rhat_max: float
    probe_seed: tuple[int, int]
    tuning_verification_seed: tuple[int, int]
    warmup_seed: tuple[int, int]
    retained_seed: tuple[int, int]
    repair_basis: str
    parent_result_path: str | None = None
    parent_result_sha256: str | None = None

    def payload(self) -> Mapping[str, Any]:
        return {
            "profile_id": self.profile_id,
            "step_sizes": self.step_sizes,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "tuning_rhat_max": self.tuning_rhat_max,
            "probe_seed": self.probe_seed,
            "tuning_verification_seed": self.tuning_verification_seed,
            "warmup_seed": self.warmup_seed,
            "retained_seed": self.retained_seed,
            "repair_basis": self.repair_basis,
            "parent_result_path": self.parent_result_path,
            "parent_result_sha256": self.parent_result_sha256,
        }


INITIAL_PROFILE_ID = "initial_l8_grid_v1"
REPAIR_PROFILE_ID = "stable_step_005_l8_tuning_rhat_105_repair_v1"
FINAL_REPAIR_PROFILE_ID = "stable_step_005_l12_tuning_rhat_105_repair_v2"
KERNEL_PROFILES = {
    INITIAL_PROFILE_ID: KernelProfile(
        profile_id=INITIAL_PROFILE_ID,
        step_sizes=(0.02, 0.05, 0.10, 0.20, 0.30, 0.40),
        num_leapfrog_steps=8,
        tuning_rhat_max=1.01,
        probe_seed=(20260717, 42000),
        tuning_verification_seed=(20260717, 42100),
        warmup_seed=(20260717, 42201),
        retained_seed=(20260717, 42301),
        repair_basis="initial_predeclared_step_size_grid",
    ),
    REPAIR_PROFILE_ID: KernelProfile(
        profile_id=REPAIR_PROFILE_ID,
        step_sizes=(0.05,),
        num_leapfrog_steps=8,
        tuning_rhat_max=1.05,
        probe_seed=(20260717, 43000),
        tuning_verification_seed=(20260717, 43100),
        warmup_seed=(20260717, 43201),
        retained_seed=(20260717, 43301),
        repair_basis=(
            "attempt_01_found_step_005_energy_stable_but_rhat_104044; "
            "tuning_admission_uses_warmup_rhat_105_while_retained_gate_stays_101"
        ),
        parent_result_path=(
            "docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/"
            "confirmation/attempt-01/result.json"
        ),
        parent_result_sha256=(
            "622ab339ceffdeb4850b53d9930a372dd9ea9e4ab13a7c59b4401bde10c8ffbe"
        ),
    ),
    FINAL_REPAIR_PROFILE_ID: KernelProfile(
        profile_id=FINAL_REPAIR_PROFILE_ID,
        step_sizes=(0.05,),
        num_leapfrog_steps=12,
        tuning_rhat_max=1.05,
        probe_seed=(20260717, 44000),
        tuning_verification_seed=(20260717, 44100),
        warmup_seed=(20260717, 44201),
        retained_seed=(20260717, 44301),
        repair_basis=(
            "attempt_02_confirmed_step_005_energy_stability_but_rhat_107895; "
            "increase_trajectory_length_from_04_to_06_without_changing_step_size"
        ),
        parent_result_path=(
            "docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/"
            "confirmation/attempt-02/result.json"
        ),
        parent_result_sha256=(
            "b02d22b9cfa51995c02c0df3f686770660d99a1f461600d224134b32ee200eed"
        ),
    ),
}


class StructuralUKFConfirmationError(RuntimeError):
    """Raised when the structural confirmation evidence contract fails."""


def _transport_source(tf: Any, loaded: Any, latent_samples: Any) -> Any:
    latent = tf.convert_to_tensor(latent_samples, tf.float64)
    shape = tf.shape(latent)
    source = loaded.transport.forward_batch(tf.reshape(latent, (-1, training.DIMENSION)))
    return tf.reshape(source, shape)


def _source_physical(tf: Any, source_samples: Any) -> Any:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        structural_source_chart,
    )

    source = tf.convert_to_tensor(source_samples, tf.float64)
    shape = tf.shape(source)
    physical, _derivative = structural_source_chart(
        tf.reshape(source, (-1, training.DIMENSION))
    )
    return tf.reshape(physical, shape)


def _transport_physical(tf: Any, loaded: Any, latent_samples: Any) -> Any:
    return _source_physical(tf, _transport_source(tf, loaded, latent_samples))


def run_confirmation(
    *, output_root: Path, kernel_profile_id: str = INITIAL_PROFILE_ID
) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"confirmation output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    profile = KERNEL_PROFILES[kernel_profile_id]
    _verify_kernel_profile(profile)

    import tensorflow as tf

    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
        rank_normalized_split_rhat_summary,
    )
    from bayesfilter.inference.neutra_campaign import (
        campaign_fixed_transport_adapter,
        load_campaign_neutra_transport,
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_batched_hmc,
        run_sequential_neutra_hmc,
    )

    adapter, identity, identity_reference = training.reconstruct_identity(tf)
    training_result, training_reference, loaded = _load_training(
        identity=identity,
        adapter=adapter,
        load_campaign_neutra_transport=load_campaign_neutra_transport,
    )
    transformed = campaign_fixed_transport_adapter(
        identity=identity, adapter=adapter, loaded_artifact=loaded
    )
    canary = _compiled_canary(tf, transformed, loaded)
    initial_state = tf.constant(INITIAL_OFFSETS, tf.float64)
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
    )

    probe_rows = []
    for grid_index, step_size in enumerate(profile.step_sizes):
        seed = (profile.probe_seed[0], profile.probe_seed[1] + grid_index)
        probe = run_batched_hmc(
            adapter=transformed,
            initial_state=initial_state,
            config=BatchedHMCConfig(
                num_results=PROBE_RESULTS,
                num_burnin_steps=PROBE_BURNIN,
                step_size=step_size,
                num_leapfrog_steps=profile.num_leapfrog_steps,
                seed=seed,
            ),
        )
        physical_samples = _transport_physical(tf, loaded, probe["samples"])
        diagnostic = rank_normalized_hmc_diagnostics(
            physical_samples,
            parameter_names=PHYSICAL_PARAMETER_NAMES,
            thresholds=thresholds,
        )
        health = probe["diagnostics"]
        eligible = bool(
            health["health_passed"] is True
            and diagnostic["input_all_finite"] is True
            and diagnostic["diagnostics_all_finite"] is True
        )
        probe_rows.append(
            {
                "grid_index": grid_index,
                "step_size": step_size,
                "num_leapfrog_steps": profile.num_leapfrog_steps,
                "trajectory_length": step_size * profile.num_leapfrog_steps,
                "seed": seed,
                "eligible": eligible,
                "minimum_bulk_ess": diagnostic.get("min_bulk_ess"),
                "minimum_tail_ess": diagnostic.get("min_tail_ess"),
                "maximum_modern_rhat": diagnostic.get("max_rhat"),
                "acceptance_rate": health["acceptance_rate"],
                "health": health,
                "short_chain_diagnostics": diagnostic,
                "scientific_role": "candidate_ordering_only",
            }
        )
        atomic_write_json(
            output_root / "probe_progress.json",
            {
                "schema": "bayesfilter.structural_ukf_neutra_probe_progress.v1",
                "completed_probe_count": len(probe_rows),
                "total_probe_count": len(profile.step_sizes),
                "kernel_profile": profile.payload(),
                "probe_rows": probe_rows,
                "scientific_role": "checkpoint_only_not_confirmation",
            },
        )

    candidate_order = _ordered_probe_candidates(probe_rows)
    verification_rows = []
    selected = None
    for candidate in candidate_order:
        grid_index = int(candidate["grid_index"])
        seed = (
            profile.tuning_verification_seed[0],
            profile.tuning_verification_seed[1] + grid_index,
        )
        verification = run_batched_hmc(
            adapter=transformed,
            initial_state=initial_state,
            config=BatchedHMCConfig(
                num_results=TUNING_RESULTS,
                num_burnin_steps=TUNING_BURNIN,
                step_size=float(candidate["step_size"]),
                num_leapfrog_steps=profile.num_leapfrog_steps,
                seed=seed,
            ),
        )
        source_samples = _transport_source(tf, loaded, verification["samples"])
        physical_samples = _source_physical(tf, source_samples)
        modern_rhat = rank_normalized_split_rhat_summary(
            physical_samples, rhat_max=profile.tuning_rhat_max
        )
        health = verification["diagnostics"]
        admitted = _tuning_verification_admitted(
            health=health, modern_rhat=modern_rhat
        )
        archive = _archive_triplet(
            tf=tf,
            atomic_write_json=atomic_write_json,
            destination=output_root / "tuning-verification" / f"candidate-{grid_index:04d}",
            stage="tuning_verification",
            latent_samples=verification["samples"],
            source_samples=source_samples,
            physical_samples=physical_samples,
            seed=seed,
            target_signature=identity.target_signature,
            excluded_from_posterior=True,
            cumulative=False,
            chunk_index=None,
        )
        row = {
            "grid_index": grid_index,
            "step_size": float(candidate["step_size"]),
            "seed": seed,
            "probe_nomination_rank": len(verification_rows),
            "health": health,
            "modern_rhat": modern_rhat,
            "archive": archive,
            "admitted": admitted,
            "acceptance_role": "explanatory_only",
            "scientific_role": "fixed_kernel_tuning_admission_only",
        }
        verification_rows.append(row)
        atomic_write_json(
            output_root / "tuning_verification_progress.json",
            {
                "schema": "bayesfilter.structural_ukf_neutra_tuning_progress.v1",
                "completed_candidate_count": len(verification_rows),
                "total_candidate_count": len(candidate_order),
                "kernel_profile": profile.payload(),
                "verification_rows": verification_rows,
            },
        )
        if admitted:
            selected = {**dict(candidate), "tuning_verification": row}
            break
    atomic_write_json(
        output_root / "tuning_selection.json",
        {
            "schema": "bayesfilter.structural_ukf_neutra_tuning_selection.v1",
            "probe_rows": probe_rows,
            "candidate_order": candidate_order,
            "tuning_verification_rows": verification_rows,
            "selected_probe": selected,
            "selection_rule": (
                "healthy_short_probe_lowest_modern_rhat_then_bulk_ess_then_"
                "disjoint_1000_burnin_1000_draw_modern_rhat_first_pass"
            ),
            "tuning_rhat_max": profile.tuning_rhat_max,
            "kernel_profile": profile.payload(),
            "acceptance_role": "explanatory_only",
        },
    )

    sequential = None
    if selected is not None:
        archive = _TripletArchive(
            tf=tf,
            atomic_write_json=atomic_write_json,
            output_root=output_root / "samples",
            loaded=loaded,
            target_signature=identity.target_signature,
        )

        def archive_with_progress(**kwargs: Any) -> Mapping[str, Any]:
            payload = archive(**kwargs)
            atomic_write_json(
                output_root / "sequential_progress.json",
                {
                    "schema": "bayesfilter.structural_ukf_neutra_sequential_progress.v1",
                    "latest_stage": kwargs["stage"],
                    "latest_chunk_index": kwargs["chunk_index"],
                    "latest_seed": kwargs["seed"],
                    "latest_cumulative": kwargs["cumulative"],
                    "latest_archive": payload,
                    "scientific_role": "checkpoint_only_not_confirmation",
                },
            )
            return payload

        def retained_diagnostic(draws: Any) -> Mapping[str, Any]:
            return rank_normalized_hmc_diagnostics(
                draws,
                parameter_names=PHYSICAL_PARAMETER_NAMES,
                thresholds=thresholds,
            )

        sequential = run_sequential_neutra_hmc(
            adapter=transformed,
            initial_state=initial_state,
            model_transform=lambda latent: _transport_physical(tf, loaded, latent),
            parameter_names=PHYSICAL_PARAMETER_NAMES,
            config=SequentialNeuTraHMCConfig(
                step_size=float(selected["step_size"]),
                num_leapfrog_steps=profile.num_leapfrog_steps,
                warmup_seed=profile.warmup_seed,
                retained_seed=profile.retained_seed,
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
            archive_callback=archive_with_progress,
            retained_diagnostic_fn=retained_diagnostic,
        )

    truth_tail = None
    if sequential is not None and sequential["retained_results_per_chain"] > 0:
        truth_tail = _truth_tail_diagnostic(
            tf=tf,
            tfp=tfp,
            physical_samples=sequential["private_retained_raw"],
        )
    classification = _classify_terminal(sequential, truth_tail)
    passed = classification == "NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS"
    public_sequential = (
        None
        if sequential is None
        else {
            key: value
            for key, value in sequential.items()
            if not key.startswith("private_")
        }
    )
    result = {
        "schema": "bayesfilter.structural_ukf_neutra_confirmation.v1",
        "campaign_id": training.CAMPAIGN_ID,
        "cell_id": training.CELL_ID,
        "completed": True,
        "passed": passed,
        "decision": classification,
        "claim_scope": "one_noncentral_truth_frozen_structural_ukf_fixture",
        "target_identity": identity.payload(),
        "identity_reference": identity_reference,
        "training_reference": training_reference,
        "training_result_binding": {
            "recipe_id": training_result["recipe_id"],
            "steps": training_result["steps"],
            "training_state_hash": training_result["training_state_hash"],
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
        },
        "compiled_canary": canary,
        "kernel_grid": {
            "profile": profile.payload(),
            "step_sizes": profile.step_sizes,
            "num_leapfrog_steps": profile.num_leapfrog_steps,
            "tuning_rhat_max": profile.tuning_rhat_max,
            "probe_results": PROBE_RESULTS,
            "probe_burnin": PROBE_BURNIN,
            "tuning_results": TUNING_RESULTS,
            "tuning_burnin": TUNING_BURNIN,
        },
        "probe_rows": probe_rows,
        "tuning_verification_rows": verification_rows,
        "selected_probe": selected,
        "sequential_run": public_sequential,
        "truth_tail": truth_tail,
        "elapsed_seconds": time.monotonic() - started,
        "decision_table": {
            "primary_criterion": "health_valid_converged_hmc_and_all_five_p_truth_at_least_0.05",
            "primary_status": passed,
            "veto_status": classification,
            "main_uncertainty": "single_noncentral_synthetic_fixture",
            "next_justified_action": _next_action(classification),
            "not_concluded": NONCLAIMS,
        },
        "inference_status": {
            "hard_veto_screen": classification,
            "statistically_supported_ranking": False,
            "descriptive_only_differences": "acceptance, runtime, posterior moments, intervals, and per-parameter tail magnitudes",
            "default_readiness": False,
            "next_evidence_needed": _next_evidence(classification),
        },
        "nonclaims": NONCLAIMS,
    }
    common._write_new_json(output_root / "result.json", result)
    common._write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=identity.target_signature,
            training_reference=training_reference,
            kernel_profile=profile,
            wall_time=time.monotonic() - started,
        ),
    )
    _write_result_markdown(output_root / "result.md", result)
    training.write_recursive_hashes(output_root)
    return result


def _verify_kernel_profile(profile: KernelProfile) -> None:
    if not profile.step_sizes or any(step <= 0.0 for step in profile.step_sizes):
        raise StructuralUKFConfirmationError("kernel profile step sizes must be positive")
    if profile.num_leapfrog_steps <= 0:
        raise StructuralUKFConfirmationError("kernel profile leapfrog count must be positive")
    if not 1.0 < profile.tuning_rhat_max <= 1.05:
        raise StructuralUKFConfirmationError("tuning R-hat gate must be in (1, 1.05]")
    seeds = (
        profile.probe_seed,
        profile.tuning_verification_seed,
        profile.warmup_seed,
        profile.retained_seed,
    )
    if len(set(seeds)) != len(seeds) or any(len(seed) != 2 for seed in seeds):
        raise StructuralUKFConfirmationError("kernel profile stage seeds must be disjoint pairs")
    if profile.parent_result_path is not None:
        path = Path(profile.parent_result_path)
        if not path.is_file() or common._file_sha256(path) != profile.parent_result_sha256:
            raise StructuralUKFConfirmationError("kernel repair parent result binding drift")


def _load_training(
    *, identity: Any, adapter: Any, load_campaign_neutra_transport: Any
) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    selection = training.load_selection()
    recipe_id = str(selection["selected_recipe_id"])
    root = training.default_job_root("final", recipe_id)
    result_path = root / "result.json"
    result_hash = common._file_sha256(result_path)
    reference = common._verify_result_root(root, result_hash, require_passed=True)
    result = common._read_mapping(result_path)
    if (
        result.get("job_kind") != "final"
        or result.get("recipe_id") != recipe_id
        or result.get("steps") != training.FINAL_STEPS
        or result.get("seed") != [*training.FINAL_SEED]
        or result.get("screen_weights_reused_by_final") is not False
        or result.get("target_identity", {}).get("target_signature")
        != identity.target_signature
        or result.get("frozen_trainable_parity", {}).get("passed") is not True
        or result.get("heldout_common_batches", {}).get("target_status_all_valid")
        is not True
    ):
        raise StructuralUKFConfirmationError("final training admission failed")
    payload_path = Path(str(result["payload"]["path"]))
    if common._file_sha256(payload_path) != result["payload"]["file_sha256"]:
        raise StructuralUKFConfirmationError("frozen payload hash drift")
    loaded = load_campaign_neutra_transport(
        identity=identity, adapter=adapter, payload=common._read_mapping(payload_path)
    )
    if (
        loaded.artifact_signature != result["transport_artifact_signature"]
        or loaded.manifest.transport_hash != result["transport_hash"]
    ):
        raise StructuralUKFConfirmationError("frozen transport binding drift")
    return result, {**reference, "selection": selection}, loaded


def _compiled_canary(tf: Any, transformed: Any, loaded: Any) -> Mapping[str, Any]:
    probes = tf.constant(INITIAL_OFFSETS, tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(values: Any):
        target_value, target_score = transformed.log_prob_and_grad_batch(values)
        status = transformed.target_status_telemetry(values)
        physical = _transport_physical(tf, loaded, values)
        return (
            target_value,
            target_score,
            physical,
            tf.reduce_all(tf.equal(status["status_code"], 0)),
            tf.reduce_all(status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(probes)
    passed = bool(
        tf.reduce_all(tf.math.is_finite(outputs[0])).numpy()
        and tf.reduce_all(tf.math.is_finite(outputs[1])).numpy()
        and tf.reduce_all(tf.math.is_finite(outputs[2])).numpy()
        and outputs[3].numpy()
        and outputs[4].numpy()
        and all("GPU" in str(item.device).upper() for item in outputs)
    )
    if not passed:
        raise StructuralUKFConfirmationError("compiled transformed canary failed")
    return {
        "passed": True,
        "jit_compile": True,
        "output_devices": tuple(str(item.device) for item in outputs),
        "target_status_all_valid": True,
    }


def _ordered_probe_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    eligible = [row for row in rows if row.get("eligible") is True]
    return tuple(
        dict(row)
        for row in sorted(
            eligible,
            key=lambda row: (
                float(row["maximum_modern_rhat"]),
                -float(row["minimum_bulk_ess"]),
                int(row["grid_index"]),
            ),
        )
    )


def _tuning_verification_admitted(
    *, health: Mapping[str, Any], modern_rhat: Mapping[str, Any]
) -> bool:
    return bool(
        health.get("health_passed") is True
        and modern_rhat.get("input_all_finite") is True
        and modern_rhat.get("diagnostics_all_finite") is True
        and modern_rhat.get("passed") is True
    )


class _TripletArchive:
    def __init__(
        self,
        *,
        tf: Any,
        atomic_write_json: Any,
        output_root: Path,
        loaded: Any,
        target_signature: str,
    ) -> None:
        self.tf = tf
        self.atomic_write_json = atomic_write_json
        self.output_root = output_root
        self.loaded = loaded
        self.target_signature = target_signature

    def __call__(
        self,
        *,
        stage: str,
        chunk_index: int | None,
        latent_samples: Any,
        model_samples: Any,
        seed: tuple[int, int] | None,
        cumulative: bool,
    ) -> Mapping[str, Any]:
        label = "cumulative" if cumulative else f"chunk-{int(chunk_index):04d}"
        source = _transport_source(self.tf, self.loaded, latent_samples)
        physical = self.tf.convert_to_tensor(model_samples, self.tf.float64)
        independently_physical = _source_physical(self.tf, source)
        gap = float(self.tf.reduce_max(self.tf.abs(physical - independently_physical)).numpy())
        if gap > 1.0e-12:
            raise StructuralUKFConfirmationError("archive physical transform mismatch")
        return _archive_triplet(
            tf=self.tf,
            atomic_write_json=self.atomic_write_json,
            destination=self.output_root / stage / label,
            stage=stage,
            latent_samples=latent_samples,
            source_samples=source,
            physical_samples=physical,
            seed=seed,
            target_signature=self.target_signature,
            excluded_from_posterior=(stage != "retained"),
            cumulative=cumulative,
            chunk_index=chunk_index,
        )


def _archive_triplet(
    *,
    tf: Any,
    atomic_write_json: Any,
    destination: Path,
    stage: str,
    latent_samples: Any,
    source_samples: Any,
    physical_samples: Any,
    seed: tuple[int, int] | None,
    target_signature: str,
    excluded_from_posterior: bool,
    cumulative: bool,
    chunk_index: int | None,
) -> Mapping[str, Any]:
    if destination.exists():
        raise FileExistsError(f"archive destination exists: {destination}")
    destination.mkdir(parents=True)
    latent = tf.convert_to_tensor(latent_samples, tf.float64)
    source = tf.convert_to_tensor(source_samples, tf.float64)
    physical = tf.convert_to_tensor(physical_samples, tf.float64)
    if latent.shape != source.shape or source.shape != physical.shape or latent.shape.rank != 3:
        raise StructuralUKFConfirmationError("archive tensors must share [draw, chain, parameter]")
    paths = {
        "latent": destination / "latent.tensor",
        "source": destination / "source.tensor",
        "physical": destination / "physical.tensor",
    }
    tf.io.write_file(str(paths["latent"]), tf.io.serialize_tensor(latent))
    tf.io.write_file(str(paths["source"]), tf.io.serialize_tensor(source))
    tf.io.write_file(str(paths["physical"]), tf.io.serialize_tensor(physical))
    metadata = {
        "schema": "bayesfilter.structural_ukf_neutra_triplet_archive.v1",
        "stage": stage,
        "cumulative": bool(cumulative),
        "chunk_index": chunk_index,
        "seed": seed,
        "target_signature": target_signature,
        "sample_shape": tuple(int(item) for item in latent.shape),
        "latent_path": str(paths["latent"]),
        "source_path": str(paths["source"]),
        "physical_path": str(paths["physical"]),
        "latent_sha256": common._file_sha256(paths["latent"]),
        "source_sha256": common._file_sha256(paths["source"]),
        "physical_sha256": common._file_sha256(paths["physical"]),
        "excluded_from_posterior": bool(excluded_from_posterior),
        "warmup_excluded_from_posterior": True,
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return metadata


def _truth_tail_diagnostic(
    *, tf: Any, tfp: Any, physical_samples: Any
) -> Mapping[str, Any]:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_PARAMETER_LOWER,
        STRUCTURAL_PARAMETER_UPPER,
        STRUCTURAL_TRUTH_PHYSICAL,
    )

    values = tf.convert_to_tensor(physical_samples, tf.float64)
    pooled = tf.reshape(values, (-1, training.DIMENSION))
    truth = tf.convert_to_tensor(STRUCTURAL_TRUTH_PHYSICAL, tf.float64)
    prior_center = 0.5 * (
        tf.convert_to_tensor(STRUCTURAL_PARAMETER_LOWER, tf.float64)
        + tf.convert_to_tensor(STRUCTURAL_PARAMETER_UPPER, tf.float64)
    )
    less = tf.reduce_sum(tf.cast(pooled < truth[None, :], tf.float64), axis=0)
    equal = tf.reduce_sum(tf.cast(pooled == truth[None, :], tf.float64), axis=0)
    count = tf.cast(tf.shape(pooled)[0], tf.float64)
    cdf = (less + 0.5 * equal + 0.5) / (count + 1.0)
    p_truth = 2.0 * tf.minimum(cdf, 1.0 - cdf)
    mean = tf.reduce_mean(pooled, axis=0)
    sd = tf.math.reduce_std(pooled, axis=0)
    interval = tfp.stats.percentile(
        pooled, (2.5, 97.5), axis=0, interpolation="linear"
    )
    rows = []
    for index, name in enumerate(PHYSICAL_PARAMETER_NAMES):
        current = float(p_truth[index].numpy())
        status = (
            "PASS"
            if current >= PASS_THRESHOLD
            else ("MARGINAL_RERUN" if current >= SEVERE_THRESHOLD else "SEVERE_FAILURE")
        )
        rows.append(
            {
                "parameter": name,
                "truth": float(truth[index].numpy()),
                "prior_center": float(prior_center[index].numpy()),
                "truth_at_prior_center": bool(truth[index].numpy() == prior_center[index].numpy()),
                "posterior_mean": float(mean[index].numpy()),
                "posterior_sd": float(sd[index].numpy()),
                "empirical_95_interval": (
                    float(interval[0, index].numpy()),
                    float(interval[1, index].numpy()),
                ),
                "truth_in_empirical_95_interval": bool(
                    interval[0, index] <= truth[index]
                    and truth[index] <= interval[1, index]
                ),
                "count_less_than_truth": int(less[index].numpy()),
                "count_equal_to_truth": int(equal[index].numpy()),
                "smoothed_ecdf_at_truth": float(cdf[index].numpy()),
                "p_truth": current,
                "tail_status": status,
            }
        )
    minimum = min(float(row["p_truth"]) for row in rows)
    return {
        "schema": "bayesfilter.structural_ukf_truth_tail.v1",
        "passed": minimum >= PASS_THRESHOLD,
        "fully_central_truth": False,
        "central_truth_parameter_count": 0,
        "draw_count_total": int(tf.shape(pooled)[0].numpy()),
        "chain_count": int(values.shape[1]),
        "results_per_chain": int(values.shape[0]),
        "definition": "F=(n_less+0.5*n_equal+0.5)/(N+1); p_truth=2*min(F,1-F)",
        "interpretation": "posterior-tail diagnostic_not_frequentist_p_value",
        "pass_threshold": PASS_THRESHOLD,
        "severe_threshold": SEVERE_THRESHOLD,
        "minimum_p_truth": minimum,
        "parameter_rows": rows,
        "marginal_parameters": tuple(
            row["parameter"]
            for row in rows
            if SEVERE_THRESHOLD <= float(row["p_truth"]) < PASS_THRESHOLD
        ),
        "severe_parameters": tuple(
            row["parameter"]
            for row in rows
            if float(row["p_truth"]) < SEVERE_THRESHOLD
        ),
    }


def _classify_terminal(
    sequential: Mapping[str, Any] | None, truth_tail: Mapping[str, Any] | None
) -> str:
    if sequential is None:
        return "SAMPLER_BLOCKED_NO_TUNING_ADMISSION"
    if sequential.get("hard_vetoes"):
        return "SAMPLER_BLOCKED_HEALTH_OR_TARGET_STATUS"
    if sequential.get("warmup_passed") is not True:
        return "SAMPLER_BLOCKED_WARMUP"
    if sequential.get("retained_passed") is not True:
        return "SAMPLER_BLOCKED_RETAINED_CONVERGENCE"
    if not isinstance(truth_tail, Mapping):
        return "EVIDENCE_BLOCKED_NO_TRUTH_TAIL"
    if truth_tail.get("severe_parameters"):
        return "SEVERE_TRUTH_TAIL_FAILURE"
    if truth_tail.get("marginal_parameters"):
        return "MARGINAL_TRUTH_TAIL_REQUIRES_SECOND_SEED"
    if truth_tail.get("passed") is True:
        return "NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS"
    return "EVIDENCE_BLOCKED_TRUTH_TAIL_CLASSIFICATION"


def _next_action(classification: str) -> str:
    if classification == "NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS":
        return "document_one_seed_structural_result_and_close_first_seed_campaign"
    if classification == "MARGINAL_TRUTH_TAIL_REQUIRES_SECOND_SEED":
        return "run_exactly_one_fresh_data_seed_and_check_whether_same_parameter_fails"
    if classification == "SEVERE_TRUTH_TAIL_FAILURE":
        return "investigate_target_training_and_sampler_before_any_replication_claim"
    return "repair_or_diagnose_sampler_without_interpreting_truth_tails"


def _next_evidence(classification: str) -> str:
    if classification == "NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS":
        return "additional_fixtures_only_for_broader_calibration_or_reliability_claims"
    if classification == "MARGINAL_TRUTH_TAIL_REQUIRES_SECOND_SEED":
        return "one_predeclared_fresh_data_seed"
    return "valid_converged_retained_hmc_and_nonextreme_truth_tails"


def _run_manifest(
    *,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str,
    training_reference: Mapping[str, Any],
    kernel_profile: KernelProfile,
    wall_time: float,
) -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.structural_ukf_neutra_confirmation_manifest.v1",
        "campaign_id": training.CAMPAIGN_ID,
        "git_commit": commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped new paths only",
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "visible_physical_gpus": [str(item) for item in memory_policy.get("physical_devices", ())],
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "target_signature": target_signature,
        "data_seed": [20260716, 15001],
        "kernel_profile": kernel_profile.payload(),
        "hmc_seeds": {
            "probe": kernel_profile.probe_seed,
            "tuning_verification": kernel_profile.tuning_verification_seed,
            "warmup": kernel_profile.warmup_seed,
            "retained": kernel_profile.retained_seed,
        },
        "training_reference": training_reference,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    truth = result.get("truth_tail")
    lines = [
        "# Structural UKF NeuTra confirmation",
        "",
        f"Decision: `{result['decision']}`",
        "",
    ]
    if isinstance(truth, Mapping):
        lines.extend(
            [
                f"Minimum posterior truth-tail value: `{truth['minimum_p_truth']:.8g}`.",
                "",
                "| Parameter | Truth | Mean | 95% interval | p_truth | Status |",
                "| --- | ---: | ---: | --- | ---: | --- |",
            ]
        )
        for row in truth["parameter_rows"]:
            interval = row["empirical_95_interval"]
            lines.append(
                f"| `{row['parameter']}` | {row['truth']:.8g} | "
                f"{row['posterior_mean']:.8g} | [{interval[0]:.8g}, {interval[1]:.8g}] | "
                f"{row['p_truth']:.8g} | `{row['tail_status']}` |"
            )
    lines.extend(
        [
            "",
            "This is a one-fixture posterior-tail diagnostic, not a frequentist p-value or calibration theorem.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--kernel-profile",
        choices=tuple(KERNEL_PROFILES),
        default=INITIAL_PROFILE_ID,
    )
    args = parser.parse_args(argv)
    result = run_confirmation(
        output_root=args.output_root, kernel_profile_id=args.kernel_profile
    )
    print(
        json.dumps(
            {
                "cell_id": result["cell_id"],
                "decision": result["decision"],
                "passed": result["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
