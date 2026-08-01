#!/usr/bin/env python3
"""Confirm P4 predator-prey frozen NeuTra transports with same-target HMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base
from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_training as training
from docs.benchmarks import run_multimodel_neutra_p6_sir_sgqf_hmc as tuning_hmc


PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p4-r4-tuning-admission-repair-subplan-2026-07-16.md"
)
CELLS = ("PP-UKF", "PP-SGQF")
TRAINING_ROOTS = {
    cell: base.PHASE_ROOT
    / cell
    / "training/final/wide_lr5e3/attempt-01"
    for cell in CELLS
}
EXPECTED_TRAINING_RESULT_SHA256 = {
    "PP-UKF": "1650d256577f91d54e6c351545e9a7ef0cb208844dc859f19eecc3b496af27c9",
    "PP-SGQF": "de5f7cc35f606fe6d07177d1059d24acc1187e80b4bda42963f9e2823bf64bd4",
}
STEP_SIZES = (0.025, 0.05, 0.10, 0.20, 0.40, 0.80)
NUM_LEAPFROG_STEPS = 10
PROBE_RESULTS = 64
PROBE_BURNIN = 128
INITIAL_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.10, -0.10, 0.08, -0.08, 0.06, -0.06),
    (-0.10, 0.10, -0.08, 0.08, -0.06, 0.06),
    (0.16, 0.08, -0.12, -0.10, 0.12, 0.04),
)
PROBE_SEEDS = {"PP-UKF": (20260716, 12100), "PP-SGQF": (20260716, 13100)}
PROBE_SOURCE_ROOTS = {
    "PP-UKF": base.PHASE_ROOT / "PP-UKF/neutra-confirmation/attempt-02",
    "PP-SGQF": base.PHASE_ROOT / "PP-SGQF/neutra-confirmation/attempt-01",
}
PROBE_SOURCE_RESULT_SHA256 = {
    "PP-UKF": "c69117dcdc378623f054742954bf43bfa9af60a3601b0df548960369b9433375",
    "PP-SGQF": "2fb38a2e5727ab486de8f5840c50881331cdcbd0352b97710260d5e72f4fe50e",
}
TUNING_VERIFICATION_SEEDS = {
    "PP-UKF": (20260716, 42100),
    "PP-SGQF": (20260716, 43100),
}
WARMUP_SEEDS = {"PP-UKF": (20260716, 42201), "PP-SGQF": (20260716, 43201)}
RETAINED_SEEDS = {"PP-UKF": (20260716, 42301), "PP-SGQF": (20260716, 43301)}
PHYSICAL_PARAMETER_NAMES = ("r", "K", "a", "s", "u", "v")
AGREEMENT_ALPHA = 0.05
AGREEMENT_MARGIN_SD_FRACTION = 0.10
NONCLAIMS = (
    "same-target physical-posterior-mean agreement only",
    "no full-distribution, covariance, tail, or mode equivalence claim",
    "acceptance and short-probe diagnostics are nomination or explanatory only",
    "no filter exactness, superiority, calibration, robustness, or readiness claim",
)


class P4NeuTraConfirmationError(RuntimeError):
    """Raised when the frozen R4 evidence contract cannot be replayed."""


def run_confirmation(*, cell_id: str, output_root: Path) -> Mapping[str, Any]:
    cell = _cell(cell_id)
    if output_root.exists():
        raise FileExistsError(f"R4 output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

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
        CampaignCellLedger,
        SeparateCampaignArchive,
        campaign_fixed_transport_adapter,
        load_campaign_neutra_transport,
        run_campaign_neutra_hmc,
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_batched_hmc,
    )

    adapter, identity, identity_reference = training._reconstruct_identity(tf, cell)
    training_result, training_reference, loaded = _load_training(
        cell=cell,
        identity=identity,
        adapter=adapter,
        load_campaign_neutra_transport=load_campaign_neutra_transport,
    )
    comparator, comparator_reference, comparator_samples = _load_comparator(
        tf=tf, cell=cell, expected_target_signature=identity.target_signature
    )
    transformed = campaign_fixed_transport_adapter(
        identity=identity, adapter=adapter, loaded_artifact=loaded
    )
    canary = _compiled_canary(tf, transformed, cell=cell)
    initial_state = tf.constant(INITIAL_OFFSETS, tf.float64)
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
    )

    probe_rows, probe_source_reference = _load_probe_source(cell, identity.target_signature)
    candidate_order = tuning_hmc._ordered_probe_candidates(probe_rows)
    verification_rows = []
    selected = None
    for candidate in candidate_order:
        grid_index = int(candidate["grid_index"])
        seed = (
            TUNING_VERIFICATION_SEEDS[cell][0],
            TUNING_VERIFICATION_SEEDS[cell][1] + grid_index,
        )
        verification = run_batched_hmc(
            adapter=transformed,
            initial_state=initial_state,
            config=BatchedHMCConfig(
                num_results=1000,
                num_burnin_steps=1000,
                step_size=float(candidate["step_size"]),
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=seed,
            ),
        )
        source_samples = _transport_samples(tf, loaded, verification["samples"])
        modern_rhat = rank_normalized_split_rhat_summary(
            source_samples, rhat_max=1.01
        )
        health = verification["diagnostics"]
        admitted = tuning_hmc._tuning_verification_admitted(
            health=health, modern_rhat=modern_rhat
        )
        archive = _archive_tuning_verification(
            tf=tf,
            atomic_write_json=atomic_write_json,
            output_root=output_root / "tuning-verification",
            candidate=candidate,
            latent_samples=verification["samples"],
            model_samples=source_samples,
            seed=seed,
            target_signature=identity.target_signature,
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
                "schema": "bayesfilter.multimodel_neutra_p4_r4_tuning_verification_progress.v2",
                "cell_id": cell,
                "completed_candidate_count": len(verification_rows),
                "total_candidate_count": len(candidate_order),
                "verification_rows": verification_rows,
                "scientific_role": "checkpoint_only_not_confirmation",
            },
        )
        if admitted:
            selected = {**dict(candidate), "tuning_verification": row}
            break
    atomic_write_json(
        output_root / "tuning_selection.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p4_r4_tuning_selection.v2",
            "cell_id": cell,
            "probe_rows": probe_rows,
            "probe_source_reference": probe_source_reference,
            "candidate_order": candidate_order,
            "tuning_verification_rows": verification_rows,
            "selected_probe": selected,
            "selection_rule": "hash_verified_short_probe_lowest_modern_rhat_then_bulk_ess_order_then_fresh_disjoint_1000_burnin_1000_draw_modern_rhat_at_most_1.01_first_pass",
            "acceptance_role": "explanatory_only",
        },
    )

    sequential = None
    if selected is not None:
        archive = SeparateCampaignArchive(
            output_root=output_root / "samples", identity=identity, adapter=adapter
        )

        def archive_with_progress(**kwargs: Any) -> Mapping[str, Any]:
            payload = archive(**kwargs)
            atomic_write_json(
                output_root / "sequential_progress.json",
                {
                    "schema": "bayesfilter.multimodel_neutra_p4_r4_sequential_progress.v1",
                    "cell_id": cell,
                    "latest_stage": kwargs["stage"],
                    "latest_chunk_index": kwargs["chunk_index"],
                    "latest_seed": kwargs["seed"],
                    "latest_cumulative": kwargs["cumulative"],
                    "latest_archive": payload,
                    "scientific_role": "checkpoint_only_not_confirmation",
                },
            )
            return payload

        def joint_retained_diagnostic(draws: Any) -> Mapping[str, Any]:
            convergence = rank_normalized_hmc_diagnostics(
                draws,
                parameter_names=_source_parameter_names(),
                thresholds=thresholds,
            )
            agreement = _physical_mean_agreement(
                tf=tf,
                tfp=tfp,
                candidate_source_samples=draws,
                comparator_source_samples=comparator_samples,
            )
            return {
                "schema": "bayesfilter.multimodel_neutra_p4_r4_joint_diagnostic.v1",
                "passed": bool(convergence["passed"] and agreement["passed"]),
                "convergence": convergence,
                "physical_mean_agreement": agreement,
                "stop_rule": "convergence_and_simultaneous_physical_mean_equivalence",
            }

        sequential = run_campaign_neutra_hmc(
            identity=identity,
            adapter=adapter,
            loaded_artifact=loaded,
            initial_state=initial_state,
            parameter_names=_source_parameter_names(),
            config=SequentialNeuTraHMCConfig(
                step_size=float(selected["step_size"]),
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                warmup_seed=WARMUP_SEEDS[cell],
                retained_seed=RETAINED_SEEDS[cell],
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
            retained_diagnostic_fn=joint_retained_diagnostic,
        )

    public_sequential = (
        None
        if sequential is None
        else {
            key: value
            for key, value in sequential.items()
            if not key.startswith("private_")
        }
    )
    final_joint = _final_joint_diagnostic(sequential)
    classification = _classify_terminal(sequential, final_joint)
    passed = classification == "NEUTRA_CONFIRMED"
    posterior = (
        None
        if sequential is None or sequential["retained_results_per_chain"] == 0
        else {
            "source": base._posterior_summary(
                tf,
                sequential["private_retained_raw"],
                _source_parameter_names(),
            ),
            "physical_mean_agreement": (
                None
                if final_joint is None
                else final_joint.get("physical_mean_agreement")
            ),
        }
    )
    ledger = CampaignCellLedger(
        {
            "cells": [
                {
                    "cell_id": cell,
                    "state": "TRAINING_ADMITTED",
                    "target_signature": identity.mathematical_target_signature,
                }
            ]
        },
        required_candidate_families=("plain_dense_iaf",),
        event_path=output_root / "cell_events.jsonl",
    )
    ledger_state = "NEUTRA_CONFIRMED" if passed else (
        "EVIDENCE_BLOCKED"
        if classification == "EVIDENCE_BLOCKED_AGREEMENT_PRECISION"
        else "SAMPLER_BLOCKED"
    )
    ledger.transition(
        cell_id=cell,
        new_state=ledger_state,
        evidence_path=str(output_root / "result.json"),
        target_identity=identity if passed else None,
    )
    base._write_new_json(output_root / "cell_ledger.json", ledger.payload())
    result = {
        "schema": "bayesfilter.multimodel_neutra_p4_r4_confirmation.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": cell,
        "completed": True,
        "passed": passed,
        "decision": classification,
        "terminal_state": ledger_state,
        "claim_scope": "six_physical_posterior_means_only",
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
        "comparator_reference": comparator_reference,
        "comparator_decision": comparator["decision"],
        "compiled_canary": canary,
        "kernel_grid": {
            "step_sizes": STEP_SIZES,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "probe_results": PROBE_RESULTS,
            "probe_burnin": PROBE_BURNIN,
        },
        "probe_rows": probe_rows,
        "probe_source_reference": probe_source_reference,
        "tuning_verification_rows": verification_rows,
        "selected_probe": selected,
        "sequential_run": public_sequential,
        "final_joint_diagnostic": final_joint,
        "posterior_summary": posterior,
        "elapsed_seconds": time.monotonic() - started,
        "decision_table": {
            "primary_criterion": "sampler_convergence_and_simultaneous_physical_mean_equivalence",
            "primary_status": passed,
            "veto_status": classification,
            "main_uncertainty": "single_fixture_mean_level_comparison",
            "next_justified_action": "close_cell_or_investigate_terminal_classification",
            "not_concluded": NONCLAIMS,
        },
        "inference_status": {
            "hard_veto_screen": classification,
            "statistically_supported_ranking": False,
            "descriptive_only_differences": "source summaries, acceptance, runtime, and cross-filter differences",
            "default_readiness": False,
            "next_evidence_needed": "additional distributional estimands and repeated fixtures for broader claims",
        },
        "nonclaims": NONCLAIMS,
    }
    base._write_new_json(output_root / "result.json", result)
    base._write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            cell=cell,
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=identity.target_signature,
            training_result_sha256=EXPECTED_TRAINING_RESULT_SHA256[cell],
            wall_time=time.monotonic() - started,
        ),
    )
    training._write_recursive_hashes(output_root)
    return result


def _load_training(
    *, cell: str, identity: Any, adapter: Any, load_campaign_neutra_transport: Any
) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    root = TRAINING_ROOTS[cell]
    expected = EXPECTED_TRAINING_RESULT_SHA256[cell]
    if len(expected) != 64:
        raise P4NeuTraConfirmationError(f"{cell} final training hash is not frozen")
    reference = training._verify_result_root(root, expected, require_passed=True)
    result = training._read_mapping(root / "result.json")
    if (
        result.get("job_kind") != "final"
        or result.get("recipe_id") != "wide_lr5e3"
        or result.get("steps") != 5000
        or result.get("screen_weights_reused_by_final") is not False
        or result.get("target_identity", {}).get("target_signature")
        != identity.target_signature
        or result.get("frozen_trainable_parity", {}).get("passed") is not True
        or result.get("heldout_common_batches", {}).get("target_status_all_valid")
        is not True
    ):
        raise P4NeuTraConfirmationError("final training admission fields failed")
    payload_path = Path(str(result["payload"]["path"]))
    if training._file_sha256(payload_path) != result["payload"]["file_sha256"]:
        raise P4NeuTraConfirmationError("frozen transport payload hash drift")
    payload = training._read_mapping(payload_path)
    loaded = load_campaign_neutra_transport(
        identity=identity, adapter=adapter, payload=payload
    )
    if (
        _bare_sha256(loaded.manifest.training_state_hash, "loaded training state hash")
        != _bare_sha256(result["training_state_hash"], "result training state hash")
        or loaded.artifact_signature != result["transport_artifact_signature"]
        or loaded.manifest.transport_hash != result["transport_hash"]
    ):
        raise P4NeuTraConfirmationError("frozen transport result binding drift")
    return result, reference, loaded


def _load_probe_source(
    cell: str, expected_target_signature: str
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    root = PROBE_SOURCE_ROOTS[cell]
    reference = training._verify_result_root(
        root, PROBE_SOURCE_RESULT_SHA256[cell], require_passed=True
    )
    result = training._read_mapping(root / "result.json")
    kernel_grid = result.get("kernel_grid", {})
    training_binding = result.get("training_result_binding", {})
    comparator_reference = result.get("comparator_reference", {})
    probe_rows = result.get("probe_rows")
    if (
        result.get("cell_id") != cell
        or result.get("target_identity", {}).get("target_signature")
        != expected_target_signature
        or tuple(kernel_grid.get("step_sizes", ())) != STEP_SIZES
        or kernel_grid.get("num_leapfrog_steps") != NUM_LEAPFROG_STEPS
        or kernel_grid.get("probe_results") != PROBE_RESULTS
        or kernel_grid.get("probe_burnin") != PROBE_BURNIN
        or training_binding.get("transport_hash")
        != training._read_mapping(TRAINING_ROOTS[cell] / "result.json").get(
            "transport_hash"
        )
        or comparator_reference.get("result_sha256")
        != training.EXPECTED_COMPARATOR_HASHES[cell]
        or not isinstance(probe_rows, list)
        or len(probe_rows) != len(STEP_SIZES)
    ):
        raise P4NeuTraConfirmationError("probe-source contract binding failed")
    for grid_index, row in enumerate(probe_rows):
        expected_seed = (
            PROBE_SEEDS[cell][0],
            PROBE_SEEDS[cell][1] + grid_index,
        )
        if (
            row.get("grid_index") != grid_index
            or float(row.get("step_size")) != STEP_SIZES[grid_index]
            or tuple(row.get("seed", ())) != expected_seed
            or row.get("eligible") is not True
            or row.get("health", {}).get("health_passed") is not True
            or row.get("short_chain_diagnostics", {}).get("input_all_finite")
            is not True
            or row.get("short_chain_diagnostics", {}).get(
                "diagnostics_all_finite"
            )
            is not True
        ):
            raise P4NeuTraConfirmationError(
                f"probe-source row {grid_index} failed ordering-only admission"
            )
    return tuple(dict(row) for row in probe_rows), {
        **reference,
        "scientific_role": "hash_verified_candidate_ordering_only",
        "old_warmup_and_retained_samples_reused": False,
    }


def _archive_tuning_verification(
    *,
    tf: Any,
    atomic_write_json: Any,
    output_root: Path,
    candidate: Mapping[str, Any],
    latent_samples: Any,
    model_samples: Any,
    seed: tuple[int, int],
    target_signature: str,
) -> Mapping[str, Any]:
    destination = output_root / f"candidate-{int(candidate['grid_index']):04d}"
    if destination.exists():
        raise FileExistsError(f"tuning archive destination exists: {destination}")
    destination.mkdir(parents=True)
    latent = tf.convert_to_tensor(latent_samples, tf.float64)
    model = tf.convert_to_tensor(model_samples, tf.float64)
    if latent.shape != model.shape or latent.shape.rank != 3:
        raise P4NeuTraConfirmationError(
            "tuning archives must share [draw, chain, parameter] shape"
        )
    latent_path = destination / "latent.tensor"
    model_path = destination / "model.tensor"
    tf.io.write_file(str(latent_path), tf.io.serialize_tensor(latent))
    tf.io.write_file(str(model_path), tf.io.serialize_tensor(model))
    metadata = {
        "schema": "bayesfilter.multimodel_neutra_p4_tuning_verification_archive.v2",
        "grid_index": int(candidate["grid_index"]),
        "step_size": float(candidate["step_size"]),
        "seed": seed,
        "target_signature": target_signature,
        "sample_shape": tuple(int(item) for item in latent.shape),
        "latent_path": str(latent_path),
        "model_path": str(model_path),
        "excluded_from_posterior": True,
        "scientific_role": "fixed_kernel_tuning_admission_only",
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return metadata


def _load_comparator(
    *, tf: Any, cell: str, expected_target_signature: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    root = training.COMPARATOR_ROOTS[cell]
    expected = training.EXPECTED_COMPARATOR_HASHES[cell]
    reference = training._verify_result_root(root, expected, require_passed=True)
    result = training._read_mapping(root / "result.json")
    if (
        result.get("terminal_state") != "COMPARATOR_ADMITTED"
        or result.get("target_identity", {}).get("target_signature")
        != expected_target_signature
        or result.get("sequential_run", {}).get("warmup_excluded_from_posterior")
        is not True
    ):
        raise P4NeuTraConfirmationError("plain-HMC comparator binding failed")
    metadata_path = root / "samples/retained/cumulative/metadata.json"
    metadata = training._read_mapping(metadata_path)
    if (
        metadata.get("stage") != "retained"
        or metadata.get("cumulative") is not True
        or metadata.get("target_signature") != expected_target_signature
    ):
        raise P4NeuTraConfirmationError("comparator retained archive metadata failed")
    tensor_path = Path(str(metadata["model_path"]))
    draws = tf.io.parse_tensor(tf.io.read_file(str(tensor_path)), out_type=tf.float64)
    draws = tf.ensure_shape(draws, (4000, 4, 6))
    if not bool(tf.reduce_all(tf.math.is_finite(draws)).numpy()):
        raise P4NeuTraConfirmationError("comparator retained archive is nonfinite")
    return result, {**reference, "retained_metadata": _file_reference(metadata_path)}, draws


def _compiled_canary(tf: Any, transformed: Any, *, cell: str) -> Mapping[str, Any]:
    probes = tf.constant(INITIAL_OFFSETS, tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(values):
        target_value, target_score = transformed.log_prob_and_grad_batch(values)
        status = transformed.target_status_telemetry(values)
        return (
            target_value,
            target_score,
            tf.reduce_all(tf.equal(status["status_code"], 0)),
            tf.reduce_all(status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(probes)
    passed = bool(
        tf.reduce_all(tf.math.is_finite(outputs[0])).numpy()
        and tf.reduce_all(tf.math.is_finite(outputs[1])).numpy()
        and outputs[2].numpy()
        and outputs[3].numpy()
        and all("GPU" in str(item.device).upper() for item in outputs)
    )
    if not passed:
        raise P4NeuTraConfirmationError(f"{cell} compiled transported canary failed")
    return {
        "passed": True,
        "jit_compile": True,
        "output_devices": tuple(str(item.device) for item in outputs),
        "target_status_all_valid": True,
    }


def _transport_samples(tf: Any, loaded: Any, latent_samples: Any) -> Any:
    latent = tf.convert_to_tensor(latent_samples, tf.float64)
    shape = tf.shape(latent)
    source = loaded.transport.forward_batch(tf.reshape(latent, (-1, 6)))
    return tf.reshape(source, shape)


def _physical_mean_agreement(
    *, tf: Any, tfp: Any, candidate_source_samples: Any, comparator_source_samples: Any
) -> Mapping[str, Any]:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        source_chart_physical_parameters,
    )

    candidate_source = tf.convert_to_tensor(candidate_source_samples, tf.float64)
    comparator_source = tf.convert_to_tensor(comparator_source_samples, tf.float64)
    candidate_physical, _ = source_chart_physical_parameters(
        tf.reshape(candidate_source, (-1, 6))
    )
    comparator_physical, _ = source_chart_physical_parameters(
        tf.reshape(comparator_source, (-1, 6))
    )
    candidate_physical = tf.reshape(candidate_physical, tf.shape(candidate_source))
    comparator_physical = tf.reshape(comparator_physical, tf.shape(comparator_source))
    candidate = _mean_statistics(tf, tfp, candidate_physical)
    comparator = _mean_statistics(tf, tfp, comparator_physical)
    difference = tf.abs(candidate["mean"] - comparator["mean"])
    combined_mcse = tf.sqrt(tf.square(candidate["mcse"]) + tf.square(comparator["mcse"]))
    normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
    )
    critical = normal.quantile(
        tf.constant(1.0 - AGREEMENT_ALPHA / (2.0 * len(PHYSICAL_PARAMETER_NAMES)), tf.float64)
    )
    lower = tf.maximum(tf.constant(0.0, tf.float64), difference - critical * combined_mcse)
    upper = difference + critical * combined_mcse
    margin = AGREEMENT_MARGIN_SD_FRACTION * comparator["sd"]
    parameter_pass = upper <= margin
    supported_disagreement = lower > margin
    unresolved = tf.logical_not(tf.logical_or(parameter_pass, supported_disagreement))
    rows = tuple(
        {
            "parameter": name,
            "neutra_mean": float(candidate["mean"][index].numpy()),
            "neutra_sd": float(candidate["sd"][index].numpy()),
            "neutra_mean_ess": float(candidate["ess"][index].numpy()),
            "neutra_mean_mcse": float(candidate["mcse"][index].numpy()),
            "plain_hmc_mean": float(comparator["mean"][index].numpy()),
            "plain_hmc_sd": float(comparator["sd"][index].numpy()),
            "plain_hmc_mean_ess": float(comparator["ess"][index].numpy()),
            "plain_hmc_mean_mcse": float(comparator["mcse"][index].numpy()),
            "absolute_mean_difference": float(difference[index].numpy()),
            "simultaneous_lower_bound": float(lower[index].numpy()),
            "simultaneous_upper_bound": float(upper[index].numpy()),
            "practical_margin": float(margin[index].numpy()),
            "passed": bool(parameter_pass[index].numpy()),
            "supported_disagreement": bool(supported_disagreement[index].numpy()),
            "unresolved_precision": bool(unresolved[index].numpy()),
        }
        for index, name in enumerate(PHYSICAL_PARAMETER_NAMES)
    )
    return {
        "schema": "bayesfilter.multimodel_neutra_p4_physical_mean_agreement.v1",
        "passed": bool(tf.reduce_all(parameter_pass).numpy()),
        "supported_disagreement": bool(tf.reduce_any(supported_disagreement).numpy()),
        "unresolved_precision": bool(tf.reduce_any(unresolved).numpy()),
        "physical_parameter_names": PHYSICAL_PARAMETER_NAMES,
        "familywise_alpha": AGREEMENT_ALPHA,
        "critical_value": float(critical.numpy()),
        "multiplicity_control": "bonferroni_six_two_sided_normal_mcse_intervals",
        "margin_definition": "0.10_times_same_estimand_plain_hmc_posterior_sd",
        "margin_sd_fraction": AGREEMENT_MARGIN_SD_FRACTION,
        "mean_mcse_definition": "posterior_sample_sd_divided_by_sqrt_split_chain_cross_chain_ess",
        "parameter_rows": rows,
        "candidate_quantiles": _json_tensor(candidate["quantiles"]),
        "comparator_quantiles": _json_tensor(comparator["quantiles"]),
        "candidate_correlation": _json_tensor(candidate["correlation"]),
        "comparator_correlation": _json_tensor(comparator["correlation"]),
        "distributional_diagnostics_role": "explanatory_only_not_equivalence_gate",
        "claim_scope": "six_physical_posterior_means_only",
        "nonclaims": NONCLAIMS,
    }


def _mean_statistics(tf: Any, tfp: Any, samples: Any) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(samples, tf.float64)
    draw_count = int(values.shape[0])
    chain_count = int(values.shape[1])
    half = draw_count // 2
    split = tf.reshape(
        tf.stack((values[:half], values[-half:]), axis=2),
        (half, 2 * chain_count, 6),
    )
    ess = tfp.mcmc.effective_sample_size(
        split, filter_beyond_positive_pairs=True, cross_chain_dims=1
    )
    pooled = tf.reshape(values, (-1, 6))
    mean = tf.reduce_mean(pooled, axis=0)
    centered = pooled - mean[None, :]
    sample_count = tf.cast(tf.shape(pooled)[0], tf.float64)
    covariance = tf.matmul(centered, centered, transpose_a=True) / (sample_count - 1.0)
    sd = tf.sqrt(tf.linalg.diag_part(covariance))
    correlation = covariance / (sd[:, None] * sd[None, :])
    mcse = sd / tf.sqrt(ess)
    quantiles = tfp.stats.percentile(
        pooled, (5.0, 50.0, 95.0), axis=0, interpolation="linear"
    )
    finite = tf.reduce_all(
        tf.stack(
            (
                tf.reduce_all(tf.math.is_finite(mean)),
                tf.reduce_all(tf.math.is_finite(sd)),
                tf.reduce_all(tf.math.is_finite(ess)),
                tf.reduce_all(tf.math.is_finite(mcse)),
            )
        )
    )
    if not bool(finite.numpy()) or not bool(tf.reduce_all(ess > 0.0).numpy()):
        raise P4NeuTraConfirmationError("physical mean ESS/MCSE is invalid")
    return {
        "mean": mean,
        "sd": sd,
        "ess": ess,
        "mcse": mcse,
        "quantiles": quantiles,
        "correlation": correlation,
    }


def _final_joint_diagnostic(sequential: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if sequential is None or not sequential.get("retained_checks"):
        return None
    return sequential["retained_checks"][-1].get("full_convergence")


def _classify_terminal(
    sequential: Mapping[str, Any] | None, final_joint: Mapping[str, Any] | None
) -> str:
    if sequential is None:
        return "SAMPLER_BLOCKED_NO_TUNING_ADMISSION"
    if sequential.get("hard_vetoes"):
        return "SAMPLER_BLOCKED_HEALTH_OR_TARGET_STATUS"
    if sequential.get("warmup_passed") is not True:
        return "SAMPLER_BLOCKED_WARMUP"
    if not isinstance(final_joint, Mapping):
        return "SAMPLER_BLOCKED_NO_RETAINED_DIAGNOSTIC"
    convergence = final_joint.get("convergence", {})
    agreement = final_joint.get("physical_mean_agreement", {})
    if convergence.get("passed") is not True:
        return "SAMPLER_BLOCKED_RETAINED_CONVERGENCE"
    if agreement.get("passed") is True and sequential.get("passed") is True:
        return "NEUTRA_CONFIRMED"
    if agreement.get("supported_disagreement") is True:
        return "SAMPLER_BLOCKED_SAME_TARGET_MEAN_DISAGREEMENT"
    return "EVIDENCE_BLOCKED_AGREEMENT_PRECISION"


def _source_parameter_names() -> tuple[str, ...]:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_PARAMETER_NAMES,
    )

    return tuple(PP_PARAMETER_NAMES)


def _run_manifest(
    *,
    cell: str,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str,
    training_result_sha256: str,
    wall_time: float,
) -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p4_r4_manifest.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": cell,
        "git_commit": commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_neutra_confirmation.py "
            f"--cell {cell} --output-root {output_root}"
        ),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "target_signature": target_signature,
        "training_result_sha256": training_result_sha256,
        "seeds": {
            "probe": PROBE_SEEDS[cell],
            "tuning_verification": TUNING_VERIFICATION_SEEDS[cell],
            "warmup": WARMUP_SEEDS[cell],
            "retained": RETAINED_SEEDS[cell],
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def _default_output_root(cell: str) -> Path:
    attempt = "attempt-03" if cell == "PP-UKF" else "attempt-02"
    return base.PHASE_ROOT / cell / "neutra-confirmation" / attempt


def _cell(value: str) -> str:
    cell = str(value)
    if cell not in CELLS:
        raise P4NeuTraConfirmationError(f"unsupported R4 cell: {cell}")
    return cell


def _file_reference(path: Path) -> Mapping[str, Any]:
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _bare_sha256(value: Any, label: str) -> str:
    text = str(value)
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise P4NeuTraConfirmationError(f"{label} is not a SHA-256 digest")
    return text


def _json_tensor(value: Any) -> Any:
    return value.numpy().tolist()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = run_confirmation(
        cell_id=args.cell,
        output_root=args.output_root or _default_output_root(args.cell),
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
    return 0 if result["completed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
