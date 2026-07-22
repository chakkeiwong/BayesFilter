"""Same-chart HNN-NeuTra versus exact-gradient NeuTra-HMC experiments."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.neural_force_campaign import (
    generate_neural_force_supervision,
    validate_value_only_endpoint_parity,
)
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,
    NeuralForceHMCConfig,
    run_full_chain_neural_force_hmc,
    sample_neural_force_hmc,
)
from bayesfilter.inference.fixed_transport_hmc_tuning import (
    FixedTransportHMCKernelTuningConfig,
    tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference.neural_force_training import (
    FrozenScalarResidualForce,
    load_frozen_scalar_residual_force,
)
from bayesfilter.testing import lgssm_neural_force_hmc_pilot_tf as campaign
from bayesfilter.testing import predator_prey_neural_force_hmc_tf as predator_prey
from bayesfilter.testing import sir_structural_neural_force_hmc_tf as sir_structural


CELLS = ("PP-UKF", "PP-SGQF", "SIR-SGQF", "STR-UKF")
PRIMARY_ARMS = ("learned_residual", "true_gradient")
HISTORICAL_FORCE_ROOT = Path(
    "docs/plans/artifacts/corrected-neural-force-hmc-20260717"
)
HISTORICAL_FORCE_PATHS = {
    "PP-UKF": HISTORICAL_FORCE_ROOT
    / "phase-p4/PP-UKF/attempt-01-20260717T165000Z/force-training/final/"
    "w24_lr0.005_b256/frozen_force.json",
    "PP-SGQF": HISTORICAL_FORCE_ROOT
    / "phase-p4/PP-SGQF/attempt-01-20260717T171500Z/force-training/final/"
    "w24_lr0.005_b256/frozen_force.json",
    "SIR-SGQF": HISTORICAL_FORCE_ROOT
    / "phase-p5/SIR-SGQF/attempt-01-20260718T014000Z/force-training/final/"
    "w12_lr0.005_b256/frozen_force.json",
    "STR-UKF": HISTORICAL_FORCE_ROOT
    / "phase-p5/STR-UKF/attempt-01-20260718T015500Z/force-training/final/"
    "w20_lr0.005_b256/frozen_force.json",
}


class HNNNeuTraComparisonError(RuntimeError):
    """Raised when the repaired comparison cannot satisfy its evidence contract."""


def load_context(cell: str) -> Mapping[str, Any]:
    cell = str(cell).upper()
    if cell in predator_prey.CELLS:
        return predator_prey.load_context(cell)
    if cell in sir_structural.CELLS:
        return sir_structural.load_context(cell)
    raise ValueError(f"cell must be one of {CELLS}")


def load_historical_force(context: Mapping[str, Any]) -> FrozenScalarResidualForce:
    cell = str(context["cell"])
    path = HISTORICAL_FORCE_PATHS[cell]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_frozen_scalar_residual_force(
        payload,
        expected_target_signature=_target_signature(context),
        expected_transport_signature=context["loaded"].manifest.transport_hash,
    )


def prepare_timed_supervision(context: Mapping[str, Any]) -> Mapping[str, Any]:
    dimension = _dimension(context)
    flat = tf.reshape(context["latent"], [-1, dimension])
    if int(flat.shape[0]) < 3072:
        raise HNNNeuTraComparisonError("preserved NeuTra archive has fewer than 3072 rows")
    train_positions = flat[:2048]
    heldout_positions = flat[2048:3072]
    parity = validate_value_only_endpoint_parity(
        context["binding"],
        tf.concat((train_positions[:4], heldout_positions[:4]), axis=0),
        absolute_tolerance=2.0e-8,
    )
    started = time.monotonic()
    train = generate_neural_force_supervision(context["binding"], train_positions)
    heldout = generate_neural_force_supervision(context["binding"], heldout_positions)
    _synchronize_tensors(
        train.potentials, train.forces, heldout.potentials, heldout.forces
    )
    elapsed = time.monotonic() - started
    return {
        "train": train,
        "heldout": heldout,
        "parity": parity,
        "train_rows": 2048,
        "heldout_rows": 1024,
        "disjoint_archive_slices": True,
        "supervision_generation_seconds": elapsed,
        "timing_synchronized": True,
    }


def train_fresh_force(
    context: Mapping[str, Any], supervision: Mapping[str, Any], output_root: Path
) -> Mapping[str, Any]:
    started = time.monotonic()
    if context["cell"] in predator_prey.CELLS:
        result = predator_prey.train_target_specific_grid(
            context, supervision, output_root
        )
    else:
        result = sir_structural.train_grid(context, supervision, output_root)
    _synchronize_tensors(result["final"].frozen.force(supervision["heldout"].positions[:1]))
    elapsed = time.monotonic() - started
    screen_seconds = sum(float(row["elapsed_seconds"]) for row in result["screen"])
    final_seconds = float(result["final"].runtime_metadata["elapsed_seconds"])
    return {
        **result,
        "cost": {
            "screen_optimization_seconds": screen_seconds,
            "final_optimization_seconds": final_seconds,
            "grid_wall_seconds": elapsed,
            "timing_synchronized": True,
            "optimization_timers_exclude_supervision_generation": True,
        },
    }


def force_arms(
    context: Mapping[str, Any], learned: FrozenScalarResidualForce
) -> Mapping[str, FrozenPositionOnlyForce]:
    arms = {
        "learned_residual": learned.hmc_force(),
        "true_gradient": true_gradient_force(context),
    }
    if tuple(arms) != PRIMARY_ARMS:
        raise HNNNeuTraComparisonError("primary comparison must contain exactly two arms")
    return arms


def true_gradient_force(context: Mapping[str, Any]) -> FrozenPositionOnlyForce:
    """Build the exact transformed-posterior force without loading an HNN."""

    def true_force(position: tf.Tensor) -> tf.Tensor:
        _potential, force = context["binding"].potential_and_force(position)
        return force

    return FrozenPositionOnlyForce(
        true_force,
        f"{context['cell']}-complete-transformed-exact-gradient",
    )


def run_canary(
    context: Mapping[str, Any],
    learned: FrozenScalarResidualForce,
    output_root: Path,
) -> Mapping[str, Any]:
    """Run a bounded native-tuned canary for both comparison arms.

    This is deliberately a real fixed-transport tuning run, not the former
    two-transition mechanics smoke.  Compute is bounded by running one model
    cell and at most two attempts; the native tuning and fresh four-chain
    modern-R-hat verification contracts are not shortened.
    """
    dimension = _dimension(context)
    flat = tf.reshape(context["latent"], [-1, dimension])
    initial = tf.gather(flat, tf.constant((0, 17, 33, 49), tf.int32))
    parity = validate_value_only_endpoint_parity(
        context["binding"], flat[:8], absolute_tolerance=2.0e-8
    )
    target = context["binding"].hmc_target()
    arms = force_arms(context, learned)
    rows: dict[str, Mapping[str, Any]] = {}
    for arm_index, (name, force) in enumerate(arms.items()):
        started = time.monotonic()
        try:
            tuned = _native_tune_arm(
                context=context,
                force=force,
                target=target,
                initial_position=initial[0],
                output_root=output_root / "native-tuning" / name,
                seed_offset=81000 + arm_index * 100,
                raise_on_failure=False,
            )
            native_result = tuned["result"]
            rows[name] = {
                "passed": bool(tuned["passed"]),
                "native_result": native_result.payload(),
                "selected": tuned.get("selected_payload"),
                "elapsed_seconds": float(time.monotonic() - started),
                "mass_policy": "fixed_identity_z",
                "target_accept_prob": 0.70,
                "acceptance_band": (0.65, 0.75),
                "modern_rhat_required": True,
                "verification_min_retained_results_per_chain": 1000,
            }
        except Exception as exc:  # fail closed while preserving the arm record
            rows[name] = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": float(time.monotonic() - started),
            }
    structural = _structural_contract(context)
    mass_signatures = {
        row.get("native_result", {}).get("identity_z_mass_artifact_signature")
        for row in rows.values()
        if isinstance(row.get("native_result"), Mapping)
    }
    mass_signature_match = bool(
        len(mass_signatures) == 1 and None not in mass_signatures
    )
    return {
        "schema": "bayesfilter.hnn_neutra_native_tuning_canary.v2",
        "cell": context["cell"],
        "passed": bool(
            parity["passed"]
            and mass_signature_match
            and all(row["passed"] for row in rows.values())
        ),
        "target_signature": _target_signature(context),
        "transport_signature": context["loaded"].manifest.transport_hash,
        "value_only_endpoint_parity": parity,
        "arms": rows,
        "structural_contract": structural,
        "primary_arm_set": PRIMARY_ARMS,
        "cross_arm_identity_z_mass_signature_match": mass_signature_match,
        "identity_z_mass_artifact_signature": (
            next(iter(mass_signatures)) if mass_signature_match else None
        ),
        "tuning_contract": {
            "native_tuner": "bayesfilter.inference.fixed_transport_hmc_tuning.tune_fixed_transport_hmc_kernel",
            "target_accept_prob": 0.70,
            "acceptance_band": (0.65, 0.75),
            "mass_policy": "fixed_identity_z",
            "modern_rhat": "max(rank-normalized split, folded rank-normalized split)",
            "verification_min_retained_results_per_chain": 1000,
        },
        "nonclaims": ("bounded native-tuning canary only",),
    }


def run_full_cell(context: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    total_started = time.monotonic()
    supervision = prepare_timed_supervision(context)
    training = train_fresh_force(context, supervision, output_root / "force-training")
    arms = force_arms(context, training["final"].frozen)
    target = context["binding"].hmc_target()
    initial = tf.gather(
        supervision["heldout"].positions, tf.constant((0, 17, 33, 49), tf.int32)
    )
    tuning = {}
    for arm_index, (name, force) in enumerate(arms.items()):
        tuning[name] = _native_tune_arm(
            context=context,
            force=force,
            target=target,
            initial_position=initial[0],
            output_root=output_root / "native-tuning" / name,
            seed_offset=83000 + CELLS.index(context["cell"]) * 1000 + arm_index * 100,
        )
    selected_hnn = tuning["learned_residual"]["selected"]
    matched = matched_mechanics_benchmark(
        arms=arms,
        target=target,
        initial_position=initial,
        step_size=float(selected_hnn.selected_step_size),
        num_leapfrog_steps=int(selected_hnn.num_leapfrog_steps),
        seed=(20260718, 84000 + CELLS.index(context["cell"])),
    )
    runs = {}
    common_seed_base = 85000 + CELLS.index(context["cell"]) * 5000
    for name, force in arms.items():
        selected = tuning[name]["selected"]
        runs[name] = campaign.run_sequential_arm(
            arm_id=name,
            force=force,
            target=target,
            initial_position=initial,
            transform=context["loaded"].transport,
            parameter_names=_source_names(context),
            step_size=float(selected.selected_step_size),
            num_leapfrog_steps=int(selected.num_leapfrog_steps),
            output_root=output_root / "sampling",
            seed_base=common_seed_base,
            precompile=True,
            synchronize_timing=True,
        )
    physical = {
        name: _physical_samples(context, run["private_retained_raw"])
        for name, run in runs.items()
    }
    truth = {
        name: (
            campaign.truth_tail_summary(
                samples, context["truth_physical"], _physical_names(context)
            )
            if runs[name]["passed"]
            else None
        )
        for name, samples in physical.items()
    }
    agreement = direct_posterior_agreement(
        physical["learned_residual"],
        physical["true_gradient"],
        parameter_names=_physical_names(context),
        learned_diagnostic=_final_diagnostic(runs["learned_residual"]),
        exact_diagnostic=_final_diagnostic(runs["true_gradient"]),
    ) if all(run["passed"] for run in runs.values()) else None
    accuracy = {
        name: {
            "passed": bool(runs[name]["passed"] and truth[name] and truth[name]["passed"]),
            "sampler_validity": bool(runs[name]["passed"]),
            "truth_tail": truth[name],
            "truth_status": _truth_status(truth[name]),
        }
        for name in PRIMARY_ARMS
    }
    cost = cost_ledger(
        supervision=supervision,
        training=training,
        tuning=tuning,
        runs=runs,
        matched=matched,
    )
    both_valid = all(value["passed"] for value in accuracy.values())
    direct_passed = bool(agreement and agreement["status"] == "PASS")
    second_seed_required = bool(
        any(value["truth_status"] == "MARGINAL" for value in accuracy.values())
        or (agreement and agreement["status"] == "MARGINAL")
    )
    performance_passed = bool(
        both_valid
        and direct_passed
        and matched["passed"]
        and matched["learned_faster"]
        and _strictly_lower_finite(
            cost["tuned_seconds_per_minimum_bulk_ess"]["learned_residual"],
            cost["tuned_seconds_per_minimum_bulk_ess"]["true_gradient"],
        )
    )
    return {
        "schema": "bayesfilter.hnn_neutra_exact_comparison_result.v1",
        "cell": context["cell"],
        "completed": True,
        "passed": bool(both_valid and direct_passed),
        "decision": (
            "ONE_SEED_ACCURACY_AND_DESCRIPTIVE_PERFORMANCE_PASS"
            if performance_passed
            else "ONE_SEED_ACCURACY_PASS_PERFORMANCE_NOT_DEMONSTRATED"
            if both_valid and direct_passed
            else "MARGINAL_DIRECT_AGREEMENT_SECOND_SEED_REQUIRED"
            if second_seed_required
            else "CANDIDATE_OR_BASELINE_VALIDITY_FAILURE"
        ),
        "second_seed_required": second_seed_required,
        "performance_passed": performance_passed,
        "primary_arm_set": PRIMARY_ARMS,
        "target_signature": _target_signature(context),
        "transport_signature": context["loaded"].manifest.transport_hash,
        "value_only_endpoint_parity": supervision["parity"],
        "supervision": {
            key: value for key, value in supervision.items()
            if key not in {"train", "heldout"}
        },
        "training": {
            "selected": training["selected"],
            "screen": training["screen"],
            "final_result_path": str(training["final"].result_path),
            "final_metrics": training["final"].metrics,
            "cost": training["cost"],
        },
        "tuning": {
            name: {
                "selected": value["selected_payload"],
                "native_result": value["result"].payload(),
                "elapsed_seconds": value["elapsed_seconds"],
                "target_accept_prob": 0.70,
                "acceptance_band": (0.65, 0.75),
                "mass_policy": "fixed_identity_z",
            }
            for name, value in tuning.items()
        },
        "matched_mechanics": matched,
        "runs": {name: campaign.json_ready(value) for name, value in runs.items()},
        "accuracy": accuracy,
        "direct_posterior_agreement": agreement,
        "cost_ledger": cost,
        "structural_contract": _structural_contract(context),
        "wall_time_seconds": time.monotonic() - total_started,
        "inference_status": {
            "hard_veto_screen": "passed" if both_valid else "failed",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": "all one-seed runtime and ESS differences",
            "default_readiness": False,
            "next_evidence_needed": "conditional second seed only for marginal direct or truth agreement",
        },
        "nonclaims": (
            "one fixture and one seed per arm",
            "no universal superiority or default-readiness claim",
            "filter-defined posterior accuracy is not latent-model exactness",
        ),
    }


def run_structural_exact_repair(
    context: Mapping[str, Any],
    *,
    original_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    """Repair only the failed STR-UKF exact arm at a safer measured candidate."""

    if context["cell"] != "STR-UKF":
        raise HNNNeuTraComparisonError("exact-arm repair is restricted to STR-UKF")
    original_result_path = original_root / "result.json"
    original_manifest_path = original_root / "run_manifest.json"
    original_hashes_path = original_root / "artifact_hashes.json"
    original = _read_json(original_result_path)
    hashes = _read_json(original_hashes_path)
    actual_hashes = {
        "result_sha256": _file_sha256(original_result_path),
        "run_manifest_sha256": _file_sha256(original_manifest_path),
    }
    if hashes != actual_hashes:
        raise HNNNeuTraComparisonError("original STR-UKF top-level hash ledger mismatch")
    if (
        original.get("schema") != "bayesfilter.hnn_neutra_exact_comparison_result.v1"
        or original.get("cell") != "STR-UKF"
        or original.get("target_signature") != _target_signature(context)
        or original.get("transport_signature")
        != context["loaded"].manifest.transport_hash
        or original.get("structural_contract", {}).get("passed") is not True
        or original.get("runs", {}).get("learned_residual", {}).get("passed") is not True
        or original.get("runs", {}).get("true_gradient", {}).get("hard_vetoes")
        != ["warmup_chunk_health_failed"]
    ):
        raise HNNNeuTraComparisonError("original STR-UKF repair premise mismatch")

    learned_sidecar = Path(
        str(
            original["runs"]["learned_residual"]["archives"]["retained_raw"][
                "tensor_path"
            ]
        )
        + ".json"
    )
    learned_raw = campaign.read_tensor_archive(learned_sidecar)
    learned_force_path = Path(original["training"]["final_result_path"]).with_name(
        "frozen_force.json"
    )
    learned_force = load_frozen_scalar_residual_force(
        _read_json(learned_force_path),
        expected_target_signature=_target_signature(context),
        expected_transport_signature=context["loaded"].manifest.transport_hash,
    )
    dimension = _dimension(context)
    flat = tf.reshape(context["latent"], [-1, dimension])
    heldout_positions = flat[2048:3072]
    initial = tf.gather(
        heldout_positions, tf.constant((0, 17, 33, 49), tf.int32)
    )
    exact_force = true_gradient_force(context)
    target = context["binding"].hmc_target()
    exact = campaign.run_sequential_arm(
        arm_id="true_gradient",
        force=exact_force,
        target=target,
        initial_position=initial,
        transform=context["loaded"].transport,
        parameter_names=_source_names(context),
        step_size=0.1,
        num_leapfrog_steps=8,
        output_root=output_root / "sampling",
        seed_base=85000 + CELLS.index("STR-UKF") * 5000,
        precompile=True,
        synchronize_timing=True,
    )
    matched = matched_mechanics_benchmark(
        arms={
            "learned_residual": learned_force.hmc_force(),
            "true_gradient": exact_force,
        },
        target=target,
        initial_position=initial,
        step_size=0.1,
        num_leapfrog_steps=8,
        seed=(20260718, 84000 + CELLS.index("STR-UKF")),
    )
    exact_raw = exact["private_retained_raw"]
    learned_physical = _physical_samples(context, learned_raw)
    exact_physical = _physical_samples(context, exact_raw)
    learned_truth = campaign.truth_tail_summary(
        learned_physical, context["truth_physical"], _physical_names(context)
    )
    exact_truth = (
        campaign.truth_tail_summary(
            exact_physical, context["truth_physical"], _physical_names(context)
        )
        if exact["passed"]
        else None
    )
    agreement = (
        direct_posterior_agreement(
            learned_physical,
            exact_physical,
            parameter_names=_physical_names(context),
            learned_diagnostic=_final_diagnostic(
                original["runs"]["learned_residual"]
            ),
            exact_diagnostic=_final_diagnostic(exact),
        )
        if exact["passed"]
        else None
    )
    accuracy = {
        "learned_residual": {
            "passed": bool(learned_truth["passed"]),
            "sampler_validity": True,
            "truth_tail": learned_truth,
            "truth_status": _truth_status(learned_truth),
        },
        "true_gradient": {
            "passed": bool(exact["passed"] and exact_truth and exact_truth["passed"]),
            "sampler_validity": bool(exact["passed"]),
            "truth_tail": exact_truth,
            "truth_status": _truth_status(exact_truth),
        },
    }
    runs_for_cost = {
        "learned_residual": original["runs"]["learned_residual"],
        "true_gradient": exact,
    }
    cost = cost_ledger(
        supervision=original["supervision"],
        training=original["training"],
        tuning=original["tuning"],
        runs=runs_for_cost,
        matched=matched,
    )
    both_valid = all(value["passed"] for value in accuracy.values())
    direct_passed = bool(agreement and agreement["status"] == "PASS")
    second_seed_required = bool(
        any(value["truth_status"] == "MARGINAL" for value in accuracy.values())
        or (agreement and agreement["status"] == "MARGINAL")
    )
    performance_passed = bool(
        both_valid
        and direct_passed
        and matched["passed"]
        and matched["learned_faster"]
        and _strictly_lower_finite(
            cost["tuned_seconds_per_minimum_bulk_ess"]["learned_residual"],
            cost["tuned_seconds_per_minimum_bulk_ess"]["true_gradient"],
        )
    )
    passed = bool(both_valid and direct_passed)
    return {
        "schema": "bayesfilter.hnn_neutra_structural_exact_repair_result.v1",
        "cell": "STR-UKF",
        "completed": True,
        "passed": passed,
        "decision": (
            "ONE_SEED_ACCURACY_AND_DESCRIPTIVE_PERFORMANCE_PASS"
            if performance_passed
            else "ONE_SEED_ACCURACY_PASS_PERFORMANCE_NOT_DEMONSTRATED"
            if passed
            else "MARGINAL_DIRECT_AGREEMENT_SECOND_SEED_REQUIRED"
            if second_seed_required
            else "EXACT_REPAIR_VALIDITY_FAILURE"
        ),
        "second_seed_required": second_seed_required,
        "performance_passed": performance_passed,
        "repair_selection": {
            "step_size": 0.1,
            "num_leapfrog_steps": 8,
            "role": "localized_exact_arm_energy_health_repair",
            "source_tuning_candidate": "eps0.1_l8",
            "source_tuning_maximum_absolute_delta_h": 16.04995289049518,
            "source_tuning_modern_rhat": 1.0706975657128621,
            "failed_candidate": "eps0.2_l8",
            "failed_warmup_maximum_absolute_delta_h": 4213.1896518000785,
            "seed_and_initialization_unchanged": True,
        },
        "original_attempt": {
            "root": str(original_root),
            **actual_hashes,
        },
        "target_signature": _target_signature(context),
        "transport_signature": context["loaded"].manifest.transport_hash,
        "structural_contract": _structural_contract(context),
        "runs": {
            "learned_residual": {
                "source": "verified_preserved_attempt_01",
                "result": original["runs"]["learned_residual"],
            },
            "true_gradient": campaign.json_ready(exact),
        },
        "accuracy": accuracy,
        "direct_posterior_agreement": agreement,
        "cost_ledger": cost,
        "matched_mechanics": {
            **matched,
            "status": "repaired_common_mechanics_eps0.1_l8",
            "supersedes_attempt_01_matched_health_failure": True,
            "attempt_01_matched_mechanics": original["matched_mechanics"],
        },
        "inference_status": {
            "hard_veto_screen": "passed" if both_valid else "failed",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": (
                "one-seed independently tuned sampling and attempt-01 matched timing"
            ),
            "default_readiness": False,
            "next_evidence_needed": (
                "healthy matched-mechanics repair for performance promotion"
                if passed else "classify repeated exact-arm failure"
            ),
        },
        "nonclaims": (
            "no HNN retraining or HNN resampling in this repair",
            "archived HNN weights reused only for repaired matched timing",
            "attempt-01 matched timing is superseded because it failed its energy-health gate",
            "no universal superiority or default-readiness claim",
            "common NeuTra chart-training cost remains unreconstructed",
        ),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HNNNeuTraComparisonError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_mechanics_benchmark(
    *,
    arms: Mapping[str, FrozenPositionOnlyForce],
    target: Any,
    initial_position: tf.Tensor,
    step_size: float,
    num_leapfrog_steps: int,
    seed: tuple[int, int],
    num_results: int = 500,
) -> Mapping[str, Any]:
    if tuple(arms) != PRIMARY_ARMS:
        raise HNNNeuTraComparisonError("matched benchmark requires exact primary arm set")
    dimension = int(tf.convert_to_tensor(initial_position).shape[-1])
    config = NeuralForceHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        inverse_mass_diagonal=(1.0,) * dimension,
        dtype="float64",
    )
    initial = tf.convert_to_tensor(initial_position, tf.float64)
    potential = target.function(initial)
    _synchronize_tensors(potential)
    programs = {}
    compile_seconds = {}
    for name, force in arms.items():
        @tf.function(jit_compile=True, reduce_retracing=True)
        def run(position: tf.Tensor, current: tf.Tensor, run_seed: tf.Tensor):
            return sample_neural_force_hmc(
                position,
                current,
                force,
                target,
                config,
                num_warmup=0,
                num_results=num_results,
                seed=run_seed,
            )

        started = time.monotonic()
        probe = run(initial, potential, tf.constant(seed, tf.int32))
        campaign.synchronize_chain(probe)
        compile_seconds[name] = time.monotonic() - started
        programs[name] = run
    timings = {name: [] for name in PRIMARY_ARMS}
    health = {name: [] for name in PRIMARY_ARMS}
    orders = (PRIMARY_ARMS, tuple(reversed(PRIMARY_ARMS)), PRIMARY_ARMS)
    for order in orders:
        for name in order:
            started = time.monotonic()
            chain = programs[name](initial, potential, tf.constant(seed, tf.int32))
            campaign.synchronize_chain(chain)
            timings[name].append(time.monotonic() - started)
            health[name].append(campaign.chain_health(chain))
    median = {name: statistics.median(values) for name, values in timings.items()}
    mechanics = {
        "step_size": float(step_size),
        "num_leapfrog_steps": int(num_leapfrog_steps),
        "chain_count": int(initial.shape[0]),
        "transitions_per_chain": int(num_results),
        "endpoint_batch_invocations": int(num_results),
        "endpoint_scalar_values": int(num_results * int(initial.shape[0])),
        "force_batch_invocations": int(num_results * (num_leapfrog_steps + 1)),
        "seed": tuple(int(value) for value in seed),
    }
    return {
        "schema": "bayesfilter.hnn_neutra_matched_mechanics.v1",
        "passed": all(row["passed"] for rows in health.values() for row in rows),
        "mechanics": mechanics,
        "compile_probe_seconds": compile_seconds,
        "warm_repeats_seconds": timings,
        "warm_median_seconds": median,
        "seconds_per_transition_batch": {
            name: value / num_results for name, value in median.items()
        },
        "speed_ratio_exact_over_learned": (
            median["true_gradient"] / median["learned_residual"]
        ),
        "learned_faster": median["learned_residual"] < median["true_gradient"],
        "alternating_orders": orders,
        "timing_synchronized": True,
        "role": "mechanism_cost_only_not_convergence_evidence",
    }


def direct_posterior_agreement(
    learned_samples: tf.Tensor,
    exact_samples: tf.Tensor,
    *,
    parameter_names: Sequence[str],
    learned_diagnostic: Mapping[str, Any],
    exact_diagnostic: Mapping[str, Any],
) -> Mapping[str, Any]:
    learned = tf.reshape(
        tf.cast(tf.convert_to_tensor(learned_samples), tf.float64),
        [-1, len(parameter_names)],
    )
    exact = tf.reshape(
        tf.cast(tf.convert_to_tensor(exact_samples), tf.float64),
        [-1, len(parameter_names)],
    )
    learned_mean = tf.reduce_mean(learned, axis=0)
    exact_mean = tf.reduce_mean(exact, axis=0)
    learned_sd = tf.math.reduce_std(learned, axis=0)
    exact_sd = tf.math.reduce_std(exact, axis=0)
    learned_ess = tf.constant(
        [row["bulk_ess"] for row in learned_diagnostic["parameter_diagnostics"]], tf.float64
    )
    exact_ess = tf.constant(
        [row["bulk_ess"] for row in exact_diagnostic["parameter_diagnostics"]], tf.float64
    )
    learned_mcse = learned_sd / tf.sqrt(learned_ess)
    exact_mcse = exact_sd / tf.sqrt(exact_ess)
    difference = tf.abs(learned_mean - exact_mean)
    combined_mcse = tf.sqrt(tf.square(learned_mcse) + tf.square(exact_mcse))
    z_mc = difference / tf.maximum(combined_mcse, tf.constant(1.0e-15, tf.float64))
    learned_interval = tfp.stats.percentile(learned, [2.5, 97.5], axis=0, interpolation="linear")
    exact_interval = tfp.stats.percentile(exact, [2.5, 97.5], axis=0, interpolation="linear")
    overlap = tf.maximum(learned_interval[0], exact_interval[0]) <= tf.minimum(
        learned_interval[1], exact_interval[1]
    )
    failure = tf.logical_or(z_mc >= 3.0, tf.logical_not(overlap))
    marginal = tf.logical_and(tf.logical_not(failure), z_mc > 1.96)
    rows = tuple(
        {
            "parameter": str(name),
            "learned_mean": float(learned_mean[index].numpy()),
            "exact_mean": float(exact_mean[index].numpy()),
            "absolute_mean_difference": float(difference[index].numpy()),
            "learned_mcse": float(learned_mcse[index].numpy()),
            "exact_mcse": float(exact_mcse[index].numpy()),
            "z_mc": float(z_mc[index].numpy()),
            "learned_interval_95": tuple(float(value) for value in learned_interval[:, index].numpy()),
            "exact_interval_95": tuple(float(value) for value in exact_interval[:, index].numpy()),
            "intervals_overlap": bool(overlap[index].numpy()),
            "status": (
                "FAIL" if bool(failure[index].numpy()) else
                "MARGINAL" if bool(marginal[index].numpy()) else "PASS"
            ),
        }
        for index, name in enumerate(parameter_names)
    )
    status = "FAIL" if bool(tf.reduce_any(failure).numpy()) else (
        "MARGINAL" if bool(tf.reduce_any(marginal).numpy()) else "PASS"
    )
    return {
        "schema": "bayesfilter.hnn_neutra_direct_posterior_agreement.v1",
        "status": status,
        "passed": status == "PASS",
        "second_seed_required": status == "MARGINAL",
        "rows": rows,
        "criterion": "interval overlap and pooled-MCSE z_MC; fail >=3, marginal >1.96",
        "claim_scope": "same-filter physical posterior mean agreement only",
    }


def cost_ledger(
    *,
    supervision: Mapping[str, Any],
    training: Mapping[str, Any],
    tuning: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    matched: Mapping[str, Any],
) -> Mapping[str, Any]:
    tuning_seconds = {
        name: float(value.get("elapsed_seconds", sum(
            float(row["elapsed_seconds"]) for row in value.get("rows", ())
        )))
        for name, value in tuning.items()
    }
    sampling_seconds = {
        name: float(run["sampling_execution_seconds"]) for name, run in runs.items()
    }
    minimum_bulk_ess = {
        name: _minimum_bulk_ess_or_none(run) for name, run in runs.items()
    }
    seconds_per_ess = {
        name: (
            sampling_seconds[name] / minimum_bulk_ess[name]
            if minimum_bulk_ess[name] is not None and minimum_bulk_ess[name] > 0.0
            else None
        )
        for name in PRIMARY_ARMS
    }
    preparation = (
        float(supervision["supervision_generation_seconds"])
        + float(training["cost"]["grid_wall_seconds"])
    )
    reuse_total = {
        "learned_residual": preparation + tuning_seconds["learned_residual"] + sampling_seconds["learned_residual"],
        "true_gradient": tuning_seconds["true_gradient"] + sampling_seconds["true_gradient"],
    }
    saving = (
        float(matched["warm_median_seconds"]["true_gradient"])
        - float(matched["warm_median_seconds"]["learned_residual"])
    ) / float(matched["mechanics"]["transitions_per_chain"])
    overhead = max(
        0.0,
        preparation + tuning_seconds["learned_residual"] - tuning_seconds["true_gradient"],
    )
    pre_sampling_delta = (
        preparation
        + tuning_seconds["learned_residual"]
        - tuning_seconds["true_gradient"]
    )
    return {
        "schema": "bayesfilter.hnn_neutra_cost_ledger.v1",
        "supervision_generation_seconds": float(supervision["supervision_generation_seconds"]),
        "training_grid_wall_seconds": float(training["cost"]["grid_wall_seconds"]),
        "training_optimization_only": {
            "screen_seconds": float(training["cost"]["screen_optimization_seconds"]),
            "final_seconds": float(training["cost"]["final_optimization_seconds"]),
        },
        "tuning_cold_seconds": tuning_seconds,
        "sampling_warm_seconds": sampling_seconds,
        "minimum_bulk_ess": minimum_bulk_ess,
        "tuned_seconds_per_minimum_bulk_ess": seconds_per_ess,
        "efficiency_status": {
            name: "available" if seconds_per_ess[name] is not None else "not_valid_for_efficiency"
            for name in PRIMARY_ARMS
        },
        "reuse_scenario_seconds": reuse_total,
        "common_neutra_chart_training_seconds": "not_reconstructed",
        "from_scratch_total_seconds": None,
        "from_scratch_total_status": "unsupported_missing_common_chart_training_cost",
        "matched_saving_seconds_per_transition_batch": saving,
        "hnn_preparation_seconds": preparation,
        "preparation_break_even_transition_batches": (
            math.ceil(preparation / saving) if saving > 0.0 else None
        ),
        "reuse_campaign_pre_sampling_delta_seconds": pre_sampling_delta,
        "reuse_campaign_break_even_transition_batches": (
            math.ceil(max(0.0, pre_sampling_delta) / saving)
            if saving > 0.0 else None
        ),
        # Compatibility aliases: these describe the full reuse campaign after
        # both arms have paid their independently measured tuning costs.
        "hnn_specific_overhead_seconds": overhead,
        "break_even_transition_batches": (
            math.ceil(overhead / saving) if saving > 0.0 else None
        ),
        "timing_synchronized": True,
    }


def _target_signature(context: Mapping[str, Any]) -> str:
    if "target_signature" in context:
        return str(context["target_signature"])
    return str(context["identity"].target_signature)


def _dimension(context: Mapping[str, Any]) -> int:
    return int(context.get("dimension", predator_prey.DIMENSION))


def _source_names(context: Mapping[str, Any]) -> tuple[str, ...]:
    if "source_names" in context:
        return tuple(context["source_names"])
    return (
        "r_source_probit", "K_source_probit", "a_source_probit",
        "s_source_probit", "u_source_probit", "v_source_probit",
    )


def _tuning_grid(context: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if context["cell"] in predator_prey.CELLS:
        return (0.2, 0.4, 0.6, 0.8), (6, 10)
    if context["cell"] == "SIR-SGQF":
        return (0.2, 0.4, 0.6, 0.8), (6, 10)
    return (0.025, 0.05, 0.1, 0.2), (8, 12)


def _native_tune_arm(
    *,
    context: Mapping[str, Any],
    force: FrozenPositionOnlyForce,
    target: Any,
    initial_position: tf.Tensor,
    output_root: Path,
    seed_offset: int,
    raise_on_failure: bool = True,
) -> Mapping[str, Any]:
    """Tune one arm through the BayesFilter fixed-transport native stack."""

    step_sizes, leapfrog_steps = _tuning_grid(context)
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=float(step_sizes[0]),
        leapfrog_grid=tuple(int(value) for value in leapfrog_steps),
        chain_count=4,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
        repair_band=(0.55, 0.85),
        # Attempt 1 showed that 64 adaptation steps left the exact arm just
        # outside the owner band.  Continue the same native dual-averaging
        # ladder rather than substituting a hand-selected step size.
        budget_schedule=(16, 32, 64, 128, 256),
        tune_num_results=16,
        screen_num_results=32,
        screen_num_burnin_steps=16,
        verification_num_results=1000,
        verification_num_burnin_steps=100,
        require_modern_rank_normalized_verification=True,
        verification_min_retained_results_per_chain=1000,
        verification_rhat_max=1.01,
        tune_seed_base=(20260718, int(seed_offset)),
        screen_seed_base=(20260718, int(seed_offset) + 1),
        verification_seed_base=(20260718, int(seed_offset) + 2),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=f"{context['cell']}:fixed_neutra_native_tuning",
        output_filename="native_tuning_result.json",
        source="bayesfilter.hnn_neutra_native_tuning_correction",
        proposal_dynamics_identity=force.identity,
    )
    started = time.monotonic()
    if force.identity.endswith("complete-transformed-exact-gradient"):
        result = tune_fixed_transport_hmc_kernel(
            base_adapter=context["adapter"],
            fixed_transport=context["loaded"].transport,
            initial_position=tf.convert_to_tensor(initial_position, tf.float64),
            config=config,
            output_dir=output_root,
        )
    else:
        hnn_target = target

        def run_hnn(adapter: Any, state: Any, run_config: Any) -> Any:
            return run_full_chain_neural_force_hmc(
                adapter,
                state,
                run_config,
                force=force,
                target=hnn_target,
            )

        result = tune_fixed_transport_hmc_kernel(
            base_adapter=context["adapter"],
            fixed_transport=context["loaded"].transport,
            initial_position=tf.convert_to_tensor(initial_position, tf.float64),
            config=config,
            output_dir=output_root,
            run_full_chain=run_hnn,
        )
    elapsed = time.monotonic() - started
    passed = bool(result.passed and result.final_kernel_payload is not None)
    if not passed and raise_on_failure:
        raise HNNNeuTraComparisonError(
            f"native tuning failed for {context['cell']} arm {force.identity}: "
            f"{result.final_status}"
        )
    selected = result.selected_candidate
    if (selected is None or selected.selected_step_size is None) and raise_on_failure:
        raise HNNNeuTraComparisonError("native tuning returned no selected kernel")
    return {
        "result": result,
        "selected": selected,
        "passed": passed,
        "selected_payload": (
            None
            if selected is None or selected.selected_step_size is None
            else {
                "step_size": float(selected.selected_step_size),
                "num_leapfrog_steps": int(selected.num_leapfrog_steps),
                "acceptance_rate": selected.selected_acceptance_rate,
                "final_status": selected.final_status,
                "mass_policy": "fixed_identity_z",
                "target_accept_prob": 0.70,
                "acceptance_band": (0.65, 0.75),
            }
        ),
        "elapsed_seconds": float(elapsed),
    }


def _physical_names(context: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(context.get("physical_names", predator_prey.PARAMETER_NAMES))


def _physical_samples(context: Mapping[str, Any], source: tf.Tensor) -> tf.Tensor:
    if context["cell"] in predator_prey.CELLS:
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
            source_chart_physical_parameters,
        )

        shape = tf.shape(source)
        flat, _ = source_chart_physical_parameters(
            tf.reshape(source, [-1, predator_prey.DIMENSION])
        )
        return tf.reshape(flat, shape)
    return sir_structural._physical(context, source)


def _final_diagnostic(run: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = run.get("retained_checks", ())
    if not checks:
        raise HNNNeuTraComparisonError("run lacks retained convergence diagnostic")
    return checks[-1]["full_convergence"]


def _minimum_bulk_ess_or_none(run: Mapping[str, Any]) -> float | None:
    checks = run.get("retained_checks", ())
    if not checks:
        return None
    value = checks[-1].get("full_convergence", {}).get("min_bulk_ess")
    return None if value is None else float(value)


def _strictly_lower_finite(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return bool(math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) < float(right))


def _truth_status(summary: Mapping[str, Any] | None) -> str:
    if summary is None:
        return "NOT_AVAILABLE"
    minimum = float(summary["minimum_p_truth"])
    if minimum < campaign.SEVERE_TRUTH_TAIL:
        return "FAIL"
    if minimum < campaign.PASS_TRUTH_TAIL:
        return "MARGINAL"
    return "PASS"


def _structural_contract(context: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if context["cell"] != "STR-UKF":
        return None
    manifest = context["adapter"].contract.filter_program.filter_manifest
    passed = bool(
        manifest.get("deterministic_completion") == "k_t=phi*k_(t-1)+gamma*m_t^2"
        and manifest.get("artificial_k_noise_allowed") is False
    )
    if not passed:
        raise HNNNeuTraComparisonError("structural deterministic/no-noise contract drift")
    return {
        "passed": True,
        "deterministic_completion": manifest["deterministic_completion"],
        "artificial_k_noise_allowed": False,
    }


def _synchronize_tensors(*values: tf.Tensor) -> None:
    tf.add_n(
        [tf.reduce_sum(tf.cast(tf.convert_to_tensor(value), tf.float64)) for value in values]
    ).numpy()


__all__ = [
    "CELLS",
    "HNNNeuTraComparisonError",
    "PRIMARY_ARMS",
    "cost_ledger",
    "direct_posterior_agreement",
    "force_arms",
    "load_context",
    "load_historical_force",
    "matched_mechanics_benchmark",
    "prepare_timed_supervision",
    "run_canary",
    "run_full_cell",
    "train_fresh_force",
]
