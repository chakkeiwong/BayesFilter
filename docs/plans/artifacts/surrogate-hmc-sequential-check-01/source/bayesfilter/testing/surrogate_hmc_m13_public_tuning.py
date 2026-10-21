"""M13 exploratory surrogate-HMC tuning through the public artifact authority.

The canonical analytical DCTR field is a position-only proposal force. The
separately frozen trajectory-importance endpoint supplies the MH potential.
No endpoint-gradient, posterior-convergence, or efficiency claim is made here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import tensorflow as tf

from bayesfilter.inference import (
    TensorFlowHMCKernelTuningConfig, VerifiedTensorFlowHMCConfig,
    FourChainMeanBandAcceptancePolicy, tune_hmc_kernel,
    load_verified_tensorflow_hmc_result,
)
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.inference.surrogate_hmc_materiality_selection import load_selection
from bayesfilter.testing import surrogate_hmc_m13_endpoint_mechanics as mechanics
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load
from bayesfilter.testing.surrogate_hmc_m13_materiality_controls import selection_scope, source_hashes
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import native_force, batched_endpoint, bind_components

PLAN = "docs/plans/surrogate-hmc-m13-public-tuning-2026-09-26.md"
CONTROL_ROOT = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01"
PRIOR_SD = math.log(2.)
F64 = tf.float64


def read_tensor(directory, record):
    path = Path(directory) / record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("M13 tensor checksum mismatch")
    return tf.io.parse_tensor(tf.io.read_file(str(path)), tf.as_dtype(record.get("dtype", "float64")))


def checked_control_run(root):
    """Require a completed worker and preserve any reviewed wrapper failure."""
    root = Path(root)
    parent_path = root / "report.json"
    parent = json.loads(parent_path.read_text())
    worker = parent["stages"]["m13-materiality-repair"]
    if worker.get("execution") != "COMPLETED" or worker["result"].get("qualified") is not True:
        raise ValueError("control worker did not complete with a selection")
    if parent.get("source_unchanged") is not True:
        review = json.loads((root / "source-change-review.json").read_text())
        selection_path = root / "m13-materiality-repair" / "selection.json"
        if (review.get("parent_report_sha256") != file_hash(parent_path)
                or review.get("selection_sha256") != file_hash(selection_path)
                or review.get("worker_source_hashes_unchanged") is not True
                or review.get("original_snapshot_verified") is not True
                or review.get("repair_dispatch_ast_unchanged") is not True):
            raise ValueError("control source-change reconciliation is invalid")
    return parent


def selected_components():
    """Revalidate the repository-issued selection against the actual pilot scope."""
    checked_control_run(CONTROL_ROOT)
    path = CONTROL_ROOT / "m13-materiality-repair" / "selection.json"
    _, fixture, _ = _load()
    # Read the candidate controls only to instantiate its dependency closure;
    # load_selection recomputes the decision and rejects caller-stamped choices.
    payload = json.loads(path.read_text())
    target = mechanics.make_range_bearing_target(fixture, particle_count=4096,
        controls=payload["controls"], dtype=tf.float32,
        design_seed=20260923, design_partition=9)
    scope = json.loads(json.dumps(selection_scope(fixture, repair=True)))
    selection = load_selection(path, expected_scope=scope,
        expected_source_hashes=source_hashes(target))
    _, evaluator = mechanics.endpoint_components(fixture)
    identity = "M13-data1704-selected-surrogate-HMC:" + file_hash(path)
    binding = bind_components(native_force(target), batched_endpoint(evaluator), identity)

    class Adapter:
        target_scope = identity

        def adapter_signature(self):
            return identity + ":" + binding.binding_hash

    return Adapter(), binding, target, fixture, selection, path


def tuning_config(scope):
    """Prospective bounded search; these are tuning hypotheses, not defaults."""
    nomination = TensorFlowHMCKernelTuningConfig(parameter_dimension=2,
        evidence_role="diagnostic_candidate_screen", mass_window_results=(128, 256),
        step_adaptation_results=256, verification_results=128, max_leapfrog_steps=4,
        initial_step_size=.2,
        budget_provenance="approved M13 continuation: total144000 CPU /126000 GPU seconds; parent manifest enforces smaller pass and attempt caps",
        initial_step_size_provenance="normalized .2 warm-start hypothesis; independently adapted on data1704",
        geometry_provenance="prior covariance warm start; own-target windowed covariance, 10 percent prior shrinkage",
        target_scope=scope,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.5, 1.), per_chain_band=(.3, 1.)),
        target_accept_prob=.7, verification_repair_rounds=0, step_repair_factor=2.,
        mass_shrinkage=.1, covariance_jitter=0.,
        eigenvalue_floor=64 * sys.float_info.epsilon * PRIOR_SD**2,
        max_condition_number=sys.float_info.epsilon**-.5,
        seed=(20260926, 1101), use_xla=True)
    starts = tuple(tuple(x * PRIOR_SD for x in row)
        for row in ((-1., -1.), (-.33, .33), (.33, -.33), (1., 1.)))
    return VerifiedTensorFlowHMCConfig(nomination, starts,
        "four distinct prior-scale starts; all warmup and verification excluded from retained estimates",
        warmup_results=2000, verification_results=1000, fresh_seed=(20260926, 1102),
        energy_error_max=1000.)


def metric_guard_diagnostic(config):
    """Evaluate the actual metric protection on this target's prior scale.

    C=.9 E+.1 P0 >= .1 P0 for positive semidefinite E. Thus zero additive
    jitter is justified by the positive prior term, not inherited by accident.
    This changes proposal geometry only, while the MH endpoint stays fixed.
    """
    from bayesfilter.inference.hmc_tensorflow_tuning import _regularize_covariance
    prior = tf.eye(2, dtype=F64) * PRIOR_SD**2
    healthy, _, _ = _regularize_covariance(prior, target=prior, config=config.nomination)
    rank_one = tf.linalg.diag(tf.constant([0., PRIOR_SD**2], F64))
    protected, _, eigenvalues = _regularize_covariance(rank_one, target=prior, config=config.nomination)
    expected = tf.linalg.diag(tf.constant([.1 * PRIOR_SD**2, PRIOR_SD**2], F64))
    tf.debugging.assert_near(healthy, prior, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_near(protected, expected, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_greater_equal(tf.reduce_min(eigenvalues),
        tf.constant(.1 * PRIOR_SD**2 * (1-1e-12), F64))
    return dict(passed=True, healthy_max_change=float(tf.reduce_max(tf.abs(healthy-prior))),
        rank_one_eigenvalues=eigenvalues.numpy().tolist(),
        justification="C=.9 E+.1 P0 >= .1 P0; zero additive jitter; epsilon floor is backup")


def public_tuning_stage(output, *, initialization_repair=False, sequential_recovery=False):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("M13 default force requires TF32")
    adapter, binding, target, fixture, selection, selection_path = selected_components()
    config = tuning_config(adapter.target_scope)
    repair_metadata = None
    if initialization_repair:
        from bayesfilter.testing.surrogate_hmc_m13_initialization_repair import initialization_config
        config, repair_metadata = initialization_config(config)
    if sequential_recovery:
        from dataclasses import replace
        config = replace(config, nomination=replace(config.nomination, seed=(20260928, 1101)),
                         fresh_seed=(20260928, 1102))
    report = dict(status="M13_PUBLIC_TUNING_RUNNING", qualified=False, inference_launched=True,
        plan=repair_metadata["plan"] if repair_metadata else PLAN, initialization_repair=repair_metadata, data_seed=1704, observation_checksum=fixture.observation_checksum,
        selection_path=str(selection_path), selection_sha256=file_hash(selection_path),
        multiplier=selection["multiplier"], controls=selection["controls"],
        sequential_recovery=sequential_recovery, tuning_config=config.payload(), metric_protection=metric_guard_diagnostic(config),
        target_manifest=dict(target.manifest), runner_binding=binding.payload(),
        consumer_sha256=file_hash(Path(__file__)), exact_score_admission=False,
        posterior_qualified=False, confirmation_data_used=False, ranking_supported=False,
        next_action="Existing master: fresh public tuning verification, then retained posterior comparison.")
    write_json(out / "data.json", dict(observations=fixture.observations.numpy().tolist(),
        observation_checksum=fixture.observation_checksum, data_seed=1704,
        model=fixture.model_definition()))
    report["data_sha256"] = file_hash(out / "data.json")
    write_json(output, finite_report(report))
    started = time.monotonic()
    print("M13 public nomination and fresh verification started", flush=True)
    try:
        result = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.), config=config,
            parameter_scales=(PRIOR_SD, PRIOR_SD), runner_binding=binding, output_dir=out / "tuning")
    except ValueError as exc:
        if "TensorFlow nomination failed" not in str(exc):
            raise
        report.update(status="M13_PUBLIC_TUNING_REPAIR_REQUIRED", continuation_veto=False,
            tuning=dict(passed=False, detail=str(exc), wall_seconds=time.monotonic()-started),
            next_action="Existing master: inspect saved nomination and use the bounded tuning repair.")
        return finite_report(report)
    manifest = json.loads(Path(result.artifact_manifest_path).read_text())
    fallbacks = {part: int(tf.reduce_sum(tf.cast(
        read_tensor(out / "tuning", manifest[part]["force_fallback"]), tf.int32)))
        for part in ("warmup", "verification") if manifest[part]}
    report["tuning"] = dict(passed=result.passed, checks=result.checks,
        manifest=result.artifact_manifest_path, manifest_sha256=file_hash(Path(result.artifact_manifest_path)),
        step_size=float(result.nomination.step_size), leapfrog_steps=int(result.nomination.num_leapfrog_steps),
        covariance=result.nomination.covariance.numpy().tolist(),
        adaptation_force_fallback_count=int(result.nomination.adaptation_force_fallback_count),
        nomination_force_fallback_count=int(result.nomination.verification_force_fallback_count),
        fresh_force_fallback_counts=fallbacks,
        sequential_status=manifest["sequence"]["status"],
        warmup_ready=manifest["sequence"]["warmup_ready"],
        draws_per_chain=manifest["sequence"]["counts"],
        rhat={k: read_tensor(out / "tuning", v).numpy().tolist() for k, v in manifest["rhat"].items()},
        wall_seconds=time.monotonic()-started)
    if manifest["sequence"]["status"] in ("HEALTH_VETO", "BUDGET_EXHAUSTED"):
        report.update(status="M13_PUBLIC_TUNING_" + manifest["sequence"]["status"],
            continuation_veto=True,
            next_action="Existing master: refresh the phase plan with the terminal cause; no retained launch.")
    elif any(fallbacks.values()):
        report.update(status="M13_PUBLIC_TUNING_FORCE_REPAIR_REQUIRED", continuation_veto=False,
            next_action="Existing master: localize flagged force fallback and use the bounded repair before retention.")
    elif not result.passed:
        report.update(status="M13_PUBLIC_TUNING_REPAIR_REQUIRED", continuation_veto=False,
            next_action="Existing master: inspect fresh verification and use the bounded tuning repair.")
    else:
        verified = load_verified_tensorflow_hmc_result(result.artifact_manifest_path,
            adapter=adapter, runner_binding=binding)
        if not verified.passed:
            raise ValueError("public tuning reload changed the decision")
        report.update(status="M13_PUBLIC_TUNING_VERIFIED", qualified=True,
            next_action="Existing master: retained chunks and independent posterior comparison; tuning alone is not posterior qualification.")
    return finite_report(report)
