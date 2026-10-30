"""Retained M13 surrogate-HMC pilot using the public verified runner only."""
from __future__ import annotations

import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference import (
    load_verified_tensorflow_hmc_result, BoundRetainedHMCArchiveConfig,
    build_retained_bound_hmc_archive_runner_from_tuning_result,
)
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components, read_tensor
from bayesfilter.testing.surrogate_hmc_m13_posterior_assessment import METHOD, LOOKS, assess
from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import load_reference

PLAN = "docs/plans/surrogate-hmc-m13-retained-assessment-2026-09-26.md"
SEED = (20260926, 2910)


def checked_tuning_report(root):
    root = Path(root)
    parent = json.loads((root/"report.json").read_text())
    worker = parent["stages"]["m13-public-tuning"]
    if (parent.get("source_unchanged") is not True or worker.get("execution") != "COMPLETED"
            or worker["result"].get("qualified") is not True):
        raise ValueError("M13 retained pilot requires completed verified public tuning")
    report = worker["result"]
    manifest = Path(report["tuning"]["manifest"])
    if file_hash(manifest) != report["tuning"]["manifest_sha256"]:
        raise ValueError("M13 public tuning manifest changed")
    if any(report["tuning"]["fresh_force_fallback_counts"].values()):
        raise ValueError("M13 fresh public verification used force fallback")
    return report, manifest


def consume_chunks(tuning, binding, output_dir, reference, calibration):
    """Actual consumer -> public runner -> frozen native kernel, in250-draw chunks."""
    if (calibration.get("qualified") is not True or calibration.get("method") != METHOD
            or calibration.get("looks") != list(LOOKS)
            or calibration.get("reference_sha256") != reference.get("reference_sha256")):
        raise ValueError("missing or mismatched M13 uncertainty qualification")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=tuning, runner_binding=binding)
    parent, chunks, assessments, records = None, [], [], []
    accepted, probabilities, max_energy = [], [], 0.
    started = time.monotonic()
    report = dict(status="M13_RETAINED_RUNNING", qualified=False, posterior_qualified=False,
        plan=PLAN, chunk_size=250, retained_maximum=2000, warmup_excluded=True,
        draws_per_chain=0, reference_sha256=reference["reference_sha256"],
        target_scope=tuning.config.nomination.target_scope, seed=list(SEED),
        tuning_manifest=tuning.artifact_manifest_path, binding_hash=binding.binding_hash,
        force_semantics="deterministic position-only analytical DCTR proposal field",
        endpoint_role="separately frozen full-support importance potential for MH",
        exact_endpoint_gradient_claim=False, ranking_supported=False, assessments=assessments,
        chunks=records, numerical_backend="tensorflow_only", jit_compile=True)
    write_json(out/"report.json", report)
    from bayesfilter.inference.hmc_sequential_verification import runtime_budget_guard
    permit = runtime_budget_guard()
    for index in range(8):
        if permit is not None and not permit("retained", 250):
            report.update(status="M13_RETAINED_BUDGET_EXHAUSTED", continuation_veto=True,
                next_action="Existing master: resource allowance exhausted; preserve accumulated draws and assessments.")
            break
        config = BoundRetainedHMCArchiveConfig(num_results=250,
            seed=(SEED[0], SEED[1]+index), output_dir=out/f"chunk-{index:02}",
            budget_provenance="bounded M13 master allocation; watchdog enforces actual receipt; calibrated maximum2000/chain",
            continuation_manifest=parent)
        result = runner.run(config)
        manifest_path = Path(result["archive_manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        fallback = read_tensor(manifest_path.parent, manifest["chunk"]["force_fallback"])
        no_fallback = not bool(tf.reduce_any(fallback))
        record = dict(manifest=str(manifest_path), sha256=file_hash(manifest_path),
            health_passed=result["health_passed"], no_force_fallback=no_fallback,
            checks=manifest["checks"], graph_trace_count=manifest["chunk_graph_trace_count"])
        records.append(record)
        if not result["health_passed"] or not no_fallback:
            invalid = not (manifest["checks"]["finite"] and manifest["checks"]["endpoint_valid"])
            report.update(status="M13_RETAINED_HEALTH_REPAIR_REQUIRED", continuation_veto=invalid,
                failure_class="numerical_or_target_invalidity" if invalid else "candidate_health_failure",
                failed_chunk=index, next_action="Existing master: inspect the archived health failure and use bounded repair.")
            break
        raw = read_tensor(manifest_path.parent, manifest["raw_samples"])
        tf.debugging.assert_equal(tf.shape(raw), [250, 4, 2])
        chunks.append(raw)
        accepted.append(read_tensor(manifest_path.parent, manifest["chunk"]["is_accepted"]))
        probabilities.append(tf.exp(tf.minimum(tf.constant(0., tf.float64),
            read_tensor(manifest_path.parent, manifest["chunk"]["log_accept_ratio"]))))
        max_energy = max(max_energy, float(tf.reduce_max(tf.abs(
            read_tensor(manifest_path.parent, manifest["chunk"]["delta_h"])))))
        parent = manifest_path
        count = 250*len(chunks)
        report.update(draws_per_chain=count, maximum_absolute_energy_error=max_energy,
            acceptance_per_chain=tf.reduce_mean(tf.cast(tf.concat(accepted, 0), tf.float64), 0).numpy().tolist(),
            acceptance_probability_per_chain=tf.reduce_mean(tf.concat(probabilities, 0), 0).numpy().tolist())
        if count in LOOKS:
            comparison = assess(tf.transpose(tf.concat(chunks, 0), [1, 0, 2]),
                                reference, calibration=calibration)
            assessments.append(comparison)
            if comparison["status"] == "EQUIVALENT":
                report.update(status="M13_RETAINED_PILOT_QUALIFIED", qualified=True, posterior_qualified=True,
                    next_action="Existing master: record the exploratory result and unresolved all-model gaps.")
            elif comparison["status"] == "DISCREPANCY":
                report.update(status="M13_RETAINED_MATERIAL_DISCREPANCY", continuation_veto=False,
                    next_action="Existing master: localize the discrepancy before any promotion; bounded repair only.")
            elif count == 2000:
                report.update(status="M13_RETAINED_PRECISION_OR_MIXING_INCONCLUSIVE", continuation_veto=False,
                    next_action="Existing master: assess the remaining bounded repair; no automatic sample extension.")
            print("M13 retained", count, comparison["status"], "Rhat", comparison["rhat"], flush=True)
        report["next_phase_plan"] = dict(
            phase="retained" if report["status"]=="M13_RETAINED_RUNNING" else "refresh_master_phase_plan",
            next_look=next((look for look in LOOKS if look > count), None),
            maximum_per_chain=2000, criterion="modern Rhat<=1.01, bulk/tail ESS>=400, calibrated equivalence",
            changes_target=False)
        report["wall_seconds"] = time.monotonic()-started
        write_json(out/"report.json", report)
        if report["status"] != "M13_RETAINED_RUNNING":
            break
    report["wall_seconds"] = time.monotonic()-started
    write_json(out/"report.json", report)
    return report


def retained_stage(tuning_root, output_dir, reference_path, coverage_path, baseline_report):
    report, manifest = checked_tuning_report(tuning_root)
    reference, _, _ = load_reference(reference_path)
    calibration = json.loads(Path(coverage_path).read_text())
    baseline = json.loads(Path(baseline_report).read_text())
    if (calibration.get("qualified") is not True or baseline.get("health_passed") is not True
            or baseline.get("reference_sha256") != reference["reference_sha256"]):
        raise ValueError("retained prerequisite reference, coverage or classical comparator is invalid")
    adapter, binding, _, fixture, _, _ = selected_components()
    if fixture.observation_checksum != reference["observation_checksum"]:
        raise ValueError("M13 retained/reference data mismatch")
    tuning = load_verified_tensorflow_hmc_result(manifest, adapter=adapter, runner_binding=binding)
    result = consume_chunks(tuning, binding, output_dir, reference, calibration)
    result.update(public_tuning_report_sha256=file_hash(Path(tuning_root)/"report.json"),
        coverage_sha256=file_hash(Path(coverage_path)), baseline_report_sha256=file_hash(Path(baseline_report)),
        classical_comparator_qualified=baseline["qualified"], exploration_data_seed=1704,
        untouched_confirmation_seed_3701_used=False,
        scope_limit="exploratory same-data finite-reference pilot; not all-model or efficiency promotion")
    write_json(Path(output_dir)/"report.json", result)
    return result
