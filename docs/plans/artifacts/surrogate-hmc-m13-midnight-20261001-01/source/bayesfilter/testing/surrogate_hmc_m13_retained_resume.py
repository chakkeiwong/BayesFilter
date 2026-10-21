"""Resume retained M13 archives with the same public runner and assessment.

The fresh route delegates to the original consumer. The resumed route replays
validated archives through the same cumulative look schedule before drawing
new transitions. It changes no kernel, health check or assessment formula.
"""
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
from bayesfilter.testing.surrogate_hmc_m13_staged_assessment import LOOKS, assess, assessment_scope
from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import load_reference

PLAN = "docs/plans/surrogate-hmc-m13-retained-assessment-2026-09-26.md"
SEED = (20260926, 2910)


def validated_prefix(path, tuning, binding, reference, looks):
    """Recheck every archived state, tensor checksum, health result and seed."""
    from bayesfilter.inference import hmc_tensorflow_verified as verified
    from bayesfilter.inference import hmc_tensorflow_tuning as stage
    old = json.loads(Path(path).read_text())
    manifest_path = Path(tuning.artifact_manifest_path).resolve()
    tuning_payload = json.loads(manifest_path.read_text())
    if (old.get("status") != "M13_RETAINED_BUDGET_EXHAUSTED"
            or old.get("qualified") is not False
            or Path(old["tuning_manifest"]).resolve() != manifest_path
            or old["binding_hash"] != binding.binding_hash
            or old["reference_sha256"] != reference["reference_sha256"]
            or old["planned_looks"] != list(looks) or old["seed"] != list(SEED)
            or old["chunk_size"] != 250 or old["warmup_excluded"] is not True
            or old["draws_per_chain"] != 250*len(old["chunks"])):
        raise ValueError("invalid retained continuation report")
    state = verified._read_chunk(manifest_path.parent, tuning_payload["verification"]).final_state
    used = [list(tuning.config.nomination.seed), list(tuning.config.fresh_seed)]
    used.extend(tuning_payload["sequence"]["used_seeds"])
    previous_path = None
    for index, row in enumerate(old["chunks"]):
        item = Path(row["manifest"]).resolve()
        if file_hash(item) != row["sha256"]:
            raise ValueError("retained prefix manifest checksum mismatch")
        saved = json.loads(item.read_text())
        seed = [SEED[0], SEED[1]+index]
        if (saved.get("schema") != verified.VERIFIED_TENSORFLOW_ARCHIVE_SCHEMA
                or saved.get("health_passed") is not True or row.get("health_passed") is not True
                or row.get("no_force_fallback") is not True
                or saved["tuning_manifest_sha256"] != file_hash(manifest_path)
                or saved["binding_hash"] != binding.binding_hash
                or saved["config"]["num_results"] != 250 or saved["config"]["seed"] != seed
                or saved["continuation_manifest"] != previous_path
                or seed in used or saved["used_seeds"] != used+[seed]):
            raise ValueError("retained prefix identity or seed mismatch")
        initial = stage._read_tensor_record(item.parent, saved["initial_state"], tf.float64)
        tf.debugging.assert_equal(initial, state, message="retained prefix state discontinuity")
        chunk = verified._read_chunk(item.parent, saved["chunk"])
        checks = {k: bool(v) for k,v in verified._health(chunk, initial, tuning.config.energy_error_max).items()}
        if (not all(checks.values()) or checks != saved["checks"] or checks != row["checks"]
                or bool(tf.reduce_any(chunk.force_fallback))):
            raise ValueError("retained prefix health mismatch")
        tf.debugging.assert_equal(chunk.final_state, chunk.states[-1])
        raw = read_tensor(item.parent, saved["raw_samples"])
        tf.debugging.assert_equal(tf.shape(raw), [250,4,2])
        tf.debugging.assert_equal(raw, verified._affine(tuning.nomination).latent_to_position(chunk.states))
        state, previous_path, used = chunk.final_state, str(item), used+[seed]
    return old["chunks"], old

def consume_chunks(tuning, binding, output_dir, reference, calibration, *, looks=LOOKS, max_results=None, resume_report=None):
    """Actual consumer -> public runner -> frozen native kernel, in250-draw chunks."""
    if resume_report is None:
        from .surrogate_hmc_m13_retained import consume_chunks as original
        return original(tuning, binding, output_dir, reference, calibration,
                        looks=looks, max_results=max_results)
    prefix, old_report = validated_prefix(resume_report, tuning, binding, reference, looks)
    looks = tuple(looks)
    maximum = max(looks) if max_results is None else min(max(looks), max_results)
    if old_report["retained_maximum"] != maximum or len(prefix)*250 >= maximum:
        raise ValueError("retained continuation cap changed or already exhausted")
    if type(maximum) is not int or maximum < 250 or maximum % 250:
        raise ValueError("retained cap must be a positive250-draw multiple")
    method = assessment_scope(looks)
    if (calibration.get("qualified") is not True or calibration.get("method") != method
            or calibration.get("looks") != list(looks)
            or calibration.get("reference_sha256") != reference.get("reference_sha256")):
        raise ValueError("missing or mismatched M13 uncertainty qualification")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    runner = build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=tuning, runner_binding=binding)
    parent, chunks, assessments, records = None, [], [], []
    accepted, probabilities, max_energy = [], [], 0.
    started = time.monotonic()
    report = dict(status="M13_RETAINED_RUNNING", qualified=False, posterior_qualified=False,
        plan=PLAN, resumed_from=str(resume_report),
        parent_report_sha256=file_hash(resume_report), replayed_draws_per_chain=len(prefix)*250,
        chunk_size=250, retained_maximum=maximum, planned_looks=list(looks), warmup_excluded=True,
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
    for index in range(maximum//250):
        if index >= len(prefix) and permit is not None and not permit("retained", 250):
            report.update(status="M13_RETAINED_BUDGET_EXHAUSTED", continuation_veto=True,
                next_action="Existing master: resource allowance exhausted; preserve accumulated draws and assessments.")
            break
        config = BoundRetainedHMCArchiveConfig(num_results=250,
            seed=(SEED[0], SEED[1]+index), output_dir=out/f"chunk-{index:02}",
            budget_provenance=f"bounded M13 master allocation; watchdog enforces actual receipt; calibrated maximum{max(looks)}/chain",
            continuation_manifest=parent)
        result = (dict(archive_manifest_path=prefix[index]["manifest"], health_passed=True)
                  if index < len(prefix) else runner.run(config))
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
        if count in looks:
            comparison = assess(tf.transpose(tf.concat(chunks, 0), [1, 0, 2]),
                                reference, calibration=calibration, looks=looks)
            if index < len(prefix):
                old = old_report["assessments"][len(assessments)]
                if json.loads(json.dumps(comparison)) != old:
                    raise ValueError("retained prefix assessment replay mismatch")
                if comparison["status"] in ("EQUIVALENT", "DISCREPANCY"):
                    raise ValueError("retained prefix already had a scientific stopping decision")
            assessments.append(comparison)
            if comparison["status"] == "EQUIVALENT":
                report.update(status="M13_RETAINED_PILOT_QUALIFIED", qualified=True, posterior_qualified=True,
                    next_action="Existing master: record the exploratory result and unresolved all-model gaps.")
            elif comparison["status"] == "DISCREPANCY":
                report.update(status="M13_RETAINED_MATERIAL_DISCREPANCY", continuation_veto=False,
                    next_action="Existing master: localize the discrepancy before any promotion; bounded repair only.")
            elif count == max(looks):
                report.update(status="M13_RETAINED_PRECISION_OR_MIXING_INCONCLUSIVE", continuation_veto=False,
                    next_action="Existing master: assess the remaining bounded repair; no automatic sample extension.")
            print("M13 retained", count, comparison["status"], "Rhat", comparison["rhat"], flush=True)
        if count == maximum and report["status"] == "M13_RETAINED_RUNNING":
            report.update(status="M13_RETAINED_SAMPLE_CAP_INCONCLUSIVE", continuation_veto=False,
                next_action="Existing master: cumulative sample cap reached; preserve all draws and scheduled assessments.")
        report["next_phase_plan"] = dict(
            phase="retained" if report["status"]=="M13_RETAINED_RUNNING" else "refresh_master_phase_plan",
            next_look=next((look for look in looks if look > count), None),
            maximum_per_chain=maximum, criterion="modern Rhat<=1.01, bulk/tail ESS>=400, calibrated equivalence",
            changes_target=False)
        report["wall_seconds"] = time.monotonic()-started
        write_json(out/"report.json", report)
        if report["status"] != "M13_RETAINED_RUNNING":
            break
    report["wall_seconds"] = time.monotonic()-started
    write_json(out/"report.json", report)
    return report

