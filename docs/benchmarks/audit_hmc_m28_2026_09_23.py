"""Saved-data M28 audit: source, target, receipts, counts, slots and archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from docs.benchmarks.audit_hmc_m27_2026_09_23 import read, sha, digest, audit_runtime
from docs.benchmarks.prepare_hmc_m28_2026_09_23 import check_counts


def audit_case(root, design, source, prior_search):
    import tensorflow as tf
    from bayesfilter.inference import load_frozen_neutra_artifact
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.procedures import initial_starts
    from bayesfilter.testing.inference_validation.storage import read_tensor

    name = design["design_id"]
    kind = design["options"]["transport_payload"]["construction"]["kind"]
    attempt = root / f"gpu-{kind}-{design['seed']}-r3"
    path = attempt / "fits" / name / "replication-0000"
    execution = read(attempt / "execution.json")
    meter = read(attempt / "manifest.json")
    assert meter["source_hashes"] == source["source_hashes"]
    assert meter["gpu_preflight"]["trust_basis"] == "trusted_escalated_tool_execution"
    assert meter["environment"]["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    manifest = read(path / "process-attempt-001-manifest.json")
    audit_runtime(manifest["runtime"], "gpu")
    assert manifest["source"]["files"] == source["source_hashes"]
    receipt = read(path / "process-attempt-001-exit.json")
    if receipt.get("assessment_sha256"):
        assert receipt["assessment_sha256"] == sha(path / "independent_assessment.json")
    report = {"design_id": name, "map": kind, "seed": design["seed"], "fit": str(path),
        "normal_exit": execution["exit_code"] == receipt["exit_code"] == 0,
        "elapsed_seconds": execution["elapsed_seconds"], "selected_slots": [],
        "planned_slots": 2, "posterior_passes": 0}
    if not (path / "pipeline.json").exists():
        report.update(status="execution_incomplete", process_receipt=receipt)
        return report
    output = read(path / "pipeline.json")
    tuning = read(path / "tuning/candidate_set_result.json")
    checkpoint = read(path / "tuning/tuning_checkpoint.json")
    assert not check_inventory(tuning)["failures"]
    for payload, key in ((tuning, "result_hash"), (checkpoint, "content_hash")):
        body = dict(payload)
        assert body.pop(key) == digest(body)
    for key in ("scope", "config", "candidates", "candidate_states", "work_items",
                "verification_receipts", "verified_candidate_ids", "completion_status"):
        assert tuning[key] == checkpoint["result"][key], key
    assert {k: v for k, v in tuning["config"].items() if k != "max_wall_time_seconds"} == {
        k: v for k, v in prior_search.items() if k != "max_wall_time_seconds"}
    assert tuning["scope"]["use_xla"] is True
    verified_ids = set(tuning["verified_candidate_ids"])
    assert set(output["verified_candidate_ids"]) == verified_ids
    assert {m["candidate_id"] for m in output["members"]} == verified_ids
    by_length = {}
    candidates = {c["candidate_id"]: c for c in tuning["candidates"]}
    for cid in sorted(verified_ids):
        by_length.setdefault(candidates[cid]["leapfrog_steps"], cid)
    selected = [by_length[length] for length in sorted(by_length)[:2]]
    selection = read(path / "posterior_selection.json")
    assert selection["selected_candidate_ids"] == output["selection"]["candidate_ids"] == selected
    assert selection["selection_shortfall"] == 2 - len(selected)

    evidence = {}
    for expected in checkpoint["numerical_evidence_hashes"]:
        value = read(path / "tuning/numerical_evidence" / (expected + ".json"))
        assert digest(value) == expected
        evidence[expected] = value
    for verification in tuning["verification_receipts"]:
        value = evidence[verification["numerical_evidence_hash"]]
        assert verification["seed_lineage"] == value["seed"]
        assert verification["candidate_record_hash"] == value["candidate"]["candidate_record_hash"]

    starts = read(path / "start_coordinates.json")
    target = ValidationTarget(design["scenario"]["target"], design["scenario"]["parameters"], jit_compile=False)
    transport = load_frozen_neutra_artifact(design["options"]["transport_payload"],
        expected_target_signature=target.adapter_signature()).transport
    latent = tf.constant(starts["latent_starts"], tf.float64)
    model = tf.constant(starts["model_starts"], tf.float64)
    tf.debugging.assert_near(target.to_model(transport.forward_batch(latent)), model, atol=2e-11, rtol=2e-11)
    tf.debugging.assert_near(model, initial_starts(ValidationTarget("funnel", jit_compile=False), "dispersed"),
                             atol=2e-11, rtol=2e-11)
    assert starts["target_signature"] == tuning["scope"]["target_signature"] == target.adapter_signature()
    report["scope"] = {key: tuning["scope"][key] for key in
        ("target_signature", "mass_signature", "target_preparation_identity", "start_bank_signature")}

    tensor_hashes = {str(p.relative_to(path)): sha(p)
                     for p in sorted([*path.rglob("*.tensor"), *path.rglob("*.bin")])}
    for relative, expected in tensor_hashes.items():
        metadata = Path(str(path / relative) + ".json")
        if metadata.exists():
            assert read(metadata)["sha256"] == expected
    for bundle in path.rglob("bundle.json"):
        assert bundle.with_suffix(".sha256").read_text().strip() == sha(bundle)
        def check_tree(value):
            if isinstance(value, dict):
                if "tensor" in value:
                    entry = value["tensor"]
                    assert sha(bundle.parent / entry["file"]) == entry["sha256"]
                else:
                    for child in value.values():
                        check_tree(child)
            elif isinstance(value, list):
                for child in value:
                    check_tree(child)
        check_tree(read(bundle)["tree"])

    members = {m["candidate_id"]: m for m in output["members"]}
    for slot in range(2):
        if slot >= len(selected):
            report["selected_slots"].append({"slot": slot + 1, "status": "selection_shortfall"})
            continue
        cid = selected[slot]
        member = members[cid]
        row = {"slot": slot + 1, "candidate_id": cid, "L": member["L"],
               "epsilon": member["epsilon"], "status": member["status"]}
        if member["status"] == "assessed":
            posterior = member["posterior"]
            config = posterior["config"]
            for key, count in design["options"]["posterior_settings"].items():
                assert config[key] == count, key
            assert config["max_results_per_chain"] == 60000 and config["count_budget_reason"]
            precision = config["assessment_policy"]["precision"]
            assert precision["method"] == "lugsail"
            assert {(t["name"], t["kind"]) for t in precision["targets"]} == {
                (name, kind) for name in target.spec.parameters for kind in ("mean", "quantile")}
            assert all(t["mcse_absolute_max"] == .05 for t in precision["targets"])
            assert member["warmup_exclusion_matches"] and posterior["warmup_excluded_from_posterior"]
            chunks = []
            for bundle in sorted((path / "members" / cid / "posterior_chunks/committed").glob("chunk-*/bundle.json")):
                entry = read(bundle)["tree"]["sequence"][0]["tensor"]
                chunks.append(tf.io.parse_tensor(tf.io.read_file(str(bundle.parent / entry["file"])),
                                                 out_type=tf.float64))
            latent_draws = tf.concat(chunks, 0) if chunks else tf.zeros([0, 4, 3], tf.float64)
            total = posterior["warmup_results_per_chain"] + posterior["retained_results_per_chain"]
            assert int(latent_draws.shape[0]) == total
            reconstructed = tf.reshape(target.to_model(transport.forward_batch(
                tf.reshape(latent_draws, [-1, 3]))), tf.shape(latent_draws))
            warmup_count = posterior["warmup_results_per_chain"]
            for saved_path, expected_draws in ((member["warmup_path"], reconstructed[:warmup_count]),
                                               (member["draws_path"], reconstructed[warmup_count:])):
                saved = read_tensor(saved_path)
                tf.debugging.assert_equal(tf.shape(saved), tf.shape(expected_draws))
                tf.debugging.assert_equal(tf.math.is_finite(saved), tf.math.is_finite(expected_draws))
                tf.debugging.assert_equal(tf.math.is_nan(saved), tf.math.is_nan(expected_draws))
                tf.debugging.assert_near(tf.where(tf.math.is_finite(saved), saved, 0.),
                    tf.where(tf.math.is_finite(expected_draws), expected_draws, 0.), atol=2e-10, rtol=2e-10)
            row["model_draws_reconstructed_from_latent_checkpoints"] = True
            row["warmup_exclusion_independently_checked"] = True
            fixed = member.get("fixed_comparator", {})
            if fixed.get("status") == "assessed":
                for key, count in design["options"]["fixed_comparator"].items():
                    assert fixed[key] == count, key
            row.update(passed=posterior["passed"], decision=posterior["decision"],
                hard_vetoes=posterior["hard_vetoes"], warmup=posterior["warmup_results_per_chain"],
                retained=posterior["retained_results_per_chain"],
                warmup_cap=posterior["warmup_cap_hit"], retained_cap=posterior["retained_cap_hit"],
                precision_status=posterior["precision_status"], fixed_status=fixed.get("status"),
                timing=member["timing"],
                final_check=posterior["retained_checks"][-1] if posterior["retained_checks"] else None)
            report["posterior_passes"] += int(posterior["passed"])
        report["selected_slots"].append(row)
    assert all(members[cid]["status"] == "unassessed_by_design" for cid in verified_ids - set(selected))
    report.update(status="audited", tuning_completion=tuning["completion_status"],
        candidates=len(candidates), verified=len(verified_ids), receipts=len(evidence),
        tensor_count=len(tensor_hashes), tensor_inventory_digest=digest(tensor_hashes),
        unassessed_by_design=len(verified_ids - set(selected)),
        independent_assessment=read(path / "independent_assessment.json"),
        stage_timing=output.get("timing"), selection_shortfall=2-len(selected))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design-id", help="audit one completed cell while other cells run")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh audit output required")
    started = time.monotonic()
    source = read(args.root / "source-r1-manifest.json")
    for relative, expected in source["source_hashes"].items():
        assert sha(args.root / "source-r1" / relative) == expected
    plan = read(args.root / "public-plan-r1/plan.json")
    designs = [job["design"] for job in plan["jobs"]]
    counts = check_counts(designs, (args.root / "source-r1/docs/plans/bayesfilter-hmc-post-m27-next-phase-2026-09-23.md").read_text())
    if args.design_id:
        designs = [design for design in designs if design["design_id"] == args.design_id]
        if len(designs) != 1:
            raise ValueError("unknown or duplicate M28 design")
    prior = read(args.root.parent / "m25-r1/gpu-exact-2251-r1/worker/fit/tuning/candidate_set_result.json")
    reports, errors = [], []
    for design in designs:
        try:
            reports.append(audit_case(args.root, design, source, prior["config"]))
        except Exception as exc:
            errors.append({"design_id": design["design_id"], "type": type(exc).__name__, "reason": str(exc)})
    if not errors and len(reports) == 4 and all(r["status"] == "audited" for r in reports):
        for seed in (2026092381, 2026092382):
            pair = [r for r in reports if r["seed"] == seed]
            assert pair[0]["scope"]["target_signature"] == pair[1]["scope"]["target_signature"]
            for key in ("mass_signature", "target_preparation_identity", "start_bank_signature"):
                assert pair[0]["scope"][key] != pair[1]["scope"][key], key
    result = {"phase": "M28", "count_check": counts, "planned_fits": len(designs),
        "planned_member_slots": 2 * len(designs),
        "reports": reports, "errors": errors, "evidence_integrity_passed": not errors,
        "all_fits_normal_exit": len(reports) == len(designs) and all(r["normal_exit"] for r in reports),
        "numerical_sampling_performed_by_audit": False, "elapsed_seconds": time.monotonic()-started,
        "coverage_established": False, "ranking_supported": False, "default_promotion": False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("evidence_integrity_passed", "all_fits_normal_exit", "errors")}))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
