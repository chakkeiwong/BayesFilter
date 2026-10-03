"""M27 independent saved-evidence audit; no sampler execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def audit_runtime(runtime, device):
    assert runtime["device_scope"] == device
    assert runtime["memory_policy"]["tf_force_gpu_allow_growth"] == "true"
    assert runtime["cpu_threads"] == {"inter_op": "1", "intra_op": "1"}
    if device == "cpu_reference":
        assert runtime["gpu_intentionally_hidden"] and runtime["visible_devices"] == "-1"
        assert runtime["jit_compile"] is False
        assert runtime["memory_policy"]["physical_devices"] == []
    else:
        assert runtime["jit_compile"] and not runtime["gpu_intentionally_hidden"]
        assert "GPU:0" in runtime["gpu_tensor_device"]
        assert runtime["memory_policy"]["physical_devices"]
        assert runtime["physical_gpu_details"]
        assert runtime["memory_policy"]["configured_before_logical_device_initialization"]
        assert runtime["memory_policy"]["all_physical_devices_memory_growth"]
        assert all(d["memory_growth"] is True for d in runtime["memory_policy"]["physical_devices"])


def audit_fit(path):
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    output = read(path / "pipeline.json")
    tuning = read(path / "tuning/candidate_set_result.json")
    checkpoint = read(path / "tuning/tuning_checkpoint.json")
    assert check_inventory(tuning)["failures"] == []
    for payload, key in ((tuning, "result_hash"), (checkpoint, "content_hash")):
        body = dict(payload)
        assert body.pop(key) == digest(body)
    for key in ("scope", "config", "candidates", "candidate_states", "work_items",
                "verification_receipts", "verified_candidate_ids", "completion_status"):
        assert tuning[key] == checkpoint["result"][key], key
    assert set(output["verified_candidate_ids"]) == set(tuning["verified_candidate_ids"])
    assert {m["candidate_id"] for m in output["members"]} == set(tuning["verified_candidate_ids"])
    selection = read(path / "posterior_selection.json")
    verified = {c["candidate_id"]: c for c in tuning["candidates"]
                if c["candidate_id"] in tuning["verified_candidate_ids"]}
    if selection["rule"] == "shortest_verified_l":
        by_l = {}
        for cid in sorted(verified):
            by_l.setdefault(verified[cid]["leapfrog_steps"], cid)
        expected = [by_l[length] for length in sorted(by_l)[:selection["requested_member_count"]]]
        assert selection["selection_shortfall"] == selection["requested_member_count"] - len(expected)
    else:
        assert selection["rule"] == "first_verified"
        expected = sorted(verified)[:1]
    assert output["selection"]["candidate_ids"] == selection["selected_candidate_ids"] == expected
    assert [m["candidate_id"] for m in output["members"] if m["status"] == "assessed"] == expected
    assert set(p.name for p in (path / "members").iterdir()) == set(expected)
    for cid in expected:
        assert (path / "posterior_selection.json").stat().st_mtime_ns <= (
            path / "members" / cid / "result.json").stat().st_mtime_ns
    evidence = {}
    for expected in checkpoint["numerical_evidence_hashes"]:
        value = read(path / "tuning/numerical_evidence" / (expected + ".json"))
        assert digest(value) == expected
        evidence[expected] = value
    for receipt in tuning["verification_receipts"]:
        value = evidence[receipt["numerical_evidence_hash"]]
        assert receipt["seed_lineage"] == value["seed"]
        assert receipt["candidate_record_hash"] == value["candidate"]["candidate_record_hash"]
    tensor_hashes = {}
    for p in sorted([*path.rglob("*.tensor"), *path.rglob("*.bin")]):
        tensor_hashes[str(p.relative_to(path))] = sha(p)
        metadata = Path(str(p) + ".json")
        if metadata.exists():
            assert read(metadata)["sha256"] == sha(p)
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
    members = []
    for member in output["members"]:
        if member["status"] != "assessed":
            assert member["status"] == "unassessed_by_design"
            continue
        posterior = member["posterior"]
        assert member["warmup_exclusion_matches"] and posterior["warmup_excluded_from_posterior"]
        members.append({"candidate_id": member["candidate_id"], "L": member["L"],
            "epsilon": member["epsilon"], "passed": posterior["passed"],
            "hard_vetoes": posterior["hard_vetoes"], "warmup_cap": posterior["warmup_cap_hit"],
            "retained_cap": posterior["retained_cap_hit"],
            "warmup": posterior["warmup_results_per_chain"], "retained": posterior["retained_results_per_chain"],
            "fixed_status": member.get("fixed_comparator", {}).get("status"),
            "final_check": posterior["retained_checks"][-1].get(
                posterior["retained_checks"][-1].get("diagnostic_role", "modern_rhat"))
                if posterior["retained_checks"] else None})
    return {"candidate_count": len(tuning["candidates"]), "verified_count": len(tuning["verified_candidate_ids"]),
            "receipts": len(evidence), "tensor_count": len(tensor_hashes),
            "selected_posteriors": members, "tuning_completion": tuning["completion_status"]}, tuning, evidence, tensor_hashes


def parity(left, right):
    a, x, xe, xt = audit_fit(left)
    b, y, ye, yt = audit_fit(right)
    assert xt == yt, "saved tensors differ"
    assert x["completion_status"] == y["completion_status"] == "complete"
    for key in ("scope", "config", "candidates", "candidate_states", "verified_candidate_ids"):
        assert x[key] == y[key], key
    assert len(x["verification_receipts"]) == len(y["verification_receipts"])
    for u, v in zip(x["verification_receipts"], y["verification_receipts"]):
        assert {k:w for k,w in u.items() if k != "numerical_evidence_hash"} == {
            k:w for k,w in v.items() if k != "numerical_evidence_hash"}
        xx, yy = xe[u["numerical_evidence_hash"]], ye[v["numerical_evidence_hash"]]
        for key in ("candidate", "seed", "initial_state", "samples", "trace", "analysis", "rhat_reporting_only"):
            assert xx[key] == yy[key], (u["candidate_id"], key)
    # Final diagnostic arrays/decisions are compared without runtime clocks.
    assert a == b
    return {"left": str(left), "right": str(right), "exact_tensor_and_numerical_parity": True, **a,
            "excluded_from_comparison": ["elapsed/runtime clocks", "output paths",
                "full binding/evidence integrity hashes that include clocks (individually verified)"]}


def audit_execution(root, source_files, device, mode, design):
    execution = read(root / "execution.json")
    assert execution["exit_code"] == 0
    meter = read(root / "manifest.json")
    assert meter["source_hashes"] == source_files
    assert meter["environment"]["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    fit = root / "fits" / "replication-0000"
    if mode == "isolated":
        manifest = read(fit / "process-attempt-001-manifest.json")
        receipt = read(fit / "process-attempt-001-exit.json")
        assert receipt["status"] == "complete" and receipt["exit_code"] == 0
        assert receipt["assessment_sha256"] == sha(fit / "independent_assessment.json")
        resources = read(fit / "process-attempt-001-resources.json")
        allocator = resources["after"].get("gpu_allocator_bytes")
    else:
        manifest = read(root / "fits" / "manifest.json")
        allocator = read(root / "fits" / "resources.json")[-1].get("gpu_allocator_bytes")
    assert manifest["source"]["files"] == source_files
    audit_runtime(manifest["runtime"], device)
    if device == "gpu":
        assert meter["gpu_preflight"]["trust_basis"] == "trusted_escalated_tool_execution"
        assert allocator and all(x["peak"] >= x["current"] >= 0 for x in allocator.values())
    return {"mode": mode, "runtime": manifest["runtime"], "elapsed_seconds": execution["elapsed_seconds"],
            "allocator_bytes": allocator, "normal_exit": True, "design_seed": design["seed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh audit output required")
    from bayesfilter.testing.inference_validation.designs import ValidationDesign
    from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications
    started = time.monotonic()
    summary = {"numerical_execution": False, "cpu": [], "gpu": [], "errors": []}
    cpu_source = read(Path("/tmp/bayesfilter-hmc-m27-source-r1-20260923/source_snapshot.json"))["files"]
    for design_payload in read(args.root / "rotated-sibling-suite.json")["designs"]:
        design = ValidationDesign.from_payload(design_payload)
        paths = [args.root / run / "fits" / design.design_id / "replication-0000"
                 for run in ("rotated-sibling-r2", "rotated-sibling-r3")]
        row = parity(*paths)
        original = read(paths[0].parent / "attempt-001-result.json")["assessment"]
        corrected = summarize_replications(design, original["replications"])
        assert corrected["finding"] == "pipeline_assessed"
        assert corrected["assessment_complete"] and corrected["comparison_complete"]
        for slot in corrected["member_slot_assessments"].values():
            assert slot["planned"] == 1 and slot["posterior_checks_passed"] == 1
        for path in paths:
            manifest = read(path / "process-attempt-001-manifest.json")
            assert manifest["source"]["files"] == cpu_source
            audit_runtime(manifest["runtime"], "cpu_reference")
            receipt = read(path / "process-attempt-001-exit.json")
            assert receipt["status"] == "complete" and receipt["exit_code"] == 0
            assert receipt["assessment_sha256"] == sha(path / "independent_assessment.json")
        summary["cpu"].append({"design_id": design.design_id, "independent_fits": 1,
            "duplicate_replay_fits": 1, "parity": row, "original_finding": original["finding"],
            "corrected_summary": corrected, "original_result_sha256": sha(paths[0].parent / "attempt-001-result.json")})
    if args.gpu:
        files = read(args.root / "source-r2-manifest.json")["files"]
        for target, name, seed in (("gaussian", "gaussian", 2026092283),
                                   ("beta_binomial", "beta-binomial", 2026092284)):
            design = read(args.root / (f"m27-gpu-{name}-lugsail-{seed}.json"))
            roots = [args.root / f"{target}-{mode}-gpu-r1" for mode in ("persistent", "isolated")]
            row = parity(*(root / "fits/replication-0000" for root in roots))
            runs = [audit_execution(root, files, "gpu", mode, design)
                    for mode, root in zip(("persistent", "isolated"), roots)]
            assert runs[0]["runtime"] == runs[1]["runtime"], "paired GPU runtime differs"
            preflights = [read(root / "manifest.json")["gpu_preflight"] for root in roots]
            assert preflights[0]["device"].split(",")[:3] == preflights[1]["device"].split(",")[:3], "paired GPU hardware differs"
            summary["gpu"].append({"target": target, "parity": row, "runs": runs})
    summary.update(elapsed_seconds=time.monotonic()-started, engineering_audit_passed=True,
                   posterior_coverage_established=False, ranking_supported=False,
                   heuristic_dominance="not_established; development and compatibility evidence only")
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"cpu_fits": len(summary["cpu"]), "gpu_pairs": len(summary["gpu"]),
                      "engineering_audit_passed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
