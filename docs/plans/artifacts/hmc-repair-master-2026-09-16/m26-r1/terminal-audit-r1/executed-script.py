"""Independent M26 source, numerical parity, process and posterior evidence audit."""
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


def audit_runtime(runtime):
    assert runtime["device_scope"] == "cpu_reference"
    assert runtime["gpu_intentionally_hidden"] and runtime["visible_devices"] == "-1"
    assert runtime["jit_compile"] is False
    assert runtime["memory_policy"]["tf_force_gpu_allow_growth"] == "true"
    assert runtime["memory_policy"]["physical_devices"] == []
    assert runtime["cpu_threads"] == {"inter_op": "1", "intra_op": "1"}


def audit_child(fit, manifest, design):
    from bayesfilter.testing.inference_validation.designs import ValidationDesign
    child = read(fit / "process-attempt-001-manifest.json")
    assert child["source"]["identity"] == manifest["identity"]
    assert child["source"]["files"] == manifest["files"]
    assert child["source"]["commit"] == manifest["git_commit"]
    assert child["design_identity"] == ValidationDesign.from_payload(design).identity
    assert child["seed"] == design["seed"]
    audit_runtime(child["runtime"])
    receipt = read(fit / "process-attempt-001-exit.json")
    assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert receipt["assessment_sha256"] == sha(fit / "independent_assessment.json")
    return receipt


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
    assert output["selection"]["candidate_ids"] == sorted(tuning["verified_candidate_ids"])[:1]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", default="source-r2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("gaussian", "beta_binomial"))
    args = parser.parse_args()
    started = time.monotonic()
    manifest = read(args.root / args.source / "source_snapshot.json")
    assert digest(manifest["files"]) == manifest["identity"]
    for path, expected in manifest["files"].items():
        assert sha(args.root / args.source / path) == expected, path
    result = {"source_identity": manifest["identity"], "checked_sources": len(manifest["files"]),
              "parity": [], "pilots": {}, "resource_results": {}}
    targets = (args.target,) if args.target else ("gaussian", "beta_binomial")
    for target in targets:
        roots = [args.root / f"{target}-{mode}-r2" for mode in ("persistent", "isolated")]
        for attempt in roots:
            assert read(attempt / "execution.json")["exit_code"] == 0
            parent = read(attempt / "fits/manifest.json")
            assert parent["source"]["identity"] == manifest["identity"]
            assert parent["source"]["files"] == manifest["files"]
            if parent["mode"] == "persistent":
                audit_runtime(parent["runtime"])
        for rep in range(2):
            result["parity"].append(parity(*(root / "fits" / f"replication-{rep:04d}" for root in roots)))
        resources = read(roots[0] / "fits/resources.json")
        isolated = []
        for rep in range(2):
            path = roots[1] / "fits" / f"replication-{rep:04d}"
            receipt = audit_child(path, manifest, read(roots[1] / "fits/isolated_design.json"))
            isolated.append({"exit": receipt, "resources": read(path / "process-attempt-001-resources.json")})
        result["resource_results"][target] = {"persistent": resources, "isolated": isolated}
    if not args.target:
        for target in ("gaussian", "beta_binomial", "rotated_gaussian"):
            for method in ("lugsail", "autocorrelation"):
                path = args.root / f"{target}-{method}-pilot-r1"
                assert read(path / "execution.json")["exit_code"] == 0
                fit = path / "fits/replication-0000"
                design = read(args.root / f"{target}-{method}-pilot-design.json")
                parent = read(path / "fits/manifest.json")
                assert parent["design"] == read(path / "fits/isolated_design.json") == design
                assert parent["source"]["identity"] == manifest["identity"]
                assert parent["source"]["files"] == manifest["files"]
                row, _, _, _ = audit_fit(fit)
                row["process_exit"] = audit_child(fit, manifest, design)
                assessment = read(path / "fits/assessment.json")
                assert assessment["completed"] == assessment["planned"] == 1
                assert assessment["execution_failures"] == 0
                assert assessment["framework_initialized_in_coordinator"] is False
                row["intervals"] = assessment["interval_coverage_at_stop"]
                row["stopped_versus_fixed"] = assessment["stopped_versus_fixed"]
                assert row["tuning_completion"] == "complete"
                native = read(fit / "pipeline.json")
                selected = [m for m in native["members"] if m["status"] == "assessed"]
                assert len(selected) == 1
                config = selected[0]["posterior"]["config"]
                for key, value in design["options"]["posterior_settings"].items():
                    assert config[key] == value, key
                precision = config["assessment_policy"]["precision"]
                assert precision["method"] == design["options"]["posterior_precision_method"] == method
                assert all(t["mcse_absolute_max"] == design["mcse_tolerance"] for t in precision["targets"])
                assert config["warmup_rhat_max"] == 1.05 and config["retained_rhat_max"] == 1.01
                for key, value in design["options"]["fixed_comparator"].items():
                    assert selected[0]["fixed_comparator"][key] == value, key
                row["source_design_runtime_policy_verified"] = True
                result["pilots"][target + ":" + method] = row
    result.update(elapsed_seconds=time.monotonic()-started, ranking_supported=False,
                  confirmation_established=False, result_scope="bounded CPU engineering parity and development pilots")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"parity_fits": len(result["parity"]), "pilots": len(result["pilots"]),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
