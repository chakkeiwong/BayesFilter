"""M29 saved-evidence parity and host-cost audit; never executes a sampler."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import pstats
import time

from docs.benchmarks.audit_hmc_m27_2026_09_23 import (
    audit_fit, audit_runtime, digest, parity, read, sha,
)

# Declared clock fields only: retain all thresholds, budgets, counts and seeds.
CLOCKS = {"elapsed_seconds", "resume_call_elapsed_seconds", "ensemble_call_s", "per_chain_call_s"}


def strip_clocks(value):
    if isinstance(value, dict):
        return {key: strip_clocks(item) for key, item in value.items() if key not in CLOCKS}
    if isinstance(value, list):
        return [strip_clocks(item) for item in value]
    return value


def evidence_by_work(evidence):
    rows = {}
    semantic_hashes = {}
    for checksum, original in evidence.items():
        value = strip_clocks(original)
        value.pop("binding_hash")  # Full binding contains preparation wall time.
        work = value["work"]["work_item_id"]
        assert work not in rows, "duplicate evidence for work item"
        rows[work] = value
        semantic_hashes[checksum] = digest(value)
    return rows, semantic_hashes


def normalized_tuning(tuning, evidence_hashes):
    value = json.loads(json.dumps(tuning))
    value.pop("result_hash")  # Individually checked by audit_fit.
    value["search_state"].pop("elapsed_seconds")
    for event in value["accounting_events"]:
        event.pop("seconds", None)
    for observation in value["observations"]:
        observation["observation"]["numerical_evidence_hash"] = evidence_hashes[
            observation["observation"]["numerical_evidence_hash"]]
    receipt_hashes = {}
    for receipt in value["verification_receipts"]:
        original_hash = digest(receipt)
        receipt["numerical_evidence_hash"] = evidence_hashes[receipt["numerical_evidence_hash"]]
        receipt_hashes[original_hash] = digest(receipt)
    for repair in value["repair_actions"]:
        if repair.get("source_verification_hash"):
            repair["source_verification_hash"] = receipt_hashes[repair["source_verification_hash"]]
    return value


def normalized_result(value, fit_root):
    """Normalize only known paths, complete binding hash and descriptive clocks."""
    def visit(item):
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()
                    if key not in CLOCKS | {"timing", "source_binding"}}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str) and item.startswith(str(fit_root)):
            return "<fit>" + item[len(str(fit_root)):]
        return item
    return visit(value)


def full_parity(left, right):
    result = parity(left, right)
    _, a, ae, _ = audit_fit(left)
    _, b, be, _ = audit_fit(right)
    aw, ah = evidence_by_work(ae)
    bw, bh = evidence_by_work(be)
    assert aw == bw, "numerical evidence differs"
    assert normalized_tuning(a, ah) == normalized_tuning(b, bh), "tuning lifecycle/observations differ"
    executions = []
    for fit in (left, right):
        spec = read(fit/"tuning/execution_spec.json")
        assert spec["binding_hash"] == digest(spec["execution"])
        assert spec["binding_hash"] == read(fit/"tuning/tuning_checkpoint.json")["binding_hash"]
        execution = spec["execution"]
        execution["config"].pop("preparation_elapsed_seconds")
        # A separate numerical hash excludes bootstrap-probe clocks. Require
        # that numerical geometry hash and every other binding field to match.
        assert execution["preparation"]["numerical_geometry_hash"]
        execution["preparation"].pop("geometry_hash")
        executions.append(execution)
    assert executions[0] == executions[1], "numerical execution binding differs"
    for name in ("pipeline.json", "independent_assessment.json", "posterior_selection.json"):
        assert normalized_result(read(left/name), left) == normalized_result(read(right/name), right), name
    return {**result, "all_observations_and_receipts_equal": True,
            "all_posterior_decisions_and_independent_assessments_equal": True,
            "numerical_receipts_compared": len(aw), "observation_count": len(a["observations"]),
            "clock_fields_excluded": sorted(CLOCKS),
            "other_exclusions": ["accounting event seconds", "stage timing dictionaries",
                "full binding hash containing preparation wall time",
                "result/receipt integrity hashes checked separately, then replaced by numerical digest",
                "absolute output root prefix"]}


def profile_summary(path):
    stats = pstats.Stats(str(path))
    required = {"run_replication", "execute_pipeline", "tune_hmc_kernel", "run_hmc_posterior",
                "run_fixed_comparator"}
    assert required <= {key[2] for key in stats.stats}, "profile omits numerical stages"
    def rows(items):
        return [{"file": key[0], "line": key[1], "function": key[2],
                 "primitive_calls": value[0], "calls": value[1],
                 "self_seconds": value[2], "cumulative_seconds": value[3]}
                for key, value in items]
    selected = [(key, value) for key, value in stats.stats.items()
                if key[2] in required or key[2] in {"_maybe_define_function", "trace_function",
                    "_create_concrete_function", "quick_execute", "write_json", "write_tensor",
                    "assess_hmc_posterior", "resource_snapshot"}]
    by_module = Counter()
    for key, value in stats.stats.items():
        name = key[0].split("/bayesfilter/")[-1] if "/bayesfilter/" in key[0] else key[0]
        by_module[name] += value[2]
    return {"path": str(path), "sha256": sha(path), "python_calls": stats.total_calls,
            "profile_seconds": stats.total_tt, "required_functions_found": sorted(required),
            "stages_and_framework_calls": rows(sorted(selected, key=lambda row: -row[1][3])),
            "top_cumulative": rows(sorted(stats.stats.items(), key=lambda row: -row[1][3])[:25]),
            "top_self": rows(sorted(stats.stats.items(), key=lambda row: -row[1][2])[:25]),
            "module_self_seconds": by_module.most_common(20),
            "interpretation": "Python host frames including waits; nested cumulative times overlap; no device kernel timing"}


def audit_arm(root, target, mode, design, files):
    attempt = root / f"{target}-{mode}-gpu-r1"
    fit = attempt / "fits" / design["design_id"] / "replication-0000"
    outer = read(attempt / "execution.json")
    assert outer["exit_code"] == 0
    meter = read(attempt / "manifest.json")
    assert meter["source_hashes"] == files
    assert meter["gpu_preflight"]["trust_basis"] == "trusted_escalated_tool_execution"
    index = read(attempt / "fits/run_index.json")
    job = index["jobs"][design["design_id"]]
    assert job["status"] == "complete" and len(job["attempts"]) == 1
    assert job["result_sha256"] == sha(job["result"])
    saved_design = read(fit.parent / "design.json")
    assert saved_design == design
    manifest = read(fit / "process-attempt-001-manifest.json")
    assert manifest["source"]["files"] == files
    assert manifest["design_identity"] == digest(design)
    assert manifest["seed"] == design["seed"]
    assert manifest["profiling"]["requested"] == (mode == "on")
    audit_runtime(manifest["runtime"], "gpu")
    receipt = read(fit / "process-attempt-001-exit.json")
    assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert receipt["assessment_sha256"] == sha(fit / "independent_assessment.json")
    resources = read(fit / "process-attempt-001-resources.json")
    allocator = resources["after"]["gpu_allocator_bytes"]
    assert allocator and all(value["peak"] >= value["current"] >= 0 for value in allocator.values())
    assert not (fit.parent / "attempt-001-host.prof").exists()
    summary, _, _, _ = audit_fit(fit)
    if mode == "on":
        assert job["profiling"]["status"] == "available"
        profile = profile_summary(fit / "process-attempt-001-host.prof")
        assert profile["sha256"] == job["profiling"]["profiles"][0]["sha256"]
    else:
        assert not (fit / "process-attempt-001-host.prof").exists()
        profile = None
    pipeline = read(fit / "pipeline.json")
    return fit, {"mode": mode, "normal_exit": True, "elapsed_seconds": outer["elapsed_seconds"],
        "runtime": manifest["runtime"], "resources": resources, "numerical_summary": summary,
        "setup_seconds": manifest["setup_seconds_before_profile"], "profile": profile,
        "pipeline_timing": pipeline["timing"], "member_timing": [
            {"candidate_id": m["candidate_id"], "timing": m["timing"],
             "fixed_seconds": m.get("fixed_comparator", {}).get("elapsed_seconds")}
            for m in pipeline["members"] if m["status"] == "assessed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["gaussian", "beta_binomial"])
    args = parser.parse_args()
    assert not args.output.exists(), "fresh audit output required"
    started = time.monotonic()
    root = args.root.resolve()
    files = read(root / "source-r3-manifest.json")["files"]
    suite = read(root / "designs-r1/suite.json")
    result = {"sampler_executed": False, "pairs": [], "independent_fits": len(args.targets)}
    for design in suite["designs"]:
        target = design["scenario"]["target"]
        if target not in args.targets:
            continue
        left, off = audit_arm(root, target, "off", design, files)
        right, on = audit_arm(root, target, "on", design, files)
        result["pairs"].append({"target": target, "design_id": design["design_id"],
            "parity": full_parity(left, right), "off": off, "on": on,
            "descriptive_profile_overhead_seconds": on["elapsed_seconds"] - off["elapsed_seconds"],
            "confirmation_384_fits_gpu_hours_at_observed_unprofiled_price":
                off["elapsed_seconds"] * 384 / 3600})
    assert len(result["pairs"]) == len(args.targets)
    result.update(passed=True, elapsed_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "pairs": len(result["pairs"]),
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
