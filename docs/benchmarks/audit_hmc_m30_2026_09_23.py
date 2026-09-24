"""M30 saved-evidence audit. No numerical execution or source-check overrides."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path

from docs.benchmarks.audit_hmc_m27_2026_09_23 import audit_fit, audit_runtime, digest, parity, read, sha
from docs.benchmarks.audit_hmc_m29_2026_09_23 import CLOCKS, normalized_tuning


def normalize(value, *, expected_l, fit_root=None):
    """Drop only named clocks and independently checked execution telemetry."""
    if isinstance(value, dict):
        value = dict(value)
        if value.get("runtime") == "tfp.mcmc.sample_chain_independent_chain_ensemble":
            if value.pop("dynamic_num_leapfrog_steps", False):
                assert value.pop("num_leapfrog_steps") == expected_l
                assert value.pop("num_leapfrog_steps_source") == "runtime_tensor_argument"
            value.pop("ensemble_call_count")
        return {k: normalize(v, expected_l=expected_l, fit_root=fit_root) for k, v in value.items()
                if k not in CLOCKS | {"includes_first_runner_trace_or_compilation", "timing", "source_binding"}}
    if isinstance(value, list):
        return [normalize(v, expected_l=expected_l, fit_root=fit_root) for v in value]
    if isinstance(value, str) and fit_root is not None and value.startswith(str(fit_root)):
        return "<fit>" + value[len(str(fit_root)):]
    return value


def numerical_rows(evidence):
    rows, hashes = {}, {}
    for checksum, original in evidence.items():
        value = normalize(original, expected_l=original["candidate"]["leapfrog_steps"])
        value.pop("binding_hash")
        work = value["work"]["work_item_id"]
        assert work not in rows
        rows[work] = value
        hashes[checksum] = digest(value)
    return rows, hashes


def full_pair(left, right):
    result = parity(left, right)
    a_summary, a, ae, _ = audit_fit(left)
    _, b, be, _ = audit_fit(right)
    aw, ah = numerical_rows(ae)
    bw, bh = numerical_rows(be)
    assert aw == bw, "complete numerical evidence differs"
    assert normalized_tuning(a, ah) == normalized_tuning(b, bh), "lifecycle or verification receipts differ"
    executions = []
    for index, fit in enumerate((left, right)):
        payload = read(fit / "tuning/execution_spec.json")
        assert payload["binding_hash"] == digest(payload["execution"])
        assert payload["binding_hash"] == read(fit / "tuning/tuning_checkpoint.json")["binding_hash"]
        execution = payload["execution"]
        assert execution["config"].pop("reuse_leapfrog_graphs", False) is bool(index)
        execution["config"].pop("preparation_elapsed_seconds")
        execution["preparation"].pop("geometry_hash")
        for path, checksum in execution["source_closure"].items():
            assert sha(path) == checksum, path
        executions.append(execution)
    assert executions[0] == executions[1], "execution, scope, sources or geometry differ"
    selected_l, = [row["L"] for row in a_summary["selected_posteriors"]]
    for name in ("pipeline.json", "posterior_selection.json", "independent_assessment.json"):
        assert normalize(read(left / name), expected_l=selected_l, fit_root=left) == normalize(
            read(right / name), expected_l=selected_l, fit_root=right), name
    return {**result, "observations_and_receipts_equal": True, "posterior_decisions_equal": True,
            "numerical_receipts_compared": len(aw), "observation_count": len(a["observations"]),
            "source_closure_and_full_geometry_equal": True}


def audit_arm(path, strategy, files):
    outer = read(path / "execution.json")
    assert outer["exit_code"] == 0
    outer_manifest = read(path / "manifest.json")
    assert outer_manifest["source_hashes"] == files
    root = path / "fits"
    manifest = read(root / "full-fit-manifest.json")
    receipt = read(root / "full-fit-exit.json")
    assert manifest["source"]["files"] == files
    assert manifest["runner_strategy"] == strategy
    assert receipt["status"] == "complete" and receipt["error"] is None
    assert receipt["manifest_sha256"] == sha(root / "full-fit-manifest.json")
    audit_runtime(manifest["runtime"], "gpu")
    fit = root / "replication-0000"
    summary, _, evidence, _ = audit_fit(fit)
    for item in evidence.values():
        for runtime in item["runtime"]:
            assert runtime["execution_mode"] == "serial" and runtime["use_xla"]
            assert runtime.get("dynamic_num_leapfrog_steps", False) is (strategy == "dynamic")
            if strategy == "dynamic":
                assert runtime["num_leapfrog_steps"] == item["candidate"]["leapfrog_steps"]
    resources = read(root / "full-fit-resources.json")
    return {"fit": str(fit), "design_identity": manifest["design_identity"],
        "design_file_sha256": manifest["design_file_sha256"], "source_identity": manifest["source"]["identity"],
        "summary": summary, "outer_seconds": outer["elapsed_seconds"],
        "runtime": manifest["runtime"], "resources": resources}


def replay_summary(path):
    result = read(path / "result.json")
    receipt = read(path / "execution.json")
    assert receipt["exit_code"] == 0
    manifest = read(path / "manifest.json")
    assert manifest["source_hashes"] == result["source"]["files"]
    source = Path(manifest["cwd"])
    assert read(source / "source_snapshot.json")["files"] == result["source"]["files"]
    assert all(sha(source / name) == checksum for name, checksum in result["source"]["files"].items())
    inputs = read(result["manifest"])
    assert sha(result["manifest"]) == result["manifest_sha256"]
    archived_source = read(Path(inputs["baseline_source"]) / "source_snapshot.json")["files"]
    changed = [name for name, checksum in result["source"]["files"].items()
               if archived_source[name] != checksum]
    assert changed == (["bayesfilter/inference/hmc.py"] if result["strategy"] == "dynamic" else [])
    target = "beta_binomial" if path.name.startswith("beta-binomial") else "gaussian"
    cells = {(row["L"], row["count"]): row for row in inputs["cells"] if row["target"] == target}
    def archived_fingerprints(value):
        if "tensor" in value:
            raw = base64.b64decode(value["tensor"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == value["sha256"]
            return {k: value[k] for k in ("dtype", "shape", "sha256")}
        return {k: archived_fingerprints(v) for k, v in value.items()}
    assert result["status"] == "passed" and len(result["cells"]) == 6
    audit_runtime(result["runtime"], "gpu")
    first, warmed, rows = 0., 0., []
    for cell in result["cells"]:
        original = cells[(cell["L"], cell["count"])]
        assert sha(original["chunk"]) == original["chunk_sha256"]
        chunk = read(original["chunk"])
        expected = {k: archived_fingerprints(chunk[k]) for k in ("samples", "trace")}
        assert len(cell["calls"]) == 3
        for call in cell["calls"]:
            assert call["exact_comparison_passed"]
            assert call["tensor_fingerprints"] == expected
            assert call["tracing_counts"] == [1] * 4
        times = [call["seconds"] for call in cell["calls"]]
        first += times[0]
        warmed += sum(times[1:])
        rows.append({k: cell[k] for k in ("L", "count", "epsilon", "reused", "construction_seconds")} | {"call_seconds": times})
    return {"path": str(path), "sha256": sha(path / "result.json"), "runner_groups": result["runner_count"],
        "all_archived_tensor_fingerprints_independently_checked": True,
        "source_difference_paths": changed,
        "first_call_seconds_including_reuse": first, "two_warmed_repeats_seconds": warmed,
        "cells": rows, "resources": result["resources_after"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = read(args.source_manifest)["files"]
    result = {"phase": "M30", "targets": [], "sampler_executed": False}
    for target in ("gaussian", "beta-binomial"):
        a = audit_arm(root / (target + "-full-static-gpu-r1"), "static", files)
        b = audit_arm(root / (target + "-full-dynamic-gpu-r1"), "dynamic", files)
        assert a["design_identity"] == b["design_identity"]
        assert a["design_file_sha256"] == b["design_file_sha256"]
        assert a["source_identity"] == b["source_identity"]
        pair = full_pair(Path(a["fit"]), Path(b["fit"]))
        result["targets"].append({"target": target, "static": a, "dynamic": b, "parity": pair,
            "replays": {"static": replay_summary(root / (target + "-static-gpu-r" + ("2" if target == "gaussian" else "1"))),
                        "dynamic": replay_summary(root / (target + "-dynamic-gpu-r1"))}})
    result["passed"] = True
    result["normalization"] = ["declared clocks", "output root", "reuse option and dependent binding hash",
        "ensemble invocation count", "first cache-call flag", "checked dynamic-L metadata",
        "result/evidence integrity hashes independently verified then replaced by normalized digest"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "targets": [r["parity"] for r in result["targets"]]}))


if __name__ == "__main__":
    main()
