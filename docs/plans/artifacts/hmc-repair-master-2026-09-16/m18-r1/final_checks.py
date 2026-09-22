"""Offline M18 parity, source and named-test reconciliation; no sampler work."""
from collections import Counter
from datetime import datetime, timezone
import base64
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = ROOT / "source-r2"
sys.path.insert(0, str(SOURCE))
from bayesfilter.inference.hmc_candidate_set_tuning import _sha256


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (ROOT / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checksum_record(record):
    body = dict(record)
    digest = body.pop("content_hash")
    assert _sha256(body) == digest, "JSON content hash mismatch"


def check_inline_tensors(node):
    count = 0
    if isinstance(node, dict):
        if isinstance(node.get("tensor"), str):
            raw = base64.b64decode(node["tensor"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == node["sha256"]
            count += 1
        else:
            count += sum(check_inline_tensors(value) for value in node.values())
    elif isinstance(node, list):
        count += sum(check_inline_tensors(value) for value in node)
    return count


def parity():
    cases, evidence_count, chunk_count, tensor_count = [], 0, 0, 0
    files = {}
    for case in ("gaussian", "beta_binomial"):
        projections, archive_projections = [], []
        for arm in ("generic", "native"):
            directory = ROOT / f"hash-parity-{case}-{arm}-r1"
            run = read(directory / "diagnostic-run.json")
            assert run["returncode"] == 0
            checkpoint = read(directory / "tuning/tuning_checkpoint.json")
            checksum_record(checkpoint)
            assert not checkpoint["partial_chunks"]
            spec = read(directory / "tuning/execution_spec.json")
            assert _sha256(spec["execution"]) == spec["binding_hash"] == checkpoint["binding_hash"]
            evidence = []
            for digest in checkpoint["numerical_evidence_hashes"]:
                row = read(directory / f"tuning/numerical_evidence/{digest}.json")
                assert _sha256(row) == digest
                assert row["binding_hash"] == spec["binding_hash"]
                evidence.append(row)
                evidence_count += 1
                tensor_count += check_inline_tensors(row)
            for path in (directory / "tuning/numerical_chunks").glob("*.json"):
                row = read(path)
                assert _sha256(row) == path.stem
                tensor_count += check_inline_tensors(row)
                chunk_count += 1
            evidence.sort(key=lambda row: row["work"]["ordinal"])
            result = read(directory / "tuning/candidate_set_result.json")
            assert result["completion_status"] == "complete"
            assert result["verified_candidate_ids"]
            projection = read(directory / "numerical.json")
            reconstructed = [{key: row[key] for key in
                ("candidate", "work", "initial_state", "seed", "samples", "trace", "analysis")}
                for row in evidence]
            assert projection["evidence"] == reconstructed
            for key, source_key in (("candidates", "candidates"), ("scope", "scope"),
                                    ("states", "candidate_states"), ("verified", "verified_candidate_ids")):
                assert projection[key] == result[source_key]
            assessment = read(directory / "assessment.json")
            assert _sha256(projection) == assessment["numerical_hash"]
            assert assessment["selected"] == sorted(result["verified_candidate_ids"])[:1]
            archives = {}
            for name in ("discarded", "retained"):
                archive = read(directory / name / "retained_archive.json")
                checksum_record(archive)
                tensor_count += check_inline_tensors(archive)
                archives[name] = {key: archive[key] for key in (
                    "candidate", "binding_hash", "num_results", "seed", "seed_history",
                    "initial_active_state", "final_active_state", "active_samples", "position_samples",
                    "trace", "health_failures", "acceptance_reporting_only", "rhat_reporting_only",
                    "tuning_draws_included", "warmup_draws_included")}
            assert archives["retained"]["initial_active_state"] == archives["discarded"]["final_active_state"]
            member = read(directory / "member.json")
            checksum_record(member)
            bundle_path = directory / member["evidence_bundle"]["path"]
            bundle = read(bundle_path)
            checksum_record(bundle)
            assert bundle["content_hash"] == member["evidence_bundle"]["content_hash"]
            assert member["candidate_id"] == assessment["selected"][0]
            tensor_count += check_inline_tensors(member)
            projections.append(projection)
            archive_projections.append(archives)
            for path in directory.rglob("*.json"):
                files[str(path.relative_to(ROOT))] = sha(path)
        assert projections[0] == projections[1], case + ": numerical projection differs"
        assert archive_projections[0] == archive_projections[1], case + ": retained traces differ"
        cases.append({"case": case, "exact_numerical_equality": True,
            "candidate_count": len(projections[0]["candidates"]),
            "verified_count": len(projections[0]["verified"]),
            "work_items_per_arm": len(projections[0]["evidence"]),
            "numerical_hash": _sha256(projections[0]),
            "archive_projection_hash": _sha256(archive_projections[0]),
            "discarded_per_chain": len(projections[0]["discarded_draws"]),
            "retained_per_chain": len(projections[0]["retained_draws"])})
    record = {"schema": "bayesfilter.hmc_m18_parity.v1", "cases": cases,
        "full_evidence_hashes_checked": evidence_count, "full_chunk_hashes_checked": chunk_count,
        "inline_tensor_hashes_checked": tensor_count, "artifact_files": files,
        "exclusions": "Elapsed timing/device runtime, paths and receipt/member hashes containing those fields. Numerical starts, seeds, order, settings, states, draws, traces, analyses and membership are compared exactly.",
        "interpretation": "Two-model same-source intervention parity; no posterior accuracy, speed ranking or cross-model validation claim."}
    write("parity-comparison.json", record)
    return record


def source_reconciliation():
    old_root = ROOT.parent / "m16-r1/source-r1"
    old = read(old_root / "source_snapshot.json")
    new = read(SOURCE / "source_snapshot.json")
    current = {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "bayesfilter").rglob("*.py"))}
    assert current == new["source_files"]
    assert _sha256(current) == new["source_identity"]
    changes = sorted(k for k in old["source_files"].keys() | current.keys()
                     if old["source_files"].get(k) != current.get(k))
    assert changes == ["bayesfilter/inference/__init__.py",
                       "bayesfilter/inference/hmc_candidate_set_checkpoint.py"]
    diffs = {name: "".join(difflib.unified_diff((old_root / name).read_text().splitlines(True),
                     (SOURCE / name).read_text().splitlines(True), fromfile="M16/" + name,
                     tofile="M18/" + name)) for name in changes}
    record = {"schema": "bayesfilter.hmc_m18_source_reconciliation.v1",
        "historical_source": old["source_identity"], "current_source": new["source_identity"],
        "current_matches_frozen_source": True, "files_checked": len(current),
        "changed_files": changes, "diffs": diffs, "source_files": current,
        "interpretation": "M16/M17 preserve their original whole-package source identity. Current-source GPU parity is limited to Gaussian and beta-binomial with supplied identity geometry. Shared numerical preparation, target, transition and posterior modules are byte-identical to M16; exact two-model parity covers the changed checkpoint hash path. The other change routes the same uncertainty-admission callable lazily. Historical evidence is not relabeled current-source."}
    write("current-source-reconciliation.json", record)
    return record


def tests():
    previous = read(ROOT / "final-test-inventory.json")
    selected = {line.strip() for line in (ROOT / "final-test-collection.log").read_text().splitlines()
                if line.startswith("tests/") and "::" in line}
    latest, reports = {}, []
    for path in sorted(ROOT.glob("*.xml"), key=lambda p: p.stat().st_mtime_ns):
        reports.append({"path": str(path), "sha256": sha(path), "mtime_ns": path.stat().st_mtime_ns})
        for case in ET.parse(path).getroot().iter("testcase"):
            if not case.attrib.get("classname"):
                continue
            node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
            bad = [child for child in case if child.tag in {"failure", "error"}]
            skipped = [child for child in case if child.tag == "skipped"]
            latest[node] = {"outcome": "failed" if bad else "skipped" if skipped else "passed",
                "report": str(path), "reason": (bad or skipped)[0].attrib.get("message", "") if bad or skipped else ""}
    missing = sorted(selected - latest.keys())
    failed = {key: latest[key] for key in selected & latest.keys() if latest[key]["outcome"] == "failed"}
    counts = Counter(latest[key]["outcome"] for key in selected & latest.keys())
    assert not missing and not failed, (missing, failed)
    assert counts["passed"] + counts["skipped"] == len(selected) == previous["selected_count"]
    record = {"schema": "bayesfilter.hmc_m18_test_inventory.v2", "selected_count": len(selected),
        "selected_passed": counts["passed"], "selected_skipped": {key: latest[key] for key in selected if latest[key]["outcome"] == "skipped"},
        "missing_selected": missing, "unresolved_selected_failures": failed,
        "latest_selected_outcomes": {key: latest[key] for key in sorted(selected)}, "reports": reports,
        "test_source_sha256": {name: sha(REPO / name) for name in sorted({key.split("::")[0] for key in selected})},
        "exclusions": {key: value for key, value in read(ROOT / "affected-tests-r5-source.json").items()
                       if key in {"unavailable_historical_fixtures", "unavailable_individual_nodes", "extended_selection"}},
        "selection_provenance": str(ROOT / "affected-tests-r5-source.json"),
        "supersedes": str(ROOT / "final-test-inventory.json"),
        "rule": "Latest completed named report wins, including failure and skip. Overlapping old oracle failures were followed by an unambiguous terminal module rerun. Interrupted dots are not outcomes.",
        "interpretation": "Affected self-contained selection, with documented historical and extended-run exclusions; not a full-repository test claim."}
    write("final-test-inventory-r2.json", record)
    return record


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    start = time.monotonic()
    result = {"parity": parity(), "source": source_reconciliation(), "tests": tests()}
    write("final-checks-r1-run.json", {"command": sys.argv, "device": "cpu_reference",
        "elapsed_seconds": time.monotonic() - start, "returncode": 0,
        "gpu_intentionally_hidden": True, "environment": sys.executable,
        "script_sha256": sha(Path(__file__)), "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_file": "docs/plans/bayesfilter-hmc-repair-m18-design-2026-09-21.md"})
    print(json.dumps({"parity": result["parity"]["cases"], "source": result["source"]["changed_files"],
        "tests_passed": result["tests"]["selected_passed"],
        "tests_skipped": len(result["tests"]["selected_skipped"])}, indent=2))


if __name__ == "__main__":
    main()
