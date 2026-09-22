"""Terminal M18 integrity, regression reconciliation and once-only accounting."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = ROOT / "source-r2"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT.parent))
from audit_phase import PhaseAudit, read, sha


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    started = time.monotonic()
    audit = PhaseAudit()
    audit.snapshot(ROOT / "source-r1")
    identity = audit.snapshot(SOURCE)
    for path in sorted(ROOT.glob("*-run.json")):
        audit.diagnostic(path)
    workers = sorted(ROOT.glob("hash-parity-*/diagnostic-run.json"))
    if len(workers) != 4:
        audit.outstanding.append("expected exactly four completed GPU parity workers")
    for path in workers:
        audit.diagnostic(path, artifacts=True)
        record = read(path)
        if record["returncode"] != 0:
            audit.invalid.append(str(path) + ": failed GPU parity")
        script = Path(record["command"][1])
        if sha(script) != record["script_sha256"]:
            audit.invalid.append(str(script) + ": worker script changed")
        if read(path.parent / "manifest.json")["source"]["identity"] != identity:
            audit.invalid.append(str(path) + ": worker source mismatch")
    parity = read(ROOT / "parity-comparison.json")
    for name, digest in parity["artifact_files"].items():
        if sha(ROOT / name) != digest:
            audit.invalid.append(name + ": parity input changed")
    if not all(case["exact_numerical_equality"] for case in parity["cases"]):
        audit.invalid.append("numerical parity failed")
    tests = read(ROOT / "final-test-inventory-r2.json")
    if tests["missing_selected"] or tests["unresolved_selected_failures"]:
        audit.invalid.append("missing or failing selected tests")
    for record in tests["reports"]:
        if sha(record["path"]) != record["sha256"]:
            audit.invalid.append(record["path"] + ": test XML changed")
    for name, digest in tests["test_source_sha256"].items():
        if sha(REPO / name) != digest:
            audit.invalid.append(name + ": tested source changed")
    source = read(ROOT / "current-source-reconciliation.json")
    current = {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "bayesfilter").rglob("*.py"))}
    if current != source["source_files"]:
        audit.invalid.append("current numerical source changed after parity")
    coverage = read(ROOT / "coverage-refresh-r1-run.json")
    for name, digest in coverage["reports"].items():
        if sha(name) != digest:
            audit.invalid.append(name + ": coverage input changed")
    for directory, script_name in (("guide-r1", "build_guide.py"),
                                   ("guide-r2", "build_guide_r2.py"),
                                   ("guide-r3", "build_guide_r3.py")):
        path = ROOT / directory / "build-manifest.json"
        audit.diagnostic(path)
        record = read(path)
        if sha(ROOT / directory / "main.pdf") != record["pdf_sha256"]:
            audit.invalid.append(directory + ": built PDF changed")
        if sha(ROOT / script_name) != record["script_sha256"]:
            audit.invalid.append(directory + ": build script changed")
    installed = read(ROOT / "guide-r3/install-review.json")
    if not installed["installed"] or sha(REPO / "docs/main.pdf") != installed["pdf_sha256"]:
        audit.invalid.append("official guide installation mismatch")
    for name, digest in read(ROOT / "guide-r3/build-manifest.json")["source_sha256"].items():
        if sha(REPO / name) != digest:
            audit.invalid.append(name + ": guide source differs from installed build")
    processes = read(ROOT / "terminal-process-check.json")
    audit.outstanding.extend(processes["matching_workers"])
    result = audit.finish(ROOT.parent / "m17-r1/reconciliation-terminal.json",
                         {"cpu_reference": 8000., "gpu": 2000.},
                         {"cpu_reference": 900., "gpu": 0.})
    result.update(schema="bayesfilter.hmc_m18_reconciliation.v1",
        artifact_sha256={name: sha(ROOT / name) for name in (
            "parity-comparison.json", "current-source-reconciliation.json",
            "final-test-inventory-r2.json", "guide-r3/install-review.json",
            "terminal-process-check.json")},
        parity={key: parity[key] for key in ("cases", "full_evidence_hashes_checked",
            "full_chunk_hashes_checked", "inline_tensor_hashes_checked")},
        tests={key: tests[key] for key in ("selected_count", "selected_passed",
            "selected_skipped", "missing_selected", "unresolved_selected_failures", "exclusions")},
        protected_pdf_sha256=installed["protected_baseline_sha256"],
        official_pdf_sha256=installed["pdf_sha256"],
        accounting_note="Each root run record, GPU worker record and three book builds charged once. 900 CPU seconds conservatively covers unmetered inspection, retrieval, rendering, reconciliation and local tooling. The lost timeout timer is charged at 125 seconds, explicitly marked as a conservative allowance.",
        manifest={"command": sys.argv, "environment": sys.executable, "gpu_intentionally_hidden": True,
            "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started},
        interpretation="Bounded engineering completion with scoped parity and regression evidence; difficult geometry, stopping calibration, defect sensitivity, learned training and exact MacroFinance evidence remain separate scientific gaps.")
    with (ROOT / "reconciliation-terminal.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("terminal", "budget_seconds", "invalid_artifacts",
        "outstanding_workers", "numerical_receipts_checked", "tensor_checksums_checked")}, indent=2))
    return int(bool(not result["terminal"] or result["invalid_artifacts"] or result["budget_seconds"]["exceeded"]))


if __name__ == "__main__":
    raise SystemExit(main())
