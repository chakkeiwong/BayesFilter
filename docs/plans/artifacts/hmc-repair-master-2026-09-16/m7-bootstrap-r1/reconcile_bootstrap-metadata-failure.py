"""Reconcile bounded engineering checks and extend the existing campaign ledger."""
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def definitions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


previous_path = ROOT.parent / "m7-r1/reconciliation-r2.json"
previous = load(previous_path)
baseline = load(ROOT / "baseline.json")
for row in baseline["files"]:
    assert sha(ROOT / "baseline" / row["path"]) == row["sha256"], row["path"]
assert load(ROOT / "before-r2.json") == load(ROOT / "after.json")
extraction = load(ROOT / "extraction.json")
old = definitions(ROOT / "baseline/bayesfilter/inference/hmc_kernel_tuning.py")
moved = {row["name"] for row in extraction["definitions"]}
for row in extraction["definitions"]:
    name = row["name"]
    current = definitions(REPO / "bayesfilter/inference" / row["module"])[name]
    assert hashlib.sha256(current.encode()).hexdigest() == row["after_sha256"]
    expected = old[name]
    if name == "run_hmc_bootstrap_screen":
        expected = expected.replace("value='hmc_kernel_tuning.py'", "value='hmc_bootstrap.py'")
    assert current == expected, name
current = definitions(REPO / "bayesfilter/inference/hmc_kernel_tuning.py")
assert set(old) - set(current) == moved
assert [name for name in current if current[name] != old[name]] == ["_g2_source_file_for_site"]

# The renamed test now explicitly exercises the historical wrapper it mocks.
renamed = {
    "test_p4_phase7_loop_requires_typed_registry_and_non_p4_does_not_create_one":
    "test_historical_phase7_loop_requires_typed_registry_and_non_p4_does_not_create_one"
}
cases = {}
for name in ("baseline-tests.xml", "affected-tests.xml", "affected-retry.xml"):
    for case in ET.parse(ROOT / name).findall(".//testcase"):
        nodeid = case.get("classname") + "::" + renamed.get(case.get("name"), case.get("name"))
        status = "skipped" if case.find("skipped") is not None else (
            "failed" if case.find("failure") is not None or case.find("error") is not None else "passed")
        cases[nodeid] = {"status": status, "latest_batch": name}
assert sum(row["status"] == "passed" for row in cases.values()) == 332
assert sum(row["status"] == "skipped" for row in cases.values()) == 1
assert not any(row["status"] == "failed" for row in cases.values())

attempts = []
for path in sorted(ROOT.glob("*-run.json")):
    row = load(path)
    attempts.append({**row, "manifest_path": str(path.relative_to(REPO)),
                     "manifest_sha256": sha(path), "log_sha256": sha(Path(row["log"]))})
assert len(attempts) == 7
measured = sum(row["elapsed_seconds"] for row in attempts)
assert measured <= 600
overhead = 120.
prior_budget = previous["budget_seconds"]
budget = {
    "cpu_measured": measured, "cpu_overhead": overhead, "cpu_charge": measured + overhead,
    "gpu_charge": 0.,
    "total_cpu_charged": prior_budget["total_cpu_charged"] + measured + overhead,
    "total_gpu_charged": prior_budget["total_gpu_charged"],
    "remaining_cpu": prior_budget["remaining_cpu"] - measured - overhead,
    "remaining_gpu": prior_budget["remaining_gpu"],
    "remaining_m7_cpu_allocation": prior_budget["remaining_m7_cpu_allocation"] - measured - overhead,
    "remaining_m7_gpu_allocation": prior_budget["remaining_m7_gpu_allocation"],
    "unfinished_launched_reservations": 0.,
}
installation = load(ROOT / "official-guide-install.json")
assert sha(REPO / installation["official_path"]) == installation["installed_sha256"]
guide = load(ROOT / "guide-r1/build-manifest.json")
for path, expected in guide["source_sha256"].items():
    assert sha(REPO / path) == expected

paths = ["bayesfilter/inference/" + name for name in (
    "__init__.py", "hmc_bootstrap.py", "hmc_preparation_common.py", "hmc_kernel_tuning.py",
    "hmc_geometry.py", "hmc_warmup.py", "hmc_preparation.py", "hmc_candidate_set_execution.py")]
paths += ["tests/" + name for name in ("test_hmc_bootstrap_extraction.py",
    "test_hmc_kernel_tuning_bootstrap.py", "test_hmc_kernel_tuning_windowed_mass.py",
    "test_hmc_kernel_tuning_p4_registry_public_chain.py", "test_hmc_master_repair.py", "test_hmc_warmup.py")]
paths += ["docs/reference/hmc-tuning-interface.md", "docs/chapters/ch21b_hmc_tuning_interfaces.tex",
          "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
          "docs/plans/bayesfilter-hmc-repair-m7-result-2026-09-17.md"]
report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "question": "Bootstrap implementation ownership with unchanged numerical preparation",
    "plan_file": paths[-2], "result_file": paths[-1], "command": [sys.executable, str(Path(__file__))],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    "environment": {"python": sys.executable, "tensorflow": version("tensorflow"),
                    "tensorflow_probability": version("tensorflow-probability"),
                    "gpu_status": "intentionally hidden in all numerical test workers",
                    "current_process_framework_imported": False},
    "seeds": "Recorded in capture_bootstrap.py and preserved test fixtures; unchanged by extraction",
    "data": "Synthetic diagnostic/test fixtures only",
    "previous_reconciliation": {"path": str(previous_path.relative_to(REPO)), "sha256": sha(previous_path)},
    "baseline_verified": baseline, "source_sha256": {path: sha(REPO / path) for path in paths},
    "engineering": {"exact_comparison_cases": 15, "moved_definitions": 35,
                    "identical_definition_asts": 34, "physical_owner_relocations": 1,
                    "unchanged_remaining_definitions": len(current) - 1,
                    "distinct_passed": 332, "distinct_skipped": 1, "superseded_test_names": renamed},
    "test_cases": cases, "attempts": attempts, "budget_seconds": budget,
    "guide_installation": installation,
    "scientific_interpretation": "Engineering parity only; earlier frozen GPU findings retain their source identity",
}
with (ROOT / "reconciliation.json").open("x") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")
print(json.dumps({"engineering": report["engineering"], "budget_seconds": budget}, indent=2))
