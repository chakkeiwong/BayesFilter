"""Post-run diagnostic summary; no numerical implementation or admission path."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {2621: 1, 2622: 58, 2623: 58, 2624: 17, 2625: 17, 2626: 102}
runs = []
source = None
for number, count in EXPECTED.items():
    directory = ROOT / f"run-{number:05}"
    manifest = directory / "run.json"
    row = json.loads(manifest.read_text())
    assert row["state"] == "passed", number
    assert row["test_evidence"] == {
        "passed": True, "tests": count, "failure": 0, "error": 0, "skipped": 0
    }, number
    if source is None:
        source = row["source_sha256"]
    assert row["source_sha256"] == source, number
    environment = row["environment"]
    if row["device"] == "CPU":
        assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
        gpu = None
    else:
        provenance = next(json.loads(line) for line in (directory / "process.log").read_text().splitlines()
                          if line.startswith('{"tensorflow_version"'))
        policy = provenance["gpu_memory_policy"]
        assert policy["all_physical_devices_memory_growth"] is True
        assert policy["configured_before_logical_device_initialization"] is True
        assert provenance["cuda_visible_devices"] == row["gpu_uuid"]
        gpu = {"uuid": row["gpu_uuid"], "memory_policy": policy,
               "tf32_enabled": provenance["tf32_enabled"], "trust_basis": provenance["trust_basis"]}
    runs.append({"run": number, "device": row["device"], "group": row["key"][1],
                 "tests": count, "elapsed_seconds": row["elapsed_seconds"],
                 "manifest": str(manifest.relative_to(ROOT)),
                 "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 "gpu": gpu})

diagnostics = {}
for number in (2622, 2623):
    directory = ROOT / f"run-{number:05}"
    record = json.loads((directory / "posterior-design-ill_conditioned.json").read_text())
    assert all(not any(flags.values()) for flags in record["post_rejection_execution"].values())
    for fit_record in [record["original"], *record["candidate"].values()]:
        assert fit_record["accepted"] is False
        assert fit_record["status"] == "curvature_fit_rejected"
        assert all(fit_record[key] is None for key in ("precision_z", "refined_covariance", "refined_factor"))
    poison = json.loads((directory / "posterior-rejected-precision-no-use.json").read_text())
    assert len(poison["runs"]) == 6
    assert all(item["calls"] == 21 and item["result"]["accepted"] is False for item in poison["runs"])
    diagnostics[str(number)] = record["diagnostic_precision_difference"]

all_runs = [json.loads(path.read_text()) for path in sorted(ROOT.glob("run-*/run.json"))
            if int(path.parent.name.split("-")[-1]) <= max(EXPECTED)]
supplemental = [json.loads(path.read_text()) for path in ROOT.glob("supplemental-compute-*.json")]
charged = {device: sum(row.get("elapsed_seconds", row["timeout_seconds"]) for row in all_runs
                       if row["device"] == device)
           + sum(row["charged_seconds"] for row in supplemental if row["device"] == device)
           for device in ("CPU", "GPU")}
caps = {"CPU": 32 * 3600, "GPU": 52 * 3600}
assert all(charged[device] < caps[device] for device in caps)
report = {
    "schema": "filter_repair_rejected_precision_result.v1",
    "criterion": "report_existing_fit_rejection_and_prevent_downstream_use",
    "owner_decision": "2026-09-22: report ill-conditioning error instead of demanding discarded precision equivalence",
    "baseline": "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf",
    "runtime_changed": False,
    "accepted_result_tolerances_changed": False,
    "diagnostic_only_field": "diagnostics.replicates[0].precision_z",
    "scope": "already-rejected D3 ill_conditioned fixture only; full records retained",
    "same_source_all_runs": True,
    "runs": runs,
    "posterior_checks": 150,
    "additional_focused_cpu_check": 1,
    "policy_checks": 102,
    "discarded_matrix_differences_explanatory_only": diagnostics,
    "superseded": "uninstalled 1e-8 allowance and unfinished high-precision GPU reference requirement",
    "historical_failures_preserved": [2378, 2415, 2456],
    "charged_through_run": max(EXPECTED),
    "charged_seconds": charged,
    "remaining_process_hours": {device: (caps[device] - charged[device]) / 3600 for device in caps},
    "public_posterior_wired": False,
    "campaign_complete": False,
    "main_merged": False,
}
with (ROOT / "posterior-rejection-qualification-02626.json").open("x") as output:
    json.dump(report, output, indent=2, allow_nan=False)
    output.write("\n")
print(json.dumps({key: report[key] for key in ("posterior_checks", "policy_checks", "charged_seconds", "remaining_process_hours")}, indent=2))
