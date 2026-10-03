"""Preserve compact M29 evidence; read existing runs without executing a sampler."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


audit_path = ROOT / "terminal-audit-r1.json"
audit = read(audit_path)
cost_path = ROOT / "cost-attribution-r2.json"
cost = read(cost_path)
ledger = read(ROOT / "reconciliation-terminal.json")
assert audit["passed"] and ledger["all_phase_workers_terminal"]
assert cost["audit_sha256"] == sha(audit_path)
source = read(ROOT / "source-r3-manifest.json")
for name in read(ROOT / "owned-source-paths.json"):
    if name.startswith("bayesfilter/"):
        assert sha(REPO / name) == source["files"][name], name

rows = []
for pair, attribution in zip(audit["pairs"], cost["targets"], strict=True):
    assert pair["target"] == attribution["target"]
    parity = pair["parity"]
    row = {key: pair[key] for key in (
        "target", "design_id", "descriptive_profile_overhead_seconds",
        "confirmation_384_fits_gpu_hours_at_observed_unprofiled_price")}
    row["parity"] = {key: parity[key] for key in (
        "exact_tensor_and_numerical_parity", "candidate_count", "verified_count",
        "receipts", "tensor_count", "tuning_completion",
        "all_observations_and_receipts_equal",
        "all_posterior_decisions_and_independent_assessments_equal")}
    row["selected_posteriors"] = [{key: member[key] for key in
        ("candidate_id", "L", "epsilon", "passed", "hard_vetoes",
         "warmup_cap", "retained_cap", "warmup", "retained", "fixed_status")}
        for member in parity["selected_posteriors"]]
    row["arms"] = {}
    for mode in ("off", "on"):
        arm = pair[mode]
        row["arms"][mode] = {key: arm[key] for key in
            ("normal_exit", "elapsed_seconds", "setup_seconds", "runtime",
             "pipeline_timing", "member_timing")}
        row["arms"][mode]["resources_after"] = {key: arm["resources"]["after"][key]
            for key in ("registered_tf_functions", "max_rss_kib", "gpu_allocator_bytes")}
    profile = pair["on"]["profile"]
    row["profile"] = {key: profile[key] for key in
        ("path", "sha256", "python_calls", "profile_seconds",
         "required_functions_found", "stages_and_framework_calls", "interpretation")}
    row["saved_chunk_costs"] = {key: attribution[key] for key in
        ("unique_cache_keys", "calls", "first_call_seconds", "warmed_call_seconds", "interpretation")}
    rows.append(row)
write("terminal-audit-summary.json", {
    "phase": "M29", "passed": True, "sampler_executed_by_audit": False,
    "full_audit": str(audit_path), "full_audit_sha256": sha(audit_path),
    "cost_attribution": str(cost_path), "cost_attribution_sha256": sha(cost_path),
    "source_identity": source["identity"], "base_commit": source["git_commit"],
    "independent_fits": 2, "pairs": rows,
    "tests": {"focused_regressions": 43, "documentation_contract": 15},
    "charged_seconds": ledger["charged_seconds"],
    "remaining_budget_seconds": ledger["remaining_budget_seconds"],
    "interpretation": "Engineering replay and host cost evidence. No speed ranking, posterior coverage, global exploration, learned-map or default promotion.",
})

attempts = []
for receipt in sorted(ROOT.glob("*/execution.json")):
    record = read(receipt)
    path = receipt.parent / "manifest.json"
    manifest = read(path)
    attempts.append({"name": receipt.parent.name, "receipt": str(receipt),
        "receipt_sha256": sha(receipt), "manifest": str(path),
        "manifest_sha256": sha(path), "record": record,
        "command": manifest["command"], "environment": manifest["environment"],
        "cwd": manifest["cwd"], "git_commit": manifest["git_commit"],
        "started_utc": manifest["started_utc"], "device_scope": manifest["device_scope"],
        "gpu_intentionally_hidden": manifest["gpu_intentionally_hidden"],
        "gpu_preflight": manifest.get("gpu_preflight"),
        "worker_log_sha256": sha(receipt.parent / "worker.log")})
assert len(attempts) == ledger["attempt_count"]
write("attempt-manifest-summary.json", {"phase": "M29", "attempts": attempts,
    "full_runtime_source_hashes": "source-r3-manifest.json",
    "plan_file": "docs/plans/bayesfilter-hmc-post-m28-next-phase-2026-09-23.md",
    "result_file": "docs/plans/bayesfilter-hmc-m29-profiling-result-2026-09-23.md"})

guide = Path("/tmp/hmc-m29-guide-build-r1")
log = (guide / "main.log").read_text()
assert not re.search(r"(?:undefined citations|undefined references|Citation .*undefined)", log)
assert "Output written on main.pdf (574 pages" in log
doc_names = ("docs/main.tex", "docs/chapters/ch21b_hmc_tuning_interfaces.tex",
    "docs/reference/hmc-tuning-interface.md", "docs/references.bib",
    "docs/generated/hmc_tuning_route_table.tex")
write("guide-validation.json", {
    "source_sha256": {name: sha(REPO / name) for name in doc_names},
    "pdf": str(guide / "main.pdf"), "pdf_sha256": sha(guide / "main.pdf"),
    "log_sha256": sha(guide / "main.log"), "pages": 574,
    "undefined_citations_or_references": False,
    "bibtex_completed": read(ROOT / "guide-bibtex-r2/execution.json")["exit_code"] == 0,
    "final_build_receipt": "guide-render-repair-r1/execution.json",
    "visual_inspection": {"pdf_page": 432, "printed_page": 414,
        "image": str(guide / "page-432-final.png"),
        "image_sha256": sha(guide / "page-432-final.png"),
        "checked": "profiling paragraph readable; CLI command and flag separated; no clipping"},
})
print(json.dumps({"pairs": len(rows), "attempts": len(attempts),
                  "guide_pages": 574, "source_identity": source["identity"]}))
