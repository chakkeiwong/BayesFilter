"""Reconcile outer receipts once and preserve open scientific requirements."""
import datetime
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
root = repo / "docs/plans/artifacts/hmc-repair-master-2026-09-16"
progress = json.loads((root / "program-progress.json").read_text())
opening = json.loads((root / "m31-r1/reconciliation-terminal.json").read_text())
balance = dict(opening["remaining_budget_seconds"])
classified = {
    "M32/baseline-tests-r1": "snapshot omitted scripts; fixed in baseline-tests-r2",
    "M33/policy-and-parity-tests-r1": "negative per-chain lugsail fixture and real cached-policy bug; fixed and tested",
    "M33/policy-endpoint-regressions-r2": "old sibling-error regex; earlier identity rejection is correct",
    "M38/affected-tests-r1": "new test used wrong warmup report nesting; corrected assertions passed",
    "M38/guide-build-r1": "snapshot omitted five committed figures; restored and book rebuilt",
    "M38/optional-reference-status-r1": "optional ArviZ reference module skipped because matplotlib is absent",
}
ledgers = []
for number in range(32, 39):
    phase = f"M{number}"
    path = root / f"m{number}-r1"
    path.mkdir(exist_ok=True)
    before = dict(balance)
    receipts = []
    charged = {"cpu_reference": 60., "gpu": 0.}
    for p in sorted(path.glob("*/execution.json")):
        receipt = json.loads(p.read_text())
        classification = classified.get(phase + "/" + p.parent.name)
        assert receipt["exit_code"] == 0 or classification, str(p)
        for device, field in (("cpu_reference", "cpu_worker_seconds"), ("gpu", "gpu_worker_seconds")):
            charged[device] += receipt[field]
        receipts.append({"path": str(p.relative_to(repo)), "exit_code": receipt["exit_code"],
                         "elapsed_seconds": receipt["elapsed_seconds"], "classification": classification})
    assert charged["cpu_reference"] <= progress["phases"][phase]["cpu_limit"]
    assert charged["gpu"] <= progress["phases"][phase]["gpu_limit"]
    balance = {k: balance[k]-charged[k] for k in balance}
    assert min(balance.values()) >= 0
    ledger = {"phase": phase, "terminal": True, "opening_remaining_seconds": before,
        "charged_seconds": charged, "remaining_budget_seconds": balance,
        "bookkeeping_allowance_seconds": 60, "attempts": receipts,
        "outstanding_workers": [], "scientific_gaps_closed": False,
        "accounting": "outer wall counted once; mixed integration tests charged only to their receipt phase"}
    (path / "reconciliation-terminal.json").write_text(json.dumps(ledger, indent=2)+"\n")
    ledgers.append(ledger)
    progress["phases"][phase].update(cpu_worker_seconds=charged["cpu_reference"],
        gpu_worker_seconds=charged["gpu"], ledger_file=str((path / "reconciliation-terminal.json").relative_to(repo)))
statuses = {"M32": "bounded_execution_complete_gpu_parity_audited",
    "M33": "repair_complete_confirmation_underfunded",
    "M34": "activation_complete_confirmation_underfunded",
    "M35": "diagnostic_wiring_complete_global_exploration_open",
    "M36": "inventory_complete_awaiting_inputs",
    "M37": "cpu_gpu_composition_complete_training_quality_open",
    "M38": "terminal_integration_complete"}
for phase, status in statuses.items():
    progress["phases"][phase].update(status=status,
        result_file="docs/plans/bayesfilter-hmc-m31-m38-result-2026-09-23.md")
progress.update(active_phase=None, concurrent_phase=None, next_phase=None,
    status="bounded_M31_M38_program_complete_scientific_requirements_open",
    scientific_gaps_closed=False, remaining_budget_seconds=balance,
    current_ledger="docs/plans/artifacts/hmc-repair-master-2026-09-16/m38-r1/reconciliation-terminal.json",
    next_action="R2/R3 require adequately funded confirmation; R4/R7 require validated geometry; R5 exact inputs; no automatic M39",
    active_service=None, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
progress["remaining_gaps"] = ["R2_stopped_coverage_and_delivery", "R3_full_fit_null_and_defect_power",
    "R4_global_exploration", "R5_exact_consumer_inputs", "R7_target_specific_learned_map_quality",
    "optional_ArviZ_reference_environment", "legacy_module_maintainability_debt"]
progress["terminal_requirements"] = {"R1": "closed_for_declared_engineering_scope",
    "R2": "underfunded_confirmation_prior_failures_preserved", "R3": "underfunded_confirmation",
    "R4": "diagnostics_tested_exploration_open", "R5": "awaiting_inputs",
    "R6": "declared_parity_resource_checks_closed_legacy_debt_recorded",
    "R7": "composition_tested_quality_protocol_and_cost_unresolved",
    "R8": "book_reference_registry_and_status_aligned"}
progress["next_phase_design"] = None
progress["remaining_gap_agenda"] = "Terminal requirement table in bayesfilter-hmc-m31-m38-result-2026-09-23.md"
progress["active_source_scope"] = "M32/M34 GPU fits frozen at source-final-r2; final resume source check at r4; M37 and final book at r5. No historical relabeling."
(root / "program-progress.json").write_text(json.dumps(progress, indent=2)+"\n")
summary = {"opening_M31": opening["opening_remaining_seconds"], "remaining": balance,
    "charged_M31_M38": {k: opening["opening_remaining_seconds"][k]-balance[k] for k in balance},
    "receipts_M32_M38": sum(len(l["attempts"]) for l in ledgers),
    "phase_ledgers": [f"m{n}-r1/reconciliation-terminal.json" for n in range(31,39)]}
(root / "m38-r1/program-reconciliation.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary))
