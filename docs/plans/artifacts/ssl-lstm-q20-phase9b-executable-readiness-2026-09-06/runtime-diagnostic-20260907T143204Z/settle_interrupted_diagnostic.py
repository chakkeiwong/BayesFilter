"""Settle and report this stopped diagnostic without running numerical work."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


start = read_json(OUTPUT / "run_start.json")
supervision = read_json(OUTPUT / "deadline-supervision-result.json")
if supervision["process_exited"] is not True:
    raise RuntimeError("worker exit has not been confirmed")
factor = read_json(OUTPUT / "factor/readiness.json")
for archive in factor["sample_archives"]:
    if sha256(ROOT / archive["path"]) != archive["sha256"]:
        raise RuntimeError("completed factor archive checksum mismatch")
measured = float(supervision["elapsed_seconds_at_exit_check"])
ledger = CampaignBudgetLedger(OUTPUT.parent / "campaign_budget_ledger.json")
before = ledger.read()
attempt = next(row for row in before["attempts"] if row["attempt_id"] == OUTPUT.name)
if attempt["status"] != "running" or attempt["consumed_seconds"] != 0.0:
    raise RuntimeError("attempt already settled; do not charge it twice")
write_new(OUTPUT / "campaign-ledger-before-settlement.json", before)
settlement = ledger.settle_arm(
    attempt_id=OUTPUT.name,
    arm="readiness_diagnostic",
    measured_seconds=measured,
    status="stopped_budget_infeasible",
    failure_class="complete_p1_schedule_exceeds_arm_cap",
    repair="external deadline guard attached; manual early stop after factor timing proved budget infeasible",
)
ledger.finish_attempt(
    attempt_id=OUTPUT.name,
    status="stopped_budget_infeasible",
    failure_class="complete_p1_schedule_exceeds_arm_cap",
)
after = ledger.read()
write_new(OUTPUT / "campaign-ledger-after-settlement.json", after)
factor_extrapolation = (
    factor["first_compiled_call_seconds"] + 5 * factor["steady_state_chunk_seconds"]
)
manifest = {
    "schema": "bayesfilter.phase9b_runtime_diagnostic_interrupted_closeout.v1",
    "status": "M4_P0_BUDGET_INFEASIBLE_P1_BLOCKED_P2_BLOCKED",
    "claim_boundary": "phase9b_m4_p0_executable_readiness_only",
    "written_at_utc": datetime.now(timezone.utc).isoformat(),
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-2026-09-07.md",
    "result_file": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-result-2026-09-07.md",
    "launch": start,
    "target_signature": start["target_signature"],
    "data_version": {"binding": "frozen synthetic q20 target signature", "target_signature": start["target_signature"]},
    "source_hashes_at_launch": start["source_hashes"],
    "git_at_launch": start["git"],
    "command": start["command"],
    "environment": start["runtime_provenance"],
    "gpu_selection": read_json(OUTPUT / "gpu-selection.json"),
    "headroom_after_factor": read_json(OUTPUT / "factor/gpu-headroom.json"),
    "cpu_status": "not_sampled_at_launch; no CPU benchmark claim",
    "factor": factor,
    "seed_namespaces": attempt["seed_namespace"],
    "strict_status": "started_but_interrupted; no complete readiness receipt",
    "timing": {
        "started_at_utc": start["started_at_utc"],
        "worker_exit_confirmed_at_utc": supervision["finished_at_utc"],
        "charged_worker_wall_seconds": measured,
        "measurement_basis": "supervisor monotonic exit check including at most about one second of polling delay",
        "ledger_finish_wall_field": "administrative settlement latency; not GPU worker runtime",
        "factor_first_call_seconds": factor["first_compiled_call_seconds"],
        "factor_steady_call_seconds": factor["steady_state_chunk_seconds"],
        "factor_six_chunk_extrapolation_excluding_setup_seconds": factor_extrapolation,
        "factor_arm_cap_seconds": 2600.0,
        "two_arm_forecast_status": "incomplete; strict runtime not measured",
        "interpretation": "single-run budget extrapolation, not a statistical bound or sampler ranking",
    },
    "termination": {
        "reason": "measured_factor_chunk_cost_exceeds_P1_arm_budget",
        "term_at_local": "2026-09-07T23:26:29+08:00",
        "kill_at_local": "2026-09-07T23:27:28+08:00",
        "exit_code": 137,
        "limitation": "Python signal handler did not finish during compiled work; worker failure receipt absent",
        "only_our_process_stopped": 203299,
        "deadline_supervision": supervision,
    },
    "budget": {
        "cap_seconds": after["total_budget_seconds"],
        "historical_spend_estimate_seconds": 1832.61,
        "attempt_settlement": settlement,
        "total_consumed_seconds": after["consumed_seconds"],
        "remaining_seconds": ledger.remaining_seconds(),
        "reserved_seconds": after["reserved_seconds"],
        "ledger_path": str(ledger.path.relative_to(ROOT)),
        "ledger_sha256": ledger.checksum(),
    },
    "evidence_limits": [
        "factor movement, finite states/target values, single trace, XLA IR, and archive checks only",
        "readiness arm does not preserve full target-status, log-acceptance, or energy-veto evidence",
        "worker memory-growth policy return value was not persisted before forced termination; launch environment and source requirement are preserved",
        "no complete strict-arm or two-arm timing forecast",
        "no P1 launchable closeout, posterior correctness, convergence, whitening, ranking, or default promotion",
    ],
    "artifacts": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in OUTPUT.rglob("*") if path.is_file()
    },
}
write_new(OUTPUT / "run_manifest.json", manifest)
print(json.dumps({"status": manifest["status"], "budget": manifest["budget"], "timing": manifest["timing"]}, indent=2))
