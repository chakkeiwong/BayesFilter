#!/usr/bin/env python3
"""Prepare the reviewed q20 recovery successor; no GPU or service launch."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path("/home/ubuntu/python/BayesFilter")
SOURCE = Path("/tmp/BayesFilter-q20-recovery-20260922-r1")
PREVIOUS_SOURCE = Path("/tmp/BayesFilter-q20-staged-budget-20260920-r2")
ROOT = REPO / "docs/plans/artifacts/q20-recovery-and-affordability-2026-09-22"
CONTROL = REPO / "docs/plans/artifacts/q20-master-operations-2026-09-21"
PARENT = CONTROL / "campaign-02"
PLAN = "docs/plans/bayesfilter-q20-recovery-and-affordability-repair-plan-2026-09-21.md"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    sys.path.insert(0, str(SOURCE))
    from bayesfilter.inference.q20_campaign_runtime import Campaign, atomic_json, source_snapshot
    from bayesfilter.inference.q20_production_config import digest
    from bayesfilter.inference.q20_training_resume import check_refresh_sources, import_training_checkpoint
    from bayesfilter.inference.q20_stage_budget import repair_remaining
    from types import SimpleNamespace
    state = read(PARENT / "campaign.json")
    if state["status"] != "ESTIMATION_BUDGET_PAUSED" or any(a["status"] == "running" for a in state["attempts"]):
        raise ValueError("predecessor must be settled and paused")
    current, changed = check_refresh_sources(state["sources"], PREVIOUS_SOURCE, SOURCE)
    checks = sorted(ROOT.glob("cpu-check-*.json"))
    if not checks or read(checks[-1])["returncode"] != 0:
        raise ValueError("focused engineering checks must pass before preparation")
    audit = read(ROOT / "saved-tuning-audit.json")
    if not audit["valid_for_fresh_trial"]:
        raise ValueError("saved failure requires an implementation repair before tuning")
    gpu_probe = read(ROOT / "gpu-readiness.json")
    if gpu_probe["exit_code"] != 0 or json.loads(gpu_probe["output"])["status"] != "PASSED":
        raise ValueError("trusted GPU readiness probe failed")
    probe_charge = math.ceil(gpu_probe["wall_time_seconds"])
    evidence = {"previous_sources": state["sources"], "current_sources": current,
        "changed_paths": changed, "source_root": str(SOURCE), "previous_root": str(PREVIOUS_SOURCE),
        "review": "Target, transformed Hamiltonian, integrator, learning updates and posterior tests unchanged. Host orchestration/pricing and an explicit public-tuner proposal added.",
        "source_diff": str(ROOT / "source.diff"), "source_diff_sha256": sha(ROOT / "source.diff"),
        "test_receipts": [str(p) for p in checks], "plan": PLAN}
    atomic_json(ROOT / "source-import-audit.json", evidence)
    old_training = read(state["stages"]["train-neutra"]["result_path"])["result"]["checkpoint"]
    campaign = Campaign(ROOT / "campaign-03", repo=SOURCE, config=state["config"], allowance=state["allowance"])
    if campaign.path.exists():
        raise FileExistsError("prepared campaign already exists; do not renew allocations")
    prices = {}
    old_plain = read(state["stages"]["price-neutra"]["result_path"])["result_path"]
    old_ensemble = PARENT / "attempts/00008-price-ensemble/worker/data/result.json"
    for method, path, quality in (("neutra", old_plain, "historical_no_contention_observed"),
                                  ("ensemble", old_ensemble, "resource_history_unknown")):
        result = read(path)
        if result["sources"] != state["sources"] or result["config_hash"] != state["config_hash"]:
            raise ValueError("historical price scope differs")
        destination = ROOT / f"pricing-import-{method}.json"
        atomic_json(destination, {"result_path": str(path), "result_sha256": sha(path),
            "source_root": str(PREVIOUS_SOURCE), "current_sources": current,
            "reviewed_changed_paths": changed, "timing_quality": quality,
            "role": "conditional_cost_hypothesis_only_preserving_original_sources"})
        prices[method] = str(destination)
    config = state["config"]
    per_length = audit["largest_observed_work_seconds_by_L"]
    factor = config["budget"]["forecast_safety_factor"]
    initial = sum(per_length[str(length)] for length in config["tuning"]["l_grid"])
    minimum = factor*2*initial
    base = config["tuning"]["startup"]+config["tuning"]["measurement"]
    rung_ratio = sum(config["tuning"]["startup"]+config["tuning"]["measurement"]*r
                     for r in config["tuning"]["evidence_rungs"])/base
    startup = max(read(state["stages"]["price-neutra"]["result_path"])["result"]["worker_initialization_seconds"],
                  config["execution"]["termination_grace_seconds"]*2)*factor
    minimum += startup
    maximum = (minimum-startup)*rung_ratio+startup
    old_mix = read(old_ensemble)
    # Old exchange first-time field was divided by transition count. Restore
    # whole-call units only for a labelled reservation hypothesis, not a new measurement.
    mix_quote = factor*(old_mix["exchange_first_seconds"]*config["execution"]["pricing_transitions"]+
        old_mix["exchange_steady_seconds"]*config["execution"]["pricing_transitions"])
    with campaign.locked():
        imported = import_training_checkpoint(old_training, config, campaign.root/"training-import",
            previous_root=PREVIOUS_SOURCE, current_root=SOURCE)
        prior_repair_spent = config["budget"]["repair_allocation_seconds"]-repair_remaining(SimpleNamespace(state=state, config=config))
        campaign.state.update(campaign_limit=state["campaign_limit"], diagnostic_limit=state["diagnostic_limit"],
            spent_seconds=state["spent_seconds"]+probe_charge, diagnostic_spent_seconds=state["diagnostic_spent_seconds"]+probe_charge,
            inherited_attempts=deepcopy(state["attempts"]), inherited_stages=deepcopy(state["stages"]),
            prior_infrastructure_repair_spent_seconds=prior_repair_spent,
            predecessor={"path": str(PARENT), "sha256": sha(PARENT/"campaign.json")}, training_resume=imported,
            status="CONTINUATION_PREPARED", stage_limits={},
            recovery_plan={"plan_file": PLAN, "phases": ["R1", "R2", "R3", "R4", "R5", "R6"],
                "pricing_imports": prices, "exact_selected_pricing": True,
                "pricing_block_reserves": {"ensemble": {"exchange": mix_quote}},
                "startup_reserve_seconds": startup,
                "plain_cohort": {"proposal": {"id": "interior-20260922", "epsilon": math.sqrt(.050625*.0759375)},
                    "minimum_seconds": minimum, "maximum_seconds": maximum,
                    "provenance": "factor two times observed same-L measurement costs; verification allowance, fresh rungs and startup included",
                    "maximum_role": "conditional_evidence_cap_clipped_to_funded_remainder_not_upfront_reservation"},
                "diagnostic_blocks": {
                    "qualify-beta1": {"minimum_seconds": 2*136.04413049702998, "maximum_seconds": 2*136.04413049702998},
                    "qualify-beta0.5": {"minimum_seconds": 2*139.54005348298233, "maximum_seconds": 2*139.54005348298233},
                    "price-neutra": {"minimum_seconds": startup, "maximum_seconds": 2*136.04413049702998},
                    "price-ensemble": {"minimum_seconds": startup+mix_quote, "maximum_seconds": startup+mix_quote},
                    "price-selected-neutra": {"minimum_seconds": startup, "maximum_seconds": 2*706.6891502690269},
                    "price-selected-ensemble": {"minimum_seconds": startup+mix_quote, "maximum_seconds": startup+mix_quote}},
                "diagnostic_provenance": "existing observed whole-worker qualifications and full-pricing bounds; no global process-count renewal; all diagnostic seconds remain cumulative"})
        campaign.state["attempts"].append({"stage": "gpu-readiness", "status": "completed", "diagnostic": True,
            "directory": str(ROOT), "elapsed_seconds": probe_charge, "cap_seconds": probe_charge,
            "accounting_basis": "rounded_up_measured_tool_wall", "measured_seconds": gpu_probe["wall_time_seconds"],
            "command": ["/home/ubuntu/.codex/bin/codex-gpu-probe", "--framework", "nvidia", "--gpu", "auto"]})
        campaign.save()
    driver = ROOT / "runtime-supervisor-r3.py"
    shutil.copy2(REPO / "scripts/q20_campaign_supervisor.py", driver)
    old_job = read(CONTROL / "job.json")
    job = {**old_job, "campaign_name": "campaign-03", "output_root": str(ROOT), "source_root": str(SOURCE),
        "driver_path": str(driver), "driver_sha256": sha(driver), "parent_campaign": str(PARENT),
        "plan_file": PLAN, "authority": "Owner requested refresh with additional repair phases, thorough review and execution on September 22. Existing target, total funds and deadline retained.",
        "validation": {"test_receipts": [str(p) for p in checks], "source_audit": str(ROOT/"source-import-audit.json")}}
    atomic_json(ROOT / "job.json", job)
    atomic_json(ROOT / "previous-control-job.json", old_job)
    # Activation remains a separate fixed ensure command after final review.
    atomic_json(CONTROL / "job.json", job)
    atomic_json(ROOT / "preparation.json", {"status": "READY_FOR_TRUSTED_ENSURE", "created_at": datetime.now(timezone.utc).isoformat(),
        "remaining_campaign_seconds": campaign.remaining(), "remaining_diagnostic_seconds": campaign.remaining(True),
        "source_import": imported, "plain_cohort": campaign.state["recovery_plan"]["plain_cohort"],
        "additional_campaign_seconds": 0., "deadline": job["deadline_local"]})
    print(json.dumps({"status": "READY_FOR_TRUSTED_ENSURE", "campaign": str(campaign.root),
        "remaining_campaign_hours": campaign.remaining()/3600, "remaining_diagnostic_minutes": campaign.remaining(True)/60,
        "plain_initial_reservation_hours": minimum/3600}, indent=2))


if __name__ == "__main__":
    prepare()
