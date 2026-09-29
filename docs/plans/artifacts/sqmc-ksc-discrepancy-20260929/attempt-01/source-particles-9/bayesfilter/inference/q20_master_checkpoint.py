"""Measured-cost continuation of the q20 preparation repair campaign."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from bayesfilter.inference.q20_campaign_runtime import atomic_json, source_snapshot
from bayesfilter.inference.q20_production_config import digest
from bayesfilter.inference.q20_master_repair import read_predecessor, preparation_failure_classification
from bayesfilter.inference.q20_training_resume import checksum

PLAN = "docs/plans/bayesfilter-ssl-lstm-q20-checkpointed-preparation-2026-09-19.md"
SETUP_SECONDS = 180.
CLEANUP_SECONDS = 100.  # Existing import/checkpoint/cleanup reserve.
ALLOWED_DEFINITIONS = {
    "bayesfilter/inference/hmc_kernel_tuning.py": {
        "HMCBootstrapScreenConfig", "run_hmc_bootstrap_screen", "_classify_bootstrap_screen",
        "_bootstrap_selected_kernel_payload", "_operational_windowed_mass_capture",
        "prepare_operational_windowed_mass_handoff"},
    "bayesfilter/inference/hmc_bootstrap.py": {
        "HMCBootstrapScreenConfig", "run_hmc_bootstrap_screen", "_classify_bootstrap_screen", "_bootstrap_selected_kernel_payload"},
    "bayesfilter/inference/hmc_mass_adaptation.py": {"_operational_windowed_mass_capture"},
    "bayesfilter/inference/hmc_preparation.py": {
        "HMCPreparationProgress", "prepare_operational_windowed_mass_handoff"},
    "bayesfilter/inference/hmc_candidate_set_execution.py": {"_source_closure"},
    "bayesfilter/inference/q20_master_stages.py": {"dispatch", "price_preparation"},
    "bayesfilter/inference/q20_production_hmc.py": {"tune_scope"},
    "bayesfilter/inference/q20_master_program.py": {"execute_master"},
    "bayesfilter/inference/q20_campaign_runtime.py": {"Campaign"},
    "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py": {"main"},
}
NEW_PATHS = {"bayesfilter/inference/hmc_bootstrap_checkpoint.py",
             "bayesfilter/inference/q20_master_checkpoint.py"}


def check_sources(old, previous_root, current_root):
    if source_snapshot(previous_root) != old:
        raise ValueError("preserved predecessor sources changed")
    current = source_snapshot(current_root)
    changed = sorted(p for p in old.keys() | current.keys() if old.get(p) != current.get(p))
    for path in changed:
        if path in NEW_PATHS and path not in old:
            continue
        allowed = ALLOWED_DEFINITIONS.get(path)
        if not allowed or path not in old or path not in current:
            raise ValueError("unreviewed preparation source change: " + path)
        trees = []
        for root in (previous_root, current_root):
            tree = ast.parse((Path(root) / path).read_text())
            tree.body = [node for node in tree.body if not isinstance(node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name not in allowed]
            trees.append(ast.dump(tree, include_attributes=False))
        if trees[0] != trees[1]:
            raise ValueError("unreviewed changes outside preparation definitions: " + path)
    return changed


def measured_costs(old, config):
    """Use completed calls, never interpret a killed round as a cheap transition."""
    rounds, initializers, qualifications, evidence = [], [], [], []
    for attempt in old["attempts"]:
        if attempt["stage"].startswith("repair-qualify-") and attempt["status"] == "completed":
            qualifications.append(attempt["elapsed_seconds"])
        path = Path(attempt["directory"]) / "worker/data/preparation_progress.json"
        if not path.exists():
            continue
        evidence.append({"path": str(path), "sha256": checksum(path)})
        for event in json.loads(path.read_text())["events"]:
            details = event["details"]
            if event["phase"] == "bootstrap_initialization.completed":
                initializers.append(event["elapsed_seconds"])
            if event["phase"] == "bootstrap.bootstrap_round_hmc_call_complete":
                count = (details["bootstrap_diagnostic_screen_num_results"] +
                         details["bootstrap_diagnostic_screen_num_burnin_steps"])
                rounds.append((details["elapsed_s"], count))
    if not all((rounds, initializers, qualifications)):
        raise ValueError("complete predecessor timing evidence is required")
    values = [*initializers, *qualifications, *(t for t, _ in rounds)]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("invalid predecessor timing evidence")
    factor = config["budget"]["forecast_safety_factor"]
    transition = max(t/n for t, n in rounds)
    return {"seconds_per_transition": transition, "max_initializer_seconds": max(initializers),
        "qualification_cap_seconds": factor*max(qualifications)+CLEANUP_SECONDS,
        "preparation_cap_seconds": factor*(max(initializers)+max(t for t, _ in rounds))+CLEANUP_SECONDS,
        "forecast_safety_factor": factor, "evidence": evidence,
        "counts_preserved": sorted({n for _, n in rounds}),
        "role": "scheduling_estimates_not_complete_preparation_prices"}


def checkpoint_master(campaign, *, previous_campaign, previous_source_root, fixture=False):
    if previous_campaign is None or previous_source_root is None:
        raise ValueError("checkpoint repair requires its settled predecessor and frozen sources")
    old, settled, _ = read_predecessor(previous_campaign, campaign.config)
    changed = check_sources(old["sources"], previous_source_root, campaign.repo)
    allowance = campaign.state["allowance"]
    receipt = allowance["verification"]
    if checksum(receipt["path"]) != receipt["sha256"]:
        raise ValueError("verification receipt changed")
    verified = json.loads(Path(receipt["path"]).read_text())
    seconds = verified["wall_seconds"]
    if (verified["status"] != "passed" or verified["sources"] != campaign.state["sources"] or
            not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("verification missing, stale or invalid")
    debit = seconds + SETUP_SECONDS
    if (allowance["predecessor_sha256"] != settled["source_sha256"] or
            allowance["campaign_remaining_seconds"] != settled["campaign_remaining_seconds"]-debit or
            allowance["diagnostic_remaining_seconds"] != settled["diagnostic_remaining_seconds"]-debit or
            allowance.get("unsettled_holds")):
        raise ValueError("continuation must debit verification/setup from the settled remainder")
    costs = measured_costs(old, campaign.config)
    lineage = {"campaign": str(Path(previous_campaign).resolve()), "sha256": settled["source_sha256"],
        "source_root": str(Path(previous_source_root).resolve()), "changed_sources": changed,
        "verification_seconds": seconds, "setup_seconds": SETUP_SECONDS, "costs": costs,
        "authorization": "owner requested refreshed master and continuation; new phase within remaining total budget"}
    if campaign.state.get("checkpoint_predecessor", lineage) != lineage:
        raise ValueError("checkpoint predecessor changed during continuation")
    campaign.state["checkpoint_predecessor"] = lineage
    campaign.save()
    atomic_json(campaign.root / "measured-cost-plan.json", lineage)

    def finish(status, **details):
        campaign.state.update(status=status, details=details)
        campaign.save()
        result = {"status": status, "details": details, "predecessor": lineage,
            "remaining_campaign_seconds": campaign.remaining(),
            "remaining_diagnostic_seconds": campaign.remaining(True),
            "production_qualified": False, "posterior_qualified": False}
        atomic_json(campaign.root / "result.json", result)
        atomic_json(campaign.root / "settled-allowance.json", {
            "schema": "bayesfilter.q20.recovered_allowance.v1", "source_ledger": str(campaign.path),
            "source_sha256": checksum(campaign.path), "campaign_remaining_seconds": campaign.remaining(),
            "diagnostic_remaining_seconds": campaign.remaining(True), "all_attempts_settled": True,
            "master_status": status, "scope": "New repair phase; diagnostics included in existing campaign."})
        return result

    qualification = None

    def stage(name, request, cap):
        request = {**request, "fixture": fixture, "plan_file": PLAN}
        if qualification is not None:
            request["qualification_path"] = qualification
        if name in campaign.state["stages"]:
            return campaign.numerical_stage(name, request, cap_seconds=cap, diagnostic=True, gpu="auto")
        # At most one infrastructure retry, sharing the measured stage envelope.
        # Two is an engineering attempt limit, not extra compute authority.
        for _ in range(2):
            attempts = [a for a in campaign.state["attempts"] if a["stage"] == name]
            spent = sum(a.get("elapsed_seconds", 0.) for a in attempts)
            remaining = min(cap-spent, campaign.remaining(True))
            minimum = (costs["forecast_safety_factor"]*(costs["max_initializer_seconds"] +
                       4*costs["seconds_per_transition"])+CLEANUP_SECONDS
                       if request["stage"] == "price-preparation" else cap)
            if len(attempts) >= 2 or remaining < minimum:
                return {"completed": False, "status": "phase_or_stage_budget_exhausted"}
            result = campaign.numerical_stage(name, request, cap_seconds=remaining, diagnostic=True, gpu="auto")
            if result.get("completed") or preparation_failure_classification(result) != "budget_or_runtime_interruption":
                return result
        return result

    if campaign.config["jit_compile"]:
        records = []
        for beta in campaign.config["training"]["betas"][1:]:
            result = stage(f"checkpoint-qualify-beta{beta:g}", {"stage": "qualify", "betas": [beta]},
                           costs["qualification_cap_seconds"])
            if not result.get("completed"):
                return finish("MASTER_CHECKPOINT_REPAIR_PAUSED", result=result)
            receipt = json.loads(Path(result["result_path"]).read_text())
            if receipt.pop("checksum") != digest(receipt):
                raise ValueError("qualification checksum mismatch")
            records.append(receipt)
        combined = {**records[0], "betas": {}}
        for receipt in records:
            if any(receipt[k] != combined[k] for k in receipt if k != "betas"):
                raise ValueError("qualification scopes differ")
            combined["betas"].update(receipt["betas"])
        qualification = str(campaign.root / "qualification.json")
        atomic_json(qualification, {**combined, "checksum": digest(combined)})

    preparations = []
    for beta in campaign.config["training"]["betas"][1:]:
        result = stage(f"checkpoint-preparation-beta{beta:g}", {
            "stage": "price-preparation", "beta": beta, "allow_deferred": True,
            "max_seconds": costs["preparation_cap_seconds"]-CLEANUP_SECONDS,
            "bootstrap_transition_seconds": costs["seconds_per_transition"]}, costs["preparation_cap_seconds"])
        preparations.append({"beta": beta, "result": result})
        if not result.get("completed"):
            classification = preparation_failure_classification(result)
            if classification not in ("budget_or_runtime_interruption", "candidate_preparation_failure"):
                return finish("MASTER_CHECKPOINT_REPAIR_STOPPED", preparations=preparations,
                              failure_classification=classification)
    complete = all(r["result"].get("result", {}).get("status") == "preparation_cost_measured" for r in preparations)
    startup = all(r["result"].get("result", {}).get("bootstrap_passed", complete) for r in preparations)
    return finish("MASTER_CHECKPOINT_PREPARATION_PASSED" if complete else
        "MASTER_STARTUP_PASSED_MASS_DEFERRED" if startup else "MASTER_CHECKPOINT_REPAIR_INCOMPLETE",
        preparations=preparations, bootstrap_passed=startup, mass_adaptation_completed=complete,
        next_action="price missing downstream categories" if complete else
        "use preserved bootstrap and measured mass forecast to plan runtime repair or affordable adaptation",
        historical_prices=str(Path(previous_campaign) / "remaining-prices.json"),
        missing_cost_categories=["classical_preparation", "classical_metric_specific_transition",
                                 "multi_chart_mixture_dispatch"] if not complete else
                                ["classical_metric_specific_transition", "multi_chart_mixture_dispatch"])
