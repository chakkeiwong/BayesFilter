"""Bounded bootstrap repair using the settled predecessor's remaining budget."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from bayesfilter.inference.q20_campaign_runtime import atomic_json, source_snapshot
from bayesfilter.inference.q20_production_config import digest
from bayesfilter.inference.q20_training_resume import checksum, read_training_checkpoint, _unchanged_program

PLAN = "docs/plans/bayesfilter-ssl-lstm-q20-bootstrap-repair-2026-09-18.md"
SETUP_SECONDS = 180.  # Predeclared repair setup/accounting allocation.
VERIFICATION_CAP = 900.
FUNCTION_CHANGES = {
    "bayesfilter/inference/hmc_preparation.py": {"prepare_operational_windowed_mass_handoff"},
    "bayesfilter/inference/hmc_kernel_tuning.py": {"prepare_operational_windowed_mass_handoff"},
    "bayesfilter/inference/hmc_candidate_set_execution.py": {"_source_closure"},
    "bayesfilter/inference/q20_production_hmc.py": {"tune_scope"},
    "bayesfilter/inference/q20_master_stages.py": {"price_preparation"},
    "bayesfilter/inference/q20_master_program.py": {"execute_master"},
    "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py": {"main"},
}
NEW_PATHS = {"bayesfilter/inference/hmc_bootstrap_initialization.py",
             "bayesfilter/inference/q20_master_repair.py"}


def check_repair_sources(previous, previous_root, current_root):
    previous_root, current_root = Path(previous_root), Path(current_root)
    if source_snapshot(previous_root) != previous:
        raise ValueError("preserved repair source does not match predecessor")
    current = source_snapshot(current_root)
    changes = sorted(p for p in previous.keys() | current.keys() if previous.get(p) != current.get(p))
    for path in changes:
        if path in NEW_PATHS and path not in previous and path in current:
            continue
        functions = FUNCTION_CHANGES.get(path)
        if (functions is None or path not in previous or path not in current or
                _unchanged_program(previous_root / path, functions) !=
                _unchanged_program(current_root / path, functions)):
            raise ValueError("unreviewed repair source change: " + path)
    return changes


def read_predecessor(root, config):
    from bayesfilter.inference.q20_master_refresh import PHASE_CAP
    root = Path(root).resolve()
    allowance = json.loads((root / "settled-allowance.json").read_text())
    path = root / "campaign.json"
    if (Path(allowance["source_ledger"]).resolve() != path or
            allowance["source_sha256"] != checksum(path) or not allowance["all_attempts_settled"]):
        raise ValueError("predecessor allowance is not settled against its ledger")
    state = json.loads(path.read_text())
    if state["config_hash"] != digest(config) or any(a["status"] == "running" for a in state["attempts"]):
        raise ValueError("predecessor config changed or an attempt remains running")
    remaining = max(0., state["campaign_limit"] - state["spent_seconds"])
    diagnostic = max(0., min(remaining, state["diagnostic_limit"] - state["diagnostic_spent_seconds"]))
    if (allowance["campaign_remaining_seconds"] != remaining or
            allowance["diagnostic_remaining_seconds"] != diagnostic):
        raise ValueError("predecessor allowance arithmetic differs")
    phase_spent = sum(a.get("elapsed_seconds", 0.) for a in state["attempts"]
                      if a["stage"].startswith("refresh-"))
    return state, allowance, PHASE_CAP - phase_spent


def _checked_stage(state, name):
    row = state["stages"][name]
    if checksum(row["result_path"]) != row["result_sha256"]:
        raise ValueError("predecessor stage result changed")
    for path, expected in row["artifact_hashes"].items():
        if checksum(path) != expected:
            raise ValueError("predecessor stage artifact changed: " + path)
    result = json.loads(Path(row["result_path"]).read_text())
    if not result["completed"]:
        raise ValueError("incomplete predecessor stage cannot receive credit")
    return result


def preparation_failure_classification(result):
    """Distinguish candidate failure, deadline and invalid shared execution."""
    if result.get("status") == "waiting_for_gpu":
        return "resource_unavailable"
    attempt = result.get("attempt", {})
    if (result.get("type") == "HMCPreparationBudgetExceeded" or
            result.get("status") == "phase_or_stage_budget_exhausted" or
            attempt.get("status") in ("timed_out", "interrupted", "interrupted_upper_charge")):
        return "budget_or_runtime_interruption"
    if result.get("type") != "HMCPreparationFailure":
        return "execution_failure"
    if result.get("message") in ("invalid_initial_target", "invalid_retained_target"):
        return "invalid_shared_target"
    if not attempt.get("directory"):
        return "missing_failure_evidence"
    progress = Path(attempt["directory"]) / "worker/data/preparation_progress.json"
    if not progress.exists():
        return "missing_failure_evidence"
    failure = json.loads(progress.read_text()).get("failure", {})

    def has_exception(value):
        if isinstance(value, dict):
            return any((key.endswith("error_type") and item is not None) or has_exception(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(has_exception(item) for item in value)
        return False

    return "execution_failure" if has_exception(failure) else "candidate_preparation_failure"


def prior_repair_costs(history, *, predecessor_sha256, config):
    """Sum settled worker costs across fresh-source infrastructure retries."""
    total, seen = 0., set()
    for index, row in enumerate(history):
        root = Path(row["campaign"]).resolve()
        if root in seen:
            raise ValueError("duplicate prior repair ledger")
        seen.add(root)
        ledger = root / "campaign.json"
        saved = json.loads(ledger.read_text())
        settled = json.loads((root / "settled-allowance.json").read_text())
        if (checksum(ledger) != row["sha256"] or settled["source_sha256"] != row["sha256"] or
                Path(settled["source_ledger"]).resolve() != ledger or not settled["all_attempts_settled"] or
                saved["config_hash"] != digest(config) or
                saved["repair_predecessor"]["sha256"] != predecessor_sha256 or
                saved["allowance"].get("repair_history", []) != history[:index]):
            raise ValueError("prior repair lineage or settlement changed")
        if any(a["status"] == "running" or not a["diagnostic"] for a in saved["attempts"]):
            raise ValueError("prior repair attempts are unsettled or outside diagnostic scope")
        cost = sum(a["elapsed_seconds"] for a in saved["attempts"])
        if (not math.isfinite(cost) or cost < 0 or cost != saved["spent_seconds"] or
                cost != saved["diagnostic_spent_seconds"] or
                settled["campaign_remaining_seconds"] != max(0., saved["campaign_limit"]-cost) or
                settled["diagnostic_remaining_seconds"] != max(0., min(saved["campaign_limit"], saved["diagnostic_limit"])-cost)):
            raise ValueError("prior repair cost arithmetic differs")
        total += cost
    return total


def repair_master(campaign, *, previous_campaign, previous_source_root, fixture=False):
    """Requalify changed sources, then repair preparation at both temperatures."""
    if previous_campaign is None or previous_source_root is None:
        raise ValueError("repair requires predecessor campaign and preserved sources")
    config = campaign.config
    old, settled, phase_remaining = read_predecessor(previous_campaign, config)
    changes = check_repair_sources(old["sources"], previous_source_root, campaign.repo)
    allowance = campaign.state["allowance"]
    verification = allowance["verification"]
    receipt_path = Path(verification["path"])
    if checksum(receipt_path) != verification["sha256"]:
        raise ValueError("repair verification receipt changed")
    verified = json.loads(receipt_path.read_text())
    seconds = float(verified["wall_seconds"])
    if (verified["status"] != "passed" or not math.isfinite(seconds) or
            not 0 < seconds <= VERIFICATION_CAP or verified["sources"] != campaign.state["sources"]):
        raise ValueError("repair verification is missing, stale or over budget")
    history = allowance.get("repair_history", [])
    prior_cost = prior_repair_costs(history, predecessor_sha256=settled["source_sha256"], config=config)
    prior_stage_costs = {}
    for row in history:
        previous_attempts = json.loads((Path(row["campaign"]) / "campaign.json").read_text())["attempts"]
        for attempt in previous_attempts:
            name = attempt["stage"]
            prior_stage_costs[name] = prior_stage_costs.get(name, 0.) + attempt["elapsed_seconds"]
    debit = seconds + SETUP_SECONDS + prior_cost
    if (allowance["predecessor_sha256"] != settled["source_sha256"] or
            allowance["campaign_remaining_seconds"] != settled["campaign_remaining_seconds"] - debit or
            allowance["diagnostic_remaining_seconds"] != settled["diagnostic_remaining_seconds"] - debit or
            allowance.get("unsettled_holds")):
        raise ValueError("repair must deduct verification/setup once from settled predecessor")
    phase_cap = phase_remaining - debit
    if phase_cap <= 0:
        raise ValueError("original refresh phase allowance exhausted")
    saved = old["training_resume"]
    if checksum(saved["checkpoint"]) != saved["sha256"]:
        raise ValueError("preserved training checkpoint changed")
    read_training_checkpoint(saved["checkpoint"], config, sources=old["sources"])
    training = _checked_stage(old, "refresh-price-training")
    downstream = _checked_stage(old, "refresh-price-downstream")
    prices = copy.deepcopy(downstream["result"])
    previous = str(Path(previous_campaign).resolve())
    lineage = {"campaign": previous, "sha256": settled["source_sha256"],
        "sources": str(Path(previous_source_root).resolve()), "changed_sources": changes,
        "original_phase_remaining_seconds": phase_remaining, "verification_seconds": seconds,
        "setup_seconds": SETUP_SECONDS, "phase_cap_seconds": phase_cap,
        "repair_history": history, "prior_repair_worker_seconds": prior_cost,
        "prior_stage_seconds": prior_stage_costs,
        "training_checkpoint": saved["checkpoint"], "training_sha256": saved["sha256"],
        "training_prices": training["result_path"], "downstream_prices": downstream["result_path"],
        "evidence_policy": "original sources retained; preparation repaired in fresh scope; no training admission"}
    if campaign.state.get("repair_predecessor", lineage) != lineage:
        raise ValueError("repair predecessor changed during resume")
    campaign.state["repair_predecessor"] = lineage
    campaign.save()
    qualification = None

    def remaining():
        return min(campaign.remaining(True), phase_cap - campaign.state["spent_seconds"])

    def finish(status, **details):
        campaign.state.update(status=status, details=details)
        campaign.save()
        result = {"status": status, "details": details, "predecessor": lineage,
            "remaining_phase_seconds": max(0., remaining()),
            "remaining_campaign_seconds": campaign.remaining(),
            "remaining_diagnostic_seconds": campaign.remaining(True),
            "production_qualified": False, "posterior_qualified": False}
        atomic_json(campaign.root / "result.json", result)
        atomic_json(campaign.root / "settled-allowance.json", {
            "schema": "bayesfilter.q20.recovered_allowance.v1", "source_ledger": str(campaign.path),
            "source_sha256": checksum(campaign.path), "campaign_remaining_seconds": campaign.remaining(),
            "diagnostic_remaining_seconds": campaign.remaining(True), "all_attempts_settled": True,
            "remaining_phase_seconds": max(0., remaining()), "master_status": status,
            "scope": "Diagnostics included in campaign; original refresh allocation not renewed."})
        return result

    def stage(name, request, cap):
        name = "repair-" + name
        spent = prior_stage_costs.get(name, 0.) + sum(
            a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == name)
        attempt_cap = min(cap-spent, remaining())
        request = {**request, "fixture": fixture, "plan_file": PLAN}
        if qualification:
            request["qualification_path"] = qualification
        # Persist the cooperative deadline before execution; resume uses exactly
        # the same request and shares the original per-stage and phase caps.
        requests = campaign.state.setdefault("repair_requests", {})
        if name in requests:
            if {k: v for k, v in requests[name].items() if k != "max_seconds"} != request:
                raise ValueError("repair stage request changed during resume")
            request = requests[name]
        else:
            if request["stage"] == "price-preparation":
                request["max_seconds"] = max(1., attempt_cap - 100.)
            requests[name] = request
            campaign.save()
        if name not in campaign.state["stages"]:
            cap = attempt_cap
            if cap <= 2*config["execution"]["termination_grace_seconds"]:
                return {"completed": False, "status": "phase_or_stage_budget_exhausted"}
        return campaign.numerical_stage(name, request, cap_seconds=cap, diagnostic=True, gpu="auto")

    if config["jit_compile"]:
        records = []
        for beta in config["training"]["betas"][1:]:
            result = stage(f"qualify-beta{beta:g}", {"stage": "qualify", "betas": [beta]}, 1200.)
            if not result.get("completed"):
                return finish("MASTER_REPAIR_PAUSED", stage=f"qualify-beta{beta:g}", result=result)
            receipt = json.loads(Path(result["result_path"]).read_text())
            if receipt.pop("checksum") != digest(receipt):
                raise ValueError("repair qualification checksum mismatch")
            records.append(receipt)
        combined = {**records[0], "betas": {}}
        for receipt in records:
            if any(receipt[k] != combined[k] for k in receipt if k != "betas"):
                raise ValueError("repair qualification scopes differ")
            combined["betas"].update(receipt["betas"])
        qualification = str(campaign.root / "qualification.json")
        atomic_json(qualification, {**combined, "checksum": digest(combined)})

    failures, prepared_rows = [], []
    for beta in config["training"]["betas"][1:]:
        result = stage(f"preparation-beta{beta:g}", {"stage": "price-preparation", "beta": beta}, 4000.)
        if result.get("completed"):
            prepared_rows.append(result["result"])
        else:
            classification = preparation_failure_classification(result)
            failures.append({"beta": beta, "classification": classification, "result": result})
            # Missing runtime output/timeout preserves evidence and pauses the
            # candidate. An explicit invalid shared target or execution failure
            # is a continuation veto for this repair scope.
            if classification not in ("budget_or_runtime_interruption", "candidate_preparation_failure"):
                break
    prices["hmc"].extend(prepared_rows)
    if not failures:
        prices["missing_cost_categories"] = [x for x in prices["missing_cost_categories"] if x != "classical_preparation"]
    prices["repair_predecessor"] = lineage
    prices["repaired_preparation_sources"] = campaign.state["sources"]
    prices["status"] = "historical_prices_plus_repaired_preparation"
    atomic_json(campaign.root / "remaining-prices.json", prices)
    return finish("MASTER_BOOTSTRAP_REPAIR_INCOMPLETE" if failures else "MASTER_BOOTSTRAP_REPAIR_PREPARATION_PASSED",
        preparation_failures=failures, prepared=prepared_rows,
        missing_cost_categories=prices["missing_cost_categories"],
        next_action="inspect initialization/bootstrap/mass evidence; price remaining categories before campaign admission")
