"""Bounded current-source profiling within the settled q20 campaign allowance."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from bayesfilter.inference.q20_campaign_runtime import atomic_json, source_snapshot
from bayesfilter.inference.q20_master_repair import read_predecessor, _checked_stage
from bayesfilter.inference.q20_training_resume import checksum
from bayesfilter.inference.q20_production_config import digest

PLAN = "docs/plans/bayesfilter-ssl-lstm-q20-performance-continuation-2026-09-19.md"
PHASE_SECONDS = 3000.
SETUP_SECONDS = 180.
SCRIPT = "docs/benchmarks/diagnose_q20_factor_performance_2026_09_19.py"
NEW_PATHS = {"bayesfilter/inference/q20_master_profile.py", SCRIPT}
DEFINITIONS = {
    "bayesfilter/inference/q20_campaign_runtime.py": {"source_snapshot", "Campaign"},
    "bayesfilter/inference/q20_master_program.py": {"execute_master"},
    "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py": {"main"},
}


def qualification_profile_budget(old):
    return max(0., PHASE_SECONDS-old["spent_seconds"]-
               old["profile_predecessor"]["prior_profile_seconds"])


def qualify_profile_candidate(campaign, *, previous_campaign, previous_source_root):
    """Issue fresh optional-backend qualification only after the complete profile."""
    baseline_config = json.loads(json.dumps(campaign.config))
    if baseline_config["target"]["principal_sqrt_backend"] != "tensorflow_eigh_strict_factor_cached":
        raise ValueError("cache qualification requires the explicit candidate protocol")
    baseline_config["target"]["principal_sqrt_backend"] = "tensorflow_eigh_strict"
    old, settled, _ = read_predecessor(previous_campaign, baseline_config)
    if old["status"] != "MASTER_PROFILE_PARITY_PASSED":
        raise ValueError("candidate qualification requires a completed parity profile")
    profile = _checked_stage(old, "factor-profile")["result"]
    if not profile["candidate_parity_passed"]:
        raise ValueError("candidate parity did not pass")
    if source_snapshot(previous_source_root) != old["sources"]:
        raise ValueError("profile source changed")
    changed = sorted(p for p in old["sources"].keys() | campaign.state["sources"].keys()
                     if old["sources"].get(p) != campaign.state["sources"].get(p))
    allowed = {"bayesfilter/inference/q20_master_profile.py",
        "bayesfilter/inference/q20_production_config.py", "bayesfilter/inference/q20_master_program.py",
        "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py"}
    if set(changed)-allowed:
        raise ValueError("numerical dependencies changed after profile")
    allowance = campaign.state["allowance"]
    receipt = allowance["verification"]
    if checksum(receipt["path"]) != receipt["sha256"]:
        raise ValueError("qualification verification changed")
    verified = json.loads(Path(receipt["path"]).read_text())
    seconds = verified["wall_seconds"]
    if (verified["status"] != "passed" or verified["sources"] != campaign.state["sources"] or
            not math.isfinite(seconds) or seconds <= 0 or
            allowance["predecessor_sha256"] != settled["source_sha256"] or
            allowance["campaign_remaining_seconds"] != settled["campaign_remaining_seconds"]-seconds or
            allowance["diagnostic_remaining_seconds"] != settled["diagnostic_remaining_seconds"]-seconds):
        raise ValueError("qualification verification or budget lineage mismatch")
    checkpoint_path = Path(old["profile_predecessor"]["campaign"]) / "campaign.json"
    if checksum(checkpoint_path) != old["profile_predecessor"]["sha256"]:
        raise ValueError("qualification timing predecessor changed")
    checkpoint = json.loads(checkpoint_path.read_text())
    observed = [a["elapsed_seconds"] for a in checkpoint["attempts"]
                if a["stage"].startswith("checkpoint-qualify-") and a["status"] == "completed"]
    if len(observed) != len(campaign.config["training"]["betas"])-1:
        raise ValueError("missing qualification timing evidence")
    cap = max(observed)*campaign.config["budget"]["forecast_safety_factor"]+100.
    available = qualification_profile_budget(old)
    campaign.state["qualification_predecessor"] = {"path": str(Path(previous_campaign).resolve()),
        "sha256": settled["source_sha256"], "changed_sources": changed,
        "profile_phase_remaining_seconds": available, "per_beta_cap_seconds": cap,
        "verification_seconds": seconds}
    campaign.save()

    def finish(status, **details):
        campaign.state.update(status=status, details=details)
        campaign.save()
        result = {"status": status, "details": details, "production_qualified": False,
            "posterior_qualified": False, "mass_adaptation_completed": False,
            "remaining_campaign_seconds": campaign.remaining(),
            "remaining_diagnostic_seconds": campaign.remaining(True)}
        atomic_json(campaign.root / "result.json", result)
        atomic_json(campaign.root / "settled-allowance.json", {
            "schema": "bayesfilter.q20.recovered_allowance.v1", "source_ledger": str(campaign.path),
            "source_sha256": checksum(campaign.path), "campaign_remaining_seconds": campaign.remaining(),
            "diagnostic_remaining_seconds": campaign.remaining(True), "all_attempts_settled": True,
            "master_status": status, "scope": "Optional cache qualification within original profile allocation."})
        return result

    def stage_cap(beta):
        return max(0., cap-sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"]
                               if a["stage"] == f"cache-qualify-beta{beta:g}"))

    required = sum(stage_cap(beta) for beta in campaign.config["training"]["betas"][1:]
                   if f"cache-qualify-beta{beta:g}" not in campaign.state["stages"])
    if required > min(available-campaign.state["spent_seconds"], campaign.remaining(True)):
        return finish("MASTER_CACHE_QUALIFICATION_BUDGET_DEFERRED")
    records = []
    for beta in campaign.config["training"]["betas"][1:]:
        remaining = stage_cap(beta)
        if remaining <= campaign.config["execution"]["termination_grace_seconds"] and f"cache-qualify-beta{beta:g}" not in campaign.state["stages"]:
            return finish("MASTER_CACHE_QUALIFICATION_BUDGET_DEFERRED", beta=beta)
        stage = campaign.numerical_stage(f"cache-qualify-beta{beta:g}", {
            "stage": "qualify", "betas": [beta], "plan_file": PLAN},
            cap_seconds=max(remaining, 1.), diagnostic=True, gpu="auto")
        if not stage.get("completed"):
            return finish("MASTER_CACHE_QUALIFICATION_INCOMPLETE", stage=stage)
        row = json.loads(Path(stage["result_path"]).read_text())
        if row.pop("checksum") != digest(row):
            raise ValueError("qualification result checksum changed")
        records.append(row)
    combined = {**records[0], "betas": {}}
    for row in records:
        if any(row[key] != combined[key] for key in row if key != "betas"):
            raise ValueError("candidate qualification scopes differ")
        combined["betas"].update(row["betas"])
    atomic_json(campaign.root / "qualification.json", {**combined, "checksum": digest(combined)})
    atomic_json(campaign.root / "candidate-protocol.json", campaign.config)
    forecast = sum(value["with_existing_safety_factor_seconds"]
                   for key, value in profile["mass_forecasts"].items() if key.endswith(":safe_factor_cache"))
    return finish("MASTER_CACHE_GRAPH_QUALIFIED", qualification=str(campaign.root / "qualification.json"),
        candidate_protocol=str(campaign.root / "candidate-protocol.json"),
        combined_mass_scheduling_forecast_seconds=forecast,
        combined_mass_forecast_fits_campaign=forecast <= campaign.remaining(),
        forecast_role="fixed_start_fixed_metric_extrapolation_not_complete_adaptation_price",
        next_action="plan full preparation and remaining campaign under measured costs; no short adaptation substitute")


def check_profile_sources(old, previous_root, current_root):
    """Preserve the numerical baseline while adding only the profiling harness."""
    if source_snapshot(previous_root) != old:
        raise ValueError("preserved profile baseline changed")
    current = source_snapshot(current_root)
    changed = sorted(p for p in old.keys() | current.keys() if old.get(p) != current.get(p))
    for path in changed:
        if path in NEW_PATHS and path not in old and path in current:
            continue
        if path not in DEFINITIONS or path not in old or path not in current:
            raise ValueError("unreviewed profile source change: " + path)
        trees = []
        for root in (previous_root, current_root):
            tree = ast.parse((Path(root) / path).read_text())
            tree.body = [node for node in tree.body if not isinstance(node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name not in DEFINITIONS[path]]
            trees.append(ast.dump(tree, include_attributes=False))
        if trees[0] != trees[1]:
            raise ValueError("unreviewed changes outside profiling definitions: " + path)
    return changed


def prior_profile_charges(allowance, settled):
    total, count = 0., 0
    seen = set()
    for item in allowance.get("prior_profile_ledgers", []):
        path = Path(item["path"]).resolve()
        if path in seen or checksum(path) != item["sha256"]:
            raise ValueError("duplicate or changed previous profile ledger")
        seen.add(path)
        prior = json.loads(path.read_text())
        if (prior["profile_predecessor"]["sha256"] != settled["source_sha256"] or
                any(a["status"] == "running" or a["stage"] != "factor-profile" for a in prior["attempts"])):
            raise ValueError("unsettled or unrelated previous profile")
        elapsed = sum(a["elapsed_seconds"] for a in prior["attempts"])
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed != prior["spent_seconds"]:
            raise ValueError("invalid previous profile charge")
        total += elapsed
        count += len(prior["attempts"])
    return total, count


def checked_profile_allowance(allowance, settled, sources):
    receipt = allowance["verification"]
    if checksum(receipt["path"]) != receipt["sha256"]:
        raise ValueError("profile verification receipt changed")
    verified = json.loads(Path(receipt["path"]).read_text())
    seconds = verified["wall_seconds"]
    if (verified["status"] != "passed" or verified["sources"] != sources or
            not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("profile verification missing, stale or invalid")
    prior, _ = prior_profile_charges(allowance, settled)
    debit = seconds + SETUP_SECONDS + prior
    if (allowance["predecessor_sha256"] != settled["source_sha256"] or
            allowance["campaign_remaining_seconds"] != settled["campaign_remaining_seconds"]-debit or
            allowance["diagnostic_remaining_seconds"] != settled["diagnostic_remaining_seconds"]-debit or
            allowance.get("unsettled_holds")):
        raise ValueError("profile allowance must debit verification/setup without renewing budget")
    return seconds


def profile_disposition(result):
    if result.get("completed"):
        return ("MASTER_PROFILE_PARITY_PASSED" if result["result"].get("candidate_parity_passed")
                else "MASTER_PROFILE_CANDIDATE_REJECTED")
    return "MASTER_PROFILE_INCOMPLETE"


def profile_master(campaign, *, previous_campaign, previous_source_root):
    if previous_campaign is None or previous_source_root is None:
        raise ValueError("profile requires its settled predecessor and exact source")
    old, settled, _ = read_predecessor(previous_campaign, campaign.config)
    changed = check_profile_sources(old["sources"], previous_source_root, campaign.repo)
    seconds = checked_profile_allowance(campaign.state["allowance"], settled, campaign.state["sources"])
    prior_seconds, prior_count = prior_profile_charges(campaign.state["allowance"], settled)
    starts = {}
    for beta in campaign.config["training"]["betas"][1:]:
        name = f"checkpoint-preparation-beta{beta:g}"
        result = _checked_stage(old, name)
        if not result["result"].get("bootstrap_passed"):
            raise ValueError("profile expects the completed predecessor startup")
        path = Path(old["stages"][name]["result_path"]).parent / "data/starts.json"
        starts[str(beta)] = {"path": str(path), "sha256": checksum(path)}
    lineage = {"campaign": str(Path(previous_campaign).resolve()),
        "source_root": str(Path(previous_source_root).resolve()),
        "sha256": settled["source_sha256"], "changed_sources": changed,
        "verification_seconds": seconds, "setup_seconds": SETUP_SECONDS,
        "phase_cap_seconds": PHASE_SECONDS, "starts": starts,
        "prior_profile_seconds": prior_seconds, "prior_profile_attempts": prior_count}
    if campaign.state.get("profile_predecessor", lineage) != lineage:
        raise ValueError("profile predecessor changed")
    campaign.state["profile_predecessor"] = lineage
    campaign.save()
    request = {"stage": "factor-profile", "plan_file": PLAN, "starts": starts,
               "max_seconds": PHASE_SECONDS - 60.}
    spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"]
                if a["stage"] == "factor-profile")
    cap = min(PHASE_SECONDS-spent-prior_seconds, campaign.remaining(True))
    attempts = prior_count + sum(a["stage"] == "factor-profile" for a in campaign.state["attempts"])
    if cap <= 60. or (attempts >= 3 and "factor-profile" not in campaign.state["stages"]):
        result = {"completed": False, "status": "profile_budget_exhausted"}
    else:
        result = campaign.numerical_stage("factor-profile", request,
            cap_seconds=cap, diagnostic=True, gpu="auto")
    status = profile_disposition(result)
    campaign.state.update(status=status, details=result)
    campaign.save()
    summary = {"status": status, "result": result, "predecessor": lineage,
        "remaining_campaign_seconds": campaign.remaining(),
        "remaining_diagnostic_seconds": campaign.remaining(True),
        "mass_adaptation_completed": False, "production_qualified": False,
        "next_action": "inspect parity and current costs before fresh qualification/preparation"}
    atomic_json(campaign.root / "result.json", summary)
    atomic_json(campaign.root / "settled-allowance.json", {
        "schema": "bayesfilter.q20.recovered_allowance.v1", "source_ledger": str(campaign.path),
        "source_sha256": checksum(campaign.path), "campaign_remaining_seconds": campaign.remaining(),
        "diagnostic_remaining_seconds": campaign.remaining(True), "all_attempts_settled": True,
        "master_status": status, "scope": "Profiling charged to existing campaign; diagnostics included."})
    return summary
