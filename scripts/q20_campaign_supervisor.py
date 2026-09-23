#!/usr/bin/env python3
"""Bounded q20 phase continuation over an unchanged numerical snapshot.

The owner-requested extension changes elapsed-time allocation only. The original
master completes its current attempt before this driver resumes a settled budget
pause in a fresh directory. Numerical identities and finite work budgets remain
owned by the existing repository controller. See the September 21 continuation
plan; this file's checksum is recorded separately from the preserved source tree.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TUNING_STAGE = "tune-neutra-beta1"


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def allocation(state, forecast, deadline_epoch, now):
    """A later calendar date never creates campaign funds or erases past spend."""
    from bayesfilter.inference.q20_stage_budget import repair_remaining
    from types import SimpleNamespace
    if any(a["status"] == "running" for a in state["attempts"]):
        raise ValueError("predecessor has unsettled work")
    campaign = SimpleNamespace(state=state, config=state["config"])
    protected = repair_remaining(campaign)
    for name, amount in (("sample-neutra", forecast["posterior_first_assessment_seconds"]),
                         ("reference", forecast["reference_first_assessment_seconds"]),
                         ("assess-neutra", forecast["reserves"]["assessment"])):
        if name not in state["stages"]:
            spent = sum(a.get("elapsed_seconds", 0.) for a in state["attempts"] if a["stage"] == name)
            protected += max(0., amount - spent)
    spent = sum(a.get("elapsed_seconds", 0.) for a in state["attempts"] if a["stage"] == TUNING_STAGE)
    grace = state["config"]["execution"]["termination_grace_seconds"]
    remaining = state["campaign_limit"] - state["spent_seconds"]
    extra = min(max(0., remaining - protected), max(0., deadline_epoch - now - grace))
    if extra <= 2 * grace:
        raise ValueError("no funded tuning extension before the deadline")
    return {"previous_tuning_spent_seconds": spent, "additional_tuning_seconds": extra,
            "tuning_cumulative_limit_seconds": spent + extra,
            "protected_downstream_seconds": protected,
            "remaining_campaign_seconds": remaining, "additional_campaign_seconds": 0.}


def eligible_predecessor(state):
    return (state["status"] == "ESTIMATION_BUDGET_PAUSED"
            and TUNING_STAGE in state.get("details", {}).get("reason", "")) or (
                state["status"] == "WAITING_FOR_GPU")


def canonical_request(campaign, name, request):
    """Reuse a completed qualification's actual path after checking its content."""
    result = dict(request)
    previous = [a for a in campaign.state["attempts"] if a["stage"] == name]
    if previous and result.get("qualification_path"):
        prior = read_json(Path(previous[0]["directory"]) / "request.json")
        old = prior.get("qualification_path")
        if old and old != result["qualification_path"]:
            if Path(old).read_bytes() != Path(result["qualification_path"]).read_bytes():
                raise ValueError("qualification changed during continuation")
            result["qualification_path"] = old
    return result


def prepare_campaign(job):
    """Import a settled ledger without rewriting any predecessor evidence."""
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot, _process_start
    from bayesfilter.inference.q20_production_config import digest, validate_protocol
    parent = Path(job["parent_campaign"])
    state = read_json(parent / "campaign.json")
    if not eligible_predecessor(state):
        raise ValueError("predecessor is not a tuning budget pause")
    if any(a["status"] == "running" for a in state["attempts"]):
        raise ValueError("predecessor has unsettled work")
    if any(a.get("pid") and _process_start(a["pid"]) == a.get("process_start")
           for a in state["attempts"]):
        raise ValueError("predecessor still owns a live worker")
    validate_protocol(state["config"])
    if (digest(state["config"]) != job["config_hash"]
            or state["config_hash"] != job["config_hash"]
            or state["campaign_limit"] != job["campaign_limit_seconds"]
            or state["spent_seconds"] < job["minimum_settled_seconds"]):
        raise ValueError("predecessor configuration or allowance changed")
    if source_snapshot(job["source_root"]) != state["sources"]:
        raise ValueError("numerical source snapshot changed")
    latest = next(a for a in reversed(state["attempts"]) if a["stage"] == TUNING_STAGE)
    checkpoint = Path(latest["directory"]) / "worker/data/tuning/tuning_checkpoint.json"
    if not checkpoint.is_file():
        raise ValueError("tuning checkpoint is missing")
    if read_json(checkpoint)["result"]["completion_status"] not in {"partial_budget", "paused_infrastructure", "complete"}:
        raise ValueError("tuning checkpoint is not a budget continuation")
    for stage in state["stages"].values():
        if checksum(stage["result_path"]) != stage["result_sha256"]:
            raise ValueError("completed stage receipt changed")
        for path, sha in stage["artifact_hashes"].items():
            if checksum(path) != sha:
                raise ValueError("completed stage evidence changed: " + path)
    amounts = allocation(state, read_json(parent / "forecast-neutra.json"), job["deadline_epoch"], time.time())
    successor = Path(job["output_root"]) / job.get("campaign_name", "campaign-01")
    successor.mkdir(parents=True, exist_ok=False)
    initial_attempts = sum(a["stage"] == TUNING_STAGE for a in state["attempts"])
    extension = {"schema": "bayesfilter.q20.deadline_extension.v1", "created_at": utc_now(),
        "parent_campaign": str(parent), "parent_ledger_sha256": checksum(parent / "campaign.json"),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checksum(checkpoint),
        "source_root": job["source_root"], "driver_path": str(Path(__file__).resolve()),
        "driver_sha256": checksum(__file__), "deadline_epoch": job["deadline_epoch"],
        "deadline_local": job["deadline_local"], "plan_file": job["plan_file"],
        "authority": job["authority"], "additional_attempts": 1,
        "initial_tuning_attempts": initial_attempts, **amounts}
    copied = deepcopy(state)
    copied["stage_limits"][TUNING_STAGE] = amounts["tuning_cumulative_limit_seconds"]
    copied.update(status="CONTINUATION_PREPARED", deadline_extension=extension)
    write_json(successor / "campaign.json", copied)
    write_json(Path(job["output_root"]) / "takeover.json", extension)
    return successor, copied


def resume_extended_tuning(config, bridge, root, request):
    """Resume the repository-issued controller with one explicit time extension."""
    from bayesfilter.inference.q20_production_config import frozen_scope_hash
    from bayesfilter.inference.q20_production_hmc import validate_training_export, _export_tuning_result
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    started = time.monotonic()
    remaining = float(request["cooperative_seconds"])
    extension = request["deadline_extension"]
    remaining = min(remaining, extension["deadline_epoch"] - time.time())
    if remaining <= 0:
        raise StageBudgetPause("extended tuning deadline reached")
    if request.get("method") != "neutra" or request.get("beta", 1.) != 1.:
        raise ValueError("extension is restricted to the existing beta-one NeuTra scope")
    adapter = bridge.fixed_beta_adapter(1.)
    record = read_json(request["training_export"])
    validate_training_export(record, config, adapter, 1.)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    parent = Path(request["resume_checkpoint"])
    shutil.copytree(parent.parent, root / "tuning")
    saved = root / "tuning" / parent.name
    binding, controller = load_numerical_tuning_checkpoint(saved, adapter=adapter)
    if binding._spec["target_lineage"].get("frozen_scope_hash") != frozen_scope_hash(config):
        raise ValueError("resumed tuning protocol differs")
    before = controller.result()
    if before.completion_status not in {"partial_budget", "paused_infrastructure"}:
        raise ValueError("only an incomplete recoverable controller can resume")
    elapsed = before.search_state["elapsed_seconds"]
    remaining -= time.monotonic() - started
    if remaining <= 0:
        raise StageBudgetPause("checkpoint validation consumed the tuning allocation")
    limit = elapsed + remaining
    old_limit = controller.config.max_wall_time_seconds
    if old_limit is None:
        raise ValueError("the predecessor must have a bounded time allocation")
    extended_before = any(event.get("event") == "owner_requested_wall_time_extension"
                          for event in controller._accounting)
    if extended_before:
        # A resource retry consumes the existing extension; it cannot renew it.
        limit = min(limit, old_limit)
    elif limit <= old_limit:
        raise StageBudgetPause("available funds do not extend the old controller time cap")
    if limit <= elapsed:
        raise StageBudgetPause("cumulative tuning time allocation exhausted")
    controller.config = replace(controller.config, max_wall_time_seconds=limit)
    receipt = {"old_controller_limit_seconds": old_limit, "new_controller_limit_seconds": limit,
        "preserved_controller_elapsed_seconds": elapsed, "preserved_work_units": before.budget_used_units,
        "parent_checkpoint": str(parent), "parent_checkpoint_sha256": checksum(parent),
        "execution_binding_hash": binding.binding_hash,
        "change": "elapsed-time ceiling only; numerical configuration and spent counters preserved",
        "deadline_extension": extension}
    write_json(root / "deadline-extension.json", receipt)
    controller._accounting.append({"event": "resume_existing_wall_time_allocation" if extended_before
                                  else "owner_requested_wall_time_extension", **receipt})
    run = run_typed_hmc_candidate_set(binding.typed_adapter, controller.config,
        output_dir=root / "tuning", max_work_items=request.get("max_work_items"),
        _resume_controller=controller)
    old_candidates = {c.candidate_id: c.payload() for c in before.candidates}
    after_candidates = {c.candidate_id: c.payload() for c in run.result.candidates}
    if any(after_candidates[key] != value for key, value in old_candidates.items()):
        raise ValueError("existing candidate identity changed during continuation")
    if tuple(run.result.observations[:len(before.observations)]) != tuple(before.observations):
        raise ValueError("previous numerical observations changed during continuation")
    label = "neutra-beta1-" + record["candidate"]["id"]
    return _export_tuning_result(config, root, "neutra", 1., label, run, binding)


def run_worker(request, output):
    # The normal bootstrap selects a GPU and verifies growth before dispatch.
    # Only the administrative tuning-resume dispatch differs in this process.
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("memory growth must be configured before framework import")
    extension = request["deadline_extension"]
    if checksum(__file__) != extension["driver_sha256"]:
        raise ValueError("continuation driver changed")
    from bayesfilter.inference import q20_master_stages
    original = q20_master_stages.dispatch
    def dispatch(config, bridge, root, actual_request, memory):
        if actual_request["stage"] != "tune":
            raise ValueError("deadline worker only resumes tuning")
        return resume_extended_tuning(config, bridge, root, actual_request)
    q20_master_stages.dispatch = dispatch
    try:
        return q20_master_stages.run_worker(request, output)
    finally:
        q20_master_stages.dispatch = original


def phase_event(campaign, job, stage, event, *, decision, next_action, **details):
    """Keep one current phase decision plus an append-only history."""
    row = {"observed_at_utc": utc_now(), "stage": stage, "event": event,
        "decision": decision, "next_action": next_action,
        "campaign_remaining_seconds": campaign.remaining(),
        "diagnostic_remaining_seconds": campaign.remaining(True),
        "deadline_local": job["deadline_local"], **details}
    with (Path(campaign.root) / "phase-events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_json(Path(campaign.root) / "next-phase.json", row)


def checkpoint_for(attempt, kind):
    data = Path(attempt["directory"]) / "worker/data"
    if kind == "tune":
        path = data / "tuning/tuning_checkpoint.json"
        return path if path.is_file() else None
    if kind == "train":
        paths = sorted(data.glob("cohort-*.json"))
        return paths[-1] if paths else None
    if kind == "price":
        path = data / "pricing-ledger"
        return path if (path / "identity.json").is_file() else None
    path = data / "chunks"
    return path if kind in {"sample", "ensemble", "reference"} and (path / "identity.json").is_file() else None


def recoverable_interruption(result, attempt, kind):
    """Only recorded transient infrastructure and saved numerical state retry."""
    path = checkpoint_for(attempt, kind)
    if path is None:
        return False
    reasons = []
    if result.get("type") in {"UnavailableError", "DeadlineExceededError", "AbortedError", "ResourceExhaustedError"}:
        reasons.append(result["type"])
    if kind == "tune":
        saved = read_json(path)["result"]
        if saved["completion_status"] == "shared_invalidity":
            return False
        reasons.extend(event.get("reason", "") for event in saved.get("accounting_events", [])
                       if event.get("event") == "execution_failure"
                       and event.get("type") == "HMCInfrastructureFailure")
    transient = any(reason.startswith(("UnavailableError", "DeadlineExceededError",
                                      "AbortedError", "ResourceExhaustedError")) for reason in reasons)
    return transient or attempt.get("status") == "interrupted"


def remaining_repair(campaign):
    from bayesfilter.inference.q20_stage_budget import repair_remaining
    # The original accounting already charges failed process exits. Add only
    # resource failures whose numerical controller exited normally.
    normal_exits = sum(a.get("repair_cost_seconds", 0.) for a in campaign.state["attempts"]
                       if a["status"] == "completed")
    return max(0., repair_remaining(campaign) - normal_exits)


def diagnostic_reallocation(campaign, name, request, diagnostic):
    """Apply an inspected pricing timeout repair without renewing total funds."""
    repair = campaign.state.get("operational_repairs", {}).get(name)
    if repair is None:
        return None
    if (not diagnostic or name != "price-ensemble" or request.get("stage") != "price"
            or request.get("method") != "ensemble"
            or repair.get("kind") != "diagnostic_timeout_reallocation"):
        raise ValueError("operational reallocation is restricted to ensemble pricing")
    attempt_seconds = repair["attempt_seconds"]
    total_seconds = repair["cumulative_limit_seconds"]
    if (not math.isfinite(attempt_seconds) or attempt_seconds <= 0
            or not math.isfinite(total_seconds) or total_seconds < attempt_seconds
            or repair["attempt_limit"] != 2
            or campaign.state.get("stage_limits", {}).get(name) != total_seconds):
        raise ValueError("invalid cumulative pricing repair allocation")
    return repair


def reassessment_pricing(campaign, request, result):
    """Credit cached losses without pretending source-import reassessment is free.

    This host quote preserves the worker's original measurement file and emits
    its derivation separately. The numerical source and checkpoint stay fixed.
    """
    if (not campaign.state.get("recovery_plan") or request.get("stage") != "price"
            or not request.get("training_checkpoint") or not result.get("completed")):
        return result
    from bayesfilter.inference.q20_training_resume import read_training_checkpoint
    from bayesfilter.inference.q20_production_config import training_cohort
    saved = read_training_checkpoint(request["training_checkpoint"], campaign.config)
    rows = result["result"]["training"]["rows"]
    details = []
    for candidate in training_cohort(campaign.config, method=request["method"]):
        item = saved["cohort"].get(candidate["id"])
        if item is None or item["exports"] or item["session"]["level_updates"] < campaign.config["training"]["cohort_min_updates"]:
            continue
        beta = item["session"]["map"]["beta"]
        row = next(r for r in rows if r["width"] == candidate["width"] and r["beta"] == beta
                   and r["batch_size"] == campaign.config["training"]["batch_size"])
        # One measured training setup plus two measured heldout calls covers a
        # restoration/reliability hypothesis even when every loss row is cached.
        seconds = row["setup_seconds"]+2*row["heldout_first_seconds"]
        details.append({"candidate": candidate["id"], "beta": beta, "raw_seconds": seconds})
    raw = sum(row["raw_seconds"] for row in details)
    allowance = campaign.config["budget"]["forecast_safety_factor"]*raw
    priced = deepcopy(result)
    before = priced["result"]["training_quote"]["minimum_cohort_seconds"]
    priced["result"]["training_quote"]["minimum_cohort_seconds"] += allowance
    quote = {"previous_training_floor_seconds": before, "additional_reassessment_seconds": allowance,
        "candidates": details, "basis": "measured setup plus two measured heldout first calls per imported map, factor two",
        "role": "engineering_hypothesis_not_measured_reassessment", "checkpoint": request["training_checkpoint"]}
    write_json(Path(campaign.root)/f"reassessment-quote-{request['method']}.json", quote)
    priced["result"]["reassessment_quote"] = quote
    return priced


def execute_phase(campaign, job, name, request, *, diagnostic=False, reserve=None):
    from bayesfilter.inference.q20_master_program import StageIncomplete
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    base = canonical_request(campaign, name, {**request, "fixture": False})
    if name in campaign.state["stages"]:
        return reassessment_pricing(campaign, base,
            campaign.numerical_stage(name, base, cap_seconds=0., diagnostic=diagnostic, gpu="auto"))
    reallocation = diagnostic_reallocation(campaign, name, request, diagnostic)
    recovery = campaign.state.get("recovery_plan", {})
    work = recovery.get("diagnostic_blocks", {}).get(name)
    if recovery and diagnostic and work is None:
        raise ValueError("diagnostic phase is absent from finite recovery plan: " + name)
    if work is not None and name not in campaign.state.setdefault("stage_limits", {}):
        campaign.state["stage_limits"][name] = work["maximum_seconds"]
        campaign.save()
    if reserve is not None and name not in campaign.state.setdefault("stage_limits", {}):
        if not math.isfinite(reserve) or reserve <= 0:
            raise ValueError("invalid stage reserve")
        campaign.state["stage_limits"][name] = reserve
        campaign.save()
    while True:
        spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == name)
        cfg = campaign.config
        grace = cfg["execution"]["termination_grace_seconds"]
        cap = min(campaign.stage_remaining(name), campaign.remaining(diagnostic),
            (work["maximum_seconds"] if work is not None else
             reallocation["attempt_seconds"] if reallocation is not None else
             cfg["execution"]["diagnostic_attempt_seconds"]) if diagnostic else
            max(0., reserve-spent) if reserve is not None else cfg["budget"]["arm_cap_seconds"],
            job["deadline_epoch"] - time.time() - grace)
        if cap <= 2 * grace:
            phase_event(campaign, job, name, "allocation_stop", decision="stop_for_budget",
                        next_action="inspect funded chunks; new funds or deadline require owner direction")
            raise StageBudgetPause("calendar or funded stage limit: " + name)
        minimum = work["minimum_seconds"] if work else 0.
        previous_blocks = [a for a in campaign.state["attempts"] if a["stage"] == name]
        if previous_blocks and previous_blocks[-1].get("failure_classification") == "timing_unqualified":
            minimum = max(minimum, cfg["budget"]["forecast_safety_factor"] * previous_blocks[-1]["elapsed_seconds"])
        if request["stage"] == "price" and previous_blocks:
            checkpoint = checkpoint_for(previous_blocks[-1], "price")
            if checkpoint and (checkpoint / "pending.json").exists():
                minimum = read_json(checkpoint / "pending.json")["required_seconds"] + recovery.get("startup_reserve_seconds", 0.)
        if cap < minimum:
            phase_event(campaign, job, name, "allocation_stop", decision="next_block_unfunded",
                next_action="preserve completed work; quote the next missing measurement", required_seconds=minimum,
                available_seconds=cap)
            raise StageBudgetPause("remaining cap cannot cover next useful block: " + name)
        attempts = [a for a in campaign.state["attempts"] if a["stage"] == name
                    and a.get("failure_classification") != "resource_unavailable"]
        extension = campaign.state.get("deadline_extension", {"initial_tuning_attempts": 0})
        normal_limit = (extension["initial_tuning_attempts"] + 1 if name == TUNING_STAGE
                        else cfg["execution"]["stage_attempts"])
        if reallocation is not None:
            normal_limit = reallocation["attempt_limit"]
        if recovery:
            normal_limit = (2 if any(a.get("failure_classification") == "timing_unqualified" for a in attempts) else 1)
            normal_limit += recovery.get("repair_attempt_baselines", {}).get(name, 0)
        retry = bool(attempts and attempts[-1].get("failure_classification") == "resumable_infrastructure")
        used_repairs = len(campaign.state.get("recovery_attempts", {}).get(name, []))
        is_recovery = retry
        if retry or len(attempts) >= normal_limit:
            if not retry or used_repairs >= job["max_recovery_attempts_per_phase"] or remaining_repair(campaign) <= 2*grace:
                phase_event(campaign, job, name, "repair_limit", decision="agent_repair_required",
                    next_action="inspect repeated failure; no automatic renewal of attempts or repair funds")
                raise StageIncomplete("attempt_limit:" + name)
        diagnostic_attempts = [a for a in campaign.state["attempts"] if a["diagnostic"]
            and a["stage"] != "gpu-readiness" and a.get("failure_classification") != "resource_unavailable"]
        if diagnostic and not recovery and len(diagnostic_attempts) >= cfg["execution"]["diagnostic_attempts"]:
            raise StageIncomplete("diagnostic_attempt_limit:" + name)
        decision = ("restart_pricing_with_revised_allocation" if reallocation is not None else
                    "resume_checkpoint" if attempts else "run_phase")
        phase_event(campaign, job, name, "dispatch", decision=decision,
            next_action="classify result and refresh next phase", allocated_seconds=cap)
        result = campaign.numerical_stage(name, base, cap_seconds=cap, diagnostic=diagnostic, gpu="auto")
        attempt = campaign.state["attempts"][-1]
        if is_recovery and result.get("status") != "waiting_for_gpu":
            campaign.state.setdefault("recovery_attempts", {}).setdefault(name, []).append(len(campaign.state["attempts"])-1)
            campaign.save()
        phase_event(campaign, job, name, "result", decision="inspect_recorded_result",
            next_action="apply phase outcome", result_status=result.get("status"),
            elapsed_seconds=attempt.get("elapsed_seconds"), evidence_directory=attempt["directory"])
        if result.get("status") == "waiting_for_gpu":
            phase_event(campaign, job, name, "resource_wait", decision="wait_and_retry",
                next_action="select an available GPU; preserve all completed phases",
                poll_seconds=job["poll_seconds"])
            time.sleep(min(job["poll_seconds"], max(0., job["deadline_epoch"]-time.time())))
            continue
        if result.get("completed"):
            if request["stage"] == "price-selected" and result.get("terminal_resource", {}).get("quality") not in {
                    "no_contention_observed", "cpu_reference_gpu_hidden"}:
                # Keep the complete timing artifact, but it cannot be the final
                # uncontended quote. Repeat this one block at most once, with
                # unchanged cumulative caps; never repeat training or tuning.
                campaign.state["stages"].pop(name, None)
                attempt["failure_classification"] = "timing_unqualified"
                campaign.save()
                phase_event(campaign, job, name, "timing_remeasurement", decision="preserve_and_reprice_one_block",
                    next_action="check remaining cumulative cap before one clean timing retry")
                if sum(a.get("failure_classification") == "timing_unqualified" for a in attempts+[attempt]) >= 2:
                    raise StageIncomplete("selected pricing remains contended: " + name)
                continue
            phase_event(campaign, job, name, "phase_complete", decision="return_to_master",
                next_action="evaluate scientific result and dispatch its declared next phase")
            return reassessment_pricing(campaign, base, result)
        if result.get("budget_paused"):
            phase_event(campaign, job, name, "budget_pause", decision="preserve_checkpoint",
                next_action="inspect exhausted time, funds or work units; do not switch methods")
            raise StageBudgetPause(name + ": " + result["status"])
        if recoverable_interruption(result, attempt, request["stage"]):
            attempt["failure_classification"] = "resumable_infrastructure"
            # Some tuners return a normal process exit after classifying a
            # resource failure. Its real time still consumes the repair hold.
            attempt["repair_cost_seconds"] = attempt.get("elapsed_seconds", 0.)
            campaign.save()
            if remaining_repair(campaign) <= 2*grace:
                raise StageIncomplete("repair_hold_exhausted:" + name)
            phase_event(campaign, job, name, "infrastructure_repair", decision="retry_saved_checkpoint",
                next_action="fresh worker and resource selection within cumulative limits",
                failure=result.get("type", result.get("result", {}).get("status")),
                scientific_decision="no_candidate_or_method_rejection")
            continue
        phase_event(campaign, job, name, "repair_required", decision="agent_repair_required",
            next_action="inspect the recorded exception or veto, implement a focused fix, test, then ensure",
            failure=result.get("type", result.get("status")))
        raise StageIncomplete(result.get("status", "unknown_failure"))


def continue_campaign(job, root, state):
    from bayesfilter.inference.q20_campaign_runtime import Campaign
    from bayesfilter.inference.q20_master_program import run_estimation_attempts, StageIncomplete, ResourceWait
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    extension = state.get("deadline_extension")
    class DeadlineCampaign(Campaign):
        def execute(self, name, command, **kwargs):
            if checksum(__file__) != extension["driver_sha256"]:
                raise ValueError("continuation driver changed")
            request = kwargs.get("request")
            if name == TUNING_STAGE and request is not None:
                if not request.get("resume_checkpoint"):
                    raise ValueError("extended stage must resume the existing numerical checkpoint")
                kwargs["request"] = {**request, "deadline_extension": extension,
                                     "plan_file": job["plan_file"]}
                command = list(command)
                command[1] = str(Path(__file__).resolve())
            return super().execute(name, command, **kwargs)
    campaign_class = Campaign if state.get("recovery_plan") else DeadlineCampaign
    campaign = campaign_class(root, repo=job["source_root"], config=state["config"])
    with campaign.locked():
        def finish(status, **details):
            campaign.state.update(status=status, details=details, production_qualified=False)
            campaign.save()
            result = {"status": status, "details": details, "campaign_root": str(root),
                "remaining_campaign_seconds": campaign.remaining(),
                "remaining_diagnostic_seconds": campaign.remaining(True),
                "declared_estimation_checks_passed": bool(details.get("estimation_passed")),
                "production_qualified": False, "method_ranking": "not_estimated"}
            write_json(Path(root) / "result.json", result)
            phase_event(campaign, job, "master", "terminal", decision=status,
                next_action=details.get("next_action", "inspect terminal evidence and the last phase event"),
                terminal_details=details)
            return result
        def stage(name, request, **kwargs):
            return execute_phase(campaign, job, name, request, **kwargs)
        try:
            return run_estimation_attempts(campaign, stage, finish)
        except StageBudgetPause as exc:
            return finish("ESTIMATION_BUDGET_PAUSED", reason=str(exc))
        except ResourceWait as exc:
            return finish("WAITING_FOR_GPU", resource=exc.result)
        except StageIncomplete as exc:
            return finish("MASTER_INCOMPLETE", reason=str(exc))
        except BaseException as exc:
            finish("MASTER_INFRASTRUCTURE_FAILURE", type=type(exc).__name__, reason=str(exc))
            raise


def supervise(job):
    root = Path(job["output_root"])
    def status(value, **details):
        record = {"observed_at_utc": utc_now(), "status": value,
                  "deadline_local": job["deadline_local"], **details}
        write_json(root / "status.json", record)
        return record
    if checksum(__file__) != job["driver_sha256"]:
        raise ValueError("continuation driver changed after launch preparation")
    successor = root / job.get("campaign_name", "campaign-01")
    if (successor / "campaign.json").exists():
        state = read_json(successor / "campaign.json")
        resumable = state["status"].startswith("running:") or state["status"] in {
            "CONTINUATION_PREPARED", "WAITING_FOR_GPU", "MASTER_INTERRUPTED"}
        if not resumable:
            return status("EXISTING_CAMPAIGN_REQUIRES_INSPECTION", campaign_status=state["status"],
                          next_action="read next-phase.json; no implicit budget renewal")
        status("RESUMING_EXISTING_CAMPAIGN", campaign_root=str(successor))
        result = continue_campaign(job, successor, state)
        return status(result["status"], result=result)
    while True:
        if time.time() >= job["deadline_epoch"]:
            return status("CALENDAR_DEADLINE_REACHED")
        raw = subprocess.check_output(["systemctl", "--user", "show", job["parent_unit"],
            "--property=ActiveState,SubState,MainPID"], text=True)
        service = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if service.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}:
            parent_state = read_json(Path(job["parent_campaign"]) / "campaign.json")
            status("WAITING_FOR_EXISTING_MASTER", parent_service=service,
                   parent_phase=parent_state["status"], numerical_work_launched=False,
                   next_action="existing master continues; take over its settled budget/resource pause")
            time.sleep(min(job["poll_seconds"], max(0., job["deadline_epoch"]-time.time())))
            continue
        break
    parent = read_json(Path(job["parent_campaign"]) / "campaign.json")
    if parent["status"] == "STABLE_ESTIMATE_OBTAINED":
        return status("PREDECESSOR_OBTAINED_ESTIMATE", numerical_work_launched=False)
    if not eligible_predecessor(parent):
        return status("PREDECESSOR_REQUIRES_INSPECTION", predecessor_status=parent["status"],
                      predecessor_details=parent.get("details"), numerical_work_launched=False)
    successor, state = prepare_campaign(job)
    status("CONTINUING_MASTER", campaign_root=str(successor),
           allocation=state["deadline_extension"])
    result = continue_campaign(job, successor, state)
    return status(result["status"], result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "worker"))
    parser.add_argument("--job", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    record = read_json(args.job if args.mode == "supervise" else args.request)
    source = record["source_root"] if args.mode == "supervise" else record["deadline_extension"]["source_root"]
    sys.path.insert(0, source)
    if args.mode == "supervise":
        try:
            with (Path(record["output_root"]) / "supervisor.lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = supervise(record)
        except BaseException as exc:
            write_json(Path(record["output_root"]) / "status.json", {"observed_at_utc": utc_now(),
                "status": "CONTINUATION_FAILED", "type": type(exc).__name__, "reason": str(exc)})
            raise
    else:
        result = run_worker(record, args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
