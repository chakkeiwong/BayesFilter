#!/usr/bin/env python3
"""Administrative q20 deadline continuation over an unchanged numerical snapshot.

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
            and TUNING_STAGE in state.get("details", {}).get("reason", ""))


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
    if read_json(checkpoint)["result"]["completion_status"] != "partial_budget":
        raise ValueError("tuning checkpoint is not a budget continuation")
    for stage in state["stages"].values():
        if checksum(stage["result_path"]) != stage["result_sha256"]:
            raise ValueError("completed stage receipt changed")
        for path, sha in stage["artifact_hashes"].items():
            if checksum(path) != sha:
                raise ValueError("completed stage evidence changed: " + path)
    amounts = allocation(state, read_json(parent / "forecast-neutra.json"), job["deadline_epoch"], time.time())
    successor = Path(job["output_root"]) / "campaign-01"
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
    if before.completion_status != "partial_budget":
        raise ValueError("only a budget-paused controller can receive this extension")
    elapsed = before.search_state["elapsed_seconds"]
    remaining -= time.monotonic() - started
    if remaining <= 0:
        raise StageBudgetPause("checkpoint validation consumed the tuning allocation")
    limit = elapsed + remaining
    old_limit = controller.config.max_wall_time_seconds
    if old_limit is None or limit <= old_limit:
        raise StageBudgetPause("available funds do not extend the old controller time cap")
    controller.config = replace(controller.config, max_wall_time_seconds=limit)
    receipt = {"old_controller_limit_seconds": old_limit, "new_controller_limit_seconds": limit,
        "preserved_controller_elapsed_seconds": elapsed, "preserved_work_units": before.budget_used_units,
        "parent_checkpoint": str(parent), "parent_checkpoint_sha256": checksum(parent),
        "execution_binding_hash": binding.binding_hash,
        "change": "elapsed-time ceiling only; numerical configuration and spent counters preserved",
        "deadline_extension": extension}
    write_json(root / "deadline-extension.json", receipt)
    controller._accounting.append({"event": "owner_requested_wall_time_extension", **receipt})
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


def continue_campaign(job, root, state):
    from bayesfilter.inference.q20_campaign_runtime import Campaign
    from bayesfilter.inference.q20_master_program import run_estimation_attempts, StageIncomplete, ResourceWait
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    extension = state["deadline_extension"]
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
    campaign = DeadlineCampaign(root, repo=job["source_root"], config=state["config"])
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
            return result
        def stage(name, request, *, diagnostic=False, reserve=None):
            base = canonical_request(campaign, name, {**request, "fixture": False})
            if name in campaign.state["stages"]:
                return campaign.numerical_stage(name, base, cap_seconds=0., diagnostic=diagnostic, gpu="auto")
            if reserve is not None and name not in campaign.state.setdefault("stage_limits", {}):
                if not math.isfinite(reserve) or reserve <= 0:
                    raise ValueError("invalid stage reserve")
                campaign.state["stage_limits"][name] = reserve
                campaign.save()
            spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == name)
            cfg = campaign.config
            grace = cfg["execution"]["termination_grace_seconds"]
            cap = min(campaign.stage_remaining(name), campaign.remaining(diagnostic),
                cfg["execution"]["diagnostic_attempt_seconds"] if diagnostic else
                max(0., reserve-spent) if reserve is not None else cfg["budget"]["arm_cap_seconds"],
                job["deadline_epoch"] - time.time() - grace)
            if cap <= 2 * grace:
                raise StageBudgetPause("calendar or funded stage limit: " + name)
            attempts = [a for a in campaign.state["attempts"] if a["stage"] == name
                        and a.get("failure_classification") != "resource_unavailable"]
            allowed = (extension["initial_tuning_attempts"] + 1 if name == TUNING_STAGE
                       else cfg["execution"]["stage_attempts"])
            if len(attempts) >= allowed:
                raise StageIncomplete("attempt_limit:" + name)
            diagnostic_attempts = [a for a in campaign.state["attempts"] if a["diagnostic"]
                and a["stage"] != "gpu-readiness" and a.get("failure_classification") != "resource_unavailable"]
            if diagnostic and len(diagnostic_attempts) >= cfg["execution"]["diagnostic_attempts"]:
                raise StageIncomplete("diagnostic_attempt_limit:" + name)
            result = campaign.numerical_stage(name, base, cap_seconds=cap, diagnostic=diagnostic, gpu="auto")
            if result.get("status") == "waiting_for_gpu":
                raise ResourceWait(result)
            if not result.get("completed"):
                if result.get("budget_paused"):
                    raise StageBudgetPause(name + ": " + result["status"])
                raise StageIncomplete(result["status"])
            return result
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
    while True:
        if time.time() >= job["deadline_epoch"]:
            return status("CALENDAR_DEADLINE_REACHED")
        raw = subprocess.check_output(["systemctl", "--user", "show", job["parent_unit"],
            "--property=ActiveState,SubState,MainPID"], text=True)
        service = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if service.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}:
            status("WAITING_FOR_EXISTING_MASTER", parent_service=service,
                   numerical_work_launched=False)
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
