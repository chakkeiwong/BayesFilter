"""Executable q20 campaign coordinator; no accelerator imports in this process."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import sys

from bayesfilter.inference.q20_campaign_runtime import Campaign, CampaignBudgetError, atomic_json
from bayesfilter.inference.q20_production_config import digest, training_cohort, method_betas, method_schedule, validate_protocol
from bayesfilter.inference.q20_stage_budget import StageBudgetPause, allocate_stage, repair_remaining


def forecast_campaign(config, pricing):
    """Forecast work separately from the bounded allocations that fund it."""
    if pricing["config_hash"] != digest(config):
        raise ValueError("pricing protocol differs")
    method = pricing["method"]
    positive = method_betas(config, method)
    p,h=config["posterior"],config["tuning"]
    rows=[r for r in pricing["hmc"] if r["kind"].startswith("chart-") and r["beta"] in positive]
    if not rows:
        raise ValueError("HMC costs were not measured")
    expected = {(beta, f"chart-w{width}", length) for beta in positive
                for width in config["training"]["widths"]
                for length in {min(h["l_grid"]), max(h["l_grid"])}}
    if {(row["beta"], row["kind"], row["L"]) for row in rows} != expected:
        raise ValueError("method transition pricing has missing or unexpected scopes")
    if pricing["training_quote"].get("method", method) != method:
        raise ValueError("training forecast belongs to a different method")
    # State-dependent runtime and future learned weights remain forecast risks.
    step=max(r["steady_seconds"] for r in rows)
    compile_cost=max(r["first_seconds"] for r in rows)
    scopes=1 if method == "neutra" else config["ensemble"]["charts"]*len(positive)
    transitions=p["warmup_max"]+p["retained_max"]
    if method == "ensemble":
        if (pricing.get("exchange_price_role") != "actual_multi_chart_mixture"
                or pricing.get("exchange_charts_per_temperature") != config["ensemble"]["charts"]):
            raise ValueError("ensemble forecast requires actual multi-chart dispatch pricing")
        posterior_step, posterior_compile = pricing["exchange_steady_seconds"], pricing["exchange_first_seconds"]
    else:
        posterior_step, posterior_compile = step, compile_cost
    if any(not math.isfinite(v) or v <= 0 for v in (step, compile_cost, posterior_step, posterior_compile)):
        raise ValueError("positive finite method-specific transition prices required")
    posterior=transitions*posterior_step+posterior_compile
    reference_batches=config["reference"]["banks"]*max(config["reference"]["rungs"])/config["reference"]["batch_size"]
    reference=pricing["reference_batch_first_seconds"]+reference_batches*pricing["reference_batch_steady_seconds"]
    reference+=pricing["reference_analysis_seconds_per_row"]*reference_batches*config["reference"]["batch_size"]
    checks=math.ceil(p["warmup_max"]/p["warmup_chunk"])+math.ceil(p["retained_max"]/p["retained_chunk"])
    analysis=(checks+1)*pricing["posterior_analysis_seconds"]
    startup=pricing["worker_initialization_seconds"]+pricing.get("process_overhead_seconds",0.)+2*config["execution"]["termination_grace_seconds"]
    reference+=startup
    posterior+=startup
    analysis+=startup
    factor=config["budget"]["forecast_safety_factor"]
    raw={"reference":reference,"posterior":posterior,"assessment":analysis}
    reserves={key:factor*value for key,value in raw.items()}
    reserves["tuning"] = scopes*h["max_wall_seconds"]
    reserves["localized_repair"] = config["budget"]["repair_allocation_seconds"]
    minimum=pricing["training_quote"]["minimum_cohort_seconds"]+factor*startup
    reference_rows = config["reference"]["banks"] * config["reference"]["rungs"][min(1, len(config["reference"]["rungs"])-1)]
    reference_floor = factor*(startup + pricing["reference_batch_first_seconds"]
        + reference_rows/config["reference"]["batch_size"]*pricing["reference_batch_steady_seconds"]
        + reference_rows*pricing["reference_analysis_seconds_per_row"])
    if method == "neutra":
        shortest = [r for r in rows if r["L"] == min(h["l_grid"])]
        first_step, first_compile = max(r["steady_seconds"] for r in shortest), max(r["first_seconds"] for r in shortest)
    else:
        first_step, first_compile = posterior_step, posterior_compile
    first_chunk = factor*(startup+first_compile+p["warmup_chunk"]*first_step+pricing["posterior_analysis_seconds"])
    earliest_warmup = min(p["warmup_max"], math.ceil(p["warmup_min"]/p["warmup_chunk"])*p["warmup_chunk"])
    earliest_retained = min(p["retained_max"], math.ceil(p["retained_min"]/p["retained_chunk"])*p["retained_chunk"])
    first_assessment = factor*(startup+first_compile+(earliest_warmup+earliest_retained)*first_step
        +(math.ceil(earliest_warmup/p["warmup_chunk"])+math.ceil(earliest_retained/p["retained_chunk"]))*pricing["posterior_analysis_seconds"])
    missing_cost_categories = list(pricing.get("missing_cost_categories", []))
    if pricing["training_quote"].get("missing_training_scopes") and "training" not in missing_cost_categories:
        missing_cost_categories.append("training")
    return {"schema":"bayesfilter.q20.estimation_cost_forecast.v2","config_hash":digest(config), "method": method,
        "reserves":reserves,"training_floor_seconds":minimum,
        "training_cap_seconds":pricing["training_quote"]["full_training_cap_seconds"]+factor*startup,
        "reference_first_assessment_seconds":reference_floor,
        "posterior_first_chunk_seconds":first_chunk,
        "posterior_first_assessment_seconds":first_assessment,
        "posterior_first_chunk_basis":"shortest_measured_L_conditional_until_verified_member_selected" if method == "neutra" else "measured_chart_mixture",
        "tuning_scope_limit_seconds":h["max_wall_seconds"],
        "allocation_policy":"successive_stages_clipped_to_actual_remaining_allowance",
        "reserves_role":"individual_bounds_not_simultaneous_reservation_or_expected_duration",
        "forecast_safety_factor":factor,"factor_status":"engineering_hypothesis",
        "scope_count":scopes,"status":"partial_downstream_forecast" if missing_cost_categories else "all_required_stages_forecast",
        "full_campaign_priced": not bool(missing_cost_categories),
        "missing_cost_categories": missing_cost_categories,
        "limitations":["few timings do not bound compile or runtime tails",
                       "new learned maps can change numerical behavior",
                       "precision and reference qualification may still fail at declared caps"]}


def posterior_chunk_forecast(config, pricing, members):
    """Price the selected trajectory without using runtime to select a kernel."""
    if pricing["method"] == "ensemble":
        step, first = pricing["exchange_steady_seconds"], pricing["exchange_first_seconds"]
    else:
        lengths = [r["L"] for r in pricing["hmc"]]
        length = members["neutra-beta1"].get("L", max(lengths))
        priced_length = min((n for n in lengths if n >= length), default=None)
        if priced_length is None:
            raise ValueError("selected trajectory lies outside measured L coverage")
        rows = [r for r in pricing["hmc"] if r["L"] == priced_length]
        step, first = max(r["steady_seconds"] for r in rows), max(r["first_seconds"] for r in rows)
    count = max(config["posterior"]["warmup_chunk"], config["posterior"]["retained_chunk"])
    return config["budget"]["forecast_safety_factor"] * (first+count*step+pricing["posterior_analysis_seconds"])


def selected_procedure_forecast(config, pricing, selected, members):
    """Quote the exact frozen procedure, including its first posterior assessment."""
    if selected["status"] != "selected_procedure_priced" or selected["method"] != pricing["method"]:
        raise ValueError("selected procedure pricing scope differs")
    paths = ([members["neutra-beta1"]["path"]] if pricing["method"] == "neutra" else
             [members[f"ensemble-chart-{i}-beta{b:g}"]["path"] for b in method_betas(config, "ensemble")
              for i in range(config["ensemble"]["charts"])])
    measured = selected.get("members", [p for rows in selected.get("members_by_beta", {}).values() for p in rows])
    if paths != measured:
        raise ValueError("selected price does not bind the frozen members")
    first, step = selected["first_seconds"], selected["steady_seconds"]
    if any(not math.isfinite(x) or x <= 0 for x in (first, step)):
        raise ValueError("invalid selected procedure timing units")
    p, factor = config["posterior"], config["budget"]["forecast_safety_factor"]
    startup = selected["worker_initialization_seconds"] + selected.get("process_overhead_seconds", 0.)
    startup += 2*config["execution"]["termination_grace_seconds"]
    warm = math.ceil(p["warmup_min"]/p["warmup_chunk"])*p["warmup_chunk"]
    retained = math.ceil(p["retained_min"]/p["retained_chunk"])*p["retained_chunk"]
    checks = warm//p["warmup_chunk"] + retained//p["retained_chunk"]
    analysis = pricing["posterior_analysis_seconds"]
    return {"first_assessment_seconds": factor*(startup+first+(warm+retained)*step+checks*analysis),
        "chunk_seconds": factor*(first+max(p["warmup_chunk"], p["retained_chunk"])*step+analysis),
        "maximum_seconds": factor*(startup+first+(p["warmup_max"]+p["retained_max"])*step
            +(math.ceil(p["warmup_max"]/p["warmup_chunk"])+math.ceil(p["retained_max"]/p["retained_chunk"]))*analysis),
        "role": "measured_exact_procedure_engineering_reservation", "factor": factor}


def choose_maps(config, checkpoint, *, schedule="continuation", count=None, exclude=()):
    cohort=json.loads(Path(checkpoint).read_text())["cohort"]
    chosen=[]
    used_roots=set()
    # Fixed operational ordering; no loss/ESS ranking.
    for candidate in training_cohort(config):
        if candidate["schedule"] != schedule or candidate["id"] in exclude:
            continue
        item=cohort.get(candidate["id"])
        if item is None or candidate["root"] in used_roots:
            continue
        exports=item["exports"]
        valid=True
        for beta in config["training"]["betas"][1:]:
            if schedule == "direct" and beta != 1.:
                continue
            if str(beta) not in exports:
                valid=False
                break
            row=json.loads(Path(exports[str(beta)]).read_text())
            if not row["assessment"]["map_reliability"]["passed"]:
                valid=False
            if config["role"]!="smoke" and not row["assessment"]["decision"].get("hmc_trial_eligible", False):
                valid=False
            if config["role"] != "smoke":
                from bayesfilter.inference.neutra_post_training import require_post_training_assessment
                try:
                    require_post_training_assessment(row)
                except ValueError:
                    valid = False
        if valid:
            chosen.append({"candidate":candidate,"exports":exports})
            used_roots.add(candidate["root"])
        if len(chosen)==(config["ensemble"]["charts"] if count is None else count):
            break
    return chosen


def next_training_rung(config, checkpoint, method):
    """Next funded checkpoint, never inferred from trial eligibility."""
    if checkpoint is None:
        return config["training"]["cohort_min_updates"]
    cohort = json.loads(Path(checkpoint).read_text())["cohort"]
    levels = [item["session"]["level_updates"] for name, item in cohort.items()
              if name.startswith(method_schedule(method)+"-") and "session" in item]
    if not levels:  # Includes mechanics fixtures without optimizer state.
        return config["training"]["cohort_min_updates"] if any("session" in i for i in cohort.values()) else None
    return next((r for r in config["training"]["rungs"]
                 if r > min(levels) and r >= config["training"]["cohort_min_updates"]), None)


def tune_map_cohort(config, checkpoint, method, tune):
    """Try alternatives after candidate failure; shared errors propagate."""
    excluded, failures = set(), []
    count = 1 if method == "neutra" else config["ensemble"]["charts"]
    while True:
        maps = choose_maps(config, checkpoint, schedule=method_schedule(method),
                           count=count, exclude=excluded)
        if len(maps) != count:
            return [], {}, failures
        members = {}
        failed = None
        for i, item in enumerate(maps):
            for beta in method_betas(config, method):
                label = "neutra-beta1" if method == "neutra" else f"ensemble-chart-{i}-beta{beta:g}"
                member = tune(item, beta, label, len(members))
                if member is None:
                    failed = item["candidate"]["id"]
                    break
                members[label] = member
            if failed is not None:
                break
        if failed is None:
            return maps, members, failures
        excluded.add(failed)
        failures.append({"candidate": failed, "status": "tuning_candidate_failed",
                         "scientific_rejection": False})


def selected_member(tuned):
    if tuned["status"]!="complete" or not tuned["verified_members"]:
        return None
    key=sorted(tuned["verified_members"])[0]
    return {"candidate_id":key,"path":tuned["verified_members"][key],
            **tuned.get("verified_member_parameters", {}).get(key, {}),
            "selection_policy":"lexicographic_verified_id_operational_not_efficiency_ranking"}


class StageIncomplete(RuntimeError):
    pass


class ResourceWait(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__("waiting_for_gpu")


def checked_completed_tuning_failure(config, method, saved):
    """Carry inspected negative evidence across a host-only source refresh."""
    receipt = saved["stage_receipt"]
    raw = Path(receipt["result_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["result_sha256"]:
        raise ValueError("completed tuning failure result changed")
    for path, expected in receipt["artifact_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("completed tuning failure artifact changed: " + path)
    worker = json.loads(raw)
    result = worker.get("result", {})
    if (method != "neutra" or not worker.get("completed")
            or result.get("config_hash") != digest(config) or result.get("method") != method
            or result.get("beta") != 1. or result.get("status") != "complete"
            or result.get("verified_members") != {}):
        raise ValueError("only completed plain tuning rejection may be carried forward")
    checkpoint_path = result["tuning_checkpoint"]
    if checkpoint_path not in receipt["artifact_hashes"]:
        raise ValueError("completed tuning failure lacks checked checkpoint")
    checkpoint = json.loads(Path(checkpoint_path).read_text())["result"]
    if (checkpoint["completion_status"] != "complete" or checkpoint["verified_candidate_ids"]
            or not checkpoint["candidate_states"]
            or set(checkpoint["candidate_states"].values()) != {"promotion_failed"}):
        raise ValueError("tuning checkpoint does not establish completed rejection")
    return {"method": method, "status": "tuning_candidate_failed", "verified_scopes": [],
            "preserved_result": receipt["result_path"]}


def run_estimation_attempts(campaign, stage, finish, *, stop_after=None):
    """Try permitted samplers in declared order and retain the first valid estimate.

    A completed candidate failure permits the next method. Worker exceptions and
    interruptions propagate to the supervisor; they cannot nominate a fallback.
    """
    config = campaign.config
    recovery = campaign.state.get("recovery_plan", {})
    outcomes = []
    imported = campaign.state.get("training_resume")
    checkpoint = None if imported is None else imported["checkpoint"]
    if checkpoint is None and "calibration" in campaign.state["stages"]:
        saved = campaign.state["stages"]["calibration"]
        data = Path(saved["result_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != saved["result_sha256"]:
            raise ValueError("completed calibration result changed")
        checkpoint = json.loads(data)["result"]["checkpoint"]
    for method in config["estimation"]["methods"]:
        previous_failure = recovery.get("completed_tuning_failures", {}).get(method)
        if previous_failure is not None:
            outcomes.append(checked_completed_tuning_failure(config, method, previous_failure))
            # Historical failure belongs to its frozen map; refreshed or
            # alternative transports remain eligible for fresh trials below.
        positive = method_betas(config, method)
        qualified_request = {}
        if config["jit_compile"]:
            records = []
            for beta in positive:
                qualified = stage(f"qualify-beta{beta:g}", {"stage": "qualify", "betas": [beta]}, diagnostic=True)
                record = json.loads(Path(qualified["result_path"]).read_text())
                if record.pop("checksum") != digest(record):
                    raise ValueError("qualification receipt checksum mismatch")
                records.append(record)
            combined = {**records[0], "betas": {}}
            for record in records:
                if any(record[key] != combined[key] for key in record if key != "betas"):
                    raise ValueError("qualification scopes differ")
                combined["betas"].update(record["betas"])
            path = campaign.root / f"qualification-{method}.json"
            atomic_json(path, {**combined, "checksum": digest(combined)})
            qualified_request = {"qualification_path": str(path), "qualification_betas": positive}

        request = {"stage": "price", "method": method,
                   "reservation_limit_seconds": campaign.remaining(), **qualified_request}
        if method in recovery.get("pricing_imports", {}):
            request["historical_pricing"] = recovery["pricing_imports"][method]
            request["block_reserves"] = recovery.get("pricing_block_reserves", {}).get(method, {})
        # Freeze the original allowance used for this completed/interrupted
        # request, so replay does not change its identity as time is charged.
        for attempt in campaign.state["attempts"]:
            if attempt["stage"] == "price-"+method:
                prior_request = json.loads((Path(attempt["directory"]) / "request.json").read_text())
                request["reservation_limit_seconds"] = prior_request["reservation_limit_seconds"]
                break
        if checkpoint is not None:
            request["training_checkpoint"] = checkpoint
        pricing = stage("price-"+method, request, diagnostic=True)
        if pricing["result"]["status"] == "unaffordable_under_declared_reservation":
            outcome = {"method": method, "status": "under_budgeted", "forecast": pricing["result"]["training_quote"]}
            outcomes.append(outcome)
            if stop_after == "price":
                return finish("UNDER_BUDGETED", method=method, forecast=pricing["result"], attempts=outcomes)
            return finish("ESTIMATION_BUDGET_PAUSED", method=method, forecast=pricing["result"], attempts=outcomes)
        pricing["result"]["process_overhead_seconds"] = max(0., pricing["supervisor_seconds"]-pricing["wall_seconds"])
        quote = forecast_campaign(config, pricing["result"])
        atomic_json(campaign.root / f"forecast-{method}.json", quote)
        if stop_after == "price":
            return finish("PRICING_COMPLETE", method=method, forecast=quote)
        if not quote["full_campaign_priced"]:
            return finish("PRICING_INCOMPLETE", method=method, forecast=quote, attempts=outcomes)
        reserves = quote["reserves"]
        scopes = ([f"{method}-beta1"] if method == "neutra" else
                  [f"ensemble-chart-{i}-beta{b:g}" for i in range(config["ensemble"]["charts"]) for b in positive])

        def pending(name, reserve):
            if name in campaign.state["stages"]:
                return 0.
            spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == name)
            return max(0., reserve-spent)

        def protected(*, posterior=False, reference=True, assessment=True, tuning_scopes=0):
            # Protect each remaining scope's mandatory initial L cohort, with
            # fresh verification. Unmeasured L uses the next measured length;
            # this is a conservative engineering hypothesis, never a speed claim.
            tuning_floor = 0.
            for length in config["tuning"]["l_grid"]:
                available = [row for row in pricing["result"]["hmc"] if row["L"] >= length]
                bound = min(row["L"] for row in available)
                rows = [row for row in available if row["L"] == bound]
                tuning_floor += config["budget"]["forecast_safety_factor"] * (
                    2*max(row["first_seconds"] for row in rows) +
                    (2*config["tuning"]["startup"]+config["tuning"]["measurement"]+config["tuning"]["verification"])*
                    max(row["steady_seconds"] for row in rows))
            return (repair_remaining(campaign)
                + tuning_scopes*tuning_floor
                + (pending("reference", quote["reference_first_assessment_seconds"]) if reference else 0.)
                + (pending("assess-"+method, reserves["assessment"]) if assessment else 0.)
                + (pending("sample-"+method, quote["posterior_first_assessment_seconds"]) if posterior else 0.))

        training_request = {"stage": "train", "method": method, "stop_when_trial_ready": False,
                            "stop_after_rung": next_training_rung(config, checkpoint, method), **qualified_request}
        if checkpoint is not None:
            training_request["resume_checkpoint"] = checkpoint
        training_budget = allocate_stage(campaign, "train-"+method, quote["training_cap_seconds"],
                                        protected_seconds=protected(posterior=True, tuning_scopes=len(scopes)))
        trained = stage("train-"+method, training_request, reserve=training_budget)
        checkpoint = trained["result"]["checkpoint"]
        if stop_after == "train":
            return finish("TRAINING_STAGE_COMPLETE", method=method, training=trained["result"])
        count = 1 if method == "neutra" else config["ensemble"]["charts"]
        maps = choose_maps(config, checkpoint, schedule=method_schedule(method), count=count)
        members = {}
        repair_cohort = recovery.get("plain_cohort") if method == "neutra" else None
        if repair_cohort and repair_cohort["minimum_seconds"] > campaign.remaining()-protected(posterior=True):
            outcomes.append({"method": method, "status": "repair_unaffordable", "required": repair_cohort,
                             "scientific_rejection": False})
            continue
        tuning_serial = 0
        def tune_item(item, beta, label, completed):
            nonlocal tuning_serial
            # Keep the first legacy name; every alternative/updated map gets a
            # new stage and therefore fresh fixed-map numerical verification.
            suffix = "" if tuning_serial < len(scopes) else "-map-"+digest(item["exports"])[:16]
            tuning_serial += 1
            name = "tune-"+label+suffix
            ceiling = quote["tuning_scope_limit_seconds"] if repair_cohort is None else repair_cohort["maximum_seconds"]
            tuning_budget = allocate_stage(campaign, name, ceiling,
                protected_seconds=protected(posterior=True, tuning_scopes=len(scopes)-completed-1))
            tuned = stage(name, {"stage": "tune", "method": "neutra", "beta": beta,
                "training_export": item["exports"][str(beta)], "start_label": method+"-tuning-starts",
                **({"explicit_cohort": repair_cohort["proposal"]} if repair_cohort else {}),
                **qualified_request}, reserve=tuning_budget)
            if tuned["result"]["status"] in {"partial_budget", "budget_bound", "budget_paused"}:
                raise StageBudgetPause("tuning allocation exhausted: " + label)
            if tuned["result"]["status"] == "shared_invalidity":
                raise ValueError("shared tuning invalidity: " + label)
            if tuned["result"]["status"] != "complete":
                raise StageIncomplete("tuning incomplete: " + label)
            return selected_member(tuned["result"])

        while True:
            maps, members, failures = tune_map_cohort(config, checkpoint, method, tune_item)
            if members:
                break
            rung = next_training_rung(config, checkpoint, method)
            if rung is None:
                break
            name = f"train-{method}-repair-u{rung}"
            reserve = allocate_stage(campaign, name, quote["training_cap_seconds"],
                protected_seconds=protected(posterior=True, tuning_scopes=len(scopes)))
            repaired = stage(name, {"stage": "train", "method": method,
                "resume_checkpoint": checkpoint, "stop_when_trial_ready": False,
                "stop_after_rung": rung, **qualified_request}, reserve=reserve)
            checkpoint = repaired["result"]["checkpoint"]
            atomic_json(campaign.root / f"training-repair-{method}-u{rung}.json",
                        {"trigger": failures, "checkpoint": checkpoint, "rung": rung})
        if not members:
            outcomes.append({"method": method, "status": "tuning_candidate_failed" if failures else "training_candidate_failed",
                             "map_failures": failures, "checkpoint": checkpoint})
            continue
        frozen = {"schema": "bayesfilter.q20.frozen_estimation_procedure.v1", "config_hash": digest(config),
                  "method": method, "maps": maps, "members": members, "mass": "identity_in_frozen_transport_coordinates",
                  "selection_policy": "eligible_declared_cohort_order_then_verified_id; no method ranking"}
        frozen_path = campaign.root / f"frozen-procedure-{method}.json"
        if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
            raise ValueError("frozen procedure changed on resume")
        atomic_json(frozen_path, frozen)
        sample = {"label": "estimation-"+method, "start_label": method+"-posterior-starts", **qualified_request}
        if method == "neutra":
            sample.update(stage="sample", member_path=members["neutra-beta1"]["path"])
        else:
            sample.update(stage="ensemble", members_by_beta={str(beta):
                [members[f"ensemble-chart-{i}-beta{beta:g}"]["path"] for i in range(count)] for beta in positive})
        if recovery.get("exact_selected_pricing"):
            selected = stage("price-selected-"+method, {**sample, "stage": "price-selected", "method": method}, diagnostic=True)
            observation = selected.get("terminal_resource", {}).get("quality")
            if observation not in {"no_contention_observed", "cpu_reference_gpu_hidden"}:
                raise StageIncomplete("selected procedure timing requires an uncontended remeasurement")
            selected["result"]["process_overhead_seconds"] = max(0., selected["supervisor_seconds"]-selected["wall_seconds"])
            exact = selected_procedure_forecast(config, pricing["result"], selected["result"], members)
            atomic_json(campaign.root / f"forecast-selected-{method}.json", exact)
            if exact["first_assessment_seconds"] > campaign.remaining()-protected():
                raise StageBudgetPause("actual verified procedure first assessment is unaffordable: " + method)
            sample["chunk_reserve_seconds"] = exact["chunk_seconds"]
            reserves["posterior"] = exact["maximum_seconds"]
        else:
            sample["chunk_reserve_seconds"] = posterior_chunk_forecast(config, pricing["result"], members)
        sample_budget = allocate_stage(campaign, "sample-"+method, reserves["posterior"],
                                      protected_seconds=protected())
        sampled = stage("sample-"+method, sample, reserve=sample_budget)
        if sampled["result"].get("status") == "budget_paused":
            raise StageBudgetPause("sampling allocation exhausted: " + method)
        if not sampled["result"]["summary"]["sequential_declared_checks_passed"]:
            outcomes.append({"method": method, "status": "posterior_candidate_failed", "posterior_path": sampled["result_path"]})
            continue
        reference_budget = allocate_stage(campaign, "reference", reserves["reference"],
                                         protected_seconds=protected(reference=False))
        reference = stage("reference", {"stage": "reference",
            "chunk_reserve_seconds": config["budget"]["forecast_safety_factor"] * (
                pricing["result"]["reference_batch_first_seconds"]+pricing["result"]["reference_batch_steady_seconds"])},
            reserve=reference_budget)
        assessment_budget = allocate_stage(campaign, "assess-"+method, reserves["assessment"],
                                          protected_seconds=protected(reference=False, assessment=False))
        assessed = stage("assess-"+method, {"stage": "assess", "method": method,
            "reference_path": reference["result_path"], "posterior_path": sampled["result_path"]}, reserve=assessment_budget)
        outcomes.append({"method": method, "status": assessed["result"]["status"], "assessment_path": assessed["result_path"]})
        if assessed["result"]["passed"]:
            return finish("SMOKE_ESTIMATION_CHECKS_PASSED" if config["role"] == "smoke" else "STABLE_ESTIMATE_OBTAINED",
                estimation_passed=True, method=method, posterior_path=sampled["result_path"],
                assessment=assessed["result"], attempts=outcomes)
        if assessed["result"]["status"] == "reference_unqualified":
            return finish("REFERENCE_REPAIR_REQUIRED", method=method, assessment=assessed["result"], attempts=outcomes)
    return finish("ESTIMATION_UNDER_BUDGETED" if all(row["status"] == "under_budgeted" for row in outcomes)
                  else "ESTIMATION_REPAIR_REQUIRED", attempts=outcomes,
                  next_action="repair the recorded numerical, training, sampling or cost failure within the remaining allowance")


def execute_master(config, root, *, repo, allowance=None, fixture=False, stop_after=None,
                   training_checkpoint=None, previous_source_root=None, previous_campaign=None,
                   previous_config=None):
    validate_protocol(config)
    if config["role"] != "smoke" and stop_after in {"refresh", "repair", "checkpoint-repair"}:
        raise ValueError("legacy classical-preparation phases are retired; use the NeuTra estimation master")
    campaign=Campaign(root,repo=repo,config=config,allowance=allowance)
    with campaign.locked():
        if previous_config is not None:
            if training_checkpoint is None or previous_source_root is None:
                raise ValueError("migration requires the historical checkpoint and preserved source directory")
            from bayesfilter.inference.q20_checkpoint_migration import audit_migration
            audit = audit_migration(config, previous_config, training_checkpoint, previous_source_root, campaign.repo)
            request = {"stage": "migrate-training", "previous_config": previous_config,
                "training_checkpoint": str(Path(training_checkpoint).resolve()),
                "previous_source_root": str(Path(previous_source_root).resolve()),
                "previous_sha256": audit["previous_sha256"]}
            if campaign.state.get("training_migration_request", request) != request:
                raise ValueError("training migration inputs changed on resume")
            campaign.state["training_migration_request"] = request
            campaign.save()
        elif training_checkpoint is not None and stop_after != "refresh":
            from bayesfilter.inference.q20_training_resume import import_training_checkpoint, checksum
            saved = campaign.state.get("training_resume")
            if saved is None:
                if previous_source_root is None:
                    raise ValueError("checkpoint import requires its preserved source directory")
                saved = import_training_checkpoint(training_checkpoint, config, campaign.root / "training-import",
                    previous_root=previous_source_root, current_root=campaign.repo)
                campaign.state["training_resume"] = saved
                campaign.save()
            elif (saved["source_import"]["previous_checkpoint"] != str(Path(training_checkpoint).resolve())
                  or checksum(saved["checkpoint"]) != saved["sha256"]):
                raise ValueError("imported training checkpoint changed")
        if stop_after == "cache-qualify":
            from bayesfilter.inference.q20_master_profile import qualify_profile_candidate
            return qualify_profile_candidate(campaign, previous_campaign=previous_campaign,
                previous_source_root=previous_source_root)
        if stop_after == "profile":
            from bayesfilter.inference.q20_master_profile import profile_master
            return profile_master(campaign, previous_campaign=previous_campaign,
                previous_source_root=previous_source_root)
        if stop_after == "checkpoint-repair":
            from bayesfilter.inference.q20_master_checkpoint import checkpoint_master
            return checkpoint_master(campaign, previous_campaign=previous_campaign,
                previous_source_root=previous_source_root, fixture=fixture)
        if stop_after == "repair":
            from bayesfilter.inference.q20_master_repair import repair_master
            return repair_master(campaign, previous_campaign=previous_campaign,
                previous_source_root=previous_source_root, fixture=fixture)
        if stop_after == "refresh":
            from bayesfilter.inference.q20_master_refresh import refresh_master
            return refresh_master(campaign, checkpoint=training_checkpoint,
                previous_source_root=previous_source_root, fixture=fixture)
        return _execute(campaign,fixture=fixture,stop_after=stop_after)


def _execute(campaign, *, fixture=False,stop_after=None):
    config=campaign.config
    diagnostic_cap=config["execution"]["diagnostic_attempt_seconds"]
    gpu="auto" if stop_after == "diagnose" else None
    def finish(status, **details):
        campaign.state.update(status=status,details=details,production_qualified=False)
        campaign.save()
        result={"status":status,"details":details,"campaign_root":str(campaign.root),
            "remaining_campaign_seconds":campaign.remaining(),"remaining_diagnostic_seconds":campaign.remaining(True),
            "declared_estimation_checks_passed":bool(details.get("estimation_passed")),
            "production_qualified":False,
            "method_ranking":"not_estimated"}
        atomic_json(campaign.root/"result.json",result)
        return result
    if stop_after != "diagnose" and (config["role"]!="smoke" or not config["cpu_reference"]):
        if min(campaign.remaining(True),campaign.stage_remaining("gpu-readiness"))<=config["execution"]["termination_grace_seconds"]:
            return finish("BUDGET_EXHAUSTED_BEFORE_GPU_PROBE")
        probe=campaign.execute("gpu-readiness",[sys.executable,"-m",
            "bayesfilter.inference.q20_gpu_runtime","--gpu","auto"],
            cap_seconds=min(diagnostic_cap,campaign.remaining(True)),diagnostic=True,
            environment={"BAYESFILTER_PRELOAD_CUSTOM_OP":"0"})
        log=Path(probe["directory"])/"console.log"
        try:
            receipt=json.loads(log.read_text())
        except (ValueError,OSError):
            return finish("GPU_PROBE_FAILED",attempt=probe)
        if probe["status"]!="completed" or receipt.get("status")!="PASSED":
            waiting = bool(receipt.get("resource_unavailable"))
            return finish("WAITING_FOR_GPU" if waiting else "GPU_PROBE_FAILED",probe=receipt,attempt=probe)
        gpu="auto"  # Each worker selects again after any scheduling delay.
    def stage(name, request, *, diagnostic=False, reserve=None):
        base={**request,"fixture":fixture}
        if request["stage"] == "train":
            base.pop("cooperative_seconds",None)
        if name in campaign.state["stages"]:
            return campaign.numerical_stage(name,base,cap_seconds=0.,diagnostic=diagnostic,gpu=gpu)
        # Quote-derived cumulative stage limits replace the old uniform 8-hour
        # cap. The unchanged campaign allowance still bounds every launch.
        if reserve is not None and name not in campaign.state.setdefault("stage_limits", {}):
            if not math.isfinite(reserve) or reserve <= 0:
                raise ValueError("stage reserve must be finite and positive")
            campaign.state["stage_limits"][name] = reserve
            campaign.save()
        spent=sum(a.get("elapsed_seconds",0.) for a in campaign.state["attempts"] if a["stage"]==name)
        cap=min(campaign.stage_remaining(name),campaign.remaining(diagnostic),
                diagnostic_cap if diagnostic else max(0.,reserve-spent) if reserve is not None else config["budget"]["arm_cap_seconds"])
        if cap<=config["execution"]["termination_grace_seconds"]:
            raise StageBudgetPause("budget_exhausted:"+name)
        attempts=[a for a in campaign.state["attempts"] if a["stage"]==name
                  and a.get("failure_classification") != "resource_unavailable"]
        if name not in campaign.state["stages"] and len(attempts)>=config["execution"]["stage_attempts"]:
            raise StageIncomplete("attempt_limit:"+name)
        diagnostic_attempts=[a for a in campaign.state["attempts"] if a["diagnostic"] and a["stage"]!="gpu-readiness"
                             and a.get("failure_classification") != "resource_unavailable"]
        if diagnostic and name not in campaign.state["stages"] and len(diagnostic_attempts)>=config["execution"]["diagnostic_attempts"]:
            raise StageIncomplete("diagnostic_attempt_limit:"+name)
        # Training leaves time for its final receipt; the external supervisor
        # still owns the deadline, including worker initialization.
        if request["stage"] == "train":
            base.pop("cooperative_seconds",None)
        result=campaign.numerical_stage(name,base,cap_seconds=cap,diagnostic=diagnostic,gpu=gpu)
        if result.get("status") == "waiting_for_gpu":
            raise ResourceWait(result)
        if not result.get("completed"):
            if result.get("budget_paused"):
                raise StageBudgetPause(name + ": " + result["status"])
            # Retry only interrupted/checkpointed computation with unchanged
            # inputs. Programming and candidate failures need a recorded repair.
            retryable=(result["status"] in {"partial_training_checkpointed","partial_tuning_checkpointed"} or
                       result.get("attempt",{}).get("status") in {"timed_out","interrupted"})
            if retryable and len(attempts)+1<config["execution"]["stage_attempts"] and campaign.stage_remaining(name)>config["execution"]["termination_grace_seconds"]:
                return stage(name,request,diagnostic=diagnostic,reserve=reserve)
            raise StageIncomplete(result["status"])
        return result
    try:
        if campaign.state.get("training_migration_request"):
            migrated = stage("migrate-training", campaign.state["training_migration_request"], diagnostic=True)
            campaign.state["training_resume"] = migrated["result"]
            campaign.save()
        if stop_after == "migrate":
            if not campaign.state.get("training_resume"):
                raise ValueError("migrate requires a saved training checkpoint")
            return finish("TRAINING_MIGRATION_COMPLETE", training_resume=campaign.state["training_resume"])
        if stop_after == "diagnose":
            compared = stage("status-reuse", {"stage": "status-reuse"}, diagnostic=True)
            return finish("STATUS_REUSE_DIAGNOSTIC_COMPLETE", diagnostic=compared["result"],
                next_action="price and run the declared training calibration; full campaign remains unpriced")
        if stop_after == "calibrate":
            from bayesfilter.inference.q20_campaign_costs import training_reservation
            priced = stage("price-training", {"stage": "price-training",
                "reservation_limit_seconds": campaign.state["campaign_limit"], "method": "neutra"}, diagnostic=True)
            quote = training_reservation(config, priced["result"]["rows"], method="neutra")
            startup = priced["result"]["worker_initialization_seconds"] + max(
                0., priced["supervisor_seconds"] - priced["wall_seconds"])
            reserve = quote["calibration_seconds"] + config["budget"]["forecast_safety_factor"] * (
                startup + 2 * config["execution"]["termination_grace_seconds"])
            atomic_json(campaign.root / "training-forecast.json", {**quote,
                "calibration_with_worker_overhead_seconds": reserve})
            if quote["missing_calibration_scopes"] or reserve > campaign.remaining():
                return finish("CALIBRATION_UNDER_BUDGETED", forecast=quote,
                              required_seconds=reserve, full_campaign_priced=False)
            calibrated = stage("calibration", {"stage": "train", "calibration_only": True, "method": "neutra"}, reserve=reserve)
            return finish("TRAINING_SANITY_PILOT_COMPLETE", training=calibrated["result"],
                calibration_complete=False,
                forecast=quote, next_action="inspect learning and variance before extending the same full protocol",
                full_campaign_priced=False)
        return run_estimation_attempts(campaign, stage, finish, stop_after=stop_after)
    except KeyboardInterrupt:
        finish("MASTER_INTERRUPTED", reason="owned worker settled; resume or repair within remaining allowance")
        raise
    except ResourceWait as error:
        return finish("WAITING_FOR_GPU", resource=error.result,
                      next_action="resume the same master command when a policy-permitted GPU is idle")
    except StageBudgetPause as error:
        return finish("ESTIMATION_BUDGET_PAUSED", reason=str(error),
            next_action="inspect saved progress and remaining allocations; no method was rejected by this resource stop")
    except (StageIncomplete,CampaignBudgetError) as error:
        return finish("MASTER_INCOMPLETE",reason=str(error))
    except Exception as error:
        finish("MASTER_INFRASTRUCTURE_FAILURE",type=type(error).__name__,reason=str(error))
        raise
