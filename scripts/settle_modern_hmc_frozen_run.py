#!/usr/bin/env python3
"""Read-only terminal audit/accounting for the frozen full-model launcher.

No framework imports, numerical replay, candidate selection, launch or stop.
This checks preserved evidence; it does not recompute statistical diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


PLAN = "docs/plans/bayesfilter-modern-hmc-additional-72h-2026-10-16.md"
ACTIVE = {"active", "activating", "deactivating", "reloading"}
TERMINAL = {"completed", "resource_stopped", "supervisor_budget_exhausted",
            "supervisor_interrupted", "repair_required", "worker_interrupted"}
ASSESSED = {"posterior_checks_passed", "posterior_incomplete"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "nonnegative finite seconds required")
    return value


def checked_content(value):
    body = dict(value)
    expected = body.pop("content_hash")
    require(payload_hash(body) == expected, "checkpoint content hash mismatch")


def service_state(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "-p", "ActiveState", "-p", "SubState", "-p", "ExecMainStatus"],
        capture_output=True, text=True, check=True, timeout=10)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def readiness(state, supervisor):
    if state.get("ActiveState") in ACTIVE:
        return "waiting"
    require(state.get("ActiveState") in {"inactive", "failed"}, "unknown service state")
    require(supervisor is not None and supervisor.get("status") in TERMINAL,
            "service exited without a supported terminal supervisor record")
    require(type(supervisor.get("exit_code")) is int, "missing process exit code")
    if supervisor["status"] == "completed":
        require(supervisor["exit_code"] == 0 and state["ActiveState"] == "inactive"
                and state.get("ExecMainStatus") == "0", "completed process/service disagree")
    return "ready"


def inspect_inventory(design, controller, report):
    candidates = controller["candidates"]
    ids = [c["candidate_id"] for c in candidates]
    verified = controller["verified_candidate_ids"]
    require(len(ids) == len(set(ids)) and len(verified) == len(set(verified))
            and set(verified) <= set(ids), "duplicate or unknown candidate")
    require(candidates == sorted(candidates, key=lambda c: c["creation_ordinal"]),
            "candidate creation order changed")
    require([r["candidate_id"] for r in report["rows"]] == ids
            and report["verified_candidate_ids"] == verified, "report omitted/reordered candidates")
    planned = [(length, epsilon) for length, values in design["search"]["epsilon_by_l"]
               for epsilon in values]
    roots, seen, repairs = [], {}, {}
    for candidate in candidates:
        parent = candidate["parent_candidate_id"]
        family = candidate["candidate_family_id"]
        if parent is None:
            roots.append((candidate["leapfrog_steps"], candidate["epsilon"]))
        else:
            require(parent in seen and seen[parent]["candidate_family_id"] == family,
                    "repair ancestry changed")
            repairs[family] = repairs.get(family, 0) + 1
        seen[candidate["candidate_id"]] = candidate
    require(len(set(roots)) == len(roots) and set(roots) <= set(planned), "unexpected roots")
    require(all(count <= design["max_repairs_per_family"] for count in repairs.values()),
            "repair allowance exceeded")
    require(len(ids) <= design["execution"]["acceptance_policy"]["max_candidates"],
            "candidate cap exceeded")
    complete = controller["completion_status"] == "complete"
    missing = [list(pair) for pair in planned if pair not in roots]
    require(not complete or not missing, "complete tuning omitted roots")
    assessed, delivered = [], []
    for row in report["rows"]:
        cid = row["candidate_id"]
        require(type(row.get("delivered", False)) is bool, "delivery flag must be boolean")
        if row["status"] in ASSESSED:
            require(cid in verified and row.get("result_file") and "reference" in row,
                    "assessment lacks verified identity or evidence")
            expected = (row["status"] == "posterior_checks_passed"
                        and not row["posterior_hard_vetoes"] and row["reference"]["passed"])
            require(row.get("delivered", False) == expected, "delivery contradicts assessment")
            if "campaign_resource_cap" not in row["posterior_hard_vetoes"]:
                assessed.append(cid)
        else:
            require(not row.get("delivered", False), "unfinished candidate delivered")
        if row.get("delivered", False):
            delivered.append(cid)
    require(report.get("delivered_count", 0) == len(delivered), "delivery count mismatch")
    unfinished = [cid for cid in verified if cid not in assessed]
    return dict(created_count=len(ids), requested_root_count=len(planned),
        created_root_count=len(roots), repair_descendant_count=len(ids)-len(roots),
        uncreated_primary_pairs=missing, verified_count=len(verified),
        assessed_count=len(assessed), delivered_count=len(delivered),
        failed_assessment_ids=[cid for cid in assessed if cid not in delivered],
        unfinished_verified_ids=unfinished, tuning_complete=complete,
        all_verified_assessed=not unfinished, method_ranking=None, release_qualified=False)


def settle_budget(allocation, supervisor):
    available = number(allocation["available_after_existing_held_local_reserve_seconds"])
    reserved = number(allocation["numerical_cap_seconds"])
    unallocated = number(allocation["unallocated_after_numerical_reservation_seconds"])
    require(math.isclose(available, reserved + unallocated, abs_tol=1e-6, rel_tol=0),
            "prior allocation does not reconcile")
    require(supervisor["max_wall_seconds"] == reserved, "reservation changed")
    spent = number(supervisor["elapsed_seconds"])
    balance = available - spent
    result = dict(numerical_reserved_seconds=reserved, numerical_charged_seconds=spent,
        numerical_overrun_seconds=max(0., spent-reserved),
        signed_balance_after_existing_reserve_seconds=balance,
        unallocated_worker_seconds=max(0., balance), shortfall_seconds=max(0., -balance),
        existing_local_reserve_still_held_seconds=number(allocation["existing_local_reserve_still_held_seconds"]),
        prior_balance_carried_separately_seconds=number(allocation["prior_balance_carried_separately_seconds"]),
        charge_basis="inclusive_supervisor_elapsed_only_child_not_added",
        local_reserve_released=False)
    if "funding_sources" in allocation:
        sources = allocation["funding_sources"]
        require(isinstance(sources, list) and bool(sources), "funding sources must be a nonempty list")
        identifiers, balances = [], []
        for source in sources:
            require(isinstance(source, dict), "invalid funding source")
            identifier = source.get("allocation_id")
            require(isinstance(identifier, str) and bool(identifier.strip()), "funding source needs an identifier")
            identifiers.append(identifier)
            balances.append(number(source["available_seconds"]))
        require(len(set(identifiers)) == len(identifiers), "duplicate funding source")
        require(math.isclose(math.fsum(balances), available, abs_tol=1e-6, rel_tol=0),
                "funding sources do not reconcile")
        result["funding_sources"] = sources
    else:
        # Legacy allocations predate explicit funding provenance and use only
        # the old pool. Never attach this old label to a new combined pool.
        result["additional_72h_grant_included"] = False
    return result


def check_tensors(tree, directory):
    if "tensor" in tree:
        receipt = tree["tensor"]
        require(digest(directory / receipt["file"]) == receipt["sha256"], "tensor checksum mismatch")
        return 1
    if "mapping" in tree:
        return sum(check_tensors(v, directory) for v in tree["mapping"].values())
    if "sequence" in tree:
        return sum(check_tensors(v, directory) for v in tree["sequence"])
    require("scalar" in tree, "unknown checkpoint tree")
    return 0


def check_policy(policy, design, candidate, index, quantities):
    settings = design["posterior"]
    expected = dict(settings["sequential"], step_size=candidate["epsilon"],
        num_leapfrog_steps=candidate["leapfrog_steps"], jit_compile=design["execution"]["use_xla"],
        warmup_seed=[settings["seed"][0], settings["seed"][1]+2*index],
        retained_seed=[settings["seed"][0], settings["seed"][1]+2*index+1])
    # Frozen neutra_hmc._sequential_count_budget_payload omits the inherited
    # 10,000/None budget to preserve historical checkpoint identities.
    # Nondefault budgets must still be serialized and checked explicitly.
    if (expected.get("max_results_per_chain") == 10000
            and expected.get("count_budget_reason") is None
            and "max_results_per_chain" not in policy):
        expected.pop("max_results_per_chain")
    require(all(policy.get(k) == v for k, v in expected.items()), "posterior policy or seed changed")
    assessment = policy["assessment_policy"]
    require(all(assessment.get(k) == v for k, v in settings["assessment"].items()),
            "posterior assessment changed")
    precision = assessment["precision"]
    require(all(precision.get(k) == v for k, v in settings["precision"].items()),
            "precision estimator changed")
    require([t["name"] for t in precision["targets"]] == quantities and
            all(t["kind"] == "mean" and t["mcse_sd_ratio_max"] == settings["mcse_sd_ratio_max"]
                for t in precision["targets"]), "precision target changed")


def audit_posterior(run, manifest, candidates, members, rows):
    summaries, total = [], 0
    for index, (candidate, row) in enumerate(zip(candidates, rows)):
        directory = run / f"posterior/member-{index:03d}"
        commits = sorted((directory / "chunks/committed").glob("chunk-*"))
        require([c.name for c in commits] == [f"chunk-{i:06d}" for i in range(len(commits))],
                "committed checkpoint prefix has a gap")
        summary = dict(candidate_id=row["candidate_id"], status=row["status"],
                       delivered=row.get("delivered", False), committed_chunks=len(commits))
        if not directory.exists():
            require(row["status"] not in ASSESSED, "assessed member directory missing")
            summaries.append(summary)
            continue
        member = members[row["candidate_id"]]
        policy = read(directory / "policy.json")
        check_policy(policy, manifest["design"], candidate, index, manifest["reference"]["quantities"])
        identity = read(directory / "chunks/identity.json")
        require(identity == dict(member_hash=member["member_hash"], policy=policy,
            quantities=manifest["reference"]["quantities"]), "posterior checkpoint identity mismatch")
        for chunk in commits:
            require(digest(chunk/"bundle.json") == (chunk/"bundle.sha256").read_text().strip(),
                    "bundle checksum mismatch")
            bundle = read(chunk/"bundle.json")
            inputs = read(Path(bundle["metadata"]["attempt"])/"call-start.json")["inputs"]
            require(bundle["identity_hash"] == payload_hash(identity)
                and bundle["inputs_hash"] == payload_hash(inputs)
                and inputs["config"] == policy and inputs["binding_hash"] == manifest["binding_hash"]
                and inputs["candidate_record_hash"] == candidate["candidate_record_hash"],
                "committed posterior inputs changed")
            total += check_tensors(bundle["tree"], chunk)
        if row["status"] in ASSESSED:
            require(Path(row["result_file"]).resolve() == (directory/"result.json").resolve(),
                    "posterior result path mismatch")
            result = read(directory/"result.json")
            reference = row["reference"]
            require(result["config"] == policy and result["warmup_excluded_from_posterior"] is True,
                    "result policy or warmup exclusion mismatch")
            require(row["posterior_hard_vetoes"] == result["hard_vetoes"]
                and row["status"] == ("posterior_checks_passed" if result["passed"] else "posterior_incomplete")
                and reference["precision_policy"] == policy["assessment_policy"]["precision"],
                "posterior decision evidence mismatch")
            screen = reference.get("normalized_error_screen")
            if screen is not None:
                settings = manifest["design"]["posterior"]
                require(len(screen) == len(manifest["reference"]["quantities"])
                    and reference["normalized_tolerance"] == settings["reference_normalized_tolerance"]
                    and reference["mcse_multiplier"] == settings["reference_mcse_multiplier"],
                    "reference screen target/threshold mismatch")
                passed = all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <=
                             reference["normalized_tolerance"] for v in screen)
                require(reference["passed"] == passed, "reference decision mismatch")
            else:
                require(reference["passed"] is False, "passing reference lacks screen")
            summary.update(posterior_passed=result["passed"], reference_passed=reference["passed"],
                hard_vetoes=result["hard_vetoes"], retained_results_per_chain=result["retained_results_per_chain"],
                warmup_results_per_chain=result["warmup_results_per_chain"])
        summaries.append(summary)
    return summaries, total


def audit(phase):
    """Call only after supervisor exit; never mutate the phase's numerical files."""
    run = phase / "run"
    launch, allocation = read(phase/"launch.json"), read(phase/"allocation.json")
    manifest, result, supervisor = [read(run/name) for name in
                                  ("manifest.json", "result.json", "supervisor.json")]
    status = supervisor["status"]
    require(status in TERMINAL and manifest["status"] == result["status"] == status,
            "terminal statuses disagree")
    require(type(supervisor["exit_code"]) is int and (status != "completed" or supervisor["exit_code"] == 0),
            "terminal exit code mismatch")
    expected_command = launch["command"][:2] + ["--worker"] + launch["command"][2:]
    expected_command[-1] = str(float(launch["numerical_cap_seconds"]))
    require(supervisor["command"] == expected_command and manifest["command"] == expected_command,
            "worker command changed")
    require(launch["numerical_cap_seconds"] == allocation["numerical_cap_seconds"]
        == manifest["max_wall_seconds"] == supervisor["max_wall_seconds"], "launch ceiling changed")
    for key in ("started_utc", "finished_utc"):
        require(datetime.fromisoformat(manifest[key]).tzinfo is not None, "missing zoned timestamp")
    require(datetime.fromisoformat(manifest["finished_utc"]) >= datetime.fromisoformat(manifest["started_utc"]),
            "timestamps reversed")
    require(manifest["git_commit"] == launch["source_commit"]
        and manifest["binding_hash"] == launch["expected_binding_hash"], "launch identity changed")
    require(manifest["design_sha256"] == launch["design_sha256"] == digest(launch["design_file"])
        and manifest["design"] == read(launch["design_file"]), "frozen design changed")
    require(manifest["target"]["data_sha256"] == launch["expected_data_sha256"]
        and manifest["target"]["geometry_sha256"] == launch["expected_geometry_sha256"], "target changed")
    memory = manifest["memory_policy"]
    require(memory["all_physical_devices_memory_growth"] is True
        and memory["configured_before_logical_device_initialization"] is True
        and manifest["use_xla"] is True and manifest["cpu_reference"] is False
        and manifest["tf32_enabled"] is True
        and manifest["cuda_visible_devices"] == launch["environment"]["CUDA_VISIBLE_DEVICES"],
        "GPU execution policy changed")
    preserved = {str(p.resolve()): digest(p) for p in run.rglob("*") if p.is_file()}
    for p in (phase/"launch.json", phase/"allocation.json"):
        preserved[str(p.resolve())] = digest(p)
    for path, expected in manifest["source_and_input_sha256"].items():
        require(digest(path) == expected, "source/input changed: " + path)
        preserved[path] = expected
    tuning = run/"search/tuning"
    checkpoint = read(tuning/"tuning_checkpoint.json")
    checked_content(checkpoint)
    controller = checkpoint["result"]
    execution = read(tuning/"execution_spec.json")
    require(checkpoint["binding_hash"] == execution["binding_hash"] == manifest["binding_hash"]
        == payload_hash(execution["execution"]), "tuning binding mismatch")
    evidence_files = {}
    for kind in ("numerical_chunks", "numerical_evidence"):
        files = list((tuning/kind).glob("*.json"))
        evidence_files[kind] = {p.stem for p in files}
        for path in files:
            body = read(path)
            require(payload_hash(body) == path.stem and body["binding_hash"] == manifest["binding_hash"],
                    "numerical evidence hash/binding mismatch")
    require(set(checkpoint["numerical_evidence_hashes"]) <= evidence_files["numerical_evidence"],
            "committed tuning evidence missing")
    for hashes in checkpoint["partial_chunks"].values():
        require(set(hashes) <= evidence_files["numerical_chunks"], "partial tuning chunks missing")
    members = {}
    for path in (run/"search/members").glob("*-member.json"):
        member = read(path)
        checked_content(member)
        bundle = read(path.parent/member["evidence_bundle"]["path"])
        checked_content(bundle)
        require(bundle["content_hash"] == member["evidence_bundle"]["content_hash"], "member bundle mismatch")
        require(member["candidate_id"] not in members, "duplicate retained member")
        members[member["candidate_id"]] = member
    require(set(members) <= set(controller["verified_candidate_ids"]), "unverified member export")
    if controller["completion_status"] == "complete":
        require(set(members) == set(controller["verified_candidate_ids"]), "verified export missing")
    posterior_file = run/"posterior/result.json"
    report = read(posterior_file) if posterior_file.exists() else result
    if posterior_file.exists():
        require(result["posterior"] == report and report["tuning_membership_unchanged"] is True,
                "final/posterior report mismatch")
    disposition = inspect_inventory(manifest["design"], controller, report)
    require(status != "completed" or disposition["tuning_complete"] and disposition["all_verified_assessed"],
            "completed attempt has unfinished work")
    rows, tensors = audit_posterior(run, manifest, controller["candidates"], members, report["rows"])
    for path, expected in preserved.items():
        require(digest(path) == expected, "terminal evidence changed during audit: " + path)
    settlement = settle_budget(allocation, supervisor)
    settlement.update(status="settled", cohort_status=status, disposition=disposition,
        evidence_audit_passed=True, source_commit=manifest["git_commit"], rows=rows,
        source_input_hashes_checked=len(manifest["source_and_input_sha256"]),
        numerical_chunk_count=len(evidence_files["numerical_chunks"]), checked_tensor_files=tensors,
        continuation_route=("inspect_infrastructure_failure" if status in {"repair_required", "worker_interrupted",
            "supervisor_interrupted"} else "qualification_recovery_requires_validation" if not disposition["tuning_complete"]
            else "posterior_continuation" if not disposition["all_verified_assessed"] else "next_frozen_comparator"))
    return settlement, preserved


def write(path, body):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--max-wait-seconds", type=float, required=True)
    args = parser.parse_args()
    require(number(args.max_wait_seconds) > 0, "positive bounded observer lifetime required")
    phase = args.phase.resolve()
    plan = read(phase/"allocation.json").get("phase_plan", PLAN)
    # One deterministic destination per attempt avoids a second settlement.
    output = phase/"terminal-audit-01"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(status="waiting", plan=plan, command=[sys.executable, *sys.argv],
        driver_sha256=digest(__file__), numerical_launch_authority=False,
        numerical_stop_authority=False, gpu_devices_intentionally_hidden=True,
        active_inspection_seconds=0., started_utc=datetime.now(timezone.utc).isoformat(),
        max_wait_seconds=args.max_wait_seconds, poll_seconds=60.)
    try:
        while True:
            tick = time.monotonic()
            state = service_state(args.service)
            path = phase/"run/supervisor.json"
            supervisor = read(path) if path.exists() else None
            ready = readiness(state, supervisor)
            record.update(service_state=state, last_checked_utc=datetime.now(timezone.utc).isoformat())
            if ready == "ready":
                settlement, preserved = audit(phase)
                require(readiness(service_state(args.service), supervisor) == "ready", "service restarted during audit")
                settlement.update(captured_utc=datetime.now(timezone.utc).isoformat(), plan=plan,
                    command=record["command"], driver_sha256=record["driver_sha256"],
                    role="CPU artifact audit/accounting; no statistical recomputation", attempt=str(phase))
                write(output/"preserved-files.json", preserved)
                write(output/"settlement.json", settlement)
                record.update(status="closed_out", continuation_route=settlement["continuation_route"],
                              settlement_file=str(output/"settlement.json"))
            record["active_inspection_seconds"] += time.monotonic()-tick
            record["elapsed_seconds"] = time.monotonic()-started
            if record["status"] == "closed_out":
                break
            if record["elapsed_seconds"] >= args.max_wait_seconds:
                record["status"] = "observation_limit_requires_inspection"
                break
            write(output/"observer.json", record)
            time.sleep(min(60., args.max_wait_seconds-record["elapsed_seconds"]))
    except BaseException as exc:
        record.update(status="closeout_requires_inspection", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = time.monotonic()-started
        write(output/"observer.json", record)


if __name__ == "__main__":
    main()
