"""Checked LG2 preparation reporting; this module grants no inference authority."""
import hashlib
import json
import math
from pathlib import Path


INPUT_NAMES = frozenset(("calibration_data", "claim", "consumer_archived_selection_source",
    "consumer_manifest", "consumer_report", "controls_manifest", "controls_report", "reference",
    "reference_manifest", "scope", "selection", "validation_data"))
CHECK_NAMES = frozenset(("actual_claim_probes", "bitwise_replay", "canonical_call_chain", "permutation",
    "row_independence", "endpoint_independent", "field_sign_status", "endpoint_sign", "reference_force_sign",
    "proposal_reversible", "predictive_parity_0", "predictive_parity_4", "predictive_parity_8",
    "complete_transitions", "retained_diagnostics"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lg2_family_budget(root):
    """Return cumulative family balance separately from the one-attempt reservation."""
    family = dict(cpu_worker_seconds=1800., gpu_seconds=7200.)
    attempts = 0
    scripts = {"prepare_surrogate_hmc_linear_reference.py", "qualify_surrogate_hmc_linear_consumer.py",
               "calibrate_surrogate_hmc_linear_controls.py", "qualify_surrogate_hmc_linear_endpoint.py"}
    def is_lg2_job(job):
        command = job.get("command", [])
        names = {Path(arg).name for arg in command} & scripts
        if not names:
            return False
        if "--model" in command:
            index = command.index("--model")
            return index+1 < len(command) and command[index+1] == "LG2"
        # These CLIs currently implement LG2 only. Even a failed invocation
        # missing its required model flag consumes the family's budget.
        return True
    for path in Path(root).glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        payload = json.loads(path.read_text())
        charge = payload.get("charged_this_run", payload.get("charged", {}))
        if "-lg2-" in path.parent.name:
            amounts = charge
        else:
            jobs = payload.get("jobs", [])
            relevant = [job for job in jobs if is_lg2_job(job)]
            amounts = {key: sum(float(job[key]) for job in relevant) for key in family}
            if relevant:
                amounts["cpu_worker_seconds"] += max(0., float(charge["cpu_worker_seconds"])
                    -sum(float(job["cpu_worker_seconds"]) for job in jobs))
        for key in family:
            value = float(amounts.get(key, 0.))
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid LG2 accounting")
            family[key] -= value
        attempts += "-lg2-endpoint-" in path.parent.name
    return family, attempts


def lg2_endpoint_history(root):
    """Read the latest bounded qualification without presenting it as HMC inference."""
    state = dict(status="NOT_RUN", qualified=False, retained_inference=False,
        surrogate_hmc_tested=False, authorizes_new_inference=False,
        comparisons=dict.fromkeys(("B/R", "C/D", "D/R", "C/R"), "NOT_RUN"))
    paths = sorted(Path(root).glob("docs/plans/artifacts/surrogate-hmc-all-models-*-lg2-endpoint-*/report.json"))
    if not paths:
        return state
    path = paths[-1]
    state.update(report=str(path.resolve()), report_sha256=_sha(path),
        attempt_reports=[str(item.resolve()) for item in paths])
    try:
        report = json.loads(path.read_text())
        run = json.loads((path.parent/"run-manifest.json").read_text())
        if run.get("schema") != "surrogate_hmc_lg2_endpoint_v1":
            raise ValueError("wrong LG2 endpoint scope")
        if "exit_code" not in run:
            state["status"] = "RUNNING"
            return state
        if run["exit_code"] != 0:
            state.update(status="ENDPOINT_QUALIFICATION_FAILED", exit_code=run["exit_code"])
            return state
        accounting = json.loads((path.parent/"accounting.json").read_text())
        if (report.get("status") != "QUALIFIED_INFERENCE_ENDPOINT" or report.get("qualified") is not True
                or report.get("retained_inference") is not False or report.get("surrogate_hmc_tested") is not False
                or report.get("posterior_comparisons") != state["comparisons"]
                or not CHECK_NAMES.issubset(report["checks"])
                or not all(value is True for value in report["checks"].values())
                or run.get("source_drift") or accounting.get("exit_code") != 0
                or set(run["inputs"]) != INPUT_NAMES or not run.get("source_hashes")):
            raise ValueError("invalid LG2 endpoint qualification")
        for name, digest in run["source_hashes"].items():
            if _sha(path.parent/"source"/name) != digest:
                raise ValueError("LG2 archived source changed")
            if name.startswith("bayesfilter/") and name not in (
                    "bayesfilter/inference/surrogate_hmc_program.py", "bayesfilter/inference/__init__.py"):
                if _sha(Path(root)/name) != digest:
                    raise ValueError("LG2 numerical source changed: "+name)
        for record in run["inputs"].values():
            if _sha(record["path"]) != record["sha256"]:
                raise ValueError("LG2 frozen input changed")
        from scripts.qualify_surrogate_hmc_linear_endpoint import complete_forecast
        forecast = complete_forecast(report["complete_transition_costs"], report["diagnostic_costs"])
        if report["forecast"] != forecast:
            raise ValueError("LG2 complete cost mismatch")
        family, _ = lg2_family_budget(root)
        fits = (forecast["required"]["cpu_worker_seconds"] <= family["cpu_worker_seconds"]
                and forecast["required"]["wall_seconds"] <= family["gpu_seconds"])
        state.update(status="QUALIFIED_INFERENCE_ENDPOINT", qualified=True, checks=report["checks"],
            complete_forecast=forecast, remaining_family=family,
            forecast_role="bounded_2000_draw_pilot_not_adequate_equivalence_precision",
            precision_status="PROSPECTIVE_FUNCTION_AND_DIAGNOSTIC_VALIDATION_REQUIRED",
            precision_result="docs/plans/surrogate-hmc-inference-precision-result-2026-09-22.md",
            inference_readiness="REQUIRES_INFERENCE_PLAN_AND_PUBLIC_TUNING" if fits else "DEFERRED_COMPLETE_BUDGET",
            next_action="Preserve qualification; justify diagnostic scope and equivalence precision, then fund the full schedule and tune publicly")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        state.update(status="RECONCILIATION_REQUIRED", qualified=False, error=str(exc))
    return state
