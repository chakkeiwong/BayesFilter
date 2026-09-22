"""Bounded fresh calibration; numerical work uses repository TF/TFP consumers."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PLAN = "docs/plans/younis-score-iapf-fit-calibration-fresh-2026-09-17.md"
SHAPE_PLAN = "docs/plans/younis-score-iapf-relative-shape-repair-2026-09-17.md"
LIMIT = 280
REGIMES = {"weak": (.12, .04, 0), "curved": (.35, .12, 1)}


class CampaignStop(SystemExit):
    """Stop before further numerical work when an execution veto fires."""


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def studies(regime, protocol="density"):
    if protocol not in ("density", "relative_shape"):
        raise ValueError("unknown calibration protocol")
    c, b, offset = REGIMES[regime]
    theta = [.62, -.8, -.6, .9, .25, -.3]
    settings = dict(dimension=1, observation_dimension=1, horizon=2, particles=16,
        dtype="float32", device="GPU", tf32=True, jit_compile=True, theta=theta,
        data_theta=theta, transition_curve=c, observation_curve=b,
        prepared_data_regime="nonlinear_iapf_fresh_20260917_" + regime,
        reference=dict(points=401, radius=9., relative_tolerance=1e-7, tail_tolerance=1e-9))
    baseline = dict(k=1, tau=100., max_iterations=4, max_particles=128,
        mean_bound=4., sd_lower=.2, sd_upper=4., max_fit_steps=2000,
        max_backtracks=30, fit_tolerance=1e-7, floor_ratio=.01,
        fit_theta=theta, fit_dtype="float64")
    wider = {**baseline, "tau":25., "mean_bound":8., "sd_lower":.1,
        "sd_upper":8., "max_fit_steps":4000, "max_backtracks":40,
        "fit_tolerance":1e-8, "floor_ratio":.005}
    family = [{"iapf": baseline}, {"iapf": wider},
              {"iapf": {**wider, "k":2, "tau":10.}}]
    partitions = {"calibration":[1100+offset, 1102+offset],
        "validation":[1110+offset, 1112+offset], "claim":[1120+offset, 1122+offset, 1124+offset]}
    if protocol == "relative_shape":
        family = [{"iapf": {**baseline, "fit_objective":"relative_shape"}}]
        partitions = {"calibration":[1200+offset, 1202+offset],
            "validation":[1210+offset, 1212+offset], "claim":[1220+offset]}
        settings["prepared_data_regime"] = "nonlinear_iapf_shape_repair_20260917_" + regime
    base = dict(model="nonlinear_scalar", proposal="iapf", estimator="nonlinear_analytical",
        comparison_target="model_score", comparison="approximation_error",
        replicate=0, coupling_group="baseline")
    template = dict(schema="younis_score_study_v1", phase="0F", version=1,
        plan=SHAPE_PLAN if protocol=="relative_shape" else PLAN,
        evidence_class="target_specific_calibration", seed=1000,
        settings=settings, required_proposals=["iapf"], partitions=partitions,
        tuning_candidate_family=family,
        budget=dict(wall_seconds=2900, max_attempts=24, max_attempts_per_row=1))
    source = deepcopy(template)
    source["rows"] = [dict(base, **candidate, id=f"candidate{i}-{role}-{dataset}",
        role=role, dataset=dataset) for i, candidate in enumerate(family)
        for role in ("calibration", "validation") for dataset in partitions[role]]
    frozen = deepcopy(template)
    frozen["tuning_candidate_family"] = [{"iapf":baseline}]
    frozen["rows"] = [dict(base, iapf=baseline, id=f"baseline-{role}-{dataset}",
        role=role, dataset=dataset) for role in ("calibration", "validation")
        for dataset in partitions[role]]
    heuristic = deepcopy(template)
    heuristic["evidence_class"] = "mechanics"
    heuristic.pop("tuning_candidate_family")
    heuristic["required_proposals"] = ["ekf", "ukf", "bootstrap", "local_linear"]
    heuristic["rows"] = [dict(base, id=f"{name}-{dataset}", proposal=name,
        role="mechanics", dataset=dataset) for dataset in partitions["claim"]
        for name in heuristic["required_proposals"]]
    return source, frozen, heuristic


def claims(source, selected, selection_path, arm):
    result = deepcopy(source)
    base = source["rows"][0]
    result["rows"] = [dict(base, **selected, id=f"{arm}-{dataset}-{rep}",
        role="claim", dataset=dataset, replicate=rep, tuning_selection=str(selection_path))
        for dataset in source["partitions"]["claim"] for rep in range(2)]
    return result


def selection_or_blocked(state, directory, destination, issuer):
    if state["execution_status"] != "complete":
        return {"status":"blocked", "reason":"source study contains failed or missing rows",
            "rows":{k:v["execution_status"] for k,v in state["rows"].items()
                    if v["execution_status"] != "complete"}}
    return {"status":"issued", "artifact":str(destination),
            "selection":issuer(directory, destination)}


class Budget:
    def __init__(self, save, limit=LIMIT, prior=0, wall_limit=2950., cpu_limit=6900.):
        self.save, self.limit, self.prior = save, limit, prior
        self.wall_limit, self.cpu_limit = wall_limit, cpu_limit
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.charges, self.failures = [], []

    @property
    def used(self):
        return self.prior + sum(x["charges"] for x in self.charges)

    def wrap(self, endpoint, stage):
        from bayesfilter.score_study.contracts import DiagnosticFailure
        def bounded(row, context):
            fits = row["iapf"]["max_iterations"] - 1 if row["proposal"] == "iapf" else 0
            if self.used + 1 + fits > self.limit:
                raise CampaignStop("charged attempt budget exhausted before row")
            if time.monotonic()-self.started >= self.wall_limit:
                raise CampaignStop("wall budget exhausted before row")
            if time.process_time()-self.cpu_started >= self.cpu_limit:
                raise CampaignStop("CPU budget exhausted before row")
            basis = "conservative_failed_row_upper_bound"
            try:
                result = endpoint(row, context)
                if row["proposal"] == "iapf":
                    measured = result["diagnostics"]["work_accounting"]["offline_fit_calls"]
                    if type(measured) is not int or not 0 <= measured <= fits:
                        raise ValueError("invalid fit work accounting")
                    fits = measured
                basis = "observed_fit_calls"
                return result
            except DiagnosticFailure as error:
                if row["proposal"] != "iapf":
                    raise CampaignStop("unexpected diagnostic failure outside iAPF") from error
                self.failures.append({"stage":stage, "row":row, "reason":str(error),
                    "classification":"candidate_numerical_veto", "diagnostics":error.diagnostics})
                raise
            except Exception as error:
                raise CampaignStop(f"infrastructure/validity failure: {type(error).__name__}: {error}") from error
            finally:
                self.charges.append({"stage":stage, "row":row["id"],
                    "charges":1+fits, "row_attempts":1, "recursive_fits":fits, "basis":basis})
                self.save()
        return bounded


def fit_summary(diagnostics):
    steps = [step for rec in diagnostics.get("fit_iterations", [])
             for step in rec.get("density_fit_diagnostics", [])]
    return {"underflow_steps":sum(bool(s[9]) for s in steps),
        "boundary_steps":sum(bool(s[3]) for s in steps),
        "max_shape_residual":max((s[1] for s in steps), default=None)}


def conditional_tables(rows, selections, protocol="density"):
    tables = []
    for regime in REGIMES:
        _, _, heuristic = studies(regime, protocol)
        datasets = []
        for dataset in heuristic["partitions"]["claim"]:
            subset = [r for r in rows if r["regime"] == regime and r["dataset"] == dataset]
            errors = {r["proposal"]:r["score_squared_error"] for r in subset if r["arm"] == "heuristic"}
            entry = {"dataset":dataset, "heuristics":errors,
                     "heuristic_table_complete":set(errors)==set(heuristic["required_proposals"])}
            for arm in ("selected", "baseline"):
                values = [r for r in subset if r["arm"] == arm]
                mse = sum(r["score_squared_error"] for r in values)/2 if len(values)==2 else None
                entry[arm] = {"complete":len(values)==2, "mse":mse,
                    "actual_particles":[r["actual_particles"] for r in values],
                    "boundary_veto":any(r["fit_summary"]["boundary_steps"] for r in values),
                    "observed_losses_to":[k for k,v in errors.items() if mse is not None and v < mse]}
                entry[arm]["heuristic_dominance_verdict"] = (
                    "not_evaluable" if mse is None or not entry["heuristic_table_complete"]
                    else "promotion_veto_observed_loss" if entry[arm]["observed_losses_to"]
                    else "passed_descriptive_screen_only")
            datasets.append(entry)
        tables.append({"regime":regime, "datasets":datasets,
            "selection_status":{arm:selections.get(f"{regime}-{arm}",{}).get("status", "not_run")
                                for arm in ("selected", "baseline")},
            "statistically_supported_ranking":False})
    return tables


def preflight(output, protocol="density"):
    from bayesfilter.score_study.contracts import validate_study
    from bayesfilter.score_study.registry import default_registry
    from bayesfilter.score_study.coordinator import fingerprint
    registry = default_registry()
    rows, upper = 0, 0
    all_data = set()
    for regime in REGIMES:
        source, frozen, heuristic = studies(regime, protocol)
        specs = [source, frozen, heuristic,
            claims(source, source["tuning_candidate_family"][0], Path("/preflight/selected.json"), "selected"),
            claims(frozen, frozen["tuning_candidate_family"][0], Path("/preflight/baseline.json"), "baseline")]
        for spec in specs:
            decisions = validate_study(spec, registry)
            if any(d["status"] != "runnable" for d in decisions):
                raise ValueError(f"preflight non-runnable rows: {decisions}")
            rows += len(spec["rows"])
            upper += sum(r["iapf"]["max_iterations"] if r["proposal"]=="iapf" else 1 for r in spec["rows"])
        parts = source["partitions"]
        ids = [x for values in parts.values() for x in values]
        if len(ids) != len(set(ids)) or all_data.intersection(ids):
            raise ValueError("overlapping fresh partitions")
        all_data.update(ids)
        fingerprint(source, registry)
    expected_upper, expected_rows, reserved, prior = ((104,32,5,171) if protocol=="relative_shape"
                                                    else (248,80,32,0))
    if upper != expected_upper or rows != expected_rows or upper+reserved+prior != LIMIT:
        raise ValueError("matrix or conservative cost changed")
    output.mkdir(parents=True, exist_ok=False)
    write(output/"preflight.json", {"status":"pass", "rows":rows, "upper_charges":upper,
        "reserved_charges":reserved, "prior_charges":prior, "protocol":protocol,
        "limit":LIMIT, "datasets":sorted(all_data),
        "numerical_work":False, "gpu_intentionally_hidden":os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"})


def run(args):
    from bayesfilter.score_study.coordinator import execute, fingerprint, load_endpoint
    from bayesfilter.score_study.contracts import digest, validate_result
    from bayesfilter.score_study.registry import default_registry
    from bayesfilter.score_study.tuning import issue_selection
    from bayesfilter.score_study.runtime import configure_runtime, memory_usage
    root = args.output.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if revision != args.source_revision or subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True):
        raise ValueError("clean matching source snapshot required")
    root.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    registry, rows, selections = default_registry(), [], {}
    manifest = {"schema":"iapf_fresh_calibration_v1", "status":"running", "git_commit":revision,
        "source_checkout":str(REPO), "plan":SHAPE_PLAN if args.protocol=="relative_shape" else PLAN,
        "protocol":args.protocol, "command":sys.argv,
        "environment_name":"tftwogpu", "launch_number":args.launch_number,
        "prior_charges":args.prior_charges, "prior_wall_seconds":args.prior_wall_seconds,
        "prior_cpu_seconds":args.prior_cpu_seconds, "charge_limit":LIMIT,
        "environment":{k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_PRELOAD_CUSTOM_OP")},
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed":1000, "data_version":("repository nonlinear model, fresh 1200-1221 identities"
            if args.protocol=="relative_shape" else "repository nonlinear model, fresh 1100-1125 identities"),
        "studies":[], "statistically_supported_ranking":False, "default_ready":False,
        "score_semantics":"finite_program_frozen_fit_realized_count_and_labels",
        "timing_ranking":"not_permitted_concurrent_workloads", "selections":selections}
    def save():
        manifest.update(charged_attempts=budget.used, remaining_charges=LIMIT-budget.used,
            wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu_started)
        write(root/"attempt-accounting.json", budget.charges)
        write(root/"invalid-fits.json", budget.failures)
        write(root/"consumer-evidence.json", rows)
        write(root/"conditional-heuristics.json", conditional_tables(rows, selections, args.protocol))
        write(root/"run-manifest.json", manifest)
    budget = Budget(save, prior=args.prior_charges,
        wall_limit=2950.-args.prior_wall_seconds, cpu_limit=6900.-args.prior_cpu_seconds)
    (root/"driver-used.py").write_bytes(Path(__file__).read_bytes())
    save()

    def checked(endpoint):
        def evaluate(row, context):
            result = endpoint(row, context)
            validate_result(result, row, registry)
            runtime, diag = result["runtime"], result["diagnostics"]
            if not (runtime["device"]=="GPU" and runtime["tf32"] and runtime["jit_compile"]
                    and runtime["traces"]==1 and runtime["memory_policy"]["all_physical_devices_memory_growth"]):
                raise ValueError("GPU/XLA/trace/memory evidence mismatch")
            if row["proposal"]=="iapf" and (
                    any(n!=1 for n in diag["fit_trace_counts"]+diag["run_trace_counts"])
                    or diag["fit_observation_digest"]!=diag["executed_observation_digest"]
                    or fit_summary(diag)["underflow_steps"]):
                raise ValueError("iAPF diagnostic admission mismatch")
            return result
        return evaluate

    def stage(name, regime, arm, spec):
        before = fingerprint(spec, registry)
        write(root/(name+"-study.json"), spec)
        state = execute(spec, registry, root/name,
            endpoint_loader=lambda endpoint:budget.wrap(checked(load_endpoint(endpoint)), name))
        if before != state["fingerprint"] or before != fingerprint(spec, registry):
            raise CampaignStop("source/study drift")
        for row in spec["rows"]:
            saved = state["rows"][row["id"]]
            if saved["execution_status"] != "complete":
                if not any(f["stage"]==name and f["row"]["id"]==row["id"] for f in budget.failures):
                    raise CampaignStop("unclassified failed/deferred row")
                continue
            result_path = root/name/saved["result_path"]
            result = json.loads(result_path.read_text())
            if digest(result) != saved["result_digest"]:
                raise CampaignStop("result digest mismatch")
            diag = result["diagnostics"]
            rows.append({"stage":name, "regime":regime, "arm":arm, "row":row["id"],
                "dataset":row["dataset"], "replicate":row["replicate"], "proposal":row["proposal"],
                "path":str(result_path), "digest":saved["result_digest"],
                "score_squared_error":diag["score_squared_error"],
                "actual_particles":diag.get("actual_particle_count", spec["settings"]["particles"]),
                "fit_digest":diag.get("fit_digest"), "fit_summary":fit_summary(diag),
                "final_seeds":{k:v for k,v in diag.get("fit_seed_records",{}).items() if k.startswith("iapf_final")}})
        manifest["studies"].append({"name":name, "status":state["execution_status"],
            "path":str(root/name), "rows":len(spec["rows"]), "fingerprint":state["fingerprint"]})
        save()
        return state

    try:
        manifest["runtime"] = configure_runtime(device="GPU", tf32=True, jit_compile=True)
        import tensorflow as tf
        manifest["physical_gpu_details"] = [dict(name=device.name,
            **tf.config.experimental.get_device_details(device))
            for device in tf.config.list_physical_devices("GPU")]
        for regime in REGIMES:
            source, frozen, heuristic = studies(regime, args.protocol)
            # Independent heuristics remain observable even when fitting is invalid.
            stage(regime+"-heuristics", regime, "heuristic", heuristic)
            for arm, spec in (("selected",source), ("baseline",frozen)):
                name = regime+"-"+arm
                state = stage(name+"-source", regime, arm+"-source", spec)
                destination = root/(name+"-selection.json")
                selection = selection_or_blocked(state, root/(name+"-source"), destination, issue_selection)
                selections[name] = selection
                save()
                if selection["status"] == "issued":
                    evaluation = claims(spec, selection["selection"]["selected_controls"], destination, arm)
                    stage(name+"-claim", regime, arm, evaluation)
                else:
                    manifest.setdefault("blocked_claims", []).append({"regime":regime,"arm":arm,
                        "datasets":spec["partitions"]["claim"],"rows":2*len(spec["partitions"]["claim"]),
                        "reason":selection["reason"]})
        for regime in REGIMES:
            for dataset in studies(regime, args.protocol)[0]["partitions"]["claim"]:
                relevant = [r for r in rows if r["regime"]==regime and r["dataset"]==dataset
                            and r["arm"] in ("selected","baseline")]
                for arm in ("selected","baseline"):
                    pair = [r for r in relevant if r["arm"]==arm]
                    if len(pair)==2 and (pair[0]["fit_digest"]!=pair[1]["fit_digest"]
                                         or pair[0]["final_seeds"]==pair[1]["final_seeds"]):
                        raise CampaignStop("frozen-fit/independent-stream check failed")
                for rep in range(2):
                    pair = [r for r in relevant if r["replicate"]==rep]
                    if len(pair)==2 and pair[0]["final_seeds"]!=pair[1]["final_seeds"]:
                        raise CampaignStop("selected and baseline claim streams differ")
        manifest.update(status="completed_with_numerical_vetoes" if budget.failures else "completed",
            numerical_rows=len(rows), invalid_rows=len(budget.failures),
            allocator=memory_usage("GPU"), promotion="not_issued_pilot_evidence")
    except BaseException as error:
        manifest.update(status="execution_stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps({k:manifest[k] for k in ("status","numerical_rows","invalid_rows","charged_attempts","wall_seconds")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protocol", choices=("density", "relative_shape"), default="density")
    parser.add_argument("--source-revision")
    parser.add_argument("--launch-number", type=int, default=1)
    parser.add_argument("--prior-charges", type=int, default=0)
    parser.add_argument("--prior-wall-seconds", type=float, default=0.)
    parser.add_argument("--prior-cpu-seconds", type=float, default=0.)
    args = parser.parse_args()
    if args.dry_run:
        preflight(args.output, args.protocol)
    elif (not args.source_revision or not 1 <= args.launch_number <= 3
          or not 0 <= args.prior_charges <= LIMIT
          or not 0 <= args.prior_wall_seconds < 2950
          or not 0 <= args.prior_cpu_seconds < 6900):
        parser.error("valid source revision, launch count and prior accounting required")
    else:
        run(args)


if __name__ == "__main__":
    main()
