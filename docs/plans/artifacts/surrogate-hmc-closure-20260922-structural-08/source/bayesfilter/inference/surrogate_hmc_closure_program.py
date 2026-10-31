"""Closure-roadmap orchestration. No framework imports or implicit inference.

Scientific stage verdicts and successful subprocess execution are distinct.
The closed LG1 pilot is read for reporting and cannot be launched here.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from bayesfilter.inference.surrogate_hmc_program import (
    ROOT, FAMILIES, campaign_accounting_paths, charge_previous_runs, file_hash,
    run_job, validated_budget, write_json,
)

PLAN = "docs/plans/surrogate-hmc-closure-master-program-2026-09-22.md"
SCHEMA = "bayesfilter.surrogate_hmc_closure.v1"
BASE_CHARGE = dict(cpu_worker_seconds=27189.232653, gpu_seconds=25319.807396)
PASS_CAP = dict(cpu_worker_seconds=2400., gpu_seconds=3600.)
ARTIFACTS = ROOT / "docs/plans/artifacts"
ARCHIVES = {
    "LG1": "surrogate-hmc-all-models-20260919-lg1-sample-01/P2-sample/report.json",
    "SV20": "surrogate-hmc-all-models-20260922-sv20-inference-01/report.json",
    "KSC": "surrogate-hmc-all-models-20260921-ksc-t2-inference-01/report.json",
    "M10": "surrogate-hmc-all-models-20260922-m10-inference-02/P3-M10-inference/report.json",
}


def scope_judgment(*, scope, comparisons, health, accepted=False, artifact_valid=True):
    """Represent owner acceptance independently from archived formal results."""
    if health not in ("PASSED", "FAILED", "NOT_CHECKED"):
        raise ValueError("unknown sampling-health state")
    allowed = {"INCONCLUSIVE", "EQUIVALENT", "DISCREPANCY", "NOT_RUN", "TUNING_REJECTED"}
    if not set(comparisons.values()) <= allowed:
        raise ValueError("unknown formal comparison verdict")
    if accepted and (scope != "LG1" or not artifact_valid or health != "PASSED" or "B/R" not in comparisons):
        raise ValueError("practical acceptance requires the checked closed LG1 evidence")
    practical = ("ACCEPTED_PILOT" if accepted else "INVALID_EVIDENCE" if not artifact_valid
                 else "NUMERICAL_FAILURE" if health == "FAILED" else
                 "VIABLE_PILOT_REQUIRES_SCOPE_JUDGMENT" if health == "PASSED" else "NOT_ASSESSED")
    return dict(scope=scope, implementation_validity="NO_NEW_VALIDITY_CLAIM" if artifact_valid else "INVALID_ARTIFACT",
                sampling_health=health, practical_judgment=practical,
                practical_basis="owner accepted LG1 on 2026-09-22" if accepted else None,
                formal_comparisons=dict(comparisons),
                next_action="CLOSED_NO_RERUN" if accepted else "SCOPE_SPECIFIC_ACCURACY_AND_UNCERTAINTY_ASSESSMENT",
                inference_launched=False, efficiency="NOT_ESTABLISHED")


def archived_judgments():
    rows = {}
    summary_path = ARTIFACTS/"surrogate-hmc-all-models-20260922-master-reporting-03/decision.json"
    summary = json.loads(summary_path.read_text())
    for scope, relative in ARCHIVES.items():
        path = ARTIFACTS / relative
        try:
            report = json.loads(path.read_text())
            arm = report["arms"]["B"]
            chunks = arm.get("chunks", [])
            if not chunks:
                raise ValueError("missing retained chunk health")
            if any(type(row.get("health_passed")) is not bool for row in chunks):
                raise ValueError("missing typed chunk health")
            health = "PASSED" if all(row["health_passed"] for row in chunks) else "FAILED"
            comparisons = summary["scope_results"][scope]["comparisons"]
            if comparisons["B/R"] != arm["status"]:
                raise ValueError("archived aggregate disagrees with retained B result")
            row = scope_judgment(scope=scope, comparisons=comparisons, health=health, accepted=scope == "LG1")
            row.update(report=str(path), report_sha256=file_hash(path),
                       formal_summary=str(summary_path), formal_summary_sha256=file_hash(summary_path),
                       uncertainty="ARCHIVED_RULE_AND_ORIGINAL_VERDICT_UNCHANGED")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row = dict(scope=scope, implementation_validity="INVALID_ARTIFACT", sampling_health="NOT_CHECKED",
                       practical_judgment="NOT_ASSESSED", error=str(exc), report=str(path), inference_launched=False)
        rows[scope] = row
    return rows


def coverage_register():
    known = {("M01", "LG1"): "CLOSED_ACCEPTED_PILOT", ("M01", "LG2"): "ENDPOINT_QUALIFIED_DESIGN_AND_BUDGET_PENDING",
             ("M02", "SV20"): "HEALTHY_PILOT_FORMAL_INCONCLUSIVE",
             ("M03", "KSC-mixture"): "HEALTHY_T2_PILOT_FORMAL_INCONCLUSIVE",
             ("M08", "AR2"): "STRUCTURAL_MECHANICS_CHECKED_REFERENCE_AND_SCOPED_INFERENCE_PENDING",
             ("M09", "model-B"): "GENERATIVE_CONSUMER_AND_REFERENCE_REQUIRED",
             ("M09", "model-C"): "GENERATIVE_CONSUMER_AND_REFERENCE_REQUIRED",
             ("M10", "concrete-score-study"): "HEALTHY_T2_PILOT_FORMAL_INCONCLUSIVE",
             ("M13", "range-bearing"): "OBSERVATION_REPAIR_UNDER_TEST_MARGINAL_REFERENCE_MISSING"}
    return [dict(id=ident, family=family, variants=[dict(name=variant,
        state=known.get((ident, variant), "CONCRETE_SCOPE_AND_QUALIFICATION_REQUIRED"))
        for variant in variants]) for ident, family, variants in FAMILIES]


def readiness(scope, evidence):
    if scope == "LG1":
        return dict(scope=scope, ready=False, status="CLOSED_NO_RERUN", missing=[])
    required = ("same_model_law", "canonical_consumer", "independent_reference", "scoped_dctr_controls",
                "public_hmc_tuning", "prospective_precision_design", "qualified_uncertainty", "budget_forecast")
    missing = [name for name in required if evidence.get(name) is not True]
    return dict(scope=scope, ready=not missing, status="READY" if not missing else "BLOCKED_PREREQUISITES",
                missing=missing)


def inference_readiness():
    """New-scope admission is separate from completed historical preparation."""
    prepared = {"LG2": "canonical consumer, scoped controls, model reference and inference endpoint qualified; pilot exceeds family allocation",
        "KSC": "canonical consumer, scoped controls, public tuning and both model/finite posterior references qualified for archived T2 pilot",
        "SV20": "canonical consumer, scoped controls, public tuning and model posterior reference qualified for archived pilot",
        "M10": "canonical consumer, scoped controls, public tuning and model posterior reference qualified for archived T2 pilot",
        "AR2": "structural mechanics, independent model reference and scoped FP32 controls passed structural-02/03/04; cost, uncertainty calibration and posterior inference remain pending",
        "M09": "shared/nonadditive innovation model law identified; canonical generative consumer and independent posterior reference missing",
        "M13": "wrapped observation law repair is a separate stage; integrated likelihood/reference and canonical consumer remain missing"}
    rows = []
    for scope in ("LG1", *prepared):
        row = readiness(scope, {})
        row["prepared_work"] = prepared.get(scope, "accepted practical pilot; do not rerun")
        row["missing_fields_mean"] = "not admitted for a new prospective scope; prepared historical work above is preserved"
        if scope in ("LG2", "KSC", "SV20", "M10"):
            row["next_action"] = "specify material accuracy question, qualify scoped uncertainty, forecast precision/cost, bind existing preparation and retune publicly"
        rows.append(row)
    return rows


def cpu_self():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime+usage.ru_stime


def preceding_charges(output):
    paths = [p for p in campaign_accounting_paths(ARTIFACTS) if p.parent.resolve() != output.resolve()]
    ledger = validated_budget(226800., 54000.)
    charge_previous_runs(ledger, paths)
    records = ledger["prior_runs"]
    closure = [r for r in records if Path(r["path"]).parent.name.startswith("surrogate-hmc-closure-")]
    other = [r for r in records if r not in closure]
    # Retain the opening bookkeeping floor if an older receipt is unavailable.
    # Interleaved original-master jobs are always added through fresh receipts.
    total = {key: max(BASE_CHARGE[key], math.fsum(r["charged_this_run"][key] for r in other))
             + math.fsum(r["charged_this_run"][key] for r in closure) for key in BASE_CHARGE}
    return total, records


def remaining_pass_reservation(records, pass_kind="initial"):
    if pass_kind not in ("initial", "structural"):
        raise ValueError("unknown closure pass")
    prefix = "surrogate-hmc-closure-20260922-"
    pass_records = [r for r in records if Path(r["path"]).parent.name.startswith(prefix)
                    and Path(r["path"]).parent.name.startswith(prefix+"structural-") == (pass_kind == "structural")]
    remaining = {key: cap-math.fsum(r["charged_this_run"][key] for r in pass_records)
                 for key, cap in PASS_CAP.items()}
    if any(value <= 0 for value in remaining.values()):
        raise ValueError(f"{pass_kind} closure-pass budget exhausted, including preceding attempts")
    return remaining


def jobs(selected=None):
    worker = [sys.executable, "-m", "bayesfilter.testing.surrogate_hmc_closure_diagnostics"]
    available = [dict(name="review", dependencies=[], device="cpu", timeout=240., cpu=300.,
                 command=[sys.executable, "-m", "pytest", "-q", "tests/test_surrogate_hmc_closure_program.py",
                          "tests/test_wrapped_gaussian_tf.py", "tests/test_surrogate_hmc_structural_scope.py",
                          "tests/test_surrogate_hmc_range_bearing_scope.py", "tests/test_surrogate_hmc_program.py",
                          "tests/test_surrogate_hmc_dependency_execution.py"]),
            dict(name="wrapped", dependencies=["review"], device="gpu", timeout=120., cpu=150., command=worker+["--stage", "wrapped"]),
            dict(name="structural", dependencies=["review"], device="cpu", timeout=120., cpu=150., command=worker+["--stage", "structural"]),
            dict(name="uncertainty", dependencies=["review"], device="gpu", timeout=3000., cpu=1650., command=worker+["--stage", "uncertainty"]),
            dict(name="uncertainty-confirmation", dependencies=["review"], device="gpu", timeout=3000., cpu=1650.,
                 command=worker+["--stage", "uncertainty-confirmation"]),
            dict(name="structural-integration", dependencies=["review"], device="gpu", timeout=1200., cpu=900.,
                 command=worker+["--stage", "structural-integration"]),
            dict(name="ar2-reference", dependencies=["review"], device="gpu", timeout=600., cpu=300.,
                 command=worker+["--stage", "ar2-reference"]),
            dict(name="ar2-reference-extended", dependencies=["review"], device="gpu", timeout=600., cpu=300.,
                 command=worker+["--stage", "ar2-reference-extended"]),
            dict(name="ar2-controls", dependencies=["review"], device="gpu", timeout=1500., cpu=900.,
                 command=worker+["--stage", "ar2-controls"]),
            dict(name="ar2-cost", dependencies=["review"], device="gpu", timeout=900., cpu=600.,
                 command=worker+["--stage", "ar2-cost"]),
            dict(name="ar2-profile", dependencies=["review"], device="gpu", timeout=600., cpu=450.,
                 command=worker+["--stage", "ar2-profile"]),
            dict(name="ar2-kernel-repair", dependencies=["review"], device="gpu", timeout=600., cpu=450.,
                 command=worker+["--stage", "ar2-kernel-repair"])]
    names = set(selected.split(",")) if selected else {"review", "wrapped", "structural", "uncertainty"}
    if not names or not names <= {row["name"] for row in available}:
        raise ValueError("unknown or empty closure stage selection")
    if {"uncertainty", "uncertainty-confirmation"} <= names:
        raise ValueError("run the fresh confirmation after inspecting and preserving the initial calibration")
    names.add("review")
    references = names & {"ar2-reference", "ar2-reference-extended"}
    if references:
        if len(references) != 1 or names != {"review", *references}:
            raise ValueError("inspect the AR2 reference before a separate bounded refinement")
        available[0].update(timeout=180., cpu=180., command=[sys.executable, "-m", "pytest", "-q",
                    "tests/test_surrogate_hmc_closure_program.py", "tests/test_surrogate_hmc_ar2_reference.py"])
    if "ar2-controls" in names:
        if names != {"review", "ar2-controls"}:
            raise ValueError("AR2 controls require a separate inspected reference prerequisite")
        available[0].update(timeout=180., cpu=180., command=[sys.executable, "-m", "pytest", "-q",
                    "tests/test_surrogate_hmc_closure_program.py", "tests/test_surrogate_hmc_ar2_consumer.py"])
    if names & {"ar2-cost", "ar2-profile"}:
        if len(names) != 2:
            raise ValueError("inspect AR2 cost before allocating inference")
        available[0].update(timeout=180., cpu=180., command=[sys.executable, "-m", "pytest", "-q",
                    "tests/test_surrogate_hmc_closure_program.py", "tests/test_surrogate_hmc_scoped_assessment.py"])
    if "structural-integration" in names:
        if names != {"review", "structural-integration"}:
            raise ValueError("the structural integration pass has a separate bounded reservation")
        available[0].update(timeout=600., cpu=750.)
        available[0]["command"] += ["tests/test_ledh_structural_innovation.py",
                                    "tests/highdim/test_ledh_canonical_score_full.py",
                                    "tests/highdim/test_ledh_numerical_safety_tf.py"]
    if "ar2-kernel-repair" in names:
        if names != {"review", "ar2-kernel-repair"}:
            raise ValueError("inspect kernel non-harm before scoped requalification")
        available[0].update(timeout=300., cpu=300., command=[sys.executable, "-m", "pytest", "-q",
            "tests/test_surrogate_hmc_closure_program.py", "tests/highdim/test_ledh_small_matrix_tf.py",
            "tests/highdim/test_ledh_numerical_safety_tf.py", "tests/test_ledh_structural_innovation.py",
            "tests/highdim/test_ledh_canonical_score_full.py"])
    return [row for row in available if row["name"] in names]


def write_summary(output, report):
    lines = ["# Surrogate-HMC closure execution", "", report["status"], "",
             "The goal is testing canonical DCTR-force HMC with separate endpoint acceptance. "
             "LG1 remains closed. No new posterior inference is claimed by this repair pass.", "",
             "| Stage | Execution | Scientific finding |", "|---|---|---|"]
    for name, record in report["stages"].items():
        lines.append(f'| {name} | {record["execution"]} | {record.get("result", {}).get("status", "NOT_EVALUATED")} |')
    lines += ["", "| Scope | Practical judgment | Formal evidence |", "|---|---|---|"]
    for scope, row in report["scope_judgments"].items():
        lines.append(f'| {scope} | {row["practical_judgment"]} | {row.get("formal_comparisons", "INVALID_ARTIFACT")} |')
    lines += ["", "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
              "|---|---|---|---|---|---|",
              "| Continue qualified repairs | Stage-specific findings above | Inference prerequisites remain enforced | General structural support and model-specific references | Resolve recorded prerequisites before chains | All-model posterior validity |", "",
              "| Inference status | Finding |", "|---|---|",
              "| Hard veto screen | See each numerical stage and inference readiness record |",
              "| Statistically supported ranking | None |",
              "| Descriptive-only differences | Timing and existing pilot summaries |",
              "| Default-readiness | No uncertainty or HMC runtime default promoted |",
              "| Next evidence needed | Complete canonical model support, scoped references and precision designs |", "",
              "Budget charges and commands: accounting.json and run-manifest.json. "
              "Full stage evidence: report.json and stage directories.", "",
              "Post-run review: a completed diagnostic is not a completed surrogate-HMC test. "
              "Reference-table calibration cannot prove real-chain convergence. "
              "AR2 additive structural mechanics passed in structural-02; model references, scoped controls and posterior tests remain separate tasks.",
              "Next actions belong to the existing master unless explicitly identified as a direction change. Continue within the campaign budget; any stop must name its reason.", ""]
    (output/"result.md").write_text("\n".join(lines))


def snapshot_sources(output):
    sources = [Path(__file__), ROOT/PLAN, ROOT/"bayesfilter/wrapped_gaussian_tf.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_closure_diagnostics.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_structural_integration.py",
               ROOT/"tests/test_ledh_structural_innovation.py",
               ROOT/"docs/plans/surrogate-hmc-structural-integration-2026-09-22.md",
               ROOT/"bayesfilter/testing/surrogate_hmc_ar2_reference.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py",
               ROOT/"tests/test_surrogate_hmc_ar2_reference.py",
               ROOT/"docs/plans/surrogate-hmc-ar2-reference-2026-09-23.md",
               ROOT/"bayesfilter/testing/surrogate_hmc_ar2_consumer.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_ar2_controls.py",
               ROOT/"bayesfilter/highdim/gaussian_cloud_designs_tf.py",
               ROOT/"bayesfilter/inference/surrogate_hmc_dctr_selection.py",
               ROOT/"tests/test_surrogate_hmc_ar2_consumer.py",
               ROOT/"docs/plans/surrogate-hmc-ar2-controls-2026-09-23.md",
               ROOT/"docs/plans/surrogate-hmc-ar2-inference-2026-09-23.md",
               ROOT/"bayesfilter/testing/surrogate_hmc_ar2_sampling.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_ar2_profile.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_scoped_assessment.py",
               ROOT/"tests/test_surrogate_hmc_scoped_assessment.py",
               ROOT/"bayesfilter/inference/surrogate_hmc_program.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_mcse_randomization.py",
               ROOT/"bayesfilter/inference/hmc_posterior_diagnostics.py",
               ROOT/"bayesfilter/inference/surrogate_hmc_equivalence.py",
               ROOT/"bayesfilter/testing/surrogate_hmc_nonlinear_inference_metrics.py",
               ROOT/"experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py"]
    sources += list((ROOT/"bayesfilter/highdim").glob("ledh_*.py"))
    sources += [ROOT/name for name in (
        "bayesfilter/highdim/higher_moment_contract_e.py",
        "bayesfilter/testing/surrogate_hmc_ar2_kernel_repair.py",
        "tests/highdim/test_ledh_small_matrix_tf.py",
        "docs/plans/surrogate-hmc-ar2-kernel-repair-2026-09-23.md")]
    sources += [ROOT/"tests"/name for name in ("test_surrogate_hmc_closure_program.py", "test_wrapped_gaussian_tf.py",
                                               "test_surrogate_hmc_structural_scope.py", "test_surrogate_hmc_range_bearing_scope.py",
                                               "test_surrogate_hmc_program.py", "test_surrogate_hmc_dependency_execution.py")]
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    for path in sources:
        target = output/"source"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", help="comma-separated independent stages; review is always included")
    args = parser.parse_args(argv)
    selected_jobs = jobs(args.stages)
    output = args.output.resolve()
    if output.exists():
        parser.error("use a fresh output directory; prior evidence cannot be overwritten")
    if output.parent != ARTIFACTS or not output.name.startswith("surrogate-hmc-closure-20260922-"):
        parser.error("use the plan's versioned artifact prefix so all attempts are accounted")
    previous, previous_files = preceding_charges(output)
    pass_kind = "structural" if any(job["name"] == "structural-integration" or job["name"].startswith("ar2-")
                                     for job in selected_jobs) else "initial"
    if output.name.startswith("surrogate-hmc-closure-20260922-structural-") != (pass_kind == "structural"):
        parser.error("structural/AR2 stages require the structural-NN output prefix and its own pass reservation")
    reservation = remaining_pass_reservation(previous_files, pass_kind)
    ledger = validated_budget(reservation["gpu_seconds"], reservation["cpu_worker_seconds"])
    for key, cap in dict(cpu_worker_seconds=54000., gpu_seconds=226800.).items():
        if previous[key]+reservation[key] > cap:
            parser.error("closure-pass reservation exceeds remaining campaign budget")
    if args.dry_run:
        print(json.dumps(dict(plan=PLAN, reservation=reservation, previous_charge=previous,
                              jobs=selected_jobs, pass_kind=pass_kind, inference_launched=False), indent=2))
        return 0
    output.mkdir(parents=True)
    start, parent_start = time.monotonic(), cpu_self()
    manifest = dict(schema=SCHEMA, command=sys.argv, cwd=str(ROOT), plan=PLAN,
        environment=sys.executable, pass_kind=pass_kind, previous_charge=previous, previous_accounting=previous_files,
        reservation=reservation, pass_total_cap=PASS_CAP, result_file=str(output/"result.md"), seeds="2026092200 onwards; worker protocol",
        device="orchestrator CPU; trusted GPU numerical workers; CPU reference exceptions recorded per stage")
    report = dict(schema=SCHEMA, status="RUNNING", stages={}, scope_judgments={},
                  coverage=coverage_register(), all_models_validated=False, inference_launched=False)
    # Process-local CPU timer is not inherited by children, unlike RLIMIT_CPU.
    old_cpu_handler = signal.getsignal(signal.SIGPROF)
    def parent_cpu_exhausted(signum, frame):
        raise TimeoutError("closure orchestrator CPU allowance exhausted")
    signal.signal(signal.SIGPROF, parent_cpu_exhausted)
    old_timer = signal.setitimer(signal.ITIMER_PROF, min(120., reservation["cpu_worker_seconds"]))
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        hashes = snapshot_sources(output)
        manifest["source_hashes"] = hashes
        write_json(output/"run-manifest.json", manifest)
        report["scope_judgments"] = archived_judgments()
        if any(row["implementation_validity"] == "INVALID_ARTIFACT" for row in report["scope_judgments"].values()):
            raise ValueError("archived pilot evidence could not be read; see report")
        for job in selected_jobs:
            name = job["name"]
            attempt_names = ("ar2-reference", "ar2-reference-extended") if name.startswith("ar2-reference") else (name,)
            attempts = sum(any((Path(item["path"]).parent/attempt_name/"run-manifest.json").exists()
                               for attempt_name in attempt_names)
                           for item in previous_files
                           if Path(item["path"]).parent.name.startswith("surrogate-hmc-closure-20260922-"))
            # Routine regression repairs remain bounded by their CPU allowance
            # and the cumulative pass cap. They do not consume scientific tries.
            if name != "review" and attempts >= 3:
                report["stages"][name] = dict(execution="ATTEMPT_LIMIT")
                continue
            if any(report["stages"][dependency]["execution"] != "COMPLETED" for dependency in job["dependencies"]):
                report["stages"][name] = dict(execution="BLOCKED_DEPENDENCY")
                continue
            command = list(job["command"])
            if name != "review":
                command += ["--output", str(output/name/"report.json"), "--device", job["device"]]
            # Parent work is charged before each reservation as well as at exit.
            ledger["charged"]["cpu_worker_seconds"] = sum(x["cpu_worker_seconds"] for x in ledger["jobs"])+cpu_self()-parent_start
            record = run_job(command, output/name, ledger, device=job["device"],
                             timeout=job["timeout"], cpu_allowance=job["cpu"])
            scientific = {}
            path = output/name/"report.json"
            if path.exists():
                scientific = json.loads(path.read_text())
            elif record["status"] == "COMPLETED" and name != "review":
                record["status"] = "MISSING_REQUIRED_REPORT"
            report["stages"][name] = dict(execution=record["status"], result=scientific)
            write_json(output/"report.json", report)
        report["readiness"] = inference_readiness()
        changed = [path for path, digest in hashes.items() if file_hash(ROOT/path) != digest]
        if changed:
            raise ValueError(f"source changed during execution: {changed}")
        report["source_unchanged"] = True
        failed = any(row["execution"] != "COMPLETED" for row in report["stages"].values())
        report["status"] = "EXECUTION_INCOMPLETE" if failed else "COMPLETED_BOUNDED_CLOSURE_PASS_CAMPAIGN_INCOMPLETE"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="EXECUTION_FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_PROF, *old_timer)
        signal.signal(signal.SIGPROF, old_cpu_handler)
        ledger["charged"]["cpu_worker_seconds"] = sum(x["cpu_worker_seconds"] for x in ledger["jobs"])+cpu_self()-parent_start
        accounting = dict(charged_this_run=ledger["charged"], previous_charge=previous,
                          cumulative={k: previous[k]+ledger["charged"][k] for k in previous}, jobs=ledger["jobs"])
        manifest.update(wall_seconds=time.monotonic()-start, cpu_worker_seconds=ledger["charged"]["cpu_worker_seconds"],
                        gpu_seconds=ledger["charged"]["gpu_seconds"])
        write_json(output/"accounting.json", accounting)
        write_json(output/"report.json", report)
        write_json(output/"run-manifest.json", manifest)
        write_summary(output, report)
    print(json.dumps(dict(status=report["status"], report=str(output/"report.json"), charged=ledger["charged"])))
    return 0 if report["status"] == "COMPLETED_BOUNDED_CLOSURE_PASS_CAMPAIGN_INCOMPLETE" else 1
