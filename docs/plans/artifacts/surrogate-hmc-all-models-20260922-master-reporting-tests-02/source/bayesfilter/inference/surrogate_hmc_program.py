"""Bounded orchestration and source inventory for surrogate-HMC experiments.

This module deliberately has no numerical/framework imports. A dry run cannot
initialize CUDA. A scope missing its target, reference or tuning authority is
reported as incomplete; it cannot contribute a successful inference result.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/surrogate-hmc-all-models-master-program-2026-09-18.md"
EXECUTION_PLAN = "docs/plans/surrogate-hmc-all-models-execution-2026-09-19.md"
SCHEMA = "bayesfilter.surrogate_hmc_master.v1"
GPU_CAP = 63 * 3600
CPU_CAP = 15 * 3600
M10_CPU_CAP = 4600.  # September22: +2800 transferred from unused P5; global cap unchanged.

# The master is the scientific source of this minimum register. Discovery below
# preserves additional source candidates; this list is not a completeness claim.
FAMILIES = (
    ("M01", "linear Gaussian", ("LG1", "LG2", "LG4", "LG20", "diagonal", "triangular-all-parameter")),
    ("M02", "actual stochastic volatility", ("synthetic", "real-data", "SV20", "SV40")),
    ("M03", "transformed stochastic volatility", ("exact-transformed", "KSC-mixture")),
    ("M04", "generalized stochastic volatility", ("prior-mean-projection", "native-two-state")),
    ("M05", "predator-prey", ("six-parameter", "PP20-restricted")),
    ("M06", "Austria SIR", ("physical-clipped", "preclip-distinct-law")),
    ("M07", "coupled nonlinear", ("coupled-state",)),
    ("M08", "structural", ("affine", "AR2", "nonlinear-five-parameter", "AR1-quadratic-completion")),
    ("M09", "accumulation and growth", ("model-B", "model-C")),
    ("M10", "nonlinear score-study", ("concrete-score-study",)),
    ("M11", "SSL-LSTM", ("fixed", "complexity-ladder")),
    ("M12", "hard-bound macrofinancial", ("hard-bound",)),
    ("M13", "range-bearing", ("range-bearing",)),
    ("M14", "BGS", ("concrete-upstream-instance-required",)),
    ("C01", "Gaussian mechanism controls", ("anisotropic", "d100")),
    ("C02", "non-Gaussian mechanism controls", ("funnel", "banana", "mixture", "varying-hessian")),
    ("C03", "German-credit posterior", ("data-version-required",)),
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(root: Path = ROOT) -> dict:
    """Index concrete-model candidates without importing historical modules."""
    records, errors = [], []
    roots = ("bayesfilter", "experiments", "docs/benchmarks", "tests")
    tokens = ("model", "ssm", "predator", "sir", "volatility", "lgssm", "funnel", "banana", "range_bearing")
    files = subprocess.check_output(["rg", "--files", *roots, "-g", "*.py"], cwd=root, text=True).splitlines()
    for relative in sorted(files):
        path = root / relative
        try:
            tree = ast.parse(path.read_text(), filename=relative)
        except (SyntaxError, UnicodeError) as exc:
            errors.append({"path": relative, "error": str(exc)})
            continue
        definitions = [{"name": node.name, "line": node.lineno, "kind": type(node).__name__}
                       for node in ast.walk(tree)
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                       and any(token in node.name.lower() for token in tokens)]
        if definitions:
            records.append({"path": relative, "sha256": file_hash(path), "definitions": definitions})
    return {"schema": SCHEMA, "roots": roots, "python_files_examined": len(files),
            "classification": "discovery_index_requires_concrete_instance_reconciliation",
            "all_models_coverage_established": False, "records": records, "parse_errors": errors}


def model_register() -> dict:
    rows = []
    for ident, name, variants in FAMILIES:
        rows.append({"id": ident, "family": name, "variants": list(variants),
                     "role": "model_inference" if ident.startswith("M") else "mechanism_control",
                     "status": "PARTIAL_LINEAR_IMPLEMENTATION" if ident == "M01" else
                               "ANISOTROPIC_CONTROL_IMPLEMENTED" if ident == "C01" else
                               "PARTIAL_SV_PREPARATION_AVAILABLE" if ident == "M02" else
                               "REFERENCE_CONSUMER_AVAILABLE" if ident == "M10" else
                               "PARTIAL_KSC_INFERENCE_AVAILABLE" if ident == "M03" else "SCOPE_UNRESOLVED",
                     "variant_status": {variant: "REFERENCE_CONTROL_TUNING_SAMPLING_AVAILABLE" if variant == "LG1"
                                        else "REFERENCE_CONSUMER_CALIBRATION_AVAILABLE" if variant == "LG2"
                                        else "TUNING_AND_RETAINED_COMPARISON_AVAILABLE" if ident == "C01" and variant == "anisotropic"
                                        else "REFERENCE_CONSUMER_CALIBRATION_AVAILABLE" if ident == "M02" and variant == "SV20"
                                        else "POSTERIOR_REFERENCE_CONSUMER_MECHANICS_AVAILABLE" if ident == "M10"
                                        else "TUNING_AND_RETAINED_COMPARISON_AVAILABLE" if ident == "M03" and variant == "KSC-mixture"
                                        else "SCOPE_UNRESOLVED" for variant in variants},
                     "inference_status": "NOT_RUN",
                     "missing": (["resolved posterior comparisons", "heuristic comparisons",
                                  "remaining linear scopes"] if ident == "M01"
                                 else ["affordable complete inference forecast", "public tuning and retained comparisons",
                                       "heuristic comparisons", "remaining SV scopes"] if ident == "M02"
                                 else ["inference endpoint qualification",
                                       "scope-specific calibration and tuning", "retained comparisons"] if ident == "M10"
                                 else ["resolved KSC posterior comparisons", "independent finite-D endpoint and reference",
                                       "C tuning and retained comparisons", "heuristic comparisons",
                                       "exact-transformed variant and longer horizons"] if ident == "M03"
                                 else ["concrete scope card", "qualified consumer and reference",
                                       "scope-specific tuning and budget forecast"]),
                     "source": PLAN})
    structural = next(row for row in rows if row["id"] == "M08")
    structural["variant_status"]["AR2"] = "CANONICAL_CONSUMER_SINGULAR_PROCESS_COVARIANCE"
    structural["scope_findings"] = {"AR2": {
        "classification": "IMPLEMENTATION_BLOCKER_NOT_SURROGATE_HMC_REJECTION",
        "finding": "The shared canonical batch consumer rejects the singular AR2 process covariance; adding noise changes its deterministic lag law",
        "result": "docs/plans/surrogate-hmc-structural-scope-result-2026-09-22.md",
        "evidence": "docs/plans/artifacts/surrogate-hmc-all-models-20260922-structural-scope-audit-01/run-manifest.json"}}
    tracking = next(row for row in rows if row["id"] == "M13")
    tracking["variant_status"]["range-bearing"] = "WRAPPED_MODEL_REFERENCE_UNQUALIFIED"
    tracking["scope_findings"] = {"range-bearing": {
        "classification": "REFERENCE_LAW_BLOCKER_NOT_FORCE_ACCURACY_VETO",
        "finding": "The nearest-alias callback is not the normalized wrapped simulator likelihood for general noise-scale inference; the default narrow-noise check agrees numerically",
        "result": "docs/plans/surrogate-hmc-range-bearing-scope-result-2026-09-22.md",
        "evidence": "docs/plans/artifacts/surrogate-hmc-all-models-20260922-range-bearing-scope-audit-01/run-manifest.json"}}
    return {"schema": SCHEMA, "question": "Does imperfect full-DCTR surrogate HMC recover model parameter posteriors?",
            "all_models_validated": False, "coverage_complete": False, "families": rows}


def _ksc_exact_inference_history(root: Path) -> dict:
    """Report preserved KSC evidence without authorizing another statistical look.

    Check archived source and sample hashes, not equality with today's source:
    a later orchestration repair must not erase a completed experiment.
    """
    state = dict(status="NOT_RUN", model_id="M03-KSC-T2", surrogate_hmc_tested=False,
                 retained_surrogate_hmc_tested=False, authorizes_new_inference=False,
                 comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")})
    reports = sorted(root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*-ksc-t2-inference-*/report.json"))
    if not reports:
        return state
    state["attempt_reports"] = [str(path.resolve()) for path in reports]
    if len(reports) != 1:
        state.update(status="RECONCILIATION_REQUIRED", error="Multiple KSC inference attempts require result-aware reconciliation")
        return state
    path = reports[0]
    state.update(report=str(path.resolve()), report_sha256=file_hash(path))
    try:
        report = json.loads(path.read_text())
        run = json.loads((path.parent / "run-manifest.json").read_text())
        if report.get("model_id") != "M03-KSC-T2" or run.get("scope") != "M03-KSC-T2":
            raise ValueError("wrong KSC scope")
        if "exit_code" not in run:
            state.update(status="RUNNING", closed=False)
            return state
        if report.get("status") != "COMPLETED_BOUNDED_INFERENCE" or run["exit_code"] != 0:
            state.update(status=report.get("status", "INCOMPLETE_ATTEMPT"), closed=True,
                         next_action="Inspect the preserved attempt before any repair or new look")
            return state
        accounting = json.loads((path.parent / "accounting.json").read_text())
        if accounting.get("exit_code") != 0 or not run.get("source_hashes"):
            raise ValueError("missing successful accounting or source provenance")
        for name, digest in run["source_hashes"].items():
            if file_hash(path.parent / "executed-source" / name) != digest:
                raise ValueError("archived KSC source checksum mismatch: " + name)
        if file_hash(path.parent / "data.json") != run["data_sha256"]:
            raise ValueError("archived KSC data checksum mismatch")
        summaries = {}
        for arm in ("A", "B"):
            row = report["arms"][arm]
            chunks, looks = row["chunks"], row["looks"]
            if [look["draws_per_chain"] for look in looks] not in ([], [1000], [1000, 2000]):
                raise ValueError("undeclared KSC statistical look")
            if any(look["status"] in ("EQUIVALENT", "DISCREPANCY")
                   and look.get("diagnostic_valid") is not True for look in looks):
                raise ValueError("KSC conclusion has invalid uncertainty diagnostics")
            if any(look["status"] != "INCONCLUSIVE" for look in looks[:-1]):
                raise ValueError("KSC sampling continued after a closed statistical look")
            if row["tuning"]["passed"]:
                tune = Path(row["tuning"]["manifest"])
                tuning = json.loads(tune.read_text())
                if tuning.get("passed") is not True or not all(tuning["checks"].values()):
                    raise ValueError("KSC tuning/report mismatch")
                for index, chunk in enumerate(chunks):
                    archive_path = Path(chunk["manifest"])
                    archive = json.loads(archive_path.read_text())
                    sample = archive["raw_samples"]
                    if (file_hash(archive_path.parent / sample["path"]) != sample["sha256"]
                            or archive["tuning_manifest_sha256"] != file_hash(tune)
                            or archive["health_passed"] != chunk["health_passed"]):
                        raise ValueError("KSC retained archive checksum or health mismatch")
                    expected_previous = chunks[index - 1]["manifest"] if index else None
                    if archive["continuation_manifest"] != expected_previous:
                        raise ValueError("KSC continuation chain mismatch")
                if row["status"] != "HEALTH_VETO":
                    if (not looks or any(not c["health_passed"] for c in chunks)
                            or row["retained_draws"] != 250 * len(chunks)
                            or row["status"] != looks[-1]["status"]
                            or looks[-1]["draws_per_chain"] != row["retained_draws"]
                            or (row["status"] == "INCONCLUSIVE" and row["retained_draws"] != 2000)):
                        raise ValueError("KSC retained result is not a closed declared look")
            elif row["status"] != "TUNING_REJECTED" or chunks or looks:
                raise ValueError("rejected KSC tuning has retained draws")
            summaries[arm] = dict(status=row["status"], retained_draws=row.get("retained_draws", 0),
                tuning_passed=row["tuning"]["passed"],
                diagnostic_valid=looks[-1]["diagnostic_valid"] if looks else None)
        comparisons = report["posterior_comparisons"]
        if (comparisons.get("B/R") != summaries["B"]["status"]
                or any(comparisons.get(key) != "NOT_RUN" for key in ("C/D", "D/R", "C/R"))):
            raise ValueError("KSC comparison/result mismatch")
        state.update(status="COMPLETED_BOUNDED_INFERENCE", closed=True, artifact_valid=True,
            comparisons=comparisons, arms=summaries, surrogate_hmc_tested=True,
            retained_surrogate_hmc_tested=bool(report["arms"]["B"]["chunks"]),
            heuristic_dominance=report["heuristic_dominance"], all_models_validated=False,
            run_manifest=str(path.parent / "run-manifest.json"),
            next_action="Preserve closed looks; qualify finite-D reference for independent C/D work")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(status="ARTIFACT_INVALID", artifact_valid=False, error=str(exc))
    return state


def _checked_ksc_archive(directory: Path) -> dict:
    """Validate preserved evidence against its executed source, not current code."""
    run = json.loads((directory / "run-manifest.json").read_text())
    accounting = json.loads((directory / "accounting.json").read_text())
    if run.get("exit_code") != 0 or accounting.get("exit_code") != 0 or not run.get("source_hashes"):
        raise ValueError("KSC archive lacks successful accounting or executed source")
    for name, digest in run["source_hashes"].items():
        if file_hash(directory / "executed-source" / name) != digest:
            raise ValueError("KSC archived source mismatch: " + name)
    if file_hash(directory / "data.json") != run["data_sha256"]:
        raise ValueError("KSC archived data mismatch")
    return run


def ksc_inference_history(root: Path) -> dict:
    """Combine independent, closed B/R and C/D experiments without new looks."""
    state = _ksc_exact_inference_history(root)
    paths = sorted(root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*-ksc-t2-finite-inference-*/report.json"))
    if not paths or state["status"] in ("ARTIFACT_INVALID", "RECONCILIATION_REQUIRED", "RUNNING"):
        return state
    state["finite_attempt_reports"] = [str(path.resolve()) for path in paths]
    if len(paths) != 1:
        state.update(status="RECONCILIATION_REQUIRED", artifact_valid=False,
                     error="Multiple KSC finite inference attempts require reconciliation")
        return state
    path = paths[0]
    try:
        report = json.loads(path.read_text())
        run = json.loads((path.parent / "run-manifest.json").read_text())
        if "exit_code" not in run:
            state["finite_status"] = "RUNNING"
            return state
        if report.get("status") != "COMPLETED_BOUNDED_FINITE_INFERENCE" or run["exit_code"] != 0:
            state["finite_status"] = report.get("status", "INCOMPLETE_ATTEMPT")
            return state
        run = _checked_ksc_archive(path.parent)
        if report.get("model_id") != "M03-KSC-T2" or run.get("scope") != "M03-KSC-T2":
            raise ValueError("wrong finite KSC scope")
        if state.get("artifact_valid"):
            previous = json.loads(Path(state["run_manifest"]).read_text())
            if (previous["data_sha256"] != run["data_sha256"]
                    or file_hash(Path(previous["controls_root"]) / "selection.json") != run["selection_sha256"]
                    or file_hash(Path(previous["posterior_root"]) / "report.json") != run["posterior_report_sha256"]):
                raise ValueError("independent KSC arms have different data, controls or reference")
        for key in ("finite_reference", "endpoint", "posterior"):
            if file_hash(Path(run[key + "_root"]) / "report.json") != run[key + "_report_sha256"]:
                raise ValueError("KSC linked report mismatch: " + key)
        reference_root = Path(run["finite_reference_root"])
        reference_run = _checked_ksc_archive(reference_root)
        reference = json.loads((reference_root / "report.json").read_text())
        if (reference.get("qualified") is not True or reference.get("full_prior_integrated") is not True
                or reference.get("status") != "QUALIFIED_FINITE_POSTERIOR_REFERENCE"
                or reference.get("model_id") != "M03-KSC-T2"):
            raise ValueError("KSC finite reference is unqualified")
        for key in ("data_sha256", "selection_sha256", "endpoint_identity", "endpoint_report_sha256", "posterior_report_sha256"):
            if reference_run[key] != run[key]:
                raise ValueError("KSC finite reference binding mismatch: " + key)
        if report["endpoint_identity"] != run["endpoint_identity"] or reference["endpoint_identity"] != run["endpoint_identity"]:
            raise ValueError("KSC finite endpoint identity mismatch")
        for item in [reference["log_density"], *(item for grid in reference["grids"] for item in grid.values())]:
            if file_hash(Path(item["path"])) != item["sha256"]:
                raise ValueError("KSC finite reference tensor mismatch")
        if report["D_vs_R"] != reference["D_vs_R"]:
            raise ValueError("KSC D/R result mismatch")
        row = report["arms"]["C"]
        chunks, looks = row["chunks"], row["looks"]
        if [look["draws_per_chain"] for look in looks] not in ([], [1000], [1000, 2000]):
            raise ValueError("undeclared finite KSC statistical look")
        latest = {key: "NOT_RUN" for key in ("C/D", "C/R")}
        for look in looks:
            pending = {key for key, status in latest.items() if status not in ("EQUIVALENT", "DISCREPANCY")}
            if set(look["comparisons"]) != pending:
                raise ValueError("KSC finite comparison repeated or omitted")
            for name, comparison in look["comparisons"].items():
                status = comparison["status"]
                if status not in ("INCONCLUSIVE", "EQUIVALENT", "DISCREPANCY"):
                    raise ValueError("unknown KSC finite comparison status")
                if status != "INCONCLUSIVE" and comparison.get("diagnostic_valid") is not True:
                    raise ValueError("KSC finite conclusion has invalid uncertainty diagnostics")
                latest[name] = status
        if row["tuning"]["passed"]:
            tune = Path(row["tuning"]["manifest"])
            tuning = json.loads(tune.read_text())
            if tuning.get("passed") is not True or not tuning.get("checks") or not all(tuning["checks"].values()):
                raise ValueError("KSC finite tuning/report mismatch")
            for index, chunk in enumerate(chunks):
                archive_path = Path(chunk["manifest"])
                archive = json.loads(archive_path.read_text())
                sample = archive["raw_samples"]
                if (file_hash(archive_path.parent / sample["path"]) != sample["sha256"]
                        or archive["tuning_manifest_sha256"] != file_hash(tune)
                        or archive["health_passed"] != chunk["health_passed"]
                        or archive["continuation_manifest"] != (chunks[index-1]["manifest"] if index else None)):
                    raise ValueError("KSC finite retained archive mismatch")
            if row["status"] == "HEALTH_VETO":
                if not chunks or chunks[-1]["health_passed"] or any(not c["health_passed"] for c in chunks[:-1]):
                    raise ValueError("KSC finite health veto has no terminal failed chunk")
                latest = dict.fromkeys(latest, "HEALTH_VETO")
            elif (not looks or not chunks or any(not c["health_passed"] for c in chunks)
                  or row["retained_draws"] != 250*len(chunks)
                  or looks[-1]["draws_per_chain"] != row["retained_draws"]
                  or row["status"] != latest["C/D"]
                  or ("INCONCLUSIVE" in latest.values() and row["retained_draws"] != 2000)):
                raise ValueError("KSC finite result is not a closed declared look")
        elif row["status"] != "TUNING_REJECTED" or chunks or looks:
            raise ValueError("rejected KSC finite tuning has retained draws")
        else:
            latest = dict.fromkeys(latest, "TUNING_REJECTED")
        expected = {"B/R": "NOT_RUN", "D/R": reference["D_vs_R"]["status"], **latest}
        if report["posterior_comparisons"] != expected:
            raise ValueError("KSC finite comparison/result mismatch")
        state["comparisons"].update({key: value for key, value in expected.items() if key != "B/R"})
        state.setdefault("arms", {})["C"] = dict(status=row["status"], retained_draws=row.get("retained_draws", 0),
            tuning_passed=row["tuning"]["passed"], comparisons=latest,
            diagnostic_valid={name: result["diagnostic_valid"] for look in looks for name, result in look["comparisons"].items()})
        state.setdefault("report", str(path.resolve()))
        state.setdefault("report_sha256", file_hash(path))
        state.update(status="COMPLETED_BOUNDED_INFERENCE", finite_status=report["status"],
            finite_report=str(path.resolve()), finite_report_sha256=file_hash(path),
            closed=True, artifact_valid=True, surrogate_hmc_tested=True,
            retained_surrogate_hmc_tested=state["retained_surrogate_hmc_tested"] or any(c["health_passed"] for c in chunks),
            all_models_validated=False, next_action="Preserve closed looks; continue other ready model scopes")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(status="ARTIFACT_INVALID", artifact_valid=False, error=str(exc))
    return state


def lg1_scope(stage="P1") -> dict:
    card = {"id": "M01-LG1-T10-N1024-attempt4", "family": "M01", "stage": stage,
            "inference_attempt": 4, "mechanics_only": stage in ("P0", "P1", "P2-endpoint"), "state_dimension": 1,
            "parameter_dimension": 1, "parameters": ["phi"], "horizon": 10,
            "particle_count": 1024, "chunk_extent": 1024,
            "law": "x0~N(0,1/(1-phi^2)); xt=phi*xprev+N(0,1); yt=xt+N(0,.5)",
            "prior": "Uniform(-.95,.95)", "chart": "phi=-.95+1.9*sigmoid(z), Jacobian once",
            "truth": .8, "data_partition": "diagnostic", "data_replicate": 0,
            "force": "full canonical analytical DCTR; frozen proposal field, not admitted exact score",
            "endpoint": "same-law FP64 Kalman posterior",
            "precision_classification": "FP64_INTEGRATOR_FP32_DCTR",
            "precision": {"integrator": "float64", "momentum": "float64", "force": "float32",
                          "endpoint": "float64", "reference": "float64", "force_tf32": True,
                          "device": "GPU", "jit_compile": True},
            "controls_status": "warm_start_hypothesis_not_selected_or_admitted",
            "continuation_vetoes": ["wrong target", "nonfinite accepted state", "cross-chain coupling",
                                    "nondeterministic field", "mutating endpoint", "invalid status"],
            "explanatory_only": ["force discrepancy", "filter discrepancy", "acceptance", "runtime"],
            "comparisons": {"B_vs_R": "NOT_RUN", "C_vs_D": "NOT_RUN", "D_vs_R": "NOT_RUN", "C_vs_R": "NOT_RUN"},
            "nonclaims": ["posterior recovery", "convergence", "DCTR likelihood accuracy", "default readiness"]}
    if stage in ("P2-reference", "P2-controls"):
        card["data_partition"] = "calibration_validation_claim" if stage == "P2-reference" else "calibration_validation"
    elif stage in ("P2-tune", "P2-sample", "P2-endpoint", "P2-finite-reference", "P2-C-tune", "P2-C-sample"):
        card.update(data_partition="claim", controls_status="scoped_selection_required_and_validated_by_worker")
    if stage in ("P2-endpoint", "P2-finite-reference", "P2-C-tune", "P2-C-sample"):
        card["endpoint"] = "lg1_dctr_physical_fp32_chart_fp64_v1; qualification required"
        card["precision"]["endpoint"] = "FP32 physical density plus FP64 chart Jacobian"
    return card


def lg2_scope(stage="P3-LG2-qualify") -> dict:
    """Declared first LG2 scope; availability never confers inference status."""
    return {
        "id": "M01-LG2-T20-N1024-attempt1", "family": "M01", "model": "LG2",
        "stage": stage, "inference_attempt": 1, "data_replicate": 0,
        "state_dimension": 2, "parameter_dimension": 2, "parameters": ["a", "b"],
        "horizon": 20, "particle_count": 1024, "chunk_extent": 1024,
        "law": "stationary x0; F=a*[[.7,.2],[.1,.6]], Q=exp(2*b)*[[1,.3],[.3,.8]], H=I, R=.5*I",
        "prior": "independent Uniform(.5,1.2) and Uniform(-1,1)",
        "chart": "bounded logistic chart, uniform prior and Jacobian once",
        "truth": [1., 0.],
        "data_partition": ("calibration_validation_claim" if stage in ("P0", "all", "P3-LG2-reference")
                           else "calibration_validation" if stage == "P3-LG2-controls" else "calibration"),
        "force": "full canonical analytical DCTR; frozen imperfect proposal field",
        "endpoint": "same-law FP64 matrix Kalman posterior",
        "precision": {"integrator": "float64", "force": "float32", "force_tf32": True,
                      "endpoint": "float64", "reference": "float64", "jit_compile": True},
        "mechanics_only": stage != "P3-LG2-controls", "controls_status": "unselected_hypothesis",
        "comparisons": {"B_vs_R": "NOT_RUN", "C_vs_D": "NOT_RUN", "D_vs_R": "NOT_RUN", "C_vs_R": "NOT_RUN"},
        "remaining": ["scoped control calibration", "complete inference cost forecast", "public tuning", "retained inference"],
    }


def validated_budget(gpu: float, cpu: float) -> dict:
    if not (math.isfinite(gpu) and math.isfinite(cpu) and 0 < gpu <= GPU_CAP and 0 < cpu <= CPU_CAP):
        raise ValueError("budgets must be finite, positive and within the authorized 63 GPU / 15 CPU hours")
    return {"limits": {"gpu_seconds": gpu, "cpu_worker_seconds": cpu},
            "charged": {"gpu_seconds": 0., "cpu_worker_seconds": 0.}, "jobs": []}


def charge_previous_runs(ledger: dict, paths) -> None:
    """Carry ordinary saved accounting forward; never reset spent campaign time.

    New records contain their incremental charge separately from the cumulative
    total. The first version's ``charged`` field was incremental. No launch
    token or self-issued approval is involved; these are cost records only.
    """
    prior = []
    for path in sorted({Path(path).resolve() for path in paths}):
        payload = json.loads(path.read_text())
        charged = payload["charged_this_run"] if "charged_this_run" in payload else payload["charged"]
        amounts = {key: float(charged[key]) for key in ledger["limits"]}
        if any(not math.isfinite(value) or value < 0 for value in amounts.values()):
            raise ValueError(f"invalid prior accounting: {path}")
        for key, value in amounts.items():
            ledger["charged"][key] += value
        prior.append({"path": str(path), "sha256": file_hash(path), "charged_this_run": amounts})
    ledger["prior_runs"] = prior
    if any(ledger["charged"][key] > ledger["limits"][key] for key in ledger["limits"]):
        raise ValueError("prior campaign work already exceeds the requested budget")


def reserve_job(ledger: dict, device: str, timeout: float, cpu_allowance=None) -> None:
    if device not in {"gpu", "cpu"} or not math.isfinite(timeout) or not 0 < timeout <= 4 * 3600:
        raise ValueError("invalid device or job timeout")
    cpu = 4 * timeout if cpu_allowance is None else cpu_allowance
    if not math.isfinite(cpu) or cpu <= 0:
        raise ValueError("invalid CPU allowance")
    for key, need in (("gpu_seconds", timeout if device == "gpu" else 0.),
                      ("cpu_worker_seconds", cpu)):
        if ledger["charged"][key] + need > ledger["limits"][key]:
            raise ValueError(f"insufficient budget reservation: {key}")


def run_job(command: list[str], directory: Path, ledger: dict, *, device: str, timeout: float,
            cpu_allowance=None) -> dict:
    """Single process group, timeout and cumulative CPU watchdog; preserve logs."""
    reserve_job(ledger, device, timeout, cpu_allowance)
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_NUM_INTRAOP_THREADS="2",
                       TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    # Never auto-escalate or hide GPUs for a GPU job. The calling tool must have
    # trusted permissions. Failure to see a device is preserved as a failed job.
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {"command": command, "cwd": str(ROOT), "device": device, "timeout_seconds": timeout,
              "gpu_intentionally_hidden": device == "cpu", "status": "RUNNING",
              "cpu_allowance_seconds": cpu_allowance}
    write_json(directory / "run-manifest.json", record)
    stopped, process = None, None
    try:
        with (directory / "command.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    stopped = "WALL_TIME_LIMIT"
                try:
                    stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                    cpu_used = (int(stat[11]) + int(stat[12]) + int(stat[13]) + int(stat[14])) / os.sysconf("SC_CLK_TCK")
                    if cpu_used + ledger["charged"]["cpu_worker_seconds"] >= ledger["limits"]["cpu_worker_seconds"]:
                        stopped = "CPU_BUDGET_LIMIT"
                    if cpu_allowance is not None and cpu_used >= cpu_allowance:
                        stopped = "JOB_CPU_BUDGET_LIMIT"
                except (FileNotFoundError, ProcessLookupError):
                    pass
                if stopped:
                    break
                time.sleep(.25)
    except BaseException as exc:
        stopped = "LAUNCH_ERROR" if process is None else "INTERRUPTED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        elapsed = time.monotonic() - started
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        ledger["charged"]["gpu_seconds"] += elapsed if device == "gpu" else 0.
        ledger["charged"]["cpu_worker_seconds"] += cpu
        code = process.returncode if process is not None else None
        record.update(status=stopped or ("COMPLETED" if code == 0 else "FAILED"),
                      exit_code=code, wall_seconds=elapsed, cpu_worker_seconds=cpu,
                      gpu_seconds=elapsed if device == "gpu" else 0.)
        ledger["jobs"].append(record)
        write_json(directory / "run-manifest.json", record)
    return record


def retained_forecast(tuning_root: Path, required_arm="B") -> dict:
    """Conservative scoped forecast; numerical admission is rechecked by worker."""
    report_path = tuning_root / "report.json"
    report = json.loads(report_path.read_text())
    arms = report["arms"]
    if required_arm not in ("B", "C"):
        raise ValueError("unsupported retained comparison arm")
    if arms.get(required_arm, {}).get("passed") is not True:
        raise ValueError(f"retained surrogate comparison requires verified {required_arm} tuning")
    candidates = ("A", "B") if required_arm == "B" else ("C",)
    eligible = [arm for arm in candidates if arms.get(arm, {}).get("passed") is True]
    costs = [float(arms[arm]["wall_seconds"]) for arm in eligible]
    if any(not math.isfinite(cost) or cost <= 0 for cost in costs):
        raise ValueError("invalid scoped tuning cost")
    # Charge the entire tuning wall time against only its 1250 fresh transitions;
    # then add 16 graph compilations and analysis, with 50% timing headroom.
    compilations = 8 * len(eligible)
    wall = 1.5 * sum(costs) / 1250 * 2000 + compilations * 32 + 120
    return dict(status="MEASURED_SCOPED_FORECAST", tuning_report=str(report_path),
                tuning_report_sha256=file_hash(report_path), retained_per_chain_max=2000,
                gpu_seconds=wall, cpu_worker_seconds=1.2 * wall, eligible_arms=eligible,
                assumptions=f"measured tuning cost/1250; 50% headroom; {compilations} compilations at 32s; 120s analysis")


def validate_retained_forecast(forecast, job):
    if (forecast["gpu_seconds"] > job["timeout"]
            or forecast["cpu_worker_seconds"] > job["cpu_allowance"]):
        raise ValueError("retained forecast exceeds the bounded job budget")


def retained_attempt_history(root: Path, attempt: int) -> dict:
    """Find prior LG1 looks for reporting only, never fresh inference authority.

    Even a partial look prevents restarting an attempt from scratch. A failed
    worker before any look (and without a scientific health veto) may retry.
    """
    history = {}
    for stage, arms in (("P2-sample", ("A", "B")), ("P2-C-sample", ("C",))):
        paths = root.glob(f"docs/plans/artifacts/surrogate-hmc-all-models-*/{stage}/report.json")
        for path in sorted(paths):
            report = json.loads(path.read_text())
            scope = report.get("provenance", {}).get("scope", {})
            if (scope.get("model") != "LG1" or scope.get("attempt") != attempt
                    or report.get("inference_attempt") != attempt):
                continue
            rows = {arm: report.get("arms", {}).get(arm, {}) for arm in arms}
            if not any(row.get("looks") or row.get("status") == "HEALTH_VETO"
                       for row in rows.values()):
                continue
            key = f"{stage}-job"
            if key in history:
                raise ValueError(f"multiple retained reports for LG1 attempt {attempt}, {stage}; reconcile before launch")

            def terminal(arm, row):
                if row.get("status") in ("HEALTH_VETO", "TUNING_REJECTED"):
                    return True
                if not row.get("looks"):
                    return False
                if row.get("retained_draws", 0) >= 2000:
                    return True
                if arm == "C":
                    return all(row["looks"][-1].get(name, {}).get("status")
                               in ("EQUIVALENT", "DISCREPANCY") for name in ("C/D", "C/R"))
                return row.get("status") in ("EQUIVALENT", "DISCREPANCY")

            history[key] = dict(report=str(path.resolve()), report_sha256=file_hash(path),
                                inference_attempt=attempt, model="LG1",
                                closed=all(terminal(arm, row) for arm, row in rows.items()),
                                authorizes_new_inference=False)
    return history


def read_retained_report(path, job_name, history, outcomes, *, report_only=False):
    """Load a current or recorded report without granting sampling authority."""
    outcome = outcomes.get(job_name, {})
    prior = (history.get(job_name) if report_only else
             outcome.get("prior_retained_result")
             if outcome.get("status") == "HISTORICAL_RESULT_REPORTED" else None)
    if prior:
        path = Path(prior["report"])
        if not path.is_file() or file_hash(path) != prior["report_sha256"]:
            raise ValueError(f"recorded retained report changed: {job_name}")
    return json.loads(path.read_text()) if path.exists() else {}


def preserve_retained_attempts(jobs, history, *, all_stages):
    """Closed sampling branches are reported; unfinished branches still run."""
    both_closed = all(history.get(name, {}).get("closed") for name in
                      ("P2-sample-job", "P2-C-sample-job"))
    result = []
    for job in jobs:
        prior = history.get(job["name"])
        if all_stages and history.get("P3-C01-inference-job") and job["name"].startswith("P3-C01-"):
            if not prior:
                continue
            job = dict(job, depends_on=[])
        if all_stages and both_closed and (job["name"] == "P1-LG1" or job["name"].startswith("P2-")):
            if not prior:
                continue
            job = dict(job, depends_on=[])
        if prior:
            job = dict(job, prior_retained_result=prior)
        result.append(job)
    return result


def gaussian_attempt_history(root: Path) -> dict:
    """Report the frozen C01 attempt; never repeat its retained samples."""
    history = {}
    for path in sorted(root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/P3-C01-inference/report.json")):
        report = json.loads(path.read_text())
        if report.get("provenance", {}).get("scope") != "C01-anisotropic-gaussian-d3-attempt1":
            continue
        rows = report.get("arms", {}).values()
        if not (report.get("closed") or any(row.get("chunks") or row.get("status") == "TUNING_REJECTED"
                                           for row in rows)):
            continue
        if history:
            raise ValueError("multiple retained reports for C01 attempt 1; reconcile before launch")
        history["P3-C01-inference-job"] = dict(report=str(path.resolve()), report_sha256=file_hash(path),
            inference_attempt=1, model="C01", closed=bool(report.get("closed")), authorizes_new_inference=False)
    return history


def gaussian_remaining_allocation(root, stage):
    """Bound retries by the original aggregate C01 stage allocation."""
    wall, cpu = (300., 400.) if stage == "qualify" else (900., 1000.)
    script = ("qualify_surrogate_hmc_gaussian_controls.py" if stage == "qualify"
              else "run_surrogate_hmc_gaussian_controls.py")
    for path in root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        for job in json.loads(path.read_text()).get("jobs", []):
            if any(Path(arg).name == script for arg in job.get("command", [])):
                wall -= job["gpu_seconds"]
                cpu -= job["cpu_worker_seconds"]
    return wall, cpu


def gaussian_qualification_forecast(root, job):
    report = json.loads((root/"report.json").read_text())
    if report.get("status") != "QUALIFIED_AFFORDABLE":
        raise ValueError("C01 mechanics or forecast not qualified")
    if any(file_hash(ROOT/name) != digest for name, digest in report["provenance"]["source_hashes"].items()):
        raise ValueError("C01 qualification source drift")
    forecast = report["forecast"]["total"]
    if any(not math.isfinite(forecast[key]) or forecast[key] <= 0 or forecast[key] > bound
           for key, bound in (("wall_seconds", job["timeout"]), ("cpu_worker_seconds", job["cpu_allowance"]))):
        raise ValueError("C01 inference forecast exceeds remaining bounded allocation")
    return report["forecast"]


def sv20_reference_remaining_allocation(root, stage="reference"):
    """Charge all SV20 preparation retries to their original stage allocation."""
    if stage not in ("reference", "posterior", "qualify", "controls"):
        raise ValueError("unknown SV20 preparation stage")
    wall, cpu, script = {
        "reference": (450., 700., "prepare_surrogate_hmc_sv_reference.py"),
        "posterior": (600., 900., "prepare_surrogate_hmc_sv_posterior.py"),
        "qualify": (600., 450., "qualify_surrogate_hmc_sv_consumer.py"),
        "controls": (600., 600., "calibrate_surrogate_hmc_sv_controls.py"),
    }[stage]
    attempts = 0
    for path in root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        for job in json.loads(path.read_text()).get("jobs", []):
            if any(Path(arg).name == script for arg in job.get("command", [])):
                attempts += 1
                wall -= job["gpu_seconds"]
                cpu -= job["cpu_worker_seconds"]
    return (0., 0.) if attempts >= 2 else (wall, cpu)


def sv20_family_remaining_allocation(ledger, parent_cpu_seconds=0.):
    """Enforce the shared family ceiling, including preparation, tests and retries.

    Dedicated historical SV20 roots include their complete incremental charge.
    Mixed roots contribute SV jobs plus unallocated parent CPU overhead. This
    conservatively charges shared overhead without charging another family's
    workers. Current-run jobs must count before the accounting file is written.
    """
    remaining = {"gpu_seconds": 14400., "cpu_worker_seconds": 13000.}
    scripts = {"prepare_surrogate_hmc_sv_reference.py", "prepare_surrogate_hmc_sv_posterior.py",
               "qualify_surrogate_hmc_sv_consumer.py", "calibrate_surrogate_hmc_sv_controls.py",
               "qualify_surrogate_hmc_sv_endpoint.py", "run_surrogate_hmc_sv.py"}

    def charge(payload, dedicated=False, current=False):
        jobs = payload.get("jobs", [])
        sv_jobs = [job for job in jobs if any(Path(arg).name in scripts for arg in job.get("command", []))]
        incremental = payload.get("charged_this_run", payload.get("charged", {}))
        if dedicated:
            amounts = {key: float(incremental[key]) for key in remaining}
        else:
            amounts = {key: sum(float(job[key]) for job in sv_jobs) for key in remaining}
            if sv_jobs and not current:
                amounts["cpu_worker_seconds"] += max(0., float(incremental["cpu_worker_seconds"])
                                                      - sum(float(job["cpu_worker_seconds"]) for job in jobs))
        if any(not math.isfinite(value) or value < 0 for value in amounts.values()):
            raise ValueError("invalid SV20 family accounting")
        for key, value in amounts.items():
            remaining[key] -= value

    for prior in ledger.get("prior_runs", []):
        path = Path(prior["path"])
        charge(json.loads(path.read_text()), dedicated="-sv20-" in path.parent.name.lower())
    charge(ledger, current=True)
    if not math.isfinite(parent_cpu_seconds) or parent_cpu_seconds < 0:
        raise ValueError("invalid current parent CPU charge")
    remaining["cpu_worker_seconds"] -= parent_cpu_seconds
    return remaining


def m10_reference_state(root, additional_reports=()):
    """Combine standalone and master preparation costs; reuse checked evidence.

    A local likelihood reference cannot authorize a posterior claim. Nested
    worker accounting is excluded: the master already charges its whole child.
    """
    root = Path(root)
    script = "prepare_surrogate_hmc_nonlinear_reference.py"
    cpu = wall = family_cpu = 0.
    attempts, reports = 0, []
    for path in root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        payload = json.loads(path.read_text())
        jobs = payload.get("jobs", [])
        if "-m10-" in path.parent.name:
            charge = payload.get("charged_this_run", payload.get("charged", {}))
            family_cpu += float(charge["cpu_worker_seconds"])
        else:
            family_cpu += sum(float(job["cpu_worker_seconds"]) for job in jobs
                              if any("surrogate_hmc_nonlinear" in str(arg) for arg in job.get("command", [])))
        if "-m10-reference-" in path.parent.name and "-tests-" not in path.parent.name and not jobs:
            manifest = json.loads((path.parent/"run-manifest.json").read_text())
            if any(Path(arg).name == script for arg in manifest.get("command", [])):
                cpu += float(payload["charged_this_run"]["cpu_worker_seconds"])
                wall += float(manifest["wall_seconds"])
                attempts += 1
                reports.append(path.parent/"report.json")
        for job in jobs:
            command = job.get("command", [])
            if any(Path(arg).name == script for arg in command):
                cpu += float(job["cpu_worker_seconds"])
                wall += float(job["wall_seconds"])
                attempts += 1
                reports.append(Path(command[command.index("--output")+1])/"report.json")
    if not all(math.isfinite(v) and v >= 0 for v in (cpu, wall, family_cpu)):
        raise ValueError("invalid M10 accounting")
    qualified = None
    for path in sorted(set(reports) | {Path(p) for p in additional_reports}):
        if not path.exists():
            continue  # A killed worker is an infrastructure attempt, still charged.
        report = json.loads(path.read_text())
        if report.get("status") == "REFERENCE_CHECK_FAILED":
            raise ValueError("M10 scientific reference failure requires a prospective repair plan")
        if report.get("status") != "QUALIFIED_LOCAL_LIKELIHOOD_REFERENCE":
            continue
        required = {"mesh_refinement", "domain_refinement", "domain_loss", "posterior_edge",
                    "linear_identity", "analytical_noise_scores", "independent_nonlinear_refinement",
                    "independent_nonlinear_agreement"}
        if set(report.get("checks", {})) != required or any(v is not True for v in report["checks"].values()):
            raise ValueError("incomplete M10 reference checks")
        manifest = json.loads((path.parent/"run-manifest.json").read_text())
        for name in ("bayesfilter/testing/surrogate_hmc_nonlinear_reference.py",
                     "bayesfilter/score_study/nonlinear_tf.py", "bayesfilter/score_study/gaussian_tf.py"):
            if manifest["source_hashes"][name] != file_hash(root/name):
                raise ValueError("M10 numerical reference source changed")
        for row in report["partitions"].values():
            data_path = Path(row["data_file"])
            if not data_path.is_absolute():
                data_path = root/data_path
            if file_hash(data_path) != row["data_sha256"]:
                raise ValueError("M10 data changed")
        for row in report["grids"]:
            if file_hash(path.parent/row["values_file"]) != row["sha256"]:
                raise ValueError("M10 reference grid changed")
        qualified = dict(report=str(path.resolve()), report_sha256=file_hash(path),
                         role="local_likelihood_reference", posterior_reference_qualified=False,
                         authorizes_new_inference=False)
    return dict(wall_seconds=max(0., 300.-wall) if attempts < 2 else 0.,
                cpu_worker_seconds=max(0., min(300.-cpu, M10_CPU_CAP-family_cpu)) if attempts < 2 else 0.,
                family_cpu_remaining=max(0., M10_CPU_CAP-family_cpu), attempts=attempts, qualified=qualified)


M10_CONSUMER_CHECKS = frozenset((
    "actual_call_chain_parity", "bitwise_replay", "cache_matches_endpoint",
    "endpoint_independent", "endpoint_transition_health", "field_sign_status",
    "finite_grid_endpoint_binding", "finite_states", "independent_typed_bindings",
    "per_chain_energy", "permutation", "proposal_involution", "reference_force_binding",
    "row_independence", "unqualified_is_numerical_failure"))


def m10_consumer_state(root, *, for_requalification=False, additional_reports=()):
    """Account for whole master children once and reuse only checked mechanics."""
    root = Path(root)
    script = "qualify_surrogate_hmc_nonlinear_consumer.py"
    family = dict(gpu_seconds=0., cpu_worker_seconds=0.)
    stage = dict(family)
    reports = []
    for path in root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        payload = json.loads(path.read_text())
        jobs = payload.get("jobs", [])
        charge = payload.get("charged_this_run", payload.get("charged", {}))
        if "-m10-" in path.parent.name:
            for key in family:
                family[key] += float(charge[key])
        else:
            for job in jobs:
                if any("surrogate_hmc_nonlinear" in str(arg) for arg in job.get("command", [])):
                    for key in family:
                        family[key] += float(job.get(key, 0.))
        if "-m10-consumer-" in path.parent.name and "-tests-" not in path.parent.name and not jobs:
            manifest = json.loads((path.parent/"run-manifest.json").read_text())
            if any(Path(arg).name == script for arg in manifest.get("command", [])):
                reports.append(path.parent/"report.json")
                for key in stage:
                    stage[key] += float(charge[key])
        for job in jobs:
            command = job.get("command", [])
            if any(Path(arg).name == script for arg in command):
                reports.append(Path(command[command.index("--output")+1])/"report.json")
                for key in stage:
                    stage[key] += float(job[key])
    if not all(math.isfinite(v) and v >= 0 for v in (*family.values(), *stage.values())):
        raise ValueError("invalid M10 consumer accounting")
    qualified = None
    superseded = []
    for path in sorted(set(reports) | {Path(p) for p in additional_reports}):
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report.get("status") == "QUALIFICATION_FAILED":
            raise ValueError("M10 mechanics failure requires a prospective repair plan")
        if report.get("status") != "QUALIFIED_CONSUMER_MECHANICS":
            continue
        if (report.get("qualified") is not True or set(report.get("checks", {})) != M10_CONSUMER_CHECKS
                or any(v is not True for v in report["checks"].values())):
            raise ValueError("incomplete M10 mechanics checks")
        manifest = json.loads((path.parent/"run-manifest.json").read_text())
        # Bind the numerical implementation, independently of later orchestration edits.
        required = {"bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py",
                    "bayesfilter/inference/neural_force_hmc.py"}
        sources = manifest["source_hashes"]
        if not required <= sources.keys():
            raise ValueError("missing M10 mechanics source provenance")
        if any(name.startswith("bayesfilter/") and file_hash(root/name) != digest
               for name, digest in sources.items()):
            superseded.append(str(path.resolve()))
            continue
        if file_hash(Path(manifest["data_file"])) != manifest["data_sha256"]:
            raise ValueError("M10 consumer data changed")
        if file_hash(Path(manifest["parent_reference"])/"run-manifest.json") != manifest["parent_manifest_sha256"]:
            raise ValueError("M10 consumer reference binding changed")
        qualified = dict(report=str(path.resolve()), report_sha256=file_hash(path),
            role="consumer_mechanics", posterior_reference_qualified=False, authorizes_new_inference=False)
    if superseded and qualified is None and not for_requalification:
        raise ValueError("M10 numerical consumer source changed; requalification required")
    attempts = len(reports)
    return dict(attempts=attempts, qualified=qualified,
        superseded_mechanics=superseded,
        family_remaining={"gpu_seconds":max(0.,7200.-family["gpu_seconds"]),
                          "cpu_worker_seconds":max(0.,M10_CPU_CAP-family["cpu_worker_seconds"])},
        wall_seconds=max(0., min(600.-stage["gpu_seconds"], 7200.-family["gpu_seconds"])) if attempts < 2 else 0.,
        cpu_worker_seconds=max(0., min(450.-stage["cpu_worker_seconds"], M10_CPU_CAP-family["cpu_worker_seconds"])) if attempts < 2 else 0.)


M10_POSTERIOR_CHECKS = frozenset(("finite_positive", "function_precision",
    "marginal_interpolation", "normalization_refinement", "parameter_contraction",
    "predictive_probabilities"))


def m10_posterior_state(root, additional_reports=()):
    """Reuse a checked independent R, without issuing HMC or tuning authority."""
    root = Path(root)
    script = "prepare_surrogate_hmc_nonlinear_posterior.py"
    cpu = wall = gpu = 0.
    reports = []
    for path in root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        payload = json.loads(path.read_text())
        jobs = payload.get("jobs", [])
        charge = payload.get("charged_this_run", payload.get("charged", {}))
        if "-m10-posterior-" in path.parent.name and "-tests-" not in path.parent.name and not jobs:
            manifest = json.loads((path.parent/"run-manifest.json").read_text())
            if any(Path(arg).name == script for arg in manifest.get("command", [])):
                cpu += float(charge["cpu_worker_seconds"])
                gpu += float(charge["gpu_seconds"])
                wall += float(manifest["wall_seconds"])
                reports.append(path.parent/"report.json")
        for job in jobs:
            command = job.get("command", [])
            if any(Path(arg).name == script for arg in command):
                cpu += float(job["cpu_worker_seconds"])
                gpu += float(job["gpu_seconds"])
                wall += float(job["wall_seconds"])
                reports.append(Path(command[command.index("--output")+1])/"report.json")
    if not all(math.isfinite(v) and v >= 0 for v in (cpu, wall, gpu)):
        raise ValueError("invalid M10 posterior accounting")
    qualified = None
    for path in sorted(set(reports) | {Path(p) for p in additional_reports}):
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "QUALIFIED_MODEL_POSTERIOR_REFERENCE":
            continue  # Failed numerical candidates remain preserved and charged.
        if (report.get("qualified") is not True or report.get("model") != "M10"
                or report.get("partition") != "claim-nominal"
                or set(report.get("checks", {})) != M10_POSTERIOR_CHECKS
                or any(v is not True for v in report["checks"].values())):
            raise ValueError("incomplete M10 posterior checks")
        margins = [.1]*5+[.02]*6+[.03]*2
        allowance = report.get("function_allowance", [])
        if (report.get("margins") != margins or len(allowance) != 13
                or not all(isinstance(a, (int, float)) and math.isfinite(a) and 0 <= a <= m/10
                           for a, m in zip(allowance, margins))
                or report.get("allowance_is_rigorous_bound") is not False):
            raise ValueError("invalid M10 posterior precision evidence")
        manifest = json.loads((path.parent/"run-manifest.json").read_text())
        required = {"bayesfilter/testing/surrogate_hmc_nonlinear_posterior_reference.py",
                    "bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py",
                    "bayesfilter/testing/surrogate_hmc_nonlinear_reference.py",
                    "bayesfilter/score_study/nonlinear_tf.py", "bayesfilter/score_study/gaussian_tf.py"}
        sources = manifest["source_hashes"]
        if not required <= sources.keys():
            raise ValueError("missing M10 posterior numerical provenance")
        for name, digest in sources.items():
            if name.startswith("bayesfilter/") and file_hash(root/name) != digest:
                raise ValueError("M10 posterior numerical source changed")
        if file_hash(Path(manifest["data_file"])) != manifest["data_sha256"]:
            raise ValueError("M10 posterior data changed")
        grids = report.get("grids", [])
        expected = {(40,20,128), (64,20,128), (96,20,128), (96,32,128), (96,32,64)}
        if len(grids) != 5 or {(r["order"],r["state_order"],r["normal_order"]) for r in grids} != expected:
            raise ValueError("incomplete M10 posterior refinement grids")
        if any(file_hash(Path(row["path"])) != row["sha256"] for row in grids):
            raise ValueError("M10 posterior grid changed")
        qualified = dict(report=str(path.resolve()), report_sha256=file_hash(path),
            role="independent_model_posterior_reference", posterior_reference_qualified=True,
            authorizes_new_inference=False)
    attempts = len(reports)
    family = m10_consumer_state(root)["family_remaining"]
    return dict(attempts=attempts, qualified=qualified,
        wall_seconds=max(0., min(800.-wall, 800.-gpu, family["gpu_seconds"])) if attempts < 5 else 0.,
        cpu_worker_seconds=max(0., min(850.-cpu, family["cpu_worker_seconds"])) if attempts < 5 else 0.)


def m10_control_attempts(root):
    """Count standalone and master-launched control work exactly once."""
    root = Path(root)
    script = "calibrate_surrogate_hmc_nonlinear_controls.py"
    spent = dict(cpu_worker_seconds=0., gpu_seconds=0.)
    reports = []
    for path in root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"):
        payload = json.loads(path.read_text())
        jobs = payload.get("jobs", [])
        manifest_path = path.parent/"run-manifest.json"
        if not jobs and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(Path(arg).name == script for arg in manifest.get("command", [])):
                reports.append(path.parent/"report.json")
                charge = payload.get("charged_this_run", payload.get("charged", {}))
                for key in spent:
                    spent[key] += float(charge[key])
        for job in jobs:
            command = job.get("command", [])
            if any(Path(arg).name == script for arg in command):
                reports.append(Path(command[command.index("--output")+1])/"report.json")
                for key in spent:
                    spent[key] += float(job[key])
    if not all(math.isfinite(v) and v >= 0 for v in spent.values()):
        raise ValueError("invalid M10 control accounting")
    return dict(spent=spent, reports=sorted(set(reports)))


def m10_controls_state(root):
    """Reuse scoped protection/cost evidence without importing TensorFlow."""
    from bayesfilter.inference.surrogate_hmc_dctr_selection import (
        SCHEMA, USE, MULTIPLIERS, first_stable_pair)
    root = Path(root)
    attempts = m10_control_attempts(root)
    spent, reports = attempts["spent"], attempts["reports"]
    qualified = None
    for path in reports:
        if not path.exists():
            raise ValueError("missing M10 control report requires interpretation before retry")
        report = json.loads(path.read_text())
        if report.get("status") != "SELECTED_FOR_PROPOSAL_EXPERIMENT":
            raise ValueError("M10 control attempt requires interpretation before retry")
        selection_path = path.parent/"selection.json"
        selection = json.loads(selection_path.read_text())
        manifest = json.loads((path.parent/"run-manifest.json").read_text())
        if (selection != report.get("selected") or selection.get("schema") != SCHEMA
                or selection.get("use") != USE or selection.get("exact_score_admission") is not False
                or selection.get("default_readiness") is not False
                or selection.get("scope") != report.get("scope")
                or selection.get("source_hashes") != manifest.get("source_hashes")):
            raise ValueError("M10 control selection/report mismatch")
        scope = selection["scope"]
        expected = dict(model="M10", attempt=1, horizon=2, particle_count=1024,
            dtype="float32", tf32=True, jit_compile=True, state_dimension=1, parameter_dimension=2,
            reset="contract_e_chol_v1", chunk_policy="dpf_transport_exact_divisor_cap3000_v1", chunk_size=1024)
        if any(scope.get(key) != value for key,value in expected.items()):
            raise ValueError("wrong M10 control scope")
        issuer = "bayesfilter/inference/surrogate_hmc_dctr_selection.py"
        sources = manifest["source_hashes"]
        if not {issuer, "bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py"} <= sources.keys():
            raise ValueError("missing M10 control numerical sources")
        if (file_hash(root/issuer) != selection.get("issuer_sha256")
                or any(name.startswith("bayesfilter/") and file_hash(root/name) != digest
                       for name,digest in sources.items())):
            raise ValueError("M10 control numerical source changed")
        curves = {}
        for group in ("evidence", "partition_data"):
            if set(selection[group]) != {"calibration", "validation"}:
                raise ValueError("M10 control partition mismatch")
            for partition, binding in selection[group].items():
                if file_hash(Path(binding["path"])) != binding["sha256"]:
                    raise ValueError("M10 control evidence changed")
                data = json.loads(Path(binding["path"]).read_text())
                if group == "evidence":
                    if data.get("scope") != scope or data.get("source_hashes") != sources:
                        raise ValueError("M10 curve scope/source mismatch")
                    if data["data_sha256"] != selection["partition_data"][partition]["sha256"]:
                        raise ValueError("M10 curve data mismatch")
                    if len(data["arms"]) != len(MULTIPLIERS):
                        raise ValueError("incomplete M10 control ladder")
                    for arm_index, arm in enumerate(data["arms"]):
                        if file_hash(Path(arm["path"])) != arm["sha256"]:
                            raise ValueError("M10 control arm changed")
                        measured = json.loads(Path(arm["path"]).read_text())
                        if measured.get("multiplier") != MULTIPLIERS[arm_index]:
                            raise ValueError("wrong M10 control arm order")
                        if arm_index == selection["selected_index"] and measured.get("controls") != selection.get("controls"):
                            raise ValueError("M10 selected controls differ from measured arm")
                    curves[partition] = data["pairs"]
                else:
                    if (data.get("model_id"),data.get("attempt"),data.get("partition")) != ("M10",1,partition):
                        raise ValueError("wrong M10 control data role")
                    if set(data["original_datasets"]) != {"nominal", "stress"}:
                        raise ValueError("missing M10 control regime")
                    for regime, original in data["original_datasets"].items():
                        if file_hash(Path(original["path"])) != original["sha256"]:
                            raise ValueError("M10 original control data changed")
                        measured = json.loads(Path(original["path"]).read_text())
                        if (measured.get("role"), measured.get("regime"), measured.get("horizon"),
                                measured.get("dtype")) != (partition, regime, 2, "float32"):
                            raise ValueError("wrong M10 original control data role")
                        if data.get("data", {}).get(regime) != measured:
                            raise ValueError("M10 wrapped control data differs from original")
        index = first_stable_pair(curves)
        if (index is None or selection.get("selected_index") != index
                or selection.get("selected_multiplier") != MULTIPLIERS[index]):
            raise ValueError("M10 selection does not follow measured non-harm rule")
        if report.get("forecast", {}).get("authorizes_inference") is not False:
            raise ValueError("M10 optimistic forecast cannot authorize inference")
        qualified = dict(report=str(path.resolve()), report_sha256=file_hash(path),
            role="scoped_control_and_cost_preparation", authorizes_new_inference=False)
    family = m10_consumer_state(root)["family_remaining"]
    return dict(qualified=qualified, attempts=len(reports),
        wall_seconds=max(0., min(450.-spent["gpu_seconds"], family["gpu_seconds"])),
        cpu_worker_seconds=max(0., min(350.-spent["cpu_worker_seconds"], family["cpu_worker_seconds"])))


def m10_endpoint_state(root):
    """Preserve bounded endpoint preparation; this never reports retained HMC."""
    script = "qualify_surrogate_hmc_nonlinear_endpoint.py"
    reports, spent = {}, dict(cpu_worker_seconds=0., gpu_seconds=0.)
    for path in sorted((root/"docs/plans/artifacts").glob("surrogate-hmc-all-models-*/accounting.json")):
        accounting = json.loads(path.read_text())
        manifest_path = path.parent/"run-manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(Path(arg).name == script for arg in manifest.get("command", [])):
                reports[path.parent/"report.json"] = accounting.get("charged_this_run", accounting.get("charged", {}))
        for job in accounting.get("jobs", []):
            command = job.get("command", [])
            if any(Path(arg).name == script for arg in command):
                reports[Path(command[command.index("--output")+1])/"report.json"] = job
    qualified, latest = None, None
    for path, charge in reports.items():
        for key in spent:
            value = float(charge[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid M10 endpoint accounting")
            spent[key] += value
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        latest = dict(report=str(path.resolve()), status=report.get("status"))
        if not report.get("qualified"):
            continue
        manifest = json.loads((path.parent/"run-manifest.json").read_text())
        if (manifest.get("exit_code") != 0 or report.get("status") != "QUALIFIED_INFERENCE_ENDPOINT"
                or not report.get("checks") or not all(v is True for v in report["checks"].values())
                or report.get("retained_inference") is not False):
            raise ValueError("invalid M10 endpoint qualification")
        required = {"reference_finite", "reference_refinement", "density_domain_parity",
            "underflow_log_density_recovered", "total_reference_derivative", "force_replay",
            "force_health", "claim_binding", "canonical_call_chain", "row_independence",
            "reference_binding", "independent_force_endpoint", "potential_replay", "reversal",
            "complete_L1_transitions", "complete_L2_transitions"}
        if not required.issubset(report["checks"]) or set(manifest["inputs"]) != {
                "claim", "calibration", "selection", "posterior_report", "reference_report"}:
            raise ValueError("missing M10 endpoint evidence")
        from scripts.qualify_surrogate_hmc_nonlinear_endpoint import complete_forecast
        # Recompute from the preserved measurements. The first report counted
        # only one nomination candidate; retain that report, supersede its cost
        # estimate with the complete two-candidate count before any launch.
        forecast = complete_forecast(report["complete_transition_costs"], report["timings"])
        if not manifest.get("source_hashes") or report.get("source_binding") != manifest["source_hashes"]:
            raise ValueError("M10 endpoint source binding missing")
        for name, digest in manifest["source_hashes"].items():
            if file_hash(path.parent/"source"/name) != digest:
                raise ValueError("M10 endpoint source archive changed")
            if name.startswith("bayesfilter/") and name not in (
                    "bayesfilter/inference/surrogate_hmc_program.py", "bayesfilter/inference/__init__.py"):
                if file_hash(root/name) != digest:
                    raise ValueError("M10 endpoint numerical source changed: "+name)
        for record in manifest["inputs"].values():
            if file_hash(Path(record["path"])) != record["sha256"]:
                raise ValueError("M10 endpoint input changed")
        qualified = dict(report=str(path.resolve()), report_sha256=file_hash(path),
            role="inference_endpoint_preparation", authorizes_new_inference=False,
            complete_forecast=forecast, archived_forecast_superseded=report.get("forecast") != forecast)
    return dict(qualified=qualified, latest=latest, spent=spent,
        cpu_worker_seconds=max(0., 360.-spent["cpu_worker_seconds"]),
        wall_seconds=max(0., 480.-spent["gpu_seconds"]))


def m10_inference_history(root):
    """Report a preserved M10 B/R attempt; never authorize another closed look."""
    state = dict(status="NOT_RUN", surrogate_hmc_tested=False,
        retained_surrogate_hmc_tested=False, authorizes_new_inference=False,
        comparisons=dict.fromkeys(("B/R", "C/D", "D/R", "C/R"), "NOT_RUN"))
    artifacts = Path(root)/"docs/plans/artifacts"
    candidates = sorted(set(artifacts.glob("surrogate-hmc-all-models-*-m10-inference-*/run-manifest.json")) |
        set(artifacts.glob("surrogate-hmc-all-models-*/P3-M10-inference/run-manifest.json")))
    manifests = []
    for path in candidates:
        if "-tests-" in str(path.relative_to(artifacts)):
            continue
        metadata = json.loads(path.read_text())
        # A master manifest can exist before its child starts. It is accounting
        # for orchestration, not evidence of a previous statistical attempt.
        if path.parent.name == "P3-M10-inference" or any(
                Path(arg).name == "run_surrogate_hmc_nonlinear.py" for arg in metadata.get("command", [])):
            manifests.append(path)
    if not manifests:
        return state
    state["attempt_manifests"] = [str(path.resolve()) for path in manifests]
    if len(manifests) != 1:
        state.update(status="RECONCILIATION_REQUIRED", error="Multiple M10 inference attempts require reconciliation")
        return state
    directory = manifests[0].parent
    path = directory/"report.json"
    try:
        run = json.loads(manifests[0].read_text())
        report = json.loads(path.read_text())
        state.update(report=str(path.resolve()), report_sha256=file_hash(path))
        if run.get("scope") != "M10-T2-N1024" or report.get("model_id") != "M10-T2":
            raise ValueError("wrong M10 inference scope")
        if "exit_code" not in run:
            state.update(status="RUNNING", closed=False)
            return state
        if run["exit_code"] != 0 or report.get("status") != "COMPLETED_BOUNDED_INFERENCE":
            state.update(status=report.get("status", "INCOMPLETE_ATTEMPT"), closed=True,
                next_action="Inspect preserved M10 attempt before a budgeted repair; do not repeat closed looks")
            return state
        accounting = json.loads((directory/"accounting.json").read_text())
        if accounting.get("exit_code") != 0 or not run.get("source_hashes") or run.get("source_drift"):
            raise ValueError("M10 inference lacks successful accounting or source provenance")
        # Later reporting changes must not invalidate an otherwise preserved run.
        for name, digest in run["source_hashes"].items():
            if file_hash(directory/"source"/name) != digest:
                raise ValueError("archived M10 inference source changed: "+name)
        if file_hash(directory/"data.json") != run["data_sha256"]:
            raise ValueError("M10 inference data changed")
        if set(run["inputs"]) != {"claim", "calibration", "selection", "posterior_report", "reference_report", "endpoint_report"}:
            raise ValueError("missing frozen M10 inference inputs")
        for record in run["inputs"].values():
            if file_hash(Path(record["path"])) != record["sha256"]:
                raise ValueError("frozen M10 inference input changed")
        if json.loads((directory/"data.json").read_text()) != json.loads(Path(run["inputs"]["claim"]["path"]).read_text()):
            raise ValueError("M10 retained data differs from qualified claim")
        if run.get("retained_looks") != [1000, 2000] or run.get("function_count") != 13 or run.get("comparison_families") != 4:
            raise ValueError("M10 inference evidence contract changed")
        row = report["arms"]["B"]
        chunks, looks = row["chunks"], row["looks"]
        if [look["draws_per_chain"] for look in looks] not in ([], [1000], [1000, 2000]):
            raise ValueError("undeclared or repeated M10 statistical look")
        if any(look["status"] not in ("INCONCLUSIVE", "EQUIVALENT", "DISCREPANCY") or
               (look["status"] != "INCONCLUSIVE" and look.get("diagnostic_valid") is not True) for look in looks):
            raise ValueError("invalid M10 uncertainty conclusion")
        if any(look["status"] != "INCONCLUSIVE" for look in looks[:-1]):
            raise ValueError("M10 continued after a closed statistical look")
        if row["tuning"]["passed"]:
            tune_path = Path(row["tuning"]["manifest"])
            tune = json.loads(tune_path.read_text())
            binding = json.loads((directory/"binding.json").read_text())
            if (tune.get("schema") != "bayesfilter.verified_tensorflow_hmc.v1" or
                    tune.get("artifact_authority") is not True or tune.get("handoff_eligible") is not True or
                    tune.get("passed") is not True or not tune.get("checks") or not all(tune["checks"].values()) or
                    tune["runner_binding"] != binding or binding["target_scope"] != "M10-T2-N1024-claim-attempt1:B"):
                raise ValueError("M10 public tuning or binding mismatch")
            if file_hash(tune_path.parent/"nomination/manifest.json") != tune["nomination_manifest_sha256"]:
                raise ValueError("M10 nomination archive changed")
            for phase in ("warmup", "verification"):
                for record in tune[phase].values():
                    if file_hash(tune_path.parent/record["path"]) != record["sha256"]:
                        raise ValueError("M10 tuning trace changed")
            for index, chunk in enumerate(chunks):
                archive_path = Path(chunk["manifest"])
                archive = json.loads(archive_path.read_text())
                previous = chunks[index-1]["manifest"] if index else None
                if (archive["tuning_manifest_sha256"] != file_hash(tune_path) or
                        archive["binding_hash"] != binding["binding_hash"] or
                        archive["health_passed"] != chunk["health_passed"] or
                        archive["health_passed"] != (bool(archive["checks"]) and all(archive["checks"].values())) or
                        archive["continuation_manifest"] != previous or
                        archive["config"]["continuation_manifest"] != previous or
                        archive["config"]["num_results"] != 250 or
                        archive["config"]["seed"] != [20260922, 33000+index]):
                    raise ValueError("M10 retained archive, seed or continuation mismatch")
                for record in [archive["raw_samples"], archive["initial_state"], *archive["chunk"].values()]:
                    if file_hash(archive_path.parent/record["path"]) != record["sha256"]:
                        raise ValueError("M10 retained sample or trace changed")
            if row["status"] == "HEALTH_VETO":
                if (not chunks or chunks[-1]["health_passed"] or any(not c["health_passed"] for c in chunks[:-1]) or
                        (looks and looks[-1]["status"] != "INCONCLUSIVE")):
                    raise ValueError("M10 health veto lacks a terminal failed chunk")
            elif (not looks or not chunks or any(not c["health_passed"] for c in chunks) or
                    row["retained_draws"] != 250*len(chunks) or looks[-1]["draws_per_chain"] != row["retained_draws"] or
                    row["status"] != looks[-1]["status"] or
                    (row["status"] == "INCONCLUSIVE" and row["retained_draws"] != 2000)):
                raise ValueError("M10 retained result is not a closed declared look")
        elif row["status"] != "TUNING_REJECTED" or chunks or looks:
            raise ValueError("M10 rejected tuning contains retained draws")
        expected = dict(state["comparisons"], **{"B/R": row["status"]})
        if report["comparisons"] != expected:
            raise ValueError("M10 result/comparison mismatch")
        state.update(status="COMPLETED_BOUNDED_INFERENCE", closed=True, artifact_valid=True,
            comparisons=expected, surrogate_hmc_tested=True, retained_surrogate_hmc_tested=bool(chunks),
            arms={"B": dict(status=row["status"], retained_draws=row.get("retained_draws", 0),
                tuning_passed=row["tuning"]["passed"], diagnostic_valid=looks[-1]["diagnostic_valid"] if looks else None)},
            run_manifest=str(manifests[0].resolve()),
            next_action="Preserve closed B/R looks; C/D needs an independently qualified finite-target reference")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(status="ARTIFACT_INVALID", artifact_valid=False, error=str(exc))
    return state


def execute_job_graph(jobs, output, ledger, before_job=None):
    """Continue independent branches; block descendants of failed prerequisites."""
    seen = set()
    for job in jobs:
        if (not job.get("name") or job["name"] in seen
                or any(name not in seen for name in job.get("depends_on", ()))):
            raise ValueError("invalid dependency graph: duplicate, missing or forward prerequisite")
        seen.add(job["name"])
    outcomes = {}
    for job in jobs:
        dependencies = job.get("depends_on", [])
        blocked = [name for name in dependencies if outcomes[name]["status"] != "COMPLETED"]
        record = dict(name=job["name"], depends_on=dependencies)
        phase = "PREFLIGHT"
        try:
            if job.get("prior_preparation_result"):
                prior = job["prior_preparation_result"]
                if file_hash(Path(prior["report"])) != prior["report_sha256"]:
                    raise ValueError("prior preparation report changed after discovery")
                record.update(status="COMPLETED", prior_preparation_result=prior,
                              reused_preparation=True, new_sampling=False, authorizes_new_inference=False)
            elif job.get("prior_retained_result"):
                prior = job["prior_retained_result"]
                if file_hash(Path(prior["report"])) != prior["report_sha256"]:
                    raise ValueError("prior retained report changed after discovery")
                record.update(status="HISTORICAL_RESULT_REPORTED", prior_retained_result=prior,
                              new_sampling=False, authorizes_new_inference=False)
            elif blocked:
                record.update(status="SKIPPED_PREREQUISITE", blocked_by=blocked)
            else:
                if before_job is not None:
                    record["preflight"] = before_job(job)
                phase = "LAUNCH"
                record.update(run_job(job["command"], output / job["name"], ledger,
                                      device=job["device"], timeout=job["timeout"],
                                      cpu_allowance=job.get("cpu_allowance")))
        except KeyboardInterrupt:
            record.update(status="INTERRUPTED", error="KeyboardInterrupt")
            raise
        except Exception as exc:
            record.update(status="BLOCKED_PREFLIGHT" if phase == "PREFLIGHT" else "LAUNCH_ERROR",
                          error=f"{type(exc).__name__}: {exc}")
        finally:
            outcomes[job["name"]] = record
            write_json(output / "execution-results.json", outcomes)
    return outcomes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("P0", "P1", "P2-reference", "P2-controls", "P2-tune", "P2-sample", "P2-endpoint", "P2-finite-reference", "P2-C-tune", "P2-C-sample", "P3-LG2-reference", "P3-LG2-qualify", "P3-LG2-controls", "P3-LG2-endpoint", "P3-C01-qualify", "P3-C01-inference", "P3-SV20-reference", "P3-SV20-posterior", "P3-SV20-qualify", "P3-SV20-controls", "P3-SV20-inference", "P3-M10-reference", "P3-M10-qualify", "P3-M10-posterior", "P3-M10-controls", "P3-M10-endpoint", "P3-M10-inference", "P3-KSC-inference", "P3-KSC-finite-qualify", "P3-KSC-finite-reference", "P3-KSC-finite-inference", "all"), default="all")
    parser.add_argument("--scope", choices=("all", "LG1", "LG2", "C01", "SV20", "M10", "KSC"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gpu-seconds", type=float, default=GPU_CAP)
    parser.add_argument("--max-cpu-worker-seconds", type=float, default=CPU_CAP)
    parser.add_argument("--probe-timeout", type=float, default=900.)
    parser.add_argument("--reference-root", type=Path, help="Prepared reference root for the selected standalone scope")
    parser.add_argument("--posterior-root", type=Path, help="Qualified SV20 full-prior posterior reference")
    parser.add_argument("--qualification-root", type=Path, help="Prior mechanics qualification for the selected standalone scope")
    parser.add_argument("--cost-root", type=Path, help="Checked KSC complete-transition timing root")
    parser.add_argument("--selection", type=Path, help="Scoped DCTR control selection for P2-tune")
    parser.add_argument("--tuning-timeout", type=float, default=2400.)
    parser.add_argument("--tuning-root", type=Path, help="Verified tuning root for the requested sampling stage")
    parser.add_argument("--tuning-cpu-seconds", type=float, default=3000.)
    parser.add_argument("--sampling-timeout", type=float, default=3600.)
    parser.add_argument("--sampling-cpu-seconds", type=float, default=4800.)
    parser.add_argument("--endpoint-timeout", type=float, default=300.)
    parser.add_argument("--endpoint-root", type=Path, help="Qualified named finite endpoint")
    parser.add_argument("--finite-reference-root", type=Path, help="Qualified independent finite-target quadrature")
    parser.add_argument("--previous-accounting", type=Path, action="append", default=[],
                        help="Additional cost records outside the standard campaign artifact roots")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "P3-LG2-endpoint":
        if args.scope != "LG2":
            parser.error("P3-LG2-endpoint requires --scope LG2")
        supplied = list(argv) if argv is not None else sys.argv[1:]
        allowed = {"--stage", "--scope", "--output", "--dry-run"}
        if any(token.startswith("--") and token.split("=", 1)[0] not in allowed for token in supplied):
            parser.error("LG2 endpoint qualification uses its frozen inputs and bounded reservation")
        from scripts.qualify_surrogate_hmc_linear_endpoint import main as qualify_lg2
        return qualify_lg2(["--output", str(args.output)] + (["--dry-run"] if args.dry_run else []))
    if args.stage == "P3-SV20-inference":
        if args.scope != "SV20":
            parser.error("P3-SV20-inference requires --scope SV20")
        supplied = list(argv) if argv is not None else sys.argv[1:]
        allowed = {"--stage", "--scope", "--output", "--dry-run"}
        if any(token.startswith("--") and token.split("=", 1)[0] not in allowed for token in supplied):
            parser.error("SV20 inference uses its frozen inputs and complete campaign reservation")
        from scripts.run_surrogate_hmc_sv import main as run_sv
        return run_sv(["--output", str(args.output)] + (["--dry-run"] if args.dry_run else []))
    if args.stage in ("P3-KSC-inference", "P3-KSC-finite-qualify", "P3-KSC-finite-reference", "P3-KSC-finite-inference") or (args.scope == "KSC" and args.stage == "all"):
        finite_preparation = args.stage in ("P3-KSC-finite-qualify", "P3-KSC-finite-reference")
        finite_reference = args.stage == "P3-KSC-finite-reference"
        finite_inference = args.stage == "P3-KSC-finite-inference"
        if args.scope not in ("all", "KSC"):
            parser.error("KSC inference requires scope KSC or all")
        required = (args.qualification_root, args.selection, args.posterior_root)
        if any(value is None for value in required) or (not (finite_preparation or finite_inference) and args.cost_root is None):
            parser.error("KSC inference requires qualification-root, selection, posterior-root and cost-root")
        if (finite_reference or finite_inference) and args.endpoint_root is None:
            parser.error("KSC finite reference/inference requires endpoint-root")
        if finite_inference and args.finite_reference_root is None:
            parser.error("KSC finite inference requires finite-reference-root")
        if args.selection.name != "selection.json":
            parser.error("KSC selection must name the actual selection.json artifact")
        supplied = list(sys.argv[1:] if argv is None else argv)
        unsupported = ("--probe-timeout", "--reference-root", "--tuning-timeout", "--tuning-root",
                       "--tuning-cpu-seconds", "--sampling-timeout", "--sampling-cpu-seconds",
                       "--endpoint-timeout")
        if not finite_inference:
            unsupported += ("--finite-reference-root",)
        if not (finite_reference or finite_inference):
            unsupported += ("--endpoint-root",)
        if any(token.split("=", 1)[0] in unsupported for token in supplied):
            parser.error("KSC uses the frozen joint tuning/sampling reservation; generic per-stage overrides do not apply")
        if finite_preparation or finite_inference:
            if args.cost_root is not None:
                parser.error("finite preparation uses endpoint qualification costs; cost-root does not apply")
            if finite_inference:
                from scripts.run_surrogate_hmc_ksc_finite import main as run_ksc
            else:
                from scripts.prepare_surrogate_hmc_ksc_finite import main as run_ksc
        else:
            from scripts.run_surrogate_hmc_ksc import main as run_ksc
        forwarded = ["--output", str(args.output), "--qualification-root", str(args.qualification_root),
                     "--controls-root", str(args.selection.parent), "--posterior-root", str(args.posterior_root),
                     "--max-cpu-worker-seconds", str(args.max_cpu_worker_seconds),
                     "--max-gpu-seconds", str(args.max_gpu_seconds)]
        if finite_inference:
            forwarded += ["--endpoint-root", str(args.endpoint_root), "--finite-reference-root", str(args.finite_reference_root)]
        elif finite_preparation:
            forwarded += ["--stage", "reference" if finite_reference else "qualify"]
            if finite_reference:
                forwarded += ["--endpoint-root", str(args.endpoint_root)]
        else:
            forwarded += ["--cost-root", str(args.cost_root)]
        for path in args.previous_accounting:
            forwarded += ["--previous-accounting", str(path)]
        if args.dry_run:
            forwarded.append("--dry-run")
        return run_ksc(forwarded)
    if args.scope == "KSC" and args.stage != "P0":
        parser.error("KSC supports P0, P3-KSC-inference and the three P3-KSC-finite stages")
    if args.scope not in ("all", "LG1") and (args.stage == "P1" or args.stage.startswith("P2-")):
        parser.error("P1/P2 stages belong to LG1")
    for scope in ("LG2", "C01", "SV20", "M10"):
        if args.stage.startswith(f"P3-{scope}-") and args.scope not in ("all", scope):
            parser.error(f"P3-{scope} stages require scope {scope} or all")
    include_lg1 = args.scope in ("all", "LG1") and not args.stage.startswith("P3-")
    include_lg2 = args.scope in ("all", "LG2") and (args.stage in ("P0", "all") or args.stage.startswith("P3-LG2-"))
    include_c01 = args.scope in ("all", "C01") and (args.stage in ("P0", "all") or args.stage.startswith("P3-C01-"))
    include_sv20 = args.scope in ("all", "SV20") and (args.stage in ("P0", "all") or args.stage.startswith("P3-SV20-"))
    include_m10 = args.scope in ("all", "M10") and (args.stage in ("P0", "all") or args.stage.startswith("P3-M10-"))
    include_ksc = args.scope in ("all", "KSC") and args.stage in ("P0", "all")
    ksc_history = ksc_inference_history(ROOT) if include_ksc else {}
    ledger = validated_budget(args.max_gpu_seconds, args.max_cpu_worker_seconds)
    charge_previous_runs(ledger, [*ROOT.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"),
                                 *args.previous_accounting])
    prior_charged = dict(ledger["charged"])
    output = args.output.resolve()
    jobs = []
    if include_lg1 and args.stage in ("P1", "all"):
        jobs.append(dict(name="P1-LG1", device="gpu", timeout=args.probe_timeout,
                         command=[sys.executable, str(ROOT / "scripts/surrogate_hmc_lg1_probe.py"),
                                  "--output", str(output / "P1-LG1"), "--particle-count", "1024"]))
    if include_lg1 and args.stage in ("P2-reference", "all"):
        jobs.append(dict(name="P2-reference-job", device="cpu", timeout=min(300., args.probe_timeout),
                         command=[sys.executable, str(ROOT / "scripts/prepare_surrogate_hmc_lg1_reference.py"),
                                  "--output", str(output / "P2-reference")]))
    if include_lg1 and args.stage in ("P2-controls", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        if reference_root is None:
            parser.error("P2-controls requires --reference-root from a completed P2-reference stage")
        jobs.append(dict(name="P2-controls-job", device="gpu", timeout=min(900., args.probe_timeout),
                         command=[sys.executable, str(ROOT / "scripts/calibrate_surrogate_hmc_lg1_controls.py"),
                                  "--output", str(output / "P2-controls"), "--reference-root", str(reference_root.resolve())]))
    if include_lg1 and args.stage in ("P2-tune", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        if reference_root is None or selection is None:
            parser.error("P2-tune requires --reference-root and --selection")
        jobs.append(dict(name="P2-tune-job", device="gpu", timeout=args.tuning_timeout, cpu_allowance=args.tuning_cpu_seconds,
                         command=[sys.executable, str(ROOT / "scripts/tune_surrogate_hmc_lg1.py"),
                                  "--output", str(output / "P2-tune"), "--reference-root", str(reference_root.resolve()),
                                  "--selection", str(selection.resolve())]))
    if include_lg1 and args.stage in ("P2-sample", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        tuning_root = output / "P2-tune" if args.stage == "all" else args.tuning_root
        if reference_root is None or selection is None or tuning_root is None:
            parser.error("P2-sample requires --reference-root, --selection and --tuning-root")
        jobs.append(dict(name="P2-sample-job", device="gpu", timeout=args.sampling_timeout,
                         cpu_allowance=args.sampling_cpu_seconds,
                         command=[sys.executable, str(ROOT / "scripts/sample_surrogate_hmc_lg1.py"),
                                  "--output", str(output / "P2-sample"), "--reference-root", str(reference_root.resolve()),
                                  "--selection", str(selection.resolve()), "--tuning-root", str(tuning_root.resolve())]))
    if include_lg1 and args.stage in ("P2-endpoint", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        if reference_root is None or selection is None:
            parser.error("P2-endpoint requires --reference-root and --selection")
        jobs.append(dict(name="P2-endpoint-job", device="gpu", timeout=args.endpoint_timeout,
                         command=[sys.executable, str(ROOT / "scripts/qualify_surrogate_hmc_lg1_finite_endpoint.py"),
                                  "--output", str(output / "P2-endpoint"), "--reference-root", str(reference_root.resolve()),
                                  "--selection", str(selection.resolve())]))
    if include_lg1 and args.stage in ("P2-finite-reference", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        endpoint_root = output / "P2-endpoint" if args.stage == "all" else args.endpoint_root
        if reference_root is None or selection is None or endpoint_root is None:
            parser.error("P2-finite-reference requires --reference-root, --selection and --endpoint-root")
        jobs.append(dict(name="P2-finite-reference-job", device="gpu", timeout=900., cpu_allowance=1200.,
                         command=[sys.executable,str(ROOT / "scripts/prepare_surrogate_hmc_lg1_finite_reference.py"),
                                  "--output",str(output / "P2-finite-reference"),"--reference-root",str(reference_root.resolve()),
                                  "--selection",str(selection.resolve()),"--endpoint-root",str(endpoint_root.resolve())]))
    for c_stage in ("tune", "sample"):
        if not include_lg1 or args.stage not in ("P2-C-" + c_stage, "all"):
            continue
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        endpoint_root = output / "P2-endpoint" if args.stage == "all" else args.endpoint_root
        finite_root = output / "P2-finite-reference" if args.stage == "all" else args.finite_reference_root
        c_tuning_root = output / "P2-C-tune" if args.stage == "all" else args.tuning_root
        if any(value is None for value in (reference_root, selection, endpoint_root, finite_root)):
            parser.error("C requires --reference-root, --selection, --endpoint-root and --finite-reference-root")
        command = [sys.executable, str(ROOT / "scripts/run_surrogate_hmc_lg1_finite.py"),
                   "--stage", c_stage, "--output", str(output / ("P2-C-" + c_stage)),
                   "--reference-root", str(reference_root.resolve()), "--selection", str(selection.resolve()),
                   "--endpoint-root", str(endpoint_root.resolve()), "--finite-reference-root", str(finite_root.resolve())]
        if c_stage == "sample":
            if c_tuning_root is None:
                parser.error("C sampling requires its own --tuning-root")
            command += ["--tuning-root", str(c_tuning_root.resolve())]
        jobs.append(dict(name="P2-C-" + c_stage + "-job", device="gpu", command=command,
                         timeout=args.tuning_timeout if c_stage == "tune" else args.sampling_timeout,
                         cpu_allowance=args.tuning_cpu_seconds if c_stage == "tune" else args.sampling_cpu_seconds))
    if include_lg2 and args.stage in ("P3-LG2-reference", "all"):
        jobs.append(dict(name="P3-LG2-reference-job", device="cpu", timeout=120., cpu_allowance=180.,
                         command=[sys.executable, str(ROOT / "scripts/prepare_surrogate_hmc_linear_reference.py"),
                                  "--model", "LG2", "--output", str(output / "P3-LG2-reference")]))
    if include_lg2 and args.stage in ("P3-LG2-qualify", "all"):
        lg2_reference_root = output / "P3-LG2-reference" if args.stage == "all" else args.reference_root
        if lg2_reference_root is None:
            parser.error("P3-LG2-qualify requires --reference-root from a completed LG2 reference stage")
        jobs.append(dict(name="P3-LG2-qualify-job", device="gpu", timeout=600., cpu_allowance=450.,
                         command=[sys.executable, str(ROOT / "scripts/qualify_surrogate_hmc_linear_consumer.py"),
                                  "--model", "LG2", "--output", str(output / "P3-LG2-qualify"),
                                  "--reference-root", str(lg2_reference_root.resolve())]))
    if include_lg2 and args.stage in ("P3-LG2-controls", "all"):
        lg2_reference_root = output / "P3-LG2-reference" if args.stage == "all" else args.reference_root
        if lg2_reference_root is None:
            parser.error("P3-LG2-controls requires --reference-root from a completed LG2 reference stage")
        jobs.append(dict(name="P3-LG2-controls-job", device="gpu", timeout=900., cpu_allowance=1000.,
                         command=[sys.executable, str(ROOT / "scripts/calibrate_surrogate_hmc_linear_controls.py"),
                                  "--model", "LG2", "--output", str(output / "P3-LG2-controls"),
                                  "--reference-root", str(lg2_reference_root.resolve())]))
    if include_sv20 and args.stage in ("P3-SV20-reference", "all"):
        timeout, cpu_allowance = sv20_reference_remaining_allocation(ROOT)
        jobs.append(dict(name="P3-SV20-reference-job", device="gpu", timeout=timeout, cpu_allowance=cpu_allowance,
                         command=[sys.executable, str(ROOT / "scripts/prepare_surrogate_hmc_sv_reference.py"),
                                  "--output", str(output / "P3-SV20-reference")]))
    if include_sv20 and args.stage in ("P3-SV20-posterior", "all"):
        sv20_reference_root = output/"P3-SV20-reference" if args.stage == "all" else args.reference_root
        if sv20_reference_root is None:
            parser.error("P3-SV20-posterior requires --reference-root from the qualified SV20 likelihood stage")
        timeout, cpu_allowance = sv20_reference_remaining_allocation(ROOT, "posterior")
        jobs.append(dict(name="P3-SV20-posterior-job", device="gpu", timeout=timeout, cpu_allowance=cpu_allowance,
                         command=[sys.executable, str(ROOT / "scripts/prepare_surrogate_hmc_sv_posterior.py"),
                                  "--output", str(output/"P3-SV20-posterior"),
                                  "--reference-root", str(sv20_reference_root.resolve()),
                                  "--wall-budget", str(timeout), "--cpu-budget", str(cpu_allowance)]))
    if include_sv20 and args.stage in ("P3-SV20-qualify", "all"):
        sv20_reference_root = output/"P3-SV20-reference" if args.stage == "all" else args.reference_root
        sv20_posterior_root = output/"P3-SV20-posterior" if args.stage == "all" else args.posterior_root
        if sv20_reference_root is None or sv20_posterior_root is None:
            parser.error("P3-SV20-qualify requires --reference-root and --posterior-root")
        timeout, cpu_allowance = sv20_reference_remaining_allocation(ROOT, "qualify")
        jobs.append(dict(name="P3-SV20-qualify-job", device="gpu", timeout=timeout, cpu_allowance=cpu_allowance,
                         command=[sys.executable, str(ROOT/"scripts/qualify_surrogate_hmc_sv_consumer.py"),
                                  "--output", str(output/"P3-SV20-qualify"),
                                  "--reference-root", str(sv20_reference_root.resolve()),
                                  "--posterior-root", str(sv20_posterior_root.resolve()),
                                  "--wall-budget", str(timeout), "--cpu-budget", str(cpu_allowance)]))
    if include_sv20 and args.stage in ("P3-SV20-controls", "all"):
        sv20_reference_root = output/"P3-SV20-reference" if args.stage == "all" else args.reference_root
        sv20_qualification_root = output/"P3-SV20-qualify" if args.stage == "all" else args.qualification_root
        if sv20_reference_root is None or sv20_qualification_root is None:
            parser.error("P3-SV20-controls requires --reference-root and --qualification-root")
        timeout, cpu_allowance = sv20_reference_remaining_allocation(ROOT, "controls")
        jobs.append(dict(name="P3-SV20-controls-job", device="gpu", timeout=timeout, cpu_allowance=cpu_allowance,
                         command=[sys.executable, str(ROOT/"scripts/calibrate_surrogate_hmc_sv_controls.py"),
                                  "--output", str(output/"P3-SV20-controls"),
                                  "--reference-root", str(sv20_reference_root.resolve()),
                                  "--qualification-root", str(sv20_qualification_root.resolve()),
                                  "--wall-budget", str(timeout), "--cpu-budget", str(cpu_allowance)]))
    m10_state = m10_reference_state(ROOT) if include_m10 else {}
    m10_mechanics_state = m10_consumer_state(ROOT) if include_m10 else {}
    m10_posterior = m10_posterior_state(ROOT) if include_m10 else {}
    m10_controls = m10_controls_state(ROOT) if include_m10 and args.stage in ("P0", "P3-M10-controls", "P3-M10-endpoint", "P3-M10-inference", "all") else {}
    m10_endpoint = m10_endpoint_state(ROOT) if include_m10 else {}
    m10_history = m10_inference_history(ROOT) if include_m10 else {}
    if include_m10 and args.stage in ("P3-M10-reference", "all"):
        jobs.append(dict(name="P3-M10-reference-job", device="cpu", timeout=m10_state["wall_seconds"],
                         cpu_allowance=m10_state["cpu_worker_seconds"],
                         prior_preparation_result=m10_state["qualified"],
                         command=[sys.executable, str(ROOT/"scripts/prepare_surrogate_hmc_nonlinear_reference.py"),
                                  "--output", str(output/"P3-M10-reference")]))
    if include_m10 and args.stage in ("P3-M10-qualify", "all"):
        if args.stage != "all" and not m10_state["qualified"]:
            parser.error("M10 mechanics requires a completed local reference")
        m10_reference_root = (Path(m10_state["qualified"]["report"]).parent if m10_state["qualified"]
                              else output/"P3-M10-reference")
        jobs.append(dict(name="P3-M10-qualify-job", device="gpu", timeout=m10_mechanics_state["wall_seconds"],
                         cpu_allowance=m10_mechanics_state["cpu_worker_seconds"],
                         prior_preparation_result=m10_mechanics_state["qualified"],
                         command=[sys.executable, str(ROOT/"scripts/qualify_surrogate_hmc_nonlinear_consumer.py"),
                                  "--output", str(output/"P3-M10-qualify"),
                                  "--reference-root", str(m10_reference_root)]))
    if include_m10 and args.stage in ("P3-M10-posterior", "all"):
        if args.stage != "all" and not m10_state["qualified"]:
            parser.error("M10 posterior preparation requires the frozen local-reference data")
        m10_reference_root = (Path(m10_state["qualified"]["report"]).parent if m10_state["qualified"]
                              else output/"P3-M10-reference")
        jobs.append(dict(name="P3-M10-posterior-job", device="gpu", timeout=m10_posterior["wall_seconds"],
                         cpu_allowance=m10_posterior["cpu_worker_seconds"],
                         prior_preparation_result=m10_posterior["qualified"],
                         command=[sys.executable, str(ROOT/"scripts/prepare_surrogate_hmc_nonlinear_posterior.py"),
                                  "--gpu", "--output", str(output/"P3-M10-posterior"),
                                  "--reference-root", str(m10_reference_root),
                                  "--wall-budget", str(m10_posterior["wall_seconds"]),
                                  "--cpu-budget", str(m10_posterior["cpu_worker_seconds"])]))
    if include_m10 and args.stage in ("P3-M10-controls", "all"):
        if args.stage != "all" and not m10_mechanics_state["qualified"]:
            parser.error("M10 control calibration requires qualified current-source mechanics")
        control_qualification = (Path(m10_mechanics_state["qualified"]["report"]).parent
                                 if m10_mechanics_state["qualified"] else output/"P3-M10-qualify")
        jobs.append(dict(name="P3-M10-controls-job", device="gpu", timeout=m10_controls["wall_seconds"],
            cpu_allowance=m10_controls["cpu_worker_seconds"], prior_preparation_result=m10_controls["qualified"],
            command=[sys.executable, str(ROOT/"scripts/calibrate_surrogate_hmc_nonlinear_controls.py"),
                     "--output", str(output/"P3-M10-controls"),
                     "--qualification-root", str(control_qualification),
                     "--wall-budget", str(m10_controls["wall_seconds"]),
                     "--cpu-budget", str(m10_controls["cpu_worker_seconds"]),
                     "--family-cpu-budget", str(m10_mechanics_state["family_remaining"]["cpu_worker_seconds"]),
                     "--family-gpu-budget", str(m10_mechanics_state["family_remaining"]["gpu_seconds"])]))
    if include_m10 and args.stage in ("P3-M10-endpoint", "all"):
        if args.stage != "all" and not all(s.get("qualified") for s in (m10_state, m10_posterior, m10_controls)):
            parser.error("M10 endpoint requires qualified data, posterior reference and scoped controls")
        jobs.append(dict(name="P3-M10-endpoint-job", device="gpu", timeout=m10_endpoint["wall_seconds"],
            cpu_allowance=m10_endpoint["cpu_worker_seconds"], prior_preparation_result=m10_endpoint["qualified"],
            command=[sys.executable, str(ROOT/"scripts/qualify_surrogate_hmc_nonlinear_endpoint.py"),
                     "--output", str(output/"P3-M10-endpoint"),
                     "--wall-budget", str(m10_endpoint["wall_seconds"]),
                     "--cpu-budget", str(m10_endpoint["cpu_worker_seconds"])]))
    if include_m10 and args.stage in ("P3-M10-inference", "all"):
        if m10_history["status"] != "NOT_RUN" and not m10_history.get("artifact_valid"):
            parser.error("existing M10 inference attempt requires result-aware repair before another launch")
        if args.stage != "all" and not m10_history.get("artifact_valid") and not m10_endpoint.get("qualified"):
            parser.error("M10 inference requires qualified complete endpoint and forecast")
        family = m10_mechanics_state["family_remaining"]
        jobs.append(dict(name="P3-M10-inference-job", device="gpu",
            timeout=min(3600., family["gpu_seconds"]), cpu_allowance=min(3200., family["cpu_worker_seconds"]),
            prior_retained_result=m10_history if m10_history.get("artifact_valid") else None,
            command=[sys.executable, str(ROOT/"scripts/run_surrogate_hmc_nonlinear.py"),
                     "--output", str(output/"P3-M10-inference"), "--cpu-budget", str(min(3200., family["cpu_worker_seconds"])),
                     "--wall-budget", str(min(3600., family["gpu_seconds"]))]))
    for control_stage in ("qualify", "inference"):
        if include_c01 and args.stage in (f"P3-C01-{control_stage}", "all"):
            timeout, cpu_allowance = gaussian_remaining_allocation(ROOT, control_stage)
            script = ("qualify_surrogate_hmc_gaussian_controls.py" if control_stage == "qualify"
                      else "run_surrogate_hmc_gaussian_controls.py")
            command = [sys.executable, str(ROOT/"scripts"/script), "--output", str(output/f"P3-C01-{control_stage}")]
            if control_stage == "inference":
                c01_qualification = output/"P3-C01-qualify" if args.stage == "all" else args.qualification_root
                if c01_qualification is None:
                    parser.error("C01 inference requires --qualification-root")
                command += ["--qualification-root", str(c01_qualification.resolve())]
            jobs.append(dict(name=f"P3-C01-{control_stage}-job", device="gpu", timeout=timeout,
                             cpu_allowance=cpu_allowance, command=command))
    if args.stage == "all":
        prerequisites = {
            "P1-LG1": [], "P2-reference-job": [],
            "P2-controls-job": ["P1-LG1", "P2-reference-job"],
            "P2-tune-job": ["P2-controls-job"], "P2-sample-job": ["P2-tune-job"],
            "P2-endpoint-job": ["P2-controls-job"],
            "P2-finite-reference-job": ["P2-endpoint-job"],
            "P2-C-tune-job": ["P2-finite-reference-job"],
            "P2-C-sample-job": ["P2-C-tune-job"],
            "P3-LG2-reference-job": [],
            "P3-LG2-qualify-job": ["P3-LG2-reference-job"],
            "P3-LG2-controls-job": ["P3-LG2-qualify-job"],
            "P3-C01-qualify-job": [],
            "P3-C01-inference-job": ["P3-C01-qualify-job"],
            "P3-M10-reference-job": [],
            "P3-M10-qualify-job": ["P3-M10-reference-job"],
            "P3-M10-posterior-job": ["P3-M10-reference-job"],
            "P3-M10-controls-job": ["P3-M10-qualify-job"],
            "P3-M10-endpoint-job": ["P3-M10-controls-job", "P3-M10-posterior-job"],
            "P3-M10-inference-job": ["P3-M10-endpoint-job"],
            "P3-SV20-reference-job": [],
            "P3-SV20-posterior-job": ["P3-SV20-reference-job"],
            "P3-SV20-qualify-job": ["P3-SV20-posterior-job"],
            "P3-SV20-controls-job": ["P3-SV20-qualify-job"],
        }
        for job in jobs:
            job["depends_on"] = prerequisites[job["name"]]
    history = retained_attempt_history(ROOT, lg1_scope()["inference_attempt"]) if include_lg1 else {}
    if include_c01:
        history.update(gaussian_attempt_history(ROOT))
    if include_ksc and args.stage == "all" and ksc_history.get("artifact_valid"):
        history["P3-KSC-inference-job"] = dict(report=ksc_history["report"],
            report_sha256=ksc_history["report_sha256"], closed=True, authorizes_new_inference=False)
        jobs.append(dict(name="P3-KSC-inference-job", depends_on=[]))
    jobs = preserve_retained_attempts(jobs, history, all_stages=args.stage == "all")
    if args.dry_run:
        for job in jobs:
            if not job.get("prior_retained_result") and not job.get("prior_preparation_result"):
                reserve_job(ledger, job["device"], job["timeout"], job.get("cpu_allowance"))
    scope_cards = {}
    if include_ksc:
        scope_cards["KSC"] = dict(id="M03-KSC-T2", family="M03", stage=args.stage,
            state_dimension=1, parameter_dimension=2, horizon=2,
            law="existing KSC mixture observation law; exact49-component model endpoint",
            prior="independent Uniform(.1,.9)", chart="gamma,beta=.1+.8*Phi(z)",
            role="surrogate_HMC_model_posterior_test", evidence=ksc_history,
            plan="docs/plans/surrogate-hmc-ksc-t2-2026-09-21.md")
    if include_lg1:
        scope_cards["LG1"] = lg1_scope(args.stage)
    if include_lg2:
        scope_cards["LG2"] = lg2_scope(args.stage)
    if include_sv20:
        scope_cards["SV20"] = dict(id="M02-SV20-attempt1", family="M02", stage=args.stage,
            state_dimension=1, parameter_dimension=2, parameters=["gamma", "beta"], horizon=20,
            law="stationary actual SV; transition then Normal(0,beta^2*exp(x)) observation; sigma=1",
            prior="independent Uniform(.1,.9)", chart="gamma,beta=.1+.8*Phi(z)",
            role="reference_and_full_DCTR_consumer_preparation", posterior_reference_qualified=False,
            surrogate_hmc_tested=False, plan="docs/plans/surrogate-hmc-sv20-reference-2026-09-20.md")
    if include_m10:
        scope_cards["M10"] = dict(id="M10-T2", family="M10", stage=args.stage,
            state_dimension=1, parameter_dimension=2, parameters=["lambda_Q", "lambda_R"], horizon=2,
            law="x=a*xprev+.2*sin(xprev)+sigma_Q*epsilon; y=H*x+.12*x^2+sigma_R*eta",
            prior="independent Normal(0,log(2)^2)", chart="identity log-noise scales",
            role="surrogate_HMC_model_posterior_test", posterior_reference_qualified=bool(m10_posterior.get("qualified")),
            surrogate_hmc_tested=m10_history.get("surrogate_hmc_tested", False),
            plan="docs/plans/surrogate-hmc-m10-inference-2026-09-22.md")
    if include_c01:
        scope_cards["C01"] = dict(id="C01-anisotropic-gaussian-d3-attempt1", family="C01", stage=args.stage,
            parameter_dimension=3, precision=[1., 4., 9.], fields=["exact", "half", "rotational", "zero"],
            role="mechanism_control", dctr_inference_evidence=False,
            plan="docs/plans/surrogate-hmc-gaussian-controls-2026-09-20.md")
    launch = {"schema": SCHEMA, "stage": args.stage, "scope": args.scope,
              "jobs": jobs,
              "prior_retained_results": history,
              "required_inputs": [PLAN, EXECUTION_PLAN] + ([
                  "docs/plans/surrogate-hmc-lg2-inference-preparation-2026-09-20.md",
                  "docs/plans/surrogate-hmc-lg2-consumer-qualification-2026-09-20.md",
                  "docs/plans/surrogate-hmc-lg2-controls-2026-09-20.md"] if include_lg2 else [])
                  + (["docs/plans/surrogate-hmc-gaussian-controls-2026-09-20.md"] if include_c01 else [])
                  + (["docs/plans/surrogate-hmc-sv20-reference-2026-09-20.md",
                      "docs/plans/surrogate-hmc-sv20-posterior-2026-09-20.md",
                      "docs/plans/surrogate-hmc-sv20-consumer-2026-09-20.md",
                      "docs/plans/surrogate-hmc-sv20-controls-2026-09-20.md"] if include_sv20 else [])
                  + (["docs/plans/surrogate-hmc-nonlinear-reference-2026-09-20.md",
                      "docs/plans/surrogate-hmc-m10-inference-2026-09-22.md"] if include_m10 else []),
              "scope_cards": scope_cards,
              "scope_card": next(iter(scope_cards.values())) if len(scope_cards) == 1 else None,
              "endpoint_force_identities": "issued from actual frozen inputs by worker before evaluation",
              "inference_forecast": "NOT_MEASURED; inference launch prohibited",
              "remaining_budget": {key: ledger["limits"][key]-ledger["charged"][key] for key in ledger["limits"]},
              "prior_runs": ledger["prior_runs"], "inference_ready": False}
    if args.dry_run and args.stage == "P2-sample" and not history.get("P2-sample-job"):
        launch["inference_forecast"] = retained_forecast(tuning_root)
        validate_retained_forecast(launch["inference_forecast"], jobs[0])
    elif args.dry_run and args.stage == "P2-C-sample" and not history.get("P2-C-sample-job"):
        launch["inference_forecast"] = retained_forecast(c_tuning_root, "C")
        validate_retained_forecast(launch["inference_forecast"], jobs[0])
    elif args.dry_run and args.stage == "P3-C01-inference" and not history.get("P3-C01-inference-job"):
        launch["inference_forecast"] = gaussian_qualification_forecast(c01_qualification, jobs[0])
    elif args.dry_run and args.stage == "P3-M10-inference" and not m10_history.get("artifact_valid"):
        forecast = m10_endpoint["qualified"]["complete_forecast"]
        if (forecast["required"]["cpu_worker_seconds"] > jobs[0]["cpu_allowance"] or
                forecast["required"]["wall_seconds"] > jobs[0]["timeout"]):
            parser.error("complete M10 inference forecast exceeds allocation")
        launch["inference_forecast"] = forecast
    if args.dry_run:
        print(json.dumps(launch, indent=2, allow_nan=False))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    parent_cpu = resource.getrusage(resource.RUSAGE_SELF)
    manifest = dict(launch, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                    python=sys.executable, command_line=[sys.executable, *sys.argv], cwd=str(ROOT),
                    plan=PLAN, execution_plan=EXECUTION_PLAN, result=str(output / "result.md"),
                    environment={key: os.environ.get(key) for key in ("CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH")})
    write_json(output / "run-manifest.json", manifest)
    probe = None
    error = None
    outcomes = {}
    try:
        write_json(output / "source-inventory.json", source_inventory())
        write_json(output / "model-register.json", model_register())
        for name, card in scope_cards.items():
            write_json(output / f"scopes/{name}.json", card)
        def preflight(job):
            if job["name"].startswith("P3-M10-"):
                used = resource.getrusage(resource.RUSAGE_SELF)
                parent_seconds = used.ru_utime + used.ru_stime - parent_cpu.ru_utime - parent_cpu.ru_stime
                remaining = dict(m10_mechanics_state["family_remaining"])
                remaining["cpu_worker_seconds"] -= parent_seconds
                for charged_job in ledger.get("jobs", []):
                    if any("surrogate_hmc_nonlinear" in str(arg) for arg in charged_job.get("command", [])):
                        for key in remaining:
                            remaining[key] -= float(charged_job.get(key, 0.))
                if job["device"] == "gpu":
                    job["timeout"] = min(job["timeout"], remaining["gpu_seconds"])
                job["cpu_allowance"] = min(job["cpu_allowance"], remaining["cpu_worker_seconds"])
                # Nested reference workers must inherit the reduced allowance
                # before their own watchdog/RLIMIT is configured.
                for option, value in (("--wall-budget", job["timeout"]),
                                      ("--cpu-budget", job["cpu_allowance"]),
                                      ("--family-cpu-budget", remaining["cpu_worker_seconds"]),
                                      ("--family-gpu-budget", remaining["gpu_seconds"])):
                    if option in job["command"]:
                        job["command"][job["command"].index(option)+1] = str(value)
            reserve_job(ledger, job["device"], job["timeout"], job.get("cpu_allowance"))
            if job["name"] == "P3-M10-inference-job":
                qualified = m10_endpoint_state(ROOT)["qualified"]
                if not qualified:
                    raise ValueError("qualified M10 inference endpoint required")
                forecast = qualified["complete_forecast"]
                if (forecast["required"]["cpu_worker_seconds"] > job["cpu_allowance"] or
                        forecast["required"]["wall_seconds"] > job["timeout"]):
                    raise ValueError("complete M10 inference forecast exceeds remaining reservation")
                manifest.update(inference_forecast=forecast, inference_ready=True)
                return dict(family_remaining_before_job=remaining, inference_forecast=forecast)
            if job["name"].startswith("P3-SV20-"):
                used = resource.getrusage(resource.RUSAGE_SELF)
                parent_seconds = used.ru_utime + used.ru_stime - parent_cpu.ru_utime - parent_cpu.ru_stime
                remaining = sv20_family_remaining_allocation(ledger, parent_seconds)
                for key, need in (("gpu_seconds", job["timeout"]),
                                  ("cpu_worker_seconds", job["cpu_allowance"])):
                    if need > remaining[key]:
                        raise ValueError(f"insufficient SV20 family reservation: {key}; remaining={remaining[key]}")
                return {"family_remaining_before_job": remaining}
            if job["name"] in ("P2-sample-job", "P2-C-sample-job"):
                forecast = (retained_forecast(tuning_root) if job["name"] == "P2-sample-job"
                            else retained_forecast(c_tuning_root, "C"))
                validate_retained_forecast(forecast, job)
                manifest.update(inference_forecast=forecast, inference_ready=True)
                manifest.setdefault("inference_forecasts", {})[job["name"]] = forecast
                write_json(output / "run-manifest.json", manifest)
                return forecast
            if job["name"] == "P3-C01-inference-job":
                forecast = gaussian_qualification_forecast(c01_qualification, job)
                manifest.update(inference_forecast=forecast, inference_ready=True)
                write_json(output/"run-manifest.json", manifest)
                return forecast
            return None
        outcomes = execute_job_graph(jobs, output, ledger, before_job=preflight)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # An interruption raises before execute_job_graph returns its mapping.
        if (output / "execution-results.json").exists():
            outcomes = json.loads((output / "execution-results.json").read_text())
        probe = next((outcomes[job["name"]] for job in reversed(jobs)
                      if "exit_code" in outcomes.get(job["name"], {})), None)
        used = resource.getrusage(resource.RUSAGE_SELF)
        ledger["charged"]["cpu_worker_seconds"] += used.ru_utime + used.ru_stime - parent_cpu.ru_utime - parent_cpu.ru_stime
        ledger["remaining"] = {k: ledger["limits"][k] - ledger["charged"][k] for k in ledger["limits"]}
        ledger["charged_this_run"] = {key: ledger["charged"][key]-prior_charged[key] for key in ledger["limits"]}
        write_json(output / "accounting.json", ledger)
    diagnostic_path = output / "P1-LG1/diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    reference_path = (args.reference_root or output / "P2-reference") / "report.json"
    reference = json.loads(reference_path.read_text()) if include_lg1 and reference_path.exists() else {}
    selection_path = (args.selection.parent if args.selection else output / "P2-controls") / "report.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    tuning_path = (args.tuning_root or output / ("P2-C-tune" if args.stage.startswith("P2-C") else "P2-tune")) / "report.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    sampling_path = output / ("P2-C-sample/report.json" if args.stage.startswith("P2-C") else "P2-sample/report.json")
    sampling_job = "P2-C-sample-job" if args.stage.startswith("P2-C") else "P2-sample-job"
    reporting_errors = []

    def retained_report(path, job_name):
        try:
            return read_retained_report(path, job_name, history, outcomes,
                                        report_only=args.stage == "P0")
        except (ValueError, KeyError, OSError) as exc:
            reporting_errors.append(str(exc))
            return {}

    sampling = retained_report(sampling_path, sampling_job) if include_lg1 else {}
    c_sampling_path = output / "P2-C-sample/report.json"
    c_sampling = retained_report(c_sampling_path, "P2-C-sample-job") if include_lg1 else {}
    c_tuning_path = ((args.tuning_root if args.stage == "P2-C-sample" else None)
                     or output / "P2-C-tune") / "report.json"
    c_tuning = json.loads(c_tuning_path.read_text()) if c_tuning_path.exists() else {}
    endpoint_path = (args.endpoint_root or output / "P2-endpoint") / "report.json"
    endpoint = json.loads(endpoint_path.read_text()) if endpoint_path.exists() else {}
    finite_reference_path = (args.finite_reference_root or output / "P2-finite-reference") / "report.json"
    if not args.finite_reference_root and not finite_reference_path.exists() and c_sampling:
        recorded = c_sampling.get("provenance", {})
        if recorded.get("finite_reference_path"):
            recorded_path = Path(recorded["finite_reference_path"])
            if (recorded_path.is_file() and
                    file_hash(recorded_path) == recorded.get("finite_reference_sha256")):
                finite_reference_path = recorded_path
            else:
                reporting_errors.append("recorded finite posterior reference changed")
                c_sampling = {}
                if args.stage.startswith("P2-C"):
                    sampling = {}
    finite_reference = json.loads(finite_reference_path.read_text()) if finite_reference_path.exists() else {}
    lg2_reference_path = (args.reference_root or output / "P3-LG2-reference") / "report.json"
    lg2_reference = json.loads(lg2_reference_path.read_text()) if include_lg2 and lg2_reference_path.exists() else {}
    lg2_qualification_path = output / "P3-LG2-qualify/report.json"
    lg2_qualification = json.loads(lg2_qualification_path.read_text()) if lg2_qualification_path.exists() else {}
    lg2_controls_path = output / "P3-LG2-controls/report.json"
    lg2_controls = json.loads(lg2_controls_path.read_text()) if lg2_controls_path.exists() else {}
    from bayesfilter.inference.surrogate_hmc_linear_program import lg2_endpoint_history
    lg2_endpoint = lg2_endpoint_history(ROOT) if include_lg2 else {}
    c01_qualification_path = (args.qualification_root or output/"P3-C01-qualify")/"report.json"
    c01_qualification_report = (json.loads(c01_qualification_path.read_text())
                                if include_c01 and c01_qualification_path.exists() else {})
    c01_sampling_path = output/"P3-C01-inference/report.json"
    c01_sampling = retained_report(c01_sampling_path, "P3-C01-inference-job") if include_c01 else {}
    sv20_reference_path = (args.reference_root or output/"P3-SV20-reference")/"report.json"
    sv20_reference = json.loads(sv20_reference_path.read_text()) if include_sv20 and sv20_reference_path.exists() else {}
    prior_sv20_posterior = args.posterior_root
    if include_sv20 and args.stage == "P3-SV20-controls" and not prior_sv20_posterior:
        qualification_manifest_path = args.qualification_root/"run-manifest.json"
        if qualification_manifest_path.exists():
            prior_path = json.loads(qualification_manifest_path.read_text()).get("posterior_reference")
            prior_sv20_posterior = Path(prior_path) if prior_path else None
    sv20_posterior_path = (prior_sv20_posterior or output/"P3-SV20-posterior")/"report.json"
    sv20_posterior = json.loads(sv20_posterior_path.read_text()) if include_sv20 and sv20_posterior_path.exists() else {}
    sv20_consumer_path = ((args.qualification_root if args.stage == "P3-SV20-controls" else None)
                          or output/"P3-SV20-qualify")/"report.json"
    sv20_consumer = json.loads(sv20_consumer_path.read_text()) if include_sv20 and sv20_consumer_path.exists() else {}
    sv20_controls_path = output/"P3-SV20-controls/report.json"
    sv20_controls = json.loads(sv20_controls_path.read_text()) if include_sv20 and sv20_controls_path.exists() else {}
    m10_reference_path = (Path(m10_state["qualified"]["report"]) if m10_state.get("qualified")
                          else output/"P3-M10-reference/report.json")
    m10_reference = json.loads(m10_reference_path.read_text()) if include_m10 and m10_reference_path.exists() else {}
    m10_consumer_path = (Path(m10_mechanics_state["qualified"]["report"]) if m10_mechanics_state.get("qualified")
                         else output/"P3-M10-qualify/report.json")
    m10_consumer = json.loads(m10_consumer_path.read_text()) if include_m10 and m10_consumer_path.exists() else {}
    m10_posterior_path = (Path(m10_posterior["qualified"]["report"]) if m10_posterior.get("qualified")
                          else output/"P3-M10-posterior/report.json")
    m10_posterior_report = json.loads(m10_posterior_path.read_text()) if include_m10 and m10_posterior_path.exists() else {}
    m10_controls_path = (Path(m10_controls["qualified"]["report"]) if m10_controls.get("qualified")
                         else output/"P3-M10-controls/report.json")
    m10_control_report = json.loads(m10_controls_path.read_text()) if include_m10 and m10_controls_path.exists() else {}
    m10_endpoint_record = m10_endpoint.get("qualified") or m10_endpoint.get("latest")
    m10_endpoint_path = (Path(m10_endpoint_record["report"]) if m10_endpoint_record
                         else output/"P3-M10-endpoint/report.json")
    m10_endpoint_report = json.loads(m10_endpoint_path.read_text()) if include_m10 and m10_endpoint_path.exists() else {}
    if include_m10:
        m10_history = m10_inference_history(ROOT)
    if reporting_errors:
        error = "; ".join(filter(None, (error, *reporting_errors)))
    failed = error is not None or any(row["status"] not in ("COMPLETED", "HISTORICAL_RESULT_REPORTED")
                                      for row in outcomes.values())
    completed_any = any(row["status"] == "COMPLETED" for row in outcomes.values())
    decision = {"schema": SCHEMA, "status": "EXECUTION_FAILED" if failed else "INCOMPLETE_IMPLEMENTATION",
                "error": error, "all_models_validated": False,
                "engineering_validity": diagnostic.get("status", "NOT_CHECKED"),
                "comparisons": {"B/R":"NOT_RUN", "C/D":"NOT_RUN", "D/R":"NOT_RUN", "C/R":"NOT_RUN"} | sampling.get("comparisons", {}),
                "comparison_scope": "LG1" if include_lg1 else None,
                "scope_results": {"LG2": {"reference": lg2_reference, "consumer_qualification": lg2_qualification,
                                          "control_calibration": lg2_controls,
                                          "inference_endpoint": lg2_endpoint,
                                          "comparisons": {"B/R":"NOT_RUN", "C/D":"NOT_RUN", "D/R":"NOT_RUN", "C/R":"NOT_RUN"}}} if include_lg2 else {},
                "reference_quality": "QUALIFIED_LG1_MODEL_QUADRATURE" if reference.get("qualified") else "NOT_QUALIFIED",
                "reference": reference,
                "dctr_control_selection": selection.get("verdict", "NOT_RUN"),
                "scoped_hmc_tuning": tuning,
                "finite_endpoint_hmc_tuning": c_tuning,
                "retained_sampling": sampling,
                "finite_endpoint_sampling": c_sampling,
                "prior_retained_results": history,
                "finite_endpoint_qualification": endpoint,
                "finite_posterior_reference": finite_reference,
                "execution_status": ("PARTIAL" if completed_any else "FAILED") if failed else "COMPLETED",
                "job_outcomes": outcomes,
                "mixing_mcse": "SEE_RETAINED_LOOKS" if sampling or c_sampling else "NOT_RUN", "heuristic_dominance": "NOT_RUN",
                "efficiency_inference": "NOT_ESTABLISHED", "default_readiness": False,
                "probe": probe, "remaining_budget": ledger["remaining"],
                "inference_blocker": None if sampling or tuning.get("arms", {}).get("C" if args.stage.startswith("P2-C") else "B", {}).get("passed") else "Scoped tuning prerequisites must be loaded and verified",
                "remaining_requirements": ["resolved posterior comparisons", "heuristic comparisons", "remaining model consumers and references"],
                "next_action": "Assess LG1 posterior comparisons and continue the remaining model queue" if sampling else "Resolve measured consumer/tuning prerequisites before posterior comparisons"}
    if finite_reference:
        decision["comparisons"]["D/R"] = finite_reference.get("D_vs_R", {}).get("status", "INCONCLUSIVE")
    for name, status in c_sampling.get("comparisons", {}).items():
        if status != "NOT_RUN":
            decision["comparisons"][name] = status
    if include_lg1:
        decision["scope_results"]["LG1"] = dict(
            role="surrogate_HMC_model_posterior_test",
            comparisons=dict(decision["comparisons"]),
            retained_sampling=sampling, finite_endpoint_sampling=c_sampling,
            finite_posterior_reference=finite_reference)
    if include_lg2 and not include_lg1:
        decision["reference_quality"] = "QUALIFIED_LG2_MODEL_QUADRATURE" if lg2_reference.get("qualified") else "NOT_QUALIFIED"
        decision["engineering_validity"] = "QUALIFIED_LG2_CONSUMER" if lg2_qualification.get("qualified") else "NOT_CHECKED_IN_THIS_RUN"
        decision["next_action"] = "Assess LG2 control selection and complete inference forecast; continue the ready model queue"
        if lg2_controls.get("forecast", {}).get("status") == "UNDER_BUDGETED_INITIAL_ALLOCATION":
            decision["inference_blocker"] = "LG2 measured cost exceeds initial allocation; retained inference not launched"
        if lg2_endpoint.get("qualified"):
            decision.update(reference_quality="QUALIFIED_LG2_MODEL_QUADRATURE",
                engineering_validity="QUALIFIED_LG2_CLAIM_ENDPOINT",
                inference_blocker=lg2_endpoint["inference_readiness"],
                next_action=lg2_endpoint["next_action"])
    if include_c01:
        decision["scope_results"]["C01"] = dict(role="mechanism_control", dctr_inference_evidence=False,
            qualification=c01_qualification_report, retained_sampling=c01_sampling,
            comparisons=c01_sampling.get("comparisons", {arm+"/analytic": c01_sampling.get("arms", {}).get(arm, {}).get("status", "NOT_RUN")
                                                        for arm in ("exact", "half", "rotational", "zero")}))
        if not include_lg1 and not include_lg2 and not include_sv20:
            decision["reference_quality"] = "ANALYTIC_GAUSSIAN"
            decision["engineering_validity"] = ("QUALIFIED_C01_CONSUMERS" if c01_qualification_report.get("qualified")
                                                 else "SEE_PRIOR_C01_QUALIFICATION" if c01_sampling else "NOT_CHECKED")
            decision["mixing_mcse"] = "SEE_C01_RETAINED_LOOKS" if c01_sampling else "NOT_RUN"
            decision["inference_blocker"] = None if c01_sampling else "C01 qualified mechanics and affordable cost required"
            decision["next_action"] = ("Preserve the completed C01 attempt and continue remaining model consumers"
                                        if c01_sampling.get("closed") else "Assess C01 qualification and retained diagnostics")
    if include_sv20:
        from bayesfilter.inference.surrogate_hmc_sv_program import sv20_inference_history, sv20_endpoint_state
        sv_history = sv20_inference_history(ROOT)
        sv_endpoint = sv20_endpoint_state(ROOT)
        decision["scope_results"]["SV20"] = dict(role="surrogate_HMC_model_posterior_test", reference=sv20_reference,
            posterior_reference=sv20_posterior, posterior_reference_qualified=bool(sv20_posterior.get("qualified")),
            consumer=sv20_consumer, consumer_mechanics_qualified=bool(sv20_consumer.get("qualified")),
            control_selection=sv20_controls, inference_endpoint=sv_endpoint, **sv_history)
        if not include_lg1 and not include_lg2 and not include_c01:
            decision.update(reference_quality=sv20_posterior.get("status", sv20_reference.get("status", "NOT_RUN")),
                reference=sv20_reference, engineering_validity=sv20_consumer.get("status", "SEE_REFERENCE_CHECKS"),
                inference_blocker="SV-specific DCTR calibration and public HMC tuning remain incomplete" if sv20_consumer.get("qualified") else
                    "Actual-SV DCTR consumer remains unqualified" if sv20_posterior.get("qualified") else
                    "Full-prior posterior reference and actual-SV DCTR consumer remain unqualified",
                next_action="Assess original-force health, calibrate SV controls, and forecast inference cost" if sv20_consumer.get("qualified") else
                    "Assess reference refinement, then qualify the actual-SV DCTR consumer")
            if sv20_controls:
                decision.update(inference_blocker=sv20_controls.get("forecast", {}).get("status", sv20_controls.get("verdict")),
                    next_action="Preserve control and cost evidence; continue other ready scopes if inference is under-budgeted")
            if sv_history["status"] != "NOT_RUN":
                decision.update(inference_blocker=sv_history["status"],
                    next_action=sv_history.get("next_action", "Preserve the active SV20 inference attempt"))
            elif sv_endpoint["qualified"]:
                decision.update(inference_blocker="public tuning and retained B/R inference remain",
                    next_action="Execute the frozen SV20 B/R inference plan within its complete reservation")
    if include_m10:
        decision["scope_results"]["M10"] = dict(role="surrogate_HMC_model_posterior_test", reference=m10_reference,
            consumer_mechanics=m10_consumer, posterior_reference=m10_posterior_report,
            control_selection=m10_control_report, inference_endpoint=m10_endpoint_report,
            posterior_reference_qualified=bool(m10_posterior_report.get("qualified")), **m10_history)
        if not any((include_lg1, include_lg2, include_c01, include_sv20)):
            decision.update(reference_quality=m10_posterior_report.get("status", m10_reference.get("status", "NOT_RUN")),
                reference=m10_reference, engineering_validity=m10_consumer.get("status", "SEE_LOCAL_REFERENCE_CHECKS"),
                inference_blocker="Scoped controls, inference endpoint qualification and public tuning remain incomplete"
                    if m10_posterior_report.get("qualified") else "Full-prior posterior reference, scoped controls and tuning remain incomplete",
                next_action="Calibrate M10 controls and forecast complete inference using the remaining family budget"
                    if m10_posterior_report.get("qualified") else "Qualify the full-prior reference and scoped controls before retained inference")
            if m10_control_report:
                forecast = m10_control_report.get("forecast", {})
                decision.update(inference_blocker=forecast.get("combined_status", m10_control_report.get("status")),
                    next_action="Preserve per-arm control/cost evidence; assess required endpoint/reference work and the remaining model queue")
            if m10_endpoint_report:
                decision.update(engineering_validity=m10_endpoint_report.get("status"),
                    inference_blocker="Complete inference allocation and public tuning remain required",
                    next_action="Assess the measured full-endpoint forecast before any M10 inference launch")
            if m10_history["status"] != "NOT_RUN":
                decision.update(comparisons=m10_history["comparisons"], comparison_scope="M10-T2",
                    retained_sampling=m10_history, mixing_mcse="SEE_M10_RETAINED_LOOKS",
                    inference_blocker=None if m10_history.get("artifact_valid") else m10_history["status"],
                    next_action=m10_history.get("next_action", "Inspect the preserved M10 attempt"))
    if include_ksc:
        decision["scope_results"]["KSC"] = dict(role="surrogate_HMC_model_posterior_test", **ksc_history)
    write_json(output / "decision.json", decision)
    register = model_register()
    if include_lg1 or include_lg2:
        register["families"][0].update(inference_status=decision["comparisons"]["B/R"],
                                    comparison_status=decision["comparisons"],
                                    comparison_scope=decision["comparison_scope"],
                                    scope_results={name: value for name, value in decision["scope_results"].items()
                                                   if name in ("LG1", "LG2")},
                                    executed_stage=args.stage, execution_status=decision["execution_status"])
        if include_lg2 and lg2_endpoint.get("qualified"):
            register["families"][0]["variant_status"]["LG2"] = "CLAIM_ENDPOINT_AND_COMPLETE_PILOT_COST_QUALIFIED"
    if include_c01:
        control_register = next(row for row in register["families"] if row["id"] == "C01")
        control_register.update(scope_results=decision["scope_results"]["C01"], executed_stage=args.stage,
            inference_status=c01_sampling.get("status", "NOT_RUN"),
            missing=["d100 control", "heuristic comparisons", "wider model-inference evidence"])
    if include_sv20:
        sv_register = next(row for row in register["families"] if row["id"] == "M02")
        sv_register.update(scope_results=decision["scope_results"]["SV20"], executed_stage=args.stage,
            status="PARTIAL_SV20_INFERENCE_EXECUTED" if sv_history.get("artifact_valid") else sv_history["status"],
            inference_status=sv_history["comparisons"], missing=["resolved posterior comparisons",
                "independent finite-D endpoint and reference", "C tuning and retained comparisons",
                "heuristic comparisons", "remaining SV variants"] +
                ([] if sv_history.get("arms", {}).get("B", {}).get("tuning_passed") else ["public HMC tuning"]))
        if sv_history.get("artifact_valid"):
            sv_register["variant_status"]["SV20"] = "BOUNDED_B_R_INFERENCE_EXECUTED"
    if include_m10:
        m10_register = next(row for row in register["families"] if row["id"] == "M10")
        m10_register.update(scope_results=decision["scope_results"]["M10"], executed_stage=args.stage,
            status="PARTIAL_M10_INFERENCE_EXECUTED" if m10_history.get("artifact_valid") else m10_history["status"],
            inference_status=m10_history["comparisons"], missing=["resolved posterior comparisons",
                "independent finite-D endpoint and reference", "C tuning and retained comparisons", "heuristic comparisons"] +
                ([] if m10_endpoint_report.get("qualified") else ["inference endpoint qualification"]) +
                ([] if m10_history.get("arms", {}).get("B", {}).get("tuning_passed") else ["public HMC tuning"]))
        if m10_history.get("artifact_valid"):
            m10_register["variant_status"]["concrete-score-study"] = "BOUNDED_B_R_INFERENCE_EXECUTED"
    if include_ksc:
        ksc_register = next(row for row in register["families"] if row["id"] == "M03")
        ksc_register.update(scope_results=decision["scope_results"]["KSC"],
            status="PARTIAL_KSC_INFERENCE_EXECUTED" if ksc_history.get("artifact_valid") else ksc_history["status"],
            inference_status=ksc_history["comparisons"], executed_stage=args.stage,
            missing=["resolved KSC posterior comparisons", "heuristic comparisons",
                     "exact-transformed variant and longer horizons"] +
                (["independent finite-D endpoint and reference"]
                 if ksc_history["comparisons"].get("D/R", "NOT_RUN") == "NOT_RUN" else []) +
                (["C tuning and retained comparisons"]
                 if ksc_history["comparisons"].get("C/D", "NOT_RUN") == "NOT_RUN" else []))
    write_json(output / "model-register.json", register)
    manifest.update(wall_seconds=time.monotonic() - begun, source_sha256={
        str(path.relative_to(ROOT)): file_hash(path) for path in (
            ROOT / __file__, ROOT / "scripts/run_surrogate_hmc_program.py",
            ROOT / "scripts/prepare_surrogate_hmc_nonlinear_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_nonlinear_reference.py",
            ROOT / "scripts/qualify_surrogate_hmc_nonlinear_consumer.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py",
            ROOT / "scripts/prepare_surrogate_hmc_nonlinear_posterior.py",
            ROOT / "scripts/calibrate_surrogate_hmc_nonlinear_controls.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_nonlinear_posterior_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py",
            ROOT / "scripts/prepare_surrogate_hmc_sv_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_sv_reference.py",
            ROOT / "scripts/prepare_surrogate_hmc_sv_posterior.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_sv_posterior_reference.py",
            ROOT / "scripts/qualify_surrogate_hmc_sv_consumer.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_sv_consumer.py",
            ROOT / "scripts/calibrate_surrogate_hmc_sv_controls.py",
            ROOT / "scripts/surrogate_hmc_lg1_probe.py", ROOT / "bayesfilter/testing/surrogate_hmc_lg1_tf.py",
            ROOT / "scripts/prepare_surrogate_hmc_lg1_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_lg1_reference.py",
            ROOT / "scripts/calibrate_surrogate_hmc_lg1_controls.py",
            ROOT / "bayesfilter/inference/surrogate_hmc_dctr_selection.py",
            ROOT / "scripts/tune_surrogate_hmc_lg1.py",
            ROOT / "scripts/sample_surrogate_hmc_lg1.py",
            ROOT / "bayesfilter/inference/surrogate_hmc_equivalence.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_lg1_scoped.py",
            ROOT / "scripts/qualify_surrogate_hmc_lg1_finite_endpoint.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_lg1_finite_endpoint.py",
            ROOT / "scripts/prepare_surrogate_hmc_lg1_finite_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_lg1_finite_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_lg1_finite_consumer.py",
            ROOT / "scripts/run_surrogate_hmc_lg1_finite.py",
            ROOT / "scripts/prepare_surrogate_hmc_linear_reference.py",
            ROOT / "scripts/qualify_surrogate_hmc_linear_consumer.py",
            ROOT / "scripts/calibrate_surrogate_hmc_linear_controls.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_linear_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_linear_consumer.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_gaussian_control.py",
            ROOT / "scripts/qualify_surrogate_hmc_gaussian_controls.py",
            ROOT / "scripts/run_surrogate_hmc_gaussian_controls.py")})
    write_json(output / "run-manifest.json", manifest)
    (output / "result.md").write_text(
        f"# Surrogate-HMC execution\n\nStatus: {decision['status']}. "
        "See decision.json, diagnostics and the complete per-job log. "
        f"Posterior comparisons: {decision['comparisons']}. All-model validation remains incomplete.\n\n"
        + (f"Gaussian mechanism comparisons: {decision['scope_results']['C01']['comparisons']}. "
           "These controls do not establish DCTR model-inference performance.\n\n" if include_c01 else "") +
        (f"KSC T2 posterior comparisons: {ksc_history.get('comparisons', {})}. "
         f"Retained surrogate-HMC test recorded: {ksc_history.get('retained_surrogate_hmc_tested', False)}.\n\n"
         if include_ksc else "") +
        (f"M10 T2 posterior comparisons: {m10_history.get('comparisons', {})}. "
         f"Retained surrogate-HMC test recorded: {m10_history.get('retained_surrogate_hmc_tested', False)}.\n\n"
         if include_m10 else "") +
        "| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |\n"
        "|---|---|---|---|---|---|\n"
        f"| {args.stage} | {decision['comparisons']} | See stage reports | Monte Carlo and reference precision | {decision['next_action']} | All-model posterior recovery |\n\n"
        "| Inference status | Evidence |\n|---|---|\n"
        "| Hard veto screen | See consumer checks, tuning verification and retained look diagnostics |\n| Statistically supported ranking | None |\n"
        "| Descriptive differences | Timing, acceptance and ESS alone do not rank methods |\n"
        "| Default readiness | Not established |\n| Next evidence | Resolve four posterior comparisons and heuristic checks |\n")
    print(json.dumps({"output": str(output), "decision": decision["status"], "probe": probe,
                      "charged": ledger["charged"]}, allow_nan=False))
    return 1 if failed else (0 if args.stage == "P0" else 2)
