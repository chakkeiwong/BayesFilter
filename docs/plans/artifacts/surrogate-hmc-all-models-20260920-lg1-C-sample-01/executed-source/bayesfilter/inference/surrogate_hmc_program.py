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
                     "status": "PARTIAL_LG1_IMPLEMENTATION" if ident == "M01" else "SCOPE_UNRESOLVED",
                     "variant_status": {variant: "REFERENCE_CONTROL_TUNING_SAMPLING_AVAILABLE" if variant == "LG1"
                                        else "SCOPE_UNRESOLVED" for variant in variants},
                     "inference_status": "NOT_RUN",
                     "missing": (["resolved posterior comparisons", "heuristic comparisons",
                                  "remaining linear scopes"] if ident == "M01"
                                 else ["concrete scope card", "qualified consumer and reference",
                                       "scope-specific tuning and budget forecast"]),
                     "source": PLAN})
    return {"schema": SCHEMA, "question": "Does imperfect full-DCTR surrogate HMC recover model parameter posteriors?",
            "all_models_validated": False, "coverage_complete": False, "families": rows}


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
        charged = payload.get("charged_this_run", payload["charged"])
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
            if blocked:
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
    parser.add_argument("--stage", choices=("P0", "P1", "P2-reference", "P2-controls", "P2-tune", "P2-sample", "P2-endpoint", "P2-finite-reference", "P2-C-tune", "P2-C-sample", "all"), default="all")
    parser.add_argument("--scope", choices=("all", "LG1"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gpu-seconds", type=float, default=GPU_CAP)
    parser.add_argument("--max-cpu-worker-seconds", type=float, default=CPU_CAP)
    parser.add_argument("--probe-timeout", type=float, default=900.)
    parser.add_argument("--reference-root", type=Path, help="Prepared LG1 reference root for standalone P2-controls")
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
    ledger = validated_budget(args.max_gpu_seconds, args.max_cpu_worker_seconds)
    charge_previous_runs(ledger, [*ROOT.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"),
                                 *args.previous_accounting])
    prior_charged = dict(ledger["charged"])
    output = args.output.resolve()
    jobs = []
    if args.stage in ("P1", "all"):
        jobs.append(dict(name="P1-LG1", device="gpu", timeout=args.probe_timeout,
                         command=[sys.executable, str(ROOT / "scripts/surrogate_hmc_lg1_probe.py"),
                                  "--output", str(output / "P1-LG1"), "--particle-count", "1024"]))
    if args.stage in ("P2-reference", "all"):
        jobs.append(dict(name="P2-reference-job", device="cpu", timeout=min(300., args.probe_timeout),
                         command=[sys.executable, str(ROOT / "scripts/prepare_surrogate_hmc_lg1_reference.py"),
                                  "--output", str(output / "P2-reference")]))
    if args.stage in ("P2-controls", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        if reference_root is None:
            parser.error("P2-controls requires --reference-root from a completed P2-reference stage")
        jobs.append(dict(name="P2-controls-job", device="gpu", timeout=min(900., args.probe_timeout),
                         command=[sys.executable, str(ROOT / "scripts/calibrate_surrogate_hmc_lg1_controls.py"),
                                  "--output", str(output / "P2-controls"), "--reference-root", str(reference_root.resolve())]))
    if args.stage in ("P2-tune", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        if reference_root is None or selection is None:
            parser.error("P2-tune requires --reference-root and --selection")
        jobs.append(dict(name="P2-tune-job", device="gpu", timeout=args.tuning_timeout, cpu_allowance=args.tuning_cpu_seconds,
                         command=[sys.executable, str(ROOT / "scripts/tune_surrogate_hmc_lg1.py"),
                                  "--output", str(output / "P2-tune"), "--reference-root", str(reference_root.resolve()),
                                  "--selection", str(selection.resolve())]))
    if args.stage in ("P2-sample", "all"):
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
    if args.stage in ("P2-endpoint", "all"):
        reference_root = output / "P2-reference" if args.stage == "all" else args.reference_root
        selection = output / "P2-controls/selection.json" if args.stage == "all" else args.selection
        if reference_root is None or selection is None:
            parser.error("P2-endpoint requires --reference-root and --selection")
        jobs.append(dict(name="P2-endpoint-job", device="gpu", timeout=args.endpoint_timeout,
                         command=[sys.executable, str(ROOT / "scripts/qualify_surrogate_hmc_lg1_finite_endpoint.py"),
                                  "--output", str(output / "P2-endpoint"), "--reference-root", str(reference_root.resolve()),
                                  "--selection", str(selection.resolve())]))
    if args.stage in ("P2-finite-reference", "all"):
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
        if args.stage not in ("P2-C-" + c_stage, "all"):
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
    if args.stage == "all":
        prerequisites = {
            "P1-LG1": [], "P2-reference-job": [],
            "P2-controls-job": ["P1-LG1", "P2-reference-job"],
            "P2-tune-job": ["P2-controls-job"], "P2-sample-job": ["P2-tune-job"],
            "P2-endpoint-job": ["P2-controls-job"],
            "P2-finite-reference-job": ["P2-endpoint-job"],
            "P2-C-tune-job": ["P2-finite-reference-job"],
            "P2-C-sample-job": ["P2-C-tune-job"],
        }
        for job in jobs:
            job["depends_on"] = prerequisites[job["name"]]
    if args.dry_run:
        for job in jobs:
            reserve_job(ledger, job["device"], job["timeout"], job.get("cpu_allowance"))
    launch = {"schema": SCHEMA, "stage": args.stage, "scope": args.scope,
              "jobs": jobs,
              "required_inputs": [PLAN, EXECUTION_PLAN], "scope_card": lg1_scope(args.stage),
              "endpoint_force_identities": "issued from actual frozen inputs by worker before evaluation",
              "inference_forecast": "NOT_MEASURED; inference launch prohibited",
              "remaining_budget": {key: ledger["limits"][key]-ledger["charged"][key] for key in ledger["limits"]},
              "prior_runs": ledger["prior_runs"], "inference_ready": False}
    if args.dry_run and args.stage == "P2-sample":
        launch["inference_forecast"] = retained_forecast(tuning_root)
        validate_retained_forecast(launch["inference_forecast"], jobs[0])
    elif args.dry_run and args.stage == "P2-C-sample":
        launch["inference_forecast"] = retained_forecast(c_tuning_root, "C")
        validate_retained_forecast(launch["inference_forecast"], jobs[0])
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
        write_json(output / "scopes/LG1.json", launch["scope_card"])
        def preflight(job):
            reserve_job(ledger, job["device"], job["timeout"], job.get("cpu_allowance"))
            if job["name"] in ("P2-sample-job", "P2-C-sample-job"):
                forecast = (retained_forecast(tuning_root) if job["name"] == "P2-sample-job"
                            else retained_forecast(c_tuning_root, "C"))
                validate_retained_forecast(forecast, job)
                manifest.update(inference_forecast=forecast, inference_ready=True)
                manifest.setdefault("inference_forecasts", {})[job["name"]] = forecast
                write_json(output / "run-manifest.json", manifest)
                return forecast
            return None
        outcomes = execute_job_graph(jobs, output, ledger, before_job=preflight)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # An interruption raises before execute_job_graph returns its mapping.
        if (output / "execution-results.json").exists():
            outcomes = json.loads((output / "execution-results.json").read_text())
        probe = next((row for row in reversed(list(outcomes.values())) if "exit_code" in row), None)
        used = resource.getrusage(resource.RUSAGE_SELF)
        ledger["charged"]["cpu_worker_seconds"] += used.ru_utime + used.ru_stime - parent_cpu.ru_utime - parent_cpu.ru_stime
        ledger["remaining"] = {k: ledger["limits"][k] - ledger["charged"][k] for k in ledger["limits"]}
        ledger["charged_this_run"] = {key: ledger["charged"][key]-prior_charged[key] for key in ledger["limits"]}
        write_json(output / "accounting.json", ledger)
    diagnostic_path = output / "P1-LG1/diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    reference_path = (args.reference_root or output / "P2-reference") / "report.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    selection_path = (args.selection.parent if args.selection else output / "P2-controls") / "report.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    tuning_path = (args.tuning_root or output / ("P2-C-tune" if args.stage.startswith("P2-C") else "P2-tune")) / "report.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    sampling_path = output / ("P2-C-sample/report.json" if args.stage.startswith("P2-C") else "P2-sample/report.json")
    sampling = json.loads(sampling_path.read_text()) if sampling_path.exists() else {}
    c_sampling_path = output / "P2-C-sample/report.json"
    c_sampling = json.loads(c_sampling_path.read_text()) if c_sampling_path.exists() else {}
    c_tuning_path = ((args.tuning_root if args.stage == "P2-C-sample" else None)
                     or output / "P2-C-tune") / "report.json"
    c_tuning = json.loads(c_tuning_path.read_text()) if c_tuning_path.exists() else {}
    endpoint_path = (args.endpoint_root or output / "P2-endpoint") / "report.json"
    endpoint = json.loads(endpoint_path.read_text()) if endpoint_path.exists() else {}
    finite_reference_path = (args.finite_reference_root or output / "P2-finite-reference") / "report.json"
    finite_reference = json.loads(finite_reference_path.read_text()) if finite_reference_path.exists() else {}
    failed = error is not None or any(row["status"] != "COMPLETED" for row in outcomes.values())
    completed_any = any(row["status"] == "COMPLETED" for row in outcomes.values())
    decision = {"schema": SCHEMA, "status": "EXECUTION_FAILED" if failed else "INCOMPLETE_IMPLEMENTATION",
                "error": error, "all_models_validated": False,
                "engineering_validity": diagnostic.get("status", "NOT_CHECKED"),
                "comparisons": {"B/R":"NOT_RUN", "C/D":"NOT_RUN", "D/R":"NOT_RUN", "C/R":"NOT_RUN"} | sampling.get("comparisons", {}),
                "reference_quality": "QUALIFIED_LG1_MODEL_QUADRATURE" if reference.get("qualified") else "NOT_QUALIFIED",
                "reference": reference,
                "dctr_control_selection": selection.get("verdict", "NOT_RUN"),
                "scoped_hmc_tuning": tuning,
                "finite_endpoint_hmc_tuning": c_tuning,
                "retained_sampling": sampling,
                "finite_endpoint_sampling": c_sampling,
                "finite_endpoint_qualification": endpoint,
                "finite_posterior_reference": finite_reference,
                "execution_status": ("PARTIAL" if completed_any else "FAILED") if failed else "COMPLETED",
                "job_outcomes": outcomes,
                "mixing_mcse": "SEE_RETAINED_LOOKS" if sampling else "NOT_RUN", "heuristic_dominance": "NOT_RUN",
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
    write_json(output / "decision.json", decision)
    register = model_register()
    register["families"][0].update(inference_status=decision["comparisons"]["B/R"],
                                    comparison_status=decision["comparisons"],
                                    executed_stage=args.stage, execution_status=decision["execution_status"])
    write_json(output / "model-register.json", register)
    manifest.update(wall_seconds=time.monotonic() - begun, source_sha256={
        str(path.relative_to(ROOT)): file_hash(path) for path in (
            ROOT / __file__, ROOT / "scripts/run_surrogate_hmc_program.py",
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
            ROOT / "scripts/run_surrogate_hmc_lg1_finite.py")})
    write_json(output / "run-manifest.json", manifest)
    (output / "result.md").write_text(
        f"# Surrogate-HMC execution\n\nStatus: {decision['status']}. "
        "See decision.json, diagnostics and the complete per-job log. "
        f"Posterior comparisons: {decision['comparisons']}. All-model validation remains incomplete.\n\n"
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
