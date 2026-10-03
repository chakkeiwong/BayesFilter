"""Bounded CPU driver for a noncanonical independent R paper replication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/iapf-r-paper-replication-2026-09-20.md"
ARTIFACTS = ROOT / "docs/plans/artifacts/iapf-r-paper-replication-20260920-01"


def snapshot_sources(root, sources, snapshots):
    hashes = {}
    for name in sources:
        source = root / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = snapshots / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != hashes[name]:
            raise RuntimeError(f"source changed while snapshotting: {name}")
    return hashes


def r_worker_command(snapshots, runner, output, *arguments):
    """Execute the captured runner and its captured source dependencies."""
    return ["Rscript", "--vanilla", str(snapshots / runner), str(snapshots),
            str(output), *(str(value) for value in arguments)]


def run_bounded_worker(command, *, cwd, env, log, timeout):
    """Timeout covers phase subprocesses as well as their supervising R worker."""
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True) as worker:
        try:
            return worker.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("pilot", "replication", "oracle", "sensitivity", "fit_audit", "controller_diagnostic", "floor_diagnostic", "floor_repair"), required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--campaign", choices=("paper", "repair", "validation", "floor_validation", "d40_confirmation", "small_dimension_completion"), default="paper")
    parser.add_argument("--fit-mode", choices=("paper_eq15", "relative_l2", "relative_l2_nlminb", "log_quadratic"), default="paper_eq15")
    parser.add_argument("--floor-power", type=float, choices=(1, 2, 3, 8), default=2)
    parser.add_argument("--doubling-mode", choices=("first_full_window", "after_k"), default="first_full_window")
    parser.add_argument("--fit-maxit", type=int, choices=(200, 1000, 5000), default=200)
    args = parser.parse_args()
    artifacts, plan, seconds, launches, timeout_cap = ARTIFACTS, PLAN, 1800, 8, 300
    if args.campaign == "repair":
        artifacts = ROOT / "docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01"
        plan = "docs/plans/iapf-r-reference-gap-repair-2026-09-20.md"
        seconds, launches, timeout_cap = 1550, 10, 400
        if args.fit_mode == "log_quadratic":
            plan = "docs/plans/iapf-r-log-fit-repair-2026-09-20.md"
            launches, timeout_cap = 12, 200
            pilot = args.mode == "pilot" and args.repeats <= 4 and args.data_seed == 68000020
            follow_on = (args.mode == "replication" and args.repeats == 16
                         and args.first == 101 and args.data_seed == 69000020 and args.timeout <= 165)
            if args.dimension != 20 or not (pilot or follow_on):
                parser.error("log fit requires the planned d20 pilot or fresh 16-repeat batch")
    elif args.campaign == "small_dimension_completion":
        artifacts = ROOT / "docs/plans/artifacts/iapf-r-small-dimension-completion-20260921-01"
        plan = "docs/plans/iapf-r-small-dimension-completion-2026-09-21.md"
        seconds, launches, timeout_cap = 1200, 5, 400
        planned = {(5, 84000005, 1001, 32): 300,
                   (10, 84000010, 1001, 32): 350,
                   (20, 84000020, 1001, 32): 400}
        key = (args.dimension, args.data_seed, args.first, args.repeats)
        if (key not in planned or args.timeout > planned[key]
                or args.mode != "replication" or args.fit_mode != "log_quadratic"
                or args.floor_power != 8 or args.doubling_mode != "first_full_window"
                or args.fit_maxit != 200):
            parser.error("outside the frozen small-dimension completion schedule/settings")
    elif args.campaign == "d40_confirmation":
        artifacts = ROOT / "docs/plans/artifacts/iapf-r-d40-confirmation-20260920-01"
        plan = "docs/plans/iapf-r-d40-confirmation-2026-09-20.md"
        seconds, launches, timeout_cap = 1400, 3, 600
        planned = {(40, 80000040, 801, 32), (40, 82000040, 901, 32)}
        if ((args.dimension, args.data_seed, args.first, args.repeats) not in planned
                or args.mode != "replication" or args.fit_mode != "log_quadratic"
                or args.floor_power != 8 or args.doubling_mode != "first_full_window"
                or args.fit_maxit != 200):
            parser.error("outside the frozen d40 confirmation schedule/settings")
    elif args.campaign == "floor_validation":
        artifacts = ROOT / "docs/plans/artifacts/iapf-r-positive-floor-validation-20260920-01"
        plan = "docs/plans/iapf-r-positive-floor-validation-2026-09-20.md"
        seconds, launches, timeout_cap = 3800, 4, 1500
        planned = {(80, 80000080, 501, 32): 1500,
                   (80, 81000080, 601, 32): 1500,
                   (40, 80000040, 701, 32): 600}
        key = (args.dimension, args.data_seed, args.first, args.repeats)
        if (key not in planned or args.timeout > planned[key]
                or args.mode != "replication" or args.fit_mode != "log_quadratic"
                or args.floor_power != 8 or args.doubling_mode != "first_full_window"
                or args.fit_maxit != 200):
            parser.error("outside the frozen positive-floor validation schedule/settings")
    elif args.campaign == "validation":
        artifacts = ROOT / "docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01"
        plan = "docs/plans/iapf-r-log-fit-validation-2026-09-20.md"
        seconds, launches, timeout_cap = 1800, 8, 800
        planned = {
            ("replication", 20, 70000020, 201, 32): 400,
            ("replication", 20, 71000020, 301, 32): 400,
            ("pilot", 40, 72000040, 1, 4): 300,
            ("pilot", 80, 72000080, 1, 4): 600,
            ("replication", 40, 73000040, 101, 16): 600,
            ("replication", 80, 74000080, 101, 8): 800,
            ("controller_diagnostic", 80, 72000080, 1, 1): 200,
            ("floor_diagnostic", 80, 72000080, 1, 1): 120,
            ("floor_repair", 80, 72000080, 1, 1): 600,
        }
        key = (args.mode, args.dimension, args.data_seed, args.first, args.repeats)
        if (key not in planned or args.timeout > planned[key]
                or args.fit_mode != "log_quadratic"
                or args.floor_power != (8 if args.mode == "floor_repair" else 2)
                or args.doubling_mode != "first_full_window" or args.fit_maxit != 200):
            parser.error("outside the frozen log-fit validation schedule/settings")
    elif args.fit_mode != "paper_eq15" or args.floor_power != 2 or args.doubling_mode != "first_full_window" or args.fit_maxit != 200:
        parser.error("alternative settings belong to the explicitly labeled repair campaign")
    output = (ROOT / args.output).resolve()
    if output.parent != artifacts or not output.name.startswith("attempt"):
        parser.error("use a fresh attempt directory immediately under the planned root")
    if not 0 < args.timeout <= timeout_cap or args.dimension not in (5, 10, 20, 40, 80):
        parser.error("outside planned dimension/timeout bounds")
    previous = [json.loads(path.read_text()) for path in artifacts.glob("attempt*/manifest.json")]
    charged = sum(row.get("wall_seconds", row["timeout_seconds"]) for row in previous)
    # One localized logging/status repair shares the original 600-second repair
    # allocation. It adds no scientific arm or aggregate compute allowance.
    infrastructure_retry = False
    if args.campaign == "validation" and args.mode == "floor_repair" and len(previous) == 8:
        failed = artifacts / "attempt08-positive-floor-repair"
        if (failed / "manifest.json").exists():
            prior = json.loads((failed / "manifest.json").read_text())
            log = (failed / "run.log").read_text()
            infrastructure_retry = (prior["status"] == "failed"
                and "setting stdout = TRUE" in log
                and "condition has length > 1" in log
                and args.timeout + prior["wall_seconds"] <= 600)
        if infrastructure_retry:
            launches = 9
    if args.campaign == "repair" and args.fit_mode == "log_quadratic":
        log_fit_charged = sum(row.get("wall_seconds", row["timeout_seconds"])
                             for row in previous if row.get("fit_mode") == "log_quadratic")
        if log_fit_charged + args.timeout > 200:
            parser.error("insufficient remaining log-fit pilot allocation")
    if len(previous) >= launches or charged + args.timeout > seconds:
        parser.error("insufficient remaining launch/worker-time budget")
    if sum(row["status"] == "running" for row in previous) >= 2:
        parser.error("two workers are already running")
    runner = ("docs/benchmarks/diagnose_iapf_r_paper_scale_oracles.R" if args.mode == "oracle"
              else "docs/benchmarks/replicate_iapf_paper_linear.R")
    if args.mode == "sensitivity":
        if args.campaign != "repair" or args.fit_mode != "relative_l2":
            parser.error("sensitivity mode requires the labeled relative-fit repair campaign")
        runner = "docs/benchmarks/diagnose_iapf_r_fit_sensitivity.R"
    if args.mode == "fit_audit":
        if args.campaign != "repair" or args.dimension != 20 or args.timeout > 150:
            parser.error("fit audit requires the bounded d20 repair allocation")
        plan = "docs/plans/iapf-r-paper-code-audit-2026-09-20.md"
        runner = "docs/benchmarks/diagnose_iapf_r_paper_code_audit.R"
    if args.mode == "controller_diagnostic":
        if args.campaign != "validation":
            parser.error("controller diagnostic requires the frozen validation campaign")
        plan = "docs/plans/iapf-r-d80-controller-diagnosis-2026-09-20.md"
        runner = "docs/benchmarks/diagnose_iapf_r_log_controller.R"
    if args.mode == "floor_diagnostic":
        if args.campaign != "validation":
            parser.error("floor diagnostic requires the validation campaign")
        plan = "docs/plans/iapf-r-d80-floor-diagnosis-2026-09-20.md"
        runner = "docs/benchmarks/diagnose_iapf_r_floor.R"
    if args.mode == "floor_repair":
        if args.campaign != "validation":
            parser.error("floor repair requires the validation campaign")
        plan = "docs/plans/iapf-r-d80-positive-floor-repair-2026-09-20.md"
        runner = "docs/benchmarks/diagnose_iapf_r_positive_floor.R"
    output.mkdir(parents=True, exist_ok=False)
    sources = ["docs/benchmarks/reference_iapf_paper.R",
               "docs/benchmarks/reference_iapf_support_diagnostics.R",
               "docs/benchmarks/replicate_iapf_paper_linear.R",
               "docs/benchmarks/run_iapf_r_replication.py",
               "tests/reference_iapf_paper.R", "tests/highdim/test_iapf_r_paper_reference.py", plan]
    if args.campaign in ("repair", "validation", "floor_validation", "d40_confirmation", "small_dimension_completion"):
        sources.append("tests/reference_iapf_relative_fit.R")
    if args.campaign in ("floor_validation", "d40_confirmation", "small_dimension_completion"):
        sources.extend(("docs/benchmarks/diagnose_iapf_r_validation_tails.R",
                        "docs/benchmarks/summarize_iapf_r_positive_floor_validation.py",
                        "docs/benchmarks/summarize_iapf_r_log_fit.py",
                        "docs/benchmarks/summarize_iapf_r_replication.py"))
    if args.campaign in ("d40_confirmation", "small_dimension_completion"):
        sources.append("docs/benchmarks/summarize_iapf_r_d40_confirmation.py")
    if args.campaign == "small_dimension_completion":
        sources.append("docs/benchmarks/summarize_iapf_r_small_dimension_completion.py")
    if runner not in sources:
        sources.append(runner)
    if args.mode == "fit_audit":
        sources.extend("docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01/"
                       + attempt + "/results/iapf-1-failure.rds" for attempt in
                       ("attempt08-pilot-d20", "attempt09-pilot-d20-nlminb"))
    if args.mode == "controller_diagnostic":
        sources.extend(("docs/plans/iapf-r-log-fit-validation-2026-09-20.md",
            "docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01/attempt04-d80-pilot/results/iapf-1.rds"))
    if args.mode in ("floor_diagnostic", "floor_repair"):
        saved = "docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01/attempt06-d80-diagnostic/results/"
        sources.extend(saved + name for name in (
            "model.rds", "replayed-result.rds", "prefixes.csv",
            "filter-input-15.rds", "filter-input-19.rds", "filter-input-20.rds"))
    if args.mode == "floor_repair":
        sources.extend("docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01/" + name
                       for name in ("attempt07-floor-diagnostic/results/floor-curve.csv",
                                    "attempt01-d20-a/results/prefixes.csv",
                                    "attempt02-d20-b/results/prefixes.csv"))
    snapshots = output / "sources"
    snapshots.mkdir()
    hashes = snapshot_sources(ROOT, sources, snapshots)
    command = r_worker_command(snapshots, runner, output / "results",
               args.dimension, args.repeats, args.first, args.data_seed, args.mode,
               args.fit_mode, args.floor_power, args.doubling_mode, args.fit_maxit)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    manifest = dict(schema="independent_r_iapf_replication_v1", status="running",
        started_at=datetime.now(timezone.utc).isoformat(), command=command, driver_command=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=hashes, paper_sha256=hashlib.sha256((ROOT /
            ".localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf").read_bytes()).hexdigest(),
        execution_source_root=str(snapshots), executed_captured_sources=True,
        environment=dict(python=sys.executable, R=subprocess.check_output(
            ["Rscript", "--version"], stderr=subprocess.STDOUT, text=True).strip()),
        cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
        execution_exception="owner_requested_independent_R_reference",
        data_version="newly_simulated_section_5_2_linear_Gaussian_v1", data_seed=args.data_seed,
        method_seeds="53000000+dimension*100000+replication*10+method_id",
        timeout_seconds=args.timeout, prior_charged_worker_seconds=charged,
        plan_file=plan, result_file=str(output / "results/replicates.csv"), canonical=False,
        campaign=args.campaign, dimension=args.dimension, mode=args.mode,
        first_replication=args.first, repeats=args.repeats,
        fit_mode=args.fit_mode, floor_tail_power=args.floor_power,
        doubling_mode=args.doubling_mode, equation_15_objective=args.fit_mode == "paper_eq15",
        fit_maxit=args.fit_maxit, max_iterations=20, max_particles=16000,
        worker_budget_seconds=seconds, launch_budget=launches)
    if infrastructure_retry:
        manifest.update(infrastructure_retry_of="attempt08-positive-floor-repair",
                        retry_reason="R system2 stderr capture returned text instead of numeric exit code",
                        shared_repair_timeout_seconds=600)
    if args.mode == "oracle":
        manifest.update(dimension_ladder=[5, 10, 20, 40, 80],
                        result_file=str(output / "results/oracle-checks.csv"),
                        role="mathematical_oracles_not_fitted_iapf_replication",
                        method_seeds="59000000+dimension*1000+replication")
    if args.mode == "sensitivity":
        manifest.update(dimension_ladder=[5, 10],
                        result_file=str(output / "results/sensitivity.csv"),
                        role="calibration_only_no_setting_selection",
                        method_seeds="63000000+dimension; common seed across sensitivity settings",
                        sensitivity_settings=["floor2/first_full_window", "floor1/first_full_window",
                                              "floor3/first_full_window", "floor2/after_k"])
    if args.mode == "fit_audit":
        manifest.update(result_file=str(output / "results/support.csv"),
                        role="frozen_fit_support_diagnostic_no_promotion",
                        fit_mode=["relative_l2", "relative_l2_nlminb"],
                        equation_15_objective=False, data_seed=[66000020, 67000020],
                        method_seeds="replay55000011; support68000000+case*1000+rep; smoothing+100")
    if args.mode == "controller_diagnostic":
        manifest.update(role="exact_replay_and_oracle_diagnostic_no_promotion",
                        result_file=str(output / "results/diagnostic-summary.csv"),
                        method_seeds="failed_run_replay61000011; exact_twist_oracle72000980")
    if args.mode == "floor_diagnostic":
        manifest.update(role="fixed_guide_floor_diagnostic_no_promotion",
                        result_file=str(output / "results/summary.csv"),
                        method_seeds="saved RNG states for filter calls15,19,20",
                        counterfactuals=["zero_floor_time93", "zero_floor_all_times"])
    if args.mode == "floor_repair":
        manifest.update(role="optional_positive_floor_repair_no_default_promotion",
                        result_file=str(output / "results/phases.csv"),
                        dimension_ladder=[20, 80], data_seed=[70000020, 71000020, 72000080, 74000080],
                        phases=["d20-a:201:204", "d20-b:301:304", "d80-pilot:1:2", "d80-fresh:101:104"],
                        method_seeds="healthy76000001; saved RNG calls15/19/20; standard replication formula")
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        with (output / "run.log").open("w") as log:
            returncode = run_bounded_worker(command, cwd=ROOT, env=env, log=log,
                                            timeout=args.timeout)
        manifest.update(status="complete" if returncode == 0 else "failed",
                        exit_code=returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="timeout", exit_code=124)
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        data = output / "results/observations.csv"
        if data.exists():
            manifest["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
        manifest["observation_file_sha256"] = {
            str(p.relative_to(output / "results")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output / "results").rglob("observations*.csv")}
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "exit_code", "wall_seconds", "result_file")}))
    return manifest["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
