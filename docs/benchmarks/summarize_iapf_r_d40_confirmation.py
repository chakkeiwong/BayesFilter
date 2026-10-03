"""Post-run reference diagnostics for the frozen, independent d40 confirmation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_iapf_r_replication import run_bounded_worker
from summarize_iapf_r_positive_floor_validation import inspect_attempt, CORE, CORE_HASH
from summarize_iapf_r_log_fit import read

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_ROOT = "docs/plans/artifacts/iapf-r-d40-confirmation-20260920-01"
PLAN = "docs/plans/iapf-r-d40-confirmation-2026-09-20.md"
SCHEDULE = {"attempt01-same-data": (40, 80000040, 801),
            "attempt02-fresh-data": (40, 82000040, 901)}
PRIOR = "docs/plans/artifacts/iapf-r-positive-floor-validation-20260920-01/attempt03-d40-control"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def post_used(campaign):
    paths = [*campaign.glob("attempt*/diagnostic-manifest.json"),
             *campaign.glob("report-manifest*.json")]
    rows = [json.loads(p.read_text()) for p in paths]
    return sum(r["wall_seconds"] if "wall_seconds" in r else r["timeout_seconds"] for r in rows)


def diagnose(name, campaign, repo, *, schedule=None, plan=PLAN, post_budget=90):
    attempt = campaign / name
    run = json.loads((attempt / "manifest.json").read_text())
    assert run["status"] == "complete" and run["exit_code"] == 0
    expected = (SCHEDULE if schedule is None else schedule)[name]
    assert (run["dimension"], run["data_seed"], run["first_replication"]) == expected
    assert run["plan_file"] == plan
    sources = attempt / "sources"
    for source, expected in run["source_sha256"].items():
        assert digest(sources / source) == expected, source
    assert run["source_sha256"][CORE] == CORE_HASH
    target = attempt / "diagnostic-manifest.json"
    assert not target.exists()
    timeout = min(40, post_budget - post_used(campaign))
    assert timeout > 0, "post-run allocation exhausted"
    command = ["Rscript", "--vanilla", str(sources / "docs/benchmarks/diagnose_iapf_r_validation_tails.R"),
               str(sources), str(attempt / "results")]
    manifest = dict(status="running", command=command, started_at=datetime.now(timezone.utc).isoformat(),
        timeout_seconds=timeout, environment=run["environment"], git_commit=run["git_commit"],
        cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
        role="post_run_independent_R_reference_diagnostics", source_sha256=run["source_sha256"],
        input_data_sha256=run["data_sha256"],
        random_seeds="recorded data_seed for regeneration; no new filter draws",
        plan_file=plan, result_file=str(campaign / "result.md"))
    write_json(target, manifest)
    started = time.monotonic()
    try:
        with (attempt / "diagnostics.log").open("w") as log:
            code = run_bounded_worker(command, cwd=repo,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                log=log, timeout=timeout)
        manifest.update(status="complete" if code == 0 else "failed", exit_code=code)
    except subprocess.TimeoutExpired:
        manifest.update(status="timeout", exit_code=124)
    except Exception as error:
        manifest.update(status="failed", exit_code=1, error=repr(error))
        raise
    finally:
        manifest["diagnostic_sha256"] = {str(p.relative_to(attempt)): digest(p)
            for p in (attempt / "results/diagnostics").glob("*.csv")}
        manifest["input_result_sha256"] = {str(p.relative_to(attempt)): digest(p)
            for p in (attempt / "results").glob("*.csv")}
        manifest["input_saved_guide_sha256"] = {str(p.relative_to(attempt)): digest(p)
            for p in (attempt / "results").glob("iapf-*.rds")}
        manifest["orchestrator_sha256"] = digest(Path(__file__))
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(target, manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "exit_code", "wall_seconds")}))
    return manifest["exit_code"]


def summarize(campaign, repo):
    prior = repo / PRIOR
    prior_run = json.loads((prior / "manifest.json").read_text())
    assert prior_run["status"] == "complete" and prior_run["exit_code"] == 0
    assert prior_run["source_sha256"][CORE] == CORE_HASH
    assert digest(prior / "results/observations.csv") == prior_run["data_sha256"]
    prior_report = json.loads((prior / "inspection.json").read_text())
    seeds = {int(r["seed"]) for r in read(prior / "results/replicates.csv")}
    reports = []
    for name in SCHEDULE:
        attempt = campaign / name
        run = json.loads((attempt / "manifest.json").read_text())
        if run["status"] != "complete":
            reports.append(dict(attempt=name, screen_passed=False,
                hard_validity_vetoes=["incomplete_replication_set"], worker_status=run["status"]))
            continue
        same_data = name == "attempt01-same-data"
        assert (run["data_sha256"] == prior_run["data_sha256"]) == same_data
        own_seeds = {int(r["seed"]) for r in read(attempt / "results/replicates.csv")}
        assert len(own_seeds) == 128 and seeds.isdisjoint(own_seeds)
        seeds.update(own_seeds)
        report = inspect_attempt(name, campaign=campaign, schedule=SCHEDULE)
        lo, hi = report["mean_ratio_bootstrap95"]
        report.update(interval_contains_one=lo <= 1 <= hi,
            role="same_data_independent_randomness" if same_data else "fresh_data_confirmation")
        write_json(attempt / "inspection.json", report)
        reports.append(report)
    runs = [json.loads(p.read_text()) for p in campaign.glob("attempt*/manifest.json")]
    assert all(r["status"] != "running" for r in runs)
    worker = sum(r["wall_seconds"] for r in runs)
    assert worker <= 1400 and len(runs) <= 3
    return dict(schema="independent_R_d40_confirmation_v1", datasets=reports,
        prior_selected_dataset=dict(attempt=str(prior), data_seed=80000040,
            mean_ratio_bootstrap95=prior_report["mean_ratio_bootstrap95"],
            evidence_role="motivating_previous_result_not_pooled_with_confirmation"),
        screen_passed=all(r["screen_passed"] for r in reports), worker_seconds=worker,
        budget_seconds=1500, equation_15_objective=False, default_changed=False,
        default_readiness=False, statistically_supported_ranking=False, plan=PLAN,
        interval_includes_one_role="explanatory_only; neither unbiasedness proof nor promotion veto")


def finalize(campaign, repo):
    target = campaign / "report-manifest.json"
    assert not target.exists()
    used = post_used(campaign)
    assert used < 90, "reserve ten seconds for reporting"
    first = json.loads((campaign / "attempt01-same-data/manifest.json").read_text())
    manifest = dict(status="running", command=sys.argv, started_at=datetime.now(timezone.utc).isoformat(),
        timeout_seconds=100-used, git_commit=first["git_commit"], environment=sys.executable,
        cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
        source_sha256=first["source_sha256"], orchestrator_sha256=digest(Path(__file__)),
        random_seeds="bootstrap=data_seed+900, 2000 resamples", plan_file=PLAN,
        data_version="recorded per-attempt observations and reference results",
        result_file=str(campaign / "result.md"), role="post_run_independent_reference_reporting")
    write_json(target, manifest)
    started = time.monotonic()
    try:
        summary = summarize(campaign, repo)
        manifest.update(status="complete", exit_code=0)
    except Exception as error:
        manifest.update(status="failed", exit_code=1, error=repr(error))
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic()-started
        write_json(target, manifest)
    summary["post_run_seconds"] = post_used(campaign)
    summary["remaining_seconds"] = 1500-summary["worker_seconds"]-summary["post_run_seconds"]
    assert summary["post_run_seconds"] <= 100
    write_json(campaign / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                     ("screen_passed", "worker_seconds", "post_run_seconds", "remaining_seconds")}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnose", choices=tuple(SCHEDULE))
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    campaign, repo = args.campaign_root.resolve(), args.repo_root.resolve()
    if args.diagnose:
        return diagnose(args.diagnose, campaign, repo)
    return finalize(campaign, repo)


if __name__ == "__main__":
    raise SystemExit(main())
