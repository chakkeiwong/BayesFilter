"""Diagnostic reporting for the frozen optional R reference's missing dimensions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from summarize_iapf_r_d40_confirmation import diagnose, digest, post_used, write_json
from summarize_iapf_r_positive_floor_validation import inspect_attempt
from summarize_iapf_r_log_fit import read

PLAN = "docs/plans/iapf-r-small-dimension-completion-2026-09-21.md"
SCHEDULE = {"attempt01-d5": (5, 84000005, 1001),
            "attempt02-d10": (10, 84000010, 1001),
            "attempt03-d20": (20, 84000020, 1001)}
BUDGET = 1333.4925125938026
POST_BUDGET = BUDGET - 1200


def finalize(campaign):
    target = campaign / "report-manifest.json"
    assert not target.exists()
    used = post_used(campaign)
    assert used < POST_BUDGET, "post-run budget exhausted"
    first = json.loads((campaign / "attempt01-d5/manifest.json").read_text())
    manifest = dict(status="running", command=sys.argv,
        started_at=datetime.now(timezone.utc).isoformat(),
        timeout_seconds=POST_BUDGET-used, git_commit=first["git_commit"],
        environment=sys.executable, cpu_only=True, gpu_intentionally_hidden=True,
        jit_compile=False, source_sha256=first["source_sha256"],
        orchestrator_sha256=digest(Path(__file__)),
        random_seeds="bootstrap=data_seed+900, 2000 resamples", plan_file=PLAN,
        data_version="per-attempt observations and retained R results",
        result_file=str(campaign / "result.md"), role="post_run_reference_reporting")
    write_json(target, manifest)
    started = time.monotonic()
    try:
        reports, seeds, data_hashes = [], set(), set()
        for name in SCHEDULE:
            attempt = campaign / name
            run = json.loads((attempt / "manifest.json").read_text())
            assert run["plan_file"] == PLAN and run["campaign"] == "small_dimension_completion"
            assert run["status"] == "complete" and run["exit_code"] == 0
            own_seeds = {int(r["seed"]) for r in read(attempt / "results/replicates.csv")}
            assert len(own_seeds) == 128 and seeds.isdisjoint(own_seeds)
            assert run["data_sha256"] not in data_hashes
            seeds.update(own_seeds)
            data_hashes.add(run["data_sha256"])
            report = inspect_attempt(name, campaign=campaign, schedule=SCHEDULE)
            lo, hi = report["mean_ratio_bootstrap95"]
            report["interval_contains_one"] = lo <= 1 <= hi
            write_json(attempt / "inspection.json", report)
            reports.append(report)
        runs = [json.loads(p.read_text()) for p in campaign.glob("attempt*/manifest.json")]
        assert all(r["status"] != "running" for r in runs)
        worker = sum(r["wall_seconds"] for r in runs)
        assert worker <= 1200 and len(runs) <= 5
        summary = dict(schema="independent_R_small_dimension_completion_v1",
            datasets=reports, screen_passed=all(r["screen_passed"] for r in reports),
            worker_seconds=worker, budget_seconds=BUDGET, equation_15_objective=False,
            default_changed=False, default_readiness=False,
            statistically_supported_ranking=False, plan=PLAN,
            interval_includes_one_role="explanatory; not unbiasedness proof or promotion veto")
        manifest.update(status="complete", exit_code=0)
    except Exception as error:
        manifest.update(status="failed", exit_code=1, error=repr(error))
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic()-started
        write_json(target, manifest)
    summary["post_run_seconds"] = post_used(campaign)
    summary["remaining_seconds"] = BUDGET-worker-summary["post_run_seconds"]
    assert summary["post_run_seconds"] <= POST_BUDGET
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
        return diagnose(args.diagnose, campaign, repo, schedule=SCHEDULE, plan=PLAN,
                        post_budget=POST_BUDGET-10)
    return finalize(campaign)


if __name__ == "__main__":
    raise SystemExit(main())
