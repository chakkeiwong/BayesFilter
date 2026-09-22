"""Post-run diagnostics for the bounded R log-quadratic fit experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys
import time

from summarize_iapf_r_replication import bootstrap_indices, bootstrap_mean

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01"
OUTPUT = ROOT / "docs/plans/artifacts/iapf-r-log-fit-repair-20260920-01"
PLAN = "docs/plans/iapf-r-log-fit-repair-2026-09-20.md"
METHODS = ("iapf", "bpf", "fully_adapted", "sis")


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(name, count, first, data_seed, campaign=CAMPAIGN):
    path = campaign / name
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["exit_code"] == 0
    assert manifest["fit_mode"] == "log_quadratic" and manifest["data_seed"] == data_seed
    assert manifest["executed_captured_sources"] is True
    for source, expected in manifest["source_sha256"].items():
        assert digest(path / "sources" / source) == expected, source
    assert digest(path / "results/observations.csv") == manifest["data_sha256"]
    rows = read(path / "results/replicates.csv")
    expected = {(rep, method) for rep in range(first, first + count) for method in METHODS}
    keys = [(int(row["replication"]), row["method"]) for row in rows]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert all(row["status"] == "complete" and math.isfinite(float(row["log_ratio"])) for row in rows)
    fits = read(path / "results/fits.csv")
    assert fits and all(row["optimizer"] == "QR" and row["convergence"] == "0"
                        and row["design_rank"] == row["design_columns"]
                        and row["boundary"] == "FALSE" for row in fits)
    prefixes = read(path / "results/prefixes.csv")
    prefix_keys = [(int(row["replication"]), row["method"], int(row["time"])) for row in prefixes]
    assert len(prefix_keys) == len(set(prefix_keys))
    assert set(prefix_keys) == {(rep, method, t) for rep, method in expected for t in range(1, 101)}
    conditional = {}
    for situation in ("ordinary", "large_innovation"):
        conditional[situation] = {}
        for method in METHODS:
            per_rep = []
            for rep in range(first, first + count):
                errors = [float(row["log_prefix_error"]) for row in prefixes
                          if int(row["replication"]) == rep and row["method"] == method
                          and row["situation"] == situation]
                assert errors and all(math.isfinite(error) for error in errors)
                per_rep.append(stats.fmean(error * error for error in errors))
            conditional[situation][method] = stats.fmean(per_rep)
    method_summary = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        ratios = [float(row["ratio"]) for row in group]
        method_summary[method] = dict(repeats=count, mean_ratio=stats.fmean(ratios),
            sd_ratio=stats.stdev(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
            min_log_ratio=min(float(row["log_ratio"]) for row in group),
            max_log_ratio=max(float(row["log_ratio"]) for row in group),
            mean_particles=stats.fmean(float(row["particles"]) for row in group),
            mean_wall_seconds=stats.fmean(float(row["wall_seconds"]) for row in group))
    vetoes = [f"{situation}:{method}" for situation, values in conditional.items()
              for method in METHODS[1:] if values["iapf"] > values[method]]
    return dict(attempt=name, data_seed=data_seed, worker_seconds=manifest["wall_seconds"],
        methods=method_summary, conditional_log_prefix_mse=conditional,
        heuristic_dominance_verdict="veto" if vetoes else "no_observed_veto", heuristic_vetoes=vetoes,
        fits=len(fits), max_design_condition=max(float(row["design_condition"]) for row in fits),
        max_normal_equation_gradient=max(float(row["gradient_max"]) for row in fits),
        max_training_relative_residual=max(float(row["relative_residual"]) for row in fits)), rows, manifest


def main():
    started = time.monotonic()
    pilot, _, pilot_manifest = inspect("attempt11-pilot-d20-log-quadratic", 4, 1, 68000020)
    follow_on, rows, follow_manifest = inspect("attempt12-replication-d20-log-quadratic", 16, 101, 69000020)
    for source in ("docs/benchmarks/reference_iapf_paper.R", "docs/benchmarks/replicate_iapf_paper_linear.R"):
        assert pilot_manifest["source_sha256"][source] == follow_manifest["source_sha256"][source]
    values = [float(row["ratio"]) for row in rows if row["method"] == "iapf"]
    ci = bootstrap_mean(values, bootstrap_indices(len(values), 69000920))
    follow_on["mean_ratio_bootstrap95"] = ci
    follow_on["conditional_accuracy_screen"] = "pass" if .9 <= ci[0] <= ci[1] <= 1.1 else "fail"
    prior = [json.loads(path.read_text()) for path in CAMPAIGN.glob("attempt*/manifest.json")]
    assert all(row["status"] != "running" for row in prior)
    charged = sum(row["wall_seconds"] for row in prior)
    summary = dict(schema="independent_r_iapf_log_fit_diagnostic_v1", pilot=pilot, follow_on=follow_on,
        complete_repetitions=20, equation_15_objective=False, default_changed=False,
        hard_vetoes=[], statistically_supported_ranking=False, default_readiness=False,
        screen_status="pass" if follow_on["conditional_accuracy_screen"] == "pass"
        and not follow_on["heuristic_vetoes"] else "veto",
        repair_worker_seconds=charged, repair_budget_seconds=1550,
        remaining_worker_seconds=1550-charged, launches_used=len(prior), launch_budget=12,
        next_evidence="larger untouched d20 comparison, then d40/d80; resolve author settings for paper replication")
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = ["docs/benchmarks/summarize_iapf_r_log_fit.py",
               "docs/benchmarks/summarize_iapf_r_replication.py", PLAN]
    for source in sources:
        destination = OUTPUT / "sources" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source).read_bytes())
    manifest = dict(command=sys.argv, git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), environment=sys.executable,
        cpu_only=True, cuda_visible_devices="-1", role="post_run_standard_library_diagnostics",
        data_seeds=[68000020, 69000020], bootstrap_seed=69000920, bootstrap_repetitions=2000,
        wall_seconds=time.monotonic()-started, plan_file=PLAN, result_file=str(OUTPUT / "result.md"),
        source_sha256={source: digest(ROOT / source) for source in sources},
        input_sha256={str(path.relative_to(ROOT)): digest(path)
                      for attempt in (pilot["attempt"], follow_on["attempt"])
                      for path in (CAMPAIGN / attempt / "results").glob("*.csv")})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status=summary["screen_status"], mean_ratio=stats.fmean(values),
                         interval=ci, worker_seconds=charged, output=str(OUTPUT))))


if __name__ == "__main__":
    main()
