"""Bounded independent R reconstruction comparison and diagnostic reporting."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics as stats
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/iapf-r-plausible-reconstruction-2026-09-21.md"
OUT = ROOT / "docs/plans/artifacts/iapf-r-plausible-reconstruction-20260921-01"
WORKER = "docs/benchmarks/compare_iapf_r_plausible_choices.R"
BUDGET = 868.02274921973
PAPER = {5: (.09, 1000, 6.93), 10: (.14, 1000, 15.11),
         20: (.19, 1000, 27.61), 40: (.23, 1033, 42.41), 80: (.35, 1142, 71.88)}
METHODS = ("iapf", "bpf", "fully_adapted", "sis")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def interval(values):
    values = sorted(values)
    def quantile(q):
        pos = q * (len(values) - 1)
        lo = int(pos)
        return values[lo] + (pos-lo) * (values[min(lo+1, len(values)-1)]-values[lo])
    return [quantile(.025), quantile(.975)]


def bootstrap(values, statistic, seed=9102026):
    rng = random.Random(seed)
    n = len(values)
    return interval([statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(4000)])


def inspect_attempt(attempt):
    result = OUT / attempt["name"] / "results"
    rows = read_csv(result / "replicates.csv")
    good = [r for r in rows if r["status"] == "complete"]
    expected_methods = METHODS if attempt["baselines"] else ("iapf",)
    expected_ids = set(range(attempt["first"], attempt["first"]+attempt["repeats"]))
    identity_ok = all({int(r["replication"]) for r in good if r["method"] == m} == expected_ids
                      for m in expected_methods)
    identity_ok = identity_ok and len(good) == attempt["repeats"] * len(expected_methods)
    identity_ok = identity_ok and all(
        int(r["seed"]) == 53000000+attempt["dimension"]*100000+int(r["replication"])*10+
        METHODS.index(r["method"])+1 for r in good)
    tails = read_csv(result / "tails.csv")
    fits = read_csv(result / "fits.csv")
    tail_ok = bool(tails) and all(r["passed"] == "TRUE" for r in tails)
    fit_ok = bool(fits) and all(int(r["convergence"]) == 0 and r["boundary"] == "FALSE" for r in fits)
    complete_iapf = sum(r["method"] == "iapf" for r in good)
    tail_ok = tail_ok and len(tails) == 100 * complete_iapf
    fit_ok = fit_ok and len(fits) == sum((int(r["iterations"])-1)*100 for r in good if r["method"] == "iapf")
    prefixes = read_csv(result / "prefixes.csv")
    identity_ok = identity_ok and len(prefixes) == 100*len(good)
    identity_ok = identity_ok and len({(r["replication"],r["method"],r["time"])
                                       for r in prefixes}) == len(prefixes)
    finite_ok = all(math.isfinite(float(r[k])) for r in good
                    for k in ("ratio", "log_ratio", "particles", "resampling_count"))
    summary = {"attempt": attempt["name"], "arm": attempt["arm"], "stage": attempt["stage"],
               "dimension": attempt["dimension"], "data_seed": attempt["data_seed"],
               "repeats_requested": attempt["repeats"], "worker_seconds": attempt["worker_seconds"],
               "complete": attempt["exit_code"] == 0 and identity_ok and finite_ok,
               "fit_passed": fit_ok, "tail_passed": tail_ok,
               "fits_checked": len(fits), "tails_checked": len(tails),
               "minimum_tail_margin": min((float(r["relative_margin"]) for r in tails), default=None),
               "errors": [r["error"] for r in rows if r["status"] != "complete"], "methods": {}}
    for method in expected_methods:
        selected = [r for r in good if r["method"] == method]
        if not selected or not finite_ok:
            continue
        ratios = [float(r["ratio"]) for r in selected]
        summary["methods"][method] = {
            "repeats": len(selected), "mean_ratio": stats.mean(ratios),
            "sd_ratio": stats.stdev(ratios) if len(ratios)>1 else None,
            "mean_particles": stats.mean(float(r["particles"]) for r in selected),
            "mean_resampling_count": stats.mean(float(r["resampling_count"]) for r in selected),
            "mean_wall_seconds": stats.mean(float(r["wall_seconds"]) for r in selected),
            "max_floor_probability": max(float(r["floor_probability_max"]) for r in selected)}
    summary["eligible"] = summary["complete"] and fit_ok and tail_ok
    return summary


def validation_diagnostics(attempt, summary):
    if not summary["eligible"]:
        return
    result = OUT / attempt["name"] / "results"
    rows = read_csv(result / "replicates.csv")
    values = {m: [float(r["ratio"]) for r in rows if r["method"] == m] for m in METHODS}
    for method in METHODS:
        summary["methods"][method]["mean_ci95"] = bootstrap(values[method], stats.mean)
        summary["methods"][method]["sd_ci95"] = bootstrap(values[method], stats.stdev)
    method = summary["methods"]["iapf"]
    sd, particles, resampling = PAPER[attempt["dimension"]]
    mean_ci, sd_ci = method["mean_ci95"], method["sd_ci95"]
    summary["paper_targets"] = {"sd_ratio": sd, "mean_particles": particles,
                                 "mean_resampling_count": resampling}
    summary["practical_primary_passed"] = (mean_ci[0]>=.8 and mean_ci[1]<=1.2 and
        sd_ci[1]<=2*sd and method["mean_particles"]<=1.5*particles)
    summary["stricter_prior_mean_screen_passed"] = mean_ci[0]>=.9 and mean_ci[1]<=1.1
    summary["literal_variability_and_resampling_agreement"] = (
        sd_ci[0]>=.5*sd and sd_ci[1]<=2*sd and
        .5*resampling<=method["mean_resampling_count"]<=2*resampling)
    # Preserve the replicate as the resampling unit; times are correlated.
    prefixes = read_csv(result / "prefixes.csv")
    buckets = {}
    for row in prefixes:
        key = (row["method"], row["situation"], int(row["replication"]))
        err = math.expm1(float(row["log_prefix_error"]))
        buckets.setdefault(key, []).append(err*err)
    heuristic = []
    ids = range(attempt["first"], attempt["first"]+attempt["repeats"])
    for situation in ("ordinary", "large_innovation"):
        scores = {m: [stats.mean(buckets[(m,situation,i)]) for i in ids] for m in METHODS}
        for comparator in METHODS[1:]:
            diffs = [a-b for a,b in zip(scores["iapf"],scores[comparator])]
            ci = bootstrap(diffs, stats.mean)
            heuristic.append({"situation": situation, "comparator": comparator,
                "iapf_mse": stats.mean(scores["iapf"]), "comparator_mse": stats.mean(scores[comparator]),
                "paired_mse_difference_ci95": ci,
                "observed_underperformance": stats.mean(diffs)>0,
                "lower_mse_statistically_supported": ci[1]<0})
    summary["heuristic_comparisons"] = heuristic
    summary["heuristic_dominance_passed"] = not any(r["observed_underperformance"] for r in heuristic)
    summary["practical_reference_passed"] = summary["practical_primary_passed"] and summary["heuristic_dominance_passed"]
    summary["terminal_variance_difference_ci95"] = {}
    for comparator in METHODS[1:]:
        pairs = list(zip(values["iapf"],values[comparator]))
        summary["terminal_variance_difference_ci95"][comparator] = bootstrap(pairs,
            lambda sample: stats.variance(x[0] for x in sample)-stats.variance(x[1] for x in sample))


def main():
    if OUT.exists():
        raise RuntimeError("Preserve existing campaign; use a reviewed fresh output root for a retry")
    OUT.mkdir(parents=True)
    sources = [PLAN, WORKER, "docs/benchmarks/reference_iapf_paper.R",
               "docs/benchmarks/reference_iapf_plausible_choices.R",
               "docs/benchmarks/diagnose_iapf_r_validation_tails.R",
               "docs/benchmarks/run_iapf_r_replication.py",
               "docs/benchmarks/run_iapf_r_plausible_choices.py",
               "tests/reference_iapf_plausible_choices.R"]
    hashes = snapshot_sources(ROOT, sources, OUT / "source-snapshot")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    versions = subprocess.check_output(["Rscript", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    ledger = {"schema": "iapf_plausible_reconstruction_v1", "plan": PLAN,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cpu_only": True, "gpu_intentionally_hidden": True, "environment": versions,
        "thread_environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "source_hashes": hashes, "budget_seconds": BUDGET, "attempts": [], "report_seconds": 0.0,
        "paper_sha256": hashlib.sha256((ROOT / ".localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf").read_bytes()).hexdigest(),
        "result_file": str((OUT / "result.md").relative_to(ROOT)),
        "original_author_data": False, "original_author_numerical_choices": False}
    for src in ("/tmp/iapf-plausible-focused-r.log", "/tmp/iapf-plausible-focused-pytest.log"):
        if Path(src).exists():
            shutil.copy2(src, OUT / Path(src).name)

    def remaining():
        return BUDGET-sum(a["worker_seconds"] for a in ledger["attempts"])-ledger["report_seconds"]

    def checkpoint(next_action):
        ledger["remaining_seconds"] = remaining()
        write_json(OUT / "manifest.json", ledger)
        (OUT / "checkpoint.md").write_text(
            "# Active plausible iAPF reconstruction\n\n"
            f"Question: useful performance reproduction under disclosed guesses. Plan: {PLAN}.\n\n"
            f"Completed attempts: {len(ledger['attempts'])}; remaining summed worker/report seconds: {remaining():.6f}.\n\n"
            f"Next action: {next_action}. No phase approval needed.\n")

    def launch(stage, arm, dimension, repeats, first, data_seed, timeout, baselines=False):
        if remaining()<=25:
            return None
        name = f"attempt{len(ledger['attempts'])+1:02d}-{stage}-{arm}-d{dimension}"
        path = OUT / name
        path.mkdir()
        command = r_worker_command(OUT / "source-snapshot", WORKER, path / "results",
            arm, dimension, repeats, first, data_seed, "yes" if baselines else "no")
        attempt = {"name": name, "stage": stage, "arm": arm, "dimension": dimension,
            "repeats": repeats, "first": first, "data_seed": data_seed, "baselines": baselines,
            "command": command, "timeout_seconds": min(timeout, remaining()-25),
            "started_utc": datetime.now(timezone.utc).isoformat()}
        checkpoint(f"running {name}")
        write_json(path / "launch.json", attempt)
        print(f"BEGIN {name}: remaining={remaining():.3f}, cap={attempt['timeout_seconds']:.3f}", flush=True)
        started = time.monotonic()
        with (path / "run.log").open("w") as log:
            try:
                attempt["exit_code"] = run_bounded_worker(command, cwd=ROOT, env=env,
                    log=log, timeout=attempt["timeout_seconds"])
                attempt["status"] = "complete" if attempt["exit_code"] == 0 else "candidate_failure"
            except subprocess.TimeoutExpired:
                attempt.update(exit_code=None, status="timeout_underbudgeted")
        attempt["worker_seconds"] = time.monotonic()-started
        attempt["output_hashes"] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file() and p.name != "launch.json"}
        ledger["attempts"].append(attempt)
        started = time.monotonic()
        summary = inspect_attempt(attempt)
        ledger["report_seconds"] += time.monotonic()-started
        write_json(path / "summary.json", summary)
        write_json(path / "manifest.json", attempt)
        checkpoint("inspect candidate and execute the next predeclared cell")
        print(f"END {name}: {attempt['status']}, {attempt['worker_seconds']:.3f}s, "
              f"eligible={summary['eligible']}, iapf={summary['methods'].get('iapf')}", flush=True)
        return attempt, summary

    calibration = {}
    for arm in ("current", "delayed", "floor4", "local_eq15"):
        result = launch("calibration", arm, 10, 8, 1101, 85000010, 45)
        calibration[arm] = result[1] if result else None
    highdim = {}
    for arm in ("delayed", "floor4", "local_eq15"):
        if calibration[arm] and calibration[arm]["eligible"]:
            result = launch("calibration", arm, 80, 2, 1201, 85000080, 75)
            highdim[arm] = result[1] if result else None
    eligible = [arm for arm in highdim if highdim[arm] and highdim[arm]["eligible"]]
    scores = {}
    for arm in eligible:
        m = calibration[arm]["methods"]["iapf"]
        scores[arm] = abs(math.log(m["sd_ratio"]/.14))+abs(math.log(m["mean_particles"]/1000)) + \
            .5*abs(math.log(max(m["mean_resampling_count"], 1e-300)/15.11))
    selected = "local_eq15" if "local_eq15" in eligible else (min(scores, key=scores.get) if scores else None)
    ledger["selection"] = {"selected_arm": selected, "calibration_scores": scores,
        "rule": "eligible local Eq15 first, otherwise predeclared d10 paper-pattern discrepancy",
        "frozen_before_validation_utc": datetime.now(timezone.utc).isoformat()}
    write_json(OUT / "frozen-selection.json", ledger["selection"])
    checkpoint(f"validate frozen {selected} on new data")
    validation = []
    if selected:
        for dimension, repeats, first in ((10,32,1301),(80,16,1401),(5,8,1501),(20,8,1501),(40,8,1501)):
            # Do not knowingly launch an incomplete optional cell. Estimates
            # conservatively include all three baselines and serialization.
            minimum = {5: 18, 10: 75, 20: 35, 40: 85, 80: 290}[dimension]
            if remaining()-25 < minimum:
                ledger.setdefault("unrun_cells", []).append({"dimension": dimension,
                    "reason": "insufficient reserved budget", "repeats": repeats})
                continue
            result = launch("validation", selected, dimension, repeats, first,
                86000000+dimension, remaining()-25, True)
            if result:
                validation.append(result)
    checkpoint("assemble heldout uncertainty, heuristic comparisons and result")
    report_started = time.monotonic()
    for attempt, summary in validation:
        validation_diagnostics(attempt, summary)
        write_json(OUT / attempt["name"] / "summary.json", summary)
    summary = {"schema": "iapf_plausible_reconstruction_summary_v1", "plan": PLAN,
        "calibration": calibration, "high_dimension_calibration": highdim,
        "selection": ledger["selection"], "validation": [s for _,s in validation],
        "budget_seconds": BUDGET, "worker_seconds": sum(a["worker_seconds"] for a in ledger["attempts"]),
        "original_author_replication": False, "default_changed": False,
        "publication_scale_replication": False}
    ledger["report_seconds"] += time.monotonic()-report_started
    summary.update(report_seconds=ledger["report_seconds"], remaining_seconds=remaining())
    write_json(OUT / "summary.json", summary)
    checkpoint("terminal review of saved results; no filter worker remains running")
    print(f"COMPLETE selected={selected}, remaining={remaining():.6f}s", flush=True)


if __name__ == "__main__":
    main()
