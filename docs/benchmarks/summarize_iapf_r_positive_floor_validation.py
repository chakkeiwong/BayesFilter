"""Post-run diagnostic reporting for the frozen optional R floor candidate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from summarize_iapf_r_log_fit import inspect, read, digest, METHODS
from summarize_iapf_r_replication import bootstrap_indices, bootstrap_mean

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/plans/artifacts/iapf-r-positive-floor-validation-20260920-01"
PLAN = "docs/plans/iapf-r-positive-floor-validation-2026-09-20.md"
SCHEDULE = {"attempt01-d80-a": (80, 80000080, 501),
            "attempt02-d80-b": (80, 81000080, 601),
            "attempt03-d40-control": (40, 80000040, 701)}
CORE = "docs/benchmarks/reference_iapf_paper.R"
CORE_HASH = "c7152fea96d917bf90218bbbbc2bb31c01b0831b439625ff922454b01d748fd0"


def inspect_attempt(name, campaign=CAMPAIGN, schedule=None):
    d, seed, first = (SCHEDULE if schedule is None else schedule)[name]
    path = campaign / name
    report, rows, manifest = inspect(name, 32, first, seed, campaign=campaign)
    assert manifest["dimension"] == d and manifest["first_replication"] == first
    assert manifest["repeats"] == 32 and manifest["floor_tail_power"] == 8
    assert manifest["source_sha256"][CORE] == CORE_HASH
    assert manifest["cpu_only"] and manifest["gpu_intentionally_hidden"]
    assert manifest["doubling_mode"] == "first_full_window"
    diagnostic = json.loads((path / "diagnostic-manifest.json").read_text())
    assert diagnostic["status"] == "complete" and diagnostic["exit_code"] == 0
    assert diagnostic["input_data_sha256"] == manifest["data_sha256"]
    assert diagnostic["source_sha256"] == manifest["source_sha256"]
    for field in ("diagnostic_sha256", "input_result_sha256"):
        for filename, expected in diagnostic[field].items():
            assert digest(path / filename) == expected, filename
    settings = {r["name"]: r["value"] for r in read(path / "results/diagnostics/settings.csv")}
    required = dict(dimension=str(d), horizon="100", N0="1000", BPF_N="10000",
        FA_N="5000", SIS_N="10000", k="5", tau="0.5", kappa="0.5",
        fit_maxit="200", floor_tail_power="8", fit_mode="log_quadratic",
        doubling_mode="first_full_window", equation_15_objective="FALSE",
        data_seed=str(seed), first_replication=str(first), repeats="32",
        cpu_only="TRUE", original_author_data="FALSE", noncanonical_reference="TRUE")
    assert all(settings[k] == v for k, v in required.items())
    for row in rows:
        expected_seed = 53000000+d*100000+int(row["replication"])*10+METHODS.index(row["method"])+1
        assert int(row["seed"]) == expected_seed and int(row["dimension"]) == d
        assert all(math.isfinite(float(row[k])) for k in ("log_likelihood", "ratio", "log_ratio"))
        assert math.isclose(float(row["ratio"]), math.exp(float(row["log_ratio"])), rel_tol=1e-11, abs_tol=1e-300)
    fits = read(path / "results/fits.csv")
    fitted = {int(r["replication"]): r for r in rows if r["method"] == "iapf"}
    expected_fits = {(rep, iteration, t) for rep, row in fitted.items()
                     for iteration in range(int(row["iterations"])-1) for t in range(1, 101)}
    assert len(fits) == len(expected_fits)
    assert {(int(r["replication"]), int(r["iteration"]), int(r["time"])) for r in fits} == expected_fits
    assert all(int(r["design_rank"]) == int(r["design_columns"]) == 2*d+1 for r in fits)
    assert all(math.isfinite(float(r[k])) for r in fits for k in
               ("gradient_max", "loss", "design_condition", "log_variance_min", "log_variance_max", "log_floor"))
    kalman = read(path / "results/kalman.csv")
    assert len(kalman) == 100 and {int(r["time"]) for r in kalman} == set(range(1,101))
    assert all(math.isfinite(float(r[k])) for r in kalman for k in ("prefix", "innovation"))
    exact_logz = next(float(r["prefix"]) for r in kalman if int(r["time"]) == 100)
    assert all(math.isclose(float(r["log_likelihood"])-exact_logz, float(r["log_ratio"]),
                            rel_tol=1e-10, abs_tol=2e-9) for r in rows)
    situations = {int(r["time"]): r["situation"] for r in kalman}
    assert set(situations.values()) == {"ordinary", "large_innovation"}
    prefixes = read(path / "results/prefixes.csv")
    assert all(r["situation"] == situations[int(r["time"])] for r in prefixes)
    endings = {(int(r["replication"]), r["method"]): float(r["log_prefix_error"])
               for r in prefixes if int(r["time"]) == 100}
    assert all(math.isclose(endings[int(r["replication"]), r["method"]], float(r["log_ratio"]),
                            rel_tol=1e-10, abs_tol=2e-9) for r in rows)
    tails = read(path / "results/diagnostics/tail-margins.csv")
    expected_tails = {(rep, t) for rep in fitted for t in range(1,101)}
    assert len(tails) == len(expected_tails)
    assert {(int(r["replication"]), int(r["time"])) for r in tails} == expected_tails
    for r in tails:
        assert all(math.isfinite(float(r[k])) for k in ("minimum_eigenvalue", "scale", "relative_margin", "threshold"))
        assert float(r["scale"]) > 0
        assert math.isclose(float(r["threshold"]), 100*d*sys.float_info.epsilon, rel_tol=1e-12)
        assert math.isclose(float(r["relative_margin"]), float(r["minimum_eigenvalue"])/float(r["scale"]), rel_tol=1e-12, abs_tol=1e-15)
        assert (r["passed"] == "TRUE") == (float(r["relative_margin"]) > float(r["threshold"]))
    guides = read(path / "results/diagnostics/final-guides.csv")
    assert len(guides) == 32 and {int(r["replication"]) for r in guides} == set(fitted)
    for row in guides:
        reference = fitted[int(row["replication"])]
        assert int(row["iterations"]) == int(reference["iterations"]) <= 20
        assert int(row["final_particles"]) == int(reference["particles"]) <= 16000
        assert math.isclose(float(row["log_likelihood"]), float(reference["log_likelihood"]), abs_tol=1e-9)
        assert 1-1e-9 <= float(row["min_ess"]) <= float(row["max_ess"]) <= int(row["final_particles"])+1e-8
        assert 0 <= float(row["floor_probability_max"]) <= 1
        own_tails = [r for r in tails if r["replication"] == row["replication"]]
        assert math.isclose(float(row["minimum_tail_margin"]),
                            min(float(r["relative_margin"]) for r in own_tails), rel_tol=1e-12)
        assert (row["tail_passed"] == "TRUE") == all(r["passed"] == "TRUE" for r in own_tails)
    values = [float(r["ratio"]) for r in fitted.values()]
    interval = bootstrap_mean(values, bootstrap_indices(32, seed+900))
    accuracy = .9 <= interval[0] <= interval[1] <= 1.1
    tail_passed = all(r["passed"] == "TRUE" for r in tails)
    report.update(dimension=d, count=32, mean_ratio_bootstrap95=interval,
        accuracy_screen="pass" if accuracy else "fail", tail_screen="pass" if tail_passed else "fail",
        tail_rows=len(tails), minimum_tail_margin=min(float(r["relative_margin"]) for r in tails),
        final_iterations=sorted({int(r["iterations"]) for r in guides}),
        final_particles=sorted({int(r["final_particles"]) for r in guides}),
        max_floor_probability=max(float(r["floor_probability_max"]) for r in guides),
        min_ess=min(float(r["min_ess"]) for r in guides),
        screen_passed=accuracy and tail_passed and not report["heuristic_vetoes"],
        hard_validity_vetoes=[] if tail_passed else ["Gaussian_limit_second_moment_margin"],
        statistically_supported_ranking=False, default_readiness=False)
    (path / "inspection.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=tuple(SCHEDULE))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    started = time.monotonic()
    if args.attempt:
        report = inspect_attempt(args.attempt, campaign=campaign)
        print(json.dumps({k: report[k] for k in ("attempt", "mean_ratio_bootstrap95", "accuracy_screen", "tail_screen", "screen_passed")}))
    if args.finalize:
        reports = []
        for name in SCHEDULE:
            path = campaign / name
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest["status"] == "complete":
                reports.append(inspect_attempt(name, campaign=campaign))
            else:
                reports.append(dict(attempt=name, dimension=SCHEDULE[name][0],
                    screen_passed=False, worker_status=manifest["status"],
                    hard_validity_vetoes=["incomplete_replication_set"],
                    failure_log=str(path / "run.log")))
        manifests = [json.loads(p.read_text()) for p in campaign.glob("attempt*/manifest.json")]
        assert all(m["status"] != "running" for m in manifests)
        core_hashes = {m["source_sha256"][CORE] for m in manifests}
        assert core_hashes == {CORE_HASH}
        assert len({m["data_sha256"] for m in manifests if "data_sha256" in m}) == len(SCHEDULE)
        diagnostic_manifests = [json.loads(p.read_text()) for p in campaign.glob("attempt*/diagnostic-manifest.json")]
        worker = sum(m["wall_seconds"] for m in manifests)
        post = sum(m["wall_seconds"] for m in diagnostic_manifests)
        summary = dict(schema="independent_r_positive_floor_validation_v1", datasets=reports,
            screen_passed=all(r["screen_passed"] for r in reports), worker_seconds=worker,
            post_run_seconds=post, budget_seconds=4000, remaining_seconds=4000-worker-post,
            equation_15_objective=False, default_changed=False, default_readiness=False,
            statistically_supported_ranking=False, plan=PLAN)
        (campaign / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps({k: summary[k] for k in ("screen_passed", "worker_seconds", "post_run_seconds", "remaining_seconds")}))
    elapsed = time.monotonic()-started
    print(json.dumps(dict(report_seconds=elapsed)))


if __name__ == "__main__":
    main()
