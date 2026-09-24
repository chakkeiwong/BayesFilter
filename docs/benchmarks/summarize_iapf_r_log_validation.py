"""Post-run diagnostics only: frozen independent R log-fit validation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from summarize_iapf_r_log_fit import inspect, read, digest
from summarize_iapf_r_replication import bootstrap_indices, bootstrap_mean

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01"
PLAN = "docs/plans/iapf-r-log-fit-validation-2026-09-20.md"
CORE = "docs/benchmarks/reference_iapf_paper.R"
RUNNER = "docs/benchmarks/replicate_iapf_paper_linear.R"
PREVIOUS = ROOT / ("docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01/"
                   "attempt12-replication-d20-log-quadratic/manifest.json")


def inspect_floor(path, manifest, result):
    """Validate executed data, complete row sets and the declared diagnostic role."""
    for source, expected in manifest["source_sha256"].items():
        assert digest(path / "sources" / source) == expected, source
    for name, expected in manifest["observation_file_sha256"].items():
        assert digest(path / "results" / name) == expected, name
    result.update(diagnostic_only=True, advance=True, hard_veto=None)
    if manifest["mode"] == "floor_diagnostic":
        previous = json.loads((CAMPAIGN / "attempt06-d80-diagnostic/manifest.json").read_text())
        assert manifest["data_sha256"] == previous["data_sha256"]
        rows = read(path / "results/summary.csv")
        expected = {(call, arm) for call in (15, 19, 20)
                    for arm in ("original", "zero_at_93", "zero_all")}
        assert len(rows) == len(expected)
        assert {(int(r["call"]), r["arm"]) for r in rows} == expected
        assert all(r["baseline_parity"] == "TRUE" for r in rows if r["arm"] == "original")
        for r in rows:
            assert all(math.isfinite(float(r[k])) for k in
                       ("log_error", "min_ess", "ess93", "floor_probability93"))
            assert 1 - 1e-10 <= float(r["min_ess"]) <= int(r["N"])
        proposals = read(path / "results/proposal.csv")
        expected_proposals = {(call, arm, t) for call, arm in expected for t in range(1, 101)}
        assert len(proposals) == len(expected_proposals)
        assert {(int(r["call"]), r["arm"], int(r["time"])) for r in proposals} == expected_proposals
        for r in proposals:
            assert all(math.isfinite(float(r[k])) for k in
                       ("ess", "increment_ess", "log_prefix_error", "floor_weight_mass",
                        "overlap_log_min", "overlap_log_mean", "overlap_log_max"))
            assert 0 <= float(r["floor_p_min"]) <= float(r["floor_p_mean"]) <= float(r["floor_p_max"]) <= 1
            assert 0 <= int(r["floor_draws"]) <= int(r["N"])
        curve = read(path / "results/floor-curve.csv")
        assert len(curve) == 1500
        assert {(int(r["call"]), int(r["time"]), int(r["power"])) for r in curve} == {
            (call, t, power) for call in (15, 19, 20) for t in range(1, 101) for power in (2, 3, 4, 8, 16)}
        assert all(0 <= float(r["mean_probability"]) <= float(r["max_probability"]) <= 1 for r in curve)
        result.update(inference="exact_replay_and_floor_intervention_only", floor_interventions=rows,
            floor_curve_maxima={str(power): max(float(r["max_probability"]) for r in curve
                                              if int(r["power"]) == power) for power in (2, 3, 4, 8, 16)},
            diagnostic_screen="three exact replays; floor intervention recorded")
        return result
    phases = read(path / "results/phases.csv")
    assert [(r["phase"], int(r["dimension"]), int(r["data_seed"]), int(r["first"]), int(r["repeats"]))
            for r in phases] == [("d20-a", 20, 70000020, 201, 4), ("d20-b", 20, 71000020, 301, 4),
                                 ("d80-pilot", 80, 72000080, 1, 2), ("d80-fresh", 80, 74000080, 101, 4)]
    assert all(r["status"] == "complete" for r in phases)
    checks = read(path / "results/checks.csv")
    assert len(checks) == 7 and all(r["passed"] == "TRUE" for r in checks)
    margins = read(path / "results/tail-margins.csv")
    assert len(margins) == 1700
    assert len({(r["case"], int(r["time"])) for r in margins}) == len(margins)
    assert all(math.isfinite(float(r["relative_margin"])) and
               float(r["relative_margin"]) > float(r["threshold"]) for r in margins)
    regression = read(path / "results/regressions.csv")
    assert len(regression) == 4 and all(r["passed"] == "TRUE" for r in regression)
    summaries = []
    for phase in phases:
        p = path / "results" / phase["phase"]
        settings = (p / "settings.R").read_text()
        assert "floor_tail_power = 8" in settings and 'fit_mode = "log_quadratic"' in settings
        assert 'doubling_mode = "first_full_window"' in settings and "equation_15_objective = FALSE" in settings
        count, first, d = (int(phase[k]) for k in ("repeats", "first", "dimension"))
        ids = range(first, first+count)
        rows, prefixes, fits = (read(p / name) for name in ("replicates.csv", "prefixes.csv", "fits.csv"))
        methods = {"iapf": 1, "bpf": 2, "fully_adapted": 3, "sis": 4}
        keys = {(int(r["replication"]), r["method"]) for r in rows}
        assert len(rows) == len(keys) == count*4
        assert keys == {(i, m) for i in ids for m in methods}
        assert all(r["status"] == "complete" and math.isfinite(float(r["log_likelihood"]))
                   and int(r["dimension"]) == d
                   and int(r["seed"]) == 53000000+d*100000+int(r["replication"])*10+methods[r["method"]]
                   for r in rows)
        expected_prefixes = {(i, m, t) for i, m in keys for t in range(1, 101)}
        assert len(prefixes) == len(expected_prefixes)
        assert {(int(r["replication"]), r["method"], int(r["time"])) for r in prefixes} == expected_prefixes
        kalman = read(p / "kalman.csv")
        situations = {int(r["time"]): r["situation"] for r in kalman}
        assert set(situations) == set(range(1, 101))
        assert all(r["situation"] == situations[int(r["time"])] and
                   math.isfinite(float(r["log_prefix_error"])) for r in prefixes)
        expected_fits = {(int(r["replication"]), iteration, t) for r in rows if r["method"] == "iapf"
                         for iteration in range(int(r["iterations"])-1) for t in range(1, 101)}
        assert len(fits) == len(expected_fits)
        assert {(int(r["replication"]), int(r["iteration"]), int(r["time"])) for r in fits} == expected_fits
        assert all(r["boundary"] == "FALSE" and int(r["convergence"]) == 0 and
                   int(r["design_rank"]) == int(r["design_columns"]) == 2*d+1 for r in fits)
        assert all(math.isfinite(float(r[k])) for r in fits for k in
                   ("gradient_max", "loss", "design_condition", "log_variance_min", "log_variance_max", "log_floor"))
        conditional = {}
        for situation in set(situations.values()):
            values = {m: [float(r["log_prefix_error"])**2 for r in prefixes
                          if r["method"] == m and r["situation"] == situation] for m in methods}
            conditional[situation] = {m: sum(v)/len(v) for m, v in values.items()}
            assert conditional[situation]["iapf"] <= min(conditional[situation][m] for m in methods if m != "iapf")
        ratios = [float(r["ratio"]) for r in rows if r["method"] == "iapf"]
        assert all(.1 <= v <= 10 for v in ratios)
        indices = bootstrap_indices(count, int(phase["data_seed"])+900)
        interval = bootstrap_mean(ratios, indices)
        summaries.append(dict(phase=phase["phase"], dimension=d, count=count,
            data_seed=int(phase["data_seed"]), mean_ratio=sum(ratios)/count,
            mean_ratio_bootstrap95=interval, ratios=ratios, fits=len(fits),
            iterations=[int(r["iterations"]) for r in rows if r["method"] == "iapf"],
            particles=[int(r["particles"]) for r in rows if r["method"] == "iapf"],
            conditional_log_prefix_mse=conditional))
    result.update(inference="positive_floor_feasibility_only_no_accuracy_or_ranking_claim",
        diagnostic_screen="healthy parity, tail margins, d20 non-harm and d80 feasibility pass",
        repair_phases=summaries, regression_checks=regression,
        minimum_tail_relative_margin=min(float(r["relative_margin"]) for r in margins),
        floor_tail_power=8, heuristic_dominance_verdict="no_conditional_veto")
    return result


def inspect_attempt(name):
    path = CAMPAIGN / name
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] != "running", "wait for terminal worker status"
    assert manifest["source_sha256"][CORE] == json.loads(PREVIOUS.read_text())["source_sha256"][CORE]
    assert manifest["campaign"] == "validation" and manifest["cpu_only"] is True
    result = dict(attempt=name, dimension=manifest["dimension"], mode=manifest["mode"],
                  data_seed=manifest["data_seed"], count=manifest["repeats"],
                  worker_seconds=manifest["wall_seconds"], worker_status=manifest["status"],
                  source_sha256=manifest["source_sha256"], data_sha256=manifest.get("data_sha256"))
    if manifest["status"] != "complete":
        result.update(advance=False, hard_veto="incomplete_worker", inference="failed_attempt_no_subset_inference")
        if manifest["mode"] == "floor_repair" and "setting stdout = TRUE" in (path / "run.log").read_text():
            result.update(infrastructure_failure=True, hard_veto="invalid_harness_exit_status",
                          inference="harness_failure_no_candidate_inference")
        return result
    if manifest["mode"] in ("floor_diagnostic", "floor_repair"):
        return inspect_floor(path, manifest, result)
    if manifest["mode"] == "controller_diagnostic":
        for source, expected in manifest["source_sha256"].items():
            assert digest(path / "sources" / source) == expected, source
        assert digest(path / "results/observations.csv") == manifest["data_sha256"]
        previous = json.loads((CAMPAIGN / "attempt04-d80-pilot/manifest.json").read_text())
        assert manifest["data_sha256"] == previous["data_sha256"]
        checks = read(path / "results/replay-checks.csv")
        assert {row["check"] for row in checks} == {
            "same_status", "identical_history", "identical_counts", "identical_fits", "exact_twist_Kalman"}
        assert all(row["passed"] == "TRUE" for row in checks)
        diagnostic = read(path / "results/diagnostic-summary.csv")
        assert len(diagnostic) == 1
        assert abs(float(diagnostic[0]["oracle_log_error"])) <= 1e-7
        result.update(diagnostic_only=True, diagnostic=diagnostic[0], advance=True,
                      hard_veto=None, inference="exact_replay_only_no_final_estimate")
        return result
    count, first, dimension = manifest["repeats"], manifest["first_replication"], manifest["dimension"]
    summary, rows, _ = inspect(name, count, first, manifest["data_seed"], campaign=CAMPAIGN)
    assert all(int(row["dimension"]) == dimension for row in rows)
    fits = read(path / "results/fits.csv")
    keys = [(int(row["replication"]), int(row["iteration"]), int(row["time"])) for row in fits]
    expected = {(int(row["replication"]), iteration, t) for row in rows if row["method"] == "iapf"
                for iteration in range(int(row["iterations"])-1) for t in range(1, 101)}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert all(int(row["design_columns"]) == 2*dimension+1 for row in fits)
    for row in fits:
        assert all(math.isfinite(float(row[key])) for key in (
            "gradient_max", "design_condition", "relative_residual", "loss", "log_variance_min", "log_variance_max"))
    values = [float(row["ratio"]) for row in rows if row["method"] == "iapf"]
    assert all(math.isfinite(value) and value > 0 for value in values)
    ci = bootstrap_mean(values, bootstrap_indices(count, manifest["data_seed"]+900))
    screen = (.9 <= ci[0] <= ci[1] <= 1.1) if dimension == 20 and count == 32 else all(.1 <= v <= 10 for v in values)
    result.update(summary)
    result.update(mean_ratio_bootstrap95=ci, accuracy_screen="pass" if screen else "fail",
        accuracy_screen_role="conditional_32_repeat_screen" if count == 32 else "diagnostic_range_only",
        hard_veto=None, advance=screen and not summary["heuristic_vetoes"],
        inference="conditional_accuracy_screen_only_no_method_ranking" if count == 32
                  else "pilot_or_short_follow_on_descriptive_only")
    return result


def write_report(final):
    reports = [json.loads(path.read_text()) for path in sorted(CAMPAIGN.glob("attempt*/inspection.json"))]
    manifests = [json.loads(path.read_text()) for path in sorted(CAMPAIGN.glob("attempt*/manifest.json"))]
    assert not final or all(row["status"] != "running" for row in manifests)
    if final:
        assert len(reports) == len(manifests), "inspect all terminal attempts before finalizing"
    completed = [row for row in reports if row["worker_status"] == "complete"
                 and not row.get("diagnostic_only")]
    assert len({row["source_sha256"][CORE] for row in reports}) <= 1
    assert len({row["source_sha256"][RUNNER] for row in reports}) <= 1
    charged = sum(row.get("wall_seconds", row["timeout_seconds"]) for row in manifests)
    summary = dict(schema="independent_R_log_fit_validation_v1", stage_complete=final,
        attempts=reports, launches_used=len(manifests), launch_budget=max(row["launch_budget"] for row in manifests),
        worker_seconds=charged, worker_budget_seconds=1800, remaining_worker_seconds=1800-charged,
        complete_repetitions=sum(row["count"] for row in completed),
        fits=sum(row["fits"] for row in completed),
        candidate_vetoes=[row["attempt"] for row in reports if not row["advance"] and not row.get("infrastructure_failure")],
        infrastructure_failures=[row["attempt"] for row in reports if row.get("infrastructure_failure")],
        statistically_supported_ranking=False, default_readiness=False,
        equation_15_objective=False, paper_replication=False, cpu_reference_only=True)
    (CAMPAIGN / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Frozen R log-fit validation", "",
        "Bounded stage " + ("complete." if final else "in progress."), "",
        "The numerical core is frozen from the previous log-fit repair. Each worker executes captured sources. "
        "The reporting-only append change passed real-run numerical parity. The fitting objective is "
        "unweighted squared log-target error, different from paper Eq.15. All runs are independent R CPU references.", "",
        "| Attempt | d / repeats | Data seed | Mean ratio to Kalman | Bootstrap95% | Screen | Heuristic veto |",
        "|---|---:|---:|---:|---|---|---|"]
    for row in reports:
        if row.get("diagnostic_only"):
            screen = row.get("diagnostic_screen", "exact replay and oracle pass")
            lines.append(f'| {row["attempt"]} | {row["dimension"]} / diagnostic | {row["data_seed"]} | see scoped report | N/A | {screen} | no default promotion |')
            continue
        if row["worker_status"] != "complete":
            lines.append(f'| {row["attempt"]} | {row["dimension"]} / {row["count"]} | {row["data_seed"]} | incomplete | N/A | veto | not interpreted |')
            continue
        lo, hi = row["mean_ratio_bootstrap95"]
        lines.append(f'| {row["attempt"]} | {row["dimension"]} / {row["count"]} | {row["data_seed"]} | '
            f'{row["methods"]["iapf"]["mean_ratio"]:.6f} | [{lo:.6f}, {hi:.6f}] | '
            f'{row["accuracy_screen"]} ({row["accuracy_screen_role"]}) | {row["heuristic_dominance_verdict"]} |')
    lines += ["", "Intervals resample independent filter repetitions conditional on each fixed dataset "
        "(2,000 bootstrap resamples, seed=data_seed+900). A d20 screen requires its entire interval inside "
        "[0.9,1.1]. Fewer than32 repetitions provide descriptive feasibility evidence only. "
        "These intervals do not quantify variation across datasets or support a method ranking.", "",
        "| Attempt / situation | iAPF log-prefix MSE | Fully adapted | BPF | SIS |", "|---|---:|---:|---:|---:|"]
    for row in completed:
        for situation, values in row["conditional_log_prefix_mse"].items():
            lines.append(f'| {row["attempt"]} / {situation} | {values["iapf"]:.6g} | '
                         f'{values["fully_adapted"]:.6g} | {values["bpf"]:.6g} | {values["sis"]:.6g} |')
    veto = bool(summary["candidate_vetoes"])
    lines += ["", "The conditional heuristic comparison is a veto screen, not a statistical superiority claim. "
        "SIS likelihood ratios may underflow to zero; their finite log errors remain visible and zero sample "
        "variance of underflowed ratios is not accuracy evidence.", "",
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|",
        f'| {"Investigate rejected scope" if veto else "Retain as a reference candidate"} | Dataset screens above | '
        f'{"See rejected attempts" if veto else "No observed completed-run veto"} | Dataset coverage and higher-dimensional replication | '
        'Complete longer high-dimensional validation before published-scale runs | Paper Eq.15 replication; TF/score/HMC readiness |',
        "", "| Inference status | Finding |", "|---|---|",
        f'| Hard veto screen | {summary["candidate_vetoes"] or "No completed-run validity veto"} |',
        "| Statistically supported ranking | None; fixed, unequal particle/computing budgets |",
        "| Descriptive-only differences | Method means, SD, runtime, extreme ratios and conditional MSE |",
        "| Default-readiness | Not established; optional independent reference |",
        "| Next evidence needed | More repetitions/datasets, author-setting resolution, equal-cost comparison if ranking is sought |", "",
        f'Worker budget: {charged:.6f}/1800 seconds, {len(manifests)} process launches '
        '(eight planned plus one bounded infrastructure retry); '
        f'{1800-charged:.6f} seconds remaining. Completed repetitions: {summary["complete_repetitions"]}; '
        f'QR fits: {summary["fits"]}. Mechanics and focused checks are separate; see checkpoint.', "",
        "Post-run red-team: easy datasets and linear-Gaussian structure may explain favorable evidence. "
        "A rejected fit or likelihood discrepancy on untouched data would overturn advancement in that scope. "
        "The weakest evidence is higher-dimensional replication and unknown author fitting/floor/controller settings. "
        "Engineering parity does not imply numerical accuracy; conditional Kalman agreement does not imply "
        "nonlinear, score, HMC, KDM or canonical LEDH validity.", ""]
    if any(r["mode"] == "floor_diagnostic" for r in reports):
        lines += ["Continuation: [floor diagnosis and positive-floor repair](floor-result.md) records "
            "the analytic cause, intervention results, safeguards, repair outcomes and remaining gaps. "
            "The original power2 d80 candidate remains rejected. A positive-floor repair is assessed "
            "separately; a logging/status infrastructure failure is not a candidate rejection.", ""]
    (CAMPAIGN / "result.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if args.attempt:
        assert Path(args.attempt).name == args.attempt and args.attempt.startswith("attempt")
        result = inspect_attempt(args.attempt)
        path = CAMPAIGN / args.attempt / "inspection.json"
        assert not path.exists(), "do not overwrite an existing inspection"
        path.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({key: result.get(key) for key in (
            "attempt", "dimension", "count", "mean_ratio_bootstrap95", "accuracy_screen", "advance", "hard_veto")}))
    summary = write_report(args.finalize)
    source_names = ["docs/benchmarks/summarize_iapf_r_log_validation.py",
                    "docs/benchmarks/summarize_iapf_r_log_fit.py",
                    "docs/benchmarks/summarize_iapf_r_replication.py", PLAN]
    for source in source_names:
        generation = digest(ROOT / "docs/benchmarks/summarize_iapf_r_log_validation.py")[:12]
        dest = CAMPAIGN / "report_sources" / generation / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert digest(dest) == digest(ROOT / source), source
        else:
            dest.write_bytes((ROOT / source).read_bytes())
    manifest = dict(command=sys.argv, git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        cpu_only=True, gpu_intentionally_hidden=True, role="post_run_diagnostics_only",
        data_version="captured_R_attempt_observations", random_seeds="bootstrap=data_seed+900",
        environment=sys.executable, plan_file=PLAN, result_file=str(CAMPAIGN / "result.md"),
        source_sha256={source: digest(ROOT / source) for source in source_names},
        source_snapshot_root=str(CAMPAIGN / "report_sources" / generation),
        wall_seconds=time.monotonic()-started, stage_complete=args.finalize)
    (CAMPAIGN / (f'report-{args.attempt}.json' if args.attempt else 'report-final.json')).write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: summary[key] for key in ("launches_used", "worker_seconds", "remaining_worker_seconds", "candidate_vetoes")}))


if __name__ == "__main__":
    main()
