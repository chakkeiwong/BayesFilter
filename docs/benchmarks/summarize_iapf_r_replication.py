"""Post-run diagnostic reporting for the independent R iAPF reference.

Python standard library only. This is not runtime selection or admission code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plans/artifacts/iapf-r-paper-replication-20260920-01"
REFERENCE = "docs/benchmarks/reference_iapf_paper.R"
RUNNER = "docs/benchmarks/replicate_iapf_paper_linear.R"
METHODS = ("iapf", "bpf", "fully_adapted", "sis")
PAPER_SD = {5: (.09, .51, .10), 10: (.14, 6.4, .17),
            20: (.19, None, .53), 40: (.23, None, None), 80: (.35, None, None)}
BOOTSTRAPS = 2000


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (position - low) * (values[high] - values[low])


def interval(values):
    return [quantile(values, .025), quantile(values, .975)] if values else None


def bootstrap_indices(n, seed):
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAPS)]


def bootstrap_mean(values, indices):
    return interval([stats.fmean(values[i] for i in sample) for sample in indices])


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return f"[{value[0]:.4g}, {value[1]:.4g}]"
    return f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=("paper", "repair"), default="paper")
    args = parser.parse_args()
    out = OUT if args.campaign == "paper" else ROOT / "docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01"
    seconds, launch_limit = (1800, 8) if args.campaign == "paper" else (1550, 10)
    attempts, groups = [], {}
    for path in sorted(out.glob("attempt*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["status"] == "running":
            raise RuntimeError(f"worker still running: {path.parent.name}")
        for name, expected in manifest["source_sha256"].items():
            snapshot = path.parent / "sources" / name
            if not snapshot.exists():
                snapshot = path.parent / "sources" / Path(name).name
            if name == REFERENCE and hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
                snapshot = out / "source-recovery" / f"{expected}.R"
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"source snapshot changed: {snapshot}")
        result = path.parent / "results"
        for name, expected in manifest.get("observation_file_sha256", {}).items():
            if hashlib.sha256((result / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"observation file changed: {result / name}")
        if "data_sha256" in manifest and hashlib.sha256(
                (result / "observations.csv").read_bytes()).hexdigest() != manifest["data_sha256"]:
            raise RuntimeError(f"data changed: {result}")
        d, repetitions, first, data_seed = map(int, manifest["command"][5:9])
        mode = manifest["command"][9]
        rows = read_csv(result / "replicates.csv")
        entry = dict(attempt=path.parent.name, dimension=d, mode=mode,
                     status=manifest["status"], requested=repetitions,
                     completed_iapf=sum(r["method"] == "iapf" and r["status"] == "complete" for r in rows),
                     failures=[r for r in rows if r["status"] != "complete"],
                     wall_seconds=manifest["wall_seconds"])
        attempts.append(entry)
        if mode == "sensitivity":
            entry["sensitivity"] = read_csv(result / "sensitivity.csv")
        if mode == "oracle":
            checks = read_csv(result / "oracle-checks.csv")
            entry["oracle"] = dict(checks=len(checks),
                passed=sum(r["passed"] == "TRUE" for r in checks),
                dimensions=sorted({int(r["dimension"]) for r in checks}),
                max_absolute_log_error=max(abs(float(r["optimal_log_error"])) for r in checks),
                max_terminal_weight_spread=max(float(r["terminal_weight_spread"]) for r in checks),
                max_fully_adapted_difference=max(abs(float(r["fully_adapted_path_difference"])) for r in checks))
        if mode != "replication":
            continue
        key = (d, data_seed, manifest["data_sha256"],
               manifest["source_sha256"][REFERENCE], manifest["source_sha256"][RUNNER],
               manifest.get("fit_mode", "paper_eq15"), manifest.get("floor_tail_power", 2),
               manifest.get("doubling_mode", "first_full_window"), manifest.get("fit_maxit", 200))
        group = groups.setdefault(key, dict(rows={}, prefixes={}, fits={}, requested=set(), attempts=[]))
        group["attempts"].append(entry["attempt"])
        group["requested"].update(range(first, first + repetitions))
        for row in read_csv(result / "fits.csv"):
            fit_key = tuple(int(row[field]) for field in ("replication", "iteration", "time"))
            old = group["fits"].get(fit_key)
            if old and old != row:
                raise RuntimeError("same frozen seed produced different fitting diagnostics")
            group["fits"][fit_key] = row
        for row in rows:
            if row["status"] != "complete":
                continue
            replication, method = int(row["replication"]), row["method"]
            if int(row["seed"]) != 53000000 + d * 100000 + replication * 10 + METHODS.index(method) + 1:
                raise RuntimeError("method seed mismatch")
            old = group["rows"].get((replication, method))
            if old and old["log_likelihood"] != row["log_likelihood"]:
                raise RuntimeError("same frozen seed produced different likelihoods")
            group["rows"][(replication, method)] = row
        for row in read_csv(result / "prefixes.csv"):
            group["prefixes"][(int(row["replication"]), row["method"], int(row["time"]))] = row

    results = []
    for (d, data_seed, data_hash, source_hash, runner_hash, fit_mode, floor_power, doubling, fit_maxit), group in groups.items():
        complete = sorted(r for r in group["requested"]
                          if all((r, method) in group["rows"] for method in METHODS))
        n = len(complete)
        all_complete = n == len(group["requested"])
        # Failed/unfinished repetitions can select the reported sample.
        # Bootstrap cannot repair that selection, or a tiny repeat count.
        indices = bootstrap_indices(n, 20260920 + d + data_seed) if all_complete and n >= 32 else []
        methods, values = {}, {}
        for j, method in enumerate(METHODS):
            rows = [group["rows"][(r, method)] for r in complete]
            values[method] = [float(r["ratio"]) for r in rows]
            if not rows:
                continue
            mean_ci = bootstrap_mean(values[method], indices) if indices else None
            methods[method] = dict(n=n, mean_ratio=stats.fmean(values[method]),
                sd_ratio=stats.stdev(values[method]) if n > 1 else None, mean_ratio_ci95=mean_ci,
                mean_accuracy_screen=bool(all_complete and n >= 32 and mean_ci and
                                          mean_ci[0] >= .9 and mean_ci[1] <= 1.1),
                log_error_rmse=math.sqrt(stats.fmean(float(r["log_ratio"])**2 for r in rows)),
                mean_resampling=stats.fmean(float(r["resampling_count"]) for r in rows),
                mean_particles=stats.fmean(float(r["particles"]) for r in rows),
                mean_iterations=stats.fmean(float(r["iterations"]) for r in rows),
                mean_seconds=stats.fmean(float(r["wall_seconds"]) for r in rows),
                fit_nonconvergence=sum(int(r["fit_nonconvergence"]) for r in rows),
                paper_sd=PAPER_SD[d][j] if j < 3 else None)
        comparisons = {}
        if n > 1:
            for method in METHODS[1:]:
                samples = []
                for sample in indices:
                    denominator = stats.variance(values[method][i] for i in sample)
                    if denominator > 0:
                        samples.append(stats.variance(values["iapf"][i] for i in sample)/denominator)
                ci = interval(samples)
                # Accurate sampling is required before interpreting low variance.
                eligible = all_complete and n >= 32 and methods["iapf"]["mean_accuracy_screen"] and methods[method]["mean_accuracy_screen"]
                comparisons[method] = dict(variance_ratio_ci95=ci,
                    finite_bootstraps=len(samples), eligible_for_ranking=eligible,
                    supported_variance_ordering=("iapf_lower" if ci and ci[1] < 1 else
                        "iapf_higher" if ci and ci[0] > 1 else "unresolved") if eligible else "ineligible")
        conditional = {}
        for situation in ("ordinary", "large_innovation"):
            scores = {}
            for method in METHODS:
                scores[method] = []
                for replication in complete:
                    rows = [group["prefixes"][(replication, method, t)] for t in range(1, 101)]
                    errors = [float(r["log_prefix_error"])**2 for r in rows if r["situation"] == situation]
                    if errors:
                        scores[method].append(stats.fmean(errors))
            if not scores["iapf"]:
                continue
            result = dict(mean_log_prefix_mse={m: stats.fmean(v) for m, v in scores.items()})
            result["iapf_minus_heuristic_ci95"] = {
                m: bootstrap_mean([a-b for a, b in zip(scores["iapf"], scores[m])], indices)
                for m in METHODS[1:]} if indices else {}
            result["heuristic_veto_supported"] = bool(all_complete and n >= 32 and any(
                ci[0] > 0 for ci in result["iapf_minus_heuristic_ci95"].values()))
            result["inference_status"] = "assessable" if indices else "incomplete_descriptive_only"
            result["descriptively_lower_error_heuristics"] = [m for m in METHODS[1:]
                if stats.fmean(scores[m]) < stats.fmean(scores["iapf"])]
            conditional[situation] = result
        fits = [row for key, row in group["fits"].items() if key[0] in complete]
        results.append(dict(dimension=d, data_seed=data_seed, data_sha256=data_hash,
            fit_mode=fit_mode, equation_15_objective=fit_mode == "paper_eq15",
            floor_tail_power=floor_power, doubling_mode=doubling,
            fit_maxit=fit_maxit,
            reference_sha256=source_hash, runner_sha256=runner_hash, attempts=group["attempts"],
            requested=len(group["requested"]), complete=n, all_requested_complete=all_complete,
            missing_replicates=sorted(group["requested"]-set(complete)), methods=methods,
            comparisons=comparisons, conditional=conditional,
            fit_diagnostics=dict(count=len(fits),
                loss_underflows=sum(r.get("loss_underflow") == "TRUE" for r in fits),
                residual_over_half=sum(float(r["relative_residual"]) > .5 for r in fits))))

    payload = dict(schema="independent_r_iapf_replication_summary_v1", canonical=False,
                   bootstrap=dict(replicates=BOOTSTRAPS, method="paired percentile; resampling whole replicates"),
                   attempts=attempts, worker_seconds=sum(a["wall_seconds"] for a in attempts),
                   source_snapshots_and_data_verified=True, groups=results)
    (out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Bounded R iAPF replication: numerical tables", "",
             "Original-author data and unspecified solver/floor settings were not recovered. "
             "Published table entries are descriptive comparisons on different simulated observations. "
             "Interrupted batches cannot support a success-only ranking; their uncertainty intervals "
             "are suppressed because bootstrap cannot correct outcome-dependent missing runs. "
             "A false heuristic-veto flag in an incomplete group is not a passed screen.", "",
             f"Worker time: {payload['worker_seconds']:.2f}/{seconds} seconds; {len(attempts)}/{launch_limit} launches.", "",
             "| Attempt | d | Requested | Completed iAPF | Status | Seconds |",
             "|---|---:|---:|---:|---|---:|"]
    for a in attempts:
        lines.append(f"| {a['attempt']} | {a['dimension']} | {a['requested']} | {a['completed_iapf']} | {a['status']} | {a['wall_seconds']:.2f} |")
    for a in attempts:
        if "oracle" in a:
            lines += ["", "Exact-twist control (not fitted-iAPF replication): " + json.dumps(a["oracle"]) + "."]
    for g in results:
        lines += ["", f"## d={g['dimension']}, data seed {g['data_seed']}", "",
                  f"Fitter: {g['fit_mode']}; equation (15): {g['equation_15_objective']}; "
                  f"floor power: {g['floor_tail_power']}; doubling: {g['doubling_mode']}.", "",
                  f"Complete paired repeats: {g['complete']}/{g['requested']}. "
                  f"All requested repeats complete: {g['all_requested_complete']}.", "",
                  "| Method | Mean Zhat/Z | Mean 95% CI | SD | Paper SD | Resamples | Mean N | Seconds |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        for m, v in g["methods"].items():
            lines.append("| " + m + " | " + " | ".join(fmt(v[k]) for k in (
                "mean_ratio", "mean_ratio_ci95", "sd_ratio", "paper_sd", "mean_resampling", "mean_particles", "mean_seconds")) + " |")
        lines += ["", "| iAPF variance / comparator variance | 95% interval | Ranking status |",
                  "|---|---|---|"]
        for m, c in g["comparisons"].items():
            lines.append(f"| {m} | {fmt(c['variance_ratio_ci95'])} | {c['supported_variance_ordering']} |")
        lines += ["", "| Innovation situation | iAPF prefix log MSE | BPF | Fully adapted | SIS | Supported heuristic veto |",
                  "|---|---:|---:|---:|---:|---|"]
        for s, c in g["conditional"].items():
            lines.append(f"| {s} | " + " | ".join(fmt(c["mean_log_prefix_mse"][m]) for m in METHODS) + f" | {c['heuristic_veto_supported']} |")
        lines += ["", "Fit diagnostics: " + json.dumps(g["fit_diagnostics"]) + "."]
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(worker_seconds=payload["worker_seconds"], attempts=len(attempts),
                         groups=[dict(dimension=g["dimension"], complete=g["complete"],
                                      requested=g["requested"]) for g in results])))


if __name__ == "__main__":
    main()
