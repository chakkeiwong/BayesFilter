"""Post-run diagnostic reporting for the independent R fitting comparison."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plans/artifacts/iapf-r-full-filter-fitting-20260921-01"


def rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def ci(values, statistic=stats.mean):
    rng = random.Random(9212026)
    n = len(values)
    values = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(4000))
    def quantile(q):
        pos = q*(len(values)-1)
        lo = int(pos)
        return values[lo]+(pos-lo)*(values[min(lo+1, len(values)-1)]-values[lo])
    return [quantile(.025), quantile(.975)]


def summarize(attempt):
    path = OUT / attempt["name"] / "results"
    raw = rows(path / "replicates.csv")
    groups = defaultdict(list)
    for row in raw:
        groups[row["method"]].append(row)
    output = {}
    for method, records in groups.items():
        complete = [r for r in records if r["status"] == "complete"]
        item = dict(n=len(complete), planned=attempt["repeats"], failures=[
            {k: r[k] for k in ("replication", "error", "wall_seconds")} for r in records if r["status"] != "complete"],
            total_recorded_seconds=sum(float(r["wall_seconds"]) for r in records))
        if complete:
            ratios = [float(r["ratio"]) for r in complete]
            log_ratios = [float(r["log_ratio"]) for r in complete]
            item.update(mean=stats.mean(ratios), sd=stats.stdev(ratios) if len(ratios)>1 else None,
                        minimum=min(ratios), maximum=max(ratios),
                        minimum_log_ratio=min(log_ratios), maximum_log_ratio=max(log_ratios),
                        ratio_underflow_count=sum(value == 0 for value in ratios),
                        mean_particles=stats.mean(float(r["particles"]) for r in complete),
                        mean_seconds=stats.mean(float(r["wall_seconds"]) for r in complete),
                        mean_resampling=stats.mean(float(r["resampling_count"]) for r in complete),
                        maximum_floor_probability=max(float(r["floor_probability_max"]) for r in complete))
            if attempt["stage"] in ("validation", "controller") and len(complete) == attempt["repeats"]:
                item.update(mean_ci95=ci(ratios), sd_ci95=ci(ratios, stats.stdev))
                item["mean_interval_contains_one"] = item["mean_ci95"][0] <= 1 <= item["mean_ci95"][1]
                paper_sd, paper_n, paper_resampling = {20: (.19,1000,27.61),80: (.35,1142,71.88)}[attempt["dimension"]]
                item["practical_primary_pass"] = (.8 <= item["mean_ci95"][0]
                    and item["mean_ci95"][1] <= 1.2 and item["sd_ci95"][1] <= 2*paper_sd
                    and item["mean_particles"] <= 1.5*paper_n)
                item["strict_mean_pass"] = .9 <= item["mean_ci95"][0] and item["mean_ci95"][1] <= 1.1
                item["literal_pattern_pass"] = (.5*paper_sd <= item["sd_ci95"][0]
                    and item["sd_ci95"][1] <= 2*paper_sd and .5*paper_resampling <= item["mean_resampling"] <= 2*paper_resampling)
                if method in ("bpf", "fully_adapted", "sis"):
                    # These thresholds concern the iAPF reconstruction, not the
                    # deliberately different fixed-N heuristic comparators.
                    item["practical_primary_pass"] = item["literal_pattern_pass"] = None
        output[method] = item
    return dict(name=attempt["name"], stage=attempt["stage"], dimension=attempt["dimension"],
                arm=attempt["arm"], data_seed=attempt["data_seed"], inspection=attempt["inspection"], methods=output)


def compare(cells):
    for name in ("observations.csv", "kalman.csv"):
        hashes = {hashlib.sha256((OUT/a["name"]/"results"/name).read_bytes()).hexdigest() for a in cells}
        if len(hashes) != 1:
            raise RuntimeError("mismatched data or Kalman authority")
    values = defaultdict(lambda: defaultdict(list))
    for a in cells:
        for r in rows(OUT/a["name"]/"results"/"prefixes.csv"):
            values[(r["method"], r["situation"])][int(r["replication"])].append(math.expm1(float(r["log_prefix_error"]))**2)
    means = {key: {i: stats.mean(v) for i,v in reps.items()} for key,reps in values.items()}
    fitted = sorted({m for m,s in means if m in ("qr", "short_qr") or m.startswith("box")})
    expected = cells[0]["repeats"]
    table = []
    for method in fitted:
        for situation in ("ordinary", "large_innovation"):
            own = means[method, situation]
            for comparator in ("qr", "bpf", "fully_adapted", "sis"):
                if comparator == method or (comparator, situation) not in means:
                    continue
                other = means[comparator, situation]
                if set(own) != set(other) or len(own) != expected:
                    table.append(dict(method=method, comparator=comparator, situation=situation,
                                      status="incomplete paired evidence"))
                    continue
                delta = [own[i]-other[i] for i in sorted(own)]
                interval = ci(delta)
                table.append(dict(method=method, comparator=comparator, situation=situation, n=expected,
                                  status="complete", mean_mse=stats.mean(own.values()),
                                  comparator_mean_mse=stats.mean(other.values()), difference=stats.mean(delta),
                                  difference_ci95=interval, observed_loss=stats.mean(delta)>0,
                                  heuristic_promotion_veto=comparator != "qr" and stats.mean(delta)>0,
                                  pointwise_difference_supported=interval[0]>0 or interval[1]<0))
    return table


def main():
    target = OUT / (sys.argv[1] if len(sys.argv)>1 else "terminal-analysis")
    target.mkdir(exist_ok=False)
    ledger = json.loads((OUT / "manifest.json").read_text())
    for name, expected in ledger["source_hashes"].items():
        if hashlib.sha256((OUT/"source-snapshot"/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("source snapshot changed")
    numerical = [a for a in ledger["attempts"] if a["stage"] in ("calibration","validation","controller")
                 and not a.get("used_in_derived")]+ledger.get("derived_cells", [])
    summaries = [summarize(a) for a in numerical]
    by_data = defaultdict(list)
    for a in numerical:
        by_data[a["stage"], a["data_seed"]].append(a)
    for (stage, seed), cells in by_data.items():
        for name in ("observations.csv", "kalman.csv"):
            hashes = {hashlib.sha256((OUT/a["name"]/"results"/name).read_bytes()).hexdigest() for a in cells}
            if len(hashes) != 1:
                raise RuntimeError("comparison data mismatch")
    comparisons = {str(seed): compare(cells) for (stage,seed),cells in by_data.items()
                   if (stage == "validation" and len(cells) >= 2) or stage == "controller"}
    controller_pairs = {}
    for attempt in numerical:
        if attempt["stage"] != "controller":
            continue
        records = rows(OUT/attempt["name"]/"results"/"replicates.csv")
        grouped = {m: {r["replication"]: r for r in records if r["method"] == m and r["status"] == "complete"}
                   for m in ("qr", "short_qr")}
        if set(grouped["qr"]) != set(grouped["short_qr"]) or len(grouped["qr"]) != attempt["repeats"]:
            continue
        pair = {}
        for field in ("particles", "wall_seconds", "squared_likelihood_error"):
            def metric(row):
                return (float(row["ratio"])-1)**2 if field == "squared_likelihood_error" else float(row[field])
            differences = [metric(grouped["short_qr"][i])-metric(grouped["qr"][i]) for i in sorted(grouped["qr"])]
            pair[field] = dict(mean_difference=stats.mean(differences), difference_ci95=ci(differences))
        controller_pairs[str(attempt["data_seed"])] = pair
    result = dict(cells=summaries, conditional_comparisons=comparisons, controller_pairs=controller_pairs,
                  worker_seconds=ledger["worker_seconds"], remaining_seconds=ledger["remaining_seconds"],
                  launches=len(ledger["attempts"]), bootstrap_replicates=4000,
                  calibration_uncertainty="four repeats: descriptive only, no ranking",
                  validation_uncertainty="pointwise replicate bootstrap; not simultaneous or paper-wide inference")
    result["heuristic_dominance"] = {
        seed: {method: ("incomplete" if any(row["status"] != "complete" for row in table if row["method"] == method)
                        else "veto" if any(row.get("heuristic_promotion_veto") for row in table if row["method"] == method)
                        else "passes_observed_screen")
               for method in sorted({row["method"] for row in table})}
        for seed, table in comparisons.items()}
    result["underflow_interpretation"] = (
        "A reported ratio of zero with finite log ratio is exponentiation underflow. "
        "Its empirical SD must not be interpreted as zero theoretical variance; log ratios are retained. "
        "Prefix squared errors use expm1(log_ratio) and round to one for sufficiently negative values.")
    (target/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines = ["# Full-filter fitting comparison: numerical tables", "",
             "Calibration means/SDs describe complete replicas only. Failures and planned counts are shown;",
             "successful-subset summaries never establish a failed arm's accuracy.", "",
             "| Cell | Method | Complete/planned | Failures | Mean ratio | SD | Mean N | Total mean seconds | Resampling |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for method,r in s["methods"].items():
            def fmt(key):
                return f"{r[key]:.6g}" if r.get(key) is not None else "N/A"
            lines.append(f"| {s['name']} | {method} | {r['n']}/{r['planned']} | {len(r['failures'])} | "
                         +" | ".join(fmt(k) for k in ("mean","sd","mean_particles","mean_seconds","mean_resampling"))+" |")
    lines += ["", "All complete-cell and paired intervals are retained in summary.json.",
              f"Worker use: {ledger['worker_seconds']:.6f}/2400 seconds; {len(ledger['attempts'])}/16 launches."]
    (target/"tables.md").write_text("\n".join(lines)+"\n")
    print(target)
    for s in summaries:
        print(s["name"], {m: {k:r.get(k) for k in ("n","mean","sd","mean_particles","mean_seconds")}
                           for m,r in s["methods"].items()})


if __name__ == "__main__":
    main()
