"""Independent, standard-library-only post-run diagnostic artifact audit."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs/plans/artifacts/younis-iapf-fitting-protocol-20260919-01"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    manifest, results = read(path/"manifest.json"), read(path/"results.json")
    for name, expected in manifest["source_sha256"].items():
        assert sha(path/"source_snapshot"/name) == expected, name
    for name, expected in manifest["input_sha256"].items():
        assert sha(REPO/name) == expected, name
    report = dict(attempt=path.name, stage=manifest["stage"], status=manifest["status"],
        seconds=manifest["wall_seconds"], charges={k: v-manifest["prior_counts"][k]
            for k, v in manifest["budget"]["counts"].items()}, rows=[], source_snapshot_verified=True)
    if manifest["status"] != "complete":
        report["error"] = manifest.get("error")
        return report
    assert manifest["terminal_source_sha256_match"]
    assert manifest["cuda_visible_devices"] == "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
    physical = manifest["runtime"]["memory_policy"]["physical_devices"]
    assert len(physical) == 1 and physical[0]["memory_growth"]
    assert physical[0]["device_details"]["device_name"] == "NVIDIA GeForce RTX 5080"
    assert all(n == 1 for n in manifest["trace_counts"].values())
    assert sha(path/"frozen-calibration.json") == manifest["frozen_calibration_sha256"]
    frozen = read(path/"frozen-calibration.json")
    reserve = None
    if "frozen_reserve_sha256" in manifest:
        assert sha(path/"frozen-reserve.json") == manifest["frozen_reserve_sha256"]
        reserve = read(path/"frozen-reserve.json")
    if "selection_sha256" in manifest:
        assert sha(path/"selection.json") == manifest["selection_sha256"]
        assert read(path/"selection.json") == results["selection"]
    saved_seeds = read(path/"seeds.json")
    assert len({tuple(s) for s in saved_seeds.values()}) == len(saved_seeds)
    for dataset, record in results["datasets"].items():
        reference = record["reference"]
        assert max(reference["mesh_error"], reference["domain_error"], reference["forward_backward_error"]) <= 1e-7
        assert reference["tail_mass"] <= 1e-9
        assert reference["reference_device"].endswith("CPU:0")
        if manifest.get("shared_fit_across_counts"):
            assert record["arms"]["N4096"]["fit"] == record["arms"]["N16384"]["fit"]
        if manifest.get("known_failure_replay"):
            original = read(REPO/manifest["frozen_failure_inputs"][dataset])
            source_fit = (original["fits"][dataset] if "fits" in original else
                          original["datasets"][dataset]["arms"]["large_stable"]["fit"])
            assert record["arms"]["N4096"]["fit"] == source_fit
        for arm, item in record["arms"].items():
            fitted = item["fit"]
            row = dict(dataset=dataset, regime=record["regime"], arm=arm, fit_status=fitted["status"],
                       final_particles=item.get("final_particles", 4096))
            if fitted["status"] != "valid":
                row["failure"] = fitted["reason"]
                history = fitted["diagnostics"]["details"]["fit_iterations"]
                row["last_cv"] = history[-1]["cv"]
                row["last_particles"] = history[-1]["particles"]
                report["rows"].append(row)
                continue
            saved = (reserve if arm == "reserve_stable" else frozen)["datasets"][dataset]["arms"][arm]
            assert saved["assessment"] == []
            for name in ("fit", "calibration", "regression"):
                assert saved[name] == item[name], (dataset, arm, name)
            assert item["frozen_before_assessment"]
            assert len(item["calibration"]) == 96 and len(item["assessment"]) == 64
            assert item["kernel_trace_count"] == 1
            assert all(d.endswith("GPU:0") for d in item["actual_tensor_devices"])
            for score_name, size in (("ancestor", 12), ("innovation", 18)):
                coefficients = item["regression"][score_name]["coefficient"]
                scores = [[r["score"][j]-sum(x*coefficients[k][j] for k, x in enumerate(
                    r["ancestor"]+(r["innovation"] if size == 18 else []))) for j in range(6)]
                    for r in item["assessment"]]
                assert max(abs(a-b) for x, y in zip(scores, item["scores"][score_name]) for a, b in zip(x, y)) < 1e-10
            for name, scores in item["scores"].items():
                mse = statistics.mean(sum((a-b)**2 for a, b in zip(score, reference["score"])) for score in scores)
                assert abs(mse-item["summary"][name]["score_squared_error"]) < 1e-10*(1+mse)
            steps = fitted["details"]["fit_iterations"]
            row.update(last_cv=steps[-1]["cv"], last_particles=fitted["fit"]["particles"], iterations=len(steps),
                fitting_bound=fitted["boundary_active"], shape_residual=fitted["with_floor_shape"],
                floor_fraction=fitted["predictive_floor_fraction"],
                raw_mse=item["summary"]["raw_fisher"]["score_squared_error"],
                ancestor_mse=item["summary"]["ancestor"]["score_squared_error"],
                innovation_mse=item["summary"]["innovation"]["score_squared_error"],
                ukf_mse=record["deterministic"]["ukf"]["score_squared_error"],
                ekf_mse=record["deterministic"]["ekf"]["score_squared_error"],
                no_resampling_mse=(record["no_resampling_by_particles"][str(row["final_particles"])]["summary"]["score_squared_error"]
                                   if "no_resampling_by_particles" in record else record["no_resampling_summary"]["score_squared_error"]),
                heuristic_losses=item["heuristic_dominance"]["observed_losses"],
                bias_pass=item["summary"]["innovation"]["bias_screen_pass"])
            if "primary_comparison" in record:
                row["primary_comparison"] = record["primary_comparison"]
            report["rows"].append(row)
    report.update(selection=results["selection"], decision=results.get("decision"),
                  frozen_verified=True, independent_score_arithmetic_verified=True, seed_pairs=len(saved_seeds))
    return report


def main():
    reports = [audit(p.parent) for p in sorted(ROOT.glob("*/manifest.json"))]
    all_seeds = set()
    for p in ROOT.glob("*/seeds.json"):
        current = {tuple(s) for s in read(p).values()}
        assert not (current & all_seeds), p
        all_seeds |= current
    previous_seeds = set()
    for folder in ("younis-iapf-resampling-control-20260918-01", "younis-iapf-pinned-continuation-20260919-01"):
        for p in (ROOT.parent/folder).glob("*/seeds.json"):
            previous_seeds.update(tuple(pair) for pairs in read(p).values() for pair in pairs)
    # The old fitting seeds are archived inside result records, separately from
    # score-stream files. Reusing a frozen proposal never reuses its draw stream.
    for p in (ROOT.parent/"younis-iapf-pinned-continuation-20260919-01").glob("*/results.json"):
        for fitted in read(p).get("fits", {}).values():
            previous_seeds.update(tuple(pair) for pair in fitted.get("details", {}).get("fit_seed_records", {}).values())
    assert not (all_seeds & previous_seeds)
    charges = {k: sum(r["charges"][k] for r in reports) for k in ("filter_calls", "adaptive_fits")}
    analysis = dict(audit="pass", runs=reports, charges=charges,
                    driver_seconds=sum(r["seconds"] for r in reports), unique_seed_pairs=len(all_seeds),
                    archived_seed_pairs_checked=len(previous_seeds), archived_seed_overlap=False)
    (ROOT/"analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False)+"\n")
    fields = ["stage", "dataset", "regime", "arm", "fit_status", "final_particles", "last_particles", "last_cv", "iterations",
              "fitting_bound", "raw_mse", "ancestor_mse", "innovation_mse", "ukf_mse", "ekf_mse", "no_resampling_mse",
              "heuristic_losses", "bias_pass", "failure"]
    with (ROOT/"conditional-errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for report in reports:
            for row in report["rows"]:
                writer.writerow(dict(stage=report["stage"], **row))
    print(json.dumps(dict(audit="pass", charges=charges, driver_seconds=analysis["driver_seconds"],
                         stages=[dict(stage=r["stage"], status=r["status"], selection=r.get("selection"), decision=r.get("decision")) for r in reports])))


if __name__ == "__main__":
    main()
