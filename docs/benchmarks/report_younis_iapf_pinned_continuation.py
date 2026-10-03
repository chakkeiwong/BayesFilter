"""Post-run diagnostic reporting and independent arithmetic checks only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs/plans/artifacts/younis-iapf-pinned-continuation-20260919-01"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    manifest = read(path / "manifest.json")
    result = read(path / "results.json")
    for name, expected in manifest["source_sha256"].items():
        assert sha(path / "source_snapshot" / name) == expected, name
    for name, expected in manifest["input_artifacts_sha256"].items():
        assert sha(REPO / name) == expected, name
    report = dict(attempt=path.name, status=manifest["status"],
        stage=manifest["continuation_stage"], wall_seconds=manifest["wall_seconds"],
        charges={k: v-manifest["prior_budget"][k] for k, v in manifest["budget"]["counts"].items()},
        source_snapshots_verified=True, rows=[])
    if manifest["status"] != "complete":
        report["failure"] = manifest.get("error")
        return report
    assert manifest["terminal_source_sha256_match"]
    assert manifest["cuda_visible_devices"] == "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
    devices = manifest["runtime"]["memory_policy"]["physical_devices"]
    assert len(devices) == 1 and devices[0]["memory_growth"]
    assert devices[0]["device_details"]["device_name"] == "NVIDIA GeForce RTX 5080"
    assert all(v == 1 for v in manifest["trace_counts"].values())
    assert sha(path / "frozen-calibration.json") == manifest["frozen_calibration_sha256"]
    frozen = read(path / "frozen-calibration.json")
    assert frozen["bound_trigger_frozen_before_final"]
    assert frozen["bound_repair_trigger"] == result["bound_repair_trigger"]
    for key, dataset in result["datasets"].items():
        assert dataset["regression"] == frozen["datasets"][key]["regression"]
        assert result["fits"][key] == frozen["fits"][key]
        assert all(v.endswith("GPU:0") for v in dataset["actual_tensor_devices"].values())
        assert dataset["kernel_trace_count"] == 1
        assert len(dataset["calibration"]) == 96 and len(dataset["final"]) == 64
        reference = result["references"][key]["score"]
        recomputed = {}
        for name, regression in dataset["regression"].items():
            scores = []
            for row in dataset["final"]:
                controls = row["ancestor"]+(row["innovation"] if name == "innovation" else [])
                scores.append([raw-sum(control*coefficient[j]
                    for control, coefficient in zip(controls, regression["coefficient"]))
                    for j, raw in enumerate(row["score"])])
            mse = statistics.mean(sum((a-b)**2 for a, b in zip(score, reference)) for score in scores)
            assert abs(mse-dataset["summary"][name]["score_squared_error"]) < 1e-11
            means = [statistics.mean(column) for column in zip(*scores)]
            assert max(abs(a-b) for a, b in zip(means, dataset["summary"][name]["mean_score"])) < 1e-11
            recomputed[name] = mse
        new = recomputed["innovation"]
        for name, comparison in dataset["comparisons"].items():
            baseline = (dataset["summary"][name]["score_squared_error"] if name in dataset["summary"]
                        else dataset["deterministic"][name]["score_squared_error"])
            assert abs((new-baseline)-comparison["mean_difference"]) < 1e-11
        report["rows"].append(dict(dataset=key, regime=dataset["regime"],
            ancestor_mse=recomputed["ancestor_fresh"], innovation_mse=new,
            observed_reduction_percent=100*(1-new/recomputed["ancestor_fresh"]),
            primary_interval=dataset["comparisons"]["ancestor_fresh"]["primary_interval_99_75"],
            primary_pass=dataset["comparisons"]["ancestor_fresh"]["primary_pass"],
            bias_screen_pass=dataset["summary"]["innovation"]["bias_screen_pass"],
            heuristic_losses=dataset["heuristic_dominance"]["observed_losses"],
            ukf_mse=dataset["deterministic"]["ukf"]["score_squared_error"],
            no_resampling_mse=dataset["summary"].get("no_resampling", {}).get("score_squared_error"),
            fit_bound_contact=result["fits"][key]["boundary_active"],
            fitting_particle_count=result["fits"][key]["fit"]["particles"],
            predictive_shape_residual=result["fits"][key]["with_floor_shape"],
            predictive_pointwise_floor_fraction=result["fits"][key]["predictive_floor_fraction"],
            stopping_cv_supremum=math.sqrt(result["fits"][key]["details"]["iapf_configuration"]["k"]+1),
            stopping_tau=result["fits"][key]["details"]["iapf_configuration"]["tau"],
            stopping_test_informative=(result["fits"][key]["details"]["iapf_configuration"]["tau"] <
                                      math.sqrt(result["fits"][key]["details"]["iapf_configuration"]["k"]+1)),
            bound_sensitivity=result["fits"][key].get("bound_sensitivity")))
    report.update(pinned_device_verified=True, numerical_veto=False,
        calibration_and_arithmetic_verified=True, decision=result["decision"],
        allocator_peak_bytes=manifest["memory_usage"]["allocator_peak_bytes"])
    return report


def main():
    reports = [audit(p.parent) for p in sorted(ROOT.glob("*/manifest.json"))]
    reports.sort(key=lambda row: {"pinned": 0, "fresh": 1, "wide": 2}[row["stage"]])
    totals = dict(launches=len(reports), driver_seconds=sum(r["wall_seconds"] for r in reports),
        filter_calls=sum(r["charges"]["filter_calls"] for r in reports),
        adaptive_fits=sum(r["charges"]["adaptive_fits"] for r in reports))
    assert totals["launches"] <= 4 and totals["driver_seconds"] <= 1800
    assert totals["filter_calls"] <= 8000 and totals["adaptive_fits"] <= 8
    seeds = set()
    for path in sorted(ROOT.glob("*/seeds.json")):
        pairs = [tuple(pair) for value in read(path).values() for pair in value]
        assert len(pairs) == len(set(pairs)) and not seeds.intersection(pairs)
        seeds.update(pairs)
    analysis = dict(status="audited", reports=reports, totals=totals,
        all_continuation_control_seed_pairs_disjoint=True, seed_pairs=len(seeds),
        default_or_hmc_or_ledh_admission=False)
    analysis.update(campaign_status="planned_stages_complete_fit_budget_exhausted",
        limits=dict(launches=4, filter_calls=8000, adaptive_fits=8, driver_seconds=1800., cpu_probe_seconds=600.),
        remaining=dict(launches=4-totals["launches"], filter_calls=8000-totals["filter_calls"],
                       adaptive_fits=8-totals["adaptive_fits"], driver_seconds=1800.-totals["driver_seconds"]),
        cpu_probe_seconds_conservative=60.,
        promotion_vetoes=["weak_dataset_heuristic_losses", "persistent_offline_fit_bound_contact",
                         "mathematically_uninformative_inherited_stopping_test", "affine_Kalman_dominance"],
        next_action="Design independent-calibration larger-cloud and informative-stopping fit protocol; no further fitting under exhausted eight-fit ceiling.")
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False)+"\n")
    with (ROOT / "conditional-errors.csv").open("w", newline="") as stream:
        columns = ["stage", "dataset", "regime", "ancestor_mse", "innovation_mse", "ukf_mse",
                   "no_resampling_mse", "observed_reduction_percent", "primary_interval", "primary_pass",
                   "bias_screen_pass", "heuristic_losses", "fit_bound_contact", "fitting_particle_count"]
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for report in reports:
            for row in report["rows"]:
                writer.writerow(dict(stage=report["stage"], **row))
    lines = ["# Pinned replication passes; fresh weak cases expose fitting and accuracy failures", "",
        "2026-09-19. The device correction is confirmed, but the fresh weakly nonlinear datasets still lose "
        "to simple filters. Widening the fitting box leaves the problematic bound contact unresolved. "
        "The inherited stopping threshold is also mathematically uninformative. The candidate is not promoted.", ""]
    for report in reports:
        nonlinear = [r for r in report["rows"] if r["regime"] != "affine"]
        lines += [f"## {report['stage']} ({report['attempt']})", ""]
        if report["status"] != "complete":
            lines += [f"Stopped: {report.get('failure')}. Charges are retained.", ""]
            continue
        losses = [f"{r['dataset']}: {', '.join(r['heuristic_losses'])}" for r in nonlinear if r["heuristic_losses"]]
        lines += [f"Primary comparisons: {sum(r['primary_pass'] for r in nonlinear)}/4 pass. "
                  f"Nonlinear bias screens: {sum(r['bias_screen_pass'] for r in nonlinear)}/4 pass.", "",
                  "**Heuristic losses:** "+("; ".join(losses) if losses else "none on the four nonlinear datasets")+".", "",
                  "| Dataset | Regime | Ancestor MSE | New MSE | UKF MSE | New minus ancestor interval | Fit bound |",
                  "|---|---|---:|---:|---:|---|---|"]
        for r in report["rows"]:
            lines.append(f"| {r['dataset']} | {r['regime']} | {r['ancestor_mse']:.7f} | {r['innovation_mse']:.7f} | "
                         f"{r['ukf_mse']:.7f} | [{r['primary_interval'][0]:.7f}, {r['primary_interval'][1]:.7f}] | {r['fit_bound_contact']} |")
        lines += ["", f"[Manifest]({report['attempt']}/manifest.json), [raw results]({report['attempt']}/results.json), "
                  f"[frozen calibration]({report['attempt']}/frozen-calibration.json), [log]({report['attempt']}.log).", ""]
    lines += ["## Interpretation and decision", "",
        "The paired wider-box check leaves the three interior proposal fits exactly unchanged. Dataset 1900 "
        "still contacts a bound, and its first-step predictive shape residual remains large. Widening is rejected "
        "as a complete fitting repair; it passes the narrower interior non-harm check. Dataset 1901 also loses "
        "to UKF with an interior fit, so removing bound contacts alone cannot resolve the weak-case accuracy gap. "
        "See the [mathematical diagnosis](fitting-diagnosis.md) for the stopping-rule derivation and score limits.", "",
        "All completed attempts verify a single physical RTX5080, memory growth, GPU:0 particle/control outputs, "
        "one trace per kernel, frozen coefficients, numerical references and preserved source hashes. Independent "
        "Python arithmetic reproduces corrected scores and MSE summaries. This repairs the previous device-selection "
        "evidence gap; it does not retroactively change the previous run.", "",
        "The controls subtract independent-calibration linear projections of exactly centered ancestor statistics "
        "and same-law Gaussian moment contrasts. They preserve the raw normalized Fisher estimator's finite-N bias "
        "under the stated law and arithmetic assumptions. They are not derivatives of the finite likelihood program.", "",
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|",
        "| Keep diagnostic candidate only | Pinned 4/4; fresh and wider-box 3/4 each | Weak-case heuristic failures, persistent bound contact and vacuous fitting stop; affine Kalman still wins | Fixed short scalar datasets and one calibration each | Repair fitting-cloud/stopping protocol on separate calibration data; retain UKF | Default, population, LEDH or HMC readiness |",
        "| Close this planned campaign | Three planned stages executed | Eight fitting attempts exhausted | Larger-cloud repair remains untested | Write a fresh bounded fitting protocol before more fitting | Whole-master completion |", "",
        "| Inference status | Finding |", "|---|---|",
        "| Hard veto screen | Numerical/device checks pass for completed runs; observed heuristic losses and bound contacts are listed above |",
        "| Statistically supported ranking | Only per-dataset new-versus-matched-ancestor comparisons with negative primary upper endpoints, conditional on frozen fits |",
        "| Descriptive-only differences | Percent reductions, variance, heuristic mean differences, timings and comparisons across stages |",
        "| Default readiness | Not established; inherited tiny fitting cloud and limited model/horizon remain |",
        "| Next evidence needed | Resolve any bound/heuristic failure, then independently repeated fitting/calibration and longer horizons |", "",
        f"Budget used: {totals['launches']}/4 launches, {totals['filter_calls']}/8000 charged filter calls, "
        f"{totals['adaptive_fits']}/8 adaptive fits, {totals['driver_seconds']:.6f}/1800 driver seconds. "
        "Fitting reserves eight calls per attempt; actual fitting calls are separately retained in each fit record.", "",
        "Post-run red-team: chosen T2 scalar cases and one calibration can exaggerate general usefulness. "
        "A fresh-data success cannot certify the proposal fitting protocol; a heuristic loss is a candidate "
        "failure, not evidence against Fisher's identity or the entire iAPF/KDM/LEDH direction. Calibration "
        "uncertainty and data-population uncertainty remain outside the intervals. Do not retune against final streams.", "",
        "[Plan](../../younis-iapf-pinned-continuation-2026-09-19.md); [analysis](analysis.json); "
        "[conditional errors](conditional-errors.csv); [CPU checks](cpu-tests01.log).", ""]
    (ROOT / "result.md").write_text("\n".join(lines))
    print(json.dumps(dict(audit="pass", **totals, completed_stages=[r["stage"] for r in reports if r["status"] == "complete"])))


if __name__ == "__main__":
    main()
