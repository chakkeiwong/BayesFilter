"""CPU-only terminal inspection of recorded iAPF diagnostic paths.

This does not implement a filtering method or promote a numerical default.
The terminal CDF is inspected as recorded; uniforms are not regenerated here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = "docs/plans/artifacts/iapf-tf32-recorded-paths-20260923-01"
PLAN = "docs/plans/iapf-tf32-recorded-paths-2026-09-23.md"
MODES = ("fp64", "fp32", "tf32")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON token: {value}")


def check_trace(record, replicate, n, d, horizon):
    assert record["replicate"] == replicate and record["neutral"] is True
    assert len(record["score"]) == 6 and all(map(math.isfinite, record["score"]))
    assert len(record["cloud"]) == horizon
    for cloud in record["cloud"]:
        assert len(cloud) == n
        assert all(len(row) == d and all(map(math.isfinite, row)) for row in cloud)
    assert len(record["cdf"]) == len(record["ancestors"]) == horizon + 1
    for cdf, ancestors in zip(record["cdf"], record["ancestors"]):
        assert len(cdf) == len(ancestors) == n
        assert all(map(math.isfinite, cdf))
        assert cdf[0] >= 0 and all(x <= y for x, y in zip(cdf, cdf[1:]))
        assert abs(cdf[-1] - 1.) <= n * 2**-23
        assert all(type(x) is int and 0 <= x < n for x in ancestors)
    assert len(record["probabilities"]) == len(record["mixture_choices"]) == horizon
    for probabilities, choices in zip(record["probabilities"], record["mixture_choices"]):
        assert len(probabilities) == len(choices) == n
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
        assert all(type(x) is bool for x in choices)


def lattice_intervals(cdf):
    lattice = 2**23
    bounds = [0] + [math.ceil(lattice * min(1., max(0., x))) for x in cdf[:-1]] + [lattice]
    return list(zip(bounds[:-1], bounds[1:]))


def localize(a, b, horizon, n):
    def ancestor_event(index):
        ca, cb = a["cdf"][index], b["cdf"][index]
        agreement = sum(max(0, min(v, y) - max(u, x))
                        for (u, v), (x, y) in zip(lattice_intervals(ca), lattice_intervals(cb))) / 2**23
        return dict(operation="ancestor", cdf_index=index, can_affect_likelihood=index < horizon,
                    max_cdf_difference=max(abs(x-y) for x, y in zip(ca, cb)),
                    conditional_lattice_disagreement_probability=1-agreement,
                    expected_changed_ancestors=n*(1-agreement),
                    observed_changed_ancestors=sum(x != y for x, y in zip(a["ancestors"][index], b["ancestors"][index])))

    first = ancestor_event(0) if a["ancestors"][0] != b["ancestors"][0] else None
    pre = post = 0.
    for t in range(horizon):
        if first is None and a["mixture_choices"][t] != b["mixture_choices"][t]:
            first = dict(operation="mixture", time=t, can_affect_likelihood=True,
                         max_probability_difference=max(abs(x-y) for x, y in zip(a["probabilities"][t], b["probabilities"][t])))
        error = max(abs(x-y) for u, v in zip(a["cloud"][t], b["cloud"][t]) for x, y in zip(u, v))
        if first is None:
            pre = max(pre, error)
        else:
            post = max(post, error)
        if first is None and a["ancestors"][t+1] != b["ancestors"][t+1]:
            first = ancestor_event(t+1)

    def minimum_ess(record):
        values = []
        for cdf in record["cdf"]:
            weights = [y-x for x, y in zip([0.] + cdf[:-1], cdf)]
            values.append(math.fsum(weights)**2 / math.fsum(x*x for x in weights))
        return min(values)

    return dict(first_changed_decision=first, max_cloud_error_before_changed_decision=pre,
                max_cloud_error_after_changed_decision=post, baseline_minimum_ess=minimum_ess(a),
                tf32_minimum_ess=minimum_ess(b),
                max_fixed_label_score_difference=max(abs(x-y) for x, y in zip(a["score"], b["score"])))


def review(repo, root, out, replicates=1024, seed_offset=10000000):
    completed = json.loads((root / "completed.json").read_text())
    assert all(key in completed for key in (*MODES, "summary", "fixtures"))
    fixture = Path(completed["fixtures"])
    source_count = output_count = 0
    fixture_hashes = None
    previous = json.loads((repo / "docs/plans/artifacts/iapf-tf32-revalidation-20260923-01/completed.json").read_text())
    runtime_reference = json.loads((Path(previous["fp64"]).parent / "manifest.json").read_text())["source_sha256"]
    for path in sorted(root.glob("attempt*/manifest.json")):
        manifest = json.loads(path.read_text())
        assert manifest["status"] in ("complete", "failed"), path
        for name, digest in manifest["source_sha256"].items():
            assert sha(Path(manifest["source"]) / name) == digest, (path, name)
            if name.startswith("bayesfilter/"):
                assert runtime_reference[name] == digest, ("Runtime source changed", name)
            source_count += 1
        for name, digest in manifest["output_sha256"].items():
            assert sha(path.parent / name) == digest, (path, name)
            output_count += 1
        if fixture_hashes is not None:
            assert fixture_hashes == manifest["fixture_sha256"]
        fixture_hashes = manifest["fixture_sha256"]
    assert fixture_hashes
    for name, digest in fixture_hashes.items():
        assert sha(fixture / name) == digest, name
    original_cells = read_csv(fixture / "cells.csv")
    cells = [x for x in original_cells if int(x["d"]) == 2]
    assert len(cells) == 12
    cell_map = {(x["id"], x["arm"]): x for x in cells}
    assert len(cell_map) == 12
    original_starts = {int(x["particle_seed"]) for x in original_cells}
    fresh_starts = sorted({int(x["particle_seed"]) + seed_offset for x in cells})
    assert len(fresh_starts) == 3
    assert all(b > a + replicates - 1 for a, b in zip(fresh_starts, fresh_starts[1:]))
    assert all(not (fresh <= old + 511 and old <= fresh + replicates - 1)
               for fresh in fresh_starts for old in original_starts)
    if seed_offset == 20000000:
        phase20_starts = {int(x["particle_seed"]) + 10000000 for x in cells}
        assert all(not (fresh <= old + 1023 and old <= fresh + replicates - 1)
                   for fresh in fresh_starts for old in phase20_starts)
    oracle = read_csv(fixture / "oracle-checks.csv")
    assert len(oracle) == 11 and all(float(x["error"]) < 1e-9 for x in oracle)
    records = {}
    mode_results = {}
    for mode in MODES:
        directory = Path(completed[mode])
        result = json.loads((directory / "result.json").read_text())
        assert result["status"] == "complete" and result["jit_compile"] is True
        assert result["memory_policy"]["all_physical_devices_memory_growth"]
        assert result["cuda_visible_devices"] == "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
        assert result["tf32_verified"] == (mode == "tf32")
        assert result["distribution_records"] == 12 * replicates and result["replicates"] == replicates
        assert result["particle_seed_offset"] == seed_offset
        assert result["trace_all"] and result["compress_traces"]
        assert result["consumer"] == "bayesfilter.score_study.fitted_twist_tf.make_fitted_twist_kernel"
        checks = result["old_fixture_checks"]
        assert len(checks) == 18
        if mode != "tf32":
            assert all(x["passed"] for x in checks)
        for compilation in result["compilation"]:
            assert "GPU" in compilation["device"]
            assert sha(directory / (compilation["label"] + ".hlo.txt")) == compilation["hlo_sha256"]
        mode_results[mode] = dict(old_pass=sum(x["passed"] for x in checks), old_checks=len(checks),
                                  old_failed=[x for x in checks if not x["passed"]],
                                  compilations=len(result["compilation"]), allocator=result["allocator"],
                                  tensorflow=result["tensorflow"], memory_policy=result["memory_policy"])
        mode_records = read_csv(directory / "records.csv")
        assert len(mode_records) == 12 * replicates
        for row in mode_records:
            key = (row["cell"], row["arm"], mode, int(row["replicate"]))
            assert key not in records and row["mode"] == mode
            cell = cell_map[key[:2]]
            assert 0 <= key[3] < replicates
            assert int(row["seed"]) == int(cell["particle_seed"]) + seed_offset + key[3]
            assert all(int(row[k]) == int(cell[k]) for k in ("d", "N", "T"))
            z = float(row["normalized_likelihood"])
            assert math.isfinite(z) and z >= 0 and math.isfinite(float(row["log_likelihood"]))
            assert math.isclose(z, math.exp(float(row["log_likelihood"])-float(row["kalman"])), rel_tol=1e-12)
            records[key] = row
    summary = Path(completed["summary"])
    precision = read_csv(summary / "precision-comparisons.csv")
    heuristic = read_csv(summary / "heuristic-comparisons.csv")
    assert len(precision) == 24 and len(heuristic) == 27
    assert len(read_csv(summary / "conditional-summary.csv")) == 36
    assert len(read_csv(summary / "records.csv")) == 36 * replicates
    influence = []
    selected = set()
    for contrast in precision:
        key = (contrast["cell"], contrast["arm"])
        baseline = contrast["comparator"]
        assert baseline in ("fp64", "fp32") and key in cell_map
        assert int(contrast["n"]) == replicates and int(contrast["contrasts"]) == 24
        pairs = []
        for rep in range(replicates):
            a, b = records[(*key, baseline, rep)], records[(*key, "tf32", rep)]
            assert a["seed"] == b["seed"]
            pairs.append(dict(replicate=rep, seed=int(a["seed"]),
                              baseline_normalized_likelihood=float(a["normalized_likelihood"]),
                              candidate_normalized_likelihood=float(b["normalized_likelihood"]),
                              difference=float(b["normalized_likelihood"])-float(a["normalized_likelihood"])))
        mean = math.fsum(x["difference"] for x in pairs) / replicates
        assert math.isclose(mean, float(contrast["difference"]), rel_tol=1e-8, abs_tol=1e-12)
        squares = [(x["difference"]-mean)**2 for x in pairs]
        total = math.fsum(squares)
        for record, square in zip(pairs, squares):
            record["squared_share"] = square/total if total else 0.
        top = sorted(pairs, key=lambda x: x["squared_share"], reverse=True)[:3]
        assert math.isclose(top[0]["squared_share"], float(contrast["largest_centered_squared_share"]), rel_tol=1e-6, abs_tol=1e-8)
        passed = (float(contrast["lower"]) > -.02 and float(contrast["upper"]) < .02 and top[0]["squared_share"] <= .2)
        assert passed == (contrast["equivalence"] == "TRUE")
        for row in top:
            selected.add((*key, baseline, row["replicate"]))
            selected.add((*key, "tf32", row["replicate"]))
        influence.append(dict(cell=key[0], arm=key[1], comparator=baseline, precision_pass=passed, top_three=top))
    dump(out / "tail-influence.json", influence)
    retained = {}
    trace_count = 0
    for mode in MODES:
        directory = Path(completed[mode])
        assert len(list(directory.glob("*-traces.jsonl.gz"))) == 12
        for cell in cells:
            key = (cell["id"], cell["arm"], mode)
            count = 0
            with gzip.open(directory / f"{key[0]}-{key[1]}-traces.jsonl.gz", "rt") as stream:
                for rep, line in enumerate(stream):
                    record = json.loads(line, parse_constant=reject_constant)
                    check_trace(record, rep, int(cell["N"]), int(cell["d"]), int(cell["T"]))
                    full_key = (*key, rep)
                    assert full_key in records
                    if full_key in selected:
                        retained[full_key] = record
                    count += 1
            assert count == replicates
            trace_count += count
    assert set(retained) == selected and trace_count == 36 * replicates
    localization = []
    for contrast in influence:
        key = (contrast["cell"], contrast["arm"])
        baseline = contrast["comparator"]
        cell = cell_map[key]
        for row in contrast["top_three"]:
            rep = row["replicate"]
            localization.append(dict(cell=key[0], arm=key[1], baseline=baseline, **row,
                                     **localize(retained[(*key, baseline, rep)], retained[(*key, "tf32", rep)],
                                                int(cell["T"]), int(cell["N"]))))
    dump(out / "path-localization.json", localization)
    decision = dict(passed_contrasts=sum(x["equivalence"] == "TRUE" for x in precision), total_contrasts=24,
                    primary_precision_criterion="All 24 simultaneous approximate 99% paired intervals strictly inside [-.02,.02], largest centered squared share <=.2",
                    failed_contrasts=[x for x in precision if x["equivalence"] != "TRUE"],
                    tail_veto_contrasts=sum(x["tail_veto"] == "TRUE" for x in precision),
                    interval_violations=sum(float(x["lower"]) <= -.02 or float(x["upper"]) >= .02 for x in precision),
                    localized_selected_pairs=len(localization),
                    changed_decision_pairs=sum(x["first_changed_decision"] is not None for x in localization),
                    selected_for_explanation_only=True,
                    strict_tf32_path_parity="pass" if mode_results["tf32"]["old_pass"] == 18 else "veto_retained",
                    phase18_and_phase19_vetoes_retained=True,
                    heuristic_dominance=dict(observed_losses=sum(x["observed_quality_promotion_veto"] == "TRUE" for x in heuristic),
                                             comparisons=27, verdict="descriptive_only_no_quality_promotion"),
                    primary_scope="Fresh d2/N64/T5 streams, actual recorded compiled executions only",
                    scientific_or_default_promotion=False)
    dump(out / "decision.json", decision)
    dump(out / "verification.json", dict(status="pass", source_hashes=source_count, output_hashes=output_count,
                                         fixture_hashes=len(fixture_hashes), records=36 * replicates, neutral_valid_traces=trace_count,
                                         mode_results=mode_results, oracle_checks=oracle,
                                         fresh_seed_ranges=[[x, x+replicates-1] for x in fresh_starts],
                                         phase18_particle_seed_overlap=False,
                                         independent_ancestor_reconstruction="not checked; uniforms are not in phase20 traces"))
    return decision


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU reporting must explicitly hide GPU"
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--replicates", type=int, default=1024)
    parser.add_argument("--particle-seed-offset", type=int, default=10000000)
    args = parser.parse_args()
    assert (args.replicates, args.particle_seed_offset) in ((1024, 10000000), (4096, 20000000))
    repo = args.repository.resolve()
    root = repo / args.root
    out = root / "review-01"
    ledger_path = root / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["cpu_reserved_seconds"] == ledger["gpu_reserved_seconds"] == 0
    assert "post_run_review_seconds" not in ledger
    out.mkdir()
    status = "failed"
    try:
        decision = review(repo, root, out, args.replicates, args.particle_seed_offset)
        status = "pass"
        print(json.dumps({k: decision[k] for k in ("passed_contrasts", "total_contrasts", "tail_veto_contrasts",
                                                   "interval_violations", "changed_decision_pairs")}))
    finally:
        elapsed = time.monotonic() - started
        preflight = json.loads((root / "preflight.json").read_text())["wall_seconds"]
        manifest = dict(status=status, command=[sys.executable, *sys.argv],
                        CPU_only=True, gpu_intentionally_hidden=True, python=sys.version,
                        source=__file__, source_sha256=sha(__file__), plan=args.plan, wall_seconds=elapsed,
                        input_completed_sha256=sha(root / "completed.json"),
                        outputs={x.name: sha(x) for x in out.iterdir() if x.is_file()})
        dump(out / "manifest.json", manifest)
        charged = elapsed + preflight + 2.
        ledger["cpu_charged_seconds"] += charged
        ledger["remaining_campaign_cpu_seconds"] -= charged
        ledger["post_run_review_seconds"] = elapsed
        ledger["preflight_cpu_seconds"] = preflight + 2.
        dump(ledger_path, ledger)
        assert ledger["cpu_charged_seconds"] <= ledger["phase_cpu_cap_seconds"], "Phase reporting budget exceeded"


if __name__ == "__main__":
    main()
