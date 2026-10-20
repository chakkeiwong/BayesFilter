"""Post-run evidence and discrete-path inspection; diagnostic only."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def max_cloud_difference(x, y):
    return max(abs(a-b) for u,v in zip(x,y) for a,b in zip(u,v))


def main():
    start = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    root = repo / "docs/plans/artifacts/iapf-tf32-revalidation-20260923-01"
    completed = json.loads((root / "completed.json").read_text())
    for key in ("fp64", "fp32", "tf32", "summary"):
        if key not in completed:
            raise RuntimeError(f"Required result missing: {key}")
    out = root / "review-01"
    out.mkdir()
    source_count = output_count = 0
    for path in sorted(root.glob("attempt*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["status"] not in ("complete", "failed"):
            raise RuntimeError("Unsettled attempt")
        source = Path(manifest["source"])
        for name, digest in manifest["source_sha256"].items():
            assert sha(source / name) == digest, (path, name)
            source_count += 1
        for name, digest in manifest["output_sha256"].items():
            assert sha(path.parent / name) == digest, (path, name)
            output_count += 1
    fixture = Path(completed["fixtures"])
    for name, digest in completed["fixture_hashes"].items():
        assert sha(fixture / name) == digest, name
    cells = list(csv.DictReader((fixture / "cells.csv").open()))
    assert len(cells) == 44
    oracle = list(csv.DictReader((fixture / "oracle-checks.csv").open()))
    assert len(oracle) == 11 and all(float(x["error"]) < 1e-9 for x in oracle)
    center = json.loads((Path(completed["center-probe"]) / "center-probe.json").read_text())
    assert center["status"] == "pass"
    assert len(center["records"]) == 18
    assert all(x["independent_reference_pass"] for x in center["records"])
    rows = []
    for baseline in ("fp64", "fp32"):
        for cell in cells:
            filename = f"{cell['id']}-{cell['arm']}-traces.jsonl"
            lhs = Path(completed[baseline]) / filename
            rhs = Path(completed["tf32"]) / filename
            lines1, lines2 = lhs.read_text().splitlines(), rhs.read_text().splitlines()
            assert len(lines1) == len(lines2) == 8
            pairs = list(zip(lines1, lines2))
            assert len(pairs) == 8
            for line1, line2 in pairs:
                a,b = json.loads(line1),json.loads(line2)
                assert a["replicate"] == b["replicate"] and a["neutral"] and b["neutral"]
                event = None
                pre_error = 0.
                post_error = 0.
                if a["ancestors"][0] != b["ancestors"][0]:
                    event = dict(time=-1, operation="initial_ancestor")
                for t in range(int(cell["T"])):
                    if event is None and a["mixture_choices"][t] != b["mixture_choices"][t]:
                        event = dict(time=t, operation="gaussian_mixture")
                    error = max_cloud_difference(a["cloud"][t],b["cloud"][t])
                    if event is None:
                        pre_error = max(pre_error,error)
                    else:
                        post_error = max(post_error,error)
                    if event is None and a["ancestors"][t+1] != b["ancestors"][t+1]:
                        event = dict(time=t, operation="ancestor_after_weighting")
                rows.append(dict(cell=cell["id"],arm=cell["arm"],baseline=baseline,
                    replicate=a["replicate"],first_changed_decision=event,
                    decision_can_affect_returned_likelihood=event is not None and not (
                        event["operation"] == "ancestor_after_weighting" and event["time"] == int(cell["T"])-1),
                    max_cloud_error_before_changed_decision=pre_error,
                    max_cloud_error_after_changed_decision=post_error,
                    max_fixed_label_score_difference=max(abs(x-y) for x,y in zip(a["score"],b["score"]))))
    dump(out / "path-localization.json", rows)
    mode_results = {}
    for mode in ("fp64", "fp32", "tf32"):
        result = json.loads((Path(completed[mode]) / "result.json").read_text())
        assert result["status"] == "complete"
        assert result["memory_policy"]["all_physical_devices_memory_growth"]
        assert result["tf32_verified"] == (mode == "tf32")
        assert all("GPU" in c["device"] for c in result["compilation"])
        assert result["distribution_records"] == 22528
        assert result["consumer"] == "bayesfilter.score_study.fitted_twist_tf.make_fitted_twist_kernel"
        mode_results[mode] = dict(old_checks=len(result["old_fixture_checks"]),
            old_pass=sum(x["passed"] for x in result["old_fixture_checks"]),
            old_failed=[x for x in result["old_fixture_checks"] if not x["passed"]],
            compilations=len(result["compilation"]), allocator=result["allocator"],
            tensorflow=result["tensorflow"], memory_policy=result["memory_policy"])
    precision = list(csv.DictReader((Path(completed["summary"])/"precision-comparisons.csv").open()))
    heuristic = list(csv.DictReader((Path(completed["summary"])/"heuristic-comparisons.csv").open()))
    assert len(precision) == 88 and len(heuristic) == 99 and len(rows) == 704
    assert mode_results["fp64"]["old_pass"] == mode_results["fp32"]["old_pass"] == 18
    samples = {}
    for sample in csv.DictReader((Path(completed["summary"]) / "records.csv").open()):
        key = (sample["cell"], sample["arm"], sample["mode"], int(sample["replicate"]))
        assert key not in samples
        samples[key] = sample
    assert len(samples) == 67584
    influence = []
    for contrast in precision:
        key = (contrast["cell"], contrast["arm"])
        difference = float(contrast["difference"])
        pairs = []
        for rep in range(512):
            candidate = samples[(*key, "tf32", rep)]
            baseline = samples[(*key, contrast["comparator"], rep)]
            assert candidate["seed"] == baseline["seed"]
            delta = float(candidate["normalized_likelihood"]) - float(baseline["normalized_likelihood"])
            pairs.append(dict(replicate=rep, seed=int(candidate["seed"]), difference=delta,
                candidate_normalized_likelihood=float(candidate["normalized_likelihood"]),
                baseline_normalized_likelihood=float(baseline["normalized_likelihood"]),
                centered_square=(delta-difference)**2))
        total = math.fsum(x["centered_square"] for x in pairs)
        top = sorted(pairs, key=lambda x:x["centered_square"], reverse=True)[:3]
        for record in top:
            record["squared_share"] = record["centered_square"] / total if total else 0.
        assert math.isclose(top[0]["squared_share"], float(contrast["largest_centered_squared_share"]),
            rel_tol=1e-6, abs_tol=1e-8)
        influence.append(dict(cell=key[0], arm=key[1], comparator=contrast["comparator"],
            precision_pass=contrast["equivalence"] == "TRUE", top_three=top))
    dump(out / "tail-influence.json", influence)
    loader = json.loads((root / "loader-regression.json").read_text())
    assert loader["status"] == "pass" and loader["FP64_exact_binary32_recovered"] and loader["FP32_loader_identical"]
    fp64_manifest = json.loads((Path(completed["fp64"]).parent / "manifest.json").read_text())
    assert fp64_manifest["source_sha256"]["docs/benchmarks/diagnose_iapf_tf32_revalidation.py"] == loader["source_sha256"]
    verification = dict(status="pass", source_hashes=source_count, output_hashes=output_count,
        fixture_hashes=len(completed["fixture_hashes"]), distribution_records=67584,
        checked_trace_pairs=len(rows), trace_neutral_records=1056, mode_results=mode_results,
        common_input_loader_regression=loader, independent_center_probe=dict(status=center["status"],
            cases=len(center["records"]), max_error=max(max(x["independent_R_max_error_by_time"])
                for x in center["records"])), oracle_checks=oracle)
    dump(out / "verification.json", verification)
    decision = dict(primary_precision_criterion="all 88 simultaneous 99% paired intervals inside ±0.02 with tail share <=0.2",
        passed_contrasts=sum(x["equivalence"] == "TRUE" for x in precision), total_contrasts=len(precision),
        failed_contrasts=[x for x in precision if x["equivalence"] != "TRUE"],
        tail_veto_contrasts=sum(x["tail_veto"] == "TRUE" for x in precision),
        strict_tf32_path_parity="pass" if mode_results["tf32"]["old_pass"] == 18 else "veto_retained",
        heuristic_dominance=dict(verdict="quality_promotion_veto" if any(x["observed_quality_promotion_veto"] == "TRUE" for x in heuristic)
            else "descriptive_screen_pass", observed_losses=sum(x["observed_quality_promotion_veto"] == "TRUE" for x in heuristic),
            comparisons=len(heuristic), stress_guide_not_trained_candidate=True),
        changed_decision_trace_pairs=sum(x["first_changed_decision"] is not None for x in rows),
        scientific_or_default_promotion=False)
    dump(out / "decision.json", decision)
    manifest = dict(command="CUDA_VISIBLE_DEVICES=-1 python3 docs/benchmarks/review_iapf_tf32_revalidation.py",
        CPU_only=True, gpu_intentionally_hidden=True, role="post-run standard-library inspection",
        source=__file__, source_sha256=sha(Path(__file__)), input_completed_sha256=sha(root/"completed.json"),
        plan="docs/plans/iapf-tf32-revalidation-2026-09-23.md", wall_seconds=time.monotonic()-start,
        outputs={str(x.relative_to(out)):sha(x) for x in out.iterdir() if x.is_file()})
    dump(out / "manifest.json", manifest)
    ledger_path = root / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger["cpu_reserved_seconds"] or ledger["gpu_reserved_seconds"]:
        raise RuntimeError("Active reservations remain")
    seconds = manifest["wall_seconds"] + loader["wall_seconds"] + 2.0
    ledger["cpu_charged_seconds"] += seconds
    ledger["remaining_campaign_cpu_seconds"] -= seconds
    ledger["post_run_review_seconds"] = manifest["wall_seconds"]
    ledger["loader_regression_cpu_seconds"] = loader["wall_seconds"]
    ledger["preflight_cpu_conservative_seconds"] = 2.
    dump(ledger_path, ledger)
    print(json.dumps({k:decision[k] for k in ("passed_contrasts", "total_contrasts", "tail_veto_contrasts",
        "strict_tf32_path_parity", "changed_decision_trace_pairs")}))


if __name__ == "__main__":
    main()
