"""Saved fixed-count interval diagnostic; no sampler or runtime estimator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def close(x, y, message):
    require(math.isfinite(x) and math.isfinite(y) and
            math.isclose(x, y, rel_tol=2e-10, abs_tol=1e-12), message)

def classify(estimate, truth, half_width):
    require(all(math.isfinite(x) for x in (estimate, truth, half_width)) and
            half_width >= 0, "invalid interval values")
    return "below" if estimate + half_width < truth else (
        "above" if estimate - half_width > truth else "covered")

def check_interval(interval, law, estimate):
    close(interval["estimate"], estimate, "inconsistent pooled mean")
    close(interval["truth"], law["posterior_truth"], "inconsistent truth")
    if not interval["available"]:
        require(interval["covered"] is None, "unavailable interval counted as covered")
        return None
    require(interval["mcse"] > 0 and interval["multiplier"] > 0 and
            law["fixed_variance"] > 0, "nonpositive interval scale")
    half = interval["multiplier"] * interval["mcse"]
    close(interval["width"], 2 * half, "incorrect interval width")
    close(interval["estimated_to_exact_variance"],
          interval["mcse"] ** 2 / law["fixed_variance"], "incorrect variance ratio")
    direction = classify(estimate, law["posterior_truth"], half)
    require(interval["direction"] == direction and
            interval["covered"] == (direction == "covered"), "incorrect coverage decision")
    require(interval["transient_covered"] ==
            (classify(estimate, law["fixed_expectation"], half) == "covered"),
            "incorrect transient coverage decision")
    return direction

def same_multiplier_counterfactual(interval, law):
    if not interval["available"]:
        return None
    # A diagnostic with an oracle variance is not a realizable estimator and
    # does not imply a Student distribution for nonlinear chain means.
    return classify(interval["estimate"], law["posterior_truth"],
                    interval["multiplier"] * math.sqrt(law["fixed_variance"]))

def statistics_summary(values):
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)

def audit(evidence):
    provenance = json.loads((evidence / "provenance.json").read_text())
    for name in ("records.jsonl.gz", "manifest.json", "summary.json"):
        require(sha(evidence/name) == provenance["files"][name], "changed compact input: "+name)
    manifest = json.loads((evidence/"manifest.json").read_text())
    design = manifest["design"]
    with gzip.open(evidence/"records.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    expected = {(c, r) for c in design["cases"] for r in range(design["replications"])}
    require(len(records) == len(expected) and
            {(r["case"], r["replication"]) for r in records} == expected,
            "missing or duplicate study records")
    expected_profiles = {(p["chains"],p["draws"]) for p in design["profiles"]}
    groups = defaultdict(lambda: dict(units=0, arms={}, student_available=0,
        student_unavailable=0, student_directions=Counter(), exact_variance_directions=Counter(),
        original_misses_rescued=0, original_coverages_lost=0,
        variance_ratio_by_direction=defaultdict(list), paired_outcomes=[]))
    raw_existing = 0
    for record in records:
        raw_existing += Path(record["receipt"]["path"]).is_file()
        profiles = record["profiles"]
        require(len(profiles)==len(expected_profiles) and
                {(p["chains"],p["draws"]) for p in profiles}==expected_profiles,
                "missing or duplicate profile")
        for profile in profiles:
            require(set(profile["intervals"])==set(design["arms"]), "arm inventory mismatch")
            require(len(profile["laws"])==len(profile["means"])==2, "quantity inventory mismatch")
            for i,law in enumerate(profile["laws"]):
                key=f"{record['case']}:m{profile['chains']}:n{profile['draws']}:{law['name']}"
                group=groups[key];group["units"]+=1
                for name, intervals in profile["intervals"].items():
                    require(len(intervals)==2, "missing arm quantity")
                    interval=intervals[i]
                    direction=check_interval(interval,law,profile["means"][i])
                    group["arms"].setdefault(name,Counter())[direction or "unavailable"]+=1
                student=profile["intervals"]["replicated_student"][i]
                replacement=same_multiplier_counterfactual(student,law)
                if replacement is None:
                    group["student_unavailable"]+=1
                    continue
                group["student_available"]+=1
                direction=student["direction"]
                group["student_directions"][direction]+=1
                group["exact_variance_directions"][replacement]+=1
                group["original_misses_rescued"]+=direction!="covered" and replacement=="covered"
                group["original_coverages_lost"]+=direction=="covered" and replacement!="covered"
                group["variance_ratio_by_direction"][direction].append(student["estimated_to_exact_variance"])
                group["paired_outcomes"].append(dict(replication=record["replication"],
                    original=direction,exact_variance_same_multiplier=replacement))
    for group in groups.values():
        require(group["units"]==design["replications"], "incomplete group denominator")
        group["variance_ratio_by_direction"]={k:statistics_summary(v)
                                               for k,v in group["variance_ratio_by_direction"].items()}
    return dict(status="PASS",records=len(records),profiles=len(records)*len(expected_profiles),
        groups=dict(groups),original_screen_family=design["screen_count"],
        original_screen_results_changed=False, raw_receipts_present=raw_existing,
        raw_receipts_total=len(records),raw_receipts_replayed=False,
        variance_counterfactual_role="explanatory only; same Student multiplier; no Student-law claim",
        new_sampling=False,ranking_supported=False,coverage_established=False,
        default_promoted=False,source_provenance_sha256=sha(evidence/"provenance.json"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    manifest=dict(command=[sys.executable,*sys.argv],started_unix=time.time(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        driver_sha256=sha(__file__),no_framework_or_gpu_import=True,
        plan="docs/plans/bayesfilter-modern-hmc-c4-saved-precision-audit-2026-10-28.md")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    try:
        result=audit(args.evidence)
    except Exception:
        result=dict(status="ERROR",exception=traceback.format_exc());traceback.print_exc()
    result["inclusive_elapsed_seconds"]=time.monotonic()-start
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result.get(k) for k in ("status","records","profiles","inclusive_elapsed_seconds")}))
    return 0 if result["status"]=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())
