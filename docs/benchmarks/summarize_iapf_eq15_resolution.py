"""Post-run reference reporting; no method selection or production consumers."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plans/artifacts/iapf-r-equation15-resolution-20260921-01"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    attempts = [json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    assert all(a["exit_code"] == 0 for a in attempts)
    assert not (OUT/"summary.json").exists()
    frozen = OUT / "source-v1"
    hashes = json.loads((frozen/"sha256.json").read_text())
    for name, checksum in hashes.items():
        assert sha(frozen/name) == checksum
    provenance = {}
    for folder in (OUT/"runs").iterdir():
        for f in folder.rglob("*"):
            if f.is_file():
                provenance[str(f.relative_to(OUT))] = sha(f)
    fixtures = [r for p in (OUT/"runs").glob("fixtures*/results/fixtures.csv") for r in rows(p)]
    assert len(fixtures) == 72
    assert len({(r["data_seed"],r["time"],r["parameterization"],r["initialization"])
        for r in fixtures}) == 72
    failures = rows(OUT/"analysis-v1/fit-failures.csv")
    assert len(failures) == 32
    probe_records, probe_prefixes, data_hashes = [], [], defaultdict(set)
    for attempt in attempts:
        if attempt.get("mode") != "probe":
            continue
        folder = Path(attempt["output"])
        data_hashes[attempt["dimension"]].add(sha(folder/"observations.csv"))
        if (folder/"failure.csv").exists():
            assert not (folder/"replicates.csv").exists()
            continue
        rr = rows(folder/"replicates.csv")
        pp = rows(folder/"prefixes.csv")
        assert len(rr) == 1 and rr[0]["status"] == "complete" and rr[0]["tail_pass"] == "TRUE"
        assert len(pp) == 100 and {int(r["time"]) for r in pp} == set(range(1,101))
        for row in pp:
            row["dimension"] = str(attempt["dimension"])
        if rr[0]["method"] == "qr":
            assert len(rows(folder/"tails.csv")) == 100
            assert len(rows(folder/"fits.csv")) == (int(rr[0]["iterations"])-1)*100
        probe_records += rr
        probe_prefixes += pp
    assert len(probe_records) == 16 and all(len(v) == 1 for v in data_hashes.values())
    assert {(r["dimension"], r["method"], r["replication"]) for r in probe_records} == {
        (str(d), m, str(rep)) for d in (5,20) for m in ("qr","bpf","fully_adapted","sis")
        for rep in (101,102)}
    conditional = []
    for d in ("5","20"):
        for situation in ("ordinary","large_innovation"):
            by_method = defaultdict(list)
            for r in probe_prefixes:
                if r["dimension"] == d and r["situation"] == situation:
                    by_method[r["method"]].append(math.expm1(float(r["log_prefix_error"]))**2)
            means = {m:statistics.mean(v) for m,v in by_method.items()}
            conditional.append(dict(dimension=int(d),situation=situation,mean_prefix_mse=means,
                observed_qr_heuristic_veto=any(means["qr"] > means[m]
                    for m in ("bpf","fully_adapted","sis")),
                inference="two replicas: explanatory screen only; no ranking"))
    timing = []
    for path in sorted((OUT/"runs").glob("timing*/results/replicates.csv")):
        rr = rows(path)
        assert len(rr) == 8 and all(r["status"] == "complete" for r in rr)
        for r in rr:
            assert float(r["algorithm_seconds"]) >= 0 and float(r["diagnostic_io_seconds"]) >= 0
            assert abs(float(r["wall_seconds"])-float(r["algorithm_seconds"])-
                float(r["diagnostic_io_seconds"])) < 1e-8
        for method in ("qr","bpf","fully_adapted","sis"):
            same = [r for r in rr if r["method"] == method]
            timing.append(dict(dimension=int(same[0]["dimension"]),method=method,replicas=2,
                mean_algorithm_seconds=statistics.mean(float(r["algorithm_seconds"]) for r in same),
                mean_extra_diagnostic_io_seconds=statistics.mean(float(r["diagnostic_io_seconds"]) for r in same),
                inference="sequential measurements; neither matched-accuracy efficiency nor a ranking"))
    assert len(timing) == 8
    selection = json.loads((OUT/"selection.json").read_text())
    assert selection["selected"] is None
    fixture_counts = [dict(dimension=d,complete=sum(r["dimension"]==str(d) and r["status"]=="complete"
        for r in fixtures),failed=sum(r["dimension"]==str(d) and r["status"]=="candidate_failed"
        for r in fixtures)) for d in (5,20,80)]
    summary = dict(status="completed_negative_eq15_result",completed_utc=datetime.now(timezone.utc).isoformat(),
        plan="docs/plans/iapf-r-equation15-resolution-2026-09-21.md",
        engineering=dict(worker_or_harness_errors=0,source_snapshot_hashes_checked=True,
            completed_comparator_probe_records=16,completed_sequential_timing_records=16,
            solver_tests="PASS",independent_gaussian_integral_checks=3),
        fixtures=dict(records=72,counts=fixture_counts,
            high_KL_converged_examples=[r for r in fixtures if r["status"]=="complete" and
                float(r["gaussian_component_KL"])>5]),
        full_filter=dict(initial_attempts=16,bounded_retries=16,successful_eq15_filters=0,
            failed_by_reason=dict(Counter(r["reason"] for r in failures)),
            failed_by_arm={arm:sum(r["parameterization"]+"_"+r["initialization"]==arm for r in failures)
                for arm in ("joint_qr","profiled_qr","joint_moments","profiled_moments")},
            untouched_validation="not triggered: no candidate completed the required probes"),
        conditional_comparator_diagnostics=conditional,timing=timing,
        prefix_diagnostic="analysis-v1/prefix-concentration.json",selection=selection,
        inference=dict(hard_veto="All new local Eq15 solver/start variants fail full-filter validity",
            statistically_supported_ranking="None established by this amendment",
            descriptive_only="Fit residual, KL, rare-error concentration, probe loss and runtime",
            default_readiness="No default change; QR remains an explicitly different reference",
            next_evidence="Recover author numerical settings, or specify and test a different bounded reconstruction",
            nonclaims="No failure theorem for iAPF; no author-code identity; no LEDH/KDM/HMC conclusion"),
        worker_seconds=sum(a["wall_seconds"] for a in attempts),attempts=len(attempts),
        source_snapshot="source-v1/sha256.json",provenance=provenance,
        analysis_source_sha256=sha(Path(__file__)))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("status","engineering","full_filter","worker_seconds","attempts")},indent=2))


if __name__ == "__main__":
    main()
