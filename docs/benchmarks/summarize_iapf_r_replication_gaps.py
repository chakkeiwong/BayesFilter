"""Post-run diagnostic analysis only; no runtime or candidate selection role."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as stats
import time

import run_iapf_r_plausible_choices as reporting
from run_iapf_r_replication_gap_audit import ROOT, OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio_summary(rows):
    values = [float(r["ratio"]) for r in rows]
    assert len(values) >= 2 and all(math.isfinite(v) and v > 0 for v in values)
    ci = reporting.bootstrap(values, stats.mean)
    return dict(repeats=len(values), mean_ratio=stats.mean(values),
        sd_ratio=stats.stdev(values), mean_ci95=ci,
        sd_ci95=reporting.bootstrap(values, stats.stdev),
        mean_interval_contains_one=ci[0] <= 1 <= ci[1])


def main():
    start = time.monotonic()
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert len(manifest["attempts"]) == 10
    assert manifest["status"] == "all planned workers finished; terminal analysis pending"
    destination = OUT / "terminal-analysis"
    destination.mkdir(exist_ok=False)
    reporting.OUT = OUT
    attempts = {a["name"]: a for a in manifest["attempts"]}
    for name, attempt in attempts.items():
        expected = 124 if name == "attempt04-validation-d40" else 0
        assert attempt["exit_code"] == expected, (name, attempt["exit_code"])
    for snapshot, field in (("source-snapshot", "source_hashes"),
                            ("repair-source-snapshot", "repair_source_hashes")):
        for name, digest in manifest[field].items():
            assert sha(OUT / snapshot / name) == digest, name
    assert sha(ROOT / ".localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf") == manifest["paper_sha256"]

    # A resource timeout is preserved; only complete replicate blocks are merged.
    original = OUT / "attempt04-validation-d40/results"
    repair = OUT / "attempt09-validation-d40-completion/results"
    merged = destination / "d40-complete" / "results"
    merged.mkdir(parents=True)
    for filename in ("observations.csv", "kalman.csv"):
        assert sha(original / filename) == sha(repair / filename), filename
        shutil.copy2(original / filename, merged / filename)
    merge_sources = {}
    for filename in ("replicates.csv", "prefixes.csv", "fits.csv", "tails.csv"):
        source_rows = reporting.read_csv(original / filename)
        repair_rows = reporting.read_csv(repair / filename)
        rows = [r for r in source_rows if 1801 <= int(r["replication"]) <= 1827]
        rows.extend(r for r in repair_rows if 1828 <= int(r["replication"]) <= 1832)
        assert rows and source_rows[0].keys() == repair_rows[0].keys()
        with (merged / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        merge_sources[filename] = {str(p.relative_to(ROOT)): sha(p)
            for p in (original / filename, repair / filename)}
    derived = dict(name="terminal-analysis/d40-complete", stage="validation", arm="delayed",
        dimension=40, data_seed=87000040, first=1801, repeats=32, baselines=True,
        exit_code=0, worker_seconds=sum(attempts[n]["worker_seconds"] for n in (
            "attempt04-validation-d40", "attempt09-validation-d40-completion")),
        derived_from_preserved_attempts=True, source_hashes=merge_sources)
    reporting.write_json(merged.parent / "provenance.json", derived)
    validations = []
    for attempt in (attempts["attempt02-validation-d20-a"],
                    attempts["attempt03-validation-d20-b"], derived):
        summary = reporting.inspect_attempt(attempt)
        assert summary["eligible"], summary
        reporting.validation_diagnostics(attempt, summary)
        validations.append(summary)

    confirmation_attempt = attempts["attempt10-d20-mean-confirmation"]
    confirmation = reporting.inspect_attempt(confirmation_attempt)
    assert confirmation["eligible"], confirmation
    confirm_rows = reporting.read_csv(OUT / confirmation_attempt["name"] / "results/replicates.csv")
    confirmation.update(ratio_summary(confirm_rows))
    initial_rows = [r for r in reporting.read_csv(OUT / "attempt02-validation-d20-a/results/replicates.csv")
                    if r["method"] == "iapf"]
    assert not ({r["seed"] for r in initial_rows} & {r["seed"] for r in confirm_rows})
    pooled = ratio_summary(initial_rows + confirm_rows)
    pooled["interpretation"] = "Exploratory pooled96; initial32 triggered a fixed independent64 batch."

    guide_groups = collections.defaultdict(list)
    for attempt in ("attempt05-fixed-guides", "attempt08-pre-doubling-guides"):
        rows = reporting.read_csv(OUT / attempt / "results/fixed-guides.csv")
        assert len(rows) == (136 if attempt == "attempt05-fixed-guides" else 64)
        assert len({(r["arm"], r["guide"], r["N"], r["replication"]) for r in rows}) == len(rows)
        for r in rows:
            guide_groups[(r["arm"], int(r["guide"]), int(r["N"]))].append(r)
        tails = reporting.read_csv(OUT / attempt / "results/tails.csv")
        assert len(tails) == 400 and all(r["passed"] == "TRUE" for r in tails)
    guides = []
    for (arm, guide, n), rows in guide_groups.items():
        summary = dict(arm=arm, guide=guide, particles=n, **ratio_summary(rows))
        if arm == "exact_future":
            assert max(abs(float(r["log_ratio"])) for r in rows) < 1e-8
        guides.append(summary)
    parity = reporting.read_csv(OUT / "attempt08-pre-doubling-guides/results/history-parity.csv")
    assert len(parity) == 4 and all(float(r["history_error"]) < 1e-10 and
                                  float(r["final_error"]) < 1e-10 for r in parity)
    fit_status = reporting.read_csv(OUT / "attempt07-fit-sensitivity-fixed-scale/results/fit-status.csv")
    assert len(fit_status) == 8 and all(r["status"] == "converged" for r in fit_status)
    fits = reporting.read_csv(OUT / "attempt07-fit-sensitivity-fixed-scale/results/fit-sensitivity.csv")
    assert len(fits) == 24
    pilot = reporting.read_csv(OUT / "attempt06-constrained-pilot/results/pilot-result.csv")
    pilot_tails = reporting.read_csv(OUT / "attempt06-constrained-pilot/results/tails.csv")
    assert len(pilot) == 1 and len(pilot_tails) == 100 and all(r["passed"] == "TRUE" for r in pilot_tails)
    summary = dict(validation=validations, confirmation=confirmation,
        exploratory_pooled_d20_a=pooled, conditional_fixed_guides=guides,
        history_parity=parity, corrected_fit_sensitivity=fits, constrained_pilot=pilot,
        original_sensitivity="Confounded by optimizer scaling; retained but not used for box-effect inference.",
        inference_scope="Pointwise conditional bootstrap intervals; no simultaneous or equal-cost ranking.",
        source_snapshot_hashes_verified=True,
        worker_seconds=manifest["worker_seconds"], remaining_worker_seconds=manifest["remaining_worker_seconds"])
    reporting.write_json(destination / "summary.json", summary)
    shutil.copy2(Path(__file__), destination / Path(__file__).name)
    check_log = Path("/tmp/iapf-gap-identities-after-repair.log")
    if check_log.exists():
        shutil.copy2(check_log, OUT / check_log.name)
    manifest["terminal_analysis_seconds"] = time.monotonic() - start
    manifest["terminal_analysis_source_sha256"] = sha(Path(__file__))
    manifest["terminal_summary"] = str((destination / "summary.json").relative_to(ROOT))
    manifest["status"] = "workers and structured analysis complete; written result review pending"
    reporting.write_json(OUT / "manifest.json", manifest)
    print(json.dumps(dict(status=manifest["status"], worker_seconds=manifest["worker_seconds"],
                         terminal_analysis_seconds=manifest["terminal_analysis_seconds"])))


if __name__ == "__main__":
    main()
