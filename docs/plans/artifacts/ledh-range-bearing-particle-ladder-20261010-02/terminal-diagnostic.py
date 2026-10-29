"""Post-run inspection only; cannot affect tuning, filtering, or admission."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BASE = REPO / "docs/plans/artifacts/ledh-range-bearing-covariance-20261009-07"
report = json.loads((ROOT / "ladder-report.json").read_text())
execution = json.loads((ROOT / "execution.json").read_text())
assert report["complete"] and execution["complete"]
summary = {"role": "post-run diagnostic only", "checks": {}, "cells": [], "counts": {}}
for n in (2016, 4032):
    directory = ROOT / f"N{n}"
    rows = [json.loads(s) for s in (directory / "compare.jsonl").read_text().splitlines()]
    assert len(rows) == 192
    assert len({(r["arm"], r["data_seed"], r["point"], r["seed"]) for r in rows}) == 192
    ledger = json.loads((directory / "tuning.json").read_text())
    manifest = json.loads((directory / "compare-manifest.json").read_text())
    assert ledger["valid"] and ledger["scope"] == manifest["scope"]
    assert manifest["scope"]["particles"] == n and manifest["scope"]["chunk"] == 2016
    assert manifest["memory_policy"]["all_physical_devices_memory_growth"]
    for relative, expected in json.loads((directory / "reuse.json").read_text())["sha256"].items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((BASE / relative).read_bytes()).hexdigest() == expected
    for r in rows:
        if r["arm"] in ("new", "old"):
            assert r["valid"] and all(math.isfinite(x) for x in [r["value"], *r["score"]])
    summary["counts"][str(n)] = {
        "valid_by_arm": dict(collections.Counter(r["arm"] for r in rows if r["valid"])),
        "invalid_rows": [{k: r.get(k) for k in ("arm", "data_seed", "point", "seed", "steps_completed")}
                         for r in rows if not r["valid"]],
        "selected": ledger["selected"],
        "peak_allocator_MiB": manifest["allocator"]["peak"] / 2**20,
        "median_new_seconds": statistics.median(r["wall_seconds"] for r in rows if r["arm"] == "new"),
    }

for cell in report["cells"]:
    baseline = cell["counts"]["1008"]["arms"]["new"]
    item = {"data_seed": cell["data_seed"], "point": cell["point"], "counts": {}}
    for n, count in cell["counts"].items():
        row = count["arms"]["new"]
        assert row["valid"]
        assert math.isclose(row["score_rmse"]**2,
                            row["score_centroid_error"]**2 + 7/8 * row["score_spread"]**2,
                            rel_tol=1e-10, abs_tol=1e-10)
        item["counts"][n] = dict(row,
            score_variance_ratio=(row["score_spread"] / baseline["score_spread"])**2,
            score_error_ratio=row["score_error_mean"] / baseline["score_error_mean"],
            loglik_sd_ratio=row["loglik_sd"] / baseline["loglik_sd"],
            contrast=count["new_vs_N1008"],
            heuristic_losses=count["heuristic_losses_descriptive"])
    summary["cells"].append(item)

summary["checks"] = dict(unique_complete_rows=True, primary_finite_valid=True,
    tuning_scope_exact_match=True, reused_data_and_references_hash_match=True,
    memory_growth_verified=True, rmse_bias_variance_identity=True)
summary["worker_seconds"] = execution["worker_seconds"]
(ROOT / "terminal-diagnostic.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"checks": summary["checks"], "counts": {
    n: {k: x[k] for k in ("valid_by_arm", "invalid_rows", "peak_allocator_MiB", "median_new_seconds")}
    for n, x in summary["counts"].items()}, "cells": [
    {"data": c["data_seed"], "point": c["point"], "N4032": {
        k: c["counts"]["4032"][k] for k in
        ("score_variance_ratio", "score_error_ratio", "loglik_sd_ratio", "loglik_mean_offset")}}
    for c in summary["cells"]], "worker_seconds": summary["worker_seconds"]}, indent=2))
