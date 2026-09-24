"""Post-run reporting for the independent R fitting-repair diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01"


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return None if value in (None, "NA", "") else float(value)


def main():
    out = ROOT / "report-v1"
    out.mkdir(exist_ok=False)
    attempts = []
    for p in sorted(ROOT.glob("attempt*/manifest.json")):
        d = json.loads(p.read_text())
        assert d["status"] != "running", p
        for rel, expected in d["source_sha256"].items():
            assert hashlib.sha256((p.parent / "source" / rel).read_bytes()).hexdigest() == expected, rel
        for rel, expected in d["output_sha256"].items():
            assert hashlib.sha256((p.parent / rel).read_bytes()).hexdigest() == expected, rel
        for rel, expected in d.get("input_sha256", {}).items():
            assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == expected, rel
        attempts.append(dict(attempt=p.parent.name, **d))
    warm = read(ROOT / "attempt02-warm/results/warm-start.csv")
    solver = read(ROOT / "attempt03-solver/results/solver-controls.csv")
    shapes = read(ROOT / "attempt03-solver/results/heldout-shape.csv")
    analytic = read(ROOT / "attempt04-analytic/results/analytic-controls.csv")
    learning = []
    bridge_history = []
    authoritative_bridge = {}
    for a in attempts:
        directory = ROOT / a["attempt"] / "results"
        if (directory / "learning.csv").exists():
            learning.extend(dict(attempt=a["attempt"], **r) for r in read(directory / "learning.csv"))
        if (directory / "bridge.csv").exists():
            for r in read(directory / "bridge.csv"):
                entry = dict(attempt=a["attempt"], **r)
                bridge_history.append(entry)
                authoritative_bridge[(r["data_seed"], r["replication"])] = entry
    bridge = list(authoritative_bridge.values())
    write(out / "learning.csv", learning)
    write(out / "bridge-all-attempts.csv", bridge_history)
    write(out / "bridge-authoritative.csv", bridge)
    write(out / "analytic-controls.csv", analytic)
    comparisons = []
    for candidate in (r for r in learning if r["method"] == "ridge_bounded1"):
        baseline_rows = [r for r in learning if r["method"] == "qr" and r["data_seed"] == candidate["data_seed"]]
        baseline_rows += [r for r in analytic if r["data_seed"] == candidate["data_seed"] and r["replication"] == "1"]
        for baseline in baseline_rows:
            error, base_error = number(candidate["terminal_error"]), number(baseline["terminal_error"])
            if error is None or base_error is None:
                verdict = "candidate_or_comparator_incomplete"
            elif baseline["method"] == "exact_full":
                verdict = "oracle_distance_only"
            elif abs(error) > abs(base_error):
                verdict = "observed_heuristic_underperformance_promotion_veto"
            else:
                verdict = "no_underperformance_in_this_record_not_superiority"
            comparisons.append(dict(dimension=candidate["dimension"], data_seed=candidate["data_seed"],
                method=candidate["method"], comparator=baseline["method"],
                absolute_terminal_error=abs(error) if error is not None else None,
                comparator_absolute_terminal_error=abs(base_error) if base_error is not None else None,
                candidate_final_particles=candidate["final_particles"],
                comparator_particles=baseline.get("final_particles", baseline.get("N")),
                verdict=verdict, interpretation="conditional descriptive sanity check; unmatched learning cost; no ranking"))
    write(out / "conditional-heuristics.csv", comparisons)
    shape_comparisons = []
    for row in (r for r in shapes if r["method"] == "ridge_bounded1"):
        for baseline in (r for r in shapes if r["case"] == row["case"] and r["region"] == row["region"]
                         and r["method"] in ("constant", "observation_only", "svd0", "moment_diagonal", "precision_diagonal")):
            shape_comparisons.append(dict(case=row["case"], region=row["region"], comparator=baseline["method"],
                candidate_RMS=float(row["shape_rms"]), comparator_RMS=float(baseline["shape_rms"]),
                verdict="observed_underperformance_promotion_veto" if float(row["shape_rms"]) > float(baseline["shape_rms"])
                else "no_observed_underperformance"))
    write(out / "conditional-shape-heuristics.csv", shape_comparisons)
    check_rows = []
    for a in attempts:
        path = ROOT / a["attempt"] / "results/checks.csv"
        if path.exists():
            check_rows.extend(dict(attempt=a["attempt"], **row) for row in read(path))
    assert all(r["pass"] == "TRUE" for r in check_rows)
    write(out / "checks.csv", check_rows)
    summary = dict(
        status="complete", attempts=len(attempts), statuses=dict(Counter(a["status"] for a in attempts)),
        worker_seconds=sum(a["wall_seconds"] for a in attempts),
        warm_replay_records=len(warm), warm_replay_guard_accepts=sum(r["status"] == "accepted" for r in warm),
        warm_shape_worsens=sum(float(r["final_shape"]) > float(r["initial_shape"]) + 1e-8 for r in warm),
        solver_records=len(solver),
        solver_accepts={m: sum(r["method"] == m and r["status"] == "accepted" for r in solver)
                        for m in sorted({r["method"] for r in solver})},
        fresh_learning_records=len(learning), fresh_learning_statuses=dict(Counter(r["status"] for r in learning)),
        bridge_records=len(bridge), bridge_statuses=dict(Counter(r["status"] for r in bridge)),
        preserved_bridge_attempts=len(bridge_history),
        bridge_terminal_error_range=[min(number(r["terminal_error"]) for r in bridge if r["status"] == "complete"),
                                     max(number(r["terminal_error"]) for r in bridge if r["status"] == "complete")],
        total_check_evaluations=len(check_rows),
        heuristic_dominance_verdict="promotion_veto_observed_conditional_underperformance",
        conditional_heuristic_veto_records=sum(r["verdict"].startswith("observed_") for r in comparisons),
        conditional_shape_veto_records=sum(r["verdict"].startswith("observed_") for r in shape_comparisons),
        statistical_ranking="not_supported", default_readiness=False,
        nonclaims=["original-author identity", "full paper replication", "statistical superiority", "matched-cost performance", "production promotion"],
        next_action="Freeze the explicit repaired reference; independent multi-seed validation across five dimensions, preserving paper-equation and extension distinctions.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
