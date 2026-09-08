"""Aggregate Phase8D frozen-control replication records.

The reader is diagnostic/reporting code. It deliberately reports ranges and
sign counts rather than ranking stochastic methods from three paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _load(path: Path) -> dict:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def aggregate(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        result = _load(path)
        active_start = int(result.get("proposal_active_start_time", 1))
        candidate = next(
            row for row in result["records"]
            if row["label"] == "exact_likelihood_laplace_k1"
        )
        baselines = {
            row["label"]: row
            for row in result["records"]
            if row["label"] != "exact_likelihood_laplace_k1"
        }
        contrasts = {}
        candidate_valid = bool(candidate.get("all_checks_pass", False))
        if candidate_valid:
            if "ess_by_time" not in candidate:
                raise ValueError(f"valid candidate is missing ESS for {path}")
            for label, baseline in baselines.items():
                if "ess_by_time" not in baseline:
                    raise ValueError(f"baseline {label} is missing ESS for {path}")
                all_differences = [
                    float(a) - float(b)
                    for a, b in zip(candidate["ess_by_time"], baseline["ess_by_time"])
                ]
                active_differences = all_differences[active_start:]
                if not active_differences:
                    raise ValueError(f"no active transition contrasts for {path}")
                contrasts[label] = {
                    "initial_difference": all_differences[0],
                    "minimum_active_difference": min(active_differences),
                    "median_active_difference": statistics.median(active_differences),
                    "positive_active_count": sum(value > 0.0 for value in active_differences),
                    "tie_active_count": sum(value == 0.0 for value in active_differences),
                    "active_comparison_count": len(active_differences),
                    "proposal_active_start_time": active_start,
                }
        rows.append({
            "path": str(path),
            "fixture_sha256": result["sources"]["fixture"]["sha256"],
            "status": result["status"],
            "all_branch_validity": result["checks"]["all_branch_validity"],
            "candidate_branch_validity": candidate_valid,
            "candidate_failure_class": candidate.get("failure_class"),
            "candidate_error": candidate.get("error"),
            "candidate_minimum_ess": (float(candidate["minimum_ess"]) if candidate_valid else None),
            "candidate_time14_ess": (float(candidate["ess_by_time"][14]) if candidate_valid else None),
            "proposal_active_start_time": active_start,
            "heuristic_verdict": result["heuristic_dominance"]["exact_likelihood_laplace_k1"]["verdict"],
            "contrasts": contrasts,
        })
    return {
        "schema_version": "c2_phase8d_replication_aggregate_v1",
        "path_count": len(rows),
        "rows": rows,
        "validity_pass_count": sum(bool(row["all_branch_validity"]) for row in rows),
        "interpretation": "descriptive replication only; no statistical ranking or default claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    paths = [Path(value).resolve() for value in args.paths]
    result = aggregate(paths)
    output = Path(args.output)
    if not output.is_absolute():
        output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "path_count": len(paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
