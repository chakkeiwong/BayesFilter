"""Aggregate frozen per-seed LGSSM moment-teacher claim shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


LABELS = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")
EXPECTED_SEEDS = tuple(range(81910, 81916))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows, kalman):
    error_rows = [
        [row["objective"] - kalman["value"]]
        + [row["score"][index] - kalman["score"][index] for index in range(5)]
        for row in rows
    ]
    errors = {}
    for index, label in enumerate(LABELS):
        values = [row[index] for row in error_rows]
        mean = statistics.mean(values)
        mcse = statistics.stdev(values) / math.sqrt(len(values))
        errors[label] = {
            "mean_error": mean,
            "mcse": mcse,
            "absolute_mean_over_mcse": abs(mean) / mcse if mcse else float("inf"),
            "signs": [
                "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
                for value in values
            ],
        }
    return {
        "seed_count": len(rows),
        "all_valid": all(row["finite_valid"] for row in rows),
        "errors_to_kalman": errors,
        "maximum_mean_residual": max(row["maximum_mean_residual"] for row in rows),
        "maximum_covariance_residual": max(
            row["maximum_covariance_residual"] for row in rows
        ),
        "paired_difference_to_empirical_contract_e": {
            "value_mean": statistics.mean(
                row["paired_objective_difference"] for row in rows
            ),
            "value_mcse": statistics.stdev(
                row["paired_objective_difference"] for row in rows
            )
            / math.sqrt(len(rows)),
            "score_mean": [
                statistics.mean(row["paired_score_difference"][index] for row in rows)
                for index in range(5)
            ],
            "score_mcse": [
                statistics.stdev(
                    row["paired_score_difference"][index] for row in rows
                )
                / math.sqrt(len(rows))
                for index in range(5)
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in arguments.shards]
    if any(payload.get("status") != "pass" for payload in payloads):
        raise ValueError("all claim shards must pass their hard gates")
    rows = [payload["claim_rows"][0] for payload in payloads]
    seeds = tuple(row["seed"] for row in rows)
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"claim shards must have ordered frozen seeds {EXPECTED_SEEDS}")
    if any(len(payload["claim_rows"]) != 1 for payload in payloads):
        raise ValueError("each shard must contain exactly one claim seed")
    kalman = payloads[0]["kalman"]
    if any(payload["kalman"] != kalman for payload in payloads[1:]):
        raise ValueError("claim shards do not share one exact Kalman comparator")
    tuning_ids = {
        payload["selected_tuning_artifact"]["artifact_id"] for payload in payloads
    }
    if len(tuning_ids) != 1:
        raise ValueError("claim shards do not share one selected tuning artifact")
    identities = [row["route_identity"]["identity_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("per-seed route identities must be distinct")
    aggregate = {
        "schema": "bayesfilter.zhao_cui_moment_teacher_lgssm_claim_aggregate.v1",
        "status": "pass",
        "horizon": payloads[0]["horizon"],
        "particle_count": payloads[0]["particle_count"],
        "claim_seeds": list(seeds),
        "kalman": kalman,
        "selected_tuning_artifact_id": next(iter(tuning_ids)),
        "claim_summary": _summary(rows, kalman),
        "shards": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": row["seed"],
                "route_identity_sha256": row["route_identity"]["identity_sha256"],
            }
            for path, row in zip(arguments.shards, rows, strict=True)
        ],
        "nonclaims": [
            "six seeds do not support a method ranking",
            "not nonlinear validity",
            "not HMC or posterior readiness",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
