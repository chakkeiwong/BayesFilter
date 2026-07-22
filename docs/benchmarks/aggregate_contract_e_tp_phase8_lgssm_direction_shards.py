#!/usr/bin/env python3
"""Aggregate five independently executed LGSSM direction shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    rows.sort(key=lambda row: row["direction_index"])
    if [row["direction_index"] for row in rows] != list(range(5)):
        raise ValueError("direction shards must cover exactly 0 through 4")
    identity = {
        (
            row["preparation"]["sha256"],
            row["feature_mode"],
            row.get("lookahead_steps"),
            row["time_steps"],
        )
        for row in rows
    }
    if len(identity) != 1:
        raise ValueError("direction shard identities differ")
    objective = rows[0]["objective"]
    kalman_value = rows[0]["kalman_value"]
    for row in rows[1:]:
        if not math.isclose(row["objective"], objective, rel_tol=0.0, abs_tol=2e-12):
            raise ValueError("direction shard objectives differ")
        if not math.isclose(row["kalman_value"], kalman_value, rel_tol=0.0, abs_tol=2e-12):
            raise ValueError("direction shard Kalman values differ")
    score = [row["score"] for row in rows]
    fd = [row["finite_difference"] for row in rows]
    oracle = [row["kalman_score"] for row in rows]
    relative = [abs(value - reference) / abs(reference) for value, reference in zip(score, oracle)]
    signs = [math.copysign(1.0, value) != math.copysign(1.0, reference) for value, reference in zip(score, oracle)]
    value_pass = abs(objective - kalman_value) <= 0.001
    score_pass = max(relative) <= 0.05 and not any(signs)
    same_scalar_pass = all(row["status"] == "PASS_SAME_SCALAR_DIRECTION" for row in rows)
    payload = {
        "schema": "bayesfilter.contract_e_tp.phase8_lgssm_direction_aggregate.v1",
        "status": "PASS_ENGINEERING" if same_scalar_pass and all(row["chart"]["valid"] for row in rows) else "FAIL_ENGINEERING",
        "preparation": rows[0]["preparation"],
        "feature_mode": rows[0]["feature_mode"],
        "lookahead_steps": rows[0].get("lookahead_steps"),
        "time_steps": rows[0]["time_steps"],
        "value": {"contract_e_tp": objective, "kalman": kalman_value, "difference": objective - kalman_value, "center_screen_pass": value_pass},
        "score": {"contract_e_tp": score, "kalman": oracle, "same_scalar_finite_difference": fd, "same_scalar_fd_relative_error": [row["finite_difference_relative_error"] for row in rows], "componentwise_relative_error": relative, "sign_reversal": signs, "center_screen_pass": score_pass, "same_scalar_fd_pass": same_scalar_pass},
        "chart": rows[0]["chart"],
        "decision": {"engineering_pass": same_scalar_pass and all(row["chart"]["valid"] for row in rows), "center_value_screen_pass": value_pass, "center_gradient_screen_pass": score_pass, "lgssm_candidate_pass": same_scalar_pass and value_pass and score_pass and all(row["chart"]["valid"] for row in rows)},
        "shards": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "wall_time_seconds": row["execution"]["wall_time_seconds"],
                "command": row["execution"]["command"],
            }
            for path, row in zip(args.shards, [
                next(item for item in rows if item["direction_index"] == index)
                for index in range(5)
            ])
        ],
        "execution": {
            "git_commit": rows[0]["execution"]["git_commit"],
            "backend": "five independent TensorFlow float64 CPU-hidden scalar JVP shards",
            "total_shard_wall_time_seconds": sum(
                row["execution"]["wall_time_seconds"] for row in rows
            ),
            "aggregation_command": " ".join(__import__("sys").argv),
        },
        "nonclaims": ["center-only diagnostic", "no nonlinear transfer", "no canonical, leaderboard, default, or HMC admission"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"value": payload["value"], "score": payload["score"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
