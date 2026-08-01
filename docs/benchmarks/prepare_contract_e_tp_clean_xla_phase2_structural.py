#!/usr/bin/env python3
"""Prepare fixed per-step charts for the clean-XLA structural fixture."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("structural preparation requires CUDA_VISIBLE_DEVICES=-1")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_structural_tf as structural
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


DTYPE = tf.float64
THETA = tf.constant([0.70, 0.25, 0.55, 0.80], DTYPE)
PARENTS = tf.constant(
    [[-0.8, 0.2], [-0.1, -0.3], [0.6, 0.4], [1.0, -0.2]], DTYPE
)
PARENT_WEIGHTS = tf.constant([0.15, 0.35, 0.30, 0.20], DTYPE)
INNOVATIONS = tf.constant([[-1.2], [0.0], [1.2]], DTYPE)
INNOVATION_WEIGHTS = tf.constant([0.25, 0.50, 0.25], DTYPE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preparation() -> dict[str, object]:
    parents = PARENTS
    weights = PARENT_WEIGHTS
    steps: list[dict[str, object]] = []
    for time_index in range(5):
        repeated = tf.repeat(parents, tf.shape(INNOVATIONS)[0], axis=0)
        tiled = tf.tile(INNOVATIONS, [tf.shape(parents)[0], 1])
        candidates = structural.structural_fixture_transition_components_tf(
            repeated, tiled, THETA
        )["candidates"]
        teacher_weights = tf.reshape(
            weights[:, None] * INNOVATION_WEIGHTS[None, :], [-1]
        )
        features = structural.structural_fixture_features_tf(candidates)
        target = tp._dense_teacher_reduce_core(
            tf.math.log(teacher_weights), features
        )["target"]
        row_scale = tf.maximum(
            tf.reduce_max(tf.abs(features), axis=1), tf.abs(target)
        )
        matrix = (features / row_scale[:, None]).numpy()
        scaled_target = (target / row_scale).numpy()
        viable: list[dict[str, object]] = []
        for indices in itertools.combinations(range(12), 4):
            active = matrix[:, indices]
            if np.linalg.matrix_rank(active) != 4:
                continue
            try:
                student_weights = np.linalg.solve(active, scaled_target)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(student_weights)) or np.min(student_weights) <= 0.0:
                continue
            viable.append(
                {
                    "active_indices": list(indices),
                    "minimum_weight": float(np.min(student_weights)),
                    "student_weights": student_weights.tolist(),
                    "condition_number_numpy": float(np.linalg.cond(active)),
                }
            )
        viable.sort(
            key=lambda item: (
                -float(item["minimum_weight"]),
                tuple(item["active_indices"]),
            )
        )
        if not viable:
            raise RuntimeError(f"no positive full-rank chart at step {time_index}")
        selected = viable[0]
        projection = tp._contract_e_tp_dense_square_forward_core(
            candidates,
            tf.math.log(teacher_weights),
            features,
            tf.constant(selected["active_indices"], tf.int32),
            row_scale,
        )
        steps.append(
            {
                "time_index": time_index,
                "active_indices": selected["active_indices"],
                "row_scales": row_scale.numpy().tolist(),
                "minimum_weight": selected["minimum_weight"],
                "condition_number_numpy": selected["condition_number_numpy"],
                "viable_count": len(viable),
                "runner_up_minimum_weight": (
                    viable[1]["minimum_weight"] if len(viable) > 1 else None
                ),
                "all_viable_candidates": viable,
            }
        )
        parents = projection["student_points"]
        weights = projection["student_weights"]
    source_path = ROOT / "bayesfilter/highdim/ledh_contract_e_tp_structural_tf.py"
    return {
        "schema": "contract_e_tp.clean_xla_phase2_structural_preparation.v1",
        "status": "PASS_FIXED_PER_STEP_STRUCTURAL_CHART_PREPARATION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device_policy": "CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
        "selection_rule": {
            "candidate_set": "all lexicographic 4-of-12 teacher indices",
            "eligibility": "numpy full rank and finite strictly positive solved weights",
            "objective": "maximize minimum student weight",
            "tie_break": "lexicographic active indices",
            "runtime_selection": False,
        },
        "fixture": {
            "theta": THETA.numpy().tolist(),
            "initial_parents": PARENTS.numpy().tolist(),
            "initial_weights": PARENT_WEIGHTS.numpy().tolist(),
            "innovations": INNOVATIONS.numpy().tolist(),
            "innovation_weights": INNOVATION_WEIGHTS.numpy().tolist(),
            "value_operation_count": structural.STRUCTURAL_FIXTURE_VALUE_OPERATION_COUNT,
            "tangent_operation_count": structural.STRUCTURAL_FIXTURE_TANGENT_OPERATION_COUNT,
            "support_perturbation": 1.0e-3,
        },
        "steps": steps,
        "source": {
            "path": str(source_path.relative_to(ROOT)),
            "sha256": _sha256(source_path),
        },
        "nonclaims": [
            "not a parameter-region chart certificate",
            "not runtime chart selection",
            "not a DSGE, NAWM, or SIR model",
            "not scientific filtering accuracy",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    payload = build_preparation()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": [step["active_indices"] for step in payload["steps"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
