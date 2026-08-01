#!/usr/bin/env python3
"""Aggregate paired canonical LGSSM particle diagnostic arms."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim.transport_chunk_policy import (
    TRANSPORT_CHUNK_POLICY_ID,
    select_transport_chunks,
)
from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


CAMPAIGN_ID = "canonical-lgssm-balancing-kalman-repair-20260717"
DATASET_SEED = 81100
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
TIME_STEPS = 2
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
MARGINS = (0.001, 0.05, 0.05, 0.05, 0.05, 0.05)
CRITICAL_VALUE = 3.036283222821165


def _require_transport_chunk_identity(
    preparation_identity: dict[str, Any], num_particles: int
) -> None:
    chunks = select_transport_chunks(num_particles)
    required = {
        "transport_chunk_policy_id": TRANSPORT_CHUNK_POLICY_ID,
        "row_chunk_size": chunks.row_chunk_size,
        "col_chunk_size": chunks.col_chunk_size,
        "transport_block_grid": [chunks.row_blocks, chunks.col_blocks],
    }
    actual = {name: preparation_identity.get(name) for name in required}
    if actual != required:
        raise ValueError(
            f"ineligible transport chunk identity: expected {required}, got {actual}"
        )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _kalman() -> tuple[float, list[float]]:
    observations = tf.convert_to_tensor(
        _lgssm_dataset(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    theta = tf.constant(THETA, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        phi = theta[:3]
        q_scale = theta[3]
        r_scale = theta[4]
        value = tf_kalman_log_likelihood(
            observations=observations,
            transition_offset=tf.zeros([3], tf.float64),
            transition_matrix=tf.linalg.diag(phi),
            transition_covariance=tf.square(q_scale) * tf.eye(3, dtype=tf.float64),
            observation_offset=tf.zeros([3], tf.float64),
            observation_matrix=canonical._observation_matrix(tf.float64),
            observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=tf.float64),
            initial_state_mean=tf.zeros([3], tf.float64),
            initial_state_covariance=tf.linalg.diag(
                tf.square(q_scale) / (1.0 - tf.square(phi))
            ),
        )
    physical = tape.gradient(value, theta)
    chain = tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)
    return float(value.numpy()), (physical * chain).numpy().tolist()


def _interval(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    stddev = statistics.stdev(values)
    standard_error = stddev / math.sqrt(len(values))
    radius = CRITICAL_VALUE * standard_error
    return {
        "mean": mean,
        "standard_deviation": stddev,
        "standard_error": standard_error,
        "critical_value": CRITICAL_VALUE,
        "lower": mean - radius,
        "upper": mean + radius,
    }


def _screen(intervals: list[dict[str, float]], hard_valid: bool) -> str:
    if not hard_valid:
        return "screen_fail"
    if all(
        interval["lower"] >= -margin and interval["upper"] <= margin
        for interval, margin in zip(intervals, MARGINS, strict=True)
    ):
        return "screen_pass"
    if any(
        interval["lower"] > margin or interval["upper"] < -margin
        for interval, margin in zip(intervals, MARGINS, strict=True)
    ):
        return "screen_fail"
    return "inconclusive"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "arm_complete":
        raise ValueError(f"arm is not complete: {path}")
    return payload["result"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-e", type=Path, required=True)
    parser.add_argument("--no-reset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract_e = _load(args.contract_e.resolve())
    no_reset = _load(args.no_reset.resolve())
    identity_fields = (
        "num_particles",
        "time_steps",
        "estimator_seeds",
        "observation_sha256",
        "theta",
        "source_sha256",
    )
    identity_equal = all(contract_e[field] == no_reset[field] for field in identity_fields)
    if not identity_equal:
        raise ValueError("paired arm identity mismatch")
    ce_preparation = dict(contract_e["preparation_identity"])
    nr_preparation = dict(no_reset["preparation_identity"])
    _require_transport_chunk_identity(
        ce_preparation, int(contract_e["num_particles"])
    )
    _require_transport_chunk_identity(
        nr_preparation, int(no_reset["num_particles"])
    )
    ce_hashes = dict(ce_preparation.pop("tensor_sha256"))
    nr_hashes = dict(nr_preparation.pop("tensor_sha256"))
    ce_mask = ce_hashes.pop("fixed_reset_mask")
    nr_mask = nr_hashes.pop("fixed_reset_mask")
    preparation_identity = {
        "metadata_equal": ce_preparation == nr_preparation,
        "non_mask_hashes_equal": ce_hashes == nr_hashes,
        "reset_mask_hashes_differ": ce_mask != nr_mask,
    }
    if not all(preparation_identity.values()):
        raise ValueError(f"paired preparation mismatch: {preparation_identity}")
    kalman_value, kalman_score = _kalman()
    scales = [abs(kalman_value), *[abs(value) for value in kalman_score]]
    if any(not math.isfinite(scale) or scale == 0.0 for scale in scales):
        raise ValueError(f"invalid Kalman scale: {scales}")
    arms = {
        "all_active_contract_e": contract_e,
        "no_reset_weighted": no_reset,
    }
    z: dict[str, list[list[float]]] = {}
    intervals: dict[str, list[dict[str, float]]] = {}
    screens: dict[str, str] = {}
    for name, arm in arms.items():
        rows = []
        for value, score in zip(
            arm["per_seed_value"], arm["per_seed_hmc_score"], strict=True
        ):
            rows.append(
                [
                    (float(value) - kalman_value) / scales[0],
                    *[
                        (float(candidate) - oracle) / scale
                        for candidate, oracle, scale in zip(
                            score, kalman_score, scales[1:], strict=True
                        )
                    ],
                ]
            )
        z[name] = rows
        intervals[name] = [
            _interval([row[index] for row in rows]) for index in range(6)
        ]
        screens[name] = _screen(intervals[name], bool(arm["hard_valid"]))
    paired_loss = [
        [abs(left) - abs(right) for left, right in zip(ce, nr, strict=True)]
        for ce, nr in zip(
            z["all_active_contract_e"], z["no_reset_weighted"], strict=True
        )
    ]
    paired_intervals = [
        _interval([row[index] for row in paired_loss]) for index in range(6)
    ]
    paired_directions = []
    for interval in paired_intervals:
        if interval["upper"] < 0.0:
            paired_directions.append("contract_e_lower_error")
        elif interval["lower"] > 0.0:
            paired_directions.append("contract_e_higher_error")
        else:
            paired_directions.append("inconclusive")
    global_direction = (
        paired_directions[0]
        if len(set(paired_directions)) == 1
        and paired_directions[0] != "inconclusive"
        else "mixed_or_inconclusive"
    )
    labels = ("value", *PARAMETER_NAMES)
    payload = {
        "schema_version": "bayesfilter.canonical_lgssm_particle_diagnostic.v1",
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "aggregate_complete",
        "num_particles": contract_e["num_particles"],
        "time_steps": TIME_STEPS,
        "estimator_seeds": contract_e["estimator_seeds"],
        "paired_identity": {
            "common_fields_equal": identity_equal,
            **preparation_identity,
        },
        "kalman": {
            "value": kalman_value,
            "hmc_score": kalman_score,
            "scales": scales,
        },
        "analysis": {
            "familywise_level": 0.95,
            "family_size": 6,
            "degrees_of_freedom": 15,
            "critical_value": CRITICAL_VALUE,
            "margins": dict(zip(labels, MARGINS, strict=True)),
            "student_model_no_power_guarantee": True,
        },
        "arm_screens": screens,
        "arm_intervals": {
            name: dict(zip(labels, values, strict=True))
            for name, values in intervals.items()
        },
        "paired_loss_intervals": dict(zip(labels, paired_intervals, strict=True)),
        "paired_directions": dict(zip(labels, paired_directions, strict=True)),
        "global_reset_direction": global_direction,
        "per_seed_normalized_errors": z,
        "source_arms": {
            "contract_e": str(args.contract_e),
            "no_reset": str(args.no_reset),
        },
        "nonclaims": [
            "not a particle-count optimum",
            "not a parameter-region certificate",
            "not HMC readiness",
            "not method superiority",
            "not leaderboard admission",
        ],
    }
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "screens": screens}))


if __name__ == "__main__":
    main()
