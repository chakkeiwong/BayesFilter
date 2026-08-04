"""Run the bounded parent-measure ratio-bridge diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_conditional_reference_tf import (
    finite_value_and_analytical_score,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t1_score_tf import (
    t1_complete_data_parameter_score,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_t1_proposal_cloud,
)
from bayesfilter.highdim.zhao_cui_austria_sir_ratio_bridge_tf import (
    bridge_manifest,
    load_admitted_parent,
    local_to_physical,
    parent_measure_ratio_bridge,
    parent_measure_ratio_bridge_autodiff,
    sample_parent_local_points,
)


def _jsonable(value):
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--seed-a", type=int, default=92101)
    parser.add_argument("--seed-b", type=int, default=92102)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    parent = load_admitted_parent()
    theta = tf.zeros([3], tf.float64)
    rows = []
    for seed in (args.seed_a, args.seed_b):
        local = sample_parent_local_points(
            sample_count=args.sample_count, seed=seed, parent=parent
        )
        physical = local_to_physical(local, parent)
        bridge = parent_measure_ratio_bridge(theta, physical, parent=parent)
        autodiff = parent_measure_ratio_bridge_autodiff(theta, physical)
        physical_cloud = generate_t1_proposal_cloud(
            sample_count=args.sample_count, seed=seed + 1000, role=f"physical_{seed}"
        )
        physical_weights = tf.nn.softmax(physical_cloud.log_likelihood)
        physical_score_rows = t1_complete_data_parameter_score(physical_cloud.joint_points)
        physical_score = tf.reduce_sum(
            physical_weights[:, tf.newaxis] * physical_score_rows, axis=0
        )
        physical_reference = {
            "log_value": tf.reduce_logsumexp(physical_cloud.log_likelihood)
            - tf.math.log(tf.cast(args.sample_count, tf.float64)),
            "score": physical_score,
            "standard_error": tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        tf.exp(physical_cloud.log_likelihood - tf.reduce_max(physical_cloud.log_likelihood))[:, tf.newaxis]
                        * (physical_score_rows - physical_score[tf.newaxis, :])
                        / tf.reduce_mean(tf.exp(physical_cloud.log_likelihood - tf.reduce_max(physical_cloud.log_likelihood)))
                    ),
                    axis=0,
                )
                / tf.cast(args.sample_count * (args.sample_count - 1), tf.float64)
            ),
            "effective_sample_size": tf.math.reciprocal(tf.reduce_sum(tf.square(physical_weights))),
        }
        rows.append(
            {
                "seed": seed,
                "bridge": {
                    "log_value": bridge["log_value"],
                    "score": bridge["score"],
                    "standard_error": bridge["standard_error"],
                    "effective_sample_size": bridge["effective_sample_size"],
                    "autodiff_log_value": autodiff["log_value"],
                    "autodiff_score": autodiff["score"],
                },
                "physical_reference": {
                    "log_value": physical_reference["log_value"],
                    "score": physical_reference["score"],
                    "standard_error": physical_reference["standard_error"],
                    "effective_sample_size": physical_reference["effective_sample_size"],
                },
                "manifest": bridge_manifest(parent, local),
            }
        )
    payload = {
        "bridge_id": "zhao_cui_austria_sir_t1_parent_measure_ratio_bridge_v1",
        "classification": "extension_or_invention",
        "parent_identity": parent.identity.hash.value,
        "sample_count": args.sample_count,
        "rows": rows,
        "nonclaims": (
            "not exact physical likelihood unless q0_equals_pi0",
            "not source-faithful Zhao-Cui",
            "not full-horizon score",
            "not HMC evidence",
        ),
    }
    (output / "result.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "bayesfilter.zhao_cui.austria_sir.ratio_bridge_manifest.v1",
                "command": "CUDA_VISIBLE_DEVICES=-1 python scripts/run_zhao_cui_austria_sir_ratio_bridge_t1.py",
                "sample_count": args.sample_count,
                "seeds": [args.seed_a, args.seed_b],
                "parent_identity": parent.identity.hash.value,
                "artifact": "result.json",
                "nonclaims": payload["nonclaims"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
