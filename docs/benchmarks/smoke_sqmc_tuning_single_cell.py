#!/usr/bin/env python3
"""Single-cell wiring smoke check for the SQMC tuning runner.

Debug-only: verifies one control configuration evaluates end-to-end against the
exact Kalman oracle and returns finite metrics.  Not a research-decision run.
"""

import os
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
print(f"GPU memory growth verified: {policy.get('devices')}")

import run_sqmc_tuning as R  # noqa: E402


def main() -> int:
    horizon = 20
    particle_count = 1008

    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=R.DTYPE)
    observations = R._frozen_observations(horizon)
    print(f"observations: shape={observations.shape} dtype={observations.dtype.name}")

    oracle_score = R._oracle_score(observations, theta)
    print(f"oracle score: {oracle_score.numpy()}")

    controls = R._tuning_grid()[0]
    print(f"controls: {controls}")

    result = R._evaluate_controls(
        route="iid_dual_cap",
        controls=controls,
        observations=observations,
        theta=theta,
        oracle_score=oracle_score,
        seed=50001,
        horizon=horizon,
        particle_count=particle_count,
    )

    print("\n--- result ---")
    for key in (
        "valid",
        "score_l2_error",
        "cosine_similarity",
        "relative_norm_error",
        "fisher_scaled_errors",
        "induced_hmc_error",
        "tuning_score",
        "veto_reason",
    ):
        print(f"{key}: {result.get(key)}")
    if "error" in result:
        print(f"error: {result['error']}")

    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
