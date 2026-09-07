"""Focused checks for the Phase 7 integrated diagnostic harness."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = (
    ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase7_integrated_20260904.py"
)
SPEC = importlib.util.spec_from_file_location("c2_phase7_integrated_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)

DTYPE = tf.float64


def test_chunked_exact_target_matches_dense_contraction() -> None:
    transition = tf.constant([[0.7, 0.1], [-0.2, 0.6]], DTYPE)
    theta = tf.constant([0.55, math.log(0.4)], DTYPE)
    sigma = 0.8
    parents = tf.constant(
        [[-0.3, 0.2], [0.7, -0.4], [0.1, 0.8], [-0.6, -0.2]], DTYPE
    )
    weights = tf.constant([0.1, 0.2, 0.3, 0.4], DTYPE)
    observation = tf.constant([0.5, -0.9], DTYPE)
    rows = tf.constant([[0.2, -0.1], [0.8, 0.3], [-0.5, 0.6]], DTYPE)
    kernel = DRIVER._make_chunked_target_kernel(
        particle_count=4,
        state_dim=2,
        row_count=3,
        transition=transition,
        theta=theta,
        sigma=sigma,
        parent_chunk=2,
        jit_compile=False,
    )
    chunked = kernel(parents, weights, observation, rows)

    means = tf.linalg.matmul(parents, transition, transpose_b=True)
    residual = rows[:, None, :] - means[None, :, :]
    log_transition = -0.5 * (
        tf.constant(2.0 * math.log(2.0 * math.pi * sigma**2), DTYPE)
        + tf.reduce_sum(tf.square(residual), axis=2)
    )
    log_predictive = tf.reduce_logsumexp(
        log_transition + tf.math.log(weights)[None, :], axis=1
    )
    log_observation = tf.reduce_sum(
        -0.5 * tf.constant(math.log(2.0 * math.pi), DTYPE)
        - theta[1]
        - 0.5 * rows
        - 0.5
        * tf.square(observation)[None, :]
        * tf.exp(-rows - 2.0 * theta[1]),
        axis=1,
    )
    tf.debugging.assert_near(chunked, log_predictive + log_observation, atol=2e-12)


def test_chunked_target_rejects_nondivisible_parent_extent() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DRIVER._make_chunked_target_kernel(
            particle_count=5,
            state_dim=2,
            row_count=3,
            transition=tf.eye(2, dtype=DTYPE),
            theta=tf.constant([0.5, -0.7], DTYPE),
            sigma=1.0,
            parent_chunk=2,
            jit_compile=False,
        )


def test_heuristic_loss_is_a_promotion_veto() -> None:
    conditional_candidate = {str(time): 10.0 for time in DRIVER.MAP_TIMES}
    conditional_heuristic = {str(time): 9.0 for time in DRIVER.MAP_TIMES}
    conditional_heuristic["10"] = 11.0
    summary = {
        "ukf_apf_k1": {
            "evaluated_count": 3,
            "conditional_ess_mean": conditional_candidate,
        },
        **{
            family: {
                "evaluated_count": 3,
                "conditional_ess_mean": conditional_heuristic,
            }
            for family in DRIVER.HEURISTIC_FAMILIES
        },
    }
    verdict = DRIVER._heuristic_dominance(summary)["ukf_apf_k1"]
    assert verdict["verdict"] == "PROMOTION_VETO_HEURISTIC_LOSS"
    assert any(row["candidate_loses"] for row in verdict["comparisons"])
