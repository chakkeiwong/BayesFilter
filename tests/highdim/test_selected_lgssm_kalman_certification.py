from __future__ import annotations

import math

import tensorflow as tf

from docs.benchmarks import aggregate_selected_lgssm_kalman_certification as cert


def test_interval_and_screen_require_simultaneous_containment() -> None:
    intervals = [
        {"lower": -cert.VALUE_MARGIN / 2.0, "upper": cert.VALUE_MARGIN / 2.0},
        *[
            {"lower": -cert.SCORE_MARGIN / 2.0, "upper": cert.SCORE_MARGIN / 2.0}
            for _ in range(5)
        ],
    ]
    assert cert._screen(intervals, True) == "screen_pass"
    intervals[1] = {"lower": 0.04, "upper": 0.06}
    assert cert._screen(intervals, True) == "inconclusive"
    intervals[1] = {"lower": 0.051, "upper": 0.07}
    assert cert._screen(intervals, True) == "screen_fail"
    assert cert._screen(intervals, False) == "screen_fail"


def test_binding_requires_selected_controls_seeds_and_current_source_revalidation() -> None:
    horizon = 10
    expected = cert.EXPECTED[horizon]
    result = {
        "time_steps": horizon,
        "num_particles": 1024,
        "sinkhorn_steps": expected["sinkhorn_steps"],
        "balance_steps": expected["balance_steps"],
        "estimator_seeds": expected["seeds"],
        "device": {"dtype": "float32", "tf32_enabled": True, "jit_compile": True},
        "preparation_identity": {"row_chunk_size": 1024, "col_chunk_size": 1024},
        "graph": {
            "python_horizon_unroll": False,
            "while_operation_types": ["StatelessWhile"],
        },
        "per_seed_value": [-1.0] * 16,
        "per_seed_physical_score": [[0.0] * 5] * 16,
    }
    selection = {
        "sinkhorn_steps": expected["sinkhorn_steps"],
        "balance_steps": expected["balance_steps"],
        "kalman_used": False,
    }
    claim = {
        "time_steps": horizon,
        "num_particles": 1024,
        "sinkhorn_steps": expected["sinkhorn_steps"],
        "balance_steps": expected["balance_steps"],
        "estimator_seeds": expected["seeds"],
        "per_seed_value": [-1.0] * 16,
        "per_seed_physical_score": [[0.0] * 5] * 16,
    }
    assert cert._require_binding(
        horizon=horizon, node=result, selection=selection, claim=claim
    )["all_valid"]

    result["per_seed_value"][0] = -1.0001
    binding = cert._require_binding(
        horizon=horizon, node=result, selection=selection, claim=claim
    )
    assert binding["all_valid"]
    assert not binding["historical_float32_exact_replay"]["per_seed_value_exact"]
    assert not binding["historical_float32_exact_replay_used_as_gate"]

    result["balance_steps"] = 8
    assert not cert._require_binding(
        horizon=horizon, node=result, selection=selection, claim=claim
    )["all_valid"]


def test_hmc_chain_matches_declared_coordinate_map() -> None:
    chain = cert._hmc_chain().numpy().tolist()
    assert chain == [1.0 - 0.72**2, 1.0 - 0.55**2, 1.0 - 0.35**2, 0.35, 0.45]


def test_production_kalman_uses_float32_rounded_observation_target() -> None:
    value, score = cert._production_kalman(10)
    assert math.isclose(value, -32.052615747575366, rel_tol=0.0, abs_tol=1.0e-14)
    tf.debugging.assert_near(
        score,
        tf.constant(
            [
                5.432444913597522,
                -0.2120687345477662,
                -1.1451714387700966,
                3.321030884533205,
                6.330751006058259,
            ],
            tf.float64,
        ),
        atol=1.0e-14,
        rtol=1.0e-14,
    )
