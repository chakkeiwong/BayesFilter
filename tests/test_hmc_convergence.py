from __future__ import annotations

import numpy as np
import pytest
import scipy.stats
import tensorflow as tf

from bayesfilter.inference.hmc_convergence import (
    RankNormalizedHMCThresholds,
    _rank_normalize,
    rank_normalized_hmc_diagnostics,
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.hmc import _rhat_summary_from_retained_samples


def _thresholds(*, rhat: float = 1.05, bulk: float = 50.0, tail: float = 20.0):
    return RankNormalizedHMCThresholds(
        rhat_max=rhat,
        bulk_ess_min=bulk,
        tail_ess_min=tail,
    )


def test_iid_chains_pass_relaxed_synthetic_gate() -> None:
    draws = tf.random.stateless_normal(
        [800, 4, 3],
        seed=(20260711, 1),
        dtype=tf.float64,
    )

    payload = rank_normalized_hmc_diagnostics(
        draws,
        parameter_names=("a", "b", "c"),
        thresholds=_thresholds(),
    )

    assert payload["passed"] is True
    assert payload["input_all_finite"] is True
    assert payload["split_chain_count"] == 8
    assert [row["parameter"] for row in payload["parameter_diagnostics"]] == [
        "a",
        "b",
        "c",
    ]


def test_shifted_chain_fails_rhat_gate() -> None:
    base = tf.random.stateless_normal([400, 4, 2], seed=(20260711, 2), dtype=tf.float64)
    shift = tf.constant([0.0, 0.0, 0.0, 4.0], tf.float64)[tf.newaxis, :, tf.newaxis]

    payload = rank_normalized_hmc_diagnostics(
        base + shift,
        parameter_names=("a", "b"),
        thresholds=_thresholds(rhat=1.01, bulk=10.0, tail=10.0),
    )

    assert payload["passed"] is False
    assert payload["max_rhat"] > 1.01


def test_tuning_and_phase7_share_folded_scale_mismatch_rhat() -> None:
    base = tf.random.stateless_normal(
        [800, 4, 1],
        seed=(20260713, 91),
        dtype=tf.float64,
    )
    draws = base * tf.constant([0.5, 1.0, 2.0, 3.0], tf.float64)[
        tf.newaxis, :, tf.newaxis
    ]

    shared = rank_normalized_split_rhat_summary(draws, rhat_max=1.01)
    phase7 = rank_normalized_hmc_diagnostics(
        draws,
        parameter_names=("scale_mismatch",),
        thresholds=_thresholds(rhat=1.01, bulk=1.0, tail=1.0),
    )
    tuning = _rhat_summary_from_retained_samples(draws, threshold=1.01)

    assert shared["max_rank_normalized_split_rhat"] < 1.01
    assert shared["max_folded_rank_normalized_split_rhat"] > 1.01
    assert shared["passed"] is False
    assert phase7["passed"] is False
    assert tuning["passed"] is False
    assert tuning["rhat_definition"] == phase7["definitions"]["rhat"]
    assert tuning["max_finite_rhat"] == pytest.approx(phase7["max_rhat"])
    assert tuning["max_finite_rhat"] == pytest.approx(shared["max_finite_rhat"])


def test_ties_are_average_ranked_and_finite() -> None:
    values = tf.cast(
        tf.math.floormod(tf.range(320 * 4 * 2), 7),
        tf.float64,
    )
    draws = tf.reshape(values, [320, 4, 2])

    payload = rank_normalized_hmc_diagnostics(
        draws,
        parameter_names=("a", "b"),
        thresholds=_thresholds(rhat=2.0, bulk=1.0, tail=1.0),
    )

    assert payload["diagnostics_all_finite"] is True
    assert all(row["rhat"] is not None for row in payload["parameter_diagnostics"])


def test_odd_draw_count_discards_middle_draw_for_split() -> None:
    draws = tf.random.stateless_normal([401, 4, 1], seed=(20260711, 3), dtype=tf.float64)

    payload = rank_normalized_hmc_diagnostics(
        draws,
        parameter_names=("a",),
        thresholds=_thresholds(rhat=2.0, bulk=1.0, tail=1.0),
    )

    assert payload["split_draw_count_per_chain"] == 200
    assert payload["split_chain_count"] == 8


def test_nonfinite_draws_fail_closed() -> None:
    draws = np.zeros((20, 4, 2), dtype=np.float64)
    draws[3, 1, 0] = np.nan

    payload = rank_normalized_hmc_diagnostics(
        draws,
        parameter_names=("a", "b"),
        thresholds=_thresholds(),
    )

    assert payload["passed"] is False
    assert payload["hard_vetoes"] == ("nonfinite_input_draws",)


def test_shape_and_parameter_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        rank_normalized_hmc_diagnostics(
            tf.zeros([10, 4], tf.float64),
            parameter_names=("a",),
            thresholds=_thresholds(),
        )
    with pytest.raises(ValueError, match="parameter_names"):
        rank_normalized_hmc_diagnostics(
            tf.zeros([10, 4, 2], tf.float64),
            parameter_names=("a",),
            thresholds=_thresholds(),
        )


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="rhat_max"):
        RankNormalizedHMCThresholds(rhat_max=1.0)
    with pytest.raises(ValueError, match="bulk_ess_min"):
        RankNormalizedHMCThresholds(bulk_ess_min=0.0)


def test_rank_normalization_matches_independent_scipy_reference_with_ties() -> None:
    draws = np.asarray(
        [
            [[1.0, 4.0], [1.0, 2.0]],
            [[3.0, 2.0], [2.0, 2.0]],
            [[3.0, 7.0], [5.0, 4.0]],
            [[5.0, 7.0], [5.0, 9.0]],
        ],
        dtype=np.float64,
    )
    flat = draws.reshape(-1, 2)
    count = flat.shape[0]
    reference = np.column_stack(
        [
            scipy.stats.norm.ppf(
                (scipy.stats.rankdata(flat[:, column], method="average") - 3.0 / 8.0)
                / (count + 1.0 / 4.0)
            )
            for column in range(flat.shape[1])
        ]
    ).reshape(draws.shape)

    observed = _rank_normalize(tf.constant(draws, tf.float64)).numpy()

    np.testing.assert_allclose(observed, reference, rtol=0.0, atol=1.0e-12)
