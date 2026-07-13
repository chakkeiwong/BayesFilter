from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from bayesfilter.inference.hmc_posterior_diagnostics import (
    Phase29DiagnosticThresholds,
    epoch_drift_statistics,
    evaluate_phase29_posterior_pilot,
    evaluate_phase29_warmup_epoch,
    per_chain_ebfmi,
    rank_normalized_split_rhat,
)


def _iid(seed: int = 12) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(5, 512, 4))


def _mechanics(seed: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    energy = rng.normal(size=(5, 512))
    accepted = rng.uniform(size=(5, 512)) < 0.7
    delta_h = rng.normal(scale=0.1, size=(5, 512))
    return energy, accepted, delta_h


def _coordinates(samples: np.ndarray) -> dict[str, np.ndarray]:
    return {"final_latent": samples, "named_model": 2.0 * samples + 1.0}


def _independent_bulk_rhat(samples: np.ndarray) -> np.ndarray:
    chains, draws, parameters = samples.shape
    half = draws // 2
    split = np.concatenate((samples[:, :half], samples[:, half:]), axis=0)
    sample_major = split.transpose(1, 0, 2)
    normalized = np.empty_like(sample_major)
    total = half * chains * 2
    for parameter in range(parameters):
        values = sample_major[:, :, parameter].reshape(-1)
        ranks = stats.rankdata(values, method="average")
        z = stats.norm.ppf((ranks - 3.0 / 8.0) / (total - 1.0 / 4.0))
        normalized[:, :, parameter] = z.reshape(half, chains * 2)
    chain_mean = normalized.mean(axis=0)
    within = normalized.var(axis=0, ddof=1).mean(axis=0)
    between_over_n = chain_mean.var(axis=0, ddof=1)
    variance_plus = ((half - 1.0) / half) * within + between_over_n
    return np.sqrt(variance_plus / within)


def test_rank_normalized_bulk_rhat_matches_independent_scipy_reference() -> None:
    samples = _iid()
    result = rank_normalized_split_rhat(tf.constant(samples, tf.float64))
    np.testing.assert_allclose(
        result["bulk"].numpy(),
        _independent_bulk_rhat(samples),
        rtol=0.0,
        atol=2e-14,
    )


def test_iid_fixture_passes_complete_warmup_screen() -> None:
    samples = _iid()
    energy, accepted, delta_h = _mechanics()
    result = evaluate_phase29_warmup_epoch(
        _coordinates(samples),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=delta_h,
    )
    assert result["passed"] is True
    assert result["hard_vetoes"] == []
    assert result["promotion_vetoes"] == []
    assert result["coordinate_diagnostics"]["final_latent"]["epoch_drift"][
        "status"
    ] == "not_applicable_first_epoch"


def test_shifted_chain_fails_rhat_and_initialization_memory() -> None:
    samples = _iid()
    samples[0] += 1.5
    energy, accepted, delta_h = _mechanics()
    result = evaluate_phase29_warmup_epoch(
        _coordinates(samples),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=delta_h,
    )
    text = " ".join(result["promotion_vetoes"])
    assert result["passed"] is False
    assert "rank_normalized_split_rhat_above_threshold" in text
    assert "initialization_memory_above_threshold" in text


def test_nonstationary_adjacent_epoch_drift_fails() -> None:
    previous = _iid(30)
    current = previous + 0.8
    energy, accepted, delta_h = _mechanics()
    result = evaluate_phase29_warmup_epoch(
        _coordinates(current),
        previous_coordinate_samples=_coordinates(previous),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=delta_h,
    )
    assert any("epoch_mean_drift_above_threshold" in item for item in result["promotion_vetoes"])


def test_sticky_tail_fixture_fails_tail_ess() -> None:
    samples = _iid(40)
    for chain in range(5):
        samples[chain, :, 0] = np.repeat(samples[chain, ::64, 0], 64)
    energy, accepted, delta_h = _mechanics()
    result = evaluate_phase29_posterior_pilot(
        _coordinates(samples),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=delta_h,
    )
    assert any("tail_ess_below_threshold" in item for item in result["promotion_vetoes"])


def test_low_ebfmi_fixture_fails() -> None:
    samples = _iid(50)
    energy = np.tile(np.linspace(-10.0, 10.0, 512), (5, 1))
    accepted = np.random.default_rng(51).uniform(size=(5, 512)) < 0.7
    result = evaluate_phase29_posterior_pilot(
        _coordinates(samples),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=np.zeros((5, 512)),
    )
    assert "per_chain_ebfmi_below_threshold" in result["promotion_vetoes"]
    assert np.all(per_chain_ebfmi(energy).numpy() < 0.30)


def test_constant_nonfinite_shape_and_hard_energy_fixtures_fail_closed() -> None:
    energy, accepted, delta_h = _mechanics()
    constant = np.ones((5, 512, 4))
    result = evaluate_phase29_posterior_pilot(
        _coordinates(constant),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=delta_h,
    )
    assert result["passed"] is False
    assert "unmoved_chain" in result["hard_vetoes"]

    nonfinite = _iid(60)
    nonfinite[0, 0, 0] = np.nan
    result = evaluate_phase29_posterior_pilot(
        _coordinates(nonfinite),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=delta_h,
    )
    assert "final_latent_samples_nonfinite" in result["hard_vetoes"]

    explosive = delta_h.copy()
    explosive[2, 3] = 1000.1
    result = evaluate_phase29_posterior_pilot(
        _coordinates(_iid(61)),
        initial_energy=energy,
        is_accepted=accepted,
        delta_h=explosive,
    )
    assert "absolute_delta_h_above_hard_limit" in result["hard_vetoes"]

    with pytest.raises(ValueError, match="shape"):
        evaluate_phase29_posterior_pilot(
            {"final_latent": np.zeros((5, 512))},
            initial_energy=energy,
            is_accepted=accepted,
            delta_h=delta_h,
        )
    with pytest.raises(ValueError, match="final_latent"):
        evaluate_phase29_posterior_pilot(
            {"named_model": _iid(62)},
            initial_energy=energy,
            is_accepted=accepted,
            delta_h=delta_h,
        )


def test_drift_and_threshold_contracts_validate() -> None:
    samples = tf.constant(_iid(70), tf.float64)
    drift = epoch_drift_statistics(samples, samples)
    np.testing.assert_array_equal(
        drift["abs_standardized_mean_difference"].numpy(), np.zeros(4)
    )
    np.testing.assert_allclose(
        drift["sd_ratio_current_over_previous"].numpy(), np.ones(4)
    )
    with pytest.raises(ValueError, match="rhat_max"):
        Phase29DiagnosticThresholds(rhat_max=1.0)
