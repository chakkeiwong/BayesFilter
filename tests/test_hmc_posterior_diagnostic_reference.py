"""Independent reference checks for the native reporting repair, not HMC."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.inference import hmc_posterior_diagnostics as native
from scipy.signal import lfilter

pytest.importorskip("matplotlib.style.core")
az = pytest.importorskip("arviz")
reference = az.stats.diagnostics


@pytest.mark.parametrize("draws", [4, 5, 6, 7, 8, 15, 16, 31, 64, 127, 256, 640])
@pytest.mark.parametrize("chains", [1, 4])
def test_ess_matches_independent_reference_across_lengths(draws, chains):
    rng = np.random.default_rng(847 + draws)
    innovations = rng.normal(size=(chains, draws + 512, 4))
    values = lfilter([1.0], [1.0, -.85], innovations, axis=1)[:, 512:]
    values[:, :, 1] = lfilter([1.0], [1.0, .95], innovations[:, :, 1], axis=1)[:, 512:]
    values[:, :, 2] = innovations[:, 512:, 2]
    values[:, :, 3] = np.round(innovations[:, 512:, 3])
    got = native._cross_chain_ess(tf.constant(values.transpose(1, 0, 2), tf.float64)).numpy()
    expected = [reference._ess(values[:, :, i]) for i in range(4)]
    np.testing.assert_allclose(got, expected, rtol=2e-11, atol=2e-11)
    assert np.all(got > 0)


def test_bulk_tail_rank_and_all_mean_mcse_callers_match_reference():
    rng = np.random.default_rng(131)
    values = rng.normal(size=(4, 512, 4))
    values[:, :, 1] = np.repeat(values[:, ::16, 1], 16, axis=1)
    values[:, :, 2] = lfilter([1.0], [1.0, .95], rng.normal(size=(4, 1024)), axis=1)[:, 512:]
    values[:, :, 3] = np.round(values[:, :, 3] * 2) / 2
    tensor = tf.constant(values, tf.float64)
    rhat = native.rank_normalized_split_rhat(tensor)["maximum"].numpy()
    ess = native.rank_normalized_bulk_tail_ess(tensor)
    mean = native.posterior_mean_diagnostics(tensor)
    for i in range(4):
        np.testing.assert_allclose(rhat[i], az.rhat(values[:, :, i]), rtol=1e-12)
        for key in ("bulk", "tail"):
            np.testing.assert_allclose(ess[key].numpy()[i], az.ess(values[:, :, i], method=key), rtol=2e-11)
        expected_mean_ess = reference._ess(values[:, :, i])
        np.testing.assert_allclose(mean["pooled_mean_ess"].numpy()[i], expected_mean_ess, rtol=2e-11)
        expected_chain_ess = [reference._ess(row[None, :]) for row in values[:, :, i]]
        np.testing.assert_allclose(mean["per_chain_ess"].numpy()[:, i], expected_chain_ess, rtol=2e-11)
        np.testing.assert_allclose(mean["per_chain_mean_mcse"].numpy()[:, i],
            values[:, :, i].std(axis=1, ddof=1) / np.sqrt(expected_chain_ess), rtol=2e-11)


def test_constant_ess_remains_nonpromotable():
    ess = native._cross_chain_ess(tf.ones([16, 4, 2], tf.float64)).numpy()
    assert np.isnan(ess).all()


@pytest.mark.parametrize("draws", [16, 160, 257])
def test_constant_ess_does_not_depend_on_xla_mean_rounding(draws):
    values = tf.broadcast_to(tf.constant([1., .1, 1.e6], tf.float64), [draws, 4, 3])
    ess = native._cross_chain_ess(values).numpy()
    assert np.isnan(ess).all()


def test_percentile_keeps_equal_endpoints_and_tail_cdf_boundary():
    endpoint = np.float64.fromhex("0x1.7bc004f322299p-12")
    values = np.full((4, 256, 1), endpoint)
    values[:, :10] = endpoint / 2
    values[:, -10:] = endpoint * 2
    tensor = tf.constant(values, tf.float64)
    assert native._pooled_percentile(tensor, .05).numpy()[0] == endpoint
    assert native._pooled_percentile(tensor, .95).numpy()[0] == endpoint
    tails = native.rank_normalized_bulk_tail_ess(tensor)
    np.testing.assert_allclose(tails["upper_95pct"].numpy()[0],
        az.ess(values[:, :, 0], method="quantile", prob=.95), rtol=2e-11)


def test_graph_ess_matches_eager_on_saved_negative_ess_fixture():
    path = os.environ.get("DZ5_ESS_REFERENCE_FIXTURES")
    if not path:
        pytest.skip("optional preserved DZ5 reproducer is not configured")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["saved_L8_physical_coordinate_10"]
    normalized = reference._z_scale(reference._split_chains(values))
    tensor = tf.constant(normalized.T[..., None], tf.float64)
    expected = reference._ess(normalized)
    eager = native._cross_chain_ess(tensor).numpy()[0]
    graph = tf.function(native._cross_chain_ess)(tensor).numpy()[0]
    np.testing.assert_allclose([eager, graph], expected, rtol=2e-11)
    assert eager > 0


def test_ess_xla_matches_eager_on_antithetic_fixture():
    rng = np.random.default_rng(844)
    values = lfilter([1.0], [1.0, .95], rng.normal(size=(4, 640)), axis=1)[:, 512:]
    tensor = tf.constant(values.T[..., None], tf.float64)
    compiled = tf.function(native._cross_chain_ess, jit_compile=True)(tensor).numpy()
    np.testing.assert_allclose(compiled, native._cross_chain_ess(tensor).numpy(), rtol=2e-11)


def test_initialization_memory_is_invariant_to_exact_large_location_shift():
    rng = np.random.default_rng(812)
    values = np.stack([rng.permutation(64) + chain for chain in range(4)])[..., None].astype(float)
    reference_values = native.initialization_memory_statistics(tf.constant(values, tf.float64))
    shifted = native.initialization_memory_statistics(tf.constant(values + 2**30, tf.float64))
    np.testing.assert_allclose(shifted["standardized_difference"].numpy(),
        reference_values["standardized_difference"].numpy(), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("fixture", ["odd", "antithetic", "tied", "constant"])
def test_public_convergence_api_reaches_repaired_independent_reference(fixture):
    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds, rank_normalized_hmc_diagnostics,
    )
    from bayesfilter.inference.hmc_ess import STAN_ESS_VERSION
    rng = np.random.default_rng(5521)
    values = rng.normal(size=(4, 257, 2))
    if fixture == "antithetic":
        values = lfilter([1.], [1., .95], rng.normal(size=(4, 769, 2)), axis=1)[:, 512:]
    elif fixture == "tied":
        endpoint = np.float64.fromhex("0x1.7bc004f322299p-12")
        values[:] = endpoint
        for chain in range(4):
            values[chain, :4+chain] = endpoint/2
            values[chain, -6-chain:] = endpoint*2
    elif fixture == "constant":
        values[:] = 1.
    got = rank_normalized_hmc_diagnostics(tf.constant(values.transpose(1, 0, 2)),
        parameter_names=("a", "b"), thresholds=RankNormalizedHMCThresholds())
    assert got["schema"] == "bayesfilter.rank_normalized_hmc_diagnostics.v2"
    assert got["bulk_tail_ess_method"] == STAN_ESS_VERSION
    assert got["split_draw_count_per_chain"] == 128
    assert got["split_chain_count"] == 8
    if fixture == "constant":
        # Deliberate stricter convention than ArviZ's constant ESS sentinel:
        # constant chains provide no estimable mixing or posterior admission.
        assert not got["passed"]
        assert got["hard_vetoes"] == ("nonfinite_convergence_diagnostic",)
        assert all(np.isnan(row["bulk_ess"]) for row in got["parameter_diagnostics"])
        return
    for i, row in enumerate(got["parameter_diagnostics"]):
        np.testing.assert_allclose(row["rhat"], az.rhat(values[:, :, i]), rtol=2e-11, atol=2e-11)
        for key in ("bulk", "tail"):
            np.testing.assert_allclose(row[key + "_ess"], az.ess(values[:, :, i], method=key),
                                       rtol=2e-11, atol=2e-11)
        for key, cutoff in (("lower", .05), ("upper", .95)):
            np.testing.assert_allclose(row[key + "_tail_ess"],
                az.ess(values[:, :, i], method="quantile", prob=cutoff), rtol=2e-11, atol=2e-11)
