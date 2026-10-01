"""Independent CPU references and adversarial acceptance-diagnostic regressions."""
import json
from dataclasses import replace

import numpy as np  # Independent reference arithmetic only.
import pytest
import tensorflow as tf

from bayesfilter.inference.mcmc_uncertainty import chain_batch_covariance
from bayesfilter.inference.hmc_acceptance_uncertainty import (
    AcceptanceUncertaintyPolicy, acceptance_uncertainty_report,
    evaluate_acceptance_uncertainty,
)


def policy(**kwargs):
    return replace(AcceptanceUncertaintyPolicy(batch_size=8, min_batches=8,
        family_alpha=.1, temporal_tolerance=.1, chain_tolerance=.1,
        planned_looks=1, planned_candidates=1), **kwargs)


def trace(n=512):
    return np.random.default_rng(194).uniform(.68, .72, (n, 4))


def reference_covariance(x, batch):
    # Separate loops and np.cov, not the production einsum/reshape expression.
    rows = []
    for chain in range(x.shape[1]):
        means = np.array([x[i:i+batch, chain].mean(axis=0)
                          for i in range(0, len(x)-batch+1, batch)])
        rows.append(batch*np.atleast_2d(np.cov(means, rowvar=False, ddof=1)))
    return np.array(rows)


@pytest.mark.parametrize("method", ["batch_means", "lugsail"])
@pytest.mark.parametrize("jit", [False, True])
def test_covariance_matches_independent_reference_with_remainder(method, jit):
    x = np.random.default_rng(872).normal(size=(515, 3, 2))
    x[..., 1] += .6*x[..., 0]
    report = chain_batch_covariance(x, batch_size=24, min_batches=8,
                                   method=method, jit_compile=jit)
    expected = reference_covariance(x, 24)
    if method == "lugsail":
        expected = 2*expected-reference_covariance(x, 8)
    np.testing.assert_allclose(report["long_run_covariance_by_chain"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(report["pooled_variance_of_mean"],
        np.diagonal(expected, axis1=1, axis2=2).sum(axis=0)/(9*515), rtol=1e-12)
    np.testing.assert_allclose(report["pooled_mean"], x.mean((0, 1)), atol=1e-14)
    assert report["unused_terminal_draws"] == 11
    assert report["small_unused_terminal_draws"] == (3 if method == "lugsail" else 11)


@pytest.mark.parametrize("case", ["short", "constant", "nan", "alternating"])
def test_no_spurious_precision_from_unavailable_variance(case):
    x = np.ones((64, 4, 1))
    if case == "short":
        x = np.random.default_rng(183).normal(size=x.shape)
    elif case == "nan":
        x[1] = np.nan
    elif case == "alternating":
        x[1::2] = -1  # Every length-8 and length-24 batch mean is zero.
    r = chain_batch_covariance(x, batch_size=24, min_batches=8)
    assert not np.any(r["valid_by_chain"])
    assert np.all(np.isnan(r["mcse_by_chain"]))


def test_nonpositive_lugsail_is_reported_without_clipping():
    # Big batches cancel but smaller batches do not; lugsail is negative.
    x = np.tile(np.repeat([-.2, .2], 12), 32)[:, None, None]
    r = chain_batch_covariance(x, batch_size=24, min_batches=8)
    assert float(r["long_run_covariance_by_chain"][0, 0, 0]) < 0
    assert not bool(r["valid_by_chain"][0, 0])


def test_three_way_decision_and_opposing_drift_cannot_cancel():
    baseline = acceptance_uncertainty_report(trace(), policy=policy())
    assert baseline["experimental_decision"] == "compatible_for_fresh_verification"
    x = trace()
    x[:256, :2] -= .15
    x[256:, :2] += .15
    x[:256, 2:] += .15
    x[256:, 2:] -= .15
    report = acceptance_uncertainty_report(x, policy=policy())
    assert abs(report["pooled_mean"]-baseline["pooled_mean"]) < 1e-12
    assert report["experimental_decision"] == "material_conflict"
    assert report["temporal_conflict"]
    noisy = np.random.default_rng(192).uniform(.45, .95, (512, 4))
    r = acceptance_uncertainty_report(noisy, policy=policy())
    assert r["experimental_decision"] in {"unresolved_contrasts", "insufficient_information"}
    assert not r["temporal_compatible"]


@pytest.mark.parametrize("case", ["common_drift", "single_drift", "permanent_offset", "oscillation"])
def test_material_changes_across_chains_or_time(case):
    x = trace()
    if case == "common_drift":
        x[:256] -= .15
        x[256:] += .15
    elif case == "single_drift":
        x[:256, 0] -= .15
        x[256:, 0] += .15
    elif case == "permanent_offset":
        x[:, 0] -= .2
    else:
        # Early/late halves have equal means; four-window contrasts expose it.
        x[:128] -= .15
        x[128:256] += .15
        x[256:384] -= .15
        x[384:] += .15
    assert acceptance_uncertainty_report(x, policy=policy())["experimental_decision"] == "material_conflict"


@pytest.mark.parametrize("offset,expected", [(-.6, "repair_step_lower"), (.27, "repair_step_higher")])
def test_near_probability_boundaries_and_direction(offset, expected):
    r = acceptance_uncertainty_report(trace()+offset, policy=policy())
    assert r["experimental_decision"] == expected
    assert all(np.isfinite(r["pooled_interval"]))


def test_chain_permutation_time_reversal_and_policy_roundtrip():
    # Both lugsail batch sizes at both sensitivity scales must divide the
    # window, otherwise reversal changes which incomplete batch is discarded.
    x = trace(1536)
    a = acceptance_uncertainty_report(x, policy=policy(batch_size=12))
    b = acceptance_uncertainty_report(x[::-1, [2, 0, 3, 1]], policy=policy(batch_size=12))
    assert a["experimental_decision"] == b["experimental_decision"]
    np.testing.assert_allclose(a["pooled_interval"], b["pooled_interval"], atol=1e-12)
    payload = json.loads(json.dumps(policy().payload(), allow_nan=False))
    assert AcceptanceUncertaintyPolicy.from_payload(payload) == policy()
    payload["tuning_artifact_authority"] = True
    with pytest.raises(ValueError, match="mismatch"):
        AcceptanceUncertaintyPolicy.from_payload(payload)


def test_larger_multiplicity_budget_cannot_narrow_intervals():
    a = acceptance_uncertainty_report(trace(), policy=policy())
    b = acceptance_uncertainty_report(trace(), policy=policy(planned_looks=3, planned_candidates=10))
    assert b["pooled_interval"][0] < a["pooled_interval"][0]
    assert b["pooled_interval"][1] > a["pooled_interval"][1]


def evaluate(x, **kwargs):
    n = len(x)
    kwargs.setdefault("fixed_kernel", True)
    return evaluate_acceptance_uncertainty(samples=np.broadcast_to(
        np.arange(n)[:, None, None], (n, 4, 1)), log_accept_ratio=np.log(x),
        is_accepted=np.ones((n, 4), bool), uncertainty_policy=policy(), **kwargs)


def test_legacy_decision_codec_and_health_veto_are_preserved():
    from bayesfilter.inference.hmc_verification import hmc_acceptance_evidence_from_payload
    r = evaluate(trace(), native_divergence_status="available", native_divergence_count=1)
    legacy = hmc_acceptance_evidence_from_payload(json.loads(json.dumps(r["legacy_evidence"])))
    assert legacy.acceptance_decision == "passed"
    assert not legacy.passed
    assert r["recommendation"] == "blocked_by_health_or_cost"
    assert not r["tuning_artifact_authority"]
    assert evaluate(trace(), fixed_kernel=False)["recommendation"] == "unavailable_invalid_or_nonfixed_kernel"


@pytest.mark.parametrize("mutation", ["stuck", "cycle", "nan_state", "nan_ratio", "truncated"])
def test_health_and_incomplete_traces_cannot_be_rescued_by_acceptance(mutation):
    n = 16 if mutation == "truncated" else 512
    x = trace(n)
    states = np.broadcast_to(np.arange(n)[:, None, None], (n, 4, 1)).copy().astype(float)
    if mutation == "stuck":
        states[:] = 1
    elif mutation == "cycle":
        states[::2] = 1
        states[1::2] = -1
    elif mutation == "nan_state":
        states[1, 1, 0] = np.nan
    elif mutation == "nan_ratio":
        x[1, 1] = np.nan
    r = evaluate_acceptance_uncertainty(samples=states, log_accept_ratio=np.log(x),
        is_accepted=np.ones((n, 4), bool), uncertainty_policy=policy(), fixed_kernel=True)
    assert r["recommendation"] != "compatible_for_fresh_verification"


def test_support_rejection_binary_rates_and_nonfinite_rules():
    x = trace()
    ratios = np.log(x)
    mask = np.zeros_like(x, bool)
    mask[::20] = True
    ratios[mask] = -np.inf
    states = np.broadcast_to(np.arange(512)[:, None, None], (512, 4, 1))
    r = evaluate_acceptance_uncertainty(samples=states, log_accept_ratio=ratios,
        is_accepted=~mask, support_rejection=mask, uncertainty_policy=policy(), fixed_kernel=True)
    expected = np.where(mask, 0., x).mean()
    assert r["uncertainty"]["pooled_mean"] == pytest.approx(expected)
    assert r["legacy_evidence"]["realized_acceptance_rate"] != pytest.approx(expected)
    with pytest.raises(ValueError, match="support rejection"):
        evaluate_acceptance_uncertainty(samples=states, log_accept_ratio=ratios,
            is_accepted=np.ones_like(mask), support_rejection=mask, uncertainty_policy=policy(), fixed_kernel=True)


def test_probability_shape_bounds_and_cpu_graph_parity():
    p = policy()
    a = acceptance_uncertainty_report(trace(), policy=p)
    b = acceptance_uncertainty_report(trace(), policy=replace(p, jit_compile=False))
    np.testing.assert_allclose(a["pooled_interval"], b["pooled_interval"], atol=1e-12)
    for bad in (trace()-1, trace()+1, trace()*np.nan):
        assert acceptance_uncertainty_report(bad, policy=p)["experimental_decision"] == "invalid_probability_trace"
    with pytest.raises(ValueError, match="draw, chain"):
        acceptance_uncertainty_report(trace()[:, 0], policy=p)


def test_short_constant_and_nearly_constant_do_not_imply_exact_certainty():
    r = acceptance_uncertainty_report(np.full((512, 4), .7), policy=policy())
    assert r["experimental_decision"] == "insufficient_information"
    assert r["pooled_interval"] == [None, None]
    x = .7+(trace()-.7)*1e-8
    r = acceptance_uncertainty_report(x, policy=policy())
    assert r["variance_estimate_available"]
    assert r["pooled_interval"][1] > r["pooled_interval"][0]


def test_identical_seed_trace_replication_cannot_reduce_uncertainty():
    x = np.repeat(trace()[:, :1], 4, axis=1)
    r = acceptance_uncertainty_report(x, policy=policy())
    assert r["duplicate_chain_traces"]
    assert r["experimental_decision"] == "insufficient_information"


def test_current_paired_screen_misses_exactly_opposing_drifts():
    from bayesfilter.inference.hmc_verification import temporal_block_conflicts
    blocks = tf.constant([[[.5, .5, .9, .9], [.9, .9, .5, .5],
                           [.5, .5, .9, .9], [.9, .9, .5, .5]]], tf.float64)
    assert not bool(temporal_block_conflicts(blocks, practical_width=.1)[0])


@pytest.mark.parametrize("method,schema", [
    ("raw_block_crossing_v1", "v5"),
    ("paired_chain_block_contrasts_v1", "v6"),
])
def test_temporal_policy_roundtrips_through_execution_config(method, schema):
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
    from bayesfilter.inference.hmc_verification import (
        HMCAcceptancePolicy, _acceptance_policy_from_payload,
    )
    acceptance_policy = HMCAcceptancePolicy(temporal_conflict_method=method)
    payload = acceptance_policy.payload()
    assert payload["schema"] == f"bayesfilter.hmc_acceptance_policy.{schema}"
    assert _acceptance_policy_from_payload(json.loads(json.dumps(payload))) == acceptance_policy
    config = HMCCandidateExecutionConfig(
        measurement_num_results=64, verification_num_results=64,
        num_warmup_steps=0, seed=(11, 9), acceptance_policy=acceptance_policy,
        target_status_trace_policy="none",
    )
    assert HMCCandidateExecutionConfig.from_payload(config.payload()) == config
    payload["schema"] = "bayesfilter.hmc_acceptance_policy.v6" if schema == "v5" else "bayesfilter.hmc_acceptance_policy.v5"
    with pytest.raises(ValueError):
        _acceptance_policy_from_payload(payload)


def test_optional_screen_separates_common_drift_from_noisy_single_chain():
    from bayesfilter.inference.hmc_verification import temporal_block_conflicts
    blocks = tf.constant([
        [[.68, .79, .635, .768], [.7, .7, .7, .7],
         [.69, .71, .70, .70], [.71, .69, .70, .70]],
        [[.2, .4, .8, .95]] * 4,
    ], tf.float64)
    assert temporal_block_conflicts(blocks, practical_width=.1).numpy().tolist() == [False, True]


def test_multivariate_and_replication_axes_are_not_pooled_as_chains():
    x = np.random.default_rng(526).normal(size=(2, 512, 3, 2))
    all_rows = chain_batch_covariance(x, batch_size=24, min_batches=8)
    for i in range(2):
        single = chain_batch_covariance(x[i], batch_size=24, min_batches=8)
        np.testing.assert_allclose(all_rows["long_run_covariance_by_chain"][i],
                                   single["long_run_covariance_by_chain"], atol=1e-12)


def test_exact_finite_markov_reference_has_the_correct_small_n_variance():
    from bayesfilter.testing.acceptance_uncertainty_validation import exact_mean_variance
    assert exact_mean_variance(2, .8, .2) == pytest.approx(.2**2*(1+.8)/2)
    assert exact_mean_variance(512, 0., .2) == pytest.approx(.2**2/512)
    assert exact_mean_variance(512, .995, .2) > 100*exact_mean_variance(512, 0., .2)


def test_confidence_below_float_resolution_is_rejected():
    with pytest.raises(ValueError, match="resolution"):
        acceptance_uncertainty_report(trace(), policy=policy(family_alpha=1e-30))
