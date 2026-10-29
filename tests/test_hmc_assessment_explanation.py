"""CPU diagnostic checks of posterior explanations and unchanged decisions."""
import json

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.inference.hmc_posterior_assessment import (
    HMCPosteriorAssessmentPolicy, assess_posterior,
)
from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget


def assess(draws, *, policy=None, stage="warmup", extra=None, quantities_fn=None):
    rhat = rank_normalized_split_rhat_summary(draws, rhat_max=1.05)
    report = assess_posterior(draws, ("x",), policy=policy, stage=stage,
        rhat=rhat, extra=extra, quantities_fn=quantities_fn)
    # This is the pre-change public decision, independent of the new labels.
    expected = bool(report["modern_rhat"]["passed"] and report["information_passed"]
        and report["precision"]["passed"]
        and (extra is None or (extra["passed"] and not extra.get("hard_vetoes"))))
    assert report["passed"] == expected == all(report["checks"].values())
    json.dumps(report, allow_nan=False)
    return report


@pytest.fixture
def draws():
    # Inherited deterministic reference bank; not a posterior calibration run.
    return tf.random.stateless_normal([2048, 4, 1], (823, 19), dtype=tf.float64)


def test_window_counts_and_disabled_information_floors(draws):
    report = assess(draws)
    assert report["passed"] and not report["failed_checks"]
    assert report["sample_counts"] == {"draws_per_chain": 2048, "chain_count": 4,
        "total_draws": 8192, "scope": "assessed_window_only"}
    assert report["requirements"]["bulk_ess_min"] == 0.
    assert report["requirements"]["rhat_max"] == 1.05


def test_incompatible_chain_locations_identify_rhat(draws):
    offsets = tf.reshape(tf.constant([-10., -5., 5., 10.], tf.float64), [1, 4, 1])
    report = assess(draws + offsets)
    assert not report["passed"]
    assert report["failed_checks"] == ("rhat",)


def test_information_shortfall_is_separate_from_rhat(draws):
    base = assess(draws)
    policy = HMCPosteriorAssessmentPolicy(
        warmup_bulk_ess_min=2 * base["bulk_ess"][0],
        warmup_tail_ess_min=2 * base["tail_ess"][0], warmup_consecutive_checks=3)
    report = assess(draws, policy=policy)
    assert report["checks"]["rhat"]
    assert report["failed_checks"] == ("bulk_ess", "tail_ess")
    assert report["requirements"]["consecutive_warmup_checks_required"] == 3
    assert report["requirements"]["persistence_scope"] == "enforced_by_sequential_controller"


def test_insufficient_batches_blocks_precision_only_in_retained_phase(draws):
    policy = HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy(
        (HMCPrecisionTarget("x", mcse_absolute_max=1.),), method="lugsail",
        batch_size=2048, min_batches=20, jit_compile=False))
    warmup = assess(draws, policy=policy)
    retained = assess(draws, policy=policy, stage="retained")
    assert warmup["passed"] and not warmup["requirements"]["precision_required"]
    assert retained["failed_checks"] == ("precision",)
    assert retained["precision"]["batch_metadata"]["unavailable_reason"] == "insufficient_complete_batches"


def test_binary_missing_outcome_reported_with_disabled_ess_floors(draws):
    policy = HMCPosteriorAssessmentPolicy(quantities_id="event-v1",
        binary_quantity_names=("event",))
    report = assess(draws, policy=policy,
        quantities_fn=lambda x: {"event": tf.zeros(x.shape[:2], tf.float64)})
    assert "binary_outcomes_observed" in report["failed_checks"]
    assert report["checks"]["bulk_ess"] and report["checks"]["tail_ess"]
    assert report["quantity_names"] == ("x", "event")


def test_consumer_veto_is_explained_and_cannot_be_overridden(draws):
    report = assess(draws, extra={"passed": True, "hard_vetoes": ("reference_failed",)})
    assert report["failed_checks"] == ("consumer_diagnostic",)


def test_unassessable_windows_explain_early_return(draws):
    short = assess_posterior(draws[:3], ("x",), policy=None, stage="warmup", rhat={})
    assert short["failed_checks"] == ("minimum_draws",)
    assert short["sample_counts"]["total_draws"] == 12
    invalid = tf.tensor_scatter_nd_update(draws, [[0, 0, 0]], [float("nan")])
    report = assess_posterior(invalid, ("x",), policy=None, stage="warmup", rhat={"passed": False})
    assert report["failed_checks"] == ("finite_monitored_quantities",)
    assert not report["passed"]
