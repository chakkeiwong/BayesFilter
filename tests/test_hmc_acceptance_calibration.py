"""Independent oracle and controller checks for the diagnostic calibration driver."""
import pytest

from bayesfilter.testing.acceptance_decision_calibration import binomial_interval, search_once, truth, window_law
from tests.test_hmc_acceptance_protocol import policy


def test_exact_binomial_bounds_have_known_zero_and_all_success_limits():
    assert binomial_interval(0, 100, .05) == pytest.approx((0., 1-.025**.01), abs=1e-14)
    assert binomial_interval(100, 100, .05) == pytest.approx((.025**.01, 1.), abs=1e-14)
    lo, hi = binomial_interval(30, 100, .01)
    other_lo, other_hi = binomial_interval(70, 100, .01)
    assert (lo, hi) == pytest.approx((1-other_hi, 1-other_lo), abs=1e-14)


def test_transient_truth_is_the_actual_finite_horizon_quantity():
    expected = .7+.25*(1-.995**65)/(65*(1-.995))
    assert truth("initial_transient", 1.3, 3, 65) == pytest.approx((expected,)*4)
    assert expected > .85  # Comparing against stationary .70 would mislabel errors.
    discarded = truth("initial_transient", 1.3, 3, 65, discarded_prefix=3)[0]
    assert discarded == pytest.approx(.7+(expected-.7)*.995**3)


@pytest.mark.parametrize("cell", ["interior", "misleading_pooled", "opposite_l_repairs"])
def test_calibration_uses_all_retained_members_and_real_repair_children(cell):
    p = policy(base_repetitions=32, max_repetitions=128, max_candidates=8)
    row = search_once(cell=cell, index=0, namespace="unit-diagnostic-20261002", policy=p, rungs=(1, 2, 4))
    assert not row["artifact_authority"]
    assert not row["false_verified"]
    assert not row["search_direction_error"]
    if cell == "interior":
        assert row["verified_count"] == 2
        assert row["useful_delivery"]
    elif cell == "misleading_pooled":
        assert row["verified_count"] == 0
        assert set(row["candidate_states"].values()) == {"preparation_review_required"}
    else:
        directions = {o["decision"] for o in row["observations"]}
        assert {"repair_step_lower", "repair_step_higher"} <= directions
        assert row["candidate_count"] > 2
        assert row["verified_count"] >= 2


def test_full_cap_means_one_hundred_actual_candidates_with_fresh_verification():
    p = policy(base_repetitions=256, max_repetitions=256, max_candidates=100)
    row = search_once(cell="constant_law", index=0, namespace="hundred-actual-pairs-regression",
                      policy=p, rungs=(1,), initial_candidate_count=100)
    assert row["candidate_count"] == row["initial_candidate_count"] == 100
    assert row["verified_count"] == 100
    assert row["useful_delivery"]
    seen = {(o["candidate_id"], o["stage"]) for o in row["observations"]}
    assert len(seen) == 200
    assert {o["stage"] for o in row["observations"]} == {"measurement", "verification"}
    assert len({tuple(o["seed"]) for o in row["observations"]}) == 200
    assert row["observed_trial_vectors"] == row["generated_trial_vectors"] == 200*256


@pytest.mark.parametrize("cell,expected", [
    ("temporal_stationary", [False]*4), ("temporal_opposing", [True]*4),
    ("temporal_single_start", [True, False, False, False]), ("temporal_nonlinear", [True]*4),
])
def test_per_start_temporal_oracle_detection_does_not_change_admission(cell, expected):
    p = policy(trial_num_results=65, base_repetitions=256, max_repetitions=256, max_candidates=8)
    row = search_once(cell=cell, index=0, namespace="window-oracle-regression", policy=p, rungs=(1,))
    assert row["verified_count"] == 2
    assert not row["false_preparation"]
    assert not row["false_temporal_alert"]
    for observation in row["observations"]:
        assert observation["expected_temporal_alerts"] == expected
        assert observation["temporal_alerts"] == expected
    if any(expected):
        assert row["temporal_detection"]
    else:
        assert row["temporal_detection"] is None
    # Deliberately odd T: the final window contains 17, not 16, observations.
    exact = [sum(n*m for n,m in zip([16,16,16,17],w))/65 for w in window_law(cell)]
    assert truth(cell,1.3,1,65) == pytest.approx(exact)


@pytest.mark.parametrize("cell", ["shared_starts", "opposed_starts", "rare_two_sided",
                                  "persistent_0", "persistent_08", "persistent_095", "persistent_negative_08"])
def test_new_laws_keep_frozen_trial_expectation_and_explicit_work_counts(cell):
    p = policy(base_repetitions=32, max_repetitions=128, max_candidates=8)
    row = search_once(cell=cell, index=1, namespace="dependence-regression", policy=p, rungs=(1,2,4))
    assert truth(cell,1.3,1,65) == (.7,)*4
    assert not row["false_verified"]
    assert not row["search_direction_error"]
    assert not row["false_preparation"]
    observed = sum(o["trial_range"][1]-o["trial_range"][0] for o in row["observations"])
    pairs = {(o["candidate_id"], o["stage"]) for o in row["observations"]}
    assert row["observed_trial_vectors"] == observed
    assert row["generated_trial_vectors"] == len(pairs)*128 >= observed


def test_shared_and_opposed_start_laws_have_actual_within_trial_dependence():
    import numpy as np
    import tensorflow as tf
    from bayesfilter.testing.acceptance_decision_calibration import _generator
    for cell in ("shared_starts", "opposed_starts"):
        scores = _generator(128,65,cell,3)(tf.constant([3,7],tf.int32),tf.constant([.7]*4,tf.float64)).numpy()
        np.testing.assert_array_equal(scores[:,0],scores[:,2])
        if cell == "shared_starts":
            np.testing.assert_array_equal(scores[:,0],scores[:,1])
        else:
            np.testing.assert_allclose(scores[:,0]+scores[:,1],1.4,atol=1e-14,rtol=0)


def test_invalid_full_cohort_is_rejected_before_sampling():
    p = policy(max_candidates=8)
    for count in (1,9,True):
        with pytest.raises(ValueError,match="cohort"):
            search_once(cell="interior",index=0,namespace="invalid",policy=p,rungs=(1,),initial_candidate_count=count)
