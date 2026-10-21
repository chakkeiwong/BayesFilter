"""Known-law generator and preflight failures for AR2 uncertainty qualification."""
import copy

import pytest
import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_closure_program import jobs
from bayesfilter.testing.surrogate_hmc_ar2_calibration import (
    calibration_budget_check, calibration_protocol, confirmation_evidence,
    reference_generator, validate_cost_contract,
)


def cost_fixture():
    return dict(qualified=True, status="AR2_COMPLETE_TRANSITION_COST_MEASURED",
        precision=dict(precision_forecast_sufficient=True, retained_looks=[4000, 8000]),
        conservative_inference_forecast={arm: dict(cpu_worker_seconds=2000., wall_seconds=2100.)
                                         for arm in ("A", "B")})


def test_missing_precision_and_unaffordable_chains_cannot_launch_calibration():
    cost = cost_fixture()
    validate_cost_contract(cost)
    for field in ("qualified", "precision"):
        invalid = copy.deepcopy(cost)
        if field == "qualified":
            invalid[field] = False
        else:
            invalid[field]["precision_forecast_sufficient"] = False
        with pytest.raises(ValueError, match="feasible precision"):
            validate_cost_contract(invalid)
    invalid = copy.deepcopy(cost)
    invalid["conservative_inference_forecast"]["B"]["cpu_worker_seconds"] = float("nan")
    with pytest.raises(ValueError, match="positive"):
        validate_cost_contract(invalid)
    with pytest.raises(ValueError, match="does not fit"):
        calibration_budget_check(dict(previous_charge=dict(cpu_worker_seconds=52000., gpu_seconds=0.)), cost)
    assert calibration_budget_check(dict(previous_charge=dict(cpu_worker_seconds=30000., gpu_seconds=30000.)), cost)


def test_grid_generator_preserves_support_and_refresh_dependence():
    arrays = dict(axis=tf.constant([-1., 0., 1.], tf.float64),
        density=tf.ones([3, 3], tf.float64), predictive=tf.ones([3, 3, 2], tf.float64)*.5,
        means=tf.zeros([2], tf.float64), scales=tf.ones([2], tf.float64),
        cuts=tf.constant([[-.5, 0., .5], [-.5, 0., .5]], tf.float64))
    generate = reference_generator(1000, arrays)
    seed = tf.constant([31, 42], tf.int32)
    points, values = generate(seed, tf.constant(1., tf.float64), tf.constant(0, tf.int32))
    tf.debugging.assert_equal(points, tf.broadcast_to(points[:, :1], [4, 1000, 2]))
    tf.debugging.assert_equal(values, tf.broadcast_to(values[:, :1], [4, 1000, 13]))
    fresh, fresh_values = generate(seed, tf.constant(0., tf.float64), tf.constant(0, tf.int32))
    assert bool(tf.reduce_any(fresh[:, 1:] != fresh[:, :-1]))
    tf.debugging.assert_less_equal(tf.reduce_max(tf.abs(fresh)), tf.constant(1., tf.float64))
    tf.debugging.assert_equal(fresh_values[:, :, :2], fresh)
    tf.debugging.assert_equal(fresh_values[:, :, -2:], tf.ones([4, 1000, 2], tf.float64)*.5)
    shifted, shifted_values = generate(seed, tf.constant(0., tf.float64), tf.constant(1, tf.int32))
    tf.debugging.assert_near(shifted[:, :, 0]-fresh[:, :, 0],
        tf.broadcast_to(tf.constant([-.8, -.25, .25, .8], tf.float64)[:, None], [4, 1000]))
    tf.debugging.assert_equal(shifted_values[:, :, -2:], fresh_values[:, :, -2:])


def test_calibration_is_an_isolated_checked_master_stage():
    selected = jobs("ar2-calibration")
    assert [row["name"] for row in selected] == ["review", "ar2-calibration"]
    assert "tests/test_surrogate_hmc_ar2_calibration.py" in selected[0]["command"]
    with pytest.raises(ValueError, match="inspect scoped uncertainty"):
        jobs("ar2-calibration,wrapped")
    assert [row["name"] for row in jobs("ar2-calibration-confirm")] == ["review", "ar2-calibration-confirm"]
    repair = jobs("ar2-calibration-repair")
    assert [row["name"] for row in repair] == ["review", "ar2-calibration-repair"]
    assert "tests/test_surrogate_hmc_ar2_assessment.py" in repair[0]["command"]
    repair_r2 = jobs("ar2-calibration-repair-r2")
    assert [row["name"] for row in repair_r2] == ["review", "ar2-calibration-repair-r2"]
    assert "tests/test_surrogate_hmc_ar2_assessment.py" in repair_r2[0]["command"]
    with pytest.raises(ValueError, match="fresh calibration"):
        calibration_budget_check(dict(previous_charge={}), cost_fixture(), confirmation=True, repair=True)


def test_confirmation_cannot_hide_another_failed_regime_or_change_schedule():
    protocol = calibration_protocol([4000, 8000])
    previous = dict(status="AR2_UNCERTAINTY_REPAIR_REQUIRED", protocol=protocol,
        groups={name: dict(passed=name != "refresh_nine_tenths") for name, *_ in protocol["groups"]})
    assert len(confirmation_evidence(previous, [4000, 8000])) == 5
    with pytest.raises(ValueError, match="precision shortfall"):
        confirmation_evidence(previous, [2000, 4000])
    previous["groups"]["iid"]["passed"] = False
    with pytest.raises(ValueError, match="precision shortfall"):
        confirmation_evidence(previous, [4000, 8000])
