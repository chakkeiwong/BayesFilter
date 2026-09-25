"""Independent energy null, multiplicity, and actual experiment wiring."""
import numpy as np
import pytest
from scipy import stats

from bayesfilter.testing.inference_validation.engines.statistics import gaussian_energy_test
from bayesfilter.testing.inference_validation.engines import invariance


def test_energy_null_uses_all_independent_coordinates_and_both_tails():
    # Choose a chi-square quantile: its exact two-sided p-value is known.
    for tail in (.001, .4, .999):
        squared = stats.chi2.ppf(tail, 20)
        endpoints = np.full((10, 2), 2*np.sqrt(squared/20))
        result = gaussian_energy_test(endpoints, scale=2., alpha=.05, multiplicity=3)
        assert result["p_value"] == pytest.approx(2*min(tail, 1-tail), abs=1e-12)
        assert result["threshold"] == .05/3
        assert result["degrees_of_freedom"] == 20
        assert (result["finding"] == "discrepancy_detected") == (tail != .4)


@pytest.mark.parametrize("target,engine,options", [
    ("funnel", "invariance", {"gaussian_energy_test": True}),
    ("gaussian", "mechanics", {"gaussian_energy_test": True}),
    ("gaussian", "invariance", {"invariance_quantities": []}),
    ("gaussian", "invariance", {"invariance_quantities": ["made_up"]}),
    ("gaussian", "invariance", {"invariance_quantities": ["bounded_radius"]*2}),
])
def test_invalid_statistical_contract_rejected_before_execution(design, target, engine, options):
    with pytest.raises(ValueError):
        design(engine, target=target, options=options)


def test_energy_test_is_wired_to_independent_endpoint_and_family(design, tmp_path, monkeypatch):
    received = []
    original = invariance.gaussian_energy_test
    def capture(values, **kwargs):
        received.append(np.array(values))
        return original(values, **kwargs)
    monkeypatch.setattr(invariance, "gaussian_energy_test", capture)
    result = invariance.run(design("invariance", replications=32,
        options={"gaussian_energy_test":True, "invariance_quantities":["bounded_radius"]}), tmp_path)
    assert len(received) == 1 and received[0].shape == (32, 2)
    assert result["multiplicity"] == 3
    assert set(result["rank_tests"]) == set(result["two_sample_tests"]) == {"bounded_radius"}
    for test in [*result["rank_tests"].values(), *result["two_sample_tests"].values(),
                 *result["analytic_tests"].values()]:
        assert test["threshold"] == .05/3
    assert result["analytic_tests"]["gaussian_energy"]["statistic"] == pytest.approx((received[0]**2).sum())
