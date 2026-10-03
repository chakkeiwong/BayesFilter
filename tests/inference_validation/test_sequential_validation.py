"""Source equations, independent experiments and actual TF/TFP integration."""
import itertools
import json
import math

import pytest

from bayesfilter.testing.inference_validation.engines.sequential import sequential_test
from bayesfilter.testing.inference_validation.engines import invariance


def run(values, *, dimension=1, **kwargs):
    records = []
    def observe(count, index):
        records.append((count, index))
        return values[index]
    result = sequential_test(observe, alpha=.05, max_looks=3, initial_samples=10,
        sample_multiplier=4, dimension=dimension, **kwargs)
    return result, records


def test_wrapper_rejection_and_nonrejection_boundaries():
    beta = .05/3
    gamma = beta**(1/3)
    rejected, calls = run([[beta]])
    assert rejected["finding"] == "discrepancy_detected" and len(calls) == 1
    passed, calls = run([[math.nextafter(beta + gamma, math.inf)]])
    assert passed["decision"] == "early_nonrejection" and len(calls) == 1
    continued, calls = run([[beta + gamma], [1.]])
    assert len(calls) == 2
    assert continued["looks"][1]["beta"] == pytest.approx(beta/gamma)


def test_source_increases_samples_once_and_corrects_raw_pvalues_once():
    beta = .05/3
    gamma = beta**(1/3)
    values = [[(beta/gamma**i + .01)/2, 1.] for i in range(3)]
    result, calls = run(values, dimension=2)
    assert calls == [(10, 0), (40, 1), (40, 2)]
    assert result["decision"] == "look_cap_without_rejection"
    assert not result["arbitrary_callback_validity_established"]
    assert not result["mixing_established"]


@pytest.mark.parametrize("dimension", [1, 2])
def test_exhaustive_independent_discrete_null_obeys_bound(dimension):
    # Independent uniform ranks on {1,...,8}/8 are superuniform. With d=2
    # the components are deliberately identical; independence within a look
    # is unnecessary. Enumeration gives exact probability, not a flaky test.
    rejected = 0
    for path in itertools.product(range(1, 9), repeat=3):
        result = sequential_test(lambda n, i: [path[i]/8]*dimension,
            alpha=.3, max_looks=3, initial_samples=1, sample_multiplier=1, dimension=dimension)
        rejected += result["decision"] == "reject"
    assert rejected/(8**3) <= .3


@pytest.mark.parametrize("values", [[[float("nan")]], [[-1.]], [[1.01]], [[True]], [[]], [[.4, .5]]])
def test_invalid_pvalues_are_execution_errors_not_rejections(values):
    with pytest.raises(ValueError, match="p-value"):
        run(values)


def test_invariance_wrapper_uses_fresh_complete_looks(design, tmp_path, monkeypatch):
    seen = []
    def single(child, root, deadline):
        seen.append(child)
        assert "sequential" not in child.options
        p = (.05/3/((.05/3)**(1/3))**(len(seen)-1) + .01)/3
        return {"rank_tests": {"bounded_radius": {"p_value": p}},
                "two_sample_tests": {"bounded_radius": {"p_value": 1.}},
                "analytic_tests": {"gaussian_energy": {"p_value": 1.}},
                "finding": "discrepancy_detected"}  # fixed-look finding must be ignored
    monkeypatch.setattr(invariance, "_run_single", single)
    d = design("invariance", replications=8, options={"invariance_quantities": ["bounded_radius"],
        "gaussian_energy_test": True, "sequential": {"max_looks": 3, "sample_multiplier": 2}})
    result = invariance.run(d, tmp_path)
    assert [c.replications for c in seen] == [8, 16, 16]
    assert len(set(c.seed for c in seen)) == 3
    assert len(set(c.design_id for c in seen)) == 3
    assert result["finding"] == "no_discrepancy_detected"
    assert result["dimension"] == 3
    assert json.loads((tmp_path/"sequential-progress.json").read_text())["observations"]


def test_actual_tf_invariance_runs_through_wrapper(design, tmp_path):
    d = design("invariance", replications=16, rank_draws=3,
        options={"invariance_quantities": ["bounded_radius"], "gaussian_energy_test": True,
                 "sequential": {"max_looks": 2, "sample_multiplier": 2}})
    result = invariance.run(d, tmp_path)
    assert 1 <= len(result["looks"]) <= 2
    assert not result["reused_cumulative_samples"]
    for look, observed in zip(result["looks"], result["observations"]):
        stored = json.loads((tmp_path/f'look-{look["look_index"]:03}'/"invariance.json").read_text())
        assert stored["replications"] == look["sample_count"]
        assert list(test["p_value"] for test in observed["raw_tests"].values()) == look["p_values"]


@pytest.mark.parametrize("options", [
    {"sequential": None},
    {"sequential": {"max_looks": 3, "sample_multiplier": 4}},
    {"invariance_quantities": ["bounded_radius"], "sequential": {"max_looks": True, "sample_multiplier": 4}},
    {"invariance_quantities": ["bounded_radius"], "sequential": {"max_looks": 3, "sample_multiplier": .5}},
])
def test_invalid_sequential_design_fails_before_execution(design, options):
    with pytest.raises(ValueError, match="sequential"):
        design("invariance", options=options)


def test_unresolved_mc_pvalues_do_not_enter_sequential_campaign(design):
    with pytest.raises(ValueError, match="null resolution"):
        design("invariance", alpha=1e-5, null_draws=1999,
            options={"invariance_quantities": ["bounded_radius"],
                     "sequential": {"max_looks": 7, "sample_multiplier": 4}})
