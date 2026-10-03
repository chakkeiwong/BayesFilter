import numpy as np
import pytest
from scipy import stats

from bayesfilter.testing.inference_validation.catalog import TARGETS
from bayesfilter.testing.inference_validation.engines.mechanics import run
from bayesfilter.testing.inference_validation.references import analytic
from bayesfilter.testing.inference_validation.targets import ValidationTarget


@pytest.mark.parametrize("target", [t for t, spec in TARGETS.items() if spec.available])
def test_every_target_density_score_against_independent_reference(design, tmp_path, target):
    result = run(design(target=target), tmp_path)
    assert result["finding"] == "mechanics_passed", result


@pytest.mark.parametrize("target,data", [("normal_conjugate", [2., -1., 3.]),
    ("beta_binomial", [5, 9]), ("lgssm_location", [1., .1, 2., -1.])])
def test_likelihood_including_kalman_recursion(design, tmp_path, target, data):
    result = run(design(target=target, options={"data": data}), tmp_path)
    assert result["finding"] == "mechanics_passed", result


@pytest.mark.parametrize("target", ["gamma", "beta", "dirichlet", "beta_binomial"])
def test_coordinate_roundtrip_and_model_dimension(target):
    q = np.array([[.2, -.3], [1., 2.]]) if target == "dirichlet" else np.array([[.2], [-.7]])
    ref = analytic.model_coordinates(target, q)
    np.testing.assert_allclose(analytic.active_coordinates(target, ref), q, atol=1e-14)
    actual = ValidationTarget(target, jit_compile=False)
    np.testing.assert_allclose(actual.to_model(q), ref, atol=1e-14)
    assert len(actual.spec.parameters) == ref.shape[-1]
    assert len(actual.parameter_names()) == q.shape[-1]


def test_dirichlet_reference_contains_full_jacobian():
    q = np.array([[.2, -.3], [1., 2.]])
    p = analytic.model_coordinates("dirichlet", q)
    expected = stats.dirichlet.logpdf(p.T, [2, 3, 4]) + np.log(p).sum(-1)
    np.testing.assert_allclose(analytic.log_density("dirichlet", q), expected, atol=1e-14)


@pytest.mark.parametrize("target,params", [("gamma", {"rate": 0}),
    ("lgssm_location", {"rho": 1.}), ("dirichlet", {"concentration": [1, 2]}),
    ("student_t", {"df": 2}), ("normal_conjugate", {"n": 2.5})])
def test_invalid_laws_are_rejected(target, params):
    with pytest.raises(ValueError):
        ValidationTarget(target, params, jit_compile=False)


@pytest.mark.parametrize("target,control", [("gaussian", "wrong_score"), ("gamma", "omit_jacobian"),
    ("beta", "omit_jacobian"), ("dirichlet", "omit_jacobian")])
def test_target_defects_activate_and_are_detected(design, tmp_path, target, control):
    result = run(design(target=target, control=control), tmp_path)
    assert result["finding"] == "mechanics_discrepancy"
    assert not result["score_passed"]


def test_noop_is_numerically_identical(design, tmp_path):
    baseline = run(design(), tmp_path / "baseline")
    noop = run(design(control="noop"), tmp_path / "noop")
    assert baseline["probes"] == noop["probes"]
    assert baseline["score_relative_error"] == noop["score_relative_error"]
