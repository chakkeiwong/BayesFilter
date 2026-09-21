"""Target-law and public-dispatch checks for a controlled full-fit defect."""
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.designs import ScenarioSpec


@pytest.mark.parametrize("severity", [0., .25, .5, -.5])
def test_translation_has_declared_normal_mode_curvature_and_density(severity):
    data = [2., -1., .5]
    params = {"tau": 2., "sigma": 1., "location_shift_posterior_sd": severity}
    target = ValidationTarget("normal_conjugate", params, data, "location_shift", jit_compile=False)
    variance = 1 / (.25 + 3)
    mode = variance * sum(data) + severity * math.sqrt(variance)
    q = np.array([[mode - .7], [mode], [mode + .9]])
    value, score = target.log_prob_and_grad(q)
    np.testing.assert_allclose(score, -(q-mode)/variance, atol=1e-13)
    np.testing.assert_allclose(value - value[1], -.5*(q[:,0]-mode)**2/variance, atol=1e-13)
    baseline = ValidationTarget("normal_conjugate", params, data, jit_compile=False)
    assert target.adapter_signature() != baseline.adapter_signature()
    if severity == 0.:
        ref = baseline.log_prob_and_grad(q)
        np.testing.assert_array_equal(value, ref[0])
        np.testing.assert_array_equal(score, ref[1])


@pytest.mark.parametrize("target,data,severity", [
    ("gaussian", None, .5), ("normal_conjugate", None, .5),
    ("normal_conjugate", [1.], None), ("normal_conjugate", [1.], float("nan")),
])
def test_unactivated_mutation_is_rejected(target, data, severity):
    with pytest.raises(ValueError, match="location_shift"):
        ValidationTarget(target, {"location_shift_posterior_sd":severity}, data,
                         "location_shift", jit_compile=False)


def test_reference_route_cannot_silently_ignore_translation(design):
    with pytest.raises(ValueError, match="location_shift"):
        design("sbc", "normal_conjugate", "reference", "location_shift",
               scenario=ScenarioSpec("normal_conjugate", "reference", "location_shift",
                                     parameters={"location_shift_posterior_sd":.5}))


def test_public_ordinary_preparation_receives_translated_target(design, tmp_path, monkeypatch):
    import bayesfilter.inference as public
    from bayesfilter.testing.inference_validation.procedures import execute_pipeline
    def inspect(**kwargs):
        target = kwargs["adapter"]
        assert target.control == "location_shift"
        _, score = target.log_prob_and_grad(tf.constant([[1.8]], tf.float64))
        # For tau=2,sigma=1,n=6,y=2: intended mean 1.92, SD .4; shifted mean 2.12.
        assert float(score[0,0]) == pytest.approx((2.12-1.8)/.16)
        assert kwargs["search_config"] is None
        raise RuntimeError("checked actual public preparation boundary")
    monkeypatch.setattr(public, "tune_hmc_kernel", inspect)
    d = design("sbc", "normal_conjugate", "ordinary", "location_shift",
        scenario=ScenarioSpec("normal_conjugate", "ordinary", "location_shift",
            parameters={"tau":2.,"sigma":1.,"location_shift_posterior_sd":.5}),
        l_grid=(3,5,9,13,18,25),options={"native_search":True})
    with pytest.raises(RuntimeError, match="checked actual public"):
        execute_pipeline(d, tmp_path, data=[2.]*6)
