"""Actual K0--K7 filters and independently implemented reference checks.

CPU graph tests are mechanics evidence. The separate GPU preflight must check
placement/XLA and complete-fit cost before a campaign can run.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.ssm_campaign_profiles import PROFILES, generate_data
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.references import ssm


def _target(name, **kwargs):
    data = None if PROFILES[name].case == "K6" else generate_data(name, (20260926, 71))
    return ValidationTarget(name, data=data, jit_compile=False, **kwargs)


@pytest.mark.parametrize("name", list(PROFILES))
def test_campaign_real_filter_value_and_total_score_against_independent_reference(name):
    target = _target(name)
    q = np.array(target._ssm.prior_mean)
    if PROFILES[name].case == "K2": q[0] = np.arctanh(.97/.999)
    if PROFILES[name].case == "K3": q[1] = np.log(.02)
    value, score = target.log_prob_and_grad(q)
    reference = ssm.log_density(name, q, target.data)
    np.testing.assert_allclose(value.numpy(), reference, rtol=1e-6, atol=1e-6)
    for h in (1.e-5, 2.e-5):
        eye = h*np.eye(len(q))
        finite_difference = (ssm.log_density(name, q+eye, target.data)-ssm.log_density(name, q-eye, target.data))/(2*h)
        np.testing.assert_allclose(score.numpy(), finite_difference, rtol=2e-5, atol=2e-5)
    status = target.target_status_telemetry(q)
    assert bool(status["valid_pre_regularized_score"])
    assert target.parameter_names() == PROFILES[name].raw_names
    assert target.spec.parameters == PROFILES[name].names


@pytest.mark.parametrize("name", [n for n,p in PROFILES.items() if p.case not in {"K6", "K7"}])
def test_kalman_observation_marginal_and_innovations_are_same_law(name):
    target = _target(name)
    q = np.array(target._ssm.prior_mean)
    if PROFILES[name].case == "K2": q[0] = np.arctanh(.97/.999)
    if PROFILES[name].case == "K3": q[1] = np.log(.02)
    np.testing.assert_allclose(ssm.scalar_likelihood(name,q,target.data),
        ssm.dense_scalar_likelihood(name,q,target.data), rtol=1e-12, atol=1e-10)


def test_campaign_profiles_reject_silent_model_or_data_changes():
    name = "ssm_campaign_interior"
    target = _target(name)
    with pytest.raises(ValueError, match="frozen campaign"):
        ValidationTarget(name, {"rho": .9}, target.data, jit_compile=False)
    with pytest.raises(ValueError, match="shape"):
        ValidationTarget(name, data=target.data[:-1], jit_compile=False)
    changed = ValidationTarget(name, data=[target.data[0]+.1, *target.data[1:]], jit_compile=False)
    assert changed.adapter_signature() != target.adapter_signature()
    assert not np.allclose(changed.log_prob_and_grad(target._ssm.prior_mean)[0],
                           target.log_prob_and_grad(target._ssm.prior_mean)[0])


def test_campaign_event_dimension_survives_shared_filter_shape_generalization():
    # Shared QR functions see two parameter dimensions before the scalar
    # location case. A generalized inner graph must not erase its public shape.
    for name in ("ssm_lgssm_qr", "ssm_campaign_interior", "ssm_campaign_location"):
        target = _target(name) if name.startswith("ssm_campaign_") else ValidationTarget(name,jit_compile=False)
        width = target.parameter_dim
        graph = tf.function(target.log_prob_and_grad,
            input_signature=[tf.TensorSpec([4,width],tf.float64)],autograph=False)
        concrete = graph.get_concrete_function()
        assert concrete.output_shapes == (tf.TensorShape([4]),tf.TensorShape([4,width]))


def test_reference_grid_refuses_accuracy_if_sensitivity_fails():
    name = "ssm_campaign_interior"
    data = generate_data(name, (20260926,71))
    draws, metadata = ssm.posterior_reference(name, 64, 21, data, resolution=41, sensitivity=(1e-14,1e-14))
    assert draws is None
    assert metadata["checked"] is False
    assert metadata["expanded_radius"] > metadata["prior_sd_radius"]
    assert metadata["coarse_resolution"] < metadata["resolution"]


def test_nonlinear_repeated_evaluation_reuses_stable_graph_and_actual_status():
    target = _target("ssm_campaign_nonlinear")
    q = tf.constant([.70, .80], tf.float64)
    first = target.log_prob_and_grad(q)
    second = target.log_prob_and_grad(q+.001)
    assert not np.array_equal(first[0].numpy(), second[0].numpy())
    assert target._ssm._graphs[1].experimental_get_tracing_count() == 1
    status = target.target_status_telemetry(q)
    assert float(status["min_innovation_eigenvalue"]) > 0
    assert int(status["floor_count_value"]) == 0


@pytest.mark.parametrize("case", ["K0", "K1", "K4", "K6", "K7"])
def test_campaign_xla_matches_graph_and_reuses_compilation(case):
    name = next(n for n, p in PROFILES.items() if p.case == case)
    baseline = _target(name)
    compiled = ValidationTarget(name, data=baseline.data, jit_compile=True)
    q = baseline._ssm.initial_starts()
    expected = baseline.log_prob_and_grad(q)
    actual = compiled.log_prob_and_grad(q)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-5, atol=2e-5)
    compiled.log_prob_and_grad(q + tf.constant(.0001, tf.float64))
    assert compiled._ssm._graphs[4].experimental_get_tracing_count() == 1
    assert compiled.target_status_telemetry(q)["valid_pre_regularized_score"].numpy().all()


@pytest.mark.parametrize("control,field", [("wrong_score","score_passed"), ("ignore_data","density_passed")])
def test_actual_filter_defects_activate_specific_independent_checks(tmp_path, control, field):
    from bayesfilter.testing.inference_validation.designs import ValidationDesign, ScenarioSpec
    from bayesfilter.testing.inference_validation.engines.mechanics import run
    name = "ssm_campaign_interior"
    d = ValidationDesign(design_id="ssm-defect", engine="mechanics",
        scenario=ScenarioSpec(name,"frozen",control), replications=2, draws=64,
        seed=617, budget_seconds=120, purpose="activated filter defect regression",
        numerical_provenance="independent dense/scalar filtering reference", device="cpu_reference",
        options={"data":generate_data(name,(20260926,71))})
    report = run(d,tmp_path)
    assert report["finding"] == "mechanics_discrepancy"
    assert report[field] is False
