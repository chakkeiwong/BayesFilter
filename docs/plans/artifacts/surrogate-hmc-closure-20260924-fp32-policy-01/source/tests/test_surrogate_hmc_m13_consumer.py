"""Independent M13 callback identities and executable canonical wiring."""
from dataclasses import replace
from types import MappingProxyType
import math

import pytest
import tensorflow as tf

import bayesfilter.highdim.ledh_range_bearing_target_tf as target_module
from bayesfilter.highdim.ledh_range_bearing_target_tf import (
    make_range_bearing_model, make_range_bearing_target, polar_observation,
    polar_observation_tangent, polar_jacobian, polar_jacobian_tangent)
from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls
from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import build_range_bearing_fixture_tf


def test_polar_geometry_total_directional_derivatives():
    x = tf.constant([[1.2,.7,.18,-.06],[-.8,.01,.2,.3],[.001,-.002,0.,0.]],tf.float64)
    dx = tf.constant([[.1,-.2,.3,.4],[.4,.2,.1,-.1],[.0002,-.0001,.1,.1]],tf.float64)
    h = tf.constant(1e-6,tf.float64)
    tf.debugging.assert_near(polar_observation_tangent(x,dx),
        (polar_observation(x+h*dx)-polar_observation(x-h*dx))/(2*h),atol=2e-9,rtol=2e-9)
    tf.debugging.assert_near(polar_observation_tangent(x,dx),
        tf.einsum("noi,ni->no",polar_jacobian(x),dx),atol=1e-13,rtol=1e-13)
    tf.debugging.assert_near(polar_jacobian_tangent(x,dx),
        (polar_jacobian(x+h*dx)-polar_jacobian(x-h*dx))/(2*h),atol=1e-7,rtol=1e-8)
    assert not bool(tf.reduce_all(tf.math.is_finite(polar_jacobian(tf.zeros([1,4],tf.float64)))))


def test_all_parameter_rows_and_residual_covariance_terms_reach_shared_density(monkeypatch):
    calls = []
    original = target_module.wrapped_gaussian_batch_kernel
    def tracked_factory(**kwargs):
        kernel = original(**kwargs)
        def tracked(*args):
            calls.append(args[1])
            return kernel(*args)
        return tracked
    monkeypatch.setattr(target_module,"wrapped_gaussian_batch_kernel",tracked_factory)
    fixture = build_range_bearing_fixture_tf()
    model = make_range_bearing_model(fixture.A,fixture.Q,fixture.R,dtype=tf.float64)
    x = tf.constant([[1.2,.7,.18,-.06],[1.3,.8,.1,.0],[1.1,.6,.1,-.1]],tf.float64)
    dx = tf.constant([[.01,-.02,.0,.0],[-.03,.01,.0,.0],[.02,.03,.0,.0]],tf.float64)
    theta = tf.constant([[.2,-.4],[-.3,.2],[.5,.6]],tf.float64)
    direction = tf.constant([[.3,.2],[-.4,-.3],[.1,.5]],tf.float64)
    obs = fixture.observations[0]
    value = model.observation_log_density_fn(theta,x,obs)
    tangent = model.observation_log_density_tangent_fn(theta,x,obs,dx,direction)
    h = tf.constant(1e-6,tf.float64)
    plus = model.observation_log_density_fn(theta+h*direction,x+h*dx,obs)
    minus = model.observation_log_density_fn(theta-h*direction,x-h*dx,obs)
    tf.debugging.assert_near(tangent,(plus-minus)/(2*h),atol=1e-7,rtol=1e-7)
    # Independent diagonal normal law at these interior points (other aliases negligible).
    residual = obs-polar_observation(x)
    sigma = tf.sqrt(tf.linalg.diag_part(fixture.R))[None]*tf.exp(theta[:,1,None])
    oracle = tf.reduce_sum(-.5*(residual/sigma)**2-tf.math.log(sigma)-.5*math.log(2*math.pi),-1)
    tf.debugging.assert_near(value,oracle,atol=2e-12,rtol=2e-12)
    assert len(calls) == 4
    assert not bool(tf.reduce_all(calls[0][0] == calls[0][1]))
    qdot = model.process_covariance_tangent_fn(theta[0],direction[0])
    qfd = (model.process_covariance_fn(theta[0]+h*direction[0])
           -model.process_covariance_fn(theta[0]-h*direction[0]))/(2*h)
    tf.debugging.assert_near(qdot,qfd,atol=1e-12,rtol=1e-9)


def test_fixed_design_and_likelihood_only_canonical_consumer_wiring(monkeypatch):
    fixture = build_range_bearing_fixture_tf()
    fixture = replace(fixture, observations=fixture.observations[:2], states=fixture.states[:2])
    target = make_range_bearing_target(fixture,particle_count=16,controls=warm_start_controls(),dtype=tf.float64)
    design = target.controls["reset_design"]
    tf.debugging.assert_near(tf.reduce_mean(design,0),tf.zeros([4],tf.float64),atol=2e-8,rtol=0.)
    tf.debugging.assert_near(tf.transpose(design)@design/16,tf.eye(4,dtype=tf.float64),atol=8e-8,rtol=0.)
    calls = []
    def canonical(model,theta,directions,**kwargs):
        calls.append((model,theta,directions,kwargs))
        value = tf.constant([1.25,-2.5],tf.float64)
        score = tf.constant([[3.,4.],[5.,6.]],tf.float64)
        ok = tf.constant([True,True])
        return value,score,dict(value_valid=ok,force_valid=ok,program_valid=ok)
    monkeypatch.setattr(target_module,"canonical_batch_fused_value_score",canonical)
    points = tf.constant([[.2,-.3],[-.1,.4]],tf.float64)
    value,score,status = target.evaluate(points)
    tf.debugging.assert_equal(value,tf.constant([1.25,-2.5],tf.float64))
    tf.debugging.assert_equal(score,tf.constant([[3.,4.],[5.,6.]],tf.float64))
    tf.debugging.assert_equal(calls[0][2],tf.broadcast_to(tf.eye(2,dtype=tf.float64),[2,2,2]))
    assert calls[0][0] is target.model
    assert calls[0][3]["reset_policy"] == "contract_e"
    assert calls[0][3]["observations"].shape == [2,2]
    assert target.manifest["prior"] is None
    assert target.manifest["control_status"] == "diagnostic_warm_start_not_scope_tuned"
    assert not bool(tf.reduce_any(status["outside_declared_support"]))


def test_alias_numerical_failure_is_not_zero_density_or_support():
    fixture = build_range_bearing_fixture_tf()
    model = make_range_bearing_model(fixture.A,fixture.Q,fixture.R,dtype=tf.float64)
    x = fixture.m0[None]
    value = model.observation_log_density_fn(tf.constant([[0.,8.]],tf.float64),x,fixture.observations[0])
    assert bool(tf.reduce_all(tf.math.is_nan(value)))


@pytest.mark.parametrize("stage",["m13-consumer","m13-consumer-localize","m13-consumer-parity","m13-consumer-repair"])
def test_master_has_bounded_consumer_job_and_focused_review(stage):
    from bayesfilter.inference.surrogate_hmc_closure_program import jobs,PASS_CAPS
    selected = jobs("review,"+stage)
    assert [r["name"] for r in selected] == ["review",stage]
    assert "tests/test_surrogate_hmc_m13_consumer.py" in selected[0]["command"]
    assert "tests/highdim/test_ledh_canonical_batch_fused.py::test_all_computed_correction_diagnostics_reach_batch_consumer" in selected[0]["command"]
    assert selected[1]["device"] == "gpu"
    assert selected[1]["timeout"] <= PASS_CAPS["m13-consumer"]["gpu_seconds"]
    with pytest.raises(ValueError):
        jobs("review,m13-consumer,m13-reference")


def test_repair_worker_calls_existing_full_parity_implementation(monkeypatch, tmp_path):
    import json
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    from bayesfilter.testing import surrogate_hmc_m13_consumer as consumer
    from bayesfilter.inference.surrogate_hmc_closure_program import M13_REPAIR_PLAN
    calls = []
    def parity(output):
        calls.append(output)
        return dict(status="MOCK_ENGINEERING_CHECK", qualified=False,
                    posterior_qualified=False, inference_launched=False)
    monkeypatch.setattr(consumer, "full_parity_stage", parity)
    output = tmp_path/"report.json"
    assert worker.main(["--stage", "m13-consumer-repair", "--device", "cpu",
                        "--output", str(output)]) == 0
    assert calls == [output]
    result = json.loads(output.read_text())
    assert result["plan"] == M13_REPAIR_PLAN
    assert result["repair_replay"] and result["original_parity_thresholds_unchanged"]
    assert result["gpu_intentionally_hidden"]
    assert not result["posterior_qualified"] and not result["inference_launched"]


def _engineering_evidence():
    return dict(parameter_points=[[0., 0.], [.1, -.1]],
        checks=dict(all_fp64_points_valid=True, trace_value_score_parity=True,
            correction_condition_diagnostic_present=True, correction_feasibility_diagnostic_present=True,
            deterministic=True, full_total_derivative_parity=True, fp32_observability_no_harm=False),
        fp32_replay=dict(value=[1., 2.], score=[[.1, .2], [.3, .4]],
            status_tensors=dict(program_valid=[True, True]), manifest=dict(dtype="float32", tf32=True),
            value_max_scaled_error=.00497577665, score_max_scaled_error=.0138047338))


def test_saved_number_mismatch_warns_without_posterior_promotion():
    from copy import deepcopy
    from bayesfilter.testing.surrogate_hmc_m13_consumer import assess_full_parity
    evidence = _engineering_evidence()
    original = deepcopy(evidence)
    result = assess_full_parity(evidence)
    assert evidence == original
    assert result["qualified"] and not result["failed_required_checks"]
    assert result["warnings"][0]["code"] == "FP32_SAVED_RUN_DIFFERENCE"
    assert result["warnings"][0]["blocks_continuation"] is False
    assert result["warnings"][0]["threshold"] == 3e-5
    assert not result["posterior_qualified"] and not result["scope_controls_qualified"]
    assert not result["inference_launched"] and not result["all_models_validated"]


@pytest.mark.parametrize("check", [
    "all_fp64_points_valid", "trace_value_score_parity", "deterministic",
    "full_total_derivative_parity", "correction_condition_diagnostic_present",
    "correction_feasibility_diagnostic_present",
])
@pytest.mark.parametrize("missing", [False, True])
def test_required_numerical_failure_or_missing_check_still_blocks(check, missing):
    from bayesfilter.testing.surrogate_hmc_m13_consumer import assess_full_parity
    evidence = _engineering_evidence()
    if missing:
        del evidence["checks"][check]
    else:
        evidence["checks"][check] = False
    result = assess_full_parity(evidence)
    assert not result["qualified"] and check in result["failed_required_checks"]
    assert result["warnings"]


@pytest.mark.parametrize("defect", ["invalid_point", "nan_value", "infinite_score", "missing_score",
                                     "missing_comparison", "missing_validity", "missing_precision"])
def test_fp32_validity_is_not_lost_when_saved_comparison_becomes_warning(defect):
    from bayesfilter.testing.surrogate_hmc_m13_consumer import assess_full_parity
    evidence = _engineering_evidence()
    fp32 = evidence["fp32_replay"]
    if defect == "invalid_point":
        fp32["status_tensors"]["program_valid"][1] = False
    elif defect == "nan_value":
        fp32["value"][0] = float("nan")
    elif defect == "infinite_score":
        fp32["score"][1][1] = float("inf")
    elif defect == "missing_score":
        del fp32["score"]
    elif defect == "missing_comparison":
        del evidence["checks"]["fp32_observability_no_harm"]
    elif defect == "missing_validity":
        del fp32["status_tensors"]
    else:
        del fp32["manifest"]["dtype"]
    assert not assess_full_parity(evidence)["qualified"]


def test_empty_evidence_is_not_qualified_and_matching_saved_values_need_no_warning():
    from bayesfilter.testing.surrogate_hmc_m13_consumer import assess_full_parity
    assert not assess_full_parity({})["qualified"]
    evidence = _engineering_evidence()
    evidence["checks"]["fp32_observability_no_harm"] = True
    evidence["fp32_replay"].update(value_max_scaled_error=0., score_max_scaled_error=0.)
    result = assess_full_parity(evidence)
    assert result["qualified"] and not result["warnings"]


def test_master_reports_fp32_default_and_does_not_require_saved_run_localization():
    from bayesfilter.inference.surrogate_hmc_closure_program import inference_readiness
    row = next(row for row in inference_readiness() if row["scope"] == "M13")
    assert row["saved_run_comparison_role"] == "explanatory_only"
    assert row["saved_run_warning_blocks_continuation"] is False
    assert row["force_default_precision"] == "float32"
    assert row["force_tf32_default"] and row["force_jit_compile_default"]
    assert "scoped controls" in row["next_action"]
    assert "optional" in row["next_action"]
    assert not row["ready"]  # Engineering checks do not supply posterior prerequisites.


def test_default_fp32_target_preserves_dtype_through_canonical_consumer(monkeypatch):
    fixture = build_range_bearing_fixture_tf()
    fixture = replace(fixture, observations=fixture.observations[:2], states=fixture.states[:2])
    target = make_range_bearing_target(fixture, particle_count=16, controls=warm_start_controls())
    assert target.manifest["dtype"] == "float32"
    assert target.manifest["tf32"] and target.manifest["jit_compile"]
    assert all(value.dtype == tf.float32 for value in target.inputs.values())
    assert target.controls["reset_design"].dtype == tf.float32
    calls = []
    def canonical(model, theta, directions, **kwargs):
        calls.append((theta.dtype, directions.dtype))
        value, score = tf.reduce_sum(theta, -1), tf.ones_like(theta)
        valid = tf.ones([2], tf.bool)
        return value, score, dict(value_valid=valid, force_valid=valid, program_valid=valid)
    monkeypatch.setattr(target_module, "canonical_batch_fused_value_score", canonical)
    compiled = target.compile(2)
    assert compiled.input_signature[0].dtype == tf.float32
    assert compiled.function_spec.jit_compile
    value, score, status = compiled(tf.constant([[.1, -.1], [.2, -.2]], tf.float32))
    assert calls == [(tf.float32, tf.float32)]
    assert value.dtype == score.dtype == tf.float32
    assert bool(tf.reduce_all(status["program_valid"]))


def test_worker_executes_full_parity_decision_with_saved_run_warning(monkeypatch, tmp_path):
    """Execute the worker->consumer->decision chain, replacing only the costly kernel."""
    import json
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    from bayesfilter.testing import surrogate_hmc_m13_consumer as consumer

    class QuadraticTarget:
        def __init__(self, dtype):
            self.dtype = dtype
            self.manifest = dict(dtype=dtype.name, tf32=dtype == tf.float32)
            self.identity = "test-only-quadratic-mechanics"
        def compile(self, batch_size, **kwargs):
            def evaluate(points):
                value = tf.reduce_sum(points*points, -1)
                score = 2*points
                valid = tf.ones([batch_size], tf.bool)
                return value, score, dict(program_valid=valid,
                    higher_moment_maximum_diagonal_scaled_system_condition=tf.ones([batch_size], self.dtype),
                    higher_moment_minimum_pearson_feasibility_margin=tf.ones([batch_size], self.dtype))
            return evaluate

    monkeypatch.setattr(consumer, "make_range_bearing_target",
        lambda fixture, *, particle_count, controls, dtype: QuadraticTarget(dtype))
    monkeypatch.setattr(tf.config.experimental, "get_memory_info", lambda device: dict(current=0, peak=0))
    output = tmp_path/"report.json"
    assert worker.main(["--stage", "m13-consumer-repair", "--device", "cpu", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["qualified"] and result["checks"]["full_total_derivative_parity"]
    assert result["checks"]["fp32_observability_no_harm"] is False
    assert result["warnings"] and all(result["required_checks"].values())
    assert result["decision_policy"] == consumer.MATERIALITY_POLICY
    assert result["repair_replay"] and result["original_parity_thresholds_unchanged"]
    assert not result["posterior_qualified"]
