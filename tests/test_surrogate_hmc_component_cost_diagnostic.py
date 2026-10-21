"""Tiny deliberate CPU/XLA parity of stage instrumentation, never admission."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
from dataclasses import replace
import pytest
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.testing import surrogate_hmc_component_cost_diagnostic as diagnostic
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared


@pytest.mark.parametrize("name", ["lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing"])
@pytest.mark.parametrize("policy", ["uniform", "fp64_recursion_fp32_pairs"])
def test_stages_match_actual_factory_for_every_model_and_precision(name, policy, monkeypatch):
    monkeypatch.setattr(runtime, "N", 64)
    case = runtime.configure_precision(runtime.load_case(name), policy)
    controls = replace(case.controls, flow_steps=2, reset_steps=4, correction_steps=0)
    dtype = case.force_dtype
    theta = tf.stack([case.theta0, case.chart(tf.fill([case.p], tf.cast(.01, dtype)))[0]])
    stages = diagnostic.make_stages(case.spec, 64, controls, dtype, policy)
    p = stages[0](theta, case.fixed[0], case.observations[0],
                  case.fixed[1][0], case.fixed[2][0], case.beta)
    q = stages[1](*p, case.beta, theta)
    z = stages[2](*p[:2], *q[2:4])
    fused = shared.make_batched_score_filter(case.spec, 64, 1, controls, dtype,
                                             precision_policy=policy)
    reference = fused(theta, case.observations[:1], case.fixed[0],
                      case.fixed[1][:1], case.fixed[2][:1], case.beta)
    compared = diagnostic.compare_outputs((p, q, z), reference)
    assert compared["passed"], compared
    assert set(compared["differences"]) == {"value", "score", "particles", "particle_tangent"}
    for function in (*stages, fused):
        assert function.experimental_get_tracing_count() == 1
        assert function.get_concrete_function().function_def.attr["_XlaMustCompile"].b
    if name == "lgssm":
        broken = dict(reference, score=reference["score"] + 100.)
        assert not diagnostic.compare_outputs((p, q, z), broken)["passed"]
        broken = dict(reference, steps_completed=tf.zeros([2], tf.int32))
        assert not diagnostic.compare_outputs((p, q, z), broken)["passed"]
        broken = dict(reference, valid=tf.constant([True, False]))
        assert not diagnostic.compare_outputs((p, q, z), broken)["passed"]


def test_component_diagnostic_writes_timings_only_after_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "N", 64)
    case = runtime.load_case("lgssm")
    case.controls = replace(case.controls, flow_steps=2, reset_steps=4, correction_steps=0)
    result = diagnostic.run_diagnostic(case, tmp_path, repetitions=1)
    assert result["status"] == "COMPONENT_PARITY_PASS_COST_MEASURED"
    assert result["comparison"]["passed"]
    assert result["cpu_only"] is True
    assert result["posterior_sampling_complete"] is False
    assert result["trace_counts"] == [1, 1, 1, 1]
    assert result["controls"]["flow_steps"] == 2
    assert result["design_seed"] == 261008301
    assert len(result["input_sha256"]) == 4
    assert len(result["parameter_positions"]) == 2
    assert all(len(v) == 1 and v[0] > 0 for v in result["warmed_seconds"].values())
    assert (tmp_path / "component-cost.json").is_file()


@pytest.mark.parametrize("name", ["lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing"])
def test_full_sequence_stages_match_fused_filter(tmp_path, monkeypatch, name):
    monkeypatch.setattr(runtime, "N", 64)
    case = runtime.configure_precision(runtime.load_case(name), "fp64_recursion_fp32_pairs")
    case.observations = case.observations[:3]
    case.fixed = (case.fixed[0], case.fixed[1][:3], case.fixed[2][:3])
    case.controls = replace(case.controls, flow_steps=2, reset_steps=4, correction_steps=0)
    result = diagnostic.run_sequence_diagnostic(case, tmp_path, repetitions=1)
    assert result["status"] == "COMPONENT_SEQUENCE_PARITY_PASS_COST_MEASURED"
    assert result["comparison"]["passed"]
    assert len(result["per_time_warmed"][0]) == 3
    assert result["trace_counts"] == [1, 1, 1, 1, 1]
    assert result["cpu_only"] is True
