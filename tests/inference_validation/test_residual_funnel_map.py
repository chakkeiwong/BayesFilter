"""Independent supplied-map algebra and public-pipeline failure continuation."""
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import FixedTransportValueScoreAdapter, load_frozen_neutra_artifact
from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
from bayesfilter.testing.inference_validation.targets import ValidationTarget


def residual_parts(amplitude=.5):
    spec = supplied_funnel_map("residual", residual_amplitude=amplitude)
    target = ValidationTarget(spec["target"], spec["parameters"], jit_compile=False)
    transport = load_frozen_neutra_artifact(spec["transport_payload"],
        expected_target_signature=target.adapter_signature()).transport
    adapter = FixedTransportValueScoreAdapter(base_adapter=target, transport=transport,
        target_scope="inference_validation", require_batch_native=True)
    return target, transport, adapter


@pytest.mark.parametrize("amplitude", [0., .5])
def test_residual_density_jacobian_analytic_score_and_directional_difference(amplitude):
    target, transport, adapter = residual_parts(amplitude)
    z = tf.constant([[-30., .3, -2.], [-1., 2., .1], [0., -1., 1.],
                     [2., .2, 3.], [30., 1e3, -1e3]], tf.float64)
    v = 3. * z[:, 0]
    delta = amplitude * tf.math.tanh(v / 6.)
    delta_prime = amplitude / 2. * (1. - tf.math.tanh(v / 6.)**2)
    curvature = tf.exp(2. * delta)
    square_sum = tf.reduce_sum(z[:, 1:]**2, -1)
    model = tf.concat([v[:, None], tf.exp((v / 2. + delta)[:, None]) * z[:, 1:]], -1)
    mapped = transport.forward_batch(z)
    np.testing.assert_allclose(target.to_model(mapped), model, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(transport.inverse_theta_to_z_batch(mapped), z, rtol=2e-11, atol=2e-11)
    logp, score = adapter.log_prob_and_grad_batch(z)
    normalized = -.5 * z[:, 0]**2 - .5 * curvature * square_sum + 2. * delta - 1.5 * math.log(2.*math.pi)
    analytic_score = tf.concat([(-z[:, 0] + delta_prime * (2. - curvature * square_sum))[:, None],
                                -curvature[:, None] * z[:, 1:]], -1)
    np.testing.assert_allclose(logp, normalized, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(score, analytic_score, rtol=2e-11, atol=2e-11)
    centered = ValidationTarget("funnel", {"scale": 3.}, jit_compile=False)
    total_logdet = math.log(3.) + v + 2. * delta
    np.testing.assert_allclose(logp, centered.log_density(model) + total_logdet,
                               rtol=2e-11, atol=2e-11)
    assert not np.allclose(logp, centered.log_density(model))  # Omitted-Jacobian defect control.
    direction = tf.constant([[.3, -.2, .7]] * 4, tf.float64)
    h = 1e-5  # Central difference; truncation O(h^2), independent score check.
    plus = adapter.log_prob_and_grad_batch(z[:4] + h * direction)[0]
    minus = adapter.log_prob_and_grad_batch(z[:4] - h * direction)[0]
    np.testing.assert_allclose((plus-minus)/(2*h), tf.reduce_sum(score[:4]*direction, -1),
                               rtol=2e-8, atol=2e-8)
    if amplitude == 0.:
        np.testing.assert_allclose(score, -z, atol=2e-11)


def test_child_curvature_is_bounded_but_complete_hessian_is_not():
    _, _, adapter = residual_parts()
    z = tf.constant([[-30., 2., 1.], [0., 1., 1.], [0., 1e3, 1e3], [30., 2., 1.]], tf.float64)
    # The public value/score adapter deliberately freezes base scores during
    # pullback; it promises first derivatives, not differentiable Hessians.
    # Difference the actual score numerically rather than differentiating that
    # first-order interface. Both directions are native batched evaluations.
    h = 1e-5
    child_direction = tf.constant([[0., h, 0.]], tf.float64)
    height_direction = tf.constant([[h, 0., 0.]], tf.float64)
    child_curvature = -(adapter.log_prob_and_grad_batch(z + child_direction)[1][:, 1]
                        - adapter.log_prob_and_grad_batch(z - child_direction)[1][:, 1]) / (2*h)
    height_curvature = -(adapter.log_prob_and_grad_batch(z + height_direction)[1][:, 0]
                         - adapter.log_prob_and_grad_batch(z - height_direction)[1][:, 0]) / (2*h)
    expected = tf.exp(tf.math.tanh(z[:, 0] / 2.))
    np.testing.assert_allclose(child_curvature, expected, rtol=2e-8, atol=2e-8)
    assert bool(tf.reduce_all(child_curvature >= math.exp(-1.) - 2e-8))
    assert bool(tf.reduce_all(child_curvature <= math.exp(1.) + 2e-8))
    # At z0=0, U_00=1+S/8. Arbitrarily large children disprove a global bound.
    np.testing.assert_allclose(height_curvature[1:3], [1.25, 250001.], rtol=2e-8)


@pytest.mark.parametrize("amplitude", [-.1, float("inf"), float("nan"), True])
def test_invalid_residual_amplitudes_fail(amplitude):
    with pytest.raises(ValueError, match="residual_amplitude"):
        supplied_funnel_map("residual", residual_amplitude=amplitude)


@pytest.mark.parametrize("kind", ["exact", "residual"])
def test_public_supplied_map_continues_after_failed_first_member(tmp_path, monkeypatch, kind):
    import bayesfilter.inference as public
    from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
    from bayesfilter.testing.inference_validation.procedures import execute_pipeline
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.testing.inference_validation.storage import read_json, file_hash

    original = public.run_hmc_posterior
    calls = []
    def fail_first(**kwargs):
        calls.append(kwargs["member"].num_leapfrog_steps)
        if len(calls) == 1:
            kwargs["budget_check"] = lambda _: False
        return original(**kwargs)
    monkeypatch.setattr(public, "run_hmc_posterior", fail_first)
    spec = supplied_funnel_map(kind)
    design = ValidationDesign(design_id="residual-funnel-continuation-" + kind, engine="accuracy",
        scenario=ScenarioSpec(spec["target"], "fixed_transport", parameters=spec["parameters"]),
        replications=1, draws=64, seed=2026092380, budget_seconds=180,
        device="cpu_reference", purpose="real supplied-map sibling continuation mechanics",
        numerical_provenance="M28 smoke: broad acceptance region funds both fixture lengths",
        l_grid=(3, 5), step_size=.8, posterior_cap=128,
        options={"transport_payload": spec["transport_payload"], "posterior_members": "selected",
            "member_rule": "shortest_verified_l", "posterior_member_count": 2,
            "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
            "search": {"pilot_enabled": False, "refinement_rounds": 0,
                       "total_budget_units": 48, "repair_reserve_units": 8, "evidence_rungs": (1,)}})
    result = execute_pipeline(design, tmp_path)
    assert not check_inventory(read_json(result["tuning_path"]))["failures"]
    assert calls == [3, 5]
    selected = result["selection"]["candidate_ids"]
    members = {m["candidate_id"]: m for m in result["members"]}
    assert set(members) == set(result["verified_candidate_ids"])
    assert members[selected[0]]["recorded_retained_count"] == 0
    assert not members[selected[0]]["posterior"]["passed"]
    assert members[selected[1]]["posterior"]["warmup_results_per_chain"] > 0
    assert all(members[c]["warmup_exclusion_matches"] for c in selected)
    hashes = {c: file_hash(members[c]["warmup_path"]) for c in selected}
    again = execute_pipeline(design, tmp_path)
    assert again["verified_candidate_ids"] == result["verified_candidate_ids"]
    assert calls == [3, 5]
    assert hashes == {m["candidate_id"]: file_hash(m["warmup_path"])
                      for m in again["members"] if m["candidate_id"] in selected}
