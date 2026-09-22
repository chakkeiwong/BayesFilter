"""Independent map algebra and real public-tuner integrations, no training."""
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import FixedTransportValueScoreAdapter, load_frozen_neutra_artifact
from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.procedures import fixed_transport_starts, initial_starts


def map_parts(kind):
    spec = supplied_funnel_map(kind)
    target = ValidationTarget(spec["target"], spec["parameters"], jit_compile=False)
    transport = load_frozen_neutra_artifact(spec["transport_payload"],
        expected_target_signature=target.adapter_signature()).transport
    return spec, target, transport


@pytest.mark.parametrize("kind", ["exact", "partial", "partial_half"])
def test_frozen_funnel_map_matches_independent_density_score_and_inverse(kind):
    spec, target, transport = map_parts(kind)
    z = tf.constant([[-3., .3, -2.], [-1., 2., .1], [0., -1., 1.], [3., .2, 3.]], tf.float64)
    v = 3. * z[:, 0]
    a = spec["construction"]["strength"]
    s = v / 2 if a is None else 6. * tf.math.tanh(a * v / 12.)
    expected_model = tf.concat([v[:, None], tf.exp(s[:, None]) * z[:, 1:]], -1)
    mapped = transport.forward_batch(z)
    np.testing.assert_allclose(target.to_model(mapped), expected_model, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(transport.inverse_theta_to_z_batch(mapped), z, atol=2e-11, rtol=2e-11)
    centered = ValidationTarget("funnel", {"scale": 3.}, jit_compile=False)
    actual = FixedTransportValueScoreAdapter(base_adapter=target, transport=transport,
        target_scope="inference_validation", require_batch_native=True)
    values, score = actual.log_prob_and_grad_batch(z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        v = 3. * z[:, 0]
        correction = v / 2 if a is None else 6. * tf.math.tanh(a * v / 12.)
        residual = v / 2 - correction
        logp = (-.5 * z[:, 0]**2 - .5 * math.log(2 * math.pi)
                + tf.reduce_sum(-.5 * (z[:, 1:] * tf.exp(-residual[:, None]))**2
                                - residual[:, None] - .5 * math.log(2 * math.pi), -1))
    np.testing.assert_allclose(values, logp, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(score, tape.gradient(logp, z), atol=2e-11, rtol=2e-11)
    total_logdet = math.log(3.) + 2 * s
    np.testing.assert_allclose(values, centered.log_density(expected_model) + total_logdet,
                               atol=2e-11, rtol=2e-11)
    # This control is sensitive to the omitted total model Jacobian.
    assert not np.allclose(values, centered.log_density(expected_model))
    if kind == "exact":
        np.testing.assert_allclose(score, -z, atol=2e-11)
    else:
        assert not np.allclose(score, -z)


@pytest.mark.parametrize("regime", ["dispersed", "remote"])
def test_all_maps_preserve_the_physical_start_bank(regime):
    model_starts = initial_starts(ValidationTarget("funnel", jit_compile=False), regime)
    for kind in ("exact", "partial", "partial_half"):
        _, target, transport = map_parts(kind)
        starts = initial_starts(target, regime)
        latent = fixed_transport_starts(transport, starts)
        np.testing.assert_allclose(target.to_model(transport.forward_batch(latent)), model_starts,
                                   atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("scale", [0., -1., float("nan"), float("inf"), True])
def test_invalid_funnel_map_scale_fails(scale):
    with pytest.raises(ValueError, match="scale"):
        supplied_funnel_map("exact", scale=scale)


def test_map_identities_differ_and_mismatched_target_fails():
    maps = [supplied_funnel_map(k) for k in ("exact", "partial", "partial_half")]
    from bayesfilter.inference import stable_frozen_neutra_artifact_signature
    assert len({stable_frozen_neutra_artifact_signature(load_frozen_neutra_artifact(
        m["transport_payload"], expected_target_signature=m["transport_payload"]["target_signature"]))
        for m in maps}) == 3
    with pytest.raises(ValueError, match="target"):
        load_frozen_neutra_artifact(maps[0]["transport_payload"],
                                   expected_target_signature=maps[1]["transport_payload"]["target_signature"])


def test_affine_inverse_preserves_base_start_coordinates():
    target = ValidationTarget("gaussian", jit_compile=False)
    transport = load_frozen_neutra_artifact({
        "schema": "bayesfilter.neutra.frozen_affine_diag.v1", "transport_id": "start-check",
        "dimension": 2, "target_signature": target.adapter_signature(),
        "log_jacobian_available": True, "shift": [1., -2.], "raw_scale": [math.log(3.), 0.],
    }, expected_target_signature=target.adapter_signature()).transport
    starts = initial_starts(target, "dispersed")
    latent = fixed_transport_starts(transport, starts)
    np.testing.assert_allclose(latent, (starts - [1., -2.]) / [3., 1.], atol=2e-11)


def test_supplied_exact_map_uses_real_public_tuning_and_replay(tmp_path):
    from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
    from bayesfilter.testing.inference_validation.procedures import execute_pipeline
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.testing.inference_validation.storage import read_json, file_hash

    spec = supplied_funnel_map("exact")
    design = ValidationDesign(design_id="supplied-funnel-integration", engine="accuracy",
        scenario=ScenarioSpec(spec["target"], "fixed_transport", parameters=spec["parameters"]),
        replications=1, draws=64, seed=2026092251, budget_seconds=180,
        device="cpu_reference", purpose="actual supplied-map candidate and replay checks",
        numerical_provenance="September 22 supplied-whitening plan; CPU mechanics only",
        step_size=.5, posterior_cap=128,
        options={"transport_payload": spec["transport_payload"], "posterior_members": "selected",
                 "member_rule": "first_verified"})
    result = execute_pipeline(design, tmp_path)
    payload = read_json(result["tuning_path"])
    assert not check_inventory(payload)["failures"]
    assert len(result["verified_candidate_ids"]) > 1
    starts = read_json(tmp_path / "start_coordinates.json")
    _, target, transport = map_parts("exact")
    np.testing.assert_allclose(target.to_model(transport.forward_batch(starts["latent_starts"])),
                               starts["model_starts"], atol=2e-11)
    from bayesfilter.inference import load_numerical_tuning_checkpoint
    binding, _ = load_numerical_tuning_checkpoint(tmp_path / "tuning/tuning_checkpoint.json", adapter=target)
    np.testing.assert_allclose(binding.initial_active_state, starts["latent_starts"], atol=2e-11)
    assert binding.scope.coordinate_system == "fixed_transport"
    assessed = [m for m in result["members"] if m["status"] == "assessed"]
    assert len(assessed) == 1
    assert assessed[0]["warmup_exclusion_matches"]
    old_hash = file_hash(assessed[0]["draws_path"])
    resumed = execute_pipeline(design, tmp_path)
    assert resumed["verified_candidate_ids"] == result["verified_candidate_ids"]
    assert file_hash(assessed[0]["draws_path"]) == old_hash
