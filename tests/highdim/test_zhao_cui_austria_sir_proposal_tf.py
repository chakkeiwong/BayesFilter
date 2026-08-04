from __future__ import annotations

import copy
import json

import pytest
import tensorflow as tf

from scripts.run_zhao_cui_austria_sir_observed_data_score import (
    _proposal_t1_smoke_stage,
)

from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)

from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
    EVENT_ORDER,
    make_austria_sir_observed_data_target,
    prepare_austria_sir_source_order_program,
)
from bayesfilter.highdim.zhao_cui_austria_sir_proposal_tf import (
    T1_CALIBRATION_SEED_BASE,
    T1_VALIDATION_SEED,
    AustriaSIRT1ProposalSpec,
    WhitenedAlgebraicCoordinateMap,
    WhitenedGaussianQuantileCoordinateMap,
    _json_ready,
    _training_batch,
    compile_austria_sir_t1_proposal_branch,
    fit_austria_sir_t1_proposal,
    load_t1_artifact,
    make_source_order_t1_ukf_guide,
)


DTYPE = tf.float64


def _tiny_spec() -> AustriaSIRT1ProposalSpec:
    return AustriaSIRT1ProposalSpec(
        degree=2,
        rank=1,
        batch_size=8,
        train_batches=1,
        learning_rate=1e-3,
        l1_weight=1e-9,
        l2_weight=1e-8,
        defensive_tau=1e-8,
        cdf_grid_size=9,
        cdf_bisection_steps=4,
        kr_max_batch_working_bytes=64 * 1024 * 1024,
    )


@pytest.fixture(scope="module")
def tiny_artifact():
    return fit_austria_sir_t1_proposal(
        _tiny_spec(), seed=T1_CALIBRATION_SEED_BASE
    )


def test_algebraic_coordinate_map_roundtrip_and_jacobian() -> None:
    matrix = tf.constant([[1.4, 0.2], [-0.1, 0.8]], DTYPE)
    coordinate_map = WhitenedAlgebraicCoordinateMap(
        tf.constant([0.3, -0.2], DTYPE), matrix
    )
    reference = tf.constant([[-0.7, 0.25], [0.0, 0.8]], DTYPE)
    physical, forward_log_det = coordinate_map.forward(reference)
    recovered, inverse_log_det = coordinate_map.inverse(physical)
    tf.debugging.assert_near(recovered, reference, atol=2e-12)
    tf.debugging.assert_near(
        forward_log_det + inverse_log_det, tf.zeros([2], DTYPE), atol=2e-12
    )


def test_gaussian_quantile_map_roundtrip_and_exact_prior_density(
    tiny_artifact,
) -> None:
    assert isinstance(
        tiny_artifact.previous_map, WhitenedGaussianQuantileCoordinateMap
    )
    reference = tf.constant(
        [[-0.7, 0.25] if axis % 2 == 0 else [0.0, 0.8] for axis in range(18)],
        DTYPE,
    )
    physical, forward_log_det = tiny_artifact.previous_map.forward(
        tf.transpose(reference)
    )
    recovered, inverse_log_det = tiny_artifact.previous_map.inverse(physical)
    tf.debugging.assert_near(recovered, tf.transpose(reference), atol=2e-12)
    tf.debugging.assert_near(
        forward_log_det + inverse_log_det, tf.zeros([2], DTYPE), atol=2e-12
    )
    log_q = (
        tf.math.log(tiny_artifact.initial_transport().eval_pdf(reference))
        - forward_log_det
    )
    log_p = latent_preclip_zhao_cui_sir_austria_model().initial_log_density(
        tf.zeros([3], DTYPE), physical
    )
    tf.debugging.assert_near(log_q, log_p, atol=2e-11)
def test_ukf_guide_uses_transition_then_active_y1_and_finite_covariance() -> None:
    target = make_austria_sir_observed_data_target()
    guide = make_source_order_t1_ukf_guide()
    assert guide.manifest["event_order"] == "x0_then_transition_then_y1"
    assert (
        guide.manifest["source_observation_sha256"]
        == target.manifest["source_observation_sha256"]
    )
    covariance = guide.manifest["current_covariance"]
    tf.debugging.assert_near(covariance, tf.transpose(covariance), atol=2e-12)
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariance))
    assert guide.manifest["claim_class"] == "scout_not_truth"


def test_training_batch_is_exact_likelihood_weighted_sampling_measure() -> None:
    guide = make_source_order_t1_ukf_guide()
    batch, manifest = _training_batch(
        guide=guide,
        sample_count=8,
        seed=T1_CALIBRATION_SEED_BASE,
        label="unit_test_calibration",
    )
    tf.debugging.assert_equal(batch.target_values, tf.ones([8], DTYPE))
    tf.debugging.assert_near(tf.reduce_max(batch.weights), 1.0, atol=2e-15)
    tf.debugging.assert_non_negative(batch.weights)
    assert manifest["seed"] != T1_VALIDATION_SEED
    assert "weighted_by_g_y1_given_z1" in manifest["objective_identity"]


def test_tiny_fit_artifact_roundtrip_and_tamper_rejection(tiny_artifact) -> None:
    payload = json.loads(json.dumps(_json_ready(tiny_artifact.payload())))
    reloaded = load_t1_artifact(payload)
    assert reloaded.artifact_id == tiny_artifact.artifact_id
    assert len(reloaded.cores) == 36
    assert reloaded.diagnostics["calibration_validation_seed_disjoint"] is True
    assert reloaded.spec.l1_weight > 0.0
    assert reloaded.initial_transport().dimension == 18
    assert reloaded.transport().dimension == 36
    tf.debugging.assert_positive(reloaded.density().normalizer())

    tampered = copy.deepcopy(payload)
    tampered["cores"][0]["values"][0][0][0] += 1e-3
    with pytest.raises(ValueError, match="serialized TT core identity rejected"):
        load_t1_artifact(tampered)

    stamped = copy.deepcopy(payload)
    stamped["diagnostics"]["target_id"] = "caller_override"
    with pytest.raises(ValueError, match="caller-stamped T1 artifact identity rejected"):
        load_t1_artifact(stamped)


def test_t1_fitted_transport_compiles_complete_source_order_program(
    tiny_artifact,
) -> None:
    particle_count = 2
    initial_reference = tf.constant(
        [
            [0.2, 0.8] if axis % 2 == 0 else [0.35, 0.65]
            for axis in range(18)
        ],
        DTYPE,
    )
    transition_reference = tf.constant(
        [
            [
                [0.25, 0.75] if axis % 2 == 0 else [0.4, 0.6]
                for axis in range(18)
            ]
        ],
        DTYPE,
    )
    compilation = compile_austria_sir_t1_proposal_branch(
        tiny_artifact,
        initial_reference_points=initial_reference,
        ancestor_uniforms=tf.constant([[0.1, 0.9]], DTYPE),
        transition_reference_points=transition_reference,
        inverse_microbatch_size=1,
    )
    branch = compilation.branch
    assert branch.states.shape == (2, particle_count, 18)
    assert branch.transition_log_proposal_density.shape == (1, particle_count)
    assert branch.event_order == EVENT_ORDER
    assert compilation.manifest["austria_t1_artifact_id"] == tiny_artifact.artifact_id
    assert compilation.manifest["production_kr_closure"] is False
    tf.debugging.assert_all_finite(
        branch.transition_log_proposal_density, "proposal density must be finite"
    )

    target = make_austria_sir_observed_data_target()
    program = prepare_austria_sir_source_order_program(branch, target=target)
    result = program.evaluate(tf.zeros([3], tf.float32))
    assert bool(result["finite"].numpy())
    assert result["log_increments"].shape == (2,)
    assert result["score"].shape == (3,)


def test_runner_t1_smoke_payload_is_explicitly_non_promotional() -> None:
    payload = _proposal_t1_smoke_stage(
        particle_count=2,
        seed=T1_CALIBRATION_SEED_BASE + 20,
    )
    assert payload["primary_pass"] is True
    assert payload["status"] == "PASS_T1_PROPOSAL_MECHANICS_SMOKE"
    assert payload["artifact_role"] == (
        "tiny_mechanics_smoke_not_tuning_or_proposal_quality"
    )
    assert payload["production_kr_closure"] is False
    assert payload["frozen_artifact"]["diagnostics"]["core_count"] == 36
