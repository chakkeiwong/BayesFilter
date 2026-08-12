from __future__ import annotations

import tensorflow as tf

from scripts.run_zhao_cui_austria_sir_observed_data_score import _mechanics_stage

from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
    AustriaSIRLatentPreclipFP32Model,
    CLAIM_PARTICLE_COUNT,
    EVENT_ORDER,
    ROUTE_CLASSIFICATION,
    RUNTIME_FP32_OBSERVATION_SHA256,
    make_austria_sir_observed_data_target,
    make_bootstrap_mechanics_branch,
    prepare_austria_sir_source_order_program,
)
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
    SIR_OBSERVATION_SHA256,
    SIR_STATE_SHA256,
)


def _direct_scalar(model, theta, branch) -> tf.Tensor:
    log_n = tf.math.log(tf.cast(branch.particle_count, tf.float32))
    current = (
        model.initial_log_density(theta, branch.states[0])
        - branch.initial_log_proposal_density
    )
    log_sum = tf.reduce_logsumexp(current)
    value = log_sum - log_n
    previous_log_weights = current - log_sum
    for time_index in range(1, branch.transition_count + 1):
        row = time_index - 1
        ancestor = branch.ancestors[row]
        parent = tf.gather(branch.states[time_index - 1], ancestor)
        current = (
            tf.gather(previous_log_weights, ancestor)
            + model.transition_log_density(
                theta, parent, branch.states[time_index], time_index
            )
            + model.observation_log_density(
                theta,
                branch.states[time_index],
                branch.observations[row],
                time_index,
            )
            - tf.gather(branch.auxiliary_log_probabilities[row], ancestor)
            - branch.transition_log_proposal_density[row]
        )
        log_sum = tf.reduce_logsumexp(current)
        value = value + log_sum - log_n
        previous_log_weights = current - log_sum
    return value


def test_exact_target_factory_binds_source_and_runtime_hashes() -> None:
    target = make_austria_sir_observed_data_target()
    assert target.source_states.shape == (21, 18)
    assert target.source_observations.shape == (20, 9)
    assert target.observations.shape == (20, 9)
    assert target.manifest["source_state_sha256"] == SIR_STATE_SHA256
    assert target.manifest["source_observation_sha256"] == SIR_OBSERVATION_SHA256
    assert (
        target.manifest["runtime_fp32_observation_sha256"]
        == RUNTIME_FP32_OBSERVATION_SHA256
    )
    assert target.manifest["event_order"] == EVENT_ORDER
    assert target.manifest["route_classification"] == ROUTE_CLASSIFICATION


def test_fp32_latent_density_and_scores_match_fp64_reference() -> None:
    model32 = AustriaSIRLatentPreclipFP32Model()
    model64 = latent_preclip_zhao_cui_sir_austria_model()
    theta32 = tf.constant([0.015, -0.01, 0.025], tf.float32)
    theta64 = tf.cast(theta32, tf.float64)
    target = make_austria_sir_observed_data_target()
    previous64 = tf.stack(
        [
            target.source_states[0],
            tf.tensor_scatter_nd_update(
                target.source_states[1], [[0], [2]], [-2.0, -1.0]
            ),
        ]
    )
    previous32 = tf.cast(previous64, tf.float32)
    for time_index in (1, 2):
        mean64 = model64.transition_mean(
            theta64, previous64, time_index=time_index
        )
        mean32 = model32.transition_mean(
            theta32, previous32, tf.constant(time_index, tf.int32)
        )
        current64 = mean64 + tf.constant(0.1, tf.float64)
        current32 = tf.cast(current64, tf.float32)
        score64 = model64.transition_log_density_parameter_score(
            theta64, previous64, current64, time_index
        )
        score32 = model32.transition_log_density_parameter_score(
            theta32,
            previous32,
            current32,
            tf.constant(time_index, tf.int32),
        )
        tf.debugging.assert_near(
            tf.cast(mean64, tf.float32), mean32, atol=3e-4, rtol=3e-5
        )
        tf.debugging.assert_near(
            tf.cast(score64, tf.float32), score32, atol=2e-3, rtol=2e-3
        )

    observation64 = target.source_observations[0]
    observation32 = target.observations[0]
    state64 = target.source_states[1:3]
    state32 = tf.cast(state64, tf.float32)
    value64 = model64.observation_log_density(theta64, state64, observation64, 1)
    value32 = model32.observation_log_density(theta32, state32, observation32, 1)
    score64 = model64.observation_log_density_parameter_score(
        theta64, state64, observation64, 1
    )
    score32 = model32.observation_log_density_parameter_score(
        theta32, state32, observation32, 1
    )
    tf.debugging.assert_near(tf.cast(value64, tf.float32), value32, atol=2e-3)
    tf.debugging.assert_near(tf.cast(score64, tf.float32), score32, atol=2e-3)


def test_t2_program_matches_direct_scalar_manual_score_fd_and_tape() -> None:
    target = make_austria_sir_observed_data_target()
    branch = make_bootstrap_mechanics_branch(
        particle_count=12, horizon=2, proposal_seed=30701, target=target
    )
    program = prepare_austria_sir_source_order_program(branch, target=target)
    theta = tf.constant([0.01, -0.015, 0.02], tf.float32)
    result = program.evaluate(theta)
    direct = _direct_scalar(program.model, theta, branch)
    tf.debugging.assert_near(result["log_likelihood"], direct, atol=2e-4)
    tf.debugging.assert_near(
        result["log_likelihood"], tf.reduce_sum(result["log_increments"]), atol=2e-4
    )
    tf.debugging.assert_near(
        result["score"], tf.reduce_sum(result["increment_scores"], axis=0), atol=2e-4
    )

    step = tf.constant(1e-3, tf.float32)
    finite_difference = []
    for parameter_index in range(3):
        direction = tf.one_hot(parameter_index, 3, dtype=tf.float32)
        finite_difference.append(
            (
                _direct_scalar(program.model, theta + step * direction, branch)
                - _direct_scalar(program.model, theta - step * direction, branch)
            )
            / (2.0 * step)
        )
    tf.debugging.assert_near(
        result["score"], tf.stack(finite_difference), atol=0.35, rtol=3e-3
    )

    with tf.GradientTape() as tape:
        tape.watch(theta)
        tape_value = _direct_scalar(program.model, theta, branch)
    tape_score = tape.gradient(tape_value, theta)
    tf.debugging.assert_near(result["score"], tape_score, atol=3e-3, rtol=3e-4)

    graph_result = program.compiled(jit_compile=False)(theta)
    tf.debugging.assert_near(
        graph_result["log_likelihood"], result["log_likelihood"], atol=2e-4
    )
    # Graph/eager FP32 reductions can differ by a few ulps after the RK4 loop.
    tf.debugging.assert_near(graph_result["score"], result["score"], atol=1e-2)
    assert bool(result["finite"].numpy())


def test_full_horizon_bootstrap_replays_but_is_not_claim_eligible() -> None:
    target = make_austria_sir_observed_data_target()
    branch = make_bootstrap_mechanics_branch(
        particle_count=8, horizon=20, proposal_seed=30702, target=target
    )
    program = prepare_austria_sir_source_order_program(branch, target=target)
    theta = tf.zeros([3], tf.float32)
    first = program.evaluate(theta)
    second = program.evaluate(theta)
    tf.debugging.assert_equal(first["log_likelihood"], second["log_likelihood"])
    tf.debugging.assert_equal(first["score"], second["score"])
    assert first["log_increments"].shape == (21,)
    assert first["increment_scores"].shape == (21, 3)
    assert bool(first["finite"].numpy())

    claim_sized = make_bootstrap_mechanics_branch(
        particle_count=CLAIM_PARTICLE_COUNT,
        horizon=20,
        proposal_seed=30703,
        target=target,
    )
    try:
        prepare_austria_sir_source_order_program(
            claim_sized, target=target, require_claim_scope=True
        )
    except ValueError as exc:
        assert "bootstrap mechanics branch" in str(exc)
    else:
        raise AssertionError("bootstrap branch must not be claim eligible")


def test_campaign_mechanics_payload_uses_evaluator_output_contract() -> None:
    payload = _mechanics_stage(
        device={
            "execution_class": "explicit_cpu_reference",
            "online_device": "/CPU:0",
        },
        particle_count=4,
        horizon=1,
        seed=30704,
        jit_compile=False,
    )
    assert payload["primary_pass"] is True
    assert len(payload["ess_by_time"]) == 2
    assert payload["artifact_role"] == (
        "mechanics_baseline_not_zhao_cui_proposal_quality"
    )
