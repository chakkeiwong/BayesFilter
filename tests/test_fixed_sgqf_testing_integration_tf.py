import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import (
    TFFixedSGQFDerivatives,
    tf_fixed_sgqf_score,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    TFFixedSGQFBranchConfig,
    TFFixedSGQFNonlinearModel,
    tf_fixed_sgqf_cloud,
    tf_fixed_sgqf_filter,
    tf_fixed_sgqf_p47_one_step_oracle,
)
from bayesfilter.testing import (
    fixed_sgqf_branch_summary,
    fixed_sgqf_diagnostic_snapshot,
    fixed_sgqf_failure_label,
)


def _score_fixture():
    oracle, cloud = tf_fixed_sgqf_p47_one_step_oracle()
    model = oracle.model()
    derivatives = TFFixedSGQFDerivatives(
        d_initial_mean=tf.zeros([1, 1], dtype=tf.float64),
        d_initial_covariance=tf.zeros([1, 1, 1], dtype=tf.float64),
        d_process_covariance=tf.zeros([1, 1, 1], dtype=tf.float64),
        d_observation_covariance=tf.zeros([1, 1, 1], dtype=tf.float64),
        transition_state_jacobian_fn=lambda points: tf.ones([tf.shape(points)[0], 1, 1], dtype=tf.float64),
        d_transition_fn=lambda points: tf.convert_to_tensor(points, dtype=tf.float64)[tf.newaxis, :, :],
        observation_state_jacobian_fn=lambda points: 2.0 * tf.convert_to_tensor(points, dtype=tf.float64)[:, tf.newaxis, :],
        d_observation_fn=lambda points: tf.zeros([1, tf.shape(points)[0], 1], dtype=tf.float64),
        name="narrow_integration_score_fixture",
    )
    branch = TFFixedSGQFBranchConfig(predictive_epsilon=1e-10, innovation_epsilon=1e-10)
    return oracle.observation, model, derivatives, cloud, branch


def test_fixed_sgqf_testing_helpers_expose_diagnostics_for_successful_value_and_score() -> None:
    observations, model, derivatives, cloud, branch = _score_fixture()
    value_result = tf_fixed_sgqf_filter(observations, model, cloud=cloud, branch_config=branch, return_filtered=True)
    score_result = tf_fixed_sgqf_score(
        observations,
        model,
        derivatives,
        cloud=cloud,
        branch_config=branch,
        expected_branch_identity=value_result.branch_identity,
    )

    value_snapshot = fixed_sgqf_diagnostic_snapshot(value_result)
    score_snapshot = fixed_sgqf_diagnostic_snapshot(score_result)

    assert value_snapshot.branch_hash == value_result.branch_identity.hash.value
    assert score_snapshot.branch_hash == score_result.branch_identity.hash.value
    assert value_snapshot.runtime_mode == "xla_tensor_recurrence_python_branch_records"
    assert score_snapshot.runtime_mode == "xla_tensor_recurrence_python_branch_records"
    assert value_snapshot.failure_stage is None
    assert score_snapshot.failure_stage is None
    assert score_snapshot.derivative_method == "analytic_first_order_fixed_branch"
    assert score_snapshot.same_branch_signature == (
        score_result.branch_identity.hash.value,
        "accepted",
        -1,
    )


def test_fixed_sgqf_branch_summary_counts_mixed_success_and_failure() -> None:
    observations, model, derivatives, cloud, branch = _score_fixture()
    ok_value = tf_fixed_sgqf_filter(observations, model, cloud=cloud, branch_config=branch, return_filtered=True)
    ok_score = tf_fixed_sgqf_score(
        observations,
        model,
        derivatives,
        cloud=cloud,
        branch_config=branch,
        expected_branch_identity=ok_value.branch_identity,
    )
    blocked_model = TFFixedSGQFNonlinearModel(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        process_covariance=tf.constant([[-2.0]], dtype=tf.float64),
        observation_covariance=model.observation_covariance,
        transition_fn=model.transition_fn,
        observation_fn=model.observation_fn,
        name="narrow_integration_blocked_fixture",
    )
    blocked_value = tf_fixed_sgqf_filter(observations, blocked_model, cloud=cloud, branch_config=branch, return_filtered=True)
    blocked_score = tf_fixed_sgqf_score(
        observations,
        blocked_model,
        derivatives,
        cloud=cloud,
        branch_config=branch,
    )

    summary = fixed_sgqf_branch_summary([ok_value, ok_score, blocked_value, blocked_score])

    assert summary.total_count == 4
    assert summary.ok_count == 2
    assert summary.predictive_covariance_failure_count == 2
    assert summary.failure_labels == ("predictive_covariance:positive_definiteness_veto",)
    assert fixed_sgqf_failure_label(blocked_value.failure) == "predictive_covariance:positive_definiteness_veto"
