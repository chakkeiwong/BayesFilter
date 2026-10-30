"""Tiny CPU diagnostic of a missing structural capability, not inference."""
import json

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel, canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky


def test_ar2_singular_innovation_status_reaches_actual_batch_consumer():
    dtype = tf.float64
    f = tf.constant([[.45, -.15], [1., 0.]], dtype)
    h = tf.constant([[1., 0.]], dtype)
    q = tf.constant([[.09, 0.], [0., 0.]], dtype)
    model = PerPointScoreModel(
        transition_mean_fn=lambda theta, points: tf.matmul(points, f, transpose_b=True),
        transition_mean_tangent_fn=lambda theta, points, tangent, direction: tf.matmul(tangent, f, transpose_b=True),
        observation_fn=lambda points: tf.matmul(points, h, transpose_b=True),
        observation_jacobian_fn=lambda points: tf.broadcast_to(h, [tf.shape(points)[0], 1, 2]),
        observation_tangent_fn=lambda points, tangent: tf.matmul(tangent, h, transpose_b=True),
        process_covariance=q, observation_covariance=tf.constant([[.04]], dtype),
    )
    cloud = tf.constant([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]*2, dtype)

    @tf.function(input_signature=[tf.TensorSpec([2, 1], dtype)], jit_compile=True, autograph=False)
    def evaluate(theta):
        return canonical_batch_fused_value_score(
            model, theta, tf.ones([2, 1, 1], dtype), cloud,
            tf.eye(2, batch_shape=[8], dtype=dtype), tf.zeros([1, 8, 2], dtype),
            tf.constant([[.1]], dtype), substeps=1,
            reset_policy="contract_e", reset_design=cloud, reset_sinkhorn_steps=8,
            reset_balance_steps=8, correction_steps=1, pairwise_steps=1,
            coordinate_cap=2., k_batch_mode="sequential")

    value, score, status = evaluate(tf.zeros([2, 1], dtype))
    record = {key: status[key].numpy().tolist() for key in
              ("value_valid", "force_valid", "value_failure_code", "first_failure_time", "first_failure_stage")}
    print(json.dumps(dict(scope="M08-AR2-canonical-capability-diagnostic", status=record,
                         posterior_comparisons="NOT_RUN", gpu_intentionally_hidden=True)))
    assert not bool(tf.reduce_any(status["value_valid"]))
    assert not bool(tf.reduce_any(status["force_valid"]))
    assert bool(tf.reduce_all(status["value_failure_code"] == int(FailureCode.FACTORIZATION)))
    assert bool(tf.reduce_all(status["first_failure_time"] == -1))


def test_positive_process_covariance_does_not_fire_factorization_guard():
    valid, factor = safe_cholesky(tf.constant([[.09, 0.], [0., .01]], tf.float64), "process")
    assert bool(valid)
    tf.debugging.assert_near(factor @ tf.transpose(factor), tf.constant([[.09, 0.], [0., .01]], tf.float64))
