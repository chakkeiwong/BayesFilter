"""Independent mechanics/reference diagnostics for additive structural DCTR.

No posterior or tuning authority. The actual batch and single-cloud consumers
are exercised with the full reset and dual-cap correction enabled.
"""
from dataclasses import replace

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel, _single_cloud_model, canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_score_tf import _value_and_analytical_score_impl


def fixture(dtype=tf.float64, *, nonlinear=False, count=16, horizon=3):
    def transition(theta, x):
        return tf.stack([theta[:, 0]*x[:, 0]+theta[:, 1]*x[:, 1], x[:, 0]], -1)

    def tangent(theta, x, dx, direction):
        return tf.stack([theta[:, 0]*dx[:, 0]+theta[:, 1]*dx[:, 1]
                         + direction[:, 0]*x[:, 0]+direction[:, 1]*x[:, 1], dx[:, 0]], -1)

    curvature = tf.constant(.1 if nonlinear else 0., dtype)
    model = PerPointScoreModel(
        transition_mean_fn=transition, transition_mean_tangent_fn=tangent,
        observation_fn=lambda x: x[:, :1]+curvature*x[:, 1:]**2,
        observation_tangent_fn=lambda x, dx: dx[:, :1]+2*curvature*x[:, 1:]*dx[:, 1:],
        observation_jacobian_fn=lambda x: tf.stack([tf.ones_like(x[:, 0]), 2*curvature*x[:, 1]], -1)[:, None, :],
        observation_jacobian_tangent_fn=lambda x, dx: tf.stack([tf.zeros_like(x[:, 0]), 2*curvature*dx[:, 1]], -1)[:, None, :],
        process_covariance=tf.constant([[.09]], dtype),
        process_covariance_fn=lambda theta: tf.reshape(tf.exp(2*theta[2]), [1, 1]),
        process_covariance_tangent_fn=lambda theta, direction: tf.reshape(2*tf.exp(2*theta[2])*direction[2], [1, 1]),
        observation_covariance=tf.constant([[.04]], dtype),
        innovation_matrix_fn=lambda theta: tf.reshape(tf.stack([tf.constant(1., dtype), theta[3]]), [2, 1]),
        innovation_matrix_tangent_fn=lambda theta, direction: tf.reshape(tf.stack([tf.constant(0., dtype), direction[3]]), [2, 1]),
    )
    cloud = tf.random.stateless_normal([count, 2], [20260922, 381], dtype=dtype)*tf.constant(.5, dtype)
    args = dict(initial_states=cloud,
                initial_covariances=tf.eye(2, batch_shape=[count], dtype=dtype)*tf.constant(.25, dtype),
                noises=tf.random.stateless_normal([horizon, count, 1], [20260922, 382], dtype=dtype),
                observations=tf.reshape(tf.linspace(tf.constant(.2, dtype), tf.constant(-.15, dtype), horizon), [horizon, 1]))
    controls = dict(reset_policy="contract_e", reset_design=cloud,
                    reset_sinkhorn_steps=8, reset_balance_steps=8,
                    correction_steps=1, pairwise_steps=1, coordinate_cap=2.)
    theta = tf.stack([tf.constant(.45, dtype), tf.constant(-.15, dtype),
                      tf.math.log(tf.constant(.3, dtype)), tf.constant(0., dtype)])
    return model, args, controls, theta


def batch_kernel(model, args, controls):
    dtype = args["initial_states"].dtype

    @tf.function(input_signature=[tf.TensorSpec([None, 4], dtype), tf.TensorSpec([None, 4], dtype)],
                 jit_compile=True, autograph=False)
    def evaluate(theta, directions):
        return canonical_batch_fused_value_score(model, theta, directions, **args,
                    **controls, substeps=3, k_batch_mode="sequential")
    return evaluate


def full_rank_pair():
    model, args, controls, theta = fixture()
    q = tf.constant([[.09, .01], [.01, .08]], tf.float64)
    plain = replace(model, innovation_matrix_fn=None, innovation_matrix_tangent_fn=None,
                    process_covariance=q, process_covariance_fn=None, process_covariance_tangent_fn=None)
    structural = replace(plain, innovation_matrix=tf.eye(2, dtype=tf.float64))
    args["noises"] = tf.random.stateless_normal([3, 16, 2], [20260922, 383], dtype=tf.float64)
    return plain, structural, args, controls, theta


def integration_stage():
    records = []
    for nonlinear in (False, True):
        model, args, controls, theta = fixture(nonlinear=nonlinear)
        evaluate = batch_kernel(model, args, controls)
        directions = tf.eye(4, dtype=tf.float64)
        values, scores, status = evaluate(tf.repeat(theta[None], 4, axis=0), directions)
        eps = tf.constant(1e-5, tf.float64)
        trial = tf.concat([theta[None]+eps*directions, theta[None]-eps*directions], 0)
        shifted, _, shifted_status = evaluate(trial, tf.zeros_like(trial))
        fd = (shifted[:4]-shifted[4:])/(2*eps)
        error = float(tf.reduce_max(tf.abs(scores-fd)/tf.maximum(1., tf.abs(fd))))
        valid = bool(tf.reduce_all(status["force_valid"] & status["value_valid"])) and bool(tf.reduce_all(shifted_status["force_valid"]))

        # Graph reference exception: expose intermediate proposals and verify
        # their support. The claim-bearing batch calculation above uses XLA.
        @tf.function(input_signature=[tf.TensorSpec([4], tf.float64)], autograph=False, jit_compile=False)
        def traced(point):
            bound = _single_cloud_model(model, point, directions[0])
            return _value_and_analytical_score_impl(bound, point, **args, **controls,
                        flow_substeps=3, with_score=True, return_trace=True, return_status=True)

        single_value, single_score, trace, single_status = traced(theta)
        residual = max(float(tf.reduce_max(tf.abs(step["children"][:, 1]-step["ancestors"][:, 0]))) for step in trace)
        pre_residual = max(float(tf.reduce_max(tf.abs(step["pre_flow"][:, 1]-step["ancestors"][:, 0]))) for step in trace)
        parity = max(abs(float(single_value-values[0])), abs(float(single_score[0]-scores[0])))
        valid = valid and bool(single_status["force_valid"])
        records.append(dict(observation="quadratic_control" if nonlinear else "AR2_linear", valid=valid,
                            value=float(values[0]), score=scores.numpy().tolist(), finite_difference=fd.numpy().tolist(),
                            relative_score_error=error, deterministic_lag_residual=residual,
                            pre_flow_lag_residual=pre_residual, batch_single_error=parity))

    plain, structural, args, controls, theta = full_rank_pair()
    direction = tf.constant([[1., -.2, 0., 0.]], tf.float64)
    a = batch_kernel(plain, args, controls)(theta[None], direction)
    b = batch_kernel(structural, args, controls)(theta[None], direction)
    identity_error = max(float(tf.reduce_max(tf.abs(a[0]-b[0]))), float(tf.reduce_max(tf.abs(a[1]-b[1]))))
    passed = all(r["valid"] and r["relative_score_error"] < 2e-5 and
                 r["deterministic_lag_residual"] < 1e-13 and r["pre_flow_lag_residual"] < 1e-13
                 and r["batch_single_error"] < 2e-8 for r in records) and identity_error < 2e-8
    return dict(status="STRUCTURAL_CANONICAL_MECHANICS_QUALIFIED" if passed else "STRUCTURAL_IMPLEMENTATION_FAILED",
                qualified=passed, cases=records, identity_chart_error=identity_error,
                jit_compile=True, trace_exception="non-XLA graph for independent intermediate-state diagnostics",
                full_canonical_controls=controls | {"reset_design": "fixed initial cloud", "flow_substeps": 3},
                seed_policy=[20260922, 381, 382, 383],
                target="additive Gaussian innovations with deterministic affine completion",
                next_action="Existing master: independent AR2 reference, scoped DCTR tuning, public HMC tuning and posterior comparison",
                posterior_comparisons="NOT_RUN", all_models_validated=False)
