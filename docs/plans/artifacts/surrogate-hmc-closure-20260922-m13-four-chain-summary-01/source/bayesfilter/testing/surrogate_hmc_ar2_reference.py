"""Independent FP64 Kalman and joint-Gaussian AR2 reference diagnostics.

The fixed model has a one-dimensional Gaussian innovation and exact lag shift.
Autodiff here is the endpoint reference derivative, never the DCTR force.
"""
import json
import math
from pathlib import Path

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import (
    trapezoid_weights, summarize_grid, quantile_graph, expectation_graph,
)

HORIZON = 10
PRIOR_SD = math.log(2.)
NAMES = ["lambda_Q_mean", "lambda_R_mean", "lambda_Q_variance", "lambda_R_variance",
         "lambda_Q_lambda_R", "lambda_Q_CDF10", "lambda_Q_CDF50", "lambda_Q_CDF90",
         "lambda_R_CDF10", "lambda_R_CDF50", "lambda_R_CDF90",
         "next_y_above_zero", "next_y_above_point_two"]
MARGINS = [.1]*5 + [.02]*8


def make_reference(horizon=HORIZON):
    dtype = tf.float64
    f = tf.constant([[.45, -.15], [1., 0.]], dtype)
    h = tf.constant([[1., 0.]], dtype)
    q0 = tf.constant([[.09, 0.], [0., 0.]], dtype)
    eye = tf.eye(2, dtype=dtype)
    signature = [tf.TensorSpec([None, 2], dtype), tf.TensorSpec([horizon, 1], dtype)]

    def calculate(theta, observations):
        batch = tf.shape(theta)[0]
        q = tf.exp(2*theta[:, 0, None, None])*q0
        r = .04*tf.exp(2*theta[:, 1])
        mean, cov = tf.zeros([batch, 2, 1], dtype), tf.broadcast_to(eye, [batch, 2, 2])
        def step(t, mean, cov, logp):
            pm = f @ mean
            pc = f @ cov @ tf.transpose(f)+q
            s = pc[:, 0, 0]+r
            residual = observations[t, 0]-pm[:, 0, 0]
            logp -= .5*(math.log(2*math.pi)+tf.math.log(s)+residual**2/s)
            gain = pc[:, :, :1]/s[:, None, None]
            mean = pm+gain*residual[:, None, None]
            a = eye-gain @ h
            cov = a @ pc @ tf.linalg.matrix_transpose(a)+r[:, None, None]*(gain @ tf.linalg.matrix_transpose(gain))
            return t+1, mean, cov, logp
        _, mean, cov, logp = tf.while_loop(lambda t, *_: t < horizon, step,
                    (tf.constant(0), mean, cov, tf.zeros([batch], dtype)), maximum_iterations=horizon)
        forecast_mean = (f @ mean)[:, 0, 0]
        forecast_var = (f @ cov @ tf.transpose(f)+q)[:, 0, 0]+r
        probability = .5*tf.math.erfc((tf.constant([0., .2], dtype)[None]-forecast_mean[:, None])
                                     /tf.sqrt(2*forecast_var[:, None]))
        return logp, probability

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def statistics(theta, observations):
        return calculate(theta, observations)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def value_score(theta, observations):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            logp = calculate(theta, observations)[0]
            logp -= tf.reduce_sum(.5*(theta/PRIOR_SD)**2+math.log(PRIOR_SD)+.5*math.log(2*math.pi), axis=-1)
        return logp, tape.gradient(logp, theta)

    # Independent direct marginal covariance: stack loadings of x0, every
    # scalar process innovation and observation error. No Kalman recursion.
    powers = [eye]
    for _ in range(horizon):
        powers.append(powers[-1] @ f)
    initial_loadings = tf.concat([h @ powers[t+1] for t in range(horizon)], 0)
    process_loadings = tf.stack([tf.stack([
        tf.constant(.3, dtype)*powers[t-k][0, 0] if k <= t else tf.constant(0., dtype)
        for k in range(horizon)]) for t in range(horizon)])
    initial_cov = initial_loadings @ tf.transpose(initial_loadings)
    process_cov = process_loadings @ tf.transpose(process_loadings)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def joint_gaussian(theta, observations):
        cov = (initial_cov[None]+tf.exp(2*theta[:, 0, None, None])*process_cov[None]
               +.04*tf.exp(2*theta[:, 1, None, None])*tf.eye(horizon, dtype=dtype)[None])
        chol = tf.linalg.cholesky(cov)
        solved = tf.linalg.triangular_solve(chol, tf.broadcast_to(observations, [tf.shape(theta)[0], horizon, 1]))
        return -.5*(horizon*math.log(2*math.pi)+2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), -1)
                     +tf.reduce_sum(solved**2, [1, 2]))
    return statistics, value_score, joint_gaussian


def datasets(horizon=HORIZON):
    result = {}
    f = tf.constant([[.45, -.15], [1., 0.]], tf.float64)
    for code, name in enumerate(("calibration", "validation", "claim"), start=1):
        seed = [20260923, 500+code]
        noise = tf.random.stateless_normal([2+2*horizon], seed, dtype=tf.float64)
        state, observations = noise[:2], []
        for t in range(horizon):
            state = tf.linalg.matvec(f, state)+tf.stack([.3*noise[2+2*t], tf.constant(0., tf.float64)])
            observations.append(state[0]+.2*noise[3+2*t])
        obs = tf.cast(tf.cast(tf.stack(observations)[:, None], tf.float32), tf.float64)
        result[name] = dict(model_id="AR2", horizon=horizon, partition=name, seed=seed,
                            truth=[0., 0.], initial_law="N(0,I2)", transition_before_observation=True,
                            observations=obs.numpy().tolist())
    return result


def prepare_reference(observations, *, extended=False):
    statistics, value_score, joint = make_reference()
    obs = tf.convert_to_tensor(observations, tf.float64)
    checks = tf.constant([[0., 0.], [-1., 1.], [1., -1.], [.2, -.1]], tf.float64)
    kalman, _ = statistics(checks, obs)
    direct = joint(checks, obs)
    law_error = float(tf.reduce_max(tf.abs(kalman-direct)))
    _, score = value_score(checks, obs)
    eps = tf.constant(1e-5, tf.float64)
    fd = tf.stack([(value_score(checks+eps*direction, obs)[0]-value_score(checks-eps*direction, obs)[0])/(2*eps)
                   for direction in tf.unstack(tf.eye(2, dtype=tf.float64))], -1)
    derivative_error = float(tf.reduce_max(tf.abs(score-fd)/tf.maximum(1., tf.abs(fd))))
    protocol = [(64, 4.), (128, 4.), (256, 4.), (384, 6.)]
    if extended:
        protocol += [(512, 4.), (768, 6.), (1024, 8.)]
    grids, normalizers = [], []
    for intervals, width in protocol:
        axis = tf.linspace(tf.constant(-width*PRIOR_SD, tf.float64), tf.constant(width*PRIOR_SD, tf.float64), intervals+1)
        points = tf.reshape(tf.stack(tf.meshgrid(axis, axis, indexing="ij"), -1), [-1, 2])
        values, probabilities = [], []
        for start in range(0, int(points.shape[0]), 2048):
            batch = points[start:start+2048]
            count = int(batch.shape[0])
            batch = tf.concat([batch, tf.repeat(batch[:1], 2048-count, 0)], 0)
            ll, probability = statistics(batch, obs)
            lp = ll-tf.reduce_sum(.5*(batch/PRIOR_SD)**2+math.log(PRIOR_SD)+.5*math.log(2*math.pi), -1)
            tf.debugging.assert_all_finite(lp, "AR2 posterior integrand")
            values.append(lp[:count]); probabilities.append(probability[:count])
        logp = tf.reshape(tf.concat(values, 0), [intervals+1, intervals+1])
        predictive = tf.reshape(tf.concat(probabilities, 0), [intervals+1, intervals+1, 2])
        shift = tf.reduce_max(logp)
        unnormalized = tf.exp(logp-shift)
        mass = tf.reduce_sum(unnormalized*trapezoid_weights(axis)[:, None]*trapezoid_weights(axis)[None])
        tf.debugging.assert_all_finite(mass, "AR2 posterior normalizer")
        tf.debugging.assert_positive(mass, "AR2 posterior normalizer")
        tf.debugging.assert_all_finite(predictive, "AR2 predictive probabilities")
        normalizers.append(float(tf.math.log(mass)+shift))
        grids.append((axis, axis, unnormalized/mass, predictive))
    axis, _, density, predictive = grids[-1]
    means, scales = summarize_grid(axis, axis, density)
    tf.debugging.assert_all_finite(scales, "AR2 posterior scales")
    tf.debugging.assert_positive(scales, "AR2 posterior scales")
    cuts = tf.stack([quantile_graph(axis, tf.linalg.matvec(density, trapezoid_weights(axis))),
                     quantile_graph(axis, tf.linalg.matvec(density, trapezoid_weights(axis), transpose_a=True))])
    expectations = tf.stack([expectation_graph(*grid, means, scales, cuts) for grid in grids])
    tf.debugging.assert_all_finite(expectations, "AR2 posterior function expectations")
    changes = tf.abs(expectations[1:]-expectations[:-1])
    if extended:
        refinement = tf.abs(expectations[4]-expectations[2]) <= .5*tf.abs(expectations[2]-expectations[1])+1e-5
        allowance = 2*tf.reduce_max(tf.stack([tf.abs(expectations[4]-expectations[2]),
                         tf.abs(expectations[5]-expectations[4]), tf.abs(expectations[6]-expectations[5])]), 0)
    else:
        refinement = changes[1] <= .5*changes[0]+1e-5
        allowance = 2*tf.reduce_max(changes[1:], 0)
    precision = allowance <= tf.constant(MARGINS, tf.float64)/10
    qualified = law_error < 2e-10 and derivative_error < 2e-6 and bool(tf.reduce_all(precision & refinement))
    report = dict(status="AR2_REFERENCE_QUALIFIED" if qualified else "AR2_REFERENCE_NEEDS_REPAIR_OR_REFINEMENT",
        qualified=qualified, model_id="AR2", horizon=HORIZON,
        likelihood_joint_gaussian_max_error=law_error, endpoint_gradient_relative_fd_error=derivative_error,
        posterior_mean=means.numpy().tolist(), posterior_sd=scales.numpy().tolist(),
        cuts=cuts.numpy().tolist(), reference_expectations=expectations[-1].numpy().tolist(),
        function_names=NAMES, margins=MARGINS, empirical_allowance=allowance.numpy().tolist(),
        allowance_is_rigorous_bound=False, grids=protocol, log_normalizers=normalizers,
        grid_expectations=expectations.numpy().tolist(), precision_passed=precision.numpy().tolist(),
        refinement_passed=refinement.numpy().tolist(), posterior_comparisons="NOT_RUN",
        prior=dict(family="independent_normal", mean=[0., 0.], sd=PRIOR_SD),
        observation_precision="FP32 once, then exact promotion to FP64")
    return report, dict(axis=axis, density=density, predictive=predictive, means=means, scales=scales, cuts=cuts)


def reference_stage(output, *, extended=False):
    output = Path(output)
    data = datasets()
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent/"datasets.json").write_text(json.dumps(data, indent=2)+"\n")
    reports = {}
    for partition, dataset in data.items():
        report, tensors = prepare_reference(dataset["observations"], extended=extended)
        reports[partition] = report
        directory = output.parent/partition
        directory.mkdir()
        (directory/"reference.json").write_text(json.dumps(report, indent=2)+"\n")
        for name, tensor in tensors.items():
            tf.io.write_file(str(directory/(name+".tensor")), tf.io.serialize_tensor(tensor))
    qualified = all(report["qualified"] for report in reports.values())
    return dict(status="AR2_REFERENCES_QUALIFIED" if qualified else "AR2_REFERENCE_REFINEMENT_REQUIRED",
                qualified=qualified, partitions=reports, jit_compile=True,
                seeds={name: record["seed"] for name, record in data.items()},
                all_models_validated=False, posterior_comparisons="NOT_RUN",
                next_action="Existing master: scoped DCTR controls, public HMC tuning and posterior comparison")
