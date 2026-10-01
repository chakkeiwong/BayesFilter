"""Native Gaussian/Gauss-Hermite moment preparation for frozen TT hints.

This preserves the existing C2 Gaussian companion-filter extension. It is not
the particle-estimated bridge in Zhao-Cui Section 5.2 or author full_sol.m.
The complete horizon is compiled; engine callbacks only index frozen tensors.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig


def _normal_hermite_rule(gh_points, jit_compile):
    """Golub-Welsch nodes and probability weights for a standard normal."""
    d = tf.float64
    diagonal = tf.sqrt(tf.cast(tf.range(1, gh_points), d))
    jacobi = tf.linalg.diag(diagonal, k=1) + tf.linalg.diag(diagonal, k=-1)
    if jit_compile:
        nodes, _ = xla_self_adjoint_eig(
            jacobi, lower=True, max_iter=100, epsilon=2.220446049250313e-16,
        )
        nodes.set_shape([gh_points])
    else:
        nodes = tf.linalg.eigvalsh(jacobi)
    # XLA's Jacobi eigenvalues are accurate here, but its eigenvectors lose
    # about 1e-9 in the probability weights even with FP64 stopping epsilon.
    # Use the equivalent Hermite formula w_i = 1/(n*p_{n-1}(x_i)^2), where
    # p_k = He_k/sqrt(k!). This native normalized recurrence also avoids
    # factorial overflow. It is the same Gauss-Hermite rule, not a retuning.
    def polynomial_step(k, previous, current):
        order = tf.cast(k, d)
        following = (nodes * current - tf.sqrt(order-1.0) * previous) / tf.sqrt(order)
        return k+1, current, following

    _, _, polynomial = tf.while_loop(
        lambda k, *_: k < gh_points, polynomial_step,
        (tf.constant(1), tf.zeros_like(nodes), tf.ones_like(nodes)),
        maximum_iterations=gh_points-1, parallel_iterations=1,
    )
    weights = tf.math.reciprocal(tf.cast(gh_points, d) * tf.square(polynomial))
    return nodes, weights / tf.reduce_sum(weights)


@dataclass(frozen=True)
class PreparedMomentHints:
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    joint_means: tf.Tensor
    joint_covariances: tf.Tensor

    def callbacks(self):
        return (
            lambda observation: (self.initial_mean, self.initial_covariance),
            lambda date, observation: (
                tf.gather(self.joint_means, date - 1),
                tf.gather(self.joint_covariances, date - 1),
            ),
        )

    def filtered_covariances(self):
        n = self.initial_mean.shape[0]
        return tf.concat(
            [self.initial_covariance[None], self.joint_covariances[:, :n, :n]], 0,
        )


@lru_cache(maxsize=16)
def _sv_hint_kernel(horizon, n, gh_points, jit_compile):
    d = tf.float64

    @tf.function(
        input_signature=[
            tf.TensorSpec([horizon, n], d), tf.TensorSpec([n, n], d),
            tf.TensorSpec([n, n], d), tf.TensorSpec([n, n], d),
            tf.TensorSpec([], d),
        ],
        jit_compile=jit_compile, autograph=False,
    )
    def evaluate(observations, transition, process_covariance, initial_covariance, beta):
        nodes, weights = _normal_hermite_rule(gh_points, jit_compile)
        powers = tf.pow(tf.cast(gh_points, tf.int64), tf.range(n-1, -1, -1, dtype=tf.int64))
        digits = (tf.range(gh_points**n, dtype=tf.int64)[:, None] // powers[None]) % gh_points
        z = tf.gather(nodes, digits)
        w = tf.reduce_prod(tf.gather(weights, digits), axis=1)

        def update(mean, covariance, observation):
            cloud = mean[None] + tf.linalg.matmul(z, tf.linalg.cholesky(covariance), transpose_b=True)
            logw = tf.reduce_sum(
                -0.5 * tf.constant(math.log(2.0 * math.pi), d) - tf.math.log(beta)
                - cloud / 2.0 - tf.square(observation)[None] * tf.exp(-cloud) / (2.0 * beta**2),
                axis=-1,
            )
            weighted = w * tf.exp(logw - tf.reduce_max(logw))
            weighted /= tf.reduce_sum(weighted)
            filtered_mean = tf.linalg.matvec(cloud, weighted, transpose_a=True)
            centered = cloud - filtered_mean[None]
            filtered_covariance = tf.linalg.matmul(centered * weighted[:, None], centered, transpose_a=True)
            return filtered_mean, filtered_covariance

        initial_mean, initial_cov = update(tf.zeros([n], d), initial_covariance, observations[0])
        if horizon == 1:
            return (
                initial_mean, initial_cov,
                tf.zeros([0, 2*n], d), tf.zeros([0, 2*n, 2*n], d),
            )

        def step(t, mean, cov, means, covariances):
            mean_p = tf.linalg.matvec(transition, mean)
            cov_p = tf.linalg.matmul(tf.linalg.matmul(transition, cov), transition, transpose_b=True) + process_covariance
            cross = tf.linalg.matmul(cov, transition, transpose_b=True)
            filtered_mean, filtered_cov = update(mean_p, cov_p, observations[t])
            gain = tf.linalg.matmul(cross, tf.linalg.inv(cov_p))
            previous_mean = mean + tf.linalg.matvec(gain, filtered_mean - mean_p)
            previous_cov = cov - tf.linalg.matmul(tf.linalg.matmul(gain, cov_p), gain, transpose_b=True) + tf.linalg.matmul(tf.linalg.matmul(gain, filtered_cov), gain, transpose_b=True)
            cross_f = tf.linalg.matmul(gain, filtered_cov)
            joint_mean = tf.concat([filtered_mean, previous_mean], 0)
            joint_cov = tf.concat([
                tf.concat([filtered_cov, tf.transpose(cross_f)], 1),
                tf.concat([cross_f, previous_cov], 1),
            ], 0)
            return (
                t + 1, filtered_mean, filtered_cov,
                tf.tensor_scatter_nd_update(means, [[t-1]], joint_mean[None]),
                tf.tensor_scatter_nd_update(covariances, [[t-1]], joint_cov[None]),
            )

        _, _, _, means, covariances = tf.while_loop(
            lambda t, *_: t < horizon, step,
            (tf.constant(1), initial_mean, initial_cov,
             tf.zeros([horizon-1, 2*n], d), tf.zeros([horizon-1, 2*n, 2*n], d)),
            maximum_iterations=horizon-1, parallel_iterations=1,
        )
        return initial_mean, initial_cov, means, covariances

    return evaluate


def prepare_sv_gaussian_moment_hints(model, observations, *, gh_points=9, jit_compile=True):
    """Prepare the original C2 joint filtered moments for fixed observations."""
    observations = tf.convert_to_tensor(observations, tf.float64)
    n = int(model["n"])
    if observations.shape.rank != 2 or observations.shape[1] != n or not observations.shape[0]:
        raise ValueError("observations must have static shape [positive horizon,n]")
    if int(gh_points) < 2:
        raise ValueError("joint covariance hints require at least two GH points")
    values = _sv_hint_kernel(int(observations.shape[0]), n, int(gh_points), jit_compile)(
        observations, tf.convert_to_tensor(model["A"], tf.float64),
        tf.convert_to_tensor(model["Q"], tf.float64),
        tf.convert_to_tensor(model["P0"], tf.float64), tf.convert_to_tensor(model["beta"], tf.float64),
    )
    tf.debugging.assert_all_finite(values[0], "nonfinite initial hint")
    tf.debugging.assert_all_finite(values[1], "nonfinite initial covariance")
    tf.debugging.assert_all_finite(values[2], "nonfinite joint hints")
    tf.debugging.assert_all_finite(values[3], "nonfinite joint covariances")
    return PreparedMomentHints(*values)


@lru_cache(maxsize=16)
def _lgssm_hint_kernel(horizon, n, m, jit_compile):
    d = tf.float64

    @tf.function(
        input_signature=[
            tf.TensorSpec([horizon, m], d), tf.TensorSpec([n, n], d),
            tf.TensorSpec([n, n], d), tf.TensorSpec([m, n], d),
            tf.TensorSpec([m, m], d), tf.TensorSpec([n], d),
            tf.TensorSpec([n, n], d),
        ],
        jit_compile=jit_compile, autograph=False,
    )
    def evaluate(observations, transition, process_covariance, observation_matrix,
                 observation_covariance, initial_mean, initial_covariance):
        def update(mean, covariance, observation):
            projected = tf.linalg.matmul(covariance, observation_matrix, transpose_b=True)
            innovation_cov = tf.linalg.matmul(observation_matrix, projected) + observation_covariance
            inverse = tf.linalg.inv(innovation_cov)
            gain = tf.linalg.matmul(projected, inverse)
            innovation = observation - tf.linalg.matvec(observation_matrix, mean)
            filtered_mean = mean + tf.linalg.matvec(gain, innovation)
            filtered_cov = covariance - tf.linalg.matmul(
                tf.linalg.matmul(gain, innovation_cov), gain, transpose_b=True,
            )
            return filtered_mean, filtered_cov, gain, innovation, innovation_cov, inverse

        first_mean, first_cov, _, _, _, _ = update(initial_mean, initial_covariance, observations[0])
        if horizon == 1:
            return first_mean, first_cov, tf.zeros([0, 2*n], d), tf.zeros([0, 2*n, 2*n], d)

        def step(t, mean, cov, means, covariances):
            predicted_mean = tf.linalg.matvec(transition, mean)
            predicted_cov = tf.linalg.matmul(tf.linalg.matmul(transition, cov), transition, transpose_b=True) + process_covariance
            cross = tf.linalg.matmul(cov, transition, transpose_b=True)
            filtered_mean, filtered_cov, gain, innovation, innovation_cov, inverse = update(
                predicted_mean, predicted_cov, observations[t],
            )
            previous_gain = tf.linalg.matmul(
                tf.linalg.matmul(cross, observation_matrix, transpose_b=True), inverse,
            )
            previous_mean = mean + tf.linalg.matvec(previous_gain, innovation)
            previous_cov = cov - tf.linalg.matmul(
                tf.linalg.matmul(previous_gain, innovation_cov), previous_gain, transpose_b=True,
            )
            cross_filtered = cross - tf.linalg.matmul(
                tf.linalg.matmul(previous_gain, innovation_cov), gain, transpose_b=True,
            )
            joint_mean = tf.concat([filtered_mean, previous_mean], 0)
            joint_cov = tf.concat([
                tf.concat([filtered_cov, tf.transpose(cross_filtered)], 1),
                tf.concat([cross_filtered, previous_cov], 1),
            ], 0)
            return (
                t+1, filtered_mean, filtered_cov,
                tf.tensor_scatter_nd_update(means, [[t-1]], joint_mean[None]),
                tf.tensor_scatter_nd_update(covariances, [[t-1]], joint_cov[None]),
            )

        _, _, _, means, covariances = tf.while_loop(
            lambda t, *_: t < horizon, step,
            (tf.constant(1), first_mean, first_cov,
             tf.zeros([horizon-1, 2*n], d), tf.zeros([horizon-1, 2*n, 2*n], d)),
            maximum_iterations=horizon-1, parallel_iterations=1,
        )
        return first_mean, first_cov, means, covariances

    return evaluate


def prepare_lgssm_moment_hints(
    observations, transition, process_covariance, observation_matrix,
    observation_covariance, initial_mean, initial_covariance, *, jit_compile=True,
):
    """Prepare frozen Kalman joint hints, observing before the first transition.

    The LGSSM benchmark supplies its original frozen model/data arrays. This
    function does not generate new fixtures or alter their seeded draws.
    """
    tensors = tuple(tf.convert_to_tensor(value, tf.float64) for value in (
        observations, transition, process_covariance, observation_matrix,
        observation_covariance, initial_mean, initial_covariance,
    ))
    observations, transition = tensors[:2]
    if observations.shape.rank != 2 or not observations.shape.is_fully_defined() or not observations.shape[0]:
        raise ValueError("observations must have static shape [positive horizon,m]")
    if transition.shape.rank != 2 or not transition.shape.is_fully_defined():
        raise ValueError("transition must have a static matrix shape")
    values = _lgssm_hint_kernel(
        int(observations.shape[0]), int(transition.shape[0]), int(observations.shape[1]), jit_compile,
    )(*tensors)
    tf.debugging.assert_all_finite(values[0], "nonfinite initial hint")
    tf.debugging.assert_all_finite(values[1], "nonfinite initial covariance")
    tf.debugging.assert_all_finite(values[2], "nonfinite joint hints")
    tf.debugging.assert_all_finite(values[3], "nonfinite joint covariances")
    return PreparedMomentHints(*values)
