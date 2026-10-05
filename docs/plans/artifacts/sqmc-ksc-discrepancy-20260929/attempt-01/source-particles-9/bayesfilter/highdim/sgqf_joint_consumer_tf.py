"""Optional exact SGQF-joint conditional consumer; no filtering promotion.

The Gaussian approximation is preserved analytically instead of converted to
a polynomial TT. A05 derives its backwards joint and forward conditional.
Gaussian setup is host-side TensorFlow; repeated sampling is batch-native XLA.
"""
from dataclasses import dataclass
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal

D = tf.float64


@dataclass(frozen=True)
class GaussianRetainedProposal:
    chart: obs.Chart

    def physical_log_density(self, x):
        return self.chart.log_prob(x)


@dataclass(frozen=True)
class SGQFJointStep:
    current_chart: obs.Chart
    previous_marginal: obs.Chart
    conditional_factor: tf.Tensor
    gain: tf.Tensor
    time_index: int
    retained_proposal: GaussianRetainedProposal

    def conditional_log_density(self, x, previous):
        mean = self.current_chart.mean + tf.linalg.matmul(
            previous-self.previous_marginal.mean, self.gain, transpose_b=True)
        standardized = tf.transpose(tf.linalg.triangular_solve(
            self.conditional_factor, tf.transpose(x-mean)))
        return _log_standard_normal(standardized)-tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(self.conditional_factor)))


def make_sgqf_joint_step(model, current, incoming, time_index):
    """Preserve q_t(x) times the incoming Gaussian's backward conditional."""
    d = model.dimension
    if incoming is None:
        if time_index != 0:
            raise ValueError('A noninitial joint needs its incoming Gaussian')
        return SGQFJointStep(current, current, current.factor, tf.zeros([d, d], D),
                             time_index, GaussianRetainedProposal(current))
    P = incoming.factor @ tf.transpose(incoming.factor)
    Pt = current.factor @ tf.transpose(current.factor)
    A = model.transition
    S = A @ P @ tf.transpose(A)+model.sigma**2*tf.eye(d, dtype=D)
    Ls = obs.spd_factor(S, 'SGQF predictive covariance')
    K = tf.transpose(tf.linalg.cholesky_solve(Ls, A @ P))
    B = P-K @ S @ tf.transpose(K)
    obs.spd_factor(B, 'SGQF backward covariance')
    b = incoming.mean+tf.linalg.matvec(K, current.mean-tf.linalg.matvec(A, incoming.mean))
    previous = obs.Chart.from_moments(b, B+K @ Pt @ tf.transpose(K))
    cross = Pt @ tf.transpose(K)
    gain = tf.transpose(tf.linalg.cholesky_solve(previous.factor, tf.transpose(cross)))
    factor = obs.spd_factor(Pt-gain @ tf.transpose(cross), 'SGQF forward conditional covariance')
    return SGQFJointStep(current, previous, factor, gain, time_index,
                         GaussianRetainedProposal(current))


@lru_cache(None)
def compiled_gaussian_conditional(dimension, jit_compile=True):
    vector = tf.TensorSpec([dimension], D)
    matrix = tf.TensorSpec([dimension, dimension], D)
    batch = tf.TensorSpec([None, dimension], D)

    @tf.function(input_signature=[batch, batch, vector, vector, matrix, matrix],
                 jit_compile=jit_compile, autograph=False)
    def sample(previous, noise, current_mean, previous_mean, gain, factor):
        mean = current_mean+tf.linalg.matmul(previous-previous_mean, gain, transpose_b=True)
        x = mean+tf.linalg.matmul(noise, factor, transpose_b=True)
        logq = _log_standard_normal(noise)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
        return x, logq
    return sample


def sample_joint_step(step, previous, seed, jit_compile=True):
    """Consumer endpoint for a frozen Gaussian/TT choice at each time."""
    if isinstance(step, obs.PairTTStep):
        return obs.sample_pair_tt_step(step, previous, seed, jit_compile)
    if not isinstance(step, SGQFJointStep):
        raise TypeError('Expected SGQFJointStep or PairTTStep')
    d = int(step.current_chart.mean.shape[0])
    noise = tf.random.stateless_normal(tf.shape(previous), [seed, 2], dtype=D)
    x, logq = compiled_gaussian_conditional(d, jit_compile)(
        previous, noise, step.current_chart.mean, step.previous_marginal.mean,
        step.gain, step.conditional_factor)
    obs.finite(x, 'SGQF joint conditional draw')
    obs.finite(logq, 'SGQF joint conditional density')
    return x, logq, {'sgqf_joint_preserved': True, 'finite': True}
