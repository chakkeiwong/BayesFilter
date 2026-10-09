"""Rare-region research kernels, with explicit measures and no promotion defaults.

EMUS: Thiede et al. (2016), equations 14--17; author emus.py retained locally.
Splitting: fixed-level conditional probability identity, not idealized AMS.
Global MH/MALA reuse the source-mapped Gabrié primitives. All numerical
arrays are TensorFlow; static loops enumerate components, never samples.
"""
from __future__ import annotations

import math
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.neutra_warm_start_tf import (
    independence_step, mala_step, value_score, systematic_indices)
from bayesfilter.testing.neutra_warm_start_targets_tf import F64, BroadStudentProposal


def seed_fold(seed, index):
    return tf.random.experimental.stateless_fold_in(seed, index)


def event_features(target, x):
    y = x[:, 1] - (.1*(x[:, 0]**2-26.) if target.name == 'warped_mixture' else 0.)
    return tf.stack((tf.cast(tf.abs(x[:, 0]) < 2., F64),
        tf.cast(tf.abs(x[:, 0]) < .1, F64), tf.cast(x[:, 0] > 0., F64),
        x[:, 0], y, x[:, 0]**2, y*y), axis=1)


def known_masses():
    # Stable complementary-error-function subtraction, independent of TF density.
    def mass(b):
        return .5*(math.erfc((5-b)/math.sqrt(2))-math.erfc((5+b)/math.sqrt(2)))
    right = 2/3 - (1/3)*.5*math.erfc(5/math.sqrt(2))
    return [mass(2.), mass(.1), right, 5/3, 0., 26., 1.]


class Proposal:
    """Normalized Laplace/bridge/Student mixture; true masses are never supplied."""
    def __init__(self, target, *, bridge_weight=0., narrow_scale=.08, flows=None):
        centers = target.known_representatives()
        self.dimension = 2
        self.flows = flows
        self.logits = tf.Variable(tf.zeros([2], F64), trainable=False)
        self.defensive = BroadStudentProposal(2).distribution
        h = -target.hessian(centers)
        self.local = tfp.distributions.MultivariateNormalTriL(
            centers, tf.linalg.cholesky(tf.linalg.inv(h)))
        middle = [0., -2.6 if target.name == 'warped_mixture' else 0.]
        self.bridge = tfp.distributions.MultivariateNormalDiag(
            tf.constant([middle, middle], F64), tf.constant([[1., 1.], [narrow_scale, 1.]], F64))
        self.bridge_weight = float(bridge_weight)
        if not 0 <= self.bridge_weight < .95:
            raise ValueError('bridge weight outside its declared range')

    def component_log_prob(self, x):
        if self.flows is None:
            local = self.local.log_prob(x[:, None, :])
        else:
            local = tf.stack([f.log_prob(x) for f in self.flows], axis=1)
        return tf.concat((local, self.bridge.log_prob(x[:, None, :]),
                          self.defensive.log_prob(x)[:, None]), axis=1)

    def weights(self):
        b = self.bridge_weight
        return tf.concat(((.95-b)*tf.nn.softmax(self.logits), tf.constant([b/2, b/2, .05], F64)), 0)

    def log_prob(self, x):
        return tf.reduce_logsumexp(self.component_log_prob(x)+tf.math.log(self.weights())[None, :], 1)

    def sample(self, count, seed):
        labels = tf.random.stateless_categorical(tf.math.log(self.weights())[None, :], count, seed)[0]
        if self.flows is None:
            local = self.local.sample(count, seed=seed_fold(seed, 1))
        else:
            local = tf.stack([f.forward_and_logdet(tf.random.stateless_normal(
                [count, 2], seed_fold(seed, 1+k), dtype=F64))[0] for k, f in enumerate(self.flows)], axis=1)
        all_x = tf.concat((local, self.bridge.sample(count, seed=seed_fold(seed, 3)),
                          self.defensive.sample(count, seed=seed_fold(seed, 4))[:, None, :]), axis=1)
        return tf.gather(all_x, labels, axis=1, batch_dims=1)


class ImportanceProgram:
    def __init__(self, target, proposal, count, *, jit_compile=True):
        def run(seed):
            x = proposal.sample(count, seed)
            lw = tf.nn.log_softmax(target.log_prob_kernel(x)-proposal.log_prob(x))
            f = event_features(target, x)
            w = tf.exp(lw)
            estimate = tf.reduce_sum(w[:, None]*f, 0)
            se = tf.sqrt(tf.reduce_sum(w[:, None]**2*(f-estimate)**2, 0))
            return x, lw, estimate, se
        self.run = tf.function(run, input_signature=[tf.TensorSpec([2], tf.int32)],
                               jit_compile=jit_compile, autograph=False)


class GlobalProgram:
    """Frozen global/local composition; optional logit adaptation is warmup only."""
    def __init__(self, target, proposal, steps, *, adapt=False, halfspace=0, jit_compile=True):
        def run(x, dt, seed):
            trace = tf.TensorArray(F64, size=steps)
            def body(i, current, trace, ga, la, bad):
                key = seed_fold(seed, i)
                if halfspace:
                    proposed = current
                    acc = tf.zeros([tf.shape(x)[0]], tf.bool)
                    valid = tf.ones_like(acc)
                else:
                    proposed = proposal.sample(tf.shape(x)[0], key)
                    current, acc, _, valid = independence_step(target.log_prob_kernel,
                        proposal.log_prob, current, proposed,
                        tf.random.stateless_uniform([tf.shape(x)[0]], seed_fold(key, 10), dtype=F64))
                candidate, acc_local, _, valid_local = mala_step(target._value_score, current,
                    tf.random.stateless_normal(tf.shape(x), seed_fold(key, 11), dtype=F64),
                    tf.random.stateless_uniform([tf.shape(x)[0]], seed_fold(key, 12), dtype=F64), dt)
                if halfspace:
                    acc_local &= (candidate[:, 0]*halfspace > 0)
                    candidate = tf.where(acc_local[:, None], candidate, current)
                current = candidate
                if adapt:
                    # Gradient of cross entropy for softmax mixture logits,
                    # with nonlocal and defensive weights held fixed.
                    scores = proposal.component_log_prob(tf.stop_gradient(current))
                    responsibilities = tf.nn.softmax(scores+tf.math.log(proposal.weights()), axis=1)
                    r = responsibilities[:, :2]
                    gradient = tf.reduce_mean(tf.reduce_sum(r, 1)[:, None]*tf.nn.softmax(proposal.logits)-r, 0)
                    proposal.logits.assign_sub(tf.constant(.01, F64)*gradient)
                return (i+1, current, trace.write(i, current), ga+tf.reduce_mean(tf.cast(acc, F64)),
                    la+tf.reduce_mean(tf.cast(acc_local, F64)),
                    bad+tf.reduce_sum(tf.cast(~valid | ~valid_local, tf.int32)))
            _, x, trace, ga, la, bad = tf.while_loop(lambda i, *_: i < steps, body,
                (tf.constant(0), x, trace, tf.constant(0., F64), tf.constant(0., F64), tf.constant(0)))
            return x, trace.stack(), ga/steps, la/steps, bad
        self.run = tf.function(run, input_signature=[tf.TensorSpec([None, 2], F64),
            tf.TensorSpec([], F64), tf.TensorSpec([2], tf.int32)], jit_compile=jit_compile, autograph=False)


class UmbrellaProgram:
    def __init__(self, target, centers, scales, walkers, steps, *, jit_compile=True):
        self.centers, self.scales = tf.constant(centers, F64), tf.constant(scales, F64)
        k = len(centers)
        own_c = tf.repeat(self.centers, walkers)
        own_s = tf.repeat(self.scales, walkers)
        def biased(x):
            return target.log_prob_kernel(x)-.5*((x[:, 0]-own_c)/own_s)**2
        def run(x, dt, seed):
            trace = tf.TensorArray(F64, size=steps)
            def body(i, current, trace, accepted, invalid):
                key = seed_fold(seed, i)
                current, acc, _, valid = mala_step(lambda y: value_score(biased, y), current,
                    tf.random.stateless_normal(tf.shape(x), key, dtype=F64),
                    tf.random.stateless_uniform([k*walkers], seed_fold(key, 1), dtype=F64), dt)
                return i+1, current, trace.write(i, current), accepted+tf.cast(acc, F64), invalid+tf.cast(~valid, tf.int32)
            _, last, trace, acc, bad = tf.while_loop(lambda i, *_: i < steps, body,
                (tf.constant(0), x, trace, tf.zeros([k*walkers], F64), tf.zeros([k*walkers], tf.int32)))
            rows = tf.transpose(tf.reshape(trace.stack(), [steps, k, walkers, 2]), [1, 0, 2, 3])
            return last, tf.reshape(rows, [k, steps*walkers, 2]), acc/steps, bad
        self.run = tf.function(run, input_signature=[tf.TensorSpec([k*walkers, 2], F64),
            tf.TensorSpec([], F64), tf.TensorSpec([2], tf.int32)], jit_compile=jit_compile, autograph=False)


@tf.function(input_signature=[tf.TensorSpec([None, None], F64)], jit_compile=True, autograph=False)
def emus_stationary(matrix):
    n = tf.shape(matrix)[0]
    a = tf.transpose(matrix)-tf.eye(n, dtype=F64)
    a = tf.concat((a[:-1], tf.ones([1, n], F64)), 0)
    b = tf.concat((tf.zeros([n-1], F64), tf.ones([1], F64)), 0)
    z = tf.linalg.solve(a, b[:, None])[:, 0]
    residual = tf.reduce_max(tf.abs(tf.linalg.matvec(matrix, z, transpose_a=True)-z))
    return z, residual


class EmusProgram:
    def __init__(self, centers, scales, *, jit_compile=True):
        c, s = tf.constant(centers, F64), tf.constant(scales, F64)
        k = len(centers)
        def run(rows):
            logpsi = -.5*((rows[:, :, 0, None]-c[None, None, :])/s[None, None, :])**2
            logs = tf.reduce_logsumexp(logpsi, -1)
            matrix = tf.reduce_mean(tf.exp(logpsi-logs[:, :, None]), 1)
            z, residual = emus_stationary(matrix)
            lw = tf.math.log(z)[:, None]-logs-tf.math.log(tf.cast(tf.shape(rows)[1], F64))
            return tf.reshape(rows, [-1, 2]), tf.nn.log_softmax(tf.reshape(lw, [-1])), z, matrix, residual
        self.run = tf.function(run, input_signature=[tf.TensorSpec([k, None, 2], F64)],
                               jit_compile=jit_compile, autograph=False)


@tf.function(input_signature=[tf.TensorSpec([None], F64), tf.TensorSpec([], F64)],
             jit_compile=True, autograph=False)
def reflected_coordinate(x, bound):
    # Fold a Gaussian random walk into [-bound,bound]. Its transition is
    # symmetric (sum of reflected Gaussian images), so no proposal ratio.
    period = 4*bound
    y = tf.math.floormod(x+bound, period)
    return tf.where(y <= 2*bound, y-bound, 3*bound-y)


class ConstrainedProgram:
    def __init__(self, target, steps, *, jit_compile=True):
        def run(x, bound, seed):
            scale = tf.stack((.5*tf.minimum(bound, 1.), tf.constant(.75, F64)))
            def body(i, x, accepted):
                key = seed_fold(seed, i)
                y = x+scale*tf.random.stateless_normal(tf.shape(x), key, dtype=F64)
                y = tf.stack((reflected_coordinate(y[:, 0], bound), y[:, 1]), 1)
                ratio = target.log_prob_kernel(y)-target.log_prob_kernel(x)
                acc = tf.math.log(tf.random.stateless_uniform([tf.shape(x)[0]], seed_fold(key, 1), dtype=F64)) < tf.minimum(ratio, 0.)
                return i+1, tf.where(acc[:, None], y, x), accepted+tf.reduce_mean(tf.cast(acc, F64))
            _, x, accepted = tf.while_loop(lambda i, *_: i < steps, body,
                (tf.constant(0), x, tf.constant(0., F64)))
            return x, accepted/steps
        self.run = tf.function(run, input_signature=[tf.TensorSpec([None, 2], F64),
            tf.TensorSpec([], F64), tf.TensorSpec([2], tf.int32)], jit_compile=jit_compile, autograph=False)


def shell_weights(mass, count):
    return tf.math.log(tf.cast(mass, F64))-tf.math.log(tf.cast(count, F64))


def stratified_bank(rows, log_weights):
    """Four disjoint regions; omit absent strata, never invent observations."""
    a = tf.abs(rows[:, 0])
    labels = tf.where(a < .1, 0, tf.where(a < 2., 1, tf.where(rows[:, 0] < 0, 2, 3)))
    ids, conditional, masses = [], [], []
    normalized = tf.nn.log_softmax(log_weights)
    for k in range(4):
        index = tf.cast(tf.where(labels == k)[:, 0], tf.int32)
        if int(tf.size(index)):
            lw = tf.gather(normalized, index)
            mass = tf.reduce_logsumexp(lw)
            ids.append(index); conditional.append(lw-mass); masses.append(mass)
    width = max(int(tf.size(x)) for x in ids)
    indices = tf.stack([tf.pad(x, [[0, width-int(tf.size(x))]]) for x in ids])
    logs = tf.stack([tf.pad(x, [[0, width-int(tf.size(x))]], constant_values=-math.inf) for x in conditional])
    return indices, logs, tf.stack(masses)
