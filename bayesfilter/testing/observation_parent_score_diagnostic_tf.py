"""Diagnostic SIR conditional proposals using the complete shared recurrence.

Not a canonical route, tuned candidate, or HMC score. Categorical parent
indices are locally fixed when differentiating. No autodiff or pfor is used.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, mv, stack, exp, log, sqrt, logsumexp
from bayesfilter.highdim.covariance_proposal_score_tf import draw_from_maps
from bayesfilter.highdim.covariance_proposal_score_batch_tf import gaussian_mixture
from bayesfilter.highdim.covariance_proposal_tf import fixed_design_indices


def choose(mask, yes, no):
    return Jet(tf.where(mask, yes.v, no.v),
               tf.where(mask[None], yes.t, no.t))


def gather_parents(x, indices):
    p = x.t.shape[0]
    return Jet(tf.gather(x.v, indices, axis=1, batch_dims=1),
               tf.gather(x.t, tf.broadcast_to(indices[None], [p, *indices.shape]),
                         axis=2, batch_dims=2))


def conditional_geometry(models, means, observation):
    if models.kind != 'sir_d18':
        raise ValueError('diagnostic requires SIR Q=I and odd-coordinate Gaussian observation')
    r = 100*exp(2*models.q[:, 2])
    residual = models.c(observation)-models.observation(means)
    lp = -.5*(9*math.log(2*math.pi)+9*log(1+r)[:, None]
               +(residual*residual).sum(-1)/(1+r)[:, None])
    normalizer = logsumexp(lp, 1)
    log_alpha = lp-normalizer[:, None]
    log_z = normalizer-math.log(means.v.shape[1])
    cm = means+mv(models.H.transpose()[:, None], residual)/(1+r)[:, None, None]
    mask = models.c(tf.cast(tf.range(18) % 2, means.v.dtype))
    diag = 1-mask/(1+r)[:, None]
    cov = Jet(tf.linalg.diag(diag.v)[:, None], tf.linalg.diag(diag.t)[:, :, None])
    return cm, diag, cov, log_alpha, log_z


def adapted_indices(log_alpha, uniforms):
    cdf = tf.cumsum(tf.exp(log_alpha.v), axis=1)
    cdf = tf.concat([cdf[:, :-1], tf.ones_like(cdf[:, -1:])], axis=1)
    u = tf.broadcast_to(uniforms[None, :, 1], cdf.shape)
    return tf.searchsorted(cdf, u, side='right', out_type=tf.int32)


def make_hook(modes, controls):
    """Rows: 0 existing proposal, 1 conditional/uniform, 2 conditional/adapted."""
    modes = tuple(modes)
    if any(mode not in (0, 1, 2) for mode in modes):
        raise ValueError('unknown diagnostic proposal mode')

    def hook(models, maps, beta, uniforms, innovations, observation):
        means, cov = maps[:2]
        n = innovations.shape[0]
        mode = tf.constant(modes, tf.int32)
        branch, uniform_parent = fixed_design_indices(beta, uniforms, n)
        cm, diag, conditional_cov, log_alpha, log_z = conditional_geometry(models, maps[4], observation)
        adapted_parent = adapted_indices(log_alpha, uniforms)
        parents = tf.where(mode[:, None] == 2, adapted_parent, uniform_parent[None])
        exact_child = gather_parents(cm, parents)+sqrt(diag)[:, None]*models.c(innovations[None])
        original_child = draw_from_maps(maps, beta, uniforms, innovations)
        replace = (mode[:, None, None] != 0) & (branch[None, :, None] == 1)
        child = choose(replace, exact_child, original_child)
        parts = [gaussian_mixture(child, means[:, b], cov[:, b],
                  shared_covariance=True, contraction=controls.shared_gaussian_contraction)
                 for b in range(3)]
        conditional, cv = gaussian_mixture(child, cm, conditional_cov,
            shared_covariance=True, contraction=controls.shared_gaussian_contraction)
        # Exact Bayes identity avoids another weighted N-by-N kernel:
        # q_adapt(x) = g(y|x) p_N(x) / Z_N(y).
        adapted = models.observation_density(child, observation)+parts[0][0]-log_z[:, None]
        middle = choose(mode[:, None] == 2, adapted, conditional)
        middle = choose(mode[:, None] == 0, parts[1][0], middle)
        densities = stack([parts[0][0], middle, parts[2][0]], -1)
        valid = tf.reduce_all(tf.stack([x[1] for x in parts]+[cv]), axis=0)
        selected = tf.where((mode[:, None] == 2) & (branch[None] == 1), parents, uniform_parent[None])
        return child, densities, valid, selected
    return hook
