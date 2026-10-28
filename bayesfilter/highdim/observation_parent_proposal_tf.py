"""Conditional Gaussian proposal with observation-weighted parent selection.

Shared analytical recurrence for affine Gaussian observations and additive
Gaussian transitions. This is a covariance-proposal policy, not canonical LEDH
admission. Parent indices are locally fixed in the analytical derivative.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, mm, mv, solve, log, logsumexp, stack, sym
from bayesfilter.highdim.covariance_proposal_score_tf import checked_chol, draw_from_maps
from bayesfilter.highdim.covariance_proposal_tf import fixed_design_indices

POLICY_ID = 'conditional_observation_weighted'


def choose(mask, yes, no):
    return Jet(tf.where(mask, yes.v, no.v), tf.where(mask[None], yes.t, no.t))


def gather_parents(x, indices):
    p = x.t.shape[0]
    return Jet(tf.gather(x.v, indices, axis=1, batch_dims=1),
               tf.gather(x.t, tf.broadcast_to(indices[None], [p, *indices.shape]),
                         axis=2, batch_dims=2))


def gaussian_log_density(residual, factor):
    white = solve(factor[:, None], residual[..., None], triangular=True)[..., 0]
    diagonal = Jet(tf.linalg.diag_part(factor.v), tf.linalg.diag_part(factor.t))
    return -.5*(residual.v.shape[-1]*math.log(2*math.pi)
                 + 2*log(diagonal).sum(-1)[:, None] + (white*white).sum(-1))


def conditional_geometry(models, means, observation):
    if models.kind in ('ksc_mixture', 'range_bearing') or not models.constant_observation_jacobian:
        raise ValueError('conditional proposal requires affine Gaussian observations; '
                         'select original_three_branch explicitly for this model')
    h, q, r = models.H, models.Q, models.R
    s = sym(mm(mm(h, q), h.transpose())+r)
    ls, vs, _ = checked_chol(s, batch_ndims=1)
    # Cholesky solves reuse the checked factor and propagate all partials.
    qht = mm(q, h.transpose())
    gain = solve(ls.transpose(), solve(ls, qht.transpose(), triangular=True)).transpose()
    residual = models.c(observation)-models.observation(means)
    cm = means+mv(gain[:, None], residual)
    eye = models.c(tf.eye(q.v.shape[-1], batch_shape=[models.batch], dtype=q.v.dtype))
    a = eye-mm(gain, h)
    # Joseph identity: algebraically Q-K S K', avoiding subtractive cancellation.
    covariance = sym(mm(mm(a, q), a.transpose())+mm(mm(gain, r), gain.transpose()))
    lc, vc, _ = checked_chol(covariance, batch_ndims=1)
    lr, vr, _ = checked_chol(r, batch_ndims=1)
    lp = gaussian_log_density(residual, ls)
    normalizer = logsumexp(lp, 1)
    return cm, covariance[:, None], lc, lp-normalizer[:, None], normalizer-math.log(means.v.shape[1]), lr, vs & vc & vr


def adapted_indices(log_alpha, uniforms):
    cdf = tf.cumsum(tf.exp(log_alpha.v), axis=1)
    cdf = tf.concat([cdf[:, :-1], tf.ones_like(cdf[:, -1:])], axis=1)
    return tf.searchsorted(cdf, tf.broadcast_to(uniforms[None, :, 1], cdf.shape),
                           side='right', out_type=tf.int32)


def make_conditional_proposal(spec, controls, batch_size, *, precision_policy='uniform'):
    """Replace global branch only; preserve defensive identity/local branches."""
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import precision_gaussian_mixture
    weighted = controls.proposal_policy == POLICY_ID
    if controls.proposal_policy not in (POLICY_ID, 'conditional_uniform'):
        raise ValueError('conditional proposal policy required')

    def hook(models, maps, beta, uniforms, innovations, observation):
        means, cov = maps[:2]
        n = innovations.shape[0]
        branch, uniform_parent = fixed_design_indices(beta, uniforms, n)
        cm, cc, lc, alpha, log_z, lr, geometry_valid = conditional_geometry(models, maps[4], observation)
        parents = adapted_indices(alpha, uniforms) if weighted else tf.broadcast_to(uniform_parent[None], [batch_size, n])
        conditional_child = gather_parents(cm, parents)+mv(lc[:, None], models.c(innovations[None]))
        child = choose(branch[None, :, None] == 1, conditional_child,
                       draw_from_maps(maps, beta, uniforms, innovations))
        def density(mu, covariance):
            return precision_gaussian_mixture(child, mu, covariance, shared_covariance=True,
                contraction=controls.shared_gaussian_contraction, precision_policy=precision_policy)
        prior, pv = density(means[:, 0], cov[:, 0])
        local, lv = density(means[:, 2], cov[:, 2])
        if weighted:
            # Use the observation law of the proposal geometry, including its
            # finite-precision R expression; target density remains independent.
            likelihood = gaussian_log_density(models.c(observation)-models.observation(child), lr)
            middle, cv = likelihood+prior-log_z[:, None], geometry_valid
        else:
            middle, cv = density(cm, cc)
        selected = tf.where(branch[None] == 1, parents, uniform_parent[None])
        return child, stack([prior, middle, local], -1), geometry_valid & pv & lv & cv, selected
    return hook
