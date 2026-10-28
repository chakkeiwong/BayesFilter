"""Conditional Gaussian proposal with observation-weighted parent selection.

Shared analytical recurrence for additive Gaussian transitions. Affine Gaussian
and KSC-mixture observations admit exact conditioning; range-bearing uses an
explicit local Gaussian proposal with the true wrapped target left intact. This is a covariance-proposal policy, not canonical LEDH
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


def gaussian_log_density(residual, factor, *, shared=True):
    white = solve(factor[:, None] if shared else factor,
                  residual[..., None], triangular=True)[..., 0]
    diagonal = Jet(tf.linalg.diag_part(factor.v), tf.linalg.diag_part(factor.t))
    logdet = 2*log(diagonal).sum(-1)
    return -.5*(residual.v.shape[-1]*math.log(2*math.pi)
                 + (logdet[:, None] if shared else logdet) + (white*white).sum(-1))


def ksc_conditional_geometry(models, means, observation):
    """Exact parent/component conditioning for the existing seven-mixture law."""
    b, n, d = means.v.shape
    if d != 1:
        raise ValueError('KSC mixture conditioning requires its scalar state')
    q = models.Q[:, 0, 0]
    s = q[:, None]+models.variances[None]
    offset = models.means[None, None]+2*models.q[:, 1, None, None]
    residual = models.c(observation[0])-means[..., 0, None]-offset
    lp = log(models.weights)[None, None]-.5*(
        math.log(2*math.pi)+log(s)[:, None]+residual*residual/s[:, None])
    cm = (means[..., 0, None]+q[:, None, None]*residual/s[:, None])[..., None]
    covariance = (q[:, None]*models.variances[None]/s)[:, None, :, None, None]
    cm = cm.reshape([b, n*7, 1])
    covariance = covariance.broadcast([b, n, 7, 1, 1]).reshape([b, n*7, 1, 1])
    lc, valid, _ = checked_chol(covariance, batch_ndims=1)
    lp = lp.reshape([b, n*7])
    normalizer = logsumexp(lp, 1)
    return cm, covariance, lc, lp-normalizer[:, None], normalizer-math.log(n), None, valid


def range_bearing_conditional_geometry(models, means, observation):
    """Local polar conditioning; its density is evaluated explicitly downstream."""
    h = models._polar(means, jacobian=True)
    q, r = models.Q[:, None], models.R[:, None]
    s = sym(mm(mm(h, q), h.transpose())+r)
    ls, vs, _ = checked_chol(s, batch_ndims=1)
    gain = solve(ls.transpose(), solve(ls, mm(q, h.transpose()).transpose(),
                                      triangular=True)).transpose()
    raw = models.c(observation)-models.observation(means)
    angle = Jet(tf.atan2(tf.sin(raw.v[..., 1]), tf.cos(raw.v[..., 1])), raw.t[..., 1])
    residual = stack([raw[..., 0], angle], -1)
    cm = means+mv(gain, residual)
    eye = models.c(tf.eye(models.d, batch_shape=[models.batch, 1], dtype=means.v.dtype))
    a = eye-mm(gain, h)
    covariance = sym(mm(mm(a, q), a.transpose())+mm(mm(gain, r), gain.transpose()))
    lc, vc, _ = checked_chol(covariance, batch_ndims=1)
    lp = gaussian_log_density(residual, ls, shared=False)
    normalizer = logsumexp(lp, 1)
    return cm, covariance, lc, lp-normalizer[:, None], normalizer-math.log(means.v.shape[1]), None, vs & vc


def conditional_geometry(models, means, observation):
    if models.kind == 'ksc_mixture':
        return ksc_conditional_geometry(models, means, observation)
    if models.kind == 'range_bearing':
        return range_bearing_conditional_geometry(models, means, observation)
    if not models.constant_observation_jacobian:
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
    return tf.searchsorted(cdf, tf.broadcast_to(uniforms[None, :, 1], [cdf.shape[0], uniforms.shape[0]]),
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
        if models.kind == 'ksc_mixture' and not weighted:
            within = alpha.reshape([batch_size, n, 7])
            alpha = (within-logsumexp(within, 2)[:, :, None]-math.log(n)).reshape([batch_size, n*7])
        parents = (adapted_indices(alpha, uniforms) if weighted or models.kind == 'ksc_mixture'
                   else tf.broadcast_to(uniform_parent[None], [batch_size, n]))
        selected_factor = gather_parents(lc, parents) if lc.v.shape.rank == 4 else lc[:, None]
        conditional_child = gather_parents(cm, parents)+mv(selected_factor, models.c(innovations[None]))
        child = choose(branch[None, :, None] == 1, conditional_child,
                       draw_from_maps(maps, beta, uniforms, innovations))
        def density(mu, covariance, *, shared=True, log_weights=None):
            return precision_gaussian_mixture(child, mu, covariance, shared_covariance=shared,
                contraction=controls.shared_gaussian_contraction, precision_policy=precision_policy,
                log_weights=log_weights)
        prior, pv = density(means[:, 0], cov[:, 0])
        local, lv = density(means[:, 2], cov[:, 2], shared=models.constant_observation_jacobian)
        if models.kind == 'range_bearing':
            middle, cv = density(cm, cc, shared=False, log_weights=alpha if weighted else None)
        elif models.kind == 'ksc_mixture':
            if weighted:
                middle, cv = models.observation_density(child, observation)+prior-log_z[:, None], geometry_valid
            else:
                cmk, cck = cm.reshape([batch_size, n, 7, 1]), cc.reshape([batch_size, n, 7, 1, 1])
                ak = alpha.reshape([batch_size, n, 7])
                terms = [density(cmk[:, :, k], cck[:, :, k], shared=False, log_weights=ak[:, :, k]) for k in range(7)]
                middle = logsumexp(stack([term[0] for term in terms], -1), -1)
                cv = tf.reduce_all(tf.stack([term[1] for term in terms]), 0)
        elif weighted:
            # Use the observation law of the proposal geometry, including its
            # finite-precision R expression; target density remains independent.
            likelihood = gaussian_log_density(models.c(observation)-models.observation(child), lr)
            middle, cv = likelihood+prior-log_z[:, None], geometry_valid
        else:
            middle, cv = density(cm, cc)
        selected = tf.where(branch[None] == 1, parents, uniform_parent[None])
        return child, stack([prior, middle, local], -1), geometry_valid & pv & lv & cv, selected
    return hook
