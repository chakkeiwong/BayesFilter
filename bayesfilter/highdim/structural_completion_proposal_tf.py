"""Innovation proposals for deterministic-completion state transitions.

This is a structural adaptation of the shared covariance proposal, not a
Gaussian density on a singular ambient transition. Model laws implement
complete(parent, innovation), observe(state), innovation_observation_jacobian,
and R using analytical Jets. Every leading parameter row remains batch native.
The flow/UKF map builder and subsequent filtering/reset recurrence are shared.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, mv, log, stack
from bayesfilter.highdim.covariance_proposal_score_tf import build_maps, checked_chol
from bayesfilter.highdim.covariance_proposal_tf import fixed_design_indices


class _ConditionalInnovationModel:
    def __init__(self, model, parents, innovation_dimension):
        self.model, self.parents = model, parents[..., None, :]
        self.batch, self.n = parents.v.shape[:2]
        self.r = innovation_dimension
        self.constant_observation_jacobian = False

    def apply(self, value_name, tangent_name, x, *args, theta=False):
        if value_name == "transition_mean_fn":
            return x * 0
        if value_name == "observation_fn":
            return self.model.observe(self.model.complete(self.parents, x))
        if value_name == "observation_jacobian_fn":
            return self.model.innovation_observation_jacobian(self.parents, x)
        raise ValueError("Unsupported innovation model operation: " + value_name)

    def covariance(self, value_name, tangent_name):
        if value_name == "process_covariance":
            return self.parents.constant(tf.eye(self.r, batch_shape=[self.batch, self.n],
                                                dtype=self.parents.v.dtype))
        if value_name == "observation_covariance":
            o = self.model.R.v.shape[-1]
            return self.model.R[:, None].broadcast([self.batch, self.n, o, o])
        raise ValueError(value_name)


def structural_proposal(model, x, observation, controls, beta, uniforms,
                        normals, *, indices=None):
    """Draw (j,e) and evaluate q(e|j), with no ambient-state Jacobian.

    Parents are uniform and fixed by the draw design. The same parent is used
    for all three innovation densities in the defensive mixture. Summing over
    other parents would be wrong when their completion curves differ.
    """
    if controls.proposal_policy == "conditional_observation_weighted":
        raise ValueError(
            "structural completion currently uses uniform parents; set "
            "proposal_policy='conditional_uniform'"
        )
    batch, n, _ = x.v.shape
    r = normals.shape[-1]
    b, j = fixed_design_indices(beta, uniforms, n) if indices is None else indices
    parents = x.gather(j, axis=1)
    local = _ConditionalInnovationModel(model, parents, r)
    zero = x.constant(tf.zeros([batch, n, 1, r], x.v.dtype))
    maps = build_maps(local, zero, observation, controls)
    means, covariance, a, c = maps[:4]
    selector = tf.one_hot(b, 3, dtype=x.v.dtype)[None, :, :, None, None]
    ab = (a * selector[..., None]).sum(2)[..., 0, :, :]
    cb = (c * selector).sum(2)[..., 0, :]
    e = mv(ab, x.constant(normals)) + cb
    child = model.complete(parents, e)
    # Each local map has one innovation Gaussian, not N ambient Gaussians.
    l, density_valid, _ = checked_chol(covariance[..., 0, :, :], 1)
    residual = e[:, :, None, :] - means[..., 0, :]
    from bayesfilter.highdim.analytical_jet_tf import solve
    standardized = solve(l, residual[..., None], triangular=True)[..., 0]
    diagonal = Jet(tf.linalg.diag_part(l.v), tf.linalg.diag_part(l.t))
    densities = -.5 * (r * math.log(2 * math.pi) +
                       2 * log(diagonal).sum(-1) +
                       (standardized * standardized).sum(-1))
    # Preserve the shared recurrence's B-row map-health diagnostics.
    reduced = (*maps[:6], tf.reduce_all(maps[6], 1),
               tf.reduce_min(maps[7], 1), tf.reduce_min(maps[8], 1))
    return child, densities, density_valid, reduced
