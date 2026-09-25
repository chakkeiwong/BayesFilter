"""Optional A10 guide/chart/physical-mixture extension for SV pair TT.

TensorFlow float64 kernels; stable signatures and XLA by default. Finite
quadrature diagnostics do not certify accuracy. Tensor positive quadrature is
a d<=4 diagnostic repair, not a scalable Zhao--Cui retained-grid route. Rule
construction, branch decisions and checked chart setup are explicit host-side
setup exceptions. No covariance clipping or silent ridge is used.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import tensorflow as tf

from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim import pair_block_tt_tf as pair
from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    _log_standard_normal, _normalized_hermite_values,
)

D = tf.float64
EPS = 2.220446049250313e-16


@dataclass(frozen=True)
class GuideChecks:
    """A10 numerical/resolution hypotheses, not certified error tolerances."""
    roundoff_factor: float = 64.
    mean_tolerance: float = .02
    covariance_tolerance: float = .05
    log_integral_tolerance: float = .02


@lru_cache(None)
def moment_kernel(dimension, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([None, dimension], D),
                                  tf.TensorSpec([None], D), tf.TensorSpec([], D)],
                 jit_compile=jit_compile, autograph=False)
    def moments(points, signed, margin_factor):
        mass = tf.reduce_sum(signed)
        absolute_mass = tf.reduce_sum(tf.abs(signed))
        safe_mass = tf.where(mass != 0., mass, tf.constant(1., D))
        weights = signed/safe_mass
        mean = tf.einsum('n,ni->i', weights, points)
        centered = points-mean
        covariance = tf.einsum('n,ni,nj->ij', weights, centered, centered)
        covariance = .5*(covariance+tf.transpose(covariance))
        eigenvalues = tf.linalg.eigvalsh(covariance)
        absolute_second = tf.reduce_sum(tf.abs(weights)*tf.reduce_sum(centered**2, axis=1))
        rounding = margin_factor*EPS*tf.cast(tf.shape(points)[0], D)
        covariance_margin = rounding*tf.maximum(tf.constant(1., D), absolute_second)
        finite = (tf.reduce_all(tf.math.is_finite(covariance))
                  & tf.reduce_all(tf.math.is_finite(mean)) & tf.math.is_finite(mass))
        return dict(mean=mean, covariance=covariance, eigenvalues=eigenvalues,
                    covariance_margin=covariance_margin, scaled_mass=mass,
                    absolute_mass=absolute_mass,
                    cancellation=absolute_mass/tf.abs(safe_mass),
                    dominant_absolute_fraction=tf.reduce_max(tf.abs(signed))/absolute_mass,
                    mass_valid=(mass > rounding*absolute_mass) & finite,
                    covariance_valid=tf.reduce_min(eigenvalues) > covariance_margin,
                    finite=finite)
    return moments


@lru_cache(None)
def likelihood_kernel(dimension, beta, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([None, dimension], D),
                                  tf.TensorSpec([dimension], D)],
                 jit_compile=jit_compile, autograph=False)
    def logg(x, y):
        return -.5*tf.reduce_sum(math.log(2*math.pi)+2*math.log(beta)+x
                                 +y*y*tf.exp(-2*math.log(beta)-x), axis=1)
    return logg


def checked_rule(predictive, points, signed, shift, *, checks=GuideChecks(), jit_compile=True):
    """Moments in predictive units; rejected rules retain every diagnostic."""
    d = int(predictive.mean.shape[0])
    stats = moment_kernel(d, jit_compile)(points, signed, tf.constant(checks.roundoff_factor, D))
    info = dict(stats, points=int(points.shape[0]))
    if not bool(stats['mass_valid']):
        info.update(status='invalid', reason='nonfinite_or_unresolved_signed_mass')
        return None, info
    info['log_evidence'] = tf.math.log(stats['scaled_mass'])+shift
    if not bool(stats['covariance_valid']):
        info.update(status='invalid', reason='covariance_unresolved_in_predictive_units')
        return None, info
    mean = predictive.forward(stats['mean'][None, :])[0]
    cov = predictive.factor @ stats['covariance'] @ tf.transpose(predictive.factor)
    try:
        chart = obs.Chart.from_moments(mean, cov)
    except ValueError as exc:
        info.update(status='invalid', reason=str(exc))
        return None, info
    standard = chart.inverse(predictive.forward(points))
    weights = signed/stats['scaled_mass']
    info.update(status='valid', physical_mean=mean, physical_covariance=cov,
                standardized_third=tf.einsum('n,ni->i', weights, standard**3),
                standardized_fourth=tf.einsum('n,ni->i', weights, standard**4))
    return chart, info


def resolution(records, checks=GuideChecks()):
    valid = [r for r in records if r['status'] == 'valid']
    if len(valid) < 2:
        return dict(resolved=False, reason='fewer_than_two_numerically_valid_rules')
    a, b = valid[-2:]
    mean = float(tf.linalg.norm(a['mean']-b['mean']))
    covariance = float(tf.linalg.norm(a['covariance']-b['covariance'])/
                       tf.maximum(tf.constant(1., D), tf.linalg.norm(a['covariance'])))
    logz = float(tf.abs(a['log_evidence']-b['log_evidence']))
    return dict(resolved=(mean <= checks.mean_tolerance and covariance <= checks.covariance_tolerance
                          and logz <= checks.log_integral_tolerance),
                mean_difference=mean, covariance_difference=covariance, log_integral_difference=logz)


@lru_cache(None)
def positive_gaussian_rule(dimension, order):
    """Golub--Welsch rule for N(0,I); small CPU setup, no NumPy."""
    if dimension > 4 or dimension < 1 or order < 2:
        raise ValueError('A10 positive quadrature is restricted to dimensions 1..4 and order>=2')
    with tf.device('/CPU:0'):
        off = tf.sqrt(tf.cast(tf.range(1, order), D))
        jacobi = tf.linalg.diag(off, k=1)+tf.linalg.diag(off, k=-1)
        nodes, vectors = tf.linalg.eigh(jacobi)
        weights = tf.square(vectors[0])
        indices = tf.stack([tf.reshape(v, [-1]) for v in
                            tf.meshgrid(*([tf.range(order)]*dimension), indexing='ij')], axis=1)
        return tf.gather(nodes, indices), tf.reduce_prod(tf.gather(weights, indices), axis=1)


@lru_cache(None)
def laplace_mode_kernel(dimension, beta, jit_compile=True):
    vector, matrix = tf.TensorSpec([dimension], D), tf.TensorSpec([dimension, dimension], D)
    @tf.function(input_signature=[vector, matrix, vector], jit_compile=jit_compile, autograph=False)
    def mode(mean, factor, y):
        precision = tf.linalg.cholesky_solve(factor, tf.eye(dimension, dtype=D))
        c = (y/beta)**2
        def terms(x):
            expterm = c*tf.exp(-x)
            delta = x-mean
            value = .5*tf.tensordot(delta, tf.linalg.matvec(precision, delta), 1)+.5*tf.reduce_sum(x+expterm)
            gradient = tf.linalg.matvec(precision, delta)+.5*(1-expterm)
            hessian = precision+.5*tf.linalg.diag(expterm)
            return value, gradient, hessian
        tolerance = 1e-10*(1+tf.reduce_max(tf.abs(mean)))
        def condition(i, x, valid):
            _, gradient, _ = terms(x)
            return (i < 64) & valid & (tf.reduce_max(tf.abs(gradient)) > tolerance)
        def body(i, x, valid):
            value, gradient, hessian = terms(x)
            factor_h = tf.linalg.cholesky(hessian)
            step = tf.linalg.cholesky_solve(factor_h, gradient[:, None])[:, 0]
            descent = tf.tensordot(gradient, step, 1)
            def backtrack(j, alpha):
                candidate = terms(x-alpha*step)[0]
                return (j < 40) & (~tf.math.is_finite(candidate) | (candidate > value-1e-4*alpha*descent))
            _, alpha = tf.while_loop(backtrack, lambda j,a:(j+1,a*.5), (0,tf.constant(1., D)))
            next_x = x-alpha*step
            next_value = terms(next_x)[0]
            okay = tf.math.is_finite(next_value) & tf.reduce_all(tf.math.is_finite(next_x)) & (next_value <= value)
            return i+1, next_x, valid & okay
        iterations, x, valid = tf.while_loop(condition, body, (0, mean, tf.constant(True)))
        value, gradient, hessian = terms(x)
        residual = tf.reduce_max(tf.abs(gradient))
        covariance = tf.linalg.cholesky_solve(tf.linalg.cholesky(hessian), tf.eye(dimension, dtype=D))
        return x, covariance, dict(iterations=iterations, residual=residual, tolerance=tolerance,
                                   resolved=valid & (residual <= tolerance), potential=value)
    return mode


def repaired_update(model, predictive, observation, clouds, *, checks=GuideChecks(), jit_compile=True):
    records, valid = [], []
    logg = likelihood_kernel(model.dimension, model.beta, jit_compile)
    for level, cloud in clouds:
        values = logg(predictive.forward(cloud.points), observation)
        shift = tf.reduce_max(values)
        chart, info = checked_rule(predictive, cloud.points, cloud.weights*tf.exp(values-shift), shift,
                                   checks=checks, jit_compile=jit_compile)
        info.update(level=level, negative_rule_weights=int(tf.reduce_sum(tf.cast(cloud.weights < 0, tf.int32))))
        records.append(info)
        if chart is not None:
            valid.append(chart)
    resolved = resolution(records, checks)
    if resolved['resolved']:
        return valid[-1], dict(method='signed_sgqf_resolved', rules=records, resolution=resolved,
                               fallback=False, log_evidence=[r for r in records if r['status']=='valid'][-1]['log_evidence'])
    info = dict(rules=records, signed_resolution=resolved, positive_rules=[], fallback=False)
    m, cov, mode_info = laplace_mode_kernel(model.dimension, model.beta, jit_compile)(
        predictive.mean, predictive.factor, observation)
    info['mode'] = mode_info
    positive_valid = []
    if bool(mode_info['resolved']):
        laplace = obs.Chart.from_moments(m, cov)
        for order in (5, 7, 9):
            nodes, weights = positive_gaussian_rule(model.dimension, order)
            x = laplace.forward(nodes)
            logratio = predictive.log_prob(x)+logg(x, observation)-laplace.log_prob(x)
            shift = tf.reduce_max(logratio)
            chart, rule = checked_rule(predictive, predictive.inverse(x), weights*tf.exp(logratio-shift),
                                       shift, checks=checks, jit_compile=jit_compile)
            rule['order'] = order
            info['positive_rules'].append(rule)
            if chart is not None:
                positive_valid.append(chart)
        positive_resolution = resolution(info['positive_rules'], checks)
        info['positive_resolution'] = positive_resolution
        if positive_resolution['resolved']:
            info.update(method='positive_laplace_quadrature_resolved',
                        log_evidence=[r for r in info['positive_rules'] if r['status']=='valid'][-1]['log_evidence'])
            return positive_valid[-1], info
    info.update(method='predictive_fallback_unresolved', fallback=True, log_evidence=None)
    return predictive, info


def build_guide_path(model, observations, *, levels=(2,3,4,5), checks=GuideChecks(), jit_compile=True):
    clouds = [(level, obs.tf_fixed_sgqf_cloud(model.dimension, level)) for level in levels]
    mean, covariance = tf.zeros([model.dimension], D), model.covariance0
    path, records = [], []
    for t, y in enumerate(tf.unstack(observations)):
        if t:
            mean = tf.linalg.matvec(model.transition, mean)
            covariance = model.transition @ covariance @ tf.transpose(model.transition)+model.sigma**2*tf.eye(model.dimension, dtype=D)
        predictive = obs.Chart.from_moments(mean, covariance)
        posterior, info = repaired_update(model, predictive, y, clouds, checks=checks, jit_compile=jit_compile)
        mean, covariance = posterior.mean, posterior.factor @ tf.transpose(posterior.factor)
        path.append((predictive, posterior)); records.append(info)
    return path, records


def stable_charts(model, guide):
    covariance, charts = model.covariance0, []
    for t, (_, posterior) in enumerate(guide):
        if t:
            covariance = model.transition @ covariance @ tf.transpose(model.transition)+model.sigma**2*tf.eye(model.dimension, dtype=D)
        charts.append(obs.Chart.from_moments(posterior.mean, covariance))
    return charts


@lru_cache(None)
def conditional_density_kernel(shapes, jit_compile=True):
    d = len(shapes)
    @tf.function(input_signature=[tuple(tf.TensorSpec(s, D) for s in shapes),
                                  tf.TensorSpec([None, d], D), tf.TensorSpec([None, d], D),
                                  tf.TensorSpec([], D)], jit_compile=jit_compile, autograph=False)
    def log_density(cores, u, v, tau):
        conditional = pair.pair_conditional_cores(cores, v)
        z = pair.conditional_normalizer_batched(conditional)
        state = tf.ones([tf.shape(u)[0], 1], D)
        for axis, core in enumerate(conditional):
            basis = _normalized_hermite_values(u[:, axis], int(core.shape[2])-1)
            state = tf.einsum('na,nakb,nk->nb', state, core, basis)
        return tf.math.log(state[:,0]**2+tau)+_log_standard_normal(u)-tf.math.log(z+tau)
    return log_density


def conditional_log_density(step, x, previous, jit_compile=True):
    if isinstance(step, joint.SGQFJointStep):
        return gaussian_density_kernel(int(x.shape[1]), jit_compile)(
            x, previous, step.current_chart.mean, step.previous_marginal.mean,
            step.gain, step.conditional_factor)
    if not isinstance(step, obs.PairTTStep):
        raise TypeError('Expected a frozen Gaussian or pair TT step')
    u, v = step.current_chart.inverse(x), step.conditioning_chart.inverse(previous)
    kernel = conditional_density_kernel(tuple(tuple(c.shape) for c in step.cores), jit_compile)
    return kernel(step.cores, u, v, step.tau)-step.current_chart.logdet


@lru_cache(None)
def gaussian_density_kernel(dimension, jit_compile=True):
    batch, vector, matrix = (tf.TensorSpec([None, dimension], D), tf.TensorSpec([dimension], D),
                             tf.TensorSpec([dimension, dimension], D))
    @tf.function(input_signature=[batch,batch,vector,vector,matrix,matrix],
                 jit_compile=jit_compile, autograph=False)
    def density(x, previous, current_mean, previous_mean, gain, factor):
        mean = current_mean+tf.linalg.matmul(previous-previous_mean,gain,transpose_b=True)
        residual = tf.transpose(tf.linalg.triangular_solve(factor,tf.transpose(x-mean)))
        return _log_standard_normal(residual)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
    return density


@lru_cache(None)
def physical_draw_kernel(dimension, initial, jit_compile=True):
    batch, matrix = tf.TensorSpec([None,dimension],D),tf.TensorSpec([dimension,dimension],D)
    @tf.function(input_signature=[batch,batch,batch,tf.TensorSpec([None],tf.bool),matrix,matrix,tf.TensorSpec([],D)],
                 jit_compile=jit_compile,autograph=False)
    def draw(previous,fitted,noise,choose,transition,prior_factor,sigma):
        if initial:
            physical=tf.linalg.matmul(noise,prior_factor,transpose_b=True)
        else:
            physical=tf.linalg.matmul(previous,transition,transpose_b=True)+sigma*noise
        x=tf.where(choose[:,None],physical,fitted)
        if initial:
            standardized=tf.transpose(tf.linalg.triangular_solve(prior_factor,tf.transpose(x)))
            logf=_log_standard_normal(standardized)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_factor)))
        else:
            standardized=(x-tf.linalg.matmul(previous,transition,transpose_b=True))/sigma
            logf=_log_standard_normal(standardized)-dimension*tf.math.log(sigma)
        return x,logf
    return draw


@lru_cache(None)
def mixture_density_kernel(jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([None],D),tf.TensorSpec([None],D),tf.TensorSpec([],D)],
                 jit_compile=jit_compile,autograph=False)
    def density(logtt,logf,epsilon):
        logq=tf.reduce_logsumexp(tf.stack([tf.math.log(1-epsilon)+logtt,tf.math.log(epsilon)+logf]),axis=0)
        return logq,tf.reduce_min(logq-(tf.math.log(epsilon)+logf))
    return density


@dataclass(frozen=True)
class PhysicalDefenseStep:
    proposal: object
    model: obs.SVModel
    epsilon: float

    def __post_init__(self):
        if not 0. < self.epsilon < 1.:
            raise ValueError('Physical defense epsilon must lie in (0,1)')


def sample_physical_defense(step, previous, seed, jit_compile=True):
    """Sample mixture and evaluate BOTH components at the selected physical x."""
    proposal, model, epsilon = step.proposal, step.model, step.epsilon
    fitted, _, diagnostics = joint.sample_joint_step(proposal, previous, seed, jit_compile)
    n, d = int(previous.shape[0]), model.dimension
    noise = tf.random.stateless_normal([n,d], [seed,6], dtype=D)
    choose = tf.random.stateless_uniform([n], [seed,5], dtype=D) < epsilon
    x,logf = physical_draw_kernel(d,proposal.time_index==0,jit_compile)(
        previous,fitted,noise,choose,model.transition,tf.linalg.cholesky(model.covariance0),tf.constant(model.sigma,D))
    logtt = conditional_log_density(proposal, x, previous, jit_compile)
    logq, margin = mixture_density_kernel(jit_compile)(logtt,logf,tf.constant(epsilon,D))
    obs.finite(x, 'physical-mixture draws'); obs.finite(logq, 'physical-mixture density')
    if float(margin) < -1e-12:
        raise ValueError('physical mixture density violates its lower bound')
    return x, logq, dict(diagnostics, physical_defense_epsilon=epsilon,
                         physical_defense_fraction=tf.reduce_mean(tf.cast(choose,D)),
                         log_density_lower_bound_margin=margin, finite=True)
