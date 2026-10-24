"""Independent NumPy diagnostics only; never imported by a runtime target.

Short Model B integrates m0 and standard-normal innovations with Gauss-Hermite
quadrature and integrates initial k exactly. Long Model B uses independently
seeded bootstrap filters. Their score is the Fisher path-score estimate, not
the derivative of the finite resampling program. Replication error is essential.
"""
import math
import numpy as np


def logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    v = m + np.log(np.sum(np.exp(x-m), axis=axis, keepdims=True))
    return np.squeeze(v, axis=axis) if axis is not None else float(v.ravel()[0])


def kalman_a(q, observations):
    sigma, tau = np.exp(q)
    f = np.array([[.35, -.10], [1., 0.]])
    mu, p, ell = np.zeros(2), np.eye(2), 0.
    for y in np.asarray(observations).reshape(-1):
        mu = f @ mu
        p = f @ p @ f.T + np.diag([sigma*sigma, 0.])
        s, r = p[0, 0]+tau*tau, y-mu[0]
        ell -= .5*(math.log(2*math.pi*s)+r*r/s)
        gain = p[:, 0]/s
        mu = mu+gain*r
        p = p-np.outer(gain, p[0])
    return ell


def model_b_quadrature(q, observations, order=20, block=32768):
    y = np.asarray(observations, dtype=float).reshape(-1)
    t = len(y)
    if t > 3 or order < 2:
        raise ValueError('Path quadrature is restricted to T<=3, order>=2')
    rho, sigma, beta = np.tanh(q[0]), np.exp(q[1]), np.exp(q[2])
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, lw = nodes*np.sqrt(2.), np.log(weights)-.5*np.log(np.pi)
    v = .55**np.arange(1, t+1)
    covariance = .09*np.eye(t)+.20*np.outer(v, v)
    inv = np.linalg.inv(covariance)
    norm = t*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]
    totals = []
    for first in range(0, order**(t+1), block):
        ids = np.arange(first, min(first+block, order**(t+1)))
        index = np.stack([(ids//order**j)%order for j in range(t+1)], -1)
        m, k = .5*nodes[index[:, 0]], np.zeros(len(ids))
        means = []
        for j in range(t):
            m = rho*m+sigma*nodes[index[:, j+1]]
            k = .55*k+beta*np.tanh(m)
            means.append(m+k)
        residual = y-np.stack(means, -1)
        ll = -.5*(norm+np.einsum('bi,ij,bj->b', residual, inv, residual))
        totals.append(logsumexp(ll+lw[index].sum(-1)))
    return logsumexp(np.asarray(totals))


def quadrature_value_score(q, observations, order=20, step=1e-4):
    q = np.asarray(q, float)
    value = model_b_quadrature(q, observations, order)
    score = [(model_b_quadrature(q+step*d, observations, order)-
              model_b_quadrature(q-step*d, observations, order))/(2*step)
             for d in np.eye(3)]
    return dict(value=float(value), score=score, order=order, fd_step=step)


def bootstrap_value_score(q, observations, particles, seed):
    """Independent path-space Fisher score; no differentiating ancestors."""
    q = np.asarray(q, float)
    rho, sigma, beta = np.tanh(q[0]), np.exp(q[1]), np.exp(q[2])
    rng = np.random.default_rng(seed)
    m, k = rng.normal(size=(2, particles))*np.array([.5, math.sqrt(.20)])[:, None]
    dk = np.zeros(particles)
    score = np.zeros((particles, 3))
    ell, minimum_ess = 0., float(particles)
    for y in np.asarray(observations).reshape(-1):
        old = m.copy()
        e = rng.normal(size=particles)
        m = rho*old+sigma*e
        k = .55*k+beta*np.tanh(m)
        dk = .55*dk+beta*np.tanh(m)
        residual = y-m-k
        score[:, 0] += e*old*(1-rho*rho)/sigma
        score[:, 1] += e*e-1
        score[:, 2] += residual*dk/.09
        logw = -.5*(math.log(2*math.pi*.09)+residual*residual/.09)
        z = logsumexp(logw)
        w = np.exp(logw-z)
        ell += z-math.log(particles)
        minimum_ess = min(minimum_ess, 1/np.sum(w*w))
        final_score = np.sum(w[:, None]*score, axis=0)
        points = (rng.random()+np.arange(particles))/particles
        cdf = np.cumsum(w)
        cdf[-1] = 1.
        ancestors = np.searchsorted(cdf, points)
        m, k, dk, score = m[ancestors], k[ancestors], dk[ancestors], score[ancestors]
    return dict(value=ell, score=final_score.tolist(), seed=seed, particles=particles,
                minimum_ess=minimum_ess, score_kind='Fisher_path_score')


def replicated_reference(q, observations, particles, seeds):
    rows = [bootstrap_value_score(q, observations, particles, s) for s in seeds]
    v = np.asarray([[r['value'], *r['score']] for r in rows])
    return dict(rows=rows, mean=v.mean(0).tolist(),
                mcse=(v.std(0, ddof=1)/math.sqrt(len(rows))).tolist(),
                interpretation='replication uncertainty; finite-particle bias unresolved',
                role='independent_reference_diagnostic_only')


def gaussian_reference(model, q, observations, method='ekf'):
    """Independent diagnostic EKF or spherical-radial cubature Gaussian filter."""
    if model == 'model_a':
        return kalman_a(q, observations)
    rho, sigma, beta = np.tanh(q[0]), np.exp(q[1]), np.exp(q[2])
    mu, cov, value = np.zeros(2), np.diag([.25, .20]), 0.
    h = np.array([1., 1.])
    for y in np.asarray(observations).reshape(-1):
        if method == 'ekf':
            m = rho*mu[0]
            slope = beta/np.cosh(m)**2
            f = np.array([[rho, 0.], [slope*rho, .55]])
            g = np.array([sigma, slope*sigma])
            pred = np.array([m, .55*mu[1]+beta*np.tanh(m)])
            p = f@cov@f.T+np.outer(g, g)
        elif method == 'cubature':
            augmented = np.zeros((3, 3)); augmented[:2, :2] = cov; augmented[2, 2] = 1.
            factor = np.linalg.cholesky(augmented)
            points = np.r_[mu, 0.]+np.sqrt(3.)*np.concatenate([factor.T, -factor.T])
            m = rho*points[:, 0]+sigma*points[:, 2]
            cloud = np.stack([m, .55*points[:, 1]+beta*np.tanh(m)], -1)
            pred = cloud.mean(0); centered = cloud-pred
            p = centered.T@centered/6.
        else:
            raise ValueError(method)
        innovation = float(y-h@pred); variance = float(h@p@h+.09)
        value -= .5*(np.log(2*np.pi*variance)+innovation**2/variance)
        gain = p@h/variance
        mu = pred+gain*innovation; cov = p-np.outer(gain, h@p)
        cov = .5*(cov+cov.T)
    return float(value)


def gaussian_value_score(model, q, observations, method):
    q = np.asarray(q, float); step = 1e-4
    score = [(gaussian_reference(model,q+step*d,observations,method)-
              gaussian_reference(model,q-step*d,observations,method))/(2*step)
             for d in np.eye(len(q))]
    return dict(value=gaussian_reference(model,q,observations,method),score=score)


def kalman_posterior_grid(observations, mean, scale, size=257, radius=7.):
    """Independent 2D chart posterior quadrature; refinement checked by caller."""
    mean, scale = np.asarray(mean), np.asarray(scale)
    axes = [np.linspace(m-radius*s,m+radius*s,size) for m,s in zip(mean,scale)]
    q = np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
    sigma2,tau2 = np.exp(2*q).T
    mu = np.zeros((len(q),2)); p = np.broadcast_to(np.eye(2),(len(q),2,2)).copy()
    f = np.array([[.35,-.1],[1.,0.]])
    ll = np.zeros(len(q))
    for y in np.asarray(observations).reshape(-1):
        mu = mu@f.T; p = f@p@f.T; p[:,0,0] += sigma2
        variance = p[:,0,0]+tau2; residual = y-mu[:,0]
        ll -= .5*(np.log(2*np.pi*variance)+residual**2/variance)
        gain = p[:,:,0]/variance[:,None]
        mu += gain*residual[:,None]
        p -= gain[:,:,None]*p[:,None,0,:]
    lp = ll-.5*np.sum(((q-mean)/scale)**2,axis=-1)
    weights = np.exp(lp-logsumexp(lp)).reshape(size,size)
    marginals = [weights.sum(1),weights.sum(0)]
    average = np.array([np.sum(a*w) for a,w in zip(axes,marginals)])
    sd = [np.sqrt(np.sum(w*(a-m)**2)) for a,w,m in zip(axes,marginals,average)]
    quantiles = [[float(np.interp(v,np.cumsum(w),a)) for v in (.025,.5,.975)]
                 for a,w in zip(axes,marginals)]
    return dict(mean=average.tolist(),sd=sd,quantiles=quantiles,size=size,radius=radius,
                boundary_mass=float(weights[0,:].sum()+weights[-1,:].sum()+weights[:,0].sum()+weights[:,-1].sum()))
