"""Independent NumPy/SciPy reference laws for post-run validation only.

These functions do not import the TensorFlow target implementation. References
never initialize/tune the ordinary procedure. Gaussian conjugate moments follow
completion of the square. The LGSSM likelihood is independently computed by
scalar Kalman recursions instead of the target's dense marginal Gaussian.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import special, stats


def conjugate(target, params, data):
    if target == "beta_binomial":
        a, b = params.get("alpha", 2.), params.get("beta", 3.)
        if data is not None:
            a += data[0]
            b += data[1]-data[0]
        return a, b
    prior_precision = 1/params.get("tau", 2.)**2
    if not data:
        return 0., 1/prior_precision
    y = np.asarray(data)
    if target == "normal_conjugate":
        variance = params.get("sigma", 1.)**2
        v = 1/(prior_precision+len(y)/variance)
        return v*y.sum()/variance, v
    # Likelihood as a quadratic in location, obtained independently from
    # innovations at locations -1,0,1, not the target covariance computation.
    a, b, c = [kalman_log_likelihood(x, params, data) for x in (-1.,0.,1.)]
    precision = -(a+c-2*b)
    linear = (c-a)/2
    v = 1/(prior_precision+precision)
    return v*linear, v


def kalman_log_likelihood(location, params, data):
    rho, variance, noise = params.get("rho", .6), params.get("state_variance", 1.), params.get("sigma", .5)**2
    mean, covariance, total = 0., variance, 0.
    for y in data:
        residual, innovation = y-location-mean, covariance+noise
        total += stats.norm.logpdf(residual, scale=math.sqrt(innovation))
        gain = covariance/innovation
        mean += gain*residual
        covariance *= 1-gain
        mean *= rho
        covariance = rho*rho*covariance+(1-rho*rho)*variance
    return float(total)


def model_coordinates(target, raw):
    raw = np.asarray(raw)
    if target == "funnel_noncentered":
        return np.concatenate([raw[..., :1], np.exp(raw[..., :1]/2)*raw[..., 1:]], -1)
    if target == "gamma":
        return np.exp(raw)
    if target in {"beta", "beta_binomial"}:
        return special.expit(raw)
    if target == "dirichlet":
        return special.softmax(np.concatenate([raw, np.zeros(raw.shape[:-1]+(1,))], -1), -1)
    return raw


def active_coordinates(target, values):
    x = np.asarray(values)
    if target == "funnel_noncentered":
        return np.concatenate([x[..., :1], np.exp(-x[..., :1]/2)*x[..., 1:]], -1)
    if target == "gamma":
        return np.log(x)
    if target in {"beta", "beta_binomial"}:
        return special.logit(x)
    if target == "dirichlet":
        return np.log(x[..., :2]/x[..., 2:])
    return x


def covariance(params):
    a = params.get("angle", .6)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return rotation @ np.diag([1., params.get("condition", 9.)]) @ rotation.T


def log_density(target, q, params=None, data=None):
    p = params or {}
    q = np.asarray(q, dtype=float)
    x = model_coordinates(target, q)
    if target == "gaussian":
        return stats.norm.logpdf(q, scale=p.get("scale", 1.)).sum(-1)
    if target == "rotated_gaussian":
        return stats.multivariate_normal.logpdf(q, cov=covariance(p))
    if target == "banana":
        return stats.norm.logpdf(q[...,0])+stats.norm.logpdf(q[...,1]-p.get("bend",.5)*(q[...,0]**2-1))
    if target == "funnel":
        return stats.norm.logpdf(q[...,0],scale=p.get("scale",3.))+stats.norm.logpdf(q[...,1:],scale=np.exp(q[...,:1]/2)).sum(-1)
    if target == "funnel_noncentered":
        # Independent centered density plus the change-of-variable Jacobian.
        return log_density("funnel", x, p) + q[..., 0]
    if target in {"student_t", "cauchy"}:
        return stats.t.logpdf(q, df=p.get("df",5.) if target=="student_t" else 1).sum(-1)
    if target == "mixture":
        a,w = p.get("separation",5.),p.get("weight",.3)
        return np.logaddexp(np.log(w)+stats.norm.logpdf(q[...,0]+a),np.log1p(-w)+stats.norm.logpdf(q[...,0]-a))+stats.norm.logpdf(q[...,1])
    if target == "gamma":
        return (stats.gamma.logpdf(x, a=p.get("alpha",2.),scale=1/p.get("rate",1.))+q).sum(-1)
    if target in {"beta", "beta_binomial"}:
        a,b=conjugate("beta_binomial",p,data if target=="beta_binomial" else None)
        return (stats.beta.logpdf(x,a,b)-np.logaddexp(0.,-q)-np.logaddexp(0.,q)).sum(-1)
    if target == "dirichlet":
        alpha=np.asarray(p.get("concentration",[2.,3.,4.]))
        return (alpha*np.log(x)).sum(-1)+special.gammaln(alpha.sum())-special.gammaln(alpha).sum()
    if target in {"normal_conjugate", "lgssm_location"}:
        out=stats.norm.logpdf(q[...,0],scale=p.get("tau",2.))
        if data is not None:
            if target=="normal_conjugate":
                out += stats.norm.logpdf(np.asarray(data)-q[...,:1],scale=p.get("sigma",1.)).sum(-1)
            else:
                out += np.array([kalman_log_likelihood(t,p,data) for t in q[...,0].flat]).reshape(q.shape[:-1])
        return out
    raise ValueError(target)


def draw(target, count, seed, params=None, data=None):
    p, rng = params or {}, np.random.default_rng(seed)
    if target == "gaussian": x=rng.normal(size=(count,2))*p.get("scale",1.)
    elif target == "rotated_gaussian": x=rng.multivariate_normal([0.,0.],covariance(p),count)
    elif target == "banana":
        x=rng.normal(size=(count,2)); x[:,1]+=p.get("bend",.5)*(x[:,0]**2-1)
    elif target in {"funnel", "funnel_noncentered"}:
        v=rng.normal(scale=p.get("scale",3.),size=count)
        x=np.column_stack([v, rng.normal(size=(count,2))*np.exp(v[:,None]/2)])
    elif target in {"student_t","cauchy"}: x=rng.standard_t(p.get("df",5.) if target=="student_t" else 1,size=(count,2))
    elif target == "mixture":
        x=rng.normal(size=(count,2)); x[:,0]+=np.where(rng.random(count)<p.get("weight",.3),-1,1)*p.get("separation",5.)
    elif target == "gamma": x=rng.gamma(p.get("alpha",2.),1/p.get("rate",1.),size=(count,1))
    elif target in {"beta","beta_binomial"}:
        a,b=conjugate("beta_binomial",p,data if target=="beta_binomial" else None)
        x=rng.beta(a,b,size=(count,1))
    elif target == "dirichlet": x=rng.dirichlet(p.get("concentration",[2.,3.,4.]),size=count)
    elif target in {"normal_conjugate","lgssm_location"}:
        m,v=conjugate(target,p,data); x=rng.normal(m,math.sqrt(v),size=(count,1))
    else: raise ValueError(target)
    return active_coordinates(target,x)


def simulate(target, seed, params=None):
    p,rng=params or {},np.random.default_rng(seed)
    if target=="beta_binomial":
        theta=rng.beta(p.get("alpha",2.),p.get("beta",3.)); n=p.get("n",12)
        return [float(theta)],[int(rng.binomial(n,theta)),int(n)]
    theta=rng.normal(scale=p.get("tau",2.)); n=p.get("n",6)
    if target=="normal_conjugate": y=rng.normal(theta,p.get("sigma",1.),n)
    elif target=="lgssm_location":
        rho=p.get("rho",.6); variance=p.get("state_variance",1.)
        latent=rng.normal(scale=math.sqrt(variance)); y=[]
        for _ in range(n):
            y.append(theta+latent+rng.normal(scale=p.get("sigma",.5)))
            latent=rho*latent+rng.normal(scale=math.sqrt((1-rho*rho)*variance))
    else: raise ValueError("proper generative model required")
    return [float(theta)],list(map(float,y))


def test_quantities(target, raw, params=None, data=None):
    x=model_coordinates(target,raw)
    quantities={f"parameter_{i}":x[...,i] for i in range(x.shape[-1])}
    quantities["bounded_radius"]=np.arctan((x*x).sum(-1))
    if x.shape[-1]>1:
        quantities["product"]=np.arctan(x[...,0]*x[...,1])
    if data is not None:
        p=params or {}
        if target=="beta_binomial":
            quantities["log_likelihood"]=stats.binom.logpmf(data[0],data[1],x[...,0])
        elif target=="normal_conjugate":
            quantities["log_likelihood"]=stats.norm.logpdf(np.asarray(data)-x[...,:1],scale=p.get("sigma",1.)).sum(-1)
        elif target=="lgssm_location":
            quantities["log_likelihood"]=np.array([kalman_log_likelihood(t,p,data) for t in x[...,0].flat]).reshape(x.shape[:-1])
    return quantities


def exact_functionals(target, params=None, data=None):
    """Available model-coordinate means/medians; absent entries stay unavailable."""
    p=params or {}
    if target in {"funnel", "funnel_noncentered"}:
        # A positive lognormal scale times an independent centered normal has
        # zero mean and median; all moments here are finite.
        return {(kind, i): 0. for i in range(3) for kind in ("mean", "quantile")}
    if target in {"normal_conjugate", "lgssm_location"}:
        m,_=conjugate(target,p,data)
        return {("mean",0):m,("quantile",0):m}
    if target in {"gaussian","rotated_gaussian","student_t","cauchy"}:
        return {(kind,i):0. for i in range(2) for kind in
                (("quantile",) if target=="cauchy" else ("mean","quantile"))}
    if target in {"beta","beta_binomial"}:
        a,b=conjugate("beta_binomial",p,data if target=="beta_binomial" else None)
        return {("mean",0):a/(a+b),("quantile",0):float(stats.beta.ppf(.5,a,b))}
    if target=="gamma":
        a,rate=p.get("alpha",2.),p.get("rate",1.)
        return {("mean",0):a/rate,("quantile",0):float(stats.gamma.ppf(.5,a,scale=1/rate))}
    if target=="dirichlet":
        a=p.get("concentration",[2.,3.,4.]); total=sum(a)
        return {(kind,i):float(v/total if kind=="mean" else stats.beta.ppf(.5,v,total-v))
                for i,v in enumerate(a) for kind in ("mean","quantile")}
    if target=="mixture":
        from scipy.optimize import brentq
        a,w=p.get("separation",5.),p.get("weight",.3)
        median=brentq(lambda x:w*stats.norm.cdf(x+a)+(1-w)*stats.norm.cdf(x-a)-.5,-abs(a)-10,abs(a)+10)
        return {("mean",0):(1-2*w)*a,("mean",1):0.,("quantile",0):median,("quantile",1):0.}
    return {}
