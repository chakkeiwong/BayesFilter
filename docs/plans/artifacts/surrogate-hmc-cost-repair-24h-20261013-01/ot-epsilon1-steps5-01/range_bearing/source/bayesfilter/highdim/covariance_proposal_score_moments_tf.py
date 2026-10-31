"""All-parameter analytical version of the unchanged finite Cayley correction."""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import mm, solve, sqrt, concat
from bayesfilter.highdim.covariance_proposal_score_tf import checked_chol
from bayesfilter.highdim.covariance_proposal_moments_tf import feature_indices, skew_basis


def features(s,gs,indices):
    n=s.v.shape[0]; r=gs.v.shape[0]
    s=concat([s,s.constant(tf.ones([n,1],s.v.dtype))],-1)
    gs=concat([gs,gs.constant(tf.zeros([r,n,1],s.v.dtype))],-1)
    ids=tf.constant(indices,tf.int32)
    y=s.constant(tf.ones([n,len(indices)],s.v.dtype))
    gy=s.constant(tf.zeros([r,n,len(indices)],s.v.dtype))
    for k in range(4):
        x=s.gather(ids[:,k],-1); gx=gs.gather(ids[:,k],-1)
        y,gy=y*x,gy*x[None]+y[None]*gx
    return y,gy


def cayley(v,basis,kappa):
    g=(v[:,None,None]*basis).sum(0)
    scale=sqrt(1+(g*g).sum())
    gs=(g[None]*basis).sum([-2,-1])/scale
    k=kappa*g/scale
    gk=kappa*(basis/scale-g[None]*gs[:,None,None]/scale**2)
    eye=g.constant(tf.eye(g.v.shape[0],dtype=g.v.dtype))
    h=eye-k/2
    o=solve(h,eye+k/2)
    go=solve(h[None],mm(gk,(eye+o)[None])/2)
    return o,go


def repair_moments(x,w,z,mu,p,controls):
    n,d=x.v.shape; dtype=x.v.dtype; r=controls.correction_basis
    l,_,_=checked_chol(p)
    sx=solve(l,(x-mu).transpose()).transpose()
    sz=solve(l,(z-mu).transpose()).transpose()
    zeros=x.constant(tf.zeros([r,n,d],dtype)); ids=feature_indices(d)
    fx,_=features(sx,zeros,ids)
    target=(w[:,None]*fx).sum(0)
    basis=x.constant(skew_basis(n,r,dtype))
    kappa=controls.correction_radius/math.sqrt(n)
    def objective(v):
        o,go=cayley(v,basis,kappa)
        s=mm(o.transpose(),sz)
        gs=mm(go.transpose(),sz[None])
        f,gf=features(s,gs,ids)
        residual=f.mean(0)-target
        loss=.5*(residual*residual).sum()
        grad=(gf.mean(1)*residual[None]).sum(1)
        return loss,grad,o
    v=x.constant(tf.zeros([r],dtype))
    initial=objective(v)[0].v
    def step(k,value,tangent):
        from bayesfilter.highdim.analytical_jet_tf import Jet
        v=Jet(value,tangent)
        _,g,_=objective(v)
        v=v-controls.correction_rate*g
        return k+1,v.v,v.t
    _,vv,vt=tf.while_loop(lambda k,*_:k<controls.correction_steps,step,(0,v.v,v.t),parallel_iterations=1)
    from bayesfilter.highdim.analytical_jet_tf import Jet
    loss,_,o=objective(Jet(vv,vt))
    output=mu+mm(o.transpose(),z-mu)
    displacement=tf.transpose(tf.linalg.triangular_solve(l.v,tf.transpose(output.v-z.v)))
    return output,tf.stack([tf.reduce_max(tf.linalg.norm(displacement,axis=1)),initial,loss.v])
