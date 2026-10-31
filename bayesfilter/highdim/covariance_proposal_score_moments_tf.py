"""All-parameter analytical version of the unchanged finite Cayley correction."""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import mm, solve, sqrt, concat
from bayesfilter.highdim.covariance_proposal_score_tf import checked_chol
from bayesfilter.highdim.covariance_proposal_moments_tf import feature_indices, skew_basis_factors


def features(s,gs,indices):
    n=s.v.shape[-2]; r=gs.v.shape[-3]; batch=s.v.shape.as_list()[:-2]
    s=concat([s,s.constant(tf.ones([*batch,n,1],s.v.dtype))],-1)
    gs=concat([gs,gs.constant(tf.zeros([*batch,r,n,1],s.v.dtype))],-1)
    ids=tf.constant(indices,tf.int32)
    y=s.constant(tf.ones([*batch,n,len(indices)],s.v.dtype))
    gy=s.constant(tf.zeros([*batch,r,n,len(indices)],s.v.dtype))
    for k in range(4):
        x=s.gather(ids[:,k],-1); gx=gs.gather(ids[:,k],-1)
        y,gy=y*x,gy*x[...,None,:,:]+y[...,None,:,:]*gx
    return y,gy


def cayley(v,basis,kappa):
    g=(v[...,None,None]*basis).sum(-3)
    scale=sqrt(1+(g*g).sum([-2,-1]))
    gs=(g[...,None,:,:]*basis).sum([-2,-1])/scale[...,None]
    k=kappa*g/scale[...,None,None]
    gk=kappa*(basis/scale[...,None,None,None]-g[...,None,:,:]*gs[...,None,None]/scale[...,None,None,None]**2)
    eye=g.constant(tf.eye(g.v.shape[-1],dtype=g.v.dtype))
    h=eye-k/2
    o=solve(h,eye+k/2)
    go=solve(h[...,None,:,:],mm(gk,(eye+o)[...,None,:,:])/2)
    return o,go


def compact_cayley_action(v,columns,cores,kappa,z):
    """Return O' z and its v gradient using the exact 2r-dimensional solve.

    U need not be orthogonal. The Gram matrix preserves the finite cosine
    basis, normalization, and all theta/v mixed derivatives. No N-square
    matrix or parameter/basis-by-N-square derivative is materialized.
    """
    gram=mm(columns.transpose(),columns)
    a=(v[...,None,None]*cores).sum(-3)
    sag=mm(gram,mm(a,gram))
    scale=sqrt(1+(a*sag).sum([-2,-1]))
    ds=(cores*sag[...,None,:,:]).sum([-2,-1])/scale[...,None]
    d=kappa*a/scale[...,None,None]
    dd=kappa*(cores/scale[...,None,None,None]-
        a[...,None,:,:]*ds[...,None,None]/scale[...,None,None,None]**2)
    eye=v.constant(tf.eye(a.v.shape[-1],dtype=a.v.dtype))
    h=eye+mm(d,gram)/2
    projected=mm(columns.transpose(),z)
    w=solve(h,mm(d,projected))
    dw=solve(h[...,None,:,:],mm(dd,(projected-mm(gram,w)/2)[...,None,:,:]))
    return z-mm(columns,w),-mm(columns,dw)


def repair_moments(x,w,z,mu,p,controls):
    n,d=x.v.shape[-2:]; batch=x.v.shape.as_list()[:-2]; dtype=x.v.dtype; r=controls.correction_basis
    l,_,_=checked_chol(p)
    sx=solve(l,(x-mu[...,None,:]).transpose()).transpose()
    sz=solve(l,(z-mu[...,None,:]).transpose()).transpose()
    zeros=x.constant(tf.zeros([*batch,r,n,d],dtype)); ids=feature_indices(d)
    fx,_=features(sx,zeros,ids)
    target=(w[...,None]*fx).sum(-2)
    columns,cores=[x.constant(t) for t in skew_basis_factors(n,r,dtype)]
    kappa=controls.correction_radius/math.sqrt(n)
    def objective(v):
        s,gs=compact_cayley_action(v,columns,cores,kappa,sz)
        f,gf=features(s,gs,ids)
        residual=f.mean(-2)-target
        loss=.5*(residual*residual).sum(-1)
        grad=(gf.mean(-2)*residual[...,None,:]).sum(-1)
        return loss,grad
    v=x.constant(tf.zeros([*batch,r],dtype))
    initial=objective(v)[0].v
    def step(k,value,tangent):
        from bayesfilter.highdim.analytical_jet_tf import Jet
        v=Jet(value,tangent)
        _,g=objective(v)
        v=v-controls.correction_rate*g
        return k+1,v.v,v.t
    _,vv,vt=tf.while_loop(lambda k,*_:k<controls.correction_steps,step,(0,v.v,v.t),parallel_iterations=1)
    from bayesfilter.highdim.analytical_jet_tf import Jet
    final_v=Jet(vv,vt)
    loss,_=objective(final_v)
    output=mu[...,None,:]+compact_cayley_action(final_v,columns,cores,kappa,z-mu[...,None,:])[0]
    displacement=tf.linalg.matrix_transpose(tf.linalg.triangular_solve(l.v,tf.linalg.matrix_transpose(output.v-z.v)))
    return output,tf.stack([tf.reduce_max(tf.linalg.norm(displacement,axis=-1),-1),initial,loss.v],-1)
