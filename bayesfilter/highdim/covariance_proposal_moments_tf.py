"""Algorithm 3: bounded ensemble Cayley rotation with analytical mixed partials.

Fixed cosine-pair basis, all marginal and pairwise degree-3/4 monomials, unit
feature scales. Basis/rate/radius are declared controls, not promoted defaults.
The optimizer derivative includes the target moments, whitening and each step.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_tf import checked_chol, mm, solve_pair

def feature_indices(d):
    f=[]
    for k in range(d): f.extend([(k,k,k,d),(k,k,k,k)])
    for k in range(d):
        for j in range(k+1,d):
            f.extend([(k,k,j,d),(k,j,j,d),(k,k,k,j),(k,j,j,j),(k,k,j,j)])
    return f

def feature_jets(s,ds,gs,dgs,indices):
    """Polynomial value, theta tangent, v gradient and mixed total derivative."""
    n=tf.shape(s)[0]; r=tf.shape(gs)[0]
    s=tf.concat([s,tf.ones([n,1],s.dtype)],-1)
    ds=tf.concat([ds,tf.zeros([n,1],s.dtype)],-1)
    gs=tf.concat([gs,tf.zeros([r,n,1],s.dtype)],-1)
    dgs=tf.concat([dgs,tf.zeros([r,n,1],s.dtype)],-1)
    f=len(indices); ids=tf.constant(indices,tf.int32)
    y=tf.ones([n,f],s.dtype); dy=tf.zeros_like(y)
    gy=tf.zeros([r,n,f],s.dtype); dgy=tf.zeros_like(gy)
    for k in range(4):
        x=tf.gather(s,ids[:,k],axis=-1); dx=tf.gather(ds,ids[:,k],axis=-1)
        gx=tf.gather(gs,ids[:,k],axis=-1); dgx=tf.gather(dgs,ids[:,k],axis=-1)
        y,dy,gy,dgy=y*x,dy*x+y*dx,gy*x[None]+y[None]*gx,dgy*x[None]+gy*dx[None]+dy[None]*gx+y[None]*dgx
    return y,dy,gy,dgy

def skew_basis(n,r,dtype):
    if 2*r>=n: raise ValueError("Cayley basis requires 2*basis < N")
    ix=tf.cast(tf.range(n),dtype)+.5
    cols=tf.cos(tf.constant(math.pi/n,dtype)*ix[:,None]*tf.cast(tf.range(1,2*r+1),dtype)[None])
    cols-=tf.reduce_mean(cols,0,keepdims=True)
    cols/=tf.linalg.norm(cols,axis=0,keepdims=True)
    a=tf.transpose(cols[:,::2]); b=tf.transpose(cols[:,1::2])
    return (a[:,:,None]*b[:,None,:]-b[:,:,None]*a[:,None,:])/tf.sqrt(tf.constant(2.,dtype))

def cayley_jets(v,dv,basis,kappa):
    g=tf.einsum('r,rij->ij',v,basis); dg=tf.einsum('r,rij->ij',dv,basis)
    scale=tf.sqrt(1+tf.reduce_sum(g*g)); ds=tf.reduce_sum(g*dg)/scale
    gs=tf.einsum('ij,rij->r',g,basis)/scale
    dgs=tf.einsum('ij,rij->r',dg,basis)/scale-gs*ds/scale
    k=kappa*g/scale; dk=kappa*(dg/scale-g*ds/scale**2)
    gk=kappa*(basis/scale-g[None]*gs[:,None,None]/scale**2)
    dgk=kappa*(-basis*ds/scale**2-dg[None]*gs[:,None,None]/scale**2-g[None]*dgs[:,None,None]/scale**2+2*g[None]*gs[:,None,None]*ds/scale**3)
    eye=tf.eye(tf.shape(g)[0],dtype=g.dtype); h=eye-k/2
    o=tf.linalg.solve(h,eye+k/2)
    do=tf.linalg.solve(h,mm(dk,eye+o)/2)
    go=tf.linalg.solve(tf.broadcast_to(h[None],tf.shape(gk)),mm(gk,(eye+o)[None])/2)
    dgo=tf.linalg.solve(tf.broadcast_to(h[None],tf.shape(gk)),(mm(dk[None],go)+mm(dgk,(eye+o)[None])+mm(gk,do[None]))/2)
    return o,do,go,dgo

def repair_moments(x,dx,w,dw,z,dz,mu,dmu,p,dp,controls):
    n,d=x.shape; dtype=x.dtype; r=controls.correction_basis
    l,dl,valid,margin=checked_chol(p,dp)
    # A failed factor is flagged by the enclosing reset; no accepted ridge.
    sx,dsx=solve_pair(l,dl,tf.transpose(x-mu),tf.transpose(dx-dmu)); sx=tf.transpose(sx); dsx=tf.transpose(dsx)
    sz,dsz=solve_pair(l,dl,tf.transpose(z-mu),tf.transpose(dz-dmu)); sz=tf.transpose(sz); dsz=tf.transpose(dsz)
    zeros=tf.zeros([r,n,d],dtype); ids=feature_indices(d)
    fx,dfx,_,_=feature_jets(sx,dsx,zeros,zeros,ids)
    target=tf.einsum('n,nf->f',w,fx)
    dtarget=tf.einsum('n,nf->f',dw,fx)+tf.einsum('n,nf->f',w,dfx)
    basis=skew_basis(n,r,dtype)
    kappa=tf.constant(controls.correction_radius/math.sqrt(n),dtype)
    def objective(v,dv):
        o,do,go,dgo=cayley_jets(v,dv,basis,kappa)
        s=mm(tf.transpose(o),sz)
        ds=mm(tf.transpose(do),sz)+mm(tf.transpose(o),dsz)
        gs=mm(tf.linalg.matrix_transpose(go),sz[None])
        dgs=mm(tf.linalg.matrix_transpose(dgo),sz[None])+mm(tf.linalg.matrix_transpose(go),dsz[None])
        f,df,gf,dgf=feature_jets(s,ds,gs,dgs,ids)
        residual=tf.reduce_mean(f,0)-target; dr=tf.reduce_mean(df,0)-dtarget
        gr=tf.reduce_mean(gf,1); dgr=tf.reduce_mean(dgf,1)
        loss=.5*tf.reduce_sum(residual*residual)
        grad=tf.reduce_sum(gr*residual[None],1)
        dgrad=tf.reduce_sum(dgr*residual[None]+gr*dr[None],1)
        return loss,grad,dgrad,o,do
    v=tf.zeros([r],dtype); dv=tf.zeros_like(v)
    initial=objective(v,dv)[0]
    def step(k,v,dv):
        _,g,dg,_,_=objective(v,dv)
        return k+1,v-tf.cast(controls.correction_rate,dtype)*g,dv-tf.cast(controls.correction_rate,dtype)*dg
    _,v,dv=tf.while_loop(lambda k,*_:k<controls.correction_steps,step,(0,v,dv),parallel_iterations=1)
    loss,_,_,o,do=objective(v,dv)
    e=z-mu; de=dz-dmu
    output=mu+mm(tf.transpose(o),e)
    derivative=dmu+mm(tf.transpose(do),e)+mm(tf.transpose(o),de)
    displacement=tf.transpose(tf.linalg.triangular_solve(l,tf.transpose(output-z)))
    return output,derivative,tf.stack([tf.reduce_max(tf.linalg.norm(displacement,axis=1)),initial,loss])
