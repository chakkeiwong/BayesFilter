"""Optional covariance-guided mixture filter, Algorithms 1--2 (2026-10-08).

One numerical authority for every supported model/dtype. Analytical directional
recurrences differentiate the actual finite computation; no runtime autodiff.
This is not the admitted historical Contract-E residual-injection route.
Transitions must be additive Gaussian; observation densities may be non-Gaussian.
"""
from dataclasses import dataclass
import math
import math
import tensorflow as tf
from bayesfilter.highdim.ledh_marginal_weights_tf import gaussian_mixture_log_density_tangent
from bayesfilter.highdim.ledh_canonical_score_stages_tf import quadrature_update_with_parameter_tangent
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

ALGORITHM_ID = "covariance_guided_defensive_mixture_finite_v1"
TRACE_FIELDS = ("log_increment", "score_increment", "ess", "max_weight", "max_prior_proposal_ratio",
                "reset_mean_error", "reset_covariance_relative_error", "reset_margin",
                "guide_margin", "map_margin", "correction_displacement", "moment_loss_before",
                "moment_loss_after", "valid", "sinkhorn_row_error", "reset_displacement",
                "sinkhorn_column_error", "correction_coordinate_displacement")

@dataclass(frozen=True)
class ProposalControls:
    flow_steps: int = 16
    reset_steps: int = 40
    reset_epsilon: float = 1.0
    correction_steps: int = 0
    correction_basis: int = 4
    correction_rate: float = 0.1
    correction_radius: float = 0.25
    def __post_init__(self):
        if min(self.flow_steps, self.reset_steps, self.correction_basis) < 1:
            raise ValueError("iteration counts/basis must be positive")
        if self.correction_steps < 0 or not all(math.isfinite(v) and v>0 for v in (self.reset_epsilon,self.correction_rate,self.correction_radius)):
            raise ValueError("invalid positive numerical controls")

def sym(a):
    return .5*(a+tf.linalg.matrix_transpose(a))

def mm(a,b):
    return tf.linalg.matmul(a,b)

def mv(a,b):
    return tf.linalg.matvec(a,b)

def chol_tangent(l,dp):
    a=tf.linalg.triangular_solve(l,dp)
    a=tf.linalg.matrix_transpose(tf.linalg.triangular_solve(l,tf.linalg.matrix_transpose(a)))
    return mm(l,tf.linalg.band_part(a,-1,0)-.5*tf.linalg.diag(tf.linalg.diag_part(a)))

def checked_chol(p,dp):
    """Reject scale-relative rank loss; identity substitute only for flagged runs."""
    p=sym(p); eig=tf.linalg.eigvalsh(p)
    scale=tf.reduce_max(tf.abs(eig),-1)
    margin=tf.math.divide_no_nan(eig[...,0],scale)
    eps=tf.constant(2.220446049250313e-16 if p.dtype==tf.float64 else 1.1920928955078125e-7,p.dtype)
    valid=tf.reduce_all(tf.math.is_finite(p),[-2,-1]) & (margin>eps*tf.cast(tf.shape(p)[-1],p.dtype))
    eye=tf.eye(tf.shape(p)[-1],batch_shape=tf.shape(p)[:-2],dtype=p.dtype)
    l=tf.linalg.cholesky(tf.where(valid[...,None,None],p,eye))
    dl=chol_tangent(l,tf.where(valid[...,None,None],sym(dp),tf.zeros_like(dp)))
    return l,dl,tf.reduce_all(valid),tf.reduce_min(margin)

def solve_pair(a,da,b,db):
    batch=tf.broadcast_dynamic_shape(tf.shape(a)[:-2],tf.shape(b)[:-2])
    a=tf.broadcast_to(a,tf.concat([batch,tf.shape(a)[-2:]],0))
    da=tf.broadcast_to(da,tf.shape(a))
    b=tf.broadcast_to(b,tf.concat([batch,tf.shape(b)[-2:]],0))
    db=tf.broadcast_to(db,tf.shape(b))
    x=tf.linalg.solve(a,b)
    return x,tf.linalg.solve(a,db-mm(da,x))

def recolour(p,dp,s,ds):
    lp,dlp,vp,mp=checked_chol(p,dp); ls,dls,vs,ms=checked_chol(s,ds)
    it=tf.linalg.matrix_transpose
    at,dat=solve_pair(it(ls),it(dls),it(lp),it(dlp))
    return it(at),it(dat),vp & vs,tf.minimum(mp,ms)

def moments(x,dx,w,dw):
    mu=tf.einsum('n,nd->d',w,x)
    dmu=tf.einsum('n,nd->d',dw,x)+tf.einsum('n,nd->d',w,dx)
    e=x-mu; de=dx-dmu
    p=tf.einsum('n,ni,nj->ij',w,e,e)
    dp=tf.einsum('n,ni,nj->ij',dw,e,e)+tf.einsum('n,ni,nj->ij',w,de,e)+tf.einsum('n,ni,nj->ij',w,e,de)
    return mu,dmu,sym(p),sym(dp)

def build_maps(model,theta,ancestors,d_ancestors,observation,controls):
    """Chapter bridge-moments, local-ode, affine-composition and global-map."""
    n,d=ancestors.shape; dtype=ancestors.dtype
    m=model.transition_mean_fn(theta,ancestors)
    dm=model.transition_mean_tangent_fn(theta,ancestors,d_ancestors)
    q=tf.cast(model.process_covariance,dtype)
    dq=(tf.zeros_like(q) if model.process_covariance_tangent_fn is None else model.process_covariance_tangent_fn(theta))
    r=tf.cast(model.observation_covariance,dtype)
    dr=(tf.zeros_like(r) if model.observation_covariance_tangent_fn is None else model.observation_covariance_tangent_fn(theta))
    w=tf.fill([n],tf.constant(1./n,dtype)); z=tf.zeros([n],dtype)
    mean,dmean,p,dp=moments(m,dm,w,z); p+=q; dp+=dq
    upd=quadrature_update_with_parameter_tangent(mean[None],p[None],dmean[None],dp[None],
        model.observation_fn,model.observation_tangent_fn,r,observation,d_observation_covariance=dr,jitter=0.)
    up,pc,dup,dpc,vu=upd
    ag,dag,vg,guide_margin=recolour(pc[0],dpc[0],p,dp)
    cg=up[0]-mv(ag,mean); dcg=dup[0]-mv(dag,mean)-mv(ag,dmean)
    eye=tf.eye(d,dtype=dtype); eye_n=tf.broadcast_to(eye,[n,d,d])
    # Deterministic conditional-mean pilot, fixed before transition innovations.
    h=model.observation_jacobian_fn(m)
    dh=tf.zeros_like(h) if model.observation_jacobian_tangent_fn is None else model.observation_jacobian_tangent_fn(m,dm)
    hm=model.observation_fn(m); dhm=model.observation_tangent_fn(m,dm)
    offset=hm-mv(h,m); doffset=dhm-mv(dh,m)-mv(h,dm)
    rh,drh=solve_pair(r[None],dr[None],h,dh)
    info=mm(tf.linalg.matrix_transpose(h),rh)
    dinfo=mm(tf.linalg.matrix_transpose(dh),rh)+mm(tf.linalg.matrix_transpose(h),drh)
    residual=observation-offset; dresidual=-doffset
    rz,drz=solve_pair(r[None],dr[None],residual[...,None],dresidual[...,None])
    a=mv(tf.linalg.matrix_transpose(h),rz[...,0])
    da=mv(tf.linalg.matrix_transpose(dh),rz[...,0])+mv(tf.linalg.matrix_transpose(h),drz[...,0])
    lq,dlq,vq,qmargin=checked_chol(q,dq)
    qi=tf.linalg.cholesky_solve(lq,eye); dqi=-mm(mm(qi,dq),qi)
    qm=mv(qi,m); dqm=mv(dqi,m)+mv(qi,dm)
    af=eye_n; daf=tf.zeros_like(af); cf=tf.zeros_like(m); dcf=tf.zeros_like(m)
    step=tf.constant(1./controls.flow_steps,dtype)
    def bridge(k,af,daf,cf,dcf):
        lam=tf.cast(k,dtype)*step
        precision=qi[None]+lam*info; dprecision=dqi[None]+lam*dinfo
        pp,dpp=solve_pair(precision,dprecision,eye_n,tf.zeros_like(eye_n))
        rhs=qm+lam*a; drhs=dqm+lam*da
        pm=mv(pp,rhs); dpm=mv(dpp,rhs)+mv(pp,drhs)
        force=a-mv(info,pm); dforce=da-mv(dinfo,pm)-mv(info,dpm)
        mdot=mv(pp,force); dmdot=mv(dpp,force)+mv(pp,dforce)
        f=-.5*mm(pp,info); df=-.5*(mm(dpp,info)+mm(pp,dinfo))
        b=mdot-mv(f,pm); db=dmdot-mv(df,pm)-mv(f,dpm)
        mat=eye_n+step*f; dmat=step*df
        return k+1,mm(mat,af),mm(dmat,af)+mm(mat,daf),mv(mat,cf)+step*b,mv(dmat,cf)+mv(mat,dcf)+step*db
    _,af,daf,cf,dcf=tf.while_loop(lambda k,*_:k<controls.flow_steps,bridge,(0,af,daf,cf,dcf),parallel_iterations=1)
    aa=tf.stack([eye_n,tf.broadcast_to(ag,[n,d,d]),af]); daa=tf.stack([tf.zeros_like(eye_n),tf.broadcast_to(dag,[n,d,d]),daf])
    cc=tf.stack([tf.zeros_like(m),tf.broadcast_to(cg,[n,d]),cf]); dcc=tf.stack([tf.zeros_like(m),tf.broadcast_to(dcg,[n,d]),dcf])
    means=mv(aa,m[None])+cc; dmeans=mv(daa,m[None])+mv(aa,dm[None])+dcc
    cov=mm(mm(aa,q),tf.linalg.matrix_transpose(aa))
    dcov=mm(mm(daa,q),tf.linalg.matrix_transpose(aa))+mm(mm(aa,dq),tf.linalg.matrix_transpose(aa))+mm(mm(aa,q),tf.linalg.matrix_transpose(daa))
    _,_,vm,map_margin=checked_chol(cov,dcov)
    return means,dmeans,cov,dcov,aa,daa,cc,dcc,m,dm,lq,dlq,vu & vg & vq & vm,tf.minimum(guide_margin,qmargin),map_margin

def component_densities(x,dx,means,dmeans,cov,dcov):
    n=x.shape[0]; dtype=x.dtype
    lw=tf.fill([n],-tf.math.log(tf.cast(n,dtype))); dlw=tf.zeros_like(lw)
    vals=[]; derivs=[]; valid=tf.constant(True)
    for b in range(3):
        v,dv,vok,dok=gaussian_mixture_log_density_tangent(x,dx,means[b],dmeans[b],cov[b],dcov[b],lw,dlw)
        vals.append(v); derivs.append(dv); valid &= vok & dok
    return tf.stack(vals,-1),tf.stack(derivs,-1),valid

def observation_density(model,theta,x,dx,y):
    """Actual observation density; Gaussian fallback when no explicit law exists."""
    if model.observation_log_density_fn is not None:
        return (model.observation_log_density_fn(theta,x,y),
                model.observation_log_density_tangent_fn(theta,x,y,dx))
    from bayesfilter.highdim.ledh_canonical_score_stages_tf import _gaussian_log_density_and_tangent
    mean=model.observation_fn(x); dmean=model.observation_tangent_fn(x,dx)
    r=tf.cast(model.observation_covariance,x.dtype)
    dr=None if model.observation_covariance_tangent_fn is None else model.observation_covariance_tangent_fn(theta)
    return _gaussian_log_density_and_tangent(tf.broadcast_to(y,tf.shape(mean)),tf.zeros_like(mean),mean,dmean,tf.linalg.cholesky(r),d_covariance=dr)

def draw_from_maps(maps,beta,uniforms,normals):
    means,dm,cov,dcov,a,da,c,dc,m,dmu,l,dl,*_=maps
    n=normals.shape[0]
    b=tf.reduce_sum(tf.cast(uniforms[:,0,None]>=tf.cumsum(beta)[None,:-1],tf.int32),-1)
    j=tf.minimum(tf.cast(uniforms[:,1]*tf.cast(n,uniforms.dtype),tf.int32),n-1)
    ij=tf.stack([b,j],-1)
    u=tf.gather(m,j)+mv(l,normals); du=tf.gather(dmu,j)+mv(dl,normals)
    ab=tf.gather_nd(a,ij); dab=tf.gather_nd(da,ij)
    return mv(ab,u)+tf.gather_nd(c,ij),mv(dab,u)+mv(ab,du)+tf.gather_nd(dc,ij)

def reset_cloud(x,dx,logw,dlogw,controls):
    """Algorithm 2: finite log-Sinkhorn, actual column normalization, Cholesky."""
    n,d=x.shape; dtype=x.dtype
    if select_transport_chunks(n).row_chunk_size != n:
        raise ValueError("dense candidate reset limited to N<=3000; streaming extension not implemented")
    w=tf.exp(logw); dw=w*dlogw
    mu,dmu,p,dp=moments(x,dx,w,dw)
    delta=x[:,None]-x[None]; ddelta=dx[:,None]-dx[None]
    lk=-tf.reduce_sum(delta*delta,-1)/tf.cast(controls.reset_epsilon,dtype)
    dlk=-2*tf.reduce_sum(delta*ddelta,-1)/tf.cast(controls.reset_epsilon,dtype)
    logn=tf.math.log(tf.cast(n,dtype)); gamma=tf.zeros([n],dtype); dg=tf.zeros_like(gamma)
    def body(k,gamma,dg,alpha,da):
        z=lk+gamma[None]; dz=dlk+dg[None]
        alpha=logw-tf.reduce_logsumexp(z,1)
        da=dlogw-tf.reduce_sum(tf.nn.softmax(z,axis=1)*dz,1)
        z=lk+alpha[:,None]; dz=dlk+da[:,None]
        gamma=-logn-tf.reduce_logsumexp(z,0)
        dg=-tf.reduce_sum(tf.nn.softmax(z,axis=0)*dz,0)
        return k+1,gamma,dg,alpha,da
    _,gamma,dg,alpha,da=tf.while_loop(lambda k,*_:k<controls.reset_steps,body,(0,gamma,dg,gamma,dg),parallel_iterations=1)
    logpi=lk+alpha[:,None]+gamma[None]; dlogpi=dlk+da[:,None]+dg[None]
    trans=tf.nn.softmax(logpi,axis=0)
    dtrans=trans*(dlogpi-tf.reduce_sum(trans*dlogpi,0,keepdims=True))
    y=tf.einsum('ij,id->jd',trans,x)
    dy=tf.einsum('ij,id->jd',dtrans,x)+tf.einsum('ij,id->jd',trans,dx)
    unif=tf.fill([n],tf.constant(1./n,dtype)); zero=tf.zeros([n],dtype)
    ym,dym,sy,dsy=moments(y,dy,unif,zero)
    ar,dar,valid,margin=recolour(p,dp,sy,dsy)
    e=y-ym; de=dy-dym
    z=mu+mv(ar,e); dz=dmu+mv(dar,e)+mv(ar,de)
    zm,_,zp,_=moments(z,dz,unif,zero)
    rowerr=tf.reduce_max(tf.abs(tf.reduce_sum(tf.exp(logpi),1)-w))
    diagnostics=tf.stack([tf.reduce_max(tf.abs(zm-mu)),tf.linalg.norm(zp-p)/tf.maximum(tf.linalg.norm(p),tf.constant(1.e-30,dtype)),margin,rowerr,tf.reduce_max(tf.linalg.norm(z-y,axis=1)),tf.reduce_max(tf.abs(tf.reduce_sum(trans,axis=0)-1))])
    return z,dz,mu,dmu,p,dp,valid,diagnostics

def make_filter(spec,n,horizon,controls,dtype=tf.float64,jit_compile=True):
    """Compile a stable-signature value/analytical directional-score program."""
    dtype=tf.as_dtype(dtype); d=spec.dimension; p=spec.parameter_count
    o=getattr(spec,'observation_dimension',d)
    if n<=d or horizon<1: raise ValueError("require N>d and T>=1")
    sig=[tf.TensorSpec([p],dtype),tf.TensorSpec([p],dtype),tf.TensorSpec([horizon,o],dtype),tf.TensorSpec([n,d],dtype),tf.TensorSpec([horizon,n,d],dtype),tf.TensorSpec([horizon,n,2],dtype),tf.TensorSpec([3],dtype)]
    @tf.function(input_signature=sig,jit_compile=jit_compile,autograph=False)
    def program(theta,direction,observations,initial,innovations,uniforms,beta):
        model,_=spec.model(theta,direction)
        x,_,dx,_=spec.initial_cloud(theta,initial,direction)
        trace=tf.TensorArray(dtype,size=horizon,element_shape=[len(TRACE_FIELDS)]).unstack(tf.fill([horizon,len(TRACE_FIELDS)],tf.constant(float("nan"),dtype)))
        history=tf.TensorArray(dtype,size=horizon,element_shape=[n,d]).unstack(tf.fill([horizon,n,d],tf.constant(float("nan"),dtype)))
        beta_valid=tf.reduce_all(tf.math.is_finite(beta)) & tf.reduce_all(beta>=0) & (beta[0]>0) & (tf.abs(tf.reduce_sum(beta)-1)<tf.cast(1.e-6,dtype))
        def step(t,x,dx,ell,score,valid,trace,history):
            history=history.write(t,x)
            maps=build_maps(model,theta,x,dx,observations[t],controls)
            child,dchild=draw_from_maps(maps,beta,uniforms[t],innovations[t])
            logdens,ddens,vd=component_densities(child,dchild,*maps[:4])
            logmix=logdens+tf.math.log(beta)[None]
            lq=tf.reduce_logsumexp(logmix,-1); dlq=tf.reduce_sum(tf.nn.softmax(logmix,-1)*ddens,-1)
            lg,dlg=observation_density(model,theta,child,dchild,observations[t])
            lr=lg+logdens[:,0]-lq; dlr=dlg+ddens[:,0]-dlq
            normalizer=tf.reduce_logsumexp(lr); w=tf.nn.softmax(lr); increment=normalizer-tf.math.log(tf.cast(n,dtype))
            ds=tf.reduce_sum(w*dlr); logw=lr-normalizer; dlogw=dlr-ds
            z,dz,mu,dmu,pw,dpw,vr,rd=reset_cloud(child,dchild,logw,dlogw,controls)
            reset_particles=z
            cd=tf.zeros([3],dtype)
            if controls.correction_steps:
                from bayesfilter.highdim.covariance_proposal_moments_tf import repair_moments
                z,dz,cd=repair_moments(child,dchild,w,w*dlogw,z,dz,mu,dmu,pw,dpw,controls)
            unif=tf.fill([n],tf.constant(1./n,dtype)); zero=tf.zeros([n],dtype)
            zm,_,zp,_=moments(z,dz,unif,zero)
            eps=tf.constant(2.220446049250313e-16 if dtype==tf.float64 else 1.1920928955078125e-7,dtype)
            mean_error=tf.reduce_max(tf.abs(zm-mu))
            cov_error=tf.linalg.norm(zp-pw)/tf.maximum(tf.linalg.norm(pw),tf.constant(1e-30,dtype))
            moment_ok=(mean_error<=100*eps*d*tf.maximum(tf.constant(1.,dtype),tf.reduce_max(tf.abs(mu)))) & (cov_error<=100*eps*d)
            coordinate_move=tf.reduce_max(tf.abs(z-reset_particles)/tf.sqrt(tf.linalg.diag_part(pw))[None])
            cap_limit=tf.constant(controls.correction_radius,dtype)*(1+100*eps*n)
            cap_ok=(cd[0]<=cap_limit) & (coordinate_move<=cap_limit) & tf.reduce_all(tf.math.is_finite(cd))
            good=moment_ok & cap_ok & maps[12] & vd & vr & tf.reduce_all(tf.math.is_finite(z)) & tf.reduce_all(tf.math.is_finite(dz)) & tf.math.is_finite(increment) & tf.math.is_finite(ds)
            row=tf.stack([increment,ds,1/tf.reduce_sum(w*w),tf.reduce_max(w),tf.reduce_max(tf.exp(logdens[:,0]-lq)),mean_error,cov_error,rd[2],maps[13],maps[14],cd[0],cd[1],cd[2],tf.cast(good,dtype),rd[3],rd[4],rd[5],coordinate_move])
            return t+1,z,dz,ell+increment,score+ds,valid & good,trace.write(t,row),history
        completed,x,dx,ell,score,valid,trace,history=tf.while_loop(lambda t,x,dx,ell,score,valid,trace,history:(t<horizon) & valid,step,(0,x,dx,tf.constant(0.,dtype),tf.constant(0.,dtype),beta_valid,trace,history),parallel_iterations=1)
        return dict(value=ell,score=score,valid=valid,steps_completed=completed,particles=x,particle_tangent=dx,trace=trace.stack(),ancestor_history=history.stack())
    return program
