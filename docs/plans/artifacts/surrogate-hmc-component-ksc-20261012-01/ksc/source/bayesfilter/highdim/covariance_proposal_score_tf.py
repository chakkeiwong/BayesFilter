"""All-parameter analytical recurrence for the covariance-guided mixture.

One time sweep, shared primal maps/densities and cached finite Sinkhorn tiles.
All models use this implementation. Model callbacks supply analytical local
partials; the old directional filter is an independent parity reference.
"""
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, mm, mv, sym, solve, stack, concat, exp, logsumexp, cholesky_tangent
from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


def checked_chol(a):
    a = sym(a)
    eig = tf.linalg.eigvalsh(a.v)
    margin = tf.math.divide_no_nan(eig[...,0], tf.reduce_max(tf.abs(eig),-1))
    eps = 2.220446049250313e-16 if a.v.dtype==tf.float64 else 1.1920928955078125e-7
    valid = tf.reduce_all(tf.math.is_finite(a.v),[-2,-1]) & (margin>eps*a.v.shape[-1])
    eye = tf.eye(a.v.shape[-1], batch_shape=a.v.shape[:-2], dtype=a.v.dtype)
    value = tf.linalg.cholesky(tf.where(valid[...,None,None],a.v,eye))
    tangent = cholesky_tangent(value,tf.where(valid[...,None,None],a.t,tf.zeros_like(a.t)))
    return Jet(value,tangent),tf.reduce_all(valid),tf.reduce_min(margin)


def safe_chol(a, name):
    from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
    valid,value = safe_cholesky(a.v,name)
    return Jet(value,cholesky_tangent(value,a.t)),tf.reduce_all(valid)


def recolour(p,s):
    lp,vp,mp = checked_chol(p)
    ls,vs,ms = checked_chol(s)
    a = solve(ls.transpose(),lp.transpose()).transpose()
    return a,vp & vs,tf.minimum(mp,ms)


def moments(x,w):
    from bayesfilter.highdim.analytical_jet_tf import normalized_moments
    return normalized_moments(x,w)


class Models:
    """Lift existing analytical local partial callbacks, without filter replay."""
    def __init__(self,spec,theta,directions):
        self.theta=theta
        self.models=[spec.model(theta,directions[k])[0] for k in range(directions.shape[0])]
        # The callback protocol permits None only for H constant in both
        # state and theta along that direction. This is a model property,
        # not a heuristic equality check on the current particles.
        self.constant_observation_jacobian=all(
            model.observation_jacobian_tangent_fn is None for model in self.models)

    def apply(self, value_name, tangent_name, x, *args, theta=False):
        prefix=(self.theta,) if theta else ()
        value=getattr(self.models[0],value_name)(*prefix,x.v,*args)
        derivatives=[]
        for k,model in enumerate(self.models):
            callback=getattr(model,tangent_name)
            if callback is None:
                derivatives.append(tf.zeros_like(value))
            elif value_name=="observation_log_density_fn":
                derivatives.append(callback(*prefix,x.v,*args,x.t[k]))
            else:
                derivatives.append(callback(*prefix,x.v,x.t[k],*args))
        return Jet(value,tf.stack(derivatives))

    def covariance(self,value_name,tangent_name):
        value=tf.cast(getattr(self.models[0],value_name),self.theta.dtype)
        return Jet(value,tf.stack([getattr(m,tangent_name)(self.theta) if getattr(m,tangent_name) is not None else tf.zeros_like(value) for m in self.models]))


def quadrature(mean,p,r,models,observation):
    from bayesfilter.highdim.ledh_ukf_lifecycle_tf import _unscented_weights
    d=mean.v.shape[0]
    wm,wc,scale=_unscented_weights(d,mean.v.dtype,alpha=1.,beta=2.,kappa=0.)
    l,valid=safe_chol(scale*p,"sigma_points")
    points=concat([mean[None],mean[None]+l.transpose(),mean[None]-l.transpose()])
    observed=models.apply("observation_fn","observation_tangent_fn",points)
    om=(observed*wm[:,None]).sum(0)
    cx=points-mean; cy=observed-om
    innovation=sym(mm((cy*wc[:,None]).transpose(),cy)+r)
    cross=mm((cx*wc[:,None]).transpose(),cy)
    il,iv=safe_chol(innovation,"innovation")
    # Match the checked Cholesky authority, including its flagged substitute.
    gain=solve(il.transpose(),solve(il,cross.transpose(),triangular=True)).transpose()
    pm=mean+mv(gain,mean.constant(observation)-om)
    pc=sym(p-mm(mm(gain,innovation),gain.transpose()))
    return pm,pc,valid & iv


def build_maps(models,x,observation,controls):
    n,d=x.v.shape
    m=models.apply("transition_mean_fn","transition_mean_tangent_fn",x,theta=True)
    q=models.covariance("process_covariance","process_covariance_tangent_fn")
    r=models.covariance("observation_covariance","observation_covariance_tangent_fn")
    mean,p=moments(m,x.constant(tf.fill([n],tf.constant(1./n,x.v.dtype))))
    p=p+q
    up,pc,vu=quadrature(mean,p,r,models,observation)
    ag,vg,guide_margin=recolour(pc,p)
    cg=up-mv(ag,mean)
    eye=x.constant(tf.eye(d,dtype=x.v.dtype)); eye_n=eye.broadcast([n,d,d])
    shared_h=models.constant_observation_jacobian
    h=models.apply("observation_jacobian_fn","observation_jacobian_tangent_fn",m[:1] if shared_h else m)
    hm=models.apply("observation_fn","observation_tangent_fn",m)
    offset=hm-mv(h,m)
    rh=solve(r[None],h)
    info=mm(h.transpose(),rh)
    rz=solve(r[None],(x.constant(observation)-offset)[...,None])
    a=mv(h.transpose(),rz[...,0])
    lq,vq,qmargin=checked_chol(q)
    # Same primal Cholesky solve, with every covariance partial together.
    qi_value=tf.linalg.cholesky_solve(lq.v,eye.v)
    qi=Jet(qi_value,-tf.linalg.matmul(tf.linalg.matmul(qi_value,q.t),qi_value))
    qm=mv(qi,m)
    flow_eye=eye.broadcast([1 if shared_h else n,d,d])
    af=flow_eye; cf=x.constant(tf.zeros_like(m.v))
    step=tf.constant(1./controls.flow_steps,x.v.dtype)
    def bridge(k,av,at,cv,ct):
        af=Jet(av,at); cf=Jet(cv,ct)
        lam=tf.cast(k,x.v.dtype)*step
        pp=solve(qi[None]+info*lam,flow_eye)
        pm=mv(pp,qm+a*lam)
        mdot=mv(pp,a-mv(info,pm))
        f=-.5*mm(pp,info); b=mdot-mv(f,pm)
        mat=flow_eye+step*f
        af=mm(mat,af); cf=mv(mat,cf)+step*b
        return k+1,af.v,af.t,cf.v,cf.t
    _,av,at,cv,ct=tf.while_loop(lambda k,*_:k<controls.flow_steps,bridge,(0,af.v,af.t,cf.v,cf.t),parallel_iterations=1)
    aa=stack([eye_n,ag.broadcast([n,d,d]),Jet(av,at).broadcast([n,d,d])])
    cc=stack([x.constant(tf.zeros_like(m.v)),cg.broadcast([n,d]),Jet(cv,ct)])
    means=mv(aa,m[None])+cc
    cov=mm(mm(aa,q),aa.transpose())
    _,vm,map_margin=checked_chol(cov)
    return means,cov,aa,cc,m,lq,vu & vg & vq & vm,tf.minimum(guide_margin,qmargin),map_margin


def draw_from_maps(maps,beta,uniforms,normals):
    _,_,a,c,m,l,*_=maps
    n=normals.shape[0]
    b=tf.reduce_sum(tf.cast(uniforms[:,0,None]>=tf.cumsum(beta)[None,:-1],tf.int32),-1)
    j=tf.minimum(tf.cast(uniforms[:,1]*tf.cast(n,uniforms.dtype),tf.int32),n-1)
    # Flatten only the component/ancestor axes; derivative axis is preserved.
    index=b*n+j
    ab=a.reshape([3*n,*a.v.shape[-2:]]).gather(index)
    cb=c.reshape([3*n,c.v.shape[-1]]).gather(index)
    u=m.gather(j)+mv(l,m.constant(normals))
    return mv(ab,u)+cb


def tile(value,index,k,axis=0):
    """A fixed-size dynamic slice: XLA must know the tile extent."""
    begin=[0]*value.shape.rank
    begin[axis]=index*k
    sizes=value.shape.as_list()
    sizes[axis]=k
    return tf.slice(value,begin,sizes)


def jet_tile(value,index,k,axis=0):
    return Jet(tile(value.v,index,k,axis),tile(value.t,index,k,axis+1))


def gaussian_mixture(x,means,cov,shared_covariance=False,contraction="direct"):
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import gaussian_mixture as batch_mixture
    result,valid=batch_mixture(x[None],means[None],cov[None],shared_covariance,contraction=contraction)
    return result[0],valid[0]


def component_densities(x,means,cov,shared_flow_covariance=False,contraction="direct"):
    parts=[gaussian_mixture(x,means[b],cov[b],b<2 or shared_flow_covariance,contraction=contraction) for b in range(3)]
    return stack([part[0] for part in parts],-1),tf.reduce_all(tf.stack([part[1] for part in parts]))


def observation_density(models,x,y):
    if models.models[0].observation_log_density_fn is not None:
        return models.apply("observation_log_density_fn","observation_log_density_tangent_fn",x,y,theta=True)
    # Local analytical observation law is small; no filter or pairwise replay.
    from bayesfilter.highdim.covariance_proposal_tf import observation_density as directional
    results=[directional(m,models.theta,x.v,x.t[k],y) for k,m in enumerate(models.models)]
    return Jet(results[0][0],tf.stack([r[1] for r in results]))


def transport(x,logw,controls):
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import transport as batch_transport
    result,row,column=batch_transport(x[None],logw[None],controls)
    return result[0],row[0],column[0]


def reset_cloud(x,logw,controls):
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import reset_cloud as batch_reset
    z,mu,p,valid,diagnostics=batch_reset(x[None],logw[None],controls)
    return z[0],mu[0],p[0],valid[0],diagnostics[0]


def make_score_filter(spec,n,horizon,controls,dtype=tf.float64,jit_compile=True):
    """The scalar entry point delegates to the same batch recurrence at B=1."""
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
    batch=make_batched_score_filter(spec,n,horizon,controls,dtype,jit_compile,batch_size=1)
    signature=list(batch.input_signature)
    signature[0]=tf.TensorSpec([spec.parameter_count],tf.as_dtype(dtype))
    @tf.function(input_signature=signature,jit_compile=jit_compile,autograph=False)
    def program(theta,*args):
        return {key:value[0] for key,value in batch(theta[None],*args).items()}
    return program
