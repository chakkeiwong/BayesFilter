"""Shared batch particle kernels for analytical covariance-proposal scores.

The leading model-position batch is retained in mixture and transport algebra.
Model laws, proposals and reset preserve B and every parameter direction.
No scalar target or model body is mapped over sample rows.
Single-row and multi-row endpoints both call this same recurrence.
"""
import math
from bayesfilter.highdim.covariance_proposal_shared_gaussian_tf import shared_location_terms
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, mm, mv, stack, exp, logsumexp, cholesky_tangent
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.covariance_proposal_score_tf import (
    build_maps, draw_from_maps, recolour, tile, jet_tile)
from bayesfilter.highdim.covariance_proposal_tf import (
    TRACE_FIELDS, VALIDITY_FIELDS, covariance_restoration_tolerance)


def _covariance_quadratic(x,means,precision,covariance_tangent):
    """All-P Gaussian quadratic without a [P,B,N,N,D] intermediate."""
    batch,n,d=x.shape
    p=covariance_tangent.shape[0]
    anchor=x[:,:1]
    xc=x-anchor
    mc=means-anchor
    # u^T dC u = (x-m)^T P^T dC P (x-m), with u=P(x-m).
    h=tf.linalg.matmul(precision[None],
        tf.linalg.matmul(covariance_tangent,precision[None]),transpose_a=True)
    h=.5*(h+tf.linalg.matrix_transpose(h))
    hm=tf.linalg.matvec(h,mc[None])
    xx=tf.reshape(xc[:,:,:,None]*xc[:,:,None,:],[batch,n,d*d])
    hh=tf.reshape(h,[p,batch,h.shape[2],d*d])
    self_x=tf.linalg.matmul(xx[None],hh,transpose_b=True)
    cross=tf.linalg.matmul(xc[None],hm,transpose_b=True)
    self_m=tf.reduce_sum(mc[None]*hm,-1)[:,:,None]
    return self_x-2*cross+self_m


def gaussian_mixture(x,means,cov,shared_covariance=False,contraction="direct",log_weights=None):
    if contraction not in ("direct", "matrix"):
        raise ValueError("unknown shared Gaussian contraction")
    if shared_covariance:
        if log_weights is not None:
            raise ValueError("weighted components require the nonshared density kernel")
        from bayesfilter.highdim.covariance_proposal_shared_gaussian_tf import shared_gaussian_mixture
        return shared_gaussian_mixture(x, means, cov, contraction)
    from bayesfilter.highdim.covariance_proposal_nonshared_gaussian_tf import nonshared_gaussian_mixture
    return nonshared_gaussian_mixture(x, means, cov, log_weights=log_weights)


def transport(x,logw,controls):
    batch,n,d=x.v.shape; p=x.t.shape[0]; dtype=x.v.dtype
    k=select_transport_chunks(n).row_chunk_size; blocks=n//k
    cache=tf.TensorArray(dtype,blocks*blocks,element_shape=[batch,k,k])
    epsilon=tf.constant(controls.reset_epsilon,dtype)
    # Keep the dense block layout: the symmetric cache's conditional transposes
    # reduced memory but materially slowed the enclosing GPU/XLA recurrence.
    def cost(index,cache):
        i=index//blocks; j=index%blocks
        xi=tile(x.v,i,k,1); xj=tile(x.v,j,k,1)
        def coordinate(c,total):
            delta=xi[:,:,c,None]-xj[:,None,:,c]
            return c+1,total+delta*delta
        _,distance=tf.while_loop(lambda c,*_:c<d,coordinate,
            (0,tf.zeros([batch,k,k],dtype)),parallel_iterations=1)
        return index+1,cache.write(index,-distance/epsilon)
    _,cache=tf.while_loop(lambda i,*_:i<blocks*blocks,cost,(0,cache),parallel_iterations=1)
    # Shared position/tangent origins cancel exactly in pairwise differences.
    # Centring prevents large absolute states from destroying FP32 derivatives.
    centred_x=x.v-x.v[:,:1]
    centred_t=x.t-x.t[:,:,:1]
    radial=tf.reduce_sum(centred_x*centred_t,-1)
    invariant_features=tf.concat([centred_x,
        tf.reshape(tf.transpose(centred_t,[1,2,0,3]),[batch,n,p*d]),
        tf.transpose(radial,[1,2,0])],2)
    def reduce_tiles(potential,rows):
        out=tf.TensorArray(dtype,blocks,element_shape=[batch,k]); dout=tf.TensorArray(dtype,blocks,element_shape=[p,batch,k])
        def outer(i,out,dout):
            def inner(j,total,dtotal):
                index=i*blocks+j if rows else j*blocks+i
                cost_tile=cache.read(index)
                if not rows: cost_tile=tf.linalg.matrix_transpose(cost_tile)
                part=jet_tile(potential,j,k,1)
                logits=cost_tile+part.v[:,None]
                primal=tf.reduce_logsumexp(logits,2); weights=tf.nn.softmax(logits,axis=2)
                xi=tile(centred_x,i,k,1); xj=tile(centred_x,j,k,1)
                di=tile(centred_t,i,k,2); dj=tile(centred_t,j,k,2)
                ai=tile(radial,i,k,2); aj=tile(radial,j,k,2)
                packed=tf.concat([tf.transpose(part.t,[1,2,0]),
                    tile(invariant_features,j,k,1)],2)
                averaged=tf.linalg.matmul(weights,packed)
                da=tf.transpose(averaged[:,:,:p],[2,0,1]); mx=averaged[:,:,p:p+d]
                mdx=tf.transpose(tf.reshape(averaged[:,:,p+d:p+d+p*d],[batch,k,p,d]),[2,0,1,3])
                ma=tf.transpose(averaged[:,:,p+d+p*d:],[2,0,1])
                derivative=da-2/epsilon*(ai+ma-tf.reduce_sum(di*mx,-1)-tf.reduce_sum(xi*mdx,-1))
                result=logsumexp(stack([Jet(total,dtotal),Jet(primal,derivative)],-1),-1)
                return j+1,result.v,result.t
            _,v,t=tf.while_loop(lambda j,*_:j<blocks,inner,(0,tf.fill([batch,k],tf.constant(-math.inf,dtype)),tf.zeros([p,batch,k],dtype)),parallel_iterations=1)
            return i+1,out.write(i,v),dout.write(i,t)
        _,out,dout=tf.while_loop(lambda i,*_:i<blocks,outer,(0,out,dout),parallel_iterations=1)
        return Jet(tf.reshape(tf.transpose(out.stack(),[1,0,2]),[batch,n]),tf.reshape(tf.transpose(dout.stack(),[1,2,0,3]),[p,batch,n]))
    zero=x.constant(tf.zeros([batch,n],dtype)); logn=tf.math.log(tf.cast(n,dtype))
    def balance(it,gv,gt,av,at):
        alpha=logw-reduce_tiles(Jet(gv,gt),True)
        gamma=-logn-reduce_tiles(alpha,False)
        return it+1,gamma.v,gamma.t,alpha.v,alpha.t
    _,gv,gt,av,at=tf.while_loop(lambda it,*_:it<controls.reset_steps,balance,(0,zero.v,zero.t,zero.v,zero.t),parallel_iterations=1)
    alpha=Jet(av,at); gamma=Jet(gv,gt)
    normalizer=reduce_tiles(alpha,False)
    rowmass=exp(alpha+reduce_tiles(gamma,True))
    rowerr=tf.reduce_max(tf.abs(rowmass.v-tf.exp(logw.v)),1)
    out=tf.TensorArray(dtype,blocks,element_shape=[batch,k,d]); dout=tf.TensorArray(dtype,blocks,element_shape=[p,batch,k,d]); columns=tf.TensorArray(dtype,blocks,element_shape=[batch,k])
    def output(j,out,dout,columns):
        def source(i,y,dy,col,dcol):
            xi=tile(centred_x,i,k,1); xj=tile(centred_x,j,k,1)
            di=tile(centred_t,i,k,2); dj=tile(centred_t,j,k,2)
            ai=jet_tile(alpha,i,k,1); normal=jet_tile(normalizer,j,k,1)
            trans=tf.exp(cache.read(i*blocks+j)+ai.v[:,:,None]-normal.v[:,None])
            # dC_ij=-2/epsilon*(xi-xj)'(di-dj). Contract its four
            # terms with T before adding parameter axes: no [P,B,K,K].
            radial_i=tile(radial,i,k,2); radial_j=tile(radial,j,k,2)
            def pack(t):
                # [P,B,K,...] -> [B,K,P*...]
                return tf.reshape(tf.transpose(t,[1,2,0,*range(3,t.shape.rank)]),[batch,k,-1])
            features=tf.concat([xi,pack(di),pack(ai.t),pack(radial_i),
                pack(xi[None]*ai.t[...,None]),pack(xi[None]*radial_i[...,None]),
                pack(xi[None,...,:,None]*di[...,None,:]),
                tf.reshape(xi[:,:,:,None]*xi[:,:,None,:],[batch,k,d*d])],-1)
            average=tf.linalg.matmul(trans,features,transpose_a=True)
            cursor=0
            def take(width):
                nonlocal cursor
                part=average[:,:,cursor:cursor+width]; cursor+=width
                return part
            def unpack(width,shape):
                part=tf.reshape(take(width),[batch,k,p,*shape])
                return tf.transpose(part,[2,0,1,*range(3,part.shape.rank)])
            value=take(d); mean_di=unpack(p*d,[d]); mean_da=unpack(p,[])
            mean_radial=unpack(p,[]); mean_xda=unpack(p*d,[d]); mean_xradial=unpack(p*d,[d])
            mean_xdi=unpack(p*d*d,[d,d]); mean_xx=tf.reshape(take(d*d),[batch,k,d,d])
            mass=tf.reduce_sum(trans,1)
            dmass=mean_da-normal.t*mass[None]-2/epsilon*(
                mean_radial+radial_j*mass[None]-tf.reduce_sum(value[None]*dj,-1)-tf.reduce_sum(xj[None]*mean_di,-1))
            derivative=mean_di+mean_xda-normal.t[...,None]*value[None]-2/epsilon*(
                mean_xradial+radial_j[...,None]*value[None]
                -tf.linalg.matvec(mean_xx[None],dj)-tf.linalg.matvec(mean_xdi,xj[None]))
            return i+1,y+value,dy+derivative,col+mass,dcol+dmass
        _,y,dy,col,dcol=tf.while_loop(lambda i,*_:i<blocks,source,
            (0,tf.zeros([batch,k,d],dtype),tf.zeros([p,batch,k,d],dtype),
             tf.zeros([batch,k],dtype),tf.zeros([p,batch,k],dtype)),parallel_iterations=1)
        normalized=Jet(y,dy)/Jet(col,dcol)[:,:,None]
        return j+1,out.write(j,normalized.v),dout.write(j,normalized.t),columns.write(j,col)
    _,out,dout,columns=tf.while_loop(lambda j,*_:j<blocks,output,(0,out,dout,columns),parallel_iterations=1)
    value=tf.reshape(tf.transpose(out.stack(),[1,0,2,3]),[batch,n,d])+x.v[:,:1]
    tangent=tf.reshape(tf.transpose(dout.stack(),[1,2,0,3,4]),[p,batch,n,d])+x.t[:,:,:1]
    column_error=tf.reduce_max(tf.abs(columns.stack()-1),[0,2])
    return Jet(value,tangent),rowerr,column_error


def moments(x,w):
    from bayesfilter.highdim.analytical_jet_tf import normalized_moments
    return normalized_moments(x,w)


PRECISION_POLICIES = ("uniform", "fp64_recursion_fp32_pairs")


def _cast_jet(x, dtype):
    return Jet(tf.cast(x.v, dtype), tf.cast(x.t, dtype))


def precision_gaussian_mixture(x, means, covariance, shared_covariance=False,
                               contraction="direct", precision_policy="uniform", log_weights=None):
    """Same density kernel, with an optional centered FP32 arithmetic boundary."""
    if precision_policy == "uniform":
        return gaussian_mixture(x, means, covariance, shared_covariance, contraction, log_weights)
    if precision_policy != "fp64_recursion_fp32_pairs" or x.v.dtype != tf.float64:
        raise ValueError("mixed pair arithmetic requires FP64 recurrence")
    origin = Jet(x.v[:, :1], x.t[:, :, :1])
    density, valid = gaussian_mixture(
        _cast_jet(x-origin, tf.float32), _cast_jet(means-origin, tf.float32),
        _cast_jet(covariance, tf.float32), shared_covariance, contraction,
        None if log_weights is None else _cast_jet(log_weights, tf.float32))
    return _cast_jet(density, x.v.dtype), valid


def precision_transport(x, log_weights, controls, precision_policy="uniform"):
    """Same transport kernel; restore the moving origin and its sensitivity."""
    if precision_policy == "uniform":
        return transport(x, log_weights, controls)
    if precision_policy != "fp64_recursion_fp32_pairs" or x.v.dtype != tf.float64:
        raise ValueError("mixed pair arithmetic requires FP64 recurrence")
    origin = Jet(x.v[:, :1], x.t[:, :, :1])
    y, row_error, column_error = transport(
        _cast_jet(x-origin, tf.float32), _cast_jet(log_weights, tf.float32), controls)
    return (_cast_jet(y, x.v.dtype)+origin,
            tf.cast(row_error, x.v.dtype), tf.cast(column_error, x.v.dtype))


def reset_cloud(x,logw,controls,precision_policy="uniform",reset_precision_policy="uniform",return_fallback=False):
    """Optional FP64 moments/factors for failed FP32 rows; reuse transport.

    Healthy rows retain their original values and analytical tangents exactly.
    The fallback has its own factorization guard and applies no diagonal ridge.
    """
    if reset_precision_policy not in ("uniform","conditional_fp64"):
        raise ValueError("unknown reset precision policy")
    if reset_precision_policy=="conditional_fp64" and x.v.dtype!=tf.float32:
        raise ValueError("conditional FP64 reset requires FP32 recurrence")
    batch,n,d=x.v.shape
    mu,p=moments(x,exp(logw))
    y,rowerr,colerr=precision_transport(x,logw,controls,precision_policy)
    ym,sy=moments(y,x.constant(tf.fill([batch,n],tf.constant(1./n,x.v.dtype))))
    ar,valid,margin=recolour(p,sy,batch_ndims=1)
    centered=mv(ar[:,None],y-ym[:,None]); centered=centered-centered.mean(1)[:,None]
    z=mu[:,None]+centered
    rd=tf.stack([margin,rowerr,tf.reduce_max(tf.linalg.norm(z.v-y.v,axis=2),1),colerr],1)
    fallback_used=tf.zeros([batch],tf.bool)
    if reset_precision_policy=="conditional_fp64":
        fallback_used=~valid
        def fallback():
            x64,y64,lw64=(_cast_jet(a,tf.float64) for a in (x,y,logw))
            mu64,p64=moments(x64,exp(lw64))
            ym64,sy64=moments(y64,x64.constant(tf.fill([batch,n],tf.constant(1./n,tf.float64))))
            ar64,good64,margin64=recolour(p64,sy64,batch_ndims=1)
            centered64=mv(ar64[:,None],y64-ym64[:,None])
            centered64=centered64-centered64.mean(1)[:,None]
            repaired=_cast_jet(mu64[:,None]+centered64,x.v.dtype)
            zv=tf.where(valid[:,None,None],z.v,repaired.v)
            zt=tf.where(valid[None,:,None,None],z.t,repaired.t)
            diag=tf.stack([tf.cast(margin64,x.v.dtype),rowerr,tf.reduce_max(tf.linalg.norm(zv-y.v,axis=2),1),colerr],1)
            return zv,zt,valid|good64,tf.where(valid[:,None],rd,diag)
        zv,zt,valid,rd=tf.cond(tf.reduce_all(valid),lambda:(z.v,z.t,valid,rd),fallback)
        z=Jet(zv,zt)
    result=(z,mu,p,valid,rd)
    return result+(fallback_used,) if return_fallback else result


def make_batched_score_filter(spec,n,horizon,controls,dtype=tf.float64,jit_compile=True,batch_size=2,record_history=False,precision_policy="uniform",fixed_draw_design=None,reset_precision_policy="uniform",diagnostic_proposal_hook=None):
    """Build the shared recurrence, optionally binding a fixed draw design.

    fixed_draw_design=(beta, uniforms) snapshots the immutable mixture weights
    and uniforms. Original-policy indices are prepared once on CPU; conditional
    weighted-parent indices are recomputed from the current observation. That
    callable takes only (theta, observations, initial, innovations), so callers
    cannot supply mismatched weights/uniforms at execution. The default dynamic
    six-input callable is unchanged. Preparation uses the same FP64 index
    authority; the continuous FP32 recurrence needs no FP64 index arithmetic.

    diagnostic_proposal_hook is an optional diagnostic-only draw/density
    callback. Density slot zero must always retain the transition-mixture
    numerator. A fifth returned item may supply a different proposal log
    density, allowing pure controls without altering the target numerator.
    """
    dtype=tf.as_dtype(dtype); d=spec.dimension; p=spec.parameter_count; o=getattr(spec,"observation_dimension",d); batch=int(batch_size)
    if reset_precision_policy not in ("uniform","conditional_fp64"):
        raise ValueError("unknown reset precision policy")
    if reset_precision_policy=="conditional_fp64" and dtype!=tf.float32:
        raise ValueError("conditional FP64 reset requires FP32 recurrence")
    if precision_policy not in PRECISION_POLICIES:
        raise ValueError("unknown score precision policy")
    if precision_policy != "uniform" and dtype != tf.float64:
        raise ValueError("mixed pair arithmetic requires FP64 recurrence")
    if batch<1 or n<=d or horizon<1: raise ValueError("require B>=1, N>d, T>=1")
    if diagnostic_proposal_hook is not None and fixed_draw_design is not None:
        raise ValueError("observation-dependent diagnostic selection cannot use fixed indices")
    structural_factory = getattr(spec, "structural_model", None)
    if (diagnostic_proposal_hook is None
            and controls.proposal_policy != "original_three_branch"
            and structural_factory is None):
        from bayesfilter.highdim.observation_parent_proposal_tf import make_conditional_proposal
        diagnostic_proposal_hook = make_conditional_proposal(
            spec, controls, batch, precision_policy=precision_policy)
    prepared_indices = None
    if fixed_draw_design is not None:
        from bayesfilter.highdim.covariance_proposal_tf import fixed_design_indices
        with tf.device("/CPU:0"):
            fixed_beta, fixed_uniforms = (tf.identity(tf.convert_to_tensor(value,dtype=dtype))
                                          for value in fixed_draw_design)
            if fixed_beta.shape != tf.TensorShape([3]) or fixed_uniforms.shape != tf.TensorShape([horizon,n,2]):
                raise ValueError("fixed draw design requires beta [3] and uniforms [T,N,2]")
            prepared_indices = tuple(tf.reshape(value,[horizon,n]) for value in
                fixed_design_indices(fixed_beta,tf.reshape(fixed_uniforms,[horizon*n,2]),n))
    innovation_d = getattr(spec, "innovation_dimension", d)
    if structural_factory is not None and controls.proposal_policy == "conditional_observation_weighted":
        raise ValueError(
            "structural completion currently uses uniform parents; set "
            "proposal_policy='conditional_uniform'"
        )
    if structural_factory is not None and diagnostic_proposal_hook is not None:
        raise ValueError("diagnostic proposal hooks are not supported by structural completion")
    if structural_factory is not None and precision_policy != "uniform":
        raise ValueError("structural completion currently requires uniform precision")
    sig=[tf.TensorSpec([batch,p],dtype),tf.TensorSpec([horizon,o],dtype),tf.TensorSpec([n,d],dtype),tf.TensorSpec([horizon,n,innovation_d],dtype),tf.TensorSpec([horizon,n,2],dtype),tf.TensorSpec([3],dtype)]
    @tf.function(input_signature=sig,jit_compile=jit_compile,autograph=False)
    def program(theta,observations,initial,innovations,uniforms,beta):
        from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
        models=NativeModels(spec,theta) if structural_factory is None else structural_factory(theta)
        x=models.initial_cloud(initial)
        trace=tf.TensorArray(dtype,horizon,element_shape=[batch,len(TRACE_FIELDS)]).unstack(tf.fill([horizon,batch,len(TRACE_FIELDS)],tf.constant(float("nan"),dtype)))
        score_trace=tf.TensorArray(dtype,horizon,element_shape=[batch,p]).unstack(tf.fill([horizon,batch,p],tf.constant(float("nan"),dtype)))
        validity_trace=tf.TensorArray(tf.bool,horizon,element_shape=[batch,len(VALIDITY_FIELDS)]).unstack(tf.zeros([horizon,batch,len(VALIDITY_FIELDS)],tf.bool))
        reset_fallback_trace=tf.TensorArray(tf.bool,horizon,element_shape=[batch]).unstack(tf.zeros([horizon,batch],tf.bool))
        covariance_tolerance=covariance_restoration_tolerance(n,d,dtype)
        beta_valid=tf.reduce_all(tf.math.is_finite(beta)) & tf.reduce_all(beta>=0) & (beta[0]>0) & (tf.abs(tf.reduce_sum(beta)-1)<tf.cast(1e-6,dtype))
        ancestors=tf.TensorArray(dtype,horizon if record_history else 0,element_shape=[batch,n,d])
        if record_history:
            ancestors=ancestors.unstack(tf.fill([horizon,batch,n,d],tf.constant(float("nan"),dtype)))
        # Retain discrete selections for conditional-policy derivative checks.
        # Stopped rows stay unavailable while healthy rows continue.
        selections=tf.TensorArray(tf.int32,horizon if diagnostic_proposal_hook is not None else 0,element_shape=[batch,n])
        if diagnostic_proposal_hook is not None:
            selections=selections.unstack(tf.fill([horizon,batch,n],-1))
        def step(t,xv,xt,ell,score,valid,completed,trace,score_trace,ancestors,validity_trace,reset_fallback_trace,selections):
            if record_history:
                # History after a row stops is unavailable, independently of
                # whether other rows (or other shards) continue to execute.
                ancestors=ancestors.write(t,tf.where(valid[:,None,None],xv,
                    tf.constant(float("nan"),dtype)))
            x=Jet(xv,xt)
            indices = None if prepared_indices is None else tuple(value[t] for value in prepared_indices)
            if structural_factory is None:
                maps=build_maps(models,x,observations[t],controls)
                if diagnostic_proposal_hook is None:
                    child=draw_from_maps(maps,beta,uniforms[t],innovations[t],indices=indices)
                    means,cov=maps[:2]
                    density_parts=[precision_gaussian_mixture(child,means[:,b],cov[:,b],shared_covariance=b<2 or models.constant_observation_jacobian,contraction=controls.shared_gaussian_contraction,precision_policy=precision_policy) for b in range(3)]
                    densities=stack([part[0] for part in density_parts],-1)
                    vd=tf.reduce_all(tf.stack([part[1] for part in density_parts]),0)
                else:
                    diagnostic_draw=diagnostic_proposal_hook(models,maps,beta,uniforms[t],innovations[t],observations[t])
                    child,densities,vd,selected=diagnostic_draw[:4]
                    selections=selections.write(t,tf.where(valid[:,None],selected,-1))
            else:
                from bayesfilter.highdim.structural_completion_proposal_tf import structural_proposal
                child,densities,vd,maps=structural_proposal(models,x,observations[t],controls,beta,uniforms[t],innovations[t],indices=indices)
            lq=logsumexp(densities+tf.math.log(beta)[None,None],-1)
            if diagnostic_proposal_hook is not None and structural_factory is None and len(diagnostic_draw)==5:
                lq=diagnostic_draw[4]
            lg=models.observation_density(child,observations[t])
            lr=lg+densities[:,:,0]-lq
            normalizer=logsumexp(lr,1)
            inc=normalizer-tf.math.log(tf.cast(n,dtype)); logw=lr-normalizer[:,None]; w=exp(logw)
            z,mu,pw,vr,rd,fallback_used=reset_cloud(child,logw,controls,precision_policy,reset_precision_policy,True)
            reset_particles=z.v; cd=tf.zeros([batch,3],dtype)
            if controls.correction_steps:
                pre_mean,pre_covariance=moments(z,x.constant(tf.fill([batch,n],tf.constant(1./n,dtype))))
                pre_mean_error=tf.reduce_max(tf.abs(pre_mean.v-mu.v),1)
                pre_covariance_error=tf.linalg.norm(pre_covariance.v-pw.v,axis=[1,2])/tf.maximum(tf.linalg.norm(pw.v,axis=[1,2]),tf.constant(1e-30,dtype))
                from bayesfilter.highdim.covariance_proposal_score_moments_tf import repair_moments
                z,cd=repair_moments(child,w,z,mu,pw,controls)
            zm,zp=moments(z,x.constant(tf.fill([batch,n],tf.constant(1./n,dtype))))
            eps=tf.constant(2.220446049250313e-16 if dtype==tf.float64 else 1.1920928955078125e-7,dtype)
            mean_error=tf.reduce_max(tf.abs(zm.v-mu.v),1)
            cov_error=tf.linalg.norm(zp.v-pw.v,axis=[1,2])/tf.maximum(tf.linalg.norm(pw.v,axis=[1,2]),tf.constant(1e-30,dtype))
            if not controls.correction_steps:
                pre_mean_error,pre_covariance_error=mean_error,cov_error
            mean_scale=tf.maximum(tf.reduce_max(tf.abs(mu.v),1),tf.sqrt(tf.reduce_max(tf.linalg.diag_part(pw.v),1)))
            mean_limit=tf.sqrt(eps)*mean_scale
            roundoff_mean_limit=100*eps*d*tf.maximum(tf.constant(1.,dtype),tf.reduce_max(tf.abs(mu.v),1))
            roundoff_warning=(mean_error>roundoff_mean_limit) | (cov_error>100*eps*d)
            mean_ok=mean_error<=mean_limit
            covariance_ok=tf.math.is_finite(cov_error) & (cov_error<=covariance_tolerance)
            coordinate_move=tf.reduce_max(tf.abs(z.v-reset_particles)/tf.sqrt(tf.linalg.diag_part(pw.v))[:,None],[1,2])
            cap=tf.constant(controls.correction_radius,dtype)*(1+100*eps*n)
            checks=tf.stack([mean_ok,covariance_ok,cd[:,0]<=cap,coordinate_move<=cap,
                tf.reduce_all(tf.math.is_finite(cd),1),maps[6],vd,vr,
                tf.reduce_all(tf.math.is_finite(z.v),[1,2]),tf.reduce_all(tf.math.is_finite(z.t),[0,2,3]),
                tf.math.is_finite(inc.v),tf.reduce_all(tf.math.is_finite(inc.t),0)],1)
            good=tf.reduce_all(checks,1)
            row=tf.stack([inc.v,inc.t[0],1/tf.reduce_sum(w.v*w.v,1),tf.reduce_max(w.v,1),tf.reduce_max(tf.exp(densities.v[:,:,0]-lq.v),1),mean_error,cov_error,rd[:,0],maps[7],maps[8],cd[:,0],cd[:,1],cd[:,2],tf.cast(good,dtype),rd[:,1],rd[:,2],rd[:,3],coordinate_move],1)
            extra=tf.stack([pre_mean_error,pre_covariance_error,mean_limit,
                tf.cast(mean_ok & covariance_ok,dtype),tf.cast(tf.reduce_all(checks[:,2:5],1),dtype),
                tf.fill([batch],covariance_tolerance),tf.cast(roundoff_warning,dtype),mean_scale,
                roundoff_mean_limit,tf.fill([batch],100*eps*d)],1)
            row=tf.concat([row,extra],1)
            # Freeze failed rows after their first failing step. Healthy rows
            # continue independently; a bad row cannot poison another chain.
            accepted=valid & good
            return (t+1,tf.where(accepted[:,None,None],z.v,xv),tf.where(accepted[None,:,None,None],z.t,xt),
                ell+tf.where(accepted,inc.v,0),score+tf.where(accepted[None],inc.t,0),accepted,
                completed+tf.cast(valid,tf.int32),trace.write(t,tf.where(valid[:,None],row,tf.constant(float("nan"),dtype))),
                score_trace.write(t,tf.where(valid[:,None],tf.transpose(inc.t),tf.constant(float("nan"),dtype))),ancestors,validity_trace.write(t,valid[:,None] & checks),
                reset_fallback_trace.write(t,valid & fallback_used),selections)
        initial_valid=tf.fill([batch],beta_valid)
        _,xv,xt,ell,score,valid,completed,trace,st,ancestors,vt,rft,selections=tf.while_loop(lambda t,xv,xt,ell,score,valid,*_:(t<horizon)&tf.reduce_any(valid),step,
            (0,x.v,x.t,tf.zeros([batch],dtype),tf.zeros([p,batch],dtype),initial_valid,tf.zeros([batch],tf.int32),trace,score_trace,ancestors,validity_trace,reset_fallback_trace,selections),parallel_iterations=1)
        result=dict(value=ell,score=tf.transpose(score),valid=valid,steps_completed=completed,particles=xv,particle_tangent=tf.transpose(xt,[1,0,2,3]),trace=tf.transpose(trace.stack(),[1,0,2]),score_trace=tf.transpose(st.stack(),[1,0,2]),
            validity_trace=tf.transpose(vt.stack(),[1,0,2]),covariance_tolerance=covariance_tolerance,
            reset_precision_fallback=tf.transpose(rft.stack(),[1,0]))
        if record_history:
            result["ancestor_history"]=tf.transpose(ancestors.stack(),[1,0,2,3])
        if diagnostic_proposal_hook is not None:
            result["diagnostic_parent_indices"]=tf.transpose(selections.stack(),[1,0,2])
        return result
    if fixed_draw_design is None:
        return program
    @tf.function(input_signature=sig[:4],jit_compile=jit_compile,autograph=False)
    def fixed_program(theta,observations,initial,innovations):
        return program(theta,observations,initial,innovations,fixed_uniforms,fixed_beta)
    return fixed_program
