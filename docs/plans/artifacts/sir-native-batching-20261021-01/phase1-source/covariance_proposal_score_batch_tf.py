"""Shared batch particle kernels for analytical covariance-proposal scores.

The leading model-position batch is retained in mixture and transport algebra.
Small model callback/map assemblies use a fixed trace-time tuple because the
model protocol takes one parameter vector. No complete filter is row-mapped.
Single-row and multi-row endpoints both call this same recurrence.
"""
import math
from bayesfilter.highdim.covariance_proposal_shared_gaussian_tf import shared_location_terms
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, mm, mv, stack, exp, logsumexp, cholesky_tangent
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.covariance_proposal_score_tf import (
    Models, build_maps, draw_from_maps, observation_density, recolour, tile, jet_tile)
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


def gaussian_mixture(x,means,cov,shared_covariance=False,contraction="direct"):
    if contraction not in ("direct", "matrix"):
        raise ValueError("unknown shared Gaussian contraction")
    if shared_covariance:
        from bayesfilter.highdim.covariance_proposal_shared_gaussian_tf import shared_gaussian_mixture
        return shared_gaussian_mixture(x, means, cov, contraction)
    from bayesfilter.highdim import ledh_small_matrix_tf as sm
    from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
    batch,n,d=x.v.shape; p=x.t.shape[0]; dtype=x.v.dtype
    k=select_transport_chunks(n).row_chunk_size; blocks=n//k
    if shared_covariance: cov=cov[:,:1]
    valid,factor=safe_cholesky(cov.v,"importance_component")
    precision=sm.cholesky_solve(factor,tf.broadcast_to(tf.eye(d,dtype=dtype),tf.shape(factor)))
    norm=-tf.constant(.5*d*math.log(2*math.pi)+math.log(n),dtype)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)),-1)
    trace=tf.einsum("bnij,pbnji->pbn",precision,cov.t)
    out=tf.TensorArray(dtype,blocks,element_shape=[batch,k])
    dout=tf.TensorArray(dtype,blocks,element_shape=[p,batch,k])
    def rows(i,out,dout):
        xv=tile(x.v,i,k,1); xt=tile(x.t,i,k,2)
        def cols(j,total,dtotal):
            mu=tile(means.v,j,k,1); dm=tile(means.t,j,k,2)
            pr=precision if shared_covariance else tile(precision,j,k,1)
            dc=cov.t if shared_covariance else tile(cov.t,j,k,2)
            no=norm if shared_covariance else tile(norm,j,k,1)
            tr=trace if shared_covariance else tile(trace,j,k,2)
            if shared_covariance and contraction == "matrix":
                mahal,linear=shared_location_terms(xv,mu,xt,dm,pr[:,0])
                terms=no[:,None]+mahal
            else:
                residual=xv[:,:,None]-mu[:,None]
                u=tf.linalg.matvec(pr[:,None],residual)
                terms=no[:,None]-.5*tf.reduce_sum(residual*u,-1)
                # A shared sensitivity translation cancels from d(x-m).
                # Remove it before contracting to avoid subtracting large dot
                # products. This changes no primal density or derivative formula.
                origin=xt[:,:,:1]
                linear=-tf.einsum("bijd,pbid->pbij",u,xt-origin)+tf.einsum("bijd,pbjd->pbij",u,dm-origin)
            quadratic=_covariance_quadratic(xv,mu,pr,dc)
            block=logsumexp(Jet(terms,linear+.5*quadratic-.5*tr[:,:,None]),-1)
            joined=logsumexp(stack([Jet(total,dtotal),block],-1),-1)
            return j+1,joined.v,joined.t
        _,v,t=tf.while_loop(lambda j,*_:j<blocks,cols,(0,tf.fill([batch,k],tf.constant(-math.inf,dtype)),tf.zeros([p,batch,k],dtype)),parallel_iterations=1)
        return i+1,out.write(i,v),dout.write(i,t)
    _,out,dout=tf.while_loop(lambda i,*_:i<blocks,rows,(0,out,dout),parallel_iterations=1)
    value=tf.reshape(tf.transpose(out.stack(),[1,0,2]),[batch,n])
    tangent=tf.reshape(tf.transpose(dout.stack(),[1,2,0,3]),[p,batch,n])
    valid=tf.reduce_all(valid,1) & tf.reduce_all(tf.math.is_finite(value),1) & tf.reduce_all(tf.math.is_finite(tangent),[0,2]) & tf.reduce_all(tf.math.is_finite(x.v),[1,2]) & tf.reduce_all(tf.math.is_finite(means.v),[1,2])
    return Jet(value,tangent),valid


def transport(x,logw,controls):
    batch,n,d=x.v.shape; p=x.t.shape[0]; dtype=x.v.dtype
    k=select_transport_chunks(n).row_chunk_size; blocks=n//k
    cache=tf.TensorArray(dtype,blocks*blocks,element_shape=[batch,k,k])
    epsilon=tf.constant(controls.reset_epsilon,dtype)
    def cost(index,cache):
        i=index//blocks; j=index%blocks
        delta=tile(x.v,i,k,1)[:,:,None]-tile(x.v,j,k,1)[:,None]
        return index+1,cache.write(index,-tf.reduce_sum(delta*delta,-1)/epsilon)
    _,cache=tf.while_loop(lambda i,*_:i<blocks*blocks,cost,(0,cache),parallel_iterations=1)
    # Shared position/tangent origins cancel exactly in pairwise differences.
    # Centring prevents large absolute states from destroying FP32 derivatives.
    centred_x=x.v-x.v[:,:1]
    centred_t=x.t-x.t[:,:,:1]
    radial=tf.reduce_sum(centred_x*centred_t,-1)
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
                packed=tf.concat([tf.transpose(part.t,[1,2,0]),xj,
                    tf.reshape(tf.transpose(dj,[1,2,0,3]),[batch,k,p*d]),tf.transpose(aj,[1,2,0])],2)
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
            xi=tile(x.v,i,k,1); xj=tile(x.v,j,k,1)
            di=tile(centred_t,i,k,2); dj=tile(centred_t,j,k,2)
            delta=xi[:,:,None]-xj[:,None]
            derivative=(-2/epsilon)*(tf.einsum("bijd,pbid->pbij",delta,di)-tf.einsum("bijd,pbjd->pbij",delta,dj))
            cost_tile=Jet(cache.read(i*blocks+j),derivative)
            trans=exp(cost_tile+jet_tile(alpha,i,k,1)[:,:,None]-jet_tile(normalizer,j,k,1)[:,None])
            # Normalize the centered primal AND its total derivative.
            # This is identical in real arithmetic to sum_i T_ij*x_i.
            local=Jet(tile(centred_x,i,k,1),tile(centred_t,i,k,2))
            contribution=mm(trans.transpose(),local)
            mass=trans.sum(1)
            return i+1,y+contribution.v,dy+contribution.t,col+mass.v,dcol+mass.t
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
                               contraction="direct", precision_policy="uniform"):
    """Same density kernel, with an optional centered FP32 arithmetic boundary."""
    if precision_policy == "uniform":
        return gaussian_mixture(x, means, covariance, shared_covariance, contraction)
    if precision_policy != "fp64_recursion_fp32_pairs" or x.v.dtype != tf.float64:
        raise ValueError("mixed pair arithmetic requires FP64 recurrence")
    origin = Jet(x.v[:, :1], x.t[:, :, :1])
    density, valid = gaussian_mixture(
        _cast_jet(x-origin, tf.float32), _cast_jet(means-origin, tf.float32),
        _cast_jet(covariance, tf.float32), shared_covariance, contraction)
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


def reset_cloud(x,logw,controls,precision_policy="uniform"):
    batch,n,d=x.v.shape
    mu,p=moments(x,exp(logw))
    y,rowerr,colerr=precision_transport(x,logw,controls,precision_policy)
    ym,sy=moments(y,x.constant(tf.fill([batch,n],tf.constant(1./n,x.v.dtype))))
    ar,valid,margin=recolour(p,sy,batch_ndims=1)
    centered=mv(ar[:,None],y-ym[:,None]); centered=centered-centered.mean(1)[:,None]
    z=mu[:,None]+centered
    return z,mu,p,valid,tf.stack([margin,rowerr,tf.reduce_max(tf.linalg.norm(z.v-y.v,axis=2),1),colerr],1)


def make_batched_score_filter(spec,n,horizon,controls,dtype=tf.float64,jit_compile=True,batch_size=2,record_history=False,precision_policy="uniform",fixed_draw_design=None):
    """Build the shared recurrence, optionally binding a fixed draw design.

    fixed_draw_design=(beta, uniforms) snapshots the immutable mixture weights
    and uniforms and prepares their exact integer indices once on CPU. That
    callable takes only (theta, observations, initial, innovations), so callers
    cannot supply mismatched weights/uniforms at execution. The default dynamic
    six-input callable is unchanged. Preparation uses the same FP64 index
    authority; the continuous FP32 recurrence needs no FP64 index arithmetic.
    """
    dtype=tf.as_dtype(dtype); d=spec.dimension; p=spec.parameter_count; o=getattr(spec,"observation_dimension",d); batch=int(batch_size)
    if precision_policy not in PRECISION_POLICIES:
        raise ValueError("unknown score precision policy")
    if precision_policy != "uniform" and dtype != tf.float64:
        raise ValueError("mixed pair arithmetic requires FP64 recurrence")
    if batch<1 or n<=d or horizon<1: raise ValueError("require B>=1, N>d, T>=1")
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
    sig=[tf.TensorSpec([batch,p],dtype),tf.TensorSpec([horizon,o],dtype),tf.TensorSpec([n,d],dtype),tf.TensorSpec([horizon,n,d],dtype),tf.TensorSpec([horizon,n,2],dtype),tf.TensorSpec([3],dtype)]
    @tf.function(input_signature=sig,jit_compile=jit_compile,autograph=False)
    def program(theta,observations,initial,innovations,uniforms,beta):
        from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
        models=NativeModels(spec,theta)
        x=models.initial_cloud(initial)
        trace=tf.TensorArray(dtype,horizon,element_shape=[batch,len(TRACE_FIELDS)]).unstack(tf.fill([horizon,batch,len(TRACE_FIELDS)],tf.constant(float("nan"),dtype)))
        score_trace=tf.TensorArray(dtype,horizon,element_shape=[batch,p]).unstack(tf.fill([horizon,batch,p],tf.constant(float("nan"),dtype)))
        validity_trace=tf.TensorArray(tf.bool,horizon,element_shape=[batch,len(VALIDITY_FIELDS)]).unstack(tf.zeros([horizon,batch,len(VALIDITY_FIELDS)],tf.bool))
        covariance_tolerance=covariance_restoration_tolerance(n,d,dtype)
        beta_valid=tf.reduce_all(tf.math.is_finite(beta)) & tf.reduce_all(beta>=0) & (beta[0]>0) & (tf.abs(tf.reduce_sum(beta)-1)<tf.cast(1e-6,dtype))
        ancestors=tf.TensorArray(dtype,horizon if record_history else 0,element_shape=[batch,n,d])
        if record_history:
            ancestors=ancestors.unstack(tf.fill([horizon,batch,n,d],tf.constant(float("nan"),dtype)))
        def step(t,xv,xt,ell,score,valid,completed,trace,score_trace,ancestors,validity_trace):
            if record_history:
                ancestors=ancestors.write(t,xv)
            x=Jet(xv,xt)
            maps=build_maps(models,x,observations[t],controls)
            indices = None if prepared_indices is None else tuple(value[t] for value in prepared_indices)
            child=draw_from_maps(maps,beta,uniforms[t],innovations[t],indices=indices)
            means,cov=maps[:2]
            density_parts=[precision_gaussian_mixture(child,means[:,b],cov[:,b],shared_covariance=b<2 or models.constant_observation_jacobian,contraction=controls.shared_gaussian_contraction,precision_policy=precision_policy) for b in range(3)]
            densities=stack([part[0] for part in density_parts],-1)
            vd=tf.reduce_all(tf.stack([part[1] for part in density_parts]),0)
            lq=logsumexp(densities+tf.math.log(beta)[None,None],-1)
            lg=models.observation_density(child,observations[t])
            lr=lg+densities[:,:,0]-lq
            normalizer=logsumexp(lr,1)
            inc=normalizer-tf.math.log(tf.cast(n,dtype)); logw=lr-normalizer[:,None]; w=exp(logw)
            z,mu,pw,vr,rd=reset_cloud(child,logw,controls,precision_policy)
            reset_particles=z.v; cd=tf.zeros([batch,3],dtype)
            if controls.correction_steps:
                from bayesfilter.highdim.covariance_proposal_score_moments_tf import repair_moments
                z,cd=repair_moments(child,w,z,mu,pw,controls)
            zm,zp=moments(z,x.constant(tf.fill([batch,n],tf.constant(1./n,dtype))))
            eps=tf.constant(2.220446049250313e-16 if dtype==tf.float64 else 1.1920928955078125e-7,dtype)
            mean_error=tf.reduce_max(tf.abs(zm.v-mu.v),1)
            cov_error=tf.linalg.norm(zp.v-pw.v,axis=[1,2])/tf.maximum(tf.linalg.norm(pw.v,axis=[1,2]),tf.constant(1e-30,dtype))
            mean_ok=mean_error<=100*eps*d*tf.maximum(tf.constant(1.,dtype),tf.reduce_max(tf.abs(mu.v),1))
            covariance_ok=tf.math.is_finite(cov_error) & (cov_error<=covariance_tolerance)
            coordinate_move=tf.reduce_max(tf.abs(z.v-reset_particles)/tf.sqrt(tf.linalg.diag_part(pw.v))[:,None],[1,2])
            cap=tf.constant(controls.correction_radius,dtype)*(1+100*eps*n)
            checks=tf.stack([mean_ok,covariance_ok,cd[:,0]<=cap,coordinate_move<=cap,
                tf.reduce_all(tf.math.is_finite(cd),1),maps[6],vd,vr,
                tf.reduce_all(tf.math.is_finite(z.v),[1,2]),tf.reduce_all(tf.math.is_finite(z.t),[0,2,3]),
                tf.math.is_finite(inc.v),tf.reduce_all(tf.math.is_finite(inc.t),0)],1)
            good=tf.reduce_all(checks,1)
            row=tf.stack([inc.v,inc.t[0],1/tf.reduce_sum(w.v*w.v,1),tf.reduce_max(w.v,1),tf.reduce_max(tf.exp(densities.v[:,:,0]-lq.v),1),mean_error,cov_error,rd[:,0],maps[7],maps[8],cd[:,0],cd[:,1],cd[:,2],tf.cast(good,dtype),rd[:,1],rd[:,2],rd[:,3],coordinate_move],1)
            # Freeze failed rows after their first failing step. Healthy rows
            # continue independently; a bad row cannot poison another chain.
            accepted=valid & good
            return (t+1,tf.where(accepted[:,None,None],z.v,xv),tf.where(accepted[None,:,None,None],z.t,xt),
                ell+tf.where(accepted,inc.v,0),score+tf.where(accepted[None],inc.t,0),accepted,
                completed+tf.cast(valid,tf.int32),trace.write(t,tf.where(valid[:,None],row,tf.constant(float("nan"),dtype))),
                score_trace.write(t,tf.where(valid[:,None],tf.transpose(inc.t),tf.constant(float("nan"),dtype))),ancestors,validity_trace.write(t,valid[:,None] & checks))
        initial_valid=tf.fill([batch],beta_valid)
        _,xv,xt,ell,score,valid,completed,trace,st,ancestors,vt=tf.while_loop(lambda t,xv,xt,ell,score,valid,*_:(t<horizon)&tf.reduce_any(valid),step,
            (0,x.v,x.t,tf.zeros([batch],dtype),tf.zeros([p,batch],dtype),initial_valid,tf.zeros([batch],tf.int32),trace,score_trace,ancestors,validity_trace),parallel_iterations=1)
        result=dict(value=ell,score=tf.transpose(score),valid=valid,steps_completed=completed,particles=xv,particle_tangent=tf.transpose(xt,[1,0,2,3]),trace=tf.transpose(trace.stack(),[1,0,2]),score_trace=tf.transpose(st.stack(),[1,0,2]),
            validity_trace=tf.transpose(vt.stack(),[1,0,2]),covariance_tolerance=covariance_tolerance)
        if record_history:
            result["ancestor_history"]=tf.transpose(ancestors.stack(),[1,0,2,3])
        return result
    if fixed_draw_design is None:
        return program
    @tf.function(input_signature=sig[:4],jit_compile=jit_compile,autograph=False)
    def fixed_program(theta,observations,initial,innovations):
        return program(theta,observations,initial,innovations,fixed_uniforms,fixed_beta)
    return fixed_program
