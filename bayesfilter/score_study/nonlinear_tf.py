"""Scalar nonlinear fixtures, analytical tangents and quadrature reference.

The physical transition is a*x+c*sin(x)+Gaussian noise; observation mean is
H*x+curvature*x**2. The same six parameter meanings and initial law as the
Gaussian fixture are retained. The finite grid requires convergence checks;
it is not an exact oracle. c=curvature=0 must recover the Kalman reference.
"""
from dataclasses import replace
from functools import lru_cache
import tensorflow as tf
from .gaussian_tf import parameterized_model, gaussian_log_density_and_tangent
from .canonical_adapter_tf import gaussian_direction_inputs, make_canonical_kernel


def direction_inputs(theta,direction,initial_noise,transition_curve,observation_curve):
    model,initial,dinitial,P,dP,*_=gaussian_direction_inputs(theta,direction,initial_noise,1,1)
    A,dA,H,dH,*_=parameterized_model(theta,1,1)
    dA=tf.tensordot(direction,dA,1);dH=tf.tensordot(direction,dH,1)
    c=tf.cast(transition_curve,theta.dtype);b=tf.cast(observation_curve,theta.dtype)
    model=replace(model,
        transition_mean_fn=lambda parameter,x:A[0,0]*x+c*tf.sin(x),
        transition_mean_tangent_fn=lambda parameter,x,dx:dA[0,0]*x+(A[0,0]+c*tf.cos(x))*dx,
        observation_fn=lambda x:H[0,0]*x+b*x*x,
        observation_jacobian_fn=lambda x:(H[0,0]+2*b*x)[:,:,None],
        observation_tangent_fn=lambda x,dx:dH[0,0]*x+(H[0,0]+2*b*x)*dx,
        observation_jacobian_tangent_fn=lambda x,dx:(dH[0,0]+2*b*dx)[:,:,None])
    return model,initial,dinitial,P,dP


@lru_cache(maxsize=24)
def make_data_kernel(T,c,b,dtype_name="float64",jit_compile=True):
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([2],tf.int32)],jit_compile=jit_compile)
    def kernel(theta,seed):
        A,_,H,_,m,_,P,_,Q,_,R,_=parameterized_model(theta,1,1)
        draw=lambda shape,i:tf.random.stateless_normal(shape,tf.random.experimental.stateless_fold_in(seed,i),dtype=dtype)
        x=m[0]+tf.sqrt(P[0,0])*draw([],0)
        process=draw([T],1);obsnoise=draw([T],2)
        output=tf.TensorArray(dtype,T)
        def step(t,x,output):
            x=A[0,0]*x+tf.cast(c,dtype)*tf.sin(x)+tf.sqrt(Q[0,0])*process[t]
            y=H[0,0]*x+tf.cast(b,dtype)*x*x+tf.sqrt(R[0,0])*obsnoise[t]
            return t+1,x,output.write(t,y)
        result=tf.while_loop(lambda t,*_:t<T,step,(0,x,output),parallel_iterations=1)
        return result[2].stack()[:,None]
    return kernel


@lru_cache(maxsize=24)
def make_grid_reference(T,points,radius,c,b,dtype_name="float64",jit_compile=True):
    """Trapezoidal forward recursion with explicit six-parameter derivatives.

    Initial and transition tail mass is not renormalized away. Refinement in
    both grid spacing and domain is required before using this as a reference.
    """
    if points<33 or points%2!=1 or radius<=0:raise ValueError("odd grid >=33 and positive domain radius required")
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,1],dtype)],jit_compile=jit_compile)
    def kernel(theta,observations):
        A,dA,H,dH,m,dm,P,dP,Q,dQ,R,dR=parameterized_model(theta,1,1)
        grid=tf.linspace(tf.cast(-radius,dtype),tf.cast(radius,dtype),points)
        measure=tf.concat([tf.constant([.5],dtype),tf.ones([points-2],dtype),tf.constant([.5],dtype)],0)*(2*radius/(points-1))
        def lognormal(residual,dresidual,C,dC):
            return gaussian_log_density_and_tangent(residual[...,None],dresidual[...,None],C,
                                                    dC[:,None,:,:])
        initial,dinitial=lognormal(grid-m[0],-tf.broadcast_to(dm,[6,points]),P,dP)
        mass=tf.exp(initial)*measure;dmass=mass[None,:]*dinitial
        initial_mass=tf.reduce_sum(mass)
        pred=A[0,0]*grid+tf.cast(c,dtype)*tf.sin(grid)
        dpred=dA[:,0,0,None]*grid[None,:]
        residual=grid[None,:]-pred[:,None]
        transition,dtransition=gaussian_log_density_and_tangent(residual[...,None],
            -tf.broadcast_to(dpred[:,:,None,None],[6,points,points,1]),Q,dQ[:,None,None,:,:])
        matrix=tf.exp(transition)*measure[None,:]
        dmatrix=matrix[None,:,:]*dtransition
        means=H[0,0]*grid+tf.cast(b,dtype)*grid*grid
        dmeans=dH[:,0,0,None]*grid[None,:]
        def step(t,mass,dmass,value,score,loss,edge):
            predicted=tf.einsum("i,ij->j",mass,matrix)
            dpredicted=tf.einsum("pi,ij->pj",dmass,matrix)+tf.einsum("i,pij->pj",mass,dmatrix)
            loss=tf.maximum(loss,tf.maximum(tf.cast(0.,dtype),1-tf.reduce_sum(predicted)))
            lg,dlg=lognormal(observations[t,0]-means,-dmeans,R,dR)
            shift=tf.reduce_max(lg);g=tf.exp(lg-shift)
            raw=predicted*g;draw=g[None,:]*(dpredicted+predicted[None,:]*dlg)
            z=tf.reduce_sum(raw);dz=tf.reduce_sum(draw,1)
            mass=raw/z;dmass=(draw-mass[None,:]*dz[:,None])/z
            edge=tf.maximum(edge,tf.reduce_sum(tf.where(tf.abs(grid)>.9*radius,mass,tf.zeros_like(mass))))
            return t+1,mass,dmass,value+tf.math.log(z)+shift,score+dz/z,loss,edge
        out=tf.while_loop(lambda t,*_:t<T,step,(0,mass,dmass,tf.zeros([],dtype),tf.zeros([6],dtype),
            tf.maximum(tf.cast(0.,dtype),1-initial_mass),tf.zeros([],dtype)),parallel_iterations=1)
        mean=tf.reduce_sum(out[1]*grid)
        variance=tf.reduce_sum(out[1]*(grid-mean)**2)
        return out[3],out[4],mean,variance,out[5],out[6]
    return kernel


@lru_cache(maxsize=32)
def make_ledh_kernel(N,T,controls_tuple,c,b,provider="ledh",provider_control=1.,dtype_name="float64",jit_compile=True,
                     return_diagnostics=False):
    from bayesfilter.highdim.ledh_canonical_score_tf import _value_and_analytical_score_impl
    from .mixture_covariance_tf import mixture_schedule,gaussian_mixture_initial
    from .covariance_adapter_tf import SGQFCovarianceProvider
    make_canonical_kernel(1,1,N,T,controls_tuple,dtype_name,jit_compile)
    if provider not in ("ledh","kdm_covariance","sgqf"):raise ValueError("unsupported nonlinear LEDH provider")
    if provider=="kdm_covariance" and not 0<provider_control<=1:raise ValueError("invalid within fraction")
    dtype=tf.as_dtype(dtype_name)
    quadrature = SGQFCovarianceProvider(1,int(provider_control),dtype_name) if provider=="sgqf" else None
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([6],dtype),
        tf.TensorSpec([T,1],dtype),tf.TensorSpec([N,1],dtype),tf.TensorSpec([T,N,1],dtype),tf.TensorSpec([N,1],dtype)],jit_compile=jit_compile)
    def kernel(theta,direction,observations,initial_noise,noise,design):
        model,initial,dinitial,P,dP=direction_inputs(theta,direction,initial_noise,c,b)
        kwargs={}
        if provider=="kdm_covariance":
            kwargs["moment_schedule"]=mixture_schedule(model,theta,observations,
                *gaussian_mixture_initial(theta,direction,1,1,provider_control))
        elif provider=="sgqf":
            kwargs["moment_provider"]=(quadrature.predict,quadrature.update)
        result=_value_and_analytical_score_impl(model,theta,initial,tf.broadcast_to(P,[N,1,1]),
            noise,observations,with_score=True,initial_state_tangent=dinitial,
            initial_covariance_tangent=tf.broadcast_to(dP,[N,1,1]),reset_design=design,
            return_trace=return_diagnostics,**dict(controls_tuple),**kwargs)
        value,score=result[:2]
        if return_diagnostics:
            from .control_diagnostics_tf import compact_control_diagnostics
            return value,score[0],compact_control_diagnostics(result[2])
        return value,score[0]
    return kernel


@lru_cache(maxsize=24)
def make_moment_filter(T,c,b,method="ukf",dtype_name="float64",jit_compile=True):
    """EKF/UKF Gaussian likelihood approximations with total derivatives.

    UKF uses the same alpha=1,beta=2,kappa=0 quadrature as the shared LEDH.
    These approximate the physical filtering distribution; they are baselines,
    not exact nonlinear likelihood oracles.
    """
    from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
        quadrature_predict_with_parameter_tangent,quadrature_update_with_parameter_tangent)
    if method not in ("ukf","ekf"):raise ValueError("unknown moment filter")
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([6],dtype),
        tf.TensorSpec([T,1],dtype)],jit_compile=jit_compile)
    def kernel(theta,direction,observations):
        model,*_=direction_inputs(theta,direction,tf.zeros([1,1],dtype),c,b)
        A,dA,H,dH,m,dm,P,dP,Q,dQ,R,dR=parameterized_model(theta,1,1)
        dA,dH,dm,dP,dQ,dR=[tf.tensordot(direction,x,1) for x in (dA,dH,dm,dP,dQ,dR)]
        def step(t,m,P,dm,dP,value,score,margin,healthy):
            if method=="ukf":
                pred=quadrature_predict_with_parameter_tangent(m,P,dm,dP,
                    lambda x:model.transition_mean_fn(theta,x),
                    lambda x,dx:model.transition_mean_tangent_fn(theta,x,dx),Q,
                    d_process_noise_covariance=dQ,jitter=0.)
                m,P,dm,dP,update_valid,lg,dlg=quadrature_update_with_parameter_tangent(*pred[:4],
                    model.observation_fn,model.observation_tangent_fn,R,observations[t],
                    d_observation_covariance=dR,jitter=0.,return_evidence=True)
                healthy=healthy&tf.reduce_all(pred[4])&tf.reduce_all(update_valid)
            else:
                old=m[0,0];dold=dm[0,0];cov=P[0,0,0];dcov=dP[0,0,0]
                slope=A[0,0]+tf.cast(c,dtype)*tf.cos(old)
                dslope=dA[0,0]-tf.cast(c,dtype)*tf.sin(old)*dold
                pred=A[0,0]*old+tf.cast(c,dtype)*tf.sin(old)
                dpred=dA[0,0]*old+slope*dold
                variance=slope*slope*cov+Q[0,0]
                dvariance=2*slope*dslope*cov+slope*slope*dcov+dQ[0,0]
                h=H[0,0]*pred+tf.cast(b,dtype)*pred*pred
                jac=H[0,0]+2*tf.cast(b,dtype)*pred
                dh=dH[0,0]*pred+jac*dpred;djac=dH[0,0]+2*tf.cast(b,dtype)*dpred
                S=jac*jac*variance+R[0,0]
                dS=2*jac*djac*variance+jac*jac*dvariance+dR[0,0]
                K=variance*jac/S;dK=(dvariance*jac+variance*djac-K*dS)/S
                residual=observations[t,0]-h
                lg,dlg=gaussian_log_density_and_tangent(tf.reshape(residual,[1,1]),
                    tf.reshape(-dh,[1,1,1]),tf.reshape(S,[1,1]),tf.reshape(dS,[1,1,1,1]))
                m=tf.reshape(pred+K*residual,[1,1]);dm=tf.reshape(dpred+dK*residual-K*dh,[1,1])
                # Equivalent scalar Joseph covariance, preserving positivity.
                post=variance*R[0,0]/S
                dpost=(dvariance*R[0,0]+variance*dR[0,0]-post*dS)/S
                P=tf.reshape(post,[1,1,1]);dP=tf.reshape(dpost,[1,1,1])
            margin=tf.minimum(margin,tf.reduce_min(P))
            return t+1,m,P,dm,dP,value+tf.reduce_sum(lg),score+tf.reduce_sum(dlg),margin,healthy
        out=tf.while_loop(lambda t,*_:t<T,step,(0,m[None,:],P[None,:,:],dm[None,:],dP[None,:,:],
            tf.zeros([],dtype),tf.zeros([],dtype),tf.reduce_min(P),tf.constant(True)),parallel_iterations=1)
        valid=out[8]&tf.math.is_finite(out[5])&tf.math.is_finite(out[6])&(out[7]>0)
        return tf.where(valid,out[5],tf.cast(float("nan"),dtype)),tf.where(valid,out[6],tf.cast(float("nan"),dtype)),out[7]
    return kernel


@lru_cache(maxsize=24)
def make_particle_filter(N,T,c,b,adapted=False,resampling=True,dtype_name="float64",jit_compile=True):
    """Bootstrap or locally linearized Gaussian proposal, physical f*g/q.

    Multinomial labels are locally fixed in the derivative. This is a finite
    program derivative, not an unbiased gradient of its expectation.
    """
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,1],dtype),
        tf.TensorSpec([N,1],dtype),tf.TensorSpec([T,N,1],dtype),tf.TensorSpec([T,N],dtype)],jit_compile=jit_compile)
    def kernel(theta,observations,initial_noise,noises,uniforms):
        A,dA,H,dH,m,dm,P,dP,Q,dQ,R,dR=parameterized_model(theta,1,1)
        a,da,h,dh=A[0,0],dA[:,0,0],H[0,0],dH[:,0,0]
        q,dq,r,dr=Q[0,0],dQ[:,0,0],R[0,0],dR[:,0,0]
        x=m[0]+tf.sqrt(P[0,0])*initial_noise[:,0]
        dx=dm+(.5*dP[:,0,0]/tf.sqrt(P[0,0]))[:,None]*initial_noise[None,:,0]
        def density(residual,dresidual,variance,dvariance):
            ell=-.5*(tf.math.log(tf.cast(2.,dtype)*tf.acos(tf.cast(-1.,dtype))*variance)+residual**2/variance)
            dell=-residual[None,:]*dresidual/variance-.5*dvariance/variance+.5*residual[None,:]**2*dvariance/variance**2
            return ell,dell
        def step(t,x,dx,lw,dlw,value,score,ess,margin):
            pred=a*x+tf.cast(c,dtype)*tf.sin(x)
            dpred=da[:,None]*x[None,:]+(a+tf.cast(c,dtype)*tf.cos(x))[None,:]*dx
            if adapted:
                hp=h*pred+tf.cast(b,dtype)*pred*pred
                jac=h+2*tf.cast(b,dtype)*pred
                dhp=dh[:,None]*pred[None,:]+jac[None,:]*dpred
                djac=dh[:,None]+2*tf.cast(b,dtype)*dpred
                S=jac*jac*q+r
                dS=2*jac[None,:]*djac*q+jac[None,:]**2*dq[:,None]+dr[:,None]
                K=q*jac/S;dK=(dq[:,None]*jac[None,:]+q*djac-K[None,:]*dS)/S[None,:]
                residual=observations[t,0]-hp
                mean=pred+K*residual;dmean=dpred+dK*residual[None,:]-K[None,:]*dhp
                variance=q*r/S;dvariance=(dq[:,None]*r+q*dr[:,None]-variance[None,:]*dS)/S[None,:]
                x=mean+tf.sqrt(variance)*noises[t,:,0]
                dx=dmean+.5*dvariance/tf.sqrt(variance)[None,:]*noises[t,None,:,0]
                lf,dlf=density(x-pred,dx-dpred,q,dq[:,None])
                lp,dlp=density(x-mean,dx-dmean,variance,dvariance)
                correction,dcorrection=lf-lp,dlf-dlp
                margin=tf.minimum(margin,tf.reduce_min(variance))
            else:
                x=pred+tf.sqrt(q)*noises[t,:,0]
                dx=dpred+.5*dq[:,None]/tf.sqrt(q)*noises[t,None,:,0]
                correction,dcorrection=tf.zeros([N],dtype),tf.zeros([6,N],dtype)
            obsmean=h*x+tf.cast(b,dtype)*x*x
            dobsmean=dh[:,None]*x[None,:]+(h+2*tf.cast(b,dtype)*x)[None,:]*dx
            lg,dlg=density(observations[t,0]-obsmean,-dobsmean,r,dr[:,None])
            lw,dlw=lw+lg+correction,dlw+dlg+dcorrection
            weights=tf.nn.softmax(lw);ess=tf.minimum(ess,1/tf.reduce_sum(weights**2))
            if resampling:
                value+=tf.reduce_logsumexp(lw)-tf.math.log(tf.cast(N,dtype))
                score+=tf.reduce_sum(weights[None,:]*dlw,1)
                cdf=tf.concat([tf.cumsum(weights)[:-1],tf.ones([1],dtype)],0)
                ancestors=tf.searchsorted(cdf,uniforms[t],side="right")
                x,dx=tf.gather(x,ancestors),tf.gather(dx,ancestors,axis=1)
                lw,dlw=tf.zeros_like(lw),tf.zeros_like(dlw)
            return t+1,x,dx,lw,dlw,value,score,ess,margin
        out=tf.while_loop(lambda t,*_:t<T,step,(0,x,dx,tf.zeros([N],dtype),tf.zeros([6,N],dtype),
            tf.zeros([],dtype),tf.zeros([6],dtype),tf.cast(N,dtype),q),parallel_iterations=1)
        value=out[5]+tf.reduce_logsumexp(out[3])-tf.math.log(tf.cast(N,dtype))
        score=out[6]+tf.reduce_sum(tf.nn.softmax(out[3])[None,:]*out[4],1)
        valid=tf.math.is_finite(value)&tf.reduce_all(tf.math.is_finite(score))&(out[8]>0)
        return tf.where(valid,value,tf.cast(float("nan"),dtype)),tf.where(valid,score,tf.cast(float("nan"),dtype)),out[7]
    return kernel
