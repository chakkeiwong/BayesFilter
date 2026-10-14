"""Full-covariance Gaussian controlled SMC, paper-derived TensorFlow kernels.

All quadratic cross terms are retained. A nonlinear transition-mean callable
is allowed; transition covariance is constant. Precision failures are flagged by default; named optional projections
are isolated repair candidates, never implicit numerical defaults. The explicit centered-QR option solves the same least-squares
objective and is not a ridge or damping policy. Its admission needs evidence.
"""
from __future__ import annotations
import math
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from bayesfilter.replication.controlled_smc_tf import systematic_ancestors


def lorenz_rk4(x, forcing, interval=.1):
    """Author Lorenz drift and ten classical RK4 substeps; leading axes batch."""
    step = interval/10
    def drift(z):
        return (tf.roll(z,-1,-1)-tf.roll(z,2,-1))*tf.roll(z,1,-1)-z+forcing
    def body(i, z):
        k1=drift(z); k2=drift(z+.5*step*k1)
        k3=drift(z+.5*step*k2); k4=drift(z+step*k3)
        return i+1,z+(step/6)*(k1+2*k2+2*k3+k4)
    return tf.while_loop(lambda i,_:i<10,body,(tf.constant(0),x),parallel_iterations=1)[1]


def quadratic(x, a, b, c):
    return tf.einsum('...ni,...ij,...nj->...n',x,a,x)+tf.einsum('...ni,...i->...n',x,b)+c[...,None]


def covariance_from_precision(precision):
    """Checked Cholesky; rejected inputs cannot produce valid accepted output."""
    values=tf.linalg.eigvalsh(precision)
    scale=tf.reduce_max(tf.abs(values),axis=-1)
    margin=tf.reduce_min(values,axis=-1)/scale
    asymmetry=tf.reduce_max(tf.abs(precision-tf.linalg.matrix_transpose(precision)),axis=[-2,-1])
    valid=(tf.reduce_all(tf.math.is_finite(precision),axis=[-2,-1])
           & (margin>64*2.220446049250313e-16)
           & (asymmetry<=64*2.220446049250313e-16*scale))
    size=tf.shape(precision)[-1]
    identity=tf.eye(size,batch_shape=tf.shape(precision)[:-2],dtype=precision.dtype)
    # This substitution is evaluated only for rejected inputs; valid=False is
    # propagated to the consumer. It never alters an accepted matrix.
    factor=tf.linalg.cholesky(tf.where(valid[...,None,None],precision,identity))
    covariance=tf.linalg.cholesky_solve(factor,identity)
    logdet=-2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)),axis=-1)
    return covariance,logdet,margin,valid


def multivariate_twist(mean,base_precision,base_logdet,a,b,c):
    covariance,logdet,margin,valid=covariance_from_precision(base_precision+2*a)
    shift=tf.einsum('...ni,...ij->...nj',mean,base_precision)-b[...,None,:]
    twisted_mean=tf.einsum('...ni,...ij->...nj',shift,covariance)
    integral=.5*(logdet-base_logdet)-c
    integral=integral[...,None]+.5*tf.reduce_sum(shift*twisted_mean,axis=-1)
    integral-=.5*tf.einsum('...ni,...ij,...nj->...n',mean,base_precision,mean)
    return twisted_mean,covariance,integral,margin,valid


def centered_correction_integral(mean,base_precision,base_logdet,old_a,old_b,old_c,
                                 center,scale,a,b,c):
    """Exact log-integral ratio via a correction in its fitted coordinates."""
    old_mean,old_covariance,_,_,old_valid=multivariate_twist(
        mean,base_precision,base_logdet,old_a,old_b,old_c)
    zmean=(old_mean-center[...,None,:])/scale[...,None,:]
    zprecision=(base_precision+2*old_a)*scale[..., :,None]*scale[...,None,:]
    old_logdet=2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(old_covariance))),-1)
    zlogdet=old_logdet-2*tf.reduce_sum(tf.math.log(scale),-1)
    _,_,integral,_,valid=multivariate_twist(zmean,zprecision,zlogdet,a,b,c)
    return integral,tf.reduce_all(old_valid)&tf.reduce_all(valid)



def projected_correction_integral(mean,base_precision,base_logdet,old_a,old_b,old_c,
                                  center,scale,a,b,c,added):
    """Same projected polynomial, evaluated without expanding its added quadratic."""
    def original():
        return centered_correction_integral(mean,base_precision,base_logdet,
            old_a,old_b,old_c,center,scale,a,b,c)
    def protected():
        old_mean,old_cov,_,_,old_valid=multivariate_twist(
            mean,base_precision,base_logdet,old_a,old_b,old_c)
        old_precision=base_precision+2*old_a
        raw_a=a/(scale[..., :,None]*scale[...,None,:])
        raw_b=b/scale-2*tf.einsum('bi,bij->bj',center,raw_a)
        covariance,logdet,_,valid=covariance_from_precision(old_precision+2*(raw_a+added))
        shift=tf.einsum('bni,bij->bnj',old_mean,old_precision)-raw_b[:,None,:]
        new_mean=tf.einsum('bni,bij->bnj',shift,covariance)
        z=(new_mean-center[:,None,:])/scale[:,None,:]
        old_logdet=2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(old_cov))),-1)
        distance=new_mean-old_mean
        integral=.5*(logdet-old_logdet)[:,None]-quadratic(z,a,b,c)
        integral-=tf.einsum('bni,bij,bnj->bn',new_mean,added,new_mean)
        integral-=.5*tf.einsum('bni,bij,bnj->bn',distance,old_precision,distance)
        # Nonprojected rows retain the original evaluation in a mixed batch.
        expanded_a=a+added*scale[..., :,None]*scale[...,None,:]
        expanded_b=b+2*scale*tf.einsum('bi,bij->bj',center,added)
        expanded_c=c+tf.einsum('bi,bij,bj->b',center,added,center)
        fallback,fallback_valid=centered_correction_integral(mean,base_precision,base_logdet,
            old_a,old_b,old_c,center,scale,expanded_a,expanded_b,expanded_c)
        active=tf.reduce_any(added!=0,axis=[-2,-1])
        integral=tf.where(active[:,None],integral,fallback)
        return integral,tf.reduce_all(old_valid)&tf.reduce_all(valid)&fallback_valid&tf.reduce_all(tf.math.is_finite(integral))
    return tf.cond(tf.reduce_any(added!=0),protected,original)


def refine_symmetric_eigenbasis(matrix, vectors, tolerance):
    """Native batched Jacobi refinement of a nearly diagonal Rayleigh quotient.

    R is an orthogonal plane rotation, so B <- R'BR and V <- VR preserve
    the eigendecomposition target. Acceptance uses the original-matrix residual
    at the caller, not merely this iteration's stopping condition.
    """
    dimension=matrix.shape[-1]
    identity=tf.eye(dimension,batch_shape=tf.shape(matrix)[:-2],dtype=matrix.dtype)
    transformed=tf.linalg.matmul(vectors,tf.linalg.matmul(matrix,vectors),transpose_a=True)
    transformed=.5*(transformed+tf.linalg.matrix_transpose(transformed))
    scale=tf.sqrt(tf.reduce_sum(matrix**2,[-2,-1]))
    def offdiag(value):
        return value-tf.linalg.diag(tf.linalg.diag_part(value))
    def active(value):
        return tf.sqrt(tf.reduce_sum(offdiag(value)**2,[-2,-1]))>tolerance*scale
    def step(count,value,basis):
        off=offdiag(value)
        flat=tf.argmax(tf.reshape(tf.abs(off),[-1,dimension*dimension]),axis=-1,output_type=tf.int32)
        p,q=flat//dimension,flat%dimension
        ep=tf.one_hot(p,dimension,dtype=matrix.dtype)
        eq=tf.one_hot(q,dimension,dtype=matrix.dtype)
        ap=tf.einsum('bi,bij,bj->b',ep,value,ep)
        aq=tf.einsum('bi,bij,bj->b',eq,value,eq)
        pivot=tf.einsum('bi,bij,bj->b',ep,value,eq)
        do_rotate=active(value)&(pivot!=0)
        tau=(aq-ap)/(2*tf.where(do_rotate,pivot,tf.ones_like(pivot)))
        sign=tf.where(tau>=0,tf.ones_like(tau),-tf.ones_like(tau))
        tangent=sign/(tf.abs(tau)+tf.sqrt(1+tau*tau))
        tangent=tf.where(do_rotate,tangent,tf.zeros_like(tangent))
        cosine=tf.math.rsqrt(1+tangent*tangent);sine=tangent*cosine
        pp=ep[..., :,None]*ep[...,None,:];qq=eq[..., :,None]*eq[...,None,:]
        pq=ep[..., :,None]*eq[...,None,:]
        rotation=identity+(cosine-1)[...,None,None]*(pp+qq)+sine[...,None,None]*(pq-tf.linalg.matrix_transpose(pq))
        value=tf.linalg.matmul(rotation,tf.linalg.matmul(value,rotation),transpose_a=True)
        value=.5*(value+tf.linalg.matrix_transpose(value))
        return count+1,value,tf.linalg.matmul(basis,rotation)
    count,transformed,vectors=tf.while_loop(
        lambda count,value,basis:(count<20*dimension*dimension)&tf.reduce_any(active(value)),
        step,(tf.constant(0),transformed,vectors),parallel_iterations=1)
    values=tf.linalg.diag_part(transformed)
    order=tf.argsort(values,axis=-1)
    return tf.gather(values,order,batch_dims=1),tf.gather(vectors,order,axis=2,batch_dims=1),count


def constrain_precision_increment(base_precision, old_a, fitted, policy,
                                  relative_floor=2.220446049250313e-16**.5, eigen_tolerance=None, eigen_refinement=False, activation_margin=None):
    """Optional policy repair; every changed coefficient enters the same fit.

    nearest_precision implements a Frobenius projection of proposed precision.
    backtrack_precision scales the whole correction by a feasible dyadic step.
    Neither changes the model density or silently modifies a covariance.
    An explicit activation_margin separates feasibility from repair conditioning;
    None preserves the historical optional policy's coupled threshold.
    """
    dtype=old_a.dtype
    threshold=tf.constant(relative_floor,dtype)
    raw=base_precision+2*(old_a+fitted['A'])
    if eigen_tolerance is None or eigen_refinement:
        eigen,vectors=tf.linalg.eigh(raw)
    else:
        eigen,vectors=gen_xla_ops.xla_self_adjoint_eig(
            raw,lower=True,max_iter=100,epsilon=eigen_tolerance)
    rotations=tf.constant(0)
    if eigen_refinement:
        eigen,vectors,rotations=refine_symmetric_eigenbasis(raw,vectors,eigen_tolerance)
    residual=tf.linalg.matmul(raw,vectors)-vectors*eigen[...,None,:]
    eig_residual=tf.sqrt(tf.reduce_sum(residual**2,[-2,-1]))/tf.sqrt(tf.reduce_sum(raw**2,[-2,-1]))
    identity=tf.eye(tf.shape(raw)[-1],batch_shape=tf.shape(raw)[:-2],dtype=dtype)
    eig_orthogonality=tf.reduce_max(tf.abs(
        tf.linalg.matmul(vectors,vectors,transpose_a=True)-identity),[-2,-1])
    magnitude=tf.reduce_max(tf.abs(eigen),axis=-1)
    raw_margin=eigen[...,0]/magnitude
    unsafe=(raw_margin<threshold if activation_margin is None
            else raw_margin<=tf.constant(activation_margin,dtype))
    fraction=tf.ones_like(raw_margin)
    result=dict(fitted)
    if eigen_tolerance is not None:
        eig_limit=256*tf.cast(tf.shape(raw)[-1],dtype)*tf.constant(2.220446049250313e-16,dtype)
        result['valid'] &= tf.reduce_all((eig_residual<=eig_limit)&(eig_orthogonality<=eig_limit))
    if policy in ('nearest_precision','nearest_covariance'):
        if policy=='nearest_covariance':
            nonsingular=tf.reduce_all(eigen!=0,axis=-1)
            inverse=1/tf.where(eigen!=0,eigen,tf.ones_like(eigen))
            floor=threshold*tf.reduce_max(tf.abs(inverse),axis=-1,keepdims=True)
            clipped=1/tf.maximum(inverse,floor)
            result['valid'] &= tf.reduce_all((~unsafe)|nonsingular)
        else:
            clipped=tf.maximum(eigen,threshold*magnitude[...,None])
        projected=tf.linalg.matmul(vectors*clipped[...,None,:],vectors,transpose_b=True)
        projected=.5*(projected+tf.linalg.matrix_transpose(projected))
        added=tf.where(unsafe[...,None,None],.5*(projected-raw),tf.zeros_like(raw))
        result['A']=tf.where(unsafe[...,None,None],fitted['A']+added,fitted['A'])
        mu,s=fitted['center'],fitted['scale']
        result['centered_A']=tf.where(unsafe[...,None,None],
            fitted['centered_A']+added*s[..., :,None]*s[...,None,:],fitted['centered_A'])
        linear=2*s*tf.einsum('bi,bij->bj',mu,added)
        constant=tf.einsum('bi,bij,bj->b',mu,added,mu)
        result['centered_b']=tf.where(unsafe[...,None],fitted['centered_b']+linear,fitted['centered_b'])
        result['centered_c']=tf.where(unsafe,fitted['centered_c']+constant,fitted['centered_c'])
        result['projection_added_A']=added
    elif policy=='backtrack_precision':
        def insufficient(alpha):
            candidate=base_precision+2*(old_a+alpha[...,None,None]*fitted['A'])
            values=tf.linalg.eigvalsh(candidate)
            return values[...,0] < threshold*tf.reduce_max(tf.abs(values),-1)
        _,fraction=tf.while_loop(lambda i,alpha:(i<53)&tf.reduce_any(insufficient(alpha)),
            lambda i,alpha:(i+1,tf.where(insufficient(alpha),alpha*.5,alpha)),
            (tf.constant(0),fraction),parallel_iterations=1)
        for name in ('A','b','c','centered_A','centered_b','centered_c'):
            shape=[tf.shape(fraction)[0]]+[1]*(fitted[name].shape.rank-1)
            result[name]=fitted[name]*tf.reshape(fraction,shape)
    else:
        raise ValueError('Explicit recognized precision repair required')
    changed=base_precision+2*(old_a+result['A'])
    correction=tf.sqrt(tf.reduce_sum((changed-raw)**2,axis=[-2,-1]))
    size=tf.sqrt(tf.reduce_sum(raw**2,axis=[-2,-1]))
    old_valid=covariance_from_precision(base_precision+2*old_a)[3]
    result['valid'] &= tf.reduce_all(old_valid)
    for key in ('A','b','c','centered_A','centered_b','centered_c'):
        result['valid'] &= tf.reduce_all(tf.math.is_finite(result[key]))
    return result,dict(protection_applied=tf.cast(unsafe,dtype),protection_fraction=fraction,
        raw_precision_margin=raw_margin,relative_precision_change=correction/size,
        eigen_residual=eig_residual,eigen_orthogonality=eig_orthogonality,
        eigen_rotations=tf.fill(tf.shape(eig_residual),tf.cast(rotations,dtype)))

class MultivariateGaussianControlledSMC:
    """Controlled-SMC implementation with an optional diagnostic guard candidate.

    ``first_failure_guards=True`` is experimental: full healthy coefficient
    parity failed in attempts132/133. Local guard tests do not admit this
    candidate as an equivalent numerical or production route.
    """
    def __init__(self,*,observations,particles,replicas,dimension,observation_dimension,
                 initial_mean,initial_covariance,transition_covariance,transition_mean,
                 observation_log_density,solver,jit_compile=True,precision_policy="none",
                 precision_floor=None,precision_eigen_tolerance=None,precision_eigen_refinement=False,precision_stable_integral=False,precision_activation_margin=None,first_failure_guards=False):
        if solver not in ('source_normal_equations','centered_qr'):
            raise ValueError('Explicit least-squares solver required')
        if dimension<2 or min(observations,particles,replicas,observation_dimension)<1:
            raise ValueError('Positive sizes and a multivariate state are required')
        self.h,self.n,self.b,self.d=observations,particles,replicas,dimension
        self.f,self.logg,self.solver=transition_mean,observation_log_density,solver
        if precision_policy not in ('none','nearest_precision','nearest_covariance','backtrack_precision'):
            raise ValueError('Unknown precision policy')
        if precision_policy!='none' and solver!='centered_qr':
            raise ValueError('Precision repair requires the reviewed centered QR route')
        if precision_floor is not None and (precision_policy=='none' or
                not math.isfinite(precision_floor) or not 0 < precision_floor < 1):
            raise ValueError('An explicit relative floor in (0,1) requires a repair policy')
        if precision_eigen_tolerance is not None and (
                precision_policy=='none' or not jit_compile or
                not math.isfinite(precision_eigen_tolerance) or not 0 < precision_eigen_tolerance < 1):
            raise ValueError('Explicit eigen tolerance requires an XLA repair policy')
        if precision_eigen_refinement and precision_eigen_tolerance is None:
            raise ValueError('Eigen refinement requires an explicit tolerance')
        if precision_stable_integral and precision_policy not in ("nearest_covariance","nearest_precision"):
            raise ValueError("Stable projected integral requires a spectral projection")
        self.precision_stable_integral=precision_stable_integral
        self.precision_eigen_refinement=precision_eigen_refinement
        self.precision_eigen_tolerance=precision_eigen_tolerance
        self.precision_policy=precision_policy
        self.precision_floor=(2.220446049250313e-16**.5 if precision_floor is None
                              else float(precision_floor))
        if precision_activation_margin is not None and (
                precision_policy!="nearest_covariance" or
                not math.isfinite(precision_activation_margin) or
                not 0<=precision_activation_margin<self.precision_floor):
            raise ValueError("Explicit activation margin requires covariance repair and 0<=margin<floor")
        self.precision_activation_margin=precision_activation_margin
        self.dtype=tf.float64; d=self.dtype
        self.m0=tf.constant(initial_mean,d); self.c0=tf.constant(initial_covariance,d)
        self.q=tf.constant(transition_covariance,d)
        for covariance in (self.c0,self.q):
            tf.debugging.assert_equal(tf.shape(covariance),[dimension,dimension])
            tf.debugging.assert_equal(covariance_from_precision(covariance)[3],True,
                                      message='Base covariance must be symmetric positive definite')
        self.p0=tf.linalg.inv(self.c0); self.p=tf.linalg.inv(self.q)
        self.ld0=tf.linalg.slogdet(self.c0)[1]; self.ldq=tf.linalg.slogdet(self.q)[1]
        self.pairs=[(i,j) for i in range(dimension) for j in range(i+1,dimension)]
        self.width=1+2*dimension+len(self.pairs)
        basis=[]
        for k in range(self.width):
            matrix=[[0.]*dimension for _ in range(dimension)]
            if 1+dimension<=k<1+dimension+len(self.pairs):
                i,j=self.pairs[k-1-dimension];matrix[i][j]=matrix[j][i]=.5
            elif k>=1+dimension+len(self.pairs):
                i=k-1-dimension-len(self.pairs);matrix[i][i]=1.
            basis.append(matrix)
        self.basis=tf.constant(basis,d)
        a=tf.TensorSpec([observations,replicas,dimension,dimension],d)
        b=tf.TensorSpec([observations,replicas,dimension],d)
        c=tf.TensorSpec([observations,replicas],d)
        y=tf.TensorSpec([observations,replicas,observation_dimension],d)
        x=tf.TensorSpec([observations,replicas,particles,dimension],d)
        lw=tf.TensorSpec([observations,replicas,particles],d)
        self.forward=tf.function(self._forward_guarded if first_failure_guards else self._forward,input_signature=[a,b,c,y,x,c],jit_compile=jit_compile,autograph=False)
        self.cloud_weights=tf.function(self._cloud_weights,input_signature=[a,b,c,y,x],jit_compile=jit_compile,autograph=False)
        self.fit=tf.function(self._fit_guarded if first_failure_guards else self._fit,input_signature=[a,b,c,x,lw],jit_compile=jit_compile,autograph=False)
        self.regress=tf.function(self._regress,input_signature=[tf.TensorSpec([replicas,particles,dimension],d),tf.TensorSpec([replicas,particles],d)],jit_compile=jit_compile,autograph=False)

    def _design(self,x):
        cross=tf.stack([x[...,i]*x[...,j] for i,j in self.pairs],axis=-1)
        return tf.concat([tf.ones_like(x[...,:1]),x,cross,x*x],axis=-1)

    def _regress(self,x,target):
        center=tf.reduce_mean(x,axis=-2,keepdims=True)
        scale=tf.sqrt(tf.reduce_mean((x-center)**2,axis=-2,keepdims=True))
        values=(x-center)/scale if self.solver=='centered_qr' else x
        design=self._design(values)
        gram=tf.linalg.matmul(design,design,transpose_a=True)
        if self.solver=='centered_qr':
            q,r=tf.linalg.qr(design,full_matrices=False)
            inverse_r=tf.linalg.triangular_solve(r,tf.eye(self.width,batch_shape=[self.b],dtype=self.dtype),lower=False)
            qr_rcond=1/(tf.sqrt(tf.reduce_sum(r*r,axis=[-2,-1]))*
                        tf.sqrt(tf.reduce_sum(inverse_r*inverse_r,axis=[-2,-1])))
            response_center=tf.reduce_mean(target,-1,keepdims=True)
            beta=tf.squeeze(tf.linalg.triangular_solve(r,tf.linalg.matmul(q,(target-response_center)[...,None],transpose_a=True),lower=False),-1)
            beta=tf.concat([beta[:,:1]+response_center,beta[:,1:]],axis=-1)
        else:
            beta=tf.squeeze(tf.linalg.solve(gram,tf.linalg.matmul(design,target[...,None],transpose_a=True)),-1)
        a=tf.einsum('bk,kij->bij',beta,self.basis)
        b=beta[:,1:1+self.d]; c=beta[:,0]
        centered_coefficients=(a,b,c)
        if self.solver=='centered_qr':
            mu,s=center[:,0],scale[:,0]
            a=a/(s[:,:,None]*s[:,None,:])
            linear=b/s
            b=linear-2*tf.einsum('bi,bij->bj',mu,a)
            c=c-tf.reduce_sum(linear*mu,-1)+tf.einsum('bi,bij,bj->b',mu,a,mu)
        residual=quadratic(x,a,b,c)-target
        eigenvalues=tf.linalg.eigvalsh(gram)
        ratio=eigenvalues[:,0]/eigenvalues[:,-1]
        valid=tf.reduce_all(tf.math.is_finite(beta)) & tf.reduce_all(tf.math.is_finite(residual))
        if self.solver=='centered_qr':
            valid &= tf.reduce_all(tf.math.is_finite(qr_rcond)) & tf.reduce_all(
                qr_rcond>(64*2.220446049250313e-16)**.5) & tf.reduce_all(scale>0)
        result=dict(A=a,b=b,c=c,valid=valid,gram_eigenvalue_ratio=ratio,
                    residual_sum_squares=tf.reduce_sum(residual**2,-1),max_residual=tf.reduce_max(tf.abs(residual),-1))
        if self.solver=='centered_qr':
            result.update(qr_reciprocal_condition_lower_bound=qr_rcond,center=center[:,0],scale=scale[:,0],centered_A=centered_coefficients[0],
                          centered_b=centered_coefficients[1],centered_c=centered_coefficients[2])
        return result

    def _cloud_weights(self,a,b,c,y,x):
        next_mean=self.f(x[:-1])
        _,_,lookahead,margin,valid=multivariate_twist(next_mean,self.p,self.ldq,a[1:],b[1:],c[1:])
        lookahead=tf.concat([lookahead,tf.zeros([1,self.b,self.n],self.dtype)],axis=0)
        potential=self.logg(y[:,:,None,:],x)+lookahead+quadratic(x,a,b,c)
        _,_,initial,initial_margin,initial_valid=multivariate_twist(
            tf.broadcast_to(self.m0,[self.b,1,self.d]),self.p0,self.ld0,a[0],b[0],c[0])
        logz=tf.cumsum(tf.reduce_logsumexp(potential,-1)-tf.math.log(tf.cast(self.n,self.dtype)),axis=0)+tf.transpose(initial)
        weights=tf.nn.softmax(potential,-1)
        return dict(log_incremental_weight=potential,logZ=logz,ess=1/tf.reduce_sum(weights**2,-1),
            precision_margin=tf.concat([initial_margin[None,:],margin],axis=0),
            valid=tf.reduce_all(valid)&tf.reduce_all(initial_valid)&tf.reduce_all(tf.math.is_finite(potential)))

    # Optional fail-fast candidate. Full-scope coefficient no-fire checks failed
    # in132/133; keep the unchanged numerical kernels as the default until repaired.
    def _forward_guarded(self,a,b,c,y,normal,uniform):
        """Fail at the first invalid operation, before normalization/resampling.

        Codes: 1 transition, 2 proposal Gaussian, 3 particle draw,
        4 lookahead transition, 5 lookahead Gaussian, 6 log weight,
        7 resampling uniform. A failure rejects the whole batch; row codes
        identify the rows that actually failed. Unexecuted fields are NaN.
        """
        nan=tf.constant(float('nan'),self.dtype)
        clouds=tf.TensorArray(self.dtype,size=self.h,element_shape=[self.b,self.n,self.d]).unstack(
            tf.fill([self.h,self.b,self.n,self.d],nan))
        ancestors=tf.TensorArray(tf.int32,size=self.h,element_shape=[self.b,self.n]).unstack(
            tf.fill([self.h,self.b,self.n],tf.constant(-1,tf.int32)))
        zero_x=tf.zeros([self.b,self.n,self.d],self.dtype)
        zero_w=tf.zeros([self.b,self.n],self.dtype)
        zero_codes=tf.zeros([self.b],tf.int32)
        identity=tf.eye(self.d,batch_shape=[self.b],dtype=self.dtype)
        finite=lambda value:tf.reduce_all(tf.math.is_finite(value),axis=list(range(1,value.shape.rank)))
        def mark(codes,good,code):
            # A batch-wide stop leaves later operations unexecuted. Their
            # placeholder values must not label healthy rows as failures.
            return tf.cond(tf.reduce_all(codes==0),
                lambda:tf.where(~good,tf.fill([self.b],code),codes),lambda:codes)
        def step(t,previous,clouds,ancestors,valid,codes,failed_mean,failed_cov,failed_margin):
            mean=tf.cond(t==0,lambda:tf.broadcast_to(self.m0,[self.b,self.n,self.d]),lambda:self.f(previous))
            codes=mark(zero_codes,finite(mean),1)
            precision=tf.where(t==0,self.p0,self.p);ld=tf.where(t==0,self.ld0,self.ldq)
            cm,cov,integral,margin,good=tf.cond(tf.reduce_all(codes==0),
                lambda:multivariate_twist(mean,precision,ld,a[t],b[t],c[t]),
                lambda:(zero_x,identity,zero_w,tf.zeros([self.b],self.dtype),tf.zeros([self.b],tf.bool)))
            codes=mark(codes,good&finite(cm)&finite(cov)&finite(integral),2)
            x=tf.cond(tf.reduce_all(codes==0),
                lambda:cm+tf.linalg.matmul(normal[t],tf.linalg.cholesky(cov),transpose_b=True),
                lambda:tf.fill([self.b,self.n,self.d],nan))
            codes=mark(codes,finite(x),3)
            def lookahead_step():
                next_mean=self.f(x)
                next_good=finite(next_mean)
                result=tf.cond(tf.reduce_all(next_good),
                    lambda:multivariate_twist(next_mean,self.p,self.ldq,a[t+1],b[t+1],c[t+1]),
                    lambda:(zero_x,identity,zero_w,tf.zeros([self.b],self.dtype),tf.zeros([self.b],tf.bool)))
                local=mark(codes,next_good,4)
                return result[2],mark(local,result[4]&finite(result[2]),5)
            lookahead,codes=tf.cond((t<self.h-1)&tf.reduce_all(codes==0),
                lookahead_step,lambda:(zero_w,codes))
            potential=tf.cond(tf.reduce_all(codes==0),
                lambda:self.logg(y[t,:,None,:],x)+lookahead+quadratic(x,a[t],b[t],c[t]),
                lambda:tf.fill([self.b,self.n],nan))
            codes=mark(codes,finite(potential),6)
            codes=mark(codes,(uniform[t]>=0)&(uniform[t]<1),7)
            valid=tf.reduce_all(codes==0)
            indices=tf.cond(valid,lambda:systematic_ancestors(tf.nn.softmax(potential,-1),uniform[t]),
                lambda:tf.fill([self.b,self.n],tf.constant(-1,tf.int32)))
            following=tf.cond(valid,lambda:tf.gather(x,indices,batch_dims=1),lambda:previous)
            return (t+1,following,clouds.write(t,x),ancestors.write(t,indices),valid,codes,
                    cm,cov,margin)
        t,_,clouds,ancestors,valid,codes,cm,cov,margin=tf.while_loop(
            lambda t,p,x,anc,v,*_: (t<self.h)&v,step,
            (tf.constant(0),zero_x,clouds,ancestors,tf.constant(True),zero_codes,
             zero_x,identity,tf.zeros([self.b],self.dtype)),parallel_iterations=1)
        x=clouds.stack()
        result=tf.cond(valid,lambda:self._cloud_weights(a,b,c,y,x),lambda:dict(
            log_incremental_weight=tf.fill([self.h,self.b,self.n],nan),
            logZ=tf.fill([self.h,self.b],nan),ess=tf.fill([self.h,self.b],nan),
            precision_margin=tf.fill([self.h,self.b],nan),valid=tf.constant(False)))
        result.update(x=x,ancestors=ancestors.stack(),first_failure_time=tf.where(valid,-1,t-1),
            failure_codes=codes,completed_steps=t-tf.cast(~valid,tf.int32),
            failure_mean=cm,failure_covariance=cov,failure_precision_margin=margin)
        return result

    def _empty_fit(self,a,b,c):
        shapes=([self.b,self.d,self.d],[self.b,self.d],[self.b],[self.b],[self.b],[self.b],
             [self.b],[self.b],[self.b],[self.b],[self.b],[self.b],[self.b],[self.b],
             [self.b,self.n],[self.b,self.d],[self.b,self.d],[self.b,self.d,self.d],
             [self.b,self.d],[self.b],[self.b,self.d,self.d])
        keys=('A','b','c','precision_margin','gram_eigenvalue_ratio','residual_sum_squares',
              'protection_applied','protection_fraction','raw_precision_margin','relative_precision_change',
              'qr_reciprocal_condition_lower_bound','eigen_residual','eigen_orthogonality','eigen_rotations',
              'correction_log_integral','increment_center','increment_scale','increment_centered_A',
              'increment_centered_b','increment_centered_c','projection_added_A')
        result={key:tf.zeros([self.h]+list(shape),self.dtype) for key,shape in zip(keys,shapes)}
        result.update(A=a,b=b,c=c,valid=tf.constant(True),first_failure_time=tf.constant(-1),
                      failure_code=tf.constant(0),failure_rows=tf.zeros([self.b],tf.bool),
                      attempted_times=tf.zeros([self.h],tf.bool))
        return result

    def _fit_guarded(self,a,b,c,x,logweights):
        empty=self._empty_fit(a,b,c)
        arrays=[tf.TensorArray(self.dtype,size=self.h,element_shape=shape).unstack(empty[key]) for key,shape in zip(list(empty)[:21],
            ([self.b,self.d,self.d],[self.b,self.d],[self.b],[self.b],[self.b],[self.b],
             [self.b],[self.b],[self.b],[self.b],[self.b],[self.b],[self.b],[self.b],
             [self.b,self.n],[self.b,self.d],[self.b,self.d],[self.b,self.d,self.d],
             [self.b,self.d],[self.b],[self.b,self.d,self.d]))]
        def step(t,na,nb,nc,increment,valid,arrays,code,rows):
            def correction():
                mean=self.f(x[t])
                if self.solver=='centered_qr':
                    integral=(projected_correction_integral if self.precision_stable_integral
                              else centered_correction_integral)
                    return integral(mean,self.p,self.ldq,a[t+1],b[t+1],c[t+1],*increment)
                value=(multivariate_twist(mean,self.p,self.ldq,na,nb,nc)[2]
                       -multivariate_twist(mean,self.p,self.ldq,a[t+1],b[t+1],c[t+1])[2])
                return value,tf.constant(True)
            delta,delta_valid=tf.cond(t<self.h-1,correction,
                lambda:(tf.zeros([self.b,self.n],self.dtype),tf.constant(True)))
            target=-logweights[t]-delta
            input_rows=tf.reduce_all(tf.math.is_finite(target),axis=-1)&tf.reduce_all(tf.math.is_finite(x[t]),axis=[-2,-1])
            delta_valid &= tf.reduce_all(input_rows)
            # Invalid integral targets never reach QR or a matrix projection.
            def rejected_fit():
                result=dict(A=tf.zeros_like(a[t]),b=tf.zeros_like(b[t]),c=tf.zeros_like(c[t]),
                    valid=tf.constant(False),gram_eigenvalue_ratio=tf.zeros_like(c[t]),
                    residual_sum_squares=tf.zeros_like(c[t]),max_residual=tf.zeros_like(c[t]))
                if self.solver=='centered_qr':
                    result.update(qr_reciprocal_condition_lower_bound=tf.zeros_like(c[t]),
                        center=tf.zeros_like(b[t]),scale=tf.ones_like(b[t]),
                        centered_A=tf.zeros_like(a[t]),centered_b=tf.zeros_like(b[t]),centered_c=tf.zeros_like(c[t]))
                return result
            fit=tf.cond(delta_valid,lambda:self._regress(x[t],target),rejected_fit)
            code=tf.where(~delta_valid,9,tf.where(~fit['valid'],10,0))
            # Only attribute QR failures when QR actually executed. A bad
            # target in one row rejects the batch before any row reaches QR.
            rows=~input_rows
            fit_rows=~tf.reduce_all(tf.math.is_finite(fit['A']),[-2,-1])|~tf.reduce_all(tf.math.is_finite(fit['b']),-1)|~tf.math.is_finite(fit['c'])
            if self.solver=='centered_qr':
                fit_rows |= ~tf.math.is_finite(fit['qr_reciprocal_condition_lower_bound'])|(fit['qr_reciprocal_condition_lower_bound']<=(64*2.220446049250313e-16)**.5)
            rows |= delta_valid&fit_rows
            if self.solver=='centered_qr':
                unprotected=tuple(fit[k] for k in ('center','scale','centered_A','centered_b','centered_c'))
            base=tf.where(t==0,self.p0,self.p)
            safety=dict(protection_applied=tf.zeros([self.b],self.dtype),
                protection_fraction=tf.ones([self.b],self.dtype),
                relative_precision_change=tf.zeros([self.b],self.dtype),
                raw_precision_margin=tf.zeros([self.b],self.dtype),
                eigen_residual=tf.zeros([self.b],self.dtype),eigen_orthogonality=tf.zeros([self.b],self.dtype),eigen_rotations=tf.zeros([self.b],self.dtype))
            if self.precision_policy!='none':
                fit['projection_added_A']=tf.zeros_like(a[t])
                fit,safety=tf.cond(fit['valid'],lambda:constrain_precision_increment(base,a[t],fit,self.precision_policy,
                    self.precision_floor,self.precision_eigen_tolerance,self.precision_eigen_refinement,self.precision_activation_margin),
                    lambda:(fit,safety))
                residual=quadratic(x[t],fit['A'],fit['b'],fit['c'])-target
                fit['residual_sum_squares']=tf.reduce_sum(residual**2,-1)
                fit['valid'] &= tf.reduce_all(tf.math.is_finite(residual))
            na,nb,nc=a[t]+fit['A'],b[t]+fit['b'],c[t]+fit['c']
            if self.solver=='centered_qr':
                increment=tuple(fit[k] for k in ('center','scale','centered_A','centered_b','centered_c'))
                if self.precision_stable_integral:
                    increment=(*unprotected,fit['projection_added_A'])
            _,_,margin,good=covariance_from_precision(tf.where(t==0,self.p0,self.p)+2*na)
            valid &= fit['valid'] & tf.reduce_all(good) & delta_valid
            rows |= ~good
            code=tf.where((code==0)&~valid,11,code)
            if self.precision_policy=='none':safety['raw_precision_margin']=margin
            values=(na,nb,nc,margin,fit['gram_eigenvalue_ratio'],fit['residual_sum_squares'],
                safety['protection_applied'],safety['protection_fraction'],
                safety['raw_precision_margin'],safety['relative_precision_change'],
                fit.get('qr_reciprocal_condition_lower_bound',tf.zeros([self.b],self.dtype)),
                safety['eigen_residual'],safety['eigen_orthogonality'],safety['eigen_rotations'],
                delta,*(unprotected if self.solver=='centered_qr' else
                        (tf.zeros_like(nb),tf.ones_like(nb),tf.zeros_like(na),tf.zeros_like(nb),tf.zeros_like(nc))),
                fit.get('projection_added_A',tf.zeros_like(na)))
            arrays=[arr.write(t,val) for arr,val in zip(arrays,values)]
            return t-1,na,nb,nc,increment,valid,arrays,code,rows
        initial_increment=(tf.zeros_like(b[0]),tf.ones_like(b[0]),
                           tf.zeros_like(a[0]),tf.zeros_like(b[0]),tf.zeros_like(c[0]))
        if self.precision_stable_integral:initial_increment=(*initial_increment,tf.zeros_like(a[0]))
        last,_,_,_,_,valid,arrays,code,rows=tf.while_loop(lambda t,na,nb,nc,inc,valid,*_: (t>=0)&valid,step,
            (tf.constant(self.h-1),tf.zeros_like(a[0]),tf.zeros_like(b[0]),tf.zeros_like(c[0]),
             initial_increment,tf.constant(True),arrays,tf.constant(0),tf.zeros([self.b],tf.bool)),parallel_iterations=1)
        result=dict(zip(('A','b','c','precision_margin','gram_eigenvalue_ratio','residual_sum_squares',
                         'protection_applied','protection_fraction','raw_precision_margin','relative_precision_change','qr_reciprocal_condition_lower_bound','eigen_residual','eigen_orthogonality','eigen_rotations','correction_log_integral','increment_center','increment_scale','increment_centered_A','increment_centered_b','increment_centered_c','projection_added_A'),(arr.stack() for arr in arrays)))
        result.update(valid=valid,first_failure_time=tf.where(valid,-1,last+1),failure_code=code,
            failure_rows=rows,attempted_times=tf.range(self.h)>last)
        return result

    def _forward(self,a,b,c,y,normal,uniform):
        clouds=tf.TensorArray(self.dtype,size=self.h,element_shape=[self.b,self.n,self.d])
        ancestors=tf.TensorArray(tf.int32,size=self.h,element_shape=[self.b,self.n])
        def step(t,previous,clouds,ancestors):
            mean=tf.cond(t==0,lambda:tf.broadcast_to(self.m0,[self.b,self.n,self.d]),lambda:self.f(previous))
            precision=tf.where(t==0,self.p0,self.p);ld=tf.where(t==0,self.ld0,self.ldq)
            cm,cov,_,_,_=multivariate_twist(mean,precision,ld,a[t],b[t],c[t])
            x=cm+tf.linalg.matmul(normal[t],tf.linalg.cholesky(cov),transpose_b=True)
            lookahead=tf.cond(t<self.h-1,lambda:multivariate_twist(self.f(x),self.p,self.ldq,a[t+1],b[t+1],c[t+1])[2],lambda:tf.zeros([self.b,self.n],self.dtype))
            potential=self.logg(y[t,:,None,:],x)+lookahead+quadratic(x,a[t],b[t],c[t])
            indices=systematic_ancestors(tf.nn.softmax(potential,-1),uniform[t])
            return t+1,tf.gather(x,indices,batch_dims=1),clouds.write(t,x),ancestors.write(t,indices)
        _,_,clouds,ancestors=tf.while_loop(lambda t,*_:t<self.h,step,
            (tf.constant(0),tf.zeros([self.b,self.n,self.d],self.dtype),clouds,ancestors),parallel_iterations=1)
        x=clouds.stack();result=self._cloud_weights(a,b,c,y,x)
        result.update(x=x,ancestors=ancestors.stack())
        result['valid'] &= tf.reduce_all((uniform>=0)&(uniform<1)) & tf.reduce_all(tf.math.is_finite(x))
        return result

    def _fit(self,a,b,c,x,logweights):
        arrays=[tf.TensorArray(self.dtype,size=self.h,element_shape=shape) for shape in
            ([self.b,self.d,self.d],[self.b,self.d],[self.b],[self.b],[self.b],[self.b],
             [self.b],[self.b],[self.b],[self.b],[self.b],[self.b],[self.b],[self.b],
             [self.b,self.n],[self.b,self.d],[self.b,self.d],[self.b,self.d,self.d],
             [self.b,self.d],[self.b],[self.b,self.d,self.d])]
        def step(t,na,nb,nc,increment,valid,arrays):
            def correction():
                mean=self.f(x[t])
                if self.solver=='centered_qr':
                    integral=(projected_correction_integral if self.precision_stable_integral
                              else centered_correction_integral)
                    return integral(mean,self.p,self.ldq,a[t+1],b[t+1],c[t+1],*increment)
                value=(multivariate_twist(mean,self.p,self.ldq,na,nb,nc)[2]
                       -multivariate_twist(mean,self.p,self.ldq,a[t+1],b[t+1],c[t+1])[2])
                return value,tf.constant(True)
            delta,delta_valid=tf.cond(t<self.h-1,correction,
                lambda:(tf.zeros([self.b,self.n],self.dtype),tf.constant(True)))
            target=-logweights[t]-delta
            fit=self._regress(x[t],target)
            if self.solver=='centered_qr':
                unprotected=tuple(fit[k] for k in ('center','scale','centered_A','centered_b','centered_c'))
            base=tf.where(t==0,self.p0,self.p)
            safety=dict(protection_applied=tf.zeros([self.b],self.dtype),
                protection_fraction=tf.ones([self.b],self.dtype),
                relative_precision_change=tf.zeros([self.b],self.dtype),
                eigen_residual=tf.zeros([self.b],self.dtype),eigen_orthogonality=tf.zeros([self.b],self.dtype),eigen_rotations=tf.zeros([self.b],self.dtype))
            if self.precision_policy!='none':
                fit,safety=constrain_precision_increment(base,a[t],fit,self.precision_policy,
                                                        self.precision_floor,self.precision_eigen_tolerance,self.precision_eigen_refinement,self.precision_activation_margin)
                residual=quadratic(x[t],fit['A'],fit['b'],fit['c'])-target
                fit['residual_sum_squares']=tf.reduce_sum(residual**2,-1)
                fit['valid'] &= tf.reduce_all(tf.math.is_finite(residual))
            na,nb,nc=a[t]+fit['A'],b[t]+fit['b'],c[t]+fit['c']
            if self.solver=='centered_qr':
                increment=tuple(fit[k] for k in ('center','scale','centered_A','centered_b','centered_c'))
                if self.precision_stable_integral:
                    increment=(*unprotected,fit['projection_added_A'])
            _,_,margin,good=covariance_from_precision(tf.where(t==0,self.p0,self.p)+2*na)
            valid &= fit['valid'] & tf.reduce_all(good) & delta_valid
            if self.precision_policy=='none':safety['raw_precision_margin']=margin
            values=(na,nb,nc,margin,fit['gram_eigenvalue_ratio'],fit['residual_sum_squares'],
                safety['protection_applied'],safety['protection_fraction'],
                safety['raw_precision_margin'],safety['relative_precision_change'],
                fit.get('qr_reciprocal_condition_lower_bound',tf.zeros([self.b],self.dtype)),
                safety['eigen_residual'],safety['eigen_orthogonality'],safety['eigen_rotations'],
                delta,*(unprotected if self.solver=='centered_qr' else
                        (tf.zeros_like(nb),tf.ones_like(nb),tf.zeros_like(na),tf.zeros_like(nb),tf.zeros_like(nc))),
                fit.get('projection_added_A',tf.zeros_like(na)))
            arrays=[arr.write(t,val) for arr,val in zip(arrays,values)]
            return t-1,na,nb,nc,increment,valid,arrays
        initial_increment=(tf.zeros_like(b[0]),tf.ones_like(b[0]),
                           tf.zeros_like(a[0]),tf.zeros_like(b[0]),tf.zeros_like(c[0]))
        if self.precision_stable_integral:initial_increment=(*initial_increment,tf.zeros_like(a[0]))
        _,_,_,_,_,valid,arrays=tf.while_loop(lambda t,*_:t>=0,step,
            (tf.constant(self.h-1),tf.zeros_like(a[0]),tf.zeros_like(b[0]),tf.zeros_like(c[0]),
             initial_increment,tf.constant(True),arrays),parallel_iterations=1)
        result=dict(zip(('A','b','c','precision_margin','gram_eigenvalue_ratio','residual_sum_squares',
                         'protection_applied','protection_fraction','raw_precision_margin','relative_precision_change','qr_reciprocal_condition_lower_bound','eigen_residual','eigen_orthogonality','eigen_rotations','correction_log_integral','increment_center','increment_scale','increment_centered_A','increment_centered_b','increment_centered_c','projection_added_A'),(arr.stack() for arr in arrays)))
        result['valid']=valid
        return result
