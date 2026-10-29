"""Finite Eq.15 reconstruction, not the unrecovered original-author recipe.

TensorFlow implementation of the frozen R 4.1 optim/BFGS procedure. The
algorithmic specification is R src/appl/optim.c::vmmin (R Core, GPL >= 2);
this implementation is distributed under GPL >= 2. It preserves finite
stopping/reset semantics, not bitwise equivalence across BLAS/XLA backends.
No score or differentiation through adaptive fitting is claimed.
"""
from functools import lru_cache
import tensorflow as tf
from .iapf_paper_tf import _log_normal, _add_floor


def _initial(points, targets):
    dtype = points.dtype
    d = points.shape[-1]
    center = tf.reduce_mean(points, axis=0)
    scale = tf.sqrt(tf.reduce_mean(tf.square(points-center), axis=0))
    standardized = (points-center)/scale
    design = tf.concat([tf.ones([points.shape[0], 1], dtype), standardized,
                        tf.square(standardized)], axis=1)
    q, r = tf.linalg.qr(design, full_matrices=False)
    diag = tf.abs(tf.linalg.diag_part(r))
    # Restricted full-rank domain; no claim of reproducing pivoted QR at rank boundaries.
    margin = tf.reduce_min(diag/tf.sqrt(tf.reduce_sum(tf.square(design), axis=0)))
    full_rank = margin > tf.constant(1e-7, dtype)
    coefficients = tf.linalg.triangular_solve(
        r, tf.matmul(q, (targets-tf.reduce_max(targets))[:, None], transpose_a=True),
        lower=False)[:, 0]
    linear, quadratic = coefficients[1:1+d], coefficients[1+d:]
    use_quadratic = tf.logical_and(full_rank, tf.reduce_all(
        quadratic < -tf.constant(1.4901161193847656e-8, dtype)))
    w = tf.nn.softmax(targets)
    weighted_mean = tf.reduce_sum(points*w[:, None], axis=0)
    weighted_var = tf.reduce_sum(tf.square(points-weighted_mean)*w[:, None], axis=0)
    mean, variance = tf.cond(use_quadratic,
        lambda: (center-scale*linear/(2*quadratic), -tf.square(scale)/(2*quadratic)),
        lambda: (weighted_mean, weighted_var))
    theta = tf.concat([mean, tf.math.log(variance)], axis=0)
    valid = tf.logical_and(full_rank, tf.logical_and(tf.reduce_all(scale > 0),
        tf.logical_and(tf.reduce_all(variance > 0), tf.reduce_all(tf.math.is_finite(theta)))))
    return theta, tf.cast(use_quadratic, tf.int32), margin, valid


def _problem(points, targets, theta0, scaled):
    dtype, d = points.dtype, points.shape[-1]
    units = tf.concat([tf.exp(theta0[d:]/2), tf.ones([d], dtype)], axis=0)
    def log_density(theta):
        return -.5*(tf.cast(d, dtype)*tf.math.log(tf.constant(6.283185307179586, dtype))+
            tf.reduce_sum(theta[d:])+tf.reduce_sum(
                tf.square(points-theta[:d])/tf.exp(theta[d:]), axis=1))
    b = tf.exp(targets-tf.reduce_max(targets))
    offset = tf.where(scaled, tf.reduce_max(log_density(theta0)), tf.zeros([], dtype))
    p0 = tf.exp(log_density(theta0)-offset)
    a0 = tf.reduce_sum(p0*b)/tf.reduce_sum(tf.square(b))
    def evaluate(z):
        theta = theta0 + units*z[:2*d]
        variance = tf.exp(theta[d:])
        delta = points-theta[:d]
        p = tf.exp(log_density(theta)-offset)
        amplitude = a0*(1+z[-1])
        residual = p-amplitude*b
        score = tf.concat([delta/variance, .5*(tf.square(delta)/variance-1)], axis=1)
        jac = tf.concat([score*units[None, :]*p[:, None], (-a0*b)[:, None]], axis=1)
        gradient = 2*tf.reduce_sum(jac*residual[:, None], axis=0)
        loss = tf.reduce_sum(tf.square(residual))
        valid = tf.logical_and(tf.reduce_all(tf.math.is_finite(theta)),
            tf.logical_and(tf.reduce_all(tf.math.is_finite(variance)),
            tf.logical_and(tf.reduce_all(variance > 0),
            tf.logical_and(tf.math.is_finite(loss),tf.reduce_all(tf.math.is_finite(gradient))))))
        return loss, gradient, theta, amplitude, valid
    return evaluate, offset, a0


def _vmmin(evaluate, z, maxit=100, initial_valid=None):
    """Shared finite R-vmmin reconstruction for the declared full-mask scope.

    No damping or objective changes. Trial function nonfiniteness backtracks;
    accepted invalid derivatives and explicit iteration/line caps fail closed.
    Diagnostics do not establish convergence or original iAPF recipe identity.
    """
    if maxit < 2:
        raise ValueError("The reconstruction requires maxit >= 2")
    dtype, p = z.dtype, z.shape[0]
    def value_gradient_valid(q):
        ev = evaluate(q)
        return ev[0], ev[1], ev[4]
    f, g, ok = value_gradient_valid(z)
    if initial_valid is not None:
        ok = tf.logical_and(ok, initial_valid)
    eye = tf.eye(p, dtype=dtype)
    tolerance = tf.constant(1.4901161193847656e-8, dtype)
    # R may restart without incrementing iter. This extra cap only rejects a stuck solve.
    outer_cap = 4*maxit+10
    trace = tf.zeros([outer_cap, p+9], dtype)

    def body(outer, z, f, g, B, iteration, nfev, ngev, last, done, healthy, trace):
        reset = last == ngev
        B = tf.cond(reset, lambda: eye, lambda: B)
        direction = -tf.linalg.matvec(B, g)
        projection = tf.reduce_sum(direction*g)
        def downhill():
            def ls_body(j, step, trial, value, grad, valid, accepted, same, calls):
                candidate = z+step*direction
                unchanged = tf.reduce_all(tf.constant(10., dtype)+z ==
                                          tf.constant(10., dtype)+candidate)
                ev = tf.cond(unchanged, lambda: (value, grad, valid),
                             lambda: value_gradient_valid(candidate))
                accept = tf.logical_and(tf.math.is_finite(ev[0]), ev[0] <= f+
                    projection*step*tf.constant(1e-4, dtype))
                return (j+1, tf.where(tf.logical_or(accept, unchanged), step,
                                     step*tf.constant(.2, dtype)),
                        candidate, ev[0], ev[1], ev[2], accept, unchanged,
                        calls+tf.cast(tf.logical_not(unchanged), tf.int32))
            ls = tf.while_loop(
                lambda j, step, trial, value, grad, valid, accepted, same, calls:
                    tf.logical_and(j < 128,
                        tf.logical_not(tf.logical_or(accepted, same))),
                ls_body, (tf.constant(0), tf.ones([], dtype), z, f, g, healthy,
                          tf.constant(False), tf.constant(False), tf.constant(0)),
                parallel_iterations=1)
            _, step, candidate, value, grad, valid, accepted, same, calls = ls
            # Nonfinite trial functions backtrack; accepted invalid gradients
            # still fail closed, including the final reevaluation.
            valid = tf.logical_and(tf.logical_or(same, valid), tf.logical_or(accepted, same))
            curvature = tf.reduce_sum((step*direction)*(grad-g))
            enough = tf.abs(value-f) > tolerance*(tf.abs(f)+tolerance)
            progress = tf.logical_and(valid, tf.logical_and(tf.logical_not(same), enough))
            def update():
                s = step*direction
                y = grad-g
                def positive():
                    By = tf.linalg.matvec(B, y)
                    factor = 1+tf.reduce_sum(y*By)/curvature
                    update_B = B+(factor*s[:, None]*s[None, :]
                        -By[:, None]*s[None, :]-s[:, None]*By[None, :])/curvature
                    # R stores a lower triangle and reads it symmetrically.
                    update_B = tf.where(tf.range(p)[:,None] >= tf.range(p)[None,:],
                                        update_B, tf.transpose(update_B))
                    return update_B, last
                newB, newlast = tf.cond(curvature > 0, positive, lambda: (B, ngev+1))
                return candidate, value, grad, newB, iteration+1, ngev+1, newlast, tf.constant(False)
            def no_progress():
                restart = last < ngev
                return (candidate, tf.where(enough,f,value), g, B, iteration, ngev, ngev,
                        tf.logical_not(restart))
            newz, newf, newg, newB, newit, newng, newlast, newdone = tf.cond(progress, update, no_progress)
            return newz,newf,newg,newB,newit,nfev+calls,newng,newlast,newdone,valid,step,curvature,calls-tf.cast(tf.logical_and(accepted,tf.logical_not(same)),tf.int32)
        def uphill():
            return z,f,g,B,iteration,nfev,ngev,ngev,last==ngev,healthy,tf.zeros([],dtype),tf.zeros([],dtype),tf.constant(0)
        z,f,g,B,iteration,nfev,ngev,last,done,healthy,step,curvature,rejected = tf.cond(projection < 0, downhill, uphill)
        last = tf.where(ngev-last > 2*p, ngev, last)
        healthy = tf.logical_and(healthy, tf.reduce_all(tf.math.is_finite(B)))
        row = tf.concat([z, tf.stack([f,tf.reduce_max(tf.abs(g)),step,
                        tf.cast(iteration,dtype),tf.cast(healthy,dtype),
                        tf.cast(rejected,dtype),tf.cast(reset,dtype),curvature,
                        tf.reduce_max(tf.abs(B-tf.transpose(B)))])], axis=0)
        trace = tf.tensor_scatter_nd_update(trace, [[outer]], row[None,:])
        return outer+1,z,f,g,B,iteration,nfev,ngev,last,done,healthy,trace

    result = tf.while_loop(
        lambda outer,z,f,g,B,it,nf,ng,last,done,healthy,trace:
            tf.logical_and(outer < outer_cap, tf.logical_and(it < maxit,
                tf.logical_and(healthy, tf.logical_not(done)))),
        body, (tf.constant(0),z,f,g,eye,tf.constant(1),tf.constant(1),
               tf.constant(1),tf.constant(1),tf.constant(False),ok,trace),
        parallel_iterations=1)
    outer,z,algorithm_loss,g,B,iteration,nfev,ngev,_,done,healthy,trace = result

    final_f, final_g, _, _, final_valid = evaluate(z)
    valid = tf.logical_and(final_valid, tf.logical_and(healthy, outer < outer_cap))
    return dict(z=z,loss=final_f,gradient=final_g,algorithm_loss=algorithm_loss,
                code=tf.where(valid,tf.cast(iteration >= maxit,tf.int32),tf.constant(-1)),
                valid=valid,iterations=iteration,function_calls=nfev,gradient_calls=ngev,
                trace=trace[:,:p+5],trace_length=outer,
                numerical_trace=trace[:,p+5:],inverse_hessian=B)


def _finite_fit(points, targets, scaled, maxit=100, diagnostic_initial=None):
    dtype, d = points.dtype, points.shape[-1]
    theta0, method, margin, initial_valid = _initial(points, targets)
    if diagnostic_initial is not None:
        theta0 = diagnostic_initial
    evaluate, offset, a0 = _problem(points, targets, theta0, scaled)
    result = _vmmin(evaluate, tf.zeros([2*d+1], dtype), maxit,
                   tf.logical_and(initial_valid,a0 > 0))
    f,g,theta,amplitude,valid = evaluate(result["z"])
    valid = tf.logical_and(valid,tf.logical_and(result["valid"],amplitude > 0))
    result.update(theta=theta,amplitude=amplitude,loss=f,gradient=g,valid=valid,
                  code=tf.where(valid,result["code"],tf.constant(-1)),
                  initial=theta0,initial_method=method,rank_margin=margin,scale=offset)
    return result


@lru_cache(maxsize=16)
def make_diagnostic_fixed_initial_fit_kernel(d,n,dtype_name="float64",jit_compile=True):
    """Diagnostic-only matched-initializer replay; never the filter initializer."""
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([n,d],dtype),tf.TensorSpec([n],dtype),
        tf.TensorSpec([2*d],dtype),tf.TensorSpec([],tf.bool)],jit_compile=jit_compile)
    def kernel(points,targets,initial,scaled):
        return _finite_fit(points,targets,scaled,diagnostic_initial=initial)
    return kernel


@lru_cache(maxsize=16)
def make_finite_fit_kernel(d, n, dtype_name="float64", jit_compile=True):
    dtype = tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([n,d],dtype),
        tf.TensorSpec([n],dtype),tf.TensorSpec([],tf.bool)],jit_compile=jit_compile)
    def kernel(points,targets,scaled):
        return _finite_fit(points,targets,scaled)
    return kernel


@lru_cache(maxsize=16)
def make_finite_objective_kernel(d,n,dtype_name="float64",jit_compile=True):
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([n,d],dtype),tf.TensorSpec([n],dtype),
        tf.TensorSpec([2*d],dtype),tf.TensorSpec([2*d+1],dtype),
        tf.TensorSpec([],tf.bool)],jit_compile=jit_compile)
    def kernel(points,targets,initial,z,scaled):
        return _problem(points,targets,initial,scaled)[0](z)
    return kernel


@lru_cache(maxsize=16)
def make_finite_backward_kernel(d,o,n,horizon,dtype_name="float64",jit_compile=True):
    dtype=tf.as_dtype(dtype_name)
    spec=lambda shape:tf.TensorSpec(shape,dtype)
    @tf.function(input_signature=[spec([d,d]),spec([o,d]),spec([d,d]),spec([o,o]),
        spec([horizon,o]),spec([horizon,n,d]),tf.TensorSpec([],tf.bool)],
        jit_compile=jit_compile)
    def kernel(A,C,Q,R,observations,clouds,scaled):
        centers=tf.zeros([horizon,d],dtype)
        covs=tf.zeros([horizon,d,d],dtype)
        floors=tf.zeros([horizon],dtype)
        codes=tf.fill([horizon],tf.constant(-2,tf.int32))
        summary=tf.zeros([horizon,8],dtype)
        # Full last fit is returned on failure; arrays preserve already-completed times.
        p=2*d+1
        scalar=tf.zeros([],dtype)
        integer=tf.zeros([],tf.int32)
        empty_fit=dict(theta=tf.zeros([2*d],dtype),amplitude=scalar,loss=scalar,
            gradient=tf.zeros([p],dtype),code=integer,valid=tf.constant(False),
            initial=tf.zeros([2*d],dtype),initial_method=integer,rank_margin=scalar,
            z=tf.zeros([p],dtype),iterations=integer,function_calls=integer,
            gradient_calls=integer,trace=tf.zeros([410,p+5],dtype),
            trace_length=integer,scale=scalar,algorithm_loss=scalar,
            numerical_trace=tf.zeros([410,4],dtype),inverse_hessian=tf.eye(p,dtype=dtype))
        def body(t,centers,covs,floors,codes,summary,healthy,last_fit,last_targets):
            x=clouds[t]
            targets=_log_normal(tf.broadcast_to(observations[t],[n,o]),
                               tf.matmul(x,C,transpose_b=True),R)
            targets+=tf.cond(t+1 < horizon,
                lambda:_add_floor(_log_normal(tf.matmul(x,A,transpose_b=True),
                                    centers[t+1],Q+covs[t+1]),floors[t+1]),
                lambda:tf.zeros([n],dtype))
            fit=_finite_fit(x,targets,scaled)
            theta=fit["theta"]
            covariance=tf.linalg.diag(tf.exp(theta[d:]))
            peak=-.5*tf.cast(d,dtype)*tf.math.log(tf.constant(6.283185307179586,dtype))
            peak-=tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(Q+covariance))))
            delta=tf.constant(1./n,dtype)
            floor=tf.math.log(delta)-tf.math.log1p(-delta)+peak
            healthy=tf.logical_and(fit["valid"],tf.math.is_finite(floor))
            centers=tf.tensor_scatter_nd_update(centers,[[t]],theta[None,:d])
            covs=tf.tensor_scatter_nd_update(covs,[[t]],covariance[None,:,:])
            floors=tf.tensor_scatter_nd_update(floors,[[t]],floor[None])
            codes=tf.tensor_scatter_nd_update(codes,[[t]],fit["code"][None])
            row=tf.stack([fit["loss"],fit["amplitude"],tf.cast(fit["iterations"],dtype),
                tf.cast(fit["function_calls"],dtype),tf.cast(fit["gradient_calls"],dtype),
                tf.reduce_min(tf.exp(theta[d:])),tf.reduce_max(tf.exp(theta[d:])),
                fit["rank_margin"]])
            summary=tf.tensor_scatter_nd_update(summary,[[t]],row[None,:])
            return t-1,centers,covs,floors,codes,summary,healthy,fit,targets
        r=tf.while_loop(lambda t,c,v,f,co,s,h,lf,lt:tf.logical_and(t>=0,h),body,
            (tf.constant(horizon-1),centers,covs,floors,codes,summary,tf.constant(True),
             empty_fit,tf.zeros([n],dtype)),parallel_iterations=1)
        return dict(centers=r[1],covariances=r[2],floors=r[3],codes=r[4],summary=r[5],
                    valid=r[6],last_fit=r[7],last_targets=r[8],last_time=r[0]+1)
    return kernel


@lru_cache(maxsize=16)
def make_controller_decision_kernel(max_iterations, k=5, dtype_name="float64", jit_compile=True):
    """Paper Algorithm 4 decision, with sample SD and the k+1 history window."""
    dtype = tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([max_iterations], dtype),
        tf.TensorSpec([max_iterations], tf.int32), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], dtype)], jit_compile=jit_compile)
    def kernel(history, counts, iteration, tau):
        recent = history[tf.maximum(0, iteration-k):iteration+1]
        likelihood = tf.exp(recent-tf.reduce_max(recent))
        mean = tf.reduce_mean(likelihood)
        cv = tf.cond(tf.size(recent)>1,
            lambda: tf.sqrt(tf.reduce_sum(tf.square(likelihood-mean))/
                            tf.cast(tf.size(recent)-1,dtype))/mean,
            lambda: tf.constant(float("inf"),dtype))
        stop = tf.logical_and(iteration>k,cv<tau)
        double = tf.cond(iteration>=k,
            lambda: tf.logical_and(counts[iteration-k]==counts[iteration],
                                    tf.reduce_any(recent[1:]<recent[:-1])),
            lambda: tf.constant(False))
        return stop, double, cv
    return kernel


def run_finite_iapf(A, C, initial_mean, initial_covariance, Q, R, observations,
                   entropy_provider, *, n0=64, max_particles=512, max_iterations=12,
                   k=5, tau=.5, kappa=.5, scaled=False, jit_compile=True, observer=None):
    """Adaptive reconstruction using the same shared filter and finite fitter.

    Python orchestrates bounded adaptive passes and artifact callbacks. Particle,
    time, BFGS, and controller arithmetic run in stable TensorFlow graphs.
    entropy_provider(call_index, N) returns [T,N,d], [T,N], [T,N] tensors.
    No gradient through the adaptive procedure is claimed.
    """
    from .iapf_paper_tf import make_paper_apf_value_kernel
    if n0 < 2*A.shape[0]+1 or max_iterations <= k+1 or tau <= 0 or k < 1:
        raise ValueError("Invalid adaptive reconstruction scope")
    d, o, horizon = A.shape[0], C.shape[0], observations.shape[0]
    dtype = A.dtype
    n = n0
    history = tf.zeros([max_iterations], dtype)
    counts = tf.zeros([max_iterations], tf.int32)
    centers = tf.zeros([horizon,d],dtype)
    covariances = tf.eye(d,batch_shape=[horizon],dtype=dtype)
    floors = tf.zeros([horizon],dtype)
    constants = tf.ones([horizon],tf.bool)
    calls, fit_calls = 0, 0
    signatures = set()
    decisions = []
    decide = make_controller_decision_kernel(max_iterations,k,dtype.name,jit_compile)
    def observe(kind,index,payload):
        if observer is not None:
            observer(kind,index,payload)
    def filter_once():
        nonlocal calls
        calls += 1
        entropy = entropy_provider(calls,n)
        kernel = make_paper_apf_value_kernel(d,o,n,horizon,dtype.name,jit_compile)
        signatures.add((n,"filter"))
        value = kernel(A,C,initial_mean,initial_covariance,Q,R,observations,
                       centers,covariances,floors,constants,*entropy,tf.constant(kappa,dtype))
        observe("filter",calls,{"particles":n,"value":value})
        return value
    def outcome(status,iteration,final=None):
        return dict(status=status,stop_iteration=iteration,final_particles=n,
                    history=history[:iteration+1],counts=counts[:iteration+1],
                    filter_calls=calls,backward_calls=fit_calls,final=final,
                    decisions=decisions,signatures=sorted(signatures))
    for iteration in range(max_iterations):
        if n>max_particles:
            return outcome("particle_cap",iteration-1)
        value = filter_once()
        if not bool(tf.reduce_all(tf.math.is_finite(value[0])).numpy()):
            return outcome("invalid_filter",iteration)
        history = tf.tensor_scatter_nd_update(history,[[iteration]],value[0][None])
        counts = tf.tensor_scatter_nd_update(counts,[[iteration]],[n])
        stop,double,cv = decide(history,counts,tf.constant(iteration),tf.constant(tau,dtype))
        decisions.append(dict(iteration=iteration,stop=bool(stop.numpy()),
                              double=bool(double.numpy()),cv=float(cv.numpy())))
        if bool(stop.numpy()):
            final = filter_once()
            if not bool(tf.math.is_finite(final[0]).numpy()):
                return outcome("invalid_final",iteration,final)
            return outcome("complete",iteration,final)
        fit = make_finite_backward_kernel(d,o,n,horizon,dtype.name,jit_compile)(
            A,C,Q,R,observations,value[1],tf.constant(scaled))
        fit_calls += 1
        signatures.add((n,"backward"))
        observe("backward",fit_calls,fit)
        if not bool(fit["valid"].numpy()):
            return outcome("invalid_fit",iteration)
        centers,covariances,floors = fit["centers"],fit["covariances"],fit["floors"]
        constants = tf.zeros([horizon],tf.bool)
        if bool(double.numpy()):
            n *= 2
    return outcome("iteration_cap",max_iterations-1)
