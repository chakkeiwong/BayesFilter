"""Native execution of the existing C2 Gaussian-reference TT value program.

The row law, fit objective, defensive floor and frozen maps are unchanged.
This is execution infrastructure; it introduces no source-route claim.
"""

import weakref

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.retained_quadratic_form_tf import prefix_gram_matrix, suffix_gram_matrix
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    _hermite_product_basis, _log_eta, _logdet_lower, TAU_MIN, TAU_MAX,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import DiscreteIndicatorBasis1D, _initial_tt_cores
from bayesfilter.highdim.squared_tt_engine_xla_tf import _fit_als_graph
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import random_core_starts
from bayesfilter.highdim.tt_preparation_tf import frozen_design_rows

D = tf.float64
_CACHE = weakref.WeakKeyDictionary()
REPORT_NAMES = ("log_increment", "tau_t", "eps_rel_sq", "row_ess", "worst_condition",
    "weighted_fit_rms", "u_old_max", "gram_lambda_min", "gram_lambda_max", "gram_cond",
    "gram_sym_err", "eps_rel_sq_counting", "branch_count", "z_h", "corrected_increment",
    "shift", "valid_hint", "z_complete", "tau_abs")


def _floor(rms, z_h):
    # The old floor controller materialized both scalars on the host, so it
    # never differentiated this adaptation. Preserve that derivative boundary.
    relative = tf.stop_gradient(rms*rms / tf.maximum(z_h, tf.constant(1e-300,D)))
    return tf.clip_by_value(relative, TAU_MIN, TAU_MAX), relative


def make_gaussian_value_filter(adapter, observation_shape, config, *, defensive_nu=None,
    full_capture=False, jit_compile=True):
    from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (
        _assemble_transition_target, _transition_target_summary,
    )
    n, horizon = adapter.state_dim, int(observation_shape[0])
    if horizon < 1 or config.quadrature_order is not None:
        raise ValueError("Gaussian TT requires a positive horizon and scattered rows")
    key = (config, tuple(observation_shape), defensive_nu, full_capture, jit_compile)
    cache = _CACHE.setdefault(adapter,{})
    if key in cache:
        return cache[key]
    basis = _hermite_product_basis(n,config.basis_degree)
    degree = basis.bases[0].basis_dim
    fitter, ridge = FixedTTFitter(), tf.constant(config.ridge,D)
    initial = tuple(c.values for c in _initial_tt_cores(n,degree,config.rank))
    initial_shapes = tuple(tuple(v.shape) for v in initial)
    rows0, weights0, ess0, valid0 = tf.nest.map_structure(lambda value: value[0],
        frozen_design_rows(config, config.row_count, n, tf.constant([0]), 17,
            gaussian_degree=config.basis_degree, jit_compile=jit_compile))
    rows, weights, esses, valid = frozen_design_rows(config, config.row_count, 2*n,
        tf.range(1, max(2, horizon)), 100, gaussian_degree=config.basis_degree,
        jit_compile=jit_compile)
    if not bool((valid0 & tf.reduce_all(valid)).numpy()):
        raise ValueError("christoffel rows: endpoint hit or non-finite inverse CDF")
    rank = max(1,config.rank)
    all_core_shape = (2*n+1,rank,max(degree,rank+1),rank)

    def pack_prefix(values):
        return tf.stack(tuple(tf.pad(v,((0,rank-v.shape[0]),(0,degree-v.shape[1]),
            (0,rank-v.shape[2]))) for v in values))

    def pack_gram(gram):
        return tf.pad(gram,((0,rank-gram.shape[0]),(0,rank-gram.shape[1])))

    def phase(previous_shapes,dates):
        branch_count = previous_shapes[-1][-1]+1
        mixed_basis = ProductBasis(list(basis.bases)+[DiscreteIndicatorBasis1D(branch_count)]+
            list(basis.bases),basis.convention)
        dims = (degree,)*n + (branch_count,) + (degree,)*n
        shapes = tuple((1 if axis == 0 else rank,dim,1 if axis == 2*n else rank)
            for axis,dim in enumerate(dims))
        starts = random_core_starts(shapes,dates,config.seed,jit_compile=jit_compile)

        def step(t,state,y,joint_mean,joint_cov,start):
            prefix,gram,tau_abs,zc,m_old,l_old = state
            chol = tf.linalg.cholesky(joint_cov)
            m_c,l_cc = joint_mean[:n],chol[:n,:n]
            target = _assemble_transition_target(adapter=adapter,current_basis=basis,
                prefix_shapes=previous_shapes,branch_gram_floor=config.branch_gram_floor,
                defensive_nu=defensive_nu,prefix_values=prefix,gram=gram,tau_abs_prev=tau_abs,
                u_rows=rows[t-1],u_weights=weights[t-1],y=y,m_c=m_c,l_cc=l_cc,
                m_p=joint_mean[n:],l_pc=chol[n:,:n],l_pp=chol[n:,n:],l_old=l_old,m_old=m_old)
            cores,worst,rms = _fit_als_graph(fitter,mixed_basis,target.expanded_rows,
                target.sqrt_target,target.fit_weights,start,shapes,config.sweeps,ridge)
            new_gram = suffix_gram_matrix(tuple(cores[n:]),mixed_basis,axis_offset=n)
            zh = tf.einsum("ab,ab->",prefix_gram_matrix(tuple(cores[:n]),mixed_basis),new_gram)
            tau,eps = _floor(rms,zh)
            new_zc = (1.+tau)*zh
            increment = target.shift + tf.math.log(new_zc)-tf.math.log(zc)
            corrected = increment - tf.math.log1p(tau)
            eigenvalues = tf.linalg.eigvalsh(new_gram)
            condition = eigenvalues[-1]/tf.maximum(tf.abs(eigenvalues[0]),tf.constant(1e-300,D))
            symmetry = tf.linalg.norm(new_gram-tf.transpose(new_gram))/tf.maximum(tf.linalg.norm(new_gram),tf.constant(1e-300,D))
            valid_hint = (tf.reduce_all(tf.math.is_finite(joint_mean)) & tf.reduce_all(tf.math.is_finite(chol))
                & _retained_valid(new_gram,new_zc,tau))
            report = tf.stack((increment,tau,eps,esses[t-1],worst,rms,target.u_old_max,
                eigenvalues[0],eigenvalues[-1],condition,symmetry,eps*branch_count,tf.cast(branch_count,D),
                zh,corrected,target.shift,tf.cast(valid_hint,D),new_zc,tau*zh))
            new_prefix = tuple(c.values for c in cores[:n])
            outputs = (report,pack_prefix(new_prefix),pack_gram(new_gram),m_c,l_cc)
            if full_capture:
                packed = tf.stack(tuple(tf.pad(c.values,((0,rank-c.left_rank),
                    (0,all_core_shape[2]-c.basis_dim),(0,rank-c.right_rank))) for c in cores))
                outputs += (_transition_target_summary(target),packed,chol)
            return (new_prefix,new_gram,tau*zh,new_zc,m_c,l_cc),outputs
        return step,starts,shapes[:n]

    first,first_starts,prefix_shapes = phase(initial_shapes,tf.constant([1],tf.int32))
    later,starts,_ = phase(prefix_shapes,tf.range(2,max(3,horizon)))

    def evaluate(observations,initial_mean,initial_cov,joint_means,joint_covs):
        chol = tf.linalg.cholesky(initial_cov)
        x = initial_mean[None]+tf.einsum("ij,nj->ni",chol,rows0)
        log_f = adapter.initial_log_density(x)+adapter.observation_log_density(x,observations[0])+_logdet_lower(chol)-_log_eta(rows0)
        shift = tf.reduce_logsumexp(log_f)-tf.math.log(tf.cast(tf.shape(log_f)[0],D))
        cores,worst,rms = _fit_als_graph(fitter,basis,rows0,tf.exp(.5*(log_f-shift)),
            weights0,initial,initial_shapes,config.sweeps,ridge)
        suffix = tf.tensor_scatter_nd_update(tf.zeros([cores[-1].right_rank,degree,1],D),[[0,0,0]],[1.])
        gram = suffix_gram_matrix((TTCore(suffix),),_hermite_product_basis(n+1,config.basis_degree),axis_offset=n)
        zh = tf.einsum("ab,ab->",prefix_gram_matrix(tuple(cores),basis),gram)
        tau,eps = _floor(rms,zh)
        zc = (1.+tau)*zh
        total = shift+tf.math.log(zc)
        valid_hint = (tf.reduce_all(tf.math.is_finite(initial_mean)) & tf.reduce_all(tf.math.is_finite(chol))
            & _retained_valid(gram,zc,tau))
        zero = tf.constant(0.,D)
        report = tf.stack((total,tau,eps,ess0,worst,rms,zero,zero,zero,zero,zero,zero,
            tf.constant(1.,D),zh,total-tf.math.log1p(tau),shift,tf.cast(valid_hint,D),zc,tau*zh))
        prefix = tuple(c.values for c in cores)
        outputs = (report,pack_prefix(prefix),pack_gram(gram),initial_mean,chol)
        if full_capture:
            outputs += (tf.zeros([9],D),tf.zeros(all_core_shape,D),tf.zeros([2*n,2*n],D))
        histories = tf.nest.map_structure(lambda v: tf.TensorArray(D,size=horizon,
            element_shape=v.shape,clear_after_read=False).write(0,v),outputs)
        if horizon > 1:
            state = (prefix,gram,tau*zh,zc,initial_mean,chol)
            state,outputs = first(tf.constant(1),state,observations[1],joint_means[0],joint_covs[0],
                tuple(v[0] for v in first_starts))
            total += outputs[0][0]
            histories = tf.nest.map_structure(lambda a,v: a.write(1,v),histories,outputs)
            def advance(t,state,total,histories):
                state,outputs = later(t,state,observations[t],joint_means[t-1],joint_covs[t-1],
                    tuple(v[t-2] for v in starts))
                return t+1,state,total+outputs[0][0],tf.nest.map_structure(lambda a,v: a.write(t,v),histories,outputs)
            if horizon > 2:
                _,_,total,histories = tf.while_loop(lambda t,*_: t<horizon,advance,
                    (tf.constant(2),state,total,histories),maximum_iterations=horizon-2,parallel_iterations=1)
        return total,tf.nest.map_structure(lambda a: a.stack(),histories)
    call = tf.function(evaluate,input_signature=[tf.TensorSpec(observation_shape,D),tf.TensorSpec([n],D),
        tf.TensorSpec([n,n],D),tf.TensorSpec([horizon-1,2*n],D),tf.TensorSpec([horizon-1,2*n,2*n],D)],
        jit_compile=jit_compile,autograph=False)
    call.prepared_rows,call.prepared_weights = rows,weights
    call.initial_shapes,call.prefix_shapes = initial_shapes,prefix_shapes
    cache[key] = call
    return call


def _retained_valid(gram,zc,tau):
    scale = tf.maximum(tf.reduce_max(tf.abs(gram)),tf.constant(1.,D))
    return ((tf.reduce_max(tf.abs(gram-tf.transpose(gram))) <= tf.constant(1e-12,D)*scale)
        & (tau >= 0.) & (zc > 0.) & tf.math.is_finite(zc))
