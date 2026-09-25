"""Bounded Algorithm-4 iteration with density-fit twists and frozen final score.

The diagonal density objective follows GJL (2017), eq. (15); box constraints,
projected optimization, sample CV and a particle approximation of initial x0
are explicit local choices. Adaptive counts and all fitting work are reported.
"""
import math
import time
from .contracts import DiagnosticFailure, digest


FIT_DIAGNOSTIC_COLUMNS = ["scaled_loss", "normalized_shape_residual", "log_lambda",
                          "boundary_active", "iterations", "projected_gradient",
                          "minimum_sd", "maximum_sd", "density_log_amplitude",
                          "objective_underflow"]


def iteration_decision(log_values,particle_counts,*,k,tau,max_particles):
    """Pure controller; log scaling preserves the sample coefficient of variation."""
    if type(k) is not int or k<1 or not math.isfinite(tau) or tau<=0:
        raise ValueError("positive k and tau required")
    if not log_values or len(log_values)!=len(particle_counts) or not all(math.isfinite(v) for v in log_values):
        raise ValueError("finite equally sized nonempty iteration history required")
    if any(type(n) is not int or n<2 or n>max_particles for n in particle_counts):
        raise ValueError("particle history exceeds declared bounds")
    iteration=len(log_values)-1;N=particle_counts[-1]
    if iteration<k:
        return {"action":"fit","next_particles":N,"cv":None,"complete_window":False}
    window=log_values[-k-1:];maximum=max(window)
    values=[math.exp(x-maximum) for x in window]
    mean=sum(values)/len(values)
    cv=math.sqrt(sum((x-mean)**2 for x in values)/k)/mean
    if iteration>k and cv<tau:
        return {"action":"final","next_particles":N,"cv":cv,"complete_window":True}
    equal_n=all(n==N for n in particle_counts[-k-1:])
    monotone=all(a<=b for a,b in zip(window,window[1:]))
    next_n=2*N if equal_n and not monotone else N
    return {"action":"capacity_veto" if next_n>max_particles else "fit",
            "next_particles":next_n,"cv":cv,"complete_window":True}


def execute_iapf(row,settings,theta,observations,seed,context=None):
    if row.get("role") == "claim":
        if context is None:
            raise ValueError("claim iAPF requires a verified study context and tuning selection")
        from .tuning import selected_row
        row = selected_row(row, context)
    import tensorflow as tf
    from .fitted_twist_tf import make_fitted_twist_kernel
    from .iapf_fit_tf import make_density_recursive_fit_kernel
    from .iapf_scope import cost_record, validate_adaptive_ledger
    d,o,N,T=(settings[k] for k in ("dimension","observation_dimension","particles","horizon"))
    if d!=1 or o!=1:
        raise ValueError("density-fit iAPF consumer currently requires scalar state and observation")
    config=dict(row["iapf"])
    required={"k","tau","max_iterations","max_particles","mean_bound","sd_lower","sd_upper",
              "max_fit_steps","max_backtracks","fit_tolerance","floor_ratio","fit_theta"}
    if not required<=set(config) or set(config)-required-{"fit_dtype"}:
        raise ValueError("explicit complete iAPF controls required")
    fit_dtype_name=config.get("fit_dtype",settings["dtype"])
    if fit_dtype_name not in ("float32","float64"):
        raise ValueError("iAPF fit_dtype must be float32 or float64")
    if settings["dtype"]=="float64" and fit_dtype_name!="float64":
        raise ValueError("offline fit precision must not be below the filter precision")
    fit_dtype=tf.as_dtype(fit_dtype_name)
    if type(config["max_iterations"]) is not int or not config["k"]+2<=config["max_iterations"]<=50:
        raise ValueError("declare a finite iteration cap allowing the first stopping check")
    if type(config["max_particles"]) is not int or config["max_particles"]<N:
        raise ValueError("particle cap below initial count")
    iteration_decision([0.],[N],k=config["k"],tau=config["tau"],max_particles=config["max_particles"])
    fit_theta=tf.constant(config["fit_theta"],theta.dtype)
    tf.debugging.assert_equal(tf.shape(fit_theta),[6]);tf.debugging.assert_all_finite(fit_theta,"invalid nominal theta")
    centers=tf.zeros([T,d],theta.dtype);covariances=tf.eye(d,batch_shape=[T],dtype=theta.dtype)
    floors=tf.zeros([T],theta.dtype)
    histories=[];counts=[];details=[];seed_records={};fit_kernels=[];run_kernels=[];calls=0
    fit_precision={"fit_dtype":fit_dtype_name,"filter_dtype":settings["dtype"],
                   "cast_policy":"explicit_offline_fit_then_cast_coefficients_once",
                   "automatic_precision_fallback":False}
    def fail(message):
        raise DiagnosticFailure(message, {"iapf_configuration": config,
            "fit_iterations": details, "fit_seed_records": seed_records,
            "log_values": histories, "particle_counts": counts,
            "fit_diagnostic_columns": FIT_DIAGNOSTIC_COLUMNS,"fit_precision":fit_precision})
    def streams(label,fitting):
        def draw(name,shape,normal):
            key=label+"_"+name
            stream=seed(key,replicate=0,group="offline_density_iapf") if fitting else seed(key)
            seed_records[key]=stream
            return (tf.random.stateless_normal if normal else tf.random.stateless_uniform)(shape,stream,dtype=theta.dtype)
        return draw("initial",[N,d],True),draw("process",[T,N,d],True),draw("ancestors",[T+1,N],False),draw("mixture",[T,N],False)
    offline_started=time.monotonic()
    for iteration in range(config["max_iterations"]):
        kernel=make_fitted_twist_kernel(d,o,N,T,settings["dtype"],settings["jit_compile"],constant_twist=iteration==0)
        run_kernels.append(kernel)
        output=kernel(fit_theta,observations,*streams(f"iapf_fit{iteration}",True),centers,covariances,floors);calls+=1
        histories.append(float(output[0].numpy()));counts.append(N)
        if not math.isfinite(histories[-1]):
            fail(f"nonfinite iAPF likelihood at iteration {iteration}")
        decision=iteration_decision(histories,counts,k=config["k"],tau=config["tau"],max_particles=config["max_particles"])
        record={"iteration":iteration,"particles":N,"log_value":histories[-1],**decision};details.append(record)
        if decision["action"]=="final":break
        if decision["action"]=="capacity_veto":fail(f"iAPF particle cap exhausted at iteration {iteration}")
        if iteration+1==config["max_iterations"]:fail("iAPF iteration cap exhausted before convergence")
        fitter=make_density_recursive_fit_kernel(d,o,N,T,config["mean_bound"],config["sd_lower"],config["sd_upper"],
            config["max_fit_steps"],config["max_backtracks"],config["fit_tolerance"],config["floor_ratio"],
            fit_dtype_name,settings["jit_compile"])
        fit_kernels.append(fitter)
        fitted_centers,fitted_covariances,fitted_floors,valid,converged,fit_details=fitter(
            tf.cast(fit_theta,fit_dtype),tf.cast(observations,fit_dtype),tf.cast(output[2],fit_dtype));calls+=1
        fitted=(fitted_centers,fitted_covariances,fitted_floors)
        centers,covariances,floors=(tf.cast(x,theta.dtype) for x in fitted)
        record["coefficient_cast_max_abs_error"]={key:float(tf.reduce_max(tf.abs(original-tf.cast(cast,fit_dtype))).numpy())
            for key,original,cast in zip(("centers","covariances","log_floors"),fitted,(centers,covariances,floors))}
        cast_valid=all(bool(tf.reduce_all(tf.math.is_finite(x)).numpy()) for x in (centers,covariances,floors))
        # This fitter returns a diagonal covariance; no eigensolver or ridge is needed.
        cast_valid=cast_valid and bool(tf.reduce_all(tf.linalg.diag_part(covariances)>0).numpy())
        record["coefficient_cast_valid"]=cast_valid
        record["density_fit_diagnostics"]=fit_details.numpy().tolist()
        record["density_fit_valid"]=bool(valid.numpy())
        record["density_fit_converged"]=bool(converged.numpy())
        record["density_fit_parameters"]={"centers":centers.numpy().tolist(),
            "covariances":covariances.numpy().tolist(),"log_floors":floors.numpy().tolist()}
        if not bool(valid.numpy()) or not bool(converged.numpy()):
            fail(f"iAPF density fit invalid or unconverged at iteration {iteration}")
        if not cast_valid:fail(f"iAPF fitted coefficient cast invalid at iteration {iteration}")
        N=decision["next_particles"]
    frozen={"centers":centers.numpy().tolist(),"covariances":covariances.numpy().tolist(),
            "log_floors":floors.numpy().tolist(),"fit_theta":fit_theta.numpy().tolist(),"particles":N}
    offline_seconds=time.monotonic()-offline_started
    count_ledger=validate_adaptive_ledger(details,initial_particles=settings["particles"],
        max_particles=config["max_particles"],k=config["k"],tau=config["tau"])
    final_started=time.monotonic()
    kernel=make_fitted_twist_kernel(d,o,N,T,settings["dtype"],settings["jit_compile"])
    final=kernel(theta,observations,*streams("iapf_final",False),centers,covariances,floors);calls+=1
    # Materialize the score to include asynchronous device completion in timing.
    final[1].numpy()
    final_seconds=time.monotonic()-final_started
    work=cost_record(ledger=details,fit_calls=len(fit_kernels),
        fit_optimizer_steps=sum(int(step[4]) for rec in details
            for step in rec.get("density_fit_diagnostics",[])),
        final_particle_count=N,horizon=T,offline_seconds=offline_seconds,
        final_seconds=final_seconds)
    final_seeds={tuple(v) for key,v in seed_records.items() if key.startswith("iapf_final")}
    fitting_seeds={tuple(v) for key,v in seed_records.items() if not key.startswith("iapf_final")}
    if final_seeds&fitting_seeds:raise ValueError("iAPF final streams overlap fitting streams")
    return kernel,final,{"iapf_configuration":config,"fit":frozen,"fit_digest":digest(frozen),
        "candidate_configuration":{"iapf":config},
        "fit_iterations":details,"fit_seed_records":seed_records,"actual_particle_count":N,
        "adaptive_count_ledger":count_ledger,"work_accounting":work,
        "fit_observation_digest":digest(observations.numpy().tolist()),
        "fit_method":"bounded_diagonal_density_scale_least_squares_profiled_scale",
        "fit_stopping":"GJL_algorithm4_l_gt_k_sample_cv_with_iteration_and_particle_caps",
        "fit_diagnostic_columns":FIT_DIAGNOSTIC_COLUMNS,
        "fit_search_step_bound":"max_parameter_box_extent_divided_by_tolerance",
        "fit_arithmetic":"log_scaled_density_objective_original_analytical_gradient",
        "fit_precision":fit_precision,
        "fit_trace_counts":[f.experimental_get_tracing_count() for f in set(fit_kernels)],
        "run_trace_counts":[f.experimental_get_tracing_count() for f in set(run_kernels+[kernel])],
        "final_randomness":"independent_of_all_offline_iterations",
        "fit_parameter_derivative":"frozen_coefficients_at_declared_nominal_theta",
        "derivative_semantics":"fixed_ancestor_and_mixture_labels_a.e._frozen_fit_and_realized_N_finite_program",
        "expectation_gradient_unbiasedness":"not_claimed",
        "canonical_status":"separate_bounded_density_iapf_with_particle_initialization"},calls
