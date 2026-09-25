"""Bounded Algorithm-4 iteration with density-fit twists and frozen final score.

The diagonal density objective follows GJL (2017), eq. (15); box constraints,
projected optimization, sample CV and a particle approximation of initial x0
are explicit local choices. Claim rows await realized-N selection support.
"""
import math
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


def execute_iapf(row,settings,theta,observations,seed):
    import tensorflow as tf
    from .fitted_twist_tf import make_fitted_twist_kernel
    from .iapf_fit_tf import make_density_recursive_fit_kernel
    if row.get("role","mechanics")!="mechanics":
        raise ValueError("iAPF realized-particle-count tuning scope is not yet admitted; mechanics only")
    d,o,N,T=(settings[k] for k in ("dimension","observation_dimension","particles","horizon"))
    if d!=1 or o!=1:
        raise ValueError("density-fit iAPF consumer currently requires scalar state and observation")
    config=dict(row["iapf"])
    required={"k","tau","max_iterations","max_particles","mean_bound","sd_lower","sd_upper",
              "max_fit_steps","max_backtracks","fit_tolerance","floor_ratio","fit_theta"}
    if set(config)!=required:raise ValueError("explicit complete iAPF controls required")
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
    def fail(message):
        raise DiagnosticFailure(message, {"iapf_configuration": config,
            "fit_iterations": details, "fit_seed_records": seed_records,
            "log_values": histories, "particle_counts": counts,
            "fit_diagnostic_columns": FIT_DIAGNOSTIC_COLUMNS})
    def streams(label,fitting):
        def draw(name,shape,normal):
            key=label+"_"+name
            stream=seed(key,replicate=0,group="offline_density_iapf") if fitting else seed(key)
            seed_records[key]=stream
            return (tf.random.stateless_normal if normal else tf.random.stateless_uniform)(shape,stream,dtype=theta.dtype)
        return draw("initial",[N,d],True),draw("process",[T,N,d],True),draw("ancestors",[T+1,N],False),draw("mixture",[T,N],False)
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
            settings["dtype"],settings["jit_compile"])
        fit_kernels.append(fitter)
        centers,covariances,floors,valid,converged,fit_details=fitter(fit_theta,observations,output[2]);calls+=1
        record["density_fit_diagnostics"]=fit_details.numpy().tolist()
        record["density_fit_valid"]=bool(valid.numpy())
        record["density_fit_converged"]=bool(converged.numpy())
        record["density_fit_parameters"]={"centers":centers.numpy().tolist(),
            "covariances":covariances.numpy().tolist(),"log_floors":floors.numpy().tolist()}
        if not bool(valid.numpy()) or not bool(converged.numpy()):
            fail(f"iAPF density fit invalid or unconverged at iteration {iteration}")
        N=decision["next_particles"]
    frozen={"centers":centers.numpy().tolist(),"covariances":covariances.numpy().tolist(),
            "log_floors":floors.numpy().tolist(),"fit_theta":fit_theta.numpy().tolist(),"particles":N}
    kernel=make_fitted_twist_kernel(d,o,N,T,settings["dtype"],settings["jit_compile"])
    final=kernel(theta,observations,*streams("iapf_final",False),centers,covariances,floors);calls+=1
    final_seeds={tuple(v) for key,v in seed_records.items() if key.startswith("iapf_final")}
    fitting_seeds={tuple(v) for key,v in seed_records.items() if not key.startswith("iapf_final")}
    if final_seeds&fitting_seeds:raise ValueError("iAPF final streams overlap fitting streams")
    return kernel,final,{"iapf_configuration":config,"fit":frozen,"fit_digest":digest(frozen),
        "candidate_configuration":{"iapf":config},
        "fit_iterations":details,"fit_seed_records":seed_records,"actual_particle_count":N,
        "fit_method":"bounded_diagonal_density_scale_least_squares_profiled_scale",
        "fit_stopping":"GJL_algorithm4_l_gt_k_sample_cv_with_iteration_and_particle_caps",
        "fit_diagnostic_columns":FIT_DIAGNOSTIC_COLUMNS,
        "fit_search_step_bound":"max_parameter_box_extent_divided_by_tolerance",
        "fit_arithmetic":"log_scaled_density_objective_original_analytical_gradient",
        "fit_trace_counts":[f.experimental_get_tracing_count() for f in set(fit_kernels)],
        "run_trace_counts":[f.experimental_get_tracing_count() for f in set(run_kernels+[kernel])],
        "final_randomness":"independent_of_all_offline_iterations",
        "fit_parameter_derivative":"frozen_coefficients_at_declared_nominal_theta",
        "derivative_semantics":"fixed_ancestor_and_mixture_labels_a.e._frozen_fit_and_realized_N_finite_program",
        "expectation_gradient_unbiasedness":"not_claimed",
        "canonical_status":"separate_bounded_density_iapf_with_particle_initialization"},calls
