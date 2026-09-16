"""Offline fixed-iteration fitting followed by an independent psi-APF run.

The fit is conditional on a declared nominal parameter and the observations.
All fitted coefficients are frozen for the final finite-program derivative.
This is the log-quadratic local adaptation documented in fitted_twist_tf.
"""
from .contracts import digest


def execute_fitted_twist(row, settings, theta, observations, seed):
    import tensorflow as tf
    from .fitted_twist_tf import make_fitted_twist_kernel, make_recursive_fit_kernel
    from .conditional_means_tf import model_curves
    curves=model_curves(row,settings)
    d,o,N,T=(settings[k] for k in ("dimension","observation_dimension","particles","horizon"))
    iterations=row["fit_iterations"]
    initial_variance=row["fit_initial_variance"]
    floor_ratio=row["fit_floor_ratio"]
    if type(iterations) is not int or not 1 <= iterations <= 20:
        raise ValueError("declare between one and twenty offline fit iterations")
    if not 0 < initial_variance < float("inf") or not 0 < floor_ratio < float("inf"):
        raise ValueError("positive finite initial variance and floor ratio required")
    # The full positive Gaussian terminal fit is presently justified only in
    # this fixture. A subspace/ridge extension must be evaluated separately.
    if d != 1 or o != 1:
        raise ValueError("fitted_twist currently requires dimension=observation_dimension=1")
    dtype=theta.dtype
    fit_theta=tf.constant(row["fit_theta"],dtype)
    tf.debugging.assert_equal(tf.shape(fit_theta),[6])
    kernel=make_fitted_twist_kernel(d,o,N,T,settings["dtype"],settings["jit_compile"],**curves)
    fitter=make_recursive_fit_kernel(d,o,N,T,floor_ratio,settings["dtype"],settings["jit_compile"],**curves)
    centers=tf.zeros([T,d],dtype)
    covariances=tf.eye(d,batch_shape=[T],dtype=dtype)*tf.cast(initial_variance,dtype)
    log_floors=tf.fill([T],tf.math.log(tf.cast(floor_ratio,dtype))-
        .5*tf.cast(d,dtype)*tf.math.log(tf.cast(2*3.141592653589793*initial_variance,dtype)))
    seed_records={}
    def streams(label,fitting):
        def draw(name,shape,normal):
            stream=label+"_"+name
            draw_seed=seed(stream,replicate=0,group="offline_fitted_twist") if fitting else seed(stream)
            seed_records[stream]=draw_seed
            return (tf.random.stateless_normal if normal else tf.random.stateless_uniform)(shape,draw_seed,dtype=dtype)
        return (draw("initial",[N,d],True),draw("process",[T,N,d],True),
                draw("ancestors",[T+1,N],False),draw("mixture",[T,N],False))
    iterations_out=[]
    for iteration in range(iterations):
        fitted_run=kernel(fit_theta,observations,*streams(f"fit{iteration}",True),centers,covariances,log_floors)
        centers,covariances,log_floors,valid,residual=fitter(fit_theta,observations,fitted_run[2])
        if not bool(valid.numpy()):
            raise ValueError(f"offline psi fit rank/positive-precision veto at iteration {iteration}")
        iterations_out.append({"iteration":iteration,"fit_log_value":float(fitted_run[0].numpy()),
                               "log_fit_rmse":residual.numpy().tolist()})
    frozen={"centers":centers.numpy().tolist(),"covariances":covariances.numpy().tolist(),
            "log_floors":log_floors.numpy().tolist(),"fit_theta":fit_theta.numpy().tolist()}
    final=kernel(theta,observations,*streams("final_twist",False),centers,covariances,log_floors)
    fitting_seeds={tuple(v) for k,v in seed_records.items() if not k.startswith("final_")}
    final_seeds={tuple(v) for k,v in seed_records.items() if k.startswith("final_")}
    if fitting_seeds & final_seeds:
        raise ValueError("offline fitting and final streams overlap")
    return kernel,final,{"fit":frozen,"fit_digest":digest(frozen),"fit_iterations":iterations_out,"fit_model":curves,
        "fit_observation_digest":digest(observations.numpy().tolist()),
        "fit_seed_records":seed_records,"fit_stopping":"fixed_declared_iteration_count",
        "fit_method":"local_full_log_quadratic_qr_with_positive_precision_guard",
        "fit_parameter_derivative":"frozen_coefficients_at_declared_nominal_theta",
        "final_randomness":"independent_of_offline_fit_streams",
        "initial_normalizer":"mean_f_x0_psi1","terminal_future_factor":1.,
        "sampling_law":"normalized_gaussian_transition_times_gaussian_plus_positive_floor",
        "derivative_semantics":"fixed_ancestor_and_mixture_labels_a.e._frozen_fit_finite_program",
        "expectation_gradient_unbiasedness":"not_claimed",
        "canonical_status":"separate_fitted_psi_apf_local_adaptation"},2*iterations+1
