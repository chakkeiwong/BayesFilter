"""Offline fixed-iteration fitting followed by an independent psi-APF run.

The fit is conditional on a declared nominal parameter and the observations.
All fitted coefficients are frozen for the final finite-program derivative.
This is the log-quadratic local adaptation documented in fitted_twist_tf.
"""
from .contracts import digest


def execute_fitted_twist(row, settings, theta, observations, seed):
    import tensorflow as tf
    from .fitted_twist_execution_tf import make_fitted_twist_execution
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
    # Label construction is configuration metadata; random tensor generation
    # and every numerical fit update execute inside the shared XLA owner.
    seed_records={}
    def stream_seeds(label,fitting):
        records=[]
        for name in ("initial","process","ancestors","mixture"):
            stream=label+"_"+name
            draw_seed=seed(stream,replicate=0,group="offline_fitted_twist") if fitting else seed(stream)
            seed_records[stream]=draw_seed
            records.append(draw_seed)
        return records
    table=[stream_seeds(f"fit{iteration}",True) for iteration in range(iterations)]
    table.append(stream_seeds("final_twist",False))
    fitting_seeds={tuple(v) for k,v in seed_records.items() if not k.startswith("final_")}
    final_seeds={tuple(v) for k,v in seed_records.items() if k.startswith("final_")}
    if fitting_seeds & final_seeds:
        raise ValueError("offline fitting and final streams overlap")
    kernel=make_fitted_twist_execution(d,o,N,T,iterations,initial_variance,floor_ratio,
        settings["dtype"],settings["jit_compile"],**curves)
    output=kernel(theta,fit_theta,observations,tf.constant(table,tf.int32))
    invalid=int(output["invalid_iteration"].numpy())
    if invalid >= 0:
        raise ValueError(f"offline psi fit rank/positive-precision veto at iteration {invalid}")
    centers,covariances,log_floors=output["fit"]
    values=output["fit_log_values"].numpy().tolist()
    residuals=output["fit_errors"].numpy().tolist()
    iterations_out=[{"iteration":i,"fit_log_value":value,"log_fit_rmse":residual}
                    for i,(value,residual) in enumerate(zip(values,residuals))]
    frozen={"centers":centers.numpy().tolist(),"covariances":covariances.numpy().tolist(),
            "log_floors":log_floors.numpy().tolist(),"fit_theta":fit_theta.numpy().tolist()}
    final=output["final"]
    return kernel,final,{"fit":frozen,"fit_digest":digest(frozen),"fit_iterations":iterations_out,"fit_model":curves,
        "fit_observation_digest":digest(observations.numpy().tolist()),
        "fit_seed_records":seed_records,"fit_stopping":"fixed_declared_iteration_count",
        "fit_execution":"enclosing_tensorflow_loop","fit_enclosing_calls":1,
        "fit_method":"local_full_log_quadratic_qr_with_positive_precision_guard",
        "fit_parameter_derivative":"frozen_coefficients_at_declared_nominal_theta",
        "final_randomness":"independent_of_offline_fit_streams",
        "initial_normalizer":"mean_f_x0_psi1","terminal_future_factor":1.,
        "sampling_law":"normalized_gaussian_transition_times_gaussian_plus_positive_floor",
        "derivative_semantics":"fixed_ancestor_and_mixture_labels_a.e._frozen_fit_finite_program",
        "expectation_gradient_unbiasedness":"not_claimed",
        "canonical_status":"separate_fitted_psi_apf_local_adaptation"},2*iterations+1
