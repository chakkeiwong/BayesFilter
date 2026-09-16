"""Study endpoints that call the real Gaussian and canonical numerical kernels.

Host code constructs streams and artifacts; all numerical calculations use TF.
Scientific comparisons must use the separate calibration/uncertainty protocol.
"""
from __future__ import annotations

import time

from .contracts import digest, seed_pair
from .runtime import configure_runtime, memory_usage


def evaluate_gaussian(row, context):
    from .tuning import TUNABLE_PROPOSALS, candidate_configuration, selected_row
    if row.get("role") == "claim" and row["proposal"] in TUNABLE_PROPOSALS:
        row = selected_row(row, context)
    settings = context["study"]["settings"]
    runtime = configure_runtime(device=settings["device"], tf32=settings["tf32"],
                                jit_compile=settings["jit_compile"])
    import tensorflow as tf
    from .gaussian_tf import make_data_kernel, make_gaussian_kernel, make_particle_kernel, parameterized_model

    dtype = tf.as_dtype(settings["dtype"])
    d, o, T, N = (settings[k] for k in ("dimension", "observation_dimension", "horizon", "particles"))
    theta = tf.constant(settings["theta"], dtype)
    data_theta = tf.constant(settings["data_theta"], dtype)
    def seed(stream, *, replicate=None, group=None):
        return seed_pair(master_seed=context["study"]["seed"], model=row["model"],
                         dataset=row["dataset"], replicate=row["replicate"] if replicate is None else replicate,
                         stream=stream, coupling_group=group or row.get("coupling_group", "baseline"))
    data_seed = seed("observations", replicate=0, group="common_data")
    with tf.device("/GPU:0" if settings["device"] == "GPU" else "/CPU:0"):
        data_kernel = make_data_kernel(d, o, T, settings["dtype"], settings["jit_compile"])
        observations = data_kernel(data_theta, tf.constant(data_seed, tf.int32))
        initial = tf.random.stateless_normal([N, d], seed("initial"), dtype=dtype)
        process = tf.random.stateless_normal([T, N, d], seed("process"), dtype=dtype)
        uniforms = tf.random.stateless_uniform([T, N], seed("resampling"), dtype=dtype)
        reset_design = tf.random.stateless_normal([N, d], seed("reset_design"), dtype=dtype)
        oracle_kernel = make_gaussian_kernel(d, o, 6, settings["dtype"], settings["jit_compile"])
        oracle = oracle_kernel(observations, *parameterized_model(theta, d, o))
        diagnostics = {"data_seed": data_seed, "data_version": digest(observations.numpy().tolist()),
                       "initial_seed": seed("initial"), "process_seed": seed("process"),
                       "resampling_seed": seed("resampling"), "reset_seed": seed("reset_design"),
                       "state_dimension": d, "parameter_dimension": 6,
                       "particle_count": N, "horizon": T, "uses_autodiff": False,
                       "data_model": "affine_gaussian_all_six_parameters"}
        if row.get("collect_kdm_control", False):
            from .combinations_tf import make_frozen_kdm_controls
            # The pilot has its own seed namespace and uses no observations or
            # evaluation innovations. Its density parameters are held fixed in
            # the artificial translation/scale score used as a control.
            pilot_seed = seed_pair(master_seed=context["study"]["seed"], model=row["model"],
                                   dataset=0, replicate=0, stream="independent_kdm_control_pilot",
                                   coupling_group="fixed_pilot")
            M = 4
            model_parts = parameterized_model(theta, d, o)
            pilot_noise = tf.random.stateless_normal([M, d], pilot_seed, dtype=dtype)
            means = model_parts[4] + pilot_noise @ tf.transpose(tf.linalg.cholesky(model_parts[6]))
            covariance = tf.broadcast_to(model_parts[6][None, ...], [M, d, d])
            component_uniforms = tf.random.stateless_uniform([N], seed("kdm_control_component"), dtype=dtype)
            controls_kernel = make_frozen_kdm_controls(M, d, N, settings["dtype"], settings["jit_compile"])
            zero_controls, control_valid = controls_kernel(
                tf.fill([M], tf.cast(1/M, dtype)), means, covariance, component_uniforms, initial)
            if not bool(control_valid.numpy()):
                raise ValueError("KDM control density validity veto")
            diagnostics.update(kdm_zero_control=tf.reduce_mean(zero_controls, axis=0).numpy().tolist(),
                               kdm_control_pilot_seed=pilot_seed,
                               kdm_control_law="fixed_prior_pilot_mixture_translation_log_scale_density_score",
                               kdm_control_center=[0.] * (d+1),
                               kdm_control_shared_innovations="initial",
                               kdm_control_component_seed=seed("kdm_control_component"),
                               kdm_control_pilot_components=M,
                               kdm_control_pilot_provenance="untuned_mechanics_hypothesis")
        proposal = row["proposal"]
        started = time.monotonic()
        if proposal in ("kalman", "ukf"):
            kernel = make_gaussian_kernel(d, o, 6, settings["dtype"], settings["jit_compile"], proposal == "ukf")
            value, score, mean, covariance, minimum_eigenvalue = kernel(observations, *parameterized_model(theta, d, o))
            diagnostics.update(final_mean=mean.numpy().tolist(), final_covariance=covariance.numpy().tolist(),
                               minimum_covariance_eigenvalue=float(minimum_eigenvalue.numpy()))
            if not bool((minimum_eigenvalue > 0).numpy()):
                raise ValueError("Gaussian covariance validity veto")
            calls = 1
        elif proposal == "fitted_twist":
            from .fitted_twist_adapter import execute_fitted_twist
            kernel, fitted_output, fitted_diagnostics, calls = execute_fitted_twist(
                row, settings, theta, observations, seed)
            value, score = fitted_output[:2]
            diagnostics.update(fitted_diagnostics)
        elif proposal == "twist":
            from .twist_tf import make_twist_kernel
            kernel = make_twist_kernel(d, o, N, T, row["twist_power"], settings["dtype"], settings["jit_compile"])
            twist_uniforms = tf.concat([tf.random.stateless_uniform([1,N], seed("twist_initial_ancestor"), dtype=dtype),uniforms],axis=0)
            value, score, ess = kernel(theta, observations, initial, process, twist_uniforms)
            diagnostics.update(minimum_ess=float(ess.numpy()), twist_power=row["twist_power"],
                               initial_normalizer="mean_f_x0_psi1", terminal_future_factor=1.,
                               sampling_law="normalized_transition_times_observation_density_to_fixed_power",
                               derivative_semantics="fixed_ancestor_a.e._finite_program",
                               expectation_gradient_unbiasedness="not_claimed",
                               canonical_status="separate_psi_apf_comparator")
            calls = 1
        elif proposal in ("bootstrap", "adapted", "prior_sis", "adapted_sis"):
            resampling = proposal in ("bootstrap", "adapted")
            kernel = make_particle_kernel(d, o, N, T, settings["dtype"], settings["jit_compile"],
                                          proposal in ("adapted", "adapted_sis"), resampling)
            args = (theta, observations, initial, process) + ((uniforms,) if resampling else ())
            value, score, ess = kernel(*args)
            diagnostics.update(minimum_ess=float(ess.numpy()), resampling="multinomial_each_step" if resampling else "none",
                               derivative_semantics="fixed_ancestor_a.e._finite_program" if resampling else "fixed_base_noise_finite_program",
                               expectation_gradient_unbiasedness="not_claimed")
            calls = 1
        elif proposal in ("ledh", "integrated_kdm", "resampling_kdm", "sgqf", "kdm_covariance"):
            from .canonical_adapter_tf import make_canonical_kernel
            controls = row["controls"]
            if proposal == "ledh":
                kernel = make_canonical_kernel(d, o, N, T, tuple(sorted(controls.items())), settings["dtype"], settings["jit_compile"])
                outputs = [kernel(theta, direction, observations, initial, process, reset_design)
                           for direction in tf.unstack(tf.eye(6, dtype=dtype))]
            elif proposal == "sgqf":
                from .covariance_adapter_tf import make_covariance_kernel
                kernel, metadata = make_covariance_kernel(d, o, N, T, tuple(sorted(controls.items())),
                    row["sgqf_level"], settings["dtype"], settings["jit_compile"])
                outputs = [kernel(theta, direction, observations, initial, process, reset_design)
                           for direction in tf.unstack(tf.eye(6, dtype=dtype))]
                diagnostics.update(covariance_provider=metadata, canonical_status="alternative_covariance_candidate")
            elif proposal == "kdm_covariance":
                from .mixture_covariance_tf import make_mixture_covariance_kernel
                kernel = make_mixture_covariance_kernel(d, o, N, T, tuple(sorted(controls.items())),
                    row["within_fraction"], settings["dtype"], settings["jit_compile"])
                outputs = [kernel(theta, direction, observations, initial, process, reset_design)
                           for direction in tf.unstack(tf.eye(6, dtype=dtype))]
                schedule = outputs[0][2]
                diagnostics.update(covariance_provider="local_gaussian_mixture_assumed_density_filter",
                    within_fraction=row["within_fraction"], component_count=2*d,
                    component_lifecycle="independent_prediction_observation_conditioning_and_evidence_reweighting",
                    canonical_status="alternative_covariance_candidate_not_younis_filter_reproduction",
                    final_component_log_weights=schedule["component_log_weights"][-1].numpy().tolist(),
                    final_provider_covariance=schedule["post_covariances"][-1].numpy().tolist())
            else:
                from .kdm_adapter_tf import make_kdm_kernel
                bandwidth = row["bandwidth_scale"]
                if not isinstance(bandwidth, (int, float)) or not 0 < bandwidth < float("inf"):
                    raise ValueError("positive finite KDM bandwidth scale required")
                kernel = make_kdm_kernel(d, o, N, T, tuple(sorted(controls.items())), proposal,
                                         settings["dtype"], settings["jit_compile"])
                mixture_noise = tf.random.stateless_normal([T, N, d], seed("kdm_noise"), dtype=dtype)
                mixture_uniforms = tf.random.stateless_uniform([T, N], seed("kdm_components"), dtype=dtype)
                outputs = [kernel(theta, direction, observations, initial, process, reset_design,
                                  mixture_uniforms, mixture_noise, tf.constant(bandwidth, dtype))
                           for direction in tf.unstack(tf.eye(6, dtype=dtype))]
                if not all(bool(x[2].numpy()) for x in outputs):
                    raise ValueError("KDM consumer validity veto")
                diagnostics.update(bandwidth_scale=bandwidth, bandwidth_law="scale_squared_times_Q_theta",
                                   bandwidth_derivative="total_model_parameter_dependence",
                                   kdm_target="KDM-FINITE" if proposal == "integrated_kdm" else "RESKDM-IWSG-FINITE",
                                   expectation_gradient_unbiasedness="not_claimed_for_log_program",
                                   kdm_random_seed=seed("kdm_noise"), kdm_components_seed=seed("kdm_components"))
            values = tf.stack([x[0] for x in outputs])
            value, score = values[0], tf.stack([x[1] for x in outputs])
            tf.debugging.assert_near(values, tf.fill([6], value))
            diagnostics.update(reset_contract_id="contract_e_chol_v1", controls=controls,
                               chunk_policy="dpf_transport_exact_divisor_cap3000_v1", chunk_size=N,
                               score_consumer=({"ledh": "make_canonical_kernel -> canonical_value_and_analytical_score",
                                                "sgqf": "make_covariance_kernel -> shared executor -> SGQFCovarianceProvider",
                                                "kdm_covariance": "mixture_schedule -> shared executor -> scheduled prediction/conditioning moments"}.get(proposal, "make_kdm_kernel -> KDM consumer -> shared canonical executor")),
                               derivative_semantics="analytical_total_initial_moments_flow_genut_contract_e",
                               tuning_status="verified_selection" if row.get("role") == "claim" else "mechanics_or_tuning_candidate")
            calls = 6
        else:
            raise ValueError(f"unimplemented Gaussian proposal: {proposal}")
        if value.shape != () or score.shape != (6,) or value.dtype != dtype or score.dtype != dtype:
            raise ValueError("numerical result shape/dtype veto")
        tf.debugging.assert_all_finite(score, "score")
        tf.debugging.assert_all_finite(value, "value")
        # Synchronize before reporting wall time, device and allocator usage.
        value_number, score_numbers = float(value.numpy()), score.numpy().tolist()
        runtime.update(dtype=dtype.name, value_device=value.device, score_device=score.device,
                       kernel_wall_seconds=time.monotonic() - started, kernel_calls=calls,
                       traces=kernel.experimental_get_tracing_count(), **memory_usage(settings["device"]))
    diagnostics["score_squared_error"] = float(tf.reduce_sum((score - oracle[1])**2).numpy())
    diagnostics["value_squared_error"] = float(((value - oracle[0])**2).numpy())
    if proposal in TUNABLE_PROPOSALS:
        diagnostics["candidate_configuration"] = candidate_configuration(row)
    estimator = context["registry"].estimators[row["estimator"]]
    return {"value": value_number, "score": score_numbers,
            "oracle_value": float(oracle[0].numpy()), "oracle_score": oracle[1].numpy().tolist(),
            "value_target": ({"kalman": "exact_gaussian_log_likelihood", "ukf": "exact_gaussian_log_likelihood", "integrated_kdm": "KDM-FINITE", "resampling_kdm": "RESKDM-IWSG-FINITE"}.get(proposal, "finite_particle_log_likelihood")),
            "derivative_target": estimator.target, "comparison_target": row["comparison_target"],
            "derivative_id": estimator.derivative, "proposal_law": context["registry"].proposals[proposal].law,
            "initial_terms": True, "numerical_validity": "pass",
            "inference_status": "mechanics_only" if context["study"].get("evidence_class") == "mechanics" or row.get("role", "mechanics") == "mechanics" else "descriptive_only",
            "runtime": runtime, "diagnostics": diagnostics}
