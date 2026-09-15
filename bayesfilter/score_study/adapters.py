"""Study endpoints that call the real Gaussian and canonical numerical kernels.

Host code constructs streams and artifacts; all numerical calculations use TF.
Scientific comparisons must use the separate calibration/uncertainty protocol.
"""
from __future__ import annotations

import time

from .contracts import digest, seed_pair
from .runtime import configure_runtime, memory_usage


def evaluate_gaussian(row, context):
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
        elif proposal == "ledh":
            from .canonical_adapter_tf import make_canonical_kernel
            if row.get("role") == "claim":
                from .tuning import consume_selection
                controls = consume_selection(row, context)
            else:
                controls = row["controls"]
            kernel = make_canonical_kernel(d, o, N, T, tuple(sorted(controls.items())), settings["dtype"], settings["jit_compile"])
            outputs = [kernel(theta, direction, observations, initial, process, reset_design)
                       for direction in tf.unstack(tf.eye(6, dtype=dtype))]
            values = tf.stack([x[0] for x in outputs])
            value, score = values[0], tf.stack([x[1] for x in outputs])
            tf.debugging.assert_near(values, tf.fill([6], value))
            diagnostics.update(reset_contract_id="contract_e_chol_v1", controls=controls,
                               chunk_policy="dpf_transport_exact_divisor_cap3000_v1", chunk_size=N,
                               score_consumer="make_canonical_kernel -> canonical_value_and_analytical_score",
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
    estimator = context["registry"].estimators[row["estimator"]]
    return {"value": value_number, "score": score_numbers,
            "oracle_value": float(oracle[0].numpy()), "oracle_score": oracle[1].numpy().tolist(),
            "value_target": "exact_gaussian_log_likelihood" if proposal in ("kalman", "ukf") else "finite_particle_log_likelihood",
            "derivative_target": estimator.target, "comparison_target": row["comparison_target"],
            "derivative_id": estimator.derivative, "proposal_law": context["registry"].proposals[proposal].law,
            "initial_terms": True, "numerical_validity": "pass",
            "inference_status": "mechanics_only" if context["study"].get("evidence_class") == "mechanics" or row.get("role", "mechanics") == "mechanics" else "descriptive_only",
            "runtime": runtime, "diagnostics": diagnostics}
