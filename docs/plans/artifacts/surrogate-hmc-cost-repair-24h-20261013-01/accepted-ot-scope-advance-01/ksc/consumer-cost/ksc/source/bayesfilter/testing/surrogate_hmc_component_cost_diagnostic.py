"""Diagnostic-only stage timing of the unchanged analytical filter primitives.

Stages materialize intermediates and can change XLA fusion. They are neither a
runtime implementation nor a tuning/admission authority. Compare all final
outputs with the actual one-observation factory before interpreting timings.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared
from bayesfilter.highdim.analytical_jet_tf import Jet, stack, exp, logsumexp


def make_stages(spec, n, controls, dtype, precision_policy="uniform", batch_size=2, recursive=False):
    """Three XLA graphs calling exactly the shared proposal/density/reset kernels."""
    dtype = tf.as_dtype(dtype)
    if precision_policy not in shared.PRECISION_POLICIES:
        raise ValueError("unknown precision policy")
    if precision_policy != "uniform" and dtype != tf.float64:
        raise ValueError("mixed pair arithmetic requires FP64 recurrence")
    b, p, d = batch_size, spec.parameter_count, spec.dimension
    o = getattr(spec, "observation_dimension", d)
    tensor = lambda shape, kind=dtype: tf.TensorSpec(shape, kind)
    cloud, tangent = tensor([b, n, d]), tensor([p, b, n, d])
    proposal_signature = [
        tensor([b, p]), tensor([p+1, b, n, d] if recursive else [n, d]), tensor([o]), tensor([n, d]),
        tensor([n, 2]), tensor([3]),
    ]

    @tf.function(input_signature=proposal_signature, jit_compile=True, autograph=False)
    def proposal(theta, initial, observation, innovations, uniforms, beta):
        directions = tf.eye(p, dtype=dtype)
        models = tuple(shared.Models(spec, theta[row], directions) for row in range(b))
        if recursive:
            x = Jet(initial[0], initial[1:])
        else:
            clouds = []
            for row in range(b):
                initialized = [spec.initial_cloud(theta[row], initial, directions[k])
                               for k in range(p)]
                clouds.append(Jet(initialized[0][0], tf.stack([x[2] for x in initialized])))
            x = stack(clouds)
        maps = tuple(shared.build_maps(models[row], x[row], observation, controls)
                     for row in range(b))
        child = stack([shared.draw_from_maps(m, beta, uniforms, innovations) for m in maps])
        means, covariance = stack([m[0] for m in maps]), stack([m[1] for m in maps])
        likelihood = stack([shared.observation_density(models[row], child[row], observation)
                            for row in range(b)])
        valid = tf.stack([m[6] for m in maps])
        return (child.v, child.t, means.v, means.t, covariance.v, covariance.t,
                likelihood.v, likelihood.t, valid)

    density_signature = [
        cloud, tangent, tensor([b, 3, n, d]), tensor([p, b, 3, n, d]),
        tensor([b, 3, n, d, d]), tensor([p, b, 3, n, d, d]),
        tensor([b, n]), tensor([p, b, n]), tensor([b], tf.bool), tensor([3]), tensor([b, p]),
    ]

    @tf.function(input_signature=density_signature, jit_compile=True, autograph=False)
    def density(xv, xt, mv, mt, cv, ct, lv, lt, proposal_valid, beta, theta):
        models = tuple(shared.Models(spec, theta[row], tf.eye(p, dtype=dtype)) for row in range(b))
        constant_jacobian = all(model.constant_observation_jacobian for model in models)
        x, means, covariance = Jet(xv, xt), Jet(mv, mt), Jet(cv, ct)
        parts = [shared.precision_gaussian_mixture(
            x, means[:, k], covariance[:, k],
            shared_covariance=k < 2 or constant_jacobian,
            contraction=controls.shared_gaussian_contraction,
            precision_policy=precision_policy) for k in range(3)]
        densities = stack([part[0] for part in parts], -1)
        lq = logsumexp(densities + tf.math.log(beta)[None, None], -1)
        lr = Jet(lv, lt) + densities[:, :, 0] - lq
        normalizer = logsumexp(lr, 1)
        inc = normalizer - tf.math.log(tf.cast(n, dtype))
        logw = lr - normalizer[:, None]
        valid = proposal_valid & tf.reduce_all(tf.stack([part[1] for part in parts]), 0)
        beta_valid = (tf.reduce_all(tf.math.is_finite(beta)) & tf.reduce_all(beta >= 0)
                      & (beta[0] > 0) & (tf.abs(tf.reduce_sum(beta)-1) < tf.cast(1e-6, dtype)))
        valid = valid & beta_valid & tf.math.is_finite(inc.v) & tf.reduce_all(tf.math.is_finite(inc.t), 0)
        return inc.v, tf.transpose(inc.t), logw.v, logw.t, valid

    @tf.function(input_signature=[cloud, tangent, tensor([b, n]), tensor([p, b, n])],
                 jit_compile=True, autograph=False)
    def reset(xv, xt, lv, lt):
        child, logw = Jet(xv, xt), Jet(lv, lt)
        z, mu, covariance, valid, diagnostics = shared.reset_cloud(
            child, logw, controls, precision_policy)
        correction = tf.zeros([b, 3], dtype)
        if controls.correction_steps:
            from bayesfilter.highdim.covariance_proposal_score_moments_tf import repair_moments
            w = exp(logw)
            parts = [repair_moments(child[row], w[row], z[row], mu[row],
                                   covariance[row], controls) for row in range(b)]
            z, correction = stack([part[0] for part in parts]), tf.stack([part[1] for part in parts])
        return z.v, tf.transpose(z.t, [1, 0, 2, 3]), valid, diagnostics, correction

    return proposal, density, reset


def compare_outputs(stages, reference, tolerance=.01):
    """Fail closed on every row and every final output, not just the scalar."""
    proposal, density, reset = stages
    candidate = dict(value=density[0], score=density[1],
                     particles=reset[0], particle_tangent=reset[1])
    healthy = (bool(tf.reduce_all(reference["valid"]).numpy())
               and bool(tf.reduce_all(reference["steps_completed"] == 1).numpy())
               and bool(tf.reduce_all(proposal[-1] & density[-1] & reset[2]).numpy()))
    differences = {}
    for name, x in candidate.items():
        y = reference[name]
        finite = bool(tf.reduce_all(tf.math.is_finite(x) & tf.math.is_finite(y)).numpy())
        absolute = tf.abs(x-y)
        scaled = absolute / tf.maximum(tf.ones_like(y), tf.abs(y))
        differences[name] = dict(finite=finite,
            maximum_absolute=float(tf.reduce_max(absolute).numpy()),
            maximum_scaled=float(tf.reduce_max(scaled).numpy()))
        healthy = healthy and finite
    # Include materialized internal diagnostics in the validity screen.
    healthy = healthy and all(bool(tf.reduce_all(tf.math.is_finite(x)).numpy())
                              for x in (reset[3], reset[4], density[2], density[3]))
    return dict(healthy=healthy, differences=differences, tolerance=tolerance,
                passed=healthy and all(x["maximum_scaled"] <= tolerance
                                       for x in differences.values()))


def _timed(function, args):
    started = time.perf_counter()
    value = function(*args)
    # Synchronize every output, including stages whose tensors would otherwise
    # be unused by a host-side timer. No NumPy computation enters this path.
    for leaf in tf.nest.flatten(value):
        leaf.numpy()
    return value, time.perf_counter()-started


def run_diagnostic(case, output, repetitions=3):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    if repetitions < 1:
        raise ValueError("positive repetitions required")
    destination = Path(output) / "component-cost.json"
    result = dict(schema="surrogate_hmc_component_cost_v1", status="RUNNING",
        evidence_role="first_observation_unchanged_primitive_cost_diagnostic",
        scope=case.scope(), controls=asdict(case.controls), beta=runtime.json_ready(case.beta),
        design_seed=31001 if case.name=="range_bearing" else 261008301,
        diagnostic_horizon=1, batch_size=2,
        cpu_only=not bool(tf.config.list_physical_devices("GPU")),
        jit_compile=True, posterior_sampling_complete=False,
        timing_includes_materialization=True,
        timing_interpretation="descriptive; staging may change XLA fusion and memory traffic",
        first_call_seconds={}, warmed_seconds={})
    def save():
        destination.write_text(json.dumps(runtime.json_ready(result), indent=2, allow_nan=False))
    save()
    dtype = case.force_dtype
    n = int(case.fixed[0].shape[0])
    theta = tf.stack([case.theta0, case.chart(tf.fill([case.p], tf.cast(.01, dtype)))[0]])
    result["parameter_positions"] = runtime.json_ready(theta)
    result["input_sha256"] = {name: hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()
        for name, value in (("observation", case.observations[0]), ("initial", case.fixed[0]),
                            ("innovations", case.fixed[1][0]), ("uniforms", case.fixed[2][0]))}
    stages = make_stages(case.spec, n, case.controls, dtype, case.precision_policy)
    fused = shared.make_batched_score_filter(case.spec, n, 1, case.controls, dtype,
        precision_policy=case.precision_policy)
    proposal_args = (theta, case.fixed[0], case.observations[0],
                     case.fixed[1][0], case.fixed[2][0], case.beta)
    fused_args = (theta, case.observations[:1], case.fixed[0],
                  case.fixed[1][:1], case.fixed[2][:1], case.beta)
    values = []
    p, elapsed = _timed(stages[0], proposal_args)
    values.append(p); result["first_call_seconds"]["proposal_and_observation"] = elapsed; save()
    density_args = (*p, case.beta, theta)
    q, elapsed = _timed(stages[1], density_args)
    values.append(q); result["first_call_seconds"]["densities_and_weights"] = elapsed; save()
    reset_args = (*p[:2], *q[2:4])
    z, elapsed = _timed(stages[2], reset_args)
    values.append(z); result["first_call_seconds"]["transport_and_moment_reset"] = elapsed; save()
    full, elapsed = _timed(fused, fused_args)
    result["first_call_seconds"]["fused_step"] = elapsed
    result["comparison"] = compare_outputs(values, full)
    result["reset_diagnostics"] = runtime.json_ready(z[3])
    result["correction_diagnostics"] = runtime.json_ready(z[4])
    save()
    if not result["comparison"]["passed"]:
        result["status"] = "COMPONENT_PARITY_FAILED"; save()
        return result
    jobs = (("proposal_and_observation", stages[0], proposal_args),
            ("densities_and_weights", stages[1], density_args),
            ("transport_and_moment_reset", stages[2], reset_args),
            ("fused_step", fused, fused_args))
    result["warmed_seconds"] = {name: [] for name, _, _ in jobs}
    for repeat in range(repetitions):
        # Rotate order to reduce a fixed ordering bias; these remain descriptive.
        for name, function, args in jobs[repeat % 4:] + jobs[:repeat % 4]:
            _, elapsed = _timed(function, args)
            result["warmed_seconds"][name].append(elapsed); save()
    result["trace_counts"] = [f.experimental_get_tracing_count() for f in (*stages, fused)]
    if tf.config.list_physical_devices("GPU"):
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    means = {name: sum(times)/len(times) for name, times in result["warmed_seconds"].items()}
    result["staged_to_fused_ratio"] = sum(v for k, v in means.items() if k != "fused_step")/means["fused_step"]
    result["status"] = "COMPONENT_PARITY_PASS_COST_MEASURED"
    save()
    return result


def run_sequence_diagnostic(case, output, repetitions=3):
    """Host-staged diagnostic across T; the production comparator is fused XLA.

    The Python observation loop deliberately measures materialized boundaries.
    It is instrumentation, never a replacement runtime or training lane.
    """
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import compare_outputs
    if repetitions < 1:
        raise ValueError("positive repetitions required")
    destination = Path(output) / "component-sequence-cost.json"
    n, horizon = int(case.fixed[0].shape[0]), case.horizon
    dtype = case.force_dtype
    theta = tf.stack([case.theta0, case.chart(tf.fill([case.p], tf.cast(.01, dtype)))[0]])
    result = dict(schema="surrogate_hmc_full_sequence_component_cost_v1", status="RUNNING",
        evidence_role="diagnostic_only_materialized_sequence_attribution",
        scope=case.scope(), controls=asdict(case.controls), beta=runtime.json_ready(case.beta),
        parameter_positions=runtime.json_ready(theta), horizon=horizon, batch_size=2,
        cpu_only=not bool(tf.config.list_physical_devices("GPU")), jit_compile=True,
        host_observation_loop="diagnostic instrumentation only; fused runtime has one TF loop",
        timing_interpretation="descriptive; staging changes fusion and memory traffic",
        posterior_sampling_complete=False, first_call_seconds={}, per_time_warmed=[],
        warmed_seconds={"staged_sequence": [], "fused_sequence": []})
    def save():
        destination.write_text(json.dumps(runtime.json_ready(result), indent=2, allow_nan=False))
    args = (theta, case.observations, *case.fixed, case.beta)
    result["input_sha256"] = [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest()
                              for x in args]
    first = make_stages(case.spec, n, case.controls, dtype, case.precision_policy)
    recurrent = make_stages(case.spec, n, case.controls, dtype, case.precision_policy,
                            recursive=True)[0]
    fused = shared.make_batched_score_filter(case.spec, n, horizon, case.controls, dtype,
                                            precision_policy=case.precision_policy)
    names = ("proposal_and_observation", "densities_and_weights", "transport_and_moment_reset")
    def sequence():
        ell, score = tf.zeros([2], dtype), tf.zeros([2, case.p], dtype)
        healthy, completed = tf.ones([2], tf.bool), tf.zeros([2], tf.int32)
        timings, increments, score_increments = [], [], []
        state = case.fixed[0]
        for t in range(horizon):
            proposal = first[0] if t == 0 else recurrent
            p, tp = _timed(proposal, (theta, state, case.observations[t],
                                      case.fixed[1][t], case.fixed[2][t], case.beta))
            q, tq = _timed(first[1], (*p, case.beta, theta))
            z, tz = _timed(first[2], (*p[:2], *q[2:4]))
            row_health = p[-1] & q[-1] & z[2]
            for tensor in (q[0], q[1], z[0], z[1]):
                row_health &= tf.reduce_all(tf.math.is_finite(tensor),
                                             axis=tf.range(1, tf.rank(tensor)))
            healthy &= row_health
            completed += tf.cast(healthy, tf.int32)
            ell, score = ell + q[0], score + q[1]
            increments.append(q[0]); score_increments.append(q[1])
            timings.append(dict(time=t, **dict(zip(names, (tp, tq, tz)))))
            state = tf.concat([z[0][None], tf.transpose(z[1], [1, 0, 2, 3])], 0)
            if not bool(tf.reduce_all(healthy).numpy()):
                break
        return dict(value=ell, score=score, particles=z[0], particle_tangent=z[1],
                    valid=healthy, steps_completed=completed,
                    value_increments=tf.stack(increments, 1),
                    score_trace=tf.stack(score_increments, 1)), timings
    save()
    reference, elapsed = _timed(fused, args)
    result["first_call_seconds"]["fused_sequence"] = elapsed
    started = time.perf_counter()
    staged, compile_timings = sequence()
    result["first_call_seconds"]["staged_sequence"] = time.perf_counter()-started
    result["first_sequence_per_time_seconds"] = compile_timings
    result["comparison"] = compare_outputs(staged, reference, horizon)
    result["reference_health"] = runtime.json_ready(reference["valid"])
    if staged["score_trace"].shape == reference["score_trace"].shape:
        delta = tf.abs(staged["score_trace"]-reference["score_trace"])
        result["per_time_score_maximum_scaled"] = runtime.json_ready(
            tf.reduce_max(delta/tf.maximum(tf.ones_like(delta), tf.abs(reference["score_trace"])), [0, 2]))
    save()
    if not result["comparison"]["passed"]:
        result["status"] = "COMPONENT_SEQUENCE_PARITY_FAILED"; save(); return result
    for repeat in range(repetitions):
        order = ("staged_sequence", "fused_sequence") if repeat % 2 == 0 else ("fused_sequence", "staged_sequence")
        for name in order:
            if name == "staged_sequence":
                started = time.perf_counter(); value, timing = sequence()
                elapsed = time.perf_counter()-started
                result["per_time_warmed"].append(timing)
                if not compare_outputs(value, reference, horizon)["passed"]:
                    result["status"] = "COMPONENT_SEQUENCE_REPEAT_FAILED"; save(); return result
            else:
                _, elapsed = _timed(fused, args)
            result["warmed_seconds"][name].append(elapsed); save()
    result["trace_counts"] = [f.experimental_get_tracing_count() for f in (*first, recurrent, fused)]
    result["stage_mean_seconds"] = {name: sum(row[name] for run in result["per_time_warmed"]
        for row in run)/repetitions for name in names}
    result["staged_to_fused_ratio"] = (sum(result["warmed_seconds"]["staged_sequence"]) /
                                      sum(result["warmed_seconds"]["fused_sequence"]))
    if tf.config.list_physical_devices("GPU"):
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    result["status"] = "COMPONENT_SEQUENCE_PARITY_PASS_COST_MEASURED"
    save(); return result
