"""Identical-input precision localization using the actual shared filter stages.

Diagnostic only: the host observation loop materializes stages and changes XLA
fusion. Fused comparisons are mandatory before attributing a fused failure to a
staged trajectory. This module does not select controls or change runtime paths.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_component_cost_diagnostic import make_stages
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared

PROPOSAL_FIELDS = (("child", 0), ("child_tangent", 1), ("means", 0),
                   ("means_tangent", 1), ("covariance", 0),
                   ("covariance_tangent", 1), ("observation_density", 0),
                   ("observation_score", 1))
DENSITY_FIELDS = (("increment", 0), ("score", 0), ("log_weights", 0),
                  ("log_weight_tangent", 1))
RESET_FIELDS = (("particles", 0), ("particle_tangent", 0))


def cast_inputs(inputs, dtype):
    return tuple(tf.cast(x, dtype) if x.dtype.is_floating else x for x in inputs)


def difference(x, y, batch_axis=0):
    """Arithmetic diagnostics only; scaling is the owner's final-output rule."""
    x, y = tf.cast(x, tf.float64), tf.cast(y, tf.float64)
    error = tf.abs(x-y)
    axes = [i for i in range(x.shape.rank) if i != batch_axis]
    scaled = error / tf.maximum(tf.ones_like(y), tf.abs(y))
    return dict(finite=bool(tf.reduce_all(tf.math.is_finite(x) & tf.math.is_finite(y)).numpy()),
        maximum_absolute=tf.reduce_max(error), maximum_scaled=tf.reduce_max(scaled),
        per_row_maximum_absolute=tf.reduce_max(error, axes),
        per_row_maximum_scaled=tf.reduce_max(scaled, axes),
        relative_l2=tf.sqrt(tf.reduce_sum(error*error, axes)) /
                    tf.maximum(tf.ones([x.shape[batch_axis]], tf.float64),
                               tf.sqrt(tf.reduce_sum(y*y, axes))))


def compare(values, reference, fields):
    return {name: difference(x, y, axis)
            for (name, axis), x, y in zip(fields, values, reference)}


def _save_tensors(directory, names, values):
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, value in zip(names, values):
        data = tf.io.serialize_tensor(value).numpy()
        path = directory / (name + ".tensor")
        path.write_bytes(data)
        saved[name] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                           dtype=value.dtype.name, shape=value.shape.as_list())
    return saved


def run_diagnostic(case, output, horizon=10):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    if case.precision_policy != "uniform" or not 1 <= horizon <= case.horizon:
        raise ValueError("localization requires uniform arithmetic and a saved-data prefix")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "stage-arithmetic.json"
    n, p, d = int(case.fixed[0].shape[0]), case.p, case.spec.dimension
    theta = tf.stack([tf.cast(case.theta0, tf.float32),
                     tf.cast(case.chart(tf.fill([p], tf.constant(.01, tf.float32)))[0], tf.float32)])
    obs = tf.cast(case.observations[:horizon], tf.float32)
    fixed = tuple(tf.cast(x, tf.float32) for x in
                  (case.fixed[0], case.fixed[1][:horizon], case.fixed[2][:horizon]))
    beta = tf.cast(case.beta, tf.float32)
    base_inputs = (theta, obs, *fixed, beta)
    result = dict(schema="surrogate_hmc_stage_arithmetic_v1", status="RUNNING",
        evidence_role="identical_input_primitive_arithmetic_localization_only",
        model=case.name, particles=n, horizon=horizon, state_dimension=d,
        parameter_count=p, controls=asdict(case.controls), beta=beta,
        parameter_positions=theta, cpu_only=not bool(tf.config.list_physical_devices("GPU")),
        jit_compile=True, host_observation_loop="diagnostic instrumentation only",
        posterior_complete=False, local_inputs="FP64 obtained by exact casting of FP32 stage inputs",
        input_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in base_inputs],
        stages=[], fused_comparison={})
    def save():
        runtime.save(destination, result)
    save()
    started = time.perf_counter()
    funcs = {}
    for dtype in (tf.float32, tf.float64):
        funcs[dtype] = (*make_stages(case.spec, n, case.controls, dtype),
                       make_stages(case.spec, n, case.controls, dtype, recursive=True)[0])
    state32 = fixed[0]
    state64 = tf.cast(fixed[0], tf.float64)
    cumulative = {dtype: (tf.zeros([2], dtype), tf.zeros([2, p], dtype))
                  for dtype in (tf.float32, tf.float64)}
    healthy = True
    for t in range(horizon):
        proposal_args32 = (theta, state32, obs[t], fixed[1][t], fixed[2][t], beta)
        # The independent FP64 recurrence has its own state; all other inputs
        # are exactly the same representable numbers as the FP32 execution.
        proposal_args64 = (tf.cast(theta, tf.float64), state64,
                          *cast_inputs(proposal_args32[2:], tf.float64))
        proposal_index = 0 if t == 0 else 3
        a32 = funcs[tf.float32][proposal_index](*proposal_args32)
        a64local = funcs[tf.float64][proposal_index](*cast_inputs(proposal_args32, tf.float64))
        a64 = funcs[tf.float64][proposal_index](*proposal_args64)
        local = dict(proposal=compare(a32, a64local, PROPOSAL_FIELDS))
        valid = dict(proposal32=a32[-1], proposal64_local=a64local[-1], proposal64=a64[-1])
        del a64local
        density_args32 = (*a32, beta, theta)
        q32 = funcs[tf.float32][1](*density_args32)
        q64local = funcs[tf.float64][1](*cast_inputs(density_args32, tf.float64))
        q64 = funcs[tf.float64][1](*a64, tf.cast(beta, tf.float64), tf.cast(theta, tf.float64))
        density_input_note = dict(original_fp64_valid=q64local[-1], symmetrized_for_local_pair=False)
        q32local = q32
        if not bool(tf.reduce_all(q64local[-1]).numpy()):
            # A cast retains FP32 antisymmetric roundoff, which can violate
            # FP64's stricter input guard. Compare BOTH precisions at a common
            # exactly symmetric input. Never modify the propagated trajectory.
            common = list(density_args32)
            for index in (4, 5):
                common[index] = .5*(common[index] + tf.linalg.matrix_transpose(common[index]))
            q32local = funcs[tf.float32][1](*common)
            q64local = funcs[tf.float64][1](*cast_inputs(common, tf.float64))
            density_input_note.update(symmetrized_for_local_pair=True,
                covariance_input_change=difference(common[4], density_args32[4]),
                covariance_tangent_input_change=difference(common[5], density_args32[5], 1),
                fp32_output_change=compare(q32local, q32, DENSITY_FIELDS),
                common_input_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in common])
        local["density"] = compare(q32local, q64local, DENSITY_FIELDS)
        valid.update(density32=q32[-1], density32_local=q32local[-1],
                     density64_local=q64local[-1], density64=q64[-1])
        del q64local, q32local
        reset_args32 = (*a32[:2], *q32[2:4])
        z32 = funcs[tf.float32][2](*reset_args32)
        z64local = funcs[tf.float64][2](*cast_inputs(reset_args32, tf.float64))
        z64 = funcs[tf.float64][2](*a64[:2], *q64[2:4])
        local["reset"] = compare(z32, z64local, RESET_FIELDS)
        valid.update(reset32=z32[2], reset64_local=z64local[2], reset64=z64[2])
        del z64local
        for dtype, q in ((tf.float32, q32), (tf.float64, q64)):
            v, score = cumulative[dtype]
            cumulative[dtype] = (v+q[0], score+q[1])
        trajectory = compare(cumulative[tf.float32], cumulative[tf.float64], (("value", 0), ("score", 0)))
        step_healthy = (all(bool(tf.reduce_all(x).numpy()) for x in valid.values())
                        and all(item["finite"] for stage in local.values() for item in stage.values())
                        and all(item["finite"] for item in trajectory.values()))
        step = dict(observation=t+1, healthy=step_healthy, local=local,
                    independent_trajectory=trajectory, validity=valid, density_input_note=density_input_note,
                    increments32=q32[:2], increments64=q64[:2],
                    reset_diagnostics32=z32[3], reset_diagnostics64=z64[3],
                    saved_reset_inputs=_save_tensors(output / ("step-%02d" % (t+1)),
                        ("child", "child_tangent", "log_weights", "log_weight_tangent"), reset_args32),
                    saved_proposal_inputs=_save_tensors(output / ("step-%02d" % (t+1)),
                        ("theta", "initial_or_recursive_state", "observation", "innovations", "uniforms", "beta"), proposal_args32))
        result["stages"].append(step)
        healthy = healthy and step_healthy
        result["wall_seconds"] = time.perf_counter()-started
        save()
        state32 = tf.concat([z32[0][None], tf.transpose(z32[1], [1, 0, 2, 3])], 0)
        state64 = tf.concat([z64[0][None], tf.transpose(z64[1], [1, 0, 2, 3])], 0)
        if not step_healthy:
            break
    if healthy:
        for dtype, z in ((tf.float32, z32), (tf.float64, z64)):
            fused = shared.make_batched_score_filter(case.spec, n, horizon, case.controls, dtype)
            reference = fused(*cast_inputs(base_inputs, dtype))
            v, score = cumulative[dtype]
            candidate = (v, score, z[0], z[1])
            fields = (("value", 0), ("score", 0), ("particles", 0), ("particle_tangent", 0))
            comp = compare(candidate, tuple(reference[name] for name, _ in fields), fields)
            valid = (bool(tf.reduce_all(reference["valid"]).numpy()) and
                     bool(tf.reduce_all(reference["steps_completed"] == horizon).numpy()))
            passed = valid and all(x["finite"] and float(x["maximum_scaled"].numpy()) <= .01 for x in comp.values())
            result["fused_comparison"][dtype.name] = dict(passed=passed, valid=valid,
                differences=comp, staged_value=v, staged_score=score,
                fused_value=reference["value"], fused_score=reference["score"],
                fused_score_trace=reference["score_trace"],
                trace_count=fused.experimental_get_tracing_count())
            save()
        result["staging_preserves_fused_outputs"] = all(x["passed"] for x in result["fused_comparison"].values())
    result["trace_counts"] = {dtype.name: [f.experimental_get_tracing_count() for f in fs]
                              for dtype, fs in funcs.items()}
    result["status"] = "STAGE_ARITHMETIC_LOCALIZED" if healthy else "STAGE_ARITHMETIC_INVALID"
    result["wall_seconds"] = time.perf_counter()-started
    save()
    return result
