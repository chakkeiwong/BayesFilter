"""Read-only numerical instrumentation for an ordinary preparation replay.

This diagnostic adds trace outputs to TFP without modifying its transition or
seed stream. It cannot issue a tuning artifact or recover a failed window.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path


def install(output):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability.python.mcmc.internal import leapfrog_integrator
    from tensorflow_probability.python.mcmc.internal import util as mcmc_util
    from bayesfilter.inference import hmc_warmup as warmup
    from bayesfilter.testing.inference_validation.storage import write_json

    output = Path(output)
    output.mkdir(exist_ok=False)
    original_chain = tfp.mcmc.sample_chain
    original_validate = warmup._validate_operational_window_trace
    original_target = warmup.reviewed_value_score_target_fn
    adapters = {}
    runners = []
    windows = []

    def write_tensor(path, value):
        value = tf.convert_to_tensor(value)
        raw = tf.io.serialize_tensor(value).numpy()
        with Path(path).open("xb") as stream:
            stream.write(raw)
        write_json(str(path) + ".json", {"sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": value.dtype.name, "shape": value.shape.as_list()})

    def target_factory(adapter, **kwargs):
        target = original_target(adapter, **kwargs)
        adapters[target] = adapter
        return target

    def chain(*args, **kwargs):
        original_trace = kwargs.get("trace_fn")
        kernel = kwargs.get("kernel")
        target = getattr(getattr(kernel, "inner_kernel", None), "target_log_prob_fn", None)
        if target not in adapters:
            return original_chain(*args, **kwargs)
        index = len(runners)
        runners.append((target, adapters[target]))

        def trace(state, results):
            fields = dict(original_trace(state, results))
            inner = results.inner_results
            proposal = inner.proposed_results
            fields["diagnostic"] = {
                "runner": tf.constant(index, tf.int32),
                "accepted_state": state,
                "accepted_score": inner.accepted_results.grads_target_log_prob[0],
                "proposed_state": inner.proposed_state,
                "proposed_target": proposal.target_log_prob,
                "proposed_score": proposal.grads_target_log_prob[0],
                "initial_momentum": proposal.initial_momentum[0],
                "final_momentum": proposal.final_momentum[0],
                "log_acceptance_correction": proposal.log_acceptance_correction,
                "seed": proposal.seed,
                "initial_state": kwargs["current_state"],
                "initial_score": kwargs["previous_kernel_results"].inner_results.accepted_results.grads_target_log_prob[0],
                "leapfrog_steps": proposal.num_leapfrog_steps,
            }
            return fields

        return original_chain(*args, **{**kwargs, "trace_fn": trace})

    def serialized(value):
        if isinstance(value, dict):
            return {key: serialized(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serialized(item) for item in value]
        if tf.is_tensor(value):
            return serialized(value.numpy().tolist())
        if isinstance(value, float):
            import math
            return value if math.isfinite(value) else str(value)
        return value

    def finite(value):
        return bool(tf.reduce_all(tf.math.is_finite(value)))

    def diagnose(trace, row, dest):
        diagnostic = trace["diagnostic"]
        target, adapter = runners[int(diagnostic["runner"][row])]
        q = diagnostic["initial_state"][row] if row == 0 else diagnostic["accepted_state"][row - 1]
        score = diagnostic["initial_score"][row] if row == 0 else diagnostic["accepted_score"][row - 1]
        momentum = diagnostic["initial_momentum"][row]
        epsilon = trace["consumed_step_size"][row]
        steps = int(diagnostic["leapfrog_steps"][row])
        dim = int(q.shape[0])

        @tf.function(input_signature=[tf.TensorSpec([dim], tf.float64)] * 3
                     + [tf.TensorSpec([], tf.float64)], jit_compile=True, autograph=False)
        def replay(position, p, initial_score, step):
            half = p + .5 * step * initial_score
            arrays = tuple(tf.TensorArray(tf.float64, size=steps) for _ in range(4))

            def body(i, position, half, qs, ps, vs, gs):
                position = position + step * half
                value, grads = mcmc_util.maybe_call_fn_and_grads(target, [position])
                half = half + step * grads[0]
                return (i + 1, position, half, qs.write(i, position), ps.write(i, half),
                        vs.write(i, value), gs.write(i, grads[0]))

            result = tf.while_loop(lambda i, *_: i < steps, body,
                                   (tf.constant(0), position, half, *arrays))
            qs, ps, vs, gs = tuple(a.stack() for a in result[3:])
            return qs, ps, vs, gs, ps[-1] - .5 * step * gs[-1]

        qs, ps, vs, gs, final_p = replay(q, momentum, score, epsilon)
        saved = {"position": qs, "half_momentum": ps, "target": vs, "score": gs,
                 "final_momentum": final_p}
        for name, value in saved.items():
            write_tensor(dest / (name + ".tensor"), value)
        first = None
        for substep in range(steps):
            for field in ("position", "target", "score", "half_momentum"):
                if not finite(saved[field][substep]):
                    first = {"leapfrog_index": substep, "field": field}
                    break
            if first is not None:
                break
        # A finite neighbor provides an endpoint arithmetic check independent
        # of the nonfinite masks on the failed trajectory.
        comparisons = {}
        for name, actual, expected in (
            ("position", qs[-1], diagnostic["proposed_state"][row]),
            ("score", gs[-1], diagnostic["proposed_score"][row]),
            ("momentum", final_p, diagnostic["final_momentum"][row]),
            ("target", vs[-1], diagnostic["proposed_target"][row]),
        ):
            same = tf.logical_or(tf.equal(actual, expected),
                                 tf.logical_and(tf.math.is_nan(actual), tf.math.is_nan(expected)))
            comparisons[name] = {"equal_including_nan": bool(tf.reduce_all(same)),
                "finite_mask_equal": bool(tf.reduce_all(tf.equal(tf.math.is_finite(actual), tf.math.is_finite(expected)))),
                "max_absolute_difference": float(tf.reduce_max(tf.abs(actual - expected))) if finite(actual) and finite(expected) else None}
        # Preserve each affine layer needed to interpret the recorded position.
        layers = []
        active = adapter
        for _ in range(8):
            layers.append({"class": type(active).__name__,
                           "attributes": sorted(vars(active))})
            if hasattr(active, "transform"):
                layers[-1]["transform"] = {"factor": tf.convert_to_tensor(active.transform.factor),
                                            "center": tf.convert_to_tensor(active.transform.center)}
            active = getattr(active, "base_adapter", None)
            if active is None:
                break
        record = {"transition_index": row, "epsilon": epsilon, "L": steps,
            "initial_position": q, "initial_score": score, "initial_momentum": momentum,
            "accepted": trace["is_accepted"][row], "log_accept_ratio": trace["log_accept_ratio"][row],
            "endpoint": {key: value[row] for key, value in diagnostic.items()},
            "first_nonfinite": first, "endpoint_comparison": comparisons, "affine_layers": layers,
            "trajectory": {key: value for key, value in saved.items()},
            "diagnostic_only": True, "candidate_authority": False}
        write_json(dest / "trajectory.json", serialized(record))

    def validate(**kwargs):
        trace = kwargs["trace"]
        if "diagnostic" not in trace:
            raise ValueError("diagnostic replay did not receive proposal telemetry")
        number = len(windows)
        dest = output / f"window-{number:02d}"
        dest.mkdir()
        for name, value in trace.items():
            if name == "diagnostic":
                for key, item in value.items():
                    write_tensor(dest / ("proposal-" + key + ".tensor"), item)
            else:
                write_tensor(dest / (name + ".tensor"), value)
        indices = tf.reshape(tf.where(~tf.math.is_finite(trace["log_accept_ratio"])), [-1]).numpy().tolist()
        record = {"window_index": number, "draw_count": kwargs["expected_draw_count"],
                  "step_ceiling": kwargs["step_size_upper_bound"], "failed_indices": indices,
                  "retained_state_finite": finite(trace["diagnostic"]["accepted_state"]),
                  "retained_score_finite": finite(trace["diagnostic"]["accepted_score"]),
                  "retained_target_finite": finite(trace["target_log_prob"])}
        windows.append(record)
        write_json(output / "windows.json", windows)
        for row in sorted(set(indices + ([indices[0] - 1] if indices and indices[0] > 0 else []))):
            trajectory = dest / f"transition-{row:04d}"
            trajectory.mkdir()
            diagnose(trace, row, trajectory)
        return original_validate(**kwargs)

    tfp.mcmc.sample_chain = chain
    warmup.reviewed_value_score_target_fn = target_factory
    warmup._validate_operational_window_trace = validate
    sources = {}
    for module in (leapfrog_integrator, mcmc_util, inspect.getmodule(tfp.mcmc.HamiltonianMonteCarlo)):
        path = Path(module.__file__)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "instrumentation.json", {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tfp_version": tfp.__version__, "tensorflow_version": tf.__version__,
        "sources": sources, "target_and_transition_unchanged": True,
        "extra_trace_may_change_compiler_fusion": True,
    })
