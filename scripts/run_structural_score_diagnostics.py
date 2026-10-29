#!/usr/bin/env python3
"""Bounded diagnostics for the nonlinear structural Model B score.

This is diagnostic code.  It calls the shared production recurrence and does
not alter the HMC target or select production controls.  TensorFlow remains
the numerical backend for the production recurrence; NumPy is used only for
the independent Gauss--Hermite reference and reset-isolation nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


GPU_ID = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
CONTROLS = {
    2: dict(flow_steps=16, reset_steps=10, reset_epsilon=1.0,
            correction_steps=1),
    50: dict(flow_steps=16, reset_steps=5, reset_epsilon=1.0,
             correction_steps=1),
}
N = 4032
TRUTH = [math.atanh(.70), math.log(.25), math.log(.80)]
STEPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
ACTIVE_OUTPUT_ROOT = None


def configure(gpu: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID if gpu else "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    if gpu:
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    else:
        memory = {"mode": "CPU_reference_GPU_intentionally_hidden"}
    # The completed campaign's production Model B arm is FP32 with TF32 off.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    return tf, memory


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_list(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "numpy"):
        value = value.numpy()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_list(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_list(v) for v in value]
    return value


def selected_data(source_root: Path, horizon: int):
    base = source_root / "docs/plans/artifacts/structural-tanh-model-b-20261025-01/prepared"
    data_path = base / f"model_b-T{horizon}-final.json"
    design_path = base / f"model_b-T{horizon}-final-design.json"
    if not data_path.exists() or not design_path.exists():
        raise FileNotFoundError(f"missing final data/design: {data_path} {design_path}")
    return load_json(data_path), load_json(design_path), data_path, design_path


def prefix_dataset(data, horizon):
    return dict(data, observations=data["observations"][:horizon],
                states=data["states"][:horizon], horizon=horizon)


def prefix_design(design, horizon):
    return dict(design, innovations=design["innovations"][:horizon],
                uniforms=design["uniforms"][:horizon])


def target(tf, data, design, horizon, controls, dtype=None,
           reset_precision_policy="conditional_fp64", jit_compile=True):
    from bayesfilter.testing.structural_ledh_target import StructuralTarget
    dtype = tf.float32 if dtype is None else dtype
    return StructuralTarget(
        prefix_dataset(data, horizon), prefix_design(design, horizon), controls,
        batch_size=4, jit_compile=jit_compile, dtype=dtype,
        reset_precision_policy=reset_precision_policy,
    )


def repeated_q(tf, q, dtype=None):
    dtype = tf.float32 if dtype is None else dtype
    return tf.broadcast_to(tf.constant(q, dtype)[None, :], [4, 3])


def one_value_score(tf, tgt, q):
    value, score, valid = tgt.likelihood(repeated_q(tf, q, tgt.dtype))
    v = float(value.numpy()[0])
    s = [float(x) for x in score.numpy()[0]]
    good = bool(valid.numpy()[0])
    return v, s, good


def derivative_ladder(tf, data, design, horizon, steps=None, dtype=None,
                      reset_precision_policy="conditional_fp64",
                      jit_compile=True):
    controls = CONTROLS[horizon if horizon in CONTROLS else 2]
    steps = STEPS if steps is None else steps
    tgt = target(tf, data, design, horizon, controls, dtype=dtype,
                 reset_precision_policy=reset_precision_policy,
                 jit_compile=jit_compile)
    value, score, valid = one_value_score(tf, tgt, TRUTH)
    baseline = tgt.program(repeated_q(tf, TRUTH, tgt.dtype), tgt.observations,
                           tgt.design['initial'], tgt.design['innovations'])
    diagnostics = {name: as_list(baseline[name]) for name in (
        'valid', 'steps_completed', 'validity_trace',
        'reset_precision_fallback', 'trace', 'score_trace')}
    rows = []
    for step in steps:
        central = []
        plus_values, minus_values = [], []
        central_ok = True
        for direction in range(3):
            plus = list(TRUTH); minus = list(TRUTH)
            plus[direction] += step; minus[direction] -= step
            vp, _, gp = one_value_score(tf, tgt, plus)
            vm, _, gm = one_value_score(tf, tgt, minus)
            plus_values.append(vp); minus_values.append(vm)
            central.append((vp - vm) / (2.0 * step))
            central_ok = central_ok and gp and gm
        diff = [a - b for a, b in zip(score, central)]
        rows.append(dict(step=step, finite_difference=central, score=score,
                         plus_values=plus_values, minus_values=minus_values,
                         difference=diff,
                         max_abs_difference=max(abs(x) for x in diff),
                         finite_difference_valid=central_ok))
    return dict(horizon=horizon, value=value, score=score, valid=valid,
                rows=rows, controls=controls, dtype=tgt.dtype.name,
                reset_precision_policy=reset_precision_policy,
                baseline_diagnostics=diagnostics, target_signature=tgt.target_signature,
                interpretation="finite-program derivative consistency; not an oracle")


def quadrature_reference(tf, data):
    # The independent reference is deliberately imported only at the diagnostic
    # boundary. It is NumPy-based and never feeds the production recurrence.
    from bayesfilter.testing.structural_reference_diagnostics import (
        quadrature_value_score,
    )
    observations = data["observations"]
    rows = [quadrature_value_score(TRUTH, observations, order=o, step=h)
            for o, h in ((20, 1e-4), (24, 1e-4), (28, 1e-4), (32, 1e-4),
                         (32, 3e-5), (32, 3e-4))]
    return dict(rows=rows, order_refinement=[
        rows[i]["value"] - rows[i-1]["value"] for i in range(1, 4)],
        score_step_refinement=[[a-b for a,b in zip(rows[i]['score'],rows[3]['score'])]
                               for i in (4,5)],
        interpretation="independent short-horizon quadrature; order and FD-step recorded")


def design_replicates(tf, data, horizon, designs):
    controls = CONTROLS[horizon]
    rows = []
    for seed, design in designs:
        tgt = target(tf, data, design, horizon, controls)
        value, score, valid = one_value_score(tf, tgt, TRUTH)
        rows.append(dict(seed=seed, value=value, score=score, valid=valid))
    vectors = [[r["value"], *r["score"]] for r in rows]
    means = [sum(v[i] for v in vectors) / len(vectors) for i in range(4)]
    sds = [statistics.stdev(v[i] for v in vectors) for i in range(4)]
    mcse = [sd / math.sqrt(len(vectors)) for sd in sds]
    return dict(horizon=horizon, rows=rows, mean=means, sample_sd=sds, mcse=mcse,
                particles=N, controls=controls,
                interpretation="fixed-data independent particle-design variation")


def reset_isolation(tf, data, design, order=32):
    """Compare weighted pre-reset and uniform post-reset predictive laws.

    The next-observation predictive density integrates the scalar innovation
    with Gauss-Hermite quadrature. Both arms carry the shared Jet
    tangents; the pre-reset arm retains the normalized importance weights.
    """
    from bayesfilter.highdim.analytical_jet_tf import Jet, exp, log, logsumexp
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import (
        moments, reset_cloud,
    )
    from bayesfilter.highdim.covariance_proposal_score_moments_tf import (
        repair_moments,
    )
    from bayesfilter.highdim.covariance_proposal_tf import (
        ProposalControls, fixed_design_indices,
    )
    from bayesfilter.highdim.structural_completion_proposal_tf import (
        structural_proposal,
    )
    from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec

    controls = ProposalControls(**CONTROLS[2])
    import tensorflow as tf
    q = repeated_q(tf, TRUTH)
    spec = StructuralModelSpec("model_b")
    models = spec.structural_model(q)
    initial = tf.constant(prefix_design(design, 2)["initial"], tf.float32)
    uniforms = tf.constant(prefix_design(design, 2)["uniforms"], tf.float32)
    innovations = tf.constant(prefix_design(design, 2)["innovations"], tf.float32)
    beta = tf.constant(design["beta"], tf.float32)
    x = models.initial_cloud(initial)
    indices = tuple(tf.reshape(v, [2, N]) for v in fixed_design_indices(
        beta, tf.reshape(uniforms, [2 * N, 2]), N))
    child, densities, valid_density, _ = structural_proposal(
        models, x, tf.constant(data["observations"][0], tf.float32), controls,
        beta, uniforms[0], innovations[0], indices=tuple(v[0] for v in indices))
    logq = tf.math.log(tf.cast(beta, tf.float32))[None, None, :]
    log_mixture = logsumexp(densities + logq, axis=-1)
    log_g = models.observation_density(child, tf.constant(data["observations"][0], tf.float32))
    log_ratio = log_g + densities[:, :, 0] - log_mixture
    normalizer = logsumexp(log_ratio, axis=1)
    logw = log_ratio - normalizer[:, None]
    weights = exp(logw)
    post, reset_mu, reset_cov, valid_reset, reset_diag, fallback = reset_cloud(
        child, logw, controls, reset_precision_policy="conditional_fp64",
        return_fallback=True)
    correction = tf.zeros([4, 3], tf.float32)
    if controls.correction_steps:
        post, correction = repair_moments(
            child, weights, post, reset_mu, reset_cov, controls)

    # Gauss-Hermite nodes/weights are fixed constants.  This is a diagnostic
    # expectation of the true transition/observation law, not the LEDH step.
    import numpy as np
    nodes, ghw = np.polynomial.hermite.hermgauss(order)
    nodes = tf.constant(nodes * math.sqrt(2.0), tf.float32)
    ghw = tf.constant(ghw / math.sqrt(math.pi), tf.float32)

    def predictive(parent, observation):
        b = tf.shape(parent.v)[0]
        n = tf.shape(parent.v)[1]
        e = parent.constant(tf.broadcast_to(nodes[None, None, :, None], [b, n, order, 1]))
        repeated_parent = Jet(
            tf.broadcast_to(parent.v[:, :, None, :], [b, n, order, 2]),
            tf.broadcast_to(parent.t[:, :, :, None, :], [3, b, n, order, 2]),
        )
        state = models.complete(repeated_parent, e)
        mean_y = models.observe(state)[..., 0]
        residual = mean_y - observation[0]
        ld = -.5 * (math.log(2 * math.pi * .09) + residual * residual / .09)
        density = (exp(ld) * ghw[None, None, :]).sum(-1)
        return log(density)

    observation1 = tf.constant(data["observations"][1], tf.float32)
    pre = predictive(child, observation1)
    post_pred = predictive(post, observation1)
    pre_joint = log((exp(pre) * weights).sum(-1))
    post_joint = log(exp(post_pred).mean(-1))
    return dict(
        horizon=2, data_observation_index=1, quadrature_order=order,
        valid_density=as_list(valid_density), valid_reset=as_list(valid_reset),
        fallback=as_list(fallback), reset_diagnostics=as_list(reset_diag),
        correction_diagnostics=as_list(correction),
        pre_reset_log_predictive=as_list(pre_joint.v),
        post_reset_log_predictive=as_list(post_joint.v),
        log_predictive_difference=as_list((post_joint - pre_joint).v),
        score_pre_reset=as_list(tf.transpose(pre_joint.t, [1, 0])),
        score_post_reset=as_list(tf.transpose(post_joint.t, [1, 0])),
        score_difference=as_list(tf.transpose((post_joint - pre_joint).t, [1, 0])),
        interpretation=("Reset attribution diagnostic. Pre-reset retains normalized "
                       "weights; post-reset is uniform. It is not an oracle."),
    )


def main():
    global ACTIVE_OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reset-order", type=int, default=32)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reset-only", action="store_true")
    mode.add_argument("--fp64-only", action="store_true")
    mode.add_argument("--t50-only", action="store_true")
    args = parser.parse_args()
    if args.reset_order < 2:
        parser.error("--reset-order must be at least 2")
    if args.fp64_only and not args.cpu:
        parser.error("--fp64-only requires --cpu in this diagnostic")
    started = time.time()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    ACTIVE_OUTPUT_ROOT = root
    tf, memory = configure(not args.cpu)
    from bayesfilter.testing.structural_ledh_models_tf import make_structural_design

    data2, design2, data2_path, design2_path = selected_data(args.source_root, 2)
    data50, design50, data50_path, design50_path = selected_data(args.source_root, 50)
    manifest = dict(schema="structural_score_diagnostic_v1",
                    started_unix=started, git_commit=subprocess.check_output(
                        ['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip(),
                    command=" ".join(sys.argv), particles=N,
                    dtype="float64" if args.fp64_only else "float32",
                    python_executable=sys.executable,tensorflow_version=tf.__version__,
                    script_sha256=digest(Path(__file__)),
                    stage="fp64_derivative" if args.fp64_only else (
                        "t50_derivative" if args.t50_only else (
                        "reset_isolation" if args.reset_only else "full")),
                    tf32=False, gpu=not args.cpu, memory_policy=memory,
                    source_data={"T2": digest(data2_path), "T2_design": digest(design2_path),
                                 "T50": digest(data50_path), "T50_design": digest(design50_path)},
                    controls=CONTROLS)
    write_json(root / "manifest.json", as_list(manifest))
    if args.t50_only:
        result=derivative_ladder(tf,data50,design50,50)
        write_json(root / 'results.json',dict(derivative_T50=result,
                                              elapsed_seconds=time.time()-started))
        print(json.dumps(dict(output=str(root),stage='t50_derivative',
                              elapsed_seconds=time.time()-started)))
        return

    if args.fp64_only:
        if not args.cpu:
            raise ValueError("--fp64-only requires --cpu in this diagnostic")
        steps = [1e-3, 3e-4, 1e-4, 3e-5]
        fp64 = {
            "T1": derivative_ladder(
                tf, data2, design2, 1, steps=steps, dtype=tf.float64,
                reset_precision_policy="uniform", jit_compile=False,
            ),
            "T2": derivative_ladder(
                tf, data2, design2, 2, steps=steps, dtype=tf.float64,
                reset_precision_policy="uniform", jit_compile=False,
            ),
            "interpretation": (
                "FP64 diagnostic uses the same fixed data/design and controls. "
                "The structural FP32 conditional-fallback policy is unavailable "
                "for an FP64 recurrence; uniform reset is identical on the "
                "healthy FP32 rows because no fallback fired."
            ),
        }
        write_json(root / "results.json", as_list({
            "fp64_derivative": fp64,
            "elapsed_seconds": time.time() - started,
        }))
        print(json.dumps({"output": str(root),
                          "elapsed_seconds": time.time() - started,
                          "stage": "fp64_derivative"}, indent=2))
        return

    if args.reset_only:
        # This mode is intentionally CPU-only for the explanatory reset
        # diagnostic; it avoids repeating completed derivative/design work.
        with tf.device("/CPU:0"):
            reset_graph = tf.function(
                lambda: reset_isolation(tf, data2, design2,args.reset_order),
                # Graph mode preserves TensorArray read semantics.  XLA is
                # deliberately disabled here because the explanatory return
                # dictionary contains a Python string; production score and
                # reset kernels remain XLA-compiled on GPU.
                autograph=False, jit_compile=False,
            )
            reset_result = as_list(reset_graph())
        write_json(root / "results.json", {"reset_isolation_T2": reset_result,
                                            "elapsed_seconds": time.time() - started})
        print(json.dumps({"output": str(root),
                          "elapsed_seconds": time.time() - started,
                          "stage": "reset_isolation"}, indent=2))
        return

    results = {}
    def checkpoint(stage):
        write_json(root / "partial_results.json", dict(stage=stage,
                                                        results=results))

    results["derivative_T1"] = derivative_ladder(tf, data2, design2, 1)
    results["derivative_T2"] = derivative_ladder(tf, data2, design2, 2)
    results["derivative_T50"] = derivative_ladder(tf, data50, design50, 50)
    results["quadrature_T2"] = quadrature_reference(tf, data2)
    checkpoint("derivatives_and_quadrature")

    # TensorFlow's stateless random CUDA kernel is unreliable on the RTX 5080
    # with this environment after the first XLA cluster is initialized.  The
    # design is fixed data, so generate it on CPU and leave the score/reset
    # kernels on the selected device.  This changes no scientific quantity.
    with tf.device("/CPU:0"):
        design_rows2 = [(seed, as_list(make_structural_design(N, 2, seed)))
                        for seed in (91001, 91002, 91003, 91004)]
        design_rows50 = [(seed, as_list(make_structural_design(N, 50, seed)))
                         for seed in (92001, 92002, 92003, 92004)]
    results["design_replication_T2"] = design_replicates(tf, data2, 2, design_rows2)
    results["design_replication_T50"] = design_replicates(tf, data50, 50, design_rows50)
    checkpoint("design_replication")
    # The production score is measured on GPU.  This explanatory diagnostic
    # contains Jet algebra that triggers an invalid CUDA handle on the RTX 5080
    # in this TensorFlow build, so run it deliberately on CPU.  Its target is
    # unchanged; only the device is different.
    with tf.device("/CPU:0"):
        reset_graph = tf.function(
            lambda: reset_isolation(tf, data2, design2,args.reset_order),
            autograph=False, jit_compile=False,
        )
        results["reset_isolation_T2"] = as_list(reset_graph())
    checkpoint("reset_isolation")
    results["elapsed_seconds"] = time.time() - started
    write_json(root / "results.json", as_list(results))
    print(json.dumps({
        "output": str(root),
        "elapsed_seconds": results["elapsed_seconds"],
        "derivative_T2_max": max(r["max_abs_difference"] for r in results["derivative_T2"]["rows"]),
        "design_T2_mcse": results["design_replication_T2"]["mcse"],
        "design_T50_mcse": results["design_replication_T50"]["mcse"],
    }, indent=2))


if __name__ == "__main__":
    launched = time.time()
    status = 'completed'
    failure = None
    try:
        main()
    except BaseException as exc:
        status = 'failed'
        failure = dict(type=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc())
        raise
    finally:
        if ACTIVE_OUTPUT_ROOT is not None:
            write_json(ACTIVE_OUTPUT_ROOT / 'execution.json',
                       dict(status=status, elapsed_seconds=time.time()-launched,
                            command=sys.argv, failure=failure))
