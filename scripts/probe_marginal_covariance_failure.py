#!/usr/bin/env python3
"""Diagnostic only: capture actual first-observation mixture covariances.

No candidate numerics or safety rule changes. The shorter graph consumes the
first observation and innovation from the exact calibration T10 design.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="2")
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Trusted GPU access required")
    from bayesfilter.testing import marginal_weight_campaign as base
    from bayesfilter.testing import marginal_expanded as ex
    from bayesfilter.highdim import ledh_marginal_weights_tf as mixture
    from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky

    started = time.monotonic()
    original_density = mixture.gaussian_mixture_log_density_tangent
    original_ratio = mixture.marginal_prior_ratio_tangent
    rows = []

    def describe(covariance):
        valid, _ = safe_cholesky(covariance, "diagnostic_captured_covariance")
        scale = tf.reduce_max(tf.abs(covariance), axis=(-2, -1))
        asym = tf.reduce_max(tf.abs(covariance - tf.linalg.matrix_transpose(covariance)),
                             axis=(-2, -1)) / scale
        symmetric = .5 * (tf.cast(covariance, tf.float64)
                         + tf.linalg.matrix_transpose(tf.cast(covariance, tf.float64)))
        eig = tf.linalg.eigvalsh(symmetric)
        return base.serial(dict(
            all_finite=tf.reduce_all(tf.math.is_finite(covariance)),
            valid_fraction=tf.reduce_mean(tf.cast(valid, tf.float64)),
            maximum_relative_asymmetry=tf.reduce_max(asym),
            symmetry_tolerance=tf.constant(8 * covariance.shape[-1] * 2**-23, tf.float64),
            minimum_symmetric_eigenvalue=tf.reduce_min(eig),
            minimum_symmetric_relative_eigenvalue=tf.reduce_min(eig[:, 0] / eig[:, -1])))

    try:
        for tf32 in (True, False):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            for dim in (10, 20):
                spec = base.specification(f"lg{dim}-t10")
                short = dict(spec, t=1)
                observed = tf.cast(base.dataset(spec, "calibration", dataset_id=1), tf.float32)[:1]
                design = base.designs(spec, args.n, "calibration", 0, dataset_id=1)
                design = (design[0], design[1][:1], design[2])
                mixture.gaussian_mixture_log_density_tangent = original_density
                mixture.marginal_prior_ratio_tangent = original_ratio
                untouched = ex.compile_batch(short, args.n, base.controls(0), "marginal_mixture")
                original_rows = [base.serial(untouched(
                    tf.constant([theta], tf.float32), *design, observed))
                    for theta in spec["points"]]
                captured = [tf.Variable(tf.zeros([args.n, dim, dim]), trainable=False)
                            for _ in range(3)]
                flags = [tf.Variable(tf.zeros([2], tf.bool), trainable=False)
                         for _ in range(2)]
                trace_counter = [0]

                def density(*inputs):
                    result = original_density(*inputs)
                    index = trace_counter[0] % 2
                    trace_counter[0] += 1
                    writes = (captured[index].assign(inputs[4]),
                              flags[index].assign(tf.stack(result[2:])))
                    with tf.control_dependencies(writes):
                        return tuple(tf.identity(x) for x in result)

                def ratio(*inputs):
                    result = original_ratio(*inputs)
                    with tf.control_dependencies([captured[2].assign(inputs[8])]):
                        return tuple(tf.identity(x) for x in result)

                mixture.gaussian_mixture_log_density_tangent = density
                mixture.marginal_prior_ratio_tangent = ratio
                evaluate = ex.compile_batch(short, args.n, base.controls(0), "marginal_mixture")
                for point, theta in enumerate(spec["points"]):
                    call_started = time.monotonic()
                    result = base.serial(evaluate(tf.constant([theta], tf.float32), *design, observed))
                    prior, proposal, b = [v.read_value() for v in captured]
                    b64, q64 = tf.cast(b, tf.float64), tf.cast(prior, tf.float64)
                    reference = tf.cast(tf.matmul(tf.matmul(b64, q64), b64, transpose_b=True), tf.float32)
                    row = dict(scope=spec["name"], prefix_horizon=1, n=args.n, point=point,
                               tf32=tf32, result=result, untouched_result=original_rows[point],
                               capture_status_matches=(result["valid"] == original_rows[point]["valid"]),
                               mixture_flags=[base.serial(v.read_value()) for v in flags],
                               prior=describe(prior), proposal=describe(proposal),
                               fp64_product_cast_fp32=describe(reference),
                               maximum_scaled_product_discrepancy=base.serial(
                                   tf.reduce_max(tf.abs(proposal-reference))/(1+tf.reduce_max(tf.abs(reference)))),
                               wall_seconds=time.monotonic()-call_started)
                    rows.append(row)
                    base.save(args.output/"progress.json", dict(rows=rows))
                    print("d",dim,"point",point,"TF32",tf32,
                          "valid",result["valid"],
                          "proposal_asym",row["proposal"]["maximum_relative_asymmetry"],
                          "tol",row["proposal"]["symmetry_tolerance"],
                          "fp64_product_valid",row["fp64_product_cast_fp32"]["valid_fraction"],flush=True)
    finally:
        mixture.gaussian_mixture_log_density_tangent = original_density
        mixture.marginal_prior_ratio_tangent = original_ratio
    base.save(args.output/"result.json", dict(
        status="COMPLETE", rows=rows, wall_seconds=time.monotonic()-started,
        command=sys.argv, environment=dict(python=sys.executable,tensorflow=tf.__version__,
            gpu=os.environ["CUDA_VISIBLE_DEVICES"], memory_policy=memory, jit_compile=True,
            tf32_arms=[True,False], cpu_only=False),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        seeds="T10 calibration dataset 1, particle design 0, control 0",
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/"bayesfilter/highdim/ledh_marginal_weights_tf.py",
             ROOT/"bayesfilter/highdim/ledh_canonical_score_tf.py")},
        nonclaim="Diagnostic attribution only. No changed candidate, accuracy ranking, or HMC inference."))


if __name__ == "__main__":
    main()
