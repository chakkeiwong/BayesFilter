"""Matched-array precision diagnostics; not learned-map quality evidence."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp64", "fp32", "tf32"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="One physical GPU index; set before TensorFlow import")
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        parser.error("gpu must be one physical device index")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = {"mode": args.mode, "scope": "engineering_precision_only", "command": sys.argv,
        "python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "arms": [], "passed": False}
    try:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
            raise RuntimeError("memory growth must precede framework import")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        result["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(args.mode == "tf32")
        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.neutra_transport import (
            NeuTraTransport, NeuTraTransportConfig, NeuTraTransportTrainer, NeuTraOptimizerConfig)
        from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
        result.update(tensorflow=tf.__version__, tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
                      transport_dtype="float64" if args.mode == "fp64" else "float32", target_dtype="float64",
                      jit_compile=True, batch_size=32, seeds=[247, 19, 73, 91],
                      cpu_reference="small analytic float64 equation check; not CPU training")
        dtype = tf.as_dtype(result["transport_dtype"])

        def target(x):
            if x.dtype != tf.float64:
                raise ValueError("target precision changed")
            scale = tf.constant([.7, 1.3, 2., .5], tf.float64)
            center = tf.constant([.2, -.1, .3, -.2], tf.float64)
            value = -.5*tf.reduce_sum(tf.square((x-center)/scale), axis=-1)-.02*tf.reduce_sum(x**4, axis=-1)
            score = -(x-center)/tf.square(scale)-.08*x**3
            return value, score, tf.ones(tf.shape(x)[0], tf.bool)

        with tf.device("/CPU:0"):
            # Identical representable inputs across all precision modes.
            base_rows = tf.random.stateless_normal([32, 4], [73, 91], dtype=tf.float32)
        for kind in ("iaf", "naf_dsf"):
            base_config = NeuTraTransportConfig(4, kind, (16, 16), 3, "elu", (247, 19), 2.,
                mixture_components=8, final_weight_scale=.05,
                mask_policy="hoffman_block_masks_v1" if kind == "iaf" else "legacy_degree_masks_v1")
            for estimator in ("standard", "path"):
                opt = NeuTraOptimizerConfig(32, estimator, .001, .9, .999, 1.e-7, None)
                with tf.device("/CPU:0"):
                    initial = NeuTraTransport(base_config)
                    if kind == "naf_dsf":
                        for stage in initial.stages:
                            for weight in stage.weights[:-1]:
                                weight.assign(weight*100.)
                    rounded = initial.as_dtype(dtype)
                    reference = rounded.as_dtype("float64", trainable=True)
                    expected = NeuTraTransportTrainer(reference, target, opt, target_signature="a"*64)._evaluate(
                        tf.cast(base_rows, tf.float64))
                with tf.device("/GPU:0"):
                    flow = NeuTraTransport(replace(base_config, dtype=dtype.name, inverse_atol=None, inverse_rtol=None))
                    flow.restore_parameters(rounded.parameter_state())
                    trainer = NeuTraTransportTrainer(flow, target, opt, target_signature="a"*64)
                    x = tf.cast(base_rows, dtype)
                    started = time.monotonic()
                    actual = trainer.evaluate(x)
                    actual["loss"].numpy()
                    compile_evaluation = time.monotonic()-started
                    difference = tf.concat([tf.reshape(tf.cast(g, tf.float64)-r, [-1])
                        for g, r in zip(actual["gradients"], expected["gradients"], strict=True)], axis=0)
                    reference_gradient = tf.concat([tf.reshape(g, [-1]) for g in expected["gradients"]], axis=0)
                    gradient_error = float((tf.linalg.norm(difference)/(1.+tf.linalg.norm(reference_gradient))).numpy())
                    loss_error = float((tf.abs(actual["loss"]-expected["loss"])/(1.+tf.abs(expected["loss"]))).numpy())
                    limit = {"fp64": 1.e-9, "fp32": 1.e-4, "tf32": .01}[args.mode]
                    if not bool(actual["valid"].numpy()) or max(gradient_error, loss_error) > limit:
                        raise RuntimeError(f"{kind}/{estimator}: precision mismatch {gradient_error}, {loss_error}")
                    started = time.monotonic()
                    first = trainer.train_step(x)
                    first["loss"].numpy()
                    compile_train = time.monotonic()-started
                    started = time.monotonic()
                    for _ in range(7):
                        final = trainer.train_step(x)
                        if not bool(final["valid"].numpy()):
                            raise RuntimeError("invalid analytic update")
                    steady = (time.monotonic()-started)/7
                    @tf.function(input_signature=[tf.TensorSpec([32, 4], dtype)], jit_compile=True)
                    def roundtrip(z):
                        y, ld = flow.forward_and_logdet(z)
                        inverse, ild = flow.inverse_and_forward_logdet(y)
                        return tf.reduce_max(tf.abs(inverse-z)/(1.+tf.abs(z))), tf.reduce_max(tf.abs(ild-ld))
                    # Tail and center reconstruction exercise trained parameters.
                    tail = x*10.
                    inverse_error, ld_error = [float(v.numpy()) for v in roundtrip(tail)]
                    if not all(__import__("math").isfinite(v) for v in (inverse_error, ld_error)) or inverse_error > (1.e-7 if args.mode == "fp64" else 2.e-3):
                        raise RuntimeError(f"inverse precision failure: {inverse_error}, {ld_error}")
                    frozen = trainer.transport.as_dtype("float64").frozen_payload(target_signature="a"*64)
                    restored = load_frozen_neutra_artifact(frozen, expected_target_signature="a"*64).transport
                    @tf.function(input_signature=[tf.TensorSpec([32, 4], tf.float64)], jit_compile=True)
                    def frozen_check(z):
                        y, ld = restored.forward_and_logdet(z)
                        back, _ = restored.inverse_and_forward_logdet(y)
                        return tf.reduce_max(tf.abs(back-z)), tf.reduce_all(tf.math.is_finite(ld))
                    frozen_error, frozen_valid = frozen_check(tf.cast(x, tf.float64))
                    if not bool(frozen_valid.numpy()) or float(frozen_error.numpy()) > 1.e-7:
                        raise RuntimeError("FP64 exported map failed reconstruction")
                    hlo = trainer.train_step.experimental_get_compiler_ir(x)(stage="optimized_hlo")
                    (args.output/f"{kind}-{estimator}.hlo.txt").write_text(hlo)
                    arm = {"kind": kind, "estimator": estimator, "config": flow.config.payload(),
                        "gradient_scaled_l2_error": gradient_error, "loss_scaled_error": loss_error,
                        "precision_limit": limit, "inverse_scaled_max_error": inverse_error,
                        "inverse_logdet_max_error": ld_error, "frozen_inverse_max_error": float(frozen_error.numpy()),
                        "compile_evaluation_seconds": compile_evaluation, "compile_train_seconds": compile_train,
                        "steady_step_seconds": steady, "accepted_updates": int(final["iteration"].numpy()),
                        "evaluate_traces": trainer.evaluate.experimental_get_tracing_count(),
                        "train_traces": trainer.train_step.experimental_get_tracing_count(),
                        "weight_dtype": flow.trainable_variables[0].dtype.name, "weight_device": flow.trainable_variables[0].device,
                        "target_loss_dtype": actual["loss"].dtype.name,
                        "hlo_tf32_mentions": hlo.lower().count("tf32"),
                        "hlo_cublas_mentions": hlo.lower().count("cublas"),
                        "gpu_allocator": tf.config.experimental.get_memory_info("GPU:0")}
                    result["arms"].append(arm)
                    write(args.output/"progress.json", result)
        result["passed"] = True
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-began
        paths = [Path(__file__), ROOT/"bayesfilter/inference/neutra_transport.py", ROOT/"bayesfilter/inference/neutra_transport_core.py"]
        result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        write(args.output/"result.json", result)


if __name__ == "__main__":
    main()
