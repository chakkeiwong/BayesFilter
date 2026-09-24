"""Bounded GPU/XLA engineering verification, never a learned-quality result."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true must precede framework import")
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.inference.neutra_transport import (
        NeuTraTransport, NeuTraTransportConfig, NeuTraTransportTrainer, NeuTraOptimizerConfig,
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    tf.config.experimental.enable_tensor_float_32_execution(False)

    def target(x):
        return -.5*tf.reduce_sum(x*x, axis=-1), -x, tf.ones(tf.shape(x)[0], tf.bool)

    cfgs = [NeuTraTransportConfig.hoffman_author_iaf(3, conditional_scale_cap=2., seed=(247, 19)),
            NeuTraTransportConfig.huang_dsf(3, hidden_layers=(6, 6), stages=2, mixture_components=4, seed=(247, 19))]
    opt = NeuTraOptimizerConfig(4, "path", .001, .9, .999, 1.e-7, None)
    arms = []
    for cfg in cfgs:
        with tf.device("/CPU:0"):
            reference = NeuTraTransportTrainer(NeuTraTransport(cfg), target, opt, target_signature="a"*64)
            rows = tf.constant([[.4, -.7, 1.1], [-1.3, .8, -.5], [2., 1.3, -.4], [.1, -.3, .6]], tf.float64)
            expected = reference._evaluate(rows)
        with tf.device("/GPU:0"):
            trainer = NeuTraTransportTrainer(NeuTraTransport(cfg), target, opt, target_signature="a"*64)
            trainer.restore(reference.checkpoint())
            device_rows = tf.identity(rows)
            t0 = time.monotonic()
            actual = trainer.evaluate(device_rows)
            actual["loss"].numpy()
            compile_evaluate_seconds = time.monotonic()-t0
            errors = [tf.reduce_max(tf.abs(a-b)) for a, b in zip(actual["gradients"], expected["gradients"])]
            gradient_error = float(tf.reduce_max(tf.stack(errors)).numpy())
            if not bool(actual["valid"].numpy()) or gradient_error > 1.e-9:
                raise RuntimeError("GPU path gradient failed CPU reference agreement")
            t0 = time.monotonic()
            first = trainer.train_step(device_rows)
            first["loss"].numpy()
            compile_step_seconds = time.monotonic()-t0
            if not bool(first["valid"].numpy()):
                raise RuntimeError("GPU training update rejected")
            t0 = time.monotonic()
            for _ in range(3):
                last = trainer.train_step(device_rows)
                if not bool(last["valid"].numpy()):
                    raise RuntimeError("GPU repeated update rejected")
            steady_seconds = (time.monotonic()-t0)/3
            payload = trainer.transport.frozen_payload(target_signature="a"*64)
            frozen = load_frozen_neutra_artifact(payload, expected_target_signature="a"*64).transport
            @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], jit_compile=True)
            def roundtrip(x):
                y, _ = frozen.forward_and_logdet(x)
                restored, _ = frozen.inverse_and_forward_logdet(y)
                return tf.reduce_max(tf.abs(restored-x))
            inverse_error = float(roundtrip(device_rows).numpy())
            if inverse_error > 1.e-8:
                raise RuntimeError("GPU frozen inverse failed")
            arms.append({"config": cfg.payload(), "optimizer": vars(opt),
                "gradient_max_absolute_error_vs_cpu": gradient_error,
                "inverse_max_absolute_error": inverse_error,
                "evaluate_compile_and_first_call_seconds": compile_evaluate_seconds,
                "train_compile_and_first_call_seconds": compile_step_seconds,
                "mean_repeated_step_seconds": steady_seconds,
                "accepted_updates": int(last["iteration"].numpy()),
                "train_traces": trainer.train_step.experimental_get_tracing_count(),
                "evaluate_traces": trainer.evaluate.experimental_get_tracing_count(),
                "device": actual["loss"].device,
                "gpu_allocator": tf.config.experimental.get_memory_info("GPU:0")})
    sources = [ROOT/"bayesfilter/inference"/p for p in (
        "neutra_transport.py", "neutra_transport_core.py", "neutra_artifacts.py")]
    sources.append(Path(__file__))
    result = {"schema": "bayesfilter.neutra.single_authority_gpu_smoke.v1", "passed": True,
        "scope": "small_analytic_engineering_only", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "command": sys.argv, "python": sys.executable, "tensorflow": tf.__version__,
        "memory_policy": memory, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        "cpu_reference": "explicit small analytic eager derivative reference", "dtype": "float64", "tf32": False,
        "jit_compile": True, "sample_wise_loop": False, "pfor": False, "data_version": "N/A: analytic Gaussian fixture",
        "plan_file": "docs/plans/bayesfilter-neutra-single-authority-plan-2026-09-24.md",
        "result_file": str(args.output), "arms": arms,
        "wall_seconds": time.monotonic()-start, "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "nonclaims": ["no q20 quality result", "no variance ranking", "no posterior convergence or HMC admission"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"passed": True, "wall_seconds": result["wall_seconds"], "arms": arms}, allow_nan=False))


if __name__ == "__main__":
    main()
