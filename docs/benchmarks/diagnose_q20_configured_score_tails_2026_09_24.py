"""Bounded debugging of saved validation-score tails; no training or admission."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
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


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        parser.error("gpu must be a physical device index")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if os.environ["TF_FORCE_GPU_ALLOW_GROWTH"].lower() != "true":
        raise ValueError("memory growth must be enabled before import")
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    manifest = {"command": sys.argv, "python": sys.executable,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": sha(__file__), "plan_file": "docs/plans/bayesfilter-neutra-precision-training-plan-2026-09-24.md",
        "cuda_visible_devices": args.gpu, "worker_cap_seconds": 300,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    save(args.output/"manifest.json", manifest)
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        bridge = make_q20_tempered_bridge(20, jit_compile=True,
            principal_sqrt_backend="tensorflow_eigh_strict_factor_cached")
        adapter = bridge.fixed_beta_adapter(1.)
        manifest.update(target_signature=bridge.target_signature, bridge_signature=bridge.signature,
            tensorflow=tf.__version__, dtype="float64", jit_compile=True, tf32=True,
            sources={p:sha(ROOT/p) for p in ("bayesfilter/inference/neutra_transport_core.py",
                "bayesfilter/inference/neutra_transport.py", "bayesfilter/inference/tempered_target_tf.py")})
        save(args.output/"manifest.json", manifest)
        rows = []
        for seed in (0, 2):
            source = args.root/"continuation-queue-01"/f"naf16-seed{seed}-u8192"
            checkpoint = read(source/"checkpoint-008192.json")
            finalized = read(source/"finalized.json")
            if checkpoint != finalized["checkpoint"]:
                raise ValueError("finalized map differs from endpoint checkpoint")
            bank = read(source/"validation-008192.json")
            order = sorted(range(bank["rows"]), key=lambda i:sum(x*x for x in bank["blocks"]["residual"][i]))
            selected = order[-4:]+order[len(order)//2-2:len(order)//2+2]
            flow = NeuTraTransport(replace(NeuTraTransportConfig(**checkpoint["transport_config"]),
                dtype="float64", inverse_atol=None, inverse_rtol=None), trainable=False)
            flow.restore_parameters(checkpoint["parameters"])

            @tf.function(input_signature=[tf.TensorSpec([4, 4], tf.float64)],
                jit_compile=True, autograph=False)
            def components(z):
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    tape.watch(z)
                    x, ld = flow.forward_and_logdet(z)
                value, score = adapter.log_prob_and_grad(x)
                pulled = tape.gradient(x, z, output_gradients=score)
                jacobian_score = tape.gradient(ld, z)
                return x, value+ld, score, pulled, jacobian_score

            @tf.function(input_signature=[tf.TensorSpec([4, 4], tf.float64), tf.TensorSpec([], tf.float64)],
                jit_compile=True, autograph=False)
            def differences(z, multiplier):
                h = tf.constant((2.**-52)**(1./3.), tf.float64)*multiplier*(1.+tf.abs(z))
                offset = tf.eye(4, dtype=tf.float64)[None, :, :]*h[:, :, None]
                probes = tf.reshape(tf.concat((z[:, None, :]+offset, z[:, None, :]-offset), axis=1), [32, 4])
                x, ld = flow.forward_and_logdet(probes)
                value, _ = adapter.log_prob_and_grad(x)
                values = tf.reshape(value+ld, [4, 8])
                return (values[:, :4]-values[:, 4:])/(2.*h)

            @tf.function(input_signature=[tf.TensorSpec([4, 4], tf.float64), tf.TensorSpec([], tf.float64)],
                jit_compile=True, autograph=False)
            def physical_differences(x, multiplier):
                h = tf.constant((2.**-52)**(1./3.), tf.float64)*multiplier*(1.+tf.abs(x))
                offset = tf.eye(4, dtype=tf.float64)[None, :, :]*h[:, :, None]
                probes = tf.reshape(tf.concat((x[:, None, :]+offset, x[:, None, :]-offset), axis=1), [32, 4])
                value, _ = adapter.log_prob_and_grad(probes)
                values = tf.reshape(value, [4, 8])
                return (values[:, :4]-values[:, 4:])/(2.*h)

            record = {"seed":seed, "checkpoint_sha256":sha(source/"checkpoint-008192.json"),
                "validation_sha256":sha(source/"validation-008192.json"), "blocks":[]}
            rows.append(record)
            for start in (0, 4):
                indices = selected[start:start+4]
                z = tf.constant([bank["blocks"]["latent"][i] for i in indices], tf.float64)
                x, density, score, pulled, jacobian_score = components(z)
                expected = tf.constant([bank["blocks"]["residual"][i] for i in indices], tf.float64)
                calculated = pulled+jacobian_score+z
                for tensor in (x, density, score, pulled, jacobian_score):
                    tf.debugging.assert_all_finite(tensor, "invalid score component")
                agreement = float(tf.reduce_max(tf.abs(calculated-expected)/(1.+tf.abs(expected))))
                if agreement > 1.e-8:
                    raise ValueError("saved residual does not reproduce")
                block = {"validation_indices":indices, "latent":z.numpy().tolist(),
                    "physical":x.numpy().tolist(), "saved_residual_scaled_error":agreement,
                    "target_pullback":pulled.numpy().tolist(), "logdet_score":jacobian_score.numpy().tolist(),
                    "transformed_score":(pulled+jacobian_score).numpy().tolist(), "differences":[]}
                record["blocks"].append(block)
                for multiplier in (1., .1, .01):
                    fd = differences(z, tf.constant(multiplier, tf.float64))
                    fd_x = physical_differences(x, tf.constant(multiplier, tf.float64))
                    tf.debugging.assert_all_finite(fd, "invalid transformed difference")
                    tf.debugging.assert_all_finite(fd_x, "invalid physical difference")
                    block["differences"].append({"step_multiplier":multiplier,
                        "transformed_max_scaled_error":float(tf.reduce_max(tf.abs(fd-pulled-jacobian_score)/(1.+tf.abs(pulled+jacobian_score)))),
                        "physical_max_scaled_error":float(tf.reduce_max(tf.abs(fd_x-score)/(1.+tf.abs(score)))),
                        "transformed_finite_difference":fd.numpy().tolist()})
                save(args.output/"progress.json", {"rows":rows,"wall_seconds":time.monotonic()-began})
        passed = all(block["differences"][-1][key] <= 1.e-4
            for row in rows for block in row["blocks"]
            for key in ("transformed_max_scaled_error", "physical_max_scaled_error"))
        save(args.output/"result.json", {"rows":rows,"passed_pointwise_score_check":passed,
            "limit":1.e-4,"role":"debugging-only selected validation points; no global correctness or posterior claim",
            "selection":"four largest residuals and four middle residual-order points per seed",
            "wall_seconds":time.monotonic()-began})
        manifest["status"] = "complete"
        print(json.dumps({"pointwise_check_passed":passed,"wall_seconds":time.monotonic()-began}), flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        save(args.output/"failure.json", {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic()-began
        save(args.output/"manifest.json", manifest)


if __name__ == "__main__":
    main()
