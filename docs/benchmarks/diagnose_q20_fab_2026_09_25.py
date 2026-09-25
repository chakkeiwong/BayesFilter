"""Independent post-fit q20 importance-weight diagnostics, never promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def host(value):
    if isinstance(value, dict):
        return {k: host(v) for k, v in value.items()}
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, choices=["0", "1", "2"])
    args = parser.parse_args()
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("memory growth must precede TensorFlow import")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = {"schema": "bayesfilter.q20_fab_weight_diagnostic.v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": sys.argv, "environment": sys.executable, "run": str(args.run),
        "started_utc": datetime.now(timezone.utc).isoformat(), "gpu": args.gpu,
        "plan": "docs/plans/bayesfilter-fab-iaf-plan-2026-09-25.md",
        "output": str(args.output), "result": str(args.output / "result.json"),
        "purpose": "independent_posterior_importance_weights_and_region_diagnostics",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result = {"status": "started", "posterior_ready": False, "default_ready": False}
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
        from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
        previous = json.loads((args.run / "manifest.json").read_text())
        initial_path = args.run / "initial-checkpoint.json"
        final_path = args.run / "frozen-map.json"
        initial = json.loads(initial_path.read_text())
        final = json.loads(final_path.read_text())
        bridge = make_q20_tempered_bridge(20, jit_compile=True,
            principal_sqrt_backend="tensorflow_eigh_strict_factor_cached")
        signature = bridge.fixed_beta_adapter(1.).adapter_signature()
        if signature != previous["target_signature"] or initial["target_signature"] != signature:
            raise ValueError("target changed since training")
        load_frozen_neutra_artifact(final, expected_target_signature=signature)
        maps = []
        for checkpoint in (initial, final):
            cfg = checkpoint.get("transport_config", checkpoint.get("config"))
            flow = NeuTraTransport(NeuTraTransportConfig(**cfg), trainable=False)
            flow.restore_parameters(checkpoint["parameters"])
            maps.append(flow.as_dtype("float64"))
        digest = hashlib.sha256(f"q20-fab-20260925:{previous['root_seed']}:weighted-target-verification".encode()).digest()
        seed = [int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4)]
        rows, batch, dimension = 1000, 20, bridge.parameter_dim
        manifest.update(target_signature=signature, target="q20_T30_UKF_approximate_posterior",
            data_version=previous["data_version"], seeds=seed, rows_per_map=rows,
            target_batch_size=2*batch, batch_native_target=True, sample_wise_fallback=False,
            jit_compile=True, tf32=True, dtype="float64", tensorflow=tf.__version__,
            source_artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [initial_path, final_path]},
            source_sha256_numerical={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "bayesfilter").rglob("*.py")})
        save(args.output / "manifest.json", manifest)

        @tf.function(input_signature=[tf.TensorSpec([batch, dimension], tf.float64)], jit_compile=True, autograph=False)
        def evaluate(z):
            pairs = [flow.forward_and_logdet(z) for flow in maps]
            x = tf.concat([p[0] for p in pairs], axis=0)
            ld = tf.concat([p[1] for p in pairs], axis=0)
            base_q = -.5*tf.reduce_sum(z*z, -1)-tf.constant(.5*dimension*math.log(2*math.pi), tf.float64)
            log_q = tf.concat([base_q, base_q], axis=0)-ld
            log_p, score, status = bridge.value_score_status(x, tf.constant(1., tf.float64))
            valid = status["bridge_valid"] & tf.math.is_finite(log_p) & tf.math.is_finite(log_q)
            valid &= tf.reduce_all(tf.math.is_finite(x), -1) & tf.reduce_all(tf.math.is_finite(score), -1)
            return {"x":tf.reshape(x,[2,batch,dimension]),
                "log_weight":tf.reshape(log_p-log_q,[2,batch]),
                "valid":tf.reshape(valid,[2,batch])}

        with tf.device("/CPU:0"):
            latent = tf.random.stateless_normal([rows, dimension], seed, dtype=tf.float64)
        blocks = []
        for offset in range(0, rows, batch):
            row = evaluate(latent[offset:offset+batch])
            record = host(row)
            blocks.append(record)
            save(args.output / "rows.json", {"seed":seed, "batch_size":batch, "blocks":blocks})
            if not bool(tf.reduce_all(row["valid"])):
                raise ValueError("invalid target or proposal in independent weight diagnostic")
        x = tf.concat([tf.constant(b["x"],tf.float64) for b in blocks],axis=1)
        log_w = tf.concat([tf.constant(b["log_weight"],tf.float64) for b in blocks],axis=1)

        @tf.function(input_signature=[tf.TensorSpec([2,rows,dimension],tf.float64),
            tf.TensorSpec([2,rows],tf.float64)],jit_compile=True,autograph=False)
        def summarize(x, log_w):
            weights = tf.nn.softmax(log_w,axis=1)
            sign = tf.cast(x[:,:,2] > 0.,tf.float64)
            return {"ess":1./tf.reduce_sum(weights**2,axis=1),
                "max_weight":tf.reduce_max(weights,axis=1),
                "positive_unweighted_fraction":tf.reduce_mean(sign,axis=1),
                "positive_weighted_fraction":tf.reduce_sum(weights*sign,axis=1),
                "weighted_mean":tf.reduce_sum(weights[:,:,None]*x,axis=1),
                "log_mean_weight":tf.reduce_logsumexp(log_w,axis=1)-tf.math.log(tf.constant(rows,tf.float64))}
        result.update(status="complete", rows_per_map=rows, map_order=["initial", "final"],
            diagnostics=host(summarize(x,log_w)),
            interpretation="descriptive_only; no independently established region masses; concentrated weights cannot establish coverage or posterior moments",
            gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"))
    except Exception as exc:
        result.update(status="failed",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print(result["traceback"],flush=True)
    finally:
        result["wall_seconds"] = time.monotonic()-start
        manifest["wall_seconds"] = result["wall_seconds"]
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output/"manifest.json",manifest)
        save(args.output/"result.json",result)
    print(json.dumps(result,allow_nan=False),flush=True)
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
