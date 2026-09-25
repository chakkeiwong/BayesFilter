"""Bounded GPU FAB calibration/training; never posterior/default promotion."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
    if isinstance(value, (list, tuple)):
        return [host(v) for v in value]
    if hasattr(value, "numpy"):
        return host(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def seed(root, role):
    value = hashlib.sha256(f"q20-fab-20260925:{root}:{role}".encode()).digest()
    return tuple(int.from_bytes(value[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, choices=["0", "1", "2"])
    parser.add_argument("--worker-seconds", required=True, type=float)
    parser.add_argument("--passes", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--temperatures", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=.01)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--transition", choices=["hmc", "metropolis"], default="hmc")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.worker_seconds <= 0 or args.passes < 1:
        raise ValueError("positive bounds required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("memory growth must be enabled in launch environment")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    manifest = {"schema": "bayesfilter.q20_fab_run.v1", "command": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan": "docs/plans/bayesfilter-fab-iaf-plan-2026-09-25.md",
        "output": str(args.output), "result": str(args.output / "result.json"),
        "worker_seconds": args.worker_seconds, "root_seed": args.seed,
        "seeds": {role: seed(args.seed, role) for role in ["map", "fab", "probe", "coverage"]},
        "requested_gpu": args.gpu, "gpu_memory_growth_env": True,
        "purpose": "exploratory_calibration_or_training_not_posterior_promotion"}
    save(args.output / "manifest.json", manifest)
    result = {"status": "started", "posterior_ready": False, "default_ready": False}
    trainer = None
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
        from bayesfilter.inference.neutra_fab import FABConfig, FABTrainer
        bridge = make_q20_tempered_bridge(20, jit_compile=True,
            principal_sqrt_backend="tensorflow_eigh_strict_factor_cached")
        target_signature = bridge.fixed_beta_adapter(1.).adapter_signature()
        def target(x):
            v, s, status = bridge.value_score_status(x, tf.constant(1., tf.float64))
            return v, s, status["bridge_valid"]
        center = tuple(host(bridge.prior_center))
        scale = (math.sqrt(bridge.prior_variance),) * bridge.parameter_dim
        cfg = replace(NeuTraTransportConfig.hoffman_author_iaf(bridge.parameter_dim,
            conditional_scale_cap=2., seed=seed(args.seed, "map"), dtype="float32"),
            hidden_layers=(16, 16), affine_center=center, affine_scale=scale)
        flow = NeuTraTransport(cfg)
        fc = FABConfig(32, args.temperatures, 5, 1, args.step_size, args.learning_rate,
            .9, .999, 1.e-8, 12800 if args.replay else 0, 1280 if args.replay else 0,
            4 if args.replay else 1, 10., None, True, .65, 1.02,
            transition_operator=args.transition)
        trainer = FABTrainer(flow, target, fc, target_signature=target_signature, seed=seed(args.seed, "fab"))
        if args.resume:
            trainer.restore(json.loads(args.resume.read_text()))
            manifest["resume"] = {"checkpoint": str(args.resume),
                "sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                "pass_index": trainer.pass_index, "optimizer_updates": int(trainer.optimizer.iterations)}
        manifest.update(config=asdict(fc), transport_config=cfg.payload(),
            target_signature=target_signature, underlying_target_signature=bridge.target_signature,
            bridge_signature=bridge.signature,
            target="q20_T30_UKF_approximate_posterior", data_version=bridge.component_target.manifest_payload() if hasattr(bridge.component_target, "manifest_payload") else bridge.signature,
            tensorflow=tf.__version__, tf32=True, jit_compile=True, batch_native_target=True,
            sample_wise_target_fallback=False, target_dtype="float64", transport_dtype="float32",
            device=flow.trainable_variables[0].device,
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "bayesfilter").rglob("*.py")},
            native_op_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "bayesfilter/ops").glob("*.so")})
        save(args.output / "manifest.json", manifest)

        def elapsed():
            return time.monotonic() - began
        deadline = datetime.fromisoformat("2026-09-25T10:00:00+00:00")
        def time_available(reserve=0.):
            return elapsed() + reserve < args.worker_seconds and datetime.now(timezone.utc) < deadline

        # One compiled batch checks radii on each signed coordinate axis and
        # common points for map dtype sensitivity. It can expose tail problems,
        # but finitely many rays cannot prove auxiliary-target integrability.
        evaluation = flow.as_dtype("float64")
        @tf.function(input_signature=[tf.TensorSpec([32, bridge.parameter_dim], tf.float64)], jit_compile=True, autograph=False)
        def inspect(x):
            v, s, ok = target(x)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x)
                q = evaluation.log_prob(x)
            score_q = tape.gradient(q, x)
            with tf.GradientTape(watch_accessed_variables=False) as tape32:
                x32 = tf.cast(x, tf.float32)
                tape32.watch(x32)
                q32 = flow.log_prob(x32)
            return dict(target=v, log_q=q, log_g=2*v-q, valid=ok,
                q_error=tf.cast(q32, tf.float64)-q,
                q_score_error=tf.cast(tape32.gradient(q32, x32),tf.float64)-score_q)
        rays = []
        for radius in [1., 2., 4., 8.]:
            for j in range(bridge.parameter_dim):
                for sign in [-1., 1.]:
                    rays.append([center[k] + (sign*radius*scale[k] if k == j else 0.) for k in range(bridge.parameter_dim)])
        tail = inspect(tf.constant(rays, tf.float64))
        save(args.output / "initial-rays.json", {"points": rays, **host(tail), "integrability_proved": False})
        if not bool(tf.reduce_all(tail["valid"])):
            raise ValueError("initial tail target validity failed")
        save(args.output / "initial-checkpoint.json", trainer.checkpoint())
        if not args.calibrate_only:
            from bayesfilter.inference.neutra_post_training import PostTrainingProbe
            initial_probe = PostTrainingProbe(flow.as_dtype("float64"), bridge, 1., jit_compile=True)
            initial_report = initial_probe(seed(args.seed, "probe"))
            save(args.output / "initial-1000.json", initial_report)
            if not (initial_report["complete"] and initial_report["finite"]
                    and initial_report["valid_rows"] == initial_report["rows"]):
                raise ValueError("initial 1000-point target/geometry diagnostic invalid")
        records, durations = [], []
        for i in range(args.passes):
            # Reserve a measured pass (minimum 120s convenience estimate until
            # pricing) plus 180s diagnostic allowance; external timeout bounds
            # synchronous compilation that cannot be interrupted from Python.
            reserve = max(durations[-2:] or [120.]) + (0. if args.calibrate_only else 180.)
            if not time_available(reserve):
                result["status"] = "partial_budget"
                break
            start = time.monotonic()
            row = trainer.step(train=not args.calibrate_only)
            duration = time.monotonic()-start
            durations.append(duration)
            record = {k: host(v) for k, v in row.items() if k not in ["x", "log_q", "log_w", "updates"]}
            record.update(seconds=duration, elapsed_seconds=elapsed(), positive_ais_count=int(tf.reduce_sum(tf.cast(row["x"][:, 2] > 0., tf.int32))),
                positive_ais_weight=float(tf.reduce_sum(tf.nn.softmax(row["log_w"]) * tf.cast(row["x"][:, 2] > 0., tf.float32))),
                updates=[{k: host(v) for k, v in u.items() if k not in ["log_q", "log_w_adjustment"]} for u in row["updates"]])
            records.append(record)
            save(args.output / "progress.json", records)
            save(args.output / "checkpoint.json", trainer.checkpoint())
            print(json.dumps(record, allow_nan=False), flush=True)
        else:
            result["status"] = "calibration_complete" if args.calibrate_only else "planned_passes_complete"
        result.update(passes=len(records), lifetime_passes=trainer.pass_index,
            optimizer_updates=int(trainer.optimizer.iterations),
            pass_seconds=durations, last_pass=records[-1] if records else None,
            initial_tail_integrability_proved=False)
        if not args.calibrate_only and time_available(150.):
            from bayesfilter.inference.neutra_post_training import PostTrainingProbe
            frozen = flow.as_dtype("float64")
            probe = PostTrainingProbe(frozen, bridge, 1., jit_compile=True)
            report = probe(seed(args.seed, "probe"))
            save(args.output / "post-training-1000.json", report)
            result["post_training_complete"] = report["complete"]
            result["post_training_finite"] = report["finite"]
            @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
            def coverage(key):
                z = tf.random.stateless_normal([4096, bridge.parameter_dim], key, dtype=tf.float64)
                x, ld = frozen.forward_and_logdet(z)
                return {"positive_count":tf.reduce_sum(tf.cast(x[:, 2] > 0.,tf.int32)), "rows":tf.constant(4096),
                    "finite":tf.reduce_all(tf.math.is_finite(x)) & tf.reduce_all(tf.math.is_finite(ld))}
            result["coverage"] = host(coverage(tf.constant(seed(args.seed, "coverage"))))
            payload = frozen.frozen_payload(target_signature=target_signature,
                training_state_hash=trainer.checkpoint()["checkpoint_hash"])
            from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
            loaded = load_frozen_neutra_artifact(payload, expected_target_signature=target_signature)
            result["frozen_map_binding_verified"] = loaded is not None
            save(args.output / "frozen-map.json", payload)
        result["gpu_allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    except KeyboardInterrupt:
        result.update(status="interrupted", error_type="KeyboardInterrupt",
            checkpoint_note="last completed pass is preserved in checkpoint.json")
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
        if trainer is not None:
            try:
                save(args.output / "failure-checkpoint.json", trainer.checkpoint())
            except Exception:
                pass
    finally:
        result["wall_seconds"] = time.monotonic()-began
        manifest["wall_seconds"] = result["wall_seconds"]
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "manifest.json", manifest)
        save(args.output / "result.json", result)
    return 130 if result["status"] == "interrupted" else (1 if result["status"] == "failed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
