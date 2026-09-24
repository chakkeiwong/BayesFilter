"""Replay preserved invalid NAF updates; diagnostic evidence only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("training_worker", ROOT/"docs/benchmarks/run_q20_configured_training_2026_09_24.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    request = worker.read(args.campaign/"training-request-naf16-seed0-u4096-01.json")
    request["worker_seconds"] = 600.
    manifest = {"command": sys.argv, "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "exact failed minibatch replay; debugging only", "plan_file": request["plan_file"],
        "cuda_visible_devices": args.gpu, "request": request}
    runtime = worker.Runtime(request, args.output, manifest, began)
    tf = runtime.tf
    from bayesfilter.inference import neutra_transport_core as core
    from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig, NeuTraTransportTrainer, NeuTraOptimizerConfig
    def summary(value):
        value = tf.convert_to_tensor(value)
        finite = tf.math.is_finite(value)
        good = tf.boolean_mask(tf.reshape(value, [-1]), tf.reshape(finite, [-1]))
        return {"shape": value.shape.as_list(), "finite": bool(tf.reduce_all(finite)),
            "nonfinite_count": int(tf.reduce_sum(tf.cast(~finite, tf.int32))),
            "min_finite": float(tf.reduce_min(good)) if int(tf.size(good)) else None,
            "max_finite": float(tf.reduce_max(good)) if int(tf.size(good)) else None}
    rows = []
    for seed in (0, 1):
        path = args.campaign/"training-queue-01"/f"naf16-seed{seed}-u4096"/"last-valid-checkpoint.json"
        checkpoint = worker.read(path)
        flow = NeuTraTransport(NeuTraTransportConfig(**checkpoint["transport_config"]))
        trainer = NeuTraTransportTrainer(flow, runtime.target, NeuTraOptimizerConfig(**checkpoint["optimizer_config"]),
            target_signature=checkpoint["target_signature"])
        trainer.restore(checkpoint)
        iteration = int(trainer.optimizer.iterations.numpy())
        noise = runtime.noise(32, f"training-{seed}", iteration)
        row = {"seed": seed, "checkpoint": str(path), "checkpoint_sha256": worker.sha(path),
            "accepted_updates": iteration, "failed_noise_seed": worker.seed(f"training-{seed}", iteration), "modes": []}
        for dtype in ("float32", "float64"):
            candidate = flow if dtype == "float32" else flow.as_dtype("float64", trainable=True)
            z = tf.cast(noise, dtype)
            @tf.function(input_signature=[tf.TensorSpec([32, 4], z.dtype)], jit_compile=True, autograph=False)
            def inspect(points):
                layer_rows = []
                for layer in core._score_layers(candidate):
                    out, ld, jacobian, ld_score = core.value_jacobian_logdet_score(layer, points)
                    layer_rows.append((points, out, ld, tf.linalg.diag_part(jacobian), ld_score))
                    points = out
                value, score, status = runtime.target(tf.cast(points, tf.float64))
                return layer_rows, value, score, status
            layer_rows, value, score, status = inspect(z)
            mode = {"dtype": dtype, "target_valid": bool(tf.reduce_all(status)),
                "invalid_target_rows": tf.where(~status).numpy().tolist(),
                "target_value": summary(value), "target_score": summary(score),
                "layers": [{key: summary(v) for key, v in zip(("input", "output", "logdet", "jacobian_diagonal", "logdet_score"), values)}
                    for values in layer_rows], "estimators": {}}
            for estimator in ("path", "standard"):
                runtime.check_budget()
                evaluator = runtime.trainer(candidate, 32, estimator)
                result = evaluator.evaluate(z)
                mode["estimators"][estimator] = {"valid": bool(result["valid"]), "loss": summary(result["loss"]),
                    "proposal_score": summary(result["proposal_score"]),
                    "gradients": summary(tf.concat([tf.reshape(g, [-1]) for g in result["gradients"]], 0))}
            row["modes"].append(mode)
            worker.save(args.output/"partial.json", {"rows": [*rows, row]})
        result = trainer.train_step(noise)
        row["exact_replay"] = {"valid": bool(result["valid"]), "iteration_after": int(result["iteration"]),
            "gradient_norm": summary(result["gradient_norm"])}
        rows.append(row)
        worker.save(args.output/"partial.json", {"rows": rows})
    result = {"status": "diagnostic_complete", "rows": rows, "wall_seconds": time.monotonic()-began,
        "scientific_quality_evidence": False}
    worker.save(args.output/"result.json", result)
    manifest.update(status="diagnostic_complete", wall_seconds=result["wall_seconds"])
    worker.save(args.output/"manifest.json", manifest)
    print(json.dumps({"wall_seconds": result["wall_seconds"], "result": str(args.output/"result.json")}))


if __name__ == "__main__":
    main()
