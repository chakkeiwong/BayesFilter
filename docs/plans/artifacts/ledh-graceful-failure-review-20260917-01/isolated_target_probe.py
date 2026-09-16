"""Fresh-process diagnostic of the unpatched dual-target consumer only."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
repo = root.parents[3]
sys.path.insert(0, str(repo))
if args.device == "gpu":
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True)
    rows = [line.split(", ") for line in hardware.strip().splitlines()]
    selected = next(row for row in rows if "5080" in row[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = selected[1]
else:
    hardware, selected = "GPU intentionally hidden", None
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
policy = configure_tensorflow_gpu_memory_growth(tf) if args.device == "gpu" else {"mode": "cpu_reference", "gpu_intentionally_hidden": True}
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel
from bayesfilter.inference.ledh_dual_parameter_target import DualParameterLEDHTarget

record = {"purpose": "unpatched consumer mechanics; no posterior claim", "command": [sys.executable, *sys.argv], "device_mode": args.device, "hardware": hardware, "selected_gpu": selected, "memory_policy": policy, "tensorflow": tf.__version__, "checks": {}}
started = time.monotonic()

def clean(v):
    if isinstance(v, tf.Tensor):
        return {"shape": v.shape.as_list(), "device": v.device, "value": clean(v.numpy().tolist())}
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN" if math.isnan(v) else ("+Infinity" if v > 0 else "-Infinity")
    if isinstance(v, (tuple, list)):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v

def save():
    record["wall_seconds"] = time.monotonic() - started
    (root / f"isolated-{args.device}.json").write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")

def check(name, fn):
    begin = time.monotonic()
    try:
        record["checks"][name] = {"status": "returned", "output": clean(fn())}
    except Exception as exc:
        record["checks"][name] = {"status": "raised", "exception": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    record["checks"][name]["wall_seconds"] = time.monotonic() - begin
    save()

dtype = tf.float64
with tf.device("/GPU:0" if args.device == "gpu" else "/CPU:0"):
    check("device_probe", lambda: tf.reduce_sum(tf.eye(2, dtype=dtype)))
    model = PerPointScoreModel(
        transition_mean_fn=lambda theta, x: .8 * x + theta[:, :1],
        transition_mean_tangent_fn=lambda theta, x, dx, direction: .8 * dx + direction[:, :1],
        observation_fn=lambda x: x,
        observation_jacobian_fn=lambda x: tf.ones([tf.shape(x)[0], 1, 1], dtype),
        observation_tangent_fn=lambda x, dx: dx,
        process_covariance=tf.constant([[.1]], dtype),
        observation_covariance=tf.constant([[.5]], dtype),
    )
    states = tf.constant([[-1.], [-.3], [.3], [1.]], dtype)
    noises = tf.zeros([1, 4, 1], dtype)
    observations = tf.constant([[.1]], dtype)
    settings = dict(substeps=1, reset_policy="contract_e", reset_design=tf.constant([[-1.], [-1.], [1.], [1.]], dtype), correction_steps=1, pairwise_steps=1, coordinate_cap=2.)

    def evaluate(theta, covariance):
        target = DualParameterLEDHTarget(model, states, covariance, noises, observations, exact_params=settings, biased_params=settings)
        with tf.GradientTape() as tape:
            tape.watch(theta)
            value = target(theta)
        return value, tape.gradient(value, theta)

    modes = ("graph", "xla") if args.device == "cpu" else ("xla",)
    for mode in modes:
        fn = tf.function(evaluate, input_signature=[tf.TensorSpec([1], dtype), tf.TensorSpec([4, 1, 1], dtype)], jit_compile=mode == "xla")
        check(f"{mode}/healthy", lambda fn=fn: fn(tf.zeros([1], dtype), tf.ones([4, 1, 1], dtype) * .2))
        check(f"{mode}/invalid_initial_covariance", lambda fn=fn: fn(tf.zeros([1], dtype), -tf.ones([4, 1, 1], dtype)))

save()
print(json.dumps({"path": str(root / f"isolated-{args.device}.json"), "wall_seconds": record["wall_seconds"], "checks": len(record["checks"])}))
