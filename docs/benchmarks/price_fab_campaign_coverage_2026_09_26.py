"""Debug-only price of the exact native coverage batch; no training updates."""
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("campaign", ROOT / "docs/benchmarks/run_fab_iaf_training_campaign_2026_09_26.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig

started = time.monotonic()
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.config.set_soft_device_placement(False)
bridge, _, center, scale, signature, _ = campaign.build_target("q20")
source = ROOT / "docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26/q20-r1/fab-seed0-gpu1/initial-checkpoint.json"
initial = json.loads(source.read_text())
flow = NeuTraTransport(NeuTraTransportConfig(**initial["transport_config"]))
flow.restore_parameters(initial["parameters"])
flow = flow.as_dtype("float64")

@tf.function(input_signature=[tf.TensorSpec([32, 4], tf.float64)], jit_compile=True, autograph=False)
def evaluate(latent):
    mapped, ld = flow.forward_and_logdet(latent)
    reference = tf.constant(center, tf.float64) + tf.constant(scale, tf.float64) * latent
    points = tf.concat([mapped, reference], axis=0)
    log_p, score, valid = campaign.target_log_score(bridge, points)
    log_q = flow.log_prob(reference)
    log_g = campaign.diagonal_gaussian_log_prob(reference, center, scale)
    return log_p, score, valid, log_q, log_g, ld

latent = tf.random.stateless_normal([32, 4], [713, 419], dtype=tf.float64)
durations = []
for _ in range(3):
    call = time.monotonic()
    values = evaluate(latent)
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for i, v in enumerate(values) if i != 2)
    assert bool(tf.reduce_all(values[2]))
    durations.append(time.monotonic() - call)
result = {"role": "debug_timing_only", "target_signature": signature, "gpu": 0,
          "memory_policy": memory, "tf32": False, "jit_compile": True,
          "batch_size": 64, "call_seconds": durations, "wall_seconds": time.monotonic() - started}
destination = ROOT / "docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26/coverage-pricing-r1.json"
if destination.exists():
    raise FileExistsError(destination)
campaign.save(destination, result)
print(json.dumps(result))
