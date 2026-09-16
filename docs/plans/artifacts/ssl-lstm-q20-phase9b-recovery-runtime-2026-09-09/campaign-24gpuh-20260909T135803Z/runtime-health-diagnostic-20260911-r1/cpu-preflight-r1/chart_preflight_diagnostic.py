"""Debug the failed fixed preoptimizer banks without training or tuning."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--gpu-uuid")
parser.add_argument("--cpu-only", action="store_true")
parser.add_argument("--native-reference", action="store_true")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
EXECUTION = CAMPAIGN / "numerical-repairs/eigh-refinement-r1"
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(exist_ok=False, parents=True)
sys.path.insert(0, str(ROOT))
if args.cpu_only:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU diagnostic requires intentionally hidden GPUs")
elif os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("GPU pinning and growth must precede framework import")
started = time.monotonic()
import tensorflow as tf
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

memory = {"gpu_intentionally_hidden": True} if args.cpu_only else configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
tf.config.set_soft_device_placement(False)
if not args.cpu_only and len(tf.config.list_logical_devices("GPU")) != 1:
    raise RuntimeError("exactly one GPU required")
specification = importlib.util.spec_from_file_location("preflight_recovery_diagnostic", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(specification)
specification.loader.exec_module(recovery)
start, ledger = recovery.initialize_campaign(CAMPAIGN, resume=True)
sources = recovery.source_hashes()
parallel = recovery.load_parallel()
source = parallel._load_source()
from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
from bayesfilter.inference.tempered_transport_ensemble_tf import prepare_transport_initialization
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core

if args.native_reference:
    if not args.cpu_only:
        raise RuntimeError("native eigensolver comparator is independent CPU reference only")
    core._refined_symmetric_eigh = lambda matrix, **kwargs: tf.linalg.eigh(matrix)
reports = {}
with DurableTensorCheckpoint(OUTPUT / "tensors", {"sources": sources, "cpu_only": args.cpu_only,
                                                  "native_reference": args.native_reference}) as store:
    for arm in recovery.ARMS:
        profile = parallel._profile(source, EXECUTION, arm, recovery.ARM_CAP_SECONDS)
        bridge = make_q20_tempered_bridge(20, jit_compile=not args.cpu_only,
                                         principal_sqrt_backend=profile.principal_sqrt_backend)
        config = WeightedNeuTraConfig(dimension=source.DIMENSION, hidden_layers=(16, 16), stages=2,
            activation="tanh", initialization_scale=0.02, initialization_seed=profile.initialization_roots[0],
            learning_rate=1.0e-3, jit_compile=not args.cpu_only)
        raw = WeightedDenseIAFTransport(config)
        seed = source._seed(tf, profile.preflight_roots[0], 0)
        prepared = prepare_transport_initialization(raw, bridge, component_id=source.COMPONENT_IDS[0],
            seed=seed, batch_size=source.BATCH_SIZE, repair_scales=(1.0,), beta=0.0,
            reference_center=tf.constant(bridge.prior_center, tf.float64),
            reference_scale=tf.fill([source.DIMENSION], tf.sqrt(tf.constant(float(bridge.prior_variance), tf.float64))))
        reports[arm] = {"receipt": prepared.receipt.payload(), "profile": profile.payload()}
        durable_json(OUTPUT / f"{arm}-receipt.json", reports[arm])
        latent = tf.random.stateless_normal((source.BATCH_SIZE, source.DIMENSION), seed=seed, dtype=tf.float64)
        physical = prepared.transport.forward_batch(latent)
        store.run(arm + "-inputs", {}, lambda: {"latent": latent, "physical": physical, "seed": seed})
        try:
            value, score, status = store.run(arm + "-bridge", {}, lambda: bridge.value_score_status(physical, tf.constant(0.0, tf.float64)))
            payload = {"physical": physical.numpy().tolist(), "value": value.numpy().tolist(),
                       "score": score.numpy().tolist(),
                       "status": {name: tensor.numpy().tolist() for name, tensor in status.items()}}
            durable_json(OUTPUT / f"{arm}-status.json", recovery.diagnostic_payload(payload))
            reports[arm]["invalid_rows"] = tf.where(~status["bridge_valid"])[:, 0].numpy().tolist()
        except Exception as error:
            reports[arm]["uncaught_bridge_exception"] = traceback.format_exc()
            durable_json(OUTPUT / f"{arm}-exception.json", {"type": type(error).__name__, "traceback": traceback.format_exc()})
        durable_json(OUTPUT / "preflight-result.json", reports)
        print(json.dumps({"arm": arm, "receipt": reports[arm]["receipt"],
                          "invalid_rows": reports[arm].get("invalid_rows"),
                          "exception": reports[arm].get("uncaught_bridge_exception")}), flush=True)
if sources != recovery.source_hashes():
    raise RuntimeError("source closure changed during diagnostic")
durable_json(OUTPUT / "run_manifest.json", {
    "status": "completed", "evidence_role": "preoptimizer_failure_localization_only", "sources": sources,
    "memory_policy": memory, "gpu_uuid": args.gpu_uuid, "gpu_intentionally_hidden": args.cpu_only,
    "native_eigensolver_cpu_reference": args.native_reference, "jit_compile": not args.cpu_only,
    "tensorflow": tf.__version__, "tf32_enabled": True,
    "command": [sys.executable, *sys.argv], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "data_sha256": hashlib.sha256(tf.io.serialize_tensor(bridge.component_target.config.observations).numpy()).hexdigest(),
    "seeds": {arm: reports[arm]["profile"]["seed_namespace"] for arm in recovery.ARMS},
    "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "preflight-result.json"), "optimizer_updates": 0, "tuning_launched": False,
})
