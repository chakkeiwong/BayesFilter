"""q20 configured NeuTra calibration/training under an explicit request.

Uses the shared flow/trainer and unchanged batch-native FP64 q20 target. Host
loops schedule batched updates and save checkpoints; numerical work uses XLA.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(role, index=0):
    digest = hashlib.sha256(f"q20-configured-training-20260924:{role}:{index}".encode()).digest()
    return [int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4)]


def host(value):
    if isinstance(value, dict):
        return {k: host(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [host(v) for v in value]
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    return value


class Runtime:
    def __init__(self, request, root, manifest, began):
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge, FixedBetaBridgeAdapter
        from bayesfilter.inference.tempered_transport_ensemble_tf import restore_trainable_transport_checkpoint
        from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
        self.tf, self.request, self.root, self.manifest, self.began = tf, request, root, manifest, began
        self.bridge = make_q20_tempered_bridge(20, jit_compile=True,
            principal_sqrt_backend="tensorflow_eigh_strict_factor_cached")
        self.adapter = FixedBetaBridgeAdapter(self.bridge, beta=1.)
        checkpoint = Path(request["baseline_checkpoint"])
        if sha(checkpoint) != request["baseline_sha256"]:
            raise ValueError("baseline file checksum changed")
        cohort = read(checkpoint)
        from bayesfilter.inference.q20_production_config import digest
        if digest({k:v for k,v in cohort.items() if k != "checkpoint_hash"}) != cohort["checkpoint_hash"]:
            raise ValueError("baseline cohort checksum mismatch")
        state = cohort["cohort"]["direct-w16-lr0.0005-r0"]["session"]
        original = restore_trainable_transport_checkpoint(state["map"], expected_context={
            "target_signature": self.bridge.target_signature, "bridge_signature": self.bridge.signature})
        cfg = original.inner.config
        if cfg.stage_scale_linear_skip or cfg.stage_unbounded_scale_linear or cfg.stage_s_max:
            raise ValueError("baseline has unsupported local extensions")
        self.baseline_config = NeuTraTransportConfig(4, "iaf", cfg.hidden_layers, cfg.stages,
            cfg.activation, tuple(cfg.initialization_seed), cfg.s_max,
            scale_transform="bounded_tanh", permutation_policy=cfg.permutation_policy,
            affine_center=tuple(host(original.center)), affine_scale=tuple(host(original.scale)))
        self.baseline = NeuTraTransport(self.baseline_config, trainable=False)
        self.baseline.restore_parameters([{"weights":host(stage.weights), "biases":host(stage.biases), "extras":{}}
                                         for stage in original.inner.stages])
        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True)
        def initialize(key):
            z = tf.random.stateless_normal([8192, 4], key, dtype=tf.float64)
            points, ld = self.baseline.forward_and_logdet(z)
            original_points, original_ld = original.forward_and_logdet(z)
            center = tf.reduce_mean(points, axis=0)
            scale = tf.sqrt(tf.reduce_mean(tf.square(points-center), axis=0))
            return center, scale, tf.reduce_max(tf.abs(points-original_points)), tf.reduce_max(tf.abs(ld-original_ld))
        center, scale, parity, ld_parity = initialize(tf.constant(seed("initialization-moments")))
        if float(parity) > 1.e-10 or float(ld_parity) > 1.e-10 or not bool(tf.reduce_all(tf.math.is_finite(scale) & (scale > 0.))):
            raise ValueError("baseline reconstruction or affine initialization invalid")
        self.center, self.scale = tuple(host(center)), tuple(host(scale))
        manifest.update(tensorflow=tf.__version__, target_signature=self.bridge.target_signature,
            bridge_signature=self.bridge.signature, adapter_signature=self.adapter.adapter_signature(),
            target_dtype="float64", transport_dtype="float32", tf32=True, jit_compile=True,
            batch_native_target=True, sample_wise_target_fallback=False,
            baseline_lifetime_updates=state["iteration"], baseline_reconstruction_max_error=float(parity),
            affine_center=self.center, affine_scale=self.scale, internal_initialization_rows=8192,
            gpu_name=tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name"),
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"bayesfilter").rglob("*.py"))},
            native_op_sha256={p.name:sha(p) for p in (ROOT/"bayesfilter/ops").glob("*.so")})
        save(root/"manifest.json", manifest)
        self.noise_graphs = {}

    def check_budget(self):
        if time.monotonic()-self.began >= self.request["worker_seconds"]:
            raise TimeoutError("declared worker allowance exhausted")
        deadline = datetime.fromisoformat(self.request["deadline_utc"])
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("campaign deadline reached")

    def target(self, x):
        value, score, status = self.bridge.value_score_status(x, self.tf.constant(1., self.tf.float64))
        return value, score, status["bridge_valid"]

    def noise(self, count, role, index=0, dtype="float32"):
        tf = self.tf
        key = (count, dtype)
        if key not in self.noise_graphs:
            @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
            def draw(key):
                return tf.random.stateless_normal([count, 4], key, dtype=tf.as_dtype(dtype))
            self.noise_graphs[key] = draw
        return self.noise_graphs[key](tf.constant(seed(role, index)))

    def flow(self, family, root_seed=0, dtype="float32"):
        from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
        if family == "legacy_control":
            flow = NeuTraTransport(replace(self.baseline_config, dtype=dtype,
                inverse_atol=None, inverse_rtol=None))
            flow.restore_parameters(self.baseline.parameter_state())
            return flow
        width = 32 if family == "naf32" else 16
        cfg = NeuTraTransportConfig(4, "iaf" if family == "iaf16" else "naf_dsf",
            (width, width), 3, "elu", tuple(seed("initialization", root_seed)), 2.,
            dtype=dtype, mixture_components=16,
            mask_policy="hoffman_block_masks_v1" if family == "iaf16" else "legacy_degree_masks_v1",
            affine_center=self.center, affine_scale=self.scale)
        return NeuTraTransport(cfg)

    def trainer(self, flow, batch, estimator, learning_rate=.001, epsilon=1.e-8, clip=None):
        from bayesfilter.inference.neutra_transport import NeuTraTransportTrainer, NeuTraOptimizerConfig
        return NeuTraTransportTrainer(flow, self.target,
            NeuTraOptimizerConfig(batch, estimator, learning_rate, .9, .999, epsilon, clip),
            target_signature=self.adapter.adapter_signature())

    def pricing(self):
        tf = self.tf
        prices = []
        for family in self.request["families"]:
            for batch in self.request["batch_sizes"]:
                for estimator in self.request["estimators"]:
                    self.check_budget()
                    flow = self.flow(family)
                    trainer = self.trainer(flow, batch, estimator)
                    started = time.monotonic()
                    first = trainer.train_step(self.noise(batch, "pricing", 0))
                    if not bool(first["valid"].numpy()):
                        prices.append({"family":family,"batch":batch,"estimator":estimator,"valid":False})
                        save(self.root/"pricing-progress.json",prices)
                        continue
                    compile_seconds = time.monotonic()-started
                    times, norms = [], []
                    for i in range(self.request["repeated_updates"]):
                        self.check_budget()
                        started = time.monotonic()
                        result = trainer.train_step(self.noise(batch, "pricing", i+1))
                        if not bool(result["valid"].numpy()):
                            raise ValueError(f"invalid pricing update: {family}/{batch}/{estimator}")
                        times.append(time.monotonic()-started)
                        norms.append(float(result["gradient_norm"].numpy()))
                    row = {"family":family,"batch":batch,"estimator":estimator,"valid":True,
                        "compile_and_first_step_seconds":compile_seconds,"step_seconds":times,
                        "median_step_seconds":statistics.median(times), "gradient_norms":norms,
                        "accepted_updates":int(trainer.optimizer.iterations.numpy()),
                        "training_traces":trainer.train_step.experimental_get_tracing_count(),
                        "parameter_count":sum(int(tf.size(v)) for v in flow.trainable_variables),
                        "config":flow.config.payload(),"optimizer":vars(trainer.config),
                        "device":flow.trainable_variables[0].device}
                    prices.append(row)
                    save(self.root/"pricing-progress.json",prices)
        return {"status":"pricing_complete","prices":prices,
                "quality_evidence":False,"gpu_allocator":tf.config.experimental.get_memory_info("GPU:0")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args=parser.parse_args()
    request=read(args.request)
    if not args.gpu.isdecimal():
        parser.error("gpu must be one physical index")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
    if os.environ["TF_FORCE_GPU_ALLOW_GROWTH"].lower() != "true":
        raise ValueError("memory growth must be enabled before import")
    for key in ("TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","OMP_NUM_THREADS"):
        os.environ.setdefault(key,"2")
    args.output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    manifest={"command":sys.argv,"python":sys.executable,"request":request,
        "runner_sha256":sha(__file__),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "started_at":datetime.now(timezone.utc).isoformat(),"status":"starting",
        "cuda_visible_devices":args.gpu,"plan_file":request["plan_file"],"result_file":str(args.output/"result.json")}
    save(args.output/"manifest.json",manifest)
    try:
        runtime=Runtime(request,args.output,manifest,began)
        if request["stage"] != "pricing":
            raise ValueError("unsupported execution stage")
        result=runtime.pricing()
        result["wall_seconds"]=time.monotonic()-began
        save(args.output/"result.json",result)
        manifest["status"]=result["status"]
    except BaseException as error:
        manifest["status"]="failed"
        save(args.output/"failure.json",{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-began
        save(args.output/"manifest.json",manifest)


if __name__=="__main__":
    main()
