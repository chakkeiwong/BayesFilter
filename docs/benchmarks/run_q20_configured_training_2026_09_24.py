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
        self.measure_graphs = {}

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

    def measurement(self, flow, rows, role):
        """FP64 evaluation of the represented map, on an independent bank.

        A mutable evaluation copy uses assignable variables so the same stable
        graph measures distinct checkpoints without capturing stale constants.
        No optimizer updates are applied to the evaluation copy.
        """
        tf = self.tf
        batch = self.request.get("evaluation_batch_size", 32)
        if rows % batch:
            raise ValueError("evaluation must use whole native batches")
        key = (json.dumps(flow.config.payload(), sort_keys=True), batch)
        if key not in self.measure_graphs:
            from bayesfilter.inference.neutra_transport import NeuTraTransport
            evaluation = NeuTraTransport(replace(flow.config, dtype="float64",
                inverse_atol=None, inverse_rtol=None))
            @tf.function(input_signature=[tf.TensorSpec([batch, 4], tf.float64)],
                         jit_compile=True, autograph=False)
            def measure(z):
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    tape.watch(z)
                    x, ld = evaluation.forward_and_logdet(z)
                value, score, valid = self.target(x)
                residual = tape.gradient(x, z, output_gradients=score)+tape.gradient(ld, z)+z
                return {"loss":-value-ld, "residual":residual, "physical":x,
                    "latent":z, "log_ratio":value+ld+.5*tf.reduce_sum(z*z,axis=1),
                    "valid":valid}
            self.measure_graphs[key] = (evaluation, measure)
        evaluation, measure = self.measure_graphs[key]
        evaluation.restore_parameters(flow.parameter_state())
        blocks = []
        for start in range(0, rows, batch):
            self.check_budget()
            block = measure(self.noise(batch, role, start//batch, dtype="float64"))
            if not bool(tf.reduce_all(block["valid"])) or not all(
                    bool(tf.reduce_all(tf.math.is_finite(v))) for k,v in block.items() if k != "valid"):
                raise ValueError("invalid independent target/map evaluation")
            blocks.append(block)
        values = {k:tf.concat([b[k] for b in blocks],axis=0) for k in blocks[0]}
        norms = tf.linalg.norm(values["residual"], axis=1)
        ordered = tf.sort(norms)
        result = {"rows":rows,"role":role,"target_dtype":"float64","evaluation_dtype":"float64",
            "mean_loss":float(tf.reduce_mean(values["loss"])),
            "vector_residual_rms":float(tf.sqrt(tf.reduce_mean(norms**2))),
            "mean_residual_norm":float(tf.reduce_mean(norms)),
            "median_residual_norm":float(ordered[(rows-1)//2]),
            "p95_residual_norm":float(ordered[int(.95*(rows-1))]),
            "p99_residual_norm":float(ordered[int(.99*(rows-1))]),
            "max_residual_norm":float(ordered[-1]),
            "coordinate_residual_rms":host(tf.sqrt(tf.reduce_mean(values["residual"]**2,axis=0))),
            "log_ratio_range":float(tf.reduce_max(values["log_ratio"])-tf.reduce_min(values["log_ratio"])),
            "log_ratio_sd":float(tf.math.reduce_std(values["log_ratio"])),
            "map_parameters_sha256":hashlib.sha256(json.dumps(flow.parameter_state(),sort_keys=True).encode()).hexdigest(),
            "blocks":host(values)}
        return result

    def gradient_screen(self, family, batch):
        tf = self.tf
        flow = self.flow(family)
        gradients = {}
        for estimator in ("standard", "path"):
            trainer = self.trainer(flow, batch, estimator)
            rows=[]
            for i in range(self.request["gradient_batches"]):
                self.check_budget()
                result=trainer.evaluate(self.noise(batch,"gradient-calibration",i))
                if not bool(result["valid"]):
                    raise ValueError("invalid calibration gradient")
                rows.append(tf.concat([tf.reshape(tf.cast(g,tf.float64),[-1]) for g in result["gradients"]],0))
            gradients[estimator]=tf.stack(rows)
        reference=self.flow(family,dtype="float64")
        reference.restore_parameters(flow.parameter_state())
        trainer64=self.trainer(reference,batch,"path")
        drift=[]
        for i in range(self.request["gradient_batches"]):
            self.check_budget()
            result=trainer64.evaluate(tf.cast(self.noise(batch,"gradient-calibration",i),tf.float64))
            if not bool(result["valid"]):
                raise ValueError("invalid reference gradient")
            grad=tf.concat([tf.reshape(g,[-1]) for g in result["gradients"]],0)
            drift.append(tf.reduce_sum((grad-gradients["path"][i])**2))
        summary={}
        for estimator,g in gradients.items():
            mean=tf.reduce_mean(g,axis=0)
            variance=tf.reduce_sum((g-mean)**2)/tf.cast(tf.shape(g)[0]-1,tf.float64)
            summary[estimator]={"mean_gradient_norm":float(tf.linalg.norm(mean)),
                "minibatch_rms_variability":float(tf.sqrt(variance)),
                "norms":host(tf.linalg.norm(g,axis=1))}
            # Keras Adam uses epsilon with the uncorrected second moment.
            # Compare the first-step direction to epsilon=0 on the same
            # gradient; this is an algebraic sensitivity check, not a new fit.
            effective_epsilon=self.request["adam_epsilon"]/math.sqrt(1.-.999)
            actual=tf.math.divide_no_nan(g[0],tf.abs(g[0])+effective_epsilon)
            limit=tf.sign(g[0])
            summary[estimator]["initial_adam_direction_relative_epsilon_effect"]=float(
                tf.math.divide_no_nan(tf.linalg.norm(actual-limit),tf.linalg.norm(limit)))
        noise=summary["path"]["minibatch_rms_variability"]
        error=float(tf.sqrt(tf.reduce_mean(drift)))
        result={"family":family,"batch":batch,"batches":self.request["gradient_batches"],
            "estimators":summary,"fp32_tf32_vs_fp64_gradient_rms":error,
            "precision_drift_over_minibatch_noise":error/noise if noise else None,
            "precision_investigation_trigger":noise == 0. or error>.1*noise,
            "classification":"low_cost_sanity_screen_not_variance_optimization"}
        save(self.root/f"gradients-{family}-b{batch}.json",result)
        return result

    def calibration(self):
        """Target-specific bounded LR/estimator checks, never final evidence."""
        tf = self.tf
        rows=[]
        screens=[]
        for family in self.request["families"]:
            for batch in self.request["gradient_batch_sizes"]:
                screens.append(self.gradient_screen(family,batch))
            batch=self.request["training_batch_size"]
            chosen=next(s for s in screens if s["family"]==family and s["batch"]==batch)
            if chosen["precision_investigation_trigger"]:
                raise ValueError("precision drift needs investigation before training")
            for estimator in self.request["estimators"]:
                clip=self.request["clip_multiplier"]*max(chosen["estimators"][estimator]["norms"])
                for lr in self.request["learning_rates"]:
                    self.check_budget()
                    flow=self.flow(family)
                    trainer=self.trainer(flow,batch,estimator,learning_rate=lr,
                        epsilon=self.request["adam_epsilon"],clip=clip)
                    initial=self.measurement(flow,self.request["validation_rows"],"calibration-validation")
                    path=[]
                    began=time.monotonic()
                    rejected=None
                    for i in range(self.request["updates"]):
                        self.check_budget()
                        old=[tf.identity(v) for v in flow.trainable_variables]
                        result=trainer.train_step(self.noise(batch,"calibration-training",i))
                        if not bool(result["valid"]):
                            rejected="nonfinite_or_invalid_target_update"
                            break
                        changes=[a-b for a,b in zip(flow.trainable_variables,old)]
                        offsets=[]
                        offset=0
                        for stage in flow.stages:
                            count=len(stage.trainable_variables)
                            offsets.append(float(tf.math.divide_no_nan(
                                tf.linalg.global_norm(changes[offset:offset+count]),
                                tf.linalg.global_norm(old[offset:offset+count]))))
                            offset+=count
                        path.append({"step":i+1,"loss":float(result["loss"]),
                            "gradient_norm":float(result["gradient_norm"]),
                            "clipped":float(result["gradient_norm"])>clip,
                            "update_norm":float(tf.linalg.global_norm(changes)),
                            "relative_update_by_stage":offsets})
                    final=self.measurement(flow,self.request["validation_rows"],"calibration-validation") if rejected is None else None
                    arm=f"{family}-{estimator}-lr{lr:g}"
                    # Calibration states are preserved as evidence, never used
                    # as an undisclosed final-training warm start.
                    save(self.root/f"checkpoint-{arm}.json",trainer.checkpoint())
                    row={"arm":arm,"family":family,"estimator":estimator,"learning_rate":lr,
                        "gradient_clip_norm":clip,"initial":initial,"final":final,
                        "rejection":rejected,"updates":path,"wall_seconds":time.monotonic()-began,
                        "optimizer":vars(trainer.config),"transport_config":flow.config.payload()}
                    save(self.root/f"calibration-{arm}.json",row)
                    rows.append({k:v for k,v in row.items() if k not in ("initial","final","updates")})
                    rows[-1].update(initial_loss=initial["mean_loss"],
                        final_loss=None if final is None else final["mean_loss"],
                        clipped_updates=sum(r["clipped"] for r in path),completed_updates=len(path))
                    save(self.root/"calibration-progress.json",rows)
                    print(json.dumps(rows[-1]),flush=True)
        return {"status":"calibration_complete","arms":rows,"gradient_screens":screens,
            "training_quality_established":False,"posterior_qualified":False}

    def training(self):
        tf = self.tf
        r=self.request
        flow=self.flow(r["family"],r["root_seed"])
        trainer=self.trainer(flow,r["batch_size"],r["estimator"],r["learning_rate"],
            r["adam_epsilon"],r["gradient_clip_norm"])
        first=0
        if r.get("resume_checkpoint"):
            checkpoint=read(r["resume_checkpoint"])
            trainer.restore(checkpoint)
            first=int(trainer.optimizer.iterations.numpy())
        history=[]
        assessments=[]
        validation_role=f"training-validation-{r['root_seed']}"
        initial=self.measurement(flow,r["validation_rows"],validation_role)
        save(self.root/"initial-validation.json",initial)
        save(self.root/f"checkpoint-{first:06d}.json",trainer.checkpoint())
        started=time.monotonic()
        for i in range(first,r["updates"]):
            self.check_budget()
            result=trainer.train_step(self.noise(r["batch_size"],f"training-{r['root_seed']}",i))
            if not bool(result["valid"]):
                save(self.root/"last-valid-checkpoint.json",trainer.checkpoint())
                save(self.root/"history.json",history)
                raise ValueError(f"invalid training candidate at update {i+1}")
            history.append({"step":i+1,"loss":float(result["loss"]),
                "gradient_norm":float(result["gradient_norm"]),
                "clipped":float(result["gradient_norm"])>r["gradient_clip_norm"]})
            if (i+1)%r["checkpoint_every"]==0 or i+1==r["updates"]:
                save(self.root/f"checkpoint-{i+1:06d}.json",trainer.checkpoint())
                save(self.root/"history.json",history)
                progress={"family":r["family"],"seed":r["root_seed"],"updates":i+1,
                    "worker_seconds":time.monotonic()-self.began,
                    "updates_per_second":(i+1-first)/(time.monotonic()-started),
                    "clipped_updates":sum(x["clipped"] for x in history)}
                save(self.root/"progress.json",progress)
                print(json.dumps(progress),flush=True)
            if i+1 in r["validation_rungs"] or i+1==r["updates"]:
                check=self.measurement(flow,r["validation_rows"],validation_role)
                check["updates"]=i+1
                save(self.root/f"validation-{i+1:06d}.json",check)
                assessments.append({k:v for k,v in check.items() if k!="blocks"})
                # A sustained clipper-role violation triggers repair without
                # consuming the rest of this candidate's allocation blindly.
                recent=history[-r["checkpoint_every"]:]
                if sum(x["clipped"] for x in recent)>.5*len(recent):
                    raise ValueError("gradient guard clips a majority of recent updates; repair required")
        self.check_budget()
        # Every completed training arm gets the owner-required 1,000-point
        # standard probe, with untouched seed and exact exported FP64 weights.
        finalized=trainer.finalize(self.bridge,diagnostic_seed=seed(f"post-training-{r['root_seed']}"))
        save(self.root/"finalized.json",finalized)
        return {"status":"training_complete","family":r["family"],"seed":r["root_seed"],
            "updates":r["updates"],"checkpoint":str(self.root/f"checkpoint-{r['updates']:06d}.json"),
            "finalized":str(self.root/"finalized.json"),"validation":assessments,
            "post_training":finalized["post_training"],"clipped_updates":sum(x["clipped"] for x in history),
            "training_quality_established":False,"posterior_qualified":False}

    def final_evaluation(self):
        from bayesfilter.inference.neutra_transport import NeuTraTransport,NeuTraTransportConfig
        r=self.request
        if r.get("checkpoint"):
            checkpoint=read(r["checkpoint"])
            flow=NeuTraTransport(NeuTraTransportConfig(**checkpoint["transport_config"]))
            flow.restore_parameters(checkpoint["parameters"])
        else:
            flow=self.baseline
        result=self.measurement(flow,r["rows"],"untouched-final-bank")
        save(self.root/"final-bank.json",result)
        return {"status":"final_evaluation_complete","measurement":str(self.root/"final-bank.json"),
            **{k:v for k,v in result.items() if k!="blocks"}}


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
        if request["stage"] not in ("pricing","calibration","training","final_evaluation"):
            raise ValueError("unsupported execution stage")
        result=getattr(runtime,request["stage"])()
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
