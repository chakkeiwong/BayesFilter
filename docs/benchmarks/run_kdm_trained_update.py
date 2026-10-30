"""Bounded diagnostic of actual learned KDM checkpoints; never a training route."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PLAN = "docs/plans/controlled-smc-kdm-trained-update-localization-20261008.md"
RESULT = "docs/plans/controlled-smc-kdm-trained-update-localization-result-20261008.md"
SOURCE_RUN = BASE / "kdm-complete-training-20261007-03/diagnostic"
TF_RUN = BASE / "kdm-complete-training-20261007-04/diagnostic"
SOURCE_PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
TF_PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
CAPS = {"cpu": 900.0, "gpu": 1800.0}
SEED, BATCH, OFFSET = 202610086, 64, 384

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False) + "\n")
    temp.replace(path)

def worker(out):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if not sys.flags.isolated:
        raise RuntimeError("Author reference requires isolated Python")
    started = time.monotonic()
    helper = ROOT / "docs/benchmarks/reference_kdm_neural_exact_replay.py"
    spec = importlib.util.spec_from_file_location("author_helpers", helper)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch = ref.torch
    yaml = ref.yaml
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    reviewed_path = BASE / "kdm-complete-training-reviewed-20261007.json"
    reviewed = read(reviewed_path)
    assert reviewed["decision"]["status"] == "PASS"
    for run in reviewed["training_runs"]:
        for key in ("manifest", "result", "audit"):
            assert sha(ROOT / run[key]) == run[key + "_sha256"]
    configs = {x["name"]: x for x in read(SOURCE_RUN / "configurations.json")["executed_stages"]}
    source_stages = {x["stage"]: x for x in read(SOURCE_RUN / "result.json")["stages"]}
    tf_stages = {x["stage"]: x for x in read(TF_RUN / "result.json")["completed_stages"]}
    data_path = SOURCE_RUN / "datasets/training.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    get_item = ref.source_member(ref.AUTHOR / "src/datasets/bearings_only_dataset.py", "BearingsOnlyDataset", "__getitem__")
    data_owner = SimpleNamespace(car_states=data["car_states"], sensor_obs=data["sensor_obs"], sparse_ground_truth_keep_modulo=4)
    rows = [get_item(data_owner, i) for i in range(OFFSET, OFFSET+BATCH)]
    batch = {k: torch.stack([r[k] for r in rows]) for k in ("states", "observations", "ground_truth_mask")}
    assert batch["states"].shape == (BATCH, 17, 3)
    assert torch.equal(batch["ground_truth_mask"], (torch.arange(17) % 4 == 0).expand(BATCH, 17))
    transformer = ref.source_member(ref.AUTHOR / "src/problems/bearings_only/bearings_only_problem.py", "BearingsOnlyObservationTransformer")
    trainer_file = ref.AUTHOR / "src/trainers/specific_trainers/full_sequence_trainer.py"
    forward = ref.source_member(trainer_file, "FullSequenceTrainer", "do_forward_pass")
    convert = ref.source_member(trainer_file, "FullSequenceTrainer", "convert_to_model_dtype")
    cases, controls, provenance = [], [], []

    def update(base, training, architecture, dtype_name, tape=None):
        dtype = getattr(torch, dtype_name)
        model = copy.deepcopy(base).to(dtype=dtype)
        for module in model.modules():
            for attribute in ("min_bandwidths", "max_bandwidths", "residual_scale_factor"):
                value = getattr(module, attribute, None)
                if torch.is_tensor(value):
                    setattr(module, attribute, value.to(dtype=dtype))
        torch.set_default_dtype(dtype)
        model.train()
        class Collector:
            def __init__(self):
                self.optimizers, self.rates = [], {}
            def add_optimizer_and_lr_scheduler(self, modules, names, lr):
                for module in modules:
                    if lr == "Freeze":
                        for parameter in module.parameters():
                            parameter.requires_grad_(False)
                    else:
                        parameters = list(module.parameters())
                        self.optimizers.append(torch.optim.Adam(parameters, lr=lr, weight_decay=0.0))
                        for name, parameter in model.named_parameters():
                            if any(parameter is p for p in parameters):
                                assert name not in self.rates
                                self.rates[name] = lr
        collector = Collector()
        model.create_and_add_optimizers(training, collector, "full")
        parameters = dict(model.named_parameters())
        trainable = {n:p for n,p in parameters.items() if p.requires_grad}
        assert set(trainable) == set(collector.rates)
        initial_parameters = ref.host(parameters)
        constants = dict(residual_scale=ref.host(model.proposal_model.residual_scale_factor),
            min_bandwidths=ref.host(model.weighted_bandwidth_predictor.min_bandwidths),
            resampling_min_bandwidths=ref.host(model.resampling_weighted_bandwidth_predictor.min_bandwidths)
                if model.resampling_weighted_bandwidth_predictor is not None else None)
        trainer = SimpleNamespace(main_model=model, device=torch.device("cpu"),
            problem=SimpleNamespace(observation_transformer=transformer()), number_of_particles=25,
            skip_first_step_for_loss=False, max_sequence_length=None, truncated_bptt_modulo=4,
            loss_shaper=None, loss_function=ref.KDE_NLL(dict(kde_params=architecture["dpf_full_training"]["kde"]), model))
        trainer.convert_to_model_dtype = MethodType(convert, trainer)
        values = {k:v.to(dtype=torch.bool if k=="ground_truth_mask" else dtype) for k,v in batch.items()}
        captured = dict(samples=[], noise=[], outputs=[], proposal_before_normalization=[])
        original_init, original_sample = model.create_initial_dpf_state, ref.KernelDensityEstimator.sample
        def initialize(*args, **kwargs):
            result = original_init(*args, **kwargs) if tape is None else {
                k:torch.tensor(v, dtype=dtype) if isinstance(v,list) else v for k,v in tape["initial"].items()}
            captured["initial"] = ref.host(result)
            return result
        def sample(kde, *args, **kwargs):
            result = original_sample(kde, *args, **kwargs) if tape is None else torch.tensor(tape["samples"][len(captured["samples"])], dtype=dtype)
            captured["samples"].append(ref.host(result))
            return result
        def proposal_inputs(module, inputs):
            old, encoded, obs = inputs
            noise = (torch.distributions.Normal(torch.as_tensor(0.,dtype=dtype),torch.as_tensor(1.,dtype=dtype))
                .sample((old.shape[0],old.shape[1],4))) if tape is None else torch.tensor(tape["noise"][len(captured["noise"])], dtype=dtype)
            captured["noise"].append(ref.host(noise))
            return old, encoded, obs, noise
        def proposal_output(module, inputs, result):
            captured["proposal_before_normalization"].append(ref.host(result[0]))
        def model_output(module, inputs, result):
            captured["outputs"].append(ref.host({k:v for k,v in result.items() if torch.is_tensor(v)}))
        model.create_initial_dpf_state = initialize
        ref.KernelDensityEstimator.sample = sample
        hooks = [model.proposal_model.register_forward_pre_hook(proposal_inputs),
                 model.proposal_model.register_forward_hook(proposal_output), model.register_forward_hook(model_output)]
        torch.manual_seed(SEED)
        tick = time.monotonic()
        try:
            objective, effective = forward(trainer, values, None, 0)
            assert effective == BATCH and bool(torch.isfinite(objective))
            objective.backward()
        finally:
            ref.KernelDensityEstimator.sample = original_sample
            model.create_initial_dpf_state = original_init
            for hook in hooks: hook.remove()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable.values())
        raw = {n:ref.host(p.grad) for n,p in trainable.items()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),100.0)
        clipped = {n:ref.host(p.grad) for n,p in trainable.items()}
        for optimizer in collector.optimizers: optimizer.step()
        moments = {}
        for optimizer in collector.optimizers:
            for name,parameter in trainable.items():
                if parameter in optimizer.state: moments[name] = ref.host(optimizer.state[parameter])
        assert len(captured["samples"]) == 15 and len(captured["noise"]) == 16
        record = dict(data=ref.host(values), captured=captured, loss=ref.host(objective),
            gradient_norm=ref.host(norm), raw_gradients=raw, clipped_gradients=clipped,
            parameters_after=ref.host(trainable), adam_after=moments, wall_seconds=time.monotonic()-tick)
        case = dict(dtype=dtype_name, batch_size=BATCH, parameters=initial_parameters,
            trainable_names=list(trainable), frozen_names=[n for n,p in parameters.items() if not p.requires_grad],
            learning_rates=collector.rates, updates=[record], constants=constants,
            checkpoint_role="actual_completed_best_checkpoint_with_fresh_diagnostic_Adam")
        torch.set_default_dtype(torch.float32)
        return case

    for variant, directory in (("mdpf","experiment0002_importance"),("am_dpf","experiment0003_importance_init")):
        cfg = configs[variant]
        architecture = {}
        for path in (ref.EXPERIMENT/"configs/models.yaml",ref.EXPERIMENT/directory/"config.yaml"):
            architecture.update(yaml.safe_load(ref.verify(path))["models"])
        assert json.loads(json.dumps(architecture)) == cfg["architecture"]
        training = cfg["configuration"]["training_params"]
        assert training["gradient_clip_value"] == 100.0 and training["truncated_bptt_modulo"] == 4
        torch.manual_seed(SEED)
        model = ref.KDEParticleFilter(cfg["configuration"]["model_params"], architecture)
        checkpoints = source_stages[variant]["checkpoints"]
        checkpoint = next(Path(p) for p in checkpoints if Path(p).name=="full_dpf_model_best.pt")
        assert sha(checkpoint) == checkpoints[str(checkpoint)]
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
        tf_path = TF_RUN/(variant+"-best.json")
        assert sha(tf_path) == tf_stages[variant]["best_checkpoint_sha256"]
        tf_state = read(tf_path)["model"]
        tf_model = copy.deepcopy(model)
        assert set(dict(tf_model.named_parameters())) == set(tf_state["parameters"])
        with torch.no_grad():
            for name, parameter in tf_model.named_parameters():
                value = torch.tensor(tf_state["parameters"][name],dtype=torch.float32)
                assert value.shape == parameter.shape
                parameter.copy_(value)
                assert parameter.tolist() == tf_state["parameters"][name]
        provenance.append(dict(variant=variant,source=str(checkpoint),source_sha256=sha(checkpoint),
            tf=str(tf_path),tf_sha256=sha(tf_path)))
        for origin, base in (("source_trained",model),("tf_trained",tf_model)):
            native = update(base,training,architecture,"float32")
            replay = update(base,training,architecture,"float32",native["updates"][0]["captured"])
            for field in ("data","captured","loss","gradient_norm","raw_gradients","clipped_gradients","parameters_after","adam_after"):
                passed = native["updates"][0][field] == replay["updates"][0][field]
                controls.append(dict(name=variant+"/"+origin+"/"+field,pass_check=passed))
                assert passed, controls[-1]["name"]
            double = update(base,training,architecture,"float64",native["updates"][0]["captured"])
            for case in (native,double):
                case.update(fixture_id=origin,variant=variant)
                cases.append(case)
            write(out.with_name("source-progress.json"),dict(variant=variant,origin=origin,cases=len(cases)))
    for module in list(sys.modules.values()):
        path = getattr(module,"__file__",None)
        if path and Path(path).suffix==".py" and Path(path).is_relative_to(ref.AUTHOR): ref.verify(Path(path))
    inventory = out.with_name("checkpoint-inventory.json")
    write(inventory,provenance)
    payload = dict(schema="kdm_trained_update_reference_v1",cpu_only=True,gpu_intentionally_hidden=True,
        author_commit=ref.PIN,author_hashes=ref.HASHES,torch_version=torch.__version__,
        checkpoint=str(inventory),checkpoint_sha256=sha(inventory),dataset=str(data_path),dataset_sha256=sha(data_path),
        checkpoint_provenance=provenance,cases=cases,replay_controls=controls,historical_cost={},
        diagnostic_seeds=[SEED],reference_components=[dict(path=str(reviewed_path),sha256=sha(reviewed_path))],
        data_offset=OFFSET,batch_size=BATCH,updates=1,wall_seconds=time.monotonic()-started,
        optimizer_state_role="Fresh diagnostic moments; not continued training")
    write(out,payload)
    print(json.dumps(dict(cases=len(cases),replay_controls=len(controls),wall_seconds=payload["wall_seconds"])))
    return 0

def supervise(out):
    if out.parent != BASE or not out.name.startswith("kdm-trained-update-20261008-"):
        raise ValueError("Use the planned fresh output root")
    prior = list(BASE.glob("kdm-trained-update-20261008-*/manifest.json"))
    if len(prior) >= 2: raise RuntimeError("Attempt cap exhausted")
    used = {"cpu":120.0,"gpu":0.0}
    for path in prior:
        record = read(path)
        if record["status"] == "RUNNING": raise RuntimeError("A diagnostic is active")
        for resource in used: used[resource] += record["resource_wall_seconds"][resource]
    if any(used[r] >= CAPS[r] for r in used): raise RuntimeError("Phase cap exhausted")
    out.mkdir(exist_ok=False)
    manifest = dict(schema="kdm_trained_update_manifest_v1",status="RUNNING",
        command=[sys.executable,*sys.argv],cwd=str(ROOT),plan_file=PLAN,result_file=RESULT,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_pid=os.getpid(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/PLAN,
            ROOT/"docs/benchmarks/diagnose_kdm_neural_precision_tf.py",ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py",
            ROOT/"bayesfilter/replication/kdm_neural_replay_tf.py",ROOT/"bayesfilter/replication/kdm_normal_von_mises_tf.py"]},
        phase_cap_seconds=CAPS,prior_charged_seconds=used,resource_wall_seconds={"cpu":0.0,"gpu":0.0},executions=[],
        seeds=[SEED],data_version="Preserved development training rows 384:448; no qualification data",replication="INCOMPLETE")
    def save(): write(out/"manifest.json",manifest)
    save()
    def run(command,resource,label,env):
        remaining = CAPS[resource]-used[resource]-manifest["resource_wall_seconds"][resource]
        if remaining <= 0: raise RuntimeError("Resource cap exhausted")
        execution = dict(command=command,resource=resource,status="RUNNING",timeout_seconds=remaining,
            environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","OMP_NUM_THREADS","TF_FORCE_GPU_ALLOW_GROWTH","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS")},log=label+".log")
        manifest["executions"].append(execution);save()
        process = None
        tick=time.monotonic()
        try:
            with (out/(label+".log")).open("w") as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                execution["pid"]=process.pid;save()
                code=process.wait(timeout=remaining)
            execution.update(returncode=code,status="COMPLETED" if code==0 else "FAILED")
            if code: raise RuntimeError(label+" exited "+str(code))
        except BaseException:
            execution["status"]="FAILED"
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            elapsed=time.monotonic()-tick
            execution["wall_seconds"]=elapsed
            manifest["resource_wall_seconds"][resource]+=elapsed
            save()
    try:
        source_env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        run([SOURCE_PYTHON,"-I","-B",str(Path(__file__).resolve()),"--worker","--output",str(out/"author.json")],"cpu","source",source_env)
        reference=read(out/"author.json")
        assert len(reference["cases"])==8 and all(x["pass_check"] for x in reference["replay_controls"])
        gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH="true",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",OMP_NUM_THREADS="1")
        gpu_env.pop("LD_LIBRARY_PATH",None)
        run([TF_PYTHON,"-B",str(ROOT/"docs/benchmarks/diagnose_kdm_neural_precision_tf.py"),"--output",str(out/"tf"),
            "--reference",str(out/"author.json"),"--plan",PLAN,"--result-note",RESULT],"gpu","tf",gpu_env)
        manifest.update(status="COMPLETED",decision=read(out/"tf/decision.json"),reference_sha256=sha(out/"author.json"),
            comparison_sha256=sha(out/"tf/comparison.json"))
    except BaseException:
        manifest["status"]="FAILED"
        (out/"error.txt").write_text(traceback.format_exc())
        raise
    finally: save()
    print(json.dumps(dict(status=manifest["status"],decision=manifest.get("decision"),resource_wall_seconds=manifest["resource_wall_seconds"])))
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    raise SystemExit(worker(args.output.resolve()) if args.worker else supervise(args.output.resolve()))
