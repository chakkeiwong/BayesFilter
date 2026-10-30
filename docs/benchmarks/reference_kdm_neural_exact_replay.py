"""Independent CPU author reference for neural KDM replay; never a runtime backend."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace, MethodType, ModuleType

# The pinned author environment omits this optional visualization dependency.
# The audited source imports it but the replayed KDM path never calls it.
sys.modules.setdefault("torchviz", ModuleType("torchviz"))

ROOT = Path("/home/chakwong/BayesFilter")
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
EXPERIMENT = AUTHOR / "src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs"
HISTORY = ROOT / ".localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/stage0"
sys.path[:0] = [str(AUTHOR / "src"),
               str(AUTHOR / "packages/kernel-density-estimator-bandwdidth-prediction/packages")]
source_models = ModuleType('models')
source_models.__path__ = [str(AUTHOR / 'src/models')]
sys.modules['models'] = source_models
import torch
import yaml
from models.kde_particle_filter.kde_particle_filter import KDEParticleFilter
from kernel_density_estimation.kernel_density_estimator import KernelDensityEstimator
from loss_functions.loss_functions import KDE_NLL

HASHES = {}
def verify(path):
    path = Path(path)
    relative = path.relative_to(AUTHOR)
    data = path.read_bytes()
    expected = subprocess.check_output(["git", "-C", str(AUTHOR), "show", PIN + ":" + str(relative)])
    if data != expected:
        raise RuntimeError("Source mismatch: " + str(relative))
    HASHES[str(relative)] = hashlib.sha256(data).hexdigest()
    return data

def source_member(path, classname, name=None):
    tree = ast.parse(verify(path))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == classname)
    node = cls if name is None else next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    ns = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), ns)
    return ns[classname if name is None else name]

def host(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: host(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [host(v) for v in value]
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-offset", type=int, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    replay_cases = {c["variant"]:c for c in replay["cases"] if c["dtype"] == "float32"}
    replay_controls = []
    batch_size = int(args.batch_size)
    updates_count = int(args.updates)
    if updates_count != 1 or batch_size != 4:
        raise ValueError("Exact-input localization is one update of four rows")
    if batch_size < 1 or updates_count < 1:
        raise ValueError("batch-size and updates must be positive")
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = HISTORY / "attempt16/full_training/models/full_dpf_model_best.pt"
    contract = json.loads((HISTORY / "attempt17/contract.json").read_text())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != contract["checkpoint_sha256"]:
        raise RuntimeError("Historical checkpoint hash mismatch")
    common = yaml.safe_load(verify(EXPERIMENT / "configs/common_parameters_training.yaml"))
    architecture = yaml.safe_load(verify(EXPERIMENT / "configs/models.yaml"))["models"]
    historical = json.loads((HISTORY / "attempt16/diagnostic-configurations.json").read_text())
    training = historical["configurations"]["full_training"]["training_params"]
    if common["model_params"] != historical["configurations"]["full_training"]["model_params"]:
        raise RuntimeError("Historical model wiring differs from current source")
    data_path = HISTORY / "attempt16/datasets/training.pt"
    saved = torch.load(data_path, map_location="cpu", weights_only=False)
    get_item = source_member(AUTHOR / "src/datasets/bearings_only_dataset.py", "BearingsOnlyDataset", "__getitem__")
    data_owner = SimpleNamespace(car_states=saved["car_states"], sensor_obs=saved["sensor_obs"],
                                 sparse_ground_truth_keep_modulo=4)
    observation_class = source_member(AUTHOR / "src/problems/bearings_only/bearings_only_problem.py",
                                      "BearingsOnlyObservationTransformer")
    trainer_path = AUTHOR / "src/trainers/specific_trainers/full_sequence_trainer.py"
    forward = source_member(trainer_path, "FullSequenceTrainer", "do_forward_pass")
    convert = source_member(trainer_path, "FullSequenceTrainer", "convert_to_model_dtype")
    cases = []
    for variant, directory in [("mdpf", "experiment0002_importance"),
                               ("am_dpf", "experiment0003_importance_init")]:
        arch = dict(architecture)
        arch.update(yaml.safe_load(verify(EXPERIMENT / directory / "config.yaml"))["models"])
        for dtype_name in ("float64", "float32"):
            dtype = getattr(torch, dtype_name)
            # The original networks/checkpoint/constants are float32, then cast
            # for the explicit double diagnostic, including non-buffer constants.
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(args.seed)
            model = KDEParticleFilter(common["model_params"], arch)
            model.load_pretrained({"dpf_model": str(checkpoint)}, torch.device("cpu"))
            model.to(dtype=dtype)
            for module in model.modules():
                for attribute in ("min_bandwidths", "max_bandwidths", "residual_scale_factor"):
                    value = getattr(module, attribute, None)
                    if torch.is_tensor(value):
                        setattr(module, attribute, value.to(dtype=dtype))
            torch.set_default_dtype(dtype)
            source_case = replay_cases[variant]
            if set(dict(model.named_parameters())) != set(source_case["parameters"]):
                raise RuntimeError("Exact replay parameter coverage mismatch")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(torch.tensor(source_case["parameters"][name], dtype=dtype))
            model.train()
            class Collector:
                def __init__(self):
                    self.optimizers = []
                    self.rates = {}
                def add_optimizer_and_lr_scheduler(self, modules, names, lr):
                    for module in modules:
                        if lr == "Freeze":
                            for p in module.parameters():
                                p.requires_grad_(False)
                        else:
                            ps = list(module.parameters())
                            self.optimizers.append(torch.optim.Adam(ps, lr=lr, weight_decay=0.0))
                            for name, p in model.named_parameters():
                                if any(p is x for x in ps):
                                    if name in self.rates:
                                        raise RuntimeError("Parameter included in multiple optimizers")
                                    self.rates[name] = lr
            collector = Collector()
            model.create_and_add_optimizers(training, collector, "full")
            params = dict(model.named_parameters())
            trainable = {k: p for k, p in params.items() if p.requires_grad}
            if set(trainable) != set(collector.rates):
                raise RuntimeError("Optimizer trainable coverage mismatch")
            loss_params = dict(kde_params=arch["dpf_full_training"]["kde"])
            loss = KDE_NLL(loss_params, model)
            trainer = SimpleNamespace(main_model=model, device=torch.device("cpu"),
                problem=SimpleNamespace(observation_transformer=observation_class()),
                number_of_particles=25, skip_first_step_for_loss=False,
                max_sequence_length=None, truncated_bptt_modulo=4,
                loss_shaper=None, loss_function=loss)
            trainer.convert_to_model_dtype = MethodType(convert, trainer)
            initial_parameters = host(params)
            updates = []
            for update in range(updates_count):
                torch.manual_seed(args.seed + update)
                rows = [get_item(data_owner, i) for i in range(args.data_offset + batch_size * update, args.data_offset + batch_size * update + batch_size)]
                data = {k: torch.stack([row[k] for row in rows]).to(dtype=dtype)
                        for k in ("states", "observations")}
                data["ground_truth_mask"] = torch.stack([row["ground_truth_mask"] for row in rows])
                source_update = source_case["updates"][0]
                source_capture = source_update["captured"]
                data = {key: torch.tensor(value, dtype=torch.bool if key=="ground_truth_mask" else dtype)
                        for key,value in source_update["data"].items()}
                captured = dict(samples=[], noise=[], outputs=[], proposal_before_normalization=[])
                original_init = model.create_initial_dpf_state
                def capture_init(*a, **kw):
                    result = {key: torch.tensor(value,dtype=dtype) if isinstance(value,list) else value
                              for key,value in source_capture["initial"].items()}
                    captured["initial"] = host(result)
                    return result
                model.create_initial_dpf_state = capture_init
                original_sample = KernelDensityEstimator.sample
                def sample(kde, *a, **kw):
                    result = torch.tensor(source_capture["samples"][len(captured["samples"])], dtype=dtype)
                    captured["samples"].append(host(result))
                    return result
                KernelDensityEstimator.sample = sample
                def proposal_inputs(module, inputs):
                    old, encoded, obs = inputs
                    noise = torch.tensor(source_capture["noise"][len(captured["noise"])], dtype=dtype)
                    captured["noise"].append(host(noise))
                    return old, encoded, obs, noise
                proposal_hook = model.proposal_model.register_forward_pre_hook(proposal_inputs)
                def capture_proposal(module, inputs, result):
                    captured["proposal_before_normalization"].append(host(result[0]))
                proposal_output_hook = model.proposal_model.register_forward_hook(capture_proposal)
                def capture_output(module, inputs, result):
                    captured["outputs"].append(host({k: v for k, v in result.items() if torch.is_tensor(v)}))
                output_hook = model.register_forward_hook(capture_output)
                model.zero_grad(set_to_none=True)
                tick = time.monotonic()
                try:
                    objective, _effective_batch_size = forward(trainer, data, None, 0)
                    objective.backward()
                finally:
                    KernelDensityEstimator.sample = original_sample
                    model.create_initial_dpf_state = original_init
                    proposal_hook.remove()
                    proposal_output_hook.remove()
                    output_hook.remove()
                raw = {k: None if p.grad is None else host(p.grad) for k, p in trainable.items()}
                if any(p.grad is None for p in trainable.values()):
                    raise RuntimeError("Missing trainable gradient in actual full-sequence consumer")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                clipped = {k: host(p.grad) for k, p in trainable.items()}
                for opt in collector.optimizers:
                    opt.step()
                moments = {}
                for opt in collector.optimizers:
                    for name, p in trainable.items():
                        if p in opt.state:
                            moments[name] = host(opt.state[p])
                if len(captured["samples"]) != 15 or len(captured["noise"]) != 16:
                    raise RuntimeError("Unexpected source sequence/draw count")
                updates.append(dict(data=host(data), captured=captured, loss=host(objective),
                    gradient_norm=host(norm), raw_gradients=raw, clipped_gradients=clipped,
                    parameters_after=host(trainable), adam_after=moments,
                    wall_seconds=time.monotonic()-tick))
            if dtype_name == "float32":
                for field in ("data", "captured", "loss", "gradient_norm", "raw_gradients",
                              "clipped_gradients", "parameters_after", "adam_after"):
                    passed = updates[0][field] == source_update[field]
                    replay_controls.append(dict(variant=variant,field=field,pass_check=passed))
                    if not passed:
                        raise RuntimeError("Float32 replay control failed: "+variant+"/"+field)
            cases.append(dict(fixture_id="exact_float32_inputs",variant=variant, dtype=dtype_name, batch_size=batch_size,
                parameters=initial_parameters,
                trainable_names=list(trainable), frozen_names=[k for k,p in params.items() if not p.requires_grad],
                learning_rates=collector.rates, updates=updates,
                constants=dict(residual_scale=host(model.proposal_model.residual_scale_factor),
                    min_bandwidths=host(model.weighted_bandwidth_predictor.min_bandwidths),
                    resampling_min_bandwidths=host(model.resampling_weighted_bandwidth_predictor.min_bandwidths)
                      if model.resampling_weighted_bandwidth_predictor is not None else None),
                checkpoint_role="fitted_MDPF" if variant == "mdpf" else "source_converted_MDPF_warm_start"))
    # Verify every author Python module actually imported, plus extracted endpoints.
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).suffix == ".py" and Path(filename).is_relative_to(AUTHOR):
            verify(Path(filename))
    historical_result = json.loads((HISTORY / "attempt16/result.json").read_text())
    cost = {}
    for phase, d in historical_result["phases"].items():
        cost[phase] = {k:v for k,v in d.items() if isinstance(v,(int,float,str,bool)) and
                       any(x in k for x in ("second","epoch","wall","time","status"))}
    payload = dict(schema="kdm_neural_author_reference_v1", cpu_only=True, gpu_intentionally_hidden=True,
        torch_version=torch.__version__, author_commit=PIN, author_hashes=HASHES,
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        dataset=str(data_path), dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        training_config=training, cases=cases, historical_cost=cost,
        sampler_status="author samples replayed; no TensorFlow sampling-law claim",
        wall_seconds=time.monotonic()-start,
        batch_size=batch_size, updates=updates_count, seed=args.seed, data_offset=args.data_offset,
        source_training_scope=f"{updates_count} diagnostic update(s), four cases",
        replay_controls=replay_controls, replay_source=str(args.replay.resolve()),
        replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(cases), wall_seconds=payload["wall_seconds"], historical_cost=cost)))

if __name__ == "__main__":
    main()
