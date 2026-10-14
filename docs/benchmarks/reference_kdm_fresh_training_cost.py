"""Independent CPU author training/cost reference, never a BayesFilter runtime."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT/".localresources/code/younis-mdpf-neurips-2023"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
EXP = Path("experiments/bearings_only_experiments_methodical_25_particles_multiple_runs")
OLD = ROOT/".localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks"

def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False)+"\n")
    temporary.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", choices=("cost","smoke","full"), default="cost")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--constructor-seed-base", type=int, default=202610072)
    args = parser.parse_args()
    if not 0 <= args.constructor_seed_base <= 2**32-3:
        raise ValueError("Constructor seed base outside NumPy seed range")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference=None
    if args.epochs!="cost":
        if args.reference is None: raise ValueError("Fresh constructor reference required")
        reference=json.loads(args.reference.read_text())
        manifest=json.loads((args.reference.parent.parent/"manifest.json").read_text())
        if reference["status"]!="PASS" or manifest["status"]!="COMPLETED" or manifest["result_sha256"]!=sha(args.reference):
            raise ValueError("Invalid fresh source reference")
    if not sys.flags.isolated:
        raise RuntimeError("Author reference requires isolated Python")
    import torch
    import numpy as np  # Independent author data-generation/reference exception.
    import yaml
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    subprocess.run(["git", "-C", str(AUTHOR), "diff", "--quiet", PIN, "--", "src", "packages"], check=True)
    old = load_module("author_training_reference_helpers", OLD/"diagnose_replication_training.py")
    source = old.source_copy(AUTHOR, out, torch)
    sys.modules.setdefault("torchviz", ModuleType("torchviz"))
    sys.path[:0] = [str(source), str(AUTHOR/"packages/kernel-density-estimator-bandwdidth-prediction/packages")]
    os.chdir(source)
    from models.model_creator import create_model
    from trainers import training_runner as runner
    from trainers.trainer import Trainer
    from trainers.specific_trainers.weight_model_trainer import WeightModelTrainer
    from trainers.specific_trainers.full_sequence_trainer import FullSequenceTrainer
    from trainers.trainer_utils import CustomReduceLROnPlateau, EarlyStopping
    old.check_scheduler_compatibility(torch, CustomReduceLROnPlateau, device)

    def host(value):
        if torch.is_tensor(value): return value.detach().cpu().tolist()
        if isinstance(value, dict): return {k:host(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)): return [host(v) for v in value]
        return value
    def read(relative):
        return yaml.safe_load((source/relative).read_text())
    config_helper = load_module("source_configuration_helpers", OLD/"diagnose_replication_stage0.py")
    helpers = config_helper.source_functions(source/"main.py",
        {"recursive_update", "load_common_parameters_file"}, {"yaml":yaml})
    base_arch = read(EXP/"configs/models.yaml")["models"]
    pre_entry = read(EXP/"individual_training/config.yaml")
    pre = copy.deepcopy(next(v for e in pre_entry["experiments"] for k,v in e.items()
                             if k=="full_dpf_weight_training_fixed_bands"))
    stages = [("weights", pre, {**base_arch, **pre_entry["models"]})]
    for name, config, entry in (("mdpf","our_experiments.yaml","experiment0002_importance"),
                                ("am_dpf","our_experiments_exp3_init.yaml","experiment0003_importance_init")):
        spec = copy.deepcopy(next(v for e in read(EXP/"configs"/config)["experiments"]
                                  for k,v in e.items() if k=="full_dpf_full_training_fixed_bands"))
        spec = helpers["load_common_parameters_file"](spec["common_parameters_file"], spec)
        arch = {**base_arch, **read(EXP/entry/"config.yaml")["models"]}
        method = arch["dpf_full_training"]
        if method["differentiable_resampling_method"] != "ImportanceSampling":
            raise ValueError("Wrong source gradient method")
        for key in ("decouple_weights_for_resampling", "decouple_bandwidths_for_resampling"):
            if method[key] != (name=="am_dpf"): raise ValueError("Wrong source variant")
        stages.append((name,spec,arch))
    data = read(EXP/"configs/dataset_params_training.yaml")["dataset_params"]
    assert data["dataset_sizes"] == {"training":5000,"validation":1000}
    data["dataset_saves"] = {s:str(out/"datasets"/(s+".pt")) for s in ("training","validation")}
    source_configs = {}
    for name, cfg, arch in stages:
        expected_cap = 400 if name=="weights" else 800
        assert cfg["training_params"]["epochs"] == expected_cap
        source_configs[name] = copy.deepcopy(cfg["training_params"])
        cfg["training_params"]["epochs"] = expected_cap if args.epochs=="full" else (2 if args.epochs=="smoke" else 1)
        cfg["training_params"]["num_cpu_cores_for_dataloader"] = 1
        cfg["dataset_params"] = copy.deepcopy(data)
        cfg.pop("dataset_params_file", None)
        cfg["save_dir"] = str(out/name)
        cfg["do_run"] = True
        cfg["device"] = "cpu"
        if name=="mdpf":
            cfg["pre_trained_models"] = {k:str(out/"weights/models"/Path(v).name)
                                          for k,v in cfg["pre_trained_models"].items()}
        elif name=="am_dpf":
            cfg["pre_trained_models"] = {"dpf_model":str(out/"mdpf/models/full_dpf_model_best.pt")}
    save(out/"configurations.json", {"source_training_configs":source_configs,
        "executed_stages":[{"name":n,"configuration":c,"architecture":a} for n,c,a in stages],
        "data_seed":202610071,"constructor_seeds":[args.constructor_seed_base+i for i in range(3)]})
    result = dict(schema="kdm_fresh_source_cost_v1", status="RUNNING", replication="INCOMPLETE",
        source_pin=PIN, source_training_configs=source_configs, datasets={}, fixtures={}, stages=[],
        backend="CPU author PyTorch/NumPy independent reference", gpu_intentionally_hidden=True,
        torch_version=torch.__version__, numpy_version=np.__version__, threads=1,
        initialization="fresh source constructors; own preceding best-checkpoint transfers",
        source_training_protocol_complete=False, existing_float32_failures="PRESERVED", epoch_mode=args.epochs,
        data_seed=202610071, constructor_seed_base=args.constructor_seed_base)
    if args.epochs!="cost": result["schema"]="kdm_complete_source_training_v1"
    active = {}
    original_problem = runner.create_problem_training
    def create_problem(*items, **kwargs):
        np_state, random_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(202610071); np.random.seed(202610071); random.seed(202610071)
                return original_problem(*items, **kwargs)
        finally:
            np.random.set_state(np_state); random.setstate(random_state)
    runner.create_problem_training = create_problem
    original_train = Trainer.train
    def train(trainer):
        model = active["model"]
        params = dict(model.named_parameters())
        rates = {}
        for opt in trainer.optimizers:
            for group in opt.param_groups:
                for name,p in params.items():
                    if any(p is q for q in group["params"]):
                        if name in rates: raise ValueError("Duplicate optimizer parameter")
                        rates[name] = group["lr"]
        fixture = dict(dtype="float32",variant="am_dpf" if active["name"]=="am_dpf" else "mdpf",
            batch_size=64,parameters=active["constructor"],
            trainable_names=[n for n in params if n in rates],learning_rates=rates,
            constants=dict(residual_scale=host(model.proposal_model.residual_scale_factor),
                min_bandwidths=host(model.weighted_bandwidth_predictor.min_bandwidths),
                resampling_min_bandwidths=host(model.resampling_weighted_bandwidth_predictor.min_bandwidths)
                    if model.resampling_weighted_bandwidth_predictor is not None else None))
        path = out/(active["name"]+"-constructor.json")
        save(path,fixture)
        result["fixtures"][active["name"]] = dict(path=str(path),sha256=sha(path))
        if reference is not None and sha(path)!=reference["fixtures"][active["name"]]["sha256"]:
            raise ValueError("Fresh constructor differs from cost reference")
        for split,dataset in (("training",trainer.training_dataset),("validation",trainer.validation_dataset)):
            if split in result["datasets"]: continue
            rows = [dataset[i] for i in range(len(dataset))]
            tensor_data = {k:torch.stack([row[k] for row in rows])
                for k in ("states","observations","ground_truth_mask")}
            path = out/(split+".json")
            save(path,dict(rows=len(rows),row_ids=list(range(len(rows))),data=host(tensor_data)))
            result["datasets"][split] = dict(path=str(path),sha256=sha(path),rows=len(rows),
                source_sha256=sha(out/"datasets"/(split+".pt")))
            if reference is not None and sha(path)!=reference["datasets"][split]["sha256"]:
                raise ValueError("Regenerated data differs from fresh development reference")
        active["trainer"] = trainer
        active["frozen"] = {n:p.detach().clone() for n,p in params.items() if n not in rates}
        save(out/"result.partial.json",result)
        return original_train(trainer)
    Trainer.train = train
    original_step = torch.optim.Adam.step
    def optimizer_step(optimizer, *args, **kwargs):
        old.check_optimizer_gradients(torch,optimizer)
        active["optimizer_steps"] += 1
        return original_step(optimizer,*args,**kwargs)
    torch.optim.Adam.step = optimizer_step
    for cls in (WeightModelTrainer,FullSequenceTrainer):
        original = cls.do_forward_pass
        def forward(trainer,data,dataset,epoch,_original=original):
            tick=time.monotonic()
            loss,effective=_original(trainer,data,dataset,epoch)
            if loss is None or not bool(torch.isfinite(loss).all()): raise RuntimeError("Nonfinite author loss")
            split="training" if dataset is trainer.training_dataset else "validation"
            row=dict(stage=active["name"],split=split,epoch=epoch,row_ids=host(data["dataset_index"]),
                effective_batch=int(effective),loss=float(loss.detach()),seconds=time.monotonic()-tick)
            active["batches"].append(row)
            with (out/"batches.jsonl").open("a") as handle: handle.write(json.dumps(row)+"\n")
            save(out/"progress.json",dict(stage=active["name"],epoch=epoch,split=split,batches=len(active["batches"]),
                elapsed_seconds=time.monotonic()-active["started"],terminal=False))
            return loss,effective
        cls.do_forward_pass = forward
    for method_name in ("do_training_epoch","do_validation_epoch"):
        original=getattr(Trainer,method_name)
        def epoch(trainer,number,_original=original,_method=method_name):
            active["batches"]=[]
            tick=time.monotonic(); loss=_original(trainer,number)
            split="training" if _method=="do_training_epoch" else "validation"
            count=5000 if split=="training" else 1000
            rows=active["batches"]
            indices=[i for row in rows for i in row["row_ids"]]
            if sorted(indices)!=list(range(count)) or len(rows)!=(79 if split=="training" else 4):
                raise RuntimeError("Source epoch coverage mismatch")
            if len(rows[-1]["row_ids"])!=(8 if split=="training" else 232):
                raise RuntimeError("Source partial-batch mismatch")
            if any(row["epoch"]!=number or row["split"]!=split for row in rows):
                raise RuntimeError("Source epoch instrumentation mismatch")
            record=dict(function=_method,epoch=number,loss=float(loss),seconds=time.monotonic()-tick,
                coverage=dict(rows=count,batches=len(rows),partial_batch=len(rows[-1]["row_ids"])))
            active["epochs"].append(record)
            with (out/"epochs.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(stage=active["name"],**record))+"\n")
            for n,param in active["model"].named_parameters():
                if not bool(torch.isfinite(param).all()): raise RuntimeError("Nonfinite source parameter")
                if n in active["frozen"] and not torch.equal(param,active["frozen"][n]):
                    raise RuntimeError("Source frozen parameter changed")
            active["batches"]=[]
            return loss
        setattr(Trainer,method_name,epoch)
    original_early=EarlyStopping.__call__
    def early(stopper,value):
        returned=original_early(stopper,value)
        if "trainer" in active:
            trainer=active["trainer"]
            rates=[group["lr"] for optimizer in trainer.optimizers for group in optimizer.param_groups]
            number=active["epochs"][-1]["epoch"]
            record=dict(stage=active["name"],epoch=number,validation_loss=float(value),
                learning_rates=rates,early_stopping=vars(stopper).copy(),
                optimizer_steps=active["optimizer_steps"])
            with (out/"controller.jsonl").open("a") as handle: handle.write(json.dumps(record)+"\n")
            latest=out/(active["name"]+"-latest.pt")
            temporary=latest.with_suffix(".pt.tmp")
            torch.save(dict(model=active["model"].state_dict(),
                optimizers=[optimizer.state_dict() for optimizer in trainer.optimizers],
                schedulers=[scheduler.state_dict() for scheduler in trainer.lr_schedulers],
                early_stopping=vars(stopper).copy(),next_epoch=number+1,
                torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),temporary)
            temporary.replace(latest)
        return returned
    EarlyStopping.__call__=early
    for index,(name,cfg,arch) in enumerate(stages):
        seed=args.constructor_seed_base+index
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model=create_model(cfg["model_params"],arch).to(device)
        constructor=host(dict(model.named_parameters()))
        inputs={}
        if name!="weights":
            inputs={k:dict(path=v,sha256=sha(v)) for k,v in cfg["pre_trained_models"].items()}
            model.load_pretrained(cfg["pre_trained_models"],device)
        Path(cfg["save_dir"]).mkdir()
        active.clear(); active.update(name=name,model=model,constructor=constructor,epochs=[],batches=[],
                                     optimizer_steps=0,started=time.monotonic())
        runner.setup_training(cfg,model,cfg["save_dir"],device)
        trainer=active["trainer"]
        summary=dict(stage=name,wall_seconds=time.monotonic()-active["started"],epochs=active["epochs"],
            input_checkpoints=inputs,optimizer_steps=active["optimizer_steps"],splits={},
            source_skipped_gradients=trainer.total_number_of_skipped_gradients)
        completed=len(active["epochs"])//2
        if len(active["epochs"])!=2*completed or completed<1:
            raise RuntimeError("Incomplete source epochs")
        cap=cfg["training_params"]["epochs"]
        stopped=bool(trainer.early_stopping.early_stop)
        if completed!=cap and not stopped: raise RuntimeError("Source trainer stopped without its rule")
        if args.epochs!="full" and completed!=cap: raise RuntimeError("Diagnostic ended early")
        summary.update(completed_epochs=completed,source_cap=source_configs[name]["epochs"],
            termination="SOURCE_EARLY_STOP" if stopped else ("SOURCE_CAP" if args.epochs=="full" else "DIAGNOSTIC_CAP"),
            full_protocol=args.epochs=="full",latest_checkpoint=str(out/(name+"-latest.pt")))
        for split,function in (("training","do_training_epoch"),("validation","do_validation_epoch")):
            epochs=[e for e in active["epochs"] if e["function"]==function]
            if [e["epoch"] for e in epochs]!=list(range(completed)): raise RuntimeError("Source epoch index mismatch")
            summary["splits"][split]=dict(epochs=completed,**epochs[-1]["coverage"],weighted_loss=epochs[-1]["loss"])
        expected=79*completed*len(trainer.optimizers)
        if summary["optimizer_steps"]!=expected or summary["source_skipped_gradients"]!=0:
            raise RuntimeError("Source optimizer skipped or duplicated an update")
        for n,p in model.named_parameters():
            assert bool(torch.isfinite(p).all())
            if n in active["frozen"]: assert torch.equal(p,active["frozen"][n])
        summary["frozen_parameters_unchanged"]=True
        summary["checkpoints"]={str(p):sha(p) for p in (out/name/"models").glob("*_best.pt")}
        if not summary["checkpoints"]: raise RuntimeError("Missing source best checkpoint")
        result["stages"].append(summary)
        save(out/"result.partial.json",result)
    result["source_training_protocol_complete"]=args.epochs=="full"
    result["status"]="PASS"
    save(out/"result.json",result)
    save(out/"progress.json",dict(terminal=True,status="PASS"))

if __name__=="__main__":
    try:
        main()
    except Exception as error:
        output=Path(sys.argv[sys.argv.index("--output")+1]).resolve()
        if output.exists():
            partial=output/"result.partial.json"
            result=json.loads(partial.read_text()) if partial.exists() else {}
            result.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
            save(output/"result.json",result)
            save(output/"progress.json",dict(terminal=True,status="FAIL"))
        raise
