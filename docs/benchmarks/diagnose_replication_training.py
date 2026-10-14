#!/usr/bin/env python3
"""Diagnostic-only author PyTorch/NumPy training reference; never BayesFilter runtime.

Runs short wiring/cost probes or an explicitly budgeted complete source trial.
An author-reference trial does not establish learned quality or paper replication.
"""
from __future__ import annotations
import argparse
import copy
import difflib
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback


class SourceNumericalFailure(RuntimeError):
    """Nonfinite author candidate; distinct from a harness failure."""


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_yaml_dimensions(value):
    """JSON stringifies YAML integer dimension keys; restore the source contract."""
    if isinstance(value, list):
        return [restore_yaml_dimensions(item) for item in value]
    if isinstance(value, dict):
        result = {key: restore_yaml_dimensions(item) for key, item in value.items()}
        if "dims" in result and isinstance(result["dims"], dict):
            result["dims"] = {int(key): item for key, item in result["dims"].items()}
        return result
    return value



def execution_protocol(mode, source_caps, source_sizes):
    """Pure protocol validation; trial_smoke exercises guards without a completion claim."""
    guarded = mode in ("source_trial", "trial_smoke")
    if guarded:
        if source_caps != {"pretraining": 400, "full_training": 800}:
            raise ValueError("Complete-trial source epoch caps changed")
        if source_sizes != {"training": 5000, "validation": 1000}:
            raise ValueError("Complete-trial source dataset sizes changed")
    return {
        "mode": mode, "guarded": guarded,
        "source_caps": source_caps,
        "epochs": dict(source_caps) if mode == "source_trial" else
                  {label: 1 if label == "pretraining" else 2 for label in source_caps},
        "dataset_sizes": dict(source_sizes) if guarded else
                         ({"training": 5000, "validation": 1000} if mode == "cost"
                          else {"training": 128, "validation": 256}),
        "data_seed": 925201 if guarded else 924011,
        "complete_trial_eligible": mode == "source_trial",
    }


def verify_epoch_completion(records, label, cap, early_stop):
    train = [r["epoch"] for r in records if r["phase"] == label and r["function"] == "do_training_epoch"]
    validation = [r["epoch"] for r in records if r["phase"] == label and r["function"] == "do_validation_epoch"]
    if not train or train != validation or train != list(range(len(train))) or len(train) > cap:
        raise RuntimeError("Source train/validation epoch coverage mismatch")
    if len(train) < cap and not early_stop:
        raise RuntimeError("Source returned early without its early-stopping condition")
    return {"epochs_completed": len(train), "epoch_cap": cap,
            "termination": "SOURCE_EARLY_STOP" if early_stop else "SOURCE_EPOCH_CAP"}


def check_optimizer_gradients(torch, optimizer):
    gradients = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
        raise SourceNumericalFailure("Adam received absent/nonfinite gradients")


def source_copy(author, output, torch):
    """Isolate all writes and preserve exact source/compatibility provenance."""
    src = output / "author-src"
    src.mkdir()
    records = {}
    for path in sorted((author / "src").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml"):
            relative = path.relative_to(author / "src")
            target = src / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            records[str(relative)] = digest(path)
    patches = []
    scheduler = src / "trainers/trainer_utils.py"
    before = scheduler.read_text()
    after = before
    base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    if "verbose" not in inspect.signature(base_scheduler).parameters:
        old = '        super(CustomReduceLROnPlateau, self).__init__(*args, **kwargs)'
        assert after.count(old) == 1
        after = after.replace(old, '        kwargs.pop("verbose", None)  # Removed logging-only API keyword\n' + old)
    if not hasattr(base_scheduler, "is_better"):
        assert hasattr(base_scheduler, "_is_better")
        old = "        if self.is_better(current, self.best):"
        assert after.count(old) == 1
        after = after.replace(old, "        if self._is_better(current, self.best):")
    scheduler.write_text(after)
    patches += list(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                     fromfile="author/trainers/trainer_utils.py", tofile="adapted/trainers/trainer_utils.py"))
    dataset = src / "datasets/bearings_only_dataset.py"
    before = dataset.read_text()
    old = "all_data = torch.load(self.save_location)"
    assert before.count(old) == 1
    # Only locally generated, isolated reference datasets enter this source route.
    after = before.replace(old, "all_data = torch.load(self.save_location, weights_only=False)")
    dataset.write_text(after)
    patches += list(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile="author/datasets/bearings_only_dataset.py",
                    tofile="adapted/datasets/bearings_only_dataset.py"))
    (output / "reference-compatibility.patch").write_text("".join(patches))
    save(output / "author-source-hashes.json", records)
    return src


def check_scheduler_compatibility(torch, scheduler_class, device):
    for mode in ("min", "max"):
        for threshold_mode in ("rel", "abs"):
            params = dict(mode=mode, threshold_mode=threshold_mode, threshold=.001,
                          factor=.1, patience=1, cooldown=1, min_lr=0.)
            left = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0., device=device))], lr=1.)
            right = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0., device=device))], lr=1.)
            adapted = scheduler_class(left, start_epoch=0, verbose=True, **params)
            reference = torch.optim.lr_scheduler.ReduceLROnPlateau(right, **params)
            for value in (1., 1., 1., .9, .9, .9, -.2, -.2, -.2, .2, .2, .2):
                adapted.step(value)
                reference.step(value)
                assert left.param_groups[0]["lr"] == right.param_groups[0]["lr"]
                assert adapted.best == reference.best
                assert adapted.num_bad_epochs == reference.num_bad_epochs
                assert adapted.cooldown_counter == reference.cooldown_counter
    parameter = torch.nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.SGD([parameter], lr=1.)
    scheduler = scheduler_class(optimizer, mode="min", start_epoch=0,
        threshold=.001, factor=.1, patience=1, cooldown=0, min_lr=0., verbose=True)
    flags = [scheduler.step(value) for value in (1., 1., 1.)]
    assert flags == [False, False, True] and math.isclose(optimizer.param_groups[0]["lr"], .1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("wiring", "cost", "trial_smoke", "source_trial"), default="wiring")
    parser.add_argument("--source-arm", choices=("mdpf", "tg", "irg", "dis"), default="mdpf")
    args = parser.parse_args()
    if args.source_arm != "mdpf" and args.mode != "cost":
        parser.error("Comparator arms are bounded cost references only")
    output = args.output.resolve()
    author = args.author_root.resolve()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("Use isolated python -I -B")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("This reference training probe requires escalated GPU access")
    from_diag = Path(__file__).with_name("diagnose_replication_stage0.py")
    spec = importlib.util.spec_from_file_location("source_config_check", from_diag)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    if args.source_arm == "mdpf":
        config_path = args.configuration
    else:
        resolved = output / "source-configuration"
        resolved.mkdir()
        diagnostic.mdpf(author, resolved, args.source_arm, configuration_only=True)
        config_path = resolved / "resolved-configuration.json"
    original = restore_yaml_dimensions(json.loads(config_path.read_text()))
    diagnostic.check_method(original["architecture"]["dpf_full_training"], args.source_arm)

    nvidia = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"],
        text=True, timeout=20)
    (output / "nvidia-smi.txt").write_text(nvidia)
    import torch
    import numpy as np  # Explicit independent author reference only.
    if not torch.cuda.is_available():
        raise RuntimeError("Reference interpreter cannot use CUDA")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    chosen = next((i for i, name in enumerate(names) if "4080" in name), 0)
    torch.cuda.set_device(chosen)
    device = torch.device("cuda", chosen)
    probe = torch.arange(8, dtype=torch.float32, device=device, requires_grad=True)
    probe.square().sum().backward()
    assert torch.equal(probe.grad, 2 * probe.detach())
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    save(output / "environment.json", {
        "interpreter": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "visible_devices": names, "selected_device": str(device),
        "selected_device_name": names[chosen], "actual_interpreter_cuda_probe": "PASS",
        "role": "Independent author reference, eager PyTorch; not BayesFilter GPU/XLA",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "loaded_cpp_libraries": [line.strip() for line in Path("/proc/self/maps").read_text().splitlines()
                                 if "libstdc++" in line or "libgcc_s" in line]})
    src = source_copy(author, output, torch)
    os.chdir(output)
    sys.path.insert(0, str(author / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
    sys.path.insert(0, str(src))
    model_spec = importlib.util.find_spec("models")
    if list(model_spec.submodule_search_locations or []) != [str(src / "models")]:
        raise RuntimeError("Author models namespace is shadowed")
    from models.model_creator import create_model
    from trainers.training_runner import setup_training
    from trainers.trainer_utils import CustomReduceLROnPlateau

    # Compare cadence/state against the installed scheduler and the known source return flag.
    check_scheduler_compatibility(torch, CustomReduceLROnPlateau, device)

    full_name = ("full_dpf_full_training_fixed_bands_mse_000" if args.source_arm == "dis"
                 else "full_dpf_full_training_fixed_bands_000")
    full = copy.deepcopy(next(e["parameters"] for e in original["experiments"] if e["name"] == full_name))
    expected_loss = "MSE" if args.source_arm == "dis" else "KDE_NLL"
    if full["training_params"]["loss_params"]["loss_type"] != expected_loss:
        raise RuntimeError("Source comparator objective mismatch")
    pre_entry = original["pretraining_entry"]
    pre = copy.deepcopy(next(iter(pre_entry["experiments"][0].values())))
    pre_arch = copy.deepcopy(original["architecture"])
    pre_arch.update(pre_entry["models"])
    data = copy.deepcopy(full["resolved_dataset_parameters"]["dataset_params"])
    source_dataset_sizes = copy.deepcopy(data["dataset_sizes"])
    data["dataset_saves"] = {split: str(output / "datasets" / (split + ".pt"))
                             for split in ("training", "validation")}
    if data["subsequence_length"] != 17 or data["sparse_ground_truth_keep_modulo"] != 4:
        raise RuntimeError("Wrong source sequence/label protocol")
    stages = [("pretraining", pre, pre_arch, 924012),
              ("full_training", full, original["architecture"], 924013)]
    if args.source_arm == "dis":
        bandwidth = copy.deepcopy(next(e["parameters"] for e in original["experiments"]
            if e["name"] == "full_dpf_full_training_fixed_bands_bands_000"))
        if bandwidth["training_params"]["loss_params"]["loss_type"] != "KDE_NLL":
            raise RuntimeError("DIS bandwidth stage must use source likelihood loss")
        bandwidth["pre_trained_models"] = {
            "dpf_model": str(output / "full_training/models/full_dpf_model_best.pt")}
        stages.append(("bandwidth_fit", bandwidth, original["architecture"], 924014))
    protocol = execution_protocol(args.mode,
        {label: parameters["training_params"]["epochs"] for label, parameters, _, _ in stages},
        source_dataset_sizes)
    data["dataset_sizes"] = protocol["dataset_sizes"]
    stages = [(label, parameters, architecture, protocol["data_seed"] + i + 1)
              for i, (label, parameters, architecture, _) in enumerate(stages)]
    save(output / "protocol.json", protocol)
    configurations = {}
    for label, parameters, architecture, seed in stages:
        stage_epochs = protocol["epochs"][label]
        parameters.pop("dataset_params_file", None)
        parameters.pop("resolved_dataset_parameters", None)
        parameters.pop("common_parameters_file", None)
        parameters["dataset_params"] = copy.deepcopy(data)
        parameters["save_dir"] = str(output / label)
        parameters["device"] = str(device)
        parameters["do_run"] = True
        parameters["training_params"]["epochs"] = stage_epochs
        parameters["training_params"]["num_cpu_cores_for_dataloader"] = 1
        if args.mode == "wiring":
            parameters["training_params"]["save_intermediate_models"] = True
        configurations[label] = parameters
    pretrained = full["pre_trained_models"]
    full["pre_trained_models"] = {
        key: str(output / "pretraining/models" / Path(value).name)
        for key, value in pretrained.items()}
    save(output / "diagnostic-configurations.json", {
        "source_arm": args.source_arm, "resolved_source_configuration_sha256": digest(config_path),
        "configurations": configurations, "seeds": {"data": protocol["data_seed"], **{v[0]: v[3] for v in stages}},
        "overrides": [str(data["dataset_sizes"]) + " source sequences", ("original epoch caps and source early stopping" if args.mode == "source_trial"
                         else "1 pretraining/2 epochs per later stage"),
                      "one persistent loader worker", "isolated absolute data/save/pretrain paths",
                      "intermediate checkpoint policy: source default in cost mode; enabled in wiring mode"],
        "unchanged": ["source architecture, loss and learning rates", "T=17", "full N=25",
                      "training batch 64 / validation batch 256", "labels/BPTT modulo 4"],
        "purpose": ("Complete single source NLL training trial; quality remains unassessed"
                    if args.mode == "source_trial" else "Execution wiring and provisional component costs only")})
    np.random.seed(protocol["data_seed"])
    random.seed(protocol["data_seed"])
    torch.manual_seed(protocol["data_seed"])
    calls, epochs, frame_times, trainers, problems = {}, [], {}, [], []
    calls_by_phase = {}
    gradient_checks = 0
    active_label = ""
    def profile(frame, event, arg):
        nonlocal gradient_checks
        filename = Path(frame.f_code.co_filename)
        name = frame.f_code.co_name
        if filename.is_relative_to(src):
            key = str(filename.relative_to(src)) + ":" + name
            if event == "call":
                calls[key] = calls.get(key, 0) + 1
                if name in ("do_training_epoch", "do_validation_epoch"):
                    torch.cuda.synchronize(device)
                    frame_times[id(frame)] = time.monotonic()
            elif event == "return":
                if name == "create_trainer" and arg is not None:
                    trainers.append(arg)
                elif name == "create_problem_training" and arg is not None:
                    problems.append(arg)
                elif id(frame) in frame_times:
                    torch.cuda.synchronize(device)
                    if arg is None:  # Preserve the original exception on an aborted epoch.
                        frame_times.pop(id(frame))
                        return
                    loss = float(arg)
                    if not math.isfinite(loss):
                        raise SourceNumericalFailure("Nonfinite source epoch loss")
                    epochs.append({"phase": active_label, "function": name,
                                   "epoch": frame.f_locals["epoch"], "loss": loss,
                                   "seconds": time.monotonic() - frame_times.pop(id(frame))})
        elif event == "call" and name == "step" and filename.name == "adam.py":
            opt = frame.f_locals.get("self")
            if isinstance(opt, torch.optim.Adam):
                grads = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
                if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    raise AssertionError("Adam received absent/nonfinite gradients")
                gradient_checks += 1

    results = {}
    sparse_events, restorations = [], []
    if args.mode != "wiring":
        # Sparse stage boundaries; no pervasive frame profiler.
        from trainers import training_runner as runner_module
        from trainers.trainer import Trainer
        from datasets.bearings_only_dataset import BearingsOnlyDataset
        from models.kde_particle_filter.kde_particle_filter import KDEParticleFilter, KernelDensityEstimator
        from models.comparison_model_base import DiscreteSamplingComparisonModelBase
        from models.importance_sampling_particle_filter.importance_sampling_particle_filter import ImportanceSamplingParticleFilterLearnedBand
        from matplotlib.figure import Figure
        def instrument(owner, name, key, capture=None, timed=True, epoch=False):
            original_function = getattr(owner, name)
            restorations.append((owner, name, original_function))
            def wrapped(*items, **kwargs):
                nonlocal gradient_checks
                calls[key] = calls.get(key, 0) + 1
                phase_calls = calls_by_phase.setdefault(active_label, {})
                phase_calls[key] = phase_calls.get(key, 0) + 1
                if timed:
                    torch.cuda.synchronize(device)
                    started = time.monotonic()
                if key == "torch.optim.Adam:step" and protocol["guarded"]:
                    check_optimizer_gradients(torch, items[0])
                value = original_function(*items, **kwargs)
                if capture is not None:
                    capture.append(value)
                if key == "torch.optim.Adam:step":
                    gradient_checks += 1  # Count only; avoid per-gradient synchronization in cost mode.
                if timed:
                    torch.cuda.synchronize(device)
                    event = {"phase": active_label, "function": key, "seconds": time.monotonic()-started}
                    sparse_events.append(event)
                    with (output / "events.jsonl").open("a") as handle:
                        handle.write(json.dumps(event, allow_nan=False) + "\n")
                    if epoch:
                        loss = float(value)
                        if not math.isfinite(loss):
                            raise SourceNumericalFailure("Nonfinite source epoch loss")
                        epochs.append({"phase": active_label, "function": name,
                                       "epoch": items[1], "loss": loss, "seconds": event["seconds"]})
                        trainer = items[0]
                        if protocol["guarded"]:
                            model_parameters = [p for model, _ in trainer.models.values() for p in model.parameters()]
                            if not bool(torch.stack([torch.isfinite(p).all() for p in model_parameters]).all()):
                                raise SourceNumericalFailure("Nonfinite parameter at epoch boundary")
                        save(output / "progress.json", {
                            "phase": active_label, "last_epoch_event": epochs[-1],
                            "epoch_cap": protocol["epochs"][active_label],
                            "optimizer_steps": gradient_checks, "gradient_checks_enabled": protocol["guarded"],
                            "learning_rates": [g["lr"] for opt in trainer.optimizers for g in opt.param_groups],
                            "source_skipped_gradients": trainer.total_number_of_skipped_gradients,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "updated_unix_seconds": time.time(), "terminal": False})
                return value
            wrapped.__wrapped__ = original_function
            setattr(owner, name, wrapped)
        instrument(runner_module, "create_problem_training", "trainers/training_runner.py:create_problem_training", problems)
        instrument(runner_module, "create_trainer", "trainers/training_runner.py:create_trainer", trainers)
        instrument(Trainer, "__init__", "trainers/trainer.py:__init__")
        instrument(Trainer, "train", "trainers/trainer.py:train")
        instrument(Trainer, "do_training_epoch", "trainers/trainer.py:do_training_epoch", epoch=True)
        instrument(Trainer, "do_validation_epoch", "trainers/trainer.py:do_validation_epoch", epoch=True)
        instrument(BearingsOnlyDataset, "generate_dataset", "datasets/bearings_only_dataset.py:generate_dataset")
        instrument(KDEParticleFilter, "resample_particles", "models/kde_particle_filter/kde_particle_filter.py:resample_particles", timed=False)
        instrument(DiscreteSamplingComparisonModelBase, "resample_particles", "models/comparison_model_base.py:resample_particles", timed=False)
        instrument(ImportanceSamplingParticleFilterLearnedBand, "specific_resampling_method", "models/importance_sampling_particle_filter/importance_sampling_particle_filter.py:specific_resampling_method", timed=False)
        instrument(KernelDensityEstimator, "inject_gradient", "author_kde:inject_gradient", timed=False)
        instrument(torch.optim.Adam, "step", "torch.optim.Adam:step", timed=False)
        instrument(torch, "save", "external:torch.save")
        instrument(Figure, "savefig", "external:Figure.savefig")
    try:
        if args.mode == "wiring":
            sys.setprofile(profile)
        for label, parameters, architecture, seed in stages:
            active_label = label
            torch.manual_seed(seed)
            model = create_model(parameters["model_params"], architecture).to(device)
            input_checkpoints = {}
            if label != "pretraining":
                for path in parameters["pre_trained_models"].values():
                    if not Path(path).is_file():
                        raise AssertionError("Pretraining did not produce required checkpoint")
                input_checkpoints = {key: {"path": path, "sha256": digest(path)}
                                     for key, path in parameters["pre_trained_models"].items()}
                model.load_pretrained(parameters["pre_trained_models"], device)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            start = time.monotonic()
            Path(parameters["save_dir"]).mkdir()
            if args.mode != "wiring":
                calls["trainers/training_runner.py:setup_training"] = calls.get("trainers/training_runner.py:setup_training", 0) + 1
            setup_training(parameters, model, parameters["save_dir"], device)
            torch.cuda.synchronize(device)
            changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise SourceNumericalFailure("Nonfinite parameters after source training")
            if not changed:
                raise AssertionError("No update after source training")
            trainer = trainers[-1]
            completion = verify_epoch_completion(epochs, label, parameters["training_params"]["epochs"],
                                                  trainer.early_stopping.early_stop)
            optimized_ids = {id(p) for opt in trainer.optimizers for group in opt.param_groups for p in group["params"]}
            frozen_changed = [name for name, p in model.named_parameters()
                              if id(p) not in optimized_ids and name in changed]
            if frozen_changed:
                raise AssertionError("Frozen parameters changed: " + repr(frozen_changed))
            if label == "bandwidth_fit" and any("bandwidth" not in name for name in changed):
                raise AssertionError("DIS bandwidth fit altered another model")
            resampler = inspect.unwrap(model.resample_particles.__func__)
            results[label] = {"status": "PASS", "wall_seconds": time.monotonic()-start, **completion,
                "source_skipped_gradients": trainer.total_number_of_skipped_gradients,
                "model_class": type(model).__name__, "loss_class": type(trainer.loss_function).__name__,
                "loss_parameters": parameters["training_params"]["loss_params"],
                "resampler_file": inspect.getsourcefile(resampler), "resampler_line": resampler.__code__.co_firstlineno,
                "input_checkpoints": input_checkpoints, "frozen_parameter_invariance": "PASS",
                "changed_parameter_tensors": len(changed), "changed_parameters": changed,
                "train_batches": len(trainer.train_loader), "validation_batches": len(trainer.validation_loader),
                "training_batch_size": trainer.training_batch_size,
                "validation_batch_size": trainer.validation_batch_size,
                "optimizer_count": len(trainer.optimizers), "epochs": parameters["training_params"]["epochs"]}
            save(output / "completed-phases.json", {"source_arm": args.source_arm, "phases": results,
                 "calls_by_phase": calls_by_phase, "epochs": epochs})
    finally:
        sys.setprofile(None)
        for owner, name, original_function in reversed(restorations):
            setattr(owner, name, original_function)
        for trainer in trainers:
            for loader in (trainer.train_loader, trainer.validation_loader):
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()
    for split in ("training", "validation"):
        datasets = [p.get_dataset(split) for p in problems]
        for right in datasets[1:]:
            assert torch.equal(datasets[0].car_states, right.car_states)
            assert torch.equal(datasets[0].sensor_obs, right.sensor_obs)
    if calls.get("datasets/bearings_only_dataset.py:generate_dataset") != 2:
        raise AssertionError("Data were unexpectedly regenerated instead of loaded")
    for endpoint in ("trainers/training_runner.py:setup_training",
                     "trainers/trainer.py:__init__", "trainers/trainer.py:train"):
        if calls.get(endpoint) != len(stages):
            raise AssertionError("Missing real source endpoint: " + endpoint)
    expected_resampler = ("models/comparison_model_base.py:resample_particles" if args.source_arm == "dis"
                          else "models/kde_particle_filter/kde_particle_filter.py:resample_particles")
    if not calls.get(expected_resampler) or not gradient_checks:
        raise AssertionError("Missing resampling or optimizer evidence")
    if args.mode != "wiring":
        implicit_calls = calls_by_phase["full_training"].get("author_kde:inject_gradient", 0)
        if (args.source_arm == "irg") != (implicit_calls > 0):
            raise AssertionError("IRG consumer identity mismatch")
        if args.source_arm == "dis" and not calls_by_phase["full_training"].get(
                "models/importance_sampling_particle_filter/importance_sampling_particle_filter.py:specific_resampling_method"):
            raise AssertionError("DIS training did not call its importance-weight resampler")
    checkpoints = {}
    for path in sorted(output.glob("*/models/*_best.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state and all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v))
        checkpoints[str(path.relative_to(output))] = digest(path)
    source_hashes = {key.rsplit(":", 1)[0]: digest(src / key.rsplit(":", 1)[0]) for key in calls
                     if (src / key.rsplit(":", 1)[0]).is_file()}
    timings = {"epochs": epochs, "profiler": args.mode == "wiring", "mode": args.mode,
               "warning": ("Complete single source trial; no stochastic ranking or full-suite budget claim"
                           if args.mode == "source_trial" else "Short reference timing; no full budget or ranking"),
               "dataset_sizes": data["dataset_sizes"], "sparse_events": sparse_events,
               "unmeasured": ["untouched T=150 evaluation", "other comparator arms not measured by this child", "Appendix B", "iAPF",
                              "long-training behavior, optional MSE refinement, complete comparator evaluations"]}
    save(output / "source-training-wiring.json", {"source_arm": args.source_arm, "calls": calls, "calls_by_phase": calls_by_phase, "executed_source_sha256": source_hashes,
         "optimizer_gradient_checks": gradient_checks if args.mode == "wiring" or protocol["guarded"] else 0,
         "optimizer_steps": gradient_checks, "checkpoint_sha256": checkpoints,
         "dataset_round_trip": "PASS", "scheduler_cadence": "PASS"})
    save(output / "component-timings.json", timings)
    save(output / "sparse-timings.json", {"events": sparse_events, "profiler": False,
         "nesting": "Inclusive timings; do not sum nested events."})
    save(output / "progress.json", {"terminal": True, "phases": results,
         "updated_unix_seconds": time.time(), "optimizer_steps": gradient_checks})
    save(output / "result.json", {
        "check": "PASS", "source_arm": args.source_arm, "full_source_training_call_chain": ("PASS_ON_COMPLETE_SOURCE_TRIAL" if args.mode == "source_trial"
                                                                  else "PASS_ON_SHORT_REFERENCE"),
        "source_training_protocol_complete": args.mode == "source_trial",
        "protocol": protocol,
        "phases": results, "actual_interpreter_cuda_probe": "PASS",
        "source_dataset_round_trip": "PASS", "scheduler_compatibility": "PASS",
        "optimizer_gradient_checks": gradient_checks if args.mode == "wiring" or protocol["guarded"] else 0,
        "optimizer_steps": gradient_checks, "checkpoints_reloaded": len(checkpoints), "mode": args.mode,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "backend_role": "Independent author PyTorch/NumPy reference; eager, no XLA",
        "scientific_replication": "INCOMPLETE", "learning_quality": "NOT_ESTABLISHED",
        "gradient_estimator_correctness": "NOT_ESTABLISHED", "full_budget": "NOT_MEASURED"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output")+1])
            (output / "training-error.log").write_text(traceback.format_exc())
            save(output / "result.json", {"check": "FAILED", "error": repr(exc),
                 "scientific_replication": "INCOMPLETE", "classification": ("source_candidate_numerical_failure" if isinstance(exc, SourceNumericalFailure)
                                    else "reference_environment_failure" if isinstance(exc, ImportError) and
                                    ("CXXABI_" in str(exc) or "GLIBCXX_" in str(exc))
                                    else "reference_execution_failure")})
        raise
