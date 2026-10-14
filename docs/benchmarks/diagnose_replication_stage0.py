#!/usr/bin/env python3
"""Stage 0 diagnostics only: author-reference configuration and GPU/XLA smoke.

Original PyTorch/NumPy author code is an independent reference, never a
BayesFilter runtime. Configuration extraction cannot prove the training chain.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_ARMS = {
    "mdpf": ("experiment0002_importance", "KDEParticleFilter", True, "ImportanceSampling"),
    "tg": ("experiment0001", "KDEParticleFilter", False, None),
    "irg": ("experiment0002_implicit", "KDEParticleFilter", True, "ReparameterizationGradients"),
    "dis": ("importance_sampling_pf_learned_band", "ImportanceSamplingParticleFilterLearnedBand", False, None),
}


def check_method(model, arm="mdpf"):
    _, model_type, differentiable, gradient_method = SOURCE_ARMS[arm]
    expected = {
        "type": model_type, "use_differentiable_resampling": differentiable,
        "decouple_weights_for_resampling": False,
        "decouple_bandwidths_for_resampling": False,
    }
    if differentiable:
        expected["differentiable_resampling_method"] = gradient_method
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError("Wrong base source method " + arm + ": " + key)
    distributions = [v["distribution_type"] for k, v in
                     sorted(model["kde"]["dims"].items(), key=lambda pair: int(pair[0]))]
    if distributions != ["Normal", "Normal", "Von_Mises"]:
        raise ValueError("Wrong bearings kernel family")


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("Author configuration helper missing")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def mdpf(author, output, arm="mdpf", configuration_only=False):
    if not configuration_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This author-reference diagnostic must hide GPUs before import")
    import yaml

    if not sys.flags.isolated:
        raise RuntimeError("Launch the author reference with python -I to exclude unrelated PYTHONPATH projects")
    src = author / "src"
    relative = Path("experiments/bearings_only_experiments_methodical_25_particles_multiple_runs")
    experiment = src / relative
    entry_path = experiment / SOURCE_ARMS[arm][0] / "config.yaml"
    generator_path = experiment / "scripts/generate_multiple_runs_config_files.py"
    consumed = set()

    def read(path):
        path = Path(path).resolve()
        consumed.add(path)
        return yaml.load(path.read_text(), Loader=yaml.FullLoader)

    # Match main.py's shallow architecture updates and subsequent recursive
    # common-parameter / experiment / local-override precedence.
    entry = read(entry_path)
    architecture = {}
    for filename in entry["model_files"]:
        architecture.update(read(src / filename)["models"])
    architecture.update(entry["models"])
    check_method(architecture["dpf_full_training"], arm)
    generator_tree = ast.parse(generator_path.read_text())
    trial_values = [node.value.value for node in ast.walk(generator_tree)
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and any(isinstance(t, ast.Name) and t.id == "NUMBER_OF_TRIALS"
                            for t in node.targets)]
    if trial_values != [11]:
        raise ValueError("Unexpected author repeated-run protocol")
    generator = runpy.run_path(str(generator_path))
    consumed.add(generator_path)
    consumed.add(src / "main.py")
    helpers = source_functions(src / "main.py",
                               {"recursive_update", "load_common_parameters_file",
                                "make_save_dir"}, {"yaml": yaml})
    all_experiments = []
    include_state = []
    for filename in entry.get("experiments_import", []):
        original = src / filename
        if original.exists():
            expanded = read(original)
        else:
            base_path = original.parent.parent / original.name
            base = read(base_path)
            expanded = {"experiments": generator["generate_tasks_one_large_config"](
                copy.deepcopy(base["experiments"]), trial_values[0])}
        generated_path = output / ("generated-" + original.name)
        generated_path.write_text(yaml.safe_dump(expanded, sort_keys=False))
        include_state.append({"original": str(original), "original_present": original.exists(),
                              "resolved_copy": str(generated_path)})
        all_experiments.extend(expanded["experiments"])
    all_experiments.extend(entry.get("experiments", []))
    overrides = {}
    for item in entry.get("experiments_local_override", []):
        name, value = next(iter(item.items()))
        if name in overrides:
            raise ValueError("Duplicate local override")
        overrides[name] = value
    resolved = []
    paths = []
    os.chdir(src)
    for item in all_experiments:
        name, value = next(iter(item.items()))
        value = copy.deepcopy(value)
        if "common_parameters_file" in value:
            consumed.add((src / value["common_parameters_file"]).resolve())
            value = helpers["load_common_parameters_file"](value["common_parameters_file"], value)
        if name in overrides:
            value = helpers["recursive_update"](value, copy.deepcopy(overrides[name]))
        root_save = entry.get("save_dir_root", "")
        replacement = entry.get("experiment_name_replacement", "")
        value["save_dir"] = helpers["make_save_dir"](value["save_dir"], root_save)
        pretrained = value.get("pre_trained_models_local_override",
                               value.get("pre_trained_models", {}))
        for label, filename in pretrained.items():
            final = filename.replace("<root_save_dir>", root_save).replace(
                "<experiment_name_replacement>", replacement)
            paths.append({"experiment": name, "role": label, "path": str(src / final),
                          "exists": (src / final).exists()})
        if "dataset_params_file" in value:
            value["resolved_dataset_parameters"] = read(src / value["dataset_params_file"])
        resolved.append({"name": name, "parameters": value})
    pretraining = read(experiment / "individual_training/config.yaml")
    save(output / "resolved-configuration.json", {
        "architecture": architecture, "experiments": resolved, "source_arm": arm,
        "source_sha256": {str(p): sha(p) for p in sorted(consumed)},
        "pretraining_entry": pretraining, "source_includes": include_state,
        "note": "Inspection copy; original do_run/save/pretrain choices preserved; not a launch config"})
    save(output / "referenced-checkpoints.json", paths)
    if configuration_only:
        save(output / "result.json", {
            "check": "PASS", "source_arm": arm, "trials": trial_values[0],
            "configuration_only": True, "numerical_framework_imported": False,
            "scientific_replication": "INCOMPLETE"})
        return
    dataset_paths = []
    for path in sorted(consumed):
        if path.name.startswith("dataset_params"):
            data = read(path)["dataset_params"]
            for kind, filename in data.get("dataset_saves", {}).items():
                candidate = src / filename
                dataset_paths.append({"split": kind, "path": str(candidate),
                                      "exists": candidate.exists()})
    # Try the real imported modules, not extracted resampling functions.
    # Importing main does not execute its guarded main() or training.
    sys.path.insert(0, str(author / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
    sys.path.insert(0, str(src))
    models_spec = importlib.util.find_spec("models")
    package_paths = list(models_spec.submodule_search_locations or []) if models_spec else []
    if package_paths != [str(src / "models")]:
        raise RuntimeError("Author models package is shadowed: " + repr(package_paths))
    modules = ["yaml", "torch", "numpy", "PIL", "matplotlib", "scipy", "tqdm", "torchviz", "pynvml"]
    dependencies = {name: importlib.util.find_spec(name) is not None for name in modules}
    imports = {}
    for name in ("models.kde_particle_filter.kde_particle_filter", "main"):
        try:
            importlib.import_module(name)
            imports[name] = {"status": "PASS"}
        except Exception as exc:
            imports[name] = {"status": "FAILED", "error": repr(exc)}
            (output / (name.replace(".", "-") + "-import.log")).write_text(traceback.format_exc())
    construction = {"status": "NOT_CHECKED", "reason": "Real model-factory import unavailable"}
    if imports["main"]["status"] == "PASS":
        try:
            import torch
            torch.manual_seed(924001)
            factory = importlib.import_module("models.model_creator")
            full = next(e["parameters"] for e in resolved
                        if e["name"] == "full_dpf_full_training_fixed_bands_000")
            model = factory.create_model(full["model_params"], architecture).to("cpu")
            construction = {"status": "PASS", "class": type(model).__name__,
                            "full_training_forward_backward": "NOT_CHECKED"}
            try:
                construction["sequence_loss_backward_fixture"] = sequence_fixture(model, full, src, output)
            except Exception as exc:
                construction["sequence_loss_backward_fixture"] = {"status": "FAILED", "error": repr(exc)}
                (output / "source-sequence-wiring-error.log").write_text(traceback.format_exc())
        except Exception as exc:
            construction = {"status": "FAILED", "error": repr(exc)}
            (output / "model-construction.log").write_text(traceback.format_exc())
    active = [e["name"] for e in resolved if e["parameters"].get("do_run", True)]
    active_training = [e["name"] for e in resolved if e["parameters"].get("do_run", True)
                       and e["parameters"]["experiement_type"] == "training"]
    missing = sorted({p["path"] for p in paths if not p["exists"]})
    blockers = []
    if any(v["status"] == "FAILED" for v in imports.values()):
        blockers.append("Real author module imports fail")
    if not active_training:
        blockers.append("Full training is disabled in the source configuration")
    if missing:
        blockers.append("Referenced pretrained / full-model checkpoints are absent")
    if construction.get("sequence_loss_backward_fixture", {}).get("status") == "FAILED":
        blockers.append("Source sequence/loss/backward fixture FAILED; inspect its error log")
    blockers += ["Full trainer construction/dataset/optimizer endpoint NOT_CHECKED",
                 "Appendix B learning and paper bearings experiments NOT_RUN",
                 "Full training and comparison budget NOT_MEASURED"]
    result = {
        "check": "INCOMPLETE", "configuration_resolution": "PASS",
        "scientific_replication": "INCOMPLETE", "gpu_intentionally_hidden": True,
        "backend_role": "Original author PyTorch/NumPy independent reference",
        "interpreter": sys.executable,
        "isolated_python": bool(sys.flags.isolated), "models_package_paths": package_paths,
        "author_root": str(author), "trials": trial_values[0],
        "expanded_experiments": len(resolved), "active_experiments": active,
        "active_training": active_training, "missing_checkpoint_count": len(missing),
        "missing_checkpoints": missing, "dataset_paths": dataset_paths,
        "dependencies": dependencies, "real_module_imports": imports,
        "model_construction": construction, "blockers": blockers,
        "source_sha256": {str(p): sha(p) for p in sorted(consumed)},
        "full_training_call_chain": "NOT_CHECKED",
    }
    save(output / "result.json", result)



def protocol_inventory(author, output):
    """Configuration-only independent reference; no framework imports or training."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Inventory must explicitly hide GPUs")
    inventory = {}
    for arm in SOURCE_ARMS:
        target = output / arm
        target.mkdir()
        mdpf(author, target, arm, configuration_only=True)
        config = json.loads((target / "resolved-configuration.json").read_text())
        stages = []
        for item in config["experiments"]:
            if not item["name"].endswith("_000"):
                continue
            params = item["parameters"]
            stages.append({
                "name": item["name"], "type": params.get("experiement_type"),
                "do_run": params.get("do_run", True),
                "model_params": params.get("model_params"),
                "training_params": params.get("training_params", {}),
                "pre_trained_models": params.get("pre_trained_models"),
                "dataset": params.get("resolved_dataset_parameters"),
                "evaluation_params": params.get("evaluation_params")})
        inventory[arm] = {
            "configuration": str(target / "resolved-configuration.json"),
            "architecture": config["architecture"]["dpf_full_training"],
            "first_trial_stages": stages, "pretraining_entry": config["pretraining_entry"],
            "source_sha256": config["source_sha256"]}
    save(output / "protocol-inventory.json", inventory)
    save(output / "result.json", {
        "check": "PASS", "arms": list(inventory), "trials_per_arm": 11,
        "scope": "Author configuration and checkpoint dependencies only",
        "gpu_intentionally_hidden": True, "numerical_framework_imported": False,
        "scientific_replication": "INCOMPLETE", "full_budget": "NOT_MEASURED"})


def sequence_fixture(model, parameters, src, output):
    """Execute the source sequence/loss endpoint on a small diagnostic fixture.

    The source trainer constructor, loaders, pretraining and optimizer epoch are
    deliberately outside this call-chain claim. No replacement method is defined.
    """
    import torch
    from types import SimpleNamespace
    from trainers.specific_trainers.full_sequence_trainer import FullSequenceTrainer
    from problems.bearings_only.bearings_only_problem import BearingsOnlyObservationTransformer
    from loss_functions.loss_functions import create_loss_function

    trainer = FullSequenceTrainer.__new__(FullSequenceTrainer)
    trainer.main_model = model
    trainer.device = "cpu"
    trainer.problem = SimpleNamespace(observation_transformer=BearingsOnlyObservationTransformer())
    trainer.number_of_particles = parameters["training_params"]["number_of_particles"]
    trainer.skip_first_step_for_loss = False
    trainer.max_sequence_length = None
    trainer.truncated_bptt_modulo = parameters["training_params"]["truncated_bptt_modulo"]
    trainer.loss_shaper = None
    trainer.loss_function = create_loss_function(parameters["training_params"]["loss_params"], model)
    torch.manual_seed(924002)
    model.train()
    t = torch.arange(9, dtype=torch.float32)
    x = (2.0 + .1 * t).repeat(2, 1)
    y = torch.tensor([.2, -.2]).unsqueeze(1).expand(2, 9)
    angle = torch.tensor([3.1, -3.1]).unsqueeze(1).expand(2, 9)
    states = torch.stack([x, y, angle], dim=-1)
    observations = torch.atan2(y, x + 5.).unsqueeze(-1)
    mask = (torch.arange(9) % 4 == 0).repeat(2, 1)
    calls = {}
    def profile(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if str(src.parent) in filename:
                key = str(Path(filename).relative_to(src.parent)) + ":" + frame.f_code.co_name
                calls[key] = calls.get(key, 0) + 1
    start = time.monotonic()
    try:
        sys.setprofile(profile)
        loss, batch = trainer.do_forward_pass(
            {"states": states, "observations": observations, "ground_truth_mask": mask}, None, 0)
        if batch != 2 or not bool(torch.isfinite(loss)):
            raise AssertionError("Invalid source loss")
        loss.backward()
    finally:
        sys.setprofile(None)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.all(torch.isfinite(g))) for g in grads):
        raise AssertionError("Missing or nonfinite source gradients")
    gradient_l1 = sum(float(g.abs().sum().item()) for g in grads)
    if gradient_l1 <= 0:
        raise AssertionError("Source fixture produced no gradient signal")
    required = {
        "src/trainers/specific_trainers/full_sequence_trainer.py:do_forward_pass": 1,
        "src/models/kde_particle_filter/kde_particle_filter.py:forward": 8,
        "src/models/kde_particle_filter/kde_particle_filter.py:resample_particles": 8,
        "src/loss_functions/loss_functions.py:compute_loss": 8,
    }
    for name, count in required.items():
        if calls.get(name) != count:
            raise AssertionError("Unexpected source consumer call count: " + name)
    if not any("kernel_density_estimator.py:log_prob" in key for key in calls):
        raise AssertionError("Mixture-density log probability was not called")
    executed = {}
    for key in calls:
        filename = key.rsplit(":", 1)[0]
        source = src.parent / filename
        if source.exists():
            executed[filename] = sha(source)
    result = {"status": "PASS", "fixture": "Synthetic wiring-only: B=2,T=9,N=25",
              "model_seed": 924001, "sampling_seed": 924002,
              "loss": float(loss.item()), "gradient_l1_descriptive_only": gradient_l1,
              "parameters_with_gradients": len(grads), "wall_seconds": time.monotonic()-start,
              "kernel_families": ["Normal", "Normal", "Von_Mises"],
              "sparse_labels_modulo": 4, "truncated_bptt_modulo": 4,
              "actual_source_calls": calls, "executed_source_sha256": executed,
              "trainer_constructor": "NOT_CHECKED", "optimizer_epoch": "NOT_CHECKED",
              "pretrained_weights": "NOT_USED: random initialization for wiring only",
              "gradient_estimator_correctness": "NOT_ESTABLISHED by finite gradients",
              "full_training_cost": "NOT_ESTIMATED from this fixture",
              "scientific_replication": "INCOMPLETE"}
    save(output / "source-sequence-wiring.json", result)
    return {key: value for key, value in result.items()
            if key not in ("actual_source_calls", "executed_source_sha256")}


def gpu(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("GPU diagnostic was launched with devices hidden")
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    query = subprocess.run(["nvidia-smi",
                            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
                            "--format=csv"], text=True, capture_output=True, timeout=20)
    (output / "nvidia-smi.log").write_text(query.stdout + query.stderr)
    if query.returncode:
        raise RuntimeError("Escalated nvidia-smi failed; inspect saved log")
    import tensorflow as tf
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("No physical TensorFlow GPU in the escalated diagnostic")
    growth = []
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
        enabled = tf.config.experimental.get_memory_growth(device)
        if enabled is not True:
            raise RuntimeError("Cannot verify GPU memory growth")
        growth.append({"name": device.name, "memory_growth": enabled})
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)

    @tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float32)], jit_compile=True)
    def kernel(x):
        return tf.linalg.matmul(x, x)

    with tf.device("/CPU:0"):
        cpu_x = tf.constant([[1., 2.], [3., 4.]])
        reference = tf.linalg.matmul(cpu_x, cpu_x)
    with tf.device("/GPU:0"):
        x = tf.constant([[1., 2.], [3., 4.]])
        start = time.monotonic()
        first = kernel(x)
        first_values = first.numpy().tolist()  # Diagnostic serialization boundary only.
        compile_and_first = time.monotonic() - start
        elapsed = []
        for _ in range(3):
            start = time.monotonic()
            value = kernel(x)
            values = value.numpy().tolist()
            elapsed.append(time.monotonic() - start)
            if values != first_values:
                raise AssertionError("Repeated deterministic GPU result changed")
    if first_values != reference.numpy().tolist() or "GPU:0" not in first.device:
        raise AssertionError("GPU/CPU diagnostic identity or device placement failed")
    traces = kernel.experimental_get_tracing_count()
    if traces != 1:
        raise AssertionError("Unexpected retracing in fixed-signature diagnostic")
    hlo = kernel.experimental_get_compiler_ir(x)(stage="hlo")
    (output / "kernel.hlo.txt").write_text(hlo)
    result = {
        "check": "PASS", "scientific_replication": "INCOMPLETE",
        "tensorflow_version": tf.__version__, "device": first.device,
        "memory_policy": {"mode": "memory_growth", "physical_devices": growth,
                          "logical_device_memory_limit": None},
        "allocator": tf.config.experimental.get_memory_info("GPU:0"),
        "jit_compile": True, "input_signature": "[2,2] float32", "trace_count": traces,
        "tf32": False, "tf32_role": "Tiny diagnostic reference exception",
        "reference": "TensorFlow CPU matrix product on exact small-integer inputs",
        "values": first_values, "compile_and_first_seconds": compile_and_first,
        "warm_seconds_descriptive_only": elapsed,
        "trust_basis": "Escalated supervisor and child process",
        "method_level_xla": "NOT_CHECKED", "full_step_cost": "NOT_MEASURED",
        "hlo_sha256": sha(output / "kernel.hlo.txt"),
    }
    save(output / "result.json", result)



def torch_gpu(output):
    """Device check for the original-author PyTorch reference, not a new method."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("Author GPU probe was launched with devices hidden")
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used",
                            "--format=csv"], capture_output=True, text=True, timeout=20)
    (output / "nvidia-smi.log").write_text(query.stdout + query.stderr)
    if query.returncode:
        raise RuntimeError("Escalated nvidia-smi failed")
    import torch
    available = torch.cuda.is_available()
    result = {"check": "PASS" if available else "INCOMPLETE",
              "interpreter": sys.executable, "torch_version": torch.__version__,
              "compiled_cuda_version": torch.version.cuda,
              "gpu_available": available, "backend_role": "Independent author reference",
              "full_training_call_chain": "NOT_CHECKED",
              "scientific_replication": "INCOMPLETE",
              "trust_basis": "Escalated supervisor and child process"}
    if available:
        value = torch.ones(2, device="cuda:0")
        total = (value + value).sum().item()
        if total != 4.0:
            raise AssertionError("PyTorch device arithmetic failed")
        result.update(device=torch.cuda.get_device_name(0), value=total,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    else:
        result["blocker"] = "No CUDA device available to this PyTorch build"
    save(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mdpf", "protocol", "gpu", "torch_gpu"))
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "mdpf":
            if not args.author_root:
                parser.error("--author-root is required for mdpf")
            mdpf(args.author_root.resolve(), args.output.resolve())
        elif args.mode == "protocol":
            if not args.author_root:
                parser.error("--author-root is required for protocol")
            protocol_inventory(args.author_root.resolve(), args.output.resolve())
        elif args.mode == "gpu":
            gpu(args.output.resolve())
        else:
            torch_gpu(args.output.resolve())
    except Exception as exc:
        save(args.output / "result.json", {"check": "FAILED", "error": repr(exc),
                                          "scientific_replication": "INCOMPLETE"})
        raise


if __name__ == "__main__":
    main()
