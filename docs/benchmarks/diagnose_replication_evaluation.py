#!/usr/bin/env python3
"""Independent author PyTorch/NumPy evaluation reference.

Supports short-trained cost probes or a completed source-trial checkpoint.
One trial's descriptive metrics do not establish quality or paper replication.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def validate_training_result(prior, mode):
    expected = "source_trial" if mode == "source_trial" else "cost"
    if prior.get("check") != "PASS" or prior.get("mode") != expected:
        raise RuntimeError("Evaluation requires a successful training result for its declared mode")
    if mode == "source_trial" and not prior.get("source_training_protocol_complete"):
        raise RuntimeError("Incomplete source protocol cannot supply the full-trial evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cost", "source_trial"), default="cost")
    args = parser.parse_args()
    output, author = args.output.resolve(), args.author_root.resolve()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("Use isolated python -I -B")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("Requires escalated GPU access")
    spec = importlib.util.spec_from_file_location("reference_training", Path(__file__).with_name("diagnose_replication_training.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    save, digest = helper.save, helper.digest
    training = args.training_artifact.resolve()
    prior = json.loads((training / "result.json").read_text())
    validate_training_result(prior, args.mode)
    data_seed = 925231 if args.mode == "source_trial" else 924131
    initialization_seed = data_seed + 1
    checkpoint = training / "full_training/models/full_dpf_model_best.pt"
    wiring = json.loads((training / "source-training-wiring.json").read_text())
    if digest(checkpoint) != wiring["checkpoint_sha256"]["full_training/models/full_dpf_model_best.pt"]:
        raise RuntimeError("Training checkpoint hash mismatch")
    original = helper.restore_yaml_dimensions(json.loads(args.configuration.read_text()))
    parameters = copy.deepcopy(next(e["parameters"] for e in original["experiments"]
        if e["name"] == "full_dpf_evaluation_fixed_bands_000"))
    (output / "nvidia-smi.txt").write_text(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"], text=True, timeout=20))
    import torch
    import numpy as np  # Independent author reference only; never a runtime candidate.
    if not torch.cuda.is_available():
        raise RuntimeError("Reference interpreter cannot use CUDA")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    chosen = next((i for i, name in enumerate(names) if "4080" in name), 0)
    device = torch.device("cuda", chosen)
    torch.cuda.set_device(chosen)
    assert torch.arange(8, device=device).sum().item() == 28
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    save(output / "environment.json", {"interpreter": sys.executable, "torch": torch.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda, "visible_devices": names,
        "selected_device": str(device), "selected_device_name": names[chosen],
        "actual_interpreter_cuda_probe": "PASS", "jit_compile": False,
        "role": "Independent author reference; eager PyTorch, not BayesFilter GPU/XLA",
        "allocator": "PyTorch caching allocator; TensorFlow not imported"})
    src = helper.source_copy(author, output, torch)
    os.chdir(output)
    sys.path.insert(0, str(author / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
    sys.path.insert(0, str(src))
    from models.model_creator import create_model
    from models.kde_particle_filter.kde_particle_filter import KDEParticleFilter
    from evaluations import evaluations_runner
    from evaluations.full_sequence_evaluations.full_sequence_evaluation_base import FullSequenceEvaluationBase, NonSweepEvaluationMetric
    from datasets.bearings_only_dataset import BearingsOnlyDataset
    from matplotlib.figure import Figure
    data = copy.deepcopy(parameters["resolved_dataset_parameters"]["dataset_params"])
    assert data["subsequence_length"] == 150 and data["dataset_sizes"]["evaluation"] == 5000
    data["dataset_saves"] = {"evaluation": str(output / "diagnostic-evaluation-data.pt")}
    for key in ("dataset_params_file", "resolved_dataset_parameters", "common_parameters_file"):
        parameters.pop(key, None)
    parameters["dataset_params"] = data
    parameters["save_dir"] = str(output / "evaluation")
    parameters["device"] = str(device)
    parameters["pre_trained_models"] = {"dpf_model": str(checkpoint)}
    # The source ignores configured random_seed and actually seeds from default argument 0.
    parameters["evaluation_params"]["random_seed"] = 0
    qp = parameters["evaluation_params"]["quantitative_evaluation_params"]
    assert qp["batch_size"] == 1024 and qp["save_raw_sequence_outputs"]
    assert parameters["evaluation_params"]["number_of_particles"] == 25
    save(output / "contract.json", {"question": ("Untouched T=150 evaluation of completed NLL source trial" if args.mode == "source_trial"
                      else "Source T=150 evaluation execution and cost"),
        "mode": args.mode, "trained_objective": "KDE_NLL", "mse_refinement": False,
        "pass": "Finite metrics, five sequence calls/745 transitions, raw and metric persistence",
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "training_result_sha256": digest(training / "result.json"),
        "configuration_sha256": digest(args.configuration), "data_seed": data_seed,
        "initialization_seed": initialization_seed, "source_effective_evaluation_seed": 0,
        "not_concluded": ["learned accuracy", "replication", "performance ranking", "full budget"],
        "configuration": parameters, "source_dataloader_workers": 6})
    torch.manual_seed(initialization_seed)
    model = create_model(parameters["model_params"], original["architecture"]).to(device)
    model.load_pretrained(parameters["pre_trained_models"], device)
    model.eval()
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    np.random.seed(data_seed)
    random.seed(data_seed)
    torch.manual_seed(data_seed)
    calls, events, patches = {}, [], []
    def wrap(owner, name, label, timed=True):
        original_function = getattr(owner, name)
        def measured(*a, **kw):
            calls[label] = calls.get(label, 0) + 1
            if not timed:
                return original_function(*a, **kw)
            torch.cuda.synchronize(device)
            start = time.monotonic()
            try:
                return original_function(*a, **kw)
            finally:
                torch.cuda.synchronize(device)
                events.append({"operation": label, "seconds": time.monotonic() - start})
                save(output / "timings.json", {"events": events, "calls": calls,
                    "timing_semantics": "inclusive; nested operations must not be added"})
        setattr(owner, name, measured)
        patches.append((owner, name, original_function))
    wrap(BearingsOnlyDataset, "generate_dataset", "dataset_generation")
    wrap(FullSequenceEvaluationBase, "run_the_model", "sequence_batch")
    wrap(FullSequenceEvaluationBase, "evaluate_quantatitively", "quantitative_evaluation")
    wrap(NonSweepEvaluationMetric, "compute_and_add_metric", "metric_batch")
    wrap(KDEParticleFilter, "resample_particles", "resampling", timed=False)
    wrap(torch, "save", "torch_save")
    wrap(Figure, "savefig", "figure_save")
    Path(parameters["save_dir"]).mkdir()
    start = time.monotonic()
    try:
        evaluations_runner.setup_evaluation(parameters, model, parameters["save_dir"], device)
    finally:
        for owner, name, original_function in reversed(patches):
            setattr(owner, name, original_function)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    assert calls.get("sequence_batch") == 5 and calls.get("resampling") == 745, calls
    assert calls.get("metric_batch") == 10 and calls.get("dataset_generation") == 1, calls
    evaldir = Path(parameters["save_dir"])
    table = torch.load(evaldir / "table_results.pt", map_location="cpu", weights_only=False)
    metrics = {key: [float(x) for x in values] for key, values in table.items()}
    assert set(metrics) == {"nll", "rmse"}, metrics
    assert all(math.isfinite(x) for values in metrics.values() for x in values), metrics
    raw = evaldir / "raw_all_sequence_evaluation_run_data.pt"
    assert raw.is_file() and raw.stat().st_size > 0
    save(output / "result.json", {"check": "PASS", "mode": "source_trial_evaluation" if args.mode == "source_trial" else "evaluation_cost",
        "source_evaluation_call_chain": ("PASS_ON_COMPLETE_SOURCE_TRIAL" if args.mode == "source_trial"
                                         else "PASS_ON_SHORT_TRAINED_CHECKPOINT"),
        "trained_objective": "KDE_NLL", "mse_refinement": False,
        "checkpoint_sha256": digest(checkpoint),
        "heuristic_dominance": "NOT_EVALUATED_QUALITY_PROMOTION_BLOCKED",
        "statistically_supported_ranking": False, "wall_seconds": elapsed,
        "source_metrics_descriptive_only": metrics, "calls": calls,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "raw_output_bytes": raw.stat().st_size, "raw_output_sha256": digest(raw),
        "diagnostic_dataset_sha256": digest(output / "diagnostic-evaluation-data.pt"),
        "source_effective_evaluation_seed": 0, "data_seed": data_seed,
        "scientific_replication": "INCOMPLETE", "learning_quality": "NOT_ESTABLISHED",
        "backend_role": "Independent author PyTorch/NumPy reference; eager, no XLA"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        (output / "error.txt").write_text(traceback.format_exc())
        (output / "result.json").write_text(json.dumps({"check": "FAILED", "detail": repr(exc)}) + "\n")
        raise
