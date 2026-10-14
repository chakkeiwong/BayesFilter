"""Independent CPU metadata diagnostic; no framework or numerical execution."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
BASE = Path("src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs")
PLAN = "docs/plans/controlled-smc-kdm-checkpoint-wiring-20261006.md"
RESULT = "docs/plans/controlled-smc-kdm-checkpoint-wiring-result-20261006.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes, anchors = {}, {}
    def source_bytes(relative):
        relative = Path(relative)
        actual = (SOURCE / relative).read_bytes()
        expected = subprocess.check_output(
            ["git", "-C", str(SOURCE), "show", PIN + ":" + relative.as_posix()])
        if actual != expected:
            raise ValueError("Source differs from pinned blob: " + str(relative))
        hashes[relative.as_posix()] = hashlib.sha256(actual).hexdigest()
        return actual

    def source_yaml(relative):
        return yaml.safe_load(source_bytes(relative))

    def extract(relative, names, namespace, class_name=None):
        tree = ast.parse(source_bytes(relative), filename=str(relative))
        candidates = tree.body
        if class_name:
            candidates = next(n.body for n in tree.body
                              if isinstance(n, ast.ClassDef) and n.name == class_name)
        for name in names:
            node = next(n for n in candidates
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            assert not node.decorator_list
            unit = ast.Module(body=[node], type_ignores=[])
            exec(compile(unit, str(relative), "exec"), namespace)
            anchors[str(relative) + "::" + name] = [node.lineno, node.end_lineno]

    namespace = {"copy": copy, "os": os, "yaml": yaml}
    generator = BASE / "scripts/generate_multiple_runs_config_files.py"
    extract(generator, ["dict_replace_item", "generate_tasks_one_large_config"], namespace)
    extract(Path("src/main.py"), ["recursive_update", "load_common_parameters_file",
                                  "run_experiment"], namespace)
    calls, loaded_state, evaluations = [], [], []
    def record_load(path, map_location):
        calls.append({"path": path, "map_location": map_location})
        return {}
    model_namespace = {"torch": SimpleNamespace(load=record_load)}
    extract(Path("src/models/kde_particle_filter/kde_particle_filter.py"),
            ["load_pretrained"], model_namespace, "KDEParticleFilter")

    class FakeModel:
        load_pretrained = model_namespace["load_pretrained"]
        def __init__(self, architecture):
            self.decouple_weights_for_resampling = architecture.get(
                "decouple_weights_for_resampling", False)
            self.decouple_bandwidths_for_resampling = architecture.get(
                "decouple_bandwidths_for_resampling", False)
            self.do_convert_bandwidth_model_to_bandwidths_for_resampling = architecture.get(
                "do_convert_bandwidth_model_to_bandwidths_for_resampling", True)
            self.resampling_weight_model = None
            self.resampling_weighted_bandwidth_predictor = None
        def load_state_dict(self, state, strict=True):
            loaded_state.append({"empty_recording_state": state == {}, "strict": strict})
        def to(self, device):
            assert device == "cpu_metadata_only"
            return self

    namespace["create_model"] = lambda params, architecture: FakeModel(architecture[params["main_model"]])
    namespace["setup_evaluation"] = lambda experiment, model, save_dir, device: evaluations.append(
        copy.deepcopy(experiment["evaluation_params"]["quantitative_evaluation_params"]))
    def forbidden(*args, **kwargs):
        raise RuntimeError("Training/data generation is outside this metadata audit")
    namespace["setup_training"] = forbidden
    namespace["setup_data_generator"] = forbidden
    rows, active_rows = [], []
    for lane, config_name in (
        ("experiment0002_importance", "our_experiments.yaml"),
        ("experiment0003_importance_init", "our_experiments_exp3_init.yaml"),
    ):
        config = source_yaml(BASE / lane / "config.yaml")
        assert not config.get("experiments_local_override")
        base_tasks = source_yaml(BASE / "configs" / config_name)["experiments"]
        generated = namespace["generate_tasks_one_large_config"](base_tasks, 11)
        assert len(generated) == 55
        imported = Path(config["experiments_import"][0])
        assert imported.name == config_name
        root_save = config["save_dir_root"]
        assert root_save.endswith("/" + lane + "/saves/")
        for task in generated:
            name, raw = next(iter(task.items()))
            common = Path("src") / raw["common_parameters_file"]
            source_bytes(common)
            experiment = namespace["load_common_parameters_file"](
                str(SOURCE / common), copy.deepcopy(raw))
            pre = copy.deepcopy(experiment.get("pre_trained_models_local_override",
                                                experiment.get("pre_trained_models", {})))
            resolved = {k: v.replace("<root_save_dir>", root_save)
                        for k, v in pre.items()}
            trial = int(name.rsplit("_", 1)[1])
            savedir = experiment["save_dir"].replace("<root_save_dir>", root_save)
            for path in [savedir] + list(resolved.values()):
                assert "<" not in path and ">" not in path, path
            row = {"lane": lane, "task": name, "trial": trial, "do_run": experiment["do_run"],
                   "type": experiment["experiement_type"], "save_dir": savedir,
                   "checkpoint_inputs": resolved,
                   "checkpoint_exists": {k: (SOURCE / "src" / v).exists()
                                         for k, v in resolved.items()}}
            rows.append(row)
            if not experiment["do_run"]:
                continue
            assert name.startswith("full_dpf_evaluation_fixed_bands_mse_")
            assert experiment["experiement_type"] == "evaluation"
            assert list(resolved) == ["dpf_model"]
            run_output = output / lane / name
            run_output.mkdir(parents=True)
            before = len(calls), len(loaded_state), len(evaluations)
            namespace["run_experiment"](name, experiment, config["models"], str(run_output),
                                        "cpu_metadata_only", root_save, "")
            assert (len(calls), len(loaded_state), len(evaluations)) == tuple(n + 1 for n in before)
            assert calls[-1]["path"] == resolved["dpf_model"]
            normalized = Path(calls[-1]["path"]).as_posix()
            expected_path = (Path(root_save) / ("run_%03d" % trial) /
                             "full_dpf_fixed_bands/models/full_dpf_model_best.pt").as_posix()
            assert normalized == expected_path
            metrics = evaluations[-1]["metrics"]
            assert set(metrics) == {"nll", "rmse"}
            row.update(actual_torch_load=calls[-1], metrics=metrics,
                       checkpoint_training_stage="NLL", mse_trained_checkpoint_loaded=False)
            active_rows.append(row)
    assert len(active_rows) == 22
    summary = dict(status="PASS", scope="AUTHOR_CHECKPOINT_PATH_WIRING_ONLY",
                   replication="INCOMPLETE", scientific_verdict="MSE_LABELLED_EVALUATIONS_LOAD_NLL_CHECKPOINT",
                   active_evaluations=len(active_rows), task_count=len(rows),
                   actual_loader_calls=len(calls), nll_stage_loads=len(active_rows),
                   mse_stage_loads=0,
                   locally_present_active_checkpoints=sum(
                       any(row["checkpoint_exists"].values()) for row in active_rows),
                   compute_device="CPU_METADATA_ONLY", cuda_visible_devices="-1",
                   frameworks_imported=False, numerical_loading_or_training_checked=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "tasks.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "source.json").write_text(json.dumps(
        {"commit": PIN, "sha256": hashes, "callable_line_ranges": anchors}, indent=2) + "\n")
    manifest = dict(git_commit=subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        command=[sys.executable]+sys.argv, environment=sys.executable,
        cpu_gpu="CPU metadata only; CUDA intentionally hidden", random_seeds="N/A: deterministic metadata",
        data_version=PIN, wall_seconds=time.monotonic()-started, plan_file=PLAN, result_file=RESULT,
        output_artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.iterdir() if p.is_file()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mocked_boundaries=["model construction/state", "torch.load I/O", "setup_evaluation"],
        original_function_bodies_unmodified=True, no_numerical_claim=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
