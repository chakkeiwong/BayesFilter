"""CPU-only independent diagnostic at the current MDPF training checkpoints.

Reuses the unchanged bounded update/capture function from the earlier reference
diagnostic, with its free variables supplied explicitly. No runtime imports this
module. Author code and diagnostic helper hashes are preserved in the output.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
CAL = BASE / "paired-calibration-20261011/calibration-01"
SEED, BATCH, OFFSET, UPDATES = 202610121, 64, 384, 2


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + "\n")


def main(out, *, saved_optimizer=False):
    assert sys.flags.isolated
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    helper = ROOT / "docs/benchmarks/reference_kdm_neural_exact_replay.py"
    spec = importlib.util.spec_from_file_location("current_gap_author", helper)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch = ref.torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    trainer_path = ref.AUTHOR / "src/trainers/specific_trainers/full_sequence_trainer.py"
    forward = ref.source_member(trainer_path, "FullSequenceTrainer", "do_forward_pass")
    convert = ref.source_member(trainer_path, "FullSequenceTrainer", "convert_to_model_dtype")
    transformer = ref.source_member(ref.AUTHOR / "src/problems/bearings_only/bearings_only_problem.py",
                                    "BearingsOnlyObservationTransformer")
    template = ROOT / "docs/benchmarks/reference_kdm_coupled_updates.py"
    worker = next(n for n in ast.parse(template.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "worker")
    update_node = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "update")
    def restore_diagnostic_optimizer(collector, model, dtype_name):
        # Same saved state in both libraries; this is a local recurrence test,
        # not a claim to continue the author's independently trained model.
        dtype = getattr(torch, dtype_name)
        names = {id(p): n for n, p in model.named_parameters()}
        assert set(collector.rates) == set(state["trainable_names"])
        for optimizer in collector.optimizers:
            for group in optimizer.param_groups:
                group_rates = {state["rates"][names[id(p)]] for p in group["params"]}
                assert len(group_rates) == 1
                group["lr"] = group_rates.pop()
                for p in group["params"]:
                    n = names[id(p)]
                    first = torch.tensor(state["first_moments"][n], dtype=dtype)
                    second = torch.tensor(state["second_moments"][n], dtype=dtype)
                    assert first.shape == p.shape == second.shape
                    assert first.tolist() == state["first_moments"][n]
                    assert second.tolist() == state["second_moments"][n]
                    optimizer.state[p] = dict(step=torch.tensor(state["step"], dtype=torch.float32),
                                              exp_avg=first, exp_avg_sq=second)
                    collector.rates[n] = state["rates"][n]
    if saved_optimizer:
        points = [i for i,n in enumerate(update_node.body)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Attribute)
                  and n.value.func.attr == "create_and_add_optimizers"]
        assert len(points) == 1
        update_node.body.insert(points[0]+1, ast.parse(
            "restore_diagnostic_optimizer(collector, model, dtype_name)").body[0])
        ast.fix_missing_locations(update_node)
    namespace = dict(torch=torch, copy=copy, ref=ref, transformer=transformer,
                     SimpleNamespace=SimpleNamespace, MethodType=MethodType,
                     convert=convert, forward=forward, time=time, SEED=SEED,
                     BATCH=BATCH, OFFSET=OFFSET, UPDATES=UPDATES,
                     restore_diagnostic_optimizer=restore_diagnostic_optimizer)
    exec(compile(ast.Module(body=[update_node], type_ignores=[]), str(template), "exec"), namespace)
    assert read(CAL / "manifest.json")["status"] == "COMPLETED"
    cases, controls, provenance, datasets = [], [], [], []
    for unit in ("unit01", "unit02"):
        tf_run = CAL / unit / "training/diagnostic"
        source = BASE / "kdm-source-variability-20261008-01" / (unit + "-full/diagnostic")
        audit = read(CAL / unit / "audit.json")
        assert audit["verdict"] == "PASS"
        cfg = next(x for x in read(source / "configurations.json")["executed_stages"] if x["name"] == "mdpf")
        training = cfg["configuration"]["training_params"]
        # JSON makes integer dimension keys into strings. Rebuild from pinned YAML.
        architecture = {}
        for path in (ref.EXPERIMENT / "configs/models.yaml",
                     ref.EXPERIMENT / "experiment0002_importance/config.yaml"):
            architecture.update(ref.yaml.safe_load(ref.verify(path))["models"])
        assert json.loads(json.dumps(architecture)) == cfg["architecture"]
        assert training["gradient_clip_value"] == 100.0 and training["truncated_bptt_modulo"] == 4
        data_path = source / "training.json"
        data = read(data_path)["data"]
        def make_batch(index):
            lo = OFFSET + BATCH * index
            return {k: torch.tensor(v[lo:lo+BATCH], dtype=torch.bool if k == "ground_truth_mask" else torch.float32)
                    for k, v in data.items()}
        namespace["make_batch"] = make_batch
        datasets.append(dict(path=str(data_path), sha256=sha(data_path)))
        for role in (("best",) if saved_optimizer else ("starting", "best")):
            path = tf_run / ("mdpf-starting-state.json" if role == "starting" else "mdpf-best.json")
            saved = read(path)
            state = saved if role == "starting" else saved["model"]
            assert state["schema"] == "kdm_neural_training_state_v2"
            if role == "best":
                stage = next(x for x in read(tf_run / "result.json")["completed_stages"] if x["stage"] == "mdpf")
                assert sha(path) == stage["best_checkpoint_sha256"]
            torch.manual_seed(SEED)
            model = ref.KDEParticleFilter(cfg["configuration"]["model_params"], architecture)
            params = dict(model.named_parameters())
            assert set(params) == set(state["parameters"])
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(torch.tensor(state["parameters"][name], dtype=torch.float32))
                    assert p.tolist() == state["parameters"][name]
            update = namespace["update"]
            native = update(model, training, architecture, "float32")
            assert native["trainable_names"] == state["trainable_names"]
            tapes = [u["captured"] for u in native["updates"]]
            replay = update(model, training, architecture, "float32", tapes)
            for i in range(UPDATES):
                for field in ("data", "captured", "loss", "gradient_norm", "raw_gradients",
                              "clipped_gradients", "parameters_after", "adam_after"):
                    good = native["updates"][i][field] == replay["updates"][i][field]
                    controls.append(dict(name=f"{unit}/{role}/{i}/{field}", pass_check=good))
                    assert good, controls[-1]["name"]
            double = update(model, training, architecture, "float64", tapes)
            for case in (native, double):
                case.update(fixture_id=unit + "_" + role + ("_saved_optimizer" if saved_optimizer else ""),
                            variant="mdpf")
                if saved_optimizer:
                    case["initial_optimizer_state"] = {k: state[k] for k in
                        ("step", "first_moments", "second_moments", "rates")}
                    case["checkpoint_role"] = "actual_saved_parameters_moments_step_and_rates"
                    assert case["learning_rates"] == state["rates"]
                cases.append(case)
            provenance.append(dict(unit=unit, role=role, path=str(path), sha256=sha(path),
                                   actual_saved_step=state["step"], configuration=str(source / "configurations.json"),
                                   configuration_sha256=sha(source / "configurations.json")))
            write(out / "progress.json", dict(unit=unit, role=role, cases=len(cases)))
            print(json.dumps(dict(unit=unit, role=role, cases=len(cases))), flush=True)
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).is_relative_to(ref.AUTHOR):
            ref.verify(Path(path))
    inventory = out / "checkpoint-inventory.json"
    data_inventory = out / "data-inventory.json"
    write(inventory, provenance)
    write(data_inventory, datasets)
    payload = dict(schema="kdm_current_training_gap_reference_v1", cpu_only=True,
                   gpu_intentionally_hidden=True, author_commit=ref.PIN, author_hashes=ref.HASHES,
                   torch_version=torch.__version__, checkpoint=str(inventory), checkpoint_sha256=sha(inventory),
                   dataset=str(data_inventory), dataset_sha256=sha(data_inventory), cases=cases,
                   replay_controls=controls, historical_cost={}, diagnostic_seeds=[SEED],
                   reference_components=[dict(path=str(p), sha256=sha(p)) for p in
                                         (Path(__file__), helper, template, CAL / "manifest.json")],
                   optimizer_state_role=("Actual saved moments/step/rates in both frameworks"
                       if saved_optimizer else "Fresh diagnostic Adam; not continuation of saved accumulated moments"),
                   stochastic_control_role="Common realized draws, initial tensors and data; not native RNG equivalence",
                   wall_seconds=time.monotonic()-started)
    write(out / "author.json", payload)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-optimizer", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.output.resolve(), saved_optimizer=args.saved_optimizer))
