"""CPU-only independent author diagnostic; never a BayesFilter runtime route."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import MethodType, SimpleNamespace

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
CAL = BASE / "paired-calibration-20261011/calibration-01"
SEED = 2026101300


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main(out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("author", ROOT / "docs/benchmarks/reference_kdm_neural_exact_replay.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch = ref.torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    trainer_path = ref.AUTHOR / "src/trainers/trainer.py"
    full_path = ref.AUTHOR / "src/trainers/specific_trainers/full_sequence_trainer.py"
    forward = ref.source_member(full_path, "FullSequenceTrainer", "do_forward_pass")
    convert = ref.source_member(full_path, "FullSequenceTrainer", "convert_to_model_dtype")
    train_epoch = ref.source_member(trainer_path, "Trainer", "do_training_epoch")
    validation_epoch = ref.source_member(trainer_path, "Trainer", "do_validation_epoch")
    # Suppress progress display only; the extracted numerical source bodies are unchanged.
    for function in (train_epoch, validation_epoch):
        function.__globals__["tqdm"] = lambda iterable, **kw: iterable
    transformer = ref.source_member(ref.AUTHOR / "src/problems/bearings_only/bearings_only_problem.py", "BearingsOnlyObservationTransformer")
    architecture = {}
    for p in (ref.EXPERIMENT / "configs/models.yaml", ref.EXPERIMENT / "experiment0002_importance/config.yaml"):
        architecture.update(ref.yaml.safe_load(ref.verify(p))["models"])

    checks, rows, training_rows, inputs = [], [], [], {}
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def register(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    def progress(stage, **kw):
        record = dict(stage=stage, wall_seconds=time.monotonic()-started, **kw)
        write(out / "progress.json", record)
        print(json.dumps(record), flush=True)
    def rng_hash():
        return hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()

    class Plotter:
        def __init__(self):
            self.values = []
        def add_value(self, value):
            check("finite plotted loss/gradient", math.isfinite(value))
            self.values.append(value)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path, count):
            register(path)
            data = read(path)
            check(str(path)+" rows", data["rows"] == count and data["row_ids"] == list(range(count)))
            self.data = {k: torch.tensor(v, dtype=torch.bool if k == "ground_truth_mask" else torch.float32) for k, v in data["data"].items()}
            check(str(path)+" shapes", all(v.shape[:2] == (count, 17) for v in self.data.values()))
            self.count = count
        def __len__(self):
            return self.count
        def __getitem__(self, index):
            return {**{k: v[index] for k, v in self.data.items()}, "dataset_index": index}

    def build(cfg, state, directory):
        model = ref.KDEParticleFilter(cfg["configuration"]["model_params"], architecture)
        params = dict(model.named_parameters())
        check("parameter names", set(params) == set(state["parameters"]))
        with torch.no_grad():
            for name, value in params.items():
                expected = torch.tensor(state["parameters"][name])
                value.copy_(expected)
                check("restored parameter " + name, torch.equal(value, expected))
        trainer = SimpleNamespace(main_model=model, device=torch.device("cpu"), problem=SimpleNamespace(observation_transformer=transformer()),
            number_of_particles=25, skip_first_step_for_loss=False, max_sequence_length=None,
            truncated_bptt_modulo=4, loss_shaper=None, loss_function=ref.KDE_NLL(dict(kde_params=architecture["dpf_full_training"]["kde"]), model),
            models={}, models_to_optimize=[], optimizers=[], lr_schedulers=[], regularization_params=None,
            gradient_clip_value=100., gradient_clip_params=None, gradient_value_for_skip=None,
            accumulate_gradients_counter=1, total_number_of_skipped_gradients=0, save_dir=str(directory),
            all_data_plotters={"training_loss_plotter": Plotter(), "validation_loss_plotter": Plotter()})
        trainer.convert_to_model_dtype = MethodType(convert, trainer)
        trainer.do_forward_pass = MethodType(forward, trainer)
        trainer.do_freeze_rnn_batchnorm_layers = lambda: False
        def add_model(module, name, do_grad_clipping=False):
            trainer.models[name] = (module, do_grad_clipping)
            trainer.all_data_plotters[name+"_gradient_plotter"] = Plotter()
        trainer.add_model = add_model
        trainer.add_model_for_training = trainer.models_to_optimize.append
        def add_optimizer(modules, names, lr):
            check("one native optimizer module", len(modules) == len(names) == 1)
            module = modules[0]
            if lr == "Freeze":
                for value in module.parameters():
                    value.requires_grad_(False)
                module.eval()
            else:
                trainer.optimizers.append(torch.optim.Adam(module.parameters(), lr=lr, weight_decay=0.))
                trainer.models_to_optimize.extend(names)
        trainer.add_optimizer_and_lr_scheduler = add_optimizer
        model.add_models(trainer, "full")
        model.create_and_add_optimizers(cfg["configuration"]["training_params"], trainer, "full")
        check("trainable names", [n for n, v in params.items() if v.requires_grad] == state["trainable_names"])
        names = {id(v): n for n, v in params.items()}
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                rates = {state["rates"][names[id(v)]] for v in group["params"]}
                check("single saved rate", len(rates) == 1)
                group["lr"] = rates.pop()
                for value in group["params"]:
                    name = names[id(value)]
                    optimizer.state[value] = dict(step=torch.tensor(state["step"], dtype=torch.float32),
                        exp_avg=torch.tensor(state["first_moments"][name]), exp_avg_sq=torch.tensor(state["second_moments"][name]))
        return model, trainer

    def snapshot(model, trainer, template):
        state = copy.deepcopy(template)
        state["parameters"] = ref.host(dict(model.named_parameters()))
        moments = {p: s for opt in trainer.optimizers for p, s in opt.state.items()}
        names = dict(model.named_parameters())
        state["first_moments"] = {n: ref.host(moments[p]["exp_avg"]) for n, p in names.items() if p.requires_grad}
        state["second_moments"] = {n: ref.host(moments[p]["exp_avg_sq"]) for n, p in names.items() if p.requires_grad}
        steps = {int(s["step"]) for s in moments.values()}
        check("uniform Adam steps", len(steps) == 1)
        state["step"] = next(iter(steps))
        for values in (state["parameters"], state["first_moments"], state["second_moments"]):
            check("finite checkpoint arrays", all(bool(torch.isfinite(torch.tensor(v)).all()) for v in values.values()))
        return state

    def observe(trainer, split, destination):
        def call(data, dataset, epoch):
            loss, count = forward(trainer, data, dataset, epoch)
            check("finite "+split+" loss", loss is not None and bool(torch.isfinite(loss)))
            destination.append(dict(split=split, epoch=epoch, rows=data["dataset_index"].tolist(), count=count, loss=float(loss.detach())))
            return loss, count
        trainer.do_forward_pass = call

    for unit in (1, 2):
        unitname = "unit%02d" % unit
        source = BASE / "kdm-source-variability-20261008-01" / (unitname+"-full/diagnostic")
        cfgpath = source / "configurations.json"
        register(cfgpath)
        cfg = next(x for x in read(cfgpath)["executed_stages"] if x["name"] == "mdpf")
        check("architecture identity", json.loads(json.dumps(architecture)) == cfg["architecture"])
        expected = dict(number_of_particles=25, training_batch_size=64, validation_batch_size=256,
            truncated_bptt_modulo=4, gradient_clip_value=100., optimizer_type="Adam", initializer_model_learning_rate="Freeze")
        check("native training settings", all(cfg["configuration"]["training_params"][k] == v for k, v in expected.items()))
        checkpointpath = CAL / unitname / "training/diagnostic/mdpf-best.json"
        register(checkpointpath)
        template = read(checkpointpath)["model"]
        directory = out / unitname
        directory.mkdir()
        model, trainer = build(cfg, template, directory)
        check("full saved optimizer restoration", snapshot(model, trainer, template) == template)
        validation_data = Dataset(source / "validation.json", 1000)
        trainer.validation_dataset = validation_data
        trainer.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=256, shuffle=False,
            num_workers=1, pin_memory=True, persistent_workers=True)
        # Initialize the persistent worker before resetting RNG for controlled calls.
        check("validation loader coverage", [len(b["dataset_index"]) for b in trainer.validation_loader] == [256, 256, 256, 232])
        candidates = {"tf_best": copy.deepcopy(model.state_dict())}
        authorpath = source / "mdpf/models/full_dpf_model_best.pt"
        register(authorpath)
        author = torch.load(authorpath, map_location="cpu", weights_only=True)
        model.load_state_dict(author, strict=True)
        candidates["author_best"] = copy.deepcopy(model.state_dict())
        model.load_state_dict(candidates["tf_best"], strict=True)
        if unit == 1:
            training_data = Dataset(source / "training.json", 5000)
            trainer.training_dataset = training_data
            trainer.train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True,
                num_workers=1, pin_memory=True, persistent_workers=True)
            write(directory / "window0.json", template)
            for epoch in (1, 2):
                torch.manual_seed(SEED+epoch)
                write(directory / ("training-rng%d.json" % epoch), torch.get_rng_state().tolist())
                batches = []
                observe(trainer, "training", batches)
                progress("training", unit=unit, epoch=epoch)
                value = train_epoch(trainer, epoch)
                check("native training coverage", sorted(i for b in batches for i in b["rows"]) == list(range(5000)))
                check("native training batch sizes", [b["count"] for b in batches] == [64]*78+[8])
                state = snapshot(model, trainer, template)
                check("79 actual updates per epoch", state["step"] == template["step"]+79*epoch)
                write(directory / ("window%d.json" % epoch), state)
                write(directory / ("training-batches%d.json" % epoch), batches)
                training_rows.append(dict(epoch=epoch, loss=value, step=state["step"], seed=SEED+epoch))
                candidates["window%d" % epoch] = copy.deepcopy(model.state_dict())
            trainer.train_loader._iterator._shutdown_workers()
        write(directory / "candidate-state-hashes.json", {name: hashlib.sha256(json.dumps(ref.host(state), sort_keys=True).encode()).hexdigest() for name, state in candidates.items()})

        # Call-chain control: extracted native validation epoch equals the same native
        # forward calls and weighting when started from the identical RNG state.
        model.load_state_dict(candidates["tf_best"], strict=True)
        control_batches = []
        observe(trainer, "validation-control", control_batches)
        torch.manual_seed(SEED+unit*100000+999)
        native_control = validation_epoch(trainer, 0)
        control_rng = rng_hash()
        torch.manual_seed(SEED+unit*100000+999)
        manual_sum = 0.
        with torch.no_grad():
            for batch in trainer.validation_loader:
                loss, count = forward(trainer, batch, None, 0)
                manual_sum += float(loss)*count
        check("native validation call-chain exact", native_control == manual_sum/1000.)
        check("native validation RNG exact", control_rng == rng_hash())
        write(directory / "validation-control.json", dict(native=native_control, manual=manual_sum/1000., rng_sha256=control_rng, batches=control_batches))
        for mode in ("independent", "shared_initial_rng"):
            for index, (name, state) in enumerate(candidates.items()):
                model.load_state_dict(state, strict=True)
                for repetition in range(8):
                    seed = SEED+unit*100000+(10000 if mode == "shared_initial_rng" else 0)+(index*100 if mode == "independent" else 0)+repetition
                    batches = []
                    observe(trainer, "validation", batches)
                    torch.manual_seed(seed)
                    before = rng_hash()
                    tick = time.monotonic()
                    value = validation_epoch(trainer, 0)
                    check("native validation count", [b["count"] for b in batches] == [256, 256, 256, 232])
                    check("native validation order", [i for b in batches for i in b["rows"]] == list(range(1000)))
                    check("no parameter changes in validation", all(torch.equal(v, state[k]) for k, v in model.state_dict().items()))
                    row = dict(unit=unit, mode=mode, checkpoint=name, repetition=repetition, seed=seed,
                        loss=value, rng_before=before, rng_after=rng_hash(), batches=batches, wall_seconds=time.monotonic()-tick)
                    rows.append(row)
                    with (out / "evaluations.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    progress("validation", unit=unit, mode=mode, checkpoint=name, repetition=repetition, loss=value)
        trainer.validation_loader._iterator._shutdown_workers()

    summaries, comparisons, selections = [], [], []
    for unit in (1, 2):
        for mode in ("independent", "shared_initial_rng"):
            groups = {}
            for row in rows:
                if row["unit"] == unit and row["mode"] == mode:
                    groups.setdefault(row["checkpoint"], []).append(row["loss"])
            for name, values in groups.items():
                summaries.append(dict(unit=unit, mode=mode, checkpoint=name, mean=statistics.mean(values), sd=statistics.stdev(values), n=len(values), values=values))
            delta = [a-b for a,b in zip(groups["tf_best"], groups["author_best"])]
            comparisons.append(dict(unit=unit, mode=mode, tf_minus_author_mean=statistics.mean(delta), delta_sd=statistics.stdev(delta), differences=delta))
            if unit == 1:
                names = ("tf_best", "window1", "window2")
                heldout = {name: statistics.mean(groups[name][4:]) for name in names}
                chosen = [min(names, key=lambda n: groups[n][i]) for i in range(4)]
                selections.append(dict(mode=mode, choices_first4=chosen, heldout_means_last4=heldout,
                    heldout_reference_min=min(heldout.values()), regret=[heldout[n]-min(heldout.values()) for n in chosen],
                    all8_winner_counts={n: sum(min(names, key=lambda k: groups[k][i]) == n for i in range(8)) for n in names}))
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).is_relative_to(ref.AUTHOR):
            ref.verify(Path(path))
    write(out / "inputs.json", inputs)
    write(out / "author-source-hashes.json", ref.HASHES)
    write(out / "checks.json", checks)
    result = dict(status="COMPLETED", classification="conditional_native_validation_diagnostic_only", replication="INCOMPLETE",
        cpu_only=True, gpu_intentionally_hidden=True, source_commit=ref.PIN, torch_version=torch.__version__,
        seed_base=SEED, rows=1000, particles=25, horizon=17, validation_batch_sizes=[256,256,256,232],
        native_numerical_bodies_unchanged=True, fixed_rate_window=True, historical_selection_reconstructed=False,
        shared_rng_is_not_guaranteed_common_variates=True, evaluations=len(rows), checks=len(checks),
        summaries=summaries, comparisons=comparisons, selections=selections, training=training_rows,
        inference="conditional descriptive only; eight draws are not eight training units", wall_seconds=time.monotonic()-started)
    write(out / "result.json", result)
    progress("complete", evaluations=len(rows), checks=len(checks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
