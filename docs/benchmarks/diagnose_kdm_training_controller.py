"""Independent CPU diagnostic: actual author train loop vs host controller."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import ast
import contextlib
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesfilter.replication.kdm_training_controller import SourceEpochController


def selected_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Missing pinned author classes")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--saved-pretraining", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import torch
    src = ROOT / ".localresources/code/younis-mdpf-neurips-2023/src/trainers"
    namespace = dict(torch=torch, warnings=warnings, tqdm=lambda x: x)
    selected_classes(src / "trainer_utils.py",
                     ["EarlyStopping", "CustomReduceLROnPlateau"], namespace)
    source_scheduler = namespace["CustomReduceLROnPlateau"]
    adaptations = ["omit obsolete logging-only verbose constructor argument"]
    if not hasattr(source_scheduler, "is_better"):
        source_scheduler.is_better = source_scheduler._is_better
        adaptations.append("alias inherited is_better to inherited _is_better")
    tree = ast.parse((src / "trainer.py").read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(src / "trainer.py"), "exec"), namespace)
    train = namespace["train"]
    config = dict(threshold=.001, factor=.1, patience=2, cooldown=1, min_lr=0.,
                  start_epoch=0)
    cases = [
        ("ties_and_floor", [2.] * 45, config, 3, 0),
        ("negative_and_improvements", [-2., -2., -1.9995, -1.99, -2.1, -2.1,
                                     -2.09, -2.08, -2.07] * 4, config, 4, 2),
        ("strict_stop_and_cooldown", [1. + .01*i for i in range(70)], config, 4, 2),
        ("floor_events", [1. + .01*i for i in range(50)],
         dict(config, min_lr=.0005), 4, 0),
        ("delayed_start", [3., 3., 2., 2., 2., 2., 4., 4., 4., 1., 1., 1.],
         dict(config, start_epoch=4), 8, 0),
        ("full_stage_defaults", [1. + .001*i for i in range(800)],
         dict(config, patience=20, cooldown=4), 25, 200),
        ("pretraining_defaults", [2. + .001*i for i in range(400)],
         dict(config, patience=20, cooldown=5, min_lr=1e-7), 26, 60),
    ]
    saved_histories, history_rates, input_hashes = {}, {}, {}
    if args.saved_pretraining:
        import subprocess
        pin = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
        author = src.parents[1]
        for path in (src/"trainer.py", src/"trainer_utils.py"):
            relative = path.relative_to(author)
            assert path.read_bytes() == subprocess.check_output(
                ["git", "-C", str(author), "show", pin+":"+str(relative)])
        base = ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
        cases = []
        def read_input(path):
            input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            return json.loads(path.read_text())
        for unit in ("unit01", "unit02"):
            source = base/"kdm-source-variability-20261008-01"/(unit+"-full")/"diagnostic"
            target = base/"paired-calibration-20261011/calibration-01"/unit/"training/diagnostic"
            config_file = read_input(source/"configurations.json")
            training = next(x for x in config_file["executed_stages"] if x["name"]=="weights")["configuration"]["training_params"]
            scheduler = {k:v for k,v in training["lr_scheduler_params"].items()
                         if k not in ("scheduler_type", "verbose")}
            scheduler.setdefault("start_epoch", 0)
            rates = read_input(target/"weights-starting-state.json")["rates"]
            for lane, path in (("author", source/"epochs.jsonl"), ("tf", target/"epochs.jsonl")):
                input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows = [json.loads(x) for x in path.read_text().splitlines()]
                rows = [x for x in rows if x["stage"]=="weights" and
                        (lane=="tf" or x["function"]=="do_validation_epoch")]
                assert [x["epoch"] for x in rows] == list(range(len(rows)))
                values = [x["event"]["validation_loss"] if lane=="tf" else x["loss"] for x in rows]
                label = unit+"/"+lane
                cases.append((label, values, scheduler,
                              training["early_stopping_patience"], training["early_stopping_start_offset"]))
                history_rates[label] = rates
                saved_histories[label] = dict(rows=rows, lane=lane)
                if lane=="tf":
                    saved_histories[label]["best"] = read_input(target/"weights-best.json")["next_epoch"]-1
                    for x in rows:
                        assert x["event"]["validation_loss"] == x["splits"]["validation"]["weighted_loss"]
    checks, outputs = [], []
    for label, values, scheduler_config, patience, offset in cases:
        model = torch.nn.Linear(1, 1)
        rates = history_rates.get(label, dict(neural=.0005, bandwidth=.00005))
        group_rates = {}
        for name, rate in rates.items():
            group = name.split(".")[0]
            assert group not in group_rates or group_rates[group] == rate
            group_rates[group] = rate
        opt_groups = {name:torch.optim.Adam(model.parameters(), lr=lr) for name,lr in group_rates.items()}
        opts = list(opt_groups.values())
        scheduler_groups = {name:source_scheduler(opt, "min", **scheduler_config) for name,opt in opt_groups.items()}
        schedulers = list(scheduler_groups.values())
        opts_by_parameter = {name:opt_groups[name.split(".")[0]] for name in rates}
        schedulers_by_parameter = {name:scheduler_groups[name.split(".")[0]] for name in rates}
        early = namespace["EarlyStopping"](patience=patience, start_offset=offset)
        rows, saved = [], []
        def capture(train_loss, val_loss, event, epoch):
            rows.append(dict(epoch=epoch, validation_loss=val_loss,
                             scheduler_event=bool(event),
                             learning_rates={k: o.param_groups[0]["lr"] for k, o in opts_by_parameter.items()},
                             schedulers={k: {field: (None if field == "best" and math.isinf(s.best) else getattr(s, field))
                                for field in ("last_epoch", "best", "num_bad_epochs", "cooldown_counter")}
                                for k, s in schedulers_by_parameter.items()}))
        original_call = type(early).__call__
        class CapturingEarly(type(early)):
            def __call__(self, value):
                original_call(self, value)
                rows[-1]["early"] = {k: v for k, v in self.__dict__.items()}
        early.__class__ = CapturingEarly
        fake = SimpleNamespace(models={"model": (model,)}, model_save_dir="diagnostic",
            epochs=len(values), do_training_epoch=lambda epoch: 0.,
            do_validation_epoch=lambda epoch: values[epoch], lr_schedulers=schedulers,
            all_data_plotters={"model_l2_norm_plotter":
                SimpleNamespace(add_value=lambda x: None, plot_and_save=lambda: None)},
            add_and_plot_losses=capture, early_stopping=early, save_intermediate_models=True)
        def capture_save(state, name):
            saved.append(dict(epoch=len(rows)-1, name=name))
        with patch.object(torch, "save", capture_save), contextlib.redirect_stdout(io.StringIO()):
            train(fake)
        saved_best_epochs = {s["epoch"] for s in saved if s["name"].endswith("_best.pt")}
        port = SourceEpochController(rates, scheduler_config, patience=patience, start_offset=offset)
        best_epoch = None
        for row in rows:
            if row["epoch"] in saved_best_epochs:
                best_epoch = row["epoch"]
            row.update(stop=row["early"]["early_stop"],
                       save_best=row["epoch"] in saved_best_epochs,
                       best_checkpoint_epoch=best_epoch)
            event = port.observe(row["validation_loss"])
            if label in saved_histories and saved_histories[label]["lane"]=="tf":
                checks.append(dict(name=f"{label}/epoch{row['epoch']}/saved_event",
                    passed=event == saved_histories[label]["rows"][row["epoch"]]["event"]))
            state = port.state_dict()
            port_row = {k: event[k] for k in ("epoch", "validation_loss", "scheduler_event",
                                             "learning_rates", "stop", "save_best", "best_checkpoint_epoch")}
            port_row["early"] = state["early"]
            port_row["schedulers"] = {k: {field: v[field] for field in (
                "last_epoch", "best", "num_bad_epochs", "cooldown_counter")}
                for k, v in state["schedulers"].items()}
            checks.append(dict(name=f"{label}/epoch{row['epoch']}", passed=row == port_row))
            if row != port_row:
                outputs.append(dict(case=label, source=row, port=port_row))
            # Exercise actual JSON resume after every epoch.
            port = SourceEpochController.from_state_dict(json.loads(json.dumps(state, allow_nan=False)))
        checks.append(dict(name=label+"/untrained_checkpoint",
                           passed=saved[0]["name"].endswith("_untrained.pt") and saved[0]["epoch"] == -1))
        checks.append(dict(name=label+"/stopped_epoch_not_saved",
                           passed=not rows[-1]["stop"] or
                           not any(s["epoch"] == rows[-1]["epoch"] for s in saved)))
        if label in saved_histories:
            history = saved_histories[label]
            checks.append(dict(name=label+"/complete_saved_history",
                passed=len(rows)==len(history["rows"]) and rows[-1]["stop"]))
            if history["lane"]=="tf":
                checks.append(dict(name=label+"/saved_best_checkpoint",
                    passed=best_epoch == history["best"]))
        outputs.append(dict(case=label, epochs=len(rows), final=rows[-1],
                            best_checkpoint_epochs=sorted(saved_best_epochs),
                            scheduler_event_epochs=[r["epoch"] for r in rows if r["scheduler_event"]]))
    invalid = SourceEpochController({"x":.1}, config, patience=2, start_offset=0)
    before = invalid.state_dict()
    try:
        invalid.observe(float("nan"))
    except ValueError:
        rejected = True
    else:
        rejected = False
    checks.append(dict(name="nonfinite_rejected_without_state_mutation",
                       passed=rejected and invalid.state_dict() == before))
    result = dict(schema="kdm_training_controller_diagnostic_v1",
        status="PASS" if all(c["passed"] for c in checks) else "FAIL",
        checks=checks, cases=outputs, input_hashes=input_hashes, saved_pretraining=args.saved_pretraining, torch_version=torch.__version__,
        device="CPU; CUDA_VISIBLE_DEVICES=-1", source_extraction="unchanged AST bodies: EarlyStopping, CustomReduceLROnPlateau, Trainer.train",
        source_runtime_adaptations=adaptations,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (src/"trainer.py", src/"trainer_utils.py", Path(__file__),
                          ROOT/"bayesfilter/replication/kdm_training_controller.py")},
        wall_seconds=time.monotonic()-started, replication="INCOMPLETE",
        scope="CONTROLLER_DIAGNOSTIC_NOT_NEURAL_TRAINING")
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","wall_seconds","scope")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
