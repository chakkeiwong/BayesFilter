"""Independent CPU diagnostic: actual author control on saved validation histories."""
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import torch
    src = ROOT / ".localresources/code/younis-mdpf-neurips-2023/src/trainers"
    import subprocess
    helper=ROOT/"docs/benchmarks/reference_kdm_neural.py"
    tree=ast.parse(helper.read_text())
    pin=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PIN" for t in n.targets))
    for path in (src/"trainer.py",src/"trainer_utils.py"):
        rel=str(path.relative_to(src.parent.parent))
        blob=subprocess.check_output(["git","-C",str(src.parent.parent),"show",pin+":"+rel])
        assert blob==path.read_bytes(),"Author source drift"
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
    base=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
    cases=[];input_hashes={}
    for unit in ("unit01","unit02"):
        source=base/"kdm-source-variability-20261008-01"/(unit+"-full")/"diagnostic"
        port=base/"paired-calibration-20261011/calibration-01"/unit/"training/diagnostic"
        config_path=source/"configurations.json"
        config=json.loads(config_path.read_text())
        input_hashes[str(config_path)]=hashlib.sha256(config_path.read_bytes()).hexdigest()
        for engine,root in (("source",source),("tf",port)):
            path=root/"epochs.jsonl";input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            history=[json.loads(line) for line in path.read_text().splitlines()]
            for spec in config["executed_stages"]:
                stage=spec["name"];training=spec["configuration"]["training_params"]
                values=[r["loss"] if engine=="source" else r["splits"]["validation"]["weighted_loss"]
                    for r in history if r["stage"]==stage and (engine=="tf" or r["function"]=="do_validation_epoch")]
                assert values, (unit,engine,stage)
                scheduler=dict(training["lr_scheduler_params"]);scheduler.pop("scheduler_type");scheduler.pop("verbose",None)
                scheduler.setdefault("start_epoch",0)
                rates={k:float(v) for k,v in training.items() if k.endswith("_learning_rate") and v!="Freeze"}
                assert rates
                cases.append((unit+"/"+engine+"/"+stage,values,scheduler,
                    training["early_stopping_patience"],training["early_stopping_start_offset"],rates))
    # Retain explicit tie/negative fixtures in addition to every observed history.
    config=dict(threshold=.001,factor=.1,patience=2,cooldown=1,min_lr=0.,start_epoch=0)
    for label,values in (("ties",[2.]*45),("negative",[-2.,-2.,-1.99,-2.1,-2.1,-2.09]*6)):
        cases.append((label,values,config,4,2,dict(neural=.0005,bandwidth=.00005)))
    checks, outputs = [], []
    for label, values, scheduler_config, patience, offset, rates in cases:
        model = torch.nn.Linear(1, 1)
        opts = [torch.optim.Adam(model.parameters(), lr=lr) for lr in rates.values()]
        schedulers = [source_scheduler(o, "min", **scheduler_config) for o in opts]
        early = namespace["EarlyStopping"](patience=patience, start_offset=offset)
        rows, saved = [], []
        def capture(train_loss, val_loss, event, epoch):
            rows.append(dict(epoch=epoch, validation_loss=val_loss,
                             scheduler_event=bool(event),
                             learning_rates={k: o.param_groups[0]["lr"] for k, o in zip(rates, opts)},
                             schedulers={k: {field: (None if field == "best" and math.isinf(s.best) else getattr(s, field))
                                for field in ("last_epoch", "best", "num_bad_epochs", "cooldown_counter")}
                                for k, s in zip(rates, schedulers)}))
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
        checks=checks, cases=outputs, torch_version=torch.__version__,
        device="CPU; CUDA_VISIBLE_DEVICES=-1", source_extraction="unchanged AST bodies: EarlyStopping, CustomReduceLROnPlateau, Trainer.train",
        source_runtime_adaptations=adaptations,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (src/"trainer.py", src/"trainer_utils.py", Path(__file__),
                          ROOT/"bayesfilter/replication/kdm_training_controller.py")},
        wall_seconds=time.monotonic()-started, replication="INCOMPLETE",
        input_hashes=input_hashes, scope="SAVED_HISTORY_CONTROLLER_DIAGNOSTIC_NOT_NEURAL_TRAINING")
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","wall_seconds","scope")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
