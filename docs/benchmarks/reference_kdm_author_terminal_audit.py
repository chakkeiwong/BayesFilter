"""Independent CPU-only audit of the actual author's completed KDM training."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import ast
from collections import defaultdict
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
SOURCE = AUTHOR / "src/trainers/trainer_utils.py"

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            d.update(block)
    return d.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh audit output")
    started = time.monotonic()
    import torch
    torch.set_num_threads(1)
    manifest = read(args.run / "manifest.json")
    if manifest["mode"] != "source" or manifest["epoch_mode"] != "full":
        raise ValueError("Expected the complete-protocol author run")
    diagnostic = args.run / "diagnostic"
    result = read(diagnostic / "result.json")
    reference = read(manifest["reference"])
    errors, checks = [], 0
    def check(condition, name):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(name)
    check(manifest["status"] == "COMPLETED" and manifest["returncode"] == 0, "terminal_success")
    check(sha(diagnostic/"result.json") == manifest["result_sha256"], "terminal_result_hash")
    check(result["status"] == "PASS" and result["source_training_protocol_complete"], "all_stages_complete")
    check(result["gpu_intentionally_hidden"] and result["threads"] == 1, "cpu_reference")
    check(sha(manifest["reference"]) == manifest["reference_sha256"], "baseline_hash")
    for path,digest in manifest["hashes"].items():
        check(sha(ROOT/path) == digest, "helper_hash/"+path)
    expected = subprocess.check_output(["git","-C",str(AUTHOR),"show",
                                        result["source_pin"]+":src/trainers/trainer_utils.py"])
    check(SOURCE.read_bytes() == expected, "pinned_controller_source")
    tree = ast.parse(expected)
    classes = [node for node in tree.body if isinstance(node,ast.ClassDef)
               and node.name in ("EarlyStopping","CustomReduceLROnPlateau")]
    check(len(classes)==2, "actual_controller_classes")
    namespace = {"torch":torch}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(SOURCE),"exec"),namespace)
    scheduler_class = namespace["CustomReduceLROnPlateau"]
    compatibility = ["omit_removed_verbose_argument"]
    if not hasattr(scheduler_class,"is_better"):
        scheduler_class.is_better = torch.optim.lr_scheduler.ReduceLROnPlateau._is_better
        compatibility.append("is_better_to__is_better_author_worker_compatibility")
    for split,meta in result["datasets"].items():
        check(sha(meta["path"]) == meta["sha256"] == reference["datasets"][split]["sha256"], split+"/data_hash")
    stages = result["stages"]
    check([s["stage"] for s in stages] == ["weights","mdpf","am_dpf"], "stage_order")
    controllers = defaultdict(list)
    for line in (diagnostic/"controller.jsonl").read_text().splitlines():
        row = json.loads(line)
        controllers[row["stage"]].append(row)
    aggregates, current, rows = {}, None, []
    def flush():
        if current is None:
            return
        name,epoch,split = current
        total,width,partial = (5000,64,8) if split=="training" else (1000,256,232)
        ids = [i for row in rows for i in row["row_ids"]]
        label = str(current)
        check(sorted(ids)==list(range(total)),label+"/coverage")
        check(len(rows)==(79 if split=="training" else 4),label+"/batch_count")
        check(len(rows[-1]["row_ids"])==partial and all(len(row["row_ids"])==width for row in rows[:-1]),
              label+"/partial_batches")
        check(all(row["effective_batch"]==len(row["row_ids"])*(5 if name=="weights" else 1) for row in rows),
              label+"/loss_weighting")
        aggregates[current] = math.fsum(row["loss"]*row["effective_batch"] for row in rows)/sum(row["effective_batch"] for row in rows)
    with (diagnostic/"batches.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["stage"],row["epoch"],row["split"]
            if key != current:
                flush()
                current,rows=key,[]
            check(math.isfinite(row["loss"]),str(key)+"/finite_batch_loss")
            rows.append(row)
    flush()
    summaries=[]
    for stage in stages:
        name=stage["stage"]; count=stage["completed_epochs"]
        cfg=result["source_training_configs"][name]
        fixture_meta=result["fixtures"][name];fixture=read(fixture_meta["path"])
        check(sha(fixture_meta["path"])==fixture_meta["sha256"]==reference["fixtures"][name]["sha256"],name+"/constructor_hash")
        check(stage["full_protocol"] and stage["source_cap"]==cfg["epochs"],name+"/source_cap")
        check(len(stage["epochs"])==2*count and len(controllers[name])==count,name+"/epoch_count")
        losses={}
        for split,function in [("training","do_training_epoch"),("validation","do_validation_epoch")]:
            records=[r for r in stage["epochs"] if r["function"]==function]
            check([r["epoch"] for r in records]==list(range(count)),name+"/"+split+"/epoch_order")
            for row in records:
                computed=aggregates[name,row["epoch"],split]
                check(math.isclose(computed,row["loss"],rel_tol=1e-12,abs_tol=1e-12),name+"/"+split+"/recomputed_loss")
                if split=="validation":
                    losses[row["epoch"]]=row["loss"]
        for path,digest in stage["checkpoints"].items():
            check(sha(path)==digest,name+"/best_checkpoint_hash/"+Path(path).name)
            state=torch.load(path,map_location="cpu",weights_only=True)
            check(all(bool(torch.isfinite(t).all()) for t in state.values()),name+"/finite_best/"+Path(path).name)
        for role,meta in stage["input_checkpoints"].items():
            check(sha(meta["path"])==meta["sha256"],name+"/input_checkpoint_hash/"+role)
            prior=stages[0] if name=="mdpf" else stages[1]
            check(prior["checkpoints"].get(meta["path"])==meta["sha256"],name+"/preceding_best/"+role)
        latest=torch.load(stage["latest_checkpoint"],map_location="cpu",weights_only=False)
        opts=latest["optimizers"]; nopts=len(opts)
        check(stage["optimizer_steps"]==79*count*nopts and stage["source_skipped_gradients"]==0,name+"/optimizer_calls")
        check(latest["next_epoch"]==count,name+"/latest_epoch")
        check(all(bool(torch.isfinite(t).all()) for t in latest["model"].values()),name+"/finite_latest")
        for optimizer in opts:
            check(bool(optimizer["state"]),name+"/nonempty_adam_state")
            for state in optimizer["state"].values():
                check(int(state["step"])==79*count,name+"/adam_step")
                check(bool(torch.isfinite(state["exp_avg"]).all()) and bool(torch.isfinite(state["exp_avg_sq"]).all()),
                      name+"/finite_adam")
        rates=controllers[name][0]["learning_rates"]
        check(len(rates)==nopts and set(rates)==set(fixture["learning_rates"].values()),name+"/initial_rates")
        optimizers=[torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))],lr=lr) for lr in rates]
        scheduler_args=dict(cfg["lr_scheduler_params"]);scheduler_args.pop("scheduler_type");scheduler_args.pop("verbose",None)
        scheduler_args.setdefault("start_epoch",0)
        schedulers=[scheduler_class(opt,**scheduler_args) for opt in optimizers]
        early=namespace["EarlyStopping"](patience=cfg["early_stopping_patience"],start_offset=cfg["early_stopping_start_offset"])
        best_loss,best_epoch=math.inf,-1
        for row in controllers[name]:
            epoch=row["epoch"];loss=losses[epoch];label=name+"/epoch"+str(epoch)
            check(row["validation_loss"]==loss,label+"/controller_input")
            improved=loss<best_loss
            best_loss=min(best_loss,loss)
            changed=[scheduler.step(loss) for scheduler in schedulers]
            if any(changed):
                early.lr_changed()
            with contextlib.redirect_stdout(io.StringIO()):
                early(loss)
            if improved and not early.early_stop:
                best_epoch=epoch
            check(row["early_stopping"]==vars(early),label+"/actual_early_stopping")
            check(row["learning_rates"]==[opt.param_groups[0]["lr"] for opt in optimizers],label+"/actual_schedule")
            check(row["optimizer_steps"]==79*(epoch+1)*nopts,label+"/optimizer_count")
            check(not early.early_stop or epoch==count-1,label+"/no_post_stop_epoch")
        check(stage["termination"]==("SOURCE_EARLY_STOP" if early.early_stop else "SOURCE_CAP"),name+"/termination")
        check(early.early_stop or count==cfg["epochs"],name+"/no_truncation")
        check(latest["early_stopping"]==vars(early),name+"/saved_early_state")
        for actual,scheduler in zip(latest["schedulers"],schedulers):
            replay=scheduler.state_dict()
            for key in ["best","num_bad_epochs","cooldown_counter","last_epoch","_last_lr"]:
                check(actual[key]==replay[key],name+"/saved_scheduler/"+key)
        trainable=set(fixture["trainable_names"])
        # A-MDPF receives the preceding MDPF best model before freezing;
        # its fresh constructor is not the post-transfer frozen baseline.
        transferred = (torch.load(stage["input_checkpoints"]["dpf_model"]["path"],
                                  map_location="cpu",weights_only=True)
                       if name=="am_dpf" else None)
        for key,value in latest["model"].items():
            if key in fixture["parameters"] and key not in trainable:
                expected = (transferred[key.removeprefix("resampling_")] if transferred is not None
                            else torch.tensor(fixture["parameters"][key],dtype=value.dtype))
                check(torch.equal(value,expected),name+"/frozen/"+key)
        summaries.append(dict(stage=name,epochs=count,best_epoch=best_epoch,best_validation_loss=best_loss,
            termination=stage["termination"],optimizer_count=nopts,updates=79*count,wall_seconds=stage["wall_seconds"]))
    payload=dict(schema="kdm_author_terminal_audit_v1",verdict="PASS" if not errors else "FAIL",
        checks=checks,failures=errors,stages=summaries,command=[sys.executable]+sys.argv,
        run=str(args.run),run_manifest_sha256=sha(args.run/"manifest.json"),
        audit_sha256=sha(__file__),controller_source_sha256=sha(SOURCE),
        controller_api_compatibility=compatibility,torch_version=torch.__version__,
        cpu_only=True,gpu_intentionally_hidden=True,wall_seconds=time.monotonic()-started,
        replication="INCOMPLETE",not_concluded="Numerical compatibility, trained-model noninferiority or full replication")
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:payload[k] for k in ["verdict","checks","failures","stages","wall_seconds"]}))
    return 1 if errors else 0

if __name__=="__main__":
    lib=str(Path(sys.executable).resolve().parent.parent/"lib")
    if os.environ.get("LD_LIBRARY_PATH")!=lib:
        os.execve(sys.executable,[sys.executable,"-B",str(Path(__file__).resolve())]+sys.argv[1:],
                  dict(os.environ,LD_LIBRARY_PATH=lib))
    raise SystemExit(main())
