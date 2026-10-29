"""CPU-only artifact/controller audit of complete TF KDM development training.

Replays the pinned author's actual scheduler and EarlyStopping classes on saved
losses. No model training, GPU access, or numerical/replication promotion.
"""
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
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".localresources/code/younis-mdpf-neurips-2023/src/trainers/trainer_utils.py"

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, (int, float)):
        yield value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-start", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior audit evidence; choose a fresh output")
    started = time.monotonic()
    import torch
    torch.set_num_threads(1)
    manifest = read(args.run / "manifest.json")
    if manifest["mode"] != "tf" or manifest["epoch_mode"] != "full":
        raise ValueError("This audit requires a complete-protocol TF training run")
    reference = read(manifest["reference"])
    checks, errors = 0, []

    def check(condition, name):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(name)

    check(sha(manifest["reference"]) == manifest["reference_sha256"], "reference_hash")
    for path, digest in manifest["hashes"].items():
        check(sha(ROOT / path) == digest, "source_hash/" + path)
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("EarlyStopping", "CustomReduceLROnPlateau")]
    check(len(classes) == 2, "two_actual_source_controller_classes")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    # Match the source worker's Torch API adaptation on the extracted class only.
    compatibility = ["omit_removed_verbose_argument"]
    scheduler_class = namespace["CustomReduceLROnPlateau"]
    if not hasattr(scheduler_class, "is_better"):
        scheduler_class.is_better = torch.optim.lr_scheduler.ReduceLROnPlateau._is_better
        compatibility.append("is_better_to__is_better_author_worker_compatibility")
    diagnostic = args.run / "diagnostic"
    stages = read(diagnostic / "stages.partial.json")
    stage_names = {s["stage"] for s in stages}
    if args.isolated_start is not None:
        check(manifest.get("selected_stages") == ["mdpf"] and stage_names == {"mdpf"}, "explicit_isolated_mdpf")
        check(sha(args.isolated_start) == manifest["starting_state_sha256"], "isolated_start_hash")
    epochs = defaultdict(list)
    for line in (diagnostic / "epochs.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["stage"] in stage_names:
            epochs[row["stage"]].append(row)

    # Stream batches; retain only one epoch/split's IDs at a time.
    aggregates, keys, current, group = {}, set(), None, []

    def flush():
        if current is None:
            return
        stage, epoch, split = current
        count, width, partial = (5000, 64, 8) if split == "training" else (1000, 256, 232)
        ids = [i for row in group for i in row["row_ids"]]
        name = "/".join(map(str, current))
        check(sorted(ids) == list(range(count)), name + "/coverage")
        check(len(group) == (79 if split == "training" else 4), name + "/batches")
        check(len(group[-1]["row_ids"]) == partial, name + "/partial_batch")
        check(all(len(row["row_ids"]) == width for row in group[:-1]), name + "/full_batches")
        check(all(row["sequence_batch"] == len(row["row_ids"]) and
                  row["effective_batch"] == len(row["row_ids"]) * (5 if stage == "weights" else 1)
                  for row in group), name + "/weights")
        weights = [row["effective_batch"] for row in group]
        aggregates[current] = math.fsum(row["loss"] * weight for row, weight in zip(group, weights)) / sum(weights)

    with (diagnostic / "batches.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["stage"] not in stage_names:
                continue
            key = (row["stage"], row["epoch"], row["split"])
            if key != current:
                flush()
                current, group = key, []
            group.append(row)
            check(row["valid"] and math.isfinite(row["loss"]), str(key) + "/finite_valid_batch")
            if row["split"] == "training":
                check(row["committed"], str(key) + "/committed")
            random_key = tuple(row["seed"])
            check(random_key not in keys, str(key) + "/unique_native_key")
            keys.add(random_key)
    flush()
    summaries = []
    prior_best = None
    for stage in stages:
        name = stage["stage"]
        cfg = reference["source_training_configs"][name]
        fixture_meta = reference["fixtures"][name]
        check(sha(fixture_meta["path"]) == fixture_meta["sha256"], name + "/fixture_hash")
        fixture = read(fixture_meta["path"])
        state = read(diagnostic / (name + "-starting-state.json"))
        best = read(stage["best_checkpoint"])
        latest = read(diagnostic / (name + "-latest.json"))
        rows = epochs[name]
        check(len(rows) == stage["epochs"], name + "/epoch_count")
        check([row["epoch"] for row in rows] == list(range(stage["epochs"])), name + "/epoch_order")
        check(stage["source_cap"] == cfg["epochs"] and stage["full_protocol"], name + "/source_cap")
        check(sha(stage["best_checkpoint"]) == stage["best_checkpoint_sha256"], name + "/best_hash")
        check(state["step"] == 0 and all(x == 0 for x in leaves(state["first_moments"]))
              and all(x == 0 for x in leaves(state["second_moments"])), name + "/optimizer_reset")
        check(all(math.isfinite(x) for x in leaves(latest["model"])) and
              all(math.isfinite(x) for x in leaves(best["model"])), name + "/finite_checkpoints")
        if args.isolated_start is not None:
            expected = read(args.isolated_start)["parameters"]
            check(state == read(args.isolated_start), name + "/exact_isolated_start")
        elif prior_best is None:
            expected = fixture["parameters"]
        elif name == "mdpf":
            prefixes = ("observation_encoder.", "particle_encoder_for_weights_model.", "weight_model.")
            expected = {k: (prior_best["parameters"][k] if k.startswith(prefixes) else v)
                        for k, v in fixture["parameters"].items()}
        else:
            expected = {}
            for key in fixture["parameters"]:
                source_key = key.removeprefix("resampling_") if key.startswith("resampling_") else key
                expected[key] = prior_best["parameters"][source_key]
        check(state["parameters"] == expected, name + "/actual_best_checkpoint_transfer")
        frozen = {k: v for k, v in state["parameters"].items() if k not in fixture["trainable_names"]}
        check(all(latest["model"]["parameters"][k] == v and best["model"]["parameters"][k] == v
                  for k, v in frozen.items()), name + "/frozen_parameters")
        rates = fixture["learning_rates"]
        groups = [{"params": [torch.nn.Parameter(torch.zeros(()))], "lr": rate}
                  for rate in rates.values()]
        optimizer = torch.optim.Adam(groups)
        scheduler_args = dict(cfg["lr_scheduler_params"])
        scheduler_args.pop("scheduler_type")
        scheduler_args.pop("verbose", None)  # recorded source API compatibility change
        scheduler_args.setdefault("start_epoch", 0)
        scheduler = namespace["CustomReduceLROnPlateau"](optimizer, **scheduler_args)
        early = namespace["EarlyStopping"](patience=cfg["early_stopping_patience"],
                                           start_offset=cfg["early_stopping_start_offset"])
        best_loss, best_epoch = math.inf, -1
        for row in rows:
            epoch, event = row["epoch"], row["event"]
            label = name + "/epoch" + str(epoch)
            for split in ("training", "validation"):
                computed = aggregates[(name, epoch, split)]
                check(computed == row["splits"][split]["weighted_loss"], label + "/" + split + "/loss_recomputed")
            loss = aggregates[(name, epoch, "validation")]
            improved = loss < best_loss
            best_loss = min(loss, best_loss)
            changed = scheduler.step(loss)
            if changed:
                early.lr_changed()
            with contextlib.redirect_stdout(io.StringIO()):
                early(loss)
            if improved and not early.early_stop:
                best_epoch = epoch
            expected_event = dict(epoch=epoch, validation_loss=loss, new_best_loss=improved,
                                  scheduler_event=changed, learning_rates={k: g["lr"] for k, g in zip(rates, optimizer.param_groups)},
                                  stop=early.early_stop, save_best=improved and not early.early_stop,
                                  best_checkpoint_epoch=best_epoch)
            check(event == expected_event, label + "/actual_author_controller")
            check(all(value == 1 for value in row["trace_counts"].values()) and
                  set(row["trace_counts"]) == {"64", "8", "256", "232"}, label + "/stable_trace")
            check(row["splits"]["training"]["optimizer_step"] == 79 * (epoch + 1) and
                  row["splits"]["validation"]["optimizer_step"] == 79 * (epoch + 1), label + "/step_count")
            check(not early.early_stop or epoch == len(rows) - 1, label + "/no_post_stop_epoch")
        check(stage["termination"] == ("SOURCE_EARLY_STOP" if early.early_stop else "SOURCE_CAP"), name + "/termination")
        check(early.early_stop or len(rows) == cfg["epochs"], name + "/uncapped")
        check(latest["controller"]["early"] == vars(early), name + "/saved_source_controller_state")
        check(stage["best_epoch"] == best_epoch == best["next_epoch"] - 1, name + "/best_epoch")
        check(latest["next_epoch"] == len(rows) and latest["model"]["step"] == 79 * len(rows), name + "/latest_step")
        prior_best = best["model"]
        summaries.append(dict(stage=name, epochs=len(rows), best_epoch=best_epoch, termination=stage["termination"],
                              best_validation_loss=best_loss, batches=len(rows) * 83,
                              peak_allocator_bytes=max(row["allocator"]["peak"] for row in rows)))
    terminal = manifest["status"] != "RUNNING"
    expected_stages = {"mdpf"} if args.isolated_start is not None else {"weights", "mdpf", "am_dpf"}
    check(not terminal or stage_names == expected_stages, "terminal_all_declared_stages")
    if terminal:
        check(manifest["status"] == "COMPLETED" and manifest["returncode"] == 0, "terminal_success")
        result = read(diagnostic / "result.json")
        check(sha(diagnostic / "result.json") == manifest["result_sha256"], "terminal_result_hash")
        check(result["status"] == "PASS" and (result["selected_stages_complete"] if args.isolated_start is not None else result["source_training_protocol_complete"])
              and result["checks_failed"] == 0, "terminal_result_validity")
    verdict = "FAIL" if errors else ("PASS" if terminal else "IN_PROGRESS_COMPLETED_STAGES_PASS")
    report = dict(schema="kdm_training_terminal_audit_v1", verdict=verdict,
                  replication="INCOMPLETE", run=str(args.run),
                  run_manifest_sha256=sha(args.run/"manifest.json"), command=sys.argv,
                  source_pin=reference["source_pin"], source_controller_sha256=sha(SOURCE),
                  audit_sha256=sha(__file__), torch_version=torch.__version__,
                  controller_api_compatibility=compatibility,
                  gpu_intentionally_hidden=True, stages=summaries, checks=checks,
                  failures=errors, native_keys=len(keys), wall_seconds=time.monotonic()-started,
                  not_concluded="Numerical compatibility, downstream accuracy, statistical no-worse performance, or replication acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("verdict", "checks", "failures", "wall_seconds", "stages")}))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    # Match the already reviewed CPU reference environment before loading Torch.
    reference_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
    if os.environ.get("LD_LIBRARY_PATH") != reference_lib:
        environment = dict(os.environ, LD_LIBRARY_PATH=reference_lib)
        os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve())] + sys.argv[1:], environment)
    main()

