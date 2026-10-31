#!/usr/bin/env python3
"""Continue a frozen full-model posterior cohort using its original numerical source.

Only orchestration lives here. Tuning is never repeated; retained loading,
chunk replay, posterior assessment and reference checks use the frozen source.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def native(value):
    return json.loads(json.dumps(value, allow_nan=False))


def tree_hashes(path):
    return {str(p.resolve()): digest(p) for p in Path(path).rglob("*")
            if p.is_file() and p.name != ".writer.lock"}


def check_hashes(records):
    for path, expected in records.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError("continuation source/input changed: " + path)


def validate_inventory(candidates, members, rows):
    ids = [c["candidate_id"] for c in candidates]
    prior_ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids) or len(set(prior_ids)) != len(prior_ids):
        raise ValueError("duplicate candidate identity")
    if set(ids) != set(prior_ids) or not set(members) <= set(ids):
        raise ValueError("continuation candidate inventory changed")
    return {r["candidate_id"]: r for r in rows}


def checked_gpu_identity(expected):
    command = ["nvidia-smi", f"--id={expected['index']}",
               "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    current = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    fields = next(csv.reader(current.stdout.splitlines()))
    index, uuid, name, total, used, utilization = (value.strip() for value in fields)
    if int(index) != expected["index"] or uuid != expected["uuid"]:
        raise ValueError("physical GPU identity changed")
    return dict(index=int(index), uuid=uuid, name=name, memory_total_mib=int(total),
                memory_used_mib=int(used), utilization_percent=int(utilization))


def preparation(previous, source):
    previous, source = Path(previous).resolve(), Path(source).resolve()
    prior = read(previous / "manifest.json")
    if prior["status"] in {"running", "preparing", "qualifying", "posterior"}:
        raise ValueError("cannot continue an active run")
    tuning_run = Path(prior.get("tuning_run", previous)).resolve()
    original = read(tuning_run / "manifest.json")
    device_file = tuning_run.parent / "trusted-gpu-probe.json"
    device = next(g for g in read(device_file)["gpu_inventory"]
                  if str(g["index"]) == original["cuda_visible_devices"])
    if Path(original["command"][1]).resolve().parents[1] != source:
        raise ValueError("numerical source differs from original command")
    if original["design"]["method"] not in {"fixed_hmc", "randomized_hmc", "snaper", "mams"}:
        raise ValueError("excluded or unsupported method")
    if original["tuning_completion_status"] != "complete":
        raise ValueError("posterior continuation requires complete tuning")
    check_hashes(original["source_and_input_sha256"])
    design = read(original["design_file"])
    if digest(original["design_file"]) != original["design_sha256"] or design != original["design"]:
        raise ValueError("frozen design changed")
    controller = read(tuning_run / "search/tuning/tuning_checkpoint.json")["result"]
    candidates = controller["candidates"]
    if candidates != sorted(candidates, key=lambda c: c["creation_ordinal"]):
        raise ValueError("original candidate creation order changed")
    member_paths = sorted((tuning_run / "search/members").glob("*-member.json"))
    member_ids = [read(path)["candidate_id"] for path in member_paths]
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != set(controller["verified_candidate_ids"]):
        raise ValueError("verified member export inventory changed")
    report = read(previous / "posterior/result.json")
    validate_inventory(candidates, member_ids, report["rows"])
    if set(report["verified_candidate_ids"]) != set(member_ids):
        raise ValueError("previous verified inventory changed")
    inherited = dict(prior.get("continuation_inputs_sha256", {}))
    audit_path = tuning_run.parent / "terminal-audit-01/preserved-files.json"
    if audit_path.exists():
        inherited.update(read(audit_path))
    check_hashes(inherited)
    inherited.update(tree_hashes(previous / "posterior"))
    inherited.update(tree_hashes(tuning_run / "search"))
    for path in (previous / "manifest.json", tuning_run / "manifest.json", device_file):
        inherited[str(path)] = digest(path)
    return dict(original=original, design=design, tuning_run=str(tuning_run),
                candidates=candidates, member_paths=member_paths, report=report,
                device=device, inputs={**original["source_and_input_sha256"], **inherited})


def member_source(row):
    value = row.get("result_file")
    return Path(value).parent if value is not None else None


def validate_checkpoint_prefix(store):
    """Missing committed work is corruption; unfinished attempts may be replayed."""
    keys = sorted(path.name for path in (store / "committed").iterdir()) if (store / "committed").exists() else []
    if keys != [f"chunk-{index:06d}" for index in range(len(keys))]:
        raise ValueError("saved committed checkpoint prefix has a gap")
    for receipt in (store / "attempts").glob("*/committed.json"):
        if read(receipt)["key"] not in keys:
            raise ValueError("saved committed checkpoint is missing")


def validate_member_source(member, config, names, source):
    identity = dict(member_hash=member.member_hash,
                    policy=native(config.payload(chain_count=int(member.initial_active_state.shape[0]))),
                    quantities=list(names))
    if source is not None and not source.is_dir():
        raise ValueError("saved member directory missing")
    if source is not None:
        if read(source / "policy.json") != identity["policy"]:
            raise ValueError("saved member policy or seeds changed")
        store = source / "chunks"
        if (store / "identity.json").exists():
            if read(store / "identity.json") != identity:
                raise ValueError("saved checkpoint member or quantities changed")
            validate_checkpoint_prefix(store)
        elif store.exists() and any(store.iterdir()):
            raise ValueError("checkpoint identity missing")
        if (source / "result.json").exists():
            result = read(source / "result.json")
            if result["config"] != identity["policy"]:
                raise ValueError("saved terminal posterior policy changed")
            if result.get("retained_results_per_chain", 0) and not (source / "retained_raw.tensor").exists():
                raise ValueError("saved terminal posterior draws missing")
    return identity


def copy_store(source, destination):
    with (source / ".writer.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = tree_hashes(source)
        shutil.copytree(source, destination)
        relative = {str(Path(p).relative_to(source)): h for p, h in before.items()}
        copied = {str(Path(p).relative_to(destination)): h for p, h in tree_hashes(destination).items()}
        if relative != copied or tree_hashes(source) != before:
            raise ValueError("checkpoint copy changed content")


def run_cohort(*, candidates, members, previous_report, output, factory, names,
               transform, reference_check, check_inputs):
    """Resume by original candidate index; all numerical work uses shared APIs."""
    import tensorflow as tf
    from bayesfilter.inference import run_hmc_posterior
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.runtime.execution_budget import ExecutionBudgetExceeded, require_execution_budget

    prior = validate_inventory(candidates, members, previous_report["rows"])
    configs = {}
    for index, candidate in enumerate(candidates):
        cid = candidate["candidate_id"]
        if cid in members:
            config = factory(members[cid], index)
            source = member_source(prior[cid])
            identity = validate_member_source(members[cid], config, names, source)
            configs[cid] = (config, source, identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [dict(prior[c["candidate_id"]]) for c in candidates]
    report = dict(status="running", rows=rows, verified_candidate_ids=list(members),
                  tuning_membership_unchanged=True, method_ranking=None, release_qualified=False)

    def save():
        report["delivered_count"] = sum(bool(r.get("delivered")) for r in rows)
        write(output / "result.json", report)

    save()
    try:
        for index, row in enumerate(rows):
            cid = row["candidate_id"]
            if cid not in members:
                continue
            require_execution_budget()
            check_inputs()
            config, source, identity = configs[cid]
            terminal = read(source / "result.json") if source is not None and (source / "result.json").exists() else None
            if terminal is not None and "campaign_resource_cap" not in terminal.get("hard_vetoes", []):
                comparison = row.get("reference")
                reference_was_saved = comparison is not None
                if comparison is None:
                    path = source / "retained_raw.tensor"
                    values = tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64) if path.exists() else None
                    comparison = reference_check(values, config)
                expected = bool(terminal["passed"] and not terminal["hard_vetoes"] and comparison["passed"])
                if reference_was_saved and "delivered" in row and bool(row["delivered"]) != expected:
                    raise ValueError("saved delivered decision contradicts member evidence")
                row.update(status="posterior_checks_passed" if terminal["passed"] else "posterior_incomplete",
                    delivered=expected, reference=comparison, posterior_hard_vetoes=terminal["hard_vetoes"],
                    continuation_action="carried_terminal_result")
                save()
                continue

            destination = output / f"member-{index:03d}"
            destination.mkdir()
            write(destination / "policy.json", identity["policy"])
            if source is not None and (source / "chunks/identity.json").exists():
                copy_store(source / "chunks", destination / "chunks")
            row.update(status="posterior_running", result_file=str(destination / "result.json"),
                       delivered=False, continuation_action="resumed_prefix" if source is not None else "started")
            row.pop("reference", None)
            row.pop("posterior_hard_vetoes", None)
            save()
            records = []

            def progress(event, payload):
                write(output.parent / "progress.json", dict(candidate_id=cid, member_index=index,
                    event=event, chunk=payload.get("key"), at_utc=utc(),
                    replayed_chunks=sum(r["replayed"] for r in records),
                    new_chunks=sum(not r["replayed"] for r in records)))

            def budget(_):
                progress("controller_check", {})
                require_execution_budget()

            with DurableTensorCheckpoint(destination / "chunks", identity, event_callback=progress) as store:
                records = store.records
                try:
                    result = run_hmc_posterior(member=members[cid], config=config,
                        model_transform=transform, parameter_names=names,
                        checkpoint_store=store, budget_check=budget)
                finally:
                    write(destination / "checkpoint-replay.json", dict(records=records,
                        replayed_count=sum(r["replayed"] for r in records),
                        new_count=sum(not r["replayed"] for r in records)))
            for key in ("private_warmup_raw", "private_retained_raw"):
                if result.get(key) is not None:
                    tf.io.write_file(str(destination / (key.removeprefix("private_") + ".tensor")),
                                     tf.io.serialize_tensor(result[key]))
            write(destination / "result.json", {k: v for k, v in result.items() if not k.startswith("private_")})
            comparison = reference_check(result.get("private_retained_raw"), config)
            row.update(status="posterior_checks_passed" if result["passed"] else "posterior_incomplete",
                       reference=comparison, posterior_hard_vetoes=result["hard_vetoes"],
                       delivered=bool(result["passed"] and not result["hard_vetoes"] and comparison["passed"]))
            check_inputs()
            save()
        report["status"] = "completed"
    except ExecutionBudgetExceeded:
        report["status"] = "resource_stopped"
    except BaseException as exc:
        report.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if report["status"] != "completed":
            for row in rows:
                if row["status"] == "posterior_running":
                    row["status"] = "posterior_interrupted_" + report["status"]
        save()
    return report


def worker(args):
    started = time.monotonic()
    prepared = preparation(args.previous, args.source_root)
    original, design = prepared["original"], prepared["design"]
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("GPU memory growth must be enabled before framework import")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != original["cuda_visible_devices"]:
        raise ValueError("continuation must preserve original visible GPU identity")
    device = checked_gpu_identity(prepared["device"])
    sys.path.insert(0, str(args.source_root))
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.runtime.execution_budget import execution_budget
    from bayesfilter.inference import run_hmc_posterior
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.inference.hmc_candidate_set_retained import load_hmc_candidate_retained_runners
    from bayesfilter.testing.inference_validation.modern_nonlinear import make_nonlinear_comparison_target
    from bayesfilter.testing.inference_validation.modern_full_model_consumer import reference_screen
    from scripts.run_modern_hmc_frozen_full_model import posterior_factory
    from scripts.audit_mams_reference_compatibility import load_compatible_reference, model_quantities, quantity_names
    import bayesfilter.inference.hmc_candidate_set_retained as retained_module
    authorities = (run_hmc_posterior, DurableTensorCheckpoint, load_hmc_candidate_retained_runners,
                   make_nonlinear_comparison_target, reference_screen, posterior_factory,
                   load_compatible_reference, model_quantities, quantity_names)
    imports = {value.__name__: str(Path(inspect.getsourcefile(value)).resolve()) for value in authorities}
    if any(not Path(path).is_relative_to(args.source_root) for path in imports.values()):
        raise ValueError("numerical imports escaped the immutable source")
    if tf.__version__ != original["tf_version"] or tfp.__version__ != original["tfp_version"]:
        raise ValueError("framework version changed")
    tf.config.experimental.enable_tensor_float_32_execution(original["tf32_enabled"])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="preparing", started_utc=utc(), command=[sys.executable, *sys.argv],
        plan=args.plan, tuning_run=prepared["tuning_run"], previous_run=str(args.previous),
        numerical_source=str(args.source_root), git_commit=original["git_commit"],
        driver_sha256=digest(__file__), environment=sys.executable, tf_version=tf.__version__,
        tfp_version=tfp.__version__, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        physical_gpu=device,
        memory_policy=memory, gpu_trust_basis="trusted_escalated_gpu_execution",
        use_xla=True, tf32_enabled=original["tf32_enabled"], design=design,
        design_sha256=original["design_sha256"], source_and_input_sha256=original["source_and_input_sha256"],
        continuation_inputs_sha256=prepared["inputs"], max_wall_seconds=args.max_wall_seconds,
        method_ranking=None, release_qualified=False)
    write(args.output / "manifest.json", manifest)
    try:
        fixture = make_nonlinear_comparison_target(design["target"], jit_compile=True)
        reference = load_compatible_reference(design["reference_compatibility"], current_source=args.source_root,
            target=design["target"], data_sha256=original["target"]["data_sha256"],
            data=fixture.observations.numpy().tolist())
        if reference != original["reference"]:
            raise ValueError("frozen reference changed")
        members = load_hmc_candidate_retained_runners(prepared["member_paths"], adapter=fixture.adapter)
        factory = posterior_factory(design["posterior"], target=design["target"], use_xla=True)
        settings = design["posterior"]
        for index, candidate in enumerate(prepared["candidates"]):
            cid = candidate["candidate_id"]
            if cid in members:
                row = next(r for r in prepared["report"]["rows"] if r["candidate_id"] == cid)
                validate_member_source(members[cid], factory(members[cid], index),
                                       reference["quantities"], member_source(row))
                comparison = row.get("reference")
                if comparison is not None and (
                    comparison.get("precision_policy") != native(factory(members[cid], index).assessment_policy.precision.payload())
                    or ("mcse_multiplier" in comparison and comparison["mcse_multiplier"] != settings["reference_mcse_multiplier"])
                    or ("normalized_tolerance" in comparison and comparison["normalized_tolerance"] != settings["reference_normalized_tolerance"])
                ):
                    raise ValueError("saved reference-screen policy changed")
        check_hashes(prepared["inputs"])
        manifest.update(status="preflight_passed" if args.preflight_only else "posterior",
            verified_candidate_ids=list(members), numerical_import_file=retained_module.__file__,
            numerical_import_files=imports)
        write(args.output / "manifest.json", manifest)
        if args.preflight_only:
            write(args.output / "result.json", dict(status="preflight_passed", verified_count=len(members),
                sampling_performed=False, original_policy_and_seed_checks_passed=True))
            return 0
        with execution_budget(check=lambda: time.monotonic() - started < args.max_wall_seconds,
                              replace_wall_time_limit=True):
            report = run_cohort(candidates=prepared["candidates"], members=members,
                previous_report=prepared["report"], output=args.output / "posterior", factory=factory,
                names=quantity_names(design["target"]),
                transform=lambda q: model_quantities(q, target=design["target"]),
                reference_check=lambda values, config: reference_screen(values, reference,
                    mcse_multiplier=settings["reference_mcse_multiplier"],
                    normalized_tolerance=settings["reference_normalized_tolerance"],
                    precision_policy=config.assessment_policy.precision),
                check_inputs=lambda: check_hashes(prepared["inputs"]))
        check_hashes(prepared["inputs"])
        manifest["status"] = report["status"]
        write(args.output / "result.json", report)
    except BaseException as exc:
        manifest.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic() - started, finished_utc=utc())
        write(args.output / "manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 1


def finalize_stop(output, prepared, status):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "posterior/result.json"
    report = read(path) if path.exists() else native(prepared["report"])
    for row in report["rows"]:
        if row["status"] == "posterior_running":
            row["status"] = "posterior_interrupted_" + status
    report.update(status=status, delivered_count=sum(bool(r.get("delivered")) for r in report["rows"]))
    write(path, report)
    write(output / "result.json", report)
    manifest_path = output / "manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {}
    manifest.update(status=status, tuning_run=prepared["tuning_run"], finished_utc=utc(),
                    continuation_inputs_sha256=prepared.get("inputs", {}))
    write(manifest_path, manifest)


def supervise(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    prepared = preparation(args.previous, args.source_root)
    args.output.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--previous", str(args.previous),
        "--source-root", str(args.source_root), "--output", str(args.output / "run"),
        "--max-wall-seconds", str(max(1., args.max_wall_seconds - args.stop_margin_seconds)),
        "--plan", args.plan]
    if args.preflight_only:
        command.append("--preflight-only")
    manifest = dict(status="running", started_utc=utc(), command=[sys.executable, *sys.argv],
        child_command=command, source_root=str(args.source_root), previous_run=str(args.previous),
        driver_sha256=digest(__file__), environment=sys.executable, plan=args.plan,
        max_wall_seconds=args.max_wall_seconds, stop_margin_seconds=args.stop_margin_seconds,
        gpu_trust_basis="trusted_escalated_gpu_execution",
        memory_growth_environment=os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"))
    write(args.output / "manifest.json", manifest)
    code = 1
    with (args.output / "worker.log").open("x") as log:
        with subprocess.Popen(command, cwd=args.source_root, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                while True:
                    remaining = args.max_wall_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, args.max_wall_seconds)
                    try:
                        code = process.wait(timeout=min(30., remaining))
                        break
                    except subprocess.TimeoutExpired:
                        manifest["last_checked_utc"] = utc()
                        progress = args.output / "run/progress.json"
                        if progress.exists():
                            manifest["last_numerical_progress"] = read(progress)
                        write(args.output / "manifest.json", manifest)
            except BaseException as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = "supervisor_budget_exhausted" if isinstance(exc, subprocess.TimeoutExpired) else "supervisor_interrupted"
                finalize_stop(args.output / "run", prepared, status)
                manifest["status"] = status
            else:
                child = args.output / "run/manifest.json"
                manifest["status"] = read(child)["status"] if child.exists() else "worker_failed_before_manifest"
                if manifest["status"] not in {"completed", "preflight_passed", "resource_stopped"}:
                    finalize_stop(args.output / "run", prepared, manifest["status"])
    manifest.update(exit_code=code, elapsed_seconds=time.monotonic() - started, finished_utc=utc())
    write(args.output / "manifest.json", manifest)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--stop-margin-seconds", type=float, default=60.)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("previous", "source_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("finite positive allocation required")
    if not math.isfinite(args.stop_margin_seconds) or args.stop_margin_seconds < 0:
        parser.error("finite nonnegative finalization margin required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        parser.error("set TF_FORCE_GPU_ALLOW_GROWTH=true before launching")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
