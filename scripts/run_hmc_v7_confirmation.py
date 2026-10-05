"""Execute one frozen 96-slot v7 confirmation design; never declare a release."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


FAMILIES = ("lgssm_qr", "nonlinear", "funnel_residual")
SCHEMA = "bayesfilter.hmc_v7_confirmation_campaign.v1"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_design(design):
    required = {"schema", "source_manifest_sha256", "gpu_uuid", "budget_seconds",
        "search_cap_seconds", "closeout_cap_seconds", "settlement_reserve_seconds",
        "slots", "prices", "provenance", "readiness"}
    if set(design) != required or design["schema"] != SCHEMA:
        raise ValueError("unsupported confirmation design")
    if not re.fullmatch("[0-9a-f]{64}", design["source_manifest_sha256"]):
        raise ValueError("frozen source identity required")
    if not str(design["gpu_uuid"]).startswith("GPU-") or not design["provenance"]:
        raise ValueError("explicit device and design provenance required")
    for name in ("budget_seconds", "search_cap_seconds", "closeout_cap_seconds", "settlement_reserve_seconds"):
        value = design[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("budgets must be finite and positive")
    if (max(design["search_cap_seconds"], design["closeout_cap_seconds"]) > 3600
            or design["budget_seconds"] <= design["settlement_reserve_seconds"]):
        raise ValueError("invalid search/closeout/settlement budgets")
    if set(design["prices"]) != set(FAMILIES):
        raise ValueError("complete prices for all three families required")
    if any(not isinstance(path, str) or not Path(path).is_absolute() for path in design["prices"].values()):
        raise ValueError("price directories must be absolute")
    readiness = design["readiness"]
    readiness_fields = {"wait_cap_seconds", "poll_seconds", "minimum_free_mib", "provenance"}
    if (not isinstance(readiness, dict)
            or set(readiness) not in (readiness_fields, readiness_fields | {"require_no_foreign_compute"})
            or not readiness["provenance"]):
        raise ValueError("explicit readiness policy and provenance required")
    if type(readiness.get("require_no_foreign_compute", False)) is not bool:
        raise ValueError("require_no_foreign_compute must be Boolean")
    for name in ("wait_cap_seconds", "poll_seconds"):
        value = readiness[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("readiness times must be finite and positive")
    if (readiness["wait_cap_seconds"] > 3600
            or readiness["poll_seconds"] > min(60, readiness["wait_cap_seconds"])
            or type(readiness["minimum_free_mib"]) is not int or readiness["minimum_free_mib"] <= 0):
        raise ValueError("invalid bounded readiness policy")
    ids, seeds, counts = set(), set(), Counter()
    for slot in design["slots"]:
        if set(slot) != {"slot_id", "family", "seed"}:
            raise ValueError("unexpected slot fields")
        sid, family, seed = slot["slot_id"], slot["family"], slot["seed"]
        if (not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sid)
                or sid in ids or family not in FAMILIES or not isinstance(seed, (tuple, list))
                or len(seed) != 2 or any(type(v) is not int or not 0 <= v < 2**31 for v in seed)
                or tuple(seed) in seeds or tuple(seed) in {(20261002, 2501), (20261002, 2502), (20261002, 2503)}):
            raise ValueError("invalid, duplicate or development slot/seed")
        ids.add(sid); seeds.add(tuple(seed)); counts[family] += 1
    if counts != Counter({family: 32 for family in FAMILIES}):
        raise ValueError("preserve all 32 slots per family")


def slot_configuration(profile, slot, search_cap):
    if profile["wall_seconds"] != search_cap or profile["expected_outcome"] != "positive_delivery":
        raise ValueError("confirmation must preserve the priced search profile")
    if profile["case_id"] != "development-" + slot["family"] or profile["classification"] != "development":
        raise ValueError("wrong family or classification in development profile")
    result = deepcopy(profile)
    result.update(seed=list(slot["seed"]), classification="confirmation",
        case_id="confirmation-" + slot["family"],
        provenance="Frozen prepared-model confirmation; identical numerical settings to the linked complete development price. Independent confirmation seed; no posterior or default-promotion claim.")
    return result


def execute_slots(design, attempt, save, *, now=time.monotonic, started=None):
    """Bounded serial supervisor. Tests inject a child and clock, never evidence."""
    validate_design(design)
    started = now() if started is None else started
    limit = design["budget_seconds"] - design["settlement_reserve_seconds"]
    cap = (design["search_cap_seconds"] + design["closeout_cap_seconds"]
           + design["readiness"]["wait_cap_seconds"])
    records, invalid = [], False
    for slot in design["slots"]:
        remaining = limit - (now() - started)
        before = now()
        if invalid or remaining <= 0:
            result = dict(exit_code=3, disposition="unstarted_after_harness_failure" if invalid else "budget_deferred")
        else:
            try:
                result = dict(attempt(slot, min(cap, remaining)))
            except subprocess.TimeoutExpired:
                result = dict(exit_code=124, disposition="timeout")
            except Exception as error:
                result = dict(exit_code=1, disposition="harness_failure",
                    error_type=type(error).__name__, error=str(error))
            if type(result.get("exit_code")) is not int or result["exit_code"] not in (0, 2, 3, 124):
                invalid = True
                result["unclassified_return_code"] = result.get("exit_code")
                result["exit_code"] = 1
                result["disposition"] = "harness_failure"
        record = {**result, **slot, "source_manifest_sha256": design["source_manifest_sha256"],
            "wall_seconds": now() - before}
        records.append(record)
        save(records)
    return dict(status="harness_failure" if invalid else "execution_complete", outcomes=records,
        original_denominator=len(design["slots"]), wall_seconds=now()-started,
        release_ready=False, default_promoted=False)


def wait_for_memory(gpu_uuid, policy, probe, save, *, process_probe=None,
                    now=time.monotonic, sleep=time.sleep):
    """Bounded pre-import resource recovery; no claim about later GPU load."""
    require_idle = policy.get("require_no_foreign_compute", False)
    if require_idle and process_probe is None:
        raise ValueError("process-aware readiness requires a compute-process probe")
    started = now()
    observations = []
    while True:
        remaining = policy["wait_cap_seconds"] - (now() - started)
        if remaining <= 0:
            return dict(ready=False, gpu_initialized=False, observations=observations,
                        wall_seconds=now()-started, disposition="resource_deferred")
        try:
            inventory = probe(remaining)
            devices = [line for line in inventory.splitlines() if line.split(",")[0].strip() == gpu_uuid]
            if len(devices) != 1:
                raise ValueError("selected GPU is missing or duplicated")
            free = int(devices[0].split(",")[1])
            processes, foreign, process_checked = None, [], False
            remaining = policy["wait_cap_seconds"]-(now()-started)
            if require_idle and remaining > 0:
                processes = process_probe(remaining).splitlines()
                seen = set()
                for line in processes:
                    fields = [value.strip() for value in line.split(",")]
                    if len(fields) != 2 or not fields[0].startswith("GPU-"):
                        raise ValueError("malformed GPU compute-process inventory")
                    pid = int(fields[1])
                    key = (fields[0], pid)
                    if pid <= 0 or key in seen:
                        raise ValueError("invalid or duplicate GPU compute-process identity")
                    seen.add(key)
                    if fields[0] == gpu_uuid:
                        foreign.append(pid)
                process_checked = True
            # The worker has not imported TensorFlow yet, so every reported
            # compute process on its selected GPU belongs to another workload.
            ready = (free >= policy["minimum_free_mib"] and (not require_idle or process_checked)
                     and not foreign and now()-started < policy["wait_cap_seconds"])
            observations.append(dict(elapsed_seconds=now()-started, inventory=inventory.splitlines(),
                compute_processes=processes, foreign_pids=foreign,
                process_inventory_checked=process_checked, ready=ready))
        except Exception as error:
            save(dict(ready=False, gpu_initialized=False, observations=observations,
                      wall_seconds=now()-started, disposition="harness_failure", error=str(error)))
            raise
        record = dict(ready=ready, gpu_initialized=False, observations=observations,
                      wall_seconds=now()-started, disposition="ready" if ready else "resource_deferred")
        save(record)
        if ready:
            return record
        remaining = policy["wait_cap_seconds"]-(now()-started)
        if remaining > 0:
            sleep(min(policy["poll_seconds"], remaining))


def read_model_result(path, exit_code):
    model = json.loads(path.read_text()) if path and path.exists() else {}
    if not isinstance(model, dict) or (exit_code == 0 and not model):
        raise ValueError("successful child lacks a structured model result")
    return model


def worker(args):
    source, root = args.source.resolve(), args.output.resolve()
    sys.path.insert(0, str(source)); os.chdir(source)
    from scripts.run_hmc_v7_release_prices import check_source, full_search_delivered
    signature = check_source(source)
    config = json.loads(args.config.read_text())
    if config["classification"] != "confirmation":
        raise ValueError("confirmation worker requires a frozen confirmation configuration")
    design = json.loads(args.design.read_text()); validate_design(design)
    if signature != design["source_manifest_sha256"] or args.gpu != design["gpu_uuid"]:
        raise ValueError("worker source/device differs from frozen design")
    readiness = wait_for_memory(args.gpu, design["readiness"],
        lambda remaining: subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=remaining),
        lambda record: write(root/"resource.json", record),
        process_probe=lambda remaining: subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True, timeout=remaining))
    write(root/"resource.json", readiness)
    if not readiness["ready"]:
        return 3
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.testing.acceptance_decision_models import run_model
    manifest = dict(scope="independent fixed-design prepared-model v7 confirmation",
        source_manifest_sha256=signature, git_commit=json.loads((source.parent/"assembly.json").read_text())["git_commit"],
        gpu_uuid=args.gpu, device_inventory=readiness["observations"][-1]["inventory"], memory_policy=memory,
        readiness=readiness, readiness_policy=design["readiness"],
        tensorflow_version=tf.__version__, tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        jit_compile=True, dtype="float64", seed=config["seed"], data=config["data"],
        command=sys.argv, environment=sys.executable, CPU_affinity=sorted(os.sched_getaffinity(0)),
        configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md",
        started_utc=datetime.now(timezone.utc).isoformat())
    write(root/"manifest.json", manifest)
    result = run_model(config, root/"model", manifest=manifest)
    actual_devices = set()
    for path in (root/"model/tuning/numerical_chunks").glob("*.json"):
        chunk = json.loads(path.read_text())
        actual_devices.add(chunk["samples_device"])
        if not chunk["runtime"]["jit_compile"] or not chunk["runtime"]["use_xla"]:
            raise ValueError("confirmation chunk did not use XLA")
    if not actual_devices or any("GPU:0" not in device for device in actual_devices):
        raise ValueError("confirmation lacks GPU sample evidence")
    if check_source(source) != signature:
        raise ValueError("source changed during confirmation")
    write(root/"device_result.json", dict(sample_devices=sorted(actual_devices),
        allocator=tf.config.experimental.get_memory_info("GPU:0")))
    return 0 if full_search_delivered(result) else 2


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--design", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--gpu")
    args = p.parse_args()
    os.environ.update(TF_FORCE_GPU_ALLOW_GROWTH="true", TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
        BAYESFILTER_PRELOAD_CUSTOM_OP="0", TF_CPP_MIN_LOG_LEVEL="2",
        CUDA_VISIBLE_DEVICES=args.gpu if args.config else "-1")
    os.sched_setaffinity(0, {8, 9, 10, 11})
    if args.config:
        if not args.gpu or not args.design:
            raise ValueError("explicit GPU and frozen design required")
        return worker(args)
    if not args.design:
        raise ValueError("frozen design required")
    source, root = args.source.resolve(), args.output.resolve()
    design = json.loads(args.design.read_text()); validate_design(design)
    sys.path.insert(0, str(source))
    from scripts.run_hmc_v7_release_prices import check_source
    from scripts.analyze_hmc_v7_confirmation import price_report, delivery_report
    if check_source(source) != design["source_manifest_sha256"]:
        raise ValueError("design and actual source differ")
    prices = [str(Path(directory)/"result.json") for directory in dict.fromkeys(design["prices"].values())]
    forecast = price_report(prices, replications_per_family=32,
        available_gpu_seconds=design["budget_seconds"]-design["settlement_reserve_seconds"])
    if (not forecast["point_forecast_fits_budget"] or forecast["source_manifest_sha256"] != design["source_manifest_sha256"]
            or forecast["gpu_uuid"] != design["gpu_uuid"]):
        raise ValueError("complete source/device-consistent prices and affordable design required")
    if max(forecast["complete_case_seconds"].values()) > design["search_cap_seconds"] + design["closeout_cap_seconds"]:
        raise ValueError("slot process cap is below an observed complete price")
    forecast["readiness_allowance_seconds"] = len(design["slots"]) * design["readiness"]["wait_cap_seconds"]
    if (forecast["point_forecast_seconds"] + forecast["readiness_allowance_seconds"]
            > design["budget_seconds"] - design["settlement_reserve_seconds"]):
        raise ValueError("confirmation forecast must fund readiness waiting")
    profiles = {}
    for family in FAMILIES:
        price = Path(design["prices"][family])
        configuration_path = price/(family+"-config.json")
        profile = json.loads(configuration_path.read_text())
        recorded = json.loads((price/family/"model/configuration.json").read_text())
        manifest = json.loads((price/family/"manifest.json").read_text())
        if (profile != recorded or hashlib.sha256(configuration_path.read_bytes()).hexdigest() != manifest["configuration_sha256"]
                or manifest["source_manifest_sha256"] != design["source_manifest_sha256"]):
            raise ValueError("priced profile or source changed")
        profiles[family] = profile
    configs = {slot["slot_id"]: slot_configuration(profiles[slot["family"]], slot, design["search_cap_seconds"])
               for slot in design["slots"]}
    root.mkdir(parents=True, exist_ok=False)
    runner = root/"runner.py"; runner.write_bytes(Path(__file__).read_bytes())
    write(root/"design.json", design); write(root/"price-forecast.json", forecast)
    write(root/"manifest.json", dict(command=sys.argv, environment=sys.executable,
        started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid=design["gpu_uuid"],
        source_manifest_sha256=design["source_manifest_sha256"],
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md"))
    def attempt(slot, cap):
        directory = root/slot["slot_id"]; directory.mkdir()
        config = directory/"configuration.json"; write(config, configs[slot["slot_id"]])
        command = [sys.executable, str(runner), "--source", str(source), "--output", str(directory),
            "--config", str(config), "--gpu", design["gpu_uuid"], "--design", str(root/"design.json")]
        write(directory/"launch.json", dict(command=command, process_cap_seconds=cap))
        with (directory/"run.log").open("x") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=cap).returncode
        model_path = directory/"model/result.json"
        try:
            model = read_model_result(model_path, code)
            if model and (model.get("sampling_streams") != slot["seed"]
                          or model.get("classification") != "confirmation"):
                raise ValueError("model result differs from confirmation slot")
        except Exception as error:
            return dict(exit_code=1, model_result_path=str(model_path), result_read_error=str(error))
        return dict(exit_code=code, model_result_path=str(model_path))
    result = execute_slots(design, attempt, lambda rows: write(root/"progress.json", dict(outcomes=rows, release_ready=False)), started=started)
    outcomes = []
    for row in result["outcomes"]:
        path = Path(row["model_result_path"]) if row.get("model_result_path") else None
        try:
            model = read_model_result(path, row["exit_code"])
            outcomes.append({**row, "model_result": model})
        except Exception as error:
            result["status"] = "harness_failure"
            outcomes.append({**row, "model_result": {}, "result_read_error": str(error)})
    write(root/"outcomes.json", outcomes)
    try:
        if check_source(source) != design["source_manifest_sha256"]:
            raise ValueError("source identity changed")
    except Exception as error:
        result["status"] = "harness_failure"
        result["source_check_error"] = str(error)
    if result["status"] != "harness_failure":
        try:
            result["delivery"] = delivery_report(slots=design["slots"], outcomes=outcomes,
                source_manifest_sha256=design["source_manifest_sha256"])
        except Exception as error:
            result["status"] = "harness_failure"
            result["report_error"] = str(error)
    result["wall_seconds"] = time.monotonic()-started
    write(root/"result.json", result)
    print(json.dumps({k:result[k] for k in ("status", "wall_seconds", "release_ready")}))
    return 0 if result["status"] != "harness_failure" else 1


if __name__ == "__main__":
    raise SystemExit(main())
