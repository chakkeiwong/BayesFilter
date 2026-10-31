#!/usr/bin/env python3
"""Diagnostic paired OT ablation; never issues filter or HMC tuning authority."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_performance_repair as master

CAMPAIGN = ROOT / "docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01"
PLAN = ROOT / "docs/plans/surrogate-hmc-ot-five-iterations-2026-10-14.md"
FRESH_SEEDS = (26101401, 26101402)
TOLERANCE = .01
PASS = "WITHIN_ONE_PERCENT_ON_INSPECTED_INPUTS"


def read(path):
    return json.loads(Path(path).read_text())


def candidate_controls(controls):
    return replace(controls, reset_epsilon=1., reset_steps=5)


def comparison(baseline, candidate):
    """Post-run diagnostic only; never changes runtime selection or computation."""
    b, c = [baseline["value"], *baseline["score"]], [candidate["value"], *candidate["score"]]
    finite = len(b) == len(c) and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in b+c)
    healthy = finite and baseline["valid"] and candidate["valid"]
    if not healthy:
        return dict(status="INVALID_ARM", finite=finite, passed=False)
    delta = [y-x for x, y in zip(b, c)]
    errors = [abs(d)/max(1., abs(x)) for x, d in zip(b, delta)]
    passed = max(errors) <= TOLERANCE
    return dict(status=PASS if passed else "MATERIAL_DIFFERENCE", finite=True,
        passed=passed, log_likelihood_delta=delta[0], score_delta=delta[1:],
        likelihood_scaled_error=errors[0], score_scaled_errors=errors[1:],
        max_score_scaled_error=max(errors[1:]),
        score_norm_scaled_error=math.sqrt(sum(d*d for d in delta[1:])) /
            max(1., math.sqrt(sum(x*x for x in b[1:]))))


def summarize(rows):
    checks = [r["comparison"] for r in rows]
    if not rows or any(c["status"] == "INVALID_ARM" for c in checks):
        return dict(status="INVALID_ARM", paired_rows=len(rows),
                    invalid_rows=sum(c["status"] == "INVALID_ARM" for c in checks))
    centered = []
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        nominal = next(r for r in subset if r["position"] == 0)
        anchor = nominal["comparison"]["log_likelihood_delta"]
        centered.extend(r["comparison"]["log_likelihood_delta"]-anchor for r in subset)
    return dict(status=PASS if all(c["passed"] for c in checks) else "MATERIAL_DIFFERENCE",
        paired_rows=len(rows), material_rows=sum(not c["passed"] for c in checks),
        max_likelihood_scaled_error=max(c["likelihood_scaled_error"] for c in checks),
        max_score_scaled_error=max(c["max_score_scaled_error"] for c in checks),
        max_abs_log_likelihood_delta=max(abs(c["log_likelihood_delta"]) for c in checks),
        max_abs_centered_log_likelihood_delta=max(map(abs, centered)))


def available_timeout(ledger, campaign_left, diagnostic_left):
    if any(a["status"] == "RUNNING" for a in ledger["attempts"]):
        raise RuntimeError("campaign has an active worker")
    if master.campaign_attempts_remaining(ledger) <= 0:
        return 0.
    return max(0., min(2400., campaign_left, diagnostic_left))


def worker(name, campaign, output):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
    from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS, VALIDITY_FIELDS
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    selection = campaign / "fixed-design-scope-advance-01" / name / "filter-tuning" / name / "filter-selection.json"
    saved = read(selection)
    case = runtime.configure_precision(runtime.load_case(name),
        saved["scope"]["precision_policy"], saved["scope"]["tuning_seed_offset"])
    runtime.load_selection(case, selection)
    used_seeds = set(saved["calibration_seeds"]+saved["validation_seeds"]+[saved["claim_data_seed"]])
    if used_seeds.intersection(FRESH_SEEDS):
        raise ValueError("fresh datasets overlap prior tuning or saved claim data")
    baseline, candidate = case.controls, candidate_controls(case.controls)
    dtype = case.force_dtype
    q = tf.constant([[0.]*case.p, [-.15]*case.p, [-.05]*case.p,
        [.05]*case.p, [.15]*case.p, [.15 if k % 2 == 0 else -.15 for k in range(case.p)]], dtype)
    theta, _ = case.chart(q)
    sources = dict(case.scope()["source_sha256"])
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).resolve().is_relative_to(ROOT):
            p = Path(path).resolve()
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for relative, digest in sources.items():
        p = ROOT/relative
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed before snapshot: "+relative)
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    result = dict(status="RUNNING", model=name, scope=case.scope(),
        plan=str(PLAN), command=sys.argv, git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=sys.prefix, tensorflow=tf.__version__, cpu_only=False,
        memory_policy=memory_policy, tf32=False, jit_compile=True,
        gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
        selection=str(selection), selection_hash=case.selection_hash,
        baseline_controls=asdict(baseline), candidate_controls=asdict(candidate),
        beta=runtime.json_ready(case.beta), positions=runtime.json_ready(q),
        physical_parameters=runtime.json_ready(theta), source_sha256=sources,
        tolerance=TOLERANCE, reference_role="selected_finite_program_not_model_oracle",
        canonical_admission=False, posterior_sampling_complete=False,
        datasets=[], rows=[], calls=[], warmed_timings={"baseline":[], "candidate":[]})
    destination = output/"result.json"
    def save():
        result["wall_seconds"] = time.monotonic()-started
        runtime.save(destination, result)
    save()
    programs = {label: make_batched_score_filter(case.spec, runtime.N, case.horizon,
        controls, dtype, jit_compile=True, batch_size=2, precision_policy=case.precision_policy)
        for label, controls in (("baseline", baseline), ("candidate", candidate))}
    datasets = [("saved", case.observations, case.fixed, saved["claim_data_seed"], "saved_case_design")]
    for seed in FRESH_SEEDS:
        datasets.append((str(seed), tf.cast(runtime.fresh_observations(case, seed), dtype),
            tuple(tf.cast(x, dtype) for x in runtime.design(case, seed+700)), seed, seed+700))
    warmed = set()
    def evaluate(label, parameters, obs, fixed):
        start = time.monotonic()
        value = programs[label](parameters, obs, *fixed, case.beta)
        tf.nest.map_structure(lambda x: x.numpy(), value)
        seconds = time.monotonic()-start
        result["calls"].append(dict(arm=label, seconds=seconds, includes_compile=label not in warmed))
        warmed.add(label)
        return value, seconds
    for dataset_index, (label, obs, fixed, seed, design_seed) in enumerate(datasets):
        result["datasets"].append(dict(label=label, data_seed=seed, design_seed=design_seed,
            observations=runtime.json_ready(obs), input_sha256=[hashlib.sha256(
                tf.io.serialize_tensor(x).numpy()).hexdigest() for x in (theta, obs, *fixed, case.beta)]))
        for group in range(3):
            result["active"] = dict(dataset=label, group=group)
            save()
            runs = {}
            order = ("baseline", "candidate") if (dataset_index+group)%2 == 0 else ("candidate", "baseline")
            for arm in order:
                raw, _ = evaluate(arm, theta[group*2:group*2+2], obs, fixed)
                runs[arm] = runtime.json_ready({k:raw[k] for k in (
                    "value", "score", "valid", "steps_completed", "trace", "validity_trace", "covariance_tolerance")})
            runtime.save(output/f"{label}-group-{group}-health.json",
                dict(trace_fields=TRACE_FIELDS, validity_fields=VALIDITY_FIELDS, arms=runs))
            for j in range(2):
                row = dict(dataset=label, position=2*group+j)
                for arm in ("baseline", "candidate"):
                    raw = runs[arm]
                    row[arm] = dict(value=raw["value"][j], score=raw["score"][j],
                        valid=raw["valid"][j] and raw["steps_completed"][j] == case.horizon,
                        steps_completed=raw["steps_completed"][j])
                row["comparison"] = comparison(row["baseline"], row["candidate"])
                result["rows"].append(row)
            save()
    _, obs, fixed, _, _ = datasets[0]
    for repeat in range(3):
        for arm in (("candidate", "baseline") if repeat%2 == 0 else ("baseline", "candidate")):
            run, seconds = evaluate(arm, theta[:2], obs, fixed)
            valid = bool(tf.reduce_all(run["valid"] & (run["steps_completed"] == case.horizon)).numpy())
            result["warmed_timings"][arm].append(dict(seconds=seconds, valid=valid))
        save()
    result["summary"] = summarize(result["rows"])
    result["status"] = result["summary"]["status"]
    timing_valid = all(r["valid"] for rows in result["warmed_timings"].values() for r in rows)
    result["timing_valid"] = timing_valid
    if not timing_valid:
        result["status"] = "INVALID_TIMING_ARM"
    result["timing_median_seconds"] = {arm:statistics.median(r["seconds"] for r in rows)
        for arm, rows in result["warmed_timings"].items()}
    result["candidate_over_baseline_time"] = result["timing_median_seconds"]["candidate"]/result["timing_median_seconds"]["baseline"]
    result["source_mismatches"] = [p for p, h in sources.items()
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h or
           hashlib.sha256((output/"source"/p).read_bytes()).hexdigest() != h]
    if result["source_mismatches"]:
        result["status"] = "SOURCE_DRIFT"
    result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    result["active"] = None
    save()
    return 0 if result["status"] in (PASS, "MATERIAL_DIFFERENCE", "INVALID_ARM", "INVALID_TIMING_ARM") else 1


def execute(campaign, output, model=None, budget_seconds=10800.):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="RUNNING", plan=str(PLAN), models=[], diagnostic_used_seconds=0.,
        max_seconds=budget_seconds, canonical_admission=False, posterior_sampling_complete=False)
    master.save(output/"result.json", state)
    for name in ((model,) if model else master.MODELS):
        ledger, left = master.campaign_remaining(campaign/"budget.json")
        timeout = available_timeout(ledger, left, budget_seconds-state["diagnostic_used_seconds"])
        if timeout <= 0:
            state["status"] = "BUDGET_OR_ATTEMPT_LIMIT_REACHED"
            break
        child_output = output/name
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--model", name,
            "--campaign", str(campaign), "--output", str(child_output)]
        entry = dict(model=name, output=str(child_output), command=command, status="RUNNING",
            started_utc=datetime.now(timezone.utc).isoformat(), reserved_seconds=timeout)
        ledger["attempts"].append(entry)
        ledger["next_action"] = "running epsilon=1, Sinkhorn=5 diagnostic: "+name
        master.save(campaign/"budget.json", ledger)
        state["active_model"] = name
        master.save(output/"result.json", state)
        start = time.monotonic()
        try:
            with (output/(name+".log")).open("w") as log:
                child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = child.returncode
            result = read(child_output/"result.json") if (child_output/"result.json").is_file() else {}
            status = result.get("status", "MISSING_RESULT")
            if code != 0:
                status = "WORKER_FAILURE"
        except subprocess.TimeoutExpired:
            code, status = None, "TIMEOUT"
        seconds = time.monotonic()-start
        entry.update(status=status, exit_code=code, wall_seconds=seconds,
            result=str(child_output/"result.json"))
        ledger["used_seconds"] = sum(float(a.get("wall_seconds", a.get("reserved_seconds", 0))) for a in ledger["attempts"])
        ledger["remaining_seconds"] = max(0., ledger["authorized_gpu_wall_seconds"]-ledger["used_seconds"])
        ledger["next_action"] = "complete five-model OT materiality diagnostic"
        master.save(campaign/"budget.json", ledger)
        state["diagnostic_used_seconds"] += seconds
        state["models"].append(dict(model=name, status=status, returncode=code,
            seconds=seconds, result=entry["result"]))
        master.save(output/"result.json", state)
        if result.get("source_mismatches") if code is not None else False:
            state["status"] = "SOURCE_DRIFT"
            break
    if state["status"] == "RUNNING":
        state["status"] = "COMPARISON_COMPLETE" if all(r["returncode"] == 0 for r in state["models"]) else "COMPARISON_INCOMPLETE"
    state["active_model"] = None
    master.save(output/"result.json", state)
    return 0 if state["status"] == "COMPARISON_COMPLETE" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--model", choices=master.MODELS)
    args = parser.parse_args()
    if args.mode == "worker" and not args.model:
        parser.error("worker requires a model")
    try:
        sys.exit(worker(args.model, args.campaign.resolve(), args.output.resolve())
            if args.mode == "worker" else execute(args.campaign.resolve(), args.output.resolve(), args.model))
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"failure.txt").write_text(traceback.format_exc())
        raise
