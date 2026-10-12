#!/usr/bin/env python3
"""Cost diagnostics for explicit frozen cohorts using an immutable library.

This is research orchestration, not a tuner. Short trajectories never qualify
members, select a policy, or establish posterior performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def planned_rows(design_paths):
    rows = []
    seen = set()
    for path in design_paths:
        design = read(path)
        key = (design["target"], design["method"], design["seed"])
        if key in seen:
            raise ValueError("duplicate price cell")
        seen.add(key)
        if design.get("schema") != "bayesfilter.modern_frozen_full_model_design.v1":
            raise ValueError("unsupported full-model design")
        pairs = [(length, epsilon) for length, values in design["search"]["epsilon_by_l"]
                 for epsilon in values]
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError("distinct nonempty root cohort required")
        rows.append(dict(target=key[0], method=key[1], seed=key[2],
            design_file=str(Path(path).resolve()), design_sha256=digest(path),
            status="not_started", roots=[dict(index=i, leapfrog_steps=l, epsilon=e,
                status="not_started", calls=[]) for i, (l, e) in enumerate(pairs)]))
    return rows


def cost_envelope(design, priced, *, load_multiplier, overhead_seconds, trial_overhead_seconds):
    """Planning envelope, not a probabilistic upper bound or measured runtime.

    Every reserved repair receives its parent's rate times a count-based
    expansion factor. Rejected roots stay priced. Looks reuse trials within
    each stage, but verification is a separate stream with its own total.
    """
    for number in (load_multiplier, overhead_seconds, trial_overhead_seconds):
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError("invalid planning factor")
    if load_multiplier < 1:
        raise ValueError("load multiplier must be at least one")
    policy = design["execution"]["acceptance_policy"]
    search = design["search"]
    counts = [min(policy["base_repetitions"] * rung, policy["max_repetitions"])
              for rung in search["evidence_rungs"]]
    if not counts or counts[0] != policy["base_repetitions"] or sorted(set(counts)) != counts:
        raise ValueError("distinct increasing scheduled repetitions required")
    expected = [(l, e) for l, values in search["epsilon_by_l"] for e in values]
    if [(row["leapfrog_steps"], row["epsilon"]) for row in priced["roots"]] != expected:
        raise ValueError("prices do not cover the exact declared cohort")
    repairs = design["max_repairs_per_family"]
    if type(repairs) is not int or repairs < 0:
        raise ValueError("invalid family repair allowance")
    if search["max_candidates"] != len(expected) * (repairs + 1):
        raise ValueError("cost design requires the complete exact candidate allowance")
    if policy["max_candidates"] < search["max_candidates"]:
        raise ValueError("statistical allowance misses reserved candidates")
    posterior = design.get("posterior")
    if posterior is None:
        raise ValueError("posterior settings required for an all-member cost")
    sequential = posterior["sequential"]
    posterior_counts = dict(
        minimum=sequential["warmup_min_results"] + sequential["retained_min_results"],
        maximum=sequential["warmup_max_results"] + sequential["retained_max_results"])
    rates, rows = [], []
    for row in priced["roots"]:
        calls = row["calls"]
        if (len(calls) != 2 or any(c.get("status") != "returned" for c in calls)
                or any(c["transitions"] <= 0 or not math.isfinite(c["elapsed_seconds"])
                       or c["elapsed_seconds"] <= 0 for c in calls)):
            raise ValueError("two complete positive native prices required for every root")
        # Cold compilation is already paid here; retaining max(first,repeat)
        # deliberately overprices it again in case another process retraces.
        rate = max(c["elapsed_seconds"] / c["transitions"] for c in calls)
        # Duration-preserving lower-epsilon repairs can increase step count.
        # Each native step also evaluates status; ceil+1 covers discretization.
        length = row["leapfrog_steps"]
        upper_count, expansion = length, 1.
        for _ in range(repairs):
            upper_count = math.ceil((upper_count + 1) * design["repair_factor"])
            expansion += (upper_count + 1) / (length + 1)
        rates.append(rate * expansion)
        rows.append(dict(index=row["index"], root_seconds_per_transition=rate,
                         root_and_repair_rate_factor=expansion,
                         numerical_veto=any(c.get("hard_vetoes") for c in calls)))
    aggregate_rate = sum(rates) * load_multiplier
    # All chains are already included in the wall price; do not multiply by 4.
    trial_transitions = policy["trial_num_results"] + design["execution"]["num_warmup_steps"]
    qualification = {str(n): aggregate_rate * 2 * n * trial_transitions
        + trial_overhead_seconds * 2 * n * search["max_candidates"] for n in counts}
    posterior_cost = {key: aggregate_rate * n for key, n in posterior_counts.items()}
    return dict(load_multiplier=load_multiplier, overhead_seconds=overhead_seconds,
        trial_overhead_seconds=trial_overhead_seconds,
        scheduled_repetitions=counts, stage_incremental_repetitions=[counts[0]] + [
            b-a for a, b in zip(counts, counts[1:])],
        statistical_candidate_allowance=policy["max_candidates"],
        reserved_member_count=search["max_candidates"],
        qualification_seconds_by_final_rung=qualification,
        all_member_posterior_seconds=posterior_cost,
        minimum_path_seconds=overhead_seconds + qualification[str(counts[0])] + posterior_cost["minimum"],
        maximum_count_scenario_seconds=overhead_seconds + qualification[str(counts[-1])] + posterior_cost["maximum"],
        root_rates=rows, probabilistic_upper_bound=False, power_established=False,
        actual_candidate_membership=None, method_ranking=None)


def finalize(output, *, status, elapsed):
    path = Path(output) / "result.json"
    report = read(path)
    report.update(status=status, supervisor_elapsed_seconds=elapsed,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    for cell in report["rows"]:
        if cell["status"] in {"running", "preparing", "not_started"}:
            cell["status"] += "_" + status
        for root in cell["roots"]:
            if root["status"] in {"running", "not_started"}:
                root["status"] += "_" + status
            for call in root["calls"]:
                if call["status"] == "running":
                    call["status"] = "interrupted_" + status
    write(path, report)
    return report


def worker(args):
    sys.path.insert(0, str(args.source_root))
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.runtime.execution_budget import execution_budget, require_execution_budget
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.inference import HMCCandidateExecutionConfig, HMCControllerConfig
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord
    from bayesfilter.testing.inference_validation.modern_full_model_consumer import full_model_binding
    from bayesfilter.testing.inference_validation.modern_tuning_consumer import (
        validate_frozen_search, frozen_search_evidence_preflight, require_attainable_qualification)
    from scripts.run_modern_hmc_frozen_full_model import execution_source_paths, posterior_factory

    if not Path(sys.modules[full_model_binding.__module__].__file__).resolve().is_relative_to(args.source_root):
        raise ValueError("numerical implementation did not load from the declared source")
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    report = read(args.output / "result.json")
    started = time.monotonic()
    report.update(status="running", memory_policy=memory,
        tf_version=tf.__version__, tfp_version=tfp.__version__,
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    write(args.output / "result.json", report)
    with execution_budget(check=lambda: time.monotonic()-started < args.max_wall_seconds,
                          replace_wall_time_limit=True):
        for cell_index, cell in enumerate(report["rows"]):
            require_execution_budget()
            tick = time.monotonic()
            cell["status"] = "preparing"
            write(args.output / "result.json", report)
            design = read(cell["design_file"])
            if digest(cell["design_file"]) != cell["design_sha256"]:
                raise ValueError("price design changed")
            execution = HMCCandidateExecutionConfig.from_payload(design["execution"])
            search = HMCControllerConfig.from_payload(design["search"])
            if not execution.use_xla:
                raise ValueError("GPU prices require XLA")
            require_attainable_qualification(frozen_search_evidence_preflight(execution, search))
            posterior_factory(design["posterior"], target=design["target"], use_xla=True)
            binding, _, metadata = full_model_binding(target=design["target"],
                geometry_directory=design["geometry_directory"], method=design["method"],
                seed=design["seed"], execution=execution, epsilon_domain=design["epsilon_domain"],
                repair_factor=design["repair_factor"], max_repairs_per_family=design["max_repairs_per_family"],
                source_paths=execution_source_paths())
            validate_frozen_search(binding, search)
            cell.update(status="running", binding_hash=binding.binding_hash,
                initial_active_state_sha256=DurableTensorCheckpoint.tensor_hash(binding.initial_active_state),
                data_sha256=metadata["data_sha256"], geometry_sha256=metadata["geometry_sha256"],
                source_sha256=binding._spec["source_closure"], runtime=binding._runtime,
                preparation_seconds=time.monotonic()-tick)
            write(args.output / "result.json", report)
            with DurableTensorCheckpoint(args.output / f"cell-{cell_index:02d}",
                    dict(binding_hash=binding.binding_hash, role="cost_only")) as store:
                for row in cell["roots"]:
                    candidate = HMCTuningCandidateRecord.create(binding.scope,
                        leapfrog_steps=row["leapfrog_steps"], epsilon=row["epsilon"],
                        creation_ordinal=row["index"]+1)
                    row.update(status="running", candidate_id=candidate.candidate_id,
                               transition_spec=candidate.transition_spec.payload())
                    for repeat in range(2):
                        require_execution_budget()
                        seed = (20261012, 800000 + 10000*cell_index + 2*row["index"] + repeat)
                        call = dict(status="running", seed=seed, transitions=64, repeat=repeat)
                        row["calls"].append(call)
                        write(args.output / "result.json", report)
                        call_tick = time.monotonic()
                        def compute():
                            native = binding._run(candidate, binding.initial_active_state, 64, seed)
                            return dict(samples=native.samples, trace=native.trace, metadata=native.metadata)
                        value = store.run(f"root-{row['index']:03d}-repeat-{repeat}",
                            dict(candidate=candidate.payload(), seed=seed, transitions=64,
                                 start_sha256=cell["initial_active_state_sha256"]), compute)
                        vetoes = binding.health_failures(binding.initial_active_state, value["samples"], value["trace"])
                        call.update(status="returned", elapsed_seconds=time.monotonic()-call_tick,
                            hard_vetoes=vetoes, native=value["metadata"],
                            mean_acceptance=float(tf.reduce_mean(tf.exp(tf.minimum(value["trace"]["log_accept_ratio"], 0.)))),
                            sample_device=value["samples"].device)
                        write(args.output / "result.json", report)
                    row["status"] = "priced_with_veto" if any(c["hard_vetoes"] for c in row["calls"]) else "priced"
            binding.validate()
            if digest(cell["design_file"]) != cell["design_sha256"]:
                raise ValueError("price design changed")
            cell.update(status="completed", elapsed_seconds=time.monotonic()-tick)
            write(args.output / "result.json", report)
    report.update(status="completed", elapsed_seconds=time.monotonic()-started,
                  gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"))
    write(args.output / "result.json", report)
    return 0


def supervise(args):
    rows = planned_rows(args.design)
    provenance = read(args.source_root / ".validation-source.json")
    relative = "bayesfilter/ops/_symmetric_sylvester_ops.so"
    expected = provenance["compiled_operator_sha256"]
    expected = expected[relative] if isinstance(expected, dict) else expected
    if digest(args.source_root / relative) != expected:
        raise ValueError("compiled operator changed from recorded numerical snapshot")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "result.json", dict(status="starting", rows=rows,
        command=[sys.executable, *sys.argv], environment=sys.executable,
        plan=args.plan, started_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(args.source_root), numerical_source=provenance,
        compiled_operator_sha256={relative:expected},
        driver_sha256=digest(__file__), gpu_trust_basis="trusted_escalated_gpu_execution",
        memory_growth_environment=os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        max_wall_seconds=args.max_wall_seconds, qualification_authority=False,
        posterior_authority=False, method_ranking=None, release_qualified=False))
    command = [sys.executable, str(Path(__file__).resolve()), "--source-root", str(args.source_root),
        "--design", *(str(p) for p in args.design), "--output", str(args.output), "--plan", args.plan,
        "--max-wall-seconds", str(args.max_wall_seconds), "--worker"]
    started = time.monotonic()
    status = None
    with subprocess.Popen(command, cwd=args.source_root, start_new_session=True) as process:
        try:
            code = process.wait(timeout=args.max_wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "supervisor_budget_exhausted"
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "supervisor_interrupted"
    current = read(args.output / "result.json")["status"]
    status = status or ("completed" if code == 0 and current == "completed" else "repair_required")
    finalize(args.output, status=status, elapsed=time.monotonic()-started)
    return 0 if status == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_root, args.output = args.source_root.resolve(), args.output.resolve()
    args.design = [p.resolve() for p in args.design]
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("positive finite allocation required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("set memory growth before framework import")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
