#!/usr/bin/env python3
"""Cost diagnostics for explicit frozen cohorts using an immutable library.

This is research orchestration, not a tuner. Short trajectories never qualify
members, select a policy, or establish posterior performance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def pricing_seed(namespace, cell_index, root_index, repeat):
    """Keep the historical stream layout; require a new namespace for new work."""
    if (type(namespace) is not int or not 0 <= namespace < 2**31
            or type(cell_index) is not int or cell_index < 0
            or type(root_index) is not int or root_index < 0
            or type(repeat) is not int or repeat not in (0, 1)):
        raise ValueError("invalid pricing stream scope")
    second = 800000 + 10000*cell_index + 2*root_index + repeat
    if second >= 2**31:
        raise ValueError("pricing stream exceeds stateless seed range")
    return (namespace, second)


def worker_command(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--source-root", str(args.source_root),
        "--design", *(str(p) for p in args.design), "--output", str(args.output), "--plan", args.plan,
        "--max-wall-seconds", str(args.max_wall_seconds),
        "--seed-namespace", str(args.seed_namespace)]
    if getattr(args, "resume_prices", None) is not None:
        command.extend(["--resume-prices", str(args.resume_prices)])
    return command + ["--worker"]


def attempt_receipt(prices, directory):
    """Inclusive charge and driver identity for one settled price supervisor."""
    elapsed = prices.get("current_supervisor_elapsed_seconds",
                         prices.get("supervisor_elapsed_seconds"))
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0
            or prices.get("status") not in {"completed", "supervisor_budget_exhausted"}):
        raise ValueError("pricing attempt needs a supported terminal inclusive charge")
    return dict(output_directory=str(Path(directory).resolve()),
        driver_path=str(Path(prices["command"][1]).resolve()),
        driver_sha256=prices["driver_sha256"], seed_namespace=prices["pricing_seed_namespace"],
        status=prices["status"], supervisor_elapsed_seconds=elapsed)


def audit_attempts(prices, directory):
    """Check the original receipt and at most one cost-only timeout recovery."""
    from scripts.audit_mams_full_model_pricing import check_hashes
    current = attempt_receipt(prices, directory)
    expected = [current]
    if prices.get("output_directory", current["output_directory"]) != current["output_directory"]:
        raise ValueError("pricing output directory changed")
    if prices.get("resume_of"):
        prior_path = Path(prices["resume_of"]).resolve()
        prior = read(prior_path)
        if prior.get("resume_of") or prior.get("status") != "supervisor_budget_exhausted":
            raise ValueError("only one recovery of an original price timeout is supported")
        if digest(prior_path) != prices.get("resume_sha256"):
            raise ValueError("original terminal price receipt changed")
        expected = audit_attempts(prior, prior_path.parent) + expected
        if expected[0]["seed_namespace"] == current["seed_namespace"]:
            raise ValueError("recovery pricing namespace was reused")
    if prices.get("pricing_attempts", expected) != expected:
        raise ValueError("pricing attempt ledger disagrees with terminal receipts")
    total = math.fsum(a["supervisor_elapsed_seconds"] for a in expected)
    if not math.isclose(prices["supervisor_elapsed_seconds"], total, abs_tol=1e-6, rel_tol=0):
        raise ValueError("pricing cumulative charge does not reconcile")
    for item in expected:
        check_hashes({item["driver_path"]: item["driver_sha256"]})
    return expected


def resume_rows(prior, design_paths, *, seed_namespace, prior_directory):
    """Reuse only returned checkpoints from a timed-out complete design run."""
    if prior.get("status") != "supervisor_budget_exhausted":
        raise ValueError("only a terminal budget-exhausted pricing attempt may resume")
    if prior.get("resume_of"):
        raise ValueError("a second timeout recovery needs a revised campaign plan")
    audit_attempts(prior, prior_directory)
    if prior.get("pricing_seed_namespace") == seed_namespace:
        raise ValueError("pricing resume requires a fresh random namespace")
    old_rows = prior.get("rows")
    new_rows = planned_rows(design_paths)
    if not isinstance(old_rows, list) or len(old_rows) != len(new_rows):
        raise ValueError("resume must preserve the complete ordered method inventory")
    checkpoint_base = Path(prior_directory).resolve()
    from scripts.audit_mams_full_model_pricing import check_hashes
    source = Path(prior["source_root"])
    if read(source / ".validation-source.json") != prior["numerical_source"]:
        raise ValueError("prior numerical source provenance changed")
    check_hashes({str(source / name): value
                  for name, value in prior["compiled_operator_sha256"].items()})
    memory = prior.get("memory_policy", {})
    if (prior.get("gpu_trust_basis") != "trusted_escalated_gpu_execution"
            or prior.get("memory_growth_environment", "").lower() != "true"
            or memory.get("all_physical_devices_memory_growth") is not True
            or memory.get("configured_before_logical_device_initialization") is not True):
        raise ValueError("prior pricing GPU memory policy is ineligible")
    prior_seeds = set()
    for index, (old, new) in enumerate(zip(old_rows, new_rows)):
        for key in ("target", "method", "seed", "design_file", "design_sha256"):
            if old.get(key) != new.get(key):
                raise ValueError("pricing resume changes a design cell: " + key)
        if len(old.get("roots", [])) != len(new["roots"]):
            raise ValueError("pricing resume changes the root count")
        row = deepcopy(old)
        check_hashes(old.get("source_sha256", {}))
        geometry_dir = Path(read(new["design_file"])["geometry_directory"])
        if "binding_hash" in old:
            if (digest(geometry_dir / "geometry.json") != old["geometry_sha256"]
                    or read(geometry_dir / "design.json")["metadata"]["data_sha256"] != old["data_sha256"]
                    or not old.get("source_sha256")):
                raise ValueError("prior pricing source, data or geometry changed")
        row["resume_previous_status"] = old.get("status")
        row["status"] = "resume_pending"
        roots = []
        for prior_root, planned_root in zip(old["roots"], new["roots"]):
            for key in ("index", "leapfrog_steps", "epsilon"):
                if prior_root.get(key) != planned_root[key]:
                    raise ValueError("pricing resume changes a candidate root")
            for call in prior_root.get("calls", []):
                expected_seed = pricing_seed(prior["pricing_seed_namespace"], index,
                                             prior_root["index"], call["repeat"])
                seed = tuple(call.get("seed", ()))
                if seed != expected_seed or seed in prior_seeds:
                    raise ValueError("prior attempted price stream differs or was reused")
                prior_seeds.add(seed)
            returned = [call for call in prior_root.get("calls", [])
                        if call.get("status") == "returned"]
            repeats = [call.get("repeat") for call in returned]
            if len(set(repeats)) != len(repeats) or any(x not in (0, 1) for x in repeats):
                raise ValueError("prior price calls have invalid repeat identities")
            root = deepcopy(prior_root)
            root["prior_attempts"] = deepcopy(prior_root.get("calls", []))
            root["calls"] = []
            checkpoint_dir = checkpoint_base / f"cell-{index:02d}"
            dimension = len(read(geometry_dir / "geometry.json")["center"])
            for call in returned:
                reused = deepcopy(call)
                location = Path(reused.get("checkpoint_directory", checkpoint_dir)).resolve()
                if location != checkpoint_dir:
                    raise ValueError("prior price checkpoint is outside its original cell")
                from scripts.report_modern_hmc_frozen_prices import audit_call
                audit_call(location, old, prior_root, reused, dimension=dimension)
                reused["checkpoint_directory"] = str(location)
                root["calls"].append(reused)
            root["resume_missing_repeats"] = [repeat for repeat in (0, 1) if repeat not in repeats]
            root["status"] = ("resume_pending" if root["resume_missing_repeats"] else
                "priced_with_veto" if any(c.get("hard_vetoes") for c in returned) else "priced")
            roots.append(root)
        row["roots"] = roots
        row["resume_binding_identity"] = {key: old[key] for key in (
            "binding_hash", "initial_active_state_sha256", "data_sha256", "geometry_sha256",
            "source_sha256") if key in old}
        row["status"] = "completed" if all(not r["resume_missing_repeats"] for r in roots) else "resume_pending"
        new_rows[index] = row
    return new_rows


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
    attempts = list(report.get("pricing_attempts", []))
    if "command" in report:
        current_attempt = dict(output_directory=str(Path(output).resolve()),
            driver_path=str(Path(report["command"][1]).resolve()),
            driver_sha256=report["driver_sha256"], seed_namespace=report["pricing_seed_namespace"],
            status=status, supervisor_elapsed_seconds=elapsed)
        attempts.append(current_attempt)
    total = (math.fsum(float(item["supervisor_elapsed_seconds"]) for item in attempts)
             if attempts else elapsed)
    report.update(status=status, output_directory=str(Path(output).resolve()),
        current_supervisor_elapsed_seconds=elapsed, pricing_attempts=attempts,
        supervisor_elapsed_seconds=total,
        finished_utc=datetime.now(timezone.utc).isoformat())
    for cell in report["rows"]:
        if cell["status"] in {"running", "preparing", "not_started", "resume_pending"}:
            cell["status"] += "_" + status
        for root in cell["roots"]:
            if root["status"] in {"running", "not_started", "resume_pending"}:
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
            if cell["status"] == "completed":
                continue
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
            active_sha = DurableTensorCheckpoint.tensor_hash(binding.initial_active_state)
            resume_identity = cell.get("resume_binding_identity", {})
            actual_identity = dict(binding_hash=binding.binding_hash,
                initial_active_state_sha256=active_sha, data_sha256=metadata["data_sha256"],
                geometry_sha256=metadata["geometry_sha256"], source_sha256=binding._spec["source_closure"])
            if resume_identity and resume_identity != actual_identity:
                raise ValueError("price resume source, start, geometry or binding identity changed")
            cell.update(status="running", **actual_identity, runtime=binding._runtime,
                preparation_seconds=time.monotonic()-tick)
            write(args.output / "result.json", report)
            with DurableTensorCheckpoint(args.output / f"cell-{cell_index:02d}",
                    dict(binding_hash=binding.binding_hash, role="cost_only")) as store:
                for row in cell["roots"]:
                    candidate = HMCTuningCandidateRecord.create(binding.scope,
                        leapfrog_steps=row["leapfrog_steps"], epsilon=row["epsilon"],
                        creation_ordinal=row["index"]+1)
                    if row.get("candidate_id") not in (None, candidate.candidate_id):
                        raise ValueError("price resume candidate identity changed")
                    row.update(status="running", candidate_id=candidate.candidate_id,
                               transition_spec=candidate.transition_spec.payload())
                    completed_repeats = {call.get("repeat") for call in row["calls"]
                                         if call.get("status") == "returned"}
                    for repeat in range(2):
                        if repeat in completed_repeats:
                            continue
                        require_execution_budget()
                        seed = pricing_seed(args.seed_namespace, cell_index, row["index"], repeat)
                        call = dict(status="running", seed=seed, transitions=64, repeat=repeat,
                                    checkpoint_directory=str(args.output / f"cell-{cell_index:02d}"))
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
                    returned = [c for c in row["calls"] if c.get("status") == "returned"]
                    if len(returned) != 2 or {c["repeat"] for c in returned} != {0, 1}:
                        raise ValueError("pricing resume did not produce both successful calls")
                    row["calls"].sort(key=lambda c: c["repeat"])
                    row.pop("resume_missing_repeats", None)
                    row["status"] = "priced_with_veto" if any(c["hard_vetoes"] for c in returned) else "priced"
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
    prior_path = getattr(args, "resume_prices", None)
    prior = read(prior_path) if prior_path is not None else None
    rows = (planned_rows(args.design) if prior is None else
        resume_rows(prior, args.design, seed_namespace=args.seed_namespace,
                    prior_directory=args.resume_prices.parent))
    seeds = []
    prior_attempts = []
    if prior is not None:
        prior_attempts = audit_attempts(prior, prior_path.parent)
        seeds.extend(tuple(call["seed"]) for cell in rows for root in cell["roots"]
                     for call in root["calls"] if call.get("status") == "returned")
    for i, cell in enumerate(rows):
        for root in cell["roots"]:
            completed = {call.get("repeat") for call in root["calls"]
                         if call.get("status") == "returned"}
            missing = root.get("resume_missing_repeats", [0, 1])
            seeds.extend(pricing_seed(args.seed_namespace, i, root["index"], repeat)
                         for repeat in missing if repeat not in completed)
    runtime_seeds = set()
    for path in args.design:
        design = read(path)
        if design.get("seed_namespace") == args.seed_namespace:
            raise ValueError("pricing requires a separate random namespace")
        runtime_seeds.update(tuple(design[key]["seed"]) for key in ("execution", "posterior"))
    if len(set(seeds)) != len(seeds) or set(seeds) & runtime_seeds:
        raise ValueError("pricing streams overlap each other or a runtime base seed")
    provenance = read(args.source_root / ".validation-source.json")
    if prior is not None and (prior.get("source_root") != str(args.source_root)
                             or prior.get("numerical_source") != provenance):
        raise ValueError("pricing resume changed the immutable numerical source")
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
        pricing_seed_namespace=args.seed_namespace,
        pricing_attempts=prior_attempts, resume_of=None if prior_path is None
            else str(prior_path.resolve()),
        resume_sha256=None if prior is None else digest(prior_path),
        max_wall_seconds=args.max_wall_seconds, qualification_authority=False,
        posterior_authority=False, method_ranking=None, release_qualified=False))
    command = worker_command(args)
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
    parser.add_argument("--seed-namespace", type=int, default=20261012,
                        help="independent pricing namespace; default preserves historical streams")
    parser.add_argument("--resume-prices", type=Path,
                        help="terminal budget-exhausted price result to resume without rerunning returned calls")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_root, args.output = args.source_root.resolve(), args.output.resolve()
    if args.resume_prices is not None:
        args.resume_prices = args.resume_prices.resolve()
    args.design = [p.resolve() for p in args.design]
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("positive finite allocation required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("set memory growth before framework import")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
