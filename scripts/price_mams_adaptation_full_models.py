#!/usr/bin/env python3
"""Cost-only discarded MAMS adaptation on four full nonlinear targets.

No tuner, reference, posterior or method-ranking authority is issued here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_mams_adaptation import (
    execute_matrix, finalize_interrupted_matrix, source_revision, write_json)

TARGETS = ("NL-PP-SGQF", "NL-PP-UKF", "NL-SIR", "NL-STR")
PLAN = "docs/plans/bayesfilter-mams-d4r4a-full-model-pricing-2026-10-12.md"
MODEL_SOURCES = (
    "bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py",
    "bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py",
    "bayesfilter/testing/sir_filter_neutra_target_design_tf.py",
    "bayesfilter/testing/structural_ukf_neutra_target_design_tf.py",
)


def finalize_prices(output, targets, *, status, error=None):
    """Keep priced and unfinished policies explicit after any worker exit."""
    aggregate = finalize_interrupted_matrix(output, targets, status=status, error=error)
    for row in aggregate["rows"]:
        directory = output / row["target"]
        pilot_path = directory / "pilot/pilot-progress.json"
        if pilot_path.exists():
            pilot = json.loads(pilot_path.read_text())
            row.update(pilot_status=pilot["status"],
                pilot_actual_work=pilot["actual_gradient_work"],
                pilot_reserved_work=pilot["reserved_gradient_work"],
                pilot_hypotheses_before_pricing=pilot["candidate_rows"])
        for member in row["rows"]:
            if member["status"] in ("price_running", "price_not_started"):
                member["status"] += "_" + status
        if directory.exists():
            write_json(directory / "result.json", row)
    aggregate.update(tuning_artifact_authority=False, posterior_convergence_authority=False)
    write_json(output / "result.json", aggregate)
    return aggregate


def geometry_directory(root: Path, target: str) -> Path:
    if target not in TARGETS:
        raise ValueError("unsupported full-model target")
    parent = "full-model-geometry-02-sir" if target == "NL-SIR" else "full-model-geometry-01"
    return root / parent / target.lower()


def prepare(target: str, root: Path, seed: int, *, use_xla: bool):
    """Reissue a current-source MAMS binding; old local factor is only a seed."""
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    from bayesfilter.testing.inference_validation.modern_prepared_nonlinear import prepared_binding
    return prepared_binding(target, geometry_directory(root, target),
        geometry_kind="local", seed=seed, use_xla=use_xla, source_paths=(__file__,),
        geometry_source_policy="warm_start_only",
        transition_profile=HMCTransitionProfile("mams", "uniform_duration"),
        epsilon_domain=(.001, 4.), max_repairs_per_family=1)


def pilot_config(binding, seed: int):
    from bayesfilter.inference.hmc_mams_trajectory_proposals import MAMSPilotConfig
    dimension = int(binding.initial_active_state.shape[-1])
    return MAMSPilotConfig(initial_epsilon=.2,
        initial_mean_duration=math.sqrt(dimension), epsilon_domain=(.001, 4.),
        max_steps_cap=128, epsilon_adaptation_windows=6, probe_transitions=4,
        fixed_settling_transitions=32, fixed_window_transitions=64,
        fixed_window_count=2, max_autocorrelation_lag=16,
        target_accept_probability=.9, log_epsilon_gain=math.log(2),
        source_length_coefficient=.3, window_agreement_factor=3.,
        duration_factors=(1., 2.), epsilon_factors=(1.,),
        max_requested_duration=8*math.sqrt(dimension), max_candidates=4,
        max_pilot_gradient_work=2_000_000, max_pilot_wall_seconds=7200.,
        frozen_coordinate_scale=(1.,)*dimension,
        frozen_projection_directions=(tuple(1. for _ in range(dimension)),),
        seed=(20261012, seed+200))


def run_case(target, output, seed, deadline, *, use_xla, source_hashes, geometry_root):
    import tensorflow as tf
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord
    from bayesfilter.inference.hmc_mams_trajectory_proposals import (
        MAMSPilotBudgetExceeded, MAMSPilotUnavailable,
        propose_mams_trajectory_candidates)
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.runtime.execution_budget import execution_budget, require_execution_budget, ExecutionBudgetExceeded

    output.mkdir()
    started = time.monotonic()
    result = dict(target=target, status="preparing", rows=[],
                  tuning_artifact_authority=False, posterior_convergence_authority=False,
                  ranking_supported=False, release_qualified=False)
    write_json(output / "progress.json", result)
    try:
        binding, fixture, metadata = prepare(target, geometry_root, seed, use_xla=use_xla)
        binding.validate()
        if not bool(tf.reduce_all(tf.math.is_finite(binding.initial_active_state))):
            raise ValueError("prepared starts are nonfinite")
        native, score, status = fixture.adapter.log_prob_and_grad_status(metadata["start_positions"])
        if not (bool(tf.reduce_all(tf.math.is_finite(native)))
                and bool(tf.reduce_all(tf.math.is_finite(score)))
                and bool(tf.reduce_all(status["status_code"] == 0))
                and bool(tf.reduce_all(status["valid_pre_regularized_score"]))
                and bool(tf.reduce_all(status["floor_count_value"] == 0))):
            raise ValueError("current full-model target is invalid at prepared starts")
        pilot = pilot_config(binding, seed)
        write_json(output / "design.json", dict(target=target, data_sha256=metadata["data_sha256"],
            horizon=metadata["horizon"], parameter_dim=metadata["parameter_dim"],
            geometry_sha256=metadata["geometry_sha256"],
            geometry_design_sha256=metadata["geometry_design_sha256"],
            geometry_source_policy=metadata["geometry_source_policy"],
            geometry_source_changes=metadata["geometry_source_changes"],
            global_coverage_established=False, binding_hash=binding.binding_hash,
            pilot=asdict(pilot), source_hashes=source_hashes, seed=seed,
            measurement_role="cost_only_no_qualification"))
        result["status"] = "pilot"
        write_json(output / "progress.json", result)
        with execution_budget(check=lambda: time.monotonic() < deadline,
                              replace_wall_time_limit=True):
            proposals = propose_mams_trajectory_candidates(binding, pilot,
                                                           output_dir=output / "pilot")
            result.update(pilot_status="completed", pilot_actual_work=proposals["actual_gradient_work"],
                          pilot_reserved_work=proposals["reserved_gradient_work"],
                          omitted_candidates=proposals["omitted_candidates"])
            if not proposals["candidates"]:
                result["status"] = "no_frozen_proposals"
                return result
            for index, spec in enumerate(proposals["candidates"]):
                result["rows"].append(dict(index=index, frozen_spec=spec,
                                           status="price_not_started"))
            result["status"] = "pricing"
            write_json(output / "progress.json", result)
            with DurableTensorCheckpoint(output / "prices", dict(
                    binding_hash=binding.binding_hash, role="cost_only")) as store:
                for row in result["rows"]:
                    require_execution_budget()
                    index, spec = row["index"], row["frozen_spec"]
                    candidate = HMCTuningCandidateRecord.create(binding.scope,
                        leapfrog_steps=spec["max_steps"], epsilon=spec["epsilon"],
                        creation_ordinal=index+1)
                    if candidate.transition_spec.payload() != spec:
                        raise ValueError("priced frozen spec differs from pilot proposal")
                    seed_pair = (20261012, seed+1000+index)
                    row.update(status="price_running", candidate_id=candidate.candidate_id,
                               seed=list(seed_pair))
                    write_json(output / "progress.json", result)
                    tick = time.monotonic()
                    def compute():
                        native_result = binding._run(candidate, binding.initial_active_state,
                                                     32, seed_pair)
                        return dict(samples=native_result.samples, trace=native_result.trace,
                                    metadata=native_result.metadata)
                    try:
                        priced = store.run(f"candidate-{index:02d}", dict(
                            candidate=candidate.payload(), seed=seed_pair, transitions=32,
                            start_sha256=store.tensor_hash(binding.initial_active_state)), compute)
                        vetoes = binding.health_failures(binding.initial_active_state,
                                                          priced["samples"], priced["trace"])
                        trace = priced["trace"]
                        acceptance = tf.reduce_mean(tf.exp(tf.minimum(
                            trace["log_accept_ratio"], 0.)))
                        row.update(status="candidate_veto" if vetoes else "healthy_short_path",
                            hard_vetoes=vetoes, metadata=priced["metadata"],
                            mean_acceptance=float(acceptance),
                            realized_steps_min=int(tf.reduce_min(trace["trajectory_steps"])),
                            realized_steps_max=int(tf.reduce_max(trace["trajectory_steps"])))
                    except ExecutionBudgetExceeded:
                        raise
                    except Exception as exc:
                        row.update(status="candidate_execution_error_unclassified",
                                   error_type=type(exc).__name__, error=str(exc))
                        row["elapsed_seconds"] = time.monotonic() - tick
                        write_json(output / "progress.json", result)
                        raise
                    row["elapsed_seconds"] = time.monotonic() - tick
                    write_json(output / "progress.json", result)
        result["status"] = "completed"
    except (MAMSPilotBudgetExceeded, MAMSPilotUnavailable, ExecutionBudgetExceeded) as exc:
        result.update(status="resource_or_pilot_incomplete", error_type=type(exc).__name__,
                      error=str(exc))
    except BaseException as exc:
        result.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "result.json", result)
    return result


def worker(args):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(command=[sys.executable, *sys.argv], plan=PLAN,
        environment=sys.executable, **source_revision(), seed=args.seed,
        targets=list(args.targets), started_utc=datetime.now(timezone.utc).isoformat(),
        max_wall_seconds=args.max_wall_seconds,
        gpu_intentionally_hidden=args.cpu_reference,
        status="starting", result_file=str(args.output / "result.json"))
    write_json(args.output / "manifest.json", manifest)
    try:
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not args.cpu_reference)
        paths = source_dependencies([__file__, ROOT / "scripts/validate_mams_adaptation.py",
            ROOT / "bayesfilter/testing/inference_validation/modern_prepared_nonlinear.py",
            *(ROOT / name for name in MODEL_SOURCES)], repository_root=ROOT)
        hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
        geometry_inputs = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for target in args.targets
            for path in (geometry_directory(args.geometry_root, target) / "design.json",
                         geometry_directory(args.geometry_root, target) / "geometry.json")}
        operators = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT / "bayesfilter/ops").glob("*.so")}
        manifest.update(status="running", source_hashes=hashes,
            geometry_root=str(args.geometry_root.resolve()), geometry_input_sha256=geometry_inputs,
            compiled_operator_sha256=operators, memory_policy=memory,
            tf_version=tf.__version__, tfp_version=tfp.__version__,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            use_xla=not args.cpu_reference,
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled())
        write_json(args.output / "manifest.json", manifest)
        def case_runner(target, output, seed, deadline, *, use_xla, source_hashes):
            for name, digest in geometry_inputs.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError("geometry input changed during pricing: " + name)
            return run_case(target, output, seed, deadline, use_xla=use_xla,
                            source_hashes=source_hashes, geometry_root=args.geometry_root)
        result = execute_matrix(args.targets, args.output, args.seed,
            started + args.max_wall_seconds, use_xla=not args.cpu_reference,
            source_hashes=hashes, case_runner=case_runner)
        manifest["status"] = result["status"]
        finalize_prices(args.output, args.targets, status=result["status"])
    except BaseException as exc:
        manifest.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        finalize_prices(args.output, args.targets, status="repair_required", error=exc)
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-started,
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 1


def supervise(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    grace = min(60., args.max_wall_seconds / 10.)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output", str(args.output.resolve()), "--geometry-root", str(args.geometry_root.resolve()),
        "--targets", *args.targets,
        "--seed", str(args.seed), "--max-wall-seconds", str(args.max_wall_seconds-grace)]
    if args.cpu_reference:
        command.append("--cpu-reference")
    started = time.monotonic()
    with subprocess.Popen(command, cwd=ROOT, start_new_session=True) as process:
        try:
            code = process.wait(timeout=args.max_wall_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            args.output.mkdir(parents=True, exist_ok=True)
            finalize_prices(args.output, args.targets,
                            status="supervisor_budget_exhausted", error=exc)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    args.output.mkdir(parents=True, exist_ok=True)
    if code and not (args.output / "result.json").exists():
        finalize_prices(args.output, args.targets, status="worker_failed")
    write_json(args.output / "supervisor.json", dict(command=command, exit_code=code,
        max_wall_seconds=args.max_wall_seconds, cooperative_grace_seconds=grace,
        elapsed_seconds=time.monotonic()-started))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", required=True, choices=TARGETS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-wall-seconds", required=True, type=float)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.targets)) != len(args.targets) or not 0 < args.max_wall_seconds <= 7200:
        raise ValueError("unique targets and a bounded positive allocation required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("GPU memory growth must be configured before framework import")
    if args.cpu_reference and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU reference must hide GPUs before framework import")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
