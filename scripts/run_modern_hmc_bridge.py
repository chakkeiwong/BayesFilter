#!/usr/bin/env python3
"""Run the bounded N5 public randomized-HMC bridge.

The bridge is an engineering diagnostic.  It runs the repository public
ordinary tuner on exact and nonlinear fixtures, records complete replicated
trial pricing, and deliberately does not issue a claim-bearing tuning or
posterior artifact.  CPU reference and GPU/XLA modes are explicit so that a
manifest cannot confuse a reference run with a GPU qualification run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


TARGETS = ("LGSSM", "NL-ACC", "NL-PP-UKF", "NL-PP-SGQF", "NL-SIR", "NL-STR")


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                allow_nan=False, default=str) + "\n")


def _tensor_sha256(tensor) -> str:
    """Hash fixture data at the artifact boundary using TensorFlow serialization."""
    import tensorflow as tf

    encoded = tf.io.serialize_tensor(tf.convert_to_tensor(tensor)).numpy()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_file(value) -> str | None:
    try:
        source = inspect.getsourcefile(value)
    except TypeError:
        source = inspect.getsourcefile(type(value))
    return str(Path(source).resolve()) if source else None


def _policy():
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy

    return HMCReplicatedAcceptancePolicy(
        trial_num_results=64,
        discarded_prefix=0,
        base_repetitions=16,
        max_repetitions=16,
        max_candidates=2,
        search_family_alpha=0.05,
        verification_family_alpha=0.05,
        diagnostic_family_alpha=0.05,
        temporal_tolerance=0.1,
        qualification_mode="lower_bound",
        qualification_region=(0.1, 1.0),
        min_normalized_return_displacement=0.0,
    )


def _fixture(target_id: str, *, use_xla: bool = False):
    import tensorflow as tf

    if target_id == "LGSSM":
        from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
            make_lgssm_generic_target_fixture,
        )

        fixture = make_lgssm_generic_target_fixture()
        offsets = tf.constant((
            (-0.15, -0.15),
            (-0.05, 0.05),
            (0.05, -0.05),
            (0.15, 0.15),
        ), dtype=tf.float64)
        starts = fixture.initial_batch + offsets
        source_paths = [str(Path(__file__).resolve()),
                        _source_file(type(fixture.adapter)),
                        _source_file(make_lgssm_generic_target_fixture),
                        _source_file(type(fixture.source_target))]
        metadata = {
            "target_id": target_id,
            "model": "static_qr_lgssm",
            "target_semantics": "exact_linear_gaussian_kalman_likelihood",
            "data_identity": "lgssm-static-qr-fixture-observations",
            "initialization": "data_independent_mechanics_fixture_not_tuned_geometry",
        }
        metadata["data_sha256"] = _tensor_sha256(fixture.source_target.observations)
    elif target_id in set(TARGETS) - {"LGSSM"}:
        from bayesfilter.testing.inference_validation.modern_nonlinear import (
            make_nonlinear_comparison_target,
        )

        fixture = make_nonlinear_comparison_target(target_id, jit_compile=use_xla)
        starts = fixture.initial_state
        source_paths = [str(Path(__file__).resolve()), *fixture.source_paths,
                        _source_file(fixture.adapter),
                        _source_file(__import__("bayesfilter.ssm", fromlist=["SSMTargetContract"]))]
        metadata = dict(fixture.metadata)
        metadata["target_semantics"] = {
            "NL-ACC": "nonlinear_accumulation_sigma_point_likelihood",
            "NL-PP-UKF": "nonlinear_predator_prey_ukf_likelihood",
            "NL-PP-SGQF": "nonlinear_predator_prey_sparse_quadrature_likelihood",
            "NL-SIR": "nonlinear_sir_sparse_quadrature_likelihood",
            "NL-STR": "nonlinear_structural_ukf_likelihood",
        }[target_id]
        metadata["data_sha256"] = _tensor_sha256(fixture.observations)
    else:
        raise ValueError(f"unsupported bridge target: {target_id}")
    from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
    source_paths = source_dependencies(
        [path for path in source_paths if path], repository_root=REPOSITORY_ROOT)
    metadata["source_paths_checked"] = list(source_paths)
    return fixture, starts, source_paths, metadata


def _binding(target_id, fixture, starts, source_paths, policy, *, data_sha256: str, use_xla: bool,
             non_xla_reason: str | None, epsilon: float = 0.1):
    import tensorflow as tf

    from bayesfilter.inference import HMCCandidateExecutionConfig, PrecomputedMassArtifact
    from bayesfilter.inference.hmc_candidate_set_execution import (
        bind_hmc_candidate_set_execution,
    )
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    from bayesfilter.inference.posterior_adapter import value_score_capability

    dimension = int(starts.shape[-1])
    identity = tf.eye(dimension, dtype=tf.float64)
    adapter = fixture.adapter
    mass = PrecomputedMassArtifact(
        position=tf.zeros([dimension], tf.float64),
        covariance=identity,
        factor=identity,
        adapter_signature=adapter.adapter_signature(),
        position_role="reference_center",
        covariance_source="explicit N5 bridge identity mechanics geometry",
    )
    status_policy = (
        "per_chain_step"
        if callable(getattr(adapter, "target_status_telemetry", None))
        else "none"
    )
    config = HMCCandidateExecutionConfig(
        measurement_num_results=policy.trial_num_results,
        verification_num_results=policy.trial_num_results,
        num_warmup_steps=policy.discarded_prefix,
        seed=(20261008, 31 + TARGETS.index(target_id)),
        acceptance_policy=policy,
        target_status_trace_policy=status_policy,
        use_xla=use_xla,
        non_xla_reason=non_xla_reason,
        chain_mode="batched",
        chunk_max_results=policy.trial_num_results,
        transition_profile=HMCTransitionProfile("hmc", "uniform_duration"),
    )
    return bind_hmc_candidate_set_execution(
        adapter=adapter,
        initial_position=starts,
        target_scope=(value_score_capability(adapter).target_scope
                      or f"modern-hmc-n5-bridge-{target_id.lower()}"),
        target_lineage={
            "model": target_id,
            "data": data_sha256,
            "geometry": "explicit_identity_reference_mechanics_geometry",
            "profile": "hmc_uniform_duration_v1",
        },
        config=config,
        source_paths=source_paths,
        scope_id=f"modern-hmc-n5-{target_id.lower()}",
        search_id="bridge-20261008-r1",
        epsilon_domain=(min(0.01, epsilon), max(1.0, epsilon)),
        repair_factor=1.25,
        max_repairs_per_family=0,
        mass_artifact=mass,
    ), config


def _search(policy, *, max_wall_time_seconds: float, epsilon: float = 0.1):
    from bayesfilter.inference import HMCControllerConfig

    return HMCControllerConfig(
        primary_l_grid=(2, 3),
        epsilon_by_l=((2, (epsilon,)), (3, (epsilon,))),
        total_budget_units=10,
        repair_reserve_units=1,
        candidate_reserve_units=3,
        max_candidates=2,
        evidence_rungs=(1,),
        max_gradient_work=150_000,
        max_wall_time_seconds=max_wall_time_seconds,
        replicated_acceptance_policy=policy,
    )


def _work_summary(result, binding):
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    return audit_native_evidence(result, binding)


def _qualification_status(result, work):
    expected = {c.candidate_id for c in result.candidates}
    if (result.completion_status == "complete" and len(expected) == 2
            and set(result.verified_candidate_ids) == expected
            and work["unobserved_attempted_chunk_count"] == 0
            and work["charged_gradient_work"] == work["expected_completed_gradient_work"]):
        return "mechanics_verified"
    return "mechanics_incomplete"


def _failure_class(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name or "budget" in name:
        return "bounded_execution_failure"
    if "resourceexhausted" in name or "unavailable" in name or "aborted" in name:
        return "infrastructure_failure"
    return "target_or_harness_failure"


def run(output: Path, targets: tuple[str, ...], *, execution_mode: str,
        max_wall_time_seconds: float, epsilon: float = 0.1) -> int:
    if output.exists():
        raise FileExistsError(f"fresh output root required: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    tf = None
    memory_policy = None
    try:
        if execution_mode not in {"cpu_reference", "gpu_xla"}:
            raise ValueError("execution_mode must be cpu_reference or gpu_xla")
        if not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive")
        if execution_mode == "cpu_reference":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
            raise ValueError("TF_FORCE_GPU_ALLOW_GROWTH=true is required before TensorFlow import")
        if execution_mode == "cpu_reference" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise ValueError("CPU reference mode must hide CUDA explicitly before import")
        import tensorflow as tf
        if execution_mode == "gpu_xla":
            from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
            memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        elif os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise ValueError("CPU reference mode must hide CUDA explicitly")
        policy = _policy()
    except Exception as exc:
        failure = {"failure_class": _failure_class(exc),
                   "exception_type": type(exc).__name__, "message": str(exc)}
        _json_write(output / "failure.json", failure)
        _json_write(output / "manifest.json", {
            "schema": "bayesfilter.modern_hmc_n5_bridge_manifest.v2",
            "command": sys.argv, "execution_mode": execution_mode,
            "max_wall_time_seconds": max_wall_time_seconds,
            "decision": "launch_failed", "failure": failure,
            "evidence_role": "diagnostic_failure_record",
        })
        return 1
    manifest = {
        "schema": "bayesfilter.modern_hmc_n5_bridge_manifest.v2",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "command": [sys.executable, *sys.argv], "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": sys.executable, "tensorflow": tf.__version__,
        "execution_mode": execution_mode, "gpu_intentionally_hidden": execution_mode == "cpu_reference",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_memory_policy": memory_policy,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "plan": "docs/plans/bayesfilter-modern-hmc-n5-bridge-2026-10-08.md",
        "result_file": str(output / "manifest.json"), "targets": list(targets),
        "policy": policy.payload(), "max_wall_time_seconds": max_wall_time_seconds,
        "epsilon": epsilon,
        "status": "running", "rows": rows,
        "nonclaims": ["no posterior convergence or burn-in sufficiency", "no sampler ranking",
                      "no scientific validity or release qualification"],
    }
    _json_write(output / "manifest.json", manifest)
    for target_id in targets:
        target_started = time.monotonic()
        target_output = output / target_id.lower()
        target_output.mkdir()
        try:
            if time.monotonic() - started >= max_wall_time_seconds:
                raise TimeoutError("bridge total wall-time cap reached before target")
            fixture, starts, source_paths, metadata = _fixture(target_id, use_xla=execution_mode == "gpu_xla")
            binding, execution_config = _binding(
                target_id, fixture, starts, source_paths, policy,
                data_sha256=metadata["data_sha256"], use_xla=execution_mode == "gpu_xla",
                epsilon=epsilon,
                non_xla_reason=("N5 CPU reference/mechanics bridge; explicit CPU mode"
                                if execution_mode == "cpu_reference" else None),
            )
            remaining = max_wall_time_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("wall cap reached during preparation")
            search = _search(policy, max_wall_time_seconds=remaining, epsilon=epsilon)
            from bayesfilter.inference import tune_hmc_kernel
            run_result = tune_hmc_kernel(
                adapter=binding._base_adapter,
                initial_position=binding.initial_active_state,
                config=search,
                candidate_set_adapter=binding.typed_adapter,
                output_dir=target_output,
            )
            result = run_result.result
            work = _work_summary(result, binding)
            qualification = _qualification_status(result, work)
            exported = {}
            if result.verified_candidate_ids:
                from bayesfilter.inference.hmc_candidate_set_retained import (
                    export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners)
                exported = export_hmc_candidate_retained_runners(candidate_set_result=result,
                    retained_binding=binding, output_dir=target_output / "members")
                loaded = load_hmc_candidate_retained_runners(tuple(exported.values()), adapter=fixture.adapter)
                if set(loaded) != set(result.verified_candidate_ids):
                    raise ValueError("export/reload lost a verified member")
            row = {
                **metadata, "target_id": target_id,
                "target_adapter_signature": fixture.adapter.adapter_signature(),
                "source_paths": source_paths, "start_shape": starts.shape.as_list(),
                "execution": execution_config.payload(), "search": search.payload(),
                "scheduler_completion_status": result.completion_status,
                "qualification_status": qualification, "final_status": result.final_status,
                "result_path": str(target_output / "candidate_set_result.json"),
                "artifact_authority": run_result.artifact_authority,
                "numerical_handoff_authority": run_result.numerical_handoff_authority,
                "work": work, "wall_seconds": time.monotonic() - target_started,
                "source_sha256": binding._spec["source_closure"],
                "reloaded_members": {key: str(path) for key, path in exported.items()},
                "evidence_role": ("gpu_xla_mechanics_smoke" if execution_mode == "gpu_xla"
                                   else "cpu_reference_mechanics_bridge_only"),
            }
            _json_write(target_output / "bridge_summary.json", row)
            rows.append(row)
            print(json.dumps({"target": target_id, "status": qualification,
                              "verified": len(result.verified_candidate_ids),
                              "gradient_work": work["charged_gradient_work"],
                              "seconds": row["wall_seconds"]}), flush=True)
        except Exception as exc:
            failure = {"target_id": target_id, "failure_class": _failure_class(exc),
                       "exception_type": type(exc).__name__, "message": str(exc),
                       "wall_seconds": time.monotonic() - target_started}
            _json_write(target_output / "failure.json", failure)
            rows.append({**failure, "qualification_status": "failed",
                         "evidence_role": "diagnostic_failure_record"})
            print(json.dumps(failure), flush=True)
        manifest.update(wall_seconds=time.monotonic() - started, rows=rows)
        _json_write(output / "manifest.json", manifest)
    manifest.update(status="finished", decision=("mechanics_bridge_complete"
        if rows and all(row["qualification_status"] == "mechanics_verified" for row in rows)
        else "repair_needed"))
    if execution_mode == "gpu_xla":
        manifest["gpu_allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    _json_write(output / "manifest.json", manifest)
    return 0 if manifest["decision"] == "mechanics_bridge_complete" else 1


def supervise(args):
    """Enforce the total wall cap even when a native TensorFlow call blocks."""
    if not math.isfinite(args.max_wall_time_seconds) or args.max_wall_time_seconds <= 0:
        raise ValueError("positive finite wall cap required")
    epsilon = getattr(args, 'epsilon', 0.1)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive finite epsilon required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="3",
               OMP_NUM_THREADS="2", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
    manifest = dict(command=[sys.executable, *sys.argv], status="running",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(), environment=sys.executable,
        execution_mode=args.execution_mode, wall_cap_seconds=args.max_wall_time_seconds,
        plan="docs/plans/bayesfilter-modern-hmc-n5-bridge-2026-10-08.md",
        result_file=str(args.output/"cells/manifest.json"))
    _json_write(args.output / "manifest.json", manifest)
    try:
        if args.execution_mode == "cpu_reference":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            probe = subprocess.run([str(Path.home()/".codex/bin/codex-gpu-probe"),
                "--framework", "nvidia", "--gpu", "auto"], capture_output=True, text=True,
                timeout=min(15, args.max_wall_time_seconds), check=True)
            payload = json.loads(probe.stdout)
            _json_write(args.output / "gpu-probe.json", payload)
            if payload["status"] != "PASSED":
                raise RuntimeError("trusted GPU probe failed")
            env["CUDA_VISIBLE_DEVICES"] = str(payload["selected_host_gpu"])
            manifest["gpu_trust_basis"] = "trusted_escalated_gpu_execution"
        remaining = args.max_wall_time_seconds - (time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError("wall cap reached before worker")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
            "--output", str((args.output/"cells").resolve()), "--targets", *args.targets,
            "--execution-mode", args.execution_mode, "--max-wall-time-seconds", str(remaining),
            "--epsilon", str(epsilon)]
        manifest.update(worker_command=command, cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"],
                        gpu_intentionally_hidden=args.execution_mode == "cpu_reference")
        _json_write(args.output / "manifest.json", manifest)
        with (args.output/"worker.log").open("w") as log:
            result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=remaining)
        manifest.update(status="finished" if result.returncode == 0 else "repair_needed", returncode=result.returncode)
    except (subprocess.TimeoutExpired, TimeoutError):
        manifest.update(status="wall_budget_exhausted", partial_evidence_preserved=True)
    except Exception as exc:
        manifest.update(status="launch_failed", error_type=type(exc).__name__, error=str(exc))
    manifest["wall_seconds"] = time.monotonic()-started
    _json_write(args.output/"manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "wall_seconds")}))
    return 0 if manifest["status"] == "finished" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--execution-mode", choices=("cpu_reference", "gpu_xla"),
                        default="gpu_xla")
    parser.add_argument("--max-wall-time-seconds", type=float, default=600.0)
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Explicit mechanics hypothesis, not a universal step size")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run(args.output, tuple(args.targets), execution_mode=args.execution_mode,
                   max_wall_time_seconds=args.max_wall_time_seconds, epsilon=args.epsilon)
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
