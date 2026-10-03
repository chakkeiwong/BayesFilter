"""Bounded diagnostic reconstruction of a recorded ordinary HMC preparation.

Calls the public preparation helper from a frozen source and preserves its
existing window decisions and discarded draws. Issues no tuning authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    sys.path.insert(0, str(args.source))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json, write_tensor

    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write_json(args.output / "runtime.json", runtime)
    source = source_state()
    if source["identity"] != read(args.source / "source_snapshot.json")["source_identity"]:
        raise ValueError("source snapshot changed")
    from bayesfilter.testing.inference_validation.posteriordb_targets import (
        PosteriordbTarget, load_case, UPSTREAM_COMMIT,
    )
    from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig
    from bayesfilter.inference.hmc_preparation import (
        HMCPreparationProgress, _progress_json_value, prepare_operational_windowed_mass_handoff,
    )
    import tensorflow as tf

    if args.trajectory_diagnostics:
        from hmc_preparation_trajectory_diagnostic import install
        install(args.output / "trajectory_diagnostics")

    previous = read(args.recorded_manifest)
    checkout = REPO / ".localresources/posteriordb-20260918"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != UPSTREAM_COMMIT or commit != previous["upstream_commit"]:
        raise ValueError("posteriordb checkout mismatch")
    case = load_case(checkout, previous["case"])
    for record in case["files"].values():
        path = Path(record["path"])
        blob = subprocess.check_output(["git", "show", commit + ":" + str(path.relative_to(checkout))], cwd=checkout)
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("upstream input differs from pinned tree")
    target = PosteriordbTarget(previous["case"], case["data"], jit_compile=True)
    initial = previous["initialization"]
    if (not initial["passed"] or initial["reference_draws_used"]
            or initial["target_signature"] != target.adapter_signature()
            or initial["data_sha256"] != case["files"]["data"]["sha256"]):
        raise ValueError("initialization identity mismatch")
    theta = tf.constant(initial["initial_position"], tf.float64)
    value, score = target.log_prob_and_grad(theta)
    if not bool(tf.math.is_finite(value)) or float(tf.reduce_max(tf.abs(score))) > initial["gradient_tolerance"]:
        raise ValueError("initial target check failed")
    saved = previous["preparation"]
    kwargs = {key: saved[key] for key in (
        "preset", "use_xla", "target_scope", "candidate_search_bound_expansion_steps", "public_timeout_budget_s"
    )}
    kwargs["seed"] = tuple(previous["seed"])
    for optional_field in ("bootstrap_initialization_rounds", "metric_evidence_policy", "metric_probe_num_results", "preparation_max_restarts"):
        if optional_field in saved:
            kwargs[optional_field] = saved[optional_field]
    config = HMCKernelTuningConfig(**kwargs)
    config_payload = json.loads(json.dumps(config.payload()))
    mismatches = [key for key, value in saved.items() if config_payload.get(key) != value]
    if mismatches:
        raise ValueError("configuration reconstruction mismatch: " + str(mismatches))
    if any(value is not None for value in (args.preparation_preset, args.metric_evidence_policy,
                                           args.metric_probe_num_results, args.preparation_max_restarts)):
        from dataclasses import replace
        config = replace(config, **{key: value for key, value in {
            "preset": args.preparation_preset, "metric_evidence_policy": args.metric_evidence_policy,
            "metric_probe_num_results": args.metric_probe_num_results,
            "preparation_max_restarts": args.preparation_max_restarts,
        }.items() if value is not None})
    resolved_payload = json.loads(json.dumps(config.payload()))
    hint = previous.get("geometry_hint")
    if hint is not None and (not hint["passed"] or hint["reference_draws_used"]
            or hint["target_signature"] != target.adapter_signature()
            or hint["data_sha256"] != case["files"]["data"]["sha256"]):
        raise ValueError("geometry hint identity mismatch")
    write_json(args.output / "manifest.json", {
        "command": sys.argv, "source": source, "runtime": runtime, "plan_file": PLAN,
        "result_file": str(args.output / "assessment.json"), "environment": sys.executable,
        "recorded_manifest": str(args.recorded_manifest), "recorded_manifest_sha256": file_hash(args.recorded_manifest),
        "data_version": case["files"]["data"]["sha256"], "upstream_commit": commit,
        "target_signature": target.adapter_signature(), "seed": list(config.seed),
        "config": resolved_payload, "geometry_hint_supplied": hint is not None,
        "configuration_changes": {key: {"baseline": config_payload.get(key), "current": value}
                                  for key, value in resolved_payload.items() if config_payload.get(key) != value},
        "reference_draws_used": False, "role": "preparation diagnosis only",
    })
    with HMCPreparationProgress(args.output, max_wall_time_seconds=args.seconds - 30.) as progress:
        handoff = prepare_operational_windowed_mass_handoff(
            adapter=target, initial_position=theta, config=config,
            negative_hessian=None if hint is None else tf.constant(hint["negative_hessian"], tf.float64),
            progress_callback=progress.phase,
        )
    result = handoff["windowed_stage"].operational_warmup_result
    windows = []
    for index, window in enumerate(result.windows):
        record = dict(window.public_payload())
        record["adaptation_shape"] = list(window.adaptation_latent_states.shape)
        path = args.output / f"window-{index:02d}-latent.tensor"
        write_tensor(path, window.adaptation_latent_states)
        record["latent_draws_path"] = str(path)
        record["latent_draws_sha256"] = file_hash(path)
        windows.append(record)
    summary = {
        "status": result.status, "metric_adaptation_status": result.metric_adaptation_status,
        "operational_metric_update_count": result.operational_metric_update_count,
        "windows": windows, "config": result.config.payload(),
        "final_transform": {
            "signature": result.final_kernel_state.transform.signature,
            "factor": tf.convert_to_tensor(result.final_kernel_state.transform.factor).numpy().tolist(),
            "center": tf.convert_to_tensor(result.final_kernel_state.transform.center).numpy().tolist(),
        },
        "final_epsilon": result.final_kernel_state.epsilon,
        "preparation_recovery": result.preparation_recovery,
        "elapsed_seconds": progress.elapsed_seconds,
        "candidate_authority": False, "reports_posterior_convergence": False,
    }
    write_json(args.output / "assessment.json", _progress_json_value(summary))
    print(json.dumps({"updates": summary["operational_metric_update_count"], "windows": len(windows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recorded-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--preparation-preset", choices=("serious",))
    parser.add_argument("--metric-evidence-policy", choices=("finite_window",))
    parser.add_argument("--metric-probe-num-results", type=int)
    parser.add_argument("--preparation-max-restarts", type=int)
    parser.add_argument("--campaign-stage", choices=("m10", "m11", "m12", "m13"), default="m10")
    parser.add_argument("--trajectory-diagnostics", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    for name in ("source", "recorded_manifest", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("GPU memory growth must be set before framework import")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2"}:
        raise RuntimeError("one explicitly selected trusted GPU is required")
    if not 30. < args.seconds <= 900.:
        raise ValueError("diagnosis must fit the recorded 900-second ceiling")
    if args.worker:
        try:
            worker(args)
        except Exception as exc:
            (args.output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
            raise
        return
    costs = [read(path) for path in args.output.parent.glob("*/gpu-diagnostic-run.json")]
    attempt_cap, seconds_cap = {"m10": (10, 9000.), "m11": (6, 7200.), "m12": (8, 9000.), "m13": (10, 9000.)}[args.campaign_stage]
    if len(costs) >= attempt_cap or sum(row["elapsed_seconds"] for row in costs) + args.seconds > seconds_cap:
        raise RuntimeError(args.campaign_stage + " attempt or worker-wall budget exhausted")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py.txt").write_bytes(Path(__file__).read_bytes())
    if args.trajectory_diagnostics:
        (args.output / "trajectory_harness.py.txt").write_bytes(
            Path(__file__).with_name("hmc_preparation_trajectory_diagnostic.py").read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    started = time.monotonic()
    when = datetime.now(timezone.utc).isoformat()
    with (args.output / "worker.log").open("x") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record = {
        "command": command, "started_utc": when, "elapsed_seconds": time.monotonic() - started,
        "returncode": code, "device": "gpu", "environment": sys.executable,
        "plan_file": PLAN, "result_file": str(args.output / "assessment.json"),
        "script_sha256": file_hash(__file__), "script_path": str(Path(__file__).resolve()),
        "git_commit": read(args.source / "source_snapshot.json")["git_commit"],
        "runtime": read(args.output / "runtime.json") if (args.output / "runtime.json").exists() else None,
    }
    (args.output / "gpu-diagnostic-run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
