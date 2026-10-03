"""M14 bounded paired preparation diagnosis; no candidate/posterior authority."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = ROOT / "source-schedule-r1"
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(row, out):
    sys.path.insert(0, str(SOURCE))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json, write_tensor
    from bayesfilter.testing.inference_validation.designs import seed_for
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write_json(out / "runtime.json", runtime)
    source = source_state()
    assert source["identity"] == read(SOURCE / "source_snapshot.json")["source_identity"]
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget, load_case, UPSTREAM_COMMIT
    from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig
    from bayesfilter.inference.hmc_preparation import HMCPreparationProgress, prepare_operational_windowed_mass_handoff, _progress_json_value
    model = row["model"]
    hint = {}
    inputs = {}
    if model in ("sblrc-blr", "eight_schools-eight_schools_noncentered"):
        checkout = REPO / ".localresources/posteriordb-20260918"
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == UPSTREAM_COMMIT
        case = load_case(checkout, model)
        for name in ("model", "data", "posterior"):
            p = Path(case["files"][name]["path"])
            blob = subprocess.check_output(["git", "show", UPSTREAM_COMMIT + ":" + str(p.relative_to(checkout))], cwd=checkout)
            assert hashlib.sha256(blob).hexdigest() == sha(p)
        target = PosteriordbTarget(model, case["data"], jit_compile=True)
        inputs = {name: case["files"][name] for name in ("model", "data", "posterior")}
        theta = tf.zeros(target.parameter_dim, tf.float64)
        if model == "sblrc-blr":
            p = ROOT.parent / "m8-r1/regression-data-initialization.json"
            init = read(p)
            assert init["passed"] and not init["reference_draws_used"] and init["target_signature"] == target.adapter_signature()
            assert init["data_sha256"] == case["files"]["data"]["sha256"]
            inputs["initialization"] = {"path": str(p), "sha256": sha(p)}
            theta = tf.constant(init["initial_position"], tf.float64)
            if row["hint"]:
                p = ROOT.parent / "m8-r1/regression-geometry-check.json"
                geometry = read(p)
                assert geometry["passed"] and not geometry["reference_draws_used"]
                assert geometry["target_signature"] == target.adapter_signature() and geometry["data_sha256"] == init["data_sha256"]
                hint["negative_hessian"] = tf.constant(geometry["negative_hessian"], tf.float64)
                inputs["geometry"] = {"path": str(p), "sha256": sha(p)}
    else:
        options = {"scaled_gaussian": ("gaussian", {"scale": .02}, None),
                   "rotated_gaussian": ("rotated_gaussian", {"condition": 100., "angle": .6}, None),
                   "beta_binomial": ("beta_binomial", {"alpha": 2., "beta": 3.}, [7, 20]),
                   "lgssm_location": ("lgssm_location", {}, [1., -1., .5])}
        kind, params, data = options[model]
        target = ValidationTarget(kind, params, data, jit_compile=True)
        theta = tf.zeros(target.parameter_dim, tf.float64)
        inputs = {"target": kind, "parameters": params, "data": data}
        if row["hint"]:
            angle = tf.constant(.6, tf.float64)
            c, s = tf.cos(angle), tf.sin(angle)
            rot = tf.stack([tf.stack([c, s]), tf.stack([-s, c])])
            hint["initial_covariance"] = tf.transpose(rot) @ tf.linalg.diag(tf.constant([1., 100.], tf.float64)) @ rot
    value, score = target.log_prob_and_grad(theta)
    assert bool(tf.reduce_all(tf.math.is_finite(value))) and bool(tf.reduce_all(tf.math.is_finite(score)))
    seed = seed_for(row["seed_root"], "m14-preparation-comparison", model)
    cfg = HMCKernelTuningConfig(preset=row["preset"], seed=seed, use_xla=True,
        target_scope="inference_validation", metric_evidence_policy=row["metric_evidence_policy"],
        metric_probe_num_results=16, preparation_max_restarts=3,
        bootstrap_initialization_rounds=row["startup"], public_timeout_budget_s=row["seconds"] - 30.)
    write_json(out / "manifest.json", {"command": sys.argv, "environment": sys.executable,
        "source": source, "runtime": runtime, "config": cfg.payload(), "row": row,
        "inputs": inputs, "initial_position": theta.numpy().tolist(), "seed": list(seed),
        "target_signature": target.adapter_signature(), "plan_file": PLAN,
        "result_file": str(out / "assessment.json"), "reference_draws_used": False,
        "harness_sha256": sha(Path(__file__)), "candidate_authority": False})
    try:
        with HMCPreparationProgress(out, max_wall_time_seconds=row["seconds"] - 30.) as progress:
            handoff = prepare_operational_windowed_mass_handoff(adapter=target,
                initial_position=theta, config=cfg, progress_callback=progress.phase, **hint)
        result = handoff["windowed_stage"].operational_warmup_result
        windows = []
        for i, w in enumerate(result.windows):
            path = out / f"window-{i:02}-latent.tensor"
            write_tensor(path, w.adaptation_latent_states)
            windows.append({**w.public_payload(), "draws_path": str(path), "draws_sha256": sha(path)})
        summary = {"status": result.status, "operational_metric_update_count": result.operational_metric_update_count,
            "metric_adaptation_status": result.metric_adaptation_status, "windows": windows,
            "final_epsilon": result.final_kernel_state.epsilon,
            "factor": tf.convert_to_tensor(result.final_kernel_state.transform.factor).numpy().tolist(),
            "coordinate_signature": result.final_kernel_state.transform.signature,
            "preparation_recovery": result.preparation_recovery,
            "elapsed_seconds": progress.elapsed_seconds, "candidate_authority": False,
            "reports_posterior_convergence": False}
        write_json(out / "assessment.json", _progress_json_value(summary))
    except Exception as exc:
        write_json(out / "failure.json", {"exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("memory growth must precede framework import")
    design = read(ROOT / "schedule-repair-design.json")
    if args.worker:
        row = next(r for r in design["rows"] if r["job_id"] == args.worker)
        global SOURCE
        SOURCE = (ROOT / row["source"]).resolve()
        worker(row, ROOT / row["job_id"])
        return
    for row in design["rows"]:
        SOURCE = (ROOT / row["source"]).resolve()
        if args.only and row["job_id"] not in args.only:
            continue
        out = ROOT / row["job_id"]
        if (out / "gpu-diagnostic-run.json").exists():
            continue
        costs = sum(read(p)["elapsed_seconds"] for p in ROOT.glob("schedule-*/gpu-diagnostic-run.json"))
        if costs + row["seconds"] > 1080.:
            raise RuntimeError("paired preparation budget exhausted")
        out.mkdir(exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", row["job_id"]]
        started = time.monotonic()
        when = datetime.now(timezone.utc).isoformat()
        with (out / "worker.log").open("x") as log:
            try:
                code = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, timeout=row["seconds"]).returncode
            except subprocess.TimeoutExpired:
                code = 124
        record = {"command": command, "started_utc": when, "elapsed_seconds": time.monotonic() - started,
            "returncode": code, "device": "gpu", "environment": sys.executable, "plan_file": PLAN,
            "script_sha256": sha(Path(__file__)), "source_identity": read(SOURCE / "source_snapshot.json")["source_identity"],
            "runtime": read(out / "runtime.json") if (out / "runtime.json").exists() else None}
        (out / "gpu-diagnostic-run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(row["job_id"], row["model"], code, record["elapsed_seconds"], flush=True)
        if not (out / "manifest.json").exists():
            raise RuntimeError("preparation harness failed before establishing its inputs; repair before continuing")


if __name__ == "__main__":
    main()
