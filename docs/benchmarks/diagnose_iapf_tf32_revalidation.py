"""CPU reference/GPU diagnostic campaign; calls the actual shared iAPF kernel.

No fitting, runtime policy, or canonical LEDH implementation lives here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

GPU = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
PLAN = "docs/plans/iapf-tf32-revalidation-2026-09-23.md"
ROOT = "docs/plans/artifacts/iapf-tf32-revalidation-20260923-01"
PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def worker(mode, fixture, out, *, replicates=512, dimensions=None,
           particle_seed_offset=0, trace_all=False, compress_traces=False):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy = configure_tensorflow_gpu_memory_growth(tf)
    tf.config.set_soft_device_placement(False)
    provenance = dict(tensorflow=tf.__version__, build=tf.sysconfig.get_build_info(),
                      memory_policy=policy, tf32=mode == "tf32", jit_compile=True,
                      python=sys.version, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "probe":
        with tf.device("/GPU:0"):
            value = tf.linalg.matvec(tf.eye(2), tf.ones([2]))
        provenance.update(device=value.device, value=value.numpy().tolist())
        dump(out / "probe.json", provenance)
        return 0
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    tf.config.experimental.enable_tensor_float_32_execution(mode == "tf32")
    dtype = tf.float64 if mode == "fp64" else tf.float32
    dtype_name = dtype.name
    provenance["tf32_verified"] = tf.config.experimental.tensor_float_32_execution_enabled()
    kernels, compilations, checks = {}, [], []

    def read(path, shape, common_fp32=False):
        values = [float(row[0]) for row in csv.reader(Path(path).open())]
        # R's text writer need not preserve all bits of an FP32 value promoted
        # to FP64. Quantize through FP32 explicitly in distributional arms.
        tensor = tf.constant(values, tf.float32 if common_fp32 else dtype)
        return tf.reshape(tf.cast(tensor, dtype), shape)

    def kernel(d, N, T, constant, traced=False):
        key = (d, N, T, constant, traced)
        if key not in kernels:
            kernels[key] = make_fitted_twist_kernel(d, d, N, T, dtype_name, True,
                constant_twist=constant, include_numerical_trace=traced)
        return kernels[key]

    def finite(result):
        return all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in result[:3])

    def compile_record(fn, args, label, result):
        hlo = fn.experimental_get_compiler_ir(*args)(stage="optimized_hlo")
        path = out / (label + ".hlo.txt")
        path.write_text(hlo)
        compilations.append(dict(label=label, hlo_sha256=sha(path), device=result[0].device,
                                 trace_count=fn.experimental_get_tracing_count()))
        if "GPU" not in result[0].device:
            raise RuntimeError("GPU placement veto")

    # Preserve the exact original tolerances and the original R-supplied draws.
    with tf.device("/GPU:0"):
        for d in (1, 2, 5):
            directory = fixture / "old" / f"d{d}"
            N, T = 64, 5
            arrays = [read(directory / f"{name}.csv", shape) for name, shape in (
                ("theta", [6]), ("observations", [T, d]), ("initial", [N, d]),
                ("noise", [T, N, d]), ("uniforms", [T + 1, N]), ("mixture", [T, N]),
                ("centers", [T, d]), ("covariances", [T, d, d]), ("floors", [T]))]
            for constant in (False, True):
                prefix = "constant" if constant else "fitted"
                fn = kernel(d, N, T, constant)
                result = fn(*arrays)
                compile_record(fn, arrays, f"old-d{d}-{prefix}", result)
                if not finite(result):
                    raise RuntimeError("Nonfinite old-fixture result")
                for name, actual, shape in zip(("value", "score", "clouds"), result,
                                               ([], [6], [T, N, d])):
                    expected = read(directory / f"{prefix}-{name}.csv", shape)
                    if dtype == tf.float64:
                        atol, rtol = (1e-5, 1e-7) if name == "score" else (1e-9, 1e-9)
                    else:
                        atol, rtol = (5e-4, 1e-4) if name == "score" else (1e-4, 1e-5)
                    error = tf.abs(actual - expected)
                    checks.append(dict(d=d, arm=prefix, quantity=name,
                        max_error=float(tf.reduce_max(error).numpy()), atol=atol, rtol=rtol,
                        passed=bool(tf.reduce_all(error <= atol + rtol * tf.abs(expected)).numpy())))
    dump(out / "old-fixture-checks.json", checks)
    if mode != "tf32" and not all(x["passed"] for x in checks):
        raise RuntimeError("Independent R old-fixture validity veto")

    cells = list(csv.DictReader((fixture / "cells.csv").open()))
    if dimensions is not None:
        cells = [cell for cell in cells if int(cell["d"]) in dimensions]
    records = []
    started = time.monotonic()
    with (out / "records.csv").open("w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=["cell", "d", "N", "T", "arm", "mode",
            "replicate", "seed", "log_likelihood", "kalman", "normalized_likelihood"])
        writer.writeheader()
        for cell in cells:
            name, arm = cell["id"], cell["arm"]
            d, N, T = (int(cell[k]) for k in ("d", "N", "T"))
            directory = fixture / name
            with tf.device("/GPU:0"):
                theta = read(directory / "theta.csv", [6], common_fp32=True)
                observations = read(directory / "observations.csv", [T, d], common_fp32=True)
                guides = [read(directory / f"{arm}-{suffix}.csv", shape, common_fp32=True) for suffix, shape in (
                    ("centers", [T, d]), ("covariances", [T, d, d]), ("floors", [T]))]
            fn = kernel(d, N, T, arm == "constant")
            traced = kernel(d, N, T, arm == "constant", True)
            trace_path = out / f"{name}-{arm}-traces.jsonl"
            if compress_traces:
                import gzip
                trace_stream = gzip.open(str(trace_path) + ".gz", "wt")
            else:
                trace_stream = trace_path.open("w")
            with trace_stream as trace_file:
                for rep in range(replicates):
                    seed = int(cell["particle_seed"]) + particle_seed_offset + rep
                    # Explicit CPU diagnostic random-input generation, identical across modes.
                    with tf.device("/CPU:0"):
                        init = tf.random.stateless_normal([N, d], [seed, 1], dtype=tf.float32)
                        noise = tf.random.stateless_normal([T, N, d], [seed, 2], dtype=tf.float32)
                        unif = tf.cast(tf.random.stateless_uniform([T + 1, N], [seed, 3],
                            minval=0, maxval=2**23, dtype=tf.int32), tf.float32) / 2**23
                        mix = tf.cast(tf.random.stateless_uniform([T, N], [seed, 4],
                            minval=0, maxval=2**23, dtype=tf.int32), tf.float32) / 2**23
                        draws = [tf.cast(v, dtype) for v in (init, noise, unif, mix)]
                    with tf.device("/GPU:0"):
                        args = [theta, observations, *draws, *guides]
                        result = fn(*args)
                        value = float(result[0].numpy())
                        if not finite(result):
                            raise RuntimeError(f"Nonfinite consumer: {name}/{arm}/{rep}")
                        if rep == 0 and name in ("old_d1", "old_d2", "old_d5", "long1_d5"):
                            compile_record(fn, args, f"distribution-{name}-{arm}", result)
                        relative = float(tf.exp(tf.cast(result[0], tf.float64) -
                            tf.constant(float(cell["kalman"]), tf.float64)).numpy())
                        if trace_all or rep < 8:
                            traced_result = traced(*args)
                            equal = all(bool(tf.reduce_all(a == b).numpy())
                                        for a, b in zip(result, traced_result[:3]))
                            if not equal:
                                raise RuntimeError(f"Numerical tracing changed {name}/{arm}/{rep}")
                            tr = traced_result[-1]
                            record = dict(replicate=rep, neutral=True,
                                cloud=result[2].numpy().tolist(), score=result[1].numpy().tolist(),
                                ancestors=tr["ancestor_indices"].numpy().tolist(),
                                cdf=tr["ancestor_cdf"].numpy().tolist(),
                                probabilities=tr["gaussian_probability"].numpy().tolist(),
                                mixture_choices=(tf.cast(mix, dtype) < tr["gaussian_probability"]).numpy().tolist())
                            trace_file.write(json.dumps(record, allow_nan=False) + "\n")
                    row = dict(cell=name, d=d, N=N, T=T, arm=arm, mode=mode, replicate=rep,
                        seed=seed, log_likelihood=value, kalman=cell["kalman"], normalized_likelihood=relative)
                    writer.writerow(row)
                log.flush()
            records.append(dict(cell=name, arm=arm, replicates=replicates))
            dump(out / "progress.json", dict(completed=records, seconds=time.monotonic()-started))
    provenance.update(status="complete", compilation=compilations, old_fixture_checks=checks,
        allocator=tf.config.experimental.get_memory_info("GPU:0"), distribution_records=replicates * len(cells),
        replicates=replicates,particle_seed_offset=particle_seed_offset,
        trace_all=trace_all,compress_traces=compress_traces,
        consumer="bayesfilter.score_study.fitted_twist_tf.make_fitted_twist_kernel",
        fixed_label_score_only=True, default_promotion=False)
    dump(out / "result.json", provenance)
    return 0


def supervise():
    repo = Path(__file__).resolve().parents[2]
    root = repo / ROOT
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "budget.json"
    if ledger.exists():
        budget = json.loads(ledger.read_text())
    else:
        previous = json.loads((repo / "docs/plans/artifacts/iapf-score-failure-remedies-20260923-01/budget.json").read_text())
        budget = dict(cpu_charged_seconds=0., gpu_charged_seconds=30.,
            phase_cpu_cap_seconds=3600., phase_gpu_cap_seconds=7200., attempts=0,
            remaining_campaign_cpu_seconds=previous["remaining_campaign_cpu_seconds"],
            remaining_campaign_gpu_seconds=previous["remaining_campaign_gpu_seconds"]-30,
            cpu_reserved_seconds=0., gpu_reserved_seconds=0., probe_charge_note="30s conservative NVIDIA availability probe")
        dump(ledger, budget)
    if budget["cpu_reserved_seconds"] or budget["gpu_reserved_seconds"]:
        raise RuntimeError("An existing attempt needs settlement before resuming")
    # Scientific sources frozen for this launch, including the plan and shared call chain.
    attempt_number = 1 + len(list(root.glob("supervisor-*.json")))
    source = root / f"source-{attempt_number:02d}"
    source.mkdir()
    files = ["bayesfilter/__init__.py", "bayesfilter/runtime/__init__.py",
        "bayesfilter/runtime/gpu_memory_policy.py", PLAN,
        "docs/benchmarks/reference_iapf_paper.R", "docs/benchmarks/reference_iapf_tf_adaptive.R",
        "docs/benchmarks/export_iapf_tf_filter_reference.R", "docs/benchmarks/diagnose_iapf_tf_center_probe.py",
        "docs/benchmarks/export_iapf_tf32_revalidation.R", "docs/benchmarks/diagnose_iapf_tf32_revalidation.py",
        "docs/benchmarks/summarize_iapf_tf32_revalidation.R"]
    files += [str(x.relative_to(repo)) for x in (repo / "bayesfilter/score_study").glob("*.py")]
    for name in files:
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, dest)
    sources = {name: sha(source / name) for name in files}
    state = dict(status="running", source=str(source), sources=sources, plan=PLAN,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        commands=[], output_root=str(root), result_file=str(root / "result.md"))
    state_path = root / f"supervisor-{attempt_number:02d}.json"
    dump(state_path, state)
    environment = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", BAYESFILTER_PRELOAD_CUSTOM_OP="0",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2", PYTHONPATH=str(source))
    # No inherited compiler workaround: this tests the installed default compiler route.
    if environment.get("XLA_FLAGS", ""):
        raise RuntimeError("Unexpected XLA_FLAGS; review before launching")
    completed_path = root / "completed.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}
    fixture = Path(completed["fixtures"]) if "fixtures" in completed else root / f"fixtures-{attempt_number:02d}"
    fixture.mkdir(exist_ok=True)

    def run(label, command, gpu=False, cap=300):
        kind = "gpu" if gpu else "cpu"
        if budget["attempts"] >= 12 or budget[f"{kind}_charged_seconds"] + cap > budget[f"phase_{kind}_cap_seconds"]:
            raise RuntimeError("Phase attempt/time budget veto")
        if cap > budget[f"remaining_campaign_{kind}_seconds"]:
            raise RuntimeError("Campaign time budget veto")
        budget["attempts"] += 1
        job = root / f"attempt{budget['attempts']:02d}-{label}"
        job.mkdir()
        budget[f"{kind}_reserved_seconds"] = cap
        dump(ledger, budget)
        command = [str(x).replace("{OUT}", str(job)) for x in command]
        env = dict(environment, CUDA_VISIBLE_DEVICES=GPU if gpu else "-1")
        manifest = dict(label=label, status="running", command=command, cwd=str(source),
            source_sha256=sources, source=str(source), git_commit=state["git_commit"], plan=PLAN,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), cpu_only=not gpu,
            gpu_intentionally_hidden=not gpu, cap_seconds=cap,
            environment={k:env.get(k, "") for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
                "BAYESFILTER_PRELOAD_CUSTOM_OP", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "XLA_FLAGS")},
            data_version="phase18_frozen_R_fixtures", fixture=str(fixture),
            seeds="cells.csv plus particle_seed+replicate[0,511]; old R seeds 93600001/2/5")
        dump(job / "manifest.json", manifest)
        start = time.monotonic()
        try:
            with (job / "command.log").open("w") as log:
                status = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=cap)
            manifest.update(returncode=status.returncode, status="complete" if status.returncode == 0 else "failed")
        except BaseException as error:
            manifest.update(status="failed", error=repr(error))
        finally:
            seconds = time.monotonic()-start
            manifest["wall_seconds"] = seconds
            budget[f"{kind}_reserved_seconds"] = 0.
            budget[f"{kind}_charged_seconds"] += seconds
            budget[f"remaining_campaign_{kind}_seconds"] -= seconds
            manifest["output_sha256"] = {str(x.relative_to(job)):sha(x) for x in job.rglob("*")
                if x.is_file() and x.name != "manifest.json"}
            dump(job / "manifest.json", manifest)
            dump(ledger, budget)
            state["commands"].append(str(job / "manifest.json"))
            dump(state_path, state)
        print(json.dumps(dict(label=label, status=manifest["status"], seconds=seconds, path=str(job))), flush=True)
        if manifest["status"] != "complete":
            state.update(status="repair_required", failed_attempt=str(job))
            dump(state_path, state)
            raise RuntimeError(f"Attempt failed: {job}")
        return job

    script = source / "docs/benchmarks/diagnose_iapf_tf32_revalidation.py"
    if "fixtures" not in completed:
        run("old-fixtures", ["/usr/bin/Rscript", "--vanilla", source / "docs/benchmarks/export_iapf_tf_filter_reference.R", fixture / "old"])
        run("fixtures", ["/usr/bin/Rscript", "--vanilla", source / "docs/benchmarks/export_iapf_tf32_revalidation.R", fixture])
        completed["fixtures"] = str(fixture)
        completed["fixture_hashes"] = {str(x.relative_to(fixture)):sha(x) for x in fixture.rglob("*") if x.is_file()}
        dump(completed_path, completed)
    else:
        if any(sha(fixture / name) != val for name, val in completed["fixture_hashes"].items()):
            raise RuntimeError("Fixture mutation veto")
    for label in ("probe", "center-probe", "fp64", "fp32", "tf32"):
        if label in completed:
            continue
        cap = 120 if label == "probe" else 300 if label == "center-probe" else 1500
        job = run(label, [PYTHON, script, "--mode", label, "--fixture", fixture, "--output", "{OUT}/outputs"], gpu=True, cap=cap)
        completed[label] = str(job / "outputs")
        dump(completed_path, completed)
    if "summary" not in completed:
        job = run("summary", ["/usr/bin/Rscript", "--vanilla", source / "docs/benchmarks/summarize_iapf_tf32_revalidation.R",
            fixture, completed["fp64"], completed["fp32"], completed["tf32"], "{OUT}/outputs"], cap=300)
        completed["summary"] = str(job / "outputs")
        dump(completed_path, completed)
    state.update(status="complete_pending_interpretation", completed=completed)
    dump(state_path, state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=["probe", "center-probe", "fp64", "fp32", "tf32"])
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute:
        return supervise()
    if args.mode == "center-probe":
        args.output.mkdir(parents=True)
        from diagnose_iapf_tf_center_probe import run
        return run(args.fixture / "old", args.output / "center-probe.json")
    if args.mode:
        return worker(args.mode, args.fixture, args.output)
    parser.error("Select --execute or --mode")


if __name__ == "__main__":
    raise SystemExit(main())
