#!/usr/bin/env python3
"""Bounded qualification and cost forecast for the independent KSC finite target."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
from scripts.prepare_surrogate_hmc_ksc import (
    budgets, check_selection, run_bounded_worker, save, sha, sources, verify_qualification,
)
from scripts.run_surrogate_hmc_ksc import check_reference

PLAN = "docs/plans/surrogate-hmc-ksc-finite-target-2026-09-21.md"
GRID_BATCH = 16
GRID_INTERVALS = (16, 32, 64, 128)
FAMILY_CPU = 6500.
FAMILY_GPU = 7200.
ENDPOINT_CHECKS = frozenset({
    "batch_independent_density_identity", "batch_size_value_parity", "bitwise_replay",
    "canonical_value_call_chain_parity", "chart_value_and_jacobian",
    "different_exact_and_finite_endpoint_bindings", "endpoint_independent_of_force_calls",
    "finite_tail_rounding_is_not_support_exclusion", "force_independent_of_endpoint_calls",
    "healthy_values", "nonfinite_rejected", "original_force_healthy", "proposal_involution",
    "rounding_observed", *(f"endpoint_cache_{i}" for i in range(5)),
    *(f"transition_health_{i}" for i in range(5)), *(f"grid_health_{i}" for i in range(3)),
})


def check_endpoint(root, *, selection_sha, posterior_sha):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    manifest = json.loads((root / "run-manifest.json").read_text())
    if (report.get("status") != "QUALIFIED_FINITE_ENDPOINT" or report.get("qualified") is not True
            or set(report.get("checks", {})) != ENDPOINT_CHECKS or not all(v is True for v in report["checks"].values())
            or manifest.get("exit_code") != 0):
        raise ValueError("qualified finite endpoint required")
    if manifest["selection_sha256"] != selection_sha or manifest["posterior_report_sha256"] != posterior_sha:
        raise ValueError("finite endpoint scope mismatch")
    for name, digest in manifest["source_hashes"].items():
        if (name.startswith("bayesfilter/") and name != "bayesfilter/inference/surrogate_hmc_program.py"
                and sha(ROOT / name) != digest):
            raise ValueError("finite endpoint numerical source drift")
    if (report["endpoint_identity"] != manifest["endpoint_identity"]
            or sha(root / "data.json") != manifest["data_sha256"]):
        raise ValueError("finite endpoint identity or data drift")
    for kind in ("cpu_worker_seconds", "gpu_seconds"):
        for metric in ("reference", "C_inference", "grid_batch_max", "transition_L2_max"):
            value = float(report["forecast"][kind][metric])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid complete finite-target forecast")
    return report, manifest


def qualification(out, manifest, tf, selection):
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_consumer
    from bayesfilter.testing.surrogate_hmc_ksc_finite_endpoint import make_finite_endpoint, ENDPOINT_VERSION
    from bayesfilter.testing.surrogate_hmc_ksc_reference import physical_parameters
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, neural_force_proposal, neural_force_hmc_transition,
    )
    data = json.loads((Path(manifest["posterior_root"]) / "data.json").read_text())
    save(out / "data.json", data)
    consumer = make_ksc_consumer(data, controls=selection["controls"], batch_size=4)
    endpoint, physical, chart = make_finite_endpoint(consumer)
    save(out / "consumer.json", consumer.manifest)
    manifest.update(data_sha256=sha(out / "data.json"), endpoint_identity=endpoint.identity,
                    endpoint_version=ENDPOINT_VERSION, force_identity=consumer.force.identity,
                    seeds=dict(data=data["seeds"], fixed_inputs=consumer.fixture.manifest["seeds"],
                               transitions=[[20260921, 28100 + i] for i in range(5)]))
    save(out / "run-manifest.json", manifest)
    report = dict(status="RUNNING", checks={}, timings={}, surrogate_hmc_tested=False,
                  endpoint_identity=endpoint.identity, force_identity=consumer.force.identity,
                  posterior_comparisons={k: "NOT_RUN" for k in ("B/R", "C/D", "D/R", "C/R")})

    def timed(name, callback):
        start, cpu = time.monotonic(), time.process_time()
        result = callback()
        for item in tf.nest.flatten(result):
            item.numpy()
        report["timings"][name] = dict(wall_seconds=time.monotonic()-start,
                                       cpu_worker_seconds=time.process_time()-cpu)
        save(out / "report.json", report)
        return result

    def equal(a, b):
        return all(bool(tf.reduce_all(x == y)) for x, y in zip(tf.nest.flatten(a), tf.nest.flatten(b)))

    checks = report["checks"]
    q = tf.constant([[-.5, -.3], [0., 0.], [.4, -.2], [.8, .6]], tf.float64)
    theta = physical_parameters(q)
    value, status = timed("physical_compile_first", lambda: physical(theta))
    density, chart_status = timed("chart_compile_first", lambda: chart(q))
    jacobian = tf.reduce_sum(tf.math.log(tf.constant(.8, tf.float64)) - .5*q**2
                            - .5*tf.math.log(tf.constant(2*math.pi, tf.float64)), axis=-1)
    checks["chart_value_and_jacobian"] = bool(tf.reduce_all(tf.abs(density-value-jacobian) < 1e-12))
    checks["healthy_values"] = bool(tf.reduce_all(status["value_valid"] & chart_status["value_valid"]))
    direct = consumer.fixture.compile(4)
    direct_value, _, direct_status = timed("direct_canonical", lambda: direct(tf.cast(theta, tf.float32)))
    checks["canonical_value_call_chain_parity"] = equal(value, tf.cast(direct_value, tf.float64)) and equal(status["value_valid"], direct_status["value_valid"])
    checks["bitwise_replay"] = equal((density, chart_status), chart(q))
    baseline_force = timed("force_compile_first", lambda: consumer.evaluate_force(q))
    chart(q + .1)
    checks["force_independent_of_endpoint_calls"] = equal(baseline_force, consumer.evaluate_force(q))
    consumer.evaluate_force(q + .1)
    checks["endpoint_independent_of_force_calls"] = equal((density, chart_status), chart(q))
    checks["original_force_healthy"] = bool(tf.reduce_all(baseline_force[2]["force_valid"]))
    checks["different_exact_and_finite_endpoint_bindings"] = endpoint.identity != consumer.endpoint.identity
    extreme_value, extreme = chart(tf.constant([[-12., -12.], [12., 12.], [-9., 9.], [0., 0.]], tf.float64))
    checks["finite_tail_rounding_is_not_support_exclusion"] = bool(tf.reduce_all(extreme["value_valid"] & tf.math.is_finite(extreme_value))) and not bool(tf.reduce_any(extreme["outside_declared_support"]))
    checks["rounding_observed"] = extreme["coordinate_boundary_rounding"].numpy().tolist() == [True, True, True, False]
    _, invalid = chart(tf.constant([[float("inf"), 0.], [float("nan"), 0.], [-float("inf"), 0.], [0., 0.]], tf.float64))
    checks["nonfinite_rejected"] = invalid["value_valid"].numpy().tolist() == [False, False, False, True]
    save(out / "endpoint-probes.json", dict(q=q, physical=value, density=density, status=chart_status,
                                            tail_value=extreme_value, tail_status=extreme, invalid_status=invalid))
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=2, inverse_mass_diagonal=(1., 1.), dtype="float64")

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)]*2, jit_compile=True, autograph=False)
    def proposal(x, p):
        return neural_force_proposal(x, p, consumer.force, cfg)

    momentum = tf.constant([[.2, -.1], [-.4, .3], [.6, -.5], [-.8, .7]], tf.float64)
    forward = timed("proposal_compile_first", lambda: proposal(q, momentum))
    reverse = proposal(forward.position, forward.momentum)
    reversal = float(tf.maximum(tf.reduce_max(tf.abs(q-reverse.position)), tf.reduce_max(tf.abs(momentum-reverse.momentum))))
    report["reversal_absolute_error"] = reversal
    checks["proposal_involution"] = reversal <= 32*2**-23

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                 tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(x, potential, seed):
        return neural_force_hmc_transition(x, potential, consumer.force, endpoint, cfg, seed)

    state, potential = q, -density
    traces = []
    for i in range(5):
        step = timed(f"transition_{i}", lambda: transition(state, potential, tf.constant([20260921, 28100+i])))
        state, potential = step.position, step.potential
        traces.append(step.trace._asdict())
        checks[f"transition_health_{i}"] = bool(tf.reduce_all(tf.math.is_finite(step.trace.delta_h)
            & ~step.trace.endpoint_status["numerical_failure"] & ~step.trace.force_fallback & ~step.trace.divergence))
        checks[f"endpoint_cache_{i}"] = bool(tf.reduce_all(tf.abs(potential-endpoint.function(state).potential) < 1e-10))
    save(out / "transitions.json", traces)

    grid_consumer = make_ksc_consumer(data, controls=selection["controls"], batch_size=GRID_BATCH)
    grid_endpoint, grid_physical, _ = make_finite_endpoint(grid_consumer)
    checks["batch_independent_density_identity"] = grid_endpoint.identity == endpoint.identity
    grid_theta = tf.tile(theta, [GRID_BATCH//4, 1])
    grid_first = timed("grid_compile_first", lambda: grid_physical(grid_theta))
    checks["batch_size_value_parity"] = bool(tf.reduce_all(tf.abs(grid_first[0][:4]-value) < 2e-5))
    for i in range(3):
        grid_value, grid_status = timed(f"grid_steady_{i}", lambda: grid_physical(grid_theta))
        checks[f"grid_health_{i}"] = bool(tf.reduce_all(grid_status["value_valid"] & tf.math.is_finite(grid_value)))
    # Use worst observed steady cost with a factor of two and compile/host allowance.
    batches = math.ceil((GRID_INTERVALS[-1]+1)**2 / GRID_BATCH)
    forecasts = {}
    for kind, key, overhead in (("cpu_worker_seconds", "cpu_worker_seconds", 60.), ("gpu_seconds", "wall_seconds", 90.)):
        grid_cost = max(report["timings"][f"grid_steady_{i}"][key] for i in range(3))
        transition_cost = max(report["timings"][f"transition_{i}"][key] for i in range(1, 5))
        forecasts[kind] = dict(reference=2*batches*grid_cost+overhead,
                              C_inference=1.5*(3586*transition_cost+240+90),
                              grid_batch_max=grid_cost, transition_L2_max=transition_cost)
    report.update(qualified=all(checks.values()), status="QUALIFIED_FINITE_ENDPOINT" if all(checks.values()) else "QUALIFICATION_FAILED",
                  grid_intervals=GRID_INTERVALS, grid_batch=GRID_BATCH, forecast=forecasts,
                  allocator=tf.config.experimental.get_memory_info("GPU:0"))
    save(out / "report.json", report)
    return 0 if report["qualified"] else 2


def worker(out):
    manifest = json.loads((out / "run-manifest.json").read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise ValueError("one designated GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    manifest.update(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
                    gpu_status=[d.name for d in tf.config.list_logical_devices("GPU")])
    save(out / "run-manifest.json", manifest)
    selection = check_selection(manifest["controls_root"], validate_controls=True)
    return (qualification if manifest["stage"] == "qualify" else reference_preparation)(out, manifest, tf, selection)


def reference_preparation(out, manifest, tf, selection):
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_consumer
    from bayesfilter.testing.surrogate_hmc_ksc_finite_endpoint import make_finite_endpoint
    from bayesfilter.testing.surrogate_hmc_ksc_finite_reference import summarize_finite_grids
    ref_root = Path(manifest["posterior_root"])
    reference = check_reference(ref_root)
    data = json.loads((ref_root / "data.json").read_text())
    endpoint_report, _ = check_endpoint(manifest["endpoint_root"], selection_sha=manifest["selection_sha256"],
                                        posterior_sha=manifest["posterior_report_sha256"])
    consumer = make_ksc_consumer(data, controls=selection["controls"], batch_size=GRID_BATCH)
    endpoint, physical, _ = make_finite_endpoint(consumer)
    if endpoint.identity != endpoint_report["endpoint_identity"]:
        raise ValueError("finite density changed after qualification")
    save(out / "consumer.json", consumer.manifest)
    save(out / "data.json", data)
    manifest.update(endpoint_identity=endpoint.identity, data_sha256=sha(out / "data.json"),
                    seeds=dict(data=data["seeds"], fixed_inputs=consumer.fixture.manifest["seeds"]))
    save(out / "run-manifest.json", manifest)

    def load_tensor(record):
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("reference grid checksum drift")
        return tf.io.parse_tensor(tf.io.read_file(record["path"]), tf.float64)

    last = reference["grids"][-1]
    first, second = load_tensor(last["gamma"]), load_tensor(last["beta"])
    tf.debugging.assert_shapes([(first, (257,)), (second, (257,))])
    tf.debugging.assert_equal(first, second)
    knots = first[::2]
    predictive = load_tensor(last["predictive"])[::2, ::2]
    points = tf.reshape(tf.stack(tf.meshgrid(knots, knots, indexing="ij"), -1), [-1, 2])
    # Trapezoid endpoint values use one-sided limits of the declared density.
    # The adapter then applies its named FP32 interior-rounding rule.
    lo, hi = tf.constant(.1, tf.float64), tf.constant(.9, tf.float64)
    points = tf.clip_by_value(points, tf.math.nextafter(lo, hi), tf.math.nextafter(hi, lo))
    values, roundings, forces = [], [], []
    count_total = int(points.shape[0])
    progress = dict(status="RUNNING", points_total=count_total, points_done=0,
                    surrogate_hmc_tested=False, endpoint_identity=endpoint.identity)
    for offset in range(0, count_total, GRID_BATCH):
        count = min(GRID_BATCH, count_total-offset)
        batch = points[offset:offset+count]
        if count < GRID_BATCH:
            batch = tf.concat([batch, tf.repeat(batch[:1], GRID_BATCH-count, axis=0)], 0)
        value, status = physical(batch)
        if not bool(tf.reduce_all(status["value_valid"] & tf.math.is_finite(value))):
            save(out / "failed-points.json", dict(points=batch, value=value, status=status))
            raise ValueError("finite target invalid on full-prior reference grid")
        values.append(value[:count])
        roundings.append(status["coordinate_boundary_rounding"][:count])
        forces.append(status["force_valid"][:count])
        if offset % (64*GRID_BATCH) == 0:
            progress["points_done"] = offset+count
            save(out / "report.json", progress)
            print("GRID", offset+count, count_total, flush=True)
    log_density = tf.reshape(tf.concat(values, 0), [129, 129])
    report, grids = summarize_finite_grids(knots, log_density, predictive, reference)
    files = []
    for intervals, grid in zip(GRID_INTERVALS, grids):
        directory = out / f"grid-{intervals}"
        directory.mkdir()
        row = {}
        for name, tensor in zip(("gamma", "beta", "density", "predictive"), grid):
            path = directory / (name+".tensor")
            tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
            row[name] = dict(path=str(path), sha256=sha(path))
        files.append(row)
    tf.io.write_file(str(out / "log-density.tensor"), tf.io.serialize_tensor(log_density))
    report.update(grids=files, endpoint_identity=endpoint.identity, points_done=count_total,
        model_id="M03-KSC-T2", selection_sha256=manifest["selection_sha256"], data_sha256=manifest["data_sha256"],
        reference_report_sha256=manifest["posterior_report_sha256"], endpoint_report_sha256=manifest["endpoint_report_sha256"],
        rounded_grid_points=tf.reduce_sum(tf.cast(tf.concat(roundings, 0), tf.int32)),
        force_invalid_grid_points=tf.reduce_sum(tf.cast(~tf.concat(forces, 0), tf.int32)),
        log_density=dict(path=str(out / "log-density.tensor"), sha256=sha(out / "log-density.tensor")),
        allocator=tf.config.experimental.get_memory_info("GPU:0"))
    save(out / "report.json", report)
    return 0 if report["qualified"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "qualification-root", "controls-root", "posterior-root", "endpoint-root", "worker"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--stage", choices=("qualify", "reference"), default="qualify")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--previous-accounting", type=Path, action="append", default=[])
    parser.add_argument("--max-cpu-worker-seconds", type=float, default=54000.)
    parser.add_argument("--max-gpu-seconds", type=float, default=226800.)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if any(getattr(args, key) is None for key in ("output", "qualification_root", "controls_root", "posterior_root")):
        parser.error("output, qualification, controls and posterior roots required")
    if not (math.isfinite(args.max_cpu_worker_seconds) and 0 < args.max_cpu_worker_seconds <= 54000
            and math.isfinite(args.max_gpu_seconds) and 0 < args.max_gpu_seconds <= 226800):
        raise ValueError("invalid or expanded campaign budget")
    started, parent_cpu = time.monotonic(), time.process_time()
    out = args.output.resolve()
    if args.stage == "reference" and args.endpoint_root is None:
        parser.error("finite reference requires endpoint-root")
    prefix = f"surrogate-hmc-all-models-20260921-ksc-t2-finite-{args.stage}-"
    if out.parent != ROOT / "docs/plans/artifacts" or not out.name.startswith(prefix):
        raise ValueError("fresh versioned KSC finite-qualification root required")
    if out.exists():
        raise FileExistsError("preserve prior evidence")
    total, family, _ = budgets(args.previous_accounting)
    prior = dict(cpu_worker_seconds=0., gpu_seconds=0.)
    for path in out.parent.glob(prefix+"*/accounting.json"):
        record = json.loads(path.read_text())
        for key, value in record.get("charged_this_run", record.get("charged")).items():
            prior[key] += value
    cpu_cap, wall_cap = (180., 240.) if args.stage == "qualify" else (1000., 1100.)
    cpu = min(cpu_cap-prior["cpu_worker_seconds"], FAMILY_CPU-family["cpu_worker_seconds"], args.max_cpu_worker_seconds-total["cpu_worker_seconds"])
    wall = min(wall_cap-prior["gpu_seconds"], FAMILY_GPU-family["gpu_seconds"], args.max_gpu_seconds-total["gpu_seconds"])
    if min(cpu, wall) < 10:
        raise ValueError("qualification/family/campaign allocation exhausted")
    provenance = sources()
    verify_qualification(args.qualification_root, provenance)
    check_selection(args.controls_root)
    check_reference(args.posterior_root)
    if args.stage == "reference":
        endpoint_report, _ = check_endpoint(args.endpoint_root, selection_sha=sha(args.controls_root / "selection.json"),
                                            posterior_sha=sha(args.posterior_root / "report.json"))
        for key, reservation, family_cap, global_cap in (("cpu_worker_seconds", cpu, FAMILY_CPU, args.max_cpu_worker_seconds),
                                                       ("gpu_seconds", wall, FAMILY_GPU, args.max_gpu_seconds)):
            forecast = endpoint_report["forecast"][key]
            if forecast["reference"] > reservation:
                raise ValueError("full finite reference forecast exceeds reservation")
            if reservation+forecast["C_inference"] > min(family_cap-family[key], global_cap-total[key]):
                raise ValueError("reference plus complete C inference forecast exceeds remaining budget")
    for name in ("scripts/prepare_surrogate_hmc_ksc_finite.py", "scripts/run_surrogate_hmc_ksc.py",
                 "bayesfilter/testing/surrogate_hmc_ksc_finite_endpoint.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_finite_endpoint.py",
                 "bayesfilter/testing/surrogate_hmc_ksc_finite_reference.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_finite_reference.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_reference.py",
                 "bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py",
                 "bayesfilter/testing/surrogate_hmc_linear_reference.py", PLAN):
        provenance[name] = sha(ROOT / name)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, cpu_budget=cpu, wall_budget=wall, scope="KSC T2 finite target", stage=args.stage)))
        return 0
    out.mkdir()
    for name in provenance:
        destination = out / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    save(out / "executed-source/sha256.json", provenance)
    manifest = dict(command=[sys.executable, *sys.argv], environment=sys.prefix,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=provenance, plan=PLAN, result_file=str(out / "report.json"), stage=args.stage,
        qualification_root=str(args.qualification_root.resolve()), controls_root=str(args.controls_root.resolve()),
        posterior_root=str(args.posterior_root.resolve()), selection_sha256=sha(args.controls_root / "selection.json"),
        posterior_report_sha256=sha(args.posterior_root / "report.json"), cpu_budget=cpu, wall_budget=wall,
        memory_policy="pending trusted GPU configuration", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        data_version="frozen KSC T2 nominal claim partition; independent C endpoint preparation")
    if args.endpoint_root:
        manifest.update(endpoint_root=str(args.endpoint_root.resolve()), endpoint_report_sha256=sha(args.endpoint_root / "report.json"))
    save(out / "run-manifest.json", manifest)
    save(out / "report.json", dict(status="RUNNING", surrogate_hmc_tested=False))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, watchdog = run_bounded_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(out)],
        log_path=out / "worker.log", cpu_budget=cpu-2, wall_budget=wall-2)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    charge = dict(cpu_worker_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu,
                  gpu_seconds=time.monotonic()-started)
    save(out / "accounting.json", dict(charged_this_run=charge, exit_code=code, watchdog=watchdog))
    report = json.loads((out / "report.json").read_text())
    if code and report["status"] == "RUNNING":
        report.update(status="INFRASTRUCTURE_OR_RESOURCE_FAILURE", exit_code=code)
        save(out / "report.json", report)
    manifest = json.loads((out / "run-manifest.json").read_text())
    manifest.update(exit_code=code, wall_seconds=charge["gpu_seconds"], cpu_worker_seconds=charge["cpu_worker_seconds"])
    save(out / "run-manifest.json", manifest)
    if any(sha(ROOT / name) != digest for name, digest in provenance.items()):
        raise ValueError("source drift during endpoint qualification")
    print(json.dumps(dict(status=report["status"], accounting=charge, output=str(out))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
