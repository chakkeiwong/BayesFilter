#!/usr/bin/env python3
"""Bounded M10 claim-endpoint and complete-transition qualification, no inference."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/surrogate-hmc-m10-endpoint-2026-09-22.md"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(ROOT))
from scripts.qualify_surrogate_hmc_nonlinear_consumer import save, serial, sha, snapshot


def sources(extra=()):
    pending = ["scripts/qualify_surrogate_hmc_nonlinear_endpoint.py",
               "bayesfilter/testing/surrogate_hmc_nonlinear_inference_consumer.py"]
    pending.extend(extra)
    found = {}
    while pending:
        name = pending.pop()
        path = ROOT / name
        if name in found or not path.is_file():
            continue
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if node.level:
                    base = Path(name).parent.parts
                    module = ".".join((*base[:len(base)-node.level+1], module))
                names.append(module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for module in names:
                if module.startswith(("bayesfilter.", "scripts.")):
                    relative = module.replace(".", "/")
                    pending.append(relative+".py")
                    pending.append(relative+"/__init__.py")
    found[PLAN] = sha(ROOT/PLAN)
    return found


def complete_forecast(costs, reference_timings):
    """One B arm, public nomination, fresh verification and both retained looks."""
    if set(costs) != {"L1", "L2"}:
        raise ValueError("both complete transition costs required")
    for rows in costs.values():
        if len(rows) != 5 or [r["compilation_included"] for r in rows] != [True, False, False, False, False]:
            raise ValueError("one compilation and four steady costs required")
        for row in rows:
            if not row.get("healthy") or any(not math.isfinite(row[k]) or row[k] <= 0
                    for k in ("cpu_worker_seconds", "wall_seconds")):
                raise ValueError("invalid complete transition cost")
    # Two mass windows, both L1/L2 candidates (each adaptation+verification),
    # 48 conservative extra setup calls, 250 discarded verification warmup,
    # 1000 fresh verification and 2000 retained transitions. The tuner may
    # stop at the first passing candidate; do not assume that in the forecast.
    count = 32+64+2*(128+64)+48+250+1000+2000
    reference_costs = [reference_timings[f"reference_fine_{i}"] for i in range(1, 15)]
    if any(not math.isfinite(row[k]) or row[k] <= 0 for row in reference_costs
           for k in ("cpu_worker_seconds", "wall_seconds")):
        raise ValueError("invalid measured reference functional cost")
    reference = {key: 8000*max(row[key] for row in reference_costs)
                 for key in ("cpu_worker_seconds", "wall_seconds")}
    needed = {key: 1.5*(count*max(row[key] for row in costs["L2"][1:])+reference[key]+240+90)
              for key in ("cpu_worker_seconds", "wall_seconds")}
    return dict(required=needed, transitions_per_four_chain_batch=count,
        max_leapfrog_steps=2, trajectory_candidates=2, safety_factor=1.5, additional_compilation_seconds=240,
        analysis_seconds=90, retained_reference_functional_cost=reference,
        reference_cost_basis="slowest steady single-row value/score/predictive probe times 8000 retained states",
        includes_all_public_tuning_and_retained_stages=True,
        arms=["B"], retained_looks=[1000, 2000], authorizes_inference=False,
        heuristic_dominance="NOT_RUN", C_D_requires_separate_endpoint_and_reference=True)


def worker(out):
    manifest = json.loads((out/"run-manifest.json").read_text())
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT/name) != digest or sha(out/"source"/name) != digest:
            raise ValueError("source changed: "+name)
    for record in manifest["inputs"].values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen input changed")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    devices = [d.name for d in tf.config.list_logical_devices("GPU")]
    if len(devices) != 1:
        raise ValueError("one designated GPU required")
    manifest.update(memory_policy=memory, gpu_status=devices, tensorflow=tf.__version__,
                    jit_compile=True, tf32=True)
    save(out/"run-manifest.json", manifest)
    from bayesfilter.testing.surrogate_hmc_nonlinear_log_reference import make_log_reference
    from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference
    from bayesfilter.testing.surrogate_hmc_nonlinear_inference_consumer import make_inference_consumer
    from bayesfilter.testing.surrogate_hmc_nonlinear_reference import PRIOR_SD
    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, bind_neural_force_hmc_tuning_runner,
        neural_force_proposal, neural_force_hmc_transition)
    claim = json.loads(Path(manifest["inputs"]["claim"]["path"]).read_text())
    calibration = json.loads(Path(manifest["inputs"]["calibration"]["path"]).read_text())
    selection = json.loads(Path(manifest["inputs"]["selection"]["path"]).read_text())
    observations = tf.constant(claim["raw_observations"], tf.float64)
    report = dict(status="RUNNING", checks={}, timings={}, reference_probes=[], complete_transition_costs={},
        qualified=False, retained_inference=False, surrogate_hmc_tested=False,
        heuristic_dominance="NOT_RUN", source_binding=manifest["source_hashes"])

    def persist():
        save(out/"report.json", report)

    def timed(name, operation):
        started, cpu = time.monotonic(), time.process_time()
        result = operation()
        for item in tf.nest.flatten(result):
            if hasattr(item, "numpy"):
                item.numpy()
        report["timings"][name] = dict(wall_seconds=time.monotonic()-started,
                                      cpu_worker_seconds=time.process_time()-cpu)
        persist()
        return result

    def relative(a, b):
        return float(tf.reduce_max(tf.abs(a-b)/(1+tf.abs(b))))

    def equal(a, b):
        return all(bool(tf.reduce_all(x == y)) for x, y in zip(tf.nest.flatten(a), tf.nest.flatten(b)))

    fine, coarse, previous = make_log_reference(32, 128), make_log_reference(20, 128), make_reference(32, 128)
    interior = [(0., 0.), (-1., 1.), (1., -1.), (-3., -3.), (-3., 3.), (3., -3.), (3., 3.)]
    probes = [(a*PRIOR_SD, b*PRIOR_SD) for a, b in interior]
    probes += [(a*PRIOR_SD, b*PRIOR_SD) for a, b in ((-6., -6.), (6., 6.), (-9., 0.), (0., -9.), (9., 0.), (0., 9.))]
    probes += [(-3.3648781186133125, -3.192114385833288), (-6., -6.)]
    report["checks"].update(reference_finite=True, reference_refinement=True, density_domain_parity=True)
    for index, point in enumerate(probes):
        z = tf.constant(point, tf.float64)
        result = timed(f"reference_fine_{index}", lambda: fine(z, observations))
        lower = timed(f"reference_coarse_{index}", lambda: coarse(z, observations))
        delta = {key: relative(result[key], lower[key]) for key in ("log_likelihood", "score", "predictive")}
        row = dict(point=point, fine=serial(result), coarse=serial(lower), refinement_relative=delta,
                   role="refinement_veto" if index < len(interior) else "tail_observability")
        report["checks"]["reference_finite"] &= bool(result["valid"])
        if index < len(interior):
            report["checks"]["reference_refinement"] &= bool(lower["valid"]) and delta["log_likelihood"] <= 1e-6 and delta["score"] <= 1e-5 and delta["predictive"] <= 1e-6
            old = timed(f"reference_old_{index}", lambda: previous(z, observations))
            parity = {key: relative(result[key], old[key]) for key in ("log_likelihood", "score")}
            row["previous_parity_relative"] = parity
            report["checks"]["density_domain_parity"] &= bool(old["valid"]) and max(parity.values()) <= 1e-9
        report["reference_probes"].append(row)
        persist()
    report["checks"]["underflow_log_density_recovered"] = report["reference_probes"][-1]["fine"]["log_likelihood"] < -745.
    point = tf.constant([.15, -.2], tf.float64)
    result = fine(point, observations)
    fd_rows = []
    for step in (1e-4, 5e-5):
        gradient = tf.stack([(fine(point+axis*step, observations)["log_likelihood"]-
                              fine(point-axis*step, observations)["log_likelihood"])/(2*step)
                             for axis in tf.unstack(tf.eye(2, dtype=tf.float64))])
        fd_rows.append(dict(step=step, score=serial(gradient), relative_error=relative(gradient, result["score"])))
    report["finite_differences"] = fd_rows
    report["checks"]["total_reference_derivative"] = max(row["relative_error"] for row in fd_rows) <= 2e-5
    persist()
    if not all(report["checks"].values()):
        report["status"] = "REFERENCE_QUALIFICATION_FAILED"
        persist()
        return 2

    consumer = make_inference_consumer(claim, calibration, controls=selection["controls"], batch_size=4)
    save(out/"consumer.json", consumer.manifest)
    fixture = consumer.fixture
    z = tf.constant([[0., 0.], [.15, -.1], [-.15, .1], [-.3, -.3]], tf.float64)
    baseline = timed("force_compile_first", lambda: consumer.evaluate_force(z))
    replay = timed("force_replay", lambda: consumer.evaluate_force(z))
    report["checks"]["force_replay"] = equal(baseline, replay)
    report["checks"]["force_health"] = bool(tf.reduce_all(baseline[2]["force_valid"] & baseline[2]["value_valid"]))
    report["checks"]["claim_binding"] = equal(fixture.observations, tf.cast(observations, tf.float32)) and fixture.manifest["data_role"] == "claim"
    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float32)], jit_compile=True, autograph=False)
    def direct(q):
        value, score, status = canonical_batch_fused_value_score(
            fixture.model, q, tf.broadcast_to(tf.eye(2)[None], [4, 2, 2]),
            fixture.initial_base, fixture.initial_covariances, fixture.noises,
            fixture.observations, **dict(fixture.controls))
        variance = tf.constant(PRIOR_SD**2, q.dtype)
        prior = -tf.reduce_sum(q*q, -1)/(2*variance)-tf.math.log(2*math.pi*variance)
        return value+prior, score-q/variance, status
    authority = timed("canonical_call_chain", lambda: direct(tf.cast(z, tf.float32)))
    report["checks"]["canonical_call_chain"] = equal(baseline[:2], authority[:2]) and all(equal(baseline[2][k], authority[2][k]) for k in ("value_valid", "force_valid", "value_failure_code", "force_failure_code"))
    perm = tf.constant([2, 0, 3, 1])
    permuted = timed("force_permutation", lambda: consumer.evaluate_force(tf.gather(z, perm)))
    report["checks"]["row_independence"] = all(equal(tf.gather(baseline[i], perm), permuted[i]) for i in (0, 1))
    values, predictive, valid = timed("claim_reference_compile", lambda: consumer.evaluate_reference(z))
    expected = tf.stack([fine(row, observations)["log_likelihood"]-tf.reduce_sum(row**2)/(2*PRIOR_SD**2)-math.log(2*math.pi*PRIOR_SD**2) for row in tf.unstack(z)])
    endpoint = consumer.endpoint.function(z)
    report["checks"]["reference_binding"] = relative(values, expected) <= 1e-9 and equal(endpoint.potential, -values) and bool(tf.reduce_all(valid))
    report["checks"]["independent_force_endpoint"] = consumer.force.identity != consumer.endpoint.identity
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
        target_scope="M10-T2-N1024-claim-inference-endpoint-qualification")
    save(out/"binding.json", dict(binding.payload()))
    report["checks"]["potential_replay"] = equal(endpoint, consumer.endpoint.function(z))
    momentum = tf.constant([[.2, -.1], [-.4, .3], [.6, -.5], [-.8, .7]], tf.float64)
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=2, inverse_mass_diagonal=(1., 1.), dtype="float64")
    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)]*2, jit_compile=True, autograph=False)
    def proposal(q, p):
        return neural_force_proposal(q, p, consumer.force, cfg)
    forward = timed("proposal_compile", lambda: proposal(z, momentum))
    backward = timed("proposal_reverse", lambda: proposal(forward.position, forward.momentum))
    reversal = max(float(tf.reduce_max(tf.abs(backward.position-z))), float(tf.reduce_max(tf.abs(backward.momentum-momentum))))
    report["reversal_absolute_error"] = reversal
    report["checks"]["reversal"] = reversal <= 1e-8
    for length in (1, 2):
        config = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=length, inverse_mass_diagonal=(1., 1.), dtype="float64")
        @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64), tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
        def transition(q, potential, seed):
            return neural_force_hmc_transition(q, potential, consumer.force, consumer.endpoint, config, seed)
        position, potential = z, endpoint.potential
        costs = []
        for index in range(5):
            name = f"L{length}_transition_{index}"
            seed = tf.constant([20260922, 31000+100*length+index], tf.int32)
            result = timed(name, lambda: transition(position, potential, seed))
            trace = result.trace
            delta = trace.final_potential+trace.final_kinetic-trace.initial_potential-trace.initial_kinetic
            healthy = (bool(tf.reduce_all(tf.math.is_finite(result.position))) and
                bool(tf.reduce_all(tf.math.is_finite(trace.delta_h))) and
                not bool(tf.reduce_any(trace.force_fallback | trace.divergence | trace.endpoint_status["numerical_failure"])) and
                relative(delta, trace.delta_h) < 1e-10 and
                relative(result.potential, consumer.endpoint.function(result.position).potential) < 1e-9)
            costs.append(dict(report["timings"][name], compilation_included=index == 0,
                              healthy=healthy, seed=serial(seed), trace=serial(trace._asdict())))
            report["complete_transition_costs"][f"L{length}"] = costs
            position, potential = result.position, result.potential
            persist()
        report["checks"][f"complete_L{length}_transitions"] = all(row["healthy"] for row in costs)
    if all(report["checks"].values()):
        report["forecast"] = complete_forecast(report["complete_transition_costs"], report["timings"])
    report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    report["qualified"] = all(report["checks"].values())
    report["status"] = "QUALIFIED_INFERENCE_ENDPOINT" if report["qualified"] else "ENDPOINT_QUALIFICATION_FAILED"
    persist()
    return 0 if report["qualified"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cpu-budget", type=float, default=360.)
    parser.add_argument("--wall-budget", type=float, default=480.)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    from scripts.prepare_surrogate_hmc_ksc import budgets, run_bounded_worker
    from bayesfilter.inference.surrogate_hmc_program import m10_reference_state, m10_controls_state, m10_posterior_state, m10_endpoint_state
    started, parent_cpu = time.monotonic(), time.process_time()
    reference, controls, posterior = m10_reference_state(ROOT), m10_controls_state(ROOT), m10_posterior_state(ROOT)
    if not all(state.get("qualified") for state in (reference, controls, posterior)):
        raise ValueError("qualified M10 data, controls and full-prior posterior required")
    total = budgets()[0]
    endpoint_state = m10_endpoint_state(ROOT)
    if endpoint_state["qualified"]:
        raise ValueError("M10 endpoint already qualified; reuse its report")
    cpu = min(args.cpu_budget, endpoint_state["cpu_worker_seconds"], reference["family_cpu_remaining"], 54000-total["cpu_worker_seconds"])
    wall = min(args.wall_budget, endpoint_state["wall_seconds"], 226800-total["gpu_seconds"])
    if not all(math.isfinite(x) and x > 5 for x in (cpu, wall)):
        raise ValueError("insufficient finite M10 endpoint allocation")
    reference_root, controls_root, posterior_root = (Path(s["qualified"]["report"]).parent for s in (reference, controls, posterior))
    data_root = Path(json.loads((posterior_root/"run-manifest.json").read_text())["data_file"]).parent.parent
    paths = dict(claim=data_root/"claim-nominal/data.json", calibration=data_root/"calibration-nominal/data.json",
        selection=controls_root/"selection.json", posterior_report=posterior_root/"report.json", reference_report=reference_root/"report.json")
    inputs = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
    if args.dry_run:
        print(json.dumps(dict(inputs=inputs, cpu_budget=cpu, wall_budget=wall, inference=False)))
        return 0
    if not ("-m10-endpoint-" in out.name or out.name == "P3-M10-endpoint"):
        raise ValueError("versioned M10 endpoint output required")
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = sources()
    snapshot(out, source_hashes)
    manifest = dict(schema="surrogate_hmc_m10_endpoint_v1", command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=sys.prefix, plan=PLAN, result_file=str(out/"report.json"), inputs=inputs,
        data_version="M10 original nominal claim, once rounded FP32", source_hashes=source_hashes,
        seeds=dict(transitions=[[20260922, 31000+100*length+i] for length in (1, 2) for i in range(5)],
                   fixed_banks=[[20260920, i] for i in (16220, 16221, 16222)]),
        cpu_budget=cpu, wall_budget=wall, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_status="pending trusted GPU initialization", memory_policy="not initialized",
        retained_inference=False, previous_campaign_charge=total)
    save(out/"run-manifest.json", manifest)
    save(out/"report.json", dict(status="RUNNING", qualified=False, retained_inference=False))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, watchdog = run_bounded_worker([sys.executable, __file__, "--worker", "--output", str(out)],
        log_path=out/"worker.log", cpu_budget=cpu-(time.process_time()-parent_cpu)-2,
        wall_budget=wall-(time.monotonic()-started)-1)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_used = time.monotonic()-started
    cpu_used = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu
    charge = dict(cpu_worker_seconds=cpu_used, gpu_seconds=wall_used)
    save(out/"accounting.json", dict(charged_this_run=charge, wall_seconds=wall_used, exit_code=code, watchdog=watchdog))
    manifest = json.loads((out/"run-manifest.json").read_text())
    manifest.update(wall_seconds=wall_used, cpu_worker_seconds=cpu_used, exit_code=code, watchdog=watchdog)
    save(out/"run-manifest.json", manifest)
    report = json.loads((out/"report.json").read_text())
    if code and report["status"] == "RUNNING":
        report.update(status="EXECUTION_FAILED", qualified=False, exit_code=code, watchdog=watchdog)
        save(out/"report.json", report)
    print(json.dumps(dict(status=report["status"], output=str(out), charged=charge, exit_code=code)))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
