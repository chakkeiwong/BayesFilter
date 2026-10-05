#!/usr/bin/env python3
"""Bounded actual-SV claim endpoint qualification and complete B/R cost."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/surrogate-hmc-sv20-endpoint-2026-09-22.md"
PROBES = tuple((a, b) for a in (.11, .5, .89) for b in (.11, .5, .89))
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(ROOT))
from scripts.qualify_surrogate_hmc_nonlinear_consumer import save, serial, sha, snapshot


def sources(extra=()):
    pending, found = ["scripts/qualify_surrogate_hmc_sv_endpoint.py", *extra], {}
    while pending:
        name = pending.pop()
        path = ROOT/name
        if name in found or not path.is_file():
            continue
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if node.level:
                    base = Path(name).parent.parts
                    module = ".".join((*base[:len(base)-node.level+1], module))
                modules.append(module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module.startswith(("bayesfilter.", "scripts.")):
                    pending.extend((module.replace(".", "/")+".py", module.replace(".", "/")+"/__init__.py"))
    found[PLAN] = sha(ROOT/PLAN)
    return found


def complete_forecast(transitions, predictive):
    """Full cost of one prospectively restricted L1 B arm, never admission."""
    for rows in (transitions, predictive):
        if (len(rows) != 5 or [r["compilation_included"] for r in rows] != [True, False, False, False, False]
                or any(not row.get("healthy") or any(not math.isfinite(row[k]) or row[k] <= 0
                    for k in ("cpu_worker_seconds", "wall_seconds")) for row in rows)):
            raise ValueError("one compilation and four positive healthy steady costs required")
    count = 32+64+128+64+48+250+1000+2000
    needed = {key: 1.5*(count*max(row[key] for row in transitions[1:])
        +256*max(row[key] for row in predictive[1:])+240+90)
        for key in ("cpu_worker_seconds", "wall_seconds")}
    return dict(required=needed, transitions_per_four_chain_batch=count,
        trajectory_candidates=1, max_leapfrog_steps=1, predictive_batches=256,
        predictive_batch_size=32, safety_factor=1.5, compilation_allowance_seconds=240,
        analysis_allowance_seconds=90, scope="one public-tuned B arm; no C arm",
        admission=False)


def inputs():
    """Verify prior evidence against its data and unchanged numerical sources."""
    base = ROOT/"docs/plans/artifacts"
    roots = {kind: base/f"surrogate-hmc-all-models-20260920-sv20-{kind}-01"/f"P3-SV20-{child}"
             for kind, child in (("reference", "reference"), ("posterior", "posterior"),
                                 ("consumer", "qualify"), ("controls", "controls"))}
    paths = {}
    manifests = {}
    for kind, path in roots.items():
        report = json.loads((path/"report.json").read_text())
        if kind != "controls" and (report.get("qualified") is not True or not all(report["checks"].values())):
            raise ValueError("unqualified SV evidence: "+kind)
        if kind == "controls" and report.get("verdict") != "SELECTED_FOR_PROPOSAL_EXPERIMENT":
            raise ValueError("unselected SV proposal controls")
        manifest = json.loads((path/"run-manifest.json").read_text())
        for name, digest in manifest["source_hashes"].items():
            if sha(ROOT/name) != digest:
                raise ValueError("SV numerical source drift: "+name)
        manifests[kind] = manifest
        paths[kind+"_report"] = path/"report.json"
        paths[kind+"_manifest"] = path/"run-manifest.json"
    reference_hash = sha(paths["reference_manifest"])
    if (manifests["posterior"]["parent_manifest_sha256"] != reference_hash
            or manifests["consumer"]["parent_manifest_sha256"] != reference_hash
            or manifests["controls"]["reference_manifest_sha256"] != reference_hash
            or manifests["controls"]["qualification_manifest_sha256"] != sha(paths["consumer_manifest"])):
        raise ValueError("SV evidence parent mismatch")
    paths["claim"] = roots["reference"]/"claim/data.json"
    if (sha(paths["claim"]) != manifests["posterior"]["data_sha256"]
            or sha(paths["claim"]) != sha(roots["posterior"]/"data.json")):
        raise ValueError("claim/posterior data mismatch")
    paths["selection"] = roots["controls"]/"selection.json"
    paths["scope"] = roots["controls"]/"scope.json"
    selection = json.loads(paths["selection"].read_text())
    if selection["source_hashes"] != manifests["controls"]["source_hashes"]:
        raise ValueError("selection/source mismatch")
    if sha(paths["claim"]) in {row["sha256"] for row in selection["partition_data"].values()}:
        raise ValueError("claim observations reused for control selection")
    return {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}


def remaining(root=ROOT):
    """Two attempts share the single qualification reservation, including failures."""
    from bayesfilter.inference.surrogate_hmc_program import sv20_family_remaining_allocation
    paths = sorted(root.glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json"))
    family = sv20_family_remaining_allocation(dict(prior_runs=[dict(path=str(p)) for p in paths]))
    cpu, wall, attempts = 450., 600., 0
    for path in paths:
        if "-sv20-endpoint-" in path.parent.name:
            payload = json.loads(path.read_text())
            charge = payload["charged_this_run"]
            cpu -= charge["cpu_worker_seconds"]
            wall -= charge["gpu_seconds"]
            attempts += 1
    if attempts >= 2:
        cpu = wall = 0.
    return min(cpu, family["cpu_worker_seconds"]), min(wall, family["gpu_seconds"])


def worker(out):
    manifest = json.loads((out/"run-manifest.json").read_text())
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT/name) != digest or sha(out/"source"/name) != digest:
            raise ValueError("executed source drift: "+name)
    for record in manifest["inputs"].values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen input drift")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    devices = [d.name for d in tf.config.list_logical_devices("GPU")]
    if len(devices) != 1:
        raise ValueError("one designated GPU required")
    manifest.update(memory_policy=memory, gpu_status=devices, tensorflow=tf.__version__, tf32=True, jit_compile=True)
    save(out/"run-manifest.json", manifest)
    from bayesfilter.testing.surrogate_hmc_sv_inference_consumer import make_sv_inference_consumer
    from bayesfilter.testing.surrogate_hmc_sv_reference import chart_and_log_prior, make_sv_reference
    from bayesfilter.testing.surrogate_hmc_sv_posterior_reference import make_statistics
    from bayesfilter.inference.surrogate_hmc_dctr_selection import load_selection
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, bind_neural_force_hmc_tuning_runner,
        neural_force_proposal, neural_force_hmc_transition)
    records = manifest["inputs"]
    def read(key):
        return json.loads(Path(records[key]["path"]).read_text())
    scope = read("scope")
    if (scope["model"], scope["horizon"], scope["attempt"], scope["particle_count"], scope["chunk_extent"]) != ("SV20", 20, 1, 1024, 1024):
        raise ValueError("wrong SV20 control scope")
    selection = load_selection(records["selection"]["path"], expected_scope=scope,
                               expected_source_hashes=read("controls_manifest")["source_hashes"])
    data = read("claim")
    consumer = make_sv_inference_consumer(data, controls=selection["controls"], batch_size=4)
    save(out/"consumer.json", consumer.manifest)
    save(out/"data.json", data)
    report = dict(status="RUNNING", checks={}, timings={}, reference_probes=[],
        complete_transition_costs=[], predictive_costs=[], qualified=False,
        retained_inference=False, surrogate_hmc_tested=False, heuristic_dominance="NOT_RUN",
        posterior_comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")})
    def persist():
        save(out/"report.json", report)
    def timed(name, callback):
        start, cpu = time.monotonic(), time.process_time()
        result = callback()
        for value in tf.nest.flatten(result):
            if hasattr(value, "numpy"):
                value.numpy()
        report["timings"][name] = dict(cpu_worker_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start)
        report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        persist()
        if report["allocator"]["peak"] > 4*1024**3:
            raise ValueError("4 GiB allocation limit exceeded")
        print(name, report["timings"][name], flush=True)
        return result
    def equal(left, right):
        tf.nest.assert_same_structure(left, right)
        return all(bool(tf.reduce_all(a == b)) for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)))
    def coordinates(physical):
        return tf.sqrt(tf.constant(2., tf.float64))*tf.math.erfinv(2*(physical-.1)/.8-1)
    observations = tf.cast(tf.constant(data["raw_observations"], tf.float32), tf.float64)
    fine = make_sv_reference(intervals=1280, half_width=20., batch_size=4)
    observation_hash = hashlib.sha256(tf.io.serialize_tensor(observations).numpy()).hexdigest()
    report["checks"].update(claim_data_bound=consumer.manifest["endpoint"]["raw_observations_sha256"] == observation_hash,
                              reference_refinement=True, domain_mass=True,
                              unchanged_value_and_score=True)
    for offset in range(0, len(PROBES), 4):
        points = list(PROBES[offset:offset+4])
        points += [points[-1]]*(4-len(points))
        physical = tf.constant(points, tf.float64)
        z = coordinates(physical)
        value, score, loss, valid = timed(f"endpoint_probe_{offset}", lambda: consumer.evaluate_reference_diagnostics(z))
        old = timed(f"unguarded_probe_{offset}", lambda: consumer.evaluate_reference(z))
        # Use the same chart-rounded physical values as the endpoint.
        theta, derivative, prior = chart_and_log_prior(z)
        refined = timed(f"refined_probe_{offset}", lambda: fine.physical_value_score(theta, observations))
        value_error = float(tf.reduce_max(tf.abs(value-prior-refined["log_likelihood"])))
        refined_score = refined["score"]*derivative-z
        score_error = float(tf.reduce_max(tf.abs(score-refined_score)/(1+tf.abs(refined_score))))
        domain_ok = bool(tf.reduce_all(valid & tf.math.is_finite(refined["maximum_predicted_domain_loss"])
                                      & (refined["maximum_predicted_domain_loss"] <= 1e-9)))
        report["checks"]["reference_refinement"] &= math.isfinite(value_error+score_error) and value_error <= 1e-6 and score_error <= 1e-5
        report["checks"]["domain_mass"] &= domain_ok
        report["checks"]["unchanged_value_and_score"] &= equal((value, score), old)
        report["reference_probes"].append(serial(dict(physical=physical, value=value, score=score,
            domain_loss=loss, refined=refined, value_error=value_error, relative_score_error=score_error)))
        persist()
    z = coordinates(tf.constant([[.3, .3], [.5, .5], [.7, .7], [.8, .2]], tf.float64))
    baseline = timed("force_compile", lambda: consumer.evaluate_force(z))
    report["checks"]["force_healthy"] = bool(tf.reduce_all(baseline[2]["force_valid"]))
    report["checks"]["bitwise_replay"] = equal(baseline, timed("force_replay", lambda: consumer.evaluate_force(z)))
    direct = consumer.fixture.compile(4, coordinates="unconstrained")
    report["checks"]["canonical_call_chain"] = equal(baseline, timed("direct_fixture", lambda: direct(tf.cast(z, tf.float32))))
    permutation = tf.constant([2, 0, 3, 1])
    permuted = timed("permuted_force", lambda: consumer.evaluate_force(tf.gather(z, permutation)))
    report["checks"]["permutation"] = equal(tf.nest.map_structure(lambda x: tf.gather(x, permutation), baseline), permuted)
    changed = tf.tensor_scatter_nd_add(z, [[3, 0]], tf.constant([.1], tf.float64))
    changed_result = timed("changed_force", lambda: consumer.evaluate_force(changed))
    report["checks"]["row_independence"] = equal(tf.nest.map_structure(lambda x: x[:3], baseline),
                                                   tf.nest.map_structure(lambda x: x[:3], changed_result))
    endpoint = timed("endpoint_binding", lambda: consumer.endpoint.function(z))
    consumer.force.function(changed)
    report["checks"]["endpoint_independent"] = equal(endpoint, consumer.endpoint.function(z))
    field = consumer.force.function(z)
    report["checks"]["field_sign_status"] = equal((field.force, field.force_valid),
        (-tf.cast(baseline[1], tf.float64), baseline[2]["force_valid"]))
    logp, score, _, _ = consumer.evaluate_reference_diagnostics(z)
    report["checks"]["endpoint_sign"] = bool(tf.reduce_all(endpoint.potential == -logp))
    report["checks"]["reference_force_sign"] = bool(tf.reduce_all(consumer.exact_force.function(z).force == -score))
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
        target_scope="SV20-attempt1-T20-N1024-claim-endpoint-qualification")
    save(out/"binding.json", dict(binding.payload()))
    report["checks"]["independent_typed_bindings"] = consumer.force.identity != consumer.endpoint.identity
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=1, inverse_mass_diagonal=(1., 1.), dtype="float64")
    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)]*2, jit_compile=True, autograph=False)
    def proposal(q, p):
        return neural_force_proposal(q, p, consumer.force, cfg)
    momentum = tf.constant([[.2, -.1], [-.4, .3], [.6, -.5], [-.8, .7]], tf.float64)
    forward = timed("proposal_forward", lambda: proposal(z, momentum))
    backward = timed("proposal_reverse", lambda: proposal(forward.position, forward.momentum))
    error = float(tf.maximum(tf.reduce_max(tf.abs(backward.position-z)), tf.reduce_max(tf.abs(backward.momentum-momentum))))
    report["reversal_absolute_error"] = error
    report["checks"]["proposal_reversible"] = error <= 2e-6 and not bool(tf.reduce_any(
        forward.force_fallback | backward.force_fallback | forward.divergence | backward.divergence))
    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                  tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(q, u, seed):
        return neural_force_hmc_transition(q, u, consumer.force, consumer.endpoint, cfg, seed)
    q, potential = z, endpoint.potential
    for i in range(5):
        step = timed(f"complete_transition_{i}", lambda: transition(q, potential, tf.constant([20260922, 34100+i], tf.int32)))
        healthy = (bool(tf.reduce_all(tf.math.is_finite(step.position)))
            and bool(tf.reduce_all(tf.math.is_finite(step.trace.delta_h)))
            and bool(tf.reduce_all(step.trace.finite_status))
            and not bool(tf.reduce_any(step.trace.endpoint_status["numerical_failure"]
                | step.trace.force_fallback | step.trace.divergence)))
        report["complete_transition_costs"].append(dict(report["timings"][f"complete_transition_{i}"],
            compilation_included=i == 0, healthy=healthy, trace=serial(step.trace._asdict())))
        q, potential = step.position, step.potential
        persist()
        if not healthy:
            break
    statistics = make_statistics(intervals=1024, half_width=16., batch_size=32)
    theta = tf.tile(chart_and_log_prior(q)[0], [8, 1])
    for i in range(5):
        stats = timed(f"predictive_batch_{i}", lambda: statistics(theta, observations))
        healthy = all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in stats)
        healthy &= bool(tf.reduce_all((stats[1] >= 0) & (stats[1] <= 1))) and bool(tf.reduce_all(stats[2] <= 1e-9))
        report["predictive_costs"].append(dict(report["timings"][f"predictive_batch_{i}"], compilation_included=i == 0, healthy=healthy))
    report["checks"]["complete_transitions"] = len(report["complete_transition_costs"]) == 5 and all(r["healthy"] for r in report["complete_transition_costs"])
    report["checks"]["predictive_diagnostics"] = all(r["healthy"] for r in report["predictive_costs"])
    if report["checks"]["complete_transitions"] and report["checks"]["predictive_diagnostics"]:
        report["forecast"] = complete_forecast(report["complete_transition_costs"], report["predictive_costs"])
    report["qualified"] = all(report["checks"].values())
    report["status"] = "QUALIFIED_INFERENCE_ENDPOINT" if report["qualified"] else "ENDPOINT_QUALIFICATION_FAILED"
    persist()
    return 0 if report["qualified"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-budget", type=float, default=450.)
    parser.add_argument("--wall-budget", type=float, default=600.)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    if not all(math.isfinite(x) and x > 0 for x in (args.cpu_budget, args.wall_budget)):
        raise ValueError("finite positive resource limits required")
    from scripts.prepare_surrogate_hmc_ksc import budgets, run_bounded_worker
    start, parent_cpu = time.monotonic(), time.process_time()
    records = inputs()
    total = budgets()[0]
    cpu, wall = remaining()
    cpu, wall = min(cpu, args.cpu_budget, 54000-total["cpu_worker_seconds"]), min(wall, args.wall_budget, 226800-total["gpu_seconds"])
    if not all(math.isfinite(x) and x > 5 for x in (cpu, wall)):
        raise ValueError("insufficient finite endpoint allocation")
    if args.dry_run:
        print(json.dumps(dict(cpu_budget=cpu, wall_budget=wall, inputs=records, retained_inference=False)))
        return 0
    if "-sv20-endpoint-" not in out.name:
        raise ValueError("fresh versioned SV20 endpoint output required")
    out.mkdir(parents=True, exist_ok=False)
    hashes = sources()
    snapshot(out, hashes)
    manifest = dict(schema="surrogate_hmc_sv20_endpoint_v1", command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=sys.prefix, plan=PLAN, result_file=str(out/"report.json"), inputs=records,
        data_version="actual SV20 original attempt1 claim, once rounded FP32", source_hashes=hashes,
        seeds=dict(transitions=[[20260922, 34100+i] for i in range(5)],
                   fixed_banks=[[20260920, i] for i in (12320, 12321, 12322)]),
        cpu_budget=cpu, wall_budget=wall, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_status="pending trusted GPU initialization", memory_policy="not initialized",
        retained_inference=False, previous_campaign_charge=total)
    save(out/"run-manifest.json", manifest)
    save(out/"report.json", dict(status="RUNNING", qualified=False, retained_inference=False))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, watchdog = run_bounded_worker([sys.executable, __file__, "--worker", "--output", str(out)],
        log_path=out/"worker.log", cpu_budget=cpu-(time.process_time()-parent_cpu)-2,
        wall_budget=wall-(time.monotonic()-start)-1)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.monotonic()-start
    charge = dict(cpu_worker_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu,
                  gpu_seconds=elapsed)
    drift = [name for name, digest in hashes.items() if sha(ROOT/name) != digest]
    if drift:
        code = 2
    save(out/"accounting.json", dict(charged_this_run=charge, wall_seconds=elapsed, exit_code=code, watchdog=watchdog))
    manifest = json.loads((out/"run-manifest.json").read_text())
    manifest.update(wall_seconds=elapsed, cpu_worker_seconds=charge["cpu_worker_seconds"], exit_code=code, watchdog=watchdog, source_drift=drift)
    save(out/"run-manifest.json", manifest)
    report = json.loads((out/"report.json").read_text())
    if code:
        report.update(status="SOURCE_DRIFT" if drift else "ENDPOINT_QUALIFICATION_FAILED", qualified=False, exit_code=code, watchdog=watchdog)
        save(out/"report.json", report)
    print(json.dumps(dict(status=report["status"], output=str(out), charged=charge, exit_code=code)))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
