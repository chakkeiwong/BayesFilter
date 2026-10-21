#!/usr/bin/env python3
"""Bounded LG2 claim-consumer qualification and complete prospective B/R cost."""
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
PLAN = "docs/plans/surrogate-hmc-lg2-endpoint-2026-09-22.md"
PROBES = ((0., 0.), (-2., -2.), (-2., 2.), (2., -2.), (2., 2.),
          (-4., 0.), (4., 0.), (0., -4.), (0., 4.))
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(ROOT))
from scripts.qualify_surrogate_hmc_nonlinear_consumer import save, serial, sha, snapshot


def sources():
    pending, found = ["scripts/qualify_surrogate_hmc_linear_endpoint.py"], {}
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


def complete_forecast(transitions, diagnostics):
    """Full restricted L1 workload, using the slowest healthy steady timings."""
    for rows in (transitions, diagnostics):
        if (len(rows) != 5 or [r["compilation_included"] for r in rows] != [True, False, False, False, False]
                or any(not row.get("healthy") or any(not math.isfinite(row[k]) or row[k] <= 0
                    for k in ("cpu_worker_seconds", "wall_seconds")) for row in rows)):
            raise ValueError("one compilation and four positive healthy steady costs required")
    count = 32+64+128+64+48+250+1000+2000
    compilation = {key: max(240., sum(max(0., rows[0][key]-max(row[key] for row in rows[1:]))
        for rows in (transitions, diagnostics)))
        for key in ("cpu_worker_seconds", "wall_seconds")}
    needed = {key: 1.5*(count*max(row[key] for row in transitions[1:])
        +8*max(row[key] for row in diagnostics[1:])+compilation[key]+90)
        for key in ("cpu_worker_seconds", "wall_seconds")}
    return dict(required=needed, transitions_per_four_chain_batch=count,
        trajectory_candidates=1, max_leapfrog_steps=1, retained_diagnostic_chunks=8,
        diagnostic_positions_per_chunk=1000, safety_factor=1.5,
        compilation_allowance_seconds=compilation, minimum_compilation_allowance_seconds=240,
        analysis_allowance_seconds=90,
        scope="one public-tuned B arm; no C arm", admission=False)


def inputs():
    base = ROOT/"docs/plans/artifacts"
    roots = {kind: base/f"surrogate-hmc-all-models-20260920-lg2-{kind}-01"/child
             for kind, child in (("reference", "reference"), ("consumer", "qualification"),
                                 ("controls", "P3-LG2-controls"))}
    paths, manifests = {}, {}
    for kind, path in roots.items():
        manifest = json.loads((path/"run-manifest.json").read_text())
        for name, digest in manifest["source_hashes"].items():
            if sha(ROOT/name) != digest:
                # The earlier mechanics report predates the issuer's model
                # validation repair; current selected controls postdate it.
                archived = path/"executed-source"/name
                if (kind != "consumer" or name != "bayesfilter/inference/surrogate_hmc_dctr_selection.py"
                        or not archived.is_file() or sha(archived) != digest):
                    raise ValueError("LG2 evidence source drift: "+name)
                old_tree, new_tree = ast.parse(archived.read_text()), ast.parse((ROOT/name).read_text())
                functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                  and node.name == "protection_controls") for tree in (old_tree, new_tree)]
                if ast.dump(functions[0], include_attributes=False) != ast.dump(functions[1], include_attributes=False):
                    raise ValueError("prior consumer control computation changed")
                paths["consumer_archived_selection_source"] = archived
        manifests[kind] = manifest
        paths[kind+"_manifest"] = path/"run-manifest.json"
    paths["claim"] = roots["reference"]/"claim/data.json"
    paths["reference"] = roots["reference"]/"claim/reference.json"
    reference = json.loads(paths["reference"].read_text())
    claim = json.loads(paths["claim"].read_text())
    if ((claim["model_id"], claim["horizon"], claim["attempt"], claim["partition"], claim["replicate"])
            != ("LG2", 20, 1, "claim", 0)):
        raise ValueError("wrong original LG2 claim")
    if (reference.get("qualified") is not True or not all(reference["refinement_passed"])
            or not all(reference["precision_passed"]) or reference["data_sha256"] != sha(paths["claim"])
            or manifests["reference"]["partitions"]["claim"]["data_sha256"] != sha(paths["claim"])):
        raise ValueError("unqualified or mismatched LG2 claim reference")
    for kind in ("consumer", "controls"):
        paths[kind+"_report"] = roots[kind]/"report.json"
    consumer = json.loads(paths["consumer_report"].read_text())
    controls = json.loads(paths["controls_report"].read_text())
    if consumer.get("qualified") is not True or not all(consumer["checks"].values()):
        raise ValueError("unqualified prior LG2 consumer")
    if controls.get("verdict") != "SELECTED_FOR_PROPOSAL_EXPERIMENT":
        raise ValueError("unselected LG2 proposal controls")
    if Path(manifests["controls"]["reference_manifest"]).resolve() != paths["reference_manifest"]:
        raise ValueError("wrong control reference parent")
    paths["selection"] = roots["controls"]/"selection.json"
    paths["scope"] = roots["controls"]/"scope.json"
    selection = json.loads(paths["selection"].read_text())
    if (selection["source_hashes"] != manifests["controls"]["source_hashes"]
            or selection["selected_multiplier"] != .01):
        raise ValueError("LG2 selection/source mismatch")
    for kind in ("calibration", "validation"):
        row = selection["partition_data"][kind]
        path = Path(row["path"])
        if sha(path) != row["sha256"] or sha(path) != manifests["reference"]["partitions"][kind]["data_sha256"]:
            raise ValueError("LG2 selection data drift")
        if sha(paths["claim"]) == row["sha256"]:
            raise ValueError("claim reused for selection")
        paths[kind+"_data"] = path
    return {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}


def remaining(root=ROOT):
    """Charge all LG2 preparation and tests; do not reset the family allocation."""
    from bayesfilter.inference.surrogate_hmc_linear_program import lg2_family_budget
    family, attempts = lg2_family_budget(root)
    if attempts:
        return 0., 0.
    return min(350., family["cpu_worker_seconds"]), min(450., family["gpu_seconds"])


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
    from bayesfilter.testing.surrogate_hmc_linear_consumer import make_linear_consumer
    from bayesfilter.testing.surrogate_hmc_linear_inference_metrics import make_functionals
    from bayesfilter.inference.surrogate_hmc_dctr_selection import load_selection
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, bind_neural_force_hmc_tuning_runner,
        neural_force_proposal, neural_force_hmc_transition)
    records = manifest["inputs"]
    def read(key):
        return json.loads(Path(records[key]["path"]).read_text())
    scope = read("scope")
    if (scope["model"], scope["horizon"], scope["attempt"], scope["particle_count"], scope["chunk_extent"]) != ("LG2", 20, 1, 1024, 1024):
        raise ValueError("wrong LG2 control scope")
    selection = load_selection(records["selection"]["path"], expected_scope=scope,
                               expected_source_hashes=read("controls_manifest")["source_hashes"])
    data, reference = read("claim"), read("reference")
    consumer = make_linear_consumer(data, controls=selection["controls"], batch_size=4)
    save(out/"consumer.json", consumer.manifest)
    save(out/"data.json", data)
    report = dict(status="RUNNING", checks={}, timings={}, reference_probes=[],
        complete_transition_costs=[], diagnostic_costs=[], qualified=False,
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
    observations = tf.cast(tf.constant(data["observations"], tf.float32), tf.float64)
    # The reference uses exact FP64 prior bounds; the force's FP32 chart is a
    # separate proposal approximation. Do not inherit rounded force bounds.
    lower = tf.constant(consumer.reference.specification["lower"], tf.float64)
    upper = tf.constant(consumer.reference.specification["upper"], tf.float64)
    probes = list(PROBES)
    probes += [probes[-1]]*(-len(probes) % 4)
    for offset in range(0, len(probes), 4):
        z = tf.constant(probes[offset:offset+4], tf.float64)
        logp, score = timed(f"endpoint_probes_{offset}", lambda: consumer.evaluate_reference(z))
        theta = lower+(upper-lower)*tf.math.sigmoid(z)
        likelihood = consumer.reference.physical_log_likelihood(theta, observations)
        expected = likelihood-tf.reduce_sum(tf.nn.softplus(z)+tf.nn.softplus(-z), axis=1)
        field = timed(f"force_probes_{offset}", lambda: consumer.evaluate_force(z))
        healthy = (bool(tf.reduce_all(tf.math.is_finite(logp)))
                   and bool(tf.reduce_all(tf.math.is_finite(score)))
                   and bool(tf.reduce_all(field[2]["force_valid"]))
                   and bool(tf.reduce_all(tf.abs(logp-expected) <= 1e-10)))
        report["reference_probes"].append(dict(positions=serial(z), logp=serial(logp),
            max_chart_difference=float(tf.reduce_max(tf.abs(logp-expected))), healthy=healthy))
    report["checks"]["actual_claim_probes"] = all(row["healthy"] for row in report["reference_probes"])
    z = tf.constant(PROBES[1:5], tf.float64)
    baseline = timed("force_baseline", lambda: consumer.evaluate_force(z))
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
    logp, score = consumer.evaluate_reference(z)
    report["checks"]["endpoint_sign"] = bool(tf.reduce_all(endpoint.potential == -logp))
    report["checks"]["reference_force_sign"] = bool(tf.reduce_all(consumer.exact_force.function(z).force == -score))
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
        target_scope="LG2-attempt1-T20-N1024-claim-endpoint-qualification")
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
    tolerance = 32*2**-23*max(1., float(tf.reduce_max(tf.abs(z))), float(tf.reduce_max(tf.abs(momentum))))
    report.update(reversal_absolute_error=error, reversal_tolerance=tolerance)
    report["checks"]["proposal_reversible"] = error <= tolerance and not bool(tf.reduce_any(
        forward.force_fallback | backward.force_fallback | forward.divergence | backward.divergence))
    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                  tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(q, u, seed):
        return neural_force_hmc_transition(q, u, consumer.force, consumer.endpoint, cfg, seed)
    q, potential = z, endpoint.potential
    for i in range(5):
        step = timed(f"complete_transition_{i}", lambda: transition(q, potential, tf.constant([20260922, 37100+i], tf.int32)))
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
    metrics = make_functionals(data, reference)
    sample = tf.reshape(tf.tile(tf.constant(PROBES, tf.float64), [112, 1])[:1000], [250, 4, 2])
    for i in range(5):
        stats = timed(f"diagnostic_chunk_{i}", lambda: metrics(sample))
        healthy = bool(tf.reduce_all(stats[2])) and bool(tf.reduce_all(tf.math.is_finite(stats[1])))
        report["diagnostic_costs"].append(dict(report["timings"][f"diagnostic_chunk_{i}"], compilation_included=i == 0, healthy=healthy))
    parameters, values, valid = stats
    for offset in range(0, len(probes), 4):
        theta = lower+(upper-lower)*tf.math.sigmoid(tf.constant(probes[offset:offset+4], tf.float64))
        _, predictions = consumer.reference.physical_statistics(theta, observations)
        count = min(4, len(PROBES)-offset)
        actual = tf.reshape(values, [1000, 13])[offset:offset+count, -2:]
        report["checks"][f"predictive_parity_{offset}"] = bool(tf.reduce_all(tf.abs(actual-predictions[:count]) <= 1e-11))
    report["checks"]["complete_transitions"] = len(report["complete_transition_costs"]) == 5 and all(r["healthy"] for r in report["complete_transition_costs"])
    report["checks"]["retained_diagnostics"] = all(r["healthy"] for r in report["diagnostic_costs"])
    if report["checks"]["complete_transitions"] and report["checks"]["retained_diagnostics"]:
        report["forecast"] = complete_forecast(report["complete_transition_costs"], report["diagnostic_costs"])
    report["qualified"] = all(report["checks"].values())
    report["status"] = "QUALIFIED_INFERENCE_ENDPOINT" if report["qualified"] else "ENDPOINT_QUALIFICATION_FAILED"
    persist()
    return 0 if report["qualified"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-budget", type=float, default=350.)
    parser.add_argument("--wall-budget", type=float, default=450.)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
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
        raise ValueError("insufficient finite LG2 endpoint allocation")
    if args.dry_run:
        print(json.dumps(dict(cpu_budget=cpu, wall_budget=wall, inputs=records, retained_inference=False)))
        return 0
    if "-lg2-endpoint-" not in out.name:
        raise ValueError("fresh versioned LG2 endpoint output required")
    out.mkdir(parents=True, exist_ok=False)
    hashes = sources()
    snapshot(out, hashes)
    manifest = dict(schema="surrogate_hmc_lg2_endpoint_v1", command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=sys.prefix, plan=PLAN, result_file=str(out/"report.json"), inputs=records,
        data_version="LG2 original attempt1 claim, once rounded FP32", source_hashes=hashes,
        seeds=dict(transitions=[[20260922, 37100+i] for i in range(5)],
                   claim=read_claim_seeds(records)),
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


def read_claim_seeds(records):
    return json.loads(Path(records["claim"]["path"]).read_text())["seeds"]


if __name__ == "__main__":
    raise SystemExit(main())
