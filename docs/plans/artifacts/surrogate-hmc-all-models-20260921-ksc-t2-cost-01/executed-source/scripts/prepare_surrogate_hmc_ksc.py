#!/usr/bin/env python3
"""Bounded KSC T2 consumer qualification and scoped controls, without inference."""
import argparse
import ast
import hashlib
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
PLAN = "docs/plans/surrogate-hmc-ksc-t2-2026-09-21.md"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(ROOT))
STAGE_CAPS = {"qualify": (120., 180.), "controls": (250., 350.), "cost": (65., 100.)}


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if hasattr(value, "numpy"):
        return serial(value.numpy().tolist())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    pending = ["bayesfilter/testing/surrogate_hmc_ksc_consumer.py",
               "bayesfilter/inference/surrogate_hmc_dctr_selection.py",
               "scripts/calibrate_surrogate_hmc_nonlinear_controls.py",
               "scripts/calibrate_surrogate_hmc_sv_controls.py",
               "scripts/prepare_surrogate_hmc_ksc.py"]
    found = {}
    while pending:
        name = pending.pop()
        path = ROOT / name
        if name in found or not path.is_file():
            continue
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("bayesfilter.", "scripts.")):
                pending.append(node.module.replace(".", "/") + ".py")
    for name in (PLAN, "tests/test_surrogate_hmc_ksc_reference.py"):
        found[name] = sha(ROOT / name)
    return found


def budgets():
    total = dict(cpu_worker_seconds=0., gpu_seconds=0.)
    family = dict(total)
    stages = {stage: dict(total) for stage in STAGE_CAPS}
    for path in (ROOT / "docs/plans/artifacts").glob("surrogate-hmc-all-models-*/accounting.json"):
        payload = json.loads(path.read_text())
        charge = payload.get("charged_this_run", payload.get("charged", {}))
        for key in total:
            value = float(charge[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid campaign accounting")
            total[key] += value
            if "-ksc-t2-" in path.parent.name:
                family[key] += value
                for stage in stages:
                    if f"-ksc-t2-{stage}-" in path.parent.name:
                        stages[stage][key] += value
    return total, family, stages


def verify_qualification(path, current_sources):
    report = json.loads((path / "report.json").read_text())
    manifest = json.loads((path / "run-manifest.json").read_text())
    if report.get("status") != "QUALIFIED_CONSUMER_MECHANICS" or report.get("qualified") is not True:
        raise ValueError("qualified KSC consumer required")
    for name, digest in manifest["source_hashes"].items():
        if name.startswith("bayesfilter/") and (current_sources.get(name) != digest or sha(ROOT / name) != digest):
            raise ValueError("qualified numerical source drift: " + name)
    data = path / "data.json"
    if sha(data) != manifest["data_sha256"]:
        raise ValueError("qualification data drift")
    return report, manifest


def check_selection(root, *, validate_controls=False):
    """Recover completed calibration after a later, recorded resource failure."""
    from bayesfilter.inference import surrogate_hmc_dctr_selection as authority
    MULTIPLIERS = authority.MULTIPLIERS
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    manifest = json.loads((root / "run-manifest.json").read_text())
    status = report.get("status")
    if status == "INFRASTRUCTURE_OR_RESOURCE_FAILURE":
        accounting = json.loads((root / "accounting.json").read_text())
        if accounting.get("exit_code") not in (-24, 124) or report.get("exit_code") != accounting["exit_code"]:
            raise ValueError("unclassified control failure cannot be reused")
    elif status != "SELECTED_FOR_PROPOSAL_EXPERIMENT":
        raise ValueError("completed scoped KSC control selection required")
    if report.get("scope", {}).get("model") != "M03-KSC-T2":
        raise ValueError("wrong control scope")
    for name, digest in manifest["source_hashes"].items():
        if name.startswith("bayesfilter/") and sha(ROOT / name) != digest:
            raise ValueError("selected numerical source drift: " + name)
    payload = json.loads((root / "selection.json").read_text())
    if (payload.get("schema") != authority.SCHEMA or payload.get("use") != authority.USE
            or payload.get("scope") != report["scope"] or payload.get("source_hashes") != manifest["source_hashes"]
            or payload.get("issuer_sha256") != sha(authority.__file__)
            or payload.get("exact_score_admission") is not False or payload.get("default_readiness") is not False):
        raise ValueError("invalid selection authority or scope")
    for group in ("evidence", "partition_data"):
        if set(payload[group]) != {"calibration", "validation"}:
            raise ValueError("incomplete tuning partitions")
        for record in payload[group].values():
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("selection evidence checksum mismatch")
    if report.get("selected") != payload or payload["selected_multiplier"] != MULTIPLIERS[payload["selected_index"]]:
        raise ValueError("incomplete or inconsistent recorded selection")
    pairs = {}
    for partition, record in payload["evidence"].items():
        curve = json.loads(Path(record["path"]).read_text())
        pairs[partition] = curve["pairs"]
        if curve["scope"] != payload["scope"] or curve["source_hashes"] != payload["source_hashes"]:
            raise ValueError("control curve scope/source mismatch")
        if curve["data_sha256"] != payload["partition_data"][partition]["sha256"]:
            raise ValueError("control curve/data mismatch")
        if len(curve.get("arms", [])) != len(MULTIPLIERS):
            raise ValueError("incomplete control arms")
        for multiplier, arm in zip(MULTIPLIERS, curve["arms"]):
            if sha(arm["path"]) != arm["sha256"]:
                raise ValueError("control arm checksum mismatch")
            saved = json.loads(Path(arm["path"]).read_text())
            if saved["multiplier"] != multiplier or set(saved["situations"]) != {"nominal", "stress"}:
                raise ValueError("incomplete control situations")
    if authority.first_stable_pair(pairs) != payload["selected_index"]:
        raise ValueError("selection disagrees with measured curves")
    # The public issuer's full settings validation imports TensorFlow. Run it
    # only in the GPU worker, after memory policy is configured.
    if validate_controls:
        authority.load_selection(root / "selection.json", expected_scope=report["scope"],
                                  expected_source_hashes=manifest["source_hashes"])
    return payload


def run_bounded_worker(command, *, log_path, cpu_budget, wall_budget):
    """Bound a single worker using four CPU cores and a live CPU/wall watchdog."""
    cores = sorted(os.sched_getaffinity(0))[:4]
    ticks = os.sysconf("SC_CLK_TCK")
    started = time.monotonic()

    def limit():
        os.sched_setaffinity(0, cores)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.floor(cpu_budget)), max(2, math.ceil(cpu_budget))))

    reason = None
    with Path(log_path).open("w") as stream:
        with subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              preexec_fn=limit) as child:
            while child.poll() is None:
                try:
                    fields = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
                    used = (int(fields[11]) + int(fields[12])) / ticks
                except FileNotFoundError:
                    break
                if used >= cpu_budget - 4:
                    reason = "CPU_WATCHDOG"
                elif time.monotonic() - started >= wall_budget:
                    reason = "WALL_WATCHDOG"
                if reason:
                    child.kill()
                    break
                try:
                    child.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
    return code, dict(cpu_affinity=cores, guard_seconds=4., stop_reason=reason)


def qualify(out, manifest, tf, check):
    from bayesfilter.testing.surrogate_hmc_ksc_reference import make_ksc_dataset, physical_parameters
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_consumer
    from bayesfilter.inference.surrogate_hmc_dctr_selection import protection_controls
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, bind_neural_force_hmc_tuning_runner,
        neural_force_proposal, neural_force_hmc_transition,
    )
    data = make_ksc_dataset(partition="calibration")
    save(out / "data.json", data)
    consumer = make_ksc_consumer(data, controls=protection_controls(1.), batch_size=4)
    save(out / "consumer.json", consumer.manifest)
    manifest.update(data_sha256=sha(out / "data.json"), data_file=str(out / "data.json"),
                    seeds=dict(data=data["seeds"], fixed_inputs=consumer.fixture.manifest["seeds"],
                               transitions=[[20260921, 25101], [20260921, 25102]]))
    save(out / "run-manifest.json", manifest)
    report = dict(status="RUNNING", checks={}, timings={}, surrogate_hmc_tested=False,
                  posterior_comparisons={name: "NOT_RUN" for name in ("B/R", "C/D", "D/R", "C/R")})

    def equal(left, right):
        return all(bool(tf.reduce_all(a == b)) for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)))

    def timed(name, callback):
        check()
        start, cpu = time.monotonic(), time.process_time()
        result = callback()
        for item in tf.nest.flatten(result):
            item.numpy()
        report["timings"][name] = dict(wall_seconds=time.monotonic() - start,
                                       cpu_worker_seconds=time.process_time() - cpu)
        save(out / "report.json", report)
        check()
        return result

    q = tf.constant([[-.5, -.3], [0., 0.], [.4, -.2], [.8, .6]], tf.float64)
    baseline = timed("force_compile_first", lambda: consumer.evaluate_force(q))
    save(out / "force-probes.json", dict(q=q, result=baseline))
    checks = report["checks"]
    checks["healthy_nominal"] = bool(tf.reduce_all(baseline[2]["value_valid"] & baseline[2]["force_valid"]))
    checks["bitwise_replay"] = equal(baseline, timed("force_replay", lambda: consumer.evaluate_force(q)))
    direct = consumer.fixture.compile(4, coordinates="unconstrained")
    checks["actual_canonical_call_chain_parity"] = equal(
        baseline, timed("direct_canonical", lambda: direct(tf.cast(q, tf.float32))))
    permutation = tf.constant([2, 0, 3, 1])
    permuted = timed("permutation", lambda: consumer.evaluate_force(tf.gather(q, permutation)))
    checks["permutation"] = equal(tf.nest.map_structure(lambda v: tf.gather(v, permutation), baseline), permuted)
    changed = tf.tensor_scatter_nd_add(q, [[3, 0]], tf.constant([.1], tf.float64))
    changed_result = timed("row_independence", lambda: consumer.evaluate_force(changed))
    checks["row_independence"] = equal(tf.nest.map_structure(lambda v: v[:3], baseline),
                                      tf.nest.map_structure(lambda v: v[:3], changed_result))
    physical = tf.cast(physical_parameters(q), tf.float32)
    with_moments = consumer.fixture.compile(4, filter_moments=True)
    moments = timed("moment_observability", lambda: with_moments(physical))
    save(out / "moment-probes.json", dict(physical=physical, result=moments))
    plain = consumer.fixture.compile(4)
    plain_result = timed("physical_without_moments", lambda: plain(physical))
    checks["moment_observability_no_numerical_change"] = equal(plain_result[:2], moments[:2])
    before = timed("reference_endpoint", lambda: consumer.endpoint.function(q))
    consumer.evaluate_force(changed)
    checks["endpoint_independent_of_force_calls"] = equal(before, consumer.endpoint.function(q))
    field = consumer.force.function(q)
    checks["field_sign_and_original_status"] = equal((field.force, field.force_valid),
                                                     (-tf.cast(baseline[1], tf.float64), baseline[2]["force_valid"]))
    value, score = consumer.evaluate_reference(q)
    checks["reference_endpoint_and_score"] = equal((before.potential, consumer.exact_force.function(q).force),
                                                   (-value, -score))
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
                                                target_scope="M03-KSC-T2-N1024-calibration-qualification")
    save(out / "binding.json", dict(binding.payload()))
    checks["typed_independent_force_endpoint"] = consumer.force.identity != consumer.endpoint.identity
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=2, inverse_mass_diagonal=(1., 1.), dtype="float64")

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)] * 2, jit_compile=True, autograph=False)
    def proposal(q, p):
        return neural_force_proposal(q, p, consumer.force, cfg)

    p = tf.constant([[.2, -.1], [-.4, .3], [.6, -.5], [-.8, .7]], tf.float64)
    forward = timed("proposal_compile_first", lambda: proposal(q, p))
    reverse = timed("proposal_reverse", lambda: proposal(forward.position, forward.momentum))
    error = float(tf.maximum(tf.reduce_max(tf.abs(reverse.position - q)), tf.reduce_max(tf.abs(reverse.momentum - p))))
    report.update(reversal_absolute_error=error, reversal_tolerance=32 * 2**-23)
    checks["proposal_involution"] = error <= report["reversal_tolerance"]

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                 tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(q, potential, seed):
        return neural_force_hmc_transition(q, potential, consumer.force, consumer.endpoint, cfg, seed)

    first = timed("transition_compile_first", lambda: transition(q, before.potential, tf.constant([20260921, 25101])))
    second = timed("transition_steady", lambda: transition(first.position, first.potential, tf.constant([20260921, 25102])))
    trace = second.trace
    difference = trace.final_potential + trace.final_kinetic - trace.initial_potential - trace.initial_kinetic
    checks["acceptance_uses_endpoint_hamiltonian"] = bool(tf.reduce_all(tf.abs(difference - trace.delta_h) <= 1e-12))
    checks["endpoint_transition_health"] = all(bool(tf.reduce_all(
        tf.math.is_finite(step.trace.delta_h) & ~step.trace.endpoint_status["numerical_failure"]))
        for step in (first, second))
    checks["no_force_fallback_or_divergence"] = all(bool(tf.reduce_all(
        ~step.trace.force_fallback & ~step.trace.divergence)) for step in (first, second))
    checks["cache_matches_endpoint"] = bool(tf.reduce_all(
        tf.abs(second.potential - consumer.endpoint.function(second.position).potential) < 1e-10))
    checks["finite_states"] = bool(tf.reduce_all(tf.math.is_finite(second.position)))
    checks["original_force_healthy"] = bool(tf.reduce_all(baseline[2]["force_valid"]))
    save(out / "transitions.json", [first.trace._asdict(), trace._asdict()])
    checks["stable_graph_trace"] = consumer.evaluate_force.experimental_get_tracing_count() == 1
    report.update(qualified=all(checks.values()), original_force_healthy=checks["original_force_healthy"],
                  status="QUALIFIED_CONSUMER_MECHANICS" if all(checks.values()) else "QUALIFICATION_FAILED")
    save(out / "report.json", report)
    return 0 if report["qualified"] else 2


def controls(out, manifest, tf, check):
    from bayesfilter.testing.surrogate_hmc_ksc_reference import make_ksc_dataset
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_fixture, make_ksc_consumer
    from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, protection_controls, issue_selection
    from scripts.calibrate_surrogate_hmc_sv_controls import no_new_failures
    from scripts.calibrate_surrogate_hmc_nonlinear_controls import optimistic_forecast
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig, neural_force_hmc_transition
    scope = dict(model="M03-KSC-T2", attempt=1, replicate=0, horizon=2, particle_count=1024,
                 state_dimension=1, parameter_dimension=2, dtype="float32", tf32=True, jit_compile=True,
                 reset="contract_e_chol_v1", chunk_policy="dpf_transport_exact_divisor_cap3000_v1",
                 chunk_size=1024, control_family="full_canonical_geometric_protection_curve_v1")
    fields = ("filtering_mean", "filtering_covariance", "post_reset_mean", "post_reset_covariance")
    nominal = tf.constant([[.25, .3], [.45, .6], [.65, .4], [.8, .75]], tf.float32)
    stress = tf.constant([[.1001, .1001], [.8999, .8999], [.1001, .8999], [.8999, .1001]], tf.float32)
    data_paths, curves, datasets = {}, {}, {}
    for partition in ("calibration", "validation"):
        directory = out / partition
        directory.mkdir()
        datasets[partition] = {s: make_ksc_dataset(partition=partition, situation=s) for s in ("nominal", "stress")}
        data_paths[partition] = directory / "data.json"
        save(data_paths[partition], dict(model_id="M03-KSC-T2", attempt=1, partition=partition, data=datasets[partition]))
        outputs, diagnostics = [], []
        for index, multiplier in enumerate(MULTIPLIERS):
            current, situations, fixtures = {}, {}, {}
            for situation, positions in (("nominal", nominal), ("stress", stress)):
                check()
                fixture = make_ksc_fixture(datasets[partition][situation], controls=protection_controls(multiplier))
                value, force, status = fixture.compile(4, filter_moments=True)(positions)
                value.numpy()
                check()
                valid = status["value_valid"] & status["force_valid"]
                finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(force), axis=1)
                bounded = tf.ones([4], tf.bool)
                for key in fields:
                    axes = list(range(1, status[key].shape.rank))
                    finite &= tf.reduce_all(tf.math.is_finite(status[key]), axis=axes)
                    bounded &= tf.reduce_all(tf.abs(status[key]) < 1e6, axis=axes)
                healthy = valid & finite & bounded
                flagged = ~valid & ((status["value_failure_code"] != 0) | (status["force_failure_code"] != 0))
                situations[situation] = serial(dict(positions=positions, value=value, force=force, status=status,
                    healthy_rows=healthy, all_healthy=bool(tf.reduce_all(healthy)),
                    all_failures_flagged=bool(tf.reduce_all(healthy | flagged))))
                current[situation] = dict(value=value, force=force, **{key: status[key] for key in fields})
                fixtures[situation] = dict(fixture.manifest)
            arm = dict(multiplier=multiplier, controls=protection_controls(multiplier), fixtures=fixtures, situations=situations)
            save(directory / f"arm-{index}.json", arm)
            diagnostics.append(arm)
            outputs.append(current)
        pairs = []
        for index in range(3):
            changes = {key: float(tf.reduce_max(tf.abs(outputs[index]["nominal"][key]
                         - outputs[index + 1]["nominal"][key]) / (1 + tf.abs(outputs[index + 1]["nominal"][key]))))
                       for key in outputs[index]["nominal"]}
            healthy = all(diagnostics[i]["situations"]["nominal"]["all_healthy"] for i in (index, index + 1))
            safe = all(diagnostics[i]["situations"]["stress"]["all_failures_flagged"] for i in (index, index + 1))
            no_new = no_new_failures(diagnostics[index]["situations"]["stress"]["healthy_rows"],
                                     diagnostics[index + 1]["situations"]["stress"]["healthy_rows"])
            maximum = max(changes.values())
            pairs.append(dict(stronger_multiplier=MULTIPLIERS[index], weaker_multiplier=MULTIPLIERS[index + 1],
                component_changes=changes, normalized_maximum_change=maximum, tolerance=1e-3,
                healthy_nominal=healthy, stress_safe=safe, no_new_stress_failure=no_new,
                non_harm_passed=healthy and safe and no_new and maximum <= 1e-3))
        curves[partition] = directory / "curve.json"
        save(curves[partition], dict(scope=scope, source_hashes=manifest["source_hashes"],
             data_sha256=sha(data_paths[partition]), pairs=pairs,
             arms=[dict(path=str(directory / f"arm-{i}.json"), sha256=sha(directory / f"arm-{i}.json")) for i in range(4)]))
    try:
        selection = issue_selection(output=out / "selection.json", scope=scope, evidence_paths=curves,
            partition_data_paths=data_paths, source_hashes=manifest["source_hashes"])
    except ValueError as exc:
        if str(exc) != "NO_STABLE_CONTROL_PAIR":
            raise
        selection = None
    report = dict(status="SELECTED_FOR_PROPOSAL_EXPERIMENT" if selection else "NO_STABLE_CONTROL_PAIR",
                  selected=selection, scope=scope, posterior_comparisons="NOT_RUN", forecast={},
                  heuristic_dominance="NOT_RUN", inference_attempts_consumed=0, scoped_control_attempts_consumed=1)
    save(out / "report.json", report)
    if selection:
        consumer = make_ksc_consumer(datasets["calibration"]["nominal"], controls=selection["controls"], batch_size=4)
        cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=1, inverse_mass_diagonal=(1., 1.), dtype="float64")

        @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                     tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
        def transition(q, potential, seed):
            return neural_force_hmc_transition(q, potential, consumer.force, consumer.endpoint, cfg, seed)

        q = tf.constant([[-.5, -.3], [0., 0.], [.4, -.2], [.8, .6]], tf.float64)
        potential = consumer.endpoint.function(q).potential
        records = []
        for i in range(5):
            check()
            start, cpu = time.monotonic(), time.process_time()
            step = transition(q, potential, tf.constant([20260921, 25200 + i]))
            step.position.numpy()
            records.append(dict(wall_seconds=time.monotonic() - start, cpu_worker_seconds=time.process_time() - cpu,
                                compilation_included=i == 0))
            save(out / f"cost-B-{i}.json", dict(timing=records[-1], trace=step.trace._asdict()))
            if not bool(tf.reduce_all(tf.math.is_finite(step.trace.delta_h)
                                     & ~step.trace.endpoint_status["numerical_failure"]
                                     & ~step.trace.force_fallback & ~step.trace.divergence)):
                raise ValueError("invalid full-DCTR transition in cost screen")
            q, potential = step.position, step.potential
        _, family, _ = budgets()
        remaining = {"cpu_worker_seconds": 1800 - family["cpu_worker_seconds"] - time.process_time(),
                     "gpu_seconds": 7200 - family["gpu_seconds"] - (time.monotonic() - manifest["started_monotonic"])}
        report["forecast"] = {"B": optimistic_forecast(records, remaining),
                              "C": {"status": "NOT_MEASURED_UNQUALIFIED_FINITE_ENDPOINT"}}
        report["complete_transition_costs"] = records
    manifest["partition_data"] = {p: dict(path=str(path), sha256=sha(path)) for p, path in data_paths.items()}
    manifest["seeds"] = {p: {s: d["seeds"] for s, d in values.items()} for p, values in datasets.items()}
    save(out / "run-manifest.json", manifest)
    save(out / "report.json", report)
    return 0 if selection else 2


def cost(out, manifest, tf, check):
    from bayesfilter.testing.surrogate_hmc_ksc_consumer import make_ksc_consumer
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig, neural_force_hmc_transition
    from scripts.calibrate_surrogate_hmc_nonlinear_controls import optimistic_forecast
    root = Path(manifest["controls_root"])
    selection = check_selection(root, validate_controls=True)
    data_record = selection["partition_data"]["calibration"]
    data = json.loads(Path(data_record["path"]).read_text())["data"]["nominal"]
    save(out / "data.json", data)
    consumer = make_ksc_consumer(data, controls=selection["controls"], batch_size=4)
    manifest.update(data_sha256=sha(out / "data.json"), seeds=data["seeds"],
                    selection_sha256=sha(root / "selection.json"), consumer=dict(consumer.manifest))
    save(out / "run-manifest.json", manifest)
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=1,
                              inverse_mass_diagonal=(1., 1.), dtype="float64")

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                 tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def transition(q, potential, seed):
        return neural_force_hmc_transition(q, potential, consumer.force, consumer.endpoint, cfg, seed)

    q = tf.constant([[-.5, -.3], [0., 0.], [.4, -.2], [.8, .6]], tf.float64)
    potential = consumer.endpoint.function(q).potential
    records = []
    for i in range(5):
        check()
        start, cpu = time.monotonic(), time.process_time()
        step = transition(q, potential, tf.constant([20260921, 25200 + i]))
        step.position.numpy()
        records.append(dict(wall_seconds=time.monotonic() - start,
                            cpu_worker_seconds=time.process_time() - cpu, compilation_included=i == 0))
        save(out / f"cost-B-{i}.json", dict(timing=records[-1], trace=step.trace._asdict()))
        if not bool(tf.reduce_all(tf.math.is_finite(step.trace.delta_h)
                                 & ~step.trace.endpoint_status["numerical_failure"]
                                 & ~step.trace.force_fallback & ~step.trace.divergence)):
            raise ValueError("invalid full-DCTR transition in cost screen")
        q, potential = step.position, step.potential
    _, family, _ = budgets()
    remaining = {"cpu_worker_seconds": 1800 - family["cpu_worker_seconds"] - time.process_time(),
                 "gpu_seconds": 7200 - family["gpu_seconds"] - (time.monotonic() - manifest["started_monotonic"])}
    report = dict(status="COMPLETE_TRANSITION_COST_MEASURED", selected_multiplier=selection["selected_multiplier"],
        controls_root=str(root), selection_sha256=manifest["selection_sha256"],
        recovered_completed_calibration=True, parent_status=json.loads((root / "report.json").read_text())["status"],
        forecast={"B": optimistic_forecast(records, remaining), "C": {"status": "NOT_MEASURED_UNQUALIFIED_FINITE_ENDPOINT"}},
        complete_transition_costs=records, inference_attempts_consumed=0,
        posterior_comparisons={key: "NOT_RUN" for key in ("B/R", "C/D", "D/R", "C/R")})
    save(out / "report.json", report)
    return 0


def worker(out):
    manifest = json.loads((out / "run-manifest.json").read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise ValueError("exactly one designated GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    manifest.update(memory_policy=memory, tensorflow=tf.__version__, tf32=True, jit_compile=True,
                    gpu_status=[d.name for d in tf.config.list_logical_devices("GPU")])
    save(out / "run-manifest.json", manifest)

    def check():
        if (time.process_time() > manifest["cpu_budget"] - 2
                or time.monotonic() - manifest["started_monotonic"] > manifest["wall_budget"] - 2):
            raise TimeoutError("KSC stage resource budget exhausted")

    result = {"qualify": qualify, "controls": controls, "cost": cost}[manifest["stage"]](out, manifest, tf, check)
    manifest["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    save(out / "run-manifest.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGE_CAPS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--controls-root", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.stage or not args.output:
        parser.error("--stage and --output are required")
    out = args.output.resolve()
    if out.parent != ROOT / "docs/plans/artifacts" or not out.name.startswith(
            "surrogate-hmc-all-models-20260921-ksc-t2-" + args.stage + "-"):
        raise ValueError("fresh versioned KSC stage root required")
    started, parent_cpu = time.monotonic(), time.process_time()
    total, family, stages = budgets()
    cpu_cap, wall_cap = STAGE_CAPS[args.stage]
    cpu = min(cpu_cap - stages[args.stage]["cpu_worker_seconds"], 1800 - family["cpu_worker_seconds"],
              54000 - total["cpu_worker_seconds"])
    wall = min(wall_cap - stages[args.stage]["gpu_seconds"], 7200 - family["gpu_seconds"], 226800 - total["gpu_seconds"])
    if min(cpu, wall) < 5:
        raise ValueError("KSC stage/family/campaign allocation exhausted")
    provenance = sources()
    if args.stage == "cost":
        if not args.controls_root:
            raise ValueError("--controls-root required")
        check_selection(args.controls_root)
    if args.stage == "controls":
        if not args.qualification_root:
            raise ValueError("--qualification-root required")
        verify_qualification(args.qualification_root, provenance)
        if stages[args.stage]["cpu_worker_seconds"] > 0:
            raise ValueError("prior controls attempt requires checked reuse or prospective repair")
    out.mkdir(exist_ok=False)
    for name in provenance:
        target = out / "executed-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save(out / "executed-source/sha256.json", provenance)
    manifest = dict(command=[sys.executable, *sys.argv], stage=args.stage, environment=sys.prefix,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan=PLAN, result_file=str(out / "report.json"), source_hashes=provenance,
        cpu_budget=cpu, wall_budget=wall, started_monotonic=started,
        family_remaining={"cpu_worker_seconds": 1800 - family["cpu_worker_seconds"],
                          "gpu_seconds": 7200 - family["gpu_seconds"]},
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), memory_policy="pending trusted GPU configuration",
        qualification_root=str(args.qualification_root.resolve()) if args.qualification_root else None,
        controls_root=str(args.controls_root.resolve()) if args.controls_root else None,
        data_version="M03 KSC T2 new frozen calibration/validation instances; claim untouched")
    save(out / "run-manifest.json", manifest)
    save(out / "report.json", dict(status="RUNNING", posterior_comparisons="NOT_RUN"))
    child_cpu = cpu - (time.process_time() - parent_cpu) - 1
    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    code, watchdog = run_bounded_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(out)],
        log_path=out / "worker.log", cpu_budget=child_cpu, wall_budget=max(1, wall - (time.monotonic() - started)))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    charge = dict(cpu_worker_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
                 + time.process_time() - parent_cpu, gpu_seconds=time.monotonic() - started)
    save(out / "accounting.json", dict(charged=charge, charged_this_run=charge, exit_code=code,
                                      wall_seconds=charge["gpu_seconds"], watchdog=watchdog))
    report = json.loads((out / "report.json").read_text())
    if code and report.get("status") in ("RUNNING", "SELECTED_FOR_PROPOSAL_EXPERIMENT"):
        report.update(status="INFRASTRUCTURE_OR_RESOURCE_FAILURE", exit_code=code)
        save(out / "report.json", report)
    manifest = json.loads((out / "run-manifest.json").read_text())
    manifest.update(wall_seconds=charge["gpu_seconds"], cpu_worker_seconds=charge["cpu_worker_seconds"], exit_code=code)
    save(out / "run-manifest.json", manifest)
    if any(sha(ROOT / name) != digest for name, digest in provenance.items()):
        raise ValueError("source changed during KSC execution")
    print(json.dumps(dict(status=report["status"], accounting=charge, output=str(out))), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
