"""Diagnostic four-chain public-kernel wiring and cost; no tuning authority."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import hmc_tensorflow_tuning as native
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation, bind_neural_force_hmc_tuning_runner,
)
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing import surrogate_hmc_m13_endpoint_mechanics as mechanics
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load

PLAN = "docs/plans/surrogate-hmc-m13-four-chain-cost-2026-09-25.md"
DOMAIN_ROOT = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-domain-sensitivity-01"
F64 = tf.float64
F32 = tf.float32
STARTS = ((-.25, -.25), (-.25, .25), (.25, -.25), (.25, .25))
COUNT = 8


def native_force(target, batch_size=4):
    """Keep the leading chain dimension in the actual canonical implementation."""
    evaluate = target.compile(batch_size, filter_moments=False,
                              all_correction_diagnostics=False)

    @tf.function(input_signature=[tf.TensorSpec([batch_size, 2], F64)],
                 jit_compile=True, autograph=False)
    def force(position):
        position32 = tf.cast(position, F32)
        _, score, status = evaluate(position32)
        gradient = -score + position32 / tf.cast(tf.square(mechanics.PRIOR_SD), F32)
        valid = status["program_valid"] & tf.reduce_all(tf.math.is_finite(gradient), axis=-1)
        safe = tf.where(valid[:, None], gradient, tf.zeros_like(gradient))
        return PositionOnlyForceEvaluation(tf.cast(safe, F64), valid)

    return force


def batched_endpoint(evaluator):
    """Reference evaluation may loop over rows; force evaluation is batch-native."""
    @tf.function(input_signature=[tf.TensorSpec([None, 2], F64)],
                 jit_compile=True, autograph=False)
    def endpoint(position):
        count = tf.shape(position)[0]
        values = tf.TensorArray(F64, size=count, element_shape=[])
        valid_rows = tf.TensorArray(tf.bool, size=count, element_shape=[])

        def body(i, values, valid_rows):
            likelihood, valid, _ = evaluator(position[i])
            potential = -(likelihood + mechanics.prior_log_density(position[i]))
            valid = valid & tf.math.is_finite(potential)
            potential = tf.where(valid, potential, tf.constant(float("inf"), F64))
            return i+1, values.write(i, potential), valid_rows.write(i, valid)

        _, values, valid_rows = tf.while_loop(
            lambda i, *_: i < count, body, (tf.constant(0), values, valid_rows),
            parallel_iterations=1)
        valid = valid_rows.stack()
        return TargetPotentialEvaluation(values.stack(), dict(
            value_valid=valid, force_valid=valid, numerical_failure=~valid,
            outside_declared_support=tf.zeros_like(valid)))

    return endpoint


def bind_components(force, endpoint, scope):
    return bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(function=force, identity=scope+":canonical-fp32-force",
            semantics="deterministic_position_only_proposal_field"),
        target=FrozenTargetPotential(function=endpoint, identity=scope+":fp64-endpoint"),
        target_scope=scope)


def compiled_chunk(binding, scope, leapfrogs, *, center=None, factor=None, count=COUNT):
    transform = native._TensorFlowAffineTransform(
        tf.zeros([2], F64) if center is None else center,
        tf.eye(2, dtype=F64) if factor is None else factor)
    adapter = native._TensorFlowAffineAdapter(transform, scope, scope)
    kernel = binding.build_tensor_kernel(
        adapter, step_size=tf.constant(.002, F64), num_leapfrog_steps=leapfrogs,
        target_scope=scope, target_status_trace_policy="per_chain_step",
        chain_execution_mode="tf_function", use_xla=True)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], F64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def run(position, seed):
        return native._run_tensor_steps(kernel=kernel, initial_state=position,
            num_results=count, seed=seed, adaptive=False)

    return run, kernel, adapter


def timed(function, *args):
    started = time.monotonic()
    value = function(*args)
    for tensor in tf.nest.flatten(value):
        if tf.is_tensor(tensor):
            tensor.numpy()  # Synchronize at this diagnostic timing boundary.
    return value, time.monotonic()-started


def chunk_health(chunk, start):
    from bayesfilter.inference.hmc_tensorflow_verified import _health
    # Same finite/endpoint/movement/energy screen as fresh public verification.
    return _health(chunk, start, tf.constant(1000., F64))


def cost_stage(output):
    started = time.monotonic()
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    _, fixture, mechanics_path = _load()
    domain_path = DOMAIN_ROOT / "m13-domain-sensitivity/report.json"
    domain = json.loads(domain_path.read_text())
    parent = json.loads((DOMAIN_ROOT / "report.json").read_text())
    if not (domain.get("qualified") and parent.get("source_unchanged")
            and domain["observation_checksum"] == fixture.observation_checksum):
        raise ValueError("requires valid completed same-data domain check")
    target = mechanics.make_range_bearing_target(fixture, particle_count=4096,
        controls=mechanics.protection_controls(.01), dtype=F32,
        design_seed=20260923, design_partition=9)
    _, evaluator = mechanics.endpoint_components(fixture)
    endpoint = batched_endpoint(evaluator)
    force = native_force(target)
    source_hashes = mechanics._source_hashes(target)
    scope_record = dict(data_seed=1704, observation_checksum=fixture.observation_checksum,
        particles=4096, design_seed=20260923, design_partition=9, controls_selected=False,
        controls="full_geometric_protection_controls(.01)", force_dtype="float32",
        coordinate_dtype="float64", endpoint_dtype="float64", endpoint_particles=4096,
        endpoint_replicates=8, endpoint_seed=[2026092402, 9101], prior_support="R^2",
        source_hashes=source_hashes)
    scope = "m13-four-chain-cost:"+hashlib.sha256(
        json.dumps(scope_record, sort_keys=True).encode()).hexdigest()
    binding = bind_components(force, endpoint, scope)
    position = tf.constant(STARTS, F64)
    endpoint_a, endpoint_compile = timed(endpoint, position)
    endpoint_b, endpoint_repeat = timed(endpoint, position)
    direct = tf.stack([-(evaluator(row)[0]+mechanics.prior_log_density(row))
                       for row in tf.unstack(position)])
    endpoint_replay = bool(tf.reduce_all(endpoint_a.potential == endpoint_b.potential))
    endpoint_error = float(tf.reduce_max(tf.abs(endpoint_a.potential-direct)))
    endpoint_valid = bool(tf.reduce_all(endpoint_a.status["value_valid"]))
    endpoint_parity = bool(tf.reduce_all(tf.abs(endpoint_a.potential-direct) <=
        tf.constant(1e-10, F64)*(1.+tf.abs(direct))))
    if not (endpoint_replay and endpoint_valid and endpoint_parity):
        raise ValueError("batched endpoint failed direct scalar parity/replay/validity")
    print("endpoint parity/replay passed", flush=True)
    force_a, force_compile = timed(force, position)
    force_b, force_repeat = timed(force, position)
    scalar_force = native_force(target, 1)
    scalar_values = []
    scalar_valid = []
    scalar_started = time.monotonic()
    for row in tf.unstack(position):
        result = scalar_force(row[None, :])
        scalar_values.append(result.force[0])
        scalar_valid.append(result.force_valid[0])
    direct_force = tf.stack(scalar_values)
    scalar_seconds = time.monotonic()-scalar_started
    force_valid = bool(tf.reduce_all(force_a.force_valid & force_b.force_valid))
    scalar_valid = bool(tf.reduce_all(tf.stack(scalar_valid)))
    difference = tf.abs(force_a.force-direct_force)
    report = dict(stage="m13-four-chain-cost", plan=PLAN, status="RUNNING",
        qualified=False, posterior_qualified=False, tuning_admitted=False,
        inference_launched=False, diagnostic_transitions_launched=False,
        scope=scope_record, binding_identity=binding.binding_hash,
        baseline_report=str(mechanics_path), baseline_sha256=file_hash(mechanics_path),
        domain_report=str(domain_path), domain_sha256=file_hash(domain_path),
        endpoint=dict(valid=endpoint_valid, replay_exact=endpoint_replay,
            scalar_parity=endpoint_parity, max_absolute_scalar_difference=endpoint_error,
            compile_seconds=endpoint_compile, repeat_seconds=endpoint_repeat),
        force=dict(valid=force_valid, scalar_valid=scalar_valid,
            replay_exact=bool(tf.reduce_all(force_a.force == force_b.force)),
            native_compile_batch=4, compile_seconds=force_compile, repeat_seconds=force_repeat,
            scalar_compile_and_four_calls_seconds=scalar_seconds,
            native_values=mechanics.serial(force_a.force), scalar_values=mechanics.serial(direct_force),
            absolute_difference=mechanics.serial(difference),
            normalized_difference=mechanics.serial(difference/(1.+tf.abs(direct_force))),
            discrepancy_role="explanatory_FP32_warning_not_posterior_veto"),
        timing_arms=[], ranking_supported=False,
        heuristic_dominance=dict(role="harness_comparators_only",
            endpoint_scalar_parity_passed=endpoint_parity,
            force_scalar_parity="recorded_per_start", posterior_methods="NOT_EVALUATED"))
    write_json(out/"progress.json", report)
    if not (force_valid and scalar_valid):
        report.update(status="M13_FOUR_CHAIN_FORCE_INVALID", wall_seconds=time.monotonic()-started)
        return report
    print("native batch4/scalar canonical forces valid", flush=True)
    all_healthy = True
    for leapfrogs in (2, 4):
        run, _, _ = compiled_chunk(binding, scope, leapfrogs)
        seed = tf.constant([20260925, 4200+leapfrogs], tf.int32)
        first, compilation = timed(run, position, seed)
        first_health = chunk_health(first, position)
        write_json(out/f"l{leapfrogs}-compile-chunk.json", mechanics.serial(first._asdict()))
        print("L", leapfrogs, "first chunk seconds", compilation, flush=True)
        repeats = []
        state = first.final_state
        for i in range(2):
            before = state
            chunk, seconds = timed(run, state,
                tf.constant([20260925, 4300+100*leapfrogs+i], tf.int32))
            health = chunk_health(chunk, before)
            healthy = bool(tf.reduce_all(tf.stack(list(health.values())))) and not bool(
                tf.reduce_any(chunk.force_fallback))
            all_healthy &= healthy
            record = dict(seconds=seconds, seconds_per_four_chain_transition=seconds/COUNT,
                health=mechanics.serial(health), no_force_fallback=not bool(tf.reduce_any(chunk.force_fallback)),
                acceptance_by_chain=mechanics.serial(tf.reduce_mean(tf.cast(chunk.is_accepted, F64), axis=0)),
                mean_acceptance_probability=mechanics.serial(tf.reduce_mean(
                    tf.exp(tf.minimum(chunk.log_accept_ratio, 0.)), axis=0)))
            write_json(out/f"l{leapfrogs}-repeat{i}.json",
                dict(metrics=record, chunk=mechanics.serial(chunk._asdict())))
            repeats.append(record)
            state = chunk.final_state
            print("L", leapfrogs, "repeat", i, "seconds", seconds, "healthy", healthy, flush=True)
        first_healthy = bool(tf.reduce_all(tf.stack(list(first_health.values())))) and not bool(
            tf.reduce_any(first.force_fallback))
        all_healthy &= first_healthy
        report["timing_arms"].append(dict(leapfrogs=leapfrogs, count=COUNT,
            first_call_seconds=compilation, first_call_health=mechanics.serial(first_health),
            first_call_no_force_fallback=not bool(tf.reduce_any(first.force_fallback)), repeats=repeats))
        write_json(out/"progress.json", report)
    t2, t4 = [max(row["seconds_per_four_chain_transition"] for row in arm["repeats"])
              for arm in report["timing_arms"]]
    # Explanatory forecast with 50% planning headroom, not a confidence interval.
    report["cost_forecast"] = dict(
        measured_seconds_per_four_chain_transition={"L2": t2, "L4": t4},
        extrapolated_L8_seconds_with_headroom=1.5*max(t4, 3.*t4-2.*t2),
        basis="two repeats per arm; linear L extrapolation; not statistical uncertainty",
        compilation_excluded=True)
    report["gpu_allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    report.update(status="M13_FOUR_CHAIN_COST_PASSED" if all_healthy else "M13_FOUR_CHAIN_HEALTH_FAILED",
        qualified=all_healthy, diagnostic_transitions_launched=True,
        transition_count_per_chain=COUNT*3*2,
        source_unchanged=(source_hashes == mechanics._source_hashes(target)),
        wall_seconds=time.monotonic()-started,
        next_action="scope-specific control selection and public tuning/retained plan using measured budget")
    if not report["source_unchanged"]:
        report.update(qualified=False, status="M13_FOUR_CHAIN_SOURCE_CHANGED")
    return report
