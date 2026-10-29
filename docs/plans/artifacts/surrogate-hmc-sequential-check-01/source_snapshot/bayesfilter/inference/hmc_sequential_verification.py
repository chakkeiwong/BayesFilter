"""Bounded fresh warm-up and cumulative verification for a frozen HMC kernel.

This controller takes a typed kernel callback; it never substitutes an exact
gradient sampler for a position-only proposal field. TensorFlow owns numerics.
"""
from __future__ import annotations

import math
import os
import resource
import time
import tensorflow as tf


def combined_chunks(chunks):
    if not chunks:
        return None
    cls = type(chunks[0])
    return cls(**{
        name: getattr(chunks[-1], name) if name in ("final_state", "final_step_size")
        else tf.concat([getattr(chunk, name) for chunk in chunks], axis=0)
        for name in chunks[0]._fields
    })


def rhat_values(diagnostic):
    def convert(value):
        return [float(x) if math.isfinite(float(x)) else None for x in value.numpy().tolist()]
    return {key: convert(value) for key, value in diagnostic.items()}


def rhat_passed(diagnostic, threshold):
    maximum = diagnostic["maximum"]
    return bool(tf.reduce_all(tf.math.is_finite(maximum) & (maximum <= threshold)))


def run_sequential_verification(*, config, initial_state, run_chunk, health,
                                diagnose, transform, archive, persist,
                                verification_threshold, permit_chunk=None):
    """Same states/kernel across chunks; warm-up window, cumulative verification.

    Callback arguments are explicit to enable call-chain and deterministic
    controller tests without replacing the scientific numerical kernel.
    """
    state = initial_state
    phases = {"warmup": [], "verification": []}
    records, events, seeds = [], [], []
    counts = {"warmup": 0, "verification": 0}
    status = "RUNNING"
    latest_rhat = {}
    warmup_ready = False
    for phase_index, phase in enumerate(("warmup", "verification")):
        maximum = getattr(config, phase + "_max_results")
        minimum = getattr(config, phase + "_results")
        while counts[phase] < maximum:
            count = min(config.chunk_results, maximum - counts[phase])
            if permit_chunk is not None and not permit_chunk(phase, count):
                status = "BUDGET_EXHAUSTED"
                break
            index = len(phases[phase])
            seed = tf.random.experimental.stateless_fold_in(
                tf.constant(config.fresh_seed, tf.int32), phase_index * 100000 + index)
            seed_value = seed.numpy().tolist()
            if seed_value in seeds:
                raise ValueError("sequential seed reuse")
            chunk = run_chunk(state, seed, count)
            tf.debugging.assert_equal(
                tf.shape(chunk.states), [count, 4, config.nomination.parameter_dimension])
            final_matches = tf.reduce_all(tf.equal(chunk.final_state, chunk.states[-1]) |
                (tf.math.is_nan(chunk.final_state) & tf.math.is_nan(chunk.states[-1])))
            checks = {k: bool(v) for k, v in health(chunk, state).items()}
            checks["final_state_matches"] = bool(final_matches)
            saved = archive(phase, index, chunk, state, seed_value)
            records.append(dict(phase=phase, index=index, seed=seed_value, count=count,
                                archive=saved))
            seeds.append(seed_value)
            phases[phase].append(chunk)
            counts[phase] += count
            state = chunk.final_state
            event = dict(phase=phase, index=index, count=counts[phase], checks=checks,
                         check_due=False, rhat=None, passed=False)
            if not all(checks.values()):
                status = "HEALTH_VETO"
            elif counts[phase] >= minimum and (
                    counts[phase] % config.check_every_results == 0 or counts[phase] == maximum):
                all_states = tf.concat([part.states for part in phases[phase]], axis=0)
                checked = (all_states[-config.warmup_check_window_results:]
                           if phase == "warmup" else all_states)
                latest_rhat = diagnose(transform(checked))
                threshold = config.warmup_rhat_max if phase == "warmup" else verification_threshold
                passed = rhat_passed(latest_rhat, threshold)
                event.update(check_due=True, rhat=rhat_values(latest_rhat),
                             rhat_draws_per_chain=int(checked.shape[0]),
                             threshold=threshold, passed=passed)
                if passed:
                    if phase == "warmup":
                        warmup_ready = True
                    else:
                        status = "VERIFIED"
                elif counts[phase] == maximum:
                    status = phase.upper() + "_MAXIMUM_INCONCLUSIVE"
            events.append(event)
            next_phase = ("verification" if phase == "warmup" and warmup_ready else
                          "retained_assessment" if status == "VERIFIED" else
                          phase if status == "RUNNING" else "review_terminal_result")
            progress = dict(schema="bayesfilter.sequential_verification_progress.v1",
                status=status, counts=dict(counts), warmup_ready=warmup_ready,
                chunks=records, events=events, used_seeds=seeds,
                next_phase_plan=dict(phase=next_phase,
                    action="continue_same_frozen_kernel" if status == "RUNNING" else
                           "refresh_master_phase_plan",
                    required_criterion="recent_window_readiness" if next_phase == "warmup"
                                       else "cumulative_rhat_and_health",
                    maximum_per_chain=maximum, continuation_veto=status == "HEALTH_VETO"))
            persist(progress)
            if status != "RUNNING" or (phase == "warmup" and warmup_ready):
                break
        if status != "RUNNING" or not warmup_ready:
            break
    if status == "RUNNING":
        status = "VERIFICATION_MAXIMUM_INCONCLUSIVE" if warmup_ready else "WARMUP_MAXIMUM_INCONCLUSIVE"
    if status == "BUDGET_EXHAUSTED":
        persist(dict(schema="bayesfilter.sequential_verification_progress.v1",
            status=status, counts=dict(counts), warmup_ready=warmup_ready,
            chunks=records, events=events, used_seeds=seeds,
            next_phase_plan=dict(phase="review_terminal_result", action="budget_exhausted",
                                 continuation_veto=True)))
    return dict(status=status, warmup_ready=warmup_ready, counts=counts,
                phases=phases, chunks=records, events=events, used_seeds=seeds,
                rhat=latest_rhat, passed=status == "VERIFIED")


def runtime_budget_guard():
    """Stop between archived chunks before independent wall/CPU watchdogs fire."""
    deadline = os.environ.get("BAYESFILTER_WORKER_WALL_DEADLINE")
    cpu_cap = os.environ.get("BAYESFILTER_WORKER_CPU_ALLOWANCE")
    if deadline is None and cpu_cap is None:
        return None
    deadline = float(deadline) if deadline is not None else None
    cpu_cap = float(cpu_cap) if cpu_cap is not None else None
    if any(value is not None and (not math.isfinite(value) or value<=0) for value in (deadline,cpu_cap)):
        raise ValueError("invalid worker resource limit")
    def cpu_used():
        own=resource.getrusage(resource.RUSAGE_SELF)
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        return own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime
    last_wall, last_cpu = time.monotonic(), cpu_used()
    previous_wall, previous_cpu = 0.,0.
    def permit(phase, count):
        nonlocal last_wall,last_cpu,previous_wall,previous_cpu
        wall,cpu=time.monotonic(),cpu_used()
        if wall-last_wall>1.:
            previous_wall=wall-last_wall
            previous_cpu=max(0.,cpu-last_cpu)
        last_wall,last_cpu=wall,cpu
        return ((deadline is None or deadline-wall>max(60.,1.2*previous_wall)) and
                (cpu_cap is None or cpu_cap-cpu>max(60.,1.2*previous_cpu)))
    return permit
