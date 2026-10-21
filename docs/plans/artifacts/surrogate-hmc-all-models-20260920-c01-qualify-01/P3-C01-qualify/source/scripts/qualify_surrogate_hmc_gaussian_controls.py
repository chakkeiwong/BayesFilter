#!/usr/bin/env python3
"""Qualify the actual shared Gaussian-control consumers and measure their cost."""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.testing.surrogate_hmc_gaussian_control import (
        ARMS, PRECISION, bind_control, initial_positions, run_provenance,
        write_report, cpu_seconds, cost_forecast)
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, neural_force_proposal, neural_force_hmc_transition)

    started, started_cpu = time.monotonic(), cpu_seconds()
    report = dict(provenance=run_provenance(output, memory, "mechanics and cost; no retained inference"),
                  arms={}, qualified=False, retained_inference=False, heuristic_dominance="NOT_RUN")
    save = lambda: write_report(output/"report.json", report)
    save()
    cfg = NeuralForceHMCConfig(step_size=.1, num_leapfrog_steps=4,
                              inverse_mass_diagonal=(1., 1., 1.), dtype="float64")
    q = initial_positions()
    p = tf.constant([[.3, -.2, .1], [.2, .4, -.2], [-.4, .3, .2], [-.2, -.1, -.3]], tf.float64)
    signature = [tf.TensorSpec([4, 3], tf.float64)]*2
    timings = {}
    try:
        for index, arm in enumerate(ARMS):
            force, target, binding, _ = bind_control(arm)
            checks = {}
            row = report["arms"][arm] = dict(checks=checks, binding=dict(binding.payload()))
            def propose(q0, p0):
                proposal = neural_force_proposal(q0, p0, force, cfg)
                return proposal.position, proposal.momentum
            graph = tf.function(propose, input_signature=signature, jit_compile=False, autograph=False)
            compiled = tf.function(propose, input_signature=signature, jit_compile=True, autograph=False)
            ordinary, candidate = graph(q, p), compiled(q, p)
            parity = max(float(tf.reduce_max(tf.abs(a-b))) for a, b in zip(ordinary, candidate))
            reverse = compiled(*candidate)
            reverse_error = max(float(tf.reduce_max(tf.abs(a-b))) for a, b in zip((q, p), reverse))
            checks["graph_xla_parity"] = parity < 1e-12
            checks["momentum_flip_involution"] = reverse_error < 1e-12
            row.update(parity_max=parity, reverse_max=reverse_error)
            expected = .5*tf.reduce_sum(q*q*tf.constant(PRECISION, tf.float64), axis=-1)
            baseline_endpoint = target.function(q)
            checks["per_chain_exact_endpoint"] = bool(tf.reduce_all(tf.abs(baseline_endpoint-expected) < 1e-12))
            changed = tf.tensor_scatter_nd_add(q, [[3, 0]], tf.constant([.5], tf.float64))
            checks["endpoint_row_independence"] = bool(tf.reduce_all(target.function(changed)[:3] == baseline_endpoint[:3]))
            changed_proposal = compiled(changed, p)
            checks["proposal_row_independence"] = all(bool(tf.reduce_all(a[:3] == b[:3]))
                for a, b in zip(candidate, changed_proposal))

            @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64), tf.TensorSpec([2], tf.int32)],
                         jit_compile=True, autograph=False)
            def transition(q0, seed):
                value = neural_force_hmc_transition(q0, target.function(q0), force, target, cfg, seed)
                trace = value.trace
                energy = trace.final_potential + .5*tf.reduce_sum(trace.final_momentum**2, axis=-1)
                energy -= trace.initial_potential + .5*tf.reduce_sum(trace.initial_momentum**2, axis=-1)
                return value.position, trace.delta_h, energy, trace.log_acceptance_ratio
            seed = tf.constant([20260920, 8100+index], tf.int32)
            actual, replay = transition(q, seed), transition(q, seed)
            checks["bitwise_transition_replay"] = all(bool(tf.reduce_all(a == b)) for a, b in zip(actual, replay))
            checks["true_per_chain_energy"] = bool(tf.reduce_all(tf.abs(actual[1]-actual[2]) < 1e-12))
            checks["exact_endpoint_log_acceptance"] = bool(tf.reduce_all(
                tf.abs(actual[3]-tf.minimum(tf.zeros_like(actual[2]), -actual[2])) < 1e-12))

            @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64), tf.TensorSpec([2], tf.int32)],
                         jit_compile=True, autograph=False)
            def cost_chunk(q0, seed0):
                def body(i, position, potential, accepted, largest, healthy):
                    value = neural_force_hmc_transition(position, potential, force, target, cfg,
                        tf.random.experimental.stateless_fold_in(seed0, i))
                    trace = value.trace
                    valid = trace.finite_status & ~trace.divergence & ~trace.force_fallback
                    valid &= ~trace.endpoint_out_of_support & ~trace.endpoint_status["numerical_failure"]
                    return (i+1, value.position, value.potential,
                            accepted+tf.cast(trace.accepted, tf.int32),
                            tf.maximum(largest, tf.abs(trace.delta_h)), healthy & valid)
                return tf.while_loop(lambda i, *unused: i < 256, body,
                    (tf.constant(0), q0, target.function(q0), tf.zeros([4], tf.int32),
                     tf.zeros([4], tf.float64), tf.ones([4], tf.bool)))
            measured = []
            for call in range(3):
                tick, cpu_tick = time.monotonic(), cpu_seconds()
                values = cost_chunk(q, tf.constant([20260920, 8200+10*index+call], tf.int32))
                host = [tensor.numpy().tolist() for tensor in values]
                measured.append(dict(wall_seconds=time.monotonic()-tick,
                                     cpu_worker_seconds=cpu_seconds()-cpu_tick,
                                     accepted_by_chain=host[3], max_abs_energy_by_chain=host[4],
                                     healthy_by_chain=host[5]))
            checks["cost_chunks_healthy"] = all(all(row["healthy_by_chain"]) for row in measured)
            timings[arm] = dict(compile=measured[0], steady=measured[1:])
            row["timings"] = timings[arm]
            save()
            print(arm, "mechanics", all(checks.values()), flush=True)
        report["qualified"] = all(all(row["checks"].values()) for row in report["arms"].values())
        report["forecast"] = cost_forecast(timings)
        report["status"] = ("QUALIFIED_AFFORDABLE" if report["qualified"] and report["forecast"]["affordable"]
                            else "UNDER_BUDGETED" if report["qualified"] else "MECHANICS_VETO")
    except Exception as exc:
        report.update(status="INFRASTRUCTURE_FAILURE", error=repr(exc))
        raise
    finally:
        report.update(wall_seconds=time.monotonic()-started, cpu_worker_seconds=cpu_seconds()-started_cpu,
                      allocator=tf.config.experimental.get_memory_info("GPU:0"))
        save()
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
