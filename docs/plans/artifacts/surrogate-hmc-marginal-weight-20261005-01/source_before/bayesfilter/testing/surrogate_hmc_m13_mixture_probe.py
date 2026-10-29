"""Discarded GPU/XLA M13 step-mixture diagnostic; no tuning authority."""
import argparse
import json
from pathlib import Path
import time

import tensorflow as tf

F64 = tf.float64
ARMS = ((.125, 1.), (.25, 1.))
COUNT = 32


def run(output, memory):
    from bayesfilter.inference.surrogate_hmc_program import ROOT, write_json, file_hash
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    from bayesfilter.inference import hmc_tensorflow_tuning as native
    from bayesfilter.inference.hmc_tensorflow_verified import _save_chunk, _health
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components, read_tensor
    from bayesfilter.testing.surrogate_hmc_m13_step_mixture import bind_step_mixture, scaled_step

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prior = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-development-01/mixing-probe-01"
    receipt = json.loads((prior / "guard-receipt.json").read_text())
    if receipt.get("status") != "COMPLETED" or not receipt.get("accounting_complete"):
        raise ValueError("original probe must finish before the serial GPU mixture probe")
    root = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-sequential-recovery-03/m13-public-tuning/tuning"
    progress = json.loads((root / "fresh-progress.json").read_text())
    saved = json.loads((root / "nomination/manifest.json").read_text())
    center = read_tensor(root / "nomination", saved["tensors"]["center"])
    factor = read_tensor(root / "nomination", saved["tensors"]["factor"])
    epsilon = float(read_tensor(root / "nomination", saved["tensors"]["step_size"]))
    starts = [read_tensor(root, progress["chunks"][i]["archive"]["tensors"]["final_state"])
              for i in (18, 24)]
    adapter, base, target, fixture, selection, path = selected_components()
    affine = native._TensorFlowAffineAdapter(
        native._TensorFlowAffineTransform(center, factor), adapter.adapter_signature(), adapter.target_scope)
    report = dict(status="RUNNING", qualified=False, posterior_qualified=False, ranking_supported=False,
        role="discarded M13 mixing diagnostic; public tuner still required",
        memory_policy=memory, device="gpu", jit_compile=True, tf32=True, tensorflow=tf.__version__,
        source_root=str(root), selection_sha256=file_hash(path), data_seed=1704,
        observation_checksum=fixture.observation_checksum, epsilon=epsilon,
        center=center.numpy().tolist(), factor=factor.numpy().tolist(),
        phase_plan="docs/plans/surrogate-hmc-m13-48h-continuation-2026-09-29.md",
        source_sha256=file_hash(Path(__file__)), baseline_report_sha256=file_hash(prior / "report.json"),
        count_per_replication=COUNT, common_random_numbers_across_arms=True, arms=[])
    write_json(output / "report.json", finite_report(report))

    def build(binding, step):
        return binding.build_tensor_kernel(affine, step_size=tf.constant(step, F64),
            num_leapfrog_steps=1, target_scope=adapter.target_scope,
            target_status_trace_policy="none", chain_execution_mode="tf_function", use_xla=True)

    # Same XLA backend and stateless seed are essential for this pathwise check.
    large, direct = build(base, epsilon * 8), build(base, epsilon)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], F64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def parity_chart(x, seed):
        return scaled_step(large, x, large.bootstrap_results(x), tf.constant(.125, F64), seed)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], F64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def parity_direct(x, seed):
        return direct.one_step(x, direct.bootstrap_results(x), seed=seed)

    seed = tf.constant([20260929, 2199], tf.int32)
    got, expected = parity_chart(starts[0], seed), parity_direct(starts[0], seed)
    tf.debugging.assert_near(got[0], expected[0], atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(got[1].delta_h, expected[1].delta_h, atol=1e-9, rtol=1e-9)
    for name in ("is_accepted", "divergence", "force_fallback", "finite_status"):
        tf.debugging.assert_equal(getattr(got[1], name), getattr(expected[1], name))
    for name in got[1].endpoint_status:
        tf.debugging.assert_equal(got[1].endpoint_status[name], expected[1].endpoint_status[name])
    report["gpu_xla_component_parity"] = dict(passed=True,
        maximum_state_difference=float(tf.reduce_max(tf.abs(got[0] - expected[0]))),
        maximum_energy_difference=float(tf.reduce_max(tf.abs(got[1].delta_h - expected[1].delta_h))))
    write_json(output / "report.json", finite_report(report))
    print("M13 GPU/XLA component parity passed", flush=True)

    # Two distinct fixed kernels need two graphs; each graph traces once.
    for arm_index, multipliers in enumerate(ARMS):
        binding = bind_step_mixture(base, multipliers)
        kernel = build(binding, epsilon * 8)

        @tf.function(input_signature=[tf.TensorSpec([4, 2], F64), tf.TensorSpec([2], tf.int32)],
                     jit_compile=True, autograph=False)
        def chunk(state, rng):
            return native._run_tensor_steps(kernel=kernel, initial_state=state,
                num_results=COUNT, seed=rng, adaptive=False)

        arm = dict(multipliers=list(multipliers), base_step_size=epsilon * 8,
                   leapfrog_steps=1, binding=binding.payload(), replications=[])
        report["arms"].append(arm)
        for replication, start in enumerate(starts):
            rng = [20260929, 2200 + replication]
            begin = time.monotonic()
            sample = chunk(start, tf.constant(rng, tf.int32))
            sample.states.numpy()
            wall = time.monotonic() - begin
            previous = tf.concat([start[None, ...], sample.states[:-1]], axis=0)
            delta = sample.states - previous
            checks = {k: bool(v) for k, v in _health(sample, start, 1000.).items()}
            rep = dict(seed=rng, wall_seconds=wall, checks=checks,
                health_passed=all(checks.values()), graph_trace_count=chunk.experimental_get_tracing_count(),
                archive=_save_chunk(output, f"arm-{arm_index}-rep-{replication}", sample),
                acceptance_by_chain=tf.reduce_mean(tf.cast(sample.is_accepted, F64), axis=0).numpy().tolist(),
                expected_acceptance_by_chain=tf.reduce_mean(tf.exp(tf.minimum(sample.log_accept_ratio, 0.)), axis=0).numpy().tolist(),
                squared_jump_by_chain_coordinate=tf.reduce_mean(delta * delta, axis=0).numpy().tolist(),
                squared_total_displacement_by_chain_coordinate=tf.square(sample.final_state - start).numpy().tolist())
            arm["replications"].append(rep)
            write_json(output / "report.json", finite_report(report))
            print("Mixture probe", arm_index, replication, "health", rep["health_passed"], "wall", wall, flush=True)
    report["status"] = "DIAGNOSTIC_COMPLETE"
    write_json(output / "report.json", finite_report(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    run(args.output, memory)


if __name__ == "__main__":
    main()
