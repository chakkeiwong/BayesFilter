"""Diagnostic-only stage trace for the frozen guided SIR merge comparison."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=int, default=263043001)
    parser.add_argument("--beta", type=float, default=0.)
    parser.add_argument("--direction", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    arguments = parser.parse_args()
    started = time.monotonic()
    arguments.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(arguments.source.resolve()))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    import bayesfilter
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot

    if Path(bayesfilter.__file__).resolve().parent != arguments.source.resolve() / "bayesfilter":
        raise ValueError("Wrong source imported")
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    source_index = json.loads((arguments.source / "index.json").read_text())
    if not verify_source_snapshot(arguments.source, source_index)["unchanged"]:
        raise ValueError("Source changed")
    from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec

    study = arguments.repository / "docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-16-finalized/final-holdout"
    original = json.loads((study / "manifest.json").read_text())
    record = json.loads((study / f"dataset-{arguments.dataset}.json").read_text())
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    reset = tf.random.stateless_normal([64, 18], [263040002, 0], dtype=tf.float64)
    kernel = build_backward_guided_ledh_kernel(
        lambda parameter, direction: spec.model(parameter, direction)[0], None,
        horizon=5, particle_count=64, state_dimension=18, observation_dimension=9,
        parameter_dimension=3, dtype=tf.float64, jit_compile=True, return_trace=True,
        reset_design=reset, physical_mixture_weight=arguments.beta, **original["controls"])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
    def prepare(seed):
        normals = tf.random.stateless_normal([64, 18], tf.stack([seed, 0]), dtype=tf.float64)
        states, covariances, d_states, d_covariances = spec.initial_cloud(theta, normals)
        noises = tf.random.stateless_normal([5, 64, 18], tf.stack([seed, 1]), dtype=tf.float64)
        uniforms = tf.random.stateless_uniform([5, 64], tf.stack([seed, 2]), dtype=tf.float64)
        direction = (tf.zeros([3], tf.float64) if arguments.direction < 0
                     else tf.one_hot(arguments.direction, 3, dtype=tf.float64))
        return (theta, direction, states, covariances, noises,
                tf.constant(record["observations"], tf.float64), d_states, d_covariances,
                uniforms)

    inputs = prepare(tf.constant(283030000 + 4 * (arguments.dataset - 263043001)))
    input_digest = hashlib.sha256()
    for tensor in (*inputs, reset):
        input_digest.update(tf.io.serialize_tensor(tensor).numpy())
    result = kernel(*inputs)
    for tensor in tf.nest.flatten(result):
        tensor.numpy()
    elapsed = time.monotonic() - started
    fields = {}
    for step, row in enumerate(result[2]):
        for name, tensor in row.items():
            filename = f"step-{step}-{name}.tensor"
            (arguments.output / filename).write_bytes(tf.io.serialize_tensor(tensor).numpy())
            fields[filename] = dict(dtype=tensor.dtype.name, shape=tensor.shape.as_list())
    times = []
    if arguments.profile:
        tf.profiler.experimental.start(str(arguments.output / "profile"), options=
            tf.profiler.experimental.ProfilerOptions(host_tracer_level=2, python_tracer_level=0))
    for repeat in range(3):
        began = time.perf_counter()
        outputs = kernel(*inputs)
        for tensor in tf.nest.flatten(outputs):
            tensor.numpy()
        times.append(time.perf_counter() - began)
    if arguments.profile:
        tf.profiler.experimental.stop()
    save(arguments.output / "manifest.json", dict(command=sys.argv, environment=sys.executable,
        git_commit=source_index["commit"], source=str(arguments.source),
        input_sha256=input_digest.hexdigest(), dataset=arguments.dataset, direction=arguments.direction,
        beta=arguments.beta, controls=original["controls"], reset_seed=263040002,
        gpu_memory_policy=memory, gpu_devices=[device.name for device in tf.config.list_physical_devices("GPU")],
        tf32=False, dtype="float64", jit_compile=True, cpu_only=False, diagnostic_only=True,
        plan="docs/plans/ledh-sir-merge-repair-20261022.md", fields=fields,
        value=float(result[0]), score=result[1].numpy().tolist(), warm_seconds=times,
        initialization_seconds=elapsed, wall_seconds=time.monotonic()-started,
        source_unchanged=verify_source_snapshot(arguments.source, source_index)["unchanged"],
        allocator=tf.config.experimental.get_memory_info("GPU:0")))
    print(json.dumps(dict(output=str(arguments.output), value=float(result[0]),
                         score=result[1].numpy().tolist(), wall_seconds=time.monotonic()-started)))


if __name__ == "__main__":
    main()
