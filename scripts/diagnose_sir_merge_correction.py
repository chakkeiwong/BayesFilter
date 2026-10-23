"""CPU-only diagnostic of identical-input correction across frozen merge sources."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2)
    arguments = parser.parse_args()
    started = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(arguments.source.resolve()))
    import tensorflow as tf
    from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp

    manifest = json.loads((arguments.trace / "manifest.json").read_text())

    def field(name):
        filename = f"step-{arguments.step}-{name}.tensor"
        return tf.io.parse_tensor((arguments.trace / filename).read_bytes(),
                                  tf.as_dtype(manifest["fields"][filename]["dtype"]))

    source = field("children")
    weights = field("posterior_weights")
    source_tangent = field("d_children")[..., None]
    weights_tangent = field("d_posterior_weights")[..., None]
    points = field("higher_moment_stage_raw_reset")
    tangent = field("higher_moment_stage_raw_reset_tangent")
    if tangent.shape.rank == 2:
        tangent = tangent[..., None]
    controls = manifest["controls"]
    inputs = (source, weights, source_tangent, weights_tangent, points, tangent)

    @tf.function(input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in inputs],
                 autograph=False, jit_compile=False)
    def evaluate(*values):
        return higher_moment_shape_jvp(*values,
            correction_steps=controls["correction_steps"], strength=controls["correction_strength"],
            floor=1e-5, diagonal_lm_damping=controls["correction_lm_damping"],
            diagonal_lm_scale_floor=controls["correction_lm_scale_floor"],
            diagonal_trust_radius=controls["correction_trust_radius"],
            pairwise_correction_steps=controls["pairwise_steps"],
            pairwise_strength=controls["pairwise_strength"], pairwise_floor=1e-5,
            pairwise_particle_rms_cap=controls["pairwise_rms_cap"],
            coordinatewise_standardized_cap=controls["coordinate_cap"],
            coordinatewise_standardized_cap_power=controls["coordinate_cap_power"],
            moment_safety=controls["moment_safety"], return_stages=True)

    output = evaluate(*inputs)
    mean = tf.reduce_sum(weights[:, None] * source, axis=0)
    centered = source - mean
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    eigenvalues = tf.linalg.eigvalsh(covariance)
    report = dict(cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
        diagnostic_only=True, source=str(arguments.source), trace=str(arguments.trace),
        step=arguments.step, controls=controls, command=sys.argv,
        environment=sys.executable, wall_seconds=time.monotonic()-started,
        weight_ess=float(1 / tf.reduce_sum(tf.square(weights))),
        max_weight=float(tf.reduce_max(weights)),
        covariance_eigenvalues=eigenvalues.numpy().tolist(),
        covariance_condition=float(eigenvalues[-1] / eigenvalues[0]),
        output={name: tensor.numpy().tolist() for name, tensor in output.items()})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
        ("source", "step", "weight_ess", "max_weight", "covariance_condition", "wall_seconds")}))


if __name__ == "__main__":
    main()
