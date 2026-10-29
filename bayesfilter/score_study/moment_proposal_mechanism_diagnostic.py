"""CPU saved-trace diagnostic; Q-flow is not a canonical filter candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import _flow_substeps_with_tangent
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec


def gaussian_log_density(points, means, covariance):
    factor = tf.linalg.cholesky(covariance)
    difference = (points - means)[..., None]
    solved = tf.linalg.triangular_solve(factor, difference)
    normalizer = tf.cast(tf.shape(points)[-1], points.dtype) * math.log(2 * math.pi)
    return -.5 * (tf.reduce_sum(solved ** 2, axis=(-2, -1))
                  + 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)), axis=-1)
                  + normalizer)


def scalar(value):
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("This diagnostic requires explicitly hidden GPU devices")
    arguments.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    model, _ = spec.model(theta, tf.zeros_like(theta))
    process = model.process_covariance
    observation_covariance = model.observation_covariance
    eye = tf.eye(18, dtype=tf.float64)
    observation_matrix = model.observation_jacobian_fn(tf.zeros([64, 18], tf.float64))[0]
    innovation = observation_matrix @ process @ tf.transpose(observation_matrix) + observation_covariance
    gain = tf.transpose(tf.linalg.solve(innovation, observation_matrix @ process))
    conditional_covariance = process - gain @ observation_matrix @ process
    conditional_covariance = (conditional_covariance + tf.transpose(conditional_covariance)) / 2
    batched_conditional_covariance = tf.broadcast_to(conditional_covariance, [64, 18, 18])
    inverse_observation = tf.linalg.inv(observation_covariance)

    @tf.function(input_signature=[tf.TensorSpec([64, 18], tf.float64),
        tf.TensorSpec([64, 18], tf.float64), tf.TensorSpec([64, 18, 18], tf.float64),
        tf.TensorSpec([9], tf.float64)], autograph=False, jit_compile=False)
    def flow(points, anchors, covariance, observation):
        result = _flow_substeps_with_tangent(model, points, tf.zeros_like(points),
            anchors, tf.zeros_like(anchors), covariance, tf.zeros_like(covariance),
            observation, observation_covariance, None, inverse_observation, None,
            substeps=24, eye=eye, return_affine=True)
        children, _, determinant, _, mean, _, matrix, _ = result
        return children, determinant, mean, matrix @ process @ tf.linalg.matrix_transpose(matrix), matrix

    rows = []
    input_hashes = {}
    for dataset_seed, particle_seed in ((262023017, 274000034), (262023001, 274000002)):
        source = arguments.input / f"trace-off-{dataset_seed}.json"
        input_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        records = json.loads(source.read_text())
        noises = tf.random.stateless_normal([5, 64, 18], [particle_seed, 1], dtype=tf.float64)
        for index, record in enumerate(records):
            if time.monotonic() - started > 180:
                raise TimeoutError("Diagnostic budget exhausted")
            tensor = lambda name: tf.constant(record[name], tf.float64)
            pre_flow, observation = tensor("pre_flow"), tensor("observation")
            anchors = pre_flow - tf.einsum("ij,nj->ni", tf.linalg.cholesky(process), noises[index])
            conditional_mean = anchors + tf.einsum("io,no->ni", gain,
                observation[None] - tf.einsum("oi,ni->no", observation_matrix, anchors))
            conditional_log_normalizer = gaussian_log_density(observation[None],
                tf.einsum("oi,ni->no", observation_matrix, anchors), innovation)
            base_density = gaussian_log_density(pre_flow, anchors, process)
            reconstruction_error = scalar(tf.reduce_max(tf.abs(base_density - tensor("proposal_log_density"))))
            covariance_carry = tf.einsum("ij,jmn->imn", tensor("reset_transport"), tensor("post_covariances"))
            carry_error = scalar(tf.reduce_max(tf.abs(covariance_carry - tensor("covariances_after_reset")))
                                 / tf.reduce_max(tf.abs(tensor("covariances_after_reset"))))
            exact_weights = tf.nn.softmax(tensor("incoming_log_weights") + conditional_log_normalizer)
            row = dict(dataset_seed=dataset_seed, time=index + 1,
                proposal_density_replay_error=reconstruction_error, covariance_carry_relative_error=carry_error,
                predicted_covariance_trace_mean=scalar(tf.reduce_mean(tf.linalg.trace(tensor("predicted_covariances")))),
                process_covariance_trace=scalar(tf.linalg.trace(process)),
                exact_conditional_ess=scalar(1 / tf.reduce_sum(exact_weights ** 2)), arms={})
            ordering = tf.argsort(tensor("posterior_logits"), direction="DESCENDING")
            for name in ("prior_observation_logits", "observation_log_density", "posterior_logits"):
                row[name + "_winner_gap"] = scalar(tf.gather(tensor(name), ordering[0]) - tf.gather(tensor(name), ordering[1]))
            for name, covariance in (("canonical_P_flow", tensor("predicted_covariances")),
                                     ("diagnostic_conditional_Q_flow", tf.broadcast_to(process, [64, 18, 18]))):
                children, determinant, proposal_mean, proposal_covariance, flow_matrix = flow(pre_flow, anchors, covariance, observation)
                transition = gaussian_log_density(children, anchors, process)
                likelihood = gaussian_log_density(observation[None], model.observation_fn(children), observation_covariance)
                proposal = gaussian_log_density(children, proposal_mean, proposal_covariance)
                posterior = gaussian_log_density(children, conditional_mean, conditional_covariance)
                completion_error = scalar(tf.reduce_max(tf.abs(transition + likelihood - posterior - conditional_log_normalizer)))
                proposal_change_error = scalar(tf.reduce_max(tf.abs(proposal - base_density + determinant)))
                weights = tf.nn.softmax(tensor("incoming_log_weights") + transition + likelihood - proposal)
                difference = (proposal_mean - conditional_mean)[..., None]
                inverse_times_covariance = tf.linalg.solve(batched_conditional_covariance, proposal_covariance)
                quadratic = tf.reduce_sum(difference * tf.linalg.solve(batched_conditional_covariance, difference), axis=(-2, -1))
                divergence = .5 * (tf.linalg.trace(inverse_times_covariance) + quadratic - 18
                    + tf.linalg.logdet(conditional_covariance) - tf.linalg.logdet(proposal_covariance))
                metrics = dict(conditional_kl_mean=scalar(tf.reduce_mean(divergence)),
                    conditional_kl_min=scalar(tf.reduce_min(divergence)), conditional_kl_max=scalar(tf.reduce_max(divergence)),
                    ess=scalar(1 / tf.reduce_sum(weights ** 2)), max_weight=scalar(tf.reduce_max(weights)),
                    mean_displacement_from_anchor=scalar(tf.reduce_mean(tf.norm(proposal_mean - anchors, axis=-1))),
                    gaussian_completion_error=completion_error, proposal_change_of_variables_error=proposal_change_error)
                if name == "canonical_P_flow":
                    metrics["child_replay_error"] = scalar(tf.reduce_max(tf.abs(children - tensor("children"))))
                    metrics["logdet_replay_error"] = scalar(tf.reduce_max(tf.abs(determinant - tensor("flow_log_determinant"))))
                    metrics["weight_replay_error"] = scalar(tf.reduce_max(tf.abs(weights - tensor("posterior_weights"))))
                    desired_samples = conditional_mean + tf.einsum("ij,nj->ni",
                        tf.linalg.cholesky(conditional_covariance), noises[index])
                    preimage_samples = anchors + tf.linalg.solve(flow_matrix,
                        (desired_samples - proposal_mean)[..., None])[..., 0]
                    replayed_samples = flow(preimage_samples, anchors, covariance, observation)[0]
                    inverse_matrix = tf.linalg.solve(flow_matrix, tf.broadcast_to(eye, [64, 18, 18]))
                    preimage_covariance = inverse_matrix @ conditional_covariance @ tf.linalg.matrix_transpose(inverse_matrix)
                    preimage_mean = anchors + tf.linalg.solve(flow_matrix,
                        (conditional_mean - proposal_mean)[..., None])[..., 0]
                    preimage_log_density = gaussian_log_density(preimage_samples, preimage_mean, preimage_covariance)
                    desired_log_density = gaussian_log_density(desired_samples, conditional_mean, conditional_covariance)
                    preimage_weights = (gaussian_log_density(replayed_samples, anchors, process)
                        + gaussian_log_density(observation[None], model.observation_fn(replayed_samples), observation_covariance)
                        + determinant - preimage_log_density)
                    required_information = (tf.linalg.matrix_transpose(flow_matrix)
                        @ tf.linalg.solve(batched_conditional_covariance, flow_matrix) - tf.linalg.inv(process))
                    required_eigenvalues = tf.linalg.eigvalsh((required_information + tf.linalg.matrix_transpose(required_information)) / 2)
                    metrics["preimage_sample_replay_error"] = scalar(tf.reduce_max(tf.abs(replayed_samples - desired_samples)))
                    metrics["preimage_density_identity_error"] = scalar(tf.reduce_max(tf.abs(preimage_log_density - desired_log_density - determinant)))
                    metrics["preimage_conditional_weight_error"] = scalar(tf.reduce_max(tf.abs(preimage_weights - conditional_log_normalizer)))
                    metrics["required_information_minimum_eigenvalue"] = scalar(tf.reduce_min(required_eigenvalues))
                    metrics["required_information_maximum_eigenvalue"] = scalar(tf.reduce_max(required_eigenvalues))
                    metrics["preimage_mean_displacement_from_anchor"] = scalar(tf.reduce_mean(tf.norm(preimage_mean - anchors, axis=-1)))
                    if max(metrics[key] for key in ("preimage_sample_replay_error",
                            "preimage_density_identity_error", "preimage_conditional_weight_error")) > 1e-8:
                        raise ValueError("Exact preimage repair diagnostic failed")
                if any(not math.isfinite(value) for value in metrics.values()) or metrics["conditional_kl_min"] < -1e-8:
                    raise ValueError("Nonfinite or invalid conditional Gaussian diagnostic")
                if completion_error > 1e-8 or proposal_change_error > 1e-8:
                    raise ValueError("Independent Gaussian identities failed")
                row["arms"][name] = metrics
            if reconstruction_error > 1e-8 or carry_error > 1e-8 or row["arms"]["canonical_P_flow"]["child_replay_error"] > 1e-8:
                raise ValueError("Saved trace replay failed")
            rows.append(row)
    manifest = dict(command=sys.argv, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment=sys.executable, tensorflow=tf.__version__, cpu_only=True, gpu_intentionally_hidden=True,
        jit_compile=False, exception="saved_trace_reference_diagnostic_only", seeds=[262023017, 262023001, 274000034, 274000002],
        wall_seconds=time.monotonic() - started, plan="docs/plans/ledh-moment-collapse-repair-20261020.md",
        input_sha256=input_hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        result=str(arguments.output / "results.json"), canonical_algorithm_changed=False, statistical_ranking=False)
    (arguments.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (arguments.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=len(rows), replay_pass=True, output=str(arguments.output), wall_seconds=manifest["wall_seconds"])))


if __name__ == "__main__":
    main()
