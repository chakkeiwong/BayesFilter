#!/usr/bin/env python3
"""CPU-only diagnostic reconstruction of the completed five-model posterior.

Reads archived draws; never changes a sampler, target, or completion decision.
The equal-tailed intervals are empirical, not normal approximations. This is an
explicit eager TensorFlow postprocessing exception to the GPU/XLA runtime policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from docs.benchmarks.run_ledh_covariance_proposal import spec_for
    from bayesfilter.testing.surrogate_hmc_literature_priors import chart

    inputs = {}

    def resolve(path):
        path = Path(path)
        if path.is_absolute() and "/docs/plans/" in str(path):
            return ROOT / ("docs/plans/" + str(path).split("/docs/plans/", 1)[1])
        return path if path.is_absolute() else ROOT / path

    def read_bytes(path, expected=None):
        path = resolve(path)
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"checksum mismatch: {path}")
        inputs[str(path.relative_to(ROOT))] = digest
        return blob

    def read(path, expected=None):
        return json.loads(read_bytes(path, expected))

    def tensor(root, entry):
        return tf.io.parse_tensor(read_bytes(root / entry["path"], entry["sha256"]),
                                  tf.as_dtype(entry.get("dtype", "float64")))

    def plain(x):
        return x.numpy().tolist() if tf.is_tensor(x) else x

    def maximum_difference(actual, expected):
        difference = float(tf.reduce_max(tf.abs(actual - tf.constant(expected, actual.dtype))))
        if not difference <= 1e-8:
            raise ValueError(f"archived summary reconstruction mismatch: {difference}")
        return difference

    def physical(name, theta):
        if name == "lgssm":
            return tf.stack([.55 * tf.tanh(theta[..., 0]), tf.exp(theta[..., 1]),
                             tf.exp(theta[..., 2]), theta[..., 3]], axis=-1)
        if name == "ksc":
            normal = tfp.distributions.Normal(tf.constant(0., tf.float64), tf.constant(1., tf.float64))
            return tf.stack([normal.cdf(theta[..., 0]), tf.exp(theta[..., 1])], axis=-1)
        if name == "range_bearing":
            return tf.exp(2. * theta)
        if name == "sir_d18":
            # Dimensionless multipliers of the benchmark's kappa, nu and noise SD.
            return tf.exp(theta)
        return theta

    names = {"LGSSM": "lgssm", "KSC SV": "ksc", "Predator-prey": "predator_prey",
             "Range bearing": "range_bearing", "SIR NeuTra": "sir_d18"}
    labels = {
        "lgssm": ["phi (first state)", "process variance multiplier", "observation variance multiplier", "initial mean amplitude"],
        "ksc": ["gamma", "beta"],
        "predator_prey": ["r", "K", "a", "s", "u", "v"],
        "range_bearing": ["Q variance multiplier", "R variance multiplier"],
        "sir_d18": ["infection-rate multiplier", "recovery-rate multiplier", "observation-noise SD multiplier"],
    }
    artifacts = ROOT / "docs/plans/artifacts"
    completion = read(artifacts / "surrogate-hmc-five-model-completion-20261023-01/summary.json")
    rows = []
    for row in completion["rows"]:
        name = names[row["model"]]
        raw = read(row["result"], row["sha256"])
        hmc = raw.get("result", raw)
        manifest_path = resolve(hmc["manifest"])
        manifest = read(manifest_path)
        root = manifest_path.parent
        nomination = read(root / "nomination/manifest.json", manifest["nomination_manifest_sha256"])
        states = tensor(root, manifest["verification"]["states"])
        center = tensor(root / "nomination", nomination["tensors"]["center"])
        factor = tensor(root / "nomination", nomination["tensors"]["factor"])
        q = center + tf.linalg.matmul(states, factor, transpose_b=True)
        run_manifest = read(root.parent / "manifest.json")
        map_hash = None
        if name == "sir_d18":
            freeze = read(run_manifest["job"]["frozen"])
            frozen = read(freeze["frozen_map"])
            loaded = load_frozen_neutra_artifact(frozen, expected_target_signature=freeze["target_signature"])
            map_hash = loaded.manifest.transport_hash
            if not manifest["runner_binding"]["endpoint_target_identity"].endswith(":" + map_hash):
                raise ValueError("frozen map differs from the HMC endpoint map")
            q = tf.reshape(loaded.transport.forward_batch(tf.reshape(q, [-1, q.shape[-1]])), q.shape)
            scope = freeze["target_scope"]
        else:
            scope = run_manifest["scope"]
        if q.shape != (hmc["counts"]["verification"], 4, scope["parameter_dimension"]):
            raise ValueError("retained dimensions differ from the completed run")
        tf.debugging.assert_all_finite(q, "reconstructed draws")
        errors = {
            "mean": maximum_difference(tf.reduce_mean(q, [0, 1]), hmc["q_mean"]),
            "standard_deviation": maximum_difference(tf.math.reduce_std(q, [0, 1]), hmc["q_standard_deviation"]),
        }
        accepted = tensor(root, manifest["verification"]["is_accepted"])
        acceptance = tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
        errors["acceptance"] = maximum_difference(acceptance, hmc["acceptance_per_chain"])
        q_rhat = tf.constant(rank_normalized_split_rhat(tf.transpose(q, [1, 0, 2]))["maximum"], tf.float64)
        errors["q_rhat"] = maximum_difference(q_rhat, hmc["diagnostics"]["rank_normalized_split_rhat"]["maximum"])
        prior = scope["prior"]
        if prior["id"] != "literature-model-priors-v1":
            raise ValueError("unexpected prior chart")
        nominal = tf.constant(prior["nominal"], tf.float64)
        theta, _ = chart(name, q, nominal)
        if name == "range_bearing":
            truth_theta = tf.zeros([2], tf.float64)
            data = artifacts / "ledh-range-bearing-particle-ladder-20261010-02/N4032/data/1704-t20.json"
        else:
            spec = spec_for(name, lgssm_family="p44")
            truth_theta = spec.default_theta(tf.float64)
            data = artifacts / f"ledh-independent-validation-20261008-01/inputs/{name}-d26100831/dataset.json"
        dataset = read(data, scope["data_sha256"])
        if name != "range_bearing":
            if dataset["data_seed"] != 26100831 or dataset["target_id"] != spec.target_id:
                raise ValueError("simulation truth belongs to different data")
            obs = tf.constant(dataset["observations"], tf.float64)
            if hashlib.sha256(tf.io.serialize_tensor(obs).numpy()).hexdigest() != dataset["observation_sha256"]:
                raise ValueError("simulation observation checksum mismatch")
        draws = physical(name, theta)
        truth = physical(name, truth_theta)
        flat = tf.reshape(draws, [-1, q.shape[-1]])
        intervals = tfp.stats.percentile(flat, [2.5, 97.5], axis=0, interpolation="linear")
        physical_rhat = plain(rank_normalized_split_rhat(tf.transpose(draws, [1, 0, 2]))["maximum"])
        parameters = []
        for i, label in enumerate(labels[name]):
            lower, upper, value = float(intervals[0, i]), float(intervals[1, i]), float(truth[i])
            parameters.append(dict(parameter=label, truth=value, posterior_mean=float(tf.reduce_mean(flat[:, i])),
                lower_95=lower, upper_95=upper, truth_in_95_interval=lower <= value <= upper,
                q_rhat=float(q_rhat[i]), physical_rhat=physical_rhat[i]))
        rows.append(dict(model=row["model"], parameters=parameters,
            coverage_count=sum(p["truth_in_95_interval"] for p in parameters),
            parameter_count=len(parameters), max_q_rhat=float(tf.reduce_max(q_rhat)),
            max_physical_rhat=max(physical_rhat), all_parameters_rhat_below_1_01=max(physical_rhat) < 1.01,
            acceptance_per_chain=plain(acceptance), all_chain_acceptance_in_65_75=bool(tf.reduce_all((acceptance >= .65) & (acceptance <= .75))),
            counts_per_chain=hmc["counts"], nominal_chart_theta=plain(nominal), simulation_truth_theta=plain(truth_theta),
            reconstruction_errors=errors, frozen_transport_hash=map_hash, source_result=row["result"]))
    report = dict(schema="five_model_archived_posterior_audit_v1", rows=rows, input_sha256=inputs,
        truth_provenance={
            "four_T50": ["docs/benchmarks/run_ledh_independent_validation.py:97", "docs/benchmarks/run_ledh_covariance_proposal.py:61", "bayesfilter/highdim/sqmc_lgssm_tf.py:37", "bayesfilter/highdim/sqmc_ksc_tf.py:48", "bayesfilter/highdim/sqmc_nonlinear_tf.py:50"],
            "range_bearing_T20": ["bayesfilter/testing/marginal_weight_campaign.py:27", "bayesfilter/testing/surrogate_hmc_five_model.py:205"]},
        interval="equal-tailed empirical 2.5% and 97.5% quantiles, linear interpolation, all retained chains pooled",
        cpu_only=True, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], gpu_seconds=0,
        backend="TensorFlow/TFP post-run diagnostic", jit_compile=False, tensorflow=tf.__version__,
        command=[sys.executable, *sys.argv], git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), wall_seconds=time.monotonic()-started,
        plan="docs/plans/surrogate-hmc-posterior-closeout-2026-10-23.md", warmup_excluded=True,
        exact_posterior_accuracy="not established by this audit", repeated_dataset_coverage="not assessed")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Archived posterior interval audit", "", "All intervals use retained draws only, with four chains pooled. R-hat is recomputed after returning to physical parameter coordinates.", "", "| Model | Parameter | Truth | 95% credible interval | Contains truth | R-hat |", "|---|---|---:|---|---|---:|"]
    for row in rows:
        for p in row["parameters"]:
            lines.append(f"| {row['model']} | {p['parameter']} | {p['truth']:.7g} | [{p['lower_95']:.7g}, {p['upper_95']:.7g}] | {'Yes' if p['truth_in_95_interval'] else 'No'} | {p['physical_rhat']:.6f} |")
    lines += ["", "| Model | Truths inside intervals | Maximum R-hat | Chain acceptance | Within 65–75% |", "|---|---:|---:|---|---|"]
    for row in rows:
        acceptance = row["acceptance_per_chain"]
        lines.append(f"| {row['model']} | {row['coverage_count']}/{row['parameter_count']} | {row['max_physical_rhat']:.6f} | {min(acceptance):.1%}–{max(acceptance):.1%} | {'Yes' if row['all_chain_acceptance_in_65_75'] else 'No'} |")
    lines += ["", "The SIR parameters are dimensionless multipliers of benchmark infection rate, recovery rate and observation-noise standard deviation. Range-bearing multipliers apply to variances. LGSSM phi is the first state's coefficient; the other coefficients are fixed multiples.", "", "A single simulated dataset per model cannot establish repeated-dataset coverage or exact posterior accuracy. These are descriptive checks; the original sampling completion decisions are unchanged.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(args.output / "report.json"), "wall_seconds": report["wall_seconds"], "models": [{k:r[k] for k in ("model", "coverage_count", "parameter_count", "max_physical_rhat", "acceptance_per_chain")} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
