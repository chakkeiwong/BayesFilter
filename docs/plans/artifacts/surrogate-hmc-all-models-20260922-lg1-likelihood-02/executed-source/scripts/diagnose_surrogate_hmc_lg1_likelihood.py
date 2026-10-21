#!/usr/bin/env python3
"""Diagnostic only: same-data LG1 likelihood ratios and fresh DCTR spot checks."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (510, 530))
    signal.alarm(540)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
    from bayesfilter.testing.surrogate_hmc_lg1_tf import make_lg1_consumer
    from bayesfilter.testing.surrogate_hmc_lg1_finite_endpoint import make_finite_endpoint
    from bayesfilter.testing.surrogate_hmc_lg1_tf import kalman_physical

    base = Path("docs/plans/artifacts")
    reference_root = base / "surrogate-hmc-all-models-20260919-lg1-reference-master-01/P2-reference"
    finite_root = base / "surrogate-hmc-all-models-20260920-lg1-finite-reference-01/P2-finite-reference"
    selection = base / "surrogate-hmc-all-models-20260919-lg1-controls-02/P2-controls/selection.json"
    # The issuer module acquired a validation-only model/attempt check after
    # the frozen selection was issued. Load that exact archived issuer in an
    # isolated module so the selection's original authority is checked rather
    # than silently bypassed. The numerical LG1 consumer is still the current
    # checkout, whose LG1 implementation hashes are checked below.
    issuer_path = base / "surrogate-hmc-all-models-20260919-lg1-sample-01/executed-source/bayesfilter/inference/surrogate_hmc_dctr_selection.py"
    spec = importlib.util.spec_from_file_location("archived_dctr_selection", issuer_path)
    issuer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(issuer)
    scope = dict(model="LG1", horizon=10, attempt=4, state_dimension=1, parameter_dimension=1,
        particle_count=1024, chunk_extent=1024, dtype="float32", tf32=True, jit_compile=True,
        route="canonical_batch_fused_value_score", reset_contract_id="contract_e_chol_v1",
        law="stationary x0, Q1 R.5, uniform phi[-.95,.95]",
        data_regime="synthetic nominal LG1; separate calibration/validation/claim partitions",
        control_family="full_canonical_geometric_protection_curve_v1")
    expected_hashes = source_closure()
    expected_hashes["bayesfilter/inference/surrogate_hmc_dctr_selection.py"] = digest(issuer_path)
    selected = issuer.load_selection(selection, expected_scope=scope,
                                     expected_source_hashes=expected_hashes)
    data_path = reference_root / "claim/data.json"
    data = json.loads(data_path.read_text())
    reference = json.loads((reference_root / "claim/reference.json").read_text())
    assert reference["qualified"] and reference["data_sha256"] == digest(data_path)
    for name, expected in json.loads((reference_root / "run-manifest.json").read_text())["source_hashes"].items():
        assert digest(name) == expected, name
    assert (data.get("attempt"), data.get("partition"), data.get("horizon"), data.get("model_id")) == (4, "claim", 10, "LG1")
    consumer = make_lg1_consumer(data, batch_size=4, controls=selected["controls"])
    provenance = dict(scope=scope, selection_path=str(selection.resolve()),
        selection_sha256=digest(selection), claim_data_sha256=digest(data_path),
        consumer=consumer.manifest, archived_issuer_path=str(issuer_path.resolve()),
        archived_issuer_sha256=digest(issuer_path), current_selection_module_sha256=digest("bayesfilter/inference/surrogate_hmc_dctr_selection.py"))
    endpoint, physical, _ = make_finite_endpoint(consumer)
    finite = json.loads((finite_root / "report.json").read_text())
    reference = json.loads((reference_root / "claim/reference.json").read_text())
    assert finite["qualified"] and reference["qualified"]
    assert finite["endpoint_identity"] == endpoint.identity
    for key in ("claim_data_sha256", "selection_sha256"):
        assert finite["provenance"][key] == provenance[key], key
    for name, expected in finite["source_hashes"].items():
        assert digest(name) == expected, name
    archived = finite["grids"][-1]
    inputs = {}
    def read_tensor(root, record):
        path = root / record["path"]
        assert digest(path) == record["sha256"], str(path)
        inputs[str(path)] = record["sha256"]
        return tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64)
    phi = read_tensor(finite_root, archived["tensors"]["phi"])
    logp = read_tensor(finite_root, archived["tensors"]["log_density"])
    summary = read_tensor(reference_root / "claim", reference["tensors"]["summary"])
    observations = tf.cast(consumer.fixture.observations, tf.float64)
    # This is the exact FP32 prior constant used by fixture.physical.
    prior = -tf.reduce_sum(tf.math.log(consumer.fixture.upper-consumer.fixture.lower))
    finite_logl = logp-tf.cast(prior, tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float64)],
                 jit_compile=True, autograph=False)
    def exact(points):
        logl, score, _ = kalman_physical(points, observations)
        lag = tf.abs(tf.range(10)[:, None]-tf.range(10)[None, :])
        covariance = (points[:, None, None] ** tf.cast(lag[None, :, :], tf.float64)
                      / (1-points[:, None, None]**2) + .5*tf.eye(10, dtype=tf.float64))
        chol = tf.linalg.cholesky(covariance)
        y = tf.broadcast_to(observations[None], [tf.size(points), 10, 1])
        whitened = tf.linalg.triangular_solve(chol, y)
        dense = (-5*tf.constant(math.log(2*math.pi), tf.float64)
                 -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
                 -.5*tf.reduce_sum(tf.square(whitened), axis=[1, 2]))
        return logl, score, dense

    logl, score, dense = exact(phi)
    independent_error = float(tf.reduce_max(tf.abs(logl-dense)))
    assert independent_error < 1e-10, independent_error
    exact_index, finite_index = int(tf.argmax(logl)), int(tf.argmax(finite_logl))
    indices = [exact_index, finite_index] + [int(tf.argmin(tf.abs(phi-q))) for q in (summary[2], summary[4])]
    fresh_points = tf.gather(phi, indices)[:, None]
    fresh_value, fresh_status = physical(fresh_points)
    repeat_value, repeat_status = physical(fresh_points)
    for status in (fresh_status, repeat_status):
        assert bool(tf.reduce_all(status["value_valid"]))
        assert bool(tf.reduce_all(status["force_valid"]))
    fresh_error = float(tf.reduce_max(tf.abs(fresh_value-tf.gather(logp, indices))))
    repeat_error = float(tf.reduce_max(tf.abs(fresh_value-repeat_value)))
    assert fresh_error < 5e-4, fresh_error
    assert repeat_error == 0., repeat_error

    def host(tensor):
        return tensor.numpy().tolist()
    def trap(values):
        return tf.reduce_sum((phi[1:]-phi[:-1])*(values[1:]+values[:-1])*.5)
    def normalized(values):
        peak = tf.reduce_max(values)
        weights = tf.exp(values-peak)
        area = trap(weights)
        return weights/area, peak+tf.math.log(area)
    exact_density, exact_logz = normalized(logl)
    finite_density, finite_logz = normalized(finite_logl)
    delta = finite_logl-logl
    percentage = 100*tf.math.expm1(delta)
    central = (phi >= summary[2]) & (phi <= summary[4])
    exact_cdf = tf.concat([tf.zeros([1], tf.float64), tf.cumsum(
        .5*(exact_density[1:]+exact_density[:-1])*(phi[1:]-phi[:-1]))], axis=0)
    finite_cdf = tf.concat([tf.zeros([1], tf.float64), tf.cumsum(
        .5*(finite_density[1:]+finite_density[:-1])*(phi[1:]-phi[:-1]))], axis=0)
    rows = []
    for label, index in zip(("exact_grid_mle", "dctr_grid_mle", "exact_q10_nearest", "exact_q90_nearest"), indices):
        rows.append(dict(label=label, phi=float(phi[index]), exact_log_likelihood=float(logl[index]),
            dctr_log_likelihood=float(finite_logl[index]), log_likelihood_difference=float(delta[index]),
            likelihood_difference_percent=float(percentage[index]),
            log_likelihood_magnitude_difference_percent=float(100*tf.abs(delta[index])/tf.abs(logl[index]))))
    report = dict(classification="diagnostic_only", data=provenance,
        point_comparisons=rows, grid_intervals=archived["intervals"], grid_spacing=float(phi[1]-phi[0]),
        prior_log_constant_removed=float(prior), observations_representation="FP32 data promoted to FP64",
        mle_shift=float(phi[finite_index]-phi[exact_index]),
        mle_shift_in_exact_posterior_sd=float((phi[finite_index]-phi[exact_index])/summary[1]),
        exact_likelihood_loss_at_dctr_mle_percent=float(100*(-tf.math.expm1(logl[finite_index]-logl[exact_index]))),
        ratio_of_maximum_likelihoods_percent=float(100*tf.math.expm1(finite_logl[finite_index]-logl[exact_index])),
        likelihood_percentage_range=host(tf.stack([tf.reduce_min(percentage), tf.reduce_max(percentage)])),
        central_80_percent_posterior_region=host(tf.gather(summary, [2, 4])),
        likelihood_percentage_range_in_central_80_percent=host(tf.stack([
            tf.reduce_min(tf.boolean_mask(percentage, central)), tf.reduce_max(tf.boolean_mask(percentage, central))])),
        integrated_likelihood_difference_percent=float(100*tf.math.expm1(finite_logz-exact_logz)),
        exact_posterior_weighted_absolute_likelihood_percent=float(trap(exact_density*tf.abs(percentage))),
        posterior_total_variation_grid=float(.5*trap(tf.abs(finite_density-exact_density))),
        posterior_cdf_max_abs_difference_grid=float(tf.reduce_max(tf.abs(finite_cdf-exact_cdf))),
        preserved_D_vs_R=finite["D_vs_R"],
        checks=dict(input_hashes_and_scope_match=True, endpoint_identity_match=True,
            kalman_dense_max_log_error=independent_error, fresh_archived_max_log_error=fresh_error,
            repeated_dctr_max_log_error=repeat_error, fresh_force_and_value_valid=True),
        no_new_posterior_chain=True, sampler_promotion=False, default_readiness=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    columns = [host(t) for t in (phi, logl, finite_logl, percentage, exact_density, finite_density)]
    with (args.output / "likelihood-curve.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phi", "exact_log_likelihood", "dctr_log_likelihood", "likelihood_difference_percent", "exact_posterior_density", "dctr_posterior_density"])
        writer.writerows(zip(*columns))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(columns[0], host(tf.exp(logl-tf.reduce_max(logl))), label="Exact Kalman")
    axes[0].plot(columns[0], host(tf.exp(finite_logl-tf.reduce_max(logl))), label="DCTR", linestyle="--")
    axes[0].set_ylabel("Likelihood / exact maximum")
    axes[0].legend()
    axes[0].set_title("LG1: same observations, frozen DCTR controls")
    axes[1].plot(columns[0], columns[3], color="#a54a10")
    axes[1].axhline(0, color="gray", linewidth=.8)
    axes[1].set_ylabel("DCTR likelihood difference (%)")
    axes[1].set_xlabel("Autoregressive parameter φ")
    for ax in axes:
        ax.axvspan(float(summary[2]), float(summary[4]), color="#3678a6", alpha=.09)
        ax.grid(alpha=.18)
    fig.savefig(args.output / "likelihood-comparison.png", dpi=160)
    plt.close(fig)
    inputs.update({str(path): digest(path) for path in (finite_root/"report.json", reference_root/"claim/reference.json", selection)})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    manifest = dict(command=sys.argv, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        conda_env=sys.prefix, tensorflow=tf.__version__, cpu_only=False, gpu_visible=[x.name for x in tf.config.list_physical_devices("GPU")],
        gpu_details=[tf.config.experimental.get_device_details(x) for x in tf.config.list_physical_devices("GPU")],
        memory_policy=memory, tf32=True, jit_compile=True, seeds="Frozen attempt-4 data/design; no new random draws",
        inputs=inputs, source_hashes={str(Path(__file__).resolve().relative_to(Path.cwd())):digest(__file__), **source_closure(), **finite["source_hashes"]},
        wall_seconds=time.monotonic()-start, cpu_seconds=usage.ru_utime+usage.ru_stime,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),
        plan="docs/plans/surrogate-hmc-lg1-likelihood-diagnostic-2026-09-22.md",
        result="docs/plans/surrogate-hmc-lg1-likelihood-result-2026-09-22.md",
        outputs={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
    (args.output / "run-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"points":rows, "checks":report["checks"], "wall_seconds":manifest["wall_seconds"], "cpu_seconds":manifest["cpu_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
