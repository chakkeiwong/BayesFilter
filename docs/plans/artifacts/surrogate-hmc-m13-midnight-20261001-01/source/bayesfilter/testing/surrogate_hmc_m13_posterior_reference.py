"""Independent finite-box M13 posterior diagnostic; never an HMC endpoint."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_posterior_assessment import (
    F64, grid_density, linear_density_moments, piecewise_cdf, reference_quantiles,
)
from bayesfilter.testing.surrogate_hmc_m13_same_data_refine import checked_grid, DEPENDENCIES
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import FRESH_SEEDS

SCHEMA = "m13_bilinear_independent_posterior_reference_v1"
REFINED = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-refine-01"
DOMAIN = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-domain-sensitivity-01"
SOURCE_PATHS = (
    "bayesfilter/testing/surrogate_hmc_m13_posterior_reference.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_assessment.py",
)


def payload_hash(payload):
    return hashlib.sha256(json.dumps({k: v for k, v in payload.items()
        if k != "reference_sha256"}, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def tensor_record(path, tensor):
    tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
    return dict(path=path.name, sha256=file_hash(path), dtype=tensor.dtype.name,
                shape=tensor.shape.as_list())


def read_tensor(root, record):
    path = Path(root) / record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("posterior reference tensor checksum mismatch")
    value = tf.io.parse_tensor(tf.io.read_file(str(path)), tf.as_dtype(record["dtype"]))
    if value.shape.as_list() != record["shape"]:
        raise ValueError("posterior reference tensor shape mismatch")
    return value


def checked_report(root, stage):
    parent = json.loads((root / "report.json").read_text())
    if parent.get("source_unchanged") is not True:
        raise ValueError("reference parent source verification failed")
    snapshot = json.loads((root / "run-manifest.json").read_text())
    for name in DEPENDENCIES:
        if snapshot["source_hashes"].get(name) != file_hash(ROOT / name):
            raise ValueError("reference numerical source changed: " + name)
    path = root / stage / "report.json"
    report = json.loads(path.read_text())
    if report.get("qualified") is not True or report.get("invalid_nodes"):
        raise ValueError("saved reference did not pass its scoped checks")
    return path, report


def reference_summary(axis, logs):
    n, replicates = int(axis.shape[0]), int(logs.shape[-1])
    if n < 3 or replicates < 2 or logs.shape.as_list() != [n, n, replicates]:
        raise ValueError("invalid reference shape")
    tf.debugging.assert_all_finite(logs, "nonfinite reference")
    tf.debugging.assert_positive(axis[1:]-axis[:-1])
    tf.debugging.assert_near(axis[1:]-axis[:-1],
        tf.fill([n-1], (axis[-1]-axis[0])/tf.cast(n-1, F64)), atol=1e-14, rtol=1e-14)
    prior = -(axis[:, None]**2 + axis[None, :]**2)/(2*math.log(2.)**2)
    density, marginal, replicate_marginal, ratios = grid_density(axis, logs+prior[..., None])
    mean, sd = linear_density_moments(axis, marginal)
    tf.debugging.assert_positive(sd)
    cuts = reference_quantiles(axis, marginal)
    per_mean, per_sd = linear_density_moments(axis, replicate_marginal)
    second = sd**2+mean**2
    mean_if = ratios[:, None]*(per_mean-mean)
    sd_if = ratios[:, None]*((per_sd**2+per_mean**2-second) -
                              2*mean*(per_mean-mean))/(2*sd)
    cdf = piecewise_cdf(axis, replicate_marginal, cuts[None])
    cdf_if = ratios[:, None, None]*(cdf-tf.constant([.25, .5, .75], F64))
    influence = tf.concat([mean_if/sd, sd_if/sd,
                            tf.reshape(cdf_if, [replicates, 6])], -1)
    mcse = tf.sqrt(tf.reduce_sum((influence-tf.reduce_mean(influence, 0))**2, 0)/
                   (replicates*(replicates-1)))
    return dict(density=density, marginal=marginal, mean=mean, sd=sd, cuts=cuts,
                mcse=mcse, influence=influence)


def prepare_reference(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    refined_path, refined = checked_report(REFINED, "m13-same-data-refine")
    domain_path, domain = checked_report(DOMAIN, "m13-domain-sensitivity")
    if (domain["baseline_sha256"] != file_hash(refined_path)
            or domain["observation_checksum"] != refined["observation_checksum"]):
        raise ValueError("reference/domain data linkage mismatch")
    tensors, source_grids, axis = [], {}, None
    for seed in FRESH_SEEDS:
        key = f"n16384-s{seed[0]}"
        path = refined_path.parent / key / "grid.json"
        grid = checked_grid(path, expected_hash=refined["diagnostics"][key]["grid_sha256"],
            checksum=refined["observation_checksum"], particles=16384, seed=seed, order=129)
        current_axis = tf.constant(grid["axis"], F64)
        if axis is not None:
            tf.debugging.assert_equal(axis, current_axis)
        axis = current_axis
        tensors.append(tf.constant(grid["log_likelihood"], F64))
        source_grids[str(path)] = file_hash(path)
    logs = tf.concat(tensors, -1)
    fine = reference_summary(axis, logs)
    coarse = reference_summary(axis[::2], logs[::2, ::2])
    mean, sd, cuts = fine["mean"], fine["sd"], fine["cuts"]
    resolution = tf.abs(tf.concat([(fine["mean"]-coarse["mean"])/sd,
        (fine["sd"]-coarse["sd"])/sd,
        tf.reshape(piecewise_cdf(axis[::2], coarse["marginal"], cuts)-
                   tf.constant([.25, .5, .75], F64), [6])], 0))
    # Existing positive-weight domain quadrature supplies measured shell effects.
    # These are empirical envelopes, not full-support tail bounds.
    domain_check = domain["domain_comparisons"][1]
    domain_allowance = tf.constant(domain_check["envelope_in_posterior_sd"]+
        [domain_check["shell_probability_envelope"]]*6, F64)
    count = int(logs.shape[-1])
    critical = tfp.distributions.StudentT(tf.constant(float(count-1), F64),
        tf.constant(0., F64), tf.constant(1., F64)).quantile(
            tf.constant(1-.025/(2*10), F64))
    replication = critical*fine["mcse"]
    payload = dict(schema=SCHEMA, data_seed=1704, horizon=20,
        observation_checksum=refined["observation_checksum"],
        law="normalized bilinear posterior-density interpolant on the saved finite box",
        full_support_target=False, global_tail_bound=False, replicates=count,
        mean=mean.numpy().tolist(), sd=sd.numpy().tolist(), cuts=cuts.numpy().tolist(),
        replication_mcse=fine["mcse"].numpy().tolist(),
        replication_allowance=replication.numpy().tolist(),
        resolution_allowance=resolution.numpy().tolist(),
        domain_allowance=domain_allowance.numpy().tolist(),
        empirical_allowance=(replication+resolution+domain_allowance).numpy().tolist(),
        axis=tensor_record(out/"axis.tfbin", axis),
        density=tensor_record(out/"density.tfbin", fine["density"]),
        sources={str(refined_path): file_hash(refined_path),
                 str(domain_path): file_hash(domain_path), **source_grids},
        source_hashes={p: file_hash(ROOT/p) for p in SOURCE_PATHS},
        numerical_backend="tensorflow_only", ranking_supported=False)
    payload["reference_sha256"] = payload_hash(payload)
    write_json(out/"reference.json", payload)
    return payload


def load_reference(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("reference_sha256") != payload_hash(payload):
        raise ValueError("posterior reference identity mismatch")
    for name, digest in payload["source_hashes"].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError("posterior reference numerical source changed")
    for name, digest in payload["sources"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("posterior reference source artifact changed")
    axis = read_tensor(path.parent, payload["axis"])
    density = read_tensor(path.parent, payload["density"])
    tf.debugging.assert_all_finite(density, "invalid reference density")
    tf.debugging.assert_non_negative(density)
    return payload, axis, density
