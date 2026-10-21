"""Independent quadrature and qualification loading for the LG1 finite endpoint."""
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from .surrogate_hmc_lg1_reference import _trapz, grid_functions, evaluate_kalman


def validate_endpoint_qualification(root, endpoint, provenance):
    path = Path(root) / "report.json"
    report = json.loads(path.read_text())
    if report.get("verdict") != "PASS" or not all(report["checks"].values()):
        raise ValueError("finite endpoint qualification failed")
    if report["endpoint_identity"] != endpoint.identity:
        raise ValueError("finite endpoint identity changed after qualification")
    for key in ("selection_sha256", "claim_data_sha256"):
        if report["provenance"][key] != provenance[key]:
            raise ValueError("finite endpoint qualification scope mismatch")
    for name, digest in report["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("finite endpoint qualification source changed")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@tf.function(input_signature=[tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([5], tf.float64)], jit_compile=True, autograph=False)
def normalize_grid(phi, log_density, predictive, summary):
    maximum = tf.reduce_max(log_density)
    weights = tf.exp(log_density-maximum)
    normalizer = _trapz(phi, weights)
    density = weights/normalizer
    return density, grid_functions(phi, density, predictive, summary), maximum+tf.math.log(normalizer)


def compare_references(value, allowance, reference, reference_allowance, margins, qualified):
    difference = value-reference
    radius = allowance+reference_allowance
    lower, upper = difference-radius, difference+radius
    equivalent = (lower > -margins) & (upper < margins)
    discrepant = (lower > margins) | (upper < -margins)
    status = ("EQUIVALENT" if qualified and bool(tf.reduce_all(equivalent)) else
              "DISCREPANCY" if qualified and bool(tf.reduce_any(discrepant)) else "INCONCLUSIVE")
    return dict(status=status, difference=difference.numpy().tolist(), lower=lower.numpy().tolist(),
                upper=upper.numpy().tolist(), equivalent=equivalent.numpy().tolist(),
                discrepant=discrepant.numpy().tolist(), rigorous_error_bound=False,
                uncertainty="sum of empirical grid refinement allowances; not a confidence interval")
