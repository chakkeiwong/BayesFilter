"""Diagnostic precision isolation; never a tuning or HMC implementation.

Execute the unchanged shared filter in a private module namespace. Only pairwise
stage boundaries change arithmetic precision. The canonical runtime module is
not patched. Mixed precision does not establish FP32 or canonical LEDH admission.
"""
import hashlib
import importlib.util
from pathlib import Path
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared

POLICIES = ("both_fp32", "gaussian_fp32", "transport_fp32")


def cast_jet(value, dtype):
    return Jet(tf.cast(value.v, dtype), tf.cast(value.t, dtype))


def private_module(policy):
    if policy not in POLICIES:
        raise ValueError("unknown diagnostic pairwise precision policy")
    source = Path(shared.__file__).resolve()
    spec = importlib.util.spec_from_file_location(
        "_surrogate_hmc_pairwise_precision_diagnostic", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_gaussian = module.gaussian_mixture
    original_transport = module.transport

    def gaussian(x, means, covariance, shared_covariance=False, contraction="direct"):
        # Subtract the same moving origin from points, means and all tangents.
        origin = Jet(x.v[:, :1], x.t[:, :, :1])
        density, valid = original_gaussian(
            cast_jet(x-origin, tf.float32), cast_jet(means-origin, tf.float32),
            cast_jet(covariance, tf.float32), shared_covariance, contraction)
        return cast_jet(density, x.v.dtype), valid

    def transport(x, log_weights, controls):
        origin = Jet(x.v[:, :1], x.t[:, :, :1])
        transported, row_error, column_error = original_transport(
            cast_jet(x-origin, tf.float32), cast_jet(log_weights, tf.float32), controls)
        return (cast_jet(transported, x.v.dtype)+origin,
                tf.cast(row_error, x.v.dtype), tf.cast(column_error, x.v.dtype))

    if policy in ("both_fp32", "gaussian_fp32"):
        module.gaussian_mixture = gaussian
    if policy in ("both_fp32", "transport_fp32"):
        module.transport = transport
    return module


def make_program(spec, n, horizon, controls, *, policy, batch_size=2):
    module = private_module(policy)
    program = module.make_batched_score_filter(
        spec, n, horizon, controls, tf.float64, jit_compile=True, batch_size=batch_size)
    metadata = dict(
        policy=policy, recurrence_dtype="float64",
        gaussian_pair_dtype="float32" if policy != "transport_fp32" else "float64",
        transport_pair_dtype="float32" if policy != "gaussian_fp32" else "float64",
        centering_before_cast=True, evidence_role="arithmetic_localization_only",
        runtime_module_patched=False,
        numerical_source_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return program, metadata
