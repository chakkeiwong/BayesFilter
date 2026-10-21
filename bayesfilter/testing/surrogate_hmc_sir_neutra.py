"""Matched LEDH training density and frozen SIR NeuTra inference connection.

Training uses log prior + the finite LEDH likelihood and its analytical score.
MH uses the separately declared inference endpoint. No independent-reference
value is paired with an LEDH score in a training objective.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation, bind_neural_force_hmc_tuning_runner,
    _evaluate_force, _evaluate_endpoint_potential)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class SIRNeuTraTarget:
    """Native [B,3] target; B>1. Numerical telemetry remains available."""
    def __init__(self, case, batch_size, *, jit_compile=True):
        if case.name != "sir_d18" or batch_size < 2:
            raise ValueError("SIR training requires a native batch larger than one")
        self.case = replace(case)
        self.batch_size = int(batch_size)
        self.dtype = case.force_dtype
        self.parameter_dim = case.p
        n = int(case.fixed[0].shape[0])
        self.program = make_batched_score_filter(case.spec, n, case.horizon,
            case.controls, self.dtype, jit_compile=jit_compile,
            batch_size=batch_size, precision_policy=case.precision_policy)
        self.scope = training_scope(case, batch_size, jit_compile=jit_compile)
        self.target_signature = digest(self.scope)
        signature = [tf.TensorSpec([batch_size, case.p], self.dtype)]
        self.evaluate = tf.function(self._evaluate, input_signature=signature,
                                    jit_compile=jit_compile, autograph=False)
        self.value_score = tf.function(self._value_score, input_signature=signature,
                                      jit_compile=jit_compile, autograph=False)

    def _evaluate(self, q):
        case = self.case
        theta, jacobian = case.chart(q)
        result = self.program(theta, tf.cast(case.observations, self.dtype),
            *(tf.cast(x, self.dtype) for x in case.fixed), tf.cast(case.beta, self.dtype))
        prior, prior_score = case.log_prior(q)
        value = result["value"]+prior
        score = result["score"]*jacobian+prior_score
        valid = (result["valid"] & (result["steps_completed"] == case.horizon)
                 & tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), -1))
        return {**result, "log_likelihood":result["value"], "likelihood_score":result["score"],
                "value":value, "score":score, "valid":valid}

    def _value_score(self, q):
        result = self._evaluate(q)
        return result["value"], result["score"], result["valid"]


def training_scope(case, batch_size, *, jit_compile=True):
    """Recompute training identity from the current case, without tracing."""
    if case.name != "sir_d18" or not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("SIR training scope requires native batch size >1")
    root = Path(__file__).resolve().parents[2]
    files = ("bayesfilter/testing/surrogate_hmc_sir_neutra.py",
             "bayesfilter/highdim/covariance_proposal_native_models_tf.py",
             "bayesfilter/highdim/covariance_proposal_nonshared_gaussian_tf.py")
    scope = case.scope()
    scope.update(particles=int(case.fixed[0].shape[0]), force_batch_size=batch_size,
                 training_target="matched_finite_LEDH_loglik_plus_prior",
                 controls=asdict(case.controls), beta=case.beta.numpy().tolist(),
                 observations_sha256=hashlib.sha256(tf.io.serialize_tensor(case.observations).numpy()).hexdigest(),
                 chart_center_sha256=hashlib.sha256(tf.io.serialize_tensor(case.theta0).numpy()).hexdigest(),
                 fixed_design_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest()
                                      for x in case.fixed],
                 additional_source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest()
                                           for f in files},
                 jit_compile=bool(jit_compile))
    return scope


def verify_sir_training_scope(case, payload, scope):
    """Reject a map from different data, prior, controls or numerical source."""
    if not isinstance(scope, dict) or scope.get("jit_compile") is not True:
        raise ValueError("a SIR map requires its recorded GPU/XLA training scope")
    expected = training_scope(case, scope.get("force_batch_size"))
    signature = digest(expected)
    if digest(scope) != signature or payload.get("target_signature") != signature:
        raise ValueError("SIR map training scope differs from the current case")
    cfg = payload.get("config", {})
    profile = dict(kind="iaf", dimension=case.p, stages=3, activation="elu",
        mask_policy="hoffman_block_masks_v1", permutation_policy="full_reverse",
        scale_transform="neutra_conditional_tanh", iaf_initializer="hoffman_variance_scaling",
        iaf_variance_scale=.02, conditional_scale_cap=2., dtype="float64")
    if any(cfg.get(k) != v for k, v in profile.items()) or len(cfg.get("hidden_layers", ())) != 2:
        raise ValueError("SIR HMC requires the canonical author IAF with explicit FP64 conversion")
    return signature


def frozen_neutra_bindings(payload, *, target_signature, raw_force, raw_target,
                           target_scope):
    """Compose a verified fixed map with force and independent MH endpoint.

    U_z(z)=U_q(T(z))-log|det DT(z)| and
    F_z=DT(z)^T F_q(T(z))-grad_z log|det DT(z)|.
    The map derivative is analytical; the filter score remains analytical.
    """
    if raw_force.coordinate_system != "raw" or raw_target.coordinate_system != "raw":
        raise ValueError("expected physical-coordinate force and endpoint")
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=target_signature)
    transport = loaded.transport
    dimension = transport.parameter_dim
    dtype = transport.dtype
    identity = loaded.manifest.transport_hash

    @tf.function(input_signature=[tf.TensorSpec([None, dimension], dtype)],
                 jit_compile=True, autograph=False)
    def force(z):
        q = transport.forward_batch(z)
        value, fallback = _evaluate_force(raw_force, q, dimension=dimension, dtype=dtype)
        valid = ~fallback
        pulled = (transport.pullback_score_batch(z, value)
                  -transport.log_abs_det_jacobian_score_batch(z))
        valid &= tf.reduce_all(tf.math.is_finite(pulled), -1)
        return PositionOnlyForceEvaluation(pulled, valid)

    @tf.function(input_signature=[tf.TensorSpec([None, dimension], dtype)],
                 jit_compile=True, autograph=False)
    def endpoint(z):
        q, ld = transport.forward_and_logdet(z)
        value, status = _evaluate_endpoint_potential(raw_target, q, dtype=dtype,
                                                     return_status=True)
        potential = value-ld
        finite_map = tf.math.is_finite(ld) & tf.reduce_all(tf.math.is_finite(q), -1)
        status = {**status, "value_valid":status["value_valid"] & finite_map,
                  "numerical_failure":status["numerical_failure"] | ~finite_map}
        return TargetPotentialEvaluation(potential, status)

    fixed_force = FrozenPositionOnlyForce(force, raw_force.identity+":"+identity,
        semantics=raw_force.semantics, coordinate_system="transformed")
    fixed_target = FrozenTargetPotential(endpoint, raw_target.identity+":"+identity,
        coordinate_system="transformed", includes_chart_log_jacobian=True)
    binding = bind_neural_force_hmc_tuning_runner(force=fixed_force, target=fixed_target,
        target_scope=target_scope+":"+identity, frozen_transport=loaded)
    return dict(force=fixed_force, target=fixed_target, binding=binding,
                transport=transport, artifact=loaded)
