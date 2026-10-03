"""Assessed batched reverse-KL training with exact Adam continuation.

Uses the existing TensorFlow numerical trainer and frozen transport codec.
Heldout intervals describe IID base-bank Monte Carlo error; adaptive reuse is
explicit and does not provide anytime-valid coverage or posterior evidence.
"""
from __future__ import annotations

from dataclasses import fields
import json
import time

import tensorflow as tf

from bayesfilter.inference.q20_production_config import digest, write_json
from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport
from bayesfilter.inference.tempered_transport_ensemble_tf import (
    IndependentTemperedReverseKLTrainer, ReferenceAffineTransport,
    capture_trainable_transport_checkpoint, restore_trainable_transport_checkpoint,
    prepare_transport_initialization,
)
from bayesfilter.inference.neutra_artifacts import (
    finalize_dense_iaf_neutra_artifact_payload, load_frozen_neutra_artifact,
)


def _variables(optimizer):
    values = optimizer.variables
    return tuple(values() if callable(values) else values)


def _tensor_record(variable):
    tensor = tf.convert_to_tensor(variable)
    return {"shape": tensor.shape.as_list(), "dtype": tensor.dtype.name,
            "value": tensor.numpy().tolist()}


def _checked_assignments(variables, records):
    if len(variables) != len(records):
        raise ValueError("optimizer checkpoint variable count differs")
    assignments = []
    for variable, record in zip(variables, records, strict=True):
        dtype = tf.as_dtype(variable.dtype)
        if list(variable.shape) != record["shape"] or dtype.name != record["dtype"]:
            raise ValueError("optimizer checkpoint dtype or shape differs")
        value = tf.convert_to_tensor(record["value"], dtype)
        if dtype.is_floating:
            tf.debugging.assert_all_finite(value, "optimizer checkpoint")
        assignments.append((variable, value))
    return assignments


class TrainingSession:
    """One beta/seed scope; its complete state can be restarted without drift."""

    def __init__(self, *, trainer, scope, root_seed, rng_index=0, history=None,
                 level_updates=0, parent_hash=None):
        self.trainer = trainer
        self.scope = json.loads(json.dumps(scope, allow_nan=False))
        self.root_seed = tuple(root_seed)
        self.rng_index = int(rng_index)
        self.level_updates = int(level_updates)
        self.history = json.loads(json.dumps(history or [], allow_nan=False))
        self.parent_hash = parent_hash

    def advance(self, count, *, log_path=None, budget_check=None):
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        for _ in range(count):
            if budget_check is not None and not budget_check():
                return "budget_exhausted"
            seed = tf.random.experimental.stateless_fold_in(
                tf.constant(self.root_seed, tf.int32), self.rng_index)
            seed_value = seed.numpy().tolist()
            self.rng_index += 1
            variables = (*self.trainer.variables, *_variables(self.trainer.optimizer), self.trainer.step)
            saved = [tf.identity(variable) for variable in variables]
            started = time.monotonic()
            try:
                step = self.trainer.train_step(seed)
                for variable in variables:
                    if tf.as_dtype(variable.dtype).is_floating:
                        tf.debugging.assert_all_finite(tf.convert_to_tensor(variable), "post-update state")
                row = {field.name: getattr(step, field.name).numpy().item() for field in fields(step)}
                self.level_updates += 1
                row.update(status="accepted", level_updates=self.level_updates)
            except Exception as error:
                for variable, value in zip(variables, saved, strict=True):
                    variable.assign(value)
                row = {"status": "rejected", "error_type": type(error).__name__,
                       "error": str(error), "level_updates": self.level_updates}
                row.update(seed=seed_value, rng_index=self.rng_index,
                           beta=self.trainer.beta, wall_seconds=time.monotonic() - started)
                self.history.append(row)
                if log_path is not None:
                    with open(log_path, "a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                raise
            row.update(seed=seed_value, rng_index=self.rng_index,
                       beta=self.trainer.beta, wall_seconds=time.monotonic() - started)
            self.history.append(row)
            if log_path is not None:
                with open(log_path, "a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        return "completed_requested_updates"

    def checkpoint(self):
        trainer = self.trainer
        map_state = capture_trainable_transport_checkpoint(
            trainer.transport, component_id=trainer.component_id, beta=trainer.beta,
            bridge_signature=trainer.target_bridge.signature,
            target_signature=trainer.target_bridge.target_signature,
            parent_checkpoint_hash=self.parent_hash, update_count=int(trainer.step.numpy()),
            checkpoint_scope=self.scope)
        state = {"schema": "bayesfilter.neutra.complete_training_state.v1",
                 "map": map_state, "scope": self.scope, "root_seed": list(self.root_seed),
                 "rng_index": self.rng_index, "level_updates": self.level_updates,
                 "batch_size": trainer.batch_size,
                 "optimizer": [_tensor_record(v) for v in _variables(trainer.optimizer)],
                 "iteration": int(trainer.step.numpy()), "history": json.loads(json.dumps(self.history)),
                 "parent_hash": self.parent_hash}
        return {**state, "state_hash": digest(state)}

    @classmethod
    def restore(cls, state, *, bridge, config, expected_scope, preflight_seed):
        payload = dict(state)
        expected_hash = payload.pop("state_hash")
        if digest(payload) != expected_hash or payload["schema"] != "bayesfilter.neutra.complete_training_state.v1":
            raise ValueError("complete training checkpoint checksum/schema differs")
        if payload["scope"] != expected_scope:
            raise ValueError("training scope changed")
        transport = restore_trainable_transport_checkpoint(payload["map"], expected_context={
            "bridge_signature": bridge.signature, "target_signature": bridge.target_signature})
        inner = transport.inner if isinstance(transport, ReferenceAffineTransport) else transport
        if inner.config.manifest_payload() != config.manifest_payload():
            raise ValueError("training configuration changed")
        prepared = prepare_transport_initialization(transport, bridge,
            component_id=payload["map"]["component_id"], beta=payload["map"]["beta"],
            seed=preflight_seed, batch_size=payload["batch_size"], repair_scales=(1.0,))
        trainer = IndependentTemperedReverseKLTrainer(config, bridge,
            beta=payload["map"]["beta"], component_id=payload["map"]["component_id"],
            batch_size=payload["batch_size"], prepared_initialization=prepared)
        assignments = _checked_assignments(_variables(trainer.optimizer), payload["optimizer"])
        if int(payload["optimizer"][0]["value"]) != payload["iteration"]:
            raise ValueError("Adam iteration and training iteration differ")
        for variable, value in assignments:
            variable.assign(value)
        trainer.step.assign(payload["iteration"])
        return cls(trainer=trainer, scope=expected_scope, root_seed=payload["root_seed"],
                   rng_index=payload["rng_index"], level_updates=payload["level_updates"],
                   history=payload["history"], parent_hash=payload["parent_hash"])

    def next_beta(self, beta, *, root_seed, preflight_seed):
        if not self.trainer.beta < beta <= 1.0:
            raise ValueError("continuation beta must increase")
        previous = self.checkpoint()
        prepared = prepare_transport_initialization(self.trainer.transport,
            self.trainer.target_bridge, component_id=self.trainer.component_id,
            beta=beta, seed=preflight_seed, batch_size=self.trainer.batch_size,
            repair_scales=(1.0,))
        trainer = IndependentTemperedReverseKLTrainer(self.trainer.config,
            self.trainer.target_bridge, beta=beta, component_id=self.trainer.component_id,
            batch_size=self.trainer.batch_size, prepared_initialization=prepared)
        for variable, value in _checked_assignments(_variables(trainer.optimizer), previous["optimizer"]):
            variable.assign(value)
        trainer.step.assign(previous["iteration"])
        return TrainingSession(trainer=trainer, scope=self.scope, root_seed=root_seed,
                               parent_hash=previous["state_hash"], history=self.history)


def export_weighted_transport(transport, *, target_signature, training_state_hash, transport_id):
    """Export the exact trained map, including its outer affine and permutations."""
    inner = transport.inner if isinstance(transport, ReferenceAffineTransport) else transport
    if not isinstance(inner, WeightedDenseIAFTransport):
        raise ValueError("export requires weighted dense IAF with optional outer affine")
    config, dimension = inner.config, inner.parameter_dim
    components = []
    for index, stage in enumerate(inner.stages):
        if stage.scale_linear_skip_enabled or stage.unbounded_scale_linear_enabled:
            raise ValueError("frozen codec cannot represent extra scale-linear paths")
        components.append({"kind": "dense_autoregressive_iaf", "component_id": f"iaf-{index}",
            "dim": dimension, "hidden_layers": list(config.hidden_layers),
            "activation": config.activation, "masks_policy": "legacy_degree_masks_v1",
            "s_max": stage.s_max, "scale_transform": "bounded_tanh",
            "weights": [v.numpy().tolist() for v in stage.weights],
            "biases": [v.numpy().tolist() for v in stage.biases]})
        if index + 1 < len(inner.stages):
            if config.permutation_policy == "full_reverse":
                permutation = list(reversed(range(dimension)))
            elif config.permutation_policy == "root_preserving_reverse":
                permutation = [0, *reversed(range(1, dimension))]
            else:
                raise ValueError("unsupported between-stage permutation")
            components.append({"kind": "mixing_linear", "component_id": f"mix-{index}",
                "dim": dimension, "matrix": [[float(j == permutation[i]) for j in range(dimension)]
                                                for i in range(dimension)]})
    if isinstance(transport, ReferenceAffineTransport):
        components.append({"kind": "affine", "component_id": "prior-affine", "dim": dimension,
                           "offset": transport.center.numpy().tolist(), "scale": transport.scale.numpy().tolist()})
    payload = finalize_dense_iaf_neutra_artifact_payload({
        "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
        "transport_id": transport_id, "dimension": dimension, "target_signature": target_signature,
        "training_state_hash": training_state_hash, "log_jacobian_available": True,
        "procedure": "bayesfilter_assessed_weighted_reverse_kl_v1",
        "components": components, "component_order": [v["component_id"] for v in components]})
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=target_signature)
    return payload, loaded.transport


def transport_parity(transport, frozen, latent, *, rtol, atol):
    """Map engineering checks only; these points cannot establish mode coverage."""
    physical, logdet = transport.forward_and_logdet(latent)
    frozen_physical = frozen.forward_batch(latent)
    frozen_logdet = frozen.log_abs_det_jacobian_batch(latent)
    recovered = frozen.inverse_theta_to_z_batch(physical)
    score = tf.ones_like(physical)  # Independent cotangent; no target evaluation.
    comparisons = {
        "forward": (physical, frozen_physical), "inverse": (latent, recovered),
        "logdet": (logdet, frozen_logdet),
        "pullback": (transport.pullback_score_batch(latent, score), frozen.pullback_score_batch(latent, score)),
        "logdet_score": (transport.log_abs_det_jacobian_score_batch(latent),
                         frozen.log_abs_det_jacobian_score_batch(latent)),
    }
    result = {}
    for name, (left, right) in comparisons.items():
        finite = bool(tf.reduce_all(tf.math.is_finite(left) & tf.math.is_finite(right)).numpy())
        residual = tf.abs(left - right)
        passed = finite and bool(tf.reduce_all(residual <= atol + rtol * tf.abs(left)).numpy())
        result[name] = {"passed": passed, "max_absolute_error": float(tf.reduce_max(residual).numpy()) if finite else None}
    return {"passed": all(row["passed"] for row in result.values()), "checks": result,
            "scope": "heldout_map_numerics_only", "posterior_coverage_checked": False}


class HeldoutLoss:
    """Bounded stable-shape batched validation, with no parameter updates."""

    def __init__(self, transport, bridge, beta, *, batch_size, jit_compile):
        self.dimension, self.batch_size = bridge.parameter_dim, batch_size
        def values(latent):
            physical, logdet = transport.forward_and_logdet(latent)
            target, score, status = bridge.value_score_status(physical, tf.constant(beta, tf.float64))
            valid = status["bridge_valid"] & tf.math.is_finite(target) & tf.math.is_finite(logdet)
            valid &= tf.reduce_all(tf.math.is_finite(score), axis=-1)
            return -target - logdet, valid
        self.compiled = tf.function(values,
            input_signature=(tf.TensorSpec([batch_size, self.dimension], tf.float64),),
            jit_compile=jit_compile, reduce_retracing=False)

    def __call__(self, count, seed):
        rows = []
        # A CPU bank is independent of GPU optimizer randomness. The same chunks
        # extend the bank prefix deterministically at later validation rungs.
        for start in range(0, count, self.batch_size):
            with tf.device("/CPU:0"):
                z = tf.random.stateless_normal([self.batch_size, self.dimension],
                    tf.random.experimental.stateless_fold_in(seed, start // self.batch_size), dtype=tf.float64)
            loss, valid = self.compiled(z)
            length = min(self.batch_size, count - start)
            if not bool(tf.reduce_all(valid[:length]).numpy()):
                raise ValueError("invalid heldout target/map row; no resampling allowed")
            rows.append(loss[:length])
        return tf.concat(rows, axis=0)


def paired_loss_statistics(before, after, *, multiplier):
    before, after = tf.convert_to_tensor(before, tf.float64), tf.convert_to_tensor(after, tf.float64)
    if before.shape != after.shape:
        raise ValueError("paired banks must have equal shapes; broadcasting is forbidden")
    difference = after - before
    if difference.shape.rank != 1 or difference.shape[0] < 2:
        raise ValueError("paired banks require the same nontrivial vector shape")
    tf.debugging.assert_all_finite(difference, "paired losses")
    n = int(difference.shape[0])
    mean = tf.reduce_mean(difference)
    se = tf.sqrt(tf.reduce_sum(tf.square(difference - mean)) / (n * (n - 1)))
    mean, se = float(mean.numpy()), float(se.numpy())
    half = multiplier * se
    return {"mean": mean, "standard_error": se, "half_width": half,
            "lower": mean - half, "upper": mean + half, "rows": n,
            "inference": "descriptive_normal_interval_on_adaptively_reused_iid_base_bank"}


def assess_training_rung(*, baseline, increment, reliability, prior_plateaus,
                         at_cap, minimum_improvement, maximum_half_width, plateau_comparisons):
    if not reliability:
        return {"status": "numerically_invalid", "plateaus": 0, "development_eligible": False}
    precise = max(baseline["half_width"], increment["half_width"]) <= maximum_half_width
    learning = baseline["upper"] < -minimum_improvement
    plateau = precise and increment["lower"] >= -minimum_improvement and increment["upper"] <= minimum_improvement
    plateaus = prior_plateaus + 1 if plateau else 0
    if learning and plateaus >= plateau_comparisons:
        status = "plateau_nominee"
    elif increment["lower"] > minimum_improvement:
        status = "deterioration_repair_trigger"
    elif at_cap:
        status = "cap_learning_observed" if learning else "cap_learning_unresolved"
    else:
        status = "continue_training" if precise else "expand_validation_or_continue"
    return {"status": status, "plateaus": plateaus,
            "learning_observed": learning, "precision_screen": precise,
            "development_eligible": status == "plateau_nominee",
            "posterior_qualified": False}
