"""Configured NeuTra maps and batched training using one numerical authority.

See docs/reference/neutra-implementation.md for source anchors and limitations.
Old public trainers retain their saved configurations and delegate to the same
core. New artifacts use an explicit schema rather than relabeling NAF as IAF.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import tensorflow as tf

from bayesfilter.inference import neutra_transport_core as core

FROZEN_SCHEMA = "bayesfilter.neutra.configured_frozen_transport.v1"
CHECKPOINT_SCHEMA = "bayesfilter.neutra.configured_training_checkpoint.v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class NeuTraTransportConfig:
    dimension: int
    kind: str
    hidden_layers: tuple[int, ...]
    stages: int
    activation: str
    seed: tuple[int, int]
    conditional_scale_cap: float
    scale_transform: str = "neutra_conditional_tanh"
    mixture_components: int = 4
    slope_floor: float = 1.e-6
    final_weight_scale: float = .001
    permutation_policy: str = "full_reverse"
    inverse_atol: float | None = None
    inverse_rtol: float | None = None
    inverse_max_iterations: int = 100
    mask_policy: str = "legacy_degree_masks_v1"
    naf_conditioner: str = "author_cmade"
    iaf_initializer: str = "hoffman_variance_scaling"
    iaf_variance_scale: float = .02
    dtype: str = "float64"
    affine_center: tuple[float, ...] = ()
    affine_scale: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        object.__setattr__(self, "seed", tuple(self.seed))
        if self.dtype not in ("float32", "float64"):
            raise ValueError("transport dtype must be float32 or float64")
        # Sixteen FP32 epsilons allow finite-precision log-sum-exp/bisection
        # rounding. This is an engineering hypothesis, checked on tails, not
        # an alteration to any posterior/convergence threshold.
        inverse_tolerance = 16.*2.**-23 if self.dtype == "float32" else 1.e-11
        for name in ("inverse_atol", "inverse_rtol"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, inverse_tolerance)
        for name in ("affine_center", "affine_scale"):
            values = tuple(getattr(self, name))
            if values and (len(values) != self.dimension or not all(math.isfinite(v) for v in values)):
                raise ValueError(f"invalid {name}")
            if name == "affine_scale" and any(v <= 0. for v in values):
                raise ValueError("affine_scale must be positive")
            object.__setattr__(self, name, values)
        for name in ("dimension", "stages", "mixture_components", "inverse_max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.hidden_layers or any(type(w) is not int or w < 1 for w in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive integers")
        if len(self.seed) != 2 or any(type(s) is not int or not -(2**31) <= s < 2**31 for s in self.seed):
            raise ValueError("seed must contain two int32 integers")
        if self.kind not in ("iaf", "naf_dsf"):
            raise ValueError("kind must be iaf or naf_dsf")
        if self.naf_conditioner not in ("author_cmade", "paper_made"):
            raise ValueError("unsupported naf_conditioner")
        if self.iaf_initializer not in ("hoffman_variance_scaling", "glorot_small_final"):
            raise ValueError("unsupported iaf_initializer")
        if self.kind == "naf_dsf" and self.naf_conditioner == "author_cmade" and (
                len(set(self.hidden_layers)) != 1 or self.hidden_layers[0] < self.dimension):
            raise ValueError("author cMADE requires equal hidden widths at least dimension")
        if self.mask_policy not in ("legacy_degree_masks_v1", "hoffman_block_masks_v1"):
            raise ValueError("unsupported mask_policy")
        if self.mask_policy == "hoffman_block_masks_v1" and (self.kind != "iaf" or any(w % self.dimension for w in self.hidden_layers)):
            raise ValueError("author IAF block masks require dimension-multiple widths")
        if self.activation not in ("elu", "tanh", "relu"):
            raise ValueError("unsupported activation")
        if self.permutation_policy not in ("full_reverse", "root_preserving_reverse"):
            raise ValueError("unsupported permutation_policy")
        if self.scale_transform not in ("neutra_conditional_tanh", "bounded_tanh", "dsge_bounded_tanh", "identity"):
            raise ValueError("unsupported scale_transform")
        for name in ("conditional_scale_cap", "slope_floor", "final_weight_scale", "inverse_atol", "inverse_rtol", "iaf_variance_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.slope_floor >= 1.:
            raise ValueError("slope_floor must be less than one")

    @classmethod
    def hoffman_author_iaf(cls, dimension, *, conditional_scale_cap, seed, dtype="float64"):
        """Paper §4.1.1 architecture with the author's optional pre-clamp scale.

        Cap is caller supplied. Training settings are deliberately separate.
        Block masks reproduce the author's TFP masked_dense topology. Stateless
        initialization is not bitwise reproduction of the TF1 experiment.
        """
        return cls(dimension, "iaf", (dimension, dimension), 3, "elu", tuple(seed), conditional_scale_cap,
                   mask_policy="hoffman_block_masks_v1", dtype=dtype)

    @classmethod
    def huang_dsf(cls, dimension, *, hidden_layers, stages, mixture_components, seed, dtype="float64"):
        return cls(dimension, "naf_dsf", tuple(hidden_layers), stages, "elu", tuple(seed),
                   2., mixture_components=mixture_components, dtype=dtype)

    def payload(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def manifest_payload(self):
        return {"schema": "bayesfilter.neutra.transport_config.v1", **self.payload()}


class NeuTraTransport:
    """IAF or conditional NAF, usable by RKL, weighted densities and HMC."""
    def __init__(self, config: NeuTraTransportConfig, *, trainable=True):
        self.config = config
        self.dtype = tf.as_dtype(config.dtype)
        self.parameter_dim = config.dimension
        self.stages = tuple(core.AutoregressiveStage(config, i, trainable=trainable) for i in range(config.stages))
        self._permutation = core.Permutation(config.dimension, config.permutation_policy)
        layers = []
        for i, stage in enumerate(self.stages):
            layers.append(stage)
            if i+1 < len(self.stages):
                layers.append(self._permutation)
        if config.affine_center or config.affine_scale:
            layers.append(core.AffineLayer(
                tf.constant(config.affine_center or (0.,)*config.dimension, self.dtype),
                tf.constant(config.affine_scale or (1.,)*config.dimension, self.dtype)))
        self.score_layers = self.components = tuple(layers)
        self.trainable_variables = tuple(v for s in self.stages for v in s.trainable_variables)
        self._manifest = None
        self._precision_conversion = None

    def as_dtype(self, dtype, *, trainable=False):
        """Copy parameters explicitly; conversion never resumes Adam moments.

        Higher-precision evaluation changes rounding, not the learned parameter
        values. The new map requires its own diagnostics and HMC qualification.
        """
        dtype = tf.as_dtype(dtype).name
        config = self.config if dtype == self.config.dtype else replace(
            self.config, dtype=dtype, inverse_atol=None, inverse_rtol=None)
        converted = NeuTraTransport(config, trainable=trainable)
        converted.restore_parameters(self.parameter_state())
        if dtype != self.config.dtype:
            converted._precision_conversion = {
                "source_dtype": self.config.dtype, "evaluation_dtype": dtype,
                "source_config_hash": _hash(self.config.payload()),
                "source_parameters_hash": _hash(self.parameter_state()),
                "optimizer_state_transferred": False,
                "requires_fresh_evaluation": True}
        else:
            converted._precision_conversion = self._precision_conversion
        return converted

    def _between_stage_permutation(self, values):
        return self._permutation.inverse(values)

    def forward_and_logdet(self, latent):
        values = tf.ensure_shape(tf.convert_to_tensor(latent, self.dtype), [None, self.parameter_dim])
        return core.compose_forward(self.components, values)

    def inverse_and_forward_logdet(self, physical):
        values = tf.ensure_shape(tf.convert_to_tensor(physical, self.dtype), [None, self.parameter_dim])
        for layer in reversed(self.components):
            values = layer.inverse(values)
        return values, self.forward_and_logdet(values)[1]

    def log_prob(self, physical):
        z, ld = self.inverse_and_forward_logdet(physical)
        return -.5*(tf.reduce_sum(tf.square(z), axis=-1)+self.parameter_dim*math.log(2.*math.pi))-ld

    def forward_batch(self, latent):
        return self.forward_and_logdet(latent)[0]

    def forward(self, latent):
        values = tf.convert_to_tensor(latent, self.dtype)
        return self.forward_batch(values[None, :])[0] if values.shape.rank == 1 else self.forward_batch(values)

    forward_z_to_theta = forward
    forward_z_to_theta_batch = forward_batch

    def inverse_theta_to_z_batch(self, physical):
        return self.inverse_and_forward_logdet(physical)[0]

    def inverse_theta_to_z(self, physical):
        values = tf.convert_to_tensor(physical, self.dtype)
        return self.inverse_theta_to_z_batch(values[None, :])[0]

    def log_abs_det_jacobian_batch(self, latent):
        return self.forward_and_logdet(latent)[1]

    def log_abs_det_jacobian(self, latent):
        values = tf.convert_to_tensor(latent, self.dtype)
        return self.log_abs_det_jacobian_batch(values[None, :])[0] if values.shape.rank == 1 else self.log_abs_det_jacobian_batch(values)

    def pullback_score_batch(self, latent, score):
        return core.compose_pullback(self.components, tf.convert_to_tensor(latent, self.dtype), tf.convert_to_tensor(score, self.dtype))

    def pullback_score(self, latent, score):
        values, score = tf.convert_to_tensor(latent, self.dtype), tf.convert_to_tensor(score, self.dtype)
        return self.pullback_score_batch(values[None, :], score[None, :])[0] if values.shape.rank == 1 else self.pullback_score_batch(values, score)

    def log_abs_det_jacobian_score_batch(self, latent):
        return core.compose_pullback(self.components, tf.convert_to_tensor(latent, self.dtype))

    def log_abs_det_jacobian_score(self, latent):
        values = tf.convert_to_tensor(latent, self.dtype)
        return self.log_abs_det_jacobian_score_batch(values[None, :])[0] if values.shape.rank == 1 else self.log_abs_det_jacobian_score_batch(values)

    def parameter_state(self):
        return [{"weights": [w.numpy().tolist() for w in stage.weights],
                 "biases": [b.numpy().tolist() for b in stage.biases],
                 "extras": {k: [v.numpy().tolist() for v in values]
                            for k, values in stage.extra_parameters.items()}} for stage in self.stages]

    def restore_parameters(self, state):
        if self._manifest is not None:
            raise ValueError("a bound frozen transport cannot be restored or mutated")
        if len(state) != len(self.stages):
            raise ValueError("stage count mismatch")
        pending = []
        for stage, saved in zip(self.stages, state):
            if set(saved.get("extras", {})) != set(stage.extra_parameters):
                raise ValueError("conditioner extra parameter inventory mismatch")
            for key in ("weights", "biases", *stage.extra_parameters):
                current = getattr(stage, key)
                entries = saved[key] if key in ("weights", "biases") else saved["extras"][key]
                if len(entries) != len(current):
                    raise ValueError("parameter count mismatch")
                restored = tuple(tf.convert_to_tensor(v, self.dtype) for v in entries)
                for old, new in zip(current, restored):
                    if old.shape != new.shape or not bool(tf.reduce_all(tf.math.is_finite(new)).numpy()):
                        raise ValueError("invalid parameter shape or value")
                pending.append((stage, key, current, restored))
        for stage, key, current, restored in pending:
            if stage.trainable_variables:
                for old, new in zip(current, restored):
                    old.assign(new)
            else:
                setattr(stage, key, restored)

    def frozen_payload(self, *, target_signature, training_state_hash=None):
        if not isinstance(target_signature, str) or len(target_signature) != 64 or any(c not in "0123456789abcdef" for c in target_signature):
            raise ValueError("target_signature must be SHA256 hex")
        payload = {"schema": FROZEN_SCHEMA, "authority": core.AUTHORITY,
                   "transport_id": "configured_neutra_"+self.config.kind,
                   "dimension": self.parameter_dim, "target_signature": target_signature,
                   "log_jacobian_available": True, "config": self.config.payload(),
                   "parameters": self.parameter_state(), "training_state_hash": training_state_hash}
        if self._precision_conversion is not None:
            payload["precision_conversion"] = self._precision_conversion
        return {**payload, "transport_hash": _hash(payload)}

    def manifest_payload(self):
        if self._manifest is None:
            raise ValueError("freeze and reload the map before HMC use")
        return self._manifest.manifest_payload()

    @property
    def target_signature(self):
        return None if self._manifest is None else self._manifest.target_signature


def load_configured_frozen(payload, *, expected_target_signature, binding_type):
    from bayesfilter.inference.neutra_artifacts import (
        FrozenNeuTraArtifactManifest, InvalidNeuTraArtifact, LoadedFrozenNeuTraArtifact,
    )
    try:
        body = {k: v for k, v in payload.items() if k != "transport_hash"}
        if payload["schema"] != FROZEN_SCHEMA or payload["authority"] != core.AUTHORITY:
            raise ValueError("unsupported configured transport authority")
        if _hash(body) != payload["transport_hash"]:
            raise ValueError("transport_hash mismatch")
        if payload["target_signature"] != expected_target_signature:
            raise ValueError("target_signature mismatch")
        if len(expected_target_signature) != 64 or any(c not in "0123456789abcdef" for c in expected_target_signature):
            raise ValueError("target_signature must be SHA256 hex")
        config = NeuTraTransportConfig(**payload["config"])
        if config.dimension != payload["dimension"] or payload["log_jacobian_available"] is not True:
            raise ValueError("invalid dimension or Jacobian declaration")
        transport = NeuTraTransport(config, trainable=False)
        transport.restore_parameters(payload["parameters"])
        transport._precision_conversion = payload.get("precision_conversion")
        manifest = FrozenNeuTraArtifactManifest(
            schema=FROZEN_SCHEMA, transport_id=payload["transport_id"], dimension=config.dimension,
            target_signature=expected_target_signature, transport_hash=payload["transport_hash"],
            log_jacobian_available=True, training_state_hash=payload["training_state_hash"],
            topology_hash=_hash(payload["config"]), tensor_hash=_hash(payload["parameters"]))
        transport._manifest = manifest
        binding = binding_type(transport_id=manifest.transport_id, dimension=config.dimension,
            target_signature=expected_target_signature, log_jacobian_available=True,
            transport_manifest={"transport_id": manifest.transport_id,
                "transport_hash": manifest.transport_hash, "target_signature": expected_target_signature,
                "schema": FROZEN_SCHEMA, "topology_hash": manifest.topology_hash, "tensor_hash": manifest.tensor_hash})
        signature = _hash({"schema": "bayesfilter.neutra.loaded_frozen_artifact.v1",
                           "manifest": manifest.manifest_payload(), "binding": binding.manifest_payload()})
        return LoadedFrozenNeuTraArtifact(transport, manifest, binding, signature)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidNeuTraArtifact(str(exc)) from exc


@dataclass(frozen=True)
class NeuTraOptimizerConfig:
    batch_size: int
    estimator: str
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    gradient_clip_norm: float | None
    jit_compile: bool = True
    target_dtype: str = "float64"

    def __post_init__(self):
        if self.target_dtype not in ("float32", "float64"):
            raise ValueError("target_dtype must be float32 or float64")
        if type(self.batch_size) is not int or self.batch_size < 2:
            raise ValueError("batch_size must exceed one")
        if self.estimator not in ("standard", "path"):
            raise ValueError("estimator must be standard or path")
        for name in ("learning_rate", "epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("invalid Adam moments")
        if self.gradient_clip_norm is not None and (not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0):
            raise ValueError("gradient_clip_norm must be positive or None")


class NeuTraTransportTrainer:
    """Stable batched Adam with standard or layerwise path RKL and full resume.

    Target returns batch values, exact first scores, and per-row validity. It
    must itself evaluate a native batch. Target Hessians are never requested.
    No q20 training defaults are installed by this engineering API.
    """
    def __init__(self, transport, target_value_score, config, *, target_signature):
        if not isinstance(transport, NeuTraTransport) or not transport.trainable_variables:
            raise ValueError("a trainable configured NeuTra transport is required")
        if len(target_signature) != 64 or any(c not in "0123456789abcdef" for c in target_signature):
            raise ValueError("target_signature must be SHA256 hex")
        self.transport, self.target_value_score = transport, target_value_score
        self.config, self.target_signature = config, target_signature
        self.variables = transport.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate,
            beta_1=config.beta1, beta_2=config.beta2, epsilon=config.epsilon)
        self.optimizer.build(self.variables)
        signature = (tf.TensorSpec([config.batch_size, transport.parameter_dim], transport.dtype),)
        self.evaluate = tf.function(self._evaluate, input_signature=signature, jit_compile=config.jit_compile, autograph=False)
        self.train_step = tf.function(self._train_step, input_signature=signature, jit_compile=config.jit_compile, autograph=False)

    def _evaluate(self, latent):
        return core.reverse_kl_evaluate(self.transport, self.target_value_score, latent,
            estimator=self.config.estimator, variables=self.variables,
            target_dtype=tf.as_dtype(self.config.target_dtype))

    def _train_step(self, latent):
        result = self._evaluate(latent)
        gradients = result["gradients"]
        norm = tf.linalg.global_norm(gradients)
        if self.config.gradient_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm, use_norm=norm)
        all_variables = (*self.variables, *self.optimizer.variables)
        saved = tuple(tf.identity(v) for v in all_variables)

        def attempt():
            self.optimizer.apply_gradients(zip(gradients, self.variables))
            output, ld = self.transport.forward_and_logdet(latent)
            okay = tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(v))
                for v in (*self.variables, output, ld)] + [tf.reduce_all(tf.math.is_finite(v))
                    for v in self.optimizer.variables if tf.as_dtype(v.dtype).is_floating]))
            def rollback():
                for variable, value in zip(all_variables, saved):
                    variable.assign(value)
                return tf.constant(False)
            return tf.cond(okay, lambda: tf.constant(True), rollback)

        valid = tf.cond(result["valid"] & tf.math.is_finite(norm), attempt, lambda: tf.constant(False))
        return {**result, "valid": valid, "gradient_norm": norm,
                "clipped_gradient_norm": tf.linalg.global_norm(gradients),
                "iteration": tf.identity(self.optimizer.iterations)}

    def checkpoint(self):
        payload = {"schema": CHECKPOINT_SCHEMA, "target_signature": self.target_signature,
            "transport_config": self.transport.config.payload(), "optimizer_config": asdict(self.config),
            "parameters": self.transport.parameter_state(),
            "optimizer": [{"dtype": tf.as_dtype(v.dtype).name, "shape": list(v.shape),
                           "value": v.numpy().tolist()} for v in self.optimizer.variables]}
        return {**payload, "checkpoint_hash": _hash(payload)}

    def restore(self, checkpoint):
        body = {k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}
        if checkpoint.get("checkpoint_hash") != _hash(body):
            raise ValueError("checkpoint_hash mismatch")
        expected = (CHECKPOINT_SCHEMA, self.target_signature, self.transport.config.payload(), asdict(self.config))
        actual = (checkpoint["schema"], checkpoint["target_signature"],
            NeuTraTransportConfig(**checkpoint["transport_config"]).payload(),
            asdict(NeuTraOptimizerConfig(**checkpoint["optimizer_config"])))
        if expected != actual:
            raise ValueError("checkpoint configuration or target mismatch")
        if len(checkpoint["optimizer"]) != len(self.optimizer.variables):
            raise ValueError("optimizer state length mismatch")
        pending = []
        for var, saved in zip(self.optimizer.variables, checkpoint["optimizer"]):
            dtype = tf.as_dtype(var.dtype)
            value = tf.convert_to_tensor(saved["value"], dtype)
            if saved["dtype"] != dtype.name or saved["shape"] != list(var.shape) or value.shape != var.shape:
                raise ValueError("optimizer shape/dtype mismatch")
            if dtype.is_floating and not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
                raise ValueError("nonfinite optimizer state")
            pending.append((var, value))
        self.transport.restore_parameters(checkpoint["parameters"])
        for var, value in pending:
            var.assign(value)

    def post_training_diagnostic(self, bridge, *, beta=1., seed):
        from bayesfilter.inference.neutra_post_training import PostTrainingProbe
        # The shared downstream target/score diagnostic has an FP64 contract.
        # Evaluate an explicit frozen copy; never silently downcast its inputs.
        evaluation_map = self.transport.as_dtype("float64")
        probe = PostTrainingProbe(evaluation_map, bridge, beta, jit_compile=self.config.jit_compile)
        report = probe(seed)
        report["training_dtype"] = self.transport.config.dtype
        report["evaluation_dtype"] = "float64"
        return report

    def finalize(self, bridge, *, beta=1., diagnostic_seed):
        """Training handoff always includes the standard 1,000-point probe.

        Residual magnitudes are explanatory; numerical/target invalidity vetoes
        handoff. A finite report is not training-quality or HMC admission.
        """
        report = self.post_training_diagnostic(bridge, beta=beta, seed=diagnostic_seed)
        if not report["complete"] or not report["finite"] or report["valid_rows"] != 1000:
            raise ValueError("post-training verification is invalid")
        checkpoint = self.checkpoint()
        return {"checkpoint": checkpoint, "post_training": report,
                "frozen_transport": self.transport.as_dtype("float64").frozen_payload(target_signature=self.target_signature,
                    training_state_hash=checkpoint["checkpoint_hash"]),
                "training_quality_established": False, "posterior_qualified": False}
