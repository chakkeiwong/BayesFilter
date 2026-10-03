"""Transport ensembles and fresh-Gaussian reverse-KL training.

This module is the implementation boundary for the tempered NeuTra ensemble.
Each map remains an individual bijective chart.  A mixture is represented by a
categorical component index and a log-sum-exp density; maps are never averaged.
All numerical kernels operate on statically shaped TensorFlow tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
    WeightedNeuTraTrainingError,
)


ENSEMBLE_SCHEMA = "bayesfilter.tempered.transport_ensemble.v1"
TRAINABLE_TRANSPORT_CHECKPOINT_SCHEMA = (
    "bayesfilter.tempered.trainable_transport_checkpoint.v2"
)
ENSEMBLE_NONCLAIMS = (
    "categorical mixture of charts, not an averaged map",
    "reverse-KL training is not posterior sampling",
    "mixture weights are not posterior mode masses",
    "no mode-discovery, convergence, or high-dimensional scaling claim",
)


class TemperedEnsembleError(ValueError):
    """Raised when an ensemble or training contract is invalid."""


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_checkpoint_scope(scope: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and detach the execution/data scope bound by a checkpoint."""

    if not isinstance(scope, Mapping):
        raise TemperedEnsembleError("checkpoint_scope must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(
                {str(key): value for key, value in scope.items()},
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise TemperedEnsembleError(
            "checkpoint_scope must contain finite JSON values"
        ) from exc
    required = {
        "data_identity",
        "dtype",
        "backend",
        "jit_compile",
        "training_seed_derivation",
        "validation_bank_ids",
    }
    missing = sorted(required - normalized.keys())
    if missing:
        raise TemperedEnsembleError(
            "checkpoint_scope missing: " + ", ".join(missing)
        )
    for key in ("data_identity", "dtype", "backend"):
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise TemperedEnsembleError(
                f"checkpoint_scope {key} must be a nonempty string"
            )
    if not isinstance(normalized["jit_compile"], bool):
        raise TemperedEnsembleError(
            "checkpoint_scope jit_compile must be boolean"
        )
    if not isinstance(normalized["training_seed_derivation"], dict):
        raise TemperedEnsembleError(
            "checkpoint_scope training_seed_derivation must be a mapping"
        )
    validation_ids = normalized["validation_bank_ids"]
    if (
        not isinstance(validation_ids, list)
        or not validation_ids
        or any(not isinstance(item, str) or not item for item in validation_ids)
    ):
        raise TemperedEnsembleError(
            "checkpoint_scope validation_bank_ids must be nonempty strings"
        )
    return normalized


def _static_rank2(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[0] is None or tensor.shape[-1] != int(dimension):
        raise TemperedEnsembleError(
            f"{name} must have static shape [batch,{int(dimension)}]"
        )
    if int(tensor.shape[0]) <= 0:
        raise TemperedEnsembleError(f"{name} batch must be nonempty")
    return tensor


def _static_rank3(value: Any, component_count: int, batch_size: int, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    expected = (int(component_count), int(batch_size), int(dimension))
    if tensor.shape.rank != 3 or tuple(tensor.shape.as_list()) != expected:
        raise TemperedEnsembleError(f"{name} must have static shape {expected}")
    return tensor


def _finite_tensor(value: tf.Tensor, name: str) -> tf.Tensor:
    tf.debugging.assert_all_finite(value, name)
    return value


def _value_with_reviewed_score(
    physical: tf.Tensor,
    value: tf.Tensor,
    score: tf.Tensor,
) -> tf.Tensor:
    """Attach an analytic target score without differentiating its value graph.

    The q=20 filter value program contains a custom symmetric-Sylvester
    operation whose reverse-mode gradient is intentionally not registered.
    The target supplies the analytic score for precisely this boundary.  The
    value and score are evaluated once on ``stop_gradient(physical)`` and this
    custom gradient exposes the reviewed score to the transport parameters.
    """

    physical_tensor = tf.convert_to_tensor(physical, tf.float64)
    value_tensor = tf.stop_gradient(tf.convert_to_tensor(value, tf.float64))
    score_tensor = tf.stop_gradient(tf.convert_to_tensor(score, tf.float64))
    if physical_tensor.shape.rank != 2 or score_tensor.shape.rank != 2:
        raise TemperedEnsembleError(
            "reviewed-score bridge requires rank-2 physical and score tensors"
        )
    if physical_tensor.shape != score_tensor.shape:
        raise TemperedEnsembleError(
            "reviewed-score physical and score shapes must match"
        )

    @tf.custom_gradient
    def attach(x: tf.Tensor) -> tuple[tf.Tensor, Any]:
        del x

        def grad(
            upstream: Any,
            variables: Sequence[tf.Variable] | None = None,
        ) -> Any:
            upstream_tensor = tf.convert_to_tensor(upstream, tf.float64)
            if upstream_tensor.shape.rank == 0:
                scaled_score = upstream_tensor * score_tensor
            else:
                rank_delta = score_tensor.shape.rank - upstream_tensor.shape.rank
                if rank_delta < 0:
                    raise TemperedEnsembleError(
                        "target upstream gradient has too many dimensions"
                    )
                if rank_delta:
                    upstream_tensor = tf.reshape(
                        upstream_tensor,
                        tf.concat(
                            (
                                tf.shape(upstream_tensor),
                                tf.ones([rank_delta], tf.int32),
                            ),
                            axis=0,
                        ),
                    )
                scaled_score = upstream_tensor * score_tensor
            if variables is None:
                return scaled_score
            return scaled_score, tuple(tf.zeros_like(variable) for variable in variables)

        return value_tensor, grad

    return attach(physical_tensor)


@dataclass(frozen=True)
class CrossDensityTelemetry:
    """Measured transport work for one cross-density evaluation."""

    component_count: int
    batch_size: int
    cross_density_work: int
    target_work: int
    elapsed_seconds: float = 0.0

    def payload(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReverseKLStep:
    loss: tf.Tensor
    target_finite: tf.Tensor
    gradient_norm: tf.Tensor
    clipped_gradient_norm: tf.Tensor
    clipping_applied: tf.Tensor
    step: tf.Tensor
    valid: tf.Tensor
    target_call_count: tf.Tensor
    cross_density_work: tf.Tensor


@dataclass(frozen=True)
class PullbackGaussianizationDiagnostic:
    """Held-out local density/score residuals under standard-Gaussian draws."""

    reverse_kl_per_sample: tf.Tensor
    pullback_log_density_residual: tf.Tensor
    centered_log_density_residual: tf.Tensor
    centered_log_density_rms: tf.Tensor
    centered_log_density_median_abs: tf.Tensor
    centered_log_density_q90_abs: tf.Tensor
    pullback_score_residual: tf.Tensor
    pullback_score_rms_per_coordinate: tf.Tensor
    pullback_score_maximum_row_norm: tf.Tensor
    valid_row_count: tf.Tensor
    batch_size: tf.Tensor
    finite: tf.Tensor


@dataclass(frozen=True)
class InitializationPreflight:
    component_id: str
    seed: tuple[int, int]
    scale: float
    beta: float
    bridge_signature: str
    transport_state_hash: str
    valid: bool
    finite_rows: int
    batch_size: int
    repair_index: int
    optimizer_state_absent: bool
    actual_map_repaired: bool
    reason: str

    def payload(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedTransportInitialization:
    """The exact map admitted by a pre-optimizer fixed-bank screen."""

    transport: Any
    receipt: InitializationPreflight

    def payload(self) -> Mapping[str, Any]:
        return self.receipt.payload()


class AffineDiagonalTransport:
    """Small exact affine chart used by analytic fixtures and mechanics smokes."""

    def __init__(
        self,
        center: Any,
        scale: Any,
        *,
        component_id: str = "affine",
    ) -> None:
        center_tensor = tf.convert_to_tensor(center, tf.float64)
        scale_tensor = tf.convert_to_tensor(scale, tf.float64)
        if center_tensor.shape.rank != 1 or scale_tensor.shape != center_tensor.shape:
            raise TemperedEnsembleError("affine center and scale must share rank-1 shape")
        if not bool(tf.reduce_all(tf.math.is_finite(center_tensor)).numpy()):
            raise TemperedEnsembleError("affine center is nonfinite")
        if not bool(tf.reduce_all(tf.math.is_finite(scale_tensor)).numpy()) or not bool(
            tf.reduce_all(scale_tensor > 0.0).numpy()
        ):
            raise TemperedEnsembleError("affine scales must be finite and positive")
        self.center = tf.identity(center_tensor)
        self.scale = tf.identity(scale_tensor)
        self.component_id = str(component_id)
        if not self.component_id:
            raise TemperedEnsembleError("component_id must be nonempty")
        self.parameter_dim = int(center_tensor.shape[0])

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _static_rank2(latent, self.parameter_dim, "latent")
        return values * self.scale + self.center, tf.fill(
            [tf.shape(values)[0]], tf.reduce_sum(tf.math.log(self.scale))
        )

    def forward_batch(self, latent: Any) -> tf.Tensor:
        return self.forward_and_logdet(latent)[0]

    def forward(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return values * self.scale + self.center
        return self.forward_batch(values)

    def inverse_and_forward_logdet(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _static_rank2(physical, self.parameter_dim, "physical")
        return (values - self.center) / self.scale, tf.fill(
            [tf.shape(values)[0]], tf.reduce_sum(tf.math.log(self.scale))
        )

    def inverse_theta_to_z_batch(self, physical: Any) -> tf.Tensor:
        return self.inverse_and_forward_logdet(physical)[0]

    def log_abs_det_jacobian_batch(self, latent: Any) -> tf.Tensor:
        return self.forward_and_logdet(latent)[1]

    def log_abs_det_jacobian(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        result = tf.reduce_sum(tf.math.log(self.scale))
        return result if values.shape.rank == 1 else tf.fill([tf.shape(values)[0]], result)

    def pullback_score_batch(self, latent: Any, output_score: Any) -> tf.Tensor:
        values = _static_rank2(latent, self.parameter_dim, "latent")
        score = _static_rank2(output_score, self.parameter_dim, "output_score")
        return score * self.scale

    def pullback_score(self, latent: Any, output_score: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        score = tf.convert_to_tensor(output_score, tf.float64)
        if values.shape.rank == 1:
            return score * self.scale
        return self.pullback_score_batch(values, score)

    def log_abs_det_jacobian_score_batch(self, latent: Any) -> tf.Tensor:
        values = _static_rank2(latent, self.parameter_dim, "latent")
        return tf.zeros_like(values)

    def log_abs_det_jacobian_score(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        return tf.zeros_like(values)

    def log_prob(self, physical: Any) -> tf.Tensor:
        latent, logdet = self.inverse_and_forward_logdet(physical)
        d = tf.cast(self.parameter_dim, tf.float64)
        return -0.5 * (
            tf.reduce_sum(tf.square(latent), axis=-1)
            + d * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        ) - logdet

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.tempered.affine_diagonal_transport.v1",
            "component_id": self.component_id,
            "parameter_dim": self.parameter_dim,
            "center": self.center.numpy().tolist(),
            "scale": self.scale.numpy().tolist(),
        }
        return {**payload, "transport_hash": _hash_payload(payload)}

    def initialization_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.tempered.affine_initialization.v1",
            "component_id": self.component_id,
            "center": self.center.numpy().tolist(),
            "scale": self.scale.numpy().tolist(),
        }


class ReferenceAffineTransport:
    """Persist a reference-law affine repair around an invertible transport.

    For an inner map ``T``, this object represents

    ``theta = center + scale * T(z)``.

    The selected scale is therefore part of every later density, training, and
    HMC calculation.  It is not merely a narrower diagnostic batch.
    """

    def __init__(
        self,
        inner: Any,
        *,
        center: Any,
        scale: Any,
        component_id: str,
    ) -> None:
        dimension = _transport_dimension(inner)
        center_tensor = tf.convert_to_tensor(center, tf.float64)
        scale_tensor = tf.convert_to_tensor(scale, tf.float64)
        if center_tensor.shape != (dimension,):
            raise TemperedEnsembleError(
                "reference center must match the transport dimension"
            )
        if scale_tensor.shape.rank == 0:
            scale_tensor = tf.fill([dimension], scale_tensor)
        if scale_tensor.shape != (dimension,):
            raise TemperedEnsembleError(
                "reference scale must be scalar or match the transport dimension"
            )
        if not bool(tf.reduce_all(tf.math.is_finite(center_tensor)).numpy()):
            raise TemperedEnsembleError("reference center must be finite")
        if not bool(tf.reduce_all(tf.math.is_finite(scale_tensor)).numpy()) or not bool(
            tf.reduce_all(scale_tensor > 0.0).numpy()
        ):
            raise TemperedEnsembleError(
                "reference scale must be finite and strictly positive"
            )
        self.inner = inner
        self.center = tf.identity(center_tensor)
        self.scale = tf.identity(scale_tensor)
        self.component_id = str(component_id)
        if not self.component_id:
            raise TemperedEnsembleError("component_id must be nonempty")
        self.parameter_dim = dimension

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(getattr(self.inner, "trainable_variables", ()))

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _static_rank2(latent, self.parameter_dim, "latent")
        inner_values, inner_logdet = self.inner.forward_and_logdet(values)
        physical = self.center + self.scale * inner_values
        outer_logdet = tf.reduce_sum(tf.math.log(self.scale))
        return physical, inner_logdet + outer_logdet

    def forward_batch(self, latent: Any) -> tf.Tensor:
        return self.forward_and_logdet(latent)[0]

    def forward(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.forward_batch(values[tf.newaxis, :])[0]
        return self.forward_batch(values)

    def inverse_and_forward_logdet(
        self, physical: Any
    ) -> tuple[tf.Tensor, tf.Tensor]:
        values = _static_rank2(physical, self.parameter_dim, "physical")
        normalized = (values - self.center) / self.scale
        latent, inner_logdet = self.inner.inverse_and_forward_logdet(normalized)
        outer_logdet = tf.reduce_sum(tf.math.log(self.scale))
        return latent, inner_logdet + outer_logdet

    def inverse_theta_to_z_batch(self, physical: Any) -> tf.Tensor:
        return self.inverse_and_forward_logdet(physical)[0]

    def log_abs_det_jacobian_batch(self, latent: Any) -> tf.Tensor:
        return self.forward_and_logdet(latent)[1]

    def log_abs_det_jacobian(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.log_abs_det_jacobian_batch(values[tf.newaxis, :])[0]
        return self.log_abs_det_jacobian_batch(values)

    def pullback_score_batch(
        self, latent: Any, output_score: Any
    ) -> tf.Tensor:
        values = _static_rank2(latent, self.parameter_dim, "latent")
        score = _static_rank2(output_score, self.parameter_dim, "output_score")
        return self.inner.pullback_score_batch(values, score * self.scale)

    def pullback_score(self, latent: Any, output_score: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        score = tf.convert_to_tensor(output_score, tf.float64)
        if values.shape.rank == 1:
            return self.pullback_score_batch(
                values[tf.newaxis, :], score[tf.newaxis, :]
            )[0]
        return self.pullback_score_batch(values, score)

    def log_abs_det_jacobian_score_batch(self, latent: Any) -> tf.Tensor:
        return self.inner.log_abs_det_jacobian_score_batch(latent)

    def log_abs_det_jacobian_score(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.log_abs_det_jacobian_score_batch(values[tf.newaxis, :])[0]
        return self.log_abs_det_jacobian_score_batch(values)

    def log_prob(self, physical: Any) -> tf.Tensor:
        latent, logdet = self.inverse_and_forward_logdet(physical)
        dimension = tf.cast(self.parameter_dim, tf.float64)
        base = -0.5 * (
            tf.reduce_sum(tf.square(latent), axis=-1)
            + dimension * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        )
        return base - logdet

    def bind_frozen_identity(self, identity: Mapping[str, Any]) -> None:
        binder = getattr(self.inner, "bind_frozen_identity", None)
        if not callable(binder):
            raise TemperedEnsembleError(
                "inner transport does not expose bind_frozen_identity"
            )
        binder(identity)

    def initialization_payload(self) -> Mapping[str, Any]:
        inner_config = getattr(self.inner, "config", None)
        config_payload = getattr(inner_config, "manifest_payload", None)
        return {
            "schema": "bayesfilter.tempered.reference_affine_initialization.v1",
            "component_id": self.component_id,
            "parameter_dim": self.parameter_dim,
            "center": self.center.numpy().tolist(),
            "scale": self.scale.numpy().tolist(),
            "inner_module": self.inner.__class__.__module__,
            "inner_class": self.inner.__class__.__qualname__,
            "inner_config": config_payload() if callable(config_payload) else None,
        }

    def manifest_payload(self) -> Mapping[str, Any]:
        manifest = getattr(self.inner, "manifest_payload", None)
        if not callable(manifest):
            raise TemperedEnsembleError(
                "inner transport must expose a frozen manifest"
            )
        payload = {
            "schema": "bayesfilter.tempered.reference_affine_transport.v1",
            "component_id": self.component_id,
            "parameter_dim": self.parameter_dim,
            "center": self.center.numpy().tolist(),
            "scale": self.scale.numpy().tolist(),
            "inner": dict(manifest()),
        }
        return {**payload, "transport_hash": _hash_payload(payload)}


class TransportBank:
    """A fixed-size categorical bank of individual invertible transports."""

    def __init__(
        self,
        transports: Sequence[Any],
        *,
        component_ids: Sequence[str] | None = None,
        alpha_logits: Any | None = None,
    ) -> None:
        values = tuple(transports)
        if not values:
            raise TemperedEnsembleError("transport bank must contain at least one component")
        self.transports = values
        self.component_count = len(values)
        dimensions = tuple(_transport_dimension(item) for item in values)
        if len(set(dimensions)) != 1:
            raise TemperedEnsembleError("all transports must share parameter dimension")
        self.parameter_dim = dimensions[0]
        ids = tuple(
            str(item)
            for item in (component_ids if component_ids is not None else _default_ids(self.component_count))
        )
        if len(ids) != self.component_count or any(not item for item in ids):
            raise TemperedEnsembleError("component_ids must match the bank and be nonempty")
        if len(set(ids)) != len(ids):
            raise TemperedEnsembleError("component_ids must be unique")
        self.component_ids = ids
        if alpha_logits is None:
            self._alpha_logits = tf.zeros([self.component_count], tf.float64)
            self.alpha_trainable = False
        else:
            logits = (
                alpha_logits
                if isinstance(alpha_logits, tf.Variable)
                else tf.convert_to_tensor(alpha_logits, tf.float64)
            )
            if logits.shape != (self.component_count,):
                raise TemperedEnsembleError("alpha_logits must have shape [component_count]")
            if tf.as_dtype(logits.dtype) != tf.float64:
                raise TemperedEnsembleError("alpha_logits must use float64")
            self._alpha_logits = logits
            self.alpha_trainable = isinstance(logits, tf.Variable) and bool(logits.trainable)

    @property
    def alpha_logits(self) -> tf.Tensor:
        return self._alpha_logits

    @property
    def alpha(self) -> tf.Tensor:
        return tf.nn.softmax(self._alpha_logits)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        variables: list[tf.Variable] = []
        for transport in self.transports:
            variables.extend(tuple(getattr(transport, "trainable_variables", ())))
        if isinstance(self._alpha_logits, tf.Variable) and self._alpha_logits.trainable:
            variables.append(self._alpha_logits)
        return tuple(variables)

    def signature_payload(self) -> Mapping[str, Any]:
        transports = []
        for component_id, transport in zip(self.component_ids, self.transports, strict=True):
            manifest = getattr(transport, "manifest_payload", None)
            if callable(manifest):
                try:
                    payload = dict(manifest())
                except (TemperedEnsembleError, WeightedNeuTraTrainingError):
                    payload = {"parameter_dim": _transport_dimension(transport)}
            else:
                payload = {"parameter_dim": _transport_dimension(transport)}
            transports.append({"component_id": component_id, "transport": payload})
        return {
            "schema": ENSEMBLE_SCHEMA,
            "component_count": self.component_count,
            "parameter_dim": self.parameter_dim,
            "component_ids": list(self.component_ids),
            "alpha_logits": self.alpha_logits.numpy().tolist(),
            "transports": transports,
            "nonclaims": list(ENSEMBLE_NONCLAIMS),
        }

    def forward_bank(self, latent_bank: Any) -> tuple[tf.Tensor, tf.Tensor]:
        latent_tensor = tf.convert_to_tensor(latent_bank, tf.float64)
        if latent_tensor.shape.rank != 3 or latent_tensor.shape[1] is None:
            raise TemperedEnsembleError("latent_bank must have a static batch dimension")
        values = _static_rank3(
            latent_tensor,
            self.component_count,
            int(latent_tensor.shape[1]),
            self.parameter_dim,
            "latent_bank",
        )
        physical_rows = []
        logdet_rows = []
        for index, transport in enumerate(self.transports):
            forward = getattr(transport, "forward_and_logdet", None)
            if not callable(forward):
                raise TemperedEnsembleError("each transport must expose forward_and_logdet")
            physical, logdet = forward(values[index])
            physical_rows.append(_static_rank2(physical, self.parameter_dim, "physical component"))
            logdet_rows.append(tf.ensure_shape(tf.convert_to_tensor(logdet, tf.float64), [values.shape[1]]))
        return tf.stack(physical_rows, axis=0), tf.stack(logdet_rows, axis=0)

    def component_log_prob(self, physical: Any) -> tf.Tensor:
        values = _static_rank2(physical, self.parameter_dim, "physical")
        rows = []
        for transport in self.transports:
            method = getattr(transport, "log_prob", None)
            if not callable(method):
                raise TemperedEnsembleError("each transport must expose log_prob")
            rows.append(tf.ensure_shape(tf.convert_to_tensor(method(values), tf.float64), [values.shape[0]]))
        return tf.stack(rows, axis=0)

    def cross_component_log_prob(self, physical_bank: Any) -> tf.Tensor:
        values_tensor = tf.convert_to_tensor(physical_bank, tf.float64)
        if values_tensor.shape.rank != 3 or values_tensor.shape[0] != self.component_count or values_tensor.shape[2] != self.parameter_dim:
            raise TemperedEnsembleError("physical_bank must have shape [component,batch,dimension]")
        batch_size = values_tensor.shape[1]
        if batch_size is None:
            raise TemperedEnsembleError("physical_bank batch size must be static")
        flattened = tf.reshape(values_tensor, [self.component_count * int(batch_size), self.parameter_dim])
        rows = []
        for transport in self.transports:
            method = getattr(transport, "log_prob", None)
            if not callable(method):
                raise TemperedEnsembleError("each transport must expose log_prob")
            density = tf.convert_to_tensor(method(flattened), tf.float64)
            rows.append(tf.reshape(density, [self.component_count, int(batch_size)]))
        return tf.stack(rows, axis=0)

    def mixture_log_prob(self, physical: Any) -> tf.Tensor:
        component = self.component_log_prob(physical)
        return tf.reduce_logsumexp(tf.math.log(self.alpha)[:, tf.newaxis] + component, axis=0)

    def sample(self, batch_size: int, seed: Any) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        size = int(batch_size)
        if size <= 0:
            raise TemperedEnsembleError("batch_size must be positive")
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        if seed_tensor.shape != (2,):
            raise TemperedEnsembleError("seed must have shape [2]")
        indices = tf.cast(
            tf.squeeze(tf.random.stateless_categorical(tf.math.log(self.alpha)[tf.newaxis, :], size, seed_tensor), axis=0),
            tf.int32,
        )
        latent = tf.random.stateless_normal(
            [size, self.parameter_dim],
            tf.random.experimental.stateless_fold_in(seed_tensor, 1),
            dtype=tf.float64,
        )
        latent_bank = tf.broadcast_to(latent[tf.newaxis, :, :], [self.component_count, size, self.parameter_dim])
        physical_bank, _ = self.forward_bank(latent_bank)
        selected = tf.einsum(
            "bk,kbd->bd",
            tf.one_hot(indices, self.component_count, dtype=tf.float64),
            physical_bank,
        )
        return selected, indices, latent

    def state_payload(self) -> Mapping[str, Any]:
        rows = []
        for transport in self.transports:
            variables = tuple(getattr(transport, "trainable_variables", ()))
            rows.append([variable.numpy().tolist() for variable in variables])
        payload = {
            "schema": "bayesfilter.tempered.transport_bank_state.v1",
            "signature": self.signature_payload(),
            "variables": rows,
            "alpha_logits": self.alpha_logits.numpy().tolist(),
        }
        return {**payload, "state_hash": _hash_payload(payload)}

    def restore_state_payload(self, state: Mapping[str, Any]) -> None:
        """Restore an exact bank checkpoint after structural validation."""
        payload = dict(state)
        expected_hash = str(payload.pop("state_hash", ""))
        if not expected_hash or _hash_payload(payload) != expected_hash:
            raise TemperedEnsembleError("transport-bank checkpoint hash mismatch")
        if payload.get("schema") != "bayesfilter.tempered.transport_bank_state.v1":
            raise TemperedEnsembleError("transport-bank checkpoint schema mismatch")
        signature = payload.get("signature")
        if not isinstance(signature, Mapping):
            raise TemperedEnsembleError("transport-bank checkpoint lacks a signature")
        if (
            int(signature.get("component_count", -1)) != self.component_count
            or int(signature.get("parameter_dim", -1)) != self.parameter_dim
            or tuple(signature.get("component_ids", ())) != self.component_ids
        ):
            raise TemperedEnsembleError(
                "transport-bank checkpoint structure does not match the bank"
            )
        rows = payload.get("variables")
        if not isinstance(rows, Sequence) or len(rows) != self.component_count:
            raise TemperedEnsembleError(
                "transport-bank checkpoint variable rows are invalid"
            )
        assignments: list[tuple[tf.Variable, tf.Tensor]] = []
        for transport, values in zip(self.transports, rows, strict=True):
            variables = tuple(getattr(transport, "trainable_variables", ()))
            if not isinstance(values, Sequence) or len(values) != len(variables):
                raise TemperedEnsembleError(
                    "transport-bank checkpoint variable count mismatch"
                )
            for variable, value in zip(variables, values, strict=True):
                tensor = tf.convert_to_tensor(value, tf.float64)
                if tensor.shape != variable.shape:
                    raise TemperedEnsembleError(
                        "transport-bank checkpoint variable shape mismatch"
                    )
                assignments.append((variable, tensor))
        logits = tf.convert_to_tensor(payload.get("alpha_logits"), tf.float64)
        if logits.shape != (self.component_count,):
            raise TemperedEnsembleError(
                "transport-bank checkpoint alpha shape mismatch"
            )
        if not isinstance(self._alpha_logits, tf.Variable) and not bool(
            tf.reduce_all(self._alpha_logits == logits).numpy()
        ):
            raise TemperedEnsembleError(
                "cannot restore different logits into a nontrainable bank"
            )
        for variable, tensor in assignments:
            variable.assign(tensor)
        if isinstance(self._alpha_logits, tf.Variable):
            self._alpha_logits.assign(logits)
        if self.state_payload()["state_hash"] != expected_hash:
            raise TemperedEnsembleError(
                "transport-bank checkpoint did not round-trip exactly"
            )


def mixture_reverse_kl_terms(
    bank: TransportBank,
    physical_bank: Any,
    target_values_bank: Any,
    forward_logdet_bank: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return joint mixture-RKL loss and its exact component terms.

    ``physical_bank[i,b]`` is an outer draw from chart ``i``.  The target is
    evaluated once on all ``K*B`` rows; every component density is then
    evaluated on those same physical rows, yielding the required ``K^2 B``
    transport work without additional target calls.
    """
    physical = tf.convert_to_tensor(physical_bank, tf.float64)
    if physical.shape.rank != 3 or physical.shape[0] != bank.component_count or physical.shape[2] != bank.parameter_dim:
        raise TemperedEnsembleError("physical_bank shape does not match bank")
    batch_size = physical.shape[1]
    if batch_size is None:
        raise TemperedEnsembleError("physical_bank batch must be static")
    target = tf.ensure_shape(tf.convert_to_tensor(target_values_bank, tf.float64), [bank.component_count, batch_size])
    logdet = tf.ensure_shape(tf.convert_to_tensor(forward_logdet_bank, tf.float64), [bank.component_count, batch_size])
    cross = bank.cross_component_log_prob(physical)
    log_alpha = tf.math.log(bank.alpha)
    mixture = tf.reduce_logsumexp(log_alpha[:, tf.newaxis, tf.newaxis] + cross, axis=0)
    per_sample = mixture - target - logdet
    loss = tf.reduce_sum(bank.alpha * tf.reduce_mean(per_sample, axis=1))
    return loss, per_sample, mixture, cross


def pullback_gaussianization_diagnostic(
    transport: Any,
    target_bridge: Any,
    *,
    beta: float,
    latent: Any,
) -> PullbackGaussianizationDiagnostic:
    """Evaluate exact Gaussian-pullback identities on a held-out base bank.

    For an exact chart, ``log pi(T(z)) + log|det J_T(z)| - log phi(z)`` is
    constant and its score residual relative to ``-z`` is zero.  The base bank
    only probes regions reached by the chart; this is not a global coverage
    result.
    """

    dimension = _transport_dimension(transport)
    rows = _static_rank2(latent, dimension, "latent")
    beta_value = float(beta)
    if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
        raise TemperedEnsembleError("diagnostic beta must lie in [0,1]")
    forward = getattr(transport, "forward_and_logdet", None)
    pullback = getattr(transport, "pullback_score_batch", None)
    logdet_score = getattr(transport, "log_abs_det_jacobian_score_batch", None)
    if not callable(forward) or not callable(pullback) or not callable(logdet_score):
        raise TemperedEnsembleError(
            "Gaussianization diagnostic requires forward and exact score operations"
        )
    physical, logdet = forward(rows)
    target, score, status = target_bridge.value_score_status(
        physical, tf.constant(beta_value, tf.float64)
    )
    status_key = (
        "bridge_valid"
        if "bridge_valid" in status
        else "valid_pre_regularized_score"
    )
    valid_rows = tf.convert_to_tensor(status[status_key], tf.bool)
    dimension_value = tf.cast(dimension, tf.float64)
    log_phi = -0.5 * (
        tf.reduce_sum(tf.square(rows), axis=-1)
        + dimension_value * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
    )
    residual = target + logdet - log_phi
    centered = residual - tf.reduce_mean(residual)
    absolute_centered = tf.sort(tf.abs(centered))
    batch_size = int(rows.shape[0])
    median_index = (batch_size - 1) // 2
    q90_index = max(0, min(batch_size - 1, math.ceil(0.9 * batch_size) - 1))
    score_residual = (
        pullback(rows, score) + logdet_score(rows) + rows
    )
    score_row_norm = tf.linalg.norm(score_residual, axis=-1)
    values = (
        target,
        score,
        logdet,
        residual,
        centered,
        score_residual,
    )
    finite = tf.logical_and(
        tf.reduce_all(valid_rows),
        tf.reduce_all(
            tf.stack(
                tuple(tf.reduce_all(tf.math.is_finite(value)) for value in values)
            )
        ),
    )
    return PullbackGaussianizationDiagnostic(
        reverse_kl_per_sample=-target - logdet,
        pullback_log_density_residual=residual,
        centered_log_density_residual=centered,
        centered_log_density_rms=tf.sqrt(tf.reduce_mean(tf.square(centered))),
        centered_log_density_median_abs=absolute_centered[median_index],
        centered_log_density_q90_abs=absolute_centered[q90_index],
        pullback_score_residual=score_residual,
        pullback_score_rms_per_coordinate=tf.sqrt(
            tf.reduce_mean(tf.square(score_residual), axis=0)
        ),
        pullback_score_maximum_row_norm=tf.reduce_max(score_row_norm),
        valid_row_count=tf.reduce_sum(tf.cast(valid_rows, tf.int32)),
        batch_size=tf.constant(batch_size, tf.int32),
        finite=finite,
    )


def _aggregate_pullback_gaussianization_diagnostics(
    diagnostics: Sequence[PullbackGaussianizationDiagnostic],
) -> PullbackGaussianizationDiagnostic:
    """Combine per-row diagnostic chunks without changing their target calls.

    This helper is intentionally host-orchestrated: each input diagnostic was
    produced by one static, non-singleton TensorFlow batch.  Only the
    per-row tensors are concatenated here, then the same finite-bank summary
    reductions as the one-shot diagnostic are applied once.
    """

    parts = tuple(diagnostics)
    if not parts:
        raise TemperedEnsembleError("at least one diagnostic chunk is required")
    dimension = int(parts[0].pullback_score_residual.shape[-1])
    if dimension <= 0:
        raise TemperedEnsembleError("diagnostic score dimension must be positive")
    for item in parts:
        if not isinstance(item, PullbackGaussianizationDiagnostic):
            raise TemperedEnsembleError("diagnostic chunks have an invalid type")
        if item.reverse_kl_per_sample.shape.rank != 1:
            raise TemperedEnsembleError("diagnostic loss rows must be rank one")
        if item.pullback_log_density_residual.shape.rank != 1:
            raise TemperedEnsembleError("diagnostic density rows must be rank one")
        if item.pullback_score_residual.shape.rank != 2:
            raise TemperedEnsembleError("diagnostic score rows must be rank two")
        if item.pullback_score_residual.shape[-1] != dimension:
            raise TemperedEnsembleError("diagnostic chunks have different dimensions")
        expected_rows = int(item.reverse_kl_per_sample.shape[0])
        if int(item.pullback_log_density_residual.shape[0]) != expected_rows:
            raise TemperedEnsembleError("diagnostic chunk row counts disagree")
        if int(item.pullback_score_residual.shape[0]) != expected_rows:
            raise TemperedEnsembleError("diagnostic score row count disagrees")

    reverse_kl = tf.concat(
        tuple(item.reverse_kl_per_sample for item in parts), axis=0
    )
    residual = tf.concat(
        tuple(item.pullback_log_density_residual for item in parts), axis=0
    )
    score_residual = tf.concat(
        tuple(item.pullback_score_residual for item in parts), axis=0
    )
    valid_row_count = tf.reduce_sum(
        tf.stack(tuple(tf.convert_to_tensor(item.valid_row_count, tf.int32) for item in parts))
    )
    finite = tf.logical_and(
        tf.reduce_all(tf.stack(tuple(tf.convert_to_tensor(item.finite, tf.bool) for item in parts))),
        tf.reduce_all(
            tf.stack(
                tuple(
                    tf.reduce_all(tf.math.is_finite(value))
                    for value in (reverse_kl, residual, score_residual)
                )
            )
        ),
    )
    centered = residual - tf.reduce_mean(residual)
    absolute_centered = tf.sort(tf.abs(centered))
    row_count = int(reverse_kl.shape[0])
    median_index = (row_count - 1) // 2
    q90_index = max(0, min(row_count - 1, math.ceil(0.9 * row_count) - 1))
    score_row_norm = tf.linalg.norm(score_residual, axis=-1)
    return PullbackGaussianizationDiagnostic(
        reverse_kl_per_sample=reverse_kl,
        pullback_log_density_residual=residual,
        centered_log_density_residual=centered,
        centered_log_density_rms=tf.sqrt(tf.reduce_mean(tf.square(centered))),
        centered_log_density_median_abs=absolute_centered[median_index],
        centered_log_density_q90_abs=absolute_centered[q90_index],
        pullback_score_residual=score_residual,
        pullback_score_rms_per_coordinate=tf.sqrt(
            tf.reduce_mean(tf.square(score_residual), axis=0)
        ),
        pullback_score_maximum_row_norm=tf.reduce_max(score_row_norm),
        valid_row_count=tf.cast(valid_row_count, tf.int32),
        batch_size=tf.constant(row_count, tf.int32),
        finite=finite,
    )


def chunked_pullback_gaussianization_diagnostic(
    transport: Any,
    target_bridge: Any,
    *,
    beta: float,
    latent: Any,
    chunk_size: int,
) -> PullbackGaussianizationDiagnostic:
    """Evaluate a held-out bank through fixed, non-singleton target chunks.

    The latent bank is partitioned only along its leading sample axis.  Every
    target call receives a statically shaped ``[chunk_size, dimension]`` batch,
    and the total bank size must be an exact multiple of that size.  This is a
    graph-cost diagnostic for large validation banks; it does not introduce a
    scalar or row-mapped target path and is not itself a training or posterior
    route.
    """

    dimension = _transport_dimension(transport)
    rows = _static_rank2(latent, dimension, "latent")
    size = int(rows.shape[0])
    chunk = int(chunk_size)
    if chunk <= 1:
        raise TemperedEnsembleError("chunk_size must be greater than one")
    if chunk > size or size % chunk:
        raise TemperedEnsembleError(
            "chunk_size must divide the static latent-bank row count"
        )
    chunks = tuple(
        pullback_gaussianization_diagnostic(
            transport,
            target_bridge,
            beta=beta,
            latent=rows[start : start + chunk],
        )
        for start in range(0, size, chunk)
    )
    result = _aggregate_pullback_gaussianization_diagnostics(chunks)
    if int(result.batch_size.numpy()) != size:
        raise TemperedEnsembleError("chunked diagnostic lost latent rows")
    return result


def paired_reverse_kl_improvement(
    start: PullbackGaussianizationDiagnostic,
    final: PullbackGaussianizationDiagnostic,
) -> Mapping[str, tf.Tensor]:
    """Return a paired 95% Monte Carlo interval for final-minus-start loss."""

    initial = tf.convert_to_tensor(start.reverse_kl_per_sample, tf.float64)
    terminal = tf.convert_to_tensor(final.reverse_kl_per_sample, tf.float64)
    if initial.shape.rank != 1 or terminal.shape != initial.shape:
        raise TemperedEnsembleError(
            "paired reverse-KL diagnostics must share a static rank-1 bank"
        )
    count = int(initial.shape[0])
    if count <= 1:
        raise TemperedEnsembleError(
            "paired reverse-KL improvement requires at least two rows"
        )
    difference = terminal - initial
    mean = tf.reduce_mean(difference)
    centered = difference - mean
    sample_variance = tf.reduce_sum(tf.square(centered)) / tf.constant(
        float(count - 1), tf.float64
    )
    standard_error = tf.sqrt(
        sample_variance / tf.constant(float(count), tf.float64)
    )
    radius = tf.constant(1.959963984540054, tf.float64) * standard_error
    finite = tf.reduce_all(
        tf.math.is_finite(
            tf.stack((mean, standard_error, mean - radius, mean + radius))
        )
    )
    return {
        "mean_final_minus_start": mean,
        "standard_error": standard_error,
        "two_sided_95_lower": mean - radius,
        "two_sided_95_upper": mean + radius,
        "row_count": tf.constant(count, tf.int32),
        "finite": finite,
        "training_viable": tf.logical_and(finite, mean + radius < 0.0),
    }


class IndependentTemperedReverseKLTrainer:
    """One component trained with fresh IID standard-Gaussian batches."""

    def __init__(
        self,
        config: WeightedNeuTraConfig,
        target_bridge: Any,
        *,
        beta: float,
        component_id: str,
        batch_size: int,
        prepared_initialization: PreparedTransportInitialization,
    ) -> None:
        if not callable(getattr(target_bridge, "value_score_status", None)):
            raise TemperedEnsembleError("target_bridge must expose value_score_status")
        if int(batch_size) <= 1:
            raise TemperedEnsembleError("reverse-KL training batch_size must exceed one")
        beta_value = float(beta)
        if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
            raise TemperedEnsembleError("beta must lie in [0,1]")
        if int(config.dimension) != int(getattr(target_bridge, "parameter_dim", config.dimension)):
            raise TemperedEnsembleError("transport and bridge dimensions differ")
        self.config = config
        self.target_bridge = target_bridge
        self.beta = beta_value
        self.component_id = str(component_id)
        self.batch_size = int(batch_size)
        if not isinstance(prepared_initialization, PreparedTransportInitialization):
            raise TemperedEnsembleError(
                "prepared_initialization must come from the fixed-bank preflight"
            )
        self.transport = prepared_initialization.transport
        _require_admitted_preflight(
            prepared_initialization.receipt,
            transport=self.transport,
            target_bridge=target_bridge,
            beta=beta_value,
            component_id=self.component_id,
        )
        if _transport_dimension(self.transport) != int(config.dimension):
            raise TemperedEnsembleError(
                "prepared transport and training config dimensions differ"
            )
        self.initialization_preflight = prepared_initialization.receipt
        self.variables = self.transport.trainable_variables
        if not self.variables:
            raise TemperedEnsembleError("transport has no trainable variables")
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name=f"{self.component_id}_rkl_step")
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(config.learning_rate),
            beta_1=float(config.beta1),
            beta_2=float(config.beta2),
            epsilon=float(config.epsilon),
        )
        self.optimizer.build(self.variables)
        self._compiled_train_step = tf.function(
            self._train_step_impl,
            input_signature=(tf.TensorSpec([2], tf.int32),),
            jit_compile=bool(config.jit_compile),
            reduce_retracing=False,
        )

    def train_step(self, seed: Any) -> ReverseKLStep:
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        if seed_tensor.shape != (2,):
            raise TemperedEnsembleError("seed must have shape [2]")
        result = self._compiled_train_step(seed_tensor)
        if not bool(result[6].numpy()):
            raise WeightedNeuTraTrainingError(
                f"reverse-KL update rejected for component {self.component_id}"
            )
        return ReverseKLStep(*result)

    def fresh_latent_batch(self, seed: Any) -> tf.Tensor:
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        return tf.random.stateless_normal(
            [self.batch_size, int(self.config.dimension)],
            seed_tensor,
            dtype=tf.float64,
        )

    def _train_step_impl(self, seed: tf.Tensor) -> tuple[tf.Tensor, ...]:
        latent = self.fresh_latent_batch(seed)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            physical, logdet = self.transport.forward_and_logdet(latent)
            target, score, status = self.target_bridge.value_score_status(
                physical, tf.constant(self.beta, tf.float64)
            )
            target = _value_with_reviewed_score(physical, target, score)
            loss = tf.reduce_mean(-target - logdet)
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(value is None for value in gradients):
            raise TemperedEnsembleError("reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(value, tf.float64) for value in gradients)
        norm = tf.linalg.global_norm(gradients)
        clipped, _ = tf.clip_by_global_norm(
            gradients, tf.constant(float(self.config.gradient_clip_norm), tf.float64), use_norm=norm
        )
        clipped_norm = tf.linalg.global_norm(clipped)
        status_valid = tf.reduce_all(
            tf.convert_to_tensor(status["bridge_valid"], tf.bool)
        )
        finite = tf.reduce_all(
            tf.stack(
                (
                    status_valid,
                    tf.reduce_all(tf.math.is_finite(loss)),
                    tf.reduce_all(tf.math.is_finite(target)),
                    tf.reduce_all(tf.math.is_finite(logdet)),
                    tf.reduce_all(tf.math.is_finite(norm)),
                    tf.reduce_all(tf.math.is_finite(clipped_norm)),
                    *(tf.reduce_all(tf.math.is_finite(value)) for value in clipped),
                )
            )
        )

        def update() -> tf.Tensor:
            self.optimizer.apply_gradients(zip(clipped, self.variables))
            return tf.cast(self.optimizer.iterations, tf.int64)

        next_step = tf.cond(finite, update, lambda: tf.identity(self.step))
        self.step.assign(next_step)
        return (
            loss,
            tf.reduce_all(tf.math.is_finite(target)),
            norm,
            clipped_norm,
            norm > tf.constant(float(self.config.gradient_clip_norm), tf.float64),
            tf.identity(self.step),
            finite,
            tf.constant(1, tf.int64),
            tf.constant(0, tf.int64),
        )


class JointTemperedMixtureReverseKLTrainer:
    """Optional exact mixture-RKL refinement with measured quadratic work."""

    def __init__(
        self,
        bank: TransportBank,
        target_bridge: Any,
        *,
        beta: float,
        batch_size: int,
        preflight_receipts: Sequence[InitializationPreflight],
        learning_rate: float = 1.0e-3,
        gradient_clip_norm: float = 10.0,
        jit_compile: bool = True,
        train_alpha: bool = True,
    ) -> None:
        if not callable(getattr(target_bridge, "value_score_status", None)):
            raise TemperedEnsembleError("target_bridge must expose value_score_status")
        if int(batch_size) <= 1:
            raise TemperedEnsembleError("joint reverse-KL batch_size must exceed one")
        beta_value = float(beta)
        if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
            raise TemperedEnsembleError("beta must lie in [0,1]")
        learning_rate = float(learning_rate)
        gradient_clip_norm = float(gradient_clip_norm)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise TemperedEnsembleError("learning_rate must be positive")
        if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
            raise TemperedEnsembleError("gradient_clip_norm must be positive")
        self.bank = bank
        self.target_bridge = target_bridge
        self.beta = beta_value
        self.batch_size = int(batch_size)
        self.gradient_clip_norm = gradient_clip_norm
        self.train_alpha = bool(train_alpha)
        receipts = tuple(preflight_receipts)
        if len(receipts) != bank.component_count:
            raise TemperedEnsembleError(
                "joint training requires one current preflight per component"
            )
        for component_id, transport, receipt in zip(
            bank.component_ids, bank.transports, receipts, strict=True
        ):
            _require_admitted_preflight(
                receipt,
                transport=transport,
                target_bridge=target_bridge,
                beta=beta_value,
                component_id=component_id,
            )
        self.initialization_preflights = receipts
        if self.train_alpha and not isinstance(bank.alpha_logits, tf.Variable):
            bank._alpha_logits = tf.Variable(
                bank.alpha_logits, trainable=True, dtype=tf.float64, name="tempered_mixture_alpha_logits"
            )
        self.variables = bank.trainable_variables
        if not self.variables:
            raise TemperedEnsembleError(
                "joint reverse-KL trainer requires at least one trainable transport variable"
            )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer.build(self.variables)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name="joint_rkl_step")
        self._compiled_train_step = tf.function(
            self._train_step_impl,
            input_signature=(tf.TensorSpec([2], tf.int32),),
            jit_compile=bool(jit_compile),
            reduce_retracing=False,
        )

    def train_step(self, seed: Any) -> ReverseKLStep:
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        if seed_tensor.shape != (2,):
            raise TemperedEnsembleError("seed must have shape [2]")
        result = self._compiled_train_step(seed_tensor)
        if not bool(result[6].numpy()):
            raise WeightedNeuTraTrainingError("joint mixture reverse-KL update rejected")
        return ReverseKLStep(*result)

    def _train_step_impl(self, seed: tf.Tensor) -> tuple[tf.Tensor, ...]:
        rows = []
        for component in range(self.bank.component_count):
            component_seed = tf.random.experimental.stateless_fold_in(seed, component)
            rows.append(
                tf.random.stateless_normal(
                    [self.batch_size, self.bank.parameter_dim], component_seed, dtype=tf.float64
                )
            )
        latent_bank = tf.stack(rows, axis=0)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            physical_bank, logdet_bank = self.bank.forward_bank(latent_bank)
            flattened = tf.reshape(
                physical_bank,
                [
                    self.bank.component_count * self.batch_size,
                    self.bank.parameter_dim,
                ],
            )
            target, _score, status = self.target_bridge.value_score_status(
                flattened, tf.constant(self.beta, tf.float64)
            )
            target = _value_with_reviewed_score(flattened, target, _score)
            target_bank = tf.reshape(target, [self.bank.component_count, self.batch_size])
            loss, _per_sample, _mixture, _cross = mixture_reverse_kl_terms(
                self.bank, physical_bank, target_bank, logdet_bank
            )
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(value is None for value in gradients):
            raise TemperedEnsembleError("joint reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(value, tf.float64) for value in gradients)
        norm = tf.linalg.global_norm(gradients)
        clipped, _ = tf.clip_by_global_norm(
            gradients, tf.constant(self.gradient_clip_norm, tf.float64), use_norm=norm
        )
        clipped_norm = tf.linalg.global_norm(clipped)
        valid = tf.reduce_all(tf.convert_to_tensor(status["bridge_valid"], tf.bool))
        finite = tf.reduce_all(
            tf.stack(
                (
                    valid,
                    tf.reduce_all(tf.math.is_finite(loss)),
                    tf.reduce_all(tf.math.is_finite(target)),
                    tf.reduce_all(tf.math.is_finite(logdet_bank)),
                    tf.reduce_all(tf.math.is_finite(norm)),
                    tf.reduce_all(tf.math.is_finite(clipped_norm)),
                    *(tf.reduce_all(tf.math.is_finite(value)) for value in clipped),
                )
            )
        )

        def update() -> tf.Tensor:
            self.optimizer.apply_gradients(zip(clipped, self.variables))
            return tf.cast(self.optimizer.iterations, tf.int64)

        next_step = tf.cond(finite, update, lambda: tf.identity(self.step))
        self.step.assign(next_step)
        return (
            loss,
            tf.reduce_all(tf.math.is_finite(target)),
            norm,
            clipped_norm,
            norm > tf.constant(self.gradient_clip_norm, tf.float64),
            tf.identity(self.step),
            finite,
            # All K component rows are evaluated in one flattened target call.
            # The quadratic work is transport-density work only.
            tf.constant(1, tf.int64),
            tf.constant(self.bank.component_count * self.bank.component_count * self.batch_size, tf.int64),
        )


def transport_preflight_state_hash(transport: Any) -> str:
    """Hash the exact pre-optimizer map state at the artifact boundary."""
    initializer = getattr(transport, "initialization_payload", None)
    config = getattr(transport, "config", None)
    config_payload = getattr(config, "manifest_payload", None)
    variables = []
    for index, variable in enumerate(
        tuple(getattr(transport, "trainable_variables", ()))
    ):
        tensor = tf.convert_to_tensor(variable, tf.float64)
        variables.append(
            {
                "index": index,
                "shape": tensor.shape.as_list(),
                "value": tensor.numpy().tolist(),
            }
        )
    payload = {
        "schema": "bayesfilter.tempered.transport_preflight_state.v1",
        "module": transport.__class__.__module__,
        "class": transport.__class__.__qualname__,
        "parameter_dim": _transport_dimension(transport),
        "initialization": initializer() if callable(initializer) else None,
        "config": config_payload() if callable(config_payload) else None,
        "trainable_variables": variables,
    }
    return _hash_payload(payload)


def _trainable_transport_structure(transport: Any) -> Mapping[str, Any]:
    if isinstance(transport, ReferenceAffineTransport):
        if not isinstance(transport.inner, WeightedDenseIAFTransport):
            raise TemperedEnsembleError(
                "checkpoint supports only a weighted dense IAF inside the "
                "reference-affine wrapper"
            )
        return {
            "kind": "reference_affine_weighted_dense_iaf",
            "component_id": transport.component_id,
            "center": transport.center.numpy().tolist(),
            "scale": transport.scale.numpy().tolist(),
            "inner_config": transport.inner.config.manifest_payload(),
        }
    if isinstance(transport, WeightedDenseIAFTransport):
        return {
            "kind": "weighted_dense_iaf",
            "inner_config": transport.config.manifest_payload(),
        }
    raise TemperedEnsembleError(
        "checkpoint supports WeightedDenseIAFTransport with an optional "
        "ReferenceAffineTransport wrapper"
    )


def _weighted_config_from_manifest(payload: Mapping[str, Any]) -> WeightedNeuTraConfig:
    values = dict(payload)
    schema = values.pop("schema", None)
    if schema != "bayesfilter.neutra.weighted_forward_kl_config.v1":
        raise TemperedEnsembleError("checkpoint contains an unsupported transport config")
    for name in (
        "hidden_layers",
        "stage_s_max",
        "stage_scale_linear_skip",
        "stage_unbounded_scale_linear",
        "initialization_seed",
    ):
        if name in values:
            values[name] = tuple(values[name])
    try:
        return WeightedNeuTraConfig(**values)
    except (TypeError, ValueError) as exc:
        raise TemperedEnsembleError(
            "checkpoint transport config is invalid"
        ) from exc


def _restore_trainable_transport_structure(payload: Mapping[str, Any]) -> Any:
    values = dict(payload)
    config_payload = values.get("inner_config")
    if not isinstance(config_payload, Mapping):
        raise TemperedEnsembleError("checkpoint is missing the transport config")
    inner = WeightedDenseIAFTransport(
        _weighted_config_from_manifest(config_payload)
    )
    kind = values.get("kind")
    if kind == "weighted_dense_iaf":
        return inner
    if kind == "reference_affine_weighted_dense_iaf":
        return ReferenceAffineTransport(
            inner,
            center=values.get("center"),
            scale=values.get("scale"),
            component_id=str(values.get("component_id", "")),
        )
    raise TemperedEnsembleError("checkpoint contains an unsupported transport kind")


def capture_trainable_transport_checkpoint(
    transport: Any,
    *,
    component_id: str,
    beta: float,
    bridge_signature: str,
    target_signature: str,
    parent_checkpoint_hash: str | None,
    update_count: int,
    checkpoint_scope: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Capture one immutable temperature-chart state at an artifact boundary."""

    component = str(component_id)
    bridge_id = str(bridge_signature)
    target_id = str(target_signature)
    beta_value = float(beta)
    if not component or not bridge_id or not target_id:
        raise TemperedEnsembleError(
            "checkpoint component, bridge, and target identities are required"
        )
    if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
        raise TemperedEnsembleError("checkpoint beta must lie in [0,1]")
    updates = int(update_count)
    if updates < 0:
        raise TemperedEnsembleError("checkpoint update_count must be nonnegative")
    parent = None if parent_checkpoint_hash is None else str(parent_checkpoint_hash)
    if parent is not None and not parent:
        raise TemperedEnsembleError("parent checkpoint hash must be nonempty")
    scope = _normalize_checkpoint_scope(checkpoint_scope)

    variables = []
    for index, variable in enumerate(
        tuple(getattr(transport, "trainable_variables", ()))
    ):
        tensor = tf.convert_to_tensor(variable)
        variables.append(
            {
                "index": index,
                "dtype": tensor.dtype.name,
                "shape": tensor.shape.as_list(),
                "value": tensor.numpy().tolist(),
            }
        )
    if not variables:
        raise TemperedEnsembleError(
            "trainable transport checkpoint requires trainable variables"
        )
    payload = {
        "schema": TRAINABLE_TRANSPORT_CHECKPOINT_SCHEMA,
        "component_id": component,
        "beta": beta_value,
        "bridge_signature": bridge_id,
        "target_signature": target_id,
        "parent_checkpoint_hash": parent,
        "update_count": updates,
        "checkpoint_scope": scope,
        "structure": _trainable_transport_structure(transport),
        "variables": variables,
        "transport_state_hash": transport_preflight_state_hash(transport),
    }
    return {**payload, "checkpoint_hash": _hash_payload(payload)}


def restore_trainable_transport_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> Any:
    """Restore a fresh chart and verify its complete tensor/identity hash."""

    payload = dict(checkpoint)
    checkpoint_hash = str(payload.pop("checkpoint_hash", ""))
    if not checkpoint_hash or _hash_payload(payload) != checkpoint_hash:
        raise TemperedEnsembleError("trainable transport checkpoint hash mismatch")
    if payload.get("schema") != TRAINABLE_TRANSPORT_CHECKPOINT_SCHEMA:
        raise TemperedEnsembleError("unsupported trainable transport checkpoint schema")
    for key, expected in dict(expected_context or {}).items():
        if payload.get(str(key)) != expected:
            raise TemperedEnsembleError(
                f"trainable transport checkpoint context mismatch: {key}"
            )
    structure = payload.get("structure")
    if not isinstance(structure, Mapping):
        raise TemperedEnsembleError("checkpoint transport structure is missing")
    transport = _restore_trainable_transport_structure(structure)
    stored_variables = payload.get("variables")
    current_variables = tuple(getattr(transport, "trainable_variables", ()))
    if not isinstance(stored_variables, Sequence) or len(stored_variables) != len(
        current_variables
    ):
        raise TemperedEnsembleError("checkpoint variable count mismatch")
    for index, (record, variable) in enumerate(
        zip(stored_variables, current_variables, strict=True)
    ):
        if not isinstance(record, Mapping) or int(record.get("index", -1)) != index:
            raise TemperedEnsembleError("checkpoint variable order mismatch")
        if record.get("dtype") != variable.dtype.name:
            raise TemperedEnsembleError("checkpoint variable dtype mismatch")
        if tuple(record.get("shape", ())) != tuple(variable.shape.as_list()):
            raise TemperedEnsembleError("checkpoint variable shape mismatch")
        value = tf.convert_to_tensor(record.get("value"), dtype=variable.dtype)
        tf.debugging.assert_all_finite(value, "checkpoint variable")
        variable.assign(value)
    if transport_preflight_state_hash(transport) != payload.get(
        "transport_state_hash"
    ):
        raise TemperedEnsembleError("restored transport state hash mismatch")
    return transport


def _bridge_signature(target_bridge: Any) -> str:
    signature = getattr(target_bridge, "signature", None)
    if callable(signature):
        signature = signature()
    if signature is None:
        signature_fn = getattr(target_bridge, "adapter_signature", None)
        signature = signature_fn() if callable(signature_fn) else signature_fn
    if not signature:
        raise TemperedEnsembleError("target bridge must expose a stable signature")
    return str(signature)


def prepare_transport_initialization(
    transport: Any,
    target_bridge: Any,
    *,
    component_id: str,
    seed: tuple[int, int],
    batch_size: int,
    repair_scales: Sequence[float] = (1.0, 0.5, 0.25),
    beta: float = 0.0,
    reference_center: Any | None = None,
    reference_scale: Any | None = None,
) -> PreparedTransportInitialization:
    """Return the exact map admitted by a fixed pre-optimizer Gaussian bank.

    The same stateless bank is used for every declared scale.  No rows are
    replaced until a lucky valid batch appears.  A selected scale is persisted
    in a ``ReferenceAffineTransport`` and is therefore the map later trained;
    this prevents a diagnostic-only narrowed batch from masquerading as an
    initialization repair.
    """
    if int(batch_size) <= 1:
        raise TemperedEnsembleError("preflight batch_size must exceed one")
    seed_tensor = tf.convert_to_tensor(seed, tf.int32)
    if seed_tensor.shape != (2,):
        raise TemperedEnsembleError("preflight seed must have shape [2]")
    beta_value = float(beta)
    if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
        raise TemperedEnsembleError("preflight beta must lie in [0,1]")
    scales = tuple(float(value) for value in repair_scales)
    if not scales or any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise TemperedEnsembleError("repair scales must be finite and positive")
    dimension = _transport_dimension(transport)
    if (reference_center is None) != (reference_scale is None):
        raise TemperedEnsembleError(
            "reference_center and reference_scale must be supplied together"
        )
    if reference_center is None:
        center = tf.zeros([dimension], tf.float64)
        base_scale = tf.ones([dimension], tf.float64)
        has_reference_affine = False
    else:
        center = tf.convert_to_tensor(reference_center, tf.float64)
        base_scale = tf.convert_to_tensor(reference_scale, tf.float64)
        if base_scale.shape.rank == 0:
            base_scale = tf.fill([dimension], base_scale)
        if center.shape != (dimension,) or base_scale.shape != (dimension,):
            raise TemperedEnsembleError(
                "reference center and scale must match the transport dimension"
            )
        has_reference_affine = True
    bridge_signature = _bridge_signature(target_bridge)
    latent = tf.random.stateless_normal(
        [int(batch_size), dimension], seed_tensor, dtype=tf.float64
    )
    selected_transport = transport
    count = 0
    for repair_index, scale in enumerate(scales):
        if has_reference_affine or scale != 1.0:
            candidate = ReferenceAffineTransport(
                transport,
                center=center,
                scale=base_scale * tf.constant(scale, tf.float64),
                component_id=str(component_id),
            )
        else:
            candidate = transport
        selected_transport = candidate
        try:
            physical = candidate.forward_batch(latent)
            value, _score, status = target_bridge.value_score_status(
                physical, tf.constant(beta_value, tf.float64)
            )
            status_key = (
                "bridge_valid"
                if "bridge_valid" in status
                else "valid_pre_regularized_score"
            )
            valid_rows = tf.convert_to_tensor(status[status_key], tf.bool)
            finite_rows = tf.logical_and(valid_rows, tf.math.is_finite(value))
            count = int(tf.reduce_sum(tf.cast(finite_rows, tf.int32)).numpy())
            if count == int(batch_size):
                receipt = InitializationPreflight(
                    component_id=str(component_id),
                    seed=tuple(int(item) for item in seed),
                    scale=scale,
                    beta=beta_value,
                    bridge_signature=bridge_signature,
                    transport_state_hash=transport_preflight_state_hash(candidate),
                    valid=True,
                    finite_rows=count,
                    batch_size=int(batch_size),
                    repair_index=repair_index,
                    optimizer_state_absent=True,
                    actual_map_repaired=repair_index > 0,
                    reason="fixed_preoptimizer_bank_passed",
                )
                return PreparedTransportInitialization(candidate, receipt)
        except (tf.errors.InvalidArgumentError, ValueError, TypeError):
            count = 0
    receipt = InitializationPreflight(
        component_id=str(component_id),
        seed=tuple(int(item) for item in seed),
        scale=scales[-1],
        beta=beta_value,
        bridge_signature=bridge_signature,
        transport_state_hash=transport_preflight_state_hash(selected_transport),
        valid=False,
        finite_rows=count,
        batch_size=int(batch_size),
        repair_index=len(scales) - 1,
        optimizer_state_absent=True,
        actual_map_repaired=len(scales) > 1,
        reason="finite_preoptimizer_repair_ladder_exhausted",
    )
    return PreparedTransportInitialization(selected_transport, receipt)


def preflight_transport_initialization(
    transport: Any,
    target_bridge: Any,
    *,
    component_id: str,
    seed: tuple[int, int],
    batch_size: int,
    repair_scales: Sequence[float] = (1.0, 0.5, 0.25),
    beta: float = 0.0,
) -> InitializationPreflight:
    """Compatibility receipt for a screen that leaves the map unchanged.

    Call ``prepare_transport_initialization`` when a repair scale may be
    selected.  This compatibility boundary refuses to discard a repaired map.
    """
    prepared = prepare_transport_initialization(
        transport,
        target_bridge,
        component_id=component_id,
        seed=seed,
        batch_size=batch_size,
        repair_scales=repair_scales,
        beta=beta,
    )
    if prepared.transport is not transport:
        raise TemperedEnsembleError(
            "preflight selected a persistent map repair; use "
            "prepare_transport_initialization and train the returned transport"
        )
    return prepared.receipt


def _require_admitted_preflight(
    receipt: InitializationPreflight,
    *,
    transport: Any,
    target_bridge: Any,
    beta: float,
    component_id: str,
) -> None:
    if not isinstance(receipt, InitializationPreflight) or receipt.valid is not True:
        raise TemperedEnsembleError("a passed initialization preflight is required")
    if receipt.optimizer_state_absent is not True:
        raise TemperedEnsembleError("preflight must precede optimizer state")
    if receipt.component_id != str(component_id):
        raise TemperedEnsembleError("preflight component identity mismatch")
    if receipt.bridge_signature != _bridge_signature(target_bridge):
        raise TemperedEnsembleError("preflight bridge identity mismatch")
    if receipt.beta != float(beta):
        raise TemperedEnsembleError("preflight beta mismatch")
    if receipt.transport_state_hash != transport_preflight_state_hash(transport):
        raise TemperedEnsembleError("transport changed after initialization preflight")


def _transport_dimension(transport: Any) -> int:
    value = getattr(transport, "parameter_dim", None)
    if value is None:
        config = getattr(transport, "config", None)
        value = getattr(config, "dimension", None)
    if value is None or int(value) <= 0:
        raise TemperedEnsembleError("transport must expose a positive parameter_dim")
    for name in ("forward_and_logdet", "log_prob"):
        if not callable(getattr(transport, name, None)):
            raise TemperedEnsembleError(f"transport must expose {name}")
    return int(value)


def _default_ids(count: int) -> tuple[str, ...]:
    return tuple(f"chart-{index}" for index in range(int(count)))


__all__ = [
    "AffineDiagonalTransport",
    "CrossDensityTelemetry",
    "ENSEMBLE_NONCLAIMS",
    "ENSEMBLE_SCHEMA",
    "TRAINABLE_TRANSPORT_CHECKPOINT_SCHEMA",
    "IndependentTemperedReverseKLTrainer",
    "InitializationPreflight",
    "JointTemperedMixtureReverseKLTrainer",
    "PreparedTransportInitialization",
    "PullbackGaussianizationDiagnostic",
    "ReferenceAffineTransport",
    "ReverseKLStep",
    "TemperedEnsembleError",
    "TransportBank",
    "capture_trainable_transport_checkpoint",
    "chunked_pullback_gaussianization_diagnostic",
    "mixture_reverse_kl_terms",
    "paired_reverse_kl_improvement",
    "prepare_transport_initialization",
    "preflight_transport_initialization",
    "restore_trainable_transport_checkpoint",
    "pullback_gaussianization_diagnostic",
    "transport_preflight_state_hash",
]
