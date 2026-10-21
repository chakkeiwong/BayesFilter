"""Opt-in stochastic density training helpers for P75.

This module is deliberately not exported from ``bayesfilter.highdim``.  It is
an experimental fixed-variant surface, not a source-faithful Zhao--Cui route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import (
    HighDimStatus,
    MeasureConvention,
    assert_density_matches_mass,
    assert_tf_float64,
    freeze_mapping,
)
from bayesfilter.highdim.squared_tt import SquaredTTDensity, TensorProductReferenceDensity
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.validation import ComplexityBudget


P75_ROUTE_CLASSIFICATION = "extension_or_invention"
P75_SCHEMA_VERSION = "p75_stochastic_density_training.v1"
P75_SMOKE_STATUS = "P75_SMOKE_COMPLETED_NOT_PILOT_EVIDENCE"
P75_SCHEMA_STATUS = "P75_SCHEMA_READY_PHASE4_NOT_EXECUTED"
P75_NONCLAIMS = (
    "not source-faithful Zhao-Cui",
    "not lower-gate repair evidence",
    "not validation evidence",
    "not HMC readiness evidence",
    "not rank or degree promotion evidence",
    "smoke loss is not pilot evidence",
)
_AUDIT_ROLES = frozenset({"audit", "audit_line"})
P76_CORRECTED_HELDOUT_METRIC_SCHEMA_VERSION = "p76_corrected_heldout_metric.v1"
P76_CORRECTED_HELDOUT_METRIC_STATUS = "P76_CORRECTED_HELDOUT_METRIC_OK"
P76_CORRECTED_HELDOUT_METRIC_CLASSIFICATION = "extension_or_invention"
P76_CORRECTED_HELDOUT_METRIC_NONCLAIMS = P75_NONCLAIMS + (
    "corrected heldout metric is explanatory only",
    "not training evidence",
    "not stopping or selection evidence",
    "not fit-quality evidence",
)
_P76_METRIC_ROLES = frozenset({"heldout_metric", "audit_metric"})
_P76_ALLOWED_METRIC_PROVENANCE = frozenset(
    {
        "reviewed_target_bridge",
        "unit_test_reviewed_target_bridge",
        "manual_metric_fixture",
    }
)
_P76_FORBIDDEN_PROVENANCE_MARKERS = (
    "train",
    "fit",
    "prefit",
    "source_prefit",
    "source-guided-prefit",
    "selection",
    "stopping",
    "tuning",
    "unreviewed_target_bridge",
)


@dataclass(frozen=True)
class P75TrainableTTConfig:
    """Configuration for an opt-in trainable squared-TT density."""

    product_basis: ProductBasis
    ranks: tuple[int, ...]
    tau: tf.Tensor = tf.constant(1e-8, dtype=tf.float64)
    normalizer_floor: tf.Tensor = tf.constant(1e-14, dtype=tf.float64)
    denominator_floor: tf.Tensor = tf.constant(1e-300, dtype=tf.float64)
    l1_weight: tf.Tensor = tf.constant(0.0, dtype=tf.float64)
    l2_weight: tf.Tensor = tf.constant(0.0, dtype=tf.float64)
    logz_anchor_weight: tf.Tensor = tf.constant(0.0, dtype=tf.float64)
    logz_reference: tf.Tensor | None = None
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 10.0
    seed: int = 7501
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.product_basis, ProductBasis):
            raise TypeError("product_basis must be a ProductBasis")
        ranks = tuple(int(rank) for rank in self.ranks)
        if len(ranks) != self.product_basis.dimension + 1:
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        if ranks[0] != 1 or ranks[-1] != 1 or any(rank <= 0 for rank in ranks):
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        object.__setattr__(self, "ranks", ranks)
        for name in (
            "tau",
            "normalizer_floor",
            "denominator_floor",
            "l1_weight",
            "l2_weight",
            "logz_anchor_weight",
        ):
            tensor = tf.convert_to_tensor(getattr(self, name), dtype=tf.float64)
            if tensor.shape.rank != 0:
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
            if not bool(tf.math.is_finite(tensor).numpy()):
                raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
            if bool((tensor < 0.0).numpy()):
                raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
            object.__setattr__(self, name, tensor)
        if bool((self.tau <= 0.0).numpy()):
            raise ValueError("tau must be positive for P75 log rho_theta")
        if bool((self.denominator_floor <= 0.0).numpy()):
            raise ValueError("denominator_floor must be positive")
        if bool((self.normalizer_floor <= 0.0).numpy()):
            raise ValueError("normalizer_floor must be positive")
        if self.logz_reference is not None:
            logz_reference = tf.convert_to_tensor(self.logz_reference, dtype=tf.float64)
            if logz_reference.shape.rank != 0:
                raise ValueError(f"logz_reference: {HighDimStatus.INVALID_SHAPE.value}")
            if not bool(tf.math.is_finite(logz_reference).numpy()):
                raise ValueError(f"logz_reference: {HighDimStatus.NONFINITE_VALUE.value}")
            object.__setattr__(self, "logz_reference", logz_reference)
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "gradient_clip_norm", float(self.gradient_clip_norm))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class P75ObjectiveBatch:
    """One stochastic objective batch for weighted empirical cross-entropy."""

    points: tf.Tensor
    target_values: tf.Tensor
    weights: tf.Tensor
    point_records: tuple[Mapping[str, object], ...] = ()
    forbidden_audit_records: tuple[Mapping[str, object], ...] = ()
    provenance_label: str = "unrecorded_test_only"

    def __post_init__(self) -> None:
        points = tf.convert_to_tensor(self.points, dtype=tf.float64)
        targets = tf.convert_to_tensor(self.target_values, dtype=tf.float64)
        weights = tf.convert_to_tensor(self.weights, dtype=tf.float64)
        if points.shape.rank != 2:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if points.shape[0] is None or points.shape[1] is None:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if targets.shape != (int(points.shape[0]),):
            raise ValueError(f"target_values: {HighDimStatus.INVALID_SHAPE.value}")
        if weights.shape != targets.shape:
            raise ValueError(f"weights: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("points", points)
        assert_tf_float64("target_values", targets)
        assert_tf_float64("weights", weights)
        if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
            raise ValueError(f"points: {HighDimStatus.NONFINITE_VALUE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(targets)).numpy()):
            raise ValueError(f"target_values: {HighDimStatus.NONFINITE_VALUE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()):
            raise ValueError(f"weights: {HighDimStatus.NONFINITE_VALUE.value}")
        if bool(tf.reduce_any(weights < 0.0).numpy()):
            raise ValueError("weights must be nonnegative")
        if bool((tf.reduce_sum(weights) <= 0.0).numpy()):
            raise ValueError("weights must have positive mass")
        records = tuple(MappingProxyType(dict(record)) for record in self.point_records)
        forbidden = tuple(MappingProxyType(dict(record)) for record in self.forbidden_audit_records)
        if records and len(records) != int(points.shape[0]):
            raise ValueError("point_records length must match point count")
        _assert_no_audit_leakage(records, forbidden)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "target_values", targets)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "point_records", records)
        object.__setattr__(self, "forbidden_audit_records", forbidden)
        object.__setattr__(self, "provenance_label", str(self.provenance_label))


@dataclass(frozen=True)
class P76CorrectedHeldoutMetricBatch:
    """Dedicated non-training batch for the corrected P76 heldout metric."""

    points: tf.Tensor
    target_sqrt_values: tf.Tensor
    integration_weights: tf.Tensor
    role: str
    provenance_label: str
    point_records: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        role = _validate_p76_metric_role(self.role, "role")
        provenance_label = _validate_p76_metric_provenance(
            self.provenance_label,
            "provenance_label",
        )
        points = tf.convert_to_tensor(self.points, dtype=tf.float64)
        targets = tf.convert_to_tensor(self.target_sqrt_values, dtype=tf.float64)
        weights = tf.convert_to_tensor(self.integration_weights, dtype=tf.float64)
        if points.shape.rank != 2:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if points.shape[0] is None or points.shape[1] is None:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if targets.shape != (int(points.shape[0]),):
            raise ValueError(f"target_sqrt_values: {HighDimStatus.INVALID_SHAPE.value}")
        if weights.shape != targets.shape:
            raise ValueError(f"integration_weights: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("points", points)
        assert_tf_float64("target_sqrt_values", targets)
        assert_tf_float64("integration_weights", weights)
        if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
            raise ValueError(f"points: {HighDimStatus.NONFINITE_VALUE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(targets)).numpy()):
            raise ValueError(f"target_sqrt_values: {HighDimStatus.NONFINITE_VALUE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()):
            raise ValueError(f"integration_weights: {HighDimStatus.NONFINITE_VALUE.value}")
        if bool(tf.reduce_any(weights < 0.0).numpy()):
            raise ValueError("integration_weights must be nonnegative")
        if bool((tf.reduce_sum(weights) <= 0.0).numpy()):
            raise ValueError("integration_weights must have positive mass")
        target_mass = tf.reduce_sum(weights * tf.square(targets))
        if (
            not bool(tf.math.is_finite(target_mass).numpy())
            or bool((target_mass <= 0.0).numpy())
        ):
            raise ValueError("corrected heldout target mass must be positive and finite")
        records = tuple(MappingProxyType(dict(record)) for record in self.point_records)
        if records and len(records) != int(points.shape[0]):
            raise ValueError("point_records length must match point count")
        _assert_p76_metric_records(records)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "target_sqrt_values", targets)
        object.__setattr__(self, "integration_weights", weights)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "provenance_label", provenance_label)
        object.__setattr__(self, "point_records", records)


@dataclass(frozen=True)
class P75ObjectiveTerms:
    """Objective terms from one differentiable P75 batch evaluation."""

    total_loss: tf.Tensor
    weighted_empirical_cross_entropy: tf.Tensor
    log_normalizer: tf.Tensor
    regularization: tf.Tensor
    normalizer: tf.Tensor
    alpha_min: tf.Tensor
    alpha_max: tf.Tensor
    alpha_sum: tf.Tensor
    rho_min: tf.Tensor
    rho_max: tf.Tensor
    gradient_norm: tf.Tensor | None = None
    status: str = "ok"
    nonclaims: tuple[str, ...] = P75_NONCLAIMS

    def with_gradient_norm(self, gradient_norm: tf.Tensor) -> "P75ObjectiveTerms":
        return replace(
            self,
            gradient_norm=tf.convert_to_tensor(gradient_norm, dtype=tf.float64),
        )


@dataclass(frozen=True)
class P75PrefitTerms:
    """Supervised square-root prefit terms for a guide-only warm start."""

    total_loss: tf.Tensor
    normalized_weighted_square_error: tf.Tensor
    target_square_scale: tf.Tensor
    regularization: tf.Tensor
    prediction_min: tf.Tensor
    prediction_max: tf.Tensor
    target_min: tf.Tensor
    target_max: tf.Tensor
    gradient_norm: tf.Tensor | None = None
    status: str = "ok"
    nonclaims: tuple[str, ...] = P75_NONCLAIMS

    def with_gradient_norm(self, gradient_norm: tf.Tensor) -> "P75PrefitTerms":
        return replace(
            self,
            gradient_norm=tf.convert_to_tensor(gradient_norm, dtype=tf.float64),
        )


@dataclass(frozen=True)
class P76CorrectedHeldoutMetricTerms:
    """Corrected target-only heldout density metric terms for P76."""

    heldout_cross_entropy: tf.Tensor
    negative_weighted_log_rho: tf.Tensor
    log_normalizer: tf.Tensor
    normalizer: tf.Tensor
    target_mass: tf.Tensor
    integration_weight_mass: tf.Tensor
    alpha_min: tf.Tensor
    alpha_max: tf.Tensor
    alpha_sum: tf.Tensor
    alpha_effective_sample_size: tf.Tensor
    rho_min: tf.Tensor
    rho_max: tf.Tensor
    target_sqrt_min: tf.Tensor
    target_sqrt_max: tf.Tensor
    raw_sqrt_residual_rms: tf.Tensor
    optimal_scale_sqrt_residual_rms: tf.Tensor
    centered_log_shape_rms: tf.Tensor
    role: str
    provenance_label: str
    status: str = P76_CORRECTED_HELDOUT_METRIC_STATUS
    classification: str = P76_CORRECTED_HELDOUT_METRIC_CLASSIFICATION
    explanatory_only: bool = True
    not_training_or_selection: bool = True
    nonclaims: tuple[str, ...] = P76_CORRECTED_HELDOUT_METRIC_NONCLAIMS


class TrainableFunctionalTT:
    """Trainable fixed-rank TT square-root used by the P75 pilot."""

    def __init__(
        self,
        config: P75TrainableTTConfig,
        initial_cores: Sequence[TTCore | tf.Tensor] | None = None,
    ) -> None:
        if not isinstance(config, P75TrainableTTConfig):
            raise TypeError("config must be a P75TrainableTTConfig")
        self.config = config
        tensors = (
            self._random_initial_core_tensors()
            if initial_cores is None
            else tuple(_core_tensor(core) for core in initial_cores)
        )
        self._validate_core_tensors(tensors)
        self.cores = tuple(
            tf.Variable(tensor, dtype=tf.float64, trainable=True, name=f"p75_tt_core_{axis}")
            for axis, tensor in enumerate(tensors)
        )
        self.defensive_density = TensorProductReferenceDensity(
            config.product_basis,
            config.product_basis.convention,
        )

    @property
    def variables(self) -> tuple[tf.Variable, ...]:
        return self.cores

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        values = _validate_points(points, self.config.product_basis.dimension)
        vector = tf.ones([tf.shape(values)[0], 1], dtype=tf.float64)
        for axis, core in enumerate(self.cores):
            basis_values = self.config.product_basis.evaluate_axis(axis, values[:, axis])
            matrices = tf.einsum("nl,alb->nab", basis_values, core)
            vector = tf.einsum("na,nab->nb", vector, matrices)
        return tf.reshape(vector, [tf.shape(values)[0]])

    def sqrt_square_normalizer(self) -> tf.Tensor:
        vector = tf.ones([1], dtype=tf.float64)
        active_measure = self.config.product_basis.convention.mass_measure
        for axis, core in enumerate(self.cores):
            left_rank = int(core.shape[0])
            right_rank = int(core.shape[2])
            mass = self.config.product_basis.bases[axis].mass_matrix(active_measure)
            paired = tf.einsum("alb,AmB,lm->aAbB", core, core, mass)
            matrix = tf.reshape(
                paired,
                [left_rank * left_rank, right_rank * right_rank],
            )
            vector = tf.einsum("a,ab->b", vector, matrix)
        return tf.reshape(vector, [])

    def normalizer(self) -> tf.Tensor:
        defensive_mass = self.defensive_density.normalizer(
            self.config.product_basis.convention.mass_measure
        )
        return self.sqrt_square_normalizer() + self.config.tau * defensive_mass

    def rho_theta(self, points: tf.Tensor) -> tf.Tensor:
        values = _validate_points(points, self.config.product_basis.dimension)
        h_values = self.evaluate(values)
        q0 = tf.exp(self.defensive_density.log_density(values))
        return tf.square(h_values) + self.config.tau * q0

    def log_density(self, points: tf.Tensor) -> tf.Tensor:
        rho = self.rho_theta(points)
        z = self.normalizer()
        _assert_positive_tensor(rho, "rho_theta")
        _assert_normalizer(z, self.config.normalizer_floor)
        return tf.math.log(rho) - tf.math.log(z)

    def weighted_empirical_cross_entropy_weights(
        self,
        batch: P75ObjectiveBatch,
    ) -> tf.Tensor:
        _validate_batch_dimension(batch, self.config.product_basis.dimension)
        q0 = tf.exp(self.defensive_density.log_density(batch.points))
        raw = batch.weights * (tf.square(batch.target_values) + self.config.tau * q0)
        _assert_positive_tensor(tf.reduce_sum(raw), "weighted_empirical_cross_entropy_mass")
        return raw / tf.reduce_sum(raw)

    def corrected_heldout_metric_weights(
        self,
        batch: P76CorrectedHeldoutMetricBatch,
    ) -> tf.Tensor:
        _validate_p76_metric_batch_dimension(batch, self.config.product_basis.dimension)
        raw = batch.integration_weights * tf.square(batch.target_sqrt_values)
        target_mass = tf.reduce_sum(raw)
        _assert_positive_tensor(target_mass, "corrected_heldout_target_mass")
        return raw / target_mass

    def corrected_heldout_density_metric(
        self,
        batch: P76CorrectedHeldoutMetricBatch,
    ) -> P76CorrectedHeldoutMetricTerms:
        _validate_p76_metric_batch_dimension(batch, self.config.product_basis.dimension)
        alpha = self.corrected_heldout_metric_weights(batch)
        rho = self.rho_theta(batch.points)
        normalizer = self.normalizer()
        _assert_positive_tensor(rho, "rho_theta")
        _assert_normalizer(normalizer, self.config.normalizer_floor)
        negative_weighted_log_rho = -tf.reduce_sum(alpha * tf.math.log(rho))
        log_normalizer = tf.math.log(normalizer)
        heldout_cross_entropy = negative_weighted_log_rho + log_normalizer
        predictions = self.evaluate(batch.points)
        integration_weight_mass = tf.reduce_sum(batch.integration_weights)
        target_mass = tf.reduce_sum(
            batch.integration_weights * tf.square(batch.target_sqrt_values)
        )
        raw_sqrt_residual_rms = _weighted_rms(
            predictions - batch.target_sqrt_values,
            batch.integration_weights,
            self.config.denominator_floor,
        )
        optimal_scale = _optimal_weighted_scale(
            predictions,
            batch.target_sqrt_values,
            batch.integration_weights,
            self.config.denominator_floor,
        )
        optimal_scale_sqrt_residual_rms = _weighted_rms(
            optimal_scale * predictions - batch.target_sqrt_values,
            batch.integration_weights,
            self.config.denominator_floor,
        )
        centered_log_shape_rms = _centered_log_shape_rms(
            rho,
            tf.square(batch.target_sqrt_values),
            alpha,
            self.config.denominator_floor,
        )
        alpha_effective_sample_size = 1.0 / tf.reduce_sum(tf.square(alpha))
        _assert_all_finite(
            (
                heldout_cross_entropy,
                negative_weighted_log_rho,
                log_normalizer,
                normalizer,
                target_mass,
                integration_weight_mass,
                tf.reduce_min(alpha),
                tf.reduce_max(alpha),
                tf.reduce_sum(alpha),
                alpha_effective_sample_size,
                tf.reduce_min(rho),
                tf.reduce_max(rho),
                tf.reduce_min(batch.target_sqrt_values),
                tf.reduce_max(batch.target_sqrt_values),
                raw_sqrt_residual_rms,
                optimal_scale_sqrt_residual_rms,
                centered_log_shape_rms,
            ),
            "corrected_heldout_metric_terms",
        )
        return P76CorrectedHeldoutMetricTerms(
            heldout_cross_entropy=tf.reshape(heldout_cross_entropy, []),
            negative_weighted_log_rho=tf.reshape(negative_weighted_log_rho, []),
            log_normalizer=tf.reshape(log_normalizer, []),
            normalizer=tf.reshape(normalizer, []),
            target_mass=tf.reshape(target_mass, []),
            integration_weight_mass=tf.reshape(integration_weight_mass, []),
            alpha_min=tf.reduce_min(alpha),
            alpha_max=tf.reduce_max(alpha),
            alpha_sum=tf.reduce_sum(alpha),
            alpha_effective_sample_size=tf.reshape(alpha_effective_sample_size, []),
            rho_min=tf.reduce_min(rho),
            rho_max=tf.reduce_max(rho),
            target_sqrt_min=tf.reduce_min(batch.target_sqrt_values),
            target_sqrt_max=tf.reduce_max(batch.target_sqrt_values),
            raw_sqrt_residual_rms=tf.reshape(raw_sqrt_residual_rms, []),
            optimal_scale_sqrt_residual_rms=tf.reshape(
                optimal_scale_sqrt_residual_rms,
                [],
            ),
            centered_log_shape_rms=tf.reshape(centered_log_shape_rms, []),
            role=batch.role,
            provenance_label=batch.provenance_label,
        )

    def objective(self, batch: P75ObjectiveBatch) -> P75ObjectiveTerms:
        _validate_batch_dimension(batch, self.config.product_basis.dimension)
        alpha = self.weighted_empirical_cross_entropy_weights(batch)
        rho = self.rho_theta(batch.points)
        normalizer = self.normalizer()
        _assert_positive_tensor(rho, "rho_theta")
        _assert_normalizer(normalizer, self.config.normalizer_floor)
        cross_entropy = -tf.reduce_sum(alpha * tf.math.log(rho))
        log_normalizer = tf.math.log(normalizer)
        regularization = self._regularization(log_normalizer)
        total = cross_entropy + log_normalizer + regularization
        _assert_all_finite(
            (total, cross_entropy, log_normalizer, regularization, normalizer),
            "objective_terms",
        )
        return P75ObjectiveTerms(
            total_loss=tf.reshape(total, []),
            weighted_empirical_cross_entropy=tf.reshape(cross_entropy, []),
            log_normalizer=tf.reshape(log_normalizer, []),
            regularization=tf.reshape(regularization, []),
            normalizer=tf.reshape(normalizer, []),
            alpha_min=tf.reduce_min(alpha),
            alpha_max=tf.reduce_max(alpha),
            alpha_sum=tf.reduce_sum(alpha),
            rho_min=tf.reduce_min(rho),
            rho_max=tf.reduce_max(rho),
        )

    def train_step(
        self,
        batch: P75ObjectiveBatch,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> P75ObjectiveTerms:
        with tf.GradientTape() as tape:
            terms = self.objective(batch)
        gradients = tape.gradient(terms.total_loss, self.variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("missing gradient for at least one trainable core")
        checked = tuple(tf.convert_to_tensor(gradient, dtype=tf.float64) for gradient in gradients)
        _assert_all_finite(checked, "gradients")
        clipped, gradient_norm = tf.clip_by_global_norm(
            checked,
            clip_norm=tf.constant(self.config.gradient_clip_norm, dtype=tf.float64),
        )
        _assert_all_finite(tuple(clipped) + (gradient_norm,), "clipped_gradients")
        optimizer.apply_gradients(zip(clipped, self.variables))
        _assert_all_finite(tuple(tf.convert_to_tensor(core) for core in self.variables), "parameters")
        return terms.with_gradient_norm(gradient_norm)

    def square_root_prefit_objective(
        self,
        batch: P75ObjectiveBatch,
        *,
        reference_cores: Sequence[tf.Tensor] | None = None,
        reference_l2_weight: tf.Tensor | float | None = None,
        scale_floor: tf.Tensor | float | None = None,
    ) -> P75PrefitTerms:
        """Return a guide-only loss for initializing the square-root TT.

        This objective is not a validation criterion.  It fits \(h_\theta\) to
        training-eligible source-route square-root targets before the reviewed
        density cross-entropy objective is evaluated.
        """

        _validate_batch_dimension(batch, self.config.product_basis.dimension)
        predictions = self.evaluate(batch.points)
        residual = predictions - batch.target_values
        weighted_square = tf.reduce_sum(batch.weights * tf.square(residual))
        floor = (
            self.config.denominator_floor
            if scale_floor is None
            else tf.convert_to_tensor(scale_floor, dtype=tf.float64)
        )
        _assert_positive_tensor(floor, "prefit_scale_floor")
        target_square_scale = (
            tf.reduce_sum(batch.weights * tf.square(batch.target_values)) + floor
        )
        _assert_positive_tensor(target_square_scale, "prefit_target_square_scale")
        data_loss = weighted_square / target_square_scale
        regularization = self._prefit_regularization(
            reference_cores=reference_cores,
            reference_l2_weight=reference_l2_weight,
        )
        total = data_loss + regularization
        _assert_all_finite(
            (total, data_loss, target_square_scale, regularization),
            "prefit_terms",
        )
        return P75PrefitTerms(
            total_loss=tf.reshape(total, []),
            normalized_weighted_square_error=tf.reshape(data_loss, []),
            target_square_scale=tf.reshape(target_square_scale, []),
            regularization=tf.reshape(regularization, []),
            prediction_min=tf.reduce_min(predictions),
            prediction_max=tf.reduce_max(predictions),
            target_min=tf.reduce_min(batch.target_values),
            target_max=tf.reduce_max(batch.target_values),
        )

    def square_root_prefit_step(
        self,
        batch: P75ObjectiveBatch,
        optimizer: tf.keras.optimizers.Optimizer,
        *,
        reference_cores: Sequence[tf.Tensor] | None = None,
        reference_l2_weight: tf.Tensor | float | None = None,
        scale_floor: tf.Tensor | float | None = None,
    ) -> P75PrefitTerms:
        with tf.GradientTape() as tape:
            terms = self.square_root_prefit_objective(
                batch,
                reference_cores=reference_cores,
                reference_l2_weight=reference_l2_weight,
                scale_floor=scale_floor,
            )
        gradients = tape.gradient(terms.total_loss, self.variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("missing gradient for at least one trainable core")
        checked = tuple(tf.convert_to_tensor(gradient, dtype=tf.float64) for gradient in gradients)
        _assert_all_finite(checked, "prefit_gradients")
        clipped, gradient_norm = tf.clip_by_global_norm(
            checked,
            clip_norm=tf.constant(self.config.gradient_clip_norm, dtype=tf.float64),
        )
        _assert_all_finite(tuple(clipped) + (gradient_norm,), "prefit_clipped_gradients")
        optimizer.apply_gradients(zip(clipped, self.variables))
        _assert_all_finite(tuple(tf.convert_to_tensor(core) for core in self.variables), "parameters")
        return terms.with_gradient_norm(gradient_norm)

    def snapshot_functional_tt(self) -> FunctionalTT:
        cores = tuple(TTCore(tf.identity(core)) for core in self.cores)
        return FunctionalTT(
            cores,
            self.config.product_basis,
            self.config.product_basis.convention,
            complexity_budget=ComplexityBudget(max_elements=10_000_000, max_bytes=80_000_000),
        )

    def snapshot_density(self) -> SquaredTTDensity:
        ftt = self.snapshot_functional_tt()
        identity = SquaredTTDensity.expected_branch_identity(
            sqrt_tt=ftt,
            defensive_density=self.defensive_density,
            tau=self.config.tau,
            normalizer_floor=self.config.normalizer_floor,
            denominator_floor=self.config.denominator_floor,
            measure_convention=self.config.product_basis.convention,
        )
        return SquaredTTDensity(
            sqrt_tt=ftt,
            defensive_density=self.defensive_density,
            tau=self.config.tau,
            normalizer_floor=self.config.normalizer_floor,
            denominator_floor=self.config.denominator_floor,
            measure_convention=self.config.product_basis.convention,
            branch_identity=identity,
        )

    def _regularization(self, log_normalizer: tf.Tensor) -> tf.Tensor:
        l1 = tf.add_n([tf.reduce_sum(tf.abs(core)) for core in self.cores])
        l2 = tf.add_n([tf.reduce_sum(tf.square(core)) for core in self.cores])
        penalty = self.config.l1_weight * l1 + self.config.l2_weight * l2
        if self.config.logz_anchor_weight > 0.0:
            reference = (
                tf.stop_gradient(log_normalizer)
                if self.config.logz_reference is None
                else self.config.logz_reference
            )
            penalty = penalty + self.config.logz_anchor_weight * tf.square(
                log_normalizer - reference
            )
        return tf.reshape(penalty, [])

    def _prefit_regularization(
        self,
        *,
        reference_cores: Sequence[tf.Tensor] | None,
        reference_l2_weight: tf.Tensor | float | None,
    ) -> tf.Tensor:
        weight = (
            self.config.l2_weight
            if reference_l2_weight is None
            else tf.convert_to_tensor(reference_l2_weight, dtype=tf.float64)
        )
        if weight.shape.rank != 0:
            raise ValueError(f"reference_l2_weight: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.math.is_finite(weight).numpy()) or bool((weight < 0.0).numpy()):
            raise ValueError(f"reference_l2_weight: {HighDimStatus.NONFINITE_VALUE.value}")
        if reference_cores is None:
            penalty = weight * tf.add_n([tf.reduce_sum(tf.square(core)) for core in self.cores])
            return tf.reshape(penalty, [])
        references = tuple(tf.convert_to_tensor(core, dtype=tf.float64) for core in reference_cores)
        self._validate_core_tensors(references)
        penalty = weight * tf.add_n(
            [
                tf.reduce_sum(tf.square(core - reference))
                for core, reference in zip(self.cores, references)
            ]
        )
        return tf.reshape(penalty, [])

    def _random_initial_core_tensors(self) -> tuple[tf.Tensor, ...]:
        tensors = []
        for axis, basis_dim in enumerate(self.config.product_basis.basis_dim_tuple()):
            shape = (
                self.config.ranks[axis],
                basis_dim,
                self.config.ranks[axis + 1],
            )
            seed = tf.constant([self.config.seed, axis + 1], dtype=tf.int32)
            tensors.append(
                0.05
                * tf.random.stateless_normal(shape, seed=seed, dtype=tf.float64)
            )
        return tuple(tensors)

    def _validate_core_tensors(self, tensors: Sequence[tf.Tensor]) -> None:
        if len(tensors) != self.config.product_basis.dimension:
            raise ValueError(f"initial_cores: {HighDimStatus.INVALID_SHAPE.value}")
        for axis, (tensor, basis_dim) in enumerate(
            zip(tensors, self.config.product_basis.basis_dim_tuple())
        ):
            value = tf.convert_to_tensor(tensor, dtype=tf.float64)
            expected = (
                self.config.ranks[axis],
                basis_dim,
                self.config.ranks[axis + 1],
            )
            if value.shape != expected:
                raise ValueError(f"core {axis}: {HighDimStatus.INVALID_SHAPE.value}")
            assert_tf_float64(f"core {axis}", value)
            if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
                raise ValueError(f"core {axis}: {HighDimStatus.NONFINITE_VALUE.value}")


def make_adam_optimizer(config: P75TrainableTTConfig) -> tf.keras.optimizers.Adam:
    """Return the reviewed first optimizer for P75 smoke and pilot runs."""

    return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)


def _core_tensor(core: TTCore | tf.Tensor) -> tf.Tensor:
    return tf.convert_to_tensor(core.values if isinstance(core, TTCore) else core, dtype=tf.float64)


def _validate_points(points: tf.Tensor, dimension: int) -> tf.Tensor:
    values = tf.convert_to_tensor(points, dtype=tf.float64)
    assert_tf_float64("points", values)
    if values.shape.rank != 2 or values.shape[1] != int(dimension):
        raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
    return values


def _validate_batch_dimension(batch: P75ObjectiveBatch, dimension: int) -> None:
    if int(batch.points.shape[1]) != int(dimension):
        raise ValueError(f"batch.points: {HighDimStatus.INVALID_SHAPE.value}")


def _validate_p76_metric_batch_dimension(
    batch: P76CorrectedHeldoutMetricBatch,
    dimension: int,
) -> None:
    if int(batch.points.shape[1]) != int(dimension):
        raise ValueError(f"metric_batch.points: {HighDimStatus.INVALID_SHAPE.value}")


def _validate_p76_metric_role(value: object, name: str) -> str:
    role = str(value)
    if not role:
        raise ValueError(f"{name} must declare a metric role")
    if role not in _P76_METRIC_ROLES:
        raise ValueError(f"{name} must be a non-training metric role")
    return role


def _validate_p76_metric_provenance(value: object, name: str) -> str:
    provenance = str(value)
    if not provenance:
        raise ValueError(f"{name} must declare metric provenance")
    lowered = provenance.lower()
    if any(marker in lowered for marker in _P76_FORBIDDEN_PROVENANCE_MARKERS):
        raise ValueError(f"{name} contains forbidden training or selection provenance")
    if provenance not in _P76_ALLOWED_METRIC_PROVENANCE:
        raise ValueError(f"{name} must be an approved metric provenance")
    return provenance


def _assert_p76_metric_records(point_records: tuple[Mapping[str, object], ...]) -> None:
    for record in point_records:
        if "role" not in record:
            raise ValueError("point record must declare metric role")
        _validate_p76_metric_role(record.get("role"), "point_record.role")
        if "provenance_label" not in record:
            raise ValueError("point record must declare metric provenance")
        _validate_p76_metric_provenance(
            record.get("provenance_label"),
            "point_record.provenance_label",
        )


def _weighted_rms(
    residual: tf.Tensor,
    weights: tf.Tensor,
    floor: tf.Tensor,
) -> tf.Tensor:
    weighted_mass = tf.reduce_sum(weights)
    _assert_positive_tensor(weighted_mass, "weighted_rms_mass")
    scale = weighted_mass + tf.convert_to_tensor(floor, dtype=tf.float64)
    _assert_positive_tensor(scale, "weighted_rms_scale")
    return tf.sqrt(tf.reduce_sum(weights * tf.square(residual)) / scale)


def _optimal_weighted_scale(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    weights: tf.Tensor,
    floor: tf.Tensor,
) -> tf.Tensor:
    denominator = tf.reduce_sum(weights * tf.square(predictions)) + tf.convert_to_tensor(
        floor,
        dtype=tf.float64,
    )
    _assert_positive_tensor(denominator, "optimal_scale_denominator")
    return tf.reduce_sum(weights * predictions * targets) / denominator


def _centered_log_shape_rms(
    rho: tf.Tensor,
    target_square: tf.Tensor,
    alpha: tf.Tensor,
    floor: tf.Tensor,
) -> tf.Tensor:
    floor_tensor = tf.convert_to_tensor(floor, dtype=tf.float64)
    _assert_positive_tensor(floor_tensor, "centered_log_shape_floor")
    safe_rho = tf.maximum(rho, floor_tensor)
    safe_target = tf.maximum(target_square, floor_tensor)
    log_rho = tf.math.log(safe_rho)
    log_target = tf.math.log(safe_target)
    centered_rho = log_rho - tf.reduce_sum(alpha * log_rho)
    centered_target = log_target - tf.reduce_sum(alpha * log_target)
    return tf.sqrt(tf.reduce_sum(alpha * tf.square(centered_rho - centered_target)))


def _assert_positive_tensor(value: tf.Tensor, name: str) -> None:
    tf.debugging.assert_positive(value, message=f"{name} must be positive")
    tf.debugging.assert_all_finite(value, message=f"{name} must be finite")


def _assert_normalizer(value: tf.Tensor, floor: tf.Tensor) -> None:
    tf.debugging.assert_greater(value, floor, message="normalizer floor exceeded")
    tf.debugging.assert_all_finite(value, message="normalizer must be finite")


def _assert_all_finite(values: Sequence[tf.Tensor], name: str) -> None:
    for value in values:
        tf.debugging.assert_all_finite(value, message=f"{name} must be finite")


def _assert_no_audit_leakage(
    point_records: tuple[Mapping[str, object], ...],
    forbidden_audit_records: tuple[Mapping[str, object], ...],
) -> None:
    for record in point_records:
        if str(record.get("role", "")) in _AUDIT_ROLES:
            raise ValueError("audit role cannot appear in P75 training records")
    training_hashes = {
        str(record.get("cloud_hash"))
        for record in point_records
        if record.get("cloud_hash") not in (None, "")
    }
    audit_hashes = {
        str(record.get("cloud_hash"))
        for record in forbidden_audit_records
        if record.get("cloud_hash") not in (None, "")
    }
    if training_hashes.intersection(audit_hashes):
        raise ValueError("P75 training records overlap forbidden audit cloud hashes")


def config_payload(config: P75TrainableTTConfig) -> Mapping[str, object]:
    """Return a JSON-friendly P75 config summary."""

    assert_density_matches_mass(config.product_basis.convention)
    return {
        "schema_version": P75_SCHEMA_VERSION,
        "classification": P75_ROUTE_CLASSIFICATION,
        "dimension": config.product_basis.dimension,
        "basis_dim_tuple": config.product_basis.basis_dim_tuple(),
        "ranks": config.ranks,
        "tau": float(config.tau.numpy()),
        "normalizer_floor": float(config.normalizer_floor.numpy()),
        "denominator_floor": float(config.denominator_floor.numpy()),
        "l1_weight": float(config.l1_weight.numpy()),
        "l2_weight": float(config.l2_weight.numpy()),
        "logz_anchor_weight": float(config.logz_anchor_weight.numpy()),
        "learning_rate": config.learning_rate,
        "gradient_clip_norm": config.gradient_clip_norm,
        "seed": config.seed,
        "measure_convention": _measure_convention_payload(config.product_basis.convention),
        "metadata": dict(config.metadata),
        "nonclaims": P75_NONCLAIMS,
    }


def terms_payload(terms: P75ObjectiveTerms) -> Mapping[str, object]:
    """Return a JSON-friendly objective-term summary."""

    return {
        "status": terms.status,
        "total_loss": float(terms.total_loss.numpy()),
        "weighted_empirical_cross_entropy": float(
            terms.weighted_empirical_cross_entropy.numpy()
        ),
        "log_normalizer": float(terms.log_normalizer.numpy()),
        "regularization": float(terms.regularization.numpy()),
        "normalizer": float(terms.normalizer.numpy()),
        "alpha_min": float(terms.alpha_min.numpy()),
        "alpha_max": float(terms.alpha_max.numpy()),
        "alpha_sum": float(terms.alpha_sum.numpy()),
        "rho_min": float(terms.rho_min.numpy()),
        "rho_max": float(terms.rho_max.numpy()),
        "gradient_norm": (
            None
            if terms.gradient_norm is None
            else float(tf.convert_to_tensor(terms.gradient_norm).numpy())
        ),
        "nonclaims": terms.nonclaims,
    }


def prefit_terms_payload(terms: P75PrefitTerms) -> Mapping[str, object]:
    """Return a JSON-friendly square-root prefit-term summary."""

    return {
        "status": terms.status,
        "total_loss": float(terms.total_loss.numpy()),
        "normalized_weighted_square_error": float(
            terms.normalized_weighted_square_error.numpy()
        ),
        "target_square_scale": float(terms.target_square_scale.numpy()),
        "regularization": float(terms.regularization.numpy()),
        "prediction_min": float(terms.prediction_min.numpy()),
        "prediction_max": float(terms.prediction_max.numpy()),
        "target_min": float(terms.target_min.numpy()),
        "target_max": float(terms.target_max.numpy()),
        "gradient_norm": (
            None
            if terms.gradient_norm is None
            else float(tf.convert_to_tensor(terms.gradient_norm).numpy())
        ),
        "nonclaims": terms.nonclaims,
    }


def corrected_heldout_metric_terms_payload(
    terms: P76CorrectedHeldoutMetricTerms,
) -> Mapping[str, object]:
    """Return a JSON-friendly corrected heldout metric summary."""

    return {
        "schema_version": P76_CORRECTED_HELDOUT_METRIC_SCHEMA_VERSION,
        "status": terms.status,
        "classification": terms.classification,
        "role": terms.role,
        "provenance_label": terms.provenance_label,
        "explanatory_only": bool(terms.explanatory_only),
        "not_training_or_selection": bool(terms.not_training_or_selection),
        "heldout_cross_entropy": float(terms.heldout_cross_entropy.numpy()),
        "negative_weighted_log_rho": float(terms.negative_weighted_log_rho.numpy()),
        "log_normalizer": float(terms.log_normalizer.numpy()),
        "normalizer": float(terms.normalizer.numpy()),
        "target_mass": float(terms.target_mass.numpy()),
        "integration_weight_mass": float(terms.integration_weight_mass.numpy()),
        "alpha_min": float(terms.alpha_min.numpy()),
        "alpha_max": float(terms.alpha_max.numpy()),
        "alpha_sum": float(terms.alpha_sum.numpy()),
        "alpha_effective_sample_size": float(
            terms.alpha_effective_sample_size.numpy()
        ),
        "rho_min": float(terms.rho_min.numpy()),
        "rho_max": float(terms.rho_max.numpy()),
        "target_sqrt_min": float(terms.target_sqrt_min.numpy()),
        "target_sqrt_max": float(terms.target_sqrt_max.numpy()),
        "raw_sqrt_residual_rms": float(terms.raw_sqrt_residual_rms.numpy()),
        "optimal_scale_sqrt_residual_rms": float(
            terms.optimal_scale_sqrt_residual_rms.numpy()
        ),
        "centered_log_shape_rms": float(terms.centered_log_shape_rms.numpy()),
        "finite_flags": {
            "heldout_cross_entropy": bool(tf.math.is_finite(terms.heldout_cross_entropy).numpy()),
            "normalizer": bool(tf.math.is_finite(terms.normalizer).numpy()),
            "target_mass": bool(tf.math.is_finite(terms.target_mass).numpy()),
            "alpha": bool(
                tf.reduce_all(
                    tf.math.is_finite(
                        tf.stack((terms.alpha_min, terms.alpha_max, terms.alpha_sum))
                    )
                ).numpy()
            ),
            "rho": bool(
                tf.reduce_all(
                    tf.math.is_finite(tf.stack((terms.rho_min, terms.rho_max)))
                ).numpy()
            ),
        },
        "nonclaims": terms.nonclaims,
    }


def _measure_convention_payload(convention: MeasureConvention) -> Mapping[str, object]:
    return {
        "density_measure": convention.density_measure.value,
        "mass_measure": convention.mass_measure.value,
        "reference_weight_name": convention.reference_weight_name,
        "physical_coordinate_name": convention.physical_coordinate_name,
        "reference_coordinate_name": convention.reference_coordinate_name,
        "dtype_name": convention.dtype_name,
    }


__all__ = [
    "P75_NONCLAIMS",
    "P75_ROUTE_CLASSIFICATION",
    "P75_SCHEMA_STATUS",
    "P75_SCHEMA_VERSION",
    "P75_SMOKE_STATUS",
    "P76_CORRECTED_HELDOUT_METRIC_CLASSIFICATION",
    "P76_CORRECTED_HELDOUT_METRIC_NONCLAIMS",
    "P76_CORRECTED_HELDOUT_METRIC_SCHEMA_VERSION",
    "P76_CORRECTED_HELDOUT_METRIC_STATUS",
    "P76CorrectedHeldoutMetricBatch",
    "P76CorrectedHeldoutMetricTerms",
    "P75ObjectiveBatch",
    "P75ObjectiveTerms",
    "P75PrefitTerms",
    "P75TrainableTTConfig",
    "TrainableFunctionalTT",
    "config_payload",
    "corrected_heldout_metric_terms_payload",
    "make_adam_optimizer",
    "prefit_terms_payload",
    "terms_payload",
]
