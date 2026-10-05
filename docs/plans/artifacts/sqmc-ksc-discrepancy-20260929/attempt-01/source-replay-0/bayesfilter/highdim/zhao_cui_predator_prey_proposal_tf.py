"""Offline predator-prey squared-TT proposal fitting for the fixed variant.

This module is deliberately an offline compiler. It fits deterministic
FP64 squared-TT densities on disjoint calibration/validation designs, freezes
the selected objects, and hands them to the source-order branch compiler. The
fit, coordinate-map choices, and recursive APF assembly are classified as
`extension_or_invention`; only the squared-TT and paired-core operations are
source-grounded Zhao-Cui primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
import string
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.filtering import HighDimCoordinateMap, legendre_gauss_nodes_weights
from bayesfilter.highdim.squared_tt import SquaredTTDensity
from bayesfilter.highdim.stochastic_density_training import (
    P75ObjectiveBatch,
    P75TrainableTTConfig,
    TrainableFunctionalTT,
    make_adam_optimizer,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.zhao_cui_predator_prey_fixed_variant_tf import (
    EVENT_ORDER,
    TARGET_ID,
    TARGET_OBSERVATION_SHA256,
    TARGET_SEED,
    TARGET_STATE_SHA256,
    SourceOrderTTSIRTCompilation,
    compile_predator_prey_source_order_ttsirt_proposal_branch,
)
from bayesfilter.highdim.models import PredatorPreySSM


FIT_DTYPE = tf.float64
ROUTE_CLASSIFICATION = "extension_or_invention"
PROPOSAL_FITTER_ID = "zhao_cui_predator_prey_scope_tuned_ttsirt_fitter_v1"
DEFAULT_TAU = 1.0e-6
DEFAULT_RIDGE = 1.0e-8
DEFAULT_L2 = 1.0e-8
DEFAULT_LEARNING_RATE = 3.0e-4
DEFAULT_CDF_GRID = 65
DEFAULT_CDF_BISECTION = 22
PROPOSAL_MAX_RESIDUAL_RMS = 0.75
PROPOSAL_MAX_RESIDUAL_ABS = 40.0


@dataclass(frozen=True)
class PredatorPreyProposalSpec:
    """One scope-bound proposal-control candidate."""

    degree: int = 4
    rank: int = 8
    coordinate_map_family: str = "shifted_algebraic"
    coordinate_scale: float = 8.0
    defensive_tau: float = DEFAULT_TAU
    l1_weight: float = 0.0
    ridge: float = DEFAULT_RIDGE
    prefit_steps: int = 0
    train_steps: int = 32
    cdf_grid_size: int = DEFAULT_CDF_GRID
    cdf_bisection_steps: int = DEFAULT_CDF_BISECTION

    def __post_init__(self) -> None:
        if int(self.degree) < 0 or int(self.rank) < 1:
            raise ValueError("degree and rank must be nonnegative/positive")
        if str(self.coordinate_map_family) not in {
            "shifted_algebraic",
            "gaussian_quantile",
        }:
            raise ValueError("unsupported coordinate_map_family")
        for name in ("coordinate_scale", "defensive_tau", "ridge"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.l1_weight)) or float(self.l1_weight) < 0.0:
            raise ValueError("l1_weight must be finite and nonnegative")
        if int(self.prefit_steps) < 0 or int(self.train_steps) < 0:
            raise ValueError("prefit_steps and train_steps must be nonnegative")
        if int(self.cdf_grid_size) < 3 or int(self.cdf_bisection_steps) < 1:
            raise ValueError("CDF settings are too small")
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(
            self, "coordinate_map_family", str(self.coordinate_map_family)
        )
        object.__setattr__(self, "coordinate_scale", float(self.coordinate_scale))
        object.__setattr__(self, "defensive_tau", float(self.defensive_tau))
        object.__setattr__(self, "l1_weight", float(self.l1_weight))
        object.__setattr__(self, "ridge", float(self.ridge))
        object.__setattr__(self, "prefit_steps", int(self.prefit_steps))
        object.__setattr__(self, "train_steps", int(self.train_steps))
        object.__setattr__(self, "cdf_grid_size", int(self.cdf_grid_size))
        object.__setattr__(self, "cdf_bisection_steps", int(self.cdf_bisection_steps))

    def payload(self) -> Mapping[str, object]:
        return {
            "degree": self.degree,
            "rank": self.rank,
            "coordinate_map_family": self.coordinate_map_family,
            "coordinate_scale": self.coordinate_scale,
            "defensive_tau": self.defensive_tau,
            "l1_weight": self.l1_weight,
            "ridge": self.ridge,
            "prefit_steps": self.prefit_steps,
            "train_steps": self.train_steps,
            "cdf_grid_size": self.cdf_grid_size,
            "cdf_bisection_steps": self.cdf_bisection_steps,
        }

    def structural_payload(self) -> Mapping[str, object]:
        """Controls held fixed while comparing the L1 arms."""

        return {
            key: value
            for key, value in self.payload().items()
            if key != "l1_weight"
        }


@dataclass(frozen=True)
class PredatorPreyProposalCandidate:
    """A fitted per-time proposal candidate with held-out diagnostics.

    ``observation_hash`` is the hash of the observation path used to fit the
    frozen, observation-specific proposal.  ``validation_observation_hash``
    is retained for schema compatibility and records the path used for the
    candidate diagnostics.  Claim-bearing candidates require both hashes to
    be the sealed target; tuning candidates normally use the same path for
    fitting and diagnostics because the author route refits its SIRT at each
    observation path.
    """

    spec: PredatorPreyProposalSpec
    reference_theta: tf.Tensor
    previous_maps: tuple[HighDimCoordinateMap, ...]
    current_maps: tuple[HighDimCoordinateMap, ...]
    transports: tuple[FixedTTSIRTTransport, ...]
    calibration_diagnostics: tuple[Mapping[str, object], ...]
    validation_diagnostics: tuple[Mapping[str, object], ...]
    observation_hash: str
    validation_observation_hash: str
    data_role: str
    fit_manifest: Mapping[str, object]
    scope_id: str

    def __post_init__(self) -> None:
        theta = tf.convert_to_tensor(self.reference_theta, dtype=FIT_DTYPE)
        if theta.shape != (6,):
            raise ValueError("reference_theta must have shape [6]")
        if not bool(tf.reduce_all(tf.math.is_finite(theta)).numpy()):
            raise ValueError("reference_theta must be finite")
        if len(self.transports) != 21:
            raise ValueError("predator-prey T20 candidate requires one initial and 20 transition transports")
        if len(self.previous_maps) != 20 or len(self.current_maps) != 20:
            raise ValueError("predator-prey T20 candidate requires 20 map pairs")
        if len(self.calibration_diagnostics) != 21 or len(self.validation_diagnostics) != 21:
            raise ValueError("initial plus per-transition diagnostics are required")
        if len(str(self.scope_id)) != 64:
            raise ValueError("scope_id must be a SHA-256 digest")
        for name in ("observation_hash", "validation_observation_hash"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if str(self.data_role) not in {
            "calibration",
            "validation",
            "audit",
            "sealed_claim_preparation",
            "debug_smoke",
        }:
            raise ValueError("unsupported proposal data role")
        if self.data_role == "sealed_claim_preparation":
            if self.observation_hash != TARGET_OBSERVATION_SHA256:
                raise ValueError("sealed claim fit observations must match the sealed target")
            if self.validation_observation_hash != TARGET_OBSERVATION_SHA256:
                raise ValueError("sealed claim validation observations must match the sealed target")
        else:
            if self.observation_hash == TARGET_OBSERVATION_SHA256:
                raise ValueError("sealed observations cannot fit tuning candidates")
            if self.validation_observation_hash == TARGET_OBSERVATION_SHA256:
                raise ValueError("sealed observations cannot select tuning controls")
        expected_scope_id = _proposal_candidate_scope_id(
            spec=self.spec,
            reference_theta=theta,
            previous_maps=self.previous_maps,
            current_maps=self.current_maps,
            transports=self.transports,
            calibration_diagnostics=self.calibration_diagnostics,
            validation_diagnostics=self.validation_diagnostics,
            observation_hash=str(self.observation_hash),
            validation_observation_hash=str(self.validation_observation_hash),
            data_role=str(self.data_role),
            fit_manifest=self.fit_manifest,
        )
        if str(self.scope_id) != expected_scope_id:
            raise ValueError("caller-stamped proposal candidate identity rejected")
        object.__setattr__(self, "reference_theta", theta)
        object.__setattr__(self, "fit_manifest", dict(self.fit_manifest))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "schema": "bayesfilter.predator_prey_ttsirt_candidate.v1",
            "fitter_id": PROPOSAL_FITTER_ID,
            "classification": ROUTE_CLASSIFICATION,
            "scope_id": self.scope_id,
            "target_id": TARGET_ID,
            "target_seed": TARGET_SEED,
            "event_order": EVENT_ORDER,
            "sealed_claim_state_sha256": TARGET_STATE_SHA256,
            "sealed_claim_observation_sha256": TARGET_OBSERVATION_SHA256,
            "fit_observation_hash": self.observation_hash,
            "validation_observation_hash": self.validation_observation_hash,
            "data_role": self.data_role,
            "fit_manifest": self.fit_manifest,
            "reference_theta": self.reference_theta,
            "spec": self.spec.payload(),
            "transport_count": len(self.transports),
            "calibration_diagnostics": self.calibration_diagnostics,
            "validation_diagnostics": self.validation_diagnostics,
            "nonclaims": (
                "no exact likelihood oracle",
                "no source-faithful assembled-route claim",
                "no HMC or posterior-readiness claim",
                "no statistical superiority claim",
            ),
        }

    def compile_branch(
        self,
        *,
        observations: tf.Tensor,
        initial_reference_points: tf.Tensor,
        ancestor_uniforms: tf.Tensor,
        auxiliary_log_probabilities: tf.Tensor,
        transition_reference_points: tf.Tensor,
        online_dtype: tf.dtypes.DType = tf.float32,
        tuning_artifact: "PredatorPreyTuningArtifact",
    ) -> SourceOrderTTSIRTCompilation:
        if self.data_role != "sealed_claim_preparation":
            raise ValueError("only a sealed-claim preparation candidate may compile the claim branch")
        tuning_artifact.require_claim_candidate(self)
        return compile_predator_prey_source_order_ttsirt_proposal_branch(
            observations=observations,
            initial_transport=self.transports[0],
            transition_transports=self.transports[1:],
            previous_coordinate_maps=self.previous_maps,
            current_coordinate_maps=self.current_maps,
            initial_reference_points=initial_reference_points,
            ancestor_uniforms=ancestor_uniforms,
            auxiliary_log_probabilities=auxiliary_log_probabilities,
            transition_reference_points=transition_reference_points,
            tuning_artifact_id=tuning_artifact.artifact_id,
            online_dtype=online_dtype,
        )


@dataclass(frozen=True)
class PredatorPreyProposalAudit:
    """Repository-issued frozen-candidate audit on an untouched data path."""

    candidate_scope_id: str
    observation_hash: str
    diagnostics: tuple[Mapping[str, object], ...]
    design_order: int
    design_seed: int
    data_role: str
    audit_id: str

    def __post_init__(self) -> None:
        for name in ("candidate_scope_id", "observation_hash", "audit_id"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.observation_hash == TARGET_OBSERVATION_SHA256:
            raise ValueError("sealed observations cannot be used for proposal audit")
        if len(self.diagnostics) != 21 or not all(
            bool(row.get("finite", False)) for row in self.diagnostics
        ):
            raise ValueError("proposal audit requires 21 finite diagnostics")
        if int(self.design_order) < 2:
            raise ValueError("proposal audit design order must be at least two")
        if str(self.data_role) not in {"audit", "debug_smoke"}:
            raise ValueError("unsupported proposal audit role")
        expected = _proposal_audit_id(
            candidate_scope_id=str(self.candidate_scope_id),
            observation_hash=str(self.observation_hash),
            diagnostics=self.diagnostics,
            design_order=int(self.design_order),
            design_seed=int(self.design_seed),
            data_role=str(self.data_role),
        )
        if str(self.audit_id) != expected:
            raise ValueError("caller-stamped proposal audit identity rejected")

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "schema": "bayesfilter.predator_prey_ttsirt_frozen_audit.v1",
            "audit_id": self.audit_id,
            "candidate_scope_id": self.candidate_scope_id,
            "observation_hash": self.observation_hash,
            "design_order": self.design_order,
            "design_seed": self.design_seed,
            "data_role": self.data_role,
            "diagnostics": self.diagnostics,
            "used_for_selection": False,
        }


@dataclass(frozen=True)
class PredatorPreyTuningArtifact:
    """Repository-issued controls selected without the sealed claim data."""

    selected_spec: PredatorPreyProposalSpec
    reference_theta: tf.Tensor
    calibration_observation_hash: str
    validation_observation_hash: str
    audit_observation_hash: str
    calibration_candidate_scope_id: str
    validation_candidate_scope_id: str
    audit_candidate_scope_id: str
    audit_id: str
    audit_design_order: int
    audit_design_seed: int
    audit_diagnostics: tuple[Mapping[str, object], ...]
    selection_diagnostics: Mapping[str, object]
    tuning_scope_id: str
    artifact_id: str

    def __post_init__(self) -> None:
        theta = tf.convert_to_tensor(self.reference_theta, FIT_DTYPE)
        if theta.shape != (6,):
            raise ValueError("reference_theta must have shape [6]")
        hashes = (
            str(self.calibration_observation_hash),
            str(self.validation_observation_hash),
            str(self.audit_observation_hash),
        )
        if any(len(item) != 64 for item in hashes):
            raise ValueError("tuning data hashes must be SHA-256 digests")
        if len(set(hashes)) != 3 or TARGET_OBSERVATION_SHA256 in hashes:
            raise ValueError("tuning datasets must be disjoint from one another and the sealed claim")
        if len(self.audit_diagnostics) != 21:
            raise ValueError("audit diagnostics must contain initial plus 20 transition rows")
        if not all(bool(row.get("finite", False)) for row in self.audit_diagnostics):
            raise ValueError("audit diagnostics must be finite before issuing tuning")
        expected_audit = _proposal_audit_id(
            candidate_scope_id=str(self.audit_candidate_scope_id),
            observation_hash=hashes[2],
            diagnostics=self.audit_diagnostics,
            design_order=int(self.audit_design_order),
            design_seed=int(self.audit_design_seed),
            data_role="audit",
        )
        if str(self.audit_id) != expected_audit:
            raise ValueError("tuning artifact contains an invalid proposal audit")
        if len(str(self.tuning_scope_id)) != 64 or len(str(self.artifact_id)) != 64:
            raise ValueError("tuning_scope_id and artifact_id must be SHA-256 digests")
        for name in (
            "calibration_candidate_scope_id",
            "validation_candidate_scope_id",
            "audit_candidate_scope_id",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        expected = _tuning_artifact_id(
            selected_spec=self.selected_spec,
            reference_theta=theta,
            calibration_observation_hash=hashes[0],
            validation_observation_hash=hashes[1],
            audit_observation_hash=hashes[2],
            calibration_candidate_scope_id=str(self.calibration_candidate_scope_id),
            validation_candidate_scope_id=str(self.validation_candidate_scope_id),
            audit_candidate_scope_id=str(self.audit_candidate_scope_id),
            audit_id=str(self.audit_id),
            audit_diagnostics=self.audit_diagnostics,
            selection_diagnostics=self.selection_diagnostics,
            tuning_scope_id=str(self.tuning_scope_id),
        )
        if str(self.artifact_id) != expected:
            raise ValueError("caller-stamped tuning artifact identity rejected")
        object.__setattr__(self, "reference_theta", theta)

    def require_claim_candidate(self, candidate: PredatorPreyProposalCandidate) -> None:
        if not self.proposal_quality_pass:
            raise ValueError(
                "tuning artifact failed the proposal-quality promotion veto"
            )
        if candidate.data_role != "sealed_claim_preparation":
            raise ValueError("candidate is not a sealed-claim preparation")
        if candidate.observation_hash != TARGET_OBSERVATION_SHA256:
            raise ValueError("claim candidate observation hash mismatch")
        if candidate.validation_observation_hash != TARGET_OBSERVATION_SHA256:
            raise ValueError("claim candidate validation observation hash mismatch")
        if candidate.spec != self.selected_spec:
            raise ValueError("claim candidate controls differ from frozen tuning artifact")
        tf.debugging.assert_equal(candidate.reference_theta, self.reference_theta)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "schema": "bayesfilter.predator_prey_ttsirt_tuning.v1",
            "artifact_id": self.artifact_id,
            "tuning_scope_id": self.tuning_scope_id,
            "selected_spec": self.selected_spec.payload(),
            "reference_theta": self.reference_theta,
            "calibration_observation_hash": self.calibration_observation_hash,
            "validation_observation_hash": self.validation_observation_hash,
            "audit_observation_hash": self.audit_observation_hash,
            "calibration_candidate_scope_id": self.calibration_candidate_scope_id,
            "validation_candidate_scope_id": self.validation_candidate_scope_id,
            "audit_candidate_scope_id": self.audit_candidate_scope_id,
            "audit_id": self.audit_id,
            "audit_design_order": self.audit_design_order,
            "audit_design_seed": self.audit_design_seed,
            "audit_diagnostics": self.audit_diagnostics,
            "selection_diagnostics": self.selection_diagnostics,
            "sealed_claim_observation_hash": TARGET_OBSERVATION_SHA256,
            "audit_used_for_selection": False,
            "proposal_quality_gate": {
                "max_residual_rms_threshold": PROPOSAL_MAX_RESIDUAL_RMS,
                "max_residual_abs_threshold": PROPOSAL_MAX_RESIDUAL_ABS,
                "observed_max_residual_rms": max(
                    float(row["residual_rms"]) for row in self.audit_diagnostics
                ),
                "observed_max_residual_abs": max(
                    float(row["residual_abs_max"]) for row in self.audit_diagnostics
                ),
                "pass": self.proposal_quality_pass,
            },
        }

    @property
    def proposal_quality_pass(self) -> bool:
        """Return the declared audit promotion-veto status."""

        return bool(
            max(float(row["residual_rms"]) for row in self.audit_diagnostics)
            <= PROPOSAL_MAX_RESIDUAL_RMS
            and max(float(row["residual_abs_max"]) for row in self.audit_diagnostics)
            <= PROPOSAL_MAX_RESIDUAL_ABS
        )


def make_predator_prey_reference_theta() -> tf.Tensor:
    """Return the neutral prior-box midpoint used as the first fit hypothesis."""

    return tf.constant([0.6, 120.0, 25.0, 0.6, 0.5, 0.5], FIT_DTYPE)


def make_predator_prey_coordinate_map(
    *,
    locations: tf.Tensor,
    scale: float,
    family: str = "shifted_algebraic",
) -> HighDimCoordinateMap:
    """Build one declared target-specific full-support coordinate map."""

    from bayesfilter.highdim.zhao_cui_coupled_nonlinear import (
        GaussianQuantileCoordinateMap,
        ShiftedAlgebraicCoordinateMap,
    )

    location_vector = tf.reshape(tf.convert_to_tensor(locations, FIT_DTYPE), [-1])
    map_type = {
        "shifted_algebraic": ShiftedAlgebraicCoordinateMap,
        "gaussian_quantile": GaussianQuantileCoordinateMap,
    }.get(str(family))
    if map_type is None:
        raise ValueError("unsupported predator-prey coordinate map family")
    return map_type(
        locations=location_vector,
        scales=tf.fill(tf.shape(location_vector), tf.constant(scale, FIT_DTYPE)),
    )


def make_reference_design(
    *, dimension: int, order: int, design_seed: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Create deterministic full-domain Gauss-Legendre reference quadrature."""

    dimension = int(dimension)
    order = int(order)
    if dimension < 1:
        raise ValueError("design dimension must be positive")
    if order < 2:
        raise ValueError("design order must be at least 2")
    nodes, weights = legendre_gauss_nodes_weights(order)
    node_mesh = tf.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_mesh = tf.meshgrid(*([0.5 * weights] * dimension), indexing="ij")
    points = tf.stack([tf.reshape(item, [-1]) for item in node_mesh], axis=1)
    product_weights = tf.ones([tf.shape(points)[0]], FIT_DTYPE)
    for item in weight_mesh:
        product_weights = product_weights * tf.reshape(item, [-1])
    # Different orders provide disjoint node sets; the seed remains bound to
    # initialization and candidate identity rather than perturbing quadrature.
    del design_seed
    return points, product_weights


def _basis(dimension: int, degree: int) -> ProductBasis:
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="uniform_probability_on_minus_one_one",
    )
    return ProductBasis(
        [LegendreBasis1D(BoundedInterval(-1.0, 1.0), int(degree)) for _ in range(dimension)],
        convention,
    )


def _fit_one_density(
    *,
    product_basis: ProductBasis,
    points: tf.Tensor,
    weights: tf.Tensor,
    log_target: tf.Tensor,
    spec: PredatorPreyProposalSpec,
    seed: int,
) -> tuple[SquaredTTDensity, Mapping[str, object]]:
    log_target = tf.convert_to_tensor(log_target, FIT_DTYPE)
    weights = tf.convert_to_tensor(weights, FIT_DTYPE)
    log_normalizer = tf.reduce_logsumexp(tf.math.log(weights) + log_target)
    normalized = log_target - log_normalizer
    target_sqrt = tf.exp(0.5 * normalized)
    ranks = tuple([1] + [min(spec.rank, spec.degree + 1)] * (product_basis.dimension - 1) + [1])
    config = P75TrainableTTConfig(
        product_basis=product_basis,
        ranks=ranks,
        tau=tf.constant(spec.defensive_tau, FIT_DTYPE),
        normalizer_floor=tf.constant(1e-12, FIT_DTYPE),
        denominator_floor=tf.constant(1e-12, FIT_DTYPE),
        l1_weight=tf.constant(spec.l1_weight, FIT_DTYPE),
        l2_weight=tf.constant(spec.ridge, FIT_DTYPE),
        learning_rate=DEFAULT_LEARNING_RATE,
        gradient_clip_norm=10.0,
        seed=int(seed),
        metadata={
            "scope": "predator_prey_t20_fixed_variant",
            "training_role": "calibration_only",
            "route_classification": ROUTE_CLASSIFICATION,
            "ridge": spec.ridge,
            "ridge_role": "l2_core_penalty",
            "l1_weight_tuned": True,
        },
    )
    initial_cores, realized_ranks, initializer = _quadrature_projection_tt_svd(
        product_basis=product_basis,
        points=points,
        weights=weights,
        target_sqrt=target_sqrt,
        rank_cap=spec.rank,
    )
    if realized_ranks != ranks:
        config = P75TrainableTTConfig(
            product_basis=product_basis,
            ranks=realized_ranks,
            tau=config.tau,
            normalizer_floor=config.normalizer_floor,
            denominator_floor=config.denominator_floor,
            l1_weight=config.l1_weight,
            l2_weight=config.l2_weight,
            learning_rate=config.learning_rate,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=config.seed,
            metadata={**dict(config.metadata), "realized_ranks": realized_ranks},
        )
    trainer = TrainableFunctionalTT(config, initial_cores=initial_cores)
    optimizer = make_adam_optimizer(config)
    batch = P75ObjectiveBatch(
        points=points,
        target_values=target_sqrt,
        weights=weights,
        provenance_label="predator_prey_calibration_only",
    )
    prefit_losses = []
    for _ in range(spec.prefit_steps):
        terms = trainer.square_root_prefit_step(
            batch,
            optimizer,
            reference_l2_weight=tf.constant(0.0, FIT_DTYPE),
        )
        prefit_losses.append(float(terms.total_loss.numpy()))
    losses = []
    for _ in range(spec.train_steps):
        terms = trainer.train_step(batch, optimizer)
        losses.append(float(terms.total_loss.numpy()))
    density = trainer.snapshot_density()
    return density, {
        "log_normalizer": float(log_normalizer.numpy()),
        "prefit_steps": spec.prefit_steps,
        "final_prefit_loss": prefit_losses[-1] if prefit_losses else None,
        "prefit_loss_trace": tuple(prefit_losses),
        "train_steps": spec.train_steps,
        "final_loss": losses[-1] if losses else None,
        "loss_trace": tuple(losses),
        "density_normalizer": float(density.normalizer().numpy()),
        "fit_dtype": FIT_DTYPE.name,
        "l1_weight": spec.l1_weight,
        "ridge": spec.ridge,
        "initializer": initializer,
        "realized_ranks": realized_ranks,
    }


def _quadrature_projection_tt_svd(
    *,
    product_basis: ProductBasis,
    points: tf.Tensor,
    weights: tf.Tensor,
    target_sqrt: tf.Tensor,
    rank_cap: int,
) -> tuple[tuple[tf.Tensor, ...], tuple[int, ...], Mapping[str, object]]:
    """Project the normalized square root and compress it with TensorFlow TT-SVD."""

    dimension = product_basis.dimension
    if dimension > len(string.ascii_lowercase):
        raise ValueError("projection dimension exceeds einsum label budget")
    labels = string.ascii_lowercase[:dimension]
    basis_values = tuple(
        product_basis.evaluate_axis(axis, points[:, axis])
        for axis in range(dimension)
    )
    equation = "n," + ",".join(f"n{label}" for label in labels) + "->" + labels
    coefficients = tf.einsum(
        equation,
        tf.convert_to_tensor(weights, FIT_DTYPE)
        * tf.convert_to_tensor(target_sqrt, FIT_DTYPE),
        *basis_values,
    )
    if not bool(tf.reduce_all(tf.math.is_finite(coefficients)).numpy()):
        raise ValueError("TT-SVD projection coefficients must be finite")

    cores = []
    ranks = [1]
    left_rank = 1
    remainder = coefficients
    discarded_square = tf.constant(0.0, FIT_DTYPE)
    for axis in range(dimension - 1):
        basis_dim = product_basis.bases[axis].basis_dim
        matrix = tf.reshape(remainder, [left_rank * basis_dim, -1])
        singular_values, left_vectors, right_vectors = tf.linalg.svd(
            matrix, full_matrices=False
        )
        realized_rank = min(int(rank_cap), int(singular_values.shape[0]))
        if realized_rank < int(singular_values.shape[0]):
            discarded_square += tf.reduce_sum(
                tf.square(singular_values[realized_rank:])
            )
        cores.append(
            tf.reshape(
                left_vectors[:, :realized_rank],
                [left_rank, basis_dim, realized_rank],
            )
        )
        remainder = (
            singular_values[:realized_rank, tf.newaxis]
            * tf.transpose(right_vectors[:, :realized_rank])
        )
        left_rank = realized_rank
        ranks.append(realized_rank)
    cores.append(
        tf.reshape(
            remainder,
            [left_rank, product_basis.bases[-1].basis_dim, 1],
        )
    )
    ranks.append(1)
    return tuple(cores), tuple(ranks), {
        "family": "deterministic_quadrature_projection_tt_svd",
        "classification": ROUTE_CLASSIFICATION,
        "rank_cap": int(rank_cap),
        "realized_ranks": tuple(ranks),
        "coefficient_tensor_shape": tuple(int(item) for item in coefficients.shape),
        "discarded_singular_value_square_sum": float(discarded_square.numpy()),
    }


def _initial_log_target(
    model: PredatorPreySSM,
    coordinate_map: HighDimCoordinateMap,
    reference_points: tf.Tensor,
    theta: tf.Tensor,
) -> tf.Tensor:
    physical, log_det = coordinate_map.forward(reference_points)
    return (
        model.initial_log_density(theta, physical)
        + log_det
        - _reference_log_density(2)
    )


def _adjacent_log_target(
    model: PredatorPreySSM,
    previous_density: SquaredTTDensity,
    previous_map: HighDimCoordinateMap,
    current_map: HighDimCoordinateMap,
    reference_points: tf.Tensor,
    observation: tf.Tensor,
    time_index: int,
    theta: tf.Tensor,
) -> tf.Tensor:
    previous_local = reference_points[:, :2]
    current_local = reference_points[:, 2:]
    previous_physical, _previous_log_det = previous_map.forward(previous_local)
    current_physical, current_log_det = current_map.forward(current_local)
    previous_axes = (
        (0, 1)
        if len(previous_density.sqrt_tt.cores) == 2
        else (2, 3)
    )
    previous_log = tf.math.log(
        previous_density.normalized_marginal_density_values(previous_axes, previous_local)
    )
    transition_log = model.transition_log_density(
        theta, previous_physical, current_physical, time_index
    )
    observation_log = model.observation_log_density(
        theta, current_physical, observation, time_index
    )
    return (
        previous_log
        + transition_log
        + observation_log
        + current_log_det
        - _reference_log_density(2)
    )


def _reference_log_density(dimension: int) -> tf.Tensor:
    """Log density of the uniform probability reference on ``[-1,1]^d``."""

    return -tf.cast(int(dimension), FIT_DTYPE) * tf.math.log(
        tf.constant(2.0, FIT_DTYPE)
    )


def _validation_diagnostic(
    density: SquaredTTDensity,
    points: tf.Tensor,
    weights: tf.Tensor,
    log_target: tf.Tensor,
) -> Mapping[str, object]:
    log_z = tf.reduce_logsumexp(tf.math.log(weights) + log_target)
    normalized_target = log_target - log_z
    fitted = density.log_density(points)
    residual = normalized_target - fitted
    weighted = tf.exp(normalized_target) * weights
    weighted = weighted / tf.reduce_sum(weighted)
    centered = residual - tf.reduce_sum(weighted * residual)
    return {
        "cross_entropy": float((-tf.reduce_sum(weighted * fitted)).numpy()),
        "target_quadrature_log_normalizer": float(log_z.numpy()),
        "density_internal_log_normalizer": float(
            tf.math.log(density.normalizer()).numpy()
        ),
        "residual_rms": float(tf.sqrt(tf.reduce_sum(weighted * tf.square(centered))).numpy()),
        "residual_abs_max": float(tf.reduce_max(tf.abs(residual)).numpy()),
        "finite": bool(
            tf.reduce_all(
                tf.math.is_finite(
                    tf.stack([log_z, tf.math.log(density.normalizer()), tf.reduce_max(tf.abs(residual))])
                )
            ).numpy()
        ),
    }


def fit_predator_prey_proposal_candidate(
    *,
    observations: tf.Tensor,
    spec: PredatorPreyProposalSpec,
    calibration_order: int = 5,
    validation_order: int = 6,
    calibration_seed: int = 8110401,
    validation_seed: int = 8110402,
    reference_theta: tf.Tensor | None = None,
    data_role: str = "calibration",
    tuning_artifact: PredatorPreyTuningArtifact | None = None,
    frozen_control_source_scope_id: str | None = None,
) -> PredatorPreyProposalCandidate:
    """Fit one observation-specific T20 proposal on disjoint quadrature designs."""

    model = PredatorPreySSM(dtype=FIT_DTYPE)
    y = tf.convert_to_tensor(observations, FIT_DTYPE)
    if y.shape != (20, 2):
        raise ValueError("predator-prey proposal fitting requires observations [20,2]")
    observation_hash = _tensor_hash(y)
    validation_observation_hash = observation_hash
    role = str(data_role)
    if role not in {
        "calibration",
        "validation",
        "audit",
        "sealed_claim_preparation",
        "debug_smoke",
    }:
        raise ValueError("proposal fitting is not allowed for this data role")
    if role != "sealed_claim_preparation":
        if TARGET_OBSERVATION_SHA256 in {
            observation_hash,
            validation_observation_hash,
        }:
            raise ValueError(
                "sealed seed-81104 observations cannot be used for proposal-control tuning"
            )
    if role == "sealed_claim_preparation" and tuning_artifact is None:
        raise ValueError("sealed claim preparation requires a repository-issued tuning artifact")
    if role == "sealed_claim_preparation" and (
        observation_hash != TARGET_OBSERVATION_SHA256
        or validation_observation_hash != TARGET_OBSERVATION_SHA256
    ):
        raise ValueError("sealed claim preparation requires the sealed observation path")
    if role == "audit":
        source_scope = str(frozen_control_source_scope_id or "")
        if len(source_scope) != 64 or any(
            character not in "0123456789abcdef" for character in source_scope
        ):
            raise ValueError(
                "audit proposal fit requires a validation-issued frozen control scope"
            )
    elif frozen_control_source_scope_id is not None:
        raise ValueError(
            "frozen_control_source_scope_id is reserved for final-only audit fitting"
        )
    theta = (
        make_predator_prey_reference_theta()
        if reference_theta is None
        else tf.convert_to_tensor(reference_theta, FIT_DTYPE)
    )
    if theta.shape != (6,):
        raise ValueError("reference_theta must have shape [6]")
    if tuning_artifact is not None:
        if spec != tuning_artifact.selected_spec:
            raise ValueError("claim controls differ from the tuning artifact")
        tf.debugging.assert_equal(theta, tuning_artifact.reference_theta)
        if not tuning_artifact.proposal_quality_pass:
            raise ValueError(
                "tuning artifact failed the proposal-quality promotion veto"
            )

    previous_density: SquaredTTDensity | None = None
    previous_maps: list[HighDimCoordinateMap] = []
    current_maps: list[HighDimCoordinateMap] = []
    transports: list[FixedTTSIRTTransport] = []
    calibration_rows = []
    validation_rows = []
    last_current_map: HighDimCoordinateMap | None = None
    for time_index in range(21):
        if time_index == 0:
            current_map = make_predator_prey_coordinate_map(
                locations=model.initial_mean,
                scale=spec.coordinate_scale,
                family=spec.coordinate_map_family,
            )
            previous_map = current_map
            joint_map = current_map
            calibration_points, calibration_weights = make_reference_design(
                dimension=2, order=calibration_order, design_seed=calibration_seed
            )
            validation_points, validation_weights = make_reference_design(
                dimension=2, order=validation_order, design_seed=validation_seed
            )
            calibration_target = _initial_log_target(
                model, joint_map, calibration_points, theta
            )
            validation_target = _initial_log_target(
                model, joint_map, validation_points, theta
            )
            target_dimension = 2
        else:
            if last_current_map is None:
                raise RuntimeError("transition fit requires the previous coordinate map")
            previous_map = last_current_map
            current_map = make_predator_prey_coordinate_map(
                locations=y[time_index - 1],
                scale=spec.coordinate_scale,
                family=spec.coordinate_map_family,
            )
            calibration_points, calibration_weights = make_reference_design(
                dimension=4,
                order=calibration_order,
                design_seed=calibration_seed + time_index,
            )
            validation_points, validation_weights = make_reference_design(
                dimension=4,
                order=validation_order,
                design_seed=validation_seed + time_index,
            )
            if previous_density is None:
                raise RuntimeError("adjacent fit requires previous density")
            calibration_target = _adjacent_log_target(
                model,
                previous_density,
                previous_map,
                current_map,
                calibration_points,
                y[time_index - 1],
                time_index,
                theta,
            )
            validation_target = _adjacent_log_target(
                model,
                previous_density,
                previous_map,
                current_map,
                validation_points,
                y[time_index - 1],
                time_index,
                theta,
            )
            target_dimension = 4

        density, fit_diag = _fit_one_density(
            product_basis=_basis(target_dimension, spec.degree),
            points=calibration_points,
            weights=calibration_weights,
            log_target=calibration_target,
            spec=spec,
            seed=calibration_seed + 17 * time_index,
        )
        calibration_rows.append({"time_index": time_index, **fit_diag})
        validation_rows.append(
            {
                "time_index": time_index,
                **_validation_diagnostic(
                    density, validation_points, validation_weights, validation_target
                ),
            }
        )
        transports.append(
            FixedTTSIRTTransport(
                density=density,
                cdf_config=KRCDFConfig(
                    grid_size=spec.cdf_grid_size,
                    bisection_steps=spec.cdf_bisection_steps,
                    monotonicity_tolerance=1e-12,
                    bracket_tolerance=1e-12,
                    denominator_floor=1e-12,
                    max_floor_count=0,
                ),
            )
        )
        previous_density = density
        last_current_map = current_map
        if time_index > 0:
            previous_maps.append(previous_map)
            current_maps.append(current_map)

    fit_manifest = {
        "fitter_id": PROPOSAL_FITTER_ID,
        "design_family": "full_domain_tensor_gauss_legendre_distinct_orders_v1",
        "calibration_order": int(calibration_order),
        "validation_order": int(validation_order),
        "calibration_seed": int(calibration_seed),
        "validation_seed": int(validation_seed),
        "calibration_design_hash_initial": _tensor_hash(
            make_reference_design(
                dimension=2,
                order=calibration_order,
                design_seed=calibration_seed,
            )[0]
        ),
        "validation_design_hash_initial": _tensor_hash(
            make_reference_design(
                dimension=2,
                order=validation_order,
                design_seed=validation_seed,
            )[0]
        ),
        "calibration_design_hash_transition": _tensor_hash(
            make_reference_design(
                dimension=4,
                order=calibration_order,
                design_seed=calibration_seed + 1,
            )[0]
        ),
        "validation_design_hash_transition": _tensor_hash(
            make_reference_design(
                dimension=4,
                order=validation_order,
                design_seed=validation_seed + 1,
            )[0]
        ),
        "fit_dtype": FIT_DTYPE.name,
        "source_dependency_sha256": _source_dependency_sha256(),
        "frozen_control_source_scope_id": (
            None
            if role != "audit"
            else str(frozen_control_source_scope_id)
        ),
        "audit_controls_selected_elsewhere": role == "audit",
    }
    scope_id = _proposal_candidate_scope_id(
        spec=spec,
        reference_theta=theta,
        previous_maps=tuple(previous_maps),
        current_maps=tuple(current_maps),
        transports=tuple(transports),
        calibration_diagnostics=tuple(calibration_rows),
        validation_diagnostics=tuple(validation_rows),
        observation_hash=observation_hash,
        validation_observation_hash=validation_observation_hash,
        data_role=role,
        fit_manifest=fit_manifest,
    )
    return PredatorPreyProposalCandidate(
        spec=spec,
        reference_theta=theta,
        previous_maps=tuple(previous_maps),
        current_maps=tuple(current_maps),
        transports=tuple(transports),
        calibration_diagnostics=tuple(calibration_rows),
        validation_diagnostics=tuple(validation_rows),
        observation_hash=observation_hash,
        validation_observation_hash=validation_observation_hash,
        data_role=role,
        fit_manifest=fit_manifest,
        scope_id=scope_id,
    )


def evaluate_predator_prey_proposal_candidate(
    candidate: PredatorPreyProposalCandidate,
    *,
    observations: tf.Tensor,
    design_order: int = 6,
    design_seed: int = 8110701,
    data_role: str = "audit",
) -> PredatorPreyProposalAudit:
    """Evaluate a frozen candidate on a third data path without refitting.

    The candidate's maps, TT densities, and transport settings are never
    rebuilt here.  Only the observation-dependent target used for a held-out
    diagnostic is evaluated.  This is the audit lane required before issuing
    a tuning artifact.
    """

    if not isinstance(candidate, PredatorPreyProposalCandidate):
        raise TypeError("candidate must be a PredatorPreyProposalCandidate")
    if str(data_role) not in {"audit", "debug_smoke"}:
        raise ValueError("frozen candidate evaluation is restricted to audit diagnostics")
    y = tf.convert_to_tensor(observations, FIT_DTYPE)
    if y.shape != (20, 2):
        raise ValueError("predator-prey audit requires observations [20,2]")
    observation_hash = _tensor_hash(y)
    if observation_hash == TARGET_OBSERVATION_SHA256:
        raise ValueError("sealed observations cannot be used for proposal audit")
    same_candidate_path = observation_hash == candidate.observation_hash == candidate.validation_observation_hash
    if observation_hash in {
        candidate.observation_hash,
        candidate.validation_observation_hash,
    } and not (same_candidate_path and candidate.data_role == "audit"):
        raise ValueError("audit observations must be fresh or an audit candidate's own fit path")
    rows: list[Mapping[str, object]] = []
    initial_map = candidate.previous_maps[0]
    initial_points, initial_weights = make_reference_design(
        dimension=2, order=design_order, design_seed=design_seed
    )
    rows.append(
        {
            "time_index": 0,
            **_validation_diagnostic(
                candidate.transports[0].density,
                initial_points,
                initial_weights,
                _initial_log_target(
                    PredatorPreySSM(dtype=FIT_DTYPE),
                    initial_map,
                    initial_points,
                    candidate.reference_theta,
                ),
            ),
        }
    )
    model = PredatorPreySSM(dtype=FIT_DTYPE)
    for time_index in range(1, 21):
        points, weights = make_reference_design(
            dimension=4,
            order=design_order,
            design_seed=design_seed + time_index,
        )
        target = _adjacent_log_target(
            model,
            candidate.transports[time_index - 1].density,
            candidate.previous_maps[time_index - 1],
            candidate.current_maps[time_index - 1],
            points,
            y[time_index - 1],
            time_index,
            candidate.reference_theta,
        )
        rows.append(
            {
                "time_index": time_index,
                **_validation_diagnostic(
                    candidate.transports[time_index].density,
                    points,
                    weights,
                    target,
                ),
            }
        )
    diagnostics = tuple(rows)
    audit_id = _proposal_audit_id(
        candidate_scope_id=candidate.scope_id,
        observation_hash=observation_hash,
        diagnostics=diagnostics,
        design_order=int(design_order),
        design_seed=int(design_seed),
        data_role=str(data_role),
    )
    return PredatorPreyProposalAudit(
        candidate_scope_id=candidate.scope_id,
        observation_hash=observation_hash,
        diagnostics=diagnostics,
        design_order=int(design_order),
        design_seed=int(design_seed),
        data_role=str(data_role),
        audit_id=audit_id,
    )


def select_l1_candidate(
    candidates: Sequence[PredatorPreyProposalCandidate],
    *,
    positive_margin: float = 0.005,
) -> tuple[PredatorPreyProposalCandidate, Mapping[str, object]]:
    """Select L1 only from validation diagnostics, retaining a zero arm."""

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("L1 selection requires at least one candidate")
    if any(item.data_role != "validation" for item in candidates):
        raise ValueError("L1 selection accepts validation-role candidates only")
    fit_hashes = {item.observation_hash for item in candidates}
    validation_hashes = {item.validation_observation_hash for item in candidates}
    if len(fit_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("L1 arms must use the same validation path")
    structural = {
        tuple(sorted(item.spec.structural_payload().items())) for item in candidates
    }
    if len(structural) != 1:
        raise ValueError("L1 arms must hold all non-L1 controls fixed")
    zero = tuple(item for item in candidates if item.spec.l1_weight == 0.0)
    positive = tuple(item for item in candidates if item.spec.l1_weight > 0.0)
    if len(zero) != 1:
        raise ValueError("L1 selection requires exactly one zero-L1 comparator")
    def score(item: PredatorPreyProposalCandidate) -> float:
        return max(float(row["residual_rms"]) for row in item.validation_diagnostics)
    zero_score = score(zero[0])
    if not positive:
        return zero[0], {
            "selected_l1_weight": 0.0,
            "zero_validation_residual_rms": zero_score,
            "best_positive_validation_residual_rms": None,
            "selection_reason": "zero_l1_only_debug_arm",
        }
    best_positive = min(positive, key=lambda item: (score(item), item.spec.l1_weight))
    positive_score = score(best_positive)
    selected = (
        best_positive
        if positive_score <= zero_score - float(positive_margin)
        else zero[0]
    )
    return selected, {
        "selected_l1_weight": selected.spec.l1_weight,
        "zero_validation_residual_rms": zero_score,
        "best_positive_validation_residual_rms": positive_score,
        "positive_margin": float(positive_margin),
        "selection_reason": (
            "positive_l1_margin_met"
            if selected.spec.l1_weight > 0.0
            else "zero_l1_retained_margin_not_met"
        ),
        "audit_data_used_for_selection": False,
    }


def select_structural_candidate(
    candidates: Sequence[PredatorPreyProposalCandidate],
) -> tuple[PredatorPreyProposalCandidate, Mapping[str, object]]:
    """Nominate structural controls using calibration-path diagnostics only."""

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("structural selection requires at least one candidate")
    if any(item.data_role != "calibration" for item in candidates):
        raise ValueError("structural selection accepts calibration-role candidates only")
    if len({item.observation_hash for item in candidates}) != 1:
        raise ValueError("structural candidates must use one calibration split")
    if any(item.observation_hash != item.validation_observation_hash for item in candidates):
        raise ValueError("calibration structural diagnostics must use the calibration path")

    def score(item: PredatorPreyProposalCandidate) -> float:
        return max(float(row["residual_rms"]) for row in item.validation_diagnostics)

    selected = min(candidates, key=lambda item: (score(item), item.scope_id))
    return selected, {
        "selected_structural_spec": selected.spec.structural_payload(),
        "selected_calibration_max_residual_rms": score(selected),
        "candidate_count": len(candidates),
        "audit_data_used_for_selection": False,
        "validation_data_used_for_selection": False,
    }


def make_tuning_artifact(
    *,
    calibration_candidate: PredatorPreyProposalCandidate,
    selected_candidate: PredatorPreyProposalCandidate,
    audit_candidate: PredatorPreyProposalCandidate,
    calibration_observation_hash: str,
    validation_observation_hash: str,
    audit: PredatorPreyProposalAudit,
    selection_diagnostics: Mapping[str, object],
) -> PredatorPreyTuningArtifact:
    """Issue a tuning identity from the selected candidate and split hashes."""

    if calibration_candidate.data_role != "calibration":
        raise ValueError("tuning calibration candidate must come from calibration")
    if selected_candidate.data_role != "validation":
        raise ValueError("tuning selection candidate must come from validation")
    if audit_candidate.data_role != "audit":
        raise ValueError(
            "tuning audit candidate must come from final-only audit fitting"
        )
    if calibration_candidate.observation_hash != str(calibration_observation_hash):
        raise ValueError("calibration candidate does not match calibration observation hash")
    if selected_candidate.observation_hash != str(validation_observation_hash):
        raise ValueError("selected candidate does not match validation observation hash")
    if calibration_candidate.spec.structural_payload() != selected_candidate.spec.structural_payload():
        raise ValueError("validation candidate changed frozen non-L1 controls")
    if audit_candidate.spec != selected_candidate.spec:
        raise ValueError("audit candidate controls differ from selected validation controls")
    if (
        audit_candidate.fit_manifest.get("frozen_control_source_scope_id")
        != selected_candidate.scope_id
    ):
        raise ValueError(
            "audit candidate is not bound to the validation-selected frozen controls"
        )
    if not isinstance(audit, PredatorPreyProposalAudit):
        raise TypeError("audit must be a PredatorPreyProposalAudit")
    if audit.data_role != "audit":
        raise ValueError("tuning issuance requires an audit-role result")
    if audit.candidate_scope_id != audit_candidate.scope_id:
        raise ValueError("audit was not computed from the audit candidate")
    if audit.observation_hash in {
        str(calibration_observation_hash),
        str(validation_observation_hash),
        TARGET_OBSERVATION_SHA256,
    }:
        raise ValueError("audit observations must be distinct and non-sealed")
    audit_rows = tuple(dict(row) for row in audit.diagnostics)
    tuning_scope_id = _tuning_scope_id(
        selected_spec=selected_candidate.spec,
        reference_theta=selected_candidate.reference_theta,
        calibration_observation_hash=str(calibration_observation_hash),
        validation_observation_hash=str(validation_observation_hash),
        audit_observation_hash=audit.observation_hash,
    )
    artifact_id = _tuning_artifact_id(
        selected_spec=selected_candidate.spec,
        reference_theta=selected_candidate.reference_theta,
        calibration_observation_hash=str(calibration_observation_hash),
        validation_observation_hash=str(validation_observation_hash),
        audit_observation_hash=audit.observation_hash,
        calibration_candidate_scope_id=calibration_candidate.scope_id,
        validation_candidate_scope_id=selected_candidate.scope_id,
        audit_candidate_scope_id=audit_candidate.scope_id,
        audit_id=audit.audit_id,
        audit_diagnostics=audit_rows,
        selection_diagnostics=selection_diagnostics,
        tuning_scope_id=str(tuning_scope_id),
    )
    return PredatorPreyTuningArtifact(
        selected_spec=selected_candidate.spec,
        reference_theta=selected_candidate.reference_theta,
        calibration_observation_hash=str(calibration_observation_hash),
        validation_observation_hash=str(validation_observation_hash),
        audit_observation_hash=audit.observation_hash,
        calibration_candidate_scope_id=calibration_candidate.scope_id,
        validation_candidate_scope_id=selected_candidate.scope_id,
        audit_candidate_scope_id=audit_candidate.scope_id,
        audit_id=audit.audit_id,
        audit_design_order=audit.design_order,
        audit_design_seed=audit.design_seed,
        audit_diagnostics=audit_rows,
        selection_diagnostics=dict(selection_diagnostics),
        tuning_scope_id=str(tuning_scope_id),
        artifact_id=artifact_id,
    )


def _tuning_artifact_id(
    *,
    selected_spec: PredatorPreyProposalSpec,
    reference_theta: tf.Tensor,
    calibration_observation_hash: str,
    validation_observation_hash: str,
    audit_observation_hash: str,
    calibration_candidate_scope_id: str,
    validation_candidate_scope_id: str,
    audit_candidate_scope_id: str,
    audit_id: str,
    audit_diagnostics: Sequence[Mapping[str, object]],
    selection_diagnostics: Mapping[str, object],
    tuning_scope_id: str,
) -> str:
    digest = hashlib.sha256()
    for item in (
        "bayesfilter.predator_prey_ttsirt_tuning.v1",
        selected_spec.payload(),
        reference_theta,
        calibration_observation_hash,
        validation_observation_hash,
        audit_observation_hash,
        calibration_candidate_scope_id,
        validation_candidate_scope_id,
        audit_candidate_scope_id,
        audit_id,
        tuple(audit_diagnostics),
        selection_diagnostics,
        tuning_scope_id,
    ):
        _hash_update(digest, item)
    return digest.hexdigest()


def _tuning_scope_id(
    *,
    selected_spec: PredatorPreyProposalSpec,
    reference_theta: tf.Tensor,
    calibration_observation_hash: str,
    validation_observation_hash: str,
    audit_observation_hash: str,
) -> str:
    digest = hashlib.sha256()
    for item in (
        "bayesfilter.predator_prey_ttsirt_tuning_scope.v1",
        TARGET_ID,
        EVENT_ORDER,
        20,
        2,
        6,
        "fit_float64_online_float32_tf32_xla",
        selected_spec.payload(),
        reference_theta,
        calibration_observation_hash,
        validation_observation_hash,
        audit_observation_hash,
        _source_dependency_sha256(),
    ):
        _hash_update(digest, item)
    return digest.hexdigest()


def _proposal_candidate_scope_id(
    *,
    spec: PredatorPreyProposalSpec,
    reference_theta: tf.Tensor,
    previous_maps: Sequence[HighDimCoordinateMap],
    current_maps: Sequence[HighDimCoordinateMap],
    transports: Sequence[FixedTTSIRTTransport],
    calibration_diagnostics: Sequence[Mapping[str, object]],
    validation_diagnostics: Sequence[Mapping[str, object]],
    observation_hash: str,
    validation_observation_hash: str,
    data_role: str,
    fit_manifest: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    for item in (
        "bayesfilter.predator_prey_ttsirt_candidate.v1",
        TARGET_ID,
        TARGET_STATE_SHA256,
        TARGET_OBSERVATION_SHA256,
        EVENT_ORDER,
        spec.payload(),
        reference_theta,
        observation_hash,
        validation_observation_hash,
        data_role,
        fit_manifest,
        tuple(item.manifest_payload() for item in previous_maps),
        tuple(item.manifest_payload() for item in current_maps),
        tuple(
            {
                "transport": item.manifest_payload(),
                "density": item.density.manifest_payload(),
            }
            for item in transports
        ),
        tuple(calibration_diagnostics),
        tuple(validation_diagnostics),
    ):
        _hash_update(digest, item)
    return digest.hexdigest()


def _proposal_audit_id(
    *,
    candidate_scope_id: str,
    observation_hash: str,
    diagnostics: Sequence[Mapping[str, object]],
    design_order: int,
    design_seed: int,
    data_role: str,
) -> str:
    digest = hashlib.sha256()
    for item in (
        "bayesfilter.predator_prey_ttsirt_frozen_audit.v1",
        candidate_scope_id,
        observation_hash,
        tuple(diagnostics),
        int(design_order),
        int(design_seed),
        str(data_role),
        _source_dependency_sha256(),
    ):
        _hash_update(digest, item)
    return digest.hexdigest()


def _source_dependency_sha256() -> Mapping[str, str]:
    from bayesfilter.highdim import models as model_module
    from bayesfilter.highdim import stochastic_density_training as training_module
    from bayesfilter.highdim import transport as transport_module

    return {
        "proposal": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": hashlib.sha256(Path(inspect.getfile(model_module)).read_bytes()).hexdigest(),
        "training": hashlib.sha256(
            Path(inspect.getfile(training_module)).read_bytes()
        ).hexdigest(),
        "transport": hashlib.sha256(
            Path(inspect.getfile(transport_module)).read_bytes()
        ).hexdigest(),
    }


def _hash_update(digest: object, value: object) -> None:
    if isinstance(value, tf.Tensor):
        digest.update(b"tensor\0")
        digest.update(value.dtype.name.encode("ascii"))
        digest.update(repr(value.shape.as_list()).encode("ascii"))
        digest.update(tf.io.serialize_tensor(value).numpy())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_update(digest, str(key))
            _hash_update(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_update(digest, item)
        return
    if isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode("ascii"))
        return
    if isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii"))
        return
    if value is None:
        digest.update(b"none\0")
        return
    encoded = str(value).encode("utf-8")
    digest.update(b"string\0" + str(len(encoded)).encode("ascii") + b"\0" + encoded)


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


__all__ = [
    "FIT_DTYPE",
    "PredatorPreyProposalCandidate",
    "PredatorPreyProposalAudit",
    "PredatorPreyProposalSpec",
    "PredatorPreyTuningArtifact",
    "PROPOSAL_MAX_RESIDUAL_ABS",
    "PROPOSAL_MAX_RESIDUAL_RMS",
    "PROPOSAL_FITTER_ID",
    "ROUTE_CLASSIFICATION",
    "evaluate_predator_prey_proposal_candidate",
    "fit_predator_prey_proposal_candidate",
    "make_tuning_artifact",
    "make_predator_prey_coordinate_map",
    "make_predator_prey_reference_theta",
    "make_reference_design",
    "select_l1_candidate",
    "select_structural_candidate",
]
