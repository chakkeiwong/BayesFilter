"""Deterministic T1 training-base baseline for the Austria SIR Lane-B route."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import (
    AlgebraicMap,
    ProductBasis,
    p85_author_sir_lagrangep_algebraic_product_basis_spec,
)
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.source_route import (
    SourceRouteCoordinateFrame,
    source_route_recenter,
)
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.stochastic_density_training import (
    P75ObjectiveBatch,
    P75TrainableTTConfig,
    TrainableFunctionalTT,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    LANE_B_TARGET_ID,
    SIR_JOINT_DIM,
    LaneBT1ProposalCloud,
    target_manifest,
    tensor_sha256,
    t1_joint_log_density,
)


DTYPE = tf.float64
BASELINE_ID = "zhao_cui_austria_sir_fixed_variant_training_base_v1"
ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t1.v1"
IDENTITY_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t1_identity.v1"
REFERENCE_LOG_DENSITY_ID = "affine_algebraic_uniform_probability_reference_v1"
VALUE_DEFINITION_ID = "sum_log_tt_normalizer_minus_shift_v1"
TRAINING_KERNEL_ID = "lane_b_training_base_batch_native_xla_v1"
INITIALIZATION_ID = "lane_b_unit_constant_path_seeded_channels_1e-6_v1"
MASS_PRECOMPUTE_ID = "lane_b_setup_static_cpu_mass_integral_constants_v1"
NORMALIZER_CALIBRATION_ID = "independent_mc_then_exact_core_rescale_tau_fixed_v1"
LOG_REFERENCE_DENSITY_CONSTANT = SIR_JOINT_DIM * math.log(2.0)


def lane_b_measure_convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="uniform_probability_density_2^-36",
        physical_coordinate_name="affine_algebraic_r",
        reference_coordinate_name="u",
    )


@dataclass(frozen=True)
class _LaneBFrozenMassBasis:
    """Delegate basis evaluation while freezing tiny setup-static contractions."""

    delegate: object
    reference_mass: tf.Tensor
    reference_integral: tf.Tensor
    lebesgue_mass: tf.Tensor
    lebesgue_integral: tf.Tensor

    @property
    def basis_dim(self) -> int:
        return int(self.delegate.basis_dim)

    @property
    def dtype(self) -> tf.DType:
        return self.delegate.dtype

    @property
    def domain(self):
        return self.delegate.domain

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        return self.delegate.evaluate(points)

    def derivative(self, points: tf.Tensor) -> tf.Tensor:
        return self.delegate.derivative(points)

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor:
        if measure is MassMeasure.REFERENCE_MEASURE:
            return tf.identity(self.reference_mass)
        if measure is MassMeasure.REFERENCE_LEBESGUE:
            return tf.identity(self.lebesgue_mass)
        raise TypeError("measure must be a MassMeasure")

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor:
        if measure is MassMeasure.REFERENCE_MEASURE:
            return tf.identity(self.reference_integral)
        if measure is MassMeasure.REFERENCE_LEBESGUE:
            return tf.identity(self.lebesgue_integral)
        raise TypeError("measure must be a MassMeasure")

    def manifest_payload(self) -> Mapping[str, object]:
        payload = dict(self.delegate.manifest_payload())
        payload.update(
            {
                "lane_b_mass_precompute_id": MASS_PRECOMPUTE_ID,
                "reference_mass_sha256": tensor_sha256(self.reference_mass),
                "reference_integral_sha256": tensor_sha256(self.reference_integral),
                "lebesgue_mass_sha256": tensor_sha256(self.lebesgue_mass),
                "lebesgue_integral_sha256": tensor_sha256(self.lebesgue_integral),
            }
        )
        return payload


def _freeze_basis_contractions(basis: object) -> _LaneBFrozenMassBasis:
    with tf.device("/CPU:0"):
        reference_mass = tf.identity(basis.mass_matrix(MassMeasure.REFERENCE_MEASURE))
        reference_integral = tf.identity(
            basis.integral_vector(MassMeasure.REFERENCE_MEASURE)
        )
        lebesgue_mass = tf.identity(basis.mass_matrix(MassMeasure.REFERENCE_LEBESGUE))
        lebesgue_integral = tf.identity(
            basis.integral_vector(MassMeasure.REFERENCE_LEBESGUE)
        )
    return _LaneBFrozenMassBasis(
        delegate=basis,
        reference_mass=reference_mass,
        reference_integral=reference_integral,
        lebesgue_mass=lebesgue_mass,
        lebesgue_integral=lebesgue_integral,
    )


def lane_b_product_basis(*, order: int, num_elems: int) -> ProductBasis:
    raw = p85_author_sir_lagrangep_algebraic_product_basis_spec(
        dimension=SIR_JOINT_DIM,
        convention=lane_b_measure_convention(),
        order=int(order),
        num_elems=int(num_elems),
    ).build_product_basis()
    return ProductBasis(
        tuple(_freeze_basis_contractions(basis) for basis in raw.bases),
        raw.convention,
    )


@dataclass(frozen=True)
class LaneBT1Settings:
    """All setup-static hypotheses owned by one T1 tuning arm."""

    arm_id: str
    rank: int
    basis_order: int
    basis_num_elems: int
    learning_rate: float
    l1_weight: float
    l2_weight: float
    batch_size: int
    train_steps: int
    expansion_factor: float
    covariance_jitter: float
    quantile_fraction: float
    use_quantile_scale: bool
    tau: float
    gradient_clip_norm: float
    cdf_grid_size: int
    cdf_bisection_steps: int
    cdf_max_working_bytes: int

    def __post_init__(self) -> None:
        if not str(self.arm_id):
            raise ValueError("arm_id must be nonempty")
        positive_ints = (
            "rank",
            "basis_order",
            "basis_num_elems",
            "batch_size",
            "train_steps",
            "cdf_grid_size",
            "cdf_bisection_steps",
            "cdf_max_working_bytes",
        )
        if any(int(getattr(self, name)) <= 0 for name in positive_ints):
            raise ValueError("integer Lane-B settings must be positive")
        if int(self.batch_size) <= 1:
            raise ValueError("Lane-B training requires batch_size greater than one")
        positive = (
            "learning_rate",
            "expansion_factor",
            "covariance_jitter",
            "quantile_fraction",
            "tau",
            "gradient_clip_norm",
        )
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive
        ):
            raise ValueError("positive Lane-B settings must be finite")
        if not 0.0 < float(self.quantile_fraction) < 0.5:
            raise ValueError("quantile_fraction must lie in (0,0.5)")
        for name in ("l1_weight", "l2_weight"):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) < 0.0:
                raise ValueError("regularization weights must be finite and nonnegative")

    def ranks(self) -> tuple[int, ...]:
        return (1, *([int(self.rank)] * (SIR_JOINT_DIM - 1)), 1)

    def cdf_config(self) -> KRCDFConfig:
        return KRCDFConfig(
            grid_size=int(self.cdf_grid_size),
            bisection_steps=int(self.cdf_bisection_steps),
            monotonicity_tolerance=1e-10,
            bracket_tolerance=1e-8,
            denominator_floor=1e-14,
            max_floor_count=0,
            max_batch_working_bytes=int(self.cdf_max_working_bytes),
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "arm_id": self.arm_id,
            "rank": int(self.rank),
            "basis_order": int(self.basis_order),
            "basis_num_elems": int(self.basis_num_elems),
            "learning_rate": float(self.learning_rate),
            "l1_weight": float(self.l1_weight),
            "l2_weight": float(self.l2_weight),
            "batch_size": int(self.batch_size),
            "train_steps": int(self.train_steps),
            "expansion_factor": float(self.expansion_factor),
            "covariance_jitter": float(self.covariance_jitter),
            "quantile_fraction": float(self.quantile_fraction),
            "use_quantile_scale": bool(self.use_quantile_scale),
            "tau": float(self.tau),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "cdf_grid_size": int(self.cdf_grid_size),
            "cdf_bisection_steps": int(self.cdf_bisection_steps),
            "cdf_max_working_bytes": int(self.cdf_max_working_bytes),
        }


@dataclass(frozen=True)
class LaneBLogNormalizerEstimate:
    """Stable iid Monte Carlo estimate of c + log p(y1)."""

    role: str
    seed: int
    sample_count: int
    shift_constant: tf.Tensor
    log_evidence: tf.Tensor
    log_shifted_normalizer: tf.Tensor
    log_standard_error: tf.Tensor
    log_likelihood_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "shift_constant",
            "log_evidence",
            "log_shifted_normalizer",
            "log_standard_error",
        ):
            value = tf.reshape(tf.convert_to_tensor(getattr(self, name), DTYPE), [])
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
            object.__setattr__(self, name, value)
        if int(self.sample_count) < 2:
            raise ValueError("normalizer estimate requires at least two samples")
        if bool((self.log_standard_error < 0.0).numpy()):
            raise ValueError("log_standard_error must be nonnegative")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "role": str(self.role),
            "seed": self.seed,
            "sample_count": self.sample_count,
            "shift_constant": float(self.shift_constant.numpy()),
            "log_evidence": float(self.log_evidence.numpy()),
            "log_shifted_normalizer": float(self.log_shifted_normalizer.numpy()),
            "log_standard_error": float(self.log_standard_error.numpy()),
            "log_likelihood_sha256": str(self.log_likelihood_sha256),
            "uncertainty_method": "iid_sample_mean_delta_method_v1",
        }


@dataclass(frozen=True)
class LaneBT1TrainingBatch:
    """Measure-correct training batch evaluated in local algebraic coordinates."""

    points: tf.Tensor
    target_sqrt_values: tf.Tensor
    integration_weights: tf.Tensor
    reference_points: tf.Tensor
    log_target_reference: tf.Tensor
    role: str

    def __post_init__(self) -> None:
        points = tf.convert_to_tensor(self.points, DTYPE)
        targets = tf.convert_to_tensor(self.target_sqrt_values, DTYPE)
        weights = tf.convert_to_tensor(self.integration_weights, DTYPE)
        reference = tf.convert_to_tensor(self.reference_points, DTYPE)
        log_target = tf.convert_to_tensor(self.log_target_reference, DTYPE)
        if points.shape.rank != 2 or points.shape[1] != SIR_JOINT_DIM:
            raise ValueError("training points must have shape [sample,36]")
        expected = (points.shape[0],)
        if targets.shape != expected or weights.shape != expected or log_target.shape != expected:
            raise ValueError("training vectors must match the static sample count")
        if reference.shape != points.shape:
            raise ValueError("reference_points must match points")
        for name, value in (
            ("points", points),
            ("target_sqrt_values", targets),
            ("integration_weights", weights),
            ("reference_points", reference),
            ("log_target_reference", log_target),
        ):
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
        tf.debugging.assert_positive(targets, "target_sqrt_values must be positive")
        tf.debugging.assert_positive(weights, "integration_weights must be positive")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "target_sqrt_values", targets)
        object.__setattr__(self, "integration_weights", weights)
        object.__setattr__(self, "reference_points", reference)
        object.__setattr__(self, "log_target_reference", log_target)

    def objective_batch(self) -> P75ObjectiveBatch:
        return P75ObjectiveBatch(
            points=self.points,
            target_values=self.target_sqrt_values,
            weights=self.integration_weights,
            provenance_label=f"lane_b_{self.role}_measure_correct_v1",
        )


def build_lane_b_frame(
    cloud: LaneBT1ProposalCloud,
    settings: LaneBT1Settings,
) -> SourceRouteCoordinateFrame:
    """Build the deterministic target-weighted author-form affine frame."""

    return source_route_recenter(
        samples=tf.transpose(cloud.joint_points),
        log_weights=cloud.log_likelihood,
        expansion_factor=settings.expansion_factor,
        covariance_jitter=settings.covariance_jitter,
        quantile_fraction=settings.quantile_fraction,
        use_quantile_scale=settings.use_quantile_scale,
    )


def physical_to_local_and_reference(
    joint_points: tf.Tensor,
    frame: SourceRouteCoordinateFrame,
) -> tuple[tf.Tensor, tf.Tensor]:
    physical = tf.convert_to_tensor(joint_points, DTYPE)
    local_columns = tf.linalg.triangular_solve(
        frame.matrix,
        tf.transpose(physical) - frame.mu[:, tf.newaxis],
        lower=True,
    )
    local = tf.transpose(local_columns)
    reference = AlgebraicMap(1.0).to_reference(local)
    return local, reference


def t1_log_density_relative_to_reference_measure(
    joint_points: tf.Tensor,
    frame: SourceRouteCoordinateFrame,
) -> Mapping[str, tf.Tensor]:
    """Convert the physical joint density to density relative to dnu=2^-36 du."""

    local, reference = physical_to_local_and_reference(joint_points, frame)
    log_physical = t1_joint_log_density(joint_points)
    log_affine_jacobian = tf.ones_like(log_physical) * frame.log_abs_det()
    log_algebraic_jacobian = tf.reduce_sum(
        AlgebraicMap(1.0).reference_to_domain_log_density(reference),
        axis=1,
    )
    log_inverse_reference_density = tf.ones_like(log_physical) * tf.constant(
        LOG_REFERENCE_DENSITY_CONSTANT,
        DTYPE,
    )
    total = (
        log_physical
        + log_affine_jacobian
        + log_algebraic_jacobian
        + log_inverse_reference_density
    )
    return {
        "local_points": local,
        "reference_points": reference,
        "log_physical_density": log_physical,
        "log_affine_jacobian": log_affine_jacobian,
        "log_algebraic_jacobian": log_algebraic_jacobian,
        "log_inverse_reference_density": log_inverse_reference_density,
        "log_density_relative_to_reference_measure": total,
    }


def select_shift_constant(
    cloud: LaneBT1ProposalCloud,
    frame: SourceRouteCoordinateFrame,
) -> tf.Tensor:
    """Choose c=-log p_hat(y1), giving calibration shifted mass exactly one."""

    del frame
    zero_shift = estimate_shifted_log_normalizer(cloud, tf.constant(0.0, DTYPE))
    return -zero_shift.log_evidence


def build_training_batch(
    cloud: LaneBT1ProposalCloud,
    frame: SourceRouteCoordinateFrame,
    shift_constant: tf.Tensor,
) -> LaneBT1TrainingBatch:
    """Build importance weights for integration under the uniform probability measure."""

    shift = tf.reshape(tf.convert_to_tensor(shift_constant, DTYPE), [])
    terms = t1_log_density_relative_to_reference_measure(cloud.joint_points, frame)
    points = terms["local_points"]
    reference = terms["reference_points"]
    log_target = terms["log_density_relative_to_reference_measure"] + shift
    z1 = cloud.joint_points[:, :18]
    z0 = cloud.joint_points[:, 18:]
    from bayesfilter.highdim.sir_latent_preclip_tf import (  # local to keep closure explicit
        latent_preclip_zhao_cui_sir_austria_model,
    )

    model = latent_preclip_zhao_cui_sir_austria_model()
    theta = tf.zeros([3], DTYPE)
    log_proposal_physical = model.initial_log_density(
        theta, z0
    ) + model.transition_log_density(theta, z0, z1, 1)
    log_coordinate_jacobian = (
        terms["log_affine_jacobian"] + terms["log_algebraic_jacobian"]
    )
    log_integration_weight = (
        -tf.constant(LOG_REFERENCE_DENSITY_CONSTANT, DTYPE)
        - log_proposal_physical
        - log_coordinate_jacobian
    )
    # A common scale cancels from the training-base normalized quadrature weights.
    log_integration_weight -= tf.reduce_max(log_integration_weight)
    return LaneBT1TrainingBatch(
        points=points,
        target_sqrt_values=tf.exp(0.5 * log_target),
        integration_weights=tf.exp(log_integration_weight),
        reference_points=reference,
        log_target_reference=log_target,
        role=cloud.role,
    )


def estimate_shifted_log_normalizer(
    cloud: LaneBT1ProposalCloud,
    shift_constant: tf.Tensor,
) -> LaneBLogNormalizerEstimate:
    log_likelihood = cloud.log_likelihood
    maximum = tf.reduce_max(log_likelihood)
    scaled = tf.exp(log_likelihood - maximum)
    mean_scaled = tf.reduce_mean(scaled)
    centered = scaled - mean_scaled
    variance_scaled = tf.reduce_sum(tf.square(centered)) / tf.cast(
        cloud.sample_count - 1, DTYPE
    )
    standard_error_scaled = tf.sqrt(
        variance_scaled / tf.cast(cloud.sample_count, DTYPE)
    )
    log_evidence = maximum + tf.math.log(mean_scaled)
    log_standard_error = standard_error_scaled / mean_scaled
    shift = tf.reshape(tf.convert_to_tensor(shift_constant, DTYPE), [])
    return LaneBLogNormalizerEstimate(
        role=cloud.role,
        seed=cloud.seed,
        sample_count=cloud.sample_count,
        shift_constant=shift,
        log_evidence=log_evidence,
        log_shifted_normalizer=shift + log_evidence,
        log_standard_error=log_standard_error,
        log_likelihood_sha256=tensor_sha256(log_likelihood),
    )


def normalizer_estimates_agree(
    left: LaneBLogNormalizerEstimate,
    right: LaneBLogNormalizerEstimate,
    *,
    sigma: float = 3.0,
    absolute_floor: float = 1e-6,
) -> tf.Tensor:
    difference = tf.abs(left.log_shifted_normalizer - right.log_shifted_normalizer)
    uncertainty = float(sigma) * tf.sqrt(
        tf.square(left.log_standard_error) + tf.square(right.log_standard_error)
    ) + tf.constant(float(absolute_floor), DTYPE)
    return difference <= uncertainty


def trainer_config(settings: LaneBT1Settings) -> P75TrainableTTConfig:
    return P75TrainableTTConfig(
        product_basis=lane_b_product_basis(
            order=settings.basis_order,
            num_elems=settings.basis_num_elems,
        ),
        ranks=settings.ranks(),
        tau=tf.constant(settings.tau, DTYPE),
        normalizer_floor=tf.constant(1e-14, DTYPE),
        denominator_floor=tf.constant(1e-300, DTYPE),
        l1_weight=tf.constant(settings.l1_weight, DTYPE),
        l2_weight=tf.constant(settings.l2_weight, DTYPE),
        logz_anchor_weight=tf.constant(0.0, DTYPE),
        learning_rate=settings.learning_rate,
        gradient_clip_norm=settings.gradient_clip_norm,
        seed=73001,
        metadata={
            "baseline_id": BASELINE_ID,
            "training_kernel_id": TRAINING_KERNEL_ID,
            "classification": "extension_or_invention",
        },
    )


def balanced_initial_cores(
    settings: LaneBT1Settings,
    product_basis: ProductBasis,
    *,
    seeded_channel_epsilon: float = 1e-6,
) -> tuple[tf.Tensor, ...]:
    """Return an O(1) constant path plus small connected rank channels."""

    if product_basis.dimension != SIR_JOINT_DIM:
        raise ValueError("Lane-B initialization requires the 36D product basis")
    if not math.isfinite(float(seeded_channel_epsilon)) or float(
        seeded_channel_epsilon
    ) <= 0.0:
        raise ValueError("seeded_channel_epsilon must be positive and finite")
    ranks = settings.ranks()
    extra_count = max(int(settings.rank) - 1, 0)
    seeded_scale = (
        float(seeded_channel_epsilon) / float(extra_count)
        if extra_count
        else 0.0
    )
    cores = []
    for axis, basis_dim in enumerate(product_basis.basis_dim_tuple()):
        left_rank = int(ranks[axis])
        right_rank = int(ranks[axis + 1])
        values = tf.zeros([left_rank, int(basis_dim), right_rank], DTYPE)
        # Lagrange cardinal coefficients of one are the constant function.
        constant_indices = tf.constant(
            [[0, basis_index, 0] for basis_index in range(int(basis_dim))],
            tf.int64,
        )
        values = tf.tensor_scatter_nd_update(
            values,
            constant_indices,
            tf.ones([int(basis_dim)], DTYPE),
        )
        indices: list[list[int]] = []
        updates: list[float] = []
        for channel in range(1, min(left_rank, right_rank)):
            # Carry the seeded channel as a constant so its amplitude does not
            # become a product of 36 nonconstant cardinal values.
            for basis_index in range(int(basis_dim)):
                indices.append([channel, basis_index, channel])
                updates.append(1.0)
        if axis == 0:
            for channel in range(1, right_rank):
                basis_index = 1 + ((axis + channel - 1) % max(int(basis_dim) - 1, 1))
                indices.append([0, basis_index, channel])
                updates.append(seeded_scale)
        if axis == SIR_JOINT_DIM - 1:
            for channel in range(1, left_rank):
                for basis_index in range(int(basis_dim)):
                    indices.append([channel, basis_index, 0])
                    updates.append(1.0)
        if indices:
            values = tf.tensor_scatter_nd_update(
                values,
                tf.constant(indices, tf.int64),
                tf.constant(updates, DTYPE),
            )
        cores.append(values)
    return tuple(cores)


def make_compiled_train_step(
    trainer: TrainableFunctionalTT,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, ...]]:
    """Compile the exact training-base density objective and one Adam update."""

    if hasattr(optimizer, "build"):
        optimizer.build(trainer.variables)
    config = trainer.config

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_step(
        points: tf.Tensor,
        target_values: tf.Tensor,
        integration_weights: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape() as tape:
            rho = trainer.rho_theta(points)
            normalizer = trainer.normalizer()
            raw_alpha = integration_weights * (
                tf.square(target_values) + config.tau
            )
            alpha = raw_alpha / tf.reduce_sum(raw_alpha)
            cross_entropy = -tf.reduce_sum(alpha * tf.math.log(rho))
            log_normalizer = tf.math.log(normalizer)
            l1 = tf.add_n([tf.reduce_sum(tf.abs(core)) for core in trainer.variables])
            l2 = tf.add_n([tf.reduce_sum(tf.square(core)) for core in trainer.variables])
            regularization = config.l1_weight * l1 + config.l2_weight * l2
            total_loss = cross_entropy + log_normalizer + regularization
        gradients = tape.gradient(total_loss, trainer.variables)
        clipped, gradient_norm = tf.clip_by_global_norm(
            gradients,
            tf.constant(config.gradient_clip_norm, DTYPE),
        )
        optimizer.apply_gradients(zip(clipped, trainer.variables))
        return (
            total_loss,
            cross_entropy,
            log_normalizer,
            regularization,
            gradient_norm,
            tf.reduce_min(rho),
            tf.reduce_max(rho),
        )

    return compiled_step


def calibrate_trainer_normalizer(
    trainer: TrainableFunctionalTT,
    target_log_normalizer: tf.Tensor,
) -> tf.Tensor:
    """Scale h through one core so integral(h^2)+tau equals the target mass."""

    target = tf.exp(tf.reshape(tf.convert_to_tensor(target_log_normalizer, DTYPE), []))
    square_mass = trainer.sqrt_square_normalizer()
    defensive_mass = trainer.defensive_density.normalizer(
        trainer.config.product_basis.convention.mass_measure
    )
    defensive = trainer.config.tau * defensive_mass
    tf.debugging.assert_greater(target, defensive, "normalizer target must exceed tau")
    tf.debugging.assert_positive(square_mass, "square-root TT mass must be positive")
    scale = tf.sqrt((target - defensive) / square_mass)
    trainer.variables[0].assign(trainer.variables[0] * scale)
    tf.debugging.assert_near(
        trainer.normalizer(),
        target,
        atol=tf.constant(1e-12, DTYPE) * (1.0 + tf.abs(target)),
    )
    return scale


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure() -> Mapping[str, str]:
    from bayesfilter.highdim import bases, diagnostics, fixed_branch, models
    from bayesfilter.highdim import sir_latent_preclip_tf, source_route, squared_tt
    from bayesfilter.highdim import stochastic_density_training, transport, tt
    from bayesfilter.highdim import zhao_cui_austria_sir_lane_b_target_tf

    modules = (
        bases,
        diagnostics,
        fixed_branch,
        models,
        sir_latent_preclip_tf,
        source_route,
        squared_tt,
        stochastic_density_training,
        transport,
        tt,
        zhao_cui_austria_sir_lane_b_target_tf,
    )
    paths = [Path(inspect.getfile(module)).resolve() for module in modules]
    paths.append(Path(__file__).resolve())
    root = Path(__file__).resolve().parents[2]
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(set(paths))
    }


def _target_contract_hash() -> str:
    manifest = BranchManifest(
        version="lane_b_target_contract.v1",
        payload=target_manifest(),
    )
    return manifest.sha256().value


@dataclass(frozen=True)
class LaneBT1Artifact:
    """Repository-issued, reloadable identity for one calibrated T1 baseline."""

    settings: LaneBT1Settings
    frame: SourceRouteCoordinateFrame
    cores: tuple[tf.Tensor, ...]
    shift_constant: tf.Tensor
    calibration_estimate: LaneBLogNormalizerEstimate
    validation_estimate: LaneBLogNormalizerEstimate
    frozen_reference_points: tf.Tensor
    training_cloud_manifest: Mapping[str, object]
    validation_cloud_manifest: Mapping[str, object]
    source_hashes: Mapping[str, str]
    identity: BranchIdentity

    def __post_init__(self) -> None:
        cores = tuple(tf.convert_to_tensor(core, DTYPE) for core in self.cores)
        reference = tf.convert_to_tensor(self.frozen_reference_points, DTYPE)
        shift = tf.reshape(tf.convert_to_tensor(self.shift_constant, DTYPE), [])
        if len(cores) != SIR_JOINT_DIM:
            raise ValueError("Lane-B T1 artifact requires 36 cores")
        if reference.shape.rank != 2 or reference.shape[0] != SIR_JOINT_DIM:
            raise ValueError("frozen references must have shape [36,sample]")
        if not bool(tf.reduce_all((reference >= 0.0) & (reference <= 1.0)).numpy()):
            raise ValueError("frozen reference points must lie in [0,1]")
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "frozen_reference_points", reference)
        object.__setattr__(self, "shift_constant", shift)
        expected = issue_lane_b_t1_identity(
            settings=self.settings,
            frame=self.frame,
            cores=cores,
            shift_constant=shift,
            calibration_estimate=self.calibration_estimate,
            validation_estimate=self.validation_estimate,
            frozen_reference_points=reference,
            training_cloud_manifest=self.training_cloud_manifest,
            validation_cloud_manifest=self.validation_cloud_manifest,
            source_hashes=self.source_hashes,
        )
        if self.identity != expected:
            raise ValueError("Lane-B artifact identity mismatch")
        if not bool(
            normalizer_estimates_agree(
                self.calibration_estimate,
                self.validation_estimate,
            ).numpy()
        ):
            raise ValueError("Lane-B calibration and validation normalizers disagree")
        target = self.calibration_estimate.log_shifted_normalizer
        tf.debugging.assert_near(
            tf.math.log(self.density().normalizer()),
            target,
            atol=tf.constant(1e-10, DTYPE),
        )

    def density(self) -> SquaredTTDensity:
        product_basis = lane_b_product_basis(
            order=self.settings.basis_order,
            num_elems=self.settings.basis_num_elems,
        )
        ftt = FunctionalTT(
            tuple(TTCore(core) for core in self.cores),
            product_basis,
            lane_b_measure_convention(),
        )
        defensive = TensorProductReferenceDensity(
            product_basis,
            lane_b_measure_convention(),
        )
        tau = tf.constant(self.settings.tau, DTYPE)
        floor = tf.constant(1e-14, DTYPE)
        denominator = tf.constant(1e-300, DTYPE)
        branch = SquaredTTDensity.expected_branch_identity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=denominator,
            measure_convention=lane_b_measure_convention(),
        )
        return SquaredTTDensity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=denominator,
            measure_convention=lane_b_measure_convention(),
            branch_identity=branch,
        )

    def transport(self) -> FixedTTSIRTTransport:
        return FixedTTSIRTTransport(self.density(), self.settings.cdf_config())

    def value(self) -> tf.Tensor:
        return tf.math.log(self.density().normalizer()) - self.shift_constant


def issue_lane_b_t1_identity(
    *,
    settings: LaneBT1Settings,
    frame: SourceRouteCoordinateFrame,
    cores: Sequence[tf.Tensor],
    shift_constant: tf.Tensor,
    calibration_estimate: LaneBLogNormalizerEstimate,
    validation_estimate: LaneBLogNormalizerEstimate,
    frozen_reference_points: tf.Tensor,
    training_cloud_manifest: Mapping[str, object],
    validation_cloud_manifest: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> BranchIdentity:
    """Issue identity from actual arrays and repository-owned fixed semantics."""

    payload = {
        "baseline_id": BASELINE_ID,
        "target_id": LANE_B_TARGET_ID,
        "target_contract_sha256": _target_contract_hash(),
        "reference_log_density_id": REFERENCE_LOG_DENSITY_ID,
        "reference_inverse_density_log_constant": LOG_REFERENCE_DENSITY_CONSTANT,
        "normalizer_calibration_id": NORMALIZER_CALIBRATION_ID,
        "training_kernel_id": TRAINING_KERNEL_ID,
        "initialization_id": INITIALIZATION_ID,
        "mass_precompute_id": MASS_PRECOMPUTE_ID,
        "trainer_initialization_seed": 73001,
        "frozen_reference_seed": 73501,
        "value_definition_id": VALUE_DEFINITION_ID,
        "axis_order": ("z1", "z0"),
        "retained_axes": tuple(range(18)),
        "settings": settings.manifest_payload(),
        "frame": {
            "mu_sha256": tensor_sha256(frame.mu),
            "matrix_sha256": tensor_sha256(frame.matrix),
            "log_abs_det": float(frame.log_abs_det().numpy()),
        },
        "core_sha256": tuple(tensor_sha256(core) for core in cores),
        "shift_constant": float(tf.convert_to_tensor(shift_constant, DTYPE).numpy()),
        "calibration_estimate": calibration_estimate.manifest_payload(),
        "validation_estimate": validation_estimate.manifest_payload(),
        "frozen_reference_sha256": tensor_sha256(frozen_reference_points),
        "training_cloud": dict(training_cloud_manifest),
        "validation_cloud": dict(validation_cloud_manifest),
        "source_closure": dict(source_hashes),
        "classification": "extension_or_invention",
        "whole_route_source_faithful": False,
        "hmc_authorized": False,
    }
    manifest = BranchManifest(version=IDENTITY_SCHEMA, payload=payload)
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())


def make_lane_b_t1_artifact(
    *,
    settings: LaneBT1Settings,
    frame: SourceRouteCoordinateFrame,
    trainer: TrainableFunctionalTT,
    shift_constant: tf.Tensor,
    calibration_estimate: LaneBLogNormalizerEstimate,
    validation_estimate: LaneBLogNormalizerEstimate,
    frozen_reference_points: tf.Tensor,
    training_cloud_manifest: Mapping[str, object],
    validation_cloud_manifest: Mapping[str, object],
) -> LaneBT1Artifact:
    cores = tuple(tf.identity(core) for core in trainer.variables)
    hashes = source_closure()
    identity = issue_lane_b_t1_identity(
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift_constant,
        calibration_estimate=calibration_estimate,
        validation_estimate=validation_estimate,
        frozen_reference_points=frozen_reference_points,
        training_cloud_manifest=training_cloud_manifest,
        validation_cloud_manifest=validation_cloud_manifest,
        source_hashes=hashes,
    )
    return LaneBT1Artifact(
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift_constant,
        calibration_estimate=calibration_estimate,
        validation_estimate=validation_estimate,
        frozen_reference_points=frozen_reference_points,
        training_cloud_manifest=dict(training_cloud_manifest),
        validation_cloud_manifest=dict(validation_cloud_manifest),
        source_hashes=hashes,
        identity=identity,
    )


def _write_tensor(path: Path, value: tf.Tensor) -> Mapping[str, object]:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": tf.convert_to_tensor(value).dtype.name,
        "shape": tf.convert_to_tensor(value).shape.as_list(),
    }


def save_lane_b_t1_artifact(artifact: LaneBT1Artifact, directory: Path) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    tensors: dict[str, Mapping[str, object]] = {}
    tensor_values = {
        "frame_mu": artifact.frame.mu,
        "frame_matrix": artifact.frame.matrix,
        "frozen_reference_points": artifact.frozen_reference_points,
        **{f"core_{axis:02d}": core for axis, core in enumerate(artifact.cores)},
    }
    for name, value in tensor_values.items():
        tensors[name] = _write_tensor(output / f"{name}.tensor", value)
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "identity_sha256": artifact.identity.hash.value,
        "settings": artifact.settings.manifest_payload(),
        "shift_constant": float(artifact.shift_constant.numpy()),
        "calibration_estimate": artifact.calibration_estimate.manifest_payload(),
        "validation_estimate": artifact.validation_estimate.manifest_payload(),
        "training_cloud_manifest": dict(artifact.training_cloud_manifest),
        "validation_cloud_manifest": dict(artifact.validation_cloud_manifest),
        "source_closure": dict(artifact.source_hashes),
        "tensors": tensors,
        "value": float(artifact.value().numpy()),
        "nonclaims": (
            "no exact nonlinear likelihood theorem",
            "no score or HMC readiness",
            "no T2 or T20 admission",
            "no source-faithful assembled-route claim",
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _estimate_from_payload(payload: Mapping[str, object]) -> LaneBLogNormalizerEstimate:
    return LaneBLogNormalizerEstimate(
        role=str(payload["role"]),
        seed=int(payload["seed"]),
        sample_count=int(payload["sample_count"]),
        shift_constant=tf.constant(float(payload["shift_constant"]), DTYPE),
        log_evidence=tf.constant(float(payload["log_evidence"]), DTYPE),
        log_shifted_normalizer=tf.constant(
            float(payload["log_shifted_normalizer"]), DTYPE
        ),
        log_standard_error=tf.constant(float(payload["log_standard_error"]), DTYPE),
        log_likelihood_sha256=str(payload["log_likelihood_sha256"]),
    )


def _settings_from_payload(payload: Mapping[str, object]) -> LaneBT1Settings:
    return LaneBT1Settings(**{name: payload[name] for name in LaneBT1Settings.__dataclass_fields__})


def load_lane_b_t1_artifact(directory: Path) -> LaneBT1Artifact:
    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text())
    if payload.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("Lane-B artifact schema mismatch")
    if payload.get("source_closure") != dict(source_closure()):
        raise ValueError("Lane-B artifact source closure is stale")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("Lane-B artifact tensor ledger missing")

    def read_tensor(name: str) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Lane-B artifact tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError(f"Lane-B artifact tensor hash mismatch: {name}")
        dtype = tf.dtypes.as_dtype(str(row["dtype"]))
        value = tf.io.parse_tensor(serialized, out_type=dtype)
        return tf.ensure_shape(value, row["shape"])

    settings = _settings_from_payload(payload["settings"])
    frame = SourceRouteCoordinateFrame(
        mu=read_tensor("frame_mu"),
        matrix=read_tensor("frame_matrix"),
        expansion_factor=settings.expansion_factor,
    )
    cores = tuple(read_tensor(f"core_{axis:02d}") for axis in range(SIR_JOINT_DIM))
    reference = read_tensor("frozen_reference_points")
    calibration = _estimate_from_payload(payload["calibration_estimate"])
    validation = _estimate_from_payload(payload["validation_estimate"])
    identity = issue_lane_b_t1_identity(
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=tf.constant(float(payload["shift_constant"]), DTYPE),
        calibration_estimate=calibration,
        validation_estimate=validation,
        frozen_reference_points=reference,
        training_cloud_manifest=payload["training_cloud_manifest"],
        validation_cloud_manifest=payload["validation_cloud_manifest"],
        source_hashes=payload["source_closure"],
    )
    if identity.hash.value != payload.get("identity_sha256"):
        raise ValueError("Lane-B artifact manifest identity mismatch")
    return LaneBT1Artifact(
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=tf.constant(float(payload["shift_constant"]), DTYPE),
        calibration_estimate=calibration,
        validation_estimate=validation,
        frozen_reference_points=reference,
        training_cloud_manifest=payload["training_cloud_manifest"],
        validation_cloud_manifest=payload["validation_cloud_manifest"],
        source_hashes=payload["source_closure"],
        identity=identity,
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "BASELINE_ID",
    "LOG_REFERENCE_DENSITY_CONSTANT",
    "LaneBLogNormalizerEstimate",
    "LaneBT1Artifact",
    "LaneBT1Settings",
    "LaneBT1TrainingBatch",
    "build_lane_b_frame",
    "build_training_batch",
    "calibrate_trainer_normalizer",
    "estimate_shifted_log_normalizer",
    "issue_lane_b_t1_identity",
    "lane_b_product_basis",
    "load_lane_b_t1_artifact",
    "make_compiled_train_step",
    "make_lane_b_t1_artifact",
    "normalizer_estimates_agree",
    "physical_to_local_and_reference",
    "save_lane_b_t1_artifact",
    "select_shift_constant",
    "source_closure",
    "t1_log_density_relative_to_reference_measure",
    "trainer_config",
]
