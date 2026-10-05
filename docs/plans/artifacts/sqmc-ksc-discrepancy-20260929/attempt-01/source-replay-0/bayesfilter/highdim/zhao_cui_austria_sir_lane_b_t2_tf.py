"""Scope-specific T2 training-base program for the Lane-B Austria SIR route."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import AlgebraicMap
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame, source_route_recenter
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.squared_tt import SquaredTTDensity, TensorProductReferenceDensity
from bayesfilter.highdim.stochastic_density_training import (
    P75TrainableTTConfig,
    TrainableFunctionalTT,
)
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_sampler_tf import (
    LaneBRetainedGridSampler,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    LANE_B_TARGET_ID,
    SIR_JOINT_DIM,
    generate_sealed_lane_b_dataset,
    tensor_sha256,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_boundary_tf import (
    LaneBT1RetainedBoundary,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LOG_REFERENCE_DENSITY_CONSTANT,
    LaneBT1Artifact,
    LaneBT1Settings,
    LaneBT1TrainingBatch,
    calibrate_trainer_normalizer,
    lane_b_measure_convention,
    lane_b_product_basis,
    normalizer_estimates_agree,
    source_closure as t1_source_closure,
)


DTYPE = tf.float64
T2_BASELINE_ID = "zhao_cui_austria_sir_fixed_variant_training_base_t2_v1"
T2_TARGET_ID = "zhao_cui_austria_sir_latent_preclip_t2_previous_tt_transition_observation_v1"
T2_ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t2.v1"
T2_IDENTITY_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t2_identity.v1"
T2_VALUE_DEFINITION_ID = "parent_value_plus_log_t2_tt_normalizer_minus_shift_v1"
B2_ADMISSION_RESULT_PATH = Path(
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-b2-sampler-20260731/"
    "attempt-05-cpu-fresh-replay/result.json"
)
B2_ADMISSION_RESULT_SHA256 = (
    "1649792818cdcf2e7621ea9497dce26cc557fac3582dc6e568133ef94a2906df"
)


@dataclass(frozen=True)
class LaneBT2ProposalCloud:
    """Samples from q_grid(z1) f(z2|z1), ordered as [z2,z1]."""

    joint_points: tf.Tensor
    previous_log_target: tf.Tensor
    previous_log_proposal: tf.Tensor
    previous_correction: tf.Tensor
    transition_log_density: tf.Tensor
    log_likelihood: tf.Tensor
    reference_uniforms: tf.Tensor
    reference_seed: int
    transition_seed: int
    role: str

    def __post_init__(self) -> None:
        points = tf.convert_to_tensor(self.joint_points, DTYPE)
        reference = tf.convert_to_tensor(self.reference_uniforms, DTYPE)
        if points.shape.rank != 2 or points.shape[1] != SIR_JOINT_DIM:
            raise ValueError("T2 joint_points must have shape [sample,36]")
        sample_count = int(points.shape[0])
        if reference.shape != (18, sample_count):
            raise ValueError("T2 reference_uniforms must have shape [18,sample]")
        vectors = {}
        for name in (
            "previous_log_target",
            "previous_log_proposal",
            "previous_correction",
            "transition_log_density",
            "log_likelihood",
        ):
            value = tf.convert_to_tensor(getattr(self, name), DTYPE)
            if value.shape != (sample_count,):
                raise ValueError(f"{name} must match T2 sample count")
            if name == "log_likelihood":
                invalid = tf.math.is_nan(value) | tf.math.is_inf(value) & (value > 0.0)
                if bool(tf.reduce_any(invalid).numpy()):
                    raise ValueError("log_likelihood contains NaN or positive infinity")
            else:
                tf.debugging.assert_all_finite(value, f"{name} must be finite")
            vectors[name] = value
        tf.debugging.assert_all_finite(points, "T2 joint_points must be finite")
        tf.debugging.assert_near(
            vectors["previous_correction"],
            vectors["previous_log_target"] - vectors["previous_log_proposal"],
            atol=2e-12,
        )
        if not str(self.role):
            raise ValueError("T2 cloud role must be nonempty")
        object.__setattr__(self, "joint_points", points)
        object.__setattr__(self, "reference_uniforms", reference)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reference_seed", int(self.reference_seed))
        object.__setattr__(self, "transition_seed", int(self.transition_seed))
        object.__setattr__(self, "role", str(self.role))

    @property
    def sample_count(self) -> int:
        return int(self.joint_points.shape[0])

    @property
    def log_proposal_physical(self) -> tf.Tensor:
        return self.previous_log_proposal + self.transition_log_density

    @property
    def log_target_physical(self) -> tf.Tensor:
        return (
            self.previous_log_target
            + self.transition_log_density
            + self.log_likelihood
        )

    @property
    def log_importance_weight(self) -> tf.Tensor:
        return self.previous_correction + self.log_likelihood

    @property
    def zero_target_mask(self) -> tf.Tensor:
        return tf.math.is_inf(self.log_likelihood) & (self.log_likelihood < 0.0)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "role": self.role,
            "sample_count": self.sample_count,
            "reference_seed": self.reference_seed,
            "transition_seed": self.transition_seed,
            "joint_axis_order": ("z2", "z1"),
            "proposal_law": "q_grid_T1(z1) f(z2|z1,theta=0)",
            "target_law": "p1_TT(z1) f(z2|z1,theta=0) g(y2|z2,theta=0)",
            "joint_points_sha256": tensor_sha256(self.joint_points),
            "reference_uniforms_sha256": tensor_sha256(self.reference_uniforms),
            "previous_log_target_sha256": tensor_sha256(self.previous_log_target),
            "previous_log_proposal_sha256": tensor_sha256(
                self.previous_log_proposal
            ),
            "previous_correction_sha256": tensor_sha256(self.previous_correction),
            "transition_log_density_sha256": tensor_sha256(
                self.transition_log_density
            ),
            "log_likelihood_sha256": tensor_sha256(self.log_likelihood),
            "zero_target_mask_sha256": tensor_sha256(self.zero_target_mask),
            "zero_target_count": int(tf.reduce_sum(tf.cast(self.zero_target_mask, tf.int32)).numpy()),
            "log_importance_weight_sha256": tensor_sha256(
                self.log_importance_weight
            ),
        }


def generate_t2_proposal_cloud(
    *,
    t1_artifact: LaneBT1Artifact,
    reference_uniforms: tf.Tensor,
    reference_seed: int,
    transition_seed: int,
    role: str,
) -> LaneBT2ProposalCloud:
    """Generate one batch-native T2 cloud from the admitted finite sampler."""

    reference = tf.convert_to_tensor(reference_uniforms, DTYPE)
    retained = LaneBRetainedGridSampler(t1_artifact).inverse(reference)
    sample_count = int(reference.shape[1])
    model = latent_preclip_zhao_cui_sir_austria_model()
    _states, observations, _all = generate_sealed_lane_b_dataset()
    theta = tf.zeros([3], DTYPE)
    noise = tf.random.stateless_normal(
        [sample_count, 18],
        seed=tf.constant([int(transition_seed), 2], tf.int32),
        dtype=DTYPE,
    )
    z1 = retained.physical_points
    z2 = model.transition_push_from_standard_normal(theta, z1, noise, 2)
    transition = model.transition_log_density(theta, z1, z2, 2)
    likelihood = model.observation_log_density(theta, z2, observations[1], 2)
    return LaneBT2ProposalCloud(
        joint_points=tf.concat([z2, z1], axis=1),
        previous_log_target=retained.target_log_density,
        previous_log_proposal=retained.proposal_log_density,
        previous_correction=retained.correction_log_weights,
        transition_log_density=transition,
        log_likelihood=likelihood,
        reference_uniforms=reference,
        reference_seed=reference_seed,
        transition_seed=transition_seed,
        role=role,
    )


@dataclass(frozen=True)
class LaneBT2LogNormalizerEstimate:
    role: str
    reference_seed: int
    transition_seed: int
    sample_count: int
    shift_constant: tf.Tensor
    log_increment: tf.Tensor
    log_shifted_normalizer: tf.Tensor
    log_standard_error: tf.Tensor
    log_importance_weight_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "shift_constant",
            "log_increment",
            "log_shifted_normalizer",
            "log_standard_error",
        ):
            value = tf.reshape(tf.convert_to_tensor(getattr(self, name), DTYPE), [])
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
            object.__setattr__(self, name, value)
        if int(self.sample_count) < 2:
            raise ValueError("T2 normalizer estimate requires at least two samples")
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "role": self.role,
            "reference_seed": self.reference_seed,
            "transition_seed": self.transition_seed,
            "sample_count": self.sample_count,
            "shift_constant": self.shift_constant,
            "log_increment": self.log_increment,
            "log_shifted_normalizer": self.log_shifted_normalizer,
            "log_standard_error": self.log_standard_error,
            "log_importance_weight_sha256": self.log_importance_weight_sha256,
            "uncertainty_method": "iid_sample_mean_delta_method_v1",
        }


def estimate_t2_shifted_log_normalizer(
    cloud: LaneBT2ProposalCloud, shift_constant: tf.Tensor
) -> LaneBT2LogNormalizerEstimate:
    log_weight = cloud.log_importance_weight
    invalid = tf.math.is_nan(log_weight) | tf.math.is_inf(log_weight) & (log_weight > 0.0)
    if bool(tf.reduce_any(invalid).numpy()):
        raise ValueError("T2 importance weight contains NaN or positive infinity")
    if not bool(tf.reduce_any(tf.math.is_finite(log_weight)).numpy()):
        raise ValueError("T2 importance weights have no finite support")
    maximum = tf.reduce_max(log_weight)
    scaled = tf.exp(log_weight - maximum)
    mean_scaled = tf.reduce_mean(scaled)
    variance_scaled = tf.reduce_sum(tf.square(scaled - mean_scaled)) / tf.cast(
        cloud.sample_count - 1, DTYPE
    )
    standard_error_scaled = tf.sqrt(
        variance_scaled / tf.cast(cloud.sample_count, DTYPE)
    )
    log_increment = maximum + tf.math.log(mean_scaled)
    log_standard_error = standard_error_scaled / mean_scaled
    shift = tf.reshape(tf.convert_to_tensor(shift_constant, DTYPE), [])
    return LaneBT2LogNormalizerEstimate(
        role=cloud.role,
        reference_seed=cloud.reference_seed,
        transition_seed=cloud.transition_seed,
        sample_count=cloud.sample_count,
        shift_constant=shift,
        log_increment=log_increment,
        log_shifted_normalizer=shift + log_increment,
        log_standard_error=log_standard_error,
        log_importance_weight_sha256=tensor_sha256(log_weight),
    )


def select_t2_shift_constant(cloud: LaneBT2ProposalCloud) -> tf.Tensor:
    zero = estimate_t2_shifted_log_normalizer(cloud, tf.constant(0.0, DTYPE))
    return -zero.log_increment


def build_t2_frame(
    cloud: LaneBT2ProposalCloud, settings: LaneBT1Settings
) -> SourceRouteCoordinateFrame:
    return source_route_recenter(
        samples=tf.transpose(cloud.joint_points),
        log_weights=cloud.log_importance_weight,
        expansion_factor=settings.expansion_factor,
        covariance_jitter=settings.covariance_jitter,
        quantile_fraction=settings.quantile_fraction,
        use_quantile_scale=settings.use_quantile_scale,
    )


def t2_log_density_relative_to_reference_measure(
    cloud: LaneBT2ProposalCloud, frame: SourceRouteCoordinateFrame
) -> Mapping[str, tf.Tensor]:
    physical = cloud.joint_points
    local = tf.transpose(
        tf.linalg.triangular_solve(
            frame.matrix,
            tf.transpose(physical) - frame.mu[:, tf.newaxis],
            lower=True,
        )
    )
    reference = AlgebraicMap(1.0).to_reference(local)
    log_affine = tf.ones([cloud.sample_count], DTYPE) * frame.log_abs_det()
    log_algebraic = tf.reduce_sum(
        AlgebraicMap(1.0).reference_to_domain_log_density(reference), axis=1
    )
    log_inverse_reference = tf.ones([cloud.sample_count], DTYPE) * tf.constant(
        LOG_REFERENCE_DENSITY_CONSTANT, DTYPE
    )
    total = (
        cloud.log_target_physical
        + log_affine
        + log_algebraic
        + log_inverse_reference
    )
    return {
        "local_points": local,
        "reference_points": reference,
        "log_physical_density": cloud.log_target_physical,
        "log_affine_jacobian": log_affine,
        "log_algebraic_jacobian": log_algebraic,
        "log_inverse_reference_density": log_inverse_reference,
        "log_density_relative_to_reference_measure": total,
    }


def build_t2_training_batch(
    cloud: LaneBT2ProposalCloud,
    frame: SourceRouteCoordinateFrame,
    shift_constant: tf.Tensor,
) -> "LaneBT2LogTrainingBatch":
    shift = tf.reshape(tf.convert_to_tensor(shift_constant, DTYPE), [])
    terms = t2_log_density_relative_to_reference_measure(cloud, frame)
    log_target = terms["log_density_relative_to_reference_measure"] + shift
    return LaneBT2LogTrainingBatch(
        points=terms["local_points"],
        reference_points=terms["reference_points"],
        log_target_reference=log_target,
        log_importance_weight=cloud.log_importance_weight,
        role=f"t2_{cloud.role}",
    )


@dataclass(frozen=True)
class LaneBT2LogTrainingBatch:
    """T2 empirical target represented without exponentiating tail densities."""

    points: tf.Tensor
    reference_points: tf.Tensor
    log_target_reference: tf.Tensor
    log_importance_weight: tf.Tensor
    role: str

    def __post_init__(self) -> None:
        points = tf.convert_to_tensor(self.points, DTYPE)
        reference = tf.convert_to_tensor(self.reference_points, DTYPE)
        log_target = tf.convert_to_tensor(self.log_target_reference, DTYPE)
        log_weight = tf.convert_to_tensor(self.log_importance_weight, DTYPE)
        if points.shape.rank != 2 or points.shape[1] != SIR_JOINT_DIM:
            raise ValueError("T2 log batch points must have shape [sample,36]")
        sample_count = int(points.shape[0])
        if reference.shape != points.shape:
            raise ValueError("T2 reference points must match local points")
        if log_target.shape != (sample_count,) or log_weight.shape != (sample_count,):
            raise ValueError("T2 log batch vectors must match sample count")
        for name, value in (
            ("points", points),
            ("reference_points", reference),
            ("log_target_reference", log_target),
            ("log_importance_weight", log_weight),
        ):
            if name == "log_target_reference":
                invalid = tf.math.is_nan(value) | tf.math.is_inf(value) & (value > 0.0)
                if bool(tf.reduce_any(invalid).numpy()):
                    raise ValueError("T2 log target contains NaN or positive infinity")
            elif name == "log_importance_weight":
                invalid = tf.math.is_nan(value) | tf.math.is_inf(value) & (value > 0.0)
                if bool(tf.reduce_any(invalid).numpy()):
                    raise ValueError("T2 log weight contains NaN or positive infinity")
            else:
                tf.debugging.assert_all_finite(value, f"{name} must be finite")
        if not str(self.role):
            raise ValueError("T2 log batch role must be nonempty")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "reference_points", reference)
        object.__setattr__(self, "log_target_reference", log_target)
        object.__setattr__(self, "log_importance_weight", log_weight)
        object.__setattr__(self, "role", str(self.role))


def make_t2_compiled_train_step(
    trainer: TrainableFunctionalTT,
    optimizer: tf.keras.optimizers.Optimizer,
    *,
    microbatch_size: int,
) -> Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, ...]]:
    """Build one exact full-cloud update from fixed-parameter microbatches.

    Each compiled microbatch kernel evaluates a scaled contribution while the
    parameters remain fixed. Raw gradients are averaged over the complete
    deterministic cycle, clipped once, and only then passed to the optimizer.
    """

    if hasattr(optimizer, "build"):
        optimizer.build(trainer.variables)
    config = trainer.config
    batch_size = int(microbatch_size)
    if batch_size <= 1:
        raise ValueError("T2 training microbatch size must exceed one")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_microbatch_gradients(
        points: tf.Tensor,
        log_importance_weight: tf.Tensor,
        global_log_weight_sum: tf.Tensor,
        microbatch_scale: tf.Tensor,
    ) -> tuple[tuple[tf.Tensor, ...], tuple[tf.Tensor, ...]]:
        with tf.GradientTape() as tape:
            rho = trainer.rho_theta(points)
            normalizer = trainer.normalizer()
            alpha = tf.exp(log_importance_weight - global_log_weight_sum)
            cross_entropy = -microbatch_scale * tf.reduce_sum(
                alpha * tf.math.log(rho)
            )
            log_normalizer = tf.math.log(normalizer)
            l1 = tf.add_n(
                [tf.reduce_sum(tf.abs(core)) for core in trainer.variables]
            )
            l2 = tf.add_n(
                [tf.reduce_sum(tf.square(core)) for core in trainer.variables]
            )
            regularization = config.l1_weight * l1 + config.l2_weight * l2
            total_loss = cross_entropy + log_normalizer + regularization
        gradients = tape.gradient(total_loss, trainer.variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("missing T2 training gradient")
        checked = tuple(tf.convert_to_tensor(gradient, DTYPE) for gradient in gradients)
        for gradient in checked:
            tf.debugging.assert_all_finite(gradient, "T2 microbatch gradient")
        return (
            (
                total_loss,
                cross_entropy,
                log_normalizer,
                regularization,
                tf.reduce_min(rho),
                tf.reduce_max(rho),
                tf.reduce_min(alpha),
                tf.reduce_max(alpha),
            ),
            checked,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_apply_gradients(
        gradients: tuple[tf.Tensor, ...],
    ) -> tf.Tensor:
        clipped, gradient_norm = tf.clip_by_global_norm(
            gradients, tf.constant(config.gradient_clip_norm, DTYPE)
        )
        for gradient in clipped:
            tf.debugging.assert_all_finite(gradient, "T2 clipped full-cloud gradient")
        optimizer.apply_gradients(zip(clipped, trainer.variables))
        for variable in trainer.variables:
            tf.debugging.assert_all_finite(variable, "T2 updated core")
        return gradient_norm

    def full_cloud_step(
        points: tf.Tensor,
        log_importance_weight: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        values = tf.convert_to_tensor(points, DTYPE)
        weights = tf.convert_to_tensor(log_importance_weight, DTYPE)
        if values.shape.rank != 2 or values.shape[1] != SIR_JOINT_DIM:
            raise ValueError("T2 training points must have shape [sample,36]")
        sample_count = values.shape[0]
        if sample_count is None or weights.shape != (int(sample_count),):
            raise ValueError("T2 training weights must match a static sample count")
        if int(sample_count) % batch_size:
            raise ValueError("T2 full cloud must divide exactly into microbatches")
        microbatch_count = int(sample_count) // batch_size
        global_log_weight_sum = tf.reduce_logsumexp(weights)
        tf.debugging.assert_all_finite(
            global_log_weight_sum, "T2 global log-weight sum"
        )

        term_rows: list[tuple[tf.Tensor, ...]] = []
        gradient_rows: list[tuple[tf.Tensor, ...]] = []
        scale = tf.constant(float(microbatch_count), DTYPE)
        for first in range(0, int(sample_count), batch_size):
            terms, gradients = compiled_microbatch_gradients(
                values[first : first + batch_size],
                weights[first : first + batch_size],
                global_log_weight_sum,
                scale,
            )
            term_rows.append(terms)
            gradient_rows.append(gradients)

        inverse_count = tf.constant(1.0 / float(microbatch_count), DTYPE)
        accumulated = tuple(
            tf.add_n([row[index] for row in gradient_rows]) * inverse_count
            for index in range(len(trainer.variables))
        )
        gradient_norm = compiled_apply_gradients(accumulated)
        averaged = tuple(
            tf.add_n([row[index] for row in term_rows]) * inverse_count
            for index in range(4)
        )
        return (
            *averaged,
            gradient_norm,
            tf.reduce_min(tf.stack([row[4] for row in term_rows])),
            tf.reduce_max(tf.stack([row[5] for row in term_rows])),
            tf.reduce_min(tf.stack([row[6] for row in term_rows])),
            tf.reduce_max(tf.stack([row[7] for row in term_rows])),
        )

    return full_cloud_step


def t2_log_weight_cross_entropy(
    *,
    log_rho: tf.Tensor,
    log_importance_weight: tf.Tensor,
    global_log_weight_sum: tf.Tensor | None = None,
    scale: tf.Tensor | float = 1.0,
) -> tf.Tensor:
    """Evaluate one full or scaled-microbatch empirical cross-entropy term."""

    log_density = tf.convert_to_tensor(log_rho, DTYPE)
    log_weight = tf.convert_to_tensor(log_importance_weight, DTYPE)
    if log_density.shape != log_weight.shape or log_density.shape.rank != 1:
        raise ValueError("T2 log density and weights must be matching vectors")
    normalizer = (
        tf.reduce_logsumexp(log_weight)
        if global_log_weight_sum is None
        else tf.reshape(tf.convert_to_tensor(global_log_weight_sum, DTYPE), [])
    )
    multiplier = tf.reshape(tf.convert_to_tensor(scale, DTYPE), [])
    return -multiplier * tf.reduce_sum(
        tf.exp(log_weight - normalizer) * log_density
    )


def make_t2_compiled_validation_metric(
    trainer: TrainableFunctionalTT,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, ...]]:
    """Return shape diagnostics under the same T2 empirical target measure."""

    @tf.function(jit_compile=True, reduce_retracing=True)
    def metric(
        points: tf.Tensor,
        log_importance_weight: tf.Tensor,
        log_target_reference: tf.Tensor,
        target_log_normalizer: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        alpha = tf.nn.softmax(log_importance_weight)
        safe_target = tf.where(
            alpha > 0.0, log_target_reference, tf.zeros_like(log_target_reference)
        )
        candidate_log_density = tf.math.log(trainer.rho_theta(points)) - tf.math.log(
            trainer.normalizer()
        )
        target_log_density = safe_target - target_log_normalizer
        candidate_rms = tf.sqrt(
            tf.reduce_sum(alpha * tf.square(candidate_log_density - target_log_density))
        )
        constant_rms = tf.sqrt(
            tf.reduce_sum(alpha * tf.square(target_log_density))
        )
        centered_candidate = candidate_log_density - tf.reduce_sum(
            alpha * candidate_log_density
        )
        centered_target = target_log_density - tf.reduce_sum(
            alpha * target_log_density
        )
        centered_rms = tf.sqrt(
            tf.reduce_sum(alpha * tf.square(centered_candidate - centered_target))
        )
        return (
            candidate_rms,
            constant_rms,
            centered_rms,
            tf.reduce_min(alpha),
            tf.reduce_max(alpha),
        )

    return metric


def t2_trainer_config(settings: LaneBT1Settings) -> P75TrainableTTConfig:
    return P75TrainableTTConfig(
        product_basis=lane_b_product_basis(
            order=settings.basis_order, num_elems=settings.basis_num_elems
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
        seed=73800,
        metadata={
            "baseline_id": T2_BASELINE_ID,
            "target_id": T2_TARGET_ID,
            "classification": "extension_or_invention",
        },
    )


def verify_b2_admission(root: Path) -> str:
    path = root / B2_ADMISSION_RESULT_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != B2_ADMISSION_RESULT_SHA256:
        raise ValueError("B2 admission result hash mismatch")
    import json

    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS_B2_RETAINED_SAMPLER_ADMISSION":
        raise ValueError("B2 sampler is not admitted")
    return digest


def t2_source_closure() -> Mapping[str, str]:
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_lane_b_sampler_tf,
        zhao_cui_austria_sir_lane_b_t2_boundary_tf,
    )

    modules = (
        zhao_cui_austria_sir_lane_b_sampler_tf,
        zhao_cui_austria_sir_lane_b_t2_boundary_tf,
        inspect.getmodule(t2_source_closure),
    )
    result = dict(t1_source_closure())
    for module in modules:
        if module is None or module.__file__ is None:
            raise RuntimeError("T2 source closure module path missing")
        path = Path(module.__file__).resolve()
        result[path.relative_to(Path(__file__).resolve().parents[2]).as_posix()] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return result


def issue_lane_b_t2_identity(
    *,
    parent_artifact: LaneBT1Artifact,
    settings: LaneBT1Settings,
    frame: SourceRouteCoordinateFrame,
    cores: Sequence[tf.Tensor],
    shift_constant: tf.Tensor,
    calibration_estimate: LaneBT2LogNormalizerEstimate,
    validation_estimate: LaneBT2LogNormalizerEstimate,
    training_cloud_manifest: Mapping[str, object],
    validation_cloud_manifest: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> BranchIdentity:
    """Issue non-overridable T2 identity from the actual numerical program."""

    payload = {
        "baseline_id": T2_BASELINE_ID,
        "target_id": T2_TARGET_ID,
        "parent_t1_identity": parent_artifact.identity.hash.value,
        "parent_t1_value": float(parent_artifact.value().numpy()),
        "b2_admission_result_sha256": B2_ADMISSION_RESULT_SHA256,
        "axis_order": ("z2", "z1"),
        "retained_axes": tuple(range(18)),
        "event_order": "retained_z1_then_transition_to_z2_then_observe_sealed_y2",
        "value_definition_id": T2_VALUE_DEFINITION_ID,
        "reference_inverse_density_log_constant": LOG_REFERENCE_DENSITY_CONSTANT,
        "proposal_law": "q_grid_T1(z1) f(z2|z1,theta=0)",
        "proposal_correction": "log_p1_TT_minus_log_q_grid",
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
        "training_cloud": _t2_cloud_identity_payload(training_cloud_manifest),
        "validation_cloud": _t2_cloud_identity_payload(validation_cloud_manifest),
        "source_closure": dict(source_hashes),
        "classification": "extension_or_invention",
        "whole_route_source_faithful": False,
        "hmc_authorized": False,
    }
    manifest = BranchManifest(T2_IDENTITY_SCHEMA, payload)
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())


def _t2_cloud_identity_payload(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    result = dict(payload)
    order = result.get("joint_axis_order")
    if order is not None:
        if tuple(order) != ("z2", "z1"):
            raise ValueError("T2 cloud joint axis order mismatch")
        result["joint_axis_order"] = ("z2", "z1")
    return result


@dataclass(frozen=True)
class LaneBT2Artifact:
    """Reloadable T2 object tied to its admitted T1 parent."""

    parent_artifact: LaneBT1Artifact
    settings: LaneBT1Settings
    frame: SourceRouteCoordinateFrame
    cores: tuple[tf.Tensor, ...]
    shift_constant: tf.Tensor
    calibration_estimate: LaneBT2LogNormalizerEstimate
    validation_estimate: LaneBT2LogNormalizerEstimate
    training_cloud_manifest: Mapping[str, object]
    validation_cloud_manifest: Mapping[str, object]
    source_hashes: Mapping[str, str]
    identity: BranchIdentity

    def __post_init__(self) -> None:
        cores = tuple(tf.convert_to_tensor(core, DTYPE) for core in self.cores)
        shift = tf.reshape(tf.convert_to_tensor(self.shift_constant, DTYPE), [])
        if len(cores) != SIR_JOINT_DIM:
            raise ValueError("Lane-B T2 artifact requires 36 cores")
        expected = issue_lane_b_t2_identity(
            parent_artifact=self.parent_artifact,
            settings=self.settings,
            frame=self.frame,
            cores=cores,
            shift_constant=shift,
            calibration_estimate=self.calibration_estimate,
            validation_estimate=self.validation_estimate,
            training_cloud_manifest=self.training_cloud_manifest,
            validation_cloud_manifest=self.validation_cloud_manifest,
            source_hashes=self.source_hashes,
        )
        if self.identity != expected:
            raise ValueError("Lane-B T2 artifact identity mismatch")
        if not bool(
            normalizer_estimates_agree(
                self.calibration_estimate, self.validation_estimate
            ).numpy()
        ):
            raise ValueError("T2 calibration and validation normalizers disagree")
        tf.debugging.assert_near(
            tf.math.log(self.density().normalizer()),
            self.calibration_estimate.log_shifted_normalizer,
            atol=tf.constant(1e-10, DTYPE),
        )
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "shift_constant", shift)

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
            product_basis, lane_b_measure_convention()
        )
        tau = tf.constant(self.settings.tau, DTYPE)
        normalizer_floor = tf.constant(1e-14, DTYPE)
        denominator_floor = tf.constant(1e-300, DTYPE)
        branch = SquaredTTDensity.expected_branch_identity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=normalizer_floor,
            denominator_floor=denominator_floor,
            measure_convention=lane_b_measure_convention(),
        )
        return SquaredTTDensity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=normalizer_floor,
            denominator_floor=denominator_floor,
            measure_convention=lane_b_measure_convention(),
            branch_identity=branch,
        )

    def increment(self) -> tf.Tensor:
        return tf.math.log(self.density().normalizer()) - self.shift_constant

    def value(self) -> tf.Tensor:
        return self.parent_artifact.value() + self.increment()


def make_lane_b_t2_artifact(
    *,
    parent_artifact: LaneBT1Artifact,
    settings: LaneBT1Settings,
    frame: SourceRouteCoordinateFrame,
    trainer: TrainableFunctionalTT,
    shift_constant: tf.Tensor,
    calibration_estimate: LaneBT2LogNormalizerEstimate,
    validation_estimate: LaneBT2LogNormalizerEstimate,
    training_cloud_manifest: Mapping[str, object],
    validation_cloud_manifest: Mapping[str, object],
) -> LaneBT2Artifact:
    cores = tuple(tf.identity(core) for core in trainer.variables)
    hashes = t2_source_closure()
    identity = issue_lane_b_t2_identity(
        parent_artifact=parent_artifact,
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift_constant,
        calibration_estimate=calibration_estimate,
        validation_estimate=validation_estimate,
        training_cloud_manifest=training_cloud_manifest,
        validation_cloud_manifest=validation_cloud_manifest,
        source_hashes=hashes,
    )
    return LaneBT2Artifact(
        parent_artifact=parent_artifact,
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift_constant,
        calibration_estimate=calibration_estimate,
        validation_estimate=validation_estimate,
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
        "dtype": value.dtype.name,
        "shape": value.shape.as_list(),
    }


def save_lane_b_t2_artifact(artifact: LaneBT2Artifact, directory: Path) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    tensors = {
        "frame_mu": _write_tensor(output / "frame_mu.tensor", artifact.frame.mu),
        "frame_matrix": _write_tensor(
            output / "frame_matrix.tensor", artifact.frame.matrix
        ),
    }
    for axis, core in enumerate(artifact.cores):
        tensors[f"core_{axis:02d}"] = _write_tensor(
            output / f"core_{axis:02d}.tensor", core
        )
    payload = {
        "schema_version": T2_ARTIFACT_SCHEMA,
        "identity_sha256": artifact.identity.hash.value,
        "parent_t1_identity": artifact.parent_artifact.identity.hash.value,
        "settings": artifact.settings.manifest_payload(),
        "shift_constant": float(artifact.shift_constant.numpy()),
        "calibration_estimate": artifact.calibration_estimate.manifest_payload(),
        "validation_estimate": artifact.validation_estimate.manifest_payload(),
        "training_cloud_manifest": dict(artifact.training_cloud_manifest),
        "validation_cloud_manifest": dict(artifact.validation_cloud_manifest),
        "source_closure": dict(artifact.source_hashes),
        "tensors": tensors,
        "increment": float(artifact.increment().numpy()),
        "cumulative_value": float(artifact.value().numpy()),
        "nonclaims": (
            "no exact nonlinear likelihood theorem",
            "no score, T20, HMC, or production KR readiness",
            "no source-faithful assembled-route claim",
        ),
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return path


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            return value.numpy().item()
        return value.numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _t2_estimate_from_payload(
    payload: Mapping[str, object],
) -> LaneBT2LogNormalizerEstimate:
    return LaneBT2LogNormalizerEstimate(
        role=str(payload["role"]),
        reference_seed=int(payload["reference_seed"]),
        transition_seed=int(payload["transition_seed"]),
        sample_count=int(payload["sample_count"]),
        shift_constant=tf.constant(float(payload["shift_constant"]), DTYPE),
        log_increment=tf.constant(float(payload["log_increment"]), DTYPE),
        log_shifted_normalizer=tf.constant(
            float(payload["log_shifted_normalizer"]), DTYPE
        ),
        log_standard_error=tf.constant(
            float(payload["log_standard_error"]), DTYPE
        ),
        log_importance_weight_sha256=str(
            payload["log_importance_weight_sha256"]
        ),
    )


def load_lane_b_t2_artifact(
    directory: Path, *, parent_artifact: LaneBT1Artifact
) -> LaneBT2Artifact:
    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text())
    if payload.get("schema_version") != T2_ARTIFACT_SCHEMA:
        raise ValueError("Lane-B T2 artifact schema mismatch")
    if payload.get("parent_t1_identity") != parent_artifact.identity.hash.value:
        raise ValueError("Lane-B T2 parent identity mismatch")
    if payload.get("source_closure") != dict(t2_source_closure()):
        raise ValueError("Lane-B T2 source closure is stale")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("Lane-B T2 tensor ledger missing")

    def read_tensor(name: str) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Lane-B T2 tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError(f"Lane-B T2 tensor hash mismatch: {name}")
        value = tf.io.parse_tensor(
            serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"]))
        )
        return tf.ensure_shape(value, row["shape"])

    settings = LaneBT1Settings(
        **{
            name: payload["settings"][name]
            for name in LaneBT1Settings.__dataclass_fields__
        }
    )
    frame = SourceRouteCoordinateFrame(
        mu=read_tensor("frame_mu"),
        matrix=read_tensor("frame_matrix"),
        expansion_factor=settings.expansion_factor,
    )
    cores = tuple(read_tensor(f"core_{axis:02d}") for axis in range(SIR_JOINT_DIM))
    calibration = _t2_estimate_from_payload(payload["calibration_estimate"])
    validation = _t2_estimate_from_payload(payload["validation_estimate"])
    shift = tf.constant(float(payload["shift_constant"]), DTYPE)
    identity = issue_lane_b_t2_identity(
        parent_artifact=parent_artifact,
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift,
        calibration_estimate=calibration,
        validation_estimate=validation,
        training_cloud_manifest=payload["training_cloud_manifest"],
        validation_cloud_manifest=payload["validation_cloud_manifest"],
        source_hashes=payload["source_closure"],
    )
    if identity.hash.value != payload.get("identity_sha256"):
        raise ValueError("Lane-B T2 manifest identity mismatch")
    return LaneBT2Artifact(
        parent_artifact=parent_artifact,
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift,
        calibration_estimate=calibration,
        validation_estimate=validation,
        training_cloud_manifest=payload["training_cloud_manifest"],
        validation_cloud_manifest=payload["validation_cloud_manifest"],
        source_hashes=payload["source_closure"],
        identity=identity,
    )


__all__ = [
    "B2_ADMISSION_RESULT_PATH",
    "B2_ADMISSION_RESULT_SHA256",
    "LaneBT2LogTrainingBatch",
    "LaneBT2LogNormalizerEstimate",
    "LaneBT2ProposalCloud",
    "LaneBT2Artifact",
    "T2_BASELINE_ID",
    "T2_TARGET_ID",
    "build_t2_frame",
    "build_t2_training_batch",
    "estimate_t2_shifted_log_normalizer",
    "generate_t2_proposal_cloud",
    "issue_lane_b_t2_identity",
    "load_lane_b_t2_artifact",
    "make_lane_b_t2_artifact",
    "make_t2_compiled_train_step",
    "make_t2_compiled_validation_metric",
    "save_lane_b_t2_artifact",
    "select_t2_shift_constant",
    "t2_log_density_relative_to_reference_measure",
    "t2_log_weight_cross_entropy",
    "t2_source_closure",
    "t2_trainer_config",
    "verify_b2_admission",
]
