"""T1 origin-tangent training for the Zhao-Cui Austria SIR Lane-B child.

The parent squared-TT density is immutable. Only three compact tangent-core
banks are trained, and every score returned by the admitted child is a manual
derivative of that child's finite value program.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    LaneBT1ProposalCloud,
    generate_sealed_lane_b_dataset,
    generate_t1_proposal_cloud,
    tensor_sha256,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LaneBT1Artifact,
    lane_b_product_basis,
    physical_to_local_and_reference,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
    _paired_mass_product,
)


DTYPE = tf.float64
PARAMETER_DIM = 3
T1_SCORE_BATCH_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_batch.v1"
T1_SCORE_TRAINER_ID = (
    "frozen_parent_full_defensive_log_score_tangent_adam_plus_training_fisher_gauge_v2"
)
T1_FISHER_ESTIMATE_ID = "self_normalized_iid_ratio_delta_method_v1"
T1_SCORE_CLASSIFICATION = "extension_or_invention"
T1_SCORE_ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_artifact.v1"
T1_SCORE_IDENTITY_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_identity.v1"
T1_SCORE_PLAN = (
    "docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t1-score-plan-2026-07-31.md"
)
EXPECTED_T1_PARENT_IDENTITY = (
    "e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59"
)


def _require_t1_parent(parent: LaneBT1Artifact) -> None:
    if not isinstance(parent, LaneBT1Artifact):
        raise TypeError("T1 score training requires a LaneBT1Artifact parent")
    if parent.identity.hash.value != EXPECTED_T1_PARENT_IDENTITY:
        raise ValueError("T1 score parent identity mismatch")


def t1_complete_data_parameter_score(
    joint_points: tf.Tensor,
) -> tf.Tensor:
    """Return the analytical score of p0(z0) f(z1|z0) g(y1|z1)."""

    points = tf.convert_to_tensor(joint_points, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != 36:
        raise ValueError("joint_points must have shape [sample,36]")
    model = latent_preclip_zhao_cui_sir_austria_model()
    _states, observations, _all_observations = generate_sealed_lane_b_dataset()
    theta = tf.zeros([PARAMETER_DIM], DTYPE)
    z1 = points[:, :18]
    z0 = points[:, 18:]
    score = (
        model.initial_log_density_parameter_score(theta, z0)
        + model.transition_log_density_parameter_score(theta, z0, z1, 1)
        + model.observation_log_density_parameter_score(
            theta, z1, observations[0], 1
        )
    )
    score = tf.ensure_shape(score, [points.shape[0], PARAMETER_DIM])
    tf.debugging.assert_all_finite(score, "T1 complete-data score")
    return score


def normalized_likelihood_weights(log_likelihood: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(log_likelihood, DTYPE)
    if values.shape.rank != 1:
        raise ValueError("log_likelihood must be a vector")
    maximum = tf.reduce_max(values)
    scaled = tf.exp(values - maximum)
    weights = scaled / tf.reduce_sum(scaled)
    tf.debugging.assert_all_finite(weights, "normalized likelihood weights")
    tf.debugging.assert_near(tf.reduce_sum(weights), tf.constant(1.0, DTYPE))
    return weights


@dataclass(frozen=True)
class LaneBT1ScoreBatch:
    physical_points: tf.Tensor
    local_points: tf.Tensor
    log_likelihood: tf.Tensor
    target_score: tf.Tensor
    target_weights: tf.Tensor
    score_scale: tf.Tensor
    seed: int
    role: str

    def __post_init__(self) -> None:
        physical = tf.convert_to_tensor(self.physical_points, DTYPE)
        local = tf.convert_to_tensor(self.local_points, DTYPE)
        likelihood = tf.convert_to_tensor(self.log_likelihood, DTYPE)
        score = tf.convert_to_tensor(self.target_score, DTYPE)
        weights = tf.convert_to_tensor(self.target_weights, DTYPE)
        scale = tf.reshape(tf.convert_to_tensor(self.score_scale, DTYPE), [PARAMETER_DIM])
        if physical.shape.rank != 2 or physical.shape[1] != 36:
            raise ValueError("score batch physical points must have shape [sample,36]")
        if local.shape != physical.shape:
            raise ValueError("score batch local points must match physical points")
        sample_count = physical.shape[0]
        if sample_count is None or int(sample_count) < 2:
            raise ValueError("score batch requires a static sample count of at least two")
        if likelihood.shape != (sample_count,) or weights.shape != (sample_count,):
            raise ValueError("score batch weights must match sample count")
        if score.shape != (sample_count, PARAMETER_DIM):
            raise ValueError("score batch target score must have shape [sample,3]")
        for name, value in (
            ("physical_points", physical),
            ("local_points", local),
            ("log_likelihood", likelihood),
            ("target_score", score),
            ("target_weights", weights),
            ("score_scale", scale),
        ):
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
        tf.debugging.assert_positive(weights, "target weights must be positive")
        tf.debugging.assert_positive(scale, "score scale must be positive")
        tf.debugging.assert_near(tf.reduce_sum(weights), tf.constant(1.0, DTYPE))
        if not str(self.role):
            raise ValueError("score batch role must be nonempty")
        object.__setattr__(self, "physical_points", physical)
        object.__setattr__(self, "local_points", local)
        object.__setattr__(self, "log_likelihood", likelihood)
        object.__setattr__(self, "target_score", score)
        object.__setattr__(self, "target_weights", weights)
        object.__setattr__(self, "score_scale", scale)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "role", str(self.role))

    @property
    def sample_count(self) -> int:
        return int(self.physical_points.shape[0])

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "schema": T1_SCORE_BATCH_SCHEMA,
            "role": self.role,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "physical_points_sha256": tensor_sha256(self.physical_points),
            "local_points_sha256": tensor_sha256(self.local_points),
            "log_likelihood_sha256": tensor_sha256(self.log_likelihood),
            "target_score_sha256": tensor_sha256(self.target_score),
            "target_weights_sha256": tensor_sha256(self.target_weights),
            "score_scale": tuple(float(value) for value in self.score_scale.numpy()),
            "proposal_law": "p0(z0) f0(z1|z0)",
            "target_score": "initial_plus_transition_plus_observation_manual_score",
        }


def build_t1_score_batch_from_cloud(
    *,
    parent: LaneBT1Artifact,
    cloud: LaneBT1ProposalCloud,
    score_scale: tf.Tensor | None = None,
) -> LaneBT1ScoreBatch:
    _require_t1_parent(parent)
    local, _reference = physical_to_local_and_reference(cloud.joint_points, parent.frame)
    target_score = t1_complete_data_parameter_score(cloud.joint_points)
    weights = normalized_likelihood_weights(cloud.log_likelihood)
    if score_scale is None:
        scale = tf.sqrt(
            tf.reduce_sum(weights[:, tf.newaxis] * tf.square(target_score), axis=0)
        )
        scale = tf.maximum(scale, tf.constant(1e-6, DTYPE))
    else:
        scale = tf.reshape(tf.convert_to_tensor(score_scale, DTYPE), [PARAMETER_DIM])
    return LaneBT1ScoreBatch(
        physical_points=cloud.joint_points,
        local_points=local,
        log_likelihood=cloud.log_likelihood,
        target_score=target_score,
        target_weights=weights,
        score_scale=scale,
        seed=cloud.seed,
        role=cloud.role,
    )


def generate_t1_score_batch(
    *,
    parent: LaneBT1Artifact,
    sample_count: int,
    seed: int,
    role: str,
    score_scale: tf.Tensor | None = None,
) -> LaneBT1ScoreBatch:
    cloud = generate_t1_proposal_cloud(
        sample_count=int(sample_count), seed=int(seed), role=str(role)
    )
    return build_t1_score_batch_from_cloud(
        parent=parent, cloud=cloud, score_scale=score_scale
    )


@dataclass(frozen=True)
class LaneBT1FisherScoreEstimate:
    role: str
    seed: int
    sample_count: int
    score: tf.Tensor
    standard_error: tf.Tensor
    effective_sample_size: tf.Tensor
    score_sha256: str
    log_likelihood_sha256: str

    def __post_init__(self) -> None:
        score = tf.reshape(tf.convert_to_tensor(self.score, DTYPE), [PARAMETER_DIM])
        standard_error = tf.reshape(
            tf.convert_to_tensor(self.standard_error, DTYPE), [PARAMETER_DIM]
        )
        ess = tf.reshape(tf.convert_to_tensor(self.effective_sample_size, DTYPE), [])
        tf.debugging.assert_all_finite(score, "Fisher score")
        tf.debugging.assert_all_finite(standard_error, "Fisher score standard error")
        tf.debugging.assert_non_negative(standard_error)
        tf.debugging.assert_positive(ess)
        if int(self.sample_count) < 2:
            raise ValueError("Fisher estimate requires at least two samples")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "effective_sample_size", ess)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "estimator_id": T1_FISHER_ESTIMATE_ID,
            "role": self.role,
            "seed": int(self.seed),
            "sample_count": int(self.sample_count),
            "score": tuple(float(value) for value in self.score.numpy()),
            "standard_error": tuple(
                float(value) for value in self.standard_error.numpy()
            ),
            "effective_sample_size": float(self.effective_sample_size.numpy()),
            "score_sha256": self.score_sha256,
            "log_likelihood_sha256": self.log_likelihood_sha256,
        }


def estimate_t1_fisher_score(batch: LaneBT1ScoreBatch) -> LaneBT1FisherScoreEstimate:
    weights = batch.target_weights
    score = tf.reduce_sum(weights[:, tf.newaxis] * batch.target_score, axis=0)
    mean_scaled_weight = tf.reduce_mean(
        tf.exp(batch.log_likelihood - tf.reduce_max(batch.log_likelihood))
    )
    scaled_weight = tf.exp(batch.log_likelihood - tf.reduce_max(batch.log_likelihood))
    influence = (
        scaled_weight[:, tf.newaxis]
        * (batch.target_score - score[tf.newaxis, :])
        / mean_scaled_weight
    )
    variance = tf.reduce_sum(tf.square(influence), axis=0) / tf.cast(
        batch.sample_count - 1, DTYPE
    )
    standard_error = tf.sqrt(variance / tf.cast(batch.sample_count, DTYPE))
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))
    return LaneBT1FisherScoreEstimate(
        role=batch.role,
        seed=batch.seed,
        sample_count=batch.sample_count,
        score=score,
        standard_error=standard_error,
        effective_sample_size=ess,
        score_sha256=tensor_sha256(batch.target_score),
        log_likelihood_sha256=tensor_sha256(batch.log_likelihood),
    )


@dataclass(frozen=True)
class LaneBT1TangentTrainingConfig:
    arm_id: str
    learning_rate: float
    l1_weight: float
    l2_weight: float
    gradient_clip_norm: float
    batch_size: int
    train_steps: int
    seed: int

    def __post_init__(self) -> None:
        if not str(self.arm_id):
            raise ValueError("tangent arm_id must be nonempty")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("l1_weight", "l2_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if int(self.batch_size) <= 1 or int(self.train_steps) <= 0:
            raise ValueError("tangent batch_size and train_steps must be positive")

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class LaneBT1TangentTrainer:
    """Train only origin tangent banks while treating all parent fields as constants."""

    def __init__(self, parent: LaneBT1Artifact) -> None:
        _require_t1_parent(parent)
        self.parent = parent
        self.parent_cores = tuple(tf.identity(core) for core in parent.cores)
        self.product_basis = lane_b_product_basis(
            order=parent.settings.basis_order,
            num_elems=parent.settings.basis_num_elems,
        )
        self.cores = tuple(
            tuple(
                tf.Variable(
                    tf.zeros_like(core),
                    trainable=True,
                    name=f"lane_b_t1_tangent_axis_{axis:02d}_parameter_{parameter}",
                )
                for parameter in range(PARAMETER_DIM)
            )
            for axis, core in enumerate(self.parent_cores)
        )

    @property
    def variables(self) -> tuple[tf.Variable, ...]:
        return tuple(variable for bank in self.cores for variable in bank)

    def amplitude_and_tangent(self, local_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        points = tf.convert_to_tensor(local_points, DTYPE)
        if points.shape.rank != 2 or points.shape[1] != len(self.parent_cores):
            raise ValueError("tangent trainer points have wrong shape")
        sample_count = tf.shape(points)[0]
        value = tf.ones([sample_count, 1], DTYPE)
        derivative = tf.zeros([PARAMETER_DIM, sample_count, 1], DTYPE)
        for axis, (parent_core, tangent_bank) in enumerate(
            zip(self.parent_cores, self.cores)
        ):
            basis_values = self.product_basis.evaluate_axis(axis, points[:, axis])
            parent_matrix = tf.einsum("nl,alb->nab", basis_values, parent_core)
            next_value = tf.einsum("na,nab->nb", value, parent_matrix)
            next_derivative = []
            for parameter in range(PARAMETER_DIM):
                tangent_matrix = tf.einsum(
                    "nl,alb->nab", basis_values, tangent_bank[parameter]
                )
                next_derivative.append(
                    tf.einsum("na,nab->nb", value, tangent_matrix)
                    + tf.einsum("na,nab->nb", derivative[parameter], parent_matrix)
                )
            value = next_value
            derivative = tf.stack(next_derivative, axis=0)
        return tf.reshape(value, [sample_count]), tf.transpose(
            tf.reshape(derivative, [PARAMETER_DIM, sample_count])
        )

    def unnormalized_log_density_score(self, local_points: tf.Tensor) -> tf.Tensor:
        amplitude, derivative = self.amplitude_and_tangent(local_points)
        density = tf.square(amplitude) + tf.constant(self.parent.settings.tau, DTYPE)
        score = 2.0 * amplitude[:, tf.newaxis] * derivative / density[:, tf.newaxis]
        tf.debugging.assert_all_finite(score, "trained origin density score")
        return score

    def freeze_child(self) -> LaneBParameterChild:
        return LaneBParameterChild(
            self.parent,
            tuple(tuple(tf.identity(value) for value in bank) for bank in self.cores),
        )

    def calibrate_normalizer_score(self, target_score: tf.Tensor) -> tf.Tensor:
        """Add a training-only parent-amplitude gauge to match normalizer score."""

        target_mean = tf.reshape(
            tf.convert_to_tensor(target_score, DTYPE), [PARAMETER_DIM]
        )
        child = self.freeze_child()
        _current_increment, current_score = child.increment_and_score(
            tf.zeros([PARAMETER_DIM], DTYPE)
        )
        # H is the square-root parent mass and Z=H+tau for this fixed branch.
        parent_density = child.density(tf.zeros([PARAMETER_DIM], DTYPE))
        h_mass = parent_density.sqrt_square_normalizer()
        z_mass = parent_density.normalizer()
        alpha = (target_mean - current_score) * z_mass / (2.0 * h_mass)
        for parameter in range(PARAMETER_DIM):
            self.cores[0][parameter].assign_add(alpha[parameter] * self.parent_cores[0])
        tf.debugging.assert_all_finite(alpha, "normalizer-score gauge")
        return alpha


def make_compiled_tangent_train_step(
    trainer: LaneBT1TangentTrainer,
    optimizer: tf.keras.optimizers.Optimizer,
    config: LaneBT1TangentTrainingConfig,
):
    if hasattr(optimizer, "build"):
        optimizer.build(trainer.variables)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(
        local_points: tf.Tensor,
        target_score: tf.Tensor,
        target_weights: tf.Tensor,
        score_scale: tf.Tensor,
        population_size: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        batch_count = tf.cast(tf.shape(local_points)[0], DTYPE)
        population = tf.cast(population_size, DTYPE)
        with tf.GradientTape() as tape:
            prediction = trainer.unnormalized_log_density_score(local_points)
            normalized_residual = (
                prediction - target_score
            ) / score_scale[tf.newaxis, :]
            data_loss = population / batch_count * tf.reduce_sum(
                target_weights * tf.reduce_sum(tf.square(normalized_residual), axis=1)
            )
            l1 = tf.add_n([tf.reduce_sum(tf.abs(value)) for value in trainer.variables])
            l2 = tf.add_n([tf.reduce_sum(tf.square(value)) for value in trainer.variables])
            regularization = (
                tf.constant(config.l1_weight, DTYPE) * l1
                + tf.constant(config.l2_weight, DTYPE) * l2
            )
            loss = data_loss + regularization
        gradients = tape.gradient(loss, trainer.variables)
        clipped, gradient_norm = tf.clip_by_global_norm(
            gradients, tf.constant(config.gradient_clip_norm, DTYPE)
        )
        optimizer.apply_gradients(zip(clipped, trainer.variables))
        return loss, data_loss, regularization, gradient_norm

    return train_step


def tangent_validation_metrics(
    trainer: LaneBT1TangentTrainer,
    batch: LaneBT1ScoreBatch,
) -> Mapping[str, tf.Tensor]:
    prediction = trainer.unnormalized_log_density_score(batch.local_points)
    residual = prediction - batch.target_score
    weights = batch.target_weights[:, tf.newaxis]
    target_mean = tf.reduce_sum(weights * batch.target_score, axis=0)
    prediction_mean = tf.reduce_sum(weights * prediction, axis=0)
    target_centered = batch.target_score - target_mean[tf.newaxis, :]
    prediction_centered = prediction - prediction_mean[tf.newaxis, :]
    covariance = tf.reduce_sum(weights * target_centered * prediction_centered, axis=0)
    target_variance = tf.reduce_sum(weights * tf.square(target_centered), axis=0)
    prediction_variance = tf.reduce_sum(
        weights * tf.square(prediction_centered), axis=0
    )
    correlation = covariance / tf.sqrt(
        tf.maximum(target_variance * prediction_variance, tf.constant(1e-300, DTYPE))
    )
    raw_rms = tf.sqrt(tf.reduce_sum(weights * tf.square(residual), axis=0))
    normalized_rms = raw_rms / batch.score_scale
    return {
        "raw_rms": raw_rms,
        "normalized_rms": normalized_rms,
        "weighted_target_mean": target_mean,
        "weighted_prediction_mean": prediction_mean,
        "weighted_correlation": correlation,
        "maximum_absolute_residual": tf.reduce_max(tf.abs(residual), axis=0),
        "tangent_l1": tf.add_n(
            [tf.reduce_sum(tf.abs(value)) for value in trainer.variables]
        ),
        "tangent_l2": tf.sqrt(
            tf.add_n([tf.reduce_sum(tf.square(value)) for value in trainer.variables])
        ),
    }


def make_compiled_child_origin_score(child: LaneBParameterChild):
    """Compile the frozen child's origin value/score contraction for tie-out."""

    basis = lane_b_product_basis(
        order=child.settings.basis_order,
        num_elems=child.settings.basis_num_elems,
    )
    parent_cores = child.parent_cores
    tangent_cores = child.tangent_cores
    tau = tf.constant(child.settings.tau, DTYPE)
    shift = tf.reshape(tf.convert_to_tensor(child.parent.shift_constant, DTYPE), [])

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled() -> tuple[tf.Tensor, tf.Tensor]:
        square_mass, derivative = _paired_mass_product(
            basis=basis,
            parent_cores=parent_cores,
            tangent_cores=tangent_cores,
            theta=tf.zeros([PARAMETER_DIM], DTYPE),
        )
        normalizer = square_mass + tau
        return tf.math.log(normalizer) - shift, derivative / normalizer

    return compiled


def tangent_workspace_estimate_bytes(
    *, parent: LaneBT1Artifact, batch_size: int
) -> int:
    _require_t1_parent(parent)
    n = int(batch_size)
    if n <= 1:
        raise ValueError("batch_size must be greater than one")
    max_rank = max(max(int(core.shape[0]), int(core.shape[2])) for core in parent.cores)
    tangent_elements = PARAMETER_DIM * sum(int(tf.size(core)) for core in parent.cores)
    # Conservative allowance for forward values, three tangents, reverse-mode
    # intermediates, optimizer slots, and XLA overlap.
    live_batch_slots = n * len(parent.cores) * (8 * max_rank * max_rank + 32)
    persistent_slots = 10 * tangent_elements
    return int(DTYPE.size * (live_batch_slots + persistent_slots))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def t1_score_source_closure() -> Mapping[str, str]:
    from bayesfilter.highdim import fixed_branch, models, sir_latent_preclip_tf
    from bayesfilter.highdim import zhao_cui_austria_sir_lane_b_target_tf
    from bayesfilter.highdim import zhao_cui_austria_sir_lane_b_tf
    from bayesfilter.highdim import zhao_cui_austria_sir_parameter_child_tf

    modules = (
        fixed_branch,
        models,
        sir_latent_preclip_tf,
        zhao_cui_austria_sir_lane_b_target_tf,
        zhao_cui_austria_sir_lane_b_tf,
        zhao_cui_austria_sir_parameter_child_tf,
    )
    root = Path(__file__).resolve().parents[2]
    paths = [Path(inspect.getfile(module)).resolve() for module in modules]
    paths.append(Path(__file__).resolve())
    paths.append(root / T1_SCORE_PLAN)
    paths.append(root / "scripts/run_zhao_cui_austria_sir_lane_b_t1_score.py")
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(set(paths))
    }


def _metric_payload(metrics: Mapping[str, tf.Tensor]) -> Mapping[str, object]:
    payload: dict[str, object] = {}
    for name, raw_value in sorted(metrics.items()):
        value = tf.convert_to_tensor(raw_value, DTYPE)
        tf.debugging.assert_all_finite(value, f"validation metric {name}")
        flattened = tf.reshape(value, [-1])
        values = tuple(float(item) for item in flattened.numpy())
        payload[name] = values[0] if value.shape.rank == 0 else values
    return payload


def issue_t1_score_artifact_identity(
    *,
    parent: LaneBT1Artifact,
    tangent_cores: Sequence[Sequence[tf.Tensor]],
    config: LaneBT1TangentTrainingConfig,
    training_manifest: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    score_scale: tf.Tensor,
    validation_metrics: Mapping[str, tf.Tensor],
    source_hashes: Mapping[str, str],
) -> BranchIdentity:
    _require_t1_parent(parent)
    child = LaneBParameterChild(
        parent,
        tuple(tuple(tf.convert_to_tensor(value, DTYPE) for value in bank) for bank in tangent_cores),
    )
    if training_manifest.get("role") != "score_training":
        raise ValueError("trained score artifact requires score_training role")
    if validation_manifest.get("role") != "score_validation":
        raise ValueError("trained score artifact requires score_validation role")
    if int(training_manifest.get("seed", -1)) == int(validation_manifest.get("seed", -1)):
        raise ValueError("training and validation seeds must be disjoint")
    scale = tf.reshape(tf.convert_to_tensor(score_scale, DTYPE), [PARAMETER_DIM])
    tf.debugging.assert_positive(scale, "artifact score scale")
    payload = {
        "schema": T1_SCORE_IDENTITY_SCHEMA,
        "trainer_id": T1_SCORE_TRAINER_ID,
        "classification": T1_SCORE_CLASSIFICATION,
        "parent_identity": parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
        "config": config.manifest_payload(),
        "training_manifest": dict(training_manifest),
        "validation_manifest": dict(validation_manifest),
        "score_scale_sha256": tensor_sha256(scale),
        "score_scale": tuple(float(value) for value in scale.numpy()),
        "validation_metrics": _metric_payload(validation_metrics),
        "source_closure": dict(source_hashes),
        "plan": T1_SCORE_PLAN,
        "parent_cores_immutable": True,
        "theta_integration_forbidden": True,
        "runtime_autodiff_score": False,
        "runtime_finite_difference_score": False,
        "hmc_authorized": False,
    }
    # This identity is a persisted JSON schema. Normalize sequences before
    # hashing so a fresh JSON decode cannot change tuple/list type tags.
    persisted_payload = json.loads(json.dumps(payload, sort_keys=True))
    manifest = BranchManifest(T1_SCORE_IDENTITY_SCHEMA, persisted_payload)
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())


@dataclass(frozen=True)
class LaneBT1ScoreArtifact:
    parent: LaneBT1Artifact
    tangent_cores: tuple[tuple[tf.Tensor, ...], ...]
    config: LaneBT1TangentTrainingConfig
    training_manifest: Mapping[str, object]
    validation_manifest: Mapping[str, object]
    score_scale: tf.Tensor
    validation_metrics: Mapping[str, tf.Tensor]
    source_hashes: Mapping[str, str]
    identity: BranchIdentity

    def __post_init__(self) -> None:
        _require_t1_parent(self.parent)
        tangents = tuple(
            tuple(tf.convert_to_tensor(value, DTYPE) for value in bank)
            for bank in self.tangent_cores
        )
        scale = tf.reshape(tf.convert_to_tensor(self.score_scale, DTYPE), [PARAMETER_DIM])
        expected = issue_t1_score_artifact_identity(
            parent=self.parent,
            tangent_cores=tangents,
            config=self.config,
            training_manifest=self.training_manifest,
            validation_manifest=self.validation_manifest,
            score_scale=scale,
            validation_metrics=self.validation_metrics,
            source_hashes=self.source_hashes,
        )
        if self.identity != expected:
            raise ValueError("T1 score artifact identity mismatch")
        object.__setattr__(self, "tangent_cores", tangents)
        object.__setattr__(self, "score_scale", scale)

    def child(self) -> LaneBParameterChild:
        return LaneBParameterChild(self.parent, self.tangent_cores)


def make_t1_score_artifact(
    *,
    trainer: LaneBT1TangentTrainer,
    config: LaneBT1TangentTrainingConfig,
    training_batch: LaneBT1ScoreBatch,
    validation_batch: LaneBT1ScoreBatch,
    validation_metrics: Mapping[str, tf.Tensor],
) -> LaneBT1ScoreArtifact:
    tangents = tuple(
        tuple(tf.identity(value) for value in bank) for bank in trainer.cores
    )
    hashes = t1_score_source_closure()
    identity = issue_t1_score_artifact_identity(
        parent=trainer.parent,
        tangent_cores=tangents,
        config=config,
        training_manifest=training_batch.manifest_payload(),
        validation_manifest=validation_batch.manifest_payload(),
        score_scale=training_batch.score_scale,
        validation_metrics=validation_metrics,
        source_hashes=hashes,
    )
    return LaneBT1ScoreArtifact(
        parent=trainer.parent,
        tangent_cores=tangents,
        config=config,
        training_manifest=training_batch.manifest_payload(),
        validation_manifest=validation_batch.manifest_payload(),
        score_scale=training_batch.score_scale,
        validation_metrics=dict(validation_metrics),
        source_hashes=hashes,
        identity=identity,
    )


def _write_tensor(path: Path, value: tf.Tensor) -> Mapping[str, object]:
    tensor = tf.convert_to_tensor(value, DTYPE)
    serialized = tf.io.serialize_tensor(tensor)
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
    }


def save_t1_score_artifact(artifact: LaneBT1ScoreArtifact, directory: Path) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    tensors: dict[str, Mapping[str, object]] = {
        "score_scale": _write_tensor(output / "score_scale.tensor", artifact.score_scale)
    }
    for axis, bank in enumerate(artifact.tangent_cores):
        for parameter, value in enumerate(bank):
            name = f"tangent_{axis:02d}_{parameter}"
            tensors[name] = _write_tensor(output / f"{name}.tensor", value)
    payload = {
        "schema_version": T1_SCORE_ARTIFACT_SCHEMA,
        "identity_sha256": artifact.identity.hash.value,
        "parent_identity": artifact.parent.identity.hash.value,
        "child_identity": artifact.child().identity.hash.value,
        "config": artifact.config.manifest_payload(),
        "training_manifest": dict(artifact.training_manifest),
        "validation_manifest": dict(artifact.validation_manifest),
        "validation_metrics": _metric_payload(artifact.validation_metrics),
        "source_closure": dict(artifact.source_hashes),
        "tensors": tensors,
        "nonclaims": (
            "no T2 or later score",
            "no exact nonlinear likelihood theorem",
            "no HMC or posterior readiness",
            "no source-faithful parameter algorithm",
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_t1_score_artifact(
    directory: Path, *, parent: LaneBT1Artifact
) -> LaneBT1ScoreArtifact:
    _require_t1_parent(parent)
    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text())
    if payload.get("schema_version") != T1_SCORE_ARTIFACT_SCHEMA:
        raise ValueError("T1 score artifact schema mismatch")
    if payload.get("parent_identity") != parent.identity.hash.value:
        raise ValueError("T1 score artifact parent mismatch")
    current_closure = dict(t1_score_source_closure())
    if payload.get("source_closure") != current_closure:
        raise ValueError("T1 score artifact source closure is stale")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("T1 score artifact tensor ledger missing")

    def read_tensor(name: str) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"T1 score artifact tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        if hashlib.sha256(bytes(serialized.numpy())).hexdigest() != row.get("sha256"):
            raise ValueError(f"T1 score artifact tensor hash mismatch: {name}")
        value = tf.io.parse_tensor(serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"])))
        return tf.ensure_shape(value, row["shape"])

    tangents = tuple(
        tuple(read_tensor(f"tangent_{axis:02d}_{parameter}") for parameter in range(PARAMETER_DIM))
        for axis in range(len(parent.cores))
    )
    scale = read_tensor("score_scale")
    config = LaneBT1TangentTrainingConfig(**payload["config"])
    validation_metrics = {
        name: tf.constant(value, DTYPE)
        for name, value in payload["validation_metrics"].items()
    }
    identity = issue_t1_score_artifact_identity(
        parent=parent,
        tangent_cores=tangents,
        config=config,
        training_manifest=payload["training_manifest"],
        validation_manifest=payload["validation_manifest"],
        score_scale=scale,
        validation_metrics=validation_metrics,
        source_hashes=current_closure,
    )
    if identity.hash.value != payload.get("identity_sha256"):
        raise ValueError("T1 score artifact identity hash mismatch")
    artifact = LaneBT1ScoreArtifact(
        parent=parent,
        tangent_cores=tangents,
        config=config,
        training_manifest=payload["training_manifest"],
        validation_manifest=payload["validation_manifest"],
        score_scale=scale,
        validation_metrics=validation_metrics,
        source_hashes=current_closure,
        identity=identity,
    )
    if artifact.child().identity.hash.value != payload.get("child_identity"):
        raise ValueError("T1 score artifact child identity mismatch")
    return artifact


__all__ = [
    "EXPECTED_T1_PARENT_IDENTITY",
    "LaneBT1FisherScoreEstimate",
    "LaneBT1ScoreArtifact",
    "LaneBT1ScoreBatch",
    "LaneBT1TangentTrainer",
    "LaneBT1TangentTrainingConfig",
    "T1_FISHER_ESTIMATE_ID",
    "T1_SCORE_CLASSIFICATION",
    "T1_SCORE_TRAINER_ID",
    "build_t1_score_batch_from_cloud",
    "estimate_t1_fisher_score",
    "generate_t1_score_batch",
    "issue_t1_score_artifact_identity",
    "load_t1_score_artifact",
    "make_t1_score_artifact",
    "make_compiled_tangent_train_step",
    "normalized_likelihood_weights",
    "save_t1_score_artifact",
    "t1_complete_data_parameter_score",
    "tangent_validation_metrics",
    "tangent_workspace_estimate_bytes",
    "t1_score_source_closure",
]
