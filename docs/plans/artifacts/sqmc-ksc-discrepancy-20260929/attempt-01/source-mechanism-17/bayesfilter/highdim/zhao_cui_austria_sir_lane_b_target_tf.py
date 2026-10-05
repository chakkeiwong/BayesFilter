"""Sealed data and coherent T1 law for the Austria SIR Lane-B baseline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    LatentPreclipSIRSSM,
    latent_preclip_zhao_cui_sir_austria_model,
)


DTYPE = tf.float64
LANE_B_TARGET_ID = "zhao_cui_austria_sir_lane_b_latent_preclip_t1_v1"
LANE_B_EVENT_ORDER = "z0_then_transition_to_z1_then_observe_sealed_y1"
SIR_DATASET_ID = "zhao_cui_austria_parameter_extension_y1_y20"
SIR_DATASET_SEED = 81120
SIR_HORIZON = 20
SIR_STATE_DIM = 18
SIR_OBSERVATION_DIM = 9
SIR_JOINT_DIM = 36
SIR_STATE_SHA256 = "8cd5a079f5799f0e0b769e5ac21a4bdf460475a72319f07dc27fb037eb5774e0"
SIR_OBSERVATION_SHA256 = (
    "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07"
)
SIR_WRONG_TIME_ORDER_SHA256 = (
    "c4df0ac33a28bde16cad169892f49705f1dfa4f3541eeac9ae4afa4aa33cf041"
)
SIR_RUNTIME_FP32_OBSERVATION_SHA256 = (
    "40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009"
)


def tensor_sha256(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def generate_sealed_lane_b_dataset() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Reproduce the comparison-row data without importing UKF/SGQF code."""

    model = latent_preclip_zhao_cui_sir_austria_model().physical_model.base_model
    with tf.device("/CPU:0"):
        states, all_observations = model.simulate(
            final_time=SIR_HORIZON,
            seed=SIR_DATASET_SEED,
        )
        states = tf.ensure_shape(tf.convert_to_tensor(states, DTYPE), [21, 18])
        all_observations = tf.ensure_shape(
            tf.convert_to_tensor(all_observations, DTYPE), [21, 9]
        )
        observations = tf.ensure_shape(all_observations[1:], [20, 9])
    expected = (
        ("states", states, SIR_STATE_SHA256),
        ("observations", observations, SIR_OBSERVATION_SHA256),
        ("wrong_time_order", all_observations[:-1], SIR_WRONG_TIME_ORDER_SHA256),
        (
            "runtime_fp32_observations",
            tf.cast(observations, tf.float32),
            SIR_RUNTIME_FP32_OBSERVATION_SHA256,
        ),
    )
    for name, value, digest in expected:
        if tensor_sha256(value) != digest:
            raise ValueError(f"sealed Lane-B {name} hash mismatch")
    if bool(tf.reduce_any(states[:, 0::2] < 0.0).numpy()):
        raise ValueError("sealed Lane-B fixture unexpectedly activates clipping")
    return states, observations, all_observations


@dataclass(frozen=True)
class LaneBT1ProposalCloud:
    """An iid cloud from p0(z0) f(z1|z0), ordered as [z1,z0]."""

    joint_points: tf.Tensor
    log_likelihood: tf.Tensor
    seed: int
    role: str

    def __post_init__(self) -> None:
        points = tf.convert_to_tensor(self.joint_points, DTYPE)
        log_likelihood = tf.convert_to_tensor(self.log_likelihood, DTYPE)
        if points.shape.rank != 2 or points.shape[1] != SIR_JOINT_DIM:
            raise ValueError("joint_points must have shape [sample,36]")
        if points.shape[0] is None or log_likelihood.shape != (points.shape[0],):
            raise ValueError("log_likelihood must match the static sample count")
        tf.debugging.assert_all_finite(points, "joint_points must be finite")
        tf.debugging.assert_all_finite(log_likelihood, "log_likelihood must be finite")
        if int(points.shape[0]) < 2:
            raise ValueError("Lane-B proposal clouds require at least two samples")
        if not str(self.role):
            raise ValueError("cloud role must be nonempty")
        object.__setattr__(self, "joint_points", points)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "role", str(self.role))

    @property
    def sample_count(self) -> int:
        return int(self.joint_points.shape[0])

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "role": self.role,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "joint_axis_order": ("z1", "z0"),
            "joint_points_sha256": tensor_sha256(self.joint_points),
            "log_likelihood_sha256": tensor_sha256(self.log_likelihood),
            "proposal_law": "p0(z0) f(z1|z0)",
        }


def generate_t1_proposal_cloud(
    *,
    sample_count: int,
    seed: int,
    role: str,
    model: LatentPreclipSIRSSM | None = None,
) -> LaneBT1ProposalCloud:
    """Draw a stateless, batch-native proposal cloud for the sealed y1 target."""

    if int(sample_count) < 2:
        raise ValueError("sample_count must be at least two")
    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    if not isinstance(active_model, LatentPreclipSIRSSM):
        raise TypeError("model must be LatentPreclipSIRSSM")
    _states, observations, _all = generate_sealed_lane_b_dataset()
    theta = tf.zeros([3], DTYPE)
    initial_noise = tf.random.stateless_normal(
        [int(sample_count), SIR_STATE_DIM],
        seed=tf.constant([int(seed), 101], tf.int32),
        dtype=DTYPE,
    )
    transition_noise = tf.random.stateless_normal(
        [int(sample_count), SIR_STATE_DIM],
        seed=tf.constant([int(seed), 211], tf.int32),
        dtype=DTYPE,
    )
    scaled = active_model.physical_model.scaled_model(theta)
    z0 = scaled.initial_mean[tf.newaxis, :] + tf.linalg.matmul(
        initial_noise,
        tf.linalg.cholesky(scaled.initial_covariance),
        transpose_b=True,
    )
    z1 = active_model.transition_push_from_standard_normal(
        theta,
        z0,
        transition_noise,
        1,
    )
    log_likelihood = active_model.observation_log_density(
        theta,
        z1,
        observations[0],
        1,
    )
    return LaneBT1ProposalCloud(
        joint_points=tf.concat([z1, z0], axis=1),
        log_likelihood=log_likelihood,
        seed=int(seed),
        role=role,
    )


def t1_joint_log_density(
    joint_points: tf.Tensor,
    *,
    model: LatentPreclipSIRSSM | None = None,
) -> tf.Tensor:
    """Return log p0(z0)+log f(z1|z0)+log g(y1|z1)."""

    points = tf.convert_to_tensor(joint_points, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != SIR_JOINT_DIM:
        raise ValueError("joint_points must have shape [sample,36]")
    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    _states, observations, _all = generate_sealed_lane_b_dataset()
    theta = tf.zeros([3], DTYPE)
    z1 = points[:, :SIR_STATE_DIM]
    z0 = points[:, SIR_STATE_DIM:]
    return (
        active_model.initial_log_density(theta, z0)
        + active_model.transition_log_density(theta, z0, z1, 1)
        + active_model.observation_log_density(theta, z1, observations[0], 1)
    )


def target_manifest() -> Mapping[str, object]:
    model = latent_preclip_zhao_cui_sir_austria_model()
    return {
        "target_id": LANE_B_TARGET_ID,
        "dataset_id": SIR_DATASET_ID,
        "dataset_seed": SIR_DATASET_SEED,
        "source_observation_sha256": SIR_OBSERVATION_SHA256,
        "runtime_fp32_observation_sha256": SIR_RUNTIME_FP32_OBSERVATION_SHA256,
        "wrong_time_order_sentinel_sha256": SIR_WRONG_TIME_ORDER_SHA256,
        "event_order": LANE_B_EVENT_ORDER,
        "joint_axis_order": ("z1", "z0"),
        "joint_dimension": SIR_JOINT_DIM,
        "retained_axes": tuple(range(SIR_STATE_DIM)),
        "marginalized_axes": tuple(range(SIR_STATE_DIM, SIR_JOINT_DIM)),
        "theta": (0.0, 0.0, 0.0),
        "latent_model": model.manifest_payload(),
        "classification": "extension_or_invention",
        "forbidden_baseline_dependencies": (
            "APF",
            "UKF",
            "SGQF",
            "ALS",
            "source_replica",
            "retained_grid",
        ),
    }


__all__ = [
    "LANE_B_EVENT_ORDER",
    "LANE_B_TARGET_ID",
    "LaneBT1ProposalCloud",
    "SIR_JOINT_DIM",
    "SIR_OBSERVATION_SHA256",
    "SIR_RUNTIME_FP32_OBSERVATION_SHA256",
    "SIR_STATE_DIM",
    "generate_sealed_lane_b_dataset",
    "generate_t1_proposal_cloud",
    "target_manifest",
    "tensor_sha256",
    "t1_joint_log_density",
]
