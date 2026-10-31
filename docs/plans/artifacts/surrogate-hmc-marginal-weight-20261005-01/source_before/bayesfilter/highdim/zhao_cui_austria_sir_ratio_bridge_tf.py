"""Parent-measure Radon-Nikodym bridge for the Austria SIR T1 diagnostic.

The bridge defines an explicit extension target

    L_ZC(theta) = L_parent(0) + log E_q0[pi_theta / pi_0],

where q0 is the admitted fixed T1 TT density and pi is the unnormalized
physical T1 joint target.  This is not the exact physical likelihood unless
q0 equals the normalized physical target at theta zero.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    LatentPreclipSIRSSM,
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import LaneBT1Artifact


DTYPE = tf.float64
PARAMETER_DIM = 3
STATE_DIM = 18
BRIDGE_ID = "zhao_cui_austria_sir_t1_parent_measure_ratio_bridge_v1"
CLASSIFICATION = "extension_or_invention"
EXPECTED_PARENT_IDENTITY = (
    "e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59"
)
EXPECTED_PARENT_VALUE = -31.1290512231882
ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _tensor_hash(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def _theta(value: tf.Tensor) -> tf.Tensor:
    result = tf.reshape(tf.convert_to_tensor(value, DTYPE), [PARAMETER_DIM])
    tf.debugging.assert_all_finite(result, "theta must be finite")
    return result


def load_admitted_parent(directory: Path = PARENT_DIR) -> LaneBT1Artifact:
    parent = load_lane_b_t1_artifact_v1_compat(Path(directory))
    if parent.identity.hash.value != EXPECTED_PARENT_IDENTITY:
        raise ValueError("ratio bridge parent identity mismatch")
    value = float(parent.value().numpy())
    if abs(value - EXPECTED_PARENT_VALUE) > 2.0e-13:
        raise ValueError("ratio bridge parent value mismatch")
    return parent


def sample_parent_local_points(
    *, sample_count: int, seed: int, parent: LaneBT1Artifact
) -> tf.Tensor:
    """Draw local coordinates from the fixed parent KR transport."""

    n = int(sample_count)
    if n < 2:
        raise ValueError("sample_count must be at least two")
    uniforms = tf.random.stateless_uniform(
        [STATE_DIM * 2, n],
        seed=tf.constant([int(seed), 4401], tf.int32),
        dtype=DTYPE,
    )
    # Reuse the exact fixed uniforms while keeping the parent's KR grid
    # workspace below its configured cap.
    transport = parent.transport()
    chunk_size = 1024
    chunks = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunks.append(transport.inverse_transport(uniforms[:, start:stop]))
    local = tf.concat(chunks, axis=1)
    if local.shape != (STATE_DIM * 2, n):
        raise ValueError("parent transport returned an invalid local cloud")
    tf.debugging.assert_all_finite(local, "parent local cloud")
    return local


def local_to_physical(local_points: tf.Tensor, parent: LaneBT1Artifact) -> tf.Tensor:
    local = tf.convert_to_tensor(local_points, DTYPE)
    if local.shape.rank != 2 or local.shape[0] != STATE_DIM * 2:
        raise ValueError("local_points must have shape [36,sample]")
    physical = tf.linalg.matmul(parent.frame.matrix, local) + parent.frame.mu[:, tf.newaxis]
    result = tf.transpose(physical)
    tf.debugging.assert_all_finite(result, "parent physical cloud")
    return result


def _physical_joint_log_density(
    theta: tf.Tensor, physical_points: tf.Tensor, model: LatentPreclipSIRSSM
) -> tf.Tensor:
    values = tf.convert_to_tensor(physical_points, DTYPE)
    if values.shape.rank != 2 or values.shape[1] != STATE_DIM * 2:
        raise ValueError("physical_points must have shape [sample,36]")
    parameters = _theta(theta)
    z1 = values[:, :STATE_DIM]
    z0 = values[:, STATE_DIM:]
    _states, observations, _all = generate_sealed_lane_b_dataset()
    result = (
        model.initial_log_density(parameters, z0)
        + model.transition_log_density(parameters, z0, z1, 1)
        + model.observation_log_density(parameters, z1, observations[0], 1)
    )
    tf.debugging.assert_all_finite(result, "physical joint log density")
    return result


def _physical_joint_score(
    theta: tf.Tensor, physical_points: tf.Tensor, model: LatentPreclipSIRSSM
) -> tf.Tensor:
    values = tf.convert_to_tensor(physical_points, DTYPE)
    parameters = _theta(theta)
    z1 = values[:, :STATE_DIM]
    z0 = values[:, STATE_DIM:]
    _states, observations, _all = generate_sealed_lane_b_dataset()
    result = (
        model.initial_log_density_parameter_score(parameters, z0)
        + model.transition_log_density_parameter_score(parameters, z0, z1, 1)
        + model.observation_log_density_parameter_score(
            parameters, z1, observations[0], 1
        )
    )
    tf.debugging.assert_all_finite(result, "physical joint score")
    return result


def parent_measure_ratio_bridge(
    theta: tf.Tensor,
    physical_points: tf.Tensor,
    *,
    parent: LaneBT1Artifact,
    model: LatentPreclipSIRSSM | None = None,
) -> Mapping[str, tf.Tensor]:
    """Evaluate the finite parent-measure ratio value and analytical score."""

    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    parameters = _theta(theta)
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    target_log = _physical_joint_log_density(parameters, physical_points, active_model)
    origin_log = _physical_joint_log_density(origin, physical_points, active_model)
    log_ratio = target_log - origin_log
    log_value_ratio = tf.reduce_logsumexp(log_ratio) - tf.math.log(
        tf.cast(tf.shape(log_ratio)[0], DTYPE)
    )
    weights = tf.nn.softmax(log_ratio)
    point_score = _physical_joint_score(parameters, physical_points, active_model)
    score = tf.reduce_sum(weights[:, tf.newaxis] * point_score, axis=0)
    centered = point_score - score[tf.newaxis, :]
    scaled = tf.exp(log_ratio - tf.reduce_max(log_ratio))
    mean_scaled = tf.reduce_mean(scaled)
    influence = scaled[:, tf.newaxis] * centered / mean_scaled
    standard_error = tf.sqrt(
        tf.reduce_sum(tf.square(influence), axis=0)
        / tf.cast(tf.shape(log_ratio)[0] * (tf.shape(log_ratio)[0] - 1), DTYPE)
    )
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))
    for name, value in (("log_value_ratio", log_value_ratio), ("score", score), ("standard_error", standard_error), ("ess", ess)):
        tf.debugging.assert_all_finite(value, name)
    return {
        "log_value": tf.constant(EXPECTED_PARENT_VALUE, DTYPE) + log_value_ratio,
        "log_value_ratio": log_value_ratio,
        "score": score,
        "standard_error": standard_error,
        "effective_sample_size": ess,
        "log_ratio": log_ratio,
    }


def parent_measure_ratio_bridge_autodiff(
    theta: tf.Tensor,
    physical_points: tf.Tensor,
    *,
    model: LatentPreclipSIRSSM | None = None,
) -> Mapping[str, tf.Tensor]:
    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    parameters = tf.convert_to_tensor(theta, DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(parameters)
        result = parent_measure_ratio_bridge(
            parameters,
            physical_points,
            parent=load_admitted_parent(),
            model=active_model,
        )
        value = result["log_value"]
    score = tape.gradient(value, parameters)
    if score is None:
        raise ValueError("ratio bridge value is disconnected from theta")
    return {"log_value": value, "score": tf.reshape(score, [PARAMETER_DIM])}


def bridge_manifest(parent: LaneBT1Artifact, local_points: tf.Tensor) -> Mapping[str, object]:
    return {
        "bridge_id": BRIDGE_ID,
        "classification": CLASSIFICATION,
        "parent_identity": parent.identity.hash.value,
        "parent_value": EXPECTED_PARENT_VALUE,
        "local_points_sha256": _tensor_hash(local_points),
        "measure": "admitted_parent_normalized_reference_measure_q0",
        "target": "unnormalized_physical_p0_f0_g_y1_ratio_at_theta",
        "definition": "L_parent(0)+log E_q0[pi_theta/pi_0]",
        "source_faithfulness": "not_claimed",
        "nonclaims": (
            "not exact physical likelihood unless q0_equals_pi0",
            "not source-faithful Zhao-Cui",
            "not full-horizon score",
            "not HMC evidence",
        ),
    }


__all__ = [
    "BRIDGE_ID",
    "CLASSIFICATION",
    "EXPECTED_PARENT_IDENTITY",
    "EXPECTED_PARENT_VALUE",
    "bridge_manifest",
    "load_admitted_parent",
    "local_to_physical",
    "parent_measure_ratio_bridge",
    "parent_measure_ratio_bridge_autodiff",
    "sample_parent_local_points",
]
