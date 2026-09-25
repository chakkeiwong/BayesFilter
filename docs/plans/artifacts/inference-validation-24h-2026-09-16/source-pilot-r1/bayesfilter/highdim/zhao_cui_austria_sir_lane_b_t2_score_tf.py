"""Manual carried target score for the Lane-B Austria SIR T2 program.

The T2 proposal cloud is frozen at theta zero. Its theta-dependent target is
the issued T1 normalized retained marginal times the physical transition and
observation factors. The proposal and coordinate frame remain fixed.
"""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
)


DTYPE = tf.float64
PARAMETER_DIM = 3
T2_SCORE_TARGET_ID = "lane_b_t2_carried_t1_marginal_transition_observation_v1"


def physical_z1_to_parent_local_prefix(
    child: LaneBParameterChild, z1: tf.Tensor
) -> tf.Tensor:
    """Apply the frozen parent prefix frame to physical retained states."""

    physical = tf.convert_to_tensor(z1, DTYPE)
    if physical.shape.rank != 2 or physical.shape[1] != 18:
        raise ValueError("z1 must have shape [sample,18]")
    parent = child.parent
    prefix_matrix = parent.frame.matrix[:18, :18]
    lower_left = parent.frame.matrix[:18, 18:]
    tf.debugging.assert_equal(
        lower_left,
        tf.zeros_like(lower_left),
        message="T1 retained prefix depends on marginalized coordinates",
    )
    return tf.transpose(
        tf.linalg.triangular_solve(
            prefix_matrix,
            tf.transpose(physical) - parent.frame.mu[:18, tf.newaxis],
            lower=True,
        )
    )


def t2_target_log_value_and_manual_score(
    child: LaneBParameterChild,
    theta: tf.Tensor,
    joint_points: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Return the target log value and total carried score at fixed rows."""

    if len(child.parent_cores) != 36:
        raise TypeError("T2 carried score requires a 36-axis T1 child")
    parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    points = tf.convert_to_tensor(joint_points, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != 36:
        raise ValueError("joint_points must have shape [sample,36]")
    z2 = points[:, :18]
    z1 = points[:, 18:]
    local_z1 = physical_z1_to_parent_local_prefix(child, z1)
    previous_log_relative, previous_score = child.prefix_log_marginal_and_score(
        parameters, local_z1
    )
    model = latent_preclip_zhao_cui_sir_austria_model()
    _states, observations, _all = generate_sealed_lane_b_dataset()
    transition = model.transition_log_density(parameters, z1, z2, 2)
    likelihood = model.observation_log_density(
        parameters, z2, observations[1], 2
    )
    transition_score = model.transition_log_density_parameter_score(
        parameters, z1, z2, 2
    )
    observation_score = model.observation_log_density_parameter_score(
        parameters, z2, observations[1], 2
    )
    score = previous_score + transition_score + observation_score
    log_value = previous_log_relative + transition + likelihood
    tf.debugging.assert_all_finite(log_value, "T2 carried target log value")
    tf.debugging.assert_all_finite(score, "T2 carried target score")
    return {
        "log_value": log_value,
        "score": score,
        "previous_log_relative": previous_log_relative,
        "previous_score": previous_score,
        "transition": transition,
        "transition_score": transition_score,
        "likelihood": likelihood,
        "observation_score": observation_score,
    }


__all__ = [
    "T2_SCORE_TARGET_ID",
    "physical_z1_to_parent_local_prefix",
    "t2_target_log_value_and_manual_score",
]
