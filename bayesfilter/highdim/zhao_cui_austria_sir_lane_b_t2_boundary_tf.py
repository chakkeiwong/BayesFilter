"""Independent T1 retained-marginal and T2 input boundary for Lane B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import AlgebraicMap
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import LaneBT1Artifact


DTYPE = tf.float64
T2_BOUNDARY_ID = "zhao_cui_austria_sir_lane_b_t2_previous_marginal_boundary_v1"
KEEP_AXES = tuple(range(18))
INTEGRATED_AXES = tuple(range(18, 36))


def _physical_z1_to_local_prefix(
    artifact: LaneBT1Artifact,
    z1: tf.Tensor,
) -> tf.Tensor:
    physical = tf.convert_to_tensor(z1, DTYPE)
    if physical.shape.rank != 2 or physical.shape[1] != 18:
        raise ValueError("z1 must have shape [sample,18]")
    prefix_matrix = artifact.frame.matrix[:18, :18]
    return tf.transpose(
        tf.linalg.triangular_solve(
            prefix_matrix,
            tf.transpose(physical) - artifact.frame.mu[:18, tf.newaxis],
            lower=True,
        )
    )


def _retained_log_measure_to_physical(
    artifact: LaneBT1Artifact,
    local_prefix: tf.Tensor,
    log_relative_density: tf.Tensor,
) -> tf.Tensor:
    local = tf.convert_to_tensor(local_prefix, DTYPE)
    reference = AlgebraicMap(1.0).to_reference(local)
    log_du_dr = tf.reduce_sum(
        AlgebraicMap(1.0).domain_to_reference_log_density(local), axis=1
    )
    prefix_log_det = tf.math.log(tf.abs(tf.linalg.det(artifact.frame.matrix[:18, :18])))
    return (
        tf.convert_to_tensor(log_relative_density, DTYPE)
        - tf.constant(18.0 * math.log(2.0), DTYPE)
        + log_du_dr
        - prefix_log_det
    )


def independent_prefix_marginal_relative_density(
    artifact: LaneBT1Artifact,
    local_prefix: tf.Tensor,
) -> tf.Tensor:
    """Contract a right paired-core environment independently of SquaredTT."""

    points = tf.convert_to_tensor(local_prefix, DTYPE)
    if points.shape.rank != 2 or points.shape[1] != 18:
        raise ValueError("local_prefix must have shape [sample,18]")
    density = artifact.density()
    cores = density.sqrt_tt.cores
    bases = density.sqrt_tt.product_basis.bases
    right = tf.ones([1, 1], DTYPE)
    active_measure = density.measure_convention.mass_measure
    for axis in reversed(INTEGRATED_AXES):
        core = cores[axis].values
        mass = bases[axis].mass_matrix(active_measure)
        right = tf.einsum("alb,AmB,lm,bB->aA", core, core, mass, right)

    sample_count = tf.shape(points)[0]
    left = tf.ones([sample_count, 1, 1], DTYPE)
    for axis in KEEP_AXES:
        core = cores[axis].values
        basis = density.sqrt_tt.product_basis.evaluate_axis(axis, points[:, axis])
        evaluated = tf.einsum("nl,alb->nab", basis, core)
        left = tf.einsum("naA,nab,nAB->nbB", left, evaluated, evaluated)
    square_marginal = tf.einsum("naA,aA->n", left, right)
    relative = (square_marginal + density.tau) / density.normalizer()
    tf.debugging.assert_positive(relative, "independent retained density must be positive")
    tf.debugging.assert_all_finite(relative, "independent retained density must be finite")
    return relative


def independent_total_mass_from_cut(artifact: LaneBT1Artifact) -> tf.Tensor:
    """Integrate both sides of the independent cut contraction."""

    density = artifact.density()
    cores = density.sqrt_tt.cores
    bases = density.sqrt_tt.product_basis.bases
    active_measure = density.measure_convention.mass_measure
    right = tf.ones([1, 1], DTYPE)
    for axis in reversed(INTEGRATED_AXES):
        core = cores[axis].values
        right = tf.einsum(
            "alb,AmB,lm,bB->aA",
            core,
            core,
            bases[axis].mass_matrix(active_measure),
            right,
        )
    left = tf.ones([1, 1], DTYPE)
    for axis in KEEP_AXES:
        core = cores[axis].values
        left = tf.einsum(
            "aA,alb,AmB,lm->bB",
            left,
            core,
            core,
            bases[axis].mass_matrix(active_measure),
        )
    return tf.einsum("aA,aA->", left, right) + density.tau


@dataclass(frozen=True)
class LaneBT1RetainedBoundary:
    """Repository-owned retained prefix issued from a selected T1 artifact."""

    artifact: LaneBT1Artifact

    def api_log_physical_density(self, z1: tf.Tensor) -> tf.Tensor:
        local = _physical_z1_to_local_prefix(self.artifact, z1)
        relative = self.artifact.density().normalized_marginal_density_values(
            KEEP_AXES, local
        )
        return _retained_log_measure_to_physical(
            self.artifact, local, tf.math.log(relative)
        )

    def independent_log_physical_density(self, z1: tf.Tensor) -> tf.Tensor:
        local = _physical_z1_to_local_prefix(self.artifact, z1)
        relative = independent_prefix_marginal_relative_density(self.artifact, local)
        return _retained_log_measure_to_physical(
            self.artifact, local, tf.math.log(relative)
        )

    def t2_log_target(self, z2: tf.Tensor, z1: tf.Tensor) -> Mapping[str, tf.Tensor]:
        next_state = tf.convert_to_tensor(z2, DTYPE)
        previous = tf.convert_to_tensor(z1, DTYPE)
        if next_state.shape != previous.shape or next_state.shape.rank != 2:
            raise ValueError("z2 and z1 must have matching [sample,18] shapes")
        model = latent_preclip_zhao_cui_sir_austria_model()
        _states, observations, _all = generate_sealed_lane_b_dataset()
        theta = tf.zeros([3], DTYPE)
        previous_api = self.api_log_physical_density(previous)
        previous_independent = self.independent_log_physical_density(previous)
        transition = model.transition_log_density(theta, previous, next_state, 2)
        likelihood = model.observation_log_density(
            theta, next_state, observations[1], 2
        )
        return {
            "previous_api": previous_api,
            "previous_independent": previous_independent,
            "transition": transition,
            "likelihood": likelihood,
            "api_total": previous_api + transition + likelihood,
            "independent_total": previous_independent + transition + likelihood,
        }

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "boundary_id": T2_BOUNDARY_ID,
            "source_artifact_identity": self.artifact.identity.hash.value,
            "source_artifact_value": self.artifact.value(),
            "joint_axis_order_t1": ("z1", "z0"),
            "retained_state": "z1",
            "keep_axes": KEEP_AXES,
            "integrated_axes": INTEGRATED_AXES,
            "t2_joint_axis_order": ("z2", "z1"),
            "t2_event_order": "retained_z1_then_transition_to_z2_then_observe_sealed_y2",
            "relative_measure": "uniform_probability_measure_on_u1_in_[-1,1]^18",
            "physical_conversion": (
                "log p_z1=log p_nu-18log2+log|du/dr|-log|det L11|"
            ),
            "classification": "source_faithful_marginal_operation_with_fixed_hmc_adaptation",
            "t2_training_status": "not_started_boundary_only",
            "nonclaims": (
                "no T2 fit or value",
                "no score or HMC readiness",
                "no T20 or production readiness",
            ),
        }


__all__ = [
    "INTEGRATED_AXES",
    "KEEP_AXES",
    "LaneBT1RetainedBoundary",
    "T2_BOUNDARY_ID",
    "independent_prefix_marginal_relative_density",
    "independent_total_mass_from_cut",
]
