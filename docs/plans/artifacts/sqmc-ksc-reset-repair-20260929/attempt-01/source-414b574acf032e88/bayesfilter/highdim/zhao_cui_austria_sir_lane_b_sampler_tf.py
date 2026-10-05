"""Correctly scored finite retained sampler for the Lane-B T1 artifact."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import AlgebraicMap
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_boundary_tf import (
    KEEP_AXES,
    LaneBT1RetainedBoundary,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import LaneBT1Artifact


DTYPE = tf.float64
SAMPLER_ID = "lane_b_retained_gauss3_interval_mass_piecewise_linear_cdf_v1"
SAMPLER_CLASSIFICATION = "extension_or_invention_finite_grid_proposal"


def _gather_rows(table: tf.Tensor, columns: tf.Tensor) -> tf.Tensor:
    rows = tf.range(tf.shape(table)[0], dtype=tf.int32)
    return tf.gather_nd(table, tf.stack([rows, columns], axis=1))


@dataclass(frozen=True)
class LaneBRetainedSample:
    """A frozen retained draw with its actual proposal and TT correction."""

    reference_uniforms: tf.Tensor
    local_points: tf.Tensor
    physical_points: tf.Tensor
    proposal_log_density: tf.Tensor
    target_log_density: tf.Tensor
    correction_log_weights: tf.Tensor
    raw_conditional_mass_residuals: tf.Tensor

    def __post_init__(self) -> None:
        reference = tf.convert_to_tensor(self.reference_uniforms, DTYPE)
        local = tf.convert_to_tensor(self.local_points, DTYPE)
        physical = tf.convert_to_tensor(self.physical_points, DTYPE)
        proposal = tf.convert_to_tensor(self.proposal_log_density, DTYPE)
        target = tf.convert_to_tensor(self.target_log_density, DTYPE)
        correction = tf.convert_to_tensor(self.correction_log_weights, DTYPE)
        residuals = tf.convert_to_tensor(self.raw_conditional_mass_residuals, DTYPE)
        if reference.shape.rank != 2 or reference.shape[0] != len(KEEP_AXES):
            raise ValueError("reference_uniforms must have shape [18,sample]")
        sample_count = int(reference.shape[1])
        if local.shape != (sample_count, len(KEEP_AXES)):
            raise ValueError("local_points must have shape [sample,18]")
        if physical.shape != local.shape:
            raise ValueError("physical_points must match local_points")
        if proposal.shape != (sample_count,) or target.shape != proposal.shape:
            raise ValueError("sample log densities must have shape [sample]")
        if correction.shape != proposal.shape:
            raise ValueError("correction_log_weights must match proposal")
        if residuals.shape != (len(KEEP_AXES), sample_count):
            raise ValueError("mass residuals must have shape [18,sample]")
        for name, value in (
            ("reference_uniforms", reference),
            ("local_points", local),
            ("physical_points", physical),
            ("proposal_log_density", proposal),
            ("target_log_density", target),
            ("correction_log_weights", correction),
            ("raw_conditional_mass_residuals", residuals),
        ):
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
        tf.debugging.assert_near(correction, target - proposal, atol=2e-12)
        object.__setattr__(self, "reference_uniforms", reference)
        object.__setattr__(self, "local_points", local)
        object.__setattr__(self, "physical_points", physical)
        object.__setattr__(self, "proposal_log_density", proposal)
        object.__setattr__(self, "target_log_density", target)
        object.__setattr__(self, "correction_log_weights", correction)
        object.__setattr__(self, "raw_conditional_mass_residuals", residuals)

    def manifest_payload(self) -> Mapping[str, object]:
        shifted = self.correction_log_weights - tf.reduce_max(
            self.correction_log_weights
        )
        normalized = tf.exp(shifted) / tf.reduce_sum(tf.exp(shifted))
        ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
        return {
            "sample_count": int(self.reference_uniforms.shape[1]),
            "reference_uniforms_sha256": tensor_sha256(self.reference_uniforms),
            "local_points_sha256": tensor_sha256(self.local_points),
            "physical_points_sha256": tensor_sha256(self.physical_points),
            "proposal_log_density_sha256": tensor_sha256(self.proposal_log_density),
            "target_log_density_sha256": tensor_sha256(self.target_log_density),
            "correction_log_weights_sha256": tensor_sha256(
                self.correction_log_weights
            ),
            "maximum_raw_conditional_mass_residual": tf.reduce_max(
                self.raw_conditional_mass_residuals
            ),
            "correction_log_weight_min": tf.reduce_min(
                self.correction_log_weights
            ),
            "correction_log_weight_max": tf.reduce_max(
                self.correction_log_weights
            ),
            "correction_effective_sample_size": ess,
        }


@dataclass(frozen=True)
class LaneBRetainedGridSampler:
    """Invert and score the exact finite grid law on T1 retained axes."""

    artifact: LaneBT1Artifact

    def __post_init__(self) -> None:
        lower_left = self.artifact.frame.matrix[:18, 18:]
        tf.debugging.assert_equal(
            lower_left,
            tf.zeros_like(lower_left),
            message="retained physical prefix depends on marginalized coordinates",
        )

    @property
    def grid_size(self) -> int:
        return int(self.artifact.settings.cdf_grid_size)

    def _grid(self) -> tf.Tensor:
        return tf.linspace(tf.constant(-1.0, DTYPE), tf.constant(1.0, DTYPE), self.grid_size)

    def _conditional_table(
        self, axis: int, prefixes: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # FP64 GPU paired-core contractions are not a numerical authority for
        # this route. Tables are setup/reference work and stay explicitly CPU.
        with tf.device("/CPU:0"):
            prefix = tf.convert_to_tensor(prefixes, DTYPE)
            if prefix.shape.rank != 2 or prefix.shape[1] != axis:
                raise ValueError("prefixes have the wrong retained dimension")
            sample_count = int(prefix.shape[0])
            grid = self._grid()
            if self.grid_size % 2 != 1:
                raise ValueError("grid must include the Lagrange element breakpoint")
            tf.debugging.assert_equal(grid[self.grid_size // 2], tf.constant(0.0, DTYPE))
            left = grid[:-1]
            right = grid[1:]
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            gauss_nodes = tf.constant(
                [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)], DTYPE
            )
            gauss_weights = tf.constant([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], DTYPE)
            reference_nodes = (
                midpoint[:, tf.newaxis]
                + half_width[:, tf.newaxis] * gauss_nodes[tf.newaxis, :]
            )
            flat_local_nodes = AlgebraicMap(1.0).from_reference(
                tf.reshape(reference_nodes, [-1])
            )
            node_count = (self.grid_size - 1) * 3
            tiled_prefix = tf.repeat(
                prefix[:, tf.newaxis, :], repeats=node_count, axis=1
            )
            tiled_nodes = tf.broadcast_to(
                flat_local_nodes[tf.newaxis, :, tf.newaxis],
                [sample_count, node_count, 1],
            )
            points = tf.reshape(
                tf.concat([tiled_prefix, tiled_nodes], axis=2),
                [sample_count * node_count, axis + 1],
            )
            density = self.artifact.density()
            numerator = tf.reshape(
                density.normalized_marginal_density_values(
                    tuple(range(axis + 1)), points
                ),
                [sample_count, self.grid_size - 1, 3],
            )
            if axis == 0:
                denominator = tf.ones([sample_count], DTYPE)
            else:
                denominator = density.normalized_marginal_density_values(
                    tuple(range(axis)), prefix
                )
            conditional = numerator / denominator[:, tf.newaxis, tf.newaxis]
            # The density is relative to dnu=du/2. Gauss-3 exactly integrates
            # each degree-four conditional interval for the order-two basis.
            interval_mass = 0.5 * half_width[tf.newaxis, :] * tf.einsum(
                "nij,j->ni", conditional, gauss_weights
            )
        tf.debugging.assert_positive(interval_mass, "grid interval mass must be positive")
        raw_mass = tf.reduce_sum(interval_mass, axis=1)
        tf.debugging.assert_positive(raw_mass, "conditional grid mass must be positive")
        interval_probability = interval_mass / raw_mass[:, tf.newaxis]
        cdf = tf.concat(
            [
                tf.zeros([sample_count, 1], DTYPE),
                tf.cumsum(interval_probability, axis=1),
            ],
            axis=1,
        )
        return grid, cdf, interval_probability, tf.abs(raw_mass - 1.0)

    def _invert_axis(
        self, axis: int, prefixes: tf.Tensor, uniforms: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        target = tf.convert_to_tensor(uniforms, DTYPE)
        grid, cdf, probability, residual = self._conditional_table(axis, prefixes)
        right = tf.searchsorted(cdf, target[:, tf.newaxis], side="right")[:, 0]
        right = tf.clip_by_value(right, 1, self.grid_size - 1)
        left = right - 1
        cdf_left = _gather_rows(cdf, left)
        interval_probability = _gather_rows(probability, left)
        delta = tf.gather(grid, right) - tf.gather(grid, left)
        fraction = (target - cdf_left) / interval_probability
        reference = tf.gather(grid, left) + fraction * delta
        local = AlgebraicMap(1.0).from_reference(reference)
        log_q_reference = tf.math.log(interval_probability / delta)
        return local, log_q_reference, residual

    def inverse(self, reference_uniforms: tf.Tensor) -> LaneBRetainedSample:
        reference = tf.convert_to_tensor(reference_uniforms, DTYPE)
        if reference.shape.rank != 2 or reference.shape[0] != len(KEEP_AXES):
            raise ValueError("reference_uniforms must have shape [18,sample]")
        if not bool(tf.reduce_all((reference > 0.0) & (reference < 1.0)).numpy()):
            raise ValueError("reference uniforms must lie strictly inside (0,1)")
        sample_count = int(reference.shape[1])
        prefixes = tf.zeros([sample_count, 0], DTYPE)
        local_columns = []
        log_q_reference = tf.zeros([sample_count], DTYPE)
        residuals = []
        for axis in KEEP_AXES:
            local, log_q_axis, residual = self._invert_axis(
                axis, prefixes, reference[axis]
            )
            local_columns.append(local)
            prefixes = tf.concat([prefixes, local[:, tf.newaxis]], axis=1)
            log_q_reference += log_q_axis
            residuals.append(residual)
        local_points = tf.stack(local_columns, axis=1)
        frame = self.artifact.frame
        prefix_matrix = frame.matrix[:18, :18]
        physical = tf.transpose(
            tf.linalg.matmul(prefix_matrix, tf.transpose(local_points))
            + frame.mu[:18, tf.newaxis]
        )
        log_du_dr = tf.reduce_sum(
            AlgebraicMap(1.0).domain_to_reference_log_density(local_points), axis=1
        )
        log_det = tf.math.log(tf.abs(tf.linalg.det(prefix_matrix)))
        proposal = log_q_reference + log_du_dr - log_det
        target = LaneBT1RetainedBoundary(self.artifact).api_log_physical_density(
            physical
        )
        return LaneBRetainedSample(
            reference_uniforms=reference,
            local_points=local_points,
            physical_points=physical,
            proposal_log_density=proposal,
            target_log_density=target,
            correction_log_weights=target - proposal,
            raw_conditional_mass_residuals=tf.stack(residuals, axis=0),
        )

    def forward_and_log_proposal(
        self, local_points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        local = tf.convert_to_tensor(local_points, DTYPE)
        if local.shape.rank != 2 or local.shape[1] != len(KEEP_AXES):
            raise ValueError("local_points must have shape [sample,18]")
        sample_count = int(local.shape[0])
        uniforms = []
        log_q_reference = tf.zeros([sample_count], DTYPE)
        residuals = []
        grid = self._grid()
        for axis in KEEP_AXES:
            prefix = local[:, :axis]
            _grid, cdf, probability, residual = self._conditional_table(axis, prefix)
            reference = AlgebraicMap(1.0).to_reference(local[:, axis])
            right = tf.searchsorted(grid, reference, side="right")
            right = tf.clip_by_value(right, 1, self.grid_size - 1)
            left = right - 1
            cdf_left = _gather_rows(cdf, left)
            interval_probability = _gather_rows(probability, left)
            delta = tf.gather(grid, right) - tf.gather(grid, left)
            fraction = (reference - tf.gather(grid, left)) / delta
            uniforms.append(cdf_left + fraction * interval_probability)
            log_q_reference += tf.math.log(interval_probability / delta)
            residuals.append(residual)
        prefix_matrix = self.artifact.frame.matrix[:18, :18]
        log_du_dr = tf.reduce_sum(
            AlgebraicMap(1.0).domain_to_reference_log_density(local), axis=1
        )
        log_det = tf.math.log(tf.abs(tf.linalg.det(prefix_matrix)))
        return (
            tf.stack(uniforms, axis=0),
            log_q_reference + log_du_dr - log_det,
            tf.stack(residuals, axis=0),
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "sampler_id": SAMPLER_ID,
            "classification": SAMPLER_CLASSIFICATION,
            "source_artifact_identity": self.artifact.identity.hash.value,
            "retained_axes": KEEP_AXES,
            "grid_size": self.grid_size,
            "cdf_semantics": "gauss3_exact_piecewise_quartic_interval_mass_piecewise_linear_cdf",
            "proposal_density_semantics": "exact_piecewise_constant_cdf_slope_with_coordinate_jacobians",
            "correction_semantics": "log_normalized_tt_retained_density_minus_log_exact_grid_proposal",
            "production_kr_closure": False,
            "paper_anchors": (
                ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807-924",
            ),
            "author_source_anchors": (
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_irt_reference.m:43-71",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:19-85",
            ),
            "nonclaims": (
                "no production KR closure",
                "no source-faithful numerical CDF claim",
                "no T2 value, score, T20, or HMC readiness",
            ),
        }


def retained_sampler_workspace_estimate_bytes(
    *,
    sample_count: int,
    grid_size: int,
    max_rank: int,
    dimension: int = 18,
    quadrature_order: int = 3,
) -> int:
    """Conservative largest-axis live tensor estimate for the B2 sampler."""

    n = int(sample_count)
    g = int(grid_size)
    r = int(max_rank)
    d = int(dimension)
    q = int(quadrature_order)
    if min(n, g, r, d, q) <= 0:
        raise ValueError("workspace dimensions must be positive")
    evaluation_nodes = (g - 1) * q
    scalar_slots = n * evaluation_nodes * (2 * d + 4 + 4 * r * r) + n * (
        3 * g + d + 8
    )
    return int(2 * DTYPE.size * scalar_slots)


__all__ = [
    "LaneBRetainedGridSampler",
    "LaneBRetainedSample",
    "SAMPLER_CLASSIFICATION",
    "SAMPLER_ID",
    "retained_sampler_workspace_estimate_bytes",
]
