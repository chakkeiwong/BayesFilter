"""Parity oracle: batch general-route port vs single-cloud general implementation.

R4/R5 gates of
docs/plans/bayesfilter-guardrail-general-route-rectification-plan-2026-08-20.md:
the batch-size-1 output of the batch higher-moment correction (with pairwise
and coordinate-cap controls enabled) must match the general single-cloud
implementation `higher_moment_shape_jvp` on identical inputs, and the batch
route must expose the full general capability surface (fork-regrowth guard).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import inspect

import tensorflow as tf

from bayesfilter.highdim import cubature_genut_batch_tf as batch
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp


def _make_cloud(seed: int, count: int = 96, dim: int = 4):
    generator = tf.random.Generator.from_seed(seed)
    source = generator.normal([count, dim], dtype=tf.float32)
    skewer = generator.normal([count, dim], dtype=tf.float32)
    source = source + 0.35 * tf.square(skewer) - 0.35
    weights_raw = generator.uniform([count], 0.5, 1.5, dtype=tf.float32)
    weights = weights_raw / tf.reduce_sum(weights_raw)
    points = generator.normal([count, dim], dtype=tf.float32)
    return source, weights, points


CONTROLS = dict(
    correction_steps=2,
    strength=0.2,
    floor=1.0e-5,
    lm_damping=1.0e-2,
    lm_scale_floor=1.0e-4,
    trust_radius=0.5,
    pairwise_correction_steps=2,
    pairwise_strength=0.02,
    pairwise_floor=1.0e-5,
    pairwise_particle_rms_cap=2.0,
    coordinate_cap=0.98,
    coordinate_cap_power=8,
)


def _run_batch(source, weights, points):
    return batch._higher_moment_batch_value(  # noqa: SLF001
        source[None, :, :],
        weights[None, :],
        points[None, :, :],
        **CONTROLS,
    )


def _run_general(source, weights, points):
    count, dim = source.shape
    zero_cloud_tangent = tf.zeros([count, dim, 1], tf.float32)
    zero_weight_tangent = tf.zeros([count, 1], tf.float32)
    return higher_moment_shape_jvp(
        source,
        weights,
        zero_cloud_tangent,
        zero_weight_tangent,
        points,
        zero_cloud_tangent,
        correction_steps=CONTROLS["correction_steps"],
        strength=CONTROLS["strength"],
        floor=CONTROLS["floor"],
        diagonal_lm_damping=CONTROLS["lm_damping"],
        diagonal_lm_scale_floor=CONTROLS["lm_scale_floor"],
        diagonal_trust_radius=CONTROLS["trust_radius"],
        pairwise_correction_steps=CONTROLS["pairwise_correction_steps"],
        pairwise_strength=CONTROLS["pairwise_strength"],
        pairwise_floor=CONTROLS["pairwise_floor"],
        pairwise_particle_rms_cap=CONTROLS["pairwise_particle_rms_cap"],
        coordinatewise_standardized_cap=CONTROLS["coordinate_cap"],
        coordinatewise_standardized_cap_power=CONTROLS["coordinate_cap_power"],
    )


def test_batch_size_one_parity_with_general_route():
    for seed in (7, 11, 23):
        source, weights, points = _make_cloud(seed)
        batch_result = _run_batch(source, weights, points)
        general_result = _run_general(source, weights, points)
        batch_particles = batch_result["particles"][0]
        general_particles = general_result["particles"]
        assert bool(
            tf.reduce_all(tf.math.is_finite(batch_particles)).numpy()
        ), f"seed {seed}: batch particles nonfinite"
        assert bool(
            tf.reduce_all(tf.math.is_finite(general_particles)).numpy()
        ), f"seed {seed}: general particles nonfinite"
        difference = float(
            tf.reduce_max(tf.abs(batch_particles - general_particles)).numpy()
        )
        scale = float(
            tf.reduce_max(tf.abs(general_particles)).numpy()
        )
        # FP32 op-order tolerance: the two implementations contract moments in
        # different einsum orders. Declared tolerance, recorded max diff.
        assert difference <= 5.0e-4 * max(scale, 1.0), (
            f"seed {seed}: parity gap {difference} exceeds tolerance "
            f"(scale {scale})"
        )


def test_batch_route_default_off_matches_diagonal_only():
    source, weights, points = _make_cloud(31)
    with_defaults = batch._higher_moment_batch_value(  # noqa: SLF001
        source[None, :, :],
        weights[None, :],
        points[None, :, :],
        correction_steps=2,
        strength=0.2,
        floor=1.0e-5,
        lm_damping=0.0,
        lm_scale_floor=1.0e-6,
        trust_radius=0.0,
    )
    explicit_off = batch._higher_moment_batch_value(  # noqa: SLF001
        source[None, :, :],
        weights[None, :],
        points[None, :, :],
        correction_steps=2,
        strength=0.2,
        floor=1.0e-5,
        lm_damping=0.0,
        lm_scale_floor=1.0e-6,
        trust_radius=0.0,
        pairwise_correction_steps=0,
        pairwise_strength=0.0,
        coordinate_cap=0.0,
    )
    assert bool(
        tf.reduce_all(
            tf.equal(with_defaults["particles"], explicit_off["particles"])
        ).numpy()
    ), "default-off controls must be bitwise inert"


def test_batch_route_exposes_general_capability_surface():
    """Fork-regrowth guard: the batch value route must accept the general
    capability controls (pairwise correction, radial step cap, coordinate
    clamp, LM damping, trust radius)."""

    signature = inspect.signature(
        batch._higher_moment_batch_value  # noqa: SLF001
    )
    required = {
        "correction_steps",
        "strength",
        "floor",
        "lm_damping",
        "lm_scale_floor",
        "trust_radius",
        "pairwise_correction_steps",
        "pairwise_strength",
        "pairwise_floor",
        "pairwise_particle_rms_cap",
        "coordinate_cap",
        "coordinate_cap_power",
    }
    missing = required - set(signature.parameters)
    assert not missing, f"batch route lost general capabilities: {missing}"
