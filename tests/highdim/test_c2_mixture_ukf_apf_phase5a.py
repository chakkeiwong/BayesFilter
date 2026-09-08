"""Focused Phase 5A representation-contract regressions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.bases import HermiteBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import DensityMeasure, MassMeasure, MeasureConvention
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.tt import TTCore


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_mixture_ukf_apf_phase5a_hermite_20260904.py"


def _driver_module():
    spec = importlib.util.spec_from_file_location("phase5a_driver", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 5A driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._load_modules()
    return module


def _convention():
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="standard_normal",
        physical_coordinate_name="x",
        reference_coordinate_name="u",
    )


def test_squared_tt_gram_constant_channel_is_one():
    module = _driver_module()
    convention = _convention()
    basis = ProductBasis([HermiteBasis1D(2) for _ in range(4)], convention)
    cores = tuple(
        TTCore(tf.one_hot([0], 3, dtype=tf.float64)[:, :, None])
        for _ in range(4)
    )
    value = module._gram_squared_normalizer(cores, basis)
    tf.debugging.assert_near(value, tf.constant(1.0, tf.float64), atol=1.0e-14)


def test_reference_target_square_root_scaling_is_exact():
    module = _driver_module()
    rows = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, -0.5, 0.25, -1.25]], tf.float64)
    coordinate_map = AffineCoordinateMap(
        offset=tf.constant([0.2, -0.1, 0.3, -0.4], tf.float64),
        matrix=tf.eye(4, dtype=tf.float64),
    )

    @tf.function(input_signature=[tf.TensorSpec([2, 4], tf.float64)], autograph=False)
    def target(_physical):
        return tf.constant([0.0, -1.0], tf.float64)

    log_h, sqrt_target, shift = module._reference_target_values(
        rows,
        coordinate_map=coordinate_map,
        target_kernel=target,
    )
    tf.debugging.assert_near(
        tf.square(sqrt_target), tf.exp(log_h - shift), atol=1.0e-14
    )
    tf.debugging.assert_near(
        shift,
        tf.reduce_logsumexp(log_h) - tf.math.log(tf.cast(2, tf.float64)),
        atol=1.0e-14,
    )
