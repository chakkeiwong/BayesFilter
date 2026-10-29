"""Exact reference draws and placement for posterior cloud preparation."""

import dataclasses
import hashlib

import pytest
import tensorflow as tf

from bayesfilter.inference.posterior_cloud_preparation_tf import (
    PosteriorCloudPreparation,
    posterior_seed_keys,
)
from bayesfilter.inference.posterior_local_initializer import (
    PosteriorLocalInitializerConfig,
    _cloud_row_counts,
    _sample_ball,
)
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from bayesfilter.inference.quadratic_geometry_full_tf import prepare_geometry_inputs
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import save
from tests.test_filter_repair_posterior_movement import verified_reference_tree

D = tf.float64


@pytest.mark.parametrize("dimension", [1, 3])
def test_exact_original_clouds_and_cpu_placement(dimension, request):
    _, hashes = verified_reference_tree()
    config = PosteriorLocalInitializerConfig(max_movement_attempts=3, max_curvature_attempts=2,
        training_rows_per_replicate=3 * dimension, selection_rows_per_replicate=2 * dimension,
        audit_rows=2 * dimension + 1, curvature_radius=.08, seed=(31, 43))
    movement_config = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=12,
        pilot_direction_count=6, trust_radius=.3, seed=(17, 23))
    program = PosteriorCloudPreparation(dimension, config, movement_config)
    observations, hlos = [], []
    placement = "/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"
    for shift in (0, 19, 0):
        cfg = dataclasses.replace(config, seed=(31 + shift, 43), curvature_radius=.08 + shift * .001)
        move = dataclasses.replace(movement_config, seed=(17, 23 + shift), trust_radius=.3 + shift * .002)
        expected_movement = [prepare_geometry_inputs(dimension,
            dataclasses.replace(move, seed=(move.seed[0], move.seed[1] + i))) for i in range(cfg.max_movement_attempts)]
        rows = _cloud_row_counts(cfg, dimension)
        counts = [rows[0]] * cfg.replicate_count + [rows[1]] * cfg.replicate_count + [rows[2]]
        capacity = max(counts)
        curvature = tf.stack([tf.stack([tf.pad(_sample_ball(n, dimension, radius=cfg.curvature_radius,
            seed=(cfg.seed[0] + a, cfg.seed[1] + 1000 * a + p)), [[0, capacity - n], [0, 0]])
            for p, n in enumerate(counts)]) for a in range(cfg.max_curvature_attempts)])
        expected = {"directions": tf.stack([row[0] for row in expected_movement]),
            "movement_offsets": tf.stack([row[1] for row in expected_movement]),
            "permutation_keys": tf.stack([row[2] for row in expected_movement]), "curvature_offsets": curvature}
        keys = posterior_seed_keys(dimension, cfg, move)
        with tf.device(placement):
            operands = (*keys, tf.constant(move.trust_radius, D), tf.constant(cfg.curvature_radius, D))
            with tf.GradientTape() as tape:
                tape.watch(operands[2:])
                actual = program(*operands)
                scalar = tf.reduce_sum(actual["movement_offsets"]) + tf.reduce_sum(actual["curvature_offsets"])
            assert tape.gradient(scalar, operands[2:]) == (None, None)
        report = {}
        for name, tensor in actual.items():
            assert "device:CPU:0" in tensor.device
            assert bool(tf.reduce_all(tensor == expected[name])), name
            report[name] = {"shape": tensor.shape.as_list(), "device": tensor.device,
                "sha256": hashlib.sha256(tf.io.serialize_tensor(tensor).numpy()).hexdigest()}
        observations.append(report)
        with tf.device("/CPU:0"):
            hlos.append(stable_hlo(program.compiled.experimental_get_compiler_ir(*operands)(stage="hlo")))
    save(request, f"posterior-cloud-preparation-{dimension}.json", {
        "reference_dependency_sha256": hashes, "stream_id": program.stream_id,
        "outer_placement": placement, "comparisons": observations,
        "target_evaluations": 0, "target_interface_present": False,
        "trace_count": program.compiled.experimental_get_tracing_count(),
        "stable_hlo": hlos[0] == hlos[1] == hlos[2]})
    assert observations[0] == observations[2] and observations[0] != observations[1]
    assert hlos[0] == hlos[1] == hlos[2]
    assert program.compiled.experimental_get_tracing_count() == 1
