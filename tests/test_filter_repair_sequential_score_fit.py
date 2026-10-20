"""Pinned complete-fit and frozen-design checks for sequential preparation."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as candidate
from bayesfilter.inference import sequential_score_fit_tf as native

D = tf.float64


def _pinned(relative, name):
    source = subprocess.check_output(["git", "show",
        f"3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:{relative}"], text=True)
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102 - pinned reference only
    return module


@pytest.fixture(scope="module")
def baseline():
    module = _pinned("bayesfilter/inference/sequential_map_covariance.py", "score_fit_pinned")
    incumbent = _pinned("bayesfilter/inference/_exact_incumbent.py", "score_fit_incumbent_pinned")
    module.candidates_from_rows = incumbent.candidates_from_rows
    module.select_exact_incumbent = incumbent.select_exact_incumbent
    return module


def _targets(indefinite=False, nonlinear=0.):
    precision = tf.constant([[-1. if indefinite else 2., .2], [.2, 4.]], D)

    def scalar(point):
        score = -tf.linalg.matvec(precision, point)
        return (.5 * tf.reduce_sum(point * score) - nonlinear * tf.reduce_sum(point ** 4),
            score - 4 * nonlinear * point ** 3)

    def batch(points):
        scores = -tf.einsum("ij,bj->bi", precision, points)
        return (.5 * tf.reduce_sum(points * scores, 1) - nonlinear * tf.reduce_sum(points ** 4, 1),
            scores - 4 * nonlinear * points ** 3)

    return scalar, batch


def _assert_payload(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], (str, bool)) or expected[key] is None:
            assert actual[key] == expected[key], key
        else:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=1e-10, err_msg=key)


@pytest.mark.parametrize("pair_disjoint", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("indefinite,nonlinear", [(False, 0.), (True, .2)])
def test_complete_seeded_fit_preserves_all_fields_and_fixed_signature(baseline, pair_disjoint, batched, indefinite, nonlinear):
    scalar, batch = _targets(indefinite, nonlinear)
    center, scale = tf.constant([.13, -.22], D), tf.constant([.7, 1.2], D)
    center_score = scalar(center)[1]
    cfg = candidate.SequentialMapCovarianceConfig(pair_disjoint_score_holdout=pair_disjoint,
        regression_sample_count=16, terminal_sample_count=16)
    options = {"dimension": 2, "radius": .3, "sample_count": 16, "seed": (2026, 919),
        "config": cfg, "evaluations": 3, "batched_value_and_score_fn": batch if batched else None}
    expected, old_count = baseline._fit_score_curvature(scalar, center, center_score, scale, **options)
    actual, count = candidate._fit_score_curvature(scalar, center, center_score, scale, **options)
    assert count == old_count == 19
    _assert_payload(actual, expected)
    train, heldout = native.partition_schema(16, cfg.holdout_fraction, pair_disjoint=pair_disjoint)
    program = native.score_fit_program(scalar, batch if batched else None, 16, 2, train, heldout)
    inputs = (center, center_score, scale, tf.constant(.3, D), tf.constant([2026, 919]),
        tf.constant(cfg.ridge, D), tf.constant(cfg.eigenvalue_floor, D),
        tf.constant(cfg.max_condition_number, D), tf.constant(cfg.score_holdout_relative_rmse, D))
    program(*inputs)
    assert program.experimental_get_tracing_count() == 1
    assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")


@pytest.mark.parametrize("rank_deficient", [False, True])
@pytest.mark.parametrize("holdout_tolerance", [.35, 1e-12])
def test_frozen_cloud_preserves_rank_rejection_projection_and_holdout_status(baseline, monkeypatch, rank_deficient, holdout_tolerance):
    scalar, _batch = _targets(indefinite=True, nonlinear=.3)
    z = tf.reshape(tf.sin(tf.cast(tf.range(32), D) * .73), [16, 2]) * .3
    if rank_deficient:
        z = z * tf.constant([1., 0.], D)
    monkeypatch.setattr(baseline, "_antithetic_cloud", lambda *_: z)
    center, scale = tf.constant([.13, -.22], D), tf.constant([.7, 1.2], D)
    center_score = scalar(center)[1]
    cfg = candidate.SequentialMapCovarianceConfig(score_holdout_relative_rmse=holdout_tolerance)
    expected, _ = baseline._fit_score_curvature(scalar, center, center_score, scale,
        dimension=2, radius=.3, sample_count=16, seed=(2026, 919), config=cfg,
        evaluations=0, batched_value_and_score_fn=None)
    scores = tf.stack([scalar(row)[1] for row in center[None, :] + z * scale[None, :]])
    train, heldout = native.partition_schema(16, .25, pair_disjoint=False)

    @tf.function(input_signature=[tf.TensorSpec([16, 2], D), tf.TensorSpec([16, 2], D)],
        jit_compile=True, autograph=False)
    def fit(offsets, scores):
        return native.fit_numerics(offsets, scores, center_score, scale,
            tf.constant(cfg.ridge, D), tf.constant(cfg.eigenvalue_floor, D),
            tf.constant(cfg.max_condition_number, D), tf.constant(holdout_tolerance, D),
            training_indices=train, holdout_indices=heldout)

    result = fit(z, scores)
    statuses = ("rank_deficient_symmetric_fit", "usable", "score_holdout_failed")
    assert statuses[int(result["status"])] == expected["status"]
    assert int(result["rank"]) == expected["rank"]
    if not rank_deficient:
        for key in result.keys() - {"status", "rank"}:
            np.testing.assert_allclose(result[key], expected[key], atol=1e-10, rtol=1e-10, err_msg=key)


def test_insufficient_support_does_not_evaluate_target(baseline):
    def forbidden(_):
        raise AssertionError("insufficient support must reject before evaluating")

    cfg = candidate.SequentialMapCovarianceConfig()
    inputs = (forbidden, tf.zeros([2], D), tf.zeros([2], D), tf.ones([2], D))
    options = {"dimension": 2, "radius": .3, "sample_count": 1, "seed": (2026, 919),
        "config": cfg, "evaluations": 7, "batched_value_and_score_fn": None}
    assert candidate._fit_score_curvature(*inputs, **options) == baseline._fit_score_curvature(*inputs, **options)


@pytest.mark.parametrize("all_invalid", [False, True])
def test_nonfinite_value_candidates_preserve_exact_winner_and_fit(baseline, all_invalid):
    scalar, _ = _targets()

    def target(point):
        value, score = scalar(point)
        invalid = tf.constant(True) if all_invalid else point[0] > .13
        return tf.where(invalid, tf.constant(float("nan"), D), value), score

    center = tf.constant([.13, -.22], D)
    args = (target, center, scalar(center)[1], tf.ones([2], D))
    options = {"dimension": 2, "radius": .3, "sample_count": 16, "seed": (2026, 919),
        "config": candidate.SequentialMapCovarianceConfig(), "evaluations": 0,
        "batched_value_and_score_fn": None}
    expected, _ = baseline._fit_score_curvature(*args, **options)
    actual, _ = candidate._fit_score_curvature(*args, **options)
    _assert_payload(actual, expected)


def test_score_fit_graph_size_is_independent_of_sample_count():
    scalar, batch = _targets()
    counts = []
    for sample_count in (16, 64):
        train, heldout = native.partition_schema(sample_count, .25, pair_disjoint=False)
        program = native.score_fit_program(scalar, batch, sample_count, 2, train, heldout)
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)
        counts.append(len(nodes))
    assert counts[0] == counts[1]


def test_complete_program_counts_target_rows_and_preserves_frozen_fit_boundary():
    counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    scalar, _ = _targets()

    def counted(point):
        counter.assign_add(1)
        return scalar(point)

    cfg = candidate.SequentialMapCovarianceConfig()
    center = tf.constant([.13, -.22], D)
    with tf.GradientTape() as tape:
        tape.watch(center)
        fit, count = candidate._fit_score_curvature(counted, center, scalar(center)[1], tf.ones([2], D),
            dimension=2, radius=.3, sample_count=16, seed=(2026, 919), config=cfg,
            evaluations=7, batched_value_and_score_fn=None)
        fitted = tf.reduce_sum(fit["projected_precision_z"])
    assert int(counter) == 16 and count == 23
    assert tape.gradient(fitted, center) is None
