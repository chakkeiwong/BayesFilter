"""Graph reference execution must not be reported as an XLA fit."""

import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.batched_local_center import BatchedLocalCenterConfig
from bayesfilter.inference.dense_initializer_reporting import (
    format_dense_initializer_result,
)
from bayesfilter.inference.dense_initializer_seeded_tf import (
    SeededDenseInitializerProgram,
)
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.posterior_local_initializer import (
    PosteriorLocalInitializerConfig,
)
from tests.test_filter_repair_dense_validated_fit import _thresholds
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def test_graph_reference_reports_its_actual_compilation_setting(request):
    loadings = tf.constant([.2, .35, -.15], D)
    marginal = tf.constant([.9, 1.2, 1.4], D)
    correlation = tf.linalg.diag(1. - loadings**2) + loadings[:, None] * loadings[None, :]
    precision = tf.linalg.inv(marginal[:, None] * correlation * marginal[None, :])

    def target(points):
        scores = -tf.matmul(points, precision, transpose_b=True)
        return .5 * tf.reduce_sum(points * scores, axis=1), scores, tf.ones([points.shape[0]], tf.bool)

    config = PosteriorLocalInitializerConfig(locator_config=JointCenterLocatorConfig(
        max_iterations=4, max_objective_evaluations=40, gradient_tolerance=1e-8),
        max_curvature_attempts=1, training_rows_per_replicate=9,
        selection_rows_per_replicate=6, audit_rows=7, replicate_count=2, seed=(215, 91))
    thresholds = _thresholds(fixed)
    locator = BatchedLocalCenterConfig(box_radius=config.locator_box_radius,
        trust_refinement_rounds=1, max_iterations=4, gradient_tolerance=1e-8,
        max_optimizer_callback_batches_per_round=40, jit_compile=False)
    program = SeededDenseInitializerProgram(target, 3, 2, 9, 6, 7,
        locator_config=locator, thresholds=thresholds, max_attempts=1,
        max_exact_evaluations=config.max_exact_evaluations, center_score_max=1e-6)
    raw = program(tf.zeros([3], D), tf.ones([3], D), tf.constant(config.seed, tf.int32), tf.constant(.13, D))
    report, _archives = format_dense_initializer_result(raw, config, thresholds, [9, 9, 6, 6, 7],
        trace_count=program.controller.locator.compiled.experimental_get_tracing_count(),
        planned_rows=program.planned_rows)
    flags = [fit['diagnostics']['jit_compile'] for attempt in report['attempts']
        for fit in attempt['curvature']['fits'] if 'jit_compile' in fit['diagnostics']]
    actual_setting = program.compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    save(request, 'dense-execution-reporting.json', clean({'outer_jit_compile': actual_setting,
        'raw_jit_compile': raw.get('jit_compile'), 'structured_fit_flags': flags, 'result': report,
        'role': 'explicit_CPU_graph_reference_reporting_regression',
        'nonclaims': ['No graph-mode default, performance or numerical-equivalence promotion.']}))
    assert actual_setting is False
    assert flags and all(flag is False for flag in flags), flags
    assert bool(raw['jit_compile']) is False
