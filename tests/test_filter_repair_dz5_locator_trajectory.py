"""Diagnostic original/reusable CDF locator callbacks in isolated CPU workers."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_initializer_consumer as consumer
from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_snapshot as snapshot_test
from tests.test_filter_repair_dz5_initializer_target import SNAPSHOT, audited_child

ROOT = Path(__file__).resolve().parents[1]
LOCATOR_PATH = 'bayesfilter/inference/batched_local_center.py'

LOCATOR_CHECK = r'''
    import dataclasses
    import types
    from bayesfilter.inference.batched_local_center import BatchedLocalCenterResult
    from bayesfilter.inference.batched_local_center_tf import BatchedLocalCenterProgram
    from bayesfilter.inference.tensor_npz_archive import write_tensor_npz
    from bayesfilter_estimation_runtime import initializer_configurations
    assert not gpu
    source = ORIGINAL_SOURCE
    assert hashlib.sha256(source.encode()).hexdigest() == ORIGINAL_SHA
    baseline_source = source
    if LOCATOR_ARM == 'original_operands':
        replacements = {
            '    def run():': '    def run(initial_operand, scale_operand):\n        nonlocal initial, scale_tensor\n        initial, scale_tensor = initial_operand, scale_operand',
            'compiled = tf.function(run, jit_compile=cfg.jit_compile, autograph=False)':
                'compiled = tf.function(run, input_signature=[tf.TensorSpec(initial.shape, tf.float64), tf.TensorSpec(scale_tensor.shape, tf.float64)], jit_compile=cfg.jit_compile, autograph=False)',
            '    tensors = compiled()': '    tensors = compiled(initial, scale_tensor)',
        }
        for before, after in replacements.items():
            assert source.count(before) == 1
            source = source.replace(before, after)
        (OUT / 'pinned-original-batched-local-center.py').write_text(baseline_source)
    (OUT / 'original-batched-local-center.py').write_text(source)
    original = types.ModuleType('pinned_original_batched_locator_diagnostic')
    original.__file__ = str(OUT / 'original-batched-local-center.py')
    sys.modules[original.__name__] = original
    exec(compile(source, original.__file__, 'exec'), original.__dict__)
    observed = []
    class ObservedTF:
        def __getattr__(self, name):
            if name == 'function':
                def function(*args, **kwargs):
                    compiled = tf.function(*args, **kwargs)
                    observed.append(compiled)
                    return compiled
                return function
            return getattr(tf, name)
    original.tf = ObservedTF()
    configuration, _, _ = initializer_configurations(recipe)
    locator = configuration.locator_config
    settings = dict(box_radius=configuration.locator_box_radius, trust_refinement_rounds=1,
        num_correction_pairs=locator.num_correction_pairs, max_iterations=locator.max_iterations,
        max_line_search_iterations=locator.max_line_search_iterations,
        gradient_tolerance=locator.gradient_tolerance,
        max_optimizer_callback_batches_per_round=locator.max_objective_evaluations, jit_compile=True)
    capacity = configuration.max_exact_evaluations
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    trace_positions = tf.Variable(tf.zeros([capacity, 23], tf.float64), trainable=False)
    trace_values = tf.Variable(tf.zeros([capacity], tf.float64), trainable=False)
    trace_scores = tf.Variable(tf.zeros([capacity, 23], tf.float64), trainable=False)
    trace_valid = tf.Variable(tf.zeros([capacity], tf.bool), trainable=False)
    callback = model.training_target.batch_value_score_and_validity
    use_barriers = LOCATOR_ARM.endswith('_barrier')
    from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier
    def measured_callback(points):
        assert points.shape == (1, 23)
        if use_barriers:
            points = xla_optimization_barrier(input=[points])[0]
        value, score, valid = callback(points)
        if use_barriers:
            value, score, valid = xla_optimization_barrier(input=[value, score, valid])
        index = calls.assign_add(1) - 1
        indices = tf.reshape(index, [1, 1])
        writes = (trace_positions.scatter_nd_update(indices, points),
            trace_values.scatter_nd_update(indices, value),
            trace_scores.scatter_nd_update(indices, score),
            trace_valid.scatter_nd_update(indices, valid))
        with tf.control_dependencies(writes):
            return tf.identity(value), tf.identity(score), tf.identity(valid)
    initial = tf.constant(model.initial_position, tf.float64)[None, :]
    scale = tf.constant(model.coordinate_scale, tf.float64)
    report.update(role='frozen_CDF_locator_first_divergence_localization', arm=LOCATOR_ARM,
        jit_compile=True, cpu_reference_exception=True, target_execution_attempted=True,
        original_locator_sha256=ORIGINAL_SHA, settings=settings,
        executed_original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_operand_binding_diagnostic=LOCATOR_ARM == 'original_operands',
        callback_optimization_barriers='inputs_and_outputs' if use_barriers else 'none',
        observer='fixed_capacity_TensorFlow_callback_trace; no numerical output changed',
        nonclaims=['No current SVD source admission, complete initializer, training, HMC or matched timing.'])
    tick = time.monotonic()
    if LOCATOR_ARM.startswith('original'):
        location = original.locate_batched_local_center(measured_callback, initial, scale,
            config=original.BatchedLocalCenterConfig(**settings))
        assert len(observed) == 1
        compiled = observed[0]
    else:
        from bayesfilter.inference.batched_local_center import BatchedLocalCenterConfig
        owner = BatchedLocalCenterProgram(measured_callback, 1, 23, BatchedLocalCenterConfig(**settings))
        tensors = owner(initial, scale)
        compiled = owner.compiled
        location = BatchedLocalCenterResult(**tensors,
            trace_count=compiled.experimental_get_tracing_count())
    payload = location.payload()
    report['locator_seconds'] = time.monotonic() - tick
    count = int(calls)
    assert 0 < count <= capacity
    trace = {'positions': trace_positions[:count], 'values': trace_values[:count],
        'scores': trace_scores[:count], 'valid': trace_valid[:count]}
    with (OUT / 'locator-callbacks.npz').open('xb') as stream:
        write_tensor_npz(stream, trace)
    (OUT / 'locator-result.json').write_text(json.dumps(diagnostic_json(payload), indent=2, allow_nan=False) + '\n')
    report.update(target_evaluated=True, locator_result=payload, callback_rows=count,
        callbacks_sha256=sha(OUT / 'locator-callbacks.npz'),
        trace_count=compiled.experimental_get_tracing_count(),
        input_signature=str(compiled.input_signature),
        target_trace_counts={str(key): value.experimental_get_tracing_count() for key, value in base._compiled.items()},
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert count == int(location.physical_target_rows)
    assert report['trace_count'] == 1
    assert compiled.function_spec.jit_compile
    concrete = compiled.get_concrete_function() if LOCATOR_ARM.startswith('original') else compiled.get_concrete_function(initial, scale)
    definition = concrete.graph.as_graph_def()
    nodes = [*definition.node, *(node for f in definition.library.function for node in f.node_def)]
    report['host_callbacks'] = sorted({node.op for node in nodes if any(term in node.op.lower()
        for term in ('pyfunc', 'xlahostcompute'))})
    report['graph_bytes'] = definition.ByteSize()
    report['graph_sha256'] = hashlib.sha256(definition.SerializeToString()).hexdigest()
    assert not report['host_callbacks']
    assert base.assert_source_pinned()
'''


def child_source(arm):
    original = subprocess.check_output(['git', 'show', f'd6a568384:{LOCATOR_PATH}'], cwd=ROOT, text=True)
    setup = consumer.INITIALIZER_CHECK.split('    events, owners, refs = [], [], []')[0]
    body = setup + LOCATOR_CHECK
    replacements = {'RECIPE_PATH': repr(str(consumer.ACCEPTED_INPUTS / 'recipe.json')),
        'ADMISSION_PATH': repr(str(consumer.ADMISSION)), 'ACCEPTED_CASE': 'True',
        'ACCEPTED_INPUT_PATH': repr(str(consumer.ACCEPTED_INPUTS)),
        'ORIGINAL_SOURCE': repr(original), 'ORIGINAL_SHA': repr(hashlib.sha256(original.encode()).hexdigest()),
        'LOCATOR_ARM': repr(arm)}
    for key, value in replacements.items():
        body = body.replace(key, value)
    audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    return audited_child(merged.target_child().replace(merged.TARGET_CHECK, body + audit))


@pytest.mark.parametrize('arm', ['original', 'candidate', 'original_operands', 'original_barrier', 'candidate_barrier'])
def test_actual_cdf_locator_trace(arm, request):
    report = snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child_source(arm), scope='original_current_CDF_locator_trace_no_new_admission',
        child_timeout_seconds=840, read_only_paths=(consumer.RAW,))
    assert report['target_evaluated'] and report['callback_rows'] > 0
    assert not report['host_callbacks'] and report['trace_count'] == 1
