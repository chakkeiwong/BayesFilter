"""Isolated merged-source DZ5 engineering checks; no training or admission."""

from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_snapshot as snapshot_test

MERGED_SNAPSHOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
    'filter-gradient-repair-20260917/dz5-candidate-source-merged-9d8202b77-r2')
ARCHIVED_SNAPSHOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
    'filter-gradient-repair-20260917/dz5-archived-cdf-source-31f0067f9-r1')

TARGET_CHECK = r'''
    import time
    import resource
    from two_currency_double_zlb_credit_target import (
        credit_target_value_and_score, credit_target_numeric_outputs)
    batch = int(os.environ['FILTER_REPAIR_DZ5_BATCH'])
    tf.config.experimental.enable_tensor_float_32_execution(False)
    # Generate identical diagnostic inputs on CPU, then copy to the tested device.
    with tf.device('/CPU:0'):
        center = tf.constant(fixture.parameter_truth, tf.float64)
        scale = tf.constant(fixture.prior_scale, tf.float64)
        direction = tf.reshape(tf.sin(tf.cast(tf.range(batch * 23), tf.float64)), [batch, 23])
        initial = center[None, :] + 0.001 * scale[None, :] * direction
        if batch == 4:
            prior = tf.constant(fixture.prior_mean, tf.float64)
            initial = tf.stack([center, prior, center + 0.005 * scale, prior - 0.005 * scale])
        changed = initial + 0.0002 * scale[None, :]
    target_device = '/GPU:0' if gpu else '/CPU:0'
    with tf.device(target_device):
        initial, changed = tf.identity(initial), tf.identity(changed)
    def evaluate(function, positions):
        with tf.device(target_device):
            return function(positions)
    signature = [tf.TensorSpec([batch, 23], tf.float64)]
    def numeric_target(p, jit):
        if jit:
            fields = credit_target_numeric_outputs(p, fixture)
        else:
            value, score, diagnostics = credit_target_value_and_score(p, fixture,
                jit_compile=False, evaluation_policy='hmc_rejection')
            valid = diagnostics['valid_pre_regularized_score']
            fields = (value, score, tf.where(valid, 0, 9001), valid,
                tf.zeros_like(diagnostics['branch_status_code']), diagnostics['branch_status_code'],
                diagnostics['minimum_chart_pivot'], diagnostics['maximum_chart_residual'],
                diagnostics['maximum_support_residual'])
        return fields[0], fields[1], {
            'status_code': fields[2], 'valid_pre_regularized_score': fields[3],
            'regularization_count': fields[4], 'branch_status_code': fields[5],
            'minimum_chart_pivot': fields[6], 'maximum_chart_residual': fields[7],
            'maximum_support_residual': fields[8]}
    xla = tf.function(lambda p: numeric_target(p, True),
        input_signature=signature, jit_compile=True, autograph=False)
    graph = tf.function(lambda p: numeric_target(p, False),
        input_signature=signature, jit_compile=False, autograph=False)
    report.update(role='isolated_DZ5_target_graph_XLA_engineering_comparison',
        target_evaluated=False, target_execution_attempted=True, batch=batch, jit_compile=True,
        reference='same frozen actual CDF source with explicit candidate graph reference',
        source_role=manifest['role'],
        archived_source_comparison_complete=False,
        nonclaims=['No admission, training, HMC, cost ranking or full archived-source qualification.'])
    comparisons = []
    report['comparisons'] = comparisons
    first = None
    for label, positions in (('initial', initial), ('changed', changed), ('replay', initial)):
        tick = time.monotonic()
        actual = evaluate(xla, positions)
        tf.nest.map_structure(lambda v: v.numpy(), actual)
        report['target_evaluated'] = True
        xla_seconds = time.monotonic() - tick
        expected = evaluate(graph, positions)
        tf.nest.map_structure(lambda v: v.numpy(), expected)
        comparisons.append({'label': label, 'positions': positions.numpy().tolist(),
            'value': actual[0].numpy().tolist(), 'score': actual[1].numpy().tolist(),
            'reference_value': expected[0].numpy().tolist(), 'reference_score': expected[1].numpy().tolist(),
            'status': {key: value.numpy().tolist() for key, value in actual[2].items()
                if value.dtype == tf.bool or value.dtype.is_integer},
            'floating_diagnostics': {key: value.numpy().tolist() for key, value in actual[2].items()
                if value.dtype.is_floating},
            'result_device': actual[0].device,
            'xla_seconds_in_shared_reference_process': xla_seconds})
        assert all(value.device.endswith('device:' + target_device[1:]) for value in actual[:2]), target_device
        tf.debugging.assert_equal(actual[2]['valid_pre_regularized_score'], tf.ones([batch], tf.bool))
        tf.debugging.assert_equal(actual[2]['branch_status_code'], tf.zeros([batch], tf.int32))
        errors = {}
        for key, left, right in [('value', actual[0], expected[0]), ('score', actual[1], expected[1])]:
            tf.debugging.assert_all_finite(left, key)
            delta = tf.abs(left - right)
            limit = tf.constant(1e-8, tf.float64) + tf.constant(1e-7, tf.float64) * tf.abs(right)
            tf.debugging.assert_less_equal(delta, limit)
            errors[key] = float(tf.reduce_max(delta / limit).numpy())
        for key, value in actual[2].items():
            if value.dtype == tf.bool or value.dtype.is_integer:
                tf.debugging.assert_equal(value, expected[2][key])
            else:
                tf.debugging.assert_less_equal(tf.abs(value - expected[2][key]),
                    tf.constant(1e-8, tf.float64) + tf.constant(1e-7, tf.float64) * tf.abs(expected[2][key]))
        if label == 'initial':
            first = actual
        if label == 'replay':
            for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(first), strict=True):
                tf.debugging.assert_equal(left, right)
        comparisons[-1]['max_scaled_errors'] = errors
    if batch == 4:
        invalid = tf.tensor_scatter_nd_update(initial, [[1, 18], [2, 22]],
            tf.constant([float('nan'), -100.0], tf.float64))
        rejected = evaluate(xla, invalid)
        graph_rejected = evaluate(graph, invalid)
        report['invalid_rows'] = {
            'valid': rejected[2]['valid_pre_regularized_score'].numpy().tolist(),
            'branch_status_code': rejected[2]['branch_status_code'].numpy().tolist(),
            'value': rejected[0].numpy().tolist(), 'score': rejected[1].numpy().tolist()}
        tf.debugging.assert_equal(rejected[2]['valid_pre_regularized_score'], [True, False, False, True])
        tf.debugging.assert_equal(rejected[2]['branch_status_code'], graph_rejected[2]['branch_status_code'])
        tf.debugging.assert_equal(tf.gather(rejected[0], [1, 2]), tf.constant([-1e100, -1e100], tf.float64))
        tf.debugging.assert_equal(tf.gather(rejected[1], [1, 2]), tf.zeros([2, 23], tf.float64))
        for left, right in zip(rejected[:2], first[:2], strict=True):
            tf.debugging.assert_equal(tf.gather(left, [0, 3]), tf.gather(right, [0, 3]))
        report['invalid_row_isolation'] = True
    definition = xla.get_concrete_function().graph.as_graph_def()
    nodes = [*definition.node, *(n for f in definition.library.function for n in f.node_def)]
    forbidden = sorted({n.op for n in nodes if any(term in n.op.lower() for term in
        ('pyfunc', 'sylvester', 'principalsqrt', 'xlahostcompute'))})
    assert not forbidden, forbidden
    with tf.device(target_device):
        hlo = xla.experimental_get_compiler_ir(initial)(stage='hlo')
        changed_hlo = xla.experimental_get_compiler_ir(changed)(stage='hlo')
    assert hlo == changed_hlo
    assert xla.experimental_get_tracing_count() == 1
    (OUT / 'dz5-target.hlo.txt').write_text(hlo)
    report.update(comparisons=comparisons, trace_count=1, host_callbacks=forbidden,
        hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(), hlo_changed_input_equal=True,
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled())
    report['gpu_allocator'] = tf.config.experimental.get_memory_info('GPU:0') if devices else None
    # Re-audit all actual project imports after tracing numerical callbacks.
    unexpected, loaded_after = {}, {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            owned = path.is_relative_to(BF) or path.is_relative_to(MF)
            named = name.startswith(('bayesfilter', 'two_currency_double_zlb'))
            if owned or named:
                loaded_after[name] = {'path': str(path), 'sha256': sha(path)}
                if not owned or expected_sources.get(str(path)) != sha(path):
                    unexpected[name] = str(path)
    report['post_target_loaded_modules'] = loaded_after
    report['post_target_unexpected_modules'] = unexpected
    assert not unexpected, unexpected
    assert all(sha(path) == digest for path, digest in expected_sources.items())
'''



def test_merged_dz5_snapshot_import(request):
    snapshot_test.run_isolated_snapshot(request, snapshot=MERGED_SNAPSHOT,
        child=snapshot_test.CHILD, scope='merged_source_CPU_import_reference_only')


def target_child(*, archived=False):
    # Substitute only diagnostic code; the source snapshots remain read-only.
    child = snapshot_test.CHILD.replace(
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'",
        "    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'")
    child = child.replace('configure_tensorflow_gpu_memory_growth(tf, require_gpu=False)',
        'configure_tensorflow_gpu_memory_growth(tf, require_gpu=gpu)')
    child = child.replace("    assert not tf.config.list_physical_devices('GPU')",
        "    devices = tf.config.list_physical_devices('GPU')\n"
        "    assert len(devices) == (1 if gpu else 0)\n"
        "    report.update(device='GPU' if gpu else 'CPU',\n"
        "        trust_basis='owner_designated_managed_session_visible_gpu_trusted' if gpu else 'CPU_reference',\n"
        "        physical_gpus=[device.name for device in devices])")
    child = child.replace("    report['passed'] = True",
        "    expected_sources = expected.copy()\n" + TARGET_CHECK + "\n    report['passed'] = True")
    if archived:
        # Match the original CDF qualification's metadata-class import. Its
        # old lazy facade otherwise imports unrelated, unsnapshotted HMC code.
        child = child.replace('    from bayesfilter_estimation import audit_sources, declared_source_closure',
            '    import bayesfilter.inference as native\n'
            '    from bayesfilter.inference.posterior_adapter import ValueScoreCapability\n'
            '    native.ValueScoreCapability = ValueScoreCapability\n'
            "    report['archived_metadata_import'] = 'original_CDF_qualification_priming'\n"
            '    from bayesfilter_estimation import audit_sources, declared_source_closure')
        # The old target does not depend on the repaired initializer or its
        # preparation script. Audit its actual imports against archived bytes.
        for module in ('bayesfilter_estimation_initialization',
                       'two_currency_double_zlb_credit_recovery',
                       'bayesfilter.inference.dense_initializer_seeded_tf',
                       'bayesfilter.inference.tensor_npz_archive'):
            child = child.replace(f"    importlib.import_module('{module}')\n", '')
        start = child.index('    closure = audit_sources(')
        end = child.index('    loaded, unexpected = {}, {}', start)
        child = child[:start] + '    closure = {}\n' + child[end:]
        child = child.replace('explicit candidate graph reference', 'explicit archived graph reference')
    return child


@pytest.mark.parametrize('batch', [1, 4, 46, 68])
def test_merged_dz5_target_graph_xla(request, monkeypatch, batch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_BATCH', str(batch))
    snapshot_test.run_isolated_snapshot(request, snapshot=MERGED_SNAPSHOT,
        child=target_child(), child_timeout_seconds=840,
        scope='actual_CDF_target_graph_XLA_engineering_comparison')


@pytest.mark.parametrize('batch', [1, 4, 46, 68])
def test_archived_dz5_target_graph_xla(request, monkeypatch, batch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_BATCH', str(batch))
    snapshot_test.run_isolated_snapshot(request, snapshot=ARCHIVED_SNAPSHOT,
        child=target_child(archived=True), child_timeout_seconds=840,
        scope='archived_CDF_target_independent_reference_only')
