"""Truncated first-objective context diagnostic; never a runtime optimizer."""

import hashlib

import pytest

from tests import test_filter_repair_dz5_locator_trajectory as trajectory

PROBE_SOURCE = '''def single_objective(value_and_gradients_function, initial_position, **kwargs):
    value, gradient = value_and_gradients_function(initial_position)
    shape = tf.shape(initial_position)[:-1]
    return SimpleNamespace(position=initial_position, objective_value=value,
        objective_gradient=gradient, converged=tf.zeros(shape, tf.bool),
        failed=tf.zeros(shape, tf.bool))
'''


def child_source(arm):
    child = trajectory.child_source(arm)
    before = '    tick = time.monotonic()\n'
    replacement = '''    # Diagnostic dispatch only: evaluate the first objective once.
    from types import SimpleNamespace
    from bayesfilter.inference import batched_local_center_tf as candidate_module
    probe_source = PROBE_SOURCE
    (OUT / 'diagnostic-optimizer-dispatch.py').write_text(probe_source)
    namespace = {'tf': tf, 'SimpleNamespace': SimpleNamespace}
    exec(compile(probe_source, '<single_objective_diagnostic>', 'exec'), namespace)
    proxy = SimpleNamespace(optimizer=SimpleNamespace(
        lbfgs_minimize=namespace['single_objective'],
        converged_all=original.tfp.optimizer.converged_all))
    original.tfp = proxy
    candidate_module.tfp = proxy
    report['role'] = 'truncated_first_objective_context_diagnostic_not_optimizer'
    report['probe_dispatch_sha256'] = hashlib.sha256(probe_source.encode()).hexdigest()
    report['nonclaims'].append('L-BFGS is replaced in this isolated diagnostic; no optimization, convergence or runtime repair.')
    tick = time.monotonic()
'''.replace('PROBE_SOURCE', repr(PROBE_SOURCE))
    assert child.count(before) == 1
    child = child.replace(before, replacement)
    before = '    assert base.assert_source_pinned()\n'
    replacement = '''    # Preserve compiler context without running an optimizer trajectory.
    arguments = () if LOCATOR_ARM.startswith('original') else (initial, scale)
    hlo = compiled.experimental_get_compiler_ir(*arguments)(stage='hlo')
    (OUT / 'first-objective-context.hlo.txt').write_text(hlo)
    report['context_hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
    report['context_hlo_bytes'] = len(hlo.encode())
    report['optimizer_callback_batches'] = int(location.optimizer_target_batches)
    assert report['optimizer_callback_batches'] == 1
    assert base.assert_source_pinned()
'''.replace('LOCATOR_ARM', repr(arm))
    assert child.count(before) == 1
    return child.replace(before, replacement)


@pytest.mark.parametrize('arm', ['original', 'candidate'])
def test_first_objective_context(request, arm):
    report = trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=trajectory.SNAPSHOT, child=child_source(arm),
        scope='truncated_first_objective_context_no_new_admission',
        child_timeout_seconds=280, read_only_paths=(trajectory.consumer.RAW,))
    assert report['target_evaluated']
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report['probe_dispatch_sha256'] == hashlib.sha256(PROBE_SOURCE.encode()).hexdigest()
