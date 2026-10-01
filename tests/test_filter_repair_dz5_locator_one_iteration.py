"""Real L-BFGS limited to one iteration; compiler-context diagnostic only."""

import hashlib

import pytest

from tests import test_filter_repair_dz5_locator_context as context
from tests import test_filter_repair_dz5_locator_context_readback as readback

PROBE_SOURCE = '''def one_iteration(value_and_gradients_function, initial_position, **kwargs):
    kwargs['max_iterations'] = 1
    return real_lbfgs_minimize(value_and_gradients_function,
        initial_position=initial_position, **kwargs)
'''


def child_source(arm):
    child = context.child_source(arm)
    before = repr(context.PROBE_SOURCE)
    assert child.count(before) == 1
    child = child.replace(before, repr(PROBE_SOURCE))
    before = "namespace = {'tf': tf, 'SimpleNamespace': SimpleNamespace}"
    assert child.count(before) == 1
    child = child.replace(before, "namespace = {'tf': tf, 'SimpleNamespace': SimpleNamespace, "
        "'real_lbfgs_minimize': original.tfp.optimizer.lbfgs_minimize}")
    child = child.replace("namespace['single_objective']", "namespace['one_iteration']")
    child = child.replace('truncated_first_objective_context_diagnostic_not_optimizer',
        'genuine_one_iteration_optimizer_context_diagnostic')
    child = child.replace('L-BFGS is replaced in this isolated diagnostic; no optimization, convergence or runtime repair.',
        'Real L-BFGS is limited to one iteration; no full optimization, convergence or runtime repair.')
    before = "assert report['optimizer_callback_batches'] == 1"
    assert child.count(before) == 1
    child = child.replace(before, "assert 1 <= report['optimizer_callback_batches'] <= 128")
    compile(child, '<genuine_one_iteration_context>', 'exec')
    return child


@pytest.mark.parametrize('arm', ['original', 'candidate'])
def test_real_optimizer_one_iteration(request, arm):
    report = context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=context.trajectory.SNAPSHOT, child=child_source(arm),
        scope='genuine_one_iteration_context_no_new_admission', child_timeout_seconds=280,
        read_only_paths=(context.trajectory.consumer.RAW,))
    assert report['target_evaluated']
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report['role'] == 'genuine_one_iteration_optimizer_context_diagnostic'
    assert report['probe_dispatch_sha256'] == hashlib.sha256(PROBE_SOURCE.encode()).hexdigest()


def test_saved_real_optimizer_contexts(request):
    readback.check_saved_contexts(request,
        group_prefix='dz5_locator_one_iteration',
        output_name='dz5-one-iteration-context-readback.json', maximum_callbacks=128,
        schema='filter_dz5_one_iteration_context.v1')
