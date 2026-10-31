"""CPU diagnostics separating reporting resource storage and integer arithmetic."""

import pytest

from tests import test_filter_repair_dz5_locator_accounting_families as families
from tests import test_filter_repair_dz5_locator_counter_context as counter

COUNTERS = ('attempts', 'optimizer_calls', 'replays')
ARMS = (*COUNTERS, 'logical_int32')
INCREMENTS = tuple((f'{name}.assign_add(1)',
    f'{name}.assign(tf.cast(tf.cast({name}.read_value(), tf.int32) + 1, tf.int64))')
    for name in COUNTERS)
INCREMENT_TRANSFORM = '''    class AccountingDtype(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            for before, after in replacements:
                if ast.dump(node, include_attributes=False) == ast.dump(ast.parse(before, mode='eval').body, include_attributes=False):
                    self.count += 1
                    return ast.parse(after, mode='eval').body
            return self.generic_visit(node)
'''


def patch(arm):
    if arm in COUNTERS:
        original = families.family_patch('progress')
        original = original.replace(repr(families.FAMILIES['progress']),
            repr((f'        {arm} = variable(0, tf.int64)',)))
        original = original.replace('assert change.count == 4', 'assert change.count == 1')
        original = original.replace("'family': 'progress'", f"'family': {arm!r}")
        return original.replace('cpu-family-progress-diagnostic.py', f'cpu-reporting-{arm}-diagnostic.py')
    assert arm == 'logical_int32'
    original = counter.PATCH
    before = "    diagnostic_source = candidate_source.replace('tf.int64', 'tf.int32')\n"
    replacement = f'    replacements = {INCREMENTS!r}\n' + '''    diagnostic_source = candidate_source
    for before, after in replacements:
        assert diagnostic_source.count(before) == (2 if before.startswith('replays.') else 1)
        diagnostic_source = diagnostic_source.replace(before, after)
'''
    original = original.replace(before, replacement)
    start = original.index('    class AccountingDtype(')
    end = original.index('    change = AccountingDtype()', start)
    original = original[:start] + INCREMENT_TRANSFORM + original[end:]
    original = original.replace('assert change.count > 0', 'assert change.count == 4')
    original = original.replace('only_tf_int64_to_int32_AST_change', 'only_reporting_increment_AST_change')
    original = original.replace("'original_sha256':", "'family': 'logical_int32', 'original_sha256':")
    return original.replace('cpu-accounting-int32-diagnostic.py', 'cpu-reporting-logical_int32-diagnostic.py')


def child_source(arm):
    child = counter.child_source(patch(arm))
    anchor = "    report['context_hlo_bytes'] = len(hlo.encode())\n"
    assert child.count(anchor) == 1
    addition = '''    report['captured_resource_dtypes'] = [variable.dtype.name
        for variable in compiled.get_concrete_function().variables]
    report['reporting_counter_values'] = {
        'attempts': int(location.optimizer_callback_attempts),
        'optimizer_calls': int(location.optimizer_target_batches),
        'replays': int(location.replay_batches)}
    assert all(0 <= value < 2**31 for value in report['reporting_counter_values'].values())
'''
    if arm == 'logical_int32':
        addition += "    assert 'int32' not in report['captured_resource_dtypes']\n"
    child = child.replace(anchor, anchor + addition)
    compile(child, '<reporting_storage_context>', 'exec')
    return child


@pytest.mark.parametrize('arm', ARMS)
def test_reporting_storage_context(request, arm):
    report = counter.probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=counter.probe.context.trajectory.SNAPSHOT, child=child_source(arm),
        scope=f'CPU_reporting_storage_{arm}_not_runtime_candidate', child_timeout_seconds=280,
        read_only_paths=(counter.probe.context.trajectory.consumer.RAW,))
    evidence = report['counter_context_patch']
    assert evidence['family'] == arm
    assert evidence['changed_dtype_attributes'] == (4 if arm == 'logical_int32' else 1)
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report['reporting_counter_values']['optimizer_calls'] == report['optimizer_callback_batches']
    if arm == 'logical_int32':
        assert evidence['only_reporting_increment_AST_change']
        assert 'int32' not in report['captured_resource_dtypes']
        assert 'int64' in report['captured_resource_dtypes']
    else:
        assert evidence['only_selected_family_AST_change']


@pytest.mark.parametrize('arm', ARMS)
def test_saved_reporting_storage(request, arm):
    result = counter.check_saved_counter_context(request,
        group=f'dz5_locator_reporting_{arm}_cpu',
        output=f'dz5-reporting-{arm}-readback.json',
        diagnostic_name=f'cpu-reporting-{arm}-diagnostic.py')
    root = counter.probe.context.trajectory.consumer.RAW
    actual = counter.json.loads((root / result['run'] / 'dz5-snapshot-import.json').read_text())
    names = {'attempts': 'optimizer_callback_attempts',
        'optimizer_calls': 'optimizer_target_batches', 'replays': 'replay_batches'}
    for number in (4635, 4636):
        reference = counter.json.loads((root / f'run-{number:05d}' / 'dz5-snapshot-import.json').read_text())
        assert actual['reporting_counter_values'] == {
            name: reference['locator_result'][field] for name, field in names.items()}
