"""Resource-free replay reporting diagnostic; runtime remains unchanged."""

import hashlib
import json
import struct

import numpy as np

from tests import test_filter_repair_dz5_locator_counter_context as counter
from tests import test_filter_repair_dz5_locator_reporting_storage as storage

EDITS = (
    ('        replays = variable(0, tf.int64)\n', ''),
    ('                replays.assign(0),\n', ''),
    ('                replays.assign_add(1)\n', ''),
    ('replays.read_value()', 'tf.cast(rounds, tf.int64) + tf.cast(has_incumbent, tf.int64)'),
)
TRANSFORM = '''    class AccountingDtype(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(ast.parse('replays = variable(0, tf.int64)').body[0], include_attributes=False):
                self.count += 1
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(ast.parse('replays.assign_add(1)').body[0], include_attributes=False):
                self.count += 1
                return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            for before, after in (('replays.assign(0)', None),
                ('replays.read_value()', 'tf.cast(rounds, tf.int64) + tf.cast(has_incumbent, tf.int64)')):
                if ast.dump(node, include_attributes=False) == ast.dump(ast.parse(before, mode='eval').body, include_attributes=False):
                    self.count += 1
                    return None if after is None else ast.parse(after, mode='eval').body
            return self.generic_visit(node)
'''


def child_source():
    patch = counter.PATCH
    before = "    diagnostic_source = candidate_source.replace('tf.int64', 'tf.int32')\n"
    replacement = f'    replacements = {EDITS!r}\n' + '''    diagnostic_source = candidate_source
    for before, after in replacements:
        assert diagnostic_source.count(before) == (2 if 'assign_add' in before else 1)
        diagnostic_source = diagnostic_source.replace(before, after)
'''
    patch = patch.replace(before, replacement)
    start = patch.index('    class AccountingDtype(')
    end = patch.index('    change = AccountingDtype()', start)
    patch = patch[:start] + TRANSFORM + patch[end:]
    patch = patch.replace('assert change.count > 0', 'assert change.count == 5')
    patch = patch.replace('changed_dtype_attributes', 'changed_replay_AST_sites')
    patch = patch.replace('only_tf_int64_to_int32_AST_change', 'only_derived_replay_AST_change')
    patch = patch.replace('cpu-accounting-int32-diagnostic.py', 'cpu-derived-replay-diagnostic.py')
    patch = patch.replace("'original_sha256':", "'family': 'derived_replay', 'original_sha256':")
    # Use the same recorded resource/count observer, with only this source patch changed.
    child = storage.child_source('logical_int32')
    old = storage.patch('logical_int32')
    assert child.count(old) == 1
    child = child.replace(old, patch)
    compile(child, '<derived_replay_context>', 'exec')
    return child


def test_derived_replay_context(request):
    report = counter.probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=counter.probe.context.trajectory.SNAPSHOT, child=child_source(),
        scope='CPU_derived_replay_context_not_runtime_candidate', child_timeout_seconds=280,
        read_only_paths=(counter.probe.context.trajectory.consumer.RAW,))
    assert report['counter_context_patch']['only_derived_replay_AST_change']
    assert report['counter_context_patch']['changed_replay_AST_sites'] == 5
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert 'int32' not in report['captured_resource_dtypes']


def record_differences(actual, expected, prefix=''):
    if isinstance(actual, float) and isinstance(expected, float):
        return [] if struct.pack('>d', actual) == struct.pack('>d', expected) else [prefix]
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(actual.keys() | expected.keys()):
            path = f'{prefix}.{key}' if prefix else key
            result.extend([path] if key not in actual or key not in expected else
                record_differences(actual[key], expected[key], path))
        return result
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [prefix + '.length']
        return [path for index, (a, b) in enumerate(zip(actual, expected, strict=True))
            for path in record_differences(a, b, f'{prefix}[{index}]')]
    return [] if actual == expected and type(actual) is type(expected) else [prefix]


def test_saved_derived_replay(request):
    result = counter.check_saved_counter_context(request, group='dz5_locator_derived_replay_cpu',
        output='dz5-derived-replay-first-readback.json', diagnostic_name='cpu-derived-replay-diagnostic.py')
    root = counter.probe.context.trajectory.consumer.RAW
    directory = root / result['run']
    actual_report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    actual_record = json.loads((directory / 'locator-result.json').read_text())
    assert actual_record['replay_batches'] == actual_record['rounds_completed'] + int(any(actual_record['valid_rows']))
    assert 'int32' not in actual_report['captured_resource_dtypes']
    with np.load(directory / 'locator-callbacks.npz', allow_pickle=False) as payload:
        actual_arrays = {key: payload[key].copy() for key in payload.files}
    comparisons = {}
    for number in (4635, 4636):
        previous = root / f'run-{number:05d}'
        prior_record = json.loads((previous / 'locator-result.json').read_text())
        with np.load(previous / 'locator-callbacks.npz', allow_pickle=False) as payload:
            prior_arrays = {key: payload[key].copy() for key in payload.files}
        comparisons[str(number)] = {
            'callbacks': {key: counter.array_comparison(actual_arrays[key], prior_arrays[key]) for key in actual_arrays},
            'different_record_paths': record_differences(actual_record, prior_record),
            'record_sha256': hashlib.sha256((previous / 'locator-result.json').read_bytes()).hexdigest()}
        assert actual_report['reporting_counter_values'] == {
            'attempts': prior_record['optimizer_callback_attempts'],
            'optimizer_calls': prior_record['optimizer_target_batches'], 'replays': prior_record['replay_batches']}
    counter.save(request, 'dz5-derived-replay-complete-readback.json', {
        'schema': 'filter_dz5_derived_replay_complete.v1', 'run': result['run'],
        'comparisons': comparisons,
        'nonclaims': ['No runtime, GPU, full-trajectory, convergence or main admission.']})
