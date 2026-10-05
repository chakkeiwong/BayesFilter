"""Isolated CPU accounting-family diagnostics; no runtime/GPU candidate."""

import ast
from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_locator_counter_context as counter

FAMILIES = {
    'index_calls': (
        '        best_indices = variable(tf.fill([batch_size], tf.constant(-1, tf.int64)), tf.int64)',
        '        calls = variable(0, tf.int64)',
        'tf.range(batch_size, dtype=tf.int64)',
        'best_indices.assign(tf.fill([batch_size], tf.constant(-1, tf.int64)))',
    ),
    'progress': tuple(f'        {name} = variable(0, tf.int64)' for name in
        ('attempts', 'optimizer_calls', 'round_calls', 'replays')),
    'invalid_rows': (
        '        invalid_rows = variable(0, tf.int64)',
        'invalid_rows.assign_add(tf.reduce_sum(tf.cast(~valid, tf.int64)))',
    ),
}
COUNTS = {'index_calls': 5, 'progress': 4, 'invalid_rows': 2}

TRANSFORM = '''    class AccountingDtype(ast.NodeTransformer):
        count = 0
        def generic_visit(self, node):
            for needle in needles:
                parsed = ast.parse(needle.strip()).body[0]
                target = parsed.value if isinstance(parsed, ast.Expr) else parsed
                if ast.dump(node, include_attributes=False) == ast.dump(target, include_attributes=False):
                    self.count += needle.count('tf.int64')
                    changed = ast.parse(needle.strip().replace('tf.int64', 'tf.int32')).body[0]
                    return changed.value if isinstance(changed, ast.Expr) else changed
            return super().generic_visit(node)
'''


def family_patch(family):
    patch = counter.PATCH
    before = "    diagnostic_source = candidate_source.replace('tf.int64', 'tf.int32')\n"
    replacement = f"    needles = {FAMILIES[family]!r}\n" + '''    diagnostic_source = candidate_source
    for needle in needles:
        assert diagnostic_source.count(needle) == 1
        diagnostic_source = diagnostic_source.replace(needle, needle.replace('tf.int64', 'tf.int32'))
'''
    assert patch.count(before) == 1
    patch = patch.replace(before, replacement)
    start = patch.index('    class AccountingDtype(')
    end = patch.index('    change = AccountingDtype()', start)
    patch = patch[:start] + TRANSFORM + patch[end:]
    patch = patch.replace('assert change.count > 0', f'assert change.count == {COUNTS[family]}')
    patch = patch.replace('cpu-accounting-int32-diagnostic.py', f'cpu-family-{family}-diagnostic.py')
    patch = patch.replace('only_tf_int64_to_int32_AST_change', 'only_selected_family_AST_change')
    patch = patch.replace("'original_sha256':", f"'family': {family!r}, 'original_sha256':")
    return patch


def test_family_partition():
    source = (Path(__file__).parents[1] / 'bayesfilter/inference/batched_local_center_tf.py').read_text()
    changed = source
    for family, needles in FAMILIES.items():
        assert sum(needle.count('tf.int64') for needle in needles) == COUNTS[family]
        for needle in needles:
            assert changed.count(needle) == 1
            changed = changed.replace(needle, needle.replace('tf.int64', 'tf.int32'))
        compile(counter.child_source(family_patch(family)), '<family_context>', 'exec')
    assert ast.dump(ast.parse(changed)) == ast.dump(ast.parse(source.replace('tf.int64', 'tf.int32')))


@pytest.mark.parametrize('family', tuple(FAMILIES))
def test_family_context(request, family):
    report = counter.probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=counter.probe.context.trajectory.SNAPSHOT,
        child=counter.child_source(family_patch(family)),
        scope=f'CPU_accounting_family_{family}_not_runtime_candidate', child_timeout_seconds=280,
        read_only_paths=(counter.probe.context.trajectory.consumer.RAW,))
    patch = report['counter_context_patch']
    assert patch['family'] == family and patch['changed_dtype_attributes'] == COUNTS[family]
    assert patch['only_selected_family_AST_change']
    assert report['trace_count'] == 1 and not report['host_callbacks']


@pytest.mark.parametrize('family', tuple(FAMILIES))
def test_saved_family_context(request, family):
    counter.check_saved_counter_context(request,
        group=f'dz5_locator_family_{family}_cpu',
        output=f'dz5-family-{family}-readback.json',
        diagnostic_name=f'cpu-family-{family}-diagnostic.py')
