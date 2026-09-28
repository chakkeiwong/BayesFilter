"""CPU-only control/reporting counter bisection; no runtime promotion."""

import ast
from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_locator_accounting_families as families
from tests import test_filter_repair_dz5_locator_counter_context as counter

FAMILIES = {
    'round_budget': ('        round_calls = variable(0, tf.int64)',),
    'reporting': tuple(f'        {name} = variable(0, tf.int64)' for name in
        ('attempts', 'optimizer_calls', 'replays')),
}


def patch(family):
    original = families.family_patch('progress')
    original = original.replace(repr(families.FAMILIES['progress']), repr(FAMILIES[family]))
    original = original.replace('assert change.count == 4', f'assert change.count == {len(FAMILIES[family])}')
    original = original.replace("'family': 'progress'", f"'family': {family!r}")
    original = original.replace('cpu-family-progress-diagnostic.py', f'cpu-progress-{family}-diagnostic.py')
    return original


def test_progress_partition():
    assert sorted(needle for needles in FAMILIES.values() for needle in needles) == sorted(families.FAMILIES['progress'])
    source = (Path(__file__).parents[1] / 'bayesfilter/inference/batched_local_center_tf.py').read_text()
    for family, needles in FAMILIES.items():
        changed = source
        for needle in needles:
            assert changed.count(needle) == 1
            changed = changed.replace(needle, needle.replace('tf.int64', 'tf.int32'))
        assert len([node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute) and node.attr == 'int64']) - len([
            node for node in ast.walk(ast.parse(changed)) if isinstance(node, ast.Attribute) and node.attr == 'int64']) == len(needles)
        compile(counter.child_source(patch(family)), '<progress_context>', 'exec')


@pytest.mark.parametrize('family', tuple(FAMILIES))
def test_progress_context(request, family):
    report = counter.probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=counter.probe.context.trajectory.SNAPSHOT,
        child=counter.child_source(patch(family)),
        scope=f'CPU_progress_counter_{family}_not_runtime_candidate', child_timeout_seconds=280,
        read_only_paths=(counter.probe.context.trajectory.consumer.RAW,))
    evidence = report['counter_context_patch']
    assert evidence['family'] == family and evidence['changed_dtype_attributes'] == len(FAMILIES[family])
    assert evidence['only_selected_family_AST_change']
    assert report['trace_count'] == 1 and not report['host_callbacks']


@pytest.mark.parametrize('family', tuple(FAMILIES))
def test_saved_progress_context(request, family):
    counter.check_saved_counter_context(request,
        group=f'dz5_locator_progress_{family}_cpu',
        output=f'dz5-progress-{family}-readback.json',
        diagnostic_name=f'cpu-progress-{family}-diagnostic.py')
