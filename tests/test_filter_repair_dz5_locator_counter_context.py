"""CPU-only accounting dtype bisection; never a GPU/runtime candidate."""

import hashlib
import json

import numpy as np

from tests import test_filter_repair_dz5_locator_one_iteration as probe
from tests.test_filter_repair_dz5_locator_context_readback import array_comparison
from tests.test_filter_repair_geometry_control import save

PATCH = '''
    import ast
    candidate_source = Path(candidate_module.__file__).read_text()
    diagnostic_source = candidate_source.replace('tf.int64', 'tf.int32')
    class AccountingDtype(ast.NodeTransformer):
        count = 0
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'tf' and node.attr == 'int64':
                node.attr = 'int32'
                self.count += 1
            return self.generic_visit(node)
    change = AccountingDtype()
    changed_tree = change.visit(ast.parse(candidate_source))
    assert change.count > 0
    assert ast.dump(changed_tree, include_attributes=False) == ast.dump(ast.parse(diagnostic_source), include_attributes=False)
    diagnostic_path = OUT / 'cpu-accounting-int32-diagnostic.py'
    diagnostic_path.write_text(diagnostic_source)
    (OUT / 'frozen-candidate-locator.py').write_text(candidate_source)
    diagnostic_module = types.ModuleType('cpu_int32_locator_context_diagnostic')
    diagnostic_module.__file__ = str(diagnostic_path)
    exec(compile(diagnostic_source, str(diagnostic_path), 'exec'), diagnostic_module.__dict__)
    diagnostic_module.tfp = proxy
    BatchedLocalCenterProgram = diagnostic_module.BatchedLocalCenterProgram
    report['counter_context_patch'] = {
        'original_sha256': hashlib.sha256(candidate_source.encode()).hexdigest(),
        'diagnostic_sha256': hashlib.sha256(diagnostic_source.encode()).hexdigest(),
        'changed_dtype_attributes': change.count,
        'only_tf_int64_to_int32_AST_change': True,
        'cpu_diagnostic_only_not_GPU_candidate': True}
'''


def child_source(patch=PATCH):
    child = probe.child_source('candidate')
    anchor = '    candidate_module.tfp = proxy\n'
    assert child.count(anchor) == 1
    child = child.replace(anchor, anchor + patch)
    compile(child, '<counter_width_context>', 'exec')
    return child


def test_int32_accounting_context(request):
    report = probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=probe.context.trajectory.SNAPSHOT, child=child_source(),
        scope='CPU_int32_accounting_context_not_runtime_candidate', child_timeout_seconds=280,
        read_only_paths=(probe.context.trajectory.consumer.RAW,))
    assert report['counter_context_patch']['only_tf_int64_to_int32_AST_change']
    assert report['trace_count'] == 1 and not report['host_callbacks']


def check_saved_counter_context(request, group='dz5_locator_counter_int32_cpu',
        output='dz5-counter-context-readback.json',
        diagnostic_name='cpu-accounting-int32-diagnostic.py'):
    root = probe.context.trajectory.consumer.RAW
    found = []
    for path in sorted(root.glob('run-*/run.json')):
        run = json.loads(path.read_text())
        if run['key'][1] == group and run['state'] == 'passed':
            assert run['device'] == 'CPU'
            found.append(path.parent)
    assert len(found) == 1
    directory = found[0]
    report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    patch = report['counter_context_patch']
    assert patch['original_sha256'] == hashlib.sha256((directory / 'frozen-candidate-locator.py').read_bytes()).hexdigest()
    assert patch['diagnostic_sha256'] == hashlib.sha256((directory / diagnostic_name).read_bytes()).hexdigest()
    assert report['context_hlo_sha256'] == hashlib.sha256((directory / 'first-objective-context.hlo.txt').read_bytes()).hexdigest()
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert 1 <= report['optimizer_callback_batches'] <= 128
    with np.load(directory / 'locator-callbacks.npz', allow_pickle=False) as payload:
        actual = {key: payload[key][:2].copy() for key in payload.files}
    comparisons = {}
    for number in (4584, 4585, 4635, 4636):
        previous = root / f'run-{number:05d}'
        prior = json.loads((previous / 'dz5-snapshot-import.json').read_text())
        assert prior['snapshot_manifest_sha256'] == report['snapshot_manifest_sha256']
        with np.load(previous / 'locator-callbacks.npz', allow_pickle=False) as payload:
            reference = {key: payload[key][:2].copy() for key in payload.files}
        comparisons[str(number)] = {key: array_comparison(actual[key], reference[key]) for key in actual}
        assert all(comparisons[str(number)][key]['bitwise_equal'] for key in ('positions', 'values', 'valid'))
    result = {'schema': 'filter_dz5_counter_context.v1', 'run': directory.name,
        'patch': patch, 'comparisons': comparisons,
        'supports_accounting_width_context_effect': comparisons['4635']['scores']['bitwise_equal']
            and not comparisons['4636']['scores']['bitwise_equal'],
        'nonclaims': ['No GPU-compatible repair, exact compiler cause, full-record waiver or admission.']}
    save(request, output, result)


def test_saved_counter_context(request):
    check_saved_counter_context(request)
