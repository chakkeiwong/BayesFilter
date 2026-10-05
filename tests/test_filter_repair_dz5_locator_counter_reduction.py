"""Keep int64 resources and isolate only invalid-row reduction width on CPU."""

from tests import test_filter_repair_dz5_locator_counter_context as counter

BEFORE = 'tf.reduce_sum(tf.cast(~valid, tf.int64))'
AFTER = 'tf.cast(tf.reduce_sum(tf.cast(~valid, tf.int32)), tf.int64)'
TRANSFORM = '''    class AccountingDtype(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            before = ast.parse(BEFORE, mode='eval').body
            if ast.dump(node, include_attributes=False) == ast.dump(before, include_attributes=False):
                self.count += 1
                return ast.parse(AFTER, mode='eval').body
            return self.generic_visit(node)
'''.replace('BEFORE', repr(BEFORE)).replace('AFTER', repr(AFTER))


def child_source():
    patch = counter.PATCH
    patch = patch.replace("candidate_source.replace('tf.int64', 'tf.int32')",
        f'candidate_source.replace({BEFORE!r}, {AFTER!r})')
    start = patch.index('    class AccountingDtype(')
    end = patch.index('    change = AccountingDtype()', start)
    patch = patch[:start] + TRANSFORM + patch[end:]
    patch = patch.replace('assert change.count > 0', 'assert change.count == 1')
    patch = patch.replace('only_tf_int64_to_int32_AST_change', 'only_invalid_row_reduction_AST_change')
    patch = patch.replace('cpu-accounting-int32-diagnostic.py', 'cpu-count-reduction-diagnostic.py')
    return counter.child_source(patch)


def test_int32_reduction_context(request):
    report = counter.probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=counter.probe.context.trajectory.SNAPSHOT, child=child_source(),
        scope='CPU_int32_reduction_int64_resources_not_runtime_candidate', child_timeout_seconds=280,
        read_only_paths=(counter.probe.context.trajectory.consumer.RAW,))
    assert report['counter_context_patch']['only_invalid_row_reduction_AST_change']
    assert report['counter_context_patch']['changed_dtype_attributes'] == 1
    assert report['trace_count'] == 1 and not report['host_callbacks']


def test_saved_reduction_context(request):
    counter.check_saved_counter_context(request, group='dz5_locator_counter_reduction_cpu',
        output='dz5-counter-reduction-readback.json', diagnostic_name='cpu-count-reduction-diagnostic.py')
