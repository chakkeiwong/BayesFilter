"""Diagnostic-only binary-add intervention; never a runtime admission route."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_snapshot as snapshot_test
from tests.test_filter_repair_dz5_score_oracle import ORACLE_CHECK, SNAPSHOT


def _analyzer():
    path = Path(__file__).resolve().parents[1] / 'docs/plans/artifacts/filter-gradient-repair-20260917/analyze-dz5-addition-order-20260925.py'
    spec = importlib.util.spec_from_file_location('ordered_add_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_addition_order_evidence(request):
    result = _analyzer().analyze(SNAPSHOT.parent)
    assert not result['runs']['3908']['all_replays_exact']
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'addition-order-evidence.json').write_text(json.dumps(result, indent=2) + '\n')


@pytest.mark.parametrize('field', ['scores', 'bank', 'values', 'validity'])
def test_addition_oracle_rejects_corruption(field):
    oracle = json.loads((SNAPSHOT.parent / 'run-03908/dz5-score-oracle.json').read_text())
    if field in ('scores', 'bank'):
        oracle[field][0][0] += 1.
    elif field == 'values':
        oracle[field][1] += 1.
    else:
        oracle[field][0] = False
    with pytest.raises(AssertionError):
        _analyzer().oracle_errors(oracle)


OVERLAY = r'''
    import inspect
    import bayesfilter.nonlinear.rectangular_srukf_tf as rectangular
    original = rectangular.tf_rectangular_srukf_value_and_score
    source = inspect.getsource(original)
    needle = ('            d_predicted_points = (tf.einsum("brij,bprj->bpri", state_j, d_previous_points)\n'
        '                                  + tf.einsum("brij,bprj->bpri", process_j, d_process_points) + direct)')
    replacement = ('            d_predicted_points = tf.ensure_shape(ordered_binary_add(\n'
        '                ordered_binary_add(tf.einsum("brij,bprj->bpri", state_j, d_previous_points),\n'
        '                    tf.einsum("brij,bprj->bpri", process_j, d_process_points)), direct),\n'
        '                [b, p, 1 + 2 * latent_rank, n])')
    helper = ('    @tf.function(input_signature=[tf.TensorSpec([b, p, None, n], tf.float64)] * 2,\n'
        '        autograph=False, jit_compile=jit_compile, experimental_attributes={"_noinline": True})\n'
        '    def ordered_binary_add(left, right):\n'
        '        return left + right\n\n')
    assert source.count(needle) == 1
    assert source.count('    def run(obs: tf.Tensor):') == 1
    changed = source.replace(needle, replacement).replace('    def run(obs: tf.Tensor):',
        helper + '    def run(obs: tf.Tensor):')
    overlay_path = OUT / 'diagnostic-srukf-overlay.py'
    overlay_path.write_text(changed)
    namespace = dict(rectangular.__dict__)
    exec(compile(changed, str(overlay_path), 'exec'), namespace)
    rectangular.tf_rectangular_srukf_value_and_score = namespace[original.__name__]
    import two_currency_double_zlb_credit_target as credit_target
    credit_target.tf_rectangular_srukf_value_and_score = namespace[original.__name__]
    report['executable_overlay'] = {'role': 'diagnostic_only_not_runtime_qualification',
        'original_function_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'modified_function_sha256': sha(overlay_path), 'path': overlay_path.name,
        'description': 'No-inline binary transition tangent additions; enclosing JIT inherited.'}
'''

GRAPH_CAPTURE = r'''
    from tensorflow.python.eager import context as eager_context
    eager_context.enable_graph_collection()
    report['optimizer_options'] = tf.config.optimizer.get_experimental_options()
    assert report['optimizer_options'].get('arithmetic_optimization', True)
'''

GRAPH_EXPORT = r'''
    metadata = eager_context.export_run_metadata()
    eager_context.disable_graph_collection()
    graph_records = []
    for index, graphs in enumerate(metadata.function_graphs):
        for stage in ('pre_optimization_graph', 'post_optimization_graph'):
            definition = getattr(graphs, stage)
            path = OUT / f'ordered-graph-{index}-{stage}.pb'
            path.write_bytes(definition.SerializeToString())
            nodes = [*definition.node, *(node for function in definition.library.function
                for node in function.node_def)]
            functions = [{'name': function.signature.name,
                'noinline': function.attr['_noinline'].b,
                'ops': [node.op for node in function.node_def]}
                for function in definition.library.function
                if 'ordered_binary_add' in function.signature.name]
            graph_records.append({'stage': stage, 'path': path.name, 'sha256': sha(path),
                'node_count': len(nodes), 'ordered_functions': functions,
                'addn': [{'name': node.name, 'input': list(node.input)}
                    for node in nodes if node.op == 'AddN']})
    (OUT / 'ordered-additions.json').write_text(json.dumps(graph_records, indent=2) + '\n')
    assert graph_records
    post_functions = [function for row in graph_records if row['stage'] == 'post_optimization_graph'
        for function in row['ordered_functions']]
    assert post_functions and all(function['noinline'] and 'AddN' not in function['ops']
        and function['ops'].count('AddV2') == 1 for function in post_functions)
    report['ordered_function_count'] = len(post_functions)
'''


@pytest.mark.parametrize('horizon', [48, 96])
def test_dz5_transition_add_order_overlay(request, monkeypatch, horizon):
    """Original target outputs; no intermediate fetch instrumentation."""
    monkeypatch.setenv('FILTER_REPAIR_DZ5_SCORE_JIT', '0')
    check = ORACLE_CHECK
    needle = "        for left, right in zip(arrays, replay_arrays, strict=True):\n            np.testing.assert_array_equal(left, right)\n        report['replay_exact'] = True"
    replay = r'''
        replays = [replay_record]
        for repeat in range(2, 5):
            tick = time.monotonic()
            with tf.device(target_device):
                extra = tuple(t.numpy() for t in kernel(positions))
            record = {'repeat': repeat, 'seconds': time.monotonic() - tick,
                'values': extra[0].tolist(), 'scores': extra[1].tolist(),
                'validity': extra[2].tolist(), 'branch_status': extra[3].tolist(),
                'value_max_absolute_error': float(np.max(np.abs(arrays[0] - extra[0]))),
                'score_max_absolute_error': float(np.max(np.abs(arrays[1] - extra[1])))}
            replays.append(record)
            (OUT / f'dz5-score-replay-{repeat}.json').write_text(json.dumps(
                diagnostic_json(record), indent=2, allow_nan=False) + '\n')
        report['replay_exact'] = all(np.array_equal(arrays[i], row[key])
            for row in replays for i, key in enumerate(('values', 'scores', 'validity', 'branch_status')))
'''
    assert check.count(needle) == 1
    if horizon == 96:
        replay = replay.replace('range(2, 5)', 'range(2, 2)')
    check = check.replace(needle, replay)
    check = check.replace('    import resource', GRAPH_CAPTURE + '\n    import resource')
    check = check.replace("role='fresh_DZ5_independent_five_point_score_oracle'",
        "role='diagnostic_transition_addition_order_overlay_not_admitted'")
    # The complete fixture identity is checked first; only observations are sliced.
    prefix = f'''
    original_fixture = fixture
    prepared = fixture.likelihood_observations()
    class PrefixFixture:
        def likelihood_observations(self):
            return prepared[:{horizon}]
        def __getattr__(self, name):
            return getattr(original_fixture, name)
    if {horizon} != 96:
        fixture = PrefixFixture()
    report['evaluated_observations'] = {horizon}
'''
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK,
        prefix + check + GRAPH_EXPORT + loaded_audit + "\n    assert report['replay_exact']\n")
    import_needle = '    from two_currency_double_zlb_credit_target import CREDIT_FILTER_CONTRACT'
    assert child.count(import_needle) == 1
    child = child.replace(import_needle, OVERLAY + '\n' + import_needle)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='CPU_graph_diagnostic_transition_addition_order_overlay_only')


@pytest.mark.parametrize('jit', [False, True])
def test_ordered_add_primitive(request, jit):
    """Independent cancellation/gradient diagnostic before actual target use."""
    import numpy as np  # Independent comparison only.
    import tensorflow as tf
    from tensorflow.python.eager import context as eager_context

    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    tf.config.experimental.enable_op_determinism()
    eager_context.enable_graph_collection()
    directory = Path(request.config.getoption('xmlpath')).parent
    rows = []
    data = tf.constant([[5e15] * 257, [-5e15] * 257, [.5] * 257], tf.float64)
    def make_kernel(expose):
        @tf.function(input_signature=[tf.TensorSpec([257], tf.float64)] * 2,
                     autograph=False, jit_compile=jit, experimental_attributes={'_noinline': True})
        def ordered_binary_add(left, right):
            return left + right

        def calculate(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                terms = (2. * x[0], 2. * x[1], 2. * x[2])
                total = ordered_binary_add(ordered_binary_add(terms[0], terms[1]), terms[2])
                objective = tf.reduce_sum(total)
            gradient = tape.gradient(objective, x)
            if expose == 'first_two':
                return total, gradient, terms[0], terms[1]
            if expose == 'all':
                return total, gradient, *terms
            return total, gradient

        return tf.function(calculate, input_signature=[tf.TensorSpec([3, 257], tf.float64)],
            jit_compile=jit, autograph=False)

    for expose in ('none', 'first_two', 'all'):
        kernel = make_kernel(expose)
        observed = [tuple(t.numpy() for t in kernel(data)) for _ in range(4)]
        rows.append({'expose': expose, 'sum': [x[0].tolist() for x in observed],
            'gradient': observed[0][1].tolist(), 'trace_count': kernel.experimental_get_tracing_count()})
        for result in observed:
            np.testing.assert_array_equal(result[0], np.ones(257))
            np.testing.assert_array_equal(result[1], np.full([3, 257], 2.))
        assert kernel.experimental_get_tracing_count() == 1
        if jit:
            hlo = kernel.experimental_get_compiler_ir(data)(stage='hlo', device_name='/CPU:0')
            (directory / f'ordered-primitive-{expose}.hlo.txt').write_text(hlo)
    metadata = eager_context.export_run_metadata()
    eager_context.disable_graph_collection()
    protected = [function for graph in metadata.function_graphs
        for function in graph.post_optimization_graph.library.function
        if 'ordered_binary_add' in function.signature.name
        and function.attr['_noinline'].b]
    if not jit:
        assert protected
        assert all('AddN' not in {node.op for node in function.node_def} for function in protected)
    graph_path = directory / f'ordered-primitive-{jit}.pb'
    graph_path.write_bytes(metadata.SerializeToString())
    (directory / f'ordered-primitive-{jit}.json').write_text(json.dumps({
        'role': 'CPU_diagnostic_only', 'jit_compile': jit, 'records': rows,
        'graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest()}, indent=2) + '\n')
