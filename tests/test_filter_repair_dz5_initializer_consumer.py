"""Actual frozen CDF initializer and native resource lifetime diagnostics.

The training continuation is never called. The actual external target class,
configuration parser, initializer adapter and parent supervisor are exercised.
"""

import ast
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_snapshot as snapshot_test
from tests.test_filter_repair_dz5_initializer_target import SNAPSHOT, audited_child

RAW = SNAPSHOT.parent
ADMISSION = RAW / 'dz5-initializer-adapter-target-admission-20260928-r1.json'
ACCEPTED_INPUTS = RAW / 'dz5-initializer-accepted-inputs-20260928-r1'
MF = Path('/home/ubuntu/workspace/MacroFinance-dz5-neutra')
RECIPE = str(MF / 'docs/plans/artifacts/dz5-multi-asset-estimation-20260908/CDF/proposal-contract-r1/recipe.json')

INITIALIZER_CHECK = r'''
    import contextlib
    import gc
    import resource
    import time
    import weakref
    from dataclasses import replace
    from types import SimpleNamespace
    from bayesfilter_estimation import EstimationRecipe
    from bayesfilter_estimation_initialization import initialize_dense_local
    from two_currency_double_zlb_credit_estimation import credit_model
    from two_currency_double_zlb_credit_recovery import CreditTrainingTarget
    from bayesfilter.inference import dense_initializer_seeded_tf as seeded
    tf.config.experimental.enable_tensor_float_32_execution(False)
    recipe_path = Path(RECIPE_PATH)
    admission_path = Path(ADMISSION_PATH)
    recipe = EstimationRecipe(**json.loads(recipe_path.read_text()))
    admission = json.loads(admission_path.read_text())
    assert admission['role'] == 'fresh_target_only_isolated_initializer_regression'
    assert admission['snapshot_manifest_sha256'] == report['snapshot_manifest_sha256']
    assert not any(admission[key] for key in ('old_admission_reused', 'training_authorized',
        'hmc_authorized', 'retained_authorized'))
    model = credit_model(stage='CDF', qualification=admission_path, recipe=recipe)
    base = model.sampling_target
    assert base.fixture_identity_sha256 == report['fixture_identity']
    model = replace(model, training_target=CreditTrainingTarget(base, base.fixture.prior_mean, model.data_identity))
    if ACCEPTED_CASE:
        replay_started = time.monotonic()
        inputs_root = Path(ACCEPTED_INPUT_PATH)
        inputs = json.loads((inputs_root / 'manifest.json').read_text())
        assert all(sha(row['saved']) == row['sha256'] for row in inputs['files'].values())
        point_row = inputs['files']['start_point']
        point = json.loads(Path(point_row['saved']).read_text())
        assert point['fixture_identity'] == base.fixture_identity_sha256
        assert point['admission_sha256'] == inputs['files']['original_admission']['sha256']
        assert point['optimizer_state_reused'] is False
        for source, digest in point['artifacts'].items():
            rows = [row for row in inputs['files'].values() if row['source'] == source]
            assert len(rows) == 1 and rows[0]['sha256'] == digest
        with tf.device('/GPU:0' if gpu else '/CPU:0'):
            position = tf.constant([point['point']], tf.float64)
            value, score, status, valid, *_ = base._evaluate(position)
        assert bool(tf.reduce_all(valid & (status == 0)))
        tf.debugging.assert_less_equal(tf.abs(value - tf.constant([point['value']], tf.float64)),
            tf.constant(1e-8, tf.float64) + tf.constant(1e-9, tf.float64) * tf.abs(tf.constant([point['value']], tf.float64)))
        tf.debugging.assert_less_equal(tf.abs(score - tf.constant([point['score']], tf.float64)),
            tf.constant(1e-8, tf.float64) + tf.constant(1e-7, tf.float64) * tf.abs(tf.constant([point['score']], tf.float64)))
        fresh_point = {'role': 'fresh_value_score_replay_of_frozen_prior_origin_initializer_start',
            'point': position[0].numpy().tolist(), 'value': float(value[0]), 'score': score[0].numpy().tolist(),
            'original_point_sha256': point_row['sha256'],
            'original_admission_sha256': point['admission_sha256'],
            'current_admission_sha256': sha(admission_path), 'fixture_identity': base.fixture_identity_sha256,
            'optimizer_state_reused': False, 'training_or_HMC_authorized': False}
        (OUT / 'initializer-start-replay.json').write_text(json.dumps(fresh_point, indent=2, allow_nan=False) + '\n')
        model = replace(model, initial_position=position[0])
        report.update(start_replay_seconds=time.monotonic() - replay_started,
            accepted_input_manifest_sha256=sha(inputs_root / 'manifest.json'),
            start_replay=fresh_point)
    events, owners, refs = [], [], []
    original = seeded.SeededDenseInitializerProgram
    class ObservedProgram(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owners.append(self)
            refs.append(weakref.ref(self))
    seeded.SeededDenseInitializerProgram = ObservedProgram
    def memory():
        status = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSize', 'RssAnon', 'RssFile'):
                status[key + '_bytes'] = int(value.split()[0]) * 1024
        return {'process': status, 'map_count': len(Path('/proc/self/maps').read_text().splitlines()),
            'allocator': tf.config.experimental.get_memory_info('GPU:0') if gpu else None,
            'live_observed_owners': sum(ref() is not None for ref in refs)}
    @contextlib.contextmanager
    def boundary(name):
        events.append({'event': 'start', 'name': name, 'monotonic': time.monotonic()})
        try:
            yield
        finally:
            events.append({'event': 'end', 'name': name, 'monotonic': time.monotonic()})
    context = SimpleNamespace(root=OUT, boundary=boundary)
    report.update(role='actual_CDF_initializer_engineering_regression',
        initializer_case='historically_accepted_r2' if ACCEPTED_CASE else 'historically_rejected_r1',
        recipe_sha256=sha(recipe_path), admission_sha256=sha(admission_path),
        target_execution_attempted=True, target_evaluated=False,
        jit_compile=True, cpu_reference_exception=not gpu,
        nonclaims=['No training continuation, HMC, posterior admission or native cache eviction.'])
    report['memory_before'] = memory()
    tick = time.monotonic()
    target_device = '/GPU:0' if gpu else '/CPU:0'
    with tf.device(target_device):
        result = initialize_dense_local(model, recipe, context)
    report['initializer_seconds'] = time.monotonic() - tick
    report['target_evaluated'] = True
    report['initializer_result'] = result
    report['initializer_accepted'] = result['passed']
    report['boundaries'] = events
    report['memory_after_call'] = memory()
    # Preserve the complete rejection/acceptance before any qualification gate.
    (OUT / 'initialize.json').write_text(json.dumps(diagnostic_json(result), indent=2, allow_nan=False) + '\n')
    assert len(owners) == 1
    owner = owners.pop()
    report['compiled_owner_count'] = 1
    report['trace_count'] = owner.compiled.experimental_get_tracing_count()
    report['input_signature'] = str(owner.compiled.input_signature)
    report['target_compiled_batches'] = sorted(base._compiled)
    report['target_trace_counts'] = {str(key): value.experimental_get_tracing_count()
        for key, value in base._compiled.items()}
    assert report['trace_count'] == 1
    assert all(count == 1 for count in report['target_trace_counts'].values())
    definition = owner.compiled.get_concrete_function().graph.as_graph_def()
    nodes = [*definition.node, *(node for f in definition.library.function for node in f.node_def)]
    report['host_callbacks'] = sorted({n.op for n in nodes if any(term in n.op.lower()
        for term in ('pyfunc', 'xlahostcompute'))})
    assert not report['host_callbacks']
    from bayesfilter_estimation_runtime import initializer_configurations
    configuration, _, _ = initializer_configurations(recipe)
    hlo_started = time.monotonic()
    with tf.device(target_device):
        operands = (tf.constant(model.initial_position, tf.float64),
            tf.constant(model.coordinate_scale, tf.float64),
            tf.constant(configuration.seed, tf.int32),
            tf.constant(configuration.curvature_radius, tf.float64))
        hlo = owner.compiled.experimental_get_compiler_ir(*operands)(stage='hlo')
    (OUT / 'dz5-initializer.hlo.txt').write_text(hlo)
    report['hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
    report['hlo_inspection_seconds'] = time.monotonic() - hlo_started
    assert base.assert_source_pinned()
    report['memory_before_release'] = memory()
    del owner, definition, nodes, operands, base, model
    gc.collect()
    report['memory_after_python_release'] = memory()
    assert report['memory_after_python_release']['live_observed_owners'] == 0
    report['host_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
'''


def child_source(*, accepted=False):
    audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    recipe = str(ACCEPTED_INPUTS / 'recipe.json') if accepted else RECIPE
    body = (INITIALIZER_CHECK.replace('RECIPE_PATH', repr(recipe))
        .replace('ADMISSION_PATH', repr(str(ADMISSION)))
        .replace('ACCEPTED_CASE', repr(accepted))
        .replace('ACCEPTED_INPUT_PATH', repr(str(ACCEPTED_INPUTS))))
    return audited_child(merged.target_child().replace(merged.TARGET_CHECK, body + audit))


def test_actual_cdf_initializer(request):
    """Preserved initial mis-specified acceptance assertion; explanatory only."""
    assert ADMISSION.is_file(), 'Fresh target evidence must precede actual initializer execution'
    report = snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child_source(), scope='actual_initializer_only_no_training_or_HMC',
        child_timeout_seconds=840, read_only_paths=(RAW,))
    assert report['initializer_accepted'], report['initializer_result']


def test_saved_r1_is_rejection_evidence_only():
    directory = RAW / 'run-04569'
    run = json.loads((directory / 'run.json').read_text())
    report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    result = json.loads((directory / 'initialize.json').read_text())
    assert run['state'] == 'failed' and report['passed']
    assert report['initializer_result'] == result
    assert result['passed'] is False and result['status'] == 'dense_center_score_above_cap'
    assert result['exact_evaluation_count'] == 252
    assert result['initial_output_shift'] is None and result['initial_output_scale_log'] is None
    assert result['attempts'][0]['scaled_center_score_l2'] > .02
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report['memory_after_python_release']['live_observed_owners'] == 0
    assert hashlib.sha256((directory / 'dz5-initializer.hlo.txt').read_bytes()).hexdigest() == report['hlo_sha256']


@pytest.mark.parametrize('accepted', [False, True])
def test_declared_initializer_case(request, accepted):
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    report = snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child_source(accepted=accepted), scope='declared_actual_initializer_case_no_training_or_HMC',
        child_timeout_seconds=840 if gpu else 1700, read_only_paths=(RAW,))
    assert report['initializer_accepted'] is accepted, report['initializer_result']
    if not accepted:
        assert report['initializer_result']['status'] == 'dense_center_score_above_cap'
        assert report['initializer_result']['initial_output_scale_log'] is None


def supervisor():
    source = SNAPSHOT / str(MF / 'scripts/run_bayesfilter_estimation.py').lstrip('/')
    caller = RAW / 'dz5-initializer-supervisor-callsite-20260928-r1/run_dz5_cdf_proposal.py'
    callsite = json.loads((caller.parent / 'manifest.json').read_text())
    assert hashlib.sha256(caller.read_bytes()).hexdigest() == callsite['sha256']
    caller_tree = ast.parse(caller.read_text())
    assert any(isinstance(node, ast.ImportFrom) and node.module == 'scripts.run_bayesfilter_estimation'
        and any(alias.name == 'supervise' for alias in node.names) for node in ast.walk(caller_tree))
    calls = [node for node in ast.walk(caller_tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'supervise']
    assert len(calls) == 1 and any(k.arg == 'timeout_seconds' and ast.unparse(k.value) == 'cap'
        for k in calls[0].keywords)
    text = source.read_text()
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'supervise')
    imports = '\n'.join(ast.get_source_segment(text, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    excerpt = imports + '\n' + ast.get_source_segment(text, function)
    namespace = {'REPO_ROOT': MF}
    exec(compile(excerpt, str(source), 'exec'), namespace)  # noqa: S102 - exact diagnostic source
    return namespace['supervise'], {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'caller_sha256': hashlib.sha256(caller.read_bytes()).hexdigest(),
        'function_sha256': hashlib.sha256(excerpt.encode()).hexdigest()}


def parent_memory():
    rss = next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
    return {'rss_bytes': int(rss.split()[1]) * 1024, 'maps': len(Path('/proc/self/maps').read_text().splitlines())}


@pytest.mark.parametrize('workers', [2])
def test_actual_initializer_supervisor_lifetime(request, workers):
    directory = Path(request.config.getoption('xmlpath')).parent
    exact_supervise, provenance = supervisor()
    observations, results, receipts = [parent_memory()], [], []

    def measured_supervise(command, **kwargs):
        receipt = exact_supervise(command, **kwargs)
        receipts.append(receipt)
        assert not Path(f"/proc/{receipt['pid']}").exists(), 'Initializer child was not reaped'
        return receipt

    for index in range(workers):
        child_dir = directory / f'initializer-worker-{index}'
        child_dir.mkdir()
        child_request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _, d=child_dir: str(d / 'child.xml')))
        report = snapshot_test.run_isolated_snapshot(child_request, snapshot=SNAPSHOT,
            child=child_source(accepted=True), scope='actual_initializer_supervised_lifetime_no_training_or_HMC',
            child_timeout_seconds=1700 if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 850,
            read_only_paths=(RAW,), supervisor=measured_supervise)
        results.append(report)
        gc.collect()
        observations.append(parent_memory())
    report = {'schema': 'filter_dz5_initializer_supervisor_lifetime.v1',
        'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU',
        'supervisor': provenance, 'receipts': receipts, 'parent_memory': observations,
        'initializer_accepted': [r['initializer_accepted'] for r in results],
        'child_memory': [{k: r[k] for k in ('memory_before', 'memory_after_call', 'memory_after_python_release')}
            for r in results], 'all_children_reaped': True,
        'nonclaims': ['Two complete workers bound the observed parent trajectory; no general eviction or posterior claim.']}
    (directory / 'dz5-initializer-lifetime.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(r['initializer_accepted'] for r in results)
    assert all(not row['timed_out'] and row['returncode'] == 0 for row in receipts)
    assert observations[-1]['rss_bytes'] - observations[0]['rss_bytes'] < 256 * 1024**2
