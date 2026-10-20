"""Diagnostic healthy/rejected LEDH fixtures; no canonical admission authority."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_value_program_tf import (
    make_canonical_value_program,
)
from tests import test_filter_repair_ledh_value_native as native

authorities = native.authorities


@pytest.mark.parametrize('case,horizon,stages,cap,dual,expected_valid', [
    ('composed', 3, 3, 0.8, False, True),
    ('dual_trust', 3, 1, float('inf'), True, False),
    ('one_step', 1, 1, float('inf'), False, False),
])
def test_checked_validity_boundary(
    authorities, request, case, horizon, stages, cap, dual, expected_valid
):
    baseline, fixture, baseline_sha = authorities
    model = fixture._lgssm_model(13, horizon=horizon)
    callbacks = fixture._callbacks_for_lgssm(model)
    observations = tf.constant(model['observations'], tf.float64)
    rng = np.random.default_rng(123)
    inputs = (tf.constant(rng.normal(size=(8, 2)), tf.float64),
        tf.constant(rng.normal(size=(horizon, 8, 2)), tf.float64),
        tf.constant(rng.uniform(size=(horizon, stages)), tf.float64))
    controls = {'flow_substeps': 3, 'temper_stages': stages, 'annealed_resampling': False,
        'flow_prior_cap': cap, 'sinkhorn_steps': 2, 'balance_steps': 2,
        'dual_cap_enabled': dual, 'trust_region_enabled': dual}
    program = make_canonical_value_program(callbacks,
        tf.TensorSpec(observations.shape, tf.float64), particle_count=8, **controls)
    reference = native._frozen_reference(baseline, callbacks, observations, inputs, controls)
    actual = program(observations, *inputs)
    report = {'case': case, 'baseline_sha256': baseline_sha,
        'flow_baseline_sha256': baseline._flow_source_sha256, 'controls': controls,
        'source_device': actual['value'].device, 'tensorflow': tf.__version__,
        'reference': {}, 'actual': {},
        'nonclaims': ['Supplied-input owner qualification only; rejected raw programs are not equivalent usable likelihoods.',
            'No registered-consumer, canonical LEDH, performance or scientific admission.']}
    for name, result in (('reference', reference), ('actual', actual)):
        report[name] = {key: value.numpy().tolist() for key, value in result.items() if key != 'model_id'}
    report['input_sha256'] = [hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        for tensor in (observations, *inputs)]
    destination = Path(request.config.getoption('xmlpath')).parent / f'ledh-validity-{case}.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    assert bool(reference['numerical_valid']) is expected_valid
    assert bool(actual['numerical_valid']) is expected_valid
    assert bool(reference['program_valid']) and bool(actual['finite_program_valid'])
    assert bool(actual['program_valid']) is expected_valid
    np.testing.assert_array_equal(actual['per_step_reset_valid'], reference['per_step_reset_valid'])
    if expected_valid:
        report['healthy_complete_record_errors'] = native._compare(actual, reference)
        assert int(actual['numerical_failure_code']) == 0
        assert int(actual['first_rejected_reset_index']) == -1
        np.testing.assert_array_equal(actual['value'], actual['raw_value'])
    else:
        assert np.isnan(float(actual['value']))
        assert int(actual['numerical_failure_code']) == 2
        assert int(actual['first_rejected_reset_index']) == 0
        assert np.isfinite(float(actual['raw_value']))
    report['raw_value_absolute_difference'] = abs(float(actual['raw_value']) - float(reference['value']))
    replay = program(observations, *inputs)
    for key, value in actual.items():
        np.testing.assert_array_equal(value, replay[key], err_msg=key)
    changed_inputs = (inputs[0] + 0.15, inputs[1] - 0.05, 1.0 - inputs[2])
    changed_observations = observations + 0.1
    changed = program(changed_observations, *changed_inputs)
    changed_reference = native._frozen_reference(
        baseline, callbacks, changed_observations, changed_inputs, controls)
    assert bool(changed['program_valid']) == bool(changed_reference['numerical_valid'])
    np.testing.assert_array_equal(changed['per_step_reset_valid'], changed_reference['per_step_reset_valid'])
    if bool(changed['program_valid']):
        report['changed_healthy_record_errors'] = native._compare(changed, changed_reference)
    else:
        assert np.isnan(float(changed['value']))
    report['changed'] = {key: value.numpy().tolist() for key, value in changed.items()}
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    operations.update(node.op for function in graph.library.function for node in function.node_def)
    assert not operations & {'PyFunc', 'PyFuncStateless', 'EagerPyFunc', 'XlaHostCompute'}
    hlo = program.experimental_get_compiler_ir(observations, *inputs)(stage='hlo')
    hlo_path = destination.with_suffix('.hlo.txt')
    hlo_path.write_text(hlo)
    report['hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
    report['trace_count'] = 1
    report['passed'] = True
    destination.write_text(json.dumps(report, indent=2) + '\n')


def test_saved_boundary(request):
    raw = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
    before, after = raw / 'run-04662', raw / 'run-04663'
    runs = [json.loads((root / 'run.json').read_text()) for root in (before, after)]
    assert all(row['state'] == 'passed' and row['device'] == 'CPU' for row in runs)
    assert runs[0]['environment'] == runs[1]['environment']
    sources = set(runs[0]['source_sha256']) | set(runs[1]['source_sha256'])
    changed_runtime = [name for name in sources if name.startswith('bayesfilter/') and
        runs[0]['source_sha256'].get(name) != runs[1]['source_sha256'].get(name)]
    assert changed_runtime == ['bayesfilter/highdim/ledh_canonical_value_program_tf.py']
    changed_path = Path(__file__).resolve().parents[1] / changed_runtime[0]
    assert hashlib.sha256(changed_path.read_bytes()).hexdigest() == runs[1]['source_sha256'][changed_runtime[0]]
    cases = {}
    for case in ('composed', 'dual_trust', 'one_step'):
        files = [root / f'ledh-validity-{case}.json' for root in (before, after)]
        old, current = [json.loads(path.read_text()) for path in files]
        for field in ('input_sha256', 'baseline_sha256', 'flow_baseline_sha256', 'controls'):
            assert old[field] == current[field]
        for key, value in old['actual'].items():
            destination = {'value': 'raw_value', 'program_valid': 'finite_program_valid'}.get(key, key)
            lhs, rhs = np.asarray(value), np.asarray(current['actual'][destination])
            assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape
            assert lhs.tobytes() == rhs.tobytes(), (case, key)
        valid = current['actual']['program_valid']
        assert valid is (case == 'composed')
        assert current['actual']['numerical_failure_code'] == (0 if valid else 2)
        assert current['actual']['first_rejected_reset_index'] == (-1 if valid else 0)
        if not valid:
            assert np.isnan(current['actual']['value'])
        cases[case] = {'all_prior_raw_fields_bitwise_unchanged': True,
            'program_valid': valid, 'failure_code': current['actual']['numerical_failure_code'],
            'before_sha256': hashlib.sha256(files[0].read_bytes()).hexdigest(),
            'after_sha256': hashlib.sha256(files[1].read_bytes()).hexdigest()}
    result = {'schema': 'filter_ledh_validity_boundary_readback.v1', 'cases': cases,
        'runs': [4662, 4663], 'manifest_sha256': [hashlib.sha256((root / 'run.json').read_bytes()).hexdigest()
            for root in (before, after)], 'only_changed_runtime': changed_runtime,
        'nonclaims': ['CPU supplied-input owner qualification only; no registered wrapper or GPU admission.',
            'Before/after raw invariance does not make rejected eager/XLA raw values equivalent.']}
    destination = Path(request.config.getoption('xmlpath')).parent / 'ledh-validity-boundary-readback.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')


def test_saved_gpu_boundary(request):
    raw = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
    source = 'bayesfilter/highdim/ledh_canonical_value_program_tf.py'
    source_sha = hashlib.sha256((Path(__file__).resolve().parents[1] / source).read_bytes()).hexdigest()
    evidence = {}
    for number, group, count in ((4666, 'ledh_validity_boundary_gpu', 3),
            (4667, 'ledh_validity_regressions_gpu', 9)):
        root = raw / f'run-{number:05d}'
        manifest = json.loads((root / 'run.json').read_text())
        assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
        assert manifest['key'][1] == group
        assert manifest['test_evidence']['tests'] == count
        assert manifest['source_sha256'][source] == source_sha
        assert all(not sample['desktop_fallback'] for sample in manifest['gpu_preflight'])
        contexts = [json.loads(line) for line in (root / 'process.log').read_text().splitlines()
            if line.startswith('{"tensorflow_version"')]
        assert len(contexts) == 1
        context = contexts[0]
        assert context['cuda_visible_devices'] == manifest['gpu_uuid']
        assert context['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        memory = context['gpu_memory_policy']
        assert memory['mode'] == 'memory_growth'
        assert memory['configured_before_logical_device_initialization']
        assert memory['all_physical_devices_memory_growth']
        assert memory['full_device_preallocation_disabled']
        assert all(device['memory_growth'] for device in memory['physical_devices'])
        evidence[number] = {'manifest_sha256': hashlib.sha256((root / 'run.json').read_bytes()).hexdigest(),
            'log_sha256': hashlib.sha256((root / 'process.log').read_bytes()).hexdigest(),
            'context': context, 'tests': count}
    cases = {}
    root = raw / 'run-04666'
    for case in ('composed', 'dual_trust', 'one_step'):
        path = root / f'ledh-validity-{case}.json'
        row = json.loads(path.read_text())
        assert row['passed'] and row['trace_count'] == 1
        assert 'GPU:0' in row['source_device']
        assert hashlib.sha256(path.with_suffix('.hlo.txt').read_bytes()).hexdigest() == row['hlo_sha256']
        valid = case == 'composed'
        assert row['actual']['program_valid'] is valid
        assert row['actual']['numerical_failure_code'] == (0 if valid else 2)
        if not valid:
            assert np.isnan(row['actual']['value'])
        cases[case] = {'program_valid': valid,
            'failure_code': row['actual']['numerical_failure_code'],
            'raw_value_absolute_difference': row['raw_value_absolute_difference'],
            'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {'schema': 'filter_ledh_validity_GPU_readback.v1', 'runs': evidence,
        'cases': cases, 'source_sha256': source_sha,
        'nonclaims': ['GPU supplied-input owner qualification; registered wrapper remains unmigrated.',
            'No before/after GPU cost or bitwise raw-invariance measurement, canonical LEDH, or scientific admission.']}
    destination = Path(request.config.getoption('xmlpath')).parent / 'ledh-validity-GPU-readback.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
