"""Independent diagnostic value differences for the frozen actual DZ5 target."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_snapshot as snapshot_test

SNAPSHOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
    'filter-gradient-repair-20260917/dz5-candidate-source-score-4c37f9f40-r1')

ORACLE_CHECK = r'''
    import resource
    import threading
    import time
    import numpy as np  # Independent finite-difference diagnostic only.
    from two_currency_double_zlb_credit_target import (
        credit_target_value_and_score, credit_target_numeric_outputs)
    jit = os.environ['FILTER_REPAIR_DZ5_SCORE_JIT'] == '1'
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    qualifier = MF / 'scripts/qualify_dz5_cdf_runtime.py'
    assert sha(qualifier) == manifest['frozen_qualifier_sha256']
    center, scale = np.asarray(fixture.parameter_truth), np.asarray(fixture.prior_scale)
    steps = (1e-3, 5e-4)
    offsets = np.diag(scale)
    bank = np.concatenate((center[None], *(center + multiplier * step * offsets
        for step in steps for multiplier in (-2, -1, 1, 2))))
    assert bank.shape == (185, 23) and bank.dtype == np.float64
    target_device = '/GPU:0' if gpu else '/CPU:0'
    with tf.device(target_device):
        positions = tf.constant(bank)
    def numeric(p):
        if jit:
            fields = credit_target_numeric_outputs(p, fixture)
            return fields[0], fields[1], fields[3], fields[5]
        value, score, diagnostics = credit_target_value_and_score(p, fixture,
            jit_compile=False, evaluation_policy='hmc_rejection')
        return value, score, diagnostics['valid_pre_regularized_score'], diagnostics['branch_status_code']
    kernel = tf.function(numeric, input_signature=[tf.TensorSpec([185, 23], tf.float64)],
        jit_compile=jit, autograph=False)
    report.update(role='fresh_DZ5_independent_five_point_score_oracle',
        source_role=manifest['role'], jit_compile=jit, batch=185,
        target_evaluated=False, target_execution_attempted=True,
        frozen_qualifier_sha256=sha(qualifier), bayesfilter_commit=manifest['bayesfilter_commit'],
        nonclaims=['No new adapter admission, initializer, HMC, posterior or performance qualification.'],
        cpu_reference_exception=not gpu, graph_reference_exception=not jit,
        op_determinism_enabled=True,
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        affinity=sorted(os.sched_getaffinity(0)))
    def rss():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
        raise RuntimeError('VmRSS unavailable')
    samples = [rss()]
    stop = threading.Event()
    def sample_memory():
        while not stop.wait(.05):
            samples.append(rss())
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        tick = time.monotonic()
        with tf.device(target_device):
            result = kernel(positions)
        arrays = tuple(tensor.numpy() for tensor in result)
        report['cold_seconds'] = time.monotonic() - tick
        report['target_evaluated'] = True
        report['result_device'] = result[0].device
        assert ('DEVICE:GPU:0' if gpu else 'DEVICE:CPU:0') in result[0].device.upper()
        values, scores, validity, branch_status = arrays
        checks = []
        for index, step in enumerate(steps):
            blocks = np.split(values[1 + 92 * index:1 + 92 * (index + 1)], 4)
            finite_difference = (blocks[0] - 8 * blocks[1] + 8 * blocks[2] - blocks[3]) / (12 * step * scale)
            difference = np.abs(scores[0] - finite_difference)
            allowed = 1e-8 + 1e-7 * np.abs(finite_difference)
            checks.append({'step_prior_sd': step, 'finite_difference': finite_difference.tolist(),
                'absolute_error': difference.tolist(), 'allowed_error': allowed.tolist(),
                'max_absolute_error': float(np.max(difference)),
                'max_scaled_error': float(np.max(difference / allowed)),
                'passed': bool(np.all(difference <= allowed))})
        oracle = {'schema': 'filter_repair_dz5_fresh_score_oracle.v1',
            'bank': bank.tolist(), 'values': values.tolist(), 'scores': scores.tolist(),
            'validity': validity.tolist(), 'branch_status': branch_status.tolist(),
            'parameter_names': fixture.parameter_names, 'prior_scale': scale.tolist(),
            'checks': checks, 'jit_compile': jit, 'device': report['device'],
            'snapshot_manifest_sha256': report['snapshot_manifest_sha256'],
            'frozen_qualifier_sha256': report['frozen_qualifier_sha256']}
        # Save actual residuals before enforcing the unchanged gate.
        (OUT / 'dz5-score-oracle.json').write_text(json.dumps(diagnostic_json(oracle), indent=2, allow_nan=False) + '\n')
        report['score_checks'] = checks
        assert np.isfinite(values).all() and np.isfinite(scores).all()
        assert validity.all(), 'An oracle row was rejected'
        assert all(check['passed'] for check in checks), checks
        tick = time.monotonic()
        with tf.device(target_device):
            replay = kernel(positions)
        replay_arrays = tuple(tensor.numpy() for tensor in replay)
        report['replay_seconds'] = time.monotonic() - tick
        replay_record = {'values': replay_arrays[0].tolist(), 'scores': replay_arrays[1].tolist(),
            'validity': replay_arrays[2].tolist(), 'branch_status': replay_arrays[3].tolist(),
            'value_max_absolute_error': float(np.max(np.abs(arrays[0] - replay_arrays[0]))),
            'score_max_absolute_error': float(np.max(np.abs(arrays[1] - replay_arrays[1])))}
        (OUT / 'dz5-score-replay.json').write_text(json.dumps(diagnostic_json(replay_record), indent=2, allow_nan=False) + '\n')
        for left, right in zip(arrays, replay_arrays, strict=True):
            np.testing.assert_array_equal(left, right)
        report['replay_exact'] = True
        assert kernel.experimental_get_tracing_count() == 1
        definition = kernel.get_concrete_function().graph.as_graph_def()
        nodes = [*definition.node, *(node for function in definition.library.function for node in function.node_def)]
        forbidden = sorted({node.op for node in nodes if any(term in node.op.lower() for term in
            ('pyfunc', 'sylvester', 'principalsqrt', 'xlahostcompute'))})
        assert not forbidden, forbidden
        report.update(trace_count=1, host_callbacks=forbidden, graph_node_count=len(nodes))
        if jit:
            with tf.device(target_device):
                hlo = kernel.experimental_get_compiler_ir(positions)(stage='hlo', device_name=target_device)
            (OUT / 'dz5-score-oracle.hlo.txt').write_text(hlo)
            report['hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
    finally:
        stop.set()
        sampler.join(timeout=2)
        samples.append(rss())
        report.update(sampled_rss_before_bytes=samples[0], sampled_peak_rss_bytes=max(samples),
            sampled_rss_after_bytes=samples[-1], rss_sample_count=len(samples),
            host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:
            report['allocator'] = tf.config.experimental.get_memory_info('GPU:0' if gpu else 'CPU:0')
        except (ValueError, RuntimeError) as error:
            report['allocator'] = {'available': False, 'reason': str(error)}
'''


@pytest.mark.parametrize('jit', [False, True])
def test_dz5_fresh_score_oracle(request, monkeypatch, jit):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_SCORE_JIT', '1' if jit else '0')
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, ORACLE_CHECK + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='fresh_DZ5_five_point_score_oracle_reference_only_no_admission')


REPLAY_LOCALIZATION = r'''
    import numpy as np  # Independent diagnostic report only.
    from two_currency_double_zlb_credit_target import (
        build_credit_factor_model_and_derivatives, credit_fixed_branch)
    from bayesfilter.nonlinear.rectangular_srukf_tf import tf_rectangular_srukf_value_and_score
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    center, scale = np.asarray(fixture.parameter_truth), np.asarray(fixture.prior_scale)
    bank = np.concatenate((center[None], *(center + multiplier * step * np.diag(scale)
        for step in (1e-3, 5e-4) for multiplier in (-2, -1, 1, 2))))
    def evaluate(parameters):
        model, derivatives = build_credit_factor_model_and_derivatives(
            parameters, fixture, prepare_fixed_cir_points=True)
        observations = tf.constant(fixture.likelihood_observations()[:2], tf.float64)
        result = tf_rectangular_srukf_value_and_score(
            tf.broadcast_to(observations, [185, *observations.shape.as_list()]),
            model, derivatives, branch=credit_fixed_branch(fixture), jit_compile=False)
        return {'initial_mean': model.initial_mean, 'initial_factor': model.initial_factor,
            'd_initial_mean': derivatives.d_initial_mean, 'd_initial_factor': derivatives.d_initial_factor,
            'value': result.log_likelihood, 'score': result.score, 'mean': result.filtered_mean,
            'factor': result.filtered_factor, 'd_mean': result.d_filtered_mean,
            'd_factor': result.d_filtered_factor, 'valid': result.diagnostics['score_valid']}
    kernel = tf.function(evaluate, input_signature=[tf.TensorSpec([185, 23], tf.float64)],
        jit_compile=False, autograph=False)
    first = None
    summaries = []
    for repeat in range(3):
        current = {key: value.numpy() for key, value in kernel(tf.constant(bank)).items()}
        assert all(np.isfinite(value).all() for value in current.values())
        assert current['valid'].all()
        np.savez_compressed(OUT / f'replay-localization-{repeat}.npz', bank=bank, **current)
        if first is None:
            first = current
        else:
            summaries.append({key: {'changed_elements': int(np.count_nonzero(value != first[key])),
                'max_absolute_error': float(np.max(np.abs(value.astype(float) - first[key].astype(float))))}
                for key, value in current.items()})
    assert kernel.experimental_get_tracing_count() == 1
    report.update(role='CPU_graph_replay_localization_two_observations_not_oracle_qualification',
        threads=int(os.environ['TF_NUM_INTRAOP_THREADS']), target_evaluated=True,
        jit_compile=False, op_determinism_enabled=True, original_fixture_observations=96,
        evaluated_observations=2, batch=185, trace_count=1, replay_summaries=summaries)
    (OUT / 'dz5-replay-localization.json').write_text(json.dumps(report, indent=2) + '\n')
'''


def test_dz5_graph_replay_thread_localization(request, monkeypatch):
    directory = Path(request.config.getoption('xmlpath')).parent
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, REPLAY_LOCALIZATION + loaded_audit)
    for threads in (2, 1):
        target = directory / f'threads-{threads}'
        target.mkdir()
        monkeypatch.setenv('TF_NUM_INTRAOP_THREADS', str(threads))
        child_request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _name, p=target: str(p / 'junit.xml')))
        snapshot_test.run_isolated_snapshot(child_request, snapshot=SNAPSHOT,
            child=child, child_timeout_seconds=120,
            scope='CPU_graph_thread_replay_localization_only')
