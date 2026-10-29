"""Diagnostic-only prefix/operator localization of the frozen DZ5 graph replay."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_snapshot as snapshot_test
from tests.test_filter_repair_dz5_score_oracle import ORACLE_CHECK, SNAPSHOT

PREFIX_CHECK = r'''
    import resource
    import time
    import numpy as np  # Independent replay inspection/serialization only.
    from two_currency_double_zlb_credit_target import (
        build_credit_factor_model_and_derivatives, credit_fixed_branch,
        credit_target_value_and_score)
    from bayesfilter.nonlinear.rectangular_srukf_tf import tf_rectangular_srukf_value_and_score
    assert not gpu
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    horizon = int(os.environ['FILTER_REPAIR_DZ5_PREFIX'])
    route = os.environ['FILTER_REPAIR_DZ5_REPLAY_ROUTE']
    assert route in ('direct', 'public', 'public_tuple') and 1 <= horizon <= 96
    prepared = fixture.likelihood_observations()
    class PrefixFixture:
        def likelihood_observations(self):
            return prepared[:horizon]
        def __getattr__(self, name):
            return getattr(fixture, name)
    prefix_fixture = PrefixFixture()
    np.testing.assert_array_equal(prefix_fixture.likelihood_observations(), prepared[:horizon])
    center, scale = np.asarray(fixture.parameter_truth), np.asarray(fixture.prior_scale)
    bank = np.concatenate((center[None], *(center + multiplier * step * np.diag(scale)
        for step in (1e-3, 5e-4) for multiplier in (-2, -1, 1, 2))))
    assert bank.shape == (185, 23)
    def evaluate(parameters):
        if route in ('public', 'public_tuple'):
            value, score, diagnostics = credit_target_value_and_score(
                parameters, prefix_fixture, jit_compile=False, evaluation_policy='hmc_rejection')
            if route == 'public_tuple':
                return value, score, diagnostics['valid_pre_regularized_score'], diagnostics['branch_status_code']
            return {'value': value, 'score': score,
                'valid': diagnostics['valid_pre_regularized_score'],
                'branch_status': diagnostics['branch_status_code']}
        model, derivatives = build_credit_factor_model_and_derivatives(
            parameters, fixture, prepare_fixed_cir_points=True)
        observations = tf.constant(prefix_fixture.likelihood_observations(), tf.float64)
        result = tf_rectangular_srukf_value_and_score(
            tf.broadcast_to(observations, [185, *observations.shape.as_list()]),
            model, derivatives, branch=credit_fixed_branch(fixture), jit_compile=False)
        return {'initial_mean': model.initial_mean, 'initial_factor': model.initial_factor,
            'd_initial_mean': derivatives.d_initial_mean, 'd_initial_factor': derivatives.d_initial_factor,
            'value': result.log_likelihood, 'score': result.score, 'mean': result.filtered_mean,
            'factor': result.filtered_factor, 'd_mean': result.d_filtered_mean,
            'd_factor': result.d_filtered_factor, 'valid': result.diagnostics['score_valid'],
            'branch_status': result.diagnostics['branch_status_code']}
    kernel = tf.function(evaluate, input_signature=[tf.TensorSpec([185, 23], tf.float64)],
        jit_compile=False, autograph=False)
    def rss():
        return next(int(line.split()[1]) * 1024 for line in
            Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
    report.update(role='CPU_graph_prefix_replay_explanatory_only', route=route,
        threads=int(os.environ['TF_NUM_INTRAOP_THREADS']), jit_compile=False,
        op_determinism_enabled=True, tf32_enabled=False, original_fixture_observations=96,
        evaluated_observations=horizon, batch=185, affinity=sorted(os.sched_getaffinity(0)),
        rss_before_bytes=rss(), replay_summaries=[], call_seconds=[],
        input_tensor_policy='same_tensor_reused_and_verified_unchanged',
        nonclaims=['No full-horizon score oracle, admission, HMC or performance qualification.'])
    first = None
    positions = tf.constant(bank)
    for repeat in range(3):
        tick = time.monotonic()
        returned = kernel(positions)
        if route == 'public_tuple':
            returned = dict(zip(('value', 'score', 'valid', 'branch_status'), returned, strict=True))
        current = {key: value.numpy() for key, value in returned.items()}
        report['call_seconds'].append(time.monotonic() - tick)
        report['target_evaluated'] = True
        np.testing.assert_array_equal(positions.numpy(), bank)
        np.savez_compressed(OUT / f'prefix-replay-{repeat}.npz', bank=bank,
            prepared_observations=prepared[:horizon], **current)
        assert all(np.isfinite(value).all() for value in current.values())
        assert current['valid'].all() and not current['branch_status'].any()
        if first is None:
            first = current
        else:
            report['replay_summaries'].append({key: {
                'changed_elements': int(np.count_nonzero(value != first[key])),
                'max_absolute_error': float(np.max(np.abs(value.astype(float) - first[key].astype(float))))}
                for key, value in current.items()})
        (OUT / 'dz5-prefix-replay.json').write_text(json.dumps(report, indent=2) + '\n')
    assert kernel.experimental_get_tracing_count() == 1
    definition = kernel.get_concrete_function().graph.as_graph_def()
    nodes = [*definition.node, *(node for function in definition.library.function for node in function.node_def)]
    forbidden = sorted({node.op for node in nodes if any(term in node.op.lower() for term in
        ('pyfunc', 'sylvester', 'principalsqrt', 'xlahostcompute'))})
    assert not forbidden, forbidden
    report.update(trace_count=1, host_callbacks=forbidden, graph_node_count=len(nodes),
        graph_sha256=hashlib.sha256(definition.SerializeToString()).hexdigest(),
        replay_exact=all(row['changed_elements'] == 0 for summary in report['replay_summaries']
            for row in summary.values()), rss_after_bytes=rss(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT / 'dz5-prefix-replay.json').write_text(json.dumps(report, indent=2) + '\n')
'''


@pytest.mark.parametrize('route', ['direct', 'public', 'public_tuple'])
@pytest.mark.parametrize('horizon', [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96])
def test_dz5_graph_prefix_replay(request, monkeypatch, route, horizon):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_PREFIX', str(horizon))
    monkeypatch.setenv('FILTER_REPAIR_DZ5_REPLAY_ROUTE', route)
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, PREFIX_CHECK + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='CPU_graph_prefix_replay_localization_only_not_qualification')


@pytest.mark.parametrize('horizon', [8, 16, 32, 48, 64, 96])
def test_dz5_original_oracle_prefix_replay(request, monkeypatch, horizon):
    """Retain the failing harness's placement, output lifetimes and RSS sampler."""
    monkeypatch.setenv('FILTER_REPAIR_DZ5_SCORE_JIT', '0')
    monkeypatch.setenv('FILTER_REPAIR_DZ5_PREFIX', str(horizon))
    prefix = r'''
    original_fixture = fixture
    horizon = int(os.environ['FILTER_REPAIR_DZ5_PREFIX'])
    prepared = fixture.likelihood_observations()
    class PrefixFixture:
        def likelihood_observations(self):
            return prepared[:horizon]
        def __getattr__(self, name):
            return getattr(original_fixture, name)
    fixture = PrefixFixture()
    report.update(original_fixture_observations=96, evaluated_observations=horizon,
        replay_harness='original_oracle_with_exact_prepared_prefix_only')
'''
    replacements = {
        "role='fresh_DZ5_independent_five_point_score_oracle'":
            "role='CPU_original_oracle_prefix_replay_explanatory_only'",
        "        assert all(check['passed'] for check in checks), checks":
            "        report['prefix_oracle_passed'] = all(check['passed'] for check in checks)",
        "        for left, right in zip(arrays, replay_arrays, strict=True):\n"
        "            np.testing.assert_array_equal(left, right)\n"
        "        report['replay_exact'] = True":
            "        report['replay_exact'] = all(np.array_equal(left, right)\n"
            "            for left, right in zip(arrays, replay_arrays, strict=True))\n"
            "        np.testing.assert_array_equal(positions.numpy(), bank)",
    }
    check = ORACLE_CHECK
    for before, after in replacements.items():
        assert check.count(before) == 1
        check = check.replace(before, after)
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, prefix + check + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='CPU_original_oracle_prefix_replay_diagnostic_not_qualification')


LATE_STEP_CHECK = PREFIX_CHECK[:PREFIX_CHECK.index('    kernel =')].replace(
    'tf.constant(prefix_fixture.likelihood_observations(), tf.float64)',
    'tf.constant(prefix_fixture.likelihood_observations()[:-1], tf.float64)') + r'''
    from dataclasses import replace
    assert route == 'direct'
    target_device = '/CPU:0'
    frozen_path = os.environ.get('FILTER_REPAIR_DZ5_FROZEN_STEP_INPUT')
    state_fields = ('mean', 'factor', 'd_mean', 'd_factor')
    with tf.device(target_device):
        positions = tf.constant(bank)
        if frozen_path:
            with np.load(frozen_path, allow_pickle=False) as archive:
                frozen = {name: archive[name] for name in archive.files}
            np.testing.assert_array_equal(frozen['bank'], bank)
            np.testing.assert_array_equal(frozen['observation'], prepared[horizon - 1:horizon])
            assert int(frozen['prefix_steps']) == horizon - 1
            report['frozen_input_sha256'] = sha(frozen_path)
            report['prefix_preparation_seconds'] = 0.
        else:
            prepare = tf.function(evaluate, input_signature=[tf.TensorSpec([185, 23], tf.float64)],
                jit_compile=False, autograph=False)
            tick = time.monotonic()
            prefix_result = {key: value.numpy() for key, value in prepare(positions).items()}
            report['prefix_preparation_seconds'] = time.monotonic() - tick
            assert all(np.isfinite(value).all() for value in prefix_result.values())
            assert prefix_result['valid'].all() and not prefix_result['branch_status'].any()
            frozen = {name: prefix_result[name] for name in state_fields}
            frozen.update(bank=bank, observation=prepared[horizon - 1:horizon],
                prefix_steps=np.asarray(horizon - 1))
            np.savez_compressed(OUT / 'frozen-step-inputs.npz', **frozen)
            report['frozen_input_sha256'] = sha(OUT / 'frozen-step-inputs.npz')
        inputs = (positions, *(tf.constant(frozen[name]) for name in state_fields))
    def one_step(parameters, mean, factor, d_mean, d_factor):
        model, derivatives = build_credit_factor_model_and_derivatives(
            parameters, fixture, prepare_fixed_cir_points=True)
        model = replace(model, initial_mean=mean, initial_factor=factor)
        derivatives = replace(derivatives, d_initial_mean=d_mean, d_initial_factor=d_factor)
        observation = tf.constant(prepared[horizon - 1:horizon], tf.float64)
        result = tf_rectangular_srukf_value_and_score(
            tf.broadcast_to(observation, [185, *observation.shape.as_list()]),
            model, derivatives, branch=credit_fixed_branch(fixture), jit_compile=False)
        return {'value': result.log_likelihood, 'score': result.score,
            'mean': result.filtered_mean, 'factor': result.filtered_factor,
            'd_mean': result.d_filtered_mean, 'd_factor': result.d_filtered_factor,
            'valid': result.diagnostics['score_valid'],
            'branch_status': result.diagnostics['branch_status_code']}
    kernel = tf.function(one_step, input_signature=[tf.TensorSpec(x.shape, x.dtype) for x in inputs],
        jit_compile=False, autograph=False)
    report.update(role='CPU_frozen_late_step_replay_explanatory_only',
        evaluated_observations=1, prefix_steps=horizon - 1, jit_compile=False,
        original_fixture_observations=96, threads=int(os.environ['TF_NUM_INTRAOP_THREADS']),
        op_determinism_enabled=True, tf32_enabled=False, batch=185,
        input_tensor_policy='same_tensors_reused_and_verified_unchanged',
        call_seconds=[], replay_summaries=[], target_evaluated=False,
        nonclaims=['No oracle, full-trajectory, HMC, performance or default qualification.'])
    first = None
    retained_first_tensors = None
    for repeat in range(20):
        tick = time.monotonic()
        with tf.device(target_device):
            current_tensors = kernel(*inputs)
        current = {key: tensor.numpy() for key, tensor in current_tensors.items()}
        report['call_seconds'].append(time.monotonic() - tick)
        report['target_evaluated'] = True
        np.savez_compressed(OUT / f'late-step-replay-{repeat:02d}.npz', **current)
        assert all(np.isfinite(value).all() for value in current.values())
        assert current['valid'].all() and not current['branch_status'].any()
        for tensor, name in zip(inputs, ('bank', *state_fields), strict=True):
            np.testing.assert_array_equal(tensor.numpy(), frozen[name])
        if first is None:
            first = current
            retained_first_tensors = current_tensors
        else:
            report['replay_summaries'].append({key: {
                'changed_elements': int(np.count_nonzero(value != first[key])),
                'max_absolute_error': float(np.max(np.abs(value.astype(float) - first[key].astype(float))))}
                for key, value in current.items()})
        (OUT / 'dz5-late-step-replay.json').write_text(json.dumps(report, indent=2) + '\n')
    assert retained_first_tensors is not None and kernel.experimental_get_tracing_count() == 1
    report.update(trace_count=1, replay_exact=all(row['changed_elements'] == 0
        for summary in report['replay_summaries'] for row in summary.values()),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT / 'dz5-late-step-replay.json').write_text(json.dumps(report, indent=2) + '\n')
'''


@pytest.mark.parametrize('horizon', [48, 64])
def test_dz5_frozen_late_step_threads(request, monkeypatch, horizon):
    directory = Path(request.config.getoption('xmlpath')).parent
    monkeypatch.setenv('FILTER_REPAIR_DZ5_PREFIX', str(horizon))
    monkeypatch.setenv('FILTER_REPAIR_DZ5_REPLAY_ROUTE', 'direct')
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, LATE_STEP_CHECK + loaded_audit)
    reports = []
    for threads in (2, 1):
        target = directory / f'threads-{threads}'
        target.mkdir()
        monkeypatch.setenv('TF_NUM_INTRAOP_THREADS', str(threads))
        if threads == 1:
            shutil.copyfile(directory / 'threads-2' / 'frozen-step-inputs.npz',
                target / 'frozen-step-inputs.npz')
            monkeypatch.setenv('FILTER_REPAIR_DZ5_FROZEN_STEP_INPUT',
                '/tmp/dz5-output/frozen-step-inputs.npz')
        else:
            monkeypatch.delenv('FILTER_REPAIR_DZ5_FROZEN_STEP_INPUT', raising=False)
        child_request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _name, p=target: str(p / 'junit.xml')))
        reports.append(snapshot_test.run_isolated_snapshot(child_request, snapshot=SNAPSHOT,
            child=child, child_timeout_seconds=600,
            scope='CPU_identical_frozen_late_step_thread_replay_localization_only'))
    assert reports[0]['frozen_input_sha256'] == reports[1]['frozen_input_sha256']


TRAJECTORY_INSTRUMENTATION = r'''
    original_while_loop = tf.while_loop
    capture = {}
    state_names = ('mean', 'factor', 'd_mean', 'd_factor', 'value', 'score',
        'valid', 'minimum_pivot', 'maximum_chart_residual', 'maximum_support_residual')
    def instrumented_while_loop(cond, body, loop_vars, **kwargs):
        if body.__qualname__ != 'tf_rectangular_srukf_value_and_score.<locals>.run.<locals>.body':
            return original_while_loop(cond, body, loop_vars, **kwargs)
        assert len(loop_vars) == 11 and not capture
        buffers = tuple(tf.TensorArray(x.dtype, size=horizon, element_shape=x.shape)
            for x in loop_vars[1:])
        def recorded_body(*args):
            following = body(*args[:-1])
            updated = tuple(buf.write(args[0], x) for buf, x in zip(args[-1], following[1:], strict=True))
            return (*following, updated)
        recorded = original_while_loop(lambda *args: cond(*args[:-1]), recorded_body,
            (*loop_vars, buffers), **kwargs)
        capture.update({f'history_{name}': buf.stack()
            for name, buf in zip(state_names, recorded[-1], strict=True)})
        return recorded[:-1]
    plain_evaluate = evaluate
    def evaluate(parameters):
        tf.while_loop = instrumented_while_loop
        try:
            result = plain_evaluate(parameters)
            assert len(capture) == 10
            return {**result, **capture}
        finally:
            tf.while_loop = original_while_loop
'''

TRAJECTORY_CHECK = PREFIX_CHECK.replace(
    '    kernel = tf.function(evaluate,', TRAJECTORY_INSTRUMENTATION + '\n    kernel = tf.function(evaluate,')
TRAJECTORY_CHECK = TRAJECTORY_CHECK.replace(
    "'CPU_graph_prefix_replay_explanatory_only'", "'CPU_graph_carried_history_replay_explanatory_only'")
TRAJECTORY_CHECK = TRAJECTORY_CHECK.replace(
    "    for repeat in range(3):", "    retained_first_tensors = None\n    for repeat in range(3):")
TRAJECTORY_CHECK = TRAJECTORY_CHECK.replace(
    '        returned = kernel(positions)',
    "        with tf.device('/CPU:0'):\n            returned = kernel(positions)")
TRAJECTORY_CHECK = TRAJECTORY_CHECK.replace(
    '            first = current', '            first = current\n            retained_first_tensors = returned')
TRAJECTORY_CHECK = TRAJECTORY_CHECK.replace(
    "        (OUT / 'dz5-prefix-replay.json').write_text", r'''
        assert current['history_valid'].all()
        report['first_differing_time'] = {key: {
            'observation': (int(np.flatnonzero(np.any(value != first[key], axis=tuple(range(1, value.ndim))))[0]) + 1)
                if np.any(value != first[key]) else None,
            'changed_by_time': np.count_nonzero(value != first[key], axis=tuple(range(1, value.ndim))).tolist()}
            for key, value in current.items() if key.startswith('history_')}
        (OUT / 'dz5-prefix-replay.json').write_text''')


def test_dz5_graph_carried_history(request, monkeypatch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_PREFIX', '48')
    monkeypatch.setenv('FILTER_REPAIR_DZ5_REPLAY_ROUTE', 'direct')
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, TRAJECTORY_CHECK + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='CPU_carried_history_replay_diagnostic_not_qualification')


STEP_OPERANDS_INSTRUMENTATION = r'''
    original_step = one_step
    original_while_loop = tf.while_loop
    captured_step = {}
    observed_names = (
        'augmented_mean', 'augmented_factor', 'd_augmented_mean', 'd_augmented_factor',
        'points', 'd_points', 'predicted_points', 'state_j', 'process_j', 'direct',
        'd_predicted_points', 'predicted_mean', 'd_predicted_mean',
        'state_stack', 'd_state_stack', 'observation_points', 'd_observation_points',
        'predicted_observation', 'd_predicted_observation', 'y_stack', 'dy_stack',
        'observation_stack', 'd_observation_stack', 'state_joint_stack', 'd_state_joint_stack',
        'innovation', 'd_innovation', 'inc', 'inc_score', 'increment', 'd_increment',
        'new_factor', 'new_d_factor', 'new_mean', 'new_d_mean')
    def instrumented_while_loop(cond, body, loop_vars, **kwargs):
        if body.__qualname__ != 'tf_rectangular_srukf_value_and_score.<locals>.run.<locals>.body':
            return original_while_loop(cond, body, loop_vars, **kwargs)
        assert len(loop_vars) == 11 and not captured_step
        def trace_body(*args):
            locals_at_return = {}
            def profile(frame, event, _argument):
                if frame.f_code is body.__code__ and event == 'return':
                    locals_at_return.update({name: frame.f_locals[name] for name in observed_names})
            previous_profile = sys.getprofile()
            sys.setprofile(profile)
            try:
                following = body(*args)
            finally:
                sys.setprofile(previous_profile)
            assert len(locals_at_return) == len(observed_names)
            return following, locals_at_return
        step_kernel = tf.function(trace_body,
            input_signature=[tf.TensorSpec(x.shape, x.dtype) for x in loop_vars],
            autograph=False, jit_compile=False)
        template = step_kernel.get_concrete_function().structured_outputs[1]
        empty = tf.nest.map_structure(lambda x: tf.zeros(x.shape, x.dtype), template)
        def recorded_body(*args):
            following, intermediate = step_kernel(*args[:-1])
            return (*following, intermediate)
        recorded = original_while_loop(lambda *args: cond(*args[:-1]), recorded_body,
            (*loop_vars, empty), **kwargs)
        captured_step.update({f'operand_{key}': value for key, value in recorded[-1].items()})
        return recorded[:-1]
    def one_step(*args):
        tf.while_loop = instrumented_while_loop
        try:
            result = original_step(*args)
            assert len(captured_step) == len(observed_names)
            return {**result, **captured_step}
        finally:
            tf.while_loop = original_while_loop
'''

STEP_OPERANDS_CHECK = LATE_STEP_CHECK.replace(
    '    kernel = tf.function(one_step,', STEP_OPERANDS_INSTRUMENTATION + '\n    kernel = tf.function(one_step,')
STEP_OPERANDS_CHECK = STEP_OPERANDS_CHECK.replace(
    'CPU_frozen_late_step_replay_explanatory_only', 'CPU_frozen_step_operands_replay_explanatory_only')
STEP_OPERANDS_CHECK = STEP_OPERANDS_CHECK.replace(
    "        np.savez_compressed(OUT / f'late-step-replay-{repeat:02d}.npz', **current)",
    "        if first is None or any(not np.array_equal(value, first[key]) for key, value in current.items()):\n"
    "            np.savez_compressed(OUT / f'late-step-replay-{repeat:02d}.npz', **current)")


def test_dz5_first_difference_step_operands(request, monkeypatch):
    import numpy as np  # Diagnostic witness extraction from completed histories.

    directory = Path(request.config.getoption('xmlpath')).parent
    witness = SNAPSHOT.parent / 'run-03896'
    with np.load(witness / 'prefix-replay-0.npz', allow_pickle=False) as first, \
            np.load(witness / 'prefix-replay-2.npz', allow_pickle=False) as changed:
        fields = ('mean', 'factor', 'd_mean', 'd_factor')
        for name in fields:
            np.testing.assert_array_equal(first[f'history_{name}'][:34], changed[f'history_{name}'][:34])
        assert np.any(first['history_d_mean'][34] != changed['history_d_mean'][34])
        frozen = {name: first[f'history_{name}'][33] for name in fields}
        frozen.update(bank=first['bank'], observation=first['prepared_observations'][34:35],
            prefix_steps=np.asarray(34))
    np.savez_compressed(directory / 'frozen-step-inputs.npz', **frozen)
    monkeypatch.setenv('FILTER_REPAIR_DZ5_PREFIX', '35')
    monkeypatch.setenv('FILTER_REPAIR_DZ5_REPLAY_ROUTE', 'direct')
    monkeypatch.setenv('FILTER_REPAIR_DZ5_FROZEN_STEP_INPUT', '/tmp/dz5-output/frozen-step-inputs.npz')
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, STEP_OPERANDS_CHECK + loaded_audit)
    reports = []
    for threads in (2, 1):
        target = directory / f'threads-{threads}'
        target.mkdir()
        shutil.copyfile(directory / 'frozen-step-inputs.npz', target / 'frozen-step-inputs.npz')
        monkeypatch.setenv('TF_NUM_INTRAOP_THREADS', str(threads))
        child_request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _name, p=target: str(p / 'junit.xml')))
        reports.append(snapshot_test.run_isolated_snapshot(child_request, snapshot=SNAPSHOT,
            child=child, child_timeout_seconds=360,
            scope='CPU_first_difference_step_operands_diagnostic_only'))
    assert reports[0]['frozen_input_sha256'] == reports[1]['frozen_input_sha256']


def test_dz5_graph_mean_tangent_history(request, monkeypatch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_PREFIX', '48')
    monkeypatch.setenv('FILTER_REPAIR_DZ5_REPLAY_ROUTE', 'direct')
    instrumentation = TRAJECTORY_INSTRUMENTATION.replace(
        '    def instrumented_while_loop',
        "    extra_names = ('d_predicted_mean', 'd_increment', 'inc_score', 'd_innovation')\n"
        '    def instrumented_while_loop')
    instrumentation = instrumentation.replace(
        '            for x in loop_vars[1:])',
        '            for x in loop_vars[1:]) + tuple(tf.TensorArray(tf.float64, size=horizon,\n'
        '                infer_shape=False) for _name in extra_names)')
    instrumentation = instrumentation.replace('            following = body(*args[:-1])', r'''
            locals_at_return = {}
            def profile(frame, event, _argument):
                if frame.f_code is body.__code__ and event == 'return':
                    locals_at_return.update({name: frame.f_locals[name] for name in extra_names})
            previous_profile = sys.getprofile()
            sys.setprofile(profile)
            try:
                following = body(*args[:-1])
            finally:
                sys.setprofile(previous_profile)
            assert len(locals_at_return) == len(extra_names)
            observed = (*following[1:], *(locals_at_return[name] for name in extra_names))''')
    instrumentation = instrumentation.replace('zip(args[-1], following[1:], strict=True)',
        'zip(args[-1], observed, strict=True)')
    instrumentation = instrumentation.replace('zip(state_names, recorded[-1], strict=True)',
        'zip((*state_names, *extra_names), recorded[-1], strict=True)')
    instrumentation = instrumentation.replace('assert len(capture) == 10', 'assert len(capture) == 14')
    assert TRAJECTORY_CHECK.count(TRAJECTORY_INSTRUMENTATION) == 1
    check = TRAJECTORY_CHECK.replace(TRAJECTORY_INSTRUMENTATION, instrumentation)
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, check + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='CPU_mean_tangent_intermediate_history_diagnostic_only')


@pytest.mark.parametrize('control', ['arithmetic_off', 'one_thread'])
def test_dz5_original_prefix_execution_control(request, monkeypatch, control):
    if control == 'one_thread':
        monkeypatch.setenv('TF_NUM_INTRAOP_THREADS', '1')
    else:
        before = ORACLE_CHECK
        after = before.replace("    tf.config.experimental.enable_op_determinism()",
            "    tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})\n"
            "    report['optimizer_options'] = tf.config.optimizer.get_experimental_options()\n"
            "    tf.config.experimental.enable_op_determinism()")
        monkeypatch.setitem(globals(), 'ORACLE_CHECK', after)
    test_dz5_original_oracle_prefix_replay(request, monkeypatch, 48)


def test_dz5_optimized_addition_graph(request, monkeypatch):
    additions = r'''
    from tensorflow.python.eager import context as eager_context
    eager_context.enable_graph_collection()
'''
    export = r'''
    metadata = eager_context.export_run_metadata()
    eager_context.disable_graph_collection()
    graph_records = []
    for index, graphs in enumerate(metadata.function_graphs):
        for stage in ('pre_optimization_graph', 'post_optimization_graph'):
            definition = getattr(graphs, stage)
            path = OUT / f'dz5-graph-{index}-{stage}.pb'
            path.write_bytes(definition.SerializeToString())
            nodes = [*definition.node, *(node for function in definition.library.function
                for node in function.node_def)]
            graph_records.append({'index': index, 'stage': stage, 'path': path.name,
                'sha256': sha(path), 'nodes': len(nodes),
                'addn': [{'name': node.name, 'input': list(node.input),
                    'original_node_names': list(node.experimental_debug_info.original_node_names)}
                    for node in nodes if node.op == 'AddN']})
    assert graph_records, 'No executed optimized graph was captured'
    (OUT / 'dz5-optimized-additions.json').write_text(json.dumps(graph_records, indent=2) + '\n')
    report['optimized_addn_counts'] = [len(row['addn']) for row in graph_records]
    # Independent diagnostic of summation order versus output-buffer eligibility.
    from decimal import Decimal
    expected = float(Decimal('1e16') + Decimal('-1e16') + Decimal('1'))
    primitive_records = []
    primitive_input = tf.constant([[5e15] * 257, [-5e15] * 257, [.5] * 257], tf.float64)
    for expose in ('none', 'first_two', 'all'):
        def sum_terms(x):
            terms = (2. * x[0], 2. * x[1], 2. * x[2])
            total = tf.raw_ops.AddN(inputs=list(terms))
            if expose == 'first_two':
                return total, terms[0], terms[1]
            if expose == 'all':
                return total, *terms
            return (total,)
        primitive = tf.function(sum_terms,
            input_signature=[tf.TensorSpec([3, 257], tf.float64)],
            jit_compile=False, autograph=False)
        observed = [primitive(primitive_input)[0].numpy().tolist() for _repeat in range(4)]
        primitive_records.append({'expose': expose, 'expected_exact_sum': expected,
            'observed': observed, 'trace_count': primitive.experimental_get_tracing_count()})
    (OUT / 'addn-buffer-primitive.json').write_text(json.dumps(primitive_records, indent=2) + '\n')
'''
    check = ORACLE_CHECK.replace("    import resource", additions + '\n    import resource') + export
    monkeypatch.setitem(globals(), 'ORACLE_CHECK', check)
    test_dz5_original_oracle_prefix_replay(request, monkeypatch, 8)


def test_dz5_full_graph_oracle_arithmetic_off(request, monkeypatch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_SCORE_JIT', '0')
    check = ORACLE_CHECK.replace("    tf.config.experimental.enable_op_determinism()",
        "    tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})\n"
        "    report['optimizer_options'] = tf.config.optimizer.get_experimental_options()\n"
        "    report['reference_control'] = 'explicit_graph_arithmetic_optimizer_disabled'\n"
        "    tf.config.experimental.enable_op_determinism()")
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, check + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, child_timeout_seconds=840,
        scope='full_CPU_graph_score_oracle_explicit_arithmetic_off_reference_only')
