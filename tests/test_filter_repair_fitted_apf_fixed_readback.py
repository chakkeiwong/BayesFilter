"""Independent readback of complete fitted-APF qualification records."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import RAW, ROOT, _latest


def test_fitted_apf_fixed_readback(request):
    rows = []
    for device in ('cpu', 'gpu'):
        for model in ('gaussian', 'nonlinear_scalar'):
            for dtype in ('float64', 'float32'):
                number, run = _latest(f'fitted_apf_fixed_owner_{model}_{dtype}_{device}')
                assert run['state'] == 'passed' and run['device'] == device.upper()
                assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
                name = f'fitted-apf-owner-{model}-{dtype}'
                report = _load(number, f'{name}.json')
                graph = report['graph']
                assert graph['trace_count'] == 1 and graph['jit_compile'] and not graph['host_callbacks']
                assert hashlib.sha256((RAW/f'run-{number:05d}'/f'{name}.hlo.txt').read_bytes()).hexdigest() == graph['hlo_sha256']
                provenance = _provenance(number)
                assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
                if device == 'gpu':
                    assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
                else:
                    assert provenance['cuda_visible_devices'] == '-1'
                maximum = 0.
                for row in report['rows']:
                    expected, actual = row['reference'], row['candidate']
                    assert actual['attempted_iterations'] == 2 and actual['invalid_iteration'] == -1
                    for key in ('fit', 'final'):
                        pairs = zip(expected[key], actual[key], strict=True)
                        for a, b in pairs:
                            a, b = np.asarray(a), np.asarray(b)
                            tolerance = 2e-10 if dtype == 'float64' else 5e-5
                            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
                            maximum = max(maximum, float(np.max(np.abs(a-b))))
                    for key in ('fit_log_values', 'fit_errors', 'fit_centers', 'fit_covariances', 'fit_log_floors'):
                        a, b = np.asarray(expected[key]), np.asarray(actual[key])
                        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
                        maximum = max(maximum, float(np.max(np.abs(a-b))))
                    if dtype == 'float64':
                        assert len(row['fixed_fit_fd_errors']) == 2
                        assert max(row['fixed_fit_fd_errors']) < 2e-7
                rows.append({'run': number, 'device': device, 'model': model,
                             'dtype': dtype, 'maximum_absolute_error': maximum})
            number, run = _latest(f'fitted_apf_fixed_endpoint_{model}_{device}')
            assert run['state'] == 'passed'
            endpoint = _load(number, f'fitted-apf-endpoint-{model}.json')
            assert endpoint['passed'] and endpoint['factory_identity_verified']
            assert endpoint['candidate']['diagnostics']['fit_execution'] == 'enclosing_tensorflow_loop'
        number, run = _latest(f'fitted_apf_fixed_failure_{device}')
        assert run['state'] == 'passed'
        failed = _load(number, 'fitted-apf-failure.json')
        assert failed['invalid_iteration'] == 1 and failed['attempted_iterations'] == 2
        assert np.isnan(failed['fit_log_values'][2])
    sources = ['bayesfilter/score_study/fitted_twist_adapter.py',
               'bayesfilter/score_study/fitted_twist_execution_tf.py',
               'bayesfilter/score_study/fitted_twist_tf.py', 'bayesfilter/ops/stateless_random_tf.py']
    result = {'schema': 'filter_repair_fitted_apf_fixed_readback.v1', 'rows': rows,
              'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
              'scope': 'Fixed-fit execution/finite-score qualification only; costs and adaptive iAPF remain open.'}
    (Path(request.config.getoption('xmlpath')).parent/'fitted-apf-fixed-readback.json').write_text(
        json.dumps(result, indent=2)+'\n')
