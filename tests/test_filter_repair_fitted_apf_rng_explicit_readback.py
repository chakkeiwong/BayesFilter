"""Recompute the exact seed/word and floating-transform localization results."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import RAW, ROOT, _latest


def test_fitted_apf_explicit_rng_readback(request):
    devices = []
    for device in ('CPU', 'GPU'):
        number, run = _latest(f'fitted_apf_rng_explicit_{device.lower()}')
        assert run['state'] == 'passed' and run['device'] == device
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        report = _load(number, 'fitted-apf-rng-explicit.json')
        assert report['jit_compile'] and report['trace_count'] == 1
        directory = RAW/f'run-{number:05d}'
        assert hashlib.sha256((directory/'fitted-apf-rng-explicit.hlo.txt').read_bytes()).hexdigest() == report['hlo_sha256']
        assert hashlib.sha256((directory/'tensorflow-random_ops_util.py').read_bytes()).hexdigest() == report['tensorflow_seed_source_sha256']
        source = ROOT/'bayesfilter/ops/ledh_random_compat_tf.py'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report['shared_normal_source_sha256']
        provenance = _provenance(number)
        assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
        if device == 'GPU':
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        rows = []
        for row in report['rows']:
            states = row['states']
            for name in ('key', 'counter'):
                state = states[name]
                assert (state['auto'] == state['explicit'] == state['compiled']) is state['all_exact']
            assert (row['words_eager'] == row['words_compiled']) is states['words_exact']
            fields = {}
            for name, field in row['fields'].items():
                original, eager, compiled = (np.asarray(field[k]) for k in
                    ('reference', 'eager_explicit', 'compiled_explicit'))
                assert bool(np.array_equal(original, eager)) is field['eager_explicit_exact']
                assert bool(np.array_equal(original, compiled)) is field['compiled_explicit_exact']
                assert float(np.max(np.abs(original-compiled))) == field['compiled_explicit_error']
                fields[name] = {key: field[key] for key in
                    ('eager_explicit_exact', 'compiled_explicit_error', 'compiled_explicit_exact')}
                if 'compiled_shared' in field:
                    shared = np.asarray(field['compiled_shared'])
                    assert float(np.max(np.abs(original-shared))) == field['compiled_shared_error']
                    assert bool(np.array_equal(original, shared)) is field['compiled_shared_exact']
                    fields[name]['compiled_shared_error'] = field['compiled_shared_error']
                    fields[name]['compiled_shared_exact'] = field['compiled_shared_exact']
            rows.append({'seed': row['seed'], 'states': states, 'fields': fields})
        devices.append({'device': device, 'run': number, 'rows': rows})
    result = {'schema': 'filter_repair_fitted_apf_rng_explicit_summary.v1', 'devices': devices,
              'runtime_repaired': False, 'scope': 'Input primitive diagnosis only; no stream migration or filter admission.'}
    (Path(request.config.getoption('xmlpath')).parent/'fitted-apf-rng-explicit-readback.json').write_text(
        json.dumps(result, indent=2)+'\n')
