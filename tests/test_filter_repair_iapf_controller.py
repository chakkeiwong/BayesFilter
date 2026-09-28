"""Frozen independent decision reference, including adversarial CV thresholds."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.score_study.iapf_controller_tf import make_iapf_iteration_decision
from tests.test_filter_repair_fitted_apf_fixed import _graph, _host, _save

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/filter_repair_iapf_controller_20260929'
FIXTURE = json.loads((FIXTURES/'fixture.json').read_text())


def _reference():
    path = FIXTURES/'iapf_adapter_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE['source_sha256']
    spec = importlib.util.spec_from_file_location('bayesfilter.score_study._iapf_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.iteration_decision


def test_iapf_native_decision(request):
    reference = _reference()
    owner = make_iapf_iteration_decision(8, 2, 64)
    rows = []
    cases = [(case, .01) for case in FIXTURE['cases']]
    boundary = FIXTURE['cases'][-1]
    cv = reference(boundary['log_values'], boundary['counts'], k=2, tau=.01, max_particles=64)['cv']
    cases += [(boundary, tau) for tau in (math.nextafter(cv, 0.), cv, math.nextafter(cv, math.inf))]
    actions = {-1: 'invalid', 0: 'fit', 1: 'final', 2: 'capacity_veto'}
    for case, tau in cases:
        logs, counts = case['log_values'], case['counts']
        args = (tf.constant(logs + [float('nan')]*(8-len(logs)), tf.float64),
                tf.constant(counts + [-1]*(8-len(counts)), tf.int64),
                tf.constant(len(logs)), tf.constant(tau, tf.float64))
        expected = reference(logs, counts, k=2, tau=tau, max_particles=64)
        actual = _host(owner(*args))
        replay = _host(owner(*args))
        row = {'case': case, 'tau': tau, 'reference': expected, 'candidate': actual}
        rows.append(row)
        _save(request, 'iapf-controller', {'rows': rows, 'complete': False})
        assert actual['valid']
        for key in ('action', 'next_particles', 'complete_window'):
            value = actions[actual[key]] if key == 'action' else actual[key]
            assert value == expected[key], (case['name'], tau, key, expected, actual)
            assert actual[key] == replay[key]
        if expected['cv'] is None:
            assert math.isnan(actual['cv']) and math.isnan(replay['cv'])
        else:
            np.testing.assert_allclose(actual['cv'], expected['cv'], rtol=0, atol=1e-14)
            assert actual['cv'] == replay['cv']
    invalid = []
    for logs, counts, length, tau in [([float('nan')], [8], 1, .01),
        ([0.], [1], 1, .01), ([0.], [65], 1, .01), ([0.], [8], 0, .01),
        ([0.], [8], 9, .01), ([0.], [8], 1, 0.), ([0.], [8], 1, float('nan'))]:
        actual = _host(owner(tf.constant(logs+[0.]*(8-len(logs)), tf.float64),
            tf.constant(counts+[8]*(8-len(counts)), tf.int64), tf.constant(length), tf.constant(tau, tf.float64)))
        invalid.append(actual)
        assert not actual['valid'] and actual['action'] == -1
    assert owner.experimental_get_tracing_count() == 1
    graph = _graph(request, 'iapf-controller', owner, args)
    _save(request, 'iapf-controller', {'rows': rows, 'invalid': invalid, 'graph': graph,
        'complete': True, 'runtime_migrated': False,
        'scope': 'Native decision primitive only; adaptive endpoint and accounting remain open.'})
