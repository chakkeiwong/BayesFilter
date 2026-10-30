"""Independent Decimal checks of the iAPF numerical-resolution guard."""

import decimal
import math

import tensorflow as tf

from bayesfilter.score_study.iapf_controller_tf import (
    make_checked_iapf_iteration_decision,
)
from tests.test_filter_repair_fitted_apf_fixed import _graph, _host, _save
from tests.test_filter_repair_iapf_controller import FIXTURE, _reference


def _decimal_cv(logs, k):
    with decimal.localcontext() as context:
        context.prec = 80
        values = [decimal.Decimal.from_float(x) for x in logs[-k-1:]]
        maximum = max(values)
        values = [(x-maximum).exp() for x in values]
        mean = sum(values)/len(values)
        return (sum((x-mean)**2 for x in values)/k).sqrt()/mean


def _cases():
    reference = _reference()
    cases = [(case['name'], case['log_values'], case['counts'], 2, .01, False)
             for case in FIXTURE['cases']]
    boundary = FIXTURE['cases'][-1]
    cv = reference(boundary['log_values'], boundary['counts'], k=2, tau=.01, max_particles=64)['cv']
    cases += [(f'boundary_{i}', boundary['log_values'], boundary['counts'], 2, tau, True)
              for i, tau in enumerate((math.nextafter(cv, 0.), cv, math.nextafter(cv, math.inf)))]
    for k in (1, 2, 3):
        for j in range(8):
            logs = [-float(((j+3)*(i+2)) % 19)/(j+1) - (1000. if j % 2 else 0.)
                    for i in range(k+2)]
            cases.append((f'generated_{k}_{j}', logs, [8]*(k+2), k, .01, False))
    return cases


def test_iapf_resolution_interval(request):
    rows = []
    owners = {}
    actions = {0: 'fit', 1: 'final', 2: 'capacity_veto'}
    for label, logs, counts, k, tau, ambiguous in _cases():
        owner = make_checked_iapf_iteration_decision(8, k, 64)
        args = (tf.constant(logs+[float('nan')]*(8-len(logs)), tf.float64),
                tf.constant(counts+[-1]*(8-len(counts)), tf.int64),
                tf.constant(len(logs)), tf.constant(tau, tf.float64))
        expected = _reference()(logs, counts, k=k, tau=tau, max_particles=64)
        actual = _host(owner(*args))
        replay = _host(owner(*args))
        high = _decimal_cv(logs, k) if expected['complete_window'] else None
        row = {'label': label, 'logs': logs, 'counts': counts, 'k': k, 'tau': tau,
               'expect_unresolved': ambiguous, 'reference': expected,
               'candidate': actual, 'decimal_cv': str(high) if high is not None else None}
        rows.append(row)
        _save(request, 'iapf-resolution', {'rows': rows, 'complete': False})
        assert actual['valid'] and actual['complete_window'] == expected['complete_window']
        assert actual['next_particles'] == expected['next_particles']
        if high is not None:
            lo, hi = actual['cv_lower'], actual['cv_upper']
            assert actual['cv_interval_valid'] and math.isfinite(lo) and math.isfinite(hi)
            assert decimal.Decimal.from_float(lo) <= high <= decimal.Decimal.from_float(hi)
            assert lo <= expected['cv'] <= hi
            assert actual == replay
        if ambiguous:
            assert not actual['decision_resolved'] and actual['action'] == -2
        else:
            assert actual['decision_resolved'] and actions[actual['action']] == expected['action']
        owners[k] = owner, args
    invalid = []
    owner = make_checked_iapf_iteration_decision(8, 2, 64)
    for logs, counts, length, tau in [([float('nan')], [8], 1, .01),
        ([0.], [1], 1, .01), ([0.], [65], 1, .01), ([0.], [8], 0, .01),
        ([0.], [8], 9, .01), ([0.], [8], 1, 0.), ([0.], [8], 1, float('nan'))]:
        actual = _host(owner(tf.constant(logs+[0.]*(8-len(logs)), tf.float64),
            tf.constant(counts+[8]*(8-len(counts)), tf.int64), tf.constant(length), tf.constant(tau, tf.float64)))
        invalid.append(actual)
        assert not actual['valid'] and not actual['decision_resolved'] and actual['action'] == -1
    graphs = {str(k): _graph(request, f'iapf-resolution-k{k}', owner, args)
              for k, (owner, args) in owners.items()}
    assert all(graph['trace_count'] == 1 for graph in graphs.values())
    _save(request, 'iapf-resolution', {'rows': rows, 'graphs': graphs, 'invalid': invalid, 'complete': True,
        'runtime_migrated': False, 'scope': 'Resolution primitive; actual adaptive controller remains untouched.'})
