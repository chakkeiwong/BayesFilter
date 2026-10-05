"""Diagnostic complete-original C2 fixed/defensive preparation comparisons."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.test_filter_repair_c2_preparation import (
    BASELINE, MaterializedCheckpoint, compare, fixture, record,
)

FAMILIES = (('mixture', 2), ('mixture', 4), ('defensive', 1),
            ('defensive', 2), ('defensive', 4))
CASES = tuple((family, k, h, n, seed, changed)
              for family, k in FAMILIES
              for h, n, seed, changed in ((3, 16, 9104, False),
                  (4, 20, -9104 if k == 2 else 4294976400, True)))


def invoke(adapter, model, theta, observed, family, components,
           count=16, seed=9104, **overrides):
    arguments = dict(model=model, theta_reference=theta, observations=observed,
                     particle_count=count, seed=seed, offset=.35)
    if family == 'mixture':
        arguments['component_count'] = components
        function = adapter.compile_c2_per_ancestor_ukf_apf_mixture
    else:
        arguments.update(local_component_count=components, nu=8.,
                         epsilon_min=.05, epsilon_max=.20)
        function = adapter.compile_c2_per_ancestor_ukf_apf_defensive_mixture
    arguments.update(overrides)
    return function(**arguments)


def error_records(adapter, models, family, components):
    model, theta, observed = fixture(models, 3)
    cases = [('stationarity', {'theta_reference': tf.constant([2., 0.], tf.float64)}),
             ('particle_count', {'particle_count': 1}),
             ('component_count', {('component_count' if family == 'mixture'
                                  else 'local_component_count'): 3})]
    if components != 1:
        cases.append(('offset', {'offset': 1.}))
    if family == 'defensive':
        cases.extend((('nu', {'nu': 2.}), ('epsilon', {'epsilon_max': 1.})))
    for name, index, value in (('initial_nonfinite', 0, float('nan')),
                               ('later_nonfinite', 1, float('inf')),
                               ('later_zero', 1, 0.)):
        cases.append((name, {'observations': tf.tensor_scatter_nd_update(
            observed, [[index, 0]], tf.constant([value], tf.float64))}))
    errors = {}
    for name, overrides in cases:
        try:
            invoke(adapter, model, theta, observed, family, components, **overrides)
        except (ValueError, TypeError, tf.errors.OpError) as error:
            errors[name] = {'type': type(error).__name__, 'message': str(error)}
        else:
            pytest.fail(f'{family}/{components}/{name} did not reject invalid input')
    return errors


def test_original_mixture_records(request):
    old = MaterializedCheckpoint(BASELINE, 'c2_mixture_original')
    adapter = old.load('bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter')
    models = old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    apf = old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    results = []
    for case in CASES:
        family, k, horizon, count, seed, changed = case
        model, theta, observed = fixture(models, horizon, changed)
        value = invoke(adapter, model, theta, observed, family, k, count, seed)
        result = record(value, model, theta, apf)
        assert result['value_score']['finite']
        results.append({'case': case, 'record': result})
    errors = {f'{family}/{k}': error_records(adapter, models, family, k)
              for family, k in FAMILIES}
    output = Path(request.config.getoption('xmlpath')).parent
    (output/'c2-mixture-original.json').write_text(json.dumps({
        'cases': results, 'errors': errors, 'sources': old.hashes()}, indent=2)+'\n')


def saved_original():
    from scripts import run_filter_repair_campaign as runner
    device = 'cpu' if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' else 'gpu'
    group = f'c2_preparation_mixture_original_{device}'
    rows = [row for row in runner.records()
            if row['key'][1] == group and row['state'] == 'passed']
    assert rows, f'Missing completed original freeze: {group}'
    path = Path(rows[-1]['result']).parent/'c2-mixture-original.json'
    return json.loads(path.read_text())


@pytest.mark.parametrize('case', CASES)
def test_full_mixture_matches_original(case, request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as adapter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    family, k, horizon, count, seed, changed = case
    frozen = next(row['record'] for row in saved_original()['cases']
                  if row['case'] == list(case))
    model, theta, observed = fixture(models, horizon, changed)
    value = invoke(adapter, model, theta, observed, family, k, count, seed)
    current = record(value, model, theta, apf)
    report = {'max_abs': 0.}
    for field in ('branch', 'diagnostics', 'value_score'):
        compare(frozen[field], current[field], (field,), report)
    compare({k: v for k, v in frozen['manifest'].items() if k != 'branch_id'},
            {k: v for k, v in current['manifest'].items() if k != 'branch_id'},
            ('manifest',), report)
    output = Path(request.config.getoption('xmlpath')).parent
    (output/f'c2-{family}-k{k}-t{horizon}-comparison.json').write_text(json.dumps({
        'maximum_absolute_error': report['max_abs'], 'original': frozen,
        'current': current}, indent=2)+'\n')


@pytest.mark.parametrize('family,k', FAMILIES)
def test_mixture_original_error_order(family, k):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as adapter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    assert error_records(adapter, models, family, k) == saved_original()['errors'][f'{family}/{k}']


@pytest.mark.parametrize('family,k', FAMILIES)
def test_mixture_live_inputs_and_enclosing_xla(family, k, request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as adapter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    model, theta, observed = fixture(models, 3)
    first = invoke(adapter, model, theta, observed, family, k)
    owner = next(iter(model._c2_preparation_owners.values()))
    changed = tf.tensor_scatter_nd_add(observed, [[1, 0]], tf.constant([.7], tf.float64))
    for overrides in ({'seed': 9105}, {'observations': changed},
                      {'theta_reference': theta+tf.constant([.001, .01], tf.float64)}):
        result = invoke(adapter, model, theta, observed, family, k, **overrides)
        assert not np.array_equal(first.branch.states.numpy(), result.branch.states.numpy())
        assert next(iter(model._c2_preparation_owners.values())) is owner
    assert len(model._c2_preparation_owners) == owner.experimental_get_tracing_count() == 1
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    graph = concrete.graph.as_graph_def()
    nodes = [*graph.node, *(node for f in graph.library.function for node in f.node_def)]
    assert any(node.op in ('While', 'StatelessWhile') for node in nodes)
    assert not any(node.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless') for node in nodes)
    hlo = owner.experimental_get_compiler_ir(observed, theta, tf.constant(9104, tf.int64))(stage='hlo')
    assert 'while' in hlo.lower()
    output = Path(request.config.getoption('xmlpath')).parent
    (output/f'c2-{family}-k{k}-owner.json').write_text(json.dumps({
        'trace_count': owner.experimental_get_tracing_count(),
        'cache_count': len(model._c2_preparation_owners), 'enclosing_xla': True,
        'graph_nodes': len(nodes), 'hlo_bytes': len(hlo),
        'placement': first.branch.states.device}, indent=2)+'\n')


@pytest.mark.parametrize('family', ('mixture', 'defensive'))
def test_mixture_graph_size_is_bounded(family, request):
    from collections import Counter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim.c2_mixture_ukf_apf_tf import BatchedUKFConfig
    from bayesfilter.highdim.c2_ukf_preparation_tf import make_ukf_preparation

    model, _, _ = fixture(models, 3)
    sizes = []
    for horizon in (3, 7, 11, 3):
        owner = make_ukf_preparation(model, horizon, 16, BatchedUKFConfig(),
                                    family=family, component_count=4)
        graph = owner.get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
        sizes.append({'horizon': horizon, 'op_counts': dict(Counter(n.op for n in nodes)),
                      'all_nodes': len(nodes), 'functions': len(graph.library.function)})
    output = Path(request.config.getoption('xmlpath')).parent
    (output/f'c2-{family}-graph-size.json').write_text(json.dumps(sizes, indent=2)+'\n')
    arithmetic = [{name: count for name, count in row['op_counts'].items()
                   if name not in ('Const', 'Fill')} for row in sizes]
    assert all(operations == arithmetic[0] for operations in arithmetic)
    assert sizes[1]['op_counts'] == sizes[2]['op_counts']
    assert sizes[0]['op_counts'] == sizes[3]['op_counts']
    assert all(row['functions'] == sizes[0]['functions'] for row in sizes)


@pytest.mark.parametrize('components', (2, 4))
def test_fixed_mixture_moments_and_analytical_score(components):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as adapter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models

    model, theta, observed = fixture(models, 3)
    compilation = invoke(adapter, model, theta, observed, 'mixture', components)
    for row in compilation.proposal_diagnostics:
        for field in ('moment_recomposition_max_abs', 'label_permutation_max_abs',
                      'proposal_density_recomposition_max_abs'):
            assert float(row[field]) <= 2.e-12
    program = compilation.bind_program(model)
    result = program.evaluate(theta)
    difference = []
    for index in range(2):
        step = 1.e-5*tf.one_hot(index, 2, dtype=tf.float64)
        difference.append((program.evaluate(theta+step)['log_likelihood']
                           - program.evaluate(theta-step)['log_likelihood'])/2.e-5)
    tf.debugging.assert_near(result['score'], tf.stack(difference), atol=2.e-7, rtol=2.e-7)
