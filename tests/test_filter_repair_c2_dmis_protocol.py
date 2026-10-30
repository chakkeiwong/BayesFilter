"""Independent-original diagnostics for the native DMIS public protocol."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import tensorflow as tf

from tests.test_filter_repair_c2_preparation import BASELINE, MaterializedCheckpoint, compare
from tests.test_filter_repair_c2_branch_preparation import (
    inputs, invoke, record, varied_retained_proposal, _without_payload_id, _frozen_branch_original,
)

D = tf.float64


def test_dmis_heterogeneous_live_inputs(request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    old = MaterializedCheckpoint(BASELINE, 'c2_dmis_protocol_original')
    original = old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    old_hermite = old.load('bayesfilter.highdim.c2_gaussian_hermite_proposal_tf')
    old_apf = old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    records = []
    for module, proposal_module, score in ((original, old_hermite, old_apf), (models, hermite, apf)):
        model, theta, observed = inputs(module, 4)
        retained = tuple(varied_retained_proposal(proposal_module, t, t-1) for t in (1, 2, 3))
        default = module.transformed_student_proposals(model=model, observations=observed, theta_reference=theta, nu=8.)
        defensive = (default[0], replace(default[1], nu=6.), replace(default[2], nu=11.))
        args = dict(model=model, observations=observed, theta_reference=theta, particle_count=16,
                    seed=813, alpha=.23, nu=8., transition_proposals=retained, defensive_proposals=defensive)
        compilation = module.compile_c2_dmis_proposal_branch(**args)
        records.append(record(compilation, model, theta, module, score))
    report = {'max_abs': 0.}
    for field in ('branch', 'diagnostics', 'value_score', 'manifest'):
        compare(_without_payload_id(records[0][field]), _without_payload_id(records[1][field]), (field,), report)
    owner = next(iter(model._c2_branch_preparation_owners.values()))
    changed_defensive = (replace(defensive[0], transformed_observation=defensive[0].transformed_observation+.1), *defensive[1:])
    for change in ({'seed': 814}, {'theta_reference': theta+tf.constant([.01, .02], D)},
                   {'alpha': .37}, {'defensive_proposals': changed_defensive},
                   {'transition_proposals': (replace(retained[0], coordinate_offset=retained[0].coordinate_offset+.05), *retained[1:])}):
        actual = models.compile_c2_dmis_proposal_branch(**{**args, **change})
        assert next(iter(model._c2_branch_preparation_owners.values())) is owner
        assert not np.array_equal(actual.branch.states.numpy(), compilation.branch.states.numpy())
    reported_only = models.compile_c2_dmis_proposal_branch(**{**args, 'nu': 9.})
    np.testing.assert_array_equal(reported_only.branch.states.numpy(), compilation.branch.states.numpy())
    assert float(reported_only.proposal_diagnostics[0]['nu']) == 9.
    assert owner.experimental_get_tracing_count() == len(model._c2_branch_preparation_owners) == 1
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    graph = concrete.graph.as_graph_def()
    nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
    assert any(n.op in ('While', 'StatelessWhile') for n in nodes)
    assert not any(n.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless') for n in nodes)
    output = Path(request.config.getoption('xmlpath')).parent
    (output/'c2-dmis-protocol.json').write_text(json.dumps({'max_abs': report['max_abs'],
        'original': records[0], 'current': records[1], 'live_inputs': True,
        'retained_configurations': 3, 'student_configurations': 3, 'one_trace': True}, indent=2)+'\n')


def test_dmis_errors_and_initial_only(request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    old = MaterializedCheckpoint(BASELINE, 'c2_dmis_errors_original')
    original = old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    old_hermite = old.load('bayesfilter.highdim.c2_gaussian_hermite_proposal_tf')
    old_apf = old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    records = []
    for module, proposal_module, score in ((original, old_hermite, old_apf), (models, hermite, apf)):
        model, theta, compilation = invoke(module, proposal_module, 'dmis', 1, 16, 813)
        records.append(record(compilation, model, theta, module, score))
    for field in ('branch', 'diagnostics', 'value_score', 'manifest'):
        compare(_without_payload_id(records[0][field]), _without_payload_id(records[1][field]), (field,))
    _, _, observed = inputs(models, 3)
    errors = []
    for frozen in _frozen_branch_original()['errors']:
        if frozen['family'] != 'dmis':
            continue
        change = {
            'stationarity': {'theta_reference': tf.constant([2., 0.], D)},
            'small_count': {'particle_count': 1},
            'initial_nonfinite': {'observations': tf.tensor_scatter_nd_update(observed, [[0, 0]], tf.constant([float('nan')], D))},
            'later_nonfinite': {'observations': tf.tensor_scatter_nd_update(observed, [[1, 0]], tf.constant([float('inf')], D))},
            'later_zero': {'observations': tf.tensor_scatter_nd_update(observed, [[1, 0]], tf.constant([0.], D))},
        }[frozen['case']]
        try:
            invoke(models, hermite, 'dmis', 3, 16, 813, **change)
        except (ValueError, TypeError, tf.errors.OpError) as error:
            assert type(error).__name__ == frozen['type'] and str(error) == frozen['message']
            errors.append(frozen)
        else:
            assert frozen.get('accepted')
    output = Path(request.config.getoption('xmlpath')).parent
    (output/'c2-dmis-errors.json').write_text(json.dumps({'errors': errors, 'initial_only': True}, indent=2)+'\n')


def test_dmis_fixed_configuration_graph_growth(request):
    from collections import Counter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim.c2_branch_preparation_tf import make_branch_preparation
    from bayesfilter.highdim.c2_independent_preparation_tf import pack_proposals
    from bayesfilter.highdim.c2_student_preparation_tf import pack_student_proposals
    from bayesfilter.highdim.c2_dmis_preparation_tf import dmis_step, diagnostic_specs
    model, theta, observed = inputs(models, 3)
    template = models.transformed_student_proposals(model=model, observations=observed, theta_reference=theta, nu=8.)[0]
    graphs = []
    for horizon in (3, 7, 11, 3):
        retained = tuple(varied_retained_proposal(hermite, t, t % 2) for t in range(1, horizon))
        defensive = tuple(replace(template, time_index=t, nu=8.+t % 2) for t in range(1, horizon))
        rc, _, rs = pack_proposals(retained)
        sc, _, ss = pack_student_proposals(defensive)
        assert len(rc) == len(sc) == 2
        owner = make_branch_preparation(model, horizon, 16, dmis_step(rc, sc, 16),
            (rs, ss, tf.TensorSpec([], D), tf.TensorSpec([], D)), diagnostic_specs(), bank_step=True)
        graph = owner.get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
        graphs.append({'horizon': horizon, 'op_counts': dict(Counter(n.op for n in nodes)),
                       'functions': len(graph.library.function), 'configurations': [len(rc), len(sc)]})
    output = Path(request.config.getoption('xmlpath')).parent
    (output/'c2-dmis-graph-growth.json').write_text(json.dumps(graphs, indent=2)+'\n')
    numerical = [{name: count for name, count in row['op_counts'].items() if name not in ('Const', 'Fill')}
                 for row in graphs]
    assert all(counts == numerical[0] for counts in numerical)
    assert graphs[0]['op_counts'] == graphs[3]['op_counts']
    assert graphs[1]['op_counts'] == graphs[2]['op_counts']
    assert all(row['functions'] == graphs[0]['functions'] for row in graphs)
