"""Diagnostic full-original C2 bootstrap/independent/Student/DMIS records."""
import json
import math
from pathlib import Path

import tensorflow as tf

from tests.test_filter_repair_c2_preparation import BASELINE, MaterializedCheckpoint, NAMES, jsonable

D=tf.float64
FAMILIES=('bootstrap','gaussian','stationary','hermite','mixed','student','dmis')
CASES=tuple((family,horizon,count,seed) for family in FAMILIES
            for horizon,count,seed in ((3,16,813),(4,20,-814)))


def inputs(models,horizon):
    model=models.C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=tf.constant([[0.,.04],[-.02,0.]],D),sigma=1.)
    theta=tf.constant([.58,math.log(.4)],D)
    observations=tf.constant([[.2,-.1],[.35,.16],[-.22,.31],[.17,-.28]],D)[:horizon]
    return model,theta,observations


def retained_proposal(hermite,time):
    first=tf.constant([1.,.12],D);second=tf.constant([1.,-.08],D)
    return hermite.GaussianHermiteRetainedProposal(
        prefix_core_values=(tf.reshape(first,[1,2,1]),tf.reshape(second,[1,2,1])),
        suffix_gram=tf.ones([1,1],D),z_h=tf.reduce_sum(first**2)*tf.reduce_sum(second**2),
        tau_abs=tf.constant(.02,D),coordinate_offset=tf.constant([.1*time,-.05*time],D),
        coordinate_matrix=tf.constant([[1.,0.],[.1,.9]],D),defensive_nu=5.,time_index=time,
        source_snapshot_fingerprint=f'{time:064x}')


def gaussian_proposal(models,time):
    return models.FrozenGaussianStateProposal(
        mean=tf.constant([.1*time,-.2],D),chol=tf.constant([[1.1,0.],[.12,.9]],D),
        time_index=time,family='gaussian_hint_marginal')


def invoke(models,hermite,family,horizon,count,seed,**overrides):
    model,theta,observations=inputs(models,horizon)
    arguments=dict(model=model,theta_reference=theta,observations=observations,
                   particle_count=count,seed=seed)
    if family=='bootstrap':
        endpoint=models.compile_c2_bootstrap_proposal_branch
    elif family=='student':
        endpoint=models.compile_c2_transformed_student_proposal_branch
        arguments['nu']=8.
    else:
        if family=='stationary':
            proposals=models.stationary_gaussian_proposals(model,theta,horizon)
        else:
            proposals=tuple(gaussian_proposal(models,t) if family=='gaussian' or (family=='mixed' and t%2)
                            else retained_proposal(hermite,t) for t in range(1,horizon))
        arguments['transition_proposals']=proposals
        if family=='dmis':
            endpoint=models.compile_c2_dmis_proposal_branch
            arguments.update(alpha=.5,nu=8.)
        else:
            endpoint=models.compile_c2_independent_proposal_branch
            arguments['family']=family
    arguments.update(overrides)
    return model,theta,endpoint(**arguments)


def record(compilation,model,theta,models,apf):
    branch=compilation.branch
    assert compilation.compiler_id==models._compilation_fingerprint(compilation.manifest)
    program=apf.prepare_frozen_proposal_apf_program(model,branch)
    return jsonable({'branch':{name:getattr(branch,name) for name in NAMES},
        'diagnostics':compilation.proposal_diagnostics,'value_score':program.evaluate(theta),
        'manifest':compilation.manifest,'identities':{'branch_id':branch.branch_id,
        'compiler_id':compilation.compiler_id,'program_id':program.program_id}})


def test_original_branch_records(request):
    original=MaterializedCheckpoint(BASELINE,'c2_branch_original')
    models=original.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    hermite=original.load('bayesfilter.highdim.c2_gaussian_hermite_proposal_tf')
    apf=original.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    rows=[]
    for case in CASES:
        model,theta,compilation=invoke(models,hermite,*case)
        value=record(compilation,model,theta,models,apf)
        assert value['value_score']['finite']
        rows.append({'case':case,'record':value})
    # Ordered invalid observations, count and stationarity have differing
    # original family boundaries; capture exact public errors before editing.
    errors=[]
    for family in FAMILIES:
        _,_,observed=inputs(models,3)
        cases=[('stationarity',{'theta_reference':tf.constant([2.,0.],D)}),
               ('small_count',{'particle_count':1}),
               ('initial_nonfinite',{'observations':tf.tensor_scatter_nd_update(observed,[[0,0]],tf.constant([float('nan')],D))}),
               ('later_nonfinite',{'observations':tf.tensor_scatter_nd_update(observed,[[1,0]],tf.constant([float('inf')],D))})]
        if family in ('student','dmis'):
            cases.append(('later_zero',{'observations':tf.tensor_scatter_nd_update(observed,[[1,0]],tf.constant([0.],D))}))
        for label,arguments in cases:
            try:
                _,_,value=invoke(models,hermite,family,3,16,813,**arguments)
            except (ValueError,TypeError,tf.errors.OpError) as error:
                errors.append({'family':family,'case':label,'type':type(error).__name__,'message':str(error)})
            else:
                errors.append({'family':family,'case':label,'accepted':True})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-branch-original.json').write_text(json.dumps({'cases':rows,'errors':errors,
        'sources':original.hashes()},indent=2)+'\n')


def _frozen_branch_original():
    import os
    from scripts import run_filter_repair_campaign as runner
    device='cpu' if os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' else 'gpu'
    rows=[row for row in runner.records() if row['key'][1]==f'c2_preparation_branch_original_{device}' and row['state']=='passed']
    assert rows
    return json.loads((Path(rows[-1]['result']).parent/'c2-branch-original.json').read_text())


def test_native_bootstrap_complete_records(request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    from tests.test_filter_repair_c2_preparation import compare
    reports=[]
    for frozen in _frozen_branch_original()['cases']:
        if frozen['case'][0]!='bootstrap':
            continue
        model,theta,compilation=invoke(models,hermite,*frozen['case'])
        actual=record(compilation,model,theta,models,apf)
        report={'max_abs':0.}
        for field in ('branch','diagnostics','value_score'):
            compare(frozen['record'][field],actual[field],(field,),report)
        compare({k:v for k,v in frozen['record']['manifest'].items() if k!='branch_id'},
                {k:v for k,v in actual['manifest'].items() if k!='branch_id'},('manifest',),report)
        reports.append({'case':frozen['case'],'maximum_absolute_error':report['max_abs'],'current':actual})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-bootstrap-comparison.json').write_text(json.dumps(reports,indent=2)+'\n')


def test_native_bootstrap_errors_live_inputs_and_horizon_one(request):
    import numpy as np
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    from tests.test_filter_repair_c2_preparation import compare
    old=MaterializedCheckpoint(BASELINE,'c2_bootstrap_edge_original')
    original=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    old_hermite=old.load('bayesfilter.highdim.c2_gaussian_hermite_proposal_tf')
    old_apf=old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    records=[]
    for module,proposal_module,score_module in ((original,old_hermite,old_apf),(models,hermite,apf)):
        model,theta,compilation=invoke(module,proposal_module,'bootstrap',1,16,813)
        records.append(record(compilation,model,theta,module,score_module))
    for field in ('branch','diagnostics','value_score'):
        compare(records[0][field],records[1][field],(field,))
    model,theta,observed=inputs(models,3)
    args=dict(model=model,observations=observed,theta_reference=theta,particle_count=16,seed=813)
    first=models.compile_c2_bootstrap_proposal_branch(**args)
    owner=next(iter(model._c2_branch_preparation_owners.values()))
    changed=tf.tensor_scatter_nd_add(observed,[[1,0]],tf.constant([.7],D))
    for overrides in ({'seed':814},{'observations':changed},{'theta_reference':theta+tf.constant([.01,.02],D)}):
        result=models.compile_c2_bootstrap_proposal_branch(**{**args,**overrides})
        assert not np.array_equal(result.branch.states.numpy(),first.branch.states.numpy())
        assert next(iter(model._c2_branch_preparation_owners.values())) is owner
    assert owner.experimental_get_tracing_count()==len(model._c2_branch_preparation_owners)==1
    assert owner.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    graph=owner.get_concrete_function().graph.as_graph_def()
    nodes=[*graph.node,*(n for f in graph.library.function for n in f.node_def)]
    assert any(n.op in ('While','StatelessWhile') for n in nodes)
    assert not any(n.op in ('PyFunc','EagerPyFunc','PyFuncStateless') for n in nodes)
    hlo=owner.experimental_get_compiler_ir(observed,theta,tf.constant(813,tf.int64),())(stage='hlo')
    assert 'while' in hlo.lower()
    for case,overrides in (('stationarity',{'theta_reference':tf.constant([2.,0.],D)}),
        ('small_count',{'particle_count':1}),
        ('initial_nonfinite',{'observations':tf.tensor_scatter_nd_update(observed,[[0,0]],tf.constant([float('nan')],D))}),
        ('later_nonfinite',{'observations':tf.tensor_scatter_nd_update(observed,[[1,0]],tf.constant([float('inf')],D))})):
        expected=next(row for row in _frozen_branch_original()['errors'] if row['family']=='bootstrap' and row['case']==case)
        try:
            models.compile_c2_bootstrap_proposal_branch(**{**args,**overrides})
        except (ValueError,TypeError,tf.errors.OpError) as error:
            assert type(error).__name__==expected['type'] and str(error)==expected['message']
        else:
            raise AssertionError(f'{case} did not reject')
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-bootstrap-owner.json').write_text(json.dumps({'one_trace':True,
        'horizon_one_equal':True,'enclosing_xla':True,'hlo_bytes':len(hlo)},indent=2)+'\n')


def _without_payload_id(value):
    if isinstance(value,dict):
        return {k:_without_payload_id(v) for k,v in value.items() if k not in ('proposal_id','branch_id')}
    if isinstance(value,list):
        return [_without_payload_id(v) for v in value]
    return value


def test_native_gaussian_complete_records(request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    from tests.test_filter_repair_c2_preparation import compare
    reports=[]
    for frozen in _frozen_branch_original()['cases']:
        if frozen['case'][0] not in ('gaussian','stationary'):
            continue
        model,theta,compilation=invoke(models,hermite,*frozen['case'])
        actual=record(compilation,model,theta,models,apf)
        report={'max_abs':0.}
        for field in ('branch','diagnostics','value_score','manifest'):
            compare(_without_payload_id(frozen['record'][field]),_without_payload_id(actual[field]),(field,),report)
        reports.append({'case':frozen['case'],'maximum_absolute_error':report['max_abs'],'current':actual})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-gaussian-comparison.json').write_text(json.dumps(reports,indent=2)+'\n')


def test_gaussian_preparation_live_parameters_geometry_and_error_order(request):
    import hashlib
    import numpy as np
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from tests.test_filter_repair_c2_preparation import compare
    old=MaterializedCheckpoint(BASELINE,'c2_gaussian_geometry_original')
    original=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    shapes=[]
    for module in (original,models):
        model,theta,observed=inputs(module,3)
        proposals=module.stationary_gaussian_proposals(model,theta,3)
        values=[]
        for p in proposals:
            h=hashlib.sha256()
            h.update(str(p.family).encode());h.update(str(int(p.time_index)).encode('ascii'))
            h.update(bytes(tf.io.serialize_tensor(p.mean).numpy()));h.update(bytes(tf.io.serialize_tensor(p.chol).numpy()))
            assert h.hexdigest()==p.proposal_id
            values.append(jsonable({'mean':p.mean,'chol':p.chol}))
        shapes.append(values)
    compare(shapes[0],shapes[1])
    assert model._c2_stationary_proposal_owner.experimental_get_tracing_count()==1
    first_owner=model._c2_stationary_proposal_owner
    models.stationary_gaussian_proposals(model,theta+tf.constant([.01,0.],D),3)
    assert model._c2_stationary_proposal_owner is first_owner
    assert first_owner.experimental_get_tracing_count()==1
    proposals=tuple(gaussian_proposal(models,t) for t in (1,2))
    args=dict(model=model,observations=observed,theta_reference=theta,particle_count=16,
              seed=813,family='gaussian',transition_proposals=proposals)
    first=models.compile_c2_independent_proposal_branch(**args)
    owner=next(iter(model._c2_branch_preparation_owners.values()))
    alternate=(models.FrozenGaussianStateProposal(mean=proposals[0].mean+.05,
                chol=proposals[0].chol,time_index=1,family='gaussian_hint_marginal'),proposals[1])
    for changes in ({'seed':814},{'theta_reference':theta+tf.constant([.01,.02],D)},
                    {'transition_proposals':alternate}):
        value=models.compile_c2_independent_proposal_branch(**{**args,**changes})
        assert not np.array_equal(value.branch.states.numpy(),first.branch.states.numpy())
        assert next(iter(model._c2_branch_preparation_owners.values())) is owner
    assert len(model._c2_branch_preparation_owners)==owner.experimental_get_tracing_count()==1
    assert owner.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    graph=owner.get_concrete_function().graph.as_graph_def()
    nodes=[*graph.node,*(n for f in graph.library.function for n in f.node_def)]
    assert any(n.op in ('While','StatelessWhile') for n in nodes)
    assert not any(n.op in ('PyFunc','EagerPyFunc','PyFuncStateless') for n in nodes)
    mismatch=models.FrozenGaussianStateProposal(mean=proposals[0].mean,chol=proposals[0].chol,
                    time_index=2,family='gaussian_hint_marginal')
    for invalid_observation in (False,True):
        errors=[]
        for module in (original,models):
            m,parameters,obs=inputs(module,3)
            if invalid_observation:
                obs=tf.tensor_scatter_nd_update(obs,[[0,0]],tf.constant([float('nan')],D))
            bad=module.FrozenGaussianStateProposal(mean=mismatch.mean,chol=mismatch.chol,time_index=2,family=mismatch.family)
            try:
                module.compile_c2_independent_proposal_branch(model=m,observations=obs,theta_reference=parameters,
                    particle_count=16,seed=813,family='gaussian',transition_proposals=(bad,gaussian_proposal(module,2)))
            except ValueError as e:
                errors.append(str(e))
            else: raise AssertionError('time-index mismatch accepted')
        assert errors[0]==errors[1]
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-gaussian-owner.json').write_text(json.dumps({'live_proposal_operands':True,
        'one_trace':True,'ordered_metadata_error_preserved':True,'true_proposal_ids_verified':True},indent=2)+'\n')


def test_native_hermite_complete_records(request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    from tests.test_filter_repair_c2_preparation import compare
    reports=[]
    for frozen in _frozen_branch_original()['cases']:
        if frozen['case'][0] not in ('hermite','mixed'):
            continue
        model,theta,compilation=invoke(models,hermite,*frozen['case'])
        actual=record(compilation,model,theta,models,apf)
        report={'max_abs':0.}
        for field in ('branch','diagnostics','value_score','manifest'):
            compare(_without_payload_id(frozen['record'][field]),_without_payload_id(actual[field]),(field,),report)
        reports.append({'case':frozen['case'],'maximum_absolute_error':report['max_abs'],'current':actual})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-hermite-comparison.json').write_text(json.dumps(reports,indent=2)+'\n')



def varied_retained_proposal(hermite,time,variant):
    if variant == 0:
        return retained_proposal(hermite,time)
    first=tf.constant([[[1.,.05],[.12,.01],[.02,-.03]]],D)
    second=tf.constant([[[1.],[-.08],[.01]],[[.1],[.03],[-.02]]],D)
    coefficients=tf.einsum('akb,bjc->kj',first,second)
    return hermite.GaussianHermiteRetainedProposal(prefix_core_values=(first,second),
        suffix_gram=tf.ones([1,1],D),z_h=tf.reduce_sum(coefficients**2),tau_abs=tf.constant(.02,D),
        coordinate_offset=tf.constant([.1*time,-.05*time],D),coordinate_matrix=tf.constant([[1.,0.],[.1,.9]],D),
        defensive_nu=None if variant==1 else 7.,time_index=time,source_snapshot_fingerprint=f'{time:064x}')


def test_hermite_heterogeneous_protocol_and_live_inputs(request):
    import numpy as np
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    from tests.test_filter_repair_c2_preparation import compare
    old=MaterializedCheckpoint(BASELINE,'c2_hermite_heterogeneous_original')
    original=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    old_hermite=old.load('bayesfilter.highdim.c2_gaussian_hermite_proposal_tf')
    old_apf=old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    records=[]
    for module,proposal_module,score in ((original,old_hermite,old_apf),(models,hermite,apf)):
        model,theta,observed=inputs(module,4)
        proposals=tuple(varied_retained_proposal(proposal_module,t,t-1) for t in (1,2,3))
        args=dict(model=model,observations=observed,theta_reference=theta,particle_count=16,
                  seed=813,family='heterogeneous_retained',transition_proposals=proposals)
        compilation=module.compile_c2_independent_proposal_branch(**args)
        records.append(record(compilation,model,theta,module,score))
    report={'max_abs':0.}
    for field in ('branch','diagnostics','value_score','manifest'):
        compare(_without_payload_id(records[0][field]),_without_payload_id(records[1][field]),(field,),report)
    owner=next(iter(model._c2_branch_preparation_owners.values()))
    from dataclasses import replace
    changed=(replace(proposals[0],coordinate_offset=proposals[0].coordinate_offset+.05),*proposals[1:])
    for override in ({'seed':814},{'theta_reference':theta+tf.constant([.01,.02],D)},
                     {'transition_proposals':changed}):
        value=models.compile_c2_independent_proposal_branch(**{**args,**override})
        assert not np.array_equal(compilation.branch.states.numpy(),value.branch.states.numpy())
        assert next(iter(model._c2_branch_preparation_owners.values())) is owner
    assert owner.experimental_get_tracing_count()==len(model._c2_branch_preparation_owners)==1
    assert owner.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    graph=owner.get_concrete_function().graph.as_graph_def()
    nodes=[*graph.node,*(n for f in graph.library.function for n in f.node_def)]
    assert any(n.op in ('While','StatelessWhile') for n in nodes)
    assert not any(n.op in ('PyFunc','EagerPyFunc','PyFuncStateless') for n in nodes)
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-hermite-protocol.json').write_text(json.dumps({'max_abs':report['max_abs'],
        'configurations':3,'rank_degree_and_nu_vary':True,'one_trace':True,
        'original':records[0],'current':records[1]},indent=2)+'\n')


def test_hermite_fixed_configuration_graph_growth(request):
    from collections import Counter
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim.c2_branch_preparation_tf import make_branch_preparation
    from bayesfilter.highdim.c2_independent_preparation_tf import pack_proposals,independent_step,diagnostic_specs
    model,_,_=inputs(models,3)
    graphs=[]
    for horizon in (3,7,11,3):
        proposals=tuple(varied_retained_proposal(hermite,t,t%2) for t in range(1,horizon))
        configurations,operands,specs=pack_proposals(proposals)
        assert len(configurations)==2
        owner=make_branch_preparation(model,horizon,16,independent_step(configurations,16),specs,diagnostic_specs())
        graph=owner.get_concrete_function().graph.as_graph_def()
        nodes=[*graph.node,*(n for f in graph.library.function for n in f.node_def)]
        graphs.append({'horizon':horizon,'configurations':len(configurations),
                       'op_counts':dict(Counter(n.op for n in nodes)),'functions':len(graph.library.function)})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-hermite-graph-growth.json').write_text(json.dumps(graphs,indent=2)+'\n')
    numerical=[{name:count for name,count in row['op_counts'].items() if name not in ('Const','Fill')}
               for row in graphs]
    assert all(counts==numerical[0] for counts in numerical)
    assert graphs[0]['op_counts']==graphs[3]['op_counts']
    assert graphs[1]['op_counts']==graphs[2]['op_counts']
    assert all(row['functions']==graphs[0]['functions'] for row in graphs)
