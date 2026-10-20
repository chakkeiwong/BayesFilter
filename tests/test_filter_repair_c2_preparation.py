"""Diagnostic original/current C2 K=1 public qualification."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint

D = tf.float64
BASELINE = '385a348b9'
CASES = [(3,16,9104,False), (4,24,9102,False), (5,16,9101,False),
         (3,20,9103,True), (3,16,-9104,False), (3,16,4294976400,False)]
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json'
BASELINE_ROOT = Path('/tmp/bayesfilter-c2-preparation-original-385a348b9')
NAMES = ('observations', 'states', 'initial_log_proposal_density', 'ancestors',
    'auxiliary_log_probabilities', 'transition_log_proposal_density',
    'initial_log_base_mass', 'transition_log_base_mass')


class MaterializedCheckpoint(FrozenCheckpoint):
    """Diagnostic source loader with real files for source-bound fingerprints."""
    def load(self, name):
        module = super().load(name)
        if hasattr(module, '__file__'):
            relative = name.replace('.', '/') + '.py'
            if name == 'bayesfilter.runtime':
                relative = 'bayesfilter/runtime/__init__.py'
            path = BASELINE_ROOT / relative
            assert path.read_text() == self.sources[relative]
            module.__file__ = str(path)
        return module


def fixture(module, horizon, changed=False):
    data = json.loads(FIXTURE.read_text())
    with tf.device('/CPU:0'):
        theta = tf.constant([data['gamma'], math.log(data['beta'])], D)
        coupling = tf.constant(data['transition_matrix'], D)-theta[0]*tf.eye(4,dtype=D)
        observed = tf.constant(data['observations'][:horizon], D)
        if changed:
            observed = tf.tensor_scatter_nd_add(observed, [[1,0]], tf.constant([.7],D))
    # Explicit identity copies under the selected default device are needed;
    # convert_to_tensor alone may keep a CPU tensor on CPU.
    theta, coupling, observed = tf.identity(theta), tf.identity(coupling), tf.identity(observed)
    model = module.C2StochasticVolatilityFrozenAPFModel(coupling_matrix=coupling,
        sigma=float(data['sigma']))
    return model, theta, observed


def compare(a, b, path=(), report=None):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], (*path,key), report)
    elif isinstance(a, (tuple,list)):
        assert len(a) == len(b), path
        for index,(left,right) in enumerate(zip(a,b,strict=True)):
            compare(left,right,(*path,index),report)
    elif a is None or b is None:
        assert a is None and b is None, path
    elif isinstance(a,str):
        assert a == b, path
    else:
        a,b = np.asarray(a),np.asarray(b)
        assert a.shape == b.shape, path
        if a.dtype.kind in 'biu':
            np.testing.assert_array_equal(a,b,err_msg=str(path))
        else:
            assert np.isfinite(a).all() and np.isfinite(b).all(), path
            np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10,err_msg=str(path))
            if report is not None and a.size:
                report['max_abs']=max(report['max_abs'],float(np.max(np.abs(a-b))))


def jsonable(value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value,dict):
        return {key:jsonable(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [jsonable(item) for item in value]
    return value


def record(compilation, model, theta, apf):
    branch = compilation.branch
    program = apf.prepare_frozen_proposal_apf_program(model,branch)
    compiler_digest = hashlib.sha256(json.dumps(jsonable(compilation.manifest),sort_keys=True).encode('utf-8'))
    compiler_digest.update(branch.branch_id.encode('ascii'))
    assert compiler_digest.hexdigest() == compilation.compiler_id
    return jsonable({'branch':{name:getattr(branch,name) for name in NAMES},
        'diagnostics':compilation.proposal_diagnostics,
        'value_score':program.evaluate(theta),'manifest':compilation.manifest,
        'identities':{'branch_id':branch.branch_id,'compiler_id':compilation.compiler_id,
            'program_id':program.program_id}})


@pytest.mark.parametrize('horizon,count,seed,changed', CASES)
def test_complete_k1_public_reference(horizon,count,seed,changed,request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as current
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    old=MaterializedCheckpoint(BASELINE,'c2_preparation')
    original=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter')
    old_models=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    old_apf=old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    results=[]
    for module,model_module,score_module in ((original,old_models,old_apf),(current,models,apf)):
        model,theta,observed=fixture(model_module,horizon,changed)
        compilation=module.compile_c2_per_ancestor_ukf_apf_k1(model=model,
            observations=observed,theta_reference=theta,particle_count=count,seed=seed)
        results.append(record(compilation,model,theta,score_module))
    report={'max_abs':0.}
    for field in ('branch','diagnostics','value_score'):
        compare(results[0][field],results[1][field],(field,),report)
    for left,right in [(results[0]['manifest'],results[1]['manifest'])]:
        compare({k:v for k,v in left.items() if k != 'branch_id'},
                {k:v for k,v in right.items() if k != 'branch_id'},('manifest',),report)
    # Identity validation must bind each actual payload/source; ordinary
    # float64 rounding may change a valid branch/compiler/program identifier.
    output=Path(request.config.getoption('xmlpath')).parent
    path=output/f'c2-k1-t{horizon}-n{count}-seed{seed}-changed{int(changed)}.json'
    assert not path.exists()
    path.write_text(json.dumps({'original':results[0],'current':results[1],
        'maximum_absolute_error':report['max_abs'],
        'baseline_sources':old.hashes(),
        'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()},indent=2)+'\n')


def test_original_k1_records(request):
    old=MaterializedCheckpoint(BASELINE,'c2_preparation_original')
    adapter=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter')
    models=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    apf=old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    output=Path(request.config.getoption('xmlpath')).parent
    records=[]
    for horizon,count,seed,changed in CASES:
        model,theta,observed=fixture(models,horizon,changed)
        compilation=adapter.compile_c2_per_ancestor_ukf_apf_k1(model=model,
            observations=observed,theta_reference=theta,particle_count=count,seed=seed)
        value=record(compilation,model,theta,apf)
        assert value['value_score']['finite']
        records.append({'case':[horizon,count,seed,changed], 'record':value,
                        'device':compilation.branch.states.device})
    errors=[]
    model,theta,observed=fixture(models,3)
    for label,arguments in (
        ('invalid_stationarity',{'theta_reference':tf.constant([1.2,-.9],D)}),
        ('initial_nan',{'observations':tf.tensor_scatter_nd_update(observed,[[0,0]],tf.constant([float('nan')],D))}),
        ('later_nan',{'observations':tf.tensor_scatter_nd_update(observed,[[1,0]],tf.constant([float('nan')],D))}),
        ('small_count',{'particle_count':1}),
    ):
        base={'model':model,'observations':observed,'theta_reference':theta,'particle_count':16,'seed':9104}
        base.update(arguments)
        try:
            adapter.compile_c2_per_ancestor_ukf_apf_k1(**base)
        except (ValueError, tf.errors.InvalidArgumentError) as exc:
            errors.append({'case':label,'type':type(exc).__name__,'message':str(exc)})
        else:
            raise AssertionError(f'{label} did not fail')
    (output/'c2-k1-original-records.json').write_text(json.dumps({
        'baseline':BASELINE,'cases':records,'errors':errors,'sources':old.hashes(),
        'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()},indent=2)+'\n')


def test_recover_c2_frozen_fixture(request):
    """Recover original seeded diagnostic data; never a runtime implementation."""
    import importlib.util
    import os

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    source=BASELINE_ROOT/'docs/benchmarks/sv_fixture_c2_20260826.py'
    spec=importlib.util.spec_from_file_location('c2_frozen_diagnostic_source',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model=module.sv_model(4,52)
    observed=module.sv_simulate(model,20,42)
    initial,predictive=module.sv_gh_hint_factory(model,gh_points=9)
    hints=[]
    for time in range(20):
        mean,covariance=initial(observed[0]) if time == 0 else predictive(time,observed[time])
        hints.append({'time_index':time,'mean':mean.numpy().tolist(),
                      'covariance':covariance.numpy().tolist()})
    payload={'schema_id':'bayesfilter.c2_sv_frozen_fixture.v1',
        'classification':'cpu_only_numpy_diagnostic_fixture_freeze',
        'source_generator':'docs/benchmarks/sv_fixture_c2_20260826.py',
        'source_generator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'cuda_visible_devices':'-1','state_dimension':4,'model_seed':52,
        'observation_seed':42,'horizon':20,'gamma':.6,'sigma':1.,'beta':.4,
        'transition_matrix':model['A'].tolist(),'process_covariance':model['Q'].tolist(),
        'stationary_covariance':model['P0'].tolist(),'observations':observed.tolist(),
        'gauss_hermite_points_per_axis':9,'moment_hints':hints,
        'runtime_contract':'Load frozen JSON; no NumPy runtime numerical computation'}
    content=json.dumps(payload,indent=2)+'\n'
    if FIXTURE.exists():
        assert json.loads(FIXTURE.read_text()) == payload
    else:
        FIXTURE.parent.mkdir(parents=True,exist_ok=True)
        FIXTURE.write_text(content)
    output=Path(request.config.getoption('xmlpath')).parent
    (output/FIXTURE.name).write_text(content)
    (output/'c2-fixture-recovery.json').write_text(json.dumps({
        'source_commit':BASELINE,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'numpy_version':np.__version__,'tensorflow_version':tf.__version__,
        'status':'recovered_original_seeded_reference_inputs',
        'nonclaims':['No archived historical JSON byte identity claimed; original data recipe recovered.',
                     'All before/current comparisons use these identical frozen bytes.']},indent=2)+'\n')


def test_k1_owner_live_inputs_and_enclosing_xla(request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as current
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models

    model,theta,observed=fixture(models,3)
    arguments={'model':model,'observations':observed,'theta_reference':theta,
               'particle_count':16,'seed':9104}
    first=current.compile_c2_per_ancestor_ukf_apf_k1(**arguments)
    owner=next(iter(model._c2_preparation_owners.values()))
    changed_theta=theta+tf.constant([.015,.03],D)
    changed_observed=tf.tensor_scatter_nd_add(observed,[[1,0]],tf.constant([.7],D))
    variants=[]
    for change in ({'seed':9105},{'observations':changed_observed},{'theta_reference':changed_theta}):
        result=current.compile_c2_per_ancestor_ukf_apf_k1(**(arguments|change))
        assert not np.array_equal(first.branch.states.numpy(),result.branch.states.numpy())
        assert next(iter(model._c2_preparation_owners.values())) is owner
        variants.append(result.branch.branch_id)
    assert len(model._c2_preparation_owners) == 1
    assert owner.experimental_get_tracing_count() == 1
    concrete=owner.get_concrete_function()
    assert bool(concrete.function_def.attr['_XlaMustCompile'].b)
    graph=concrete.graph.as_graph_def()
    nodes=[*graph.node,*(node for function in graph.library.function for node in function.node_def)]
    assert any(node.op in ('While','StatelessWhile') for node in nodes)
    assert not any(node.op in ('PyFunc','EagerPyFunc','PyFuncStateless') for node in nodes)
    hlo=owner.experimental_get_compiler_ir(observed,theta,tf.constant(9104,tf.int64))(stage='hlo')
    assert 'while' in hlo.lower()
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-k1-owner.json').write_text(json.dumps({'trace_count':owner.experimental_get_tracing_count(),
        'cache_count':len(model._c2_preparation_owners),'graph_nodes':len(nodes),
        'graph_functions':len(graph.library.function),'hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest(),
        'distinct_changed_input_branches':variants,'device':first.branch.states.device},indent=2)+'\n')


def test_k1_original_error_order():
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as current
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    old=MaterializedCheckpoint(BASELINE,'c2_preparation_errors')
    original=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter')
    old_models=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    for label,index,value,parameters,count in (
        ('stationarity',None,None,[1.2,-.9],16),
        ('initial_nan',[0,0],float('nan'),None,16),
        ('later_nan',[1,0],float('nan'),None,16),
        ('later_zero',[1,0],0.,None,16),
        ('initial_zero',[0,0],0.,None,16),
        ('small_count',None,None,None,1),
    ):
        errors=[]
        for adapter,module in ((original,old_models),(current,models)):
            model,theta,observed=fixture(module,3)
            if index is not None:
                observed=tf.tensor_scatter_nd_update(observed,[index],tf.constant([value],D))
            if parameters is not None:
                theta=tf.constant(parameters,D)
            try:
                adapter.compile_c2_per_ancestor_ukf_apf_k1(model=model,observations=observed,
                    theta_reference=theta,particle_count=count,seed=9104)
            except (ValueError,tf.errors.InvalidArgumentError) as exc:
                errors.append((type(exc).__name__,str(exc)))
            else:
                errors.append(None)
        assert errors[0] == errors[1], (label,errors)
        if label != 'initial_zero':
            assert errors[0] is not None,label


def test_shared_default_and_bounded_prefix_match_original():
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    old=MaterializedCheckpoint(BASELINE,'c2_preparation_prefix')
    original=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter')
    old_models=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    old_apf=old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    old_model,theta,observed=fixture(old_models,4)
    model,_,_=fixture(models,4)
    branch=original.compile_c2_per_ancestor_ukf_apf_k1(model=old_model,observations=observed,
        theta_reference=theta,particle_count=16,seed=9104).branch
    prefix=tf.function(lambda theta,limit:apf._evaluate_core(model,branch,theta,evaluation_steps=limit),
        input_signature=[tf.TensorSpec([2],D),tf.TensorSpec([],tf.int32)],jit_compile=True,autograph=False)
    full=tf.function(lambda theta:apf._evaluate_core(model,branch,theta),
        input_signature=[tf.TensorSpec([2],D)],jit_compile=True,autograph=False)
    history=('log_increments','increment_scores','ess_by_time',
             'log_weight_spread_by_time','maximum_normalized_weight_by_time')
    for limit in range(1,5):
        partial=old_apf.prepare_frozen_proposal_branch(observations=branch.observations[:limit],
            states=branch.states[:limit],initial_log_proposal_density=branch.initial_log_proposal_density,
            ancestors=branch.ancestors[:limit-1],auxiliary_log_probabilities=branch.auxiliary_log_probabilities[:limit-1],
            transition_log_proposal_density=branch.transition_log_proposal_density[:limit-1])
        reference=old_apf.prepare_frozen_proposal_apf_program(old_model,partial).evaluate(theta)
        actual=prefix(theta,tf.constant(limit,tf.int32))
        common=dict(actual)
        for name in history:
            np.testing.assert_array_equal(actual[name][limit:].numpy(),tf.zeros_like(actual[name][limit:]).numpy())
            common[name]=actual[name][:limit]
        compare(dict(reference),common)
        if limit == 4:
            compare(dict(reference),dict(full(theta)))
    assert prefix.experimental_get_tracing_count() == 1


def test_k1_graph_size_is_bounded(request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim.c2_mixture_ukf_apf_tf import BatchedUKFConfig
    from bayesfilter.highdim.c2_ukf_preparation_tf import make_k1_preparation

    model,_,_=fixture(models,3)
    sizes=[]
    for horizon in (3,7,11,3):
        owner=make_k1_preparation(model,horizon,16,BatchedUKFConfig())
        graph=owner.get_concrete_function().graph.as_graph_def()
        from collections import Counter
        nodes=[*graph.node,*(node for f in graph.library.function for node in f.node_def)]
        sizes.append({'horizon':horizon,'op_counts':dict(Counter(n.op for n in nodes)), 'outer':len(graph.node),
            'all_nodes':len(graph.node)+sum(len(f.node_def) for f in graph.library.function),
            'functions':len(graph.library.function)})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-k1-graph-size.json').write_text(json.dumps(sizes,indent=2)+'\n')
    # TensorFlow uses Const for small zero tensors and Fill above its static
    # element threshold. Compare every computational operation separately.
    arithmetic=[{name:count for name,count in row['op_counts'].items() if name not in ('Const','Fill')}
                for row in sizes]
    assert all(operations == arithmetic[0] for operations in arithmetic)
    assert sizes[1]['op_counts'] == sizes[2]['op_counts']
    assert sizes[0]['op_counts'] == sizes[3]['op_counts']
    assert all(row['functions'] == sizes[0]['functions'] for row in sizes)
