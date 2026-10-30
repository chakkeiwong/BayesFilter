"""Actual C2 benchmark wiring check; no full benchmark or tuning run."""

import importlib.util
import json
import os
from pathlib import Path

from tests.test_filter_repair_c2_branch_preparation import (
    gaussian_proposal,
    inputs,
    retained_proposal,
)
from tests.test_filter_repair_c2_preparation import NAMES, compare, jsonable


def test_actual_candidate_factories_use_complete_compiled_owners(monkeypatch, request):
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as adapter
    from bayesfilter.highdim import c2_mixture_ukf_apf_tf as ukf
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf

    def forbid_reference(*args, **kwargs):
        raise AssertionError('active candidate used a standalone/reference sampling helper')

    monkeypatch.setattr(ukf, 'compile_k1_apf_proposal', forbid_reference)
    monkeypatch.setattr(models.FrozenGaussianStateProposal, 'sample_with_seed', forbid_reference)
    originals={}
    calls=[]

    def wrap(name, function):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return function(**kwargs)
        return invoke

    for module, names in ((models, ('compile_c2_bootstrap_proposal_branch',
            'compile_c2_transformed_student_proposal_branch', 'compile_c2_independent_proposal_branch')),
            (adapter, ('compile_c2_per_ancestor_ukf_apf_k1',
                'compile_c2_per_ancestor_ukf_apf_mixture',
                'compile_c2_per_ancestor_ukf_apf_defensive_mixture'))):
        for name in names:
            originals[name]=getattr(module,name)
            monkeypatch.setattr(module,name,wrap(name,originals[name]))

    root=Path(__file__).resolve().parents[1]
    path=root/'docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py'
    spec=importlib.util.spec_from_file_location('c2_actual_caller_qualification',path)
    benchmark=importlib.util.module_from_spec(spec)
    # The benchmark defers this environment setting during its own imports;
    # this test worker has already configured and verified GPU memory growth.
    growth=os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')
    try:
        spec.loader.exec_module(benchmark)
        benchmark._load_algorithm_modules()
    finally:
        if growth is not None:
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH']=growth

    model,theta,observed=inputs(models,3)
    proposal_sets={
        'gaussian_hint_marginal':tuple(gaussian_proposal(models,t) for t in (1,2)),
        'stationary_independence':models.stationary_gaussian_proposals(model,theta,3),
        'retained_tt':tuple(retained_proposal(hermite,t) for t in (1,2)),
    }
    controls={'nu': 8.,'epsilon_min': .05,'epsilon_max': .2,
        'gate_center': benchmark.PHASE4_GATE_CENTER,
        'gate_temperature': benchmark.PHASE4_GATE_TEMPERATURE}
    candidates=benchmark._phase4_candidate_specs(model=model,observations=observed,
        theta=theta,proposal_sets=proposal_sets,particle_count=16,seed=813,
        mixture_offset=.35,defensive_config=controls)
    assert len(candidates)==11
    records=[]

    def record(compilation):
        program=apf.prepare_frozen_proposal_apf_program(model,compilation.branch)
        value_score=program.evaluate(theta)
        return jsonable({'branch':{name:getattr(compilation.branch,name) for name in NAMES},
            'diagnostics':compilation.proposal_diagnostics,'manifest':compilation.manifest,
            'value_score':value_score,'compiler_id':compilation.compiler_id})

    for family,invoke,_ in candidates:
        start=len(calls)
        result=invoke()
        assert len(calls)==start+1
        name,kwargs=calls[-1]
        assert kwargs.get('jit_compile',kwargs.get('jit_compile_sampler')) is True
        assert kwargs['particle_count']==16 and kwargs['seed']==813
        actual=record(result)
        expected=record(originals[name](**kwargs))
        compare(expected,actual)
        records.append({'family':family,'endpoint':name,'record':actual})
    assert {r['endpoint'] for r in records}==set(originals)
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-actual-candidate-callers.json').write_text(json.dumps({
        'source':str(path.relative_to(root)),'fixture':{'horizon':3,'particles':16,'seed':813},
        'records':records,'reference_and_standalone_fallbacks_blocked':True,
        'scope':'actual candidate factories only; no tuning, training, HMC or full benchmark run'},indent=2)+'\n')
