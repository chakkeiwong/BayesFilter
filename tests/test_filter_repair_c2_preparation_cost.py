"""Diagnostic matched C2 K=1 complete-public preparation costs and reuse."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_c2_preparation import (
    BASELINE,
    FIXTURE,
    MaterializedCheckpoint,
    fixture,
    jsonable,
    record,
)
from tests.test_filter_repair_resource_owners import memory


def synchronize(compilation):
    compilation.branch.states.numpy()


def payload(compilation):
    branch=compilation.branch
    return jsonable({'branch':{name:getattr(branch,name) for name in (
        'observations','states','initial_log_proposal_density','ancestors',
        'auxiliary_log_probabilities','transition_log_proposal_density',
        'initial_log_base_mass','transition_log_base_mass')},
        'diagnostics':compilation.proposal_diagnostics})


@pytest.mark.parametrize('arm',('original','graph','xla'))
def test_k1_complete_public_cost(arm,request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_c2_adapter as current
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    old=MaterializedCheckpoint(BASELINE,'c2_preparation_cost')
    if arm == 'original':
        adapter=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter')
        models=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
        apf=old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    else:
        adapter=current
    output=Path(request.config.getoption('xmlpath')).parent
    model,theta,observed=fixture(models,3)
    assert 'GPU:0' in observed.device
    def invoke():
        return adapter.compile_c2_per_ancestor_ukf_apf_k1(model=model,
            observations=observed,theta_reference=theta,particle_count=16,
            seed=9104,jit_compile=arm != 'graph')
    tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(True) as monitor:
        start_memory=memory()
        start=time.perf_counter(); first=invoke(); synchronize(first)
        cold_seconds=time.perf_counter()-start
        cold_memory=memory()
        expected=payload(first)
        warm=[]
        for _ in range(20):
            start=time.perf_counter(); result=invoke(); synchronize(result)
            elapsed=time.perf_counter()-start
            assert result.branch.branch_id == first.branch.branch_id
            warm.append(elapsed)
        warm_memory=memory()
        reuse=None
        if arm == 'xla':
            samples={'0':memory()}
            for index in range(1,129):
                result=invoke(); synchronize(result)
                assert result.branch.branch_id == first.branch.branch_id
                if index in (64,128):
                    samples[str(index)]=memory()
            owner=next(iter(model._c2_preparation_owners.values()))
            reuse={'calls':128,'samples':samples,'cache_count':len(model._c2_preparation_owners),
                   'trace_count':owner.experimental_get_tracing_count(),
                   'late_rss_growth_bytes':samples['128']['VmRSS']-samples['64']['VmRSS']}
            assert reuse['cache_count'] == reuse['trace_count'] == 1
            assert reuse['late_rss_growth_bytes'] <= 16*2**20
            assert samples['128']['allocator']['current'] == samples['64']['allocator']['current']
        # Value/score and full record comparisons occur after primary costs.
        complete=record(first,model,theta,apf)
        assert complete['value_score']['finite']
    provenance=next(json.loads(line) for line in (output/'process.log').read_text().splitlines()
                    if line.startswith('{"tensorflow_version"'))
    result={'schema':'filter_repair.c2_preparation_cost.v1','arm':arm,
        'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'original_source_sha256':old.hashes(),'placement':observed.device,
        'cold_seconds':cold_seconds,'warm_seconds':warm,
        'memory':{'prepared_inputs':start_memory,'cold':cold_memory,'warm':warm_memory},
        'reuse':reuse,'full_record':complete,'device_observation':monitor.payload(),
        'device_provenance':provenance,
        'scope':'Complete public K=1 preparation, T3/N16/D4/seed9104; synchronization included.',
        'nonclaims':['Three fresh process blocks only; no universal throughput or capacity.',
                     'Old public host control is a reference, not a fully-XLA original.']}
    (output/'c2-k1-cost.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert expected == payload(first)
    assert not monitor.errors
    assert all(p['pid'] == os.getpid() for sample in monitor.samples for p in sample['processes'])


def test_k1_cost_readback(request):
    import math
    import statistics

    from scripts import run_filter_repair_campaign as runner
    from scripts.filter_repair_cost_provenance import validate_cost_device
    from tests.test_filter_repair_c2_preparation import compare

    rows=[r for r in runner.records() if r['key'][1].startswith('c2_preparation_k1_cost_')]
    saved={}
    frozen=runner.source_hashes()
    for row in rows:
        if row['state'] != 'passed':
            continue
        assert row['source_sha256'] == frozen
        path=Path(row['result']).parent/'c2-k1-cost.json'
        value=json.loads(path.read_text())
        validate_cost_device(row,value['device_provenance'],value['device_observation'])
        exit_record=row['process_exit_observation']
        assert not exit_record['errors'] and not exit_record['proc_entry_present']
        assert not any(p['pid'] == exit_record['pid'] for p in exit_record['gpu_processes'])
        assert len(value['warm_seconds']) == 20
        saved[(row['key'][6],value['arm'])]=(row,value)
    assert set(saved)=={(pair,arm) for pair in range(3) for arm in ('original','graph','xla')}
    comparisons=[]
    for control in ('original','graph'):
        pairs=[]
        for pair in range(3):
            before,a=saved[(pair,control)];after,b=saved[(pair,'xla')]
            for field in ('branch','diagnostics','value_score'):
                compare(a['full_record'][field],b['full_record'][field])
            assert a['fixture_sha256'] == b['fixture_sha256']
            assert before['gpu_uuid'] == after['gpu_uuid']
            reuse=b['reuse']
            assert reuse['cache_count'] == reuse['trace_count'] == 1
            assert reuse['calls'] == 128 and reuse['late_rss_growth_bytes'] <= 16*2**20
            metrics=lambda v:{'warm_seconds':statistics.median(v['warm_seconds']),
                'cold_seconds':v['cold_seconds'],'warm_rss_bytes':v['memory']['warm']['VmRSS'],
                'allocator_peak_bytes':v['memory']['warm']['allocator']['peak']}
            av,bv=metrics(a),metrics(b)
            pairs.append({'pair':pair,'before_run':str(Path(before['result']).parent),
                'after_run':str(Path(after['result']).parent),'before':av,'after':bv,
                'warm_ratio':bv['warm_seconds']/av['warm_seconds']})
        logs=[math.log(p['warm_ratio']) for p in pairs]
        center=statistics.mean(logs);half=4.30265272975*statistics.stdev(logs)/math.sqrt(3)
        comparisons.append({'control':control,'pairs':pairs,'geometric_mean_ratio':math.exp(center),
                            'paired_log_t_95_interval':[math.exp(center-half),math.exp(center+half)]})
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-k1-cost-readback.json').write_text(json.dumps({'comparisons':comparisons,
        'failed_workers_preserved':[str(Path(r['result']).parent) for r in rows if r['state']!='passed']},indent=2)+'\n')
