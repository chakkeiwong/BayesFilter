"""Descriptive helper costs and bounded XLA ownership; no speed ranking."""

import gc
import json
import os
import time
from pathlib import Path

import pytest
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_c2_preparation import (
    BASELINE,
    MaterializedCheckpoint,
    jsonable,
)
from tests.test_filter_repair_c2_public_helpers import proposal, sample_inputs
from tests.test_filter_repair_resource_owners import memory


@pytest.mark.parametrize('arm', ('original','current'))
def test_public_helper_descriptive_cost(arm,request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_tf as ukf
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as gaussian
    old=MaterializedCheckpoint(BASELINE,'c2_public_helper_cost_original')
    if arm=='original':
        gaussian=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
        ukf=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_tf')
    p=proposal(gaussian)
    inputs=sample_inputs()
    assert 'GPU:0' in p.mean.device
    rows=[]
    with GPUProcessMonitor(True) as monitor:
        for name,invoke in (
                ('gaussian',lambda:p.sample_with_seed(16,(813,17),jit_compile=True)),
                ('k1',lambda:ukf.sample_k1_apf_step(**inputs,jit_compile=True))):
            tf.config.experimental.reset_memory_stats('GPU:0')
            before=memory()
            start=time.perf_counter();first=invoke();tf.test.experimental.sync_devices()
            cold=time.perf_counter()-start
            after_cold=memory()
            warm=[]
            for _ in range(20):
                start=time.perf_counter();result=invoke();tf.test.experimental.sync_devices()
                warm.append(time.perf_counter()-start)
            after_warm=memory()
            del result
            gc.collect();tf.test.experimental.sync_devices()
            snapshots={'0':memory()}
            owner=None
            if arm=='current':
                owner=(gaussian._gaussian_seed_sampler(16,2,True) if name=='gaussian'
                       else ukf._retained_k1_apf_sampler(4,2,True))
                for index in range(1,129):
                    result=invoke();tf.test.experimental.sync_devices()
                    if index in (64,128):
                        del result
                        gc.collect();tf.test.experimental.sync_devices()
                        snapshots[str(index)]=memory()
            rows.append({'helper':name,'cold_seconds':cold,'warm_seconds':warm,
                'memory':{'before':before,'cold':after_cold,'warm':after_warm},
                'fixed_root_snapshots':snapshots,'first_result':jsonable(first),
                'trace_count':owner.experimental_get_tracing_count() if owner is not None else None})
            del first,owner
            gc.collect();tf.test.experimental.sync_devices()
    output=Path(request.config.getoption('xmlpath')).parent
    provenance=next(json.loads(line) for line in (output/'process.log').read_text().splitlines()
                    if line.startswith('{"tensorflow_version"'))
    report={'schema':'filter_repair.c2_public_helper_cost.v1','arm':arm,'rows':rows,
        'original_sources':old.hashes(),'device_provenance':provenance,
        'device_observation':monitor.payload(),
        'scope':'Complete public helper calls with synchronization; shared process, Gaussian then K1.',
        'nonclaims':['One process pair is descriptive, not a statistically supported speed ranking.',
                     'Fixed-shape lifetime does not prove native eviction or arbitrary capacity.']}
    (output/'c2-public-helper-cost.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not monitor.errors
    assert all(p['pid']==os.getpid() for s in monitor.samples for p in s['processes'])
    if arm=='current':
        for row in rows:
            samples=row['fixed_root_snapshots']
            assert row['trace_count']==1
            assert samples['128']['VmRSS']-samples['64']['VmRSS']<=16*2**20
            assert samples['128']['allocator']['current']==samples['64']['allocator']['current']
