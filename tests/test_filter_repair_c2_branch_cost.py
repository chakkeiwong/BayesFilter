"""Diagnostic complete-public preparation costs for the remaining C2 families."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_c2_preparation import BASELINE, MaterializedCheckpoint, jsonable
from tests.test_filter_repair_c2_branch_preparation import (
    inputs, gaussian_proposal, retained_proposal, record, _without_payload_id,
)
from tests.test_filter_repair_resource_owners import memory

FAMILIES = ('bootstrap', 'stationary', 'mixed', 'student', 'dmis')
ARMS = ('original', 'graph', 'xla')


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('arm', ARMS)
def test_branch_complete_public_cost(family, arm, request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as models
    from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as hermite
    from bayesfilter.highdim import zhao_cui_frozen_proposal_apf_tf as apf
    old = MaterializedCheckpoint(BASELINE, 'c2_branch_cost_original')
    if arm == 'original':
        models = old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
        hermite = old.load('bayesfilter.highdim.c2_gaussian_hermite_proposal_tf')
        apf = old.load('bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf')
    model, theta, observed = inputs(models, 4)
    assert 'GPU:0' in observed.device
    proposals = ()
    if family in ('mixed', 'dmis'):
        proposals = tuple(gaussian_proposal(models, t) if family == 'mixed' and t % 2
                          else retained_proposal(hermite, t) for t in (1, 2, 3))
    common = dict(model=model, observations=observed, theta_reference=theta,
                  particle_count=20, seed=-814, jit_compile_sampler=arm != 'graph')

    def invoke():
        if family == 'bootstrap':
            return models.compile_c2_bootstrap_proposal_branch(**common)
        if family == 'student':
            return models.compile_c2_transformed_student_proposal_branch(**common, nu=8.)
        if family == 'dmis':
            return models.compile_c2_dmis_proposal_branch(**common,
                transition_proposals=proposals, alpha=.5, nu=8.)
        supplied = models.stationary_gaussian_proposals(model, theta, 4) if family == 'stationary' else proposals
        return models.compile_c2_independent_proposal_branch(**common,
            transition_proposals=supplied, family=family)

    def synchronize(compilation):
        compilation.branch.states.numpy()

    output = Path(request.config.getoption('xmlpath')).parent
    fixture = jsonable({'coupling': model.coupling_matrix, 'sigma': model.sigma,
        'theta': theta, 'observations': observed, 'horizon': 4, 'particle_count': 20,
        'seed': -814, 'proposal_inputs': [_without_payload_id(p.manifest_payload()) for p in proposals]})
    fixture_digest = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
    tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(True) as monitor:
        start_memory = memory()
        start = time.perf_counter(); first = invoke(); synchronize(first)
        cold_seconds = time.perf_counter()-start
        cold_memory = memory()
        warm = []
        for _ in range(20):
            start = time.perf_counter(); result = invoke(); synchronize(result)
            warm.append(time.perf_counter()-start)
            assert result.branch.branch_id == first.branch.branch_id
        warm_memory = memory()
        reuse = None
        if arm == 'xla':
            samples = {'0': memory()}
            for index in range(1, 129):
                result = invoke(); synchronize(result)
                assert result.branch.branch_id == first.branch.branch_id
                if index in (64, 128):
                    samples[str(index)] = memory()
            owner = next(iter(model._c2_branch_preparation_owners.values()))
            reuse = {'calls': 128, 'samples': samples,
                'cache_count': len(model._c2_branch_preparation_owners),
                'trace_count': owner.experimental_get_tracing_count(),
                'late_rss_growth_bytes': samples['128']['VmRSS']-samples['64']['VmRSS']}
            assert reuse['cache_count'] == reuse['trace_count'] == 1
            assert owner.get_concrete_function().function_def.attr['_XlaMustCompile'].b
            assert reuse['late_rss_growth_bytes'] <= 16*2**20
            assert samples['128']['allocator']['current'] == samples['64']['allocator']['current']
        complete = record(first, model, theta, models, apf)
        assert complete['value_score']['finite']
    provenance = next(json.loads(line) for line in (output/'process.log').read_text().splitlines()
                      if line.startswith('{"tensorflow_version"'))
    result = {'schema': 'filter_repair.c2_branch_cost.v1', 'family': family, 'arm': arm,
        'fixture': fixture, 'fixture_sha256': fixture_digest, 'original_source_sha256': old.hashes(),
        'placement': observed.device, 'cold_seconds': cold_seconds, 'warm_seconds': warm,
        'memory': {'prepared_inputs': start_memory, 'cold': cold_memory, 'warm': warm_memory},
        'reuse': reuse, 'full_record': complete, 'device_observation': monitor.payload(),
        'device_provenance': provenance, 'jit_compile_sampler': arm != 'graph',
        'scope': 'Complete public C2 preparation, T4/N20/D2/seed-814; includes synchronization and Student/stationary geometry construction.',
        'nonclaims': ['Supplied frozen Hermite/Gaussian proposal construction is outside timing in every arm.',
            'Original public Python control is a reference; graph control disables the enclosing owner but retains shared compiled primitives.',
            'Three tiny process blocks cannot establish universal throughput or capacity.']}
    (output/'c2-branch-cost.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    assert not monitor.errors
    assert all(p['pid'] == os.getpid() for sample in monitor.samples for p in sample['processes'])


def test_branch_cost_readback(request):
    import math
    import statistics

    from scripts import run_filter_repair_campaign as runner
    from scripts.filter_repair_cost_provenance import validate_cost_device
    from tests.test_filter_repair_c2_preparation import compare

    rows = [r for r in runner.records() if r['key'][1].startswith('c2_preparation_branch_cost_')
            and r['key'][1].endswith('_gpu')]
    saved = {}
    frozen = runner.source_hashes()
    for row in rows:
        if row['state'] != 'passed':
            continue
        assert row['source_sha256'] == frozen
        value = json.loads((Path(row['result']).parent/'c2-branch-cost.json').read_text())
        validate_cost_device(row, value['device_provenance'], value['device_observation'])
        exited = row['process_exit_observation']
        assert not exited['errors'] and not exited['proc_entry_present']
        assert not any(p['pid'] == exited['pid'] for p in exited['gpu_processes'])
        assert len(value['warm_seconds']) == 20
        key = value['family'], row['key'][6], value['arm']
        assert key not in saved
        saved[key] = row, value
    assert set(saved) == {(family, pair, arm) for family in FAMILIES for pair in range(3) for arm in ARMS}
    comparisons = []
    for family in FAMILIES:
        for control in ('original', 'graph'):
            pairs = []
            for pair in range(3):
                before, a = saved[(family, pair, control)]; after, b = saved[(family, pair, 'xla')]
                for field in ('branch', 'diagnostics', 'value_score', 'manifest'):
                    compare(_without_payload_id(a['full_record'][field]), _without_payload_id(b['full_record'][field]))
                assert a['fixture_sha256'] == b['fixture_sha256']
                assert before['gpu_uuid'] == after['gpu_uuid']
                reuse = b['reuse']
                assert reuse['cache_count'] == reuse['trace_count'] == 1
                assert reuse['calls'] == 128 and reuse['late_rss_growth_bytes'] <= 16*2**20
                def metrics(value):
                    return {'warm_seconds': statistics.median(value['warm_seconds']),
                        'cold_seconds': value['cold_seconds'], 'warm_rss_bytes': value['memory']['warm']['VmRSS'],
                        'allocator_peak_bytes': value['memory']['warm']['allocator']['peak']}
                av, bv = metrics(a), metrics(b)
                pairs.append({'pair': pair, 'before_run': str(Path(before['result']).parent),
                    'after_run': str(Path(after['result']).parent), 'before': av, 'after': bv,
                    'warm_ratio': bv['warm_seconds']/av['warm_seconds']})
            logs = [math.log(p['warm_ratio']) for p in pairs]
            center = statistics.mean(logs); half = 4.30265272975*statistics.stdev(logs)/math.sqrt(3)
            comparisons.append({'family': family, 'control': control, 'pairs': pairs,
                'geometric_mean_ratio': math.exp(center),
                'paired_log_t_95_interval': [math.exp(center-half), math.exp(center+half)]})
    output = Path(request.config.getoption('xmlpath')).parent
    (output/'c2-branch-cost-readback.json').write_text(json.dumps({'comparisons': comparisons,
        'failed_workers_preserved': [str(Path(r['result']).parent) for r in rows if r['state'] != 'passed']}, indent=2)+'\n')
