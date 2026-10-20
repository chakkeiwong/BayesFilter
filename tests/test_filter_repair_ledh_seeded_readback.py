"""Read saved LEDH evidence independently; retain numerical and policy failures."""

import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
DEPENDENCIES = (
    'bayesfilter/highdim/genut_shape_lm_tf.py',
    'bayesfilter/highdim/ledh_canonical_filter_tf.py',
    'bayesfilter/highdim/ledh_canonical_value_program_tf.py',
    'bayesfilter/ops/ledh_random_compat_tf.py',
    'bayesfilter/highdim/genut_guided_proposal_tf.py',
    'bayesfilter/highdim/higher_moment_contract_e.py',
    'bayesfilter/highdim/ledh_flow_perparticle_tf.py',
    'bayesfilter/highdim/ledh_ukf_lifecycle_tf.py',
)


def _load(number, filename):
    return json.loads((RAW / f'run-{number:05d}' / filename).read_text())


def _provenance(number):
    log = (RAW / f'run-{number:05d}' / 'process.log').read_text()
    return next(json.loads(line) for line in log.splitlines() if line.startswith('{"tensorflow_version"'))


def _check(number, count, *, current=True):
    run = _load(number, 'run.json')
    assert run['state'] == 'passed'
    assert run['test_evidence'] == {'passed': True, 'tests': count, 'failure': 0, 'error': 0, 'skipped': 0}
    if current:
        for relative in DEPENDENCIES:
            assert run['source_sha256'][relative] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), relative
    provenance = _provenance(number)
    assert provenance['tf32_enabled'] is True
    policy = provenance['gpu_memory_policy']
    assert policy['configured_before_logical_device_initialization']
    assert policy['all_physical_devices_memory_growth']
    assert provenance['cuda_visible_devices'] == run['environment']['CUDA_VISIBLE_DEVICES']
    if run['device'] == 'GPU':
        assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        assert all(row['memory_growth'] for row in policy['physical_devices'])
    else:
        assert provenance['cuda_visible_devices'] == '-1'
        assert provenance['trust_basis'] == 'explicit_cpu_reference'
    return {'run': number, 'count': count, 'device': run['device'], 'elapsed_seconds': run['elapsed_seconds']}


def test_current_endpoint_and_preserved_failures(request):
    runs = [_check(n, count) for n,count in ((4678,7),(4679,7),(4680,8),(4681,8),(4683,36))]
    prior_failure = _load(4672, 'run.json')
    assert prior_failure['state'] == 'failed' and prior_failure['test_evidence']['failure'] == 1
    old = _load(4672, 'seeded-endpoint-dual_trust.json')['records'][0]
    new = _load(4681, 'seeded-endpoint-dual_trust.json')['records'][0]
    assert old['actual'] == new['actual']  # repair changes the defective eager comparator, not this XLA record
    assert abs(old['actual']['per_step_ess'][1] - old['expected']['per_step_ess'][1]) > 1e-5
    assert abs(new['actual']['per_step_ess'][1] - new['expected']['per_step_ess'][1]) < 2e-6
    for n in (4680,4681):
        for name in ('composed','annealed','dual_trust','one_step','invalid_initial','invalid_prediction','invalid_observation'):
            record = _load(n, f'seeded-endpoint-{name}.json')
            assert record['trace_count'] == 1 and record['owner_collected'] and record['enclosing_xla']
            assert len(record['records']) == 2
            for row in record['records']:
                assert row['actual']['program_valid'] == row['expected']['numerical_valid']
                if row['actual']['program_valid']:
                    assert row['comparison']['comparison_role'] == 'healthy_complete_record_equivalence'
                else:
                    assert row['comparison']['comparison_role'] == 'rejected_raw_diagnostics_only'
    regression = _load(4682, 'run.json')
    assert regression['state'] == 'failed' and regression['test_evidence']['tests'] == 47
    failures = [case.attrib['name'] for case in ET.parse(RAW/'run-04682/junit.xml').iter('testcase') if case.find('failure') is not None]
    assert failures == ['test_batch_claim_paths_ban_python_fanout_and_pfor_apis']
    report = {'schema':'ledh_seeded_current_readback.v1', 'runs':runs,
              'old_gpu_failure_retained':True, 'old_new_xla_record_equal':True,
              'existing_F14_failure':failures, 'whole_master_complete':False}
    path = Path(request.config.getoption('xmlpath')).parent / 'seeded-current-readback.json'
    path.write_text(json.dumps(report, indent=2)+'\n')


def test_cpu_cost_readback(request):
    from scripts.filter_repair_cost_provenance import validate_cost_device

    reports = []
    for n, arm in ((4684,'prior_eager'), (4685,'candidate_graph'), (4686,'candidate_xla')):
        _check(n, 1)
        run = _load(n, 'run.json')
        record = _load(n, 'seeded-cost.json')
        assert record['arm'] == arm and record['numerical_passed'] is True
        assert record['validation_outside_cost_measurements'] is True
        assert len(record['warm_seconds']) == 15
        assert min(record['warm_seconds']) > 0
        assert record['jit_compile'] == (arm == 'candidate_xla')
        assert validate_cost_device(run, _provenance(n), record['cost_provenance']) is None
        if arm != 'prior_eager':
            assert record['trace_count'] == 1 and record['owner_collected']
            assert len(record['one_shot_seconds']) == 2
        reports.append({'run':n, 'arm':arm, 'cold_seconds':record['cold_seconds'],
            'warm_median_seconds':statistics.median(record['warm_seconds']),
            'one_shot_seconds':record['one_shot_seconds'],
            'rss_growth_mib':(record['after_warm']['rss_bytes']-record['before']['rss_bytes'])/2**20,
            'sampled_peak_growth_mib':(record['reuse_sampled_peak_rss_bytes']-record['before']['rss_bytes'])/2**20,
            'post_release_growth_mib':(record['after_release']['rss_bytes']-record['before']['rss_bytes'])/2**20,
            'allocator':record['after_warm']['allocator']})
    report = {'schema':'ledh_seeded_CPU_cost_readback.v1','rows':reports,
              'role':'single_process_per_arm_descriptive_only',
              'GPU_costs':'pending_uncontended_device', 'long_horizon_capacity':'open',
              'baseline_reset':'current_shared_repaired_authority'}
    path = Path(request.config.getoption('xmlpath')).parent / 'seeded-cpu-cost-readback.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
