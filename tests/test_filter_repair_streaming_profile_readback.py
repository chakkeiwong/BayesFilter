"""Source-bound explanatory profile; preserve the full-filter cost veto."""

import hashlib
import json
from pathlib import Path

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_streaming_profile_readback(request):
    hlo = _load(4750, 'streaming-hlo-profile.json')
    summary = []
    for pair in hlo['comparisons']:
        for arm in ('buffered', 'streaming'):
            row = pair[arm]
            source = RAW / f"run-{row['run']:05d}" / 'paired-optimized_hlo.txt'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['hlo_sha256']
            loop = next(loop for loop in row['entry_loops'] if loop['observation_loop_shape'])
            assert bool(loop['philox_metadata_anchors']) == (arm == 'streaming')
    for horizon, numbers in ((32, (4751, 4752)), (128, (4753, 4754))):
        arms = {}
        for number in numbers:
            run = _load(number, 'run.json')
            assert run['state'] == 'passed' and run['device'] == 'CPU'
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            assert _provenance(number)['cuda_visible_devices'] == '-1'
            path = 'bayesfilter/ops/ledh_random_compat_tf.py'
            assert run['source_sha256'][path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            result = _load(number, 'streaming-rng-profile.json')
            assert result['horizon'] == horizon and result['trace_count'] == 1
            assert len(result['warm_seconds']) == 30
            assert result['complete_cloud_and_state_exact_for_seeds'] == [123, 124]
            source = RAW / f'run-{number:05d}' / 'streaming-rng-optimized_hlo.txt'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == result['hlo_sha256']
            assert bool(result['process_buffer_shape_occurrences']) == (result['arm'] == 'buffered')
            arms[result['arm']] = {'run': number, 'cold_seconds': result['cold_seconds'],
                'warm_ms': 1000 * result['warm_median_seconds'], 'rss': result['rss']}
        summary.append({'horizon': horizon, 'arms': arms,
            'streaming_minus_buffered_warm_ms': arms['streaming']['warm_ms'] - arms['buffered']['warm_ms'],
            'role': 'single_process_per_arm_component_explanation_only'})
    paired = _load(4749, 'streaming-paired-analysis.json')
    assert all(c['requires_runtime_profiling'] for c in paired['comparisons'])
    report = {'schema': 'filter_repair_streaming_profile_summary.v1', 'rng_components': summary,
        'hlo': hlo, 'full_filter_runtime_cost_accepted': False,
        'nonclaims': ['Component timing is not the full-filter intervention or a statistical ranking.',
            'Static opcode counts are not executed multiplicities or causal evidence.']}
    (Path(request.config.getoption('xmlpath')).parent / 'streaming-profile-readback.json').write_text(
        json.dumps(report, indent=2) + '\n')
