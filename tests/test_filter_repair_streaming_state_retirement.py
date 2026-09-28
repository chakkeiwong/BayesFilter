"""Preserve a rejected compiler-layout trial and verify its runtime removal."""

import hashlib
import json
import tarfile
from pathlib import Path


def test_rejected_scalar_state_layout_is_archived_and_not_current(request):
    root = Path(__file__).resolve().parents[1]
    source = 'bayesfilter/highdim/ledh_canonical_value_program_tf.py'
    baseline = '150c30a6970dc6e69b053c7ee24308feab4185112c2ed07e95a773810f51df68'
    candidate = 'c7e428521ed7569c423f400008708d2cf8b575f79c9229602d963a564a5c0a39'
    archive = root/'docs/plans/artifacts/filter-gradient-repair-20260917/streaming-state-layout-04832-evidence.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'f3df7d4aebbb431200e9d18a2a5052ae014376605689cf4ba246aa735326356a'
    assert hashlib.sha256((root/source).read_bytes()).hexdigest() == baseline
    with tarfile.open(archive, 'r:gz') as bundle:
        assert hashlib.sha256(bundle.extractfile('checkpoint-source/'+source).read()).hexdigest() == candidate
        readback = json.load(bundle.extractfile('run-04832/streaming-state-layout-readback.json'))
        assert readback['qualification'] == 4827 and not readback['performance_accepted']
        assert {row['horizon'] for row in readback['comparisons']} == {32, 128}
        for row in readback['comparisons']:
            assert row['exact_full_record_parity']
            assert row['scalar_over_vector_warm_ratio'] > 1
            assert row['observation_loop']['scalar']['unique_reachable_opcodes']['copy'] == 363
            assert row['observation_loop']['vector']['unique_reachable_opcodes']['copy'] == 360
        # The tested numerical intervention remains available solely as
        # archived evidence. Its launch harness is no longer an active route.
        assert bundle.getmember('checkpoint-source/tests/test_filter_repair_streaming_state_layout.py').isfile()
    assert not (root/'tests/test_filter_repair_streaming_state_layout.py').exists()
    summary = {'baseline_source_sha256': baseline, 'rejected_source_sha256': candidate,
               'archived_trial_preserved': True, 'runtime_restored_exactly': True,
               'cost_acceptance': False, 'conditional_gpu_trial_launched': False}
    (Path(request.config.getoption('xmlpath')).parent/'streaming-state-retirement.json').write_text(
        json.dumps(summary, indent=2)+'\n')
