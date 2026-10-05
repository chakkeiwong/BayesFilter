"""CPU diagnostic of real checkpoint serialization; no numerical replay claim."""
import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import shutil
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, checkpoint, output = (p.resolve() for p in
                                 (args.source, args.checkpoint, args.output))
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    sys.path.insert(0, str(source))
    from bayesfilter.inference.hmc_candidate_set_tuning import (
        HMCTuningCandidateSetController, _json_native_sha256)
    from bayesfilter.inference.hmc_candidate_set_artifacts import (
        _canonical, _validate_result_payload)
    from bayesfilter.inference.hmc_candidate_set_checkpoint import (
        write_numerical_tuning_checkpoint)

    raw = checkpoint.read_bytes()
    (output/'snapshot.json').write_bytes(raw)
    payload = json.loads(raw)
    checksum = payload.pop('content_hash')
    assert checksum == _json_native_sha256(payload)
    result_payload = dict(payload['result'])
    checksum = result_payload.pop('result_hash')
    assert checksum == hashlib.sha256(_canonical(result_payload)).hexdigest()
    _validate_result_payload(payload['result'])
    controller = HMCTuningCandidateSetController.from_result_payload(payload['result'])
    result = controller.result()
    spec = json.loads((checkpoint.parent/'execution_spec.json').read_text())
    assert spec['binding_hash'] == payload['binding_hash']
    assert spec['binding_hash'] == _json_native_sha256(spec['execution'])
    binding = SimpleNamespace(_spec=spec['execution'], binding_hash=spec['binding_hash'],
                              _evidence={}, _partial={}, _persisted_files={})
    copy = output/'copy'
    copy.mkdir()
    def preserve(relative, expected=None):
        original, target = checkpoint.parent/relative, copy/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(original.read_text())
        if expected is not None:
            assert _json_native_sha256(data) == expected
        shutil.copyfile(original, target)
        stat = target.stat()
        binding._persisted_files[str(target.resolve())] = (stat.st_mtime_ns, stat.st_size)
        return data
    preserve(Path('execution_spec.json'))
    for digest in payload['numerical_evidence_hashes']:
        binding._evidence[digest] = preserve(Path('numerical_evidence')/(digest+'.json'), digest)
    for work_id, digests in payload['partial_chunks'].items():
        binding._partial[work_id] = [preserve(Path('numerical_chunks')/(digest+'.json'), digest)
                                     for digest in digests]
    report = dict(scope='writer-only diagnostic; hashes checked; numerical evidence not replayed',
        source=str(source), source_manifest_sha256=hashlib.sha256(
            (source.parent/'source-manifest.json').read_bytes()).hexdigest(),
        checkpoint=str(checkpoint), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_utc=started_utc,
        gpu_intentionally_hidden=True, native_execution=False,
        evidence_records=len(binding._evidence),
        partial_chunks=sum(len(rows) for rows in binding._partial.values()),
        snapshot_loading_seconds=time.monotonic()-started, timings={})
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    for mode in ('full', 'incremental'):
        profile = cProfile.Profile()
        timings = []
        try:
            for _ in range(3):
                before = time.monotonic()
                profile.runcall(write_numerical_tuning_checkpoint, binding, result, copy,
                                _incremental=mode == 'incremental')
                timings.append(time.monotonic()-before)
        finally:
            profile.dump_stats(str(output/(mode+'.pstats')))
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(35)
            (output/(mode+'.txt')).write_text(stream.getvalue())
            report['timings'][mode] = timings
            if (copy/'tuning_checkpoint.json').exists():
                digest = hashlib.sha256((copy/'tuning_checkpoint.json').read_bytes()).hexdigest()
                previous = report.setdefault('written_checkpoint_sha256', digest)
                assert digest == previous, 'full and incremental writes changed the payload'
            report['wall_seconds'] = time.monotonic()-started
            (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in
        ('snapshot_loading_seconds','evidence_records','partial_chunks','timings','wall_seconds')}))


if __name__ == '__main__':
    main()
