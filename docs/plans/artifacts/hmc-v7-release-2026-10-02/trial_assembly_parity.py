"""Compare full runtime trial assembly on saved data under CPU/GPU placement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2', BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0, {24, 25, 26, 27})
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
        runner = args.output/'runner.py'
        runner.write_bytes(Path(__file__).read_bytes())
        command = [sys.executable, str(runner), '--source', str(args.source),
                   '--output', str(args.output), '--gpu', args.gpu, '--worker']
        started = time.monotonic()
        manifest = dict(command=command, environment=sys.executable, gpu=args.gpu,
            started_utc=datetime.now(timezone.utc).isoformat(), resource='gpu',
            runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
            plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',
            scope='full runtime saved-trial assembly parity; no sampling or artifact authority',
            jit_compile=True)
        write('manifest.json', manifest)
        with (args.output/'run.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=150).returncode
            except subprocess.TimeoutExpired:
                code = 124
        receipt = dict(manifest, exit_code=code, wall_seconds=time.monotonic()-started)
        write('receipt.json', receipt)
        print(json.dumps({'exit_code':code, 'wall_seconds':receipt['wall_seconds']}))
        return code
    sys.path.insert(0, str(args.source))
    from scripts.run_hmc_v7_release_prices import check_source, DEVELOPMENT_SEEDS
    source_hash = check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.inference import PrecomputedMassArtifact
    from bayesfilter.inference.hmc_candidate_set_execution import _rebuild_geometry, _probe, _tensor_payload
    from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
    write('worker-manifest.json', dict(source_manifest_sha256=source_hash, memory_policy=memory,
        tensorflow_version=tf.__version__, git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(), native_sampling=False))
    from types import SimpleNamespace
    from dataclasses import replace
    from bayesfilter.inference.hmc_candidate_set_execution import (
        HMCCandidateExecutionBinding, HMCCandidateExecutionConfig,
        _tensor_from_payload, _trace_from_payload)
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem
    from bayesfilter.inference.hmc_acceptance_trials import _assemble_trials
    base=Path(__file__).resolve().parents[1]/'gpu-price-stable-probe-01/nonlinear/model/tuning'
    spec=json.loads((base/'execution_spec.json').read_text())['execution']
    checkpoint=json.loads((base/'tuning_checkpoint.json').read_text())['result']
    work=min(checkpoint['work_items'],key=lambda w:w['ordinal'])
    # Choose by issued work order before inspecting numerical results.
    record=None;record_path=None
    for receipt in checkpoint['verification_receipts']:
        path=base/'numerical_evidence'/(receipt['numerical_evidence_hash']+'.json')
        candidate=json.loads(path.read_text())
        if candidate['work']['work_item_id']==work['work_item_id']:
            record=candidate;record_path=path;break
    assert record is not None and len(record['trials'])==32
    runtime=SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(spec['config']),
        initial_active_state=_tensor_from_payload(spec['initial_active_state']),
        scope=SimpleNamespace(payload=lambda:spec['scope']))
    runtime.health_failures=lambda initial,samples,trace:HMCCandidateExecutionBinding.health_failures(runtime,initial,samples,trace)
    runtime.analyze_trial=lambda initial,samples,trace:HMCCandidateExecutionBinding.analyze_trial(runtime,initial,samples,trace)
    def normalize(value):return json.loads(json.dumps(value,allow_nan=False))
    work=HMCWorkItem.from_payload(record['work'])
    chunks=record['chunks']
    expected=record['trials']
    from contextlib import nullcontext
    results=[]
    for device in ('original_default','/CPU:0','/GPU:0'):
        with (nullcontext() if device=='original_default' else tf.device(device)):
            # Match native tensor placement while preserving the archived bytes.
            place=(lambda x:x) if device=='original_default' else tf.identity
            decoded=[(place(_tensor_from_payload(c['samples'])),
                tf.nest.map_structure(place,_trace_from_payload(c['trace']))) for c in chunks]
            initial=runtime.initial_active_state
            runtime.initial_active_state=place(initial)
            baseline=None
            for workers in (4,1):
                runtime.config=replace(runtime.config,trial_analysis_workers=workers)
                before=time.monotonic()
                values=normalize(_assemble_trials(runtime,work,chunks,decoded_chunks=decoded))
                elapsed=time.monotonic()-before
                if baseline is None:
                    baseline=values
                    if values!=expected:
                        differences=[]
                        for i,(actual,saved) in enumerate(zip(values,expected)):
                            for key in saved:
                                if actual.get(key)!=saved[key]:
                                    differences.append(dict(trial=i,field=key,actual=actual.get(key) if key in ('health','scores','seed','ordinal') else 'tensor_or_chunk',saved=saved[key] if key in ('health','scores','seed','ordinal') else 'tensor_or_chunk'))
                        write('placement-differences-'+device.replace('/','').replace(':','')+'.json',differences)
                else:
                    assert values==baseline, ('worker parity failed',device)
                if device=='original_default':
                    assert values==expected, 'original saved full records changed'
                results.append(dict(device_scope=device,workers=workers,seconds=elapsed,
                    exact_within_device=True,exact_saved_record=(values==expected),
                    sample_devices=sorted({row[0].device for row in decoded})))
            runtime.initial_active_state=initial
    decoded=[(_tensor_from_payload(c['samples']),_trace_from_payload(c['trace'])) for c in chunks]
    warm_timings=[]
    for pair,order in enumerate(((1,4),(4,1),(1,4))):
        for workers in order:
            runtime.config=replace(runtime.config,trial_analysis_workers=workers)
            before=time.monotonic()
            values=normalize(_assemble_trials(runtime,work,chunks,decoded_chunks=decoded))
            elapsed=time.monotonic()-before
            assert values==expected, ('warm original record changed',pair,workers)
            warm_timings.append(dict(pair=pair,workers=workers,seconds=elapsed,exact=True))
    check_source(args.source)
    write('result.json',dict(status='exact_full_assembly_passed',source_manifest_sha256=source_hash,
        baseline=str(record_path),baseline_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        trials=32,placements=results,warm_timings=warm_timings,native_sampling=False,release_ready=False,default_promoted=False,
        interpretation='Exact archived complete records under original default placement, plus sequential/concurrent equality under each explicit CPU/GPU scope. Timings descriptive only; no new tuning authority.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
