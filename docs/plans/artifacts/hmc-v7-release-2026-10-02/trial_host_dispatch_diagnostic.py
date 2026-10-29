"""Compare exact saved-trial assembly under bounded host dispatch concurrency."""
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
            scope='saved-trial host dispatch diagnostic; no sampling',
            jit_compile=True)
        write('manifest.json', manifest)
        with (args.output/'run.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=110).returncode
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
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from bayesfilter.inference.hmc_candidate_set_execution import (
        HMCCandidateExecutionBinding, HMCCandidateExecutionConfig, _tensor_from_payload, _trace_from_payload, _trace_payload)
    from bayesfilter.inference.hmc_acceptance_statistics import complete_trial_scores
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
    rows=[(_tensor_from_payload(t['samples']),_trace_from_payload(t['trace'])) for t in record['trials']]
    runtime=SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(spec['config']),
        initial_active_state=_tensor_from_payload(spec['initial_active_state']))
    runtime.health_failures=lambda initial,samples,trace:HMCCandidateExecutionBinding.health_failures(runtime,initial,samples,trace)
    def assemble(row):
        samples,trace=row
        health=HMCCandidateExecutionBinding.analyze_trial(runtime,runtime.initial_active_state,samples,trace)
        scores=None
        if health['evidence_validity']=='valid':
            scores=complete_trial_scores(trace['log_accept_ratio'][runtime.config.num_warmup_steps:],
                policy=runtime.config.acceptance_policy,jit_compile=runtime.config.use_xla)
        return dict(samples=_tensor_payload(samples),trace=_trace_payload(trace),health=health,scores=scores)
    def normalize(value):return json.loads(json.dumps(value,allow_nan=False))
    expected=[{k:t[k] for k in ('samples','trace','health','scores')} for t in record['trials']]
    assert normalize([assemble(row) for row in rows])==expected
    timings=[]
    for workers in (1,2,4):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for repeat in range(3):
                before=time.monotonic()
                values=list(map(assemble,rows)) if workers==1 else list(pool.map(assemble,rows))
                elapsed=time.monotonic()-before
                assert normalize(values)==expected, ('record changed',workers,repeat)
                timings.append(dict(workers=workers,repeat=repeat,seconds=elapsed,exact=True))
    check_source(args.source)
    write('result.json',dict(status='exact_ordered_assembly_passed',source_manifest_sha256=source_hash,
        baseline=str(record_path),baseline_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        trials=32,timings=timings,sample_devices=sorted({row[0].device for row in rows}),
        native_sampling=False,release_ready=False,default_promoted=False,
        interpretation='Descriptive saved-data host dispatch timings; complete original outputs preserved. No concurrency runtime adopted.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
