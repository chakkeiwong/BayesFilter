"""Replay saved original trial streams with the repaired likelihood graph."""
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
            scope='saved-seed native GPU parity; no tuning or release authority',
            jit_compile=True)
        write('manifest.json', manifest)
        with (args.output/'run.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
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
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(), native_sampling=True))
    baseline = Path(__file__).resolve().parents[1]/'source22-nonlinear-work-profile-01/plain/model'
    configuration=json.loads((baseline/'configuration.json').read_text())
    tuning=baseline/'tuning'
    spec=json.loads((tuning/'execution_spec.json').read_text())['execution']
    checkpoint=json.loads((tuning/'tuning_checkpoint.json').read_text())
    assert len(checkpoint['numerical_evidence_hashes'])==1
    raw_path=tuning/'numerical_evidence'/(checkpoint['numerical_evidence_hashes'][0]+'.json')
    raw=json.loads(raw_path.read_text())
    chunks=sorted(raw['chunks'],key=lambda x:x['trial_ordinal'])
    assert len(chunks)==32 and all(c['trial_chunk_index']==0 for c in chunks)
    target=ValidationTarget(configuration['target'],configuration['parameters'],configuration['data'])
    active,_=_rebuild_geometry(target,spec['layers'],spec['target_scope'])
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload, _trace_payload
    from bayesfilter.inference.hmc import FullChainHMCConfig
    from bayesfilter.inference.hmc_replicated_batch import ReplicatedTrialBatchRunner
    starts=_tensor_from_payload(spec['initial_active_state'])
    assert _probe(active,starts,target_status=True)==spec['probe'], 'live probe changed'
    seeds=[c['seed'] for c in chunks]
    assert seeds==chunks[0]['runtime']['trial_batch_seeds']
    candidate=raw['candidate']; count=chunks[0]['count']
    config=FullChainHMCConfig(num_results=count,num_burnin_steps=0,
        step_size=candidate['epsilon'],num_leapfrog_steps=candidate['leapfrog_steps'],
        seed=tuple(seeds[0]),use_xla=True,target_scope=spec['target_scope'],
        target_status_trace_policy='per_chain_step',capture_candidate_health=True)
    runner=ReplicatedTrialBatchRunner(active,starts,config,batch_size=32)
    samples,trace,metadata=runner.run(states=tf.broadcast_to(starts,[32,*starts.shape]),seeds=seeds,
        step_size=candidate['epsilon'],num_leapfrog_steps=candidate['leapfrog_steps'])
    assert 'GPU:0' in samples.device and metadata['jit_compile']
    checked=[]
    for i,c in enumerate(chunks):
        actual_samples=_tensor_payload(samples[:,i])
        actual_trace=_trace_payload(tf.nest.map_structure(lambda value:value[:,i],trace))
        assert actual_samples==c['samples'], ('samples changed',i)
        assert actual_trace==c['trace'], ('trace changed',i)
        checked.append(dict(ordinal=i,seed=seeds[i],samples_sha256=actual_samples['sha256'],exact_trace=True))
    check_source(args.source)
    write('result.json',dict(status='exact_saved_stream_parity',trials=checked,
        source_manifest_sha256=source_hash, baseline=str(raw_path),
        baseline_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        exact_probe=True,original_base_seed=configuration['seed'],
        native_runtime=metadata,allocator=tf.config.experimental.get_memory_info('GPU:0'),
        sample_device=samples.device,release_ready=False,
        interpretation='Same saved source22 trial seeds and starts; direct numerical runner has no tuning authority. New public runs retain original base seed but derive new source-bound trial identities.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
