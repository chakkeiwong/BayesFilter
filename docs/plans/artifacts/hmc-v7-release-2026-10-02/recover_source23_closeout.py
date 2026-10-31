"""Recover every saved member and checkpoint from the interrupted source23 price."""
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
            scope='all-member numerical replay recovery; no new sampling',
            jit_compile=True)
        write('manifest.json', manifest)
        with (args.output/'run.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1440).returncode
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
    base=Path(__file__).resolve().parents[1]/'gpu-price-stable-probe-01/nonlinear/model'
    config=json.loads((base/'configuration.json').read_text())
    saved=json.loads((base/'tuning/tuning_checkpoint.json').read_text())['result']
    assert saved['completion_status']=='complete'
    target=ValidationTarget(config['target'],config['parameters'],config['data'])
    paths=sorted(base.glob('*-member.json'))
    assert len(paths)==19
    from bayesfilter.inference import load_hmc_candidate_retained_runners, load_numerical_tuning_checkpoint
    from bayesfilter.inference.hmc_candidate_set_execution import _fresh_numerical_replay_scope
    from bayesfilter.testing.acceptance_decision_models import evidence_accounting
    from bayesfilter.inference import hmc_verification as verification
    spans={};originals={}
    def time_validator(name):
        original=getattr(verification,name);originals[name]=original
        spans[name]=dict(calls=0,seconds=0.)
        def wrapped(*pos,**kw):
            started=time.monotonic()
            try:return original(*pos,**kw)
            finally:
                spans[name]['seconds']+=time.monotonic()-started
                spans[name]['calls']+=1
        setattr(verification,name,wrapped)
    for name in ('_validate_signed_proxy_summary','_validate_valid_acceptance_summary'):
        time_validator(name)
    stages=[];started=time.monotonic()
    def stage(name):
        stages.append(dict(stage=name,elapsed_seconds=time.monotonic()-started))
        write('stages.json',stages)
    stage('replay_started')
    try:
        with _fresh_numerical_replay_scope():
            members=load_hmc_candidate_retained_runners(paths,adapter=target)
            assert set(members)==set(saved['verified_candidate_ids'])
            stage('all_members_reloaded')
            binding,controller=load_numerical_tuning_checkpoint(base/'tuning/tuning_checkpoint.json',adapter=target)
        stage('checkpoint_reconstructed')
        result=controller.result()
        assert result.candidate_states==saved['candidate_states']
        assert set(result.verified_candidate_ids)==set(members)
        accounting=evidence_accounting(binding,result)
        stage('accounting_checked')
        check_source(args.source)
        write('result.json',dict(status='complete_search_closeout_recovered',verified_members=len(members),
            verified_candidate_ids=sorted(members),candidate_states=result.candidate_states,
            completion_status=result.completion_status,checkpoint_recomputed=True,
            source_manifest_sha256=source_hash,original_search_root=str(base),
            original_configuration=config,stages=stages,evidence_accounting=accounting,
            new_numerical_sampling=False,release_ready=False,interrupted_price_preserved=True,
            validator_spans=spans,span_interpretation='Inclusive instrumented validator timings; no sampling or speed ranking',
            allocator=tf.config.experimental.get_memory_info('GPU:0')))
    finally:
        for name,original in originals.items():setattr(verification,name,original)
        write('validator-spans.json',spans)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
