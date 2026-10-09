"""Native batch capacity diagnostic; no tuning or statistical admission authority."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(args):
    sys.path.insert(0, str(args.source))
    os.chdir(args.source)
    from scripts.run_hmc_v7_release_prices import check_source
    signature = check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.inference import (
        HMCCandidateExecutionConfig, PrecomputedMassArtifact, bind_hmc_candidate_set_execution)
    from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionBinding
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig

    inventory = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    write(args.output/'worker-manifest.json', dict(source_manifest_sha256=signature,
        memory_policy=memory, gpu_uuid=args.gpu, gpu_inventory=inventory.splitlines(),
        tensorflow_version=tf.__version__, tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        jit_compile=True, dtype='float64', seeds=[[20261003,2800+i] for i in range(18)],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        scope='native capacity only; shape-dependent streams; no tuning, reliability or release claim'))
    rows = []
    for model_index, case in enumerate(('lgssm_qr', 'nonlinear')):
        config = full_search_configuration(case, seed=(20261002,2501+model_index), wall_seconds=1800)
        write(args.output/(case+'-config.json'), config)
        policy = HMCControllerConfig.from_payload(config['search']).replicated_acceptance_policy
        target = ValidationTarget(config['target'], config['parameters'], config['data'])
        starts = tf.constant(config['active_starts'], tf.float64)
        factor = tf.linalg.diag(tf.constant(config['geometry']['scale'], tf.float64))
        mass = PrecomputedMassArtifact(position=config['geometry']['center'], factor=factor,
            covariance=tf.matmul(factor,factor,transpose_b=True), adapter_signature=target.adapter_signature(),
            position_role='declared_diagnostic_preparation', covariance_source=config['provenance'])
        execution = HMCCandidateExecutionConfig(measurement_num_results=65, verification_num_results=65,
            num_warmup_steps=3, seed=(20261003,2800), acceptance_policy=policy,
            target_status_trace_policy='per_chain_step', use_xla=True, chain_mode='batched',
            reuse_leapfrog_graphs=True, chunk_max_results=68)
        binding = bind_hmc_candidate_set_execution(adapter=target, initial_position=starts,
            start_coordinates='active', target_scope='inference_validation', config=execution,
            target_lineage={'model':target.target_id,'data':target.data,'parameters':target.parameters},
            source_paths=[str(args.source/'bayesfilter/testing/inference_validation/targets.py'),
                str(args.source/'bayesfilter/testing/inference_validation/ssm_targets.py'),
                target.value_score_capability().evidence_path], mass_artifact=mass,
            scope_id='native-batch-capacity', search_id=case,
            epsilon_domain=tuple(config['epsilon_domain']), repair_factor=config['repair_factor'],
            max_repairs_per_family=config['max_repairs_per_family'])
        adapter = binding._active_adapter
        base_values = adapter.log_prob_and_grad(starts)
        base_status = adapter.target_status_telemetry(starts)
        epsilon = config['epsilon_by_l'][0][1][0]
        for shape_index, copies in enumerate((1,8,32)):
            state = tf.tile(starts, [copies,1])
            for base, tiled in zip(base_values, adapter.log_prob_and_grad(state)):
                tf.debugging.assert_equal(tiled, tf.tile(base, [copies]+[1]*(base.shape.rank-1)))
            status = adapter.target_status_telemetry(state)
            for key, base in base_status.items():
                tf.debugging.assert_equal(status[key], tf.tile(base,[copies]+[1]*(base.shape.rank-1)))
            native = ReusableFullChainHMCRunner(adapter, state, FullChainHMCConfig(
                num_results=68, num_burnin_steps=0, step_size=epsilon, num_leapfrog_steps=25,
                seed=(20261003,2800), use_xla=True, target_scope='inference_validation',
                target_status_trace_policy='per_chain_step', capture_candidate_health=True),
                dynamic_num_leapfrog_steps=True)
            health_binding = SimpleNamespace(initial_active_state=state, config=execution)
            for repeat in range(3):
                seed = (20261003,2800+model_index*9+shape_index*3+repeat)
                before = time.monotonic()
                result = native.run(current_state=state, seed=seed, step_size=epsilon, num_leapfrog_steps=25)
                result.samples.numpy()
                elapsed = time.monotonic()-before
                failures = HMCCandidateExecutionBinding.health_failures(health_binding,state,result.samples,result.trace)
                row = dict(case=case,copies=copies,chains=4*copies,repeat=repeat,seed=seed,
                    enclosing_seconds=elapsed, seconds_per_bank=elapsed/copies,
                    runtime=result.metadata, sample_device=result.samples.device,
                    native_health_failures=failures, trace_count=native._runner.experimental_get_tracing_count(),
                    allocator=tf.config.experimental.get_memory_info('GPU:0'))
                rows.append(row)
                write(args.output/'calls.json', rows)
                assert not failures, row
                assert 'GPU:0' in result.samples.device
    write(args.output/'result.json',dict(status='native_capacity_checked',calls=rows,
        timing_scope='descriptive cold/warm calls; shape-dependent streams; no speed ranking',
        native_sampling=True,tuning_evidence=False,release_ready=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    args.source=args.source.resolve();args.output=args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{24,25,26,27})
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True,exist_ok=False)
    runner=args.output/'runner.py';runner.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(runner),*sys.argv[1:],'--worker']
    started=time.monotonic()
    record=dict(command=command,started_utc=datetime.now(timezone.utc).isoformat(),
        environment=sys.executable,resource='gpu',
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md')
    write(args.output/'manifest.json',record)
    with (args.output/'run.log').open('x') as log:
        try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=540).returncode
        except subprocess.TimeoutExpired:code=124
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    write(args.output/'receipt.json',record);print(json.dumps(record))
    return code


if __name__=='__main__':raise SystemExit(main())
