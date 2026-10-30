"""Bounded diagnostic of existing static/dynamic L execution; no tuning authority."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.inference import HMCCandidateExecutionConfig, PrecomputedMassArtifact, bind_hmc_candidate_set_execution
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig, HMCTuningCandidateSetController
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    config = json.loads(args.config.read_text())
    search = HMCControllerConfig.from_payload(config['search'])
    policy = search.replicated_acceptance_policy
    target = ValidationTarget(config['target'], config['parameters'], config['data'])
    starts = tf.constant(config['active_starts'], tf.float64)
    factor = tf.linalg.diag(tf.constant(config['geometry']['scale'], tf.float64))
    mass = PrecomputedMassArtifact(position=config['geometry']['center'], factor=factor,
        covariance=tf.matmul(factor, factor, transpose_b=True), adapter_signature=target.adapter_signature(),
        position_role='declared_diagnostic_preparation', covariance_source=config['provenance'])
    bindings = {}
    candidates = {}
    for dynamic in (False, True):
        execution = HMCCandidateExecutionConfig(measurement_num_results=policy.trial_num_results,
            verification_num_results=policy.trial_num_results, num_warmup_steps=policy.discarded_prefix,
            seed=(20261003, 2600), acceptance_policy=policy, target_status_trace_policy='per_chain_step',
            use_xla=True, chain_mode='batched', reuse_leapfrog_graphs=dynamic,
            chunk_max_results=policy.trial_num_results+policy.discarded_prefix)
        binding = bind_hmc_candidate_set_execution(adapter=target, initial_position=starts,
            start_coordinates='active', target_scope='inference_validation', config=execution,
            target_lineage={'model':target.target_id,'data':target.data,'parameters':target.parameters},
            source_paths=[str(args.source/'bayesfilter/testing/inference_validation/targets.py'),
                          str(args.source/'bayesfilter/testing/inference_validation/ssm_targets.py'),
                          target.value_score_capability().evidence_path],
            mass_artifact=mass, scope_id='native-graph-development', search_id='static-dynamic-L',
            epsilon_domain=tuple(config['epsilon_domain']), repair_factor=config['repair_factor'],
            max_repairs_per_family=config['max_repairs_per_family'])
        bindings[dynamic] = binding
        candidates[dynamic] = {c.leapfrog_steps:c for c in HMCTuningCandidateSetController(binding.scope, search).result().candidates}
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    manifest = dict(source_manifest_sha256=signature, configuration=config, gpu_uuid=args.gpu,
        gpu_inventory=inventory.splitlines(), memory_policy=memory, tensorflow_version=tf.__version__,
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(), jit_compile=True,
        CPU_affinity=sorted(os.sched_getaffinity(0)), seeds=[[20261003,2600+i] for i in range(8)],
        parity_criterion='exact samples and trace tensors, matching existing graph-reuse regression',
        scope='descriptive execution diagnostic; no delivery/release/posterior/default claim')
    write(args.output/'worker-manifest.json', manifest)
    rows = []
    for offset, steps in enumerate((3, 25)):
        for pair in range(4):
            seed = (20261003, 2600+4*offset+pair)
            outputs = {}
            order = (False, True) if pair % 2 == 0 else (True, False)
            for dynamic in order:
                binding = bindings[dynamic]
                before = time.monotonic()
                result = binding._run(candidates[dynamic][steps], starts, 68, seed)
                # Materialization includes outstanding device work in enclosing time.
                tf.debugging.assert_all_finite(result.samples, 'nonfinite sample')
                tf.reduce_sum(result.samples).numpy()
                row = dict(L=steps, pair=pair, dynamic=dynamic, seed=seed,
                    enclosing_seconds=time.monotonic()-before, runtime=result.metadata,
                    sample_device=result.samples.device)
                rows.append(row)
                outputs[dynamic] = result
                write(args.output/'calls.json', rows)
            tf.debugging.assert_equal(outputs[False].samples, outputs[True].samples)
            tf.nest.assert_same_structure(outputs[False].trace, outputs[True].trace)
            for left,right in zip(tf.nest.flatten(outputs[False].trace), tf.nest.flatten(outputs[True].trace)):
                tf.debugging.assert_equal(left, right)
                if left.dtype.is_floating:
                    tf.debugging.assert_all_finite(left, 'nonfinite trace')
    write(args.output/'result.json', dict(status='parity_passed', calls=rows,
        allocator=tf.config.experimental.get_memory_info('GPU:0'),
        graph_counts={str(k):[r._runner.experimental_get_tracing_count() for r in b._runners.values()]
                      for k,b in bindings.items()}, release_ready=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    for name in ('source','output','config'):
        setattr(args,name,getattr(args,name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2', BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{24,25,26,27})
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    frozen_runner=args.output/'runner.py'
    frozen_runner.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(frozen_runner),*sys.argv[1:],'--worker']
    record=dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
        environment=sys.executable, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md', resource='gpu',
        scope='existing execution option comparison; no numerical-policy promotion')
    write(args.output/'manifest.json',record)
    with (args.output/'run.log').open('x') as log:
        try:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=225).returncode
        except subprocess.TimeoutExpired:
            code=124
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    write(args.output/'receipt.json',record)
    print(json.dumps(record))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
