"""Check singleton likelihood graph reuse against the original live scalar probe."""
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
            scope='exact scalar likelihood graph diagnostic; no HMC',
            jit_compile=False, non_xla_reason='inspect existing eager target mutation probe')
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
    rows = []
    for case in ('nonlinear',):
        config = full_search_configuration(case, seed=DEVELOPMENT_SEEDS.get(case,(20261002,2500)), wall_seconds=1800)
        target = ValidationTarget(config['target'], config['parameters'], config['data'])
        g = config['geometry']
        if config['route'] == 'ordinary':
            factor = tf.linalg.diag(tf.constant(g['scale'], tf.float64))
            mass = PrecomputedMassArtifact(position=g['center'], factor=factor,
                covariance=tf.matmul(factor, factor, transpose_b=True), adapter_signature=target.adapter_signature(),
                position_role='diagnostic_preparation', covariance_source=config['provenance'])
            layers = [{'kind':'fixed_mass', 'artifact':mass.to_payload(include_arrays=True)}]
        else:
            layers = [{'kind':'frozen_transport', 'artifact':supplied_funnel_map(g['kind'])['transport_payload']}]
        active, _ = _rebuild_geometry(target, layers, 'inference_validation')
        state = tf.constant(config['active_starts'], tf.float64)
        before = time.monotonic()
        baseline = _probe(active, state, target_status=True)
        scalar_seconds = time.monotonic()-before
        from bayesfilter.testing.simple_nonlinear_generic_target_adapter_tf import simple_nonlinear_svd_ukf_log_likelihood_and_grad
        observations = target._ssm.data_tensor
        @tf.function(input_signature=[tf.TensorSpec([1,3],tf.float64),
            tf.TensorSpec(observations.shape,tf.float64)], autograph=False, jit_compile=False)
        def compiled(theta, data):
            return simple_nonlinear_svd_ukf_log_likelihood_and_grad(theta, observations=data)
        original = target._ssm.adapter.filter_log_likelihood_and_grad
        def replacement(theta):
            if tf.executing_eagerly():
                return compiled(theta, observations)
            return original(theta)
        target._ssm.adapter.filter_log_likelihood_and_grad = replacement
        timings=[]; outputs=[]
        for _ in range(3):
            before=time.monotonic()
            outputs.append(_probe(active,state,target_status=True))
            timings.append(time.monotonic()-before)
        exact=all(output==baseline for output in outputs)
        position=active.latent_to_position(state[:1])
        original_values=compiled(position,observations)
        changed_data=observations+tf.ones_like(observations)*tf.constant(.1,tf.float64)
        changed_values=compiled(position,changed_data)
        changed_reference=simple_nonlinear_svd_ukf_log_likelihood_and_grad(position,observations=changed_data)
        data_change_detected=any(_tensor_payload(x)!=_tensor_payload(y) for x,y in zip(original_values,changed_values))
        data_change_exact=all(_tensor_payload(x)==_tensor_payload(y) for x,y in zip(changed_values,changed_reference))
        rows.append(dict(case=case, baseline=baseline, graph_outputs=outputs,
            exact_value_score=exact, scalar_seconds=scalar_seconds, graph_seconds=timings,
            changed_data_detected=data_change_detected, changed_data_exact=data_change_exact,
            graph_trace_count=compiled.experimental_get_tracing_count(), configuration=config,
            tensor_devices=[x.device for x in changed_values]))
        write('result.json', dict(rows=rows, status='running', release_ready=False))
    check_source(args.source)
    write('result.json', dict(rows=rows, status='complete', all_exact=all(r['exact_value_score'] for r in rows),
        release_ready=False, default_promoted=False, scientific_claim='none; probe-output diagnostic only'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
