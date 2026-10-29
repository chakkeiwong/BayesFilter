#!/usr/bin/env python3
"""Bounded GPU qualification of M10 full-DCTR consumer mechanics, not inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/plans/surrogate-hmc-nonlinear-consumer-2026-09-20.md'
MARKER = '-m10-consumer-'
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
sys.path.insert(0, str(ROOT))


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    if hasattr(value, 'numpy'):
        return serial(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(serial(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(out, sources):
    for name, digest in sources.items():
        if sha(ROOT/name) != digest:
            raise ValueError('source changed before snapshot: '+name)
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    save(out/'source/sha256.json', sources)


def execute(out):
    manifest = json.loads((out/'run-manifest.json').read_text())
    data = json.loads(Path(manifest['data_file']).read_text())
    if sha(manifest['data_file']) != manifest['data_sha256']:
        raise ValueError('frozen M10 calibration data changed')
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_nonlinear_consumer import make_nonlinear_consumer
    from bayesfilter.inference.surrogate_hmc_dctr_selection import protection_controls
    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
    from bayesfilter.testing.surrogate_hmc_nonlinear_reference import PRIOR_SD
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig, bind_neural_force_hmc_tuning_runner,
        neural_force_proposal, neural_force_hmc_transition)
    devices = [device.name for device in tf.config.list_logical_devices('GPU')]
    if len(devices) != 1:
        raise ValueError('exactly one designated visible GPU required')
    manifest.update(memory_policy=memory, gpu_status=devices, tensorflow=tf.__version__,
                    tf32=True, jit_compile=True)
    save(out/'run-manifest.json', manifest)
    consumer = make_nonlinear_consumer(data, controls=protection_controls(1.), batch_size=4)
    save(out/'consumer.json', consumer.manifest)
    manifest['source_hashes'].update(consumer.manifest['fixture']['source_sha256'])
    manifest['seeds'] = dict(data=data['seed'], fixed_inputs=consumer.manifest['fixture']['seeds'],
                             transitions=[[20260920,16231],[20260920,16232]])
    snapshot(out, manifest['source_hashes'])
    save(out/'run-manifest.json', manifest)
    report = dict(status='RUNNING', classification='M10_consumer_mechanics_only',
                  checks={}, timings={}, retained_inference=False,
                  full_prior_posterior_reference=False, surrogate_hmc_tested=False)

    def timed(name, function):
        wall, cpu = time.monotonic(), time.process_time()
        result = function()
        # Materialize all returned tensors before charging the GPU interval.
        for value in tf.nest.flatten(result):
            if hasattr(value, 'numpy'):
                value.numpy()
        report['timings'][name] = dict(wall_seconds=time.monotonic()-wall,
                                      cpu_worker_seconds=time.process_time()-cpu)
        save(out/'report.json', report)
        print(name, report['timings'][name], flush=True)
        return result

    def equal(left, right):
        tf.nest.assert_same_structure(left, right)
        return all(bool(tf.reduce_all(a == b)) for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)))

    z = tf.constant([[0.,0.], [.15,-.1], [-.15,.1], [-.3,-.3]], tf.float64)
    baseline = timed('consumer_compile_first', lambda: consumer.evaluate_force(z))
    report['baseline'] = serial(dict(value=baseline[0], score=baseline[1], status=baseline[2], coordinates=z))
    report['original_force_healthy'] = bool(tf.reduce_all(baseline[2]['force_valid']))
    replay = timed('consumer_replay', lambda: consumer.evaluate_force(z))
    report['checks']['bitwise_replay'] = equal(baseline, replay)
    fixture = consumer.fixture

    @tf.function(input_signature=[tf.TensorSpec([4,2], tf.float32)], jit_compile=True, autograph=False)
    def direct(q):
        value, score, status = canonical_batch_fused_value_score(
            fixture.model, q, tf.broadcast_to(tf.eye(2)[None], [4,2,2]),
            fixture.initial_base, fixture.initial_covariances, fixture.noises,
            fixture.observations, **dict(fixture.controls))
        variance = tf.constant(PRIOR_SD**2, q.dtype)
        prior = -tf.reduce_sum(q*q, -1)/(2*variance)-tf.math.log(2*math.pi*variance)
        return value+prior, score-q/variance, status

    authority = timed('direct_canonical_engine', lambda: direct(tf.cast(z, tf.float32)))
    report['checks']['actual_call_chain_parity'] = (
        equal(baseline[:2], authority[:2]) and all(equal(baseline[2][key], authority[2][key])
            for key in ('value_valid','force_valid','value_failure_code','force_failure_code')))
    perm = tf.constant([2,0,3,1])
    permuted = timed('permutation', lambda: consumer.evaluate_force(tf.gather(z, perm)))
    report['checks']['permutation'] = equal(tf.nest.map_structure(lambda a: tf.gather(a, perm), baseline), permuted)
    changed = tf.tensor_scatter_nd_add(z, [[3,0]], tf.constant([.1], tf.float64))
    changed_output = timed('row_independence', lambda: consumer.evaluate_force(changed))
    report['checks']['row_independence'] = equal(tf.nest.map_structure(lambda a: a[:3], baseline),
                                                               tf.nest.map_structure(lambda a: a[:3], changed_output))
    endpoint = timed('endpoint_compile_first', lambda: consumer.endpoint.function(z))
    consumer.evaluate_force(changed)
    report['checks']['endpoint_independent'] = equal(endpoint, consumer.endpoint.function(z))
    field = consumer.force.function(z)
    report['checks']['field_sign_status'] = equal((field.force,field.force_valid),
                                                  (-tf.cast(baseline[1],tf.float64),baseline[2]['force_valid']))
    logp, score = consumer.evaluate_reference(z)
    report['checks']['finite_grid_endpoint_binding'] = equal(endpoint.potential, -logp)
    report['checks']['reference_force_binding'] = equal(consumer.reference_force.function(z).force, -score)
    outside = consumer.endpoint.function(tf.ones_like(z))
    report['checks']['unqualified_is_numerical_failure'] = bool(tf.reduce_all(
        outside.status['numerical_failure'] & ~outside.status['outside_declared_support']))
    binding = bind_neural_force_hmc_tuning_runner(force=consumer.force, target=consumer.endpoint,
                                                target_scope='M10-T2-N1024-calibration-mechanics-only')
    save(out/'binding.json', dict(binding.payload()))
    report['checks']['independent_typed_bindings'] = consumer.force.identity != consumer.endpoint.identity
    cfg = NeuralForceHMCConfig(step_size=.02, num_leapfrog_steps=2,
                              inverse_mass_diagonal=(1.,1.), dtype='float64')

    @tf.function(input_signature=[tf.TensorSpec([4,2],tf.float64)]*2, jit_compile=True, autograph=False)
    def proposal(q,p):
        return neural_force_proposal(q,p,consumer.force,cfg)

    momentum = tf.constant([[.2,-.1],[-.4,.3],[.6,-.5],[-.8,.7]],tf.float64)
    forward = timed('proposal_compile_first', lambda: proposal(z,momentum))
    backward = timed('proposal_reverse', lambda: proposal(forward.position,forward.momentum))
    error = float(tf.maximum(tf.reduce_max(tf.abs(backward.position-z)),
                            tf.reduce_max(tf.abs(backward.momentum-momentum))))
    tolerance = 32*2**-23*max(1.,float(tf.reduce_max(tf.abs(z))),float(tf.reduce_max(tf.abs(momentum))))
    report.update(reversal_absolute_error=error,reversal_tolerance=tolerance)
    report['checks']['proposal_involution'] = error <= tolerance

    @tf.function(input_signature=[tf.TensorSpec([4,2],tf.float64),tf.TensorSpec([4],tf.float64),
                                  tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
    def transition(q,potential,seed):
        return neural_force_hmc_transition(q,potential,consumer.force,consumer.endpoint,cfg,seed)

    first = timed('transition_compile_first', lambda: transition(z,endpoint.potential,tf.constant([20260920,16231])))
    second = timed('transition_steady', lambda: transition(first.position,first.potential,tf.constant([20260920,16232])))
    trace = second.trace
    difference = trace.final_potential+trace.final_kinetic-trace.initial_potential-trace.initial_kinetic
    report['checks']['per_chain_energy'] = trace.delta_h.shape == (4,) and bool(tf.reduce_all(tf.abs(trace.delta_h-difference)<1e-10))
    report['checks']['cache_matches_endpoint'] = bool(tf.reduce_all(tf.abs(second.potential-consumer.endpoint.function(second.position).potential)<1e-10))
    report['checks']['finite_states'] = all(bool(tf.reduce_all(tf.math.is_finite(step.position))) for step in (first,second))
    report['checks']['endpoint_transition_health'] = all(bool(tf.reduce_all(
        tf.math.is_finite(step.trace.delta_h) & ~step.trace.endpoint_status['numerical_failure'])) for step in (first,second))
    report['transitions'] = serial([first.trace._asdict(),second.trace._asdict()])
    report['finite_grid_logp'] = serial(logp)
    report['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    report['qualified'] = all(report['checks'].values())
    report['status'] = 'QUALIFIED_CONSUMER_MECHANICS' if report['qualified'] else 'QUALIFICATION_FAILED'
    report['field_status'] = 'HEALTHY_AT_PROBES' if report['original_force_healthy'] else 'REPAIR_REQUIRED'
    save(out/'report.json', report)
    return 0 if report['qualified'] else 2


def main():
    start = time.monotonic()
    parent_cpu = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return execute(out)
    if MARKER not in out.name or '-tests-' in out.name:
        parser.error('fresh M10 consumer accounting marker required')
    from bayesfilter.inference.surrogate_hmc_program import m10_reference_state
    state = m10_reference_state(ROOT)
    if not state['qualified']:
        raise ValueError('qualified, unchanged M10 local reference required')
    totals = dict(gpu_seconds=0.,cpu_worker_seconds=0.)
    family = dict(totals)
    stage = dict(totals)
    attempts = 0
    for path in (ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'):
        entry = json.loads(path.read_text())
        charge = entry['charged_this_run'] if 'charged_this_run' in entry else entry['charged']
        for key in totals:
            value = float(charge[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid historical resource charge')
            totals[key] += value
            if '-m10-' in path.parent.name:
                family[key] += value
            if MARKER in path.parent.name and '-tests-' not in path.parent.name:
                stage[key] += value
        if '-m10-' not in path.parent.name:
            family['gpu_seconds'] += sum(float(job.get('gpu_seconds', 0.))
                for job in entry.get('jobs', []) if any('surrogate_hmc_nonlinear' in str(arg)
                    for arg in job.get('command', [])))
        if MARKER in path.parent.name and '-tests-' not in path.parent.name:
            attempts += 1
            old_report = path.parent/'report.json'
            if old_report.exists() and json.loads(old_report.read_text()).get('status') in (
                    'QUALIFIED_CONSUMER_MECHANICS','QUALIFICATION_FAILED'):
                raise ValueError('completed mechanics requires interpretation, not automatic rerun')
    wall_budget = min(600-stage['gpu_seconds'],7200-family['gpu_seconds'],226800-totals['gpu_seconds'])
    cpu_budget = min(450-stage['cpu_worker_seconds'],state['family_cpu_remaining'],54000-totals['cpu_worker_seconds'])
    if attempts >= 2 or min(cpu_budget,wall_budget) < 2:
        raise ValueError('M10 consumer attempt/resource allocation exhausted')
    reference = Path(state['qualified']['report']).parent
    reference_report = json.loads((reference/'report.json').read_text())
    partition = reference_report['partitions']['calibration-nominal']
    data_file = Path(partition['data_file']).resolve()
    if sha(data_file) != partition['data_sha256']:
        raise ValueError('calibration data hash mismatch')
    out.mkdir(parents=True,exist_ok=False)
    source_names = [PLAN,'scripts/qualify_surrogate_hmc_nonlinear_consumer.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_consumer.py',
        'tests/test_surrogate_hmc_nonlinear_consumer.py',
        'bayesfilter/inference/neural_force_hmc.py',
        'bayesfilter/inference/surrogate_hmc_dctr_selection.py']
    sources = {name:sha(ROOT/name) for name in source_names}
    snapshot(out,sources)
    manifest = dict(command=[sys.executable,*sys.argv],environment=sys.prefix,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan=PLAN,result_file=str(out/'report.json'),source_hashes=sources,
        data_file=str(data_file),data_sha256=sha(data_file),data_version='frozen M10 calibration nominal, once rounded FP32',
        parent_reference=str(reference),parent_manifest_sha256=sha(reference/'run-manifest.json'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),memory_policy='not initialized',
        gpu_status='pending trusted GPU probe',cpu_budget=cpu_budget,wall_budget=wall_budget)
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',dict(status='RUNNING',checks={},retained_inference=False))
    # Leave a small allowance for parent bookkeeping and preserving final evidence.
    child_cpu_budget = cpu_budget-(time.process_time()-parent_cpu)-2.
    child_wall_budget = wall_budget-(time.monotonic()-start)-1.
    if min(child_cpu_budget, child_wall_budget) < 1:
        raise ValueError('insufficient allocation after launch preparation')
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU,(max(1,math.floor(child_cpu_budget)),max(2,math.ceil(child_cpu_budget))))
    try:
        with (out/'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit)
            try:
                code = process.wait(timeout=child_wall_budget)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
    except Exception as exc:
        code = 125
        save(out/'launch-error.json',dict(error=f'{type(exc).__name__}: {exc}'))
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu
    wall = time.monotonic()-start
    report = json.loads((out/'report.json').read_text())
    manifest = json.loads((out/'run-manifest.json').read_text())
    if code != 0 and report['status'] == 'RUNNING':
        report['status'] = 'STOPPED_RESOURCE_LIMIT' if code in (124,-24,-9) else 'EXECUTION_FAILED'
    if any(sha(ROOT/name) != digest for name,digest in manifest['source_hashes'].items()):
        report['status'], code = 'SOURCE_DRIFT', 2
    report.update(exit_code=code,wall_seconds=wall,cpu_worker_seconds=cpu)
    manifest.update(status=report['status'],exit_code=code,wall_seconds=wall,cpu_worker_seconds=cpu)
    save(out/'report.json',report)
    save(out/'run-manifest.json',manifest)
    save(out/'accounting.json',dict(charged_this_run=dict(gpu_seconds=wall,cpu_worker_seconds=cpu)))
    print(report['status'],'CPU',round(cpu,3),'GPU wall',round(wall,3),flush=True)
    return code if code >= 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
