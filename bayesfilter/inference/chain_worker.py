"""Private fresh-process worker for HMCChainExecutor; not an application launcher."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path


def _memory(tf, gpu):
    rss = None
    status = Path('/proc/self/status')
    if status.exists():
        fields = dict(line.split(':', 1) for line in status.read_text().splitlines() if ':' in line)
        rss = int(fields['VmRSS'].split()[0]) * 1024
    scale = 1 if sys.platform == 'darwin' else 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'host_rss_bytes': rss, 'host_peak_rss_bytes': usage.ru_maxrss * scale,
            'process_cpu_seconds': usage.ru_utime + usage.ru_stime,
            'gpu_allocator': tf.config.experimental.get_memory_info('GPU:0') if gpu else None}


def _save_tensor(tf, directory, name, tensor):
    tensor = tf.convert_to_tensor(tensor)
    # Serialization is a CPU-only artifact boundary, outside the GPU kernel.
    with tf.device('/CPU:0'):
        data = tf.io.serialize_tensor(tensor).numpy()
    path = directory / f'{name}.tensor'
    with path.open('xb') as stream:
        stream.write(data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'dtype': tensor.dtype.name, 'shape': tensor.shape.as_list()}


def _save_trace(tf, directory, tree, prefix='trace'):
    if isinstance(tree, dict):
        return {key: _save_trace(tf, directory, item, prefix + '-' + key) for key, item in tree.items()}
    return _save_tensor(tf, directory, prefix, tree)


def _status_health(tf, telemetry, shape):
    from bayesfilter.inference.hmc import _target_status_telemetry_diagnostics

    if not telemetry:
        return {'available': False, 'valid': None}
    for name in ('status_code', 'valid_pre_regularized_score', 'floor_count_value'):
        if telemetry[name].shape != shape:
            raise ValueError(f'{name} must have one status per chain/transition')
    diagnostics = _target_status_telemetry_diagnostics(telemetry)
    return {'available': True, 'valid': bool(diagnostics['all_status_valid'].numpy()),
            'nonvalid_count': int(diagnostics['status_nonvalid_count'].numpy()),
            'floor_count_total': int(diagnostics['floor_count_total'].numpy())}


class _CheckedBatchTarget:
    """Preserve the adapter's authority while checking both value and HMC calls."""

    def __init__(self, adapter, dtype):
        self.adapter, self.dtype = adapter, dtype

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def log_prob_and_grad(self, position):
        from bayesfilter.inference.batched_value_score import (
            _validate_value_score_shapes,
        )

        value, score = self.adapter.log_prob_and_grad(position)
        _validate_value_score_shapes(theta=position, value=value, score=score)
        if value.dtype != self.dtype or score.dtype != self.dtype:
            raise TypeError('target value and score must preserve the configured dtype')
        return value, score


def serve(request, reply):
    if request['layout']['pin_cpu']:
        os.sched_setaffinity(0, request['cpu_set'])
        if sorted(os.sched_getaffinity(0)) != request['cpu_set']:
            raise RuntimeError('worker CPU affinity did not match the requested budget')
    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    layout, kernel = request['layout'], request['kernel']
    gpu = layout['device'] == 'GPU'
    growth = configure_tensorflow_gpu_memory_growth(tf, require_gpu=gpu)
    if not gpu and tf.config.list_physical_devices('GPU'):
        raise RuntimeError('CPU worker must hide all GPUs before import')
    if gpu and len(tf.config.list_physical_devices('GPU')) != 1:
        raise RuntimeError('each GPU worker must see exactly one selected GPU')
    tf.config.threading.set_intra_op_parallelism_threads(layout['cores_per_worker'])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(layout['tf32'])
    tf.config.set_soft_device_placement(False)
    module_name, factory_name = request['target']['factory'].split(':')
    module = importlib.import_module(module_name)
    factory = module
    for name in factory_name.split('.'):
        factory = getattr(factory, name)
    device = '/GPU:0' if gpu else '/CPU:0'
    from bayesfilter.inference.hmc import (
        FullChainHMCConfig,
        ReusableFullChainHMCRunner,
        stable_adapter_signature,
    )
    from bayesfilter.inference.hmc_verification import TARGET_STATUS_TELEMETRY_FIELDS

    directory = Path(request['output_dir'])
    dtype = tf.as_dtype(kernel['dtype'])
    with tf.device(device):
        adapter = factory(**request['target']['kwargs'])
        if not callable(getattr(adapter, 'log_prob_and_grad', None)):
            raise TypeError('a native batched log_prob_and_grad adapter is required')
        adapter = _CheckedBatchTarget(adapter, dtype)
        signature = stable_adapter_signature(adapter)
        status_policy = kernel['target_status_trace_policy']
        if status_policy == 'auto':
            status_policy = ('per_chain_step' if callable(getattr(adapter, 'target_status_telemetry', None))
                             else 'none')
        config = FullChainHMCConfig(num_results=kernel['num_results'], num_burnin_steps=0,
            step_size=kernel['step_size'], num_leapfrog_steps=kernel['num_leapfrog_steps'],
            seed=kernel['seed'], use_xla=kernel['jit_compile'], target_scope=kernel['target_scope'],
            target_status_trace_policy=status_policy, capture_candidate_health=True)

        def value_score(position):
            value, score = adapter.log_prob_and_grad(position)
            status = adapter.target_status_telemetry(position) if status_policy == 'per_chain_step' else {}
            status = {key: status[key] for key in TARGET_STATUS_TELEMETRY_FIELDS if key in status}
            return value, score, status

        owners = []
        compiled = {}
        for part in request['partition']:
            state = tf.constant(request['initial_state'][part['start']:part['stop']], dtype)
            batch_size = part['stop'] - part['start']
            if batch_size not in compiled:
                runner = ReusableFullChainHMCRunner(adapter, state, config, state_dtype=dtype)
                evaluator = tf.function(value_score, input_signature=[tf.TensorSpec(state.shape, dtype)],
                                        jit_compile=kernel['jit_compile'], autograph=False)
                compiled[batch_size] = (runner, evaluator)
            runner, evaluator = compiled[batch_size]
            owners.append({'part': part, 'state': state, 'runner': runner, 'evaluator': evaluator})
        probe = tf.reduce_sum(tf.ones([2], dtype))
        if layout['device'] not in probe.device:
            raise RuntimeError('worker executed on the wrong device')
        factory_path = Path(module.__file__)
        cpu_info = Path('/proc/cpuinfo')
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), platform.processor()) if cpu_info.exists() else platform.processor()
        reply({'kind': 'ready', 'pid': os.getpid(), 'worker': request['worker'],
               'tensorflow': tf.__version__, 'tensorflow_probability': tfp.__version__,
               'python': sys.version, 'platform': platform.platform(),
               'python_executable': sys.executable, 'cpu_model': cpu_model,
               'execution_source_sha256': {
                   name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ('chain_worker.py', 'chain_execution.py', 'hmc.py', 'batched_value_score.py')},
               'target_signature': signature, 'factory_path': str(factory_path),
               'factory_sha256': hashlib.sha256(factory_path.read_bytes()).hexdigest(),
               'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
               'intra_op_threads': tf.config.threading.get_intra_op_parallelism_threads(),
               'inter_op_threads': tf.config.threading.get_inter_op_parallelism_threads(),
               'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'device': probe.device,
               'jit_compile': kernel['jit_compile'], 'target_status_trace_policy': status_policy,
               'gpu_intentionally_hidden': not gpu,
               'tf32': tf.config.experimental.tensor_float_32_execution_enabled(),
               'gpu_memory_policy': growth, 'memory': _memory(tf, gpu)})

        for line in sys.stdin:
            command = json.loads(line)
            if command['operation'] not in ('hmc', 'value_score'):
                raise ValueError('unknown worker operation')
            if stable_adapter_signature(adapter) != signature:
                raise RuntimeError('target identity changed during worker reuse')
            call_dir = directory / f'call-{command["request_id"]:05d}'
            call_dir.mkdir(exist_ok=False)
            if gpu:
                tf.config.experimental.reset_memory_stats('GPU:0')
            started = time.perf_counter()
            shards = []
            for owner in owners:
                part = owner['part']
                state = owner['state'] if command['state'] is None else tf.constant(
                    command['state'][part['start']:part['stop']], dtype)
                begun = time.perf_counter()
                shard_dir = call_dir / f'batch-{part["batch"]:03d}'
                shard_dir.mkdir()
                if command['operation'] == 'value_score':
                    value, score, telemetry = owner['evaluator'](state)
                    values, scores = value.numpy().tolist(), score.numpy().tolist()
                    elapsed = time.perf_counter() - begun
                    finite = bool((tf.reduce_all(tf.math.is_finite(value)) &
                                   tf.reduce_all(tf.math.is_finite(score))).numpy())
                    status = _status_health(tf, telemetry, value.shape)
                    record = {'values': values if finite else None, 'scores': scores if finite else None,
                              'health': {'finite': finite, 'target_status': status,
                                         'valid': finite and status['valid'] is not False},
                              'tensors': {'value': _save_tensor(tf, shard_dir, 'value', value),
                                          'score': _save_tensor(tf, shard_dir, 'score', score)},
                              'target_status': _save_trace(tf, shard_dir, telemetry, 'target-status'),
                              'compute_device': value.device,
                              'trace_count': owner['evaluator'].experimental_get_tracing_count(),
                              'xla_must_compile': owner['evaluator'].get_concrete_function().function_def.attr['_XlaMustCompile'].b}
                else:
                    seed = tf.random.experimental.stateless_fold_in(command['seed'], command['call_index'])
                    seed = tf.random.experimental.stateless_fold_in(seed, part['start'])
                    result = owner['runner'].run(current_state=state, seed=seed,
                                                 step_size=command['step_size'])
                    # Synchronize before stopping the numerical-call timer.
                    last = result.samples[-1]
                    last_values = last.numpy().tolist()
                    elapsed = time.perf_counter() - begun
                    finite = tf.reduce_all(tf.math.is_finite(result.samples))
                    for name in ('target_log_prob', 'log_accept_ratio', 'proposed_target_log_prob',
                                 'proposed_state', 'initial_momentum', 'final_momentum'):
                        if name in result.trace:
                            finite &= tf.reduce_all(tf.math.is_finite(result.trace[name]))
                    finite &= tf.reduce_all(result.trace['target_score_finite'])
                    finite = bool(finite.numpy())
                    moved = tf.reduce_any(result.samples != state[None, ...], axis=[0, 2]).numpy().tolist()
                    status = {name: _status_health(tf, result.trace.get(name, {}), result.samples.shape[:2])
                              for name in ('target_status_telemetry', 'proposed_target_status_telemetry')}
                    divergence = (bool(tf.reduce_any(result.trace['divergence']).numpy())
                                  if 'divergence' in result.trace else None)
                    health = {'finite': finite, 'chain_moved': moved, 'native_divergence': divergence,
                              'status': status, 'valid': finite and divergence is not True
                              and all(s['valid'] is not False for s in status.values())}
                    if 'is_accepted' in result.trace:
                        health['acceptance_rate'] = tf.reduce_mean(
                            tf.cast(result.trace['is_accepted'], tf.float64), axis=0).numpy().tolist()
                    if finite:
                        health['maximum_absolute_log_accept_ratio'] = float(tf.reduce_max(
                            tf.abs(result.trace['log_accept_ratio'])).numpy())
                    record = {'health': health, 'last_state': last_values if finite else None,
                              'seed': seed.numpy().tolist(), 'compute_device': result.samples.device,
                              'tensors': {'samples': _save_tensor(tf, shard_dir, 'samples', result.samples)},
                              'trace': _save_trace(tf, shard_dir, result.trace),
                              'trace_count': owner['runner']._runner.experimental_get_tracing_count(),
                              'xla_must_compile': owner['runner']._runner.get_concrete_function().function_def.attr['_XlaMustCompile'].b}
                    owner['state'] = last
                if layout['device'] not in record['compute_device']:
                    raise RuntimeError('numerical output has wrong placement')
                shards.append({**part, **record, 'call_seconds': elapsed,
                               'with_artifacts_seconds': time.perf_counter() - begun})
            reply({'kind': 'result', 'request_id': command['request_id'],
                   'pid': os.getpid(), 'shards': shards,
                   'worker_seconds': time.perf_counter() - started, 'memory': _memory(tf, gpu)})


def main():
    channel = sys.stdout

    def reply(value):
        channel.write(json.dumps(value, allow_nan=False) + '\n')
        channel.flush()

    try:
        request = json.loads(Path(sys.argv[1]).read_text())
        # Application prints belong in worker.log, never in the control channel.
        with contextlib.redirect_stdout(sys.stderr):
            serve(request, reply)
    except BaseException as error:  # noqa: BLE001 -- report every worker failure before exit
        traceback.print_exc(file=sys.stderr)
        reply({'kind': 'error', 'error': repr(error)})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
