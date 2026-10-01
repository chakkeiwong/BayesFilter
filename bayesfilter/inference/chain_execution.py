"""Persistent process execution of native chain batches using the shared HMC runner.

This module is framework-free. It schedules complete compiled calls, never
maps a scalar filter over chain rows, and does not issue tuning authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class HMCChainExecutionError(RuntimeError):
    """An execution worker failed; its logs and partial artifacts are retained."""


def _positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _seed(seed):
    seed = tuple(seed)
    if len(seed) != 2 or any(type(x) is not int or not -(2**31) <= x < 2**31 for x in seed):
        raise ValueError('seed must contain two signed int32 integers')
    return seed


def _json_write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _states(value):
    # Materialization is only an IPC/input-validation boundary.
    if hasattr(value, 'numpy'):
        value = value.numpy().tolist()
    rows = [list(row) for row in value]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('initial/current state must have shape [chains, parameters]')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           for row in rows for x in row):
        raise ValueError('state must contain finite real numbers')
    return rows


@dataclass(frozen=True)
class HMCChainTarget:
    """An importable ``module:factory`` returning a native batched target adapter.

    Factories are imported only after worker device/thread setup. Pass data
    paths or JSON parameters, not live TensorFlow objects or pickled closures.
    """

    factory: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        parts = self.factory.split(':')
        if len(parts) != 2 or not all(parts) or '<locals>' in self.factory:
            raise ValueError('factory must be an importable module:callable')
        object.__setattr__(self, 'kwargs', json.loads(json.dumps(dict(self.kwargs), allow_nan=False)))


@dataclass(frozen=True)
class HMCChainKernel:
    """Fixed-kernel mechanics; use the existing public tuner for qualification."""

    num_results: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int] = (20261001, 17)
    dtype: str = 'float64'
    jit_compile: bool = True
    non_xla_reason: str | None = None
    target_scope: str | None = None
    target_status_trace_policy: str = 'auto'

    def __post_init__(self):
        _positive(self.num_results, 'num_results')
        _positive(self.num_leapfrog_steps, 'num_leapfrog_steps')
        if isinstance(self.step_size, bool) or not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError('step_size must be positive and finite')
        if self.dtype not in ('float32', 'float64'):
            raise ValueError('dtype must be float32 or float64')
        if type(self.jit_compile) is not bool:
            raise TypeError('jit_compile must be boolean')
        if not self.jit_compile and not str(self.non_xla_reason or '').strip():
            raise ValueError('non-XLA execution requires a diagnostic non_xla_reason')
        if self.target_status_trace_policy not in ('auto', 'none', 'per_chain_step'):
            raise ValueError('invalid target_status_trace_policy')
        object.__setattr__(self, 'seed', _seed(self.seed))


@dataclass(frozen=True)
class HMCChainLayout:
    """Chain batching and process resources; cores are logical CPU affinity IDs.

    A worker shares its core budget across its batch. For a per-chain budget,
    use ``one_process_per_chain``. GPU processes use verified memory growth.
    """

    chains_per_batch: int
    workers: int = 1
    cores_per_worker: int = 1
    device: str = 'GPU'
    gpu_devices: tuple[str, ...] = ()
    pin_cpu: bool = True
    tf32: bool = True

    def __post_init__(self):
        for name in ('chains_per_batch', 'workers', 'cores_per_worker'):
            _positive(getattr(self, name), name)
        if self.device not in ('CPU', 'GPU'):
            raise ValueError('device must be CPU or GPU')
        if type(self.pin_cpu) is not bool or type(self.tf32) is not bool:
            raise TypeError('pin_cpu and tf32 must be boolean')
        devices = tuple(self.gpu_devices)
        if devices and (self.device != 'GPU' or len(devices) not in (1, self.workers)):
            raise ValueError('gpu_devices requires GPU and one ID or one ID per worker')
        if any(not isinstance(x, str) or not x or ',' in x or x == '-1' for x in devices):
            raise ValueError('each GPU ID must identify one visible device')
        object.__setattr__(self, 'gpu_devices', devices)

    @classmethod
    def one_process_per_chain(cls, chains, *, cores_per_chain=1, **kwargs):
        return cls(chains_per_batch=1, workers=chains,
                   cores_per_worker=cores_per_chain, **kwargs)

    def partition(self, chain_count):
        _positive(chain_count, 'chain_count')
        batches = [(start, min(start + self.chains_per_batch, chain_count))
                   for start in range(0, chain_count, self.chains_per_batch)]
        if self.workers > len(batches):
            raise ValueError('workers cannot exceed the number of chain batches')
        return tuple({'batch': i, 'start': start, 'stop': stop, 'worker': i % self.workers}
                     for i, (start, stop) in enumerate(batches))


@dataclass(frozen=True)
class HMCChainResult:
    """Ordered tensor shards and per-worker provenance for one complete call."""

    operation: str
    shards: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    def load(self, name='samples'):
        """Load completed tensor shards in chain order at a host output boundary."""
        os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
        import tensorflow as tf

        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )
        configure_tensorflow_gpu_memory_growth(tf, require_gpu=False)
        tensors = []
        with tf.device('/CPU:0'):
            for shard in self.shards:
                item = shard['tensors'][name]
                data = Path(item['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise HMCChainExecutionError(f'changed tensor artifact: {item["path"]}')
                tensor = tf.io.parse_tensor(data, out_type=tf.as_dtype(item['dtype']))
                if tensor.shape.as_list() != item['shape']:
                    raise HMCChainExecutionError('tensor shape does not match its record')
                tensors.append(tensor)
            return tf.concat(tensors, axis=1 if name == 'samples' else 0)


class HMCChainExecutor:
    """Own persistent workers and compiled native chain-batch runners.

    ``run()`` continues each batch from its last state. Supplying current_state
    restarts all batches explicitly. Explicit seeds replay independently of
    call index; otherwise the call index is folded into the configured seed.
    The same batch partition has the same streams regardless of worker count.
    This class does not discard warmup or declare posterior convergence.
    """

    def __init__(self, target: HMCChainTarget, initial_state, kernel: HMCChainKernel,
                 layout: HMCChainLayout, *, output_dir=None, timeout_seconds=300.):
        if not isinstance(target, HMCChainTarget) or not isinstance(kernel, HMCChainKernel):
            raise TypeError('target and kernel must use the shared configuration types')
        if not isinstance(layout, HMCChainLayout):
            raise TypeError('layout must be HMCChainLayout')
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be finite and positive')
        self.initial_state = _states(initial_state)
        self.partition = layout.partition(len(self.initial_state))
        self.target, self.kernel, self.layout = target, kernel, layout
        self.timeout_seconds = float(timeout_seconds)
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
        if layout.pin_cpu and not hasattr(os, 'sched_setaffinity'):
            raise ValueError('CPU affinity unavailable; explicitly select pin_cpu=False')
        if layout.workers * layout.cores_per_worker > len(available):
            raise ValueError('worker core budgets exceed the available CPU affinity')
        self.cpu_sets = [available[i * layout.cores_per_worker:(i + 1) * layout.cores_per_worker]
                         for i in range(layout.workers)]
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix='bayesfilter-chains-'))
        if output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        self.output_dir = self.output_dir.resolve()
        self._processes, self._logs = [], []
        self._lock = threading.Lock()
        self._call_index = 0
        self._closed = False
        self._request_id = 0
        self.startup = ()
        started = time.perf_counter()
        try:
            devices = layout.gpu_devices
            selection = None
            if layout.device == 'GPU' and not devices:
                from bayesfilter.runtime.display_gpu_policy import (
                    probe_inventory,
                    select_gpu,
                )
                selection = select_gpu(probe_inventory())
                if selection['selected'] is None:
                    raise HMCChainExecutionError('no eligible GPU is available')
                if selection['selected']['is_display']:
                    raise HMCChainExecutionError(
                        'no eligible non-display GPU; select gpu_devices explicitly for a display GPU')
                devices = (selection['selected']['uuid'],)
            for worker in range(layout.workers):
                directory = self.output_dir / f'worker-{worker:03d}'
                directory.mkdir()
                device = '-1' if layout.device == 'CPU' else devices[0 if len(devices) == 1 else worker]
                request = {'target': asdict(target), 'kernel': asdict(kernel), 'layout': asdict(layout),
                           'initial_state': self.initial_state, 'worker': worker,
                           'partition': [p for p in self.partition if p['worker'] == worker],
                           'cpu_set': self.cpu_sets[worker], 'output_dir': str(directory)}
                _json_write(directory / 'request.json', request)
                environment = dict(os.environ)
                environment.update(CUDA_VISIBLE_DEVICES=device, TF_FORCE_GPU_ALLOW_GROWTH='true',
                    TF_NUM_INTRAOP_THREADS=str(layout.cores_per_worker), TF_NUM_INTEROP_THREADS='1',
                    OMP_NUM_THREADS=str(layout.cores_per_worker), OPENBLAS_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', BAYESFILTER_PRELOAD_CUSTOM_OP='0', PYTHONUNBUFFERED='1')
                # Preserve caller module discovery without importing its factory in the parent.
                environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(
                    [os.getcwd(), *[str(Path(p).resolve()) for p in sys.path if p],
                     environment.get('PYTHONPATH', '')]))
                log = (directory / 'worker.log').open('wb')
                self._logs.append(log)
                process = subprocess.Popen([sys.executable, '-m',
                    'bayesfilter.inference.chain_worker', str(directory / 'request.json')],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                    env=environment, start_new_session=True)
                self._processes.append(process)
            self.startup = tuple(self._receive('ready'))
            if len({(r['target_signature'], r['factory_sha256']) for r in self.startup}) != 1:
                raise HMCChainExecutionError('workers constructed different target identities')
            if any(r['execution_source_sha256'] != self.startup[0]['execution_source_sha256']
                   for r in self.startup):
                raise HMCChainExecutionError('execution source changed during worker startup')
            self.startup_seconds = time.perf_counter() - started
            _json_write(self.output_dir / 'manifest.json', {
                'schema': 'bayesfilter.hmc_chain_execution.v1', 'target': asdict(target),
                'kernel': asdict(kernel), 'layout': asdict(layout), 'partition': self.partition,
                'workers': self.startup, 'startup_seconds': self.startup_seconds,
                'gpu_selection': selection, 'seed_policy': 'fold_call_index_then_batch_start_v1',
                'role': 'fixed_kernel_execution; no tuning or posterior admission'})
        except BaseException as error:
            try:
                _json_write(self.output_dir / 'startup-failure.json', {'error': repr(error)})
            finally:
                self.close()
            raise

    def _send(self, data, deadline):
        # Nonblocking writes keep large input banks subject to the same deadline.
        with selectors.DefaultSelector() as selector:
            for process in self._processes:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, memoryview(data))
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HMCChainExecutionError('worker input deadline exceeded')
                for key, _ in selector.select(min(remaining, .25)):
                    try:
                        count = os.write(key.fd, key.data)
                    except BlockingIOError:
                        continue
                    if count == len(key.data):
                        selector.unregister(key.fileobj)
                    else:
                        selector.modify(key.fileobj, selectors.EVENT_WRITE, key.data[count:])

    def _receive(self, expected, deadline=None):
        deadline = time.monotonic() + self.timeout_seconds if deadline is None else deadline
        replies, buffers = {}, {i: b'' for i in range(len(self._processes))}
        with selectors.DefaultSelector() as selector:
            for i, process in enumerate(self._processes):
                selector.register(process.stdout, selectors.EVENT_READ, i)
            while len(replies) < len(self._processes):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HMCChainExecutionError(f'worker deadline exceeded; logs: {self.output_dir}')
                for key, _ in selector.select(min(remaining, .25)):
                    index = key.data
                    data = os.read(key.fd, 65536)
                    if not data:
                        raise HMCChainExecutionError(f'worker {index} exited before {expected}; logs: {self.output_dir}')
                    buffers[index] += data
                    if len(buffers[index]) > 16 * 1024**2:
                        raise HMCChainExecutionError('worker response exceeds metadata size limit')
                    if b'\n' in buffers[index]:
                        line, tail = buffers[index].split(b'\n', 1)
                        reply = json.loads(line)
                        if tail or reply.get('kind') != expected:
                            raise HMCChainExecutionError(f'worker {index}: {reply}; logs: {self.output_dir}')
                        if expected == 'result' and reply.get('request_id') != self._request_id:
                            raise HMCChainExecutionError('worker reply belongs to a different call')
                        replies[index] = reply
                        selector.unregister(key.fileobj)
        return [replies[i] for i in range(len(self._processes))]

    def _call(self, operation, current_state, seed, step_size):
        if not self._lock.acquire(blocking=False):
            raise HMCChainExecutionError('concurrent executor calls are unsupported')
        try:
            if self._closed:
                raise HMCChainExecutionError('executor is closed')
            states = None if current_state is None else _states(current_state)
            if states is not None and (len(states), len(states[0])) != (len(self.initial_state), len(self.initial_state[0])):
                raise ValueError('current_state shape differs from the initial chain bank')
            actual_seed = self.kernel.seed if seed is None else _seed(seed)
            step = self.kernel.step_size if step_size is None else float(step_size)
            if not math.isfinite(step) or step <= 0:
                raise ValueError('step_size must be finite and positive')
            self._request_id += 1
            request = {'operation': operation, 'state': states, 'seed': actual_seed,
                       'call_index': self._call_index if seed is None else 0,
                       'step_size': step, 'request_id': self._request_id}
            started = time.perf_counter()
            try:
                data = (json.dumps(request, allow_nan=False) + '\n').encode()
                deadline = time.monotonic() + self.timeout_seconds
                self._send(data, deadline)
                replies = self._receive('result', deadline)
                shards = tuple(sorted((shard for reply in replies for shard in reply['shards']),
                                      key=lambda shard: shard['start']))
                if [(s['start'], s['stop'], s['worker']) for s in shards] != [
                        (p['start'], p['stop'], p['worker']) for p in self.partition]:
                    raise HMCChainExecutionError('incomplete or reordered worker chain coverage')
                metadata = {'request_id': self._request_id, 'operation': operation,
                            'end_to_end_seconds': time.perf_counter() - started,
                            'startup_seconds': self.startup_seconds, 'workers': replies,
                            'all_healthy': all(s['health']['valid'] for s in shards),
                            'target_signature': self.startup[0]['target_signature'],
                            'seed': actual_seed, 'call_index': request['call_index'],
                            'step_size': step, 'kernel': asdict(self.kernel),
                            'layout': asdict(self.layout)}
                _json_write(self.output_dir / f'call-{self._request_id:05d}.json', metadata)
                if not metadata['all_healthy']:
                    raise HMCChainExecutionError(
                        f'invalid target or HMC health; tensors and diagnostics: {self.output_dir}')
            except BaseException as error:
                try:
                    _json_write(self.output_dir / f'failure-{self._request_id:05d}.json',
                                {'error': repr(error), 'request': request})
                finally:
                    self.close()
                raise
            self._call_index += operation == 'hmc'
            return HMCChainResult(operation, shards, metadata)
        finally:
            self._lock.release()

    def run(self, *, current_state=None, seed=None, step_size=None):
        return self._call('hmc', current_state, seed, step_size)

    def value_and_score(self, state=None):
        return self._call('value_score', self.initial_state if state is None else state, None, None)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for process in self._processes:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass  # A failed worker may already have closed its read end.
        deadline = time.monotonic() + 3.
        for process in self._processes:
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in self._processes:
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if process.stdout:
                process.stdout.close()
        for log in self._logs:
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
