"""Bounded engineering validation/timing; posterior interpretation is separate."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/benchmarks'))
from . import api
from .. import stan_nuts as core
import benchmark_nuts_xla as base
import benchmark_stan_transplant as common


def sync(tf, result):
    # Every floating/integer output is consumed by the host only after timing.
    return result.completion_token.numpy()


def timed(tf, fn, args):
    begin = time.perf_counter(); out = fn(*args); sync(tf, out)
    return out, time.perf_counter()-begin


def hlo_report(tf, fn, args, folder, name, device):
    hlo = fn.experimental_get_compiler_ir(*args)(stage='optimized_hlo')
    path = folder/(name+'.hlo.txt'); path.write_text(hlo)
    graph = base.verify_graph(fn)
    forbidden = [s for s in ('BfNativeSampler', 'PyFunc', 'EagerPyFunc', 'host_callback') if s in hlo]
    if forbidden: raise RuntimeError('forbidden execution boundary: '+str(forbidden))
    calls = sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo)))
    allowed = ('cublas', 'cusolver', 'cudnn', 'triton', 'lapack', 'blas', 'xla.gpu')
    unknown = [s for s in calls if not any(token in s.lower() for token in allowed)]
    if unknown: raise RuntimeError('unreviewed custom call: '+str(unknown))
    if fn.experimental_get_tracing_count() != 1: raise RuntimeError('unexpected retracing')
    report = dict(path=str(path), sha256=hashlib.sha256(hlo.encode()).hexdigest(),
        bytes=len(hlo.encode()), fusion_instructions=len(re.findall(r'\bfusion\(', hlo)),
        while_instructions=len(re.findall(r'\bwhile\(', hlo)), custom_calls=calls,
        model_ops_present='tf_model' in hlo, trace_count=1, graph=graph)
    op_names = re.findall(r'op_name="([^"]+)"', hlo)
    report['model_gradient_inside_loop'] = any(
        'tf_model' in name and 'gradient_tape' in name and 'while' in name for name in op_names)
    return report


def setup(args):
    if args.device == 'gpu' and os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
        raise RuntimeError('set TF_FORCE_GPU_ALLOW_GROWTH=true before launching')
    conf = base.parser().parse_args(['--output', str(args.output), '--device', args.device,
                                    '--gpu', args.gpu, '--threads', '4'])
    tf, tfp, memory = base.setup(conf)
    return tf, tfp, memory


def fixtures(tf, name):
    d = 32 if name == 'regression' else 10
    if name == 'gaussian':
        def model(q): return -.5*tf.reduce_sum(q*q, -1)
        reference = lambda q: q
        data, specs, kind = (), (), 0
    elif name == 'banana':
        def reference(q):
            pair = tf.reshape(q, [-1, d//2, 2]); x = pair[..., 0]
            return tf.reshape(tf.stack((x, pair[..., 1]-.3*(x*x-1.)), -1), tf.shape(q))
        def model(q): return -.5*tf.reduce_sum(reference(q)**2, -1)
        data, specs, kind = (), (), 2
    elif name == 'funnel':
        def reference(q):
            return tf.concat((q[..., :1]/3., q[..., 1:]*tf.exp(-q[..., :1]/2.)), -1)
        def model(q):
            return -.5*tf.reduce_sum(reference(q)**2, -1)-.5*(d-1)*q[..., 0]
        data, specs, kind = (), (), 3
    elif name == 'regression':
        x = tf.random.stateless_normal([512, d], [20261027, 61], dtype=tf.float64)
        response = tf.zeros([512], tf.float64)
        def model(q, x, response):
            residual = tf.matmul(q, x, transpose_b=True)-response
            return -.5*tf.reduce_sum(q*q, -1)-.5*tf.reduce_sum(residual**2, -1)/512.
        chol = tf.linalg.cholesky(tf.eye(d, dtype=tf.float64)+tf.matmul(x, x, transpose_a=True)/512.)
        reference = lambda q: tf.matmul(q, chol)
        data = (x, response); specs = tuple(tf.TensorSpec(t.shape, t.dtype) for t in data); kind = 4
    else: raise ValueError('unknown fixture '+name)
    # Supplied starts are matched across engines; deterministic dispersal is a
    # convenience start bank, not evidence of representative posterior starts.
    q = tf.random.stateless_normal([4, d], [20261028, 321], dtype=tf.float64)*.3
    stan_data = dict(d=d, kind=kind, n=512 if kind == 4 else 0, bend=.3,
        inverse_transform=tf.eye(d, dtype=tf.float64).numpy().tolist(),
        design=data[0].numpy().tolist() if kind == 4 else [])
    return d, model, reference, data, specs, q, stan_data


def profile(tf, fn, args, folder):
    destination = folder/'profile'; destination.mkdir()
    # Inputs, executable and outputs are resident before profiling. async_wait
    # is completion synchronization, not a tensor-to-host materialization.
    from tensorflow.python.eager import context
    context.async_wait()
    tf.profiler.experimental.start(str(destination), options=tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2, python_tracer_level=0, device_tracer_level=1))
    try:
        with tf.profiler.experimental.Trace('stan_gpu_tf_sample', step_num=0, _r=1):
            result = fn(*args); context.async_wait()
    finally: tf.profiler.experimental.stop()
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    names, gpu_names, copies = Counter(), Counter(), []
    files = list(destination.rglob('*.xplane.pb'))
    for path in files:
        space = xplane_pb2.XSpace(); space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            for line in plane.lines:
                for event in line.events:
                    name = plane.event_metadata[event.metadata_id].name
                    names[name] += 1
                    if 'gpu' in plane.name.lower():
                        gpu_names[name] += 1
                    if 'memcpy' in name.lower():
                        copies.append(dict(plane=plane.name, line=line.name, name=name,
                            stats=[dict(name=plane.stat_metadata[s.metadata_id].name,
                                value=plane.stat_metadata[s.ref_value].name if s.WhichOneof('value') == 'ref_value'
                                    else str(s)) for s in event.stats]))
    if not files or not gpu_names: raise RuntimeError('GPU profiler did not capture device activity')
    out = dict(files=[str(p) for p in files], device_events=sum(gpu_names.values()),
        gpu_event_counts=dict(gpu_names), copy_events=copies, host_top_events=names.most_common(20),
        note='Device events include kernels and transfers; inspect transfer sizes/labels before claiming no host traffic.')
    out['device_to_host_events'] = sum(v for k, v in gpu_names.items() if 'memcpyd2h' in k.lower())
    out['host_to_device_events'] = sum(v for k, v in gpu_names.items() if 'memcpyh2d' in k.lower())
    out['execution_scope'] = ('TF 2.20 profiler disables command buffers; this trace describes fallback execution'
        if 'WHILE' in os.environ.get('XLA_FLAGS', '') else 'default backend under TF profiler')
    common.write(folder/'profile-summary.json', out)
    return out


def validate(tf, args, report):
    report['rng_checks'] = []
    for chains in (1, 4, 16):
        for normal in (False, True):
            cfg_rng = core.Config(3, chains=chains)
            def build_rng(cfg_rng, normal):
                @tf.function(input_signature=[tf.TensorSpec([cfg_rng.chains, 2], tf.int32),
                    tf.TensorSpec([cfg_rng.chains], tf.bool)], jit_compile=True, autograph=False)
                def compare(seed, mask):
                    target_rng = lambda q: -.5*tf.reduce_sum(q*q, -1)
                    a = core.Kernel(tf, target_rng, cfg_rng).random(seed, mask, normal=normal, size=(3,))
                    b = api.StaticChainKernel(tf, target_rng, cfg_rng).random(seed, mask, normal=normal, size=(3,))
                    return a, b
                return compare
            pair = build_rng(cfg_rng, normal)(tf.constant([[31+i, 7*i] for i in range(chains)], tf.int32),
                                               tf.constant([i % 2 == 0 for i in range(chains)]))
            if not all(bool(tf.reduce_all(a == b)) for a, b in zip(*pair)):
                raise RuntimeError('per-chain RNG mismatch')
            report['rng_checks'].append(dict(chains=chains, normal=normal, exact=True))
    if not args.oracle or not args.oracle.is_file(): raise RuntimeError('fresh native oracle required')
    from ..audit_full_stan import target
    cache = {}; max_error = 0.; count = 0
    for line in args.oracle.read_text().splitlines():
        row = json.loads(line); values = list(map(float, row['stdin'].split()))
        d, kind, depth = map(int, values[:3]); e, jitter, threshold = values[3:6]
        q, momentum = values[6:8], values[8:10]; matrix = [values[10:12], values[12:14]]
        tape = values[15:]
        if len(tape) != 256: raise ValueError('oracle tape length')
        metric_kind = 'unit' if matrix == [[1., 0.], [0., 1.]] else ('diag' if matrix[0][1] == 0. else 'dense')
        key = kind, depth, jitter, threshold, metric_kind
        if key not in cache:
            cfg = core.Config(2, metric=metric_kind, depth=depth, jitter=jitter, max_delta=threshold)
            def build(cfg, kind):
                shape = [1, 2, 2] if cfg.metric == 'dense' else [1, 2]
                @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float64),
                    tf.TensorSpec([1, 2], tf.float64), tf.TensorSpec([1], tf.float64),
                    tf.TensorSpec(shape, tf.float64), tf.TensorSpec([1, 256], tf.float64)],
                    autograph=False, jit_compile=True)
                def transition(q, p, e, m, tape):
                    k = api.StaticChainKernel(tf, target(tf, kind), cfg, uniform_tape=tape, momentum=p)
                    return k.transition(q, e, m, k.factor(m), tf.zeros([1, 2], tf.int32))
                return transition
            cache[key] = build(cfg, kind)
        m = matrix if metric_kind == 'dense' else [matrix[0][0], matrix[1][1]]
        out = cache[key](*[tf.constant(v, tf.float64) for v in ([q], [momentum], [e], [m], [tape])])
        expected = [list(map(float, line.split())) for line in row['stdout'].splitlines()]; top = expected[0]
        discrete = [int(out.leaps[0]), int(out.depth[0]), int(out.divergent[0]), int(out.seed[0, 1])]
        if discrete != [int(top[j]) for j in (1, 2, 3, 5)]: raise RuntimeError('native branch mismatch '+str(key))
        got = [float(out.acceptance[0]), float(out.energy[0]), float(out.epsilon[0]),
            *out.phase.q.numpy().tolist()[0], *out.phase.p.numpy().tolist()[0],
            *out.phase.g.numpy().tolist()[0], float(out.phase.v[0])]
        want = [top[0], top[4], top[6], *expected[1], *expected[2], *expected[3], expected[4][0]]
        error = max(abs(x-y)/(1.+abs(y)) for x, y in zip(got, want))
        if error > 2e-12: raise RuntimeError('native numerical mismatch '+str(error))
        max_error = max(error, max_error); count += 1
    report['native_transition_oracle'] = dict(cases=count, maximum_scaled_error=max_error,
        sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(), discrete_choices_exact=True)
    print('Native transition checks passed: '+str(count), flush=True)
    report['common_state_layout_checks'] = []
    for kind in (0, 1, 2):
        cfg_compare = core.Config(2, chains=4, depth=6, warmup=0, draws=1, adapt=False)
        def build_layout(layout, kind):
            @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64),
                tf.TensorSpec([4], tf.float64), tf.TensorSpec([4, 2], tf.int32)],
                jit_compile=True, autograph=False)
            def fn(q, epsilon, seeds):
                k = layout(tf, target(tf, kind), cfg_compare)
                metric = tf.ones_like(q)
                return k.transition(q, epsilon, metric, metric, seeds)
            return fn
        old = build_layout(core.Kernel, kind); new = build_layout(api.StaticChainKernel, kind)
        err = 0.
        for rep in range(8):
            q = tf.reshape(tf.sin(tf.cast(tf.range(8)+rep, tf.float64)), [4, 2])
            epsilon = tf.constant([.05, .1, .25, .5], tf.float64)
            seeds = tf.constant([[31+i, 17*rep] for i in range(4)], tf.int32)
            a, b = old(q, epsilon, seeds), new(q, epsilon, seeds)
            for left, right in zip(tf.nest.flatten(a), tf.nest.flatten(b)):
                if left.dtype.is_floating:
                    err = max(err, float(tf.reduce_max(tf.abs(left-right)/(1.+tf.abs(right)))))
                elif not bool(tf.reduce_all(left == right)): raise RuntimeError('controlled layout branch mismatch')
        if err > 2e-12: raise RuntimeError('controlled layout numerical mismatch')
        report['common_state_layout_checks'].append(dict(kind=kind, transitions=32, max_scaled_error=err,
                                                         discrete_choices_exact=True))
    # Generic data tensors, analytical gradient, full-fit composition, replay and
    # changed data are tested together. A downstream TF reduction forces consumption.
    c, d = 4, 4
    cfg = core.Config(d, chains=c, warmup=160, draws=16, depth=6)
    def model(q, x, y):
        return -.5*tf.reduce_sum(q*q, -1)-.5*tf.reduce_sum((tf.matmul(q, x, transpose_b=True)-y)**2, -1)/16.
    specs = [tf.TensorSpec([16, d], tf.float64), tf.TensorSpec([16], tf.float64)]
    functions = api.make_functions(tf, model, cfg, specs)
    baseline = api.make_functions(tf, model, cfg, specs, rng_layout='map')
    q = tf.ones([c, d], tf.float64)*.1
    x = tf.reshape(tf.sin(tf.cast(tf.range(16*d), tf.float64)), [16, d]); y = tf.zeros([16], tf.float64)
    inputs = (q, tf.ones_like(q, tf.bool), tf.ones([c], tf.float64), tf.ones_like(q),
              tf.constant([[101+i, 0] for i in range(c)], tf.int32), x, y)
    print('Compiling full-fit static RNG', flush=True)
    a, first = timed(tf, functions.fit, inputs)
    print('Compiling full-fit map RNG', flush=True)
    b, baseline_first = timed(tf, baseline.fit, inputs)
    if not bool(tf.reduce_all(a.valid & b.valid)): raise RuntimeError('invalid full-fit state')
    errors = []
    for left, right in zip(tf.nest.flatten(a), tf.nest.flatten(b)):
        if left.dtype.is_floating:
            errors.append(float(tf.reduce_max(tf.abs(left-right)/(1.+tf.abs(right)))))
        elif not bool(tf.reduce_all(left == right)): raise RuntimeError('layout discrete mismatch')
    if max(errors) > 2e-12: raise RuntimeError('RNG layout trajectory mismatch '+str(max(errors)))
    again = functions.fit(*inputs)
    if not bool(tf.reduce_all(a.sampling.draws == again.sampling.draws)): raise RuntimeError('replay mismatch')
    changed = functions.fit(*inputs[:-1], y+1.)
    if bool(tf.reduce_all(changed.sampling.draws == a.sampling.draws)): raise RuntimeError('model data frozen at trace time')
    value, gradient = functions.value_gradient(q, x, y)
    expected = -q-tf.matmul(tf.matmul(q, x, transpose_b=True)-y, x)/16.
    gradient_error = float(tf.reduce_max(tf.abs(expected-gradient)))
    if gradient_error > 2e-12: raise RuntimeError('wrong target gradient')
    h = hlo_report(tf, functions.fit, inputs, args.output, 'full-fit', args.device)
    if not h['model_ops_present']: raise RuntimeError('model absent from full-fit HLO')
    if not h['model_gradient_inside_loop']: raise RuntimeError('model gradient absent from loop HLO')
    if args.device.upper()+':0' not in a.sampling.draws.device: raise RuntimeError('wrong device')
    @tf.function(input_signature=functions.fit.input_signature, jit_compile=True, autograph=False)
    def composed(q, present, e, m, seed, x, y):
        result = functions.fit(q+.125, present, e, m, seed, x*1.1, y)
        return tf.reduce_mean(result.sampling.draws, axis=0)*2., result.valid
    print('Checking enclosing TF composition', flush=True)
    actual, valid = composed(*inputs)
    expected_fit = functions.fit(q+.125, inputs[1], inputs[2], inputs[3], inputs[4], x*1.1, y)
    error = float(tf.reduce_max(tf.abs(actual-2.*tf.reduce_mean(expected_fit.sampling.draws, axis=0))))
    if error > 2e-12 or not bool(tf.reduce_all(valid)): raise RuntimeError('composition mismatch')
    sample_args = (*a.warmup[:4], x, y)
    print('Compiling retained sample for profiler', flush=True)
    functions.sample(*sample_args).completion_token.numpy()
    if args.device == 'gpu': report['profile'] = profile(tf, functions.sample, sample_args, args.output)
    report['integration'] = dict(first_call_seconds=first, baseline_first_call_seconds=baseline_first,
        gradient_max_error=gradient_error, baseline_max_scaled_error=max(errors),
        composition_max_error=error, changed_data_changes_draws=True, replay_exact=True,
        output_device=a.sampling.draws.device, hlo=h, full_fit_trace_count=functions.fit.experimental_get_tracing_count())


def benchmark(tf, args, report):
    report['targets'] = []
    for name in args.targets.split(','):
        print('Benchmarking '+name, flush=True)
        folder = args.output/name; folder.mkdir()
        d, model, reference, data, specs, q, stan_data = fixtures(tf, name)
        cfg = core.Config(d, chains=4, warmup=args.warmup, draws=args.draws, depth=args.depth)
        inputs = (q, tf.ones_like(q, tf.bool), tf.ones([4], tf.float64), tf.ones_like(q),
                  tf.constant([[20261028*4+i, 0] for i in range(4)], tf.int32), *data)
        funcs, compiled, outputs = {}, {}, {}
        for layout in ('map', 'static'):
            functions = api.make_functions(tf, model, cfg, specs, rng_layout=layout)
            output, first = timed(tf, functions.fit, inputs)
            if not bool(tf.reduce_all(output.valid)): raise RuntimeError(name+' invalid '+layout)
            compiled[layout] = dict(first_call_including_execution_seconds=first,
                hlo=hlo_report(tf, functions.fit, inputs, folder, layout, args.device))
            funcs[layout] = functions.fit; outputs[layout] = output
        # Alternate order so the second implementation is not always advantaged.
        times = {'map': [], 'static': []}
        for rep in range(3):
            for layout in (('map', 'static') if rep % 2 == 0 else ('static', 'map')):
                output, seconds = timed(tf, funcs[layout], inputs)
                times[layout].append(seconds); outputs[layout] = output
        if not all(bool(tf.reduce_all(output.valid)) for output in outputs.values()):
            raise RuntimeError('invalid timed retained state')
        discrete_equal = bool(tf.reduce_all(outputs['map'].sampling.seed == outputs['static'].sampling.seed))
        scaled = float(tf.reduce_max(tf.abs(outputs['map'].sampling.draws-outputs['static'].sampling.draws)/
                                      (1.+tf.abs(outputs['map'].sampling.draws))))
        for layout, output in outputs.items():
            for label, value in [('draws', output.sampling.draws), ('telemetry', output.sampling.telemetry),
                ('warmup', output.warmup.draws), ('warmup-telemetry', output.warmup.telemetry)]:
                common.tensor_save(tf, folder/(layout+'-'+label+'.tensor'), value)
        item = dict(name=name, config=asdict(cfg), compile=compiled, times_seconds=times,
            median_seconds={k: statistics.median(v) for k, v in times.items()},
            rng_layout_final_seed_equal=discrete_equal, rng_layout_max_scaled_draw_error=scaled,
            timing_boundary='full initialization + warmup + retained sampling; compilation and output serialization excluded',
            seed=20261028, starts=q.numpy().tolist(), native=None)
        common.write(folder/'model-data.json', stan_data)
        item['data_sha256'] = hashlib.sha256((folder/'model-data.json').read_bytes()).hexdigest()
        # Native Stan runs once per target in the CPU reference worker. The
        # subprocess + CSV scope is explicit and differs from TF numerical fits.
        if args.device == 'cpu':
            native_args = argparse.Namespace(stan=ROOT/'docs/benchmarks/stan_targets',
                warmup=args.warmup, draws=args.draws, depth=args.depth)
            native_folder = folder/'native-stan'; native_folder.mkdir()
            from types import SimpleNamespace
            tf_target = SimpleNamespace(dimension=d, log_prob=lambda state: model(state, *data))
            item['native_target_parity'] = common.density_check(tf, tf_target, q, stan_data,
                                                               native_args, native_folder)
            draws, stats, native_report = common.stan_run(tf, q, stan_data, 20261028, native_args, native_folder)
            common.tensor_save(tf, folder/'native-draws.tensor', draws)
            common.tensor_save(tf, folder/'native-telemetry.tensor', stats)
            item['native'] = native_report
            # Separate native in-process CPU scope avoids pricing CSV and
            # process startup against numerical TF execution. Four chains are
            # sequential here, whereas the CmdStan subprocesses run in parallel.
            if name != 'regression':
                from ..native_sampler_bridge.validate import Native
                from ..native_sampler_bridge.wrapper import FitConfig
                native = Native()
                native_cfg = FitConfig('stan', stan_data['kind'], d, 4, args.warmup, args.draws, args.depth)
                starts = [v for row in q.numpy().tolist() for v in row]
                inverse = [v for row in stan_data['inverse_transform'] for v in row]
                seeds = [20261028+i for i in range(4)]
                native_times = []
                for rep in range(3):
                    begin = time.perf_counter()
                    native.fit(native_cfg, [.3, 1., .8, .5], starts, inverse, seeds)
                    native_times.append(time.perf_counter()-begin)
                item['native_inprocess'] = dict(times_seconds=native_times,
                    median_seconds=statistics.median(native_times), chains='four sequential',
                    timing_boundary='ctypes dispatch, allocation, initialization, warmup, retained; no CSV/startup',
                    seeds=seeds)
        report['targets'].append(item); common.write(args.output/'results.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--mode', choices=['validate', 'benchmark', 'qualification'], required=True)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--targets', default='gaussian,banana,funnel,regression')
    p.add_argument('--warmup', type=int, default=500); p.add_argument('--draws', type=int, default=500)
    p.add_argument('--depth', type=int, default=10)
    args = p.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); tf, tfp, memory = setup(args)
    report = dict(command=sys.argv, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        environment=sys.executable, tensorflow=tf.__version__, tfp=tfp.__version__, python=platform.python_version(),
        device=args.device, gpu_intentionally_hidden=args.device == 'cpu', memory_policy=memory,
        gpu_details=[tf.config.experimental.get_device_details(g) for g in tf.config.list_physical_devices('GPU')],
        trust_basis='trusted_escalated_tool_execution' if args.device == 'gpu' else 'explicit_cpu_reference',
        jit_compile=True, tf32=False, xla_flags=os.environ.get('XLA_FLAGS', ''),
        plan='docs/plans/bayesfilter-stan-gpu-tf-master-2026-10-28.md',
        data_version='analytic targets; regression design stateless seed 20261027,61', status='running')
    common.write(args.output/'results.json', report)
    try:
        with tf.device('/'+args.device.upper()+':0'):
            if args.mode in ('validate', 'qualification'): validate(tf, args, report)
            if args.mode in ('benchmark', 'qualification'): benchmark(tf, args, report)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'; report['error'] = repr(error); raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        report['host_max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.device == 'gpu': report['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
        common.write(args.output/'results.json', report)
    print(json.dumps({'status': report['status'], 'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__': main()
