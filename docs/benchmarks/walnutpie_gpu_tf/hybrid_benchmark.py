"""Diagnostic engineering experiment: CPU/GPU/hybrid on an expensive TF model.

Numerical kernels and generated data use TensorFlow/XLA. Standard-library host
code records provenance, transfers tensors and summarizes timing diagnostics.
Short fixed kernels cannot establish posterior convergence/default readiness.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time

PLAN = 'docs/plans/bayesfilter-walnutpie-hybrid-large-model-2026-10-29.md'


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def neural_model(tf, q, x, w, a, y):
    scale = tf.sqrt(tf.cast(tf.shape(x)[0], tf.float64))
    h = tf.tanh(x[None, :, :] + q[:, None, :] / scale)
    h = tf.tanh(tf.matmul(h, w))
    pred = tf.linalg.matvec(h, a)
    return -.5*tf.reduce_sum(q*q, -1)-.5*tf.reduce_sum((pred-y[None, :])**2, -1)


def data_fixture(tf, n, d):
    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def generate():
        x = tf.random.stateless_normal([n, d], [20261029, 1], dtype=tf.float64, alg='philox')
        w = tf.random.stateless_normal([d, d], [20261029, 2], dtype=tf.float64, alg='philox')/tf.sqrt(tf.cast(d, tf.float64))
        a = tf.random.stateless_normal([d], [20261029, 3], dtype=tf.float64, alg='philox')/tf.sqrt(tf.cast(d, tf.float64))
        pred = tf.linalg.matvec(tf.tanh(tf.matmul(tf.tanh(x), w)), a)
        y = pred + tf.random.stateless_normal([n], [20261029, 4], dtype=tf.float64, alg='philox')
        return x, w, a, y
    with tf.device('/CPU:0'):
        return generate()


def make_full(tf, cfg, signature):
    from .api import initial_state, make_kernel, run, state_spec
    @tf.function(input_signature=[tf.TensorSpec([cfg.chains, cfg.dimension], tf.float64),
        tf.TensorSpec([cfg.chains, 2], tf.int32), tf.TensorSpec([], tf.float64), *signature],
        jit_compile=True, autograph=False)
    def initialize(q, seed, epsilon, *data):
        k = make_kernel(tf, lambda q: neural_model(tf, q, *data), cfg)
        return initial_state(k, q, seed, mass=tf.ones_like(q), epsilon=tf.fill([cfg.chains], epsilon), adapt_step=False)
    @tf.function(input_signature=[state_spec(tf, cfg), *signature], jit_compile=True, autograph=False)
    def sample(state, *data):
        k = make_kernel(tf, lambda q: neural_model(tf, q, *data), cfg)
        return run(k, state, cfg.draws, False)
    @tf.function(input_signature=[tf.TensorSpec([cfg.chains, cfg.dimension], tf.float64), *signature],
        jit_compile=True, autograph=False)
    def value(q, *data):
        k = make_kernel(tf, lambda q: neural_model(tf, q, *data), cfg)
        return k.value_gradient(q)
    return initialize, sample, value


def close(tf, left, right, rtol=2e-9, atol=2e-9):
    largest = 0.
    for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)):
        with tf.device('/CPU:0'):
            a, b = tf.identity(a), tf.identity(b)
            if a.dtype.is_floating:
                tf.debugging.assert_near(a, b, rtol=rtol, atol=atol)
                largest = max(largest, float(tf.reduce_max(tf.abs(a-b)).numpy()))
            else:
                tf.debugging.assert_equal(a, b)
    return largest


def timed(tf, call):
    begin = time.perf_counter(); value = call()
    # Force every returned dependency, not merely the first output.
    for x in tf.nest.flatten(value):
        x.numpy()
    return value, time.perf_counter()-begin


def interval(values):
    logs = [math.log(v) for v in values]; mean = statistics.mean(logs)
    half = 2.446912*statistics.stdev(logs)/math.sqrt(len(logs))
    return dict(ratio=math.exp(mean), interval95=[math.exp(mean-half), math.exp(mean+half)],
        per_seed_ratios=values, method='Student-t on seven paired seed-block log ratios, df=6')


def small_model_check(tf):
    from .core import Config
    data = data_fixture(tf, 16, 3); cfg = Config(4, 3)
    sig = [tf.TensorSpec(v.shape, v.dtype) for v in data]
    _, _, value = make_full(tf, cfg, sig)
    q = tf.constant([[.1,.2,.3],[-.3,.1,.2],[.2,-.1,.3],[.5,.2,-.1]], tf.float64)
    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64), *sig], jit_compile=True, autograph=False)
    def analytic(q, x, w, a, y):
        n = tf.cast(tf.shape(x)[0], tf.float64)
        h1 = tf.tanh(x[None]+q[:,None]/tf.sqrt(n)); h2 = tf.tanh(tf.matmul(h1, w))
        residual = tf.linalg.matvec(h2, a)-y[None]
        gh2 = -residual[:,:,None]*a[None,None,:]*(1-h2*h2)
        return -q+tf.reduce_sum(tf.matmul(gh2, w, transpose_b=True)*(1-h1*h1), axis=1)/tf.sqrt(n)
    with tf.device('/CPU:0'):
        lp, g = value(q, *data); ag = analytic(q, *data)
        tf.debugging.assert_near(g, ag, atol=2e-11, rtol=2e-11)
        changed = tf.tensor_scatter_nd_add(q, [[0,1]], tf.constant([.7], tf.float64))
        lp2, g2 = value(changed, *data)
        tf.debugging.assert_equal(lp[1:], lp2[1:]); tf.debugging.assert_equal(g[1:], g2[1:])
    return dict(analytic_gradient=True, row_isolation=True, points=4, dimension=3, observations=16)


def execute(a, report):
    from .worker import setup, graph_report
    from .core import Config
    from .hybrid import make_hybrid
    tf, env = setup('gpu'); report['environment'] = env
    report['small_model_checks'] = small_model_check(tf); write(a.output/'result.json', report)
    start = time.monotonic()
    def budget():
        if time.monotonic()-start > a.seconds:
            raise TimeoutError('experiment wall-time budget exhausted')
        if tf.config.experimental.get_memory_info('GPU:0')['peak'] > 8*1024**3:
            raise MemoryError('eight GiB tensor-memory ceiling exceeded')
    report['pilots'] = []
    for n in (16384, 32768, 65536):
        budget(); cpu_data = data_fixture(tf, n, 128)
        with tf.device('/GPU:0'):
            gpu_data = tuple(tf.identity(v) for v in cpu_data)
        cfg = Config(4,128,warmup=0,draws=2,max_depth=4,max_halvings=4,variant='r2p',
            r2p_max_exponent=4,max_error=.3,initial_step=.2)
        signature = [tf.TensorSpec(v.shape,v.dtype) for v in cpu_data]
        full = {name:make_full(tf,cfg,signature) for name in ('cpu','gpu')}
        with tf.device('/CPU:0'):
            q = tf.fill([4,128], tf.constant(.03,tf.float64))
        qdev = {'cpu':q}; data = {'cpu':cpu_data,'gpu':gpu_data}
        with tf.device('/GPU:0'):
            qdev['gpu'] = tf.identity(q)
        first = {}; values = {}
        for name in ('cpu','gpu'):
            with tf.device('/'+name.upper()+':0'):
                values[name],first[name] = timed(tf,lambda:full[name][2](qdev[name],*data[name]))
        error = close(tf,values['cpu'],values['gpu']); times = {'cpu':[],'gpu':[]}
        for rep in range(7):
            for name in (('cpu','gpu') if rep%2==0 else ('gpu','cpu')):
                with tf.device('/'+name.upper()+':0'):
                    _,t = timed(tf,lambda:full[name][2](qdev[name],*data[name])); times[name].append(t)
        pilot = dict(observations=n,width=128,chains=4,first_call_seconds=first,times=times,
            max_abs_value_gradient_difference=error,
            median_gpu_over_cpu=statistics.median(times['gpu'])/statistics.median(times['cpu']),
            model_data_bytes=sum(v.shape.num_elements()*v.dtype.size for v in cpu_data))
        report['pilots'].append(pilot); write(a.output/'result.json',report)
        print(json.dumps({'pilot':pilot}),flush=True)
        if pilot['median_gpu_over_cpu'] < 1:
            break
    else:
        raise RuntimeError('No reviewed size demonstrates GPU-beneficial model evaluation')
    report['selected_model'] = dict(observations=n,width=128,chains=4,dtype='float64',tf32=False,
        formula='-.5||q||^2-.5||tanh(tanh(X+q/sqrt(N)) W) a-y||^2',
        data_sha256={name:hashlib.sha256(v.numpy().tobytes()).hexdigest() for name,v in zip(('X','W','a','y'),cpu_data)},
        data_devices={k:[v.device for v in vals] for k,vals in data.items()})
    h = make_hybrid(tf,lambda q,*x:neural_model(tf,q,*x),cfg,signature)
    graph = {}
    for name in ('cpu','gpu'):
        with tf.device('/'+name.upper()+':0'):
            graph[name+'-value'] = graph_report(full[name][2],(qdev[name],*data[name]),a.output/(name+'-value.hlo.txt'))
    seed = tf.constant([[20261028+i,0] for i in range(4)],tf.int32); epsilon = .2
    states = {}; first = {}; initial = {}
    for name in ('cpu','gpu'):
        with tf.device('/'+name.upper()+':0'):
            states[name] = full[name][0](qdev[name],seed,tf.constant(epsilon,tf.float64),*data[name])
            initial[name],first[name] = timed(tf,lambda:full[name][1](states[name],*data[name]))
        budget()
    initial['hybrid'],first['hybrid'] = timed(tf,lambda:h.sample(states['cpu'],2,*gpu_data))
    parity = {name:close(tf,initial['gpu'],initial[name]) for name in ('cpu','hybrid')}
    for r in initial.values():
        tf.debugging.assert_equal(r.state.valid,tf.ones([4],tf.bool))
        if not bool(tf.reduce_all(tf.reduce_any(r.draws!=q[None],axis=[0,2])).numpy()):
            raise ValueError('calibration has a chain with no movement')
    report['calibration'] = dict(epsilon=epsilon,first_call_seconds=first,output_max_abs_differences=parity,
        evaluations=int(tf.reduce_sum(initial['gpu'].evaluations).numpy()),round_trips=h.last_round_trips)
    report['config'] = asdict(cfg)
    for name in ('cpu','gpu'):
        with tf.device('/'+name.upper()+':0'):
            graph[name+'-sample'] = graph_report(full[name][1],(states[name],*data[name]),a.output/(name+'-sample.hlo.txt'))
    for name,fn in h.functions.items():
        with tf.device('/GPU:0' if name=='leaf' else '/CPU:0'):
            graph['hybrid-'+name] = graph_report(fn,h.graph_examples[name],a.output/('hybrid-'+name+'.hlo.txt'))
    report['graphs'] = graph; report['bytes_per_hybrid_leaf_round_trip'] = h.bytes_per_round_trip
    write(a.output/'result.json',report)
    report['records'] = []; report['gradient_records'] = []
    for block in range(7):
        budget(); key = 20261029+1000*block
        with tf.device('/CPU:0'):
            q = .03*tf.random.stateless_normal([4,128],[key,99],dtype=tf.float64,alg='philox')
            seed = tf.constant([[key+i,0] for i in range(4)],tf.int32)
        for name in ('cpu','gpu'):
            with tf.device('/'+name.upper()+':0'):
                qdev[name] = tf.identity(q)
                states[name] = full[name][0](qdev[name],seed,tf.constant(epsilon,tf.float64),*data[name])
        gradtimes = {}
        for name in (('cpu','gpu') if block%2==0 else ('gpu','cpu')):
            with tf.device('/'+name.upper()+':0'):
                _,gradtimes[name] = timed(tf,lambda:full[name][2](qdev[name],*data[name]))
        report['gradient_records'].append(dict(seed=key,seconds=gradtimes))
        for rep in range(2):
            offset = (block+rep)%3; order = ['cpu','gpu','hybrid']; order = order[offset:]+order[:offset]
            outputs = {}; seconds = {}; work = {}
            for name in order:
                budget()
                if name=='hybrid':
                    outputs[name],seconds[name] = timed(tf,lambda:h.sample(states['cpu'],2,*gpu_data))
                else:
                    with tf.device('/'+name.upper()+':0'):
                        outputs[name],seconds[name] = timed(tf,lambda:full[name][1](states[name],*data[name]))
                r = outputs[name]; tf.debugging.assert_equal(r.state.valid,tf.ones([4],tf.bool))
                work[name] = dict(evaluations=int(tf.reduce_sum(r.evaluations).numpy()),
                    macros=int(tf.reduce_sum(r.macros).numpy()),reverse_zero=int(tf.reduce_sum(tf.cast(r.reverse_failed,tf.int32)).numpy()))
            diffs = {name:close(tf,outputs['gpu'],outputs[name]) for name in ('cpu','hybrid')}
            row = dict(seed=key,repeat=rep,order=order,seconds=seconds,work=work,output_max_abs_difference=diffs,
                hybrid_round_trips=h.last_round_trips,hybrid_transfer_bytes=h.last_round_trips*h.bytes_per_round_trip)
            report['records'].append(row); write(a.output/'result.json',report)
            print(json.dumps({'trial':row}),flush=True)
    report['comparisons'] = {}
    for left,right in (('hybrid','gpu'),('hybrid','cpu'),('gpu','cpu')):
        ratios = []
        for block in range(7):
            rows = [r for r in report['records'] if r['seed']==20261029+1000*block]
            ratios.append(math.exp(statistics.median(math.log(r['seconds'][left]/r['seconds'][right]) for r in rows)))
        report['comparisons'][left+'_over_'+right] = interval(ratios)
    report['gradient_gpu_over_cpu'] = interval([r['seconds']['gpu']/r['seconds']['cpu'] for r in report['gradient_records']])
    report['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    report['peak_host_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['timed_wall_seconds'] = time.monotonic()-start
    report['status'] = 'completed'; report['engineering_validity'] = True
    report['hybrid_nominated'] = report['comparisons']['hybrid_over_gpu']['interval95'][1]<1 and report['gradient_gpu_over_cpu']['interval95'][1]<1
    write(a.output/'result.json',report)
    lines = ['# Hybrid Walnutpie large-model feasibility','',
        'Synthetic nonlinear regression; four chains, FP64, fixed kernel, two transitions per trial. This is an execution experiment, not a posterior convergence study.',
        '',f'Observations: {n}; hidden width: 128. Numerical components are XLA compiled; hybrid uses explicit host dispatch between CPU tree and GPU integration.',
        '', '| Comparison (lower favors numerator) | Paired ratio | 95% interval |','|---|---:|---|']
    for name,v in report['comparisons'].items():
        lines.append(f"| {name} | {v['ratio']:.4f} | {v['interval95']} |")
    lines.extend(['','## Decision table','',
        '| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |',
        '|---|---|---|---|---|---|',
        f"| {'Nominate optional hybrid' if report['hybrid_nominated'] else 'Do not promote hybrid'} | Paired transition timings | Matched fields/RNG/work and finite caches passed | One synthetic target, sharing, small timing sample | Validate another target or inspect transfer overhead | Posterior quality or universal backend preference |",
        '', '## Inference status','', '| Item | Finding |','|---|---|',
        '| Hard veto screen | Engineering replay passed; no posterior screen claimed |',
        '| Statistically supported ranking | Only paired intervals excluding one, conditional on this target/session |',
        '| Descriptive-only differences | Work, memory, compilation, transfer counts |',
        '| Default readiness | None |',
        '| Next evidence | Independent target and hardware/session replication |',
        '', '## Result review','',
        'Alternative explanations include workload selection, short trajectories, CPU sharing and device-specific launch behavior. Another target, chain count, CPU threading or isolated session could overturn the result. The candidate includes Python dispatch; it does not bound an optimized C++ dispatcher.'])
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='2'); p.add_argument('--seconds',type=int,default=2700); a=p.parse_args()
    if not 0<a.seconds<=2700:
        raise ValueError('reviewed campaign cap is 2700 seconds')
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',BAYESFILTER_PRELOAD_CUSTOM_OP='0',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS','')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
    a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]; hashes={}
    for f in list(Path(__file__).parent.glob('*.py'))+[root/PLAN,root/'tests/test_walnutpie_hybrid.py']:
        rel=f.relative_to(root); target=a.output/'source'/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
        hashes[str(rel)]=hashlib.sha256(f.read_bytes()).hexdigest()
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),plan=PLAN,command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_hashes=hashes,
        python=sys.executable,pid=os.getpid(),physical_gpu=a.gpu,budget_seconds=a.seconds,
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True),
        dtype='float64',tf32=False,posterior_claim=False)
    write(a.output/'manifest.json',report); begin=time.monotonic()
    try:
        execute(a,report)
    except BaseException as e:
        report.update(status='failed',error=repr(e),error_message=str(e)); write(a.output/'result.json',report); raise
    finally:
        write(a.output/'manifest.json',{**{k:v for k,v in report.items() if k not in ('records','graphs','pilots')},'wall_seconds':time.monotonic()-begin})

if __name__=='__main__':
    main()
