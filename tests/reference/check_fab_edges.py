"""CPU-only boundary and native-RNG reference comparisons; not fit evidence."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true',
    TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def run(args):
    start = time.monotonic()
    import tensorflow as tf
    from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
    from bayesfilter.inference.neutra_fab import FABConfig, FABTrainer, Point, Replay, fab_weighted_loss, fab_replay_loss
    data = json.loads(args.fixture.read_text())
    reference = json.loads(args.reference.read_text())
    edge = json.loads(args.edges.read_text())
    dtype = tf.float64
    a = lambda x: tf.convert_to_tensor(x, dtype)
    c = data['config']
    f = NeuTraTransport(replace(NeuTraTransportConfig.hoffman_author_iaf(c['dimension'], conditional_scale_cap=2., seed=(12, 31)),
        hidden_layers=tuple(c['hidden_layers']), affine_center=tuple(c['affine_center']), affine_scale=tuple(c['affine_scale'])))
    for v, x in zip(f.trainable_variables, data['params']): v.assign(x)
    cfg = FABConfig(8, 3, 3, 2, .17, .002, .9, .999, 1e-3, 16, 16, 1, 2., None, True, .65, 1.02, jit_compile=False)
    target = lambda x: (-.5*tf.reduce_sum(x*x, -1), -x, tf.reduce_all(tf.math.is_finite(x), -1))
    tr = FABTrainer(f, target, cfg, target_signature='a'*64, seed=(8, 9))
    rows = []
    def check(name, got, expected):
        got = tf.convert_to_tensor(got)
        expected = tf.convert_to_tensor(expected, got.dtype)
        if got.dtype.is_floating:
            ok = tf.reduce_all(tf.math.is_finite(got) == tf.math.is_finite(expected))
            finite = tf.math.is_finite(expected)
            err = tf.reduce_max(tf.abs(tf.where(finite, got-expected, 0.)))
            ok &= tf.reduce_all(tf.where(finite, tf.abs(got-expected) <= 1e-10*(1+tf.abs(expected)), got == expected))
            error = float(err)
        else:
            ok = tf.reduce_all(got == expected); error = 0.
        rows.append({'name': name, 'passed': bool(ok), 'maximum_error': error})
    r = reference['density']
    bad_x = tf.tensor_scatter_nd_update(a(data['x']), [[2, 0]], a([math.nan]))
    bad_q = tf.tensor_scatter_nd_update(a(r['log_q']), [[1]], a([math.nan]))
    point = Point(bad_x, bad_q, a(r['log_p']), a(r['grad_q']), a(r['grad_p']),
        tf.constant([True, False, False, True, True, True, True, True]))
    out = tr._replace_invalid_initial(point, tf.constant(edge['replacements'], tf.int32))
    for local, source in [('x','x'), ('log_q','log_q'), ('log_p','log_p'), ('grad_q','grad_log_q'), ('grad_p','grad_log_p')]:
        check('initial_replacement/'+local, getattr(out, local), edge['replaced'][source])
    check('initial_replacement/valid', out.valid, [True]*8)
    b = edge['buffer_initial']
    state = Replay(a(b['x']), a(b['log_w']), a(b['log_q_old']), tf.constant(0), tf.constant(16))
    new_x = tf.tensor_scatter_nd_update(a(data['x'])-1, [[2, 0]], a([math.nan]))
    new_w = tf.tensor_scatter_nd_update(tf.linspace(a(1.), a(2.), 8), [[1]], a([math.nan]))
    new_q = tf.tensor_scatter_nd_update(tf.linspace(a(-2.), a(-4.), 8), [[3]], a([math.inf]))
    state = tr.replay_add(state, new_x, new_w, new_q)
    for field in ['x', 'log_w', 'log_q_old', 'index']:
        check('buffer/invalid_add/'+field, getattr(state, field), edge['buffer_after_add'][field])
    state = tr.replay_adjust(state, tf.range(8), tf.linspace(a(-3.), a(-5.), 8), a([.1, .3, math.inf, math.nan, -.2, .3, .7, -.8]))
    for field in ['x', 'log_w', 'log_q_old', 'index']:
        expected = edge['buffer_after_adjust'][field]
        if field == 'log_w': expected = [(-math.inf if x is None else x) for x in expected]
        check('buffer/invalid_adjust/'+field, getattr(state, field), expected)
    for i, row in enumerate(edge['extreme_loss']):
        q = tf.Variable(r['log_q'], dtype=dtype)
        with tf.GradientTape() as tape: value = fab_weighted_loss(q, a(row['weights']))
        check(f'extreme_loss/{i}/value', value, row['value'])
        check(f'extreme_loss/{i}/gradient', tape.gradient(value, q), row['gradient'])
    q = tf.Variable(r['log_q'], dtype=dtype)
    with tf.GradientTape() as tape: loss, correction, _ = fab_replay_loss(q, a(edge['extreme_replay']['old_q']), 2.)
    check('extreme_replay/loss', loss, edge['extreme_replay']['loss'])
    check('extreme_replay/gradient', tape.gradient(loss, q), edge['extreme_replay']['gradient'])
    check('extreme_replay/correction', correction, edge['extreme_replay']['adjustment'])

    rng_tr = FABTrainer(f, target, replace(cfg, batch_size=2, replay_capacity=4, replay_min_size=4,
        jit_compile=True), target_signature='b'*64, seed=(31, 51))
    rng_state = Replay(tf.zeros([4, c['dimension']], dtype), tf.math.log(a([.1,.2,.3,.4])),
        tf.zeros([4], dtype), tf.constant(0), tf.constant(4))
    n = edge['rng']['n']
    @tf.function(input_signature=[tf.TensorSpec([2],tf.int32)], jit_compile=True, autograph=False)
    def frequencies(key):
        def body(i, counts):
            k = tf.random.experimental.stateless_fold_in(key, i)
            indices = rng_tr.replay_sample(rng_state, k)
            code = tf.reduce_min(indices)*4+tf.reduce_max(indices)
            return i+1, tf.tensor_scatter_nd_add(counts, tf.reshape(code,[1,1]), tf.ones([1],tf.int32))
        return tf.while_loop(lambda i, _: i<n, body, (tf.constant(0),tf.zeros([16],tf.int32)),parallel_iterations=1)[1]
    counts = frequencies(tf.constant([31051, 0]))
    pairs = edge['rng']['pairs']
    observed = [float(counts[i*4+j])/n for i,j in pairs]
    normals = tf.random.stateless_normal([n], [31052,0], dtype=dtype)
    uniforms = tf.random.stateless_uniform([n], [31053,0], dtype=dtype)
    observed += [float(tf.reduce_mean(tf.cast(normals <= t,dtype))) for t in [-1.,0.,1.]]
    observed += [float(tf.reduce_mean(tf.cast(uniforms <= t,dtype))) for t in [.1,.5,.9]]
    other = edge['rng']['pair_frequencies']+edge['rng']['normal_cdf']+edge['rng']['uniform_cdf']
    laws = edge['rng']['pair_law']+[.5*(1+math.erf(t/math.sqrt(2))) for t in [-1.,0.,1.]]+[.1,.5,.9]
    radius = math.sqrt(math.log(2*24/.01)/(2*n))
    rng_checks = []
    for i,(a_,b_,law) in enumerate(zip(observed,other,laws)):
        passed = abs(a_-law)+radius <= .025 and abs(b_-law)+radius <= .025 and abs(a_-b_)+2*radius <= .025
        rng_checks.append({'event':i,'tf':a_,'jax':b_,'law':law,'radius':radius,
            'difference_interval':[a_-b_-2*radius,a_-b_+2*radius],'margin':.025,'passed':passed})
    result = {'status':'passed' if all(r['passed'] for r in rows+rng_checks) else 'failed',
        'checks':rows,'rng':rng_checks,'n':n,'wall_seconds':time.monotonic()-start,'gpu_intentionally_hidden':True,
        'rng_status':'bounded event-frequency equivalence only',
        'exclusions':['all-invalid batch has undefined upstream replacement law',
            'local finite-score and target-status guards remain stricter']}
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(rows),'rng_checks':len(rng_checks),'wall_seconds':result['wall_seconds']}))
    return 0 if result['status']=='passed' else 1

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['fixture','reference','edges','output']: p.add_argument('--'+name,required=True,type=Path)
    sys.exit(run(p.parse_args()))
