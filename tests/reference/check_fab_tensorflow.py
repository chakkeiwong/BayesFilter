"""Compare actual TensorFlow FAB with executable author-reference artifacts."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def run(args):
    start = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu is not None else '-1'
    import tensorflow as tf
    memory = None
    if args.gpu is not None:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(args.tf32)
    from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
    from bayesfilter.inference.neutra_fab import FABConfig, FABTrainer, FABAdam, fab_weighted_loss, fab_replay_loss
    data = json.loads(args.fixture.read_text())
    ref = json.loads(args.reference.read_text())
    if ref['fixture_sha256'] != hashlib.sha256(args.fixture.read_bytes()).hexdigest():
        raise ValueError('reference/fixture mismatch')
    dtype = tf.as_dtype(args.dtype)
    cf = data['config']
    cfg = replace(NeuTraTransportConfig.hoffman_author_iaf(cf['dimension'], conditional_scale_cap=cf['conditional_scale_cap'],
        seed=tuple(cf['seed']), dtype=args.dtype), hidden_layers=tuple(cf['hidden_layers']),
        affine_center=tuple(cf['affine_center']), affine_scale=tuple(cf['affine_scale']))

    def array(x):
        return tf.convert_to_tensor(x, dtype)

    def flow():
        f = NeuTraTransport(cfg)
        for v, value in zip(f.trainable_variables, data['params']):
            v.assign(value)
        return f

    def target(x):
        means = tf.constant([[.6, -.4, .2, .1], [-.7, .8, 1., -.3]], x.dtype)[:, :cf['dimension']]
        scales = tf.constant([[.8, 1.1, .9, 1.2], [1.2, .7, 1.1, .8]], x.dtype)[:, :cf['dimension']]
        diff = (x[:, None, :]-means)/scales
        terms = -.5*tf.reduce_sum(diff*diff, -1)-tf.reduce_sum(tf.math.log(scales), -1)-cf['dimension']/2*math.log(2*math.pi)
        terms += tf.math.log(tf.constant([.4, .6], x.dtype))
        log_p = tf.reduce_logsumexp(terms, -1)
        score = tf.reduce_sum(tf.nn.softmax(terms, -1)[..., None]*(-diff/scales), axis=1)
        return log_p, score, tf.reduce_all(tf.math.is_finite(x), -1)

    checks = []
    elementary = 1e-10 if args.dtype == 'float64' else 2e-5
    composed = 1e-8 if args.dtype == 'float64' else 2e-4
    def check(name, got, expected, *, tol=None, exact=False):
        a = tf.convert_to_tensor(got)
        b = tf.convert_to_tensor(expected, dtype=a.dtype)
        if a.shape != b.shape:
            checks.append(dict(name=name, passed=False, reason='shape', got=str(a.shape), expected=str(b.shape)))
            return
        if exact:
            ok = bool(tf.reduce_all(a == b))
            err = 0. if ok else 1.
            bound = 0.
        else:
            a, b = tf.cast(a, tf.float64), tf.cast(b, tf.float64)
            err = float(tf.reduce_max(tf.abs(a-b))) if int(tf.size(a)) else 0.
            bound = elementary if tol is None else tol
            ok = bool(tf.reduce_all(tf.math.is_finite(a) & tf.math.is_finite(b)
                & (tf.abs(a-b) <= bound*(1+tf.abs(b)))))
        checks.append(dict(name=name, passed=ok, max_absolute_error=err, atol=bound, rtol=bound))

    def check_params(prefix, got, expected, tol=composed):
        for i, (a, b) in enumerate(zip(got, expected)):
            check(f'{prefix}/{i}', a, b, tol=tol)
        if len(got) != len(expected):
            raise ValueError('parameter count mismatch')

    f = flow()
    x, w = array(data['x']), array(data['log_w'])
    def evaluate_callback(xx, ww):
        fx, ld = f.forward_and_logdet(xx)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xx)
            qq = f.log_prob(xx)
            ll = fab_weighted_loss(qq, ww)
            total = tf.reduce_sum(qq)
        return (fx, ld, f.inverse_and_forward_logdet(xx)[0], qq,
            tape.gradient(qq, xx), tape.gradient(total, f.trainable_variables),
            ll, tape.gradient(ll, f.trainable_variables))
    evaluator = evaluate_callback
    if args.compile_callback:
        evaluator = tf.function(evaluate_callback, input_signature=[
            tf.TensorSpec(x.shape, dtype), tf.TensorSpec(w.shape, dtype)],
            jit_compile=True, autograph=False)
    fx, ld, inverse, q, spatial_grad, param_grad, loss, fresh_grads = evaluator(x, w)
    check('callback/forward', fx, ref['density']['forward'][0])
    check('callback/logdet', ld, ref['density']['forward'][1])
    check('callback/inverse', inverse, ref['density']['inverse'])
    check('callback/log_q', q, ref['density']['log_q'])
    check('callback/spatial_score', spatial_grad, ref['density']['grad_q'])
    check_params('callback/parameter_gradient', param_grad, ref['density']['parameter_gradient'])
    check('fresh/loss', loss, ref['fresh']['loss'])
    check_params('fresh/gradient', fresh_grads, ref['fresh']['gradients'], elementary)
    p, gp, _ = target(x)
    check('callback/log_p', p, ref['density']['log_p'])
    check('callback/target_score', gp, ref['density']['grad_p'])
    baseline_ratio = data['baseline_fresh_loss']/ref['fresh']['loss']
    # The protected original 2D fixture has the old SUM loss. Later fixtures
    # may be exported after the repair; report the ratio without re-labeling.
    if cf['dimension'] == 2:
        check('baseline_detected/fresh_scale_ratio', tf.constant(baseline_ratio, tf.float64), 8.)

    base = ref['runs'][0]['config']
    def config(operator='hmc', replay=False, **kw):
        conf = {**base, 'transition_operator': operator, 'jit_compile': args.jit}
        if not replay:
            conf.update(replay_capacity=0, replay_min_size=0, updates_per_pass=1)
        conf.update(kw)
        return FABConfig(**conf)

    trainer = FABTrainer(f, target, config(), target_signature='a'*64, seed=(8, 9))
    point = trainer._point(x)
    for row in ref['bridge']:
        beta = array(row['beta'])
        check(f"bridge/{row['beta']}/value", trainer._log_prob(point, beta), row['value'])
        check(f"bridge/{row['beta']}/score", trainer._score(point, beta), row['score'])
    for row in ref['trajectories']:
        tr = FABTrainer(flow(), target, config(leapfrog_steps=row['steps']), target_signature='a'*64, seed=(8, 9))
        proposal, momentum, valid = tr._integrate(tr._point(x), array(ref['momentum']), array(.75), array(.17))
        check(f"leapfrog/{row['steps']}/position", proposal.x, row['position'])
        check(f"leapfrog/{row['steps']}/momentum", momentum, row['momentum'])
        check(f"leapfrog/{row['steps']}/log_p", proposal.log_p, row['log_p'])
        check(f"leapfrog/{row['steps']}/valid", valid, [True]*8, exact=True)

    for row in ref['optimizer']:
        ff = flow()
        optimizer = FABAdam(ff.trainable_variables, config(adam_epsilon=row['epsilon']))
        for i, (factor, expected) in enumerate(zip([1., -2., .1], row['trace'])):
            optimizer.apply_gradients(zip([array(factor)*g for g in fresh_grads], ff.trainable_variables))
            check_params(f"adam/{row['epsilon']}/{i}/params", ff.trainable_variables, expected['params'], elementary)
            check_params(f"adam/{row['epsilon']}/{i}/m", optimizer.m, expected['m'], elementary)
            check_params(f"adam/{row['epsilon']}/{i}/v", optimizer.v, expected['v'], elementary)
            check(f"adam/{row['epsilon']}/{i}/count", optimizer.iterations, expected['count'], exact=True)

    # Replay loss/gradient are tested independently of any sampling or optimizer.
    with tf.GradientTape() as tape:
        q = f.log_prob(x)
        replay_loss, adj, _ = fab_replay_loss(q, array(ref['replay_loss']['old_q']), 2.)
    check('replay/loss', replay_loss, ref['replay_loss']['loss'])
    check_params('replay/gradient', tape.gradient(replay_loss, f.trainable_variables), ref['replay_loss']['gradients'])

    for row in ref.get('stress_mutations', []):
        tr = FABTrainer(flow(), target, config(row['operator']), target_signature='a'*64, seed=(8, 9))
        result = tr.ais_from_random(x, array(row['steps']), array(row['noise']), array(row['log_uniform']), tf.range(8))
        for name in ['x', 'log_w', 'log_q', 'steps', 'stage_positions', 'log_weight_increments', 'acceptance']:
            check('stress/'+row['operator']+'/'+name, result[name], row['ais'][name], tol=composed)

    def check_state(prefix, tr, expected, replay):
        check_params(prefix+'/params', tr.variables, expected['params'])
        check_params(prefix+'/m', tr.optimizer.m, expected['m'])
        check_params(prefix+'/v', tr.optimizer.v, expected['v'])
        check(prefix+'/count', tr.optimizer.iterations, expected['count'], exact=True)
        check(prefix+'/steps', tr.steps, expected['steps'], tol=composed)
        if replay:
            b = expected['buffer']
            check(prefix+'/buffer_index', tr.replay.index, b['index'], exact=True)
            check(prefix+'/buffer_size', tr.replay.size, b['size'], exact=True)
            active = [i for i, v in enumerate(b['log_w']) if v is not None]
            for name in ['x', 'log_w', 'log_q_old']:
                check(prefix+'/buffer_'+name, tf.gather(getattr(tr.replay, name), active),
                    [b[name][i] for i in active], tol=composed)

    for run in ref['runs']:
        operator, replay = run['operator'], run['replay']
        tr = FABTrainer(flow(), target, config(operator, replay), target_signature='a'*64, seed=(8, 9))
        prefix = f'complete/{operator}/{"replay" if replay else "fresh"}'
        tape_holder = {}
        def draw(key):
            return tr.transport.forward_and_logdet(array(tape_holder['tape']['z']))[0]
        def ais(xx, key, steps):
            row = tape_holder['tape']
            r = tr.ais_from_random(xx, steps, array(row['noise']), array(row['log_uniform']), tf.range(8))
            expected = row.get('ais', row)
            for name in ['x', 'log_w', 'log_q', 'steps']:
                check(tape_holder['label']+'/ais_'+name, r[name], expected[name], tol=composed)
            for name in ['stage_positions', 'log_weight_increments', 'acceptance']:
                if name in expected:
                    check(tape_holder['label']+'/'+name, r[name], expected[name], tol=composed)
            return r
        def select(buffer, key):
            row = tape_holder['tape']
            selected = tr.replay_select(buffer, array(row['gumbel']), tf.constant(row['order'], tf.int32))
            check(tape_holder['label']+'/selected_indices', selected, row['indices'], exact=True)
            return selected
        # Only random-source adapters are replaced; numerical kernels and the
        # actual production step orchestration execute unchanged.
        tr.draw, tr.ais = draw, ais
        if replay: tr.replay_sample = select
        for i, row in enumerate(run['initialization']):
            tape_holder.update(tape=row, label=prefix+f'/fill{i}')
            out = tr.step()
            check(prefix+f'/fill{i}/updates', tf.constant(len(out['updates'])), 0, exact=True)
        check_state(prefix+'/init', tr, run['initial_state'], replay)
        for i, row in enumerate(run['iterations']):
            tape_holder.update(tape=row, label=prefix+f'/step{i}')
            out = tr.step()
            check_state(prefix+f'/step{i}', tr, row['state'], replay)
            if i == 1:
                saved = json.loads(json.dumps(tr.checkpoint(), allow_nan=False))
                tr.restore(saved)
                check(prefix+'/checkpoint_roundtrip', tf.constant(tr.checkpoint()['checkpoint_hash'] == saved['checkpoint_hash']), True, exact=True)
        print(json.dumps({'completed': prefix, 'failures_so_far': sum(not c['passed'] for c in checks)}), flush=True)

    result = {'status': 'passed' if all(c['passed'] for c in checks) else 'failed',
        'check_count': len(checks), 'failed': [c for c in checks if not c['passed']], 'checks': checks,
        'baseline_fresh_loss_ratio': baseline_ratio, 'dtype': args.dtype, 'jit_compile': args.jit,
        'callback_jit_compile': args.compile_callback,
        'tf32': args.tf32, 'gpu_intentionally_hidden': args.gpu is None, 'memory_policy': memory,
        'wall_seconds': time.monotonic()-start, 'environment': sys.executable,
        'source_sha256': hashlib.sha256((ROOT/'bayesfilter/inference/neutra_fab.py').read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'limits': ['finite valid target domain', 'alpha=2 linear AIS without resampling',
            'identity mass', 'canonical IAF callback', 'no q20 learned-map claim']}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'check_count', 'wall_seconds']}), flush=True)
    return 0 if result['status'] == 'passed' else 1

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    p.add_argument('--jit', action='store_true')
    p.add_argument('--gpu')
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--compile-callback', action='store_true')
    sys.exit(run(p.parse_args()))
