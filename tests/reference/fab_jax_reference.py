"""Independent CPU diagnostic. Executes pinned author FAB; no runtime imports.

The IAF callback below is a test-only mathematical evaluation, validated against
the canonical TensorFlow map before any FAB comparison is accepted. All FAB,
replay, HMC, AIS, loss and optimizer operations come from upstream dependencies.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value

def run(args):
    start = time.monotonic()
    sys.path.insert(0, str(args.author))
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import optax
    from blackjax.mcmc.metrics import gaussian_euclidean
    from fabjax.flow.flow import Flow
    from fabjax.sampling.base import Point, create_point, get_intermediate_log_prob, get_grad_intermediate_log_prob
    from fabjax.sampling.smc import build_smc, log_weight_contribution_point
    from fabjax.sampling.mcmc.hmc import build_blackjax_hmc
    from fabjax.sampling.mcmc.metropolis import build_metropolis
    from fabjax.sampling.mcmc.blackjax_hmc_rewrite import velocity_verlet, IntegratorState
    from fabjax.buffer.prioritised_buffer import build_prioritised_buffer
    from fabjax.train.fab_without_buffer import fab_loss_smc_samples, build_fab_no_buffer_init_step_fns
    from fabjax.train.fab_with_buffer import fab_loss_buffer_samples_fn, build_fab_with_buffer_init_step_fns

    fixture = json.loads(args.fixture.read_text())
    cfg = fixture['config']
    dtype = jnp.float64
    params = tuple(jnp.asarray(v, dtype) for v in fixture['params'])
    masks = [[jnp.asarray(v, dtype) for v in stage] for stage in fixture['masks']]
    d, stages = cfg['dimension'], cfg['stages']
    layers = len(masks[0])
    center = jnp.asarray(cfg['affine_center'], dtype)
    scale = jnp.asarray(cfg['affine_scale'], dtype)

    def network(p, stage, x):
        off = 2*layers*stage
        for k in range(layers):
            x = x @ (p[off+2*k]*masks[stage][k]) + p[off+2*k+1]
            if k+1 < layers:
                x = jax.nn.elu(x)
        bias = p[off+2*layers-1][:d]
        s = bias + cfg['conditional_scale_cap']*jnp.tanh((x[..., :d]-bias)/cfg['conditional_scale_cap'])
        return s, x[..., d:]

    def forward(p, z):
        x, ld = z, jnp.zeros(z.shape[:-1], dtype)
        for stage in range(stages):
            s, shift = network(p, stage, x)
            x, ld = x*jnp.exp(s)+shift, ld+jnp.sum(s, -1)
            if stage+1 < stages:
                x = x[..., ::-1]
        return x*scale+center, ld+jnp.sum(jnp.log(scale))

    def inverse(p, x):
        z = (x-center)/scale
        for stage in reversed(range(stages)):
            if stage+1 < stages:
                z = z[..., ::-1]
            y = z
            def solve(k, values):
                s, shift = network(p, stage, values)
                return values.at[..., k].set((y[..., k]-shift[..., k])*jnp.exp(-s[..., k]))
            z = jax.lax.fori_loop(0, d, solve, jnp.zeros_like(y))
        return z

    def log_q(p, x):
        z = inverse(p, x)
        return -.5*(jnp.sum(z*z, -1)+d*math.log(2*math.pi))-forward(p, z)[1]

    def sample(p, key, shape):
        return forward(p, jax.random.normal(key, (*shape, d), dtype))[0]

    flow = Flow(init=lambda key, x: params, log_prob_apply=log_q,
        sample_and_log_prob_apply=lambda p, key, shape: (sample(p, key, shape), log_q(p, sample(p, key, shape))),
        sample_apply=sample, log_prob_with_extra_apply=None,
        sample_and_log_prob_with_extra_apply=None, config=cfg, dim=d)

    # Full-support nonlinear target with independently known mixture density.
    def log_p(x):
        means = jnp.asarray([[.6, -.4, .2, .1], [-.7, .8, 1., -.3]], dtype)[:, :d]
        scales = jnp.asarray([[.8, 1.1, .9, 1.2], [1.2, .7, 1.1, .8]], dtype)[:, :d]
        diff = (x[..., None, :]-means)/scales
        terms = -.5*jnp.sum(diff*diff, -1)-jnp.sum(jnp.log(scales), -1)-d/2*math.log(2*math.pi)
        return jax.nn.logsumexp(terms+jnp.log(jnp.asarray([.4, .6], dtype)), axis=-1)

    x = jnp.asarray(fixture['x'], dtype)
    w = jnp.asarray(fixture['log_w'], dtype)
    old_q = log_q(params, x)+jnp.asarray([-3., -1., -.2, 0., .2, 1., 3., 6.], dtype)
    fresh = jax.value_and_grad(fab_loss_smc_samples)(params, x, w, log_q)
    (replay_value, replay_aux), replay_grad = jax.value_and_grad(fab_loss_buffer_samples_fn, has_aux=True)(
        params, x, old_q, 2., log_q, 2.)
    point = jax.vmap(create_point, in_axes=(0, None, None, None))(x, lambda v: log_q(params, v), log_p, True)
    result = {'schema': 'fab_author_reference.v1', 'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        'density': {'forward': forward(params, x), 'inverse': inverse(params, x), 'log_q': point.log_q,
            'grad_q': point.grad_log_q, 'log_p': point.log_p, 'grad_p': point.grad_log_p,
            'parameter_gradient': jax.grad(lambda p: jnp.sum(log_q(p, x)))(params)},
        'fresh': {'loss': fresh[0], 'gradients': fresh[1]},
        'replay_loss': {'old_q': old_q, 'loss': replay_value, 'adjustment': replay_aux[0],
            'log_q': replay_aux[1], 'gradients': replay_grad},
        'bridge': [{'beta': beta, 'value': get_intermediate_log_prob(point.log_q, point.log_p, beta, 2.),
            'score': get_grad_intermediate_log_prob(point.grad_log_q, point.grad_log_p, beta, 2.)}
            for beta in [0., .25, .5, .75, 1.]], 'trajectories': [], 'optimizer': [], 'runs': [], 'stress_mutations': []}

    momentum = jnp.asarray([[.2, -.4], [1., -.1], [-.3, .6], [.5, .7], [-1., -.2], [.8, .3], [-.5, .9], [.1, -.8]], dtype)
    if d != 2:
        momentum = jnp.linspace(-.9, .8, 8*d).reshape((8, d))
    kinetic = lambda p: .5*jnp.sum(p*p)
    integrator = velocity_verlet(lambda v: log_q(params, v), log_p, kinetic)
    state = IntegratorState(point.x, momentum, point.log_q, point.log_p, point.grad_log_q, point.grad_log_p,
        jnp.full([len(x)], .75, dtype), jnp.full([len(x)], 2., dtype))
    for leap in range(1, 4):
        state = jax.vmap(integrator, in_axes=(0, None))(state, .17)
        result['trajectories'].append({'steps': leap, 'position': state.position, 'momentum': state.momentum,
            'log_q': state.log_q, 'log_p': state.log_p})
    result['momentum'] = momentum

    for epsilon in [1e-8, 1e-3]:
        opt = optax.adam(.002, b1=.9, b2=.999, eps=epsilon)
        ps, os_ = params, opt.init(params)
        trace = []
        for factor in [1., -2., .1]:
            gradients = jax.tree_util.tree_map(lambda g: factor*g, fresh[1])
            upd, os_ = opt.update(gradients, os_, ps)
            ps = optax.apply_updates(ps, upd)
            trace.append({'params': ps, 'count': os_[0].count, 'm': os_[0].mu, 'v': os_[0].nu})
        result['optimizer'].append({'epsilon': epsilon, 'trace': trace})

    config = dict(batch_size=8, intermediate_distributions=3, leapfrog_steps=3, hmc_steps=2,
        initial_step_size=.17, learning_rate=.002, beta1=.9, beta2=.999, adam_epsilon=1e-3,
        replay_capacity=64, replay_min_size=32, updates_per_pass=2, correction_clip=2.,
        gradient_clip=.4, adapt_step_size=True, target_acceptance=.65, step_size_multiplier=1.02)

    def random_tape(smc_state, operator):
        noises, thresholds = [], []
        for stage_key in smc_state.transition_operator_state.key:
            _, subkey = jax.random.split(stage_key)
            ns, us = [], []
            for mutation_key in jax.random.split(subkey, config['hmc_steps']):
                bkeys = jax.random.split(mutation_key, config['batch_size'])
                k0, k1 = jax.vmap(lambda k: jax.random.split(k))(bkeys).transpose(1, 0, 2)
                if operator == 'hmc':
                    generator, _, _ = gaussian_euclidean(jnp.ones(d, dtype))
                    n = jax.vmap(lambda k: generator(k, jnp.zeros(d, dtype)))(k0)
                    u = jnp.log(jax.vmap(lambda k: jax.random.uniform(k, dtype=dtype))(k1))
                else:
                    n = jax.vmap(lambda k: jax.random.normal(k, (d,)))(k0)
                    u = -jax.vmap(lambda k: jax.random.exponential(k))(k1)
                ns.append(n); us.append(u)
            noises.append(jnp.stack(ns)); thresholds.append(jnp.stack(us))
        return {'noise': jnp.stack(noises), 'log_uniform': jnp.stack(thresholds)}

    def snapshot(state, replay):
        adam = state.opt_state[1][0]
        row = {'params': state.flow_params, 'count': adam.count, 'm': adam.mu, 'v': adam.nu,
            'steps': state.smc_state.transition_operator_state.step_size}
        if replay:
            b = state.buffer_state
            row['buffer'] = {'x': b.data.x, 'log_w': b.data.log_w, 'log_q_old': b.data.log_q_old,
                'index': b.current_index, 'size': config['replay_capacity'] if bool(b.is_full) else int(b.current_index)}
        return row

    for operator in ['hmc', 'metropolis']:
        if operator == 'hmc':
            transition = build_blackjax_hmc(d, n_outer_steps=2, n_inner_steps=3, init_step_size=.17,
                adapt_step_size=True, target_p_accept=.65, step_size_multiplier=1.02)
        else:
            transition = build_metropolis(d, n_steps=2, init_step_size=.17, tune_step_size=True,
                target_p_accept=.65, step_size_multiplier=1.02)
        def observed_step(*a, _step=transition.step, **kw):
            pt, ts, info = _step(*a, **kw)
            return pt, ts, {**info, 'trace_x': pt.x, 'trace_log_q': pt.log_q, 'trace_log_p': pt.log_p}
        transition = transition._replace(step=observed_step)
        smc = build_smc(transition, 3, alpha=2., use_resampling=False)
        smc_step = jax.jit(lambda xx, ss, ps: smc.step(xx, ss, lambda v: log_q(ps, v), log_p))
        def ais_record(pt, lw, ss, info, x0, ps):
            points = [Point(x0, log_q(ps, x0), log_p(x0))]
            points.extend(Point(info[f'dist{k}_trace_x'], info[f'dist{k}_trace_log_q'], info[f'dist{k}_trace_log_p']) for k in range(1, 4))
            return {'x': pt.x, 'log_w': lw, 'log_q': pt.log_q, 'steps': ss.transition_operator_state.step_size,
                'stage_positions': jnp.stack([p.x for p in points]),
                'log_weight_increments': jnp.stack([log_weight_contribution_point(p, k, smc.betas, 2.) for k, p in enumerate(points)]),
                'acceptance': jnp.asarray([info[f'dist{k}_'+('mean_acceptance_rate' if operator=='hmc' else 'mean_p_accept')] for k in range(1, 4)])}
        stress_state = smc.init(jax.random.PRNGKey(833))
        stress_steps = jnp.asarray([1.7, 2., 2.3] if operator=='hmc' else [2.5, 3., 4.])
        stress_state = stress_state._replace(transition_operator_state=
            stress_state.transition_operator_state._replace(step_size=stress_steps))
        stress_tape = random_tape(stress_state, operator)
        sp, sw, ss, si = smc_step(x, stress_state, params)
        result['stress_mutations'].append({'operator': operator, 'steps': stress_steps, **stress_tape,
            'ais': ais_record(sp, sw, ss, si, x, params)})
        optimizer = optax.chain(optax.clip_by_global_norm(.4), optax.adam(.002, eps=1e-3))
        for replay in [False, True]:
            root_key = jax.random.PRNGKey(731)
            buffer = build_prioritised_buffer(d, 64, 32)
            if replay:
                init, step = build_fab_with_buffer_init_step_fns(flow, log_p, smc, buffer, optimizer,
                    batch_size=8, n_updates_per_smc_forward_pass=2, w_adjust_clip=2.)
            else:
                init, step = build_fab_no_buffer_init_step_fns(flow, log_p, smc, optimizer, 8)
            state = init(root_key)
            row = {'operator': operator, 'replay': replay, 'config': config,
                'initialization': [], 'initial_state': snapshot(state, replay), 'iterations': []}
            if replay:
                _, key2, _, key4 = jax.random.split(root_key, 4)
                ss = smc.init(key2)
                q = lambda v: log_q(params, v)
                for k in jax.random.split(key4, 5):
                    tape = random_tape(ss, operator)
                    z = jax.random.normal(k, (8, d), dtype)
                    pt, lw, ss, info = smc_step(forward(params, z)[0], ss, params)
                    row['initialization'].append({'z': z, **tape,
                        **ais_record(pt, lw, ss, info, forward(params, z)[0], params)})
            for iteration in range(4):
                key, sample_key = jax.random.split(state.key)
                tape = random_tape(state.smc_state, operator)
                if replay:
                    _, _, indices = buffer.sample_n_batches(sample_key, state.buffer_state, 8, 2)
                    a, b = jax.random.split(sample_key)
                    c, e = jax.random.split(a)
                    gumbel = jax.random.gumbel(c, (64,))
                    order = jax.random.permutation(b, jax.random.permutation(e, jnp.arange(16)))
                    tape.update(gumbel=gumbel, order=order, indices=indices.reshape(-1))
                    key, _ = jax.random.split(key)
                    key, sample_key = jax.random.split(key)
                z = jax.random.normal(sample_key, (8, d), dtype)
                q = lambda v: log_q(state.flow_params, v)
                pt, lw, ss, info = smc_step(forward(state.flow_params, z)[0], state.smc_state, state.flow_params)
                # The actual author complete trainer, including its own AIS,
                # replay selection, loss/autodiff and optimizer, is the oracle.
                new_state, train_info = step(state)
                row['iterations'].append({'z': z, **tape,
                    'ais': ais_record(pt, lw, ss, info, forward(state.flow_params, z)[0], state.flow_params),
                    'state': snapshot(new_state, replay)})
                state = new_state
            result['runs'].append(row)
            print(json.dumps({'completed': operator, 'replay': replay, 'wall_seconds': time.monotonic()-start}), flush=True)

    result.update(wall_seconds=time.monotonic()-start, gpu_intentionally_hidden=True,
        environment=sys.executable, versions={n: importlib.metadata.version(n) for n in
            ['jax', 'jaxlib', 'optax', 'chex', 'blackjax-nightly', 'distrax', 'flax', 'numpy', 'scipy']})
    # JSON null preserves unoccupied author buffer slots; they have -inf priority.
    def clean(v):
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v): return None
        return v
    with args.output.open('x') as f:
        json.dump(clean(plain(result)), f, allow_nan=False)
        f.write('\n')
    print(json.dumps({'reference': str(args.output), 'wall_seconds': result['wall_seconds']}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--author', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
