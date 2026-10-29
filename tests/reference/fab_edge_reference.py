"""Actual-author boundary and native RNG checks; CPU independent reference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import argparse
import json
import math
from pathlib import Path
import sys
import time

def run(args):
    start = time.monotonic()
    sys.path.insert(0, str(args.author))
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from fabjax.sampling.base import Point
    from fabjax.sampling.point_is_valid import default_point_is_valid_fn
    from fabjax.sampling.smc import replace_invalid_samples_with_valid_ones
    from fabjax.buffer.prioritised_buffer import build_prioritised_buffer
    from fabjax.train.fab_without_buffer import fab_loss_smc_samples
    from fabjax.train.fab_with_buffer import fab_loss_buffer_samples_fn
    fixture = json.loads(args.fixture.read_text())
    ref = json.loads(args.reference.read_text())
    x = jnp.asarray(fixture['x'])
    r = ref['density']
    point = Point(x, jnp.asarray(r['log_q']), jnp.asarray(r['log_p']), jnp.asarray(r['grad_q']), jnp.asarray(r['grad_p']))
    bad = point._replace(log_q=point.log_q.at[1].set(jnp.nan), x=point.x.at[2, 0].set(jnp.nan))
    valid = jax.vmap(default_point_is_valid_fn)(bad)
    key = jax.random.PRNGKey(517)
    # Preserve the upstream probability dtype as well as its values: bool p is
    # internally promoted differently from an explicitly FP64 p by JAX choice.
    probabilities_valid = jnp.where(valid, jnp.ones_like(valid), jnp.zeros_like(valid))
    replacements = jax.random.choice(key, jnp.arange(8), p=probabilities_valid, shape=(8,))
    replaced = replace_invalid_samples_with_valid_ones(bad, key, default_point_is_valid_fn)
    buffer = build_prioritised_buffer(x.shape[1], 16, 16)
    bx = jnp.concatenate([x, x+1])
    bw = jnp.linspace(-.8, 1.3, 16)
    bq = jnp.linspace(-1.8, -3.3, 16)
    bs = buffer.init(bx, bw, bq)
    new_x = (x-1).at[2, 0].set(jnp.nan)
    new_w = jnp.linspace(1., 2., 8).at[1].set(jnp.nan)
    new_q = jnp.linspace(-2., -4., 8).at[3].set(jnp.inf)
    after_add = buffer.add(new_x, new_w, new_q, bs)
    adjustments = jnp.asarray([.1, .3, jnp.inf, jnp.nan, -.2, .3, .7, -.8])
    log_q = jnp.linspace(-3., -5., 8)
    after_adjust = buffer.adjust(log_q, adjustments, jnp.arange(8), after_add)
    def buf(s):
        return {'x': s.data.x, 'log_w': s.data.log_w, 'log_q_old': s.data.log_q_old, 'index': s.current_index}
    # Use the actual loss functions with log_q as the trainable parameter. This
    # isolates normalization and stopped correction derivatives at extreme weights.
    q = jnp.asarray(r['log_q'])
    q_fn = lambda params, _: params
    extremes = []
    for values in [[-10000., -1000., -100., -1., 0., 1., 100., 10000.], [0.]*8]:
        value, grad = jax.value_and_grad(fab_loss_smc_samples)(q, x, jnp.asarray(values), q_fn)
        extremes.append({'weights': values, 'value': value, 'gradient': grad})
    old = q+jnp.asarray([-1000., -100., -1., 0., .1, 1., 100., 1000.])
    (rv, (ra, rq)), rg = jax.value_and_grad(fab_loss_buffer_samples_fn, has_aux=True)(q, x, old, 2., q_fn, 2.)

    n = 65536
    probabilities = jnp.asarray([1., 2., 3., 4.]) / 10
    rb = build_prioritised_buffer(2, 4, 4)
    rs = rb.init(jnp.zeros((4, 2)), jnp.log(probabilities), jnp.zeros(4))
    def sample(k):
        indices = rb.sample(k, rs, 2)[2]
        return jnp.min(indices)*4+jnp.max(indices)
    codes = jax.jit(jax.vmap(sample))(jax.random.split(jax.random.PRNGKey(31051), n))
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
    frequencies = [jnp.mean(codes == i*4+j) for i, j in pairs]
    # Analytic unordered Plackett-Luce pair probabilities.
    law = [probabilities[i]*probabilities[j]*(1/(1-probabilities[i])+1/(1-probabilities[j])) for i, j in pairs]
    normals = jax.random.normal(jax.random.PRNGKey(31052), (n,))
    uniforms = jnp.exp(-jax.random.exponential(jax.random.PRNGKey(31053), (n,)))
    rng = {'n': n, 'pairs': pairs, 'pair_frequencies': frequencies, 'pair_law': law,
        'normal_cdf': [jnp.mean(normals <= a) for a in [-1., 0., 1.]],
        'uniform_cdf': [jnp.mean(uniforms <= a) for a in [.1, .5, .9]], 'seeds': [31051, 31052, 31053]}
    result = {'replacements': replacements, 'replaced': replaced._asdict(),
        'buffer_initial': buf(bs), 'buffer_after_add': buf(after_add), 'buffer_after_adjust': buf(after_adjust),
        'extreme_loss': extremes, 'extreme_replay': {'old_q': old, 'loss': rv, 'adjustment': ra, 'gradient': rg},
        'rng': rng, 'wall_seconds': time.monotonic()-start, 'gpu_intentionally_hidden': True}
    def clean(v):
        if hasattr(v, 'tolist'): return clean(v.tolist())
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)): return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v): return None
        return v
    with args.output.open('x') as f:
        json.dump(clean(result), f, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.output), 'wall_seconds': result['wall_seconds']}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['author', 'fixture', 'reference', 'output']: p.add_argument('--'+name, required=True, type=Path)
    run(p.parse_args())
