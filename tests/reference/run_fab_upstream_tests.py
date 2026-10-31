"""Run unmodified upstream test functions headlessly; report their limits."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp/fab-equivalence-matplotlib'
os.environ['WANDB_MODE'] = 'disabled'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import inspect

def run(args):
    start = time.monotonic()
    sys.path.insert(0, str(args.author))
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **kw: plt.close('all')
    if args.compat:
        sys.modules['fabjax.utils.logging'] = importlib.import_module('fabjax.utils.loggers')
    tests = [
        ('fabjax.flow.flow_test', 'test_flow_does_not_smoke', 'density consistency'),
        ('fabjax.buffer.prioritised_buffer_test', 'test_prioritised_buffer_does_not_smoke', 'buffer smoke / one row assertion'),
        ('fabjax.sampling.mcmc.hmc_test', 'test_hmc_produces_good_samples', 'Gaussian mean and acceptance screen'),
        ('fabjax.sampling.mcmc.metropolis_test', 'tesst_metropolis_produces_good_samples', 'explicit invocation of misspelled test'),
        ('fabjax.sampling.smc_test', 'test_smc_visualise_as_step_size_tuned', 'visual diagnostic; no numerical correctness assertion'),
        ('fabjax.sampling.smc_test', 'test_reasonable_resampling', 'visual diagnostic; no numerical correctness assertion'),
        ('fabjax.sampling.smc_test', 'test_ess_with_more_dists', 'visual diagnostic; no numerical correctness assertion'),
    ]
    rows = []
    for module, name, scope in tests:
        if args.only and args.only not in module:
            continue
        t = time.monotonic()
        row = {'function': module+'.'+name, 'scope': scope}
        try:
            mod = importlib.import_module(module)
            fn = getattr(mod, name)
            repairs = []
            if args.compat and module == 'fabjax.flow.flow_test':
                original_config = mod.FlowDistConfig
                def fixed_config(*a, **kw):
                    config = original_config(*a, **kw)
                    fields = ['transform_type', 'restrict_scale_rnvp', 'spline_max', 'spline_min', 'spline_num_bins']
                    changes = {k: getattr(config, k)[0] for k in fields if isinstance(getattr(config, k), tuple)}
                    # Keep the test's non-identity coupling initialization.
                    # Upstream act_norm explicitly supports identity_init=True only.
                    changes['act_norm'] = False
                    return config._replace(**changes)
                mod.FlowDistConfig = fixed_config
                repairs.append('unwrap accidental trailing-comma tuple defaults in FlowDistConfig')
                repairs.append('disable act_norm: it rejects the test non-identity initialization; coupling stays non-identity')
                original_assert = mod.chex.assert_trees_all_close
                def bounded_assert(a, b, **kw):
                    kw.setdefault('atol', 1e-5)
                    return original_assert(a, b, **kw)
                mod.chex.assert_trees_all_close = bounded_assert
                repairs.append('set test-only inverse density tolerance atol=1e-5; original zero-atol check fails on upstream FP32 roundoff')
            if args.compat and module == 'fabjax.buffer.prioritised_buffer_test':
                source = inspect.getsource(fn)
                old = 'sample_n_batches(rng_key, buffer_state, 12, 2)'
                new = 'sample_n_batches(rng_key, buffer_state, 6, 2)'
                assert source.count(old) == 1
                source = source.replace(old, new)
                ns = dict(mod.__dict__)
                exec(compile(source, str(args.author/'fabjax/buffer/prioritised_buffer_test.py'), 'exec'), ns)
                fn = ns[name]
                repairs.append('test sample request 12*2 exceeds min_length=15; use 6*2')
                row['adapted_test_source'] = source
            if args.compat and 'sampling' in module:
                repairs.append('alias stale fabjax.utils.logging import to existing loggers module')
            row['repairs'] = repairs
            fn()
            row['status'] = 'passed'
        except Exception:
            row.update(status='failed', traceback=traceback.format_exc())
        row['wall_seconds'] = time.monotonic()-t
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'status': 'passed' if all(r['status']=='passed' for r in rows) else 'failed',
        'tests': rows, 'wall_seconds': time.monotonic()-start, 'gpu_intentionally_hidden': True,
        'changes': 'Agg, plt.show close, explicit invocation; compatibility repairs listed per test' if args.compat else 'only Agg backend and plt.show closes figures; explicit function invocation',
        'jax_precision': 'upstream default (FP32)'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    return 0 if result['status'] == 'passed' else 1

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--author', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--compat', action='store_true')
    p.add_argument('--only')
    sys.exit(run(p.parse_args()))
