"""Post-run CPU diagnostic reader; cannot influence sampling or adaptation."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/benchmarks'))
import benchmark_nuts_xla as base
import benchmark_stan_transplant as common
from .worker import fixtures
from .api import StaticChainKernel
from .. import stan_nuts as core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('runs', type=Path, nargs='+')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    conf = base.parser().parse_args(['--output', str(args.output), '--device', 'cpu'])
    started = time.monotonic(); tf, _, memory = base.setup(conf)
    report = dict(command=sys.argv, cpu_gpu_intentionally_hidden=True, memory_policy=memory,
        interpretation='short-chain descriptive diagnostics; no sampler/default ranking', cases=[])
    # The long CPU fits may accumulate compiler-rounding differences. Compare
    # both graph layouts at the same state/step/seed instead of attributing
    # divergent adaptive trajectories to RNG without a discriminating check.
    report['controlled_cpu_layout_checks'] = []
    for name in ('gaussian', 'banana', 'funnel'):
        d, model, _, data, _, _, _ = fixtures(tf, name)
        cfg = core.Config(d, chains=4, depth=6, adapt=False, warmup=0, draws=1)
        def build(kind, cfg, model):
            @tf.function(input_signature=[tf.TensorSpec([4, d], tf.float64),
                tf.TensorSpec([4], tf.float64), tf.TensorSpec([4, 2], tf.int32)],
                autograph=False, jit_compile=True)
            def step(q, e, seeds):
                k = kind(tf, model, cfg); m = tf.ones_like(q)
                return k.transition(q, e, m, m, seeds)
            return step
        old, new = build(core.Kernel, cfg, model), build(StaticChainKernel, cfg, model)
        maximum = 0.
        for rep in range(8):
            q = tf.reshape(tf.sin(tf.cast(tf.range(4*d)+rep, tf.float64)), [4, d])
            e = tf.constant([.05, .1, .25, .5], tf.float64)
            seed = tf.constant([[31+i, 17*rep] for i in range(4)], tf.int32)
            a, b = old(q, e, seed), new(q, e, seed)
            for x, y in zip(tf.nest.flatten(a), tf.nest.flatten(b)):
                if x.dtype.is_floating:
                    maximum = max(maximum, float(tf.reduce_max(tf.abs(x-y)/(1.+tf.abs(y)))))
                elif not bool(tf.reduce_all(x == y)):
                    raise RuntimeError('CPU controlled branch/RNG mismatch')
        if maximum > 2e-12: raise RuntimeError('CPU controlled transition mismatch')
        report['controlled_cpu_layout_checks'].append(dict(target=name, transitions=32,
            discrete_choices_exact=True, maximum_scaled_error=maximum))
    for run in args.runs:
        metadata = json.loads((run/'results.json').read_text())
        for item in metadata['targets']:
            name = item['name']; d, model, reference, *_ = fixtures(tf, name)
            target = SimpleNamespace(dimension=d, to_reference=reference)
            for engine in ['map', 'static']+(['native'] if item['native'] else []):
                def read(label):
                    return tf.io.parse_tensor((run/name/(engine+'-'+label+'.tensor')).read_bytes(), out_type=tf.float64)
                samples, telemetry = read('draws'), read('telemetry')
                diagnostics = common.assess(tf, target, samples, telemetry)
                divergence_count = int(tf.reduce_sum(tf.cast(telemetry[..., 4] != 0, tf.int32)))
                case = dict(run=str(run), device=metadata['device'], target=name, engine=engine,
                            diagnostics=diagnostics, divergences=divergence_count,
                            efficiency_rank_eligible=diagnostics['posterior_screen_pass'] and divergence_count == 0)
                if name == 'funnel':
                    case['neck_fraction_v_lt_minus3'] = float(tf.reduce_mean(tf.cast(samples[..., 0] < -3., tf.float64)))
                report['cases'].append(case)
                common.write(args.output/'diagnostics.json', report)
    report['cross_execution_checks'] = []
    gpu_runs = [run for run in args.runs if json.loads((run/'results.json').read_text())['device'] == 'gpu']
    if len(gpu_runs) > 1:
        for run in gpu_runs[1:]:
            for name in ('gaussian', 'banana', 'funnel', 'regression'):
                a_path, b_path = gpu_runs[0]/name/'static-draws.tensor', run/name/'static-draws.tensor'
                if not a_path.exists() or not b_path.exists(): continue
                a = tf.io.parse_tensor(a_path.read_bytes(), out_type=tf.float64)
                b = tf.io.parse_tensor(b_path.read_bytes(), out_type=tf.float64)
                report['cross_execution_checks'].append(dict(baseline=str(gpu_runs[0]), candidate=str(run),
                    target=name, draws_exact=bool(tf.reduce_all(a == b)),
                    maximum_scaled_difference=float(tf.reduce_max(tf.abs(a-b)/(1.+tf.abs(a))))))
    report['cpu_rounding_investigation'] = []
    for run in args.runs:
        if json.loads((run/'results.json').read_text())['device'] != 'cpu': continue
        for name in ('gaussian', 'banana', 'funnel', 'regression'):
            def draw_file(layout):
                return tf.io.parse_tensor((run/name/(layout+'-warmup.tensor')).read_bytes(), out_type=tf.float64)
            a, b = draw_file('map'), draw_file('static')
            diff = tf.reduce_max(tf.abs(a-b), axis=[1, 2])
            changed = tf.where(diff != 0)
            first = int(changed[0, 0]) if int(tf.size(changed)) else None
            report['cpu_rounding_investigation'].append(dict(target=name, first_changed_warmup_iteration=first,
                maximum_difference_at_first_change=float(diff[first]) if first is not None else 0.,
                interpretation='first perturbation diagnostic; does not prove all later adaptive differences'))
    report['wall_seconds'] = time.monotonic()-started
    common.write(args.output/'diagnostics.json', report)
    print(json.dumps({'cases':len(report['cases']), 'wall_seconds':report['wall_seconds']}))


if __name__ == '__main__': main()
