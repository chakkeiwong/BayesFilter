"""Bounded diagnostic replay of saved pair fits; no fitting or selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--dimension', type=int, default=4)
    args = p.parse_args()
    start = time.monotonic()
    out, source = Path(args.output_root), Path(args.source)
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((source/'run_manifest.json').read_text())
    config = old['config']
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    manifest = dict(command=sys.argv, plan='docs/plans/observation-tt-pair-sampler-recovery-20260914.md',
        result=str(out/'result.json'), started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source=str(source), seeds=config['seeds'], cpu_only=False,
        gpu_intentionally_hidden=False, jit_compile=True, numerical_dtype='float64',
        environment={k: os.environ.get(k) for k in old['environment']}, status='STARTING')
    def write(name, value):
        (out/name).write_text(json.dumps(driver.plain(value), indent=2)+'\n')
    import run_observation_aware_tt_complete as driver
    paths = [Path(__file__).relative_to(ROOT), Path(manifest['plan']),
             source/'run_manifest.json', Path(config['fixture_json']),
             source/f'd{args.dimension}/tt_pair_block_proposals.json']
    paths.extend(Path(x) for x in old['source_hashes'] if x.endswith('.py'))
    manifest['source_hashes'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    write('run_manifest.json', manifest)
    try:
        import tensorflow as tf
        D = tf.float64
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('trusted GPU replay requires a visible physical GPU')
        devices = []
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            growth = tf.config.experimental.get_memory_growth(gpu)
            if not growth:
                raise RuntimeError('GPU memory growth not verified')
            devices.append(dict(name=gpu.name, memory_growth=growth,
                                details=tf.config.experimental.get_device_details(gpu)))
        manifest.update(gpu_memory_policy=dict(mode='memory_growth', devices=devices),
            tensorflow=tf.__version__, python=sys.version, conda_prefix=sys.prefix,
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(), status='RUNNING')
        write('run_manifest.json', manifest)
        from bayesfilter.highdim import observation_guided_tt_tf as lib
        driver.tf, driver.D, driver.lib = tf, D, lib
        fixture = json.loads(Path(config['fixture_json']).read_text())
        data = fixture['dimensions'][str(args.dimension)]
        model = lib.SVModel(tf.constant(data['A'], D), tf.constant(data['P0'], D), fixture['beta'], fixture['sigma'])
        observations = tf.constant(data['observations'][:config['horizon']], D)
        saved = json.loads((source/f'd{args.dimension}/tt_pair_block_proposals.json').read_text())
        steps = []
        for s in saved:
            current = lib.Chart(tf.constant(s['mean'], D), tf.constant(s['factor'], D))
            condition = (lib.Chart(tf.constant(s['condition_mean'], D), tf.constant(s['condition_factor'], D))
                         if s['condition_mean'] is not None else None)
            cores = tuple(tf.constant(c, D) for c in s['cores'])
            steps.append(lib.PairTTStep(cores, current, condition, tf.constant(s['tau'], D), {}, s['time'], None))
        guide = [(None, s.current_chart) for s in steps]
        original = lib.sample_pair_tt_step
        progress = []
        def observed(step, previous, seed, jit):
            try:
                result = original(step, previous, seed, jit)
            except lib.PairConditionalSamplingError as exc:
                write('failure_inputs.json', dict(step=saved[step.time_index], previous=previous,
                    v=step.conditioning_chart.inverse(previous), seed=seed,
                    mixture_uniform=tf.random.stateless_uniform([config['particles']], [seed, 0], dtype=D),
                    uniforms=tf.random.stateless_uniform([config['particles'], args.dimension], [seed, 1],
                        minval=lib.EPS, maxval=1.-lib.EPS, dtype=D),
                    normal_noise=tf.random.stateless_normal([config['particles'], args.dimension], [seed, 2], dtype=D)))
                write('failure_diagnostics.json', dict(time=exc.time_index, seed=exc.seed,
                                                     diagnostics=exc.diagnostics))
                raise
            progress.append(dict(time=step.time_index, seed=seed, diagnostics=result[2]))
            write('progress.json', progress)
            return result
        lib.sample_pair_tt_step = observed
        runs = []
        for seed in config['seeds']:
            run, _ = driver.particle_filter(model, observations, guide, steps, 'tt_pair_block',
                                           config['particles'], seed*100, True)
            runs.append(run)
            write('completed_runs.json', runs)
        manifest['status'] = 'COMPLETED'
        write('result.json', dict(status='COMPLETED', runs=runs, no_refitting=True))
    except Exception as exc:
        manifest['status'] = 'REPRODUCED_GUARD_FAILURE' if type(exc).__name__ == 'PairConditionalSamplingError' else 'FAILED'
        write('result.json', dict(status=manifest['status'], error_type=type(exc).__name__, error=str(exc)))
        (out/'traceback.log').write_text(traceback.format_exc())
    finally:
        manifest.update(wall_seconds=time.monotonic()-start, completed_utc=datetime.now(timezone.utc).isoformat())
        if 'tf' in locals() and manifest.get('gpu_memory_policy'):
            manifest['gpu_allocator'] = tf.config.experimental.get_memory_info('GPU:0')
        write('run_manifest.json', manifest)
        print(json.dumps({k: manifest[k] for k in ['status', 'wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
