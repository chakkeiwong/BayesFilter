#!/usr/bin/env python3
"""Bounded fresh-data representation diagnostic, not a production evaluator.

All numerical work uses TensorFlow. The diagnostic incoming Gaussian is
explicitly different from the recursive TT incoming law in the full master.
"""
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
PLAN = ROOT / 'docs/plans/observation-tt-pair-block-remedy-20260914.md'


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, 'numpy'):
        x = x.numpy()
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


def write(path, x):
    path.write_text(json.dumps(plain(x), indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', required=True)
    p.add_argument('--fixture-source', required=True)
    p.add_argument('--wall-budget-seconds', type=int, default=650)
    args = p.parse_args()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('The planned GPU is not visible')
    growth = []
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
        value = tf.config.experimental.get_memory_growth(device)
        if not value:
            raise RuntimeError('Memory growth was not established')
        growth.append(dict(name=device.name, memory_growth=value))
    from bayesfilter.highdim import observation_guided_tt_tf as lib
    from bayesfilter.highdim import pair_block_tt_tf as pair
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal
    D = tf.float64
    manifest = dict(command=[sys.executable, *sys.argv], plan=str(PLAN), result=str(out/'result.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, environment=sys.prefix,
        tensorflow=tf.__version__, dtype='float64', gpu_memory_policy=dict(mode='memory_growth', devices=growth),
        cpu_only=False, jit_compile=True, status='RUNNING',
        source_hashes={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
          [Path(__file__).resolve(), ROOT/'bayesfilter/highdim/pair_block_tt_tf.py',
           ROOT/'bayesfilter/highdim/observation_guided_tt_tf.py',
           ROOT/'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py']},
        setup_exception='TF simulation, SGQF, polynomial evaluation and postfit spectra outside XLA; fitter and conditional sampler compiled',
        seeds=dict(calibration=2026091401, downstream=2026091402, training=48100, audit=58100))
    write(out/'run_manifest.json', manifest)
    result = dict(status='RUNNING', dimensions={}, ranking='descriptive only', classification='extension_or_invention')
    def budget():
        if time.monotonic()-start > args.wall_budget_seconds:
            raise TimeoutError('diagnostic budget exhausted')
    try:
        # Explicit CPU/GPU parity of the actual compiled conditional consumer.
        cores = (tf.constant([[[[1.], [.2]], [[.3], [.5]]]], D),)
        v = tf.constant([[-.7], [.9]], D)
        uniform = tf.constant([[.2], [.8]], D)
        sampler = pair.compiled_pair_sampler([c.shape for c in cores], True)
        parity = []
        for device in ('/CPU:0', '/GPU:0'):
            with tf.device(device):
                parity.append(sampler(tuple(tf.identity(c) for c in cores), tf.identity(v),
                    tf.constant(.1, D), tf.zeros([2], D), tf.identity(uniform), tf.zeros([2, 1], D)))
        gap = tf.maximum(tf.reduce_max(tf.abs(parity[0][0]-parity[1][0])),
                         tf.reduce_max(tf.abs(parity[0][1]-parity[1][1])))
        if float(gap) > 1e-9 or not all(bool(x[2]['finite']) for x in parity):
            raise ValueError('CPU/GPU consumer parity failed')
        write(out/'cpu_gpu_parity.json', dict(status='PASS', maximum_absolute_gap=gap, cpu_role='explicit tiny reference'))

        source = json.loads(Path(args.fixture_source).read_text())
        fixtures = {}
        for partition, seed in [('calibration', 2026091401), ('downstream', 2026091402)]:
            fixture = dict(beta=source['beta'], sigma=source['sigma'], dimensions={},
                observation_seed=seed, source_sha256=hashlib.sha256(Path(args.fixture_source).read_bytes()).hexdigest(),
                provenance='Frozen model parameters only; fresh TF stateless simulations, independent partitions')
            for d in (1, 2, 4):
                A = tf.constant(source['dimensions']['1' if d == 1 else '4']['A'], D)[:d, :d]
                Q = source['sigma']**2 * tf.eye(d, dtype=D)
                P = tf.identity(Q)
                for _ in range(256):
                    P = A @ P @ tf.transpose(A) + Q
                L = tf.linalg.cholesky(P)
                x = tf.linalg.matvec(L, tf.random.stateless_normal([d], [seed, d], dtype=D))
                ys = []
                for t in range(20):
                    if t:
                        x = tf.linalg.matvec(A, x) + source['sigma']*tf.random.stateless_normal([d], [seed+d, t], dtype=D)
                    ys.append(source['beta']*tf.exp(.5*x)*tf.random.stateless_normal([d], [seed+10+d, t], dtype=D))
                fixture['dimensions'][str(d)] = dict(A=A, P0=P, observations=tf.stack(ys))
            write(out/(partition+'_fixture.json'), fixture)
            fixtures[partition] = fixture

        for d in (2, 4):
            budget()
            data = fixtures['calibration']['dimensions'][str(d)]
            model = lib.SVModel(data['A'], data['P0'], source['beta'], source['sigma'])
            guide, _ = lib.build_guide_path(model, data['observations'][:2])
            previous, current = guide[0][1], guide[1][1]
            y = data['observations'][1]
            def target(row):
                u, v = row[:, :, 0], row[:, :, 1]
                x, z = current.forward(u), previous.forward(v)
                return (previous.log_prob(z)+model.transition_log_prob(x, z)+model.observation_log_prob(x, y)
                    +current.logdet+previous.logdet-_log_standard_normal(u)-_log_standard_normal(v))
            def grouped_target(row):
                return target(tf.stack([row[:, :d], row[:, d:]], -1))
            def interleaved_target(row):
                return target(tf.reshape(row, [-1, d, 2]))
            candidates = []
            # Identical scalar fitter for the two orders isolates its ordering change.
            for order, fun in [('grouped', grouped_target), ('interleaved', interleaved_target)]:
                budget()
                print(f'd={d} scalar={order}', flush=True)
                c, info = lib.fit_amplitude(fun, 2*d, 48100+d, rank=3, rows=1024, sweeps=4, jit_compile=True)
                candidates.append(dict(name='scalar_'+order, cores=c, info=info, rank=3, sweeps=4, steps=128, weighted=False))
            for weighted in (False, True):
                for rank in (2, 3, 4):
                    budget()
                    print(f'd={d} pair rank={rank} weighted={weighted}', flush=True)
                    row_sampler = (lambda count, seed: lib.joint_sgqf_row_sampler(model, current, previous, count, seed)) if weighted else None
                    c, info = lib.pair_fit_from_log_target(target, d, 48100+d, rows=1024, rank=rank,
                        sweeps=4, proximal_steps=128, penalty=-1, row_sampler=row_sampler, jit_compile=True)
                    candidate = dict(name=f'pair_r{rank}_weighted{int(weighted)}', cores=c, info=info,
                        rank=rank, sweeps=4, steps=128, weighted=weighted)
                    candidates.append(candidate)
            baseline = next(c for c in candidates if c['name']=='pair_r3_weighted1')
            if float(baseline['info']['fit']['kkt_residual']) > .001:
                budget()
                print(f'd={d} bounded_solver_repair', flush=True)
                c, info = lib.pair_fit_from_log_target(target, d, 48100+d, rows=1024, rank=3,
                    sweeps=8, proximal_steps=256, penalty=-1,
                    row_sampler=lambda count, seed: lib.joint_sgqf_row_sampler(model, current, previous, count, seed), jit_compile=True)
                candidates.append(dict(name='pair_r3_weighted1_repair', cores=c, info=info,
                    rank=3, sweeps=8, steps=256, weighted=True))
            eligible = [c for c in candidates if c['name'].startswith('pair') and c['weighted']]
            chosen = min(eligible, key=lambda c: float(c['info']['validation_relative_rms']))
            selected = {k: chosen[k] for k in ('name', 'rank', 'sweeps', 'steps')}
            # Selection is frozen before this common independent audit exists.
            write(out/f'd{d}_selection.json', dict(**selected, criterion='validation_relative_rms_only', audit_used=False))
            audit = tf.random.stateless_normal([8192, d, 2], [58100+d, 0], dtype=D)
            logH2 = target(audit)
            scale = tf.reduce_logsumexp(logH2)-tf.math.log(tf.constant(8192., D))
            H = tf.exp(.5*(logH2-scale))
            for c in candidates:
                if c['name'].startswith('pair'):
                    h = pair.evaluate_pair_cores(c['cores'], audit)
                else:
                    rows = (tf.concat([audit[:, :, 0], audit[:, :, 1]], 1) if c['name']=='scalar_grouped'
                            else tf.reshape(audit, [-1, 2*d]))
                    h = lib.evaluate_cores(c['cores'], lib.features(rows, 3))
                h *= tf.exp(.5*(c['info']['log_target_scale']-scale))
                c['common_audit_relative_rms'] = tf.sqrt(tf.reduce_sum((h-H)**2)/tf.reduce_sum(H**2))
                write(out/f'd{d}_{c["name"]}.json', c)
            result['dimensions'][str(d)] = dict(selected=selected, incoming_law='SGQF Gaussian, diagnostic one-step target',
                candidates=[{k:v for k,v in c.items() if k!='cores'} for c in candidates])
            write(out/'result.json', result)
        selected = result['dimensions']['4']['selected']
        write(out/'selected_configuration.json', dict(rank=selected['rank'], pair_sweeps=selected['sweeps'],
            pair_proximal_steps=selected['steps'], reason='d4 fresh validation selection frozen before downstream observations are evaluated',
            training_seed=64100, reference_seed=74100, particle_seeds=[2101, 2102, 2103, 2104]))
        result['status'] = 'EXECUTED'
        manifest['status'] = 'EXECUTED'
    except Exception:
        manifest['status'] = 'FAILED'
        (out/'traceback.log').write_text(traceback.format_exc())
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic()-start
        write(out/'run_manifest.json', manifest)
        write(out/'result.json', result)
    print(json.dumps(dict(status=manifest['status'], wall_seconds=manifest['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
