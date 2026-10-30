#!/usr/bin/env python3
"""Diagnostic step-size convergence for the SAME finite SIR value and score."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ledh_sir_no_oracle_tuning as tuning

STEPS = (1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8)
PLAN = tuning.ROOT / 'docs/plans/ledh-sir-score-localization-20261006.md'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    args = p.parse_args()
    used = sum(tuning.read(f)['wall_seconds'] for f in
               (tuning.OUT / 'attempts').glob('*/attempt.json'))
    for directory in tuning.OUT.glob('score-localization-*'):
        record = directory / 'attempt.json'
        if not record.exists():
            record = directory / 'result.json'
        if record.exists():
            used += tuning.read(record).get('wall_seconds', 0.)
    allowance = min(3600., tuning.BUDGET - used)
    if allowance <= 0:
        raise RuntimeError('original campaign budget exhausted')
    args.output.mkdir(exist_ok=False)
    tick = time.monotonic()
    # Failed attempts consume the same campaign budget as completed ones.
    import atexit
    atexit.register(lambda: tuning.write(args.output/'attempt.json', dict(
        wall_seconds=time.monotonic()-tick,
        complete=(args.output/'result.json').exists())))
    tf, memory = tuning.prepare_tf(args.device)
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec, trace_kernel
    from bayesfilter.highdim import sqmc_campaign_tf as common
    spec = NonlinearSQMCSpec('sir_d18')
    theta = spec.default_theta(tf.float64)
    tuning.write(args.output/'manifest.json', dict(
        **tuning.source_record(), command=sys.argv, python=sys.executable,
        tensorflow=tf.__version__, device=args.device, cpu_only=args.device=='cpu',
        memory_policy=memory, dtype='float64', jit_compile=True, tf32=False,
        steps=STEPS, total_budget_seconds=tuning.BUDGET,
        prior_worker_seconds=used, local_budget_seconds=allowance,
        design_seed=tuning.SEEDS['confirmation'][0],
        observation_seed=tuning.DATA_SEEDS['confirmation'],
        controls=tuning.BASE, particles=tuning.N, horizons=tuning.HORIZONS,
        plan=str(PLAN), results=str(args.output/'result.json'),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='local_finite_program_derivative_only'))
    results = []
    for h in tuning.HORIZONS:
        if time.monotonic()-tick >= allowance:
            raise RuntimeError('localization budget exhausted')
        data = spec.observations(h, tuning.DATA_SEEDS['confirmation'],
                                 dtype=tf.float64, jit_compile=True)
        inputs = common.random_inputs(tuning.ROUTE, tuning.SEEDS['confirmation'][0],
             tuning.N, spec.dimension, h, tf.float64, jit_compile=True)
        kernel = trace_kernel(spec, tuning.ROUTE, tuning.BASE, tuning.N, h,
             tf.float64, jit_compile=True, reset_design_kind=tuning.DESIGN,
             dynamic_direction=True)
        direction0 = tf.one_hot(0, 3, dtype=tf.float64)
        def call(t):
            value, score, trace = kernel(t, *inputs, data, direction0)
            flags = [v for row in trace for k,v in row.items()
                     if v.dtype == tf.bool and (k.endswith('_valid') or k.endswith('_finite'))]
            valid = bool(tf.reduce_all(tf.stack(flags))) and math.isfinite(float(value))
            return float(value), valid
        base, base_valid = call(theta)
        public, score, public_valid = common.value_and_score(spec, tuning.ROUTE,
            tuning.BASE, theta, data, tuning.SEEDS['confirmation'][0], tuning.N,
            inputs=inputs, jit_compile=True, reset_design_kind=tuning.DESIGN)
        public_gap = abs(base-float(public))
        valid = base_valid and bool(public_valid) and public_gap <= 1e-10*(1.+abs(base))
        rows = []
        for j in range(3):
            direction = tf.one_hot(j, 3, dtype=tf.float64)
            estimates = []
            for step in STEPS:
                if time.monotonic()-tick >= allowance:
                    raise RuntimeError('localization budget exhausted')
                plus, vp = call(theta+step*direction)
                minus, vm = call(theta-step*direction)
                fd = (plus-minus)/(2.*step)
                discrepancy = abs(fd-float(score[j]))
                relative = discrepancy/(1.+abs(float(score[j])))
                row = dict(horizon=h, coordinate=j, step=step, plus=plus, minus=minus,
                    finite_difference=fd, analytical=float(score[j]),
                    absolute_error=discrepancy, scaled_error=relative,
                    valid=vp and vm and math.isfinite(fd))
                estimates.append(row)
                with (args.output/'evaluations.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            agreement = any(a['valid'] and b['valid'] and
                            a['scaled_error'] <= 1e-3 and b['scaled_error'] <= 1e-3
                            for a,b in zip(estimates, estimates[1:]))
            rows.append(dict(coordinate=j, analytical=float(score[j]),
                             agrees_at_adjacent_steps=agreement, ladder=estimates))
        repeat, repeat_valid = call(theta)
        repeat_gap = abs(repeat-base)
        valid = valid and repeat_valid and repeat_gap <= 1e-10*(1.+abs(base))
        result = dict(horizon=h, value=base, score=[float(v) for v in score],
             public_gap=public_gap, repeat_gap=repeat_gap,
             input_sha256=hashlib.sha256(bytes(tf.io.serialize_tensor(data).numpy())).hexdigest(),
             valid=valid, coordinates=rows,
             all_coordinates_agree=valid and all(r['agrees_at_adjacent_steps'] for r in rows))
        results.append(result)
        tuning.write(args.output/'horizons.json', results)
        print(json.dumps(dict(horizon=h,valid=valid,
           all_coordinates_agree=result['all_coordinates_agree'],
           minimum_scaled_errors=[min(v['scaled_error'] for v in r['ladder']) for r in rows])),flush=True)
    tuning.write(args.output/'result.json', dict(complete=True, wall_seconds=time.monotonic()-tick,
        prior_worker_seconds=used, budget_seconds=tuning.BUDGET, horizons=results,
        all_coordinates_agree=all(r['all_coordinates_agree'] for r in results),
        nonclaim='Local same-scalar derivative agreement is not oracle score accuracy.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
