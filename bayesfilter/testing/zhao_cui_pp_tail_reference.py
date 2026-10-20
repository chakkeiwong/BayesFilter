"""High-precision independent reference for rare PP transition overflows.

This diagnostic is invoked only by the Octave publication-reference adapter.
Equation (38), Zhao and Cui (JMLR 2024), and the pinned author PP odefun and
predator_step define the equations. The paper uses full-stage k4; the released
source uses a half-stage k4. This module exposes that difference explicitly.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import mpmath as mp


def _evaluate(theta, previous, current, *, sigma, dt, k4_fraction, digits):
    with mp.workdps(digits):
        r,s,u,v,K0,a0=map(mp.mpf,theta)
        K=90+20*K0; a=20+10*a0
        x=list(map(mp.mpf,previous)); target=list(map(mp.mpf,current))
        # Reuse the binary64 step size of the Octave callback, exactly.
        h=mp.mpf(float(dt/20)); k4=mp.mpf(k4_fraction)
        def f(z):
            transfer=z[0]*z[1]/(a+z[0])
            return [r*z[0]*(1-z[0]/K)-s*transfer,u*transfer-v*z[1]]
        for _ in range(20):
            f1=f(x); f2=f([x[j]+h*f1[j]/2 for j in range(2)])
            f3=f([x[j]+h*f2[j]/2 for j in range(2)])
            f4=f([x[j]+h*k4*f3[j] for j in range(2)])
            x=[x[j]+h*(f1[j]+2*f2[j]+2*f3[j]+f4[j])/6 for j in range(2)]
        logp=-mp.log(2*mp.pi)-2*mp.log(sigma)-sum(((target[j]-x[j])/sigma)**2 for j in range(2))/2
        if not mp.isfinite(logp): raise ValueError('Nonfinite high-precision density')
        return logp


def checked_log_transition(theta,previous,current,*,sigma=2.,dt=2.,k4_fraction=1.):
    if len(theta)!=6 or len(previous)!=2 or len(current)!=2:
        raise ValueError('Expected six chart parameters and two-dimensional states')
    if not all(math.isfinite(v) for v in [*theta,*previous,*current,sigma,dt,k4_fraction]):
        raise ValueError('Nonfinite reference input')
    if sigma<=0 or dt<=0 or k4_fraction not in (1.,.5):
        raise ValueError('Invalid transition specification')
    a=_evaluate(theta,previous,current,sigma=sigma,dt=dt,k4_fraction=k4_fraction,digits=100)
    b=_evaluate(theta,previous,current,sigma=sigma,dt=dt,k4_fraction=k4_fraction,digits=200)
    with mp.workdps(200):
        relative=float(abs(a-b)/max(1,abs(b)))
        if not math.isfinite(relative) or relative>1e-40:
            raise ArithmeticError('100/200-digit replay disagreement')
        value=float(b)
        if math.isnan(value) or value==math.inf:
            raise ArithmeticError('Invalid rounded log density')
        record=dict(precisions=[100,200],relative_difference=relative,
                    rounded_log_density=value if math.isfinite(value) else '-Infinity',
                    log10_absolute_log_density=float(mp.log10(abs(b))) if b else None,
                    k4_fraction=k4_fraction,dt=dt,sigma=sigma)
    return value,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--k4-fraction',type=float,required=True)
    parser.add_argument('--sigma',type=float,required=True)
    parser.add_argument('--dt',type=float,required=True)
    args=parser.parse_args(); start=time.monotonic(); values=[]; records=[]
    with args.input.open() as stream:
        for row in csv.reader(stream):
            v=list(map(float,row))
            if len(v)!=11:raise ValueError('Input row needs index plus ten model values')
            value,record=checked_log_transition(v[1:7],v[7:9],v[9:11],sigma=args.sigma,dt=args.dt,k4_fraction=args.k4_fraction)
            record.update(path_index_one_based=int(v[0]),theta_chart=v[1:7],previous=v[7:9],current=v[9:11])
            values.append(value);records.append(record)
    args.output.write_text(''.join(format(v,'.17g')+'\n' for v in values))
    artifact=dict(schema='zhao_cui_pp_tail_reference_v1',role='independent_reference_arithmetic_replay',
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpu_only=True,gpu_intentionally_hidden=True,wall_seconds=time.monotonic()-start,records=records)
    args.record.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
