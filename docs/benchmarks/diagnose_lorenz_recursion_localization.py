"""Frozen-recursion CPU/SciPy diagnostic, never a runtime implementation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as ref
import validate_lorenz_covariance_reference as independent


def load(folder,key):
    return tf.io.parse_tensor(tf.io.read_file(str(folder/(key+'.tensor'))),tf.float64).numpy()


def compare(actual,expected):
    difference=np.abs(actual-expected)
    scaled=difference/(1e-7+1e-9*np.abs(expected))
    return dict(passed=bool(np.all(np.isfinite(scaled)) and np.all(scaled<=1)),
                max_absolute=float(np.max(difference)),max_scaled=float(np.max(scaled)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    candidate=campaign/'attempt108_c2_lorenz_bounded_integral_safety/covariance_floor_0.001'
    capture=campaign/'attempt111_c2_lorenz_bounded_crossover_capture'
    fixture=json.loads((campaign/'attempt69_c1_lorenz_refinement_reference/fixture.json').read_text())['metadata']
    keys=('A','b','c','x','log_incremental_weight')
    history={k:load(candidate,k)[3,:,0] for k in keys}
    capkeys=('A','b','c','correction_log_integral','increment_center','increment_scale',
             'increment_centered_A','increment_centered_b','increment_centered_c','projection_added_A')
    captured={k:load(capture,k)[:,0] for k in capkeys}
    old=[history[k] for k in ('A','b','c')]
    h=len(history['x']);original_regress=ref.regress;original_integral=independent.projected_integral
    summary=dict(status='FAIL',scope='FROZEN_FULL_BACKWARD_RECURSION_LOCALIZATION',
                 device='CPU',gpu_intentionally_hidden=True,diagnostic_numpy_reference=True,
                 source_inputs=[str(candidate),str(capture)],arms=[],replication='INCOMPLETE')
    for mode in ('completed_square_all','match_gpu_branch'):
        rows=[];diagnostics=[]
        def regress(x,target):
            t=h-1-len(rows)
            ans=original_regress(x,target)
            a,b,c,rss,ratio,centered=ans
            delta=-history['log_incremental_weight'][t]-target
            raw=np.linalg.inv(np.asarray(fixture['initial_covariance' if t==0 else 'transition_covariance']))+2*(old[0][t]+a)
            ev=np.linalg.eigvalsh(raw)
            rows.append(dict(time=t,correction=compare(delta,captured['correction_log_integral'][t]),
                target=compare(target,-history['log_incremental_weight'][t]-captured['correction_log_integral'][t]),
                centered_fit={k:compare(centered[k],captured['increment_centered_'+k][t]) for k in ('A','b','c')},
                raw_precision_eigenvalues=ev.tolist(),singular_ratio=float(ratio),
                fit_center=centered['center'].tolist(),fit_scale=centered['scale'].tolist()))
            return ans
        def integral(old_mean,old_cov,inc):
            if mode=='completed_square_all' or np.any(inc['projection_added_A']):
                return original_integral(old_mean,old_cov,inc)
            zmean=(old_mean-inc['center'])/inc['scale']
            zcov=old_cov/(inc['scale'][:,None]*inc['scale'][None,:])
            return ref.twist(zmean,zcov,inc['unprotected_A'],inc['unprotected_b'],inc['unprotected_c'])[2]
        ref.regress=regress;independent.projected_integral=integral
        try:
            policy,ratio,protections=independent.fitted_policy(fixture,old,history['x'],history['log_incremental_weight'],
                diagnostics=diagnostics,actual=[captured[k] for k in ('A','b','c')],relative_floor=.001,stable_integral=True)
        finally:
            ref.regress=original_regress;independent.projected_integral=original_integral
        for row,diag in zip(rows,diagnostics):
            t=row['time'];row['recursion']=diag
            row['policy']={k:compare(v[t],captured[k][t]) for k,v in zip(('A','b','c'),policy)}
            # A comparison of the represented polynomial, on independent axis probes,
            # distinguishes meaningful disagreement from coordinate magnification.
            center=np.mean(history['x'][t],axis=0);scale=np.std(history['x'][t],axis=0)
            probes=np.concatenate([center[None],center+np.eye(8)*scale,center-np.eye(8)*scale])
            poly=lambda a,b,c:np.sum((probes@a)*probes,-1)+probes@b+c
            row['polynomial_probes']=compare(poly(*(v[t] for v in policy)),poly(*(captured[k][t] for k in ('A','b','c'))))
            covariance=np.asarray(fixture['initial_covariance' if t==0 else 'transition_covariance'])
            base_mean=np.zeros([1,8]) if t==0 else ref.rk4(history['x'][t-1],2.5)
            local=ref.twist(base_mean,covariance,*(v[t] for v in policy))
            gpu=ref.twist(base_mean,covariance,*(captured[k][t] for k in ('A','b','c')))
            row['gaussian']={k:compare(local[j],gpu[j]) for j,k in enumerate(('mean','covariance','integral'))}
        first=lambda key:next((row['time'] for row in rows if not row[key]['passed']),None)
        failures=[row['time'] for row in rows if not all(v['passed'] for v in row['policy'].values())]
        summary['arms'].append(dict(mode=mode,rows=rows,protections=protections,
            first_coefficient_failure=failures[0] if failures else None,
            first_correction_failure=first('correction'),first_polynomial_failure=first('polynomial_probes'),
            first_mean_failure=next((r['time'] for r in rows if not r['gaussian']['mean']['passed']),None),
            coefficient_failures=failures,
            complete_policy={k:compare(v,captured[k]) for k,v in zip(('A','b','c'),policy)}))
    summary.update(status='PASS',wall_seconds=time.monotonic()-started,
        interpretation='Diagnostic executed. Candidate agreement is recorded separately; this status does not admit a repair.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'],arms=[{k:v for k,v in arm.items() if k not in ('rows','coefficient_failures')} for arm in summary['arms']])))


if __name__=='__main__':main()
