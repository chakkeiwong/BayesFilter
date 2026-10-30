"""CPU-only independent forensic replay of saved Lorenz policies, never a runtime.

NumPy/SciPy/mpmath are diagnostic references. No policy is selected or changed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import mpmath as mp
import tensorflow as tf
import reference_lorenz_stable_fit as reference


def moments(a, b, initial=True):
    precision = np.eye(8) * (100. if initial else 1000.) + 2*a
    covariance = np.linalg.inv(precision)
    mean = np.linalg.solve(precision, -b)
    return mean, covariance, precision


def ratio_on_fixed_inputs(x, old, new):
    """Completed-square E_old(exp(-increment)), using recorded adjacent policies."""
    _, oc, op = moments(old[0], old[1], False)
    _, nc, np_ = moments(new[0], new[1], False)
    base_mean = reference.rk4(x, 2.5)
    om = (1000*base_mean-old[1]) @ oc
    nm = (1000*base_mean-new[1]) @ nc
    da, db, dc = (n-o for o,n in zip(old,new))
    # Extended precision reduces cancellation in the independent scalar evaluation.
    z = nm.astype(np.longdouble)
    distance = z-om.astype(np.longdouble)
    increment = np.einsum('ni,ij,nj->n',z,da.astype(np.longdouble),z)
    increment += z @ db.astype(np.longdouble) + np.longdouble(dc)
    quadratic = np.einsum('ni,ij,nj->n',distance,op.astype(np.longdouble),distance)
    value = .5*(np.linalg.slogdet(nc)[1]-np.linalg.slogdet(oc)[1])-increment-.5*quadratic
    return value.astype(float), (base_mean, om, nm)


def high_precision_ratio(mean, old, new):
    mp.mp.dps = 80
    mat = lambda a: mp.matrix(np.asarray(a).tolist())
    op = 1000*mp.eye(8)+2*mat(old[0]); np_ = 1000*mp.eye(8)+2*mat(new[0])
    oc = op**-1; nc = np_**-1
    om = oc*(1000*mat(mean)-mat(old[1])); nm = nc*(1000*mat(mean)-mat(new[1]))
    da = mat(new[0])-mat(old[0]); db = mat(new[1])-mat(old[1])
    dc = mp.mpf(float(new[2]))-mp.mpf(float(old[2])); distance = nm-om
    value = .5*mp.log(mp.det(nc)/mp.det(oc))-(nm.T*da*nm)[0]-(db.T*nm)[0]-dc
    value -= .5*(distance.T*op*distance)[0]
    return float(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output','root','design'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate',type=Path,action='append',required=True)
    args = p.parse_args(); start = time.monotonic()
    sys.path.insert(0,str(args.root))
    from bayesfilter.replication.lorenz_refinement_tf import LorenzRefinement
    from bayesfilter.replication import multivariate_controlled_smc_tf as kernel
    summary = dict(status='FAIL',scope='fixed_saved_policy_root_cause_audit',replication='INCOMPLETE',
        device='CPU',gpu_intentionally_hidden=True,jit_compile=False,
        role='independent_diagnostic_not_candidate',checks=[],candidates=[],input_sha256={})
    def check(name, good, **info):
        summary['checks'].append(dict(name=name,passed=bool(good),**info))
    def load(path):
        payload=path.read_bytes(); summary['input_sha256'][str(path)]=hashlib.sha256(payload).hexdigest()
        return tf.io.parse_tensor(payload,tf.float64).numpy()
    def clean(value):
        if isinstance(value,float) and not np.isfinite(value): return None
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        return value
    try:
        for module in (kernel,sys.modules[LorenzRefinement.__module__]):
            live=Path(inspect.getfile(module)); frozen=Path(__file__).parent/live.name
            check('source_snapshot_'+live.name,live.read_bytes()==frozen.read_bytes())
        engine = LorenzRefinement(forcing=2.5,particles=512,replicas=1,jit_compile=False).engine
        check('actual_controller_engine_fit_wiring',type(engine) is kernel.MultivariateGaussianControlledSMC
            and engine.fit.python_function.__func__ is kernel.MultivariateGaussianControlledSMC._fit
            and engine.forward.python_function.__func__ is kernel.MultivariateGaussianControlledSMC._forward)
        normal = load(args.design/'rejected_batch/normal.tensor')
        for candidate in args.candidate:
            meta=json.loads((candidate/'summary.json').read_text()); config=meta['candidates'][0]
            folder=candidate/config['folder']
            history={name:load(folder/(name+'.tensor')) for name in ('A','b','c','x','log_incremental_weight')}
            item=dict(attempt=candidate.name,floor=config['floor'],activation_margin=meta['activation_margin'],
                initial_geometries=[],backward_replay=[],nonclaim='Counterfactual replay is not a newly executed adaptive filter')
            summary['candidates'].append(item)
            for stage in range(5):
                a,b,c=(history[name][stage,0,0] for name in ('A','b','c'))
                mean,cov,prec=moments(a,b)
                eig=np.linalg.eigvalsh(prec)
                saved=history['x'][stage,0,0]
                reconstructed=mean+normal[stage,0,0]@np.linalg.cholesky(cov).T
                error=float(np.max(np.abs(reconstructed-saved)))
                check(candidate.name+'_initial_particles_'+str(stage),
                    np.all(np.isfinite(reconstructed)) and np.all(np.abs(reconstructed-saved)<=1e-7+1e-9*np.abs(saved)),max_error=error)
                geometry=dict(stage=stage,mean=mean.tolist(),mean_max_abs=float(np.max(np.abs(mean))),
                    cloud_max_abs=float(np.max(np.abs(saved))),covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),
                    covariance_coordinate_sd=np.sqrt(np.diag(cov)).tolist(),precision_condition=float(eig[-1]/eig[0]),
                    linear_max_abs=float(np.max(np.abs(b))))
                if stage:
                    oa,ob=(history[name][stage-1,0,0] for name in ('A','b'))
                    om,oc,op=moments(oa,ob); da=a-oa; db=b-ob
                    via_linear=-cov@db; via_quadratic=-cov@(2*da@om)
                    total=mean-om; error=float(np.max(np.abs(total-via_linear-via_quadratic)))
                    check(candidate.name+'_mean_shift_identity_'+str(stage),error<1e-8,max_error=error)
                    geometry.update(mean_shift_norm=float(np.linalg.norm(total)),
                        mean_shift_previous_mahalanobis=float(np.sqrt(total@op@total)),
                        shift_from_linear=via_linear.tolist(),shift_from_quadratic=via_quadratic.tolist())
                item['initial_geometries'].append(geometry)
            # Reconstruct each time's regression from the recorded *new next-time*
            # policy. This is a frozen-input target check, not an independent full recursion.
            for stage in (3,4):
                old=[history[k][stage-1,:,0] for k in ('A','b','c')]
                new=[history[k][stage,:,0] for k in ('A','b','c')]
                x=history['x'][stage-1,:,0]; weights=history['log_incremental_weight'][stage-1,:,0]
                for t in reversed(range(len(x))):
                    if t==len(x)-1: delta=np.zeros(len(x[t])); auxiliary=None
                    else: delta,auxiliary=ratio_on_fixed_inputs(x[t],tuple(v[t+1] for v in old),tuple(v[t+1] for v in new))
                    target=-weights[t]-delta
                    fa,fb,fc,rss,sratio,centered=reference.regress(x[t],target)
                    pa=old[0][t]+fa; pb=old[1][t]+fb; pc=old[2][t]+fc
                    raw=np.eye(8)*(100. if t==0 else 1000.)+2*pa
                    ev,vectors=np.linalg.eigh(raw); margin=float(ev[0]/np.max(np.abs(ev)))
                    activates=margin<=meta['activation_margin']
                    projected=raw.copy()
                    if activates:
                        inverse=1/ev; covariance_floor=config['floor']*np.max(np.abs(inverse))
                        values=1/np.maximum(inverse,covariance_floor)
                        projected=(vectors*values)@vectors.T
                    expected_a=pa+.5*(projected-raw)
                    errors={key:float(np.max(np.abs(actual-predicted))) for key,actual,predicted in
                        zip(('A','b','c'),(v[t] for v in new),(expected_a,pb,pc))}
                    passed=all(np.all(np.abs(actual-predicted)<=1e-6+1e-8*np.abs(actual))
                        for actual,predicted in zip((v[t] for v in new),(expected_a,pb,pc)))
                    row=dict(stage=stage,time=t,raw_precision_margin=margin,activation=activates,
                        singular_ratio=sratio,residual_rms=float(np.sqrt(rss/len(x[t]))),
                        target_centered_rms=float(np.std(target)),errors=errors,
                        coefficient_replay_within_diagnostic_tolerance=bool(passed))
                    if t==0:
                        mean=-np.linalg.solve(projected,pb); old_mean,_,_=moments(old[0][0],old[1][0])
                        row.update(old_mean=old_mean.tolist(),replayed_mean=mean.tolist(),
                            raw_precision_eigenvalues=ev.tolist(),projected_precision_eigenvalues=np.linalg.eigvalsh(projected).tolist(),
                            sample_center=centered['center'].tolist(),sample_scale=centered['scale'].tolist(),
                            unprojected_b=pb.tolist(),added_A_frobenius=float(np.linalg.norm(.5*(projected-raw))))
                        for index in (0, len(x[t])//2, len(x[t])-1):
                            high=high_precision_ratio(auxiliary[0][index],tuple(v[1] for v in old),tuple(v[1] for v in new))
                            error=abs(high-delta[index])
                            check(candidate.name+'_mp_ratio_'+str(stage)+'_'+str(index),
                                error<=1e-7+1e-10*abs(high),max_error=error,value=high)
                        actual=engine.regress(tf.constant(x[t][None],tf.float64),tf.constant(target[None],tf.float64))
                        diffs={k:float(np.max(np.abs(actual[k].numpy()[0]-value))) for k,value in zip(('A','b','c'),(fa,fb,fc))}
                        row['same_target_tf_cpu_vs_svd_errors']=diffs
                        row['tf_regression_valid']=bool(actual['valid'].numpy())
                    item['backward_replay'].append(row)
            item['replay_mismatches']=[dict(stage=r['stage'],time=r['time'],errors=r['errors']) for r in item['backward_replay'] if not r['coefficient_replay_within_diagnostic_tolerance']]
        summary['status']='PASS' if all(v['passed'] for v in summary['checks']) and all(not row['replay_mismatches'] for row in summary['candidates']) else 'FAIL'
        summary['scientific_verdict']='EXPLANATORY_AUDIT_ONLY_NO_REPAIR_ADMITTED'
    except Exception as exc:
        summary['error']=repr(exc)
    summary['wall_seconds']=time.monotonic()-start
    (args.output/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'],checks=len(summary['checks']),error=summary.get('error'),wall_seconds=summary['wall_seconds'])))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
