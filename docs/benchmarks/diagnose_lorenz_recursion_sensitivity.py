"""CPU independent-reference interventions; no safeguard is selected here."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as ref
import validate_lorenz_covariance_reference as independent
from diagnose_lorenz_recursion_localization import compare,load


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();started=time.monotonic();campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    candidate=campaign/'attempt108_c2_lorenz_bounded_integral_safety/covariance_floor_0.001'
    capture=campaign/'attempt111_c2_lorenz_bounded_crossover_capture'
    fixture=json.loads((campaign/'attempt69_c1_lorenz_refinement_reference/fixture.json').read_text())['metadata']
    history={k:load(candidate,k)[3,:,0] for k in ('A','b','c','x','log_incremental_weight')}
    captured={k:load(capture,k)[:,0] for k in ('A','b','c','correction_log_integral','increment_center','increment_scale',
        'increment_centered_A','increment_centered_b','increment_centered_c','projection_added_A')}
    old=[history[k] for k in ('A','b','c')];h=len(history['x'])
    base_integral=independent.projected_integral;base_regress=ref.regress
    summary=dict(status='FAIL',scope='FROZEN_RECURSION_SENSITIVITY_AND_GAUSSIAN_TRUST_CURVES',
        device='CPU',gpu_intentionally_hidden=True,diagnostic_numpy_reference=True,arms=[],
        gaussian_sensitivity=[],trust_curve=[],replication='INCOMPLETE')
    policies={}
    for pinned in (None,13,12,11):
        count=[0]
        def regress(*values):count[0]+=1;return base_regress(*values)
        def integral(mean,cov,increment):
            t=h-1-count[0]
            if pinned is not None and t==pinned-1:
                increment=dict(center=captured['increment_center'][pinned],scale=captured['increment_scale'][pinned],
                    projection_added_A=captured['projection_added_A'][pinned],
                    **{'unprotected_'+k:captured['increment_centered_'+k][pinned] for k in ('A','b','c')})
            return base_integral(mean,cov,increment)
        ref.regress=regress;independent.projected_integral=integral
        try:
            policy,_,_=independent.fitted_policy(fixture,old,history['x'],history['log_incremental_weight'],relative_floor=.001,stable_integral=True)
        finally:ref.regress=base_regress;independent.projected_integral=base_integral
        policies[pinned]=policy
        summary['arms'].append(dict(pinned_next_policy_time=pinned,
            comparisons={k:compare(v,captured[k]) for k,v in zip(('A','b','c'),policy)},
            time10={k:compare(v[10],captured[k][10]) for k,v in zip(('A','b','c'),policy)}))
    q=np.asarray(fixture['transition_covariance']);B=np.linalg.inv(q)
    for j in (14,13,12,11):
        incoming=ref.rk4(history['x'][j-1],2.5)
        actual=[captured[k][j] for k in ('A','b','c')];other=[v[j] for v in policies[None]]
        m,C,L,_=ref.twist(incoming,q,*actual);_,_,L_other,_=ref.twist(incoming,q,*other)
        dA,db,dc=[a-b for a,b in zip(other,actual)]
        # d log integral = -E_new[dq(X)] for a Gaussian tilted density.
        linearized=-(np.einsum('ni,ij,nj->n',m,dA,m)+np.trace(dA@C)+m@db+dc)
        difference=L_other-L
        summary['gaussian_sensitivity'].append(dict(next_time=j,proposal_mean_maxabs=float(np.max(abs(m))),
            covariance_eigenvalues=np.linalg.eigvalsh(C).tolist(),coefficient_delta_maxabs=[float(np.max(abs(v))) for v in (dA,db,dc)],
            integral_delta_maxabs=float(np.max(abs(difference))),linearized_maxabs=float(np.max(abs(linearized))),
            linearization_error_maxabs=float(np.max(abs(difference-linearized)))))
    # Model-trust curve for the already known failed initial proposal121. This
    # describes a whole-update protection; it does not choose a KL threshold.
    failed=campaign/'attempt121_c2_lorenz_bounded_trigger_safety/covariance_floor_0.01'
    saved={k:load(failed,k) for k in ('A','b','c','x')}
    old=[saved[k][3,0,0] for k in ('A','b','c')];new=[saved[k][4,0,0] for k in ('A','b','c')]
    prior=np.asarray(fixture['initial_covariance']);B0=np.linalg.inv(prior)
    mean0,C0,_,_=ref.twist(np.zeros([1,8]),prior,*old);P0=np.linalg.inv(C0)
    next_policy=[saved[k][4,1,0] for k in ('A','b','c')]
    cloud=saved['x'][3,0,0];center=cloud.mean(0);scale=cloud.std(0)
    observation_A=np.diag([5000.]*6+[0.]*2)
    # The new target at time0 has the current observation quadratic plus the
    # exact future integral. Compare only differences from the cloud center to
    # eliminate a harmless large constant from the local model-trust diagnostic.
    normal=load(campaign/'attempt76_c2_lorenz_batch_validation/rejected_batch','normal')[4,0,0]
    center_future=ref.twist(ref.rk4(center[None],2.5),q,*next_policy)[2][0]
    obs_y=load(campaign/'attempt76_c2_lorenz_batch_validation/rejected_batch','y')[0,0]
    def qvalue(x,pol):return np.sum((x@pol[0])*x,-1)+x@pol[1]+pol[2]
    center_obs=5000*np.sum((center[:6]-obs_y)**2)
    for alpha in (0.,1/1024,1/256,1/64,1/16,1/4,.5,.75,.9,.99,1.):
        pol=[a+alpha*(b-a) for a,b in zip(old,new)]
        mean,C,_,margin=ref.twist(np.zeros([1,8]),prior,*pol)
        delta=mean-mean0
        kl=.5*(np.trace(P0@C)+float(delta@P0@delta.T)-8+np.linalg.slogdet(C0)[1]-np.linalg.slogdet(C)[1])
        probes=mean+normal@np.linalg.cholesky(C).T
        with np.errstate(over='ignore',invalid='ignore'):forecast=ref.rk4(probes,2.5)
        row=dict(alpha=alpha,kl_new_to_old=float(kl),precision_margin=float(margin),
            mean_maxabs=float(np.max(abs(mean))),maximum_cloud_scale_displacement=float(np.max(abs((mean-center)/scale))),
            covariance_eigenvalues=np.linalg.eigvalsh(C).tolist(),rk4_finite=bool(np.isfinite(forecast).all()))
        if row['rk4_finite']:
            future=ref.twist(forecast,q,*next_policy)[2]
            target=5000*np.sum((probes[:,:6]-obs_y)**2,-1)-future
            target_delta=target-(center_obs-center_future)
            predicted_delta=qvalue(probes,new)-qvalue(center[None],new)[0]
            row['local_model_error_rms']=float(np.sqrt(np.mean((target_delta-predicted_delta)**2)))
            row['future_centered_target_rms']=float(np.sqrt(np.mean(target_delta**2)))
        summary['trust_curve'].append(row)
    summary.update(status='PASS',wall_seconds=time.monotonic()-started,
        verdict='Sensitivity and diagnostic trust curve only. No protection threshold selected or full-recursion admission.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'],arms=len(summary['arms']),sensitivity=summary['gaussian_sensitivity'],trust_points=len(summary['trust_curve']))))


if __name__=='__main__':main()
