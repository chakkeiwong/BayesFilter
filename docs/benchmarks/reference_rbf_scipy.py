"""Independent CPU-only SciPy NNLS reference with corrected author forward code.

NumPy is used only for this independent reference, never as a candidate runtime.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import nnls, lsq_linear
from scipy.special import logsumexp
from diagnose_rbf_reference_feasibility import boundary_projection
from reference_rbf_svd_nnls import svd_nnls


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items() if not k.startswith('__')}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--svd-reference',action='store_true');parser.add_argument('--stopping-factor',type=float,default=.01);parser.add_argument('--seeds',type=int,nargs=2,default=[20261006,20261007]);args=parser.parse_args();out=args.output;snapshot=out/'sources';start=time.monotonic();diagnostics=[]
    if not np.isfinite(args.stopping_factor) or args.stopping_factor<=0 or args.seeds[0]==args.seeds[1]:raise ValueError('Invalid stopping factor or repeated seeds')
    source=(snapshot/'cSMC_Resample.m').read_text()
    edited=source.replace('function cSMC = cSMC_Resample(','function cSMC = cSMC_Resample_fit_reference(',1)
    edited=edited.replace('DiscreteSample(NormalisedPolicyWeights(n,:), 1)','DiscreteSample(NormalisedPolicyWeights(Ancestor(n),:), 1)')
    edited=edited.replace('        SMC.Trajectory(:,1) = X;','        SMC.Trajectory(:,1) = X;\n        SMC.AuditComponents(:,1) = SelectComponents(:);')
    edited=edited.replace('            SMC.Trajectory(:,t+1) = X;','            SMC.Trajectory(:,t+1) = X;\n            SMC.AuditComponents(:,t+1) = SelectComponents(:);')
    loop='''            SelectComponents = zeros(1,N);
            for n = 1:N
                SelectComponents(n) = DiscreteSample(NormalisedPolicyWeights(Ancestor(n),:), 1);
            end'''
    vector='''            Edges = cumsum(NormalisedPolicyWeights(Ancestor,:),2);
            Total = Edges(:,end); Renormalize = abs(Total-1)>eps;
            Edges(Renormalize,:) = bsxfun(@rdivide,Edges(Renormalize,:),Total(Renormalize));
            RandomDraws = rand(2,N);
            SelectComponents = (sum(bsxfun(@ge,RandomDraws(1,:)',Edges),2)+1)';'''
    assert edited.count(loop)==1;edited=edited.replace(loop,vector)
    marker='    cSMC{1,1} = CurrentPolicy;'
    assert edited.count(marker)==1
    edited=edited.replace(marker,"    if isfield(Parameters,'ReferencePolicy'), CurrentPolicy=Parameters.ReferencePolicy; end\n"+marker)
    (snapshot/'cSMC_Resample_fit_reference.m').write_text(edited)
    (out/'corrected-source.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),edited.splitlines(True),fromfile='pinned_author',tofile='corrected_injected_reference')))
    # Reuse the inspected forward-export/replay construction verbatim.
    template=(snapshot/'reference_rbf_fit.m').read_text()
    setup=template[:template.index('% Cheap actual-endpoint')].replace("args=argv();output=args{1};","args=argv();output=args{1};mode=args{2};")
    replay=template[template.index(' rounds=cell(1,2);'):template.index('fixture=struct(')]
    finish=template[template.index('fixture=struct('):].replace("'vectorized_reference_bitwise_parity',true,","'vectorized_reference_parity_evidence','attempt36',")
    driver=setup+'''
Parameters.Iterations=0;seeds=[20261006,20261007];
if strcmp(mode,'base')
 base_cases=cell(1,2);
 for b=1:2
  rand('state',seeds(b));randn('state',seeds(b));
  base=cSMC_Resample_fit_reference(Parameters,Initial,Transition,Observation);
  base_cases{b}=struct('policy',base{1,1},'smc',base{1,2},'uniform_state',rand('state'),'normal_state',randn('state'));
 end
 save('-mat7-binary',fullfile(output,'base.mat'),'base_cases','y');
 return;
end
load(fullfile(output,'base.mat'));load(fullfile(output,'policies.mat'));
cases=cell(1,2);times=zeros(1,2);
for b=1:2
 Parameters.ReferencePolicy=policies{b};
 rand('state',base_cases{b}.uniform_state);randn('state',base_cases{b}.normal_state);tic;
 fitted_run=cSMC_Resample_fit_reference(Parameters,Initial,Transition,Observation);times(b)=toc;
 corrected=cell(2,2);corrected{1,1}=base_cases{b}.policy;corrected{1,2}=base_cases{b}.smc;
 corrected{2,1}=fitted_run{1,1};corrected{2,2}=fitted_run{1,2};
'''+replay+finish
    driver=driver.replace('seeds=[20261006,20261007];','seeds=[%d,%d];'%tuple(args.seeds))
    script=snapshot/'reference_rbf_scipy.m';script.write_text(driver)
    commands=[]
    def octave(mode):
        command=['/usr/bin/octave-cli','--no-gui','--quiet',str(script),str(out),mode]
        commands.append(command);subprocess.run(command,check=True)
    failure=None
    try:
        octave('base');base=loadmat(out/'base.mat',simplify_cells=True);policies=np.empty((1,2),dtype=object)
        for b,case in enumerate(base['base_cases']):
            smc=case['smc'];cloud=smc['Trajectory'];n,h=cloud.shape;m=16
            beta=np.zeros(h);weights=np.zeros((m,h));knots=np.zeros((m,h));constant=np.zeros(h)
            coefficients=np.zeros((n,h));targets=np.zeros((n,h));lookahead=np.zeros(n)
            for t in range(h-1,-1,-1):
                begun=time.monotonic();x=cloud[:,t];beta[t]=.3*np.std(x,ddof=1)
                logtarget=smc['LogObservationDensity'][:,t]+lookahead;constant[t]=np.max(logtarget)
                target=np.exp(logtarget-constant[t]);design=np.exp(-beta[t]*(x[:,None]-x[None,:])**2)
                tol=10*np.finfo(float).eps*np.linalg.norm(design,1)*n
                trials=[];accepted=False
                methods=(('svd_active_set',args.stopping_factor),) if args.svd_reference else (('scipy.nnls',None),('scipy.bvls',1e-14),('scipy.bvls',1e-15))
                for solver_name,accuracy in methods:
                    trial=dict(solver=solver_name,tolerance=accuracy)
                    try:
                        if solver_name=='svd_active_set':
                            candidate,info=svd_nnls(design,target,stopping_factor=accuracy)
                            converged=True;correction=dict(eligible=True);trial['solver_diagnostics']=info
                        elif solver_name=='scipy.nnls':
                            candidate,_=nnls(design,target,maxiter=5*n);converged=True;correction=dict(eligible=True)
                        else:
                            alt=lsq_linear(design,target,bounds=(0.,np.inf),method='bvls',lsq_solver='exact',tol=accuracy,max_iter=5*n)
                            candidate,correction=boundary_projection(design,alt.x);converged=bool(alt.success)
                            trial['iterations']=int(alt.nit);trial['boundary_correction']=correction
                        g=design.T@(design@candidate-target)
                        dual_alt=max(0.,float(-np.min(g)));comp_alt=float(np.max(np.abs(candidate*g)))
                        okay=bool(converged and correction['eligible'] and np.all(np.isfinite(candidate)) and np.min(candidate)>=0 and dual_alt<=20*tol and comp_alt<=20*tol*max(1.,float(np.max(candidate))))
                        trial.update(valid=okay,converged=converged,minimum=float(np.min(candidate)),maximum=float(np.max(candidate)),
                            dual_error=dual_alt,complementarity=comp_alt,rss=float(np.sum((design@candidate-target)**2)))
                        if not okay:np.save(out/('rejected_seed%d_time%d_trial%d.npy'%(args.seeds[b],t,len(trials))),candidate)
                    except RuntimeError as exc:
                        trial.update(valid=False,failure=str(exc));okay=False
                    trials.append(trial)
                    if okay:
                        coef=candidate;rnorm=float(np.linalg.norm(design@coef-target));accepted=True;break
                if not accepted:
                    np.savez(out/('nnls_failure_seed%d_time%d.npz'%(args.seeds[b],t)),design=design,target=target,x=x,beta=beta[t])
                    (out/'solver-failure.json').write_text(json.dumps(dict(seed=args.seeds[b],time=t,trials=trials),indent=2)+'\n')
                    raise RuntimeError('All independent solvers failed at '+str((b,t)))
                residual=design@coef-target;gradient=design.T@residual
                dual=max(0.,float(-np.min(gradient)));complementarity=float(np.max(np.abs(coef*gradient)))
                valid=bool(np.all(np.isfinite(coef)) and np.min(coef)>=0 and dual<=20*tol and complementarity<=20*tol*max(1.,float(np.max(coef))))
                diagnostics.append(dict(seed=args.seeds[b],time=t,solver=solver_name,solver_trials=trials,rss=float(rnorm*rnorm),dual_error=dual,complementarity=complementarity,
                    gradient_tolerance=float(tol),positive_coefficients=int(np.count_nonzero(coef)),valid=valid,seconds=time.monotonic()-begun))
                (out/'fit-progress.json').write_text(json.dumps(dict(completed=len(diagnostics),latest=diagnostics[-1]),indent=2)+'\n')
                if not valid:raise RuntimeError('Independent NNLS KKT failed at '+str((b,t)))
                coefficients[:,t]=coef;targets[:,t]=target
                ix=np.argsort(-coef,kind='stable')[:m];weights[:,t]=coef[ix];knots[:,t]=x[ix]
                if not np.sum(weights[:,t])>0:raise RuntimeError('Empty retained policy')
                if t>0:
                    mean=smc['TransitionMean'][:,t-1];denom=1+20*beta[t]
                    with np.errstate(divide='ignore'):
                        terms=np.log(weights[:,t])[None,:]-beta[t]/denom*(mean[:,None]-knots[:,t][None,:])**2
                    lookahead=constant[t]-.5*np.log(denom)+logsumexp(terms,axis=1)
            policies[0,b]=dict(Bandwidth=beta[None,:],Weights=weights,Knots=knots,MaxLogPolicyValue=constant[None,:],FullCoefficients=coefficients,FitTarget=targets)
        savemat(out/'policies.mat',{'policies':policies});octave('forward')
        export=plain(loadmat(out/'export.mat',simplify_cells=True));fixture=export['fixture']
        fixture['stopping_factor']=args.stopping_factor if args.svd_reference else None
        fixture['seeds']=args.seeds
        fixture['scope']=('SVD_NNLS' if args.svd_reference else 'SCIPY_NNLS')+'_CORRECTED_SOURCE_RBF_FULL_HORIZON_REFERENCE'
        (out/'fixture.json').write_text(json.dumps(fixture,allow_nan=False)+'\n')
    except Exception as exc:failure=repr(exc)
    summary=dict(status='FAIL' if failure else 'PASS',failure=failure,scope='RBF_SCIPY_FULL_HORIZON_REFERENCE',replication='INCOMPLETE',
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='independent_reference_only',solver='fresh_svd_active_set' if args.svd_reference else 'scipy_nnls_bvls',
        commands=commands,seeds=args.seeds,stopping_factor=args.stopping_factor if args.svd_reference else None,fits=diagnostics,wall_seconds=time.monotonic()-start,
        nonclaims=['No literal unmodified author parity','No empirical Table2 replication','No TF runtime admission'])
    (out/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','failure','wall_seconds')}))
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
