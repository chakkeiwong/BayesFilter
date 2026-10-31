"""Independent CPU-only Octave/SciPy reference for the RBF NNLS consumer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import time
import numpy as np  # Explicit independent reference, never imported by runtime.
from scipy.io import loadmat
from scipy.optimize import nnls


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items() if not k.startswith('__')}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--diagnose-only',action='store_true')
    args=parser.parse_args();started=time.monotonic();snapshot=args.output/'sources'
    source=(snapshot/'cSMC_Resample.m').read_text()
    replacements={
      'function cSMC = cSMC_Resample(':'function cSMC = cSMC_Resample_fit_reference(',
      'DiscreteSample(NormalisedPolicyWeights(n,:), 1)':'DiscreteSample(NormalisedPolicyWeights(Ancestor(n),:), 1)',
      '        SMC.Trajectory(:,1) = X;':'        SMC.Trajectory(:,1) = X;\n        SMC.AuditComponents(:,1) = SelectComponents(:);',
      '            SMC.Trajectory(:,t+1) = X;':'            SMC.Trajectory(:,t+1) = X;\n            SMC.AuditComponents(:,t+1) = SelectComponents(:);',
      'FittedWeights = lsqnonneg( RBF(Bandwidth, PairwiseDistancesq), PolicyValue );':'FittedWeights = lsqnonneg( RBF(Bandwidth, PairwiseDistancesq), PolicyValue );\n                    NewPolicy.FullCoefficients(:,t) = FittedWeights;\n                    NewPolicy.FitTarget(:,t) = PolicyValue;'
    }
    edited=source
    for old,new in replacements.items():
        assert source.count(old)==(2 if old.startswith('FittedWeights') else 1),old
        edited=edited.replace(old,new)
    baseline=edited.replace('cSMC_Resample_fit_reference(', 'cSMC_Resample_fit_unvectorized(',1)
    (snapshot/'cSMC_Resample_fit_unvectorized.m').write_text(baseline)
    loop='''            SelectComponents = zeros(1,N);
            for n = 1:N
                SelectComponents(n) = DiscreteSample(NormalisedPolicyWeights(Ancestor(n),:), 1);
            end'''
    vector='''            Edges = cumsum(NormalisedPolicyWeights(Ancestor,:),2);
            Total = Edges(:,end); Renormalize = abs(Total-1)>eps;
            Edges(Renormalize,:) = bsxfun(@rdivide,Edges(Renormalize,:),Total(Renormalize));
            % Octave DiscreteSample consumes rand(1), then randperm(1).
            RandomDraws = rand(2,N);
            SelectComponents = (sum(bsxfun(@ge,RandomDraws(1,:)',Edges),2)+1)';'''
    assert edited.count(loop)==1
    edited=edited.replace(loop,vector)
    call='FittedWeights = lsqnonneg( RBF(Bandwidth, PairwiseDistancesq), PolicyValue );'
    timed='''fprintf('nnls_begin time=%d\\n',t);fflush(stdout);FitStarted=time();
                    [FittedWeights, FitRSS, FitResidual, FitFlag, FitInfo] = lsqnonneg( RBF(Bandwidth, PairwiseDistancesq), PolicyValue, [], struct('MaxIter',5*N) );
                    fprintf('nnls_end time=%d seconds=%.6f iterations=%d active=%d\\n',t,time()-FitStarted,FitInfo.iterations,sum(FittedWeights>0));fflush(stdout);
                    if FitFlag<=0
                        Matrix=RBF(Bandwidth,PairwiseDistancesq);
                        save('-mat7-binary',fullfile(Parameters.AuditOutput,'failed-nnls.mat'),'Matrix','PolicyValue','FittedWeights','FitRSS','FitFlag','FitInfo','Particles','Bandwidth','t','SMC');
                        error('RBF_NNLS_DID_NOT_CONVERGE');
                    end'''
    edited=edited.replace(call,timed)
    (snapshot/'cSMC_Resample_fit_reference.m').write_text(edited)
    (args.output/'corrected-source.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),edited.splitlines(True),fromfile='pinned_author',tofile='corrected_instrumented')))
    command=['/usr/bin/octave-cli','--no-gui','--quiet',str(snapshot/'reference_rbf_fit.m'),str(args.output)]
    completed=subprocess.run(command,check=False)
    if args.diagnose_only:
        diagnose(args.output,command,completed.returncode,started)
        return
    completed.check_returncode()
    export=plain(loadmat(args.output/'export.mat',simplify_cells=True));fixture=export['fixture'];summary=export['summary'];checks=[]
    for case in fixture['cases']:
        cloud=np.array(case['rounds'][0]['x']);beta=np.array(case['rounds'][1]['beta'])
        targets=np.array(case['fit_target']);coeff=np.array(case['full_coefficients'])
        for t in (0,50,100):
            design=np.exp(-beta[t]*(cloud[t,:,None]-cloud[t,None,:])**2);target=targets[t]
            tol=10*np.finfo(float).eps*np.linalg.norm(design,1)*len(target)
            alternative,rnorm=nnls(design,target,maxiter=5*len(target),atol=tol)
            prediction=design@coeff[t];gradient=design.T@(prediction-target);rss=np.sum((prediction-target)**2)
            feasible=bool(np.min(coeff[t])>=0 and np.all(np.isfinite(coeff[t])))
            kkt=bool(np.min(gradient)>=-20*tol and np.max(np.abs(coeff[t]*gradient))<=20*tol*max(1,np.max(coeff[t])))
            objective=bool(abs(rss-rnorm*rnorm)<=1e-8+1e-6*rnorm*rnorm)
            checks.append(dict(seed=case['seed'],time=t,feasible=feasible,kkt=kkt,objective_agreement=objective,
                source_rss=float(rss),scipy_rss=float(rnorm*rnorm),gradient_tolerance=float(tol),
                source_positive_coefficients=int(np.count_nonzero(coeff[t])),scipy_positive_coefficients=int(np.count_nonzero(alternative)),
                prediction_difference=float(np.max(np.abs(prediction-design@alternative)))))
    summary.update(status='PASS' if all(c['feasible'] and c['kkt'] and c['objective_agreement'] for c in checks) else 'FAIL',
        independent_nnls_checks=checks,reference_command=command,wall_seconds=time.monotonic()-started,
        source_target='CORRECTED_ANCESTOR_WITH_PARITY_CHECKED_VECTORIZED_CATEGORICAL_REFERENCE',numpy_role='INDEPENDENT_REFERENCE_ONLY',
        nonclaims=['No Table2 repetition matrix','No general NNLS or runtime admission','No empirical ranking'])
    (args.output/'fixture.json').write_text(json.dumps(fixture,allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],checks=len(checks),source_seconds=summary['source_seconds'],wall_seconds=summary['wall_seconds'])))
    if summary['status']!='PASS':raise SystemExit(1)


def diagnose(output,command,returncode,started):
    from scipy.optimize import lsq_linear
    failure=loadmat(output/'failed-nnls.mat',simplify_cells=True)
    G=failure['Matrix'];target=failure['PolicyValue'];original=failure['FittedWeights']
    tol=10*np.finfo(float).eps*np.linalg.norm(G,1)*len(target)
    singular=np.linalg.svd(G,compute_uv=False)
    def metrics(w):
        residual=G@w-target;gradient=G.T@residual
        feasible=bool(np.all(np.isfinite(w)) and np.min(w)>=0)
        dual=float(max(0,-np.min(gradient)));comp=float(np.max(np.abs(w*gradient)))
        return dict(rss=float(np.dot(residual,residual)),dual_error=dual,complementarity=comp,
            positive_coefficients=int(np.count_nonzero(w)),valid=bool(feasible and dual<=20*tol and comp<=20*tol*max(1,np.max(w))))
    alternatives=[];reference=None
    for name in ('scipy_nnls','scipy_bvls'):
        begin=time.monotonic()
        try:
            if name=='scipy_nnls':w,_=nnls(G,target,maxiter=5*len(target),atol=tol);converged=True
            else:
                result=lsq_linear(G,target,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=5*len(target),lsq_solver='exact')
                w=result.x;converged=bool(result.success)
            entry=dict(name=name,seconds=time.monotonic()-begin,converged=converged,**metrics(w))
            if entry['valid'] and converged and reference is None:reference=w
            np.save(output/(name+'-coefficients.npy'),w)
        except Exception as exc:entry=dict(name=name,seconds=time.monotonic()-begin,valid=False,failure=repr(exc))
        alternatives.append(entry)
    fixture=dict(scope='FROZEN_RBF_NNLS_FAILURE',time=int(failure['t'])-1,seed=20261006,
        particles=len(target),x=failure['Particles'].tolist(),beta=float(failure['Bandwidth']),
        target=target.tolist(),source_coefficients=original.tolist(),reference_coefficients=None if reference is None else reference.tolist(),
        gradient_tolerance=float(tol))
    summary=dict(status='PASS',scope='RBF_NNLS_ROOT_CAUSE_DIAGNOSIS',replication='INCOMPLETE',
        reference_valid=reference is not None,source_returncode=returncode,source_exitflag=int(failure['FitFlag']),
        source_iterations=int(failure['FitInfo']['iterations']),source_metrics=metrics(original),
        singular_max=float(singular[0]),singular_min=float(singular[-1]),
        numerical_rank=int(np.sum(singular>np.finfo(float).eps*len(target)*singular[0])),
        gradient_tolerance=float(tol),alternatives=alternatives,wall_seconds=time.monotonic()-started,
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='INDEPENDENT_REFERENCE_ONLY',
        reference_command=command,nonclaims=['No completed full-horizon reference','No Table2 replication','No policy promotion'])
    (output/'fixture.json').write_text(json.dumps(fixture,allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
