"""Independent CPU/SciPy-SVD reference, never imported by a TF runtime.

NumPy is used only for this diagnostic authority. Every fitted basis is full
rank; no truncated solve or regularization is accepted.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import lstsq
from scipy.special import logsumexp


def rk4(x,forcing):
    def drift(z):
        return (np.roll(z,-1,axis=-1)-np.roll(z,2,axis=-1))*np.roll(z,1,axis=-1)-z+forcing
    z=x.copy()
    for _ in range(10):
        k1=drift(z);k2=drift(z+.005*k1)
        k3=drift(z+.005*k2);k4=drift(z+.01*k3)
        z=z+(.01/6)*(k1+2*k2+2*k3+k4)
    return z


def twist(mean,covariance,a,b,c):
    precision=np.linalg.inv(covariance);p=precision+2*a
    eigenvalues=np.linalg.eigvalsh(p)
    margin=float(eigenvalues[0]/eigenvalues[-1])
    if margin<=64*np.finfo(float).eps:raise ValueError('Invalid twisted precision')
    k=np.linalg.inv(p);shift=mean@precision-b;cm=shift@k
    integral=.5*(np.linalg.slogdet(k)[1]-np.linalg.slogdet(covariance)[1])-c
    integral=integral+.5*np.sum(shift*cm,axis=-1)-.5*np.sum((mean@precision)*mean,axis=-1)
    return cm,k,integral,margin


def regress(x,target):
    center=x.mean(0);scale=x.std(0)
    if not np.all(scale>0):raise ValueError('Zero coordinate spread')
    z=(x-center)/scale;d=x.shape[1]
    pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
    design=np.column_stack([np.ones(len(x)),z]+[z[:,i]*z[:,j] for i,j in pairs]+[z*z])
    response_center=target.mean()
    beta,_,rank,s=lstsq(design,target-response_center,lapack_driver='gelsd')
    beta[0]+=response_center
    if rank!=design.shape[1]:raise ValueError('Rank loss; truncated SVD forbidden')
    a=np.diag(beta[-d:])
    for coefficient,(i,j) in zip(beta[1+d:-d],pairs):a[i,j]=a[j,i]=.5*coefficient
    centered=dict(center=center,scale=scale,A=a.copy(),b=beta[1:d+1].copy(),c=float(beta[0]))
    a=a/(scale[:,None]*scale[None,:]);linear=beta[1:d+1]/scale
    b=linear-2*center@a;c=beta[0]-linear@center+center@a@center
    residual=design@beta-target
    return a,b,float(c),float(np.sum(residual**2)),float(s[-1]/s[0]),centered


def fitted_policy(fixture,source):
    x=np.asarray(source['x']);weights=np.asarray(source['log_incremental_weight'])
    old=[np.asarray(source[k]) for k in ('A','b','c')]
    new=[np.zeros_like(v) for v in old];h=len(x)
    covariance=np.asarray(fixture['transition_covariance']);prior=np.asarray(fixture['initial_covariance'])
    rss=np.zeros(h);ratios=np.zeros(h);margins=np.zeros(h)
    increment=None
    for t in reversed(range(h)):
        correction=0.
        if t<h-1:
            mean=rk4(x[t],fixture['forcing'])
            old_mean,old_covariance,_,_=twist(mean,covariance,*(v[t+1] for v in old))
            zmean=(old_mean-increment['center'])/increment['scale']
            zcovariance=old_covariance/(increment['scale'][:,None]*increment['scale'][None,:])
            correction=twist(zmean,zcovariance,increment['A'],increment['b'],increment['c'])[2]
        fit=regress(x[t],-weights[t]-correction)
        for j in range(3):new[j][t]=old[j][t]+fit[j]
        rss[t],ratios[t]=fit[3:5];increment=fit[5]
        margins[t]=twist(np.zeros((1,x.shape[-1])),prior if t==0 else covariance,*(v[t] for v in new))[3]
    return dict(zip(('A','b','c'),new)),rss,ratios,margins


def forward(fixture,policy,random):
    a,b,c=(policy[k] for k in ('A','b','c'));h,n,d=(int(fixture[k]) for k in ('horizon','particles','dimension'))
    y=np.asarray(fixture['y']);normal=np.asarray(random['normal']);uniform=np.asarray(random['uniform'])
    prior=np.asarray(fixture['initial_covariance']);transition=np.asarray(fixture['transition_covariance'])
    initial=twist(np.zeros((n,d)),prior,a[0],b[0],c[0])[2][0]
    previous=np.zeros((n,d));logz=initial
    result={key:[] for key in ('x','ancestors','log_incremental_weight','logZ','ess')}
    for t in range(h):
        mean=np.zeros((n,d)) if t==0 else rk4(previous,fixture['forcing'])
        cm,covariance,_,_=twist(mean,prior if t==0 else transition,a[t],b[t],c[t])
        x=cm+normal[t]@np.linalg.cholesky(covariance).T
        lookahead=0. if t==h-1 else twist(rk4(x,fixture['forcing']),transition,a[t+1],b[t+1],c[t+1])[2]
        logw=-.5/fixture['observation_variance']*np.sum((y[t]-x[:,:len(y[t])])**2,axis=-1)
        logw=logw+lookahead+np.sum((x@a[t])*x,axis=-1)+x@b[t]+c[t]
        normalizer=logsumexp(logw);logz+=normalizer-np.log(n)
        w=np.exp(logw-normalizer);cdf=np.cumsum(w);cdf[-1]=1.
        idx=np.searchsorted(cdf,(np.arange(n)+uniform[t])/n,side='left')
        previous=x[idx]
        for key,value in zip(result,(x,idx,logw,logz,1./np.sum(w*w))):result[key].append(value)
    return {key:np.asarray(value) for key,value in result.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();fixture=json.loads(args.fixture.read_text());results=[];diagnostics=[]
    for rep in fixture['replicates']:
        policy,rss,ratios,margins=fitted_policy(fixture,rep['rounds'][0])
        replay=forward(fixture,policy,rep['rounds'][1])
        if not all(np.isfinite(v).all() for v in list(policy.values())+list(replay.values())):
            raise ValueError('Nonfinite independent result')
        results.append(dict(seed=rep['seed'],policy={k:v.tolist() for k,v in policy.items()},
                            forward={k:v.tolist() for k,v in replay.items()}))
        diagnostics.append(dict(seed=rep['seed'],rss=float(rss.sum()),minimum_singular_ratio=float(ratios.min()),
                                minimum_precision_margin=float(margins.min()),logZ=float(replay['logZ'][-1])))
    reference=dict(replicates=results,source_fixture=str(args.fixture),solver='scipy_gelsd_centered_full_rank_relative_integral')
    (args.output/'reference.json').write_text(json.dumps(reference,allow_nan=False)+'\n')
    summary=dict(status='PASS',scope='INDEPENDENT_LORENZ_FULL_RANK_SVD_REFERENCE',replication='INCOMPLETE',
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='INDEPENDENT_REFERENCE_ONLY',
        diagnostics=diagnostics,wall_seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
