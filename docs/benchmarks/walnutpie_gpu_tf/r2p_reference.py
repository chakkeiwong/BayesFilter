"""Independent diagnostic/reference only: author NumPy integrator + paper tree.

Never imported by runtime kernels. Whole-tree recurrence follows paper eqs
13–16 and Appendix A, including both backward weights (author tree omits one).
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PIN=ROOT/'.localresources/walnuts-r2p-895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a/WALNUTSpy'

def author_module():
    sys.path.insert(0,str(PIN))
    try:
        spec=importlib.util.spec_from_file_location('walnuts_author_integrators',PIN/'adaptiveIntegrators.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path.pop(0)
    return module

class Tape:
    def __init__(self,key):self.key=key;self.counter=0
    def uniform(self):
        v=((self.counter*73+self.key*19)%997+.5)/997.
        self.counter+=1;return v
    def normal(self,d):
        self.counter+=1;return np.array([((self.key*13+j*17)%101-50)/19. for j in range(d)])

def target(kind):
    def f(q):
        if kind==0:return -.5*np.dot(q,q),-q
        if kind==1:
            x,y=q[::2],q[1::2];z=y-.3*(x*x-1)
            g=np.empty_like(q);g[::2]=-x+.6*x*z;g[1::2]=-z
            return -.5*np.sum(x*x+z*z),g
        v=q[0];scale=np.exp(-v);sq=np.dot(q[1:],q[1:]);g=-scale*q.copy()
        g[0]=-v/9-.5*(len(q)-1)+.5*scale*sq
        return -v*v/18-.5*(len(q)-1)*v-.5*scale*sq,g
    return f

def macro(author,q,p,g,lp,h,model,tape,maxc=4,delta=.5,prob=2/3):
    old=np.random.uniform;np.random.uniform=tape.uniform
    try:
        r=author.adaptLeapFrogR2P(q,p,g,-lp+.5*np.dot(p,p),abs(h),1 if h>0 else -1,
            model,delta,author.integratorAuxPar(maxC=maxc,R2Pprob0=prob))
    finally:np.random.uniform=old
    # Author -700 sentinel means exact zero in the mathematical algorithm.
    correction=float('-inf') if r.lwt < -600 else r.lwt
    return r,correction

def transition(author,kind,q,key,epsilon,maxdepth=4,maxc=4,delta=.5,jitter=.2):
    model=target(kind);rng=Tape(key);lp,g=model(q);p=rng.normal(len(q))
    initial=dict(q=q.copy(),p=p,g=g,lp=lp,joint=lp-.5*np.dot(p,p),correction=0.)
    initial['weight']=initial['joint'];old=[initial];selected=initial
    depth=0;ev=0;macros=0;ratios=[]
    def turned(a,b):
        d=b['q']-a['q'];return np.dot(a['p'],d)<0 or np.dot(b['p'],d)<0
    def total(points):
        w=np.array([p['weight'] for p in points]);a=np.max(w)
        return -np.inf if not np.isfinite(a) else a+np.log(np.sum(np.exp(w-a)))
    def grow(start,level,forward):
        nonlocal ev,macros
        if level==0:
            h=epsilon*(1+jitter*(2*rng.uniform()-1)) if jitter else epsilon
            r,ratio=macro(author,start['q'],start['p'],start['g'],start['lp'],h if forward else -h,model,rng,maxc,delta)
            ev+=r.nEvalF+r.nEvalB;macros+=1;ratios.append(ratio)
            joint=r.lp-.5*np.dot(r.v,r.v);corr=start['correction']+ratio
            p=dict(q=r.q,p=r.v,g=r.grad,lp=r.lp,joint=joint,correction=corr,weight=joint+corr)
            finite=np.isfinite(joint) and np.all(np.isfinite(r.q)) and np.all(np.isfinite(r.grad))
            return [p],p,finite
        left,a,ok=grow(start,level-1,forward)
        if not ok:return left,a,False
        right,b,ok=grow(left[-1] if forward else left[0],level-1,forward)
        points=left+right if forward else right+left
        if not ok or turned(points[0],points[-1]):return points,a,False
        u=np.log(rng.uniform());den=np.logaddexp(total(left),total(right))
        choose=np.isfinite(den) and u < total(right)-den
        return points,b if choose else a,True
    for level in range(maxdepth):
        forward=rng.uniform()<.5
        ext,pick,ok=grow(old[-1] if forward else old[0],level,forward)
        if not ok:break
        if np.log(rng.uniform())<total(ext)-total(old):selected=pick
        old=old+ext if forward else ext+old;depth+=1
        if turned(old[0],old[-1]):break
    return dict(q=selected['q'],lp=selected['lp'],g=selected['g'],depth=depth,evaluations=ev,
        macros=macros,counter=rng.counter,ratios=ratios)
