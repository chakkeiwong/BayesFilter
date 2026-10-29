"""Independent NumPy author-reference experiments; never a TF runtime import.

The vendored files are immutable. The optional one-line correction is applied
in memory and emitted as a diff. Normalized analytic target evaluation replaces
scipy's distribution calls only after target/gradient parity is checked.
"""
import difflib
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types
import numpy as np

SOURCE = Path(__file__).parent / 'author_reference'


def check_sources():
    meta = json.loads((SOURCE/'provenance.json').read_text())
    for rel, expected in meta['sha256'].items():
        if hashlib.sha256((SOURCE/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('author source changed: '+rel)
    return meta


def load_author(corrected=False):
    check_sources()
    path = SOURCE/'WALNUTSpy'
    sys.path.insert(0, str(path))
    try:
        import adaptiveIntegrators as ai
        import targetDistr as td
        code = (path/'WALNUTS.py').read_text()
        original = code
        # The sampler never uses pyplot. Drop only its import in this headless
        # adapter, preserving all numerical author code and the copied file.
        assert code.count('plt') == 1
        code = code.replace('import matplotlib.pyplot as plt', '# Unused plotting import omitted by headless adapter')
        if corrected:
            anchor = 'timeLenB_ += HLoc1[1]'
            assert code.count(anchor) == 1
            index = code.index('                            Wnew =', code.index(anchor))
            code = code[:index]+'                            lwtSumb += intOut.lwt\n'+code[index:]
        module = types.ModuleType('walnuts_author_corrected' if corrected else 'walnuts_author_original')
        module.__file__ = str(path/'WALNUTS.py')
        exec(compile(code, module.__file__, 'exec'), module.__dict__)
    finally:
        sys.path.pop(0)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), code.splitlines(True),
                                      fromfile='author/WALNUTS.py', tofile='corrected/WALNUTS.py'))
    return module, ai, td, diff


def target(dimension):
    constant = -.5*dimension*math.log(2*math.pi)-math.log(3.)
    def model(q):
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            v=q[0]; scale=np.exp(-v); sq=np.dot(q[1:],q[1:])
            lp=constant-v*v/18.-.5*(dimension-1)*v-.5*scale*sq
            g=-scale*q.copy()
            g[0]=-v/9.-.5*(dimension-1)+.5*scale*sq
        return lp,g
    return model


def start(seed, dimension):
    np.random.seed(seed)
    q=np.zeros(dimension)
    # Literal author semantics: normal(1) has mean 1, not a size argument.
    q[0]=3.*np.random.normal(1)
    for j in range(1,dimension):q[j]=np.exp(.5*q[0])*np.random.normal(1)
    return q


def tune(dimension, output):
    wn,ai,_,_=load_author(False)
    q=start(0,dimension);begin=time.perf_counter()
    samples,d=wn.WALNUTS(target(dimension),q,integrator=ai.adaptLeapFrogR2P,
        M=12,H0=.3,delta0=.3,numIter=1000,warmupIter=1000,
        adaptDeltaTarget=.6,recordOrbitStats=False)
    elapsed=time.perf_counter()-begin
    rng=np.random.get_state()
    state=[rng[0],rng[1].tolist(),int(rng[2]),int(rng[3]),float(rng[4])]
    report=dict(dimension=dimension,step=float(d[-1,15]),delta=float(d[-1,18]),
        seconds=elapsed,q0=q.tolist(),seed=0,iterations=1000,
        rng_after_tuning=state,starts=[start(s,dimension).tolist() for s in range(4)],
        tuning='Unmodified author Appendix C quantile tuning; last reported h/delta as mainFunnel.py',
        source=check_sources(),diagnostics=d.tolist(),draws=samples.T.tolist())
    if not all(math.isfinite(report[x]) and report[x]>0 for x in ('step','delta')):
        raise ValueError('invalid author tuning')
    (output/'tuning.json').write_text(json.dumps(report,allow_nan=False))
    return {k:v for k,v in report.items() if k not in ('draws','diagnostics','rng_after_tuning','source')}


def chain(job):
    seed,dimension,draws,settle,tuning,corrected,transient=job
    wn,ai,_,diff=load_author(corrected)
    q=start(seed,dimension)
    if transient:
        q=np.zeros(dimension);q[0]=-6.
    elif seed==0:
        r=tuning['rng_after_tuning']
        np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))
    begin=time.perf_counter()
    samples,d,omin,omax=wn.WALNUTS(target(dimension),q,integrator=ai.adaptLeapFrogR2P,
        M=12,H0=tuning['step'],delta0=tuning['delta'],numIter=draws+settle,
        warmupIter=0,recordOrbitStats=True)
    return dict(seed=seed,q0=q.tolist(),seconds=time.perf_counter()-begin,
        draws=samples[:,1:].T.tolist(),diagnostics=d.tolist(),
        orbit_v_min=omin[0].tolist(),orbit_v_max=omax[0].tolist(),patch=diff)


def extreme_probe(output):
    _,ai,_,_=load_author(False)
    np.random.seed(1);q=np.zeros(11);q[0]=-30.;p=np.random.normal(size=11)
    lp,g=target(11)(q);begin=time.perf_counter()
    (output/'probe-start.json').write_text(json.dumps(dict(v=-30.,h=.3,delta=.3,
        maxC=30,seed=1,role='One author macro cost probe, not the full transient experiment')))
    r=ai.adaptLeapFrogR2P(q,p,g,-lp+.5*np.dot(p,p),.3,1,target(11),.3,
                         ai.integratorAuxPar(maxC=30))
    return dict(seconds=time.perf_counter()-begin,v=float(r.q[0]),forward=r.If,
        backward=r.Ib,chosen=r.c,evaluations=r.nEvalF+r.nEvalB)
