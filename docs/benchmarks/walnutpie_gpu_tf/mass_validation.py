"""Bounded engineering checks for supplied diagonal/dense metrics on GPU XLA."""
import argparse
from dataclasses import asdict
import json
import time
from .worker import setup,graph_report
from .core import Config
from .api import make_supplied_mass_functions
from .mass_coordinates import make_dense_mass_functions,validate_mass
from .replication import write


def main():
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','gpu'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    tf,report=setup(a.device);report['cases']=[]
    report['role']='engineering only, no posterior or performance promotion'
    model=lambda q:-.5*tf.reduce_sum(q*q,axis=-1)
    with tf.device('/GPU:0' if a.device=='gpu' else '/CPU:0'):
        q=tf.constant([[.2,.4],[-.5,.3]],tf.float64)
        seeds=tf.constant([[17,0],[29,0]],tf.int32);epsilon=tf.constant([.1,.2],tf.float64)
        for variant in ('deterministic','r2p'):
            for dense in (False,True):
                for adapt in (False,True):
                    c=Config(2,2,warmup=4,draws=4,max_depth=3,max_halvings=3,
                        variant=variant,r2p_max_exponent=3,adapt_mass=adapt)
                    f=(make_dense_mass_functions if dense else make_supplied_mass_functions)(
                        tf,model,c,adapt_step=False)
                    # Validation/eigenvalues are an explicit host boundary.
                    with tf.device('/CPU:0'):
                        mass=validate_mass(tf,[[4.,1.],[1.,2.]] if dense else [2.,3.],c,dense=dense)
                    args=(q,seeds,mass,epsilon)
                    started=time.perf_counter();r=f.fit(*args);r.sampling.draws.numpy()
                    seconds=time.perf_counter()-started
                    if not bool(tf.reduce_all(r.valid).numpy()):raise AssertionError('invalid metric lifecycle')
                    if dense:
                        initial=f.physical_mass(r.initial);warm=f.physical_mass(r.warmup.state)
                        final=f.physical_mass(r.sampling.state)
                    else:
                        initial=1/r.initial.inverse_mass;warm=1/r.warmup.state.inverse_mass
                        final=1/r.sampling.state.inverse_mass
                    tf.debugging.assert_near(initial,mass,atol=2e-11,rtol=2e-11)
                    tf.debugging.assert_equal(final,warm)
                    if adapt:
                        if not bool(tf.reduce_any(tf.abs(warm-initial)>1e-8).numpy()):raise AssertionError('metric did not adapt')
                    else:tf.debugging.assert_near(warm,initial,atol=2e-11,rtol=2e-11)
                    name='%s-%s-%s'%(variant,'dense' if dense else 'diagonal','adaptive' if adapt else 'fixed')
                    graph=graph_report(f.fit,args,a.output/(name+'.hlo.txt'))
                    report['cases'].append(dict(name=name,config=asdict(c),passed=True,graph=graph,
                        first_call_seconds=seconds,initial_mass=initial.numpy().tolist(),
                        final_mass=final.numpy().tolist()))
                    write(a.output/'progress.json',dict(completed=len(report['cases']),total=8))
    if a.device=='gpu':report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    report.update(status='passed',wall_seconds=time.monotonic()-begin)
    write(a.output/'result.json',report)
    print(json.dumps(dict(status='passed',cases=len(report['cases']),seconds=report['wall_seconds'])),flush=True)

if __name__=='__main__':main()
