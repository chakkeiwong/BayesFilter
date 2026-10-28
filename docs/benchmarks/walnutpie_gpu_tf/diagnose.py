"""Post-run diagnostic repair/analysis; never changes sampling/adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
from .worker import setup, diagnostics, write

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('runs',type=Path,nargs='+');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tf,report=setup('cpu');report['cases']=[];report['source_draw_hashes']={}
    for directory in a.runs:
        meta=json.loads((directory/'result.json').read_text())
        for item in meta['targets']:
            name=item['name'];paths=[directory/(name+'-draws.json'),directory/(name+'-native.json')]
            for path in paths:report['source_draw_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            values=json.loads(paths[0].read_text());native=json.loads(paths[1].read_text())
            draws=tf.constant(values,tf.float64)
            ndraws=tf.transpose(tf.constant([r['draws'] for r in native['chains']],tf.float64),[1,0,2])
            results={}
            for engine,x in [('tf',draws),('native',ndraws)]:
                d=diagnostics(tf,x)
                if name=='funnel':d['fraction_v_below_minus3']=float(tf.reduce_mean(tf.cast(x[:,:,0]<-3.,tf.float64)).numpy())
                # Analytic transforms have standard-normal marginal references.
                if name=='gaussian':z=x
                elif name=='banana':
                    a0=x[:,:,::2];b=x[:,:,1::2]-.3*(a0*a0-1.)
                    z=tf.reshape(tf.stack([a0,b],axis=-1),x.shape)
                elif name=='funnel':z=tf.concat([x[:,:,:1]/3.,x[:,:,1:]*tf.exp(-x[:,:,:1]/2.)],-1)
                else:z=None
                if z is not None:
                    d['reference_max_abs_mean']=float(tf.reduce_max(tf.abs(tf.reduce_mean(z,[0,1]))).numpy())
                    d['reference_max_abs_variance_error']=float(tf.reduce_max(tf.abs(tf.math.reduce_variance(z,[0,1])-1.)).numpy())
                results[engine]=d
            report['cases'].append(dict(run=str(directory),device=meta['device'],target=name,diagnostics=results,
                steady_seconds=item['steady_seconds'],native_parallel_seconds=item['native_parallel_seconds'],
                interpretation='descriptive short-chain diagnostics; not convergence or sampler efficiency promotion'))
    report['cross_device_draw_comparison']=[]
    if len(a.runs)>=2:
        left,right=a.runs[:2]
        for path in left.glob('*-draws.json'):
            other=right/path.name
            if not other.exists():continue
            x=tf.constant(json.loads(path.read_text()),tf.float64);y=tf.constant(json.loads(other.read_text()),tf.float64)
            report['cross_device_draw_comparison'].append(dict(target=path.stem,exact=bool(tf.reduce_all(x==y).numpy()),
                max_scaled_error=float(tf.reduce_max(tf.abs(x-y)/(1+tf.abs(x))).numpy()),
                note='Adaptive branch amplification may change long trajectories; shared-tape oracles test source parity.'))
    report['repair']='Corrected diagnostic API axis: expects [chain,draw,parameter], sampler emits [draw,chain,parameter]. Original benchmark diagnostics superseded; timings/draws unchanged.'
    write(a.output/'diagnostics.json',report)
    print(json.dumps({'cases':len(report['cases']),'output':str(a.output)}))

if __name__=='__main__':main()
