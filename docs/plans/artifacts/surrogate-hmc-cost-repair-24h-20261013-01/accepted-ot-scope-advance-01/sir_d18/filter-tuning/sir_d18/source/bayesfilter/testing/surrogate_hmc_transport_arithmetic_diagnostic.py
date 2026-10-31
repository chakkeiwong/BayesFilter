"""Diagnostic-only algebraic transport candidates; no runtime or tuning authority."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared
from bayesfilter.highdim.analytical_jet_tf import Jet
from bayesfilter.testing.surrogate_hmc_stage_arithmetic_diagnostic import difference,cast_inputs


def candidate_source(source,kind):
    if kind not in ('gauge','direct','direct_gauge'):
        raise ValueError('unknown diagnostic transport candidate')
    def substitute(old,new):
        nonlocal source
        if source.count(old)!=1:
            raise ValueError('transport source changed; candidate requires re-audit')
        source=source.replace(old,new)
    if 'direct' in kind:
        start=source.index('                xi=tile(centred_x,i,k,1);')
        stop=source.index('                result=logsumexp(',start)
        old=source[start:stop]
        new="""                xi=tile(x.v,i,k,1); xj=tile(x.v,j,k,1)
                di=tile(centred_t,i,k,2); dj=tile(centred_t,j,k,2)
                delta=xi[:,:,None]-xj[:,None]
                derivative=(-2/epsilon)*(tf.einsum("bijd,pbid->pbij",delta,di)-tf.einsum("bijd,pbjd->pbij",delta,dj))
                derivative=tf.reduce_sum(weights[None]*(derivative+part.t[:,: ,None,:]),-1)
"""
        substitute(old,new)
    if 'gauge' in kind:
        substitute('        return it+1,gamma.v,gamma.t,alpha.v,alpha.t',
            '        shift=gamma[:,:1]\n        gamma=gamma-shift; alpha=alpha+shift\n        return it+1,gamma.v,gamma.t,alpha.v,alpha.t')
        substitute('    alpha=Jet(av,at); gamma=Jet(gv,gt)\n    normalizer=',
            '    alpha=Jet(av,at); gamma=Jet(gv,gt)\n    shift=alpha[:,:1]; alpha=alpha-shift; gamma=gamma+shift\n    normalizer=')
    return '# DIAGNOSTIC CANDIDATE ONLY: '+kind+'\n'+source


def load_candidate(path,kind):
    source=candidate_source(Path(shared.__file__).read_text(),kind)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    spec=importlib.util.spec_from_file_location('diagnostic_transport_'+kind,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.transport,hashlib.sha256(source.encode()).hexdigest()


def graph(fn,b,n,d,p,dtype,controls):
    sig=[tf.TensorSpec(shape,dtype) for shape in ((b,n,d),(p,b,n,d),(b,n),(p,b,n))]
    @tf.function(input_signature=sig,jit_compile=True,autograph=False)
    def kernel(x,xt,w,wt):
        y,row,col=fn(Jet(x,xt),Jet(w,wt),controls)
        return y.v,y.t,row,col
    return kernel


def run_diagnostic(case,output,saved_root,observations=(2,3,4,5,6)):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from dataclasses import asdict
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    saved=json.loads((Path(saved_root)/'stage-arithmetic.json').read_text())
    if saved['status']!='STAGE_ARITHMETIC_LOCALIZED' or saved['model']!=case.name or saved['controls']!=asdict(case.controls):
        raise ValueError('saved transport scope mismatch')
    n,d,p=saved['particles'],saved['state_dimension'],saved['parameter_count']
    b=saved['stages'][0]['saved_reset_inputs']['child']['shape'][0]
    if n!=case.fixed[0].shape[0] or d!=case.spec.dimension or p!=case.p:
        raise ValueError('saved transport dimensions mismatch')
    funcs={'baseline':graph(shared.transport,b,n,d,p,tf.float32,case.controls),
           'reference':graph(shared.transport,b,n,d,p,tf.float64,case.controls)}
    result=dict(schema='transport_arithmetic_localization_v1',status='RUNNING',
        evidence_role='diagnostic_candidates_only',model=case.name,particles=n,
        state_dimension=d,parameter_count=p,controls=asdict(case.controls),
        source=str(saved_root),source_sha256={},stages=[],warm_seconds={},
        jit_compile=True,tf32=False,cpu_only=not bool(tf.config.list_physical_devices('GPU')),
        posterior_complete=False)
    for kind in ('gauge','direct','direct_gauge'):
        fn,digest=load_candidate(output/'source'/(kind+'.py'),kind)
        result['source_sha256'][kind]=digest
        funcs[kind]=graph(fn,b,n,d,p,tf.float32,case.controls)
    destination=output/'transport-arithmetic.json';runtime.save(destination,result)
    start=time.monotonic();healthy=True
    for index in observations:
        inputs=[];hashes={};step=saved['stages'][index-1]
        if step['observation']!=index: raise ValueError('observation mismatch')
        for key in ('child','child_tangent','log_weights','log_weight_tangent'):
            meta=step['saved_reset_inputs'][key];raw=Path(meta['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=meta['sha256'] or meta['dtype']!='float32':
                raise ValueError('saved input checksum or dtype mismatch')
            x=tf.io.parse_tensor(raw,tf.float32)
            if x.shape.as_list()!=meta['shape']:raise ValueError('saved input shape mismatch')
            inputs.append(x);hashes[key]=meta['sha256']
        ref=funcs['reference'](*cast_inputs(inputs,tf.float64));row=dict(observation=index,inputs_sha256=hashes,comparisons={})
        for name in ('baseline','gauge','direct','direct_gauge'):
            out=funcs[name](*inputs)
            checks={field:difference(out[j],ref[j],axis) for j,(field,axis) in enumerate((('value',0),('tangent',1)))}
            row['comparisons'][name]=checks
            row.setdefault('transport_diagnostics',{})[name]=dict(row_mass=out[2],column_mass=out[3])
            healthy=healthy and all(v['finite'] for v in checks.values())
        result['stages'].append(row);runtime.save(destination,result)
        if not healthy:break
        if index==5:
            names=('baseline','gauge','direct','direct_gauge')
            for repetition in range(3):
                for name in names[::1 if repetition%2==0 else -1]:
                    begin=time.monotonic();out=funcs[name](*inputs)
                    for x in out:x.numpy()
                    result['warm_seconds'].setdefault(name,[]).append(time.monotonic()-begin)
            runtime.save(destination,result)
    nominated=[]
    if healthy and all(index in observations for index in (4,5,6)):
        for name in ('gauge','direct','direct_gauge'):
            selected=[s for s in result['stages'] if s['observation'] in (4,5,6)]
            if len(selected)==3 and all(float(s['comparisons'][name]['tangent']['maximum_scaled'])<=.2*float(s['comparisons']['baseline']['tangent']['maximum_scaled']) and float(s['comparisons'][name]['value']['maximum_scaled'])<=.01 for s in selected):
                nominated.append(name)
    result.update(status='TRANSPORT_ARITHMETIC_LOCALIZED' if healthy else 'INVALID_TRANSPORT_ARITHMETIC',
        nominated=nominated,wall_seconds=time.monotonic()-start,
        trace_counts={name:fn.experimental_get_tracing_count() for name,fn in funcs.items()})
    runtime.save(destination,result)
    return result
