"""Diagnostic-only identical-input decomposition of the actual shared reset.

Saved SIR tensors are independent localization inputs, never runtime/tuning data.
Graphs call shared primitives; host staging can change fusion. A comparison to
actual reset_cloud is mandatory and particle/tangent gaps remain explanatory.
"""
from pathlib import Path
import hashlib
import time
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_batch_tf as shared
from bayesfilter.highdim.analytical_jet_tf import Jet, exp, mv, stack
from bayesfilter.testing.surrogate_hmc_stage_arithmetic_diagnostic import difference, cast_inputs


def make_stages(n,d,p,b,dtype,controls):
    tensor=lambda shape: tf.TensorSpec(shape,dtype)
    cloud=[tensor([b,n,d]),tensor([p,b,n,d])]
    weights=[tensor([b,n]),tensor([p,b,n])]
    matrix=[tensor([b,d,d]),tensor([p,b,d,d])]
    mean=[tensor([b,d]),tensor([p,b,d])]
    def pair(x): return x.v,x.t
    @tf.function(input_signature=cloud+weights,jit_compile=True,autograph=False)
    def weighted(xv,xt,lv,lt):
        mu,cov=shared.moments(Jet(xv,xt),exp(Jet(lv,lt)))
        return *pair(mu),*pair(cov)
    @tf.function(input_signature=cloud+weights,jit_compile=True,autograph=False)
    def transport(xv,xt,lv,lt):
        y,row,col=shared.transport(Jet(xv,xt),Jet(lv,lt),controls)
        return *pair(y),row,col
    @tf.function(input_signature=cloud,jit_compile=True,autograph=False)
    def unweighted(xv,xt):
        x=Jet(xv,xt)
        mu,cov=shared.moments(x,x.constant(tf.fill([b,n],tf.constant(1./n,dtype))))
        return *pair(mu),*pair(cov)
    @tf.function(input_signature=matrix+matrix,jit_compile=True,autograph=False)
    def recolour(pv,pt,sv,st):
        cov,sy=Jet(pv,pt),Jet(sv,st)
        factors=[shared.recolour(cov[row],sy[row]) for row in range(b)]
        a=stack([v[0] for v in factors])
        return *pair(a),tf.stack([v[1] for v in factors]),tf.stack([v[2] for v in factors])
    @tf.function(input_signature=mean+cloud+mean+matrix,jit_compile=True,autograph=False)
    def restore(mv0,mt,yv,yt,ymv,ymt,av,at):
        mu,y,ym,a=Jet(mv0,mt),Jet(yv,yt),Jet(ymv,ymt),Jet(av,at)
        centered=mv(a[:,None],y-ym[:,None])
        centered=centered-centered.mean(1)[:,None]
        return pair(mu[:,None]+centered)
    @tf.function(input_signature=cloud+weights,jit_compile=True,autograph=False)
    def fused(xv,xt,lv,lt):
        z,mu,cov,valid,diagnostics=shared.reset_cloud(Jet(xv,xt),Jet(lv,lt),controls)
        return *pair(z),valid,diagnostics
    return weighted,transport,unweighted,recolour,restore,fused


def compare(a,b,fields):
    return {name:difference(a[i],b[i],axis) for i,(name,axis) in enumerate(fields)}


def run_diagnostic(case,output,saved_stage_root,observations=(1,2,3,4,5,6)):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    output=Path(output); source=Path(saved_stage_root)
    output.mkdir(parents=True,exist_ok=True)
    data_path=source/'stage-arithmetic.json'
    import json
    saved=json.loads(data_path.read_text())
    n,d,p=saved['particles'],saved['state_dimension'],saved['parameter_count']
    b=saved['stages'][0]['saved_reset_inputs']['child']['shape'][0]
    if (saved['status']!='STAGE_ARITHMETIC_LOCALIZED' or saved['model']!=case.name
            or n!=case.fixed[0].shape[0] or d!=case.spec.dimension or p!=case.p
            or case.precision_policy!='uniform'):
        raise ValueError('saved reset scope mismatch')
    from dataclasses import asdict
    if saved['controls']!=asdict(case.controls):
        raise ValueError('saved controls differ')
    funcs={dt:make_stages(n,d,p,b,dt,case.controls) for dt in (tf.float32,tf.float64)}
    fields4=(('mean',0),('mean_tangent',1),('covariance',0),('covariance_tangent',1))
    fields2=(('value',0),('tangent',1))
    result=dict(schema='surrogate_hmc_reset_arithmetic_v1',status='RUNNING',
        evidence_role='identical_input_reset_primitive_localization_only',source=str(source),
        particles=n,state_dimension=d,parameter_count=p,batch_size=b,
        source_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        controls=saved['controls'],cpu_only=not bool(tf.config.list_physical_devices('GPU')),
        jit_compile=True,stages=[],posterior_complete=False)
    destination=output/'reset-arithmetic.json'
    runtime.save(destination,result)
    start=time.perf_counter(); healthy=True
    for index in observations:
        step=saved['stages'][index-1]
        if step['observation']!=index: raise ValueError('observation mismatch')
        args=[]; hashes={}
        for name in ('child','child_tangent','log_weights','log_weight_tangent'):
            meta=step['saved_reset_inputs'][name]
            raw=Path(meta['path']).read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=meta['sha256'] or meta['dtype']!='float32':
                raise ValueError('saved tensor checksum or dtype mismatch')
            value=tf.io.parse_tensor(raw,tf.float32)
            if value.shape.as_list()!=meta['shape']:
                raise ValueError('saved tensor shape mismatch')
            args.append(value); hashes[name]=digest
        f32,f64=funcs[tf.float32],funcs[tf.float64]
        m=f32[0](*args); mr=f64[0](*cast_inputs(args,tf.float64))
        t=f32[1](*args); tr=f64[1](*cast_inputs(args,tf.float64))
        u=f32[2](*t[:2]); ur=f64[2](*cast_inputs(t[:2],tf.float64))
        ar=f32[3](*m[2:],*u[2:]); arr=f64[3](*cast_inputs((*m[2:],*u[2:]),tf.float64))
        restore_args=(*m[:2],*t[:2],*u[:2],*ar[:2])
        z=f32[4](*restore_args); zr=f64[4](*cast_inputs(restore_args,tf.float64))
        fused=f32[5](*args); fused64=f64[5](*cast_inputs(args,tf.float64))
        validity=bool(tf.reduce_all(ar[2] & arr[2] & fused[2] & fused64[2]).numpy())
        local=dict(weighted_moments=compare(m,mr,fields4),transport=compare(t,tr,fields2),
            transported_moments=compare(u,ur,fields4),recolour=compare(ar,arr,fields2),
            restore=compare(z,zr,fields2))
        row=dict(observation=index,local=local,healthy=validity,
            full_reset=compare(fused,fused64,fields2),staged_vs_fused=compare(z,fused,fields2),
            input_sha256=hashes,relative_covariance_margin=ar[3],
            maximum_particle_tangent=tf.reduce_max(tf.abs(args[1])),
            maximum_transport_tangent=tf.reduce_max(tf.abs(t[1])))
        healthy=(healthy and validity
            and all(v['finite'] for stage in local.values() for v in stage.values())
            and all(v['finite'] for v in row['full_reset'].values())
            and all(v['finite'] for v in row['staged_vs_fused'].values()))
        result['stages'].append(row); result['wall_seconds']=time.perf_counter()-start
        runtime.save(destination,result)
        if not healthy: break
    result['status']='RESET_ARITHMETIC_LOCALIZED' if healthy else 'INVALID_RESET_ARITHMETIC'
    result['trace_counts']={dt.name:[f.experimental_get_tracing_count() for f in fs]
                            for dt,fs in funcs.items()}
    runtime.save(destination,result)
    return result
