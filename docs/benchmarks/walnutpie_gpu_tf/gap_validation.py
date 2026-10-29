"""Independent diagnostic qualification; NumPy only for reference comparisons."""
import argparse
from dataclasses import replace
import hashlib,json,math,os
from pathlib import Path
import resource,subprocess,time
from .worker import setup,graph_report,model_for
from .core import Config,Kernel
from .api import make_functions,initial_state
from .r2p import R2PKernel
from .initialization import Initialization,make_initializers
from .control import RunControl,make_controllers,drive_chunks
ROOT=Path(__file__).resolve().parents[3]


def validate(tf,out,report):
    import numpy as np  # Diagnostic independent reference only.
    from .r2p_reference import author_module,target,Tape,macro,transition
    author=author_module();checks=[]
    def close(x,y):np.testing.assert_allclose(x,y,rtol=2e-11,atol=2e-11)
    compiler=['g++','-O3','-std=c++20','-pthread',
        '-I'+str(ROOT/'.localresources/cmdstan/cmdstan-2.37.0/stan/lib/stan_math/lib/eigen_3.4.0'),
        '-I'+str(ROOT/'.localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953/include'),
        str(Path(__file__).with_name('gap_native.cc')),'-o',str(out/'gap-native')]
    build=subprocess.run(compiler,text=True,capture_output=True,timeout=90,check=True)
    (out/'native-build.log').write_text(build.stdout+build.stderr)
    native=json.loads(subprocess.check_output([str(out/'gap-native')],text=True,timeout=20))
    report['native_reference']=dict(command=compiler,values=native,
        binary_sha256=hashlib.sha256((out/'gap-native').read_bytes()).hexdigest())
    cfg=Config(3,2,init_search_limit=None)
    q=tf.constant([[.2,.4],[-.5,.3],[.7,-.8]],tf.float64)
    seeds=tf.constant([[17,0],[18,0],[19,0]],tf.int32)
    k=Kernel(tf,model_for(tf,0),cfg,tape=True)
    @tf.function(input_signature=[tf.TensorSpec([3,2],tf.float64),tf.TensorSpec([3,2],tf.int32)],jit_compile=True,autograph=False)
    def init(q,seeds):
        lp,g=k.value_gradient(q);mass=.99999*tf.abs(g)+.00001
        avg=tf.broadcast_to(tf.exp(tf.reduce_mean(tf.math.log(mass),0))[None,:],q.shape)
        return initial_state(k,q,seeds,avg)
    state=init(q,seeds);close(1/state.inverse_mass.numpy(),[native['mass']]*3);close(state.epsilon.numpy(),native['steps'])
    controller=make_controllers(tf,model_for(tf,0),cfg)
    snap=state._replace(epsilon=tf.constant([.2,.25,.18],tf.float64),inverse_mass=1/tf.constant([[1.,2.],[1.5,2.3],[.9,2.2]],tf.float64))
    stats=controller.warmup_statistics(snap);close([float(x.numpy()) for x in stats],native['warmup_statistics'])
    stat=controller.sampling_statistic(tf.constant([-2.,-2.1,-1.9],tf.float64),tf.constant([9.,10.8,7.2],tf.float64),tf.constant(10))
    close(stat.numpy(),native['sampling_statistic']);checks.append('native_initialization_and_controller_formulas')
    report['initialization_graph']=graph_report(init,(q,seeds),out/'initialize.hlo.txt')
    boundary=make_initializers(tf,model_for(tf,0),cfg,Initialization(mass='unit'))
    bad=boundary.initialize(q,seeds,tf.ones_like(q),tf.constant([0.,-.1,float('inf')],tf.float64))
    np.testing.assert_array_equal(bad.valid.numpy(),[False]*3)
    np.testing.assert_array_equal(bad.seed.numpy(),seeds.numpy())
    checks.append('uncapped_initialization_invalid_steps_terminate')
    random_init=make_initializers(tf,model_for(tf,0),cfg,
        Initialization(positions='normal',scale=2.,average_masses=True))
    inputs=(q,seeds,tf.ones_like(q),tf.fill([3],tf.constant(.1,tf.float64)))
    randomized=random_init.initialize(*inputs)
    expected,advanced=random_init.random_positions(seeds)
    close(randomized.q.numpy(),expected.numpy())
    np.testing.assert_array_equal(randomized.seed.numpy()[:,1],advanced.numpy()[:,1]+1)
    assert bool(tf.reduce_all(randomized.valid).numpy())
    zero_init=make_initializers(tf,model_for(tf,0),cfg,
        Initialization(positions='zero',mass='unit',adapt_step=False))
    zero=zero_init.initialize(*inputs)
    close(zero.q.numpy(),tf.zeros_like(q).numpy());close(zero.inverse_mass.numpy(),tf.ones_like(q).numpy())
    assert bool(tf.reduce_all(zero.valid).numpy())
    report['random_initialization_graph']=graph_report(random_init.initialize,inputs,out/'random-initialize.hlo.txt')
    checks.append('compiled_random_and_zero_initialization')
    # Source integrator branch coverage, selected using fixed fixture values.
    rows=[];coverage={'baseline':0,'finer':0,'nonzero_correction':0,'zero_reverse_support':0,'cap':0}
    for kind in range(3):
        f=target(kind)
        for key in range(1,49):
            qq=np.array([.15*((key+j*3)%11-5) for j in range(4)])
            if kind==2:qq[0]=key%9-4.
            lp,g=f(qq);p=np.array([((key*13+j*17)%101-50)/19. for j in range(4)])
            h=2.**(key%6-2);rng=Tape(key)
            r,ratio=macro(author,qq,p,g,lp,h,f,rng,maxc=4)
            rows.append((kind,key,qq,p,lp,g,h,r,ratio))
            coverage['baseline']+=r.c==r.If;coverage['finer']+=r.c==r.If+1
            coverage['nonzero_correction']+=math.isfinite(ratio) and abs(ratio)>1e-8
            coverage['zero_reverse_support']+=not math.isfinite(ratio);coverage['cap']+=r.If==4
    for kind in range(3):
        rr=[r for r in rows if r[0]==kind];n=len(rr)
        cfg=Config(n,4,variant='r2p',r2p_max_exponent=4,r2p_jitter=0,max_depth=4)
        k=R2PKernel(tf,model_for(tf,kind),cfg,tape=True)
        q=tf.constant([x[2] for x in rr],tf.float64);p=tf.constant([x[3] for x in rr],tf.float64)
        h=tf.constant([x[6] for x in rr],tf.float64);seed=tf.constant([[x[1],0] for x in rr],tf.int32)
        @tf.function(input_signature=[tf.TensorSpec(q.shape,tf.float64),tf.TensorSpec(p.shape,tf.float64),tf.TensorSpec(h.shape,tf.float64),tf.TensorSpec(seed.shape,tf.int32),tf.TensorSpec(q.shape,tf.float64)],jit_compile=True,autograph=False)
        def call(q,p,h,seed,inv):
            lp,g=k.value_gradient(q)
            return k.r2p_macro(k.point(q,p,g,lp,inv),inv,h,k.active(),seed,k.adam_init(h))
        result=call(q,p,h,seed,tf.ones_like(q))
        close(result.point.q.numpy(),[x[7].q for x in rr]);close(result.point.p.numpy(),[x[7].v for x in rr])
        close(result.point.lp.numpy(),[x[7].lp for x in rr]);close(result.point.g.numpy(),[x[7].grad for x in rr])
        close(result.log_ratio.numpy(),[x[8] for x in rr])
        for field,name in [('forward','If'),('backward','Ib'),('exponent','c'),('forward_evaluations','nEvalF'),('backward_evaluations','nEvalB')]:
            np.testing.assert_array_equal(getattr(result,field).numpy(),[getattr(x[7],name) for x in rr])
        np.testing.assert_array_equal(result.seed.numpy()[:,1],np.ones(n,dtype=int))
        # Diagonal metric conjugacy: q=sqrt(inv)*z and v=sqrt(inv)*p.
        inv_values=np.array([[.5+.1*((row+j)%9) for j in range(4)] for row in range(n)])
        result=call(q,p,h,seed,tf.constant(inv_values,tf.float64))
        for i,x in enumerate(rr):
            scale=np.sqrt(inv_values[i]);base=target(kind)
            def transformed(z):
                value,grad=base(scale*z);return value,scale*grad
            z=x[2]/scale;v=x[3]*scale;lp,gz=transformed(z)
            native,ratio=macro(author,z,v,gz,lp,x[6],transformed,Tape(x[1]),maxc=4)
            close(result.point.q.numpy()[i],native.q*scale)
            close(result.point.p.numpy()[i],native.v/scale)
            close(result.point.g.numpy()[i],native.grad/scale)
            close(result.point.lp.numpy()[i],native.lp)
            close(result.log_ratio.numpy()[i],ratio)
            assert int(result.exponent.numpy()[i])==native.c
        checks.append('diagonal_metric_conjugacy_kind_'+str(kind))
    assert all(coverage.values()),coverage
    report['author_macro_cases']=len(rows);report['diagonal_metric_cases']=len(rows);report['author_macro_coverage']={k:int(v) for k,v in coverage.items()}
    checks.append('unmodified_author_r2p_macro_parity')
    # Independent equation-level whole-tree reference uses author macro only.
    all_graphs=[];cases=0
    for kind in range(3):
        refs=[]
        for key in range(1,25):
            q=np.array([.12*((key+j*3)%11-5) for j in range(4)])
            if kind==2:q[0]=key%7-3.
            h=2.**(key%4-2)
            refs.append((q,h,key,transition(author,kind,q,key,h)))
        for layout in ('fieldwise','packed','flat'):
            cfg=Config(len(refs),4,variant='r2p',r2p_max_exponent=4,max_depth=4,execution_layout=layout)
            k=R2PKernel(tf,model_for(tf,kind),cfg,tape=True)
            q=tf.constant([r[0] for r in refs],tf.float64);h=tf.constant([r[1] for r in refs],tf.float64);seed=tf.constant([[r[2],0] for r in refs],tf.int32)
            @tf.function(input_signature=[tf.TensorSpec(q.shape,tf.float64),tf.TensorSpec(h.shape,tf.float64),tf.TensorSpec(seed.shape,tf.int32)],jit_compile=True,autograph=False)
            def step(q,h,seed):
                lp,g=k.value_gradient(q)
                return k.transition(q,lp,g,h,tf.ones_like(q),tf.ones([len(refs)],tf.int32),seed,k.adam_init(h))
            result=step(q,h,seed)
            for field in ('q','lp','g'):close(getattr(result,field).numpy(),[r[3][field] for r in refs])
            for field in ('depth','evaluations','macros'):np.testing.assert_array_equal(getattr(result,field).numpy(),[r[3][field] for r in refs])
            np.testing.assert_array_equal(result.seed.numpy()[:,1],[r[3]['counter'] for r in refs])
            cases+=len(refs)
            if layout=='packed':all_graphs.append(graph_report(step,(q,h,seed),out/f'r2p-{kind}.hlo.txt'))
    checks.append('paper_weighted_whole_tree_all_layouts');report['whole_tree_cases']=cases
    # Explicit full-fit composition, runtime data, frozen continuation and controls.
    cfg=Config(3,4,warmup=3,draws=4,max_depth=3,variant='r2p',r2p_max_exponent=4,execution_layout='packed')
    model=lambda q,shift:-.5*tf.reduce_sum((q-shift)**2,-1)
    spec=[tf.TensorSpec([4],tf.float64)];f=make_functions(tf,model,cfg,spec)
    q=tf.constant([[.2,.3,-.1,.5],[-.3,.4,.7,-.2],[.1,-.5,.2,.4]],tf.float64)
    seeds=tf.constant([[17,0],[29,0],[43,0]],tf.int32);data=tf.zeros([4],tf.float64)
    start=time.perf_counter();fit=f.fit(q,seeds,data);bool(tf.reduce_all(fit.valid).numpy());first=time.perf_counter()-start
    assert bool(tf.reduce_all(fit.valid).numpy())
    sample=f.sample(fit.warmup.state,data);close(sample.draws.numpy(),fit.sampling.draws.numpy())
    continued=f.sample(sample.state,data)
    long=make_functions(tf,model,replace(cfg,draws=8),spec).sample(fit.warmup.state,data)
    close(tf.concat([sample.draws,continued.draws],0).numpy(),long.draws.numpy())
    changed=f.fit(q,seeds,data+1);assert not np.allclose(changed.sampling.draws.numpy(),fit.sampling.draws.numpy())
    policy=RunControl(warmup_min=4,warmup_max=4,sample_min=6,sample_max=6,chunk_size=2)
    ctrl=make_controllers(tf,model,cfg,policy,spec);warm=ctrl.warmup(ctrl.begin(fit.initial),data)
    retained=ctrl.sample(ctrl.begin(warm.state.sampler),data)
    driven=drive_chunks(ctrl.sample_chunk,ctrl.begin(warm.state.sampler),data)
    close(tf.concat([x.draws for x in driven['chunks']],0).numpy(),retained.draws.numpy())
    close(retained.state.sampler.inverse_mass.numpy(),warm.state.sampler.inverse_mass.numpy())
    assert int(retained.state.count.numpy())==6
    exhausted=ctrl.sample_chunk(ctrl.begin(warm.state.sampler)._replace(count=tf.constant(6)),data)
    assert int(exhausted.state.reason.numpy())==2
    assert not bool(tf.reduce_any(exhausted.mask).numpy())
    # Early stop and padded storage; stopping is explicitly legacy agreement only.
    early=make_controllers(tf,model,cfg,RunControl(warmup_min=2,warmup_max=8,warmup_stride=1,step_tolerance=1e10,mass_tolerance=1e10,sample_min=2,sample_max=8,logp_rhat_tolerance=1e10),spec)
    e=early.warmup(early.begin(fit.initial),data);assert int(e.state.count.numpy())==2
    np.testing.assert_array_equal(e.mask.numpy(),[True,True,False,False,False,False,False,False])
    checks.extend(['r2p_runtime_data_fit_continuation','compiled_control_caps_early_stop_padding_continuation'])
    all_graphs.append(graph_report(f.fit,(q,seeds,data),out/'r2p-fit.hlo.txt'))
    all_graphs.append(graph_report(ctrl.warmup,(ctrl.begin(fit.initial),data),out/'controlled-warmup.hlo.txt'))
    all_graphs.append(graph_report(ctrl.sample,(ctrl.begin(warm.state.sampler),data),out/'controlled-sample.hlo.txt'))
    checks.append('all_macros_and_tree_traces_bounded')
    report.update(checks=checks,graphs=all_graphs,output_device=fit.sampling.draws.device,
        first_call_seconds=first,performance_interpretation='debug timing only; no comparative ranking',
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if report['device']=='gpu':
        assert 'GPU:0' in fit.sampling.draws.device
        report['allocator']=tf.config.experimental.get_memory_info('GPU:0')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','gpu'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tf,report=setup(a.device);begin=time.monotonic()
    try:
        with tf.device('/GPU:0' if a.device=='gpu' else '/CPU:0'):validate(tf,a.output,report)
        report['status']='passed'
    except Exception as e:report.update(status='failed',error=repr(e));raise
    finally:
        report['wall_seconds']=time.monotonic()-begin
        (a.output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
