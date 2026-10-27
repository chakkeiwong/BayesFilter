"""Independent Stan-header arithmetic and complete XLA integration checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
import sys
import math
import subprocess
import hashlib
import json
import time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
import benchmark_nuts_xla as base
import stan_nuts_warmup as adaptation
from stan_nuts_warmup import (
    Config,stan_schedule,dual_update,regularized_metric,make_functions,
    welford_update,reasonable_step_size,leapfrog_probe,
    warmup_checked,freeze,sample_checked,FrozenState,
)

@pytest.fixture(scope='module')
def framework():
    args=base.parser().parse_args(['--device','cpu','--output','/tmp/stan-test'])
    return base.setup(args)[:2]

@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    """Always build current source; an audit run must never silently skip it."""
    stan=ROOT/'.localresources/cmdstan/cmdstan-2.37.0'
    if not stan.is_dir():
        if os.environ.get('BAYESFILTER_REQUIRE_STAN_ORACLE')=='1':
            pytest.fail('Required pinned CmdStan 2.37.0 installation is missing')
        pytest.skip('Optional native oracle needs the pinned CmdStan installation')
    folder=Path(os.environ.get('BAYESFILTER_STAN_AUDIT_OUTPUT',
                               str(tmp_path_factory.mktemp('stan-oracle'))))
    folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'adaptation_oracle'
    source=ROOT/'docs/benchmarks/stan_adaptation_oracle.cpp'
    mathroot=stan/'stan/lib/stan_math'
    includes=[stan/'stan/src',mathroot,mathroot/'lib/eigen_3.4.0',
              mathroot/'lib/boost_1.87.0',mathroot/'lib/tbb_2020.3/include',
              mathroot/'lib/sundials_6.1.1/include']
    command=['g++','-std=c++17','-O1','-pthread','-D_REENTRANT',
             *['-I'+str(p) for p in includes],str(source),
             '-L'+str(mathroot/'lib/tbb'),'-Wl,-rpath,'+str(mathroot/'lib/tbb'),
             '-ltbb','-o',str(binary)]
    started=time.perf_counter()
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (folder/'oracle-build.log').write_text(result.stdout+result.stderr)
    headers=[
        'stan/src/stan/mcmc/windowed_adaptation.hpp',
        'stan/src/stan/mcmc/stepsize_adaptation.hpp',
        'stan/src/stan/mcmc/var_adaptation.hpp',
        'stan/src/stan/mcmc/covar_adaptation.hpp',
        'stan/src/stan/mcmc/hmc/base_hmc.hpp',
        'stan/src/stan/mcmc/hmc/integrators/base_leapfrog.hpp',
        'stan/src/stan/mcmc/hmc/integrators/expl_leapfrog.hpp',
        'stan/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp',
        'stan/src/stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp',
        'stan/src/stan/mcmc/hmc/nuts/base_nuts.hpp',
        'stan/src/stan/services/sample/hmc_nuts_diag_e_adapt.hpp',
        'stan/src/stan/services/util/run_adaptive_sampler.hpp',
        'stan/lib/stan_math/stan/math/prim/fun/welford_var_estimator.hpp',
        'stan/lib/stan_math/stan/math/prim/fun/welford_covar_estimator.hpp']
    manifest=dict(command=command,wall_seconds=time.perf_counter()-started,
        returncode=result.returncode,stan_version='2.37.0',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        headers={p:hashlib.sha256((stan/p).read_bytes()).hexdigest() for p in headers})
    if binary.exists():manifest['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (folder/'oracle-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert result.returncode==0,result.stderr
    def call(*args,stdin=None):
        cmd=[str(binary),*map(str,args)]
        result=subprocess.run(cmd,input=stdin,capture_output=True,text=True,timeout=30,check=True)
        with (folder/'oracle-calls.jsonl').open('a') as out:
            out.write(json.dumps(dict(command=cmd,stdin=stdin,stdout=result.stdout))+'\n')
        return result.stdout.splitlines()
    return call

def test_actual_stan_header_recurrences(framework,oracle):
    tf,_=framework
    lines=oracle()
    cfg=Config(2)
    @tf.function(input_signature=[],autograph=False,jit_compile=True)
    def recurrence():
        n=tf.constant(0);h=tf.constant(0.,tf.float64);x=h;mu=tf.math.log(tf.constant(10.,tf.float64))
        out=[]
        for i in range(12):
            if i==6: n=tf.constant(0);h=tf.constant(0.,tf.float64);x=h;mu=tf.math.log(tf.constant(2.,tf.float64))
            n,h,x,e=dual_update(tf,n,h,x,mu,tf.constant(.15+.07*i,tf.float64),cfg)
            out.append(e)
        q=tf.constant([[.1*i,.01*i*i] for i in range(5,25)],tf.float64)
        center=q-tf.reduce_mean(q,axis=0)
        m2=tf.matmul(center,center,transpose_a=True)[None]
        return tf.stack(out),tf.exp(x),regularized_metric(tf,m2,20,True)
    eps,final,cov=recurrence()
    for line in lines:
        row=line.split()
        if row[0]=='schedule':
            _,ends=stan_schedule(Config(2,warmup=int(row[1])))
            assert [i for i,v in enumerate(ends) if v]==list(map(int,row[2:]))
        elif row[0]=='da': assert float(eps[int(row[1])])==pytest.approx(float(row[2]),rel=2e-12)
        elif row[0]=='final': assert float(final)==pytest.approx(float(row[1]),rel=2e-12)
        elif row[0]=='cov':
            vals=list(map(float,row[1:]));actual=tf.reshape(cov,[-1]).numpy().tolist()
            assert actual==pytest.approx(vals[:4],rel=2e-12)
            assert [actual[0],actual[3]]==pytest.approx(vals[4:],rel=2e-12)

@pytest.mark.parametrize('metric',['diag','dense'])
def test_compiled_handoff_reproducibility_and_metric_orientation(framework,oracle,metric,monkeypatch):
    tf,tfp=framework
    # Independent test instrumentation recomputes incoming/outgoing caches.
    # It does not alter a valid transition or introduce a runtime cache check.
    original=tfp.mcmc.NoUTurnSampler
    class CacheAuditedNUTS(original):
        def cache_valid(self,q,kr):
            with tf.GradientTape() as tape:
                tape.watch(q);value=self.target_log_prob_fn(q)
            grad=tape.gradient(value,q)
            return ((tf.abs(value-kr.target_log_prob)<=2e-12*(1.+tf.abs(value))) &
                tf.reduce_all(tf.abs(grad-kr.grads_target_log_prob[0])<=
                              2e-12*(1.+tf.abs(grad)),axis=-1))
        def one_step(self,q,kr,seed=None):
            ok=self.cache_valid(q,kr)
            q,kr=super().one_step(q,kr,seed=seed)
            ok &= self.cache_valid(q,kr)
            return q,kr._replace(log_accept_ratio=tf.where(
                ok,kr.log_accept_ratio,tf.constant(float('nan'),tf.float64)))
    monkeypatch.setattr(tfp.mcmc,'NoUTurnSampler',CacheAuditedNUTS)
    cfg=Config(2,chains=2,warmup=160,draws=16,metric=metric,depth=5,
               initial_buffer=10,terminal_buffer=10,window=20,
               delta=.85,gamma=.06,kappa=.65,t0=8.,initial_epsilon=.4)
    def lp(q): return -.5*tf.reduce_sum(tf.square(q/tf.constant([1.,3.],tf.float64)),axis=-1)
    warm,sample=make_functions(tf,tfp,lp,cfg)
    q=tf.constant([[.1,-1.],[-.5,2.]],tf.float64);seed=tf.constant([49,10],tf.int32)
    out,_=base.timed(warm,q,seed)
    same,_=base.timed(warm,q,seed)
    assert bool(tf.reduce_all(out[6]))
    assert bool(tf.reduce_all(out[4]==same[4]))
    assert out[4].shape==(160,2,2)
    assert float(tf.reduce_max(tf.abs(out[0]-out[4][-1])/(1.+tf.abs(out[0]))))<1e-12
    assert tf.reduce_sum(out[5][:,:,7],axis=0).numpy().tolist()==[3.,3.]
    assert not bool(tf.reduce_all(out[2][0]==out[2][1]))
    sampled,_=base.timed(sample,*out[:4])
    assert sampled[2].shape==(16,2,2)
    assert bool(tf.reduce_all(sampled[3][...,5]==1))
    assert bool(tf.reduce_all(sampled.valid))
    # Replay actual accepted positions and trajectory acceptance through native
    # Stan adapters. Searched boundary epsilon is supplied; the search itself
    # has a separate actual-header oracle below.
    for chain in range(2):
        rows=[]
        for qrow,trow in zip(out.draws[:,chain].numpy().tolist(),
                             out.telemetry[:,chain].numpy().tolist()):
            rows.append(' '.join(map(repr,[*qrow,trow[0],trow[6]])))
        native=oracle('replay',cfg.warmup,cfg.initial_buffer,cfg.terminal_buffer,
            cfg.window,metric,cfg.delta,cfg.gamma,cfg.kappa,cfg.t0,cfg.initial_epsilon,
            stdin='\n'.join(rows)+'\n')
        for i,line in enumerate(native[:-1]):
            parts=list(map(float,line.split()))
            assert parts[1]==pytest.approx(float(out.telemetry[i,chain,6]),rel=2e-12)
            assert parts[2]==float(out.telemetry[i,chain,7])
        assert float(native[-1].split()[1])==pytest.approx(float(out.step_size[chain]),rel=2e-12)
        native_cov=list(map(float,native[-2].split()[3:]))
        actual_cov=tf.matmul(out.metric_factor[chain],out.metric_factor[chain],transpose_b=True)
        assert tf.reshape(actual_cov,[-1]).numpy().tolist()==pytest.approx(native_cov,rel=2e-12,abs=2e-12)
    checked=warmup_checked(warm,q,seed)
    state=freeze(checked)
    first=sample_checked(sample,state)
    second=sample_checked(sample,FrozenState(first.position,state.step_size,state.metric_factor,first.seed))
    longer=make_functions(tf,tfp,lp,Config(**{**cfg.__dict__,'draws':32}))[1]
    combined=sample_checked(longer,state)
    assert bool(tf.reduce_all(tf.abs(tf.concat([first.draws,second.draws],axis=0)-combined.draws)<2e-12))
    for fn,args in [(warm,(q,seed)),(sample,out[:4])]:
        assert fn.experimental_get_tracing_count()==1
        assert not base.verify_graph(fn)['forbidden_ops']
        assert 'HloModule' in fn.experimental_get_compiler_ir(*args)(stage='optimized_hlo')
    a=tf.constant([[[2.,0.],[1.,3.]]],tf.float64)
    z=tf.constant([[.4,-.2]],tf.float64);pz=tf.constant([[.1,.3]],tf.float64)
    physical=tf.linalg.matvec(a,z)
    pq=tf.linalg.triangular_solve(a,pz[...,None],adjoint=True)[...,0]
    c=tf.matmul(a,a,transpose_b=True)
    assert float(tf.reduce_sum(pq*tf.linalg.matvec(c,pq)))==pytest.approx(float(tf.reduce_sum(pz*pz)))
    step=tf.constant(.03,tf.float64)
    assert tf.linalg.matvec(a,z+step*pz).numpy().tolist()[0]==pytest.approx((physical+step*tf.linalg.matvec(c,pq)).numpy().tolist()[0])

def test_short_budget_fails_explicitly():
    with pytest.raises(ValueError,match='fitting windows'): Config(2,warmup=100)


@pytest.mark.parametrize('kwargs',[
    {'dimension':True},{'chains':1.5},{'warmup':0},{'draws':-1},{'depth':31},
    {'window':0},{'terminal_buffer':0},{'initial_buffer':False},
    {'gamma':float('nan')},{'kappa':float('inf')},{'t0':True},
    {'initial_epsilon':float('inf')},{'initial_epsilon':float('nan')},
    {'initial_epsilon':0.},{'gamma':0.},{'delta':1.},{'delta':0.},
    {'metric':'unit'},{'chains':2**31},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError): Config(**{'dimension':2,**kwargs})


@pytest.mark.parametrize('n,initial,terminal,window',[
    (1,75,50,25),(19,75,50,25),(20,1,1,1),(21,1,1,2),
    (30,5,5,20),(80,5,5,1),(150,75,50,25),(151,75,50,25),
    (200,75,50,25),(250,75,50,25),(1000,75,50,25),(10000,75,50,25),
])
def test_every_schedule_flag_matches_stan(oracle,n,initial,terminal,window):
    cfg=Config(2,warmup=n,initial_buffer=initial,terminal_buffer=terminal,window=window)
    active,ends=stan_schedule(cfg)
    native=[list(map(int,row.split())) for row in oracle('schedule',n,initial,terminal,window)]
    assert [bool(row[1]) for row in native]==active
    assert [bool(row[2]) for row in native]==ends


@pytest.fixture(scope='module')
def scripted_search(framework):
    tf,_=framework
    @tf.function(input_signature=[tf.TensorSpec([1],tf.float64),
        tf.TensorSpec([16],tf.float64),tf.TensorSpec([],tf.int32)],
        autograph=False,jit_compile=True)
    def run(e,log_accepts,bound):
        def probe(e,index):
            return tf.gather(log_accepts,tf.minimum(index,15))[None],index+1
        return reasonable_step_size(tf,e,tf.constant(0),probe,max_trials=bound)
    return run


@pytest.mark.parametrize('epsilon,values',[
    (1.,[0.,0.,0.,-1.]),(1.,[-1.,-1.,-1.,0.]),
    (1.,[0.,-1.]),(1.,[-1.,0.]),(1.,[math.log(.8),math.log(.8)]),
    (1.,[float('nan'),-1.,-1.,0.]),(1.,[-float('inf'),0.]),
    (1.,[0.]),(2e7,[0.]),
])
def test_reasonable_epsilon_actual_stan_control_flow(framework,oracle,scripted_search,epsilon,values):
    tf,_=framework
    padded=values+[values[-1]]*(16-len(values))
    out=scripted_search(tf.constant([epsilon],tf.float64),tf.constant(padded,tf.float64),tf.constant(2048))
    row=oracle('search',epsilon,*values)[0].split()
    assert bool(out[3][0])==bool(int(row[0]))
    assert float(out[0][0])==float(row[1])
    assert int(out[2][0])==int(row[2])
    assert float(row[3])==0.


def test_epsilon_search_underflow_and_safety_bound(framework,oracle,scripted_search):
    tf,_=framework
    negative=tf.fill([16],tf.constant(-1.,tf.float64))
    underflow=scripted_search(tf.constant([1.],tf.float64),negative,tf.constant(2048))
    native=oracle('search',1.,-1.)[0].split()
    assert not bool(underflow[3][0]) and not bool(int(native[0]))
    assert float(underflow[0][0])==0.
    # XLA may flush subnormals; success/failure must match, not the probe count.
    bound=scripted_search(tf.constant([1.],tf.float64),negative,tf.constant(3))
    assert not bool(bound[3][0]) and int(bound[2][0])==4


@pytest.mark.parametrize('n,initial,terminal,window',[(20,1,1,1),(80,5,5,1),(1000,75,50,25)])
def test_streaming_windows_and_singleton_carry_match_stan(framework,oracle,n,initial,terminal,window):
    tf,_=framework
    cfg=Config(2,warmup=n,initial_buffer=initial,terminal_buffer=terminal,window=window)
    active,ends=stan_schedule(cfg)
    @tf.function(input_signature=[],autograph=False,jit_compile=True)
    def run():
        positions=tf.constant([[[.1*i,.01*i*i]] for i in range(n)],tf.float64)
        flags=tf.constant(active);boundaries=tf.constant(ends)
        def body(i,count,mean,m2,cov,diag,saved):
            count,mean,m2=tf.cond(flags[i],
                lambda:welford_update(tf,count,mean,m2,positions[i]),lambda:(count,mean,m2))
            def update():
                c=regularized_metric(tf,m2,count,True,cov)
                v=regularized_metric(tf,m2,count,False,diag)
                return tf.constant(0),tf.zeros_like(mean),tf.zeros_like(m2),c,v
            count,mean,m2,cov,diag=tf.cond(boundaries[i],update,lambda:(count,mean,m2,cov,diag))
            row=tf.concat((tf.reshape(cov,[-1]),tf.linalg.diag_part(diag)[0]),axis=0)
            return i+1,count,mean,m2,cov,diag,saved.write(i,row)
        eye=tf.eye(2,batch_shape=[1],dtype=tf.float64)
        out=tf.while_loop(lambda i,*_:i<n,body,(tf.constant(0),tf.constant(0),
            tf.zeros([1,2],tf.float64),tf.zeros([1,2,2],tf.float64),eye,eye,
            tf.TensorArray(tf.float64,size=n,element_shape=[6])))
        return out[-1].stack()
    actual=run()
    for row in oracle('stream',n,initial,terminal,window):
        parts=row.split();i=int(parts[0])
        assert actual[i].numpy().tolist()==pytest.approx(list(map(float,parts[1:])),rel=2e-12,abs=2e-12)


def test_dense_leapfrog_probe_matches_actual_stan_integrator(framework,oracle):
    tf,_=framework
    @tf.function(input_signature=[tf.TensorSpec([1],tf.float64)],autograph=False,jit_compile=True)
    def run(e):
        a=tf.constant([[[2.,0.],[1.,3.]]],tf.float64)
        z=tf.constant([[.4,-.2]],tf.float64);p=tf.constant([[.1,.3]],tf.float64)
        def score(z):
            with tf.GradientTape() as tape:
                tape.watch(z);q=tf.linalg.matvec(a,z)
                v=-.5*tf.reduce_sum(q*q*tf.constant([1.,.25],tf.float64),axis=-1)
            return v,tape.gradient(v,z)
        v,g=score(z)
        next_z,next_p,loga=leapfrog_probe(tf,score,z,p,e,v,g)
        return (tf.linalg.matvec(a,next_z),
                tf.linalg.triangular_solve(a,next_p[...,None],adjoint=True)[...,0],loga)
    for step in [.03,-.03,1.]:
        q,p,loga=run(tf.constant([step],tf.float64))
        assert q.numpy().tolist()[0]+p.numpy().tolist()[0]+loga.numpy().tolist()==pytest.approx(
            list(map(float,oracle('leapfrog',step)[0].split())),rel=2e-12,abs=2e-12)


@pytest.mark.parametrize('kind',['scalar','wrong_dtype','disconnected'])
def test_invalid_target_contract_fails_at_tracing(framework,kind):
    tf,tfp=framework
    def target(q):
        if kind=='scalar':return -.5*tf.reduce_sum(q*q)
        if kind=='wrong_dtype':return tf.cast(-.5*tf.reduce_sum(q*q,axis=-1),tf.float32)
        return tf.zeros([2],tf.float64)
    warm,_=make_functions(tf,tfp,target,Config(2,chains=2,warmup=1,draws=2,depth=2))
    with pytest.raises((ValueError,TypeError)):
        warm.get_concrete_function()


def test_rejected_nonfinite_proposals_are_zero_acceptance_and_valid(framework):
    tf,tfp=framework
    cfg=Config(1,chains=2,warmup=1,draws=4,depth=2)
    def target(q):
        inside=tf.reduce_all(tf.abs(q)<1.,axis=-1)
        return tf.where(inside,-.5*tf.reduce_sum(q*q,axis=-1),tf.constant(float('nan'),tf.float64))
    _,sample=make_functions(tf,tfp,target,cfg)
    state=FrozenState(tf.zeros([2,1],tf.float64),tf.constant([1e6,1e6],tf.float64),
                      tf.ones([2,1,1],tf.float64),tf.constant([49,10],tf.int32))
    result=sample_checked(sample,state)
    assert result.valid.numpy().tolist()==[True,True]
    assert bool(tf.reduce_all(result.draws==0.))
    assert bool(tf.reduce_all(result.telemetry[...,0]==0.))
    assert bool(tf.reduce_all(result.telemetry[...,4]==1.))


def test_invalid_frozen_state_and_invalid_initial_target_fail_closed(framework):
    tf,tfp=framework
    cfg=Config(2,warmup=1,draws=2,depth=2)
    def lp(q):return -.5*tf.reduce_sum(q*q,axis=-1)
    warm,sample=make_functions(tf,tfp,lp,cfg)
    q=tf.zeros([1,2],tf.float64);seed=tf.constant([23,71],tf.int32)
    eye=tf.eye(2,batch_shape=[1],dtype=tf.float64)
    for e,a in [
        ([-.1],eye),([float('nan')],eye),([.1],tf.constant([[[1.,1.],[0.,1.]]],tf.float64)),
        ([.1],tf.constant([[[-1.,0.],[0.,1.]]],tf.float64)),
        ([.1],tf.constant([[[1.,0.],[.1,1.]]],tf.float64)),
    ]:
        with pytest.raises(RuntimeError,match='Invalid'):
            sample_checked(sample,FrozenState(q,tf.constant(e,tf.float64),a,seed))
    with pytest.raises(RuntimeError,match='Invalid'):
        warmup_checked(warm,tf.constant([[float('nan'),0.]],tf.float64),seed)


def test_final_smoothed_step_size_is_checked(framework,monkeypatch):
    tf,tfp=framework
    def bad_average(tf,count,h,x,mu,acceptance,config):
        return count+1,h,tf.fill(tf.shape(x),tf.constant(float('nan'),tf.float64)),tf.ones_like(x)
    monkeypatch.setattr(adaptation,'dual_update',bad_average)
    warm,_=make_functions(tf,tfp,lambda q:-.5*tf.reduce_sum(q*q,axis=-1),
                          Config(2,warmup=1,draws=1,depth=2))
    result=warm(tf.zeros([1,2],tf.float64),tf.constant([23,71],tf.int32))
    assert not bool(result.valid[0])
    with pytest.raises(RuntimeError,match='Invalid'):freeze(result)


def test_adaptation_primitives_isolate_batch_rows(framework):
    tf,_=framework
    @tf.function(input_signature=[tf.TensorSpec([2,2],tf.float64)],autograph=False,jit_compile=True)
    def run(q):
        count,mean,m2=welford_update(tf,tf.constant(0),tf.zeros_like(q),
                                     tf.zeros([2,2,2],tf.float64),q)
        _,mean,m2=welford_update(tf,count,mean,m2,q+tf.constant([[1.,2.],[3.,4.]],tf.float64))
        _,h,x,e=dual_update(tf,tf.constant(0),tf.zeros([2],tf.float64),
            tf.zeros([2],tf.float64),tf.ones([2],tf.float64),q[:,0],Config(2))
        return mean,m2,h,x,e
    a=run(tf.constant([[.2,.3],[.4,.5]],tf.float64))
    b=run(tf.constant([[.2,.3],[.8,-50.]],tf.float64))
    assert all(bool(tf.reduce_all(x[0]==y[0])) for x,y in zip(a,b))
