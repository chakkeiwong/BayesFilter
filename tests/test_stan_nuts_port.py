"""Diagnostic native-source fidelity, lifecycle and enclosing-XLA tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
import stan_nuts as port
import benchmark_nuts_xla as base
from tests.test_stan_nuts_warmup import oracle as adaptation_oracle

@pytest.fixture(scope='module')
def tf():
    args=base.parser().parse_args(['--device','cpu','--output','/tmp/stan-port-test'])
    return base.setup(args)[0]

@pytest.fixture(scope='module')
def native(tmp_path_factory):
    stan=ROOT/'.localresources/cmdstan/cmdstan-2.37.0'
    assert stan.is_dir(),'Pinned CmdStan required; native fidelity tests may not silently skip'
    folder=Path(os.environ.get('BAYESFILTER_STAN_AUDIT_OUTPUT',str(tmp_path_factory.mktemp('port-oracle'))))
    folder.mkdir(parents=True,exist_ok=True)
    mathroot=stan/'stan/lib/stan_math'
    source=ROOT/'docs/benchmarks/stan_transition_oracle.cpp'
    binary=folder/'transition-oracle'
    includes=[stan/'stan/src',mathroot,mathroot/'lib/eigen_3.4.0',mathroot/'lib/boost_1.87.0',
              mathroot/'lib/tbb_2020.3/include',mathroot/'lib/sundials_6.1.1/include']
    cmd=['g++','-std=c++17','-O1','-pthread','-D_REENTRANT',*['-I'+str(x) for x in includes],
         str(source),'-L'+str(mathroot/'lib/tbb'),'-Wl,-rpath,'+str(mathroot/'lib/tbb'),'-ltbb','-o',str(binary)]
    began=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    (folder/'transition-build.log').write_text(r.stdout+r.stderr)
    sources=[source,stan/'stan/src/stan/mcmc/hmc/nuts/base_nuts.hpp',
             stan/'stan/src/stan/mcmc/hmc/base_hmc.hpp',
             stan/'stan/src/stan/mcmc/hmc/integrators/expl_leapfrog.hpp']
    (folder/'transition-build.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,
        wall_seconds=time.perf_counter()-began,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    assert r.returncode==0,r.stderr
    def call(kind,q,p,e,metric,tape,jitter=0.,max_delta=1000.,depth=6):
        values=[len(q),kind,depth,e,jitter,max_delta,*q,*p,*[x for row in metric for x in row],len(tape),*tape]
        stdin=' '.join(map(str,values))+'\n'
        r=subprocess.run([str(binary)],input=stdin,capture_output=True,text=True,check=True,timeout=10)
        with (folder/'transition-calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(stdin=stdin,stdout=r.stdout))+'\n')
        return [[float(x) for x in row.split()] for row in r.stdout.splitlines()]
    return call

def target(tf,kind):
    if kind==0:return lambda q: -.5*tf.reduce_sum(q*q,-1)
    if kind==1:return lambda q: -.5*(q[:,0]**2+(q[:,1]-(q[:,0]**2-1.))**2)
    if kind==2:return lambda q: -.5*q[:,0]**2/9.-.5*q[:,0]-.5*q[:,1]**2*tf.exp(-q[:,0])
    if kind==3:return lambda q: tf.where(tf.abs(q[:,0])<1.,-.5*tf.reduce_sum(q*q,-1),tf.constant(-float('inf'),tf.float64))
    return lambda q: 0.*tf.reduce_sum(q,-1)

@pytest.mark.parametrize('kind',[0,1,2,3,4])
@pytest.mark.parametrize('metric_kind',['unit','diag','dense'])
def test_actual_native_transition(tf,native,kind,metric_kind):
    cfg=port.Config(2,metric=metric_kind,depth=6,jitter=.17)
    mshape=[1,2,2] if metric_kind=='dense' else [1,2]
    @tf.function(input_signature=[tf.TensorSpec([1,2],tf.float64),tf.TensorSpec([1,2],tf.float64),
        tf.TensorSpec([1],tf.float64),tf.TensorSpec(mshape,tf.float64),tf.TensorSpec([1,256],tf.float64)],
        autograph=False,jit_compile=True)
    def transition(q,p,e,metric,tape):
        k=port.Kernel(tf,target(tf,kind),cfg,uniform_tape=tape,momentum=p)
        return k.transition(q,e,metric,k.factor(metric),tf.zeros([1,2],tf.int32))
    rnd=random.Random(20261027+kind)
    for i in range(12):
        q=[rnd.uniform(-.8,.8) for _ in range(2)]
        p=[rnd.uniform(-1.5,1.5) for _ in range(2)]
        e=[.01,.1,.5,1.2,3.,10.][i%6]
        if kind==4 and i==0:p=[0.,0.]  # strict >0 U-turn tie
        metric={'unit':[[1.,0.],[0.,1.]],'diag':[[.4,0.],[0.,2.]],'dense':[[1.3,.4],[.4,.8]]}[metric_kind]
        tape=[rnd.random() for _ in range(256)]
        if i==1:tape[1]=.5  # exact direction boundary after jitter draw
        expected=native(kind,q,p,e,metric,tape,jitter=cfg.jitter)
        m=metric if metric_kind=='dense' else [metric[0][0],metric[1][1]]
        actual=transition(*[tf.constant(x,tf.float64) for x in ([q],[p],[e],[m],[tape])])
        top=expected[0]
        assert [int(actual.leaps[0]),int(actual.depth[0]),int(actual.divergent[0]),int(actual.seed[0,1])]==[int(top[j]) for j in (1,2,3,5)],(kind,metric_kind,i,top,actual)
        assert float(actual.acceptance[0])==pytest.approx(top[0],rel=2e-12,abs=2e-12)
        assert float(actual.energy[0])==pytest.approx(top[4],rel=2e-12,abs=2e-12)
        assert float(actual.epsilon[0])==pytest.approx(top[6],rel=2e-12,abs=2e-12)
        for value,row in zip((actual.phase.q,actual.phase.p,actual.phase.g),expected[1:4]):
            assert value.numpy().tolist()[0]==pytest.approx(row,rel=2e-12,abs=2e-12)
        assert float(actual.phase.v[0])==pytest.approx(expected[4][0],rel=2e-12,abs=2e-12)
        assert bool(actual.valid[0])
    assert transition.experimental_get_tracing_count()==1
    base.verify_graph(transition)

@pytest.mark.parametrize('metric',['unit','diag','dense'])
def test_lifecycle_resume_and_cache(tf,metric):
    cfg=port.Config(2,chains=2,warmup=160,draws=16,metric=metric,depth=5)
    initialize,warm,sample=port.make_functions(tf,target(tf,0),cfg)
    q=tf.constant([[.3,-.7],[1.,-.1]],tf.float64)
    seeds=tf.constant([[171,0],[172,0]],tf.int32)
    m=tf.eye(2,batch_shape=[2],dtype=tf.float64) if metric=='dense' else tf.ones([2,2],tf.float64)
    warmed=warm(q,tf.ones([2],tf.float64),m,seeds)
    assert bool(tf.reduce_all(warmed.valid))
    frozen=port.freeze(warmed)
    a=sample(*frozen)
    assert bool(tf.reduce_all(a.valid))
    b=sample(a.position,frozen.step_size,frozen.inverse_metric,a.seed)
    _,_,long=port.make_functions(tf,target(tf,0),port.Config(**{**cfg.__dict__,'draws':32}))
    both=long(*frozen)
    assert bool(tf.reduce_all(tf.concat([a.draws,b.draws],0)==both.draws))
    assert bool(tf.reduce_all(b.seed==both.seed))
    # Stored selected energy equals fresh V+K is native-tested; retained q's
    # density is independently evaluated here and must be finite after warmup.
    assert bool(tf.reduce_all(tf.math.is_finite(target(tf,0)(a.position))))
    for fn,args,name in [(warm,(q,tf.ones([2],tf.float64),m,seeds),'warm'),(sample,frozen,'sample')]:
        hlo=fn.experimental_get_compiler_ir(*args)(stage='optimized_hlo')
        folder=os.environ.get('BAYESFILTER_STAN_PORT_OUTPUT')
        if folder:Path(folder,f'{metric}-{name}.hlo.txt').write_text(hlo)
        base.verify_graph(fn)
        assert fn.experimental_get_tracing_count()==1

def test_batch_isolation(tf):
    cfg=port.Config(2,chains=2,warmup=0,draws=20,depth=6,adapt=False)
    _,_,batch=port.make_functions(tf,target(tf,1),cfg)
    _,_,scalar=port.make_functions(tf,target(tf,1),port.Config(**{**cfg.__dict__,'chains':1}))
    q=tf.constant([[.2,.1],[5.,-10.]],tf.float64)
    e=tf.constant([.15,.8],tf.float64);m=tf.ones([2,2],tf.float64);r=tf.constant([[51,0],[73,0]],tf.int32)
    b=batch(q,e,m,r)
    for i in range(2):
        s=scalar(q[i:i+1],e[i:i+1],m[i:i+1],r[i:i+1])
        assert bool(tf.reduce_all(s.draws[:,0]==b.draws[:,i]))
        assert bool(tf.reduce_all(s.seed[0]==b.seed[i]))

def test_overlapping_random_streams_rejected(tf):
    cfg=port.Config(2,chains=2,warmup=0,draws=2,adapt=False,depth=3)
    init,warm,sample=port.make_functions(tf,target(tf,0),cfg)
    q=tf.zeros([2,2],tf.float64);e=tf.ones([2],tf.float64);m=tf.ones_like(q)
    overlap=tf.constant([[51,0],[51,17]],tf.int32)
    assert not bool(tf.reduce_any(init(q,tf.ones_like(q,tf.bool),overlap)[2]))
    with pytest.raises(RuntimeError):port.require_valid(warm(q,e,m,overlap))
    with pytest.raises(RuntimeError):port.require_valid(sample(q,e,m,overlap))
    distinct=tf.constant([[51,0],[52,0]],tf.int32)
    port.require_valid(sample(q,e,m,distinct))

@pytest.mark.parametrize('warmup,adapt,thin,dimension',[(0,False,1,2),(0,True,1,2),(12,True,3,2),(20,True,1,2),(4,False,2,0)])
def test_service_edge_cases(tf,warmup,adapt,thin,dimension):
    cfg=port.Config(dimension,warmup=warmup,draws=7,adapt=adapt,thin=thin,depth=3)
    init,warm,sample=port.make_functions(tf,target(tf,0),cfg)
    q=tf.zeros([1,dimension],tf.float64);e=tf.constant([.2],tf.float64);m=tf.ones_like(q);r=tf.constant([[18,0]],tf.int32)
    w=warm(q,e,m,r)
    assert bool(tf.reduce_all(w.valid))
    assert w.draws.shape[0]==(warmup+thin-1)//thin
    s=sample(*port.freeze(w))
    assert s.draws.shape[0]==(7+thin-1)//thin
    assert bool(tf.reduce_all(s.valid))
    if not adapt:assert w.step_size.numpy().tolist()==e.numpy().tolist()
    if thin>1 and dimension:
        _,_,unthin=port.make_functions(tf,target(tf,0),port.Config(**{**cfg.__dict__,'thin':1}))
        un=unthin(*port.freeze(w))
        assert bool(tf.reduce_all(s.draws==un.draws[::thin]))

def test_initialization_retry_and_failure(tf):
    cfg=port.Config(2,chains=2,warmup=0,draws=1,adapt=False,init_radius=2.)
    init,_,_=port.make_functions(tf,target(tf,3),cfg)
    q=tf.constant([[2.,.2],[.2,.4]],tf.float64);seeds=tf.constant([[19,0],[31,0]],tf.int32)
    _,_,good,counts=init(q,tf.ones([2,2],tf.bool),seeds)
    assert good.numpy().tolist()==[False,True]
    assert counts.numpy().tolist()==[1,1]
    initialized,_,good,counts=init(q,tf.constant([[False,True],[False,False]]),seeds)
    assert bool(tf.reduce_all(good))
    assert float(initialized[0,1])==.2
    assert bool(tf.reduce_all(tf.abs(initialized[:,0])<1.))
    assert bool(tf.reduce_all(counts<=100))
    impossible=lambda q: 0.*tf.reduce_sum(q,-1)+tf.constant(-float('inf'),tf.float64)
    fail,_,_=port.make_functions(tf,impossible,cfg)
    _,_,good,counts=fail(q,tf.zeros([2,2],tf.bool),seeds)
    assert not bool(tf.reduce_any(good))
    assert counts.numpy().tolist()==[100,100]

@pytest.mark.parametrize('metric,bad',[('diag',[[-1.,1.]]),('unit',[[2.,1.]]),('dense',[[[1.,2.],[2.,1.]]]),('dense',[[[1.,.1],[0.,1.]]])])
def test_bad_metric_fails_closed(tf,metric,bad):
    cfg=port.Config(2,metric=metric,warmup=0,draws=1,adapt=False,depth=2)
    _,_,sample=port.make_functions(tf,target(tf,0),cfg)
    result=sample(tf.zeros([1,2],tf.float64),tf.constant([.1],tf.float64),tf.constant(bad,tf.float64),tf.constant([[12,0]],tf.int32))
    with pytest.raises(RuntimeError):port.require_valid(result)

@pytest.mark.parametrize('kwargs',[{'thin':0},{'jitter':1.},{'depth':31},{'metric':'bad'},{'dimension':-1},{'initial_epsilon':0.}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):port.Config(**{'dimension':2,**kwargs})

@pytest.mark.parametrize('n,buffers',[(0,(75,50,25)),(19,(75,50,25)),(20,(75,50,25)),
    (100,(75,50,25)),(1000,(75,50,25)),(20,(0,0,1)),(20,(0,0,0)),(20,(2,1,0)),(20,(2,1,1))])
def test_actual_window_schedule(adaptation_oracle,n,buffers):
    cfg=port.Config(2,warmup=n,initial_buffer=buffers[0],terminal_buffer=buffers[1],window=buffers[2])
    rows=adaptation_oracle('schedule',n,*buffers)
    active,ends=port.schedule(cfg)
    for i,row in enumerate(rows):
        _,a,b=map(int,row.split())
        assert active[i]==bool(a)
        assert ends[i]==bool(a and b)

@pytest.mark.parametrize('metric',['diag','dense'])
def test_native_adaptation_replay_on_actual_port_trace(tf,adaptation_oracle,metric):
    cfg=port.Config(2,metric=metric,warmup=30,draws=0,depth=4,initial_buffer=2,terminal_buffer=1,window=1)
    _,warm,_=port.make_functions(tf,target(tf,1),cfg)
    m=tf.constant([[[2.,.3],[.3,1.]]],tf.float64) if metric=='dense' else tf.constant([[2.,1.]],tf.float64)
    # Oracle replay starts with identity; the first singleton covariance boundary
    # therefore needs identity here. Initial non-unit metrics are covered by
    # native transitions and the runtime warmup tests.
    m=tf.eye(2,batch_shape=[1],dtype=tf.float64) if metric=='dense' else tf.ones([1,2],tf.float64)
    w=warm(tf.constant([[.2,-.1]],tf.float64),tf.constant([1.],tf.float64),m,tf.constant([[324,0]],tf.int32))
    assert bool(w.valid[0])
    q=w.draws.numpy().tolist();trace=w.telemetry.numpy().tolist()
    stdin='\n'.join(' '.join(map(str,[*q[i][0],trace[i][0][0],trace[i][0][9]])) for i in range(cfg.warmup))+'\n'
    rows=adaptation_oracle('replay',cfg.warmup,2,1,1,metric,cfg.delta,cfg.gamma,cfg.kappa,cfg.t0,1.,stdin=stdin)
    for i,row in enumerate(rows[:-1]):
        vals=list(map(float,row.split()))
        assert trace[i][0][9]==pytest.approx(vals[1],rel=2e-12,abs=2e-12)
        assert int(trace[i][0][10])==int(vals[2])
    expected=list(map(float,rows[-2].split()))[3:]
    actual=w.inverse_metric.numpy().tolist()[0]
    actual=[x for row in actual for x in row] if metric=='dense' else [actual[0],0.,0.,actual[1]]
    assert actual==pytest.approx(expected,rel=2e-12,abs=2e-12)
    assert float(w.step_size[0])==pytest.approx(float(rows[-1].split()[1]),rel=2e-12,abs=2e-12)

@pytest.mark.parametrize('threshold',[2.,1.9999999999999998])
def test_strict_energy_threshold(tf,native,threshold):
    cfg=port.Config(2,metric='unit',depth=2,max_delta=threshold)
    tape=tf.constant([[.9]*256],tf.float64);p=tf.constant([[1.,0.]],tf.float64)
    @tf.function(input_signature=[],autograph=False,jit_compile=True)
    def step():
        k=port.Kernel(tf,target(tf,0),cfg,uniform_tape=tape,momentum=p)
        m=tf.ones([1,2],tf.float64)
        return k.transition(tf.zeros([1,2],tf.float64),tf.constant([2.],tf.float64),m,m,tf.zeros([1,2],tf.int32))
    got=step();ref=native(0,[0.,0.],[1.,0.],2.,[[1.,0.],[0.,1.]],[.9]*256,max_delta=threshold,depth=2)
    assert bool(got.divergent[0])==bool(ref[0][3])==(threshold<2.)

def test_graph_size_does_not_unroll_exponential_tree(tf):
    counts=[]
    for depth in (3,10,20):
        cfg=port.Config(2,depth=depth,draws=1,warmup=0,adapt=False)
        _,_,sample=port.make_functions(tf,target(tf,0),cfg)
        graph=sample.get_concrete_function().graph.as_graph_def()
        counts.append(len(graph.node)+sum(len(f.node_def) for f in graph.library.function))
    assert counts[0]==counts[1]==counts[2]
    folder=os.environ.get('BAYESFILTER_STAN_PORT_OUTPUT')
    if folder:Path(folder,'graph-growth.json').write_text(json.dumps(dict(depths=[3,10,20],nodes=counts)))

@pytest.mark.parametrize('epsilon,log_accept',[(1.,[-.1,-.1,-.3]),(1.,[-2.,-2.,-.1]),
    (1.,[-.2231435513142097,-.2231435513142097]),(2e7,[-2.]),(8e6,[0.,0.]),(1.,[float('nan'),-.1])])
def test_direct_step_search_against_native(tf,adaptation_oracle,epsilon,log_accept):
    cfg=port.Config(2)
    class Probe(port.Kernel):
        def momentum(self,seed,mask,factor):
            # Momentum value carries the probe index for the scripted integrator.
            return tf.repeat(tf.cast(seed[:,1,None],tf.float64),2,axis=1),seed+tf.stack((tf.zeros([1],tf.int32),tf.cast(mask,tf.int32)),-1)
        def energy(self,z,metric):return z.v
        def leapfrog(self,z,e,metric):
            index=tf.minimum(tf.cast(z.p[:,0],tf.int32),len(log_accept)-1)
            return z._replace(v=-tf.gather(tf.constant(log_accept,tf.float64),index))
    @tf.function(input_signature=[],autograph=False,jit_compile=True)
    def search():
        k=Probe(tf,lambda q: 0.*tf.reduce_sum(q,-1),cfg)
        return k.find_epsilon(tf.zeros([1,2],tf.float64),tf.constant([epsilon],tf.float64),
            tf.ones([1,2],tf.float64),tf.ones([1,2],tf.float64),tf.constant([[11,0]],tf.int32))
    e,r,n,ok=search()
    native=list(map(float,adaptation_oracle('search',epsilon,*log_accept)[0].split()))
    assert bool(ok[0])==bool(native[0])
    assert float(e[0])==native[1]
    assert int(n[0])==int(native[2])
    assert int(r[0,1])==int(native[2])


@pytest.fixture(scope='module')
def native_initializer(tmp_path_factory):
    stan=ROOT/'.localresources/cmdstan/cmdstan-2.37.0'
    mathroot=stan/'stan/lib/stan_math'
    folder=Path(os.environ.get('BAYESFILTER_STAN_AUDIT_OUTPUT',str(tmp_path_factory.mktemp('service-oracle'))))
    folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'docs/benchmarks/stan_service_oracle.cpp'
    binary=folder/'service-oracle'
    includes=[stan/'stan/src',mathroot,mathroot/'lib/eigen_3.4.0',mathroot/'lib/boost_1.87.0',
              mathroot/'lib/tbb_2020.3/include',mathroot/'lib/sundials_6.1.1/include']
    cmd=['g++','-std=c++17','-O1','-pthread','-D_REENTRANT',*['-I'+str(x) for x in includes],
         str(source),'-L'+str(mathroot/'lib/tbb'),'-Wl,-rpath,'+str(mathroot/'lib/tbb'),'-ltbb','-o',str(binary)]
    began=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    (folder/'service-build.log').write_text(r.stdout+r.stderr)
    sources=[source,stan/'stan/src/stan/services/util/initialize.hpp',
             stan/'stan/src/stan/io/random_var_context.hpp',stan/'stan/src/stan/model/log_prob_grad.hpp']
    (folder/'service-build.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,
        wall_seconds=time.perf_counter()-began,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    assert r.returncode==0,r.stderr
    def call(*args):
        cmd=[str(binary),*map(str,args)]
        r=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=10)
        with (folder/'service-calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(command=cmd,stdout=r.stdout))+'\n')
        return list(map(float,r.stdout.split()))
    return call


@pytest.mark.parametrize('kind,mask,radius,q',[
    (0,3,2.,[.25,-.5]),(0,1,2.,[.25,-.5]),(1,0,2.,[0.,0.]),
    (1,3,2.,[2.,0.]),(2,0,2.,[0.,0.]),(2,0,0.,[0.,0.]),
    (3,3,0.,[0.,0.]),(0,0,0.,[.25,-.5]),
])
def test_initialization_against_native_service(tf,native_initializer,monkeypatch,kind,mask,radius,q):
    class TapeKernel(port.Kernel):
        def __init__(self,tf,log_prob,config):
            tape=tf.constant([[[.875,.5]]+[[.375,.75]]*100],tf.float64)
            super().__init__(tf,log_prob,config,uniform_tape=tape)
    monkeypatch.setattr(port,'Kernel',TapeKernel)
    def lp(x):
        if kind==2:return 0.*tf.reduce_sum(x,-1)+tf.constant(-float('inf'),tf.float64)
        if kind==3:return tf.constant(1e308,tf.float64)*tf.reduce_sum(x,-1)
        result=-.5*tf.reduce_sum(x*x,-1)
        return tf.where(tf.abs(x[:,0])<1.,result,tf.constant(-float('inf'),tf.float64)) if kind==1 else result
    cfg=port.Config(2,init_radius=radius,warmup=0,draws=0,adapt=False)
    init,_,_=port.make_functions(tf,lp,cfg)
    value,seed,good,attempts=init(tf.constant([q],tf.float64),
        tf.constant([[bool(mask&1),bool(mask&2)]]),tf.constant([[99,0]],tf.int32))
    native=native_initializer(kind,mask,radius,*q)
    assert bool(good[0])==bool(native[0])
    assert int(attempts[0])==int(native[1])
    assert 2*int(seed[0,1])==int(native[2])
    if bool(good[0]):assert value.numpy().tolist()[0]==native[3:]
