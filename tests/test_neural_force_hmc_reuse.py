"""CPU diagnostic parity of force reuse against uncached transitions."""
import pytest
import tensorflow as tf
from bayesfilter.inference import neural_force_hmc as nf

D = tf.float64
START = tf.constant([[.1, -.2], [1.5, -1.2], [-1., .2], [.4, .6]], D)
SEED = tf.constant([91731, 77], tf.int32)
STEPS = 12

def bindings(kind, counter=None):
    def field(q):
        if counter is not None:
            counter.assign_add(1)
        f = tf.stack([.75*q[:, 0]+.2*q[:, 1], 1.5*q[:, 1]-.3*q[:, 0]], axis=1)
        if kind == "fallback":
            f = tf.where(q[:, :1] > .5, tf.constant(float("nan"), D), f)
        if kind == "reported_invalid":
            return nf.PositionOnlyForceEvaluation(f, q[:, 0] <= .5)
        if kind == "overflow":
            f = tf.where(q[:, :1] > .5, tf.constant(1.e308, D), f)
        return f
    def endpoint(q):
        u = .5*tf.reduce_sum(tf.square(q)*tf.constant([1.,4.], D), axis=1)
        if kind in ("support", "bad_endpoint"):
            u = tf.where(tf.reduce_sum(tf.square(q),axis=1)>5.,
                         tf.constant(float("inf") if kind=="support" else float("nan"),D),u)
        return u
    return nf.FrozenPositionOnlyForce(field,"cache-test-"+kind), nf.FrozenTargetPotential(endpoint,"target-"+kind)

def uncached(q,u,force,target,cfg):
    traces, states, potentials = [], [], []
    for i in range(STEPS):
        tr=nf.neural_force_hmc_transition(q,u,force,target,cfg,
                    tf.random.experimental.stateless_fold_in(SEED,i))
        q,u=tr.position,tr.potential
        traces.append(tr.trace);states.append(q);potentials.append(u)
    return tf.stack(states),tf.stack(potentials),traces

def check(chain,states,potentials,traces,atol=0.):
    def equal(a,b,key):
        if a.dtype.is_floating and atol:
            tf.debugging.assert_near(a,b,atol=atol,rtol=atol,message=key)
        else:
            tf.debugging.assert_equal(a,b,message=key)
    equal(chain.positions,states,"positions")
    equal(chain.potentials,potentials,"potentials")
    for key in ("accepted","log_acceptance_ratio","initial_potential","final_potential",
                "initial_kinetic","final_kinetic","delta_h","finite_status","endpoint_call_count",
                "divergence","force_fallback"):
        equal(getattr(chain,key),tf.stack([getattr(t,key) for t in traces]),key)
    for key,values in chain.endpoint_status.items():
        equal(values,tf.stack([t.endpoint_status[key] for t in traces]),key)

@pytest.mark.parametrize("kind",["healthy","fallback","reported_invalid","support","overflow"])
@pytest.mark.parametrize("leapfrog",[1,3])
def test_accept_reject_fallback_and_divergence_parity(kind,leapfrog):
    force,target=bindings(kind)
    cfg=nf.NeuralForceHMCConfig(4. if kind=="overflow" else .9,leapfrog,(1.,1.),dtype="float64")
    u=target.function(START)
    expected=uncached(START,u,force,target,cfg)
    chain=nf.sample_neural_force_hmc(START,u,force,target,cfg,num_warmup=3,num_results=STEPS-3,seed=SEED)
    check(chain,*expected)
    assert int(tf.reduce_sum(chain.force_call_count))==1+STEPS*leapfrog
    if kind=="healthy":
        assert bool(tf.reduce_any(chain.accepted)) and bool(tf.reduce_any(~chain.accepted))
    if kind in ("fallback","reported_invalid"):
        assert bool(tf.reduce_any(chain.force_fallback))
    if kind in ("bad_endpoint","overflow"):
        assert bool(tf.reduce_any(chain.divergence))
    if kind=="support":
        assert bool(tf.reduce_any(chain.endpoint_status["outside_declared_support"]))

def test_physical_batched_force_evaluation_accounting():
    calls=tf.Variable(0,dtype=tf.int32)
    force,target=bindings("healthy",calls)
    cfg=nf.NeuralForceHMCConfig(.3,3,(1.,1.),dtype="float64")
    u=target.function(START)
    chain=nf.sample_neural_force_hmc(START,u,force,target,cfg,num_warmup=3,num_results=STEPS-3,seed=SEED)
    assert int(calls)==1+STEPS*3==int(tf.reduce_sum(chain.force_call_count))
    calls.assign(0);uncached(START,u,force,target,cfg)
    assert int(calls)==STEPS*4

@pytest.mark.parametrize("kind",["healthy","fallback","support"])
def test_public_tfp_consumer_preserves_cached_rejected_rows(kind):
    calls=tf.Variable(0,dtype=tf.int32)
    force,target=bindings(kind,calls)
    cfg=nf.NeuralForceHMCConfig(.9,3,(1.,1.),dtype="float64")
    kernel=nf.NeuralForceTransitionKernel(force=force,target=target,config=cfg)
    kr=kernel.bootstrap_results(START)
    assert int(calls)==int(kr.force_call_count)==1
    q=START
    reference_force,_=bindings(kind)
    u=target.function(q)
    count=1
    for i in range(STEPS):
        seed=tf.random.experimental.stateless_fold_in(SEED,i)
        old=nf.neural_force_hmc_transition(q,u,reference_force,target,cfg,seed)
        q,kr=kernel.one_step(q,kr,seed=seed)
        u=old.potential
        for actual,expected in ((q,old.position),(kr.is_accepted,old.trace.accepted),
                                (kr.delta_h,old.trace.delta_h),(kr.force_fallback,old.trace.force_fallback)):
            tf.debugging.assert_equal(actual,expected)
        expected_cache=nf._evaluate_force(reference_force,q,dimension=2,dtype=D)
        tf.debugging.assert_equal(kr.retained_force[0],expected_cache[0])
        tf.debugging.assert_equal(kr.retained_force[1],expected_cache[1])
        count+=int(kr.force_call_count)
    assert int(calls)==count==1+STEPS*3

def test_changed_field_and_coordinate_bootstrap_is_fresh():
    cfg=nf.NeuralForceHMCConfig(.3,1,(1.,1.),dtype="float64")
    _,target=bindings("healthy")
    def kernel(field,identity,coordinate):
        return nf.NeuralForceTransitionKernel(
            force=nf.FrozenPositionOnlyForce(field,identity,coordinate_system=coordinate),
            target=target,config=cfg)
    old=kernel(lambda q:q,"raw","raw").bootstrap_results(START)
    new=kernel(lambda q:2*q,"affine","transformed").bootstrap_results(START+1.)
    tf.debugging.assert_equal(old.retained_force[0],START)
    tf.debugging.assert_equal(new.retained_force[0],2*(START+1.))

@pytest.mark.parametrize("jit",[False,True])
def test_stable_graph_and_cpu_xla_parity(jit):
    force,target=bindings("healthy")
    cfg=nf.NeuralForceHMCConfig(.3,3,(1.,1.),dtype="float64")
    sig=[tf.TensorSpec([4,2],D),tf.TensorSpec([4],D)]
    @tf.function(input_signature=sig,jit_compile=jit,autograph=False)
    def run(q,u):
        return nf.sample_neural_force_hmc(q,u,force,target,cfg,num_warmup=3,num_results=STEPS-3,seed=SEED)
    @tf.function(input_signature=sig,jit_compile=jit,autograph=False)
    def reference(q,u):
        return uncached(q,u,force,target,cfg)
    u=target.function(START)
    check(run(START,u),*reference(START,u),atol=2.e-12 if jit else 0.)


def test_undeclared_endpoint_nan_fails_closed_with_and_without_reuse():
    force,target=bindings("bad_endpoint")
    cfg=nf.NeuralForceHMCConfig(.9,3,(1.,1.),dtype="float64")
    u=target.function(START)
    with pytest.raises(tf.errors.InvalidArgumentError, match="target potential must be finite"):
        uncached(START,u,force,target,cfg)
    with pytest.raises(tf.errors.InvalidArgumentError, match="target potential must be finite"):
        nf.sample_neural_force_hmc(START,u,force,target,cfg,num_warmup=0,num_results=STEPS,seed=SEED)


def test_campaign_paired_program_uses_same_seed_stream():
    from bayesfilter.testing.surrogate_hmc_force_reuse_diagnostic import paired_programs, compare
    force, target = bindings("healthy")
    cfg = nf.NeuralForceHMCConfig(.3, 1, (1., 1.), dtype="float64")
    programs = paired_programs(force, target, cfg)
    values = {name: program(START, target.function(START), SEED)
              for name, program in programs.items()}
    result = compare(values["cached"], values["uncached"])
    assert result["pass_parity"], result
    assert int(tf.reduce_sum(values["cached"]["force_call_count"])) == 3
    assert int(tf.reduce_sum(values["uncached"]["force_call_count"])) == 4


def test_bootstrap_retains_and_reports_force_fallback():
    force, target = bindings("reported_invalid")
    cfg = nf.NeuralForceHMCConfig(.3, 1, (1., 1.), dtype="float64")
    kr = nf.NeuralForceTransitionKernel(force=force, target=target, config=cfg).bootstrap_results(START)
    tf.debugging.assert_equal(kr.force_fallback, START[:, 0] > .5)
    tf.debugging.assert_equal(kr.retained_force[1], kr.force_fallback)
