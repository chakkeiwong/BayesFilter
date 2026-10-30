"""Declared LG fixtures for the 2026-09-17 graceful-failure program.

This module binds model parameters, immutable random inputs and priors to the
shared canonical engine. It implements no alternative filtering algorithm and
does not issue scientific admission from a successful evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
import ast,hashlib,json
import tensorflow as tf
from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import linear_gaussian_parameter_model
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode, FailureStage
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky

MODEL_CODES={'LG1':1,'LG2':2,'LG4':3,'LG20':4}
PARTITION_CODES={'calibration':1,'validation':2,'claim':3,'diagnostic':4}

def fixture_seed(model_id,horizon,partition,attempt,replicate,purpose):
    seed=tf.constant([20260917,917],tf.int32)
    for code in (MODEL_CODES[model_id],horizon,PARTITION_CODES[partition],attempt,replicate,purpose):
        seed=tf.random.experimental.stateless_fold_in(seed,int(code))
    return seed

def linear_fixture_spec(model_id,dtype=tf.float32):
    if model_id=='LG1':
        return dict(d=1,t=10,base_f=tf.ones([1,1],dtype),base_q=tf.ones([1,1],dtype),
            h=tf.ones([1,1],dtype),r=.5*tf.ones([1,1],dtype),truth=(.8,),
            lower=(-.95,),upper=(.95,),infer_process_scale=False)
    if model_id=='LG2':
        d,t,m=2,20,2
        a=tf.constant([[.7,.2],[.1,.6]],dtype)
        q=tf.constant([[1.,.3],[.3,.8]],dtype)
    elif model_id in ('LG4','LG20'):
        d,t,m=(4,50,2) if model_id=='LG4' else (20,120,10)
        a=.6*tf.eye(d,dtype=dtype)+.1*(tf.linalg.diag(tf.ones([d-1],dtype),k=1)+tf.linalg.diag(tf.ones([d-1],dtype),k=-1))
        q=tf.eye(d,dtype=dtype) if d==4 else tf.linalg.diag(tf.linspace(tf.cast(.5,dtype),tf.cast(2.,dtype),d))
    else:raise ValueError('not a declared linear fixture: '+model_id)
    return dict(d=d,t=t,base_f=a,base_q=q,h=tf.eye(d,dtype=dtype)[:m],
        r=.5*tf.eye(m,dtype=dtype),truth=(1.,0.),lower=(.5,-1.),upper=(1.2,1.),
        infer_process_scale=True)

def source_closure():
    """Ordinary source provenance, following local imports from the actual route."""
    root=Path(__file__).resolve().parents[2]
    todo=['bayesfilter.highdim.ledh_program_fixtures_tf'];done={}
    while todo:
        name=todo.pop()
        path=root.joinpath(*name.split('.')).with_suffix('.py')
        if not path.is_file() or str(path.relative_to(root)) in done:continue
        rel=str(path.relative_to(root));raw=path.read_bytes()
        done[rel]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('bayesfilter.'):
                todo.append(node.module)
            elif isinstance(node,ast.Import):
                todo.extend(alias.name for alias in node.names if alias.name.startswith('bayesfilter.'))
    return dict(sorted(done.items()))

def _hash_tensor(value):
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()

@dataclass(frozen=True)
class BoundedLEDHFixtureTarget:
    model_id:str
    model:object
    initial_base:tf.Tensor
    initial_covariances:tf.Tensor
    noises:tf.Tensor
    observations:tf.Tensor
    lower:tf.Tensor
    upper:tf.Tensor
    controls:object
    identity:str
    manifest:object

    @property
    def parameter_count(self):return int(self.lower.shape[0])

    def physical(self,theta,*,filter_moments=False):
        theta=tf.convert_to_tensor(theta,self.initial_base.dtype)
        p=self.parameter_count
        if theta.shape.rank!=2 or theta.shape[1]!=p:raise ValueError('theta must be [batch,parameters]')
        finite=tf.reduce_all(tf.math.is_finite(theta),axis=-1)
        support=finite & tf.reduce_all((theta>self.lower)&(theta<self.upper),axis=-1)
        safe=tf.where(support[:,None],theta,(self.lower+self.upper)[None]/2)
        directions=tf.broadcast_to(tf.eye(p,dtype=theta.dtype)[None],[tf.shape(theta)[0],p,p])
        value,score,status=canonical_batch_fused_value_score(self.model,safe,directions,
            self.initial_base,self.initial_covariances,self.noises,self.observations,
            return_filter_moments=filter_moments,**dict(self.controls))
        status=dict(status)
        status['outside_declared_support']=finite & ~support
        status['value_valid'] &= support
        status['force_valid'] &= support
        code=tf.where(finite,int(FailureCode.OUTSIDE_SUPPORT),int(FailureCode.NONFINITE_INPUT))
        status['value_failure_code']=tf.where(~support,code,status['value_failure_code'])
        status['force_failure_code']=tf.where(~support,code,status['force_failure_code'])
        for key in ('first_failure_time','first_failure_particle'):
            status[key]=tf.where(~support,tf.constant(-1,tf.int32),status[key])
        status['first_failure_stage']=tf.where(~support,tf.constant(int(FailureStage.PRIOR),tf.int32),status['first_failure_stage'])
        status['program_valid']=status['value_valid'] & status['force_valid']
        status['direction_value_valid'] &= support[:,None]
        status['direction_force_valid'] &= support[:,None]
        log_prior=-tf.reduce_sum(tf.math.log(self.upper-self.lower))
        return tf.where(status['value_valid'],value+log_prior,tf.constant(-float('inf'),theta.dtype)),\
            tf.where(status['force_valid'][:,None],score,tf.zeros_like(score)),status

    def unconstrained(self,z):
        z=tf.convert_to_tensor(z,self.initial_base.dtype)
        sigmoid=tf.math.sigmoid(z)
        theta=self.lower+(self.upper-self.lower)*sigmoid
        value,score,status=self.physical(theta)
        # A finite z that rounds to a boundary is numerical exclusion, not true
        # support exclusion: the mathematical sigmoid has not left (lower,upper).
        coordinate_finite=tf.reduce_all(tf.math.is_finite(z),axis=-1)
        coordinate_failed=~status['value_valid'] & status['outside_declared_support']
        status['outside_declared_support']=tf.zeros_like(coordinate_finite)
        for key in ('value_failure_code','force_failure_code'):
            status[key]=tf.where(coordinate_failed | ~coordinate_finite,
                int(FailureCode.NONFINITE_INPUT),status[key])
        log_jac=tf.reduce_sum(tf.math.log(self.upper-self.lower)-tf.nn.softplus(-z)-tf.nn.softplus(z),axis=-1)
        raw_score=score*(self.upper-self.lower)*sigmoid*(1-sigmoid)+(1-2*sigmoid)
        valid=status['value_valid'] & coordinate_finite & tf.math.is_finite(value+log_jac)
        force_valid=valid & status['force_valid'] & tf.reduce_all(tf.math.is_finite(raw_score),axis=-1)
        status['value_valid']=valid;status['force_valid']=force_valid
        status['program_valid']=valid & force_valid
        status['direction_value_valid'] &= valid[:,None]
        status['direction_force_valid'] &= force_valid[:,None]
        return tf.where(valid,value+log_jac,tf.constant(-float('inf'),z.dtype)),\
            tf.where(force_valid[:,None],raw_score,tf.zeros_like(raw_score)),status

    def compile(self,batch_size,*,coordinates='physical',filter_moments=False):
        if coordinates not in ('physical','unconstrained'):raise ValueError('unknown coordinates')
        if coordinates=='unconstrained' and filter_moments:raise ValueError('moments are requested in physical coordinates')
        function=(lambda x:self.physical(x,filter_moments=filter_moments)) if coordinates=='physical' else self.unconstrained
        return tf.function(function,input_signature=[tf.TensorSpec([batch_size,self.parameter_count],self.initial_base.dtype)],jit_compile=True)

def make_linear_fixture_target(dataset,particle_count,controls,*,dtype=tf.float32,draw_design="iid_gaussian"):
    """Issue identity from the actual fixture, inputs, callable and settings."""
    model_id=dataset['model_id'];spec=linear_fixture_spec(model_id,dtype)
    n=int(particle_count);d=spec['d'];t=spec['t']
    chunks=select_transport_chunks(n)
    controls=dict(controls)
    if controls.get('reset_policy')!='contract_e' or controls.get('correction_steps',0)<1 or controls.get('pairwise_steps',0)<1 or controls.get('coordinate_cap',0)<=0:
        raise ValueError('the fixture requires the complete canonical reset and correction')
    model=linear_gaussian_parameter_model(spec['base_f'],spec['base_q'],spec['h'],spec['r'],infer_process_scale=spec['infer_process_scale'])
    seeds={str(p):fixture_seed(model_id,t,dataset['partition'],dataset['attempt'],dataset['replicate'],p) for p in (20,21,22)}
    if draw_design == "iid_gaussian":
        # Preserve the original finite inputs exactly for baseline reproduction.
        base=tf.cast(tf.random.stateless_normal([n,d],seeds['20'],dtype=tf.float64),dtype)
        noises=tf.cast(tf.random.stateless_normal([t,n,d],seeds['21'],dtype=tf.float64),dtype)
        design=tf.cast(tf.random.stateless_normal([n,d],seeds['22'],dtype=tf.float64),dtype)
    elif draw_design == "scrambled_halton_gaussian":
        from bayesfilter.highdim.gaussian_cloud_designs_tf import standard_normal_cloud
        # Joint path design preserves separate coordinates for x0 and every
        # process innovation. Freeze it once; retained HMC never redraws it.
        with tf.device("/CPU:0"):
            path_seed=seeds['20'].numpy().tolist()
            design_seed=seeds['22'].numpy().tolist()
            path=standard_normal_cloud(draw_design,num_particles=n,dimension=(t+1)*d,
                seed=path_seed[0],salt=path_seed[1],dtype=tf.float64)
            base=tf.cast(path[:,:d],dtype)
            noises=tf.cast(tf.transpose(tf.reshape(path[:,d:],[n,t,d]),[1,0,2]),dtype)
            design=tf.cast(standard_normal_cloud(draw_design,num_particles=n,dimension=d,
                seed=design_seed[0],salt=design_seed[1],dtype=tf.float64),dtype)
    else:
        raise ValueError("unreviewed fixture draw design: "+draw_design)
    # Fixed residual design: centered and covariance-whitened once, outside HMC.
    design-=tf.reduce_mean(design,axis=0)
    gram=tf.matmul(design,design,transpose_a=True)/tf.cast(n,dtype)
    valid,chol=safe_cholesky(gram,'fixed_residual_design')
    if not bool(valid):raise ValueError('fixed residual design is singular')
    design=tf.transpose(tf.linalg.triangular_solve(chol,tf.transpose(design)))
    tf.debugging.assert_all_finite(design,'invalid fixed residual design')
    controls['reset_design']=design
    obs=tf.convert_to_tensor(dataset['observations'],dtype)
    if obs.shape!=[t,int(spec['h'].shape[0])]:raise ValueError('dataset shape does not match fixture')
    manifest=dict(schema='bayesfilter.ledh.graceful_fixture.v1',model_id=model_id,
        parameterization='F=a*A,Q=exp(2*b)*Qbase; LG1 fixed Q',
        initial_law='stationary mean zero; separate per-particle UKF design covariance P0',
        priors=dict(lower=spec['lower'],upper=spec['upper'],family='independent_uniform'),
        coordinate_map='lower+(upper-lower)*sigmoid(z); full log Jacobian once',
        route='canonical_batch_fused_value_score',reset_contract_id='contract_e_chol_v1',
        derivative='analytical_recursive_all_parameter_dependencies',
        row_mass_normalization='actual shared Contract E implementation',
        n=n,t=t,d=d,p=len(spec['lower']),dtype=dtype.name,
        tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()) if dtype==tf.float32 else False,
        jit_compile=True,chunk_policy=chunks.__dict__,draw_design=draw_design,
        settings={k:v for k,v in controls.items() if k!='reset_design'},
        input_sha256=dict(base=_hash_tensor(base),noise=_hash_tensor(noises),
            observations=_hash_tensor(obs),residual_design=_hash_tensor(design)),
        seeds={k:v.numpy().tolist() for k,v in seeds.items()},source_sha256=source_closure())
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return BoundedLEDHFixtureTarget(model_id,model,base,tf.eye(d,batch_shape=[n],dtype=dtype),
        noises,obs,tf.constant(spec['lower'],dtype),tf.constant(spec['upper'],dtype),
        MappingProxyType(controls),identity,MappingProxyType(manifest))
