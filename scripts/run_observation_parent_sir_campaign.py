"""Bounded GPU FP64/XLA comparative diagnostic; no canonical admission."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys,json,time,math,argparse,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
MEMORY_POLICY=configure_tensorflow_gpu_memory_growth(tf)
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.observation_parent_proposal_tf import make_conditional_proposal
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
D=tf.float64;SPEC=NonlinearSQMCSpec('sir_d18')
CASES={'nominal':[0.,0.,0.],'infection09':[math.log(.9),0.,0.],'infection11':[math.log(1.1),0.,0.], 'recovery09':[0.,math.log(.9),0.],'recovery11':[0.,math.log(1.1),0.], 'noise08':[0.,0.,math.log(.8)],'noise125':[0.,0.,math.log(1.25)]}
POLICIES=['original_three_branch','conditional_uniform','conditional_observation_weighted']
ARMS=['original','conditional_uniform','conditional_weighted','identity_only','uniform_only','weighted_only']
DATA=Path('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-zhao-horizons-20261006-01/037-zhao-sir_d18-rank20-fit17-runtime-repair/sir_austria-input-dataset.json')
PLAN='docs/plans/observation-parent-default-and-sir-accuracy-20261028.md'
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def controls(policy):return ProposalControls(proposal_policy=policy,flow_steps=16,reset_steps=40,reset_epsilon=1.,correction_steps=0)
def build(n,t,policy,pure=False):
    c=controls(policy);hook=None
    if pure:
        shared=make_conditional_proposal(SPEC,c,1)
        def hook(models,maps,beta,uniforms,innovations,observation):
            child,densities,valid,parents=shared(models,maps,tf.constant([0.,1.,0.],D),uniforms,innovations,observation)
            return child,densities,valid,parents,densities[:,:,1]
    return make_batched_score_filter(SPEC,n,t,c,D,True,1,diagnostic_proposal_hook=hook)
def inputs(n,t,seed,obs):
    return [tf.constant(obs[:t],D),tf.random.stateless_normal([n,18],[seed,11],dtype=D),tf.random.stateless_normal([t,n,18],[seed,12],dtype=D),tf.random.stateless_uniform([t,n,2],[seed,13],dtype=D),tf.constant([1/3]*3,D)]
def smoke(obs):
    a=inputs(32,3,281200,obs);theta=tf.zeros([1,3],D)
    f=build(32,3,POLICIES[2]);r=f(theta,*a)
    b=[a[0][:2],a[1],a[2][:2],a[3][:2],a[4]];s=build(32,2,POLICIES[2])(theta,*b)
    tf.debugging.assert_equal(r['valid'],[True]);tf.debugging.assert_equal(s['valid'],[True])
    tf.debugging.assert_near(tf.reduce_sum(r['trace'][:,:2,0],axis=1),s['value'],atol=1e-7,rtol=1e-9)
    tf.debugging.assert_near(tf.reduce_sum(r['score_trace'][:,:2],axis=1),s['score'],atol=1e-6,rtol=1e-8)
    errors=[]
    for k in range(3):
        delta=tf.one_hot(k,3,dtype=D)[None]*1e-6;lo=f(theta-delta,*a);hi=f(theta+delta,*a)
        tf.debugging.assert_equal(lo['valid'] & hi['valid'],[True])
        tf.debugging.assert_equal(lo['diagnostic_parent_indices'],hi['diagnostic_parent_indices'])
        fd=(hi['value']-lo['value'])/2e-6
        error=float(tf.abs(fd[0]-r['score'][0,k])/(1+tf.abs(r['score'][0,k])));errors.append(error)
        assert error<2e-3,errors
    return dict(passed=True,prefix_equality=True,scaled_fd_errors=errors)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=float,default=5100);p.add_argument('--pilot',action='store_true');p.add_argument('--seeds',type=int,default=8);p.add_argument('--resume-rows',type=Path);p.add_argument('--local-only',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];durations=[]
    memory=MEMORY_POLICY
    assert tf.config.list_physical_devices('GPU'),'GPU required'
    data=json.loads(DATA.read_text());obs=data['observations'];n=1008;t=50
    source_paths=[Path(__file__),ROOT/'bayesfilter/highdim/observation_parent_proposal_tf.py',ROOT/'bayesfilter/highdim/covariance_proposal_score_batch_tf.py',ROOT/'bayesfilter/highdim/covariance_proposal_tf.py',ROOT/'bayesfilter/highdim/covariance_proposal_score_tf.py',ROOT/'bayesfilter/highdim/covariance_proposal_native_models_tf.py']
    manifest=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),environment=sys.executable,tensorflow=tf.__version__,device='GPU:0',visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],gpu_memory_policy=memory,dtype='float64',jit_compile=True,tf32=tf.config.experimental.tensor_float_32_execution_enabled(),n=n,horizon=t,chunk=n,seeds=list(range(281201,281201+a.seeds)),data_seed=26100611,data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),controls=vars(controls(POLICIES[2])),plan=PLAN,result='docs/plans/observation-parent-sir-accuracy-result-20261028.md',source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_paths},limit=a.limit,scope='untuned FP64 diagnostic; not canonical LEDH admission',status='running')
    save(a.out/'manifest.json',manifest)
    try:
        save(a.out/'smoke.json',smoke(obs))
        programs={}
        previous=json.loads(a.resume_rows.read_text()) if a.resume_rows else []
        done={(x['seed'],x['case'],x['arm']) for x in previous if x['horizon']==50}
        cases={'nominal':CASES['nominal']} if a.pilot else CASES
        local_cases={k:[math.log(v) if j==i else 0. for j in range(3)] for i,name in [(0,'infection'),(1,'recovery')] for k,v in [(name+'0999',.999),(name+'1001',1.001)]}
        if a.local_only:cases=local_cases
        elif not a.pilot:cases={**cases,**local_cases}
        stop=False
        for seed in manifest['seeds']:
            args=inputs(n,t,seed,obs)
            for case,theta in cases.items():
                for index,arm in enumerate(ARMS):
                    if (seed,case,arm) in done:continue
                    reserve=max(durations[-6:],default=60)*1.3+15
                    if time.monotonic()-start+reserve>a.limit:stop=True;break
                    tick=time.monotonic();policy=POLICIES[index if index<3 else max(index-3,0)];pure=index>=4
                    key=(policy,pure)
                    if key not in programs:programs[key]=build(n,t,policy,pure)
                    args[-1]=tf.constant([1.,0.,0.] if index>=3 else [1/3]*3,D)
                    r=programs[key](tf.constant([theta],D),*args)
                    trace=r['trace'][0].numpy().tolist();scores=r['score_trace'][0].numpy().tolist();checks=r['validity_trace'][0].numpy().tolist()
                    elapsed=time.monotonic()-tick;durations.append(elapsed)
                    for horizon in (10,20,50):
                        valid=all(all(c) for c in checks[:horizon]);finite=all(math.isfinite(z) for z in [v[0] for v in trace[:horizon]]+[v for row in scores[:horizon] for v in row])
                        row=dict(seed=seed,case=case,theta=theta,arm=arm,horizon=horizon,valid=valid and finite,elapsed_full50=elapsed,failed_steps=[j+1 for j,c in enumerate(checks[:horizon]) if not all(c)],failed_check_indices=[j for j in range(len(checks[0])) if not all(c[j] for c in checks[:horizon])])
                        if valid and finite:row.update(value=sum(x[0] for x in trace[:horizon]),score=[sum(x[k] for x in scores[:horizon]) for k in range(3)],ess_min=min(x[2] for x in trace[:horizon]),ess_mean=sum(x[2] for x in trace[:horizon])/horizon,covariance_error_max=max(x[6] for x in trace[:horizon]))
                        rows.append(row)
                    save(a.out/'rows.json',rows)
                    manifest.update(wall_seconds=time.monotonic()-start,row_count=len(rows),allocator=tf.config.experimental.get_memory_info('GPU:0'));save(a.out/'manifest.json',manifest)
                    print(json.dumps(dict(seed=seed,case=case,arm=arm,valid50=rows[-1]['valid'],seconds=elapsed,elapsed=manifest['wall_seconds'])),flush=True)
                if stop:break
            if stop:break
        manifest['status']='budget_stopped' if stop else 'complete'
    except Exception as error:
        manifest.update(status='failed',exception=repr(error));raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-start,row_count=len(rows));save(a.out/'manifest.json',manifest)
if __name__=='__main__':main()
