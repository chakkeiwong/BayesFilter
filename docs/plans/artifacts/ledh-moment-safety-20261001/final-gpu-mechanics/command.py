"""GPU/XLA mechanics checks after the strict-positive trial-covariance audit."""
import hashlib,json,os,sys,time
from pathlib import Path
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import tensorflow as tf
gpus=tf.config.list_physical_devices('GPU')
for g in gpus: tf.config.experimental.set_memory_growth(g,True)
assert gpus and all(tf.config.experimental.get_memory_growth(g) for g in gpus)
tf.config.experimental.enable_tensor_float_32_execution(True)
root=Path('/home/chakwong/BayesFilter-SQMC');sys.path.insert(0,str(root))
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.sqmc_campaign_tf import reset_design
from bayesfilter.highdim.moment_safety_tf import SAFETY_KEYS
out=root/'docs/plans/artifacts/ledh-moment-safety-20261001/final-gpu-mechanics'
out.mkdir(parents=True,exist_ok=False)
(out/'command.py').write_bytes(Path(__file__).read_bytes())
started=time.monotonic();records=[]
for dtype in [tf.float64,tf.float32]:
 for d,n in [(1,1008),(2,96)]:
  q=reset_design(n,d,dtype,'normal_quantiles');x=q+.15*q*q;w=tf.ones([n],dtype)/n
  dx=tf.random.stateless_normal([n,d,1],[991,d],dtype=dtype)*.01
  dp=tf.random.stateless_normal([n,d,1],[991,8],dtype=dtype)*.01
  dw=tf.zeros([n,1],dtype)
  specs=[tf.TensorSpec(v.shape,v.dtype) for v in (x,w,dx,dw,q,dp)]
  controls=dict(correction_steps=1,strength=.03,diagonal_lm_damping=.01,diagonal_lm_scale_floor=1e-4,
                diagonal_trust_radius=.5,pairwise_correction_steps=0 if d==1 else 1,pairwise_strength=.03,
                pairwise_particle_rms_cap=2.,coordinatewise_standardized_cap=.98,
                coordinatewise_standardized_identity_radius=8.)
  def make(safety):
   @tf.function(input_signature=specs,jit_compile=True,autograph=False)
   def kernel(x,w,dx,dw,p,dp):
    r=higher_moment_shape_jvp(x,w,dx,dw,p,dp,**controls,moment_safety=safety)
    return {k:r[k] for k in ('particles','particles_tangent','valid','fraction_coordinatewise_cap_active',*SAFETY_KEYS)}
   return kernel
  guarded,old=make(True),make(False)
  for case in ['healthy','active_tail']:
   points=q if case=='healthy' else tf.tensor_scatter_nd_update(q,[[0]],[tf.ones([d],dtype)*1000.])
   with tf.device('/GPU:0'):
    r=guarded(x,w,dx,dw,points,dp);b=old(x,w,dx,dw,points,dp)
   record=dict(dtype=dtype.name,dimension=d,case=case,valid=bool(r['valid']),
       finite=bool(tf.reduce_all(tf.math.is_finite(r['particles_tangent']))),
       cap_fraction=float(r['fraction_coordinatewise_cap_active']),
       value_difference=float(tf.reduce_max(tf.abs(r['particles']-b['particles']))),
       tangent_difference=float(tf.reduce_max(tf.abs(r['particles_tangent']-b['particles_tangent']))),
       **{k:float(r[k]) for k in SAFETY_KEYS})
   assert record['valid'] and record['finite']
   eps=2.**(-23 if dtype==tf.float32 else -52)
   assert record['moment_safety_final_loss']<=record['moment_safety_baseline_loss']+32*eps*(1+record['moment_safety_baseline_loss'])
   if case=='active_tail' and d==1: assert record['cap_fraction']>0
   if case=='healthy' and dtype==tf.float64:
    h=tf.constant(1e-5,dtype)
    with tf.device('/GPU:0'):
     plus=guarded(x+h*dx[:,:,0],w,dx,dw,points+h*dp[:,:,0],dp)
     minus=guarded(x-h*dx[:,:,0],w,dx,dw,points-h*dp[:,:,0],dp)
    for key in ['moment_safety_trials','moment_safety_final_rejected']:
     tf.debugging.assert_equal(plus[key],minus[key])
    fd=(plus['particles']-minus['particles'])/(2*h)
    record['fd_max_error']=float(tf.reduce_max(tf.abs(fd-r['particles_tangent'][:,:,0])))
    tf.debugging.assert_near(fd,r['particles_tangent'][:,:,0],atol=1e-7,rtol=1e-5)
   records.append(record)
   print(json.dumps(record),flush=True)
manifest=dict(wall_seconds=time.monotonic()-started,records=records,device='/GPU:0',jit_compile=True,tf32=True,
    environment=sys.executable,tensorflow=tf.__version__,gpu_growth=[{'name':g.name,'growth':True} for g in gpus],
    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
       ['bayesfilter/highdim/higher_moment_contract_e.py','bayesfilter/highdim/moment_safety_tf.py']})
(out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
