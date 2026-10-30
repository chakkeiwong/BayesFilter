import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import json, time, subprocess, sys, hashlib, signal
def _timeout(*args):
    raise TimeoutError('400-second GPU attempt budget reached')
signal.signal(signal.SIGALRM,_timeout)
signal.alarm(400)
from pathlib import Path
root = Path('/home/chakwong/BayesFilter-SQMC')
sys.path.insert(0, str(root))
sys.path.insert(0, '/home/chakwong/python/src')
out = root/'docs/plans/artifacts/sqmc-repair-20260925/08-final-gpu-xla'
out.mkdir(exist_ok=False)
started = time.perf_counter()
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'trusted TensorFlow GPU probe found no device'
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
assert all(tf.config.experimental.get_memory_growth(gpu) for gpu in gpus)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.sqmc_campaign_tf import value_and_score, random_inputs
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
controls = dict(reset_epsilon=.4, reset_sinkhorn_steps=20, reset_balance_steps=8,
                correction_steps=1, correction_strength=.1, pairwise_steps=1,
                pairwise_strength=.1, flow_substeps=2)
manifest = dict(command=sys.argv, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    environment=sys.executable, cpu_only=False, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    tf_version=tf.__version__, tf32=True, memory_policy='memory_growth',
    physical_gpus=[dict(name=g.name,details=tf.config.experimental.get_device_details(g),growth=tf.config.experimental.get_memory_growth(g)) for g in gpus],
    plan='docs/plans/sqmc-repair-master-program-20260925.md', seeds=[82001,82002],
    data_version='shared_p44_predict_first_v2', diagnostic_only=True, controls=controls)
manifest['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'bayesfilter/highdim').glob('*.py')}
rows=[]
try:
    with tf.device('/GPU:0'):
        probe=tf.linalg.matmul(tf.eye(3),tf.eye(3))
        assert 'GPU' in probe.device
        manifest['framework_probe_device']=probe.device
        (out/'probe.json').write_text(json.dumps(manifest,indent=2))
        for dtype in (tf.float64,tf.float32):
            spec=LGSSMSpec('p44',3); theta=spec.default_theta(dtype)
            obs=spec.simulate(theta,2,82001)
            for route in ('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation'):
                before=time.perf_counter()
                inputs=random_inputs(route,82002,12,3,2,dtype)
                reference=value_and_score(spec,route,controls,theta,obs,82002,12,jit_compile=False,inputs=inputs)
                compiled=value_and_score(spec,route,controls,theta,obs,82002,12,jit_compile=True,inputs=inputs)
                assert bool(reference[2]) and bool(compiled[2]), 'nonfinite reference or compiled result'
                dv=float(tf.abs(reference[0]-compiled[0])); ds=float(tf.norm(reference[1]-compiled[1]))
                scale=max(1.,float(tf.norm(reference[1])))
                tol=1e-7 if dtype==tf.float64 else 5e-3
                row=dict(dtype=dtype.name,route=route,value_difference=dv,score_difference=ds,
                    score_scale=scale,relative_tolerance=tol,passed=dv<=tol*max(1.,float(tf.abs(reference[0]))) and ds<=tol*scale,
                    device=compiled[0].device,wall_seconds=time.perf_counter()-before,
                    value=float(compiled[0]),score=compiled[1].numpy().tolist())
                rows.append(row); print(json.dumps(row),flush=True)
                assert row['passed'], 'GPU/XLA parity failed'
        manifest['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    manifest['exit_code']=0
except BaseException as exc:
    manifest['exit_code']=1; manifest['error']=repr(exc)
    raise
finally:
    manifest['wall_seconds']=time.perf_counter()-started; manifest['results']=rows
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
