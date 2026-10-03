"""CPU-only guard localization for one failed expanded calibration; no admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import tensorflow as tf
from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
from bayesfilter.highdim import sqmc_campaign_tf as campaign
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score

def main(output):
    tick=time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    spec=FullLGSSMSpec('full_matrix',3)
    theta=spec.default_theta(tf.float64)
    direction=tf.one_hot(0,spec.parameter_count,dtype=tf.float64)
    observations=spec.simulate(theta,2,195002,jit_compile=False)
    inputs=campaign.random_inputs('iid_dual_cap',195002,1020,3,2,tf.float64)
    controls=dict(flow_substeps=8,reset_epsilon=.4,reset_sinkhorn_steps=24,
        reset_balance_steps=12,correction_steps=1,correction_strength=.12,
        pairwise_steps=1,pairwise_strength=.03)
    settings=dict(campaign.numerical_settings(controls),**campaign.route_settings('iid_dual_cap'))
    design=campaign.reset_design(1020,3,tf.float64)
    signature=[tf.TensorSpec(t.shape,t.dtype) for t in (theta,*inputs,observations)]
    @tf.function(input_signature=signature,jit_compile=False,autograph=False)
    def trace(parameters,initial,noise,uniforms,obs):
        model,_=spec.model(parameters,direction)
        states,covs,ds,dc=spec.initial_cloud(parameters,initial,direction)
        return canonical_value_and_analytical_score(model,parameters,states,covs,noise,obs,
            with_score=True,initial_state_tangent=ds,initial_covariance_tangent=dc,
            reset_design=design,process_ancestor_uniforms=uniforms,return_trace=True,**settings)
    value,score,records=trace(theta,*inputs,observations)
    rows=[]
    trace_files={}
    for index,row in enumerate(records):
        transport=row['reset_transport']
        weights=row['posterior_weights']
        rows.append(dict(time_index=index,program_valid=bool(row['program_valid'].numpy()),
            higher_moment_valid=bool(row['higher_moment_valid'].numpy()),
            reset_row_error=float(tf.reduce_max(tf.abs(tf.reduce_sum(transport,axis=1)-1.)).numpy()),
            reset_column_tv_error=float((.5*tf.reduce_sum(tf.abs(tf.reduce_mean(transport,axis=0)-weights))).numpy()),
            posterior_ess=float((1./tf.reduce_sum(weights*weights)).numpy()),
            states_finite=bool(tf.reduce_all(tf.math.is_finite(row['states_after_reset'])).numpy()),
            state_tangents_finite=bool(tf.reduce_all(tf.math.is_finite(row['d_states_after_reset'])).numpy()),
            covariance_min_eigenvalue=float(tf.reduce_min(tf.linalg.eigvalsh(row['covariances_after_reset'])).numpy()),
            covariance_tangents_finite=bool(tf.reduce_all(tf.math.is_finite(row['d_covariances_after_reset'])).numpy())))
        for key,tensor in row.items():
            name=f'step{index}-{key}.tensor'
            raw=tf.io.serialize_tensor(tensor).numpy()
            (output/name).write_bytes(raw)
            trace_files[name]=hashlib.sha256(raw).hexdigest()
    result=dict(program='CPU-only guard localization; no accuracy or promotion evidence',
        cpu_gpu_status='CUDA_VISIBLE_DEVICES=-1; GPUs intentionally hidden',jit_compile=False,
        compared_saved_case='run-01/full_d3_T2__iid_dual_cap, calibration seed 195002, flow 8',
        raw_value=str(float(value.numpy())),raw_score=score.numpy().tolist(),controls=settings,
        trace=rows,trace_sha256=trace_files,wall_seconds=time.perf_counter()-tick)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trace_sha256'},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
