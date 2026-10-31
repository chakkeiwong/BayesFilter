#!/usr/bin/env python3
"""Debug-only first-step guard decomposition using shared filter primitives.

This is not a filtering candidate: it computes no final likelihood, score,
timing comparison, tuning decision or HMC result. It explains a rejection.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import five_model_precision_diagnostic as campaign
base=campaign.base


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    tf,memory=base.prepare(SimpleNamespace(gpu=args.gpu,dtype='float32',no_tf32=False))
    from bayesfilter.highdim import covariance_proposal_tf as runtime
    from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
    from docs.benchmarks.run_ledh_covariance_proposal import spec_for
    from bayesfilter.highdim.sqmc_campaign_tf import random_inputs
    dtype=tf.float32
    result=dict(evidence_role='first_step_guard_localization_only',memory_policy=memory,
                command=[sys.executable,*sys.argv],models=[],source_sha256=campaign.provenance())
    for name in ('range_bearing','lgssm','ksc','predator_prey','sir_d18'):
        saved_dir=args.campaign/campaign.label(name,1008,'fp32_tf32')
        manifest=base.load(saved_dir/'manifest.json')
        controls=runtime.ProposalControls(**manifest['controls'])
        n,horizon=1008,manifest['horizon']
        dataset_path=Path(next(iter(manifest['data_sha256'])))
        dataset=base.load(dataset_path)

        def common(value):
            return tf.cast(tf.convert_to_tensor(value,dtype=tf.float64),dtype)

        with tf.device('/CPU:0'):
            beta=common(manifest['beta']); obs=common(dataset['observations'])
            if name=='range_bearing':
                _,spec=base.specs(tf,dtype)
                spec64,_=base.specs(tf,tf.float64)
                fixed=base.inputs(tf,spec64,n,31001,dtype)[:3]
                theta=common(base.POINTS[0])
            else:
                spec=spec_for(name,lgssm_family='p44')
                design=random_inputs('iid_dual_cap',261008301,n,spec.dimension,horizon,tf.float64,jit_compile=True)
                extra=tf.random.stateless_uniform([horizon,n,2],[261008301,991],dtype=tf.float64)
                fixed=tuple(common(x) for x in (*design[:2],extra))
                theta=common(spec.default_theta(tf.float64))
            hashes=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in (theta,obs,*fixed,beta)]
            source_row=campaign.read_rows(saved_dir/'measurements.jsonl')[0]
            assert hashes==source_row['input_sha256']
        sig=[tf.TensorSpec([spec.parameter_count],dtype),tf.TensorSpec([spec.parameter_count],dtype),
             tf.TensorSpec(obs.shape,dtype),tf.TensorSpec(fixed[0].shape,dtype),
             tf.TensorSpec(fixed[1].shape,dtype),tf.TensorSpec(fixed[2].shape,dtype),tf.TensorSpec([3],dtype)]

        @tf.function(input_signature=sig,jit_compile=True,autograph=False)
        def inspect(theta,direction,observations,initial,noise,uniforms,beta):
            model,_=spec.model(theta,direction)
            x,_,dx,_=spec.initial_cloud(theta,initial,direction)
            maps=runtime.build_maps(model,theta,x,dx,observations[0],controls)
            child,dchild=runtime.draw_from_maps(maps,beta,uniforms[0],noise[0])
            ld,dd,density_valid=runtime.component_densities(child,dchild,*maps[:4])
            mix=ld+tf.math.log(beta)[None]
            lq=tf.reduce_logsumexp(mix,-1)
            dlq=tf.reduce_sum(tf.nn.softmax(mix,-1)*dd,-1)
            lg,dlg=runtime.observation_density(model,theta,child,dchild,observations[0])
            lr=lg+ld[:,0]-lq; dlr=dlg+dd[:,0]-dlq
            normalizer=tf.reduce_logsumexp(lr); weights=tf.nn.softmax(lr)
            ds=tf.reduce_sum(weights*dlr)
            z,dz,mu,dmu,p,dp,reset_valid,rd=runtime.reset_cloud(child,dchild,lr-normalizer,dlr-ds,controls)
            c=maps[2]
            scale=tf.reduce_max(tf.abs(c),[-2,-1],keepdims=True)
            normalized=c/tf.where(scale>0,scale,tf.ones_like(scale))
            asym=tf.reduce_max(tf.abs(normalized-tf.linalg.matrix_transpose(normalized)),[-2,-1])
            chol_valid,_=safe_cholesky(c)
            return dict(map_valid=maps[12],density_valid=density_valid,reset_valid=reset_valid,
                        finite_state=tf.reduce_all(tf.math.is_finite(z)),finite_tangent=tf.reduce_all(tf.math.is_finite(dz)),
                        finite_increment=tf.math.is_finite(normalizer),finite_score=tf.math.is_finite(ds),
                        mean_error=rd[0],covariance_error=rd[1],reset_margin=rd[2],
                        branch_max_relative_asymmetry=tf.reduce_max(asym,axis=1),
                        branch_invalid_factor_count=tf.reduce_sum(tf.cast(~chol_valid,tf.int32),axis=1),
                        symmetry_tolerance=tf.constant(8*spec.dimension*2.**-23,dtype))

        with tf.device('/GPU:0'):
            row=inspect(theta,tf.one_hot(0,spec.parameter_count,dtype=dtype),obs,*fixed,beta)
        row.update(model=name,source_full_filter_valid=source_row['valid'],input_sha256=hashes,
                   graph_trace_count=inspect.experimental_get_tracing_count())
        result['models'].append(base.clean(row))
        result['wall_seconds']=time.monotonic()-start
        base.save(args.output/'guard-diagnostic.json',result)
        print(json.dumps(result['models'][-1]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='GPU-68251639-fe82-8f81-3ccc-2953c32e805b')
    args=parser.parse_args()
    run(args)


if __name__=='__main__':
    main()
