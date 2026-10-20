"""Scope-specific transport-resolution repair; shared numerical implementation."""
from pathlib import Path
import json
import math
import time
import run_sqmc_expanded_comparison as campaign

PLAN='docs/plans/sqmc-expanded-transport-repair-20260928.md'
EPSILON_GRID=(.4,1.6,6.4,25.6,102.4)
PROFILE='transport_epsilon_calibration_v2'


def trace_summary_kernel(spec,route,controls,n,horizon,*,dynamic_epsilon=False):
    import tensorflow as tf
    from bayesfilter.highdim import sqmc_campaign_tf as numerical
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    dtype=tf.float64
    settings=dict(numerical.numerical_settings(controls),**numerical.route_settings(route))
    signature=[tf.TensorSpec([spec.parameter_count],dtype),
        tf.TensorSpec([n,spec.dimension],dtype),tf.TensorSpec([horizon,n,spec.dimension],dtype),
        tf.TensorSpec([horizon,n],dtype),tf.TensorSpec([horizon,spec.dimension],dtype)]
    if dynamic_epsilon: signature.append(tf.TensorSpec([],tf.float32))
    @tf.function(input_signature=signature,jit_compile=True,autograph=False)
    def compute(theta,initial,noise,uniforms,observations,epsilon=None):
        direction=tf.one_hot(0,spec.parameter_count,dtype=dtype)
        design=numerical.reset_design(n,spec.dimension,dtype)
        current_settings=dict(settings)
        if dynamic_epsilon: current_settings["reset_epsilon"]=epsilon
        model,_=spec.model(theta,direction)
        states,covs,ds,dc=spec.initial_cloud(theta,initial,direction)
        value,score,trace=canonical_value_and_analytical_score(model,theta,states,covs,noise,observations,
            with_score=True,initial_state_tangent=ds,initial_covariance_tangent=dc,
            reset_design=design,process_ancestor_uniforms=uniforms,return_trace=True,**current_settings)
        # Traverse the fixed output schema only, then reduce all dates with
        # tensor axes. Do not unroll one numerical reduction per time point.
        transport=tf.stack([r['reset_transport'] for r in trace])
        weights=tf.stack([r['posterior_weights'] for r in trace])
        return value,score,dict(
            program_valid=tf.stack([r['program_valid'] for r in trace]),
            higher_moment_valid=tf.stack([r['higher_moment_valid'] for r in trace]),
            row_error=tf.reduce_max(tf.abs(tf.reduce_sum(transport,axis=2)-1.),axis=1),
            column_tv=.5*tf.reduce_sum(tf.abs(tf.reduce_mean(transport,axis=1)-weights),axis=1))
    return compute


def summarize_trace(output):
    value,score,diagnostics=output
    values={k:v.numpy().tolist() for k,v in diagnostics.items()}
    raw_value=float(value.numpy())
    raw_score=score.numpy().tolist()
    valid=math.isfinite(raw_value) and all(math.isfinite(x) for x in raw_score) and all(values['program_valid'])
    row_error=max(values['row_error'])
    column_tv=max(values['column_tv'])
    return dict(valid=valid,nomination_pass=valid and row_error<=1e-8 and column_tv<=1e-6,
        raw_value=raw_value if math.isfinite(raw_value) else str(raw_value),
        raw_score=[v if math.isfinite(v) else str(v) for v in raw_score],
        maximum_row_error=row_error,maximum_column_tv=column_tv,per_time=values)


def prepare_controls(spec,route,theta,horizon,n,out):
    from bayesfilter.highdim import sqmc_campaign_tf as numerical
    observations={seed:spec.simulate(theta,horizon,seed,jit_compile=True) for seed in campaign.CAL}
    inputs={seed:numerical.random_inputs(route,seed,n,spec.dimension,horizon,theta.dtype) for seed in campaign.CAL}
    result=dict(profile=PROFILE,scope=dict(target=spec.target_id,route=route,horizon=horizon,n=n),
        calibration_seeds=campaign.CAL,epsilon_grid=EPSILON_GRID,column_nomination_limit=1e-6,
        row_nomination_limit=1e-8,rows=[],selected_balance_steps=12,selected_epsilon=None,
        role='scalar validity nomination only; full-score calibration/validation still required')
    import tensorflow as tf
    kernel=trace_summary_kernel(spec,route,campaign.CONTROLS[-1],n,horizon,dynamic_epsilon=True)
    for epsilon in EPSILON_GRID:
        current=[]
        for seed in campaign.CAL:
            tick=time.perf_counter()
            row=summarize_trace(kernel(theta,*inputs[seed],observations[seed],tf.constant(epsilon,tf.float32)))
            row.update(seed=seed,reset_epsilon=epsilon,wall_seconds=time.perf_counter()-tick)
            result['rows'].append(row);current.append(row)
            campaign.dump(out/'transport_calibration.json',result)
            progress=dict(stage='transport_calibration',flow_substeps=8,seed=seed,
                valid=row['valid'],nomination_pass=row['nomination_pass'],epsilon=epsilon,
                maximum_column_tv=row['maximum_column_tv'],wall_seconds=row['wall_seconds'])
            with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True)
        if all(r['nomination_pass'] for r in current):
            result['selected_epsilon']=epsilon
            campaign.dump(out/'transport_calibration.json',result)
            return [dict(c,reset_epsilon=epsilon) for c in campaign.CONTROLS]
    campaign.dump(out/'transport_calibration.json',result)
    return None


def configure_profile():
    campaign.SCOPES=tuple(s for s in campaign.SCOPES if s[1]=='full_matrix')
    campaign.CAL=(199001,199002)
    campaign.VAL=(200001,)
    campaign.CLAIM=(201001,201002)
    campaign.FILTER=(202001,202002)
    campaign.ENTRY_POINT=Path(__file__).resolve()
    campaign.EXTRA_SOURCE_PATHS=(Path(__file__).resolve(),campaign.ROOT/PLAN,
        campaign.ROOT/'docs/benchmarks/check_sqmc_expanded_repair_gpu.py')
    campaign.EXTRA_CONTRACT=dict(repair_profile=PROFILE,transport_epsilon_grid=EPSILON_GRID,
        column_nomination_limit=1e-6,row_nomination_limit=1e-8)
    campaign.CONTROL_PREPARER=prepare_controls


if __name__=='__main__':
    configure_profile()
    campaign.main()
