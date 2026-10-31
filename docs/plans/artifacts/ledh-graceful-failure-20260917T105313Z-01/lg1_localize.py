"""Independent scalar-flow and Gaussian-mixture reference, diagnostic only.

Runtime candidate remains the shared canonical engine. Scalar formulas here
are an independent arithmetic/importance-identity check, never a candidate.
"""
import json, math, os, sys, time
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import _single_cloud_model
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score

root=Path(sys.argv[1]);out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
prior=json.loads((root/'lg1-calibration-diagnostic-01/diagnostic.json').read_text())
controls=prior['controls'];n=4096;dtype=tf.float32
records=[]
report=dict(classification='independent_diagnostic_not_admission',memory_policy=memory,
    data_partition='calibration',particle_count=n,records=records,
    question='Does each finite flow/weight evaluation equal independent scalar algebra, and how far is its importance approximation from the exact conditional empirical mixture?',
    budgets=dict(relative_flow=1e-4,relative_logit=1e-4),
    nonclaim='A conditional mixture based on the actual incoming particles is not the exact global filtering law.')
def scalar_normal(x,m,v): return -.5*(tf.math.log(tf.constant(2*math.pi,tf.float64)*v)+tf.square(x-m)/v)
def f64(x):return tf.cast(x,tf.float64)
for replicate in (0,2):
    dataset=json.loads((root/f'lg1-data-01/data/LG1-calibration-{replicate:02}.json').read_text())
    target=make_linear_fixture_target(dataset,n,controls,dtype=dtype)
    settings=dict(target.controls);settings['flow_substeps']=settings.pop('substeps')
    @tf.function(input_signature=[tf.TensorSpec([1],dtype)],jit_compile=True)
    def trace_fn(theta):
        p=1/(1-tf.square(theta[0]));dp=2*theta[0]*p*p
        states=target.initial_base*tf.sqrt(p)
        dstates=target.initial_base*(dp/(2*tf.sqrt(p)))
        covs=tf.ones([n,1,1],dtype)*p;dcovs=tf.ones_like(covs)*dp
        value,score,trace,status=canonical_value_and_analytical_score(
            _single_cloud_model(target.model,theta,tf.ones_like(theta)),theta,
            states,covs,target.noises,target.observations,
            initial_state_tangent=dstates,initial_covariance_tangent=dcovs,
            with_score=True,return_trace=True,return_status=True,**settings)
        keys=('pre_flow','children','posterior_weights','posterior_logits',
              'predicted_covariances','states_after_reset','covariances_after_reset')
        return value,score,status,{key:tf.stack([step[key] for step in trace]) for key in keys}
    for phi in (-.9,0.,.9):
        started=time.monotonic()
        value,score,status,trace=trace_fn(tf.constant([phi],dtype))
        assert bool(status['value_valid']) and bool(status['force_valid'])
        # Match the represented float32 physical parameter in the reference.
        physical=f64(tf.constant(phi,dtype));initial_p=1/(1-physical*physical)
        incoming=f64(target.initial_base[:,0])*tf.sqrt(initial_p)
        incoming_cov=tf.ones([n],tf.float64)*initial_p
        steps=[];mixture_ll=0.
        for t in range(10):
            y=f64(target.observations[t,0]);anchor=physical*incoming
            predicted=physical*physical*incoming_cov+1
            pre=anchor+f64(target.noises[t,:,0]);moved=pre
            logdet=tf.zeros([n],tf.float64)
            count=controls['substeps'];eps=1/count
            for j in range(count):
                lam=(j+1)/count
                a=-.5*predicted/(lam*predicted+.5)
                b=(1+2*lam*a)*((1+lam*a)*predicted*2*y+a*anchor)
                moved=moved+eps*(a*moved+b)
                logdet+=tf.math.log(tf.abs(1+eps*a))
            logits=-math.log(n)+scalar_normal(moved,anchor,1.)+scalar_normal(y,moved,.5)-scalar_normal(pre,anchor,1.)+logdet
            actual=f64(trace['children'][t,:,0]);actual_logits=f64(trace['posterior_logits'][t])
            flow_error=float(tf.reduce_max(tf.abs(actual-moved)/(1+tf.abs(moved))))
            logit_error=float(tf.reduce_max(tf.abs(actual_logits-logits)/(1+tf.abs(logits))))
            covariance_error=float(tf.reduce_max(tf.abs(f64(trace['predicted_covariances'][t,:,0,0])-predicted)/(1+predicted)))
            weights=f64(trace['posterior_weights'][t]);weights/=tf.reduce_sum(weights)
            actual_mean=tf.reduce_sum(weights*actual);actual_cov=tf.reduce_sum(weights*tf.square(actual-actual_mean))
            conditional_logits=scalar_normal(y,anchor,1.5)-math.log(n)
            conditional_ll=tf.reduce_logsumexp(conditional_logits)
            component_weights=tf.nn.softmax(conditional_logits)
            component_means=(.5*anchor+y)/1.5
            mixture_mean=tf.reduce_sum(component_weights*component_means)
            mixture_cov=1/3+tf.reduce_sum(component_weights*tf.square(component_means-mixture_mean))
            mixture_ll+=float(conditional_ll)
            steps.append(dict(time=t,flow_scaled_error=flow_error,logit_scaled_error=logit_error,
                predicted_covariance_scaled_error=covariance_error,
                importance_ess=float(1/tf.reduce_sum(weights*weights)),
                empirical_mixture_mean=float(mixture_mean),empirical_mixture_covariance=float(mixture_cov),
                importance_mean=float(actual_mean),importance_covariance=float(actual_cov),
                importance_to_mixture_mean_error=float(tf.abs(actual_mean-mixture_mean)/tf.sqrt(mixture_cov)),
                importance_to_mixture_cov_error=float(tf.abs(actual_cov/mixture_cov-1)),
                importance_log_increment=float(tf.reduce_logsumexp(actual_logits)),
                exact_empirical_mixture_log_increment=float(conditional_ll)))
            assert flow_error<1e-4 and logit_error<1e-4 and covariance_error<1e-4,(replicate,phi,t,steps[-1])
            incoming=f64(trace['states_after_reset'][t,:,0])
            incoming_cov=f64(trace['covariances_after_reset'][t,:,0,0])
        row=dict(replicate=replicate,phi=phi,log_likelihood=float(value),score=float(score),
            conditional_mixture_log_likelihood=mixture_ll,steps=steps,seconds=time.monotonic()-started)
        records.append(row)
        (out/'localization.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(replicate=replicate,phi=phi,
            max_flow=max(q['flow_scaled_error'] for q in steps),
            max_logit=max(q['logit_scaled_error'] for q in steps),
            max_conditional_mean_error=max(q['importance_to_mixture_mean_error'] for q in steps),
            max_conditional_cov_error=max(q['importance_to_mixture_cov_error'] for q in steps))),flush=True)
report['complete']=True
report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
(out/'localization.json').write_text(json.dumps(report,indent=2)+'\n')
