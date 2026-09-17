"""LG1 calibration diagnostics only; never read validation or claim observations."""
import json,math,os,time,sys
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target

root=Path(sys.argv[1]);out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
data=root/'lg1-data-01/data'
controls=dict(substeps=10,reset_policy='contract_e',reset_epsilon=2.,
    reset_sinkhorn_steps=8,reset_balance_steps=8,reset_ridge=1e-5,
    correction_steps=2,correction_strength=.2,correction_lm_damping=1e-3,
    correction_lm_scale_floor=1e-6,correction_trust_radius=.1,
    pairwise_steps=2,pairwise_strength=.02,pairwise_rms_cap=2.,coordinate_cap=.98,
    coordinate_cap_power=8)
report=dict(classification='calibration_diagnostic_not_admission',memory_policy=memory,
    controls=controls,particle_count=256,prior='Uniform(-.95,.95)',tf32=True,jit_compile=True,
    calibration=[],derivatives=[],coordinate_derivatives=[])
def save():
    (out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
def host(status):return {k:v.numpy().tolist() for k,v in status.items()}
def kalman(phi,observations):
    # Independent scalar reference in this diagnostic module, float64 Python math.
    mean=0.;cov=1/(1-phi*phi);logvalue=0.;means=[];covs=[]
    for row in observations:
        mean=phi*mean;cov=phi*phi*cov+1
        innovation=row[0]-mean;variance=cov+.5
        logvalue-=.5*(math.log(2*math.pi*variance)+innovation*innovation/variance)
        gain=cov/variance;mean+=gain*innovation;cov=(1-gain)*cov
        means.append(mean);covs.append(cov)
    return logvalue,means,covs
for dtype in (tf.float64,tf.float32):
    for replicate in range(4):
        dataset=json.loads((data/f'LG1-calibration-{replicate:02}.json').read_text())
        target=make_linear_fixture_target(dataset,256,controls,dtype=dtype)
        # Filtering trace and the normal endpoint share the same canonical call.
        fn=target.compile(1,filter_moments=True)
        ordinary=target.compile(1)
        records=[]
        obs=target.observations.numpy().tolist()
        for phi in (-.9,-.6,-.3,0.,.3,.6,.9):
            started=time.monotonic();value,score,status=fn(tf.constant([[phi]],dtype))
            raw=float(value[0]);duration=time.monotonic()-started
            exact,mean,cov=kalman(float(tf.constant(phi,dtype)),obs)
            value_ok=bool(status['value_valid'][0]);force_ok=bool(status['force_valid'][0])
            fm=status['filtering_mean'][0,:,0].numpy().tolist()
            fc=status['filtering_covariance'][0,:,0,0].numpy().tolist()
            mean_errors=[abs(a-b)/math.sqrt(c) for a,b,c in zip(fm,mean,cov)]
            cov_errors=[abs(a/b-1) for a,b in zip(fc,cov)]
            records.append(dict(phi=phi,log_posterior=raw,log_likelihood=raw+math.log(1.9),
                exact_log_likelihood=exact,score=float(score[0,0]),value_valid=value_ok,force_valid=force_ok,
                max_whitened_mean_error=max(mean_errors),max_whitened_cov_error=max(cov_errors),
                mean_errors=mean_errors,cov_errors=cov_errors,filtering_mean=fm,filtering_covariance=fc,
                reference_mean=mean,reference_covariance=cov,seconds=duration,
                failure={k:v.numpy().tolist() for k,v in status.items() if k.startswith('first_failure') or k.endswith('failure_code')}))
        center=records[3]
        for row in records:
            row['centered_log_ratio_error']=(row['log_likelihood']-center['log_likelihood'])-(row['exact_log_likelihood']-center['exact_log_likelihood'])
        report['calibration'].append(dict(dtype=dtype.name,replicate=replicate,
            target_identity=target.identity,records=records))
        (out/f'target-{dtype.name}-{replicate}.json').write_text(json.dumps(dict(target.manifest),indent=2)+'\n')
        save()
        if replicate==0:
            for phi in (-.9,.3,.9):
                value,score,status=ordinary(tf.constant([[phi]],dtype))
                derivative=float(score[0,0]);steps=[]
                eps=2.**(-52 if dtype==tf.float64 else -23)
                for factor in (.125,.25,.5,1.,2.,4.,8.):
                    h=eps**(1/3)*max(1,abs(phi))*factor
                    plus,_,ps=ordinary(tf.constant([[phi+h]],dtype))
                    minus,_,ms=ordinary(tf.constant([[phi-h]],dtype))
                    fd=float((plus[0]-minus[0])/(2*h))
                    steps.append(dict(h=h,finite_difference=fd,scaled_error=abs(fd-derivative)/max(1.,abs(fd),abs(derivative)),
                        valid=bool(ps['value_valid'][0] & ms['value_valid'][0])))
                report['derivatives'].append(dict(dtype=dtype.name,phi=phi,score=derivative,
                    value_valid=bool(status['value_valid'][0]),force_valid=bool(status['force_valid'][0]),steps=steps))
                save()
            chart=target.compile(1,coordinates='unconstrained')
            for z in (-2.,0.,2.):
                v,s,status=chart(tf.constant([[z]],dtype));h=eps**(1/3)*max(1,abs(z))
                vp=chart(tf.constant([[z+h]],dtype))[0];vm=chart(tf.constant([[z-h]],dtype))[0]
                fd=float((vp[0]-vm[0])/(2*h));derivative=float(s[0,0])
                physical=-.95+1.9/(1+math.exp(-z))
                pv=ordinary(tf.constant([[physical]],dtype))[0]
                independent=float(pv[0])+math.log(1.9)-math.log1p(math.exp(-z))-math.log1p(math.exp(z))
                report['coordinate_derivatives'].append(dict(dtype=dtype.name,z=z,
                    value=float(v[0]),independent_change_of_variables=independent,
                    score=derivative,finite_difference=fd,
                    scaled_error=abs(fd-derivative)/max(1.,abs(fd),abs(derivative)),
                    value_valid=bool(status['value_valid'][0]),force_valid=bool(status['force_valid'][0])))
                save()
report['allocator']=tf.config.experimental.get_memory_info('GPU:0')
report['completed']=True;save()
print(json.dumps(dict(result=str(out/'diagnostic.json'),completed=True,
    max_mean_error=max(row['max_whitened_mean_error'] for group in report['calibration'] for row in group['records']),
    max_cov_error=max(row['max_whitened_cov_error'] for group in report['calibration'] for row in group['records']),
    max_log_ratio_error=max(abs(row['centered_log_ratio_error']) for group in report['calibration'] for row in group['records']),
    derivative_min_errors=[dict(dtype=r['dtype'],phi=r['phi'],minimum=min(s['scaled_error'] for s in r['steps'])) for r in report['derivatives']])))
