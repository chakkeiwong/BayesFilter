"""Predeclared LG1 numerical-control curves on calibration data only."""
import json,math,os,sys,time,gc
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
root=Path(sys.argv[1]);out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
previous=json.loads((root/'lg1-calibration-diagnostic-01/diagnostic.json').read_text())
baseline=previous['controls'];dtype=tf.float32
# Full features remain enabled; coarse and fine integration investigate
# whether proposal concentration, rather than derivative error, explains tails.
candidates=[('N4096-warm',4096,{}),('N4096-flow-3',4096,dict(substeps=3)),
            ('N4096-flow-1',4096,dict(substeps=1)),('N4096-flow-40',4096,dict(substeps=40))]
report=dict(classification='calibration_curves_not_admission',memory_policy=memory,
    candidate_list=[dict(name=k,n=n,controls=dict(baseline,**changes)) for k,n,changes in candidates],
    parameter_design=[-.9,0.,.9],dataset_replicates=[0,2],curves=[])
(out/'frozen-design.json').write_text(json.dumps(report,indent=2)+'\n')
def save(): (out/'curves.json').write_text(json.dumps(report,indent=2)+'\n')
for name,n,changes in candidates:
    controls=dict(baseline,**changes)
    for replicate in (0,2):
        dataset=json.loads((root/f'lg1-data-01/data/LG1-calibration-{replicate:02}.json').read_text())
        target=make_linear_fixture_target(dataset,n,controls,dtype=dtype)
        fn=target.compile(1,filter_moments=True)
        ref=next(g for g in previous['calibration'] if g['dtype']=='float32' and g['replicate']==replicate)
        rows=[]
        for phi in report['parameter_design']:
            reference=next(r for r in ref['records'] if r['phi']==phi)
            started=time.monotonic();v,s,status=fn(tf.constant([[phi]],dtype));value=float(v[0]);elapsed=time.monotonic()-started
            m=status['filtering_mean'][0,:,0].numpy().tolist();p=status['filtering_covariance'][0,:,0,0].numpy().tolist()
            rows.append(dict(phi=phi,log_likelihood=value+math.log(1.9),score=float(s[0,0]),
                exact_log_likelihood=reference['exact_log_likelihood'],
                filtering_mean=m,filtering_covariance=p,
                reference_mean=reference['reference_mean'],reference_covariance=reference['reference_covariance'],
                filtering_ess=status['filtering_ess'][0].numpy().tolist(),
                post_reset_mean=status['post_reset_mean'][0,:,0].numpy().tolist(),
                post_reset_covariance=status['post_reset_covariance'][0,:,0,0].numpy().tolist(),
                value_valid=bool(status['value_valid'][0]),force_valid=bool(status['force_valid'][0]),
                max_mean_error=max(abs(a-b)/math.sqrt(c) for a,b,c in zip(m,reference['reference_mean'],reference['reference_covariance'])),
                max_cov_error=max(abs(a/b-1) for a,b in zip(p,reference['reference_covariance'])),seconds=elapsed))
        center=rows[1]
        for row in rows:row['log_ratio_error']=(row['log_likelihood']-center['log_likelihood'])-(row['exact_log_likelihood']-center['exact_log_likelihood'])
        report['curves'].append(dict(name=name,n=n,replicate=replicate,target_identity=target.identity,records=rows))
        save();print(json.dumps(dict(name=name,n=n,replicate=replicate,
            max_mean=max(r['max_mean_error'] for r in rows),max_cov=max(r['max_cov_error'] for r in rows),
            max_log_ratio=max(abs(r['log_ratio_error']) for r in rows))),flush=True)
        del fn,target;gc.collect()
report['complete']=True;report['allocator']=tf.config.experimental.get_memory_info('GPU:0');save()
