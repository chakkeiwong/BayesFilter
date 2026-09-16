"""One bounded GPU integration process, with explicit FP64 derivative references."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path.cwd()))
from bayesfilter.score_study.coordinator import execute,write_json
from bayesfilter.score_study.registry import default_registry

study_path,output=map(Path,sys.argv[1:3])
study=json.loads(study_path.read_text())
start=time.monotonic()
state=execute(study,default_registry(),output)
checks=[]
if state['execution_status']=='complete':
    import tensorflow as tf
    from bayesfilter.score_study.gaussian_tf import make_data_kernel
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    settings=study['settings'];d=settings['dimension'];o=settings['observation_dimension'];T=settings['horizon']
    with tf.device('/GPU:0'):
        theta=tf.constant(settings['theta'],tf.float32)
        for row in study['rows']:
            if row['proposal']!='iapf':continue
            result=json.loads((output/state['rows'][row['id']]['result_path']).read_text())
            diag=result['diagnostics'];fit=diag['fit'];N=fit['particles'];seeds=diag['fit_seed_records']
            obs=make_data_kernel(d,o,T,'float32')(tf.constant(settings['data_theta'],tf.float32),
                                                   tf.constant(diag['data_seed'],tf.int32))
            def draw(name,shape,normal):
                function=tf.random.stateless_normal if normal else tf.random.stateless_uniform
                return function(shape,seeds['iapf_final_'+name],dtype=tf.float32)
            args=(obs,draw('initial',[N,d],True),draw('process',[T,N,d],True),
                  draw('ancestors',[T+1,N],False),draw('mixture',[T,N],False),
                  tf.constant(fit['centers'],tf.float32),tf.constant(fit['covariances'],tf.float32),
                  tf.constant(fit['log_floors'],tf.float32))
            kernel32=make_fitted_twist_kernel(d,o,N,T,'float32')
            kernel64=make_fitted_twist_kernel(d,o,N,T,'float64')
            value32,score32,_=kernel32(theta,*args)
            args64=tuple(tf.cast(x,tf.float64) for x in args);theta64=tf.cast(theta,tf.float64)
            value64,score64,_=kernel64(theta64,*args64)
            score_tolerance=512*2**-23*(1+tf.abs(score64))
            score_error=tf.abs(tf.cast(score32,tf.float64)-score64)
            replay_error=float(tf.reduce_max(tf.abs(score32-tf.constant(result['score'],tf.float32))))
            entry={'row':row['id'],'particles':N,'fit_digest':diag['fit_digest'],
                   'kernel_calls':14,'replay_max_score_error':replay_error,
                   'fp32_max_score_error':float(tf.reduce_max(score_error)),
                   'fp32_scaled_error':float(tf.reduce_max(score_error/score_tolerance)),
                   'value_difference':float(tf.cast(value32,tf.float64)-value64),'directions':[]}
            for j,direction in enumerate(tf.unstack(tf.eye(6,dtype=tf.float64))):
                h=tf.constant(1e-6,tf.float64)
                fd=(kernel64(theta64+h*direction,*args64)[0]-kernel64(theta64-h*direction,*args64)[0])/(2*h)
                error=float(tf.abs(fd-score64[j]));tolerance=1e-6*(1+abs(float(score64[j])))
                entry['directions'].append({'coordinate':j,'analytical':float(score64[j]),
                    'finite_difference':float(fd),'absolute_error':error,'tolerance':tolerance,
                    'pass':error<=tolerance})
            entry['pass']=(replay_error==0 and bool(tf.reduce_all(score_error<=score_tolerance))
                           and all(x['pass'] for x in entry['directions']))
            checks.append(entry)
verdict={'engineering_pass':state['execution_status']=='complete' and len(checks)==2 and all(c['pass'] for c in checks),
         'classification':'GPU FP32/TF32/XLA mechanics with GPU FP64 derivative reference; no quality claim',
         'state_status':state['execution_status'],'checks':checks,'wall_seconds':time.monotonic()-start,
         'plan':study['plan'],'command':sys.argv,'source_root':str(Path.cwd())}
write_json(output/'integration_checks.json',verdict)
print(json.dumps({'engineering_pass':verdict['engineering_pass'],'rows':len(state['rows']),
                  'direction_checks':sum(len(x['directions']) for x in checks),'wall_seconds':verdict['wall_seconds']}))
sys.exit(0 if verdict['engineering_pass'] else 2)
