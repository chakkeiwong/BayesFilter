"""GPU/XLA native KDM endpoint diagnostic; no training or algorithm changes."""
import argparse,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,allow_nan=False)+chr(10));tmp.replace(p)
def validate_recent_scope(inputs,check):
    """Validate the declared diagnostic model set before importing TensorFlow."""
    unit=inputs['unit'];scope=inputs.get('evaluation_scope','recent_checkpoints')
    check('recent unit',unit in (1,2))
    check('known evaluation scope',scope in ('recent_checkpoints','unit02_mdpf_initialization'))
    if scope=='unit02_mdpf_initialization':
        check('intervention unit',unit==2)
        expected={('mdpf',name) for name in ('source_trained','tf_trained','author_start_tf_trained')}
    else:
        expected={(variant,name) for variant in ('mdpf','am_dpf') for name in ('source_trained','tf_trained')}
        if unit==1:expected.add(('mdpf','author_start_tf_trained'))
    actual=[(arm['variant'],arm['checkpoint']) for arm in inputs['arms']]
    check('declared checkpoint count',len(actual)==len(expected))
    check('declared checkpoint identities',set(actual)==expected)
def main(out):
    started=time.monotonic();checks=[]
    result=dict(status='RUNNING',checks=checks,records=[],metric_replays=[],replication='INCOMPLETE',
        inference='DESCRIPTIVE_DEVELOPMENT_DIAGNOSTIC',rows=256,horizon=150,particles=25,batch=64,
        filter_seeds=[202610082,202610083],input_sha256=sha(out/'inputs.json'))
    def check(name,value):
        okay=bool(value);checks.append(dict(name=name,passed=okay))
        if not okay: raise RuntimeError(name)
    try:
        inputs=json.loads((out/'inputs.json').read_text())
        check('input schema',inputs['schema'] in ('kdm_native_evaluation_inputs_v1','kdm_native_evaluation_inputs_v2'))
        if inputs['schema']=='kdm_native_evaluation_inputs_v1':
            pilot=ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928/repaired-training-20261011/training-pilot-02'
            check('terminal pilot identity',inputs['pilot_manifest_sha256']==sha(pilot/'manifest.json'))
            check('common evaluator identity',inputs['common_report_sha256']==sha(pilot/'evaluation/diagnostic/result.json'))
        else:
            validate_recent_scope(inputs,check)
            check('recent input provenance present',len(inputs['provenance'])>=10)
            for name,digest in inputs['provenance'].items():
                check('input provenance '+name,sha(Path(name))==digest)
        labels=[(arm['variant'],arm['checkpoint']) for arm in inputs['arms']]
        check('distinct checkpoints',len(set(labels))==len(labels))
        result['checkpoint_count']=len(inputs['arms'])
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
        from bayesfilter.replication.kdm_source_stages_tf import SourceInitialCloud
        from bayesfilter.replication.kdm_normal_von_mises_tf import NormalVonMisesKDM
        result.update(tensorflow_version=tf.__version__,memory_policy=memory,jit_compile=True,tf32=False,
            visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),random_backend='native stateless TF Philox',
            call_chain=['SourceInitialCloud.sample','BearingsNeuralReplay.native_evaluate','NormalVonMisesKDM._posterior'])
        with tf.device('/GPU:0'):
            states=tf.constant(inputs['data']['states'],tf.float32);observations=tf.constant(inputs['data']['observations'],tf.float32)
            check('full data shapes',states.shape==[256,150,3] and observations.shape==[256,150,1])
            kde=NormalVonMisesKDM(batch_size=64*150,particles=25,circular_dimensions=[False,False,True],dtype=tf.float32,jit_compile=True)
            specs=[tf.TensorSpec([64,150,25,3],tf.float32),tf.TensorSpec([64,150,25],tf.float32),
                   tf.TensorSpec([64,150,3],tf.float32),tf.TensorSpec([64,150,3],tf.float32)]
            def calculate(p,w,b,s):
                post=kde._posterior(tf.reshape(p,[64*150,25,3]),tf.reshape(w,[64*150,25]),
                                    tf.reshape(b,[64*150,3]),tf.reshape(s,[64*150,3]))
                xy=tf.reduce_sum(p[...,:2]*w[...,None],axis=2)
                angle=tf.atan2(tf.reduce_sum(tf.sin(p[...,2])*w,axis=2),tf.reduce_sum(tf.cos(p[...,2])*w,axis=2))
                da=tf.math.floormod(angle-s[...,2]+tf.constant(math.pi,tf.float32),tf.constant(2*math.pi,tf.float32))-tf.constant(math.pi,tf.float32)
                error=tf.reduce_sum(tf.square(xy-s[...,:2]),axis=-1)+tf.square(da)
                return dict(nll=tf.reshape(post['nll'],[64,150]),squared_error=error,
                            valid=post['valid'] & tf.reduce_all(tf.math.is_finite(error)))
            metrics=tf.function(calculate,input_signature=specs,jit_compile=True,autograph=False)
            true_bearing=tf.atan2(states[...,1],states[...,0]+5.)
            residual=observations[...,0]-true_bearing
            residual=tf.abs(tf.atan2(tf.sin(residual),tf.cos(residual)))
            masks={'first_17':tf.broadcast_to(tf.range(150)[None,:]<17,[256,150]),
                   'after_17':tf.broadcast_to(tf.range(150)[None,:]>=17,[256,150]),
                   'bearing_residual_le_0p5':residual<=.5,'bearing_residual_gt_0p5':residual>.5,
                   'angle_seam_0p1':tf.abs(states[...,2])>=math.pi-.1}
            for arm in inputs['arms']:
                label=arm['variant']+'/'+arm['checkpoint']
                model=BearingsNeuralReplay(arm['fixture'],batch_size=64,particles=25,horizon=150,jit_compile=True)
                model.load_state_dict(arm['state']);initial=SourceInitialCloud(model)
                before=model.state_dict()
                for seed in result['filter_seeds']:
                    tick=time.monotonic();nlls=[];errors=[]
                    for begin in range(0,256,64):
                        write(out/'progress.json',dict(stage=label,seed=seed,rows_done=begin,elapsed_seconds=time.monotonic()-started))
                        s=states[begin:begin+64];o=observations[begin:begin+64]
                        init=initial.sample(s[:,0],tf.constant([seed,begin],tf.int64))
                        check(label+'/initial valid',init['valid'].numpy())
                        args=[init[k] for k in ('particles','particle_weights','bandwidths','resampling_particle_weights','resampling_bandwidths')]
                        filtered=model.native_evaluate(*args,o,s,tf.ones([64,150],tf.bool),tf.constant([seed+1,begin],tf.int64))
                        check(label+'/endpoint valid',filtered['valid'].numpy())
                        a=filtered['outputs']
                        check(label+'/time coverage',a[0].shape==[149,64,25,3])
                        check(label+'/GPU output', 'GPU' in a[0].device)
                        for j,value in enumerate(a): check(label+'/finite output '+str(j),tf.reduce_all(tf.math.is_finite(value)).numpy())
                        p=tf.concat([init['particles'][:,None],tf.transpose(a[0],[1,0,2,3])],axis=1)
                        w=tf.concat([init['particle_weights'][:,None],tf.transpose(a[1],[1,0,2])],axis=1)
                        b=tf.concat([init['bandwidths'][:,None],tf.transpose(a[2],[1,0,2])],axis=1)
                        for j in (1,3):
                            check(label+'/probability '+str(j),tf.reduce_all(a[j]>=0).numpy() and tf.reduce_all(tf.abs(tf.reduce_sum(a[j],-1)-1.)<=1e-5).numpy())
                        for j in (2,4): check(label+'/positive bandwidth '+str(j),tf.reduce_all(a[j]>0).numpy())
                        computed=metrics(p,w,b,s);check(label+'/metric valid',computed['valid'].numpy())
                        nlls.append(computed['nll']);errors.append(computed['squared_error'])
                        if seed==result['filter_seeds'][0] and begin==0:
                            small_nll=computed['nll'][:2];small_error=computed['squared_error'][:2]
                            payload=dict(variant=arm['variant'],checkpoint=arm['checkpoint'],states=s[:2].numpy().tolist(),
                                particles=p[:2].numpy().tolist(),particle_weights=w[:2].numpy().tolist(),bandwidths=b[:2].numpy().tolist(),
                                nll=small_nll.numpy().tolist(),squared_error=small_error.numpy().tolist(),
                                aggregate=dict(nll=float(tf.reduce_mean(small_nll).numpy()),
                                  rmse=float(tf.reduce_mean(tf.sqrt(tf.reduce_mean(small_error,axis=1))).numpy())))
                            path=out/(arm['variant']+'-'+arm['checkpoint']+'-metric-replay.json');write(path,payload)
                            result['metric_replays'].append(dict(path=path.name,sha256=sha(path)))
                    nll=tf.concat(nlls,0);error=tf.concat(errors,0)
                    check(label+'/complete metric coverage',nll.shape==[256,150] and error.shape==[256,150])
                    sequence_nll=tf.reduce_mean(nll,axis=1);sequence_rmse=tf.sqrt(tf.reduce_mean(error,axis=1))
                    conditional={}
                    for name,mask in masks.items():
                        count=tf.reduce_sum(tf.cast(mask,tf.float32),axis=1);keep=count>0
                        if not bool(tf.reduce_any(keep).numpy()):
                            conditional[name]=dict(status='UNAVAILABLE',states=0,trajectories=0);continue
                        cn=tf.boolean_mask(tf.reduce_sum(nll*tf.cast(mask,tf.float32),axis=1),keep)/tf.boolean_mask(count,keep)
                        ce=tf.boolean_mask(tf.reduce_sum(error*tf.cast(mask,tf.float32),axis=1),keep)/tf.boolean_mask(count,keep)
                        conditional[name]=dict(nll=float(tf.reduce_mean(cn).numpy()),rmse=float(tf.reduce_mean(tf.sqrt(ce)).numpy()),
                            states=int(tf.reduce_sum(count).numpy()),trajectories=int(tf.reduce_sum(tf.cast(keep,tf.int32)).numpy()))
                    file=out/(arm['variant']+'-'+arm['checkpoint']+'-'+str(seed)+'-metrics.json')
                    write(file,dict(nll=nll.numpy().tolist(),squared_error=error.numpy().tolist(),
                        sequence_nll=sequence_nll.numpy().tolist(),sequence_rmse=sequence_rmse.numpy().tolist()))
                    row=dict(variant=arm['variant'],checkpoint=arm['checkpoint'],seed=seed,
                        nll=float(tf.reduce_mean(sequence_nll).numpy()),rmse=float(tf.reduce_mean(sequence_rmse).numpy()),
                        conditional=conditional,metrics_path=file.name,metrics_sha256=sha(file),wall_seconds=time.monotonic()-tick)
                    result['records'].append(row);write(out/'partial.json',result)
                check(label+'/checkpoint immutable',model.state_dict()==before)
                check(label+'/single initializer trace',initial.sample.experimental_get_tracing_count()==1)
                check(label+'/single filter trace',model.native_evaluate.experimental_get_tracing_count()==1)
            check('single metric trace',metrics.experimental_get_tracing_count()==1)
            check('all arms completed',len(result['records'])==2*len(inputs['arms']) and len(result['metric_replays'])==len(inputs['arms']))
            result['allocator']=tf.config.experimental.get_memory_info('GPU:0')
        result['status']='PASS'
    except BaseException:
        result['status']='FAIL';result['error']=traceback.format_exc()
    result['wall_seconds']=time.monotonic()-started;write(out/'native-result.json',result)
    print(json.dumps(dict(status=result['status'],checks=len(checks),records=len(result['records']),wall_seconds=result['wall_seconds'])))
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.output))
