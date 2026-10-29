"""Diagnostic only: composed native pretraining cloud law; no runtime changes."""
import argparse,json,math,os,subprocess,sys,tempfile,time
from pathlib import Path
from multiprocessing.connection import Listener
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
from diagnose_kdm_fresh_pretraining import read,write,save,digest
SEED=202610134

def main(out):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceWeightPretraining
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def host(x):return x.numpy().tolist()
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    with tf.device('/GPU:0'):probe=tf.matmul(tf.ones([2,2]),tf.eye(2))
    check('device','GPU' in probe.device)
    k=tf.constant(1/(.1+1e-8),tf.float64)
    A=float(tf.math.bessel_i1e(k)/tf.math.bessel_i0e(k))
    expected={}
    for axis in ('x','y'):
        expected.update({f'center_{axis}_mean':0.,f'center_{axis}_second':4.,
            f'conditional_{axis}_mean':0.,f'conditional_{axis}_second':9.,
            f'marginal_{axis}_mean':0.,f'marginal_{axis}_second':13.,
            f'within_{axis}_product':4.,f'center_conditional_{axis}_product':0.})
    expected.update(center_xy_product=0.,conditional_xy_product=0.,marginal_xy_product=0.,
        center_angle_cos=A,center_angle_sin=0.,conditional_angle_cos=A,conditional_angle_sin=0.,
        marginal_angle_cos=A*A,marginal_angle_sin=0.,within_angle_cos=A*A)
    names=list(expected)
    def moments(truth,center,particles):
        # Reporting-only FP64 reductions. Each returned row is one cloud cluster.
        with tf.device('/CPU:0'):
            truth,center,particles=[tf.cast(v,tf.float64) for v in (truth,center,particles)]
            c=center-truth;u=particles-center[:,None,:];p=particles-truth[:,None,:]
            mean=lambda v:tf.reduce_mean(v,axis=1)
            vals={}
            for j,axis in enumerate(('x','y')):
                cm=c[:,j];um=u[:,:,j];pm=p[:,:,j]
                # All distinct ordered pairs, calculated without an N-by-N array.
                pairs=(tf.square(tf.reduce_sum(pm,1))-tf.reduce_sum(tf.square(pm),1))/(50.*49.)
                vals.update({f'center_{axis}_mean':cm,f'center_{axis}_second':cm*cm,
                    f'conditional_{axis}_mean':mean(um),f'conditional_{axis}_second':mean(um*um),
                    f'marginal_{axis}_mean':mean(pm),f'marginal_{axis}_second':mean(pm*pm),
                    f'within_{axis}_product':pairs,f'center_conditional_{axis}_product':cm*mean(um)})
            vals.update(center_xy_product=c[:,0]*c[:,1],conditional_xy_product=mean(u[:,:,0]*u[:,:,1]),
                        marginal_xy_product=mean(p[:,:,0]*p[:,:,1]))
            for label,angle in [('center',c[:,2]),('conditional',u[:,:,2]),('marginal',p[:,:,2])]:
                vals[label+'_angle_cos']=tf.cos(angle) if label=='center' else mean(tf.cos(angle))
                vals[label+'_angle_sin']=tf.sin(angle) if label=='center' else mean(tf.sin(angle))
            angles=particles[:,:,2]
            vals['within_angle_cos']=(tf.square(tf.reduce_sum(tf.cos(angles),1))+tf.square(tf.reduce_sum(tf.sin(angles),1))-50.)/(50.*49.)
            return tf.stack([vals[n] for n in names],axis=1)
    tmp=tempfile.TemporaryDirectory(prefix='kdm-cloud-law-',dir='/tmp')
    socket=str(Path(tmp.name)/'source.sock');listener=Listener(socket,family='AF_UNIX')
    listener._listener._socket.settimeout(90)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',LD_LIBRARY_PATH='/home/chakwong/anaconda3/envs/tf-gpu2/lib')
    log=(out/'author.log').open('w')
    command=['/home/chakwong/anaconda3/envs/tf-gpu2/bin/python','-I','-B',str(ROOT/'docs/benchmarks/diagnose_kdm_fresh_pretraining.py'),'--socket',socket]
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    conn=None
    def receive():
        assert conn.poll(120),'Author timeout'
        r=conn.recv()
        if r['type']=='error':raise RuntimeError(r['error'])
        return r
    try:
        conn=listener.accept();ready=receive()
        conn.send(dict(op='setup',unit='unit01',dtype='float32',checkpoint='constructor'))
        setup=receive();save(out/'setup.json.gz',setup)
        with tf.device('/GPU:0'):model=BearingsNeuralReplay(setup['fixture'],batch_size=256)
        stages={};all_moments={b:[] for b in ('author','tf')};cases=[]
        state=model.state_dict()
        for repeat in range(4):
            rows=[]
            for index in range(4):
                source_seed=SEED+repeat*100+index
                conn.send(dict(op='cloud_law',index=index,seed=source_seed,control=repeat==0 and index==0))
                r=receive();rows+=r['rows'];b=len(r['rows'])
                label=f'pass{repeat:02d}-batch{index:02d}'
                check(label+'/source_unchanged',r['unchanged'])
                check(label+'/effective',r['effective_batch']==b*5)
                if repeat==0 and index==0:check('author_instrumentation_control',r['instrumentation_control'])
                with tf.device('/GPU:0'):
                    if b not in stages:stages[b]=SourceWeightPretraining(model,b)
                    data=r['data'];inputs=[tf.constant(data[n],tf.bool if n=='ground_truth_mask' else tf.float32)
                        for n in ('states','observations','ground_truth_mask')]
                    result=stages[b].native_evaluate(*inputs,tf.constant([SEED,repeat*100+index],tf.int64))
                    o=result['outputs']
                    check(label+'/tf_valid',result['valid'])
                    check(label+'/tf_unchanged',state==model.state_dict())
                    check(label+'/truth_selection',host(o['selected_states'])==r['truth'])
                    native=[o[n] for n in ('selected_states','cloud_centers','particles')]
                    author=[tf.constant(r[n],tf.float32) for n in ('truth','centers','particles')]
                    transform_delta=float(tf.reduce_max(tf.abs(author[2]-tf.constant(r['sampled_particles'],tf.float32))))
                    check(label+'/author_consumer_cloud',transform_delta<=5e-6)
                block={}
                for backend,triplet in [('author',author),('tf',native)]:
                    m=moments(*triplet)
                    check(label+'/'+backend+'_finite',tf.reduce_all(tf.math.is_finite(m)))
                    all_moments[backend].append(m);block[backend]=host(m)
                save(out/(label+'-moments.json.gz'),dict(names=names,clusters=block,source_seed=source_seed,
                    tf_seed=[SEED,repeat*100+index],input_hash=digest(data),rows=r['rows']))
                cases.append(dict(pass_index=repeat,batch=index,sequences=b,clouds=b*5,source_seed=source_seed,
                    tf_seed=[SEED,repeat*100+index],author_loss=r['loss'],tf_loss=float(result['loss']),
                    max_sampling_iterations=host(o['sampling_iterations']),source_input_hash=digest(data)))
                print(json.dumps(dict(case=label,status='OK')),flush=True)
            check(f'pass{repeat}/coverage',rows==list(range(1000)))
        estimates={};flags=[]
        for backend,blocks in all_moments.items():
            with tf.device('/CPU:0'):
                matrix=tf.concat(blocks,axis=0);n=int(matrix.shape[0]);check(backend+'/cloud_count',n==20000)
                avg=tf.reduce_mean(matrix,0)
                se=tf.sqrt(tf.reduce_sum(tf.square(matrix-avg[None,:]),0)/(n*(n-1.)))
            estimates[backend]={}
            for j,name in enumerate(names):
                mu,s=float(avg[j]),float(se[j]);delta=mu-expected[name];ok=abs(delta)<=6*s+5e-6
                estimates[backend][name]=dict(mean=mu,se=s,expected=expected[name],difference=delta,
                    z=delta/s,passed=ok)
                if not ok:flags.append(backend+'/'+name)
        between={}
        for name in names:
            a,t=estimates['author'][name],estimates['tf'][name]
            se=math.hypot(a['se'],t['se']);delta=t['mean']-a['mean'];ok=abs(delta)<=6*se+1e-5
            between[name]=dict(difference=delta,se=se,z=delta/se,passed=ok)
            if not ok:flags.append('between/'+name)
        traces={str(b):st.native_evaluate.experimental_get_tracing_count() for b,st in stages.items()}
        check('stable_traces',all(n==1 for n in traces.values()))
        conn.send(dict(op='stop'));closure=receive();write(out/'source-closure.json',closure)
        proc.wait(timeout=30);check('source_exit',proc.returncode==0)
        write(out/'result.json',dict(status='PASS' if not flags else 'LOCALIZATION_REQUIRED',checks=checks,
            estimates=estimates,between=between,flags=flags,cases=cases,cluster_count=20000,particles_per_backend=1000000,
            moments=len(names),trace_counts=traces,seed=SEED,wall_seconds=time.monotonic()-start,
            scope='Native composed pretraining cloud law only; whole KDM remains incomplete',
            environment=dict(source=ready,source_command=command,tensorflow=tf.__version__,memory_growth=memory,
                gpu=[d.name for d in tf.config.list_logical_devices('GPU')],probe_device=probe.device,
                allocator=tf.config.experimental.get_memory_info('GPU:0'),tf32=False,jit_compile=True,
                reporting='CPU FP64 diagnostic reductions only'),whole_kdm_replication='INCOMPLETE'))
    finally:
        if conn is not None:conn.close()
        listener.close()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close();tmp.cleanup()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    main(p.parse_args().output)
