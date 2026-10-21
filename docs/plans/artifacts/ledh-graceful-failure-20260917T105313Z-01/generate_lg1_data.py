"""Batch-native CPU data generation for the declared LG1 fixture; no claim inspection."""
import os,json,hashlib,multiprocessing
from pathlib import Path

def worker(task):
    partition,count,outdir=task
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='1')
    import tensorflow as tf
    from bayesfilter.highdim.ledh_program_fixtures_tf import fixture_seed
    dtype=tf.float64;t=10
    seeds=[{str(p):fixture_seed('LG1',t,partition,1,i,p) for p in (1,2,3)} for i in range(count)]
    initial=tf.stack([tf.random.stateless_normal([1],s['1'],dtype=dtype) for s in seeds])
    process=tf.stack([tf.random.stateless_normal([t,1],s['2'],dtype=dtype) for s in seeds],axis=1)
    observation=tf.stack([tf.random.stateless_normal([t,1],s['3'],dtype=dtype) for s in seeds],axis=1)
    @tf.function(input_signature=[tf.TensorSpec([count,1],dtype),tf.TensorSpec([t,count,1],dtype),tf.TensorSpec([t,count,1],dtype)],jit_compile=True)
    def generate(z,e,v):
        x=z/tf.sqrt(tf.constant(1-.8**2,dtype))
        states=tf.TensorArray(dtype,size=t);obs=tf.TensorArray(dtype,size=t)
        def body(i,x,states,obs):
            x=.8*x+e[i]; y=x+tf.sqrt(tf.constant(.5,dtype))*v[i]
            return i+1,x,states.write(i,x),obs.write(i,y)
        _,_,states,obs=tf.while_loop(lambda i,*_:i<t,body,(0,x,states,obs))
        return tf.transpose(states.stack(),[1,0,2]),tf.transpose(obs.stack(),[1,0,2])
    states,observations=generate(initial,process,observation)
    records=[]
    for i in range(count):
        payload=dict(model_id='LG1',partition=partition,attempt=1,replicate=i,horizon=t,
            classification='synthetic_fixture',gpu_intentionally_hidden=True,
            generation='CPU batch-native TensorFlow/XLA; 3 partition workers',
            master_seed=[20260917,917],seeds={p:s.numpy().tolist() for p,s in seeds[i].items()},
            initial_state=(initial[i]/tf.sqrt(tf.constant(1-.8**2,dtype))).numpy().tolist(),
            states=states[i].numpy().tolist(),observations=observations[i].numpy().tolist(),
            truth=[.8],physical_prior=[[-.95,.95]],transition_before_observation=True)
        path=Path(outdir)/f'LG1-{partition}-{i:02}.json'
        text=json.dumps(payload,indent=2)+'\n'
        with path.open('x') as f:f.write(text)
        records.append(dict(path=str(path),sha256=hashlib.sha256(text.encode()).hexdigest(),
            partition=partition,replicate=i))
    return records

if __name__=='__main__':
    out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])/'data';out.mkdir(exist_ok=False)
    with multiprocessing.get_context('spawn').Pool(3) as pool:
        groups=pool.map(worker,[(p,c,str(out)) for p,c in [('calibration',4),('validation',2),('claim',3)]])
    records=sum(groups,[])
    (out/'index.json').write_text(json.dumps(dict(model_id='LG1',cpu_workers=3,datasets=records),indent=2)+'\n')
    print(json.dumps(dict(dataset_count=len(records),index=str(out/'index.json'))))
