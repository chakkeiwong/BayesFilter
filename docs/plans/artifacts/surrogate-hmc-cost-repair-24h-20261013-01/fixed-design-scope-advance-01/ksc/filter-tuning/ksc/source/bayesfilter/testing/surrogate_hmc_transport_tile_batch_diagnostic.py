"""Diagnostic-only exact-tile scheduling comparison; no runtime admission."""
import hashlib
import json
from pathlib import Path
import time
import tensorflow as tf


def _inputs(n, k, b, p, d, dtype):
    if n % k: raise ValueError("exact tile divisor required")
    x=tf.random.stateless_normal([b,n,d],[167,31],dtype=dtype)
    tangent=tf.random.stateless_normal([p,b,n,d],[168,31],dtype=dtype)*.1
    potential=tf.random.stateless_normal([b,n],[169,31],dtype=dtype)*.4
    potential_t=tf.random.stateless_normal([p,b,n],[170,31],dtype=dtype)*.1
    delta=x[:,:,None,:]-x[:,None,:,:]
    cost=-tf.reduce_sum(delta*delta,-1)/tf.cast(max(1,d),dtype)
    return x,tangent,potential,potential_t,cost


def serial_reduce(x,tangent,potential,potential_t,cost,k,rows=True):
    """The current nested-tile weighted analytic derivative, isolated."""
    b,n,d=x.shape; p=tangent.shape[0]; blocks=n//k; dtype=x.dtype
    centred_x=x-x[:,:1]
    centred_t=tangent-tangent[:,:,:1]
    radial=tf.reduce_sum(centred_x*centred_t,-1)
    xb=tf.reshape(centred_x,[b,blocks,k,d])
    tb=tf.reshape(centred_t,[p,b,blocks,k,d])
    rb=tf.reshape(radial,[p,b,blocks,k])
    vb=tf.reshape(potential,[b,blocks,k])
    vtb=tf.reshape(potential_t,[p,b,blocks,k])
    cb=tf.reshape(cost,[b,blocks,k,blocks,k])
    out=tf.TensorArray(dtype,blocks,element_shape=[b,k])
    dout=tf.TensorArray(dtype,blocks,element_shape=[p,b,k])
    def outer(i,out,dout):
        xi=tf.gather(xb,i,axis=1)
        di=tf.gather(tb,i,axis=2)
        ai=tf.gather(rb,i,axis=2)
        def inner(j,total,dtotal):
            part=tf.gather(vb,j,axis=1)
            part_t=tf.gather(vtb,j,axis=2)
            xj=tf.gather(xb,j,axis=1)
            dj=tf.gather(tb,j,axis=2)
            aj=tf.gather(rb,j,axis=2)
            if rows:
                c=tf.gather(tf.gather(cb,i,axis=1),j,axis=2)
            else:
                c=tf.linalg.matrix_transpose(tf.gather(tf.gather(cb,j,axis=1),i,axis=2))
            logits=c+part[:,None,:]
            primal=tf.reduce_logsumexp(logits,-1)
            weights=tf.nn.softmax(logits,-1)
            packed=tf.concat([tf.transpose(part_t,[1,2,0]),xj,
                tf.reshape(tf.transpose(dj,[1,2,0,3]),[b,k,p*d]),
                tf.transpose(aj,[1,2,0])],-1)
            averaged=tf.linalg.matmul(weights,packed)
            da=tf.transpose(averaged[:,:,:p],[2,0,1])
            mx=averaged[:,:,p:p+d]
            mdx=tf.transpose(tf.reshape(averaged[:,:,p+d:p+d+p*d],[b,k,p,d]),[2,0,1,3])
            ma=tf.transpose(averaged[:,:,p+d+p*d:],[2,0,1])
            derivative=da-(2/tf.cast(max(1,d),dtype))*(ai+ma-tf.reduce_sum(di*mx[None],-1)
                               -tf.reduce_sum(xi[None]*mdx,-1))
            both=tf.stack([total,primal],-1)
            next_t=tf.reduce_sum(tf.nn.softmax(both,-1)[None]*tf.stack([dtotal,derivative],-1),-1)
            return j+1,tf.reduce_logsumexp(both,-1),next_t
        _,lv,lt=tf.while_loop(lambda j,*_:j<blocks,inner,
            (0,tf.fill([b,k],tf.constant(-float("inf"),dtype)),tf.zeros([p,b,k],dtype)),
            parallel_iterations=1)
        return i+1,out.write(i,lv),dout.write(i,lt)
    _,out,dout=tf.while_loop(lambda i,*_:i<blocks,outer,(0,out,dout),parallel_iterations=1)
    return (tf.reshape(tf.transpose(out.stack(),[1,0,2]),[b,n]),
            tf.reshape(tf.transpose(dout.stack(),[1,2,0,3]),[p,b,n]))


def batched_reduce(x,tangent,potential,potential_t,cost,k,rows=True):
    """Same tile equations with row and column blocks as batch axes."""
    b,n,d=x.shape; p=tangent.shape[0]; blocks=n//k
    centred_x=x-x[:,:1]
    centred_t=tangent-tangent[:,:,:1]
    radial=tf.reduce_sum(centred_x*centred_t,-1)
    xb=tf.transpose(tf.reshape(centred_x,[b,blocks,k,d]),[1,0,2,3])
    tb=tf.transpose(tf.reshape(centred_t,[p,b,blocks,k,d]),[2,1,3,0,4])
    rb=tf.transpose(tf.reshape(radial,[p,b,blocks,k]),[2,1,3,0])
    vb=tf.transpose(tf.reshape(potential,[b,blocks,k]),[1,0,2])
    vtb=tf.transpose(tf.reshape(potential_t,[p,b,blocks,k]),[2,1,3,0])
    cb=tf.reshape(cost,[b,blocks,k,blocks,k])
    c=tf.transpose(cb,[1,3,0,2,4] if rows else [3,1,0,4,2])
    logits=c+vb[None,:, :,None,:]
    primal=tf.reduce_logsumexp(logits,-1)
    weights=tf.nn.softmax(logits,-1)
    packed=tf.concat([vtb,xb,tf.reshape(tb,[blocks,b,k,p*d]),rb],-1)
    averaged=tf.linalg.matmul(weights,packed[None])
    da=averaged[:,:,:,:,:p]
    mx=averaged[:,:,:,:,p:p+d]
    mdx=tf.reshape(averaged[:,:,:,:,p+d:p+d+p*d],[blocks,blocks,b,k,p,d])
    ma=averaged[:,:,:,:,p+d+p*d:]
    derivative=da-(2/tf.cast(max(1,d),x.dtype))*(rb[:,None]+ma-tf.reduce_sum(tb[:,None]*mx[:,:,:,:,None,:],-1)
                     -tf.reduce_sum(xb[:,None,:,: ,None,:]*mdx,-1))
    merge_weights=tf.nn.softmax(primal,axis=1)
    value=tf.reduce_logsumexp(primal,axis=1)
    score=tf.reduce_sum(merge_weights[:,:,:,:,None]*derivative,axis=1)
    return (tf.reshape(tf.transpose(value,[1,0,2]),[b,n]),
            tf.reshape(tf.transpose(score,[3,1,0,2]),[p,b,n]))


def run_diagnostic(case, output, repetitions=5):
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    from bayesfilter.testing.surrogate_hmc_five_model import json_ready
    output=Path(output)
    n=int(case.fixed[0].shape[0]);k=select_transport_chunks(n).row_chunk_size
    b,p,d=2,case.p,case.spec.dimension; dtype=tf.float32
    args=_inputs(n,k,b,p,d,dtype)
    signature=[tf.TensorSpec(a.shape,a.dtype) for a in args]
    @tf.function(input_signature=signature,jit_compile=True,autograph=False)
    def baseline(*values):
        return (*serial_reduce(*values,k,True),*serial_reduce(*values,k,False))
    @tf.function(input_signature=signature,jit_compile=True,autograph=False)
    def candidate(*values):
        return (*batched_reduce(*values,k,True),*batched_reduce(*values,k,False))
    def timed(fn):
        start=time.monotonic();result=fn(*args)
        for tensor in result: tensor.numpy()
        return result,time.monotonic()-start
    record=dict(status="RUNNING",evidence_role="synthetic_exact_tile_schedule_nomination_only",
        model=case.name,particles=n,tile=k,batch=b,parameters=p,state_dimension=d,
        dtype=dtype.name,tf32=False,jit_compile=True,repetitions=repetitions,
        seeds=[[167,31],[168,31],[169,31],[170,31]],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=[hashlib.sha256(tf.io.serialize_tensor(a).numpy()).hexdigest() for a in args],
        cold_seconds={},warmed_seconds={"baseline":[],"candidate":[]},
        allocator_by_arm={"baseline":[],"candidate":[]},posterior_sampling_complete=False,
        canonical_admission=False)
    initial={}
    for name,fn in (("baseline",baseline),("candidate",candidate)):
        initial[name],record["cold_seconds"][name]=timed(fn)
    record["maximum_scaled_errors"]=[float(tf.reduce_max(tf.abs(a-r)/tf.maximum(tf.ones_like(r),tf.abs(r))).numpy())
        for a,r in zip(initial["candidate"],initial["baseline"])]
    record["finite"]=all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in initial["candidate"])
    for repeat in range(repetitions):
        order=(("baseline",baseline),("candidate",candidate))
        for name,fn in order[::1 if repeat%2==0 else -1]:
            out,seconds=timed(fn)
            record["warmed_seconds"][name].append(seconds)
            record["allocator_by_arm"][name].append(tf.config.experimental.get_memory_info("GPU:0"))
            for before,after in zip(initial[name],out):tf.debugging.assert_equal(before,after)
    record["candidate_baseline_ratio"]=sum(record["warmed_seconds"]["candidate"])/sum(record["warmed_seconds"]["baseline"])
    record["status"]="TILE_PRIMITIVE_PARITY_PASS_COST_MEASURED" if record["finite"] and max(record["maximum_scaled_errors"])<=.01 else "TILE_PRIMITIVE_PARITY_VETO"
    (output/"transport-tile-batch.json").write_text(json.dumps(json_ready(record),indent=2,allow_nan=False)+"\n")
    return record
