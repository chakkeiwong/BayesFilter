"""Diagnostic-only fixed transport exponential cache; no runtime admission."""
import hashlib
import json
from pathlib import Path
import time
import tensorflow as tf


def cached_normalization(cost, kernel, potential):
    """Return log-normalizer, weights and whether the log-domain fallback ran."""
    dtype = cost.dtype
    shift = tf.reduce_max(potential, -1, keepdims=True)
    scaled = kernel * tf.exp(potential-shift)[:, None, :]
    denominator = tf.reduce_sum(scaled, -1, keepdims=True)
    tiny = tf.constant(2.2250738585072014e-308 if dtype == tf.float64 else 1.1754943508222875e-38, dtype)
    guard = tf.constant(3*int(cost.shape[-1]), dtype)*tiny/tf.constant(1e-6, dtype)
    safe = (tf.reduce_all(tf.math.is_finite(cost)) & tf.reduce_all(cost <= 0)
            & tf.reduce_all(tf.math.is_finite(denominator))
            & tf.reduce_all(denominator >= guard))
    def fast():
        return tf.squeeze(tf.math.log(denominator), -1)+shift, scaled/denominator
    def fallback():
        logits = cost+potential[:, None, :]
        return tf.reduce_logsumexp(logits, -1), tf.nn.softmax(logits, -1)
    primal, weights = tf.cond(safe, fast, fallback)
    return primal, weights, tf.logical_not(safe)


def combined_normalization(cost, potential):
    """One stable exponential supplies both logsumexp and softmax weights."""
    logits=cost+potential[:,None,:]
    safe=tf.reduce_all(tf.math.is_finite(logits))
    def fast():
        maximum=tf.reduce_max(logits,-1,keepdims=True)
        scaled=tf.exp(logits-maximum)
        total=tf.reduce_sum(scaled,-1,keepdims=True)
        return tf.squeeze(maximum+tf.math.log(total),-1),scaled/total
    def fallback():
        return tf.reduce_logsumexp(logits,-1),tf.nn.softmax(logits,-1)
    primal,weights=tf.cond(safe,fast,fallback)
    return primal,weights,tf.logical_not(safe)


def run_diagnostic(case, output, repetitions=10, method="cache"):
    """Primitive GPU cost nomination, not model likelihood or HMC evidence."""
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    from bayesfilter.testing.surrogate_hmc_five_model import json_ready
    if method not in ("cache","combined"): raise ValueError("unknown transport primitive")
    output=Path(output); dtype=tf.float32
    k=select_transport_chunks(int(case.fixed[0].shape[0])).row_chunk_size
    b,p,d=2,case.p,case.spec.dimension
    width=p*(d+2)+d
    x=tf.random.stateless_normal([b,k,d],[141,27],dtype=dtype)
    cost=-tf.reduce_sum(tf.square(x[:,:,None]-x[:,None]),-1)/tf.cast(max(1,d),dtype)
    kernel=tf.exp(cost)
    potential=tf.random.stateless_normal([b,k],[142,27],dtype=dtype)*3
    packed=tf.random.stateless_normal([b,k,width],[143,27],dtype=dtype)
    signature=[tf.TensorSpec(a.shape,a.dtype) for a in (cost,kernel,potential,packed)]
    @tf.function(input_signature=signature,jit_compile=True,autograph=False)
    def baseline(c,e,a,h):
        logits=c+a[:,None,:]
        return tf.reduce_logsumexp(logits,-1),tf.linalg.matmul(tf.nn.softmax(logits,-1),h),tf.constant(False)
    @tf.function(input_signature=signature,jit_compile=True,autograph=False)
    def candidate(c,e,a,h):
        value,weights,fallback=(cached_normalization(c,e,a) if method=="cache" else combined_normalization(c,a))
        return value,tf.linalg.matmul(weights,h),fallback
    args=(cost,kernel,potential,packed)
    def measure(fn):
        start=time.monotonic(); out=fn(*args)
        for tensor in out: tensor.numpy()
        return out,time.monotonic()-start
    result=dict(status="RUNNING",method=method,evidence_role="synthetic_primitive_cost_nomination_only",
        model_dimensions=case.name,batch=b,tile=k,parameter_count=p,state_dimension=d,packed_width=width,
        seeds=[[141,27],[142,27],[143,27]],dtype=dtype.name,jit_compile=True,tf32=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=[hashlib.sha256(tf.io.serialize_tensor(v).numpy()).hexdigest() for v in args],
        cold_seconds={},warmed_seconds={"baseline":[],"candidate":[]},
        posterior_sampling_complete=False,canonical_admission=False)
    initial={}
    for name,fn in (("baseline",baseline),("candidate",candidate)):
        initial[name],result["cold_seconds"][name]=measure(fn)
    result["maximum_scaled_errors"]=[float(tf.reduce_max(tf.abs(a-r)/tf.maximum(tf.ones_like(r),tf.abs(r))).numpy()) for a,r in zip(initial["candidate"][:2],initial["baseline"][:2])]
    result["fallback_used"]=bool(initial["candidate"][2].numpy())
    result["finite"]=all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in initial["candidate"][:2])
    for i in range(repetitions):
        jobs=(("baseline",baseline),("candidate",candidate))
        for name,fn in jobs[::1 if i%2==0 else -1]:
            out,elapsed=measure(fn); result["warmed_seconds"][name].append(elapsed)
            for old,new in zip(initial[name],out): tf.debugging.assert_equal(old,new)
    result["candidate_baseline_ratio"]=sum(result["warmed_seconds"]["candidate"])/sum(result["warmed_seconds"]["baseline"])
    result["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    result["status"]="PRIMITIVE_PARITY_PASS_COST_MEASURED" if result["finite"] and max(result["maximum_scaled_errors"])<=.01 else "PRIMITIVE_PARITY_VETO"
    filename="transport-cache-probe.json" if method=="cache" else "transport-combined-probe.json"
    (output/filename).write_text(json.dumps(json_ready(result),indent=2,allow_nan=False)+"\n")
    return result
