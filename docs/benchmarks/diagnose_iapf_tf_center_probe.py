"""Diagnostic caller of the actual Gaussian twisted transition; no runtime role."""
import csv
import hashlib
import json
import time

def run(fixture, output):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy=configure_tensorflow_gpu_memory_growth(tf)
    from bayesfilter.score_study.fitted_twist_tf import twisted_transition
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    N,T,d=64,5,2
    directory=fixture/"d2"
    def read(name,shape,dtype):
        values=[float(row[0]) for row in csv.reader((directory/f"{name}.csv").open())]
        return tf.reshape(tf.constant(values,dtype),shape)
    records=[];outputs={}
    with tf.device("/GPU:0"):
        for precision,tf32 in (("float64",False),("float32",False),("float32",True)):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            dtype=tf.as_dtype(precision)
            Q=parameterized_model(read("theta",[6],dtype),d,d)[8]
            dQ=tf.zeros([6,d,d],dtype)
            means=tf.tile(read("initial",[N,d],dtype)[None,:,:]*tf.constant(.3,dtype),[T,1,1])
            noise=read("noise",[T,N,d],dtype)
            covariance=read("covariances",[T,d,d],dtype)
            for center_kind in ("fixture","sentinel"):
                centers=(read("centers",[T,d],dtype) if center_kind=="fixture" else
                    tf.constant([[0.,0.],[1.,-1.],[-2.,2.],[3.,-3.],[-4.,4.]],dtype))
                inputs=[means,Q,dQ,centers,covariance,noise]
                signature=[tf.TensorSpec(x.shape,dtype) for x in inputs]
                for route,jit in (("while",True),("unrolled",True),("while",False)):
                    def build(route,jit):
                        @tf.function(input_signature=signature,jit_compile=jit)
                        def kernel(means,Q,dQ,centers,covariances,noise):
                            def one(t):
                                return twisted_transition(means[t],tf.zeros([6,N,d],dtype),Q,dQ,
                                    centers[t],covariances[t],tf.ones([N],dtype),noise[t],tf.zeros([N],dtype))
                            if route=="unrolled":
                                ans=[one(t) for t in range(T)]
                                return tf.stack([x[0] for x in ans]),tf.stack([x[1] for x in ans])
                            xs=tf.TensorArray(dtype,size=T,element_shape=[N,d])
                            ds=tf.TensorArray(dtype,size=T,element_shape=[6,N,d])
                            def step(t,xs,ds):
                                x,dx=one(t)
                                return t+1,xs.write(t,x),ds.write(t,dx)
                            r=tf.while_loop(lambda t,*_:t<T,step,(0,xs,ds),parallel_iterations=1)
                            return r[1].stack(),r[2].stack()
                        return kernel
                    kernel=build(route,jit)
                    start=time.monotonic();result=kernel(*inputs);result[0].numpy()
                    seconds=time.monotonic()-start
                    key=f"{precision}/{tf32}/{center_kind}/{route}/{jit}"
                    outputs[key]=result[0].numpy().tolist()
                    # FP64 non-XLA below is the numerical comparator; the
                    # independent formula is evaluated separately on the CPU.
                    record=dict(key=key,dtype=precision,tf32=tf32,center_kind=center_kind,
                        route=route,jit_compile=jit,seconds=seconds,device=result[0].device,
                        trace_count=kernel.experimental_get_tracing_count())
                    if jit:
                        hlo=kernel.experimental_get_compiler_ir(*inputs)(stage="hlo")
                        record["hlo_sha256"]=hashlib.sha256(hlo.encode()).hexdigest()
                    records.append(record)
    for r in records:
        a=outputs[r["key"]];b=outputs[f"float64/False/{r['center_kind']}/while/False"]
        r["max_error_by_time"]=[max(abs(x-y) for xs,ys in zip(u,v) for x,y in zip(xs,ys))
            for u,v in zip(a,b)]
        ref=read("center-probe-"+r["center_kind"],[T,N,d],tf.float64).numpy().tolist()
        r["independent_R_max_error_by_time"]=[max(abs(x-y) for xs,ys in zip(u,v) for x,y in zip(xs,ys))
            for u,v in zip(a,ref)]
        # Error bounds distinguish the observed O(1) wrong-center result from
        # O(1e-3) TF32 rounding; no full-filter precision admission is implied.
        tolerance=1e-9 if r["dtype"]=="float64" else (0.003 if r["tf32"] else 2e-6)
        r["independent_reference_pass"]=max(r["independent_R_max_error_by_time"])<=tolerance
        r["reference_tolerance"]=tolerance
    valid=all(r["independent_reference_pass"] and "GPU" in r["device"] for r in records)
    report=dict(status="pass" if valid else "fail",records=records,gpu_memory_policy=policy,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),tensorflow_version=tf.__version__,
        GPU_intentionally_selected=True,production_default_changed=False,
        scope="isolated actual twisted_transition with frozen inputs and Gaussian branch",
        independent_formula_checked=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    output.with_name("center-probe-values.json").write_text(json.dumps(outputs)+"\n")
    print(json.dumps(dict(status=report["status"],records=len(records))))
    return 0 if valid else 1


def run_broadcast(fixture, output):
    """Reduce the failure to a time-indexed broadcast matrix product."""
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy=configure_tensorflow_gpu_memory_growth(tf)
    records=[]
    with tf.device("/GPU:0"):
        for tf32 in (False,True):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            K=tf.constant([[.47,-.01],[-.01,.37]],tf.float32)
            means=tf.random.stateless_normal([64,2],[93810001,1])*.3
            cs=tf.constant([[0.,0.],[1.,-1.],[-2.,2.],[3.,-3.],[-4.,4.]])
            expected=(tf.cast(cs,tf.float64)[:,None,:]-tf.cast(means,tf.float64)[None,:,:])@tf.transpose(tf.cast(K,tf.float64))
            for split,unrolled in ((False,False),(True,False),(False,True)):
                def build(split,unrolled):
                    @tf.function(input_signature=[tf.TensorSpec([2,2],tf.float32),
                        tf.TensorSpec([64,2],tf.float32),tf.TensorSpec([5,2],tf.float32)],jit_compile=True)
                    def kernel(K,means,cs):
                        def one(t):
                            if split:return tf.linalg.matvec(K,cs[t])[None,:]-tf.einsum("ij,nj->ni",K,means)
                            return tf.einsum("ij,nj->ni",K,cs[t]-means)
                        if unrolled:return tf.stack([one(t) for t in range(5)])
                        xs=tf.TensorArray(tf.float32,size=5,element_shape=[64,2])
                        def step(t,xs):return t+1,xs.write(t,one(t))
                        answer=tf.while_loop(lambda t,*_:t<5,step,(0,xs),parallel_iterations=1)
                        return answer[1].stack()
                    return kernel
                kernel=build(split,unrolled);actual=kernel(K,means,cs)
                error=tf.reduce_max(tf.abs(tf.cast(actual,tf.float64)-expected),axis=[1,2]).numpy().tolist()
                key=f"tf32-{tf32}-split-{split}-unrolled-{unrolled}"
                hlo=kernel.experimental_get_compiler_ir(K,means,cs)(stage="optimized_hlo")
                hlo_path=output.with_name(key+".optimized.hlo.txt");hlo_path.write_text(hlo)
                records.append(dict(tf32=tf32,split=split,unrolled=unrolled,
                    max_error_by_time=error,device=actual.device,optimized_hlo=str(hlo_path),
                    hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                    values=actual.numpy().tolist()))
    controls=all(max(r["max_error_by_time"])<.003 for r in records
                 if not r["tf32"] or r["split"] or r["unrolled"])
    target=next(r for r in records if r["tf32"] and not r["split"] and not r["unrolled"])
    report=dict(status="diagnosis_complete" if controls else "diagnosis_invalid",
        standalone_broadcast_failure_reproduced=max(target["max_error_by_time"])>.1,
        controls_pass=controls,records=records,gpu_memory_policy=policy,
        tensorflow_version=tf.__version__,tensorflow_build_info=tf.sysconfig.get_build_info(),
        device_details=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
        allocator=tf.config.experimental.get_memory_info("GPU:0"),production_default_changed=False)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("status","standalone_broadcast_failure_reproduced","controls_pass")}))
    return 0 if controls else 1
