"""Diagnostic-only native parity and bridge overhead harness (no NumPy).

Short chains validate integration, never posterior convergence.
"""
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import resource
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT/".localresources/native_sampler_bridge"

class Native:
    def __init__(self):
        self.lib=ct.CDLL(str(BUILD/"libbf_native.so"))
        pdouble=ct.POINTER(ct.c_double); pint=ct.POINTER(ct.c_int64)
        self.lib.bf_native_fit.argtypes=[pint,pdouble,pdouble,pdouble,pint,pdouble,pdouble,ct.c_char_p,ct.c_size_t]
        self.lib.bf_native_fit.restype=ct.c_int
        self.lib.bf_native_gradient.argtypes=[ct.c_int64,ct.c_int64,ct.c_double,pdouble,pdouble,pdouble,ct.c_char_p,ct.c_size_t]
        self.lib.bf_native_gradient.restype=ct.c_int
    def fit(self,cfg,controls,starts,inverse,seeds):
        c,d=cfg.chains,cfg.dimension
        rows=(ct.c_double*(c*(cfg.warmup+cfg.draws)*(d+7)))()
        meta=(ct.c_double*(c*(d+4)))()
        error=ct.create_string_buffer(2048)
        rc=self.lib.bf_native_fit((ct.c_int64*8)(*cfg.native_config()),
            (ct.c_double*4)(*controls),(ct.c_double*(c*d))(*starts),
            (ct.c_double*(d*d))(*inverse),(ct.c_int64*c)(*seeds),rows,meta,error,len(error))
        if rc: raise ValueError(error.value.decode())
        return rows,meta
    def gradient(self,kind,d,bend,inverse,q):
        out=(ct.c_double*(d+1))();error=ct.create_string_buffer(2048)
        rc=self.lib.bf_native_gradient(kind,d,bend,(ct.c_double*(d*d))(*inverse),
            (ct.c_double*d)(*q),out,error,len(error))
        if rc:raise ValueError(error.value.decode())
        return list(out)

def equal(a,b):
    return len(a)==len(b) and all(x==y or (math.isnan(x) and math.isnan(y)) for x,y in zip(a,b))

def gradient_checks(native):
    d=4;bend=.3;q=[.2,-.4,.6,-.8]
    inverse=[1.,.3,0.,0., 0.,2.,.1,0., 0.,0.,.7,.2, 0.,0.,0.,1.3]
    checks=[]
    for kind in range(4):
        if kind==0: lp=-.5*sum(x*x for x in q);g=[-x for x in q]
        elif kind==1:
            y=[sum(inverse[i*d+j]*q[j] for j in range(d)) for i in range(d)]
            lp=-.5*sum(x*x for x in y)
            g=[-sum(inverse[i*d+j]*y[i] for i in range(d)) for j in range(d)]
        elif kind==2:
            lp=0.;g=[]
            for x,y in zip(q[::2],q[1::2]):
                z=y-bend*(x*x-1);lp-=.5*(x*x+z*z);g.extend([-x+2*bend*x*z,-z])
        else:
            v=q[0];ss=sum(x*x for x in q[1:]);e=math.exp(-v)
            lp=-.5*(v/3)**2-.5*(d-1)*v-.5*ss*e
            g=[-v/9-.5*(d-1)+.5*ss*e]+[-x*e for x in q[1:]]
        actual=native.gradient(kind,d,bend,inverse,q)
        err=max(abs(x-y) for x,y in zip(actual,[lp]+g))
        assert err < 1e-12,(kind,err)
        checks.append({"kind":kind,"max_absolute_error":err})
    return checks

def cmdstan_check(native,FitConfig,directory):
    d=4;cfg=FitConfig("stan",0,d,1,100,100,10)
    inv=[float(i==j) for i in range(d) for j in range(d)];q=[.2,-.4,.6,-.8]
    data=directory/"cmdstan_data.json";init=directory/"cmdstan_init.json";csv=directory/"cmdstan.csv"
    data.write_text(json.dumps({"d":d,"kind":0,"n":0,"design":[],"bend":.3,"inverse_transform":[inv[i*d:(i+1)*d] for i in range(d)]}))
    init.write_text(json.dumps({"q":q}))
    command=[str(ROOT/"docs/benchmarks/stan_targets"),"sample","num_warmup=100","num_samples=100","save_warmup=1","adapt","delta=0.8","algorithm=hmc","engine=nuts","max_depth=10","metric=diag_e","stepsize=1","data","file="+str(data),"init="+str(init),"random","seed=101","id=1","output","file="+str(csv),"sig_figs=18","refresh=0"]
    with (directory/"cmdstan.log").open("w") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    lines=[line for line in csv.read_text().splitlines() if line and not line.startswith("#")]
    actual=[float(v) for line in lines[1:] for v in line.split(",")]
    expected=list(native.fit(cfg,[.3,1.,.8,.5],q,inv,[101])[0])
    assert len(actual)==len(expected)
    max_error=max(abs(x-y) for x,y in zip(actual,expected))
    assert max_error<1e-10,max_error
    return {"command":command,"max_absolute_error":max_error,"identical":actual==expected}

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",choices=["cpu","gpu"],required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--benchmark",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    if args.device=="cpu": os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    else:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")!="true":
            raise RuntimeError("GPU memory growth must be set before import")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS","1");os.environ.setdefault("TF_NUM_INTEROP_THREADS","1")
    import tensorflow as tf
    memory=[]
    if args.device=="gpu":
        devices=tf.config.list_physical_devices("GPU");assert devices,"no GPU"
        for dev in devices:
            tf.config.experimental.set_memory_growth(dev,True)
            growth=tf.config.experimental.get_memory_growth(dev);assert growth
            memory.append({"device":dev.name,"growth":growth,"details":tf.config.experimental.get_device_details(dev)})
    tf.config.set_soft_device_placement(False)
    from wrapper import FitConfig,compiled_fit
    native=Native();device="/GPU:0" if args.device=="gpu" else "/CPU:0"
    manifest={"command":sys.argv,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python":sys.executable,"python_version":platform.python_version(),"tensorflow":tf.__version__,"device":device,
        "platform":platform.platform(),"cpu":platform.processor(),"TF32":tf.config.experimental.tensor_float_32_execution_enabled(),
        "result_file":"docs/plans/bayesfilter-native-stan-walnutpie-result-2026-10-28.md",
        "device_trust_basis":"trusted_escalated_tool_execution" if args.device=="gpu" else "explicit_cpu_reference",
        "scheduling":"sequential chains; serialized native ABI",
        "sampler_execution":"native CPU C++","jit_compile":True,"GPU_intentionally_hidden":args.device=="cpu",
        "memory_policy":memory,"plan":"docs/plans/bayesfilter-native-stan-walnutpie-master-2026-10-28.md",
        "data":"compiled stan_targets.stan synthetic targets","seeds":[101,202,303,404],
        "mode":"descriptive overhead benchmark" if args.benchmark else "engineering smoke",
        "source_hashes":json.loads((BUILD/"source_hashes.json").read_text()),"build":json.loads((BUILD/"build_manifest.json").read_text())}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    result={"gradient_checks":gradient_checks(native),"fits":[],"errors":[]}
    if args.device=="cpu":
        result["cmdstan"]=cmdstan_check(native,FitConfig,args.output)
        text=subprocess.check_output([str(BUILD/"walnut_oracle")],text=True,timeout=30)
        (args.output/"walnut_oracle.txt").write_text(text)
        lines=[[float(v) for v in line.split()] for line in text.splitlines()]
        cfg=FitConfig("walnutpie",0,4,1,100,100,5)
        rows,meta=native.fit(cfg,[.3,.1,.8,.5],[.2,-.4,.6,-.8],
            [float(i==j) for i in range(4) for j in range(4)],[101])
        expected=[[200.,*list(meta)[:5]]]+[[rows[i*11],*list(rows)[i*11+7:(i+1)*11]] for i in range(200)]
        error=max(abs(x-y) for a,b in zip(lines,expected) for x,y in zip(a,b))
        assert len(lines)==len(expected) and error<1e-12,error
        result["walnutpie_public_api"]={"max_absolute_error":error,"fixed_counts":100,"independent_analytic_target":True}
    targets=[(0,10),(2,10),(3,10)] if args.benchmark else [(0,4),(2,4),(3,4),(1,4)]
    for engine in ("stan","walnutpie"):
        for kind,d in targets:
            c=4 if args.benchmark else 2;w=s=500 if args.benchmark else 30
            cfg=FitConfig(engine,kind,d,c,w,s,10 if engine=="stan" else 5)
            starts=[.1*(j+1)*(-1 if i%2 else 1) for i in range(c) for j in range(d)]
            inv=[float(i==j)*(1.+i*.2 if kind==1 else 1.) for i in range(d) for j in range(d)]
            if kind==1:inv[1]=.3
            controls=[.3,1. if engine=="stan" else .1,.8,.5];seeds=[101,202,303,404][:c]
            func=compiled_fit(cfg,device)
            with tf.device(device):
                inputs=[tf.constant(controls,tf.float64),tf.reshape(tf.constant(starts,tf.float64),[c,d]),
                    tf.reshape(tf.constant(inv,tf.float64),[d,d]),tf.constant(seeds,tf.int64)]
            def wrapped():
                with tf.device(device):out=func(*inputs)
                # Host materialization is a diagnostic/timing boundary only.
                vals=[x.numpy() for x in out]
                assert all(args.device.upper() in x.device for x in out),[x.device for x in out]
                return vals
            t=time.monotonic();first=wrapped();first_s=time.monotonic()-t
            hlo=func.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
            assert "BfNativeSamplerV1" in hlo
            (args.output/f"{engine}-{kind}.hlo").write_text(hlo)
            direct=native.fit(cfg,controls,starts,inv,seeds)
            assert all(bytes(a)==bytes(b) for a,b in zip(direct,first)),(engine,kind,"native parity")
            assert all(bytes(a)==bytes(b) for a,b in zip(first,wrapped())),"replay mismatch"
            changed=native.fit(cfg,controls,starts,inv,[x+1 for x in seeds])
            assert not equal(list(direct[0]),list(changed[0])),"seed ignored"
            native_times=[];wrapped_times=[]
            for rep in range(3 if args.benchmark else 1):
                routes=["native","wrapped"] if rep%2==0 else ["wrapped","native"]
                for route in routes:
                    t=time.monotonic()
                    value=native.fit(cfg,controls,starts,inv,seeds) if route=="native" else wrapped()
                    elapsed=time.monotonic()-t
                    (native_times if route=="native" else wrapped_times).append(elapsed)
                    assert all(bytes(a)==bytes(b) for a,b in zip(direct,value))
            row_list=list(direct[0]);width=d+7
            retained=[row_list[(ch*(w+s)+i)*width:(ch*(w+s)+i+1)*width] for ch in range(c) for i in range(w,w+s)]
            assert all(math.isfinite(row[0]) and all(math.isfinite(x) for x in row[7:]) for row in retained)
            item={"engine":engine,"kind":kind,"dimension":d,"chains":c,"warmup":w,"retained":s,
                "controls":controls,"depth":cfg.depth,"halvings":cfg.halvings,"initial_call_including_compile_s":first_s,
                "native_s":native_times,"wrapped_s":wrapped_times,"native_median_s":statistics.median(native_times),
                "wrapped_median_s":statistics.median(wrapped_times),"overhead_ratio":statistics.median(wrapped_times)/statistics.median(native_times),
                "exact_parity":True,"finite_retained":True,"deterministic_replay":True,"seed_changes_result":True,
                "trace_count":func.experimental_get_tracing_count(),"mean_first_coordinate":statistics.mean(r[7] for r in retained),
                "funnel_neck_fraction_below_minus3":sum(r[7]<-3 for r in retained)/len(retained) if kind==3 else None,
                "divergences":sum(r[5] for r in retained) if engine=="stan" else None,
                "native_output_sha256":hashlib.sha256(bytes(direct[0])+bytes(direct[1])).hexdigest()}
            (args.output/f"{engine}-{kind}-rows.bin").write_bytes(bytes(direct[0]))
            (args.output/f"{engine}-{kind}-metadata.bin").write_bytes(bytes(direct[1]))
            result["fits"].append(item);print(json.dumps(item),flush=True)
            (args.output/"results.json").write_text(json.dumps(result,indent=2))
            if kind==0:
                for label,which,index,value in [("bad_step",0,1,-1.),("nan_start",1,0,float("nan")),("bad_seed",3,0,-1)]:
                    bad=list(inputs);flat=bad[which].numpy().reshape(-1).tolist();flat[index]=value
                    with tf.device(device):bad[which]=tf.reshape(tf.constant(flat,dtype=inputs[which].dtype),inputs[which].shape)
                    try:
                        with tf.device(device):got=func(*bad)
                        [x.numpy() for x in got]
                    except tf.errors.OpError as error:
                        result["errors"].append({"engine":engine,"case":label,"propagated":True,"message":str(error)[:300]})
                    else:raise AssertionError("invalid input accepted: "+label)
    manifest["wall_seconds"]=time.monotonic()-start
    manifest["peak_process_rss_kib_linux"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["current_harness_hashes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")}
    manifest["artifacts"]=[str(x) for x in args.output.iterdir()]
    if args.device=="gpu":manifest["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (args.output/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"complete":True,"output":str(args.output),"wall_seconds":manifest["wall_seconds"]}),flush=True)

if __name__=="__main__":main()
