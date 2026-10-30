"""GPU/XLA validation of Lorenz adaptive refinement and source APF counts."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,"/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
import lorenz_refinement_tf as controller
D=tf.float64

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    checks,cases,apf=[],[],[]
    summary=dict(status="FAIL",scope="LORENZ_REFINEMENT_AND_APF_VALIDATION",
        replication="INCOMPLETE",device="/GPU:0",jit_compile=True,dtype="float64",
        tf32=False,gpu_memory_policy=memory,checks=checks,cases=cases,apf=apf,
        seed=202609360,heuristic_dominance_verdict="NOT_ASSESSED_IMPLEMENTATION_VALIDATION_ONLY")
    def flag(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details))
    def check(name,actual,expected,atol=1e-7,rtol=1e-9):
        a=tf.cast(actual,D);e=tf.cast(expected,D)
        delta=tf.abs(a-e);limit=atol+rtol*tf.abs(e)
        good=tf.reduce_all(tf.math.is_finite(a)&tf.math.is_finite(e)&(delta<=limit))
        flag(name,good.numpy(),max_absolute_error=float(tf.reduce_max(delta).numpy()),
             atol=atol,rtol=rtol)
    def load(item):
        path=args.reference/item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError("Reference binary hash mismatch: "+item["file"])
        dtype=tf.int32 if item["dtype"]=="int32" else D
        return tf.reshape(tf.io.decode_raw(tf.io.read_file(str(path)),dtype,little_endian=True),item["shape"])
    def traces(engine):
        return dict(controller=engine.run.experimental_get_tracing_count(),
            forward=engine.engine.forward.experimental_get_tracing_count(),
            fit=engine.engine.fit.experimental_get_tracing_count())
    def save(name,value):
        tf.io.write_file(str(args.output/(name+".tensor")),tf.io.serialize_tensor(value))
    try:
        prior=json.loads((args.reference/"summary.json").read_text())
        if prior["status"]!="PASS":
            raise ValueError("Independent reference did not pass")
        fixture=json.loads((args.reference/"fixture.json").read_text())
        # The new wrapper imports existing repository kernels. Bind actual files
        # to the snapshots rather than merely asserting that a general kernel exists.
        core=sys.modules[controller.MultivariateGaussianControlledSMC.__module__]
        for function in (controller.MultivariateGaussianControlledSMC,core.systematic_ancestors):
            actual=Path(inspect.getfile(function))
            snapshot=Path(__file__).with_name(actual.name)
            flag("snapshot_"+function.__name__,actual.read_bytes()==snapshot.read_bytes())
        if not all(c["passed"] for c in checks):
            raise ValueError("Kernel dependency snapshot mismatch")
        with tf.device("/GPU:0"):
            y=tf.constant(fixture["metadata"]["y"],D)[:,None,:]
            normal=load(fixture["normal"])[:,:,None,:,:]
            uniform=load(fixture["uniform"])[:,:,None]
            boundary=tf.function(lambda i,m,e:controller.needs_refinement(i,m,e,512),
                input_signature=[tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.int32),tf.TensorSpec([1],D)],
                jit_compile=True,autograph=False)
            for name,i,m,e,expected in (
                ("at_least_one",0,4,500.,True),("strict90",1,4,460.8,True),
                ("above90",1,4,460.80001,False),("max_four",4,4,1.,False),("apf_only",0,0,1.,False)):
                flag(name,bool(boundary(tf.constant(i),tf.constant(m),tf.constant([e],D)).numpy())==expected)
            flag("boundary_single_trace",boundary.experimental_get_tracing_count()==1)
            flag("apf_count_mapping",[controller.apf_particles_for_refinements(i) for i in range(1,5)]==[1382,2867,4505,6144])
            try:
                controller.apf_particles_for_refinements(0)
                flag("zero_refinement_count_rejected",False)
            except ValueError:
                flag("zero_refinement_count_rejected",True)
            for case_index,case in enumerate(fixture["cases"]):
                engine=controller.LorenzRefinement(forcing=case["forcing"])
                before=time.monotonic()
                output=engine.run(y,normal,uniform,tf.constant(4))
                count=int(output["iterations"].numpy())
                wall=time.monotonic()-before
                flag("valid_case%d"%case_index,output["valid"].numpy())
                check("iterations_case%d"%case_index,count,case["iterations"],0,0)
                check("executed_case%d"%case_index,output["executed"],
                      tf.range(5)<=case["iterations"],0,0)
                for iteration,stage in enumerate(case["stages"]):
                    for key,item in {**stage["policy"],**stage["output"]}.items():
                        actual=output["history"][key][iteration,:,0]
                        check("case%d_stage%d_%s"%(case_index,iteration,key),
                            actual,tf.cast(load(item),D),
                            0 if key=="ancestors" else 1e-7,0 if key=="ancestors" else 1e-9)
                for key,value in output["history"].items():
                    save("case%d_%s"%(case_index,key),value)
                save("case%d_fit_gram_ratio"%case_index,output["minimum_fit_gram_ratio"])
                save("case%d_fit_rss"%case_index,output["maximum_fit_rss"])
                raw=output["history"]["logZ"][count,:,0]
                normalized=raw-.5*6*math.log(2*math.pi*1e-4)*tf.cast(tf.range(1,102),D)
                check("normalized_prefix_case%d"%case_index,output["normalized_logZ"][count,:,0],
                      normalized,1e-10,1e-12)
                trace=traces(engine)
                flag("stable_traces_case%d"%case_index,all(v==1 for v in trace.values()))
                cases.append(dict(forcing=case["forcing"],iterations=count,seconds=wall,traces=trace,
                    minimum_final_ess_fraction=float(tf.reduce_min(output["history"]["ess"][count]).numpy())/512,
                    minimum_precision_margin=float(tf.reduce_min(output["history"]["precision_margin"][:count+1]).numpy()),
                    minimum_fit_gram_ratio=float(tf.reduce_min(output["minimum_fit_gram_ratio"][1:count+1]).numpy()),
                    normalized_terminal_logZ=float(output["normalized_logZ"][count,-1,0].numpy()),
                    source_apf_particles=controller.apf_particles_for_refinements(count)))
                (args.output/"progress.json").write_text(json.dumps(cases,indent=2,allow_nan=False)+"\n")
                print(json.dumps(cases[-1]),flush=True)
                if not bool(output["valid"].numpy()):
                    raise ValueError("Invalid repeated-refinement path")
            normal_apf=load(fixture["apf_normal"])
            uniform_apf=load(fixture["apf_uniform"])[:,None]
            for case in fixture["apf"]:
                n=case["particles"]
                engine=controller.LorenzRefinement(forcing=4.8801,particles=n)
                policy=engine.initial_policy(y)
                before=time.monotonic()
                output=engine.engine.forward(*policy,y,normal_apf[:,:n][:,None],uniform_apf)
                flag("apf_valid_%d"%n,output["valid"].numpy())
                for key,item in case["stage"]["output"].items():
                    check("apf%d_%s"%(n,key),output[key][:,0],tf.cast(load(item),D),
                          0 if key=="ancestors" else 1e-7,0 if key=="ancestors" else 1e-9)
                apf.append(dict(particles=n,seconds=time.monotonic()-before,
                    forward_traces=engine.engine.forward.experimental_get_tracing_count()))
                flag("apf_single_trace_%d"%n,apf[-1]["forward_traces"]==1)
            # Independent conjugate-Gaussian APF formula at time zero.
            gauss=tf.function(lambda mean,a,b,c:core.multivariate_twist(
                mean,tf.eye(8,dtype=D)*100.,tf.constant(8*math.log(.01),D),a,b,c)[:3],
                input_signature=[tf.TensorSpec([1,3,8],D),tf.TensorSpec([1,8,8],D),
                                 tf.TensorSpec([1,8],D),tf.TensorSpec([1],D)],
                jit_compile=True,autograph=False)
            means=tf.stack([tf.zeros([8],D),tf.linspace(tf.constant(-.2,D),tf.constant(.3,D),8),tf.ones([8],D)*2])[None]
            cm,cov,logint=gauss(means,policy[0][0],policy[1][0],policy[2][0])
            expected_cm=tf.concat([(means[...,:6]/.01+y[0,:,None,:]/1e-4)/(100.+10000.),means[...,6:]],-1)
            expected_cov=tf.linalg.diag(tf.constant([1./10100.]*6+[.01,.01],D))[None]
            expected_logint=-.5*6*math.log(.0101/1e-4)-.5/.0101*tf.reduce_sum((y[0,:,None,:]-means[...,:6])**2,-1)
            check("apf_closed_gaussian_mean",cm,expected_cm,1e-11,1e-11)
            check("apf_closed_gaussian_covariance",cov,expected_cov,1e-13,1e-11)
            check("apf_closed_gaussian_integral",logint,expected_logint,1e-8,1e-10)
            summary.update(status="PASS" if all(c["passed"] for c in checks) else "FAIL",
                call_chain="LorenzRefinement.run -> MultivariateGaussianControlledSMC.forward / fit; shared systematic_ancestors",
                nonclaims=["No Figure6 or Table1 replication","No performance ranking",
                           "No cost matching","No default or HMC readiness"])
    except Exception as exc:
        summary.update(failure=type(exc).__name__+": "+str(exc))
    finally:
        summary.update(wall_seconds=time.monotonic()-started,tensorflow=tf.__version__,
            allocator=tf.config.experimental.get_memory_info("GPU:0"))
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:summary.get(k) for k in ("status","failure","wall_seconds")}),flush=True)
    if summary["status"]!="PASS":
        raise SystemExit(1)

if __name__=="__main__":
    main()
