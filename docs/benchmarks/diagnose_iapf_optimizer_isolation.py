"""Fresh-process CPU diagnostics; optimizer switches are not runtime defaults."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
OPTIONS=["default","disable_meta_optimizer","arithmetic_optimization",
         "dependency_optimization","loop_optimization","function_optimization",
         "constant_folding","shape_optimization","remapping"]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=OPTIONS)
    args=parser.parse_args()
    if args.case:
        import tensorflow as tf
        sys.path.insert(0,"/home/chakwong/BayesFilter")
        if args.case!="default":
            tf.config.optimizer.set_experimental_options(
                {args.case:args.case=="disable_meta_optimizer"})
        from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
        v=tf.float64
        inputs=(tf.constant([.42,0.,0.,0.,1.,0.],v),tf.zeros([2,1],v),
            tf.constant([[-1.],[-.25],[.25],[1.]],v),tf.zeros([2,4,1],v),
            tf.constant([[.125,.375,.625,.875],[.875,.625,.375,.125],
                         [.125,.375,.625,.875]],v),tf.zeros([2,4],v),
            tf.zeros([2,1],v),tf.ones([2,1,1],v),tf.zeros([2],v))
        kernel=make_fitted_twist_kernel(1,1,4,2,"float64",False,constant_twist=True)
        expected=kernel.python_function(*inputs)
        actual=kernel(*inputs)
        shapes=[x.shape.as_list() for x in actual]
        match=shapes==[x.shape.as_list() for x in expected]
        errors=[float(tf.reduce_max(tf.abs(x-y)).numpy()) for x,y in zip(actual,expected)] if match else []
        result=dict(case=args.case,pid=os.getpid(),tensorflow=tf.__version__,
            effective_options=tf.config.optimizer.get_experimental_options(),shapes=shapes,
            parity=match and max(errors,default=0)<=1e-10,errors=errors,
            cpu_only=True,gpu_intentionally_hidden=True,jit_compile=False)
        (args.output/(args.case+".json")).write_text(json.dumps(result,indent=2)+"\n")
        return
    results=[]
    for case in OPTIONS:
        command=[sys.executable,__file__,"--output",str(args.output),"--case",case]
        with (args.output/(case+".log")).open("w") as stream:
            proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=12,check=False)
        if proc.returncode!=0:raise RuntimeError("isolated diagnostic failed: "+case)
        result=json.loads((args.output/(case+".json")).read_text());result["command"]=command
        results.append(result)
        (args.output/"cases.json").write_text(json.dumps(results,indent=2)+"\n")
    with (args.output/"decision.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=["status","default_valid","passing_debug_options","replication"])
        writer.writeheader();writer.writerow(dict(status="DIAGNOSTIC_COMPLETE",
            default_valid=results[0]["parity"],
            passing_debug_options=";".join(x["case"] for x in results if x["parity"]),
            replication="INCOMPLETE"))
    print(json.dumps({"default_valid":results[0]["parity"],
        "passing_debug_options":[x["case"] for x in results if x["parity"]]}))
if __name__=="__main__":main()
