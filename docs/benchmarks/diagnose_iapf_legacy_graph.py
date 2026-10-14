"""CPU-only debugging of the legacy graph; no paper-replication claim."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import csv
import json
from pathlib import Path
import time
import sys
sys.path.insert(0,"/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    v=tf.float64
    values=(tf.constant([.42,0.,0.,0.,1.,0.],v),tf.zeros([2,1],v),
        tf.constant([[-1.],[-.25],[.25],[1.]],v),tf.zeros([2,4,1],v),
        tf.constant([[.125,.375,.625,.875],[.875,.625,.375,.125],
                     [.125,.375,.625,.875]],v),tf.zeros([2,4],v),
        tf.zeros([2,1],v),tf.ones([2,1,1],v),tf.zeros([2],v))
    reports=[]
    def compare(name,actual,expected,elapsed):
        shapes=[x.shape.as_list() for x in actual]
        expected_shapes=[x.shape.as_list() for x in expected]
        valid=shapes==expected_shapes
        errors=[float(tf.reduce_max(tf.abs(x-y)).numpy()) for x,y in zip(actual,expected)] if valid else []
        reports.append(dict(case=name,shapes=shapes,expected_shapes=expected_shapes,
            shape_valid=valid,max_errors=errors,parity=valid and max(errors,default=0)<=1e-10,
            seconds=elapsed))
        (args.output/"cases.json").write_text(json.dumps(reports,indent=2)+"\n")
    for name,options in [
        ("default",{}),
        ("trace_only",dict(include_numerical_trace=True)),
        ("fisher_only",dict(include_fisher_score=True)),
        ("all_arrays_written",dict(include_numerical_trace=True,
            include_fisher_score=True,include_resampling_controls=True,resampling_uniform_bits=16))]:
        kernel=make_fitted_twist_kernel(1,1,4,2,"float64",False,constant_twist=True,**options)
        expected=kernel.python_function(*values)[:3]
        start=time.monotonic();actual=kernel(*values)[:3]
        compare(name,actual,expected,time.monotonic()-start)
    for write in [False,True]:
        @tf.function(input_signature=[tf.TensorSpec([],v)],jit_compile=False)
        def tiny(value):
            unused=tf.TensorArray(v,size=2,element_shape=[])
            def step(t,x,arr):
                if write:arr=arr.write(t,x)
                return t+1,x+1,arr
            out=tf.while_loop(lambda t,*_:t<2,step,(0,value,unused),parallel_iterations=1)
            return out[1]
        start=time.monotonic();actual=tiny(tf.constant(0.,v))
        compare("minimal_array_written_"+str(write),[actual],[tf.constant(2.,v)],time.monotonic()-start)
    original_options=tf.config.optimizer.get_experimental_options()
    for name,options in [
        ("no_meta_optimizer",dict(disable_meta_optimizer=True)),
        ("no_arithmetic",dict(arithmetic_optimization=False)),
        ("no_dependency",dict(dependency_optimization=False)),
        ("no_loop",dict(loop_optimization=False)),
        ("no_function",dict(function_optimization=False)),
        ("no_constant_folding",dict(constant_folding=False)),
        ("no_shape",dict(shape_optimization=False)),
        ("no_remapping",dict(remapping=False))]:
        tf.config.optimizer.set_experimental_options({**original_options,**options})
        make_fitted_twist_kernel.cache_clear()
        kernel=make_fitted_twist_kernel(1,1,4,2,"float64",False,constant_twist=True)
        expected=kernel.python_function(*values)[:3]
        start=time.monotonic();actual=kernel(*values)[:3]
        compare(name,actual,expected,time.monotonic()-start)
        tf.config.optimizer.set_experimental_options(original_options)
    findings=dict(tensorflow=tf.__version__,cpu_only=True,gpu_intentionally_hidden=True,
        jit_compile=False,scope="legacy graph localization only",cases=reports,
        all_written_repairs_observed_failure=not reports[0]["parity"] and reports[3]["parity"])
    (args.output/"result.json").write_text(json.dumps(findings,indent=2)+"\n")
    with (args.output/"decision.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=["status","legacy_graph_default_valid","all_written_valid","replication"])
        writer.writeheader();writer.writerow(dict(status="DIAGNOSTIC_COMPLETE",
            legacy_graph_default_valid=reports[0]["parity"],
            all_written_valid=reports[3]["parity"],replication="INCOMPLETE"))
    print(json.dumps(dict(cases=len(reports),all_written_repairs=findings["all_written_repairs_observed_failure"])))
if __name__=="__main__":main()
